/*
 * NEON-accelerated math operations for embedding indexing.
 *
 * Hot loops:
 * - k-means assignment: n × K dot products per iteration
 * - kNN candidate scoring: query vs N candidates
 * - Covariance accumulation: streaming outer products
 * - Matrix-vector multiply: projection per track
 * - int16 → mono float conversion: bulk audio decoding
 *
 * ARM NEON does 4 float multiply-adds per instruction, giving ~4x
 * speedup over scalar Kotlin loops. Combined with C loop efficiency
 * (no bounds checks, register allocation), expect 6-10x total.
 */
#include <jni.h>
#include <android/log.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdint.h>
#include <limits.h>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

#define TAG "MathJNI"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

/* ── Exact deterministic Graph Explorer ──────────────────── */

static void throw_graph_explorer_error(JNIEnv *env, const char *class_name, const char *message) {
    jclass error_class = (*env)->FindClass(env, class_name);
    if (error_class) (*env)->ThrowNew(env, error_class, message);
}

/*
 * Integrate the exact terminal distribution over directed (previous,current)
 * edge states. This mirrors GraphExplorerSelector's pure Kotlin reference.
 * Cancellation is checked once per propagated link; one native link iteration
 * is bounded by edge_count * K and is short at the production 80K x 5 graph.
 */
JNIEXPORT jint JNICALL
Java_com_powerampstartradio_similarity_algorithms_NativeGraphExplorer_nativePropagate(
    JNIEnv *env, jclass cls,
    jintArray j_neighbors,
    jbyteArray j_choice_counts,
    jint node_count,
    jint k,
    jint seed,
    jfloat stop_probability,
    jint max_links,
    jdoubleArray j_terminal,
    jdoubleArray j_terminal_link_mass,
    jobject cancellation_check)
{
    (void)cls;
    if (node_count <= 0 || k <= 0 || seed < 0 || seed >= node_count ||
        max_links <= 0 || !isfinite(stop_probability) ||
        stop_probability < 0.0f || stop_probability > 1.0f) {
        throw_graph_explorer_error(
            env, "java/lang/IllegalArgumentException", "invalid Graph Explorer arguments");
        return -1;
    }

    size_t edge_count = (size_t)node_count * (size_t)k;
    if (edge_count > (size_t)INT_MAX ||
        (*env)->GetArrayLength(env, j_neighbors) != (jsize)edge_count ||
        (*env)->GetArrayLength(env, j_choice_counts) != (jsize)edge_count ||
        (*env)->GetArrayLength(env, j_terminal) != node_count ||
        (*env)->GetArrayLength(env, j_terminal_link_mass) != node_count ||
        !cancellation_check) {
        throw_graph_explorer_error(
            env, "java/lang/IllegalArgumentException", "Graph Explorer array size mismatch");
        return -1;
    }

    jint *neighbors = NULL;
    jbyte *choice_counts = NULL;
    jdouble *terminal = NULL;
    jdouble *terminal_link_mass = NULL;
    double *current_probability = NULL;
    double *next_probability = NULL;
    jint *current_states = NULL;
    jint *next_states = NULL;
    jint result = -1;

    neighbors = (*env)->GetIntArrayElements(env, j_neighbors, NULL);
    choice_counts = (*env)->GetByteArrayElements(env, j_choice_counts, NULL);
    terminal = (*env)->GetDoubleArrayElements(env, j_terminal, NULL);
    terminal_link_mass = (*env)->GetDoubleArrayElements(env, j_terminal_link_mass, NULL);
    if (!neighbors || !choice_counts || !terminal || !terminal_link_mass) goto cleanup;

    current_probability = (double *)calloc(edge_count, sizeof(double));
    next_probability = (double *)calloc(edge_count, sizeof(double));
    current_states = (jint *)malloc(edge_count * sizeof(jint));
    next_states = (jint *)malloc(edge_count * sizeof(jint));
    if (!current_probability || !next_probability || !current_states || !next_states) {
        throw_graph_explorer_error(
            env, "java/lang/OutOfMemoryError", "Graph Explorer workspace allocation failed");
        goto cleanup;
    }

    jclass callback_class = (*env)->GetObjectClass(env, cancellation_check);
    if (!callback_class) goto cleanup;
    jmethodID invoke = (*env)->GetMethodID(
        env, callback_class, "invoke", "()Ljava/lang/Object;");
    (*env)->DeleteLocalRef(env, callback_class);
    if (!invoke) goto cleanup;

    int seed_row = seed * k;
    int initial_choice_count = 0;
    for (int slot = 0; slot < k; slot++) {
        if (neighbors[seed_row + slot] >= 0) initial_choice_count++;
    }

    int current_count = 0;
    if (initial_choice_count == 0) {
        terminal[seed] = 1.0;
    } else {
        double initial_probability = 1.0 / (double)initial_choice_count;
        for (int slot = 0; slot < k; slot++) {
            int state = seed_row + slot;
            if (neighbors[state] < 0) continue;
            current_probability[state] = initial_probability;
            current_states[current_count++] = state;
        }
    }

    const double alpha = (double)stop_probability;
    const double continuation_scale = 1.0 - alpha;
    int evaluated_links = 0;

    for (int link_count = 1; link_count <= max_links && current_count > 0; link_count++) {
        jobject callback_result = (*env)->CallObjectMethod(env, cancellation_check, invoke);
        if (callback_result) (*env)->DeleteLocalRef(env, callback_result);
        if ((*env)->ExceptionCheck(env)) goto cleanup;

        evaluated_links = link_count;
        int next_count = 0;
        for (int active_offset = 0; active_offset < current_count; active_offset++) {
            int state = current_states[active_offset];
            double probability = current_probability[state];
            current_probability[state] = 0.0;
            if (probability == 0.0) continue;

            int current = neighbors[state];
            if (current < 0 || current >= node_count) {
                throw_graph_explorer_error(
                    env, "java/lang/IllegalStateException", "active graph state is invalid");
                goto cleanup;
            }
            unsigned int choice_count = (unsigned char)choice_counts[state];
            int must_terminate = link_count == max_links || choice_count == 0;
            double stopped_mass = must_terminate ? probability : alpha * probability;
            terminal[current] += stopped_mass;
            terminal_link_mass[current] += (double)link_count * stopped_mass;

            if (must_terminate || continuation_scale == 0.0) continue;

            double contribution = probability * continuation_scale / (double)choice_count;
            if (contribution == 0.0) continue;
            int previous = state / k;
            int next_row = current * k;
            for (int slot = 0; slot < k; slot++) {
                int next_state = next_row + slot;
                int following = neighbors[next_state];
                if (following < 0 || following == previous) continue;
                if (next_probability[next_state] == 0.0) {
                    next_states[next_count++] = next_state;
                }
                next_probability[next_state] += contribution;
            }
        }

        double *probability_swap = current_probability;
        current_probability = next_probability;
        next_probability = probability_swap;
        jint *state_swap = current_states;
        current_states = next_states;
        next_states = state_swap;
        current_count = next_count;
    }

    jobject callback_result = (*env)->CallObjectMethod(env, cancellation_check, invoke);
    if (callback_result) (*env)->DeleteLocalRef(env, callback_result);
    if ((*env)->ExceptionCheck(env)) goto cleanup;
    result = evaluated_links;

cleanup:
    free(current_probability);
    free(next_probability);
    free(current_states);
    free(next_states);
    if (neighbors) (*env)->ReleaseIntArrayElements(env, j_neighbors, neighbors, JNI_ABORT);
    if (choice_counts) {
        (*env)->ReleaseByteArrayElements(env, j_choice_counts, choice_counts, JNI_ABORT);
    }
    if (terminal) (*env)->ReleaseDoubleArrayElements(env, j_terminal, terminal, 0);
    if (terminal_link_mass) {
        (*env)->ReleaseDoubleArrayElements(env, j_terminal_link_mass, terminal_link_mass, 0);
    }
    return result;
}

/* ── NEON dot product ─────────────────────────────────────── */

static float dot_product(const float *a, const float *b, int dim) {
#if defined(__aarch64__)
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    float32x4_t sum3 = vdupq_n_f32(0.0f);

    int i = 0;
    /* Unrolled 16-wide loop */
    for (; i + 15 < dim; i += 16) {
        sum0 = vfmaq_f32(sum0, vld1q_f32(a + i),      vld1q_f32(b + i));
        sum1 = vfmaq_f32(sum1, vld1q_f32(a + i + 4),   vld1q_f32(b + i + 4));
        sum2 = vfmaq_f32(sum2, vld1q_f32(a + i + 8),   vld1q_f32(b + i + 8));
        sum3 = vfmaq_f32(sum3, vld1q_f32(a + i + 12),  vld1q_f32(b + i + 12));
    }
    sum0 = vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3));

    /* Handle remaining 4-wide */
    for (; i + 3 < dim; i += 4) {
        sum0 = vfmaq_f32(sum0, vld1q_f32(a + i), vld1q_f32(b + i));
    }

    float result = vaddvq_f32(sum0);

    /* Scalar tail */
    for (; i < dim; i++) {
        result += a[i] * b[i];
    }
    return result;
#else
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) sum += a[i] * b[i];
    return sum;
#endif
}

/* ── k-means assignment ───────────────────────────────────── */
/*
 * For each of n points, find the nearest centroid (by dot product / cosine sim).
 * Returns labels[n] with the index of the best centroid per point.
 *
 * This replaces the Kotlin double loop:
 *   for i in 0..n: for j in 0..k: sim = dotProduct(emb[i], centroid[j])
 * which takes ~15s per iteration at 75K × 200 × 512d in scalar Kotlin.
 */
JNIEXPORT jintArray JNICALL
Java_com_powerampstartradio_indexing_NativeMath_nativeKmeansAssign(
    JNIEnv *env, jclass cls,
    jfloatArray jEmbeddings, jint n,
    jfloatArray jCentroids, jint k,
    jint dim)
{
    float *embeddings = (*env)->GetFloatArrayElements(env, jEmbeddings, NULL);
    float *centroids = (*env)->GetFloatArrayElements(env, jCentroids, NULL);
    if (!embeddings || !centroids) {
        if (embeddings) (*env)->ReleaseFloatArrayElements(env, jEmbeddings, embeddings, JNI_ABORT);
        if (centroids) (*env)->ReleaseFloatArrayElements(env, jCentroids, centroids, JNI_ABORT);
        return NULL;
    }

    jintArray jLabels = (*env)->NewIntArray(env, n);
    if (!jLabels) {
        (*env)->ReleaseFloatArrayElements(env, jEmbeddings, embeddings, JNI_ABORT);
        (*env)->ReleaseFloatArrayElements(env, jCentroids, centroids, JNI_ABORT);
        return NULL;
    }
    int *labels = (*env)->GetIntArrayElements(env, jLabels, NULL);

    for (int i = 0; i < n; i++) {
        const float *emb = embeddings + (long)i * dim;
        int bestK = 0;
        float bestSim = -FLT_MAX;
        for (int j = 0; j < k; j++) {
            float sim = dot_product(emb, centroids + (long)j * dim, dim);
            if (sim > bestSim) {
                bestSim = sim;
                bestK = j;
            }
        }
        labels[i] = bestK;
    }

    (*env)->ReleaseFloatArrayElements(env, jEmbeddings, embeddings, JNI_ABORT);
    (*env)->ReleaseFloatArrayElements(env, jCentroids, centroids, JNI_ABORT);
    (*env)->ReleaseIntArrayElements(env, jLabels, labels, 0);
    return jLabels;
}

/* ── Batch dot products ───────────────────────────────────── */
/*
 * Compute dot product of one query against n candidates.
 * Returns float[n] of similarities. Used for kNN search.
 */
JNIEXPORT jfloatArray JNICALL
Java_com_powerampstartradio_indexing_NativeMath_nativeBatchDot(
    JNIEnv *env, jclass cls,
    jfloatArray jQuery, jfloatArray jCandidates, jint n, jint dim)
{
    float *query = (*env)->GetFloatArrayElements(env, jQuery, NULL);
    float *candidates = (*env)->GetFloatArrayElements(env, jCandidates, NULL);
    if (!query || !candidates) {
        if (query) (*env)->ReleaseFloatArrayElements(env, jQuery, query, JNI_ABORT);
        if (candidates) (*env)->ReleaseFloatArrayElements(env, jCandidates, candidates, JNI_ABORT);
        return NULL;
    }

    jfloatArray jResult = (*env)->NewFloatArray(env, n);
    if (!jResult) {
        (*env)->ReleaseFloatArrayElements(env, jQuery, query, JNI_ABORT);
        (*env)->ReleaseFloatArrayElements(env, jCandidates, candidates, JNI_ABORT);
        return NULL;
    }
    float *result = (*env)->GetFloatArrayElements(env, jResult, NULL);

    for (int i = 0; i < n; i++) {
        result[i] = dot_product(query, candidates + (long)i * dim, dim);
    }

    (*env)->ReleaseFloatArrayElements(env, jQuery, query, JNI_ABORT);
    (*env)->ReleaseFloatArrayElements(env, jCandidates, candidates, JNI_ABORT);
    (*env)->ReleaseFloatArrayElements(env, jResult, result, 0);
    return jResult;
}

/* ── Covariance accumulation ──────────────────────────────── */
/*
 * Accumulate upper-triangle of covariance matrix: C += x * x^T
 * for a batch of vectors. Uses double precision for numerical stability.
 *
 * covariance: dim×dim double array (row-major, upper triangle filled)
 * vectors: batch×dim float array (row-major)
 */
JNIEXPORT void JNICALL
Java_com_powerampstartradio_indexing_NativeMath_nativeCovarianceAccum(
    JNIEnv *env, jclass cls,
    jdoubleArray jCovariance, jfloatArray jVectors, jint batch, jint dim)
{
    double *cov = (*env)->GetDoubleArrayElements(env, jCovariance, NULL);
    float *vecs = (*env)->GetFloatArrayElements(env, jVectors, NULL);
    if (!cov || !vecs) {
        if (cov) (*env)->ReleaseDoubleArrayElements(env, jCovariance, cov, JNI_ABORT);
        if (vecs) (*env)->ReleaseFloatArrayElements(env, jVectors, vecs, JNI_ABORT);
        return;
    }

    for (int b = 0; b < batch; b++) {
        const float *x = vecs + (long)b * dim;
        for (int i = 0; i < dim; i++) {
            double xi = (double)x[i];
            if (xi == 0.0) continue;
            double *row = cov + (long)i * dim;
            for (int j = i; j < dim; j++) {
                row[j] += xi * (double)x[j];
            }
        }
    }

    (*env)->ReleaseFloatArrayElements(env, jVectors, vecs, JNI_ABORT);
    (*env)->ReleaseDoubleArrayElements(env, jCovariance, cov, 0);
}

/* ── Matrix-vector multiply ───────────────────────────────── */
/*
 * result[rows] = matrix[rows, cols] * vector[cols]
 * Used for SVD projection (512×1024 × 1024 → 512).
 */
JNIEXPORT jfloatArray JNICALL
Java_com_powerampstartradio_indexing_NativeMath_nativeMatVecMul(
    JNIEnv *env, jclass cls,
    jfloatArray jMatrix, jint rows, jint cols, jfloatArray jVector)
{
    float *matrix = (*env)->GetFloatArrayElements(env, jMatrix, NULL);
    float *vector = (*env)->GetFloatArrayElements(env, jVector, NULL);
    if (!matrix || !vector) {
        if (matrix) (*env)->ReleaseFloatArrayElements(env, jMatrix, matrix, JNI_ABORT);
        if (vector) (*env)->ReleaseFloatArrayElements(env, jVector, vector, JNI_ABORT);
        return NULL;
    }

    jfloatArray jResult = (*env)->NewFloatArray(env, rows);
    if (!jResult) {
        (*env)->ReleaseFloatArrayElements(env, jMatrix, matrix, JNI_ABORT);
        (*env)->ReleaseFloatArrayElements(env, jVector, vector, JNI_ABORT);
        return NULL;
    }
    float *result = (*env)->GetFloatArrayElements(env, jResult, NULL);

    for (int i = 0; i < rows; i++) {
        result[i] = dot_product(matrix + (long)i * cols, vector, cols);
    }

    (*env)->ReleaseFloatArrayElements(env, jMatrix, matrix, JNI_ABORT);
    (*env)->ReleaseFloatArrayElements(env, jVector, vector, JNI_ABORT);
    (*env)->ReleaseFloatArrayElements(env, jResult, result, 0);
    return jResult;
}

/* ── Jacobi eigendecomposition ──────────────────────────── */
/*
 * Full cyclic Jacobi eigendecomposition for a symmetric n×n matrix.
 * Returns eigenvalues (sorted descending) + eigenvectors in a single
 * flat array of size n + n*n: [eigenvalues[n], eigenvectors[n*n]].
 *
 * Eigenvectors are stored column-major in the output: column i is the
 * eigenvector for eigenvalue[i].
 *
 * Moving the entire algorithm to C avoids JNI overhead per rotation
 * and lets the compiler optimize the inner loops (loop unrolling,
 * register allocation, prefetch). Expected ~5-10x speedup over Kotlin.
 */
JNIEXPORT jdoubleArray JNICALL
Java_com_powerampstartradio_indexing_NativeMath_nativeJacobiEigen(
    JNIEnv *env, jclass cls,
    jdoubleArray jMatrix, jint n, jint maxSweeps, jdouble eps)
{
    double *a = (*env)->GetDoubleArrayElements(env, jMatrix, NULL);
    if (!a) return NULL;

    /* Work on a copy so we don't modify the input */
    double *work = (double *)malloc((size_t)n * n * sizeof(double));
    double *v = (double *)malloc((size_t)n * n * sizeof(double));
    if (!work || !v) {
        free(work);
        free(v);
        (*env)->ReleaseDoubleArrayElements(env, jMatrix, a, JNI_ABORT);
        return NULL;
    }
    memcpy(work, a, (size_t)n * n * sizeof(double));
    (*env)->ReleaseDoubleArrayElements(env, jMatrix, a, JNI_ABORT);
    a = work;

    /* Initialize eigenvector matrix to identity */
    memset(v, 0, (size_t)n * n * sizeof(double));
    for (int i = 0; i < n; i++) v[(long)i * n + i] = 1.0;

    for (int sweep = 0; sweep < maxSweeps; sweep++) {
        /* Compute sum of squared off-diagonal elements */
        double offDiagSum = 0.0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double val = a[(long)i * n + j];
                offDiagSum += val * val;
            }
        }
        if (offDiagSum < eps) break;

        /* Threshold: higher for first 3 sweeps */
        double threshold = (sweep < 3) ? 0.2 * offDiagSum / ((double)n * n) : 0.0;

        /* Sweep through all upper-triangle pairs */
        for (int p = 0; p < n - 1; p++) {
            for (int q = p + 1; q < n; q++) {
                double apq = a[(long)p * n + q];
                if (apq > -threshold && apq < threshold) continue;

                double app = a[(long)p * n + p];
                double aqq = a[(long)q * n + q];
                double diff = aqq - app;

                double t;
                double abs_apq = apq < 0 ? -apq : apq;
                double abs_diff = diff < 0 ? -diff : diff;
                if (abs_apq < eps * abs_diff) {
                    t = apq / diff;
                } else {
                    double phi = diff / (2.0 * apq);
                    double abs_phi = phi < 0 ? -phi : phi;
                    double sign_phi = phi >= 0 ? 1.0 : -1.0;
                    t = sign_phi / (abs_phi + sqrt(1.0 + phi * phi));
                }

                double c = 1.0 / sqrt(1.0 + t * t);
                double s = t * c;
                double tau = s / (1.0 + c);

                /* Update diagonal */
                a[(long)p * n + p] -= t * apq;
                a[(long)q * n + q] += t * apq;
                a[(long)p * n + q] = 0.0;
                a[(long)q * n + p] = 0.0;

                /* Update off-diagonal elements for rows r != p, q */
                for (int r = 0; r < n; r++) {
                    if (r == p || r == q) continue;
                    double arp = a[(long)r * n + p];
                    double arq = a[(long)r * n + q];
                    double newP = arp - s * (arq + tau * arp);
                    double newQ = arq + s * (arp - tau * arq);
                    a[(long)r * n + p] = newP;
                    a[(long)p * n + r] = newP;
                    a[(long)r * n + q] = newQ;
                    a[(long)q * n + r] = newQ;
                }

                /* Accumulate eigenvectors */
                for (int r = 0; r < n; r++) {
                    double vrp = v[(long)r * n + p];
                    double vrq = v[(long)r * n + q];
                    v[(long)r * n + p] = vrp - s * (vrq + tau * vrp);
                    v[(long)r * n + q] = vrq + s * (vrp - tau * vrq);
                }
            }
        }
    }

    /* Sort eigenvalues descending and reorder eigenvectors */
    int *indices = (int *)malloc(n * sizeof(int));
    if (!indices) { free(a); free(v); return NULL; }
    for (int i = 0; i < n; i++) indices[i] = i;

    /* Simple insertion sort on eigenvalues (n=1024, fast enough) */
    for (int i = 1; i < n; i++) {
        int key = indices[i];
        double keyVal = a[(long)key * n + key];
        int j = i - 1;
        while (j >= 0 && a[(long)indices[j] * n + indices[j]] < keyVal) {
            indices[j + 1] = indices[j];
            j--;
        }
        indices[j + 1] = key;
    }

    /* Build output: eigenvalues[n] + eigenvectors[n*n] */
    jint outSize = n + n * n;
    jdoubleArray jResult = (*env)->NewDoubleArray(env, outSize);
    if (!jResult) { free(a); free(v); free(indices); return NULL; }
    double *result = (*env)->GetDoubleArrayElements(env, jResult, NULL);

    /* Eigenvalues */
    for (int i = 0; i < n; i++) {
        result[i] = a[(long)indices[i] * n + indices[i]];
    }

    /* Eigenvectors: column i of output = column indices[i] of v */
    for (int col = 0; col < n; col++) {
        int srcCol = indices[col];
        for (int row = 0; row < n; row++) {
            result[n + (long)row * n + col] = v[(long)row * n + srcCol];
        }
    }

    (*env)->ReleaseDoubleArrayElements(env, jResult, result, 0);
    free(a);
    free(v);
    free(indices);
    return jResult;
}

/* ── Top-K search on mmap'd embedding index ─────────────── */
/*
 * Find top-K most similar tracks by scanning a mmap'd .emb file directly.
 * Replaces the scalar Kotlin dotProduct loop in EmbeddingIndex.findTopK.
 *
 * The .emb format has track IDs at trackIdsOffset (int64[N]) and embeddings
 * at embeddingsOffset (float32[N × dim]), both little-endian.
 *
 * Uses NEON dot products + C min-heap for ~30x speedup over Kotlin/mmap.
 *
 * @param byteBuffer     mmap'd .emb file (direct ByteBuffer)
 * @param trackIdsOffset byte offset to int64 track ID array
 * @param embOffset      byte offset to float32 embedding array
 * @param jQuery         query vector [dim]
 * @param numTracks      total tracks in the index
 * @param dim            embedding dimension (e.g. 768)
 * @param topK           how many results to return
 * @param jExcludeIds    track IDs to skip (nullable)
 * @param outTrackIds    pre-allocated long[topK] for result track IDs
 * @param outScores      pre-allocated float[topK] for result scores
 * @return               actual number of results (≤ topK)
 */

typedef struct {
    int idx;
    float score;
} TopKEntry;

/* NaN is invalid ranking evidence and sorts after every numeric score. */
static int topk_is_worse(TopKEntry a, TopKEntry b, const int64_t *track_ids) {
    if (isnan(a.score)) return !isnan(b.score) ||
        (isnan(b.score) && track_ids[a.idx] > track_ids[b.idx]);
    if (isnan(b.score)) return 0;
    if (a.score < b.score) return 1;
    if (a.score > b.score) return 0;
    return track_ids[a.idx] > track_ids[b.idx];
}

static int topk_is_better(TopKEntry a, TopKEntry b, const int64_t *track_ids) {
    if (isnan(a.score)) return isnan(b.score) && track_ids[a.idx] < track_ids[b.idx];
    if (isnan(b.score)) return 1;
    if (a.score > b.score) return 1;
    if (a.score < b.score) return 0;
    return track_ids[a.idx] < track_ids[b.idx];
}

static void topk_sift_down(
    TopKEntry *heap,
    int size,
    int i,
    const int64_t *track_ids)
{
    while (1) {
        int worst = i;
        int left = 2 * i + 1;
        int right = 2 * i + 2;
        if (left < size && topk_is_worse(heap[left], heap[worst], track_ids)) worst = left;
        if (right < size && topk_is_worse(heap[right], heap[worst], track_ids)) worst = right;
        if (worst == i) break;
        TopKEntry tmp = heap[i];
        heap[i] = heap[worst];
        heap[worst] = tmp;
        i = worst;
    }
}

/* A worst-first min-heap becomes best-first after extracting each root to the end. */
static void topk_sort_best_first(
    TopKEntry *heap,
    int size,
    const int64_t *track_ids)
{
    for (int i = size / 2 - 1; i >= 0; i--) {
        topk_sift_down(heap, size, i, track_ids);
    }
    for (int end = size - 1; end > 0; end--) {
        TopKEntry tmp = heap[0];
        heap[0] = heap[end];
        heap[end] = tmp;
        topk_sift_down(heap, end, 0, track_ids);
    }
}

static int int64_compare(const void *left, const void *right) {
    int64_t a = *(const int64_t *)left;
    int64_t b = *(const int64_t *)right;
    return (a > b) - (a < b);
}

static int topk_is_excluded(
    int64_t track_id,
    const int64_t *exclude_ids,
    int exclude_count,
    int exclusions_are_sorted)
{
    if (exclusions_are_sorted) {
        int low = 0;
        int high = exclude_count - 1;
        while (low <= high) {
            int middle = low + (high - low) / 2;
            int64_t candidate = exclude_ids[middle];
            if (candidate < track_id) low = middle + 1;
            else if (candidate > track_id) high = middle - 1;
            else return 1;
        }
        return 0;
    }
    for (int i = 0; i < exclude_count; i++) {
        if (exclude_ids[i] == track_id) return 1;
    }
    return 0;
}

static int topk_check_cancellation(
    JNIEnv *env,
    jobject cancellation_check,
    jmethodID invoke)
{
    if (!cancellation_check) return 1;
    jobject callback_result = (*env)->CallObjectMethod(env, cancellation_check, invoke);
    if (callback_result) (*env)->DeleteLocalRef(env, callback_result);
    return !(*env)->ExceptionCheck(env);
}

static void throw_topk_error(JNIEnv *env, const char *class_name, const char *message) {
    jclass error_class = (*env)->FindClass(env, class_name);
    if (error_class) {
        (*env)->ThrowNew(env, error_class, message);
        (*env)->DeleteLocalRef(env, error_class);
    }
}

JNIEXPORT jint JNICALL
Java_com_powerampstartradio_indexing_NativeMath_nativeFindTopK(
    JNIEnv *env, jclass cls,
    jobject byteBuffer,
    jlong trackIdsOffset,
    jlong embOffset,
    jfloatArray jQuery,
    jint numTracks,
    jint dim,
    jint topK,
    jlongArray jExcludeIds,
    jlongArray outTrackIds,
    jfloatArray outScores,
    jobject cancellation_check)
{
    (void)cls;
    if (!byteBuffer || !jQuery || !outTrackIds || !outScores) {
        throw_topk_error(env, "java/lang/IllegalArgumentException", "null top-K argument");
        return -1;
    }
    uint8_t *base = (uint8_t *)(*env)->GetDirectBufferAddress(env, byteBuffer);
    jlong capacity = (*env)->GetDirectBufferCapacity(env, byteBuffer);
    if (!base || capacity < 0) {
        throw_topk_error(env, "java/lang/IllegalArgumentException", "index is not a direct buffer");
        return -1;
    }

    if (topK <= 0 || numTracks <= 0 || dim <= 0 || topK > numTracks ||
        (*env)->GetArrayLength(env, jQuery) != dim ||
        (*env)->GetArrayLength(env, outTrackIds) < topK ||
        (*env)->GetArrayLength(env, outScores) < topK ||
        trackIdsOffset < 0 || embOffset < 0 || trackIdsOffset > capacity ||
        embOffset > capacity) {
        throw_topk_error(env, "java/lang/IllegalArgumentException", "invalid top-K arguments");
        return -1;
    }
    uint64_t track_bytes = (uint64_t)numTracks * sizeof(int64_t);
    uint64_t row_bytes = (uint64_t)dim * sizeof(float);
    uint64_t available_ids = (uint64_t)(capacity - trackIdsOffset);
    uint64_t available_embeddings = (uint64_t)(capacity - embOffset);
    if (track_bytes > available_ids ||
        (uint64_t)numTracks > available_embeddings / row_bytes) {
        throw_topk_error(env, "java/lang/IllegalArgumentException", "top-K index range exceeds buffer");
        return -1;
    }

    const int64_t *trackIds = (const int64_t *)(base + trackIdsOffset);
    const float *embeddings = (const float *)(base + embOffset);
    float *query = NULL;
    jlong *rawExcludeIds = NULL;
    int64_t *sortedExcludeIds = NULL;
    TopKEntry *heap = NULL;
    jlong *outIds = NULL;
    float *outScr = NULL;
    int result = -1;

    jmethodID cancellation_invoke = NULL;
    if (cancellation_check) {
        jclass callback_class = (*env)->GetObjectClass(env, cancellation_check);
        if (!callback_class) goto cleanup;
        cancellation_invoke = (*env)->GetMethodID(
            env, callback_class, "invoke", "()Ljava/lang/Object;");
        (*env)->DeleteLocalRef(env, callback_class);
        if (!cancellation_invoke) goto cleanup;
    }

    query = (*env)->GetFloatArrayElements(env, jQuery, NULL);
    if (!query) goto cleanup;

    int excludeCount = jExcludeIds ? (*env)->GetArrayLength(env, jExcludeIds) : 0;
    const int exclusion_linear_threshold = 8;
    const int64_t *excludeIds = NULL;
    int exclusionsAreSorted = 0;
    if (excludeCount > 0) {
        rawExcludeIds = (*env)->GetLongArrayElements(env, jExcludeIds, NULL);
        if (!rawExcludeIds) goto cleanup;
        excludeIds = (const int64_t *)rawExcludeIds;
        if (excludeCount > exclusion_linear_threshold) {
            sortedExcludeIds = (int64_t *)malloc((size_t)excludeCount * sizeof(int64_t));
            if (!sortedExcludeIds) {
                throw_topk_error(env, "java/lang/OutOfMemoryError", "top-K exclusion allocation failed");
                goto cleanup;
            }
            memcpy(sortedExcludeIds, rawExcludeIds, (size_t)excludeCount * sizeof(int64_t));
            qsort(sortedExcludeIds, (size_t)excludeCount, sizeof(int64_t), int64_compare);
            excludeIds = sortedExcludeIds;
            exclusionsAreSorted = 1;
        }
    }

    heap = (TopKEntry *)malloc((size_t)topK * sizeof(TopKEntry));
    if (!heap) {
        throw_topk_error(env, "java/lang/OutOfMemoryError", "top-K heap allocation failed");
        goto cleanup;
    }
    int heapSize = 0;

    for (int i = 0; i < numTracks; i++) {
        if ((i & 4095) == 0 &&
            !topk_check_cancellation(env, cancellation_check, cancellation_invoke)) {
            goto cleanup;
        }
        int64_t track_id = trackIds[i];
        if (excludeCount > 0 && topk_is_excluded(
                track_id, excludeIds, excludeCount, exclusionsAreSorted)) continue;

        float score = dot_product(query, embeddings + (long)i * dim, dim);
        TopKEntry candidate = { .idx = i, .score = score };

        if (heapSize < topK) {
            heap[heapSize++] = candidate;
            /* Heapify once full */
            if (heapSize == topK) {
                for (int j = topK / 2 - 1; j >= 0; j--)
                    topk_sift_down(heap, heapSize, j, trackIds);
            }
        } else if (topk_is_better(candidate, heap[0], trackIds)) {
            heap[0] = candidate;
            topk_sift_down(heap, heapSize, 0, trackIds);
        }
    }

    if (!topk_check_cancellation(env, cancellation_check, cancellation_invoke)) goto cleanup;
    topk_sort_best_first(heap, heapSize, trackIds);

    /* Write results to output arrays */
    outIds = (*env)->GetLongArrayElements(env, outTrackIds, NULL);
    outScr = (*env)->GetFloatArrayElements(env, outScores, NULL);
    if (!outIds || !outScr) goto cleanup;
    for (int i = 0; i < heapSize; i++) {
        outIds[i] = trackIds[heap[i].idx];
        outScr[i] = heap[i].score;
    }
    result = heapSize;

cleanup:
    if (outIds) (*env)->ReleaseLongArrayElements(
        env, outTrackIds, outIds, result >= 0 ? 0 : JNI_ABORT);
    if (outScr) (*env)->ReleaseFloatArrayElements(
        env, outScores, outScr, result >= 0 ? 0 : JNI_ABORT);
    free(heap);
    free(sortedExcludeIds);
    if (rawExcludeIds) {
        (*env)->ReleaseLongArrayElements(env, jExcludeIds, rawExcludeIds, JNI_ABORT);
    }
    if (query) (*env)->ReleaseFloatArrayElements(env, jQuery, query, JNI_ABORT);

    return result;
}

JNIEXPORT jint JNICALL
Java_com_powerampstartradio_indexing_NativeMath_nativeRankFromSimilarities(
    JNIEnv *env, jclass cls,
    jobject byteBuffer,
    jlong trackIdsOffset,
    jfloatArray jSimilarities,
    jint numTracks,
    jint targetIndex)
{
    (void)cls;
    if (!byteBuffer || !jSimilarities || numTracks <= 0 ||
        targetIndex < 0 || targetIndex >= numTracks ||
        (*env)->GetArrayLength(env, jSimilarities) != numTracks) {
        throw_topk_error(env, "java/lang/IllegalArgumentException", "invalid rank arguments");
        return -1;
    }
    uint8_t *base = (uint8_t *)(*env)->GetDirectBufferAddress(env, byteBuffer);
    jlong capacity = (*env)->GetDirectBufferCapacity(env, byteBuffer);
    uint64_t track_bytes = (uint64_t)numTracks * sizeof(int64_t);
    if (!base || capacity < 0 || trackIdsOffset < 0 || trackIdsOffset > capacity ||
        track_bytes > (uint64_t)(capacity - trackIdsOffset)) {
        throw_topk_error(env, "java/lang/IllegalArgumentException", "rank IDs exceed buffer");
        return -1;
    }

    const int64_t *track_ids = (const int64_t *)(base + trackIdsOffset);
    float *similarities = (*env)->GetFloatArrayElements(env, jSimilarities, NULL);
    if (!similarities) return -1;
    float target_similarity = similarities[targetIndex];
    int64_t target_track_id = track_ids[targetIndex];
    int rank = 1;
    for (int i = 0; i < numTracks; i++) {
        if (i != targetIndex &&
            (similarities[i] > target_similarity ||
             (similarities[i] == target_similarity && track_ids[i] < target_track_id))) {
            rank++;
        }
    }
    (*env)->ReleaseFloatArrayElements(env, jSimilarities, similarities, JNI_ABORT);
    return rank;
}

/* ── All-pairs similarity on mmap'd embedding index ─────── */
/*
 * Compute dot product of one query against all N embeddings in a mmap'd
 * .emb file. Returns float[N] of similarities.
 *
 * Same NEON acceleration as nativeFindTopK but returns all scores instead
 * of top-K. Used for precomputing seed similarities for rank lookups.
 */
JNIEXPORT void JNICALL
Java_com_powerampstartradio_indexing_NativeMath_nativeAllSimilarities(
    JNIEnv *env, jclass cls,
    jobject byteBuffer,
    jlong embOffset,
    jfloatArray jQuery,
    jint numTracks,
    jint dim,
    jfloatArray outScores)
{
    uint8_t *base = (uint8_t *)(*env)->GetDirectBufferAddress(env, byteBuffer);
    if (!base) {
        LOGE("nativeAllSimilarities: not a direct ByteBuffer");
        return;
    }

    const float *embeddings = (const float *)(base + embOffset);
    float *query = (*env)->GetFloatArrayElements(env, jQuery, NULL);
    if (!query) return;

    float *scores = (*env)->GetFloatArrayElements(env, outScores, NULL);
    if (!scores) {
        (*env)->ReleaseFloatArrayElements(env, jQuery, query, JNI_ABORT);
        return;
    }

    for (int i = 0; i < numTracks; i++) {
        scores[i] = dot_product(query, embeddings + (long)i * dim, dim);
    }

    (*env)->ReleaseFloatArrayElements(env, jQuery, query, JNI_ABORT);
    (*env)->ReleaseFloatArrayElements(env, outScores, scores, 0);
}

/* Score aligned pairs of mmap rows with the identical reduction used by nativeFindTopK. */
JNIEXPORT void JNICALL
Java_com_powerampstartradio_indexing_NativeMath_nativePairSimilarities(
    JNIEnv *env, jclass cls,
    jobject byteBuffer,
    jlong embOffset,
    jintArray jLeftIndices,
    jintArray jRightIndices,
    jint numTracks,
    jint dim,
    jfloatArray outScores,
    jint pair_count,
    jobject cancellation_check)
{
    (void)cls;
    if (!byteBuffer || !jLeftIndices || !jRightIndices || !outScores ||
        numTracks <= 0 || dim <= 0) {
        throw_topk_error(env, "java/lang/IllegalArgumentException", "invalid pair-dot arguments");
        return;
    }
    jsize left_capacity = (*env)->GetArrayLength(env, jLeftIndices);
    jsize right_capacity = (*env)->GetArrayLength(env, jRightIndices);
    jsize score_capacity = (*env)->GetArrayLength(env, outScores);
    if (pair_count < 0 || pair_count > left_capacity ||
        pair_count > right_capacity || pair_count > score_capacity) {
        throw_topk_error(env, "java/lang/IllegalArgumentException", "pair-dot array size mismatch");
        return;
    }
    uint8_t *base = (uint8_t *)(*env)->GetDirectBufferAddress(env, byteBuffer);
    jlong capacity = (*env)->GetDirectBufferCapacity(env, byteBuffer);
    uint64_t row_bytes = (uint64_t)dim * sizeof(float);
    if (!base || capacity < 0 || embOffset < 0 || embOffset > capacity ||
        (uint64_t)numTracks > (uint64_t)(capacity - embOffset) / row_bytes) {
        throw_topk_error(env, "java/lang/IllegalArgumentException", "pair-dot index exceeds buffer");
        return;
    }

    jmethodID cancellation_invoke = NULL;
    if (cancellation_check) {
        jclass callback_class = (*env)->GetObjectClass(env, cancellation_check);
        if (!callback_class) return;
        cancellation_invoke = (*env)->GetMethodID(
            env, callback_class, "invoke", "()Ljava/lang/Object;");
        (*env)->DeleteLocalRef(env, callback_class);
        if (!cancellation_invoke) return;
    }

    jint *left_indices = (*env)->GetIntArrayElements(env, jLeftIndices, NULL);
    jint *right_indices = (*env)->GetIntArrayElements(env, jRightIndices, NULL);
    jfloat *scores = (*env)->GetFloatArrayElements(env, outScores, NULL);
    int succeeded = 0;
    if (!left_indices || !right_indices || !scores) goto pair_cleanup;

    const float *embeddings = (const float *)(base + embOffset);
    for (jsize pair = 0; pair < pair_count; pair++) {
        if ((pair & 1023) == 0 &&
            !topk_check_cancellation(env, cancellation_check, cancellation_invoke)) {
            goto pair_cleanup;
        }
        jint left = left_indices[pair];
        jint right = right_indices[pair];
        if (left < 0 || left >= numTracks || right < 0 || right >= numTracks) {
            throw_topk_error(env, "java/lang/IllegalArgumentException", "pair-dot row is out of range");
            goto pair_cleanup;
        }
        scores[pair] = dot_product(
            embeddings + (long)left * dim,
            embeddings + (long)right * dim,
            dim);
    }
    if (!topk_check_cancellation(env, cancellation_check, cancellation_invoke)) {
        goto pair_cleanup;
    }
    succeeded = 1;

pair_cleanup:
    if (left_indices) {
        (*env)->ReleaseIntArrayElements(env, jLeftIndices, left_indices, JNI_ABORT);
    }
    if (right_indices) {
        (*env)->ReleaseIntArrayElements(env, jRightIndices, right_indices, JNI_ABORT);
    }
    if (scores) {
        (*env)->ReleaseFloatArrayElements(env, outScores, scores, succeeded ? 0 : JNI_ABORT);
    }
}

/* ── Polyphase FIR resampler (NEON-accelerated) ─────────── */
/*
 * High-quality audio resampling equivalent to scipy.signal.resample_poly.
 * Uses a Kaiser-windowed sinc FIR filter decomposed into polyphase filter
 * banks, with NEON-accelerated convolution.
 *
 * For 44100→24000Hz: up=80, down=147, filter=2941 taps, 37 taps/phase.
 * Each output sample requires 37 multiply-accumulates (~10 NEON ops).
 * Total for a 4-min track: ~60ms vs ~15000ms for soxr HQ.
 *
 * Quality: identical to scipy resample_poly (cosine 1.000 vs soxr HQ
 * in per-window MERT feature comparison across 3 test tracks).
 */

#include <time.h>

static long nanos_math(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000000L + ts.tv_nsec;
}

/* Modified Bessel function I0 (for Kaiser window) */
static double bessel_i0(double x) {
    double sum = 1.0, term = 1.0;
    double y = x * x * 0.25;
    for (int k = 1; k <= 30; k++) {
        term *= y / ((double)k * k);
        sum += term;
        if (term < sum * 1e-16) break;
    }
    return sum;
}

static int gcd_int(int a, int b) {
    while (b) { int t = b; b = a % b; a = t; }
    return a;
}

typedef struct {
    int up;
    int down;
    int taps;
    int center_tap;
    float *phases;
} polyphase_plan;

static void free_polyphase_plan(polyphase_plan *plan) {
    free(plan->phases);
    plan->phases = NULL;
}

/* Build the one canonical filter/phase layout used by whole and aligned ranges. */
static int build_polyphase_plan(int from_rate, int to_rate, polyphase_plan *plan) {
    memset(plan, 0, sizeof(*plan));
    if (from_rate <= 0 || to_rate <= 0) return 0;

    int g = gcd_int(from_rate, to_rate);
    plan->up = to_rate / g;
    plan->down = from_rate / g;
    int max_rate = (plan->up > plan->down) ? plan->up : plan->down;
    if (max_rate > (INT_MAX - 1) / 20) return 0;

    double cutoff = 1.0 / max_rate;
    int half_len = 10 * max_rate;
    int filt_len = 2 * half_len + 1;
    float *filt = (float *)malloc((size_t)filt_len * sizeof(float));
    if (!filt) return 0;

    double beta = 5.0;
    double inv_i0 = 1.0 / bessel_i0(beta);
    double half = (double)half_len;
    for (int n = 0; n < filt_len; n++) {
        double t = n - half;
        double sinc = (fabs(t) < 1e-10) ? cutoff
                    : sin(M_PI * cutoff * t) / (M_PI * t);
        double r = t / half;
        double w_arg = 1.0 - r * r;
        double window = (w_arg > 0) ? bessel_i0(beta * sqrt(w_arg)) * inv_i0 : 0.0;
        filt[n] = (float)(sinc * window * plan->up);
    }

    plan->taps = filt_len / plan->up + (filt_len % plan->up != 0);
    if ((size_t)plan->up > SIZE_MAX / (size_t)plan->taps ||
        (size_t)plan->up * (size_t)plan->taps > SIZE_MAX / sizeof(float)) {
        free(filt);
        return 0;
    }
    plan->phases = (float *)calloc(
        (size_t)plan->up * (size_t)plan->taps,
        sizeof(float));
    if (!plan->phases) {
        free(filt);
        return 0;
    }
    for (int p = 0; p < plan->up; p++) {
        for (int t = 0; t < plan->taps; t++) {
            int fi = p + t * plan->up;
            if (fi < filt_len) plan->phases[p * plan->taps + t] = filt[fi];
        }
    }
    free(filt);

    /* Convolution is correlation with each (non-symmetric) phase reversed. */
    for (int p = 0; p < plan->up; p++) {
        float *phase = &plan->phases[p * plan->taps];
        for (int t = 0; t < plan->taps / 2; t++) {
            int opposite = plan->taps - 1 - t;
            float tmp = phase[t];
            phase[t] = phase[opposite];
            phase[opposite] = tmp;
        }
    }
    plan->center_tap = (plan->taps - 1) / 2;
    return 1;
}

static int polyphase_output_length(
    long long total_input_samples,
    const polyphase_plan *plan,
    long long *output_length)
{
    if (total_input_samples < 0 ||
        total_input_samples > (LLONG_MAX - (plan->down - 1)) / plan->up) {
        return 0;
    }
    *output_length = (total_input_samples * plan->up + plan->down - 1) / plan->down;
    return 1;
}

static void polyphase_safe_region(
    long long total_input_samples,
    long long total_output_samples,
    const polyphase_plan *plan,
    long long *first_safe,
    long long *last_safe)
{
    *first_safe = 0;
    while (*first_safe < total_output_samples) {
        long long pos = *first_safe * plan->down;
        long long input_idx = pos / plan->up - plan->center_tap;
        if (input_idx >= 0) break;
        (*first_safe)++;
    }

    *last_safe = total_output_samples;
    while (*last_safe > *first_safe) {
        long long pos = (*last_safe - 1) * plan->down;
        long long input_idx = pos / plan->up - plan->center_tap + plan->taps - 1;
        if (input_idx < total_input_samples) break;
        (*last_safe)--;
    }
}

static int polyphase_has_context(
    long long input_start,
    int input_count,
    long long total_input_samples,
    long long output_start,
    int output_count,
    const polyphase_plan *plan)
{
    if (output_count == 0) return 1;
    long long first_pos = output_start * plan->down;
    long long last_pos = (output_start + output_count - 1) * plan->down;
    long long needed_start = first_pos / plan->up - plan->center_tap;
    long long needed_end = last_pos / plan->up - plan->center_tap + plan->taps;
    if (needed_start < 0) needed_start = 0;
    if (needed_end > total_input_samples) needed_end = total_input_samples;
    long long input_end = input_start + input_count;
    return input_start <= needed_start && input_end >= needed_end;
}

/* Render a globally addressed output range while preserving whole-track rounding. */
static void render_polyphase_range(
    const float *input,
    long long input_start,
    long long total_input_samples,
    long long total_output_samples,
    long long output_start,
    int output_count,
    const polyphase_plan *plan,
    float *output)
{
    long long first_safe, last_safe;
    polyphase_safe_region(
        total_input_samples, total_output_samples, plan, &first_safe, &last_safe);
    long long output_end = output_start + output_count;

    long long leading_end = output_end < first_safe ? output_end : first_safe;
    for (long long n = output_start; n < leading_end; n++) {
        long long pos = n * plan->down;
        int phase = (int)(pos % plan->up);
        long long k_start = pos / plan->up - plan->center_tap;
        float *ph = &plan->phases[phase * plan->taps];
        float sum = 0.0f;
        for (int t = 0; t < plan->taps; t++) {
            long long source_index = k_start + t;
            if (source_index >= 0 && source_index < total_input_samples) {
                sum += ph[t] * input[source_index - input_start];
            }
        }
        output[n - output_start] = sum;
    }

    long long middle_start = output_start > first_safe ? output_start : first_safe;
    long long middle_end = output_end < last_safe ? output_end : last_safe;
    for (long long n = middle_start; n < middle_end; n++) {
        long long pos = n * plan->down;
        int phase = (int)(pos % plan->up);
        long long k_start = pos / plan->up - plan->center_tap;
        float *ph = &plan->phases[phase * plan->taps];
        const float *src = &input[k_start - input_start];

#if defined(__aarch64__)
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);
        int t = 0;
        for (; t + 7 < plan->taps; t += 8) {
            acc0 = vfmaq_f32(acc0, vld1q_f32(&ph[t]),     vld1q_f32(&src[t]));
            acc1 = vfmaq_f32(acc1, vld1q_f32(&ph[t + 4]), vld1q_f32(&src[t + 4]));
        }
        for (; t + 3 < plan->taps; t += 4) {
            acc0 = vfmaq_f32(acc0, vld1q_f32(&ph[t]), vld1q_f32(&src[t]));
        }
        float sum = vaddvq_f32(vaddq_f32(acc0, acc1));
        for (; t < plan->taps; t++) sum += ph[t] * src[t];
#else
        float sum = 0.0f;
        for (int t = 0; t < plan->taps; t++) sum += ph[t] * src[t];
#endif
        output[n - output_start] = sum;
    }

    long long trailing_start = output_start > last_safe ? output_start : last_safe;
    for (long long n = trailing_start; n < output_end; n++) {
        long long pos = n * plan->down;
        int phase = (int)(pos % plan->up);
        long long k_start = pos / plan->up - plan->center_tap;
        float *ph = &plan->phases[phase * plan->taps];
        float sum = 0.0f;
        for (int t = 0; t < plan->taps; t++) {
            long long source_index = k_start + t;
            if (source_index >= 0 && source_index < total_input_samples) {
                sum += ph[t] * input[source_index - input_start];
            }
        }
        output[n - output_start] = sum;
    }
}

JNIEXPORT jfloatArray JNICALL
Java_com_powerampstartradio_indexing_NativeMath_nativeResamplePolyphase(
    JNIEnv *env, jclass cls,
    jfloatArray inputArray, jint fromRate, jint toRate)
{
    jsize n_in = (*env)->GetArrayLength(env, inputArray);
    if (n_in == 0 || fromRate == toRate) return inputArray;
    if (fromRate <= 0 || toRate <= 0) return NULL;

    long t0 = nanos_math();

    polyphase_plan plan;
    if (!build_polyphase_plan((int)fromRate, (int)toRate, &plan)) return NULL;
    long long n_out_ll;
    if (!polyphase_output_length(n_in, &plan, &n_out_ll) || n_out_ll > INT_MAX) {
        free_polyphase_plan(&plan);
        return NULL;
    }
    int n_out = (int)n_out_ll;

    jfloat *input = (*env)->GetFloatArrayElements(env, inputArray, NULL);
    if (!input) {
        free_polyphase_plan(&plan);
        return NULL;
    }

    float *output = (float *)malloc(n_out * sizeof(float));
    if (!output) {
        free_polyphase_plan(&plan);
        (*env)->ReleaseFloatArrayElements(env, inputArray, input, JNI_ABORT);
        return NULL;
    }

    long t1 = nanos_math();
    render_polyphase_range(input, 0, n_in, n_out_ll, 0, n_out, &plan, output);

    long t2 = nanos_math();

    int up = plan.up;
    int down = plan.down;
    int taps = plan.taps;
    free_polyphase_plan(&plan);
    (*env)->ReleaseFloatArrayElements(env, inputArray, input, JNI_ABORT);

    jfloatArray result = (*env)->NewFloatArray(env, n_out);
    if (!result) { free(output); return NULL; }
    (*env)->SetFloatArrayRegion(env, result, 0, n_out, output);
    free(output);

    long t3 = nanos_math();

    __android_log_print(ANDROID_LOG_INFO, TAG,
        "TIMING: polyphase_resample %d->%dHz (up=%d,down=%d,taps=%d) "
        "%d->%d samples: setup=%ldms resample=%ldms jni_out=%ldms total=%ldms",
        fromRate, toRate, up, down, taps,
        n_in, n_out,
        (t1 - t0) / 1000000, (t2 - t1) / 1000000,
        (t3 - t2) / 1000000, (t3 - t0) / 1000000);

    return result;
}

JNIEXPORT jfloatArray JNICALL
Java_com_powerampstartradio_indexing_NativeMath_nativeResamplePolyphaseAligned(
    JNIEnv *env, jclass cls,
    jfloatArray inputArray, jint fromRate, jint toRate,
    jlong inputStartSample, jlong totalInputSamples,
    jlong outputStartSample, jint outputSampleCount)
{
    jsize n_in = (*env)->GetArrayLength(env, inputArray);
    if (fromRate <= 0 || toRate <= 0 || fromRate == toRate ||
        inputStartSample < 0 || totalInputSamples < 0 ||
        inputStartSample > totalInputSamples ||
        (jlong)n_in > totalInputSamples - inputStartSample ||
        outputStartSample < 0 || outputSampleCount < 0) {
        LOGE("Invalid aligned polyphase range");
        return NULL;
    }

    polyphase_plan plan;
    if (!build_polyphase_plan((int)fromRate, (int)toRate, &plan)) return NULL;
    long long total_output_samples;
    if (!polyphase_output_length(totalInputSamples, &plan, &total_output_samples) ||
        outputStartSample > total_output_samples ||
        (jlong)outputSampleCount > total_output_samples - outputStartSample) {
        LOGE("Aligned polyphase output range exceeds whole-track output");
        free_polyphase_plan(&plan);
        return NULL;
    }
    if (!polyphase_has_context(
            inputStartSample, n_in, totalInputSamples,
            outputStartSample, outputSampleCount, &plan)) {
        LOGE("Aligned polyphase input slice lacks required FIR context");
        free_polyphase_plan(&plan);
        return NULL;
    }

    jfloatArray result = (*env)->NewFloatArray(env, outputSampleCount);
    if (!result || outputSampleCount == 0) {
        free_polyphase_plan(&plan);
        return result;
    }
    jfloat *input = (*env)->GetFloatArrayElements(env, inputArray, NULL);
    if (!input) {
        free_polyphase_plan(&plan);
        return NULL;
    }
    float *output = (float *)malloc((size_t)outputSampleCount * sizeof(float));
    if (!output) {
        (*env)->ReleaseFloatArrayElements(env, inputArray, input, JNI_ABORT);
        free_polyphase_plan(&plan);
        return NULL;
    }

    render_polyphase_range(
        input, inputStartSample, totalInputSamples, total_output_samples,
        outputStartSample, outputSampleCount, &plan, output);
    (*env)->ReleaseFloatArrayElements(env, inputArray, input, JNI_ABORT);
    free_polyphase_plan(&plan);
    (*env)->SetFloatArrayRegion(env, result, 0, outputSampleCount, output);
    free(output);
    return result;
}

/* ── TorchAudio default Hann resampler V1 ────────────────── */
/*
 * Pinned equivalent of torchaudio.transforms.Resample defaults:
 *   resampling_method=sinc_interp_hann
 *   lowpass_filter_width=6
 *   rolloff=0.99
 *   dtype=None (float64 coefficient construction, then float32 kernel)
 *
 * This is deliberately separate from the legacy Kaiser polyphase entry points.
 * TorchAudio constructs one correlation kernel per reduced output phase, pads the
 * input by [width, width + reduced_input_rate], strides by reduced_input_rate,
 * flattens time-major/phase-minor, then truncates to its float32-derived target
 * length. Global output addressing below preserves those exact coordinates.
 */

#define TORCHAUDIO_HANN_V1_FILTER_WIDTH 6
#define TORCHAUDIO_HANN_V1_ROLLOFF 0.99

typedef struct {
    int output_phases;
    int input_stride;
    int width;
    int kernel_size;
    float *kernels;
} torchaudio_hann_v1_plan;

static void free_torchaudio_hann_v1_plan(torchaudio_hann_v1_plan *plan) {
    free(plan->kernels);
    plan->kernels = NULL;
}

static int build_torchaudio_hann_v1_plan(
    int from_rate,
    int to_rate,
    torchaudio_hann_v1_plan *plan)
{
    memset(plan, 0, sizeof(*plan));
    if (from_rate <= 0 || to_rate <= 0) return 0;

    int divisor = gcd_int(from_rate, to_rate);
    plan->input_stride = from_rate / divisor;
    plan->output_phases = to_rate / divisor;
    double base_frequency =
        (plan->input_stride < plan->output_phases
            ? plan->input_stride
            : plan->output_phases) * TORCHAUDIO_HANN_V1_ROLLOFF;
    if (!(base_frequency > 0.0)) return 0;

    double width_value =
        TORCHAUDIO_HANN_V1_FILTER_WIDTH * plan->input_stride / base_frequency;
    if (width_value > INT_MAX) return 0;
    plan->width = (int)ceil(width_value);
    if (plan->width > (INT_MAX - plan->input_stride) / 2) return 0;
    plan->kernel_size = plan->input_stride + 2 * plan->width;

    if ((size_t)plan->output_phases > SIZE_MAX / (size_t)plan->kernel_size ||
        (size_t)plan->output_phases * (size_t)plan->kernel_size >
            SIZE_MAX / sizeof(float)) {
        return 0;
    }
    plan->kernels = malloc(
        (size_t)plan->output_phases * (size_t)plan->kernel_size * sizeof(float));
    if (!plan->kernels) return 0;

    const double scale = base_frequency / plan->input_stride;
    for (int phase = 0; phase < plan->output_phases; phase++) {
        for (int tap = 0; tap < plan->kernel_size; tap++) {
            double index =
                (double)(tap - plan->width) / (double)plan->input_stride;
            double t =
                -(double)phase / (double)plan->output_phases + index;
            t *= base_frequency;
            if (t < -TORCHAUDIO_HANN_V1_FILTER_WIDTH) {
                t = -TORCHAUDIO_HANN_V1_FILTER_WIDTH;
            } else if (t > TORCHAUDIO_HANN_V1_FILTER_WIDTH) {
                t = TORCHAUDIO_HANN_V1_FILTER_WIDTH;
            }

            double window_phase =
                t * M_PI / TORCHAUDIO_HANN_V1_FILTER_WIDTH / 2.0;
            double window_cosine = cos(window_phase);
            double window = window_cosine * window_cosine;
            double sinc_phase = t * M_PI;
            double sinc = sinc_phase == 0.0
                ? 1.0
                : sin(sinc_phase) / sinc_phase;
            plan->kernels[phase * plan->kernel_size + tap] =
                (float)(sinc * window * scale);
        }
    }
    return 1;
}

/* TorchAudio 2.10 materializes the Python ratio as a default float32 tensor. */
static int torchaudio_hann_v1_output_length(
    long long total_input_samples,
    const torchaudio_hann_v1_plan *plan,
    long long *output_length)
{
    if (total_input_samples < 0) return 0;
    double exact_ratio =
        (double)plan->output_phases * (double)total_input_samples /
        (double)plan->input_stride;
    float torch_scalar = (float)exact_ratio;
    if (!isfinite(torch_scalar) || torch_scalar < 0.0f) return 0;
    double rounded = ceil((double)torch_scalar);
    if (rounded > (double)LLONG_MAX) return 0;
    *output_length = (long long)rounded;
    return 1;
}

static int torchaudio_hann_v1_has_context(
    long long input_start,
    int input_count,
    long long total_input_samples,
    long long output_start,
    int output_count,
    const torchaudio_hann_v1_plan *plan)
{
    if (output_count == 0) return 1;
    long long output_end = output_start + output_count;
    long long first_block = output_start / plan->output_phases;
    long long last_block = (output_end - 1) / plan->output_phases;
    long long needed_start =
        first_block * plan->input_stride - plan->width;
    long long needed_end =
        last_block * plan->input_stride - plan->width + plan->kernel_size;
    if (needed_start < 0) needed_start = 0;
    if (needed_end > total_input_samples) needed_end = total_input_samples;
    long long input_end = input_start + input_count;
    return input_start <= needed_start && input_end >= needed_end;
}

static void torchaudio_hann_v1_safe_region(
    long long total_input_samples,
    long long total_output_samples,
    const torchaudio_hann_v1_plan *plan,
    long long *first_safe,
    long long *last_safe)
{
    *first_safe = 0;
    while (*first_safe < total_output_samples) {
        long long block = *first_safe / plan->output_phases;
        long long source_start = block * plan->input_stride - plan->width;
        if (source_start >= 0) break;
        (*first_safe)++;
    }

    *last_safe = total_output_samples;
    while (*last_safe > *first_safe) {
        long long block = (*last_safe - 1) / plan->output_phases;
        long long source_end =
            block * plan->input_stride - plan->width + plan->kernel_size;
        if (source_end <= total_input_samples) break;
        (*last_safe)--;
    }
}

static void render_torchaudio_hann_v1_range(
    const float *input,
    long long input_start,
    long long total_input_samples,
    long long total_output_samples,
    long long output_start,
    int output_count,
    const torchaudio_hann_v1_plan *plan,
    float *output)
{
    long long first_safe, last_safe;
    torchaudio_hann_v1_safe_region(
        total_input_samples, total_output_samples, plan, &first_safe, &last_safe);
    long long output_end = output_start + output_count;

    long long leading_end = output_end < first_safe ? output_end : first_safe;
    for (long long index = output_start; index < leading_end; index++) {
        long long block = index / plan->output_phases;
        int phase = (int)(index % plan->output_phases);
        long long source_start = block * plan->input_stride - plan->width;
        const float *kernel = &plan->kernels[phase * plan->kernel_size];
        float sum = 0.0f;
        for (int tap = 0; tap < plan->kernel_size; tap++) {
            long long source_index = source_start + tap;
            if (source_index >= 0 && source_index < total_input_samples) {
                sum += kernel[tap] * input[source_index - input_start];
            }
        }
        output[index - output_start] = sum;
    }

    long long middle_start = output_start > first_safe ? output_start : first_safe;
    long long middle_end = output_end < last_safe ? output_end : last_safe;
    for (long long index = middle_start; index < middle_end; index++) {
        long long block = index / plan->output_phases;
        int phase = (int)(index % plan->output_phases);
        long long source_start = block * plan->input_stride - plan->width;
        const float *kernel = &plan->kernels[phase * plan->kernel_size];
        const float *source = &input[source_start - input_start];
#if defined(__aarch64__)
        float32x4_t accumulator0 = vdupq_n_f32(0.0f);
        float32x4_t accumulator1 = vdupq_n_f32(0.0f);
        int tap = 0;
        for (; tap + 7 < plan->kernel_size; tap += 8) {
            accumulator0 = vfmaq_f32(
                accumulator0, vld1q_f32(&kernel[tap]), vld1q_f32(&source[tap]));
            accumulator1 = vfmaq_f32(
                accumulator1, vld1q_f32(&kernel[tap + 4]), vld1q_f32(&source[tap + 4]));
        }
        for (; tap + 3 < plan->kernel_size; tap += 4) {
            accumulator0 = vfmaq_f32(
                accumulator0, vld1q_f32(&kernel[tap]), vld1q_f32(&source[tap]));
        }
        float sum = vaddvq_f32(vaddq_f32(accumulator0, accumulator1));
        for (; tap < plan->kernel_size; tap++) sum += kernel[tap] * source[tap];
#else
        float sum = 0.0f;
        for (int tap = 0; tap < plan->kernel_size; tap++) {
            sum += kernel[tap] * source[tap];
        }
#endif
        output[index - output_start] = sum;
    }

    long long trailing_start = output_start > last_safe ? output_start : last_safe;
    for (long long index = trailing_start; index < output_end; index++) {
        long long block = index / plan->output_phases;
        int phase = (int)(index % plan->output_phases);
        long long source_start = block * plan->input_stride - plan->width;
        const float *kernel = &plan->kernels[phase * plan->kernel_size];
        float sum = 0.0f;
        for (int tap = 0; tap < plan->kernel_size; tap++) {
            long long source_index = source_start + tap;
            if (source_index >= 0 && source_index < total_input_samples) {
                sum += kernel[tap] * input[source_index - input_start];
            }
        }
        output[index - output_start] = sum;
    }
}

JNIEXPORT jfloatArray JNICALL
Java_com_powerampstartradio_indexing_NativeMath_nativeResampleTorchAudioHannV1(
    JNIEnv *env, jclass cls,
    jfloatArray inputArray, jint fromRate, jint toRate)
{
    jsize input_count = (*env)->GetArrayLength(env, inputArray);
    if (input_count == 0 || fromRate == toRate) return inputArray;
    if (fromRate <= 0 || toRate <= 0) return NULL;

    long started = nanos_math();
    torchaudio_hann_v1_plan plan;
    if (!build_torchaudio_hann_v1_plan(fromRate, toRate, &plan)) return NULL;
    long long output_count_long;
    if (!torchaudio_hann_v1_output_length(input_count, &plan, &output_count_long) ||
        output_count_long > INT_MAX) {
        free_torchaudio_hann_v1_plan(&plan);
        return NULL;
    }
    int output_count = (int)output_count_long;
    jfloat *input = (*env)->GetFloatArrayElements(env, inputArray, NULL);
    if (!input) {
        free_torchaudio_hann_v1_plan(&plan);
        return NULL;
    }
    float *output = malloc((size_t)output_count * sizeof(float));
    if (!output) {
        (*env)->ReleaseFloatArrayElements(env, inputArray, input, JNI_ABORT);
        free_torchaudio_hann_v1_plan(&plan);
        return NULL;
    }

    render_torchaudio_hann_v1_range(
        input, 0, input_count, output_count_long,
        0, output_count, &plan, output);
    int phases = plan.output_phases;
    int stride = plan.input_stride;
    int width = plan.width;
    int kernel_size = plan.kernel_size;
    free_torchaudio_hann_v1_plan(&plan);
    (*env)->ReleaseFloatArrayElements(env, inputArray, input, JNI_ABORT);

    jfloatArray result = (*env)->NewFloatArray(env, output_count);
    if (!result) {
        free(output);
        return NULL;
    }
    (*env)->SetFloatArrayRegion(env, result, 0, output_count, output);
    free(output);
    long finished = nanos_math();
    __android_log_print(
        ANDROID_LOG_INFO, TAG,
        "TIMING: torchaudio_hann_v1 %d->%dHz (phases=%d,stride=%d,width=%d,kernel=%d) "
        "%d->%d samples: total=%ldms",
        fromRate, toRate, phases, stride, width, kernel_size,
        input_count, output_count, (finished - started) / 1000000);
    return result;
}

JNIEXPORT jfloatArray JNICALL
Java_com_powerampstartradio_indexing_NativeMath_nativeResampleTorchAudioHannV1Aligned(
    JNIEnv *env, jclass cls,
    jfloatArray inputArray, jint fromRate, jint toRate,
    jlong inputStartSample, jlong totalInputSamples,
    jlong outputStartSample, jint outputSampleCount)
{
    jsize input_count = (*env)->GetArrayLength(env, inputArray);
    if (fromRate <= 0 || toRate <= 0 || fromRate == toRate ||
        inputStartSample < 0 || totalInputSamples < 0 ||
        inputStartSample > totalInputSamples ||
        (jlong)input_count > totalInputSamples - inputStartSample ||
        outputStartSample < 0 || outputSampleCount < 0) {
        LOGE("Invalid TorchAudio Hann V1 aligned range");
        return NULL;
    }

    torchaudio_hann_v1_plan plan;
    if (!build_torchaudio_hann_v1_plan(fromRate, toRate, &plan)) return NULL;
    long long total_output_samples;
    if (!torchaudio_hann_v1_output_length(
            totalInputSamples, &plan, &total_output_samples) ||
        outputStartSample > total_output_samples ||
        (jlong)outputSampleCount > total_output_samples - outputStartSample) {
        LOGE("TorchAudio Hann V1 output range exceeds whole-track output");
        free_torchaudio_hann_v1_plan(&plan);
        return NULL;
    }
    if (!torchaudio_hann_v1_has_context(
            inputStartSample, input_count, totalInputSamples,
            outputStartSample, outputSampleCount, &plan)) {
        LOGE("TorchAudio Hann V1 input slice lacks required FIR context");
        free_torchaudio_hann_v1_plan(&plan);
        return NULL;
    }

    jfloatArray result = (*env)->NewFloatArray(env, outputSampleCount);
    if (!result || outputSampleCount == 0) {
        free_torchaudio_hann_v1_plan(&plan);
        return result;
    }
    jfloat *input = (*env)->GetFloatArrayElements(env, inputArray, NULL);
    if (!input) {
        free_torchaudio_hann_v1_plan(&plan);
        return NULL;
    }
    float *output = malloc((size_t)outputSampleCount * sizeof(float));
    if (!output) {
        (*env)->ReleaseFloatArrayElements(env, inputArray, input, JNI_ABORT);
        free_torchaudio_hann_v1_plan(&plan);
        return NULL;
    }
    render_torchaudio_hann_v1_range(
        input, inputStartSample, totalInputSamples, total_output_samples,
        outputStartSample, outputSampleCount, &plan, output);
    (*env)->ReleaseFloatArrayElements(env, inputArray, input, JNI_ABORT);
    free_torchaudio_hann_v1_plan(&plan);
    (*env)->SetFloatArrayRegion(env, result, 0, outputSampleCount, output);
    free(output);
    return result;
}

/* ── int16 PCM → mono float conversion ──────────────────── */
/*
 * Convert interleaved int16 PCM from a direct ByteBuffer to mono float.
 * Replaces the per-sample Kotlin loop:
 *   for (frame in 0..frameCount) {
 *       for (ch in 0..channels) sample += buffer.getShort() / 32768f
 *       output[i] = sample / channels
 *   }
 * which does 21M individual getShort() calls for a 4-min stereo track.
 *
 * NEON processes 8 stereo frames at a time (16 int16 → 8 float), giving
 * ~20x speedup over the Kotlin ByteBuffer loop.
 *
 * @param byteBuffer  Direct ByteBuffer from MediaCodec.getOutputBuffer()
 * @param offsetBytes bufferInfo.offset (start of valid data)
 * @param sizeBytes   bufferInfo.size (bytes of valid data)
 * @param channels    Number of audio channels (1=mono, 2=stereo)
 * @param jOutput     Pre-allocated float array to write mono samples into
 * @param dstOffset   Write position in jOutput
 * @param maxFrames   Maximum frames to convert (for duration cap)
 * @return            Number of mono frames actually written
 */
JNIEXPORT jint JNICALL
Java_com_powerampstartradio_indexing_NativeMath_nativeInt16ToMonoFloat(
    JNIEnv *env, jclass cls,
    jobject byteBuffer, jint offsetBytes, jint sizeBytes, jint channels,
    jfloatArray jOutput, jint dstOffset, jint maxFrames)
{
    uint8_t *bufPtr = (uint8_t *)(*env)->GetDirectBufferAddress(env, byteBuffer);
    if (!bufPtr) {
        LOGE("int16ToMonoFloat: not a direct ByteBuffer");
        return 0;
    }

    const int16_t *src = (const int16_t *)(bufPtr + offsetBytes);
    int totalFrames = sizeBytes / (2 * channels);
    if (totalFrames > maxFrames) totalFrames = maxFrames;
    if (totalFrames <= 0) return 0;

    float *dst = (*env)->GetPrimitiveArrayCritical(env, jOutput, NULL);
    if (!dst) return 0;

    float *out = dst + dstOffset;

    if (channels == 2) {
        /* Stereo → mono: average L and R */
        const float scale = 1.0f / (32768.0f * 2.0f);
        int i = 0;
#if defined(__aarch64__)
        float32x4_t vscale = vdupq_n_f32(scale);
        for (; i + 7 < totalFrames; i += 8) {
            /* Load 8 stereo frames: 16 interleaved int16 → deinterleaved L[8], R[8] */
            int16x8x2_t stereo = vld2q_s16(src + i * 2);
            /* Sum L + R in int32 to avoid int16 overflow (L+R can exceed ±32767) */
            int32x4_t lo32 = vaddl_s16(vget_low_s16(stereo.val[0]),
                                        vget_low_s16(stereo.val[1]));
            float32x4_t flo = vmulq_f32(vcvtq_f32_s32(lo32), vscale);
            int32x4_t hi32 = vaddl_high_s16(stereo.val[0], stereo.val[1]);
            float32x4_t fhi = vmulq_f32(vcvtq_f32_s32(hi32), vscale);
            /* Store 8 mono floats */
            vst1q_f32(out + i, flo);
            vst1q_f32(out + i + 4, fhi);
        }
#endif
        for (; i < totalFrames; i++) {
            int sum = (int)src[i * 2] + (int)src[i * 2 + 1];
            out[i] = (float)sum * scale;
        }
    } else if (channels == 1) {
        /* Mono: just convert int16 → float */
        const float scale = 1.0f / 32768.0f;
        int i = 0;
#if defined(__aarch64__)
        float32x4_t vscale = vdupq_n_f32(scale);
        for (; i + 7 < totalFrames; i += 8) {
            int16x8_t samples = vld1q_s16(src + i);
            int32x4_t lo32 = vmovl_s16(vget_low_s16(samples));
            float32x4_t flo = vmulq_f32(vcvtq_f32_s32(lo32), vscale);
            int32x4_t hi32 = vmovl_s16(vget_high_s16(samples));
            float32x4_t fhi = vmulq_f32(vcvtq_f32_s32(hi32), vscale);
            vst1q_f32(out + i, flo);
            vst1q_f32(out + i + 4, fhi);
        }
#endif
        for (; i < totalFrames; i++) {
            out[i] = (float)src[i] * scale;
        }
    } else {
        /* Generic N-channel downmix */
        const float scale = 1.0f / (32768.0f * (float)channels);
        for (int i = 0; i < totalFrames; i++) {
            int sum = 0;
            for (int ch = 0; ch < channels; ch++) {
                sum += (int)src[i * channels + ch];
            }
            out[i] = (float)sum * scale;
        }
    }

    (*env)->ReleasePrimitiveArrayCritical(env, jOutput, dst, 0);
    return totalFrames;
}
