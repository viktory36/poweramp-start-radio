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

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

#define TAG "MathJNI"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

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

static void topk_sift_down(TopKEntry *heap, int size, int i) {
    while (1) {
        int smallest = i;
        int left = 2 * i + 1;
        int right = 2 * i + 2;
        if (left < size && heap[left].score < heap[smallest].score) smallest = left;
        if (right < size && heap[right].score < heap[smallest].score) smallest = right;
        if (smallest == i) break;
        TopKEntry tmp = heap[i];
        heap[i] = heap[smallest];
        heap[smallest] = tmp;
        i = smallest;
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
    jfloatArray outScores)
{
    uint8_t *base = (uint8_t *)(*env)->GetDirectBufferAddress(env, byteBuffer);
    if (!base) {
        LOGE("nativeFindTopK: not a direct ByteBuffer");
        return 0;
    }

    const int64_t *trackIds = (const int64_t *)(base + trackIdsOffset);
    const float *embeddings = (const float *)(base + embOffset);

    float *query = (*env)->GetFloatArrayElements(env, jQuery, NULL);
    if (!query) return 0;

    /* Exclude set (usually 1 element = seed track) */
    int excludeCount = jExcludeIds ? (*env)->GetArrayLength(env, jExcludeIds) : 0;
    int64_t *excludeIds = NULL;
    if (excludeCount > 0) {
        excludeIds = (*env)->GetLongArrayElements(env, jExcludeIds, NULL);
    }

    /* Allocate min-heap */
    TopKEntry *heap = (TopKEntry *)malloc(topK * sizeof(TopKEntry));
    if (!heap) {
        (*env)->ReleaseFloatArrayElements(env, jQuery, query, JNI_ABORT);
        if (excludeIds) (*env)->ReleaseLongArrayElements(env, jExcludeIds, excludeIds, JNI_ABORT);
        return 0;
    }
    int heapSize = 0;

    for (int i = 0; i < numTracks; i++) {
        /* Check exclude list (linear scan — typically 1 element) */
        if (excludeCount > 0) {
            int64_t tid = trackIds[i];
            int skip = 0;
            for (int e = 0; e < excludeCount; e++) {
                if (excludeIds[e] == tid) { skip = 1; break; }
            }
            if (skip) continue;
        }

        float score = dot_product(query, embeddings + (long)i * dim, dim);

        if (heapSize < topK) {
            heap[heapSize].idx = i;
            heap[heapSize].score = score;
            heapSize++;
            /* Heapify once full */
            if (heapSize == topK) {
                for (int j = topK / 2 - 1; j >= 0; j--)
                    topk_sift_down(heap, heapSize, j);
            }
        } else if (score > heap[0].score) {
            heap[0].idx = i;
            heap[0].score = score;
            topk_sift_down(heap, heapSize, 0);
        }
    }

    /* Sort by score descending (insertion sort, heapSize ≤ topK ≤ ~1500) */
    for (int i = 1; i < heapSize; i++) {
        TopKEntry key = heap[i];
        int j = i - 1;
        while (j >= 0 && heap[j].score < key.score) {
            heap[j + 1] = heap[j];
            j--;
        }
        heap[j + 1] = key;
    }

    /* Write results to output arrays */
    int64_t *outIds = (*env)->GetLongArrayElements(env, outTrackIds, NULL);
    float *outScr = (*env)->GetFloatArrayElements(env, outScores, NULL);
    for (int i = 0; i < heapSize; i++) {
        outIds[i] = trackIds[heap[i].idx];
        outScr[i] = heap[i].score;
    }
    (*env)->ReleaseLongArrayElements(env, outTrackIds, outIds, 0);
    (*env)->ReleaseFloatArrayElements(env, outScores, outScr, 0);

    /* Cleanup */
    free(heap);
    (*env)->ReleaseFloatArrayElements(env, jQuery, query, JNI_ABORT);
    if (excludeIds) (*env)->ReleaseLongArrayElements(env, jExcludeIds, excludeIds, JNI_ABORT);

    return heapSize;
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

JNIEXPORT jfloatArray JNICALL
Java_com_powerampstartradio_indexing_NativeMath_nativeResamplePolyphase(
    JNIEnv *env, jclass cls,
    jfloatArray inputArray, jint fromRate, jint toRate)
{
    jsize n_in = (*env)->GetArrayLength(env, inputArray);
    if (n_in == 0 || fromRate == toRate) return inputArray;

    long t0 = nanos_math();

    jfloat *input = (*env)->GetFloatArrayElements(env, inputArray, NULL);
    if (!input) return NULL;

    /* Compute rational resampling factors */
    int g = gcd_int((int)fromRate, (int)toRate);
    int up = (int)toRate / g;
    int down = (int)fromRate / g;
    int max_rate = (up > down) ? up : down;

    /* Design anti-aliasing FIR filter (Kaiser window, beta=5.0) */
    /* Matches scipy.signal.resample_poly defaults */
    double cutoff = 1.0 / max_rate;
    int half_len = 10 * max_rate;
    int filt_len = 2 * half_len + 1;

    float *filt = (float *)malloc(filt_len * sizeof(float));
    if (!filt) {
        (*env)->ReleaseFloatArrayElements(env, inputArray, input, JNI_ABORT);
        return NULL;
    }

    double beta = 5.0;
    double inv_i0 = 1.0 / bessel_i0(beta);
    double half = (double)half_len;

    for (int n = 0; n < filt_len; n++) {
        double t = n - half;
        /* Normalized sinc */
        double sinc = (fabs(t) < 1e-10) ? cutoff
                    : sin(M_PI * cutoff * t) / (M_PI * t);
        /* Kaiser window */
        double r = t / half;
        double w_arg = 1.0 - r * r;
        double window = (w_arg > 0) ? bessel_i0(beta * sqrt(w_arg)) * inv_i0 : 0.0;
        filt[n] = (float)(sinc * window * up);
    }

    /* Polyphase decomposition: split filter into 'up' phases */
    int taps = (filt_len + up - 1) / up;
    float *phases = (float *)calloc((size_t)up * taps, sizeof(float));
    if (!phases) {
        free(filt);
        (*env)->ReleaseFloatArrayElements(env, inputArray, input, JNI_ABORT);
        return NULL;
    }
    for (int p = 0; p < up; p++) {
        for (int t = 0; t < taps; t++) {
            int fi = p + t * up;
            if (fi < filt_len) phases[p * taps + t] = filt[fi];
        }
    }
    free(filt);

    /* Reverse each polyphase filter phase (convolution = correlation with flipped kernel).
     * The full filter is symmetric, but individual phases are NOT symmetric.
     * Without this flip, we compute correlation instead of convolution, producing
     * phase-shifted output that degrades embeddings (cosine 0.980 vs 0.997). */
    for (int p = 0; p < up; p++) {
        float *ph = &phases[p * taps];
        for (int t = 0; t < taps / 2; t++) {
            int rt = taps - 1 - t;
            float tmp = ph[t];
            ph[t] = ph[rt];
            ph[rt] = tmp;
        }
    }

    /* Compute output length */
    long long n_out_ll = ((long long)n_in * up + down - 1) / down;
    int n_out = (int)n_out_ll;

    float *output = (float *)malloc(n_out * sizeof(float));
    if (!output) {
        free(phases);
        (*env)->ReleaseFloatArrayElements(env, inputArray, input, JNI_ABORT);
        return NULL;
    }

    /* Center offset: positions the filter symmetrically around each output sample */
    int center_tap = (taps - 1) / 2;

    long t1 = nanos_math();

    /* ── Apply polyphase filter ────────────────────────────── */
    /* Three regions: leading edge (bounds checks), middle (pure NEON),
     * trailing edge (bounds checks). Middle handles >99.9% of samples. */

    /* Find safe middle region (no bounds checks needed) */
    int first_safe = 0;
    while (first_safe < n_out) {
        long long pos = (long long)first_safe * down;
        int input_idx = (int)(pos / up) - center_tap;
        if (input_idx >= 0) break;
        first_safe++;
    }

    int last_safe = n_out;
    while (last_safe > first_safe) {
        long long pos = (long long)(last_safe - 1) * down;
        int input_idx = (int)(pos / up) - center_tap + taps - 1;
        if (input_idx < n_in) break;
        last_safe--;
    }

#if defined(__aarch64__)
    int neon_taps = taps & ~3;  /* largest multiple of 4 ≤ taps */
#endif

    /* Leading edge (scalar with bounds checks) */
    for (int n = 0; n < first_safe; n++) {
        long long pos = (long long)n * down;
        int phase = (int)(pos % up);
        int k_start = (int)(pos / up) - center_tap;
        float *ph = &phases[phase * taps];
        float sum = 0.0f;
        for (int t = 0; t < taps; t++) {
            int ki = k_start + t;
            if (ki >= 0 && ki < n_in)
                sum += ph[t] * input[ki];
        }
        output[n] = sum;
    }

    /* Middle region: pure NEON, no bounds checks */
    for (int n = first_safe; n < last_safe; n++) {
        long long pos = (long long)n * down;
        int phase = (int)(pos % up);
        int k_start = (int)(pos / up) - center_tap;
        float *ph = &phases[phase * taps];
        float *src = &input[k_start];

#if defined(__aarch64__)
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);
        int t = 0;
        /* Unrolled 8-wide for better ILP */
        for (; t + 7 < taps; t += 8) {
            acc0 = vfmaq_f32(acc0, vld1q_f32(&ph[t]),     vld1q_f32(&src[t]));
            acc1 = vfmaq_f32(acc1, vld1q_f32(&ph[t + 4]), vld1q_f32(&src[t + 4]));
        }
        for (; t + 3 < taps; t += 4) {
            acc0 = vfmaq_f32(acc0, vld1q_f32(&ph[t]), vld1q_f32(&src[t]));
        }
        float sum = vaddvq_f32(vaddq_f32(acc0, acc1));
        for (; t < taps; t++) {
            sum += ph[t] * src[t];
        }
#else
        float sum = 0.0f;
        for (int t = 0; t < taps; t++) {
            sum += ph[t] * src[t];
        }
#endif
        output[n] = sum;
    }

    /* Trailing edge (scalar with bounds checks) */
    for (int n = last_safe; n < n_out; n++) {
        long long pos = (long long)n * down;
        int phase = (int)(pos % up);
        int k_start = (int)(pos / up) - center_tap;
        float *ph = &phases[phase * taps];
        float sum = 0.0f;
        for (int t = 0; t < taps; t++) {
            int ki = k_start + t;
            if (ki >= 0 && ki < n_in)
                sum += ph[t] * input[ki];
        }
        output[n] = sum;
    }

    long t2 = nanos_math();

    free(phases);
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
