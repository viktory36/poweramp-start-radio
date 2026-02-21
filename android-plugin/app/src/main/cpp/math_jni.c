/*
 * NEON-accelerated math operations for embedding fusion.
 *
 * Hot loops that dominate FusionEngine runtime:
 * - k-means assignment: n × K dot products per iteration (~15s → ~2s)
 * - kNN candidate scoring: query vs N candidates
 * - Covariance accumulation: streaming outer products
 * - Matrix-vector multiply: projection (1024d → 512d per track)
 *
 * ARM NEON does 4 float multiply-adds per instruction, giving ~4x
 * speedup over scalar Kotlin loops. Combined with C loop efficiency
 * (no bounds checks, register allocation), expect 6-10x total.
 */
#include <jni.h>
#include <android/log.h>
#include <stdlib.h>
#include <string.h>
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
            /* Sum L + R channels */
            int16x8_t sum = vaddq_s16(stereo.val[0], stereo.val[1]);
            /* Convert lower 4 frames: int16 → int32 → float, scale */
            int32x4_t lo32 = vmovl_s16(vget_low_s16(sum));
            float32x4_t flo = vmulq_f32(vcvtq_f32_s32(lo32), vscale);
            /* Convert upper 4 frames */
            int32x4_t hi32 = vmovl_s16(vget_high_s16(sum));
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
