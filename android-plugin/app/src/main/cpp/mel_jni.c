/*
 * Native mel spectrogram computation for MuLan and Flamingo.
 *
 * Matches the pure Kotlin MelSpectrogram.kt exactly:
 * - Periodic Hann window
 * - Radix-2 Cooley-Tukey FFT for power-of-2 nFft
 * - Bluestein's algorithm for non-power-of-2 nFft (e.g., Whisper's 400)
 * - Slaney + HTK mel filterbanks with optional area normalization
 * - Reflect padding for center=true
 */
#include <jni.h>
#include <android/log.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define TAG "MelJNI"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

#define MEL_HTK    0
#define MEL_SLANEY 1

/* ── Mel scale conversions ─────────────────────────────────── */

static float hz_to_mel(float hz, int scale) {
    if (scale == MEL_SLANEY) {
        if (hz < 1000.0f) return 3.0f * hz / 200.0f;
        return 15.0f + logf(hz / 1000.0f) * (27.0f / logf(6.4f));
    }
    return 2595.0f * log10f(1.0f + hz / 700.0f);
}

static float mel_to_hz(float mel, int scale) {
    if (scale == MEL_SLANEY) {
        if (mel < 15.0f) return 200.0f * mel / 3.0f;
        return 1000.0f * expf((mel - 15.0f) / (27.0f / logf(6.4f)));
    }
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

/* ── Utilities ─────────────────────────────────────────────── */

static int next_pow2(int n) {
    int v = n - 1;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return v + 1;
}

/* ── In-place radix-2 Cooley-Tukey FFT ────────────────────── */

static void fft_radix2(float *real, float *imag, int n) {
    /* Bit-reversal permutation */
    int j = 0;
    for (int i = 0; i < n - 1; i++) {
        if (i < j) {
            float t;
            t = real[i]; real[i] = real[j]; real[j] = t;
            t = imag[i]; imag[i] = imag[j]; imag[j] = t;
        }
        int k = n / 2;
        while (k <= j) { j -= k; k /= 2; }
        j += k;
    }

    /* Butterfly passes */
    for (int len = 2; len <= n; len *= 2) {
        int half = len / 2;
        float angle = -2.0f * (float)M_PI / (float)len;
        float w_r = cosf(angle), w_i = sinf(angle);

        for (int i = 0; i < n; i += len) {
            float cur_r = 1.0f, cur_i = 0.0f;
            for (int k = 0; k < half; k++) {
                int a = i + k, b = i + k + half;
                float tr = cur_r * real[b] - cur_i * imag[b];
                float ti = cur_r * imag[b] + cur_i * real[b];
                real[b] = real[a] - tr;
                imag[b] = imag[a] - ti;
                real[a] += tr;
                imag[a] += ti;
                float nr = cur_r * w_r - cur_i * w_i;
                cur_i    = cur_r * w_i + cur_i * w_r;
                cur_r    = nr;
            }
        }
    }
}

/* ── Bluestein power spectrum (exact N-point DFT) ─────────── */

static void bluestein_power_spectrum(
    const float *x, int N, int M,
    const float *chirp_r, const float *chirp_i,
    const float *h_fft_r, const float *h_fft_i,
    float *work_r, float *work_i,
    float *output, int n_freqs)
{
    /* Step 1: a[n] = x[n] * chirp[n], zero-padded to M */
    memset(work_r, 0, (size_t)M * sizeof(float));
    memset(work_i, 0, (size_t)M * sizeof(float));
    for (int n = 0; n < N; n++) {
        work_r[n] = x[n] * chirp_r[n];
        work_i[n] = x[n] * chirp_i[n];
    }

    /* Step 2: A = FFT_M(a) */
    fft_radix2(work_r, work_i, M);

    /* Step 3: C = A * H (element-wise complex multiply) */
    for (int i = 0; i < M; i++) {
        float tr = work_r[i] * h_fft_r[i] - work_i[i] * h_fft_i[i];
        float ti = work_r[i] * h_fft_i[i] + work_i[i] * h_fft_r[i];
        work_r[i] = tr;
        work_i[i] = ti;
    }

    /* Step 4: IFFT via conjugate-FFT-conjugate-scale */
    for (int i = 0; i < M; i++) work_i[i] = -work_i[i];
    fft_radix2(work_r, work_i, M);
    float scale = 1.0f / (float)M;
    for (int i = 0; i < M; i++) {
        work_r[i] *= scale;
        work_i[i] = -work_i[i] * scale;
    }

    /* Step 5: X[k] = chirp[k] * c[k], output |X[k]|^2 */
    for (int k = 0; k < n_freqs; k++) {
        float xr = work_r[k] * chirp_r[k] - work_i[k] * chirp_i[k];
        float xi = work_r[k] * chirp_i[k] + work_i[k] * chirp_r[k];
        output[k] = xr * xr + xi * xi;
    }
}

/* ── JNI: compute mel spectrogram ─────────────────────────── */

/*
 * Returns flat float array:
 *   [numFrames_as_float, mel[0][0], mel[0][1], ..., mel[nMels-1][numFrames-1]]
 * Layout is mel-major: mel[m * numFrames + t].
 */
JNIEXPORT jfloatArray JNICALL
Java_com_powerampstartradio_indexing_NativeMel_nativeComputeMel(
    JNIEnv *env,
    jclass clazz,
    jfloatArray audioArray,
    jint sampleRate,
    jint nFft,
    jint hopLength,
    jint nMels,
    jfloat fMin,
    jfloat fMax,
    jboolean center,
    jint melScale,
    jboolean normalize)
{
    jsize audioLen = (*env)->GetArrayLength(env, audioArray);
    jfloat *audio = (*env)->GetFloatArrayElements(env, audioArray, NULL);
    if (!audio) { LOGE("Failed to get audio array"); return NULL; }

    int n_freqs = nFft / 2 + 1;
    int pad = nFft / 2;

    /* ── Center padding (reflect) ── */
    float *input;
    int inputLen;
    if (center) {
        inputLen = audioLen + 2 * pad;
        input = (float *)malloc((size_t)inputLen * sizeof(float));
        if (!input) {
            (*env)->ReleaseFloatArrayElements(env, audioArray, audio, JNI_ABORT);
            LOGE("Center pad alloc failed");
            return NULL;
        }
        for (int i = 0; i < pad; i++) input[i] = audio[pad - i];
        memcpy(input + pad, audio, (size_t)audioLen * sizeof(float));
        for (int i = 0; i < pad; i++) input[pad + audioLen + i] = audio[audioLen - 2 - i];
    } else {
        inputLen = audioLen;
        input = (float *)audio;  /* no copy needed */
    }

    /* ── Frame count ── */
    int numFrames;
    if (center) {
        numFrames = 1 + (inputLen - nFft) / hopLength;
    } else {
        numFrames = audioLen / hopLength;
    }

    if (numFrames <= 0) {
        if (center) free(input);
        (*env)->ReleaseFloatArrayElements(env, audioArray, audio, JNI_ABORT);
        /* Return array with just numFrames=0 */
        jfloatArray result = (*env)->NewFloatArray(env, 1);
        if (result) {
            float zero = 0.0f;
            (*env)->SetFloatArrayRegion(env, result, 0, 1, &zero);
        }
        return result;
    }

    /* ── Hann window (periodic) ── */
    float *window = (float *)malloc((size_t)nFft * sizeof(float));
    if (!window) goto fail;
    for (int i = 0; i < nFft; i++) {
        window[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * (float)i / (float)nFft));
    }

    /* ── Mel filterbank [nMels][n_freqs] ── */
    float *filters = (float *)calloc((size_t)nMels * (size_t)n_freqs, sizeof(float));
    if (!filters) { free(window); goto fail; }
    {
        float mel_min = hz_to_mel(fMin, melScale);
        float mel_max = hz_to_mel(fMax, melScale);
        int nPoints = nMels + 2;
        float *freq_hz = (float *)malloc((size_t)nPoints * sizeof(float));
        if (!freq_hz) { free(filters); free(window); goto fail; }

        for (int i = 0; i < nPoints; i++) {
            float mel_pt = mel_min + (mel_max - mel_min) * (float)i / (float)(nMels + 1);
            freq_hz[i] = mel_to_hz(mel_pt, melScale);
        }

        for (int m = 0; m < nMels; m++) {
            float left  = freq_hz[m];
            float cent  = freq_hz[m + 1];
            float right = freq_hz[m + 2];
            float enorm = normalize ? 2.0f / (right - left) : 1.0f;

            for (int k = 0; k < n_freqs; k++) {
                float f = (float)k * (float)sampleRate / (float)nFft;
                float v;
                if      (f < left)  v = 0.0f;
                else if (f <= cent) v = (f - left) / (cent - left);
                else if (f <= right) v = (right - f) / (right - cent);
                else                 v = 0.0f;
                filters[m * n_freqs + k] = v * enorm;
            }
        }
        free(freq_hz);
    }

    /* ── FFT setup ── */
    int fft_size = next_pow2(nFft);
    int use_bluestein = (fft_size != nFft);

    float *chirp_r = NULL, *chirp_i = NULL;
    float *h_fft_r = NULL, *h_fft_i = NULL;
    int M = 0;
    int work_size;

    if (use_bluestein) {
        M = next_pow2(2 * nFft - 1);
        work_size = M;

        chirp_r = (float *)malloc((size_t)nFft * sizeof(float));
        chirp_i = (float *)malloc((size_t)nFft * sizeof(float));
        h_fft_r = (float *)calloc((size_t)M, sizeof(float));
        h_fft_i = (float *)calloc((size_t)M, sizeof(float));
        if (!chirp_r || !chirp_i || !h_fft_r || !h_fft_i) {
            free(chirp_r); free(chirp_i); free(h_fft_r); free(h_fft_i);
            free(filters); free(window); goto fail;
        }

        /* Chirp: exp(-pi*i * n^2 / N) */
        for (int n = 0; n < nFft; n++) {
            double angle = -(double)M_PI * (long)n * (long)n / (double)nFft;
            chirp_r[n] = (float)cos(angle);
            chirp_i[n] = (float)sin(angle);
        }

        /* h[n] = conj(chirp[n]), arranged for circular convolution */
        for (int n = 0; n < nFft; n++) {
            double angle = (double)M_PI * (long)n * (long)n / (double)nFft;
            float cr = (float)cos(angle), ci = (float)sin(angle);
            h_fft_r[n] = cr; h_fft_i[n] = ci;
            if (n > 0) { h_fft_r[M - n] = cr; h_fft_i[M - n] = ci; }
        }
        fft_radix2(h_fft_r, h_fft_i, M);
    } else {
        work_size = fft_size;
    }

    /* ── Working arrays ── */
    float *work_r = (float *)malloc((size_t)work_size * sizeof(float));
    float *work_i = (float *)malloc((size_t)work_size * sizeof(float));
    float *frame_buf = (float *)malloc((size_t)nFft * sizeof(float));
    float *power_frame = (float *)malloc((size_t)n_freqs * sizeof(float));
    float *mel_spec = (float *)calloc((size_t)nMels * (size_t)numFrames, sizeof(float));

    if (!work_r || !work_i || !frame_buf || !power_frame || !mel_spec) {
        free(work_r); free(work_i); free(frame_buf); free(power_frame); free(mel_spec);
        if (use_bluestein) { free(chirp_r); free(chirp_i); free(h_fft_r); free(h_fft_i); }
        free(filters); free(window); goto fail;
    }

    /* ── STFT → power → mel for each frame ── */
    for (int frame = 0; frame < numFrames; frame++) {
        int start = frame * hopLength;

        /* Apply window */
        for (int i = 0; i < nFft; i++) {
            int idx = start + i;
            frame_buf[i] = (idx < inputLen) ? input[idx] * window[i] : 0.0f;
        }

        if (use_bluestein) {
            bluestein_power_spectrum(frame_buf, nFft, M,
                chirp_r, chirp_i, h_fft_r, h_fft_i,
                work_r, work_i, power_frame, n_freqs);
        } else {
            memset(work_r, 0, (size_t)fft_size * sizeof(float));
            memset(work_i, 0, (size_t)fft_size * sizeof(float));
            memcpy(work_r, frame_buf, (size_t)nFft * sizeof(float));
            fft_radix2(work_r, work_i, fft_size);
            for (int k = 0; k < n_freqs; k++) {
                power_frame[k] = work_r[k] * work_r[k] + work_i[k] * work_i[k];
            }
        }

        /* Accumulate into mel filterbank */
        for (int m = 0; m < nMels; m++) {
            float sum = 0.0f;
            const float *filt = filters + m * n_freqs;
            for (int k = 0; k < n_freqs; k++) {
                sum += filt[k] * power_frame[k];
            }
            mel_spec[m * numFrames + frame] = sum;
        }
    }

    /* ── Build Java result array ── */
    int totalSize = nMels * numFrames;
    jfloatArray result = (*env)->NewFloatArray(env, 1 + totalSize);
    if (result) {
        float nf = (float)numFrames;
        (*env)->SetFloatArrayRegion(env, result, 0, 1, &nf);
        (*env)->SetFloatArrayRegion(env, result, 1, totalSize, mel_spec);
    }

    /* ── Cleanup ── */
    free(mel_spec);
    free(power_frame);
    free(frame_buf);
    free(work_r);
    free(work_i);
    if (use_bluestein) {
        free(chirp_r); free(chirp_i);
        free(h_fft_r); free(h_fft_i);
    }
    free(filters);
    free(window);
    if (center) free(input);
    (*env)->ReleaseFloatArrayElements(env, audioArray, audio, JNI_ABORT);
    return result;

fail:
    if (center && input != (float *)audio) free(input);
    (*env)->ReleaseFloatArrayElements(env, audioArray, audio, JNI_ABORT);
    LOGE("Allocation failed in mel computation");
    return NULL;
}
