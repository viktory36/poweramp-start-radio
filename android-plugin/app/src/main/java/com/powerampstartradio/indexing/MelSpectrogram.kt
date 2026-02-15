package com.powerampstartradio.indexing

import kotlin.math.*

/**
 * Configurable mel spectrogram computation in pure Kotlin.
 *
 * Default configuration matches WhisperFeatureExtractor (for Flamingo):
 * - STFT: n_fft=400, hop_length=160, window=hann(400) @ 16kHz → 25ms window, 10ms hop
 * - Mel filterbank: 128 filters, fmin=0, fmax=8000
 *
 * Also supports MuLan configuration:
 * - n_fft=2048, hop_length=240, window=hann(2048) @ 24kHz
 * - Mel filterbank: 128 filters, fmin=0, fmax=12000 (Nyquist)
 *
 * Normalization is applied separately by the caller:
 * - Flamingo: [whisperNormalize] (log10, clamp, scale to [0,1])
 * - MuLan: (mel - mean) / std with model-specific statistics
 */
class MelSpectrogram(
    private val sampleRate: Int = 16000,
    private val nFft: Int = 400,
    private val hopLength: Int = 160,
    private val nMels: Int = 128,
    private val fMin: Float = 0f,
    private val fMax: Float = 8000f,
) {

    // Precomputed Hann window
    private val window = FloatArray(nFft) { i ->
        (0.5 * (1 - cos(2.0 * PI * i / nFft))).toFloat()
    }

    // Precomputed mel filterbank [nMels, nFft/2+1]
    private val melFilterbank: Array<FloatArray> = buildMelFilterbank()

    // Precomputed FFT twiddle factors for radix-2
    private val fftSize = nextPowerOf2(nFft)

    /**
     * Compute raw power mel spectrogram from audio samples.
     *
     * Returns the raw power mel with no normalization applied.
     * Callers should apply the appropriate normalization:
     * - Flamingo/Whisper: [whisperNormalize]
     * - MuLan: (mel - mean) / std
     *
     * @param audio PCM samples at [sampleRate] Hz
     * @return [nMels, numFrames] raw power mel spectrogram
     */
    fun compute(audio: FloatArray): Array<FloatArray> {
        // Pad to ensure we get the expected number of frames
        val numFrames = (audio.size / hopLength) + 1
        val paddedLength = (numFrames - 1) * hopLength + nFft
        val padded = if (audio.size < paddedLength) {
            FloatArray(paddedLength).also { audio.copyInto(it) }
        } else {
            audio
        }

        // STFT -> power spectrogram
        val nFreqs = nFft / 2 + 1
        val powerSpec = Array(numFrames) { FloatArray(nFreqs) }

        val fftReal = FloatArray(fftSize)
        val fftImag = FloatArray(fftSize)

        for (frame in 0 until numFrames) {
            val start = frame * hopLength

            // Apply window and zero-pad for FFT
            fftReal.fill(0f)
            fftImag.fill(0f)
            for (i in 0 until nFft) {
                val sampleIdx = start + i
                fftReal[i] = if (sampleIdx < padded.size) padded[sampleIdx] * window[i] else 0f
            }

            // In-place FFT
            fft(fftReal, fftImag)

            // Power spectrum: |FFT|^2
            for (k in 0 until nFreqs) {
                powerSpec[frame][k] = fftReal[k] * fftReal[k] + fftImag[k] * fftImag[k]
            }
        }

        // Apply mel filterbank
        val melSpec = Array(nMels) { FloatArray(numFrames) }
        for (m in 0 until nMels) {
            val filter = melFilterbank[m]
            for (t in 0 until numFrames) {
                var sum = 0f
                for (k in 0 until nFreqs) {
                    sum += filter[k] * powerSpec[t][k]
                }
                melSpec[m][t] = sum
            }
        }

        return melSpec
    }

    companion object {
        /**
         * Whisper-style log mel normalization (in-place).
         * 1. log10(max(mel, 1e-10))
         * 2. Clamp to max_val - 8.0
         * 3. Scale to approximately [0, 1] range: (x - (max_val - 8)) / 8
         */
        fun whisperNormalize(melSpec: Array<FloatArray>) {
            var maxVal = -Float.MAX_VALUE
            for (m in melSpec.indices) {
                for (t in melSpec[m].indices) {
                    melSpec[m][t] = log10(maxOf(melSpec[m][t], 1e-10f))
                    if (melSpec[m][t] > maxVal) maxVal = melSpec[m][t]
                }
            }

            val minVal = maxVal - 8f
            for (m in melSpec.indices) {
                for (t in melSpec[m].indices) {
                    melSpec[m][t] = (maxOf(melSpec[m][t], minVal) - minVal) / 8f
                }
            }
        }
    }

    /**
     * Build mel filterbank matrix [nMels, nFft/2+1].
     *
     * Uses HTK-style mel scale: mel(f) = 2595 * log10(1 + f/700)
     */
    private fun buildMelFilterbank(): Array<FloatArray> {
        val nFreqs = nFft / 2 + 1
        val filters = Array(nMels) { FloatArray(nFreqs) }

        val melMin = hzToMel(fMin)
        val melMax = hzToMel(fMax)

        // nMels + 2 equally spaced mel points
        val melPoints = FloatArray(nMels + 2) { i ->
            melMin + (melMax - melMin) * i / (nMels + 1)
        }

        // Convert mel points to FFT bin indices
        val binIndices = FloatArray(nMels + 2) { i ->
            val freq = melToHz(melPoints[i])
            freq * nFft / sampleRate
        }

        for (m in 0 until nMels) {
            val left = binIndices[m]
            val center = binIndices[m + 1]
            val right = binIndices[m + 2]

            for (k in 0 until nFreqs) {
                val kf = k.toFloat()
                filters[m][k] = when {
                    kf < left -> 0f
                    kf <= center -> (kf - left) / (center - left)
                    kf <= right -> (right - kf) / (right - center)
                    else -> 0f
                }
            }

            // Slaney-style normalization: normalize each filter by its bandwidth
            val bandwidth = 2f / (binIndices[m + 2] - binIndices[m])
            for (k in 0 until nFreqs) {
                filters[m][k] *= bandwidth
            }
        }

        return filters
    }

    private fun hzToMel(hz: Float): Float = 2595f * log10(1f + hz / 700f)
    private fun melToHz(mel: Float): Float = 700f * (10f.pow(mel / 2595f) - 1f)

    /**
     * In-place radix-2 Cooley-Tukey FFT.
     *
     * Input arrays must be power-of-2 length. Data beyond nFft is zero-padded.
     */
    private fun fft(real: FloatArray, imag: FloatArray) {
        val n = real.size
        if (n <= 1) return

        // Bit-reversal permutation
        var j = 0
        for (i in 0 until n - 1) {
            if (i < j) {
                var temp = real[i]; real[i] = real[j]; real[j] = temp
                temp = imag[i]; imag[i] = imag[j]; imag[j] = temp
            }
            var k = n / 2
            while (k <= j) {
                j -= k
                k /= 2
            }
            j += k
        }

        // FFT butterfly
        var len = 2
        while (len <= n) {
            val halfLen = len / 2
            val angle = -2.0 * PI / len
            val wReal = cos(angle).toFloat()
            val wImag = sin(angle).toFloat()

            var i = 0
            while (i < n) {
                var curReal = 1f
                var curImag = 0f

                for (k in 0 until halfLen) {
                    val tReal = curReal * real[i + k + halfLen] - curImag * imag[i + k + halfLen]
                    val tImag = curReal * imag[i + k + halfLen] + curImag * real[i + k + halfLen]

                    real[i + k + halfLen] = real[i + k] - tReal
                    imag[i + k + halfLen] = imag[i + k] - tImag
                    real[i + k] += tReal
                    imag[i + k] += tImag

                    val newReal = curReal * wReal - curImag * wImag
                    val newImag = curReal * wImag + curImag * wReal
                    curReal = newReal
                    curImag = newImag
                }
                i += len
            }
            len *= 2
        }
    }

    private fun nextPowerOf2(n: Int): Int {
        var v = n - 1
        v = v or (v shr 1)
        v = v or (v shr 2)
        v = v or (v shr 4)
        v = v or (v shr 8)
        v = v or (v shr 16)
        return v + 1
    }
}
