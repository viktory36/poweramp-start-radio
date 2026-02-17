package com.powerampstartradio.indexing

import kotlin.math.*

/**
 * Mel frequency scale for filterbank construction.
 */
enum class MelScale {
    /** HTK formula: 2595 * log10(1 + f/700). Used by librosa/MuLan. */
    HTK,
    /** Slaney 1998: linear below 1kHz, logarithmic above. Used by Whisper. */
    SLANEY,
}

/**
 * Configurable mel spectrogram computation in pure Kotlin.
 *
 * Default configuration matches WhisperFeatureExtractor for Whisper-large-v3 (Flamingo):
 * - STFT: n_fft=400, hop_length=160, window=hann(400) @ 16kHz -> 25ms window, 10ms hop
 * - Mel filterbank: 128 Slaney-scale filters, fmin=0, fmax=8000, area-normalized
 * - Center padding: reflect-pad nFft/2 on each side
 *
 * Also supports MuLan configuration:
 * - n_fft=2048, hop_length=240, window=hann(2048) @ 24kHz
 * - Mel filterbank: 128 HTK-scale filters, fmin=0, fmax=12000 (Nyquist)
 *
 * Normalization is applied separately by the caller:
 * - Flamingo: [whisperNormalize] (log10, clamp, (x + 4) / 4)
 * - MuLan: (mel - mean) / std with model-specific statistics
 */
class MelSpectrogram(
    private val sampleRate: Int = 16000,
    private val nFft: Int = 400,
    private val hopLength: Int = 160,
    private val nMels: Int = 128,
    private val fMin: Float = 0f,
    private val fMax: Float = 8000f,
    private val center: Boolean = false,
    private val melScale: MelScale = MelScale.HTK,
) {

    // Precomputed periodic Hann window (matches np.hanning with periodic=True)
    private val window = FloatArray(nFft) { i ->
        (0.5 * (1 - cos(2.0 * PI * i / nFft))).toFloat()
    }

    // Precomputed mel filterbank [nMels, nFft/2+1]
    private val melFilterbank: Array<FloatArray> = buildMelFilterbank()

    // FFT size (next power of 2 >= nFft)
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
        // Optional center padding: reflect-pad nFft/2 on each side
        val input = if (center) reflectPad(audio, nFft / 2) else audio

        // Frame count
        val numFrames = if (center) {
            // HuggingFace convention: 1 + floor((padded_size - frame_length) / hop_length)
            1 + (input.size - nFft) / hopLength
        } else {
            audio.size / hopLength
        }

        val paddedLength = (numFrames - 1) * hopLength + nFft
        val padded = if (input.size < paddedLength) {
            FloatArray(paddedLength).also { input.copyInto(it) }
        } else {
            input
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
        // Slaney mel scale constant: 27 / ln(6.4)
        private val SLANEY_LOGSTEP = (27.0 / ln(6.4)).toFloat()

        /**
         * Whisper-style log mel normalization (in-place).
         * Matches HuggingFace WhisperFeatureExtractor exactly:
         * 1. log10(max(mel, 1e-10))
         * 2. Clamp to max_val - 8.0 (80 dB dynamic range)
         * 3. Rescale: (x + 4.0) / 4.0
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
                    melSpec[m][t] = (maxOf(melSpec[m][t], minVal) + 4f) / 4f
                }
            }
        }
    }

    /**
     * Reflect-pad audio symmetrically.
     * Matches numpy.pad(audio, padLength, mode='reflect').
     */
    private fun reflectPad(audio: FloatArray, padLength: Int): FloatArray {
        val result = FloatArray(audio.size + 2 * padLength)
        // Left reflection: audio[padLength], audio[padLength-1], ..., audio[1]
        for (i in 0 until padLength) {
            result[i] = audio[padLength - i]
        }
        // Original audio
        audio.copyInto(result, padLength)
        // Right reflection: audio[n-2], audio[n-3], ..., audio[n-1-padLength]
        for (i in 0 until padLength) {
            result[padLength + audio.size + i] = audio[audio.size - 2 - i]
        }
        return result
    }

    /**
     * Build mel filterbank matrix [nMels, nFft/2+1].
     *
     * Triangular filters constructed in Hz space with center frequencies
     * uniformly spaced in mel space. Slaney-style area normalization
     * (each filter divided by its bandwidth in Hz).
     *
     * Matches HuggingFace's mel_filter_bank(norm="slaney").
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

        // Convert mel points to Hz (filter edge/center frequencies)
        val filterFreqsHz = FloatArray(nMels + 2) { i ->
            melToHz(melPoints[i])
        }

        // FFT bin center frequencies in Hz
        val fftFreqsHz = FloatArray(nFreqs) { k ->
            k.toFloat() * sampleRate / nFft
        }

        for (m in 0 until nMels) {
            val left = filterFreqsHz[m]
            val center = filterFreqsHz[m + 1]
            val right = filterFreqsHz[m + 2]

            for (k in 0 until nFreqs) {
                val freq = fftFreqsHz[k]
                filters[m][k] = when {
                    freq < left -> 0f
                    freq <= center -> (freq - left) / (center - left)
                    freq <= right -> (right - freq) / (right - center)
                    else -> 0f
                }
            }

            // Slaney-style area normalization: divide by bandwidth in Hz
            val enorm = 2f / (right - left)
            for (k in 0 until nFreqs) {
                filters[m][k] *= enorm
            }
        }

        return filters
    }

    private fun hzToMel(hz: Float): Float = when (melScale) {
        MelScale.HTK -> 2595f * log10(1f + hz / 700f)
        MelScale.SLANEY -> if (hz < 1000f) {
            3f * hz / 200f
        } else {
            15f + ln(hz / 1000f) * SLANEY_LOGSTEP
        }
    }

    private fun melToHz(mel: Float): Float = when (melScale) {
        MelScale.HTK -> 700f * (10f.pow(mel / 2595f) - 1f)
        MelScale.SLANEY -> if (mel < 15f) {
            200f * mel / 3f
        } else {
            1000f * exp((mel - 15f) / SLANEY_LOGSTEP)
        }
    }

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
