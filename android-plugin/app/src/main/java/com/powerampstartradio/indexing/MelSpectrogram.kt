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

// Slaney mel scale constant: 27 / ln(6.4)
private val SLANEY_LOGSTEP = (27.0 / ln(6.4)).toFloat()

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
 * For non-power-of-2 nFft (e.g. Whisper's 400), uses Bluestein's algorithm
 * to compute exact N-point DFT via radix-2 FFT. This avoids frequency-bin
 * misalignment from zero-padding to the next power of 2.
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

    // Bluestein state for non-power-of-2 FFT (null if nFft is power of 2)
    private val bluestein: BluesteinState? =
        if (fftSize != nFft) BluesteinState(nFft) else null

    /**
     * Compute raw power mel spectrogram from audio samples.
     *
     * @param audio PCM samples at [sampleRate] Hz
     * @return [nMels, numFrames] raw power mel spectrogram
     */
    fun compute(audio: FloatArray): Array<FloatArray> {
        // Optional center padding: reflect-pad nFft/2 on each side
        val input = if (center) reflectPad(audio, nFft / 2) else audio

        // Frame count
        val numFrames = if (center) {
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
        val frameBuffer = FloatArray(nFft)

        val bs = bluestein
        if (bs != null) {
            // Non-power-of-2: exact N-point DFT via Bluestein's algorithm
            for (frame in 0 until numFrames) {
                val start = frame * hopLength
                for (i in 0 until nFft) {
                    val sampleIdx = start + i
                    frameBuffer[i] = if (sampleIdx < padded.size) padded[sampleIdx] * window[i] else 0f
                }
                bs.powerSpectrum(frameBuffer, powerSpec[frame])
            }
        } else {
            // Power-of-2: direct radix-2 FFT
            val fftReal = FloatArray(fftSize)
            val fftImag = FloatArray(fftSize)
            for (frame in 0 until numFrames) {
                val start = frame * hopLength
                fftReal.fill(0f)
                fftImag.fill(0f)
                for (i in 0 until nFft) {
                    val sampleIdx = start + i
                    fftReal[i] = if (sampleIdx < padded.size) padded[sampleIdx] * window[i] else 0f
                }
                fftRadix2(fftReal, fftImag)
                for (k in 0 until nFreqs) {
                    powerSpec[frame][k] = fftReal[k] * fftReal[k] + fftImag[k] * fftImag[k]
                }
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
        for (i in 0 until padLength) {
            result[i] = audio[padLength - i]
        }
        audio.copyInto(result, padLength)
        for (i in 0 until padLength) {
            result[padLength + audio.size + i] = audio[audio.size - 2 - i]
        }
        return result
    }

    /**
     * Build mel filterbank matrix [nMels, nFft/2+1].
     *
     * Triangular filters constructed in Hz space with center frequencies
     * uniformly spaced in mel space. Slaney-style area normalization.
     */
    private fun buildMelFilterbank(): Array<FloatArray> {
        val nFreqs = nFft / 2 + 1
        val filters = Array(nMels) { FloatArray(nFreqs) }

        val melMin = hzToMel(fMin)
        val melMax = hzToMel(fMax)

        val melPoints = FloatArray(nMels + 2) { i ->
            melMin + (melMax - melMin) * i / (nMels + 1)
        }

        val filterFreqsHz = FloatArray(nMels + 2) { i ->
            melToHz(melPoints[i])
        }

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
}

/**
 * Bluestein's algorithm for computing exact N-point DFT when N is not a power of 2.
 *
 * Converts the N-point DFT into circular convolution (via radix-2 FFT of size M >= 2N-1).
 * Precomputes chirp factors and H=FFT(h) once; per-frame cost is 2 forward FFTs + 1 inverse FFT.
 *
 * For Whisper (N=400), M=1024. About 3x slower than direct radix-2 FFT of 512,
 * but produces frequency bins at the correct N-point spacing.
 */
private class BluesteinState(private val N: Int) {
    private val M = nextPowerOf2(2 * N - 1)
    private val nFreqs = N / 2 + 1

    // Chirp: exp(-pi*i * n^2 / N)
    private val chirpReal = FloatArray(N)
    private val chirpImag = FloatArray(N)

    // Precomputed FFT of conjugate chirp sequence (for circular convolution)
    private val hFftReal: FloatArray
    private val hFftImag: FloatArray

    // Working arrays (reused across calls to avoid allocation)
    private val workReal = FloatArray(M)
    private val workImag = FloatArray(M)

    init {
        // Chirp: exp(-pi*i * n^2 / N)
        for (n in 0 until N) {
            val angle = -PI * n.toLong() * n / N
            chirpReal[n] = cos(angle).toFloat()
            chirpImag[n] = sin(angle).toFloat()
        }

        // h[n] = exp(+pi*i * n^2 / N), arranged for circular convolution
        val hR = FloatArray(M)
        val hI = FloatArray(M)
        for (n in 0 until N) {
            val angle = PI * n.toLong() * n / N
            val cr = cos(angle).toFloat()
            val ci = sin(angle).toFloat()
            hR[n] = cr; hI[n] = ci
            if (n > 0) { hR[M - n] = cr; hI[M - n] = ci }
        }

        // Precompute H = FFT_M(h)
        fftRadix2(hR, hI)
        hFftReal = hR
        hFftImag = hI
    }

    /**
     * Compute power spectrum |X[k]|^2 for k = 0..N/2 using exact N-point DFT.
     *
     * @param x Real-valued input of length N (windowed frame)
     * @param output Array of length N/2+1 to receive |X[k]|^2
     */
    fun powerSpectrum(x: FloatArray, output: FloatArray) {
        val w = workReal
        val wi = workImag

        // Step 1: a[n] = x[n] * chirp[n], zero-padded to M
        w.fill(0f)
        wi.fill(0f)
        for (n in 0 until N) {
            w[n] = x[n] * chirpReal[n]
            wi[n] = x[n] * chirpImag[n]
        }

        // Step 2: A = FFT_M(a)
        fftRadix2(w, wi)

        // Step 3: C = A * H (element-wise complex multiplication)
        for (i in 0 until M) {
            val tr = w[i] * hFftReal[i] - wi[i] * hFftImag[i]
            val ti = w[i] * hFftImag[i] + wi[i] * hFftReal[i]
            w[i] = tr
            wi[i] = ti
        }

        // Step 4: c = IFFT_M(C) via conjugate-FFT-conjugate-scale
        for (i in 0 until M) wi[i] = -wi[i]
        fftRadix2(w, wi)
        val scale = 1f / M
        for (i in 0 until M) {
            w[i] *= scale
            wi[i] = -wi[i] * scale
        }

        // Step 5: X[k] = chirp[k] * c[k], output |X[k]|^2
        for (k in 0 until nFreqs) {
            val xr = w[k] * chirpReal[k] - wi[k] * chirpImag[k]
            val xi = w[k] * chirpImag[k] + wi[k] * chirpReal[k]
            output[k] = xr * xr + xi * xi
        }
    }
}

/**
 * In-place radix-2 Cooley-Tukey FFT. Arrays must be power-of-2 length.
 */
private fun fftRadix2(real: FloatArray, imag: FloatArray) {
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
