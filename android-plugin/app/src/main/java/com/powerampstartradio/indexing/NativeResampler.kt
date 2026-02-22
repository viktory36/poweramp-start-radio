package com.powerampstartradio.indexing

/**
 * High-quality audio resampling via libsoxr (native).
 *
 * Uses the SoX Resampler library which produces embeddings-grade resampling
 * quality that Flamingo/Whisper requires. Standard approaches (linear interp,
 * windowed sinc, Kaiser FIR) all produce catastrophically wrong embeddings
 * due to aliasing sensitivity in the Whisper encoder.
 */
object NativeResampler {
    init {
        System.loadLibrary("soxr-jni")
    }

    const val QUALITY_MQ = 0   // ~16-bit precision, fastest
    const val QUALITY_HQ = 1   // ~20-bit precision (default)
    const val QUALITY_VHQ = 2  // ~28-bit precision, slowest

    /**
     * Resample mono audio from [fromRate] to [toRate] Hz.
     *
     * @param samples Mono PCM float samples in [-1, 1]
     * @param fromRate Source sample rate (e.g. 44100)
     * @param toRate Target sample rate (e.g. 16000)
     * @param quality One of [QUALITY_MQ], [QUALITY_HQ], [QUALITY_VHQ]
     * @return Resampled samples at [toRate], or null on error
     */
    fun resample(
        samples: FloatArray,
        fromRate: Int,
        toRate: Int,
        quality: Int = QUALITY_HQ
    ): FloatArray? {
        if (fromRate == toRate) return samples
        return nativeResample(samples, fromRate, toRate, quality)
    }

    @JvmStatic
    private external fun nativeResample(
        input: FloatArray,
        fromRate: Int,
        toRate: Int,
        quality: Int
    ): FloatArray?
}
