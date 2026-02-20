package com.powerampstartradio.indexing

import android.util.Log

/**
 * Native mel spectrogram computation via JNI.
 *
 * Provides a ~2-4x speedup over the pure Kotlin implementation for the
 * FFT-heavy mel spectrogram computation, especially for Bluestein's algorithm
 * (Whisper's non-power-of-2 nFft=400).
 *
 * Falls back gracefully: if the native library isn't available,
 * [MelSpectrogram] uses its built-in Kotlin implementation.
 */
object NativeMel {
    private const val TAG = "NativeMel"
    private val loaded: Boolean

    init {
        loaded = try {
            System.loadLibrary("mel-jni")
            true
        } catch (e: UnsatisfiedLinkError) {
            Log.w(TAG, "mel-jni not available, using Kotlin fallback")
            false
        }
    }

    val isAvailable: Boolean get() = loaded

    /**
     * Compute mel spectrogram natively.
     *
     * @return Flat array: `[numFrames_as_float, mel[0][0], ..., mel[nMels-1][numFrames-1]]`.
     *   Mel-major layout: `mel[m * numFrames + t]` starting at index 1.
     *   Returns null on failure.
     */
    @JvmStatic
    external fun nativeComputeMel(
        audio: FloatArray,
        sampleRate: Int,
        nFft: Int,
        hopLength: Int,
        nMels: Int,
        fMin: Float,
        fMax: Float,
        center: Boolean,
        melScale: Int,  // 0 = HTK, 1 = SLANEY
    ): FloatArray?
}
