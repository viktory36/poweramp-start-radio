package com.powerampstartradio.indexing

import android.util.Log
import com.google.ai.edge.litert.Accelerator
import java.io.File
import kotlin.math.sqrt

/**
 * MERT-v1-95M LiteRT inference for on-device audio feature extraction.
 *
 * Uses a TFLite model (converted from PyTorch via litert-torch):
 * - Input:  raw waveform [1, 120000] (5 seconds at 24kHz)
 * - Output: [1, 768] mean-pooled features (averaged over all transformer layers and time)
 *
 * Processing strategy:
 * - Audio is split into 5-second non-overlapping windows
 * - Each window is processed independently through MERT
 * - Output features are streamed to the caller (disk or memory)
 * - CLaMP3 audio encoder handles aggregation into a final embedding
 *
 * @param modelFile Path to the mert.tflite model file
 * @param accelerator Hardware accelerator to use (GPU or CPU)
 */
class MertInference(
    modelFile: File,
    accelerator: Accelerator = Accelerator.GPU,
) {

    companion object {
        private const val TAG = "MertInference"
        const val SAMPLE_RATE = 24000
        const val WINDOW_SEC = 5
        const val WINDOW_SAMPLES = SAMPLE_RATE * WINDOW_SEC  // 120000
        const val FEATURE_DIM = 768
        /** Max audio duration in seconds (must match desktop's max_duration=600). */
        const val MAX_DURATION_S = 600

        /**
         * Per-track zero-mean unit-variance normalization (in-place).
         * Matches Wav2Vec2FeatureExtractor(do_normalize=True) on desktop.
         */
        fun normalizeAudio(samples: FloatArray) {
            var sum = 0.0
            for (s in samples) sum += s
            val mean = (sum / samples.size).toFloat()

            var sumSq = 0.0
            for (s in samples) {
                val d = s - mean
                sumSq += d * d
            }
            val std = sqrt((sumSq / samples.size).toFloat() + 1e-7f)

            for (i in samples.indices) {
                samples[i] = (samples[i] - mean) / std
            }
        }
    }

    private val model: com.google.ai.edge.litert.CompiledModel
    private val inputBuffers: List<com.google.ai.edge.litert.TensorBuffer>
    private val outputBuffers: List<com.google.ai.edge.litert.TensorBuffer>

    /** Which accelerator is actually in use (may differ from requested if fallback occurred). */
    val activeAccelerator: Accelerator

    // Pre-allocated buffer for one 5s window (reused across all windows)
    private val windowBuffer = FloatArray(WINDOW_SAMPLES)

    init {
        val result = createReadyModel(modelFile.absolutePath, accelerator)
        model = result.model
        activeAccelerator = result.accelerator
        inputBuffers = result.inputBuffers
        outputBuffers = result.outputBuffers

        Log.i(TAG, "MERT loaded: ${modelFile.name} " +
                "(${modelFile.length() / 1024 / 1024}MB), accelerator=$activeAccelerator")
    }

    /**
     * Compute the number of 5-second windows for audio of this duration.
     * Used for pre-computing total work steps for ETA.
     */
    fun windowCount(durationS: Float): Int {
        val totalSamples = (durationS * SAMPLE_RATE).toInt()
        return totalSamples / WINDOW_SAMPLES
    }

    /**
     * Extract MERT features from decoded audio, streaming each window's
     * features via [onFeatureExtracted].
     *
     * Each output feature is a 768d vector representing one 5-second window.
     * Does NOT accumulate features in memory — the caller decides how to store them
     * (disk spill for two-phase GPU, or in-memory for small batches).
     *
     * @param audio Decoded audio at 24kHz
     * @param onFeatureExtracted Callback with 768d feature vector per window
     * @param onWindowDone Optional callback for progress tracking (called after each window)
     * @return Number of successfully processed windows (0 = failure)
     */
    fun extractFeaturesStreaming(
        audio: AudioDecoder.DecodedAudio,
        onFeatureExtracted: (FloatArray) -> Unit,
        onWindowDone: (() -> Unit)? = null,
    ): Int {
        require(audio.sampleRate == SAMPLE_RATE) {
            "MERT requires ${SAMPLE_RATE}Hz audio, got ${audio.sampleRate}Hz"
        }

        val numWindows = audio.samples.size / WINDOW_SAMPLES
        if (numWindows == 0) {
            Log.w(TAG, "Audio too short (${audio.durationS}s < ${WINDOW_SEC}s)")
            return 0
        }

        // Per-track zero-mean unit-variance normalization
        // (matches Wav2Vec2FeatureExtractor(do_normalize=True) on desktop)
        normalizeAudio(audio.samples)

        var count = 0
        var totalInferMs = 0L

        for (w in 0 until numWindows) {
            val startSample = w * WINDOW_SAMPLES
            System.arraycopy(audio.samples, startSample, windowBuffer, 0, WINDOW_SAMPLES)

            val feature = runInference(windowBuffer) { inferMs ->
                totalInferMs += inferMs
            }
            if (feature != null) {
                onFeatureExtracted(feature)
                count++
            }
            onWindowDone?.invoke()
        }

        Log.i(TAG, "TIMING: mert $numWindows windows: inference=${totalInferMs}ms, " +
            "${if (numWindows > 0) totalInferMs / numWindows else 0}ms/window")

        return count
    }

    /**
     * Run MERT inference on a single 5-second window.
     *
     * @return 768d feature vector, or null on failure
     */
    private fun runInference(
        window: FloatArray,
        onTiming: ((inferMs: Long) -> Unit)? = null,
    ): FloatArray? {
        return try {
            inputBuffers[0].writeFloat(window)

            val start = System.nanoTime()
            model.run(inputBuffers, outputBuffers)
            val output = outputBuffers[0].readFloat()
            val inferMs = (System.nanoTime() - start) / 1_000_000
            onTiming?.invoke(inferMs)

            if (output.size >= FEATURE_DIM) {
                output.copyOf(FEATURE_DIM)
            } else {
                Log.w(TAG, "Unexpected output size: ${output.size}")
                null
            }
        } catch (e: Exception) {
            Log.e(TAG, "MERT inference failed: ${e.message}")
            null
        }
    }

    fun close() {
        inputBuffers.forEach { it.close() }
        outputBuffers.forEach { it.close() }
        model.close()
    }
}
