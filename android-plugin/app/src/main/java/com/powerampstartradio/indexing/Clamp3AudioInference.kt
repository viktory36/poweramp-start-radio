package com.powerampstartradio.indexing

import android.util.Log
import com.google.ai.edge.litert.Accelerator
import java.io.File
import kotlin.math.ceil
import kotlin.math.min

/**
 * CLaMP3 audio encoder LiteRT inference for on-device embedding generation.
 *
 * Uses a TFLite model (converted from PyTorch via litert-torch):
 * - Input 0: audio_inputs [1, 128, 768] (MERT features, zero-padded)
 * - Input 1: audio_masks  [1, 128] float32 (1=real, 0=pad)
 * - Output:  [1, 768] projected features (not L2-normalized)
 *
 * The model computes a GPU-safe additive attention mask internally using -1e4
 * (fits in FP16) instead of BERT's default finfo.min (-3.4e38 → -inf → NaN).
 * ADD broadcasts [B,1,1,S] natively on GPU — no BROADCAST_TO ops needed.
 *
 * Handles segmentation: tracks with >128 MERT windows are split into
 * segments, each encoded separately, then weight-averaged by valid frame count.
 * L2 normalization is applied after averaging.
 *
 * This is the second phase of the two-phase GPU pipeline:
 * 1. MertInference extracts 768d features per 5-second window
 * 2. Clamp3AudioInference aggregates windows into a single 768d embedding
 *
 * @param modelFile Path to the clamp3_audio.tflite model file
 * @param accelerator Hardware accelerator to use (GPU or CPU)
 */
class Clamp3AudioInference(
    modelFile: File,
    accelerator: Accelerator = Accelerator.GPU,
) {

    companion object {
        private const val TAG = "Clamp3AudioInference"
        const val MAX_WINDOWS = 128
        const val FEATURE_DIM = 768
        const val EMBEDDING_DIM = 768

        /** Floats per MERT feature window. */
        const val WINDOW_FLOATS = FEATURE_DIM

        /** Bytes per MERT feature window (768 × 4 = 3072 bytes). */
        const val WINDOW_BYTES = WINDOW_FLOATS * 4
    }

    private val model: com.google.ai.edge.litert.CompiledModel
    private val inputBuffers: List<com.google.ai.edge.litert.TensorBuffer>
    private val outputBuffers: List<com.google.ai.edge.litert.TensorBuffer>

    /** Which accelerator is actually in use. */
    val activeAccelerator: Accelerator

    // Pre-allocated flat arrays for model input [1, 128, 768] and [1, 128]
    private val audioInputFlat = FloatArray(MAX_WINDOWS * FEATURE_DIM)
    private val audioMaskFlat = FloatArray(MAX_WINDOWS)

    init {
        val result = createReadyModel(modelFile.absolutePath, accelerator)
        model = result.model
        activeAccelerator = result.accelerator
        inputBuffers = result.inputBuffers
        outputBuffers = result.outputBuffers

        Log.i(TAG, "CLaMP3 audio encoder loaded: ${modelFile.name} " +
                "(${modelFile.length() / 1024 / 1024}MB), accelerator=$activeAccelerator")
    }

    /**
     * Compute the number of CLaMP3 segments needed for this many MERT windows.
     * Accounts for the prepended + appended zero vectors (CLaMP3 training convention).
     * Used for pre-computing total work steps for ETA.
     */
    fun segmentCount(numWindows: Int): Int {
        val totalFrames = numWindows + 2  // +2 for zero-vector bookends
        return maxOf(1, ceil(totalFrames.toFloat() / MAX_WINDOWS).toInt())
    }

    /**
     * Encode MERT features into a CLaMP3 embedding, streaming features
     * from a callback. Designed for disk-spilled MERT features to avoid
     * loading all windows into memory at once.
     *
     * Matches the desktop CLaMP3 pipeline:
     * 1. Prepend + append zero vector (CLaMP3 training convention)
     * 2. Segment into 128-frame windows
     * 3. Last segment uses the final 128 frames (may overlap with previous)
     * 4. Weight-average segments by valid frame count
     *
     * @param numWindows Total number of MERT feature windows for this track
     * @param readNextWindow Callback that returns the next 768d MERT feature vector
     * @param onSegmentDone Optional callback for progress tracking
     * @return 768d L2-normalized CLaMP3 embedding, or null on failure
     */
    fun encodeStreaming(
        numWindows: Int,
        readNextWindow: () -> FloatArray,
        onSegmentDone: (() -> Unit)? = null,
    ): FloatArray? {
        if (numWindows == 0) return null

        // Read all MERT windows into memory for prepend/append + overlapping last segment.
        // Memory: numWindows * 768 * 4 bytes (~180 windows for 15min track = ~540KB).
        val allWindows = Array(numWindows) { readNextWindow() }

        // Prepend + append zero vector (matching CLaMP3 training pipeline)
        val totalFrames = numWindows + 2
        val zeroVec = FloatArray(FEATURE_DIM)

        // Build segment list matching desktop: segment into 128-frame windows,
        // then replace last segment with the final 128 frames (may overlap).
        val segStarts = mutableListOf<Int>()
        var pos = 0
        while (pos < totalFrames) {
            segStarts.add(pos)
            pos += MAX_WINDOWS
        }
        // Last segment: use last MAX_WINDOWS frames (may overlap with previous)
        if (segStarts.size > 1 || totalFrames > MAX_WINDOWS) {
            segStarts[segStarts.lastIndex] = maxOf(0, totalFrames - MAX_WINDOWS)
        }

        val numSegments = segStarts.size
        val sumEmbedding = FloatArray(EMBEDDING_DIM)
        var totalWeight = 0f
        var count = 0
        var totalInferMs = 0L

        for (s in 0 until numSegments) {
            val segStart = segStarts[s]
            val segEnd = minOf(segStart + MAX_WINDOWS, totalFrames)
            val segWindows = segEnd - segStart

            // Fill input buffer with features from the virtual array
            // (zero at index 0, allWindows at 1..numWindows, zero at numWindows+1)
            audioMaskFlat.fill(0f)
            for (w in 0 until segWindows) {
                val virtualIdx = segStart + w
                val feature = when (virtualIdx) {
                    0 -> zeroVec                         // prepended zero
                    totalFrames - 1 -> zeroVec           // appended zero
                    else -> allWindows[virtualIdx - 1]   // actual MERT feature
                }
                System.arraycopy(feature, 0, audioInputFlat, w * FEATURE_DIM, FEATURE_DIM)
                audioMaskFlat[w] = 1f
            }
            // Zero-pad remainder of features
            for (w in segWindows until MAX_WINDOWS) {
                audioInputFlat.fill(0f, w * FEATURE_DIM, (w + 1) * FEATURE_DIM)
            }

            // Run inference
            val embedding = runInference { inferMs ->
                totalInferMs += inferMs
            }
            if (embedding != null) {
                // Weight by valid frame count (matches desktop weighting)
                val weight = if (s < numSegments - 1) {
                    MAX_WINDOWS.toFloat()
                } else {
                    // Last segment weight = remainder (or MAX_WINDOWS if evenly divisible)
                    val remain = totalFrames % MAX_WINDOWS
                    if (remain == 0) MAX_WINDOWS.toFloat() else remain.toFloat()
                }
                for (i in 0 until EMBEDDING_DIM) {
                    sumEmbedding[i] += embedding[i] * weight
                }
                totalWeight += weight
                count++
            }
            onSegmentDone?.invoke()
        }

        Log.i(TAG, "TIMING: clamp3_audio $numSegments segments ($numWindows windows): " +
            "inference=${totalInferMs}ms")

        if (count == 0 || totalWeight == 0f) return null

        // Weighted average and L2-normalize
        for (i in 0 until EMBEDDING_DIM) {
            sumEmbedding[i] /= totalWeight
        }
        l2Normalize(sumEmbedding)

        return sumEmbedding
    }

    /**
     * Convenience: encode from an in-memory list of MERT feature windows.
     *
     * @param features List of 768d MERT feature vectors
     * @param numWindows Number of valid windows (should equal features.size)
     * @return 768d L2-normalized CLaMP3 embedding, or null on failure
     */
    fun encode(features: List<FloatArray>, numWindows: Int): FloatArray? {
        var idx = 0
        return encodeStreaming(numWindows, readNextWindow = {
            features[idx++]
        })
    }

    private fun runInference(
        onTiming: ((inferMs: Long) -> Unit)? = null,
    ): FloatArray? {
        return try {
            inputBuffers[0].writeFloat(audioInputFlat)
            inputBuffers[1].writeFloat(audioMaskFlat)

            val start = System.nanoTime()
            model.run(inputBuffers, outputBuffers)
            val output = outputBuffers[0].readFloat()
            val inferMs = (System.nanoTime() - start) / 1_000_000
            onTiming?.invoke(inferMs)

            if (output.size >= EMBEDDING_DIM) {
                output.copyOf(EMBEDDING_DIM)
            } else {
                Log.w(TAG, "Unexpected output size: ${output.size}")
                null
            }
        } catch (e: Exception) {
            Log.e(TAG, "CLaMP3 audio inference failed: ${e.message}")
            null
        }
    }

    fun close() {
        inputBuffers.forEach { it.close() }
        outputBuffers.forEach { it.close() }
        model.close()
    }
}
