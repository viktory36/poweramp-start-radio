package com.powerampstartradio.indexing

import android.content.Context
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer
import java.io.File
import kotlin.math.ceil
import kotlin.math.min

/**
 * Music Flamingo LiteRT inference for on-device audio embedding generation.
 *
 * Uses two TFLite models (converted from PyTorch via litert-torch):
 * - Encoder: mel [1,128,3000] + audio_times [1,750] -> hidden [1,750,1280]
 *   (No mask input — the TFLite model assumes full 30s chunks with no padding)
 * - Projector: hidden [1,750,1280] -> projected [1,750,3584]
 *
 * Replicates the desktop FlamingoEmbeddingGenerator:
 * - Full non-overlapping 30s coverage, max 60 chunks
 * - Compute Whisper-compatible mel spectrogram for each chunk -> [128, 3000]
 * - Construct audio_times with absolute timestamps (frame_idx * 0.04 + chunk_start)
 * - Run encoder + projector -> [750, 3584]
 * - Mean pool over time -> 3584d per chunk
 * - Average across chunks, L2-normalize
 *
 * If no projector model is found, output is 1280-dim (encoder-only).
 *
 * @param encoderFile Path to the encoder .tflite model
 * @param projectorFile Path to the projector .tflite model (optional)
 * @param accelerator Hardware accelerator to use (NPU, GPU, CPU). Falls back automatically.
 * @param context Android context (required for NPU acceleration)
 */
class FlamingoInference(
    encoderFile: File,
    projectorFile: File? = null,
    accelerator: Accelerator = Accelerator.CPU,
    context: Context? = null,
) {

    companion object {
        private const val TAG = "FlamingoInference"
        private const val SAMPLE_RATE = 16000
        private const val CHUNK_DURATION_S = 30
        private const val CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_DURATION_S  // 480000
        private const val MAX_CHUNKS = 60
        private const val NUM_MEL_FRAMES = 3000
        private const val N_MELS = 128
        private const val NUM_FRAMES = 750  // Post-pool frames per 30s chunk
        private const val FRAME_DURATION_S = 0.04f  // 30s / 750 frames
        private const val ENCODER_DIM = 1280
        private const val PROJECTED_DIM = 3584
    }

    private val encoderModel: CompiledModel
    private val projectorModel: CompiledModel?
    private val melSpectrogram = MelSpectrogram(center = true, melScale = MelScale.SLANEY)

    val outputDim: Int

    /** Which accelerator is actually in use for the encoder. */
    val activeAccelerator: Accelerator

    // Pre-allocated TensorBuffers for encoder
    private val encoderInputBuffers: List<TensorBuffer>
    private val encoderOutputBuffers: List<TensorBuffer>

    // Pre-allocated TensorBuffers for projector (if available)
    private val projectorInputBuffers: List<TensorBuffer>?
    private val projectorOutputBuffers: List<TensorBuffer>?

    // Pre-allocated flat arrays for input data
    private val melFlat = FloatArray(N_MELS * NUM_MEL_FRAMES)

    init {
        // Load encoder
        val encResult = createModelWithFallback(encoderFile.absolutePath, accelerator, context)
        encoderModel = encResult.first
        activeAccelerator = encResult.second

        encoderInputBuffers = encoderModel.createInputBuffers()
        encoderOutputBuffers = encoderModel.createOutputBuffers()

        // Load projector (if available)
        if (projectorFile != null && projectorFile.exists()) {
            val projResult = createModelWithFallback(projectorFile.absolutePath, activeAccelerator, context)
            projectorModel = projResult.first
            projectorInputBuffers = projectorModel.createInputBuffers()
            projectorOutputBuffers = projectorModel.createOutputBuffers()
            outputDim = PROJECTED_DIM
        } else {
            projectorModel = null
            projectorInputBuffers = null
            projectorOutputBuffers = null
            outputDim = ENCODER_DIM
        }

        Log.i(TAG, "Flamingo loaded: encoder=${encoderFile.name} " +
                "(${encoderFile.length() / 1024 / 1024}MB), " +
                "projector=${projectorFile?.name ?: "none"}, " +
                "output_dim=$outputDim, accelerator=$activeAccelerator")
    }

    /**
     * Generate embedding from decoded audio.
     *
     * @param audio Decoded audio at 16kHz
     * @return 3584-dim (or 1280-dim) L2-normalized embedding, or null on failure
     */
    fun generateEmbedding(audio: AudioDecoder.DecodedAudio): FloatArray? {
        require(audio.sampleRate == SAMPLE_RATE) {
            "Flamingo requires ${SAMPLE_RATE}Hz audio, got ${audio.sampleRate}Hz"
        }

        if (audio.durationS < 3.0f) {
            Log.w(TAG, "Audio too short (${audio.durationS}s)")
            return null
        }

        // Select chunk positions (stratified, non-overlapping)
        val numChunks = calculateNumChunks(audio.durationS)
        val positions = selectChunkPositions(audio.durationS, numChunks)

        Log.d(TAG, "Processing ${positions.size} chunks from ${audio.durationS}s audio")

        val sumEmbedding = FloatArray(outputDim)
        var count = 0

        for (pos in positions) {
            val embedding = processChunk(audio.samples, pos) ?: continue
            for (i in 0 until outputDim) {
                sumEmbedding[i] += embedding[i]
            }
            count++
        }

        if (count == 0) {
            Log.w(TAG, "All chunk inferences failed")
            return null
        }

        // Average and L2-normalize
        val countF = count.toFloat()
        for (i in 0 until outputDim) {
            sumEmbedding[i] /= countF
        }
        l2Normalize(sumEmbedding)

        return sumEmbedding
    }

    /**
     * Process a single 30s chunk: compute mel, run encoder+projector, mean-pool.
     */
    private fun processChunk(samples: FloatArray, positionS: Float): FloatArray? {
        val startSample = (positionS * SAMPLE_RATE).toInt()
        val endSample = min(startSample + CHUNK_SAMPLES, samples.size)
        val chunkSamples = samples.copyOfRange(startSample, endSample)

        // Pad to full 30s if needed
        val paddedChunk = if (chunkSamples.size < CHUNK_SAMPLES) {
            FloatArray(CHUNK_SAMPLES).also { chunkSamples.copyInto(it) }
        } else {
            chunkSamples
        }

        // Compute mel spectrogram and apply Whisper log normalization.
        // Center padding produces 3001 frames; drop last to match Whisper's [:, :-1].
        val rawMel = melSpectrogram.compute(paddedChunk)
        val mel = Array(rawMel.size) { m -> rawMel[m].copyOf(rawMel[m].size - 1) }
        MelSpectrogram.whisperNormalize(mel)

        // Construct audio_times: absolute timestamps per post-pool frame
        val audioTimes = FloatArray(NUM_FRAMES) { frame ->
            frame * FRAME_DURATION_S + positionS
        }

        return runInference(mel, audioTimes)
    }

    /**
     * Run inference: encoder -> [optional projector] -> mean pool.
     *
     * @param mel [128, numFrames] mel spectrogram
     * @param audioTimes [750] absolute timestamps
     * @return Mean-pooled embedding [outputDim], or null on failure
     */
    private fun runInference(mel: Array<FloatArray>, audioTimes: FloatArray): FloatArray? {
        return try {
            // Flatten mel [128, 3000] to flat array with zero-padding
            for (m in 0 until N_MELS) {
                val rowOffset = m * NUM_MEL_FRAMES
                val srcFrames = min(mel[m].size, NUM_MEL_FRAMES)
                mel[m].copyInto(melFlat, rowOffset, 0, srcFrames)
                for (t in srcFrames until NUM_MEL_FRAMES) {
                    melFlat[rowOffset + t] = 0f
                }
            }

            // Write inputs: [mel, audio_times]
            encoderInputBuffers[0].writeFloat(melFlat)
            encoderInputBuffers[1].writeFloat(audioTimes)

            // Step 1: Encoder [1,128,3000] + [1,750] -> [1,750,1280]
            encoderModel.run(encoderInputBuffers, encoderOutputBuffers)

            // Step 2: Projector (if available) [1,750,1280] -> [1,750,3584]
            val finalOutput: FloatArray
            val finalDim: Int
            if (projectorModel != null && projectorInputBuffers != null && projectorOutputBuffers != null) {
                // Pass encoder output to projector input
                val encoderOutput = encoderOutputBuffers[0].readFloat()
                projectorInputBuffers[0].writeFloat(encoderOutput)
                projectorModel.run(projectorInputBuffers, projectorOutputBuffers)
                finalOutput = projectorOutputBuffers[0].readFloat()
                finalDim = PROJECTED_DIM
            } else {
                finalOutput = encoderOutputBuffers[0].readFloat()
                finalDim = ENCODER_DIM
            }

            // Mean pool over time dimension: [1, 750, dim] -> [dim]
            val pooled = FloatArray(finalDim)
            for (frame in 0 until NUM_FRAMES) {
                val frameOffset = frame * finalDim
                for (i in 0 until finalDim) {
                    pooled[i] += finalOutput[frameOffset + i]
                }
            }
            val numFramesF = NUM_FRAMES.toFloat()
            for (i in 0 until finalDim) {
                pooled[i] /= numFramesF
            }

            pooled
        } catch (e: Exception) {
            Log.e(TAG, "Flamingo inference failed: ${e.message}", e)
            null
        }
    }

    private fun calculateNumChunks(durationS: Float): Int {
        return maxOf(1, minOf(ceil(durationS / CHUNK_DURATION_S).toInt(), MAX_CHUNKS))
    }

    private fun selectChunkPositions(durationS: Float, numChunks: Int): List<Float> {
        val usable = durationS - CHUNK_DURATION_S
        if (usable <= 0) return listOf(0f)
        if (numChunks == 1) return listOf(usable / 2f)
        return (0 until numChunks).map { i -> usable * i / (numChunks - 1) }
    }

    fun close() {
        encoderInputBuffers.forEach { it.close() }
        encoderOutputBuffers.forEach { it.close() }
        encoderModel.close()
        projectorInputBuffers?.forEach { it.close() }
        projectorOutputBuffers?.forEach { it.close() }
        projectorModel?.close()
    }
}
