package com.powerampstartradio.indexing

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.util.Log
import java.io.File
import java.nio.FloatBuffer
import java.nio.LongBuffer
import kotlin.math.ceil
import kotlin.math.min

/**
 * Music Flamingo ONNX inference for on-device audio embedding generation.
 *
 * Replicates the desktop FlamingoEmbeddingGenerator:
 * - Full non-overlapping 30s coverage, max 60 chunks
 * - Compute Whisper-compatible mel spectrogram for each chunk -> [128, 3000]
 * - Construct audio_times with absolute timestamps (frame_idx * 0.04 + chunk_start)
 * - Run ONNX: (mel, mask, audio_times) -> [750, 3584]
 * - Mean pool over time -> 3584d per chunk
 * - Average across chunks, L2-normalize
 *
 * Mel spectrogram computed in Kotlin (MelSpectrogram class).
 * ONNX model takes: input_features [B,128,3000], mask [B,3000], audio_times [B,750]
 */
class FlamingoInference(modelFile: File) {

    companion object {
        private const val TAG = "FlamingoInference"
        private const val SAMPLE_RATE = 16000
        private const val CHUNK_DURATION_S = 30
        private const val CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_DURATION_S  // 480000
        private const val MAX_CHUNKS = 60
        private const val NUM_FRAMES = 750  // Post-pool frames per 30s chunk
        private const val FRAME_DURATION_S = 0.04f  // 30s / 750 frames
        private const val OUTPUT_DIM = 3584  // With projector
    }

    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val melSpectrogram = MelSpectrogram()

    // Detect output dimension from model
    val outputDim: Int

    init {
        val opts = OrtSession.SessionOptions().apply {
            try {
                addNnapi()
                Log.i(TAG, "Using NNAPI execution provider")
            } catch (e: Exception) {
                Log.i(TAG, "NNAPI not available, using CPU: ${e.message}")
            }
        }
        session = env.createSession(modelFile.absolutePath, opts)

        // Detect output dimension from model metadata
        val outputInfo = session.outputInfo
        val outputShape = outputInfo.values.first().info.toString()
        outputDim = if (outputShape.contains("3584")) OUTPUT_DIM else 1280
        Log.i(TAG, "Flamingo ONNX session loaded: ${modelFile.name}, output_dim=$outputDim")
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
        for (i in 0 until outputDim) {
            sumEmbedding[i] /= count
        }
        l2Normalize(sumEmbedding)

        return sumEmbedding
    }

    /**
     * Process a single 30s chunk: compute mel, run ONNX, mean-pool time dimension.
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

        // Compute mel spectrogram [128, 3000]
        val mel = melSpectrogram.compute(paddedChunk)

        // Construct audio_times: absolute timestamps per post-pool frame
        val audioTimes = FloatArray(NUM_FRAMES) { frame ->
            frame * FRAME_DURATION_S + positionS
        }

        // Run ONNX inference
        return runInference(mel, audioTimes)
    }

    /**
     * Run ONNX inference on mel spectrogram.
     *
     * @param mel [128, numFrames] mel spectrogram
     * @param audioTimes [750] absolute timestamps
     * @return Mean-pooled embedding [outputDim], or null on failure
     */
    private fun runInference(mel: Array<FloatArray>, audioTimes: FloatArray): FloatArray? {
        return try {
            val numMelFrames = mel[0].size

            // Flatten mel to [1, 128, numMelFrames]
            val melFlat = FloatArray(128 * numMelFrames)
            for (m in 0 until 128) {
                System.arraycopy(mel[m], 0, melFlat, m * numMelFrames, numMelFrames)
            }
            val melTensor = OnnxTensor.createTensor(
                env,
                FloatBuffer.wrap(melFlat),
                longArrayOf(1, 128, numMelFrames.toLong())
            )

            // All-ones mask [1, numMelFrames]
            val mask = LongArray(numMelFrames) { 1L }
            val maskTensor = OnnxTensor.createTensor(
                env,
                LongBuffer.wrap(mask),
                longArrayOf(1, numMelFrames.toLong())
            )

            // Audio times [1, 750]
            val timesTensor = OnnxTensor.createTensor(
                env,
                FloatBuffer.wrap(audioTimes),
                longArrayOf(1, NUM_FRAMES.toLong())
            )

            val results = session.run(mapOf(
                "input_features" to melTensor,
                "input_features_mask" to maskTensor,
                "audio_times" to timesTensor,
            ))

            // Output: [1, 750, outputDim]
            val outputTensor = results[0] as OnnxTensor

            @Suppress("UNCHECKED_CAST")
            val output = outputTensor.value as Array<Array<FloatArray>>
            val frames = output[0]  // [750, outputDim]

            // Mean pool over time dimension
            val pooled = FloatArray(outputDim)
            for (frame in frames) {
                for (i in 0 until outputDim) {
                    pooled[i] += frame[i]
                }
            }
            for (i in 0 until outputDim) {
                pooled[i] /= frames.size
            }

            melTensor.close()
            maskTensor.close()
            timesTensor.close()
            results.close()

            pooled
        } catch (e: Exception) {
            Log.e(TAG, "Flamingo inference failed: ${e.message}")
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
        session.close()
    }
}
