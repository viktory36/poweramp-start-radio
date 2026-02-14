package com.powerampstartradio.indexing

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.util.Log
import java.io.File
import java.nio.FloatBuffer
import kotlin.math.ceil
import kotlin.math.min
import kotlin.math.sqrt

/**
 * MuQ-MuLan ONNX inference for on-device audio embedding generation.
 *
 * Replicates the desktop MuLanEmbeddingGenerator chunking strategy:
 * - 1 chunk (30s) per minute of audio, max 30
 * - Each 30s chunk -> 3x 10s clips (matching MuQ-MuLan's _get_all_clips)
 * - Each 10s clip runs through ONNX -> 512d
 * - Average all clip embeddings, L2-normalize
 *
 * Input to ONNX model: [1, 240000] raw waveform at 24kHz (10s clip)
 * Output: [1, 512] L2-normalized embedding
 */
class MuLanInference(modelFile: File) {

    companion object {
        private const val TAG = "MuLanInference"
        private const val SAMPLE_RATE = 24000
        private const val CLIP_DURATION_S = 10
        private const val CLIP_SAMPLES = SAMPLE_RATE * CLIP_DURATION_S  // 240000
        private const val CHUNK_DURATION_S = 30
        private const val MAX_CHUNKS = 30
        private const val EMBEDDING_DIM = 512
    }

    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    init {
        val opts = OrtSession.SessionOptions().apply {
            // Try NNAPI for Snapdragon NPU, fall back to CPU
            try {
                addNnapi()
                Log.i(TAG, "Using NNAPI execution provider")
            } catch (e: Exception) {
                Log.i(TAG, "NNAPI not available, using CPU: ${e.message}")
            }
        }
        session = env.createSession(modelFile.absolutePath, opts)
        Log.i(TAG, "MuQ-MuLan ONNX session loaded: ${modelFile.name}")
    }

    /**
     * Generate a 512-dim embedding from decoded audio.
     *
     * @param audio Decoded audio at 24kHz
     * @return 512-dim L2-normalized embedding, or null on failure
     */
    fun generateEmbedding(audio: AudioDecoder.DecodedAudio): FloatArray? {
        require(audio.sampleRate == SAMPLE_RATE) {
            "MuQ-MuLan requires ${SAMPLE_RATE}Hz audio, got ${audio.sampleRate}Hz"
        }

        val durationS = audio.durationS
        if (durationS < CHUNK_DURATION_S) {
            Log.w(TAG, "Audio too short (${durationS}s < ${CHUNK_DURATION_S}s)")
            return null
        }

        // Select chunk positions (stratified sampling)
        val numChunks = calculateNumChunks(durationS)
        val positions = selectChunkPositions(durationS, numChunks)

        // Extract 10s clips from each 30s chunk position
        val clips = mutableListOf<FloatArray>()
        val chunkSamples = CHUNK_DURATION_S * SAMPLE_RATE

        for (pos in positions) {
            val startSample = (pos * SAMPLE_RATE).toInt()
            val endSample = min(startSample + chunkSamples, audio.samples.size)
            if (endSample - startSample < CLIP_SAMPLES) continue

            // Extract 3 x 10s clips from this 30s chunk (matching _get_all_clips)
            for (clipIdx in 0 until 3) {
                val clipStart = startSample + clipIdx * CLIP_SAMPLES
                val clipEnd = clipStart + CLIP_SAMPLES
                if (clipEnd <= audio.samples.size) {
                    clips.add(audio.samples.copyOfRange(clipStart, clipEnd))
                }
            }
        }

        if (clips.isEmpty()) {
            Log.w(TAG, "No valid clips extracted")
            return null
        }

        Log.d(TAG, "Running inference on ${clips.size} clips")

        // Run each clip through ONNX and accumulate embeddings
        val sumEmbedding = FloatArray(EMBEDDING_DIM)
        var count = 0

        for (clip in clips) {
            val embedding = runInference(clip) ?: continue
            for (i in 0 until EMBEDDING_DIM) {
                sumEmbedding[i] += embedding[i]
            }
            count++
        }

        if (count == 0) {
            Log.w(TAG, "All clip inferences failed")
            return null
        }

        // Average and L2-normalize
        for (i in 0 until EMBEDDING_DIM) {
            sumEmbedding[i] /= count
        }
        l2Normalize(sumEmbedding)

        return sumEmbedding
    }

    private fun runInference(clip: FloatArray): FloatArray? {
        return try {
            val inputTensor = OnnxTensor.createTensor(
                env,
                FloatBuffer.wrap(clip),
                longArrayOf(1, clip.size.toLong())
            )

            val results = session.run(mapOf("wav" to inputTensor))
            val outputTensor = results[0] as OnnxTensor

            @Suppress("UNCHECKED_CAST")
            val output = (outputTensor.value as Array<FloatArray>)[0]

            inputTensor.close()
            results.close()

            output
        } catch (e: Exception) {
            Log.e(TAG, "MuQ-MuLan inference failed: ${e.message}")
            null
        }
    }

    private fun calculateNumChunks(durationS: Float): Int {
        val minutes = durationS / 60f
        return maxOf(1, minOf(minutes.toInt(), MAX_CHUNKS))
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

/** L2-normalize a float array in place. */
internal fun l2Normalize(arr: FloatArray) {
    var norm = 0f
    for (v in arr) norm += v * v
    norm = sqrt(norm)
    if (norm > 1e-10f) {
        for (i in arr.indices) arr[i] /= norm
    }
}
