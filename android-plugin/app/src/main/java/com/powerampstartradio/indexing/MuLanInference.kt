package com.powerampstartradio.indexing

import android.util.Log
import com.google.gson.Gson
import com.google.gson.annotations.SerializedName
import org.tensorflow.lite.Interpreter
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.log10
import kotlin.math.min
import kotlin.math.sqrt

/**
 * MuQ-MuLan LiteRT inference for on-device audio embedding generation.
 *
 * Uses a TFLite model (converted from PyTorch via litert-torch):
 * - Input: mel_features [1, 128, 1000] (normalized mel spectrogram from 10s @ 24kHz)
 * - Output: [1, 512] L2-normalized embedding
 *
 * Mel parameters are read from a JSON sidecar file (mulan_audio.mel_params.json)
 * generated during TFLite export. Hardcoded defaults match MuQ-MuLan-large.
 *
 * Chunking strategy (matches desktop MuLanEmbeddingGenerator):
 * - 1 chunk (30s) per minute of audio, max 30
 * - Each 30s chunk -> 3x 10s clips (matching MuQ-MuLan's _get_all_clips)
 * - Each 10s clip: compute mel -> normalize -> TFLite -> 512d
 * - Average all clip embeddings, L2-normalize
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
        private const val EXPECTED_MEL_FRAMES = 1000  // 240000 / 240 = 1000
        private const val N_MELS = 128
    }

    /** Mel spectrogram parameters from the JSON sidecar or defaults.
     *  Numeric fields use Double because the Python export writes `2048.0` (not `2048`)
     *  and Gson won't coerce `Double` → `Int` cleanly. */
    private data class MelParams(
        @SerializedName("n_fft") val nFft: Double = 2048.0,
        @SerializedName("hop_length") val hopLength: Double = 240.0,
        @SerializedName("win_length") val winLength: Double = 2048.0,
        @SerializedName("n_mels") val nMels: Double = 128.0,
        @SerializedName("sample_rate") val sampleRate: Double = 24000.0,
        @SerializedName("f_min") val fMin: Double = 0.0,
        @SerializedName("f_max") val fMax: Double? = null,  // null = Nyquist (sampleRate / 2)
        @SerializedName("power") val power: Double = 2.0,
        @SerializedName("norm_mean") val normMean: Double = 6.768444971712967,
        @SerializedName("norm_std") val normStd: Double = 18.417922652295623,
        @SerializedName("is_db") val isDb: Boolean = true,
    )

    private val interpreter: Interpreter
    private val melSpectrogram: MelSpectrogram
    private val melParams: MelParams

    // Pre-allocated I/O buffers for inference (avoid GC pressure)
    private val inputBuffer: ByteBuffer
    private val outputBuffer: ByteBuffer

    init {
        // Load mel params from JSON sidecar if available, else use defaults
        val jsonFile = File(modelFile.parent, modelFile.nameWithoutExtension + ".mel_params.json")
        melParams = if (jsonFile.exists()) {
            try {
                Gson().fromJson(jsonFile.readText(), MelParams::class.java).also {
                    Log.i(TAG, "Loaded mel params from ${jsonFile.name}: mean=${it.normMean}, std=${it.normStd}")
                }
            } catch (e: Exception) {
                Log.w(TAG, "Failed to parse mel params, using defaults: ${e.message}")
                MelParams()
            }
        } else {
            Log.i(TAG, "No mel_params.json found, using MuLan defaults")
            MelParams()
        }

        val effectiveFMax = melParams.fMax ?: (melParams.sampleRate / 2.0)
        melSpectrogram = MelSpectrogram(
            sampleRate = melParams.sampleRate.toInt(),
            nFft = melParams.nFft.toInt(),
            hopLength = melParams.hopLength.toInt(),
            nMels = melParams.nMels.toInt(),
            fMin = melParams.fMin.toFloat(),
            fMax = effectiveFMax.toFloat(),
        )

        // Load TFLite model via memory-mapped file (avoids loading ~1.2GB into heap)
        val modelBuffer = loadMappedModel(modelFile)
        val options = Interpreter.Options().apply {
            setNumThreads(4)
        }
        interpreter = Interpreter(modelBuffer, options)

        // Pre-allocate I/O buffers
        inputBuffer = ByteBuffer.allocateDirect(1 * N_MELS * EXPECTED_MEL_FRAMES * 4)
            .order(ByteOrder.nativeOrder())
        outputBuffer = ByteBuffer.allocateDirect(1 * EMBEDDING_DIM * 4)
            .order(ByteOrder.nativeOrder())

        Log.i(TAG, "MuQ-MuLan TFLite session loaded: ${modelFile.name} " +
                "(${modelFile.length() / 1024 / 1024}MB)")
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

        // Run each clip through TFLite and accumulate embeddings
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
        val countF = count.toFloat()
        for (i in 0 until EMBEDDING_DIM) {
            sumEmbedding[i] /= countF
        }
        l2Normalize(sumEmbedding)

        return sumEmbedding
    }

    /**
     * Run TFLite inference on a single 10s clip.
     * Computes mel spectrogram, normalizes, feeds to model.
     */
    private fun runInference(clip: FloatArray): FloatArray? {
        return try {
            // Compute raw power mel spectrogram
            val mel = melSpectrogram.compute(clip)

            // Apply dB conversion if required: 10 * log10(max(mel, 1e-10))
            if (melParams.isDb) {
                for (m in mel.indices) {
                    for (t in mel[m].indices) {
                        mel[m][t] = 10f * log10(maxOf(mel[m][t], 1e-10f))
                    }
                }
            }

            // Apply MuLan normalization: (mel - mean) / std
            val normMean = melParams.normMean.toFloat()
            val normStd = melParams.normStd.toFloat()
            for (m in mel.indices) {
                for (t in mel[m].indices) {
                    mel[m][t] = (mel[m][t] - normMean) / normStd
                }
            }

            // Fill input buffer: flatten [128, 1000] to [1, 128, 1000]
            inputBuffer.rewind()
            val floatInput = inputBuffer.asFloatBuffer()
            for (m in 0 until N_MELS) {
                val srcFrames = min(mel[m].size, EXPECTED_MEL_FRAMES)
                floatInput.put(mel[m], 0, srcFrames)
                // Zero-pad if mel is shorter than expected
                repeat(EXPECTED_MEL_FRAMES - srcFrames) { floatInput.put(0f) }
            }

            // Run TFLite inference
            outputBuffer.rewind()
            interpreter.run(inputBuffer, outputBuffer)

            // Read output [1, 512]
            outputBuffer.rewind()
            val output = FloatArray(EMBEDDING_DIM)
            outputBuffer.asFloatBuffer().get(output)

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
        interpreter.close()
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

/** Memory-map a model file for efficient loading without heap allocation. */
internal fun loadMappedModel(file: File): MappedByteBuffer {
    val stream = FileInputStream(file)
    val channel = stream.channel
    return channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size()).also {
        channel.close()
        stream.close()
    }
}
