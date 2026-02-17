package com.powerampstartradio.indexing

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtLoggingLevel
import ai.onnxruntime.OrtSession
import android.util.Log
import com.google.gson.Gson
import com.google.gson.annotations.SerializedName
import java.io.File
import java.nio.FloatBuffer
import kotlin.math.log10
import kotlin.math.min
import kotlin.math.sqrt

/**
 * MuQ-MuLan ONNX inference for on-device audio embedding generation.
 *
 * Supports the split model where mel spectrogram is computed on-device:
 * - Input: mel_features [1, 128, 1000] (normalized mel spectrogram from 10s @ 24kHz)
 * - Output: [1, 512] L2-normalized embedding
 *
 * Mel parameters are read from a JSON sidecar file (mulan_audio.mel_params.json)
 * generated during desktop ONNX export. Hardcoded defaults match MuQ-MuLan-large.
 *
 * Chunking strategy (matches desktop MuLanEmbeddingGenerator):
 * - 1 chunk (30s) per minute of audio, max 30
 * - Each 30s chunk -> 3x 10s clips (matching MuQ-MuLan's _get_all_clips)
 * - Each 10s clip: compute mel -> normalize -> ONNX -> 512d
 * - Average all clip embeddings, L2-normalize
 */
class MuLanInference(modelFile: File, cacheDir: File? = null) {

    companion object {
        private const val TAG = "MuLanInference"
        private const val SAMPLE_RATE = 24000
        private const val CLIP_DURATION_S = 10
        private const val CLIP_SAMPLES = SAMPLE_RATE * CLIP_DURATION_S  // 240000
        private const val CHUNK_DURATION_S = 30
        private const val MAX_CHUNKS = 30
        private const val EMBEDDING_DIM = 512
        private const val EXPECTED_MEL_FRAMES = 1000  // 240000 / 240 = 1000
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

    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val melSpectrogram: MelSpectrogram
    private val melParams: MelParams
    private val isSplitModel: Boolean

    /** Which execution provider is active ("QNN" or "CPU"). */
    var executionProvider: String = "CPU"
        private set

    /** QNN EP error message if it failed and fell back to CPU. Null = QNN not attempted or succeeded. */
    var qnnError: String? = null
        private set

    init {
        val opts = createSessionOptions(
            "mulan_audio.ctx.onnx", cacheDir ?: modelFile.parentFile
        )
        session = if (executionProvider == "QNN") {
            try {
                env.createSession(modelFile.absolutePath, opts).also {
                    Log.i(TAG, "MuLan session created with QNN EP")
                }
            } catch (e: Exception) {
                qnnError = e.message ?: e.toString()
                Log.e(TAG, "══════ QNN EP FAILED for MuLan ══════")
                Log.e(TAG, "Error: ${e.message}")
                Log.e(TAG, "Exception: ${e.javaClass.simpleName}")
                e.cause?.let { Log.e(TAG, "Root cause: ${it.message}") }
                Log.e(TAG, "Stack trace:", e)
                Log.e(TAG, "Falling back to CPU EP")
                executionProvider = "CPU"
                env.createSession(modelFile.absolutePath, OrtSession.SessionOptions())
            }
        } else {
            env.createSession(modelFile.absolutePath, opts)
        }

        // Detect model type from ONNX input name
        val inputName = session.inputInfo.keys.first()
        isSplitModel = inputName == "mel_features"

        // Load mel params from JSON sidecar if available, else use defaults
        melParams = if (isSplitModel) {
            val jsonFile = File(modelFile.parent, modelFile.nameWithoutExtension + ".mel_params.json")
            if (jsonFile.exists()) {
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
        } else {
            MelParams()  // Not used for legacy model
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

        Log.i(TAG, "MuQ-MuLan ONNX session loaded: ${modelFile.name} " +
                "(${if (isSplitModel) "split/mel" else "legacy/wav"}, ep=$executionProvider)")
    }

    /**
     * Create ORT session options with QNN EP (Hexagon HTP) if available.
     * Falls back to XNNPACK CPU automatically if QNN EP fails.
     */
    private fun createSessionOptions(contextFileName: String, cacheDir: File): OrtSession.SessionOptions {
        val opts = OrtSession.SessionOptions()
        try {
            opts.addQnn(mapOf(
                "backend_type" to "htp",
                "enable_htp_fp16_precision" to "1",
                "htp_performance_mode" to "high_performance",
                "htp_graph_finalization_optimization_mode" to "0",
                "profiling_level" to "basic",
            ))
            val cachePath = File(cacheDir, contextFileName).absolutePath
            opts.addConfigEntry("ep.context_enable", "1")
            opts.addConfigEntry("ep.context_embed_mode", "0")
            opts.addConfigEntry("ep.context_file_path", cachePath)
            // Prevent silent CPU fallback — forces ORT to throw if QNN can't handle the graph
            opts.addConfigEntry("session.disable_cpu_ep_fallback", "1")
            // Verbose logging to dump graph partitioning info to logcat
            opts.setSessionLogLevel(OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE)
            opts.setSessionLogVerbosityLevel(0)
            executionProvider = "QNN"
            Log.i(TAG, "QNN EP configured (HTP FP16), cache: $cachePath")
        } catch (e: Exception) {
            Log.w(TAG, "QNN EP not available, using XNNPACK CPU: ${e.message}")
        }
        return opts
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
            val embedding = if (isSplitModel) {
                runSplitInference(clip)
            } else {
                runLegacyInference(clip)
            } ?: continue
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
     * Split model inference: compute mel spectrogram on-device, feed to ONNX.
     */
    private fun runSplitInference(clip: FloatArray): FloatArray? {
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

            // Trim or pad to expected frame count (ONNX model expects exactly 1000 frames)
            val numFrames = EXPECTED_MEL_FRAMES
            val nMels = melParams.nMels.toInt()
            val melFlat = FloatArray(nMels * numFrames)
            for (m in 0 until nMels) {
                val srcFrames = min(mel[m].size, numFrames)
                System.arraycopy(mel[m], 0, melFlat, m * numFrames, srcFrames)
                // Remaining frames stay 0 (zero-padded) if mel is shorter
            }

            val melTensor = OnnxTensor.createTensor(
                env,
                FloatBuffer.wrap(melFlat),
                longArrayOf(1, nMels.toLong(), numFrames.toLong())
            )

            val results = session.run(mapOf("mel_features" to melTensor))
            val outputTensor = results[0] as OnnxTensor

            @Suppress("UNCHECKED_CAST")
            val output = (outputTensor.value as Array<FloatArray>)[0]

            melTensor.close()
            results.close()

            output
        } catch (e: Exception) {
            Log.e(TAG, "MuQ-MuLan split inference failed: ${e.message}")
            null
        }
    }

    /**
     * Legacy model inference: pass raw waveform directly to ONNX.
     */
    private fun runLegacyInference(clip: FloatArray): FloatArray? {
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
            Log.e(TAG, "MuQ-MuLan legacy inference failed: ${e.message}")
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
