package com.powerampstartradio.indexing

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
 * Two-phase GPU operation to avoid dual OpenCL environments on Adreno GPUs:
 * - Phase 1 (encoder): encodeTrack() runs the encoder on GPU for all chunks
 * - Between phases: closeEncoder() frees GPU, loadProjector() claims it
 * - Phase 2 (projector): projectAndAverage() runs the projector on GPU
 *
 * Models (converted from PyTorch via litert-torch):
 * - Encoder: mel [1,128,3000] + audio_times [1,750] -> hidden [1,750,1280]
 * - Projector: hidden [1,750,1280] -> projected [1,750,3584]
 *
 * @param encoderFile Path to the encoder .tflite model
 * @param projectorFile Path to the projector .tflite model (optional)
 * @param accelerator Hardware accelerator to use (GPU or CPU)
 */
class FlamingoInference(
    encoderFile: File,
    private val projectorFile: File? = null,
    private val accelerator: Accelerator = Accelerator.GPU,
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

        /** Floats per encoder hidden state chunk (750 × 1280). */
        const val HIDDEN_STATE_FLOATS = NUM_FRAMES * ENCODER_DIM
        /** Bytes per encoder hidden state chunk (3,840,000 bytes = 3.66 MB). */
        const val HIDDEN_STATE_BYTES = HIDDEN_STATE_FLOATS * 4
    }

    private val melSpectrogram = MelSpectrogram(center = true, melScale = MelScale.SLANEY)

    val outputDim: Int = if (projectorFile != null && projectorFile.exists()) PROJECTED_DIM else ENCODER_DIM

    /** Which accelerator is actually in use for the encoder. */
    val activeAccelerator: Accelerator

    // Encoder (loaded at init, released by closeEncoder)
    private var encoderModel: CompiledModel?
    private var encoderInputBuffers: List<TensorBuffer>?
    private var encoderOutputBuffers: List<TensorBuffer>?

    // Projector (loaded on demand by loadProjector)
    private var projectorModel: CompiledModel? = null
    private var projectorInputBuffers: List<TensorBuffer>? = null
    private var projectorOutputBuffers: List<TensorBuffer>? = null

    // Pre-allocated flat array for mel input
    private val melFlat = FloatArray(N_MELS * NUM_MEL_FRAMES)

    // Pre-allocated buffer for 30s chunk (reused across all chunks in a track)
    private val chunkBuffer = FloatArray(CHUNK_SAMPLES)

    init {
        val encResult = createReadyModel(encoderFile.absolutePath, accelerator)
        encoderModel = encResult.model
        activeAccelerator = encResult.accelerator
        encoderInputBuffers = encResult.inputBuffers
        encoderOutputBuffers = encResult.outputBuffers

        Log.i(TAG, "Flamingo encoder loaded: ${encoderFile.name} " +
                "(${encoderFile.length() / 1024 / 1024}MB), " +
                "projector=${projectorFile?.name ?: "none"}, " +
                "output_dim=$outputDim, accelerator=$activeAccelerator")
    }

    /**
     * Compute the number of inference calls (chunks) for audio of this duration.
     */
    fun chunkCount(durationS: Float): Int {
        if (durationS < 3.0f) return 0
        return calculateNumChunks(durationS)
    }

    // ── Phase 1: Encoder ──────────────────────────────────────────────

    /**
     * Encode all chunks for a track, streaming each result to [onChunkEncoded].
     *
     * Unlike the list-returning variant, this writes each chunk's hidden state
     * immediately via the callback and does NOT accumulate them in memory.
     * This keeps per-track peak memory at ~3.84MB (one chunk) instead of
     * numChunks × 3.84MB (up to 230MB for a 30-min track with 60 chunks).
     *
     * @return Number of successfully encoded chunks (0 = failure)
     */
    fun encodeTrackStreaming(
        audio: AudioDecoder.DecodedAudio,
        onChunkEncoded: (FloatArray) -> Unit,
        onChunkDone: (() -> Unit)? = null,
    ): Int {
        require(audio.sampleRate == SAMPLE_RATE) {
            "Flamingo requires ${SAMPLE_RATE}Hz audio, got ${audio.sampleRate}Hz"
        }

        if (audio.durationS < 3.0f) {
            Log.w(TAG, "Audio too short (${audio.durationS}s)")
            return 0
        }

        val numChunks = calculateNumChunks(audio.durationS)
        val positions = selectChunkPositions(audio.durationS, numChunks)

        Log.d(TAG, "Encoding ${positions.size} chunks from ${audio.durationS}s audio")

        var count = 0
        var totalMelMs = 0L
        var totalEncoderMs = 0L

        for (pos in positions) {
            val hidden = encodeChunk(audio.samples, pos) { melMs, encMs ->
                totalMelMs += melMs
                totalEncoderMs += encMs
            } ?: continue
            onChunkEncoded(hidden)
            count++
            onChunkDone?.invoke()
        }

        Log.i(TAG, "TIMING: flamingo_encode ${positions.size} chunks: mel=${totalMelMs}ms, " +
            "encoder=${totalEncoderMs}ms, total=${totalMelMs + totalEncoderMs}ms")

        return count
    }

    /**
     * Convenience wrapper: encode all chunks and return as a list.
     * Accumulates all hidden states in memory — only use for short tracks
     * or benchmarking. For production indexing, prefer [encodeTrackStreaming].
     */
    fun encodeTrack(
        audio: AudioDecoder.DecodedAudio,
        onChunkDone: (() -> Unit)? = null,
    ): List<FloatArray>? {
        val results = mutableListOf<FloatArray>()
        val count = encodeTrackStreaming(audio, { results.add(it) }, onChunkDone)
        return if (count > 0) results else null
    }

    /**
     * Encode a single chunk. Returns raw encoder output [NUM_FRAMES * ENCODER_DIM].
     */
    private fun encodeChunk(
        samples: FloatArray,
        positionS: Float,
        onTiming: ((melMs: Long, encoderMs: Long) -> Unit)? = null,
    ): FloatArray? {
        val enc = encoderModel ?: return null
        val encIn = encoderInputBuffers ?: return null
        val encOut = encoderOutputBuffers ?: return null

        // Copy audio into pre-allocated chunk buffer, zero-pad remainder if needed
        val startSample = (positionS * SAMPLE_RATE).toInt()
        val actualLen = min(CHUNK_SAMPLES, samples.size - startSample)
        System.arraycopy(samples, startSample, chunkBuffer, 0, actualLen)
        if (actualLen < CHUNK_SAMPLES) {
            chunkBuffer.fill(0f, actualLen, CHUNK_SAMPLES)
        }

        // Compute mel spectrogram and apply Whisper log normalization.
        // Center padding produces 3001 frames; the flatten loop's min() trims to 3000.
        val melStart = System.nanoTime()
        val rawMel = melSpectrogram.compute(chunkBuffer)
        MelSpectrogram.whisperNormalize(rawMel)
        val melMs = (System.nanoTime() - melStart) / 1_000_000

        // Flatten mel to pre-allocated melFlat
        for (m in 0 until N_MELS) {
            val rowOffset = m * NUM_MEL_FRAMES
            val srcFrames = min(rawMel[m].size, NUM_MEL_FRAMES)
            rawMel[m].copyInto(melFlat, rowOffset, 0, srcFrames)
            for (t in srcFrames until NUM_MEL_FRAMES) {
                melFlat[rowOffset + t] = 0f
            }
        }

        // Audio times: absolute timestamps per post-pool frame
        val audioTimes = FloatArray(NUM_FRAMES) { frame ->
            frame * FRAME_DURATION_S + positionS
        }

        return try {
            encIn[0].writeFloat(melFlat)
            encIn[1].writeFloat(audioTimes)

            val encStart = System.nanoTime()
            enc.run(encIn, encOut)
            val output = encOut[0].readFloat()  // GPU sync point — blocks until inference completes
            val encoderMs = (System.nanoTime() - encStart) / 1_000_000
            onTiming?.invoke(melMs, encoderMs)

            output
        } catch (e: Exception) {
            Log.e(TAG, "Encoder inference failed: ${e.message}", e)
            null
        }
    }

    /** Release encoder GPU resources to make room for the projector. */
    fun closeEncoder() {
        encoderInputBuffers?.forEach { it.close() }
        encoderOutputBuffers?.forEach { it.close() }
        encoderModel?.close()
        encoderModel = null
        encoderInputBuffers = null
        encoderOutputBuffers = null
        Log.i(TAG, "Encoder released")
    }

    // ── Phase 2: Projector ────────────────────────────────────────────

    /** Load projector on GPU (call after closeEncoder). */
    fun loadProjector() {
        if (projectorModel != null) return
        val pf = projectorFile ?: return
        if (!pf.exists()) return
        val projResult = createReadyModel(pf.absolutePath, accelerator)
        projectorModel = projResult.model
        projectorInputBuffers = projResult.inputBuffers
        projectorOutputBuffers = projResult.outputBuffers
        Log.i(TAG, "Projector loaded on ${projResult.accelerator}")
    }

    /**
     * Project encoder outputs and produce the final embedding, reading
     * chunks lazily via [readNextChunk] to avoid loading all hidden states
     * into memory simultaneously.
     *
     * Each chunk is [NUM_FRAMES * ENCODER_DIM] floats (3.84MB). For a 30-min
     * track (60 chunks), loading all at once would require 230MB. This method
     * keeps only one chunk in memory at a time (~3.84MB).
     *
     * @param numChunks Number of chunks to project
     * @param readNextChunk Callback that returns the next encoder hidden state
     * @param onChunkDone Optional callback invoked after each chunk is projected
     * @return L2-normalized embedding, or null if all chunks fail
     */
    fun projectAndAverageStreaming(
        numChunks: Int,
        readNextChunk: () -> FloatArray,
        onChunkDone: (() -> Unit)? = null,
    ): FloatArray? {
        val proj = projectorModel
        val projIn = projectorInputBuffers
        val projOut = projectorOutputBuffers

        val dim = if (proj != null) PROJECTED_DIM else ENCODER_DIM
        val sumEmbedding = FloatArray(dim)
        var count = 0
        var totalMs = 0L

        for (c in 0 until numChunks) {
            val hidden = readNextChunk()
            try {
                if (proj != null && projIn != null && projOut != null) {
                    projIn[0].writeFloat(hidden)
                    val start = System.nanoTime()
                    proj.run(projIn, projOut)
                    totalMs += (System.nanoTime() - start) / 1_000_000
                    val output = projOut[0].readFloat()

                    // Mean pool [750, 3584] → accumulate into [3584]
                    for (frame in 0 until NUM_FRAMES) {
                        val off = frame * PROJECTED_DIM
                        for (i in 0 until PROJECTED_DIM) {
                            sumEmbedding[i] += output[off + i]
                        }
                    }
                } else {
                    // No projector — mean pool encoder output [750, 1280]
                    for (frame in 0 until NUM_FRAMES) {
                        val off = frame * ENCODER_DIM
                        for (i in 0 until ENCODER_DIM) {
                            sumEmbedding[i] += hidden[off + i]
                        }
                    }
                }
                count++
                onChunkDone?.invoke()
            } catch (e: Exception) {
                Log.e(TAG, "Projector inference failed: ${e.message}", e)
            }
        }

        Log.i(TAG, "TIMING: flamingo_project $numChunks chunks: ${totalMs}ms")

        if (count == 0) return null

        val scale = 1f / (NUM_FRAMES.toFloat() * count)
        for (i in 0 until dim) {
            sumEmbedding[i] *= scale
        }
        l2Normalize(sumEmbedding)
        return sumEmbedding
    }

    /**
     * Convenience wrapper: project from a pre-built list.
     * For production indexing, prefer [projectAndAverageStreaming].
     */
    fun projectAndAverage(
        encoderOutputs: List<FloatArray>,
        onChunkDone: (() -> Unit)? = null,
    ): FloatArray? {
        val iter = encoderOutputs.iterator()
        return projectAndAverageStreaming(encoderOutputs.size, { iter.next() }, onChunkDone)
    }

    // ── Lifecycle ─────────────────────────────────────────────────────

    fun close() {
        closeEncoder()
        projectorInputBuffers?.forEach { it.close() }
        projectorOutputBuffers?.forEach { it.close() }
        projectorModel?.close()
        projectorModel = null
        projectorInputBuffers = null
        projectorOutputBuffers = null
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
}
