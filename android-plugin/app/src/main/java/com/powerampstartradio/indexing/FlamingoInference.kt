package com.powerampstartradio.indexing

import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
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
 */
class FlamingoInference(encoderFile: File, projectorFile: File? = null) {

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

    private val encoderInterpreter: Interpreter
    private val projectorInterpreter: Interpreter?
    private val melSpectrogram = MelSpectrogram(center = true, melScale = MelScale.SLANEY)

    val outputDim: Int

    // Pre-allocated I/O buffers for encoder
    private val melBuffer: ByteBuffer
    private val timesBuffer: ByteBuffer
    private val encoderOutputBuffer: ByteBuffer

    // Pre-allocated I/O buffers for projector (if available)
    private val projectorOutputBuffer: ByteBuffer?

    init {
        // Load encoder model (memory-mapped)
        val encOptions = Interpreter.Options().apply { setNumThreads(4) }
        encoderInterpreter = Interpreter(loadMappedModel(encoderFile), encOptions)

        // Load projector model (if available)
        projectorInterpreter = if (projectorFile != null && projectorFile.exists()) {
            val projOptions = Interpreter.Options().apply { setNumThreads(4) }
            Interpreter(loadMappedModel(projectorFile), projOptions)
        } else null

        outputDim = if (projectorInterpreter != null) PROJECTED_DIM else ENCODER_DIM

        // Pre-allocate I/O buffers
        melBuffer = ByteBuffer.allocateDirect(1 * N_MELS * NUM_MEL_FRAMES * 4)
            .order(ByteOrder.nativeOrder())
        timesBuffer = ByteBuffer.allocateDirect(1 * NUM_FRAMES * 4)
            .order(ByteOrder.nativeOrder())
        encoderOutputBuffer = ByteBuffer.allocateDirect(1 * NUM_FRAMES * ENCODER_DIM * 4)
            .order(ByteOrder.nativeOrder())
        projectorOutputBuffer = if (projectorInterpreter != null) {
            ByteBuffer.allocateDirect(1 * NUM_FRAMES * PROJECTED_DIM * 4)
                .order(ByteOrder.nativeOrder())
        } else null

        Log.i(TAG, "Flamingo TFLite loaded: encoder=${encoderFile.name} " +
                "(${encoderFile.length() / 1024 / 1024}MB), " +
                "projector=${projectorFile?.name ?: "none"}, output_dim=$outputDim")
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
     * Process a single 30s chunk: compute mel, run TFLite, mean-pool time dimension.
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

        // Run TFLite inference
        return runInference(mel, audioTimes)
    }

    /**
     * Run TFLite inference: encoder -> [optional projector] -> mean pool.
     *
     * The TFLite encoder takes 2 inputs (mel + audio_times), no mask.
     * The mask was removed during TFLite export because we always use
     * full 30s chunks with no padding (mask would be all-ones).
     *
     * @param mel [128, numFrames] mel spectrogram
     * @param audioTimes [750] absolute timestamps
     * @return Mean-pooled embedding [outputDim], or null on failure
     */
    private fun runInference(mel: Array<FloatArray>, audioTimes: FloatArray): FloatArray? {
        return try {
            val numMelFrames = mel[0].size

            // Fill mel buffer: [1, 128, 3000]
            melBuffer.rewind()
            val melFloat = melBuffer.asFloatBuffer()
            for (m in 0 until N_MELS) {
                val srcFrames = min(mel[m].size, NUM_MEL_FRAMES)
                melFloat.put(mel[m], 0, srcFrames)
                repeat(NUM_MEL_FRAMES - srcFrames) { melFloat.put(0f) }
            }

            // Fill times buffer: [1, 750]
            timesBuffer.rewind()
            timesBuffer.asFloatBuffer().put(audioTimes)

            // Step 1: Encoder [1,128,3000] + [1,750] -> [1,750,1280]
            encoderOutputBuffer.rewind()
            val encInputs = arrayOf<Any>(melBuffer, timesBuffer)
            val encOutputs = HashMap<Int, Any>()
            encOutputs[0] = encoderOutputBuffer
            encoderInterpreter.runForMultipleInputsOutputs(encInputs, encOutputs)

            // Step 2: Projector (if available) [1,750,1280] -> [1,750,3584]
            val finalOutputBuffer: ByteBuffer
            val finalDim: Int
            if (projectorInterpreter != null && projectorOutputBuffer != null) {
                projectorOutputBuffer.rewind()
                encoderOutputBuffer.rewind()  // Rewind for reading by projector
                projectorInterpreter.run(encoderOutputBuffer, projectorOutputBuffer)
                finalOutputBuffer = projectorOutputBuffer
                finalDim = PROJECTED_DIM
            } else {
                finalOutputBuffer = encoderOutputBuffer
                finalDim = ENCODER_DIM
            }

            // Mean pool over time dimension: [1, 750, dim] -> [dim]
            finalOutputBuffer.rewind()
            val floatBuf = finalOutputBuffer.asFloatBuffer()
            val pooled = FloatArray(finalDim)
            for (frame in 0 until NUM_FRAMES) {
                for (i in 0 until finalDim) {
                    pooled[i] += floatBuf.get()
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
        encoderInterpreter.close()
        projectorInterpreter?.close()
    }
}
