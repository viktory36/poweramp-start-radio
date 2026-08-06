package com.powerampstartradio.indexing

import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * CLaMP3 text encoder LiteRT inference for text-to-music search.
 *
 * Uses a TFLite model (converted from PyTorch via litert-torch):
 * - Input 0: input_ids      [1, 128] INT64 (padded token IDs)
 * - Input 1: attention_mask  [1, 128] INT64 (1=valid, 0=pad)
 * - Output:  [1, 768] text embedding
 *
 * The text embedding lives in the same 768d space as CLaMP3 audio
 * embeddings, enabling cosine similarity search: text vs audio library.
 *
 * Uses the same XLM-RoBERTa Unigram tokenizer as the audio encoder.
 *
 * @param modelFile Path to the clamp3_text .tflite model file
 * @param tokenizerModelFile Path to the checkpoint's serialized SentencePiece model
 * @param accelerator Hardware accelerator to use (GPU or CPU)
 * @param strictAccelerator Refuse backend/precision fallback when deterministic replay requires it
 */
class Clamp3TextInference(
    modelFile: File,
    tokenizerModelFile: File,
    accelerator: Accelerator = Accelerator.GPU,
    strictAccelerator: Boolean = false,
) {
    companion object {
        private const val TAG = "Clamp3TextInference"
        private const val EMBEDDING_DIM = 768
        private const val SEQ_LEN = 128  // must match CLaMP3TextWrapper.SEQ_LEN
    }

    private val model: com.google.ai.edge.litert.CompiledModel
    private val tokenizer: OfficialSentencePieceTokenizer
    private val inputBuffers: List<com.google.ai.edge.litert.TensorBuffer>
    private val outputBuffers: List<com.google.ai.edge.litert.TensorBuffer>
    val activeAccelerator: Accelerator

    // Pre-allocated LongArray for writing INT64 input tensors [1, 128]
    private val longBuffer = LongArray(SEQ_LEN)

    init {
        tokenizer = OfficialSentencePieceTokenizer(tokenizerModelFile, seqLen = SEQ_LEN)

        // Text model fallback chain:
        // 1. GPU with FP32 precision (best quality, may fail on FP16-converted models)
        // 2. GPU with default/FP16 precision (single forward pass, FP16 acceptable)
        // 3. CPU (always works)
        val result = try {
            loadWithFallback(modelFile.absolutePath, accelerator, strictAccelerator)
        } catch (failure: Throwable) {
            tokenizer.close()
            throw failure
        }
        model = result.model
        activeAccelerator = result.accelerator
        inputBuffers = result.inputBuffers
        outputBuffers = result.outputBuffers

        Log.i(TAG, "CLaMP3 text encoder loaded: ${modelFile.name} " +
                "(${modelFile.length() / 1024 / 1024}MB), accelerator=$activeAccelerator")
    }

    private fun loadWithFallback(
        path: String,
        preferred: Accelerator,
        strictAccelerator: Boolean,
    ): ReadyModel {
        if (strictAccelerator) return createReadyModel(path, preferred)
        if (preferred == Accelerator.GPU) {
            // Try GPU with FP32 first
            try {
                return createReadyModel(path, Accelerator.GPU)
            } catch (e: Exception) {
                Log.w(TAG, "GPU+FP32 failed: ${e.message}")
            }

            // Try GPU with default (FP16) precision — acceptable for single-pass text
            try {
                val options = CompiledModel.Options(Accelerator.GPU)
                val model = CompiledModel.create(path, options)
                try {
                    val inputs = model.createInputBuffers()
                    val outputs = model.createOutputBuffers()
                    Log.i(TAG, "GPU+FP16 succeeded")
                    return ReadyModel(model, inputs, outputs, Accelerator.GPU)
                } catch (e2: Exception) {
                    model.close()
                    Log.w(TAG, "GPU+FP16 buffer alloc failed: ${e2.message}")
                }
            } catch (e: Exception) {
                Log.w(TAG, "GPU+FP16 compilation failed: ${e.message}")
            }
        }

        // CPU fallback
        try {
            return createReadyModel(path, Accelerator.CPU)
        } catch (e: Exception) {
            Log.w(TAG, "CPU with default options failed: ${e.message}")
        }

        // Last resort: CPU with no options at all
        val model = CompiledModel.create(path)
        val inputs = model.createInputBuffers()
        val outputs = model.createOutputBuffers()
        Log.i(TAG, "CPU (bare) succeeded")
        return ReadyModel(model, inputs, outputs, Accelerator.CPU)
    }

    /**
     * Generate a 768-dim text embedding from a query string.
     *
     * @param query Text query (e.g., "ethereal ambient", "heavy bass")
     * @param debugDir Optional directory to save the raw embedding for quality comparison.
     *                 Saves as `text_emb_<sanitized_query>.bin` (768 x float32 LE).
     * @return 768-dim L2-normalized embedding, or null on failure
     */
    fun generateEmbedding(query: String, debugDir: File? = null): FloatArray? {
        return try {
            val t0 = System.nanoTime()

            val segments = tokenizer.encodeSegments(query)

            val tokenMs = (System.nanoTime() - t0) / 1_000_000
            val inferStart = System.nanoTime()
            val weighted = FloatArray(EMBEDDING_DIM)
            var totalContributionTokens = 0
            segments.forEach { segment ->
                writeInt64Tensor(inputBuffers[0], segment.inputIds)
                writeInt64Tensor(inputBuffers[1], segment.attentionMask)
                model.run(inputBuffers, outputBuffers)
                val output = outputBuffers[0].readFloat()
                require(output.size >= EMBEDDING_DIM) {
                    "Unexpected text output size: ${output.size}"
                }
                val weight = segment.contributionTokenCount.toFloat()
                for (index in 0 until EMBEDDING_DIM) {
                    weighted[index] += output[index] * weight
                }
                totalContributionTokens += segment.contributionTokenCount
            }
            val inferMs = (System.nanoTime() - inferStart) / 1_000_000

            Log.i(
                TAG,
                "Text inference: tokenize=${tokenMs}ms, inference=${inferMs}ms, " +
                    "windows=${segments.size}, contributionTokens=$totalContributionTokens, " +
                    "total=${tokenMs + inferMs}ms, query='$query'",
            )

            require(totalContributionTokens > 0)
            for (index in weighted.indices) {
                weighted[index] /= totalContributionTokens.toFloat()
            }
            val embedding = weighted.also { l2Normalize(it) }

            // Save embedding to file for quality comparison (adb pull)
            if (debugDir != null) {
                try {
                    debugDir.mkdirs()
                    val safeName = query.replace(Regex("[^a-zA-Z0-9_-]"), "_").take(50)
                    val file = File(debugDir, "text_emb_${safeName}.bin")
                    val buf = ByteBuffer.allocate(EMBEDDING_DIM * 4).order(ByteOrder.LITTLE_ENDIAN)
                    for (v in embedding) buf.putFloat(v)
                    file.writeBytes(buf.array())
                    Log.i(TAG, "Saved embedding to ${file.absolutePath} (${file.length()} bytes)")
                } catch (e: Exception) {
                    Log.w(TAG, "Failed to save debug embedding: ${e.message}")
                }
            }

            embedding
        } catch (e: Exception) {
            Log.e(TAG, "Text inference failed: ${e.message}", e)
            null
        }
    }

    /**
     * Write an IntArray as INT64 values to a TensorBuffer.
     * TFLite text model expects INT64 inputs for token IDs and attention mask.
     */
    private fun writeInt64Tensor(buffer: com.google.ai.edge.litert.TensorBuffer, values: IntArray) {
        for (i in values.indices) {
            longBuffer[i] = values[i].toLong()
        }
        buffer.writeLong(longBuffer)
    }

    @Synchronized
    fun close() {
        if (closed) return
        closed = true
        tokenizer.close()
        inputBuffers.forEach { it.close() }
        outputBuffers.forEach { it.close() }
        model.close()
    }

    @Volatile
    private var closed = false
}
