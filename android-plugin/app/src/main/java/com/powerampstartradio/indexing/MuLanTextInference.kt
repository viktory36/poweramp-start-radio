package com.powerampstartradio.indexing

import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer
import java.io.File

/**
 * MuQ-MuLan text tower LiteRT inference for text-to-music search.
 *
 * Uses a TFLite model (converted from PyTorch via litert-torch):
 * - Input 0: input_ids    [1, 64] INT64 (padded token IDs)
 * - Input 1: attention_mask [1, 64] INT64 (1=valid, 0=pad)
 * - Output: [1, 512] L2-normalized embedding
 *
 * The text embedding lives in the same 512d space as MuQ-MuLan audio
 * embeddings, enabling cosine similarity search: text vs audio library.
 *
 * @param modelFile Path to the mulan_text .tflite model file
 * @param vocabFile Path to the xlm_roberta_vocab.json file
 * @param accelerator Hardware accelerator to use (GPU or CPU)
 */
class MuLanTextInference(
    modelFile: File,
    vocabFile: File,
    accelerator: Accelerator = Accelerator.GPU,
) {
    companion object {
        private const val TAG = "MuLanTextInference"
        private const val EMBEDDING_DIM = 512
        private const val SEQ_LEN = 64  // must match MuLanTextWrapper.SEQ_LEN
    }

    private val model: CompiledModel
    private val tokenizer: SentencePieceTokenizer
    private val inputBuffers: List<TensorBuffer>
    private val outputBuffers: List<TensorBuffer>
    val activeAccelerator: Accelerator

    // Pre-allocated LongArray for writing INT64 input tensors [1, 64]
    private val longBuffer = LongArray(SEQ_LEN)

    init {
        tokenizer = SentencePieceTokenizer(vocabFile)

        val result = createReadyModel(modelFile.absolutePath, accelerator)
        model = result.model
        activeAccelerator = result.accelerator
        inputBuffers = result.inputBuffers
        outputBuffers = result.outputBuffers

        Log.i(TAG, "MuQ-MuLan text tower loaded: ${modelFile.name} " +
                "(${modelFile.length() / 1024 / 1024}MB), accelerator=$activeAccelerator")
    }

    /**
     * Generate a 512-dim text embedding from a query string.
     *
     * @param query Text query (e.g., "ethereal ambient", "heavy bass")
     * @return 512-dim L2-normalized embedding, or null on failure
     */
    fun generateEmbedding(query: String): FloatArray? {
        return try {
            val t0 = System.nanoTime()

            // Tokenize: text → (input_ids[64], attention_mask[64])
            val (inputIds, attentionMask) = tokenizer.encode(query)

            val tokenMs = (System.nanoTime() - t0) / 1_000_000

            // Write input_ids as INT64
            writeInt64Tensor(inputBuffers[0], inputIds)

            // Write attention_mask as INT64
            writeInt64Tensor(inputBuffers[1], attentionMask)

            // Run inference
            val inferStart = System.nanoTime()
            model.run(inputBuffers, outputBuffers)

            // Read output [1, 512]
            val output = outputBuffers[0].readFloat()
            val inferMs = (System.nanoTime() - inferStart) / 1_000_000

            Log.i(TAG, "Text inference: tokenize=${tokenMs}ms, inference=${inferMs}ms, " +
                    "total=${tokenMs + inferMs}ms, query='$query'")

            if (output.size >= EMBEDDING_DIM) {
                output.copyOf(EMBEDDING_DIM)
            } else {
                Log.w(TAG, "Unexpected output size: ${output.size}")
                null
            }
        } catch (e: Exception) {
            Log.e(TAG, "Text inference failed: ${e.message}", e)
            null
        }
    }

    /**
     * Write an IntArray as INT64 values to a TensorBuffer.
     * TFLite text model expects INT64 inputs for token IDs and attention mask.
     */
    private fun writeInt64Tensor(buffer: TensorBuffer, values: IntArray) {
        for (i in values.indices) {
            longBuffer[i] = values[i].toLong()
        }
        buffer.writeLong(longBuffer)
    }

    fun close() {
        inputBuffers.forEach { it.close() }
        outputBuffers.forEach { it.close() }
        model.close()
    }
}
