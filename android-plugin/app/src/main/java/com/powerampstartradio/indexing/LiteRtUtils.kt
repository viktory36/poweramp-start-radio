package com.powerampstartradio.indexing

import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer
import kotlin.math.sqrt

/** A fully initialized LiteRT model with pre-allocated I/O buffers. */
internal data class ReadyModel(
    val model: CompiledModel,
    val inputBuffers: List<TensorBuffer>,
    val outputBuffers: List<TensorBuffer>,
    val accelerator: Accelerator,
)

/**
 * Create a CompiledModel and allocate I/O buffers.
 *
 * Uses the requested accelerator (GPU or CPU). If model compilation
 * succeeds but buffer allocation fails (e.g. GPU OOM), the model is
 * properly closed before the exception propagates.
 */
internal fun createReadyModel(
    path: String,
    accelerator: Accelerator,
): ReadyModel {
    var model: CompiledModel? = null
    try {
        val options = CompiledModel.Options(accelerator).apply {
            if (accelerator == Accelerator.GPU) {
                // Force FP32 computation on GPU. The default (FP16) causes embedding
                // collapse in deep transformers (12+ layers): FP16 accumulation with
                // Flush-To-Zero on mobile GPUs loses discriminative signal, producing
                // near-identical embeddings regardless of input content.
                gpuOptions = CompiledModel.GpuOptions(
                    precision = CompiledModel.GpuOptions.Precision.FP32,
                )
            }
        }
        model = CompiledModel.create(path, options)
        val inputBuffers = model.createInputBuffers()
        val outputBuffers = model.createOutputBuffers()
        Log.i("LiteRT", "Model ready with $accelerator accelerator (precision: " +
            "${if (accelerator == Accelerator.GPU) "FP32" else "default"})")
        return ReadyModel(model, inputBuffers, outputBuffers, accelerator)
    } catch (e: Exception) {
        model?.close()
        throw e
    }
}

/** In-place L2 normalization of a float array. */
internal fun l2Normalize(arr: FloatArray) {
    var norm = 0f
    for (v in arr) norm += v * v
    norm = sqrt(norm)
    if (norm > 1e-10f) {
        for (i in arr.indices) arr[i] /= norm
    }
}
