package com.powerampstartradio.indexing

import kotlin.math.ceil

/** Pure coordinate contract for the pinned desktop TorchAudio Hann preprocessing policy. */
internal object TorchAudioHannV1Policy {
    const val SPEC_ID = "torchaudio-hann-v1-width6-rolloff0.99-f32-target-length"
    const val FILTER_WIDTH = 6
    const val ROLLOFF = 0.99

    data class KernelShape(
        val outputPhases: Int,
        val inputStride: Int,
        val width: Int,
        val kernelSize: Int,
    )

    data class InputRange(
        val start: Long,
        val endExclusive: Long,
    )

    fun kernelShape(fromRate: Int, toRate: Int): KernelShape {
        require(fromRate > 0) { "fromRate must be positive" }
        require(toRate > 0) { "toRate must be positive" }
        val divisor = gcd(fromRate, toRate)
        val inputStride = fromRate / divisor
        val outputPhases = toRate / divisor
        val baseFrequency = minOf(inputStride, outputPhases) * ROLLOFF
        val width = ceil(FILTER_WIDTH * inputStride / baseFrequency).toInt()
        return KernelShape(
            outputPhases = outputPhases,
            inputStride = inputStride,
            width = width,
            kernelSize = inputStride + 2 * width,
        )
    }

    /** TorchAudio 2.10 uses `ceil(float32(new * length / old))`. */
    fun resampledLength(totalInputSamples: Long, fromRate: Int, toRate: Int): Long {
        require(totalInputSamples >= 0L) { "totalInputSamples must be non-negative" }
        if (fromRate == toRate) return totalInputSamples
        val shape = kernelShape(fromRate, toRate)
        val torchScalar = (
            shape.outputPhases.toDouble() * totalInputSamples.toDouble() /
                shape.inputStride.toDouble()
            ).toFloat()
        require(torchScalar.isFinite()) { "TorchAudio target length is not finite" }
        val result = ceil(torchScalar.toDouble())
        require(result <= Long.MAX_VALUE.toDouble()) { "TorchAudio target length overflows Long" }
        return result.toLong()
    }

    fun requiredInputRange(
        totalInputSamples: Long,
        fromRate: Int,
        toRate: Int,
        outputStartSample: Long,
        outputSampleCount: Int,
    ): InputRange {
        require(totalInputSamples >= 0L) { "totalInputSamples must be non-negative" }
        require(outputStartSample >= 0L) { "outputStartSample must be non-negative" }
        require(outputSampleCount >= 0) { "outputSampleCount must be non-negative" }
        val totalOutputSamples = resampledLength(totalInputSamples, fromRate, toRate)
        require(outputStartSample <= totalOutputSamples) {
            "outputStartSample exceeds resampled output length"
        }
        require(outputSampleCount.toLong() <= totalOutputSamples - outputStartSample) {
            "requested output range exceeds resampled output length"
        }
        if (outputSampleCount == 0) return InputRange(0L, 0L)
        if (fromRate == toRate) {
            return InputRange(
                start = outputStartSample,
                endExclusive = outputStartSample + outputSampleCount,
            )
        }

        val shape = kernelShape(fromRate, toRate)
        val outputEnd = outputStartSample + outputSampleCount
        val firstBlock = outputStartSample / shape.outputPhases
        val lastBlock = (outputEnd - 1L) / shape.outputPhases
        val start = (
            firstBlock * shape.inputStride - shape.width
            ).coerceAtLeast(0L)
        val end = (
            lastBlock * shape.inputStride - shape.width + shape.kernelSize
            ).coerceAtMost(totalInputSamples)
        return InputRange(start, end)
    }

    private fun gcd(a: Int, b: Int): Int {
        var left = a
        var right = b
        while (right != 0) {
            val remainder = left % right
            left = right
            right = remainder
        }
        return left
    }
}
