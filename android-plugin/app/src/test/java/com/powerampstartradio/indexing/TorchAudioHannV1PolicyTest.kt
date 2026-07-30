package com.powerampstartradio.indexing

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class TorchAudioHannV1PolicyTest {
    @Test
    fun `kernel shapes match TorchAudio default at production rates`() {
        assertEquals(
            TorchAudioHannV1Policy.KernelShape(
                outputPhases = 80,
                inputStride = 147,
                width = 12,
                kernelSize = 171,
            ),
            TorchAudioHannV1Policy.kernelShape(44_100, 24_000),
        )
        assertEquals(
            TorchAudioHannV1Policy.KernelShape(
                outputPhases = 1,
                inputStride = 2,
                width = 13,
                kernelSize = 28,
            ),
            TorchAudioHannV1Policy.kernelShape(48_000, 24_000),
        )
    }

    @Test
    fun `target length preserves TorchAudio float32 rounding including one-sample cases`() {
        val cases = listOf(
            Triple(19_556_776L, 44_100, 10_643_143L),
            Triple(19_724_591L, 44_100, 10_734_471L),
            Triple(10_452_143L, 44_100, 5_688_241L),
            Triple(16_212_626L, 44_100, 8_823_198L),
            Triple(6_300_000L, 48_000, 3_150_000L),
            Triple(33_564_540L, 48_000, 16_782_270L),
        )
        for ((inputSamples, fromRate, expected) in cases) {
            assertEquals(
                "$inputSamples @ $fromRate",
                expected,
                TorchAudioHannV1Policy.resampledLength(inputSamples, fromRate, 24_000),
            )
        }

        val exactRationalCeiling =
            (19_556_776L * 80L + 147L - 1L) / 147L
        assertEquals(10_643_144L, exactRationalCeiling)
        assertEquals(
            exactRationalCeiling - 1L,
            TorchAudioHannV1Policy.resampledLength(19_556_776L, 44_100, 24_000),
        )
    }

    @Test
    fun `awkward output boundaries map to exact source context`() {
        val totalInput = 50_003L
        assertEquals(
            TorchAudioHannV1Policy.InputRange(0L, 159L),
            TorchAudioHannV1Policy.requiredInputRange(
                totalInput, 44_100, 24_000, outputStartSample = 0L, outputSampleCount = 1,
            ),
        )
        assertEquals(
            TorchAudioHannV1Policy.InputRange(0L, 159L),
            TorchAudioHannV1Policy.requiredInputRange(
                totalInput, 44_100, 24_000, outputStartSample = 79L, outputSampleCount = 1,
            ),
        )
        assertEquals(
            TorchAudioHannV1Policy.InputRange(135L, 306L),
            TorchAudioHannV1Policy.requiredInputRange(
                totalInput, 44_100, 24_000, outputStartSample = 80L, outputSampleCount = 1,
            ),
        )

        val outputLength = TorchAudioHannV1Policy.resampledLength(
            totalInput, 44_100, 24_000,
        )
        val tail = TorchAudioHannV1Policy.requiredInputRange(
            totalInput,
            44_100,
            24_000,
            outputStartSample = outputLength - 3L,
            outputSampleCount = 3,
        )
        assertTrue(tail.start in 0 until totalInput)
        assertEquals(totalInput, tail.endExclusive)
    }

    @Test
    fun `same-rate policy is an exact slice`() {
        assertEquals(
            12_345L,
            TorchAudioHannV1Policy.resampledLength(12_345L, 24_000, 24_000),
        )
        assertEquals(
            TorchAudioHannV1Policy.InputRange(17L, 40L),
            TorchAudioHannV1Policy.requiredInputRange(
                totalInputSamples = 100L,
                fromRate = 24_000,
                toRate = 24_000,
                outputStartSample = 17L,
                outputSampleCount = 23,
            ),
        )
    }
}
