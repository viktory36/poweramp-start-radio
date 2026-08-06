package com.powerampstartradio.indexing

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class AudioDecodeBoundaryContractTest {
    @Test
    fun `physical whole file follows decoded pcm order across gapless timestamp shifts`() {
        val authority = AudioDecodeBoundaryContract.timelineAuthority(
            requestedStartUs = 0L,
            requestedEndUs = null,
        )
        assertEquals(
            AudioDecodeBoundaryContract.TimelineAuthority.PHYSICAL_DECODE_ORDER,
            authority,
        )

        // Real MP3 failure shape: SkipCutBuffer released 1,007 held PCM frames on a callback
        // timestamped at 1,775, then the next callback was timestamped at 2,927. The bytes are
        // adjacent even though those input-derived timestamps imply a 145-sample hole.
        val firstStart = AudioDecodeBoundaryContract.outputBufferStartSample(
            authority = authority,
            reportedStartSourceSample = 1_775L,
            previousEndSourceSampleExclusive = null,
            requestedStartSourceSample = 0L,
        )
        val firstEnd = firstStart + 1_007L
        val secondStart = AudioDecodeBoundaryContract.outputBufferStartSample(
            authority = authority,
            reportedStartSourceSample = 2_927L,
            previousEndSourceSampleExclusive = firstEnd,
            requestedStartSourceSample = 0L,
        )

        assertEquals(0L, firstStart)
        assertEquals(1_007L, firstEnd)
        assertEquals(firstEnd, secondStart)
    }

    @Test
    fun `bounded and seeked spans retain presentation timestamp authority`() {
        val bounded = AudioDecodeBoundaryContract.timelineAuthority(
            requestedStartUs = 1_000_000L,
            requestedEndUs = 2_000_000L,
        )
        val seekedToEos = AudioDecodeBoundaryContract.timelineAuthority(
            requestedStartUs = 1_000_000L,
            requestedEndUs = null,
        )

        assertEquals(
            AudioDecodeBoundaryContract.TimelineAuthority.PRESENTATION_TIMESTAMPS,
            bounded,
        )
        assertEquals(
            AudioDecodeBoundaryContract.TimelineAuthority.PRESENTATION_TIMESTAMPS,
            seekedToEos,
        )
        assertEquals(
            2_927L,
            AudioDecodeBoundaryContract.outputBufferStartSample(
                authority = bounded,
                reportedStartSourceSample = 2_927L,
                previousEndSourceSampleExclusive = 2_782L,
                requestedStartSourceSample = 1_000L,
            ),
        )
    }

    @Test
    fun `requested boundary trims buffer overshoot exactly`() {
        val slice = AudioDecodeBoundaryContract.frameSlice(
            bufferStartSourceSample = 90L,
            bufferFrameCount = 50,
            requestedStartSourceSample = 100L,
            requestedEndSourceSampleExclusive = 120L,
        )

        assertEquals(10, slice.firstFrame)
        assertEquals(20, slice.frameCount)
        assertEquals(100L, slice.startSourceSample)
        assertEquals(120L, slice.endSourceSampleExclusive)
        val evidence = AudioDecodeBoundaryContract.requireComplete(
            requestedStartSourceSample = 100L,
            requestedEndSourceSampleExclusive = 120L,
            observedStartSourceSample = slice.startSourceSample,
            observedEndSourceSampleExclusive = slice.endSourceSampleExclusive,
            observedSourceSampleCount = slice.frameCount.toLong(),
            endOfStreamReached = false,
            requestedBoundaryReached = true,
        )
        assertTrue(evidence.requestedBoundaryReached)
        assertFalse(evidence.endOfStreamReached)
    }

    @Test
    fun `ordinary decode requires real codec eos`() {
        assertThrows(AudioDecoder.AudioDecodeBoundaryException::class.java) {
            AudioDecodeBoundaryContract.requireComplete(
                requestedStartSourceSample = 0L,
                requestedEndSourceSampleExclusive = null,
                observedStartSourceSample = 0L,
                observedEndSourceSampleExclusive = 1_000L,
                observedSourceSampleCount = 1_000L,
                endOfStreamReached = false,
                requestedBoundaryReached = false,
            )
        }

        val evidence = AudioDecodeBoundaryContract.requireComplete(
            requestedStartSourceSample = 0L,
            requestedEndSourceSampleExclusive = null,
            observedStartSourceSample = 0L,
            observedEndSourceSampleExclusive = 1_000L,
            observedSourceSampleCount = 1_000L,
            endOfStreamReached = true,
            requestedBoundaryReached = false,
        )
        assertTrue(evidence.endOfStreamReached)
    }

    @Test
    fun `early eos and requested boundary underflow fail closed`() {
        val error = assertThrows(AudioDecoder.AudioDecodeBoundaryException::class.java) {
            AudioDecodeBoundaryContract.requireComplete(
                requestedStartSourceSample = 100L,
                requestedEndSourceSampleExclusive = 200L,
                observedStartSourceSample = 100L,
                observedEndSourceSampleExclusive = 175L,
                observedSourceSampleCount = 75L,
                endOfStreamReached = true,
                requestedBoundaryReached = false,
            )
        }
        assertEquals(175L, error.evidence.observedEndSourceSampleExclusive)
        assertTrue(error.evidence.endOfStreamReached)
    }

    @Test
    fun `coordinate gaps and sample count disagreement fail closed`() {
        assertThrows(AudioDecoder.AudioDecodeBoundaryException::class.java) {
            AudioDecodeBoundaryContract.requireComplete(
                requestedStartSourceSample = 10L,
                requestedEndSourceSampleExclusive = 30L,
                observedStartSourceSample = 11L,
                observedEndSourceSampleExclusive = 30L,
                observedSourceSampleCount = 20L,
                endOfStreamReached = false,
                requestedBoundaryReached = true,
            )
        }
    }

    @Test
    fun `cooperative cancellation is typed`() {
        assertThrows(AudioDecoder.AudioDecodeCancelledException::class.java) {
            AudioDecodeBoundaryContract.requireActive { true }
        }
        AudioDecodeBoundaryContract.requireActive { false }
    }
}
