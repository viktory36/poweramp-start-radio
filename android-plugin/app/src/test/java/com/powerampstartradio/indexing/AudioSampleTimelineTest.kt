package com.powerampstartradio.indexing

import org.junit.Assert.assertEquals
import org.junit.Test

class AudioSampleTimelineTest {
    @Test
    fun `sub-second cue span uses independent absolute boundaries`() {
        val startUs = 1_200_666_000L
        val durationUs = 209_334_000L

        val sourceSamples = AudioSampleTimeline.spanSampleCount(
            startUs = startUs,
            durationUs = durationUs,
            sampleRate = 44_100,
        )

        assertEquals(9_231_629L, sourceSamples)
        assertEquals(
            5_024_016L,
            AudioSampleTimeline.resampledSampleCount(sourceSamples, 44_100, 24_000),
        )
    }

    @Test
    fun `sample-integral ten second span is unchanged by offset phase`() {
        assertEquals(
            441_000L,
            AudioSampleTimeline.spanSampleCount(
                startUs = 1_200_666_000L,
                durationUs = 10_000_000L,
                sampleRate = 44_100,
            ),
        )
    }

    @Test
    fun `timestamp rounding recovers codec sample coordinate`() {
        val sampleIndex = 52_949_371L
        val roundedTimestampUs = sampleIndex * 1_000_000L / 44_100L

        assertEquals(
            sampleIndex,
            AudioSampleTimeline.nearestSampleForTimestamp(roundedTimestampUs, 44_100),
        )
    }

    @Test
    fun `one second is the first eligible padded MERT tail`() {
        assertEquals(0, MertWindowPolicy.windowCount(23_999))
        assertEquals(1, MertWindowPolicy.windowCount(24_000))
        assertEquals(1, MertWindowPolicy.windowCount(119_999))
        assertEquals(1, MertWindowPolicy.windowCount(120_000))
        assertEquals(1, MertWindowPolicy.windowCount(120_001))
        assertEquals(1, MertWindowPolicy.windowCount(143_999))
        assertEquals(2, MertWindowPolicy.windowCount(144_000))
    }
}
