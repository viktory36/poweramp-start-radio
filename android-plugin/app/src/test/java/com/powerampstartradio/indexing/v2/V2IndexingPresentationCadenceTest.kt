package com.powerampstartradio.indexing.v2

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class V2IndexingPresentationCadenceTest {
    @Test
    fun `intermediate progress emits no more often than its interval`() {
        val cadence = V2IndexingProgressEventCadence(intermediateIntervalMs = 5_000L)

        assertTrue(cadence.shouldEmit(0L, 20L, 1_000L))
        assertFalse(cadence.shouldEmit(1L, 20L, 1_001L))
        assertFalse(cadence.shouldEmit(8L, 20L, 5_999L))
        assertTrue(cadence.shouldEmit(9L, 20L, 6_000L))
        assertFalse(cadence.shouldEmit(15L, 20L, 10_999L))
        assertTrue(cadence.shouldEmit(16L, 20L, 11_000L))
    }

    @Test
    fun `stage start and exact final progress always emit`() {
        val cadence = V2IndexingProgressEventCadence(intermediateIntervalMs = 5_000L)

        assertTrue(cadence.shouldEmit(0L, 2L, 10_000L))
        assertFalse(cadence.shouldEmit(1L, 2L, 10_001L))
        assertTrue(cadence.shouldEmit(2L, 2L, 10_002L))
    }
}
