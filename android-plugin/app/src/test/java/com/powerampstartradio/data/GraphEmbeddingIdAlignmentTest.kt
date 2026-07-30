package com.powerampstartradio.data

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class GraphEmbeddingIdAlignmentTest {
    @Test
    fun identicalOrderedIdsAreAligned() {
        assertNull(
            GraphEmbeddingIdAlignment.firstMismatch(
                longArrayOf(10L, 20L, 30L),
                longArrayOf(10L, 20L, 30L),
            )
        )
    }

    @Test
    fun sameIdsInDifferentOrderReportTheFirstDisagreement() {
        val mismatch = GraphEmbeddingIdAlignment.firstMismatch(
            longArrayOf(10L, 20L, 30L),
            longArrayOf(10L, 30L, 20L),
        )

        assertEquals(OrderedTrackIdMismatch(1, 20L, 30L), mismatch)
    }

    @Test
    fun shorterGraphReportsMissingGraphIdentity() {
        val mismatch = GraphEmbeddingIdAlignment.firstMismatch(
            longArrayOf(10L, 20L),
            longArrayOf(10L, 20L, 30L),
        )

        assertEquals(OrderedTrackIdMismatch(2, null, 30L), mismatch)
    }

    @Test
    fun shorterEmbeddingIndexReportsMissingEmbeddingIdentity() {
        val mismatch = GraphEmbeddingIdAlignment.firstMismatch(
            longArrayOf(10L, 20L, 30L),
            longArrayOf(10L, 20L),
        )

        assertEquals(OrderedTrackIdMismatch(2, 30L, null), mismatch)
    }
}
