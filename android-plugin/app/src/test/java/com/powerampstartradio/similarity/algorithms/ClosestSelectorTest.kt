package com.powerampstartradio.similarity.algorithms

import org.junit.Assert.assertEquals
import org.junit.Test

class ClosestSelectorTest {
    @Test
    fun preservesCosineOrderAndOriginalRanks() {
        val candidates = listOf(10L to 0.9f, 20L to 0.8f, 30L to 0.7f)

        val selected = ClosestSelector.select(candidates, numSelect = 2)

        assertEquals(listOf(10L, 20L), selected.map { it.trackId })
        assertEquals(listOf(1, 2), selected.map { it.candidateRank })
    }

    @Test
    fun eligibilityIsPartOfSelectionAndRefillsTheRequest() {
        val candidates = listOf(10L to 0.9f, 20L to 0.8f, 30L to 0.7f, 40L to 0.6f)

        val selected = ClosestSelector.select(candidates, numSelect = 3) { id, _ -> id != 20L }

        assertEquals(listOf(10L, 30L, 40L), selected.map { it.trackId })
        assertEquals(listOf(1, 3, 4), selected.map { it.candidateRank })
    }

    @Test
    fun scansPastTheFormerCandidatePoolBoundaryToRefillExactly() {
        val candidates = (1L..140L).map { id -> id to (1f - id / 1_000f) }

        val selected = ClosestSelector.select(candidates, numSelect = 3) { id, _ -> id > 110L }

        assertEquals(listOf(111L, 112L, 113L), selected.map { it.trackId })
        assertEquals(listOf(111, 112, 113), selected.map { it.candidateRank })
    }

    @Test
    fun reconsidersRowsAfterAStatefulSpacingConstraintClears() {
        val candidates = listOf(
            1L to 1f,
            2L to 0.9f,
            3L to 0.8f,
            4L to 0.7f,
        )

        val selected = ClosestSelector.select(candidates, numSelect = 4) { id, selectedIds ->
            val sameArtist = id == 1L || id == 2L
            !sameArtist || selectedIds.lastOrNull() !in setOf(1L, 2L)
        }

        assertEquals(listOf(1L, 3L, 2L, 4L), selected.map { it.trackId })
        assertEquals(listOf(1, 3, 2, 4), selected.map { it.candidateRank })
    }
}
