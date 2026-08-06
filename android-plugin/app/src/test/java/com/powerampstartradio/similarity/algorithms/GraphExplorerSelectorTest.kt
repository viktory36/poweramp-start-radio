package com.powerampstartradio.similarity.algorithms

import com.powerampstartradio.data.UniformGraphSnapshot
import kotlinx.coroutines.CancellationException
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class GraphExplorerSelectorTest {
    @Test
    fun knownGraphHasExactTerminalProbabilitiesAndRouteEvidence() {
        val graph = graphOf(
            1L to longArrayOf(2L, 3L),
            2L to longArrayOf(4L),
            3L to longArrayOf(4L),
            4L to longArrayOf(),
        )

        val result = GraphExplorerSelector.compute(graph, 1L, stopProbability = 0.5f)
        val scores = result.ranking.associateBy { it.trackId }

        assertEquals(0.25, scores.getValue(2L).terminalProbability, 0.0)
        assertEquals(0.25, scores.getValue(3L).terminalProbability, 0.0)
        assertEquals(0.50, scores.getValue(4L).terminalProbability, 0.0)
        assertEquals(1.0, scores.getValue(2L).expectedRouteLinks, 0.0)
        assertEquals(2.0, scores.getValue(4L).expectedRouteLinks, 0.0)
        assertEquals(1.5, result.expectedRouteLinks, 0.0)
        assertEquals(1.0, result.totalTerminalProbability, 0.0)
        assertEquals(0.0, result.numericMassError, 0.0)
    }

    @Test
    fun conditionalRouteEvidenceCombinesShortAndLongPathsToTheSameTrack() {
        val graph = graphOf(
            1L to longArrayOf(2L, 4L),
            2L to longArrayOf(4L),
            4L to longArrayOf(),
        )

        val result = GraphExplorerSelector.compute(graph, 1L, stopProbability = 0.5f)
        val score = result.ranking.single { it.trackId == 4L }

        assertEquals(0.75, score.terminalProbability, 0.0)
        assertEquals(4.0 / 3.0, score.expectedRouteLinks, 1e-15)
    }

    @Test
    fun duplicateNeighborSlotsRetainUniformSlotMultiplicity() {
        val graph = graphOf(
            1L to longArrayOf(2L, 2L, 3L),
            2L to longArrayOf(),
            3L to longArrayOf(),
        )

        val result = GraphExplorerSelector.compute(graph, 1L, stopProbability = 1f)
        val scores = result.ranking.associateBy { it.trackId }

        assertEquals(2.0 / 3.0, scores.getValue(2L).terminalProbability, 0.0)
        assertEquals(1.0 / 3.0, scores.getValue(3L).terminalProbability, 0.0)
    }

    @Test
    fun immediatePreviousNodeIsTheOnlyExcludedContinuation() {
        val graph = graphOf(
            1L to longArrayOf(2L),
            2L to longArrayOf(1L, 3L),
            3L to longArrayOf(),
        )

        val result = GraphExplorerSelector.compute(graph, 1L, stopProbability = 0f)

        assertEquals(listOf(3L), result.ranking.map { it.trackId })
        assertEquals(1.0, result.ranking.single().terminalProbability, 0.0)
        assertEquals(2.0, result.ranking.single().expectedRouteLinks, 0.0)
    }

    @Test
    fun deadEndReceivesAllContinuationMass() {
        val graph = graphOf(
            1L to longArrayOf(2L),
            2L to longArrayOf(),
        )

        val result = GraphExplorerSelector.compute(graph, 1L, stopProbability = 0.2f)

        assertEquals(1.0, result.ranking.single().terminalProbability, 0.0)
        assertEquals(1.0, result.expectedRouteLinks, 0.0)
        assertEquals(1, result.evaluatedLinks)
    }

    @Test
    fun seedDeadEndConservesMassButReturnsNoTrack() {
        val graph = graphOf(9L to longArrayOf())

        val result = GraphExplorerSelector.compute(graph, 9L, stopProbability = 0.5f)

        assertTrue(result.ranking.isEmpty())
        assertEquals(1.0, result.excludedSeedProbability, 0.0)
        assertEquals(1.0, result.totalTerminalProbability, 0.0)
        assertEquals(0.0, result.expectedRouteLinks, 0.0)
        assertEquals(0, result.evaluatedLinks)
    }

    @Test
    fun alphaOneStopsAfterFirstLink() {
        val graph = graphOf(
            100L to longArrayOf(30L, 10L),
            30L to longArrayOf(100L),
            10L to longArrayOf(100L),
        )

        val result = GraphExplorerSelector.compute(graph, 100L, stopProbability = 1f)

        assertEquals(listOf(10L, 30L), result.ranking.map { it.trackId })
        assertTrue(result.ranking.all { it.terminalProbability == 0.5 })
        assertTrue(result.ranking.all { it.expectedRouteLinks == 1.0 })
        assertEquals(1, result.evaluatedLinks)
    }

    @Test
    fun alphaZeroAssignsResidualMassAtHundredLinkCap() {
        val graph = graphOf(
            1L to longArrayOf(2L),
            2L to longArrayOf(3L),
            3L to longArrayOf(1L),
        )

        val result = GraphExplorerSelector.compute(graph, 1L, stopProbability = 0f)

        assertEquals(listOf(2L), result.ranking.map { it.trackId })
        assertEquals(1.0, result.ranking.single().terminalProbability, 0.0)
        assertEquals(100.0, result.ranking.single().expectedRouteLinks, 0.0)
        assertEquals(100, result.evaluatedLinks)
    }

    @Test
    fun seedTerminalMassIsExcludedFromRankingButIncludedInProof() {
        val graph = graphOf(
            1L to longArrayOf(2L),
            2L to longArrayOf(3L),
            3L to longArrayOf(1L),
        )

        val result = GraphExplorerSelector.compute(graph, 1L, stopProbability = 0.5f)
        val returnedMass = result.ranking.sumOf { it.terminalProbability }

        assertTrue(result.excludedSeedProbability > 0.0)
        assertEquals(
            result.totalTerminalProbability,
            returnedMass + result.excludedSeedProbability,
            GraphExplorerSelector.MASS_TOLERANCE,
        )
        assertTrue(1L !in result.ranking.map { it.trackId })
    }

    @Test
    fun equalProbabilitiesUseCanonicalTrackIdTieBreak() {
        val graph = graphOf(
            50L to longArrayOf(90L, 10L, 70L),
            90L to longArrayOf(),
            10L to longArrayOf(),
            70L to longArrayOf(),
        )

        val result = GraphExplorerSelector.compute(graph, 50L, stopProbability = 1f)

        assertEquals(listOf(10L, 70L, 90L), result.ranking.map { it.trackId })
    }

    @Test
    fun largeDeterministicGraphConservesMassAtDeepSetting() {
        val nodeCount = 257
        val k = 5
        val ids = LongArray(nodeCount) { index -> 10_000L + index * 7L }
        val neighbors = IntArray(nodeCount * k)
        for (node in 0 until nodeCount) {
            for (slot in 0 until k) {
                neighbors[node * k + slot] =
                    (node * 37 + slot * 53 + slot * slot + 1) % nodeCount
            }
        }
        val graph = UniformGraphSnapshot.fromRaw(ids, neighbors, k)

        val result = GraphExplorerSelector.compute(graph, ids[19], stopProbability = 0.05f)

        assertEquals(1.0, result.totalTerminalProbability, GraphExplorerSelector.MASS_TOLERANCE)
        assertTrue(result.numericMassError <= GraphExplorerSelector.MASS_TOLERANCE)
        assertTrue(result.ranking.isNotEmpty())
        assertTrue(result.expectedRouteLinks > 1.0)
    }

    @Test
    fun callerCancellationIsObservedDuringPropagation() {
        val graph = graphOf(
            1L to longArrayOf(2L),
            2L to longArrayOf(3L),
            3L to longArrayOf(1L),
        )
        var checks = 0

        assertThrows(CancellationException::class.java) {
            GraphExplorerSelector.compute(graph, 1L, stopProbability = 0f) {
                checks++
                if (checks == 5) throw CancellationException("test cancellation")
            }
        }
        assertEquals(5, checks)
    }

    @Test
    fun repeatedRunsAreBitForBitDeterministic() {
        val graph = graphOf(
            8L to longArrayOf(5L, 3L),
            5L to longArrayOf(8L, 2L, 3L),
            3L to longArrayOf(8L, 2L, 5L),
            2L to longArrayOf(5L, 3L),
        )

        val first = GraphExplorerSelector.compute(graph, 8L, stopProbability = 0.17f)
        repeat(10) {
            assertEquals(first, GraphExplorerSelector.compute(graph, 8L, 0.17f))
        }
    }

    @Test
    fun rejectsInvalidAlphaAndUnknownSeed() {
        val graph = graphOf(1L to longArrayOf(2L), 2L to longArrayOf())

        assertThrows(IllegalArgumentException::class.java) {
            GraphExplorerSelector.compute(graph, 1L, Float.NaN)
        }
        assertThrows(IllegalArgumentException::class.java) {
            GraphExplorerSelector.compute(graph, 1L, -0.01f)
        }
        assertThrows(IllegalArgumentException::class.java) {
            GraphExplorerSelector.compute(graph, 99L, 0.5f)
        }
    }

    private fun graphOf(vararg rows: Pair<Long, LongArray>): UniformGraphSnapshot {
        val ids = LongArray(rows.size) { rows[it].first }
        val indexById = ids.withIndex().associate { it.value to it.index }
        val k = rows.maxOfOrNull { it.second.size }?.coerceAtLeast(1) ?: 1
        val neighbors = IntArray(ids.size * k) { -1 }
        rows.forEachIndexed { node, row ->
            row.second.forEachIndexed { slot, neighborId ->
                neighbors[node * k + slot] = requireNotNull(indexById[neighborId]) {
                    "test graph is missing neighbor $neighborId"
                }
            }
        }
        return UniformGraphSnapshot.fromRaw(ids, neighbors, k)
    }
}
