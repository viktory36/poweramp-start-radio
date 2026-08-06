package com.powerampstartradio.data

import com.powerampstartradio.similarity.algorithms.GraphExplorerSelector
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

class ActiveDomainGraphTopologyTest {
    @Test
    fun `all unique identities preserve the topology and walk exactly`() {
        val base = graph(
            ids = longArrayOf(10, 20, 30, 40),
            neighbors = arrayOf(
                longArrayOf(20, 30),
                longArrayOf(10, 40),
                longArrayOf(10, 40),
                longArrayOf(20, 30),
            ),
        )

        val identityGraph = ActiveDomainGraphTopologyBuilder.buildFromNodeSubset(
            base = base,
            orderedNodeTrackIds = base.ids.copyOf(),
            exactTopK = { trackId, _, _ ->
                error("Unique graph row $trackId should not be rescanned")
            },
        )

        assertEquals(0, identityGraph.evidence.affectedRowCount)
        assertEquals(base.ids.toList(), identityGraph.topology.ids.toList())
        assertEquals(base.neighbors.toList(), identityGraph.topology.neighbors.toList())
        assertEquals(
            GraphExplorerSelector.compute(base, seedTrackId = 10, stopProbability = 0.37f),
            GraphExplorerSelector.compute(
                identityGraph.topology,
                seedTrackId = 10,
                stopProbability = 0.37f,
            ),
        )
    }

    @Test
    fun `canonical identity nodes prevent duplicate occurrences from absorbing queue places`() {
        val occurrenceGraph = graph(
            ids = longArrayOf(10, 11, 20, 30, 40),
            neighbors = arrayOf(
                longArrayOf(11, 20),
                longArrayOf(10, 30),
                longArrayOf(10, 30),
                longArrayOf(20, 40),
                longArrayOf(30, 20),
            ),
        )

        val occurrenceWalk = GraphExplorerSelector.compute(
            occurrenceGraph,
            seedTrackId = 10,
            stopProbability = 1f,
        )
        val seedOccurrenceIds = setOf(10L, 11L)
        val oldVisibleQueue = occurrenceWalk.ranking
            .map { it.trackId }
            .filterNot(seedOccurrenceIds::contains)
            .take(2)
        assertEquals(listOf(20L), oldVisibleQueue)

        val identityGraph = ActiveDomainGraphTopologyBuilder.buildFromNodeSubset(
            base = occurrenceGraph,
            orderedNodeTrackIds = longArrayOf(10, 20, 30, 40),
            exactTopK = { trackId, k, exclusions ->
                assertEquals(10L, trackId)
                assertEquals(2, k)
                assertEquals(setOf(10L, 11L), exclusions)
                listOf(20L, 30L)
            },
        )
        val fixedWalk = GraphExplorerSelector.compute(
            identityGraph.topology,
            seedTrackId = 10,
            stopProbability = 1f,
        )

        assertEquals(listOf(10L, 20L, 30L, 40L), identityGraph.topology.ids.toList())
        assertEquals(listOf(20L, 30L), fixedWalk.ranking.take(2).map { it.trackId })
        assertEquals(1.0, fixedWalk.ranking.take(2).sumOf { it.terminalProbability }, 0.0)
    }

    @Test
    fun `removed copy nodes trigger exact refill while untouched rows are preserved`() {
        val base = graph(
            ids = longArrayOf(10, 20, 30, 40, 50, 60),
            neighbors = arrayOf(
                longArrayOf(20, 30),
                longArrayOf(10, 30),
                longArrayOf(10, 20),
                longArrayOf(30, 50),
                longArrayOf(40, 60),
                longArrayOf(50, 40),
            ),
        )
        val retained = longArrayOf(10, 30, 40, 50, 60)
        val exact = mapOf(
            10L to listOf(30L, 40L),
            30L to listOf(10L, 40L),
        )
        val exclusionsByQuery = linkedMapOf<Long, Set<Long>>()

        val result = ActiveDomainGraphTopologyBuilder.buildFromNodeSubset(
            base = base,
            orderedNodeTrackIds = retained,
            exactTopK = { trackId, k, exclusions ->
                exclusionsByQuery[trackId] = exclusions
                requireNotNull(exact[trackId]).also { assertEquals(k, it.size) }
            },
        )

        assertEquals(retained.toList(), result.topology.ids.toList())
        assertEquals(listOf(30L, 40L), result.topology.neighborIds(10))
        assertEquals(listOf(10L, 40L), result.topology.neighborIds(30))
        assertEquals(listOf(30L, 50L), result.topology.neighborIds(40))
        assertEquals(listOf(40L, 60L), result.topology.neighborIds(50))
        assertEquals(listOf(50L, 40L), result.topology.neighborIds(60))
        assertEquals(setOf(10L, 20L), exclusionsByQuery.getValue(10))
        assertEquals(setOf(20L, 30L), exclusionsByQuery.getValue(30))
        assertEquals(2, result.evidence.affectedRowCount)
        assertEquals(3, result.evidence.preservedRowCount)
        assertEquals(2, result.evidence.invalidatedSlotCount)
    }

    @Test
    fun `repeated and self slots are repaired to distinct non-self neighbors`() {
        val base = graph(
            ids = longArrayOf(10, 20, 30, 40),
            neighbors = arrayOf(
                longArrayOf(20, 20),
                longArrayOf(20, 30),
                longArrayOf(10, 40),
                longArrayOf(10, 30),
            ),
        )

        val result = ActiveDomainGraphTopologyBuilder.buildFromNodeSubset(
            base = base,
            orderedNodeTrackIds = base.ids.copyOf(),
            exactTopK = { trackId, _, _ ->
                when (trackId) {
                    10L -> listOf(20L, 30L)
                    20L -> listOf(10L, 30L)
                    else -> error("untouched row $trackId was rescanned")
                }
            },
        )

        assertEquals(listOf(20L, 30L), result.topology.neighborIds(10))
        assertEquals(listOf(10L, 30L), result.topology.neighborIds(20))
        assertEquals(2, result.evidence.affectedRowCount)
        assertEquals(2, result.evidence.invalidatedSlotCount)
    }

    @Test
    fun `exact refill fails closed on repeated self or excluded results`() {
        val base = graph(
            ids = longArrayOf(10, 20, 30, 40),
            neighbors = arrayOf(
                longArrayOf(20, 20),
                longArrayOf(10, 30),
                longArrayOf(10, 40),
                longArrayOf(10, 30),
            ),
        )

        assertThrows(IllegalArgumentException::class.java) {
            ActiveDomainGraphTopologyBuilder.buildFromNodeSubset(
                base = base,
                orderedNodeTrackIds = longArrayOf(10, 20, 30, 40),
                exactTopK = { _, _, _ -> listOf(20L, 20L) },
            )
        }
        assertThrows(IllegalArgumentException::class.java) {
            ActiveDomainGraphTopologyBuilder.buildFromNodeSubset(
                base = base,
                orderedNodeTrackIds = longArrayOf(10, 20, 30, 40),
                exactTopK = { _, _, _ -> listOf(20L, 99L) },
            )
        }
    }

    private fun graph(ids: LongArray, neighbors: Array<LongArray>): UniformGraphSnapshot {
        val indexById = ids.withIndex().associate { it.value to it.index }
        val k = neighbors.first().size
        require(neighbors.size == ids.size && neighbors.all { it.size == k })
        return UniformGraphSnapshot.fromRaw(
            trackIds = ids,
            neighborIndices = IntArray(ids.size * k) { offset ->
                val row = offset / k
                val slot = offset % k
                requireNotNull(indexById[neighbors[row][slot]])
            },
            neighborsPerNode = k,
        )
    }

    private fun UniformGraphSnapshot.neighborIds(trackId: Long): List<Long> {
        val row = indexOfTrackId(trackId)
        return (0 until neighborsPerNode).map { slot ->
            trackIdAt(neighbors[row * neighborsPerNode + slot])
        }
    }
}
