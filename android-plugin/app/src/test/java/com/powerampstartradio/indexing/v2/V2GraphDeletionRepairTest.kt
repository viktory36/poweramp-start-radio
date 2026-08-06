package com.powerampstartradio.indexing.v2

import kotlin.math.abs
import kotlin.math.sqrt
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class V2GraphDeletionRepairTest {
    @Test
    fun repairedTopologyMatchesFullExactReferenceAndPreservesUnaffectedRows() {
        val ids = longArrayOf(10L, 20L, 30L, 40L, 50L, 60L)
        val vectors = arrayOf(
            normalized(1.00f, 0.05f),
            normalized(0.98f, 0.20f),
            normalized(0.94f, 0.34f),
            normalized(0.34f, 0.94f),
            normalized(0.20f, 0.98f),
            normalized(0.05f, 1.00f),
        )
        val base = fullExactTopology(ids, vectors, k = 2)
        val retainedIndices = listOf(0, 1, 3, 4, 5)
        val retainedIds = retainedIndices.map(ids::get).toLongArray()
        val retainedVectors = retainedIndices.map(vectors::get).toTypedArray()

        val repaired = V2GraphDeletionRepairer.repairTopology(
            base = base,
            retainedTrackIds = retainedIds,
            exactTopK = exactScanner(retainedIds, retainedVectors),
        )
        assertNotNull(repaired)
        val plan = requireNotNull(repaired)
        assertTrue(plan.affectedRowCount > 0)
        assertTrue(plan.preservedRowCount > 0)
        assertEquals(retainedIds.size, plan.affectedRowCount + plan.preservedRowCount)

        val exactReference = fullExactTopology(retainedIds, retainedVectors, k = 2)
        assertTopologyEquals(exactReference, plan.topology)

        val oldById = ids.indices.associateBy { ids[it] }
        val newById = retainedIds.indices.associateBy { retainedIds[it] }
        retainedIds.indices.forEach { newRow ->
            val oldRow = requireNotNull(oldById[retainedIds[newRow]])
            val rowWasAffected = base.neighborIndices[oldRow]
                .any { oldNeighbor -> ids[oldNeighbor] !in newById }
            if (!rowWasAffected) {
                repeat(base.neighborsPerNode) { slot ->
                    assertEquals(
                        base.weights[oldRow][slot].toBits(),
                        plan.topology.weights[newRow][slot].toBits(),
                    )
                }
            }
        }
    }

    @Test
    fun affectedTieUsesLowerTrackIdAndMatchesFullReference() {
        val ids = longArrayOf(10L, 20L, 30L, 40L, 50L)
        val vectors = arrayOf(
            normalized(1f, 0f),
            normalized(0.8f, 0.6f),
            normalized(0.8f, 0.6f),
            normalized(0.6f, 0.8f),
            normalized(0f, 1f),
        )
        val base = fullExactTopology(ids, vectors, k = 2)
        val retainedIndices = listOf(0, 1, 2, 4)
        val retainedIds = retainedIndices.map(ids::get).toLongArray()
        val retainedVectors = retainedIndices.map(vectors::get).toTypedArray()
        val repaired = requireNotNull(
            V2GraphDeletionRepairer.repairTopology(
                base = base,
                retainedTrackIds = retainedIds,
                exactTopK = exactScanner(retainedIds, retainedVectors),
            ),
        )
        assertTopologyEquals(
            fullExactTopology(retainedIds, retainedVectors, k = 2),
            repaired.topology,
        )
        val row50 = retainedIds.indexOf(50L)
        assertEquals(
            listOf(20L, 30L),
            repaired.topology.neighborIndices[row50].map(retainedIds::get),
        )
    }

    @Test
    fun impossibleShapeZeroMassAndInvalidBindingFallBackToAbsent() {
        val ids = longArrayOf(10L, 20L, 30L, 40L)
        val vectors = arrayOf(
            normalized(1f, 0.1f),
            normalized(0.9f, 0.3f),
            normalized(0.3f, 0.9f),
            normalized(0.1f, 1f),
        )
        val base = fullExactTopology(ids, vectors, k = 2)

        assertNull(
            V2GraphDeletionRepairer.repairTopology(
                base = base,
                retainedTrackIds = longArrayOf(10L, 20L),
                exactTopK = { _, _ -> error("scan must not run") },
            ),
        )
        assertNull(
            V2GraphDeletionRepairer.repairTopology(
                base = base,
                retainedTrackIds = longArrayOf(10L, 99L, 40L),
                exactTopK = { _, _ -> error("scan must not run") },
            ),
        )

        val retainedIds = longArrayOf(10L, 20L, 40L)
        assertNull(
            V2GraphDeletionRepairer.repairTopology(
                base = base,
                retainedTrackIds = retainedIds,
                exactTopK = { row, k ->
                    retainedIds.indices.filter { it != row }.take(k).map { neighbor ->
                        V2RepairScoredNeighbor(neighbor, -1f)
                    }
                },
            ),
        )

        val selfLinked = base.copy(
            neighborIndices = Array(base.trackIds.size) { row ->
                base.neighborIndices[row].copyOf().also { if (row == 0) it[0] = 0 }
            },
        )
        assertNull(
            V2GraphDeletionRepairer.repairTopology(
                base = selfLinked,
                retainedTrackIds = longArrayOf(10L, 20L, 40L),
                exactTopK = exactScanner(
                    longArrayOf(10L, 20L, 40L),
                    arrayOf(vectors[0], vectors[1], vectors[3]),
                ),
            ),
        )
    }

    private fun exactScanner(
        ids: LongArray,
        vectors: Array<FloatArray>,
    ): (Int, Int) -> List<V2RepairScoredNeighbor> = { row, k ->
        vectors.indices.asSequence()
            .filter { it != row }
            .map { index -> V2RepairScoredNeighbor(index, dot(vectors[row], vectors[index])) }
            .sortedWith(
                compareByDescending<V2RepairScoredNeighbor> { it.score }
                    .thenBy { ids[it.retainedIndex] },
            )
            .take(k)
            .toList()
    }

    private fun fullExactTopology(
        ids: LongArray,
        vectors: Array<FloatArray>,
        k: Int,
    ): V2RepairGraphTopology {
        val neighbors = Array(ids.size) { IntArray(k) }
        val weights = Array(ids.size) { FloatArray(k) }
        ids.indices.forEach { row ->
            val top = exactScanner(ids, vectors)(row, k)
            val mass = top.sumOf { candidate -> maxOf(candidate.score, 0f).toDouble() }
            require(mass > 0.0)
            top.forEachIndexed { slot, candidate ->
                neighbors[row][slot] = candidate.retainedIndex
                weights[row][slot] = (maxOf(candidate.score, 0f) / mass).toFloat()
            }
        }
        return V2RepairGraphTopology(ids.copyOf(), neighbors, weights, k)
    }

    private fun assertTopologyEquals(
        expected: V2RepairGraphTopology,
        actual: V2RepairGraphTopology,
    ) {
        assertTrue(expected.trackIds.contentEquals(actual.trackIds))
        assertEquals(expected.neighborsPerNode, actual.neighborsPerNode)
        expected.trackIds.indices.forEach { row ->
            assertTrue(expected.neighborIndices[row].contentEquals(actual.neighborIndices[row]))
            expected.weights[row].indices.forEach { slot ->
                assertTrue(
                    abs(expected.weights[row][slot] - actual.weights[row][slot]) <= 1e-6f,
                )
            }
        }
    }

    private fun normalized(x: Float, y: Float): FloatArray {
        val norm = sqrt(x * x + y * y)
        return floatArrayOf(x / norm, y / norm)
    }

    private fun dot(left: FloatArray, right: FloatArray): Float =
        left.indices.sumOf { index -> (left[index] * right[index]).toDouble() }.toFloat()
}
