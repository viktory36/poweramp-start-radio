package com.powerampstartradio.data

import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.cos
import kotlin.math.sin

@RunWith(AndroidJUnit4::class)
class ActiveDomainGraphTopologyInstrumentedTest {
    @Test
    fun deletionRepairsExactlyTheAffectedRowsToTheFullActiveOnlyTopK() {
        val context = ApplicationProvider.getApplicationContext<android.content.Context>()
        val embeddingFile = File(context.cacheDir, "active-domain-repair.emb")
        val graphFile = File(context.cacheDir, "active-domain-repair.graph")
        val ids = longArrayOf(10L, 20L, 30L, 40L, 50L, 60L, 70L, 80L)
        val angles = doubleArrayOf(0.0, 0.10, 0.25, 0.45, 0.70, 1.00, 1.35, 1.75)
        val embeddings = Array(ids.size) { row ->
            floatArrayOf(cos(angles[row]).toFloat(), sin(angles[row]).toFloat())
        }
        val neighborsPerNode = 2
        val fullNeighbors = exactTopK(ids, embeddings, ids.toSet(), neighborsPerNode)
        writeEmbeddingIndex(embeddingFile, ids, embeddings)
        writeGraph(graphFile, ids, fullNeighbors, neighborsPerNode)

        try {
            val activeIds = ids.filter { it != INACTIVE_TRACK_ID }.toLongArray()
            val expectedActiveNeighbors = exactTopK(
                ids = ids,
                embeddings = embeddings,
                eligibleIds = activeIds.toSet(),
                neighborsPerNode = neighborsPerNode,
            )
            val repaired = ActiveDomainGraphTopologyBuilder.build(
                graph = GraphIndex.mmap(graphFile),
                embeddings = EmbeddingIndex.mmap(embeddingFile),
                orderedActiveTrackIds = activeIds,
            )

            assertEquals(activeIds.toList(), repaired.topology.ids.toList())
            assertEquals(2, repaired.evidence.affectedRowCount)
            assertEquals(5, repaired.evidence.preservedRowCount)
            assertEquals(2, repaired.evidence.invalidatedSlotCount)

            val activeIndexById = activeIds.withIndex().associate { it.value to it.index }
            activeIds.forEachIndexed { activeRow, trackId ->
                val oldNeighborIds = requireNotNull(fullNeighbors[trackId])
                val actualNeighborIds = repaired.topology.neighbors
                    .row(activeRow, neighborsPerNode)
                    .map(repaired.topology.ids::get)
                val expectedNeighborIds = requireNotNull(expectedActiveNeighbors[trackId])

                assertEquals(
                    "row $trackId must equal active-only exact top-K",
                    expectedNeighborIds,
                    actualNeighborIds,
                )
                assertTrue(actualNeighborIds.all { it in activeIndexById })
                assertFalse(INACTIVE_TRACK_ID in actualNeighborIds)

                if (INACTIVE_TRACK_ID !in oldNeighborIds) {
                    assertEquals(
                        "unaffected row $trackId must only be index-remapped",
                        oldNeighborIds,
                        actualNeighborIds,
                    )
                } else {
                    assertEquals(neighborsPerNode, actualNeighborIds.size)
                    assertTrue(
                        "affected row $trackId must replace its inactive slot",
                        actualNeighborIds.any { it !in oldNeighborIds },
                    )
                }
            }
        } finally {
            embeddingFile.delete()
            graphFile.delete()
        }
    }

    private fun exactTopK(
        ids: LongArray,
        embeddings: Array<FloatArray>,
        eligibleIds: Set<Long>,
        neighborsPerNode: Int,
    ): Map<Long, List<Long>> = ids.indices
        .filter { ids[it] in eligibleIds }
        .associate { queryRow ->
            ids[queryRow] to ids.indices
                .asSequence()
                .filter { candidateRow ->
                    candidateRow != queryRow && ids[candidateRow] in eligibleIds
                }
                .map { candidateRow ->
                    ids[candidateRow] to dot(embeddings[queryRow], embeddings[candidateRow])
                }
                .sortedWith(compareByDescending<Pair<Long, Float>> { it.second }.thenBy { it.first })
                .take(neighborsPerNode)
                .map { it.first }
                .toList()
        }

    private fun writeEmbeddingIndex(
        file: File,
        ids: LongArray,
        embeddings: Array<FloatArray>,
    ) {
        val dim = embeddings.first().size
        val size = 16 + ids.size * Long.SIZE_BYTES + ids.size * dim * Float.SIZE_BYTES
        val buffer = ByteBuffer.allocate(size).order(ByteOrder.LITTLE_ENDIAN)
        buffer.putInt(0x424D4550)
        buffer.putInt(1)
        buffer.putInt(ids.size)
        buffer.putInt(dim)
        ids.forEach(buffer::putLong)
        embeddings.forEach { row -> row.forEach(buffer::putFloat) }
        file.writeBytes(buffer.array())
    }

    private fun writeGraph(
        file: File,
        ids: LongArray,
        neighborIdsByTrack: Map<Long, List<Long>>,
        neighborsPerNode: Int,
    ) {
        val indexById = ids.withIndex().associate { it.value to it.index }
        val size = 8 + ids.size * Long.SIZE_BYTES +
            ids.size * neighborsPerNode * (Int.SIZE_BYTES + Float.SIZE_BYTES)
        val buffer = ByteBuffer.allocate(size).order(ByteOrder.LITTLE_ENDIAN)
        buffer.putInt(ids.size)
        buffer.putInt(neighborsPerNode)
        ids.forEach(buffer::putLong)
        ids.forEach { trackId ->
            requireNotNull(neighborIdsByTrack[trackId]).forEach { neighborId ->
                buffer.putInt(requireNotNull(indexById[neighborId]))
                buffer.putFloat(1f / neighborsPerNode)
            }
        }
        file.writeBytes(buffer.array())
    }

    private fun dot(left: FloatArray, right: FloatArray): Float {
        var result = 0f
        for (index in left.indices) result += left[index] * right[index]
        return result
    }

    private fun IntArray.row(row: Int, width: Int): List<Int> =
        (row * width until (row + 1) * width).map(::get)

    private companion object {
        const val INACTIVE_TRACK_ID = 40L
    }
}
