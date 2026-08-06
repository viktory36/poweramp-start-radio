package com.powerampstartradio.indexing.v2

import com.powerampstartradio.data.EmbeddingIndex
import java.io.File
import java.io.FileOutputStream
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max

data class V2GraphDeletionRepairResult(
    val graphFile: File,
    val affectedRowCount: Int,
    val preservedRowCount: Int,
)

internal data class V2RepairGraphTopology(
    val trackIds: LongArray,
    val neighborIndices: Array<IntArray>,
    val weights: Array<FloatArray>,
    val neighborsPerNode: Int,
)

internal data class V2RepairScoredNeighbor(
    val retainedIndex: Int,
    val score: Float,
)

internal data class V2RepairGraphPlan(
    val topology: V2RepairGraphTopology,
    val affectedRowCount: Int,
    val preservedRowCount: Int,
)

/** Exact deletion update conditional on the active base generation's bound graph contract. */
object V2GraphDeletionRepairer {
    fun repairOrNull(
        baseGraphFile: File,
        retainedEmbeddingFile: File,
        retainedEmbeddingBinding: V2OrderedEmbeddingBinding,
        target: File,
        onProgress: (String) -> Unit = {},
    ): V2GraphDeletionRepairResult? {
        target.delete()
        return try {
            val baseBinding = V2GraphGenerationFile.inspect(baseGraphFile)
            val base = readGraph(baseGraphFile)
            require(baseBinding.nodeCount == base.trackIds.size &&
                baseBinding.neighborsPerNode == base.neighborsPerNode
            ) { "Base graph shape changed during deletion repair" }
            val retainedIds = readEmbeddingIds(retainedEmbeddingFile, retainedEmbeddingBinding)
            var retainedIndex: EmbeddingIndex? = null
            val plan = repairTopology(
                base = base,
                retainedTrackIds = retainedIds,
                exactTopK = { row, k ->
                    val index = retainedIndex ?: EmbeddingIndex.mmap(retainedEmbeddingFile).also {
                        retainedIndex = it
                    }
                    val trackId = retainedIds[row]
                    index.findTopK(
                        query = index.getEmbedding(row),
                        topK = k,
                        excludeIds = setOf(trackId),
                    ).map { (neighborId, score) ->
                        val neighborIndex = retainedIds.binarySearch(neighborId)
                        require(neighborIndex >= 0) {
                            "Exact graph scan returned a track outside the retained PEMB"
                        }
                        V2RepairScoredNeighbor(neighborIndex, score)
                    }
                },
                onAffectedProgress = { completed, total ->
                    onProgress("Updating similarity graph \u00b7 $completed of $total tracks")
                },
            )
            if (plan == null) {
                onProgress("The similarity graph will be rebuilt when next needed.")
                return null
            }
            writeGraph(target, plan.topology)
            val repairedBinding = V2GraphGenerationFile.inspect(target)
            require(repairedBinding.nodeCount == retainedEmbeddingBinding.trackCount &&
                repairedBinding.orderedTrackSetSha256 ==
                retainedEmbeddingBinding.orderedTrackSetSha256
            ) { "Repaired graph is not bound to the retained PEMB" }
            V2GraphDeletionRepairResult(
                graphFile = target,
                affectedRowCount = plan.affectedRowCount,
                preservedRowCount = plan.preservedRowCount,
            )
        } catch (_: LinkageError) {
            target.delete()
            onProgress("The similarity graph will be rebuilt when next needed.")
            null
        } catch (_: Exception) {
            target.delete()
            onProgress("The similarity graph will be rebuilt when next needed.")
            null
        }
    }

    internal fun repairTopology(
        base: V2RepairGraphTopology,
        retainedTrackIds: LongArray,
        exactTopK: (retainedRow: Int, k: Int) -> List<V2RepairScoredNeighbor>,
        onAffectedProgress: (completed: Int, total: Int) -> Unit = { _, _ -> },
    ): V2RepairGraphPlan? {
        return try {
            requireValidTopologyShape(base)
            require(retainedTrackIds.isNotEmpty() && retainedTrackIds.isStrictlyIncreasing()) {
                "Retained graph IDs must be non-empty and strictly increasing"
            }
            val k = base.neighborsPerNode
            if (retainedTrackIds.size <= k) return null

        val oldToNew = IntArray(base.trackIds.size) { -1 }
        var oldCursor = 0
        retainedTrackIds.forEachIndexed { newIndex, retainedId ->
            while (oldCursor < base.trackIds.size && base.trackIds[oldCursor] < retainedId) {
                oldCursor++
            }
            require(oldCursor < base.trackIds.size && base.trackIds[oldCursor] == retainedId) {
                "Retained PEMB IDs are not an ordered subset of the base graph"
            }
            oldToNew[oldCursor] = newIndex
            oldCursor++
        }
        val newToOld = IntArray(retainedTrackIds.size)
        oldToNew.forEachIndexed { oldIndex, newIndex ->
            if (newIndex >= 0) newToOld[newIndex] = oldIndex
        }

        val affected = BooleanArray(retainedTrackIds.size)
        var affectedCount = 0
        retainedTrackIds.indices.forEach { newRow ->
            val oldRow = newToOld[newRow]
            if (base.neighborIndices[oldRow].any { oldToNew[it] < 0 }) {
                affected[newRow] = true
                affectedCount++
            }
        }

        val repairedNeighbors = Array(retainedTrackIds.size) { IntArray(k) }
        val repairedWeights = Array(retainedTrackIds.size) { FloatArray(k) }
        var completedAffected = 0
        retainedTrackIds.indices.forEach { newRow ->
            val oldRow = newToOld[newRow]
            if (!affected[newRow]) {
                repeat(k) { slot ->
                    repairedNeighbors[newRow][slot] =
                        oldToNew[base.neighborIndices[oldRow][slot]]
                    repairedWeights[newRow][slot] = base.weights[oldRow][slot]
                }
                return@forEach
            }

            val candidates = exactTopK(newRow, k)
            require(candidates.size == k &&
                candidates.all { candidate ->
                    candidate.retainedIndex in retainedTrackIds.indices &&
                        candidate.retainedIndex != newRow &&
                        candidate.score.isFinite()
                } &&
                candidates.map(V2RepairScoredNeighbor::retainedIndex).distinct().size == k
            ) { "Exact deletion scan returned an invalid top-K row" }
            val ordered = candidates.sortedWith(
                compareByDescending<V2RepairScoredNeighbor> { it.score }
                    .thenBy { retainedTrackIds[it.retainedIndex] },
            )
            var positiveMass = 0.0
            ordered.forEach { candidate -> positiveMass += max(candidate.score, 0f).toDouble() }
            if (!positiveMass.isFinite() || positiveMass <= 0.0) return null
            ordered.forEachIndexed { slot, candidate ->
                repairedNeighbors[newRow][slot] = candidate.retainedIndex
                repairedWeights[newRow][slot] =
                    (max(candidate.score, 0f).toDouble() / positiveMass).toFloat()
            }
            completedAffected++
            onAffectedProgress(completedAffected, affectedCount)
        }

            V2RepairGraphPlan(
                topology = V2RepairGraphTopology(
                    trackIds = retainedTrackIds.copyOf(),
                    neighborIndices = repairedNeighbors,
                    weights = repairedWeights,
                    neighborsPerNode = k,
                ),
                affectedRowCount = affectedCount,
                preservedRowCount = retainedTrackIds.size - affectedCount,
            )
        } catch (_: IllegalArgumentException) {
            null
        }
    }

    private fun requireValidTopologyShape(topology: V2RepairGraphTopology) {
        val n = topology.trackIds.size
        val k = topology.neighborsPerNode
        require(n > 0 && k > 0 && topology.trackIds.isStrictlyIncreasing() &&
            topology.neighborIndices.size == n && topology.weights.size == n
        ) { "Invalid base graph shape" }
        repeat(n) { row ->
            val neighbors = topology.neighborIndices[row]
            val weights = topology.weights[row]
            require(neighbors.size == k && weights.size == k &&
                neighbors.all { it in 0 until n } &&
                neighbors.distinct().size == k &&
                neighbors.none { it == row } &&
                weights.all { it.isFinite() && it >= 0f } &&
                kotlin.math.abs(weights.sum().toDouble() - 1.0) <= ROW_SUM_TOLERANCE
            ) { "Invalid base graph row $row" }
        }
    }

    private fun readGraph(file: File): V2RepairGraphTopology = RandomAccessFile(file, "r").use { input ->
        val header = ByteArray(8).also(input::readFully)
        val headerBuffer = ByteBuffer.wrap(header).order(ByteOrder.LITTLE_ENDIAN)
        val n = headerBuffer.int
        val k = headerBuffer.int
        require(n > 0 && k > 0) { "Invalid base graph header" }
        val ids = LongArray(n) { java.lang.Long.reverseBytes(input.readLong()) }
        val neighbors = Array(n) { IntArray(k) }
        val weights = Array(n) { FloatArray(k) }
        repeat(n) { row ->
            repeat(k) { slot ->
                neighbors[row][slot] = Integer.reverseBytes(input.readInt())
                weights[row][slot] = Float.fromBits(Integer.reverseBytes(input.readInt()))
            }
        }
        require(input.filePointer == input.length()) { "Base graph has trailing bytes" }
        V2RepairGraphTopology(ids, neighbors, weights, k)
    }

    private fun readEmbeddingIds(
        file: File,
        expected: V2OrderedEmbeddingBinding,
    ): LongArray = RandomAccessFile(file, "r").use { input ->
        val header = ByteArray(16).also(input::readFully)
        val values = ByteBuffer.wrap(header).order(ByteOrder.LITTLE_ENDIAN)
        require(values.int == PEMB_MAGIC && values.int == PEMB_VERSION) {
            "Retained PEMB header is invalid"
        }
        val count = values.int
        val dimension = values.int
        require(count == expected.trackCount && dimension == expected.dimension &&
            file.length() == expected.byteLength
        ) { "Retained PEMB binding changed before graph repair" }
        LongArray(count) { java.lang.Long.reverseBytes(input.readLong()) }.also { ids ->
            require(ids.isStrictlyIncreasing()) { "Retained PEMB IDs are not ordered" }
        }
    }

    private fun writeGraph(file: File, topology: V2RepairGraphTopology) {
        val n = topology.trackIds.size
        val k = topology.neighborsPerNode
        val byteLength = Math.addExact(
            8,
            Math.addExact(
                Math.multiplyExact(n, Long.SIZE_BYTES),
                Math.multiplyExact(Math.multiplyExact(n, k), 8),
            ),
        )
        val bytes = ByteBuffer.allocate(byteLength).order(ByteOrder.LITTLE_ENDIAN)
            .putInt(n)
            .putInt(k)
        topology.trackIds.forEach(bytes::putLong)
        repeat(n) { row ->
            repeat(k) { slot ->
                bytes.putInt(topology.neighborIndices[row][slot])
                bytes.putFloat(topology.weights[row][slot])
            }
        }
        file.parentFile?.let { require(it.isDirectory || it.mkdirs()) }
        FileOutputStream(file).use { output ->
            output.write(bytes.array())
            output.flush()
            output.fd.sync()
        }
    }

    private fun LongArray.isStrictlyIncreasing(): Boolean {
        if (isEmpty()) return false
        var previous = Long.MIN_VALUE
        forEach { value ->
            if (value <= 0L || value <= previous) return false
            previous = value
        }
        return true
    }

    private const val PEMB_MAGIC = 0x424D4550
    private const val PEMB_VERSION = 1
    private const val ROW_SUM_TOLERANCE = 0.005
}
