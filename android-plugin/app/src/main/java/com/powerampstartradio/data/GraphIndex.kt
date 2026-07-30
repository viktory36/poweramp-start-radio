package com.powerampstartradio.data

import android.util.Log
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * Memory-mapped kNN graph for deterministic graph exploration.
 *
 * Binary format (graph.bin):
 * ```
 * Header:     N (uint32), K (uint32)
 * ID map:     track_ids[N] (int64, little-endian)
 * Graph:      N * K entries of (neighbor_index uint32, weight float32)
 * ```
 *
 * neighbor_index values are indices into the ID map, not track IDs.
 * Weights are row-normalized transition probabilities (sum to 1 per node).
 */
class GraphIndex private constructor(
    private val buffer: MappedByteBuffer,
    val numNodes: Int,
    val k: Int,
    private val trackIdToIndex: Map<Long, Int>,
    private val indexToTrackId: LongArray
) {
    companion object {
        private const val TAG = "GraphIndex"
        private const val HEADER_SIZE = 8  // N (uint32) + K (uint32)

        /**
         * Memory-map a graph.bin file.
         */
        fun mmap(file: File): GraphIndex {
            RandomAccessFile(file, "r").use { raf ->
                val channel = raf.channel
                val buf = channel.map(FileChannel.MapMode.READ_ONLY, 0, raf.length())
                buf.order(ByteOrder.LITTLE_ENDIAN)

                val n = buf.getInt(0)
                val k = buf.getInt(4)

                // Read ID map
                val idMapOffset = HEADER_SIZE
                val indexToTrackId = LongArray(n)
                val trackIdToIndex = HashMap<Long, Int>(n)
                for (i in 0 until n) {
                    val tid = buf.getLong(idMapOffset + i * 8)
                    indexToTrackId[i] = tid
                    trackIdToIndex[tid] = i
                }

                val expectedSize = HEADER_SIZE.toLong() + n.toLong() * 8 + n.toLong() * k * 8
                require(raf.length() == expectedSize) {
                    "Graph file size mismatch: expected $expectedSize, got ${raf.length()}"
                }

                Log.i(TAG, "Graph: $n nodes, K=$k, ${raf.length() / 1024 / 1024} MB")
                return GraphIndex(buf, n, k, trackIdToIndex, indexToTrackId)
            }
        }

        /**
         * Read the node count (N) from a graph.bin file header without full mmap.
         * Returns -1 if the file is missing, too small, or unreadable.
         */
        fun readHeaderNodeCount(file: File): Int {
            if (!file.exists() || file.length() < HEADER_SIZE) return -1
            return try {
                java.io.RandomAccessFile(file, "r").use { raf ->
                    val buf = ByteArray(HEADER_SIZE)
                    raf.readFully(buf)
                    java.nio.ByteBuffer.wrap(buf).order(ByteOrder.LITTLE_ENDIAN).getInt(0)
                }
            } catch (_: Exception) { -1 }
        }

        /**
         * Extract graph binary from SQLite binary_data table and write to file.
         * Uses chunked reading to avoid Android's ~2 MB CursorWindow limit.
         */
        fun extractFromDatabase(db: EmbeddingDatabase, outFile: File): Boolean {
            if (!db.hasBinaryData("knn_graph")) return false
            val destination = outFile.absoluteFile
            destination.parentFile?.mkdirs()
            val temporary = File.createTempFile(
                ".${destination.name}.",
                ".tmp",
                destination.parentFile,
            )
            return try {
                if (!db.extractBinaryToFile("knn_graph", temporary)) {
                    Log.w(TAG, "Failed to extract graph from database")
                    false
                } else {
                    // Reopen and validate the complete binary before publication.
                    mmap(temporary)
                    EmbeddingIndex.replaceAtomically(temporary, destination)
                    Log.i(TAG, "Extracted graph: ${destination.length() / 1024 / 1024} MB")
                    true
                }
            } catch (e: Exception) {
                Log.w(TAG, "Rejected extracted graph: ${e.message}")
                false
            } finally {
                temporary.delete()
            }
        }
    }

    // Offset where graph data starts (after header + ID map)
    private val graphOffset = HEADER_SIZE.toLong() + numNodes.toLong() * 8

    @Volatile
    private var uniformGraphSnapshot: UniformGraphSnapshot? = null

    /**
     * Decode the mmap once into a compact immutable topology for deterministic traversal.
     *
     * The snapshot is about `N * K * 4` bytes (roughly 1.6 MB at 80,421 x 5) and is
     * cached for the lifetime of this graph generation. A graph slot is valid under the
     * same rules as [getNeighbors]: its index is in range and its stored weight is
     * positive. Traversal remains uniform over valid slots.
     */
    fun uniformTopology(): UniformGraphSnapshot {
        uniformGraphSnapshot?.let { return it }
        return synchronized(this) {
            uniformGraphSnapshot?.let { return@synchronized it }

            val neighbors = IntArray(numNodes * k) { -1 }
            for (node in 0 until numNodes) {
                val entryOffset = graphOffset + node.toLong() * k * 8
                val rowOffset = node * k
                for (slot in 0 until k) {
                    val offset = (entryOffset + slot * 8).toInt()
                    val neighborIndex = buffer.getInt(offset)
                    val weight = buffer.getFloat(offset + 4)
                    if (neighborIndex in 0 until numNodes && weight > 0f) {
                        neighbors[rowOffset + slot] = neighborIndex
                    }
                }
            }

            UniformGraphSnapshot(
                trackIds = indexToTrackId,
                neighborIndices = neighbors,
                neighborsPerNode = k,
                trackIdToIndex = trackIdToIndex,
                copyArrays = false,
            ).also { uniformGraphSnapshot = it }
        }
    }

    /**
     * Get the K nearest neighbors for a track, with transition probabilities.
     *
     * @return List of (trackId, weight) pairs, or empty if track not in graph
     */
    fun getNeighbors(trackId: Long): List<Pair<Long, Float>> {
        val nodeIndex = trackIdToIndex[trackId] ?: return emptyList()
        val result = mutableListOf<Pair<Long, Float>>()

        // Each entry is 8 bytes: neighbor_index (uint32) + weight (float32)
        val entryOffset = graphOffset + nodeIndex.toLong() * k * 8

        for (j in 0 until k) {
            val offset = (entryOffset + j * 8).toInt()
            val neighborIndex = buffer.getInt(offset)
            val weight = buffer.getFloat(offset + 4)

            if (neighborIndex in 0 until numNodes && weight > 0f) {
                result.add(indexToTrackId[neighborIndex] to weight)
            }
        }

        return result
    }

    /**
     * Check if a track exists in the graph.
     */
    fun hasTrack(trackId: Long): Boolean = trackId in trackIdToIndex

    /** Proves this graph uses the exact canonical row order of an embedding index. */
    fun hasSameOrderedTrackIds(embeddingIndex: EmbeddingIndex): Boolean =
        GraphEmbeddingIdAlignment.firstMismatch(
            graphCount = numNodes,
            graphTrackIdAt = indexToTrackId::get,
            embeddingCount = embeddingIndex.numTracks,
            embeddingTrackIdAt = embeddingIndex::getTrackId,
        ) == null

    /**
     * Compute shortest hop distance from a seed track to all reachable nodes via BFS.
     *
     * @return Map of trackId to hop count (seed itself is 0, direct neighbors are 1, etc.)
     */
    fun bfsFromSeed(seedTrackId: Long): Map<Long, Int> {
        val distances = HashMap<Long, Int>()
        distances[seedTrackId] = 0
        val queue = ArrayDeque<Long>()
        queue.add(seedTrackId)
        while (queue.isNotEmpty()) {
            val node = queue.removeFirst()
            val dist = distances[node]!!
            for ((neighborId, _) in getNeighbors(node)) {
                if (neighborId !in distances) {
                    distances[neighborId] = dist + 1
                    queue.add(neighborId)
                }
            }
        }
        return distances
    }
}
