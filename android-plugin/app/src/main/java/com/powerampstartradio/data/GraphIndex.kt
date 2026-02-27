package com.powerampstartradio.data

import android.util.Log
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * Memory-mapped kNN graph for random walk exploration.
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
            val ok = db.extractBinaryToFile("knn_graph", outFile)
            if (ok) Log.i(TAG, "Extracted graph: ${outFile.length() / 1024 / 1024} MB")
            else Log.w(TAG, "Failed to extract graph from database")
            return ok
        }
    }

    // Offset where graph data starts (after header + ID map)
    private val graphOffset = HEADER_SIZE.toLong() + numNodes.toLong() * 8

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
     * Sample a uniformly random neighbor for Monte Carlo random walks.
     *
     * Uniform selection lets the graph *topology* drive exploration.
     * Non-backtracking: [excludeId] prevents the walk from immediately
     * reversing direction (A→B→A oscillation wastes steps).
     *
     * Single-pass reservoir sampling: O(K) time, zero allocations.
     *
     * @param excludeId track ID to skip (previous node), or -1 to allow all
     * @return neighbor track ID, or -1 if the node has no valid neighbors
     */
    fun sampleNeighbor(trackId: Long, rand: java.util.Random, excludeId: Long = -1L): Long {
        val nodeIndex = trackIdToIndex[trackId] ?: return -1L
        val entryOffset = graphOffset + nodeIndex.toLong() * k * 8
        var selected = -1L
        var validCount = 0
        for (j in 0 until k) {
            val offset = (entryOffset + j * 8).toInt()
            val neighborIndex = buffer.getInt(offset)
            val weight = buffer.getFloat(offset + 4)
            if (neighborIndex !in 0 until numNodes || weight <= 0f) continue
            val neighborId = indexToTrackId[neighborIndex]
            if (neighborId == excludeId) continue
            validCount++
            if (rand.nextInt(validCount) == 0) {
                selected = neighborId
            }
        }
        return selected
    }

    /**
     * Check if a track exists in the graph.
     */
    fun hasTrack(trackId: Long): Boolean = trackId in trackIdToIndex

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
