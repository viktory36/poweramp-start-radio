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
     * Check if a track exists in the graph.
     */
    fun hasTrack(trackId: Long): Boolean = trackId in trackIdToIndex
}
