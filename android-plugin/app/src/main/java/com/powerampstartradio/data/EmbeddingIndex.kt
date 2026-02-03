package com.powerampstartradio.data

import android.util.Log
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.PriorityQueue

/**
 * Memory-mapped embedding index for fast similarity search.
 *
 * Binary format (.emb):
 * ```
 * Offset 0:              magic "PEMB" (4 bytes)
 * Offset 4:              version     (uint32, little-endian)
 * Offset 8:              num_tracks  (uint32, little-endian)
 * Offset 12:             dim         (uint32, little-endian)
 * Offset 16:             track_ids   (int64[num_tracks], little-endian)
 * Offset 16 + N*8:       embeddings  (float32[num_tracks * dim], row-major, little-endian)
 * ```
 *
 * Uses FileChannel.map(READ_ONLY) so the OS pages data on demand (~4 KB pages).
 * Zero Java heap allocation — both indices can be "loaded" simultaneously since
 * they're virtual memory, not heap.
 */
class EmbeddingIndex private constructor(
    private val buffer: MappedByteBuffer,
    val numTracks: Int,
    val dim: Int
) {
    companion object {
        private const val TAG = "EmbeddingIndex"
        private const val MAGIC = 0x424D4550  // "PEMB" in little-endian
        private const val VERSION = 1
        private const val HEADER_SIZE = 16  // magic + version + num_tracks + dim

        /**
         * Memory-map an existing .emb file.
         */
        fun mmap(file: File): EmbeddingIndex {
            RandomAccessFile(file, "r").use { raf ->
                val channel = raf.channel
                val buf = channel.map(FileChannel.MapMode.READ_ONLY, 0, raf.length())
                buf.order(ByteOrder.LITTLE_ENDIAN)

                val magic = buf.getInt(0)
                require(magic == MAGIC) { "Invalid magic: ${Integer.toHexString(magic)}" }

                val version = buf.getInt(4)
                require(version == VERSION) { "Unsupported version: $version" }

                val numTracks = buf.getInt(8)
                val dim = buf.getInt(12)

                val expectedSize = HEADER_SIZE.toLong() + numTracks.toLong() * 8 + numTracks.toLong() * dim * 4
                require(raf.length() == expectedSize) {
                    "File size mismatch: expected $expectedSize, got ${raf.length()}"
                }

                return EmbeddingIndex(buf, numTracks, dim)
            }
        }

        /**
         * Extract embeddings from SQLite database into a .emb binary file.
         */
        fun extractFromDatabase(db: EmbeddingDatabase, model: EmbeddingModel, outFile: File) {
            Log.d(TAG, "Extracting ${model.name} embeddings to ${outFile.name}")

            val trackIds = mutableListOf<Long>()
            val embeddings = mutableListOf<FloatArray>()

            // Read all embeddings from the database
            val allEmbeddings = db.getAllEmbeddings(model)
            for ((trackId, embedding) in allEmbeddings) {
                if (embedding.size == model.dim) {
                    trackIds.add(trackId)
                    embeddings.add(embedding)
                } else {
                    Log.w(TAG, "Skipping track $trackId: dim ${embedding.size} != ${model.dim}")
                }
            }

            val numTracks = trackIds.size
            Log.d(TAG, "Writing $numTracks embeddings (dim=${model.dim})")

            // Write binary file
            val totalSize = HEADER_SIZE.toLong() + numTracks.toLong() * 8 + numTracks.toLong() * model.dim * 4
            RandomAccessFile(outFile, "rw").use { raf ->
                raf.setLength(totalSize)
                val channel = raf.channel
                val buf = channel.map(FileChannel.MapMode.READ_WRITE, 0, totalSize)
                buf.order(ByteOrder.LITTLE_ENDIAN)

                // Header
                buf.putInt(0, MAGIC)
                buf.putInt(4, VERSION)
                buf.putInt(8, numTracks)
                buf.putInt(12, model.dim)

                // Track IDs
                buf.position(HEADER_SIZE)
                for (trackId in trackIds) {
                    buf.putLong(trackId)
                }

                // Embeddings (row-major)
                for (embedding in embeddings) {
                    for (v in embedding) {
                        buf.putFloat(v)
                    }
                }

                buf.force()
            }

            Log.d(TAG, "Wrote ${outFile.length() / 1024 / 1024} MB to ${outFile.name}")
        }
    }

    // Precomputed offsets
    private val trackIdsOffset = HEADER_SIZE
    private val embeddingsOffset = HEADER_SIZE + numTracks.toLong() * 8

    /**
     * Build a lookup map from track ID to index for fast getEmbeddingByTrackId.
     * Lazily initialized on first use.
     */
    private val trackIdToIndex: Map<Long, Int> by lazy {
        val map = HashMap<Long, Int>(numTracks)
        for (i in 0 until numTracks) {
            map[getTrackId(i)] = i
        }
        map
    }

    /**
     * Get the track ID at a given index.
     */
    fun getTrackId(index: Int): Long {
        return buffer.getLong(trackIdsOffset + index * 8)
    }

    /**
     * Compute dot product between a query vector and the embedding at the given index.
     * Since embeddings are L2-normalized, this equals cosine similarity.
     */
    fun dotProduct(query: FloatArray, index: Int): Float {
        val offset = (embeddingsOffset + index.toLong() * dim * 4).toInt()
        var dot = 0f
        for (d in 0 until dim) {
            dot += query[d] * buffer.getFloat(offset + d * 4)
        }
        return dot
    }

    /**
     * Find the top-K most similar tracks to a query embedding.
     *
     * Uses a min-heap of size K for O(N log K) scan.
     */
    fun findTopK(
        query: FloatArray,
        topK: Int,
        excludeIds: Set<Long> = emptySet()
    ): List<Pair<Long, Float>> {
        // Min-heap: smallest similarity at top, so we can evict it when we find better
        val heap = PriorityQueue<Pair<Long, Float>>(topK + 1, compareBy { it.second })

        for (i in 0 until numTracks) {
            val trackId = getTrackId(i)
            if (trackId in excludeIds) continue

            val score = dotProduct(query, i)

            if (heap.size < topK) {
                heap.add(trackId to score)
            } else if (score > heap.peek()!!.second) {
                heap.poll()
                heap.add(trackId to score)
            }
        }

        // Return sorted descending by similarity
        return heap.sortedByDescending { it.second }
    }

    /**
     * Find top-K results for multiple queries in a single corpus scan.
     *
     * Each query maintains its own min-heap. One pass over all embeddings
     * serves all queries simultaneously — much faster than N separate findTopK calls.
     *
     * @param queries Map of anchor track ID to its query embedding
     * @param topK Number of results per query
     * @param excludeIds Track IDs to exclude from all results
     * @return Map of anchor track ID to its top-K results
     */
    fun findTopKMulti(
        queries: Map<Long, FloatArray>,
        topK: Int,
        excludeIds: Set<Long> = emptySet()
    ): Map<Long, List<Pair<Long, Float>>> {
        if (queries.isEmpty()) return emptyMap()

        // One min-heap per query
        val heaps = queries.mapValues {
            PriorityQueue<Pair<Long, Float>>(topK + 1, compareBy { it.second })
        }

        val queryEntries = queries.entries.toList()

        for (i in 0 until numTracks) {
            val trackId = getTrackId(i)
            if (trackId in excludeIds) continue

            // Compute embedding offset once per track
            val offset = (embeddingsOffset + i.toLong() * dim * 4).toInt()

            for ((anchorId, queryVec) in queryEntries) {
                // Skip: don't return the anchor itself as a result
                if (trackId == anchorId) continue

                var dot = 0f
                for (d in 0 until dim) {
                    dot += queryVec[d] * buffer.getFloat(offset + d * 4)
                }

                val heap = heaps[anchorId]!!
                if (heap.size < topK) {
                    heap.add(trackId to dot)
                } else if (dot > heap.peek()!!.second) {
                    heap.poll()
                    heap.add(trackId to dot)
                }
            }
        }

        return heaps.mapValues { (_, heap) ->
            heap.sortedByDescending { it.second }
        }
    }

    /**
     * Get the embedding for a specific track ID, or null if not found.
     */
    fun getEmbeddingByTrackId(trackId: Long): FloatArray? {
        val index = trackIdToIndex[trackId] ?: return null
        val offset = (embeddingsOffset + index.toLong() * dim * 4).toInt()
        val result = FloatArray(dim)
        for (d in 0 until dim) {
            result[d] = buffer.getFloat(offset + d * 4)
        }
        return result
    }
}
