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
         *
         * Streams rows one at a time — never holds more than one embedding in memory.
         * Each row's BLOB bytes are written directly to the mmap'd output at the
         * correct offset (track ID in the IDs section, embedding in the data section).
         *
         * @param onProgress called periodically with (current, total) track counts
         */
        fun extractFromDatabase(
            db: EmbeddingDatabase,
            model: EmbeddingModel,
            outFile: File,
            onProgress: ((current: Int, total: Int) -> Unit)? = null
        ) {
            Log.d(TAG, "Extracting ${model.name} embeddings to ${outFile.name}")

            val numTracks = db.getEmbeddingCount(model)
            if (numTracks == 0) {
                Log.w(TAG, "No ${model.name} embeddings to extract")
                return
            }

            // Detect actual dimension from the database (may differ from enum
            // if embeddings were reduced via SVD projection on the desktop side)
            val actualDim = db.getEmbeddingDim(model) ?: model.dim
            val totalMB = numTracks.toLong() * actualDim * 4 / 1024 / 1024
            Log.i(TAG, "Extracting $numTracks ${model.name} embeddings (dim=$actualDim, ~${totalMB} MB)")

            val expectedBlobSize = actualDim * 4  // float32
            val totalSize = HEADER_SIZE.toLong() + numTracks.toLong() * 8 + numTracks.toLong() * actualDim * 4
            val embeddingsStart = HEADER_SIZE.toLong() + numTracks.toLong() * 8

            RandomAccessFile(outFile, "rw").use { raf ->
                raf.setLength(totalSize)
                val channel = raf.channel
                val buf = channel.map(FileChannel.MapMode.READ_WRITE, 0, totalSize)
                buf.order(ByteOrder.LITTLE_ENDIAN)

                // Write header
                buf.putInt(0, MAGIC)
                buf.putInt(4, VERSION)
                buf.putInt(8, numTracks)
                buf.putInt(12, actualDim)

                // Stream rows: write track ID and embedding blob at computed offsets
                var i = 0
                var skipped = 0
                val progressInterval = maxOf(numTracks / 20, 1)  // ~5% increments
                db.forEachEmbeddingRaw(model) { trackId, blob ->
                    if (blob.size != expectedBlobSize) {
                        skipped++
                        return@forEachEmbeddingRaw
                    }
                    if (i >= numTracks) return@forEachEmbeddingRaw

                    // Write track ID
                    val idOffset = HEADER_SIZE + i * 8
                    buf.putLong(idOffset, trackId)

                    // Write embedding blob bytes directly (already little-endian float32)
                    val embOffset = (embeddingsStart + i.toLong() * expectedBlobSize).toInt()
                    buf.position(embOffset)
                    buf.put(blob)

                    i++

                    if (i % progressInterval == 0) {
                        val pct = i * 100 / numTracks
                        Log.d(TAG, "${model.name}: $i / $numTracks ($pct%)")
                        onProgress?.invoke(i, numTracks)
                    }
                }

                onProgress?.invoke(i, numTracks)

                // Update header with actual count if some were skipped
                if (skipped > 0) {
                    Log.w(TAG, "Skipped $skipped embeddings with wrong dimension")
                    buf.putInt(8, i)
                    // Truncate file to actual size
                    val actualSize = HEADER_SIZE.toLong() + i.toLong() * 8 + i.toLong() * expectedBlobSize
                    raf.setLength(actualSize)
                }

                buf.force()
            }

            Log.i(TAG, "Wrote ${outFile.length() / 1024 / 1024} MB to ${outFile.name}")
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
