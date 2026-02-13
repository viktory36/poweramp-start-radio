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
 * Zero Java heap allocation for the embedding data.
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
         *
         * @param onProgress called periodically with (current, total) track counts
         */
        fun extractFromDatabase(
            db: EmbeddingDatabase,
            outFile: File,
            onProgress: ((current: Int, total: Int) -> Unit)? = null
        ) {
            Log.d(TAG, "Extracting fused embeddings to ${outFile.name}")

            val numTracks = db.getEmbeddingCount()
            if (numTracks == 0) {
                Log.w(TAG, "No embeddings to extract")
                return
            }

            val actualDim = db.getEmbeddingDim() ?: return
            val totalMB = numTracks.toLong() * actualDim * 4 / 1024 / 1024
            Log.i(TAG, "Extracting $numTracks embeddings (dim=$actualDim, ~${totalMB} MB)")

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
                db.forEachEmbeddingRaw { trackId, blob ->
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
                        Log.d(TAG, "Extract: $i / $numTracks ($pct%)")
                        onProgress?.invoke(i, numTracks)
                    }
                }

                onProgress?.invoke(i, numTracks)

                // Update header with actual count if some were skipped
                if (skipped > 0) {
                    Log.w(TAG, "Skipped $skipped embeddings with wrong dimension")
                    buf.putInt(8, i)
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
     * Find the single most similar track to a query embedding.
     *
     * @param cancellationCheck called every 10K tracks to allow coroutine cancellation
     */
    fun findTop1(
        query: FloatArray,
        excludeIds: Set<Long> = emptySet(),
        cancellationCheck: (() -> Unit)? = null
    ): Pair<Long, Float>? {
        var bestId = -1L
        var bestScore = Float.NEGATIVE_INFINITY
        for (i in 0 until numTracks) {
            if (i % 10000 == 0) cancellationCheck?.invoke()
            val trackId = getTrackId(i)
            if (trackId in excludeIds) continue
            val score = dotProduct(query, i)
            if (score > bestScore) {
                bestScore = score
                bestId = trackId
            }
        }
        return if (bestId >= 0) bestId to bestScore else null
    }

    /**
     * Find the top-K most similar tracks to a query embedding.
     *
     * Uses a min-heap of size K for O(N log K) scan.
     *
     * @param cancellationCheck called every 10K tracks to allow coroutine cancellation
     */
    fun findTopK(
        query: FloatArray,
        topK: Int,
        excludeIds: Set<Long> = emptySet(),
        cancellationCheck: (() -> Unit)? = null
    ): List<Pair<Long, Float>> {
        val heap = PriorityQueue<Pair<Long, Float>>(topK + 1, compareBy { it.second })

        for (i in 0 until numTracks) {
            if (i % 10000 == 0) cancellationCheck?.invoke()
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

        return heap.sortedByDescending { it.second }
    }

    /**
     * Compute similarity of every track to a reference vector in one sequential scan.
     * Returns a FloatArray indexed by internal track index (~300KB for 75K tracks).
     * Use with [rankFromSimilarities] for O(1)-amortized rank lookups.
     */
    fun computeAllSimilarities(reference: FloatArray): FloatArray {
        val sims = FloatArray(numTracks)
        for (i in 0 until numTracks) {
            sims[i] = dotProduct(reference, i)
        }
        return sims
    }

    /**
     * Compute 1-based rank of a target track from a precomputed similarity array.
     * Rank 1 = most similar in the corpus. Returns -1 if track not found.
     */
    fun rankFromSimilarities(sims: FloatArray, targetTrackId: Long): Int {
        val targetIdx = trackIdToIndex[targetTrackId] ?: return -1
        val targetSim = sims[targetIdx]
        var rank = 1
        for (i in sims.indices) {
            if (i != targetIdx && sims[i] > targetSim) rank++
        }
        return rank
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
