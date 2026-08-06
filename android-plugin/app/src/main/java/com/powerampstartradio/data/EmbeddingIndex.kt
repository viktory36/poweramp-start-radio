package com.powerampstartradio.data

import android.util.Log
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.LongBuffer
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.nio.file.AtomicMoveNotSupportedException
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import com.powerampstartradio.indexing.NativeMath

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
        fun mmap(file: File, preload: Boolean = true): EmbeddingIndex {
            RandomAccessFile(file, "r").use { raf ->
                val length = raf.length()
                require(length in HEADER_SIZE.toLong()..Int.MAX_VALUE.toLong()) {
                    "Embedding index length is unsupported: $length"
                }
                val headerBytes = ByteArray(HEADER_SIZE)
                raf.readFully(headerBytes)
                val header = ByteBuffer.wrap(headerBytes).order(ByteOrder.LITTLE_ENDIAN)

                val magic = header.getInt(0)
                require(magic == MAGIC) { "Invalid magic: ${Integer.toHexString(magic)}" }

                val version = header.getInt(4)
                require(version == VERSION) { "Unsupported version: $version" }

                val numTracks = header.getInt(8)
                val dim = header.getInt(12)
                val expectedSize = expectedFileSize(numTracks, dim)
                require(expectedSize != null && length == expectedSize) {
                    "Invalid index shape or file size: tracks=$numTracks, dim=$dim, " +
                        "expected=$expectedSize, actual=$length"
                }

                val channel = raf.channel
                val buf = channel.map(FileChannel.MapMode.READ_ONLY, 0, length)
                buf.order(ByteOrder.LITTLE_ENDIAN)

                // A retrieval scans every embedding. Synchronous prefaulting is substantially
                // faster than taking one minor fault per 4 KiB page inside the dot-product loop.
                if (preload) buf.load()

                return EmbeddingIndex(buf, numTracks, dim)
            }
        }

        /**
         * Read the track count from an .emb file header without full mmap.
         * Returns -1 if the file is missing, too small, or has invalid magic.
         */
        fun readHeaderTrackCount(file: File): Int {
            if (!file.exists() || file.length() < HEADER_SIZE) return -1
            return try {
                RandomAccessFile(file, "r").use { raf ->
                    val buf = ByteArray(HEADER_SIZE)
                    raf.readFully(buf)
                    val bb = ByteBuffer.wrap(buf).order(ByteOrder.LITTLE_ENDIAN)
                    if (bb.getInt(0) != MAGIC) return -1
                    if (bb.getInt(4) != VERSION) return -1
                    val numTracks = bb.getInt(8)
                    val dim = bb.getInt(12)
                    val expectedSize = expectedFileSize(numTracks, dim) ?: return -1
                    if (raf.length() != expectedSize) return -1
                    numTracks
                }
            } catch (_: Exception) { -1 }
        }

        private fun expectedFileSize(numTracks: Int, dim: Int): Long? {
            if (numTracks < 0 || dim <= 0) return null
            return try {
                val idsBytes = Math.multiplyExact(numTracks.toLong(), Long.SIZE_BYTES.toLong())
                val values = Math.multiplyExact(numTracks.toLong(), dim.toLong())
                val embeddingBytes = Math.multiplyExact(values, Float.SIZE_BYTES.toLong())
                Math.addExact(HEADER_SIZE.toLong(), Math.addExact(idsBytes, embeddingBytes))
            } catch (_: ArithmeticException) {
                null
            }
        }

        /**
         * Extract embeddings from SQLite database into a .emb binary file.
         *
         * Streams rows one at a time — never holds more than one embedding in memory.
         *
         * @param table Override the embedding table to extract from.
         *              When null, uses the database's default table (embeddings_clamp3).
         * @param onProgress called periodically with (current, total) track counts
         */
        fun extractFromDatabase(
            db: EmbeddingDatabase,
            outFile: File,
            table: String? = null,
            onProgress: ((current: Int, total: Int) -> Unit)? = null
        ) {
            val tableName = table ?: db.embeddingTable
            val t0 = System.nanoTime()
            Log.d(TAG, "Extracting embeddings from $tableName to ${outFile.name}")

            val numTracks = db.getEmbeddingCountForTable(tableName)
            if (numTracks == 0) {
                Log.w(TAG, "No embeddings in $tableName to extract")
                return
            }

            val actualDim = db.getEmbeddingDimForTable(tableName) ?: return
            val totalMB = numTracks.toLong() * actualDim * 4 / 1024 / 1024
            Log.i(TAG, "Extracting $numTracks embeddings (dim=$actualDim, ~${totalMB} MB)")

            val expectedBlobSize = actualDim * 4  // float32
            val totalSize = HEADER_SIZE.toLong() + numTracks.toLong() * 8 + numTracks.toLong() * actualDim * 4
            val embeddingsStart = HEADER_SIZE.toLong() + numTracks.toLong() * 8

            val destination = outFile.absoluteFile
            destination.parentFile?.mkdirs()
            val temporary = File.createTempFile(
                ".${destination.name}.",
                ".tmp",
                destination.parentFile,
            )
            try {
                RandomAccessFile(temporary, "rw").use { raf ->
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
                    val progressInterval = maxOf(numTracks / 20, 1)  // ~5% increments
                    db.forEachEmbeddingRaw(tableName) { trackId, blob ->
                        if (blob.size != expectedBlobSize) {
                            throw IllegalStateException(
                                "Track $trackId has ${blob.size} embedding bytes; " +
                                    "expected $expectedBlobSize"
                            )
                        }
                        check(i < numTracks) {
                            "Embedding table changed during extraction: more than $numTracks rows"
                        }

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

                    check(i == numTracks) {
                        "Embedding table changed during extraction: expected $numTracks rows, read $i"
                    }
                    onProgress?.invoke(i, numTracks)

                    buf.force()
                }
                replaceAtomically(temporary, destination)
                if (destination.name == "clamp3.emb") {
                    TextEmbeddingIndexGeneration.invalidate()
                }
            } finally {
                temporary.delete()
            }

            val extractMs = (System.nanoTime() - t0) / 1_000_000
            Log.i(TAG, "Wrote ${destination.length() / 1024 / 1024} MB to ${destination.name} in ${extractMs}ms")
        }

        internal fun replaceAtomically(source: File, destination: File) {
            try {
                Files.move(
                    source.toPath(),
                    destination.toPath(),
                    StandardCopyOption.ATOMIC_MOVE,
                    StandardCopyOption.REPLACE_EXISTING,
                )
            } catch (_: AtomicMoveNotSupportedException) {
                Files.move(
                    source.toPath(),
                    destination.toPath(),
                    StandardCopyOption.REPLACE_EXISTING,
                )
            }
        }
    }

    // Precomputed offsets
    private val trackIdsOffset = HEADER_SIZE
    private val embeddingsOffset = HEADER_SIZE + numTracks.toLong() * 8
    private val trackIdLongs: LongBuffer = buffer.duplicate()
        .order(ByteOrder.LITTLE_ENDIAN)
        .apply {
            position(trackIdsOffset)
            limit(embeddingsOffset.toInt())
        }
        .slice()
        .order(ByteOrder.LITTLE_ENDIAN)
        .asLongBuffer()
    private val trackIdSnapshot: LongArray by lazy {
        LongArray(numTracks).also { trackIdLongs.duplicate().get(it) }
    }
    private val embeddingFloats: FloatBuffer = buffer.duplicate()
        .order(ByteOrder.LITTLE_ENDIAN)
        .apply { position(embeddingsOffset.toInt()) }
        .slice()
        .order(ByteOrder.LITTLE_ENDIAN)
        .asFloatBuffer()

    /**
     * Get the track ID at a given index.
     */
    fun getTrackId(index: Int): Long {
        require(index in 0 until numTracks) { "Index out of range: $index" }
        return trackIdAt(index)
    }

    private fun trackIdAt(index: Int): Long = trackIdSnapshot[index]

    internal fun findTrackIndex(trackId: Long): Int? {
        val ids = trackIdSnapshot
        var low = 0
        var high = numTracks - 1
        while (low <= high) {
            val middle = (low + high).ushr(1)
            val candidate = ids[middle]
            when {
                candidate < trackId -> low = middle + 1
                candidate > trackId -> high = middle - 1
                else -> return middle
            }
        }
        // Published V2 and extracted legacy files are ordered. Retain exact support for
        // hand-authored/old unsorted PEMB artifacts without paying a boxed HashMap cost.
        val legacyIndex = ids.indexOf(trackId)
        return legacyIndex.takeIf { it >= 0 }
    }

    /** Resolve IDs once for algorithms that repeatedly score rows from the mmap'd index. */
    internal fun findTrackIndices(trackIds: LongArray): IntArray =
        IntArray(trackIds.size) { position -> findTrackIndex(trackIds[position]) ?: -1 }

    /**
     * Compute dot product between a query vector and the embedding at the given index.
     * Since embeddings are L2-normalized, this equals cosine similarity.
     */
    fun dotProduct(query: FloatArray, index: Int): Float {
        require(query.size == dim) { "Query dimension ${query.size} != index dimension $dim" }
        require(index in 0 until numTracks) { "Index out of range: $index" }
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
     * @param cancellationCheck called at bounded intervals by the native scan
     */
    fun findTop1(
        query: FloatArray,
        excludeIds: Set<Long> = emptySet(),
        cancellationCheck: (() -> Unit)? = null
    ): Pair<Long, Float>? = findTopK(
        query = query,
        topK = 1,
        excludeIds = excludeIds,
        cancellationCheck = cancellationCheck,
    ).firstOrNull()

    /**
     * Find the top-K most similar tracks to a query embedding.
     *
     * Uses NEON-accelerated dot products via JNI for ~30x speedup over scalar Kotlin.
     *
     * @param cancellationCheck called at bounded intervals by the native scan
     */
    fun findTopK(
        query: FloatArray,
        topK: Int,
        excludeIds: Set<Long> = emptySet(),
        cancellationCheck: (() -> Unit)? = null
    ): List<Pair<Long, Float>> {
        require(query.size == dim) { "Query dimension ${query.size} != index dimension $dim" }
        if (topK <= 0 || numTracks == 0) return emptyList()
        val k = topK.coerceAtMost(numTracks)
        val outTrackIds = LongArray(k)
        val outScores = FloatArray(k)
        val excludeArray = if (excludeIds.isEmpty()) null else excludeIds.toLongArray()

        val count = NativeMath.findTopK(
            buffer, trackIdsOffset.toLong(), embeddingsOffset,
            query, numTracks, dim, k,
            excludeArray, outTrackIds, outScores, cancellationCheck
        )

        return (0 until count).map { i -> outTrackIds[i] to outScores[i] }
    }

    /**
     * Compute similarity of every track to a reference vector in one sequential scan.
     * Returns a FloatArray indexed by internal track index (~300KB for 75K tracks).
     * Use with [rankFromSimilarities] for O(1)-amortized rank lookups.
     *
     * Uses NEON-accelerated dot products via JNI.
     */
    fun computeAllSimilarities(reference: FloatArray): FloatArray {
        return FloatArray(numTracks).also { similarities ->
            computeAllSimilaritiesInto(reference, similarities)
        }
    }

    /** Write a full sequential similarity scan into caller-owned storage. */
    internal fun computeAllSimilaritiesInto(
        reference: FloatArray,
        outSimilarities: FloatArray,
        cancellationCheck: (() -> Unit)? = null,
    ) {
        require(reference.size == dim) {
            "Reference dimension ${reference.size} != index dimension $dim"
        }
        require(outSimilarities.size == numTracks) {
            "Similarity count ${outSimilarities.size} != index track count $numTracks"
        }
        cancellationCheck?.invoke()
        if (System.getProperty("java.vm.name") == "Dalvik") {
            NativeMath.allSimilarities(
                buffer,
                embeddingsOffset,
                reference,
                numTracks,
                dim,
                outSimilarities,
            )
        } else {
            for (row in 0 until numTracks) {
                if ((row and 1023) == 0) cancellationCheck?.invoke()
                outSimilarities[row] = dotProduct(reference, row)
            }
        }
        cancellationCheck?.invoke()
    }

    /** Batch exact row-pair scores using the same native reduction as full top-K scans. */
    fun computePairSimilarities(
        leftIndices: IntArray,
        rightIndices: IntArray,
        cancellationCheck: (() -> Unit)? = null,
    ): FloatArray = FloatArray(leftIndices.size).also { scores ->
        computePairSimilaritiesInto(
            leftIndices = leftIndices,
            rightIndices = rightIndices,
            outScores = scores,
            cancellationCheck = cancellationCheck,
        )
    }

    /**
     * Write pair scores into caller-owned storage.
     *
     * Android uses the canonical NEON reduction shared with retrieval. Local JVM tests use the
     * scalar-order reference because Android native libraries cannot be loaded by the host JVM.
     */
    internal fun computePairSimilaritiesInto(
        leftIndices: IntArray,
        rightIndices: IntArray,
        outScores: FloatArray,
        pairCount: Int = leftIndices.size,
        cancellationCheck: (() -> Unit)? = null,
    ) {
        require(pairCount in 0..leftIndices.size &&
            pairCount <= rightIndices.size && pairCount <= outScores.size
        ) { "pair count exceeds a pair score buffer" }
        require((0 until pairCount).all { position ->
            leftIndices[position] in 0 until numTracks &&
                rightIndices[position] in 0 until numTracks
        }
        ) { "pair index is outside the embedding index" }

        if (pairCount == 0) return

        if (System.getProperty("java.vm.name") == "Dalvik") {
            NativeMath.pairSimilarities(
                buffer = buffer,
                embeddingsOffset = embeddingsOffset,
                leftIndices = leftIndices,
                rightIndices = rightIndices,
                numTracks = numTracks,
                dim = dim,
                outScores = outScores,
                pairCount = pairCount,
                cancellationCheck = cancellationCheck,
            )
            return
        }

        for (position in 0 until pairCount) {
            if ((position and 1023) == 0) cancellationCheck?.invoke()
            val leftOffset = (embeddingsOffset + leftIndices[position].toLong() * dim * 4).toInt()
            val rightOffset = (embeddingsOffset + rightIndices[position].toLong() * dim * 4).toInt()
            var score = 0f
            for (column in 0 until dim) {
                score += buffer.getFloat(leftOffset + column * 4) *
                    buffer.getFloat(rightOffset + column * 4)
            }
            outScores[position] = score
        }
        cancellationCheck?.invoke()
    }

    /**
     * Compute 1-based rank of a target track from a precomputed similarity array.
     * Rank 1 = most similar in the corpus. Returns -1 if track not found.
     */
    fun rankFromSimilarities(sims: FloatArray, targetTrackId: Long): Int {
        require(sims.size == numTracks) {
            "Similarity count ${sims.size} != index track count $numTracks"
        }
        val targetIdx = findTrackIndex(targetTrackId) ?: return -1
        return NativeMath.rankFromSimilarities(
            buffer = buffer,
            trackIdsOffset = trackIdsOffset.toLong(),
            similarities = sims,
            numTracks = numTracks,
            targetIndex = targetIdx,
        )
    }

    /**
     * Look up a track's similarity from a precomputed similarity array.
     * Returns 0f if the track ID is not found in the index.
     */
    fun getSimFromPrecomputed(sims: FloatArray, trackId: Long): Float {
        require(sims.size == numTracks) {
            "Similarity count ${sims.size} != index track count $numTracks"
        }
        val idx = findTrackIndex(trackId) ?: return 0f
        return sims[idx]
    }

    /**
     * Copy the embedding at a given internal index into [out].
     */
    fun copyEmbedding(index: Int, out: FloatArray) {
        require(index in 0 until numTracks) { "Index out of range: $index" }
        require(out.size >= dim) { "Output buffer too small: ${out.size} < $dim" }

        embeddingFloats.duplicate()
            .apply { position(index * dim) }
            .get(out, 0, dim)
    }

    /**
     * Get the embedding at a given internal index.
     */
    fun getEmbedding(index: Int): FloatArray {
        val result = FloatArray(dim)
        copyEmbedding(index, result)
        return result
    }

    /** Exact cross-generation row comparison used to prove graph-delta eligibility. */
    internal fun hasBitIdenticalEmbedding(
        index: Int,
        other: EmbeddingIndex,
        otherIndex: Int,
    ): Boolean {
        require(index in 0 until numTracks && otherIndex in 0 until other.numTracks) {
            "Embedding index is outside its generation"
        }
        if (dim != other.dim) return false
        val leftOffset = embeddingsOffset + index.toLong() * dim * Float.SIZE_BYTES
        val rightOffset = other.embeddingsOffset + otherIndex.toLong() * dim * Float.SIZE_BYTES
        for (column in 0 until dim) {
            val byteOffset = column.toLong() * Float.SIZE_BYTES
            if (buffer.getInt((leftOffset + byteOffset).toInt()) !=
                other.buffer.getInt((rightOffset + byteOffset).toInt())
            ) {
                return false
            }
        }
        return true
    }

    /**
     * Get the embedding for a specific track ID, or null if not found.
     */
    fun getEmbeddingByTrackId(trackId: Long): FloatArray? {
        val index = findTrackIndex(trackId) ?: return null
        return getEmbedding(index)
    }

    /** Strict generation-local vector equivalence without allocating either embedding. */
    internal fun hasEmbeddingWithinMaxAbsoluteDelta(
        leftIndex: Int,
        rightIndex: Int,
        maxAbsoluteDelta: Float,
    ): Boolean {
        require(leftIndex in 0 until numTracks && rightIndex in 0 until numTracks) {
            "Embedding index is outside the active generation"
        }
        require(maxAbsoluteDelta >= 0f && maxAbsoluteDelta.isFinite()) {
            "Embedding delta must be finite and non-negative"
        }
        if (leftIndex == rightIndex) return true
        val leftOffset = embeddingsOffset + leftIndex.toLong() * dim * Float.SIZE_BYTES
        val rightOffset = embeddingsOffset + rightIndex.toLong() * dim * Float.SIZE_BYTES
        for (column in 0 until dim) {
            val byteOffset = column.toLong() * Float.SIZE_BYTES
            val left = buffer.getFloat((leftOffset + byteOffset).toInt())
            val right = buffer.getFloat((rightOffset + byteOffset).toInt())
            if (!left.isFinite() || !right.isFinite() ||
                kotlin.math.abs(left - right) > maxAbsoluteDelta
            ) {
                return false
            }
        }
        return true
    }
}
