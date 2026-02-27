package com.powerampstartradio.indexing

/**
 * NEON-accelerated math operations for embedding search and indexing.
 *
 * Provides batch operations to minimize JNI call overhead:
 * - Batch dot products: 1×N similarities for kNN search
 * - Top-K mmap search: NEON dot products + C min-heap
 * - Polyphase FIR resampler: Kaiser-windowed sinc with NEON convolution
 */
object NativeMath {
    init {
        System.loadLibrary("math-jni")
    }

    /**
     * Batch dot products: one query against [n] candidates.
     *
     * @param query Float array [dim]
     * @param candidates Flat float array [n × dim], row-major
     * @param n Number of candidates
     * @param dim Embedding dimension
     * @return similarities[n] — dot product of query with each candidate
     */
    fun batchDot(
        query: FloatArray,
        candidates: FloatArray, n: Int,
        dim: Int,
    ): FloatArray? = nativeBatchDot(query, candidates, n, dim)

    /**
     * Top-K search on a mmap'd .emb file using NEON dot products + C min-heap.
     * ~30x faster than scalar Kotlin dotProduct loop over mmap.
     *
     * @param buffer mmap'd .emb file (direct ByteBuffer)
     * @param trackIdsOffset byte offset to int64 track ID array
     * @param embeddingsOffset byte offset to float32 embedding array
     * @param query query vector [dim]
     * @param numTracks total tracks in the index
     * @param dim embedding dimension
     * @param topK how many results to return
     * @param excludeIds track IDs to skip (null for none)
     * @param outTrackIds pre-allocated LongArray[topK] for result track IDs
     * @param outScores pre-allocated FloatArray[topK] for result scores
     * @return actual number of results (≤ topK)
     */
    fun findTopK(
        buffer: java.nio.ByteBuffer,
        trackIdsOffset: Long,
        embeddingsOffset: Long,
        query: FloatArray,
        numTracks: Int,
        dim: Int,
        topK: Int,
        excludeIds: LongArray?,
        outTrackIds: LongArray,
        outScores: FloatArray,
    ): Int = nativeFindTopK(
        buffer, trackIdsOffset, embeddingsOffset, query,
        numTracks, dim, topK, excludeIds, outTrackIds, outScores
    )

    /**
     * Polyphase FIR resampler — equivalent to scipy.signal.resample_poly.
     * Kaiser-windowed sinc filter with NEON-accelerated convolution.
     * ~200x faster than soxr HQ with identical embedding quality.
     *
     * @param samples Mono PCM float samples
     * @param fromRate Source sample rate (e.g. 44100)
     * @param toRate Target sample rate (e.g. 24000)
     * @return Resampled samples, or null on error
     */
    fun resamplePolyphase(
        samples: FloatArray,
        fromRate: Int,
        toRate: Int,
    ): FloatArray? {
        if (fromRate == toRate) return samples
        return nativeResamplePolyphase(samples, fromRate, toRate)
    }

    fun int16ToMonoFloat(
        buffer: java.nio.ByteBuffer,
        offsetBytes: Int, sizeBytes: Int,
        channels: Int,
        output: FloatArray, dstOffset: Int, maxFrames: Int,
    ): Int = nativeInt16ToMonoFloat(buffer, offsetBytes, sizeBytes, channels, output, dstOffset, maxFrames)

    @JvmStatic private external fun nativeBatchDot(
        query: FloatArray, candidates: FloatArray, n: Int, dim: Int): FloatArray?
    /**
     * Compute dot product of one query against all embeddings in a mmap'd .emb file.
     * Returns all scores via outScores[numTracks].
     */
    fun allSimilarities(
        buffer: java.nio.ByteBuffer,
        embeddingsOffset: Long,
        query: FloatArray,
        numTracks: Int,
        dim: Int,
        outScores: FloatArray,
    ) = nativeAllSimilarities(buffer, embeddingsOffset, query, numTracks, dim, outScores)

    @JvmStatic private external fun nativeAllSimilarities(
        buffer: java.nio.ByteBuffer, embOffset: Long,
        query: FloatArray, numTracks: Int, dim: Int, outScores: FloatArray)
    @JvmStatic private external fun nativeFindTopK(
        buffer: java.nio.ByteBuffer, trackIdsOffset: Long, embeddingsOffset: Long,
        query: FloatArray, numTracks: Int, dim: Int, topK: Int,
        excludeIds: LongArray?, outTrackIds: LongArray, outScores: FloatArray): Int
    @JvmStatic private external fun nativeResamplePolyphase(
        input: FloatArray, fromRate: Int, toRate: Int): FloatArray?
    @JvmStatic private external fun nativeInt16ToMonoFloat(
        buffer: java.nio.ByteBuffer, offsetBytes: Int, sizeBytes: Int, channels: Int,
        output: FloatArray, dstOffset: Int, maxFrames: Int): Int
}
