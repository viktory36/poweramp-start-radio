package com.powerampstartradio.indexing

/**
 * NEON-accelerated math operations for embedding fusion.
 *
 * Provides batch operations to minimize JNI call overhead:
 * - k-means assignment: n×K dot products → labels (replaces 15s Kotlin loop with ~2s native)
 * - Batch dot products: 1×N similarities for kNN search
 * - Covariance accumulation: streaming outer products
 * - Matrix-vector multiply: SVD projection
 */
object NativeMath {
    init {
        System.loadLibrary("math-jni")
    }

    /**
     * k-means assignment: for each of [n] points, find the nearest centroid.
     *
     * @param embeddings Flat float array [n × dim], row-major
     * @param n Number of points
     * @param centroids Flat float array [k × dim], row-major
     * @param k Number of centroids
     * @param dim Embedding dimension
     * @return labels[n] — index of nearest centroid per point
     */
    fun kmeansAssign(
        embeddings: FloatArray, n: Int,
        centroids: FloatArray, k: Int,
        dim: Int,
    ): IntArray? = nativeKmeansAssign(embeddings, n, centroids, k, dim)

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
     * Accumulate covariance matrix: C += Σ (x_i × x_i^T) for a batch of vectors.
     * Only fills upper triangle. Uses double precision.
     *
     * @param covariance Double array [dim × dim], modified in place
     * @param vectors Flat float array [batch × dim], row-major
     * @param batch Number of vectors
     * @param dim Dimension
     */
    fun covarianceAccum(
        covariance: DoubleArray,
        vectors: FloatArray, batch: Int,
        dim: Int,
    ) = nativeCovarianceAccum(covariance, vectors, batch, dim)

    /**
     * Matrix-vector multiply: result[rows] = matrix[rows, cols] * vector[cols].
     *
     * @param matrix Flat float array [rows × cols], row-major
     * @param rows Number of rows
     * @param cols Number of columns
     * @param vector Float array [cols]
     * @return result[rows]
     */
    fun matVecMul(
        matrix: FloatArray, rows: Int, cols: Int,
        vector: FloatArray,
    ): FloatArray? = nativeMatVecMul(matrix, rows, cols, vector)

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
     * Jacobi eigendecomposition for a symmetric matrix, fully in native code.
     *
     * @param matrix Symmetric n×n matrix as flat DoubleArray (row-major)
     * @param n Matrix dimension
     * @param maxSweeps Maximum Jacobi sweeps
     * @param eps Convergence threshold
     * @return Flat array [eigenvalues[n], eigenvectors[n×n]] sorted by descending eigenvalue,
     *         or null on failure
     */
    fun jacobiEigen(
        matrix: DoubleArray, n: Int,
        maxSweeps: Int = 50, eps: Double = 1e-10,
    ): DoubleArray? = nativeJacobiEigen(matrix, n, maxSweeps, eps)

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

    @JvmStatic private external fun nativeKmeansAssign(
        embeddings: FloatArray, n: Int, centroids: FloatArray, k: Int, dim: Int): IntArray?
    @JvmStatic private external fun nativeBatchDot(
        query: FloatArray, candidates: FloatArray, n: Int, dim: Int): FloatArray?
    @JvmStatic private external fun nativeCovarianceAccum(
        covariance: DoubleArray, vectors: FloatArray, batch: Int, dim: Int)
    @JvmStatic private external fun nativeMatVecMul(
        matrix: FloatArray, rows: Int, cols: Int, vector: FloatArray): FloatArray?
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
    @JvmStatic private external fun nativeJacobiEigen(
        matrix: DoubleArray, n: Int, maxSweeps: Int, eps: Double): DoubleArray?
    @JvmStatic private external fun nativeResamplePolyphase(
        input: FloatArray, fromRate: Int, toRate: Int): FloatArray?
    @JvmStatic private external fun nativeInt16ToMonoFloat(
        buffer: java.nio.ByteBuffer, offsetBytes: Int, sizeBytes: Int, channels: Int,
        output: FloatArray, dstOffset: Int, maxFrames: Int): Int
}
