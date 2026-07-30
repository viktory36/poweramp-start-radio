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
    /** Pinned desktop-compatible preprocessing policy; changing it requires a new spec ID. */
    const val TORCHAUDIO_HANN_V1_SPEC_ID = TorchAudioHannV1Policy.SPEC_ID

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
     * @param cancellationCheck optional callback invoked at bounded scan intervals
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
        cancellationCheck: (() -> Unit)? = null,
    ): Int = nativeFindTopK(
        buffer, trackIdsOffset, embeddingsOffset, query,
        numTracks, dim, topK, excludeIds, outTrackIds, outScores, cancellationCheck
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

    /**
     * Resample an output range from a slice of a larger, conceptual input.
     *
     * Output coordinates and filter phase are relative to the complete input, so adjacent
     * calls can be concatenated byte-for-byte with the corresponding range from
     * [resamplePolyphase]. The supplied input slice must include every in-track source
     * sample needed by the FIR filter; this returns null when context is missing. Samples
     * outside the true track boundaries are the same implicit zero padding used by the
     * whole-track resampler.
     *
     * @param samples Source PCM slice whose first sample is [inputStartSample]
     * @param inputStartSample Global source-sample index represented by `samples[0]`
     * @param totalInputSamples Sample count of the complete conceptual source
     * @param outputStartSample Global output-sample index to render
     * @param outputSampleCount Number of output samples to render
     */
    fun resamplePolyphaseAligned(
        samples: FloatArray,
        fromRate: Int,
        toRate: Int,
        inputStartSample: Long,
        totalInputSamples: Long,
        outputStartSample: Long,
        outputSampleCount: Int,
    ): FloatArray? {
        require(fromRate > 0) { "fromRate must be positive" }
        require(toRate > 0) { "toRate must be positive" }
        require(inputStartSample >= 0L) { "inputStartSample must be non-negative" }
        require(totalInputSamples >= 0L) { "totalInputSamples must be non-negative" }
        require(inputStartSample <= totalInputSamples) {
            "inputStartSample exceeds totalInputSamples"
        }
        require(samples.size.toLong() <= totalInputSamples - inputStartSample) {
            "input slice extends beyond totalInputSamples"
        }
        require(outputStartSample >= 0L) { "outputStartSample must be non-negative" }
        require(outputSampleCount >= 0) { "outputSampleCount must be non-negative" }

        val totalOutputSamples = resampledLength(totalInputSamples, fromRate, toRate)
        require(outputStartSample <= totalOutputSamples) {
            "outputStartSample exceeds resampled output length"
        }
        require(outputSampleCount.toLong() <= totalOutputSamples - outputStartSample) {
            "requested output range exceeds resampled output length"
        }
        if (outputSampleCount == 0) return FloatArray(0)

        if (fromRate == toRate) {
            val inputEndSample = inputStartSample + samples.size
            val outputEndSample = outputStartSample + outputSampleCount
            if (outputStartSample < inputStartSample || outputEndSample > inputEndSample) {
                return null
            }
            val localStart = (outputStartSample - inputStartSample).toInt()
            return samples.copyOfRange(localStart, localStart + outputSampleCount)
        }

        return nativeResamplePolyphaseAligned(
            samples,
            fromRate,
            toRate,
            inputStartSample,
            totalInputSamples,
            outputStartSample,
            outputSampleCount,
        )
    }

    /**
     * Resample with the pinned TorchAudio default Hann V1 policy used by the desktop indexer.
     *
     * This is an explicit alternative to [resamplePolyphase]; existing indexing callers remain
     * on the legacy Kaiser path until the V2 preprocessing spec is integrated end to end.
     */
    fun resampleTorchAudioHannV1(
        samples: FloatArray,
        fromRate: Int,
        toRate: Int,
    ): FloatArray? {
        require(fromRate > 0) { "fromRate must be positive" }
        require(toRate > 0) { "toRate must be positive" }
        if (fromRate == toRate) return samples
        return nativeResampleTorchAudioHannV1(samples, fromRate, toRate)
    }

    /**
     * Render one globally addressed output range using [TORCHAUDIO_HANN_V1_SPEC_ID].
     *
     * Adjacent calls concatenate to the whole-call output because output phase and TorchAudio's
     * float32 target-length truncation are resolved against [totalInputSamples]. The supplied
     * source slice must contain the FIR context needed by the requested output range.
     */
    fun resampleTorchAudioHannV1Aligned(
        samples: FloatArray,
        fromRate: Int,
        toRate: Int,
        inputStartSample: Long,
        totalInputSamples: Long,
        outputStartSample: Long,
        outputSampleCount: Int,
    ): FloatArray? {
        require(fromRate > 0) { "fromRate must be positive" }
        require(toRate > 0) { "toRate must be positive" }
        require(inputStartSample >= 0L) { "inputStartSample must be non-negative" }
        require(totalInputSamples >= 0L) { "totalInputSamples must be non-negative" }
        require(inputStartSample <= totalInputSamples) {
            "inputStartSample exceeds totalInputSamples"
        }
        require(samples.size.toLong() <= totalInputSamples - inputStartSample) {
            "input slice extends beyond totalInputSamples"
        }
        require(outputStartSample >= 0L) { "outputStartSample must be non-negative" }
        require(outputSampleCount >= 0) { "outputSampleCount must be non-negative" }

        val totalOutputSamples = torchAudioHannV1ResampledLength(
            totalInputSamples,
            fromRate,
            toRate,
        )
        require(outputStartSample <= totalOutputSamples) {
            "outputStartSample exceeds resampled output length"
        }
        require(outputSampleCount.toLong() <= totalOutputSamples - outputStartSample) {
            "requested output range exceeds resampled output length"
        }
        if (outputSampleCount == 0) return FloatArray(0)

        if (fromRate == toRate) {
            val inputEndSample = inputStartSample + samples.size
            val outputEndSample = outputStartSample + outputSampleCount
            if (outputStartSample < inputStartSample || outputEndSample > inputEndSample) {
                return null
            }
            val localStart = (outputStartSample - inputStartSample).toInt()
            return samples.copyOfRange(localStart, localStart + outputSampleCount)
        }

        return nativeResampleTorchAudioHannV1Aligned(
            input = samples,
            fromRate = fromRate,
            toRate = toRate,
            inputStartSample = inputStartSample,
            totalInputSamples = totalInputSamples,
            outputStartSample = outputStartSample,
            outputSampleCount = outputSampleCount,
        )
    }

    /**
     * TorchAudio 2.10 computes `ceil(float32(reducedTo * length / reducedFrom))`.
     * This intentionally differs from exact rational ceiling for some long tracks.
     */
    internal fun torchAudioHannV1ResampledLength(
        totalInputSamples: Long,
        fromRate: Int,
        toRate: Int,
    ): Long = TorchAudioHannV1Policy.resampledLength(totalInputSamples, fromRate, toRate)

    internal fun resampledLength(totalInputSamples: Long, fromRate: Int, toRate: Int): Long {
        return try {
            AudioSampleTimeline.resampledSampleCount(totalInputSamples, fromRate, toRate)
        } catch (_: ArithmeticException) {
            throw IllegalArgumentException("resampled output length overflows Long")
        }
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

    /**
     * Score aligned pairs of rows from one mmap'd embedding index in a single JNI call.
     * The native kernel is the same [dot_product] path used by full top-K graph scans.
     */
    fun pairSimilarities(
        buffer: java.nio.ByteBuffer,
        embeddingsOffset: Long,
        leftIndices: IntArray,
        rightIndices: IntArray,
        numTracks: Int,
        dim: Int,
        cancellationCheck: (() -> Unit)? = null,
    ): FloatArray = FloatArray(leftIndices.size).also { scores ->
        pairSimilarities(
            buffer = buffer,
            embeddingsOffset = embeddingsOffset,
            leftIndices = leftIndices,
            rightIndices = rightIndices,
            numTracks = numTracks,
            dim = dim,
            outScores = scores,
            cancellationCheck = cancellationCheck,
        )
    }

    /** Write row-pair scores into a caller-owned buffer so iterative selectors can reuse it. */
    fun pairSimilarities(
        buffer: java.nio.ByteBuffer,
        embeddingsOffset: Long,
        leftIndices: IntArray,
        rightIndices: IntArray,
        numTracks: Int,
        dim: Int,
        outScores: FloatArray,
        pairCount: Int = leftIndices.size,
        cancellationCheck: (() -> Unit)? = null,
    ) {
        require(pairCount in 0..leftIndices.size &&
            pairCount <= rightIndices.size && pairCount <= outScores.size
        ) { "pair count exceeds a pair score buffer" }
        require(numTracks > 0 && dim > 0) { "invalid embedding index shape" }
        require((0 until pairCount).all { position ->
            leftIndices[position] in 0 until numTracks &&
                rightIndices[position] in 0 until numTracks
        }
        ) { "pair index is outside the embedding index" }
        if (pairCount == 0) return
        nativePairSimilarities(
            buffer,
            embeddingsOffset,
            leftIndices,
            rightIndices,
            numTracks,
            dim,
            outScores,
            pairCount,
            cancellationCheck,
        )
    }

    /** Exact 1-based score rank with ascending track ID as the tie-break. */
    fun rankFromSimilarities(
        buffer: java.nio.ByteBuffer,
        trackIdsOffset: Long,
        similarities: FloatArray,
        numTracks: Int,
        targetIndex: Int,
    ): Int = nativeRankFromSimilarities(
        buffer,
        trackIdsOffset,
        similarities,
        numTracks,
        targetIndex,
    )

    @JvmStatic private external fun nativeAllSimilarities(
        buffer: java.nio.ByteBuffer, embOffset: Long,
        query: FloatArray, numTracks: Int, dim: Int, outScores: FloatArray)
    @JvmStatic private external fun nativePairSimilarities(
        buffer: java.nio.ByteBuffer,
        embOffset: Long,
        leftIndices: IntArray,
        rightIndices: IntArray,
        numTracks: Int,
        dim: Int,
        outScores: FloatArray,
        pairCount: Int,
        cancellationCheck: (() -> Unit)?,
    )
    @JvmStatic private external fun nativeRankFromSimilarities(
        buffer: java.nio.ByteBuffer,
        trackIdsOffset: Long,
        similarities: FloatArray,
        numTracks: Int,
        targetIndex: Int,
    ): Int
    @JvmStatic private external fun nativeFindTopK(
        buffer: java.nio.ByteBuffer, trackIdsOffset: Long, embeddingsOffset: Long,
        query: FloatArray, numTracks: Int, dim: Int, topK: Int,
        excludeIds: LongArray?, outTrackIds: LongArray, outScores: FloatArray,
        cancellationCheck: (() -> Unit)?): Int
    @JvmStatic private external fun nativeResamplePolyphase(
        input: FloatArray, fromRate: Int, toRate: Int): FloatArray?
    @JvmStatic private external fun nativeResamplePolyphaseAligned(
        input: FloatArray,
        fromRate: Int,
        toRate: Int,
        inputStartSample: Long,
        totalInputSamples: Long,
        outputStartSample: Long,
        outputSampleCount: Int,
    ): FloatArray?
    @JvmStatic private external fun nativeResampleTorchAudioHannV1(
        input: FloatArray,
        fromRate: Int,
        toRate: Int,
    ): FloatArray?
    @JvmStatic private external fun nativeResampleTorchAudioHannV1Aligned(
        input: FloatArray,
        fromRate: Int,
        toRate: Int,
        inputStartSample: Long,
        totalInputSamples: Long,
        outputStartSample: Long,
        outputSampleCount: Int,
    ): FloatArray?
    @JvmStatic private external fun nativeInt16ToMonoFloat(
        buffer: java.nio.ByteBuffer, offsetBytes: Int, sizeBytes: Int, channels: Int,
        output: FloatArray, dstOffset: Int, maxFrames: Int): Int
}
