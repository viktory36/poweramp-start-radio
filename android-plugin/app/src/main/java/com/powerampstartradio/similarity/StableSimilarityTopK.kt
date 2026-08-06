package com.powerampstartradio.similarity

import com.powerampstartradio.data.StableVisibleResultIdentity
import java.util.PriorityQueue

data class RankedSimilarity(
    val trackId: Long,
    val score: Float,
)

/**
 * Bounded top-K selection whose tie-breaker is a stable library identity, not a database row ID.
 *
 * Similarities remain the sole retrieval objective. The stable key is consulted only when two
 * float scores are exactly equal, making the result invariant to row replacement and reordering.
 */
object StableSimilarityTopK {
    /**
     * Retrieve a bounded relevance neighborhood measured in distinct, proven identities.
     *
     * [verifiedDuplicateExcessCount] is an upper bound on rows in the active domain which may
     * collapse under [identityForTrack]. Ranking those extra rows guarantees that verified copies
     * at the head of the ranking cannot consume places in the requested identity neighborhood.
     * Unverified identities are deliberately retained even if a caller repeats their token.
     */
    fun selectDistinctStableIdentities(
        orderedTrackIds: LongArray,
        similarities: FloatArray,
        requestedIdentityCount: Int,
        verifiedDuplicateExcessCount: Int,
        rankingTieKey: (Long) -> String,
        identityForTrack: (Long) -> StableVisibleResultIdentity,
        excludeIds: Set<Long> = emptySet(),
        cancellationCheck: (() -> Unit)? = null,
    ): List<RankedSimilarity> {
        require(requestedIdentityCount >= 0) {
            "Requested identity count cannot be negative"
        }
        require(verifiedDuplicateExcessCount >= 0) {
            "Verified duplicate excess count cannot be negative"
        }
        if (requestedIdentityCount == 0 || orderedTrackIds.isEmpty()) return emptyList()

        val rankedRowCount = (requestedIdentityCount.toLong() + verifiedDuplicateExcessCount)
            .coerceAtMost(orderedTrackIds.size.toLong())
            .toInt()
        val rankedRows = select(
            orderedTrackIds = orderedTrackIds,
            similarities = similarities,
            topK = rankedRowCount,
            rankingTieKey = rankingTieKey,
            excludeIds = excludeIds,
            cancellationCheck = cancellationCheck,
        )
        val distinct = StableVisibleResultReducer.reduce(
            rankedItems = rankedRows,
            requestedVisibleCount = requestedIdentityCount,
            identityOf = { ranked -> identityForTrack(ranked.trackId) },
        )
        cancellationCheck?.invoke()
        return distinct.items
    }

    fun select(
        orderedTrackIds: LongArray,
        similarities: FloatArray,
        topK: Int,
        rankingTieKey: (Long) -> String,
        excludeIds: Set<Long> = emptySet(),
        cancellationCheck: (() -> Unit)? = null,
    ): List<RankedSimilarity> {
        require(orderedTrackIds.size == similarities.size) {
            "Track ID count ${orderedTrackIds.size} != similarity count ${similarities.size}"
        }
        if (topK <= 0 || orderedTrackIds.isEmpty()) return emptyList()

        val requestedCount = topK.coerceAtMost(orderedTrackIds.size)
        if (requestedCount == orderedTrackIds.size) {
            val allCandidates = ArrayList<HeapEntry>(orderedTrackIds.size)
            orderedTrackIds.indices.forEach { index ->
                if (index and CANCELLATION_CHECK_MASK == 0) cancellationCheck?.invoke()
                val trackId = orderedTrackIds[index]
                if (trackId !in excludeIds) {
                    allCandidates += HeapEntry(
                        trackId = trackId,
                        score = similarities[index],
                        tieKey = rankingTieKey(trackId),
                    )
                }
            }
            cancellationCheck?.invoke()
            allCandidates.sortWith(BEST_FIRST)
            cancellationCheck?.invoke()
            return allCandidates.map { entry -> RankedSimilarity(entry.trackId, entry.score) }
        }

        val heap = PriorityQueue(requestedCount, WORST_FIRST)

        orderedTrackIds.indices.forEach { index ->
            if (index and CANCELLATION_CHECK_MASK == 0) cancellationCheck?.invoke()

            val trackId = orderedTrackIds[index]
            if (trackId in excludeIds) return@forEach

            val candidate = HeapEntry(
                trackId = trackId,
                score = similarities[index],
                tieKey = rankingTieKey(trackId),
            )
            if (heap.size < requestedCount) {
                heap.add(candidate)
            } else if (compareBestFirst(candidate, requireNotNull(heap.peek())) < 0) {
                heap.poll()
                heap.add(candidate)
            }
        }
        cancellationCheck?.invoke()

        return heap.sortedWith(BEST_FIRST).map { entry ->
            RankedSimilarity(entry.trackId, entry.score)
        }
    }

    private data class HeapEntry(
        val trackId: Long,
        val score: Float,
        val tieKey: String,
    )

    private val BEST_FIRST = Comparator<HeapEntry>(::compareBestFirst)
    private val WORST_FIRST = Comparator<HeapEntry> { left, right ->
        compareBestFirst(right, left)
    }

    /** NaN is invalid ranking evidence and follows every numeric score. */
    private fun compareBestFirst(left: HeapEntry, right: HeapEntry): Int {
        val leftNaN = left.score.isNaN()
        val rightNaN = right.score.isNaN()
        if (leftNaN != rightNaN) return if (leftNaN) 1 else -1

        if (!leftNaN) {
            if (left.score > right.score) return -1
            if (left.score < right.score) return 1
        }

        val stableKeyOrder = left.tieKey.compareTo(right.tieKey)
        if (stableKeyOrder != 0) return stableKeyOrder
        return left.trackId.compareTo(right.trackId)
    }

    private const val CANCELLATION_CHECK_MASK = 0x3ff
}
