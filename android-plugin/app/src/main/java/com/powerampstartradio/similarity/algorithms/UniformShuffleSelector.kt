package com.powerampstartradio.similarity.algorithms

import com.powerampstartradio.data.UniformShuffleIdentityKey
import java.util.PriorityQueue

/** One position in a deterministic uniform-style permutation of the eligible library. */
data class UniformShufflePick(
    val trackId: Long,
    val shuffleRank: Int,
    val stableIdentity: Boolean,
)

/**
 * Seeded, input-order-independent shuffle without replacement.
 *
 * Each verified source-span key is assigned a seeded 128-bit SplitMix-style priority. Sorting by
 * that priority creates a pseudo-random permutation independent of SQLite row assignment. Legacy
 * rows use a key explicitly scoped to one database generation. Metadata and embeddings do not
 * influence membership.
 */
object UniformShuffleSelector {
    fun select(
        trackIds: LongArray,
        numSelect: Int,
        seed: Long,
        identityKeyForTrack: (Long) -> UniformShuffleIdentityKey,
        excludeIds: Set<Long> = emptySet(),
        isEligible: (Long, List<Long>) -> Boolean = { _, _ -> true },
    ): List<UniformShufflePick> {
        require(numSelect >= 0) { "numSelect must be non-negative" }
        if (numSelect == 0 || trackIds.isEmpty()) return emptyList()
        val canonicalTracks = canonicalizeByIdentity(
            trackIds = trackIds,
            identityKeyForTrack = identityKeyForTrack,
            excludeIds = excludeIds,
        )
        if (canonicalTracks.isEmpty()) return emptyList()

        var prefixSize = minOf(canonicalTracks.size, maxOf(numSelect * 2, MIN_PREFIX_SIZE))
        while (true) {
            val rankedPrefix = smallestPriorityPrefix(
                canonicalTracks,
                prefixSize,
                seed,
            )
            val selectedIds = ArrayList<Long>(minOf(numSelect, rankedPrefix.size))
            val selected = ArrayList<UniformShufflePick>(minOf(numSelect, rankedPrefix.size))
            for ((index, candidate) in rankedPrefix.withIndex()) {
                if (!isEligible(candidate.trackId, selectedIds)) continue
                selected += UniformShufflePick(
                    trackId = candidate.trackId,
                    shuffleRank = index + 1,
                    stableIdentity = candidate.identityKey.isStableAcrossGenerations,
                )
                selectedIds += candidate.trackId
                if (selected.size == numSelect) return selected
            }

            if (prefixSize == canonicalTracks.size) return selected
            prefixSize = minOf(canonicalTracks.size, prefixSize * 2)
        }
    }

    private fun canonicalizeByIdentity(
        trackIds: LongArray,
        identityKeyForTrack: (Long) -> UniformShuffleIdentityKey,
        excludeIds: Set<Long>,
    ): List<CanonicalTrack> {
        val excludedTokens = excludeIds.mapTo(HashSet(excludeIds.size)) { trackId ->
            identityKeyForTrack(trackId).identityToken
        }
        val canonicalByToken = HashMap<String, CanonicalTrack>(trackIds.size)
        trackIds.forEach { trackId ->
            val identityKey = identityKeyForTrack(trackId)
            if (identityKey.identityToken in excludedTokens) return@forEach
            val previous = canonicalByToken[identityKey.identityToken]
            if (previous != null) {
                require(previous.identityKey == identityKey) {
                    "Uniform shuffle identity token maps to inconsistent keys"
                }
            }
            if (previous == null || trackId < previous.trackId) {
                canonicalByToken[identityKey.identityToken] = CanonicalTrack(trackId, identityKey)
            }
        }
        return canonicalByToken.values.toList()
    }

    /** Exact first [limit] positions without sorting or allocating one object per track. */
    private fun smallestPriorityPrefix(
        canonicalTracks: List<CanonicalTrack>,
        limit: Int,
        seed: Long,
    ): List<RankedTrack> {
        val worstFirst = ASCENDING.reversed()
        val heap = PriorityQueue(limit, worstFirst)
        for ((trackId, identityKey) in canonicalTracks) {
            val candidatePriority = priority(identityKey, seed)
            val candidate = RankedTrack(trackId, identityKey, candidatePriority.first, candidatePriority.second)
            if (heap.size < limit) {
                heap += candidate
            } else {
                val worst = requireNotNull(heap.peek())
                if (ASCENDING.compare(candidate, worst) < 0) {
                    heap.poll()
                    heap += candidate
                }
            }
        }
        return heap.sortedWith(ASCENDING)
    }

    private fun priority(identityKey: UniformShuffleIdentityKey, seed: Long): Pair<Long, Long> =
        mix64(identityKey.high xor mix64(seed)) to
            mix64(identityKey.low xor mix64(seed + GOLDEN_GAMMA))

    /** Deterministic, decorrelated seed progression for the explicit New order command. */
    fun nextSeed(seed: Long): Long {
        val next = mix64(seed + GOLDEN_GAMMA)
        return if (next == 0L) 1L else next
    }

    private fun mix64(input: Long): Long {
        var value = input
        value = (value xor (value ushr 30)) * MIX_MULTIPLIER_1
        value = (value xor (value ushr 27)) * MIX_MULTIPLIER_2
        return value xor (value ushr 31)
    }

    private data class RankedTrack(
        val trackId: Long,
        val identityKey: UniformShuffleIdentityKey,
        val priorityHigh: Long,
        val priorityLow: Long,
    )

    private data class CanonicalTrack(
        val trackId: Long,
        val identityKey: UniformShuffleIdentityKey,
    )

    private val ASCENDING = Comparator<RankedTrack> { left, right ->
        unsignedCompare(left.priorityHigh, right.priorityHigh)
            .takeUnless { it == 0 }
            ?: unsignedCompare(left.priorityLow, right.priorityLow)
                .takeUnless { it == 0 }
            ?: unsignedCompare(left.identityKey.high, right.identityKey.high)
                .takeUnless { it == 0 }
            ?: unsignedCompare(left.identityKey.low, right.identityKey.low)
                .takeUnless { it == 0 }
            ?: left.identityKey.identityToken.compareTo(right.identityKey.identityToken)
                .takeUnless { it == 0 }
            ?: left.trackId.compareTo(right.trackId)
    }

    private fun unsignedCompare(left: Long, right: Long): Int =
        java.lang.Long.compareUnsigned(left, right)

    private const val MIN_PREFIX_SIZE = 256
    private const val GOLDEN_GAMMA = -7046029254386353131L
    private const val MIX_MULTIPLIER_1 = -4658895280553007687L
    private const val MIX_MULTIPLIER_2 = -7723592293110705685L
}
