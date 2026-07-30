package com.powerampstartradio.similarity

import com.powerampstartradio.data.StableVisibleResultIdentity
import org.junit.Assert.assertEquals
import org.junit.Test

class StableSimilarityTopKTest {
    @Test
    fun `stable identity decides an exact-score cutoff instead of row ID`() {
        val result = StableSimilarityTopK.select(
            orderedTrackIds = longArrayOf(1, 2, 3),
            similarities = floatArrayOf(0.8f, 0.8f, 0.8f),
            topK = 2,
            rankingTieKey = mapOf(1L to "c", 2L to "a", 3L to "b")::getValue,
        )

        assertEquals(listOf(2L, 3L), result.map(RankedSimilarity::trackId))
    }

    @Test
    fun `stable result identities survive row replacement and input reordering`() {
        val firstKeys = mapOf(10L to "span-a", 20L to "span-b", 30L to "span-c")
        val replacementKeys = mapOf(101L to "span-c", 102L to "span-a", 103L to "span-b")

        val first = StableSimilarityTopK.select(
            orderedTrackIds = longArrayOf(10, 20, 30),
            similarities = floatArrayOf(0.5f, 0.5f, 0.5f),
            topK = 2,
            rankingTieKey = firstKeys::getValue,
        ).map { firstKeys.getValue(it.trackId) }
        val replacement = StableSimilarityTopK.select(
            orderedTrackIds = longArrayOf(101, 103, 102),
            similarities = floatArrayOf(0.5f, 0.5f, 0.5f),
            topK = 2,
            rankingTieKey = replacementKeys::getValue,
        ).map { replacementKeys.getValue(it.trackId) }

        assertEquals(listOf("span-a", "span-b"), first)
        assertEquals(first, replacement)
    }

    @Test
    fun `numeric scores outrank NaN and exclusions refill the bound`() {
        val result = StableSimilarityTopK.select(
            orderedTrackIds = longArrayOf(1, 2, 3, 4),
            similarities = floatArrayOf(Float.NaN, 0.4f, 0.9f, 0.7f),
            topK = 2,
            rankingTieKey = { "track-$it" },
            excludeIds = setOf(3),
        )

        assertEquals(listOf(4L, 2L), result.map(RankedSimilarity::trackId))
    }

    @Test
    fun `verified copies at the top cannot shrink a distinct identity neighborhood`() {
        val stableIdentity = mapOf(
            1L to StableVisibleResultIdentity("span-a", true),
            2L to StableVisibleResultIdentity("span-a", true),
            3L to StableVisibleResultIdentity("span-b", true),
            4L to StableVisibleResultIdentity("span-c", true),
        )

        val result = StableSimilarityTopK.selectDistinctStableIdentities(
            orderedTrackIds = longArrayOf(1, 2, 3, 4),
            similarities = floatArrayOf(0.99f, 0.99f, 0.90f, 0.80f),
            requestedIdentityCount = 3,
            verifiedDuplicateExcessCount = 1,
            rankingTieKey = { stableIdentity.getValue(it).identityToken },
            identityForTrack = stableIdentity::getValue,
        )

        assertEquals(listOf(1L, 3L, 4L), result.map(RankedSimilarity::trackId))
    }

    @Test
    fun `unverified rows never collapse even when their tokens match`() {
        val unverified = StableVisibleResultIdentity("unproven-same-token", false)

        val result = StableSimilarityTopK.selectDistinctStableIdentities(
            orderedTrackIds = longArrayOf(1, 2, 3),
            similarities = floatArrayOf(0.99f, 0.98f, 0.50f),
            requestedIdentityCount = 2,
            verifiedDuplicateExcessCount = 0,
            rankingTieKey = { "row-$it" },
            identityForTrack = { unverified },
        )

        assertEquals(listOf(1L, 2L), result.map(RankedSimilarity::trackId))
    }
}
