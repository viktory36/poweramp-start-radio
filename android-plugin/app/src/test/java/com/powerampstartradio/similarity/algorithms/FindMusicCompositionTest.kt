package com.powerampstartradio.similarity.algorithms

import com.powerampstartradio.data.StableVisibleResultIdentity
import com.powerampstartradio.ui.FindMusicRefineNeighborhood
import com.powerampstartradio.ui.FindMusicRefineSpec
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test
import kotlin.math.sqrt

class FindMusicCompositionTest {
    @Test
    fun `equal cosine scores receive one empirical percentile`() {
        val percentiles = GeoMeanSelector.percentileRanks(
            similarities = floatArrayOf(0.2f, 0.8f, 0.8f, -0.1f),
            trackIds = longArrayOf(40, 30, 20, 10),
            rankingTieKeys = listOf("d", "c", "b", "a"),
        )

        assertEquals(listOf(0.5f, 1f, 1f, 0.25f), percentiles.toList())
    }

    @Test
    fun `stable tie keys preserve objective identity order when row IDs change`() {
        fun rankedKeys(trackIds: LongArray, tieKeys: List<String>): List<String> {
            val keysById = trackIds.indices.associate { trackIds[it] to tieKeys[it] }
            return FindMusicComposition.rankAllOf(
                trackIds = trackIds,
                objectiveScores = FloatArray(trackIds.size) { 0.9f },
                anchorPercentiles = listOf(FloatArray(trackIds.size) { 0.9f }),
                topK = trackIds.size,
                rankingTieKeys = tieKeys,
            ).map { keysById.getValue(it.trackId) }
        }

        assertEquals(
            listOf("stable-a", "stable-b"),
            rankedKeys(longArrayOf(10, 20), listOf("stable-b", "stable-a")),
        )
        assertEquals(
            listOf("stable-a", "stable-b"),
            rankedKeys(longArrayOf(200, 100), listOf("stable-a", "stable-b")),
        )
    }

    @Test
    fun allOfIsWeightedGeometricMeanOfIngredientPercentiles() {
        val scores = FindMusicComposition.allOfObjective(
            percentiles = listOf(
                floatArrayOf(1.0f, 0.25f, 0.5f, 0.75f),
                floatArrayOf(0.25f, 1.0f, 0.75f, 0.5f),
            ),
            weights = floatArrayOf(0.5f, 0.5f),
        )

        assertEquals(sqrt(0.25f), scores[0], 1e-6f)
        assertEquals(sqrt(0.25f), scores[1], 1e-6f)
    }

    @Test
    fun refineNeighborhoodWidthsUseExactCeilingAndOnePercentDefault() {
        assertEquals(
            FindMusicRefineNeighborhood.TOP_1_PERCENT,
            FindMusicRefineNeighborhood.DEFAULT,
        )
        assertEquals(
            FindMusicRefineNeighborhood.TOP_1_PERCENT,
            FindMusicRefineSpec(primaryIngredientIndex = 0).neighborhood,
        )

        val domainSize = 85_567
        assertEquals(
            214,
            FindMusicRefineNeighborhood.TOP_0_25_PERCENT.candidateCount(domainSize),
        )
        assertEquals(
            428,
            FindMusicRefineNeighborhood.TOP_0_5_PERCENT.candidateCount(domainSize),
        )
        assertEquals(
            856,
            FindMusicRefineNeighborhood.TOP_1_PERCENT.candidateCount(domainSize),
        )
        assertEquals(
            1_712,
            FindMusicRefineNeighborhood.TOP_2_PERCENT.candidateCount(domainSize),
        )
        assertEquals(0, FindMusicRefineNeighborhood.DEFAULT.candidateCount(0))
        assertEquals(1, FindMusicRefineNeighborhood.DEFAULT.candidateCount(1))
    }

    @Test
    fun refineRanksBySecondaryOnlyInsideTheExactPrimaryNeighborhood() {
        val domainSize = 500
        val trackIds = LongArray(domainSize) { (it + 1).toLong() }
        val primary = FloatArray(domainSize) { 0.1f }.apply {
            this[0] = 0.99f
            this[1] = 0.98f
            this[2] = 0.97f
            this[3] = 0.96f
            this[4] = 0.96f
        }
        val secondary = FloatArray(domainSize) { 0.2f }.apply {
            this[0] = 0.8f
            this[1] = 0.9f
            this[2] = 0.9f
            this[3] = 0.7f
            this[4] = 0.7f
        }
        val tieKeys = List(domainSize) { "track-${it + 1}" }.toMutableList().apply {
            this[3] = "z"
            this[4] = "a"
        }

        val ranked = FindMusicComposition.rankRefine(
            trackIds = trackIds,
            anchorPercentiles = listOf(primary, secondary),
            refineSpec = FindMusicRefineSpec(
                primaryIngredientIndex = 0,
                neighborhood = FindMusicRefineNeighborhood.TOP_1_PERCENT,
            ),
            topK = 50,
            rankingTieKeys = tieKeys,
        )

        assertEquals(5, ranked.objectiveRankingDomainCount)
        assertEquals(listOf(2L, 3L, 1L, 5L, 4L), ranked.rows.map { it.trackId })
        assertEquals(0.9f, ranked.rows.first().objectiveScore, 1e-6f)
        assertEquals(listOf(0.98f, 0.9f), ranked.rows.first().anchorPercentiles)
    }

    @Test
    fun refineAppliesExclusionsBeforeSizingAndSelectingPrimaryNeighborhood() {
        val domainSize = 101
        val trackIds = LongArray(domainSize) { (it + 1).toLong() }
        val primary = FloatArray(domainSize) { index ->
            (domainSize - index).toFloat() / domainSize
        }
        val secondary = FloatArray(domainSize) { index ->
            (index + 1).toFloat() / domainSize
        }

        val ranked = FindMusicComposition.rankRefine(
            trackIds = trackIds,
            anchorPercentiles = listOf(primary, secondary),
            refineSpec = FindMusicRefineSpec(
                primaryIngredientIndex = 0,
                neighborhood = FindMusicRefineNeighborhood.TOP_1_PERCENT,
            ),
            topK = 50,
            excludedTrackIds = setOf(1L),
        )

        assertEquals(1, ranked.objectiveRankingDomainCount)
        assertEquals(listOf(2L), ranked.rows.map { it.trackId })
    }

    @Test
    fun refineRejectsAnythingOtherThanTwoAlignedIngredients() {
        assertThrows(IllegalArgumentException::class.java) {
            FindMusicComposition.rankRefine(
                trackIds = longArrayOf(1L, 2L),
                anchorPercentiles = listOf(floatArrayOf(1f, 0.5f)),
                refineSpec = FindMusicRefineSpec(primaryIngredientIndex = 0),
                topK = 2,
            )
        }
    }

    @Test
    fun allOfRankHasStableTieBreakAndObjectiveAlignedEvidence() {
        val rows = FindMusicComposition.rankAllOf(
            trackIds = longArrayOf(30L, 10L, 20L),
            objectiveScores = floatArrayOf(0.8f, 0.8f, 0.7f),
            anchorPercentiles = listOf(
                floatArrayOf(0.8f, 0.7f, 0.6f),
                floatArrayOf(0.2f, 0.9f, 0.4f),
            ),
            topK = 2,
        )

        assertEquals(listOf(10L, 30L), rows.map { it.trackId })
        assertEquals(listOf(0.7f, 0.9f), rows.first().anchorPercentiles)
    }

    @Test
    fun `verified duplicate cannot change unrelated All of order or scores`() {
        val baseline = rankIdentityDomain(
            trackIds = longArrayOf(10, 20, 30, 40),
            stableTokens = listOf("a", "b", "c", "d"),
            similarities = listOf(
                floatArrayOf(0.95f, 0.8f, 0.4f, 0.2f),
                floatArrayOf(0.2f, 0.7f, 0.9f, 0.4f),
            ),
        )
        val withCopy = rankIdentityDomain(
            trackIds = longArrayOf(10, 20, 30, 40, 50),
            stableTokens = listOf("a", "b", "c", "d", "a"),
            similarities = listOf(
                floatArrayOf(0.95f, 0.8f, 0.4f, 0.2f, 0.95f),
                floatArrayOf(0.2f, 0.7f, 0.9f, 0.4f, 0.2f),
            ),
        )

        assertEquals(baseline.map { it.first }, withCopy.map { it.first })
        baseline.zip(withCopy).forEach { (expected, actual) ->
            assertEquals(expected.second, actual.second, 1e-6f)
        }
    }

    @Test
    fun `unverified legacy rows remain separate composition members`() {
        val representatives = GeoMeanSelector.representativeIndices(
            trackIds = longArrayOf(10, 20, 30),
            identities = listOf(
                StableVisibleResultIdentity("same", false),
                StableVisibleResultIdentity("same", false),
                StableVisibleResultIdentity("stable", true),
            ),
        )

        assertEquals(listOf(0, 1, 2), representatives.toList())
    }

    private fun rankIdentityDomain(
        trackIds: LongArray,
        stableTokens: List<String>,
        similarities: List<FloatArray>,
    ): List<Pair<String, Float>> {
        val identities = stableTokens.map { StableVisibleResultIdentity(it, true) }
        val representativeIndices = GeoMeanSelector.representativeIndices(trackIds, identities)
        val domainTrackIds = LongArray(representativeIndices.size) { position ->
            trackIds[representativeIndices[position]]
        }
        val domainTokens = representativeIndices.map(stableTokens::get)
        val percentiles = similarities.map { raw ->
            val domainSimilarities = FloatArray(representativeIndices.size) { position ->
                raw[representativeIndices[position]]
            }
            GeoMeanSelector.percentileRanks(
                similarities = domainSimilarities,
                trackIds = domainTrackIds,
                rankingTieKeys = domainTokens,
            )
        }
        val scores = FindMusicComposition.allOfObjective(
            percentiles,
            FloatArray(percentiles.size) { 0.5f },
        )
        val rows = FindMusicComposition.rankAllOf(
            trackIds = domainTrackIds,
            objectiveScores = scores,
            anchorPercentiles = percentiles,
            topK = domainTrackIds.size,
            rankingTieKeys = domainTokens,
        )
        val tokenByTrackId = domainTrackIds.indices.associate { index ->
            domainTrackIds[index] to domainTokens[index]
        }
        return rows.map { row ->
            tokenByTrackId.getValue(row.trackId) to row.objectiveScore
        }
    }
}
