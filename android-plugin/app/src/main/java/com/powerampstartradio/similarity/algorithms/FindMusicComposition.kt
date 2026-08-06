package com.powerampstartradio.similarity.algorithms

import com.powerampstartradio.ui.FindMusicRefineSpec
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import java.util.PriorityQueue

data class ComposedRankingRow(
    val trackId: Long,
    val objectiveScore: Float,
    val anchorPercentiles: List<Float>,
)

data class RefinedRanking(
    val rows: List<ComposedRankingRow>,
    /** Exact non-excluded primary-neighborhood domain ranked by the secondary ingredient. */
    val objectiveRankingDomainCount: Int,
)

/** Pure deterministic objectives shared by production ranking and focused tests. */
object FindMusicComposition {
    fun allOfObjective(
        percentiles: List<FloatArray>,
        weights: FloatArray,
    ): FloatArray {
        val size = validateAlignedInputs(percentiles, weights)
        require(weights.all { it.isFinite() }) { "Ingredient weights must be finite" }
        return allOf(percentiles, weights, size)
    }

    fun rankAllOf(
        trackIds: LongArray,
        objectiveScores: FloatArray,
        anchorPercentiles: List<FloatArray>,
        topK: Int,
        excludedTrackIds: Set<Long> = emptySet(),
        rankingTieKeys: List<String>? = null,
    ): List<ComposedRankingRow> {
        require(trackIds.size == objectiveScores.size)
        require(anchorPercentiles.all { it.size == trackIds.size })
        require(rankingTieKeys == null || rankingTieKeys.size == trackIds.size)
        if (topK <= 0) return emptyList()

        val eligible = rankedAllOfIndices(
            trackIds,
            objectiveScores,
            topK,
            excludedTrackIds,
            rankingTieKeys,
        )
        return eligible.map { index ->
            val evidence = anchorPercentiles.map { it[index] }
            ComposedRankingRow(
                trackId = trackIds[index],
                objectiveScore = objectiveScores[index],
                anchorPercentiles = evidence,
            )
        }
    }

    /** Rank by the secondary ingredient inside an exact-sized primary neighborhood. */
    fun rankRefine(
        trackIds: LongArray,
        anchorPercentiles: List<FloatArray>,
        refineSpec: FindMusicRefineSpec,
        topK: Int,
        excludedTrackIds: Set<Long> = emptySet(),
        rankingTieKeys: List<String>? = null,
    ): RefinedRanking {
        require(anchorPercentiles.size == 2) { "Refine needs exactly two ingredients" }
        require(anchorPercentiles.all { it.size == trackIds.size }) {
            "Refine ingredient score arrays must align"
        }
        require(anchorPercentiles.all { values ->
            values.all { it.isFinite() && it > 0f && it <= 1f }
        }) { "Refine ingredient percentiles must be finite and in (0, 1]" }
        require(rankingTieKeys == null || rankingTieKeys.size == trackIds.size)
        require(refineSpec.primaryIngredientIndex in anchorPercentiles.indices) {
            "Refine primary ingredient is out of range"
        }
        if (trackIds.isEmpty()) {
            return RefinedRanking(emptyList(), objectiveRankingDomainCount = 0)
        }
        val primaryIndex = refineSpec.primaryIngredientIndex
        val secondaryIndex = 1 - primaryIndex
        val availableCount = trackIds.count { it !in excludedTrackIds }
        val neighborhoodCount = refineSpec.neighborhood.candidateCount(availableCount)
        val primaryNeighborhood = rankedAllOfIndices(
            trackIds = trackIds,
            scores = anchorPercentiles[primaryIndex],
            topK = neighborhoodCount,
            excludedTrackIds = excludedTrackIds,
            rankingTieKeys = rankingTieKeys,
        )
        val refinedBestFirst = compareByDescending<Int> {
            anchorPercentiles[secondaryIndex][it]
        }.thenByDescending {
            anchorPercentiles[primaryIndex][it]
        }.thenBy {
            rankingTieKeys?.get(it).orEmpty()
        }.thenBy {
            trackIds[it]
        }
        val rows = if (topK <= 0) {
            emptyList()
        } else {
            primaryNeighborhood.sortedWith(refinedBestFirst)
                .take(topK)
                .map { trackIndex ->
                    ComposedRankingRow(
                        trackId = trackIds[trackIndex],
                        objectiveScore = anchorPercentiles[secondaryIndex][trackIndex],
                        anchorPercentiles = anchorPercentiles.map { it[trackIndex] },
                    )
                }
        }
        return RefinedRanking(
            rows = rows,
            objectiveRankingDomainCount = neighborhoodCount,
        )
    }

    internal fun rankedAllOfIndices(
        trackIds: LongArray,
        scores: FloatArray,
        topK: Int,
        excludedTrackIds: Set<Long>,
        rankingTieKeys: List<String>?,
    ): List<Int> {
        require(trackIds.size == scores.size)
        if (topK <= 0) return emptyList()
        val bestFirst = compareByDescending<Int> { scores[it] }
            .thenBy { rankingTieKeys?.get(it).orEmpty() }
            .thenBy { trackIds[it] }
        val eligibleCount = trackIds.count { it !in excludedTrackIds }
        if (topK >= eligibleCount) {
            return trackIds.indices
                .filter { trackIds[it] !in excludedTrackIds }
                .sortedWith(bestFirst)
        }
        val worstFirst = bestFirst.reversed()
        val heap = PriorityQueue<Int>(worstFirst)
        for (index in trackIds.indices) {
            if (trackIds[index] in excludedTrackIds) continue
            if (heap.size < topK) {
                heap += index
            } else if (bestFirst.compare(index, heap.peek()) < 0) {
                heap.poll()
                heap += index
            }
        }
        return heap.sortedWith(bestFirst)
    }

    private fun allOf(
        percentiles: List<FloatArray>,
        weights: FloatArray,
        size: Int,
    ): FloatArray {
        val total = weights.fold(0.0) { sum, weight -> sum + abs(weight.toDouble()) }
        require(total > 1e-8) { "All of needs at least one non-zero ingredient" }
        val logScores = DoubleArray(size)
        for (anchor in percentiles.indices) {
            val normalizedWeight = abs(weights[anchor].toDouble()) / total
            for (track in 0 until size) {
                val percentile = percentiles[anchor][track].toDouble().coerceIn(1.0 / size, 1.0)
                logScores[track] += normalizedWeight * ln(percentile)
            }
        }
        return FloatArray(size) { track -> exp(logScores[track]).toFloat() }
    }

    private fun validateAlignedInputs(
        percentiles: List<FloatArray>,
        weights: FloatArray,
        expectedSize: Int? = null,
    ): Int {
        require(percentiles.isNotEmpty()) { "At least one ingredient is required" }
        require(percentiles.size == weights.size) { "One weight is required per ingredient" }
        val size = expectedSize ?: percentiles.first().size
        require(percentiles.all { it.size == size }) { "Ingredient score arrays must align" }
        require(percentiles.all { values -> values.all { it.isFinite() && it > 0f && it <= 1f } }) {
            "Ingredient percentiles must be finite and in (0, 1]"
        }
        return size
    }
}
