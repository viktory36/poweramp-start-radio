package com.powerampstartradio.similarity.algorithms

import android.util.Log
import com.powerampstartradio.data.EmbeddingIndex
import com.powerampstartradio.data.StableTrackIdentityCatalog
import com.powerampstartradio.data.StableVisibleResultIdentity
import com.powerampstartradio.ui.FindMusicOperator
import com.powerampstartradio.ui.FindMusicRefineSpec

data class RankedComposedRow(
    val objectiveRank: Int,
    val row: ComposedRankingRow,
)

data class DetailedComposedRanking(
    val rows: List<ComposedRankingRow>,
    /** Identity-representative rows used to compute every ingredient percentile. */
    val ingredientRankingDomainCount: Int,
    /** Exact domain ranked by the objective; Refine narrows this to its primary neighborhood. */
    val objectiveRankingDomainCount: Int,
)

/** Full All-of order backed by primitive arrays; row evidence is materialized only while scanned. */
class AllOfRankingSnapshot internal constructor(
    private val trackIds: LongArray,
    private val objectiveScores: FloatArray,
    private val anchorPercentiles: List<FloatArray>,
    private val rankedIndices: IntArray,
) : Iterable<RankedComposedRow> {
    val size: Int get() = rankedIndices.size

    override fun iterator(): Iterator<RankedComposedRow> = object : Iterator<RankedComposedRow> {
        private var position = 0

        override fun hasNext(): Boolean = position < rankedIndices.size

        override fun next(): RankedComposedRow {
            if (!hasNext()) throw NoSuchElementException()
            val objectiveRank = position + 1
            val trackIndex = rankedIndices[position++]
            return RankedComposedRow(
                objectiveRank = objectiveRank,
                row = ComposedRankingRow(
                    trackId = trackIds[trackIndex],
                    objectiveScore = objectiveScores[trackIndex],
                    anchorPercentiles = anchorPercentiles.map { it[trackIndex] },
                ),
            )
        }
    }
}

/**
 * Composed retrieval over per-ingredient corpus percentiles.
 *
 * For each seed, computes cosine similarity to all tracks, converts to
 * percentile ranks, then takes the weighted geometric mean. This is
 * scale-invariant and works well even for seeds in distant embedding regions
 * (where vector blending collapses).
 *
 * Algorithm:
 * 1. For each seed: dot(seed, all_tracks) → similarities
 * 2. If weight < 0, negate similarities ("less like")
 * 3. Convert to empirical upper-CDF percentiles; equal cosine scores stay tied
 * 4. Apply the explicit All of or Refine objective
 * 5. Return top-K with that objective's evidence
 */
object GeoMeanSelector {

    private const val TAG = "GeoMeanSelector"

    /**
     * Compute geo-mean-of-percentiles ranking across multiple seeds.
     *
     * @param index Mmap'd embedding index
     * @param seeds List of (embedding, weight) pairs. Weight sign:
     *              positive = "more like", negative = "less like".
     *              Magnitude controls relative importance.
     * @param topK Number of results to return
     * @param excludeTrackIds Track IDs to exclude from results (e.g. song seeds)
     * @return Ordered list of (trackId, geoMeanScore), descending by score
     */
    fun computeRanking(
        index: EmbeddingIndex,
        seeds: List<Pair<FloatArray, Float>>,
        topK: Int,
        excludeTrackIds: Set<Long> = emptySet(),
        includedTrackIds: Set<Long>? = null,
    ): List<Pair<Long, Float>> = computeRankingDetailed(
        index = index,
        seeds = seeds,
        operator = FindMusicOperator.ALL_OF,
        topK = topK,
        excludeTrackIds = excludeTrackIds,
        includedTrackIds = includedTrackIds,
    ).map { it.trackId to it.objectiveScore }

    /**
     * Rank using the selected explicit objective and retain that objective's evidence.
     * Negative signs are applied before percentile conversion. Refine requires a positive
     * primary ingredient; its secondary ingredient may be positive or negative.
     */
    fun computeRankingDetailed(
        index: EmbeddingIndex,
        seeds: List<Pair<FloatArray, Float>>,
        operator: FindMusicOperator,
        topK: Int,
        refineSpec: FindMusicRefineSpec? = null,
        excludeTrackIds: Set<Long> = emptySet(),
        identityCatalog: StableTrackIdentityCatalog? = null,
        includedTrackIds: Set<Long>? = null,
        cancellationCheck: (() -> Unit)? = null,
    ): List<ComposedRankingRow> = computeRankingDetailedSnapshot(
        index = index,
        seeds = seeds,
        operator = operator,
        topK = topK,
        refineSpec = refineSpec,
        excludeTrackIds = excludeTrackIds,
        identityCatalog = identityCatalog,
        includedTrackIds = includedTrackIds,
        cancellationCheck = cancellationCheck,
    ).rows

    fun computeRankingDetailedSnapshot(
        index: EmbeddingIndex,
        seeds: List<Pair<FloatArray, Float>>,
        operator: FindMusicOperator,
        topK: Int,
        refineSpec: FindMusicRefineSpec? = null,
        excludeTrackIds: Set<Long> = emptySet(),
        identityCatalog: StableTrackIdentityCatalog? = null,
        includedTrackIds: Set<Long>? = null,
        cancellationCheck: (() -> Unit)? = null,
    ): DetailedComposedRanking {
        val n = index.numTracks
        if (n == 0 || seeds.isEmpty()) {
            return DetailedComposedRanking(emptyList(), 0, 0)
        }

        val t0 = System.nanoTime()
        val space = computePercentileSpace(
            index,
            seeds,
            identityCatalog,
            includedTrackIds,
            cancellationCheck,
        )
        cancellationCheck?.invoke()
        val weights = FloatArray(seeds.size) { seeds[it].second }
        val representativeExclusions = representativeExclusions(
            excludeTrackIds,
            identityCatalog,
            space.trackIds,
        )
        val excludedRankingRows = representativeExclusions.count { excludedId ->
            excludedId in space.trackIds
        }
        val objectiveRankingDomainCount: Int
        val topResults = when (operator) {
            FindMusicOperator.ALL_OF -> {
                require(refineSpec == null) { "All of cannot carry a Refine specification" }
                val scores = FindMusicComposition.allOfObjective(space.percentiles, weights)
                objectiveRankingDomainCount = space.trackIds.size - excludedRankingRows
                FindMusicComposition.rankAllOf(
                    trackIds = space.trackIds,
                    objectiveScores = scores,
                    anchorPercentiles = space.percentiles,
                    topK = topK,
                    excludedTrackIds = representativeExclusions,
                    rankingTieKeys = space.rankingTieKeys,
                )
            }
            FindMusicOperator.REFINE -> {
                val exactRefineSpec = requireNotNull(refineSpec) {
                    "Refine needs a primary ingredient and neighborhood"
                }
                require(weights.size == 2 && weights.all { it.isFinite() && it != 0f }) {
                    "Refine needs exactly two finite non-zero ingredient weights"
                }
                require(exactRefineSpec.primaryIngredientIndex in weights.indices) {
                    "Refine primary ingredient is out of range"
                }
                require(weights[exactRefineSpec.primaryIngredientIndex] > 0f) {
                    "Refine primary ingredient must be positive"
                }
                val refined = FindMusicComposition.rankRefine(
                    trackIds = space.trackIds,
                    anchorPercentiles = space.percentiles,
                    refineSpec = exactRefineSpec,
                    topK = topK,
                    excludedTrackIds = representativeExclusions,
                    rankingTieKeys = space.rankingTieKeys,
                )
                objectiveRankingDomainCount = refined.objectiveRankingDomainCount
                refined.rows
            }
        }
        cancellationCheck?.invoke()
        val elapsed = (System.nanoTime() - t0) / 1_000_000
        Log.d(TAG, "computeRanking: $operator, ${seeds.size} seeds, $n tracks, top-$topK in ${elapsed}ms")

        return DetailedComposedRanking(
            rows = topResults,
            ingredientRankingDomainCount = space.trackIds.size,
            objectiveRankingDomainCount = objectiveRankingDomainCount,
        )
    }

    fun computeAllOfRankingSnapshot(
        index: EmbeddingIndex,
        seeds: List<Pair<FloatArray, Float>>,
        excludeTrackIds: Set<Long> = emptySet(),
        identityCatalog: StableTrackIdentityCatalog? = null,
        includedTrackIds: Set<Long>? = null,
        cancellationCheck: (() -> Unit)? = null,
    ): AllOfRankingSnapshot {
        if (index.numTracks == 0 || seeds.isEmpty()) {
            return AllOfRankingSnapshot(LongArray(0), FloatArray(0), emptyList(), IntArray(0))
        }
        val space = computePercentileSpace(
            index,
            seeds,
            identityCatalog,
            includedTrackIds,
            cancellationCheck,
        )
        cancellationCheck?.invoke()
        val weights = FloatArray(seeds.size) { seeds[it].second }
        val scores = FindMusicComposition.allOfObjective(space.percentiles, weights)
        val representativeExclusions = representativeExclusions(
            excludeTrackIds,
            identityCatalog,
            space.trackIds,
        )
        val rankedIndices = FindMusicComposition.rankedAllOfIndices(
            trackIds = space.trackIds,
            scores = scores,
            topK = space.trackIds.size,
            excludedTrackIds = representativeExclusions,
            rankingTieKeys = space.rankingTieKeys,
        )
        cancellationCheck?.invoke()
        return AllOfRankingSnapshot(
            trackIds = space.trackIds,
            objectiveScores = scores,
            anchorPercentiles = space.percentiles,
            rankedIndices = rankedIndices.toIntArray(),
        )
    }

    private fun computePercentileSpace(
        index: EmbeddingIndex,
        seeds: List<Pair<FloatArray, Float>>,
        identityCatalog: StableTrackIdentityCatalog?,
        includedTrackIds: Set<Long>?,
        cancellationCheck: (() -> Unit)?,
    ): PercentileSpace {
        val sourceTrackIds = LongArray(index.numTracks) { index.getTrackId(it) }
        if (includedTrackIds != null) {
            require(includedTrackIds.isNotEmpty()) { "Included track domain must not be empty" }
            require(includedTrackIds.all { sourceTrackIds.binarySearch(it) >= 0 }) {
                "Included track domain contains an ID outside the embedding index"
            }
        }
        val includedSourceIndices = if (includedTrackIds == null) {
            sourceTrackIds.indices.toList().toIntArray()
        } else {
            sourceTrackIds.indices.filter { sourceTrackIds[it] in includedTrackIds }.toIntArray()
        }
        require(includedTrackIds == null || includedSourceIndices.size == includedTrackIds.size) {
            "Included track domain does not map one-to-one onto the embedding index"
        }
        val includedSourceTrackIds = LongArray(includedSourceIndices.size) { position ->
            sourceTrackIds[includedSourceIndices[position]]
        }
        val sourceIdentities = identityCatalog?.let { catalog ->
            List(includedSourceTrackIds.size) { position ->
                catalog.visibleResultIdentity(includedSourceTrackIds[position])
            }
        }
        val representativeIndices = representativeIndices(includedSourceTrackIds, sourceIdentities)
        val trackIds = LongArray(representativeIndices.size) { position ->
            includedSourceTrackIds[representativeIndices[position]]
        }
        val representativeSourceIndices = IntArray(representativeIndices.size) { position ->
            includedSourceIndices[representativeIndices[position]]
        }
        val rankingTieKeys = identityCatalog?.let { catalog ->
            List(trackIds.size) { position -> catalog.rankingTieKey(trackIds[position]) }
        }
        val percentiles = ArrayList<FloatArray>(seeds.size)
        for ((embedding, weight) in seeds) {
            cancellationCheck?.invoke()
            val sourceSimilarities = index.computeAllSimilarities(embedding)
            val direction = if (weight < 0f) -1f else 1f
            val similarities = FloatArray(representativeIndices.size) { position ->
                sourceSimilarities[representativeSourceIndices[position]] * direction
            }
            percentiles += percentileRanks(similarities, trackIds, rankingTieKeys)
        }
        return PercentileSpace(trackIds, rankingTieKeys, percentiles)
    }

    /** One row per indexing-verified acoustic identity; every legacy row remains distinct. */
    internal fun representativeIndices(
        trackIds: LongArray,
        identities: List<StableVisibleResultIdentity>?,
    ): IntArray {
        if (identities == null) return trackIds.indices.toList().toIntArray()
        require(identities.size == trackIds.size)

        val representativeByStableIdentity = HashMap<String, Int>()
        identities.forEachIndexed { index, identity ->
            if (!identity.isCollapsibleRecording) return@forEachIndexed
            val previous = representativeByStableIdentity[identity.identityToken]
            if (previous == null || trackIds[index] < trackIds[previous]) {
                representativeByStableIdentity[identity.identityToken] = index
            }
        }

        return trackIds.indices.filter { index ->
            val identity = identities[index]
            !identity.isCollapsibleRecording ||
                representativeByStableIdentity.getValue(identity.identityToken) == index
        }.toIntArray()
    }

    private fun representativeExclusions(
        excludedTrackIds: Set<Long>,
        identityCatalog: StableTrackIdentityCatalog?,
        representativeTrackIds: LongArray,
    ): Set<Long> {
        if (identityCatalog == null || excludedTrackIds.isEmpty()) return excludedTrackIds
        val excludedIdentityTokens = excludedTrackIds.mapTo(hashSetOf()) { trackId ->
            identityCatalog.visibleResultIdentity(trackId).identityToken
        }
        return representativeTrackIds.filterTo(linkedSetOf()) { trackId ->
            identityCatalog.visibleResultIdentity(trackId).identityToken in
                excludedIdentityTokens
        }
    }

    private data class PercentileSpace(
        val trackIds: LongArray,
        val rankingTieKeys: List<String>?,
        val percentiles: List<FloatArray>,
    )

    /** Empirical upper-CDF percentile: exactly equal cosine scores receive one percentile. */
    internal fun percentileRanks(
        similarities: FloatArray,
        trackIds: LongArray,
        rankingTieKeys: List<String>? = null,
    ): FloatArray {
        require(similarities.size == trackIds.size)
        require(rankingTieKeys == null || rankingTieKeys.size == trackIds.size)
        if (similarities.isEmpty()) return FloatArray(0)
        require(similarities.all(Float::isFinite)) { "Similarity scores must be finite" }
        val sortedIndices = similarities.indices.sortedWith(
            compareBy<Int> { similarities[it] }
                .thenBy { rankingTieKeys?.get(it).orEmpty() }
                .thenBy { trackIds[it] },
        )
        val ranks = FloatArray(similarities.size)
        var start = 0
        while (start < sortedIndices.size) {
            var endExclusive = start + 1
            val score = similarities[sortedIndices[start]]
            while (endExclusive < sortedIndices.size &&
                similarities[sortedIndices[endExclusive]] == score
            ) {
                endExclusive++
            }
            val percentile = endExclusive.toFloat() / similarities.size
            for (position in start until endExclusive) {
                ranks[sortedIndices[position]] = percentile
            }
            start = endExclusive
        }
        return ranks
    }
}
