package com.powerampstartradio.similarity.algorithms

import android.util.Log
import com.powerampstartradio.data.EmbeddingIndex
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln

/**
 * Multi-seed ranking via Geometric Mean of Percentiles.
 *
 * For each seed, computes cosine similarity to all tracks, converts to
 * percentile ranks, then takes the weighted geometric mean. This is
 * scale-invariant and works well even for seeds in distant embedding regions
 * (where vector blending collapses).
 *
 * Algorithm:
 * 1. For each seed: dot(seed, all_tracks) → similarities
 * 2. If weight < 0, negate similarities ("less like")
 * 3. Convert to percentile ranks: argsort.argsort / N → (0, 1]
 * 4. Weighted geometric mean: exp(Σ w_i * ln(percentile_i))
 * 5. Return top-K by geo mean score
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
    ): List<Pair<Long, Float>> {
        val n = index.numTracks
        if (n == 0 || seeds.isEmpty()) return emptyList()

        val t0 = System.nanoTime()

        // Normalize absolute weights to sum to 1
        val totalAbsWeight = seeds.sumOf { abs(it.second).toDouble() }.toFloat()
        if (totalAbsWeight < 1e-8f) return emptyList()

        // Compute percentiles for each seed
        val logPercentiles = FloatArray(n) // accumulates weighted log(percentile)
        for ((embedding, weight) in seeds) {
            val normW = abs(weight) / totalAbsWeight
            val sims = index.computeAllSimilarities(embedding)

            // If negative weight, negate similarities
            if (weight < 0) {
                for (i in sims.indices) sims[i] = -sims[i]
            }

            // Argsort → rank mapping
            // sortedIndices[rank] = trackIndex (ascending similarity)
            val sortedIndices = sims.indices.sortedBy { sims[it] }.toIntArray()
            val ranks = IntArray(n)
            for (rank in sortedIndices.indices) {
                ranks[sortedIndices[rank]] = rank
            }

            // Accumulate weighted log(percentile)
            // percentile = (rank + 1) / N, in (0, 1]
            val logN = ln(n.toFloat())
            for (i in 0 until n) {
                logPercentiles[i] += normW * (ln((ranks[i] + 1).toFloat()) - logN)
            }
        }

        // Convert accumulated log to geo mean, find top-K
        // Use a partial sort via priority queue for efficiency
        val excludeSet = if (excludeTrackIds.isEmpty()) null else {
            // Build trackId lookup for exclusion
            val set = HashSet<Long>(excludeTrackIds.size)
            set.addAll(excludeTrackIds)
            set
        }

        // Simple approach: compute scores and find top-K
        data class Scored(val index: Int, val score: Float)
        val topResults = mutableListOf<Scored>()
        var minScore = Float.NEGATIVE_INFINITY

        for (i in 0 until n) {
            if (excludeSet != null && index.getTrackId(i) in excludeSet) continue
            val score = exp(logPercentiles[i])
            if (topResults.size < topK) {
                topResults.add(Scored(i, score))
                if (topResults.size == topK) {
                    topResults.sortByDescending { it.score }
                    minScore = topResults.last().score
                }
            } else if (score > minScore) {
                topResults[topResults.lastIndex] = Scored(i, score)
                topResults.sortByDescending { it.score }
                minScore = topResults.last().score
            }
        }

        topResults.sortByDescending { it.score }
        val elapsed = (System.nanoTime() - t0) / 1_000_000
        Log.d(TAG, "computeRanking: ${seeds.size} seeds, $n tracks, top-$topK in ${elapsed}ms")

        return topResults.map { index.getTrackId(it.index) to it.score }
    }
}
