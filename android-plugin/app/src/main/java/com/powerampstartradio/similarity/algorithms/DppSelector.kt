package com.powerampstartradio.similarity.algorithms

import com.powerampstartradio.data.EmbeddingIndex
import kotlin.math.sqrt

/**
 * Determinantal Point Process (DPP) greedy MAP selector.
 *
 * Maximizes both quality and diversity simultaneously using a DPP kernel.
 * The kernel L[i][j] = q[i] * q[j] * dot(emb_i, emb_j) captures:
 * - Quality: q[i] = relevance score to seed
 * - Diversity: dot(emb_i, emb_j) = pairwise similarity
 *
 * DPP assigns higher probability to subsets where items are both
 * high-quality and dissimilar — mathematically superior to MMR's
 * pairwise diversity because it considers all pairwise interactions.
 *
 * Uses fast greedy MAP with incremental Cholesky decomposition
 * (Chen et al., 2018). O(K²×N) per selection step.
 */
object DppSelector {

    /**
     * Select a batch of tracks using greedy DPP MAP inference.
     *
     * @param candidates List of (trackId, relevanceScore)
     * @param numSelect How many to select
     * @param index Embedding index for looking up embeddings
     * @param qualityExponent Exponent for quality scores (higher = prefer more relevant)
     * @return Selected tracks as (trackId, relevance) pairs in selection order
     */
    fun selectBatch(
        candidates: List<Pair<Long, Float>>,
        numSelect: Int,
        index: EmbeddingIndex,
        qualityExponent: Float = 1.0f
    ): List<Pair<Long, Float>> {
        if (candidates.isEmpty()) return emptyList()
        val n = candidates.size
        val k = minOf(numSelect, n)

        // Pre-load embeddings and quality scores
        val embeddings = Array(n) { FloatArray(0) }
        val quality = FloatArray(n)
        val validMask = BooleanArray(n)

        for (i in 0 until n) {
            val (trackId, relevance) = candidates[i]
            val emb = index.getEmbeddingByTrackId(trackId)
            if (emb != null) {
                embeddings[i] = emb
                quality[i] = if (qualityExponent == 1.0f) relevance
                             else Math.pow(relevance.toDouble(), qualityExponent.toDouble()).toFloat()
                validMask[i] = true
            }
        }

        // Greedy DPP MAP with incremental Cholesky
        // L[i][j] = q[i] * q[j] * dot(emb_i, emb_j)
        // We maintain a partial Cholesky factor to incrementally compute
        // the marginal gain of adding each candidate.

        val selected = mutableListOf<Int>()
        val dim = embeddings.firstOrNull { it.isNotEmpty() }?.size ?: return emptyList()

        // c[i][j] = Cholesky factor entries for candidate i at step j
        val choleskyFactors = Array(n) { FloatArray(k) }

        // d[i] = remaining diagonal term for candidate i
        // Initially d[i] = L[i][i] = q[i]² * dot(emb_i, emb_i) = q[i]²
        // (embeddings are L2-normalized, so dot(emb_i, emb_i) = 1)
        val diagRemaining = FloatArray(n) { i ->
            if (validMask[i]) quality[i] * quality[i] else 0f
        }

        for (step in 0 until k) {
            // Find candidate with maximum marginal gain (= d[i])
            var bestIdx = -1
            var bestGain = -1f

            for (i in 0 until n) {
                if (!validMask[i] || i in selected.map { it }) continue
                if (diagRemaining[i] > bestGain) {
                    bestGain = diagRemaining[i]
                    bestIdx = i
                }
            }

            if (bestIdx < 0 || bestGain <= 1e-10f) break

            selected.add(bestIdx)

            // Update Cholesky factors for remaining candidates
            val sqrtGain = sqrt(bestGain)

            for (i in 0 until n) {
                if (!validMask[i] || i in selected.map { it }) continue

                // L[i][bestIdx] = q[i] * q[bestIdx] * dot(emb_i, emb_bestIdx)
                val kernelVal = quality[i] * quality[bestIdx] * dotProduct(embeddings[i], embeddings[bestIdx])

                // Subtract contributions from previous Cholesky entries
                var subtracted = kernelVal
                for (j in 0 until step) {
                    subtracted -= choleskyFactors[i][j] * choleskyFactors[bestIdx][j]
                }

                choleskyFactors[i][step] = subtracted / sqrtGain

                // Update remaining diagonal
                diagRemaining[i] -= choleskyFactors[i][step] * choleskyFactors[i][step]
                if (diagRemaining[i] < 0f) diagRemaining[i] = 0f
            }

            // Also update the selected item's own Cholesky factor
            choleskyFactors[bestIdx][step] = sqrtGain
        }

        return selected.map { idx -> candidates[idx] }
    }

    private fun dotProduct(a: FloatArray, b: FloatArray): Float {
        var sum = 0f
        for (i in a.indices) sum += a[i] * b[i]
        return sum
    }
}
