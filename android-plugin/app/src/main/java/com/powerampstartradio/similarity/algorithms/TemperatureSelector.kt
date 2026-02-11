package com.powerampstartradio.similarity.algorithms

import com.powerampstartradio.similarity.SelectedTrack
import kotlin.math.ln
import kotlin.random.Random

/**
 * Temperature-based stochastic selection using the Gumbel-max trick
 * with rank-based score transformation.
 *
 * Raw cosine similarities cluster tightly (0.93-0.96 spread ~0.03), so
 * Gumbel noise dominates even at low T. Rank-based transformation maps
 * candidates to uniform [0,1] before applying temperature, giving the T
 * knob meaningful dynamic range.
 *
 * perturbedScore = (1 - rank/N) / T + gumbelNoise()
 * pick = argmax(perturbedScores)
 *
 * T=0: deterministic (always picks top), T→∞: uniform random.
 * Original similarity preserved in SelectedTrack.score.
 */
object TemperatureSelector {

    /**
     * Select one track from candidates with temperature sampling.
     *
     * @param candidates List of (trackId, score) — assumed sorted by score descending
     * @param temperature Controls randomness. 0 = deterministic, higher = more random.
     * @return SelectedTrack of the selected candidate, or null
     */
    fun selectOne(
        candidates: List<Pair<Long, Float>>,
        temperature: Float
    ): SelectedTrack? {
        if (candidates.isEmpty()) return null
        if (temperature <= 1e-6f) {
            var bestIdx = 0
            for (i in 1 until candidates.size) {
                if (candidates[i].second > candidates[bestIdx].second) bestIdx = i
            }
            val (id, score) = candidates[bestIdx]
            return SelectedTrack(id, score, candidateRank = bestIdx + 1)
        }

        // Compute ranks (0 = best, N-1 = worst) for rank-based transform
        val n = candidates.size
        val rankScores = computeRankScores(candidates)

        var bestIdx = -1
        var bestPerturbed = Float.NEGATIVE_INFINITY

        for (i in candidates.indices) {
            val perturbed = rankScores[i] / temperature + gumbelNoise()
            if (perturbed > bestPerturbed) {
                bestPerturbed = perturbed
                bestIdx = i
            }
        }

        if (bestIdx < 0) return null
        val (id, score) = candidates[bestIdx]
        return SelectedTrack(id, score, candidateRank = bestIdx + 1)
    }

    /**
     * Select multiple tracks with temperature sampling (without replacement).
     *
     * @param candidates List of (trackId, score)
     * @param numSelect How many to select
     * @param temperature Controls randomness
     * @return Selected tracks in selection order
     */
    fun selectBatch(
        candidates: List<Pair<Long, Float>>,
        numSelect: Int,
        temperature: Float
    ): List<SelectedTrack> {
        if (candidates.isEmpty()) return emptyList()

        // Build indexed list to track original positions
        data class IndexedCandidate(
            val origIndex: Int, val trackId: Long,
            val score: Float, var rankScore: Float
        )
        val remaining = candidates.mapIndexed { i, (id, score) ->
            IndexedCandidate(i, id, score, 0f)
        }.toMutableList()

        if (temperature <= 1e-6f) {
            return remaining.sortedByDescending { it.score }.take(numSelect).map {
                SelectedTrack(it.trackId, it.score, candidateRank = it.origIndex + 1)
            }
        }

        // Compute rank scores for initial ordering
        val rankScores = computeRankScores(candidates)
        for (i in remaining.indices) {
            remaining[i].rankScore = rankScores[i]
        }

        val result = mutableListOf<SelectedTrack>()

        for (step in 0 until minOf(numSelect, candidates.size)) {
            // Recompute rank scores among remaining candidates
            if (step > 0) {
                val n = remaining.size
                // Sort indices by original score descending to assign ranks
                val sorted = remaining.indices.sortedByDescending { remaining[it].score }
                for ((rank, idx) in sorted.withIndex()) {
                    remaining[idx].rankScore = 1f - rank.toFloat() / n
                }
            }

            var bestIdx = -1
            var bestPerturbed = Float.NEGATIVE_INFINITY

            for (i in remaining.indices) {
                val perturbed = remaining[i].rankScore / temperature + gumbelNoise()
                if (perturbed > bestPerturbed) {
                    bestPerturbed = perturbed
                    bestIdx = i
                }
            }

            if (bestIdx < 0) break
            val picked = remaining.removeAt(bestIdx)
            result.add(SelectedTrack(picked.trackId, picked.score, candidateRank = picked.origIndex + 1))
        }

        return result
    }

    /**
     * Compute rank-based scores: (1 - rank/N) mapped to [0, 1].
     * Rank 0 (best) maps to ~1.0, rank N-1 (worst) maps to ~0.0.
     */
    private fun computeRankScores(candidates: List<Pair<Long, Float>>): FloatArray {
        val n = candidates.size
        if (n == 0) return floatArrayOf()
        // Argsort by score descending
        val sortedIndices = candidates.indices.sortedByDescending { candidates[it].second }
        val rankScores = FloatArray(n)
        for ((rank, origIdx) in sortedIndices.withIndex()) {
            rankScores[origIdx] = 1f - rank.toFloat() / n
        }
        return rankScores
    }

    /**
     * Sample from Gumbel(0, 1) distribution.
     * Gumbel noise = -ln(-ln(U)) where U ~ Uniform(0, 1)
     */
    private fun gumbelNoise(): Float {
        val u = Random.nextFloat().coerceIn(1e-10f, 1f - 1e-10f)
        return (-ln(-ln(u.toDouble()))).toFloat()
    }
}
