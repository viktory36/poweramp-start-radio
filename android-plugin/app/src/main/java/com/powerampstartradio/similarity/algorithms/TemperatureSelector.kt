package com.powerampstartradio.similarity.algorithms

import com.powerampstartradio.similarity.SelectedTrack
import kotlin.math.ln
import kotlin.random.Random

/**
 * Temperature-based stochastic selection using the Gumbel-max trick.
 *
 * Instead of always picking the highest-scoring candidate, adds calibrated
 * noise so every run produces a different playlist. No softmax needed.
 *
 * perturbedScore = score / T + gumbelNoise()
 * pick = argmax(perturbedScores)
 *
 * T=0: deterministic (always picks top), T→∞: uniform random.
 */
object TemperatureSelector {

    /**
     * Select one track from candidates with temperature sampling.
     *
     * @param candidates List of (trackId, score)
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

        var bestIdx = -1
        var bestPerturbed = Float.NEGATIVE_INFINITY

        for (i in candidates.indices) {
            val perturbed = candidates[i].second / temperature + gumbelNoise()
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
        data class IndexedCandidate(val origIndex: Int, val trackId: Long, val score: Float)
        val remaining = candidates.mapIndexed { i, (id, score) ->
            IndexedCandidate(i, id, score)
        }.toMutableList()

        if (temperature <= 1e-6f) {
            return remaining.sortedByDescending { it.score }.take(numSelect).map {
                SelectedTrack(it.trackId, it.score, candidateRank = it.origIndex + 1)
            }
        }

        val result = mutableListOf<SelectedTrack>()

        for (step in 0 until minOf(numSelect, candidates.size)) {
            var bestIdx = -1
            var bestPerturbed = Float.NEGATIVE_INFINITY

            for (i in remaining.indices) {
                val perturbed = remaining[i].score / temperature + gumbelNoise()
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
     * Sample from Gumbel(0, 1) distribution.
     * Gumbel noise = -ln(-ln(U)) where U ~ Uniform(0, 1)
     */
    private fun gumbelNoise(): Float {
        val u = Random.nextFloat().coerceIn(1e-10f, 1f - 1e-10f)
        return (-ln(-ln(u.toDouble()))).toFloat()
    }
}
