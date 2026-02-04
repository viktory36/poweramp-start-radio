package com.powerampstartradio.similarity.algorithms

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
     * @return (trackId, originalScore) of the selected candidate, or null
     */
    fun selectOne(
        candidates: List<Pair<Long, Float>>,
        temperature: Float
    ): Pair<Long, Float>? {
        if (candidates.isEmpty()) return null
        if (temperature <= 1e-6f) return candidates.maxByOrNull { it.second }

        var bestId = -1L
        var bestOrigScore = 0f
        var bestPerturbed = Float.NEGATIVE_INFINITY

        for ((trackId, score) in candidates) {
            val perturbed = score / temperature + gumbelNoise()
            if (perturbed > bestPerturbed) {
                bestPerturbed = perturbed
                bestId = trackId
                bestOrigScore = score
            }
        }

        return if (bestId >= 0) bestId to bestOrigScore else null
    }

    /**
     * Select multiple tracks with temperature sampling (without replacement).
     *
     * @param candidates List of (trackId, score)
     * @param numSelect How many to select
     * @param temperature Controls randomness
     * @return Selected (trackId, originalScore) pairs in selection order
     */
    fun selectBatch(
        candidates: List<Pair<Long, Float>>,
        numSelect: Int,
        temperature: Float
    ): List<Pair<Long, Float>> {
        if (candidates.isEmpty()) return emptyList()
        if (temperature <= 1e-6f) {
            return candidates.sortedByDescending { it.second }.take(numSelect)
        }

        val remaining = candidates.toMutableList()
        val result = mutableListOf<Pair<Long, Float>>()

        for (step in 0 until minOf(numSelect, candidates.size)) {
            // Perturb all remaining scores and pick max
            var bestIdx = -1
            var bestPerturbed = Float.NEGATIVE_INFINITY

            for (i in remaining.indices) {
                val perturbed = remaining[i].second / temperature + gumbelNoise()
                if (perturbed > bestPerturbed) {
                    bestPerturbed = perturbed
                    bestIdx = i
                }
            }

            if (bestIdx < 0) break
            result.add(remaining.removeAt(bestIdx))
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
