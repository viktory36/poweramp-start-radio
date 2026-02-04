package com.powerampstartradio.similarity.algorithms

import com.powerampstartradio.ui.DecaySchedule
import com.powerampstartradio.ui.DriftMode
import com.powerampstartradio.ui.RadioConfig
import kotlin.math.exp
import kotlin.math.sqrt

/**
 * Manages query evolution across drift steps.
 *
 * Two modes:
 * - Seed Interpolation: query = normalize(alpha_t * seed + (1-alpha_t) * current)
 *   where alpha_t decays over time based on the configured schedule
 * - EMA Momentum: ema_t = normalize(beta * ema_{t-1} + (1-beta) * latest_emb)
 *   creates smooth trajectories through embedding space
 */
object DriftEngine {

    /**
     * Compute the query for the next drift step.
     *
     * @param seedEmb Original seed track embedding
     * @param currentEmb Embedding of the most recently selected track
     * @param emaState Running EMA state (only for MOMENTUM mode, null initially)
     * @param step Current step index (0-based)
     * @param totalSteps Total planned steps
     * @param config Radio configuration with drift parameters
     * @return (newQuery, newEmaState) — newEmaState should be passed to next call
     */
    fun updateQuery(
        seedEmb: FloatArray,
        currentEmb: FloatArray,
        emaState: FloatArray?,
        step: Int,
        totalSteps: Int,
        config: RadioConfig
    ): Pair<FloatArray, FloatArray> {
        return when (config.driftMode) {
            DriftMode.SEED_INTERPOLATION -> {
                val alpha = computeAlpha(config.anchorStrength, step, totalSteps, config.anchorDecay)
                val query = interpolate(seedEmb, currentEmb, alpha)
                query to currentEmb
            }
            DriftMode.MOMENTUM -> {
                val prev = emaState ?: seedEmb
                val ema = momentum(prev, currentEmb, config.momentumBeta)
                ema to ema
            }
        }
    }

    /**
     * Compute anchor strength at a given step based on decay schedule.
     *
     * @param baseAlpha Base anchor strength (0..1)
     * @param step Current step
     * @param totalSteps Total steps planned
     * @param decay Decay schedule
     * @return Effective alpha at this step
     */
    private fun computeAlpha(baseAlpha: Float, step: Int, totalSteps: Int, decay: DecaySchedule): Float {
        if (totalSteps <= 1) return baseAlpha
        val progress = step.toFloat() / (totalSteps - 1).toFloat()  // 0..1

        return when (decay) {
            DecaySchedule.NONE -> baseAlpha
            DecaySchedule.LINEAR -> baseAlpha * (1f - progress)
            DecaySchedule.EXPONENTIAL -> baseAlpha * exp(-3f * progress)
            DecaySchedule.STEP -> if (progress < 0.5f) baseAlpha else baseAlpha * 0.2f
        }
    }

    /**
     * Seed interpolation: normalize(alpha * seed + (1-alpha) * current)
     */
    private fun interpolate(seed: FloatArray, current: FloatArray, alpha: Float): FloatArray {
        val result = FloatArray(seed.size)
        for (i in result.indices) {
            result[i] = alpha * seed[i] + (1f - alpha) * current[i]
        }
        return l2Normalize(result)
    }

    /**
     * EMA momentum: normalize(beta * prev + (1-beta) * current)
     */
    private fun momentum(prev: FloatArray, current: FloatArray, beta: Float): FloatArray {
        val result = FloatArray(prev.size)
        for (i in result.indices) {
            result[i] = beta * prev[i] + (1f - beta) * current[i]
        }
        return l2Normalize(result)
    }

    private fun l2Normalize(v: FloatArray): FloatArray {
        var sumSq = 0f
        for (x in v) sumSq += x * x
        val norm = sqrt(sumSq)
        if (norm < 1e-10f) return v
        val result = FloatArray(v.size)
        for (i in v.indices) result[i] = v[i] / norm
        return result
    }
}
