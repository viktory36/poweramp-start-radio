package com.powerampstartradio.similarity.algorithms

import com.powerampstartradio.ui.DecaySchedule
import com.powerampstartradio.ui.DriftMode
import com.powerampstartradio.ui.RadioConfig
import kotlin.math.max
import kotlin.math.pow
import kotlin.math.sqrt

/**
 * Result of a drift query update, including provenance metadata.
 *
 * @param query The new query embedding for the next search step
 * @param emaState Updated EMA state to pass to next call
 * @param seedWeight The exact mathematical weight of the seed on this step's query
 */
data class DriftResult(
    val query: FloatArray,
    val emaState: FloatArray,
    val seedWeight: Float,
)

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
     * @param config Radio configuration with drift parameters
     * @return DriftResult with new query, EMA state, and seed weight for provenance
     */
    fun updateQuery(
        seedEmb: FloatArray,
        currentEmb: FloatArray,
        emaState: FloatArray?,
        step: Int,
        config: RadioConfig
    ): DriftResult {
        return when (config.driftMode) {
            DriftMode.SEED_INTERPOLATION -> {
                val alpha = computeAlpha(
                    baseAlpha = config.anchorStrength,
                    step = step,
                    decay = config.anchorDecay,
                    halfLifeTracks = config.effectiveAnchorHalfLifeTracks,
                )
                val query = interpolate(seedEmb, currentEmb, alpha)
                DriftResult(query, currentEmb, seedWeight = alpha)
            }
            DriftMode.MOMENTUM -> {
                val prev = emaState ?: seedEmb
                val beta = config.momentumBeta
                val ema = momentum(prev, currentEmb, beta)
                // Seed weight in EMA: beta^(step+1) — seed contribution decays geometrically
                val seedWeight = Math.pow(beta.toDouble(), (step + 1).toDouble()).toFloat()
                DriftResult(ema, ema, seedWeight = seedWeight)
            }
        }
    }

    /**
     * Compute anchor strength at a given step based on decay schedule.
     *
     * @param baseAlpha Base anchor strength (0..1)
     * @param step Current step
     * @param decay Decay schedule
     * @return Effective alpha at this step
     */
    internal fun computeAlpha(
        baseAlpha: Float,
        step: Int,
        decay: DecaySchedule,
        halfLifeTracks: Float,
    ): Float {
        require(baseAlpha in 0f..1f) { "baseAlpha must be between 0 and 1" }
        require(step >= 0) { "step must be non-negative" }
        require(halfLifeTracks > 0f && halfLifeTracks.isFinite()) {
            "halfLifeTracks must be finite and positive"
        }
        val elapsed = step.toFloat()
        return when (decay) {
            DecaySchedule.NONE -> baseAlpha
            DecaySchedule.LINEAR -> baseAlpha * max(0f, 1f - elapsed / (2f * halfLifeTracks))
            DecaySchedule.EXPONENTIAL ->
                baseAlpha * 0.5f.pow(elapsed / halfLifeTracks)
            DecaySchedule.STEP -> if (elapsed < halfLifeTracks) baseAlpha else baseAlpha * 0.2f
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
