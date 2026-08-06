package com.powerampstartradio.ui

import kotlin.math.roundToInt

/** User-facing names for exact selection semantics and controls. */
object SelectionControlText {
    fun modeLabel(mode: SelectionMode): String = when (mode) {
        SelectionMode.CLOSEST -> "Closest"
        SelectionMode.MMR -> "MMR"
        SelectionMode.DPP -> "DPP"
        SelectionMode.RANDOM_WALK -> "Graph Explorer"
        SelectionMode.UNIFORM_SHUFFLE -> "Uniform shuffle"
    }

    fun modeDifferentiator(mode: SelectionMode): String = when (mode) {
        SelectionMode.CLOSEST -> "Direct cosine similarity to the seed"
        SelectionMode.MMR -> "Relevance minus resemblance to earlier picks"
        SelectionMode.DPP -> "Greedy set-wide quality and diversity"
        SelectionMode.RANDOM_WALK -> "Terminal probability across similarity-graph paths"
        SelectionMode.UNIFORM_SHUFFLE -> "Uniform order without similarity ranking"
    }

    fun modeDescription(mode: SelectionMode, driftEnabled: Boolean = false): String = when (mode) {
        SelectionMode.CLOSEST ->
            "Ranks available distinct recordings by cosine similarity to the seed."
        SelectionMode.MMR -> if (driftEnabled) {
            "At each pick, subtracts the weighted maximum similarity to any earlier pick from " +
                "weighted relevance to the evolving direction."
        } else {
            "At each pick, subtracts the weighted maximum similarity to any earlier pick from " +
                "weighted seed relevance."
        }
        SelectionMode.DPP ->
            "Greedily maximizes determinant gain over a quality-weighted cosine-similarity " +
                "kernel, rewarding strong, internally diverse sets."
        SelectionMode.RANDOM_WALK ->
            "Ranks terminal probabilities from deterministic uniform non-backtracking walks " +
                "over the similarity graph."
        SelectionMode.UNIFORM_SHUFFLE ->
            "Creates a reproducible uniform permutation of the other available distinct " +
                "recordings."
    }

    fun mmrBalanceTitle(relevanceWeight: Float, driftEnabled: Boolean): String {
        require(relevanceWeight.isFinite() && relevanceWeight in 0f..1f)
        if (relevanceWeight == 0f) return "Nearest first \u00b7 then variety only"
        val relevance = (relevanceWeight * 100f).roundToInt()
        val variety = 100 - relevance
        val relevanceLabel = if (driftEnabled) {
            "Current-direction relevance"
        } else {
            "Seed relevance"
        }
        return "$relevanceLabel $relevance% \u00b7 variety $variety%"
    }

    fun dppSeedPullLabel(exponent: Float): String = when (
        SelectionKnobPolicy.nearestValue(SelectionKnobPolicy.DPP_SEED_PULL_OPTIONS, exponent)
    ) {
        0f -> "Nearest first, then set variety"
        0.25f -> "Very light seed pull"
        0.5f -> "Light seed pull"
        1f -> "Standard seed pull"
        2f -> "Strong seed pull"
        3f -> "Very strong seed pull"
        4f -> "Strongest seed pull"
        else -> error("DPP seed-pull option has no user-facing label")
    }

    fun driftModeLabel(mode: DriftMode): String = when (mode) {
        DriftMode.SEED_INTERPOLATION -> "Seed + last pick"
        DriftMode.MOMENTUM -> "Rolling direction (momentum)"
    }
}
