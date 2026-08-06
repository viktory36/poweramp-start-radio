package com.powerampstartradio.ui

import java.util.Locale
import kotlin.math.roundToInt

/** Human-scale evidence derived from exact ranks in one persisted ranking domain. */
object LibraryRankEvidenceText {
    fun rank(rank: Int, total: Int? = null): String? {
        if (rank < 1 || (total != null && total < rank)) return null
        val formattedRank = String.format(Locale.US, "%,d", rank)
        return if (total != null) {
            "#$formattedRank of ${String.format(Locale.US, "%,d", total)}"
        } else {
            "#$formattedRank"
        }
    }

    fun rankWithTopFraction(rank: Int, total: Int?): String? {
        val exactRank = rank(rank, total) ?: return null
        val topFraction = topFraction(rank, total) ?: return null
        return "$exactRank \u00b7 $topFraction"
    }

    /** Compact radio-row evidence; denominator and top fraction remain in expanded evidence. */
    fun compactNearestRank(rank: Int, total: Int?): String? {
        if (total == null || rank !in 1..total) return null
        val exactRank = rank(rank) ?: return null
        return "$exactRank nearest"
    }

    /** Compact result-row evidence when the exact denominator is stated once in the header. */
    fun compactRankAndTopFraction(rank: Int, total: Int?): String? {
        val topFraction = topFraction(rank, total) ?: return null
        val exactRank = rank(rank) ?: return null
        return "$exactRank \u00b7 $topFraction"
    }

    fun topFraction(rank: Int, total: Int?): String? {
        if (rank < 1 || total == null || total < rank) return null
        val formatted = when {
            rank.toLong() * 10_000L < total.toLong() -> "<0.01"
            rank.toLong() * 1_000L < total.toLong() ->
                containingPercent(rank = rank, total = total, decimalPlaces = 2)
            else -> containingPercent(rank = rank, total = total, decimalPlaces = 1)
        }
        return "top $formatted%"
    }

    /** A displayed "top X%" is always a containing bound, never a rounded-under claim. */
    private fun containingPercent(rank: Int, total: Int, decimalPlaces: Int): String {
        val scale = if (decimalPlaces == 2) 100L else 10L
        val numerator = rank.toLong() * 100L * scale
        val scaledCeiling = (numerator + total.toLong() - 1L) / total.toLong()
        val whole = scaledCeiling / scale
        val remainder = scaledCeiling % scale
        if (remainder == 0L) return whole.toString()
        return String.format(
            Locale.US,
            if (decimalPlaces == 2) "%d.%02d" else "%d.%01d",
            whole,
            remainder,
        )
    }

    /** Returns the first rank in a tie group from GeoMeanSelector's upper-CDF percentile. */
    fun rankFromUpperCdfPercentile(percentile: Float, total: Int): Int? {
        if (!percentile.isFinite() || percentile <= 0f || percentile > 1f || total < 1) {
            return null
        }
        val rowsAtOrBelow = (percentile * total.toFloat()).roundToInt().coerceIn(1, total)
        val canonicalPercentile = rowsAtOrBelow.toFloat() / total.toFloat()
        if (canonicalPercentile.toBits() != percentile.toBits()) return null
        return total - rowsAtOrBelow + 1
    }

    fun ingredientTopFraction(percentile: Float, total: Int): String? =
        rankFromUpperCdfPercentile(percentile, total)?.let { topFraction(it, total) }

    fun ingredientRankWithTopFraction(percentile: Float, total: Int): String? =
        rankFromUpperCdfPercentile(percentile, total)?.let { rank ->
            rankWithTopFraction(rank, total)
        }

    fun compactIngredientRankAndTopFraction(percentile: Float, total: Int): String? =
        rankFromUpperCdfPercentile(percentile, total)?.let { rank ->
            compactRankAndTopFraction(rank, total)
        }
}
