package com.powerampstartradio.ui

/** Exact control surface for the two canonical artist-credit constraints. */
data class ArtistConstraintControlState(
    val showMaximum: Boolean,
    val showSpacing: Boolean,
    val maximumOptions: List<Int>,
    val spacingOptions: List<Int>,
    val spacingOccurrenceCeiling: Int,
    val evidenceLine: String?,
)

object ArtistConstraintControlPolicy {
    const val MAXIMUM_UI_LIMIT = 10
    const val SPACING_UI_LIMIT = 20

    fun forRequest(
        recommendationCount: Int,
        maxPerArtist: Int,
        minArtistSpacing: Int,
    ): ArtistConstraintControlState {
        require(recommendationCount >= 1)
        require(maxPerArtist >= 1)
        require(minArtistSpacing >= 0)

        val spacingCeiling = maximumOccurrencesFromSpacing(
            recommendationCount = recommendationCount,
            minArtistSpacing = minArtistSpacing,
        )
        if (recommendationCount == 1) {
            return ArtistConstraintControlState(
                showMaximum = false,
                showSpacing = false,
                maximumOptions = emptyList(),
                spacingOptions = emptyList(),
                spacingOccurrenceCeiling = spacingCeiling,
                evidenceLine = "One recommendation cannot repeat an artist credit.",
            )
        }

        // If both constraints independently mean "no repeats", keep the maximum visible so
        // raising it can hand control back to the preserved spacing value.
        val showSpacing = maxPerArtist > 1
        val showMaximum = maxPerArtist == 1 || spacingCeiling > 1

        return ArtistConstraintControlState(
            showMaximum = showMaximum,
            showSpacing = showSpacing,
            maximumOptions = if (showMaximum) {
                distinctMaximumOptions(
                    spacingOccurrenceCeiling = spacingCeiling,
                    currentValue = maxPerArtist,
                )
            } else {
                emptyList()
            },
            spacingOptions = if (showSpacing) {
                distinctSpacingOptions(
                    recommendationCount = recommendationCount,
                    currentValue = minArtistSpacing,
                )
            } else {
                emptyList()
            },
            spacingOccurrenceCeiling = spacingCeiling,
            evidenceLine = when {
                !showSpacing ->
                    "Maximum 1 already prevents the same artist credit appearing twice."
                !showMaximum ->
                    "For $recommendationCount recommendations, spacing $minArtistSpacing " +
                        "permits at most one track with the same artist credit."
                spacingCeiling < maxPerArtist ->
                    "Together, these settings allow at most $spacingCeiling tracks with the " +
                        "same artist credit."
                else -> null
            },
        )
    }

    /** Maximum same-credit occurrences possible in N accepted positions: ceil(N / (S + 1)). */
    fun maximumOccurrencesFromSpacing(
        recommendationCount: Int,
        minArtistSpacing: Int,
    ): Int {
        require(recommendationCount >= 1)
        require(minArtistSpacing >= 0)
        return 1 + (recommendationCount - 1) / (minArtistSpacing + 1)
    }

    private fun distinctMaximumOptions(
        spacingOccurrenceCeiling: Int,
        currentValue: Int,
    ): List<Int> {
        require(spacingOccurrenceCeiling >= 1)
        require(currentValue >= 1)

        if (spacingOccurrenceCeiling == 1 && currentValue == 1) {
            return listOf(1, 2)
        }

        val lastDistinct = minOf(MAXIMUM_UI_LIMIT, spacingOccurrenceCeiling)
        val base = (1..lastDistinct).toMutableList()
        if (currentValue >= spacingOccurrenceCeiling) {
            base.removeAll { it >= spacingOccurrenceCeiling }
            base += currentValue
        } else if (currentValue !in base) {
            base += currentValue
        }
        return base.distinct().sorted()
    }

    private fun distinctSpacingOptions(
        recommendationCount: Int,
        currentValue: Int,
    ): List<Int> {
        require(recommendationCount >= 2)
        require(currentValue >= 0)

        val noRepeatSpacing = recommendationCount - 1
        val lastDistinct = minOf(SPACING_UI_LIMIT, noRepeatSpacing)
        val base = (0..lastDistinct).toMutableList()
        if (currentValue >= noRepeatSpacing) {
            base.removeAll { it >= noRepeatSpacing }
            base += currentValue
        } else if (currentValue !in base) {
            base += currentValue
        }
        return base.distinct().sorted()
    }
}
