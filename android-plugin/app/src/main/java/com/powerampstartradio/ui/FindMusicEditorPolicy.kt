package com.powerampstartradio.ui

/** User-visible readiness of the live Find Music editor. */
data class FindMusicEditorReadiness(
    val canSearch: Boolean,
    val reason: String?,
)

data class FindMusicEditorSnapshot(
    val textIngredients: List<TextIngredientState>,
    val songSeeds: List<SongSeedState>,
    val operator: FindMusicOperator,
    val resultLimit: Int,
)

/**
 * Pure editor rules. A visible, typed recording remains an ingredient until it is confirmed or
 * removed; query construction must never silently omit it.
 */
internal object FindMusicEditorPolicy {
    /** Two active shares have one degree of freedom; three or more need per-row allocation. */
    fun usesSharedBalance(activeIngredientCount: Int): Boolean = activeIngredientCount == 2

    fun usesPerIngredientWeightControls(activeIngredientCount: Int): Boolean =
        activeIngredientCount >= 3

    /** Refine has one primary and one modifier; larger compositions remain explicit All-of queries. */
    fun shouldShowOperatorControl(activeIngredientCount: Int): Boolean =
        activeIngredientCount == 2

    fun effectiveOperator(
        requested: FindMusicOperator,
        activeIngredientCount: Int,
    ): FindMusicOperator =
        if (requested == FindMusicOperator.REFINE && activeIngredientCount != 2) {
            FindMusicOperator.ALL_OF
        } else {
            requested
        }

    fun readiness(
        textIngredients: List<TextIngredientState>,
        songSeeds: List<SongSeedState>,
        searchRunning: Boolean,
        operator: FindMusicOperator = FindMusicOperator.ALL_OF,
        resultLimit: Int = FindMusicQuerySpec.DEFAULT_RESULT_LIMIT,
        refinePrimaryIngredientIndex: Int = 0,
    ): FindMusicEditorReadiness {
        if (searchRunning) {
            return denied("Finishing the current Find Music request...")
        }
        val unconfirmed = songSeeds.filter {
            it.query.isNotBlank() && it.confirmedTrack == null
        }
        if (unconfirmed.isNotEmpty()) {
            return denied(
                if (unconfirmed.size == 1) {
                    "Choose the exact recording for the typed recording ingredient."
                } else {
                    "Choose the exact recording for every typed recording ingredient."
                },
            )
        }
        val unfundedIngredientCount = textIngredients.count {
            it.query.isNotBlank() && it.weight <= 0f
        } + songSeeds.count {
            it.confirmedTrack != null && it.weight <= 0f
        }
        if (unfundedIngredientCount > 0) {
            return denied(
                "Release a held share so every chosen ingredient can receive at least 1%.",
            )
        }
        val activeTexts = textIngredients.filter(TextIngredientState::isActive)
        val activeSongs = songSeeds.filter(SongSeedState::isActive)
        if (activeTexts.isEmpty() && activeSongs.isEmpty()) {
            return denied("Add a description or choose a recording.")
        }
        if (activeTexts.none { !it.negative } && activeSongs.none { !it.negative }) {
            return denied("Keep at least one Like ingredient to define the sound you want.")
        }
        if (operator == FindMusicOperator.REFINE) {
            val signs = activeTexts.map { it.negative } + activeSongs.map { it.negative }
            if (signs.size != 2) {
                return denied("Refine needs exactly two ingredients.")
            }
            if (refinePrimaryIngredientIndex !in signs.indices ||
                signs[refinePrimaryIngredientIndex]
            ) {
                return denied("Choose a Like ingredient as the primary sound.")
            }
        }
        return FindMusicEditorReadiness(canSearch = true, reason = null)
    }

    fun canSetTextToAvoid(
        textIngredients: List<TextIngredientState>,
        songSeeds: List<SongSeedState>,
        index: Int,
    ): Boolean {
        val target = textIngredients.getOrNull(index) ?: return false
        return target.isActive && !target.negative && positiveCount(textIngredients, songSeeds) > 1
    }

    fun canSetSongToAvoid(
        textIngredients: List<TextIngredientState>,
        songSeeds: List<SongSeedState>,
        index: Int,
    ): Boolean {
        val target = songSeeds.getOrNull(index) ?: return false
        return target.isActive && !target.negative && positiveCount(textIngredients, songSeeds) > 1
    }

    fun shouldShowSignControl(
        negative: Boolean,
        canSetAvoid: Boolean,
        operator: FindMusicOperator,
    ): Boolean = negative || canSetAvoid

    fun defaults(): FindMusicEditorSnapshot = FindMusicEditorSnapshot(
        textIngredients = listOf(TextIngredientState()),
        songSeeds = emptyList(),
        operator = FindMusicOperator.ALL_OF,
        resultLimit = FindMusicQuerySpec.DEFAULT_RESULT_LIMIT,
    )

    fun reset(@Suppress("UNUSED_PARAMETER") current: FindMusicEditorSnapshot): FindMusicEditorSnapshot =
        defaults()

    private fun positiveCount(
        textIngredients: List<TextIngredientState>,
        songSeeds: List<SongSeedState>,
    ): Int = textIngredients.count { it.isActive && !it.negative } +
        songSeeds.count { it.isActive && !it.negative }

    private fun denied(reason: String) = FindMusicEditorReadiness(
        canSearch = false,
        reason = reason,
    )
}
