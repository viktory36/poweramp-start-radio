package com.powerampstartradio.ui

/** Current evidence for one generation-pinned row in a displayed Find Music ranking. */
internal data class CurrentDisplayedFindMusicTrack(
    val existsInGeneration: Boolean,
    val stableTrackSpanId: String?,
    val activePowerampFileId: Long?,
)

data class DisplayedFindMusicQueueEligibility(
    val eligible: Boolean,
    val reason: String?,
) {
    companion object {
        val UNAVAILABLE = DisplayedFindMusicQueueEligibility(
            false,
            "Run Find Music to create a queueable ranking.",
        )
        val CHECKING = DisplayedFindMusicQueueEligibility(
            false,
            "Verifying these results still exist in Poweramp...",
        )
    }
}

/** Fail-closed UI policy mirroring the generation, row-identity, and occurrence evidence in replay. */
internal object DisplayedFindMusicQueueEligibilityPolicy {
    fun evaluate(
        result: TextSearchResult,
        activeGeneration: RadioGenerationToken?,
        activeProviderGenerationId: String?,
        orderedActiveTrackIdsSha256: String?,
        activeTrackCount: Int?,
        currentTracks: Map<Long, CurrentDisplayedFindMusicTrack>,
    ): DisplayedFindMusicQueueEligibility {
        val binding = result.libraryBinding
        val querySpec = result.querySpec
        if (result.error != null || result.matches.isEmpty() || binding == null ||
            querySpec == null || querySpec.libraryBinding != binding ||
            result.stableResultReduction == null ||
            validateFindMusicQueryContract(querySpec) != null ||
            result.matches.any { it.objectiveRank == null } ||
            result.providerGenerationId.isNullOrBlank() ||
            result.orderedActiveTrackIdsSha256?.matches(SHA256) != true ||
            (result.activeTrackCount ?: 0) <= 0
        ) {
            return denied(REFRESH_RESULT_REASON)
        }
        val active = activeGeneration
            ?: return denied(
                "The music index is not ready. Wait for it to finish loading, " +
                    "then run Find Music again.",
            )
        if (binding.generationId != active.generationId ||
            binding.activationBindingId != active.activationBindingId ||
            binding.databaseContentSha256 != active.databaseContentSha256 ||
            binding.orderedTrackSetSha256 != active.orderedTrackSetSha256
        ) {
            return changedLibrary()
        }
        if (result.providerGenerationId != activeProviderGenerationId ||
            result.orderedActiveTrackIdsSha256 != orderedActiveTrackIdsSha256 ||
            result.activeTrackCount != activeTrackCount
        ) {
            return changedLibrary()
        }
        for (match in result.matches) {
            if (match.identity.embeddedTrackId != match.track.id) {
                return denied(REFRESH_RESULT_REASON)
            }
            val current = currentTracks[match.track.id]
            if (current?.existsInGeneration != true ||
                current.stableTrackSpanId != match.identity.stableTrackSpanId ||
                current.activePowerampFileId == null || current.activePowerampFileId <= 0L
            ) {
                return changedLibrary()
            }
        }
        return DisplayedFindMusicQueueEligibility(eligible = true, reason = null)
    }

    private fun changedLibrary() = denied(
        "The library changed since this ranking was computed. Run Find Music again.",
    )

    private fun denied(reason: String) = DisplayedFindMusicQueueEligibility(false, reason)

    private const val REFRESH_RESULT_REASON =
        "This result is no longer current. Run Find Music again before queueing it."

    private val SHA256 = Regex("^[0-9a-f]{64}$")
}
