package com.powerampstartradio.services

import com.powerampstartradio.data.StableIdentityGenerationBinding
import com.powerampstartradio.ui.RadioGenerationToken
import com.powerampstartradio.ui.RadioSeedIdentity

/** Fail-closed checks applied before a displayed result may resolve a numeric database row ID. */
internal object DisplayedResultBindingGuard {
    fun requireExactGeneration(
        displayedBinding: StableIdentityGenerationBinding,
        activeGeneration: RadioGenerationToken,
    ) {
        require(
            displayedBinding.generationId == activeGeneration.generationId &&
                displayedBinding.activationBindingId == activeGeneration.activationBindingId &&
                displayedBinding.databaseContentSha256 == activeGeneration.databaseContentSha256 &&
                displayedBinding.orderedTrackSetSha256 == activeGeneration.orderedTrackSetSha256,
        ) { "The displayed Find Music result belongs to a different library generation" }
    }

    fun requireExactTrackIdentity(
        displayedTrackId: Long,
        displayedIdentity: RadioSeedIdentity,
        activeStableTrackSpanId: String?,
    ) {
        require(displayedIdentity.embeddedTrackId == displayedTrackId) {
            "The displayed Find Music row and its pinned identity disagree"
        }
        require(displayedIdentity.stableTrackSpanId == activeStableTrackSpanId) {
            "The displayed Find Music track identity changed inside the active generation"
        }
    }
}
