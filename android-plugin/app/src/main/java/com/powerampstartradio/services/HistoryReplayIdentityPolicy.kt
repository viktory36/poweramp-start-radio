package com.powerampstartradio.services

import com.powerampstartradio.data.EmbeddedTrack
import com.powerampstartradio.ui.RadioSeedIdentity

internal data class SavedHistoryReplayIdentity(
    val track: EmbeddedTrack,
    val stableTrackSpanId: String?,
    val resolvedPowerampFileId: Long,
) {
    val embeddedTrackId: Long get() = track.id
}

/** Pure fail-closed identity policy for replaying a saved queue against today's library. */
internal object HistoryReplayIdentityPolicy {
    fun resolve(
        savedRows: List<SavedHistoryReplayIdentity>,
        /** Returns a row only when the caller has authenticated every required exact binding. */
        resolveAuthenticatedExactRow: (SavedHistoryReplayIdentity) -> Long?,
        resolveFullContentStableSpan: (SavedHistoryReplayIdentity) -> Long?,
    ): List<RadioSeedIdentity> = savedRows.mapIndexed { index, saved ->
        val exactTrackId = resolveAuthenticatedExactRow(saved)
        if (exactTrackId != null) {
            RadioSeedIdentity(exactTrackId, saved.stableTrackSpanId)
        } else {
            val stableId = requireNotNull(saved.stableTrackSpanId) {
                "Saved occurrence ${index + 1} no longer matches its exact delivered track and " +
                    "has no byte-identical indexed source-span identity"
            }
            val trackId = requireNotNull(resolveFullContentStableSpan(saved)) {
                "Saved occurrence ${index + 1} has no matching byte-identical indexed source span " +
                    "in the active generation"
            }
            RadioSeedIdentity(trackId, stableId)
        }
    }
}
