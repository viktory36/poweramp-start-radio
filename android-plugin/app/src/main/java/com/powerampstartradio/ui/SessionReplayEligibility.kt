package com.powerampstartradio.ui

import com.powerampstartradio.data.EmbeddedTrack
import com.powerampstartradio.indexing.V2ActiveLibraryCatalog
import com.powerampstartradio.similarity.ActiveRecommendationDomain

data class SessionReplayEligibility(
    val eligible: Boolean,
    val reason: String?,
) {
    companion object {
        val CHECKING = SessionReplayEligibility(
            false,
            "Verifying this session's tracks still exist in Poweramp...",
        )
    }
}

internal data class CurrentReplayTrackIdentity(
    val track: EmbeddedTrack?,
    val stableTrackSpanId: String?,
) {
    val existsInGeneration: Boolean get() = track != null
}

/** One shared exact-row contract for the UI preflight and the service mutation boundary. */
internal object ReplayExactRowAuthenticationPolicy {
    fun isAuthenticated(
        sameGeneration: Boolean,
        sameProviderGeneration: Boolean,
        savedTrack: EmbeddedTrack,
        savedStableTrackSpanId: String?,
        savedPowerampFileId: Long,
        currentTrack: EmbeddedTrack?,
        currentStableTrackSpanId: String?,
        currentPowerampFileId: Long?,
    ): Boolean {
        if (!sameGeneration && !sameProviderGeneration) return false
        if (currentTrack != savedTrack || currentStableTrackSpanId != savedStableTrackSpanId) {
            return false
        }
        return sameGeneration || currentPowerampFileId == savedPowerampFileId
    }
}

/** Deterministically selects one active byte-identical equivalent in both replay layers. */
internal object StableReplayTrackSelectionPolicy {
    fun select(
        equivalentTrackIds: List<Long>,
        savedPowerampFileId: Long,
        preferSavedPowerampOccurrence: Boolean,
        currentPowerampFileId: (Long) -> Long?,
    ): Long? {
        var firstActive: Long? = null
        for (trackId in equivalentTrackIds) {
            val powerampFileId = currentPowerampFileId(trackId) ?: continue
            if (firstActive == null) firstActive = trackId
            if (preferSavedPowerampOccurrence && powerampFileId == savedPowerampFileId) {
                return trackId
            }
        }
        return firstActive
    }
}

/**
 * Compact exact projection of the active catalog used by the completed library-readiness pass.
 *
 * Replay availability needs only this ordered ID mapping. Retaining it avoids rebuilding a
 * string-keyed matcher over the complete Poweramp library for every history refresh.
 */
internal class VerifiedReplayLibraryBinding private constructor(
    val generation: RadioGenerationToken,
    val providerGenerationId: String,
    val orderedActiveTrackIdsSha256: String,
    private val orderedTrackIds: LongArray,
    private val orderedPowerampFileIds: LongArray,
) {
    init {
        require(providerGenerationId.isNotBlank()) { "provider generation is blank" }
        require(orderedActiveTrackIdsSha256.isNotBlank()) {
            "ordered active track ID hash is blank"
        }
        require(orderedTrackIds.size == orderedPowerampFileIds.size) {
            "replay binding arrays differ in size"
        }
        for (index in orderedTrackIds.indices) {
            require(orderedTrackIds[index] > 0L && orderedPowerampFileIds[index] > 0L) {
                "replay binding contains a non-positive ID"
            }
            if (index > 0) {
                require(orderedTrackIds[index - 1] < orderedTrackIds[index]) {
                    "replay binding track IDs are not strictly ordered"
                }
            }
        }
    }

    val activeTrackCount: Int get() = orderedTrackIds.size

    fun powerampFileIdForTrack(trackId: Long): Long? {
        val index = orderedTrackIds.binarySearch(trackId)
        return if (index >= 0) orderedPowerampFileIds[index] else null
    }

    companion object {
        fun from(
            generation: RadioGenerationToken,
            catalog: V2ActiveLibraryCatalog,
        ): VerifiedReplayLibraryBinding {
            require(catalog.generationBinding.databaseGenerationId == generation.generationId) {
                "active catalog and replay generation differ"
            }
            val bindings = catalog.bindings
            val orderedTrackIds = LongArray(bindings.size) { bindings[it].trackId }
            return VerifiedReplayLibraryBinding(
                generation = generation,
                providerGenerationId = catalog.generationBinding.providerGenerationId,
                orderedActiveTrackIdsSha256 =
                    ActiveRecommendationDomain.computeOrderedActiveIdsSha256(orderedTrackIds),
                orderedTrackIds = orderedTrackIds,
                orderedPowerampFileIds = LongArray(bindings.size) {
                    bindings[it].powerampFileId
                },
            )
        }
    }
}

/** Pure fail-closed replay policy shared by UI state and focused tests. */
internal object SessionReplayEligibilityPolicy {
    fun evaluate(
        session: RadioResult,
        activeGeneration: RadioGenerationToken?,
        /** Current IDs resolved from active source-span bindings against one provider snapshot. */
        resolvedCurrentPowerampFileIds: List<Long?>?,
        resolvedByExactEmbeddedRows: List<Boolean>?,
        currentTrackIdentities: Map<Long, CurrentReplayTrackIdentity>,
        activeProviderGenerationId: String? = null,
        /** IDs successfully resolved by the active StableTrackIdentityCatalog (full hash only). */
        availableFullContentStableTrackSpanIds: Set<String>? = null,
    ): SessionReplayEligibility {
        val queuedRows = session.tracks.filter { it.status == QueueStatus.QUEUED }
        if (queuedRows.isEmpty()) {
            return denied(
                "Nothing from this session was confirmed in Poweramp, so there is nothing to queue again.",
            )
        }
        val savedGeneration = session.generation
            ?: return denied(OLDER_SESSION_REASON)
        if (activeGeneration == null) {
            return denied(
                "The music index is not ready. Wait for it to finish loading, then try again.",
            )
        }
        if (queuedRows.any { it.resolvedPowerampFileId == null }) {
            return denied(INCOMPLETE_SESSION_REASON)
        }
        if (queuedRows.any { it.resolvedPowerampQueueId == null }) {
            return denied(INCOMPLETE_SESSION_REASON)
        }
        if (queuedRows.mapNotNull { it.resolvedPowerampQueueId }.toSet().size != queuedRows.size) {
            return denied(INCOMPLETE_SESSION_REASON)
        }
        if (resolvedCurrentPowerampFileIds == null ||
            resolvedCurrentPowerampFileIds.size != queuedRows.size ||
            resolvedByExactEmbeddedRows == null ||
            resolvedByExactEmbeddedRows.size != queuedRows.size
        ) {
            return denied(CURRENT_LIBRARY_MATCH_REASON)
        }
        if (resolvedCurrentPowerampFileIds.any { it == null || it <= 0L }) {
            return denied(CURRENT_LIBRARY_MATCH_REASON)
        }
        val sameGeneration = savedGeneration == activeGeneration
        val sameProviderGeneration = session.providerGenerationId != null &&
            session.providerGenerationId == activeProviderGenerationId
        for ((index, row) in queuedRows.withIndex()) {
            if (resolvedByExactEmbeddedRows[index]) {
                val current = currentTrackIdentities[row.track.id]
                if (!ReplayExactRowAuthenticationPolicy.isAuthenticated(
                        sameGeneration = sameGeneration,
                        sameProviderGeneration = sameProviderGeneration,
                        savedTrack = row.track,
                        savedStableTrackSpanId = row.stableTrackSpanId,
                        savedPowerampFileId = checkNotNull(row.resolvedPowerampFileId),
                        currentTrack = current?.track,
                        currentStableTrackSpanId = current?.stableTrackSpanId,
                        currentPowerampFileId = resolvedCurrentPowerampFileIds[index],
                    )
                ) {
                    return denied(CHANGED_LIBRARY_REASON)
                }
            } else {
                if (sameGeneration) return denied(CHANGED_LIBRARY_REASON)
                val stableId = row.stableTrackSpanId ?: return denied(OLDER_SESSION_REASON)
                if (stableId !in availableFullContentStableTrackSpanIds.orEmpty()) {
                    return denied(CHANGED_LIBRARY_REASON)
                }
            }
        }
        return SessionReplayEligibility(true, null)
    }

    private fun denied(reason: String) = SessionReplayEligibility(false, reason)

    private const val OLDER_SESSION_REASON =
        "This older session cannot be queued again reliably. Start a new radio from the same seed."
    private const val INCOMPLETE_SESSION_REASON =
        "This session is missing information needed to queue it again. " +
            "Start a new radio from the same seed."
    private const val CHANGED_LIBRARY_REASON =
        "The library changed and some tracks in this session are no longer available. " +
            "Start a new radio from the same seed."
    private const val CURRENT_LIBRARY_MATCH_REASON =
        "Some tracks in this session could not be matched to the current Poweramp library. " +
            "Start a new radio from the same seed."
}

internal fun RadioResult.replayEligibilityKey(): String =
    requestId ?: "legacy:$timestamp:${seedTrack.realId}"
