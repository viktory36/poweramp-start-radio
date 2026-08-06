package com.powerampstartradio.poweramp

import com.google.gson.Gson
import kotlin.math.abs

internal enum class PowerampBroadcastDisposition {
    AUTHENTICATED_EXPLICIT,
    REFRESH_HINT_ONLY,
    REJECT,
}

internal enum class PowerampAuthenticatedStateOrigin {
    LIVE_EXPLICIT,
    PERSISTED_EXPLICIT,
}

/** State learned from a sender-authenticated Poweramp explicit event, never a sticky broadcast. */
internal data class PowerampAuthenticatedState(
    val track: PowerampTrack?,
    val playbackState: Int?,
    val lastEventTimestampMs: Long?,
    val origin: PowerampAuthenticatedStateOrigin,
)

/** Returning from background retires the assumption that the last explicit event is still live. */
internal object PowerampActivityResumePolicy {
    fun requireStickyRevalidation(
        state: PowerampAuthenticatedState?,
    ): PowerampAuthenticatedState? = state?.let {
        if (it.origin == PowerampAuthenticatedStateOrigin.PERSISTED_EXPLICIT) it
        else it.copy(origin = PowerampAuthenticatedStateOrigin.PERSISTED_EXPLICIT)
    }
}

/**
 * Compatibility current-track candidate read from Poweramp's sticky API. Playback state is
 * presentation only: Poweramp can publish STATE_STOPPED while retaining the exact selected track
 * that V1 allowed as a radio seed. Callers must revalidate the track against Poweramp's providers
 * and prove its identity did not change during validation.
 */
internal data class PowerampLegacyStickyCandidate(
    val track: PowerampTrack?,
    val playbackState: Int,
)

internal object PowerampLegacyStickyCandidatePolicy {
    fun fromSticky(
        stickyTrack: PowerampTrack?,
        stickyPlaybackState: Int?,
        fallbackTrack: PowerampTrack? = null,
    ): PowerampLegacyStickyCandidate {
        val state = stickyPlaybackState ?: PowerampHelper.STATE_STOPPED
        require(state in setOf(
            PowerampHelper.STATE_STOPPED,
            PowerampHelper.STATE_PLAYING,
            PowerampHelper.STATE_PAUSED,
        )) { "Poweramp sticky playback state is unavailable or invalid" }
        return PowerampLegacyStickyCandidate(
            track = stickyTrack ?: fallbackTrack.takeUnless {
                state == PowerampHelper.STATE_STOPPED
            },
            playbackState = state,
        )
    }

    fun unchanged(
        before: PowerampLegacyStickyCandidate,
        after: PowerampLegacyStickyCandidate,
    ): Boolean = before.track == after.track
}

internal object PowerampExplicitEventStateMachine {
    fun trackChanged(
        previous: PowerampAuthenticatedState?,
        track: PowerampTrack,
        eventTimestampMs: Long?,
    ): PowerampAuthenticatedState {
        if (isOlder(previous, eventTimestampMs)) return requireNotNull(previous)
        return PowerampAuthenticatedState(
            track = track,
            playbackState = previous?.playbackState
                ?.takeUnless { it == PowerampHelper.STATE_STOPPED },
            lastEventTimestampMs = mergedTimestamp(previous, eventTimestampMs),
            origin = PowerampAuthenticatedStateOrigin.LIVE_EXPLICIT,
        )
    }

    fun statusChanged(
        previous: PowerampAuthenticatedState?,
        playbackState: Int,
        eventTrack: PowerampTrack?,
        eventTimestampMs: Long?,
    ): PowerampAuthenticatedState {
        if (isOlder(previous, eventTimestampMs)) return requireNotNull(previous)
        val nextTrack = if (playbackState == PowerampHelper.STATE_STOPPED) {
            null
        } else {
            eventTrack ?: previous?.track
        }
        val nextOrigin = if (
            playbackState != PowerampHelper.STATE_STOPPED &&
            eventTrack == null &&
            previous?.track != null
        ) {
            // Status broadcasts authenticate playback state, not the retained track identity.
            previous.origin
        } else {
            PowerampAuthenticatedStateOrigin.LIVE_EXPLICIT
        }
        return PowerampAuthenticatedState(
            track = nextTrack,
            playbackState = playbackState,
            lastEventTimestampMs = mergedTimestamp(previous, eventTimestampMs),
            origin = nextOrigin,
        )
    }

    private fun isOlder(
        previous: PowerampAuthenticatedState?,
        eventTimestampMs: Long?,
    ): Boolean = previous?.lastEventTimestampMs != null &&
        eventTimestampMs != null && eventTimestampMs < previous.lastEventTimestampMs

    private fun mergedTimestamp(
        previous: PowerampAuthenticatedState?,
        eventTimestampMs: Long?,
    ): Long? = listOfNotNull(previous?.lastEventTimestampMs, eventTimestampMs).maxOrNull()
}

/** Pure identity checks shared by process-restart admission and provider revalidation. */
internal object PowerampCurrentTrackIdentityPolicy {
    private const val LEGACY_DURATION_TOLERANCE_MS = 1_100

    fun persistedStateMatchesSticky(
        persisted: PowerampAuthenticatedState,
        stickyTrack: PowerampTrack?,
        stickyPlaybackState: Int?,
    ): Boolean {
        if (persisted.origin != PowerampAuthenticatedStateOrigin.PERSISTED_EXPLICIT) return false
        return authenticatedStateMatchesSticky(persisted, stickyTrack, stickyPlaybackState)
    }

    fun authenticatedStateMatchesSticky(
        authenticated: PowerampAuthenticatedState,
        stickyTrack: PowerampTrack?,
        stickyPlaybackState: Int?,
    ): Boolean {
        if (stickyPlaybackState == null || authenticated.playbackState != stickyPlaybackState) {
            return false
        }
        if (stickyPlaybackState == PowerampHelper.STATE_STOPPED) {
            return authenticated.track == null && stickyTrack == null
        }
        return sameRecordingIdentity(authenticated.track, stickyTrack)
    }

    fun matchesProvider(authenticated: PowerampTrack, provider: PowerampFileEntry): Boolean {
        if (authenticated.realId <= 0L || authenticated.realId != provider.id) return false
        val authenticatedPath = authenticated.path?.let(TrackNormalization::normalizePath)
            ?.takeIf { it.isNotBlank() }
            ?: return false
        val providerPath = provider.path?.let(TrackNormalization::normalizePath)
            ?.takeIf { it.isNotBlank() }
            ?: return false
        if (authenticatedPath != providerPath) return false
        if (TrackNormalization.normalizeTitle(authenticated.title) !=
            TrackNormalization.normalizeTitle(provider.title)
        ) {
            return false
        }
        if (authenticated.durationMs > 0 && provider.durationMs > 0 &&
            abs(authenticated.durationMs.toLong() - provider.durationMs.toLong()) >
            LEGACY_DURATION_TOLERANCE_MS
        ) {
            return false
        }
        // ID, normalized full path, and title identify the exact library item. Other display
        // metadata may legitimately differ, such as an empty album versus "Unknown album".
        return true
    }

    private fun sameRecordingIdentity(left: PowerampTrack?, right: PowerampTrack?): Boolean {
        if (left == null || right == null || left.realId <= 0L || left.realId != right.realId) {
            return false
        }
        val leftPath = left.path?.let(TrackNormalization::normalizePath)?.takeIf { it.isNotBlank() }
        val rightPath = right.path?.let(TrackNormalization::normalizePath)?.takeIf { it.isNotBlank() }
        if (leftPath == null || rightPath == null || leftPath != rightPath) return false
        if (left.durationMs > 0 && right.durationMs > 0 &&
            abs(left.durationMs.toLong() - right.durationMs.toLong()) >
            LEGACY_DURATION_TOLERANCE_MS
        ) {
            return false
        }
        val leftQueueId = left.queueOccurrenceId
        val rightQueueId = right.queueOccurrenceId
        if ((leftQueueId != null || rightQueueId != null) && leftQueueId != rightQueueId) return false
        return true
    }
}

/** Converts Poweramp event evidence into provider-owned command input. */
internal object PowerampCommandTrackPolicy {
    fun requireProviderBacked(
        authenticated: PowerampAuthenticatedState,
        providerEntries: List<PowerampFileEntry>,
        queueEntries: List<QueueEntry>?,
    ): PowerampTrack? = requireProviderBacked(
        track = authenticated.track,
        playbackState = authenticated.playbackState,
        providerEntries = providerEntries,
        queueEntries = queueEntries,
    )

    fun requireLegacyProviderBacked(
        candidate: PowerampLegacyStickyCandidate,
        providerEntries: List<PowerampFileEntry>,
        queueEntries: List<QueueEntry>?,
    ): PowerampTrack? {
        val track = candidate.track ?: return null
        return requireProviderBackedTrack(track, providerEntries, queueEntries)
    }

    private fun requireProviderBacked(
        track: PowerampTrack?,
        playbackState: Int?,
        providerEntries: List<PowerampFileEntry>,
        queueEntries: List<QueueEntry>?,
    ): PowerampTrack? {
        if (playbackState == PowerampHelper.STATE_STOPPED) {
            require(track == null) {
                "Poweramp stopped state retained a current track"
            }
            return null
        }
        val eventTrack = requireNotNull(track) {
            "Poweramp has not exposed a current-track identity"
        }
        return requireProviderBackedTrack(eventTrack, providerEntries, queueEntries)
    }

    private fun requireProviderBackedTrack(
        eventTrack: PowerampTrack,
        providerEntries: List<PowerampFileEntry>,
        queueEntries: List<QueueEntry>?,
    ): PowerampTrack {
        val provider = providerEntries.singleOrNull { it.id == eventTrack.realId }
            ?: throw IllegalStateException(
                "Current Poweramp track is absent or duplicated in the provider snapshot",
            )
        require(PowerampCurrentTrackIdentityPolicy.matchesProvider(eventTrack, provider)) {
            "Poweramp provider identity no longer matches the current track"
        }

        val claimsQueue = PowerampQueueOccurrencePolicy.isQueueCategory(eventTrack.categoryUri)
        val queueId = eventTrack.queueOccurrenceId
        val queueIdsAreUnique = queueEntries
            ?.let { entries -> entries.mapTo(hashSetOf()) { it.queueId }.size == entries.size }
            ?: false
        val queueOccurrence = if (claimsQueue && queueId != null && queueIdsAreUnique) {
            queueEntries?.singleOrNull { it.queueId == queueId }
                ?.takeIf { it.fileId == provider.id }
        } else {
            null
        }

        return PowerampTrack(
            realId = provider.id,
            // The provider snapshot proves identity; the authenticated event retains display case.
            title = eventTrack.title,
            artist = eventTrack.artist,
            album = eventTrack.album,
            durationMs = provider.durationMs,
            path = provider.path,
            // A stale or unavailable Queue occurrence only disables anchor preservation. The
            // exact provider-backed recording remains a valid radio seed, matching V1 behavior.
            trackId = queueOccurrence?.queueId ?: -1L,
            categoryUri = queueOccurrence?.let { eventTrack.categoryUri },
            positionInList = queueOccurrence?.let { eventTrack.positionInList },
        )
    }
}

internal data class PersistedPowerampAuthenticatedState(
    val schemaVersion: Int,
    val track: PowerampTrack?,
    val playbackState: Int?,
    val lastEventTimestampMs: Long?,
)

internal object PowerampAuthenticatedStateCodec {
    private const val SCHEMA_VERSION = 1
    private val gson = Gson()

    fun encode(state: PowerampAuthenticatedState): String {
        // A status-only explicit event may update state while retaining track evidence loaded
        // from this record. Its origin remains persisted so every command still takes the sticky
        // confirmation path after the process boundary.
        validate(state.track, state.playbackState, state.lastEventTimestampMs)
        return gson.toJson(
            PersistedPowerampAuthenticatedState(
                schemaVersion = SCHEMA_VERSION,
                track = state.track,
                playbackState = state.playbackState,
                lastEventTimestampMs = state.lastEventTimestampMs,
            ),
        )
    }

    fun decode(json: String): PowerampAuthenticatedState {
        val stored = gson.fromJson(json, PersistedPowerampAuthenticatedState::class.java)
            ?: throw IllegalArgumentException("Authenticated Poweramp evidence is empty")
        require(stored.schemaVersion == SCHEMA_VERSION) {
            "Unsupported authenticated Poweramp evidence schema"
        }
        validate(stored.track, stored.playbackState, stored.lastEventTimestampMs)
        return PowerampAuthenticatedState(
            track = stored.track,
            playbackState = stored.playbackState,
            lastEventTimestampMs = stored.lastEventTimestampMs,
            origin = PowerampAuthenticatedStateOrigin.PERSISTED_EXPLICIT,
        )
    }

    private fun validate(track: PowerampTrack?, playbackState: Int?, timestampMs: Long?) {
        require(playbackState == null || playbackState in setOf(
            PowerampHelper.STATE_STOPPED,
            PowerampHelper.STATE_PLAYING,
            PowerampHelper.STATE_PAUSED,
        )) { "Authenticated Poweramp playback state is invalid" }
        require(timestampMs == null || timestampMs >= 0L) {
            "Authenticated Poweramp event timestamp is invalid"
        }
        if (playbackState == PowerampHelper.STATE_STOPPED) {
            require(track == null) { "Stopped authenticated Poweramp state contains a track" }
        } else if (track != null) {
            require(track.realId > 0L && track.title.isNotBlank()) {
                "Authenticated Poweramp track identity is invalid"
            }
            require(!track.path.isNullOrBlank()) {
                "Authenticated Poweramp track path is unavailable"
            }
            require(track.durationMs >= 0) { "Authenticated Poweramp duration is invalid" }
            if (PowerampQueueOccurrencePolicy.isQueueCategory(track.categoryUri)) {
                require(track.queueOccurrenceId != null) {
                    "Authenticated Queue track has no occurrence identity"
                }
            }
        }
    }
}
