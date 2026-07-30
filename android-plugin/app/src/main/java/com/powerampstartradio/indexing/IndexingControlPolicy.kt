package com.powerampstartradio.indexing

import com.powerampstartradio.data.EmbeddedTrack
import com.powerampstartradio.indexing.v2.IndexingJobState
import com.powerampstartradio.indexing.v2.IndexingTrackState
import com.powerampstartradio.indexing.v2.RetryTrigger
import com.powerampstartradio.indexing.v2.TrackFailureCode
import com.powerampstartradio.indexing.v2.V2IndexingPreflightFailureCode
import com.powerampstartradio.indexing.v2.V2IndexingExecutorEvent
import com.powerampstartradio.indexing.v2.V2IndexingProgressSnapshot
import java.util.Locale

/** Job states where every per-track attention action has a stable, non-racing meaning. */
internal fun canOfferIndexingFailureActions(state: IndexingJobState): Boolean = state in setOf(
    IndexingJobState.PAUSED,
    IndexingJobState.WAITING_FOR_INPUT,
    IndexingJobState.INTERRUPTED,
    IndexingJobState.READY_TO_RESUME,
)

internal fun isActionableIndexingFailure(
    jobState: IndexingJobState,
    trackState: IndexingTrackState,
    hasFailureEvidence: Boolean,
): Boolean = isVisibleUnresolvedIndexingFailure(trackState, hasFailureEvidence) &&
    canOfferIndexingFailureActions(jobState)

/** A terminal job still owes the user an explanation for every track it did not index. */
internal fun isVisibleUnresolvedIndexingFailure(
    trackState: IndexingTrackState,
    hasFailureEvidence: Boolean,
): Boolean = hasFailureEvidence && trackState in setOf(
        IndexingTrackState.RETRYABLE_FAILURE,
        IndexingTrackState.BLOCKED_FAILURE,
    )

internal fun canUserRetryIndexingFailure(
    state: IndexingJobState,
    retryTrigger: RetryTrigger,
): Boolean = canOfferIndexingFailureActions(state) &&
    retryTrigger != RetryTrigger.NEW_JOB_REQUIRED &&
    retryTrigger != RetryTrigger.NEVER

/** Terminal ledgers are immutable; retrying means explicitly selecting the track for a new job. */
internal fun canSelectIndexingFailureForNewRun(
    state: IndexingJobState,
    retryTrigger: RetryTrigger,
    failureCode: TrackFailureCode,
    sourceIdentityChanged: Boolean = false,
): Boolean = state in setOf(IndexingJobState.COMPLETE, IndexingJobState.CANCELLED) &&
    retryTrigger != RetryTrigger.NEVER &&
    (failureCode != TrackFailureCode.CONTAINER_EOS_MISMATCH || sourceIdentityChanged)

internal fun canNeverIndexFailure(state: IndexingJobState): Boolean =
    canOfferIndexingFailureActions(state) ||
        state in setOf(IndexingJobState.COMPLETE, IndexingJobState.CANCELLED)

internal fun hasUserRetryEligibleFailure(
    state: IndexingJobState,
    retryTriggers: Iterable<RetryTrigger>,
): Boolean = retryTriggers.any { canUserRetryIndexingFailure(state, it) }

/** A stale activity/notification command must never race a live executor. */
internal fun canAcceptIndexingFailureCommand(
    state: IndexingJobState,
    runnerActive: Boolean,
): Boolean = !runnerActive && canOfferIndexingFailureActions(state)

internal fun shouldOfferIndexingPauseAction(state: IndexingJobState): Boolean =
    state == IndexingJobState.RUNNING

internal fun canSelectReadyTracks(
    selectableVisibleIds: Set<Long>,
    selectedIds: Set<Long>,
): Boolean = selectableVisibleIds.any { it !in selectedIds }

internal fun matchesIndexingTrackSearch(query: String, vararg fields: String?): Boolean {
    val needle = query.trim()
    return needle.isEmpty() || fields.any { field ->
        field?.contains(needle, ignoreCase = true) == true
    }
}

internal fun formatFailureOccurrences(occurrences: Int): String {
    require(occurrences > 0) { "failure occurrences must be positive" }
    return if (occurrences == 1) "Failed once" else "Failed $occurrences times"
}

internal fun indexingFailureSummary(code: TrackFailureCode): String = when (code) {
    TrackFailureCode.SOURCE_MISSING -> "Source file is missing"
    TrackFailureCode.STORAGE_UNMOUNTED -> "Music storage is unavailable"
    TrackFailureCode.SOURCE_UNREADABLE -> "Source file cannot be read"
    TrackFailureCode.POWERAMP_PROVIDER_UNAVAILABLE -> "Poweramp library is unavailable"
    TrackFailureCode.ANDROID_AUDIO_PERMISSION_DENIED -> "Music access was denied"
    TrackFailureCode.POWERAMP_PERMISSION_DENIED -> "Poweramp library access was denied"
    TrackFailureCode.INVALID_LOGICAL_SPAN ->
        "Poweramp's saved timing does not describe playable audio"
    TrackFailureCode.CUE_SOURCE_IMAGE -> "This is a raw CUE source image"
    TrackFailureCode.SOURCE_FINGERPRINT_CHANGED -> "Source file changed during indexing"
    TrackFailureCode.PROVIDER_SNAPSHOT_CHANGED -> "Poweramp library changed during indexing"
    TrackFailureCode.CONTAINER_EOS_MISMATCH ->
        "Decoded audio ended at a different length than its container declares"
    TrackFailureCode.NO_AUDIO_STREAM -> "No audio stream was found"
    TrackFailureCode.UNSUPPORTED_CODEC_OR_CONTAINER -> "Codec or container is unsupported"
    TrackFailureCode.DRM_PROTECTED -> "Audio is DRM protected"
    TrackFailureCode.CORRUPT_OR_TRUNCATED -> "Audio appears corrupt or truncated"
    TrackFailureCode.DECODER_ERROR -> "Audio decoding failed"
    TrackFailureCode.BELOW_MINIMUM_DURATION -> "Audio is too short to embed"
    TrackFailureCode.OUT_OF_MEMORY -> "The device ran out of memory"
    TrackFailureCode.THERMAL_SHUTDOWN -> "Indexing stopped for device temperature"
    TrackFailureCode.PROCESS_INTERRUPTED -> "Android interrupted indexing"
    TrackFailureCode.STAGE_TIMEOUT -> "An indexing stage timed out"
    TrackFailureCode.MODEL_LOAD_FAILED -> "Embedding model could not be loaded"
    TrackFailureCode.INFERENCE_FAILED -> "Embedding inference failed"
    TrackFailureCode.INVALID_MODEL_OUTPUT -> "Embedding model returned invalid output"
    TrackFailureCode.PARTIAL_ARTIFACT -> "Saved intermediate work is incomplete"
    TrackFailureCode.ARTIFACT_CHECKSUM_MISMATCH -> "Saved work is damaged or incomplete"
    TrackFailureCode.STORAGE_FULL -> "Device storage is full"
    TrackFailureCode.DATABASE_BUSY -> "Music index is busy"
    TrackFailureCode.DATABASE_GENERATION_CHANGED -> "Active music index changed"
    TrackFailureCode.IMPORTED_ROW_AUTHORIZATION_CHANGED ->
        "Source identity changed before the embedding could be saved"
    TrackFailureCode.COMMIT_FAILED -> "Embedding could not be saved"
    TrackFailureCode.UNKNOWN_TRANSIENT -> "Indexing failed for an unclassified temporary reason"
    TrackFailureCode.UNKNOWN_BLOCKED -> "Indexing is blocked by an unclassified error"
}

internal fun indexingRetryGuidance(trigger: RetryTrigger): String = when (trigger) {
    RetryTrigger.IMMEDIATE -> "This track is ready to retry now."
    RetryTrigger.PROCESS_RESTART -> "Restart the app before retrying this track."
    RetryTrigger.SOURCE_AVAILABLE -> "Retry after the source file or music storage is readable."
    RetryTrigger.PERMISSION_GRANTED -> "Grant the required access before retrying."
    RetryTrigger.RESOURCE_RECOVERED ->
        "Retry after memory or device-temperature pressure has cleared."
    RetryTrigger.STORAGE_RECOVERED ->
        "Free storage or wait for the current music-index operation to finish before retrying."
    RetryTrigger.SOURCE_OR_LIBRARY_CHANGED ->
        "Retry after the source file or Poweramp library entry has been corrected."
    RetryTrigger.DECODER_OR_APP_CHANGED ->
        "Retry after an app or decoder update that can read this audio."
    RetryTrigger.USER_REQUEST -> "Retry when you want to test this track again."
    RetryTrigger.NEW_JOB_REQUIRED -> "Select this track in a new indexing run."
    RetryTrigger.NEVER -> "This saved run cannot retry this failure."
}

internal fun indexingFailureGuidance(
    code: TrackFailureCode,
    retryTrigger: RetryTrigger,
): String = when (code) {
    TrackFailureCode.CONTAINER_EOS_MISMATCH ->
        "The file is likely corrupt or truncated. Repair or replace it before indexing again."
    else -> indexingRetryGuidance(retryTrigger)
}

internal fun preflightRejectionSummary(code: V2IndexingPreflightFailureCode): String = when (code) {
    V2IndexingPreflightFailureCode.EMPTY_SELECTION -> "No tracks were selected"
    V2IndexingPreflightFailureCode.DUPLICATE_POWERAMP_ROW ->
        "The selection contains the same library item more than once"
    V2IndexingPreflightFailureCode.INVALID_SELECTION_EVIDENCE ->
        "Poweramp did not provide a complete playable track entry"
    V2IndexingPreflightFailureCode.INVALID_LOGICAL_SPAN ->
        "Poweramp's saved start or duration does not describe valid audio"
    V2IndexingPreflightFailureCode.CUE_SOURCE_IMAGE ->
        "This is a raw CUE source image rather than a playable track"
    V2IndexingPreflightFailureCode.AUDIO_TOO_SHORT ->
        "This track is too short to create a music embedding"
    V2IndexingPreflightFailureCode.SOURCE_UNREADABLE ->
        "The source file is not currently readable"
    V2IndexingPreflightFailureCode.NO_AUDIO_STREAM ->
        "No audio stream was found in the source file"
    V2IndexingPreflightFailureCode.UNSUPPORTED_OR_INVALID_AUDIO_CONTAINER ->
        "Android could not read this audio container"
    V2IndexingPreflightFailureCode.SOURCE_CHANGED ->
        "The source changed while its exact content hash was being recorded"
    V2IndexingPreflightFailureCode.SOURCE_CANONICAL_ALIAS_COLLISION ->
        "Multiple selected paths resolved to one conflicting source"
    V2IndexingPreflightFailureCode.PROVIDER_SNAPSHOT_INVALID ->
        "The current Poweramp library could not be read completely"
    V2IndexingPreflightFailureCode.MODEL_UNREADABLE ->
        "A required music model file is missing or unreadable. Review file details in Settings"
    V2IndexingPreflightFailureCode.APP_ARTIFACT_UNREADABLE ->
        "A required indexing file is missing or unreadable. Review file details in Settings"
    V2IndexingPreflightFailureCode.INSUFFICIENT_STORAGE ->
        "There is not enough storage for this indexing run"
    V2IndexingPreflightFailureCode.INVALID_PLAN ->
        "This indexing request could not be prepared. Select the tracks again"
    V2IndexingPreflightFailureCode.PERSISTENCE_FAILED ->
        "The indexing request could not be saved safely"
}

internal fun formatDurableTrackCounts(
    state: IndexingJobState,
    progress: V2IndexingProgressSnapshot,
    locale: Locale = Locale.getDefault(),
): String {
    if (state == IndexingJobState.CANCELLED) {
        return "No tracks were added; unfinished work was discarded"
    }
    if (progress.totalTracks > 0 &&
        progress.succeededTracks == progress.totalTracks &&
        progress.blockedTracks == 0 &&
        progress.skippedTracks == 0
    ) {
        val count = formatIndexingTrackCount(progress.totalTracks, locale)
        return if (state == IndexingJobState.COMPLETE) {
            val noun = if (progress.totalTracks == 1) "track" else "tracks"
            "$count $noun indexed"
        } else {
            "$count embeddings saved"
        }
    }
    val parts = buildList {
        val succeeded = formatIndexingTrackCount(progress.succeededTracks, locale)
        val total = formatIndexingTrackCount(progress.totalTracks, locale)
        add(
            if (state == IndexingJobState.COMPLETE) {
                "$succeeded of $total indexed"
            } else {
                "$succeeded of $total embeddings saved"
            },
        )
        if (progress.blockedTracks > 0) {
            val blocked = formatIndexingTrackCount(progress.blockedTracks, locale)
            add(
                if (progress.blockedTracks == 1) {
                    "$blocked needs attention"
                } else {
                    "$blocked need attention"
                },
            )
        }
        if (progress.skippedTracks > 0) {
            add("${formatIndexingTrackCount(progress.skippedTracks, locale)} skipped")
        }
    }
    return parts.joinToString(" \u00b7 ")
}

/** Verified per-track artifacts expose useful saved progress before any track fully finishes. */
internal fun formatDurableStageTrackCounts(
    state: IndexingJobState,
    progress: V2IndexingProgressSnapshot,
    locale: Locale = Locale.getDefault(),
): String? {
    if (progress.totalTracks <= 0 ||
        state in setOf(IndexingJobState.COMPLETE, IndexingJobState.CANCELLED)
    ) return null

    val total = formatIndexingTrackCount(progress.totalTracks, locale)
    val parts = buildList {
        if (progress.tracksWithMertFeatures > 0) {
            add(
                "Saved audio-analysis checkpoints: " +
                    "${formatIndexingTrackCount(progress.tracksWithMertFeatures, locale)} of $total",
            )
        }
        if (progress.tracksWithMertFeatures >=
            progress.totalTracks - progress.blockedTracks - progress.skippedTracks ||
            progress.tracksWithClampVectors > 0
        ) {
            add(
                "Saved music embeddings: " +
                    "${formatIndexingTrackCount(progress.tracksWithClampVectors, locale)} of $total",
            )
        }
    }
    return parts.takeIf { it.isNotEmpty() }?.joinToString(" \u00b7 ")
}

/** A completed per-track event must not masquerade as the current whole-library work. */
internal fun shouldShowIndexingStageEvent(
    event: V2IndexingExecutorEvent,
    progress: V2IndexingProgressSnapshot,
): Boolean = event.workId == null || (
    progress.activeTrackOrdinal != null &&
        event.trackOrdinal == progress.activeTrackOrdinal
    )

internal fun indexingStageFallbackText(
    state: IndexingJobState,
    progress: V2IndexingProgressSnapshot,
    hasVisibleStageEvent: Boolean,
): String? {
    if (hasVisibleStageEvent) return null
    return when (state) {
        IndexingJobState.RUNNING ->
            "Indexing run \u00b7 Restoring the saved checkpoint and current operation"
        IndexingJobState.PAUSE_REQUESTED ->
            "Indexing run \u00b7 Finishing the current file or model operation before pausing"
        IndexingJobState.ACTIVATING ->
            "Whole library \u00b7 Restoring the saved publication checkpoint"
        IndexingJobState.CANCELLING ->
            "Indexing run \u00b7 Removing unfinished staging files"
        else -> null
    }
}

internal fun formatCurrentIndexingTrack(
    activeTrackOrdinal: Int?,
    eventTrackOrdinal: Int?,
    eventTrackTitle: String?,
    totalTracks: Int,
): String? {
    if (totalTracks <= 0) return null
    val ordinal = activeTrackOrdinal ?: return null
    if (ordinal !in 0 until totalTracks) return null
    if (eventTrackOrdinal != ordinal) return null
    val title = eventTrackTitle
        ?.trim()
        ?.takeIf(String::isNotEmpty)
        ?: return null
    return "Track ${formatIndexingTrackCount(ordinal + 1)} of " +
        "${formatIndexingTrackCount(totalTracks)} \u00b7 $title"
}

internal fun formatPreflightAttentionSummary(count: Int): String? {
    require(count >= 0) { "preflight attention count cannot be negative" }
    if (count == 0) return null
    val formatted = formatIndexingTrackCount(count)
    return if (count == 1) {
        "$formatted selected track could not start indexing; review it below."
    } else {
        "$formatted selected tracks could not start indexing; review them below."
    }
}

internal fun shouldShowIndexingEta(state: IndexingJobState): Boolean = state in setOf(
    IndexingJobState.RUNNING,
    IndexingJobState.PAUSE_REQUESTED,
    IndexingJobState.ACTIVATING,
)

internal sealed interface DatabaseCleanupScanState {
    data object Idle : DatabaseCleanupScanState
    data class Scanning(val message: String) : DatabaseCleanupScanState
    data class Ready(
        val tracks: List<EmbeddedTrack>,
        val message: String,
    ) : DatabaseCleanupScanState
    data class Failed(val message: String) : DatabaseCleanupScanState
}

internal enum class IndexingUiOperation {
    JOB_PLANNING,
    EXPORT,
}

/** Owns the admission gap before asynchronous planning or export work starts. */
internal class IndexingUiOperationAdmission {
    private var active: IndexingUiOperation? = null

    @Synchronized
    fun tryAcquire(operation: IndexingUiOperation): Boolean {
        if (active != null) return false
        active = operation
        return true
    }

    @Synchronized
    fun release(operation: IndexingUiOperation): Boolean {
        if (active != operation) return false
        active = null
        return true
    }

    @Synchronized
    fun isOwnedBy(operation: IndexingUiOperation): Boolean = active == operation
}
