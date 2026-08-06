package com.powerampstartradio.indexing

import com.powerampstartradio.indexing.v2.V2IndexingEtaCoverageSnapshot
import com.powerampstartradio.indexing.v2.V2IndexingEventScope
import com.powerampstartradio.indexing.v2.V2IndexingExecutorEvent
import com.powerampstartradio.indexing.v2.IndexingJobState
import com.powerampstartradio.indexing.v2.V2IndexingPreflightFailureCode
import com.powerampstartradio.indexing.v2.V2IndexingPreflightIntentState
import com.powerampstartradio.indexing.v2.V2MeasuredWorkStage
import com.powerampstartradio.indexing.v2.V2StageAwareEtaEstimate
import com.powerampstartradio.indexing.v2.V2UnmeasuredIndexingWork
import java.text.NumberFormat
import java.util.Locale
import kotlin.math.roundToInt

internal const val ON_DEVICE_INDEXING_NOTIFICATION_TITLE = "On-device indexing"

internal fun formatIndexingByteCount(bytes: Long): String =
    String.format(Locale.US, "%.1f MiB", bytes / (1024.0 * 1024.0))

internal fun exactHashProgressText(
    subject: String,
    completedBytes: Long,
    totalBytes: Long,
): String = if (completedBytes == 0L) {
    "Opening $subject for SHA-256 verification · " +
        "${formatIndexingByteCount(totalBytes)} total to read"
} else {
    "Hashing $subject · ${formatIndexingByteCount(completedBytes)} of " +
        formatIndexingByteCount(totalBytes)
}

internal fun powerampLibraryReadProgressText(
    completedRows: Int,
    totalRows: Int,
    locale: Locale = Locale.getDefault(),
): String {
    require(completedRows in 0..totalRows && totalRows > 0) {
        "Poweramp row progress is invalid"
    }
    val number = NumberFormat.getIntegerInstance(locale)
    val total = number.format(totalRows)
    return if (completedRows == 0) {
        "Poweramp reports $total library rows; beginning the complete read"
    } else {
        "Read ${number.format(completedRows)} of $total Poweramp library rows"
    }
}

internal enum class IndexingStageScope {
    CURRENT_TRACK,
    INDEXING_RUN,
    WHOLE_LIBRARY,
}

internal data class IndexingStageEvidence(
    val label: String,
    val scope: IndexingStageScope,
    val fraction: Float?,
    val text: String,
)

internal data class IndexingNotificationEvidence(
    val title: String,
    val text: String,
    val fraction: Float?,
)

internal fun indexingStageEvidence(
    event: V2IndexingExecutorEvent?,
): IndexingStageEvidence? {
    event ?: return null
    val fraction = if (
        event.completedUnits != null &&
        event.totalUnits != null &&
        event.totalUnits > 0L &&
        event.completedUnits in 1L..event.totalUnits
    ) {
        (event.completedUnits.toDouble() / event.totalUnits)
            .toFloat()
            .coerceIn(0f, 1f)
    } else {
        null
    }
    val label = event.detail.trim().ifEmpty {
        V2IndexingServiceStateMapper.stageLabel(event.stage)
    }
    val scope = when (event.scope ?: if (event.workId == null) {
        V2IndexingEventScope.WHOLE_LIBRARY
    } else {
        V2IndexingEventScope.CURRENT_TRACK
    }) {
        V2IndexingEventScope.CURRENT_TRACK -> IndexingStageScope.CURRENT_TRACK
        V2IndexingEventScope.INDEXING_RUN -> IndexingStageScope.INDEXING_RUN
        V2IndexingEventScope.WHOLE_LIBRARY -> IndexingStageScope.WHOLE_LIBRARY
    }
    val scopeLabel = when (scope) {
        IndexingStageScope.CURRENT_TRACK -> "Current track"
        IndexingStageScope.INDEXING_RUN -> "Indexing run"
        IndexingStageScope.WHOLE_LIBRARY -> "Whole library"
    }
    val percent = fraction?.let {
        when {
            event.completedUnits == 0L -> null
            it < 0.01f -> "<1%"
            else -> "${(it * 100f).roundToInt()}%"
        }
    }
    val text = "$scopeLabel · $label" + percent?.let { " · $it" }.orEmpty()
    return IndexingStageEvidence(
        label = label,
        scope = scope,
        fraction = fraction,
        text = text,
    )
}

/** Foreground-notification copy reports one local stage, never a synthetic whole-job fraction. */
internal fun indexingNotificationEvidence(
    state: IndexingJobState,
    event: V2IndexingExecutorEvent?,
): IndexingNotificationEvidence {
    val stage = event
        ?.takeIf {
            state in setOf(
                IndexingJobState.RUNNING,
                IndexingJobState.PAUSE_REQUESTED,
                IndexingJobState.ACTIVATING,
            )
        }
        ?.let(::indexingStageEvidence)
    val text = stage?.text ?: when (state) {
        IndexingJobState.PLANNED -> "Ready to begin"
        IndexingJobState.RUNNING -> "Restoring the saved indexing checkpoint and current operation"
        IndexingJobState.PAUSE_REQUESTED -> "Finishing the current step, then pausing"
        IndexingJobState.PAUSED -> "Paused; progress is saved"
        IndexingJobState.WAITING_FOR_INPUT ->
            "Open the app to review tracks that need attention"
        IndexingJobState.INTERRUPTED -> "Interrupted; saved progress can be resumed"
        IndexingJobState.READY_TO_RESUME -> "Ready to resume saved progress"
        IndexingJobState.CANCELLING -> "Discarding unfinished work"
        IndexingJobState.CANCELLED -> "Indexing request cancelled"
        IndexingJobState.ACTIVATING -> "Restoring the saved music-index publication checkpoint"
        IndexingJobState.COMPLETE -> "Music index updated"
    }
    return IndexingNotificationEvidence(
        title = ON_DEVICE_INDEXING_NOTIFICATION_TITLE,
        text = text,
        fraction = stage?.fraction,
    )
}

/** Raw preflight diagnostics remain durable evidence; visible copy is selected by typed state. */
internal fun preflightStatusEvidenceText(
    state: V2IndexingPreflightIntentState,
    failureCode: V2IndexingPreflightFailureCode?,
    progressMessage: String,
): String = when (state) {
    V2IndexingPreflightIntentState.FAILED -> failureCode
        ?.let(::preflightRejectionSummary)
        ?: "The indexing request needs attention"
    V2IndexingPreflightIntentState.PLANNING -> progressMessage
    else -> preflightStateFallbackText(state)
}

private fun preflightStateFallbackText(state: V2IndexingPreflightIntentState): String = when (state) {
    V2IndexingPreflightIntentState.REQUESTED ->
        "Waiting for Android to start the foreground indexing service"
    V2IndexingPreflightIntentState.PLANNING ->
        "Reading Poweramp rows and binding selected tracks to source files"
    V2IndexingPreflightIntentState.INTERRUPTED ->
        "Your selection is saved. Try preparation again when ready"
    V2IndexingPreflightIntentState.FAILED -> "The indexing request needs attention"
    V2IndexingPreflightIntentState.CANCEL_REQUESTED -> "Cancelling the indexing request"
    V2IndexingPreflightIntentState.CANCELLED -> "Indexing request cancelled"
    V2IndexingPreflightIntentState.RESOLVED_WITH_EXECUTABLE_ROWS -> "Saving the indexing job"
    V2IndexingPreflightIntentState.RESOLVED_WITHOUT_EXECUTABLE_ROWS ->
        "No selected tracks can be indexed"
    V2IndexingPreflightIntentState.MATERIALIZED -> "Indexing job ready"
}

internal fun formatIndexingEtaEvidence(
    eta: V2StageAwareEtaEstimate,
    coverage: V2IndexingEtaCoverageSnapshot,
): String? {
    val estimate = formatEtaEstimate(eta)
    if (estimate != null) {
        val early = eta.calibratingStages.isNotEmpty()
        val calibrating = eta.calibratingStages
            .sortedBy { it.ordinal }
            .joinToString(", ") { V2IndexingServiceStateMapper.stageLabel(it) }
        val estimateText = if (early) {
            "Early estimate at current pace: $estimate"
        } else {
            "At current pace: $estimate"
        }
        return if (coverage.coversWholeJob) {
            if (early) {
                "$estimateText remaining.\n" +
                    "Still measuring: $calibrating."
            } else {
                "$estimateText remaining"
            }
        } else {
            val caveat = formatEtaOmittedWork(coverage.omittedRemainingWork)
            if (early) {
                "$estimateText of measured work left.\n" +
                    "Still measuring: $calibrating. $caveat"
            } else {
                "$estimateText of measured work left.\n$caveat"
            }
        }
    }
    if (eta.calibratingStages.isEmpty()) return null
    val calibrating = eta.calibratingStages
        .sortedBy { it.ordinal }
        .joinToString(", ") { V2IndexingServiceStateMapper.stageLabel(it) }
    return "ETA unavailable until timing samples exist for: $calibrating."
}

private fun formatEtaOmittedWork(omitted: Set<V2UnmeasuredIndexingWork>): String {
    val labels = omitted.sortedBy { it.ordinal }.map { work ->
        when (work) {
            V2UnmeasuredIndexingWork.UNKNOWN_DURATION_AUDIO_WORK ->
                "audio work for tracks without known durations"
            V2UnmeasuredIndexingWork.VALIDATION_AND_FINAL_PUBLICATION ->
                "remaining validation and final music-index publication"
            V2UnmeasuredIndexingWork.DERIVED_GRAPH_WITHOUT_STRUCTURED_PLAN ->
                "similarity-graph construction"
            V2UnmeasuredIndexingWork.DERIVED_GRAPH_BINARY_IO_WITHOUT_COMPLETE_BOUNDARIES ->
                "similarity-graph binary I/O"
        }
    }
    return if (labels.isEmpty()) {
        "Additional unmeasured work is not included in this ETA."
    } else {
        "Not included in this ETA: ${labels.joinToString("; ")}."
    }
}

private fun formatEtaEstimate(eta: V2StageAwareEtaEstimate): String? {
    val remaining = eta.remainingMs?.takeIf { it > 0L } ?: return null
    val center = formatEtaDuration(remaining)
    val lower = eta.lowerBoundMs?.takeIf { it > 0L }
    val upper = eta.upperBoundMs?.takeIf { it >= (lower ?: 0L) }
    if (lower == null || upper == null) return "about $center"
    val lowerText = formatEtaDuration(lower)
    val upperText = formatEtaDuration(upper)
    return if (lowerText == upperText) "about $center" else "$lowerText to $upperText"
}

private fun formatEtaDuration(ms: Long): String {
    val totalMinutes = ((ms + 30_000L) / 60_000L).coerceAtLeast(1L)
    if (totalMinutes < 60L) return "$totalMinutes min"
    val hours = totalMinutes / 60L
    val minutes = totalMinutes % 60L
    return if (minutes == 0L) "$hours h" else "$hours h $minutes min"
}

/** Only listener-safe, explicitly recognized stop reasons may leave the durable ledger. */
internal fun indexingStoppedReasonEvidence(
    state: IndexingJobState,
    stateReason: String?,
): String? {
    if (state !in setOf(
            IndexingJobState.PAUSED,
            IndexingJobState.INTERRUPTED,
            IndexingJobState.READY_TO_RESUME,
        )
    ) return null

    val reason = stateReason?.trim()?.takeIf(String::isNotEmpty) ?: return null
    return when (reason) {
        IndexingService.MEDIA_PROCESSING_TIMEOUT_REASON ->
            "Android's processing time limit paused indexing. Completed work is saved."
        "A radio queue is still finishing. Resume indexing after that queue completes." ->
            "A radio queue is still finishing. Resume indexing after that queue completes."
        "paused between processing steps",
        "paused at verified checkpoint",
        "pause completed at verified checkpoint after process stop",
        -> "Completed work is saved."
        "executor interrupted",
        "activation interrupted",
        -> "Indexing was interrupted. Completed work is saved."
        "ready to resume execution",
        "ready to resume activation",
        -> "Completed work is saved and ready to resume."
        else -> null
    }
}
