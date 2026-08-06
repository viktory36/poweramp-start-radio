package com.powerampstartradio.indexing

import com.powerampstartradio.indexing.v2.IndexingJobState
import com.powerampstartradio.indexing.v2.V2IndexingPreflightIntentState
import com.powerampstartradio.indexing.v2.V2IndexingPreflightIntent
import com.powerampstartradio.indexing.v2.V2IndexingExecutionProfile
import java.util.concurrent.atomic.AtomicBoolean

internal object V2IndexingRecoveryStartPolicy {
    fun mayDefer(error: RuntimeException): Boolean =
        error is SecurityException || error is IllegalStateException
}

/** States whose retained durable pointer requires service-owned recovery work. */
internal object V2IndexingRecoveryServicePolicy {
    fun shouldStart(state: IndexingJobState): Boolean = state in setOf(
        IndexingJobState.PLANNED,
        IndexingJobState.RUNNING,
        IndexingJobState.PAUSE_REQUESTED,
        IndexingJobState.INTERRUPTED,
        IndexingJobState.READY_TO_RESUME,
        IndexingJobState.ACTIVATING,
        IndexingJobState.CANCELLING,
        // A terminal pointer is retained only while idempotent private cleanup still needs work.
        IndexingJobState.CANCELLED,
        IndexingJobState.COMPLETE,
    )
}

/** Keeps terminal ownership durable until all private cleanup has been verified. */
internal object V2IndexingTerminalCleanupRecoveryPolicy {
    fun settle(
        state: IndexingJobState,
        cleanup: () -> Unit,
        onRetryRequired: (Throwable, String) -> Unit,
        onClean: () -> Unit,
    ): Boolean {
        require(state == IndexingJobState.COMPLETE || state == IndexingJobState.CANCELLED) {
            "terminal cleanup recovery requires a terminal job"
        }
        val failure = runCatching(cleanup).exceptionOrNull()
        if (failure != null) {
            onRetryRequired(failure, retryMessage(state))
            return false
        }
        onClean()
        return true
    }

    private fun retryMessage(state: IndexingJobState): String {
        val outcome = if (state == IndexingJobState.COMPLETE) {
            "The music index is ready"
        } else {
            "The indexing run was discarded"
        }
        return "$outcome, but temporary indexing files could not be removed. " +
            "Reopen On-device indexing to retry cleanup."
    }
}

/** A command received while the old runner is draining must be replayed after lease release. */
internal object V2IndexingRunnerDrainPolicy {
    fun shouldReplayStart(state: IndexingJobState): Boolean = state in setOf(
        IndexingJobState.PAUSE_REQUESTED,
        IndexingJobState.PAUSED,
        IndexingJobState.WAITING_FOR_INPUT,
        IndexingJobState.INTERRUPTED,
        IndexingJobState.READY_TO_RESUME,
    )
}

/** Only the newest start request may remove the shared foreground notification/service. */
internal object V2IndexingServiceStartOwnershipPolicy {
    fun mayStopForeground(finishingStartId: Int, latestStartId: Int): Boolean =
        finishingStartId == latestStartId
}

/**
 * The platform timeout is type-wide and has a short hard deadline. It therefore bypasses normal
 * start-id ownership and requests service shutdown before attempting any potentially large ledger
 * rewrite. A failed rewrite is not permission to keep the service alive: stage artifacts and the
 * last atomic ledger checkpoint remain the source of truth for recovery.
 */
internal object V2IndexingMediaProcessingTimeoutPolicy {
    fun <T> stopThenCheckpoint(
        stopServiceImmediately: () -> Unit,
        checkpoint: () -> T,
    ): Result<T> {
        stopServiceImmediately()
        return runCatching(checkpoint)
    }
}

/** A platform timeout permanently closes one service instance to queued or late start work. */
internal class V2IndexingServiceLifetimeGate {
    private val mediaProcessingTimedOut = AtomicBoolean(false)

    fun closeForMediaProcessingTimeout() {
        mediaProcessingTimedOut.set(true)
    }

    fun mayStartWork(): Boolean = !mediaProcessingTimedOut.get()
}

internal enum class V2IndexingNotificationActionSet {
    NONE,
    PAUSE,
    RESUME,
    RETRY,
}

/** Exact user-visible notification state used for content dedupe and transition detection. */
internal data class V2IndexingNotificationPresentation(
    val jobId: String,
    val state: IndexingJobState,
    val title: String,
    val text: String,
    val progressPercent: Int?,
    val indeterminateProgress: Boolean,
    val ongoing: Boolean,
    val actionSet: V2IndexingNotificationActionSet,
)

/**
 * RUNNING/ACTIVATING progress is deliberately quiet, while every state or action transition is
 * immediate. A settling service may bypass cadence only to replace stale visible content before
 * detaching the foreground notification; identical content remains deduplicated.
 */
internal class V2IndexingNotificationUpdatePolicy(
    private val activeProgressIntervalMs: Long = 15_000L,
) {
    private data class Published(
        val presentation: V2IndexingNotificationPresentation,
        val atElapsedMs: Long,
    )

    private var lastPublished: Published? = null

    init {
        require(activeProgressIntervalMs > 0L) { "active progress interval must be positive" }
    }

    @Synchronized
    fun shouldPublish(
        candidate: V2IndexingNotificationPresentation,
        observedAtElapsedMs: Long,
        force: Boolean = false,
        settling: Boolean = false,
    ): Boolean {
        require(observedAtElapsedMs >= 0L) { "elapsed time must not be negative" }
        val previous = lastPublished
        val contentChanged = previous?.presentation != candidate
        val transition = previous == null ||
            previous.presentation.jobId != candidate.jobId ||
            previous.presentation.state != candidate.state ||
            previous.presentation.actionSet != candidate.actionSet
        val activeProgressDue = previous != null &&
            candidate.state in ACTIVE_PROGRESS_STATES &&
            contentChanged &&
            observedAtElapsedMs - previous.atElapsedMs >= activeProgressIntervalMs
        val publish = force || transition || activeProgressDue || (settling && contentChanged)
        if (publish) lastPublished = Published(candidate, observedAtElapsedMs)
        return publish
    }

    /** A platform start/error notification replaced the last ledger presentation. */
    @Synchronized
    fun invalidate() {
        lastPublished = null
    }

    private companion object {
        val ACTIVE_PROGRESS_STATES = setOf(
            IndexingJobState.RUNNING,
            IndexingJobState.ACTIVATING,
        )
    }
}

internal object V2IndexingPreflightControlPolicy {
    fun isExplicitlyResumable(state: V2IndexingPreflightIntentState): Boolean = state in setOf(
        V2IndexingPreflightIntentState.INTERRUPTED,
        V2IndexingPreflightIntentState.FAILED,
    )

    fun shouldAutoRecover(state: V2IndexingPreflightIntentState): Boolean = state in setOf(
        V2IndexingPreflightIntentState.REQUESTED,
        V2IndexingPreflightIntentState.PLANNING,
        V2IndexingPreflightIntentState.RESOLVED_WITH_EXECUTABLE_ROWS,
        V2IndexingPreflightIntentState.CANCEL_REQUESTED,
    )

    fun isTerminalRequest(state: V2IndexingPreflightIntentState): Boolean = state in setOf(
        V2IndexingPreflightIntentState.CANCELLED,
        V2IndexingPreflightIntentState.RESOLVED_WITHOUT_EXECUTABLE_ROWS,
        V2IndexingPreflightIntentState.MATERIALIZED,
    )

    fun shouldCheckpointInterruptionOnServiceTeardown(
        state: V2IndexingPreflightIntentState,
    ): Boolean = state in setOf(
        V2IndexingPreflightIntentState.REQUESTED,
        V2IndexingPreflightIntentState.PLANNING,
    )
}

internal object V2IndexingPreflightOwnerPolicy {
    fun shouldSettleBeforeLedgerCommand(
        commandType: V2IndexingServiceCommandType,
        preflightOwnerActive: Boolean,
    ): Boolean = preflightOwnerActive && commandType == V2IndexingServiceCommandType.CANCEL
}

internal enum class V2IndexingPreflightLedgerLinkAction {
    PROMOTE_RESOLVED_INTENT,
    USE_MATERIALIZED_INTENT,
}

internal enum class V2IndexingTerminalCancellationReconciliationAction {
    FINISH_CANCELLED_INTENT,
    USE_CANCELLED_PAIR,
}

/** Pure crash-window policy for the separately atomic intent and ledger files. */
internal object V2IndexingPreflightLedgerLinkPolicy {
    fun terminalCancellationAction(
        intentState: V2IndexingPreflightIntentState,
        ledgerState: IndexingJobState,
    ): V2IndexingTerminalCancellationReconciliationAction? = when {
        ledgerState != IndexingJobState.CANCELLED -> null
        intentState == V2IndexingPreflightIntentState.CANCEL_REQUESTED ->
            V2IndexingTerminalCancellationReconciliationAction.FINISH_CANCELLED_INTENT
        intentState == V2IndexingPreflightIntentState.CANCELLED ->
            V2IndexingTerminalCancellationReconciliationAction.USE_CANCELLED_PAIR
        else -> null
    }

    fun action(
        intent: V2IndexingPreflightIntent,
        verifiedPreflightSpecId: String,
    ): V2IndexingPreflightLedgerLinkAction {
        require(verifiedPreflightSpecId.isNotBlank()) { "verified preflight spec ID is blank" }
        return when (intent.state) {
            V2IndexingPreflightIntentState.RESOLVED_WITH_EXECUTABLE_ROWS -> {
                require(intent.resolvedSpecId == verifiedPreflightSpecId) {
                    "preflight result and ledger spec disagree"
                }
                V2IndexingPreflightLedgerLinkAction.PROMOTE_RESOLVED_INTENT
            }
            V2IndexingPreflightIntentState.MATERIALIZED -> {
                require(intent.materializedSpecId == verifiedPreflightSpecId) {
                    "materialized preflight and ledger spec disagree"
                }
                V2IndexingPreflightLedgerLinkAction.USE_MATERIALIZED_INTENT
            }
            else -> throw IllegalStateException(
                "ledger exists while preflight is ${intent.state}",
            )
        }
    }

    /** Rechecks the persisted terminal handoff immediately before command dispatch. */
    fun requireMaterializedLink(
        intent: V2IndexingPreflightIntent,
        verifiedPreflightSpecId: String,
    ) {
        require(verifiedPreflightSpecId.isNotBlank()) { "verified preflight spec ID is blank" }
        require(intent.state == V2IndexingPreflightIntentState.MATERIALIZED) {
            "ledger command requires a materialized preflight, not ${intent.state}"
        }
        require(intent.materializedSpecId == verifiedPreflightSpecId) {
            "materialized preflight and ledger spec disagree"
        }
    }

    fun shouldRestoreInitialProfile(
        ledgerState: IndexingJobState,
        ledgerRevision: Long,
        ledgerProfile: V2IndexingExecutionProfile,
        requestedProfile: V2IndexingExecutionProfile,
    ): Boolean = ledgerState == IndexingJobState.PLANNED && ledgerRevision == 0L &&
        ledgerProfile != requestedProfile
}
