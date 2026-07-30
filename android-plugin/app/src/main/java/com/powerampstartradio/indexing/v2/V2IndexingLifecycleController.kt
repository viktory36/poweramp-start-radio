package com.powerampstartradio.indexing.v2

import java.io.File

interface V2IndexingLedgerAccess {
    fun list(): List<IndexingJobLedger>
    fun require(jobId: String): IndexingJobLedger
    fun update(
        jobId: String,
        expectedRevision: Long,
        transition: (IndexingJobLedger) -> IndexingJobLedger,
    ): IndexingJobLedger
    fun updateLatest(
        jobId: String,
        transition: (IndexingJobLedger) -> IndexingJobLedger,
    ): IndexingJobLedger
    fun reconcileAfterProcessRestart(
        jobId: String,
        expectedRevision: Long,
        nowEpochMs: Long,
    ): RestartReconciliation
}

class V2AtomicIndexingLedgerAccess(directory: File) : V2IndexingLedgerAccess {
    private val store = AtomicV2IndexingLedgerStore(directory)

    override fun list(): List<IndexingJobLedger> = store.list()

    override fun require(jobId: String): IndexingJobLedger = store.require(jobId)

    override fun update(
        jobId: String,
        expectedRevision: Long,
        transition: (IndexingJobLedger) -> IndexingJobLedger,
    ): IndexingJobLedger = store.update(jobId, expectedRevision, transition)

    override fun updateLatest(
        jobId: String,
        transition: (IndexingJobLedger) -> IndexingJobLedger,
    ): IndexingJobLedger = store.updateLatest(jobId, transition)

    override fun reconcileAfterProcessRestart(
        jobId: String,
        expectedRevision: Long,
        nowEpochMs: Long,
    ): RestartReconciliation = store.reconcileAfterProcessRestart(
        jobId = jobId,
        expectedRevision = expectedRevision,
        nowEpochMs = nowEpochMs,
    )
}

data class V2RetryResult(
    val retried: Int,
    val notEligible: Int,
    val ledger: IndexingJobLedger,
)

data class V2StartupReconciliationResult(
    val reconciledJobs: Int,
    val cancellationJobs: List<String>,
    val resumableJobs: List<String>,
)

/** Serializes every durable lifecycle command in the app process. */
class V2IndexingLifecycleController(
    private val ledgers: V2IndexingLedgerAccess,
    private val nowEpochMs: () -> Long = System::currentTimeMillis,
) {
    @Synchronized
    fun list(): List<IndexingJobLedger> = ledgers.list()

    @Synchronized
    fun require(jobId: String): IndexingJobLedger = ledgers.require(jobId)

    @Synchronized
    fun reconcileNonterminalJobs(): V2StartupReconciliationResult {
        var changed = 0
        val cancellations = mutableListOf<String>()
        val resumable = mutableListOf<String>()
        ledgers.list().filterNot { it.state.isTerminal() }.forEach { ledger ->
            val result = ledgers.reconcileAfterProcessRestart(
                jobId = ledger.jobSpec.jobId,
                expectedRevision = ledger.revision,
                nowEpochMs = nowEpochMs(),
            )
            if (result.changed) changed++
            if (result.action == RestartAction.FINISH_CANCELLATION) {
                cancellations += ledger.jobSpec.jobId
            }
            if (result.ledger.state == IndexingJobState.INTERRUPTED ||
                result.ledger.state == IndexingJobState.READY_TO_RESUME
            ) {
                resumable += ledger.jobSpec.jobId
            }
        }
        return V2StartupReconciliationResult(changed, cancellations, resumable)
    }

    @Synchronized
    fun start(jobId: String): IndexingJobLedger = transition(jobId) { ledger ->
        when (ledger.state) {
            IndexingJobState.PLANNED ->
                V2IndexingLedgerStateMachine.startJob(ledger, nowEpochMs())
            IndexingJobState.RUNNING -> ledger
            else -> throw InvalidIndexingLedgerException(
                "job $jobId cannot start from ${ledger.state}",
            )
        }
    }

    @Synchronized
    fun requestPause(jobId: String): IndexingJobLedger {
        var ledger = transition(jobId) { current ->
            when (current.state) {
                IndexingJobState.RUNNING ->
                    V2IndexingLedgerStateMachine.requestPause(current, nowEpochMs())
                IndexingJobState.PAUSE_REQUESTED,
                IndexingJobState.PAUSED,
                -> current
                else -> throw InvalidIndexingLedgerException(
                    "job $jobId cannot pause from ${current.state}",
                )
            }
        }
        if (ledger.state == IndexingJobState.PAUSE_REQUESTED &&
            ledger.tracks.none { it.state.isActiveStage() }
        ) {
            ledger = transition(jobId) { current ->
                V2IndexingLedgerStateMachine.finishPause(current, nowEpochMs())
            }
        }
        return ledger
    }

    @Synchronized
    fun requestMediaProcessingTimeoutPause(
        jobId: String,
        reason: String,
    ): IndexingJobLedger {
        var ledger = transition(jobId) { current ->
            when (current.state) {
                IndexingJobState.RUNNING ->
                    V2IndexingLedgerStateMachine.requestMediaProcessingTimeoutPause(
                        current,
                        nowEpochMs(),
                        reason,
                    )
                IndexingJobState.PAUSE_REQUESTED,
                IndexingJobState.PAUSED,
                -> current
                else -> throw InvalidIndexingLedgerException(
                    "job $jobId cannot pause after media-processing timeout from ${current.state}",
                )
            }
        }
        if (ledger.state == IndexingJobState.PAUSE_REQUESTED &&
            ledger.tracks.none { it.state.isActiveStage() }
        ) {
            ledger = transition(jobId) { current ->
                V2IndexingLedgerStateMachine.finishPause(current, nowEpochMs(), reason)
            }
        }
        return ledger
    }

    @Synchronized
    fun finishPauseAfterExecutorStops(
        jobId: String,
        reason: String = "paused at verified checkpoint",
    ): IndexingJobLedger {
        var ledger = ledgers.require(jobId)
        if (ledger.state != IndexingJobState.PAUSE_REQUESTED) return ledger
        val active = ledger.tracks.singleOrNull { it.state.isActiveStage() }
        if (active != null) {
            ledger = transition(jobId) { current ->
                V2IndexingLedgerStateMachine.suspendActiveStageForPause(
                    current,
                    active.workId,
                    nowEpochMs(),
                )
            }
        }
        return transition(jobId) { current ->
            V2IndexingLedgerStateMachine.finishPause(current, nowEpochMs(), reason)
        }
    }

    @Synchronized
    fun interruptActivationForMediaProcessingTimeout(
        jobId: String,
        reason: String,
    ): IndexingJobLedger = transition(jobId) { current ->
        V2IndexingLedgerStateMachine.interruptActivationForMediaProcessingTimeout(
            current,
            nowEpochMs(),
            reason,
        )
    }

    @Synchronized
    fun checkpointForMediaProcessingTimeout(
        jobId: String,
        reason: String,
    ): IndexingJobLedger = transition(jobId) { current ->
        V2IndexingLedgerStateMachine.checkpointForMediaProcessingTimeout(
            current,
            nowEpochMs(),
            reason,
        )
    }

    @Synchronized
    fun resume(jobId: String): IndexingJobLedger {
        var ledger = ledgers.require(jobId)
        if (ledger.state == IndexingJobState.RUNNING) return ledger
        if (ledger.state == IndexingJobState.PAUSED ||
            ledger.state == IndexingJobState.WAITING_FOR_INPUT ||
            ledger.state == IndexingJobState.INTERRUPTED
        ) {
            ledger = transition(jobId) { current ->
                V2IndexingLedgerStateMachine.prepareResume(current, nowEpochMs())
            }
        }
        if (ledger.state != IndexingJobState.READY_TO_RESUME) {
            throw InvalidIndexingLedgerException(
                "job $jobId cannot resume from ${ledger.state}",
            )
        }
        return transition(jobId) { current ->
            V2IndexingLedgerStateMachine.resume(current, nowEpochMs())
        }
    }

    @Synchronized
    fun requestCancel(jobId: String): IndexingJobLedger = transition(jobId) { current ->
        when (current.state) {
            IndexingJobState.CANCELLING,
            IndexingJobState.CANCELLED,
            -> current
            IndexingJobState.COMPLETE -> throw InvalidIndexingLedgerException(
                "completed job $jobId cannot be cancelled",
            )
            else -> V2IndexingLedgerStateMachine.requestCancel(current, nowEpochMs())
        }
    }

    @Synchronized
    fun finishCancellationAfterCleanup(jobId: String): IndexingJobLedger {
        var ledger = ledgers.require(jobId)
        if (ledger.state == IndexingJobState.CANCELLED) return ledger
        if (ledger.state != IndexingJobState.CANCELLING) {
            throw InvalidIndexingLedgerException(
                "job $jobId is not cancelling",
            )
        }
        if (ledger.tracks.any { it.state.isActiveStage() }) {
            ledger = transition(jobId) { current ->
                V2IndexingLedgerStateMachine.checkpointCancellation(current, nowEpochMs())
            }
        }
        return transition(jobId) { current ->
            V2IndexingLedgerStateMachine.finishCancel(current, nowEpochMs())
        }
    }

    @Synchronized
    fun retryFailed(jobId: String, trigger: RetryTrigger): V2RetryResult {
        var ledger = ledgers.require(jobId)
        var retried = 0
        var rejected = 0
        val candidates = ledger.tracks.filter {
            it.state == IndexingTrackState.RETRYABLE_FAILURE ||
                it.state == IndexingTrackState.BLOCKED_FAILURE
        }.map { it.workId }
        candidates.forEach { workId ->
            val current = ledgers.require(jobId)
            try {
                ledger = ledgers.update(jobId, current.revision) { value ->
                    V2IndexingLedgerStateMachine.retryTrack(
                        value,
                        workId,
                        trigger,
                        nowEpochMs(),
                    )
                }
                retried++
            } catch (_: InvalidIndexingLedgerException) {
                rejected++
            }
        }
        return V2RetryResult(retried, rejected, ledger)
    }

    @Synchronized
    fun retryTrack(
        jobId: String,
        workId: String,
        trigger: RetryTrigger,
    ): IndexingJobLedger = transition(jobId) { current ->
        V2IndexingLedgerStateMachine.retryTrack(
            current,
            workId,
            trigger,
            nowEpochMs(),
        )
    }

    @Synchronized
    fun changeProfile(
        jobId: String,
        profile: V2IndexingExecutionProfile,
    ): IndexingJobLedger = transition(jobId) { current ->
        V2IndexingLedgerStateMachine.changeExecutionProfile(
            current,
            profile,
            nowEpochMs(),
        )
    }

    @Synchronized
    fun skipTrack(jobId: String, workId: String): IndexingJobLedger = transition(jobId) { current ->
        V2IndexingLedgerStateMachine.skipTrack(current, workId, nowEpochMs())
    }

    @Synchronized
    fun reconcileUnexpectedExecutorStop(jobId: String): RestartReconciliation {
        val current = ledgers.require(jobId)
        return ledgers.reconcileAfterProcessRestart(
            jobId = jobId,
            expectedRevision = current.revision,
            nowEpochMs = nowEpochMs(),
        )
    }

    @Synchronized
    fun waitForInput(jobId: String): IndexingJobLedger = transition(jobId) { current ->
        V2IndexingLedgerStateMachine.waitForInput(current, nowEpochMs())
    }

    @Synchronized
    fun update(
        jobId: String,
        transition: (IndexingJobLedger) -> IndexingJobLedger,
    ): IndexingJobLedger = transition(jobId, transition)

    private fun transition(
        jobId: String,
        operation: (IndexingJobLedger) -> IndexingJobLedger,
    ): IndexingJobLedger {
        return ledgers.updateLatest(jobId) { current ->
            val updated = operation(current)
            updated
        }
    }
}

internal fun IndexingJobState.isTerminal(): Boolean =
    this == IndexingJobState.CANCELLED || this == IndexingJobState.COMPLETE
