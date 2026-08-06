package com.powerampstartradio.indexing

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import android.os.SystemClock
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.core.content.ContextCompat
import com.powerampstartradio.R
import com.powerampstartradio.services.MusicIndexMutationAdmission
import com.powerampstartradio.services.MusicIndexMutationBusyException
import com.powerampstartradio.services.RadioService
import com.powerampstartradio.services.RecommendationWorkAdmission
import com.powerampstartradio.ui.MainViewModel
import com.powerampstartradio.indexing.v2.IndexingJobLedger
import com.powerampstartradio.indexing.v2.IndexingJobState
import com.powerampstartradio.indexing.v2.IndexingTrackState
import com.powerampstartradio.indexing.v2.RetryTrigger
import com.powerampstartradio.indexing.v2.AtomicV2IndexingPreflightIntentStore
import com.powerampstartradio.indexing.v2.V2ActiveIndexingJobPointer
import com.powerampstartradio.indexing.v2.V2ActiveIndexingJobPointerInspection
import com.powerampstartradio.indexing.v2.V2DecodedEosLineage
import com.powerampstartradio.indexing.v2.V2ExecutorLeaseToken
import com.powerampstartradio.indexing.v2.V2IndexingControlFlowException
import com.powerampstartradio.indexing.v2.V2IndexingExecutionProfile
import com.powerampstartradio.indexing.v2.V2IndexingExecutor
import com.powerampstartradio.indexing.v2.V2IndexingExecutorControl
import com.powerampstartradio.indexing.v2.V2IndexingExecutorEvent
import com.powerampstartradio.indexing.v2.V2IndexingExecutorOutcome
import com.powerampstartradio.indexing.v2.V2IndexingEventScope
import com.powerampstartradio.indexing.v2.V2IndexingEtaSampleStore
import com.powerampstartradio.indexing.v2.V2IndexingEtaCoverageSnapshot
import com.powerampstartradio.indexing.v2.V2IndexingEventRateRecorder
import com.powerampstartradio.indexing.v2.V2IndexingJobRepository
import com.powerampstartradio.indexing.v2.V2IndexGenerationReader
import com.powerampstartradio.indexing.v2.V2IndexingOverallWorkPlanner
import com.powerampstartradio.indexing.v2.V2IndexingOverallWorkSnapshot
import com.powerampstartradio.indexing.v2.V2IndexingProgress
import com.powerampstartradio.indexing.v2.V2IndexingProgressSnapshot
import com.powerampstartradio.indexing.v2.V2IndexingPreflightCancelledException
import com.powerampstartradio.indexing.v2.V2IndexingPreflightException
import com.powerampstartradio.indexing.v2.V2IndexingPreflightFailureCode
import com.powerampstartradio.indexing.v2.V2IndexingPreflightFailurePolicy
import com.powerampstartradio.indexing.v2.V2IndexingPreflightFailureScope
import com.powerampstartradio.indexing.v2.V2IndexingPreflightIntent
import com.powerampstartradio.indexing.v2.V2IndexingPreflightIntentState
import com.powerampstartradio.indexing.v2.V2IndexingPreflightIntentStateMachine
import com.powerampstartradio.indexing.v2.V2IndexingPreflightObserver
import com.powerampstartradio.indexing.v2.V2IndexingPreflightPhase
import com.powerampstartradio.indexing.v2.V2IndexingPreflightProgress
import com.powerampstartradio.indexing.v2.V2IndexingPreflightProgressUnit
import com.powerampstartradio.indexing.v2.V2IndexingPreflightRequestMaterializer
import com.powerampstartradio.indexing.v2.V2IndexingPreflightResolution
import com.powerampstartradio.indexing.v2.V2IndexingJobPreflightPlanner
import com.powerampstartradio.indexing.v2.V2PowerampProviderSnapshotAcquirer
import com.powerampstartradio.indexing.v2.V2MeasuredWorkStage
import com.powerampstartradio.indexing.v2.V2PersistedStageRateSnapshot
import com.powerampstartradio.indexing.v2.V2StageAwareEtaEstimate
import com.powerampstartradio.indexing.v2.V2StageAwareWorkEstimator
import com.powerampstartradio.indexing.v2.V2VerifiedPcmProgressTracker
import java.util.UUID
import java.util.concurrent.atomic.AtomicReference
import java.util.concurrent.atomic.AtomicInteger
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import kotlin.math.roundToInt

private class RecommendationHandoffTimeoutException : IllegalStateException(
    "A radio queue is still finishing",
)

/**
 * Durable V2 foreground executor. Intents carry only immutable ledger identities; all track/model
 * plans are reloaded from the atomically persisted job before any work starts.
 */
class IndexingService : Service() {
    sealed interface IndexingState {
        data object Idle : IndexingState

        data class JobSnapshot(
            val jobId: String,
            val jobState: IndexingJobState,
            val profile: V2IndexingExecutionProfile,
            val progress: V2IndexingProgressSnapshot,
            val overallWork: V2IndexingOverallWorkSnapshot,
            val event: V2IndexingExecutorEvent?,
            val eta: V2StageAwareEtaEstimate,
            val etaCoverage: V2IndexingEtaCoverageSnapshot,
            val stateReason: String?,
        ) : IndexingState {
            val etaCoversAllRemainingWork: Boolean
                get() = etaCoverage.coversWholeJob
        }

        data class PreflightSnapshot(
            val jobId: String,
            val state: V2IndexingPreflightIntentState,
            val profile: V2IndexingExecutionProfile,
            val selectedTrackCount: Int,
            val progress: V2IndexingPreflightProgress,
            val failureCode: V2IndexingPreflightFailureCode?,
        ) : IndexingState

        data class Error(
            val jobId: String?,
            val message: String,
        ) : IndexingState
    }

    private enum class StopReason {
        PAUSE,
        CANCEL,
        TIMEOUT,
    }

    private enum class PendingRunnerCommandType {
        START,
        SETTLE,
    }

    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val commandMutex = Mutex()
    private lateinit var repository: V2IndexingJobRepository
    private lateinit var activePointer: V2ActiveIndexingJobPointer
    private lateinit var executor: V2IndexingExecutor
    private lateinit var etaStore: V2IndexingEtaSampleStore
    private lateinit var estimator: V2StageAwareWorkEstimator
    private lateinit var preflightStore: AtomicV2IndexingPreflightIntentStore
    private lateinit var preflightMaterializer: V2IndexingPreflightRequestMaterializer
    private var runnerJob: Job? = null
    private var runnerControl: ServiceExecutorControl? = null
    private var runnerToken: V2ExecutorLeaseToken? = null
    private var preflightJob: Job? = null
    private var preflightControl: ServicePreflightObserver? = null
    private val latestStartId = AtomicInteger(0)
    private val pendingRunnerCommand = AtomicReference<PendingRunnerCommand?>(null)
    private val pendingPreflightCommand = AtomicReference<PendingPreflightCommand?>(null)
    private val lifetimeGate = V2IndexingServiceLifetimeGate()
    private val executorStartLock = Any()
    private val recommendationReservationLock = Any()
    private val recommendationReservationOwners =
        mutableMapOf<String, RecommendationWorkAdmission.ReservationOwner>()
    private val recommendationServiceIdentity = UUID.randomUUID().toString()
    private var wakeLock: PowerManager.WakeLock? = null
    private var lastPreflightNotificationAtElapsedMs: Long? = null
    private val notificationUpdatePolicy = V2IndexingNotificationUpdatePolicy()
    private var lastEtaSaveAtElapsedMs = 0L
    @Volatile private var runtimeInitialized = false

    override fun onCreate() {
        super.onCreate()
        // Foreground promotion has a short platform deadline. Keep onCreate limited to the
        // lightweight state needed to parse a restart and build its first notification; a ledger
        // can contain thousands of tracks and is deliberately opened only after startForeground.
        activePointer = V2ActiveIndexingJobPointer(filesDir)
        preflightStore = createPreflightStore(this)
        createNotificationChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val command = if (intent == null) {
            when (val pointer = activePointer.inspect()) {
                V2ActiveIndexingJobPointerInspection.Missing -> null
                is V2ActiveIndexingJobPointerInspection.Readable -> {
                    V2IndexingServiceCommand(
                        V2IndexingServiceCommandType.RECOVER,
                        pointer.jobId,
                    )
                }
                is V2ActiveIndexingJobPointerInspection.Unreadable -> {
                    publishActivePointerError(null, "restore indexing", pointer)
                    null
                }
            }
        } else {
            runCatching { V2IndexingServiceIntents.parse(this, intent) }
                .onFailure { error ->
                    Log.w(TAG, "Rejected indexing service intent", error)
                }
                .getOrNull()
        }
        if (command == null) {
            stopSelfResult(startId)
            return START_NOT_STICKY
        }
        if (!lifetimeGate.mayStartWork()) {
            releaseLaunchRecommendationReservation(command.jobId)
            stopSelfResult(startId)
            return START_NOT_STICKY
        }
        reserveServiceRecommendationReservation(command.jobId)
        releaseLaunchRecommendationReservation(command.jobId)
        latestStartId.accumulateAndGet(startId, ::maxOf)

        val notification = buildNotification(
            command.jobId,
            "Opening saved indexing state",
        )
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.VANILLA_ICE_CREAM) {
                startForeground(
                    NOTIFICATION_ID,
                    notification,
                    ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROCESSING,
                )
            } else {
                startForeground(NOTIFICATION_ID, notification)
            }
            // startForeground replaces any prior ledger presentation with restoration copy.
            notificationUpdatePolicy.invalidate()
            lastPreflightNotificationAtElapsedMs = null
        } catch (error: RuntimeException) {
            // Most notably, Android 15 rejects a new mediaProcessing run while its rolling quota
            // is exhausted. The request/ledger remains durable and a visible retry can resume it.
            Log.w(TAG, "Android rejected media-processing foreground promotion", error)
            _state.value = IndexingState.Error(
                command.jobId,
                "Android cannot start indexing in the background right now. " +
                    "Open On-device indexing and retry.",
            )
            // A later command can reuse this job-scoped owner while an earlier worker is still
            // unwinding. Service teardown waits for those workers before releasing every owner.
            stopSelfResult(startId)
            return START_NOT_STICKY
        }
        serviceScope.launch {
            try {
                initializeRuntimeAfterForegroundPromotion()
                if (lifetimeGate.mayStartWork()) {
                    commandMutex.withLock {
                        if (lifetimeGate.mayStartWork()) handleCommand(command, startId)
                    }
                }
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (error: Throwable) {
                Log.e(TAG, "Unable to initialize durable indexing runtime", error)
                publishError(
                    command.jobId,
                    "Indexing state could not be opened. Reopen On-device indexing to recover.",
                )
                // Keep the job-scoped owner through teardown; this command may have failed while
                // an earlier same-job worker still owns an uninterruptible model call.
                finishForeground(startId, detach = true)
            }
        }
        return START_STICKY
    }

    @Synchronized
    private fun initializeRuntimeAfterForegroundPromotion() {
        if (runtimeInitialized) return
        repository = V2IndexingJobRepository.get(this)
        executor = V2IndexingExecutor(this, repository = repository)
        etaStore = V2IndexingEtaSampleStore(filesDir)
        estimator = V2StageAwareWorkEstimator(restoredSnapshot = etaStore.loadOrEmpty())
        preflightMaterializer = V2IndexingPreflightRequestMaterializer(this)
        runtimeInitialized = true
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        // Process/service teardown is not a user cancellation. The next process reconciles the
        // active stage from its last verified checkpoint.
        val jobsToDrain = listOfNotNull(runnerJob, preflightJob)
            .filterNot { it.isCompleted }
            .distinct()
        if (jobsToDrain.isEmpty()) {
            releaseAllServiceRecommendationReservations()
        } else {
            val remaining = AtomicInteger(jobsToDrain.size)
            jobsToDrain.forEach { job ->
                job.invokeOnCompletion {
                    if (remaining.decrementAndGet() == 0) {
                        releaseAllServiceRecommendationReservations()
                    }
                }
            }
        }
        runnerJob?.cancel()
        if (preflightJob?.isActive == true && runnerJob?.isActive != true) {
            preflightControl?.requestStop(
                "Indexing service stopped; reopen to resume preflight",
                false,
            )
            val activeJobId = when (val pointer = activePointer.inspect()) {
                V2ActiveIndexingJobPointerInspection.Missing -> null
                is V2ActiveIndexingJobPointerInspection.Readable -> pointer.jobId
                is V2ActiveIndexingJobPointerInspection.Unreadable -> {
                    publishActivePointerError(null, "checkpoint indexing shutdown", pointer)
                    null
                }
            }
            activeJobId?.let { jobId ->
                runCatching {
                    preflightStore.updateLatest(jobId) { current ->
                        if (V2IndexingPreflightControlPolicy
                                .shouldCheckpointInterruptionOnServiceTeardown(current.state)
                        ) {
                            V2IndexingPreflightIntentStateMachine.interrupt(
                                current,
                                "Indexing service stopped; reopen to resume preflight",
                                System.currentTimeMillis(),
                            )
                        } else {
                            current
                        }
                    }
                }.onFailure { error ->
                    Log.e(TAG, "Unable to checkpoint preflight service teardown", error)
                }
            }
        }
        preflightJob?.cancel()
        serviceScope.cancel()
        releaseWakeLock()
        super.onDestroy()
    }

    override fun onTimeout(startId: Int, fgsType: Int) {
        lifetimeGate.closeForMediaProcessingTimeout()
        val jobId = when (val pointer = activePointer.inspect()) {
            V2ActiveIndexingJobPointerInspection.Missing -> null
            is V2ActiveIndexingJobPointerInspection.Readable -> pointer.jobId
            is V2ActiveIndexingJobPointerInspection.Unreadable -> {
                publishActivePointerError(null, "checkpoint the indexing timeout", pointer)
                null
            }
        }
        if (jobId == null) {
            forceStopForMediaProcessingTimeout(detach = false)
            return
        }
        if (preflightJob?.isActive == true && runnerJob?.isActive != true) {
            preflightControl?.requestStop(MEDIA_PROCESSING_TIMEOUT_REASON, false)
            // onTimeout is a whole-service deadline, not completion of one start command. Stop
            // immediately even if startId is older than a command that arrived meanwhile. The
            // following durable transition is best effort; the source/model hashes and planning
            // progress already committed by preflight remain the recovery authority if killed.
            val interrupted = V2IndexingMediaProcessingTimeoutPolicy.stopThenCheckpoint(
                stopServiceImmediately = {
                    forceStopForMediaProcessingTimeout(detach = true)
                },
            ) {
                preflightStore.updateLatest(jobId) { current ->
                    V2IndexingPreflightIntentStateMachine.interrupt(
                        current,
                        MEDIA_PROCESSING_TIMEOUT_REASON,
                        System.currentTimeMillis(),
                    )
                }
            }.onFailure { error ->
                Log.e(TAG, "Unable to checkpoint media-processing preflight timeout", error)
            }.getOrNull()
            preflightJob?.cancel(CancellationException(MEDIA_PROCESSING_TIMEOUT_REASON))
            interrupted?.let { _state.value = V2IndexingServiceStateMapper.mapPreflight(it) }
            // Quota is exhausted at this point, so foreground-service notification actions cannot
            // be promised to work. The content intent brings the app forward and exposes durable
            // Retry/Cancel controls after Android resets the user-visible quota window.
            getSystemService(NotificationManager::class.java).notify(
                NOTIFICATION_ID,
                buildNotification(jobId, MEDIA_PROCESSING_TIMEOUT_REASON),
            )
            return
        }
        runnerControl?.requestStop(StopReason.TIMEOUT)
        val checkpoint = V2IndexingMediaProcessingTimeoutPolicy.stopThenCheckpoint(
            stopServiceImmediately = {
                forceStopForMediaProcessingTimeout(detach = true)
            },
        ) {
            synchronized(executorStartLock) {
                initializeRuntimeAfterForegroundPromotion()
                repository.reconcileStartup()
                val ledger = repository.require(jobId)
                val checkpointed = if (ledger.state in setOf(
                        IndexingJobState.RUNNING,
                        IndexingJobState.PAUSE_REQUESTED,
                        IndexingJobState.ACTIVATING,
                    )
                ) {
                    repository.checkpointForMediaProcessingTimeout(
                        jobId,
                        MEDIA_PROCESSING_TIMEOUT_REASON,
                    )
                } else {
                    ledger
                }
                runnerToken?.let { token ->
                    runCatching { repository.releaseExecutor(token) }
                        .onFailure { error ->
                            Log.w(TAG, "Unable to retire timed-out executor lease", error)
                        }
                }
                checkpointed
            }
        }
        checkpoint.onFailure { error ->
            Log.e(TAG, "Unable to checkpoint media-processing timeout", error)
        }
        runnerJob?.cancel(CancellationException(MEDIA_PROCESSING_TIMEOUT_REASON))
        // A start already inside executor setup may have acquired the lock after the first stop.
        releaseWakeLock()
        getSystemService(NotificationManager::class.java).notify(
            NOTIFICATION_ID,
            buildNotification(jobId, MEDIA_PROCESSING_TIMEOUT_REASON),
        )
        checkpoint.getOrNull()?.let { ledger ->
            _state.value = V2IndexingServiceStateMapper.map(
                ledger = ledger,
                event = null,
                eta = V2StageAwareEtaEstimate(null, null, null, emptySet()),
                overallWork = V2IndexingOverallWorkPlanner.snapshot(
                    ledger,
                    event = null,
                    graphPlan = null,
                    verifiedPcmWorkIds = emptySet(),
                ),
            )
        }
        if (runtimeInitialized) saveEtaSamples()
    }

    private suspend fun handleCommand(command: V2IndexingServiceCommand, startId: Int) {
        if (!lifetimeGate.mayStartWork()) return
        val pointerInspection = activePointer.inspect()
        if (pointerInspection is V2ActiveIndexingJobPointerInspection.Unreadable) {
            publishActivePointerError(command.jobId, "run an indexing command", pointerInspection)
            finishForeground(startId, detach = true)
            return
        }
        if (pointerInspection is V2ActiveIndexingJobPointerInspection.Readable &&
            pointerInspection.jobId != command.jobId
        ) {
            val activeJobId = pointerInspection.jobId
            Log.w(
                TAG,
                "Ignoring stale ${command.type} for ${command.jobId}; " +
                    "durable ownership belongs to $activeJobId",
            )
            // Reserve the durable owner before retiring the stale start's owner. Replaying RECOVER
            // keeps the latest Android startId attached to the real job without applying the stale
            // command or letting its terminal state stop a newer runner.
            reserveServiceRecommendationReservation(activeJobId)
            releaseServiceRecommendationReservation(command.jobId)
            handleCommand(
                V2IndexingServiceCommand(
                    type = V2IndexingServiceCommandType.RECOVER,
                    jobId = activeJobId,
                ),
                startId,
            )
            return
        }
        var initialLedger = runCatching { repository.require(command.jobId) }.getOrNull()
        if (initialLedger != null &&
            preflightJob?.isActive == true &&
            command.type in setOf(
                V2IndexingServiceCommandType.START,
                V2IndexingServiceCommandType.RESUME,
                V2IndexingServiceCommandType.RECOVER,
            )
        ) {
            queuePendingPreflightCommand(
                PendingPreflightCommand(
                    PendingRunnerCommandType.START,
                    command.jobId,
                    startId,
                ),
            )
            publishPreflight(
                preflightStore.require(command.jobId),
                preserveRecommendationReservation = true,
            )
            return
        }
        if (initialLedger != null && V2IndexingPreflightOwnerPolicy.shouldSettleBeforeLedgerCommand(
                commandType = command.type,
                preflightOwnerActive = preflightJob?.isActive == true,
            )
        ) {
            val owner = preflightJob
            preflightControl?.requestStop("cancel requested after ledger publication", true)
            owner?.join()
            initialLedger = repository.require(command.jobId)
        }
        if (initialLedger == null) {
            val preflight = runCatching { preflightStore.require(command.jobId) }
                .getOrElse { error ->
                    Log.e(TAG, "Indexing preflight could not be opened", error)
                    publishError(
                        command.jobId,
                        "Indexing job could not be opened. Reopen On-device indexing to recover.",
                    )
                    finishForeground(startId, detach = false)
                    return
                }
            handlePreflightCommand(command, preflight, startId)
            return
        }
        var ledger = initialLedger
        // Intent result is published before its ledger. A crash after the ledger write is
        // reconciled only when both files name the same immutable spec.
            val leftoverPreflight = runCatching { preflightStore.require(command.jobId) }
                .getOrElse { error ->
                Log.e(TAG, "Indexing result evidence could not be opened", error)
                publishError(
                    command.jobId,
                    "Indexing stopped before its result was confirmed. " +
                        "Reopen On-device indexing to recover.",
                )
                finishForeground(startId, detach = true)
                return
            }
        when (V2IndexingPreflightLedgerLinkPolicy.terminalCancellationAction(
            intentState = leftoverPreflight.state,
            ledgerState = ledger.state,
        )) {
            V2IndexingTerminalCancellationReconciliationAction.FINISH_CANCELLED_INTENT -> {
                preflightStore.updateLatest(command.jobId) { current ->
                    V2IndexingPreflightIntentStateMachine.finishCancellation(
                        current,
                        System.currentTimeMillis(),
                    )
                }
                settleRecoveredTerminalJob(ledger, startId)
                return
            }
            V2IndexingTerminalCancellationReconciliationAction.USE_CANCELLED_PAIR -> {
                settleRecoveredTerminalJob(ledger, startId)
                return
            }
            null -> Unit
        }
        if (leftoverPreflight.state == V2IndexingPreflightIntentState.CANCEL_REQUESTED) {
            val cancelling = repository.requestCancel(command.jobId)
            finishCancellationWithoutRunner(cancelling, startId)
            val cancelled = preflightStore.updateLatest(command.jobId) { current ->
                V2IndexingPreflightIntentStateMachine.finishCancellation(
                    current,
                    System.currentTimeMillis(),
                )
            }
            publishPreflight(cancelled, forceNotification = true)
            return
        }
        val linkAction = runCatching {
            val verifiedPreflightSpecId = V2DecodedEosLineage.requirePreflightSpecId(
                ledger.jobSpec,
            )
            V2IndexingPreflightLedgerLinkPolicy.action(
                leftoverPreflight,
                verifiedPreflightSpecId,
            )
        }.getOrElse { error ->
            Log.e(TAG, "Indexing plan and job link is invalid", error)
            publishError(
                command.jobId,
                "The saved indexing job could not be read. Reopen On-device indexing to recover.",
            )
            finishForeground(startId, detach = true)
            return
        }
        val reconciled = when (linkAction) {
            V2IndexingPreflightLedgerLinkAction.PROMOTE_RESOLVED_INTENT -> {
                preflightStore.updateLatest(command.jobId) { current ->
                    V2IndexingPreflightIntentStateMachine.materializeResolved(
                        current,
                        System.currentTimeMillis(),
                    )
                }
            }
            V2IndexingPreflightLedgerLinkAction.USE_MATERIALIZED_INTENT -> leftoverPreflight
        }
        runCatching {
            V2IndexingPreflightLedgerLinkPolicy.requireMaterializedLink(
                reconciled,
                V2DecodedEosLineage.requirePreflightSpecId(ledger.jobSpec),
            )
        }.getOrElse { error ->
            Log.e(TAG, "Reconciled indexing plan and job link is invalid", error)
            publishError(
                command.jobId,
                "The saved indexing job could not be read. Reopen On-device indexing to recover.",
            )
            finishForeground(startId, detach = true)
            return
        }
        if (ledger.state != IndexingJobState.CANCELLED &&
            ledger.state != IndexingJobState.COMPLETE &&
            ledger.executionProfile != V2IndexingExecutionProfile.FULL
        ) {
            ledger = repository.changeProfile(
                command.jobId,
                V2IndexingExecutionProfile.FULL,
            )
        }
        if (ledger.state == IndexingJobState.CANCELLED) {
            settleRecoveredTerminalJob(ledger, startId)
            return
        }
        try {
            if (!lifetimeGate.mayStartWork()) return
            when (command.type) {
                V2IndexingServiceCommandType.START -> startOrResume(ledger, recover = false, startId)
                V2IndexingServiceCommandType.RECOVER -> recover(ledger, startId)
                V2IndexingServiceCommandType.RESUME -> startOrResume(
                    ledger,
                    recover = ledger.state == IndexingJobState.INTERRUPTED,
                    startId = startId,
                )
                V2IndexingServiceCommandType.PAUSE -> requestPause(ledger, StopReason.PAUSE, startId)
                V2IndexingServiceCommandType.TIMEOUT -> requestPause(ledger, StopReason.TIMEOUT, startId)
                V2IndexingServiceCommandType.CANCEL -> requestCancel(ledger, startId)
                V2IndexingServiceCommandType.RETRY -> retry(command, ledger, startId)
                V2IndexingServiceCommandType.SKIP -> skip(command, ledger, startId)
                V2IndexingServiceCommandType.PROFILE -> changeProfile(command, startId)
            }
        } catch (error: Exception) {
            Log.e(TAG, "Indexing command ${command.type} failed", error)
            if (runnerJob?.isActive == true) {
                publish(repository.require(command.jobId))
            } else {
                publishError(
                    command.jobId,
                    "Indexing command could not finish. Reopen On-device indexing to recover.",
                )
                finishForeground(startId, detach = true)
            }
        }
    }

    private fun handlePreflightCommand(
        command: V2IndexingServiceCommand,
        intent: V2IndexingPreflightIntent,
        startId: Int,
    ) {
        if (intent.state in setOf(
                V2IndexingPreflightIntentState.CANCELLED,
                V2IndexingPreflightIntentState.RESOLVED_WITHOUT_EXECUTABLE_ROWS,
            )
        ) {
            activePointer.clear(intent.jobId)
            publishPreflight(intent, forceNotification = true)
            finishForeground(startId, detach = true)
            return
        }
        if (intent.state == V2IndexingPreflightIntentState.MATERIALIZED) {
            publishError(
                intent.jobId,
                "Saved indexing plan is incomplete. Reopen On-device indexing to recover.",
            )
            finishForeground(startId, detach = true)
            return
        }
        when (command.type) {
            V2IndexingServiceCommandType.START,
            V2IndexingServiceCommandType.RESUME,
            -> startOrResumePreflight(intent, startId)

            V2IndexingServiceCommandType.RECOVER -> {
                if (V2IndexingPreflightControlPolicy.shouldAutoRecover(intent.state)) {
                    startOrResumePreflight(intent, startId)
                } else {
                    publishPreflight(intent, forceNotification = true)
                    finishForeground(startId, detach = true)
                }
            }

            V2IndexingServiceCommandType.CANCEL -> requestPreflightCancellation(intent, startId)

            else -> {
                publishPreflight(intent, forceNotification = true)
                if (preflightJob?.isActive == true) {
                    queuePendingPreflightCommand(
                        PendingPreflightCommand(
                            PendingRunnerCommandType.SETTLE,
                            intent.jobId,
                            startId,
                        ),
                    )
                } else {
                    finishForeground(startId, detach = true)
                }
            }
        }
    }

    private fun startOrResumePreflight(
        initial: V2IndexingPreflightIntent,
        startId: Int,
    ) {
        reserveServiceRecommendationReservation(initial.jobId)
        if (initial.state == V2IndexingPreflightIntentState.CANCEL_REQUESTED) {
            finishPreflightCancellation(initial, startId)
            return
        }
        if (initial.state == V2IndexingPreflightIntentState.CANCELLED) {
            activePointer.clear(initial.jobId)
            publishPreflight(initial, forceNotification = true)
            finishForeground(startId, detach = true)
            return
        }
        if (initial.state ==
            V2IndexingPreflightIntentState.RESOLVED_WITHOUT_EXECUTABLE_ROWS
        ) {
            activePointer.clear(initial.jobId)
            publishPreflight(initial, forceNotification = true)
            finishForeground(startId, detach = true)
            return
        }
        if (initial.state == V2IndexingPreflightIntentState.MATERIALIZED) {
            publishError(
                initial.jobId,
                "Saved indexing plan is incomplete. Reopen On-device indexing to recover.",
            )
            finishForeground(startId, detach = true)
            return
        }
        if (preflightJob?.isActive == true) {
            queuePendingPreflightCommand(
                PendingPreflightCommand(
                    PendingRunnerCommandType.START,
                    initial.jobId,
                    startId,
                ),
            )
            publishPreflight(
                preflightStore.require(initial.jobId),
                preserveRecommendationReservation = true,
            )
            return
        }
        if (runnerJob?.isActive == true) {
            publishError(initial.jobId, "Another indexing run is already active.")
            finishForeground(startId, detach = true)
            return
        }

        activePointer.write(initial.jobId)
        acquireWakeLock()
        publishPreflight(initial, forceNotification = true)
        val control = ServicePreflightObserver(initial.jobId)
        preflightControl = control
        preflightJob = serviceScope.launch {
            runPreflight(initial.jobId, control, startId)
        }
    }

    private suspend fun runPreflight(
        jobId: String,
        control: ServicePreflightObserver,
        startId: Int,
    ) {
        var handedOffToExecutor = false
        val wakeLockHeartbeat = serviceScope.launch {
            while (isActive) {
                delay(HEARTBEAT_INTERVAL_MS)
                renewWakeLock()
            }
        }
        try {
            check(isServiceRecommendationReservationHeld(jobId)) {
                "Preflight cannot inspect indexing inputs without recommendation exclusion"
            }
            val beforeResourceHandoff = preflightStore.require(jobId)
            publishPreflight(
                beforeResourceHandoff.copy(
                    progress = V2IndexingPreflightProgress(
                        V2IndexingPreflightPhase.QUEUED,
                        "Releasing Find Music model memory before reading indexing inputs",
                    ),
                ),
                forceNotification = true,
            )
            if (!awaitExclusiveRecommendationResources()) {
                throw RecommendationHandoffTimeoutException()
            }
            // Resolve and durably bind the base before any source/model hashing. A resumed attempt
            // must use the same generation or fail visibly rather than silently rebasing.
            control.throwIfCancelled()
            val beforeBinding = preflightStore.require(jobId)
            publishPreflight(
                beforeBinding.copy(
                    progress = V2IndexingPreflightProgress(
                        V2IndexingPreflightPhase.ACTIVE_GENERATION,
                        "Binding the active index generation",
                    ),
                ),
            )
            val activeGeneration = V2IndexGenerationReader.requireActive(filesDir) { progress ->
                control.onProgress(
                    V2IndexingPreflightProgress(
                        phase = V2IndexingPreflightPhase.ACTIVE_GENERATION,
                        message = exactHashProgressText(
                            subject = "active music-index file ${progress.filename}",
                            completedBytes = progress.completedBytes,
                            totalBytes = progress.totalBytes,
                        ),
                        completedUnits = progress.completedBytes,
                        totalUnits = progress.totalBytes,
                        unit = V2IndexingPreflightProgressUnit.BYTES,
                    ),
                )
            }
            val planning = preflightStore.updateLatest(jobId) { current ->
                V2IndexingPreflightIntentStateMachine.beginOrResumePlanning(
                    current = current,
                    baseGenerationId = activeGeneration.manifest.generationId,
                    progress = V2IndexingPreflightProgress(
                        V2IndexingPreflightPhase.POWERAMP_SNAPSHOT,
                        "Reading ${current.selected.size} selected Poweramp library rows",
                    ),
                    nowEpochMs = System.currentTimeMillis(),
                )
            }
            publishPreflight(planning, forceNotification = true)
            control.throwIfCancelled()

            val providerAcquisition = V2PowerampProviderSnapshotAcquirer(this)
                .acquireSelectedWithCueFallbackBlocking(
                    fileIds = planning.selected.map { it.powerampFileId },
                ) {
                    completedRows,
                    totalRows,
                ->
                val selectedRead = totalRows == planning.selected.size
                control.onProgress(
                    V2IndexingPreflightProgress(
                        phase = V2IndexingPreflightPhase.POWERAMP_SNAPSHOT,
                        message = if (selectedRead && completedRows == 0) {
                            "Reading ${planning.selected.size} selected Poweramp rows"
                        } else if (selectedRead) {
                            "Read $completedRows of ${planning.selected.size} selected " +
                                "Poweramp rows"
                        } else {
                            powerampLibraryReadProgressText(completedRows, totalRows)
                        },
                        completedUnits = completedRows.toLong(),
                        totalUnits = totalRows.toLong(),
                    ),
                )
            }
            val providerSnapshot = providerAcquisition.snapshot
            control.throwIfCancelled()
            val current = preflightStore.require(jobId)
            val request = preflightMaterializer.materialize(current, providerSnapshot)
            val resolution = V2IndexingJobPreflightPlanner(
                context = this,
                observer = control,
            ).resolveAndPersistBlocking(request) { prepared ->
                preflightStore.updateLatest(jobId) { latest ->
                    V2IndexingPreflightIntentStateMachine.resolveWithExecutableRows(
                        current = latest,
                        planned = prepared.planned,
                        rejected = prepared.rejected,
                        specId = prepared.specId,
                        nowEpochMs = System.currentTimeMillis(),
                    )
                }
            }
            when (resolution) {
                is V2IndexingPreflightResolution.Materialized -> {
                    control.throwIfCancelled()
                    val materialized = preflightStore.updateLatest(jobId) { latest ->
                        V2IndexingPreflightIntentStateMachine.materializeResolved(
                            current = latest,
                            nowEpochMs = System.currentTimeMillis(),
                        )
                    }
                    check(materialized.materializedSpecId == resolution.job.specId) {
                        "Materialized preflight and ledger spec disagree"
                    }
                    control.throwIfCancelled()
                    repository.refresh()
                    repository.changeProfile(jobId, V2IndexingExecutionProfile.FULL)

                    releaseWakeLock()
                    val ledger = repository.require(jobId)
                    Log.i(
                        TAG,
                        "Prepared ${ledger.tracks.size} tracks for $jobId; starting executor",
                    )
                    startOrResume(ledger, recover = false, startId)
                    handedOffToExecutor = true
                }
                is V2IndexingPreflightResolution.WithoutExecutableRows -> {
                    val terminal = preflightStore.updateLatest(jobId) { latest ->
                        V2IndexingPreflightIntentStateMachine.resolveWithoutExecutableRows(
                            current = latest,
                            rejected = resolution.rejected,
                            nowEpochMs = System.currentTimeMillis(),
                        )
                    }
                    activePointer.clear(jobId)
                    publishPreflight(terminal, forceNotification = true)
                    finishForeground(startId, detach = true)
                }
            }
        } catch (error: Throwable) {
            val current = runCatching { preflightStore.require(jobId) }.getOrNull()
            val materializedLedger = runCatching { repository.require(jobId) }.getOrNull()
            if (error is RecommendationHandoffTimeoutException && current != null) {
                val interrupted = preflightStore.updateLatest(jobId) { latest ->
                    V2IndexingPreflightIntentStateMachine.interrupt(
                        current = latest,
                        message = RECOMMENDATION_DRAIN_TIMEOUT_MESSAGE,
                        nowEpochMs = System.currentTimeMillis(),
                    )
                }
                publishPreflight(interrupted, forceNotification = true)
                finishForeground(startId, detach = true)
                return
            }
            if (current != null &&
                (current.state == V2IndexingPreflightIntentState.CANCEL_REQUESTED ||
                    error is V2IndexingPreflightCancelledException)
            ) {
                if (materializedLedger != null) {
                    repository.refresh()
                    if (materializedLedger.state != IndexingJobState.CANCELLED) {
                        val cancellingLedger = repository.requestCancel(jobId)
                        finishCancellationWithoutRunner(cancellingLedger, startId)
                    } else {
                        activePointer.clear(jobId)
                        publishAndStop(materializedLedger, startId, detachNotification = true)
                    }
                    if (current.state == V2IndexingPreflightIntentState.MATERIALIZED) return
                    val cancellingIntent = if (current.state ==
                        V2IndexingPreflightIntentState.CANCEL_REQUESTED
                    ) current else preflightStore.updateLatest(jobId) { value ->
                        V2IndexingPreflightIntentStateMachine.requestCancel(
                            value,
                            System.currentTimeMillis(),
                        )
                    }
                    val cancelled = preflightStore.updateLatest(jobId) { value ->
                        check(value.revision == cancellingIntent.revision)
                        V2IndexingPreflightIntentStateMachine.finishCancellation(
                            value,
                            System.currentTimeMillis(),
                        )
                    }
                    publishPreflight(cancelled, forceNotification = true)
                    return
                }
                val cancelling = if (current.state == V2IndexingPreflightIntentState.CANCEL_REQUESTED) {
                    current
                } else {
                    preflightStore.updateLatest(jobId) { value ->
                        V2IndexingPreflightIntentStateMachine.requestCancel(
                            value,
                            System.currentTimeMillis(),
                        )
                    }
                }
                finishPreflightCancellation(cancelling, startId)
            } else if (current != null &&
                (current.state == V2IndexingPreflightIntentState.RESOLVED_WITH_EXECUTABLE_ROWS ||
                    (materializedLedger != null &&
                        current.state == V2IndexingPreflightIntentState.MATERIALIZED))
            ) {
                val message = if (materializedLedger != null) {
                    "Indexing job is saved. Reopen On-device indexing to resume."
                } else {
                    "The exact source and model plan finished, but the job was not fully saved. " +
                        "Reopen On-device indexing to continue."
                }
                Log.e(TAG, "V2 indexing ledger handoff was interrupted", error)
                publishError(
                    jobId,
                    message,
                )
                finishForeground(startId, detach = true)
            } else if (current != null && current.state in setOf(
                    V2IndexingPreflightIntentState.INTERRUPTED,
                    V2IndexingPreflightIntentState.FAILED,
                )
            ) {
                publishPreflight(current, forceNotification = true)
                finishForeground(startId, detach = true)
            } else if (current != null) {
                Log.e(TAG, "V2 indexing preflight failed", error)
                val failed = preflightStore.updateLatest(jobId) { value ->
                    val typed = error as? V2IndexingPreflightException
                    val code = typed?.code?.takeIf {
                        V2IndexingPreflightFailurePolicy.semantics(it).scope ==
                            V2IndexingPreflightFailureScope.GLOBAL_REQUEST
                    } ?: V2IndexingPreflightFailureCode.INVALID_PLAN
                    V2IndexingPreflightIntentStateMachine.fail(
                        current = value,
                        code = code,
                        message = "The indexing job could not be prepared. " +
                            "Review the selected tracks and try again.",
                        nowEpochMs = System.currentTimeMillis(),
                    )
                }
                publishPreflight(failed, forceNotification = true)
                finishForeground(startId, detach = true)
            } else {
                Log.e(TAG, "V2 indexing preflight failed before durable state was available", error)
                publishError(
                    jobId,
                    "The indexing job could not be prepared. " +
                        "Reopen On-device indexing and try again.",
                )
                finishForeground(startId, detach = true)
            }
        } finally {
            withContext(NonCancellable) {
                wakeLockHeartbeat.cancelAndJoin()
                if (!handedOffToExecutor) releaseWakeLock()
                preflightControl = null
                preflightJob = null
                drainPendingPreflightCommand()
            }
        }
    }

    private fun requestPreflightCancellation(
        initial: V2IndexingPreflightIntent,
        startId: Int,
    ) {
        val cancelling = preflightStore.updateLatest(initial.jobId) { current ->
            V2IndexingPreflightIntentStateMachine.requestCancel(
                current,
                System.currentTimeMillis(),
            )
        }
        preflightControl?.requestStop("cancel requested", true)
        publishPreflight(cancelling, forceNotification = true)
        if (preflightJob?.isActive != true) {
            finishPreflightCancellation(cancelling, startId)
        } else {
            queuePendingPreflightCommand(
                PendingPreflightCommand(
                    PendingRunnerCommandType.SETTLE,
                    cancelling.jobId,
                    startId,
                ),
            )
        }
    }

    private fun finishPreflightCancellation(
        cancelling: V2IndexingPreflightIntent,
        startId: Int,
    ) {
        val cancelled = if (cancelling.state == V2IndexingPreflightIntentState.CANCELLED) {
            cancelling
        } else {
            preflightStore.updateLatest(cancelling.jobId) { current ->
                V2IndexingPreflightIntentStateMachine.finishCancellation(
                    current,
                    System.currentTimeMillis(),
                )
            }
        }
        activePointer.clear(cancelled.jobId)
        publishPreflight(cancelled, forceNotification = true)
        finishForeground(startId, detach = true)
    }

    private fun recover(ledger: IndexingJobLedger, startId: Int) {
        when (ledger.state) {
            IndexingJobState.PLANNED,
            IndexingJobState.RUNNING,
            IndexingJobState.INTERRUPTED,
            IndexingJobState.READY_TO_RESUME,
            IndexingJobState.ACTIVATING,
            -> startOrResume(ledger, recover = true, startId)

            IndexingJobState.CANCELLING -> finishCancellationWithoutRunner(ledger, startId)
            IndexingJobState.PAUSE_REQUESTED -> requestPause(ledger, StopReason.PAUSE, startId)
            IndexingJobState.PAUSED,
            IndexingJobState.WAITING_FOR_INPUT,
            -> publishAndStop(ledger, startId, detachNotification = true)

            IndexingJobState.CANCELLED,
            IndexingJobState.COMPLETE,
            -> settleRecoveredTerminalJob(ledger, startId)
        }
    }

    /** A terminal ledger makes only its job-private staging work disposable. */
    private fun settleRecoveredTerminalJob(ledger: IndexingJobLedger, startId: Int): Boolean =
        V2IndexingTerminalCleanupRecoveryPolicy.settle(
            state = ledger.state,
            cleanup = { executor.cleanupTerminalJob(ledger) },
            onRetryRequired = { error, message ->
                Log.e(TAG, "Unable to remove terminal indexing work", error)
                publishError(ledger.jobSpec.jobId, message)
                // Keep the pointer so the next visible recovery retries this exact cleanup.
                finishForeground(startId, detach = true)
            },
            onClean = {
                activePointer.clear(ledger.jobSpec.jobId)
                publishAndStop(ledger, startId, detachNotification = true)
            },
        )

    private fun startOrResume(
        initial: IndexingJobLedger,
        recover: Boolean,
        startId: Int,
    ) {
        reserveServiceRecommendationReservation(initial.jobSpec.jobId)
        if (!lifetimeGate.mayStartWork()) return
        if (routeStartToActiveRunner(initial, recover, startId)) return
        synchronized(executorStartLock) {
            if (!lifetimeGate.mayStartWork()) return
            // A second command can pass the optimistic check while the first caller is claiming
            // the durable lease. Recheck under the same lock that publishes runnerJob.
            if (routeStartToActiveRunner(initial, recover, startId)) return
            val token = try {
                repository.claimExecutor(initial.jobSpec.jobId)
            } catch (error: Exception) {
                Log.w(TAG, "Indexing executor is still active or draining", error)
                publish(repository.require(initial.jobSpec.jobId))
                finishForeground(startId, detach = true)
                return
            }
            if (!lifetimeGate.mayStartWork()) {
                runCatching { repository.releaseExecutor(token) }
                return
            }
            var ledger = repository.require(initial.jobSpec.jobId)
            try {
                if (recover && ledger.state == IndexingJobState.INTERRUPTED) {
                    repository.retryFailed(ledger.jobSpec.jobId, RetryTrigger.PROCESS_RESTART)
                    ledger = repository.require(ledger.jobSpec.jobId)
                }
                ledger = when (ledger.state) {
                    IndexingJobState.PLANNED -> repository.startAuthorized(token)
                    IndexingJobState.PAUSED,
                    IndexingJobState.WAITING_FOR_INPUT,
                    IndexingJobState.INTERRUPTED,
                    IndexingJobState.READY_TO_RESUME,
                    -> repository.resumeAuthorized(token)
                    IndexingJobState.RUNNING,
                    IndexingJobState.ACTIVATING,
                    -> ledger
                    else -> throw IllegalStateException("Cannot execute ${ledger.state}")
                }
            } catch (error: Throwable) {
                runCatching { repository.releaseExecutor(token) }
                Log.e(TAG, "Unable to resume indexing", error)
                publishError(
                    ledger.jobSpec.jobId,
                    "Indexing could not resume. Reopen On-device indexing and try again.",
                )
                finishForeground(startId, detach = true)
                return
            }

            if (!lifetimeGate.mayStartWork()) {
                val checkpointed = if (ledger.state in setOf(
                        IndexingJobState.RUNNING,
                        IndexingJobState.PAUSE_REQUESTED,
                        IndexingJobState.ACTIVATING,
                    )
                ) {
                    repository.checkpointForMediaProcessingTimeout(
                        ledger.jobSpec.jobId,
                        MEDIA_PROCESSING_TIMEOUT_REASON,
                    )
                } else {
                    ledger
                }
                runCatching { repository.releaseExecutor(token) }
                publish(checkpointed)
                return
            }

            activePointer.write(ledger.jobSpec.jobId)
            val control = ServiceExecutorControl(
                ledger.jobSpec.jobId,
                ledger.executionProfile,
                emptySet(),
            )
            runnerControl = control
            runnerToken = token
            if (!lifetimeGate.mayStartWork()) {
                control.requestStop(StopReason.TIMEOUT)
                val checkpointed = repository.checkpointForMediaProcessingTimeout(
                    ledger.jobSpec.jobId,
                    MEDIA_PROCESSING_TIMEOUT_REASON,
                )
                runCatching { repository.releaseExecutor(token) }
                runnerControl = null
                runnerToken = null
                publish(checkpointed)
                return
            }
            acquireWakeLock()
            publish(
                ledger,
                event = runSetupEvent(
                    ledger,
                    V2MeasuredWorkStage.RECOMMENDATION_RESOURCE_HANDOFF,
                    "Waiting for recommendation resources before loading the indexing models",
                ),
                verifiedPcmWorkIds = control.verifiedPcmSnapshot(ledger),
            )
            runnerJob = serviceScope.launch {
                runExecutor(token, control, startId)
            }
        }
    }

    private fun routeStartToActiveRunner(
        initial: IndexingJobLedger,
        recover: Boolean,
        startId: Int,
    ): Boolean {
        if (runnerJob?.isActive != true) return false
        val runningJobId = runnerToken?.jobId ?: runnerControl?.jobId
        var preserveCurrentRecommendationReservation = false
        if (runningJobId != null && runningJobId != initial.jobSpec.jobId) {
            Log.w(
                TAG,
                "Deferring ${initial.jobSpec.jobId}; executor is draining $runningJobId",
            )
            queuePendingRunnerCommand(
                PendingRunnerCommand(
                    type = PendingRunnerCommandType.START,
                    jobId = initial.jobSpec.jobId,
                    recover = recover,
                    startId = startId,
                ),
            )
        }
        val current = repository.require(runningJobId ?: initial.jobSpec.jobId)
        if (runningJobId == initial.jobSpec.jobId) {
            val pendingType = if (V2IndexingRunnerDrainPolicy.shouldReplayStart(current.state)) {
                PendingRunnerCommandType.START
            } else {
                PendingRunnerCommandType.SETTLE
            }
            queuePendingRunnerCommand(
                PendingRunnerCommand(
                    pendingType,
                    initial.jobSpec.jobId,
                    recover,
                    startId,
                ),
            )
            preserveCurrentRecommendationReservation = pendingType == PendingRunnerCommandType.START
        }
        publish(
            current,
            preserveRecommendationReservation = preserveCurrentRecommendationReservation,
        )
        return true
    }

    private fun runSetupEvent(
        ledger: IndexingJobLedger,
        stage: V2MeasuredWorkStage,
        detail: String,
    ) = V2IndexingExecutorEvent(
        jobId = ledger.jobSpec.jobId,
        workId = null,
        trackOrdinal = null,
        trackTitle = null,
        stage = stage,
        completedUnits = null,
        totalUnits = null,
        detail = detail,
        scope = V2IndexingEventScope.INDEXING_RUN,
    )

    private suspend fun runExecutor(
        token: V2ExecutorLeaseToken,
        control: ServiceExecutorControl,
        startId: Int,
    ) {
        val heartbeat = serviceScope.launch {
            while (isActive) {
                delay(HEARTBEAT_INTERVAL_MS)
                repository.heartbeatExecutor(token)
                renewWakeLock()
            }
        }
        try {
            check(isServiceRecommendationReservationHeld(token.jobId)) {
                "Indexing cannot load music models without recommendation exclusion"
            }
            if (!awaitExclusiveRecommendationResources()) {
                throw RecommendationHandoffTimeoutException()
            }
            when (executor.run(token, control)) {
                V2IndexingExecutorOutcome.COMPLETE -> {
                    val complete = repository.require(token.jobId)
                    settleRecoveredTerminalJob(complete, startId)
                }
                V2IndexingExecutorOutcome.WAITING_FOR_INPUT -> {
                    publishAndStop(repository.require(token.jobId), startId, detachNotification = true)
                }
            }
        } catch (_: RecommendationHandoffTimeoutException) {
            repository.requestPause(token.jobId)
            val paused = repository.finishPauseAfterExecutorStops(
                token,
                RECOMMENDATION_DRAIN_TIMEOUT_MESSAGE,
            )
            publishAndStop(paused, startId, detachNotification = true)
        } catch (_: V2IndexingControlFlowException) {
            when (control.stopReason.get()) {
                StopReason.CANCEL -> finishCancellation(token, startId)
                StopReason.PAUSE,
                StopReason.TIMEOUT,
                -> {
                    val timedOut = control.stopReason.get() == StopReason.TIMEOUT
                    val current = repository.require(token.jobId)
                    val paused = if (timedOut && current.state == IndexingJobState.ACTIVATING) {
                        repository.interruptActivationForMediaProcessingTimeout(
                            token,
                            MEDIA_PROCESSING_TIMEOUT_REASON,
                        )
                    } else {
                        repository.finishPauseAfterExecutorStops(
                            token,
                            if (timedOut) MEDIA_PROCESSING_TIMEOUT_REASON
                            else "paused between processing steps",
                        )
                    }
                    publishAndStop(paused, startId, detachNotification = true)
                }
                null -> reconcileUnexpectedStop(
                    token,
                    "Indexing stopped unexpectedly. Completed work was saved; reopen " +
                        "On-device indexing to resume.",
                    startId,
                )
            }
        } catch (error: Throwable) {
            if (control.stopReason.get() == StopReason.TIMEOUT) {
                Log.i(TAG, "Executor unwound after durable media-processing timeout", error)
            } else {
                Log.e(TAG, "V2 indexing executor failed", error)
                val current = runCatching { repository.require(token.jobId) }.getOrNull()
                if (current?.state == IndexingJobState.COMPLETE) {
                    // Publication is already durable. Retry only disposable private cleanup, and
                    // retain ownership if it still cannot be verified.
                    settleRecoveredTerminalJob(current, startId)
                } else {
                    val listenerMessage = if (current?.state == IndexingJobState.ACTIVATING) {
                        "The music index update did not finish. The previous index remains active."
                    } else {
                        "Indexing stopped unexpectedly. Completed work was saved; reopen " +
                            "On-device indexing to resume."
                    }
                    reconcileUnexpectedStop(
                        token,
                        listenerMessage,
                        startId,
                    )
                }
            }
        } finally {
            // Service teardown cancels serviceScope. Lease release and durable ETA persistence must
            // still complete or an explicit same-process resume can be blocked by the old lease.
            withContext(NonCancellable) {
                heartbeat.cancelAndJoin()
                saveEtaSamples()
                releaseWakeLock()
                runnerControl = null
                runnerToken = null
                runnerJob = null
                runCatching { repository.releaseExecutor(token) }
                drainPendingRunnerCommand()
            }
        }
    }

    private fun drainPendingRunnerCommand() {
        if (!lifetimeGate.mayStartWork()) {
            pendingRunnerCommand.set(null)
            return
        }
        val pending = pendingRunnerCommand.getAndSet(null) ?: return
        serviceScope.launch {
            commandMutex.withLock {
                if (!lifetimeGate.mayStartWork()) return@withLock
                val current = runCatching { repository.require(pending.jobId) }.getOrNull()
                    ?: return@withLock
                when {
                    current.state == IndexingJobState.CANCELLING ->
                        finishCancellationWithoutRunner(current, pending.startId)

                    current.state in setOf(
                        IndexingJobState.CANCELLED,
                        IndexingJobState.COMPLETE,
                    ) -> settleRecoveredTerminalJob(current, pending.startId)

                    pending.type == PendingRunnerCommandType.START &&
                        V2IndexingRunnerDrainPolicy.shouldReplayStart(current.state) ->
                        startOrResume(current, pending.recover, pending.startId)

                    else -> publishAndStop(current, pending.startId, detachNotification = true)
                }
            }
        }
    }

    private fun queuePendingRunnerCommand(command: PendingRunnerCommand) {
        pendingRunnerCommand.updateAndGet { existing ->
            if (existing?.jobId == command.jobId &&
                existing.type == PendingRunnerCommandType.START &&
                command.type == PendingRunnerCommandType.SETTLE
            ) {
                existing.copy(
                    recover = existing.recover || command.recover,
                    startId = maxOf(existing.startId, command.startId),
                )
            } else {
                command
            }
        }
        // Close the boundary where the runner clears itself between the caller's active check and
        // this follow-up reservation. Either side may drain; getAndSet keeps it one-shot.
        if (runnerJob?.isActive != true) drainPendingRunnerCommand()
    }

    private fun drainPendingPreflightCommand() {
        val pending = pendingPreflightCommand.getAndSet(null) ?: return
        serviceScope.launch {
            commandMutex.withLock {
                // Preflight publishes its immutable ledger before it finishes unwinding. A Start
                // received during that handoff belongs to the ledger/runner, not to the terminal
                // MATERIALIZED intent. Dispatching it back to preflight would stop the newborn
                // executor as an "incomplete" plan.
                val ledger = runCatching { repository.require(pending.jobId) }.getOrNull()
                if (ledger != null) {
                    Log.i(
                        TAG,
                        "Routing pending preflight ${pending.type} to materialized job " +
                            pending.jobId,
                    )
                    if (pending.type == PendingRunnerCommandType.START) {
                        startOrResume(ledger, recover = false, pending.startId)
                    } else if (runnerJob?.isActive == true) {
                        transferRunnerSettlementOwnership(ledger.jobSpec.jobId, pending.startId)
                        publish(ledger)
                    } else {
                        publishAndStop(ledger, pending.startId, detachNotification = true)
                    }
                    return@withLock
                }

                val intent = runCatching { preflightStore.require(pending.jobId) }.getOrNull()
                if (intent != null) {
                    when {
                        intent.state == V2IndexingPreflightIntentState.CANCEL_REQUESTED ->
                            finishPreflightCancellation(intent, pending.startId)
                        pending.type == PendingRunnerCommandType.START ->
                            startOrResumePreflight(intent, pending.startId)
                        else -> {
                            publishPreflight(intent, forceNotification = true)
                            finishForeground(pending.startId, detach = true)
                        }
                    }
                    return@withLock
                }

                finishForeground(pending.startId, detach = true)
            }
        }
    }

    private fun queuePendingPreflightCommand(command: PendingPreflightCommand) {
        pendingPreflightCommand.set(command)
        if (preflightJob?.isActive != true) drainPendingPreflightCommand()
    }

    private fun requestPause(
        ledger: IndexingJobLedger,
        reason: StopReason,
        startId: Int,
    ) {
        if (reason == StopReason.PAUSE && ledger.state == IndexingJobState.ACTIVATING) {
            // Generation publication is one atomic filesystem transaction. A stale notification
            // action must not turn an already-publishing job into an artificial interruption.
            publish(ledger)
            if (runnerJob?.isActive == true) {
                queuePendingRunnerCommand(
                    PendingRunnerCommand(
                        PendingRunnerCommandType.SETTLE,
                        ledger.jobSpec.jobId,
                        recover = false,
                        startId = startId,
                    ),
                )
            }
            return
        }
        val updated = runCatching {
            if (reason == StopReason.TIMEOUT) {
                repository.requestMediaProcessingTimeoutPause(
                    ledger.jobSpec.jobId,
                    MEDIA_PROCESSING_TIMEOUT_REASON,
                )
            } else {
                repository.requestPause(ledger.jobSpec.jobId)
            }
        }
            .getOrElse { ledger }
        val shouldStop = updated.state == IndexingJobState.PAUSE_REQUESTED ||
            updated.state == IndexingJobState.PAUSED ||
            (reason == StopReason.TIMEOUT && updated.state == IndexingJobState.ACTIVATING)
        if (shouldStop) runnerControl?.requestStop(reason)
        if (runnerJob?.isActive != true) {
            publishAndStop(updated, startId, detachNotification = true)
        } else {
            if (shouldStop) {
                queuePendingRunnerCommand(
                    PendingRunnerCommand(
                        PendingRunnerCommandType.SETTLE,
                        updated.jobSpec.jobId,
                        recover = false,
                        startId = startId,
                    ),
                )
            }
            publish(updated)
        }
    }

    private fun requestCancel(ledger: IndexingJobLedger, startId: Int) {
        val cancelling = repository.requestCancel(ledger.jobSpec.jobId)
        runnerControl?.requestStop(StopReason.CANCEL)
        if (runnerJob?.isActive != true) {
            finishCancellationWithoutRunner(cancelling, startId)
        } else {
            queuePendingRunnerCommand(
                PendingRunnerCommand(
                    PendingRunnerCommandType.SETTLE,
                    cancelling.jobSpec.jobId,
                    recover = false,
                    startId = startId,
                ),
            )
            publish(cancelling)
        }
    }

    private fun finishCancellationWithoutRunner(ledger: IndexingJobLedger, startId: Int) {
        val token = runCatching { repository.claimExecutor(ledger.jobSpec.jobId) }
            .getOrElse { error ->
                Log.e(TAG, "Unable to acquire indexing cancellation lease", error)
                publishError(
                    ledger.jobSpec.jobId,
                    "Indexing could not stop yet. Reopen On-device indexing and try again.",
                )
                finishForeground(startId, detach = true)
                return
            }
        try {
            finishCancellation(token, startId)
        } finally {
            runCatching { repository.releaseExecutor(token) }
        }
    }

    private fun finishCancellation(token: V2ExecutorLeaseToken, startId: Int) {
        executor.cleanupCancelledJob(token)
        val cancelled = repository.finishCancellationAfterCleanup(token)
        activePointer.clear(token.jobId)
        publishAndStop(cancelled, startId, detachNotification = true)
    }

    private fun retry(
        command: V2IndexingServiceCommand,
        observedLedger: IndexingJobLedger,
        startId: Int,
    ) {
        val current = repository.require(command.jobId)
        if (settleInadmissibleFailureCommand(current, observedLedger, startId)) return
        val trigger = requireNotNull(command.retryTrigger)
        val updated = command.workId?.let { workId ->
            repository.retryTrack(command.jobId, workId, trigger)
        } ?: repository.retryFailed(command.jobId, trigger).ledger
        val unresolved = updated.tracks.any {
            it.state == IndexingTrackState.RETRYABLE_FAILURE ||
                it.state == IndexingTrackState.BLOCKED_FAILURE
        }
        if (!unresolved && updated.state == IndexingJobState.WAITING_FOR_INPUT) {
            startOrResume(updated, recover = false, startId)
        } else {
            publishAndStop(updated, startId, detachNotification = true)
        }
    }

    private fun skip(
        command: V2IndexingServiceCommand,
        observedLedger: IndexingJobLedger,
        startId: Int,
    ) {
        val current = repository.require(command.jobId)
        if (settleInadmissibleFailureCommand(current, observedLedger, startId)) return
        val updated = repository.skipTrack(command.jobId, requireNotNull(command.workId))
        val unresolved = updated.tracks.any {
            it.state == IndexingTrackState.RETRYABLE_FAILURE ||
                it.state == IndexingTrackState.BLOCKED_FAILURE
        }
        if (!unresolved && updated.state == IndexingJobState.WAITING_FOR_INPUT) {
            startOrResume(updated, recover = false, startId)
        } else {
            publishAndStop(updated, startId, detachNotification = true)
        }
    }

    /**
     * PendingIntents can outlive the state which exposed them. Preserve a live runner and hand its
     * eventual foreground teardown to this newest start ID instead of applying a stale mutation.
     */
    private fun settleInadmissibleFailureCommand(
        current: IndexingJobLedger,
        observed: IndexingJobLedger,
        startId: Int,
    ): Boolean {
        val active = runnerJob?.isActive == true
        if (current.jobSpec.specId == observed.jobSpec.specId &&
            canAcceptIndexingFailureCommand(current.state, active)
        ) {
            return false
        }
        val runningJobId = (runnerToken?.jobId ?: runnerControl?.jobId)
            ?.takeIf { active }
        if (runningJobId != null) {
            val running = repository.require(runningJobId)
            publish(running)
            transferRunnerSettlementOwnership(runningJobId, startId)
        } else {
            publishAndStop(current, startId, detachNotification = true)
        }
        return true
    }

    /** Preserve an already-requested resume while transferring teardown to the newest start ID. */
    private fun transferRunnerSettlementOwnership(jobId: String, startId: Int) {
        pendingRunnerCommand.updateAndGet { existing ->
            if (existing?.jobId == jobId) {
                existing.copy(startId = maxOf(existing.startId, startId))
            } else {
                PendingRunnerCommand(
                    PendingRunnerCommandType.SETTLE,
                    jobId,
                    recover = false,
                    startId = startId,
                )
            }
        }
        if (runnerJob?.isActive != true) drainPendingRunnerCommand()
    }

    private fun changeProfile(command: V2IndexingServiceCommand, startId: Int) {
        requireNotNull(command.profile)
        val updated = repository.changeProfile(
            command.jobId,
            V2IndexingExecutionProfile.FULL,
        )
        runnerControl?.let { control ->
            control.profile.set(updated.executionProfile)
            // Never attribute an interval straddling a profile change to either rate bucket.
            control.clearMeasurements()
        }
        publish(updated)
        if (runnerJob?.isActive != true) {
            finishForeground(startId, detach = true)
        } else {
            queuePendingRunnerCommand(
                PendingRunnerCommand(
                    PendingRunnerCommandType.SETTLE,
                    updated.jobSpec.jobId,
                    recover = false,
                    startId = startId,
                ),
            )
        }
    }

    private fun reconcileUnexpectedStop(
        token: V2ExecutorLeaseToken,
        message: String,
        startId: Int,
    ) {
        val reconciled = runCatching { repository.reconcileUnexpectedExecutorStop(token).ledger }
            .getOrElse { repository.require(token.jobId) }
        Log.w(TAG, message)
        publishAndStop(reconciled, startId, detachNotification = true)
    }

    private inner class ServiceExecutorControl(
        val jobId: String,
        initialProfile: V2IndexingExecutionProfile,
        initialVerifiedPcmWorkIds: Set<String>,
    ) : V2IndexingExecutorControl {
        val stopReason = AtomicReference<StopReason?>(null)
        val profile = AtomicReference(initialProfile)
        private val rateRecorder = V2IndexingEventRateRecorder(estimator)
        private val verifiedPcmTracker = V2VerifiedPcmProgressTracker(initialVerifiedPcmWorkIds)

        override fun throwIfStopped() {
            stopReason.get()?.let { reason ->
                throw V2IndexingControlFlowException("Indexing requested ${reason.name.lowercase()}")
            }
        }

        override fun executionProfile(): V2IndexingExecutionProfile =
            V2IndexingExecutionProfile.FULL

        override fun onProgress(event: V2IndexingExecutorEvent) {
            if (event.jobId != jobId) return
            val ledger = repository.require(jobId)
            recordMeasurement(event, ledger, this)
            publish(
                ledger,
                event,
                verifiedPcmWorkIds = verifiedPcmTracker.snapshot(ledger, event),
            )
        }

        fun requestStop(reason: StopReason) {
            stopReason.compareAndSet(null, reason)
        }

        fun recordRateEvent(
            event: V2IndexingExecutorEvent,
            sourceMime: String?,
            observedAtElapsedMs: Long,
        ): Boolean = rateRecorder.onEvent(
            event = event,
            profile = executionProfile(),
            sourceMime = sourceMime,
            observedAtElapsedMs = observedAtElapsedMs,
        )

        fun clearMeasurements() = rateRecorder.clear()

        fun verifiedPcmSnapshot(ledger: IndexingJobLedger): Set<String> =
            verifiedPcmTracker.snapshot(ledger)
    }

    private inner class ServicePreflightObserver(
        private val jobId: String,
    ) : V2IndexingPreflightObserver {
        private val stopRequest = AtomicReference<PreflightStopRequest?>(null)
        private var lastPersistedAtElapsedMs = 0L
        private var lastPersistedPhase: V2IndexingPreflightPhase? = null

        override fun onProgress(progress: V2IndexingPreflightProgress) {
            throwIfCancelled()
            val current = preflightStore.require(jobId)
            if (current.state == V2IndexingPreflightIntentState.REQUESTED) {
                publishPreflight(current.copy(progress = progress))
                return
            }
            if (current.state != V2IndexingPreflightIntentState.PLANNING) {
                throwIfCancelled()
                throw IllegalStateException("Preflight entered ${current.state}")
            }
            val nowElapsed = SystemClock.elapsedRealtime()
            val phaseChanged = progress.phase != lastPersistedPhase
            val stageComplete = progress.completedUnits != null &&
                progress.completedUnits == progress.totalUnits
            val shouldPersist = phaseChanged || stageComplete ||
                nowElapsed - lastPersistedAtElapsedMs >= PREFLIGHT_PROGRESS_SAVE_INTERVAL_MS
            val display = if (shouldPersist) {
                preflightStore.persistProgressOverlay(
                    jobId = jobId,
                    progress = progress,
                    nowEpochMs = System.currentTimeMillis(),
                ).also {
                    lastPersistedAtElapsedMs = nowElapsed
                    lastPersistedPhase = progress.phase
                }
            } else {
                current.copy(progress = progress)
            }
            publishPreflight(display)
            throwIfCancelled()
        }

        override fun throwIfCancelled() {
            stopRequest.get()?.let { stop ->
                if (stop.userCancellation) throw V2IndexingPreflightCancelledException()
                throw IllegalStateException(stop.reason)
            }
            if (preflightStore.load(jobId)?.state ==
                V2IndexingPreflightIntentState.CANCEL_REQUESTED
            ) {
                throw V2IndexingPreflightCancelledException()
            }
        }

        fun requestStop(reason: String, userCancellation: Boolean) {
            stopRequest.compareAndSet(
                null,
                PreflightStopRequest(reason, userCancellation),
            )
        }
    }

    private data class PreflightStopRequest(
        val reason: String,
        val userCancellation: Boolean,
    )

    private data class PendingRunnerCommand(
        val type: PendingRunnerCommandType,
        val jobId: String,
        val recover: Boolean,
        val startId: Int,
    )

    private data class PendingPreflightCommand(
        val type: PendingRunnerCommandType,
        val jobId: String,
        val startId: Int,
    )

    @Synchronized
    private fun recordMeasurement(
        event: V2IndexingExecutorEvent,
        ledger: IndexingJobLedger,
        control: ServiceExecutorControl,
    ) {
        val sourceMime = ledger.jobSpec.tracks
            .firstOrNull { it.workId == event.workId }
            ?.finalizedAudioSpan?.container?.mime
        val now = SystemClock.elapsedRealtime()
        if (control.recordRateEvent(event, sourceMime, now) &&
            now - lastEtaSaveAtElapsedMs >= ETA_SAVE_INTERVAL_MS
        ) {
            saveEtaSamples()
        }
    }

    private fun publish(
        ledger: IndexingJobLedger,
        event: V2IndexingExecutorEvent? = null,
        preserveRecommendationReservation: Boolean = false,
        verifiedPcmWorkIds: Set<String>? = null,
        settlingNotification: Boolean = false,
    ) {
        if (ledger.state in RECOMMENDATION_QUIESCENT_JOB_STATES &&
            !preserveRecommendationReservation
        ) {
            releaseServiceRecommendationReservation(ledger.jobSpec.jobId)
        }
        // Replacement/deletion effects and exact-base provenance are knowable only after the
        // private target embedding set exists. Until GraphUpdater emits its bound plan, omitting
        // graph work from ETA is truthful; treating every selected row as an append is not.
        val graphPlan = event?.graphWorkPlan
        val resolvedVerifiedPcmWorkIds = verifiedPcmWorkIds.orEmpty()
        val overallWork = V2IndexingOverallWorkPlanner.snapshot(
            ledger,
            event,
            graphPlan,
            resolvedVerifiedPcmWorkIds,
        )
        val eta = estimator.estimate(
            overallWork.remainingMeasuredWork(),
            V2IndexingExecutionProfile.FULL,
        )
        _state.value = V2IndexingServiceStateMapper.map(
            ledger = ledger,
            event = event,
            eta = eta,
            overallWork = overallWork,
        )
        updateNotification(
            ledger,
            event,
            settling = settlingNotification,
        )
    }

    private fun publishPreflight(
        intent: V2IndexingPreflightIntent,
        forceNotification: Boolean = false,
        preserveRecommendationReservation: Boolean = false,
    ) {
        if (intent.state in RECOMMENDATION_QUIESCENT_PREFLIGHT_STATES &&
            !preserveRecommendationReservation
        ) {
            releaseServiceRecommendationReservation(intent.jobId)
        }
        _state.value = V2IndexingServiceStateMapper.mapPreflight(intent)
        val now = SystemClock.elapsedRealtime()
        val lastPreflight = lastPreflightNotificationAtElapsedMs
        if (!forceNotification &&
            lastPreflight != null &&
            now - lastPreflight < PREFLIGHT_NOTIFICATION_THROTTLE_MS
        ) return
        lastPreflightNotificationAtElapsedMs = now
        getSystemService(NotificationManager::class.java).notify(
            NOTIFICATION_ID,
            buildPreflightNotification(intent),
        )
        notificationUpdatePolicy.invalidate()
    }

    private fun publishAndStop(
        ledger: IndexingJobLedger,
        startId: Int,
        detachNotification: Boolean,
    ) {
        // Detaching preserves the notification. Publish any changed terminal/stopped content now,
        // even inside the ordinary cadence, so running text and actions cannot be left behind.
        publish(ledger, settlingNotification = true)
        finishForeground(startId, detachNotification)
    }

    private suspend fun awaitExclusiveRecommendationResources(): Boolean {
        Log.i(TAG, "Waiting for recommendation resources before indexing")
        MainViewModel.releaseProcessRetrievalResourcesForIndexing()
        val released = RadioService.suspendAndReleaseRecommendationResources(
            timeoutMs = RECOMMENDATION_DRAIN_TIMEOUT_MS,
        )
        if (released) Log.i(TAG, "Recommendation resources released before indexing")
        return released
    }

    private fun releaseLaunchRecommendationReservation(jobId: String) {
        if (RecommendationWorkAdmission.release(
                RecommendationWorkAdmission.indexingLaunchOwner(jobId),
            )
        ) {
            RadioService.kickDeferredRecovery(applicationContext)
        }
    }

    private fun reserveServiceRecommendationReservation(jobId: String) {
        val owner = synchronized(recommendationReservationLock) {
            recommendationReservationOwners.getOrPut(jobId) {
                RecommendationWorkAdmission.indexingServiceOwner(
                    "$recommendationServiceIdentity:$jobId",
                )
            }
        }
        RecommendationWorkAdmission.reserve(owner)
    }

    private fun releaseServiceRecommendationReservation(jobId: String) {
        val owner = synchronized(recommendationReservationLock) {
            recommendationReservationOwners.remove(jobId)
        } ?: return
        if (RecommendationWorkAdmission.release(owner)) {
            RadioService.kickDeferredRecovery(applicationContext)
        }
    }

    private fun isServiceRecommendationReservationHeld(jobId: String): Boolean {
        val owner = synchronized(recommendationReservationLock) {
            recommendationReservationOwners[jobId]
        } ?: return false
        return RecommendationWorkAdmission.isReservedBy(owner)
    }

    private fun releaseAllServiceRecommendationReservations() {
        val owners = synchronized(recommendationReservationLock) {
            recommendationReservationOwners.values.toList().also {
                recommendationReservationOwners.clear()
            }
        }
        var becameAvailable = false
        owners.forEach { owner ->
            becameAvailable = RecommendationWorkAdmission.release(owner) || becameAvailable
        }
        if (becameAvailable) RadioService.kickDeferredRecovery(applicationContext)
    }

    private fun publishError(jobId: String?, message: String) {
        _state.value = IndexingState.Error(jobId, message.take(2_048))
        notificationUpdatePolicy.invalidate()
        lastPreflightNotificationAtElapsedMs = null
        val manager = getSystemService(NotificationManager::class.java)
        manager.notify(NOTIFICATION_ID, buildNotification(jobId, message.take(160)))
    }

    private fun publishActivePointerError(
        jobId: String?,
        operation: String,
        inspection: V2ActiveIndexingJobPointerInspection.Unreadable,
    ) {
        Log.e(TAG, "Refusing to $operation: active pointer is ${inspection.reason}")
        publishError(jobId, ACTIVE_POINTER_UNREADABLE_MESSAGE)
    }

    private fun updateNotification(
        ledger: IndexingJobLedger,
        event: V2IndexingExecutorEvent? = null,
        force: Boolean = false,
        settling: Boolean = false,
    ) {
        val now = SystemClock.elapsedRealtime()
        val presentation = notificationPresentation(ledger, event)
        if (!notificationUpdatePolicy.shouldPublish(
                candidate = presentation,
                observedAtElapsedMs = now,
                force = force,
                settling = settling,
            )
        ) return
        getSystemService(NotificationManager::class.java).notify(
            NOTIFICATION_ID,
            buildNotification(ledger, presentation),
        )
        lastPreflightNotificationAtElapsedMs = null
    }

    private fun notificationPresentation(
        ledger: IndexingJobLedger,
        event: V2IndexingExecutorEvent?,
    ): V2IndexingNotificationPresentation {
        val evidence = indexingNotificationEvidence(
            ledger.state,
            event,
        )
        val active = ledger.state in ACTIVE_STATES
        return V2IndexingNotificationPresentation(
            jobId = ledger.jobSpec.jobId,
            state = ledger.state,
            title = evidence.title,
            text = evidence.text,
            progressPercent = evidence.fraction?.let { (it * 100f).roundToInt() },
            indeterminateProgress = evidence.fraction == null && active,
            ongoing = active,
            actionSet = notificationActionSet(ledger),
        )
    }

    private fun buildNotification(
        ledger: IndexingJobLedger,
        presentation: V2IndexingNotificationPresentation,
    ): Notification {
        val builder = baseNotification(
            ledger.jobSpec.jobId,
            presentation.title,
            presentation.text,
        )
            .setOnlyAlertOnce(true)
            .setOngoing(presentation.ongoing)
        if (presentation.progressPercent != null) {
            builder.setProgress(100, presentation.progressPercent, false)
        } else if (presentation.indeterminateProgress) {
            builder.setProgress(0, 0, true)
        }
        addNotificationActions(builder, ledger, presentation.actionSet)
        return builder.build()
    }

    private fun buildPreflightNotification(intent: V2IndexingPreflightIntent): Notification {
        val text = preflightStatusEvidenceText(
            state = intent.state,
            failureCode = intent.failureCode,
            progressMessage = intent.progress.message,
        )
        val active = intent.state in setOf(
            V2IndexingPreflightIntentState.REQUESTED,
            V2IndexingPreflightIntentState.PLANNING,
            V2IndexingPreflightIntentState.RESOLVED_WITH_EXECUTABLE_ROWS,
            V2IndexingPreflightIntentState.CANCEL_REQUESTED,
        )
        val builder = baseNotification(intent.jobId, ON_DEVICE_INDEXING_NOTIFICATION_TITLE, text)
            .setOnlyAlertOnce(true)
            .setOngoing(active)
        val completed = intent.progress.completedUnits
        val total = intent.progress.totalUnits
        if (completed != null && total != null && total in 1..Int.MAX_VALUE &&
            completed in 0..total
        ) {
            builder.setProgress(total.toInt(), completed.toInt(), false)
        } else if (active) {
            builder.setProgress(0, 0, true)
        }
        if (V2IndexingPreflightControlPolicy.isExplicitlyResumable(intent.state)) {
            builder.addAction(
                0,
                "Retry",
                commandPendingIntent(V2IndexingServiceIntents.resume(this, intent.jobId), 21),
            )
        }
        if (intent.state in setOf(
                V2IndexingPreflightIntentState.REQUESTED,
                V2IndexingPreflightIntentState.PLANNING,
                V2IndexingPreflightIntentState.INTERRUPTED,
                V2IndexingPreflightIntentState.FAILED,
                V2IndexingPreflightIntentState.RESOLVED_WITH_EXECUTABLE_ROWS,
            )
        ) {
            builder.addAction(
                0,
                "Cancel",
                commandPendingIntent(V2IndexingServiceIntents.cancel(this, intent.jobId), 22),
            )
        }
        return builder.build()
    }

    private fun buildNotification(jobId: String?, message: String): Notification =
        baseNotification(jobId, "On-device indexing", message).build()

    private fun baseNotification(
        jobId: String?,
        title: String,
        text: String,
    ): NotificationCompat.Builder {
        val contentIntent = PendingIntent.getActivity(
            this,
            1,
            Intent(this, IndexingActivity::class.java).apply {
                jobId?.let { putExtra(V2IndexingServiceIntents.EXTRA_JOB_ID, it) }
            },
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_radio)
            .setContentTitle(title)
            .setContentText(text)
            .setStyle(NotificationCompat.BigTextStyle().bigText(text))
            .setContentIntent(contentIntent)
            .setCategory(NotificationCompat.CATEGORY_PROGRESS)
            .setVisibility(NotificationCompat.VISIBILITY_PUBLIC)
    }

    private fun addNotificationActions(
        builder: NotificationCompat.Builder,
        ledger: IndexingJobLedger,
        actionSet: V2IndexingNotificationActionSet,
    ) {
        val jobId = ledger.jobSpec.jobId
        when (actionSet) {
            V2IndexingNotificationActionSet.PAUSE -> builder.addAction(
                0,
                "Pause",
                commandPendingIntent(V2IndexingServiceIntents.pause(this, jobId), 10),
            )
            V2IndexingNotificationActionSet.RESUME -> builder.addAction(
                0,
                "Resume",
                commandPendingIntent(V2IndexingServiceIntents.resume(this, jobId), 11),
            )
            V2IndexingNotificationActionSet.RETRY -> builder.addAction(
                0,
                "Retry",
                commandPendingIntent(
                    V2IndexingServiceIntents.retry(
                        this,
                        jobId,
                        RetryTrigger.USER_REQUEST,
                    ),
                    12,
                ),
            )
            V2IndexingNotificationActionSet.NONE -> Unit
        }
        // Destructive job cancellation is confirmed in IndexingActivity. The notification keeps
        // only checkpoint-preserving controls; tapping its body opens the full job controls.
    }

    private fun notificationActionSet(
        ledger: IndexingJobLedger,
    ): V2IndexingNotificationActionSet = when {
        shouldOfferIndexingPauseAction(ledger.state) ->
            V2IndexingNotificationActionSet.PAUSE
        ledger.state in setOf(
            IndexingJobState.PAUSED,
            IndexingJobState.INTERRUPTED,
            IndexingJobState.READY_TO_RESUME,
        ) -> V2IndexingNotificationActionSet.RESUME
        ledger.state == IndexingJobState.WAITING_FOR_INPUT &&
            hasUserRetryEligibleFailure(
                ledger.state,
                ledger.tracks.mapNotNull { track ->
                    track.failures.singleOrNull { it.failureId == track.activeFailureId }
                        ?.retryTrigger
                },
            ) -> V2IndexingNotificationActionSet.RETRY
        else -> V2IndexingNotificationActionSet.NONE
    }

    private fun commandPendingIntent(intent: Intent, salt: Int): PendingIntent =
        PendingIntent.getForegroundService(
            this,
            requireNotNull(intent.getStringExtra(V2IndexingServiceIntents.EXTRA_JOB_ID)).hashCode() *
                31 + salt,
            intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )

    private fun createNotificationChannel() {
        getSystemService(NotificationManager::class.java).createNotificationChannel(
            NotificationChannel(
                CHANNEL_ID,
                "On-device indexing",
                NotificationManager.IMPORTANCE_LOW,
            ).apply {
                description = "Progress and pause/resume controls for on-device music indexing"
                setShowBadge(false)
            },
        )
    }

    @Synchronized
    private fun acquireWakeLock() {
        val lock = wakeLock ?: (getSystemService(POWER_SERVICE) as PowerManager)
            .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "$packageName:v2-indexing")
            .apply { setReferenceCounted(false) }
            .also { wakeLock = it }
        if (!lock.isHeld) lock.acquire(WAKE_LOCK_TIMEOUT_MS)
    }

    @Synchronized
    private fun renewWakeLock() {
        val lock = wakeLock ?: return
        if (lock.isHeld) lock.release()
        lock.acquire(WAKE_LOCK_TIMEOUT_MS)
    }

    @Synchronized
    private fun releaseWakeLock() {
        wakeLock?.takeIf { it.isHeld }?.release()
    }

    @Synchronized
    private fun saveEtaSamples() {
        val snapshot: V2PersistedStageRateSnapshot = estimator.snapshot()
        runCatching { etaStore.save(snapshot) }
            .onFailure { Log.w(TAG, "Unable to persist ETA calibration", it) }
        lastEtaSaveAtElapsedMs = SystemClock.elapsedRealtime()
    }

    private fun finishForeground(@Suppress("UNUSED_PARAMETER") startId: Int, detach: Boolean) {
        if (!V2IndexingServiceStartOwnershipPolicy.mayStopForeground(
                finishingStartId = startId,
                latestStartId = latestStartId.get(),
            )
        ) {
            stopSelfResult(startId)
            return
        }
        releaseWakeLock()
        stopForeground(
            if (detach) STOP_FOREGROUND_DETACH else STOP_FOREGROUND_REMOVE,
        )
        // Commands may add newer start IDs while this runner is alive. Once the durable runner
        // has quiesced, no start request should keep a non-foreground service instance around.
        stopSelfResult(startId)
    }

    /** Android's type-wide timeout must stop this service, independent of command ownership. */
    private fun forceStopForMediaProcessingTimeout(detach: Boolean) {
        releaseWakeLock()
        stopForeground(
            if (detach) STOP_FOREGROUND_DETACH else STOP_FOREGROUND_REMOVE,
        )
        stopSelf()
    }

    companion object {
        private const val TAG = "IndexingServiceV2"
        private const val CHANNEL_ID = "indexing_service_v2"
        private const val NOTIFICATION_ID = 2
        private const val WAKE_LOCK_TIMEOUT_MS = 10L * 60L * 1000L
        private const val HEARTBEAT_INTERVAL_MS = 60_000L
        private const val ETA_SAVE_INTERVAL_MS = 15_000L
        private const val PREFLIGHT_NOTIFICATION_THROTTLE_MS = 500L
        private const val RECOMMENDATION_DRAIN_TIMEOUT_MS = 60_000L
        private const val RECOMMENDATION_DRAIN_TIMEOUT_MESSAGE =
            "A radio queue is still finishing. Resume indexing after that queue completes."
        private const val PREFLIGHT_PROGRESS_SAVE_INTERVAL_MS = 1_000L
        internal const val MEDIA_PROCESSING_TIMEOUT_REASON =
            "Android's media-processing time limit was reached. Reopen the app and resume this job."
        internal const val ACTIVE_POINTER_UNREADABLE_MESSAGE =
            "Saved indexing work could not be read. It was kept; reopen " +
                "On-device indexing to recover."

        private val ACTIVE_STATES = setOf(
            IndexingJobState.RUNNING,
            IndexingJobState.PAUSE_REQUESTED,
            IndexingJobState.ACTIVATING,
            IndexingJobState.CANCELLING,
        )

        private val RECOMMENDATION_QUIESCENT_JOB_STATES = setOf(
            IndexingJobState.PAUSED,
            IndexingJobState.WAITING_FOR_INPUT,
            IndexingJobState.INTERRUPTED,
            IndexingJobState.READY_TO_RESUME,
            IndexingJobState.CANCELLED,
            IndexingJobState.COMPLETE,
        )

        private val RECOMMENDATION_QUIESCENT_PREFLIGHT_STATES = setOf(
            V2IndexingPreflightIntentState.INTERRUPTED,
            V2IndexingPreflightIntentState.FAILED,
            V2IndexingPreflightIntentState.CANCELLED,
            V2IndexingPreflightIntentState.RESOLVED_WITHOUT_EXECUTABLE_ROWS,
        )

        private val _state = kotlinx.coroutines.flow.MutableStateFlow<IndexingState>(
            IndexingState.Idle,
        )
        val state: kotlinx.coroutines.flow.StateFlow<IndexingState> = _state

        fun attach(context: Context, explicitJobId: String? = null) {
            val pointer = V2ActiveIndexingJobPointer(context.filesDir)
            if (explicitJobId != null) {
                val inspection = pointer.inspect()
                if (inspection is V2ActiveIndexingJobPointerInspection.Unreadable) {
                    Log.e(TAG, "Refusing explicit attach: active pointer is ${inspection.reason}")
                    _state.value = IndexingState.Error(null, ACTIVE_POINTER_UNREADABLE_MESSAGE)
                    return
                }
            }
            val repository = V2IndexingJobRepository.get(context)
            val jobId = explicitJobId ?: resolveActiveOrOrphanPreflight(
                context,
                pointer,
                repository,
            )
            if (jobId == null) {
                if (_state.value !is IndexingState.Error) _state.value = IndexingState.Idle
                return
            }
            runCatching { repository.require(jobId) }
                .onSuccess { ledger ->
                    val retained = (_state.value as? IndexingState.JobSnapshot)
                        ?.takeIf { it.jobId == ledger.jobSpec.jobId }
                    val retainedEvent = retained?.event
                    _state.value = V2IndexingServiceStateMapper.map(
                        ledger,
                        event = retainedEvent,
                        eta = retained?.eta
                            ?: V2StageAwareEtaEstimate(null, null, null, emptySet()),
                        overallWork = V2IndexingOverallWorkPlanner.snapshot(
                            ledger,
                            event = retainedEvent,
                            graphPlan = retainedEvent?.graphWorkPlan,
                            verifiedPcmWorkIds = emptySet(),
                        ),
                    )
                }
                .onFailure { ledgerError ->
                    Log.e(TAG, "Indexing job could not be opened for presentation", ledgerError)
                    runCatching { createPreflightStore(context).require(jobId) }
                        .onSuccess { intent ->
                            _state.value = V2IndexingServiceStateMapper.mapPreflight(intent)
                        }
                        .onFailure {
                            _state.value = IndexingState.Error(
                                jobId,
                                "Indexing job could not be opened. Reopen On-device indexing to recover.",
                            )
                        }
                }
        }

        /** Clears only the process-local presentation after a terminal job is dismissed. */
        fun resetState() {
            val snapshot = _state.value
            if (snapshot is IndexingState.JobSnapshot &&
                snapshot.jobState != IndexingJobState.COMPLETE &&
                snapshot.jobState != IndexingJobState.CANCELLED
            ) {
                return
            }
            if (snapshot is IndexingState.PreflightSnapshot &&
                snapshot.state != V2IndexingPreflightIntentState.CANCELLED &&
                snapshot.state !=
                V2IndexingPreflightIntentState.RESOLVED_WITHOUT_EXECUTABLE_ROWS
            ) {
                // MATERIALIZED can still own the durable pointer when ledger attachment failed.
                // Do not replace that recoverable evidence with a misleading process-local Idle.
                return
            }
            _state.value = IndexingState.Idle
        }

        fun submitPreflight(context: Context, intent: V2IndexingPreflightIntent) {
            val mutationOwner = MusicIndexMutationAdmission.newOwner(
                "indexing-preflight:${intent.jobId}",
            )
            if (!MusicIndexMutationAdmission.process.tryAcquire(mutationOwner)) {
                throw MusicIndexMutationBusyException()
            }
            val launchOwner = RecommendationWorkAdmission.indexingLaunchOwner(intent.jobId)
            try {
                RecommendationWorkAdmission.reserve(launchOwner)
                try {
                    val store = createPreflightStore(context)
                    val conflict = store.createIfNoConflict(intent) { existing ->
                        !V2IndexingPreflightControlPolicy.isTerminalRequest(existing.state)
                    }
                    if (conflict != null) {
                        throw IllegalStateException(
                            "An indexing job is already being prepared. Resume or cancel it first.",
                        )
                    }

                    val repository = V2IndexingJobRepository.get(context)
                    val pointer = V2ActiveIndexingJobPointer(context.filesDir)
                    try {
                        pointer.claim(intent.jobId) { existingJobId ->
                            runCatching { repository.require(existingJobId).state }.getOrNull()
                        }
                    } catch (error: Throwable) {
                        runCatching { store.delete(intent.jobId) }
                            .onFailure(error::addSuppressed)
                        throw error
                    }
                } finally {
                    check(MusicIndexMutationAdmission.process.release(mutationOwner)) {
                        "indexing preflight lost music-index mutation admission"
                    }
                }

                _state.value = V2IndexingServiceStateMapper.mapPreflight(intent)
                try {
                    ContextCompat.startForegroundService(
                        context,
                        V2IndexingServiceIntents.start(context, intent.jobId),
                    )
                } catch (error: Throwable) {
                    // The pointer and request intentionally remain durable. A later visible attach
                    // or boot recovery can start exactly the request the user submitted.
                    attach(context, intent.jobId)
                    throw error
                }
            } catch (error: Throwable) {
                releaseLaunchReservation(context, intent.jobId)
                throw error
            }
        }

        fun startJob(context: Context, jobId: String) {
            val repository = V2IndexingJobRepository.get(context)
            repository.require(jobId)
            val pointer = V2ActiveIndexingJobPointer(context.filesDir)
            val claim = pointer.claim(jobId) { existingJobId ->
                runCatching { repository.require(existingJobId).state }.getOrNull()
            }
            RecommendationWorkAdmission.reserve(
                RecommendationWorkAdmission.indexingLaunchOwner(jobId),
            )
            try {
                ContextCompat.startForegroundService(
                    context,
                    V2IndexingServiceIntents.start(context, jobId),
                )
            } catch (error: Throwable) {
                if (claim.changed) pointer.clear(jobId)
                releaseLaunchReservation(context, jobId)
                throw error
            }
        }

        fun pause(context: Context, jobId: String) = startCommand(
            context,
            jobId,
            V2IndexingServiceIntents.pause(context, jobId),
        )

        fun resume(context: Context, jobId: String) = startCommand(
            context,
            jobId,
            V2IndexingServiceIntents.resume(context, jobId),
        )

        fun cancel(context: Context, jobId: String) = startCommand(
            context,
            jobId,
            V2IndexingServiceIntents.cancel(context, jobId),
        )

        fun retry(context: Context, jobId: String, trigger: RetryTrigger) = startCommand(
            context,
            jobId,
            V2IndexingServiceIntents.retry(context, jobId, trigger),
        )

        fun retryTrack(
            context: Context,
            jobId: String,
            workId: String,
            trigger: RetryTrigger,
        ) = startCommand(
            context,
            jobId,
            V2IndexingServiceIntents.retryTrack(context, jobId, workId, trigger),
        )

        fun skip(context: Context, jobId: String, workId: String) = startCommand(
            context,
            jobId,
            V2IndexingServiceIntents.skip(context, jobId, workId),
        )

        fun changeProfile(
            context: Context,
            jobId: String,
            profile: V2IndexingExecutionProfile,
        ) = startCommand(
            context,
            jobId,
            V2IndexingServiceIntents.profile(context, jobId, profile),
        )

        fun recoverEligible(context: Context) {
            val repository = V2IndexingJobRepository.get(context)
            repository.reconcileStartup()
            val pointer = V2ActiveIndexingJobPointer(context.filesDir)
            val jobId = resolveActiveOrOrphanPreflight(context, pointer, repository) ?: return
            val ledger = runCatching { repository.require(jobId) }.getOrNull()
            if (ledger == null) {
                val intent = runCatching { createPreflightStore(context).require(jobId) }
                    .getOrNull() ?: return
                if (V2IndexingPreflightControlPolicy.shouldAutoRecover(intent.state)) {
                    try {
                        startCommand(context, jobId, V2IndexingServiceIntents.recover(context, jobId))
                    } catch (error: RuntimeException) {
                        if (V2IndexingRecoveryStartPolicy.mayDefer(error)) {
                            Log.w(TAG, "Background preflight recovery deferred", error)
                            attach(context, jobId)
                        } else {
                            throw error
                        }
                    }
                } else {
                    attach(context, jobId)
                }
                return
            }
            if (V2IndexingRecoveryServicePolicy.shouldStart(ledger.state)) {
                try {
                    startCommand(context, jobId, V2IndexingServiceIntents.recover(context, jobId))
                } catch (error: RuntimeException) {
                    if (V2IndexingRecoveryStartPolicy.mayDefer(error)) {
                        Log.w(TAG, "Background recovery deferred until a visible user action", error)
                        attach(context, jobId)
                    } else {
                        throw error
                    }
                }
            } else {
                attach(context, jobId)
            }
        }

        /**
         * BOOT_COMPLETED has a short receiver deadline. The durable pointer is enough to launch a
         * typed foreground recovery; all potentially large ledger reconciliation happens only
         * after the service has promoted itself. Paused/attention states are presented, not run.
         */
        fun recoverFromBoot(context: Context) {
            val jobId = when (
                val pointer = V2ActiveIndexingJobPointer(context.filesDir).inspect()
            ) {
                V2ActiveIndexingJobPointerInspection.Missing -> return
                is V2ActiveIndexingJobPointerInspection.Readable -> pointer.jobId
                is V2ActiveIndexingJobPointerInspection.Unreadable -> {
                    Log.e(TAG, "Refusing boot recovery: active pointer is ${pointer.reason}")
                    _state.value = IndexingState.Error(null, ACTIVE_POINTER_UNREADABLE_MESSAGE)
                    return
                }
            }
            try {
                startCommand(context, jobId, V2IndexingServiceIntents.recover(context, jobId))
            } catch (error: RuntimeException) {
                if (V2IndexingRecoveryStartPolicy.mayDefer(error)) {
                    Log.w(TAG, "Boot recovery deferred until a visible user action", error)
                } else {
                    throw error
                }
            }
        }

        fun reconcileAndAttach(context: Context) {
            val repository = V2IndexingJobRepository.get(context)
            repository.reconcileStartup()
            attach(context)
        }

        private fun startCommand(context: Context, jobId: String, intent: Intent) {
            RecommendationWorkAdmission.reserve(
                RecommendationWorkAdmission.indexingLaunchOwner(jobId),
            )
            try {
                ContextCompat.startForegroundService(context, intent)
            } catch (error: Throwable) {
                releaseLaunchReservation(context, jobId)
                throw error
            }
        }

        private fun releaseLaunchReservation(context: Context, jobId: String) {
            if (RecommendationWorkAdmission.release(
                    RecommendationWorkAdmission.indexingLaunchOwner(jobId),
                )
            ) {
                RadioService.kickDeferredRecovery(context.applicationContext)
            }
        }

        private fun createPreflightStore(context: Context) =
            AtomicV2IndexingPreflightIntentStore(
                java.io.File(context.filesDir, "indexing_v2/preflight-intents"),
            )

        private fun resolveActiveOrOrphanPreflight(
            context: Context,
            pointer: V2ActiveIndexingJobPointer,
            repository: V2IndexingJobRepository,
        ): String? {
            when (val inspection = pointer.inspect()) {
                V2ActiveIndexingJobPointerInspection.Missing -> Unit
                is V2ActiveIndexingJobPointerInspection.Readable -> return inspection.jobId
                is V2ActiveIndexingJobPointerInspection.Unreadable -> {
                    Log.e(TAG, "Refusing orphan recovery: active pointer is ${inspection.reason}")
                    _state.value = IndexingState.Error(null, ACTIVE_POINTER_UNREADABLE_MESSAGE)
                    return null
                }
            }
            val candidates = runCatching {
                createPreflightStore(context).list().filter { intent ->
                    !V2IndexingPreflightControlPolicy.isTerminalRequest(intent.state)
                }
            }.getOrElse { error ->
                Log.e(TAG, "Unable to inspect durable indexing preflight", error)
                _state.value = IndexingState.Error(
                    null,
                    "Saved indexing work could not be inspected. Reopen On-device indexing to recover.",
                )
                return null
            }
            if (candidates.isEmpty()) return null
            if (candidates.size != 1) {
                _state.value = IndexingState.Error(
                    null,
                    "More than one saved indexing job needs attention. " +
                        "Open On-device indexing to review them.",
                )
                return null
            }
            val jobId = candidates.single().jobId
            return runCatching {
                pointer.claim(jobId) { existingJobId ->
                    runCatching { repository.require(existingJobId).state }.getOrNull()
                }
                jobId
            }.getOrElse { error ->
                Log.e(TAG, "Unable to claim saved indexing preflight", error)
                _state.value = IndexingState.Error(
                    jobId,
                    "Saved indexing work could not be resumed. Reopen On-device indexing and try again.",
                )
                null
            }
        }

    }
}

internal enum class V2IndexingServiceCommandType {
    START,
    RECOVER,
    PAUSE,
    RESUME,
    CANCEL,
    RETRY,
    SKIP,
    PROFILE,
    TIMEOUT,
}

internal data class V2IndexingServiceCommand(
    val type: V2IndexingServiceCommandType,
    val jobId: String,
    val workId: String? = null,
    val retryTrigger: RetryTrigger? = null,
    val profile: V2IndexingExecutionProfile? = null,
)

/** Explicit, immutable, package-private command contract used by UI and notification actions. */
internal object V2IndexingServiceIntents {
    const val EXTRA_JOB_ID = "com.powerampstartradio.indexing.v2.extra.JOB_ID"
    private const val EXTRA_WORK_ID = "com.powerampstartradio.indexing.v2.extra.WORK_ID"
    private const val EXTRA_RETRY_TRIGGER = "com.powerampstartradio.indexing.v2.extra.RETRY_TRIGGER"
    private const val EXTRA_PROFILE = "com.powerampstartradio.indexing.v2.extra.PROFILE"
    private const val PREFIX = "com.powerampstartradio.indexing.v2.action."

    fun start(context: Context, jobId: String) = command(context, "START", jobId)
    fun recover(context: Context, jobId: String) = command(context, "RECOVER", jobId)
    fun pause(context: Context, jobId: String) = command(context, "PAUSE", jobId)
    fun resume(context: Context, jobId: String) = command(context, "RESUME", jobId)
    fun cancel(context: Context, jobId: String) = command(context, "CANCEL", jobId)

    fun retry(context: Context, jobId: String, trigger: RetryTrigger) =
        command(context, "RETRY", jobId).putExtra(EXTRA_RETRY_TRIGGER, trigger.name)

    fun retryTrack(
        context: Context,
        jobId: String,
        workId: String,
        trigger: RetryTrigger,
    ): Intent {
        require(SAFE_ID.matches(workId)) { "unsafe work ID" }
        return retry(context, jobId, trigger).putExtra(EXTRA_WORK_ID, workId)
    }

    fun skip(context: Context, jobId: String, workId: String) =
        command(context, "SKIP", jobId).putExtra(EXTRA_WORK_ID, workId)

    fun profile(
        context: Context,
        jobId: String,
        profile: V2IndexingExecutionProfile,
    ) = command(context, "PROFILE", jobId).putExtra(EXTRA_PROFILE, profile.name)

    fun parse(context: Context, intent: Intent): V2IndexingServiceCommand {
        require(intent.component?.packageName == context.packageName &&
            intent.component?.className == IndexingService::class.java.name
        ) { "indexing command must target this private service explicitly" }
        val actionName = intent.action?.takeIf { it.startsWith(PREFIX) }
            ?.removePrefix(PREFIX)
            ?: throw IllegalArgumentException("unknown indexing action")
        val type = runCatching { V2IndexingServiceCommandType.valueOf(actionName) }
            .getOrElse { throw IllegalArgumentException("unknown indexing action") }
        require(type != V2IndexingServiceCommandType.TIMEOUT) { "timeout is process-internal" }
        val jobId = intent.getStringExtra(EXTRA_JOB_ID)
            ?.takeIf(SAFE_ID::matches)
            ?: throw IllegalArgumentException("missing or unsafe job ID")
        val allowedExtras = mutableSetOf(EXTRA_JOB_ID)
        val workId = if (type == V2IndexingServiceCommandType.SKIP) {
            allowedExtras += EXTRA_WORK_ID
            intent.getStringExtra(EXTRA_WORK_ID)?.takeIf(SAFE_ID::matches)
                ?: throw IllegalArgumentException("missing or unsafe work ID")
        } else if (type == V2IndexingServiceCommandType.RETRY &&
            intent.hasExtra(EXTRA_WORK_ID)
        ) {
            allowedExtras += EXTRA_WORK_ID
            intent.getStringExtra(EXTRA_WORK_ID)?.takeIf(SAFE_ID::matches)
                ?: throw IllegalArgumentException("unsafe retry work ID")
        } else null
        val trigger = if (type == V2IndexingServiceCommandType.RETRY) {
            allowedExtras += EXTRA_RETRY_TRIGGER
            enumValue<RetryTrigger>(intent.getStringExtra(EXTRA_RETRY_TRIGGER), "retry trigger")
        } else null
        val profile = if (type == V2IndexingServiceCommandType.PROFILE) {
            allowedExtras += EXTRA_PROFILE
            enumValue<V2IndexingExecutionProfile>(intent.getStringExtra(EXTRA_PROFILE), "profile")
        } else null
        require(intent.extras?.keySet().orEmpty().all { it in allowedExtras }) {
            "indexing command contains an unsupported payload"
        }
        return V2IndexingServiceCommand(type, jobId, workId, trigger, profile)
    }

    private fun command(context: Context, action: String, jobId: String): Intent {
        require(SAFE_ID.matches(jobId)) { "unsafe job ID" }
        return Intent(context, IndexingService::class.java).apply {
            this.action = PREFIX + action
            putExtra(EXTRA_JOB_ID, jobId)
        }
    }

    private inline fun <reified T : Enum<T>> enumValue(value: String?, label: String): T =
        value?.let { runCatching { enumValueOf<T>(it) }.getOrNull() }
            ?: throw IllegalArgumentException("invalid $label")

    private val SAFE_ID = Regex("^[A-Za-z0-9._-]{1,128}$")
}

internal object V2IndexingServiceStateMapper {
    fun map(
        ledger: IndexingJobLedger,
        event: V2IndexingExecutorEvent?,
        eta: V2StageAwareEtaEstimate,
        overallWork: V2IndexingOverallWorkSnapshot,
    ): IndexingService.IndexingState.JobSnapshot = IndexingService.IndexingState.JobSnapshot(
        jobId = ledger.jobSpec.jobId,
        jobState = ledger.state,
        profile = ledger.executionProfile,
        progress = V2IndexingProgress.from(ledger),
        overallWork = overallWork,
        event = event,
        eta = eta,
        etaCoverage = overallWork.etaCoverage,
        stateReason = ledger.stateReason,
    )

    fun mapPreflight(
        intent: V2IndexingPreflightIntent,
    ): IndexingService.IndexingState.PreflightSnapshot =
        IndexingService.IndexingState.PreflightSnapshot(
            jobId = intent.jobId,
            state = intent.state,
            profile = intent.executionProfile,
            selectedTrackCount = intent.selected.size,
            progress = intent.progress,
            failureCode = intent.failureCode,
        )

    fun stageLabel(stage: V2MeasuredWorkStage): String = when (stage) {
        V2MeasuredWorkStage.PCM_24K_SAMPLES -> "Decode and resample audio"
        V2MeasuredWorkStage.MERT_WINDOWS -> "Analyze audio features"
        V2MeasuredWorkStage.CLAMP_SEGMENTS -> "Create music embedding"
        V2MeasuredWorkStage.DATABASE_COMMITS -> "Save track embedding"
        V2MeasuredWorkStage.DERIVED_EMBEDDING_ROWS -> "Update search index"
        V2MeasuredWorkStage.GRAPH_SIMILARITY_DOT_PRODUCTS -> "Calculate similarity graph"
        V2MeasuredWorkStage.GRAPH_BINARY_BYTES -> "Save similarity graph"
        V2MeasuredWorkStage.GRAPH_NODES -> "Update similarity graph"
        V2MeasuredWorkStage.ACTIVATION_TRACKS -> "Updating music index"
        V2MeasuredWorkStage.PCM_NORMALIZATION_SAMPLES -> "Calculate whole-track normalization"
        V2MeasuredWorkStage.PCM_CACHE_BYTES -> "Verify saved decoded audio"
        V2MeasuredWorkStage.POWERAMP_LIBRARY_ROWS -> "Read Poweramp library rows"
        V2MeasuredWorkStage.INDEXING_MODEL_FILES -> "Load indexing model files"
        V2MeasuredWorkStage.SOURCE_AUDIO_HASH -> "Hash source audio"
        V2MeasuredWorkStage.PRIVATE_INDEX_COPY -> "Copy current music index"
        V2MeasuredWorkStage.SIMILARITY_GRAPH_SETUP -> "Prepare similarity graph"
        V2MeasuredWorkStage.MUSIC_INDEX_PUBLICATION -> "Publish music index"
        V2MeasuredWorkStage.SAVED_ARTIFACT_INSPECTION -> "Read saved indexing artifacts"
        V2MeasuredWorkStage.RECOMMENDATION_RESOURCE_HANDOFF -> "Release radio-model memory"
        V2MeasuredWorkStage.STAGING_INDEX_INSPECTION -> "Inspect staged music index"
    }
}

class V2IndexingBootReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action == Intent.ACTION_BOOT_COMPLETED) {
            IndexingService.recoverFromBoot(context.applicationContext)
        }
    }
}
