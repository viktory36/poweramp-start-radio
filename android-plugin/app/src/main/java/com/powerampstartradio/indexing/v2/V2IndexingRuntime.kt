package com.powerampstartradio.indexing.v2

import android.content.Context
import android.util.AtomicFile
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.io.File
import java.io.FileNotFoundException
import java.io.OutputStreamWriter
import java.nio.ByteBuffer
import java.nio.charset.CodingErrorAction
import java.nio.charset.StandardCharsets
import java.util.UUID

class V2IndexingJobRepository private constructor(private val filesDir: File) {
    private val ledgerDirectory = File(filesDir, "indexing_v2/jobs")
    private val controller = V2IndexingLifecycleController(
        V2AtomicIndexingLedgerAccess(ledgerDirectory),
    )
    private val _jobs = MutableStateFlow<List<IndexingJobLedger>>(emptyList())
    val jobs: StateFlow<List<IndexingJobLedger>> = _jobs.asStateFlow()
    private val processInstanceId = UUID.randomUUID().toString()
    private val executorLeases = V2ExecutorLeaseCoordinator(
        V2AtomicExecutorLeasePersistence(filesDir),
    )
    private var startupReconciled = false
    private var startupResult: V2StartupReconciliationResult? = null

    init {
        refresh()
    }

    @Synchronized
    fun refresh(): List<IndexingJobLedger> = controller.list().also { _jobs.value = it }

    @Synchronized
    fun reconcileStartup(): V2StartupReconciliationResult {
        startupResult?.let { return it }
        val result = controller.reconcileNonterminalJobs()
        // Ledger intent is durable before an executor epoch can be retired and reissued.
        executorLeases.retirePreviousProcessLease(processInstanceId)
        startupReconciled = true
        startupResult = result
        refresh()
        return result
    }

    @Synchronized
    fun require(jobId: String): IndexingJobLedger = controller.require(jobId)

    @Synchronized
    fun claimExecutor(jobId: String): V2ExecutorLeaseToken {
        ensureStartupReconciled()
        controller.require(jobId)
        return executorLeases.claim(jobId, processInstanceId)
    }

    @Synchronized
    fun startAuthorized(token: V2ExecutorLeaseToken): IndexingJobLedger =
        withExecutor(token) { controller.start(token.jobId) }.publish()

    @Synchronized
    fun requestPause(jobId: String): IndexingJobLedger =
        controller.requestPause(jobId).publish()

    @Synchronized
    fun requestMediaProcessingTimeoutPause(
        jobId: String,
        reason: String,
    ): IndexingJobLedger = controller.requestMediaProcessingTimeoutPause(jobId, reason).publish()

    @Synchronized
    fun finishPauseAfterExecutorStops(
        token: V2ExecutorLeaseToken,
        reason: String = "paused at verified checkpoint",
    ): IndexingJobLedger = withExecutor(token) {
        controller.finishPauseAfterExecutorStops(token.jobId, reason)
    }.publish()

    @Synchronized
    fun interruptActivationForMediaProcessingTimeout(
        token: V2ExecutorLeaseToken,
        reason: String,
    ): IndexingJobLedger = withExecutor(token) {
        controller.interruptActivationForMediaProcessingTimeout(token.jobId, reason)
    }.publish()

    @Synchronized
    fun checkpointForMediaProcessingTimeout(
        token: V2ExecutorLeaseToken,
        reason: String,
    ): IndexingJobLedger = withExecutor(token) {
        controller.checkpointForMediaProcessingTimeout(token.jobId, reason)
    }.publish()

    /** Platform service timeout is authoritative even before an executor lease is claimed. */
    @Synchronized
    fun checkpointForMediaProcessingTimeout(
        jobId: String,
        reason: String,
    ): IndexingJobLedger = controller.checkpointForMediaProcessingTimeout(jobId, reason).publish()

    @Synchronized
    fun resumeAuthorized(token: V2ExecutorLeaseToken): IndexingJobLedger =
        withExecutor(token) { controller.resume(token.jobId) }.publish()

    @Synchronized
    fun requestCancel(jobId: String): IndexingJobLedger =
        controller.requestCancel(jobId).publish()

    @Synchronized
    fun finishCancellationAfterCleanup(token: V2ExecutorLeaseToken): IndexingJobLedger =
        withExecutor(token) { controller.finishCancellationAfterCleanup(token.jobId) }.publish()

    @Synchronized
    fun retryFailed(jobId: String, trigger: RetryTrigger): V2RetryResult =
        controller.retryFailed(jobId, trigger).also { refresh() }

    @Synchronized
    fun retryTrack(
        jobId: String,
        workId: String,
        trigger: RetryTrigger,
    ): IndexingJobLedger = controller.retryTrack(jobId, workId, trigger).publish()

    @Synchronized
    fun changeProfile(
        jobId: String,
        profile: V2IndexingExecutionProfile,
    ): IndexingJobLedger = controller.changeProfile(jobId, profile).publish()

    @Synchronized
    fun skipTrack(jobId: String, workId: String): IndexingJobLedger =
        controller.skipTrack(jobId, workId).publish()

    @Synchronized
    fun reconcileUnexpectedExecutorStop(
        token: V2ExecutorLeaseToken,
    ): RestartReconciliation = withExecutor(token) {
        controller.reconcileUnexpectedExecutorStop(token.jobId)
    }.also { refresh() }

    @Synchronized
    internal fun executorUpdate(
        token: V2ExecutorLeaseToken,
        transition: (IndexingJobLedger) -> IndexingJobLedger,
    ): IndexingJobLedger = withExecutor(token) {
        controller.update(token.jobId, transition)
    }.publish()

    @Synchronized
    internal fun waitForInput(token: V2ExecutorLeaseToken): IndexingJobLedger =
        withExecutor(token) { controller.waitForInput(token.jobId) }.publish()

    @Synchronized
    fun heartbeatExecutor(token: V2ExecutorLeaseToken): V2PersistedExecutorLease =
        executorLeases.heartbeat(token)

    @Synchronized
    fun releaseExecutor(token: V2ExecutorLeaseToken) {
        executorLeases.release(token)
    }

    fun artifactDirectory(jobId: String): File =
        File(filesDir, "indexing_v2/artifacts/$jobId")

    fun stagingDatabaseFile(jobId: String): File =
        File(filesDir, "indexing_v2/job-databases/$jobId.db")

    private fun ensureStartupReconciled() {
        if (!startupReconciled) reconcileStartup()
    }

    private inline fun <T> withExecutor(
        token: V2ExecutorLeaseToken,
        operation: () -> T,
    ): T {
        executorLeases.requireCurrent(token)
        return operation()
    }

    private fun IndexingJobLedger.publish(): IndexingJobLedger = also { updated ->
        val current = _jobs.value
        _jobs.value = if (current.any { it.jobSpec.jobId == updated.jobSpec.jobId }) {
            current.map { existing ->
                if (existing.jobSpec.jobId == updated.jobSpec.jobId) updated else existing
            }
        } else {
            (current + updated).sortedBy { it.jobSpec.createdAtEpochMs }
        }
    }

    companion object {
        @Volatile private var instance: V2IndexingJobRepository? = null

        fun get(context: Context): V2IndexingJobRepository = instance ?: synchronized(this) {
            instance ?: V2IndexingJobRepository(context.applicationContext.filesDir).also {
                instance = it
            }
        }

        /** Independent durable root for connected acceptance without touching the active app. */
        internal fun createIsolated(filesDir: File): V2IndexingJobRepository =
            V2IndexingJobRepository(filesDir.canonicalFile)
    }
}

/** Durable pointer used only to recover START_STICKY's null restart intent. */
class V2ActiveIndexingJobPointer(filesDir: File) {
    private val baseFile = File(filesDir, "indexing_v2/active-job-id")
    private val file = AtomicFile(baseFile)

    fun inspect(): V2ActiveIndexingJobPointerInspection = synchronized(POINTER_LOCK) {
        inspectLocked()
    }

    fun read(): String? = synchronized(POINTER_LOCK) {
        inspectLocked().jobIdOrThrow()
    }

    /**
     * Reserves the one durable user-visible indexing slot before the service is launched.
     * A paused or attention-blocked job still owns the slot; the user must resume or cancel it.
     */
    fun claim(
        jobId: String,
        stateOf: (String) -> IndexingJobState?,
    ): V2ActiveIndexingJobClaim = synchronized(POINTER_LOCK) {
        require(SAFE_JOB_ID.matches(jobId)) { "unsafe active job id" }
        val currentJobId = inspectLocked().jobIdOrThrow()
        val currentState = currentJobId
            ?.takeIf { it != jobId }
            ?.let(stateOf)
        if (!V2ActiveIndexingJobClaimPolicy.canClaim(jobId, currentJobId, currentState)) {
            throw V2ActiveIndexingJobConflictException(
                activeJobId = requireNotNull(currentJobId),
                activeState = currentState,
            )
        }
        val changed = currentJobId != jobId
        if (changed) writeLocked(jobId)
        V2ActiveIndexingJobClaim(previousJobId = currentJobId, changed = changed)
    }

    fun write(jobId: String) = synchronized(POINTER_LOCK) {
        inspectLocked().jobIdOrThrow()
        writeLocked(jobId)
    }

    fun clear(jobId: String? = null) = synchronized(POINTER_LOCK) {
        val currentJobId = inspectLocked().jobIdOrThrow()
        if (currentJobId != null && (jobId == null || currentJobId == jobId)) file.delete()
    }

    private fun inspectLocked(): V2ActiveIndexingJobPointerInspection {
        val bytes = try {
            file.openRead().use { it.readBytes() }
        } catch (_: FileNotFoundException) {
            return if (hasPointerArtifact()) {
                V2ActiveIndexingJobPointerInspection.Unreadable(
                    V2ActiveIndexingJobPointerUnreadableReason.IO_FAILURE,
                )
            } else {
                V2ActiveIndexingJobPointerInspection.Missing
            }
        } catch (_: Exception) {
            return V2ActiveIndexingJobPointerInspection.Unreadable(
                V2ActiveIndexingJobPointerUnreadableReason.IO_FAILURE,
            )
        }
        if (bytes.isEmpty()) {
            return V2ActiveIndexingJobPointerInspection.Unreadable(
                V2ActiveIndexingJobPointerUnreadableReason.EMPTY,
            )
        }
        val value = try {
            StandardCharsets.UTF_8.newDecoder()
                .onMalformedInput(CodingErrorAction.REPORT)
                .onUnmappableCharacter(CodingErrorAction.REPORT)
                .decode(ByteBuffer.wrap(bytes))
                .toString()
                .trim()
        } catch (_: Exception) {
            return V2ActiveIndexingJobPointerInspection.Unreadable(
                V2ActiveIndexingJobPointerUnreadableReason.MALFORMED_UTF8,
            )
        }
        return if (SAFE_JOB_ID.matches(value)) {
            V2ActiveIndexingJobPointerInspection.Readable(value)
        } else {
            V2ActiveIndexingJobPointerInspection.Unreadable(
                if (value.isEmpty()) V2ActiveIndexingJobPointerUnreadableReason.EMPTY
                else V2ActiveIndexingJobPointerUnreadableReason.INVALID_JOB_ID,
            )
        }
    }

    private fun hasPointerArtifact(): Boolean =
        baseFile.exists() ||
            File("${baseFile.path}.bak").exists() ||
            File("${baseFile.path}.new").exists()

    private fun V2ActiveIndexingJobPointerInspection.jobIdOrThrow(): String? = when (this) {
        V2ActiveIndexingJobPointerInspection.Missing -> null
        is V2ActiveIndexingJobPointerInspection.Readable -> jobId
        is V2ActiveIndexingJobPointerInspection.Unreadable -> {
            throw V2ActiveIndexingJobPointerUnreadableException(reason)
        }
    }

    private fun writeLocked(jobId: String) {
        require(SAFE_JOB_ID.matches(jobId)) { "unsafe active job id" }
        file.baseFile.parentFile?.let { parent ->
            require(parent.isDirectory || parent.mkdirs()) { "cannot create $parent" }
        }
        val stream = file.startWrite()
        try {
            val writer = OutputStreamWriter(stream, StandardCharsets.UTF_8)
            writer.write(jobId)
            writer.flush()
            file.finishWrite(stream)
        } catch (error: Throwable) {
            file.failWrite(stream)
            throw error
        }
    }

    private companion object {
        val POINTER_LOCK = Any()
        val SAFE_JOB_ID = Regex("^[A-Za-z0-9._-]{1,128}$")
    }
}

sealed interface V2ActiveIndexingJobPointerInspection {
    data object Missing : V2ActiveIndexingJobPointerInspection

    data class Readable(val jobId: String) : V2ActiveIndexingJobPointerInspection

    data class Unreadable(
        val reason: V2ActiveIndexingJobPointerUnreadableReason,
    ) : V2ActiveIndexingJobPointerInspection
}

enum class V2ActiveIndexingJobPointerUnreadableReason {
    EMPTY,
    MALFORMED_UTF8,
    INVALID_JOB_ID,
    IO_FAILURE,
}

class V2ActiveIndexingJobPointerUnreadableException(
    val reason: V2ActiveIndexingJobPointerUnreadableReason,
) : IllegalStateException(
    "Saved active indexing ownership cannot be read. Its evidence was preserved and no " +
        "indexing job was reassigned.",
)

data class V2ActiveIndexingJobClaim(
    val previousJobId: String?,
    val changed: Boolean,
)

class V2ActiveIndexingJobConflictException(
    val activeJobId: String,
    val activeState: IndexingJobState?,
) : IllegalStateException(
    buildString {
        if (activeState == null) {
            append("An existing indexing job could not be verified. Its recovery pointer was ")
            append("preserved instead of starting another job.")
        } else {
            append("Another indexing job is already active (")
            append(activeState.name.lowercase().replace('_', ' '))
            append("). Resume or cancel it before starting a new job.")
        }
    },
)

object V2ActiveIndexingJobClaimPolicy {
    fun canClaim(
        candidateJobId: String,
        currentJobId: String?,
        currentState: IndexingJobState?,
    ): Boolean = currentJobId == null ||
        currentJobId == candidateJobId ||
        currentState == IndexingJobState.COMPLETE ||
        currentState == IndexingJobState.CANCELLED
}
