package com.powerampstartradio.indexing.v2

import android.util.AtomicFile
import com.google.gson.Gson
import com.google.gson.GsonBuilder
import java.io.File
import java.io.IOException
import java.io.OutputStreamWriter
import java.nio.charset.StandardCharsets

data class V2ExecutorLeaseToken(
    val jobId: String,
    val epoch: Long,
    val ownerInstanceId: String,
)

data class V2PersistedExecutorLease(
    val jobId: String,
    val epoch: Long,
    val ownerInstanceId: String,
    val acquiredAtEpochMs: Long,
    val heartbeatAtEpochMs: Long,
) {
    fun token(): V2ExecutorLeaseToken = V2ExecutorLeaseToken(jobId, epoch, ownerInstanceId)
}

data class V2ExecutorLeaseState(
    val schemaVersion: Int,
    val lastIssuedEpoch: Long,
    val active: V2PersistedExecutorLease?,
)

class V2ExecutorLeaseConflictException(message: String) : IllegalStateException(message)

interface V2ExecutorLeasePersistence {
    fun read(): V2ExecutorLeaseState
    fun write(state: V2ExecutorLeaseState)
}

/**
 * Issues one process-scoped executor token at a time. Every executor callback must present the
 * exact epoch, so work posted by an old service instance cannot mutate a newly recovered job.
 */
class V2ExecutorLeaseCoordinator(
    private val persistence: V2ExecutorLeasePersistence,
    private val nowEpochMs: () -> Long = System::currentTimeMillis,
) {
    @Synchronized
    fun activeLease(): V2PersistedExecutorLease? = processSerialized {
        validated(persistence.read()).active
    }

    @Synchronized
    fun claim(jobId: String, ownerInstanceId: String): V2ExecutorLeaseToken = processSerialized {
        requireSafe(jobId, "job id")
        requireSafe(ownerInstanceId, "owner instance id")
        val state = validated(persistence.read())
        val current = state.active
        if (current != null) {
            throw V2ExecutorLeaseConflictException(
                "executor epoch ${current.epoch} already belongs to job ${current.jobId}",
            )
        }
        val epoch = Math.addExact(state.lastIssuedEpoch, 1L)
        val now = nowEpochMs().coerceAtLeast(0L)
        val lease = V2PersistedExecutorLease(
            jobId = jobId,
            epoch = epoch,
            ownerInstanceId = ownerInstanceId,
            acquiredAtEpochMs = now,
            heartbeatAtEpochMs = now,
        )
        persistence.write(
            V2ExecutorLeaseState(
                schemaVersion = SCHEMA_VERSION,
                lastIssuedEpoch = epoch,
                active = lease,
            ),
        )
        return lease.token()
    }

    @Synchronized
    fun requireCurrent(token: V2ExecutorLeaseToken): V2PersistedExecutorLease = processSerialized {
        val active = validated(persistence.read()).active
        if (active?.token() != token) {
            throw V2ExecutorLeaseConflictException(
                "stale executor callback for job ${token.jobId} epoch ${token.epoch}",
            )
        }
        return active
    }

    @Synchronized
    fun heartbeat(token: V2ExecutorLeaseToken): V2PersistedExecutorLease = processSerialized {
        val state = validated(persistence.read())
        val current = state.active
        if (current?.token() != token) {
            throw V2ExecutorLeaseConflictException(
                "cannot heartbeat stale executor epoch ${token.epoch}",
            )
        }
        val updated = current.copy(
            heartbeatAtEpochMs = maxOf(current.heartbeatAtEpochMs, nowEpochMs()),
        )
        persistence.write(state.copy(active = updated))
        return updated
    }

    @Synchronized
    fun release(token: V2ExecutorLeaseToken) = processSerialized {
        val state = validated(persistence.read())
        if (state.active?.token() != token) {
            throw V2ExecutorLeaseConflictException(
                "cannot release stale executor epoch ${token.epoch}",
            )
        }
        persistence.write(state.copy(active = null))
    }

    /** Called only after durable ledger reconciliation for the previous process has completed. */
    @Synchronized
    fun retirePreviousProcessLease(
        currentOwnerInstanceId: String,
    ): V2PersistedExecutorLease? = processSerialized {
        requireSafe(currentOwnerInstanceId, "owner instance id")
        val state = validated(persistence.read())
        val previous = state.active?.takeIf { it.ownerInstanceId != currentOwnerInstanceId }
            ?: return null
        persistence.write(state.copy(active = null))
        return previous
    }

    private fun validated(state: V2ExecutorLeaseState): V2ExecutorLeaseState {
        if (state.schemaVersion != SCHEMA_VERSION || state.lastIssuedEpoch < 0L) {
            throw V2ExecutorLeaseConflictException("invalid persisted executor lease state")
        }
        state.active?.let { lease ->
            requireSafe(lease.jobId, "persisted job id")
            requireSafe(lease.ownerInstanceId, "persisted owner instance id")
            if (lease.epoch <= 0L || lease.epoch > state.lastIssuedEpoch ||
                lease.acquiredAtEpochMs < 0L || lease.heartbeatAtEpochMs < lease.acquiredAtEpochMs
            ) {
                throw V2ExecutorLeaseConflictException("invalid persisted executor lease")
            }
        }
        return state
    }

    private fun requireSafe(value: String, label: String) {
        require(SAFE_ID.matches(value)) { "unsafe $label" }
    }

    private inline fun <T> processSerialized(operation: () -> T): T =
        synchronized(PROCESS_COORDINATION_LOCK, operation)

    companion object {
        const val SCHEMA_VERSION = 1
        private val SAFE_ID = Regex("^[A-Za-z0-9._-]{1,128}$")
        private val PROCESS_COORDINATION_LOCK = Any()
    }
}

class V2AtomicExecutorLeasePersistence(
    filesDir: File,
    private val gson: Gson = GsonBuilder().disableHtmlEscaping().create(),
) : V2ExecutorLeasePersistence {
    private val file = AtomicFile(File(filesDir, "indexing_v2/executor-lease.json"))

    @Synchronized
    override fun read(): V2ExecutorLeaseState {
        if (!file.baseFile.exists() && !File(file.baseFile.path + ".bak").exists()) {
            return emptyState()
        }
        return try {
            file.openRead().bufferedReader(StandardCharsets.UTF_8).use { reader ->
                gson.fromJson(reader, V2ExecutorLeaseState::class.java)
            } ?: throw IOException("empty executor lease file")
        } catch (error: Exception) {
            throw IOException("unable to read V2 executor lease", error)
        }
    }

    @Synchronized
    override fun write(state: V2ExecutorLeaseState) {
        file.baseFile.parentFile?.let { parent ->
            if ((!parent.exists() && !parent.mkdirs()) || !parent.isDirectory) {
                throw IOException("unable to create executor lease directory: $parent")
            }
        }
        val stream = file.startWrite()
        try {
            val writer = OutputStreamWriter(stream, StandardCharsets.UTF_8)
            gson.toJson(state, writer)
            writer.flush()
            file.finishWrite(stream)
        } catch (error: Throwable) {
            file.failWrite(stream)
            throw IOException("unable to persist V2 executor lease", error)
        }
    }

    private fun emptyState() = V2ExecutorLeaseState(
        schemaVersion = V2ExecutorLeaseCoordinator.SCHEMA_VERSION,
        lastIssuedEpoch = 0L,
        active = null,
    )
}
