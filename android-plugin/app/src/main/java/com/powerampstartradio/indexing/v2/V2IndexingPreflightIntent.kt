package com.powerampstartradio.indexing.v2

import android.util.AtomicFile
import com.google.gson.Gson
import com.google.gson.GsonBuilder
import java.io.File
import java.io.IOException
import java.io.OutputStreamWriter
import java.io.Reader
import java.io.Writer
import java.nio.charset.StandardCharsets
import java.security.MessageDigest

object V2IndexingPreflightIntentSchema {
    const val VERSION = 2
    const val FORMAT = "poweramp-start-radio-v2-indexing-preflight-intent"
}

enum class V2IndexingPreflightIntentState {
    REQUESTED,
    PLANNING,
    INTERRUPTED,
    FAILED,
    CANCEL_REQUESTED,
    CANCELLED,
    RESOLVED_WITH_EXECUTABLE_ROWS,
    RESOLVED_WITHOUT_EXECUTABLE_ROWS,
    MATERIALIZED,
}

enum class V2IndexingPreflightPhase {
    QUEUED,
    ACTIVE_GENERATION,
    POWERAMP_SNAPSHOT,
    AUDIO_SPANS,
    SOURCE_BINDINGS,
    SOURCE_FINGERPRINTS,
    MODEL_FINGERPRINTS,
    RUNTIME_FINGERPRINT,
    SOURCE_REVALIDATION,
    PERSISTING_LEDGER,
    COMPLETE,
}

enum class V2IndexingPreflightProgressUnit {
    BYTES,
}

/** Exact occurrence facts captured at the user's selection boundary. */
data class V2IndexingPreflightSelection(
    val powerampFileId: Long,
    val providerPhysicalPath: String,
    val durationMs: Long,
    val offsetMs: Long,
    val cueSourceImageFolderId: Long?,
)

data class V2IndexingPreflightProgress(
    val phase: V2IndexingPreflightPhase,
    val message: String,
    val completedUnits: Long? = null,
    val totalUnits: Long? = null,
    val unit: V2IndexingPreflightProgressUnit? = null,
)

object V2IndexingPreflightProgressValidator {
    fun requireValid(progress: V2IndexingPreflightProgress) {
        if (progress.message.isBlank() || progress.message.length > 512) {
            invalid("progress message")
        }
        if ((progress.completedUnits == null) != (progress.totalUnits == null)) {
            invalid("partial progress units")
        }
        if (progress.totalUnits != null &&
            (progress.totalUnits <= 0L || progress.completedUnits !in 0L..progress.totalUnits)
        ) {
            invalid("invalid progress units")
        }
    }

    private fun invalid(detail: String): Nothing =
        throw InvalidV2IndexingPreflightIntentException(
            "Invalid indexing preflight intent: $detail",
        )
}

object V2IndexingPreflightProgressOverlaySchema {
    const val VERSION = 1
    const val FORMAT = "poweramp-start-radio-v2-indexing-preflight-progress"
}

/**
 * Small durable progress projection. The main intent remains the state-machine authority: this
 * overlay is usable only while its exact base revision and immutable request fingerprint match.
 */
data class V2IndexingPreflightProgressOverlay(
    val schemaVersion: Int,
    val jobId: String,
    val baseIntentRevision: Long,
    val baseIntentUpdatedAtEpochMs: Long,
    val requestFingerprint: String,
    val persistedAtEpochMs: Long,
    val progress: V2IndexingPreflightProgress,
)

object V2IndexingPreflightRequestFingerprint {
    fun compute(intent: V2IndexingPreflightIntent): String {
        val digest = MessageDigest.getInstance("SHA-256")
        digest.putInt(intent.schemaVersion)
        digest.putString(intent.jobId)
        digest.putLong(intent.createdAtEpochMs)
        digest.putInt(intent.selected.size)
        intent.selected.forEach { selected ->
            digest.putLong(selected.powerampFileId)
            digest.putString(selected.providerPhysicalPath)
            digest.putLong(selected.durationMs)
            digest.putLong(selected.offsetMs)
            digest.putNullableLong(selected.cueSourceImageFolderId)
        }
        digest.putBoolean(intent.rebuildDerivedIndexes)
        digest.putString(intent.executionProfile.name)
        return digest.digest().joinToString("") { byte -> "%02x".format(byte.toInt() and 0xff) }
    }

    private fun MessageDigest.putBoolean(value: Boolean) =
        update((if (value) 1 else 0).toByte())

    private fun MessageDigest.putNullableLong(value: Long?) {
        putBoolean(value != null)
        if (value != null) putLong(value)
    }

    private fun MessageDigest.putString(value: String) {
        val bytes = value.toByteArray(StandardCharsets.UTF_8)
        putInt(bytes.size)
        update(bytes)
    }

    private fun MessageDigest.putInt(value: Int) {
        for (shift in 24 downTo 0 step 8) update((value ushr shift).toByte())
    }

    private fun MessageDigest.putLong(value: Long) {
        for (shift in 56 downTo 0 step 8) update((value ushr shift).toByte())
    }
}

object V2IndexingPreflightProgressOverlayPolicy {
    fun create(
        current: V2IndexingPreflightIntent,
        progress: V2IndexingPreflightProgress,
        nowEpochMs: Long,
    ): V2IndexingPreflightProgressOverlay {
        require(current.state == V2IndexingPreflightIntentState.PLANNING) {
            "${current.state} preflight cannot persist a progress overlay"
        }
        require(nowEpochMs >= current.updatedAtEpochMs) { "preflight clock moved backwards" }
        V2IndexingPreflightProgressValidator.requireValid(progress)
        return V2IndexingPreflightProgressOverlay(
            schemaVersion = V2IndexingPreflightProgressOverlaySchema.VERSION,
            jobId = current.jobId,
            baseIntentRevision = current.revision,
            baseIntentUpdatedAtEpochMs = current.updatedAtEpochMs,
            requestFingerprint = V2IndexingPreflightRequestFingerprint.compute(current),
            persistedAtEpochMs = nowEpochMs,
            progress = progress,
        )
    }

    fun applyIfCurrent(
        current: V2IndexingPreflightIntent,
        overlay: V2IndexingPreflightProgressOverlay,
    ): V2IndexingPreflightIntent {
        if (current.state != V2IndexingPreflightIntentState.PLANNING ||
            overlay.schemaVersion != V2IndexingPreflightProgressOverlaySchema.VERSION ||
            overlay.jobId != current.jobId ||
            overlay.baseIntentRevision != current.revision ||
            overlay.baseIntentUpdatedAtEpochMs != current.updatedAtEpochMs ||
            overlay.persistedAtEpochMs < current.updatedAtEpochMs ||
            !SAFE_FINGERPRINT.matches(overlay.requestFingerprint)
        ) return current
        return runCatching {
            V2IndexingPreflightProgressValidator.requireValid(overlay.progress)
            if (overlay.requestFingerprint != V2IndexingPreflightRequestFingerprint.compute(current)) {
                current
            } else {
                current.copy(progress = overlay.progress)
            }
        }.getOrDefault(current)
    }

    private val SAFE_FINGERPRINT = Regex("^[0-9a-f]{64}$")
}

internal object V2IndexingPreflightProgressOverlayCodec {
    fun write(gson: Gson, overlay: V2IndexingPreflightProgressOverlay, writer: Writer) {
        gson.toJson(
            Envelope(
                format = V2IndexingPreflightProgressOverlaySchema.FORMAT,
                schemaVersion = V2IndexingPreflightProgressOverlaySchema.VERSION,
                overlay = overlay,
            ),
            writer,
        )
    }

    fun readOrNull(gson: Gson, reader: Reader): V2IndexingPreflightProgressOverlay? =
        runCatching {
            val envelope = gson.fromJson(reader, Envelope::class.java) ?: return null
            envelope.overlay.takeIf {
                envelope.format == V2IndexingPreflightProgressOverlaySchema.FORMAT &&
                    envelope.schemaVersion == V2IndexingPreflightProgressOverlaySchema.VERSION &&
                    it.schemaVersion == envelope.schemaVersion
            }
        }.getOrNull()

    private data class Envelope(
        val format: String,
        val schemaVersion: Int,
        val overlay: V2IndexingPreflightProgressOverlay,
    )
}

/** Durable result for one selected occurrence that cannot enter this immutable execution plan. */
data class V2IndexingPreflightRejectedRow(
    val selected: V2IndexingPreflightSelection,
    val code: V2IndexingPreflightFailureCode,
    val disposition: FailureDisposition,
    val retryTrigger: RetryTrigger,
    val diagnostic: String,
)

/**
 * Lightweight durable request written before querying Poweramp or validating source/model files.
 * The selected occurrence facts and scheduling choice are immutable after creation.
 */
data class V2IndexingPreflightIntent(
    val schemaVersion: Int,
    val jobId: String,
    val createdAtEpochMs: Long,
    val updatedAtEpochMs: Long,
    val revision: Long,
    val state: V2IndexingPreflightIntentState,
    val selected: List<V2IndexingPreflightSelection>,
    val baseGenerationId: String?,
    val rebuildDerivedIndexes: Boolean,
    val executionProfile: V2IndexingExecutionProfile,
    val progress: V2IndexingPreflightProgress,
    val failureCode: V2IndexingPreflightFailureCode?,
    val failureMessage: String?,
    val planned: List<V2IndexingPreflightSelection>,
    val rejected: List<V2IndexingPreflightRejectedRow>,
    val resolvedSpecId: String?,
    val materializedSpecId: String?,
)

class InvalidV2IndexingPreflightIntentException(message: String) :
    IllegalStateException(message)

class V2IndexingPreflightIntentConflictException(message: String) :
    IllegalStateException(message)

data class V2IndexingPreflightIntentReadIssue(
    val jobId: String,
    val fileName: String,
    val message: String,
)

data class V2IndexingPreflightIntentInspection(
    val intents: List<V2IndexingPreflightIntent>,
    val issues: List<V2IndexingPreflightIntentReadIssue>,
) {
    val isComplete: Boolean get() = issues.isEmpty()

    fun requireComplete(): List<V2IndexingPreflightIntent> {
        if (issues.isNotEmpty()) throw V2IndexingPreflightIntentInspectionException(this)
        return intents
    }
}

class V2IndexingPreflightIntentInspectionException(
    val inspection: V2IndexingPreflightIntentInspection,
) : IOException(
    buildString {
        append("Unable to read ")
        append(inspection.issues.size)
        append(" indexing preflight intent")
        if (inspection.issues.size != 1) append('s')
        inspection.issues.firstOrNull()?.let { first ->
            append(": ")
            append(first.fileName)
            append(": ")
            append(first.message)
        }
    },
)

object V2IndexingPreflightIntentFactory {
    fun create(
        jobId: String,
        selected: List<V2IndexingPreflightSelection>,
        rebuildDerivedIndexes: Boolean,
        executionProfile: V2IndexingExecutionProfile,
        nowEpochMs: Long,
    ): V2IndexingPreflightIntent = V2IndexingPreflightIntent(
        schemaVersion = V2IndexingPreflightIntentSchema.VERSION,
        jobId = jobId,
        createdAtEpochMs = nowEpochMs,
        updatedAtEpochMs = nowEpochMs,
        revision = 0L,
        state = V2IndexingPreflightIntentState.REQUESTED,
        selected = selected.sortedBy(V2IndexingPreflightSelection::powerampFileId),
        baseGenerationId = null,
        rebuildDerivedIndexes = rebuildDerivedIndexes,
        executionProfile = executionProfile,
        progress = V2IndexingPreflightProgress(
            phase = V2IndexingPreflightPhase.QUEUED,
            message = "Waiting for durable foreground preflight",
        ),
        failureCode = null,
        failureMessage = null,
        planned = emptyList(),
        rejected = emptyList(),
        resolvedSpecId = null,
        materializedSpecId = null,
    ).also(V2IndexingPreflightIntentValidator::requireValid)
}

object V2IndexingPreflightIntentStateMachine {
    fun beginOrResumePlanning(
        current: V2IndexingPreflightIntent,
        baseGenerationId: String,
        progress: V2IndexingPreflightProgress,
        nowEpochMs: Long,
    ): V2IndexingPreflightIntent {
        require(current.state in setOf(
            V2IndexingPreflightIntentState.REQUESTED,
            V2IndexingPreflightIntentState.PLANNING,
            V2IndexingPreflightIntentState.INTERRUPTED,
            V2IndexingPreflightIntentState.FAILED,
            V2IndexingPreflightIntentState.RESOLVED_WITH_EXECUTABLE_ROWS,
        )) { "${current.state} preflight cannot begin planning" }
        require(baseGenerationId.isNotBlank()) { "base generation ID is blank" }
        current.baseGenerationId?.let { expected ->
            require(expected == baseGenerationId) {
                "preflight is bound to base generation $expected, not $baseGenerationId"
            }
        }
        return current.transition(nowEpochMs) {
            copy(
                state = V2IndexingPreflightIntentState.PLANNING,
                baseGenerationId = baseGenerationId,
                progress = progress,
                failureCode = null,
                failureMessage = null,
                planned = emptyList(),
                rejected = emptyList(),
                resolvedSpecId = null,
                materializedSpecId = null,
            )
        }
    }

    fun updateProgress(
        current: V2IndexingPreflightIntent,
        progress: V2IndexingPreflightProgress,
        nowEpochMs: Long,
    ): V2IndexingPreflightIntent {
        if (current.state == V2IndexingPreflightIntentState.CANCEL_REQUESTED) return current
        require(current.state == V2IndexingPreflightIntentState.PLANNING) {
            "${current.state} preflight cannot report planning progress"
        }
        return current.transition(nowEpochMs) { copy(progress = progress) }
    }

    fun requestCancel(
        current: V2IndexingPreflightIntent,
        nowEpochMs: Long,
    ): V2IndexingPreflightIntent {
        if (current.state in setOf(
                V2IndexingPreflightIntentState.CANCEL_REQUESTED,
                V2IndexingPreflightIntentState.CANCELLED,
            )
        ) return current
        require(current.state !in setOf(
            V2IndexingPreflightIntentState.MATERIALIZED,
            V2IndexingPreflightIntentState.RESOLVED_WITHOUT_EXECUTABLE_ROWS,
        )) {
            "terminal preflight cannot be cancelled through preflight control"
        }
        return current.transition(nowEpochMs) {
            copy(
                state = V2IndexingPreflightIntentState.CANCEL_REQUESTED,
                progress = V2IndexingPreflightProgress(
                    V2IndexingPreflightPhase.COMPLETE,
                    "Cancelling durable preflight",
                ),
                failureCode = null,
                failureMessage = null,
                planned = emptyList(),
                rejected = emptyList(),
                resolvedSpecId = null,
                materializedSpecId = null,
            )
        }
    }

    fun finishCancellation(
        current: V2IndexingPreflightIntent,
        nowEpochMs: Long,
    ): V2IndexingPreflightIntent {
        if (current.state == V2IndexingPreflightIntentState.CANCELLED) return current
        require(current.state == V2IndexingPreflightIntentState.CANCEL_REQUESTED) {
            "${current.state} preflight cannot finish cancellation"
        }
        return current.transition(nowEpochMs) {
            copy(
                state = V2IndexingPreflightIntentState.CANCELLED,
                progress = V2IndexingPreflightProgress(
                    V2IndexingPreflightPhase.COMPLETE,
                    "Indexing request cancelled before audio work began",
                ),
            )
        }
    }

    fun fail(
        current: V2IndexingPreflightIntent,
        code: V2IndexingPreflightFailureCode,
        message: String,
        nowEpochMs: Long,
    ): V2IndexingPreflightIntent {
        if (current.state == V2IndexingPreflightIntentState.CANCEL_REQUESTED) return current
        require(current.state in setOf(
            V2IndexingPreflightIntentState.REQUESTED,
            V2IndexingPreflightIntentState.PLANNING,
            V2IndexingPreflightIntentState.FAILED,
            V2IndexingPreflightIntentState.RESOLVED_WITH_EXECUTABLE_ROWS,
        )) { "${current.state} preflight cannot fail" }
        val diagnostic = message.trim().take(MAX_FAILURE_MESSAGE_CHARS)
            .ifBlank { "Preflight failed without a diagnostic" }
        require(
            V2IndexingPreflightFailurePolicy.semantics(code).scope ==
                V2IndexingPreflightFailureScope.GLOBAL_REQUEST,
        ) { "$code cannot fail a whole preflight request" }
        return current.transition(nowEpochMs) {
            copy(
                state = V2IndexingPreflightIntentState.FAILED,
                progress = V2IndexingPreflightProgress(
                    V2IndexingPreflightPhase.COMPLETE,
                    "Indexing preflight needs attention",
                ),
                failureCode = code,
                failureMessage = diagnostic,
                planned = emptyList(),
                rejected = emptyList(),
                resolvedSpecId = null,
                materializedSpecId = null,
            )
        }
    }

    fun interrupt(
        current: V2IndexingPreflightIntent,
        message: String,
        nowEpochMs: Long,
    ): V2IndexingPreflightIntent {
        if (current.state == V2IndexingPreflightIntentState.CANCEL_REQUESTED) return current
        // The exact result/spec link is the recovery authority once it has been published.
        // Android timeout or service teardown must not erase it while the separately atomic
        // ledger write is in flight.
        if (current.state == V2IndexingPreflightIntentState.RESOLVED_WITH_EXECUTABLE_ROWS) {
            return current
        }
        require(current.state in setOf(
            V2IndexingPreflightIntentState.REQUESTED,
            V2IndexingPreflightIntentState.PLANNING,
            V2IndexingPreflightIntentState.INTERRUPTED,
        )) { "${current.state} preflight cannot be interrupted" }
        val diagnostic = message.trim().take(MAX_FAILURE_MESSAGE_CHARS)
            .ifBlank { "Preflight was interrupted; resume when ready" }
        return current.transition(nowEpochMs) {
            copy(
                state = V2IndexingPreflightIntentState.INTERRUPTED,
                progress = V2IndexingPreflightProgress(
                    V2IndexingPreflightPhase.COMPLETE,
                    "Preflight paused by Android; resume when ready",
                ),
                failureCode = null,
                failureMessage = diagnostic,
                planned = emptyList(),
                rejected = emptyList(),
                resolvedSpecId = null,
                materializedSpecId = null,
            )
        }
    }

    fun resolveWithExecutableRows(
        current: V2IndexingPreflightIntent,
        planned: List<V2IndexingPreflightSelection>,
        rejected: List<V2IndexingPreflightRejectedRow>,
        specId: String,
        nowEpochMs: Long,
    ): V2IndexingPreflightIntent {
        require(current.state == V2IndexingPreflightIntentState.PLANNING) {
            "${current.state} preflight cannot record an executable result"
        }
        require(specId.isNotBlank()) { "resolved spec ID is blank" }
        return current.transition(nowEpochMs) {
            copy(
                state = V2IndexingPreflightIntentState.RESOLVED_WITH_EXECUTABLE_ROWS,
                progress = V2IndexingPreflightProgress(
                    V2IndexingPreflightPhase.PERSISTING_LEDGER,
                    "Persisting the resolved indexing plan",
                ),
                failureCode = null,
                failureMessage = null,
                planned = planned,
                rejected = rejected,
                resolvedSpecId = specId,
                materializedSpecId = null,
            )
        }
    }

    fun materializeResolved(
        current: V2IndexingPreflightIntent,
        nowEpochMs: Long,
    ): V2IndexingPreflightIntent {
        require(current.state == V2IndexingPreflightIntentState.RESOLVED_WITH_EXECUTABLE_ROWS) {
            "${current.state} preflight cannot materialize a ledger"
        }
        val specId = requireNotNull(current.resolvedSpecId)
        return current.transition(nowEpochMs) {
            copy(
                state = V2IndexingPreflightIntentState.MATERIALIZED,
                progress = V2IndexingPreflightProgress(
                    V2IndexingPreflightPhase.COMPLETE,
                    "Immutable indexing ledger persisted",
                ),
                failureCode = null,
                resolvedSpecId = null,
                materializedSpecId = specId,
            )
        }
    }

    fun resolveWithoutExecutableRows(
        current: V2IndexingPreflightIntent,
        rejected: List<V2IndexingPreflightRejectedRow>,
        nowEpochMs: Long,
    ): V2IndexingPreflightIntent {
        require(current.state == V2IndexingPreflightIntentState.PLANNING) {
            "${current.state} preflight cannot record an all-rejected result"
        }
        return current.transition(nowEpochMs) {
            copy(
                state = V2IndexingPreflightIntentState.RESOLVED_WITHOUT_EXECUTABLE_ROWS,
                progress = V2IndexingPreflightProgress(
                    V2IndexingPreflightPhase.COMPLETE,
                    "No selected tracks are currently executable",
                ),
                failureCode = null,
                failureMessage = null,
                planned = emptyList(),
                rejected = rejected,
                resolvedSpecId = null,
                materializedSpecId = null,
            )
        }
    }

    private fun V2IndexingPreflightIntent.transition(
        nowEpochMs: Long,
        transform: V2IndexingPreflightIntent.() -> V2IndexingPreflightIntent,
    ): V2IndexingPreflightIntent {
        require(nowEpochMs >= updatedAtEpochMs) { "preflight clock moved backwards" }
        val updated = transform().copy(
            revision = Math.addExact(revision, 1L),
            updatedAtEpochMs = nowEpochMs,
        )
        V2IndexingPreflightIntentValidator.requireValid(updated)
        return updated
    }

    private const val MAX_FAILURE_MESSAGE_CHARS = 2_048
}

object V2IndexingPreflightIntentValidator {
    private val SAFE_JOB_ID = Regex("^[A-Za-z0-9._-]{1,128}$")

    fun requireValid(intent: V2IndexingPreflightIntent) {
        if (intent.schemaVersion != V2IndexingPreflightIntentSchema.VERSION) invalid("schema")
        if (!SAFE_JOB_ID.matches(intent.jobId)) invalid("unsafe job ID")
        if (intent.createdAtEpochMs < 0L || intent.updatedAtEpochMs < intent.createdAtEpochMs) {
            invalid("timestamps")
        }
        if (intent.revision < 0L) invalid("negative revision")
        if (intent.selected.isEmpty()) invalid("empty selection")
        val seenIds = hashSetOf<Long>()
        intent.selected.forEach { row ->
            if (row.powerampFileId <= 0L || !seenIds.add(row.powerampFileId)) {
                invalid("invalid or duplicate Poweramp row")
            }
            if (!File(row.providerPhysicalPath).isAbsolute || row.providerPhysicalPath.isBlank()) {
                invalid("selected source path is not absolute")
            }
            if (row.durationMs < 0L || row.durationMs > Int.MAX_VALUE) {
                invalid("selected duration is invalid")
            }
            if (row.offsetMs < 0L) invalid("selected offset is invalid")
        }
        if (intent.baseGenerationId != null && intent.baseGenerationId.isBlank()) {
            invalid("blank base generation ID")
        }
        V2IndexingPreflightProgressValidator.requireValid(intent.progress)
        requireValidPartition(intent)
        when (intent.state) {
            V2IndexingPreflightIntentState.REQUESTED -> {
                if (intent.baseGenerationId != null || intent.failureCode != null ||
                    intent.failureMessage != null ||
                    hasResult(intent)
                ) invalid("requested evidence")
            }
            V2IndexingPreflightIntentState.PLANNING -> {
                if (intent.baseGenerationId == null || intent.failureCode != null ||
                    intent.failureMessage != null ||
                    hasResult(intent)
                ) invalid("planning evidence")
            }
            V2IndexingPreflightIntentState.INTERRUPTED -> {
                if (intent.failureCode != null || intent.failureMessage.isNullOrBlank() ||
                    hasResult(intent)
                ) {
                    invalid("interruption evidence")
                }
            }
            V2IndexingPreflightIntentState.FAILED -> {
                val code = intent.failureCode
                if (code == null || intent.failureMessage.isNullOrBlank() || hasResult(intent) ||
                    V2IndexingPreflightFailurePolicy.semantics(code).scope !=
                    V2IndexingPreflightFailureScope.GLOBAL_REQUEST
                ) {
                    invalid("failure evidence")
                }
            }
            V2IndexingPreflightIntentState.CANCEL_REQUESTED,
            V2IndexingPreflightIntentState.CANCELLED,
            -> if (intent.failureCode != null || intent.failureMessage != null || hasResult(intent)) {
                invalid("cancellation evidence")
            }
            V2IndexingPreflightIntentState.RESOLVED_WITH_EXECUTABLE_ROWS -> {
                if (intent.baseGenerationId == null || intent.failureCode != null ||
                    intent.failureMessage != null ||
                    intent.planned.isEmpty() || intent.resolvedSpecId.isNullOrBlank() ||
                    intent.materializedSpecId != null
                ) invalid("resolved executable evidence")
            }
            V2IndexingPreflightIntentState.RESOLVED_WITHOUT_EXECUTABLE_ROWS -> {
                if (intent.baseGenerationId == null || intent.failureCode != null ||
                    intent.failureMessage != null ||
                    intent.planned.isNotEmpty() || intent.rejected.size != intent.selected.size ||
                    intent.resolvedSpecId != null || intent.materializedSpecId != null
                ) invalid("all-rejected evidence")
            }
            V2IndexingPreflightIntentState.MATERIALIZED -> {
                if (intent.baseGenerationId == null || intent.failureCode != null ||
                    intent.failureMessage != null ||
                    intent.planned.isEmpty() || intent.resolvedSpecId != null ||
                    intent.materializedSpecId.isNullOrBlank()
                ) invalid("materialization evidence")
            }
        }
    }

    private fun requireValidPartition(intent: V2IndexingPreflightIntent) {
        val selectedById = intent.selected.associateBy(V2IndexingPreflightSelection::powerampFileId)
        val plannedIds = intent.planned.map(V2IndexingPreflightSelection::powerampFileId)
        val rejectedIds = intent.rejected.map { it.selected.powerampFileId }
        if (plannedIds.distinct().size != plannedIds.size ||
            rejectedIds.distinct().size != rejectedIds.size ||
            plannedIds.any { it in rejectedIds }
        ) invalid("duplicate result occurrence")
        if (intent.planned.any { selectedById[it.powerampFileId] != it } ||
            intent.rejected.any { selectedById[it.selected.powerampFileId] != it.selected }
        ) invalid("result does not bind immutable selection")
        if (intent.planned != intent.selected.filter { it.powerampFileId in plannedIds } ||
            intent.rejected.map(V2IndexingPreflightRejectedRow::selected) !=
            intent.selected.filter { it.powerampFileId in rejectedIds }
        ) invalid("result order differs from selection order")
        if (intent.rejected.any { rejected ->
                val semantics = V2IndexingPreflightFailurePolicy.semantics(rejected.code)
                semantics.scope != V2IndexingPreflightFailureScope.SELECTED_OCCURRENCE ||
                    semantics.disposition != rejected.disposition ||
                    semantics.retryTrigger != rejected.retryTrigger ||
                    rejected.diagnostic.isBlank() || rejected.diagnostic.length > 2_048
            }
        ) invalid("invalid rejected-row semantics")
        if (intent.planned.size + intent.rejected.size !in setOf(0, intent.selected.size)) {
            invalid("partial result partition")
        }
    }

    private fun hasResult(intent: V2IndexingPreflightIntent): Boolean =
        intent.planned.isNotEmpty() || intent.rejected.isNotEmpty() ||
            intent.resolvedSpecId != null || intent.materializedSpecId != null

    private fun invalid(detail: String): Nothing =
        throw InvalidV2IndexingPreflightIntentException("Invalid indexing preflight intent: $detail")
}

/** Atomic per-job intent store. Immutable selection fields cannot change during updates. */
class AtomicV2IndexingPreflightIntentStore(
    private val directory: File,
    private val gson: Gson = GsonBuilder().disableHtmlEscaping().create(),
) {
    private val progressDirectory = File(directory, "progress")

    init {
        synchronized(PROCESS_LOCK) {
            if ((!directory.exists() && !directory.mkdirs() && !directory.isDirectory) ||
                !directory.isDirectory
            ) {
                throw IOException("Unable to create indexing preflight directory: $directory")
            }
            if ((!progressDirectory.exists() &&
                    !progressDirectory.mkdirs() &&
                    !progressDirectory.isDirectory) ||
                !progressDirectory.isDirectory
            ) {
                throw IOException(
                    "Unable to create indexing preflight progress directory: $progressDirectory",
                )
            }
        }
    }

    fun create(intent: V2IndexingPreflightIntent) = synchronized(PROCESS_LOCK) {
        createLocked(intent)
    }

    /**
     * Checks every existing intent and creates this request under the same process-wide store lock.
     * Returns the first conflict without creating, or null after a successful create.
     */
    fun createIfNoConflict(
        intent: V2IndexingPreflightIntent,
        conflictsWith: (V2IndexingPreflightIntent) -> Boolean,
    ): V2IndexingPreflightIntent? = synchronized(PROCESS_LOCK) {
        requireNewIntent(intent)
        inspectLocked().requireComplete().firstOrNull(conflictsWith)
            ?.let { return@synchronized it }
        createLocked(intent)
        null
    }

    private fun createLocked(intent: V2IndexingPreflightIntent) {
        requireNewIntent(intent)
        val file = fileFor(intent.jobId)
        if (hasCommittedAtomicFile(file)) {
            throw V2IndexingPreflightIntentConflictException(
                "preflight intent ${intent.jobId} already exists",
            )
        }
        // A job ID is unique, so any sidecar or unfinished first-write artifact is stale.
        AtomicFile(file).delete()
        AtomicFile(progressFileFor(intent.jobId)).delete()
        writeAtomically(file, intent)
    }

    private fun requireNewIntent(intent: V2IndexingPreflightIntent) {
        V2IndexingPreflightIntentValidator.requireValid(intent)
        require(intent.revision == 0L && intent.state == V2IndexingPreflightIntentState.REQUESTED) {
            "new preflight intent must be REQUESTED at revision 0"
        }
    }

    fun load(jobId: String): V2IndexingPreflightIntent? = synchronized(PROCESS_LOCK) {
        loadLocked(jobId)
    }

    private fun loadLocked(jobId: String): V2IndexingPreflightIntent? {
        val current = loadAuthoritativeLocked(jobId) ?: return null
        val overlay = readProgressOverlayOrNull(progressFileFor(jobId)) ?: return current
        return V2IndexingPreflightProgressOverlayPolicy.applyIfCurrent(current, overlay)
    }

    private fun loadAuthoritativeLocked(jobId: String): V2IndexingPreflightIntent? {
        val file = fileFor(jobId)
        if (!hasCommittedAtomicFile(file)) return null
        return readAndValidate(file)
    }

    fun require(jobId: String): V2IndexingPreflightIntent = synchronized(PROCESS_LOCK) {
        loadLocked(jobId)
            ?: throw InvalidV2IndexingPreflightIntentException("unknown preflight intent $jobId")
    }

    fun updateLatest(
        jobId: String,
        transition: (V2IndexingPreflightIntent) -> V2IndexingPreflightIntent,
    ): V2IndexingPreflightIntent = synchronized(PROCESS_LOCK) {
        val current = loadLocked(jobId)
            ?: throw InvalidV2IndexingPreflightIntentException("unknown preflight intent $jobId")
        val updated = transition(current)
        if (updated == current) return@synchronized current
        requireImmutableRequest(current, updated)
        if (updated.revision != Math.addExact(current.revision, 1L)) {
            throw InvalidV2IndexingPreflightIntentException(
                "preflight update must advance one revision",
            )
        }
        V2IndexingPreflightIntentValidator.requireValid(updated)
        writeAtomically(fileFor(jobId), updated)
        // The main-file commit is the authority. A crash before this cleanup leaves an overlay
        // whose base revision cannot match and therefore cannot override the committed transition.
        AtomicFile(progressFileFor(jobId)).delete()
        updated
    }

    /** Persists only bounded progress evidence; no selected/result rows are rewritten. */
    fun persistProgressOverlay(
        jobId: String,
        progress: V2IndexingPreflightProgress,
        nowEpochMs: Long,
    ): V2IndexingPreflightIntent = synchronized(PROCESS_LOCK) {
        val current = loadAuthoritativeLocked(jobId)
            ?: throw InvalidV2IndexingPreflightIntentException("unknown preflight intent $jobId")
        if (current.state == V2IndexingPreflightIntentState.CANCEL_REQUESTED) {
            return@synchronized current
        }
        val overlay = V2IndexingPreflightProgressOverlayPolicy.create(
            current = current,
            progress = progress,
            nowEpochMs = nowEpochMs,
        )
        writeProgressOverlayAtomically(progressFileFor(jobId), overlay)
        current.copy(progress = progress)
    }

    fun list(): List<V2IndexingPreflightIntent> = synchronized(PROCESS_LOCK) {
        inspectLocked().requireComplete()
    }

    /** Reads every independently recoverable intent and reports corrupt files without hiding them. */
    fun inspect(): V2IndexingPreflightIntentInspection = synchronized(PROCESS_LOCK) {
        inspectLocked()
    }

    private fun inspectLocked(): V2IndexingPreflightIntentInspection {
        val intents = mutableListOf<V2IndexingPreflightIntent>()
        val issues = mutableListOf<V2IndexingPreflightIntentReadIssue>()
        jobEntriesLocked().forEach { entry ->
            try {
                loadLocked(entry.jobId)?.let(intents::add)
            } catch (error: Exception) {
                issues += V2IndexingPreflightIntentReadIssue(
                    jobId = entry.jobId,
                    fileName = entry.fileName,
                    message = (error.message ?: error::class.java.simpleName).take(512),
                )
            }
        }
        return V2IndexingPreflightIntentInspection(intents, issues)
    }

    private fun jobEntriesLocked(): List<DiscoveredIntentFile> = directory.listFiles().orEmpty()
        .mapNotNull { file ->
            val jobId = when {
                file.name.endsWith(FILE_SUFFIX) -> file.name.removeSuffix(FILE_SUFFIX)
                file.name.endsWith("$FILE_SUFFIX.bak") ->
                    file.name.removeSuffix("$FILE_SUFFIX.bak")
                else -> null
            } ?: return@mapNotNull null
            DiscoveredIntentFile(jobId, file.name)
        }
        .distinctBy(DiscoveredIntentFile::jobId)
        .sortedBy(DiscoveredIntentFile::jobId)

    fun delete(jobId: String): Boolean = synchronized(PROCESS_LOCK) {
        val file = fileFor(jobId)
        val progressFile = progressFileFor(jobId)
        AtomicFile(file).delete()
        AtomicFile(progressFile).delete()
        !hasAnyAtomicArtifact(file) && !hasAnyAtomicArtifact(progressFile)
    }

    private fun requireImmutableRequest(
        previous: V2IndexingPreflightIntent,
        updated: V2IndexingPreflightIntent,
    ) {
        if (previous.schemaVersion != updated.schemaVersion ||
            previous.jobId != updated.jobId ||
            previous.createdAtEpochMs != updated.createdAtEpochMs ||
            previous.selected != updated.selected ||
            previous.rebuildDerivedIndexes != updated.rebuildDerivedIndexes ||
            previous.executionProfile != updated.executionProfile ||
            (previous.baseGenerationId != null &&
                previous.baseGenerationId != updated.baseGenerationId)
        ) {
            throw InvalidV2IndexingPreflightIntentException(
                "preflight update cannot mutate its durable request",
            )
        }
    }

    private fun readAndValidate(file: File): V2IndexingPreflightIntent {
        try {
            val envelope = AtomicFile(file).openRead()
                .bufferedReader(StandardCharsets.UTF_8)
                .use { gson.fromJson(it, Envelope::class.java) }
                ?: throw InvalidV2IndexingPreflightIntentException("empty ${file.name}")
            if (envelope.format != V2IndexingPreflightIntentSchema.FORMAT ||
                envelope.schemaVersion != V2IndexingPreflightIntentSchema.VERSION ||
                envelope.intent.schemaVersion != envelope.schemaVersion
            ) {
                throw InvalidV2IndexingPreflightIntentException(
                    "unsupported preflight envelope ${file.name}",
                )
            }
            if (envelope.intent.jobId != file.name.removeSuffix(FILE_SUFFIX)) {
                throw InvalidV2IndexingPreflightIntentException(
                    "preflight filename/job ID mismatch",
                )
            }
            V2IndexingPreflightIntentValidator.requireValid(envelope.intent)
            return envelope.intent
        } catch (error: InvalidV2IndexingPreflightIntentException) {
            throw error
        } catch (error: Exception) {
            throw IOException("Unable to read indexing preflight ${file.name}", error)
        }
    }

    private fun writeAtomically(file: File, intent: V2IndexingPreflightIntent) {
        val atomic = AtomicFile(file)
        val stream = atomic.startWrite()
        try {
            val writer = OutputStreamWriter(stream, StandardCharsets.UTF_8)
            gson.toJson(
                Envelope(
                    format = V2IndexingPreflightIntentSchema.FORMAT,
                    schemaVersion = V2IndexingPreflightIntentSchema.VERSION,
                    intent = intent,
                ),
                writer,
            )
            writer.flush()
            atomic.finishWrite(stream)
        } catch (error: Exception) {
            atomic.failWrite(stream)
            throw IOException("Unable to persist indexing preflight ${file.name}", error)
        }
    }

    private fun readProgressOverlayOrNull(file: File): V2IndexingPreflightProgressOverlay? {
        if (!hasCommittedAtomicFile(file)) return null
        return runCatching {
            AtomicFile(file).openRead()
                .bufferedReader(StandardCharsets.UTF_8)
                .use { V2IndexingPreflightProgressOverlayCodec.readOrNull(gson, it) }
        }.getOrNull()
    }

    private fun writeProgressOverlayAtomically(
        file: File,
        overlay: V2IndexingPreflightProgressOverlay,
    ) {
        val atomic = AtomicFile(file)
        val stream = atomic.startWrite()
        try {
            val writer = OutputStreamWriter(stream, StandardCharsets.UTF_8)
            V2IndexingPreflightProgressOverlayCodec.write(gson, overlay, writer)
            writer.flush()
            atomic.finishWrite(stream)
        } catch (error: Exception) {
            atomic.failWrite(stream)
            throw IOException("Unable to persist indexing preflight progress ${file.name}", error)
        }
    }

    private fun fileFor(jobId: String): File {
        if (!SAFE_JOB_ID.matches(jobId)) {
            throw InvalidV2IndexingPreflightIntentException("unsafe preflight job ID")
        }
        return File(directory, "$jobId$FILE_SUFFIX")
    }

    private fun progressFileFor(jobId: String): File {
        if (!SAFE_JOB_ID.matches(jobId)) {
            throw InvalidV2IndexingPreflightIntentException("unsafe preflight job ID")
        }
        return File(progressDirectory, "$jobId$FILE_SUFFIX")
    }

    private fun backupFileFor(file: File): File = File(file.path + ".bak")

    private fun newFileFor(file: File): File = File(file.path + ".new")

    private fun hasCommittedAtomicFile(file: File): Boolean =
        file.exists() || backupFileFor(file).exists()

    private fun hasAnyAtomicArtifact(file: File): Boolean =
        hasCommittedAtomicFile(file) || newFileFor(file).exists()

    private data class Envelope(
        val format: String,
        val schemaVersion: Int,
        val intent: V2IndexingPreflightIntent,
    )

    private data class DiscoveredIntentFile(
        val jobId: String,
        val fileName: String,
    )

    private companion object {
        val PROCESS_LOCK = Any()
        const val FILE_SUFFIX = ".json"
        val SAFE_JOB_ID = Regex("^[A-Za-z0-9._-]{1,128}$")
    }
}
