package com.powerampstartradio.services

import com.google.gson.Gson
import com.powerampstartradio.ui.MAX_LIBRARY_ADDED_DAYS
import com.powerampstartradio.ui.RadioConfig
import com.powerampstartradio.ui.SelectionMode
import com.powerampstartradio.ui.forSelectionRequest
import com.powerampstartradio.widget.WidgetRadioSeedReference
import java.io.File
import java.util.concurrent.ConcurrentHashMap

internal enum class WidgetRadioIngressState {
    PENDING,
    FAILED,
}

/** Small command accepted from a widget before any library-scale work begins. */
internal data class WidgetRadioIngressRecord(
    val schemaVersion: Int = CURRENT_SCHEMA_VERSION,
    val commandId: String,
    val expectedSeed: WidgetRadioSeedReference,
    val config: RadioConfig,
    val state: WidgetRadioIngressState = WidgetRadioIngressState.PENDING,
    val terminalDetail: String? = null,
    val createdAtEpochMs: Long,
    val updatedAtEpochMs: Long = createdAtEpochMs,
) {
    companion object {
        const val CURRENT_SCHEMA_VERSION = 1
    }
}

internal sealed interface WidgetRadioIngressAdmission {
    data class Accepted(
        val record: WidgetRadioIngressRecord,
        val newlyPersisted: Boolean,
    ) : WidgetRadioIngressAdmission

    data class Busy(val pending: WidgetRadioIngressRecord) : WidgetRadioIngressAdmission
    data class AlreadyFailed(val record: WidgetRadioIngressRecord) : WidgetRadioIngressAdmission
}

internal data class WidgetRadioIngressFailureRecoveryResult(
    val terminalized: Boolean,
    val statusPublished: Boolean,
    val terminalizationFailure: Throwable?,
    val statusFailure: Throwable?,
)

/** Fail visibly when possible, retain a nonterminal ingress for retry, and always bound FGS life. */
internal fun recoverFailedWidgetIngress(
    store: WidgetRadioIngressStore,
    commandId: String,
    detail: String,
    publishFailure: (WidgetRadioSeedReference?) -> Unit,
    scheduleBoundedStop: () -> Unit,
): WidgetRadioIngressFailureRecoveryResult {
    val seed = runCatching { store.read(commandId)?.expectedSeed }.getOrNull()
    val terminalization = runCatching { store.markFailed(commandId, detail) }
    val status = runCatching { publishFailure(seed) }
    scheduleBoundedStop()
    return WidgetRadioIngressFailureRecoveryResult(
        terminalized = terminalization.getOrNull()?.state == WidgetRadioIngressState.FAILED,
        statusPublished = status.isSuccess,
        terminalizationFailure = terminalization.exceptionOrNull(),
        statusFailure = status.exceptionOrNull(),
    )
}

/**
 * Atomic receiver-to-service handoff. The receiver writes only this bounded record; provider,
 * database, and embedding work starts after RadioService owns a foreground execution context.
 */
internal class WidgetRadioIngressStore(
    rootDir: File,
    private val clock: () -> Long = System::currentTimeMillis,
    private val atomicWriter: RadioRequestAtomicWriter = AndroidRadioRequestAtomicWriter,
    private val atomicReader: RadioRequestAtomicReader = AndroidRadioRequestAtomicReader,
    private val atomicDeleter: RadioRequestAtomicDeleter = AndroidRadioRequestAtomicDeleter,
    private val gson: Gson = Gson(),
) {
    private val directory = File(rootDir, DIRECTORY_NAME)
    private val lock = locks.computeIfAbsent(directory.absoluteFile.normalize().path) { Any() }

    fun admit(
        commandId: String,
        expectedSeed: WidgetRadioSeedReference,
        config: RadioConfig,
    ): WidgetRadioIngressAdmission = admitAfterPrecommit(
        commandId = commandId,
        expectedSeed = expectedSeed,
        config = config,
        beforePendingPublication = {},
    )

    /**
     * Publish a new pending command only after its synchronous execution prerequisites succeed.
     * Existing/busy records never invoke [beforePendingPublication].
     */
    fun admitAfterPrecommit(
        commandId: String,
        expectedSeed: WidgetRadioSeedReference,
        config: RadioConfig,
        beforePendingPublication: (WidgetRadioIngressRecord) -> Unit,
    ): WidgetRadioIngressAdmission = synchronized(lock) {
        ensureDirectory()
        cleanupFailedLocked()
        val existing = readLocked(commandId)
        if (existing != null) {
            require(existing.expectedSeed == expectedSeed) {
                "Widget command ID belongs to a different displayed seed"
            }
            // The first accepted config remains authoritative. A later delivery may happen after
            // Settings changed, but it cannot rewrite or fail the already-journaled command.
            return@synchronized when (existing.state) {
                WidgetRadioIngressState.PENDING ->
                    WidgetRadioIngressAdmission.Accepted(existing, newlyPersisted = false)
                WidgetRadioIngressState.FAILED ->
                    WidgetRadioIngressAdmission.AlreadyFailed(existing)
            }
        }

        val pending = recordsLocked().firstOrNull { it.state == WidgetRadioIngressState.PENDING }
        if (pending != null) return@synchronized WidgetRadioIngressAdmission.Busy(pending)

        val now = clock()
        val record = WidgetRadioIngressRecord(
            commandId = commandId,
            expectedSeed = expectedSeed,
            config = config,
            createdAtEpochMs = now,
            updatedAtEpochMs = now,
        )
        beforePendingPublication(record)
        writeLocked(record)
        WidgetRadioIngressAdmission.Accepted(record, newlyPersisted = true)
    }

    fun read(commandId: String): WidgetRadioIngressRecord? = synchronized(lock) {
        readLocked(commandId)
    }

    fun pendingRecords(): List<WidgetRadioIngressRecord> = synchronized(lock) {
        ensureDirectory()
        recordsLocked()
            .filter { it.state == WidgetRadioIngressState.PENDING }
            .sortedWith(
                compareBy<WidgetRadioIngressRecord> { it.createdAtEpochMs }
                    .thenBy { it.commandId },
            )
    }

    fun markFailed(commandId: String, detail: String): WidgetRadioIngressRecord? =
        synchronized(lock) {
            val current = readLocked(commandId) ?: return@synchronized null
            if (current.state == WidgetRadioIngressState.FAILED) return@synchronized current
            val failed = current.copy(
                state = WidgetRadioIngressState.FAILED,
                terminalDetail = detail.take(MAX_TERMINAL_DETAIL_CHARS),
                updatedAtEpochMs = clock(),
            )
            writeLocked(failed)
            failed
        }

    fun delete(commandId: String): Boolean = synchronized(lock) {
        requireValidCommandId(commandId)
        atomicDeleter.delete(file(commandId))
    }

    private fun recordsLocked(): List<WidgetRadioIngressRecord> =
        directory.listFiles { candidate -> candidate.extension == FILE_EXTENSION }
            .orEmpty()
            .sortedBy(File::getName)
            .mapNotNull { candidate ->
                val commandId = candidate.nameWithoutExtension
                runCatching {
                    requireValidCommandId(commandId)
                    requireNotNull(readLocked(commandId)) {
                        "Widget ingress record disappeared while being read"
                    }
                }.getOrElse {
                    quarantineLocked(candidate)
                    null
                }
            }

    private fun quarantineLocked(candidate: File) {
        val quarantine = File(directory, QUARANTINE_DIRECTORY_NAME)
        val canRetain = (quarantine.exists() || quarantine.mkdirs()) && quarantine.isDirectory
        val suffix = ".${clock()}.invalid"
        val retained = canRetain && candidate.renameTo(File(quarantine, candidate.name + suffix))
        val backup = File(candidate.path + ".bak")
        if (backup.exists()) {
            if (!canRetain || !backup.renameTo(File(quarantine, backup.name + suffix))) {
                backup.delete()
            }
        }
        if (!retained) atomicDeleter.delete(candidate)
    }

    private fun readLocked(commandId: String): WidgetRadioIngressRecord? {
        requireValidCommandId(commandId)
        val bytes = atomicReader.read(file(commandId)) ?: return null
        require(bytes.isNotEmpty() && bytes.size <= MAX_RECORD_BYTES) {
            "Widget ingress record has invalid size"
        }
        val record = gson.fromJson(
            bytes.toString(Charsets.UTF_8),
            WidgetRadioIngressRecord::class.java,
        ) ?: throw IllegalArgumentException("Widget ingress record is empty")
        validate(record)
        require(record.commandId == commandId) { "Widget ingress filename/command mismatch" }
        return record
    }

    private fun writeLocked(record: WidgetRadioIngressRecord) {
        validate(record)
        ensureDirectory()
        val bytes = gson.toJson(record).toByteArray(Charsets.UTF_8)
        require(bytes.size <= MAX_RECORD_BYTES) { "Widget ingress record is too large" }
        atomicWriter.write(file(record.commandId), bytes)
    }

    private fun validate(record: WidgetRadioIngressRecord) {
        require(record.schemaVersion == WidgetRadioIngressRecord.CURRENT_SCHEMA_VERSION) {
            "Unsupported widget ingress schema"
        }
        requireValidCommandId(record.commandId)
        require(record.createdAtEpochMs > 0L && record.updatedAtEpochMs >= record.createdAtEpochMs) {
            "Invalid widget ingress timestamps"
        }
        if (record.state == WidgetRadioIngressState.PENDING) {
            require(record.terminalDetail == null) { "Pending widget ingress has terminal detail" }
        } else {
            require(!record.terminalDetail.isNullOrBlank()) {
                "Failed widget ingress has no terminal detail"
            }
        }
        validateSeed(record.expectedSeed)
        validateConfig(record.config)
    }

    private fun validateSeed(seed: WidgetRadioSeedReference) {
        require(seed.powerampFileId > 0L) { "Invalid widget seed file ID" }
        require(seed.normalizedTitle.isNotBlank()) { "Invalid widget seed title" }
        require(seed.displayTitle.isNotBlank() && seed.displayTitle.length <= MAX_TITLE_CHARS) {
            "Invalid widget display title"
        }
        seed.normalizedPath?.let { path ->
            require(path.isNotBlank() && path.length <= MAX_PATH_CHARS) {
                "Invalid widget seed path"
            }
        }
        seed.queueOccurrenceId?.let { require(it > 0L) { "Invalid widget queue occurrence" } }
        seed.embeddedTrackId?.let { require(it >= 0L) { "Invalid embedded widget seed" } }
    }

    private fun validateConfig(config: RadioConfig) {
        require(config.configSchemaVersion == RadioConfig.CURRENT_CONFIG_SCHEMA_VERSION)
        require(
            config.libraryAddedDays == null ||
                config.libraryAddedDays in 1..MAX_LIBRARY_ADDED_DAYS,
        )
        require(config == config.forSelectionRequest()) {
            "Widget ingress config is not a canonical selection request"
        }
        require(config.numTracks in 1..MAX_RADIO_TRACKS)
        require(config.candidatePoolSize == 0)
        requireFiniteFraction(config.mmrCandidatePoolFraction, exclusiveZero = true)
        requireFiniteFraction(config.dppFixedCandidatePoolFraction, exclusiveZero = true)
        requireFiniteFraction(config.anchorStrength)
        require(config.anchorHalfLifeTracks.isFinite() && config.anchorHalfLifeTracks > 0f)
        requireFiniteFraction(config.walkRestartAlpha)
        requireFiniteFraction(config.momentumBeta)
        requireFiniteFraction(config.diversityLambda)
        require(config.dppQualityExponent.isFinite() && config.dppQualityExponent in 0f..8f)
        require(config.maxPerArtist in 1..MAX_RADIO_TRACKS)
        require(config.minArtistSpacing in 0..MAX_RADIO_TRACKS)
        require(!config.driftEnabled || config.selectionMode == SelectionMode.MMR)
        config.selectionMode.name
        config.driftMode.name
        config.anchorDecay.name
    }

    private fun requireFiniteFraction(value: Float, exclusiveZero: Boolean = false) {
        require(
            value.isFinite() && value <= 1f &&
                if (exclusiveZero) value > 0f else value >= 0f,
        ) { "Widget ingress config contains an invalid fraction" }
    }

    private fun cleanupFailedLocked() {
        val cutoff = clock() - FAILED_RETENTION_MS
        recordsLocked().forEach { record ->
            if (record.state == WidgetRadioIngressState.FAILED &&
                record.updatedAtEpochMs < cutoff
            ) {
                atomicDeleter.delete(file(record.commandId))
            }
        }
    }

    private fun ensureDirectory() {
        require(directory.exists() || directory.mkdirs()) {
            "Could not create widget ingress directory"
        }
        require(directory.isDirectory) { "Widget ingress path is not a directory" }
    }

    private fun file(commandId: String): File = File(directory, "$commandId.$FILE_EXTENSION")

    private fun requireValidCommandId(commandId: String) {
        require(commandId.matches(COMMAND_ID_REGEX)) { "Invalid widget command ID" }
    }

    private companion object {
        const val DIRECTORY_NAME = "widget_radio_ingress_v2"
        const val QUARANTINE_DIRECTORY_NAME = "quarantine"
        const val FILE_EXTENSION = "json"
        const val MAX_RECORD_BYTES = 64 * 1024
        const val MAX_TERMINAL_DETAIL_CHARS = 1_024
        const val MAX_TITLE_CHARS = 512
        const val MAX_PATH_CHARS = 32_768
        const val MAX_RADIO_TRACKS = 1_000
        const val FAILED_RETENTION_MS = 7L * 24L * 60L * 60L * 1000L
        val COMMAND_ID_REGEX =
            Regex("[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}")
        val locks = ConcurrentHashMap<String, Any>()
    }
}
