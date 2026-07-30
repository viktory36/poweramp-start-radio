package com.powerampstartradio.indexing.v2

import android.util.AtomicFile
import com.google.gson.Gson
import com.google.gson.GsonBuilder
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.InputStream
import java.io.OutputStream
import java.io.OutputStreamWriter
import java.io.RandomAccessFile
import java.nio.charset.StandardCharsets
import java.security.MessageDigest

internal object V2IndexingLedgerFileNamespace {
    const val FILE_SUFFIX = ".json"
    const val JOURNAL_SUFFIX = ".journal-v1"
    private val SAFE_STORE_JOB_ID = Regex("^[A-Za-z0-9._-]{1,128}$")

    fun requireSafeJobId(jobId: String) {
        if (!SAFE_STORE_JOB_ID.matches(jobId)) {
            throw InvalidIndexingLedgerException("unsafe job id")
        }
        if (jobId.endsWith(
                V2ImportedRowAuthorizationFileNamespace.LEGACY_LEDGER_JOB_ID_SUFFIX,
            )
        ) {
            throw IndexingLedgerConflictException(
                "job ID collides with the legacy imported-row authorization namespace",
            )
        }
    }

    fun listedJobIds(files: Iterable<File>): List<String> = files
        .filterNot { file ->
            V2ImportedRowAuthorizationFileNamespace.isSidecarOrResidue(file.name)
        }
        .mapNotNull { file ->
            when {
                file.name.endsWith(FILE_SUFFIX) -> file.name.removeSuffix(FILE_SUFFIX)
                file.name.endsWith("$FILE_SUFFIX.bak") ->
                    file.name.removeSuffix("$FILE_SUFFIX.bak")
                else -> null
            }
        }
        .distinct()
        .sorted()
}

/** Injectable only so the ledger journal can be exercised by local JVM tests. */
internal interface V2LedgerBaseFileIo {
    fun hasCommittedFile(file: File): Boolean
    fun openRead(file: File): InputStream
    fun write(file: File, writer: (OutputStream) -> Unit)
}

private object AndroidAtomicV2LedgerBaseFileIo : V2LedgerBaseFileIo {
    override fun hasCommittedFile(file: File): Boolean =
        file.exists() || File(file.path + ".bak").exists()

    override fun openRead(file: File): InputStream = AtomicFile(file).openRead()

    override fun write(file: File, writer: (OutputStream) -> Unit) {
        val atomicFile = AtomicFile(file)
        val stream = atomicFile.startWrite()
        try {
            writer(stream)
            atomicFile.finishWrite(stream)
        } catch (error: Exception) {
            atomicFile.failWrite(stream)
            throw error
        }
    }
}

/**
 * One immutable full-ledger base file plus one append-only transition journal per durable job.
 *
 * The hot process keeps the fully validated current ledger in memory. A transition fsyncs one
 * length/checksum-framed delta containing the mutable job header and only changed descriptor and
 * track rows. A new process replays the journal and fully validates the reconstructed ledger.
 * Cross-process writers remain unsupported; V2 owns this store from one application process.
 */
class AtomicV2IndexingLedgerStore private constructor(
    private val directory: File,
    private val gson: Gson,
    private val baseFileIo: V2LedgerBaseFileIo,
    private val afterTerminalBaseWrite: (() -> Unit)?,
) {
    constructor(
        directory: File,
        gson: Gson = GsonBuilder().disableHtmlEscaping().create(),
    ) : this(directory, gson, AndroidAtomicV2LedgerBaseFileIo, null)

    internal constructor(
        directory: File,
        baseFileIo: V2LedgerBaseFileIo,
        gson: Gson = GsonBuilder().disableHtmlEscaping().create(),
        afterTerminalBaseWrite: (() -> Unit)? = null,
    ) : this(directory, gson, baseFileIo, afterTerminalBaseWrite)

    private val cache = mutableMapOf<String, CachedLedger>()

    init {
        if ((!directory.exists() && !directory.mkdirs()) || !directory.isDirectory) {
            throw IOException("Unable to create indexing ledger directory: $directory")
        }
    }

    @Synchronized
    fun create(ledger: IndexingJobLedger) {
        V2IndexingLedgerValidator.requireValid(ledger)
        if (ledger.revision != 0L || ledger.state != IndexingJobState.PLANNED) {
            throw InvalidIndexingLedgerException("new ledger must be PLANNED at revision 0")
        }
        val file = fileFor(ledger.jobSpec.jobId)
        val journal = journalFileFor(ledger.jobSpec.jobId)
        if (baseFileIo.hasCommittedFile(file) || journal.exists()) {
            throw IndexingLedgerConflictException("job ${ledger.jobSpec.jobId} already exists")
        }
        writeBaseAtomically(file, ledger)
        cache[ledger.jobSpec.jobId] = CachedLedger(
            ledger = ledger,
            summary = LedgerValidationSummary.from(ledger),
            journalLength = 0L,
        )
    }

    @Synchronized
    fun load(jobId: String): IndexingJobLedger? {
        cache[jobId]?.let { return it.ledger }
        val file = fileFor(jobId)
        val journal = journalFileFor(jobId)
        if (!baseFileIo.hasCommittedFile(file)) {
            if (journal.exists()) {
                throw InvalidIndexingLedgerException("journal exists without base ledger for $jobId")
            }
            return null
        }

        val base = readAndValidateBase(file)
        var replayed = replayJournal(base, journal)
        V2IndexingLedgerValidator.requireValid(replayed.ledger)
        if (replayed.ledger.state in TERMINAL_JOB_STATES && replayed.journalLength > 0L) {
            // Recovery may have replayed the terminal delta before its base replacement began.
            writeBaseAtomically(file, replayed.ledger)
            retireJournal(journal)
            replayed = replayed.copy(journalLength = 0L)
        }
        cache[jobId] = replayed
        return replayed.ledger
    }

    @Synchronized
    fun require(jobId: String): IndexingJobLedger = load(jobId)
        ?: throw InvalidIndexingLedgerException("unknown indexing job $jobId")

    /** Persist exactly one state-machine transition. */
    @Synchronized
    fun update(
        jobId: String,
        expectedRevision: Long,
        transition: (IndexingJobLedger) -> IndexingJobLedger,
    ): IndexingJobLedger {
        val current = requireCached(jobId)
        if (current.ledger.revision != expectedRevision) {
            throw IndexingLedgerConflictException(
                "job $jobId is revision ${current.ledger.revision}, expected $expectedRevision",
            )
        }
        return persistTransition(jobId, current, transition)
    }

    /** Cached update for the process-serialized repository/controller path. */
    @Synchronized
    fun updateLatest(
        jobId: String,
        transition: (IndexingJobLedger) -> IndexingJobLedger,
    ): IndexingJobLedger = persistTransition(jobId, requireCached(jobId), transition)

    private fun persistTransition(
        jobId: String,
        current: CachedLedger,
        transition: (IndexingJobLedger) -> IndexingJobLedger,
    ): IndexingJobLedger {
        val updated = V2IndexingLedgerValidator.withStoreManagedTransitionValidation {
            transition(current.ledger)
        }
        if (updated === current.ledger || updated == current.ledger) return current.ledger
        return persistResolvedTransition(jobId, current, updated)
    }

    private fun persistResolvedTransition(
        jobId: String,
        current: CachedLedger,
        updated: IndexingJobLedger,
    ): IndexingJobLedger {
        val delta = createDelta(current.ledger, updated)
        val nextSummary = requireValidIncrementalTransition(
            previous = current.ledger,
            updated = updated,
            previousSummary = current.summary,
            descriptorIndices = delta.descriptorChanges.mapTo(linkedSetOf()) { it.ordinal },
            trackIndices = delta.trackChanges.mapTo(linkedSetOf()) { it.ordinal },
        )
        val journal = journalFileFor(jobId)
        if (journal.length() != current.journalLength) {
            cache.remove(jobId)
            throw IndexingLedgerConflictException("job $jobId journal changed outside this store")
        }
        val newLength = try {
            appendRecord(journal, delta)
        } catch (error: Exception) {
            cache.remove(jobId)
            throw IOException("Unable to append indexing ledger journal ${journal.name}", error)
        }
        cache[jobId] = CachedLedger(updated, nextSummary, newLength)
        if (updated.state in TERMINAL_JOB_STATES) {
            try {
                writeBaseAtomically(fileFor(jobId), updated)
                afterTerminalBaseWrite?.invoke()
                retireJournal(journal)
            } catch (error: Exception) {
                cache.remove(jobId)
                throw IOException("Unable to compact terminal indexing ledger $jobId", error)
            }
            cache[jobId] = CachedLedger(updated, nextSummary, 0L)
        }
        return updated
    }

    /** Reconcile startup state and persist it only when reconciliation changed evidence. */
    @Synchronized
    fun reconcileAfterProcessRestart(
        jobId: String,
        expectedRevision: Long,
        nowEpochMs: Long,
    ): RestartReconciliation {
        val current = requireCached(jobId)
        if (current.ledger.revision != expectedRevision) {
            throw IndexingLedgerConflictException(
                "job $jobId is revision ${current.ledger.revision}, expected $expectedRevision",
            )
        }
        // Restart is the deliberate full-validation boundary before recovery mutates evidence.
        V2IndexingLedgerValidator.requireValid(current.ledger)
        val result = V2IndexingLedgerValidator.withStoreManagedTransitionValidation {
            V2IndexingLedgerStateMachine.reconcileAfterProcessRestart(
                ledger = current.ledger,
                nowEpochMs = nowEpochMs,
            )
        }
        if (!result.changed) return result
        val persisted = persistResolvedTransition(jobId, current, result.ledger)
        return result.copy(ledger = persisted)
    }

    @Synchronized
    fun list(): List<IndexingJobLedger> {
        val jobIds = V2IndexingLedgerFileNamespace.listedJobIds(
            directory.listFiles().orEmpty().asIterable(),
        )
        return jobIds.mapNotNull(::load)
    }

    private fun requireCached(jobId: String): CachedLedger {
        load(jobId)
        return cache[jobId]
            ?: throw InvalidIndexingLedgerException("unknown indexing job $jobId")
    }

    private fun readAndValidateBase(file: File): IndexingJobLedger {
        try {
            val envelope = baseFileIo.openRead(file).bufferedReader(StandardCharsets.UTF_8).use { reader ->
                gson.fromJson(reader, LedgerEnvelope::class.java)
            } ?: throw InvalidIndexingLedgerException("empty ledger file ${file.name}")
            if (envelope.format != V2IndexingLedgerSchema.FORMAT) {
                throw InvalidIndexingLedgerException("unrecognized ledger format ${envelope.format}")
            }
            if (envelope.schemaVersion != V2IndexingLedgerSchema.VERSION) {
                throw UnsupportedIndexingLedgerSchemaException(
                    "ledger ${file.name} uses schema ${envelope.schemaVersion}",
                )
            }
            if (envelope.ledger.schemaVersion != envelope.schemaVersion) {
                throw InvalidIndexingLedgerException("envelope/ledger schema mismatch")
            }
            val fileJobId = file.name.removeSuffix(V2IndexingLedgerFileNamespace.FILE_SUFFIX)
            if (envelope.ledger.jobSpec.jobId != fileJobId) {
                throw InvalidIndexingLedgerException("ledger filename/jobId mismatch")
            }
            V2IndexingLedgerValidator.requireValid(envelope.ledger)
            return envelope.ledger
        } catch (error: UnsupportedIndexingLedgerSchemaException) {
            throw error
        } catch (error: InvalidIndexingLedgerException) {
            throw error
        } catch (error: Exception) {
            throw IOException("Unable to read indexing ledger ${file.name}", error)
        }
    }

    private fun replayJournal(base: IndexingJobLedger, journal: File): CachedLedger {
        if (!journal.exists() || journal.length() == 0L) {
            return CachedLedger(base, LedgerValidationSummary.from(base), 0L)
        }
        if (base.state in TERMINAL_JOB_STATES) {
            if (journalProvesCompletedCompaction(base, journal) || journal.length() == 0L) {
                retireJournal(journal)
                return CachedLedger(base, LedgerValidationSummary.from(base), 0L)
            }
            throw InvalidIndexingLedgerException(
                "terminal base ledger has an unbound non-empty journal",
            )
        }
        var ledger = base
        var summary = LedgerValidationSummary.from(base)
        val replay = readJournalRecords(journal) { delta ->
            val applied = applyDelta(ledger, delta)
            val nextSummary = requireValidIncrementalTransition(
                previous = ledger,
                updated = applied,
                previousSummary = summary,
                descriptorIndices = delta.descriptorChanges.mapTo(linkedSetOf()) { it.ordinal },
                trackIndices = delta.trackChanges.mapTo(linkedSetOf()) { it.ordinal },
            )
            ledger = applied
            summary = nextSummary
        }
        return CachedLedger(ledger, summary, replay.validLength)
    }

    private fun readJournalRecords(
        journal: File,
        consume: (LedgerJournalDelta) -> Unit,
    ): JournalReadResult {
        var validLength = 0L
        var recordCount = 0
        var tornTail = false
        try {
            RandomAccessFile(journal, "rw").use { input ->
                while (input.filePointer < input.length()) {
                    val frameStart = input.filePointer
                    val remaining = input.length() - frameStart
                    if (remaining < JOURNAL_HEADER_BYTES) {
                        tornTail = true
                        break
                    }
                    val magic = ByteArray(JOURNAL_MAGIC.size)
                    input.readFully(magic)
                    if (!magic.contentEquals(JOURNAL_MAGIC)) {
                        throw InvalidIndexingLedgerException(
                            "invalid journal frame magic at byte $frameStart",
                        )
                    }
                    val payloadLength = input.readInt()
                    val payloadLengthComplement = input.readInt()
                    if (payloadLength.inv() != payloadLengthComplement ||
                        payloadLength <= 0 || payloadLength > MAX_JOURNAL_RECORD_BYTES
                    ) {
                        throw InvalidIndexingLedgerException(
                            "invalid journal frame length at byte $frameStart",
                        )
                    }
                    val expectedSha256 = ByteArray(SHA256_BYTES)
                    input.readFully(expectedSha256)
                    if (input.length() - input.filePointer < payloadLength.toLong()) {
                        tornTail = true
                        break
                    }
                    val payload = ByteArray(payloadLength)
                    input.readFully(payload)
                    val actualSha256 = MessageDigest.getInstance("SHA-256").digest(payload)
                    if (!actualSha256.contentEquals(expectedSha256)) {
                        throw InvalidIndexingLedgerException(
                            "journal frame checksum mismatch at byte $frameStart",
                        )
                    }
                    val delta = try {
                        gson.fromJson(
                            String(payload, StandardCharsets.UTF_8),
                            LedgerJournalDelta::class.java,
                        )
                    } catch (error: Exception) {
                        throw InvalidIndexingLedgerException(
                            "invalid journal JSON at byte $frameStart: ${error.message}",
                        )
                    } ?: throw InvalidIndexingLedgerException(
                        "empty journal record at byte $frameStart",
                    )
                    consume(delta)
                    recordCount += 1
                    validLength = input.filePointer
                }
                if (tornTail) {
                    input.setLength(validLength)
                    input.fd.sync()
                }
            }
        } catch (error: UnsupportedIndexingLedgerSchemaException) {
            throw error
        } catch (error: InvalidIndexingLedgerException) {
            throw error
        } catch (error: Exception) {
            throw IOException("Unable to replay indexing ledger journal ${journal.name}", error)
        }
        return JournalReadResult(recordCount, validLength)
    }

    private fun journalProvesCompletedCompaction(
        base: IndexingJobLedger,
        journal: File,
    ): Boolean {
        var expectedRevision: Long? = null
        var expectedJobSpecId: String? = null
        var last: LedgerJournalDelta? = null
        var valid = true
        val read = readJournalRecords(journal) { delta ->
            val prior = last
            if (prior?.terminalLedgerSha256 != null) valid = false
            val requiredRevision = expectedRevision ?: delta.previousRevision
            val requiredJobSpecId = expectedJobSpecId ?: delta.previousJobSpecId
            if (delta.format != JOURNAL_RECORD_FORMAT ||
                delta.schemaVersion != JOURNAL_SCHEMA_VERSION ||
                delta.jobId != base.jobSpec.jobId ||
                delta.previousRevision != requiredRevision ||
                delta.revision != Math.addExact(requiredRevision, 1L) ||
                delta.previousJobSpecId != requiredJobSpecId
            ) {
                valid = false
            }
            expectedRevision = delta.revision
            expectedJobSpecId = delta.jobSpecIdentity?.specId ?: requiredJobSpecId
            last = delta
        }
        val terminal = last ?: return false
        return valid && read.recordCount > 0 &&
            terminal.revision == base.revision &&
            expectedJobSpecId == base.jobSpec.specId &&
            terminal.header == base.journalHeader() &&
            terminal.terminalLedgerSha256 == ledgerSha256(base)
    }

    private fun retireJournal(journal: File) {
        if (!journal.exists()) return
        RandomAccessFile(journal, "rw").use { file ->
            file.setLength(0L)
            file.fd.sync()
        }
    }

    private fun createDelta(
        previous: IndexingJobLedger,
        updated: IndexingJobLedger,
    ): LedgerJournalDelta {
        if (updated.jobSpec.jobId != previous.jobSpec.jobId) {
            throw InvalidIndexingLedgerException("a ledger update cannot change job ID")
        }
        if (updated.revision != Math.addExact(previous.revision, 1L)) {
            throw InvalidIndexingLedgerException("a ledger update must advance one revision")
        }
        if (updated.jobSpec.tracks.size != previous.jobSpec.tracks.size ||
            updated.tracks.size != previous.tracks.size
        ) {
            throw InvalidIndexingLedgerException("a ledger update cannot resize the track plan")
        }

        val jobSpecChange = if (updated.jobSpec == previous.jobSpec) {
            null
        } else {
            try {
                V2DecodedEosPlanFinalizer.requireAllowedMutation(previous, updated)
            } catch (error: Exception) {
                throw InvalidIndexingLedgerException(
                    "a ledger update cannot mutate the job spec outside EOS finalization: " +
                        error.message,
                )
            }
            JournalJobSpecIdentity(
                specId = updated.jobSpec.specId,
                provisionalParentSpecId = updated.jobSpec.provisionalParentSpecId,
            )
        }
        val descriptorChanges = previous.jobSpec.tracks.indices.mapNotNull { ordinal ->
            val old = previous.jobSpec.tracks[ordinal]
            val fresh = updated.jobSpec.tracks[ordinal]
            if (old == fresh) null else JournalDescriptorChange(ordinal, old.workId, fresh)
        }
        val trackChanges = previous.tracks.indices.mapNotNull { ordinal ->
            val old = previous.tracks[ordinal]
            val fresh = updated.tracks[ordinal]
            if (old == fresh) null else JournalTrackChange(ordinal, old.workId, fresh)
        }
        if ((jobSpecChange == null) != descriptorChanges.isEmpty()) {
            throw InvalidIndexingLedgerException("job-spec identity and descriptor delta disagree")
        }
        return LedgerJournalDelta(
            format = JOURNAL_RECORD_FORMAT,
            schemaVersion = JOURNAL_SCHEMA_VERSION,
            jobId = previous.jobSpec.jobId,
            previousRevision = previous.revision,
            revision = updated.revision,
            previousJobSpecId = previous.jobSpec.specId,
            header = updated.journalHeader(),
            jobSpecIdentity = jobSpecChange,
            descriptorChanges = descriptorChanges,
            trackChanges = trackChanges,
            terminalLedgerSha256 = if (updated.state in TERMINAL_JOB_STATES) {
                ledgerSha256(updated)
            } else {
                null
            },
        )
    }

    private fun applyDelta(
        previous: IndexingJobLedger,
        delta: LedgerJournalDelta,
    ): IndexingJobLedger {
        if (delta.format != JOURNAL_RECORD_FORMAT) {
            throw InvalidIndexingLedgerException("unrecognized ledger journal format ${delta.format}")
        }
        if (delta.schemaVersion != JOURNAL_SCHEMA_VERSION) {
            throw UnsupportedIndexingLedgerSchemaException(
                "ledger journal uses schema ${delta.schemaVersion}",
            )
        }
        if (delta.jobId != previous.jobSpec.jobId ||
            delta.previousJobSpecId != previous.jobSpec.specId
        ) {
            throw InvalidIndexingLedgerException("journal record does not bind its base ledger")
        }
        if (delta.previousRevision != previous.revision ||
            delta.revision != Math.addExact(previous.revision, 1L)
        ) {
            throw InvalidIndexingLedgerException("journal revision chain is discontinuous")
        }

        val descriptorIndices = requireUniqueIndices(
            delta.descriptorChanges.map { it.ordinal },
            previous.jobSpec.tracks.size,
            "descriptor",
        )
        val trackIndices = requireUniqueIndices(
            delta.trackChanges.map { it.ordinal },
            previous.tracks.size,
            "track",
        )
        if ((delta.jobSpecIdentity == null) != descriptorIndices.isEmpty()) {
            throw InvalidIndexingLedgerException("journal job-spec identity and descriptor delta disagree")
        }

        val descriptors = previous.jobSpec.tracks.toMutableList()
        delta.descriptorChanges.forEach { change ->
            if (descriptors[change.ordinal].workId != change.previousWorkId) {
                throw InvalidIndexingLedgerException("journal descriptor base identity mismatch")
            }
            descriptors[change.ordinal] = change.descriptor
        }
        val spec = delta.jobSpecIdentity?.let { identity ->
            previous.jobSpec.copy(
                specId = identity.specId,
                provisionalParentSpecId = identity.provisionalParentSpecId,
                tracks = descriptors,
            )
        } ?: previous.jobSpec

        val tracks = previous.tracks.toMutableList()
        delta.trackChanges.forEach { change ->
            if (tracks[change.ordinal].workId != change.previousWorkId) {
                throw InvalidIndexingLedgerException("journal track base identity mismatch")
            }
            tracks[change.ordinal] = change.track
        }
        val updated = previous.copy(
            jobSpec = spec,
            state = delta.header.state,
            recoveryPhase = delta.header.recoveryPhase,
            revision = delta.revision,
            updatedAtEpochMs = delta.header.updatedAtEpochMs,
            stateReason = delta.header.stateReason,
            tracks = tracks,
            executionProfile = delta.header.executionProfile,
            activationEvidence = delta.header.activationEvidence,
        )
        val expectedTerminalDigest = if (updated.state in TERMINAL_JOB_STATES) {
            ledgerSha256(updated)
        } else {
            null
        }
        if (delta.terminalLedgerSha256 != expectedTerminalDigest) {
            throw InvalidIndexingLedgerException("journal terminal-ledger digest mismatch")
        }
        return updated
    }

    private fun requireValidIncrementalTransition(
        previous: IndexingJobLedger,
        updated: IndexingJobLedger,
        previousSummary: LedgerValidationSummary,
        descriptorIndices: Set<Int>,
        trackIndices: Set<Int>,
    ): LedgerValidationSummary {
        if (updated.schemaVersion != previous.schemaVersion ||
            updated.schemaVersion != V2IndexingLedgerSchema.VERSION
        ) {
            throw InvalidIndexingLedgerException("a ledger update cannot change schema version")
        }
        if (updated.jobSpec.jobId != previous.jobSpec.jobId ||
            updated.jobSpec.createdAtEpochMs != previous.jobSpec.createdAtEpochMs
        ) {
            throw InvalidIndexingLedgerException("a ledger update cannot change job identity")
        }
        if (updated.updatedAtEpochMs < previous.updatedAtEpochMs ||
            updated.updatedAtEpochMs < updated.jobSpec.createdAtEpochMs
        ) {
            throw InvalidIndexingLedgerException("a ledger update moved time backwards")
        }
        if (updated.stateReason != null && updated.stateReason.length > 2_048) {
            throw InvalidIndexingLedgerException("job state reason is too long")
        }
        if (updated.revision != Math.addExact(previous.revision, 1L)) {
            throw InvalidIndexingLedgerException("a ledger update must advance one revision")
        }

        if (updated.jobSpec == previous.jobSpec) {
            if (descriptorIndices.isNotEmpty()) {
                throw InvalidIndexingLedgerException("descriptor delta changed an immutable job spec")
            }
        } else {
            try {
                V2DecodedEosPlanFinalizer.requireAllowedMutation(previous, updated)
                V2IndexingLedgerValidator.requireValidEosJobSpecDelta(
                    updated.jobSpec,
                    descriptorIndices,
                )
            } catch (error: Exception) {
                throw InvalidIndexingLedgerException(
                    "invalid EOS job-spec mutation: ${error.message}",
                )
            }
        }

        val changedIndices = (descriptorIndices + trackIndices).toSortedSet()
        changedIndices.forEach { ordinal ->
            if (ordinal !in updated.tracks.indices || ordinal !in updated.jobSpec.tracks.indices) {
                throw InvalidIndexingLedgerException("changed track ordinal is out of bounds")
            }
            val oldTrack = previous.tracks[ordinal]
            val freshTrack = updated.tracks[ordinal]
            if (freshTrack.updatedAtEpochMs < oldTrack.updatedAtEpochMs) {
                throw InvalidIndexingLedgerException("track update moved time backwards")
            }
            V2IndexingLedgerValidator.requireValidChangedTrack(
                track = freshTrack,
                descriptor = updated.jobSpec.tracks[ordinal],
                spec = updated.jobSpec,
                ledgerUpdatedAtEpochMs = updated.updatedAtEpochMs,
            )
        }

        val summary = previousSummary.updated(previous, updated, changedIndices)
        summary.requireCompatibleWith(updated)
        V2IndexingLedgerValidator.requireValidChangedActivationEvidence(updated)
        return summary
    }

    private fun requireUniqueIndices(
        indices: List<Int>,
        size: Int,
        label: String,
    ): Set<Int> {
        if (indices.any { it !in 0 until size } || indices.toSet().size != indices.size) {
            throw InvalidIndexingLedgerException("invalid or repeated journal $label ordinal")
        }
        return indices.toSet()
    }

    private fun appendRecord(file: File, delta: LedgerJournalDelta): Long {
        val payload = gson.toJson(delta).toByteArray(StandardCharsets.UTF_8)
        if (payload.isEmpty() || payload.size > MAX_JOURNAL_RECORD_BYTES) {
            throw InvalidIndexingLedgerException("ledger journal record has invalid size ${payload.size}")
        }
        val sha256 = MessageDigest.getInstance("SHA-256").digest(payload)
        FileOutputStream(file, true).use { output ->
            output.write(JOURNAL_MAGIC)
            output.writeIntBigEndian(payload.size)
            output.writeIntBigEndian(payload.size.inv())
            output.write(sha256)
            output.write(payload)
            output.flush()
            output.fd.sync()
        }
        return file.length()
    }

    private fun IndexingJobLedger.journalHeader() = JournalJobHeader(
        state = state,
        recoveryPhase = recoveryPhase,
        updatedAtEpochMs = updatedAtEpochMs,
        stateReason = stateReason,
        executionProfile = executionProfile,
        activationEvidence = activationEvidence,
    )

    private fun ledgerSha256(ledger: IndexingJobLedger): String =
        MessageDigest.getInstance("SHA-256")
            .digest(gson.toJson(ledger).toByteArray(StandardCharsets.UTF_8))
            .joinToString("") { byte -> "%02x".format(byte.toInt() and 0xff) }

    private fun OutputStream.writeIntBigEndian(value: Int) {
        write(value ushr 24 and 0xff)
        write(value ushr 16 and 0xff)
        write(value ushr 8 and 0xff)
        write(value and 0xff)
    }

    private fun writeBaseAtomically(file: File, ledger: IndexingJobLedger) {
        try {
            baseFileIo.write(file) { stream ->
                val writer = OutputStreamWriter(stream, StandardCharsets.UTF_8)
                gson.toJson(
                    LedgerEnvelope(
                        format = V2IndexingLedgerSchema.FORMAT,
                        schemaVersion = V2IndexingLedgerSchema.VERSION,
                        ledger = ledger,
                    ),
                    writer,
                )
                writer.flush()
            }
        } catch (error: Exception) {
            throw IOException("Unable to persist indexing ledger ${file.name}", error)
        }
    }

    private fun fileFor(jobId: String): File {
        V2IndexingLedgerFileNamespace.requireSafeJobId(jobId)
        return File(directory, "$jobId${V2IndexingLedgerFileNamespace.FILE_SUFFIX}")
    }

    private fun journalFileFor(jobId: String): File {
        V2IndexingLedgerFileNamespace.requireSafeJobId(jobId)
        return File(directory, "$jobId${V2IndexingLedgerFileNamespace.JOURNAL_SUFFIX}")
    }

    private data class CachedLedger(
        val ledger: IndexingJobLedger,
        val summary: LedgerValidationSummary,
        val journalLength: Long,
    )

    private data class JournalReadResult(
        val recordCount: Int,
        val validLength: Long,
    )

    private data class LedgerEnvelope(
        val format: String,
        val schemaVersion: Int,
        val ledger: IndexingJobLedger,
    )

    private data class LedgerJournalDelta(
        val format: String,
        val schemaVersion: Int,
        val jobId: String,
        val previousRevision: Long,
        val revision: Long,
        val previousJobSpecId: String,
        val header: JournalJobHeader,
        val jobSpecIdentity: JournalJobSpecIdentity?,
        val descriptorChanges: List<JournalDescriptorChange>,
        val trackChanges: List<JournalTrackChange>,
        val terminalLedgerSha256: String?,
    )

    private data class JournalJobHeader(
        val state: IndexingJobState,
        val recoveryPhase: RecoveryPhase?,
        val updatedAtEpochMs: Long,
        val stateReason: String?,
        val executionProfile: V2IndexingExecutionProfile,
        val activationEvidence: ActivatedGenerationEvidence?,
    )

    private data class JournalJobSpecIdentity(
        val specId: String,
        val provisionalParentSpecId: String?,
    )

    private data class JournalDescriptorChange(
        val ordinal: Int,
        val previousWorkId: String,
        val descriptor: SelectedTrackDescriptor,
    )

    private data class JournalTrackChange(
        val ordinal: Int,
        val previousWorkId: String,
        val track: IndexingTrackLedger,
    )

    private data class LedgerValidationSummary(
        val activeTracks: Int,
        val unresolvedTracks: Int,
        val retryableFailures: Int,
        val databaseCommits: Int,
        val runnableProvisionalTracks: Int,
    ) {
        fun updated(
            previous: IndexingJobLedger,
            updated: IndexingJobLedger,
            changedIndices: Set<Int>,
        ): LedgerValidationSummary {
            var result = this
            changedIndices.forEach { ordinal ->
                result -= contribution(
                    previous.jobSpec.tracks[ordinal],
                    previous.tracks[ordinal],
                )
                result += contribution(
                    updated.jobSpec.tracks[ordinal],
                    updated.tracks[ordinal],
                )
            }
            return result
        }

        fun requireCompatibleWith(ledger: IndexingJobLedger) {
            if (activeTracks > 1) {
                throw InvalidIndexingLedgerException("more than one track owns the executor")
            }
            if (runnableProvisionalTracks > 0 && databaseCommits > 0) {
                throw InvalidIndexingLedgerException(
                    "database commit exists while a runnable ordinary span is provisional",
                )
            }
            when (ledger.state) {
                IndexingJobState.INTERRUPTED,
                IndexingJobState.READY_TO_RESUME,
                -> if (ledger.recoveryPhase == null) {
                    throw InvalidIndexingLedgerException(
                        "recovery phase missing for ${ledger.state}",
                    )
                }

                else -> if (ledger.recoveryPhase != null) {
                    throw InvalidIndexingLedgerException(
                        "recovery phase set outside interrupted/resume state",
                    )
                }
            }
            if (ledger.state == IndexingJobState.ACTIVATING ||
                ledger.state == IndexingJobState.COMPLETE
            ) {
                if (unresolvedTracks != 0) {
                    throw InvalidIndexingLedgerException("${ledger.state} has unresolved tracks")
                }
            }
            if (ledger.state in NO_ACTIVE_TRACK_JOB_STATES && activeTracks != 0) {
                throw InvalidIndexingLedgerException("${ledger.state} retains an active track stage")
            }
            if (ledger.state == IndexingJobState.COMPLETE && retryableFailures != 0) {
                throw InvalidIndexingLedgerException("complete job retains retryable failure")
            }
        }

        private operator fun plus(other: LedgerValidationSummary) = LedgerValidationSummary(
            activeTracks + other.activeTracks,
            unresolvedTracks + other.unresolvedTracks,
            retryableFailures + other.retryableFailures,
            databaseCommits + other.databaseCommits,
            runnableProvisionalTracks + other.runnableProvisionalTracks,
        )

        private operator fun minus(other: LedgerValidationSummary) = LedgerValidationSummary(
            activeTracks - other.activeTracks,
            unresolvedTracks - other.unresolvedTracks,
            retryableFailures - other.retryableFailures,
            databaseCommits - other.databaseCommits,
            runnableProvisionalTracks - other.runnableProvisionalTracks,
        )

        companion object {
            fun from(ledger: IndexingJobLedger): LedgerValidationSummary = ledger.tracks.indices
                .fold(ZERO) { result, ordinal ->
                    result + contribution(
                        ledger.jobSpec.tracks[ordinal],
                        ledger.tracks[ordinal],
                    )
                }

            private fun contribution(
                descriptor: SelectedTrackDescriptor,
                track: IndexingTrackLedger,
            ) = LedgerValidationSummary(
                activeTracks = if (track.state.isActiveStage()) 1 else 0,
                unresolvedTracks = if (track.state.isResolvedForActivation()) 0 else 1,
                retryableFailures = if (track.state == IndexingTrackState.RETRYABLE_FAILURE) 1 else 0,
                databaseCommits = if (track.verifiedArtifacts.any {
                        it.kind == VerifiedArtifactKind.DATABASE_COMMIT
                    }
                ) 1 else 0,
                runnableProvisionalTracks = if (
                    V2IndexingPlanFinalizationPolicy.isRunnableProvisional(descriptor, track)
                ) 1 else 0,
            )

            private val ZERO = LedgerValidationSummary(0, 0, 0, 0, 0)
        }
    }

    private companion object {
        const val JOURNAL_RECORD_FORMAT = "poweramp-start-radio-v2-indexing-ledger-journal"
        const val JOURNAL_SCHEMA_VERSION = 1
        const val SHA256_BYTES = 32
        const val MAX_JOURNAL_RECORD_BYTES = 256 * 1024 * 1024
        val JOURNAL_MAGIC = byteArrayOf(0x50, 0x53, 0x52, 0x32, 0x4a, 0x52, 0x30, 0x31)
        val JOURNAL_HEADER_BYTES = JOURNAL_MAGIC.size + Int.SIZE_BYTES * 2 + SHA256_BYTES
        val NO_ACTIVE_TRACK_JOB_STATES = setOf(
            IndexingJobState.PAUSED,
            IndexingJobState.WAITING_FOR_INPUT,
            IndexingJobState.INTERRUPTED,
            IndexingJobState.READY_TO_RESUME,
            IndexingJobState.CANCELLED,
            IndexingJobState.ACTIVATING,
            IndexingJobState.COMPLETE,
        )
        val TERMINAL_JOB_STATES = setOf(
            IndexingJobState.CANCELLED,
            IndexingJobState.COMPLETE,
        )
    }
}
