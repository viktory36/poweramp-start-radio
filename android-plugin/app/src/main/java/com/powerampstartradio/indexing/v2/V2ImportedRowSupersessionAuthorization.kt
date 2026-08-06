package com.powerampstartradio.indexing.v2

import android.database.sqlite.SQLiteDatabase
import android.util.AtomicFile
import com.google.gson.Gson
import com.google.gson.GsonBuilder
import com.powerampstartradio.indexing.V2LegacyCompatibilityBinding
import com.powerampstartradio.indexing.V2LegacyCompatibilityEvidence
import com.powerampstartradio.indexing.V2LegacyCompatibilityResolver
import com.powerampstartradio.indexing.V2LegacyDatabaseCandidate
import com.powerampstartradio.indexing.V2LegacyProviderCandidate
import com.powerampstartradio.poweramp.TrackNormalization
import java.io.File
import java.io.IOException
import java.nio.charset.StandardCharsets
import java.security.MessageDigest

internal object V2ImportedRowSupersessionAuthorizationSchema {
    const val VERSION = 1
    const val FORMAT = "poweramp-start-radio-v2-imported-row-supersession"
}

enum class V2ImportedRowCommitKind {
    ADDITION,
    SUPERSESSION,
}

data class V2ImportedPredecessorEvidence(
    val trackId: Long,
    val metadata: V2CommitTrackMetadata,
    val metadataSha256: String,
    val embeddingByteLength: Int,
    val embeddingSha256: String,
)

internal data class V2ObservedImportedPredecessorEvidence(
    val metadata: V2CommitTrackMetadata,
    val embeddingByteLength: Int,
    val embeddingSha256: String,
    val receiptCount: Long,
)

internal object V2ImportedRowPredecessorPolicy {
    fun requireExactUnreceipted(
        expected: V2ImportedPredecessorEvidence,
        observed: V2ObservedImportedPredecessorEvidence?,
        location: String,
    ) {
        require(observed != null && observed.receiptCount == 0L) {
            "authorized imported predecessor is missing, ambiguous, or receipt-bound in $location"
        }
        require(observed.metadata == expected.metadata &&
            V2CommitMetadataIdentity.sha256(observed.metadata) == expected.metadataSha256 &&
            observed.embeddingByteLength == expected.embeddingByteLength &&
            observed.embeddingSha256 == expected.embeddingSha256
        ) { "authorized imported predecessor fingerprint changed in $location" }
    }
}

data class V2ImportedRowWorkAuthorization(
    val workId: String,
    val powerampFileId: Long,
    val providerSpan: V2CommittedProviderSpan,
    val kind: V2ImportedRowCommitKind,
    val predecessor: V2ImportedPredecessorEvidence?,
)

data class V2ImportedRowSupersessionCommitAuthorization(
    val jobSpecId: String,
    val baseGenerationId: String,
    val baseManifestSha256: String,
    val baseDatabaseSha256: String,
    val privateBaseBindingId: String,
    val providerSnapshotGeneration: String,
    val predecessor: V2ImportedPredecessorEvidence,
)

/** Immutable preflight proof kept beside, but deliberately outside, the schema-v5 ledger. */
data class V2ImportedRowSupersessionAuthorization(
    val schemaVersion: Int,
    val jobId: String,
    val jobSpecId: String,
    val baseGenerationId: String,
    val baseManifestSha256: String,
    val baseDatabaseByteLength: Long,
    val baseDatabaseSha256: String,
    val baseDatabaseContentSha256: String,
    val privateBaseBindingId: String,
    val providerSnapshotGeneration: String,
    val works: List<V2ImportedRowWorkAuthorization>,
)

internal class V2ImportedRowAuthorizationException(message: String, cause: Throwable? = null) :
    IllegalStateException(message, cause)

internal object V2ImportedRowSupersessionAuthorizationPolicy {
    fun requireValid(
        authorization: V2ImportedRowSupersessionAuthorization,
        ledger: IndexingJobLedger? = null,
    ) {
        require(authorization.schemaVersion == V2ImportedRowSupersessionAuthorizationSchema.VERSION) {
            "unsupported imported-row authorization schema"
        }
        require(authorization.jobId.isNotBlank()) { "authorization job ID is blank" }
        require(authorization.jobSpecId.isNotBlank()) { "authorization job spec ID is blank" }
        require(authorization.baseGenerationId.isNotBlank()) {
            "authorization base generation is blank"
        }
        requireV2Sha256(authorization.baseManifestSha256, "base manifest SHA-256")
        require(authorization.baseDatabaseByteLength > 0L) { "base database length is invalid" }
        requireV2Sha256(authorization.baseDatabaseSha256, "base database SHA-256")
        requireV2Sha256(authorization.baseDatabaseContentSha256, "base content SHA-256")
        requireV2Sha256(authorization.privateBaseBindingId, "private base binding ID")
        require(authorization.privateBaseBindingId == V2JobPrivateDatabaseBindingIdentity.compute(
            jobId = authorization.jobId,
            jobSpecId = authorization.jobSpecId,
            baseGenerationId = authorization.baseGenerationId,
            sourceDatabaseByteLength = authorization.baseDatabaseByteLength,
            sourceDatabaseSha256 = authorization.baseDatabaseSha256,
            baseManifestSha256 = authorization.baseManifestSha256,
            baseDatabaseContentSha256 = authorization.baseDatabaseContentSha256,
        )) { "private base binding ID does not match authorization evidence" }
        require(authorization.providerSnapshotGeneration.isNotBlank()) {
            "authorization provider generation is blank"
        }
        require(authorization.works.isNotEmpty()) { "authorization has no logical CUE work" }
        require(authorization.works.mapTo(hashSetOf()) { it.workId }.size ==
            authorization.works.size
        ) { "authorization repeats a work ID" }
        require(authorization.works.mapTo(hashSetOf()) { it.powerampFileId }.size ==
            authorization.works.size
        ) { "authorization repeats a Poweramp row" }
        require(authorization.works.mapTo(hashSetOf()) { it.providerSpan }.size ==
            authorization.works.size
        ) { "authorization repeats a provider span" }
        val predecessorIds = authorization.works.mapNotNull { it.predecessor?.trackId }
        require(predecessorIds.toSet().size == predecessorIds.size) {
            "authorization reuses an imported predecessor"
        }
        authorization.works.forEach { work ->
            require(work.workId.isNotBlank() && work.powerampFileId > 0L) {
                "authorization has invalid work identity"
            }
            require(work.providerSpan.offsetMs >= 0L && work.providerSpan.durationMs >= 0L) {
                "authorization has invalid provider span"
            }
            when (work.kind) {
                V2ImportedRowCommitKind.ADDITION -> require(work.predecessor == null) {
                    "addition authorization carries a predecessor"
                }

                V2ImportedRowCommitKind.SUPERSESSION -> {
                    val predecessor = requireNotNull(work.predecessor) {
                        "supersession authorization has no predecessor"
                    }
                    require(predecessor.trackId > 0L) { "predecessor track ID is invalid" }
                    require(predecessor.metadataSha256 ==
                        V2CommitMetadataIdentity.sha256(predecessor.metadata)
                    ) { "predecessor metadata fingerprint is stale" }
                    require(predecessor.embeddingByteLength == V2_CLAMP3_BLOB_BYTES) {
                        "predecessor embedding length is invalid"
                    }
                    requireV2Sha256(predecessor.embeddingSha256, "predecessor embedding SHA-256")
                }
            }
        }
        if (ledger != null) {
            V2IndexingLedgerValidator.requireValid(ledger)
            val preflightSpecId = V2DecodedEosLineage.requirePreflightSpecId(ledger.jobSpec)
            require(authorization.jobId == ledger.jobSpec.jobId &&
                authorization.jobSpecId == preflightSpecId &&
                authorization.baseGenerationId == ledger.jobSpec.baseGenerationId &&
                authorization.providerSnapshotGeneration ==
                    ledger.jobSpec.providerSnapshot.libraryGeneration
            ) { "imported-row authorization is not bound to this immutable job" }
            val descriptors = ledger.jobSpec.tracks.associateBy(SelectedTrackDescriptor::workId)
            require(authorization.works.all { work ->
                descriptors[work.workId]?.let { descriptor ->
                    descriptor.powerampFileId == work.powerampFileId &&
                        descriptor.finalizedAudioSpan.kind == V2ResolvedAudioSpanKind.LOGICAL_CUE &&
                        descriptor.committedProviderSpan() == work.providerSpan
                } == true
            }) { "imported-row authorization does not match its logical CUE descriptors" }
            require(authorization.works.mapTo(hashSetOf()) { it.workId } ==
                descriptors.values.filter {
                    it.finalizedAudioSpan.kind == V2ResolvedAudioSpanKind.LOGICAL_CUE
                }.mapTo(hashSetOf()) { it.workId }
            ) { "imported-row authorization does not exhaust logical CUE work" }
        }
    }
}

interface V2ImportedRowAuthorizationAtomicIo {
    fun read(file: File): ByteArray
    fun write(file: File, bytes: ByteArray)
}

internal object AndroidV2ImportedRowAuthorizationAtomicIo :
    V2ImportedRowAuthorizationAtomicIo {
    override fun read(file: File): ByteArray =
        AtomicFile(file).openRead().use { input -> input.readBytes() }

    override fun write(file: File, bytes: ByteArray) {
        val atomic = AtomicFile(file)
        val stream = atomic.startWrite()
        try {
            stream.write(bytes)
            atomic.finishWrite(stream)
        } catch (error: Throwable) {
            atomic.failWrite(stream)
            throw error
        }
    }
}

internal enum class V2ImportedRowAuthorizationFileKind {
    CURRENT,
    LEGACY,
}

internal data class V2ImportedRowAuthorizationFileSelection(
    val kind: V2ImportedRowAuthorizationFileKind,
    val file: File,
)

/** Owns the non-ledger sidecar namespace and the one legacy JSON name shipped by early V2. */
internal object V2ImportedRowAuthorizationFileNamespace {
    const val CURRENT_SUFFIX = ".imported-row-supersession-v1.auth"
    const val LEGACY_SUFFIX = ".imported-row-supersession-v1.json"
    const val LEGACY_LEDGER_JOB_ID_SUFFIX = ".imported-row-supersession-v1"
    private val SAFE_JOB_ID = Regex("^[A-Za-z0-9._-]{1,128}$")

    fun currentFile(root: File, jobId: String): File {
        requireSafeJobId(jobId)
        return File(root, "$jobId$CURRENT_SUFFIX")
    }

    fun legacyFile(root: File, jobId: String): File {
        requireSafeJobId(jobId)
        return File(root, "$jobId$LEGACY_SUFFIX")
    }

    fun resolveExisting(root: File, jobId: String): V2ImportedRowAuthorizationFileSelection? {
        val current = currentFile(root, jobId)
        val legacy = legacyFile(root, jobId)
        listOf(current, legacy).forEach(::requireNoAtomicResidue)
        val candidates = buildList {
            if (current.exists()) {
                add(
                    V2ImportedRowAuthorizationFileSelection(
                        V2ImportedRowAuthorizationFileKind.CURRENT,
                        current,
                    ),
                )
            }
            if (legacy.exists()) {
                add(
                    V2ImportedRowAuthorizationFileSelection(
                        V2ImportedRowAuthorizationFileKind.LEGACY,
                        legacy,
                    ),
                )
            }
        }
        if (candidates.size > 1) {
            throw V2ImportedRowAuthorizationException(
                "Conflicting current and legacy imported-row authorizations exist; " +
                    "start a new indexing preflight",
            )
        }
        return candidates.singleOrNull()
    }

    fun isSidecarOrResidue(name: String): Boolean =
        listOf(CURRENT_SUFFIX, LEGACY_SUFFIX).any { suffix ->
            name.endsWith(suffix) || name.endsWith("$suffix.new") ||
                name.endsWith("$suffix.bak")
        }

    private fun requireSafeJobId(jobId: String) {
        require(SAFE_JOB_ID.matches(jobId)) { "unsafe job ID" }
        if (jobId.endsWith(LEGACY_LEDGER_JOB_ID_SUFFIX)) {
            throw V2ImportedRowAuthorizationException(
                "Job ID collides with the legacy imported-row authorization namespace",
            )
        }
    }

    private fun requireNoAtomicResidue(file: File) {
        val residue = listOf(File(file.path + ".new"), File(file.path + ".bak"))
            .firstOrNull(File::exists)
        if (residue != null) {
            throw V2ImportedRowAuthorizationException(
                "Imported CUE authorization has incomplete atomic state; start a new preflight",
            )
        }
    }
}

/** Atomic sidecar store. It is published before its ledger, so a crash can only orphan the sidecar. */
class V2ImportedRowSupersessionAuthorizationStore(
    ledgerDirectory: File,
    private val gson: Gson = GsonBuilder().disableHtmlEscaping().create(),
    private val atomicIo: V2ImportedRowAuthorizationAtomicIo =
        AndroidV2ImportedRowAuthorizationAtomicIo,
) {
    private val root = ledgerDirectory.canonicalFile

    @Synchronized
    fun createOrRequireExact(authorization: V2ImportedRowSupersessionAuthorization) {
        V2ImportedRowSupersessionAuthorizationPolicy.requireValid(authorization)
        require(root.isDirectory || root.mkdirs()) { "cannot create indexing ledger directory" }
        val existingFile = V2ImportedRowAuthorizationFileNamespace.resolveExisting(
            root,
            authorization.jobId,
        )
        if (existingFile != null) {
            val existing = read(existingFile.file)
            if (existing != authorization) {
                throw V2ImportedRowAuthorizationException(
                    "A different imported-row authorization already exists; start a new preflight",
                )
            }
            return
        }
        val target = V2ImportedRowAuthorizationFileNamespace.currentFile(
            root,
            authorization.jobId,
        )
        try {
            val bytes = gson.toJson(
                Envelope(
                    format = V2ImportedRowSupersessionAuthorizationSchema.FORMAT,
                    schemaVersion = V2ImportedRowSupersessionAuthorizationSchema.VERSION,
                    authorization = authorization,
                ),
            ).toByteArray(StandardCharsets.UTF_8)
            atomicIo.write(target, bytes)
        } catch (error: Throwable) {
            throw IOException("unable to persist imported-row authorization", error)
        }
        val published = checkNotNull(
            V2ImportedRowAuthorizationFileNamespace.resolveExisting(root, authorization.jobId),
        ) { "imported-row authorization disappeared after write" }
        check(published.kind == V2ImportedRowAuthorizationFileKind.CURRENT) {
            "imported-row authorization was published under the wrong namespace"
        }
        check(read(published.file) == authorization) {
            "imported-row authorization changed after write"
        }
    }

    @Synchronized
    fun requireFor(ledger: IndexingJobLedger): V2ImportedRowSupersessionAuthorization {
        val target = V2ImportedRowAuthorizationFileNamespace.resolveExisting(
            root,
            ledger.jobSpec.jobId,
        )
        if (target == null || !target.file.isFile) {
            throw V2ImportedRowAuthorizationException(
                "Imported CUE repair authorization is missing; start a new indexing preflight",
            )
        }
        return read(target.file).also {
            V2ImportedRowSupersessionAuthorizationPolicy.requireValid(it, ledger)
        }
    }

    @Synchronized
    fun requireAbsent(jobId: String) {
        val target = V2ImportedRowAuthorizationFileNamespace.resolveExisting(root, jobId)
        if (target != null) {
            throw V2ImportedRowAuthorizationException(
                "Unexpected imported-row authorization exists; start a new indexing preflight",
            )
        }
    }

    private fun read(file: File): V2ImportedRowSupersessionAuthorization = try {
        val bytes = atomicIo.read(file)
        val envelope = gson.fromJson(
            bytes.toString(StandardCharsets.UTF_8),
            Envelope::class.java,
        ) ?: throw IOException("empty imported-row authorization")
        if (envelope.format != V2ImportedRowSupersessionAuthorizationSchema.FORMAT ||
            envelope.schemaVersion != V2ImportedRowSupersessionAuthorizationSchema.VERSION ||
            envelope.authorization.schemaVersion != envelope.schemaVersion
        ) throw IOException("unsupported imported-row authorization envelope")
        envelope.authorization.also(V2ImportedRowSupersessionAuthorizationPolicy::requireValid)
    } catch (error: V2ImportedRowAuthorizationException) {
        throw error
    } catch (error: Exception) {
        throw V2ImportedRowAuthorizationException(
            "Imported CUE repair authorization is unreadable; start a new indexing preflight",
            error,
        )
    }

    private data class Envelope(
        val format: String,
        val schemaVersion: Int,
        val authorization: V2ImportedRowSupersessionAuthorization,
    )
}

/** Recomputes destructive intent from one exact provider snapshot and immutable active base. */
internal object V2ImportedRowSupersessionAuthorizer {
    fun authorize(
        ledger: IndexingJobLedger,
        activeBase: V2ResolvedActiveIndexGeneration,
        providerSnapshot: V2ProviderPathGroupSnapshot,
    ): V2ImportedRowSupersessionAuthorization? {
        V2IndexingLedgerValidator.requireValid(ledger)
        val cueDescriptors = ledger.jobSpec.tracks.filter {
            it.finalizedAudioSpan.kind == V2ResolvedAudioSpanKind.LOGICAL_CUE
        }
        if (cueDescriptors.isEmpty()) return null
        val acquisition = providerSnapshot.acquisitionEvidence
            ?: fail("Poweramp snapshot has no completion evidence")
        val providerRows = providerSnapshot.groups.flatMap { it.rows }
        if (!acquisition.cursorExhaustedNormally || acquisition.rowCount != providerRows.size ||
            providerRows.mapTo(hashSetOf()) { it.powerampFileId }.size != providerRows.size
        ) fail("Poweramp snapshot is incomplete or ambiguous")
        if (providerSnapshot.groups.mapTo(hashSetOf()) { it.physicalPath }.size !=
            providerSnapshot.groups.size
        ) fail("Poweramp snapshot repeats a physical path group")
        val providerGeneration = providerSnapshot.libraryGeneration
            ?: fail("Poweramp snapshot has no stable library generation")
        if (providerGeneration != ledger.jobSpec.providerSnapshot.libraryGeneration) {
            fail("Poweramp library changed; start a new indexing preflight")
        }
        val baseGenerationId = ledger.jobSpec.baseGenerationId
            ?: fail("Imported CUE work has no immutable base generation")
        if (activeBase.manifest.generationId != baseGenerationId) {
            fail("The active embedding generation changed; start a new indexing preflight")
        }
        val baseRows = readBaseRows(activeBase.databaseFile)
        val receiptedIds = baseRows.filter(V2ImportedBaseRow::hasV2Receipt)
            .mapTo(hashSetOf()) { it.trackId }
        val receiptSnapshot = V2ProviderSpanReceiptReader.read(activeBase.databaseFile).also { snapshot ->
            if (snapshot.invalidReceiptCount != 0 ||
                (!snapshot.compatibleSchema && receiptedIds.isNotEmpty())
            ) fail("The active generation has unreadable V2 receipt evidence")
        }
        val receiptSpans = receiptSnapshot.receipts.mapTo(hashSetOf()) { it.providerSpan }
        if (receiptSpans.size != receiptSnapshot.receipts.size) {
            fail("The active generation repeats a committed provider span")
        }
        val groupsByPath = providerSnapshot.groups.associateBy { it.physicalPath }
        val candidates = providerRows.mapNotNull { row ->
            val path = TrackNormalization.normalizePath(row.providerPhysicalPath)
                ?: fail("Poweramp row ${row.powerampFileId} has no normalized source path")
            val span = V2CommittedProviderSpan(
                normalizedPhysicalPath = V2StableProviderLexicalPathNormalizer.normalizeAbsolute(path),
                offsetMs = row.offsetMs,
                durationMs = V2ProviderDurationEvidencePolicy.canonicalMs(row.durationMs),
            )
            if (span in receiptSpans) return@mapNotNull null
            val group = groupsByPath[row.physicalPath]
                ?: fail("Poweramp snapshot omitted path group ${row.physicalPath}")
            if (group.completeness != V2ProviderPathGroupCompleteness.COMPLETE) {
                fail("Poweramp path group ${row.physicalPath} is incomplete")
            }
            val artist = TrackNormalization.normalizeArtist(row.artist)
            val album = TrackNormalization.normalizeAlbum(row.album)
            val title = TrackNormalization.normalizeTitle(row.title)
            val durationMs = Math.toIntExact(
                V2ProviderDurationEvidencePolicy.canonicalMs(row.durationMs),
            )
            val requiresRepair = group.rows.any {
                it.offsetMs > 0L || it.cueSourceImageFolderId != null
            }
            V2LegacyProviderCandidate(
                powerampFileId = row.powerampFileId,
                normalizedPhysicalPath = span.normalizedPhysicalPath,
                offsetMs = row.offsetMs,
                durationMs = durationMs,
                metadataKey = TrackNormalization.buildMetadataKey(
                    artist,
                    album,
                    title,
                    durationMs,
                ),
                compatibilityEligible = !requiresRepair && row.cueSourceImageFolderId == null,
                requiresSpanSpecificRebuild = requiresRepair,
            )
        }
        if (candidates.mapTo(hashSetOf()) {
                Triple(it.normalizedPhysicalPath, it.offsetMs, it.durationMs)
            }.size != candidates.size
        ) fail("Poweramp snapshot repeats an uncommitted provider span")
        val legacyRows = baseRows.filterNot(V2ImportedBaseRow::hasV2Receipt)
        val legacyCandidates = legacyRows.map { row ->
            val metadata = row.metadata
            val artist = TrackNormalization.normalizeArtist(metadata.artist)
            val album = TrackNormalization.normalizeAlbum(metadata.album)
            val title = TrackNormalization.normalizeTitle(metadata.title)
            V2LegacyDatabaseCandidate(
                trackId = row.trackId,
                normalizedPath = TrackNormalization.normalizePath(metadata.filePath),
                durationMs = metadata.durationMs,
                metadataKey = TrackNormalization.buildMetadataKey(
                    artist,
                    album,
                    title,
                    metadata.durationMs,
                ),
            )
        }
        val compatibility = V2LegacyCompatibilityResolver.resolve(candidates, legacyCandidates)
        val cueWorks = cueDescriptors.map { descriptor ->
            V2SelectedImportedRowWork(
                workId = descriptor.workId,
                powerampFileId = descriptor.powerampFileId,
                providerSpan = descriptor.committedProviderSpan(),
            )
        }
        val cueIds = cueWorks.mapTo(hashSetOf()) { it.powerampFileId }
        val selectedBindings = compatibility.repairBindings.filter { it.powerampFileId in cueIds }
        val plan = V2ImportedRowSupersessionPlanner.plan(cueWorks, selectedBindings)
        if (plan.blockedAmbiguousMappings.isNotEmpty()) {
            fail("Imported CUE predecessor mapping is ambiguous; start a new preflight")
        }
        val bindingsByProvider = selectedBindings.associateBy(V2LegacyCompatibilityBinding::powerampFileId)
        val providerCandidatesById = candidates.associateBy(V2LegacyProviderCandidate::powerampFileId)
        val legacyCandidateById = legacyCandidates.associateBy(V2LegacyDatabaseCandidate::trackId)
        val legacyRowsById = legacyRows.associateBy(V2ImportedBaseRow::trackId)
        val workAuthorizations = cueWorks.sortedBy(V2SelectedImportedRowWork::workId).map { work ->
            val provider = providerCandidatesById[work.powerampFileId]
                ?: fail("Selected logical CUE row is already represented or missing")
            val binding = bindingsByProvider[work.powerampFileId]
            if (binding == null) {
                val possible = legacyCandidates.filter { candidate ->
                    possibleCueRepair(provider, candidate)
                }
                if (possible.isNotEmpty()) {
                    fail("Imported CUE predecessor evidence is ambiguous or stale")
                }
                V2ImportedRowWorkAuthorization(
                    workId = work.workId,
                    powerampFileId = work.powerampFileId,
                    providerSpan = work.providerSpan,
                    kind = V2ImportedRowCommitKind.ADDITION,
                    predecessor = null,
                )
            } else {
                if (binding.evidence != V2LegacyCompatibilityEvidence.CUE_LOGICAL_METADATA_REPAIR ||
                    legacyCandidateById[binding.trackId] == null
                ) fail("Imported CUE predecessor evidence is not exact")
                val predecessor = legacyRowsById[binding.trackId]
                    ?: fail("Imported CUE predecessor disappeared")
                V2ImportedRowWorkAuthorization(
                    workId = work.workId,
                    powerampFileId = work.powerampFileId,
                    providerSpan = work.providerSpan,
                    kind = V2ImportedRowCommitKind.SUPERSESSION,
                    predecessor = predecessor.toEvidence(),
                )
            }
        }
        val preflightSpecId = V2DecodedEosLineage.requirePreflightSpecId(ledger.jobSpec)
        val authorization = V2ImportedRowSupersessionAuthorization(
            schemaVersion = V2ImportedRowSupersessionAuthorizationSchema.VERSION,
            jobId = ledger.jobSpec.jobId,
            jobSpecId = preflightSpecId,
            baseGenerationId = baseGenerationId,
            baseManifestSha256 = activeBase.manifestSha256,
            baseDatabaseByteLength = activeBase.manifest.databaseByteLength,
            baseDatabaseSha256 = activeBase.manifest.databaseSha256,
            baseDatabaseContentSha256 = activeBase.manifest.databaseContentSha256,
            privateBaseBindingId = V2JobPrivateDatabaseBindingIdentity.compute(
                jobId = ledger.jobSpec.jobId,
                jobSpecId = preflightSpecId,
                baseGenerationId = baseGenerationId,
                sourceDatabaseByteLength = activeBase.manifest.databaseByteLength,
                sourceDatabaseSha256 = activeBase.manifest.databaseSha256,
                baseManifestSha256 = activeBase.manifestSha256,
                baseDatabaseContentSha256 = activeBase.manifest.databaseContentSha256,
            ),
            providerSnapshotGeneration = providerGeneration,
            works = workAuthorizations,
        )
        return authorization.also {
            V2ImportedRowSupersessionAuthorizationPolicy.requireValid(it, ledger)
        }
    }

    private fun readBaseRows(file: File): List<V2ImportedBaseRow> =
        SQLiteDatabase.openDatabase(file.canonicalPath, null, SQLiteDatabase.OPEN_READONLY).use { db ->
            val receiptTable = V2EmbeddingCommitRepository.RECEIPT_TABLE
            val hasReceipts = db.rawQuery(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1",
                arrayOf(receiptTable),
            ).use { it.moveToFirst() }
            val receiptJoin = if (hasReceipts) {
                "LEFT JOIN $receiptTable r ON r.track_id = t.id"
            } else {
                "LEFT JOIN tracks r ON 0"
            }
            db.rawQuery(
                """
                SELECT t.id, t.metadata_key, t.filename_key, t.artist, t.album, t.title,
                       t.duration_ms, t.file_path, t.source, e.embedding,
                       ${if (hasReceipts) "r.track_id" else "NULL"}
                FROM ${V2EmbeddingCommitRepository.TRACK_TABLE} t
                JOIN ${V2EmbeddingCommitRepository.EMBEDDING_TABLE} e ON e.track_id = t.id
                $receiptJoin
                ORDER BY t.id
                """.trimIndent(),
                null,
            ).use { cursor ->
                buildList {
                    while (cursor.moveToNext()) {
                        val blob = cursor.getBlob(9)
                        V2Clamp3VectorCodec.requireValidBlob(blob)
                        add(
                            V2ImportedBaseRow(
                                trackId = cursor.getLong(0),
                                metadata = V2CommitTrackMetadata(
                                    metadataKey = cursor.getString(1),
                                    filenameKey = cursor.getString(2),
                                    artist = if (cursor.isNull(3)) null else cursor.getString(3),
                                    album = if (cursor.isNull(4)) null else cursor.getString(4),
                                    title = if (cursor.isNull(5)) null else cursor.getString(5),
                                    durationMs = cursor.getInt(6),
                                    filePath = cursor.getString(7),
                                    source = if (cursor.isNull(8)) {
                                        fail("Imported base row ${cursor.getLong(0)} has no source metadata")
                                    } else {
                                        cursor.getString(8)
                                    },
                                ),
                                embeddingByteLength = blob.size,
                                embeddingSha256 = V2ArtifactDigests.sha256(blob),
                                hasV2Receipt = !cursor.isNull(10),
                            ),
                        )
                    }
                }
            }
        }

    private fun possibleCueRepair(
        provider: V2LegacyProviderCandidate,
        database: V2LegacyDatabaseCandidate,
    ): Boolean {
        if (!provider.requiresSpanSpecificRebuild || provider.durationMs <= 0 ||
            database.durationMs <= 0 ||
            kotlin.math.abs(provider.durationMs - database.durationMs) > 5_000
        ) return false
        val providerPath = V2LegacyCompatibilityResolver.strictMusicRelativePath(
            provider.normalizedPhysicalPath,
        )
        val databasePath = database.normalizedPath?.let(
            V2LegacyCompatibilityResolver::strictMusicRelativePath,
        )
        return providerPath != null && providerPath == databasePath &&
            provider.metadataKey.substringBeforeLast('|', "") ==
            database.metadataKey.substringBeforeLast('|', "")
    }

    private fun fail(message: String): Nothing = throw V2ImportedRowAuthorizationException(message)

    private data class V2ImportedBaseRow(
        val trackId: Long,
        val metadata: V2CommitTrackMetadata,
        val embeddingByteLength: Int,
        val embeddingSha256: String,
        val hasV2Receipt: Boolean,
    ) {
        fun toEvidence(): V2ImportedPredecessorEvidence = V2ImportedPredecessorEvidence(
            trackId = trackId,
            metadata = metadata,
            metadataSha256 = V2CommitMetadataIdentity.sha256(metadata),
            embeddingByteLength = embeddingByteLength,
            embeddingSha256 = embeddingSha256,
        )
    }
}

internal fun SelectedTrackDescriptor.committedProviderSpan(): V2CommittedProviderSpan =
    V2CommittedProviderSpan(
        normalizedPhysicalPath = providerRow.physicalPath,
        offsetMs = providerOffsetMs,
        durationMs = V2ProviderDurationEvidencePolicy.canonicalMs(providerDurationMs),
    )

internal fun V2ImportedRowSupersessionAuthorization.commitAuthorizationFor(
    workId: String,
    ledger: IndexingJobLedger,
): V2ImportedRowSupersessionCommitAuthorization? {
    val executionBindingId = executionPrivateBaseBindingId(ledger)
    val work = works.singleOrNull { it.workId == workId }
        ?: throw V2ImportedRowAuthorizationException(
            "Logical CUE work is absent from its authorization; start a new preflight",
        )
    return when (work.kind) {
        V2ImportedRowCommitKind.ADDITION -> null
        V2ImportedRowCommitKind.SUPERSESSION -> V2ImportedRowSupersessionCommitAuthorization(
            jobSpecId = ledger.jobSpec.specId,
            baseGenerationId = baseGenerationId,
            baseManifestSha256 = baseManifestSha256,
            baseDatabaseSha256 = baseDatabaseSha256,
            privateBaseBindingId = executionBindingId,
            providerSnapshotGeneration = providerSnapshotGeneration,
            predecessor = requireNotNull(work.predecessor),
        )
    }
}

/** Binds the authorized base bytes to the exact post-EOS spec without rewriting preflight proof. */
internal fun V2ImportedRowSupersessionAuthorization.executionPrivateBaseBindingId(
    ledger: IndexingJobLedger,
): String {
    V2ImportedRowSupersessionAuthorizationPolicy.requireValid(this, ledger)
    return V2JobPrivateDatabaseBindingIdentity.compute(
        jobId = jobId,
        jobSpecId = ledger.jobSpec.specId,
        baseGenerationId = baseGenerationId,
        sourceDatabaseByteLength = baseDatabaseByteLength,
        sourceDatabaseSha256 = baseDatabaseSha256,
        baseManifestSha256 = baseManifestSha256,
        baseDatabaseContentSha256 = baseDatabaseContentSha256,
    )
}

internal data class V2ImportedRowActivationDisposition(
    val committedSupersessions: List<V2ImportedRowWorkAuthorization>,
    val uncommittedSupersessions: List<V2ImportedRowWorkAuthorization>,
)

/** Separates destructive work by its durable commit receipt, never by a transient job state. */
internal object V2ImportedRowActivationPolicy {
    fun partition(
        ledger: IndexingJobLedger,
        authorization: V2ImportedRowSupersessionAuthorization,
    ): V2ImportedRowActivationDisposition {
        V2ImportedRowSupersessionAuthorizationPolicy.requireValid(authorization, ledger)
        val tracks = ledger.tracks.associateBy(IndexingTrackLedger::workId)
        val authorizedWorkIds = authorization.works.mapTo(hashSetOf()) { it.workId }
        val committedWorkIds = tracks.values.filter { track ->
            track.workId in authorizedWorkIds &&
                track.checkpoint == TrackCheckpoint.COMMITTED &&
                track.verifiedArtifacts.count {
                    it.kind == VerifiedArtifactKind.DATABASE_COMMIT
                } == 1
        }.mapTo(hashSetOf(), IndexingTrackLedger::workId)
        return partition(authorization, committedWorkIds)
    }

    fun partition(
        authorization: V2ImportedRowSupersessionAuthorization,
        committedWorkIds: Set<String>,
    ): V2ImportedRowActivationDisposition {
        V2ImportedRowSupersessionAuthorizationPolicy.requireValid(authorization)
        require(committedWorkIds.all { committed ->
            authorization.works.any { it.workId == committed }
        }) { "durable commit evidence names unauthorized imported work" }
        val (committed, uncommitted) = authorization.works
            .filter { it.kind == V2ImportedRowCommitKind.SUPERSESSION }
            .partition { it.workId in committedWorkIds }
        return V2ImportedRowActivationDisposition(
            committedSupersessions = committed,
            uncommittedSupersessions = uncommitted,
        )
    }
}

internal object V2JobPrivateDatabaseBindingIdentity {
    fun compute(
        jobId: String,
        jobSpecId: String,
        baseGenerationId: String?,
        sourceDatabaseByteLength: Long,
        sourceDatabaseSha256: String,
        baseManifestSha256: String?,
        baseDatabaseContentSha256: String?,
    ): String {
        val digest = MessageDigest.getInstance("SHA-256")
        fun put(value: String?) {
            val bytes = value?.toByteArray(StandardCharsets.UTF_8)
            digest.update(if (bytes == null) 0 else 1)
            if (bytes != null) {
                digest.update((bytes.size ushr 24).toByte())
                digest.update((bytes.size ushr 16).toByte())
                digest.update((bytes.size ushr 8).toByte())
                digest.update(bytes.size.toByte())
                digest.update(bytes)
            }
        }
        put("v2-job-private-base-binding-v2")
        put(jobId)
        put(jobSpecId)
        put(baseGenerationId)
        for (shift in 56 downTo 0 step 8) {
            digest.update((sourceDatabaseByteLength ushr shift).toByte())
        }
        put(sourceDatabaseSha256)
        put(baseManifestSha256)
        put(baseDatabaseContentSha256)
        return digest.digest().toV2CommitHex()
    }
}
