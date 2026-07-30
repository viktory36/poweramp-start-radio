package com.powerampstartradio.indexing.v2

import android.content.ContentValues
import android.database.Cursor
import android.database.sqlite.SQLiteDatabase

data class V2EmbeddingCommitRequest(
    val workId: String,
    val stableTrackSpanIdentity: StableTrackSpanIdentity,
    val embeddingSpecId: String,
    val providerSpan: V2CommittedProviderSpan,
    val metadata: V2CommitTrackMetadata,
    val sourceFingerprint: SourceFingerprint,
    val clampVectorArtifact: VerifiedArtifact,
    val verifiedAtEpochMs: Long,
    val importedRowSupersession: V2ImportedRowSupersessionCommitAuthorization? = null,
)

enum class V2EmbeddingCommitDisposition {
    INSERTED,
    REUSED,
}

data class V2EmbeddingCommitResult(
    val disposition: V2EmbeddingCommitDisposition,
    val trackId: Long,
    val receiptKey: String,
    val databaseArtifact: VerifiedArtifact,
)

/**
 * Crash-idempotent terminal commit for one V2 work item.
 *
 * The track row, exact 768d blob, and `(work_id, embedding_spec_id)` receipt are inserted in one
 * SQLite transaction. A replay verifies all three pieces of committed evidence and reuses the row.
 */
class V2EmbeddingCommitRepository(
    private val database: SQLiteDatabase,
) {
    fun commit(
        request: V2EmbeddingCommitRequest,
        clampVectorBlob: ByteArray,
    ): V2EmbeddingCommitResult {
        require(!database.isReadOnly) { "embedding database is read-only" }
        require(database.isOpen) { "embedding database is closed" }
        val expected = V2EmbeddingCommitReconciler.expectation(
            workId = request.workId,
            stableTrackSpanIdentity = request.stableTrackSpanIdentity,
            embeddingSpecId = request.embeddingSpecId,
            providerSpan = request.providerSpan,
            metadata = request.metadata,
            embeddingBlob = clampVectorBlob,
        )
        requireValidClampArtifact(request, expected)

        var disposition: V2EmbeddingCommitDisposition? = null
        var committedTrackId: Long? = null
        database.beginTransaction()
        try {
            ensureReceiptSchema()
            if (request.importedRowSupersession != null) {
                ensureImportedRowSupersessionSchema()
            }
            val existingSupersession = readImportedRowSupersession(
                request.workId,
                request.embeddingSpecId,
            )
            when (val decision = V2EmbeddingCommitReconciler.decide(
                expected,
                readEvidence(request.workId, request.embeddingSpecId),
            )) {
                V2EmbeddingCommitDecision.InsertNew -> {
                    if (existingSupersession != null) {
                        throw V2ImportedRowSupersessionIntegrityException(
                            "supersession audit exists without its replacement receipt",
                        )
                    }
                    request.importedRowSupersession?.let(::requirePredecessorMatches)
                    val trackId = insertTrack(request.metadata)
                    insertEmbedding(trackId, clampVectorBlob)
                    insertReceipt(expected, trackId, request.verifiedAtEpochMs)
                    request.importedRowSupersession?.let { authorization ->
                        deleteExactImportedPredecessor(authorization.predecessor)
                        insertImportedRowSupersession(
                            request = request,
                            authorization = authorization,
                            replacementTrackId = trackId,
                        )
                    }
                    requireReusableAfterWrite(expected, trackId)
                    requireImportedRowSupersessionReplay(
                        request = request,
                        replacementTrackId = trackId,
                    )
                    disposition = V2EmbeddingCommitDisposition.INSERTED
                    committedTrackId = trackId
                }

                is V2EmbeddingCommitDecision.Reuse -> {
                    requireImportedRowSupersessionReplay(
                        request = request,
                        replacementTrackId = decision.trackId,
                    )
                    disposition = V2EmbeddingCommitDisposition.REUSED
                    committedTrackId = decision.trackId
                }

                is V2EmbeddingCommitDecision.Conflict -> {
                    throw V2EmbeddingCommitIntegrityException(decision.reasons)
                }
            }
            database.setTransactionSuccessful()
        } finally {
            database.endTransaction()
        }

        val trackId = checkNotNull(committedTrackId)
        val finalDisposition = checkNotNull(disposition)
        val receiptKey = receiptKey(request.workId, request.embeddingSpecId)
        return V2EmbeddingCommitResult(
            disposition = finalDisposition,
            trackId = trackId,
            receiptKey = receiptKey,
            databaseArtifact = VerifiedArtifact(
                kind = VerifiedArtifactKind.DATABASE_COMMIT,
                storageKey = "sqlite:$EMBEDDING_TABLE:track:$trackId:$receiptKey",
                byteLength = expected.embeddingByteLength.toLong(),
                sha256 = expected.embeddingSha256,
                completedUnits = 1,
                plannedUnits = 1,
                embeddingSpecId = request.embeddingSpecId,
                sourceFingerprint = request.sourceFingerprint,
                verifiedAtEpochMs = request.verifiedAtEpochMs,
            ),
        )
    }

    private fun requireValidClampArtifact(
        request: V2EmbeddingCommitRequest,
        expected: V2EmbeddingCommitExpectation,
    ) {
        val artifact = request.clampVectorArtifact
        require(request.verifiedAtEpochMs >= 0L) { "verifiedAtEpochMs must not be negative" }
        require(artifact.kind == VerifiedArtifactKind.CLAMP_VECTOR) {
            "commit requires a verified CLaMP vector artifact"
        }
        require(artifact.storageKey.isNotBlank()) { "CLaMP artifact storage key is blank" }
        require(artifact.embeddingSpecId == request.embeddingSpecId) {
            "CLaMP artifact embedding spec mismatch"
        }
        require(artifact.sourceFingerprint == request.sourceFingerprint) {
            "CLaMP artifact source fingerprint mismatch"
        }
        require(artifact.byteLength == expected.embeddingByteLength.toLong()) {
            "CLaMP artifact byte length mismatch"
        }
        require(artifact.sha256 == expected.embeddingSha256) {
            "CLaMP artifact SHA-256 mismatch"
        }
        require(artifact.plannedUnits > 0 && artifact.completedUnits == artifact.plannedUnits) {
            "CLaMP artifact is incomplete"
        }
        require(request.verifiedAtEpochMs >= artifact.verifiedAtEpochMs) {
            "database verification predates the CLaMP artifact"
        }
    }

    private fun ensureReceiptSchema() {
        database.execSQL(
            """
            CREATE TABLE IF NOT EXISTS $RECEIPT_TABLE (
                receipt_schema_version INTEGER NOT NULL CHECK (receipt_schema_version = $RECEIPT_SCHEMA_VERSION),
                work_id TEXT NOT NULL,
                stable_track_span_id TEXT NOT NULL,
                stable_identity_spec_id TEXT NOT NULL,
                stable_identity_strength TEXT NOT NULL,
                embedding_spec_id TEXT NOT NULL,
                provider_physical_path TEXT NOT NULL,
                provider_offset_ms INTEGER NOT NULL CHECK (provider_offset_ms >= 0),
                provider_duration_ms INTEGER NOT NULL CHECK (provider_duration_ms >= 0),
                track_id INTEGER NOT NULL,
                metadata_sha256 TEXT NOT NULL,
                embedding_byte_length INTEGER NOT NULL CHECK (embedding_byte_length = $V2_CLAMP3_BLOB_BYTES),
                embedding_sha256 TEXT NOT NULL,
                committed_at_epoch_ms INTEGER NOT NULL CHECK (committed_at_epoch_ms >= 0),
                PRIMARY KEY (work_id, embedding_spec_id),
                FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE CASCADE
            )
            """.trimIndent(),
        )
        database.execSQL(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_${RECEIPT_TABLE}_track_id " +
                "ON $RECEIPT_TABLE(track_id)",
        )
        database.execSQL(
            "CREATE INDEX IF NOT EXISTS idx_${RECEIPT_TABLE}_stable_span " +
                "ON $RECEIPT_TABLE(stable_track_span_id, embedding_spec_id)",
        )
        database.execSQL(
            "CREATE INDEX IF NOT EXISTS idx_${RECEIPT_TABLE}_provider_span " +
                "ON $RECEIPT_TABLE(provider_physical_path, provider_offset_ms, provider_duration_ms)",
        )
    }

    private fun ensureImportedRowSupersessionSchema() {
        database.execSQL(
            """
            CREATE TABLE IF NOT EXISTS $IMPORTED_ROW_SUPERSESSION_TABLE (
                supersession_schema_version INTEGER NOT NULL
                    CHECK (supersession_schema_version = $IMPORTED_ROW_SUPERSESSION_SCHEMA_VERSION),
                work_id TEXT NOT NULL,
                embedding_spec_id TEXT NOT NULL,
                job_spec_id TEXT NOT NULL,
                base_generation_id TEXT NOT NULL,
                base_manifest_sha256 TEXT NOT NULL,
                base_database_sha256 TEXT NOT NULL,
                private_base_binding_id TEXT NOT NULL,
                provider_snapshot_generation TEXT NOT NULL,
                predecessor_track_id INTEGER NOT NULL,
                predecessor_metadata_sha256 TEXT NOT NULL,
                predecessor_embedding_byte_length INTEGER NOT NULL
                    CHECK (predecessor_embedding_byte_length = $V2_CLAMP3_BLOB_BYTES),
                predecessor_embedding_sha256 TEXT NOT NULL,
                replacement_track_id INTEGER NOT NULL,
                committed_at_epoch_ms INTEGER NOT NULL CHECK (committed_at_epoch_ms >= 0),
                PRIMARY KEY (work_id, embedding_spec_id),
                UNIQUE (predecessor_track_id),
                UNIQUE (replacement_track_id),
                FOREIGN KEY (replacement_track_id) REFERENCES tracks(id) ON DELETE CASCADE
            )
            """.trimIndent(),
        )
    }

    private fun requirePredecessorMatches(
        authorization: V2ImportedRowSupersessionCommitAuthorization,
    ) {
        val predecessor = authorization.predecessor
        val track = readTrack(predecessor.trackId)
        val embedding = readEmbedding(predecessor.trackId)
        val metadata = track?.metadata
        val receiptCount = database.rawQuery(
            "SELECT COUNT(*) FROM $RECEIPT_TABLE WHERE track_id = ?",
            arrayOf(predecessor.trackId.toString()),
        ).use { cursor -> check(cursor.moveToFirst()); cursor.getLong(0) }
        if (metadata != predecessor.metadata ||
            metadata?.let(V2CommitMetadataIdentity::sha256) != predecessor.metadataSha256 ||
            embedding?.byteLength != predecessor.embeddingByteLength ||
            embedding?.sha256 != predecessor.embeddingSha256 ||
            receiptCount != 0L
        ) {
            throw V2ImportedRowSupersessionIntegrityException(
                "imported predecessor is missing, stale, or already receipt-bound",
            )
        }
    }

    private fun deleteExactImportedPredecessor(predecessor: V2ImportedPredecessorEvidence) {
        val deletedEmbedding = database.delete(
            EMBEDDING_TABLE,
            "track_id = ?",
            arrayOf(predecessor.trackId.toString()),
        )
        if (deletedEmbedding != 1) {
            throw V2ImportedRowSupersessionIntegrityException(
                "exact imported predecessor embedding deletion affected $deletedEmbedding rows",
            )
        }
        val deletedTrack = database.delete(
            TRACK_TABLE,
            "id = ?",
            arrayOf(predecessor.trackId.toString()),
        )
        if (deletedTrack != 1) {
            throw V2ImportedRowSupersessionIntegrityException(
                "exact imported predecessor track deletion affected $deletedTrack rows",
            )
        }
    }

    private fun insertImportedRowSupersession(
        request: V2EmbeddingCommitRequest,
        authorization: V2ImportedRowSupersessionCommitAuthorization,
        replacementTrackId: Long,
    ) {
        val predecessor = authorization.predecessor
        val values = ContentValues().apply {
            put("supersession_schema_version", IMPORTED_ROW_SUPERSESSION_SCHEMA_VERSION)
            put("work_id", request.workId)
            put("embedding_spec_id", request.embeddingSpecId)
            put("job_spec_id", authorization.jobSpecId)
            put("base_generation_id", authorization.baseGenerationId)
            put("base_manifest_sha256", authorization.baseManifestSha256)
            put("base_database_sha256", authorization.baseDatabaseSha256)
            put("private_base_binding_id", authorization.privateBaseBindingId)
            put("provider_snapshot_generation", authorization.providerSnapshotGeneration)
            put("predecessor_track_id", predecessor.trackId)
            put("predecessor_metadata_sha256", predecessor.metadataSha256)
            put("predecessor_embedding_byte_length", predecessor.embeddingByteLength)
            put("predecessor_embedding_sha256", predecessor.embeddingSha256)
            put("replacement_track_id", replacementTrackId)
            put("committed_at_epoch_ms", request.verifiedAtEpochMs)
        }
        val inserted = database.insertWithOnConflict(
            IMPORTED_ROW_SUPERSESSION_TABLE,
            null,
            values,
            SQLiteDatabase.CONFLICT_ABORT,
        )
        if (inserted < 0L) {
            throw V2ImportedRowSupersessionIntegrityException("supersession audit insert failed")
        }
    }

    private fun requireImportedRowSupersessionReplay(
        request: V2EmbeddingCommitRequest,
        replacementTrackId: Long,
    ) {
        val actual = readImportedRowSupersession(request.workId, request.embeddingSpecId)
        val expected = request.importedRowSupersession
        if (expected == null) {
            if (actual != null) {
                throw V2ImportedRowSupersessionIntegrityException(
                    "supersession receipt cannot be reused without its authorization",
                )
            }
            return
        }
        val predecessor = expected.predecessor
        val replacementReceipt = readReceipt(request.workId, request.embeddingSpecId)
        if (actual == null ||
            actual.schemaVersion != IMPORTED_ROW_SUPERSESSION_SCHEMA_VERSION ||
            actual.jobSpecId != expected.jobSpecId ||
            actual.baseGenerationId != expected.baseGenerationId ||
            actual.baseManifestSha256 != expected.baseManifestSha256 ||
            actual.baseDatabaseSha256 != expected.baseDatabaseSha256 ||
            actual.privateBaseBindingId != expected.privateBaseBindingId ||
            actual.providerSnapshotGeneration != expected.providerSnapshotGeneration ||
            actual.predecessorTrackId != predecessor.trackId ||
            actual.predecessorMetadataSha256 != predecessor.metadataSha256 ||
            actual.predecessorEmbeddingByteLength != predecessor.embeddingByteLength ||
            actual.predecessorEmbeddingSha256 != predecessor.embeddingSha256 ||
            actual.replacementTrackId != replacementTrackId ||
            replacementReceipt?.trackId != replacementTrackId ||
            actual.committedAtEpochMs != replacementReceipt.committedAtEpochMs ||
            readTrack(predecessor.trackId) != null ||
            readEmbedding(predecessor.trackId) != null ||
            countReceiptsForTrack(predecessor.trackId) != 0L
        ) {
            throw V2ImportedRowSupersessionIntegrityException(
                "supersession replay evidence is missing or inconsistent",
            )
        }
    }

    private fun readImportedRowSupersession(
        workId: String,
        embeddingSpecId: String,
    ): ImportedRowSupersessionEvidence? {
        if (!hasTable(IMPORTED_ROW_SUPERSESSION_TABLE)) return null
        return database.rawQuery(
            """
            SELECT supersession_schema_version, job_spec_id, base_generation_id,
                   base_manifest_sha256, base_database_sha256, private_base_binding_id,
                   provider_snapshot_generation, predecessor_track_id,
                   predecessor_metadata_sha256, predecessor_embedding_byte_length,
                   predecessor_embedding_sha256, replacement_track_id, committed_at_epoch_ms
            FROM $IMPORTED_ROW_SUPERSESSION_TABLE
            WHERE work_id = ? AND embedding_spec_id = ?
            """.trimIndent(),
            arrayOf(workId, embeddingSpecId),
        ).use { cursor ->
            if (!cursor.moveToFirst()) return@use null
            ImportedRowSupersessionEvidence(
                schemaVersion = cursor.getInt(0),
                jobSpecId = cursor.getString(1),
                baseGenerationId = cursor.getString(2),
                baseManifestSha256 = cursor.getString(3),
                baseDatabaseSha256 = cursor.getString(4),
                privateBaseBindingId = cursor.getString(5),
                providerSnapshotGeneration = cursor.getString(6),
                predecessorTrackId = cursor.getLong(7),
                predecessorMetadataSha256 = cursor.getString(8),
                predecessorEmbeddingByteLength = cursor.getInt(9),
                predecessorEmbeddingSha256 = cursor.getString(10),
                replacementTrackId = cursor.getLong(11),
                committedAtEpochMs = cursor.getLong(12),
            )
        }
    }

    private fun hasTable(table: String): Boolean = database.rawQuery(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1",
        arrayOf(table),
    ).use { it.moveToFirst() }

    private fun countReceiptsForTrack(trackId: Long): Long = database.rawQuery(
        "SELECT COUNT(*) FROM $RECEIPT_TABLE WHERE track_id = ?",
        arrayOf(trackId.toString()),
    ).use { cursor -> check(cursor.moveToFirst()); cursor.getLong(0) }

    private fun insertTrack(metadata: V2CommitTrackMetadata): Long {
        val values = ContentValues().apply {
            put("metadata_key", metadata.metadataKey)
            put("filename_key", metadata.filenameKey)
            put("artist", metadata.artist)
            put("album", metadata.album)
            put("title", metadata.title)
            put("duration_ms", metadata.durationMs)
            put("file_path", metadata.filePath)
            put("source", metadata.source)
        }
        return database.insertOrThrow(TRACK_TABLE, null, values).also { trackId ->
            check(trackId > 0L) { "SQLite returned an invalid track id: $trackId" }
        }
    }

    private fun insertEmbedding(trackId: Long, blob: ByteArray) {
        val values = ContentValues().apply {
            put("track_id", trackId)
            put("embedding", blob)
        }
        val inserted = database.insertWithOnConflict(
            EMBEDDING_TABLE,
            null,
            values,
            SQLiteDatabase.CONFLICT_ABORT,
        )
        check(inserted == trackId) {
            "embedding insert returned row id $inserted for track $trackId"
        }
    }

    private fun insertReceipt(
        expected: V2EmbeddingCommitExpectation,
        trackId: Long,
        committedAtEpochMs: Long,
    ) {
        val values = ContentValues().apply {
            put("receipt_schema_version", RECEIPT_SCHEMA_VERSION)
            put("work_id", expected.workId)
            put("stable_track_span_id", expected.stableTrackSpanIdentity.stableTrackSpanId)
            put("stable_identity_spec_id", expected.stableTrackSpanIdentity.identitySpecId)
            put("stable_identity_strength", expected.stableTrackSpanIdentity.strength.name)
            put("embedding_spec_id", expected.embeddingSpecId)
            put("provider_physical_path", expected.providerSpan.normalizedPhysicalPath)
            put("provider_offset_ms", expected.providerSpan.offsetMs)
            put("provider_duration_ms", expected.providerSpan.durationMs)
            put("track_id", trackId)
            put("metadata_sha256", expected.metadataSha256)
            put("embedding_byte_length", expected.embeddingByteLength)
            put("embedding_sha256", expected.embeddingSha256)
            put("committed_at_epoch_ms", committedAtEpochMs)
        }
        val inserted = database.insertWithOnConflict(
            RECEIPT_TABLE,
            null,
            values,
            SQLiteDatabase.CONFLICT_ABORT,
        )
        check(inserted >= 0L) { "receipt insert failed for ${expected.workId}" }
    }

    private fun requireReusableAfterWrite(expected: V2EmbeddingCommitExpectation, trackId: Long) {
        when (val verified = V2EmbeddingCommitReconciler.decide(
            expected,
            readEvidence(expected.workId, expected.embeddingSpecId),
        )) {
            is V2EmbeddingCommitDecision.Reuse -> check(verified.trackId == trackId) {
                "post-insert receipt references ${verified.trackId}, expected $trackId"
            }
            V2EmbeddingCommitDecision.InsertNew -> error("receipt vanished before transaction commit")
            is V2EmbeddingCommitDecision.Conflict -> {
                throw V2EmbeddingCommitIntegrityException(verified.reasons)
            }
        }
    }

    private fun readEvidence(workId: String, embeddingSpecId: String): V2EmbeddingCommitEvidence {
        val receipt = readReceipt(workId, embeddingSpecId)
            ?: return V2EmbeddingCommitEvidence(receipt = null, track = null, embedding = null)
        return V2EmbeddingCommitEvidence(
            receipt = receipt,
            track = readTrack(receipt.trackId),
            embedding = readEmbedding(receipt.trackId),
        )
    }

    private fun readReceipt(
        workId: String,
        embeddingSpecId: String,
    ): V2EmbeddingCommitReceiptEvidence? = database.rawQuery(
        """
        SELECT receipt_schema_version, work_id, stable_track_span_id, stable_identity_spec_id,
               stable_identity_strength, embedding_spec_id, provider_physical_path,
               provider_offset_ms, provider_duration_ms, track_id,
               metadata_sha256, embedding_byte_length, embedding_sha256, committed_at_epoch_ms
        FROM $RECEIPT_TABLE
        WHERE work_id = ? AND embedding_spec_id = ?
        """.trimIndent(),
        arrayOf(workId, embeddingSpecId),
    ).use { cursor ->
        if (!cursor.moveToFirst()) return@use null
        V2EmbeddingCommitReceiptEvidence(
            receiptSchemaVersion = cursor.getInt(0),
            workId = cursor.getString(1).orEmpty(),
            stableTrackSpanId = cursor.getString(2).orEmpty(),
            stableIdentitySpecId = cursor.getString(3).orEmpty(),
            stableIdentityStrength = StableTrackSpanIdentityStrength.valueOf(cursor.getString(4)),
            embeddingSpecId = cursor.getString(5).orEmpty(),
            providerSpan = V2CommittedProviderSpan(
                normalizedPhysicalPath = cursor.getString(6).orEmpty(),
                offsetMs = cursor.getLong(7),
                durationMs = cursor.getLong(8),
            ),
            trackId = cursor.getLong(9),
            metadataSha256 = cursor.getString(10).orEmpty(),
            embeddingByteLength = cursor.getInt(11),
            embeddingSha256 = cursor.getString(12).orEmpty(),
            committedAtEpochMs = cursor.getLong(13),
        )
    }

    private fun readTrack(trackId: Long): V2CommittedTrackEvidence? = database.rawQuery(
        """
        SELECT id, metadata_key, filename_key, artist, album, title, duration_ms, file_path, source
        FROM $TRACK_TABLE
        WHERE id = ?
        """.trimIndent(),
        arrayOf(trackId.toString()),
    ).use { cursor ->
        if (!cursor.moveToFirst()) return@use null
        V2CommittedTrackEvidence(
            trackId = cursor.getLong(0),
            metadata = cursor.readTrackMetadataOrNull(),
        )
    }

    private fun readEmbedding(trackId: Long): V2StoredEmbeddingEvidence? = database.rawQuery(
        "SELECT track_id, embedding FROM $EMBEDDING_TABLE WHERE track_id = ?",
        arrayOf(trackId.toString()),
    ).use { cursor ->
        if (!cursor.moveToFirst() || cursor.isNull(1)) return@use null
        val blob = cursor.getBlob(1)
        V2StoredEmbeddingEvidence(
            trackId = cursor.getLong(0),
            byteLength = blob.size,
            sha256 = V2ArtifactDigests.sha256(blob),
        )
    }

    private fun Cursor.readTrackMetadataOrNull(): V2CommitTrackMetadata? {
        if (isNull(1) || isNull(2) || isNull(6) || isNull(7) || isNull(8)) return null
        return V2CommitTrackMetadata(
            metadataKey = getString(1),
            filenameKey = getString(2),
            artist = if (isNull(3)) null else getString(3),
            album = if (isNull(4)) null else getString(4),
            title = if (isNull(5)) null else getString(5),
            durationMs = getInt(6),
            filePath = getString(7),
            source = getString(8),
        )
    }

    companion object {
        const val RECEIPT_SCHEMA_VERSION = V2_EMBEDDING_COMMIT_RECEIPT_SCHEMA_VERSION
        const val RECEIPT_TABLE = "v2_embedding_commit_receipts_v4"
        const val TRACK_TABLE = "tracks"
        const val EMBEDDING_TABLE = "embeddings_clamp3"
        const val IMPORTED_ROW_SUPERSESSION_SCHEMA_VERSION = 1
        const val IMPORTED_ROW_SUPERSESSION_TABLE = "v2_imported_row_supersessions_v1"

        fun receiptKey(workId: String, embeddingSpecId: String): String =
            "$RECEIPT_TABLE:$workId:$embeddingSpecId"
    }

    private data class ImportedRowSupersessionEvidence(
        val schemaVersion: Int,
        val jobSpecId: String,
        val baseGenerationId: String,
        val baseManifestSha256: String,
        val baseDatabaseSha256: String,
        val privateBaseBindingId: String,
        val providerSnapshotGeneration: String,
        val predecessorTrackId: Long,
        val predecessorMetadataSha256: String,
        val predecessorEmbeddingByteLength: Int,
        val predecessorEmbeddingSha256: String,
        val replacementTrackId: Long,
        val committedAtEpochMs: Long,
    )
}

internal class V2ImportedRowSupersessionIntegrityException(message: String) :
    IllegalStateException(message)
