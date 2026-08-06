package com.powerampstartradio.indexing.v2

import android.database.sqlite.SQLiteDatabase
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import java.util.UUID
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class V2ImportedRowSupersessionCommitInstrumentedTest {
    @Test
    fun replacementAndUniquePredecessorDeletionAreAtomicAndReplayIsIdempotent() = withDatabase { db ->
        val predecessor = insertLegacy(db, "Imported", 3)
        val vector = vector(11)
        val request = request(
            workId = "repair-work",
            title = "Repaired",
            vector = vector,
            supersession = authorization(predecessor),
            verifiedAt = 2_000L,
        )

        val first = V2EmbeddingCommitRepository(db).commit(request, vector)
        assertEquals(V2EmbeddingCommitDisposition.INSERTED, first.disposition)
        assertEquals(0L, countWhere(db, "tracks", "id = ?", predecessor.trackId))
        assertEquals(0L, countWhere(db, "embeddings_clamp3", "track_id = ?", predecessor.trackId))
        assertEquals(1L, count(db, "tracks"))
        assertEquals(1L, count(db, V2EmbeddingCommitRepository.RECEIPT_TABLE))
        assertEquals(1L, count(db, V2EmbeddingCommitRepository.IMPORTED_ROW_SUPERSESSION_TABLE))

        val replay = V2EmbeddingCommitRepository(db).commit(
            request.copy(verifiedAtEpochMs = 1_500L),
            vector,
        )
        assertEquals(V2EmbeddingCommitDisposition.REUSED, replay.disposition)
        assertEquals(first.trackId, replay.trackId)
        assertEquals(1L, count(db, "tracks"))
        assertEquals(1L, count(db, V2EmbeddingCommitRepository.IMPORTED_ROW_SUPERSESSION_TABLE))
    }

    @Test
    fun staleOrReceiptBoundPredecessorFailsBeforeAnyReplacementMutation() = withDatabase { db ->
        val predecessor = insertLegacy(db, "Imported", 4)
        val vector = vector(12)
        val stale = authorization(predecessor).copy(
            predecessor = predecessor.copy(embeddingSha256 = "f".repeat(64)),
        )
        assertThrows(V2ImportedRowSupersessionIntegrityException::class.java) {
            V2EmbeddingCommitRepository(db).commit(
                request("stale-work", "Stale", vector, stale),
                vector,
            )
        }
        assertEquals(1L, countWhere(db, "tracks", "id = ?", predecessor.trackId))
        assertEquals(0L, count(db, V2EmbeddingCommitRepository.RECEIPT_TABLE))

        val predecessorVector = vector(13)
        val receiptRequest = request("owner-work", "Owned", predecessorVector, null)
        val owner = V2EmbeddingCommitRepository(db).commit(receiptRequest, predecessorVector)
        val ownerEvidence = evidenceFor(db, owner.trackId)
        assertThrows(V2ImportedRowSupersessionIntegrityException::class.java) {
            V2EmbeddingCommitRepository(db).commit(
                request(
                    "receipt-bound-work",
                    "Receipt bound",
                    vector,
                    authorization(ownerEvidence),
                ),
                vector,
            )
        }
        assertEquals(1L, countWhere(db, "tracks", "id = ?", owner.trackId))
        assertEquals(1L, countWhere(db, "embeddings_clamp3", "track_id = ?", owner.trackId))
    }

    @Test
    fun auditInsertFailureRollsBackReplacementInsertAndPredecessorDeletion() = withDatabase { db ->
        val priorPredecessor = insertLegacy(db, "Prior imported", 19)
        val ordinaryVector = vector(20)
        V2EmbeddingCommitRepository(db).commit(
            request(
                "prior-repair-work",
                "Prior repaired",
                ordinaryVector,
                authorization(priorPredecessor),
            ),
            ordinaryVector,
        )
        val predecessor = insertLegacy(db, "Imported", 21)
        db.execSQL(
            """
            CREATE TRIGGER reject_supersession_audit
            BEFORE INSERT ON ${V2EmbeddingCommitRepository.IMPORTED_ROW_SUPERSESSION_TABLE}
            BEGIN SELECT RAISE(ABORT, 'injected audit failure'); END
            """.trimIndent(),
        )
        val replacementVector = vector(22)

        assertThrows(RuntimeException::class.java) {
            V2EmbeddingCommitRepository(db).commit(
                request(
                    "rollback-work",
                    "Replacement",
                    replacementVector,
                    authorization(predecessor),
                ),
                replacementVector,
            )
        }
        assertEquals(1L, countWhere(db, "tracks", "id = ?", predecessor.trackId))
        assertEquals(1L, countWhere(db, "embeddings_clamp3", "track_id = ?", predecessor.trackId))
        assertEquals(2L, count(db, "tracks"))
        assertEquals(1L, count(db, V2EmbeddingCommitRepository.RECEIPT_TABLE))
        assertEquals(1L, count(db, V2EmbeddingCommitRepository.IMPORTED_ROW_SUPERSESSION_TABLE))
    }

    private fun authorization(
        predecessor: V2ImportedPredecessorEvidence,
    ) = V2ImportedRowSupersessionCommitAuthorization(
        jobSpecId = "job-spec",
        baseGenerationId = "base-generation",
        baseManifestSha256 = "a".repeat(64),
        baseDatabaseSha256 = "b".repeat(64),
        privateBaseBindingId = "c".repeat(64),
        providerSnapshotGeneration = "provider-generation",
        predecessor = predecessor,
    )

    private fun request(
        workId: String,
        title: String,
        vector: ByteArray,
        supersession: V2ImportedRowSupersessionCommitAuthorization?,
        verifiedAt: Long = 2_000L,
    ): V2EmbeddingCommitRequest {
        val fingerprint = SourceFingerprint(
            fingerprintSpecId = "instrumented-source-v1",
            sizeBytes = 10_000L,
            lastModifiedEpochMs = 100L,
            fileKey = workId,
            sampledContentSha256 = "d".repeat(64),
            fullContentSha256 = null,
        )
        val stable = StableTrackSpanIdentity(
            identitySpecId = V2IndexingLedgerIds.STABLE_TRACK_SPAN_IDENTITY_SPEC_ID,
            stableTrackSpanId = "",
            strength = StableTrackSpanIdentityStrength.VERSIONED_SAMPLED_CONTENT_SHA256,
            contentFingerprintSpecId = fingerprint.fingerprintSpecId,
            contentSha256 = requireNotNull(fingerprint.sampledContentSha256),
            sourceSizeBytes = fingerprint.sizeBytes,
            sourceSampleRateHz = 48_000,
            startSourceSample = 0L,
            endSourceSampleExclusive = 480_000L,
        ).let { it.copy(stableTrackSpanId = V2IndexingLedgerIds.stableTrackSpanId(it)) }
        return V2EmbeddingCommitRequest(
            workId = workId,
            stableTrackSpanIdentity = stable,
            embeddingSpecId = SPEC_ID,
            providerSpan = V2CommittedProviderSpan(
                normalizedPhysicalPath = "/storage/Music/$workId.flac",
                offsetMs = 0L,
                durationMs = 10_000L,
            ),
            metadata = V2CommitTrackMetadata(
                metadataKey = "artist|album|$title|10000",
                filenameKey = "$workId.flac",
                artist = "Artist",
                album = "Album",
                title = title,
                durationMs = 10_000,
                filePath = "/storage/Music/$workId.flac",
            ),
            sourceFingerprint = fingerprint,
            clampVectorArtifact = VerifiedArtifact(
                kind = VerifiedArtifactKind.CLAMP_VECTOR,
                storageKey = "clamp/$workId.bin",
                byteLength = vector.size.toLong(),
                sha256 = V2ArtifactDigests.sha256(vector),
                completedUnits = 1,
                plannedUnits = 1,
                embeddingSpecId = SPEC_ID,
                sourceFingerprint = fingerprint,
                verifiedAtEpochMs = 1_000L,
            ),
            verifiedAtEpochMs = verifiedAt,
            importedRowSupersession = supersession,
        )
    }

    private fun insertLegacy(
        db: SQLiteDatabase,
        title: String,
        vectorIndex: Int,
    ): V2ImportedPredecessorEvidence {
        val metadata = V2CommitTrackMetadata(
            metadataKey = "artist|album|$title|10000",
            filenameKey = "legacy-$vectorIndex.flac",
            artist = "Artist",
            album = "Album",
            title = title,
            durationMs = 10_000,
            filePath = "/storage/Music/legacy-$vectorIndex.flac",
            source = "desktop",
        )
        db.execSQL(
            """
            INSERT INTO tracks(metadata_key, filename_key, artist, album, title,
                               duration_ms, file_path, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """.trimIndent(),
            arrayOf<Any?>(
                metadata.metadataKey,
                metadata.filenameKey,
                metadata.artist,
                metadata.album,
                metadata.title,
                metadata.durationMs,
                metadata.filePath,
                metadata.source,
            ),
        )
        val id = db.rawQuery("SELECT last_insert_rowid()", null).use {
            check(it.moveToFirst())
            it.getLong(0)
        }
        val blob = vector(vectorIndex)
        db.execSQL(
            "INSERT INTO embeddings_clamp3(track_id, embedding) VALUES (?, ?)",
            arrayOf(id, blob),
        )
        return V2ImportedPredecessorEvidence(
            trackId = id,
            metadata = metadata,
            metadataSha256 = V2CommitMetadataIdentity.sha256(metadata),
            embeddingByteLength = blob.size,
            embeddingSha256 = V2ArtifactDigests.sha256(blob),
        )
    }

    private fun evidenceFor(db: SQLiteDatabase, trackId: Long): V2ImportedPredecessorEvidence =
        db.rawQuery(
            """
            SELECT t.metadata_key, t.filename_key, t.artist, t.album, t.title,
                   t.duration_ms, t.file_path, t.source, e.embedding
            FROM tracks t JOIN embeddings_clamp3 e ON e.track_id = t.id WHERE t.id = ?
            """.trimIndent(),
            arrayOf(trackId.toString()),
        ).use { cursor ->
            check(cursor.moveToFirst())
            val metadata = V2CommitTrackMetadata(
                cursor.getString(0),
                cursor.getString(1),
                cursor.getString(2),
                cursor.getString(3),
                cursor.getString(4),
                cursor.getInt(5),
                cursor.getString(6),
                cursor.getString(7),
            )
            val blob = cursor.getBlob(8)
            V2ImportedPredecessorEvidence(
                trackId,
                metadata,
                V2CommitMetadataIdentity.sha256(metadata),
                blob.size,
                V2ArtifactDigests.sha256(blob),
            )
        }

    private fun vector(index: Int): ByteArray = V2Clamp3VectorCodec.encode(
        FloatArray(V2_CLAMP3_DIMENSION).apply { this[index] = 1f },
    )

    private fun withDatabase(block: (SQLiteDatabase) -> Unit) {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val file = File(context.cacheDir, "v2-supersession-${UUID.randomUUID()}.db")
        try {
            SQLiteDatabase.openOrCreateDatabase(file, null).use { db ->
                db.execSQL(
                    """
                    CREATE TABLE tracks (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        metadata_key TEXT NOT NULL,
                        filename_key TEXT NOT NULL,
                        artist TEXT,
                        album TEXT,
                        title TEXT,
                        duration_ms INTEGER,
                        file_path TEXT NOT NULL,
                        source TEXT DEFAULT 'desktop'
                    )
                    """.trimIndent(),
                )
                db.execSQL(
                    """
                    CREATE TABLE embeddings_clamp3 (
                        track_id INTEGER PRIMARY KEY REFERENCES tracks(id),
                        embedding BLOB NOT NULL
                    )
                    """.trimIndent(),
                )
                block(db)
            }
        } finally {
            listOf(file, File(file.path + "-wal"), File(file.path + "-shm")).forEach(File::delete)
        }
    }

    private fun count(db: SQLiteDatabase, table: String): Long =
        db.rawQuery("SELECT COUNT(*) FROM $table", null).use {
            check(it.moveToFirst())
            it.getLong(0)
        }

    private fun countWhere(
        db: SQLiteDatabase,
        table: String,
        where: String,
        id: Long,
    ): Long = db.rawQuery(
        "SELECT COUNT(*) FROM $table WHERE $where",
        arrayOf(id.toString()),
    ).use { cursor -> check(cursor.moveToFirst()); cursor.getLong(0) }

    private companion object {
        const val SPEC_ID =
            "embedding-spec-v2-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    }
}
