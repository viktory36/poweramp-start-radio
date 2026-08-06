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
class V2EmbeddingCommitRepositoryInstrumentedTest {
    @Test
    fun committedUnknownDurationReceiptSurvivesReopenAndPreventsDuplicate() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val dbFile = File(context.cacheDir, "v2-commit-${UUID.randomUUID()}.db")
        val vectorBlob = V2Clamp3VectorCodec.encode(
            FloatArray(V2_CLAMP3_DIMENSION).apply { this[11] = 1f },
        )
        val fingerprint = SourceFingerprint(
            fingerprintSpecId = "instrumented-source-v1",
            sizeBytes = 98_765L,
            lastModifiedEpochMs = 500L,
            fileKey = "instrumented-file",
            sampledContentSha256 =
                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            fullContentSha256 = null,
        )
        val artifact = VerifiedArtifact(
            kind = VerifiedArtifactKind.CLAMP_VECTOR,
            storageKey = "clamp/instrumented.bin",
            byteLength = V2_CLAMP3_BLOB_BYTES.toLong(),
            sha256 = V2ArtifactDigests.sha256(vectorBlob),
            completedUnits = 1,
            plannedUnits = 1,
            embeddingSpecId = SPEC_ID,
            sourceFingerprint = fingerprint,
            verifiedAtEpochMs = 1_000L,
        )
        val request = V2EmbeddingCommitRequest(
            workId = WORK_ID,
            stableTrackSpanIdentity = StableTrackSpanIdentity(
                identitySpecId = V2IndexingLedgerIds.STABLE_TRACK_SPAN_IDENTITY_SPEC_ID,
                stableTrackSpanId = "",
                strength = StableTrackSpanIdentityStrength.VERSIONED_SAMPLED_CONTENT_SHA256,
                contentFingerprintSpecId = fingerprint.fingerprintSpecId,
                contentSha256 = requireNotNull(fingerprint.sampledContentSha256),
                sourceSizeBytes = fingerprint.sizeBytes,
                sourceSampleRateHz = 48_000,
                startSourceSample = 0L,
                endSourceSampleExclusive = 5_925_600L,
            ).let { provisional ->
                provisional.copy(
                    stableTrackSpanId = V2IndexingLedgerIds.stableTrackSpanId(provisional),
                )
            },
            embeddingSpecId = SPEC_ID,
            providerSpan = V2CommittedProviderSpan(
                normalizedPhysicalPath = "/storage/emulated/0/Music/artist-title.flac",
                offsetMs = 0L,
                durationMs = 0L,
            ),
            metadata = V2CommitTrackMetadata(
                metadataKey = "artist|album|title|0",
                filenameKey = "artist-title.flac",
                artist = "Artist",
                album = "Album",
                title = "Title",
                durationMs = 0,
                filePath = "/storage/emulated/0/Music/artist-title.flac",
            ),
            sourceFingerprint = fingerprint,
            clampVectorArtifact = artifact,
            verifiedAtEpochMs = 2_000L,
        )

        try {
            openTestDatabase(dbFile).use { database ->
                val result = V2EmbeddingCommitRepository(database).commit(request, vectorBlob)
                assertEquals(V2EmbeddingCommitDisposition.INSERTED, result.disposition)
                assertEquals(1L, count(database, "tracks"))
                assertEquals(1L, count(database, "embeddings_clamp3"))
                assertEquals(1L, count(database, V2EmbeddingCommitRepository.RECEIPT_TABLE))
                val receiptSnapshot = V2ProviderSpanReceiptReader.read(database)
                assertEquals(true, receiptSnapshot.compatibleSchema)
                assertEquals(0, receiptSnapshot.invalidReceiptCount)
                assertEquals(
                    listOf(V2ProviderSpanReceipt(result.trackId, request.providerSpan)),
                    receiptSnapshot.receipts,
                )
            }

            openTestDatabase(dbFile).use { database ->
                val replay = V2EmbeddingCommitRepository(database).commit(request, vectorBlob)
                assertEquals(V2EmbeddingCommitDisposition.REUSED, replay.disposition)
                assertEquals(1L, count(database, "tracks"))
                assertEquals(1L, count(database, "embeddings_clamp3"))
                assertEquals(1L, count(database, V2EmbeddingCommitRepository.RECEIPT_TABLE))

                database.execSQL(
                    """
                    INSERT INTO tracks(metadata_key, filename_key, artist, album, title,
                                       duration_ms, file_path, source)
                    VALUES ('legacy', 'legacy.flac', NULL, NULL, 'Legacy', 1000,
                            '/legacy.flac', 'desktop')
                    """.trimIndent(),
                )
                val legacyTrackId = database.rawQuery("SELECT last_insert_rowid()", null).use {
                    check(it.moveToFirst())
                    it.getLong(0)
                }
                database.execSQL(
                    "INSERT INTO embeddings_clamp3(track_id, embedding) VALUES (?, ?)",
                    arrayOf(
                        legacyTrackId,
                        V2Clamp3VectorCodec.encode(
                            FloatArray(V2_CLAMP3_DIMENSION).apply { this[12] = 1f },
                        ),
                    ),
                )
                val coverage = V2EmbeddingSpecCoverage.inspect(database, 2, SPEC_ID)
                assertEquals(1, coverage.receiptBoundTrackCount)
                assertEquals(mapOf(SPEC_ID to 1), coverage.receiptSpecTrackCounts)
                assertEquals(1, coverage.compatibilityBase?.trackCount)
                assertEquals(
                    V2EmbeddingSpecCoverage.COMPATIBILITY_BASE_PROVENANCE_POLICY_ID,
                    coverage.compatibilityBase?.provenancePolicyId,
                )

                val conflictingSpec = "embedding-spec-v2-" + "b".repeat(64)
                database.execSQL(
                    "UPDATE ${V2EmbeddingCommitRepository.RECEIPT_TABLE} " +
                        "SET embedding_spec_id = ?",
                    arrayOf(conflictingSpec),
                )
                assertThrows(IllegalArgumentException::class.java) {
                    V2EmbeddingSpecCoverage.inspect(database, 2, SPEC_ID)
                }
                database.execSQL(
                    "UPDATE ${V2EmbeddingCommitRepository.RECEIPT_TABLE} " +
                        "SET embedding_spec_id = ?",
                    arrayOf(SPEC_ID),
                )

                database.execSQL(
                    "UPDATE embeddings_clamp3 SET embedding = ? WHERE track_id = ?",
                    arrayOf(ByteArray(V2_CLAMP3_BLOB_BYTES), replay.trackId),
                )
                assertThrows(V2EmbeddingCommitIntegrityException::class.java) {
                    V2EmbeddingCommitRepository(database).commit(request, vectorBlob)
                }
                assertEquals(2L, count(database, "tracks"))
            }
        } finally {
            listOf(dbFile, File("${dbFile.path}-wal"), File("${dbFile.path}-shm")).forEach(File::delete)
        }
    }

    private fun openTestDatabase(file: File): SQLiteDatabase =
        SQLiteDatabase.openOrCreateDatabase(file, null).also { database ->
            database.execSQL(
                """
                CREATE TABLE IF NOT EXISTS tracks (
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
            database.execSQL(
                """
                CREATE TABLE IF NOT EXISTS embeddings_clamp3 (
                    track_id INTEGER PRIMARY KEY REFERENCES tracks(id),
                    embedding BLOB NOT NULL
                )
                """.trimIndent(),
            )
        }

    private fun count(database: SQLiteDatabase, table: String): Long =
        database.rawQuery("SELECT COUNT(*) FROM $table", null).use { cursor ->
            check(cursor.moveToFirst())
            cursor.getLong(0)
        }

    companion object {
        private const val WORK_ID = "work-v1-instrumented"
        private const val SPEC_ID =
            "embedding-spec-v2-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    }
}
