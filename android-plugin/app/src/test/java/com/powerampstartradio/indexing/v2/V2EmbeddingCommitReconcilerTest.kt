package com.powerampstartradio.indexing.v2

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class V2EmbeddingCommitReconcilerTest {
    @Test
    fun `missing receipt is the only state that permits an insert`() {
        assertEquals(
            V2EmbeddingCommitDecision.InsertNew,
            V2EmbeddingCommitReconciler.decide(
                expectation,
                V2EmbeddingCommitEvidence(null, null, null),
            ),
        )

        val decision = V2EmbeddingCommitReconciler.decide(
            expectation,
            V2EmbeddingCommitEvidence(null, validTrack, null),
        )
        assertConflict(decision, V2EmbeddingCommitConflictReason.ORPHAN_EVIDENCE_WITHOUT_RECEIPT)
    }

    @Test
    fun `exact receipt track and blob are reused`() {
        assertEquals(
            V2EmbeddingCommitDecision.Reuse(TRACK_ID),
            V2EmbeddingCommitReconciler.decide(expectation, validEvidence),
        )
    }

    @Test
    fun `receipt identity and immutable payload mismatches are hard conflicts`() {
        val cases = listOf(
            validReceipt.copy(receiptSchemaVersion = 1) to
                V2EmbeddingCommitConflictReason.RECEIPT_SCHEMA_VERSION_MISMATCH,
            validReceipt.copy(committedAtEpochMs = -1) to
                V2EmbeddingCommitConflictReason.RECEIPT_COMMIT_TIME_INVALID,
            validReceipt.copy(workId = "other-work") to
                V2EmbeddingCommitConflictReason.RECEIPT_WORK_ID_MISMATCH,
            validReceipt.copy(stableTrackSpanId = "stable-track-span-v1-" + "e".repeat(64)) to
                V2EmbeddingCommitConflictReason.RECEIPT_STABLE_TRACK_SPAN_ID_MISMATCH,
            validReceipt.copy(stableIdentitySpecId = "other-stable-spec") to
                V2EmbeddingCommitConflictReason.RECEIPT_STABLE_IDENTITY_SPEC_MISMATCH,
            validReceipt.copy(
                stableIdentityStrength = StableTrackSpanIdentityStrength.FULL_CONTENT_SHA256,
            ) to V2EmbeddingCommitConflictReason.RECEIPT_STABLE_IDENTITY_STRENGTH_MISMATCH,
            validReceipt.copy(embeddingSpecId = "other-spec") to
                V2EmbeddingCommitConflictReason.RECEIPT_EMBEDDING_SPEC_MISMATCH,
            validReceipt.copy(
                providerSpan = providerSpan.copy(durationMs = providerSpan.durationMs + 1),
            ) to V2EmbeddingCommitConflictReason.RECEIPT_PROVIDER_SPAN_MISMATCH,
            validReceipt.copy(trackId = 0) to
                V2EmbeddingCommitConflictReason.RECEIPT_TRACK_ID_INVALID,
            validReceipt.copy(metadataSha256 = SHA_B) to
                V2EmbeddingCommitConflictReason.RECEIPT_METADATA_SHA_MISMATCH,
            validReceipt.copy(embeddingByteLength = V2_CLAMP3_BLOB_BYTES - 4) to
                V2EmbeddingCommitConflictReason.RECEIPT_EMBEDDING_LENGTH_MISMATCH,
            validReceipt.copy(embeddingSha256 = SHA_B) to
                V2EmbeddingCommitConflictReason.RECEIPT_EMBEDDING_SHA_MISMATCH,
        )

        cases.forEach { (receipt, reason) ->
            val decision = V2EmbeddingCommitReconciler.decide(
                expectation,
                validEvidence.copy(receipt = receipt),
            )
            assertConflict(decision, reason)
        }
    }

    @Test
    fun `replay verifies the referenced track row exactly`() {
        assertConflict(
            V2EmbeddingCommitReconciler.decide(
                expectation,
                validEvidence.copy(track = null),
            ),
            V2EmbeddingCommitConflictReason.TRACK_ROW_MISSING,
        )
        assertConflict(
            V2EmbeddingCommitReconciler.decide(
                expectation,
                validEvidence.copy(track = validTrack.copy(trackId = TRACK_ID + 1)),
            ),
            V2EmbeddingCommitConflictReason.TRACK_ID_MISMATCH,
        )
        assertConflict(
            V2EmbeddingCommitReconciler.decide(
                expectation,
                validEvidence.copy(
                    track = validTrack.copy(metadata = metadata.copy(title = "Changed")),
                ),
            ),
            V2EmbeddingCommitConflictReason.TRACK_METADATA_MISMATCH,
        )
        val nullMetadata = V2EmbeddingCommitReconciler.decide(
            expectation,
            validEvidence.copy(track = validTrack.copy(metadata = null)),
        )
        assertConflict(nullMetadata, V2EmbeddingCommitConflictReason.TRACK_METADATA_MISMATCH)
        assertConflict(nullMetadata, V2EmbeddingCommitConflictReason.TRACK_METADATA_SHA_MISMATCH)
    }

    @Test
    fun `replay verifies exact embedding track length and SHA`() {
        assertConflict(
            V2EmbeddingCommitReconciler.decide(
                expectation,
                validEvidence.copy(embedding = null),
            ),
            V2EmbeddingCommitConflictReason.EMBEDDING_ROW_MISSING,
        )
        assertConflict(
            V2EmbeddingCommitReconciler.decide(
                expectation,
                validEvidence.copy(
                    embedding = validEmbedding.copy(trackId = TRACK_ID + 1),
                ),
            ),
            V2EmbeddingCommitConflictReason.EMBEDDING_TRACK_ID_MISMATCH,
        )
        assertConflict(
            V2EmbeddingCommitReconciler.decide(
                expectation,
                validEvidence.copy(
                    embedding = validEmbedding.copy(byteLength = V2_CLAMP3_BLOB_BYTES + 4),
                ),
            ),
            V2EmbeddingCommitConflictReason.EMBEDDING_LENGTH_MISMATCH,
        )
        assertConflict(
            V2EmbeddingCommitReconciler.decide(
                expectation,
                validEvidence.copy(embedding = validEmbedding.copy(sha256 = SHA_B)),
            ),
            V2EmbeddingCommitConflictReason.EMBEDDING_SHA_MISMATCH,
        )
    }

    @Test
    fun `metadata digest is deterministic and preserves null versus empty`() {
        assertEquals(
            V2CommitMetadataIdentity.sha256(metadata),
            V2CommitMetadataIdentity.sha256(metadata.copy()),
        )
        assertNotEquals(
            V2CommitMetadataIdentity.sha256(metadata),
            V2CommitMetadataIdentity.sha256(metadata.copy(artist = "")),
        )
        assertNotEquals(
            V2CommitMetadataIdentity.sha256(metadata),
            V2CommitMetadataIdentity.sha256(metadata.copy(durationMs = metadata.durationMs + 1)),
        )
    }

    @Test
    fun `commit expectation rejects malformed and non-normalized vectors`() {
        assertThrows(IllegalArgumentException::class.java) {
            V2EmbeddingCommitReconciler.expectation(
                WORK_ID,
                stableIdentity,
                SPEC_ID,
                providerSpan,
                metadata,
                ByteArray(V2_CLAMP3_BLOB_BYTES - 1),
            )
        }
        assertThrows(IllegalArgumentException::class.java) {
            V2EmbeddingCommitReconciler.expectation(
                WORK_ID,
                stableIdentity,
                SPEC_ID,
                providerSpan,
                metadata,
                V2Clamp3VectorCodec.encode(FloatArray(V2_CLAMP3_DIMENSION).apply {
                    this[0] = 1f
                }).also { bytes ->
                    // Two unit components produce sqrt(2), which is not a normalized embedding.
                    java.nio.ByteBuffer.wrap(bytes)
                        .order(java.nio.ByteOrder.LITTLE_ENDIAN)
                        .putFloat(4, 1f)
                },
            )
        }
    }

    @Test
    fun `unknown provider duration remains an exact commit occurrence locator`() {
        val unknownProviderSpan = providerSpan.copy(durationMs = 0L)
        val unknownMetadata = metadata.copy(durationMs = 0)

        val unknownExpectation = V2EmbeddingCommitReconciler.expectation(
            WORK_ID,
            stableIdentity,
            SPEC_ID,
            unknownProviderSpan,
            unknownMetadata,
            vectorBlob,
        )

        assertEquals(0L, unknownExpectation.providerSpan.durationMs)
        assertEquals(0, unknownExpectation.metadata.durationMs)
        assertTrue(unknownExpectation.stableTrackSpanIdentity.endSourceSampleExclusive > 0L)
        assertThrows(IllegalArgumentException::class.java) {
            V2EmbeddingCommitReconciler.expectation(
                WORK_ID,
                stableIdentity,
                SPEC_ID,
                unknownProviderSpan.copy(durationMs = -1L),
                unknownMetadata,
                vectorBlob,
            )
        }
    }

    private fun assertConflict(
        decision: V2EmbeddingCommitDecision,
        reason: V2EmbeddingCommitConflictReason,
    ) {
        assertTrue(decision is V2EmbeddingCommitDecision.Conflict)
        assertTrue((decision as V2EmbeddingCommitDecision.Conflict).reasons.contains(reason))
    }

    private val metadata = V2CommitTrackMetadata(
        metadataKey = "artist|album|title|245",
        filenameKey = "artist-title.flac",
        artist = null,
        album = "Album",
        title = "Title",
        durationMs = 245_123,
        filePath = "/storage/emulated/0/Music/artist-title.flac",
        source = "phone",
    )
    private val vectorBlob = V2Clamp3VectorCodec.encode(
        FloatArray(V2_CLAMP3_DIMENSION).apply { this[0] = 1f },
    )
    private val stableIdentity = StableTrackSpanIdentity(
        identitySpecId = V2IndexingLedgerIds.STABLE_TRACK_SPAN_IDENTITY_SPEC_ID,
        stableTrackSpanId = "",
        strength = StableTrackSpanIdentityStrength.VERSIONED_SAMPLED_CONTENT_SHA256,
        contentFingerprintSpecId = V2FixedRegionSampling.SPEC_ID,
        contentSha256 = "c".repeat(64),
        sourceSizeBytes = 98_765L,
        sourceSampleRateHz = 48_000,
        startSourceSample = 0L,
        endSourceSampleExclusive = 5_925_600L,
    ).let { provisional ->
        provisional.copy(
            stableTrackSpanId = V2IndexingLedgerIds.stableTrackSpanId(provisional),
        )
    }
    private val providerSpan = V2CommittedProviderSpan(
        "/storage/emulated/0/Music/artist-title.flac",
        0L,
        245_123L,
    )
    private val expectation = V2EmbeddingCommitReconciler.expectation(
        WORK_ID,
        stableIdentity,
        SPEC_ID,
        providerSpan,
        metadata,
        vectorBlob,
    )
    private val validReceipt = V2EmbeddingCommitReceiptEvidence(
        receiptSchemaVersion = V2_EMBEDDING_COMMIT_RECEIPT_SCHEMA_VERSION,
        workId = WORK_ID,
        stableTrackSpanId = stableIdentity.stableTrackSpanId,
        stableIdentitySpecId = stableIdentity.identitySpecId,
        stableIdentityStrength = stableIdentity.strength,
        embeddingSpecId = SPEC_ID,
        providerSpan = providerSpan,
        trackId = TRACK_ID,
        metadataSha256 = expectation.metadataSha256,
        embeddingByteLength = expectation.embeddingByteLength,
        embeddingSha256 = expectation.embeddingSha256,
        committedAtEpochMs = 1_000L,
    )
    private val validTrack = V2CommittedTrackEvidence(TRACK_ID, metadata)
    private val validEmbedding = V2StoredEmbeddingEvidence(
        TRACK_ID,
        V2_CLAMP3_BLOB_BYTES,
        expectation.embeddingSha256,
    )
    private val validEvidence = V2EmbeddingCommitEvidence(
        validReceipt,
        validTrack,
        validEmbedding,
    )

    companion object {
        private const val WORK_ID = "work-v1-abc"
        private const val SPEC_ID = "embedding-spec-v1-abc"
        private const val TRACK_ID = 42L
        private const val SHA_B = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    }

}
