package com.powerampstartradio.indexing.v2

import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.charset.StandardCharsets
import java.security.MessageDigest
import java.util.Locale
import kotlin.math.abs
import kotlin.math.sqrt

const val V2_CLAMP3_DIMENSION = 768
const val V2_FLOAT32_BYTES = 4
const val V2_CLAMP3_BLOB_BYTES = V2_CLAMP3_DIMENSION * V2_FLOAT32_BYTES
const val V2_EMBEDDING_COMMIT_RECEIPT_SCHEMA_VERSION = 4

private val V2_SHA256_PATTERN = Regex("^[0-9a-f]{64}$")

/** Exact values persisted in the legacy-compatible tracks row. They carry no recommendation signal. */
data class V2CommitTrackMetadata(
    val metadataKey: String,
    val filenameKey: String,
    val artist: String?,
    val album: String?,
    val title: String?,
    val durationMs: Int,
    val filePath: String,
    val source: String = "phone",
)

/**
 * Cheap, exact locator for the Poweramp source span that produced one committed embedding.
 *
 * This is not a content identity. It is retained so a complete future provider snapshot can
 * prove that the same provider occurrence is already represented without re-hashing the library.
 */
data class V2CommittedProviderSpan(
    val normalizedPhysicalPath: String,
    val offsetMs: Long,
    /** Zero is Poweramp's canonical unknown-duration locator, not an empty acoustic span. */
    val durationMs: Long,
)

data class V2EmbeddingCommitExpectation(
    val workId: String,
    val stableTrackSpanIdentity: StableTrackSpanIdentity,
    val embeddingSpecId: String,
    val providerSpan: V2CommittedProviderSpan,
    val metadata: V2CommitTrackMetadata,
    val metadataSha256: String,
    val embeddingByteLength: Int,
    val embeddingSha256: String,
)

data class V2EmbeddingCommitReceiptEvidence(
    val receiptSchemaVersion: Int,
    val workId: String,
    val stableTrackSpanId: String,
    val stableIdentitySpecId: String,
    val stableIdentityStrength: StableTrackSpanIdentityStrength,
    val embeddingSpecId: String,
    val providerSpan: V2CommittedProviderSpan,
    val trackId: Long,
    val metadataSha256: String,
    val embeddingByteLength: Int,
    val embeddingSha256: String,
    val committedAtEpochMs: Long,
)

data class V2CommittedTrackEvidence(
    val trackId: Long,
    val metadata: V2CommitTrackMetadata?,
)

data class V2StoredEmbeddingEvidence(
    val trackId: Long,
    val byteLength: Int,
    val sha256: String,
)

data class V2EmbeddingCommitEvidence(
    val receipt: V2EmbeddingCommitReceiptEvidence?,
    val track: V2CommittedTrackEvidence?,
    val embedding: V2StoredEmbeddingEvidence?,
)

enum class V2EmbeddingCommitConflictReason {
    ORPHAN_EVIDENCE_WITHOUT_RECEIPT,
    RECEIPT_WORK_ID_MISMATCH,
    RECEIPT_STABLE_TRACK_SPAN_ID_MISMATCH,
    RECEIPT_STABLE_IDENTITY_SPEC_MISMATCH,
    RECEIPT_STABLE_IDENTITY_STRENGTH_MISMATCH,
    RECEIPT_EMBEDDING_SPEC_MISMATCH,
    RECEIPT_PROVIDER_SPAN_MISMATCH,
    RECEIPT_TRACK_ID_INVALID,
    RECEIPT_SCHEMA_VERSION_MISMATCH,
    RECEIPT_COMMIT_TIME_INVALID,
    RECEIPT_METADATA_SHA_MISMATCH,
    RECEIPT_EMBEDDING_LENGTH_MISMATCH,
    RECEIPT_EMBEDDING_SHA_MISMATCH,
    TRACK_ROW_MISSING,
    TRACK_ID_MISMATCH,
    TRACK_METADATA_MISMATCH,
    TRACK_METADATA_SHA_MISMATCH,
    EMBEDDING_ROW_MISSING,
    EMBEDDING_TRACK_ID_MISMATCH,
    EMBEDDING_LENGTH_MISMATCH,
    EMBEDDING_SHA_MISMATCH,
}

sealed interface V2EmbeddingCommitDecision {
    data object InsertNew : V2EmbeddingCommitDecision

    data class Reuse(
        val trackId: Long,
    ) : V2EmbeddingCommitDecision

    data class Conflict(
        val reasons: List<V2EmbeddingCommitConflictReason>,
    ) : V2EmbeddingCommitDecision
}

class V2EmbeddingCommitIntegrityException(
    val reasons: List<V2EmbeddingCommitConflictReason>,
) : IllegalStateException("V2 embedding commit integrity conflict: ${reasons.joinToString()}")

/** Pure replay decision logic, separated from SQLite so every integrity branch is unit-testable. */
object V2EmbeddingCommitReconciler {
    fun expectation(
        workId: String,
        stableTrackSpanIdentity: StableTrackSpanIdentity,
        embeddingSpecId: String,
        providerSpan: V2CommittedProviderSpan,
        metadata: V2CommitTrackMetadata,
        embeddingBlob: ByteArray,
    ): V2EmbeddingCommitExpectation {
        require(workId.isNotBlank()) { "workId must not be blank" }
        requireValidStableTrackSpanIdentity(stableTrackSpanIdentity)
        require(embeddingSpecId.isNotBlank()) { "embeddingSpecId must not be blank" }
        requireValidProviderSpan(providerSpan)
        requireValidMetadata(metadata)
        V2Clamp3VectorCodec.requireValidBlob(embeddingBlob)
        return V2EmbeddingCommitExpectation(
            workId = workId,
            stableTrackSpanIdentity = stableTrackSpanIdentity,
            embeddingSpecId = embeddingSpecId,
            providerSpan = providerSpan,
            metadata = metadata,
            metadataSha256 = V2CommitMetadataIdentity.sha256(metadata),
            embeddingByteLength = embeddingBlob.size,
            embeddingSha256 = V2ArtifactDigests.sha256(embeddingBlob),
        )
    }

    fun decide(
        expected: V2EmbeddingCommitExpectation,
        evidence: V2EmbeddingCommitEvidence,
    ): V2EmbeddingCommitDecision {
        val receipt = evidence.receipt
        if (receipt == null) {
            return if (evidence.track == null && evidence.embedding == null) {
                V2EmbeddingCommitDecision.InsertNew
            } else {
                V2EmbeddingCommitDecision.Conflict(
                    listOf(V2EmbeddingCommitConflictReason.ORPHAN_EVIDENCE_WITHOUT_RECEIPT),
                )
            }
        }

        val reasons = buildList {
            if (receipt.receiptSchemaVersion != V2_EMBEDDING_COMMIT_RECEIPT_SCHEMA_VERSION) {
                add(V2EmbeddingCommitConflictReason.RECEIPT_SCHEMA_VERSION_MISMATCH)
            }
            if (receipt.committedAtEpochMs < 0L) {
                add(V2EmbeddingCommitConflictReason.RECEIPT_COMMIT_TIME_INVALID)
            }
            if (receipt.workId != expected.workId) {
                add(V2EmbeddingCommitConflictReason.RECEIPT_WORK_ID_MISMATCH)
            }
            if (receipt.stableTrackSpanId != expected.stableTrackSpanIdentity.stableTrackSpanId) {
                add(V2EmbeddingCommitConflictReason.RECEIPT_STABLE_TRACK_SPAN_ID_MISMATCH)
            }
            if (receipt.stableIdentitySpecId != expected.stableTrackSpanIdentity.identitySpecId) {
                add(V2EmbeddingCommitConflictReason.RECEIPT_STABLE_IDENTITY_SPEC_MISMATCH)
            }
            if (receipt.stableIdentityStrength != expected.stableTrackSpanIdentity.strength) {
                add(V2EmbeddingCommitConflictReason.RECEIPT_STABLE_IDENTITY_STRENGTH_MISMATCH)
            }
            if (receipt.embeddingSpecId != expected.embeddingSpecId) {
                add(V2EmbeddingCommitConflictReason.RECEIPT_EMBEDDING_SPEC_MISMATCH)
            }
            if (receipt.providerSpan != expected.providerSpan) {
                add(V2EmbeddingCommitConflictReason.RECEIPT_PROVIDER_SPAN_MISMATCH)
            }
            if (receipt.trackId <= 0L) {
                add(V2EmbeddingCommitConflictReason.RECEIPT_TRACK_ID_INVALID)
            }
            if (receipt.metadataSha256 != expected.metadataSha256) {
                add(V2EmbeddingCommitConflictReason.RECEIPT_METADATA_SHA_MISMATCH)
            }
            if (receipt.embeddingByteLength != expected.embeddingByteLength) {
                add(V2EmbeddingCommitConflictReason.RECEIPT_EMBEDDING_LENGTH_MISMATCH)
            }
            if (receipt.embeddingSha256 != expected.embeddingSha256) {
                add(V2EmbeddingCommitConflictReason.RECEIPT_EMBEDDING_SHA_MISMATCH)
            }

            val track = evidence.track
            if (track == null) {
                add(V2EmbeddingCommitConflictReason.TRACK_ROW_MISSING)
            } else {
                if (track.trackId != receipt.trackId) {
                    add(V2EmbeddingCommitConflictReason.TRACK_ID_MISMATCH)
                }
                if (track.metadata != expected.metadata) {
                    add(V2EmbeddingCommitConflictReason.TRACK_METADATA_MISMATCH)
                }
                if (track.metadata == null ||
                    V2CommitMetadataIdentity.sha256(track.metadata) != receipt.metadataSha256
                ) {
                    add(V2EmbeddingCommitConflictReason.TRACK_METADATA_SHA_MISMATCH)
                }
            }

            val embedding = evidence.embedding
            if (embedding == null) {
                add(V2EmbeddingCommitConflictReason.EMBEDDING_ROW_MISSING)
            } else {
                if (embedding.trackId != receipt.trackId) {
                    add(V2EmbeddingCommitConflictReason.EMBEDDING_TRACK_ID_MISMATCH)
                }
                if (embedding.byteLength != expected.embeddingByteLength ||
                    embedding.byteLength != receipt.embeddingByteLength
                ) {
                    add(V2EmbeddingCommitConflictReason.EMBEDDING_LENGTH_MISMATCH)
                }
                if (embedding.sha256 != expected.embeddingSha256 ||
                    embedding.sha256 != receipt.embeddingSha256
                ) {
                    add(V2EmbeddingCommitConflictReason.EMBEDDING_SHA_MISMATCH)
                }
            }
        }.distinct()

        return if (reasons.isEmpty()) {
            V2EmbeddingCommitDecision.Reuse(receipt.trackId)
        } else {
            V2EmbeddingCommitDecision.Conflict(reasons)
        }
    }

    private fun requireValidMetadata(metadata: V2CommitTrackMetadata) {
        require(metadata.metadataKey.isNotBlank()) { "metadataKey must not be blank" }
        require(metadata.filenameKey.isNotBlank()) { "filenameKey must not be blank" }
        require(metadata.durationMs >= 0) { "durationMs must not be negative" }
        require(metadata.filePath.isNotBlank()) { "filePath must not be blank" }
        require(metadata.source.isNotBlank()) { "source must not be blank" }
    }

    private fun requireValidStableTrackSpanIdentity(identity: StableTrackSpanIdentity) {
        require(identity.identitySpecId == V2IndexingLedgerIds.STABLE_TRACK_SPAN_IDENTITY_SPEC_ID) {
            "unsupported stable track-span identity spec"
        }
        require(identity.stableTrackSpanId.matches(Regex("^stable-track-span-v1-[0-9a-f]{64}$"))) {
            "invalid stable track-span id"
        }
        require(identity.stableTrackSpanId == V2IndexingLedgerIds.stableTrackSpanId(identity)) {
            "stable track-span id does not match its evidence"
        }
        requireV2Sha256(identity.contentSha256, "stable content SHA-256")
        require(identity.contentFingerprintSpecId.isNotBlank()) {
            "stable content fingerprint spec must not be blank"
        }
        when (identity.strength) {
            StableTrackSpanIdentityStrength.FULL_CONTENT_SHA256 -> require(
                identity.contentFingerprintSpecId ==
                    V2IndexingLedgerIds.FULL_CONTENT_FINGERPRINT_SPEC_ID,
            ) { "full-content stable identity has the wrong fingerprint spec" }

            StableTrackSpanIdentityStrength.VERSIONED_SAMPLED_CONTENT_SHA256 -> require(
                identity.contentFingerprintSpecId !=
                    V2IndexingLedgerIds.FULL_CONTENT_FINGERPRINT_SPEC_ID,
            ) { "sampled stable identity claims the full-content fingerprint spec" }
        }
        require(identity.sourceSizeBytes > 0L) { "stable source size must be positive" }
        require(identity.sourceSampleRateHz > 0) { "stable source sample rate must be positive" }
        require(identity.startSourceSample >= 0L) { "stable start sample must not be negative" }
        require(identity.endSourceSampleExclusive > identity.startSourceSample) {
            "stable sample span must be non-empty"
        }
    }

    private fun requireValidProviderSpan(span: V2CommittedProviderSpan) {
        require(span.normalizedPhysicalPath.startsWith('/')) {
            "provider span path must be normalized and absolute"
        }
        require('\\' !in span.normalizedPhysicalPath) {
            "provider span path must use forward slashes"
        }
        require(span.offsetMs >= 0L) { "provider span offset must not be negative" }
        require(span.durationMs >= 0L) { "provider span duration must not be negative" }
    }
}

/** Stable hash over exact SQLite values; null and empty strings intentionally remain distinct. */
object V2CommitMetadataIdentity {
    fun sha256(metadata: V2CommitTrackMetadata): String {
        val digest = MessageDigest.getInstance("SHA-256")
        digest.putString("v2-commit-track-metadata-v1")
        digest.putString(metadata.metadataKey)
        digest.putString(metadata.filenameKey)
        digest.putNullableString(metadata.artist)
        digest.putNullableString(metadata.album)
        digest.putNullableString(metadata.title)
        digest.putInt(metadata.durationMs)
        digest.putString(metadata.filePath)
        digest.putString(metadata.source)
        return digest.digest().toV2CommitHex()
    }
}

/** Canonical little-endian float32 representation used by files and embeddings_clamp3. */
object V2Clamp3VectorCodec {
    private const val NORMALIZATION_TOLERANCE = 0.001

    fun encode(vector: FloatArray): ByteArray {
        requireValidVector(vector)
        return ByteBuffer.allocate(V2_CLAMP3_BLOB_BYTES)
            .order(ByteOrder.LITTLE_ENDIAN)
            .also { buffer -> vector.forEach(buffer::putFloat) }
            .array()
    }

    fun decode(blob: ByteArray): FloatArray {
        require(blob.size == V2_CLAMP3_BLOB_BYTES) {
            "CLaMP3 blob must be exactly $V2_CLAMP3_BLOB_BYTES bytes, got ${blob.size}"
        }
        val buffer = ByteBuffer.wrap(blob).order(ByteOrder.LITTLE_ENDIAN)
        return FloatArray(V2_CLAMP3_DIMENSION) { buffer.float }.also(::requireValidVector)
    }

    fun requireValidBlob(blob: ByteArray) {
        decode(blob)
    }

    fun requireValidVector(vector: FloatArray) {
        require(vector.size == V2_CLAMP3_DIMENSION) {
            "CLaMP3 vector must have $V2_CLAMP3_DIMENSION values, got ${vector.size}"
        }
        var squaredNorm = 0.0
        vector.forEachIndexed { index, value ->
            require(value.isFinite()) { "CLaMP3 vector contains a non-finite value at $index" }
            squaredNorm += value.toDouble() * value.toDouble()
        }
        val norm = sqrt(squaredNorm)
        require(abs(norm - 1.0) <= NORMALIZATION_TOLERANCE) {
            "CLaMP3 vector must be L2-normalized; norm=$norm"
        }
    }
}

object V2ArtifactDigests {
    fun sha256(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256")
        .digest(bytes)
        .toV2CommitHex()
}

private fun MessageDigest.putString(value: String) {
    val bytes = value.toByteArray(StandardCharsets.UTF_8)
    putInt(bytes.size)
    update(bytes)
}

private fun MessageDigest.putNullableString(value: String?) {
    update(if (value == null) 0.toByte() else 1.toByte())
    if (value != null) putString(value)
}

private fun MessageDigest.putInt(value: Int) {
    update((value ushr 24).toByte())
    update((value ushr 16).toByte())
    update((value ushr 8).toByte())
    update(value.toByte())
}

internal fun ByteArray.toV2CommitHex(): String = joinToString("") { byte ->
    String.format(Locale.ROOT, "%02x", byte.toInt() and 0xff)
}

internal fun requireV2Sha256(value: String, label: String) {
    require(V2_SHA256_PATTERN.matches(value)) { "$label must be a lowercase SHA-256" }
}
