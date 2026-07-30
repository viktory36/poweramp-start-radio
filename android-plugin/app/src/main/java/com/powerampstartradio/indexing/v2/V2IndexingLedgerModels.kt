package com.powerampstartradio.indexing.v2

/** On-disk compatibility boundary for the V2 indexing job ledger. */
object V2IndexingLedgerSchema {
    const val VERSION = 5
    const val FORMAT = "poweramp-start-radio-v2-indexing-ledger"
}

enum class IndexingJobState {
    PLANNED,
    RUNNING,
    PAUSE_REQUESTED,
    PAUSED,
    WAITING_FOR_INPUT,
    INTERRUPTED,
    READY_TO_RESUME,
    CANCELLING,
    CANCELLED,
    ACTIVATING,
    COMPLETE,
}

enum class RecoveryPhase {
    EXECUTION,
    ACTIVATION,
}

enum class IndexingTrackState {
    QUEUED,
    PREFLIGHTING,
    PREFLIGHTED,
    DECODING,
    MERT_COMPLETE,
    CLAMPING,
    CLAMP_COMPLETE,
    COMMITTING,
    COMMITTED,
    RETRYABLE_FAILURE,
    BLOCKED_FAILURE,
    SKIPPED_BY_USER,
}

/** States whose evidence is safe to resume after process death. */
enum class TrackCheckpoint {
    QUEUED,
    PREFLIGHTED,
    MERT_COMPLETE,
    CLAMP_COMPLETE,
    COMMITTED,
}

enum class IndexingStage {
    PREFLIGHT,
    DECODE_AND_MERT,
    CLAMP3,
    DATABASE_COMMIT,
    DATABASE_ACTIVATION,
    CANCELLATION,
}

enum class VerifiedArtifactKind {
    MERT_FEATURES,
    CLAMP_VECTOR,
    DATABASE_COMMIT,
}

enum class TrackFailureCategory {
    SOURCE_UNAVAILABLE,
    PERMISSION,
    SPAN_OR_IDENTITY,
    DECODE_UNSUPPORTED,
    DECODE_CORRUPT,
    TOO_SHORT,
    RESOURCE_TRANSIENT,
    INFERENCE_OR_ARTIFACT,
    STORAGE_OR_DATABASE,
}

enum class TrackFailureCode {
    SOURCE_MISSING,
    STORAGE_UNMOUNTED,
    SOURCE_UNREADABLE,
    POWERAMP_PROVIDER_UNAVAILABLE,
    ANDROID_AUDIO_PERMISSION_DENIED,
    POWERAMP_PERMISSION_DENIED,
    INVALID_LOGICAL_SPAN,
    CUE_SOURCE_IMAGE,
    SOURCE_FINGERPRINT_CHANGED,
    PROVIDER_SNAPSHOT_CHANGED,
    CONTAINER_EOS_MISMATCH,
    NO_AUDIO_STREAM,
    UNSUPPORTED_CODEC_OR_CONTAINER,
    DRM_PROTECTED,
    CORRUPT_OR_TRUNCATED,
    DECODER_ERROR,
    BELOW_MINIMUM_DURATION,
    OUT_OF_MEMORY,
    THERMAL_SHUTDOWN,
    PROCESS_INTERRUPTED,
    STAGE_TIMEOUT,
    MODEL_LOAD_FAILED,
    INFERENCE_FAILED,
    INVALID_MODEL_OUTPUT,
    PARTIAL_ARTIFACT,
    ARTIFACT_CHECKSUM_MISMATCH,
    STORAGE_FULL,
    DATABASE_BUSY,
    DATABASE_GENERATION_CHANGED,
    IMPORTED_ROW_AUTHORIZATION_CHANGED,
    COMMIT_FAILED,
    UNKNOWN_TRANSIENT,
    UNKNOWN_BLOCKED,
}

enum class FailureDisposition {
    RETRYABLE,
    BLOCKED,
}

enum class RetryTrigger {
    IMMEDIATE,
    PROCESS_RESTART,
    SOURCE_AVAILABLE,
    PERMISSION_GRANTED,
    RESOURCE_RECOVERED,
    STORAGE_RECOVERED,
    SOURCE_OR_LIBRARY_CHANGED,
    DECODER_OR_APP_CHANGED,
    USER_REQUEST,
    NEW_JOB_REQUIRED,
    NEVER,
}

/** Runtime facts retained with failures; none are recommendation inputs. */
data class IndexingRuntimeFingerprint(
    val appVersionCode: Long,
    val appBuildId: String,
    val decoderRuntimeId: String,
    val platformFingerprint: String,
)

data class EmbeddingSpecFingerprint(
    val specId: String,
    val preprocessingSpecId: String,
    val decoderPolicyId: String,
    val inferenceBackendPolicyId: String,
    val outputDimension: Int,
    val modelArtifactSha256: Map<String, String>,
)

data class TextRetrievalSpecFingerprint(
    val specId: String,
    val compatibleAudioEmbeddingSpecId: String,
    val textModelSha256: String,
    val tokenizerModelSha256: String,
    val tokenizerPolicyId: String,
    val tokenizerRuntimeContractSha256: String,
    val outputSpaceId: String,
    val outputDimension: Int,
    val inferenceBackendPolicyId: String,
)

/**
 * File identity captured before work begins. Only a full hash proves cross-path equality.
 * A versioned sampled hash may detect mutation but must never authorize reuse or collapse.
 */
data class SourceFingerprint(
    val fingerprintSpecId: String,
    val sizeBytes: Long,
    val lastModifiedEpochMs: Long?,
    val fileKey: String?,
    val sampledContentSha256: String?,
    val fullContentSha256: String?,
)

enum class StableTrackSpanIdentityStrength {
    FULL_CONTENT_SHA256,
    VERSIONED_SAMPLED_CONTENT_SHA256,
}

/**
 * Path- and provider-independent identity for one exact acoustic span.
 *
 * This is deliberately separate from [SelectedTrackDescriptor.workId], which remains a local
 * execution identity and may change after a path/provider migration.
 */
data class StableTrackSpanIdentity(
    val identitySpecId: String,
    val stableTrackSpanId: String,
    val strength: StableTrackSpanIdentityStrength,
    val contentFingerprintSpecId: String,
    val contentSha256: String,
    val sourceSizeBytes: Long,
    val sourceSampleRateHz: Int,
    val startSourceSample: Long,
    val endSourceSampleExclusive: Long,
)

data class DisplayTrackMetadata(
    val artist: String,
    val album: String,
    val title: String,
)

/** Exact output of the matching normalizer that planned this job. */
data class NormalizedTrackMetadata(
    val normalizationSpecId: String,
    val artist: String,
    val album: String,
    val title: String,
    val metadataKey: String,
)

data class ExpectedTrackWork(
    val mertWindows: Int,
    val clampSegments: Int,
)

/** One complete provider acquisition shared by every selected row in an immutable job. */
data class PowerampProviderSnapshotEvidence(
    val libraryGeneration: String,
    val acquisition: V2ProviderSnapshotAcquisitionEvidence,
)

/** Acoustic coordinates. Ordinary files are provisional until a physical-EOS decode finalizes them. */
data class FinalizedAudioSpanEvidence(
    val kind: V2ResolvedAudioSpanKind,
    val authority: V2AudioSpanAuthority,
    val executionBoundaryRequirement: V2ExecutionBoundaryRequirement,
    val providerSpan: V2ProviderSpanEvidence,
    val cueClassification: V2CueClassificationEvidence,
    val container: V2AudioContainerEvidence,
    val startUs: Long,
    val endExclusiveUs: Long,
    val startSourceSample: Long,
    val endSourceSampleExclusive: Long,
    val sourceSampleCount: Long,
    val exactSampleCount24k: Long,
    val expectedWork: ExpectedTrackWork,
)

/** Selected identity and span; ordinary work has one allowed provisional-to-decoded transition. */
data class SelectedTrackDescriptor(
    val workId: String,
    /** Initial content-addressed work identity retained after the one-way EOS finalization. */
    val provisionalWorkId: String? = null,
    val stableTrackSpanIdentity: StableTrackSpanIdentity,
    val ordinal: Int,
    val powerampFileId: Long,
    val providerSnapshotGeneration: String,
    val providerRow: V2ProviderPathRowEvidence,
    val displayMetadata: DisplayTrackMetadata,
    val normalizedMetadata: NormalizedTrackMetadata,
    val physicalPath: String,
    val canonicalPath: String,
    val sourceFingerprint: SourceFingerprint,
    val finalizedAudioSpan: FinalizedAudioSpanEvidence,
) {
    val expectedWork: ExpectedTrackWork get() = finalizedAudioSpan.expectedWork
    val providerDurationMs: Long get() = providerRow.durationMs
    val providerOffsetMs: Long get() = providerRow.offsetMs
    val cueFolderId: Long? get() = providerRow.cueSourceImageFolderId
}

data class SelectedTrackInput(
    val powerampFileId: Long,
    val providerSnapshotGeneration: String,
    val providerRow: V2ProviderPathRowEvidence,
    val displayMetadata: DisplayTrackMetadata,
    val normalizedMetadata: NormalizedTrackMetadata,
    val physicalPath: String,
    val sourceFingerprint: SourceFingerprint,
    val finalizedAudioSpan: FinalizedAudioSpanEvidence,
) {
    val expectedWork: ExpectedTrackWork get() = finalizedAudioSpan.expectedWork
}

/** Content-addressed plan. Only ordinary EOS finalization may replace its acoustic identities. */
data class IndexingJobSpec(
    val jobId: String,
    val specId: String,
    /** Initial provisional job-spec identity retained through every per-track EOS finalization. */
    val provisionalParentSpecId: String? = null,
    val createdAtEpochMs: Long,
    val providerSnapshot: PowerampProviderSnapshotEvidence,
    val embeddingSpec: EmbeddingSpecFingerprint,
    val textRetrievalSpec: TextRetrievalSpecFingerprint,
    val runtimeFingerprint: IndexingRuntimeFingerprint,
    /** Generation copied into this job's private staging database, or null for bootstrap. */
    val baseGenerationId: String?,
    val rebuildDerivedIndexes: Boolean,
    val tracks: List<SelectedTrackDescriptor>,
)

/** Only complete, checksummed artifacts may advance a checkpoint. */
data class VerifiedArtifact(
    val kind: VerifiedArtifactKind,
    val storageKey: String,
    val byteLength: Long,
    val sha256: String,
    val completedUnits: Int,
    val plannedUnits: Int,
    val embeddingSpecId: String,
    val sourceFingerprint: SourceFingerprint,
    val verifiedAtEpochMs: Long,
    val executionBoundary: VerifiedExecutionBoundaryEvidence? = null,
)

/** Observed decoder boundary that must agree with the authoritative span before MERT publication. */
data class VerifiedExecutionBoundaryEvidence(
    val requirement: V2ExecutionBoundaryRequirement,
    val observedStartSourceSample: Long,
    val observedEndSourceSampleExclusive: Long,
    val observedSourceSampleCount: Long,
    val exactSampleCount24k: Long,
    val endOfStreamReached: Boolean,
    val providerBoundaryEnforced: Boolean,
)

/**
 * Checksummed progress inside a stage. It is resumable evidence, not permission to advance the
 * major track checkpoint; [completedUnits] must remain below [plannedUnits].
 */
data class VerifiedStageProgress(
    val stage: IndexingStage,
    val storageKey: String,
    val byteLength: Long,
    val sha256: String,
    val completedUnits: Int,
    val plannedUnits: Int,
    val resumeCursor: String,
    val embeddingSpecId: String,
    val sourceFingerprint: SourceFingerprint,
    val verifiedAtEpochMs: Long,
)

/** Aggregated repeated evidence for one typed failure under the same source and model spec. */
data class TrackFailureAggregate(
    val failureId: String,
    val code: TrackFailureCode,
    val category: TrackFailureCategory,
    val stage: IndexingStage,
    val disposition: FailureDisposition,
    val retryTrigger: RetryTrigger,
    val resumeFrom: TrackCheckpoint,
    val diagnostic: String,
    val firstOccurredAtEpochMs: Long,
    val lastOccurredAtEpochMs: Long,
    val firstAttemptNumber: Int,
    val lastAttemptNumber: Int,
    val occurrences: Int,
    val sourceFingerprint: SourceFingerprint,
    val embeddingSpecId: String,
    val appBuildId: String,
)

data class IndexingTrackLedger(
    val workId: String,
    val state: IndexingTrackState,
    val checkpoint: TrackCheckpoint,
    val attemptCount: Int,
    val currentAttemptNumber: Int?,
    val activeFailureId: String?,
    val stageProgress: VerifiedStageProgress?,
    val verifiedArtifacts: List<VerifiedArtifact>,
    val failures: List<TrackFailureAggregate>,
    val updatedAtEpochMs: Long,
)

/** Durable proof required before an ACTIVATING job may become COMPLETE. */
data class ActivatedGenerationEvidence(
    val generationId: String,
    val activationBindingId: String,
    val jobSpecId: String,
    val receiptEmbeddingSpecId: String,
    val textRetrievalSpecId: String,
    val baseGenerationId: String?,
    val rebuildDerivedIndexes: Boolean,
    val manifestSha256: String,
    val databaseSha256: String,
    val databaseContentSha256: String,
    val orderedTrackSetSha256: String,
    val stableTrackUidMappingSha256: String,
    val embeddingSha256: String,
    val graphSha256: String?,
    val activatedAtEpochMs: Long,
)

data class IndexingJobLedger(
    val schemaVersion: Int,
    val jobSpec: IndexingJobSpec,
    val state: IndexingJobState,
    val recoveryPhase: RecoveryPhase?,
    val revision: Long,
    val updatedAtEpochMs: Long,
    val stateReason: String?,
    val tracks: List<IndexingTrackLedger>,
    /** Mutable scheduling only. It is deliberately excluded from every acoustic identity. */
    val executionProfile: V2IndexingExecutionProfile = V2IndexingExecutionProfile.FULL,
    val activationEvidence: ActivatedGenerationEvidence? = null,
)

enum class RestartAction {
    NONE,
    WAIT_FOR_RESUME,
    FINISH_CANCELLATION,
}

data class RestartReconciliation(
    val ledger: IndexingJobLedger,
    val action: RestartAction,
    val changed: Boolean,
)

class InvalidIndexingLedgerException(message: String) : IllegalStateException(message)

class IndexingLedgerConflictException(message: String) : IllegalStateException(message)

class UnsupportedIndexingLedgerSchemaException(message: String) : IllegalStateException(message)
