package com.powerampstartradio.indexing.v2

import java.nio.charset.StandardCharsets
import java.security.MessageDigest
import java.text.Normalizer
import java.util.Locale
import java.util.UUID

data class EmbeddingSpecInput(
    val preprocessingSpecId: String,
    val decoderPolicyId: String,
    val inferenceBackendPolicyId: String,
    val outputDimension: Int,
    val modelArtifactSha256: Map<String, String>,
)

data class TextRetrievalSpecInput(
    val compatibleAudioEmbeddingSpecId: String,
    val textModelSha256: String,
    val tokenizerModelSha256: String,
    val tokenizerPolicyId: String,
    val tokenizerRuntimeContractSha256: String,
    val outputSpaceId: String,
    val outputDimension: Int,
    val inferenceBackendPolicyId: String,
)

/** Builds content-addressed job specifications before an executor is started. */
object V2IndexingLedgerPlanner {
    fun createEmbeddingSpec(input: EmbeddingSpecInput): EmbeddingSpecFingerprint {
        val normalizedEntries = input.modelArtifactSha256.map { (name, hash) ->
            name.trim().lowercase(Locale.ROOT) to hash.lowercase(Locale.ROOT)
        }
        if (normalizedEntries.map { it.first }.toSet().size != normalizedEntries.size) {
            throw InvalidIndexingLedgerException("model artifact names collide after normalization")
        }
        val normalizedModels = normalizedEntries.toMap().toSortedMap()
        val provisional = EmbeddingSpecFingerprint(
            specId = "",
            preprocessingSpecId = input.preprocessingSpecId,
            decoderPolicyId = input.decoderPolicyId,
            inferenceBackendPolicyId = input.inferenceBackendPolicyId,
            outputDimension = input.outputDimension,
            modelArtifactSha256 = normalizedModels,
        )
        val result = provisional.copy(
            specId = V2IndexingLedgerIds.embeddingSpecId(provisional),
        )
        V2IndexingLedgerValidator.requireValidEmbeddingSpec(result)
        return result
    }

    fun createTextRetrievalSpec(
        input: TextRetrievalSpecInput,
    ): TextRetrievalSpecFingerprint {
        val provisional = TextRetrievalSpecFingerprint(
            specId = "",
            compatibleAudioEmbeddingSpecId = input.compatibleAudioEmbeddingSpecId,
            textModelSha256 = input.textModelSha256.lowercase(Locale.ROOT),
            tokenizerModelSha256 = input.tokenizerModelSha256.lowercase(Locale.ROOT),
            tokenizerPolicyId = input.tokenizerPolicyId,
            tokenizerRuntimeContractSha256 =
                input.tokenizerRuntimeContractSha256.lowercase(Locale.ROOT),
            outputSpaceId = input.outputSpaceId,
            outputDimension = input.outputDimension,
            inferenceBackendPolicyId = input.inferenceBackendPolicyId,
        )
        return provisional.copy(
            specId = V2IndexingLedgerIds.textRetrievalSpecId(provisional),
        ).also(V2IndexingLedgerValidator::requireValidTextRetrievalSpec)
    }

    fun planJob(
        providerSnapshot: PowerampProviderSnapshotEvidence,
        embeddingSpec: EmbeddingSpecFingerprint,
        textRetrievalSpec: TextRetrievalSpecFingerprint,
        runtimeFingerprint: IndexingRuntimeFingerprint,
        selectedTracks: List<SelectedTrackInput>,
        rebuildDerivedIndexes: Boolean,
        executionProfile: V2IndexingExecutionProfile = V2IndexingExecutionProfile.FULL,
        baseGenerationId: String? = null,
        createdAtEpochMs: Long,
        jobId: String = UUID.randomUUID().toString(),
    ): IndexingJobLedger {
        if (selectedTracks.any {
                it.providerSnapshotGeneration != providerSnapshot.libraryGeneration
            }
        ) {
            throw InvalidIndexingLedgerException(
                "selected track span was resolved from a different provider snapshot",
            )
        }
        val descriptors = selectedTracks.mapIndexed { ordinal, input ->
            val canonicalPath = V2IndexingLedgerIds.canonicalPath(input.physicalPath)
            val sourceFingerprint = input.sourceFingerprint.copy(
                sampledContentSha256 = input.sourceFingerprint.sampledContentSha256
                    ?.lowercase(Locale.ROOT),
                fullContentSha256 = input.sourceFingerprint.fullContentSha256
                    ?.lowercase(Locale.ROOT),
            )
            val provisional = SelectedTrackDescriptor(
                workId = "",
                provisionalWorkId = null,
                stableTrackSpanIdentity = V2IndexingLedgerIds.stableTrackSpanIdentity(
                    sourceFingerprint = sourceFingerprint,
                    span = input.finalizedAudioSpan,
                ),
                ordinal = ordinal,
                powerampFileId = input.powerampFileId,
                providerSnapshotGeneration = input.providerSnapshotGeneration,
                providerRow = input.providerRow,
                displayMetadata = input.displayMetadata,
                normalizedMetadata = input.normalizedMetadata,
                physicalPath = input.physicalPath,
                canonicalPath = canonicalPath,
                sourceFingerprint = sourceFingerprint,
                finalizedAudioSpan = input.finalizedAudioSpan,
            )
            provisional.copy(workId = V2IndexingLedgerIds.workId(provisional))
        }
        val provisionalSpec = IndexingJobSpec(
            jobId = jobId,
            specId = "",
            provisionalParentSpecId = null,
            createdAtEpochMs = createdAtEpochMs,
            providerSnapshot = providerSnapshot,
            embeddingSpec = embeddingSpec,
            textRetrievalSpec = textRetrievalSpec,
            runtimeFingerprint = runtimeFingerprint,
            baseGenerationId = baseGenerationId,
            rebuildDerivedIndexes = rebuildDerivedIndexes,
            tracks = descriptors,
        )
        val spec = provisionalSpec.copy(specId = V2IndexingLedgerIds.jobSpecId(provisionalSpec))
        val ledger = IndexingJobLedger(
            schemaVersion = V2IndexingLedgerSchema.VERSION,
            jobSpec = spec,
            state = IndexingJobState.PLANNED,
            recoveryPhase = null,
            revision = 0L,
            updatedAtEpochMs = createdAtEpochMs,
            stateReason = null,
            executionProfile = executionProfile,
            tracks = descriptors.map { descriptor ->
                IndexingTrackLedger(
                    workId = descriptor.workId,
                    state = IndexingTrackState.QUEUED,
                    checkpoint = TrackCheckpoint.QUEUED,
                    attemptCount = 0,
                    currentAttemptNumber = null,
                    activeFailureId = null,
                    stageProgress = null,
                    verifiedArtifacts = emptyList(),
                    failures = emptyList(),
                    updatedAtEpochMs = createdAtEpochMs,
                )
            },
        )
        V2IndexingLedgerValidator.requireValid(ledger)
        return ledger
    }
}

internal object V2IndexingLedgerIds {
    const val STABLE_TRACK_SPAN_IDENTITY_SPEC_ID =
        "stable-track-span-v1:content-sha256:native-half-open-sample-span"
    const val FULL_CONTENT_FINGERPRINT_SPEC_ID = "full-content-sha256-v1"

    private const val STABLE_TRACK_SPAN_ID_PREFIX = "stable-track-span-v1-"
    private const val WORK_ID_PREFIX = "work-v4-"
    private const val EMBEDDING_SPEC_ID_PREFIX = "embedding-spec-v2-"
    private const val TEXT_RETRIEVAL_SPEC_ID_PREFIX = "text-retrieval-spec-v2-"
    private const val JOB_SPEC_ID_PREFIX = "job-spec-v5-"
    private const val FAILURE_ID_PREFIX = "failure-v2-"

    fun canonicalPath(path: String): String = Normalizer.normalize(
        path.replace('\\', '/').replace(Regex("/{2,}"), "/"),
        Normalizer.Form.NFC,
    )

    fun embeddingSpecId(spec: EmbeddingSpecFingerprint): String =
        EMBEDDING_SPEC_ID_PREFIX + CanonicalDigest().apply {
            string("embedding-spec-v2")
            string(spec.preprocessingSpecId)
            string(spec.decoderPolicyId)
            string(spec.inferenceBackendPolicyId)
            int(spec.outputDimension)
            int(spec.modelArtifactSha256.size)
            spec.modelArtifactSha256.toSortedMap().forEach { (name, sha256) ->
                string(name)
                string(sha256.lowercase(Locale.ROOT))
            }
        }.hex()

    fun textRetrievalSpecId(spec: TextRetrievalSpecFingerprint): String =
        TEXT_RETRIEVAL_SPEC_ID_PREFIX + CanonicalDigest().apply {
            string("text-retrieval-spec-v2")
            string(spec.compatibleAudioEmbeddingSpecId)
            string(spec.textModelSha256)
            string(spec.tokenizerModelSha256)
            string(spec.tokenizerPolicyId)
            string(spec.tokenizerRuntimeContractSha256)
            string(spec.outputSpaceId)
            int(spec.outputDimension)
            string(spec.inferenceBackendPolicyId)
        }.hex()

    fun workId(descriptor: SelectedTrackDescriptor): String =
        WORK_ID_PREFIX + CanonicalDigest().apply {
            string("track-acoustic-span-v4")
            nullableString(descriptor.provisionalWorkId)
            string(descriptor.stableTrackSpanIdentity.stableTrackSpanId)
            string(descriptor.canonicalPath)
            fingerprint(descriptor.sourceFingerprint)
            providerAcousticEvidence(descriptor.providerRow)
            finalizedAudioSpan(descriptor.finalizedAudioSpan)
        }.hex()

    fun stableTrackSpanIdentity(
        sourceFingerprint: SourceFingerprint,
        span: FinalizedAudioSpanEvidence,
    ): StableTrackSpanIdentity {
        val fullSha256 = sourceFingerprint.fullContentSha256?.lowercase(Locale.ROOT)
        val sampledSha256 = sourceFingerprint.sampledContentSha256?.lowercase(Locale.ROOT)
        val strength: StableTrackSpanIdentityStrength
        val contentFingerprintSpecId: String
        val contentSha256: String
        if (fullSha256 != null) {
            strength = StableTrackSpanIdentityStrength.FULL_CONTENT_SHA256
            contentFingerprintSpecId = FULL_CONTENT_FINGERPRINT_SPEC_ID
            contentSha256 = fullSha256
        } else {
            strength = StableTrackSpanIdentityStrength.VERSIONED_SAMPLED_CONTENT_SHA256
            contentFingerprintSpecId = sourceFingerprint.fingerprintSpecId
            contentSha256 = sampledSha256
                ?: throw InvalidIndexingLedgerException("source has no content identity")
        }
        val provisional = StableTrackSpanIdentity(
            identitySpecId = STABLE_TRACK_SPAN_IDENTITY_SPEC_ID,
            stableTrackSpanId = "",
            strength = strength,
            contentFingerprintSpecId = contentFingerprintSpecId,
            contentSha256 = contentSha256,
            sourceSizeBytes = sourceFingerprint.sizeBytes,
            sourceSampleRateHz = span.container.sampleRateHz,
            startSourceSample = span.startSourceSample,
            endSourceSampleExclusive = span.endSourceSampleExclusive,
        )
        return provisional.copy(
            stableTrackSpanId = stableTrackSpanId(provisional),
        )
    }

    fun stableTrackSpanId(identity: StableTrackSpanIdentity): String =
        STABLE_TRACK_SPAN_ID_PREFIX + CanonicalDigest().apply {
            string(identity.identitySpecId)
            string(identity.strength.name)
            string(identity.contentFingerprintSpecId)
            string(identity.contentSha256)
            long(identity.sourceSizeBytes)
            int(identity.sourceSampleRateHz)
            long(identity.startSourceSample)
            long(identity.endSourceSampleExclusive)
        }.hex()

    fun jobSpecId(spec: IndexingJobSpec): String =
        JOB_SPEC_ID_PREFIX + CanonicalDigest().apply {
            string("job-spec-v5")
            nullableString(spec.provisionalParentSpecId)
            providerSnapshot(spec.providerSnapshot)
            string(spec.embeddingSpec.specId)
            string(spec.textRetrievalSpec.specId)
            runtime(spec.runtimeFingerprint)
            nullableString(spec.baseGenerationId)
            boolean(spec.rebuildDerivedIndexes)
            int(spec.tracks.size)
            spec.tracks.forEach { descriptor ->
                string(descriptor.workId)
                nullableString(descriptor.provisionalWorkId)
                stableTrackSpanIdentity(descriptor.stableTrackSpanIdentity)
                int(descriptor.ordinal)
                long(descriptor.powerampFileId)
                string(descriptor.providerSnapshotGeneration)
                providerRow(descriptor.providerRow)
                displayMetadata(descriptor.displayMetadata)
                normalizedMetadata(descriptor.normalizedMetadata)
                string(descriptor.physicalPath)
                string(descriptor.canonicalPath)
                fingerprint(descriptor.sourceFingerprint)
                finalizedAudioSpan(descriptor.finalizedAudioSpan)
            }
        }.hex()

    private fun CanonicalDigest.stableTrackSpanIdentity(value: StableTrackSpanIdentity) {
        string(value.identitySpecId)
        string(value.stableTrackSpanId)
        string(value.strength.name)
        string(value.contentFingerprintSpecId)
        string(value.contentSha256)
        long(value.sourceSizeBytes)
        int(value.sourceSampleRateHz)
        long(value.startSourceSample)
        long(value.endSourceSampleExclusive)
    }

    fun failureId(
        workId: String,
        code: TrackFailureCode,
        stage: IndexingStage,
        sourceFingerprint: SourceFingerprint,
        embeddingSpecId: String,
        appBuildId: String,
    ): String = FAILURE_ID_PREFIX + CanonicalDigest().apply {
        string("failure-v1")
        string(workId)
        string(code.name)
        string(stage.name)
        fingerprint(sourceFingerprint)
        string(embeddingSpecId)
        string(appBuildId)
    }.hex()

    private class CanonicalDigest {
        private val digest = MessageDigest.getInstance("SHA-256")

        fun boolean(value: Boolean) = byte(if (value) 1 else 0)

        fun byte(value: Int) {
            digest.update(value.toByte())
        }

        fun int(value: Int) {
            byte(value ushr 24)
            byte(value ushr 16)
            byte(value ushr 8)
            byte(value)
        }

        fun long(value: Long) {
            byte((value ushr 56).toInt())
            byte((value ushr 48).toInt())
            byte((value ushr 40).toInt())
            byte((value ushr 32).toInt())
            byte((value ushr 24).toInt())
            byte((value ushr 16).toInt())
            byte((value ushr 8).toInt())
            byte(value.toInt())
        }

        fun string(value: String) {
            val bytes = value.toByteArray(StandardCharsets.UTF_8)
            int(bytes.size)
            digest.update(bytes)
        }

        fun nullableString(value: String?) {
            boolean(value != null)
            if (value != null) string(value)
        }

        fun nullableLong(value: Long?) {
            boolean(value != null)
            if (value != null) long(value)
        }

        fun fingerprint(value: SourceFingerprint) {
            string(value.fingerprintSpecId)
            long(value.sizeBytes)
            nullableLong(value.lastModifiedEpochMs)
            nullableString(value.fileKey)
            nullableString(value.sampledContentSha256)
            nullableString(value.fullContentSha256)
        }

        fun providerSnapshot(value: PowerampProviderSnapshotEvidence) {
            string(value.libraryGeneration)
            val acquisition = value.acquisition
            string(acquisition.queryUri)
            strings(acquisition.requestedColumns)
            strings(acquisition.returnedColumns)
            int(acquisition.rowCount)
            boolean(acquisition.cursorExhaustedNormally)
        }

        fun providerRow(value: V2ProviderPathRowEvidence) {
            long(value.powerampFileId)
            providerAcousticEvidence(value)
            nullableString(value.artist)
            nullableString(value.album)
            nullableString(value.title)
        }

        fun providerAcousticEvidence(value: V2ProviderPathRowEvidence) {
            string(value.physicalPath)
            string(value.providerPhysicalPath)
            long(value.offsetMs)
            boolean(value.offsetWasNull)
            long(value.durationMs)
            nullableLong(value.cueSourceImageFolderId)
        }

        fun finalizedAudioSpan(value: FinalizedAudioSpanEvidence) {
            string(value.kind.name)
            string(value.authority.name)
            string(value.executionBoundaryRequirement.name)
            long(value.providerSpan.offsetUs)
            long(value.providerSpan.durationUs)
            long(value.providerSpan.endExclusiveUs)
            cueClassification(value.cueClassification)
            val container = value.container
            string(container.physicalPath)
            int(container.audioTrackIndex)
            long(container.durationUsEstimate)
            string(container.durationEstimateSource.name)
            int(container.sampleRateHz)
            int(container.channelCount)
            string(container.mime)
            long(value.startUs)
            long(value.endExclusiveUs)
            long(value.startSourceSample)
            long(value.endSourceSampleExclusive)
            long(value.sourceSampleCount)
            long(value.exactSampleCount24k)
            int(value.expectedWork.mertWindows)
            int(value.expectedWork.clampSegments)
        }

        fun cueClassification(value: V2CueClassificationEvidence) {
            int(value.providerGroupRowCount)
            int(value.logicalRowCount)
            longs(value.nonZeroOffsetRowIds)
            longs(value.rawSourceImageRowIds)
        }

        fun strings(values: List<String>) {
            int(values.size)
            values.forEach(::string)
        }

        fun longs(values: List<Long>) {
            int(values.size)
            values.forEach(::long)
        }

        fun displayMetadata(value: DisplayTrackMetadata) {
            string(value.artist)
            string(value.album)
            string(value.title)
        }

        fun normalizedMetadata(value: NormalizedTrackMetadata) {
            string(value.normalizationSpecId)
            string(value.artist)
            string(value.album)
            string(value.title)
            string(value.metadataKey)
        }

        fun runtime(value: IndexingRuntimeFingerprint) {
            long(value.appVersionCode)
            string(value.appBuildId)
            string(value.decoderRuntimeId)
            string(value.platformFingerprint)
        }

        fun hex(): String = digest.digest().joinToString("") { byte -> "%02x".format(byte) }
    }
}
