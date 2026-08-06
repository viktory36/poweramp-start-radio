package com.powerampstartradio.indexing.v2

private val SHA256 = Regex("^[0-9a-f]{64}$")
private val SAFE_JOB_ID = Regex("^[A-Za-z0-9._-]{1,128}$")

/** Full structural validation at trust boundaries, plus changed-row validation for hot writes. */
object V2IndexingLedgerValidator {
    private val storeManagedTransitionDepth = ThreadLocal<Int>()

    fun requireValid(ledger: IndexingJobLedger) {
        if ((storeManagedTransitionDepth.get() ?: 0) > 0) return
        val errors = validate(ledger)
        if (errors.isNotEmpty()) {
            throw InvalidIndexingLedgerException(errors.joinToString("; "))
        }
    }

    /**
     * State-machine helpers retain their defensive validation when used directly. The durable
     * store uses this scope because it starts from a fully validated cached ledger and validates
     * only the transition delta before fsyncing it.
     */
    internal fun <T> withStoreManagedTransitionValidation(block: () -> T): T {
        val previousDepth = storeManagedTransitionDepth.get() ?: 0
        storeManagedTransitionDepth.set(previousDepth + 1)
        return try {
            block()
        } finally {
            if (previousDepth == 0) {
                storeManagedTransitionDepth.remove()
            } else {
                storeManagedTransitionDepth.set(previousDepth)
            }
        }
    }

    internal fun requireValidEosJobSpecDelta(
        spec: IndexingJobSpec,
        changedDescriptorIndices: Set<Int>,
    ) {
        val errors = buildList {
            if (spec.specId != V2IndexingLedgerIds.jobSpecId(spec)) add("job specId mismatch")
            if (spec.provisionalParentSpecId == null ||
                !spec.provisionalParentSpecId.matches(Regex("^job-spec-v5-[0-9a-f]{64}$"))
            ) {
                add("invalid provisional parent job-spec ID")
            }
            try {
                V2DecodedEosLineage.requireValid(spec)
            } catch (error: Exception) {
                add(error.message ?: "invalid decoded EOS lineage")
            }
            changedDescriptorIndices.forEach { ordinal ->
                val descriptor = spec.tracks.getOrNull(ordinal)
                if (descriptor == null) {
                    add("changed descriptor ordinal is out of bounds")
                } else {
                    addAll(validateDescriptor(descriptor, ordinal))
                    if (descriptor.providerSnapshotGeneration !=
                        spec.providerSnapshot.libraryGeneration
                    ) {
                        add("track provider generation mismatch at $ordinal")
                    }
                }
            }
        }
        if (errors.isNotEmpty()) throw InvalidIndexingLedgerException(errors.joinToString("; "))
    }

    internal fun requireValidChangedTrack(
        track: IndexingTrackLedger,
        descriptor: SelectedTrackDescriptor,
        spec: IndexingJobSpec,
        ledgerUpdatedAtEpochMs: Long,
    ) {
        val errors = validateTrack(track, descriptor, spec, ledgerUpdatedAtEpochMs)
        if (errors.isNotEmpty()) throw InvalidIndexingLedgerException(errors.joinToString("; "))
    }

    internal fun requireValidChangedActivationEvidence(ledger: IndexingJobLedger) {
        val errors = buildList {
            val activation = ledger.activationEvidence
            if (ledger.state == IndexingJobState.COMPLETE) {
                if (activation == null) add("complete job lacks activated-generation evidence")
            } else if (activation != null) {
                add("activated-generation evidence exists before COMPLETE")
            }
            if (activation != null) {
                addAll(
                    validateActivationEvidence(
                        activation,
                        ledger.jobSpec,
                        ledger.updatedAtEpochMs,
                    ),
                )
            }
        }
        if (errors.isNotEmpty()) throw InvalidIndexingLedgerException(errors.joinToString("; "))
    }

    fun validate(ledger: IndexingJobLedger): List<String> = buildList {
        if (ledger.schemaVersion != V2IndexingLedgerSchema.VERSION) {
            add("unsupported ledger schema ${ledger.schemaVersion}")
        }
        val spec = ledger.jobSpec
        if (!SAFE_JOB_ID.matches(spec.jobId)) add("invalid jobId")
        if (spec.createdAtEpochMs < 0L) add("negative job creation time")
        if (ledger.updatedAtEpochMs < spec.createdAtEpochMs) add("ledger predates job")
        if (ledger.revision < 0L) add("negative revision")
        if (ledger.executionProfile !in V2IndexingExecutionProfile.entries) {
            add("invalid execution profile")
        }
        addAll(validateProviderSnapshot(spec.providerSnapshot))
        addAll(validateEmbeddingSpec(spec.embeddingSpec))
        addAll(validateTextRetrievalSpec(spec.textRetrievalSpec))
        if (spec.textRetrievalSpec.compatibleAudioEmbeddingSpecId != spec.embeddingSpec.specId ||
            spec.textRetrievalSpec.outputDimension != spec.embeddingSpec.outputDimension
        ) {
            add("text retrieval spec is not compatible with the audio embedding spec")
        }
        addAll(validateRuntimeFingerprint(spec.runtimeFingerprint))
        if (spec.baseGenerationId != null &&
            !spec.baseGenerationId.matches(Regex("^index-generation-v2-[0-9a-f]{64}$"))
        ) {
            add("invalid base generation ID")
        }

        if (spec.tracks.isEmpty()) add("job has no selected tracks")
        if (spec.specId != V2IndexingLedgerIds.jobSpecId(spec)) add("job specId mismatch")
        if (spec.provisionalParentSpecId != null &&
            !spec.provisionalParentSpecId.matches(Regex("^job-spec-v5-[0-9a-f]{64}$"))
        ) {
            add("invalid provisional parent job-spec ID")
        }
        val hasDecodedOrdinarySpan = spec.tracks.any { descriptor ->
            descriptor.finalizedAudioSpan.kind == V2ResolvedAudioSpanKind.WHOLE_FILE &&
                descriptor.finalizedAudioSpan.authority ==
                V2AudioSpanAuthority.DECODED_END_OF_STREAM
        }
        if (hasDecodedOrdinarySpan != (spec.provisionalParentSpecId != null)) {
            add("decoded ordinary span/job-spec provisional lineage mismatch")
        }
        try {
            V2DecodedEosLineage.requireValid(spec)
        } catch (error: Exception) {
            add(error.message ?: "invalid decoded EOS lineage")
        }

        val descriptorIds = spec.tracks.map { it.workId }
        if (descriptorIds.toSet().size != descriptorIds.size) add("duplicate workId in job spec")
        val powerampIds = spec.tracks.map { it.powerampFileId }
        if (powerampIds.toSet().size != powerampIds.size) {
            add("duplicate Poweramp row in job spec")
        }
        if (spec.providerSnapshot.acquisition.rowCount < spec.tracks.size) {
            add("provider snapshot row count is smaller than selected track count")
        }
        spec.tracks.forEachIndexed { ordinal, descriptor ->
            addAll(validateDescriptor(descriptor, ordinal))
            if (descriptor.providerSnapshotGeneration !=
                spec.providerSnapshot.libraryGeneration
            ) {
                add("track provider generation mismatch at $ordinal")
            }
        }

        val trackIds = ledger.tracks.map { it.workId }
        if (trackIds != descriptorIds) add("track ledger order/identity differs from immutable spec")
        if (trackIds.toSet().size != trackIds.size) add("duplicate workId in track ledger")
        ledger.tracks.zip(spec.tracks).forEach { (track, descriptor) ->
            addAll(validateTrack(track, descriptor, spec, ledger.updatedAtEpochMs))
        }
        val trackById = ledger.tracks.associateBy(IndexingTrackLedger::workId)
        if (spec.tracks.any { descriptor ->
                V2IndexingPlanFinalizationPolicy.isRunnableProvisional(
                    descriptor,
                    trackById.getValue(descriptor.workId),
                )
            } && ledger.tracks.any { track ->
                track.verifiedArtifacts.any { it.kind == VerifiedArtifactKind.DATABASE_COMMIT }
            }
        ) {
            add("database commit exists while a runnable ordinary span is provisional")
        }
        if (ledger.tracks.count { it.state.isActiveStage() } > 1) {
            add("more than one track owns the executor")
        }

        when (ledger.state) {
            IndexingJobState.INTERRUPTED,
            IndexingJobState.READY_TO_RESUME,
            -> if (ledger.recoveryPhase == null) add("recovery phase missing for ${ledger.state}")

            else -> if (ledger.recoveryPhase != null) {
                add("recovery phase set outside interrupted/resume state")
            }
        }

        if (ledger.state == IndexingJobState.ACTIVATING || ledger.state == IndexingJobState.COMPLETE) {
            val unresolved = ledger.tracks.filterNot { it.state.isResolvedForActivation() }
            if (unresolved.isNotEmpty()) add("${ledger.state} has unresolved tracks")
        }
        if (ledger.state in setOf(
                IndexingJobState.PAUSED,
                IndexingJobState.WAITING_FOR_INPUT,
                IndexingJobState.INTERRUPTED,
                IndexingJobState.READY_TO_RESUME,
                IndexingJobState.CANCELLED,
                IndexingJobState.ACTIVATING,
                IndexingJobState.COMPLETE,
            ) && ledger.tracks.any { it.state.isActiveStage() }
        ) {
            add("${ledger.state} retains an active track stage")
        }
        if (ledger.stateReason != null && ledger.stateReason.length > 2_048) {
            add("job state reason is too long")
        }
        if (ledger.state == IndexingJobState.COMPLETE &&
            ledger.tracks.any { it.state == IndexingTrackState.RETRYABLE_FAILURE }
        ) {
            add("complete job retains retryable failure")
        }
        val activation = ledger.activationEvidence
        if (ledger.state == IndexingJobState.COMPLETE) {
            if (activation == null) add("complete job lacks activated-generation evidence")
        } else if (activation != null) {
            add("activated-generation evidence exists before COMPLETE")
        }
        if (activation != null) addAll(validateActivationEvidence(activation, spec, ledger.updatedAtEpochMs))
    }

    private fun validateActivationEvidence(
        evidence: ActivatedGenerationEvidence,
        spec: IndexingJobSpec,
        ledgerUpdatedAtEpochMs: Long,
    ): List<String> = buildList {
        if (!evidence.generationId.matches(Regex("^index-generation-v2-[0-9a-f]{64}$"))) {
            add("invalid activated generation ID")
        }
        if (!evidence.activationBindingId.matches(Regex("^activation-binding-v3-[0-9a-f]{64}$"))) {
            add("invalid activation binding ID")
        }
        if (evidence.jobSpecId != spec.specId ||
            evidence.receiptEmbeddingSpecId != spec.embeddingSpec.specId ||
            evidence.textRetrievalSpecId != spec.textRetrievalSpec.specId ||
            evidence.baseGenerationId != spec.baseGenerationId ||
            evidence.rebuildDerivedIndexes != spec.rebuildDerivedIndexes
        ) {
            add("activated generation does not bind immutable job spec")
        }
        listOf(
            evidence.manifestSha256,
            evidence.databaseSha256,
            evidence.databaseContentSha256,
            evidence.orderedTrackSetSha256,
            evidence.stableTrackUidMappingSha256,
            evidence.embeddingSha256,
        ).forEach { digest ->
            if (!SHA256.matches(digest)) add("invalid activated-generation SHA-256")
        }
        if (evidence.graphSha256 != null && !SHA256.matches(evidence.graphSha256)) {
            add("invalid activated graph SHA-256")
        }
        if ((evidence.graphSha256 != null) != spec.rebuildDerivedIndexes) {
            add("activated graph presence disagrees with immutable job promise")
        }
        if (evidence.activatedAtEpochMs < spec.createdAtEpochMs ||
            evidence.activatedAtEpochMs > ledgerUpdatedAtEpochMs
        ) {
            add("activation timestamp is outside job lifetime")
        }
    }

    fun requireValidEmbeddingSpec(spec: EmbeddingSpecFingerprint) {
        val errors = validateEmbeddingSpec(spec)
        if (errors.isNotEmpty()) throw InvalidIndexingLedgerException(errors.joinToString("; "))
    }

    private fun validateEmbeddingSpec(spec: EmbeddingSpecFingerprint): List<String> = buildList {
        if (spec.preprocessingSpecId.isBlank()) add("blank preprocessing spec")
        if (spec.decoderPolicyId.isBlank()) add("blank decoder policy")
        if (spec.inferenceBackendPolicyId != V2IndexingWorkPolicy.INFERENCE_BACKEND_POLICY_ID) {
            add("unsupported inference backend policy")
        }
        if (spec.outputDimension <= 0) add("invalid embedding dimension")
        if (spec.modelArtifactSha256.isEmpty()) add("embedding spec has no model artifacts")
        spec.modelArtifactSha256.forEach { (name, hash) ->
            if (name.isBlank()) add("blank model artifact name")
            if (!SHA256.matches(hash)) add("invalid SHA-256 for model artifact $name")
        }
        val requiredArtifacts = setOf("mert", "clamp3_audio")
        if (spec.modelArtifactSha256.keys != requiredArtifacts) {
            add("embedding spec must bind exactly the MERT and CLaMP3 audio artifacts")
        }
        if (spec.specId != V2IndexingLedgerIds.embeddingSpecId(spec)) {
            add("embedding specId mismatch")
        }
    }

    fun requireValidTextRetrievalSpec(spec: TextRetrievalSpecFingerprint) {
        val errors = validateTextRetrievalSpec(spec)
        if (errors.isNotEmpty()) throw InvalidIndexingLedgerException(errors.joinToString("; "))
    }

    private fun validateTextRetrievalSpec(
        spec: TextRetrievalSpecFingerprint,
    ): List<String> = buildList {
        if (!spec.compatibleAudioEmbeddingSpecId.matches(
                Regex("^embedding-spec-v2-[0-9a-f]{64}$"),
            )
        ) {
            add("invalid compatible audio embedding spec ID")
        }
        if (!SHA256.matches(spec.textModelSha256)) add("invalid text model SHA-256")
        if (spec.tokenizerModelSha256 != V2IndexingWorkPolicy.TEXT_TOKENIZER_MODEL_SHA256) {
            add("unsupported SentencePiece model SHA-256")
        }
        if (spec.tokenizerPolicyId != V2IndexingWorkPolicy.TEXT_TOKENIZER_POLICY_ID) {
            add("unsupported text tokenizer policy")
        }
        if (spec.tokenizerRuntimeContractSha256 !=
            V2IndexingWorkPolicy.TEXT_TOKENIZER_RUNTIME_CONTRACT_SHA256
        ) {
            add("unsupported official SentencePiece runtime contract")
        }
        if (spec.outputSpaceId != V2IndexingWorkPolicy.TEXT_OUTPUT_SPACE_ID) {
            add("unsupported text output space")
        }
        if (spec.outputDimension <= 0) add("invalid text embedding dimension")
        if (spec.inferenceBackendPolicyId !=
            V2IndexingWorkPolicy.TEXT_INFERENCE_BACKEND_POLICY_ID
        ) {
            add("unsupported text inference backend policy")
        }
        if (spec.specId != V2IndexingLedgerIds.textRetrievalSpecId(spec)) {
            add("text retrieval specId mismatch")
        }
    }

    private fun validateRuntimeFingerprint(value: IndexingRuntimeFingerprint): List<String> = buildList {
        if (value.appVersionCode <= 0L) add("invalid app version code")
        if (value.appBuildId.isBlank()) add("blank app build id")
        if (value.decoderRuntimeId.isBlank()) add("blank decoder runtime id")
        if (value.platformFingerprint.isBlank()) add("blank platform fingerprint")
    }

    private fun validateDescriptor(
        descriptor: SelectedTrackDescriptor,
        expectedOrdinal: Int,
    ): List<String> = buildList {
        if (descriptor.ordinal != expectedOrdinal) add("non-contiguous track ordinal")
        if (descriptor.powerampFileId <= 0L) add("invalid Poweramp file id at $expectedOrdinal")
        if (descriptor.physicalPath.isBlank() || !descriptor.physicalPath.startsWith('/')) {
            add("physical path is not absolute at $expectedOrdinal")
        }
        if (descriptor.canonicalPath != V2IndexingLedgerIds.canonicalPath(descriptor.physicalPath)) {
            add("canonical path mismatch at $expectedOrdinal")
        }
        if (descriptor.normalizedMetadata.normalizationSpecId.isBlank()) {
            add("blank metadata normalization spec at $expectedOrdinal")
        }
        if (descriptor.normalizedMetadata.metadataKey.isBlank()) {
            add("blank normalized metadata key at $expectedOrdinal")
        }
        addAll(validateProviderRow(descriptor, expectedOrdinal))
        addAll(validateFinalizedSpan(descriptor, expectedOrdinal))
        addAll(validateFingerprint(descriptor.sourceFingerprint, "track $expectedOrdinal"))
        addAll(validateStableTrackSpanIdentity(descriptor, expectedOrdinal))
        val provisionalWorkId = descriptor.provisionalWorkId
        if (descriptor.finalizedAudioSpan.authority ==
            V2AudioSpanAuthority.PROVISIONAL_END_OF_STREAM
        ) {
            if (provisionalWorkId != null) add("provisional span already has work lineage at $expectedOrdinal")
        } else if (descriptor.finalizedAudioSpan.kind == V2ResolvedAudioSpanKind.WHOLE_FILE) {
            if (provisionalWorkId == null ||
                !provisionalWorkId.matches(Regex("^work-v4-[0-9a-f]{64}$"))
            ) {
                add("decoded EOS span lacks provisional work lineage at $expectedOrdinal")
            }
        } else if (provisionalWorkId != null) {
            add("CUE span has ordinary EOS work lineage at $expectedOrdinal")
        }
        if (descriptor.workId != V2IndexingLedgerIds.workId(descriptor)) {
            add("workId mismatch at $expectedOrdinal")
        }
    }

    private fun validateProviderSnapshot(
        snapshot: PowerampProviderSnapshotEvidence,
    ): List<String> = buildList {
        val generationPattern = Regex(
            "^poweramp-provider-snapshot-v[23]-sha256:[0-9a-f]{64}$",
        )
        if (!generationPattern.matches(snapshot.libraryGeneration)) {
            add("invalid Poweramp provider snapshot generation")
        }
        val acquisition = snapshot.acquisition
        if (acquisition.queryUri.isBlank()) add("blank provider query URI")
        if (acquisition.requestedColumns.isEmpty() ||
            acquisition.requestedColumns.any(String::isBlank) ||
            acquisition.requestedColumns.toSet().size != acquisition.requestedColumns.size
        ) {
            add("invalid requested provider columns")
        }
        if (acquisition.returnedColumns.isEmpty() ||
            acquisition.returnedColumns.any(String::isBlank) ||
            acquisition.returnedColumns.toSet().size != acquisition.returnedColumns.size
        ) {
            add("invalid returned provider columns")
        }
        if (acquisition.requestedColumns.any { requested ->
                !acquisition.returnedRequestedColumn(requested)
            }
        ) {
            add("provider cursor omitted requested columns")
        }
        if (acquisition.rowCount <= 0 || !acquisition.cursorExhaustedNormally) {
            add("provider acquisition is not a complete non-empty cursor result")
        }
    }

    private fun validateProviderRow(
        descriptor: SelectedTrackDescriptor,
        ordinal: Int,
    ): List<String> = buildList {
        val row = descriptor.providerRow
        if (row.powerampFileId != descriptor.powerampFileId) {
            add("provider row id mismatch at $ordinal")
        }
        if (row.providerPhysicalPath.isBlank() || !row.providerPhysicalPath.startsWith('/')) {
            add("provider path is not absolute at $ordinal")
        } else {
            val expectedProviderPathKey = try {
                V2StableProviderLexicalPathNormalizer.normalizeAbsolute(
                    row.providerPhysicalPath,
                )
            } catch (_: Exception) {
                null
            }
            if (row.physicalPath != expectedProviderPathKey) {
                add("provider lexical path mismatch at $ordinal")
            }
        }
        if (row.offsetWasNull && row.offsetMs != 0L) {
            add("null provider offset has non-zero value at $ordinal")
        }
        if (row.durationMs < 0L) {
            add("provider duration sentinel is not canonical at $ordinal")
        }
        if (row.cueSourceImageFolderId != null) {
            add("raw CUE source image entered immutable plan at $ordinal")
        }
        val providerSpan = descriptor.finalizedAudioSpan.providerSpan
        try {
            val expectedOffsetUs = Math.multiplyExact(row.offsetMs, 1_000L)
            val expectedDurationUs = Math.multiplyExact(row.durationMs, 1_000L)
            val expectedEndUs = Math.addExact(expectedOffsetUs, expectedDurationUs)
            if (providerSpan != V2ProviderSpanEvidence(
                    expectedOffsetUs,
                    expectedDurationUs,
                    expectedEndUs,
                )
            ) {
                add("provider millisecond/microsecond evidence mismatch at $ordinal")
            }
        } catch (_: ArithmeticException) {
            add("provider span overflows at $ordinal")
        }
    }

    private fun validateFinalizedSpan(
        descriptor: SelectedTrackDescriptor,
        ordinal: Int,
    ): List<String> = buildList {
        val span = descriptor.finalizedAudioSpan
        val container = span.container
        val unavailableContainerDuration =
            V2UnknownDurationOrdinarySpanPolicy.hasUnavailableDuration(container)
        val unresolvedOrdinary = V2UnknownDurationOrdinarySpanPolicy.isUnresolved(span)
        val validContainerDurationEvidence =
            (container.durationUsEstimate > 0L &&
                container.durationEstimateSource != V2DurationEstimateSource.UNAVAILABLE) ||
                (unavailableContainerDuration && span.kind == V2ResolvedAudioSpanKind.WHOLE_FILE)
        if (container.physicalPath != descriptor.physicalPath ||
            container.audioTrackIndex < 0 || !validContainerDurationEvidence ||
            container.sampleRateHz <= 0 || container.channelCount <= 0 ||
            !container.mime.startsWith("audio/")
        ) {
            add("invalid container evidence at $ordinal")
        }
        if (unresolvedOrdinary) {
            if (descriptor.providerRow.durationMs != 0L ||
                span.startUs != 0L || span.endExclusiveUs != 0L ||
                span.startSourceSample != 0L || span.endSourceSampleExclusive != 0L ||
                span.sourceSampleCount != 0L || span.exactSampleCount24k != 0L ||
                span.expectedWork != V2UnknownDurationOrdinarySpanPolicy.unresolvedWork
            ) {
                add("unknown-duration ordinary span invented provisional work at $ordinal")
            }
        } else {
            if (span.startUs < 0L || span.endExclusiveUs <= span.startUs) {
                add("invalid finalized microsecond span at $ordinal")
            }
            if (span.startSourceSample < 0L ||
                span.endSourceSampleExclusive <= span.startSourceSample ||
                span.sourceSampleCount <= 0L || span.exactSampleCount24k <= 0L
            ) {
                add("invalid finalized sample span at $ordinal")
            }
        }

        when (span.kind) {
            V2ResolvedAudioSpanKind.WHOLE_FILE -> {
                if (span.authority !in setOf(
                        V2AudioSpanAuthority.PROVISIONAL_END_OF_STREAM,
                        V2AudioSpanAuthority.DECODED_END_OF_STREAM,
                    ) || span.executionBoundaryRequirement !=
                    V2ExecutionBoundaryRequirement.VERIFY_END_OF_STREAM_AND_RECONCILE ||
                    span.startUs != 0L ||
                    (span.authority == V2AudioSpanAuthority.PROVISIONAL_END_OF_STREAM &&
                        !unresolvedOrdinary && span.endExclusiveUs != container.durationUsEstimate)
                ) {
                    add("ordinary span lacks container authority/EOS reconciliation at $ordinal")
                }
                if (span.cueClassification.nonZeroOffsetRowIds.isNotEmpty() ||
                    span.cueClassification.rawSourceImageRowIds.isNotEmpty()
                ) {
                    add("ordinary span has structural CUE evidence at $ordinal")
                }
            }

            V2ResolvedAudioSpanKind.LOGICAL_CUE -> {
                if (span.authority != V2AudioSpanAuthority.PROVIDER_CUE_HALF_OPEN_SPAN ||
                    span.executionBoundaryRequirement !=
                        V2ExecutionBoundaryRequirement.ENFORCE_PROVIDER_HALF_OPEN_SPAN ||
                    span.startUs != span.providerSpan.offsetUs ||
                    span.endExclusiveUs != span.providerSpan.endExclusiveUs ||
                    span.startUs < 0L ||
                    descriptor.providerRow.durationMs <= 0L
                ) {
                    add("CUE span lacks exact provider half-open authority at $ordinal")
                }
                if (span.cueClassification.nonZeroOffsetRowIds.isEmpty() &&
                    span.cueClassification.rawSourceImageRowIds.isEmpty()
                ) {
                    add("CUE span lacks structural group evidence at $ordinal")
                }
            }
        }

        val cue = span.cueClassification
        if (cue.providerGroupRowCount <= 0 || cue.logicalRowCount <= 0 ||
            cue.logicalRowCount > cue.providerGroupRowCount ||
            cue.nonZeroOffsetRowIds.size > cue.logicalRowCount ||
            cue.logicalRowCount + cue.rawSourceImageRowIds.size > cue.providerGroupRowCount ||
            cue.nonZeroOffsetRowIds.any { it in cue.rawSourceImageRowIds } ||
            !isSortedUniquePositive(cue.nonZeroOffsetRowIds) ||
            !isSortedUniquePositive(cue.rawSourceImageRowIds)
        ) {
            add("invalid CUE classification evidence at $ordinal")
        }

        if (!unresolvedOrdinary) {
            try {
                val expectedStartSample = V2AudioSpanMath.sampleAtOrAfter(
                    span.startUs,
                    container.sampleRateHz,
                )
                val expectedEndSample = V2AudioSpanMath.sampleAtOrAfter(
                    span.endExclusiveUs,
                    container.sampleRateHz,
                )
                val expectedSourceSamples = Math.subtractExact(expectedEndSample, expectedStartSample)
                val expected24k = V2AudioSpanMath.resampledLength(
                    expectedSourceSamples,
                    container.sampleRateHz,
                    V2AudioSpanMath.TARGET_SAMPLE_RATE_HZ,
                )
                val expectedWork = V2AudioSpanMath.expectedWorkFor24kSamples(expected24k)
                if (span.startSourceSample != expectedStartSample ||
                    span.endSourceSampleExclusive != expectedEndSample ||
                    span.sourceSampleCount != expectedSourceSamples ||
                    span.exactSampleCount24k != expected24k
                ) {
                    add("finalized time/sample coordinates disagree at $ordinal")
                }
                if (span.expectedWork != expectedWork ||
                    expectedWork.mertWindows <= 0 || expectedWork.clampSegments <= 0
                ) {
                    add("invalid exact expected work at $ordinal")
                }
            } catch (_: Exception) {
                add("finalized acoustic arithmetic failed at $ordinal")
            }
        }
    }

    private fun isSortedUniquePositive(values: List<Long>): Boolean =
        values.all { it > 0L } && values == values.distinct().sorted()

    private fun validateFingerprint(value: SourceFingerprint, label: String): List<String> = buildList {
        if (value.fingerprintSpecId.isBlank()) add("blank fingerprint spec for $label")
        if (value.sizeBytes <= 0L) add("non-positive source size for $label")
        if (value.lastModifiedEpochMs != null && value.lastModifiedEpochMs < 0L) {
            add("negative source mtime for $label")
        }
        if (value.sampledContentSha256 == null && value.fullContentSha256 == null) {
            add("source fingerprint has no content digest for $label")
        }
        if (value.fileKey?.isBlank() == true) add("blank source file key for $label")
        if (value.sampledContentSha256 != null && !SHA256.matches(value.sampledContentSha256)) {
            add("invalid sampled source SHA-256 for $label")
        }
        if (value.fullContentSha256 != null && !SHA256.matches(value.fullContentSha256)) {
            add("invalid full source SHA-256 for $label")
        }
    }

    private fun validateStableTrackSpanIdentity(
        descriptor: SelectedTrackDescriptor,
        ordinal: Int,
    ): List<String> = buildList {
        val identity = descriptor.stableTrackSpanIdentity
        if (identity.identitySpecId != V2IndexingLedgerIds.STABLE_TRACK_SPAN_IDENTITY_SPEC_ID) {
            add("unsupported stable track-span identity spec at $ordinal")
        }
        if (!identity.stableTrackSpanId.matches(Regex("^stable-track-span-v1-[0-9a-f]{64}$"))) {
            add("invalid stable track-span id at $ordinal")
        }
        if (!SHA256.matches(identity.contentSha256)) {
            add("invalid stable content SHA-256 at $ordinal")
        }
        val unresolvedOrdinary =
            V2UnknownDurationOrdinarySpanPolicy.isUnresolved(descriptor.finalizedAudioSpan)
        val validCoordinates = identity.startSourceSample >= 0L &&
            (identity.endSourceSampleExclusive > identity.startSourceSample ||
                (unresolvedOrdinary && identity.startSourceSample == 0L &&
                    identity.endSourceSampleExclusive == 0L))
        if (identity.contentFingerprintSpecId.isBlank() || identity.sourceSizeBytes <= 0L ||
            identity.sourceSampleRateHz <= 0 || !validCoordinates
        ) {
            add("invalid stable track-span coordinates at $ordinal")
        }
        val expected = try {
            V2IndexingLedgerIds.stableTrackSpanIdentity(
                sourceFingerprint = descriptor.sourceFingerprint,
                span = descriptor.finalizedAudioSpan,
            )
        } catch (_: Exception) {
            null
        }
        if (identity != expected) {
            add("stable track-span identity disagrees with source/span evidence at $ordinal")
        }
    }

    private fun validateTrack(
        track: IndexingTrackLedger,
        descriptor: SelectedTrackDescriptor,
        spec: IndexingJobSpec,
        ledgerUpdatedAtEpochMs: Long,
    ): List<String> = buildList {
        if (track.workId != descriptor.workId) add("track workId/spec mismatch")
        if (track.updatedAtEpochMs < spec.createdAtEpochMs) add("track predates job")
        if (track.updatedAtEpochMs > ledgerUpdatedAtEpochMs) add("track update is ahead of ledger")
        if (track.attemptCount < 0) add("negative attempt count for ${track.workId}")
        if (track.currentAttemptNumber != null &&
            (track.currentAttemptNumber <= 0 || track.currentAttemptNumber > track.attemptCount)
        ) {
            add("invalid current attempt for ${track.workId}")
        }

        val expectedCheckpoint = track.state.fixedCheckpointOrNull()
        if (expectedCheckpoint != null && expectedCheckpoint != track.checkpoint) {
            add("checkpoint/state mismatch for ${track.workId}")
        }
        if (track.state.isActiveStage() && track.currentAttemptNumber == null) {
            add("active stage has no attempt for ${track.workId}")
        }
        if (track.state == IndexingTrackState.COMMITTED ||
            track.state == IndexingTrackState.SKIPPED_BY_USER
        ) {
            if (track.currentAttemptNumber != null) add("terminal track has active attempt")
        }

        track.stageProgress?.let { progress ->
            addAll(
                validateStageProgress(
                    progress,
                    track,
                    descriptor,
                    spec.embeddingSpec.specId,
                    spec.createdAtEpochMs,
                ),
            )
        }

        val artifactKinds = track.verifiedArtifacts.map { it.kind }
        if (artifactKinds.toSet().size != artifactKinds.size) {
            add("duplicate verified artifact kind for ${track.workId}")
        }
        track.verifiedArtifacts.forEach { artifact ->
            addAll(
                validateArtifact(
                    artifact,
                    descriptor,
                    spec.embeddingSpec.specId,
                    spec.createdAtEpochMs,
                    track.updatedAtEpochMs,
                ),
            )
        }
        if (V2IndexingPlanFinalizationPolicy.isProvisionalOrdinary(descriptor) &&
            (track.checkpoint.rank() >= TrackCheckpoint.MERT_COMPLETE.rank() ||
                track.state in setOf(
                    IndexingTrackState.MERT_COMPLETE,
                    IndexingTrackState.CLAMPING,
                    IndexingTrackState.CLAMP_COMPLETE,
                    IndexingTrackState.COMMITTING,
                    IndexingTrackState.COMMITTED,
                ))
        ) {
            add("provisional ordinary span advanced to MERT for ${track.workId}")
        }
        addAll(validateCheckpointArtifacts(track, descriptor))

        val failureIds = track.failures.map { it.failureId }
        if (failureIds.toSet().size != failureIds.size) add("duplicate failure id for ${track.workId}")
        track.failures.forEach { failure ->
            addAll(validateFailure(failure, track, descriptor, spec))
        }

        if (track.state == IndexingTrackState.RETRYABLE_FAILURE ||
            track.state == IndexingTrackState.BLOCKED_FAILURE
        ) {
            val active = track.failures.singleOrNull { it.failureId == track.activeFailureId }
            if (active == null) {
                add("failure state lacks one active failure for ${track.workId}")
            } else {
                val expectedDisposition = if (track.state == IndexingTrackState.RETRYABLE_FAILURE) {
                    FailureDisposition.RETRYABLE
                } else {
                    FailureDisposition.BLOCKED
                }
                if (active.disposition != expectedDisposition) {
                    add("failure disposition/state mismatch for ${track.workId}")
                }
                if (active.resumeFrom != track.checkpoint) {
                    add("active failure checkpoint mismatch for ${track.workId}")
                }
            }
            if (track.currentAttemptNumber != null) add("failed track retains active attempt")
        } else if (track.activeFailureId != null) {
            add("non-failure state has active failure for ${track.workId}")
        }
    }

    private fun validateArtifact(
        artifact: VerifiedArtifact,
        descriptor: SelectedTrackDescriptor,
        embeddingSpecId: String,
        jobCreatedAtEpochMs: Long,
        trackUpdatedAtEpochMs: Long,
    ): List<String> = buildList {
        if (artifact.storageKey.isBlank()) add("blank artifact storage key for ${descriptor.workId}")
        if (artifact.byteLength <= 0L) add("empty artifact for ${descriptor.workId}")
        if (!SHA256.matches(artifact.sha256)) add("invalid artifact SHA-256 for ${descriptor.workId}")
        if (artifact.plannedUnits <= 0 || artifact.completedUnits != artifact.plannedUnits) {
            add("artifact is not complete for ${descriptor.workId}")
        }
        if (artifact.embeddingSpecId != embeddingSpecId) {
            add("artifact embedding spec mismatch for ${descriptor.workId}")
        }
        if (artifact.sourceFingerprint != descriptor.sourceFingerprint) {
            add("artifact source fingerprint mismatch for ${descriptor.workId}")
        }
        if (artifact.verifiedAtEpochMs < jobCreatedAtEpochMs ||
            artifact.verifiedAtEpochMs > trackUpdatedAtEpochMs
        ) {
            add("artifact verification time is outside track lifetime")
        }
        val expectedUnits = when (artifact.kind) {
            VerifiedArtifactKind.MERT_FEATURES -> descriptor.expectedWork.mertWindows
            VerifiedArtifactKind.CLAMP_VECTOR -> descriptor.expectedWork.clampSegments
            VerifiedArtifactKind.DATABASE_COMMIT -> 1
        }
        if (artifact.plannedUnits != expectedUnits) {
            add("artifact planned-unit mismatch for ${descriptor.workId}")
        }
        val expectedByteLength = when (artifact.kind) {
            VerifiedArtifactKind.MERT_FEATURES -> try {
                Math.multiplyExact(expectedUnits.toLong(), V2_CLAMP3_BLOB_BYTES.toLong())
            } catch (_: ArithmeticException) {
                -1L
            }
            VerifiedArtifactKind.CLAMP_VECTOR,
            VerifiedArtifactKind.DATABASE_COMMIT,
            -> V2_CLAMP3_BLOB_BYTES.toLong()
        }
        if (artifact.byteLength != expectedByteLength) {
            add("artifact format length mismatch for ${descriptor.workId}")
        }
        if (artifact.kind == VerifiedArtifactKind.MERT_FEATURES) {
            addAll(validateExecutionBoundary(artifact.executionBoundary, descriptor))
        } else if (artifact.executionBoundary != null) {
            add("non-MERT artifact carries execution-boundary evidence for ${descriptor.workId}")
        }
    }

    private fun validateExecutionBoundary(
        boundary: VerifiedExecutionBoundaryEvidence?,
        descriptor: SelectedTrackDescriptor,
    ): List<String> = buildList {
        val span = descriptor.finalizedAudioSpan
        if (boundary == null) {
            add("MERT artifact lacks verified execution boundary for ${descriptor.workId}")
            return@buildList
        }
        if (boundary.requirement != span.executionBoundaryRequirement ||
            boundary.observedStartSourceSample != span.startSourceSample ||
            boundary.observedEndSourceSampleExclusive != span.endSourceSampleExclusive ||
            boundary.observedSourceSampleCount != span.sourceSampleCount ||
            boundary.exactSampleCount24k != span.exactSampleCount24k
        ) {
            add("observed decoder boundary disagrees with plan for ${descriptor.workId}")
        }
        when (span.executionBoundaryRequirement) {
            V2ExecutionBoundaryRequirement.VERIFY_END_OF_STREAM_AND_RECONCILE -> {
                if (!boundary.endOfStreamReached || boundary.providerBoundaryEnforced) {
                    add("ordinary MERT artifact lacks EOS reconciliation for ${descriptor.workId}")
                }
            }

            V2ExecutionBoundaryRequirement.ENFORCE_PROVIDER_HALF_OPEN_SPAN -> {
                if (!boundary.providerBoundaryEnforced) {
                    add("CUE MERT artifact lacks enforced provider boundary for ${descriptor.workId}")
                }
            }
        }
    }

    private fun validateStageProgress(
        progress: VerifiedStageProgress,
        track: IndexingTrackLedger,
        descriptor: SelectedTrackDescriptor,
        embeddingSpecId: String,
        jobCreatedAtEpochMs: Long,
    ): List<String> = buildList {
        if (progress.stage != track.checkpoint.nextResumableStage()) {
            add("partial stage/checkpoint mismatch for ${track.workId}")
        }
        if (track.state.isActiveStage() && progress.stage != track.state.stageOrNull()) {
            add("partial stage/active state mismatch for ${track.workId}")
        }
        if (track.state == IndexingTrackState.COMMITTED ||
            track.state == IndexingTrackState.SKIPPED_BY_USER
        ) {
            add("terminal track retains partial stage progress")
        }
        if (progress.storageKey.isBlank() || progress.resumeCursor.isBlank()) {
            add("partial stage lacks storage/cursor evidence for ${track.workId}")
        }
        if (progress.byteLength <= 0L || !SHA256.matches(progress.sha256)) {
            add("invalid partial artifact for ${track.workId}")
        }
        val expectedUnits = when (progress.stage) {
            IndexingStage.DECODE_AND_MERT -> descriptor.expectedWork.mertWindows
            IndexingStage.CLAMP3 -> descriptor.expectedWork.clampSegments
            else -> -1
        }
        if (progress.plannedUnits != expectedUnits ||
            progress.completedUnits <= 0 ||
            progress.completedUnits >= progress.plannedUnits
        ) {
            add("invalid partial stage units for ${track.workId}")
        }
        if (progress.embeddingSpecId != embeddingSpecId ||
            progress.sourceFingerprint != descriptor.sourceFingerprint
        ) {
            add("partial stage identity mismatch for ${track.workId}")
        }
        if (progress.verifiedAtEpochMs < jobCreatedAtEpochMs ||
            progress.verifiedAtEpochMs > track.updatedAtEpochMs
        ) {
            add("partial verification time is outside track lifetime")
        }
    }

    private fun validateCheckpointArtifacts(
        track: IndexingTrackLedger,
        descriptor: SelectedTrackDescriptor,
    ): List<String> = buildList {
        val kinds = track.verifiedArtifacts.map { it.kind }.toSet()
        val checkpointRank = track.checkpoint.rank()
        val mertRequired = checkpointRank >= TrackCheckpoint.MERT_COMPLETE.rank()
        val clampRequired = checkpointRank >= TrackCheckpoint.CLAMP_COMPLETE.rank()
        val commitRequired = checkpointRank >= TrackCheckpoint.COMMITTED.rank()
        if ((VerifiedArtifactKind.MERT_FEATURES in kinds) != mertRequired) {
            add("MERT artifact/checkpoint mismatch for ${descriptor.workId}")
        }
        if ((VerifiedArtifactKind.CLAMP_VECTOR in kinds) != clampRequired) {
            add("CLaMP artifact/checkpoint mismatch for ${descriptor.workId}")
        }
        if ((VerifiedArtifactKind.DATABASE_COMMIT in kinds) != commitRequired) {
            add("commit artifact/checkpoint mismatch for ${descriptor.workId}")
        }
    }

    private fun validateFailure(
        failure: TrackFailureAggregate,
        track: IndexingTrackLedger,
        descriptor: SelectedTrackDescriptor,
        spec: IndexingJobSpec,
    ): List<String> = buildList {
        val policy = V2TrackFailurePolicies.forCode(failure.code)
        if (failure.category != policy.category) add("failure category/code mismatch")
        val exhausted = policy.maxAutomaticRetries?.let { failure.occurrences > it } ?: false
        val expectedDisposition = if (exhausted) {
            FailureDisposition.BLOCKED
        } else {
            policy.initialDisposition
        }
        val expectedTrigger = if (exhausted) policy.exhaustedTrigger else policy.retryTrigger
        if (failure.disposition != expectedDisposition || failure.retryTrigger != expectedTrigger) {
            add("failure retry policy/evidence mismatch")
        }
        if (failure.failureId != V2IndexingLedgerIds.failureId(
                track.workId,
                failure.code,
                failure.stage,
                failure.sourceFingerprint,
                failure.embeddingSpecId,
                failure.appBuildId,
            )
        ) {
            add("failure id mismatch for ${track.workId}")
        }
        if (failure.occurrences <= 0) add("non-positive failure occurrence count")
        if (failure.firstOccurredAtEpochMs > failure.lastOccurredAtEpochMs) {
            add("failure timestamps reversed")
        }
        if (failure.firstOccurredAtEpochMs < spec.createdAtEpochMs ||
            failure.lastOccurredAtEpochMs > track.updatedAtEpochMs
        ) {
            add("failure timestamps are outside track lifetime")
        }
        if (failure.firstAttemptNumber <= 0 ||
            failure.firstAttemptNumber > failure.lastAttemptNumber ||
            failure.lastAttemptNumber > track.attemptCount
        ) {
            add("invalid failure attempt range for ${track.workId}")
        }
        if (failure.diagnostic.isBlank() || failure.diagnostic.length > 2_048) {
            add("invalid failure diagnostic for ${track.workId}")
        }
        if (failure.sourceFingerprint != descriptor.sourceFingerprint) {
            add("failure source fingerprint mismatch for ${track.workId}")
        }
        if (failure.embeddingSpecId != spec.embeddingSpec.specId) {
            add("failure embedding spec mismatch for ${track.workId}")
        }
        if (failure.appBuildId != spec.runtimeFingerprint.appBuildId) {
            add("failure app build mismatch for ${track.workId}")
        }
        if (failure.resumeFrom.rank() > track.checkpoint.rank() &&
            failure.failureId == track.activeFailureId
        ) {
            add("active failure resume checkpoint is ahead of track checkpoint")
        }
    }
}

internal fun TrackCheckpoint.rank(): Int = when (this) {
    TrackCheckpoint.QUEUED -> 0
    TrackCheckpoint.PREFLIGHTED -> 1
    TrackCheckpoint.MERT_COMPLETE -> 2
    TrackCheckpoint.CLAMP_COMPLETE -> 3
    TrackCheckpoint.COMMITTED -> 4
}

internal fun IndexingTrackState.fixedCheckpointOrNull(): TrackCheckpoint? = when (this) {
    IndexingTrackState.QUEUED,
    IndexingTrackState.PREFLIGHTING,
    -> TrackCheckpoint.QUEUED

    IndexingTrackState.PREFLIGHTED,
    IndexingTrackState.DECODING,
    -> TrackCheckpoint.PREFLIGHTED

    IndexingTrackState.MERT_COMPLETE,
    IndexingTrackState.CLAMPING,
    -> TrackCheckpoint.MERT_COMPLETE

    IndexingTrackState.CLAMP_COMPLETE,
    IndexingTrackState.COMMITTING,
    -> TrackCheckpoint.CLAMP_COMPLETE

    IndexingTrackState.COMMITTED -> TrackCheckpoint.COMMITTED
    IndexingTrackState.RETRYABLE_FAILURE,
    IndexingTrackState.BLOCKED_FAILURE,
    IndexingTrackState.SKIPPED_BY_USER,
    -> null
}

internal fun IndexingTrackState.isActiveStage(): Boolean = when (this) {
    IndexingTrackState.PREFLIGHTING,
    IndexingTrackState.DECODING,
    IndexingTrackState.CLAMPING,
    IndexingTrackState.COMMITTING,
    -> true

    else -> false
}

internal fun IndexingTrackState.isResolvedForActivation(): Boolean = when (this) {
    IndexingTrackState.COMMITTED,
    IndexingTrackState.BLOCKED_FAILURE,
    IndexingTrackState.SKIPPED_BY_USER,
    -> true

    else -> false
}

internal fun TrackCheckpoint.nextResumableStage(): IndexingStage? = when (this) {
    TrackCheckpoint.QUEUED -> null
    TrackCheckpoint.PREFLIGHTED -> IndexingStage.DECODE_AND_MERT
    TrackCheckpoint.MERT_COMPLETE -> IndexingStage.CLAMP3
    TrackCheckpoint.CLAMP_COMPLETE,
    TrackCheckpoint.COMMITTED,
    -> null
}

internal fun IndexingTrackState.stageOrNull(): IndexingStage? = when (this) {
    IndexingTrackState.PREFLIGHTING -> IndexingStage.PREFLIGHT
    IndexingTrackState.DECODING -> IndexingStage.DECODE_AND_MERT
    IndexingTrackState.CLAMPING -> IndexingStage.CLAMP3
    IndexingTrackState.COMMITTING -> IndexingStage.DATABASE_COMMIT
    else -> null
}
