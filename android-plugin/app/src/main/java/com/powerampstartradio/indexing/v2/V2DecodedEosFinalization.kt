package com.powerampstartradio.indexing.v2

/** Decoder evidence produced by the single PCM materialization pass for an ordinary file. */
data class V2DecodedEosEvidence(
    val sourceSampleRateHz: Int,
    val observedStartSourceSample: Long,
    val observedEndSourceSampleExclusive: Long,
    val observedSourceSampleCount: Long,
    val exactSampleCount24k: Long,
    val endOfStreamReached: Boolean,
)

data class V2DecodedEosPlanUpdate(
    val ledger: IndexingJobLedger,
    val workIdRemap: Map<String, String>,
)

/** Pure, one-way conversion from an estimate-backed ordinary span to decoded EOS authority. */
object V2DecodedEosSpanFinalizer {
    fun finalize(
        span: FinalizedAudioSpanEvidence,
        evidence: V2DecodedEosEvidence,
    ): FinalizedAudioSpanEvidence {
        require(span.kind == V2ResolvedAudioSpanKind.WHOLE_FILE) {
            "only ordinary whole-file spans use decoded EOS finalization"
        }
        require(span.executionBoundaryRequirement ==
            V2ExecutionBoundaryRequirement.VERIFY_END_OF_STREAM_AND_RECONCILE
        ) { "ordinary span has the wrong execution-boundary requirement" }
        require(evidence.endOfStreamReached) { "physical decoder EOS was not reached" }
        require(evidence.sourceSampleRateHz == span.container.sampleRateHz) {
            "decoder sample rate changed during EOS finalization"
        }
        require(evidence.observedStartSourceSample == span.startSourceSample) {
            "decoder EOS evidence starts at the wrong native sample"
        }
        require(evidence.observedEndSourceSampleExclusive > evidence.observedStartSourceSample) {
            "decoder EOS evidence is empty"
        }
        require(evidence.observedSourceSampleCount ==
            evidence.observedEndSourceSampleExclusive - evidence.observedStartSourceSample
        ) { "decoder EOS coordinates and sample count disagree" }
        val exact24k = V2AudioSpanMath.resampledLength(
            evidence.observedSourceSampleCount,
            evidence.sourceSampleRateHz,
            V2AudioSpanMath.TARGET_SAMPLE_RATE_HZ,
        )
        require(evidence.exactSampleCount24k == exact24k) {
            "decoded EOS 24 kHz count disagrees with the pinned resampler"
        }
        val expectedWork = V2AudioSpanMath.expectedWorkFor24kSamples(exact24k)
        require(expectedWork.mertWindows > 0 && expectedWork.clampSegments > 0) {
            "decoded EOS audio is below the one-second indexing floor"
        }
        val canonicalEndUs = V2AudioSpanMath.canonicalTimeUsForSampleBoundary(
            evidence.observedEndSourceSampleExclusive,
            evidence.sourceSampleRateHz,
        )
        val finalized = span.copy(
            authority = V2AudioSpanAuthority.DECODED_END_OF_STREAM,
            endExclusiveUs = canonicalEndUs,
            endSourceSampleExclusive = evidence.observedEndSourceSampleExclusive,
            sourceSampleCount = evidence.observedSourceSampleCount,
            exactSampleCount24k = exact24k,
            expectedWork = expectedWork,
        )
        V2DecodedEosPublicationPolicy.requirePublishable(finalized)
        if (span.authority == V2AudioSpanAuthority.DECODED_END_OF_STREAM) {
            require(span == finalized) { "decoded EOS evidence changed after finalization" }
        } else {
            require(span.authority == V2AudioSpanAuthority.PROVISIONAL_END_OF_STREAM) {
                "ordinary span cannot transition from ${span.authority} to decoded EOS"
            }
        }
        return finalized
    }
}

/** Gates inference and database mutation on authoritative acoustic work. */
object V2IndexingPlanFinalizationPolicy {
    fun isProvisionalOrdinary(descriptor: SelectedTrackDescriptor): Boolean =
        descriptor.finalizedAudioSpan.kind == V2ResolvedAudioSpanKind.WHOLE_FILE &&
            descriptor.finalizedAudioSpan.authority ==
            V2AudioSpanAuthority.PROVISIONAL_END_OF_STREAM

    fun isRunnableProvisional(
        descriptor: SelectedTrackDescriptor,
        track: IndexingTrackLedger,
    ): Boolean = isProvisionalOrdinary(descriptor) && track.state !in setOf(
        IndexingTrackState.BLOCKED_FAILURE,
        IndexingTrackState.SKIPPED_BY_USER,
    )

    fun requireMertReady(descriptor: SelectedTrackDescriptor) {
        require(!isProvisionalOrdinary(descriptor)) {
            "MERT cannot run before physical EOS finalizes ${descriptor.workId}"
        }
    }

    fun requireStagingDatabaseReady(ledger: IndexingJobLedger) {
        val trackById = ledger.tracks.associateBy(IndexingTrackLedger::workId)
        val provisional = ledger.jobSpec.tracks.firstOrNull { descriptor ->
            isRunnableProvisional(descriptor, trackById.getValue(descriptor.workId))
        }
        require(provisional == null) {
            "staging database cannot bind while ${provisional?.workId} has provisional EOS work"
        }
    }
}

/** Reconstructs and verifies the one all-provisional ancestor of a partially finalized plan. */
internal object V2DecodedEosLineage {
    /**
     * Returns the one preflight-published spec identity after verifying the current content hash
     * and every decoded-EOS rewrite back to its reconstructable all-provisional ancestor.
     */
    fun requirePreflightSpecId(spec: IndexingJobSpec): String {
        require(spec.specId == V2IndexingLedgerIds.jobSpecId(spec)) {
            "job specId mismatch"
        }
        requireValid(spec)
        return spec.provisionalParentSpecId ?: spec.specId
    }

    fun requireValid(spec: IndexingJobSpec) {
        val hasDecodedOrdinary = spec.tracks.any { descriptor ->
            descriptor.finalizedAudioSpan.kind == V2ResolvedAudioSpanKind.WHOLE_FILE &&
                descriptor.finalizedAudioSpan.authority ==
                V2AudioSpanAuthority.DECODED_END_OF_STREAM
        }
        if (!hasDecodedOrdinary) {
            require(spec.provisionalParentSpecId == null) {
                "an all-provisional job cannot claim EOS parent lineage"
            }
            return
        }

        val provisionalTracks = spec.tracks.map(::reconstructProvisionalDescriptor)
        var ancestor = spec.copy(
            specId = "",
            provisionalParentSpecId = null,
            tracks = provisionalTracks,
        )
        ancestor = ancestor.copy(specId = V2IndexingLedgerIds.jobSpecId(ancestor))
        require(spec.provisionalParentSpecId == ancestor.specId) {
            "decoded EOS job-spec parent does not match its reconstructable provisional ancestor"
        }
    }

    private fun reconstructProvisionalDescriptor(
        descriptor: SelectedTrackDescriptor,
    ): SelectedTrackDescriptor {
        val span = descriptor.finalizedAudioSpan
        if (span.kind != V2ResolvedAudioSpanKind.WHOLE_FILE) {
            require(descriptor.provisionalWorkId == null) {
                "CUE work cannot claim ordinary EOS lineage"
            }
            return descriptor
        }
        if (span.authority == V2AudioSpanAuthority.PROVISIONAL_END_OF_STREAM) {
            require(descriptor.provisionalWorkId == null) {
                "provisional ordinary work cannot already claim finalization lineage"
            }
            return descriptor
        }
        require(span.authority == V2AudioSpanAuthority.DECODED_END_OF_STREAM) {
            "whole-file work has unsupported EOS authority"
        }

        val provisionalSpan = if (
            V2UnknownDurationOrdinarySpanPolicy.hasUnavailableDuration(span.container)
        ) {
            require(descriptor.providerRow.durationMs == 0L) {
                "decoded unknown-duration lineage has non-canonical provider duration evidence"
            }
            span.copy(
                authority = V2AudioSpanAuthority.PROVISIONAL_END_OF_STREAM,
                startUs = 0L,
                endExclusiveUs = 0L,
                startSourceSample = 0L,
                endSourceSampleExclusive = 0L,
                sourceSampleCount = 0L,
                exactSampleCount24k = 0L,
                expectedWork = V2UnknownDurationOrdinarySpanPolicy.unresolvedWork,
            )
        } else {
            val provisionalEndUs = span.container.durationUsEstimate
            val provisionalStartSample = V2AudioSpanMath.sampleAtOrAfter(
                0L,
                span.container.sampleRateHz,
            )
            val provisionalEndSample = V2AudioSpanMath.sampleAtOrAfter(
                provisionalEndUs,
                span.container.sampleRateHz,
            )
            val provisionalSourceSamples = Math.subtractExact(
                provisionalEndSample,
                provisionalStartSample,
            )
            require(provisionalSourceSamples > 0L) {
                "decoded EOS lineage reconstructs an empty provisional span"
            }
            val provisional24k = V2AudioSpanMath.resampledLength(
                provisionalSourceSamples,
                span.container.sampleRateHz,
                V2AudioSpanMath.TARGET_SAMPLE_RATE_HZ,
            )
            span.copy(
                authority = V2AudioSpanAuthority.PROVISIONAL_END_OF_STREAM,
                startUs = 0L,
                endExclusiveUs = provisionalEndUs,
                startSourceSample = provisionalStartSample,
                endSourceSampleExclusive = provisionalEndSample,
                sourceSampleCount = provisionalSourceSamples,
                exactSampleCount24k = provisional24k,
                expectedWork = V2AudioSpanMath.expectedWorkFor24kSamples(provisional24k),
            )
        }
        var provisional = descriptor.copy(
            workId = "",
            provisionalWorkId = null,
            stableTrackSpanIdentity = V2IndexingLedgerIds.stableTrackSpanIdentity(
                descriptor.sourceFingerprint,
                provisionalSpan,
            ),
            finalizedAudioSpan = provisionalSpan,
        )
        provisional = provisional.copy(workId = V2IndexingLedgerIds.workId(provisional))
        require(descriptor.provisionalWorkId == provisional.workId) {
            "decoded EOS work does not point to its exact provisional identity"
        }
        return provisional
    }
}

/** The only state-machine operation permitted to replace content-addressed job-plan identity. */
object V2DecodedEosPlanFinalizer {
    fun finalizeCanonicalGroup(
        ledger: IndexingJobLedger,
        canonicalWorkId: String,
        evidence: V2DecodedEosEvidence,
        nowEpochMs: Long,
    ): V2DecodedEosPlanUpdate {
        V2IndexingLedgerValidator.requireValid(ledger)
        require(ledger.state == IndexingJobState.RUNNING ||
            ledger.state == IndexingJobState.PAUSE_REQUESTED
        ) { "EOS finalization requires an executing job" }

        val group = V2CanonicalAcousticWorkPlanner.groups(ledger.jobSpec)
            .singleOrNull { candidate ->
                candidate.members.any { descriptor ->
                    descriptor.workId == canonicalWorkId ||
                        descriptor.provisionalWorkId == canonicalWorkId
                }
            }
            ?: throw InvalidIndexingLedgerException(
                "EOS finalization target is not an acoustic work-group member",
            )
        val targetDescriptor = group.members.single { descriptor ->
            descriptor.workId == canonicalWorkId || descriptor.provisionalWorkId == canonicalWorkId
        }
        val targetTrack = ledger.tracks.single { it.workId == targetDescriptor.workId }
        if (targetDescriptor.finalizedAudioSpan.authority ==
            V2AudioSpanAuthority.DECODED_END_OF_STREAM
        ) {
            group.members.forEach { descriptor ->
                V2DecodedEosSpanFinalizer.finalize(descriptor.finalizedAudioSpan, evidence)
            }
            val replayRemap = if (canonicalWorkId == targetDescriptor.workId) {
                emptyMap()
            } else {
                group.members.associate { descriptor ->
                    requireNotNull(descriptor.provisionalWorkId) to descriptor.workId
                }
            }
            return V2DecodedEosPlanUpdate(ledger, replayRemap)
        }
        if (targetTrack.state != IndexingTrackState.DECODING ||
            targetTrack.checkpoint != TrackCheckpoint.PREFLIGHTED ||
            targetTrack.verifiedArtifacts.isNotEmpty() ||
            targetTrack.stageProgress != null
        ) {
            throw InvalidIndexingLedgerException(
                "ordinary EOS may finalize only inside the artifact-free decode stage",
            )
        }
        val groupIds = group.members.map { it.workId }.toSet()
        ledger.tracks.filter { it.workId in groupIds && it.workId != targetDescriptor.workId }
            .forEach(::requireRemappableAlias)

        val replacements = linkedMapOf<String, SelectedTrackDescriptor>()
        group.members.forEach { descriptor ->
            val finalizedSpan = V2DecodedEosSpanFinalizer.finalize(
                descriptor.finalizedAudioSpan,
                evidence,
            )
            val provisionalId = descriptor.provisionalWorkId ?: descriptor.workId
            var replacement = descriptor.copy(
                workId = "",
                provisionalWorkId = provisionalId,
                finalizedAudioSpan = finalizedSpan,
                stableTrackSpanIdentity = V2IndexingLedgerIds.stableTrackSpanIdentity(
                    descriptor.sourceFingerprint,
                    finalizedSpan,
                ),
            )
            replacement = replacement.copy(workId = V2IndexingLedgerIds.workId(replacement))
            replacements[descriptor.workId] = replacement
        }
        val remap = replacements.mapValues { it.value.workId }
        require(remap.values.toSet().size == remap.size) {
            "EOS finalization collapsed distinct local work identities"
        }

        val updatedDescriptors = ledger.jobSpec.tracks.map { descriptor ->
            replacements[descriptor.workId] ?: descriptor
        }
        val provisionalParent = ledger.jobSpec.provisionalParentSpecId ?: ledger.jobSpec.specId
        var updatedSpec = ledger.jobSpec.copy(
            specId = "",
            provisionalParentSpecId = provisionalParent,
            tracks = updatedDescriptors,
        )
        updatedSpec = updatedSpec.copy(specId = V2IndexingLedgerIds.jobSpecId(updatedSpec))
        val effectiveNow = maxOf(nowEpochMs, ledger.updatedAtEpochMs)
        val updatedTracks = ledger.tracks.map { track ->
            val newWorkId = remap[track.workId] ?: return@map track
            val failures = track.failures.map { failure ->
                failure.copy(
                    failureId = V2IndexingLedgerIds.failureId(
                        newWorkId,
                        failure.code,
                        failure.stage,
                        failure.sourceFingerprint,
                        failure.embeddingSpecId,
                        failure.appBuildId,
                    ),
                )
            }
            track.copy(
                workId = newWorkId,
                activeFailureId = track.activeFailureId?.let { oldActive ->
                    failures.singleOrNull { fresh ->
                        track.failures.singleOrNull { it.failureId == oldActive }?.let { old ->
                            fresh.code == old.code && fresh.stage == old.stage
                        } == true
                    }?.failureId
                },
                failures = failures,
            )
        }
        val updated = ledger.copy(
            jobSpec = updatedSpec,
            revision = Math.addExact(ledger.revision, 1L),
            updatedAtEpochMs = effectiveNow,
            tracks = updatedTracks,
            stateReason = "physical EOS finalized; estimated work was revised".take(2_048),
        )
        V2IndexingLedgerValidator.requireValid(updated)
        requireAllowedMutation(ledger, updated)
        return V2DecodedEosPlanUpdate(updated, remap)
    }

    /** Defense in depth for the atomic ledger store's otherwise immutable-spec boundary. */
    fun requireAllowedMutation(previous: IndexingJobLedger, updated: IndexingJobLedger) {
        require(
            updated.copy(
                jobSpec = previous.jobSpec,
                revision = previous.revision,
                updatedAtEpochMs = previous.updatedAtEpochMs,
                stateReason = previous.stateReason,
                tracks = previous.tracks,
            ) == previous,
        ) { "EOS finalization changed unrelated ledger state" }
        require(previous.state == IndexingJobState.RUNNING ||
            previous.state == IndexingJobState.PAUSE_REQUESTED)
        require(updated.jobSpec.jobId == previous.jobSpec.jobId)
        require(updated.jobSpec.createdAtEpochMs == previous.jobSpec.createdAtEpochMs)
        require(updated.jobSpec.providerSnapshot == previous.jobSpec.providerSnapshot)
        require(updated.jobSpec.embeddingSpec == previous.jobSpec.embeddingSpec)
        require(updated.jobSpec.textRetrievalSpec == previous.jobSpec.textRetrievalSpec)
        require(updated.jobSpec.runtimeFingerprint == previous.jobSpec.runtimeFingerprint)
        require(updated.jobSpec.baseGenerationId == previous.jobSpec.baseGenerationId)
        require(updated.jobSpec.rebuildDerivedIndexes == previous.jobSpec.rebuildDerivedIndexes)
        require(updated.jobSpec.provisionalParentSpecId ==
            (previous.jobSpec.provisionalParentSpecId ?: previous.jobSpec.specId)
        )
        require(updated.jobSpec.tracks.size == previous.jobSpec.tracks.size)
        val changedPairs = previous.jobSpec.tracks.zip(updated.jobSpec.tracks)
            .filter { (old, fresh) -> old != fresh }
        require(changedPairs.isNotEmpty()) { "EOS finalization changed no acoustic span" }
        val changedWorkIds = changedPairs.map { (old, _) -> old.workId }.toSet()
        val changedGroup = V2CanonicalAcousticWorkPlanner.groups(previous.jobSpec)
            .singleOrNull { group ->
                group.members.map { it.workId }.toSet() == changedWorkIds
            }
            ?: error("EOS finalization must replace exactly one acoustic work group")
        val previousTrackById = previous.tracks.associateBy(IndexingTrackLedger::workId)
        val decodingMembers = changedGroup.members.filter { descriptor ->
            previousTrackById.getValue(descriptor.workId).state == IndexingTrackState.DECODING
        }
        require(decodingMembers.size == 1) {
            "EOS finalization requires exactly one decoding acoustic-group member"
        }
        val decodingWorkId = decodingMembers.single().workId
        changedGroup.members.forEach { descriptor ->
            val track = previousTrackById.getValue(descriptor.workId)
            if (descriptor.workId != decodingWorkId) requireRemappableAlias(track)
        }
        val decodingTrack = previousTrackById.getValue(decodingWorkId)
        require(decodingTrack.checkpoint == TrackCheckpoint.PREFLIGHTED)
        require(decodingTrack.verifiedArtifacts.isEmpty() && decodingTrack.stageProgress == null)

        previous.jobSpec.tracks.zip(updated.jobSpec.tracks).forEach { (old, fresh) ->
            if (old == fresh) return@forEach
            require(old.finalizedAudioSpan.authority ==
                V2AudioSpanAuthority.PROVISIONAL_END_OF_STREAM
            )
            require(fresh.finalizedAudioSpan.authority ==
                V2AudioSpanAuthority.DECODED_END_OF_STREAM
            )
            require(fresh.provisionalWorkId == (old.provisionalWorkId ?: old.workId))
            val finalSpan = fresh.finalizedAudioSpan
            require(V2DecodedEosSpanFinalizer.finalize(
                old.finalizedAudioSpan,
                V2DecodedEosEvidence(
                    sourceSampleRateHz = finalSpan.container.sampleRateHz,
                    observedStartSourceSample = finalSpan.startSourceSample,
                    observedEndSourceSampleExclusive = finalSpan.endSourceSampleExclusive,
                    observedSourceSampleCount = finalSpan.sourceSampleCount,
                    exactSampleCount24k = finalSpan.exactSampleCount24k,
                    endOfStreamReached = true,
                ),
            ) == finalSpan)
            require(old.copy(
                workId = fresh.workId,
                provisionalWorkId = fresh.provisionalWorkId,
                stableTrackSpanIdentity = fresh.stableTrackSpanIdentity,
                finalizedAudioSpan = fresh.finalizedAudioSpan,
            ) == fresh)
        }
        val workIdRemap = changedPairs.associate { (old, fresh) -> old.workId to fresh.workId }
        val expectedTracks = previous.tracks.map { oldTrack ->
            val freshWorkId = workIdRemap[oldTrack.workId] ?: return@map oldTrack
            val freshFailures = oldTrack.failures.map { failure ->
                failure.copy(
                    failureId = V2IndexingLedgerIds.failureId(
                        freshWorkId,
                        failure.code,
                        failure.stage,
                        failure.sourceFingerprint,
                        failure.embeddingSpecId,
                        failure.appBuildId,
                    ),
                )
            }
            oldTrack.copy(
                workId = freshWorkId,
                activeFailureId = oldTrack.activeFailureId?.let { oldActive ->
                    val oldFailure = oldTrack.failures.single { it.failureId == oldActive }
                    freshFailures.single {
                        it.code == oldFailure.code && it.stage == oldFailure.stage
                    }.failureId
                },
                failures = freshFailures,
            )
        }
        require(updated.tracks == expectedTracks) {
            "EOS finalization changed execution evidence beyond identity remapping"
        }
        require(updated.updatedAtEpochMs >= previous.updatedAtEpochMs)
        require(updated.stateReason == "physical EOS finalized; estimated work was revised")
        require(updated.revision == Math.addExact(previous.revision, 1L))
    }

    private fun requireRemappableAlias(track: IndexingTrackLedger) {
        require(track.state in setOf(
            IndexingTrackState.PREFLIGHTED,
            IndexingTrackState.RETRYABLE_FAILURE,
            IndexingTrackState.BLOCKED_FAILURE,
            IndexingTrackState.SKIPPED_BY_USER,
        )) { "acoustic-group peer is not preflighted, failed, or terminal" }
        require(track.checkpoint == TrackCheckpoint.QUEUED ||
            track.checkpoint == TrackCheckpoint.PREFLIGHTED
        ) { "acoustic-group peer advanced beyond its artifact-free checkpoint" }
        require(track.verifiedArtifacts.isEmpty() && track.stageProgress == null) {
            "acoustic-group peer must be artifact-free before EOS finalization"
        }
    }
}
