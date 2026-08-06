package com.powerampstartradio.indexing.v2

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class V2IndexingLifecycleControllerTest {
    @Test
    fun `media processing timeout persists a resumable pause reason`() {
        val access = InMemoryLedgerAccess(plannedLedger())
        val controller = V2IndexingLifecycleController(access, TickClock())
        val reason = "Android media-processing limit reached; reopen and resume"
        controller.start(JOB_ID)
        val workId = controller.require(JOB_ID).tracks.single().workId
        controller.update(JOB_ID) {
            V2IndexingLedgerStateMachine.beginNextTrackStage(it, workId, 1_002L)
        }

        val requested = controller.requestMediaProcessingTimeoutPause(JOB_ID, reason)
        assertEquals(IndexingJobState.PAUSE_REQUESTED, requested.state)
        assertEquals(reason, requested.stateReason)

        val paused = controller.finishPauseAfterExecutorStops(JOB_ID, reason)
        assertEquals(IndexingJobState.PAUSED, paused.state)
        assertEquals(reason, paused.stateReason)
        assertEquals(IndexingTrackState.QUEUED, paused.tracks.single().state)
    }

    @Test
    fun `media processing timeout checkpoints active work in one durable transition`() {
        val access = InMemoryLedgerAccess(plannedLedger())
        val controller = V2IndexingLifecycleController(access, TickClock())
        val reason = "Android limit reached; reopen and resume"
        controller.start(JOB_ID)
        val workId = controller.require(JOB_ID).tracks.single().workId
        controller.update(JOB_ID) {
            V2IndexingLedgerStateMachine.beginNextTrackStage(it, workId, 1_002L)
        }

        val paused = controller.checkpointForMediaProcessingTimeout(JOB_ID, reason)

        assertEquals(IndexingJobState.PAUSED, paused.state)
        assertEquals(IndexingTrackState.QUEUED, paused.tracks.single().state)
        assertEquals(reason, paused.stateReason)
    }

    @Test
    fun `pause is durable before active work is released`() {
        val access = InMemoryLedgerAccess(plannedLedger())
        val controller = V2IndexingLifecycleController(access, TickClock())
        var ledger = controller.start(JOB_ID)
        val workId = ledger.tracks.single().workId
        ledger = controller.update(JOB_ID) {
            V2IndexingLedgerStateMachine.beginNextTrackStage(it, workId, 1_002L)
        }

        ledger = controller.requestPause(JOB_ID)
        assertEquals(IndexingJobState.PAUSE_REQUESTED, ledger.state)
        assertTrue(ledger.tracks.single().state.isActiveStage())

        ledger = controller.finishPauseAfterExecutorStops(JOB_ID)
        assertEquals(IndexingJobState.PAUSED, ledger.state)
        assertEquals(IndexingTrackState.QUEUED, ledger.tracks.single().state)
        assertEquals(TrackCheckpoint.QUEUED, ledger.tracks.single().checkpoint)
    }

    @Test
    fun `cancellation intent survives active work and finishes only after cleanup`() {
        val access = InMemoryLedgerAccess(plannedLedger())
        val controller = V2IndexingLifecycleController(access, TickClock())
        var ledger = controller.start(JOB_ID)
        val workId = ledger.tracks.single().workId
        controller.update(JOB_ID) {
            V2IndexingLedgerStateMachine.beginNextTrackStage(it, workId, 1_002L)
        }

        ledger = controller.requestCancel(JOB_ID)
        assertEquals(IndexingJobState.CANCELLING, ledger.state)
        assertTrue(ledger.tracks.single().state.isActiveStage())

        ledger = controller.finishCancellationAfterCleanup(JOB_ID)
        assertEquals(IndexingJobState.CANCELLED, ledger.state)
        assertEquals(IndexingTrackState.QUEUED, ledger.tracks.single().state)
    }

    @Test
    fun `process death after durable pause request does not consume a retry`() {
        val access = InMemoryLedgerAccess(plannedLedger())
        val controller = V2IndexingLifecycleController(access, TickClock())
        var ledger = controller.start(JOB_ID)
        val workId = ledger.tracks.single().workId
        ledger = controller.update(JOB_ID) {
            V2IndexingLedgerStateMachine.beginNextTrackStage(it, workId, 1_002L)
        }
        val attemptsBeforePause = ledger.tracks.single().attemptCount
        controller.requestPause(JOB_ID)

        val result = controller.reconcileNonterminalJobs()
        ledger = controller.require(JOB_ID)

        assertEquals(1, result.reconciledJobs)
        assertEquals(IndexingJobState.PAUSED, ledger.state)
        assertEquals(IndexingTrackState.QUEUED, ledger.tracks.single().state)
        assertEquals(attemptsBeforePause, ledger.tracks.single().attemptCount)
        assertTrue(ledger.tracks.single().failures.isEmpty())
    }

    @Test
    fun `restart failure resumes from checkpoint and automatic crash retries are bounded`() {
        val access = InMemoryLedgerAccess(plannedLedger())
        val clock = TickClock()
        val controller = V2IndexingLifecycleController(access, clock)
        controller.start(JOB_ID)
        val workId = controller.require(JOB_ID).tracks.single().workId

        repeat(3) { occurrence ->
            controller.update(JOB_ID) {
                V2IndexingLedgerStateMachine.beginNextTrackStage(it, workId, clock())
            }
            val reconciliation = controller.reconcileNonterminalJobs()
            assertEquals(1, reconciliation.reconciledJobs)
            val interrupted = controller.require(JOB_ID)
            assertEquals(IndexingJobState.INTERRUPTED, interrupted.state)
            val failed = interrupted.tracks.single()
            assertEquals(occurrence + 1, failed.failures.single().occurrences)

            val retry = controller.retryFailed(JOB_ID, RetryTrigger.PROCESS_RESTART)
            if (occurrence < 2) {
                assertEquals(1, retry.retried)
                controller.resume(JOB_ID)
            } else {
                assertEquals(0, retry.retried)
                assertEquals(1, retry.notEligible)
                assertEquals(IndexingTrackState.BLOCKED_FAILURE, retry.ledger.tracks.single().state)
                assertEquals(
                    RetryTrigger.USER_REQUEST,
                    retry.ledger.tracks.single().failures.single().retryTrigger,
                )
            }
        }
    }

    @Test
    fun `profile mutation persists without changing immutable job spec`() {
        val access = InMemoryLedgerAccess(plannedLedger())
        val controller = V2IndexingLifecycleController(access, TickClock())
        val before = controller.require(JOB_ID)
        val after = controller.changeProfile(JOB_ID, V2IndexingExecutionProfile.BACKGROUND)

        assertEquals(before.jobSpec, after.jobSpec)
        assertEquals(V2IndexingExecutionProfile.BACKGROUND, after.executionProfile)
        assertEquals(before.revision + 1L, after.revision)
    }

    private class InMemoryLedgerAccess(initial: IndexingJobLedger) : V2IndexingLedgerAccess {
        private var value = initial

        override fun list(): List<IndexingJobLedger> = listOf(value)

        override fun require(jobId: String): IndexingJobLedger {
            require(jobId == value.jobSpec.jobId)
            return value
        }

        override fun update(
            jobId: String,
            expectedRevision: Long,
            transition: (IndexingJobLedger) -> IndexingJobLedger,
        ): IndexingJobLedger {
            require(jobId == value.jobSpec.jobId)
            if (value.revision != expectedRevision) throw IndexingLedgerConflictException("stale")
            value = transition(value)
            V2IndexingLedgerValidator.requireValid(value)
            return value
        }

        override fun updateLatest(
            jobId: String,
            transition: (IndexingJobLedger) -> IndexingJobLedger,
        ): IndexingJobLedger = update(jobId, value.revision, transition)

        override fun reconcileAfterProcessRestart(
            jobId: String,
            expectedRevision: Long,
            nowEpochMs: Long,
        ): RestartReconciliation {
            require(jobId == value.jobSpec.jobId)
            if (value.revision != expectedRevision) throw IndexingLedgerConflictException("stale")
            val result = V2IndexingLedgerStateMachine.reconcileAfterProcessRestart(
                value,
                nowEpochMs,
            )
            value = result.ledger
            return result
        }
    }

    private class TickClock : () -> Long {
        private var value = 1_000L
        override fun invoke(): Long = ++value
    }

    private fun plannedLedger(): IndexingJobLedger {
        val path = "/storage/emulated/0/Music/test.flac"
        val durationUs = 10_000_000L
        val sourceSamples = 480_000L
        val exact24k = 240_000L
        val generation = "poweramp-provider-snapshot-v3-sha256:" + "7".repeat(64)
        val audioSpec = V2IndexingLedgerPlanner.createEmbeddingSpec(
            EmbeddingSpecInput(
                preprocessingSpecId = V2IndexingWorkPolicy.PREPROCESSING_SPEC_ID,
                decoderPolicyId = V2IndexingWorkPolicy.DECODER_POLICY_ID,
                inferenceBackendPolicyId = V2IndexingWorkPolicy.INFERENCE_BACKEND_POLICY_ID,
                outputDimension = 768,
                modelArtifactSha256 = mapOf(
                    "mert" to "a".repeat(64),
                    "clamp3_audio" to "b".repeat(64),
                ),
            ),
        )
        return V2IndexingLedgerPlanner.planJob(
            providerSnapshot = PowerampProviderSnapshotEvidence(
                libraryGeneration = generation,
                acquisition = V2ProviderSnapshotAcquisitionEvidence(
                    queryUri = "content://com.maxmpz.audioplayer.data/files",
                    requestedColumns = listOf("_id", "duration", "path"),
                    returnedColumns = listOf("_id", "duration", "path"),
                    rowCount = 1,
                    cursorExhaustedNormally = true,
                ),
            ),
            embeddingSpec = audioSpec,
            textRetrievalSpec = V2IndexingLedgerPlanner.createTextRetrievalSpec(
                TextRetrievalSpecInput(
                    compatibleAudioEmbeddingSpecId = audioSpec.specId,
                    textModelSha256 = "1".repeat(64),
                    tokenizerModelSha256 = V2IndexingWorkPolicy.TEXT_TOKENIZER_MODEL_SHA256,
                    tokenizerPolicyId = V2IndexingWorkPolicy.TEXT_TOKENIZER_POLICY_ID,
                    tokenizerRuntimeContractSha256 =
                        V2IndexingWorkPolicy.TEXT_TOKENIZER_RUNTIME_CONTRACT_SHA256,
                    outputSpaceId = V2IndexingWorkPolicy.TEXT_OUTPUT_SPACE_ID,
                    outputDimension = audioSpec.outputDimension,
                    inferenceBackendPolicyId =
                        V2IndexingWorkPolicy.TEXT_INFERENCE_BACKEND_POLICY_ID,
                ),
            ),
            runtimeFingerprint = IndexingRuntimeFingerprint(
                appVersionCode = 2_000_000L,
                appBuildId = "test-build",
                decoderRuntimeId = "test-decoder",
                platformFingerprint = "test-platform",
            ),
            selectedTracks = listOf(
                SelectedTrackInput(
                    powerampFileId = 1L,
                    providerSnapshotGeneration = generation,
                    providerRow = V2ProviderPathRowEvidence(
                        powerampFileId = 1L,
                        physicalPath = path,
                        providerPhysicalPath = path,
                        artist = "Artist",
                        album = "Album",
                        title = "Title",
                        offsetMs = 0L,
                        durationMs = 10_000L,
                        cueSourceImageFolderId = null,
                    ),
                    displayMetadata = DisplayTrackMetadata("Artist", "Album", "Title"),
                    normalizedMetadata = NormalizedTrackMetadata(
                        normalizationSpecId = V2IndexingWorkPolicy.METADATA_NORMALIZATION_SPEC_ID,
                        artist = "artist",
                        album = "album",
                        title = "title",
                        metadataKey = "artist|album|title|10000",
                    ),
                    physicalPath = path,
                    sourceFingerprint = SourceFingerprint(
                        fingerprintSpecId = V2FixedRegionSampling.SPEC_ID,
                        sizeBytes = 1_000L,
                        lastModifiedEpochMs = 1L,
                        fileKey = null,
                        sampledContentSha256 = "c".repeat(64),
                        fullContentSha256 = null,
                    ),
                    finalizedAudioSpan = FinalizedAudioSpanEvidence(
                        kind = V2ResolvedAudioSpanKind.WHOLE_FILE,
                        authority = V2AudioSpanAuthority.PROVISIONAL_END_OF_STREAM,
                        executionBoundaryRequirement =
                            V2ExecutionBoundaryRequirement.VERIFY_END_OF_STREAM_AND_RECONCILE,
                        providerSpan = V2ProviderSpanEvidence(0L, durationUs, durationUs),
                        cueClassification = V2CueClassificationEvidence(
                            providerGroupRowCount = 1,
                            logicalRowCount = 1,
                            nonZeroOffsetRowIds = emptyList(),
                            rawSourceImageRowIds = emptyList(),
                        ),
                        container = V2AudioContainerEvidence(
                            physicalPath = path,
                            audioTrackIndex = 0,
                            durationUsEstimate = durationUs,
                            sampleRateHz = 48_000,
                            channelCount = 2,
                            mime = "audio/flac",
                        ),
                        startUs = 0L,
                        endExclusiveUs = durationUs,
                        startSourceSample = 0L,
                        endSourceSampleExclusive = sourceSamples,
                        sourceSampleCount = sourceSamples,
                        exactSampleCount24k = exact24k,
                        expectedWork = V2AudioSpanMath.expectedWorkFor24kSamples(exact24k),
                    ),
                ),
            ),
            rebuildDerivedIndexes = true,
            createdAtEpochMs = 900L,
            jobId = JOB_ID,
        )
    }

    private companion object {
        const val JOB_ID = "lifecycle-job"
    }
}
