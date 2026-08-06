package com.powerampstartradio.indexing.v2

import com.google.gson.Gson
import com.powerampstartradio.indexing.V2IndexingPreflightLedgerLinkAction
import com.powerampstartradio.indexing.V2IndexingPreflightLedgerLinkPolicy
import java.io.File
import java.nio.file.Files
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class V2IndexingLedgerStateMachineTest {
    @Test
    fun `CUE authorization survives multiple ordinary EOS revisions without sidecar rewrite`() {
        fun relocated(
            input: SelectedTrackInput,
            powerampFileId: Long,
            path: String,
            title: String,
            contentHash: Char,
        ): SelectedTrackInput = input.copy(
            powerampFileId = powerampFileId,
            providerRow = input.providerRow.copy(
                powerampFileId = powerampFileId,
                physicalPath = path,
                providerPhysicalPath = path,
                title = title,
            ),
            displayMetadata = input.displayMetadata.copy(title = title),
            normalizedMetadata = input.normalizedMetadata.copy(
                title = title.lowercase(),
                metadataKey = "artist|album|${title.lowercase()}|241200",
            ),
            physicalPath = path,
            sourceFingerprint = input.sourceFingerprint.copy(
                fileKey = "dev=12;ino=$powerampFileId",
                sampledContentSha256 = sha(contentHash),
            ),
            finalizedAudioSpan = input.finalizedAudioSpan.copy(
                cueClassification = input.finalizedAudioSpan.cueClassification.copy(
                    nonZeroOffsetRowIds = if (input.finalizedAudioSpan.kind ==
                        V2ResolvedAudioSpanKind.LOGICAL_CUE
                    ) listOf(powerampFileId) else emptyList(),
                ),
                container = input.finalizedAudioSpan.container.copy(physicalPath = path),
            ),
        )

        val ordinaryOne = relocated(
            selectedTrack(providerOffsetMs = 0L, wholeFile = true),
            991L,
            "/storage/emulated/0/Music/ordinary-one.flac",
            "Ordinary One",
            'c',
        )
        val ordinaryTwo = relocated(
            selectedTrack(providerOffsetMs = 0L, wholeFile = true),
            992L,
            "/storage/emulated/0/Music/ordinary-two.flac",
            "Ordinary Two",
            'd',
        )
        val cue = relocated(
            selectedTrack(providerOffsetMs = 7_500L, wholeFile = false),
            993L,
            "/storage/emulated/0/Music/cue-image.flac",
            "Cue Work",
            'e',
        )
        val baseGenerationId = "index-generation-v2-" + "a".repeat(64)
        var ledger = plannedLedger(
            jobId = "job-cue-eos-lineage",
            selectedTracks = listOf(ordinaryOne, ordinaryTwo, cue),
            baseGenerationId = baseGenerationId,
        )
        val preflightSpecId = ledger.jobSpec.specId
        val cueDescriptor = ledger.jobSpec.tracks.single {
            it.finalizedAudioSpan.kind == V2ResolvedAudioSpanKind.LOGICAL_CUE
        }
        val predecessorMetadata = V2CommitTrackMetadata(
            metadataKey = "artist|album|imported cue|241250",
            filenameKey = "cue-image.flac",
            artist = "Artist",
            album = "Album",
            title = "Imported Cue",
            durationMs = 241_250,
            filePath = cueDescriptor.providerRow.physicalPath,
            source = "desktop",
        )
        val authorization = V2ImportedRowSupersessionAuthorization(
            schemaVersion = V2ImportedRowSupersessionAuthorizationSchema.VERSION,
            jobId = ledger.jobSpec.jobId,
            jobSpecId = preflightSpecId,
            baseGenerationId = baseGenerationId,
            baseManifestSha256 = "1".repeat(64),
            baseDatabaseByteLength = 12_345L,
            baseDatabaseSha256 = "2".repeat(64),
            baseDatabaseContentSha256 = "3".repeat(64),
            privateBaseBindingId = V2JobPrivateDatabaseBindingIdentity.compute(
                jobId = ledger.jobSpec.jobId,
                jobSpecId = preflightSpecId,
                baseGenerationId = baseGenerationId,
                sourceDatabaseByteLength = 12_345L,
                sourceDatabaseSha256 = "2".repeat(64),
                baseManifestSha256 = "1".repeat(64),
                baseDatabaseContentSha256 = "3".repeat(64),
            ),
            providerSnapshotGeneration = ledger.jobSpec.providerSnapshot.libraryGeneration,
            works = listOf(
                V2ImportedRowWorkAuthorization(
                    workId = cueDescriptor.workId,
                    powerampFileId = cueDescriptor.powerampFileId,
                    providerSpan = cueDescriptor.committedProviderSpan(),
                    kind = V2ImportedRowCommitKind.SUPERSESSION,
                    predecessor = V2ImportedPredecessorEvidence(
                        trackId = 7L,
                        metadata = predecessorMetadata,
                        metadataSha256 = V2CommitMetadataIdentity.sha256(predecessorMetadata),
                        embeddingByteLength = V2_CLAMP3_BLOB_BYTES,
                        embeddingSha256 = "4".repeat(64),
                    ),
                ),
            ),
        )
        V2ImportedRowSupersessionAuthorizationPolicy.requireValid(authorization, ledger)

        val sidecarRoot = Files.createTempDirectory("cue-eos-authorization").toFile()
        var sidecarWrites = 0
        try {
            val store = V2ImportedRowSupersessionAuthorizationStore(
                ledgerDirectory = sidecarRoot,
                atomicIo = object : V2ImportedRowAuthorizationAtomicIo {
                    override fun read(file: File): ByteArray = file.readBytes()

                    override fun write(file: File, bytes: ByteArray) {
                        sidecarWrites += 1
                        file.parentFile?.mkdirs()
                        file.writeBytes(bytes)
                    }
                },
            )
            store.createOrRequireExact(authorization)
            ledger = V2IndexingLedgerStateMachine.startJob(ledger, 2_000L)

            listOf(991L, 992L).forEachIndexed { index, powerampFileId ->
                val descriptor = ledger.jobSpec.tracks.single {
                    it.powerampFileId == powerampFileId
                }
                ledger = V2IndexingLedgerStateMachine.beginNextTrackStage(
                    ledger,
                    descriptor.workId,
                    2_010L + index * 100L,
                )
                ledger = V2IndexingLedgerStateMachine.completeActiveTrackStage(
                    ledger,
                    descriptor.workId,
                    null,
                    2_020L + index * 100L,
                )
                ledger = V2IndexingLedgerStateMachine.beginNextTrackStage(
                    ledger,
                    descriptor.workId,
                    2_030L + index * 100L,
                )
                ledger = V2DecodedEosPlanFinalizer.finalizeCanonicalGroup(
                    ledger,
                    descriptor.workId,
                    decodedEosEvidence(
                        descriptor,
                        descriptor.finalizedAudioSpan.endSourceSampleExclusive + index + 1L,
                    ),
                    2_040L + index * 100L,
                ).ledger
                val finalized = ledger.jobSpec.tracks.single {
                    it.powerampFileId == powerampFileId
                }
                ledger = V2IndexingLedgerStateMachine.completeActiveTrackStage(
                    ledger,
                    finalized.workId,
                    artifact(
                        ledger,
                        VerifiedArtifactKind.MERT_FEATURES,
                        2_050L + index * 100L,
                        finalized.workId,
                    ),
                    2_050L + index * 100L,
                )

                assertEquals(preflightSpecId, ledger.jobSpec.provisionalParentSpecId)
                assertEquals(
                    preflightSpecId,
                    V2DecodedEosLineage.requirePreflightSpecId(ledger.jobSpec),
                )
                assertEquals(authorization, store.requireFor(ledger))
                assertEquals(1, sidecarWrites)
            }

            assertNotEquals(preflightSpecId, ledger.jobSpec.specId)
            assertEquals(cueDescriptor, ledger.jobSpec.tracks.single {
                it.powerampFileId == cueDescriptor.powerampFileId
            })
            val commitAuthorization = requireNotNull(
                authorization.commitAuthorizationFor(cueDescriptor.workId, ledger),
            )
            assertEquals(ledger.jobSpec.specId, commitAuthorization.jobSpecId)
            assertEquals(
                authorization.executionPrivateBaseBindingId(ledger),
                commitAuthorization.privateBaseBindingId,
            )
            assertNotEquals(
                authorization.privateBaseBindingId,
                commitAuthorization.privateBaseBindingId,
            )
            assertThrows(IllegalArgumentException::class.java) {
                V2ImportedRowSupersessionAuthorizationPolicy.requireValid(
                    authorization.copy(
                        jobSpecId = ledger.jobSpec.specId,
                        privateBaseBindingId = commitAuthorization.privateBaseBindingId,
                    ),
                    ledger,
                )
            }
        } finally {
            sidecarRoot.deleteRecursively()
        }
    }

    @Test
    fun qualifiedPowerampProjectionAcceptsProviderStrippedCursorNames() {
        val acquisition = providerSnapshot().acquisition.copy(
            requestedColumns = listOf(
                "folder_files._id",
                "artist",
                "folder_files.duration",
                "path",
                "folder_files.name",
                "folder_files.offset_ms",
            ),
            returnedColumns = listOf("_id", "artist", "duration", "path", "name", "offset_ms"),
        )

        val ledger = plannedLedger(
            providerSnapshot = providerSnapshot().copy(acquisition = acquisition),
        )

        assertEquals(acquisition, ledger.jobSpec.providerSnapshot.acquisition)
    }

    @Test
    fun qualifiedPowerampProjectionStillRejectsMissingSemanticColumn() {
        val acquisition = providerSnapshot().acquisition.copy(
            requestedColumns = listOf("folder_files._id", "folder_files.duration"),
            returnedColumns = listOf("_id"),
        )

        assertThrows(InvalidIndexingLedgerException::class.java) {
            plannedLedger(providerSnapshot = providerSnapshot().copy(acquisition = acquisition))
        }
    }

    @Test
    fun `execution profile changes scheduling without changing byte identities`() {
        val original = plannedLedger()
        val changed = V2IndexingLedgerStateMachine.changeExecutionProfile(
            original,
            V2IndexingExecutionProfile.BACKGROUND,
            1_100L,
        )

        assertEquals(V2IndexingExecutionProfile.FULL, original.executionProfile)
        assertEquals(V2IndexingExecutionProfile.BACKGROUND, changed.executionProfile)
        assertEquals(original.jobSpec, changed.jobSpec)
        assertEquals(original.jobSpec.specId, changed.jobSpec.specId)
        assertEquals(
            original.jobSpec.tracks.single().workId,
            changed.jobSpec.tracks.single().workId,
        )
        assertNotEquals(
            V2IndexingExecutionPolicies.schedule(V2IndexingExecutionProfile.FULL),
            V2IndexingExecutionPolicies.schedule(V2IndexingExecutionProfile.BACKGROUND),
        )
        V2IndexingExecutionPolicies.requirePinnedByteContract(changed.jobSpec.embeddingSpec)
    }

    @Test
    fun `initial execution profile survives persistence restart and resume`() {
        val full = plannedLedger()
        val planned = plannedLedger(executionProfile = V2IndexingExecutionProfile.BACKGROUND)

        assertEquals(full.jobSpec, planned.jobSpec)
        assertEquals(V2IndexingExecutionProfile.BACKGROUND, planned.executionProfile)

        val restored = Gson().fromJson(
            Gson().toJson(planned),
            IndexingJobLedger::class.java,
        )
        var ledger = V2IndexingLedgerStateMachine.startJob(restored, 1_100L)
        val interrupted = V2IndexingLedgerStateMachine.reconcileAfterProcessRestart(
            ledger,
            1_200L,
        ).ledger
        ledger = V2IndexingLedgerStateMachine.prepareResume(interrupted, 1_300L)
        ledger = V2IndexingLedgerStateMachine.resume(ledger, 1_400L)

        assertEquals(IndexingJobState.RUNNING, ledger.state)
        assertEquals(V2IndexingExecutionProfile.BACKGROUND, ledger.executionProfile)
        assertEquals(planned.jobSpec.specId, ledger.jobSpec.specId)
        assertEquals(planned.tracks.map { it.workId }, ledger.tracks.map { it.workId })
    }

    @Test
    fun `progress counters use immutable exact work and verified artifacts`() {
        var ledger = V2IndexingLedgerStateMachine.startJob(plannedLedger(), 2_000L)
        val descriptor = ledger.jobSpec.tracks.single()
        val workId = descriptor.workId
        assertEquals(
            V2IndexingProgressSnapshot(
                resolvedTracks = 0,
                succeededTracks = 0,
                blockedTracks = 0,
                skippedTracks = 0,
                totalTracks = 1,
                tracksWithMertFeatures = 0,
                tracksWithClampVectors = 0,
                mertWindows = V2DurableStageCounter(
                    completedUnits = 0L,
                    remainingUnits = descriptor.expectedWork.mertWindows.toLong(),
                    abandonedUnits = 0L,
                ),
                clampSegments = V2DurableStageCounter(
                    completedUnits = 0L,
                    remainingUnits = descriptor.expectedWork.clampSegments.toLong(),
                    abandonedUnits = 0L,
                ),
                databaseCommits = V2DurableStageCounter(0L, 1L, 0L),
                activation = V2DurableStageCounter(0L, 1L, 0L),
                activeTrackOrdinal = null,
                activeStage = null,
            ),
            V2IndexingProgress.from(ledger),
        )
        ledger = V2IndexingLedgerStateMachine.beginNextTrackStage(ledger, workId, 2_010L)
        ledger = V2IndexingLedgerStateMachine.completeActiveTrackStage(
            ledger,
            workId,
            null,
            2_020L,
        )
        ledger = V2IndexingLedgerStateMachine.beginNextTrackStage(ledger, workId, 2_030L)
        ledger = V2IndexingLedgerStateMachine.completeActiveTrackStage(
            ledger,
            workId,
            artifact(ledger, VerifiedArtifactKind.MERT_FEATURES, 2_040L),
            2_040L,
        )

        val progress = V2IndexingProgress.from(ledger)
        assertEquals(
            descriptor.expectedWork.mertWindows.toLong(),
            progress.mertWindows.completedUnits,
        )
        assertEquals(1, progress.tracksWithMertFeatures)
        assertEquals(0, progress.tracksWithClampVectors)
        assertEquals(0L, progress.mertWindows.remainingUnits)
        assertEquals(descriptor.expectedWork.clampSegments.toLong(), progress.clampSegments.remainingUnits)
        assertEquals(0.0, progress.resolvedFraction, 0.0)

        ledger = V2IndexingLedgerStateMachine.beginNextTrackStage(ledger, workId, 2_050L)
        ledger = V2IndexingLedgerStateMachine.completeActiveTrackStage(
            ledger,
            workId,
            artifact(ledger, VerifiedArtifactKind.CLAMP_VECTOR, 2_060L),
            2_060L,
        )
        val clampProgress = V2IndexingProgress.from(ledger)
        assertEquals(1, clampProgress.tracksWithMertFeatures)
        assertEquals(1, clampProgress.tracksWithClampVectors)
        assertEquals(
            descriptor.expectedWork.clampSegments.toLong(),
            clampProgress.clampSegments.completedUnits,
        )
    }

    @Test
    fun `gson round trip preserves immutable ids and validates`() {
        val original = V2IndexingLedgerStateMachine.startJob(plannedLedger(), 1_500L)
        val restored = Gson().fromJson(
            Gson().toJson(original),
            IndexingJobLedger::class.java,
        )

        assertEquals(original, restored)
        V2IndexingLedgerValidator.requireValid(restored)
    }

    @Test
    fun `plan captures exact identity and content-addresses semantic inputs`() {
        val firstEmbeddingSpec = V2IndexingLedgerPlanner.createEmbeddingSpec(
            embeddingInput(linkedMapOf("mert" to sha('a'), "clamp3_audio" to sha('b'))),
        )
        val reorderedEmbeddingSpec = V2IndexingLedgerPlanner.createEmbeddingSpec(
            embeddingInput(linkedMapOf("clamp3_audio" to sha('b'), "mert" to sha('a'))),
        )
        assertEquals(firstEmbeddingSpec, reorderedEmbeddingSpec)

        val first = plannedLedger(jobId = "job-a", embeddingSpec = firstEmbeddingSpec)
        val second = plannedLedger(jobId = "job-b", embeddingSpec = reorderedEmbeddingSpec)
        assertEquals(first.jobSpec.specId, second.jobSpec.specId)

        val descriptor = first.jobSpec.tracks.single()
        assertEquals(991L, descriptor.powerampFileId)
        assertEquals(PROVIDER_GENERATION, first.jobSpec.providerSnapshot.libraryGeneration)
        assertEquals(7_500L, descriptor.providerOffsetMs)
        assertEquals(241_250L, descriptor.providerDurationMs)
        assertEquals(7_500_000L, descriptor.finalizedAudioSpan.startUs)
        assertEquals(248_750_000L, descriptor.finalizedAudioSpan.endExclusiveUs)
        assertEquals("artist|album|title|241200", descriptor.normalizedMetadata.metadataKey)
        assertEquals(sha('c'), descriptor.sourceFingerprint.sampledContentSha256)

        val changedSpan = selectedTrack(providerOffsetMs = 7_501L)
        val changed = plannedLedger(jobId = "job-c", selectedTrack = changedSpan)
        assertNotEquals(descriptor.workId, changed.jobSpec.tracks.single().workId)
        assertNotEquals(
            descriptor.stableTrackSpanIdentity.stableTrackSpanId,
            changed.jobSpec.tracks.single().stableTrackSpanIdentity.stableTrackSpanId,
        )
        assertNotEquals(first.jobSpec.specId, changed.jobSpec.specId)

        val metadataOnlyInput = selectedTrack().let { input ->
            input.copy(providerRow = input.providerRow.copy(title = "Retagged title"))
        }
        val metadataOnly = plannedLedger(jobId = "job-metadata", selectedTrack = metadataOnlyInput)
        assertEquals(descriptor.workId, metadataOnly.jobSpec.tracks.single().workId)
        assertEquals(
            descriptor.stableTrackSpanIdentity,
            metadataOnly.jobSpec.tracks.single().stableTrackSpanIdentity,
        )
        assertNotEquals(first.jobSpec.specId, metadataOnly.jobSpec.specId)

        val reacquired = plannedLedger(
            jobId = "job-snapshot",
            providerSnapshot = providerSnapshot().copy(
                acquisition = providerSnapshot().acquisition.copy(rowCount = 10_001),
            ),
        )
        assertEquals(descriptor.workId, reacquired.jobSpec.tracks.single().workId)
        assertEquals(
            descriptor.stableTrackSpanIdentity,
            reacquired.jobSpec.tracks.single().stableTrackSpanIdentity,
        )
        assertNotEquals(first.jobSpec.specId, reacquired.jobSpec.specId)

        val movedPath = "/storage/emulated/0/Music/moved/image.flac"
        val movedInput = selectedTrack().let { input ->
            input.copy(
                physicalPath = movedPath,
                providerRow = input.providerRow.copy(
                    physicalPath = movedPath,
                    providerPhysicalPath = movedPath,
                ),
                finalizedAudioSpan = input.finalizedAudioSpan.copy(
                    container = input.finalizedAudioSpan.container.copy(physicalPath = movedPath),
                ),
            )
        }
        val moved = plannedLedger(jobId = "job-moved", selectedTrack = movedInput)
        assertNotEquals(descriptor.workId, moved.jobSpec.tracks.single().workId)
        assertEquals(
            descriptor.stableTrackSpanIdentity,
            moved.jobSpec.tracks.single().stableTrackSpanIdentity,
        )

        val mutatedWithoutNewIdentity = first.copy(
            jobSpec = first.jobSpec.copy(
                tracks = first.jobSpec.tracks.map { descriptorToMutate ->
                    descriptorToMutate.copy(
                        finalizedAudioSpan = descriptorToMutate.finalizedAudioSpan.copy(
                            endExclusiveUs = 99L,
                        ),
                    )
                },
            ),
        )
        val errors = V2IndexingLedgerValidator.validate(mutatedWithoutNewIdentity)
        assertTrue(errors.any { it.contains("specId mismatch") })
        assertTrue(errors.any { it.contains("workId mismatch") })
    }

    @Test
    fun `pause waits for checkpoint and full lifecycle commits only verified artifacts`() {
        var ledger = V2IndexingLedgerStateMachine.startJob(plannedLedger(), 2_000L)
        val workId = ledger.tracks.single().workId

        ledger = V2IndexingLedgerStateMachine.beginNextTrackStage(ledger, workId, 2_010L)
        ledger = V2IndexingLedgerStateMachine.completeActiveTrackStage(
            ledger,
            workId,
            artifact = null,
            nowEpochMs = 2_020L,
        )
        ledger = V2IndexingLedgerStateMachine.beginNextTrackStage(ledger, workId, 2_030L)
        ledger = V2IndexingLedgerStateMachine.requestPause(ledger, 2_040L)

        assertThrows(InvalidIndexingLedgerException::class.java) {
            V2IndexingLedgerStateMachine.finishPause(ledger, 2_050L)
        }

        ledger = V2IndexingLedgerStateMachine.completeActiveTrackStage(
            ledger,
            workId,
            artifact(ledger, VerifiedArtifactKind.MERT_FEATURES, 2_060L),
            2_060L,
        )
        ledger = V2IndexingLedgerStateMachine.finishPause(ledger, 2_070L)
        assertEquals(IndexingJobState.PAUSED, ledger.state)
        assertEquals(IndexingTrackState.MERT_COMPLETE, ledger.tracks.single().state)

        ledger = V2IndexingLedgerStateMachine.prepareResume(ledger, 2_080L)
        ledger = V2IndexingLedgerStateMachine.resume(ledger, 2_090L)
        ledger = V2IndexingLedgerStateMachine.beginNextTrackStage(ledger, workId, 2_100L)
        ledger = V2IndexingLedgerStateMachine.completeActiveTrackStage(
            ledger,
            workId,
            artifact(ledger, VerifiedArtifactKind.CLAMP_VECTOR, 2_110L),
            2_110L,
        )
        ledger = V2IndexingLedgerStateMachine.beginNextTrackStage(ledger, workId, 2_120L)
        ledger = V2IndexingLedgerStateMachine.completeActiveTrackStage(
            ledger,
            workId,
            artifact(ledger, VerifiedArtifactKind.DATABASE_COMMIT, 2_130L),
            2_130L,
        )
        ledger = V2IndexingLedgerStateMachine.beginActivation(ledger, 2_140L)
        assertThrows(InvalidIndexingLedgerException::class.java) {
            V2IndexingLedgerStateMachine.requestCancel(ledger, 2_145L)
        }
        ledger = V2IndexingLedgerStateMachine.completeJob(
            ledger,
            activationEvidence(ledger, 2_150L),
            2_150L,
        )

        assertEquals(IndexingJobState.COMPLETE, ledger.state)
        assertEquals(IndexingTrackState.COMMITTED, ledger.tracks.single().state)
        assertEquals(
            setOf(
                VerifiedArtifactKind.MERT_FEATURES,
                VerifiedArtifactKind.CLAMP_VECTOR,
                VerifiedArtifactKind.DATABASE_COMMIT,
            ),
            ledger.tracks.single().verifiedArtifacts.map { it.kind }.toSet(),
        )
        V2IndexingLedgerValidator.requireValid(ledger)
    }

    @Test
    fun `activation authorization change durably requires a new job`() {
        var ledger = V2IndexingLedgerStateMachine.startJob(plannedLedger(), 2_200L)
        val workId = ledger.tracks.single().workId
        ledger = V2IndexingLedgerStateMachine.beginNextTrackStage(ledger, workId, 2_210L)
        ledger = V2IndexingLedgerStateMachine.completeActiveTrackStage(
            ledger,
            workId,
            null,
            2_220L,
        )
        ledger = V2IndexingLedgerStateMachine.beginNextTrackStage(ledger, workId, 2_230L)
        ledger = V2IndexingLedgerStateMachine.completeActiveTrackStage(
            ledger,
            workId,
            artifact(ledger, VerifiedArtifactKind.MERT_FEATURES, 2_240L),
            2_240L,
        )
        ledger = V2IndexingLedgerStateMachine.beginNextTrackStage(ledger, workId, 2_250L)
        ledger = V2IndexingLedgerStateMachine.completeActiveTrackStage(
            ledger,
            workId,
            artifact(ledger, VerifiedArtifactKind.CLAMP_VECTOR, 2_260L),
            2_260L,
        )
        ledger = V2IndexingLedgerStateMachine.beginNextTrackStage(ledger, workId, 2_270L)
        ledger = V2IndexingLedgerStateMachine.completeActiveTrackStage(
            ledger,
            workId,
            artifact(ledger, VerifiedArtifactKind.DATABASE_COMMIT, 2_280L),
            2_280L,
        )
        ledger = V2IndexingLedgerStateMachine.beginActivation(ledger, 2_290L)
        val committedArtifacts = ledger.tracks.single().verifiedArtifacts

        ledger = V2IndexingLedgerStateMachine.blockImportedRowActivationForNewJob(
            ledger = ledger,
            workId = workId,
            diagnostic = "authorization sidecar no longer matches",
            nowEpochMs = 2_300L,
        )

        val track = ledger.tracks.single()
        val failure = track.failures.single { it.failureId == track.activeFailureId }
        assertEquals(V2IndexingLedgerSchema.VERSION, ledger.schemaVersion)
        assertEquals(IndexingJobState.WAITING_FOR_INPUT, ledger.state)
        assertEquals(IndexingTrackState.BLOCKED_FAILURE, track.state)
        assertEquals(TrackCheckpoint.COMMITTED, track.checkpoint)
        assertEquals(committedArtifacts, track.verifiedArtifacts)
        assertEquals(TrackFailureCode.IMPORTED_ROW_AUTHORIZATION_CHANGED, failure.code)
        assertEquals(IndexingStage.DATABASE_ACTIVATION, failure.stage)
        assertEquals(RetryTrigger.NEW_JOB_REQUIRED, failure.retryTrigger)
        assertThrows(InvalidIndexingLedgerException::class.java) {
            V2IndexingLedgerStateMachine.retryTrack(
                ledger,
                workId,
                RetryTrigger.USER_REQUEST,
                2_310L,
            )
        }
        V2IndexingLedgerValidator.requireValid(ledger)
    }

    @Test
    fun `activation source and provider drift durably block only committed results`() {
        listOf(
            TrackFailureCode.SOURCE_FINGERPRINT_CHANGED,
            TrackFailureCode.PROVIDER_SNAPSHOT_CHANGED,
        ).forEachIndexed { index, code ->
            var ledger = committedActivationLedger("activation-drift-$index")
            val trackBefore = ledger.tracks.single()
            val workId = trackBefore.workId

            ledger = V2IndexingLedgerStateMachine.blockCommittedActivationForNewJob(
                ledger = ledger,
                failures = listOf(
                    V2ActivationTrackFailure(workId, code, "identity changed before publish"),
                ),
                nowEpochMs = 2_400L + index,
            )

            val track = ledger.tracks.single()
            val failure = track.failures.single { it.failureId == track.activeFailureId }
            assertEquals(IndexingJobState.WAITING_FOR_INPUT, ledger.state)
            assertEquals(IndexingTrackState.BLOCKED_FAILURE, track.state)
            assertEquals(TrackCheckpoint.COMMITTED, track.checkpoint)
            assertEquals(trackBefore.verifiedArtifacts, track.verifiedArtifacts)
            assertEquals(code, failure.code)
            assertEquals(IndexingStage.DATABASE_ACTIVATION, failure.stage)
            assertEquals(RetryTrigger.NEW_JOB_REQUIRED, failure.retryTrigger)
            V2IndexingLedgerValidator.requireValid(ledger)
        }

        val committed = committedActivationLedger("activation-selection")
        assertEquals(1, V2ActivationSourceSelection.committedDescriptors(committed).size)
        assertTrue(
            V2ActivationSourceSelection.committedDescriptors(
                committed.copy(
                    tracks = committed.tracks.map {
                        it.copy(state = IndexingTrackState.SKIPPED_BY_USER)
                    },
                ),
            ).isEmpty(),
        )
    }

    @Test
    fun `ordinary MERT publication requires exact decoder EOS reconciliation`() {
        var ledger = V2IndexingLedgerStateMachine.startJob(
            plannedLedger(selectedTrack = selectedTrack(providerOffsetMs = 0L, wholeFile = true)),
            2_500L,
        )
        val provisionalWorkId = ledger.tracks.single().workId
        ledger = V2IndexingLedgerStateMachine.beginNextTrackStage(
            ledger,
            provisionalWorkId,
            2_510L,
        )
        ledger = V2IndexingLedgerStateMachine.completeActiveTrackStage(
            ledger,
            provisionalWorkId,
            null,
            2_512L,
        )
        ledger = V2IndexingLedgerStateMachine.beginNextTrackStage(
            ledger,
            provisionalWorkId,
            2_514L,
        )
        val provisionalSpan = ledger.jobSpec.tracks.single().finalizedAudioSpan
        ledger = V2DecodedEosPlanFinalizer.finalizeCanonicalGroup(
            ledger = ledger,
            canonicalWorkId = provisionalWorkId,
            evidence = V2DecodedEosEvidence(
                sourceSampleRateHz = provisionalSpan.container.sampleRateHz,
                observedStartSourceSample = provisionalSpan.startSourceSample,
                observedEndSourceSampleExclusive = provisionalSpan.endSourceSampleExclusive,
                observedSourceSampleCount = provisionalSpan.sourceSampleCount,
                exactSampleCount24k = provisionalSpan.exactSampleCount24k,
                endOfStreamReached = true,
            ),
            nowEpochMs = 2_515L,
        ).ledger
        val workId = ledger.tracks.single().workId
        val goodArtifact = artifact(ledger, VerifiedArtifactKind.MERT_FEATURES, 2_540L)

        assertThrows(InvalidIndexingLedgerException::class.java) {
            V2IndexingLedgerStateMachine.completeActiveTrackStage(
                ledger,
                workId,
                goodArtifact.copy(
                    executionBoundary = goodArtifact.executionBoundary?.copy(
                        endOfStreamReached = false,
                    ),
                ),
                2_540L,
            )
        }
        assertThrows(InvalidIndexingLedgerException::class.java) {
            V2IndexingLedgerStateMachine.completeActiveTrackStage(
                ledger,
                workId,
                goodArtifact.copy(
                    executionBoundary = goodArtifact.executionBoundary?.copy(
                        observedEndSourceSampleExclusive =
                            requireNotNull(goodArtifact.executionBoundary)
                                .observedEndSourceSampleExclusive - 1L,
                    ),
                ),
                2_540L,
            )
        }

        val completed = V2IndexingLedgerStateMachine.completeActiveTrackStage(
            ledger,
            workId,
            goodArtifact,
            2_540L,
        )
        assertEquals(IndexingTrackState.MERT_COMPLETE, completed.tracks.single().state)
    }

    @Test
    fun `one extra decoded sample atomically revises work and retains provisional lineage`() {
        val provisionalSamples = 143_999L
        var ledger = V2IndexingLedgerStateMachine.startJob(
            plannedLedger(
                selectedTrack = selectedTrack(
                    providerOffsetMs = 0L,
                    wholeFile = true,
                    containerDurationUs = V2AudioSpanMath.canonicalTimeUsForSampleBoundary(
                        provisionalSamples,
                        24_000,
                    ),
                ),
            ),
            2_600L,
        )
        val originalSpecId = ledger.jobSpec.specId
        val originalWorkId = ledger.tracks.single().workId
        val provisionalDescriptor = ledger.jobSpec.tracks.single()
        val provisionalProgress = V2IndexingProgress.from(ledger)
        assertEquals(ExpectedTrackWork(1, 1), provisionalDescriptor.expectedWork)
        assertThrows(IllegalArgumentException::class.java) {
            V2IndexingPlanFinalizationPolicy.requireMertReady(provisionalDescriptor)
        }
        assertThrows(IllegalArgumentException::class.java) {
            V2IndexingPlanFinalizationPolicy.requireStagingDatabaseReady(ledger)
        }

        ledger = V2IndexingLedgerStateMachine.beginNextTrackStage(ledger, originalWorkId, 2_610L)
        ledger = V2IndexingLedgerStateMachine.completeActiveTrackStage(
            ledger,
            originalWorkId,
            null,
            2_612L,
        )
        ledger = V2IndexingLedgerStateMachine.beginNextTrackStage(
            ledger,
            originalWorkId,
            2_614L,
        )
        val evidence = decodedEosEvidence(provisionalDescriptor, provisionalSamples + 1L)
        val update = V2DecodedEosPlanFinalizer.finalizeCanonicalGroup(
            ledger,
            originalWorkId,
            evidence,
            2_620L,
        )
        ledger = update.ledger
        val finalDescriptor = ledger.jobSpec.tracks.single()

        assertEquals(originalSpecId, ledger.jobSpec.provisionalParentSpecId)
        assertEquals(
            originalSpecId,
            V2DecodedEosLineage.requirePreflightSpecId(ledger.jobSpec),
        )
        assertEquals(originalWorkId, finalDescriptor.provisionalWorkId)
        assertNotEquals(originalSpecId, ledger.jobSpec.specId)
        assertNotEquals(originalWorkId, finalDescriptor.workId)
        assertEquals(V2AudioSpanAuthority.DECODED_END_OF_STREAM,
            finalDescriptor.finalizedAudioSpan.authority)
        assertEquals(provisionalSamples + 1L,
            finalDescriptor.finalizedAudioSpan.endSourceSampleExclusive)
        assertEquals(ExpectedTrackWork(2, 1), finalDescriptor.expectedWork)
        assertEquals(1L, provisionalProgress.mertWindows.remainingUnits)
        assertEquals(2L, V2IndexingProgress.from(ledger).mertWindows.remainingUnits)
        V2IndexingPlanFinalizationPolicy.requireMertReady(finalDescriptor)
        V2IndexingPlanFinalizationPolicy.requireStagingDatabaseReady(ledger)
        V2IndexingLedgerValidator.requireValid(ledger)

        val replay = V2DecodedEosPlanFinalizer.finalizeCanonicalGroup(
            ledger,
            finalDescriptor.workId,
            evidence,
            2_630L,
        )
        assertEquals(ledger, replay.ledger)
        assertTrue(replay.workIdRemap.isEmpty())
        val staleReplay = V2DecodedEosPlanFinalizer.finalizeCanonicalGroup(
            ledger,
            originalWorkId,
            evidence,
            2_635L,
        )
        assertEquals(ledger, staleReplay.ledger)
        assertEquals(mapOf(originalWorkId to finalDescriptor.workId), staleReplay.workIdRemap)
        val repeatedStaleReplay = V2DecodedEosPlanFinalizer.finalizeCanonicalGroup(
            staleReplay.ledger,
            originalWorkId,
            evidence,
            2_636L,
        )
        assertEquals(staleReplay, repeatedStaleReplay)
        assertThrows(IllegalArgumentException::class.java) {
            V2DecodedEosPlanFinalizer.finalizeCanonicalGroup(
                ledger,
                finalDescriptor.workId,
                decodedEosEvidence(finalDescriptor, provisionalSamples + 2L),
                2_640L,
            )
        }
    }

    @Test
    fun `unknown ordinary duration gains exact immutable work only after decoded EOS`() {
        var ledger = plannedLedger(
            selectedTrack = selectedTrack(
                providerOffsetMs = 0L,
                wholeFile = true,
                providerDurationMs = 0L,
                containerDurationUs = 0L,
            ),
        )
        val provisional = ledger.jobSpec.tracks.single()
        val originalSpecId = ledger.jobSpec.specId

        V2IndexingLedgerValidator.requireValid(ledger)
        assertEquals(
            originalSpecId,
            V2DecodedEosLineage.requirePreflightSpecId(ledger.jobSpec),
        )
        assertTrue(V2UnknownDurationOrdinarySpanPolicy.isUnresolved(
            provisional.finalizedAudioSpan))
        assertEquals(ExpectedTrackWork(0, 0), provisional.expectedWork)
        assertEquals(0L, V2IndexingProgress.from(ledger).mertWindows.remainingUnits)
        assertTrue(
            V2UnmeasuredIndexingWork.UNKNOWN_DURATION_AUDIO_WORK in
                V2IndexingOverallWorkPlanner.snapshot(ledger, null, null)
                    .omittedRemainingWork,
        )

        ledger = V2IndexingLedgerStateMachine.startJob(ledger, 2_700L)
        ledger = V2IndexingLedgerStateMachine.beginNextTrackStage(
            ledger,
            provisional.workId,
            2_710L,
        )
        ledger = V2IndexingLedgerStateMachine.completeActiveTrackStage(
            ledger,
            provisional.workId,
            null,
            2_712L,
        )
        ledger = V2IndexingLedgerStateMachine.beginNextTrackStage(
            ledger,
            provisional.workId,
            2_714L,
        )
        ledger = V2DecodedEosPlanFinalizer.finalizeCanonicalGroup(
            ledger,
            provisional.workId,
            decodedEosEvidence(provisional, 48_000L),
            2_720L,
        ).ledger
        val finalized = ledger.jobSpec.tracks.single()

        assertEquals(originalSpecId, ledger.jobSpec.provisionalParentSpecId)
        assertEquals(
            originalSpecId,
            V2DecodedEosLineage.requirePreflightSpecId(ledger.jobSpec),
        )
        assertEquals(provisional.workId, finalized.provisionalWorkId)
        assertEquals(V2AudioSpanAuthority.DECODED_END_OF_STREAM,
            finalized.finalizedAudioSpan.authority)
        assertEquals(48_000L, finalized.finalizedAudioSpan.exactSampleCount24k)
        assertEquals(ExpectedTrackWork(1, 1), finalized.expectedWork)
        assertFalse(
            V2UnmeasuredIndexingWork.UNKNOWN_DURATION_AUDIO_WORK in
                V2IndexingOverallWorkPlanner.snapshot(ledger, null, null)
                    .omittedRemainingWork,
        )

        val selection = V2IndexingPreflightSelection(
            powerampFileId = provisional.powerampFileId,
            providerPhysicalPath = provisional.providerRow.providerPhysicalPath,
            durationMs = provisional.providerRow.durationMs,
            offsetMs = provisional.providerRow.offsetMs,
            cueSourceImageFolderId = provisional.providerRow.cueSourceImageFolderId,
        )
        val requested = V2IndexingPreflightIntentFactory.create(
            jobId = ledger.jobSpec.jobId,
            selected = listOf(selection),
            rebuildDerivedIndexes = ledger.jobSpec.rebuildDerivedIndexes,
            executionProfile = ledger.executionProfile,
            nowEpochMs = 100L,
        )
        val planning = V2IndexingPreflightIntentStateMachine.beginOrResumePlanning(
            requested,
            "index-generation-v2-" + "0".repeat(64),
            V2IndexingPreflightProgress(
                V2IndexingPreflightPhase.PERSISTING_LEDGER,
                "Persisting ledger",
            ),
            101L,
        )
        val materialized = V2IndexingPreflightIntentStateMachine.materializeResolved(
            V2IndexingPreflightIntentStateMachine.resolveWithExecutableRows(
                current = planning,
                planned = listOf(selection),
                rejected = emptyList(),
                specId = originalSpecId,
                nowEpochMs = 102L,
            ),
            103L,
        )
        val verifiedPreflightSpecId = V2DecodedEosLineage.requirePreflightSpecId(
            ledger.jobSpec,
        )

        assertNotEquals(ledger.jobSpec.specId, verifiedPreflightSpecId)
        assertEquals(
            V2IndexingPreflightLedgerLinkAction.USE_MATERIALIZED_INTENT,
            V2IndexingPreflightLedgerLinkPolicy.action(
                materialized,
                verifiedPreflightSpecId,
            ),
        )
        V2IndexingPreflightLedgerLinkPolicy.requireMaterializedLink(
            materialized,
            verifiedPreflightSpecId,
        )
        assertThrows(IllegalArgumentException::class.java) {
            V2IndexingPreflightLedgerLinkPolicy.action(materialized, ledger.jobSpec.specId)
        }
        assertThrows(IllegalArgumentException::class.java) {
            V2IndexingPreflightLedgerLinkPolicy.requireMaterializedLink(
                materialized,
                ledger.jobSpec.specId,
            )
        }
        V2IndexingLedgerValidator.requireValid(ledger)
    }

    @Test
    fun `decoded EOS lineage rejects recomputed regex-shaped parent and work tampering`() {
        val provisionalSamples = 143_999L
        var ledger = V2IndexingLedgerStateMachine.startJob(
            plannedLedger(
                selectedTrack = selectedTrack(
                    providerOffsetMs = 0L,
                    wholeFile = true,
                    containerDurationUs = V2AudioSpanMath.canonicalTimeUsForSampleBoundary(
                        provisionalSamples,
                        24_000,
                    ),
                ),
            ),
            2_650L,
        )
        val originalWorkId = ledger.tracks.single().workId
        ledger = V2IndexingLedgerStateMachine.beginNextTrackStage(ledger, originalWorkId, 2_651L)
        ledger = V2IndexingLedgerStateMachine.completeActiveTrackStage(
            ledger,
            originalWorkId,
            null,
            2_652L,
        )
        ledger = V2IndexingLedgerStateMachine.beginNextTrackStage(ledger, originalWorkId, 2_653L)
        ledger = V2DecodedEosPlanFinalizer.finalizeCanonicalGroup(
            ledger,
            originalWorkId,
            decodedEosEvidence(ledger.jobSpec.tracks.single(), provisionalSamples + 1L),
            2_654L,
        ).ledger

        var wrongParentSpec = ledger.jobSpec.copy(
            specId = "",
            provisionalParentSpecId = "job-spec-v5-" + "f".repeat(64),
        )
        wrongParentSpec = wrongParentSpec.copy(
            specId = V2IndexingLedgerIds.jobSpecId(wrongParentSpec),
        )
        val wrongParentErrors = V2IndexingLedgerValidator.validate(
            ledger.copy(jobSpec = wrongParentSpec),
        )
        assertTrue(wrongParentErrors.any { it.contains("reconstructable provisional ancestor") })
        assertThrows(IllegalArgumentException::class.java) {
            V2DecodedEosLineage.requirePreflightSpecId(wrongParentSpec)
        }

        val descriptor = ledger.jobSpec.tracks.single()
        var wrongWork = descriptor.copy(
            workId = "",
            provisionalWorkId = "work-v4-" + "e".repeat(64),
        )
        wrongWork = wrongWork.copy(workId = V2IndexingLedgerIds.workId(wrongWork))
        var wrongWorkSpec = ledger.jobSpec.copy(specId = "", tracks = listOf(wrongWork))
        wrongWorkSpec = wrongWorkSpec.copy(
            specId = V2IndexingLedgerIds.jobSpecId(wrongWorkSpec),
        )
        val wrongWorkLedger = ledger.copy(
            jobSpec = wrongWorkSpec,
            tracks = listOf(ledger.tracks.single().copy(workId = wrongWork.workId)),
        )
        val wrongWorkErrors = V2IndexingLedgerValidator.validate(wrongWorkLedger)
        assertTrue(wrongWorkErrors.any { it.contains("exact provisional identity") })
        assertThrows(IllegalArgumentException::class.java) {
            V2DecodedEosLineage.requirePreflightSpecId(wrongWorkSpec)
        }

        val staleCurrentHash = ledger.jobSpec.copy(specId = ledger.jobSpec.provisionalParentSpecId!!)
        assertThrows(IllegalArgumentException::class.java) {
            V2DecodedEosLineage.requirePreflightSpecId(staleCurrentHash)
        }
    }

    @Test
    fun `process restart after EOS rewrite resumes final identity and replays stale id once`() {
        val provisionalSamples = 143_999L
        var ledger = V2IndexingLedgerStateMachine.startJob(
            plannedLedger(
                selectedTrack = selectedTrack(
                    providerOffsetMs = 0L,
                    wholeFile = true,
                    containerDurationUs = V2AudioSpanMath.canonicalTimeUsForSampleBoundary(
                        provisionalSamples,
                        24_000,
                    ),
                ),
            ),
            2_655L,
        )
        val oldWorkId = ledger.tracks.single().workId
        ledger = V2IndexingLedgerStateMachine.beginNextTrackStage(ledger, oldWorkId, 2_656L)
        ledger = V2IndexingLedgerStateMachine.completeActiveTrackStage(
            ledger,
            oldWorkId,
            null,
            2_657L,
        )
        ledger = V2IndexingLedgerStateMachine.beginNextTrackStage(ledger, oldWorkId, 2_658L)
        val evidence = decodedEosEvidence(
            ledger.jobSpec.tracks.single(),
            provisionalSamples + 1L,
        )
        ledger = V2DecodedEosPlanFinalizer.finalizeCanonicalGroup(
            ledger,
            oldWorkId,
            evidence,
            2_659L,
        ).ledger
        val finalWorkId = ledger.tracks.single().workId

        val interrupted = V2IndexingLedgerStateMachine.reconcileAfterProcessRestart(
            ledger,
            2_660L,
        )
        assertTrue(interrupted.changed)
        assertEquals(finalWorkId, interrupted.ledger.tracks.single().workId)
        assertEquals(IndexingTrackState.RETRYABLE_FAILURE,
            interrupted.ledger.tracks.single().state)
        assertEquals(TrackFailureCode.PROCESS_INTERRUPTED,
            interrupted.ledger.tracks.single().failures.single().code)
        val repeatedRestart = V2IndexingLedgerStateMachine.reconcileAfterProcessRestart(
            interrupted.ledger,
            2_661L,
        )
        assertFalse(repeatedRestart.changed)
        assertEquals(interrupted.ledger, repeatedRestart.ledger)

        ledger = V2IndexingLedgerStateMachine.prepareResume(interrupted.ledger, 2_662L)
        ledger = V2IndexingLedgerStateMachine.resume(ledger, 2_663L)
        ledger = V2IndexingLedgerStateMachine.retryTrack(
            ledger,
            finalWorkId,
            RetryTrigger.PROCESS_RESTART,
            2_664L,
        )
        ledger = V2IndexingLedgerStateMachine.beginNextTrackStage(
            ledger,
            finalWorkId,
            2_665L,
        )
        val replay = V2DecodedEosPlanFinalizer.finalizeCanonicalGroup(
            ledger,
            oldWorkId,
            evidence,
            2_666L,
        )
        assertEquals(ledger, replay.ledger)
        assertEquals(mapOf(oldWorkId to finalWorkId), replay.workIdRemap)
        assertEquals(ledger.revision, replay.ledger.revision)
        V2IndexingLedgerValidator.requireValid(replay.ledger)
    }

    @Test
    fun `artifact-free blocked alias is identity-remapped without changing failure evidence`() {
        val provisionalSamples = 143_999L
        val baseInput = selectedTrack(
            providerOffsetMs = 0L,
            wholeFile = true,
            containerDurationUs = V2AudioSpanMath.canonicalTimeUsForSampleBoundary(
                provisionalSamples,
                24_000,
            ),
        )
        val fullFingerprint = baseInput.sourceFingerprint.copy(
            fingerprintSpecId = V2IndexingLedgerIds.FULL_CONTENT_FINGERPRINT_SPEC_ID,
            sampledContentSha256 = null,
            fullContentSha256 = sha('c'),
        )
        val canonicalInput = baseInput.copy(sourceFingerprint = fullFingerprint)
        val aliasPath = "/storage/emulated/0/Music/copy.flac"
        val aliasInput = canonicalInput.copy(
            powerampFileId = 992L,
            providerRow = canonicalInput.providerRow.copy(
                powerampFileId = 992L,
                physicalPath = aliasPath,
                providerPhysicalPath = aliasPath,
                title = "Alias locator",
            ),
            displayMetadata = canonicalInput.displayMetadata.copy(title = "Alias locator"),
            normalizedMetadata = canonicalInput.normalizedMetadata.copy(
                title = "alias locator",
                metadataKey = "artist|album|alias locator|241200",
            ),
            physicalPath = aliasPath,
            finalizedAudioSpan = canonicalInput.finalizedAudioSpan.copy(
                container = canonicalInput.finalizedAudioSpan.container.copy(
                    physicalPath = aliasPath,
                ),
            ),
        )
        val embeddingSpec = V2IndexingLedgerPlanner.createEmbeddingSpec(
            embeddingInput(linkedMapOf("mert" to sha('a'), "clamp3_audio" to sha('b'))),
        )
        var ledger = V2IndexingLedgerPlanner.planJob(
            providerSnapshot = providerSnapshot(),
            embeddingSpec = embeddingSpec,
            textRetrievalSpec = textRetrievalSpec(embeddingSpec),
            runtimeFingerprint = IndexingRuntimeFingerprint(
                appVersionCode = 2_000_000L,
                appBuildId = "v2-test-build",
                decoderRuntimeId = "android-mediacodec-test",
                platformFingerprint = "android-test-device",
            ),
            selectedTracks = listOf(canonicalInput, aliasInput),
            rebuildDerivedIndexes = true,
            createdAtEpochMs = 1_000L,
            jobId = "job-mixed-alias",
        )
        ledger = V2IndexingLedgerStateMachine.startJob(ledger, 2_660L)
        val oldCanonicalId = ledger.jobSpec.tracks[0].workId
        val oldAliasId = ledger.jobSpec.tracks[1].workId
        ledger = V2IndexingLedgerStateMachine.beginNextTrackStage(ledger, oldCanonicalId, 2_661L)
        ledger = V2IndexingLedgerStateMachine.completeActiveTrackStage(
            ledger,
            oldCanonicalId,
            null,
            2_662L,
        )
        ledger = V2IndexingLedgerStateMachine.beginNextTrackStage(ledger, oldAliasId, 2_663L)
        ledger = V2IndexingLedgerStateMachine.recordTrackFailure(
            ledger,
            oldAliasId,
            TrackFailureCode.UNKNOWN_BLOCKED,
            IndexingStage.PREFLIGHT,
            "alias cannot currently be read",
            2_664L,
        )
        val blockedBefore = ledger.tracks[1]
        ledger = V2IndexingLedgerStateMachine.beginNextTrackStage(ledger, oldCanonicalId, 2_665L)
        val beforeFinalization = ledger

        val update = V2DecodedEosPlanFinalizer.finalizeCanonicalGroup(
            ledger,
            oldCanonicalId,
            decodedEosEvidence(ledger.jobSpec.tracks[0], provisionalSamples + 1L),
            2_666L,
        )
        val finalized = update.ledger
        val canonicalAfter = finalized.jobSpec.tracks[0]
        val aliasAfter = finalized.jobSpec.tracks[1]
        val blockedAfter = finalized.tracks[1]
        val remappedFailure = blockedBefore.failures.single().copy(
            failureId = blockedAfter.failures.single().failureId,
        )

        assertEquals(
            mapOf(oldCanonicalId to canonicalAfter.workId, oldAliasId to aliasAfter.workId),
            update.workIdRemap,
        )
        assertEquals(oldAliasId, aliasAfter.provisionalWorkId)
        assertEquals(V2AudioSpanAuthority.DECODED_END_OF_STREAM,
            aliasAfter.finalizedAudioSpan.authority)
        assertEquals(IndexingTrackState.BLOCKED_FAILURE, blockedAfter.state)
        assertEquals(TrackCheckpoint.QUEUED, blockedAfter.checkpoint)
        assertEquals(blockedBefore.updatedAtEpochMs, blockedAfter.updatedAtEpochMs)
        assertEquals(listOf(remappedFailure), blockedAfter.failures)
        assertEquals(remappedFailure.failureId, blockedAfter.activeFailureId)
        assertEquals(
            blockedBefore.copy(
                workId = blockedAfter.workId,
                activeFailureId = blockedAfter.activeFailureId,
                failures = blockedAfter.failures,
            ),
            blockedAfter,
        )
        assertTrue(blockedAfter.verifiedArtifacts.isEmpty())
        assertEquals(1, V2CanonicalAcousticWorkPlanner.groups(finalized.jobSpec).size)
        V2IndexingPlanFinalizationPolicy.requireStagingDatabaseReady(finalized)
        V2IndexingLedgerValidator.requireValid(finalized)
        assertThrows(IllegalArgumentException::class.java) {
            V2DecodedEosPlanFinalizer.requireAllowedMutation(
                beforeFinalization,
                finalized.copy(
                    tracks = finalized.tracks.toMutableList().also { tracks ->
                        tracks[1] = tracks[1].copy(state = IndexingTrackState.SKIPPED_BY_USER)
                    },
                ),
            )
        }
    }

    @Test
    fun `healthy later locator can lead decoded EOS after first duplicate locator fails`() {
        val provisionalSamples = 143_999L
        val baseInput = selectedTrack(
            providerOffsetMs = 0L,
            wholeFile = true,
            containerDurationUs = V2AudioSpanMath.canonicalTimeUsForSampleBoundary(
                provisionalSamples,
                24_000,
            ),
        )
        val fullFingerprint = baseInput.sourceFingerprint.copy(
            fingerprintSpecId = V2IndexingLedgerIds.FULL_CONTENT_FINGERPRINT_SPEC_ID,
            sampledContentSha256 = null,
            fullContentSha256 = sha('d'),
        )
        val firstInput = baseInput.copy(sourceFingerprint = fullFingerprint)
        val secondPath = "/storage/emulated/0/Music/healthy-copy.flac"
        val secondInput = firstInput.copy(
            powerampFileId = 993L,
            providerRow = firstInput.providerRow.copy(
                powerampFileId = 993L,
                physicalPath = secondPath,
                providerPhysicalPath = secondPath,
                title = "Healthy copy",
            ),
            displayMetadata = firstInput.displayMetadata.copy(title = "Healthy copy"),
            normalizedMetadata = firstInput.normalizedMetadata.copy(
                title = "healthy copy",
                metadataKey = "artist|album|healthy copy|241200",
            ),
            physicalPath = secondPath,
            finalizedAudioSpan = firstInput.finalizedAudioSpan.copy(
                container = firstInput.finalizedAudioSpan.container.copy(
                    physicalPath = secondPath,
                ),
            ),
        )
        val embeddingSpec = V2IndexingLedgerPlanner.createEmbeddingSpec(
            embeddingInput(linkedMapOf("mert" to sha('a'), "clamp3_audio" to sha('b'))),
        )
        var ledger = V2IndexingLedgerPlanner.planJob(
            providerSnapshot = providerSnapshot(),
            embeddingSpec = embeddingSpec,
            textRetrievalSpec = textRetrievalSpec(embeddingSpec),
            runtimeFingerprint = IndexingRuntimeFingerprint(
                appVersionCode = 2_000_000L,
                appBuildId = "v2-test-build",
                decoderRuntimeId = "android-mediacodec-test",
                platformFingerprint = "android-test-device",
            ),
            selectedTracks = listOf(firstInput, secondInput),
            rebuildDerivedIndexes = true,
            createdAtEpochMs = 1_000L,
            jobId = "job-later-leader",
        )
        ledger = V2IndexingLedgerStateMachine.startJob(ledger, 2_660L)
        val oldFirstId = ledger.jobSpec.tracks[0].workId
        val oldSecondId = ledger.jobSpec.tracks[1].workId

        ledger = V2IndexingLedgerStateMachine.beginNextTrackStage(ledger, oldFirstId, 2_661L)
        ledger = V2IndexingLedgerStateMachine.completeActiveTrackStage(
            ledger,
            oldFirstId,
            null,
            2_662L,
        )
        ledger = V2IndexingLedgerStateMachine.beginNextTrackStage(ledger, oldFirstId, 2_663L)
        ledger = V2IndexingLedgerStateMachine.recordTrackFailure(
            ledger,
            oldFirstId,
            TrackFailureCode.SOURCE_FINGERPRINT_CHANGED,
            IndexingStage.DECODE_AND_MERT,
            "first locator changed after planning",
            2_664L,
        )
        val failedFirst = ledger.tracks[0]
        ledger = V2IndexingLedgerStateMachine.beginNextTrackStage(ledger, oldSecondId, 2_665L)
        ledger = V2IndexingLedgerStateMachine.completeActiveTrackStage(
            ledger,
            oldSecondId,
            null,
            2_666L,
        )
        assertEquals(
            listOf(oldSecondId),
            V2CanonicalAcousticWorkExecutionPolicy.leadersInState(
                ledger,
                IndexingTrackState.PREFLIGHTED,
                VerifiedArtifactKind.MERT_FEATURES,
            ).map { it.workId },
        )
        ledger = V2IndexingLedgerStateMachine.beginNextTrackStage(ledger, oldSecondId, 2_667L)

        val update = V2DecodedEosPlanFinalizer.finalizeCanonicalGroup(
            ledger,
            oldSecondId,
            decodedEosEvidence(ledger.jobSpec.tracks[1], provisionalSamples + 1L),
            2_668L,
        )
        val finalized = update.ledger
        val firstAfter = finalized.tracks[0]
        val secondAfter = finalized.tracks[1]

        assertEquals(IndexingTrackState.BLOCKED_FAILURE, firstAfter.state)
        assertEquals(IndexingTrackState.DECODING, secondAfter.state)
        assertEquals(failedFirst.failures.single().code, firstAfter.failures.single().code)
        assertEquals(oldFirstId, finalized.jobSpec.tracks[0].provisionalWorkId)
        assertEquals(oldSecondId, finalized.jobSpec.tracks[1].provisionalWorkId)
        assertEquals(
            setOf(oldFirstId, oldSecondId),
            update.workIdRemap.keys,
        )
        assertEquals(
            ledger.jobSpec.specId,
            finalized.jobSpec.provisionalParentSpecId,
        )
        assertEquals(
            ledger.jobSpec.specId,
            V2DecodedEosLineage.requirePreflightSpecId(finalized.jobSpec),
        )
        assertTrue(finalized.jobSpec.tracks.all {
            it.finalizedAudioSpan.authority == V2AudioSpanAuthority.DECODED_END_OF_STREAM
        })
        V2IndexingLedgerValidator.requireValid(finalized)
    }

    @Test
    fun `one fewer decoded sample atomically reduces exact work`() {
        val provisionalSamples = 144_000L
        var ledger = V2IndexingLedgerStateMachine.startJob(
            plannedLedger(
                selectedTrack = selectedTrack(
                    providerOffsetMs = 0L,
                    wholeFile = true,
                    containerDurationUs = V2AudioSpanMath.canonicalTimeUsForSampleBoundary(
                        provisionalSamples,
                        24_000,
                    ),
                ),
            ),
            2_700L,
        )
        val provisionalDescriptor = ledger.jobSpec.tracks.single()
        val originalWorkId = provisionalDescriptor.workId
        assertEquals(ExpectedTrackWork(2, 1), provisionalDescriptor.expectedWork)

        ledger = V2IndexingLedgerStateMachine.beginNextTrackStage(ledger, originalWorkId, 2_710L)
        ledger = V2IndexingLedgerStateMachine.completeActiveTrackStage(
            ledger,
            originalWorkId,
            null,
            2_712L,
        )
        ledger = V2IndexingLedgerStateMachine.beginNextTrackStage(
            ledger,
            originalWorkId,
            2_714L,
        )
        ledger = V2DecodedEosPlanFinalizer.finalizeCanonicalGroup(
            ledger,
            originalWorkId,
            decodedEosEvidence(provisionalDescriptor, provisionalSamples - 1L),
            2_720L,
        ).ledger

        val finalized = ledger.jobSpec.tracks.single()
        assertEquals(provisionalSamples - 1L, finalized.finalizedAudioSpan.sourceSampleCount)
        assertEquals(ExpectedTrackWork(1, 1), finalized.expectedWork)
        assertEquals(1L, V2IndexingProgress.from(ledger).mertWindows.remainingUnits)
        V2IndexingLedgerValidator.requireValid(ledger)
    }

    @Test
    fun `restart reconciliation is idempotent and resumes last verified checkpoint`() {
        var ledger = V2IndexingLedgerStateMachine.startJob(plannedLedger(), 3_000L)
        val workId = ledger.tracks.single().workId
        ledger = V2IndexingLedgerStateMachine.beginNextTrackStage(ledger, workId, 3_010L)
        ledger = V2IndexingLedgerStateMachine.completeActiveTrackStage(
            ledger,
            workId,
            null,
            3_020L,
        )
        ledger = V2IndexingLedgerStateMachine.beginNextTrackStage(ledger, workId, 3_030L)
        ledger = V2IndexingLedgerStateMachine.checkpointActiveStageProgress(
            ledger,
            workId,
            stageProgress(ledger, completedUnits = 12, nowEpochMs = 3_035L),
            3_035L,
        )

        val first = V2IndexingLedgerStateMachine.reconcileAfterProcessRestart(ledger, 3_040L)
        assertTrue(first.changed)
        assertEquals(RestartAction.WAIT_FOR_RESUME, first.action)
        assertEquals(IndexingJobState.INTERRUPTED, first.ledger.state)
        assertEquals(RecoveryPhase.EXECUTION, first.ledger.recoveryPhase)
        val interruptedTrack = first.ledger.tracks.single()
        assertEquals(IndexingTrackState.RETRYABLE_FAILURE, interruptedTrack.state)
        assertEquals(TrackCheckpoint.PREFLIGHTED, interruptedTrack.checkpoint)
        assertEquals(TrackFailureCode.PROCESS_INTERRUPTED, interruptedTrack.failures.single().code)
        assertEquals(12, interruptedTrack.stageProgress?.completedUnits)

        val second = V2IndexingLedgerStateMachine.reconcileAfterProcessRestart(
            first.ledger,
            3_050L,
        )
        assertFalse(second.changed)
        assertEquals(first.ledger, second.ledger)

        ledger = V2IndexingLedgerStateMachine.prepareResume(first.ledger, 3_060L)
        ledger = V2IndexingLedgerStateMachine.resume(ledger, 3_070L)
        ledger = V2IndexingLedgerStateMachine.retryTrack(
            ledger,
            workId,
            RetryTrigger.PROCESS_RESTART,
            3_080L,
        )
        ledger = V2IndexingLedgerStateMachine.beginNextTrackStage(ledger, workId, 3_090L)
        assertEquals(IndexingTrackState.DECODING, ledger.tracks.single().state)
        assertEquals(2, ledger.tracks.single().attemptCount)
    }

    @Test
    fun `planned track admission promotes only queued tracks in one revision`() {
        var mixed = V2IndexingLedgerStateMachine.startJob(
            plannedLedger(
                jobId = "job-admit-mixed",
                selectedTracks = distinctSelectedTracks(3),
            ),
            6_000L,
        )
        val alreadyPreflightedWorkId = mixed.tracks.first().workId
        mixed = V2IndexingLedgerStateMachine.beginNextTrackStage(
            mixed,
            alreadyPreflightedWorkId,
            6_010L,
        )
        mixed = V2IndexingLedgerStateMachine.completeActiveTrackStage(
            mixed,
            alreadyPreflightedWorkId,
            artifact = null,
            nowEpochMs = 6_020L,
        )
        val existingTrack = mixed.tracks.first()
        val queuedWorkIds = mixed.tracks.drop(1).map { it.workId }.toSet()

        val admitted = V2IndexingLedgerStateMachine.admitPlannedTracksForExecution(
            mixed,
            6_030L,
        )

        assertEquals(mixed.revision + 1L, admitted.revision)
        assertEquals(6_030L, admitted.updatedAtEpochMs)
        assertEquals(existingTrack, admitted.tracks.first())
        admitted.tracks.filter { it.workId in queuedWorkIds }.forEach { track ->
            assertEquals(IndexingTrackState.PREFLIGHTED, track.state)
            assertEquals(TrackCheckpoint.PREFLIGHTED, track.checkpoint)
            assertEquals(6_030L, track.updatedAtEpochMs)
            assertEquals(0, track.attemptCount)
            assertEquals(null, track.currentAttemptNumber)
            assertTrue(track.failures.isEmpty())
            assertTrue(track.verifiedArtifacts.isEmpty())
        }
    }

    @Test
    fun `planned track admission starts no attempt until decode begins`() {
        val started = V2IndexingLedgerStateMachine.startJob(
            plannedLedger(jobId = "job-admit-attempt"),
            7_000L,
        )
        val workId = started.tracks.single().workId

        val admitted = V2IndexingLedgerStateMachine.admitPlannedTracksForExecution(
            started,
            7_010L,
        )
        val ready = admitted.tracks.single()
        assertEquals(IndexingTrackState.PREFLIGHTED, ready.state)
        assertEquals(TrackCheckpoint.PREFLIGHTED, ready.checkpoint)
        assertEquals(7_010L, ready.updatedAtEpochMs)
        assertEquals(0, ready.attemptCount)
        assertEquals(null, ready.currentAttemptNumber)

        val decoding = V2IndexingLedgerStateMachine.beginNextTrackStage(
            admitted,
            workId,
            7_020L,
        ).tracks.single()
        assertEquals(IndexingTrackState.DECODING, decoding.state)
        assertEquals(TrackCheckpoint.PREFLIGHTED, decoding.checkpoint)
        assertEquals(1, decoding.attemptCount)
        assertEquals(1, decoding.currentAttemptNumber)
    }

    @Test
    fun `planned track admission is an idempotent no-op without queued tracks`() {
        val started = V2IndexingLedgerStateMachine.startJob(
            plannedLedger(
                jobId = "job-admit-idempotent",
                selectedTracks = distinctSelectedTracks(2),
            ),
            8_000L,
        )
        val admitted = V2IndexingLedgerStateMachine.admitPlannedTracksForExecution(
            started,
            8_010L,
        )

        val repeated = V2IndexingLedgerStateMachine.admitPlannedTracksForExecution(
            admitted,
            9_000L,
        )

        assertTrue(admitted === repeated)
        assertEquals(admitted, repeated)
        assertEquals(admitted.revision, repeated.revision)
        assertEquals(8_010L, repeated.updatedAtEpochMs)
    }

    @Test
    fun `batched track preflight exactly matches sequential success and failure semantics`() {
        val inputs = distinctSelectedTracks(3)
        val started = V2IndexingLedgerStateMachine.startJob(
            plannedLedger(jobId = "job-preflight-oracle", selectedTracks = inputs),
            3_000L,
        )
        val workIds = started.jobSpec.tracks.map { it.workId }
        val outcomes = listOf(
            V2TrackPreflightOutcome.Verified(workIds[0], 3_010L, 3_011L),
            V2TrackPreflightOutcome.Failed(
                workId = workIds[1],
                startedAtEpochMs = 3_012L,
                finishedAtEpochMs = 3_013L,
                failure = V2ClassifiedIndexingFailure(
                    TrackFailureCode.SOURCE_FINGERPRINT_CHANGED,
                    "source bytes changed",
                ),
            ),
            V2TrackPreflightOutcome.Verified(workIds[2], 3_014L, 3_015L),
        )

        var sequential = started
        outcomes.forEach { outcome ->
            sequential = V2IndexingLedgerStateMachine.beginNextTrackStage(
                sequential,
                outcome.workId,
                outcome.startedAtEpochMs,
            )
            sequential = when (outcome) {
                is V2TrackPreflightOutcome.Verified ->
                    V2IndexingLedgerStateMachine.completeActiveTrackStage(
                        sequential,
                        outcome.workId,
                        null,
                        outcome.finishedAtEpochMs,
                    )
                is V2TrackPreflightOutcome.Failed ->
                    V2IndexingLedgerStateMachine.recordTrackFailure(
                        sequential,
                        outcome.workId,
                        outcome.failure.code,
                        IndexingStage.PREFLIGHT,
                        outcome.failure.diagnostic,
                        outcome.finishedAtEpochMs,
                    )
            }
        }

        val batched = V2IndexingLedgerStateMachine.commitTrackPreflightBatch(started, outcomes)

        assertEquals(started.revision + 1L, batched.revision)
        assertEquals(sequential.copy(revision = batched.revision), batched)
        val failure = batched.tracks[1].failures.single()
        assertEquals(TrackFailureCode.SOURCE_FINGERPRINT_CHANGED, failure.code)
        assertEquals(RetryTrigger.NEW_JOB_REQUIRED, failure.retryTrigger)
        assertEquals(TrackCheckpoint.QUEUED, failure.resumeFrom)
    }

    @Test
    fun `batched repeated preflight failure preserves aggregate retry evidence`() {
        val started = V2IndexingLedgerStateMachine.startJob(
            plannedLedger(jobId = "job-preflight-repeat"),
            3_100L,
        )
        val workId = started.tracks.single().workId
        val failure = V2ClassifiedIndexingFailure(
            TrackFailureCode.SOURCE_UNREADABLE,
            "source temporarily unavailable",
        )

        fun failSequential(
            ledger: IndexingJobLedger,
            startedAt: Long,
        ): IndexingJobLedger {
            val active = V2IndexingLedgerStateMachine.beginNextTrackStage(
                ledger,
                workId,
                startedAt,
            )
            return V2IndexingLedgerStateMachine.recordTrackFailure(
                active,
                workId,
                failure.code,
                IndexingStage.PREFLIGHT,
                failure.diagnostic,
                startedAt + 1L,
            )
        }

        fun failBatched(
            ledger: IndexingJobLedger,
            startedAt: Long,
        ): IndexingJobLedger = V2IndexingLedgerStateMachine.commitTrackPreflightBatch(
            ledger,
            listOf(
                V2TrackPreflightOutcome.Failed(
                    workId,
                    startedAt,
                    startedAt + 1L,
                    failure,
                ),
            ),
        )

        var sequential = failSequential(started, 3_110L)
        var batched = failBatched(started, 3_110L)
        sequential = V2IndexingLedgerStateMachine.retryTrack(
            sequential,
            workId,
            RetryTrigger.SOURCE_AVAILABLE,
            3_120L,
        )
        batched = V2IndexingLedgerStateMachine.retryTrack(
            batched,
            workId,
            RetryTrigger.SOURCE_AVAILABLE,
            3_120L,
        )
        sequential = failSequential(sequential, 3_130L)
        batched = failBatched(batched, 3_130L)

        assertEquals(sequential.copy(revision = batched.revision), batched)
        val aggregate = batched.tracks.single().failures.single()
        assertEquals(2, aggregate.occurrences)
        assertEquals(2, aggregate.lastAttemptNumber)
        assertEquals(RetryTrigger.SOURCE_AVAILABLE, aggregate.retryTrigger)
    }

    @Test
    fun `invalid preflight batch cannot expose a partially applied ledger`() {
        val started = V2IndexingLedgerStateMachine.startJob(
            plannedLedger(
                jobId = "job-preflight-atomic",
                selectedTracks = distinctSelectedTracks(2),
            ),
            4_000L,
        )
        val validWorkId = started.jobSpec.tracks.first().workId

        assertThrows(InvalidIndexingLedgerException::class.java) {
            V2IndexingLedgerStateMachine.commitTrackPreflightBatch(
                started,
                listOf(
                    V2TrackPreflightOutcome.Verified(validWorkId, 4_010L, 4_011L),
                    V2TrackPreflightOutcome.Verified("missing-work", 4_012L, 4_013L),
                ),
            )
        }

        assertTrue(started.tracks.all { it.state == IndexingTrackState.QUEUED })
        assertTrue(started.tracks.all { it.attemptCount == 0 })
        assertTrue(started.tracks.all { it.failures.isEmpty() })
    }

    @Test
    fun `process death before a preflight batch commit rechecks without a track failure`() {
        val started = V2IndexingLedgerStateMachine.startJob(
            plannedLedger(
                jobId = "job-preflight-uncommitted",
                selectedTracks = distinctSelectedTracks(2),
            ),
            5_000L,
        )

        val interrupted = V2IndexingLedgerStateMachine.reconcileAfterProcessRestart(
            started,
            5_010L,
        ).ledger

        assertEquals(IndexingJobState.INTERRUPTED, interrupted.state)
        assertTrue(interrupted.tracks.all { it.state == IndexingTrackState.QUEUED })
        assertTrue(interrupted.tracks.all { it.failures.isEmpty() })

        var resumed = V2IndexingLedgerStateMachine.prepareResume(interrupted, 5_020L)
        resumed = V2IndexingLedgerStateMachine.resume(resumed, 5_030L)
        resumed = V2IndexingLedgerStateMachine.commitTrackPreflightBatch(
            resumed,
            resumed.jobSpec.tracks.mapIndexed { index, descriptor ->
                V2TrackPreflightOutcome.Verified(
                    descriptor.workId,
                    5_040L + index * 2L,
                    5_041L + index * 2L,
                )
            },
        )

        assertTrue(resumed.tracks.all { it.state == IndexingTrackState.PREFLIGHTED })
        assertTrue(resumed.tracks.all { it.attemptCount == 1 })
        assertTrue(resumed.tracks.all { it.failures.isEmpty() })
    }

    @Test
    fun `global staging failure is durably recorded at current checkpoint before waiting`() {
        var ledger = V2IndexingLedgerStateMachine.startJob(plannedLedger(), 3_200L)
        val workId = ledger.tracks.single().workId
        ledger = V2IndexingLedgerStateMachine.beginNextTrackStage(ledger, workId, 3_210L)
        ledger = V2IndexingLedgerStateMachine.completeActiveTrackStage(
            ledger,
            workId,
            null,
            3_220L,
        )
        ledger = V2IndexingLedgerStateMachine.beginNextTrackStage(ledger, workId, 3_230L)
        val failure = V2IndexingFailureClassifier.classify(
            V2StagingDatabaseException(
                V2StagingDatabaseFailure.BASE_GENERATION_CHANGED,
                "planned generation is no longer active",
            ),
            IndexingStage.DECODE_AND_MERT,
            ledger.jobSpec.tracks.single().finalizedAudioSpan,
        )
        ledger = V2IndexingLedgerStateMachine.recordTrackFailure(
            ledger,
            workId,
            failure.code,
            IndexingStage.DECODE_AND_MERT,
            failure.diagnostic,
            3_240L,
        )
        ledger = V2IndexingLedgerStateMachine.waitForInput(ledger, 3_250L)

        assertEquals(IndexingJobState.WAITING_FOR_INPUT, ledger.state)
        assertEquals(IndexingTrackState.RETRYABLE_FAILURE, ledger.tracks.single().state)
        assertEquals(TrackFailureCode.DATABASE_GENERATION_CHANGED,
            ledger.tracks.single().failures.single().code)
        assertEquals(TrackCheckpoint.PREFLIGHTED, ledger.tracks.single().checkpoint)
    }

    @Test
    fun `pause can suspend within a track at checksummed partial progress`() {
        var ledger = V2IndexingLedgerStateMachine.startJob(plannedLedger(), 3_500L)
        val workId = ledger.tracks.single().workId
        ledger = V2IndexingLedgerStateMachine.beginNextTrackStage(ledger, workId, 3_510L)
        ledger = V2IndexingLedgerStateMachine.completeActiveTrackStage(
            ledger,
            workId,
            null,
            3_520L,
        )
        ledger = V2IndexingLedgerStateMachine.beginNextTrackStage(ledger, workId, 3_530L)
        ledger = V2IndexingLedgerStateMachine.checkpointActiveStageProgress(
            ledger,
            workId,
            stageProgress(ledger, completedUnits = 18, nowEpochMs = 3_540L),
            3_540L,
        )
        ledger = V2IndexingLedgerStateMachine.requestPause(ledger, 3_550L)
        ledger = V2IndexingLedgerStateMachine.suspendActiveStageForPause(
            ledger,
            workId,
            3_560L,
        )
        ledger = V2IndexingLedgerStateMachine.finishPause(ledger, 3_570L)

        assertEquals(IndexingJobState.PAUSED, ledger.state)
        assertEquals(IndexingTrackState.PREFLIGHTED, ledger.tracks.single().state)
        assertEquals(18, ledger.tracks.single().stageProgress?.completedUnits)

        ledger = V2IndexingLedgerStateMachine.prepareResume(ledger, 3_580L)
        ledger = V2IndexingLedgerStateMachine.resume(ledger, 3_590L)
        ledger = V2IndexingLedgerStateMachine.beginNextTrackStage(ledger, workId, 3_600L)
        assertEquals(IndexingTrackState.DECODING, ledger.tracks.single().state)
        assertEquals(18, ledger.tracks.single().stageProgress?.completedUnits)
    }

    @Test
    fun `bounded decode retry becomes visible blocked failure with aggregate evidence`() {
        var ledger = V2IndexingLedgerStateMachine.startJob(plannedLedger(), 4_000L)
        val workId = ledger.tracks.single().workId
        ledger = V2IndexingLedgerStateMachine.beginNextTrackStage(ledger, workId, 4_010L)
        ledger = V2IndexingLedgerStateMachine.completeActiveTrackStage(
            ledger,
            workId,
            null,
            4_020L,
        )
        ledger = V2IndexingLedgerStateMachine.beginNextTrackStage(ledger, workId, 4_030L)
        ledger = V2IndexingLedgerStateMachine.recordTrackFailure(
            ledger,
            workId,
            TrackFailureCode.CORRUPT_OR_TRUNCATED,
            IndexingStage.DECODE_AND_MERT,
            "malformed packet at 31.5 seconds",
            4_040L,
        )
        assertEquals(IndexingTrackState.RETRYABLE_FAILURE, ledger.tracks.single().state)

        ledger = V2IndexingLedgerStateMachine.retryTrack(
            ledger,
            workId,
            RetryTrigger.IMMEDIATE,
            4_050L,
        )
        ledger = V2IndexingLedgerStateMachine.beginNextTrackStage(ledger, workId, 4_060L)
        ledger = V2IndexingLedgerStateMachine.recordTrackFailure(
            ledger,
            workId,
            TrackFailureCode.CORRUPT_OR_TRUNCATED,
            IndexingStage.DECODE_AND_MERT,
            "fallback decoder confirmed malformed packet",
            4_070L,
        )

        val track = ledger.tracks.single()
        assertEquals(IndexingTrackState.BLOCKED_FAILURE, track.state)
        assertEquals(2, track.attemptCount)
        assertEquals(2, track.failures.single().occurrences)
        assertEquals(RetryTrigger.DECODER_OR_APP_CHANGED, track.failures.single().retryTrigger)
        assertThrows(InvalidIndexingLedgerException::class.java) {
            V2IndexingLedgerStateMachine.retryTrack(
                ledger,
                workId,
                RetryTrigger.IMMEDIATE,
                4_080L,
            )
        }
        V2IndexingLedgerValidator.requireValid(ledger)
    }

    @Test
    fun `cancel retains intent across restart and discards unverified active stage`() {
        var ledger = V2IndexingLedgerStateMachine.startJob(plannedLedger(), 5_000L)
        val workId = ledger.tracks.single().workId
        ledger = V2IndexingLedgerStateMachine.beginNextTrackStage(ledger, workId, 5_010L)
        ledger = V2IndexingLedgerStateMachine.requestCancel(ledger, 5_020L)

        val reconciled = V2IndexingLedgerStateMachine.reconcileAfterProcessRestart(ledger, 5_030L)
        assertEquals(RestartAction.FINISH_CANCELLATION, reconciled.action)
        assertEquals(IndexingJobState.CANCELLING, reconciled.ledger.state)
        assertEquals(IndexingTrackState.QUEUED, reconciled.ledger.tracks.single().state)
        assertTrue(reconciled.ledger.tracks.single().failures.isEmpty())

        ledger = V2IndexingLedgerStateMachine.finishCancel(reconciled.ledger, 5_040L)
        assertEquals(IndexingJobState.CANCELLED, ledger.state)
    }

    @Test
    fun `canonical acoustic work groups duplicate locators by stable span only`() {
        val base = plannedLedger().jobSpec
        val original = base.tracks.single()
        val first = original.copy(
            sourceFingerprint = original.sourceFingerprint.copy(
                fingerprintSpecId = V2IndexingLedgerIds.FULL_CONTENT_FINGERPRINT_SPEC_ID,
                sampledContentSha256 = null,
                fullContentSha256 = sha('c'),
            ),
            stableTrackSpanIdentity = original.stableTrackSpanIdentity.copy(
                strength = StableTrackSpanIdentityStrength.FULL_CONTENT_SHA256,
                contentFingerprintSpecId =
                    V2IndexingLedgerIds.FULL_CONTENT_FINGERPRINT_SPEC_ID,
                contentSha256 = sha('c'),
            ),
        )
        val alias = first.copy(
            workId = "work-v4-" + "8".repeat(64),
            ordinal = 1,
            powerampFileId = first.powerampFileId + 1,
            providerRow = first.providerRow.copy(
                powerampFileId = first.powerampFileId + 1,
                title = "Different locator metadata",
            ),
            displayMetadata = first.displayMetadata.copy(title = "Different locator metadata"),
        )
        val differentSpan = alias.copy(
            workId = "work-v4-" + "9".repeat(64),
            ordinal = 2,
            powerampFileId = first.powerampFileId + 2,
            stableTrackSpanIdentity = first.stableTrackSpanIdentity.copy(
                stableTrackSpanId = "stable-track-span-v1-" + "7".repeat(64),
            ),
            providerRow = alias.providerRow.copy(powerampFileId = first.powerampFileId + 2),
        )

        val groups = V2CanonicalAcousticWorkPlanner.groups(
            base.copy(tracks = listOf(first, alias, differentSpan)),
        )

        assertEquals(2, groups.size)
        assertEquals(first.workId, groups[0].canonical.workId)
        assertEquals(listOf(first.workId, alias.workId), groups[0].members.map { it.workId })
        assertEquals(listOf(differentSpan.workId), groups[1].members.map { it.workId })
    }

    @Test
    fun `sampled identities never alias acoustic work`() {
        val base = plannedLedger().jobSpec
        val first = base.tracks.single()
        val second = first.copy(
            workId = "work-v4-" + "8".repeat(64),
            ordinal = 1,
            powerampFileId = first.powerampFileId + 1,
            providerRow = first.providerRow.copy(powerampFileId = first.powerampFileId + 1),
        )

        val groups = V2CanonicalAcousticWorkPlanner.groups(
            base.copy(tracks = listOf(first, second)),
        )

        assertEquals(2, groups.size)
        assertTrue(groups.all { it.members.size == 1 })
    }

    private fun plannedLedger(
        jobId: String = "job-1",
        providerSnapshot: PowerampProviderSnapshotEvidence = providerSnapshot(),
        embeddingSpec: EmbeddingSpecFingerprint = V2IndexingLedgerPlanner.createEmbeddingSpec(
            embeddingInput(linkedMapOf("mert" to sha('a'), "clamp3_audio" to sha('b'))),
        ),
        textRetrievalSpec: TextRetrievalSpecFingerprint = textRetrievalSpec(embeddingSpec),
        selectedTrack: SelectedTrackInput = selectedTrack(),
        selectedTracks: List<SelectedTrackInput> = listOf(selectedTrack),
        executionProfile: V2IndexingExecutionProfile = V2IndexingExecutionProfile.FULL,
        baseGenerationId: String? = null,
    ): IndexingJobLedger = V2IndexingLedgerPlanner.planJob(
        providerSnapshot = providerSnapshot,
        embeddingSpec = embeddingSpec,
        textRetrievalSpec = textRetrievalSpec,
        runtimeFingerprint = IndexingRuntimeFingerprint(
            appVersionCode = 2_000_000L,
            appBuildId = "v2-test-build",
            decoderRuntimeId = "android-mediacodec-test",
            platformFingerprint = "android-test-device",
        ),
        selectedTracks = selectedTracks,
        rebuildDerivedIndexes = true,
        executionProfile = executionProfile,
        baseGenerationId = baseGenerationId,
        createdAtEpochMs = 1_000L,
        jobId = jobId,
    )

    private fun committedActivationLedger(jobId: String): IndexingJobLedger {
        var ledger = V2IndexingLedgerStateMachine.startJob(
            plannedLedger(jobId = jobId),
            2_200L,
        )
        val workId = ledger.tracks.single().workId
        ledger = V2IndexingLedgerStateMachine.beginNextTrackStage(ledger, workId, 2_210L)
        ledger = V2IndexingLedgerStateMachine.completeActiveTrackStage(
            ledger,
            workId,
            null,
            2_220L,
        )
        listOf(
            VerifiedArtifactKind.MERT_FEATURES,
            VerifiedArtifactKind.CLAMP_VECTOR,
            VerifiedArtifactKind.DATABASE_COMMIT,
        ).forEachIndexed { index, kind ->
            val startedAt = 2_230L + index * 20L
            ledger = V2IndexingLedgerStateMachine.beginNextTrackStage(
                ledger,
                workId,
                startedAt,
            )
            ledger = V2IndexingLedgerStateMachine.completeActiveTrackStage(
                ledger,
                workId,
                artifact(ledger, kind, startedAt + 10L),
                startedAt + 10L,
            )
        }
        return V2IndexingLedgerStateMachine.beginActivation(ledger, 2_300L)
    }

    private fun providerSnapshot() = PowerampProviderSnapshotEvidence(
        libraryGeneration = PROVIDER_GENERATION,
        acquisition = V2ProviderSnapshotAcquisitionEvidence(
            queryUri = "content://com.maxmpz.audioplayer.data/files",
            requestedColumns = listOf("_id", "duration", "path"),
            returnedColumns = listOf("_id", "duration", "path"),
            rowCount = 10_000,
            cursorExhaustedNormally = true,
        ),
    )

    private fun selectedTrack(
        providerOffsetMs: Long = 7_500L,
        wholeFile: Boolean = false,
        providerDurationMs: Long = 241_250L,
        containerDurationUs: Long = if (wholeFile) 241_987_654L else 300_000_000L,
    ): SelectedTrackInput {
        val path = "/storage/emulated/0/Music/image.flac"
        val providerSpan = V2ProviderSpanEvidence(
            offsetUs = providerOffsetMs * 1_000L,
            durationUs = providerDurationMs * 1_000L,
            endExclusiveUs = (providerOffsetMs + providerDurationMs) * 1_000L,
        )
        val startUs = if (wholeFile) 0L else providerSpan.offsetUs
        val endUs = if (wholeFile) containerDurationUs else providerSpan.endExclusiveUs
        val startSample = V2AudioSpanMath.sampleAtOrAfter(startUs, 24_000)
        val endSample = V2AudioSpanMath.sampleAtOrAfter(endUs, 24_000)
        val sourceSamples = endSample - startSample
        val exact24k = V2AudioSpanMath.resampledLength(sourceSamples, 24_000, 24_000)
        return SelectedTrackInput(
            powerampFileId = 991L,
            providerSnapshotGeneration = PROVIDER_GENERATION,
            providerRow = V2ProviderPathRowEvidence(
                powerampFileId = 991L,
                physicalPath = path,
                providerPhysicalPath = path,
                artist = "Artist",
                album = "Album",
                title = "Title",
                offsetMs = providerOffsetMs,
                durationMs = providerDurationMs,
                cueSourceImageFolderId = null,
            ),
            displayMetadata = DisplayTrackMetadata("Artist", "Album", "Title"),
            normalizedMetadata = NormalizedTrackMetadata(
                normalizationSpecId = "track-normalization-v2",
                artist = "artist",
                album = "album",
                title = "title",
                metadataKey = "artist|album|title|241200",
            ),
            physicalPath = path,
            sourceFingerprint = SourceFingerprint(
                fingerprintSpecId = "stat-plus-samples-v1",
                sizeBytes = 123_456_789L,
                lastModifiedEpochMs = 1_700_000_000_000L,
                fileKey = "dev=12;ino=99",
                sampledContentSha256 = sha('c'),
                fullContentSha256 = null,
            ),
            finalizedAudioSpan = FinalizedAudioSpanEvidence(
                kind = if (wholeFile) V2ResolvedAudioSpanKind.WHOLE_FILE
                else V2ResolvedAudioSpanKind.LOGICAL_CUE,
                authority = if (wholeFile) V2AudioSpanAuthority.PROVISIONAL_END_OF_STREAM
                else V2AudioSpanAuthority.PROVIDER_CUE_HALF_OPEN_SPAN,
                executionBoundaryRequirement = if (wholeFile) {
                    V2ExecutionBoundaryRequirement.VERIFY_END_OF_STREAM_AND_RECONCILE
                } else {
                    V2ExecutionBoundaryRequirement.ENFORCE_PROVIDER_HALF_OPEN_SPAN
                },
                providerSpan = providerSpan,
                cueClassification = V2CueClassificationEvidence(
                    providerGroupRowCount = 1,
                    logicalRowCount = 1,
                    nonZeroOffsetRowIds = if (wholeFile) emptyList() else listOf(991L),
                    rawSourceImageRowIds = emptyList(),
                ),
                container = V2AudioContainerEvidence(
                    physicalPath = path,
                    audioTrackIndex = 0,
                    durationUsEstimate = containerDurationUs,
                    durationEstimateSource = if (containerDurationUs > 0L) {
                        V2DurationEstimateSource.CONTAINER_METADATA
                    } else {
                        V2DurationEstimateSource.UNAVAILABLE
                    },
                    sampleRateHz = 24_000,
                    channelCount = 2,
                    mime = "audio/flac",
                ),
                startUs = startUs,
                endExclusiveUs = endUs,
                startSourceSample = startSample,
                endSourceSampleExclusive = endSample,
                sourceSampleCount = sourceSamples,
                exactSampleCount24k = exact24k,
                expectedWork = V2AudioSpanMath.expectedWorkFor24kSamples(exact24k),
            ),
        )
    }

    private fun distinctSelectedTracks(count: Int): List<SelectedTrackInput> {
        require(count in 1..8)
        val hashes = "cdefghij"
        return List(count) { index ->
            val base = selectedTrack()
            val powerampFileId = base.powerampFileId + index
            val path = "/storage/emulated/0/Music/preflight-$index.flac"
            val title = "Preflight $index"
            base.copy(
                powerampFileId = powerampFileId,
                providerRow = base.providerRow.copy(
                    powerampFileId = powerampFileId,
                    physicalPath = path,
                    providerPhysicalPath = path,
                    title = title,
                ),
                displayMetadata = base.displayMetadata.copy(title = title),
                normalizedMetadata = base.normalizedMetadata.copy(
                    title = title.lowercase(),
                    metadataKey = "artist|album|${title.lowercase()}|241200",
                ),
                physicalPath = path,
                sourceFingerprint = base.sourceFingerprint.copy(
                    fileKey = "dev=12;ino=${99 + index}",
                    sampledContentSha256 = sha(hashes[index]),
                ),
                finalizedAudioSpan = base.finalizedAudioSpan.copy(
                    cueClassification = base.finalizedAudioSpan.cueClassification.copy(
                        nonZeroOffsetRowIds = listOf(powerampFileId),
                    ),
                    container = base.finalizedAudioSpan.container.copy(physicalPath = path),
                ),
            )
        }
    }

    private fun embeddingInput(models: Map<String, String>) = EmbeddingSpecInput(
        preprocessingSpecId = V2IndexingWorkPolicy.PREPROCESSING_SPEC_ID,
        decoderPolicyId = V2IndexingWorkPolicy.DECODER_POLICY_ID,
        inferenceBackendPolicyId = V2IndexingWorkPolicy.INFERENCE_BACKEND_POLICY_ID,
        outputDimension = 768,
        modelArtifactSha256 = models,
    )

    private fun textRetrievalSpec(
        audioSpec: EmbeddingSpecFingerprint,
    ): TextRetrievalSpecFingerprint = V2IndexingLedgerPlanner.createTextRetrievalSpec(
        TextRetrievalSpecInput(
            compatibleAudioEmbeddingSpecId = audioSpec.specId,
            textModelSha256 = sha('1'),
            tokenizerModelSha256 = V2IndexingWorkPolicy.TEXT_TOKENIZER_MODEL_SHA256,
            tokenizerPolicyId = V2IndexingWorkPolicy.TEXT_TOKENIZER_POLICY_ID,
            tokenizerRuntimeContractSha256 =
                V2IndexingWorkPolicy.TEXT_TOKENIZER_RUNTIME_CONTRACT_SHA256,
            outputSpaceId = V2IndexingWorkPolicy.TEXT_OUTPUT_SPACE_ID,
            outputDimension = audioSpec.outputDimension,
            inferenceBackendPolicyId = V2IndexingWorkPolicy.TEXT_INFERENCE_BACKEND_POLICY_ID,
        ),
    )

    private fun activationEvidence(
        ledger: IndexingJobLedger,
        activatedAtEpochMs: Long,
    ) = ActivatedGenerationEvidence(
        generationId = "index-generation-v2-" + "1".repeat(64),
        activationBindingId = "activation-binding-v3-" + "2".repeat(64),
        jobSpecId = ledger.jobSpec.specId,
        receiptEmbeddingSpecId = ledger.jobSpec.embeddingSpec.specId,
        textRetrievalSpecId = ledger.jobSpec.textRetrievalSpec.specId,
        baseGenerationId = ledger.jobSpec.baseGenerationId,
        rebuildDerivedIndexes = ledger.jobSpec.rebuildDerivedIndexes,
        manifestSha256 = sha('3'),
        databaseSha256 = sha('4'),
        databaseContentSha256 = sha('5'),
        orderedTrackSetSha256 = sha('6'),
        stableTrackUidMappingSha256 = sha('7'),
        embeddingSha256 = sha('8'),
        graphSha256 = sha('9').takeIf { ledger.jobSpec.rebuildDerivedIndexes },
        activatedAtEpochMs = activatedAtEpochMs,
    )

    private fun artifact(
        ledger: IndexingJobLedger,
        kind: VerifiedArtifactKind,
        nowEpochMs: Long,
        workId: String? = null,
    ): VerifiedArtifact {
        val descriptor = if (workId == null) {
            ledger.jobSpec.tracks.single()
        } else {
            ledger.jobSpec.tracks.single { it.workId == workId }
        }
        val units = when (kind) {
            VerifiedArtifactKind.MERT_FEATURES -> descriptor.expectedWork.mertWindows
            VerifiedArtifactKind.CLAMP_VECTOR -> descriptor.expectedWork.clampSegments
            VerifiedArtifactKind.DATABASE_COMMIT -> 1
        }
        return VerifiedArtifact(
            kind = kind,
            storageKey = "${ledger.jobSpec.jobId}/${descriptor.workId}/${kind.name.lowercase()}",
            byteLength = if (kind == VerifiedArtifactKind.MERT_FEATURES) {
                units * V2_CLAMP3_BLOB_BYTES.toLong()
            } else {
                V2_CLAMP3_BLOB_BYTES.toLong()
            },
            sha256 = sha(
                when (kind) {
                    VerifiedArtifactKind.MERT_FEATURES -> 'd'
                    VerifiedArtifactKind.CLAMP_VECTOR -> 'e'
                    VerifiedArtifactKind.DATABASE_COMMIT -> 'f'
                },
            ),
            completedUnits = units,
            plannedUnits = units,
            embeddingSpecId = ledger.jobSpec.embeddingSpec.specId,
            sourceFingerprint = descriptor.sourceFingerprint,
            verifiedAtEpochMs = nowEpochMs,
            executionBoundary = if (kind == VerifiedArtifactKind.MERT_FEATURES) {
                boundaryEvidence(descriptor)
            } else {
                null
            },
        )
    }

    private fun decodedEosEvidence(
        descriptor: SelectedTrackDescriptor,
        endSourceSampleExclusive: Long,
    ): V2DecodedEosEvidence {
        val sourceSampleCount = endSourceSampleExclusive -
            descriptor.finalizedAudioSpan.startSourceSample
        return V2DecodedEosEvidence(
            sourceSampleRateHz = descriptor.finalizedAudioSpan.container.sampleRateHz,
            observedStartSourceSample = descriptor.finalizedAudioSpan.startSourceSample,
            observedEndSourceSampleExclusive = endSourceSampleExclusive,
            observedSourceSampleCount = sourceSampleCount,
            exactSampleCount24k = V2AudioSpanMath.resampledLength(
                sourceSampleCount,
                descriptor.finalizedAudioSpan.container.sampleRateHz,
                V2AudioSpanMath.TARGET_SAMPLE_RATE_HZ,
            ),
            endOfStreamReached = true,
        )
    }

    private fun boundaryEvidence(
        descriptor: SelectedTrackDescriptor,
    ): VerifiedExecutionBoundaryEvidence {
        val span = descriptor.finalizedAudioSpan
        return VerifiedExecutionBoundaryEvidence(
            requirement = span.executionBoundaryRequirement,
            observedStartSourceSample = span.startSourceSample,
            observedEndSourceSampleExclusive = span.endSourceSampleExclusive,
            observedSourceSampleCount = span.sourceSampleCount,
            exactSampleCount24k = span.exactSampleCount24k,
            endOfStreamReached = span.kind == V2ResolvedAudioSpanKind.WHOLE_FILE,
            providerBoundaryEnforced = span.kind == V2ResolvedAudioSpanKind.LOGICAL_CUE,
        )
    }

    private fun stageProgress(
        ledger: IndexingJobLedger,
        completedUnits: Int,
        nowEpochMs: Long,
    ): VerifiedStageProgress {
        val descriptor = ledger.jobSpec.tracks.single()
        return VerifiedStageProgress(
            stage = IndexingStage.DECODE_AND_MERT,
            storageKey = "${ledger.jobSpec.jobId}/${descriptor.workId}/mert.partial",
            byteLength = completedUnits * 768L * 4L,
            sha256 = sha('9'),
            completedUnits = completedUnits,
            plannedUnits = descriptor.expectedWork.mertWindows,
            resumeCursor = "window:$completedUnits",
            embeddingSpecId = ledger.jobSpec.embeddingSpec.specId,
            sourceFingerprint = descriptor.sourceFingerprint,
            verifiedAtEpochMs = nowEpochMs,
        )
    }

    private fun sha(character: Char): String = character.toString().repeat(64)

    private companion object {
        const val PROVIDER_GENERATION =
            "poweramp-provider-snapshot-v3-sha256:" +
                "7777777777777777777777777777777777777777777777777777777777777777"
    }
}
