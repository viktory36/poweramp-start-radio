package com.powerampstartradio.indexing.v2

import android.content.Context
import android.content.ContextWrapper
import android.database.MatrixCursor
import android.database.sqlite.SQLiteDatabase
import android.os.Process
import android.os.SystemClock
import android.util.AtomicFile
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.powerampstartradio.data.EmbeddingDatabase
import com.powerampstartradio.indexing.NewTrackDetector
import com.powerampstartradio.poweramp.TrackNormalization
import java.io.File
import java.io.OutputStreamWriter
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.charset.StandardCharsets
import java.security.MessageDigest
import java.util.UUID
import kotlin.math.sqrt
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Opt-in connected acceptance for the complete durable executor. Every database, ledger, artifact,
 * graph, and generation pointer lives below a disposable private root; only pinned model and FLAC
 * fixture bytes are read from the installed V2 package.
 */
@RunWith(AndroidJUnit4::class)
class V2IndexingExecutorLifecycleInstrumentedTest {
    @Test
    fun realMediaExtractorPartitionsValidAndCorruptRowsBeforeExecution() {
        val target = InstrumentationRegistry.getInstrumentation().targetContext
        val validSource = File(target.filesDir, "$FIXTURE_ROOT/$SOURCE_RELATIVE_PATH")
        assumeTrue("Opt-in lifecycle source fixture is not staged", validSource.isFile)
        REQUIRED_MODEL_FILES.forEach { name ->
            assumeTrue("Pinned model is not staged: $name", File(target.filesDir, name).isFile)
        }
        assertEquals(SOURCE_SHA256, V2FileSha256.digest(validSource))

        val livePointerBefore = activePointerState(target.filesDir)
        val root = File(target.cacheDir, "v2-mixed-preflight-${UUID.randomUUID()}")
        assertTrue(root.mkdirs())
        try {
            val isolatedContext = IsolatedFilesContext(target, root)
            val testPolicy = V2CurrentModelPolicyResolver.resolve(target.filesDir)
            publishTinyBase(root, testPolicy)
            val corruptSource = File(root, "deliberately-corrupt.flac").apply {
                writeText("this is deliberately not an audio container\n".repeat(128))
            }
            val snapshot = completeTwoRowSnapshot(validSource, corruptSource)
            val selected = snapshot.groups
                .flatMap { it.rows }
                .sortedBy(V2ProviderPathRowEvidence::powerampFileId)
                .map { row ->
                    val source = when (row.powerampFileId) {
                        MIXED_VALID_POWERAMP_ID -> validSource
                        MIXED_CORRUPT_POWERAMP_ID -> corruptSource
                        else -> error("unexpected mixed-preflight row ${row.powerampFileId}")
                    }
                    V2ResolvedTrackSource(
                        track = NewTrackDetector.UnindexedTrack(
                            powerampFileId = row.powerampFileId,
                            artist = TrackNormalization.normalizeArtist(row.artist),
                            album = TrackNormalization.normalizeAlbum(row.album),
                            title = TrackNormalization.normalizeTitle(row.title),
                            durationMs = Math.toIntExact(row.durationMs),
                            path = source.path,
                        ),
                        sourceFile = source,
                    )
                }
            val ledgerDirectory = File(root, "mixed-preflight-ledgers")
            val request = V2IndexingPreflightRequest(
                selectedTracks = selected,
                models = V2ResolvedIndexingModels(
                    mertModelFile = File(target.filesDir, "mert.tflite"),
                    clamp3AudioModelFile = File(target.filesDir, "clamp3_audio.tflite"),
                    clamp3TextModelFile = File(target.filesDir, "clamp3_text.tflite"),
                    sentencePieceModelFile = File(target.filesDir, "sentencepiece.bpe.model"),
                ),
                providerSnapshot = snapshot,
                baseGenerationId = V2IndexGenerationReader.requireActive(root)
                    .manifest.generationId,
                rebuildDerivedIndexes = true,
                jobId = "acceptance-mixed-${UUID.randomUUID()}",
            )
            var prepared: V2PreparedIndexingJob? = null
            val result = V2IndexingJobPreflightPlanner(
                context = isolatedContext,
                ledgerDirectory = ledgerDirectory,
                sourceFingerprinter = V2ExactSourceFingerprinter(),
                audioSpanResolver = V2AudioSpanResolver(V2MediaExtractorAudioInspector()),
            ).resolveAndPersistBlocking(request) { prepared = it }

            assertTrue(result is V2IndexingPreflightResolution.Materialized)
            result as V2IndexingPreflightResolution.Materialized
            assertEquals(1, result.planned.size)
            assertEquals(MIXED_VALID_POWERAMP_ID, result.planned.single().powerampFileId)
            assertEquals(1, result.rejected.size)
            val rejection = result.rejected.single()
            assertEquals(MIXED_CORRUPT_POWERAMP_ID, rejection.selected.powerampFileId)
            assertEquals(
                V2IndexingPreflightFailureCode.UNSUPPORTED_OR_INVALID_AUDIO_CONTAINER,
                rejection.code,
            )
            assertEquals(
                V2IndexingPreflightFailureScope.SELECTED_OCCURRENCE,
                V2IndexingPreflightFailurePolicy.semantics(rejection.code).scope,
            )
            assertEquals(result.planned, prepared?.planned)
            assertEquals(result.rejected, prepared?.rejected)

            val ledger = AtomicV2IndexingLedgerStore(ledgerDirectory).require(result.jobId)
            assertEquals(IndexingJobState.PLANNED, ledger.state)
            assertEquals(1, ledger.jobSpec.tracks.size)
            assertEquals(MIXED_VALID_POWERAMP_ID, ledger.jobSpec.tracks.single().powerampFileId)
            assertEquals(1, ledger.tracks.size)
            assertEquals(
                ledger.jobSpec.tracks.single().workId,
                ledger.tracks.single().workId,
            )
            assertFalse(
                ledger.jobSpec.tracks.any { it.powerampFileId == MIXED_CORRUPT_POWERAMP_ID },
            )
            println(
                "V2_MIXED_PREFLIGHT planned=${result.planned.size} " +
                    "rejected=${result.rejected.size} rejection_code=${rejection.code} " +
                    "ledger_tracks=${ledger.tracks.size}",
            )
        } finally {
            assertTrue("mixed-preflight root cleanup failed", !root.exists() || root.deleteRecursively())
            assertEquals(livePointerBefore, activePointerState(target.filesDir))
        }
    }

    @Test
    fun unknownDurationOrdinaryDecodesOnceFinalizesAndReusesPcmWhileZeroDurationCueIsRejected() {
        val target = InstrumentationRegistry.getInstrumentation().targetContext
        val source = File(target.filesDir, "$FIXTURE_ROOT/$SOURCE_RELATIVE_PATH")
        assumeTrue("Opt-in lifecycle source fixture is not staged", source.isFile)
        REQUIRED_MODEL_FILES.forEach { name ->
            assumeTrue("Pinned model is not staged: $name", File(target.filesDir, name).isFile)
        }
        assertEquals(SOURCE_SHA256, V2FileSha256.digest(source))

        val livePointerBefore = activePointerState(target.filesDir)
        val root = File(target.cacheDir, "v2-unknown-duration-${UUID.randomUUID()}")
        assertTrue(root.mkdirs())
        try {
            val isolatedContext = IsolatedFilesContext(target, root)
            val policy = V2CurrentModelPolicyResolver.resolve(target.filesDir)
            publishTinyBase(root, policy)
            val cueSource = File(root, "zero-duration-cue.flac")
            source.copyTo(cueSource)
            assertEquals(SOURCE_SHA256, V2FileSha256.digest(cueSource))

            val provider = unknownDurationProvider(isolatedContext, source, cueSource)
            val snapshot = provider.acquireBlocking()
            val ordinary = unknownDurationTrack(
                powerampFileId = UNKNOWN_DURATION_POWERAMP_ID,
                title = UNKNOWN_DURATION_TITLE,
                source = source,
            )
            val zeroDurationCue = unknownDurationTrack(
                powerampFileId = ZERO_DURATION_CUE_POWERAMP_ID,
                title = ZERO_DURATION_CUE_TITLE,
                source = cueSource,
                sourceReferenceCount = 2,
                sourceHasLogicalOffsets = true,
            )
            val realInspector = V2MediaExtractorAudioInspector()
            val unavailableDurationInspector = V2AudioContainerInspector { physicalPath ->
                realInspector.inspect(physicalPath).copy(
                    durationUsEstimate = 0L,
                    durationEstimateSource = V2DurationEstimateSource.UNAVAILABLE,
                )
            }
            val jobId = "acceptance-unknown-duration-${UUID.randomUUID()}"
            val resolution = V2IndexingJobPreflightPlanner(
                context = isolatedContext,
                audioSpanResolver = V2AudioSpanResolver(unavailableDurationInspector),
            ).resolveAndPersistBlocking(
                V2IndexingPreflightRequest(
                    selectedTracks = listOf(
                        V2ResolvedTrackSource(ordinary, source),
                        V2ResolvedTrackSource(zeroDurationCue, cueSource),
                    ),
                    models = V2ResolvedIndexingModels(
                        mertModelFile = File(target.filesDir, "mert.tflite"),
                        clamp3AudioModelFile = File(target.filesDir, "clamp3_audio.tflite"),
                        clamp3TextModelFile = File(target.filesDir, "clamp3_text.tflite"),
                        sentencePieceModelFile = File(
                            target.filesDir,
                            "sentencepiece.bpe.model",
                        ),
                    ),
                    providerSnapshot = snapshot,
                    baseGenerationId = V2IndexGenerationReader.requireActive(root)
                        .manifest.generationId,
                    rebuildDerivedIndexes = true,
                    jobId = jobId,
                ),
            )

            assertTrue(resolution is V2IndexingPreflightResolution.Materialized)
            resolution as V2IndexingPreflightResolution.Materialized
            assertEquals(listOf(UNKNOWN_DURATION_POWERAMP_ID), resolution.planned.map {
                it.powerampFileId
            })
            assertEquals(1, resolution.rejected.size)
            assertEquals(ZERO_DURATION_CUE_POWERAMP_ID, resolution.rejected.single()
                .selected.powerampFileId)
            assertEquals(
                V2IndexingPreflightFailureCode.INVALID_LOGICAL_SPAN,
                resolution.rejected.single().code,
            )

            val repository = V2IndexingJobRepository.createIsolated(root)
            repository.reconcileStartup()
            val provisionalLedger = repository.require(jobId)
            val provisionalDescriptor = provisionalLedger.jobSpec.tracks.single()
            assertEquals(UNKNOWN_DURATION_POWERAMP_ID, provisionalDescriptor.powerampFileId)
            assertEquals(0L, provisionalDescriptor.providerRow.durationMs)
            assertEquals(0L, provisionalDescriptor.finalizedAudioSpan.providerSpan.durationUs)
            assertEquals(
                V2DurationEstimateSource.UNAVAILABLE,
                provisionalDescriptor.finalizedAudioSpan.container.durationEstimateSource,
            )
            assertEquals(0L, provisionalDescriptor.finalizedAudioSpan.container.durationUsEstimate)
            assertEquals(
                V2AudioSpanAuthority.PROVISIONAL_END_OF_STREAM,
                provisionalDescriptor.finalizedAudioSpan.authority,
            )
            assertEquals(0L, provisionalDescriptor.finalizedAudioSpan.exactSampleCount24k)
            assertEquals(ExpectedTrackWork(0, 0), provisionalDescriptor.expectedWork)
            val provisionalSpecId = provisionalLedger.jobSpec.specId
            val provisionalWorkId = provisionalDescriptor.workId

            val firstToken = repository.claimExecutor(jobId)
            repository.startAuthorized(firstToken)
            val pauseAfterExactPcm = PauseAfterExactPcmControl(repository, jobId)
            assertThrows(V2IndexingControlFlowException::class.java) {
                executor(
                    context = isolatedContext,
                    repository = repository,
                    provider = provider,
                    modelFilesDir = target.filesDir,
                ).run(firstToken, pauseAfterExactPcm)
            }
            val paused = repository.finishPauseAfterExecutorStops(firstToken)
            repository.releaseExecutor(firstToken)
            assertEquals(IndexingJobState.PAUSED, paused.state)
            assertProgressBounds(pauseAfterExactPcm.events)
            assertEquals(
                1,
                pauseAfterExactPcm.events.countPcmPoint(
                    V2PcmRateMeasurementPoint.MATERIALIZATION_STARTED,
                ),
            )
            assertEquals(
                1,
                pauseAfterExactPcm.events.countPcmPoint(
                    V2PcmRateMeasurementPoint.MATERIALIZATION_COMPLETED_EXACT,
                ),
            )
            assertEquals(
                0,
                pauseAfterExactPcm.events.countPcmPoint(
                    V2PcmRateMeasurementPoint.VERIFIED_CACHE_REUSED_EXACT,
                ),
            )

            val finalizedDescriptor = paused.jobSpec.tracks.single()
            assertNotEquals(provisionalSpecId, paused.jobSpec.specId)
            assertEquals(provisionalSpecId, paused.jobSpec.provisionalParentSpecId)
            assertNotEquals(provisionalWorkId, finalizedDescriptor.workId)
            assertEquals(provisionalWorkId, finalizedDescriptor.provisionalWorkId)
            assertEquals(
                V2IndexingLedgerIds.workId(finalizedDescriptor),
                finalizedDescriptor.workId,
            )
            assertEquals(
                V2IndexingLedgerIds.jobSpecId(paused.jobSpec),
                paused.jobSpec.specId,
            )
            assertEquals(
                V2AudioSpanAuthority.DECODED_END_OF_STREAM,
                finalizedDescriptor.finalizedAudioSpan.authority,
            )
            assertTrue(finalizedDescriptor.finalizedAudioSpan.endExclusiveUs > 0L)
            assertTrue(finalizedDescriptor.finalizedAudioSpan.sourceSampleCount > 0L)
            assertTrue(finalizedDescriptor.finalizedAudioSpan.exactSampleCount24k > 0L)
            assertTrue(finalizedDescriptor.expectedWork.mertWindows > 0)
            assertTrue(finalizedDescriptor.expectedWork.clampSegments > 0)
            assertEquals(0L, finalizedDescriptor.providerRow.durationMs)
            assertEquals(0L, finalizedDescriptor.finalizedAudioSpan.providerSpan.durationUs)
            assertEquals(
                V2DurationEstimateSource.UNAVAILABLE,
                finalizedDescriptor.finalizedAudioSpan.container.durationEstimateSource,
            )

            val artifactDirectory = repository.artifactDirectory(jobId)
            val verifiedBeforeResume = V2VerifiedPcmCacheStore().requireVerified(
                artifactDirectory,
                jobId,
                finalizedDescriptor,
            )
            assertTrue(verifiedBeforeResume.receipt.decoderName.isNotBlank())
            assertTrue(verifiedBeforeResume.receipt.endOfStreamReached)
            assertFalse(verifiedBeforeResume.receipt.logicalBoundaryEnforced)
            assertEquals(
                finalizedDescriptor.finalizedAudioSpan.exactSampleCount24k,
                verifiedBeforeResume.receipt.exactSampleCount24k,
            )
            val verifiedPcmSha256 = verifiedBeforeResume.receipt.pcmSha256

            val resumedRepository = V2IndexingJobRepository.createIsolated(root)
            resumedRepository.reconcileStartup()
            val reopenedDescriptor = resumedRepository.require(jobId).jobSpec.tracks.single()
            assertEquals(finalizedDescriptor, reopenedDescriptor)
            assertEquals(
                verifiedPcmSha256,
                V2VerifiedPcmCacheStore().requireVerified(
                    resumedRepository.artifactDirectory(jobId),
                    jobId,
                    reopenedDescriptor,
                ).receipt.pcmSha256,
            )
            val resumedToken = resumedRepository.claimExecutor(jobId)
            resumedRepository.resumeAuthorized(resumedToken)
            val resumedControl = RecordingControl()
            val outcome = try {
                executor(
                    context = isolatedContext,
                    repository = resumedRepository,
                    provider = provider,
                    modelFilesDir = target.filesDir,
                ).run(resumedToken, resumedControl)
            } finally {
                resumedRepository.releaseExecutor(resumedToken)
            }
            assertEquals(V2IndexingExecutorOutcome.COMPLETE, outcome)
            assertEquals(IndexingJobState.COMPLETE, resumedRepository.require(jobId).state)
            assertEquals(
                0,
                resumedControl.events.countPcmPoint(
                    V2PcmRateMeasurementPoint.MATERIALIZATION_STARTED,
                ),
            )
            assertEquals(
                0,
                resumedControl.events.countPcmPoint(
                    V2PcmRateMeasurementPoint.MATERIALIZATION_COMPLETED_EXACT,
                ),
            )
            assertEquals(
                1,
                resumedControl.events.countPcmPoint(
                    V2PcmRateMeasurementPoint.VERIFIED_CACHE_REUSED_EXACT,
                ),
            )
            assertCompleteProgress(resumedControl.events, requirePcmDecode = false)

            val generation = V2IndexGenerationReader.requireActive(root)
            assertEquals(BASE_TRACK_COUNT + 1, generation.manifest.trackCount)
            assertEquals(V2_CLAMP3_DIMENSION, readCommittedFixtureEmbedding(generation, source).size)
            val normalizedSourcePath = V2StableProviderLexicalPathNormalizer.normalizeAbsolute(
                source.path,
            )
            val providerReceipt = V2ProviderSpanReceiptReader.read(generation.databaseFile)
                .receipts
                .single { receipt ->
                    receipt.providerSpan.normalizedPhysicalPath == normalizedSourcePath
                }
            assertEquals(0L, providerReceipt.providerSpan.offsetMs)
            assertEquals(0L, providerReceipt.providerSpan.durationMs)
            assertFalse(resumedRepository.artifactDirectory(jobId).exists())
            assertStagingDatabaseFamilyAbsent(root, jobId)
            println(
                "V2_UNKNOWN_DURATION provisional_work=$provisionalWorkId " +
                    "final_work=${finalizedDescriptor.workId} " +
                    "exact_24k=${finalizedDescriptor.finalizedAudioSpan.exactSampleCount24k} " +
                    "mert_windows=${finalizedDescriptor.expectedWork.mertWindows} " +
                    "clamp_segments=${finalizedDescriptor.expectedWork.clampSegments} " +
                    "decoder=${verifiedBeforeResume.receipt.decoderName} pcm_sha256=$verifiedPcmSha256",
            )
        } finally {
            assertTrue("unknown-duration root cleanup failed", !root.exists() || root.deleteRecursively())
            assertEquals(livePointerBefore, activePointerState(target.filesDir))
        }
    }

    @Test
    fun failedFirstDuplicateLocatorFallsThroughToHealthyExactCopyOnRealModels() {
        val target = InstrumentationRegistry.getInstrumentation().targetContext
        val source = File(target.filesDir, "$FIXTURE_ROOT/$SOURCE_RELATIVE_PATH")
        val expectedEmbeddingFile = File(target.filesDir, "$FIXTURE_ROOT/$EXPECTED_RELATIVE_PATH")
        assumeTrue("Opt-in lifecycle source fixture is not staged", source.isFile)
        assumeTrue("Opt-in lifecycle expected embedding is not staged", expectedEmbeddingFile.isFile)
        REQUIRED_MODEL_FILES.forEach { name ->
            assumeTrue("Pinned model is not staged: $name", File(target.filesDir, name).isFile)
        }
        assertEquals(SOURCE_SHA256, V2FileSha256.digest(source))
        assertEquals(EXPECTED_SHA256, V2FileSha256.digest(expectedEmbeddingFile))

        val livePointerBefore = activePointerState(target.filesDir)
        val root = File(target.cacheDir, "v2-duplicate-failover-${UUID.randomUUID()}")
        assertTrue(root.mkdirs())
        try {
            val isolatedContext = IsolatedFilesContext(target, root)
            val policy = V2CurrentModelPolicyResolver.resolve(target.filesDir)
            publishTinyBase(root, policy)
            val firstSource = File(root, "duplicate-first.flac")
            val healthySource = File(root, "duplicate-healthy.flac")
            source.copyTo(firstSource)
            source.copyTo(healthySource)
            assertEquals(SOURCE_SHA256, V2FileSha256.digest(firstSource))
            assertEquals(SOURCE_SHA256, V2FileSha256.digest(healthySource))

            val provider = duplicateProvider(isolatedContext, firstSource, healthySource)
            val snapshot = provider.acquireBlocking()
            val selected = listOf(
                duplicateTrack(DUPLICATE_FIRST_POWERAMP_ID, "First duplicate", firstSource),
                duplicateTrack(DUPLICATE_HEALTHY_POWERAMP_ID, "Healthy duplicate", healthySource),
            )
            val jobId = "acceptance-duplicate-failover-${UUID.randomUUID()}"
            val planned = V2IndexingJobPreflightPlanner(isolatedContext).planAndPersistBlocking(
                V2IndexingPreflightRequest(
                    selectedTracks = selected.map { track ->
                        V2ResolvedTrackSource(
                            track,
                            if (track.powerampFileId == DUPLICATE_FIRST_POWERAMP_ID) {
                                firstSource
                            } else {
                                healthySource
                            },
                        )
                    },
                    models = V2ResolvedIndexingModels(
                        mertModelFile = File(target.filesDir, "mert.tflite"),
                        clamp3AudioModelFile = File(target.filesDir, "clamp3_audio.tflite"),
                        clamp3TextModelFile = File(target.filesDir, "clamp3_text.tflite"),
                        sentencePieceModelFile = File(target.filesDir, "sentencepiece.bpe.model"),
                    ),
                    providerSnapshot = snapshot,
                    baseGenerationId = V2IndexGenerationReader.requireActive(root)
                        .manifest.generationId,
                    rebuildDerivedIndexes = true,
                    jobId = jobId,
                ),
            )
            val repository = V2IndexingJobRepository.createIsolated(root)
            repository.reconcileStartup()
            val preflightLedger = repository.require(planned.jobId)
            val group = V2CanonicalAcousticWorkPlanner.groups(preflightLedger.jobSpec).single()
            assertEquals(
                listOf(DUPLICATE_FIRST_POWERAMP_ID, DUPLICATE_HEALTHY_POWERAMP_ID),
                group.members.map { it.powerampFileId },
            )

            var firstMutated = false
            val exactFingerprinter = V2ExactSourceFingerprinter()
            val failFirstAtRuntimeHash = V2SourceFingerprintProvider { candidate ->
                if (!firstMutated && candidate.canonicalPath == firstSource.canonicalPath) {
                    RandomAccessFile(candidate, "rw").use { file ->
                        val original = file.readByte().toInt()
                        file.seek(0L)
                        file.writeByte(original xor 0x01)
                    }
                    firstMutated = true
                }
                exactFingerprinter.fingerprint(candidate)
            }
            val token = repository.claimExecutor(planned.jobId)
            repository.startAuthorized(token)
            val control = RecordingControl()
            val startedNs = System.nanoTime()
            val outcome = try {
                V2IndexingExecutor(
                    context = isolatedContext,
                    repository = repository,
                    providerSnapshots = provider,
                    sourceFingerprinter = failFirstAtRuntimeHash,
                    modelResolver = V2IndexingModelResolver(target.filesDir),
                ).run(token, control)
            } finally {
                repository.releaseExecutor(token)
            }
            val elapsedMs = elapsedMs(startedNs)

            assertTrue("first duplicate was not exercised", firstMutated)
            assertEquals(V2IndexingExecutorOutcome.COMPLETE, outcome)
            val complete = repository.require(planned.jobId)
            assertEquals(IndexingJobState.COMPLETE, complete.state)
            val firstTrack = complete.tracks.single { track ->
                complete.jobSpec.tracks.single { it.workId == track.workId }.powerampFileId ==
                    DUPLICATE_FIRST_POWERAMP_ID
            }
            val healthyTrack = complete.tracks.single { track ->
                complete.jobSpec.tracks.single { it.workId == track.workId }.powerampFileId ==
                    DUPLICATE_HEALTHY_POWERAMP_ID
            }
            assertEquals(IndexingTrackState.BLOCKED_FAILURE, firstTrack.state)
            assertEquals(TrackFailureCode.SOURCE_FINGERPRINT_CHANGED,
                firstTrack.failures.single().code)
            assertTrue(firstTrack.verifiedArtifacts.isEmpty())
            assertEquals(IndexingTrackState.COMMITTED, healthyTrack.state)
            assertTrue(healthyTrack.failures.isEmpty())
            assertEquals(
                setOf(
                    VerifiedArtifactKind.MERT_FEATURES,
                    VerifiedArtifactKind.CLAMP_VECTOR,
                    VerifiedArtifactKind.DATABASE_COMMIT,
                ),
                healthyTrack.verifiedArtifacts.mapTo(linkedSetOf()) { it.kind },
            )
            assertTrue(
                control.events.none { event ->
                    event.workId == firstTrack.workId &&
                        event.stage != V2MeasuredWorkStage.SOURCE_AUDIO_HASH
                },
            )
            assertTrue(control.events.any {
                it.workId == healthyTrack.workId && it.stage == V2MeasuredWorkStage.MERT_WINDOWS
            })
            assertTrue(control.events.any {
                it.workId == healthyTrack.workId && it.stage == V2MeasuredWorkStage.CLAMP_SEGMENTS
            })
            assertTrue(control.events.any {
                it.workId == healthyTrack.workId && it.stage == V2MeasuredWorkStage.DATABASE_COMMITS
            })
            assertProgressBounds(control.events)

            val generation = V2IndexGenerationReader.requireActive(root)
            assertEquals(BASE_TRACK_COUNT + 1, generation.manifest.trackCount)
            val expected = readFloatVector(expectedEmbeddingFile)
            val actual = readCommittedFixtureEmbedding(generation, healthySource)
            val desktopDeviceCosine = cosine(expected, actual)
            assertTrue("duplicate failover cosine $desktopDeviceCosine", desktopDeviceCosine >= 0.990)
            val database = EmbeddingDatabase.open(generation.databaseFile)
            try {
                assertTrue(database.findTracksByPath(firstSource.path).isEmpty())
                assertEquals(1, database.findTracksByPath(healthySource.path).size)
            } finally {
                database.close()
            }
            println(
                "V2_DUPLICATE_FAILOVER elapsed_ms=$elapsedMs " +
                    "failed_id=$DUPLICATE_FIRST_POWERAMP_ID " +
                    "committed_id=$DUPLICATE_HEALTHY_POWERAMP_ID " +
                    "desktop_device_cosine=$desktopDeviceCosine events=${control.events.size}",
            )
        } finally {
            assertTrue("duplicate-failover root cleanup failed", !root.exists() || root.deleteRecursively())
            assertEquals(livePointerBefore, activePointerState(target.filesDir))
        }
    }

    @Test
    fun pauseResumeFailureAndProgressAreExactWithoutTouchingTheActiveGeneration() {
        val target = InstrumentationRegistry.getInstrumentation().targetContext
        val fixtureRoot = File(target.filesDir, FIXTURE_ROOT)
        val source = File(fixtureRoot, SOURCE_RELATIVE_PATH)
        val expectedEmbeddingFile = File(fixtureRoot, EXPECTED_RELATIVE_PATH)
        assumeTrue("Opt-in lifecycle source fixture is not staged", source.isFile)
        assumeTrue("Opt-in lifecycle expected embedding is not staged", expectedEmbeddingFile.isFile)
        REQUIRED_MODEL_FILES.forEach { name ->
            assumeTrue("Pinned model is not staged: $name", File(target.filesDir, name).isFile)
        }
        assertEquals(SOURCE_SHA256, V2FileSha256.digest(source))
        assertEquals(EXPECTED_SHA256, V2FileSha256.digest(expectedEmbeddingFile))

        val livePointerBefore = activePointerState(target.filesDir)
        val root = File(target.cacheDir, "v2-executor-lifecycle-${UUID.randomUUID()}")
        assertTrue(root.mkdirs())
        try {
            val isolatedContext = IsolatedFilesContext(target, root)
            val policy = V2CurrentModelPolicyResolver.resolve(target.filesDir)
            publishTinyBase(root, policy)

            val successProvider = fixtureProvider(
                isolatedContext,
                source,
                powerampFileId = SUCCESS_POWERAMP_ID,
            )
            val successJob = planJob(
                context = isolatedContext,
                modelFilesDir = target.filesDir,
                source = source,
                provider = successProvider,
                powerampFileId = SUCCESS_POWERAMP_ID,
                jobId = "acceptance-resume-${UUID.randomUUID()}",
            )
            val repository = V2IndexingJobRepository.createIsolated(root)
            repository.reconcileStartup()
            val firstToken = repository.claimExecutor(successJob.jobId)
            repository.startAuthorized(firstToken)
            val pausingControl = PauseAfterFirstPcmControl(repository, successJob.jobId)
            val firstExecutor = executor(
                context = isolatedContext,
                repository = repository,
                provider = successProvider,
                modelFilesDir = target.filesDir,
            )
            val interruptedStartedNs = System.nanoTime()
            assertThrows(V2IndexingControlFlowException::class.java) {
                firstExecutor.run(firstToken, pausingControl)
            }
            val interruptedMs = elapsedMs(interruptedStartedNs)
            val paused = repository.finishPauseAfterExecutorStops(firstToken)
            repository.releaseExecutor(firstToken)
            assertEquals(IndexingJobState.PAUSED, paused.state)
            assertEquals(IndexingTrackState.PREFLIGHTED, paused.tracks.single().state)
            assertTrue(paused.tracks.single().verifiedArtifacts.isEmpty())
            assertProgressBounds(pausingControl.events)
            assertTrue(
                "interrupted PCM published unverified private bytes",
                repository.artifactDirectory(successJob.jobId)
                    .walkTopDown()
                    .filter(File::isFile)
                    .none(),
            )

            // A new repository instance exercises on-disk recovery rather than retaining in-memory
            // controller state from the interrupted executor.
            val resumedRepository = V2IndexingJobRepository.createIsolated(root)
            resumedRepository.reconcileStartup()
            val resumedToken = resumedRepository.claimExecutor(successJob.jobId)
            resumedRepository.resumeAuthorized(resumedToken)
            val resumedControl = RecordingControl()
            val resumedExecutor = executor(
                context = isolatedContext,
                repository = resumedRepository,
                provider = successProvider,
                modelFilesDir = target.filesDir,
            )
            val resumedStartedNs = System.nanoTime()
            val outcome = resumedExecutor.run(resumedToken, resumedControl)
            val resumedMs = elapsedMs(resumedStartedNs)
            resumedRepository.releaseExecutor(resumedToken)
            assertEquals(V2IndexingExecutorOutcome.COMPLETE, outcome)
            assertEquals(
                IndexingJobState.COMPLETE,
                resumedRepository.require(successJob.jobId).state,
            )
            assertProgressBounds(resumedControl.events)
            assertCompleteProgress(resumedControl.events)

            val successfulGeneration = V2IndexGenerationReader.requireActive(root)
            assertEquals(BASE_TRACK_COUNT + 1, successfulGeneration.manifest.trackCount)
            val expected = readFloatVector(expectedEmbeddingFile)
            val actual = readCommittedFixtureEmbedding(successfulGeneration, source)
            val desktopDeviceCosine = cosine(expected, actual)
            assertTrue("device executor cosine $desktopDeviceCosine", desktopDeviceCosine >= 0.990)
            assertFalse(resumedRepository.artifactDirectory(successJob.jobId).exists())
            assertStagingDatabaseFamilyAbsent(root, successJob.jobId)

            val failingSource = File(root, "source-changed-after-preflight.flac")
            source.copyTo(failingSource)
            val failureProvider = fixtureProvider(
                isolatedContext,
                failingSource,
                powerampFileId = FAILURE_POWERAMP_ID,
            )
            val failureJob = planJob(
                context = isolatedContext,
                modelFilesDir = target.filesDir,
                source = failingSource,
                provider = failureProvider,
                powerampFileId = FAILURE_POWERAMP_ID,
                jobId = "acceptance-failure-${UUID.randomUUID()}",
            )
            RandomAccessFile(failingSource, "rw").use { file ->
                file.seek(file.length())
                file.write(0)
            }
            val failureRepository = V2IndexingJobRepository.createIsolated(root)
            failureRepository.reconcileStartup()
            val failureToken = failureRepository.claimExecutor(failureJob.jobId)
            failureRepository.startAuthorized(failureToken)
            val failureControl = RecordingControl()
            val generationBeforeFailure = V2IndexGenerationReader.requireActive(root)
            val failureStartedNs = System.nanoTime()
            val failureOutcome = executor(
                context = isolatedContext,
                repository = failureRepository,
                provider = failureProvider,
                modelFilesDir = target.filesDir,
            ).run(failureToken, failureControl)
            val failureMs = elapsedMs(failureStartedNs)
            failureRepository.releaseExecutor(failureToken)
            assertEquals(V2IndexingExecutorOutcome.WAITING_FOR_INPUT, failureOutcome)
            val failed = failureRepository.require(failureJob.jobId)
            assertEquals(IndexingJobState.WAITING_FOR_INPUT, failed.state)
            assertEquals(IndexingTrackState.BLOCKED_FAILURE, failed.tracks.single().state)
            assertEquals(
                TrackFailureCode.SOURCE_FINGERPRINT_CHANGED,
                failed.tracks.single().failures.single().code,
            )
            assertEquals(
                generationBeforeFailure.manifest.generationId,
                V2IndexGenerationReader.requireActive(root).manifest.generationId,
            )
            assertTrue("source failure unexpectedly entered model progress", failureControl.events.isEmpty())
            assertStagingDatabaseFamilyAbsent(root, failureJob.jobId)
            val reopenedFailureRepository = V2IndexingJobRepository.createIsolated(root)
            reopenedFailureRepository.reconcileStartup()
            val reopenedFailure = reopenedFailureRepository.require(failureJob.jobId)
            assertEquals(IndexingJobState.WAITING_FOR_INPUT, reopenedFailure.state)
            assertEquals(IndexingTrackState.BLOCKED_FAILURE, reopenedFailure.tracks.single().state)
            assertEquals(
                TrackFailureCode.SOURCE_FINGERPRINT_CHANGED,
                reopenedFailure.tracks.single().failures.single().code,
            )

            println(
                "V2_INDEXING_LIFECYCLE " +
                    "interrupted_ms=$interruptedMs resumed_ms=$resumedMs failure_ms=$failureMs " +
                    "desktop_device_cosine=$desktopDeviceCosine " +
                    "resume_events=${resumedControl.events.size} " +
                    "generation=${successfulGeneration.manifest.generationId}",
            )
        } finally {
            assertTrue("isolated executor root cleanup failed", !root.exists() || root.deleteRecursively())
            assertEquals(livePointerBefore, activePointerState(target.filesDir))
        }
    }

    @Test
    fun armProcessDeathAfterVerifiedPcmAndWaitForHostKill() {
        val runId = requireProcessDeathRunId()
        val target = InstrumentationRegistry.getInstrumentation().targetContext
        val source = File(target.filesDir, "$FIXTURE_ROOT/$SOURCE_RELATIVE_PATH")
        assumeTrue("Opt-in lifecycle source fixture is not staged", source.isFile)
        REQUIRED_MODEL_FILES.forEach { name ->
            assumeTrue("Pinned model is not staged: $name", File(target.filesDir, name).isFile)
        }

        val rootName = "$PROCESS_DEATH_ROOT_PREFIX$runId"
        val jobId = "$PROCESS_DEATH_JOB_PREFIX$runId"
        val root = File(target.cacheDir, rootName)
        assertFalse("refusing pre-existing process-death root", root.exists())
        assertTrue(root.mkdirs())
        try {
            val isolatedContext = IsolatedFilesContext(target, root)
            val policy = V2CurrentModelPolicyResolver.resolve(target.filesDir)
            publishTinyBase(root, policy)
            val provider = fixtureProvider(isolatedContext, source, PROCESS_DEATH_POWERAMP_ID)
            val job = planJob(
                context = isolatedContext,
                modelFilesDir = target.filesDir,
                source = source,
                provider = provider,
                powerampFileId = PROCESS_DEATH_POWERAMP_ID,
                jobId = jobId,
            )
            val repository = V2IndexingJobRepository.createIsolated(root)
            repository.reconcileStartup()
            val token = repository.claimExecutor(job.jobId)
            repository.startAuthorized(token)
            executor(
                context = isolatedContext,
                repository = repository,
                provider = provider,
                modelFilesDir = target.filesDir,
            ).run(
                token,
                BlockForHostKillAfterVerifiedPcmControl(
                    repository = repository,
                    jobId = job.jobId,
                    root = root,
                    rootName = rootName,
                    runId = runId,
                    token = token,
                    protectedPointerSha256 = activePointerFingerprint(target.filesDir),
                ),
            )
            throw AssertionError("executor returned before the host killed its process")
        } finally {
            // An actual process kill cannot execute this block. It only cleans an unexpected return.
            assertTrue("unexpected arm-phase cleanup failed", !root.exists() || root.deleteRecursively())
        }
    }

    @Test
    fun resumeAfterHostKilledProcessWithoutDecodingPcmAgain() {
        val runId = requireProcessDeathRunId()
        val target = InstrumentationRegistry.getInstrumentation().targetContext
        val source = File(target.filesDir, "$FIXTURE_ROOT/$SOURCE_RELATIVE_PATH")
        val expectedEmbeddingFile = File(target.filesDir, "$FIXTURE_ROOT/$EXPECTED_RELATIVE_PATH")
        assumeTrue("Opt-in lifecycle source fixture is not staged", source.isFile)
        assumeTrue("Opt-in lifecycle expected embedding is not staged", expectedEmbeddingFile.isFile)
        val rootName = "$PROCESS_DEATH_ROOT_PREFIX$runId"
        val jobId = "$PROCESS_DEATH_JOB_PREFIX$runId"
        val root = File(target.cacheDir, rootName)
        val protocol = readProcessDeathProtocol(File(root, PROCESS_DEATH_MARKER))
        assertEquals(PROCESS_DEATH_PROTOCOL_SCHEMA.toString(), protocol.requireValue("schema"))
        assertEquals(runId, protocol.requireValue("run_id"))
        assertEquals(rootName, protocol.requireValue("root_name"))
        assertEquals(jobId, protocol.requireValue("job_id"))
        assertEquals("RUNNING", protocol.requireValue("ledger_state"))
        assertEquals("DECODING", protocol.requireValue("track_state"))
        assertEquals("PREFLIGHTED", protocol.requireValue("track_checkpoint"))
        assertEquals("0", protocol.requireValue("verified_artifact_count"))
        val oldPid = protocol.requireValue("pid").toInt()
        val oldEpoch = protocol.requireValue("lease_epoch").toLong()
        val oldOwner = protocol.requireValue("lease_owner")
        val oldRevision = protocol.requireValue("ledger_revision").toLong()
        assertTrue(oldPid > 0 && oldPid != Process.myPid())
        assertTrue(oldEpoch > 0L && oldRevision >= 0L)
        assertEquals(
            protocol.requireValue("protected_pointer_sha256"),
            activePointerFingerprint(target.filesDir),
        )
        try {
            val isolatedContext = IsolatedFilesContext(target, root)
            val provider = fixtureProvider(isolatedContext, source, PROCESS_DEATH_POWERAMP_ID)
            val repository = V2IndexingJobRepository.createIsolated(root)

            val killedProcessLedger = repository.require(jobId)
            assertEquals(IndexingJobState.RUNNING, killedProcessLedger.state)
            assertEquals(oldRevision, killedProcessLedger.revision)
            val killedProcessTrack = killedProcessLedger.tracks.single()
            assertEquals(IndexingTrackState.DECODING, killedProcessTrack.state)
            assertEquals(TrackCheckpoint.PREFLIGHTED, killedProcessTrack.checkpoint)
            assertTrue(killedProcessTrack.verifiedArtifacts.isEmpty())
            val killedLease = requireNotNull(V2AtomicExecutorLeasePersistence(root).read().active)
            assertEquals(jobId, killedLease.jobId)
            assertEquals(oldEpoch, killedLease.epoch)
            assertEquals(oldOwner, killedLease.ownerInstanceId)
            val killedPcm = V2VerifiedPcmCacheStore().requireVerified(
                repository.artifactDirectory(jobId),
                jobId,
                killedProcessLedger.jobSpec.tracks.single(),
            )
            assertEquals(protocol.requireValue("pcm_sha256"), killedPcm.receipt.pcmSha256)
            assertEquals(protocol.requireValue("pcm_bytes").toLong(), killedPcm.receipt.pcmByteLength)
            assertEquals(
                protocol.requireValue("pcm_path"),
                killedPcm.result.file.canonicalFile.relativeTo(root.canonicalFile).path,
            )

            val reconcileStartedNs = System.nanoTime()
            val reconciliation = repository.reconcileStartup()
            val reconcileMs = elapsedMs(reconcileStartedNs)
            assertEquals(1, reconciliation.reconciledJobs)
            assertEquals(listOf(jobId), reconciliation.resumableJobs)
            assertEquals(null, V2AtomicExecutorLeasePersistence(root).read().active)

            val interrupted = repository.require(jobId)
            assertEquals(IndexingJobState.INTERRUPTED, interrupted.state)
            assertTrue(interrupted.revision > oldRevision)
            val interruptedTrack = interrupted.tracks.single()
            assertEquals(IndexingTrackState.RETRYABLE_FAILURE, interruptedTrack.state)
            assertEquals(
                TrackFailureCode.PROCESS_INTERRUPTED,
                interruptedTrack.failures.single().code,
            )
            assertEquals(RetryTrigger.PROCESS_RESTART, interruptedTrack.failures.single().retryTrigger)
            V2VerifiedPcmCacheStore().requireVerified(
                repository.artifactDirectory(jobId),
                jobId,
                interrupted.jobSpec.tracks.single(),
            )

            val token = repository.claimExecutor(jobId)
            assertTrue("executor lease epoch was not advanced", token.epoch > oldEpoch)
            assertTrue("executor lease owner was reused", token.ownerInstanceId != oldOwner)
            val retry = repository.retryFailed(jobId, RetryTrigger.PROCESS_RESTART)
            assertEquals(1, retry.retried)
            assertEquals(0, retry.notEligible)
            repository.resumeAuthorized(token)
            val control = RecordingControl()
            val resumedStartedNs = System.nanoTime()
            val outcome = try {
                executor(
                    context = isolatedContext,
                    repository = repository,
                    provider = provider,
                    modelFilesDir = target.filesDir,
                ).run(token, control)
            } finally {
                repository.releaseExecutor(token)
            }
            val resumedMs = elapsedMs(resumedStartedNs)
            assertEquals(V2IndexingExecutorOutcome.COMPLETE, outcome)
            assertEquals(IndexingJobState.COMPLETE, repository.require(jobId).state)
            assertTrue(
                "verified PCM was decoded again after process restart",
                control.events.none { event ->
                    event.detail.startsWith("Decoding and resampling") ||
                        event.detail.startsWith("Materialized PCM chunk")
                },
            )
            assertCompleteProgress(control.events, requirePcmDecode = false)

            val generation = V2IndexGenerationReader.requireActive(root)
            val expected = readFloatVector(expectedEmbeddingFile)
            val actual = readCommittedFixtureEmbedding(generation, source)
            val desktopDeviceCosine = cosine(expected, actual)
            assertTrue("process-death cosine $desktopDeviceCosine", desktopDeviceCosine >= 0.990)
            assertFalse(repository.artifactDirectory(jobId).exists())
            assertStagingDatabaseFamilyAbsent(root, jobId)
            println(
                "V2_PROCESS_DEATH reconcile_ms=$reconcileMs resumed_ms=$resumedMs " +
                    "desktop_device_cosine=$desktopDeviceCosine " +
                    "old_pid=$oldPid new_pid=${Process.myPid()} " +
                    "old_epoch=$oldEpoch new_epoch=${token.epoch} " +
                    "resume_events=${control.events.size} " +
                    "generation=${generation.manifest.generationId}",
            )
        } finally {
            assertTrue("process-death root cleanup failed", !root.exists() || root.deleteRecursively())
            assertEquals(
                protocol.requireValue("protected_pointer_sha256"),
                activePointerFingerprint(target.filesDir),
            )
        }
    }

    private fun publishTinyBase(root: File, policy: V2FutureModelPolicy) {
        val database = File(root, "bootstrap.db")
        SQLiteDatabase.openOrCreateDatabase(database, null).use { db ->
            db.execSQL(
                """
                CREATE TABLE tracks (
                    id INTEGER PRIMARY KEY,
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
                    track_id INTEGER PRIMARY KEY,
                    embedding BLOB NOT NULL,
                    FOREIGN KEY(track_id) REFERENCES tracks(id)
                )
                """.trimIndent(),
            )
            repeat(BASE_TRACK_COUNT) { index ->
                val id = index + 1L
                db.execSQL(
                    """
                    INSERT INTO tracks(id, metadata_key, filename_key, artist, album, title,
                                       duration_ms, file_path, source)
                    VALUES (?, ?, ?, 'Fixture', 'Base', ?, 10000, ?, 'desktop')
                    """.trimIndent(),
                    arrayOf<Any>(
                        id,
                        "fixture|base|track$id|10000",
                        "base-track-$id.flac",
                        "Base track $id",
                        "/acceptance/base-track-$id.flac",
                    ),
                )
                val vector = FloatArray(V2_CLAMP3_DIMENSION)
                vector[index] = 1f
                db.execSQL(
                    "INSERT INTO embeddings_clamp3(track_id, embedding) VALUES (?, ?)",
                    arrayOf(id, EmbeddingDatabase.floatArrayToBlob(vector)),
                )
            }
        }
        V2IndexGenerationPublisher(root).publishBootstrapCompatibility(
            privateStagingDatabase = database,
            futureReceiptEmbeddingSpec = policy.receiptEmbeddingSpec,
            textRetrievalSpec = policy.textRetrievalSpec,
        )
    }

    private fun constantHashModelPolicy(): V2FutureModelPolicy {
        val embeddingSpec = V2IndexingLedgerPlanner.createEmbeddingSpec(
            EmbeddingSpecInput(
                preprocessingSpecId = V2IndexingWorkPolicy.PREPROCESSING_SPEC_ID,
                decoderPolicyId = V2IndexingWorkPolicy.DECODER_POLICY_ID,
                inferenceBackendPolicyId = V2IndexingWorkPolicy.INFERENCE_BACKEND_POLICY_ID,
                outputDimension = V2_CLAMP3_DIMENSION,
                modelArtifactSha256 = mapOf(
                    "mert" to ACCEPTANCE_ARTIFACT_SHA256,
                    "clamp3_audio" to ACCEPTANCE_ARTIFACT_SHA256,
                ),
            ),
        )
        val textSpec = V2IndexingLedgerPlanner.createTextRetrievalSpec(
            TextRetrievalSpecInput(
                compatibleAudioEmbeddingSpecId = embeddingSpec.specId,
                textModelSha256 = ACCEPTANCE_ARTIFACT_SHA256,
                tokenizerModelSha256 = V2IndexingWorkPolicy.TEXT_TOKENIZER_MODEL_SHA256,
                tokenizerPolicyId = V2IndexingWorkPolicy.TEXT_TOKENIZER_POLICY_ID,
                tokenizerRuntimeContractSha256 =
                    V2IndexingWorkPolicy.TEXT_TOKENIZER_RUNTIME_CONTRACT_SHA256,
                outputSpaceId = V2IndexingWorkPolicy.TEXT_OUTPUT_SPACE_ID,
                outputDimension = V2_CLAMP3_DIMENSION,
                inferenceBackendPolicyId =
                    V2IndexingWorkPolicy.TEXT_INFERENCE_BACKEND_POLICY_ID,
            ),
        )
        return V2FutureModelPolicy(embeddingSpec, textSpec)
    }

    private fun completeTwoRowSnapshot(
        validSource: File,
        corruptSource: File,
    ): V2ProviderPathGroupSnapshot {
        val rows = listOf(
            V2RawPowerampProviderRow(
                powerampFileId = MIXED_VALID_POWERAMP_ID,
                artist = "Acceptance Artist",
                album = "Acceptance Album",
                title = "Valid MediaExtractor Fixture",
                durationMs = PROVIDER_DURATION_MS,
                folderPath = validSource.parentFile!!.path,
                fileName = validSource.name,
                offsetMs = 0L,
                offsetWasNull = true,
                cueSourceImageFolderId = null,
            ),
            V2RawPowerampProviderRow(
                powerampFileId = MIXED_CORRUPT_POWERAMP_ID,
                artist = "Acceptance Artist",
                album = "Acceptance Album",
                title = "Corrupt MediaExtractor Fixture",
                durationMs = PROVIDER_DURATION_MS,
                folderPath = corruptSource.parentFile!!.path,
                fileName = corruptSource.name,
                offsetMs = 0L,
                offsetWasNull = true,
                cueSourceImageFolderId = null,
            ),
        )
        val acquisition = V2ProviderSnapshotAcquisitionEvidence(
            queryUri = "content://acceptance/complete-two-row-snapshot",
            requestedColumns = listOf("_id", "path", "name", "duration"),
            returnedColumns = listOf("_id", "path", "name", "duration"),
            rowCount = rows.size,
            cursorExhaustedNormally = true,
        )
        return V2PowerampProviderSnapshotAssembler().assembleAfterSuccessfulExhaustion(
            rows,
            acquisition,
        )
    }

    private fun unknownDurationProvider(
        context: Context,
        ordinarySource: File,
        cueSource: File,
    ): V2PowerampProviderSnapshotAcquirer = V2PowerampProviderSnapshotAcquirer(
        context = context,
        providerQuery = V2PowerampProviderQuery { _, _, projection ->
            MatrixCursor(projection).apply {
                addUnknownDurationProviderRow(
                    projection = projection,
                    powerampFileId = UNKNOWN_DURATION_POWERAMP_ID,
                    title = UNKNOWN_DURATION_TITLE,
                    source = ordinarySource,
                    durationMs = 0L,
                    offsetMs = null,
                )
                addUnknownDurationProviderRow(
                    projection = projection,
                    powerampFileId = ZERO_DURATION_CUE_POWERAMP_ID,
                    title = ZERO_DURATION_CUE_TITLE,
                    source = cueSource,
                    durationMs = 0L,
                    offsetMs = null,
                )
                addUnknownDurationProviderRow(
                    projection = projection,
                    powerampFileId = ZERO_DURATION_CUE_SIBLING_POWERAMP_ID,
                    title = "CUE sibling",
                    source = cueSource,
                    durationMs = 1_000L,
                    offsetMs = 1_000L,
                )
            }
        },
    )

    private fun MatrixCursor.addUnknownDurationProviderRow(
        projection: Array<String>,
        powerampFileId: Long,
        title: String,
        source: File,
        durationMs: Long,
        offsetMs: Long?,
    ) {
        addRow(
            projection.map<String, Any?> { column ->
                when (column) {
                    "folder_files._id", "_id" -> powerampFileId
                    "artist" -> UNKNOWN_DURATION_ARTIST
                    "album" -> UNKNOWN_DURATION_ALBUM
                    "title_tag" -> title
                    "folder_files.duration", "duration" -> durationMs
                    "path" -> source.parentFile!!.path
                    "folder_files.name", "name" -> source.name
                    "folder_files.offset_ms", "offset_ms" -> offsetMs
                    "cue_folder_id" -> null
                    else -> error("unexpected unknown-duration provider column $column")
                }
            }.toTypedArray(),
        )
    }

    private fun duplicateProvider(
        context: Context,
        firstSource: File,
        healthySource: File,
    ): V2PowerampProviderSnapshotAcquirer = V2PowerampProviderSnapshotAcquirer(
        context = context,
        providerQuery = V2PowerampProviderQuery { _, _, projection ->
            val returnedColumns = (
                projection.toList() + listOf("_id", "duration", "name", "offset_ms")
            ).distinct()
            MatrixCursor(returnedColumns.toTypedArray()).apply {
                listOf(
                    Triple(DUPLICATE_FIRST_POWERAMP_ID, "First duplicate", firstSource),
                    Triple(DUPLICATE_HEALTHY_POWERAMP_ID, "Healthy duplicate", healthySource),
                ).forEach { (powerampFileId, title, source) ->
                    addRow(
                        returnedColumns.map<String, Any?> { column ->
                            when (column) {
                                "folder_files._id", "_id" -> powerampFileId
                                "artist" -> "Acceptance Artist"
                                "album" -> "Duplicate Failover"
                                "title_tag" -> title
                                "folder_files.duration", "duration" -> PROVIDER_DURATION_MS
                                "path" -> source.parentFile!!.path
                                "folder_files.name", "name" -> source.name
                                "folder_files.offset_ms", "offset_ms", "cue_folder_id" -> null
                                else -> error("unexpected duplicate provider column $column")
                            }
                        }.toTypedArray(),
                    )
                }
            }
        },
    )

    private fun duplicateTrack(
        powerampFileId: Long,
        title: String,
        source: File,
    ) = NewTrackDetector.UnindexedTrack(
        powerampFileId = powerampFileId,
        artist = TrackNormalization.normalizeArtist("Acceptance Artist"),
        album = TrackNormalization.normalizeAlbum("Duplicate Failover"),
        title = TrackNormalization.normalizeTitle(title),
        durationMs = Math.toIntExact(PROVIDER_DURATION_MS),
        path = source.path,
    )

    private fun unknownDurationTrack(
        powerampFileId: Long,
        title: String,
        source: File,
        sourceReferenceCount: Int = 1,
        sourceHasLogicalOffsets: Boolean = false,
    ) = NewTrackDetector.UnindexedTrack(
        powerampFileId = powerampFileId,
        artist = TrackNormalization.normalizeArtist(UNKNOWN_DURATION_ARTIST),
        album = TrackNormalization.normalizeAlbum(UNKNOWN_DURATION_ALBUM),
        title = TrackNormalization.normalizeTitle(title),
        durationMs = 0,
        path = source.path,
        sourceReferenceCount = sourceReferenceCount,
        sourceHasLogicalOffsets = sourceHasLogicalOffsets,
    )

    private fun planJob(
        context: Context,
        modelFilesDir: File,
        source: File,
        provider: V2PowerampProviderSnapshotAcquirer,
        powerampFileId: Long,
        jobId: String,
    ): V2PersistedIndexingJob {
        val snapshot = provider.acquireBlocking()
        val row = snapshot.groups.flatMap { it.rows }.single()
        val track = NewTrackDetector.UnindexedTrack(
            powerampFileId = powerampFileId,
            artist = TrackNormalization.normalizeArtist(row.artist),
            album = TrackNormalization.normalizeAlbum(row.album),
            title = TrackNormalization.normalizeTitle(row.title),
            durationMs = Math.toIntExact(row.durationMs),
            path = source.path,
        )
        return V2IndexingJobPreflightPlanner(context).planAndPersistBlocking(
            V2IndexingPreflightRequest(
                selectedTracks = listOf(V2ResolvedTrackSource(track, source)),
                models = V2ResolvedIndexingModels(
                    mertModelFile = File(modelFilesDir, "mert.tflite"),
                    clamp3AudioModelFile = File(modelFilesDir, "clamp3_audio.tflite"),
                    clamp3TextModelFile = File(modelFilesDir, "clamp3_text.tflite"),
                    sentencePieceModelFile = File(modelFilesDir, "sentencepiece.bpe.model"),
                ),
                providerSnapshot = snapshot,
                baseGenerationId = V2IndexGenerationReader.requireActive(context.filesDir)
                    .manifest.generationId,
                rebuildDerivedIndexes = true,
                jobId = jobId,
            ),
        )
    }

    private fun executor(
        context: Context,
        repository: V2IndexingJobRepository,
        provider: V2PowerampProviderSnapshotAcquirer,
        modelFilesDir: File,
    ): V2IndexingExecutor = V2IndexingExecutor(
        context = context,
        repository = repository,
        providerSnapshots = provider,
        modelResolver = V2IndexingModelResolver(modelFilesDir),
    )

    private fun fixtureProvider(
        context: Context,
        source: File,
        powerampFileId: Long,
    ): V2PowerampProviderSnapshotAcquirer = V2PowerampProviderSnapshotAcquirer(
        context = context,
        providerQuery = V2PowerampProviderQuery { _, _, projection ->
            val returnedColumns = (
                projection.toList() + listOf("_id", "duration", "name", "offset_ms")
            ).distinct()
            MatrixCursor(returnedColumns.toTypedArray()).apply {
                addRow(
                    returnedColumns.map<String, Any?> { column ->
                        when (column) {
                            "folder_files._id", "_id" -> powerampFileId
                            "artist" -> "Acceptance Artist"
                            "album" -> "Acceptance Album"
                            "title_tag" -> "Executor Lifecycle Fixture"
                            "folder_files.duration", "duration" -> PROVIDER_DURATION_MS
                            "path" -> source.parentFile!!.path
                            "folder_files.name", "name" -> source.name
                            "folder_files.offset_ms", "offset_ms", "cue_folder_id" -> null
                            else -> error("unexpected fixture provider column $column")
                        }
                    }.toTypedArray(),
                )
            }
        },
    )

    private fun readCommittedFixtureEmbedding(
        generation: V2ResolvedActiveIndexGeneration,
        source: File,
    ): FloatArray {
        val database = EmbeddingDatabase.open(generation.databaseFile)
        return try {
            val track = database.findTracksByPath(source.path).single()
            requireNotNull(database.getEmbedding(track.id))
        } finally {
            database.close()
        }
    }

    private fun assertProgressBounds(events: List<V2IndexingExecutorEvent>) {
        assertTrue("executor emitted no progress", events.isNotEmpty())
        events.forEach { event ->
            val completed = event.completedUnits
            val total = event.totalUnits
            if (completed != null && total != null) {
                assertTrue("negative progress for ${event.stage}", completed >= 0L)
                assertTrue("non-positive total for ${event.stage}", total > 0L)
                assertTrue(
                    "progress exceeded total for ${event.stage}: $completed/$total",
                    completed <= total,
                )
            }
            CHUNK_PROGRESS.find(event.detail)?.let { match ->
                val completedChunk = match.groupValues[1].toLong()
                val totalChunks = match.groupValues[2].toLong()
                assertTrue("chunk progress exceeded total: ${event.detail}", completedChunk <= totalChunks)
            }
        }
    }

    private fun assertCompleteProgress(
        events: List<V2IndexingExecutorEvent>,
        requirePcmDecode: Boolean = true,
    ) {
        val required = setOf(
            V2MeasuredWorkStage.PCM_24K_SAMPLES,
            V2MeasuredWorkStage.MERT_WINDOWS,
            V2MeasuredWorkStage.CLAMP_SEGMENTS,
            V2MeasuredWorkStage.DATABASE_COMMITS,
            V2MeasuredWorkStage.ACTIVATION_TRACKS,
        )
        required.forEach { stage ->
            val stageEvents = events.filter { it.stage == stage }
            assertTrue("missing required stage $stage", stageEvents.isNotEmpty())
            val bounded = stageEvents.map { event ->
                requireNotNull(event.completedUnits) to requireNotNull(event.totalUnits)
            }
            assertEquals("$stage changed its exact denominator", 1, bounded.map { it.second }.distinct().size)
            if (stage != V2MeasuredWorkStage.PCM_24K_SAMPLES || requirePcmDecode) {
                assertEquals("$stage did not begin at zero", 0L, bounded.first().first)
            }
            assertEquals("$stage did not finish at its exact total", bounded.last().second, bounded.last().first)
            assertTrue(
                "$stage progress moved backward",
                bounded.zipWithNext().all { (before, after) -> after.first >= before.first },
            )
        }
        val chunkProgress = events.mapNotNull { event ->
            CHUNK_PROGRESS.find(event.detail)?.let { match ->
                match.groupValues[1].toLong() to match.groupValues[2].toLong()
            }
        }
        if (requirePcmDecode) {
            assertTrue("no exact PCM chunk progress was emitted", chunkProgress.isNotEmpty())
            assertEquals(
                "PCM chunk progress did not terminate exactly",
                chunkProgress.last().second,
                chunkProgress.last().first,
            )
        } else {
            assertTrue("verified PCM resume emitted decode chunks", chunkProgress.isEmpty())
        }
        assertTrue(
            "staging copy repeated source-database integrity_check",
            events.none { it.detail.contains("integrity_check on the active music index") },
        )
        assertTrue(
            "staging copy repeated the already validated source-database SHA",
            events.none { it.detail.contains("Hashing the active music-index database before copying") },
        )
        assertTrue(
            "new staging database was not checked",
            events.any { it.detail.contains("SQLite quick_check") },
        )
        assertPcmRateMeasurementContract(events, requirePcmDecode)
    }

    private fun assertPcmRateMeasurementContract(
        events: List<V2IndexingExecutorEvent>,
        materializedNow: Boolean,
    ) {
        val measured = events.filter { it.pcmRateMeasurement != null }
        assertTrue("executor emitted no PCM rate-measurement evidence", measured.isNotEmpty())
        val identities = measured.map { requireNotNull(it.pcmRateMeasurement).powerampFileId }.distinct()
        assertEquals("PCM measurement identity changed across work-ID finalization", 1, identities.size)

        val starts = measured.filter {
            it.pcmRateMeasurement?.point == V2PcmRateMeasurementPoint.MATERIALIZATION_STARTED
        }
        val progress = measured.filter {
            it.pcmRateMeasurement?.point == V2PcmRateMeasurementPoint.MATERIALIZATION_PROGRESS
        }
        val completed = measured.filter {
            it.pcmRateMeasurement?.point ==
                V2PcmRateMeasurementPoint.MATERIALIZATION_COMPLETED_EXACT
        }
        val reused = measured.filter {
            it.pcmRateMeasurement?.point == V2PcmRateMeasurementPoint.VERIFIED_CACHE_REUSED_EXACT
        }
        if (materializedNow) {
            assertEquals("fresh PCM materialization did not emit exactly one start", 1, starts.size)
            assertTrue("fresh PCM materialization emitted no progress evidence", progress.isNotEmpty())
            assertEquals("fresh PCM materialization did not emit one exact completion", 1, completed.size)
            assertTrue("fresh PCM materialization claimed cache reuse", reused.isEmpty())
        } else {
            assertTrue("verified PCM reuse emitted a synthetic materialization start", starts.isEmpty())
            assertTrue("verified PCM reuse emitted materialization progress", progress.isEmpty())
            assertTrue("verified PCM reuse claimed a fresh completion", completed.isEmpty())
            assertEquals("verified PCM reuse did not emit one exact reuse event", 1, reused.size)
        }
        val terminal = (completed + reused).single()
        val exact = requireNotNull(terminal.pcmRateMeasurement?.exactSampleCount24k)
        assertEquals("PCM terminal event completed count is not exact", exact, terminal.completedUnits)
        assertEquals("PCM terminal event denominator is not exact", exact, terminal.totalUnits)
    }

    private fun List<V2IndexingExecutorEvent>.countPcmPoint(
        point: V2PcmRateMeasurementPoint,
    ): Int = count { event -> event.pcmRateMeasurement?.point == point }

    private fun assertStagingDatabaseFamilyAbsent(root: File, jobId: String) {
        val directory = File(root, "indexing_v2/job-databases")
        val leftovers = directory.listFiles().orEmpty().filter { file ->
            file.name == "$jobId.db" || file.name.startsWith("$jobId.db-") ||
                file.name.startsWith("$jobId.db.")
        }
        assertTrue("staging database family remains: ${leftovers.map(File::getName)}", leftovers.isEmpty())
    }

    private fun activePointerState(filesDir: File): Map<String, Pair<Long, String>?> {
        val root = File(filesDir, "indexing_v2/generations")
        return listOf(
            "active-generation.json",
            "active-generation.json.bak",
            "active-generation.json.new",
        ).associateWith { name ->
            File(root, name).takeIf(File::isFile)?.let { file ->
                file.length() to V2FileSha256.digest(file)
            }
        }
    }

    private fun activePointerFingerprint(filesDir: File): String {
        val value = activePointerState(filesDir).entries.joinToString("\n") { (name, state) ->
            "$name=${state?.let { "${it.first}:${it.second}" } ?: "absent"}"
        }
        return MessageDigest.getInstance("SHA-256")
            .digest(value.toByteArray(StandardCharsets.UTF_8))
            .joinToString("") { byte -> "%02x".format(byte.toInt() and 0xff) }
    }

    private fun requireProcessDeathRunId(): String {
        val candidate = InstrumentationRegistry.getArguments().getString(PROCESS_DEATH_RUN_ID_ARGUMENT)
        assumeTrue("Opt-in host process-death run id was not supplied", !candidate.isNullOrBlank())
        val runId = requireNotNull(candidate)
        assertTrue("unsafe process-death run id", PROCESS_DEATH_RUN_ID.matches(runId))
        return runId
    }

    private fun readProcessDeathProtocol(file: File): Map<String, String> {
        assertTrue("host-kill marker is absent", file.isFile)
        val values = linkedMapOf<String, String>()
        file.readLines(StandardCharsets.UTF_8).forEach { line ->
            val separator = line.indexOf('=')
            require(separator > 0 && separator < line.lastIndex) { "invalid protocol line" }
            val key = line.substring(0, separator)
            val value = line.substring(separator + 1)
            require(values.put(key, value) == null) { "duplicate protocol key $key" }
        }
        require(values.keys == PROCESS_DEATH_PROTOCOL_KEYS) {
            "unexpected process-death protocol keys: ${values.keys}"
        }
        return values
    }

    private fun Map<String, String>.requireValue(key: String): String =
        requireNotNull(this[key]) { "missing process-death protocol key $key" }

    private open class RecordingControl : V2IndexingExecutorControl {
        val events = mutableListOf<V2IndexingExecutorEvent>()

        override fun throwIfStopped() = Unit

        override fun executionProfile(): V2IndexingExecutionProfile =
            V2IndexingExecutionProfile.FULL

        override fun onProgress(event: V2IndexingExecutorEvent) {
            events += event
        }
    }

    private class PauseAfterFirstPcmControl(
        private val repository: V2IndexingJobRepository,
        private val jobId: String,
    ) : RecordingControl() {
        private var stopped = false

        override fun throwIfStopped() {
            if (stopped) throw V2IndexingControlFlowException("acceptance pause")
        }

        override fun onProgress(event: V2IndexingExecutorEvent) {
            super.onProgress(event)
            if (!stopped && event.stage == V2MeasuredWorkStage.PCM_24K_SAMPLES &&
                (event.completedUnits ?: 0L) > 0L
            ) {
                repository.requestPause(jobId)
                stopped = true
                throw V2IndexingControlFlowException("acceptance pause after decoded PCM")
            }
        }
    }

    private class PauseAfterExactPcmControl(
        private val repository: V2IndexingJobRepository,
        private val jobId: String,
    ) : RecordingControl() {
        private var stopped = false

        override fun throwIfStopped() {
            if (stopped) throw V2IndexingControlFlowException("acceptance exact-PCM pause")
        }

        override fun onProgress(event: V2IndexingExecutorEvent) {
            super.onProgress(event)
            if (!stopped && event.pcmRateMeasurement?.point ==
                V2PcmRateMeasurementPoint.MATERIALIZATION_COMPLETED_EXACT
            ) {
                repository.requestPause(jobId)
                stopped = true
                throw V2IndexingControlFlowException(
                    "acceptance pause after exact verified PCM",
                )
            }
        }
    }

    private class BlockForHostKillAfterVerifiedPcmControl(
        private val repository: V2IndexingJobRepository,
        private val jobId: String,
        private val root: File,
        private val rootName: String,
        private val runId: String,
        private val token: V2ExecutorLeaseToken,
        private val protectedPointerSha256: String,
    ) : RecordingControl() {
        private val marker = AtomicFile(File(root, PROCESS_DEATH_MARKER))
        private var armed = false

        override fun onProgress(event: V2IndexingExecutorEvent) {
            super.onProgress(event)
            if (armed || event.stage != V2MeasuredWorkStage.MERT_WINDOWS ||
                event.completedUnits != 0L
            ) return
            val ledger = repository.require(jobId)
            val track = ledger.tracks.single()
            require(ledger.state == IndexingJobState.RUNNING)
            require(track.state == IndexingTrackState.DECODING)
            require(track.checkpoint == TrackCheckpoint.PREFLIGHTED)
            require(track.verifiedArtifacts.isEmpty())
            val lease = requireNotNull(V2AtomicExecutorLeasePersistence(root).read().active)
            require(lease.token() == token)
            val verified = V2VerifiedPcmCacheStore().requireVerified(
                repository.artifactDirectory(jobId),
                jobId,
                ledger.jobSpec.tracks.single(),
            )
            val relativePcmPath = verified.result.file.canonicalFile
                .relativeTo(root.canonicalFile)
                .path
            require(!relativePcmPath.startsWith(".."))
            writeAtomic(
                marker,
                buildString {
                    appendLine("schema=$PROCESS_DEATH_PROTOCOL_SCHEMA")
                    appendLine("run_id=$runId")
                    appendLine("root_name=$rootName")
                    appendLine("job_id=$jobId")
                    appendLine("pid=${Process.myPid()}")
                    appendLine("lease_epoch=${lease.epoch}")
                    appendLine("lease_owner=${lease.ownerInstanceId}")
                    appendLine("ledger_revision=${ledger.revision}")
                    appendLine("ledger_state=${ledger.state}")
                    appendLine("track_state=${track.state}")
                    appendLine("track_checkpoint=${track.checkpoint}")
                    appendLine("verified_artifact_count=${track.verifiedArtifacts.size}")
                    appendLine("work_id=${track.workId}")
                    appendLine("pcm_sha256=${verified.receipt.pcmSha256}")
                    appendLine("pcm_bytes=${verified.receipt.pcmByteLength}")
                    appendLine("pcm_path=$relativePcmPath")
                    appendLine("protected_pointer_sha256=$protectedPointerSha256")
                },
            )
            armed = true
            val deadline = SystemClock.elapsedRealtime() + PROCESS_DEATH_ARM_WATCHDOG_MS
            while (SystemClock.elapsedRealtime() < deadline) Thread.sleep(250L)
            throw AssertionError("host did not kill the armed process before the watchdog expired")
        }
    }

    private class IsolatedFilesContext(base: Context, private val root: File) :
        ContextWrapper(base) {
        override fun getApplicationContext(): Context = this
        override fun getFilesDir(): File = root
    }

    private fun readFloatVector(file: File): FloatArray {
        val bytes = file.readBytes()
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        return FloatArray(bytes.size / Float.SIZE_BYTES) { buffer.float }
    }

    private fun cosine(a: FloatArray, b: FloatArray): Double {
        require(a.size == b.size)
        var dot = 0.0
        var aa = 0.0
        var bb = 0.0
        for (index in a.indices) {
            dot += a[index] * b[index]
            aa += a[index] * a[index]
            bb += b[index] * b[index]
        }
        return dot / sqrt(aa * bb)
    }

    private fun elapsedMs(startedNs: Long): Long =
        (System.nanoTime() - startedNs) / 1_000_000L

    private companion object {
        fun writeAtomic(file: AtomicFile, value: String) {
            val stream = file.startWrite()
            try {
                val writer = OutputStreamWriter(stream, StandardCharsets.UTF_8)
                writer.write(value)
                writer.flush()
                file.finishWrite(stream)
            } catch (error: Throwable) {
                file.failWrite(stream)
                throw error
            }
        }

        const val FIXTURE_ROOT = "device_acceptance/audio_parity"
        const val SOURCE_RELATIVE_PATH = "source/vice_city_interlude_3.flac"
        const val EXPECTED_RELATIVE_PATH = "expected/vice_city_interlude_3.f32le"
        const val SOURCE_SHA256 =
            "ce0da2f5e54bab63482c0731d73a174ef683ede8e961967dd08f0540238bbfdf"
        const val EXPECTED_SHA256 =
            "66d73a6272f476ce231e46bc7b7e9f1f91d10735d37bd96af454f391e6e68f14"
        const val PROVIDER_DURATION_MS = 10_333L
        const val SUCCESS_POWERAMP_ID = 8_700_000_001L
        const val FAILURE_POWERAMP_ID = 8_700_000_002L
        const val PROCESS_DEATH_POWERAMP_ID = 8_700_000_003L
        const val MIXED_VALID_POWERAMP_ID = 8_700_000_004L
        const val MIXED_CORRUPT_POWERAMP_ID = 8_700_000_005L
        const val UNKNOWN_DURATION_POWERAMP_ID = 8_700_000_006L
        const val ZERO_DURATION_CUE_POWERAMP_ID = 8_700_000_007L
        const val ZERO_DURATION_CUE_SIBLING_POWERAMP_ID = 8_700_000_008L
        const val DUPLICATE_FIRST_POWERAMP_ID = 8_700_000_009L
        const val DUPLICATE_HEALTHY_POWERAMP_ID = 8_700_000_010L
        const val UNKNOWN_DURATION_ARTIST = "Acceptance Artist"
        const val UNKNOWN_DURATION_ALBUM = "Unknown Duration"
        const val UNKNOWN_DURATION_TITLE = "Physical EOS Fixture"
        const val ZERO_DURATION_CUE_TITLE = "Invalid Zero Duration CUE"
        const val ACCEPTANCE_ARTIFACT_SHA256 =
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        const val PROCESS_DEATH_ROOT_PREFIX = "v2-executor-process-death-"
        const val PROCESS_DEATH_JOB_PREFIX = "acceptance-process-death-"
        const val PROCESS_DEATH_MARKER = "host-kill-ready"
        const val PROCESS_DEATH_RUN_ID_ARGUMENT = "v2ProcessDeathRunId"
        const val PROCESS_DEATH_PROTOCOL_SCHEMA = 1
        const val PROCESS_DEATH_ARM_WATCHDOG_MS = 120_000L
        const val BASE_TRACK_COUNT = 6
        val PROCESS_DEATH_RUN_ID = Regex("^[A-Za-z0-9][A-Za-z0-9_-]{0,39}$")
        val PROCESS_DEATH_PROTOCOL_KEYS = linkedSetOf(
            "schema",
            "run_id",
            "root_name",
            "job_id",
            "pid",
            "lease_epoch",
            "lease_owner",
            "ledger_revision",
            "ledger_state",
            "track_state",
            "track_checkpoint",
            "verified_artifact_count",
            "work_id",
            "pcm_sha256",
            "pcm_bytes",
            "pcm_path",
            "protected_pointer_sha256",
        )
        val REQUIRED_MODEL_FILES = listOf(
            "mert.tflite",
            "clamp3_audio.tflite",
            "clamp3_text.tflite",
            "sentencepiece.bpe.model",
        )
        val CHUNK_PROGRESS = Regex("chunk (\\d+)/(\\d+)", RegexOption.IGNORE_CASE)
    }
}
