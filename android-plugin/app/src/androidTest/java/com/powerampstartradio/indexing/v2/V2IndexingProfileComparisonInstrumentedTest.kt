package com.powerampstartradio.indexing.v2

import android.content.Context
import android.content.ContextWrapper
import android.content.Intent
import android.database.MatrixCursor
import android.database.sqlite.SQLiteDatabase
import android.os.BatteryManager
import android.os.Debug
import android.os.PowerManager
import android.os.Process
import android.os.SystemClock
import android.util.AtomicFile
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.google.gson.Gson
import com.google.gson.GsonBuilder
import com.powerampstartradio.data.EmbeddingDatabase
import com.powerampstartradio.indexing.NewTrackDetector
import com.powerampstartradio.poweramp.TrackNormalization
import java.io.File
import java.io.OutputStreamWriter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.charset.StandardCharsets
import java.security.MessageDigest
import java.util.UUID
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * One opt-in, sandboxed indexing-profile observation per instrumentation invocation.
 *
 * The host stages a source below [FIXTURE_ROOT] and supplies every argument. The complete executor
 * runs below a disposable Context root. Installed model bytes and the staged source are read-only;
 * the live generation pointer is fingerprinted before and after the observation.
 */
@RunWith(AndroidJUnit4::class)
class V2IndexingProfileComparisonInstrumentedTest {
    @Test
    fun runOneIsolatedProfileCaseAndPersistEvidence() {
        val arguments = InstrumentationRegistry.getArguments()
        val runId = arguments.getString(ARG_RUN_ID)
        assumeTrue("Opt-in indexing-profile run ID was not supplied", !runId.isNullOrBlank())
        val target = InstrumentationRegistry.getInstrumentation().targetContext
        val spec = ComparisonSpec.fromArguments(requireNotNull(runId), arguments, target.filesDir)
        val source = File(target.filesDir, spec.sourceRelativePath).canonicalFile
        assertTrue("profile-comparison source is missing", source.isFile)
        assertEquals(spec.expectedSourceSha256, V2FileSha256.digest(source))
        REQUIRED_MODEL_FILES.forEach { name ->
            assertTrue("pinned model is missing: $name", File(target.filesDir, name).isFile)
        }
        val modelPolicy = V2CurrentModelPolicyResolver.resolve(target.filesDir)
        val livePointerBefore = activePointerState(target.filesDir)
        val resultFile = File(
            target.filesDir,
            "$FIXTURE_ROOT/results/${spec.resultToken}.json",
        )
        assertFalse("refusing to overwrite profile evidence", resultFile.exists())
        val root = File(target.cacheDir, "v2-index-profile-${spec.resultToken}")
        assertFalse("refusing a pre-existing disposable profile root", root.exists())
        assertTrue(root.mkdirs())

        val evidence = try {
            runComparison(
                target = target,
                root = root,
                source = source,
                spec = spec,
                modelPolicy = modelPolicy,
                livePointerBefore = activePointerBefore(livePointerBefore),
            )
        } finally {
            assertTrue("disposable profile root cleanup failed", !root.exists() || root.deleteRecursively())
            assertEquals(livePointerBefore, activePointerState(target.filesDir))
        }

        resultFile.parentFile?.let { parent ->
            assertTrue(parent.isDirectory || parent.mkdirs())
        }
        writeAtomic(resultFile, GSON.toJson(evidence))
        assertTrue("profile evidence was not published", resultFile.isFile && resultFile.length() > 0L)
        println(
            "V2_INDEX_PROFILE token=${spec.resultToken} profile=${spec.profile.name} " +
                "case=${spec.caseId} wall_ms=${evidence.executorWallMs} " +
                "cpu_ms=${evidence.executorCpuMs} result=${resultFile.relativeTo(target.filesDir)}",
        )
    }

    /**
     * Opt-in FULL-profile batch observation. All selected sources share one immutable job and one
     * executor invocation so model startup, track-to-track scheduling, and final publication are
     * measured as they are in a real indexing run.
     */
    @Test
    fun runOneIsolatedFullBatchAndPersistEvidence() {
        val arguments = InstrumentationRegistry.getArguments()
        val runId = arguments.getString(ARG_RUN_ID)
        assumeTrue("Opt-in indexing-batch run ID was not supplied", !runId.isNullOrBlank())
        val target = InstrumentationRegistry.getInstrumentation().targetContext
        val spec = FullBatchSpec.fromArguments(requireNotNull(runId), arguments, target.filesDir)
        spec.sources.forEach { sourceSpec ->
            assertTrue("batch source is missing: ${sourceSpec.caseId}", sourceSpec.source.isFile)
            assertEquals(sourceSpec.expectedSha256, V2FileSha256.digest(sourceSpec.source))
        }
        REQUIRED_MODEL_FILES.forEach { name ->
            assertTrue("pinned model is missing: $name", File(target.filesDir, name).isFile)
        }
        val modelPolicy = V2CurrentModelPolicyResolver.resolve(target.filesDir)
        val livePointerBefore = activePointerState(target.filesDir)
        val resultFile = File(
            target.filesDir,
            "$FIXTURE_ROOT/results/${spec.resultToken}.json",
        )
        assertFalse("refusing to overwrite batch evidence", resultFile.exists())
        val root = File(target.cacheDir, "v2-index-batch-${spec.resultToken}")
        assertFalse("refusing a pre-existing disposable batch root", root.exists())
        assertTrue(root.mkdirs())

        val evidence = try {
            runFullBatch(
                target = target,
                root = root,
                spec = spec,
                modelPolicy = modelPolicy,
                livePointerBefore = activePointerBefore(livePointerBefore),
            )
        } finally {
            assertTrue("disposable batch root cleanup failed", !root.exists() || root.deleteRecursively())
            assertEquals(livePointerBefore, activePointerState(target.filesDir))
        }

        resultFile.parentFile?.let { parent ->
            assertTrue(parent.isDirectory || parent.mkdirs())
        }
        writeAtomic(resultFile, GSON.toJson(evidence))
        assertTrue("batch evidence was not published", resultFile.isFile && resultFile.length() > 0L)
        println(
            "V2_INDEX_BATCH token=${spec.resultToken} profile=FULL " +
                "pcm_prefetch=${spec.pcmPrefetchEnabled} " +
                "tracks=${evidence.sourceCount} wall_ms=${evidence.executorWallMs} " +
                "cpu_ms=${evidence.executorCpuMs} result=${resultFile.relativeTo(target.filesDir)}",
        )
    }

    private fun runFullBatch(
        target: Context,
        root: File,
        spec: FullBatchSpec,
        modelPolicy: V2FutureModelPolicy,
        livePointerBefore: String,
    ): FullBatchEvidence {
        val totalStartedNs = SystemClock.elapsedRealtimeNanos()
        val initialRuntime = runtimeSnapshot(target)
        val isolatedContext = IsolatedFilesContext(target, root)
        publishTinyBase(root, modelPolicy)
        val provider = fixtureProvider(isolatedContext, spec.sources)
        val realInspector = V2MediaExtractorAudioInspector()
        val endOfStreamInspector = V2AudioContainerInspector { physicalPath ->
            realInspector.inspect(physicalPath).copy(
                durationUsEstimate = 0L,
                durationEstimateSource = V2DurationEstimateSource.UNAVAILABLE,
            )
        }
        val selected = spec.sources.map { sourceSpec ->
            NewTrackDetector.UnindexedTrack(
                powerampFileId = sourceSpec.powerampFileId,
                artist = TrackNormalization.normalizeArtist(ARTIST),
                album = TrackNormalization.normalizeAlbum(BATCH_ALBUM),
                title = TrackNormalization.normalizeTitle(sourceSpec.caseId),
                durationMs = 0,
                path = sourceSpec.source.path,
            )
        }
        val snapshot = provider.acquireBlocking()
        val preflightStartedNs = SystemClock.elapsedRealtimeNanos()
        val resolution = V2IndexingJobPreflightPlanner(
            context = isolatedContext,
            audioSpanResolver = V2AudioSpanResolver(endOfStreamInspector),
        ).resolveAndPersistBlocking(
            V2IndexingPreflightRequest(
                selectedTracks = selected.zip(spec.sources) { track, sourceSpec ->
                    V2ResolvedTrackSource(track, sourceSpec.source)
                },
                models = V2ResolvedIndexingModels(
                    mertModelFile = File(target.filesDir, "mert.tflite"),
                    clamp3AudioModelFile = File(target.filesDir, "clamp3_audio.tflite"),
                    clamp3TextModelFile = File(target.filesDir, "clamp3_text.tflite"),
                    sentencePieceModelFile = File(target.filesDir, "sentencepiece.bpe.model"),
                ),
                providerSnapshot = snapshot,
                baseGenerationId = V2IndexGenerationReader.requireActive(root).manifest.generationId,
                rebuildDerivedIndexes = true,
                executionProfile = V2IndexingExecutionProfile.FULL,
                jobId = "batch-${spec.resultToken}",
            ),
        )
        val preflightElapsedMs = elapsedMs(preflightStartedNs)
        assertTrue("batch preflight did not materialize", resolution is V2IndexingPreflightResolution.Materialized)
        resolution as V2IndexingPreflightResolution.Materialized
        assertEquals(spec.sources.size, resolution.planned.size)
        assertTrue(resolution.rejected.isEmpty())

        val repository = V2IndexingJobRepository.createIsolated(root)
        repository.reconcileStartup()
        val planned = repository.require(resolution.jobId)
        assertEquals(V2IndexingExecutionProfile.FULL, planned.executionProfile)
        assertEquals(spec.sources.size, planned.jobSpec.tracks.size)
        val sourceByPowerampId = spec.sources.associateBy(FullBatchSourceSpec::powerampFileId)
        val sourceByOrdinal = planned.jobSpec.tracks.associate { descriptor ->
            descriptor.ordinal to sourceByPowerampId.getValue(descriptor.powerampFileId)
        }
        val control = FullBatchEvidenceControl(
            repository = repository,
            jobId = resolution.jobId,
            sourceByOrdinal = sourceByOrdinal,
        )
        val token = repository.claimExecutor(resolution.jobId)
        repository.startAuthorized(token)
        val executorStartedNs = SystemClock.elapsedRealtimeNanos()
        val executorStartedCpuMs = Process.getElapsedCpuTime()
        val outcome = try {
            V2IndexingExecutor(
                context = isolatedContext,
                repository = repository,
                providerSnapshots = provider,
                modelResolver = V2IndexingModelResolver(target.filesDir),
                pcmPrefetchEnabled = spec.pcmPrefetchEnabled,
            ).run(token, control)
        } finally {
            repository.releaseExecutor(token)
        }
        val executorCpuMs = Process.getElapsedCpuTime() - executorStartedCpuMs
        val executorWallMs = elapsedMs(executorStartedNs)
        assertEquals(V2IndexingExecutorOutcome.COMPLETE, outcome)
        val complete = repository.require(resolution.jobId)
        assertEquals(IndexingJobState.COMPLETE, complete.state)
        assertEquals(V2IndexingExecutionProfile.FULL, complete.executionProfile)
        assertEquals(spec.sources.size, complete.tracks.size)

        val generation = V2IndexGenerationReader.requireActive(root)
        val database = EmbeddingDatabase.open(generation.databaseFile)
        val trackEvidence = try {
            spec.sources.map { sourceSpec ->
                val descriptor = complete.jobSpec.tracks.single {
                    it.powerampFileId == sourceSpec.powerampFileId
                }
                val ledgerTrack = complete.tracks.single { it.workId == descriptor.workId }
                assertEquals(IndexingTrackState.COMMITTED, ledgerTrack.state)
                assertTrue(descriptor.finalizedAudioSpan.exactSampleCount24k > 0L)
                val artifacts = ledgerTrack.verifiedArtifacts.associateBy(VerifiedArtifact::kind)
                assertEquals(
                    setOf(
                        VerifiedArtifactKind.MERT_FEATURES,
                        VerifiedArtifactKind.CLAMP_VECTOR,
                        VerifiedArtifactKind.DATABASE_COMMIT,
                    ),
                    artifacts.keys,
                )
                val pcm = requireNotNull(control.pcmReceiptsByOrdinal[descriptor.ordinal]) {
                    "executor completed without verified PCM evidence for ${sourceSpec.caseId}"
                }
                val committedTrack = database.findTracksByPath(sourceSpec.source.path).single()
                val finalVector = requireNotNull(database.getEmbedding(committedTrack.id))
                val finalEmbeddingSha256 = sha256(EmbeddingDatabase.floatArrayToBlob(finalVector))
                val clampSha256 = artifacts.getValue(VerifiedArtifactKind.CLAMP_VECTOR).sha256
                val commitSha256 = artifacts.getValue(VerifiedArtifactKind.DATABASE_COMMIT).sha256
                assertEquals(clampSha256, commitSha256)
                assertEquals(clampSha256, finalEmbeddingSha256)
                FullBatchTrackEvidence(
                    ordinal = descriptor.ordinal,
                    caseId = sourceSpec.caseId,
                    sourceRelativePath = sourceSpec.sourceRelativePath,
                    sourceByteLength = sourceSpec.source.length(),
                    sourceSha256 = sourceSpec.expectedSha256,
                    powerampFileId = sourceSpec.powerampFileId,
                    workId = descriptor.workId,
                    stableTrackSpanId = descriptor.stableTrackSpanIdentity.stableTrackSpanId,
                    exactSampleCount24k = pcm.exactSampleCount24k,
                    sourceSampleCount = pcm.sourceSampleCount,
                    decoderName = pcm.decoderName,
                    pcmByteLength = pcm.pcmByteLength,
                    pcmSha256 = pcm.pcmSha256,
                    mertWindows = descriptor.expectedWork.mertWindows,
                    mertByteLength = artifacts.getValue(VerifiedArtifactKind.MERT_FEATURES).byteLength,
                    mertSha256 = artifacts.getValue(VerifiedArtifactKind.MERT_FEATURES).sha256,
                    clampSegments = descriptor.expectedWork.clampSegments,
                    clampByteLength = artifacts.getValue(VerifiedArtifactKind.CLAMP_VECTOR).byteLength,
                    clampSha256 = clampSha256,
                    databaseCommitEmbeddingSha256 = commitSha256,
                    finalEmbeddingSha256 = finalEmbeddingSha256,
                )
            }.sortedBy(FullBatchTrackEvidence::ordinal)
        } finally {
            database.close()
        }
        assertEquals(spec.sources.indices.toList(), trackEvidence.map(FullBatchTrackEvidence::ordinal))
        val graphFile = requireNotNull(generation.graphFile)
        assertEquals(generation.manifest.databaseSha256, V2FileSha256.digest(generation.databaseFile))
        assertEquals(generation.manifest.embeddingSha256, V2FileSha256.digest(generation.embeddingFile))
        assertEquals(generation.manifest.graph?.sha256, V2FileSha256.digest(graphFile))
        val semanticDatabaseSha256 = semanticDatabaseSha256(generation.databaseFile)
        val finalRuntime = runtimeSnapshot(target)

        return FullBatchEvidence(
            schemaVersion = 2,
            runId = spec.runId,
            resultToken = spec.resultToken,
            runLabel = spec.runLabel,
            profile = V2IndexingExecutionProfile.FULL.name,
            sourceCount = spec.sources.size,
            pcmPrefetchEnabled = spec.pcmPrefetchEnabled,
            jobId = complete.jobSpec.jobId,
            jobSpecId = complete.jobSpec.specId,
            embeddingSpecId = complete.jobSpec.embeddingSpec.specId,
            modelArtifactSha256 = complete.jobSpec.embeddingSpec.modelArtifactSha256,
            preprocessingSpecId = complete.jobSpec.embeddingSpec.preprocessingSpecId,
            schedule = V2IndexingExecutionPolicies.schedule(V2IndexingExecutionProfile.FULL),
            tracks = trackEvidence,
            databaseFileByteLength = generation.databaseFile.length(),
            databaseFileSha256 = generation.manifest.databaseSha256,
            databaseContentSha256 = generation.manifest.databaseContentSha256,
            databaseSemanticSha256 = semanticDatabaseSha256,
            orderedTrackSetSha256 = generation.manifest.orderedTrackSetSha256,
            pembByteLength = generation.embeddingFile.length(),
            pembSha256 = generation.manifest.embeddingSha256,
            graphByteLength = graphFile.length(),
            graphSha256 = requireNotNull(generation.manifest.graph).sha256,
            graphNodes = generation.manifest.graph.nodeCount,
            graphNeighborsPerNode = generation.manifest.graph.neighborsPerNode,
            generationId = generation.manifest.generationId,
            activationBindingId = generation.manifest.activationBindingId,
            manifestSha256 = generation.manifestSha256,
            activePointerBeforeSha256 = livePointerBefore,
            preflightElapsedMs = preflightElapsedMs,
            executorWallMs = executorWallMs,
            executorCpuMs = executorCpuMs,
            totalWallMs = elapsedMs(totalStartedNs),
            stageTimings = control.stageTimings(),
            runtimeBefore = initialRuntime,
            runtimeAfter = finalRuntime,
        )
    }

    private fun runComparison(
        target: Context,
        root: File,
        source: File,
        spec: ComparisonSpec,
        modelPolicy: V2FutureModelPolicy,
        livePointerBefore: String,
    ): ProfileComparisonEvidence {
        val totalStartedNs = SystemClock.elapsedRealtimeNanos()
        val initialRuntime = runtimeSnapshot(target)
        val isolatedContext = IsolatedFilesContext(target, root)
        publishTinyBase(root, modelPolicy)
        val provider = fixtureProvider(isolatedContext, source, spec.powerampFileId, spec.caseId)
        val realInspector = V2MediaExtractorAudioInspector()
        val endOfStreamInspector = V2AudioContainerInspector { physicalPath ->
            realInspector.inspect(physicalPath).copy(
                durationUsEstimate = 0L,
                durationEstimateSource = V2DurationEstimateSource.UNAVAILABLE,
            )
        }
        val selected = NewTrackDetector.UnindexedTrack(
            powerampFileId = spec.powerampFileId,
            artist = TrackNormalization.normalizeArtist(ARTIST),
            album = TrackNormalization.normalizeAlbum(ALBUM),
            title = TrackNormalization.normalizeTitle(spec.caseId),
            durationMs = 0,
            path = source.path,
        )
        val snapshot = provider.acquireBlocking()
        val preflightStartedNs = SystemClock.elapsedRealtimeNanos()
        val resolution = V2IndexingJobPreflightPlanner(
            context = isolatedContext,
            audioSpanResolver = V2AudioSpanResolver(endOfStreamInspector),
        ).resolveAndPersistBlocking(
            V2IndexingPreflightRequest(
                selectedTracks = listOf(V2ResolvedTrackSource(selected, source)),
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
                executionProfile = spec.profile,
                jobId = "profile-${spec.resultToken}",
            ),
        )
        val preflightElapsedMs = elapsedMs(preflightStartedNs)
        assertTrue("profile preflight did not materialize", resolution is V2IndexingPreflightResolution.Materialized)
        resolution as V2IndexingPreflightResolution.Materialized
        assertEquals(1, resolution.planned.size)
        assertTrue(resolution.rejected.isEmpty())

        val repository = V2IndexingJobRepository.createIsolated(root)
        repository.reconcileStartup()
        val planned = repository.require(resolution.jobId)
        assertEquals(spec.profile, planned.executionProfile)
        val control = ProfileEvidenceControl(
            profile = spec.profile,
            repository = repository,
            jobId = resolution.jobId,
            powerampFileId = spec.powerampFileId,
        )
        val token = repository.claimExecutor(resolution.jobId)
        repository.startAuthorized(token)
        val executorStartedNs = SystemClock.elapsedRealtimeNanos()
        val executorStartedCpuMs = Process.getElapsedCpuTime()
        val outcome = try {
            V2IndexingExecutor(
                context = isolatedContext,
                repository = repository,
                providerSnapshots = provider,
                modelResolver = V2IndexingModelResolver(target.filesDir),
            ).run(token, control)
        } finally {
            repository.releaseExecutor(token)
        }
        val executorCpuMs = Process.getElapsedCpuTime() - executorStartedCpuMs
        val executorWallMs = elapsedMs(executorStartedNs)
        assertEquals(V2IndexingExecutorOutcome.COMPLETE, outcome)
        val complete = repository.require(resolution.jobId)
        assertEquals(IndexingJobState.COMPLETE, complete.state)
        assertEquals(spec.profile, complete.executionProfile)
        val track = complete.tracks.single()
        assertEquals(IndexingTrackState.COMMITTED, track.state)
        val finalDescriptor = complete.jobSpec.tracks.single()
        assertEquals(track.workId, finalDescriptor.workId)
        assertTrue(finalDescriptor.finalizedAudioSpan.exactSampleCount24k > 0L)
        val artifacts = track.verifiedArtifacts.associateBy(VerifiedArtifact::kind)
        assertEquals(
            setOf(
                VerifiedArtifactKind.MERT_FEATURES,
                VerifiedArtifactKind.CLAMP_VECTOR,
                VerifiedArtifactKind.DATABASE_COMMIT,
            ),
            artifacts.keys,
        )
        val pcm = requireNotNull(control.pcmReceipt) {
            "executor completed without verified PCM evidence"
        }
        val generation = V2IndexGenerationReader.requireActive(root)
        val database = EmbeddingDatabase.open(generation.databaseFile)
        val finalVector = try {
            val committedTrack = database.findTracksByPath(source.path).single()
            requireNotNull(database.getEmbedding(committedTrack.id))
        } finally {
            database.close()
        }
        val finalEmbeddingSha256 = sha256(EmbeddingDatabase.floatArrayToBlob(finalVector))
        val clampSha256 = artifacts.getValue(VerifiedArtifactKind.CLAMP_VECTOR).sha256
        val commitSha256 = artifacts.getValue(VerifiedArtifactKind.DATABASE_COMMIT).sha256
        assertEquals(clampSha256, commitSha256)
        assertEquals(clampSha256, finalEmbeddingSha256)
        val graphFile = requireNotNull(generation.graphFile)
        assertEquals(generation.manifest.databaseSha256, V2FileSha256.digest(generation.databaseFile))
        assertEquals(generation.manifest.embeddingSha256, V2FileSha256.digest(generation.embeddingFile))
        assertEquals(generation.manifest.graph?.sha256, V2FileSha256.digest(graphFile))
        val semanticDatabaseSha256 = semanticDatabaseSha256(generation.databaseFile)
        val finalRuntime = runtimeSnapshot(target)

        return ProfileComparisonEvidence(
            schemaVersion = 1,
            runId = spec.runId,
            resultToken = spec.resultToken,
            runLabel = spec.runLabel,
            caseId = spec.caseId,
            profile = spec.profile.name,
            sourceRelativePath = spec.sourceRelativePath,
            sourceByteLength = source.length(),
            sourceSha256 = spec.expectedSourceSha256,
            powerampFileId = spec.powerampFileId,
            jobId = complete.jobSpec.jobId,
            jobSpecId = complete.jobSpec.specId,
            workId = finalDescriptor.workId,
            stableTrackSpanId = finalDescriptor.stableTrackSpanIdentity.stableTrackSpanId,
            embeddingSpecId = complete.jobSpec.embeddingSpec.specId,
            modelArtifactSha256 = complete.jobSpec.embeddingSpec.modelArtifactSha256,
            preprocessingSpecId = complete.jobSpec.embeddingSpec.preprocessingSpecId,
            schedule = V2IndexingExecutionPolicies.schedule(spec.profile),
            exactSampleCount24k = pcm.exactSampleCount24k,
            sourceSampleCount = pcm.sourceSampleCount,
            decoderName = pcm.decoderName,
            pcmByteLength = pcm.pcmByteLength,
            pcmSha256 = pcm.pcmSha256,
            mertWindows = finalDescriptor.expectedWork.mertWindows,
            mertByteLength = artifacts.getValue(VerifiedArtifactKind.MERT_FEATURES).byteLength,
            mertSha256 = artifacts.getValue(VerifiedArtifactKind.MERT_FEATURES).sha256,
            clampSegments = finalDescriptor.expectedWork.clampSegments,
            clampByteLength = artifacts.getValue(VerifiedArtifactKind.CLAMP_VECTOR).byteLength,
            clampSha256 = clampSha256,
            databaseCommitEmbeddingSha256 = commitSha256,
            finalEmbeddingSha256 = finalEmbeddingSha256,
            databaseFileByteLength = generation.databaseFile.length(),
            databaseFileSha256 = generation.manifest.databaseSha256,
            databaseContentSha256 = generation.manifest.databaseContentSha256,
            databaseSemanticSha256 = semanticDatabaseSha256,
            orderedTrackSetSha256 = generation.manifest.orderedTrackSetSha256,
            pembByteLength = generation.embeddingFile.length(),
            pembSha256 = generation.manifest.embeddingSha256,
            graphByteLength = graphFile.length(),
            graphSha256 = requireNotNull(generation.manifest.graph).sha256,
            graphNodes = generation.manifest.graph.nodeCount,
            graphNeighborsPerNode = generation.manifest.graph.neighborsPerNode,
            generationId = generation.manifest.generationId,
            activationBindingId = generation.manifest.activationBindingId,
            manifestSha256 = generation.manifestSha256,
            activePointerBeforeSha256 = livePointerBefore,
            preflightElapsedMs = preflightElapsedMs,
            executorWallMs = executorWallMs,
            executorCpuMs = executorCpuMs,
            totalWallMs = elapsedMs(totalStartedNs),
            stageTimings = control.stageTimings(),
            runtimeBefore = initialRuntime,
            runtimeAfter = finalRuntime,
        )
    }

    private class ProfileEvidenceControl(
        private val profile: V2IndexingExecutionProfile,
        private val repository: V2IndexingJobRepository,
        private val jobId: String,
        private val powerampFileId: Long,
    ) : V2IndexingExecutorControl {
        private data class MutableTiming(
            var firstElapsedMs: Long,
            var lastElapsedMs: Long,
            var events: Int,
            var lastCompletedUnits: Long?,
            var totalUnits: Long?,
        )

        private val startedNs = SystemClock.elapsedRealtimeNanos()
        private val timings = linkedMapOf<V2MeasuredWorkStage, MutableTiming>()
        var pcmReceipt: V2VerifiedPcmCacheReceipt? = null
            private set

        override fun throwIfStopped() = Unit

        override fun executionProfile(): V2IndexingExecutionProfile = profile

        override fun onProgress(event: V2IndexingExecutorEvent) {
            val now = elapsedMs(startedNs)
            val timing = timings.getOrPut(event.stage) {
                MutableTiming(now, now, 0, null, null)
            }
            timing.lastElapsedMs = now
            timing.events++
            timing.lastCompletedUnits = event.completedUnits
            timing.totalUnits = event.totalUnits
            if (pcmReceipt == null && event.stage == V2MeasuredWorkStage.MERT_WINDOWS &&
                event.completedUnits == 0L
            ) {
                val receipt = File(
                    repository.artifactDirectory(jobId),
                    "pcm-cache-v1/$powerampFileId.receipt.json",
                )
                assertTrue("verified PCM receipt is missing at MERT start", receipt.isFile)
                pcmReceipt = receipt.bufferedReader(StandardCharsets.UTF_8).use { reader ->
                    GSON.fromJson(reader, V2VerifiedPcmCacheReceipt::class.java)
                }
                assertNotNull(pcmReceipt)
            }
        }

        fun stageTimings(): List<StageTimingEvidence> = timings.map { (stage, value) ->
            StageTimingEvidence(
                stage = stage.name,
                firstElapsedMs = value.firstElapsedMs,
                lastElapsedMs = value.lastElapsedMs,
                observedSpanMs = value.lastElapsedMs - value.firstElapsedMs,
                eventCount = value.events,
                lastCompletedUnits = value.lastCompletedUnits,
                totalUnits = value.totalUnits,
            )
        }
    }

    private class FullBatchEvidenceControl(
        private val repository: V2IndexingJobRepository,
        private val jobId: String,
        private val sourceByOrdinal: Map<Int, FullBatchSourceSpec>,
    ) : V2IndexingExecutorControl {
        private data class TimingKey(
            val trackOrdinal: Int?,
            val stage: V2MeasuredWorkStage,
        )

        private data class MutableTiming(
            val firstElapsedMs: Long,
            var lastElapsedMs: Long,
            var events: Int,
            val firstWorkId: String?,
            var lastWorkId: String?,
            val firstCompletedUnits: Long?,
            var lastCompletedUnits: Long?,
            val firstTotalUnits: Long?,
            var lastTotalUnits: Long?,
            val firstDetail: String,
            var lastDetail: String,
        )

        private val startedNs = SystemClock.elapsedRealtimeNanos()
        private val timings = linkedMapOf<TimingKey, MutableTiming>()
        val pcmReceiptsByOrdinal = mutableMapOf<Int, V2VerifiedPcmCacheReceipt>()

        override fun throwIfStopped() = Unit

        override fun executionProfile(): V2IndexingExecutionProfile = V2IndexingExecutionProfile.FULL

        override fun onProgress(event: V2IndexingExecutorEvent) {
            val now = elapsedMs(startedNs)
            val key = TimingKey(event.trackOrdinal, event.stage)
            val timing = timings.getOrPut(key) {
                MutableTiming(
                    firstElapsedMs = now,
                    lastElapsedMs = now,
                    events = 0,
                    firstWorkId = event.workId,
                    lastWorkId = event.workId,
                    firstCompletedUnits = event.completedUnits,
                    lastCompletedUnits = event.completedUnits,
                    firstTotalUnits = event.totalUnits,
                    lastTotalUnits = event.totalUnits,
                    firstDetail = event.detail,
                    lastDetail = event.detail,
                )
            }
            timing.lastElapsedMs = now
            timing.events++
            timing.lastWorkId = event.workId
            timing.lastCompletedUnits = event.completedUnits
            timing.lastTotalUnits = event.totalUnits
            timing.lastDetail = event.detail

            val ordinal = event.trackOrdinal
            if (event.stage == V2MeasuredWorkStage.MERT_WINDOWS && ordinal != null &&
                ordinal !in pcmReceiptsByOrdinal
            ) {
                val source = requireNotNull(sourceByOrdinal[ordinal]) {
                    "executor emitted an unknown batch track ordinal $ordinal"
                }
                val receipt = File(
                    repository.artifactDirectory(jobId),
                    "pcm-cache-v1/${source.powerampFileId}.receipt.json",
                )
                assertTrue("verified PCM receipt is missing at MERT start", receipt.isFile)
                pcmReceiptsByOrdinal[ordinal] = receipt.bufferedReader(StandardCharsets.UTF_8).use {
                    GSON.fromJson(it, V2VerifiedPcmCacheReceipt::class.java)
                }
            }
        }

        fun stageTimings(): List<FullBatchStageTimingEvidence> = timings.map { (key, value) ->
            FullBatchStageTimingEvidence(
                trackOrdinal = key.trackOrdinal,
                caseId = key.trackOrdinal?.let { sourceByOrdinal.getValue(it).caseId },
                stage = key.stage.name,
                firstElapsedMs = value.firstElapsedMs,
                lastElapsedMs = value.lastElapsedMs,
                observedSpanMs = value.lastElapsedMs - value.firstElapsedMs,
                eventCount = value.events,
                firstWorkId = value.firstWorkId,
                lastWorkId = value.lastWorkId,
                firstCompletedUnits = value.firstCompletedUnits,
                lastCompletedUnits = value.lastCompletedUnits,
                firstTotalUnits = value.firstTotalUnits,
                lastTotalUnits = value.lastTotalUnits,
                firstDetail = value.firstDetail,
                lastDetail = value.lastDetail,
            )
        }
    }

    private fun fixtureProvider(
        context: Context,
        source: File,
        powerampFileId: Long,
        caseId: String,
    ): V2PowerampProviderSnapshotAcquirer = V2PowerampProviderSnapshotAcquirer(
        context = context,
        providerQuery = V2PowerampProviderQuery { _, _, projection ->
            val columns = (projection.toList() + listOf(
                "_id",
                "duration",
                "name",
                "offset_ms",
            )).distinct()
            MatrixCursor(columns.toTypedArray()).apply {
                addRow(
                    columns.map<String, Any?> { column ->
                        when (column) {
                            "folder_files._id", "_id" -> powerampFileId
                            "artist" -> ARTIST
                            "album" -> ALBUM
                            "title_tag" -> caseId
                            "folder_files.duration", "duration" -> 0L
                            "path" -> source.parentFile!!.path
                            "folder_files.name", "name" -> source.name
                            "folder_files.offset_ms", "offset_ms", "cue_folder_id" -> null
                            else -> error("unexpected profile provider column $column")
                        }
                    }.toTypedArray(),
                )
            }
        },
    )

    private fun fixtureProvider(
        context: Context,
        sources: List<FullBatchSourceSpec>,
    ): V2PowerampProviderSnapshotAcquirer = V2PowerampProviderSnapshotAcquirer(
        context = context,
        providerQuery = V2PowerampProviderQuery { _, _, projection ->
            val columns = (projection.toList() + listOf(
                "_id",
                "duration",
                "name",
                "offset_ms",
            )).distinct()
            MatrixCursor(columns.toTypedArray()).apply {
                sources.forEach { sourceSpec ->
                    addRow(
                        columns.map<String, Any?> { column ->
                            when (column) {
                                "folder_files._id", "_id" -> sourceSpec.powerampFileId
                                "artist" -> ARTIST
                                "album" -> BATCH_ALBUM
                                "title_tag" -> sourceSpec.caseId
                                "folder_files.duration", "duration" -> 0L
                                "path" -> sourceSpec.source.parentFile!!.path
                                "folder_files.name", "name" -> sourceSpec.source.name
                                "folder_files.offset_ms", "offset_ms", "cue_folder_id" -> null
                                else -> error("unexpected batch provider column $column")
                            }
                        }.toTypedArray(),
                    )
                }
            }
        },
    )

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

    /** Canonical DB result digest excluding only deliberately run-specific provenance times. */
    private fun semanticDatabaseSha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        SQLiteDatabase.openDatabase(file.path, null, SQLiteDatabase.OPEN_READONLY).use { database ->
            digest.rowSet(
                database,
                """
                SELECT id, metadata_key, filename_key, artist, album, title, duration_ms,
                       file_path, source
                FROM tracks ORDER BY id
                """.trimIndent(),
            )
            digest.rowSet(
                database,
                "SELECT track_id, embedding FROM embeddings_clamp3 ORDER BY track_id",
            )
            digest.rowSet(
                database,
                """
                SELECT receipt_schema_version, work_id, stable_track_span_id,
                       stable_identity_spec_id, stable_identity_strength, embedding_spec_id,
                       provider_physical_path, provider_offset_ms, provider_duration_ms, track_id,
                       metadata_sha256, embedding_byte_length, embedding_sha256
                FROM ${V2EmbeddingCommitRepository.RECEIPT_TABLE}
                ORDER BY work_id, embedding_spec_id
                """.trimIndent(),
            )
            digest.rowSet(
                database,
                """
                SELECT singleton, receipt_schema_version, is_valid, activation_binding_id,
                       job_spec_id, receipt_embedding_spec_id, text_retrieval_spec_id,
                       embedding_coverage_sha256, compatibility_base_content_sha256,
                       database_content_sha256, ordered_track_set_sha256,
                       stable_uid_mapping_sha256, embedding_sha256, graph_sha256
                FROM v2_index_generation_guard_v2 ORDER BY singleton
                """.trimIndent(),
            )
        }
        return digest.digest().toHex()
    }

    private fun MessageDigest.rowSet(database: SQLiteDatabase, sql: String) {
        updateLengthPrefixed(sql)
        database.rawQuery(sql, null).use { cursor ->
            updateInt(cursor.columnCount)
            var rowCount = 0
            while (cursor.moveToNext()) {
                update(1)
                repeat(cursor.columnCount) { column ->
                    when (cursor.getType(column)) {
                        android.database.Cursor.FIELD_TYPE_NULL -> update(0)
                        android.database.Cursor.FIELD_TYPE_INTEGER -> {
                            update(1)
                            updateLong(cursor.getLong(column))
                        }
                        android.database.Cursor.FIELD_TYPE_FLOAT -> {
                            update(2)
                            updateLong(java.lang.Double.doubleToRawLongBits(cursor.getDouble(column)))
                        }
                        android.database.Cursor.FIELD_TYPE_STRING -> {
                            update(3)
                            updateLengthPrefixed(cursor.getString(column))
                        }
                        android.database.Cursor.FIELD_TYPE_BLOB -> {
                            update(4)
                            updateLengthPrefixed(cursor.getBlob(column))
                        }
                        else -> error("unsupported SQLite field type")
                    }
                }
                rowCount++
            }
            update(0)
            updateInt(rowCount)
        }
    }

    private fun runtimeSnapshot(context: Context): RuntimeSnapshot {
        val memory = Debug.MemoryInfo().also(Debug::getMemoryInfo)
        val runtime = Runtime.getRuntime()
        val battery = context.getSystemService(BatteryManager::class.java)
        val sticky = context.registerReceiver(null, android.content.IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val power = context.getSystemService(PowerManager::class.java)
        return RuntimeSnapshot(
            elapsedRealtimeMs = SystemClock.elapsedRealtime(),
            processCpuMs = Process.getElapsedCpuTime(),
            totalPssKb = memory.totalPss,
            nativeHeapAllocatedBytes = Debug.getNativeHeapAllocatedSize(),
            javaHeapUsedBytes = runtime.totalMemory() - runtime.freeMemory(),
            javaHeapMaxBytes = runtime.maxMemory(),
            thermalStatus = if (android.os.Build.VERSION.SDK_INT >= 29) power.currentThermalStatus else null,
            batteryCapacityPercent = battery.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY),
            batteryCurrentNowUa = battery.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW),
            batteryChargeCounterUah = battery.getIntProperty(BatteryManager.BATTERY_PROPERTY_CHARGE_COUNTER),
            batteryTemperatureTenthsC = sticky?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, -1),
            batteryStatus = sticky?.getIntExtra(BatteryManager.EXTRA_STATUS, -1),
            batteryPlugged = sticky?.getIntExtra(BatteryManager.EXTRA_PLUGGED, 0),
        )
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

    private fun activePointerBefore(state: Map<String, Pair<Long, String>?>): String =
        sha256(
            state.entries.joinToString("\n") { (name, value) ->
                "$name=${value?.let { "${it.first}:${it.second}" } ?: "absent"}"
            }.toByteArray(StandardCharsets.UTF_8),
        )

    private class IsolatedFilesContext(base: Context, private val root: File) :
        ContextWrapper(base) {
        override fun getApplicationContext(): Context = this
        override fun getFilesDir(): File = root
    }

    private data class ComparisonSpec(
        val runId: String,
        val runLabel: String,
        val caseId: String,
        val profile: V2IndexingExecutionProfile,
        val sourceRelativePath: String,
        val expectedSourceSha256: String,
        val powerampFileId: Long,
    ) {
        val resultToken: String = "$runId-$caseId-$runLabel"

        companion object {
            fun fromArguments(
                runId: String,
                arguments: android.os.Bundle,
                filesDir: File,
            ): ComparisonSpec {
                require(SAFE_ID.matches(runId)) { "unsafe comparison run ID" }
                val runLabel = arguments.required(ARG_RUN_LABEL)
                val caseId = arguments.required(ARG_CASE_ID)
                require(SAFE_ID.matches(runLabel) && SAFE_ID.matches(caseId)) {
                    "unsafe comparison label or case ID"
                }
                require("$runId-$caseId-$runLabel".length <= MAX_RESULT_TOKEN_LENGTH) {
                    "comparison result token is too long"
                }
                val profile = V2IndexingExecutionProfile.valueOf(arguments.required(ARG_PROFILE))
                require(profile == V2IndexingExecutionProfile.FULL ||
                    profile == V2IndexingExecutionProfile.BACKGROUND
                ) { "comparison supports FULL and BACKGROUND only" }
                val sourceRelativePath = arguments.required(ARG_SOURCE_RELATIVE_PATH)
                require(!sourceRelativePath.startsWith('/') && ".." !in sourceRelativePath.split('/')) {
                    "unsafe source-relative path"
                }
                val allowed = File(filesDir, FIXTURE_ROOT).canonicalFile
                val source = File(filesDir, sourceRelativePath).canonicalFile
                require(source.path.startsWith(allowed.path + File.separator)) {
                    "source is outside the opt-in fixture root"
                }
                val sourceSha256 = arguments.required(ARG_SOURCE_SHA256).lowercase()
                require(SHA256.matches(sourceSha256)) { "invalid source SHA-256" }
                val powerampFileId = arguments.required(ARG_POWERAMP_FILE_ID).toLong()
                require(powerampFileId in MIN_POWERAMP_ID..MAX_POWERAMP_ID) {
                    "profile comparison Poweramp ID is outside its reserved range"
                }
                return ComparisonSpec(
                    runId = runId,
                    runLabel = runLabel,
                    caseId = caseId,
                    profile = profile,
                    sourceRelativePath = source.relativeTo(filesDir.canonicalFile).path,
                    expectedSourceSha256 = sourceSha256,
                    powerampFileId = powerampFileId,
                )
            }
        }
    }

    private data class FullBatchSourceSpec(
        val caseId: String,
        val sourceRelativePath: String,
        val source: File,
        val expectedSha256: String,
        val powerampFileId: Long,
    )

    private data class FullBatchSpec(
        val runId: String,
        val runLabel: String,
        val pcmPrefetchEnabled: Boolean,
        val sources: List<FullBatchSourceSpec>,
    ) {
        val resultToken: String = "$runId-$runLabel"

        companion object {
            fun fromArguments(
                runId: String,
                arguments: android.os.Bundle,
                filesDir: File,
            ): FullBatchSpec {
                require(SAFE_ID.matches(runId)) { "unsafe batch run ID" }
                val runLabel = arguments.required(ARG_RUN_LABEL)
                require(SAFE_ID.matches(runLabel)) { "unsafe batch run label" }
                require("$runId-$runLabel".length <= MAX_RESULT_TOKEN_LENGTH) {
                    "batch result token is too long"
                }
                val pcmPrefetchEnabled = arguments.getString(ARG_PCM_PREFETCH_ENABLED)
                    ?.toBooleanStrictOrNull()
                    ?: true
                val count = arguments.required(ARG_BATCH_CASE_COUNT).toInt()
                require(count in MIN_BATCH_CASE_COUNT..MAX_BATCH_CASE_COUNT) {
                    "batch case count must be in $MIN_BATCH_CASE_COUNT..$MAX_BATCH_CASE_COUNT"
                }
                val allowed = File(filesDir, FIXTURE_ROOT).canonicalFile
                val sources = (0 until count).map { index ->
                    val prefix = "$ARG_BATCH_CASE_PREFIX$index"
                    val caseId = arguments.required("${prefix}Id")
                    require(SAFE_ID.matches(caseId)) { "unsafe batch case ID" }
                    val relative = arguments.required("${prefix}SourceRelativePath")
                    require(!relative.startsWith('/') && ".." !in relative.split('/')) {
                        "unsafe batch source-relative path"
                    }
                    val source = File(filesDir, relative).canonicalFile
                    require(source.path.startsWith(allowed.path + File.separator)) {
                        "batch source is outside the opt-in fixture root"
                    }
                    val sourceSha256 = arguments.required("${prefix}SourceSha256").lowercase()
                    require(SHA256.matches(sourceSha256)) { "invalid batch source SHA-256" }
                    val powerampFileId = arguments.required("${prefix}PowerampFileId").toLong()
                    require(powerampFileId in MIN_POWERAMP_ID..MAX_POWERAMP_ID) {
                        "batch Poweramp ID is outside its reserved range"
                    }
                    FullBatchSourceSpec(
                        caseId = caseId,
                        sourceRelativePath = source.relativeTo(filesDir.canonicalFile).path,
                        source = source,
                        expectedSha256 = sourceSha256,
                        powerampFileId = powerampFileId,
                    )
                }
                require(sources.map(FullBatchSourceSpec::caseId).toSet().size == sources.size) {
                    "batch case IDs must be unique"
                }
                require(sources.map(FullBatchSourceSpec::source).toSet().size == sources.size) {
                    "batch source paths must be unique"
                }
                require(sources.map(FullBatchSourceSpec::expectedSha256).toSet().size == sources.size) {
                    "batch source contents must be distinct"
                }
                require(sources.map(FullBatchSourceSpec::powerampFileId).toSet().size == sources.size) {
                    "batch Poweramp IDs must be unique"
                }
                return FullBatchSpec(runId, runLabel, pcmPrefetchEnabled, sources)
            }
        }
    }

    private data class ProfileComparisonEvidence(
        val schemaVersion: Int,
        val runId: String,
        val resultToken: String,
        val runLabel: String,
        val caseId: String,
        val profile: String,
        val sourceRelativePath: String,
        val sourceByteLength: Long,
        val sourceSha256: String,
        val powerampFileId: Long,
        val jobId: String,
        val jobSpecId: String,
        val workId: String,
        val stableTrackSpanId: String,
        val embeddingSpecId: String,
        val modelArtifactSha256: Map<String, String>,
        val preprocessingSpecId: String,
        val schedule: V2IndexingExecutionSchedule,
        val exactSampleCount24k: Long,
        val sourceSampleCount: Long,
        val decoderName: String,
        val pcmByteLength: Long,
        val pcmSha256: String,
        val mertWindows: Int,
        val mertByteLength: Long,
        val mertSha256: String,
        val clampSegments: Int,
        val clampByteLength: Long,
        val clampSha256: String,
        val databaseCommitEmbeddingSha256: String,
        val finalEmbeddingSha256: String,
        val databaseFileByteLength: Long,
        val databaseFileSha256: String,
        val databaseContentSha256: String,
        val databaseSemanticSha256: String,
        val orderedTrackSetSha256: String,
        val pembByteLength: Long,
        val pembSha256: String,
        val graphByteLength: Long,
        val graphSha256: String,
        val graphNodes: Int,
        val graphNeighborsPerNode: Int,
        val generationId: String,
        val activationBindingId: String,
        val manifestSha256: String,
        val activePointerBeforeSha256: String,
        val preflightElapsedMs: Long,
        val executorWallMs: Long,
        val executorCpuMs: Long,
        val totalWallMs: Long,
        val stageTimings: List<StageTimingEvidence>,
        val runtimeBefore: RuntimeSnapshot,
        val runtimeAfter: RuntimeSnapshot,
    )

    private data class FullBatchEvidence(
        val schemaVersion: Int,
        val runId: String,
        val resultToken: String,
        val runLabel: String,
        val profile: String,
        val sourceCount: Int,
        val pcmPrefetchEnabled: Boolean,
        val jobId: String,
        val jobSpecId: String,
        val embeddingSpecId: String,
        val modelArtifactSha256: Map<String, String>,
        val preprocessingSpecId: String,
        val schedule: V2IndexingExecutionSchedule,
        val tracks: List<FullBatchTrackEvidence>,
        val databaseFileByteLength: Long,
        val databaseFileSha256: String,
        val databaseContentSha256: String,
        val databaseSemanticSha256: String,
        val orderedTrackSetSha256: String,
        val pembByteLength: Long,
        val pembSha256: String,
        val graphByteLength: Long,
        val graphSha256: String,
        val graphNodes: Int,
        val graphNeighborsPerNode: Int,
        val generationId: String,
        val activationBindingId: String,
        val manifestSha256: String,
        val activePointerBeforeSha256: String,
        val preflightElapsedMs: Long,
        val executorWallMs: Long,
        val executorCpuMs: Long,
        val totalWallMs: Long,
        val stageTimings: List<FullBatchStageTimingEvidence>,
        val runtimeBefore: RuntimeSnapshot,
        val runtimeAfter: RuntimeSnapshot,
    )

    private data class FullBatchTrackEvidence(
        val ordinal: Int,
        val caseId: String,
        val sourceRelativePath: String,
        val sourceByteLength: Long,
        val sourceSha256: String,
        val powerampFileId: Long,
        val workId: String,
        val stableTrackSpanId: String,
        val exactSampleCount24k: Long,
        val sourceSampleCount: Long,
        val decoderName: String,
        val pcmByteLength: Long,
        val pcmSha256: String,
        val mertWindows: Int,
        val mertByteLength: Long,
        val mertSha256: String,
        val clampSegments: Int,
        val clampByteLength: Long,
        val clampSha256: String,
        val databaseCommitEmbeddingSha256: String,
        val finalEmbeddingSha256: String,
    )

    private data class FullBatchStageTimingEvidence(
        val trackOrdinal: Int?,
        val caseId: String?,
        val stage: String,
        val firstElapsedMs: Long,
        val lastElapsedMs: Long,
        val observedSpanMs: Long,
        val eventCount: Int,
        val firstWorkId: String?,
        val lastWorkId: String?,
        val firstCompletedUnits: Long?,
        val lastCompletedUnits: Long?,
        val firstTotalUnits: Long?,
        val lastTotalUnits: Long?,
        val firstDetail: String,
        val lastDetail: String,
    )

    private data class StageTimingEvidence(
        val stage: String,
        val firstElapsedMs: Long,
        val lastElapsedMs: Long,
        val observedSpanMs: Long,
        val eventCount: Int,
        val lastCompletedUnits: Long?,
        val totalUnits: Long?,
    )

    private data class RuntimeSnapshot(
        val elapsedRealtimeMs: Long,
        val processCpuMs: Long,
        val totalPssKb: Int,
        val nativeHeapAllocatedBytes: Long,
        val javaHeapUsedBytes: Long,
        val javaHeapMaxBytes: Long,
        val thermalStatus: Int?,
        val batteryCapacityPercent: Int,
        val batteryCurrentNowUa: Int,
        val batteryChargeCounterUah: Int,
        val batteryTemperatureTenthsC: Int?,
        val batteryStatus: Int?,
        val batteryPlugged: Int?,
    )

    private companion object {
        val GSON: Gson = GsonBuilder().disableHtmlEscaping().setPrettyPrinting().create()
        val SAFE_ID = Regex("^[a-z0-9][a-z0-9_-]{0,47}$")
        val SHA256 = Regex("^[0-9a-f]{64}$")
        const val FIXTURE_ROOT = "device_acceptance/indexing_profile"
        const val ARG_RUN_ID = "v2ProfileRunId"
        const val ARG_RUN_LABEL = "v2ProfileRunLabel"
        const val ARG_CASE_ID = "v2ProfileCaseId"
        const val ARG_PROFILE = "v2Profile"
        const val ARG_SOURCE_RELATIVE_PATH = "v2ProfileSourceRelativePath"
        const val ARG_SOURCE_SHA256 = "v2ProfileSourceSha256"
        const val ARG_POWERAMP_FILE_ID = "v2ProfilePowerampFileId"
        const val ARG_BATCH_CASE_COUNT = "v2BatchCaseCount"
        const val ARG_BATCH_CASE_PREFIX = "v2BatchCase"
        const val ARG_PCM_PREFETCH_ENABLED = "v2PcmPrefetchEnabled"
        const val MIN_POWERAMP_ID = 8_800_000_000L
        const val MAX_POWERAMP_ID = 8_899_999_999L
        const val MIN_BATCH_CASE_COUNT = 3
        const val MAX_BATCH_CASE_COUNT = 16
        const val MAX_RESULT_TOKEN_LENGTH = 96
        const val BASE_TRACK_COUNT = 6
        const val ARTIST = "Profile Acceptance Artist"
        const val ALBUM = "Full vs Background"
        const val BATCH_ALBUM = "One Executor Full Batch"
        val REQUIRED_MODEL_FILES = listOf(
            "mert.tflite",
            "clamp3_audio.tflite",
            "clamp3_text.tflite",
            "sentencepiece.bpe.model",
        )

        fun android.os.Bundle.required(key: String): String =
            requireNotNull(getString(key)).takeIf(String::isNotBlank)
                ?: error("missing instrumentation argument $key")

        fun elapsedMs(startedNs: Long): Long =
            (SystemClock.elapsedRealtimeNanos() - startedNs) / 1_000_000L

        fun sha256(bytes: ByteArray): String =
            MessageDigest.getInstance("SHA-256").digest(bytes).toHex()

        fun ByteArray.toHex(): String =
            joinToString("") { byte -> "%02x".format(byte.toInt() and 0xff) }

        fun MessageDigest.updateInt(value: Int) {
            update(ByteBuffer.allocate(Int.SIZE_BYTES).order(ByteOrder.BIG_ENDIAN).putInt(value).array())
        }

        fun MessageDigest.updateLong(value: Long) {
            update(ByteBuffer.allocate(Long.SIZE_BYTES).order(ByteOrder.BIG_ENDIAN).putLong(value).array())
        }

        fun MessageDigest.updateLengthPrefixed(value: String) =
            updateLengthPrefixed(value.toByteArray(StandardCharsets.UTF_8))

        fun MessageDigest.updateLengthPrefixed(value: ByteArray) {
            updateInt(value.size)
            update(value)
        }

        fun writeAtomic(file: File, value: String) {
            val atomic = AtomicFile(file)
            val stream = atomic.startWrite()
            try {
                val writer = OutputStreamWriter(stream, StandardCharsets.UTF_8)
                writer.write(value)
                writer.write("\n")
                writer.flush()
                atomic.finishWrite(stream)
            } catch (error: Throwable) {
                atomic.failWrite(stream)
                throw error
            }
        }
    }
}
