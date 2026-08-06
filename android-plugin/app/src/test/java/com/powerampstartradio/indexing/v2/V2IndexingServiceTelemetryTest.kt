package com.powerampstartradio.indexing.v2

import com.powerampstartradio.indexing.V2GraphUpdateStrategy
import com.powerampstartradio.indexing.V2GraphWorkPlan
import java.io.File
import java.nio.file.Files
import java.util.concurrent.atomic.AtomicInteger
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class V2IndexingServiceTelemetryTest {
    @Test
    fun `production start and end events converge every repeated stage with matching keys`() {
        val estimator = V2StageAwareWorkEstimator(minimumSamples = 2)
        val recorder = V2IndexingEventRateRecorder(estimator)

        listOf("track-flac-a", "track-flac-b").forEachIndexed { index, workId ->
            val base = index * 1_000L
            recorder.onEvent(
                event(
                    workId,
                    V2MeasuredWorkStage.PCM_24K_SAMPLES,
                    0L,
                    24_000L,
                    pcmMeasurement(
                        powerampFileId = 100L + index,
                        point = V2PcmRateMeasurementPoint.MATERIALIZATION_STARTED,
                    ),
                ),
                PROFILE,
                "Audio/FLAC",
                base,
            )
            recorder.onEvent(
                event(
                    workId,
                    V2MeasuredWorkStage.PCM_24K_SAMPLES,
                    24_000L,
                    24_000L,
                    pcmMeasurement(
                        powerampFileId = 100L + index,
                        point = V2PcmRateMeasurementPoint.MATERIALIZATION_COMPLETED_EXACT,
                        exactSampleCount24k = 24_000L,
                    ),
                ),
                PROFILE,
                "audio/flac",
                base + 100L,
            )
            recorder.onEvent(
                event(workId, V2MeasuredWorkStage.MERT_WINDOWS, 0L, 1L),
                PROFILE,
                "audio/flac",
                base + 110L,
            )
            recorder.onEvent(
                event(workId, V2MeasuredWorkStage.MERT_WINDOWS, 1L, 1L),
                PROFILE,
                "audio/flac",
                base + 210L,
            )
            recorder.onEvent(
                event(workId, V2MeasuredWorkStage.CLAMP_SEGMENTS, 0L, 1L),
                PROFILE,
                "audio/flac",
                base + 220L,
            )
            recorder.onEvent(
                event(workId, V2MeasuredWorkStage.CLAMP_SEGMENTS, 1L, 1L),
                PROFILE,
                "audio/flac",
                base + 230L,
            )
            recorder.onEvent(
                event(workId, V2MeasuredWorkStage.DATABASE_COMMITS, 0L, 1L),
                PROFILE,
                "audio/flac",
                base + 240L,
            )
            recorder.onEvent(
                event(workId, V2MeasuredWorkStage.DATABASE_COMMITS, 1L, 1L),
                PROFILE,
                "audio/flac",
                base + 245L,
            )
        }

        val estimate = estimator.estimate(
            remaining = listOf(
                V2RemainingStageWork(
                    V2MeasuredWorkStage.PCM_24K_SAMPLES,
                    48_000L,
                    "audio/flac",
                ),
                V2RemainingStageWork(V2MeasuredWorkStage.MERT_WINDOWS, 2L),
                V2RemainingStageWork(V2MeasuredWorkStage.CLAMP_SEGMENTS, 2L),
                V2RemainingStageWork(V2MeasuredWorkStage.DATABASE_COMMITS, 2L),
            ),
            profile = PROFILE,
        )

        assertEquals(430L, estimate.remainingMs)
        assertTrue(estimate.calibratingStages.isEmpty())
        val samples = estimator.snapshot().samples
        assertEquals(
            setOf("audio/flac"),
            samples.filter { it.stage == V2MeasuredWorkStage.PCM_24K_SAMPLES }
                .mapNotNull { it.codecClass }
                .toSet(),
        )
        assertTrue(
            samples.filter { it.stage != V2MeasuredWorkStage.PCM_24K_SAMPLES }
                .all { it.codecClass == null },
        )
        assertEquals(
            2,
            samples.count { it.stage == V2MeasuredWorkStage.DATABASE_COMMITS },
        )
    }

    @Test
    fun `terminal event without a real start never learns cached or aliased work`() {
        val estimator = V2StageAwareWorkEstimator(minimumSamples = 1)
        val recorder = V2IndexingEventRateRecorder(estimator)

        val recorded = recorder.onEvent(
            event("cached", V2MeasuredWorkStage.MERT_WINDOWS, 8L, 8L),
            PROFILE,
            "audio/flac",
            100L,
        )

        assertFalse(recorded)
        assertTrue(estimator.snapshot().samples.isEmpty())
    }

    @Test
    fun `coalesced progress records every multi-unit delta at its actual rate`() {
        val estimator = V2StageAwareWorkEstimator(minimumSamples = 1)
        val recorder = V2IndexingEventRateRecorder(estimator)

        assertFalse(
            recorder.onEvent(
                event("track", V2MeasuredWorkStage.MERT_WINDOWS, 0L, 20L),
                PROFILE,
                null,
                100L,
            ),
        )
        assertTrue(
            recorder.onEvent(
                event("track", V2MeasuredWorkStage.MERT_WINDOWS, 5L, 20L),
                PROFILE,
                null,
                600L,
            ),
        )
        assertTrue(
            recorder.onEvent(
                event("track", V2MeasuredWorkStage.MERT_WINDOWS, 12L, 20L),
                PROFILE,
                null,
                1_300L,
            ),
        )
        assertTrue(
            recorder.onEvent(
                event("track", V2MeasuredWorkStage.MERT_WINDOWS, 20L, 20L),
                PROFILE,
                null,
                2_100L,
            ),
        )

        val samples = estimator.snapshot().samples
            .filter { it.stage == V2MeasuredWorkStage.MERT_WINDOWS }
        assertEquals(3, samples.size)
        assertTrue(samples.all { it.activeMsPerUnit == 100.0 })
    }

    @Test
    fun `PCM learning ignores provisional work IDs and denominators until exact EOS completion`() {
        val estimator = V2StageAwareWorkEstimator(minimumSamples = 1)
        val recorder = V2IndexingEventRateRecorder(estimator)

        assertFalse(
            recorder.onEvent(
                event(
                    "provisional-work",
                    V2MeasuredWorkStage.PCM_24K_SAMPLES,
                    null,
                    null,
                    pcmMeasurement(501L, V2PcmRateMeasurementPoint.MATERIALIZATION_STARTED),
                ),
                PROFILE,
                "audio/flac",
                1_000L,
            ),
        )
        assertFalse(
            recorder.onEvent(
                event(
                    "provisional-work",
                    V2MeasuredWorkStage.PCM_24K_SAMPLES,
                    null,
                    null,
                    pcmMeasurement(501L, V2PcmRateMeasurementPoint.MATERIALIZATION_PROGRESS),
                ),
                PROFILE,
                "audio/flac",
                1_080L,
            ),
        )
        assertTrue(estimator.snapshot().samples.isEmpty())

        assertTrue(
            recorder.onEvent(
                event(
                    "decoded-eos-work",
                    V2MeasuredWorkStage.PCM_24K_SAMPLES,
                    24_000L,
                    24_000L,
                    pcmMeasurement(
                        501L,
                        V2PcmRateMeasurementPoint.MATERIALIZATION_COMPLETED_EXACT,
                        exactSampleCount24k = 24_000L,
                    ),
                ),
                PROFILE,
                "audio/flac",
                1_100L,
            ),
        )

        val sample = estimator.snapshot().samples.single()
        assertEquals(V2MeasuredWorkStage.PCM_24K_SAMPLES, sample.stage)
        assertEquals(100.0 / 24_000.0, sample.activeMsPerUnit, 0.0)
    }

    @Test
    fun `verified PCM reuse clears stale measurement without learning a second rate`() {
        val estimator = V2StageAwareWorkEstimator(minimumSamples = 1)
        val recorder = V2IndexingEventRateRecorder(estimator)
        val started = pcmMeasurement(601L, V2PcmRateMeasurementPoint.MATERIALIZATION_STARTED)
        val completed = pcmMeasurement(
            601L,
            V2PcmRateMeasurementPoint.MATERIALIZATION_COMPLETED_EXACT,
            exactSampleCount24k = 24_000L,
        )

        recorder.onEvent(
            event("work", V2MeasuredWorkStage.PCM_24K_SAMPLES, 0L, 24_000L, started),
            PROFILE,
            "audio/flac",
            1_000L,
        )
        assertTrue(
            recorder.onEvent(
                event("work", V2MeasuredWorkStage.PCM_24K_SAMPLES, 24_000L, 24_000L, completed),
                PROFILE,
                "audio/flac",
                1_100L,
            ),
        )
        assertEquals(1, estimator.snapshot().samples.size)

        recorder.onEvent(
            event("retry", V2MeasuredWorkStage.PCM_24K_SAMPLES, 0L, 30_000L, started),
            PROFILE,
            "audio/flac",
            2_000L,
        )
        assertFalse(
            recorder.onEvent(
                event(
                    "retry-final",
                    V2MeasuredWorkStage.PCM_24K_SAMPLES,
                    24_000L,
                    24_000L,
                    pcmMeasurement(
                        601L,
                        V2PcmRateMeasurementPoint.VERIFIED_CACHE_REUSED_EXACT,
                        exactSampleCount24k = 24_000L,
                    ),
                ),
                PROFILE,
                "audio/flac",
                2_100L,
            ),
        )
        assertEquals(1, estimator.snapshot().samples.size)
    }

    @Test
    fun `an intervening stage is never charged to a stale rate interval`() {
        val estimator = V2StageAwareWorkEstimator(minimumSamples = 1)
        val recorder = V2IndexingEventRateRecorder(estimator)

        recorder.onEvent(
            event("track", V2MeasuredWorkStage.MERT_WINDOWS, 0L, 2L),
            PROFILE,
            null,
            0L,
        )
        recorder.onEvent(
            event("track", V2MeasuredWorkStage.CLAMP_SEGMENTS, 0L, 1L),
            PROFILE,
            null,
            10L,
        )
        assertFalse(
            recorder.onEvent(
                event("track", V2MeasuredWorkStage.MERT_WINDOWS, 1L, 2L),
                PROFILE,
                null,
                1_000L,
            ),
        )
        assertTrue(
            recorder.onEvent(
                event("track", V2MeasuredWorkStage.MERT_WINDOWS, 2L, 2L),
                PROFILE,
                null,
                1_010L,
            ),
        )

        val sample = estimator.snapshot().samples.single()
        assertEquals(V2MeasuredWorkStage.MERT_WINDOWS, sample.stage)
        assertEquals(10.0, sample.activeMsPerUnit, 0.0)
    }

    @Test
    fun `multi track overall work stays monotonic and commit completion is not subtracted twice`() {
        var ledger = V2IndexingLedgerStateMachine.startJob(plannedLedger(), 2_000L)
        ledger.jobSpec.tracks.forEachIndexed { index, descriptor ->
            ledger = V2IndexingLedgerStateMachine.beginNextTrackStage(
                ledger,
                descriptor.workId,
                2_010L + index * 2L,
            )
            ledger = V2IndexingLedgerStateMachine.completeActiveTrackStage(
                ledger,
                descriptor.workId,
                null,
                2_011L + index * 2L,
            )
        }
        val first = ledger.jobSpec.tracks.first()
        ledger = V2IndexingLedgerStateMachine.beginNextTrackStage(
            ledger,
            first.workId,
            2_100L,
        )

        val snapshots = mutableListOf(
            V2IndexingOverallWorkPlanner.snapshot(ledger, null, null),
            V2IndexingOverallWorkPlanner.snapshot(
                ledger,
                event(first.workId, V2MeasuredWorkStage.PCM_24K_SAMPLES, 120_000L, 480_000L),
                null,
            ),
            V2IndexingOverallWorkPlanner.snapshot(
                ledger,
                event(first.workId, V2MeasuredWorkStage.MERT_WINDOWS, 0L, first.expectedWork.mertWindows.toLong()),
                null,
            ),
            V2IndexingOverallWorkPlanner.snapshot(
                ledger,
                event(
                    first.workId,
                    V2MeasuredWorkStage.MERT_WINDOWS,
                    first.expectedWork.mertWindows.toLong(),
                    first.expectedWork.mertWindows.toLong(),
                ),
                null,
            ),
        )

        ledger = V2IndexingLedgerStateMachine.completeActiveTrackStage(
            ledger,
            first.workId,
            artifact(ledger, first.workId, VerifiedArtifactKind.MERT_FEATURES, 2_200L),
            2_200L,
        )
        snapshots += V2IndexingOverallWorkPlanner.snapshot(ledger, null, null)
        ledger = V2IndexingLedgerStateMachine.beginNextTrackStage(ledger, first.workId, 2_210L)
        snapshots += V2IndexingOverallWorkPlanner.snapshot(
            ledger,
            event(
                first.workId,
                V2MeasuredWorkStage.CLAMP_SEGMENTS,
                first.expectedWork.clampSegments.toLong(),
                first.expectedWork.clampSegments.toLong(),
            ),
            null,
        )
        ledger = V2IndexingLedgerStateMachine.completeActiveTrackStage(
            ledger,
            first.workId,
            artifact(ledger, first.workId, VerifiedArtifactKind.CLAMP_VECTOR, 2_220L),
            2_220L,
        )
        ledger = V2IndexingLedgerStateMachine.beginNextTrackStage(ledger, first.workId, 2_230L)
        snapshots += V2IndexingOverallWorkPlanner.snapshot(
            ledger,
            event(first.workId, V2MeasuredWorkStage.DATABASE_COMMITS, 0L, 1L),
            null,
        )
        ledger = V2IndexingLedgerStateMachine.completeActiveTrackStage(
            ledger,
            first.workId,
            artifact(ledger, first.workId, VerifiedArtifactKind.DATABASE_COMMIT, 2_240L),
            2_240L,
        )
        val afterDurableCommit = V2IndexingOverallWorkPlanner.snapshot(
            ledger,
            event(first.workId, V2MeasuredWorkStage.DATABASE_COMMITS, 1L, 1L),
            null,
        )
        snapshots += afterDurableCommit

        V2MeasuredWorkStage.entries.forEach { stage ->
            val resolved = snapshots.map { snapshot ->
                snapshot.stages.filter { it.stage == stage }.sumOf { it.resolvedUnits }
            }
            assertTrue("$stage resolved work regressed: $resolved", resolved.zipWithNext().all {
                (before, after) -> after >= before
            })
        }
        val commits = afterDurableCommit.stages.single {
            it.stage == V2MeasuredWorkStage.DATABASE_COMMITS
        }
        assertEquals(1L, commits.completedUnits)
        assertEquals(2L, commits.remainingUnits)
        assertEquals(V2IndexingEtaScope.MEASURED_STAGES_ONLY, afterDurableCommit.etaCoverage.scope)
        assertFalse(afterDurableCommit.etaCoverage.coversWholeJob)
        assertEquals(
            setOf(V2UnmeasuredIndexingWork.VALIDATION_AND_FINAL_PUBLICATION),
            afterDurableCommit.etaCoverage.omittedRemainingWork,
        )

        val pcmCodecs = afterDurableCommit.stages
            .filter { it.stage == V2MeasuredWorkStage.PCM_24K_SAMPLES }
            .mapNotNull { it.codecClass }
            .toSet()
        assertEquals(setOf("audio/flac", "audio/mpeg"), pcmCodecs)
        assertTrue(
            afterDurableCommit.stages
                .filter { it.stage != V2MeasuredWorkStage.PCM_24K_SAMPLES }
                .all { it.codecClass == null },
        )
        assertTrue(afterDurableCommit.remainingMeasuredWork().isNotEmpty())
    }

    @Test
    fun `partially instrumented graph binary IO is counted but excluded from ETA`() {
        val ledger = V2IndexingLedgerStateMachine.startJob(
            plannedLedger(rebuildDerivedIndexes = true),
            2_000L,
        )
        val graphPlan = V2GraphWorkPlan(
            strategy = V2GraphUpdateStrategy.INCREMENTAL,
            targetNodes = 103,
            baseGraphNodes = 100,
            newNodes = 3,
            neighborsPerNode = 5,
            embeddingDimension = 768,
            embeddingRows = 103L,
            similarityDotProducts = 1_000L,
            graphBinaryInputBytes = 100L,
            graphBinaryOutputBytes = 200L,
        )

        val snapshot = V2IndexingOverallWorkPlanner.snapshot(ledger, null, graphPlan)

        assertEquals(
            300L,
            snapshot.stages.single {
                it.stage == V2MeasuredWorkStage.GRAPH_BINARY_BYTES
            }.remainingUnits,
        )
        assertTrue(
            snapshot.remainingMeasuredWork().none {
                it.stage == V2MeasuredWorkStage.GRAPH_BINARY_BYTES
            },
        )
        assertEquals(
            setOf(
                V2UnmeasuredIndexingWork.VALIDATION_AND_FINAL_PUBLICATION,
                V2UnmeasuredIndexingWork.DERIVED_GRAPH_BINARY_IO_WITHOUT_COMPLETE_BOUNDARIES,
            ),
            snapshot.etaCoverage.omittedRemainingWork,
        )
    }

    @Test
    fun `verified cached PCM is resolved in an event-free reopened snapshot`() {
        var ledger = V2IndexingLedgerStateMachine.startJob(plannedLedger(), 2_000L)
        val first = ledger.jobSpec.tracks.first()
        ledger = V2IndexingLedgerStateMachine.beginNextTrackStage(ledger, first.workId, 2_010L)
        ledger = V2IndexingLedgerStateMachine.completeActiveTrackStage(
            ledger,
            first.workId,
            artifact = null,
            nowEpochMs = 2_020L,
        )

        val before = V2IndexingOverallWorkPlanner.snapshot(ledger, null, null)
        val reopened = V2IndexingOverallWorkPlanner.snapshot(
            ledger,
            event = null,
            graphPlan = null,
            verifiedPcmWorkIds = setOf(first.workId),
        )
        val beforeFlac = before.stages.single {
            it.stage == V2MeasuredWorkStage.PCM_24K_SAMPLES && it.codecClass == "audio/flac"
        }
        val reopenedFlac = reopened.stages.single {
            it.stage == V2MeasuredWorkStage.PCM_24K_SAMPLES && it.codecClass == "audio/flac"
        }

        assertEquals(0L, beforeFlac.completedUnits)
        assertEquals(first.finalizedAudioSpan.exactSampleCount24k, reopenedFlac.completedUnits)
        assertEquals(
            beforeFlac.remainingUnits - first.finalizedAudioSpan.exactSampleCount24k,
            reopenedFlac.remainingUnits,
        )
    }

    @Test
    fun `live MERT events retain verified noncurrent Opus PCM and keep ETA available`() {
        var ledger = V2IndexingLedgerStateMachine.startJob(
            plannedLedger(
                selectedTracks = listOf(
                    selectedTrack(101L, "audio/flac", 'd'),
                    selectedTrack(102L, "audio/mpeg", 'e'),
                    selectedTrack(103L, "audio/opus", 'f'),
                ),
            ),
            2_000L,
        )
        ledger.jobSpec.tracks.forEachIndexed { index, descriptor ->
            ledger = V2IndexingLedgerStateMachine.beginNextTrackStage(
                ledger,
                descriptor.workId,
                2_010L + index * 2L,
            )
            ledger = V2IndexingLedgerStateMachine.completeActiveTrackStage(
                ledger,
                descriptor.workId,
                null,
                2_011L + index * 2L,
            )
        }
        val flac = ledger.jobSpec.tracks.single {
            it.finalizedAudioSpan.container.mime == "audio/flac"
        }
        val opus = ledger.jobSpec.tracks.single {
            it.finalizedAudioSpan.container.mime == "audio/opus"
        }
        ledger = V2IndexingLedgerStateMachine.beginNextTrackStage(ledger, flac.workId, 2_100L)
        val mertEvent = event(
            flac.workId,
            V2MeasuredWorkStage.MERT_WINDOWS,
            completed = 1L,
            total = flac.expectedWork.mertWindows.toLong(),
        )
        val tracker = V2VerifiedPcmProgressTracker(setOf(opus.workId))
        val verified = tracker.snapshot(ledger, mertEvent)
        val snapshot = V2IndexingOverallWorkPlanner.snapshot(
            ledger,
            event = mertEvent,
            graphPlan = null,
            verifiedPcmWorkIds = verified,
        )

        val opusPcm = snapshot.stages.single {
            it.stage == V2MeasuredWorkStage.PCM_24K_SAMPLES && it.codecClass == "audio/opus"
        }
        assertEquals(opus.finalizedAudioSpan.exactSampleCount24k, opusPcm.completedUnits)
        assertEquals(0L, opusPcm.remainingUnits)

        val estimator = V2StageAwareWorkEstimator(minimumSamples = 1)
        listOf("audio/flac", "audio/mpeg").forEach { codec ->
            estimator.recordCompleted(
                V2MeasuredWorkStage.PCM_24K_SAMPLES,
                PROFILE,
                completedUnits = 24_000L,
                activeDurationMs = 100L,
                codecClass = codec,
            )
        }
        listOf(
            V2MeasuredWorkStage.MERT_WINDOWS,
            V2MeasuredWorkStage.CLAMP_SEGMENTS,
            V2MeasuredWorkStage.DATABASE_COMMITS,
        ).forEach { stage ->
            estimator.recordCompleted(stage, PROFILE, completedUnits = 1L, activeDurationMs = 100L)
        }
        assertTrue(
            estimator.estimate(snapshot.remainingMeasuredWork(), PROFILE).remainingMs != null,
        )

        val withoutReceipts = V2IndexingOverallWorkPlanner.snapshot(
            ledger,
            event = mertEvent,
            graphPlan = null,
            verifiedPcmWorkIds = emptySet(),
        )
        assertEquals(
            null,
            estimator.estimate(withoutReceipts.remainingMeasuredWork(), PROFILE).remainingMs,
        )
    }

    @Test
    fun `verified PCM resolver caches file identity and invalidates deletion and work remap`() {
        val root = Files.createTempDirectory("v2-pcm-telemetry-").toFile()
        try {
            val ledger = plannedLedger()
            val first = ledger.jobSpec.tracks.first()
            val cacheDirectory = File(root, "pcm-cache-v1").also { assertTrue(it.mkdirs()) }
            File(cacheDirectory, "${first.powerampFileId}.pcm-24k-f32.bin")
                .writeBytes(byteArrayOf(1, 2, 3, 4))
            val receipt = File(cacheDirectory, "${first.powerampFileId}.receipt.json")
                .also { it.writeText("receipt-v1") }
            val calls = AtomicInteger(0)
            val rejectingResolver = V2VerifiedPcmCompletionResolver(
                V2VerifiedPcmReceiptVerifier { _, _, _ -> false },
            )
            assertTrue(rejectingResolver.verifiedWorkIds(ledger, root).isEmpty())

            val resolver = V2VerifiedPcmCompletionResolver(
                V2VerifiedPcmReceiptVerifier { _, _, _ ->
                    calls.incrementAndGet()
                    true
                },
            )

            assertEquals(setOf(first.workId), resolver.verifiedWorkIds(ledger, root))
            assertEquals(setOf(first.workId), resolver.verifiedWorkIds(ledger, root))
            assertEquals(1, calls.get())

            receipt.appendText("-changed")
            assertEquals(setOf(first.workId), resolver.verifiedWorkIds(ledger, root))
            assertEquals(2, calls.get())

            val remappedDescriptor = first.copy(
                workId = "decoded-${first.workId}",
                provisionalWorkId = first.workId,
            )
            val remappedLedger = ledger.copy(
                jobSpec = ledger.jobSpec.copy(
                    tracks = ledger.jobSpec.tracks.map {
                        if (it.workId == first.workId) remappedDescriptor else it
                    },
                ),
                revision = ledger.revision + 1L,
                tracks = ledger.tracks.map {
                    if (it.workId == first.workId) it.copy(workId = remappedDescriptor.workId) else it
                },
            )
            assertEquals(
                setOf(remappedDescriptor.workId),
                resolver.verifiedWorkIds(remappedLedger, root),
            )
            assertEquals(3, calls.get())

            val sourceChangedDescriptor = remappedDescriptor.copy(
                sourceFingerprint = remappedDescriptor.sourceFingerprint.copy(
                    sizeBytes = remappedDescriptor.sourceFingerprint.sizeBytes + 1L,
                ),
            )
            val sourceChangedLedger = remappedLedger.copy(
                jobSpec = remappedLedger.jobSpec.copy(
                    tracks = remappedLedger.jobSpec.tracks.map {
                        if (it.workId == remappedDescriptor.workId) sourceChangedDescriptor else it
                    },
                ),
                revision = remappedLedger.revision + 1L,
            )
            assertEquals(
                setOf(sourceChangedDescriptor.workId),
                resolver.verifiedWorkIds(sourceChangedLedger, root),
            )
            assertEquals(4, calls.get())

            assertTrue(receipt.delete())
            assertTrue(resolver.verifiedWorkIds(sourceChangedLedger, root).isEmpty())
            assertEquals(4, calls.get())
        } finally {
            root.deleteRecursively()
        }
    }

    @Test
    fun `cancelled job resolves unfinished measured work as abandoned`() {
        val running = V2IndexingLedgerStateMachine.startJob(plannedLedger(), 2_000L)
        val cancelling = V2IndexingLedgerStateMachine.requestCancel(running, 2_010L)
        val cancelled = V2IndexingLedgerStateMachine.finishCancel(cancelling, 2_020L)

        val snapshot = V2IndexingOverallWorkPlanner.snapshot(cancelled, null, null)

        assertTrue(snapshot.stages.all { it.remainingUnits == 0L })
        assertTrue(snapshot.stages.all { it.resolvedUnits == it.plannedUnits })
        assertTrue(snapshot.remainingMeasuredWork().isEmpty())
        assertEquals(V2IndexingEtaScope.WHOLE_JOB, snapshot.etaCoverage.scope)
        assertTrue(snapshot.etaCoverage.coversWholeJob)
    }

    private fun plannedLedger(
        rebuildDerivedIndexes: Boolean = false,
        selectedTracks: List<SelectedTrackInput> = listOf(
            selectedTrack(101L, "audio/flac", 'd'),
            selectedTrack(102L, "audio/flac", 'e'),
            selectedTrack(103L, "audio/mpeg", 'f'),
        ),
    ): IndexingJobLedger {
        val embeddingSpec = V2IndexingLedgerPlanner.createEmbeddingSpec(
            EmbeddingSpecInput(
                preprocessingSpecId = V2IndexingWorkPolicy.PREPROCESSING_SPEC_ID,
                decoderPolicyId = V2IndexingWorkPolicy.DECODER_POLICY_ID,
                inferenceBackendPolicyId = V2IndexingWorkPolicy.INFERENCE_BACKEND_POLICY_ID,
                outputDimension = 768,
                modelArtifactSha256 = mapOf("mert" to sha('a'), "clamp3_audio" to sha('b')),
            ),
        )
        val textSpec = V2IndexingLedgerPlanner.createTextRetrievalSpec(
            TextRetrievalSpecInput(
                compatibleAudioEmbeddingSpecId = embeddingSpec.specId,
                textModelSha256 = sha('c'),
                tokenizerModelSha256 = V2IndexingWorkPolicy.TEXT_TOKENIZER_MODEL_SHA256,
                tokenizerPolicyId = V2IndexingWorkPolicy.TEXT_TOKENIZER_POLICY_ID,
                tokenizerRuntimeContractSha256 =
                    V2IndexingWorkPolicy.TEXT_TOKENIZER_RUNTIME_CONTRACT_SHA256,
                outputSpaceId = V2IndexingWorkPolicy.TEXT_OUTPUT_SPACE_ID,
                outputDimension = embeddingSpec.outputDimension,
                inferenceBackendPolicyId = V2IndexingWorkPolicy.TEXT_INFERENCE_BACKEND_POLICY_ID,
            ),
        )
        return V2IndexingLedgerPlanner.planJob(
            providerSnapshot = PowerampProviderSnapshotEvidence(
                libraryGeneration = PROVIDER_GENERATION,
                acquisition = V2ProviderSnapshotAcquisitionEvidence(
                    queryUri = "content://com.maxmpz.audioplayer.data/files",
                    requestedColumns = listOf("_id", "duration", "path"),
                    returnedColumns = listOf("_id", "duration", "path"),
                    rowCount = selectedTracks.size,
                    cursorExhaustedNormally = true,
                ),
            ),
            embeddingSpec = embeddingSpec,
            textRetrievalSpec = textSpec,
            runtimeFingerprint = IndexingRuntimeFingerprint(
                appVersionCode = 2_000_000L,
                appBuildId = "telemetry-test",
                decoderRuntimeId = "android-mediacodec-test",
                platformFingerprint = "android-test-device",
            ),
            selectedTracks = selectedTracks,
            rebuildDerivedIndexes = rebuildDerivedIndexes,
            createdAtEpochMs = 1_000L,
            jobId = "telemetry-job",
        )
    }

    private fun selectedTrack(id: Long, mime: String, fingerprint: Char): SelectedTrackInput {
        val path = "/storage/emulated/0/Music/$id.${if (mime == "audio/flac") "flac" else "mp3"}"
        val durationMs = 20_000L
        val offsetMs = 1_000L
        val startSample = offsetMs * 24L
        val sampleCount = durationMs * 24L
        val endSample = startSample + sampleCount
        val providerSpan = V2ProviderSpanEvidence(
            offsetUs = offsetMs * 1_000L,
            durationUs = durationMs * 1_000L,
            endExclusiveUs = (offsetMs + durationMs) * 1_000L,
        )
        return SelectedTrackInput(
            powerampFileId = id,
            providerSnapshotGeneration = PROVIDER_GENERATION,
            providerRow = V2ProviderPathRowEvidence(
                powerampFileId = id,
                physicalPath = path,
                providerPhysicalPath = path,
                artist = "Artist $id",
                album = "Album",
                title = "Track $id",
                offsetMs = offsetMs,
                durationMs = durationMs,
                cueSourceImageFolderId = null,
            ),
            displayMetadata = DisplayTrackMetadata("Artist $id", "Album", "Track $id"),
            normalizedMetadata = NormalizedTrackMetadata(
                normalizationSpecId = "track-normalization-v2",
                artist = "artist $id",
                album = "album",
                title = "track $id",
                metadataKey = "artist $id|album|track $id|20000",
            ),
            physicalPath = path,
            sourceFingerprint = SourceFingerprint(
                fingerprintSpecId = "stat-plus-samples-v1",
                sizeBytes = 1_000_000L + id,
                lastModifiedEpochMs = 1_700_000_000_000L + id,
                fileKey = "dev=1;ino=$id",
                sampledContentSha256 = sha(fingerprint),
                fullContentSha256 = null,
            ),
            finalizedAudioSpan = FinalizedAudioSpanEvidence(
                kind = V2ResolvedAudioSpanKind.LOGICAL_CUE,
                authority = V2AudioSpanAuthority.PROVIDER_CUE_HALF_OPEN_SPAN,
                executionBoundaryRequirement =
                    V2ExecutionBoundaryRequirement.ENFORCE_PROVIDER_HALF_OPEN_SPAN,
                providerSpan = providerSpan,
                cueClassification = V2CueClassificationEvidence(
                    providerGroupRowCount = 1,
                    logicalRowCount = 1,
                    nonZeroOffsetRowIds = listOf(id),
                    rawSourceImageRowIds = emptyList(),
                ),
                container = V2AudioContainerEvidence(
                    physicalPath = path,
                    audioTrackIndex = 0,
                    durationUsEstimate = 21_000_000L,
                    sampleRateHz = 24_000,
                    channelCount = 2,
                    mime = mime,
                ),
                startUs = providerSpan.offsetUs,
                endExclusiveUs = providerSpan.endExclusiveUs,
                startSourceSample = startSample,
                endSourceSampleExclusive = endSample,
                sourceSampleCount = sampleCount,
                exactSampleCount24k = sampleCount,
                expectedWork = V2AudioSpanMath.expectedWorkFor24kSamples(sampleCount),
            ),
        )
    }

    private fun artifact(
        ledger: IndexingJobLedger,
        workId: String,
        kind: VerifiedArtifactKind,
        nowEpochMs: Long,
    ): VerifiedArtifact {
        val descriptor = ledger.jobSpec.tracks.single { it.workId == workId }
        val units = when (kind) {
            VerifiedArtifactKind.MERT_FEATURES -> descriptor.expectedWork.mertWindows
            VerifiedArtifactKind.CLAMP_VECTOR -> descriptor.expectedWork.clampSegments
            VerifiedArtifactKind.DATABASE_COMMIT -> 1
        }
        val span = descriptor.finalizedAudioSpan
        return VerifiedArtifact(
            kind = kind,
            storageKey = "${ledger.jobSpec.jobId}/$workId/${kind.name.lowercase()}",
            byteLength = units * V2_CLAMP3_BLOB_BYTES.toLong(),
            sha256 = sha('9'),
            completedUnits = units,
            plannedUnits = units,
            embeddingSpecId = ledger.jobSpec.embeddingSpec.specId,
            sourceFingerprint = descriptor.sourceFingerprint,
            verifiedAtEpochMs = nowEpochMs,
            executionBoundary = if (kind == VerifiedArtifactKind.MERT_FEATURES) {
                VerifiedExecutionBoundaryEvidence(
                    requirement = span.executionBoundaryRequirement,
                    observedStartSourceSample = span.startSourceSample,
                    observedEndSourceSampleExclusive = span.endSourceSampleExclusive,
                    observedSourceSampleCount = span.sourceSampleCount,
                    exactSampleCount24k = span.exactSampleCount24k,
                    endOfStreamReached = false,
                    providerBoundaryEnforced = true,
                )
            } else {
                null
            },
        )
    }

    private fun event(
        workId: String,
        stage: V2MeasuredWorkStage,
        completed: Long?,
        total: Long?,
        pcmRateMeasurement: V2PcmRateMeasurement? = null,
    ) = V2IndexingExecutorEvent(
        jobId = "telemetry-job",
        workId = workId,
        trackOrdinal = null,
        trackTitle = null,
        stage = stage,
        completedUnits = completed,
        totalUnits = total,
        detail = "production event",
        pcmRateMeasurement = pcmRateMeasurement,
    )

    private fun pcmMeasurement(
        powerampFileId: Long,
        point: V2PcmRateMeasurementPoint,
        exactSampleCount24k: Long? = null,
    ) = V2PcmRateMeasurement(powerampFileId, point, exactSampleCount24k)

    private fun sha(character: Char): String = character.toString().repeat(64)

    private companion object {
        val PROFILE = V2IndexingExecutionProfile.BALANCED
        const val PROVIDER_GENERATION =
            "poweramp-provider-snapshot-v3-sha256:" +
                "7777777777777777777777777777777777777777777777777777777777777777"
    }
}
