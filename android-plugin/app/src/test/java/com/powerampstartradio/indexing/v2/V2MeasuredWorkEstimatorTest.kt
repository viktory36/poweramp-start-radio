package com.powerampstartradio.indexing.v2

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class V2MeasuredWorkEstimatorTest {
    @Test
    fun `ETA never averages heterogeneous stage units`() {
        val estimator = V2StageAwareWorkEstimator(minimumSamples = 2)
        repeat(2) {
            estimator.recordCompleted(
                V2MeasuredWorkStage.MERT_WINDOWS,
                V2IndexingExecutionProfile.BALANCED,
                completedUnits = 1,
                activeDurationMs = 1_000,
            )
            estimator.recordCompleted(
                V2MeasuredWorkStage.CLAMP_SEGMENTS,
                V2IndexingExecutionProfile.BALANCED,
                completedUnits = 1,
                activeDurationMs = 10,
            )
        }

        val estimate = estimator.estimate(
            listOf(
                V2RemainingStageWork(V2MeasuredWorkStage.MERT_WINDOWS, 3),
                V2RemainingStageWork(V2MeasuredWorkStage.CLAMP_SEGMENTS, 10),
            ),
            V2IndexingExecutionProfile.BALANCED,
        )
        assertEquals(3_100L, estimate.remainingMs)
        assertTrue(estimate.calibratingStages.isEmpty())
    }

    @Test
    fun `unknown activation tail keeps ETA calibrating instead of pretending precision`() {
        val estimator = V2StageAwareWorkEstimator(minimumSamples = 1)
        estimator.recordCompleted(
            V2MeasuredWorkStage.MERT_WINDOWS,
            V2IndexingExecutionProfile.FULL,
            completedUnits = 2,
            activeDurationMs = 200,
        )
        val estimate = estimator.estimate(
            listOf(
                V2RemainingStageWork(V2MeasuredWorkStage.MERT_WINDOWS, 3),
                V2RemainingStageWork(V2MeasuredWorkStage.ACTIVATION_TRACKS, 100),
            ),
            V2IndexingExecutionProfile.FULL,
        )

        assertNull(estimate.remainingMs)
        assertEquals(setOf(V2MeasuredWorkStage.ACTIVATION_TRACKS), estimate.calibratingStages)
    }

    @Test
    fun `same codec other profile supplies only a wide calibrating prior`() {
        val estimator = V2StageAwareWorkEstimator(minimumSamples = 1)
        estimator.recordCompleted(
            V2MeasuredWorkStage.PCM_24K_SAMPLES,
            V2IndexingExecutionProfile.FULL,
            completedUnits = 24_000,
            activeDurationMs = 100,
            codecClass = "audio/flac",
        )
        val background = estimator.estimate(
            listOf(
                V2RemainingStageWork(
                    V2MeasuredWorkStage.PCM_24K_SAMPLES,
                    24_000,
                    "audio/flac",
                ),
            ),
            V2IndexingExecutionProfile.BACKGROUND,
        )
        assertEquals(250L, background.remainingMs)
        assertEquals(100L, background.lowerBoundMs)
        assertEquals(400L, background.upperBoundMs)
        assertEquals(setOf(V2MeasuredWorkStage.PCM_24K_SAMPLES), background.calibratingStages)

        val fullMp3 = estimator.estimate(
            listOf(
                V2RemainingStageWork(
                    V2MeasuredWorkStage.PCM_24K_SAMPLES,
                    24_000,
                    "audio/mpeg",
                ),
            ),
            V2IndexingExecutionProfile.FULL,
        )
        assertNull(fullMp3.remainingMs)
    }

    @Test
    fun `generic decode history never substitutes for a codec partition`() {
        val estimator = V2StageAwareWorkEstimator(minimumSamples = 1)
        estimator.recordCompleted(
            V2MeasuredWorkStage.PCM_24K_SAMPLES,
            V2IndexingExecutionProfile.FULL,
            completedUnits = 24_000,
            activeDurationMs = 100,
            codecClass = null,
        )

        val estimate = estimator.estimate(
            listOf(
                V2RemainingStageWork(
                    V2MeasuredWorkStage.PCM_24K_SAMPLES,
                    24_000,
                    "audio/flac",
                ),
            ),
            V2IndexingExecutionProfile.FULL,
        )

        assertNull(estimate.remainingMs)
        assertEquals(setOf(V2MeasuredWorkStage.PCM_24K_SAMPLES), estimate.calibratingStages)
    }

    @Test
    fun `exact profile observations replace and tighten a cross profile prior`() {
        val estimator = V2StageAwareWorkEstimator(minimumSamples = 2)
        repeat(2) {
            estimator.recordCompleted(
                V2MeasuredWorkStage.CLAMP_SEGMENTS,
                V2IndexingExecutionProfile.FULL,
                completedUnits = 1,
                activeDurationMs = 100,
            )
        }

        val borrowed = estimator.estimate(
            listOf(V2RemainingStageWork(V2MeasuredWorkStage.CLAMP_SEGMENTS, 10)),
            V2IndexingExecutionProfile.BACKGROUND,
        )
        assertEquals(2_500L, borrowed.remainingMs)
        assertEquals(1_000L, borrowed.lowerBoundMs)
        assertEquals(4_000L, borrowed.upperBoundMs)
        assertEquals(setOf(V2MeasuredWorkStage.CLAMP_SEGMENTS), borrowed.calibratingStages)

        estimator.recordCompleted(
            V2MeasuredWorkStage.CLAMP_SEGMENTS,
            V2IndexingExecutionProfile.BACKGROUND,
            completedUnits = 1,
            activeDurationMs = 300,
        )
        val partlyCalibrated = estimator.estimate(
            listOf(V2RemainingStageWork(V2MeasuredWorkStage.CLAMP_SEGMENTS, 10)),
            V2IndexingExecutionProfile.BACKGROUND,
        )
        assertEquals(3_000L, partlyCalibrated.remainingMs)
        assertEquals(1_000L, partlyCalibrated.lowerBoundMs)
        assertEquals(4_500L, partlyCalibrated.upperBoundMs)
        assertEquals(setOf(V2MeasuredWorkStage.CLAMP_SEGMENTS), partlyCalibrated.calibratingStages)

        estimator.recordCompleted(
            V2MeasuredWorkStage.CLAMP_SEGMENTS,
            V2IndexingExecutionProfile.BACKGROUND,
            completedUnits = 1,
            activeDurationMs = 320,
        )
        val calibrated = estimator.estimate(
            listOf(V2RemainingStageWork(V2MeasuredWorkStage.CLAMP_SEGMENTS, 10)),
            V2IndexingExecutionProfile.BACKGROUND,
        )
        assertEquals(3_100L, calibrated.remainingMs)
        assertEquals(3_040L, calibrated.lowerBoundMs)
        assertEquals(3_160L, calibrated.upperBoundMs)
        assertTrue(calibrated.calibratingStages.isEmpty())
    }

    @Test
    fun `one exact sample without a same phone prior keeps learning`() {
        val estimator = V2StageAwareWorkEstimator(minimumSamples = 2)
        estimator.recordCompleted(
            V2MeasuredWorkStage.DATABASE_COMMITS,
            V2IndexingExecutionProfile.BACKGROUND,
            completedUnits = 1,
            activeDurationMs = 40,
        )

        val estimate = estimator.estimate(
            listOf(V2RemainingStageWork(V2MeasuredWorkStage.DATABASE_COMMITS, 10)),
            V2IndexingExecutionProfile.BACKGROUND,
        )

        assertNull(estimate.remainingMs)
        assertEquals(setOf(V2MeasuredWorkStage.DATABASE_COMMITS), estimate.calibratingStages)
    }

    @Test
    fun `all profiles share exact decode partition`() {
        val chunks = V2IndexingExecutionProfile.entries.map {
            V2IndexingExecutionPolicies.schedule(it).pcmChunkDurationMs
        }.toSet()
        assertEquals(setOf(V2IndexingExecutionPolicies.BYTE_STABLE_PCM_CHUNK_DURATION_MS), chunks)
        assertTrue(
            V2IndexingExecutionPolicies.schedule(V2IndexingExecutionProfile.BACKGROUND)
                .yieldAfterCompletedUnitMs >
                V2IndexingExecutionPolicies.schedule(V2IndexingExecutionProfile.FULL)
                    .yieldAfterCompletedUnitMs,
        )
    }

    @Test
    fun `persisted active-stage samples survive estimator restart`() {
        val beforeRestart = V2StageAwareWorkEstimator(minimumSamples = 2)
        repeat(2) {
            beforeRestart.recordCompleted(
                stage = V2MeasuredWorkStage.MERT_WINDOWS,
                profile = V2IndexingExecutionProfile.BALANCED,
                completedUnits = 2,
                activeDurationMs = 400,
            )
        }

        val afterRestart = V2StageAwareWorkEstimator(
            minimumSamples = 2,
            restoredSnapshot = beforeRestart.snapshot(),
        )
        val estimate = afterRestart.estimate(
            listOf(V2RemainingStageWork(V2MeasuredWorkStage.MERT_WINDOWS, 3)),
            V2IndexingExecutionProfile.BALANCED,
        )

        assertEquals(600L, estimate.remainingMs)
        assertTrue(estimate.calibratingStages.isEmpty())
    }

    @Test
    fun `invalid restored rates fall back to calibrating and samples stay bounded`() {
        val estimator = V2StageAwareWorkEstimator(
            minimumSamples = 1,
            maximumSamples = 2,
            restoredSnapshot = V2PersistedStageRateSnapshot(
                samples = listOf(
                    V2PersistedStageRateSample(
                        V2MeasuredWorkStage.CLAMP_SEGMENTS,
                        V2IndexingExecutionProfile.FULL,
                        null,
                        Double.NaN,
                    ),
                ),
            ),
        )
        assertNull(
            estimator.estimate(
                listOf(V2RemainingStageWork(V2MeasuredWorkStage.CLAMP_SEGMENTS, 1)),
                V2IndexingExecutionProfile.FULL,
            ).remainingMs,
        )

        listOf(100L, 200L, 300L).forEach { duration ->
            estimator.recordCompleted(
                V2MeasuredWorkStage.CLAMP_SEGMENTS,
                V2IndexingExecutionProfile.FULL,
                completedUnits = 1,
                activeDurationMs = duration,
            )
        }
        assertEquals(2, estimator.snapshot().samples.size)
        assertEquals(200.0, estimator.snapshot().samples.first().activeMsPerUnit, 0.0)
    }
}
