package com.powerampstartradio.ui

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class DriftControlPolicyTest {
    @Test
    fun defaultsUseTheMeasuredV2MomentumOperatingPoint() {
        assertEquals(0.9f, RadioConfig.DEFAULT_MOMENTUM_BETA, 0f)
        assertEquals(RadioConfig.DEFAULT_MOMENTUM_BETA, RadioConfig().momentumBeta, 0f)
        assertTrue(RadioConfig.DEFAULT_MOMENTUM_BETA in DriftControlPolicy.MOMENTUM_OPTIONS)
    }

    @Test
    fun followLastPickAppearsOnlyInSeedInterpolation() {
        assertTrue(0f in DriftControlPolicy.SEED_PULL_OPTIONS)
        assertFalse(0f in DriftControlPolicy.MOMENTUM_OPTIONS)
        assertTrue(
            DriftControlPolicy.isFollowLastPick(
                mode = DriftMode.SEED_INTERPOLATION,
                anchorStrength = 0f,
                momentumBeta = 0.5f,
            )
        )
    }

    @Test
    fun fixedSeedQueryAppearsOnlyAsMomentumEndpoint() {
        assertEquals(1f, DriftControlPolicy.MOMENTUM_OPTIONS.last(), 0f)
        assertFalse(1f in DriftControlPolicy.seedPullOptions(DecaySchedule.NONE))
        assertFalse(DecaySchedule.NONE in DriftControlPolicy.decaySchedules(1f))
        assertTrue(1f in DriftControlPolicy.seedPullOptions(DecaySchedule.EXPONENTIAL))
        assertTrue(
            DriftControlPolicy.usesFixedSeedQuery(
                mode = DriftMode.MOMENTUM,
                anchorStrength = 0.5f,
                anchorDecay = DecaySchedule.EXPONENTIAL,
                momentumBeta = 1f,
            )
        )
    }

    @Test
    fun zeroSeedPullMakesEveryFadeControlInapplicable() {
        assertFalse(DriftControlPolicy.seedFadeApplies(0f))
        assertTrue(DriftControlPolicy.seedFadeApplies(0.25f))
    }

    @Test
    fun driftSlidersExposeOnlyFrozenMatrixStops() {
        assertEquals(
            listOf(0f, 0.25f, 0.5f, 0.75f, 0.85f, 1f),
            DriftControlPolicy.SEED_PULL_OPTIONS,
        )
        assertEquals(
            listOf(0.25f, 0.5f, 0.75f, 0.9f, 0.95f, 1f),
            DriftControlPolicy.MOMENTUM_OPTIONS,
        )
        assertEquals(
            listOf(1f, 3f, 7f, 10f, 15f, 30f),
            DriftControlPolicy.FADE_TIMING_OPTIONS,
        )
    }

    @Test
    fun stepTimingOnlyExposesStopsWhichCanChangeTheRequestedQueue() {
        assertEquals(
            listOf(1f, 3f, 7f),
            DriftControlPolicy.fadeTimingOptions(DecaySchedule.STEP, recommendationCount = 10),
        )
        assertEquals(
            listOf(1f, 3f, 7f, 10f, 15f, 30f),
            DriftControlPolicy.fadeTimingOptions(DecaySchedule.STEP, recommendationCount = 100),
        )
        assertFalse(DriftControlPolicy.stepTimingCanAffectQueue(10f, recommendationCount = 10))
        assertTrue(DriftControlPolicy.stepTimingCanAffectQueue(7f, recommendationCount = 10))
        assertEquals(
            7f,
            DriftControlPolicy.canonicalFadeTiming(
                decay = DecaySchedule.STEP,
                timingTracks = 30f,
                recommendationCount = 10,
            ),
            0f,
        )
        assertEquals(8, DriftControlPolicy.stepDropAfterPickCount(7f))
    }

    @Test
    fun nonStepAbsoluteTimingsRemainIndependentOfQueueLength() {
        assertEquals(
            DriftControlPolicy.FADE_TIMING_OPTIONS,
            DriftControlPolicy.fadeTimingOptions(
                DecaySchedule.EXPONENTIAL,
                recommendationCount = 10,
            ),
        )
    }

    @Test
    fun nonMmrRequestsSuspendSavedDriftWithoutErasingIt() {
        val savedMmrState = RadioConfig(
            selectionMode = SelectionMode.MMR,
            driftEnabled = true,
            driftMode = DriftMode.MOMENTUM,
            momentumBeta = 0.75f,
        )

        SelectionMode.entries.filterNot { it == SelectionMode.MMR }.forEach { mode ->
            val effective = savedMmrState.copy(selectionMode = mode).forActiveSelectionMode()
            assertFalse("$mode must not emit hidden drift", effective.driftEnabled)
            assertEquals(DriftMode.MOMENTUM, effective.driftMode)
            assertEquals(0.75f, effective.momentumBeta, 0f)
        }

        assertTrue(savedMmrState.driftEnabled)
        assertEquals(savedMmrState, savedMmrState.forActiveSelectionMode())
    }
}
