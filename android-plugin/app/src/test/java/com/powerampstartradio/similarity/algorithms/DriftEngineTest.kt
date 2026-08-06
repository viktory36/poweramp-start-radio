package com.powerampstartradio.similarity.algorithms

import com.powerampstartradio.ui.DecaySchedule
import com.powerampstartradio.ui.DriftMode
import com.powerampstartradio.ui.RadioConfig
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test

class DriftEngineTest {
    @Test
    fun exponentialFadeUsesAbsoluteTrackHalfLife() {
        assertEquals(
            0.4f,
            DriftEngine.computeAlpha(
                baseAlpha = 0.8f,
                step = 6,
                decay = DecaySchedule.EXPONENTIAL,
                halfLifeTracks = 6f,
            ),
            1e-6f,
        )
    }

    @Test
    fun allSchedulesHaveExplicitAbsoluteSemantics() {
        assertEquals(0.8f, alpha(DecaySchedule.NONE, 8), 1e-6f)
        assertEquals(0.4f, alpha(DecaySchedule.LINEAR, 8), 1e-6f)
        assertEquals(0.4f, alpha(DecaySchedule.EXPONENTIAL, 8), 1e-6f)
        assertEquals(0.16f, alpha(DecaySchedule.STEP, 8), 1e-6f)
    }

    @Test
    fun nextQueryDoesNotDependOnRequestedQueueLength() {
        val seed = floatArrayOf(1f, 0f)
        val current = floatArrayOf(0f, 1f)
        val shortRequest = RadioConfig(
            numTracks = 10,
            driftMode = DriftMode.SEED_INTERPOLATION,
            anchorStrength = 0.8f,
            anchorDecay = DecaySchedule.EXPONENTIAL,
            anchorHalfLifeTracks = 6f,
        )
        val longRequest = shortRequest.copy(numTracks = 100)

        val shortQuery = DriftEngine.updateQuery(seed, current, null, 5, shortRequest).query
        val longQuery = DriftEngine.updateQuery(seed, current, null, 5, longRequest).query

        assertArrayEquals(shortQuery, longQuery, 0f)
    }

    @Test
    fun defaultHalfLifeIsTheSevenTrackAbsoluteContract() {
        val base = 0.8f
        assertEquals(7f, RadioConfig.DEFAULT_ANCHOR_HALF_LIFE_TRACKS, 0f)
        assertEquals(
            base / 2f,
            DriftEngine.computeAlpha(
                baseAlpha = base,
                step = 7,
                decay = DecaySchedule.EXPONENTIAL,
                halfLifeTracks = RadioConfig.DEFAULT_ANCHOR_HALF_LIFE_TRACKS,
            ),
            1e-6f,
        )
        assertEquals(
            base / 4f,
            DriftEngine.computeAlpha(
                baseAlpha = base,
                step = 14,
                decay = DecaySchedule.EXPONENTIAL,
                halfLifeTracks = RadioConfig.DEFAULT_ANCHOR_HALF_LIFE_TRACKS,
            ),
            1e-6f,
        )
    }

    @Test
    fun zeroMomentumAndZeroSeedPullAreTheSameLastPickQuery() {
        val seed = floatArrayOf(1f, 0f)
        val current = floatArrayOf(0.6f, 0.8f)
        val seedInterpolation = RadioConfig(
            driftMode = DriftMode.SEED_INTERPOLATION,
            anchorStrength = 0f,
            anchorDecay = DecaySchedule.STEP,
            anchorHalfLifeTracks = 3f,
        )
        val momentum = seedInterpolation.copy(
            driftMode = DriftMode.MOMENTUM,
            momentumBeta = 0f,
        )

        assertArrayEquals(
            DriftEngine.updateQuery(seed, current, null, 12, seedInterpolation).query,
            DriftEngine.updateQuery(seed, current, seed, 12, momentum).query,
            0f,
        )
    }

    @Test
    fun zeroSeedPullMakesFadeScheduleAndTimingExactNoOps() {
        val seed = floatArrayOf(1f, 0f)
        val current = floatArrayOf(0.6f, 0.8f)
        val baseline = RadioConfig(
            driftMode = DriftMode.SEED_INTERPOLATION,
            anchorStrength = 0f,
            anchorDecay = DecaySchedule.NONE,
            anchorHalfLifeTracks = 1f,
        )
        val expected = DriftEngine.updateQuery(seed, current, null, 20, baseline).query

        DecaySchedule.entries.forEach { schedule ->
            for (timing in listOf(1f, 3f, 7f, 30f)) {
                assertArrayEquals(
                    "$schedule at $timing tracks",
                    expected,
                    DriftEngine.updateQuery(
                        seed,
                        current,
                        null,
                        20,
                        baseline.copy(anchorDecay = schedule, anchorHalfLifeTracks = timing),
                    ).query,
                    0f,
                )
            }
        }
    }

    @Test
    fun holdAtFullSeedAndFullMomentumUseTheSameFixedQuery() {
        val seed = floatArrayOf(0.6f, 0.8f)
        val current = floatArrayOf(0f, 1f)
        val heldSeed = RadioConfig(
            driftMode = DriftMode.SEED_INTERPOLATION,
            anchorStrength = 1f,
            anchorDecay = DecaySchedule.NONE,
        )
        val fullMomentum = heldSeed.copy(
            driftMode = DriftMode.MOMENTUM,
            momentumBeta = 1f,
        )

        assertArrayEquals(
            DriftEngine.updateQuery(seed, current, null, 8, heldSeed).query,
            DriftEngine.updateQuery(seed, current, seed, 8, fullMomentum).query,
            0f,
        )
    }

    private fun alpha(schedule: DecaySchedule, step: Int): Float =
        DriftEngine.computeAlpha(
            baseAlpha = 0.8f,
            step = step,
            decay = schedule,
            halfLifeTracks = 8f,
        )
}
