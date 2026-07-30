package com.powerampstartradio.ui

import com.powerampstartradio.similarity.algorithms.DriftEngine
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class RadioSettingsCodecTest {
    @Test
    fun emptyPreferencesProduceTheCanonicalFirstUseConfig() {
        val snapshot = RadioSettingsCodec.decode(emptyMap<String, Any?>())

        assertEquals(RadioConfig(), snapshot.storedConfig)
        assertEquals(RadioConfig(), snapshot.requestConfig)
    }

    @Test
    fun staleAndOffStepPreferencesSnapToTheVisibleControlSurface() {
        val snapshot = RadioSettingsCodec.decode(
            mapOf(
                "num_tracks" to 57,
                "library_added_range" to LibraryAddedRange.LAST_7_DAYS.name,
                "selection_mode" to SelectionMode.MMR.name,
                "drift_enabled" to true,
                "drift_mode" to DriftMode.MOMENTUM.name,
                "anchor_strength" to 0.71f,
                "walk_restart_alpha" to 0.31f,
                "anchor_decay" to DecaySchedule.LINEAR.name,
                "anchor_half_life_tracks" to 8f,
                "momentum_beta" to 0.89f,
                "diversity_lambda" to 0.86f,
                "mmr_candidate_pool_fraction" to 0.031f,
                "dpp_fixed_candidate_pool_fraction" to 0.42f,
                "dpp_uses_certified_full_domain" to false,
                "dpp_quality_exponent" to 1.6f,
                "shuffle_seed" to 0L,
                "artist_limits_enabled" to false,
                "max_per_artist" to 37,
                "min_artist_spacing" to 27,
            ),
        )
        val config = snapshot.storedConfig

        assertEquals(60, config.numTracks)
        assertEquals(LibraryAddedRange.ALL_DATES, config.libraryAddedRange)
        assertEquals(7, config.effectiveLibraryAddedDays)
        assertEquals(SelectionMode.MMR, config.selectionMode)
        assertTrue(config.driftEnabled)
        assertEquals(DriftMode.MOMENTUM, config.driftMode)
        assertEquals(0.75f, config.anchorStrength, 0f)
        assertEquals(0.25f, config.walkRestartAlpha, 0f)
        assertEquals(DecaySchedule.LINEAR, config.anchorDecay)
        assertEquals(RadioConfig.DEFAULT_ANCHOR_HALF_LIFE_TRACKS, config.anchorHalfLifeTracks, 0f)
        assertEquals(RadioConfig.DEFAULT_MOMENTUM_BETA, config.momentumBeta, 0f)
        assertEquals(0.9f, config.diversityLambda, 0f)
        assertEquals(0.02f, config.mmrCandidatePoolFraction, 0f)
        assertEquals(0.5f, config.dppFixedCandidatePoolFraction, 0f)
        assertFalse(config.dppUsesCertifiedFullDomain)
        assertEquals(2f, config.dppQualityExponent, 0f)
        assertEquals(RadioConfig.DEFAULT_SHUFFLE_SEED, config.shuffleSeed)
        assertFalse(config.artistLimitsEnabled)
        assertEquals(37, config.maxPerArtist)
        assertEquals(27, config.minArtistSpacing)
    }

    @Test
    fun invalidTypesEnumsAndNonFiniteValuesFallBackWithoutCreatingInvalidRequests() {
        val config = RadioSettingsCodec.decode(
            mapOf(
                "num_tracks" to Int.MIN_VALUE,
                "library_added_range" to "NOT_A_RANGE",
                "selection_mode" to "NOT_A_MODE",
                "drift_enabled" to "true",
                "drift_mode" to "NOT_A_DRIFT_MODE",
                "anchor_strength" to -100f,
                "walk_restart_alpha" to 100f,
                "anchor_decay" to "NOT_A_DECAY",
                "anchor_half_life_tracks" to 0f,
                "momentum_beta" to 2f,
                "diversity_lambda" to -100f,
                "mmr_candidate_pool_fraction" to -0.5f,
                "dpp_fixed_candidate_pool_fraction" to Float.NaN,
                "dpp_quality_exponent" to 100f,
                "shuffle_seed" to 0L,
                "max_per_artist" to 0,
                "min_artist_spacing" to 1_001,
            ),
        ).storedConfig

        assertEquals(RadioConfig().numTracks, config.numTracks)
        assertEquals(LibraryAddedRange.ALL_DATES, config.libraryAddedRange)
        assertNull(config.effectiveLibraryAddedDays)
        assertEquals(SelectionMode.MMR, config.selectionMode)
        assertFalse(config.driftEnabled)
        assertEquals(DriftMode.SEED_INTERPOLATION, config.driftMode)
        assertEquals(RadioConfig().anchorStrength, config.anchorStrength, 0f)
        assertEquals(RadioConfig().walkRestartAlpha, config.walkRestartAlpha, 0f)
        assertEquals(DecaySchedule.EXPONENTIAL, config.anchorDecay)
        assertEquals(RadioConfig.DEFAULT_ANCHOR_HALF_LIFE_TRACKS, config.anchorHalfLifeTracks, 0f)
        assertEquals(RadioConfig.DEFAULT_MOMENTUM_BETA, config.momentumBeta, 0f)
        assertEquals(RadioConfig().diversityLambda, config.diversityLambda, 0f)
        assertEquals(RadioConfig.DEFAULT_MMR_CANDIDATE_POOL_FRACTION, config.mmrCandidatePoolFraction, 0f)
        assertEquals(
            RadioConfig.DEFAULT_DPP_FIXED_CANDIDATE_POOL_FRACTION,
            config.dppFixedCandidatePoolFraction,
            0f,
        )
        assertEquals(RadioConfig.DEFAULT_DPP_QUALITY_EXPONENT, config.dppQualityExponent, 0f)
        assertEquals(RadioConfig.DEFAULT_SHUFFLE_SEED, config.shuffleSeed)
        assertEquals(RadioConfig().maxPerArtist, config.maxPerArtist)
        assertEquals(RadioConfig().minArtistSpacing, config.minArtistSpacing)
    }

    @Test
    fun exactAddedDaysTakePrecedenceOverLegacyPreset() {
        val config = RadioSettingsCodec.decode(
            mapOf(
                "library_added_days" to 17,
                "library_added_range" to LibraryAddedRange.LAST_365_DAYS.name,
            ),
        ).storedConfig

        assertEquals(LibraryAddedRange.ALL_DATES, config.libraryAddedRange)
        assertEquals(17, config.libraryAddedDays)
        assertEquals(17, config.effectiveLibraryAddedDays)
    }

    @Test
    fun invalidPresentExactDaysDoNotResurrectStaleLegacyPreset() {
        listOf(0, -1, MAX_LIBRARY_ADDED_DAYS + 1, "30").forEach { invalid ->
            val config = RadioSettingsCodec.decode(
                mapOf(
                    "library_added_days" to invalid,
                    "library_added_range" to LibraryAddedRange.LAST_30_DAYS.name,
                ),
            ).storedConfig

            assertNull(config.effectiveLibraryAddedDays)
        }
    }

    @Test
    fun longStepTimingStaysSavedButShortQueueRequestsUseALiveDropPoint() {
        val snapshot = RadioSettingsCodec.decode(
            mapOf(
                "num_tracks" to 10,
                "selection_mode" to SelectionMode.MMR.name,
                "drift_enabled" to true,
                "drift_mode" to DriftMode.SEED_INTERPOLATION.name,
                "anchor_decay" to DecaySchedule.STEP.name,
                "anchor_half_life_tracks" to 30f,
            ),
        )

        assertEquals(30f, snapshot.storedConfig.anchorHalfLifeTracks, 0f)
        assertEquals(7f, snapshot.requestConfig.anchorHalfLifeTracks, 0f)
        assertEquals(
            30f,
            snapshot.storedConfig.copy(numTracks = 100).forSelectionRequest()
                .anchorHalfLifeTracks,
            0f,
        )
    }

    @Test
    fun removedFixedSeedAliasMigratesToItsOutputEquivalentVisibleControl() {
        val config = RadioSettingsCodec.decode(
            mapOf(
                "drift_mode" to DriftMode.SEED_INTERPOLATION.name,
                "anchor_decay" to DecaySchedule.NONE.name,
                "anchor_strength" to 1f,
            ),
        ).storedConfig

        assertEquals(DriftMode.MOMENTUM, config.driftMode)
        assertEquals(1f, config.momentumBeta, 0f)
        assertEquals(DecaySchedule.NONE, config.anchorDecay)
        assertEquals(RadioConfig().anchorStrength, config.anchorStrength, 0f)

        val seed = floatArrayOf(0.6f, 0.8f)
        val current = floatArrayOf(0f, 1f)
        val removedAlias = RadioConfig(
            driftMode = DriftMode.SEED_INTERPOLATION,
            anchorStrength = 1f,
            anchorDecay = DecaySchedule.NONE,
        )
        assertArrayEquals(
            DriftEngine.updateQuery(seed, current, null, 8, removedAlias).query,
            DriftEngine.updateQuery(seed, current, null, 8, config).query,
            0f,
        )
    }

    @Test
    fun nonAliasSeedHoldPreferenceKeepsItsMeaning() {
        val config = RadioSettingsCodec.decode(
            mapOf(
                "drift_mode" to DriftMode.SEED_INTERPOLATION.name,
                "anchor_decay" to DecaySchedule.NONE.name,
                "anchor_strength" to 0.85f,
            ),
        ).storedConfig

        assertEquals(DriftMode.SEED_INTERPOLATION, config.driftMode)
        assertEquals(0.85f, config.anchorStrength, 0f)
        assertEquals(DecaySchedule.NONE, config.anchorDecay)
    }

    @Test
    fun appAndWidgetRequestProjectionsShareTheSameCanonicalSnapshot() {
        val snapshot = RadioSettingsCodec.decode(
            mapOf(
                "selection_mode" to SelectionMode.DPP.name,
                "drift_enabled" to true,
                "diversity_lambda" to 0.83f,
                "mmr_candidate_pool_fraction" to 0.049f,
                "shuffle_seed" to 73L,
            ),
        )

        assertTrue(snapshot.storedConfig.driftEnabled)
        assertFalse(snapshot.requestConfig.driftEnabled)
        assertEquals(
            snapshot.storedConfig.forSelectionRequest(),
            snapshot.requestConfig,
        )
        assertEquals(RadioConfig().diversityLambda, snapshot.requestConfig.diversityLambda, 0f)
        assertEquals(
            RadioConfig.DEFAULT_MMR_CANDIDATE_POOL_FRACTION,
            snapshot.requestConfig.mmrCandidatePoolFraction,
            0f,
        )
        assertEquals(RadioConfig.DEFAULT_SHUFFLE_SEED, snapshot.requestConfig.shuffleSeed)
    }

    @Test
    fun historicalSessionConfigsRemainExactUnlessExplicitlyDecodedAsPreferences() {
        val historical = RadioConfig(
            numTracks = 17,
            diversityLambda = 0.333f,
            mmrCandidatePoolFraction = 0.037f,
        )

        assertEquals(17, historical.numTracks)
        assertEquals(0.333f, historical.diversityLambda, 0f)
        assertEquals(0.037f, historical.mmrCandidatePoolFraction, 0f)
    }
}
