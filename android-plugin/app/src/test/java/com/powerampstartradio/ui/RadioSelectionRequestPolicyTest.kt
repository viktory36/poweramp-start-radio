package com.powerampstartradio.ui

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class RadioSelectionRequestPolicyTest {
    private val noisy = RadioConfig(
        configSchemaVersion = 1,
        candidatePoolSize = 731,
        mmrCandidatePoolFraction = 0.25f,
        dppFixedCandidatePoolFraction = 0.5f,
        dppUsesCertifiedFullDomain = false,
        driftEnabled = true,
        driftMode = DriftMode.MOMENTUM,
        anchorStrength = 0.75f,
        anchorDecay = DecaySchedule.LINEAR,
        anchorHalfLifeTracks = 14f,
        momentumBeta = 0.95f,
        walkRestartAlpha = 0.25f,
        diversityLambda = 0.8f,
        dppQualityExponent = 4f,
        shuffleSeed = 73L,
    )

    @Test
    fun eachModeRetainsOnlyInputsWhichCanAffectItsQueue() {
        val defaults = RadioConfig()

        val closest = noisy.copy(selectionMode = SelectionMode.CLOSEST).forSelectionRequest()
        assertEquals(defaults.copy(selectionMode = SelectionMode.CLOSEST), closest)

        val graph = noisy.copy(selectionMode = SelectionMode.RANDOM_WALK).forSelectionRequest()
        assertEquals(defaults.copy(
            selectionMode = SelectionMode.RANDOM_WALK,
            walkRestartAlpha = noisy.walkRestartAlpha,
        ), graph)

        val shuffle = noisy.copy(selectionMode = SelectionMode.UNIFORM_SHUFFLE)
            .forSelectionRequest()
        assertEquals(defaults.copy(
            selectionMode = SelectionMode.UNIFORM_SHUFFLE,
            shuffleSeed = noisy.shuffleSeed,
        ), shuffle)

        val dpp = noisy.copy(selectionMode = SelectionMode.DPP).forSelectionRequest()
        assertEquals(defaults.copy(
            selectionMode = SelectionMode.DPP,
            dppFixedCandidatePoolFraction = noisy.dppFixedCandidatePoolFraction,
            dppUsesCertifiedFullDomain = false,
            dppQualityExponent = noisy.dppQualityExponent,
        ), dpp)

        val mmr = noisy.copy(selectionMode = SelectionMode.MMR).forSelectionRequest()
        assertEquals(defaults.copy(
            selectionMode = SelectionMode.MMR,
            mmrCandidatePoolFraction = noisy.mmrCandidatePoolFraction,
            driftEnabled = true,
            driftMode = DriftMode.MOMENTUM,
            momentumBeta = noisy.momentumBeta,
            diversityLambda = noisy.diversityLambda,
        ), mmr)
    }

    @Test
    fun certifiedDppDoesNotClaimItsOutputNeutralProofPrefixAsAControl() {
        val request = noisy.copy(
            selectionMode = SelectionMode.DPP,
            dppUsesCertifiedFullDomain = true,
        ).forSelectionRequest()

        assertTrue(request.dppUsesCertifiedFullDomain)
        assertEquals(
            RadioConfig.DEFAULT_DPP_FIXED_CANDIDATE_POOL_FRACTION,
            request.dppFixedCandidatePoolFraction,
            0f,
        )
    }

    @Test
    fun collapsedFixedDppDomainCanonicalizesOnlyTheEffectiveRequest() {
        val stored = noisy.copy(
            selectionMode = SelectionMode.DPP,
            dppUsesCertifiedFullDomain = false,
            dppFixedCandidatePoolFraction = 0.5f,
            numTracks = 10,
        )

        val collapsedRequest = stored.forSelectionRequest(eligibleCandidateIdentityCount = 100)
        val properSubsetRequest = stored.forSelectionRequest(eligibleCandidateIdentityCount = 101)

        assertFalse(stored.dppUsesCertifiedFullDomain)
        assertTrue(collapsedRequest.dppUsesCertifiedFullDomain)
        assertEquals(
            RadioConfig.DEFAULT_DPP_FIXED_CANDIDATE_POOL_FRACTION,
            collapsedRequest.dppFixedCandidatePoolFraction,
            0f,
        )
        assertFalse(properSubsetRequest.dppUsesCertifiedFullDomain)
        assertEquals(0.5f, properSubsetRequest.dppFixedCandidatePoolFraction, 0f)
    }

    @Test
    fun disabledDriftAndArtistLimitsDoNotCarryDormantKnobsIntoEvidence() {
        val request = noisy.copy(
            selectionMode = SelectionMode.MMR,
            driftEnabled = false,
            artistLimitsEnabled = false,
            maxPerArtist = 77,
            minArtistSpacing = 44,
        ).forSelectionRequest()

        assertFalse(request.driftEnabled)
        assertEquals(DriftMode.SEED_INTERPOLATION, request.driftMode)
        assertEquals(RadioConfig().anchorStrength, request.anchorStrength, 0f)
        assertEquals(RadioConfig.DEFAULT_MOMENTUM_BETA, request.momentumBeta, 0f)
        assertFalse(request.artistLimitsEnabled)
        assertEquals(RadioConfig().maxPerArtist, request.maxPerArtist)
        assertEquals(RadioConfig().minArtistSpacing, request.minArtistSpacing)
    }

    @Test
    fun seedInterpolationRetainsFadeOnlyWhenTheScheduleUsesIt() {
        val noDecay = noisy.copy(
            selectionMode = SelectionMode.MMR,
            driftMode = DriftMode.SEED_INTERPOLATION,
            anchorDecay = DecaySchedule.NONE,
        ).forSelectionRequest()
        assertEquals(RadioConfig.DEFAULT_ANCHOR_HALF_LIFE_TRACKS, noDecay.anchorHalfLifeTracks, 0f)

        val exponential = noisy.copy(
            selectionMode = SelectionMode.MMR,
            driftMode = DriftMode.SEED_INTERPOLATION,
            anchorDecay = DecaySchedule.EXPONENTIAL,
        ).forSelectionRequest()
        assertEquals(noisy.anchorHalfLifeTracks, exponential.anchorHalfLifeTracks, 0f)
        assertEquals(RadioConfig.DEFAULT_MOMENTUM_BETA, exponential.momentumBeta, 0f)
    }

    @Test
    fun stepRequestCannotRecordADropAfterItsLastUsefulQuery() {
        val request = noisy.copy(
            numTracks = 10,
            selectionMode = SelectionMode.MMR,
            driftMode = DriftMode.SEED_INTERPOLATION,
            anchorDecay = DecaySchedule.STEP,
            anchorHalfLifeTracks = 30f,
        ).forSelectionRequest()

        assertEquals(DecaySchedule.STEP, request.anchorDecay)
        assertEquals(7f, request.anchorHalfLifeTracks, 0f)
        assertTrue(
            DriftControlPolicy.stepTimingCanAffectQueue(
                request.anchorHalfLifeTracks,
                request.numTracks,
            ),
        )
    }
}
