package com.powerampstartradio.ui

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class RadioConfigReachTest {
    @Test
    fun mmrReachDefaultsToMeasuredRecommendation() {
        val config = RadioConfig()

        assertEquals(
            RadioConfig.DEFAULT_MMR_CANDIDATE_POOL_FRACTION,
            config.effectiveMmrCandidatePoolFraction,
            0f,
        )
    }

    @Test
    fun chosenMmrReachUsesExplicitFraction() {
        val config = RadioConfig(mmrCandidatePoolFraction = 0.25f)

        assertEquals(0.25f, config.effectiveMmrCandidatePoolFraction, 0f)
    }

    @Test
    fun dppDefaultsToCertifiedFullDomainWithSeparateFixedChoice() {
        val defaults = RadioConfig()
        val fixed = RadioConfig(
            dppUsesCertifiedFullDomain = false,
            dppFixedCandidatePoolFraction = 0.1f,
        )

        assertTrue(defaults.dppUsesCertifiedFullDomain)
        assertFalse(fixed.dppUsesCertifiedFullDomain)
        assertEquals(0.1f, fixed.effectiveDppFixedCandidatePoolFraction, 0f)
        assertEquals(1_606, defaults.copy(selectionMode = SelectionMode.DPP)
            .resolveCandidatePoolSize(80_323))
        assertEquals(8_032, fixed.copy(selectionMode = SelectionMode.DPP)
            .resolveCandidatePoolSize(80_323))
    }

    @Test
    fun fixedDppDefaultsToMeasuredTightNeighborhood() {
        assertEquals(0.02f, RadioConfig.DEFAULT_DPP_FIXED_CANDIDATE_POOL_FRACTION, 0f)
        assertFalse(1f in NeighborhoodReachPolicy.DPP_FIXED_OPTIONS)
    }

    @Test
    fun reachStopsCollapseWhenTheyProduceTheSameCandidateDomain() {
        assertEquals(
            listOf(0.1f, 0.25f, 0.5f, 1f),
            NeighborhoodReachPolicy.distinctOptions(
                NeighborhoodReachPolicy.MMR_OPTIONS,
                librarySize = 1_000,
                numTracks = 30,
            ),
        )
        assertEquals(
            listOf(0.5f, 1f),
            NeighborhoodReachPolicy.distinctOptions(
                NeighborhoodReachPolicy.MMR_OPTIONS,
                librarySize = 200,
                numTracks = 30,
            ),
        )
        assertEquals(100, NeighborhoodReachPolicy.candidateCount(0.5f, 200, 30))
    }

    @Test
    fun savedReachRepresentsItsCollapsedDomainWithoutChangingFutureMeaning() {
        val compactLibraryOptions = NeighborhoodReachPolicy.distinctOptions(
            NeighborhoodReachPolicy.MMR_OPTIONS,
            librarySize = 1_000,
            numTracks = 30,
            preferredFraction = 0.02f,
        )

        assertEquals(listOf(0.02f, 0.25f, 0.5f, 1f), compactLibraryOptions)
        assertEquals(
            compactLibraryOptions.size,
            compactLibraryOptions.map {
                NeighborhoodReachPolicy.candidateCount(it, 1_000, 30)
            }.distinct().size,
        )

        val grownLibraryOptions = NeighborhoodReachPolicy.distinctOptions(
            NeighborhoodReachPolicy.MMR_OPTIONS,
            librarySize = 80_323,
            numTracks = 30,
            preferredFraction = 0.02f,
        )
        assertTrue(0.02f in grownLibraryOptions)
        assertEquals(
            1_606,
            NeighborhoodReachPolicy.candidateCount(0.02f, 80_323, 30),
        )
    }

    @Test
    fun onlyAnActualReachStopCanRepresentACollapsedDomain() {
        assertEquals(
            listOf(0.1f, 0.25f, 0.5f, 1f),
            NeighborhoodReachPolicy.distinctOptions(
                NeighborhoodReachPolicy.MMR_OPTIONS,
                librarySize = 1_000,
                numTracks = 30,
                preferredFraction = 0.019f,
            ),
        )
    }

    @Test
    fun mmrReachIsShownOnlyWhenItCanChangeSelectionOrFilteredRefill() {
        assertFalse(
            MmrControlPolicy.reachCanAffectOutput(
                relevanceWeight = 1f,
                artistLimitsEnabled = false,
                recommendationCount = 10,
                maxPerArtist = 8,
                minArtistSpacing = 3,
            ),
        )
        assertTrue(
            MmrControlPolicy.reachCanAffectOutput(
                relevanceWeight = 0.95f,
                artistLimitsEnabled = false,
                recommendationCount = 10,
                maxPerArtist = 8,
                minArtistSpacing = 3,
            ),
        )
        assertTrue(
            MmrControlPolicy.reachCanAffectOutput(
                relevanceWeight = 1f,
                artistLimitsEnabled = true,
                recommendationCount = 10,
                maxPerArtist = 8,
                minArtistSpacing = 3,
            ),
        )
    }

    @Test
    fun enabledArtistLimitsDoNotCreateReachWhenNeitherConstraintCanReject() {
        assertFalse(
            MmrControlPolicy.reachCanAffectOutput(
                relevanceWeight = 1f,
                artistLimitsEnabled = true,
                recommendationCount = 10,
                maxPerArtist = 10,
                minArtistSpacing = 0,
            ),
        )
        assertTrue(
            MmrControlPolicy.reachCanAffectOutput(
                relevanceWeight = 1f,
                artistLimitsEnabled = true,
                recommendationCount = 10,
                maxPerArtist = 9,
                minArtistSpacing = 0,
            ),
        )
        assertFalse(
            MmrControlPolicy.reachCanAffectOutput(
                relevanceWeight = 1f,
                artistLimitsEnabled = true,
                recommendationCount = 1,
                maxPerArtist = 1,
                minArtistSpacing = 20,
            ),
        )
        assertTrue(
            MmrControlPolicy.reachCanAffectOutput(
                relevanceWeight = 1f,
                artistLimitsEnabled = true,
                recommendationCount = 10,
                maxPerArtist = 10,
                minArtistSpacing = 1,
            ),
        )
    }

    @Test
    fun exactTinyIdentityDomainCollapsesReachToNoControl() {
        val config = RadioConfig(
            numTracks = 10,
            selectionMode = SelectionMode.MMR,
            diversityLambda = 0.4f,
        )

        assertEquals(
            listOf(1f),
            NeighborhoodReachPolicy.distinctOptions(
                options = NeighborhoodReachPolicy.MMR_OPTIONS,
                librarySize = 0,
                numTracks = config.numTracks,
            ),
        )
        assertFalse(
            MmrControlPolicy.shouldExposeReach(
                config = config,
                eligibleCandidateIdentityCount = 100,
            ),
        )
        assertTrue(
            MmrControlPolicy.shouldExposeReach(
                config = config,
                eligibleCandidateIdentityCount = 101,
            ),
        )
    }

    @Test
    fun fullSizedLibraryRetainsEveryMeasuredDppStopExceptRedundantFullDomain() {
        assertEquals(
            NeighborhoodReachPolicy.DPP_FIXED_OPTIONS,
            NeighborhoodReachPolicy.distinctOptions(
                NeighborhoodReachPolicy.DPP_FIXED_OPTIONS,
                librarySize = 80_323,
                numTracks = 30,
            ),
        )
    }

    @Test
    fun exactDppDomainControlRequiresAMeasuredProperSubset() {
        assertTrue(
            DppDomainControlPolicy.shouldExposeDomainControl(
                eligibleCandidateIdentityCount = null,
                numTracks = 10,
            ),
        )
        assertFalse(
            DppDomainControlPolicy.shouldExposeDomainControl(
                eligibleCandidateIdentityCount = 100,
                numTracks = 10,
            ),
        )
        assertTrue(
            DppDomainControlPolicy.shouldExposeDomainControl(
                eligibleCandidateIdentityCount = 101,
                numTracks = 10,
            ),
        )
    }

    @Test
    fun selectorKnobsExposeOnlyMeasuredStops() {
        assertEquals(
            listOf(0f, 0.25f, 0.5f, 1f, 2f, 3f, 4f),
            SelectionKnobPolicy.DPP_SEED_PULL_OPTIONS,
        )
        assertEquals(
            listOf(0.05f, 0.10f, 0.25f, 0.50f, 0.75f, 0.90f),
            SelectionKnobPolicy.GRAPH_STOP_OPTIONS,
        )
        assertEquals(
            listOf(0f, 0.25f, 0.4f, 0.5f, 0.55f, 0.6f, 0.8f, 0.9f, 1f),
            SelectionKnobPolicy.MMR_RELEVANCE_OPTIONS,
        )
        assertEquals(
            8,
            SelectionKnobPolicy.nearestIndex(
                SelectionKnobPolicy.MMR_RELEVANCE_OPTIONS,
                0.9704f,
            ),
        )
    }

    @Test
    fun currentConfigPreservesDppSeedPullAndShuffleSeed() {
        val config = RadioConfig(dppQualityExponent = 0f, shuffleSeed = -73L)

        assertEquals(0f, config.effectiveDppQualityExponent, 0f)
        assertEquals(-73L, config.effectiveShuffleSeed)
    }
}
