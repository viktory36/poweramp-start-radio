package com.powerampstartradio.debug

import com.powerampstartradio.ui.DecaySchedule
import com.powerampstartradio.ui.DriftControlPolicy
import com.powerampstartradio.ui.DriftMode
import com.powerampstartradio.ui.NeighborhoodReachPolicy
import com.powerampstartradio.ui.RadioConfig
import com.powerampstartradio.ui.SelectionKnobPolicy
import com.powerampstartradio.ui.SelectionMode
import com.powerampstartradio.ui.forSelectionRequest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class CurrentV2SelectionAcceptanceMatrixTest {
    @Test
    fun fullDomainBatchMmrIsOneCanonicalProductionCase() {
        val cases = CurrentV2SelectionAcceptanceMatrix.cases(
            CurrentV2SelectionAcceptanceMatrix.FULL_DOMAIN_BATCH_MMR_ID,
            numTracks = 30,
        )

        assertEquals(1, cases.size)
        val selectionCase = cases.single()
        assertEquals(
            "v2_mmr_full_domain_batch__relevance_0p5__reach_100pct",
            selectionCase.id,
        )
        assertEquals(SelectionMode.MMR, selectionCase.config.selectionMode)
        assertFalse(selectionCase.config.driftEnabled)
        assertEquals(
            RadioConfig.DEFAULT_MMR_RELEVANCE_WEIGHT,
            selectionCase.config.diversityLambda,
        )
        assertEquals(
            NeighborhoodReachPolicy.MMR_OPTIONS.last(),
            selectionCase.config.effectiveMmrCandidatePoolFraction,
        )
        assertEquals(
            selectionCase.config.forSelectionRequest(),
            selectionCase.config,
        )
    }

    @Test
    fun defaultsOnlyIsACompactCurrentModeSmokeSuite() {
        val cases = CurrentV2SelectionAcceptanceMatrix.cases(
            CurrentV2SelectionAcceptanceMatrix.DEFAULTS_ONLY_ID,
            numTracks = 30,
        )

        assertEquals(7, cases.size)
        assertEquals(
            setOf(
                SelectionMode.CLOSEST,
                SelectionMode.MMR,
                SelectionMode.DPP,
                SelectionMode.RANDOM_WALK,
                SelectionMode.UNIFORM_SHUFFLE,
            ),
            cases.map { it.config.selectionMode }.toSet(),
        )
        assertEquals(2, cases.count { it.config.driftEnabled })
        assertEquals(
            DriftMode.entries.toSet(),
            cases.filter { it.config.driftEnabled }.map { it.config.driftMode }.toSet(),
        )
    }

    @Test
    fun defaultsExtremesCoversEveryCurrentEndpointWithoutStaleMagicValues() {
        val cases = CurrentV2SelectionAcceptanceMatrix.cases(
            CurrentV2SelectionAcceptanceMatrix.DEFAULTS_EXTREMES_ID,
            numTracks = 30,
        )
        val configs = cases.map(FeatureSelectionCase::config)
        val plainMmr = configs.filter {
            it.selectionMode == SelectionMode.MMR && !it.driftEnabled
        }
        val fullDpp = configs.filter {
            it.selectionMode == SelectionMode.DPP && it.dppUsesCertifiedFullDomain
        }
        val fixedDpp = configs.filter {
            it.selectionMode == SelectionMode.DPP && !it.dppUsesCertifiedFullDomain
        }
        val graph = configs.filter { it.selectionMode == SelectionMode.RANDOM_WALK }
        val seedDrift = configs.filter {
            it.driftEnabled && it.driftMode == DriftMode.SEED_INTERPOLATION
        }
        val momentumDrift = configs.filter {
            it.driftEnabled && it.driftMode == DriftMode.MOMENTUM
        }

        assertEquals(30, cases.size)
        assertEquals(cases.size, cases.map(FeatureSelectionCase::id).distinct().size)
        assertTrue(cases.all { it.config.numTracks == 30 })
        assertTrue(cases.all { it.config == it.config.forSelectionRequest() })

        assertTrue(RadioConfig.DEFAULT_MMR_RELEVANCE_WEIGHT in plainMmr.map { it.diversityLambda })
        assertTrue(SelectionKnobPolicy.MMR_RELEVANCE_OPTIONS.first() in plainMmr.map { it.diversityLambda })
        assertTrue(SelectionKnobPolicy.MMR_RELEVANCE_OPTIONS.last() in plainMmr.map { it.diversityLambda })
        assertTrue(RadioConfig.DEFAULT_MMR_CANDIDATE_POOL_FRACTION in plainMmr.map { it.mmrCandidatePoolFraction })
        assertTrue(NeighborhoodReachPolicy.MMR_OPTIONS.first() in plainMmr.map { it.mmrCandidatePoolFraction })
        assertTrue(NeighborhoodReachPolicy.MMR_OPTIONS.last() in plainMmr.map { it.mmrCandidatePoolFraction })

        assertTrue(RadioConfig.DEFAULT_DPP_QUALITY_EXPONENT in fullDpp.map { it.dppQualityExponent })
        assertTrue(SelectionKnobPolicy.DPP_SEED_PULL_OPTIONS.first() in fullDpp.map { it.dppQualityExponent })
        assertTrue(SelectionKnobPolicy.DPP_SEED_PULL_OPTIONS.last() in fullDpp.map { it.dppQualityExponent })
        assertEquals(
            setOf(
                NeighborhoodReachPolicy.DPP_FIXED_OPTIONS.first(),
                NeighborhoodReachPolicy.DPP_FIXED_OPTIONS.last(),
            ),
            fixedDpp.map { it.dppFixedCandidatePoolFraction }.toSet(),
        )

        assertTrue(RadioConfig().walkRestartAlpha in graph.map { it.walkRestartAlpha })
        assertTrue(SelectionKnobPolicy.GRAPH_STOP_OPTIONS.first() in graph.map { it.walkRestartAlpha })
        assertTrue(SelectionKnobPolicy.GRAPH_STOP_OPTIONS.last() in graph.map { it.walkRestartAlpha })

        assertTrue(RadioConfig().anchorStrength in seedDrift.map { it.anchorStrength })
        assertTrue(DriftControlPolicy.SEED_PULL_OPTIONS.first() in seedDrift.map { it.anchorStrength })
        assertTrue(DriftControlPolicy.SEED_PULL_OPTIONS.last() in seedDrift.map { it.anchorStrength })
        assertEquals(DecaySchedule.entries.toSet(), seedDrift.map { it.anchorDecay }.toSet())
        assertTrue(DriftControlPolicy.FADE_TIMING_OPTIONS.first() in seedDrift.map { it.anchorHalfLifeTracks })
        assertTrue(DriftControlPolicy.FADE_TIMING_OPTIONS.last() in seedDrift.map { it.anchorHalfLifeTracks })
        assertTrue(SelectionKnobPolicy.MMR_RELEVANCE_OPTIONS.first() in seedDrift.map { it.diversityLambda })
        assertTrue(SelectionKnobPolicy.MMR_RELEVANCE_OPTIONS.last() in seedDrift.map { it.diversityLambda })
        assertTrue(NeighborhoodReachPolicy.MMR_OPTIONS.first() in seedDrift.map { it.mmrCandidatePoolFraction })
        assertTrue(NeighborhoodReachPolicy.MMR_OPTIONS.last() in seedDrift.map { it.mmrCandidatePoolFraction })

        assertTrue(RadioConfig.DEFAULT_MOMENTUM_BETA in momentumDrift.map { it.momentumBeta })
        assertTrue(DriftControlPolicy.MOMENTUM_OPTIONS.first() in momentumDrift.map { it.momentumBeta })
        assertTrue(DriftControlPolicy.MOMENTUM_OPTIONS.last() in momentumDrift.map { it.momentumBeta })
        assertTrue(SelectionKnobPolicy.MMR_RELEVANCE_OPTIONS.first() in momentumDrift.map { it.diversityLambda })
        assertTrue(SelectionKnobPolicy.MMR_RELEVANCE_OPTIONS.last() in momentumDrift.map { it.diversityLambda })
        assertTrue(NeighborhoodReachPolicy.MMR_OPTIONS.first() in momentumDrift.map { it.mmrCandidatePoolFraction })
        assertTrue(NeighborhoodReachPolicy.MMR_OPTIONS.last() in momentumDrift.map { it.mmrCandidatePoolFraction })
        assertFalse(configs.any { it.candidatePoolSize != 0 })
    }
}
