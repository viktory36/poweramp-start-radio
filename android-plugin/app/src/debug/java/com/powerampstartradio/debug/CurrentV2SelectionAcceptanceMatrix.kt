package com.powerampstartradio.debug

import com.powerampstartradio.ui.DecaySchedule
import com.powerampstartradio.ui.DriftControlPolicy
import com.powerampstartradio.ui.DriftMode
import com.powerampstartradio.ui.NeighborhoodReachPolicy
import com.powerampstartradio.ui.RadioConfig
import com.powerampstartradio.ui.SelectionKnobPolicy
import com.powerampstartradio.ui.SelectionMode
import com.powerampstartradio.ui.forSelectionRequest
import java.math.BigDecimal

internal data class FeatureSelectionCase(
    val id: String,
    val config: RadioConfig,
)

/** Debug-only installed-APK matrix for the current measured V2 selector surface. */
internal object CurrentV2SelectionAcceptanceMatrix {
    const val DEFAULTS_EXTREMES_ID = "current_v2_defaults_extremes"
    const val DEFAULTS_ONLY_ID = "current_v2_defaults"
    const val FULL_DOMAIN_BATCH_MMR_ID = "current_v2_mmr_full_domain_batch"
    const val SELECTOR_REPAIRS_ID = "current_v2_selector_repairs"

    val supportedIds: Set<String> = setOf(
        DEFAULTS_EXTREMES_ID,
        DEFAULTS_ONLY_ID,
        FULL_DOMAIN_BATCH_MMR_ID,
        SELECTOR_REPAIRS_ID,
    )

    fun cases(matrixId: String, numTracks: Int): List<FeatureSelectionCase> {
        require(numTracks > 0) { "numTracks must be positive" }
        require(matrixId in supportedIds) {
            "Unknown selection matrix '$matrixId'; expected ${supportedIds.sorted()}"
        }

        val cases = when (matrixId) {
            DEFAULTS_ONLY_ID -> defaultCases(numTracks)
            FULL_DOMAIN_BATCH_MMR_ID -> listOf(fullDomainBatchMmrCase(numTracks))
            SELECTOR_REPAIRS_ID -> selectorRepairCases(numTracks)
            else -> defaultCases(numTracks) + endpointCases(numTracks)
        }
        require(cases.map(FeatureSelectionCase::id).distinct().size == cases.size) {
            "Selection acceptance case IDs must be unique"
        }
        cases.forEach { selectionCase ->
            require(selectionCase.config == selectionCase.config.forSelectionRequest()) {
                "${selectionCase.id} is not a canonical production selection request"
            }
        }
        return cases
    }

    private fun fullDomainBatchMmrCase(numTracks: Int): FeatureSelectionCase {
        val config = RadioConfig(numTracks = numTracks).copy(
            selectionMode = SelectionMode.MMR,
            driftEnabled = false,
            diversityLambda = RadioConfig.DEFAULT_MMR_RELEVANCE_WEIGHT,
            mmrCandidatePoolFraction = NeighborhoodReachPolicy.MMR_OPTIONS.last(),
        ).forSelectionRequest()
        return FeatureSelectionCase(
            id = "v2_mmr_full_domain_batch__relevance_${scalarId(config.diversityLambda)}" +
                "__reach_${percentId(config.effectiveMmrCandidatePoolFraction)}",
            config = config,
        )
    }

    /** Small installed-APK matrix for the three selector repairs found by the full review. */
    private fun selectorRepairCases(numTracks: Int): List<FeatureSelectionCase> {
        val defaults = RadioConfig(numTracks = numTracks)
        val closest = defaults.copy(selectionMode = SelectionMode.CLOSEST)
            .forSelectionRequest()
        val exactClosestMmr = defaults.copy(
            selectionMode = SelectionMode.MMR,
            driftEnabled = false,
            diversityLambda = SelectionKnobPolicy.MMR_RELEVANCE_OPTIONS.last(),
        ).forSelectionRequest()
        val dpp = defaults.copy(selectionMode = SelectionMode.DPP)
            .forSelectionRequest()
        val graph = defaults.copy(selectionMode = SelectionMode.RANDOM_WALK)
            .forSelectionRequest()
        return listOf(
            FeatureSelectionCase(
                id = "v2_closest_default",
                config = closest,
            ),
            FeatureSelectionCase(
                id = "v2_mmr_relevance_max__${scalarId(exactClosestMmr.diversityLambda)}",
                config = exactClosestMmr,
            ),
            FeatureSelectionCase(
                id = "v2_dpp_full_default__seed_pull_${scalarId(dpp.effectiveDppQualityExponent)}",
                config = dpp,
            ),
            FeatureSelectionCase(
                id = "v2_graph_default__stop_${percentId(graph.walkRestartAlpha)}",
                config = graph,
            ),
        )
    }

    private fun defaultCases(numTracks: Int): List<FeatureSelectionCase> {
        val defaults = RadioConfig(numTracks = numTracks)
        val mmr = defaults.copy(
            selectionMode = SelectionMode.MMR,
            driftEnabled = false,
        ).forSelectionRequest()
        val dpp = defaults.copy(selectionMode = SelectionMode.DPP).forSelectionRequest()
        val graph = defaults.copy(selectionMode = SelectionMode.RANDOM_WALK)
            .forSelectionRequest()
        val seedDrift = defaults.copy(
            selectionMode = SelectionMode.MMR,
            driftEnabled = true,
            driftMode = DriftMode.SEED_INTERPOLATION,
        ).forSelectionRequest()
        val momentumDrift = defaults.copy(
            selectionMode = SelectionMode.MMR,
            driftEnabled = true,
            driftMode = DriftMode.MOMENTUM,
        ).forSelectionRequest()

        return listOf(
            FeatureSelectionCase(
                id = "v2_closest_default",
                config = defaults.copy(selectionMode = SelectionMode.CLOSEST)
                    .forSelectionRequest(),
            ),
            FeatureSelectionCase(
                id = "v2_mmr_default__relevance_${scalarId(mmr.diversityLambda)}" +
                    "__reach_${percentId(mmr.effectiveMmrCandidatePoolFraction)}",
                config = mmr,
            ),
            FeatureSelectionCase(
                id = "v2_dpp_full_default__seed_pull_${scalarId(dpp.effectiveDppQualityExponent)}",
                config = dpp,
            ),
            FeatureSelectionCase(
                id = "v2_graph_default__stop_${percentId(graph.walkRestartAlpha)}",
                config = graph,
            ),
            FeatureSelectionCase(
                id = "v2_drift_seed_default__pull_${percentId(seedDrift.anchorStrength)}" +
                    "__exp_half_life_${scalarId(seedDrift.effectiveAnchorHalfLifeTracks)}",
                config = seedDrift,
            ),
            FeatureSelectionCase(
                id = "v2_drift_momentum_default__carry_${percentId(momentumDrift.momentumBeta)}",
                config = momentumDrift,
            ),
            FeatureSelectionCase(
                id = "v2_uniform_shuffle_default",
                config = defaults.copy(selectionMode = SelectionMode.UNIFORM_SHUFFLE)
                    .forSelectionRequest(),
            ),
        )
    }

    private fun endpointCases(numTracks: Int): List<FeatureSelectionCase> {
        val defaults = RadioConfig(numTracks = numTracks)
        val mmr = defaults.copy(
            selectionMode = SelectionMode.MMR,
            driftEnabled = false,
        )
        val dpp = defaults.copy(selectionMode = SelectionMode.DPP)
        val graph = defaults.copy(selectionMode = SelectionMode.RANDOM_WALK)
        val seedDrift = defaults.copy(
            selectionMode = SelectionMode.MMR,
            driftEnabled = true,
            driftMode = DriftMode.SEED_INTERPOLATION,
        )
        val momentumDrift = defaults.copy(
            selectionMode = SelectionMode.MMR,
            driftEnabled = true,
            driftMode = DriftMode.MOMENTUM,
        )

        val minMmrRelevance = SelectionKnobPolicy.MMR_RELEVANCE_OPTIONS.first()
        val maxMmrRelevance = SelectionKnobPolicy.MMR_RELEVANCE_OPTIONS.last()
        val minMmrReach = NeighborhoodReachPolicy.MMR_OPTIONS.first()
        val maxMmrReach = NeighborhoodReachPolicy.MMR_OPTIONS.last()
        val minDppSeedPull = SelectionKnobPolicy.DPP_SEED_PULL_OPTIONS.first()
        val maxDppSeedPull = SelectionKnobPolicy.DPP_SEED_PULL_OPTIONS.last()
        val minDppFixedReach = NeighborhoodReachPolicy.DPP_FIXED_OPTIONS.first()
        val maxDppFixedReach = NeighborhoodReachPolicy.DPP_FIXED_OPTIONS.last()
        val minGraphStop = SelectionKnobPolicy.GRAPH_STOP_OPTIONS.first()
        val maxGraphStop = SelectionKnobPolicy.GRAPH_STOP_OPTIONS.last()
        val minSeedPull = DriftControlPolicy.seedPullOptions(defaults.anchorDecay).first()
        val maxSeedPull = DriftControlPolicy.seedPullOptions(defaults.anchorDecay).last()
        val minFadeTiming = DriftControlPolicy.fadeTimingOptions(
            decay = defaults.anchorDecay,
            recommendationCount = numTracks,
        ).first()
        val maxFadeTiming = DriftControlPolicy.fadeTimingOptions(
            decay = defaults.anchorDecay,
            recommendationCount = numTracks,
        ).last()
        val minMomentum = DriftControlPolicy.MOMENTUM_OPTIONS.first()
        val maxMomentum = DriftControlPolicy.MOMENTUM_OPTIONS.last()
        val stepTiming = DriftControlPolicy.canonicalFadeTiming(
            decay = DecaySchedule.STEP,
            timingTracks = defaults.anchorHalfLifeTracks,
            recommendationCount = numTracks,
        )

        return listOf(
            FeatureSelectionCase(
                id = "v2_mmr_relevance_min__${scalarId(minMmrRelevance)}",
                config = mmr.copy(diversityLambda = minMmrRelevance).forSelectionRequest(),
            ),
            FeatureSelectionCase(
                id = "v2_mmr_relevance_max__${scalarId(maxMmrRelevance)}",
                config = mmr.copy(diversityLambda = maxMmrRelevance).forSelectionRequest(),
            ),
            FeatureSelectionCase(
                id = "v2_mmr_reach_min__${percentId(minMmrReach)}",
                config = mmr.copy(mmrCandidatePoolFraction = minMmrReach)
                    .forSelectionRequest(),
            ),
            FeatureSelectionCase(
                id = "v2_mmr_reach_max__${percentId(maxMmrReach)}",
                config = mmr.copy(mmrCandidatePoolFraction = maxMmrReach)
                    .forSelectionRequest(),
            ),
            FeatureSelectionCase(
                id = "v2_dpp_full_seed_pull_min__${scalarId(minDppSeedPull)}",
                config = dpp.copy(
                    dppUsesCertifiedFullDomain = true,
                    dppQualityExponent = minDppSeedPull,
                ).forSelectionRequest(),
            ),
            FeatureSelectionCase(
                id = "v2_dpp_full_seed_pull_max__${scalarId(maxDppSeedPull)}",
                config = dpp.copy(
                    dppUsesCertifiedFullDomain = true,
                    dppQualityExponent = maxDppSeedPull,
                ).forSelectionRequest(),
            ),
            FeatureSelectionCase(
                id = "v2_dpp_fixed_reach_min__${percentId(minDppFixedReach)}",
                config = dpp.copy(
                    dppUsesCertifiedFullDomain = false,
                    dppFixedCandidatePoolFraction = minDppFixedReach,
                ).forSelectionRequest(),
            ),
            FeatureSelectionCase(
                id = "v2_dpp_fixed_reach_max__${percentId(maxDppFixedReach)}",
                config = dpp.copy(
                    dppUsesCertifiedFullDomain = false,
                    dppFixedCandidatePoolFraction = maxDppFixedReach,
                ).forSelectionRequest(),
            ),
            FeatureSelectionCase(
                id = "v2_graph_stop_min__${percentId(minGraphStop)}",
                config = graph.copy(walkRestartAlpha = minGraphStop).forSelectionRequest(),
            ),
            FeatureSelectionCase(
                id = "v2_graph_stop_max__${percentId(maxGraphStop)}",
                config = graph.copy(walkRestartAlpha = maxGraphStop).forSelectionRequest(),
            ),
            FeatureSelectionCase(
                id = "v2_drift_seed_pull_min__${percentId(minSeedPull)}",
                config = seedDrift.copy(anchorStrength = minSeedPull).forSelectionRequest(),
            ),
            FeatureSelectionCase(
                id = "v2_drift_seed_pull_max__${percentId(maxSeedPull)}",
                config = seedDrift.copy(anchorStrength = maxSeedPull).forSelectionRequest(),
            ),
            FeatureSelectionCase(
                id = "v2_drift_seed_fade_hold__pull_${percentId(defaults.anchorStrength)}",
                config = seedDrift.copy(anchorDecay = DecaySchedule.NONE)
                    .forSelectionRequest(),
            ),
            FeatureSelectionCase(
                id = "v2_drift_seed_fade_linear__half_strength_" +
                    scalarId(defaults.effectiveAnchorHalfLifeTracks),
                config = seedDrift.copy(anchorDecay = DecaySchedule.LINEAR)
                    .forSelectionRequest(),
            ),
            FeatureSelectionCase(
                id = "v2_drift_seed_fade_step__drop_after_" +
                    "${DriftControlPolicy.stepDropAfterPickCount(stepTiming)}_picks",
                config = seedDrift.copy(
                    anchorDecay = DecaySchedule.STEP,
                    anchorHalfLifeTracks = stepTiming,
                ).forSelectionRequest(),
            ),
            FeatureSelectionCase(
                id = "v2_drift_seed_exp_timing_min__half_life_${scalarId(minFadeTiming)}",
                config = seedDrift.copy(anchorHalfLifeTracks = minFadeTiming)
                    .forSelectionRequest(),
            ),
            FeatureSelectionCase(
                id = "v2_drift_seed_exp_timing_max__half_life_${scalarId(maxFadeTiming)}",
                config = seedDrift.copy(anchorHalfLifeTracks = maxFadeTiming)
                    .forSelectionRequest(),
            ),
            FeatureSelectionCase(
                id = "v2_drift_momentum_carry_min__${percentId(minMomentum)}",
                config = momentumDrift.copy(momentumBeta = minMomentum)
                    .forSelectionRequest(),
            ),
            FeatureSelectionCase(
                id = "v2_drift_momentum_carry_max__${percentId(maxMomentum)}",
                config = momentumDrift.copy(momentumBeta = maxMomentum)
                    .forSelectionRequest(),
            ),
            FeatureSelectionCase(
                id = "v2_drift_seed_query_near_corner__relevance_" +
                    "${scalarId(maxMmrRelevance)}__reach_${percentId(minMmrReach)}",
                config = seedDrift.copy(
                    diversityLambda = maxMmrRelevance,
                    mmrCandidatePoolFraction = minMmrReach,
                ).forSelectionRequest(),
            ),
            FeatureSelectionCase(
                id = "v2_drift_seed_query_far_corner__relevance_" +
                    "${scalarId(minMmrRelevance)}__reach_${percentId(maxMmrReach)}",
                config = seedDrift.copy(
                    diversityLambda = minMmrRelevance,
                    mmrCandidatePoolFraction = maxMmrReach,
                ).forSelectionRequest(),
            ),
            FeatureSelectionCase(
                id = "v2_drift_momentum_query_near_corner__relevance_" +
                    "${scalarId(maxMmrRelevance)}__reach_${percentId(minMmrReach)}",
                config = momentumDrift.copy(
                    diversityLambda = maxMmrRelevance,
                    mmrCandidatePoolFraction = minMmrReach,
                ).forSelectionRequest(),
            ),
            FeatureSelectionCase(
                id = "v2_drift_momentum_query_far_corner__relevance_" +
                    "${scalarId(minMmrRelevance)}__reach_${percentId(maxMmrReach)}",
                config = momentumDrift.copy(
                    diversityLambda = minMmrRelevance,
                    mmrCandidatePoolFraction = maxMmrReach,
                ).forSelectionRequest(),
            ),
        )
    }

    private fun percentId(fraction: Float): String = scalarId(fraction * 100f) + "pct"

    private fun scalarId(value: Float): String = BigDecimal(value.toString())
        .stripTrailingZeros()
        .toPlainString()
        .replace("-", "neg")
        .replace(".", "p")
}
