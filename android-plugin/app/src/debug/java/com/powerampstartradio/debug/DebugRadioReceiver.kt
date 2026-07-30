package com.powerampstartradio.debug

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.util.Log
import com.powerampstartradio.services.RadioService
import com.powerampstartradio.ui.DriftMode
import com.powerampstartradio.ui.RadioConfig
import com.powerampstartradio.ui.QueueOrigin
import com.powerampstartradio.ui.SelectionMode
import com.powerampstartradio.ui.forSelectionRequest

/**
 * Debug-only receiver to trigger radio from ADB for automated testing.
 *
 * Usage:
 *   adb shell am broadcast -a com.powerampstartradio.DEBUG_START_RADIO \
 *     -n com.powerampstartradio.v2/.debug.DebugRadioReceiver \
 *     --es selection_mode MMR --ef diversity_lambda 0.4 --ei num_tracks 30
 */
class DebugRadioReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        val config = RadioConfig(
            numTracks = intent.getIntExtra("num_tracks", 30),
            selectionMode = try {
                SelectionMode.valueOf(intent.getStringExtra("selection_mode") ?: "MMR")
            } catch (_: Exception) { SelectionMode.MMR },
            driftEnabled = intent.getBooleanExtra("drift_enabled", false),
            driftMode = try {
                DriftMode.valueOf(intent.getStringExtra("drift_mode") ?: "SEED_INTERPOLATION")
            } catch (_: Exception) { DriftMode.SEED_INTERPOLATION },
            anchorStrength = intent.getFloatExtra("anchor_strength", 0.5f),
            momentumBeta = intent.getFloatExtra(
                "momentum_beta",
                RadioConfig.DEFAULT_MOMENTUM_BETA,
            ),
            walkRestartAlpha = intent.getFloatExtra("walk_restart_alpha", 0.5f),
            diversityLambda = intent.getFloatExtra(
                "diversity_lambda",
                RadioConfig.DEFAULT_MMR_RELEVANCE_WEIGHT,
            ),
            mmrCandidatePoolFraction = intent.getFloatExtra(
                "mmr_candidate_pool_fraction",
                RadioConfig.DEFAULT_MMR_CANDIDATE_POOL_FRACTION,
            ),
            dppUsesCertifiedFullDomain = intent.getBooleanExtra(
                "dpp_uses_certified_full_domain",
                true,
            ),
            dppFixedCandidatePoolFraction = intent.getFloatExtra(
                "dpp_fixed_candidate_pool_fraction",
                RadioConfig.DEFAULT_DPP_FIXED_CANDIDATE_POOL_FRACTION,
            ),
            dppQualityExponent = intent.getFloatExtra(
                "dpp_quality_exponent",
                RadioConfig.DEFAULT_DPP_QUALITY_EXPONENT,
            ),
            maxPerArtist = intent.getIntExtra("max_per_artist", 8),
            minArtistSpacing = intent.getIntExtra("min_artist_spacing", 3),
        ).forSelectionRequest()
        Log.i("DebugRadio", "Triggering: ${config.selectionMode} lambda=${config.diversityLambda} " +
            "alpha=${config.walkRestartAlpha} drift=${config.driftEnabled} anchor=${config.anchorStrength}")
        RadioService.startRadio(context, config, origin = QueueOrigin.DEBUG_RADIO)
    }
}
