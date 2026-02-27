package com.powerampstartradio.debug

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.util.Log
import com.powerampstartradio.services.RadioService
import com.powerampstartradio.ui.DriftMode
import com.powerampstartradio.ui.RadioConfig
import com.powerampstartradio.ui.SelectionMode

/**
 * Debug-only receiver to trigger radio from ADB for automated testing.
 *
 * Usage:
 *   adb shell am broadcast -a com.powerampstartradio.DEBUG_START_RADIO \
 *     -n com.powerampstartradio/.debug.DebugRadioReceiver \
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
            momentumBeta = intent.getFloatExtra("momentum_beta", 0.7f),
            pageRankAlpha = intent.getFloatExtra("pagerank_alpha", 0.5f),
            diversityLambda = intent.getFloatExtra("diversity_lambda", 0.4f),
            maxPerArtist = intent.getIntExtra("max_per_artist", 8),
            minArtistSpacing = intent.getIntExtra("min_artist_spacing", 3),
        )
        Log.i("DebugRadio", "Triggering: ${config.selectionMode} lambda=${config.diversityLambda} " +
            "alpha=${config.pageRankAlpha} drift=${config.driftEnabled} anchor=${config.anchorStrength}")
        RadioService.startRadio(context, config)
    }
}
