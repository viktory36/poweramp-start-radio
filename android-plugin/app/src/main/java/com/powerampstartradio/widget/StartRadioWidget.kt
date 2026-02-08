package com.powerampstartradio.widget

import android.content.Context
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.glance.GlanceId
import androidx.glance.GlanceModifier
import androidx.glance.action.ActionParameters
import androidx.glance.action.clickable
import androidx.glance.appwidget.GlanceAppWidget
import androidx.glance.appwidget.action.ActionCallback
import androidx.glance.appwidget.action.actionRunCallback
import androidx.glance.appwidget.cornerRadius
import androidx.glance.appwidget.provideContent
import androidx.glance.background
import androidx.glance.layout.Alignment
import androidx.glance.layout.Column
import androidx.glance.layout.Row
import androidx.glance.layout.Spacer
import androidx.glance.layout.fillMaxSize
import androidx.glance.layout.padding
import androidx.glance.layout.width
import androidx.glance.color.ColorProvider
import androidx.glance.text.FontWeight
import androidx.glance.text.Text
import androidx.glance.text.TextStyle
import com.powerampstartradio.poweramp.PowerampReceiver
import com.powerampstartradio.services.RadioService
import com.powerampstartradio.ui.DecaySchedule
import com.powerampstartradio.ui.DriftMode
import com.powerampstartradio.ui.RadioConfig
import com.powerampstartradio.ui.SelectionMode
import java.io.File

class StartRadioWidget : GlanceAppWidget() {

    override suspend fun provideGlance(context: Context, id: GlanceId) {
        provideContent {
            val track = PowerampReceiver.currentTrack

            Column(
                modifier = GlanceModifier
                    .fillMaxSize()
                    .cornerRadius(16.dp)
                    .background(Color(0xFF1A1B2E))
                    .padding(14.dp)
                    .clickable(actionRunCallback<StartRadioAction>()),
                verticalAlignment = Alignment.CenterVertically,
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        text = "▶",
                        style = TextStyle(
                            color = ColorProvider(Color(0xFF89B4FA), Color(0xFF89B4FA)),
                            fontSize = 18.sp,
                            fontWeight = FontWeight.Bold
                        )
                    )
                    Spacer(modifier = GlanceModifier.width(8.dp))
                    Text(
                        text = "Start Radio",
                        style = TextStyle(
                            color = ColorProvider(Color.White, Color.White),
                            fontSize = 15.sp,
                            fontWeight = FontWeight.Medium
                        )
                    )
                }
                if (track != null) {
                    val title = track.title
                    val artist = track.artist
                    val album = track.album
                    if (!title.isNullOrEmpty()) {
                        Text(
                            text = title,
                            style = TextStyle(
                                color = ColorProvider(Color(0xFFCDD6F4), Color(0xFFCDD6F4)),
                                fontSize = 12.sp,
                                fontWeight = FontWeight.Medium
                            ),
                            maxLines = 1
                        )
                    }
                    val subtitle = listOfNotNull(artist, album)
                        .joinToString(" · ")
                    if (subtitle.isNotEmpty()) {
                        Text(
                            text = subtitle,
                            style = TextStyle(
                                color = ColorProvider(Color(0xFF9399B2), Color(0xFF9399B2)),
                                fontSize = 11.sp
                            ),
                            maxLines = 1
                        )
                    }
                }
            }
        }
    }
}

class StartRadioAction : ActionCallback {
    override suspend fun onAction(
        context: Context,
        glanceId: GlanceId,
        parameters: ActionParameters
    ) {
        val dbFile = File(context.filesDir, "embeddings.db")
        if (!dbFile.exists()) return

        val prefs = context.getSharedPreferences("settings", Context.MODE_PRIVATE)

        val config = RadioConfig(
            numTracks = prefs.getInt("num_tracks", RadioService.DEFAULT_NUM_TRACKS),
            candidatePoolSize = prefs.getInt("candidate_pool_size", 200),
            selectionMode = try {
                SelectionMode.valueOf(prefs.getString("selection_mode", SelectionMode.MMR.name)!!)
            } catch (e: IllegalArgumentException) { SelectionMode.MMR },
            driftEnabled = prefs.getBoolean("drift_enabled", true),
            driftMode = try {
                DriftMode.valueOf(prefs.getString("drift_mode", DriftMode.SEED_INTERPOLATION.name)!!)
            } catch (e: IllegalArgumentException) { DriftMode.SEED_INTERPOLATION },
            anchorStrength = prefs.getFloat("anchor_strength", 0.5f),
            anchorDecay = try {
                DecaySchedule.valueOf(prefs.getString("anchor_decay", DecaySchedule.EXPONENTIAL.name)!!)
            } catch (e: IllegalArgumentException) { DecaySchedule.EXPONENTIAL },
            momentumBeta = prefs.getFloat("momentum_beta", 0.7f),
            diversityLambda = prefs.getFloat("diversity_lambda", 0.4f),
            temperature = prefs.getFloat("temperature", 0.05f),
            maxPerArtist = prefs.getInt("max_per_artist", 8),
            minArtistSpacing = prefs.getInt("min_artist_spacing", 2),
        )

        RadioService.startRadio(context, config, showToasts = true)
    }
}
