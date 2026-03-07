package com.powerampstartradio.widget

import android.content.Context
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.glance.action.clickable
import androidx.glance.GlanceId
import androidx.glance.GlanceModifier
import androidx.glance.Image
import androidx.glance.ImageProvider
import androidx.glance.LocalSize
import androidx.glance.action.ActionParameters
import androidx.glance.appwidget.GlanceAppWidget
import androidx.glance.appwidget.SizeMode
import androidx.glance.appwidget.action.ActionCallback
import androidx.glance.appwidget.action.actionRunCallback
import androidx.glance.appwidget.cornerRadius
import androidx.glance.appwidget.provideContent
import androidx.glance.background
import androidx.glance.layout.Alignment
import androidx.glance.layout.Box
import androidx.glance.layout.Row
import androidx.glance.layout.Spacer
import androidx.glance.layout.fillMaxSize
import androidx.glance.layout.fillMaxWidth
import androidx.glance.layout.padding
import androidx.glance.layout.size
import androidx.glance.layout.width
import androidx.glance.color.ColorProvider
import androidx.glance.text.FontWeight
import androidx.glance.text.Text
import androidx.glance.text.TextStyle
import com.powerampstartradio.R
import com.powerampstartradio.poweramp.PowerampReceiver
import com.powerampstartradio.services.RadioService
import com.powerampstartradio.ui.DecaySchedule
import com.powerampstartradio.ui.DriftMode
import com.powerampstartradio.ui.RadioConfig
import com.powerampstartradio.ui.SelectionMode
import java.io.File

class StartRadioWidget : GlanceAppWidget() {
    override val sizeMode: SizeMode = SizeMode.Exact

    companion object {
        private val WidgetPrimaryText = ColorProvider(Color(0xFFF5F7FA), Color(0xFFF5F7FA))
        private val WidgetMutedText = ColorProvider(Color(0xFFA8B0BE), Color(0xFFA8B0BE))
        private val WidgetButtonBackground = ColorProvider(Color(0xB8111213), Color(0xB8111213))
    }

    override suspend fun provideGlance(context: Context, id: GlanceId) {
        provideContent {
            val track = PowerampReceiver.getCurrentTrack(context)
            val title = track?.title?.takeIf { !it.isNullOrBlank() }
            val size = LocalSize.current
            val titleFont = if (size.width < 170.dp) 12.sp else 14.sp

            Row(
                modifier = GlanceModifier
                    .fillMaxSize()
                    .padding(horizontal = 4.dp, vertical = 4.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                StartRadioButton()
                Spacer(modifier = GlanceModifier.width(8.dp))
                TrackTitle(title, titleFont, GlanceModifier.fillMaxWidth())
            }
        }
    }

    @androidx.compose.runtime.Composable
    private fun StartRadioButton() {
        Box(
            modifier = GlanceModifier
                .cornerRadius(18.dp)
                .background(WidgetButtonBackground)
                .clickable(actionRunCallback<StartRadioAction>())
                .padding(9.dp),
            contentAlignment = Alignment.Center
        ) {
            Image(
                provider = ImageProvider(R.drawable.ic_radio_waves),
                contentDescription = "Start Radio",
                modifier = GlanceModifier.size(18.dp)
            )
        }
    }

    @androidx.compose.runtime.Composable
    private fun TrackTitle(
        title: String?,
        fontSize: androidx.compose.ui.unit.TextUnit,
        modifier: GlanceModifier = GlanceModifier
    ) {
        Text(
            text = title ?: "No track playing",
            modifier = modifier,
            style = TextStyle(
                color = if (title != null) WidgetPrimaryText else WidgetMutedText,
                fontSize = fontSize,
                fontWeight = FontWeight.Bold
            ),
            maxLines = 1
        )
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

        PowerampReceiver.getCurrentTrack(context)

        val prefs = context.getSharedPreferences("settings", Context.MODE_PRIVATE)

        val config = RadioConfig(
            numTracks = prefs.getInt("num_tracks", RadioService.DEFAULT_NUM_TRACKS),
            // candidatePoolSize auto-computed by RecommendationEngine
            selectionMode = try {
                SelectionMode.valueOf(prefs.getString("selection_mode", SelectionMode.MMR.name)!!)
            } catch (e: IllegalArgumentException) { SelectionMode.MMR },
            driftEnabled = prefs.getBoolean("drift_enabled", false),
            driftMode = try {
                DriftMode.valueOf(prefs.getString("drift_mode", DriftMode.SEED_INTERPOLATION.name)!!)
            } catch (e: IllegalArgumentException) { DriftMode.SEED_INTERPOLATION },
            anchorStrength = prefs.getFloat("anchor_strength", 0.5f),
            walkRestartAlpha = prefs.getFloat("walk_restart_alpha", 0.5f),
            anchorDecay = try {
                DecaySchedule.valueOf(prefs.getString("anchor_decay", DecaySchedule.EXPONENTIAL.name)!!)
            } catch (e: IllegalArgumentException) { DecaySchedule.EXPONENTIAL },
            momentumBeta = prefs.getFloat("momentum_beta", 0.7f),
            diversityLambda = prefs.getFloat("diversity_lambda", 0.4f),
            maxPerArtist = prefs.getInt("max_per_artist", 8),
            minArtistSpacing = prefs.getInt("min_artist_spacing", 3),
        )

        RadioService.startRadio(context, config, showToasts = true)
    }
}
