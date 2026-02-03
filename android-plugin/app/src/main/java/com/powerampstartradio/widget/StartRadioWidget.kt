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
import com.powerampstartradio.data.EmbeddingModel
import com.powerampstartradio.poweramp.PowerampReceiver
import com.powerampstartradio.services.RadioService
import com.powerampstartradio.similarity.AnchorExpandConfig
import com.powerampstartradio.similarity.SearchStrategy
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
        val numTracks = prefs.getInt("num_tracks", RadioService.DEFAULT_NUM_TRACKS)

        val strategy = try {
            val stored = prefs.getString("search_strategy", SearchStrategy.ANCHOR_EXPAND.name)!!
            if (stored == "FEED_FORWARD") SearchStrategy.ANCHOR_EXPAND
            else SearchStrategy.valueOf(stored)
        } catch (e: IllegalArgumentException) {
            SearchStrategy.ANCHOR_EXPAND
        }

        val drift = prefs.getBoolean("drift", false)

        val aeConfig = if (strategy == SearchStrategy.ANCHOR_EXPAND) {
            val primaryModel = try {
                val stored = prefs.getString("anchor_expand_primary", null)
                    ?: prefs.getString("feed_forward_primary", EmbeddingModel.MULAN.name)
                EmbeddingModel.valueOf(stored!!)
            } catch (e: IllegalArgumentException) {
                EmbeddingModel.MULAN
            }
            val expansion = prefs.getInt("anchor_expand_expansion",
                prefs.getInt("feed_forward_expansion", 3))
            AnchorExpandConfig(primaryModel, expansion)
        } else null

        RadioService.startRadio(context, numTracks, strategy, aeConfig, drift, showToasts = true)
    }
}
