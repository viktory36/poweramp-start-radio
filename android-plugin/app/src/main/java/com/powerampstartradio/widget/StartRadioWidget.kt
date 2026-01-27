package com.powerampstartradio.widget

import android.content.Context
import android.content.Intent
import android.widget.Toast
import androidx.glance.GlanceId
import androidx.glance.GlanceModifier
import androidx.glance.action.ActionParameters
import androidx.glance.action.clickable
import androidx.glance.appwidget.GlanceAppWidget
import androidx.glance.appwidget.action.ActionCallback
import androidx.glance.appwidget.action.actionRunCallback
import androidx.glance.appwidget.provideContent
import androidx.glance.background
import androidx.glance.layout.Alignment
import androidx.glance.layout.Row
import androidx.glance.layout.fillMaxSize
import androidx.glance.layout.padding
import androidx.glance.text.FontWeight
import androidx.glance.text.Text
import androidx.glance.text.TextStyle
import androidx.glance.unit.ColorProvider
import android.graphics.Color
import androidx.glance.layout.Spacer
import androidx.glance.layout.width
import androidx.compose.ui.unit.dp
import com.powerampstartradio.data.EmbeddingDatabase
import com.powerampstartradio.poweramp.PowerampReceiver
import com.powerampstartradio.services.RadioService
import java.io.File

/**
 * Minimal home screen widget for one-tap radio start.
 * Shows toast feedback instead of updating widget UI.
 */
class StartRadioWidget : GlanceAppWidget() {

    override suspend fun provideGlance(context: Context, id: GlanceId) {
        provideContent {
            Row(
                modifier = GlanceModifier
                    .fillMaxSize()
                    .background(ColorProvider(Color.parseColor("#1E1E2E")))
                    .clickable(actionRunCallback<StartRadioAction>())
                    .padding(horizontal = 16.dp, vertical = 12.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "\u25B6",  // Play symbol
                    style = TextStyle(
                        color = ColorProvider(Color.parseColor("#89B4FA")),
                        fontWeight = FontWeight.Bold
                    )
                )
                Spacer(modifier = GlanceModifier.width(8.dp))
                Text(
                    text = "Start Radio",
                    style = TextStyle(
                        color = ColorProvider(Color.WHITE),
                        fontWeight = FontWeight.Medium
                    )
                )
            }
        }
    }
}

/**
 * Action callback for widget tap.
 * Shows toast and starts radio service.
 */
class StartRadioAction : ActionCallback {
    override suspend fun onAction(
        context: Context,
        glanceId: GlanceId,
        parameters: ActionParameters
    ) {
        // Check if track is playing
        val currentTrack = PowerampReceiver.currentTrack
        if (currentTrack == null) {
            Toast.makeText(context, "No track playing in Poweramp", Toast.LENGTH_SHORT).show()
            return
        }

        // Check if database exists
        val dbFile = File(context.filesDir, "embeddings.db")
        if (!dbFile.exists()) {
            Toast.makeText(context, "No embedding database", Toast.LENGTH_SHORT).show()
            return
        }

        // Show starting toast
        Toast.makeText(context, "Starting radio...", Toast.LENGTH_SHORT).show()

        // Start radio service
        RadioService.startRadio(context)

        // The service will update its state, but we can't easily listen from here.
        // For simplicity, we'll just show a toast when starting.
        // The user can check the app or notification for detailed results.
    }
}
