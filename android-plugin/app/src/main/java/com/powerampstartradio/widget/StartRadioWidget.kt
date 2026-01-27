package com.powerampstartradio.widget

import android.content.Context
import android.widget.Toast
import androidx.compose.ui.unit.dp
import androidx.glance.Button
import androidx.glance.GlanceId
import androidx.glance.GlanceModifier
import androidx.glance.action.ActionParameters
import androidx.glance.action.clickable
import androidx.glance.appwidget.GlanceAppWidget
import androidx.glance.appwidget.action.ActionCallback
import androidx.glance.appwidget.action.actionRunCallback
import androidx.glance.appwidget.provideContent
import androidx.glance.layout.Alignment
import androidx.glance.layout.Box
import androidx.glance.layout.fillMaxSize
import androidx.glance.layout.padding
import androidx.glance.text.Text
import com.powerampstartradio.poweramp.PowerampReceiver
import com.powerampstartradio.services.RadioService
import java.io.File

/**
 * Minimal home screen widget for one-tap radio start.
 */
class StartRadioWidget : GlanceAppWidget() {

    override suspend fun provideGlance(context: Context, id: GlanceId) {
        provideContent {
            Box(
                modifier = GlanceModifier
                    .fillMaxSize()
                    .padding(8.dp)
                    .clickable(actionRunCallback<StartRadioAction>()),
                contentAlignment = Alignment.Center
            ) {
                Text(text = "▶ Start Radio")
            }
        }
    }
}

/**
 * Action callback for widget tap.
 */
class StartRadioAction : ActionCallback {
    override suspend fun onAction(
        context: Context,
        glanceId: GlanceId,
        parameters: ActionParameters
    ) {
        val currentTrack = PowerampReceiver.currentTrack
        if (currentTrack == null) {
            Toast.makeText(context, "No track playing in Poweramp", Toast.LENGTH_SHORT).show()
            return
        }

        val dbFile = File(context.filesDir, "embeddings.db")
        if (!dbFile.exists()) {
            Toast.makeText(context, "No embedding database", Toast.LENGTH_SHORT).show()
            return
        }

        Toast.makeText(context, "Starting radio...", Toast.LENGTH_SHORT).show()
        RadioService.startRadio(context)
    }
}
