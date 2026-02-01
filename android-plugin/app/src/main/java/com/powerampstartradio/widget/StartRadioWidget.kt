package com.powerampstartradio.widget

import android.content.Context
import android.widget.Toast
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
import java.io.File

class StartRadioWidget : GlanceAppWidget() {

    override suspend fun provideGlance(context: Context, id: GlanceId) {
        provideContent {
            val track = PowerampReceiver.currentTrack

            Column(
                modifier = GlanceModifier
                    .fillMaxSize()
                    .cornerRadius(16.dp)
                    .background(Color(0xFF1E1E2E))
                    .padding(12.dp)
                    .clickable(actionRunCallback<StartRadioAction>()),
                verticalAlignment = Alignment.CenterVertically,
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        text = "▶",
                        style = TextStyle(
                            color = ColorProvider(Color(0xFF89B4FA)),
                            fontSize = 16.sp,
                            fontWeight = FontWeight.Bold
                        )
                    )
                    Spacer(modifier = GlanceModifier.width(8.dp))
                    Text(
                        text = "Start Radio",
                        style = TextStyle(
                            color = ColorProvider(Color.White),
                            fontSize = 14.sp
                        )
                    )
                }
                if (track != null) {
                    val label = listOfNotNull(track.artist, track.title)
                        .joinToString(" — ")
                    if (label.isNotEmpty()) {
                        Text(
                            text = label,
                            style = TextStyle(
                                color = ColorProvider(Color(0xFFBAC2DE)),
                                fontSize = 12.sp
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
        if (!dbFile.exists()) {
            Toast.makeText(context, "No embedding database", Toast.LENGTH_SHORT).show()
            return
        }

        Toast.makeText(context, "Starting radio...", Toast.LENGTH_SHORT).show()
        RadioService.startRadio(context)
    }
}
