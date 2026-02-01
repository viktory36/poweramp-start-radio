package com.powerampstartradio.widget

import android.content.Context
import android.content.Intent
import androidx.glance.appwidget.GlanceAppWidget
import androidx.glance.appwidget.GlanceAppWidgetManager
import androidx.glance.appwidget.GlanceAppWidgetReceiver
import com.powerampstartradio.poweramp.PowerampHelper
import com.powerampstartradio.poweramp.PowerampReceiver
import kotlinx.coroutines.MainScope
import kotlinx.coroutines.launch

class StartRadioWidgetReceiver : GlanceAppWidgetReceiver() {
    override val glanceAppWidget: GlanceAppWidget = StartRadioWidget()

    private val powerampReceiver = PowerampReceiver()
    private val scope = MainScope()

    override fun onReceive(context: Context, intent: Intent) {
        super.onReceive(context, intent)

        if (intent.action == PowerampHelper.ACTION_TRACK_CHANGED) {
            powerampReceiver.onReceive(context, intent)
            scope.launch {
                val manager = GlanceAppWidgetManager(context)
                val ids = manager.getGlanceIds(StartRadioWidget::class.java)
                ids.forEach { id -> glanceAppWidget.update(context, id) }
            }
        }
    }
}
