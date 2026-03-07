package com.powerampstartradio.widget

import android.app.PendingIntent
import android.appwidget.AppWidgetManager
import android.appwidget.AppWidgetProvider
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.view.View
import android.widget.RemoteViews
import com.powerampstartradio.MainActivity
import com.powerampstartradio.R
import com.powerampstartradio.poweramp.PowerampHelper
import com.powerampstartradio.poweramp.PowerampReceiver
import com.powerampstartradio.services.RadioService
import com.powerampstartradio.ui.DecaySchedule
import com.powerampstartradio.ui.DriftMode
import com.powerampstartradio.ui.RadioConfig
import com.powerampstartradio.ui.SelectionMode
import java.io.File

class StartRadioWidgetReceiver : AppWidgetProvider() {

    companion object {
        private const val ACTION_START_RADIO = "com.powerampstartradio.widget.ACTION_START_RADIO"

        fun updateAllWidgets(context: Context) {
            val manager = AppWidgetManager.getInstance(context)
            val ids = manager.getAppWidgetIds(ComponentName(context, StartRadioWidgetReceiver::class.java))
            if (ids.isNotEmpty()) {
                updateWidgets(context, manager, ids)
            }
        }

        private fun updateWidgets(
            context: Context,
            manager: AppWidgetManager,
            appWidgetIds: IntArray
        ) {
            val track = PowerampReceiver.getCurrentTrack(context)
            val title = track?.title?.takeIf { it.isNotBlank() } ?: "No track playing"
            val subtitle = listOfNotNull(
                track?.artist?.takeIf { !it.isNullOrBlank() },
                track?.album?.takeIf { !it.isNullOrBlank() }
            ).joinToString(" · ")

            val views = RemoteViews(context.packageName, R.layout.widget_start_radio).apply {
                setTextViewText(R.id.widget_track_title, title)
                if (subtitle.isNotBlank()) {
                    setViewVisibility(R.id.widget_track_subtitle, View.VISIBLE)
                    setTextViewText(R.id.widget_track_subtitle, subtitle)
                } else {
                    setViewVisibility(R.id.widget_track_subtitle, View.GONE)
                    setTextViewText(R.id.widget_track_subtitle, "")
                }
                setOnClickPendingIntent(R.id.widget_root, openAppPendingIntent(context))
                setOnClickPendingIntent(R.id.widget_start_button, startRadioPendingIntent(context))
            }

            appWidgetIds.forEach { manager.updateAppWidget(it, views) }
        }

        private fun startRadioPendingIntent(context: Context): PendingIntent {
            val intent = Intent(context, StartRadioWidgetReceiver::class.java).apply {
                action = ACTION_START_RADIO
            }
            return PendingIntent.getBroadcast(
                context,
                0,
                intent,
                PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
            )
        }

        private fun openAppPendingIntent(context: Context): PendingIntent {
            val intent = Intent(context, MainActivity::class.java).apply {
                flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP
            }
            return PendingIntent.getActivity(
                context,
                1,
                intent,
                PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
            )
        }

        private fun buildRadioConfig(context: Context): RadioConfig {
            val prefs = context.getSharedPreferences("settings", Context.MODE_PRIVATE)
            return RadioConfig(
                numTracks = prefs.getInt("num_tracks", RadioService.DEFAULT_NUM_TRACKS),
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
        }
    }

    override fun onReceive(context: Context, intent: Intent) {
        when (intent.action) {
            ACTION_START_RADIO -> {
                val dbFile = File(context.filesDir, "embeddings.db")
                if (dbFile.exists()) {
                    PowerampReceiver.getCurrentTrack(context)
                    RadioService.startRadio(context, buildRadioConfig(context), showToasts = true)
                }
                return
            }

            PowerampHelper.ACTION_TRACK_CHANGED -> {
                val track = PowerampHelper.getCurrentTrackFromIntent(intent)
                PowerampReceiver.updateCurrentTrack(context, track)
                updateAllWidgets(context)
                return
            }

            PowerampHelper.ACTION_STATUS_CHANGED,
            AppWidgetManager.ACTION_APPWIDGET_UPDATE,
            Intent.ACTION_MY_PACKAGE_REPLACED -> updateAllWidgets(context)
        }

        super.onReceive(context, intent)
    }

    override fun onUpdate(context: Context, appWidgetManager: AppWidgetManager, appWidgetIds: IntArray) {
        updateWidgets(context, appWidgetManager, appWidgetIds)
    }
}
