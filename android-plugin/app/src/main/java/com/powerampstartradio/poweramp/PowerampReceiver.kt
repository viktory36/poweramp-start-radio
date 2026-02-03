package com.powerampstartradio.poweramp

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.util.Log
import androidx.glance.appwidget.GlanceAppWidgetManager
import com.powerampstartradio.widget.StartRadioWidget
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

/**
 * Receives track change broadcasts from Poweramp.
 *
 * This is used to track the currently playing song so we know which track
 * to use as the seed when the user triggers "Start Radio".
 */
class PowerampReceiver : BroadcastReceiver() {

    companion object {
        private const val TAG = "PowerampReceiver"

        // Singleton to hold current track state
        @Volatile
        var currentTrack: PowerampTrack? = null
            private set

        @Volatile
        var isPlaying: Boolean = false
            private set

        // Listeners for track changes
        private val trackChangeListeners = mutableListOf<(PowerampTrack?) -> Unit>()

        fun addTrackChangeListener(listener: (PowerampTrack?) -> Unit) {
            trackChangeListeners.add(listener)
        }

        fun removeTrackChangeListener(listener: (PowerampTrack?) -> Unit) {
            trackChangeListeners.remove(listener)
        }

        private fun notifyTrackChanged(track: PowerampTrack?) {
            trackChangeListeners.forEach { it(track) }
        }
    }

    override fun onReceive(context: Context, intent: Intent) {
        when (intent.action) {
            PowerampHelper.ACTION_TRACK_CHANGED -> {
                val track = PowerampHelper.getCurrentTrackFromIntent(intent)
                Log.d(TAG, "Track changed: ${track?.title} by ${track?.artist}")
                currentTrack = track
                notifyTrackChanged(track)

                // Update widget with new track name
                val pendingResult = goAsync()
                CoroutineScope(Dispatchers.IO).launch {
                    try {
                        val manager = GlanceAppWidgetManager(context)
                        val widget = StartRadioWidget()
                        val ids = manager.getGlanceIds(StartRadioWidget::class.java)
                        ids.forEach { id -> widget.update(context, id) }
                    } catch (e: Exception) {
                        Log.w(TAG, "Widget update failed", e)
                    } finally {
                        pendingResult.finish()
                    }
                }
            }

            PowerampHelper.ACTION_STATUS_CHANGED -> {
                val state = intent.getIntExtra(PowerampHelper.EXTRA_STATE, -1)
                isPlaying = state == 1 // STATE_PLAYING = 1
                Log.d(TAG, "Status changed: isPlaying=$isPlaying")
            }
        }
    }
}
