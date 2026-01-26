package com.powerampstartradio.poweramp

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.util.Log

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
            }

            PowerampHelper.ACTION_STATUS_CHANGED -> {
                val state = intent.getIntExtra(PowerampHelper.EXTRA_STATE, -1)
                isPlaying = state == 1 // STATE_PLAYING = 1
                Log.d(TAG, "Status changed: isPlaying=$isPlaying")
            }
        }
    }
}
