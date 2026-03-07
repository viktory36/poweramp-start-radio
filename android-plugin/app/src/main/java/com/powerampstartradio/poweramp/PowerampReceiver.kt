package com.powerampstartradio.poweramp

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.util.Log
import com.powerampstartradio.widget.StartRadioWidgetReceiver

/**
 * Receives track change broadcasts from Poweramp.
 *
 * This is used to track the currently playing song so we know which track
 * to use as the seed when the user triggers "Start Radio".
 */
class PowerampReceiver : BroadcastReceiver() {

    companion object {
        private const val TAG = "PowerampReceiver"
        private const val PREFS_NAME = "poweramp_state"
        private const val KEY_REAL_ID = "current_track_real_id"
        private const val KEY_TITLE = "current_track_title"
        private const val KEY_ARTIST = "current_track_artist"
        private const val KEY_ALBUM = "current_track_album"
        private const val KEY_DURATION_MS = "current_track_duration_ms"
        private const val KEY_PATH = "current_track_path"

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

        fun getCurrentTrack(context: Context): PowerampTrack? {
            PowerampHelper.getStickyCurrentTrack(context)?.let { stickyTrack ->
                if (currentTrack != stickyTrack) {
                    currentTrack = stickyTrack
                    persistCurrentTrack(context, stickyTrack)
                }
                return stickyTrack
            }

            currentTrack?.let { return it }

            val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            val title = prefs.getString(KEY_TITLE, null) ?: return null
            val track = PowerampTrack(
                realId = prefs.getLong(KEY_REAL_ID, -1L),
                title = title,
                artist = prefs.getString(KEY_ARTIST, null),
                album = prefs.getString(KEY_ALBUM, null),
                durationMs = prefs.getInt(KEY_DURATION_MS, 0),
                path = prefs.getString(KEY_PATH, null),
            )
            currentTrack = track
            return track
        }

        fun updateCurrentTrack(context: Context, track: PowerampTrack?) {
            currentTrack = track
            persistCurrentTrack(context, track)
            notifyTrackChanged(track)
        }

        private fun persistCurrentTrack(context: Context, track: PowerampTrack?) {
            val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            prefs.edit().apply {
                if (track == null) {
                    remove(KEY_REAL_ID)
                    remove(KEY_TITLE)
                    remove(KEY_ARTIST)
                    remove(KEY_ALBUM)
                    remove(KEY_DURATION_MS)
                    remove(KEY_PATH)
                } else {
                    putLong(KEY_REAL_ID, track.realId)
                    putString(KEY_TITLE, track.title)
                    putString(KEY_ARTIST, track.artist)
                    putString(KEY_ALBUM, track.album)
                    putInt(KEY_DURATION_MS, track.durationMs)
                    putString(KEY_PATH, track.path)
                }
            }.apply()
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
                updateCurrentTrack(context, track)
                StartRadioWidgetReceiver.updateAllWidgets(context)
            }

            PowerampHelper.ACTION_STATUS_CHANGED -> {
                val state = intent.getIntExtra(PowerampHelper.EXTRA_STATE, -1)
                isPlaying = state == 1 // STATE_PLAYING = 1
                Log.d(TAG, "Status changed: isPlaying=$isPlaying")
            }
        }
    }
}
