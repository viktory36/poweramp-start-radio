package com.powerampstartradio.poweramp

import android.content.Context
import android.util.Log

/**
 * Manages the Poweramp queue for radio functionality.
 */
class QueueManager(
    private val context: Context
) {
    companion object {
        private const val TAG = "QueueManager"
    }

    /**
     * Result of queueing radio tracks.
     */
    data class QueueResult(
        val success: Boolean,
        val tracksQueued: Int,
        val message: String
    )

    /**
     * Clear the queue and add new tracks for radio playback.
     * Poweramp will automatically play from queue when current track ends.
     *
     * @param fileIds Poweramp file IDs to add to the queue (in similarity order)
     * @param clearExisting Whether to clear the existing queue first
     */
    fun setQueueAndPlay(
        fileIds: List<Long>,
        clearExisting: Boolean = true
    ): QueueResult {
        if (fileIds.isEmpty()) {
            return QueueResult(false, 0, "No tracks to queue")
        }

        Log.d(TAG, "Queueing ${fileIds.size} tracks")

        try {
            if (clearExisting) {
                PowerampHelper.clearQueue(context)
                Log.d(TAG, "Cleared existing queue")
            }

            val added = PowerampHelper.addTracksToQueue(context, fileIds)
            Log.d(TAG, "Added $added tracks to queue")

            if (added == 0) {
                return QueueResult(false, 0, "Failed to add tracks to queue")
            }

            PowerampHelper.reloadData(context)
            Log.d(TAG, "Queue ready - will play after current track")

            return QueueResult(true, added, "Queued $added similar tracks")

        } catch (e: Exception) {
            Log.e(TAG, "Error setting queue", e)
            return QueueResult(false, 0, "Error: ${e.message}")
        }
    }

    /**
     * Open the Poweramp queue UI without changing playback.
     */
    fun openQueueInPoweramp() {
        PowerampHelper.openQueue(context)
    }
}
