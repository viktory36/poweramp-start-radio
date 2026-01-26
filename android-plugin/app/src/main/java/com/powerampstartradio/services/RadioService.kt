package com.powerampstartradio.services

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.IBinder
import android.util.Log
import androidx.core.app.NotificationCompat
import com.powerampstartradio.MainActivity
import com.powerampstartradio.R
import com.powerampstartradio.data.EmbeddingDatabase
import com.powerampstartradio.poweramp.PowerampReceiver
import com.powerampstartradio.poweramp.PowerampTrack
import com.powerampstartradio.poweramp.QueueManager
import com.powerampstartradio.poweramp.TrackMatcher
import com.powerampstartradio.similarity.SimilarityEngine
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import java.io.File

/**
 * Foreground service that handles the "Start Radio" functionality.
 *
 * Flow:
 * 1. Get current track from Poweramp
 * 2. Match to embedding database
 * 3. Find similar tracks
 * 4. Map to Poweramp file IDs
 * 5. Queue in Poweramp and start playback
 */
class RadioService : Service() {

    companion object {
        private const val TAG = "RadioService"
        private const val NOTIFICATION_ID = 1
        private const val CHANNEL_ID = "radio_service"

        const val ACTION_START_RADIO = "com.powerampstartradio.START_RADIO"
        const val ACTION_STOP = "com.powerampstartradio.STOP"

        const val EXTRA_NUM_TRACKS = "num_tracks"
        const val EXTRA_SHUFFLE = "shuffle"

        const val DEFAULT_NUM_TRACKS = 50

        fun startRadio(context: Context, numTracks: Int = DEFAULT_NUM_TRACKS, shuffle: Boolean = true) {
            val intent = Intent(context, RadioService::class.java).apply {
                action = ACTION_START_RADIO
                putExtra(EXTRA_NUM_TRACKS, numTracks)
                putExtra(EXTRA_SHUFFLE, shuffle)
            }
            context.startForegroundService(intent)
        }
    }

    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.Main)
    private var embeddingDb: EmbeddingDatabase? = null
    private var similarityEngine: SimilarityEngine? = null

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_START_RADIO -> {
                val numTracks = intent.getIntExtra(EXTRA_NUM_TRACKS, DEFAULT_NUM_TRACKS)
                val shuffle = intent.getBooleanExtra(EXTRA_SHUFFLE, true)

                startForeground(NOTIFICATION_ID, createNotification("Starting radio..."))
                startRadio(numTracks, shuffle)
            }

            ACTION_STOP -> {
                stopSelf()
            }
        }

        return START_NOT_STICKY
    }

    private fun startRadio(numTracks: Int, shuffle: Boolean) {
        serviceScope.launch {
            try {
                // Get current track
                val currentTrack = PowerampReceiver.currentTrack
                if (currentTrack == null) {
                    updateNotification("No track playing in Poweramp")
                    Log.e(TAG, "No current track")
                    stopSelfDelayed()
                    return@launch
                }

                updateNotification("Finding similar tracks to: ${currentTrack.title}")
                Log.d(TAG, "Starting radio for: ${currentTrack.title} by ${currentTrack.artist}")

                // Load database
                val db = getOrCreateDatabase()
                if (db == null) {
                    updateNotification("No embedding database found")
                    Log.e(TAG, "No database")
                    stopSelfDelayed()
                    return@launch
                }

                // Find matching embedded track
                val matcher = TrackMatcher(db)
                val matchResult = matcher.findMatch(currentTrack)

                if (matchResult == null) {
                    updateNotification("Track not found in database")
                    Log.e(TAG, "No match found for: ${currentTrack.title}")
                    stopSelfDelayed()
                    return@launch
                }

                Log.d(TAG, "Found match (${matchResult.matchType}): ${matchResult.embeddedTrack.title}")

                // Find similar tracks
                val engine = getOrCreateEngine(db)
                engine.loadEmbeddings()

                var similarTracks = engine.findSimilarTracks(
                    seedTrackId = matchResult.embeddedTrack.id,
                    topN = numTracks,
                    excludeSeed = true
                )

                if (shuffle) {
                    similarTracks = similarTracks.shuffled()
                }

                Log.d(TAG, "Found ${similarTracks.size} similar tracks")

                if (similarTracks.isEmpty()) {
                    updateNotification("No similar tracks found")
                    stopSelfDelayed()
                    return@launch
                }

                // Map to Poweramp file IDs
                val embeddedTracks = similarTracks.map { it.track }
                val fileIds = matcher.mapEmbeddedTracksToFileIds(this@RadioService, embeddedTracks)

                if (fileIds.isEmpty()) {
                    updateNotification("Could not find tracks in Poweramp library")
                    stopSelfDelayed()
                    return@launch
                }

                // Queue and play
                val queueManager = QueueManager(this@RadioService)
                val result = queueManager.setQueueAndPlay(fileIds)

                updateNotification(result.message)
                Log.d(TAG, "Queue result: ${result.message}")

                // Stop after a short delay
                stopSelfDelayed()

            } catch (e: Exception) {
                Log.e(TAG, "Error starting radio", e)
                updateNotification("Error: ${e.message}")
                stopSelfDelayed()
            }
        }
    }

    private fun getOrCreateDatabase(): EmbeddingDatabase? {
        embeddingDb?.let { return it }

        val dbFile = File(filesDir, "embeddings.db")
        if (!dbFile.exists()) {
            Log.e(TAG, "Database file does not exist: ${dbFile.absolutePath}")
            return null
        }

        return try {
            EmbeddingDatabase.open(dbFile).also { embeddingDb = it }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to open database", e)
            null
        }
    }

    private fun getOrCreateEngine(db: EmbeddingDatabase): SimilarityEngine {
        similarityEngine?.let { return it }
        return SimilarityEngine(db).also { similarityEngine = it }
    }

    private fun stopSelfDelayed() {
        serviceScope.launch {
            kotlinx.coroutines.delay(3000)
            stopSelf()
        }
    }

    private fun createNotificationChannel() {
        val channel = NotificationChannel(
            CHANNEL_ID,
            "Radio Service",
            NotificationManager.IMPORTANCE_LOW
        ).apply {
            description = "Shows when Start Radio is finding similar tracks"
        }

        val notificationManager = getSystemService(NotificationManager::class.java)
        notificationManager.createNotificationChannel(channel)
    }

    private fun createNotification(message: String): Notification {
        val pendingIntent = PendingIntent.getActivity(
            this,
            0,
            Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_IMMUTABLE
        )

        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("Poweramp Start Radio")
            .setContentText(message)
            .setSmallIcon(R.drawable.ic_radio)
            .setContentIntent(pendingIntent)
            .setOngoing(true)
            .build()
    }

    private fun updateNotification(message: String) {
        val notificationManager = getSystemService(NotificationManager::class.java)
        notificationManager.notify(NOTIFICATION_ID, createNotification(message))
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        super.onDestroy()
        serviceScope.cancel()
        embeddingDb?.close()
        similarityEngine?.clearCache()
    }
}
