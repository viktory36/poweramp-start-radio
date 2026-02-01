package com.powerampstartradio.services

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.IBinder
import android.util.Log
import androidx.core.app.NotificationCompat
import com.powerampstartradio.MainActivity
import com.powerampstartradio.R
import com.powerampstartradio.data.EmbeddingDatabase
import com.powerampstartradio.poweramp.PowerampHelper
import com.powerampstartradio.poweramp.PowerampReceiver
import com.powerampstartradio.poweramp.TrackMatcher
import com.powerampstartradio.similarity.SimilarityEngine
import com.powerampstartradio.ui.QueueStatus
import com.powerampstartradio.ui.QueuedTrackResult
import com.powerampstartradio.ui.RadioResult
import com.powerampstartradio.ui.RadioUiState
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import java.io.File

/**
 * Foreground service that handles the "Start Radio" functionality.
 *
 * Flow:
 * 1. Get current track from Poweramp
 * 2. Match to embedding database
 * 3. Find similar tracks (dual-model interleaved if both available)
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

        // UI state shared with MainActivity
        private val _uiState = MutableStateFlow<RadioUiState>(RadioUiState.Idle)
        val uiState: StateFlow<RadioUiState> = _uiState.asStateFlow()

        fun startRadio(context: Context, numTracks: Int = DEFAULT_NUM_TRACKS, shuffle: Boolean = true) {
            _uiState.value = RadioUiState.Loading
            val intent = Intent(context, RadioService::class.java).apply {
                action = ACTION_START_RADIO
                putExtra(EXTRA_NUM_TRACKS, numTracks)
                putExtra(EXTRA_SHUFFLE, shuffle)
            }
            context.startForegroundService(intent)
        }

        fun resetState() {
            _uiState.value = RadioUiState.Idle
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
                    _uiState.value = RadioUiState.Error("No track playing in Poweramp")
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
                    _uiState.value = RadioUiState.Error("No embedding database found")
                    updateNotification("No embedding database found")
                    Log.e(TAG, "No database")
                    stopSelfDelayed()
                    return@launch
                }

                // Find matching embedded track
                val matcher = TrackMatcher(db)
                val matchResult = matcher.findMatch(currentTrack)

                if (matchResult == null) {
                    _uiState.value = RadioUiState.Error("Track not found in database")
                    updateNotification("Track not found in database")
                    Log.e(TAG, "No match found for: ${currentTrack.title}")
                    stopSelfDelayed()
                    return@launch
                }

                Log.d(TAG, "Found match (${matchResult.matchType}): ${matchResult.embeddedTrack.title}")

                // Find similar tracks — dual or single model
                val engine = getOrCreateEngine(db)
                val availableModels = db.getAvailableModels()
                val isDual = availableModels.size > 1

                val similarTracks = if (isDual) {
                    Log.d(TAG, "Using dual-model search (${availableModels.joinToString { it.name }})")
                    engine.findSimilarTracksDual(
                        seedTrackId = matchResult.embeddedTrack.id,
                        requestedCount = numTracks
                    )
                } else {
                    val model = availableModels.firstOrNull() ?: com.powerampstartradio.data.EmbeddingModel.MUQ
                    Log.d(TAG, "Using single-model search (${model.name})")
                    engine.loadEmbeddings(model)
                    engine.findSimilarTracks(
                        seedTrackId = matchResult.embeddedTrack.id,
                        topN = numTracks,
                        excludeSeed = true,
                        model = model
                    )
                }

                var orderedTracks = if (shuffle) similarTracks.shuffled() else similarTracks

                Log.d(TAG, "Found ${orderedTracks.size} similar tracks")

                if (orderedTracks.isEmpty()) {
                    _uiState.value = RadioUiState.Error("No similar tracks found")
                    updateNotification("No similar tracks found")
                    stopSelfDelayed()
                    return@launch
                }

                // Map to Poweramp file IDs (preserving similarity scores)
                val mappedTracks = matcher.mapSimilarTracksToFileIds(this@RadioService, orderedTracks)

                // Get file IDs for tracks that were found
                val fileIds = mappedTracks.mapNotNull { it.fileId }

                if (fileIds.isEmpty()) {
                    _uiState.value = RadioUiState.Error("Could not find tracks in Poweramp library")
                    updateNotification("Could not find tracks in Poweramp library")
                    stopSelfDelayed()
                    return@launch
                }

                // Clear queue and add tracks
                PowerampHelper.clearQueue(this@RadioService)
                val queuedFileIds = mutableSetOf<Long>()

                // Track which file IDs were successfully queued
                for (fileId in fileIds) {
                    val added = PowerampHelper.addTracksToQueue(this@RadioService, listOf(fileId))
                    if (added > 0) {
                        queuedFileIds.add(fileId)
                    }
                }

                // Build per-track results
                val trackResults = mappedTracks.map { mapped ->
                    val status = when {
                        mapped.fileId == null -> QueueStatus.NOT_IN_LIBRARY
                        mapped.fileId in queuedFileIds -> QueueStatus.QUEUED
                        else -> QueueStatus.QUEUE_FAILED
                    }
                    QueuedTrackResult(
                        track = mapped.similarTrack.track,
                        similarity = mapped.similarTrack.similarity,
                        status = status,
                        modelUsed = mapped.similarTrack.model
                    )
                }

                // Create result
                val radioResult = RadioResult(
                    seedTrack = currentTrack,
                    matchType = matchResult.matchType,
                    tracks = trackResults,
                    availableModels = availableModels
                )

                _uiState.value = RadioUiState.Success(radioResult)

                // Notify Poweramp to reload queue and start playback
                PowerampHelper.reloadData(this@RadioService)
                kotlinx.coroutines.delay(100)
                PowerampHelper.playQueue(this@RadioService)

                val modeLabel = if (isDual) " (dual)" else ""
                val message = "${radioResult.queuedCount} queued / ${radioResult.failedCount} failed$modeLabel"
                updateNotification(message)
                Log.d(TAG, "Queue result: $message")

                // Stop after a short delay
                stopSelfDelayed()

            } catch (e: Exception) {
                Log.e(TAG, "Error starting radio", e)
                _uiState.value = RadioUiState.Error("Error: ${e.message}")
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
