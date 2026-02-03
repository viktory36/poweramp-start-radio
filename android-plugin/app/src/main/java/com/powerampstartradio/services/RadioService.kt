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
import android.widget.Toast
import androidx.core.app.NotificationCompat
import com.powerampstartradio.MainActivity
import com.powerampstartradio.R
import com.powerampstartradio.data.EmbeddingDatabase
import com.powerampstartradio.data.EmbeddingModel
import com.powerampstartradio.poweramp.PowerampHelper
import com.powerampstartradio.poweramp.PowerampReceiver
import com.powerampstartradio.poweramp.TrackMatcher
import com.powerampstartradio.similarity.AnchorExpandConfig
import com.powerampstartradio.similarity.SearchStrategy
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
 * 3. Find similar tracks using selected strategy (MuLan/Flamingo/Interleave/Feed-Forward)
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
        const val EXTRA_SHOW_TOASTS = "show_toasts"
        const val EXTRA_STRATEGY = "strategy"
        const val EXTRA_AE_PRIMARY_MODEL = "ae_primary_model"
        const val EXTRA_AE_EXPANSION = "ae_expansion"
        const val EXTRA_DRIFT = "drift"

        const val DEFAULT_NUM_TRACKS = 50

        // UI state shared with MainActivity
        private val _uiState = MutableStateFlow<RadioUiState>(RadioUiState.Idle)
        val uiState: StateFlow<RadioUiState> = _uiState.asStateFlow()

        // Session history — accumulated radio results
        private val _sessionHistory = MutableStateFlow<List<RadioResult>>(emptyList())
        val sessionHistory: StateFlow<List<RadioResult>> = _sessionHistory.asStateFlow()

        fun startRadio(
            context: Context,
            numTracks: Int = DEFAULT_NUM_TRACKS,
            strategy: SearchStrategy = SearchStrategy.ANCHOR_EXPAND,
            anchorExpandConfig: AnchorExpandConfig? = null,
            drift: Boolean = false,
            showToasts: Boolean = false
        ) {
            _uiState.value = RadioUiState.Loading()
            val intent = Intent(context, RadioService::class.java).apply {
                action = ACTION_START_RADIO
                putExtra(EXTRA_NUM_TRACKS, numTracks)
                putExtra(EXTRA_SHOW_TOASTS, showToasts)
                putExtra(EXTRA_STRATEGY, strategy.name)
                putExtra(EXTRA_DRIFT, drift)
                if (anchorExpandConfig != null) {
                    putExtra(EXTRA_AE_PRIMARY_MODEL, anchorExpandConfig.primaryModel.name)
                    putExtra(EXTRA_AE_EXPANSION, anchorExpandConfig.expansionCount)
                }
            }
            context.startForegroundService(intent)
        }

        fun resetState() {
            _uiState.value = RadioUiState.Idle
        }

        fun clearHistory() {
            _sessionHistory.value = emptyList()
        }
    }

    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.Main)
    private var embeddingDb: EmbeddingDatabase? = null
    private var similarityEngine: SimilarityEngine? = null
    private var showToasts: Boolean = false

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_START_RADIO -> {
                val numTracks = intent.getIntExtra(EXTRA_NUM_TRACKS, DEFAULT_NUM_TRACKS)
                showToasts = intent.getBooleanExtra(EXTRA_SHOW_TOASTS, false)

                val strategy = try {
                    val stored = intent.getStringExtra(EXTRA_STRATEGY) ?: SearchStrategy.ANCHOR_EXPAND.name
                    if (stored == "FEED_FORWARD") SearchStrategy.ANCHOR_EXPAND
                    else SearchStrategy.valueOf(stored)
                } catch (e: IllegalArgumentException) {
                    SearchStrategy.ANCHOR_EXPAND
                }

                val drift = intent.getBooleanExtra(EXTRA_DRIFT, false)

                val anchorExpandConfig = if (strategy == SearchStrategy.ANCHOR_EXPAND) {
                    val primaryModel = try {
                        EmbeddingModel.valueOf(intent.getStringExtra(EXTRA_AE_PRIMARY_MODEL) ?: EmbeddingModel.MULAN.name)
                    } catch (e: IllegalArgumentException) {
                        EmbeddingModel.MULAN
                    }
                    val expansion = intent.getIntExtra(EXTRA_AE_EXPANSION, 3)
                    AnchorExpandConfig(primaryModel, expansion)
                } else null

                startForeground(NOTIFICATION_ID, createNotification("Starting radio..."))
                startRadio(numTracks, strategy, anchorExpandConfig, drift)
            }

            ACTION_STOP -> {
                stopSelf()
            }
        }

        return START_NOT_STICKY
    }

    private fun startRadio(
        numTracks: Int,
        strategy: SearchStrategy,
        anchorExpandConfig: AnchorExpandConfig?,
        drift: Boolean
    ) {
        serviceScope.launch {
            try {
                toast("Starting radio...")

                // Get current track
                val currentTrack = PowerampReceiver.currentTrack
                if (currentTrack == null) {
                    _uiState.value = RadioUiState.Error("No track playing in Poweramp")
                    updateNotification("No track playing in Poweramp")
                    toast("No track playing in Poweramp")
                    Log.e(TAG, "No current track")
                    stopSelfDelayed()
                    return@launch
                }

                updateNotification("Finding similar tracks to: ${currentTrack.title}")
                Log.d(TAG, "Starting radio for: ${currentTrack.title} by ${currentTrack.artist}")
                Log.d(TAG, "Strategy: ${strategy.name}" +
                    (if (anchorExpandConfig != null) " (${anchorExpandConfig.primaryModel.name} -> ${anchorExpandConfig.secondaryModel.name}, N=${anchorExpandConfig.expansionCount})" else "") +
                    (if (drift) " [drift]" else "")
                )

                // Load database
                val db = getOrCreateDatabase()
                if (db == null) {
                    _uiState.value = RadioUiState.Error("No embedding database found")
                    updateNotification("No embedding database found")
                    toast("No embedding database found")
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
                    toast("Track not found in database")
                    Log.e(TAG, "No match found for: ${currentTrack.title}")
                    stopSelfDelayed()
                    return@launch
                }

                Log.d(TAG, "Found match (${matchResult.matchType}): ${matchResult.embeddedTrack.title}")

                // Ensure mmap indices are ready (one-time extraction on first use)
                val engine = getOrCreateEngine(db)
                val availableModels = db.getAvailableModels()

                engine.ensureIndices { message ->
                    _uiState.value = RadioUiState.Loading(message)
                    updateNotification(message)
                }

                updateNotification("Searching for similar tracks...")

                val similarTracks = engine.findSimilarTracks(
                    seedTrackId = matchResult.embeddedTrack.id,
                    numTracks = numTracks,
                    strategy = strategy,
                    anchorExpandConfig = anchorExpandConfig,
                    drift = drift,
                    onProgress = { message ->
                        _uiState.value = RadioUiState.Loading(message)
                        updateNotification(message)
                    }
                )

                Log.d(TAG, "Found ${similarTracks.size} similar tracks")

                if (similarTracks.isEmpty()) {
                    _uiState.value = RadioUiState.Error("No similar tracks found")
                    updateNotification("No similar tracks found")
                    toast("No similar tracks found")
                    stopSelfDelayed()
                    return@launch
                }

                // Map to Poweramp file IDs (preserving rank order)
                val mappedTracks = matcher.mapSimilarTracksToFileIds(this@RadioService, similarTracks)

                // Get file IDs for tracks that were found
                val fileIds = mappedTracks.mapNotNull { it.fileId }

                if (fileIds.isEmpty()) {
                    _uiState.value = RadioUiState.Error("Could not find tracks in Poweramp library")
                    updateNotification("Could not find tracks in Poweramp library")
                    toast("Could not find tracks in Poweramp library")
                    stopSelfDelayed()
                    return@launch
                }

                // Replace queue, preserving current entry if playing from queue
                val queuedCount = PowerampHelper.replaceQueue(this@RadioService, currentTrack.realId, fileIds)
                val queuedFileIds = fileIds.take(queuedCount).toSet()

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
                    availableModels = availableModels,
                    strategy = strategy,
                    queuedFileIds = queuedFileIds
                )

                _uiState.value = RadioUiState.Success(radioResult)
                _sessionHistory.value = _sessionHistory.value + radioResult

                // Notify Poweramp to reload queue data.
                // Poweramp will switch to the queue after the current song ends.
                PowerampHelper.reloadData(this@RadioService)

                val message = "${radioResult.queuedCount} tracks queued"
                updateNotification(message)
                Log.d(TAG, "Queue result: ${radioResult.queuedCount} queued / ${radioResult.failedCount} failed (${strategy.name})")
                Toast.makeText(this@RadioService, message, Toast.LENGTH_SHORT).show()

                // Stop after a short delay
                stopSelfDelayed()

            } catch (e: Exception) {
                Log.e(TAG, "Error starting radio", e)
                _uiState.value = RadioUiState.Error("Error: ${e.message}")
                updateNotification("Error: ${e.message}")
                toast("Error: ${e.message}")
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
        return SimilarityEngine(db, filesDir).also { similarityEngine = it }
    }

    private fun toast(message: String) {
        if (showToasts) {
            Toast.makeText(this, message, Toast.LENGTH_SHORT).show()
        }
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
    }
}
