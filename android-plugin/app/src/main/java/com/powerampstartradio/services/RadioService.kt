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
import com.powerampstartradio.poweramp.PowerampHelper
import com.powerampstartradio.poweramp.PowerampReceiver
import com.powerampstartradio.poweramp.TrackMatcher
import com.powerampstartradio.similarity.RecommendationEngine
import com.powerampstartradio.similarity.SimilarTrack
import com.powerampstartradio.ui.DecaySchedule
import com.powerampstartradio.ui.DriftMode
import com.powerampstartradio.ui.QueueStatus
import com.powerampstartradio.ui.QueuedTrackResult
import com.powerampstartradio.ui.RadioConfig
import com.powerampstartradio.ui.RadioResult
import com.powerampstartradio.ui.RadioUiState
import com.powerampstartradio.ui.SelectionMode
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.launch
import java.io.File

/**
 * Foreground service that handles the "Start Radio" functionality.
 *
 * Flow:
 * 1. Get current track from Poweramp
 * 2. Match to embedding database
 * 3. Find similar tracks using RecommendationEngine
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
        const val ACTION_CANCEL = "com.powerampstartradio.CANCEL"

        // Intent extras for RadioConfig
        const val EXTRA_NUM_TRACKS = "num_tracks"
        const val EXTRA_SHOW_TOASTS = "show_toasts"
        const val EXTRA_SELECTION_MODE = "selection_mode"
        const val EXTRA_DRIFT_ENABLED = "drift_enabled"
        const val EXTRA_DRIFT_MODE = "drift_mode"
        const val EXTRA_ANCHOR_STRENGTH = "anchor_strength"
        const val EXTRA_ANCHOR_DECAY = "anchor_decay"
        const val EXTRA_MOMENTUM_BETA = "momentum_beta"
        const val EXTRA_DIVERSITY_LAMBDA = "diversity_lambda"
        const val EXTRA_TEMPERATURE = "temperature"
        const val EXTRA_MAX_PER_ARTIST = "max_per_artist"
        const val EXTRA_MIN_ARTIST_SPACING = "min_artist_spacing"
        const val EXTRA_CANDIDATE_POOL_SIZE = "candidate_pool_size"

        const val DEFAULT_NUM_TRACKS = 50

        private var activeJob: Job? = null
        val isSearchActive: Boolean get() = activeJob?.isActive == true

        private val _uiState = MutableStateFlow<RadioUiState>(RadioUiState.Idle)
        val uiState: StateFlow<RadioUiState> = _uiState.asStateFlow()

        private val _sessionHistory = MutableStateFlow<List<RadioResult>>(emptyList())
        val sessionHistory: StateFlow<List<RadioResult>> = _sessionHistory.asStateFlow()

        fun startRadio(context: Context, config: RadioConfig, showToasts: Boolean = false) {
            if (isSearchActive) return
            _uiState.value = RadioUiState.Loading()
            val intent = Intent(context, RadioService::class.java).apply {
                action = ACTION_START_RADIO
                putExtra(EXTRA_NUM_TRACKS, config.numTracks)
                putExtra(EXTRA_SHOW_TOASTS, showToasts)
                putExtra(EXTRA_SELECTION_MODE, config.selectionMode.name)
                putExtra(EXTRA_DRIFT_ENABLED, config.driftEnabled)
                putExtra(EXTRA_DRIFT_MODE, config.driftMode.name)
                putExtra(EXTRA_ANCHOR_STRENGTH, config.anchorStrength)
                putExtra(EXTRA_ANCHOR_DECAY, config.anchorDecay.name)
                putExtra(EXTRA_MOMENTUM_BETA, config.momentumBeta)
                putExtra(EXTRA_DIVERSITY_LAMBDA, config.diversityLambda)
                putExtra(EXTRA_TEMPERATURE, config.temperature)
                putExtra(EXTRA_MAX_PER_ARTIST, config.maxPerArtist)
                putExtra(EXTRA_MIN_ARTIST_SPACING, config.minArtistSpacing)
                putExtra(EXTRA_CANDIDATE_POOL_SIZE, config.candidatePoolSize)
            }
            context.startForegroundService(intent)
        }

        fun cancelSearch() {
            activeJob?.cancel()
            val current = _uiState.value
            if (current is RadioUiState.Streaming) {
                val completed = current.result.copy(isComplete = true)
                _uiState.value = RadioUiState.Success(completed)
                _sessionHistory.value = _sessionHistory.value + completed
            } else {
                _uiState.value = RadioUiState.Idle
            }
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
    private var engine: RecommendationEngine? = null
    private var showToasts: Boolean = false
    private var stopJob: Job? = null

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_START_RADIO -> {
                stopJob?.cancel()
                stopJob = null
                showToasts = intent.getBooleanExtra(EXTRA_SHOW_TOASTS, false)
                val config = extractConfig(intent)
                startForeground(NOTIFICATION_ID, createNotification("Starting radio..."))
                performRadio(config)
            }
            ACTION_CANCEL -> {
                cancelSearch()
                stopSelfDelayed()
            }
            ACTION_STOP -> {
                stopSelf()
            }
        }
        return START_NOT_STICKY
    }

    private fun extractConfig(intent: Intent): RadioConfig {
        return RadioConfig(
            numTracks = intent.getIntExtra(EXTRA_NUM_TRACKS, DEFAULT_NUM_TRACKS),
            selectionMode = try {
                SelectionMode.valueOf(intent.getStringExtra(EXTRA_SELECTION_MODE) ?: SelectionMode.MMR.name)
            } catch (e: IllegalArgumentException) { SelectionMode.MMR },
            driftEnabled = intent.getBooleanExtra(EXTRA_DRIFT_ENABLED, true),
            driftMode = try {
                DriftMode.valueOf(intent.getStringExtra(EXTRA_DRIFT_MODE) ?: DriftMode.SEED_INTERPOLATION.name)
            } catch (e: IllegalArgumentException) { DriftMode.SEED_INTERPOLATION },
            anchorStrength = intent.getFloatExtra(EXTRA_ANCHOR_STRENGTH, 0.5f),
            anchorDecay = try {
                DecaySchedule.valueOf(intent.getStringExtra(EXTRA_ANCHOR_DECAY) ?: DecaySchedule.EXPONENTIAL.name)
            } catch (e: IllegalArgumentException) { DecaySchedule.EXPONENTIAL },
            momentumBeta = intent.getFloatExtra(EXTRA_MOMENTUM_BETA, 0.7f),
            diversityLambda = intent.getFloatExtra(EXTRA_DIVERSITY_LAMBDA, 0.4f),
            temperature = intent.getFloatExtra(EXTRA_TEMPERATURE, 0.05f),
            maxPerArtist = intent.getIntExtra(EXTRA_MAX_PER_ARTIST, 8),
            minArtistSpacing = intent.getIntExtra(EXTRA_MIN_ARTIST_SPACING, 2),
            candidatePoolSize = intent.getIntExtra(EXTRA_CANDIDATE_POOL_SIZE, 200),
        )
    }

    private fun performRadio(config: RadioConfig) {
        activeJob = serviceScope.launch {
            try {
                toast("Starting radio...")

                val currentTrack = PowerampReceiver.currentTrack
                if (currentTrack == null) {
                    _uiState.value = RadioUiState.Error("No track playing in Poweramp")
                    updateNotification("No track playing in Poweramp")
                    toast("No track playing in Poweramp")
                    stopSelfDelayed()
                    return@launch
                }

                updateNotification("Finding similar tracks to: ${currentTrack.title}")
                Log.d(TAG, "Starting radio for: ${currentTrack.title} by ${currentTrack.artist}")
                Log.d(TAG, "Config: ${config.selectionMode.name}" +
                    (if (config.driftEnabled) " drift(${config.driftMode.name})" else "") +
                    " lambda=${config.diversityLambda} T=${config.temperature}")

                val db = getOrCreateDatabase()
                if (db == null) {
                    _uiState.value = RadioUiState.Error("No embedding database found")
                    updateNotification("No embedding database found")
                    toast("No embedding database found")
                    stopSelfDelayed()
                    return@launch
                }

                val matcher = TrackMatcher(db)
                val matchResult = matcher.findMatch(currentTrack)

                if (matchResult == null) {
                    _uiState.value = RadioUiState.Error("Track not found in database")
                    updateNotification("Track not found in database")
                    toast("Track not found in database")
                    stopSelfDelayed()
                    return@launch
                }

                Log.d(TAG, "Found match (${matchResult.matchType}): ${matchResult.embeddedTrack.title}")

                val eng = getOrCreateEngine(db)
                eng.ensureIndices { message ->
                    _uiState.value = RadioUiState.Loading(message)
                    updateNotification(message)
                }

                updateNotification("Searching for similar tracks...")

                // DPP+drift is degenerate (forced to batch in RecommendationEngine),
                // so use batch path here too to avoid streaming/return-value mismatch.
                val effectiveDrift = config.driftEnabled && config.selectionMode != SelectionMode.DPP
                if (effectiveDrift) {
                    // Drift path: stream search results to UI, queue to Poweramp
                    // in background batches (every 5 tracks). Queue ops are decoupled
                    // from the search loop via a Channel so Poweramp can't stall drift.
                    val streamingTracks = mutableListOf<QueuedTrackResult>()
                    val seenFileIds = mutableSetOf<Long>()
                    val pendingFileIds = mutableListOf<Long>()
                    val queuedFileIds = mutableSetOf<Long>()

                    // Background queue consumer — processes batches sequentially
                    val queueChannel = Channel<List<Long>>(Channel.UNLIMITED)
                    val queueJob = serviceScope.launch {
                        var isFirst = true
                        for (batch in queueChannel) {
                            try {
                                val count = if (isFirst) {
                                    isFirst = false
                                    PowerampHelper.replaceQueue(this@RadioService, currentTrack.realId, batch)
                                } else {
                                    PowerampHelper.addTracksToQueue(this@RadioService, batch)
                                }
                                queuedFileIds.addAll(batch.take(count))
                                PowerampHelper.reloadData(this@RadioService)
                            } catch (e: Exception) {
                                Log.e(TAG, "Queue batch failed", e)
                            }
                        }
                    }

                    _uiState.value = RadioUiState.Streaming(RadioResult(
                        seedTrack = currentTrack,
                        matchType = matchResult.matchType,
                        tracks = emptyList(),
                        config = config,
                        isComplete = false,
                        totalExpected = config.numTracks
                    ))

                    eng.generatePlaylist(
                        seedTrackId = matchResult.embeddedTrack.id,
                        config = config,
                        onProgress = { message ->
                            updateNotification(message)
                        },
                        onResult = { similarTrack ->
                            // Map to Poweramp file ID (fast HashMap after first cache build)
                            val fileId = matcher.mapSingleTrackToFileId(
                                this@RadioService, similarTrack, seenFileIds
                            )

                            streamingTracks.add(QueuedTrackResult(
                                track = similarTrack.track,
                                similarity = similarTrack.similarity,
                                similarityToSeed = similarTrack.similarityToSeed,
                                candidateRank = similarTrack.candidateRank,
                                status = if (fileId != null) QueueStatus.QUEUED else QueueStatus.NOT_IN_LIBRARY,
                                provenance = similarTrack.provenance,
                            ))

                            _uiState.value = RadioUiState.Streaming(RadioResult(
                                seedTrack = currentTrack,
                                matchType = matchResult.matchType,
                                tracks = streamingTracks.toList(),
                                config = config,
                                isComplete = false,
                                totalExpected = config.numTracks
                            ))

                            // Batch file IDs for background queuing
                            if (fileId != null) {
                                pendingFileIds.add(fileId)
                                if (pendingFileIds.size >= 5) {
                                    queueChannel.trySend(pendingFileIds.toList())
                                    pendingFileIds.clear()
                                }
                            }
                        }
                    )

                    // Flush remaining file IDs and wait for queue consumer to finish
                    if (pendingFileIds.isNotEmpty()) {
                        queueChannel.trySend(pendingFileIds.toList())
                    }
                    queueChannel.close()
                    queueJob.join()

                    val finalResult = RadioResult(
                        seedTrack = currentTrack,
                        matchType = matchResult.matchType,
                        tracks = streamingTracks.toList(),
                        config = config,
                        queuedFileIds = queuedFileIds.toSet(),
                        isComplete = true,
                        totalExpected = config.numTracks
                    )

                    _uiState.value = RadioUiState.Success(finalResult)
                    _sessionHistory.value = _sessionHistory.value + finalResult

                    PowerampHelper.reloadData(this@RadioService)

                    val message = "${finalResult.queuedCount} tracks queued"
                    updateNotification(message)
                    Log.d(TAG, "Queue result: ${finalResult.queuedCount} queued / ${finalResult.failedCount} failed")
                    Toast.makeText(this@RadioService, message, Toast.LENGTH_SHORT).show()

                } else {
                    // Non-drift path: batch search
                    _uiState.value = RadioUiState.Searching("Searching...")

                    val similarTracks = eng.generatePlaylist(
                        seedTrackId = matchResult.embeddedTrack.id,
                        config = config,
                        onProgress = { message ->
                            _uiState.value = RadioUiState.Searching(message)
                            updateNotification(message)
                        }
                    )

                    if (similarTracks.isEmpty()) {
                        _uiState.value = RadioUiState.Error("No similar tracks found")
                        updateNotification("No similar tracks found")
                        toast("No similar tracks found")
                        stopSelfDelayed()
                        return@launch
                    }

                    val mappedTracks = matcher.mapSimilarTracksToFileIds(this@RadioService, similarTracks)
                    val fileIds = mappedTracks.mapNotNull { it.fileId }

                    if (fileIds.isEmpty()) {
                        _uiState.value = RadioUiState.Error("Could not find tracks in Poweramp library")
                        updateNotification("Could not find tracks in Poweramp library")
                        toast("Could not find tracks in Poweramp library")
                        stopSelfDelayed()
                        return@launch
                    }

                    val queuedCount = PowerampHelper.replaceQueue(this@RadioService, currentTrack.realId, fileIds)
                    val queuedFileIds = fileIds.take(queuedCount).toSet()

                    val trackResults = mappedTracks.map { mapped ->
                        val status = when {
                            mapped.fileId == null -> QueueStatus.NOT_IN_LIBRARY
                            mapped.fileId in queuedFileIds -> QueueStatus.QUEUED
                            else -> QueueStatus.QUEUE_FAILED
                        }
                        QueuedTrackResult(
                            track = mapped.similarTrack.track,
                            similarity = mapped.similarTrack.similarity,
                            similarityToSeed = mapped.similarTrack.similarityToSeed,
                            candidateRank = mapped.similarTrack.candidateRank,
                            status = status,
                            provenance = mapped.similarTrack.provenance,
                        )
                    }

                    val radioResult = RadioResult(
                        seedTrack = currentTrack,
                        matchType = matchResult.matchType,
                        tracks = trackResults,
                        config = config,
                        queuedFileIds = queuedFileIds
                    )

                    _uiState.value = RadioUiState.Success(radioResult)
                    _sessionHistory.value = _sessionHistory.value + radioResult

                    PowerampHelper.reloadData(this@RadioService)

                    val message = "${radioResult.queuedCount} tracks queued"
                    updateNotification(message)
                    Log.d(TAG, "Queue result: ${radioResult.queuedCount} queued / ${radioResult.failedCount} failed")
                    Toast.makeText(this@RadioService, message, Toast.LENGTH_SHORT).show()
                }

                stopSelfDelayed()

            } catch (e: CancellationException) {
                Log.d(TAG, "Radio search cancelled")
                stopSelfDelayed()
            } catch (e: Exception) {
                Log.e(TAG, "Error starting radio", e)
                _uiState.value = RadioUiState.Error("Error: ${e.message}")
                updateNotification("Error: ${e.message}")
                toast("Error: ${e.message}")
                stopSelfDelayed()
            } finally {
                activeJob = null
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

    private fun getOrCreateEngine(db: EmbeddingDatabase): RecommendationEngine {
        engine?.let { return it }
        return RecommendationEngine(db, filesDir).also { engine = it }
    }

    private fun toast(message: String) {
        if (showToasts) {
            Toast.makeText(this, message, Toast.LENGTH_SHORT).show()
        }
    }

    private fun stopSelfDelayed() {
        stopJob?.cancel()
        stopJob = serviceScope.launch {
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
            this, 0,
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
        embeddingDb = null
        engine = null
    }
}
