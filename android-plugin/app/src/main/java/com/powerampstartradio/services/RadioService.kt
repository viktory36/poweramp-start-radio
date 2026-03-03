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
import com.powerampstartradio.data.EmbeddedTrack
import com.powerampstartradio.data.EmbeddingDatabase
import com.powerampstartradio.poweramp.PowerampHelper
import com.powerampstartradio.poweramp.PowerampReceiver
import com.powerampstartradio.poweramp.PowerampTrack
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
import com.powerampstartradio.ui.SeedSpec
import com.powerampstartradio.ui.SelectionMode
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
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
        private const val HISTORY_FILE = "session_history.json"
        private const val MAX_SESSIONS = 200

        const val ACTION_START_RADIO = "com.powerampstartradio.START_RADIO"
        const val ACTION_STOP = "com.powerampstartradio.STOP"
        const val ACTION_CANCEL = "com.powerampstartradio.CANCEL"

        // Intent extras for RadioConfig
        const val EXTRA_SEED_TRACK_ID = "seed_track_id"
        const val EXTRA_NUM_TRACKS = "num_tracks"
        const val EXTRA_SHOW_TOASTS = "show_toasts"
        const val EXTRA_SELECTION_MODE = "selection_mode"
        const val EXTRA_DRIFT_ENABLED = "drift_enabled"
        const val EXTRA_DRIFT_MODE = "drift_mode"
        const val EXTRA_ANCHOR_STRENGTH = "anchor_strength"
        const val EXTRA_ANCHOR_DECAY = "anchor_decay"
        const val EXTRA_MOMENTUM_BETA = "momentum_beta"
        const val EXTRA_WALK_RESTART_ALPHA = "walk_restart_alpha"
        const val EXTRA_DIVERSITY_LAMBDA = "diversity_lambda"
        const val EXTRA_MAX_PER_ARTIST = "max_per_artist"
        const val EXTRA_MIN_ARTIST_SPACING = "min_artist_spacing"
        const val EXTRA_MULTI_SEED = "multi_seed"
        const val ACTION_START_MULTI_SEED = "com.powerampstartradio.START_MULTI_SEED"
        const val ACTION_QUEUE_DIRECTLY = "com.powerampstartradio.QUEUE_DIRECTLY"
        const val DEFAULT_NUM_TRACKS = 50

        /** Transient seed list for multi-seed radio (too large for Intent extras). */
        @Volatile
        var pendingMultiSeeds: List<SeedSpec>? = null

        /** Transient track list for direct queueing (embedding DB track IDs). */
        @Volatile
        var pendingDirectQueue: List<EmbeddedTrack>? = null

        private var activeJob: Job? = null
        val isSearchActive: Boolean get() = activeJob?.isActive == true

        private val _uiState = MutableStateFlow<RadioUiState>(RadioUiState.Idle)
        val uiState: StateFlow<RadioUiState> = _uiState.asStateFlow()

        private val _sessionHistory = MutableStateFlow<List<RadioResult>>(emptyList())
        val sessionHistory: StateFlow<List<RadioResult>> = _sessionHistory.asStateFlow()

        /** Drift reference embeddings for lazy rank computation, keyed by track ID. */
        val driftReferences = MutableStateFlow<Map<Long, FloatArray>>(emptyMap())

        // --- Session history persistence ---
        private var historyDir: File? = null
        private val gson = Gson()
        private val historyType = object : TypeToken<List<RadioResult>>() {}.type
        private val saveScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

        fun initHistory(filesDir: File) {
            historyDir = filesDir
            try {
                val file = File(filesDir, HISTORY_FILE)
                if (file.exists()) {
                    val json = file.readText()
                    val loaded: List<RadioResult> = gson.fromJson(json, historyType) ?: emptyList()
                    _sessionHistory.value = loaded.takeLast(MAX_SESSIONS)
                    Log.d(TAG, "Loaded ${_sessionHistory.value.size} sessions from disk")
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to load session history", e)
                _sessionHistory.value = emptyList()
            }
        }

        private fun saveHistory() {
            val dir = historyDir ?: return
            val snapshot = _sessionHistory.value
            saveScope.launch {
                try {
                    val file = File(dir, HISTORY_FILE)
                    file.writeText(gson.toJson(snapshot, historyType))
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to save session history", e)
                }
            }
        }

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
                putExtra(EXTRA_WALK_RESTART_ALPHA, config.walkRestartAlpha)
                putExtra(EXTRA_DIVERSITY_LAMBDA, config.diversityLambda)
                putExtra(EXTRA_MAX_PER_ARTIST, config.maxPerArtist)
                putExtra(EXTRA_MIN_ARTIST_SPACING, config.minArtistSpacing)
                // candidatePoolSize is auto-computed in performRadio
            }
            context.startForegroundService(intent)
        }

        /**
         * Start radio from a specific seed track ID (e.g. from text search).
         * Bypasses Poweramp current-track matching — the seed is known.
         */
        fun startRadioFromSeed(context: Context, seedTrackId: Long, config: RadioConfig) {
            if (isSearchActive) return
            _uiState.value = RadioUiState.Searching("Starting radio...")
            val intent = Intent(context, RadioService::class.java).apply {
                action = ACTION_START_RADIO
                putExtra(EXTRA_SEED_TRACK_ID, seedTrackId)
                putExtra(EXTRA_NUM_TRACKS, config.numTracks)
                putExtra(EXTRA_SHOW_TOASTS, true)
                putExtra(EXTRA_SELECTION_MODE, config.selectionMode.name)
                putExtra(EXTRA_DRIFT_ENABLED, config.driftEnabled)
                putExtra(EXTRA_DRIFT_MODE, config.driftMode.name)
                putExtra(EXTRA_ANCHOR_STRENGTH, config.anchorStrength)
                putExtra(EXTRA_ANCHOR_DECAY, config.anchorDecay.name)
                putExtra(EXTRA_MOMENTUM_BETA, config.momentumBeta)
                putExtra(EXTRA_WALK_RESTART_ALPHA, config.walkRestartAlpha)
                putExtra(EXTRA_DIVERSITY_LAMBDA, config.diversityLambda)
                putExtra(EXTRA_MAX_PER_ARTIST, config.maxPerArtist)
                putExtra(EXTRA_MIN_ARTIST_SPACING, config.minArtistSpacing)
            }
            context.startForegroundService(intent)
        }

        /**
         * Start radio from multiple seeds (text + song references).
         * Seeds are passed via [pendingMultiSeeds] (too large for Intent extras).
         */
        fun startRadioFromMultiSeed(context: Context, seeds: List<SeedSpec>, config: RadioConfig) {
            if (isSearchActive) return
            pendingMultiSeeds = seeds
            _uiState.value = RadioUiState.Searching("Starting multi-seed search...")
            val intent = Intent(context, RadioService::class.java).apply {
                action = ACTION_START_MULTI_SEED
                putExtra(EXTRA_NUM_TRACKS, config.numTracks)
                putExtra(EXTRA_SHOW_TOASTS, true)
                putExtra(EXTRA_MAX_PER_ARTIST, config.maxPerArtist)
                putExtra(EXTRA_MIN_ARTIST_SPACING, config.minArtistSpacing)
            }
            context.startForegroundService(intent)
        }

        /**
         * Queue a pre-computed list of tracks directly into Poweramp.
         * No recommendation engine — just map and queue.
         */
        fun queueDirectly(context: Context, tracks: List<EmbeddedTrack>, label: String) {
            if (isSearchActive) cancelSearch()
            pendingDirectQueue = tracks
            _uiState.value = RadioUiState.Searching("Adding to queue...")
            val intent = Intent(context, RadioService::class.java).apply {
                action = ACTION_QUEUE_DIRECTLY
                putExtra("label", label)
            }
            context.startForegroundService(intent)
        }

        fun cancelSearch() {
            activeJob?.cancel()
            val current = _uiState.value
            if (current is RadioUiState.Streaming) {
                val completed = current.result.copy(isComplete = true)
                _uiState.value = RadioUiState.Success(completed)
                _sessionHistory.value = (_sessionHistory.value + completed).takeLast(MAX_SESSIONS)
                saveHistory()
            } else {
                _uiState.value = RadioUiState.Idle
            }
        }

        fun resetState() {
            _uiState.value = RadioUiState.Idle
        }

        fun clearHistory() {
            _sessionHistory.value = emptyList()
            saveHistory()
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
                val seedTrackId = intent.getLongExtra(EXTRA_SEED_TRACK_ID, -1L)
                    .takeIf { it >= 0 }
                startForeground(NOTIFICATION_ID, createNotification("Starting radio..."))
                performRadio(config, seedTrackId)
            }
            ACTION_START_MULTI_SEED -> {
                stopJob?.cancel()
                stopJob = null
                showToasts = intent.getBooleanExtra(EXTRA_SHOW_TOASTS, true)
                val config = extractConfig(intent)
                val seeds = pendingMultiSeeds
                pendingMultiSeeds = null
                startForeground(NOTIFICATION_ID, createNotification("Multi-seed search..."))
                if (seeds != null) {
                    performMultiSeedRadio(config, seeds)
                } else {
                    _uiState.value = RadioUiState.Error("No seeds provided")
                    stopSelfDelayed()
                }
            }
            ACTION_QUEUE_DIRECTLY -> {
                stopJob?.cancel()
                stopJob = null
                showToasts = true
                val tracks = pendingDirectQueue
                pendingDirectQueue = null
                val label = intent.getStringExtra("label") ?: "Direct queue"
                startForeground(NOTIFICATION_ID, createNotification("Adding to queue..."))
                if (tracks != null) {
                    performDirectQueue(tracks, label)
                } else {
                    _uiState.value = RadioUiState.Error("No tracks provided")
                    stopSelfDelayed()
                }
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
            driftEnabled = intent.getBooleanExtra(EXTRA_DRIFT_ENABLED, false),
            driftMode = try {
                DriftMode.valueOf(intent.getStringExtra(EXTRA_DRIFT_MODE) ?: DriftMode.SEED_INTERPOLATION.name)
            } catch (e: IllegalArgumentException) { DriftMode.SEED_INTERPOLATION },
            anchorStrength = intent.getFloatExtra(EXTRA_ANCHOR_STRENGTH, 0.5f),
            anchorDecay = try {
                DecaySchedule.valueOf(intent.getStringExtra(EXTRA_ANCHOR_DECAY) ?: DecaySchedule.EXPONENTIAL.name)
            } catch (e: IllegalArgumentException) { DecaySchedule.EXPONENTIAL },
            momentumBeta = intent.getFloatExtra(EXTRA_MOMENTUM_BETA, 0.7f),
            walkRestartAlpha = intent.getFloatExtra(EXTRA_WALK_RESTART_ALPHA, 0.5f),
            diversityLambda = intent.getFloatExtra(EXTRA_DIVERSITY_LAMBDA, 0.4f),
            maxPerArtist = intent.getIntExtra(EXTRA_MAX_PER_ARTIST, 8),
            minArtistSpacing = intent.getIntExtra(EXTRA_MIN_ARTIST_SPACING, 3),
            // candidatePoolSize auto-computed in performRadio
        )
    }

    private fun performRadio(config: RadioConfig, overrideSeedTrackId: Long? = null) {
        activeJob = serviceScope.launch {
            try {
                val radioStart = System.nanoTime()
                toast("Starting radio...")
                Log.i(TAG, "performRadio: mode=${config.selectionMode.name}, " +
                    "drift=${config.driftEnabled}, numTracks=${config.numTracks}, " +
                    "seedOverride=$overrideSeedTrackId")

                val db = getOrCreateDatabase()
                if (db == null) {
                    _uiState.value = RadioUiState.Error("No embedding database found")
                    updateNotification("No embedding database found")
                    toast("No embedding database found")
                    stopSelfDelayed()
                    return@launch
                }

                val matcher = TrackMatcher(db)

                // Resolve seed: either from override (text search) or current Poweramp track
                val seedTrackId: Long
                val seedDisplayTrack: PowerampTrack
                val matchType: TrackMatcher.MatchType
                // For text search: the seed file ID to prepend to recommendations
                var textSearchSeedFileId: Long? = null
                // For text search: the currently-playing track to anchor queue pos 0 (Poweramp workaround)
                var textSearchQueueAnchorId: Long? = null

                if (overrideSeedTrackId != null) {
                    // Text search path: seed track ID is known
                    seedTrackId = overrideSeedTrackId
                    val track = db.getTrackById(seedTrackId)
                    if (track == null) {
                        _uiState.value = RadioUiState.Error("Seed track not found in database")
                        updateNotification("Seed track not found")
                        stopSelfDelayed()
                        return@launch
                    }
                    // Resolve seed to Poweramp file ID
                    textSearchSeedFileId = matcher.findFileId(this@RadioService, track)

                    // Display: always show the text search pick as seed in UI/session history.
                    seedDisplayTrack = PowerampTrack(
                        realId = textSearchSeedFileId ?: -1L,
                        title = track.title ?: "Unknown",
                        artist = track.artist,
                        album = track.album,
                        durationMs = track.durationMs,
                        path = track.filePath,
                    )
                    // Queue: if currently in a Poweramp queue, keep that track at pos 0
                    // (Poweramp breaks if we replace the queue without the current track).
                    // Seed goes at pos 1, then recommendations.
                    // If NOT in a queue, seed goes first — no preservation needed.
                    val currentTrack = PowerampReceiver.currentTrack
                    val currentInQueue = currentTrack?.realId?.takeIf { it > 0 }?.let {
                        PowerampHelper.isInQueue(this@RadioService, it)
                    } == true
                    textSearchQueueAnchorId = if (currentInQueue) currentTrack!!.realId else null
                    matchType = TrackMatcher.MatchType.METADATA_EXACT
                    Log.d(TAG, "Text search seed: ${track.title} by ${track.artist} " +
                            "(seedFileId=$textSearchSeedFileId, queueAnchor=$textSearchQueueAnchorId)")
                } else {
                    // Normal path: match current Poweramp track
                    val currentTrack = PowerampReceiver.currentTrack
                    if (currentTrack == null) {
                        _uiState.value = RadioUiState.Error("No track playing in Poweramp")
                        updateNotification("No track playing in Poweramp")
                        toast("No track playing in Poweramp")
                        stopSelfDelayed()
                        return@launch
                    }

                    val matchResult = matcher.findMatch(currentTrack)
                    if (matchResult == null) {
                        _uiState.value = RadioUiState.Error("Track not found in database")
                        updateNotification("Track not found in database")
                        toast("Track not found in database")
                        stopSelfDelayed()
                        return@launch
                    }

                    seedTrackId = matchResult.embeddedTrack.id
                    seedDisplayTrack = currentTrack
                    matchType = matchResult.matchType
                    Log.d(TAG, "Found match (${matchResult.matchType}): ${matchResult.embeddedTrack.title}")
                }

                updateNotification("Finding similar tracks to: ${seedDisplayTrack.title}")
                Log.d(TAG, "Config: ${config.selectionMode.name}" +
                    (if (config.driftEnabled) " drift(${config.driftMode.name})" else "") +
                    " lambda=${config.diversityLambda}")

                val eng = getOrCreateEngine(db)
                val tIndices = System.nanoTime()
                eng.ensureIndices { message ->
                    _uiState.value = RadioUiState.Loading(message)
                    updateNotification(message)
                }
                val indicesMs = (System.nanoTime() - tIndices) / 1_000_000
                Log.d(TAG, "ensureIndices: ${indicesMs}ms")

                // Resolve auto pool size so RadioResult carries the actual value
                val resolvedConfig = if (config.candidatePoolSize <= 0) {
                    val autoPool = (db.getTrackCount() * 0.02f).toInt().coerceAtLeast(100)
                    config.copy(candidatePoolSize = autoPool)
                } else config

                updateNotification("Searching for similar tracks...")

                // DPP+drift is degenerate (forced to batch in RecommendationEngine),
                // so use batch path here too to avoid streaming/return-value mismatch.
                val effectiveDrift = resolvedConfig.driftEnabled && resolvedConfig.selectionMode != SelectionMode.DPP
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
                                    // For text search: prepend seed, anchor current track for Poweramp
                                    val firstBatch = if (textSearchSeedFileId != null) {
                                        listOf(textSearchSeedFileId) + batch.filter { it != textSearchSeedFileId }
                                    } else {
                                        batch
                                    }
                                    val driftCurrentId = textSearchQueueAnchorId ?: seedDisplayTrack.realId
                                    PowerampHelper.replaceQueue(this@RadioService, driftCurrentId, firstBatch)
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

                    // Clear drift references for new session
                    driftReferences.value = emptyMap()

                    _uiState.value = RadioUiState.Streaming(RadioResult(
                        seedTrack = seedDisplayTrack,
                        matchType = matchType,
                        tracks = emptyList(),
                        config = resolvedConfig,
                        queueAnchorId = textSearchQueueAnchorId,
                        isComplete = false,
                        totalExpected = resolvedConfig.numTracks
                    ))

                    eng.generatePlaylist(
                        seedTrackId = seedTrackId,
                        config = resolvedConfig,
                        onProgress = { message ->
                            updateNotification(message)
                        },
                        onResult = { similarTrack ->
                            // Map to Poweramp file ID (fast HashMap after first cache build)
                            val fileId = matcher.mapSingleTrackToFileId(
                                this@RadioService, similarTrack, seenFileIds
                            )

                            // Store drift reference for lazy rank computation
                            similarTrack.driftReferenceEmb?.let { ref ->
                                driftReferences.value = driftReferences.value + (similarTrack.track.id to ref)
                            }

                            streamingTracks.add(QueuedTrackResult(
                                track = similarTrack.track,
                                similarity = similarTrack.similarity,
                                similarityToSeed = similarTrack.similarityToSeed,
                                candidateRank = similarTrack.candidateRank,
                                seedRank = similarTrack.seedRank,
                                driftRank = similarTrack.driftRank,
                                graphHops = similarTrack.graphHops,
                                status = if (fileId != null) QueueStatus.QUEUED else QueueStatus.NOT_IN_LIBRARY,
                                provenance = similarTrack.provenance,
                            ))

                            _uiState.value = RadioUiState.Streaming(RadioResult(
                                seedTrack = seedDisplayTrack,
                                matchType = matchType,
                                tracks = streamingTracks.toList(),
                                config = resolvedConfig,
                                queueAnchorId = textSearchQueueAnchorId,
                                isComplete = false,
                                totalExpected = resolvedConfig.numTracks
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

                    // Compute queue quality metrics from embeddings
                    val driftSimilarTracks = streamingTracks.map { qtr ->
                        SimilarTrack(
                            track = qtr.track, similarity = qtr.similarity,
                            similarityToSeed = qtr.similarityToSeed
                        )
                    }
                    val driftMetrics = eng.computeQueueMetrics(driftSimilarTracks)

                    val finalResult = RadioResult(
                        seedTrack = seedDisplayTrack,
                        matchType = matchType,
                        tracks = streamingTracks.toList(),
                        config = resolvedConfig,
                        queuedFileIds = queuedFileIds.toSet(),
                        queueAnchorId = textSearchQueueAnchorId,
                        isComplete = true,
                        totalExpected = resolvedConfig.numTracks,
                        metrics = driftMetrics
                    )

                    _uiState.value = RadioUiState.Success(finalResult)
                    _sessionHistory.value = (_sessionHistory.value + finalResult).takeLast(MAX_SESSIONS)
                    saveHistory()

                    PowerampHelper.reloadData(this@RadioService)

                    val notFound = streamingTracks.count { it.status == QueueStatus.NOT_IN_LIBRARY }
                    val message = buildQueueResultMessage(finalResult.queuedCount, notFound)
                    updateNotification(message)
                    val totalMs = (System.nanoTime() - radioStart) / 1_000_000
                    Log.i(TAG, "TIMING: radio_drift total=${totalMs}ms, " +
                        "${finalResult.queuedCount} queued / ${finalResult.failedCount} failed / $notFound not found")
                    Toast.makeText(this@RadioService, message, Toast.LENGTH_SHORT).show()

                } else {
                    // Non-drift path: batch search
                    _uiState.value = RadioUiState.Searching("Searching...")

                    val similarTracks = eng.generatePlaylist(
                        seedTrackId = seedTrackId,
                        config = resolvedConfig,
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

                    val tMap = System.nanoTime()
                    val mappedTracks = matcher.mapSimilarTracksToFileIds(this@RadioService, similarTracks)
                    val fileIds = mappedTracks.mapNotNull { it.fileId }
                    val mapMs = (System.nanoTime() - tMap) / 1_000_000
                    Log.d(TAG, "Track mapping: ${similarTracks.size} → ${fileIds.size} file IDs in ${mapMs}ms")

                    if (fileIds.isEmpty()) {
                        _uiState.value = RadioUiState.Error("Could not find tracks in Poweramp library")
                        updateNotification("Could not find tracks in Poweramp library")
                        toast("Could not find tracks in Poweramp library")
                        stopSelfDelayed()
                        return@launch
                    }

                    // For text search: prepend seed to recommendations.
                    // If in queue, anchor current track at pos 0 for Poweramp stability.
                    // Queue: [anchor?, seed, rec1, rec2, ...]
                    val allFileIds = if (textSearchSeedFileId != null) {
                        listOf(textSearchSeedFileId) + fileIds.filter { it != textSearchSeedFileId }
                    } else {
                        fileIds
                    }

                    val queueCurrentId = textSearchQueueAnchorId ?: seedDisplayTrack.realId
                    val tQueue = System.nanoTime()
                    val queuedCount = PowerampHelper.replaceQueue(this@RadioService, queueCurrentId, allFileIds)
                    val queueMs = (System.nanoTime() - tQueue) / 1_000_000
                    Log.d(TAG, "Poweramp queue: ${queuedCount}/${allFileIds.size} queued in ${queueMs}ms")
                    val queuedFileIds = allFileIds.take(queuedCount).toSet()

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
                            seedRank = mapped.similarTrack.seedRank,
                            driftRank = mapped.similarTrack.driftRank,
                            graphHops = mapped.similarTrack.graphHops,
                            status = status,
                            provenance = mapped.similarTrack.provenance,
                        )
                    }

                    // Compute queue quality metrics from embeddings
                    val batchMetrics = eng.computeQueueMetrics(similarTracks)

                    val radioResult = RadioResult(
                        seedTrack = seedDisplayTrack,
                        matchType = matchType,
                        tracks = trackResults,
                        config = resolvedConfig,
                        queuedFileIds = queuedFileIds,
                        queueAnchorId = textSearchQueueAnchorId,
                        metrics = batchMetrics
                    )

                    _uiState.value = RadioUiState.Success(radioResult)
                    _sessionHistory.value = (_sessionHistory.value + radioResult).takeLast(MAX_SESSIONS)
                    saveHistory()

                    PowerampHelper.reloadData(this@RadioService)


                    val notFound = trackResults.count { it.status == QueueStatus.NOT_IN_LIBRARY }
                    val message = buildQueueResultMessage(radioResult.queuedCount, notFound)
                    updateNotification(message)
                    val totalMs = (System.nanoTime() - radioStart) / 1_000_000
                    Log.i(TAG, "TIMING: radio_batch total=${totalMs}ms, " +
                        "${radioResult.queuedCount} queued / ${radioResult.failedCount} failed / $notFound not found")
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

    private fun performMultiSeedRadio(config: RadioConfig, seeds: List<SeedSpec>) {
        activeJob = serviceScope.launch {
            try {
                val radioStart = System.nanoTime()
                toast("Multi-seed search...")
                Log.i(TAG, "performMultiSeedRadio: ${seeds.size} seeds, numTracks=${config.numTracks}")

                val db = getOrCreateDatabase()
                if (db == null) {
                    _uiState.value = RadioUiState.Error("No embedding database found")
                    stopSelfDelayed()
                    return@launch
                }

                val matcher = TrackMatcher(db)
                val eng = getOrCreateEngine(db)

                eng.ensureIndices { message ->
                    _uiState.value = RadioUiState.Loading(message)
                    updateNotification(message)
                }

                _uiState.value = RadioUiState.Searching("Computing multi-seed ranking...")
                updateNotification("Computing multi-seed ranking...")

                val similarTracks = eng.generateMultiSeedPlaylist(
                    seeds = seeds,
                    config = config,
                    onProgress = { message ->
                        _uiState.value = RadioUiState.Searching(message)
                        updateNotification(message)
                    }
                )

                if (similarTracks.isEmpty()) {
                    _uiState.value = RadioUiState.Error("No similar tracks found")
                    toast("No similar tracks found")
                    stopSelfDelayed()
                    return@launch
                }

                // Map to Poweramp file IDs
                val mappedTracks = matcher.mapSimilarTracksToFileIds(this@RadioService, similarTracks)
                val fileIds = mappedTracks.mapNotNull { it.fileId }

                if (fileIds.isEmpty()) {
                    _uiState.value = RadioUiState.Error("Could not find tracks in Poweramp library")
                    toast("Could not find tracks in Poweramp library")
                    stopSelfDelayed()
                    return@launch
                }

                // Build display seed track from the first labeled seed
                val displayLabel = seeds.firstOrNull()?.label ?: "Multi-seed"
                val seedDisplayTrack = PowerampTrack(
                    realId = -1L,
                    title = displayLabel,
                    artist = seeds.drop(1).joinToString(", ") { it.label }.ifEmpty { null },
                    album = null,
                    durationMs = 0,
                    path = "",
                )

                // Queue: anchor current Poweramp track if in queue, else just queue results
                val currentTrack = PowerampReceiver.currentTrack
                val currentInQueue = currentTrack?.realId?.takeIf { it > 0 }?.let {
                    PowerampHelper.isInQueue(this@RadioService, it)
                } == true
                val queueAnchorId = if (currentInQueue) currentTrack!!.realId else null
                val queueCurrentId = queueAnchorId ?: (currentTrack?.realId ?: -1L)

                val queuedCount = PowerampHelper.replaceQueue(this@RadioService, queueCurrentId, fileIds)
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
                        status = status,
                    )
                }

                val metrics = eng.computeQueueMetrics(similarTracks)
                val radioResult = RadioResult(
                    seedTrack = seedDisplayTrack,
                    matchType = TrackMatcher.MatchType.METADATA_EXACT,
                    tracks = trackResults,
                    config = config,
                    queuedFileIds = queuedFileIds,
                    queueAnchorId = queueAnchorId,
                    metrics = metrics,
                )

                _uiState.value = RadioUiState.Success(radioResult)
                _sessionHistory.value = (_sessionHistory.value + radioResult).takeLast(MAX_SESSIONS)
                saveHistory()
                PowerampHelper.reloadData(this@RadioService)

                val notFound = trackResults.count { it.status == QueueStatus.NOT_IN_LIBRARY }
                val message = buildQueueResultMessage(radioResult.queuedCount, notFound)
                updateNotification(message)
                val totalMs = (System.nanoTime() - radioStart) / 1_000_000
                Log.i(TAG, "TIMING: radio_multiseed total=${totalMs}ms, " +
                    "${radioResult.queuedCount} queued / ${radioResult.failedCount} failed / $notFound not found")
                Toast.makeText(this@RadioService, message, Toast.LENGTH_SHORT).show()

                stopSelfDelayed()
            } catch (e: CancellationException) {
                Log.d(TAG, "Multi-seed search cancelled")
                stopSelfDelayed()
            } catch (e: Exception) {
                Log.e(TAG, "Error in multi-seed radio", e)
                _uiState.value = RadioUiState.Error("Error: ${e.message}")
                toast("Error: ${e.message}")
                stopSelfDelayed()
            } finally {
                activeJob = null
            }
        }
    }

    /**
     * Directly queue a pre-computed list of tracks into Poweramp.
     * Maps embedding DB tracks to Poweramp file IDs via TrackMatcher, then queues.
     */
    private fun performDirectQueue(tracks: List<EmbeddedTrack>, label: String) {
        activeJob = serviceScope.launch {
            try {
                val radioResult = kotlinx.coroutines.withContext(Dispatchers.IO) {
                    val db = getOrCreateDatabase()
                        ?: return@withContext null
                    val matcher = TrackMatcher(db)

                    Log.i(TAG, "DIRECT_QUEUE: queueing ${tracks.size} tracks, label='$label'")
                    for ((i, t) in tracks.withIndex()) {
                        Log.d(TAG, "DIRECT_QUEUE: [$i] ${t.artist} - ${t.title} (id=${t.id})")
                    }

                    // Map to Poweramp file IDs and build track results
                    val fileIds = mutableListOf<Long>()
                    val trackResults = mutableListOf<QueuedTrackResult>()
                    var notFound = 0
                    for (track in tracks) {
                        val fileId = matcher.findFileId(this@RadioService, track)
                        val status = if (fileId != null) {
                            fileIds.add(fileId)
                            QueueStatus.QUEUED
                        } else {
                            notFound++
                            Log.w(TAG, "DIRECT_QUEUE: no Poweramp file for ${track.artist} - ${track.title}")
                            QueueStatus.NOT_IN_LIBRARY
                        }
                        trackResults.add(QueuedTrackResult(
                            track = track,
                            similarity = 0f,
                            similarityToSeed = 0f,
                            status = status,
                        ))
                    }

                    if (fileIds.isEmpty()) return@withContext null

                    // Queue: anchor current Poweramp track if in queue
                    val currentTrack = PowerampReceiver.currentTrack
                    val currentInQueue = currentTrack?.realId?.takeIf { it > 0 }?.let {
                        PowerampHelper.isInQueue(this@RadioService, it)
                    } == true
                    val queueAnchorId = if (currentInQueue) currentTrack!!.realId else null
                    val queueCurrentId = queueAnchorId ?: (currentTrack?.realId ?: -1L)

                    val queuedCount = PowerampHelper.replaceQueue(this@RadioService, queueCurrentId, fileIds)
                    PowerampHelper.reloadData(this@RadioService)

                    Log.i(TAG, "DIRECT_QUEUE: queued $queuedCount of ${fileIds.size} tracks " +
                        "($notFound not in Poweramp library)")

                    // Build RadioResult with synthetic seed
                    val syntheticSeed = PowerampTrack(
                        realId = -1L,
                        title = label,
                        artist = null,
                        album = null,
                        durationMs = 0,
                        path = null,
                    )
                    RadioResult(
                        seedTrack = syntheticSeed,
                        matchType = TrackMatcher.MatchType.METADATA_EXACT,
                        tracks = trackResults,
                        queuedFileIds = fileIds.toSet(),
                        queueAnchorId = queueAnchorId,
                        isDirectQueue = true,
                    )
                }

                if (radioResult == null) {
                    _uiState.value = RadioUiState.Error("Could not find tracks in Poweramp library")
                    toast("Could not find tracks in Poweramp library")
                    stopSelfDelayed()
                    return@launch
                }

                _sessionHistory.value = (_sessionHistory.value + radioResult).takeLast(MAX_SESSIONS)
                saveHistory()

                val message = buildQueueResultMessage(
                    radioResult.queuedCount,
                    radioResult.tracks.count { it.status == QueueStatus.NOT_IN_LIBRARY }
                )
                updateNotification(message)
                Toast.makeText(this@RadioService, message, Toast.LENGTH_SHORT).show()
                _uiState.value = RadioUiState.Success(radioResult)
                stopSelfDelayed()
            } catch (e: CancellationException) {
                Log.d(TAG, "Direct queue cancelled")
                stopSelfDelayed()
            } catch (e: Exception) {
                Log.e(TAG, "Error in direct queue", e)
                _uiState.value = RadioUiState.Error("Error: ${e.message}")
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

    private fun buildQueueResultMessage(queuedCount: Int, notFoundCount: Int): String {
        val base = "$queuedCount tracks queued"
        return if (notFoundCount > 0) "$base ($notFoundCount not found)" else base
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
