package com.powerampstartradio.indexing

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.IBinder
import android.os.PowerManager
import android.util.Log
import androidx.core.app.NotificationCompat
import com.powerampstartradio.MainActivity
import com.powerampstartradio.R
import com.powerampstartradio.data.EmbeddingDatabase
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.async
import kotlinx.coroutines.launch
import java.io.File

/**
 * Foreground service for on-device CLaMP3 embedding indexing.
 *
 * Two-phase GPU pipeline (Adreno can't run two OpenCL contexts simultaneously):
 * - Phase 1 (MERT): Decode audio → extract 768d features per 5s window → spill to disk
 * - Phase 2 (CLaMP3): Read MERT features → encode into 768d embedding → write to DB
 *
 * After indexing, optionally rebuilds the kNN graph for recommendation algorithms.
 */
class IndexingService : Service() {

    companion object {
        private const val TAG = "IndexingService"
        private const val NOTIFICATION_ID = 2
        private const val CHANNEL_ID = "indexing_service"

        /** FP32 model files required — FP16 files fail with FP32 GPU precision. */
        private val MODEL_VARIANTS = listOf("")

        const val ACTION_START_INDEXING = "com.powerampstartradio.START_INDEXING"
        const val ACTION_CANCEL = "com.powerampstartradio.CANCEL_INDEXING"

        private var activeJob: Job? = null
        val isActive: Boolean get() = activeJob?.isActive == true

        // Observable state for the UI
        private val _state = MutableStateFlow<IndexingState>(IndexingState.Idle)
        val state: StateFlow<IndexingState> = _state.asStateFlow()

        /** Tracks passed from IndexingViewModel; consumed by the service on start. */
        @Volatile
        var pendingTracks: List<NewTrackDetector.UnindexedTrack>? = null

        /** Whether to rebuild kNN graph after indexing. */
        @Volatile
        var pendingBuildGraph: Boolean = false

        fun startIndexing(
            context: Context,
            selectedTracks: List<NewTrackDetector.UnindexedTrack>? = null,
            buildGraph: Boolean = false,
        ) {
            if (isActive) return
            pendingTracks = selectedTracks
            pendingBuildGraph = buildGraph
            _state.value = IndexingState.Starting
            val intent = Intent(context, IndexingService::class.java).apply {
                action = ACTION_START_INDEXING
            }
            context.startForegroundService(intent)
        }

        fun cancelIndexing() {
            activeJob?.cancel()
            _state.value = IndexingState.Idle
        }

        /** Reset from terminal states (Complete/Error) back to Idle. */
        fun resetState() {
            val cur = _state.value
            if (cur is IndexingState.Complete || cur is IndexingState.Error) {
                _state.value = IndexingState.Idle
            }
        }
    }

    sealed class IndexingState {
        data object Idle : IndexingState()
        data object Starting : IndexingState()
        data class Detecting(val message: String) : IndexingState()
        data class Processing(
            val current: Int,
            val total: Int,
            val trackName: String,
            val passName: String = "",
            val detail: String = "",
            val progressFraction: Float = 0f,
            val estimatedRemainingMs: Long = 0,
        ) : IndexingState()
        data class RebuildingIndices(
            val message: String,
            val phaseName: String = "",
            val progressFraction: Float = -1f,
            val estimatedRemainingMs: Long = 0,
        ) : IndexingState()
        data class Complete(val indexed: Int, val failed: Int) : IndexingState()
        data class Error(val message: String) : IndexingState()
    }

    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private var wakeLock: PowerManager.WakeLock? = null

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_START_INDEXING -> {
                startForeground(NOTIFICATION_ID, createNotification("Preparing..."))
                acquireWakeLock()
                performIndexing()
            }
            ACTION_CANCEL -> {
                cancelIndexing()
                stopSelf()
            }
        }
        return START_NOT_STICKY
    }

    private fun performIndexing() {
        activeJob = serviceScope.launch {
            try {
                val dbFile = File(filesDir, "embeddings.db")
                if (!dbFile.exists()) {
                    _state.value = IndexingState.Error("No embedding database found")
                    stopSelf()
                    return@launch
                }

                val db = EmbeddingDatabase.openReadWrite(dbFile)

                // Use pre-selected tracks if provided, otherwise detect
                val unindexed = pendingTracks?.also {
                    pendingTracks = null
                    _state.value = IndexingState.Detecting("Preparing ${it.size} selected tracks...")
                    updateNotification("Preparing ${it.size} selected tracks...")
                    Log.i(TAG, "Using ${it.size} pre-selected tracks")
                } ?: run {
                    _state.value = IndexingState.Detecting("Detecting new tracks...")
                    updateNotification("Detecting new tracks...")
                    val detector = NewTrackDetector(db)
                    detector.findUnindexedTracks(this@IndexingService) { status ->
                        _state.value = IndexingState.Detecting(status)
                        updateNotification(status)
                    }
                }

                if (unindexed.isEmpty()) {
                    _state.value = IndexingState.Complete(0, 0)
                    updateNotification("All tracks already indexed")
                    db.close()
                    stopSelfDelayed()
                    return@launch
                }

                Log.i(TAG, "Found ${unindexed.size} unindexed tracks")
                _state.value = IndexingState.Detecting("Loading models for ${unindexed.size} tracks...")
                updateNotification("Loading models for ${unindexed.size} tracks...")
                val batchStartTime = System.currentTimeMillis()

                val mertFile = resolveModelFile(filesDir, "mert")
                val clamp3AudioFile = resolveModelFile(filesDir, "clamp3_audio")

                if (!mertFile.exists() || !clamp3AudioFile.exists()) {
                    _state.value = IndexingState.Error(
                        "CLaMP3 models not found. Transfer mert.tflite and " +
                        "clamp3_audio.tflite to ${filesDir.absolutePath}"
                    )
                    db.close()
                    stopSelf()
                    return@launch
                }

                val writer = EmbeddingWriter(db)
                val audioDecoder = AudioDecoder()

                var indexed = 0
                var failed = 0

                // Pre-compute total work steps for ETA.
                // Step weights calibrated for CLaMP3 pipeline:
                // - Audio decode: ~12s/track (weight=45)
                // - MERT window inference: ~0.3s/window (weight=1)
                // - CLaMP3 segment inference: ~0.5s/segment (weight=2)
                // - DB write: negligible (weight=1)
                val DECODE_WEIGHT = 45
                val CLAMP3_SEGMENT_WEIGHT = 2
                val WRITE_WEIGHT = 1
                var totalSteps = 0
                for (t in unindexed) {
                    val durS = t.durationMs / 1000f
                    if (durS < MertInference.WINDOW_SEC) continue
                    totalSteps += DECODE_WEIGHT
                    val windows = (durS * MertInference.SAMPLE_RATE).toInt() / MertInference.WINDOW_SAMPLES
                    totalSteps += windows  // MERT windows, weight=1 each
                    val segments = maxOf(1, kotlin.math.ceil(windows.toFloat() / Clamp3AudioInference.MAX_WINDOWS).toInt())
                    totalSteps += segments * CLAMP3_SEGMENT_WEIGHT
                    totalSteps += WRITE_WEIGHT
                }
                var completedSteps = 0
                var fractionFloor = 0f
                var etaStartTime = 0L
                var lastEta = 0L

                fun fraction(): Float {
                    val raw = if (totalSteps > 0) completedSteps.toFloat() / totalSteps else 0f
                    fractionFloor = maxOf(fractionFloor, raw)
                    return fractionFloor
                }

                fun eta(): Long {
                    if (completedSteps == 0 || etaStartTime == 0L) return 0L
                    val elapsed = System.currentTimeMillis() - etaStartTime
                    val rawEta = elapsed * (totalSteps - completedSteps) / completedSteps
                    lastEta = if (rawEta > lastEta && lastEta > 0L) {
                        minOf(rawEta, (lastEta * 1.2).toLong())
                    } else rawEta
                    return lastEta
                }

                // ── Phase 1: MERT Feature Extraction (GPU) ────────────────
                // Extract 768d MERT features per 5-second window. Features are
                // spilled to disk because the Adreno GPU can't have MERT and
                // CLaMP3 audio TFLite models loaded simultaneously.

                _state.value = IndexingState.Processing(
                    current = 0, total = unindexed.size,
                    trackName = "", passName = "MERT features",
                    detail = "Loading MERT model (GPU)...",
                    progressFraction = 0f,
                    estimatedRemainingMs = 0,
                )
                updateNotification("Loading MERT model...")
                val mertLoadStart = System.nanoTime()
                val mertInference = try { MertInference(mertFile) }
                catch (e: Exception) {
                    Log.e(TAG, "Failed to load MERT", e)
                    _state.value = IndexingState.Error("Failed to load MERT model: ${e.message}")
                    db.close()
                    stopSelf()
                    return@launch
                }
                Log.i(TAG, "TIMING: mert_model_load = ${(System.nanoTime() - mertLoadStart) / 1_000_000}ms")

                // Disk-spill: write all MERT features to a single temp file.
                // Each window is 768 floats = 3072 bytes. For 100 tracks × 48 windows
                // = ~14 MB disk — trivial.
                val featuresFile = File(cacheDir, "mert_features.tmp")
                val windowCounts = IntArray(unindexed.size)
                val windowBytes = MertInference.FEATURE_DIM * 4  // 768 × 4 = 3072

                etaStartTime = System.currentTimeMillis()

                try {  // ensures featuresFile is cleaned up
                val writeBuf = java.nio.ByteBuffer.allocate(windowBytes)
                    .order(java.nio.ByteOrder.LITTLE_ENDIAN)
                val writeChannel = java.io.FileOutputStream(featuresFile).channel

                // CPU prefetch: decode track N+1 while GPU runs MERT on track N.
                // Decode+resample is CPU-bound; MERT is GPU-bound. Overlapping them
                // hides decode latency for all but the first track.
                var prefetchJob: kotlinx.coroutines.Deferred<AudioDecoder.DecodedAudio?>? = null
                var prefetchIndex = -1

                fun startPrefetch(idx: Int) {
                    if (idx >= unindexed.size) return
                    val nextTrack = unindexed[idx]
                    val nextFile = resolveAudioFile(nextTrack) ?: return
                    prefetchIndex = idx
                    prefetchJob = async(Dispatchers.Default) {
                        try {
                            audioDecoder.decode(nextFile, MertInference.SAMPLE_RATE,
                                maxDurationS = MertInference.MAX_DURATION_S)
                        } catch (e: Exception) {
                            Log.e(TAG, "Prefetch decode failed for: ${nextTrack.title}", e)
                            null
                        }
                    }
                }

                try {
                for ((i, track) in unindexed.withIndex()) {
                    ensureActive()
                    val trackProgress = "${i + 1}/${unindexed.size}"
                    var windowsDone = 0
                    fun emitProgress(detail: String, etaMs: Long = eta()) {
                        val msg = "MERT $trackProgress ${track.title} \u2013 $detail${formatEta(etaMs)}"
                        _state.value = IndexingState.Processing(
                            current = i + 1,
                            total = unindexed.size,
                            trackName = "${track.artist} - ${track.title}",
                            passName = "MERT features",
                            detail = detail,
                            progressFraction = fraction(),
                            estimatedRemainingMs = etaMs,
                        )
                        updateNotification(msg, completedSteps, totalSteps)
                    }

                    val trackStart = System.nanoTime()

                    val audioFile = resolveAudioFile(track)
                    if (audioFile == null) {
                        Log.w(TAG, "Cannot resolve audio file for: ${track.title}")
                        failed++
                        continue
                    }

                    // Use prefetched decode if available, otherwise decode now
                    val audio24k = if (i == prefetchIndex && prefetchJob != null) {
                        emitProgress("awaiting prefetch")
                        val result = prefetchJob!!.await()
                        prefetchJob = null
                        prefetchIndex = -1
                        result
                    } else {
                        emitProgress("decoding")
                        audioDecoder.decode(audioFile, MertInference.SAMPLE_RATE,
                            maxDurationS = MertInference.MAX_DURATION_S)
                    }

                    if (audio24k == null) {
                        Log.w(TAG, "Decode failed for: ${track.title}")
                        failed++
                        continue
                    }
                    completedSteps += DECODE_WEIGHT

                    val expectedWindows = mertInference.windowCount(audio24k.durationS)
                    if (expectedWindows == 0) {
                        Log.w(TAG, "Audio too short for MERT: ${track.title} (${audio24k.durationS}s)")
                        failed++
                        continue
                    }

                    // Start prefetching next track BEFORE MERT inference.
                    // While GPU runs MERT (~10s), CPU decodes next track in parallel.
                    startPrefetch(i + 1)

                    windowsDone = 0
                    emitProgress("window 0/$expectedWindows")

                    // Stream MERT features to disk
                    val mertStart = System.nanoTime()
                    val numExtracted = mertInference.extractFeaturesStreaming(
                        audio24k,
                        onFeatureExtracted = { feature ->
                            writeBuf.clear()
                            writeBuf.asFloatBuffer().put(feature)
                            writeBuf.rewind()
                            writeChannel.write(writeBuf)
                        },
                        onWindowDone = {
                            completedSteps++
                            windowsDone++
                            emitProgress("window $windowsDone/$expectedWindows")
                        },
                    )
                    val mertMs = (System.nanoTime() - mertStart) / 1_000_000

                    val trackMs = (System.nanoTime() - trackStart) / 1_000_000
                    Log.i(TAG, "TIMING: mert_track ${i+1}/${unindexed.size} " +
                        "\"${track.artist} - ${track.title}\" = ${trackMs}ms " +
                        "(decode=${audio24k.decodeMs}ms, resample=${audio24k.resampleMs}ms, " +
                        "mert=${mertMs}ms, windows=$numExtracted)")

                    if (numExtracted > 0) {
                        windowCounts[i] = numExtracted
                    } else {
                        failed++
                    }
                }
                prefetchJob?.cancel()
                } finally {
                    writeChannel.close()
                }

                val totalWindows = windowCounts.sum()
                val spillMB = totalWindows.toLong() * windowBytes / (1024 * 1024)
                Log.i(TAG, "MERT phase done: $totalWindows windows spilled to disk (${spillMB}MB)")

                // ── Phase transition: MERT → CLaMP3 audio encoder ────────
                mertInference.close()

                _state.value = IndexingState.Processing(
                    current = 0, total = unindexed.size,
                    trackName = "", passName = "CLaMP3 encode",
                    detail = "Loading CLaMP3 audio encoder (GPU)...",
                    progressFraction = fraction(),
                    estimatedRemainingMs = eta(),
                )
                updateNotification("Loading CLaMP3 audio encoder...")
                val clamp3LoadStart = System.nanoTime()
                val clamp3Inference = try { Clamp3AudioInference(clamp3AudioFile) }
                catch (e: Exception) {
                    Log.e(TAG, "Failed to load CLaMP3 audio encoder", e)
                    _state.value = IndexingState.Error("Failed to load CLaMP3 model: ${e.message}")
                    db.close()
                    stopSelf()
                    return@launch
                }
                Log.i(TAG, "TIMING: clamp3_audio_load = ${(System.nanoTime() - clamp3LoadStart) / 1_000_000}ms")

                // ── Phase 2: CLaMP3 Audio Encoding ───────────────────────
                // Read MERT features from disk, encode into 768d embeddings,
                // write to database.
                val readBuf = java.nio.ByteBuffer.allocate(windowBytes)
                    .order(java.nio.ByteOrder.LITTLE_ENDIAN)
                val readChannel = java.io.FileInputStream(featuresFile).channel

                try {
                val encodableCount = windowCounts.count { it > 0 }
                var encoded = 0
                for ((i, track) in unindexed.withIndex()) {
                    val numWindows = windowCounts[i]
                    if (numWindows == 0) continue

                    ensureActive()
                    encoded++
                    val numSegments = clamp3Inference.segmentCount(numWindows)
                    val encMsg = "CLaMP3 $encoded/$encodableCount ${track.title}"
                    _state.value = IndexingState.Processing(
                        current = encoded,
                        total = encodableCount,
                        trackName = "${track.artist} - ${track.title}",
                        passName = "CLaMP3 encode",
                        detail = "encoding ($numWindows windows, $numSegments segments)",
                        progressFraction = fraction(),
                        estimatedRemainingMs = eta(),
                    )
                    updateNotification(encMsg, completedSteps, totalSteps)
                    val encStart = System.nanoTime()

                    // Stream MERT features from disk one window at a time
                    val embedding = clamp3Inference.encodeStreaming(
                        numWindows = numWindows,
                        readNextWindow = {
                            readBuf.clear()
                            readChannel.read(readBuf)
                            readBuf.flip()
                            FloatArray(MertInference.FEATURE_DIM).also { arr ->
                                readBuf.asFloatBuffer().get(arr)
                            }
                        },
                        onSegmentDone = {
                            completedSteps += CLAMP3_SEGMENT_WEIGHT
                        },
                    )

                    if (embedding == null) {
                        Log.w(TAG, "CLaMP3 encode failed for: ${track.artist} - ${track.title}")
                        failed++
                        continue
                    }

                    // Write to database
                    val trackId = writer.writeTrack(
                        metadataKey = track.metadataKey,
                        filenameKey = track.filenameKey,
                        artist = track.artist.ifEmpty { null },
                        album = track.album.ifEmpty { null },
                        title = track.title.ifEmpty { null },
                        durationMs = track.durationMs,
                        filePath = track.path ?: "",
                        embedding = embedding,
                        source = "phone",
                    )
                    completedSteps += WRITE_WEIGHT
                    val encMs = (System.nanoTime() - encStart) / 1_000_000

                    if (trackId > 0) {
                        indexed++
                    } else {
                        Log.e(TAG, "DB write failed for: ${track.artist} - ${track.title}")
                        failed++
                    }
                    Log.i(TAG, "TIMING: clamp3_encode " +
                        "\"${track.artist} - ${track.title}\" = ${encMs}ms")
                }
                } finally {
                    readChannel.close()
                }

                clamp3Inference.close()

                } finally {
                    featuresFile.delete()
                }

                val clamp3PassMs = System.currentTimeMillis() - batchStartTime
                Log.i(TAG, "TIMING: clamp3_pass_total = ${clamp3PassMs}ms " +
                    "($indexed indexed, $failed failed, " +
                    "${clamp3PassMs / maxOf(indexed, 1)}ms/track avg)")

                // Rebuild indices after indexing new tracks
                if (indexed > 0) {
                    val buildGraph = pendingBuildGraph.also { pendingBuildGraph = false }
                    val rebuildStart = System.currentTimeMillis()

                    _state.value = IndexingState.RebuildingIndices(
                        message = "Rebuilding search indices...")
                    updateNotification("Rebuilding indices...")

                    val graphUpdater = GraphUpdater(db, filesDir)
                    graphUpdater.rebuildIndices { status ->
                        _state.value = IndexingState.RebuildingIndices(message = status)
                        updateNotification(status)
                    }
                    Log.i(TAG, "TIMING: rebuild_indices = ${System.currentTimeMillis() - rebuildStart}ms")
                }

                db.close()

                val totalBatchMs = System.currentTimeMillis() - batchStartTime
                Log.i(TAG, "TIMING: batch_total = ${totalBatchMs}ms " +
                    "($indexed indexed, $failed failed, ${totalBatchMs / maxOf(indexed, 1)}ms/track avg)")

                _state.value = IndexingState.Complete(indexed, failed)
                val message = "$indexed tracks indexed" +
                    if (failed > 0) " ($failed failed)" else ""
                showCompletionNotification(message)
                Log.i(TAG, message)

                stopSelfDelayed()

            } catch (e: CancellationException) {
                Log.d(TAG, "Indexing cancelled")
                _state.value = IndexingState.Idle
                stopSelf()
            } catch (e: OutOfMemoryError) {
                Log.e(TAG, "OOM during indexing — stopping", e)
                System.gc()
                _state.value = IndexingState.Error("Out of memory — try fewer tracks")
                updateNotification("Out of memory")
                stopSelfDelayed()
            } catch (e: Exception) {
                Log.e(TAG, "Indexing failed", e)
                _state.value = IndexingState.Error("Error: ${e.message}")
                updateNotification("Error: ${e.message}")
                stopSelfDelayed()
            } finally {
                activeJob = null
                releaseWakeLock()
            }
        }
    }

    /**
     * Find the model file for a given base name.
     * FP32 model files required (FP16 files incompatible with FP32 GPU precision).
     */
    private fun resolveModelFile(dir: File, baseName: String): File {
        for (suffix in MODEL_VARIANTS) {
            val f = File(dir, "${baseName}${suffix}.tflite")
            if (f.exists()) {
                Log.i(TAG, "Model resolved: ${f.name}")
                return f
            }
        }
        return File(dir, "${baseName}.tflite")  // default (may not exist)
    }

    /**
     * Resolve a Poweramp track to an actual audio file on the filesystem.
     */
    private fun resolveAudioFile(track: NewTrackDetector.UnindexedTrack): File? {
        val path = track.path ?: return null

        val direct = File(path)
        if (direct.exists() && direct.canRead()) return direct

        val prefixes = listOf(
            "/storage/emulated/0/",
            "/sdcard/",
            "/storage/sdcard0/",
        )
        for (prefix in prefixes) {
            val withPrefix = File(prefix + path)
            if (withPrefix.exists() && withPrefix.canRead()) return withPrefix
        }

        return null
    }

    private fun acquireWakeLock() {
        val pm = getSystemService(Context.POWER_SERVICE) as PowerManager
        wakeLock = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "PowerampStartRadio:indexing")
        wakeLock?.acquire(60 * 60 * 1000L)  // 1 hour max
    }

    private fun releaseWakeLock() {
        wakeLock?.let {
            if (it.isHeld) it.release()
        }
        wakeLock = null
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
            "Indexing Service",
            NotificationManager.IMPORTANCE_LOW,
        ).apply {
            description = "Shows progress when indexing new tracks"
        }
        getSystemService(NotificationManager::class.java).createNotificationChannel(channel)
    }

    private fun createNotification(
        message: String,
        current: Int = 0,
        total: Int = 0,
    ): Notification {
        val pendingIntent = PendingIntent.getActivity(
            this, 0,
            Intent(this, IndexingActivity::class.java),
            PendingIntent.FLAG_IMMUTABLE,
        )
        val cancelIntent = PendingIntent.getService(
            this, 0,
            Intent(this, IndexingService::class.java).apply { action = ACTION_CANCEL },
            PendingIntent.FLAG_IMMUTABLE,
        )
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("Indexing Tracks")
            .setContentText(message)
            .setSmallIcon(R.drawable.ic_radio)
            .setContentIntent(pendingIntent)
            .addAction(R.drawable.ic_radio, "Cancel", cancelIntent)
            .setOngoing(true)
            .apply {
                if (total > 0) {
                    setProgress(total, current, false)
                } else {
                    setProgress(0, 0, true)
                }
            }
            .build()
    }

    private fun updateNotification(message: String, current: Int = 0, total: Int = 0) {
        getSystemService(NotificationManager::class.java)
            .notify(NOTIFICATION_ID, createNotification(message, current, total))
    }

    private fun showCompletionNotification(message: String) {
        val pendingIntent = PendingIntent.getActivity(
            this, 0,
            Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_IMMUTABLE,
        )
        val notification = NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("Indexing Complete")
            .setContentText(message)
            .setSmallIcon(R.drawable.ic_radio)
            .setContentIntent(pendingIntent)
            .setAutoCancel(true)
            .build()
        getSystemService(NotificationManager::class.java)
            .notify(NOTIFICATION_ID + 1, notification)
    }

    private fun formatEta(etaMs: Long): String {
        if (etaMs <= 0) return ""
        val minutes = etaMs / 60_000
        return if (minutes < 1) " (~<1 min left)"
        else " (~$minutes min left)"
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        super.onDestroy()
        serviceScope.cancel()
        releaseWakeLock()
    }
}
