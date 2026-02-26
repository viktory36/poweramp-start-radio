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
            // Hoisted so the finally block can release GPU resources on cancellation.
            // Without this, CancellationException skips .close() and the OpenCL context
            // leaks, causing "Failed to create input buffers" on the next attempt.
            var mertInference: MertInference? = null
            var clamp3Inference: Clamp3AudioInference? = null
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

                // ── ETA: measured wall-clock extrapolation ──────────────
                // Instead of guessing step weights, we measure actual ms/window
                // from completed MERT inference (which dominates ~95% of time)
                // and extrapolate for remaining windows. This naturally adapts
                // to any device, resampler, or model speed.
                //
                // Per-track cost breakdown (Snapdragon 8 Gen 3, 2026-02):
                //   MERT window: ~250ms  |  Decode+resample: ~3.5s (hidden by prefetch)
                //   CLaMP3: ~50ms/track   |  Graph rebuild: ~10s (separate phase)
                //
                // Since decode is hidden by CPU prefetch after track 1, the
                // wall-clock per-track ≈ MERT windows × ms/window. We measure
                // this directly.

                // Pre-compute expected window counts per track (full duration, no cap)
                val windowsPerTrack = IntArray(unindexed.size)
                var totalWindows = 0
                for ((i, t) in unindexed.withIndex()) {
                    val durS = t.durationMs / 1000f
                    if (durS < MertInference.WINDOW_SEC) continue
                    val totalSamples = (durS * MertInference.SAMPLE_RATE).toInt()
                    val full = totalSamples / MertInference.WINDOW_SAMPLES
                    val remain = totalSamples % MertInference.WINDOW_SAMPLES
                    val w = full + if (remain >= MertInference.SAMPLE_RATE) 1 else 0
                    windowsPerTrack[i] = w
                    totalWindows += w
                }

                var windowsCompleted = 0
                var tracksCompleted = 0
                var fractionFloor = 0f
                // EMA of ms/window, updated both within-track (from MERT windows)
                // and at track boundaries (from full wall-clock). This gives an
                // ETA even during the first track — crucial for single-track indexing.
                var msPerWindowEma = 0.0
                val EMA_ALPHA = 0.3  // weight for newest measurement
                var mertPhaseStartTime = 0L
                var currentTrackMertStart = 0L  // set when MERT begins for current track
                var lastEta = 0L

                fun fraction(): Float {
                    val raw = if (totalWindows > 0) windowsCompleted.toFloat() / totalWindows else 0f
                    // CLaMP3 phase is fast (~1% of total) — reserve last 5% for it + graph
                    val scaled = raw * 0.95f
                    fractionFloor = maxOf(fractionFloor, scaled)
                    return fractionFloor
                }

                fun eta(): Long {
                    if (msPerWindowEma <= 0) return 0L
                    val remainingWindows = totalWindows - windowsCompleted
                    // Add flat budget for CLaMP3 phase (~50ms/track) + graph rebuild
                    val remainingTracks = unindexed.size - tracksCompleted
                    val clamp3Budget = remainingTracks * 50L
                    val graphBudget = if (remainingWindows > 0) 10_000L else 0L
                    val rawEta = (remainingWindows * msPerWindowEma).toLong() + clamp3Budget + graphBudget
                    // Dampen upward jumps (max +20% per update) for smoother display
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
                mertInference = try { MertInference(mertFile) }
                catch (e: Exception) {
                    Log.e(TAG, "Failed to load MERT", e)
                    _state.value = IndexingState.Error("Failed to load MERT model: ${e.message}")
                    db.close()
                    stopSelf()
                    return@launch
                }
                val mert = mertInference!!  // safe: assigned above, exception returns early
                Log.i(TAG, "TIMING: mert_model_load = ${(System.nanoTime() - mertLoadStart) / 1_000_000}ms")

                // Disk-spill: write all MERT features to a single temp file.
                // Each window is 768 floats = 3072 bytes. For 100 tracks × 48 windows
                // = ~14 MB disk — trivial.
                val featuresFile = File(cacheDir, "mert_features.tmp")
                val windowCounts = IntArray(unindexed.size)
                val windowBytes = MertInference.FEATURE_DIM * 4  // 768 × 4 = 3072

                mertPhaseStartTime = System.currentTimeMillis()

                try {  // ensures featuresFile is cleaned up
                val writeBuf = java.nio.ByteBuffer.allocate(windowBytes)
                    .order(java.nio.ByteOrder.LITTLE_ENDIAN)
                val writeChannel = java.io.FileOutputStream(featuresFile).channel

                // CPU prefetch: decode first chunk of track N+1 while GPU runs
                // MERT on track N. Decode+resample is CPU-bound; MERT is GPU-bound.
                var prefetchJob: kotlinx.coroutines.Deferred<AudioDecoder.DecodedAudio?>? = null
                var prefetchIndex = -1

                fun startPrefetch(idx: Int, chunkS: Int) {
                    if (idx >= unindexed.size) return
                    val nextTrack = unindexed[idx]
                    val nextFile = resolveAudioFile(nextTrack) ?: return
                    prefetchIndex = idx
                    prefetchJob = async(Dispatchers.Default) {
                        try {
                            audioDecoder.decode(nextFile, MertInference.SAMPLE_RATE,
                                maxDurationS = chunkS)
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
                    val expectedWindows = windowsPerTrack[i]
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
                        updateNotification(msg, windowsCompleted, totalWindows)
                    }

                    val trackStart = System.nanoTime()

                    val audioFile = resolveAudioFile(track)
                    if (audioFile == null) {
                        Log.w(TAG, "Cannot resolve audio file for: ${track.title}")
                        failed++
                        continue
                    }

                    if (expectedWindows == 0) {
                        Log.w(TAG, "Audio too short for MERT: ${track.title}")
                        failed++
                        continue
                    }

                    // Chunked decode+MERT: process the full track in memory-adaptive
                    // chunks. Chunk size is recomputed per-track from available heap,
                    // so it adapts if memory pressure changes mid-run. Each chunk is
                    // decoded, resampled, fed to MERT, then freed.
                    val trackDurationS = track.durationMs / 1000
                    var trackWindowsExtracted = 0
                    var totalDecodeMs = 0L
                    var totalResampleMs = 0L
                    var totalMertMs = 0L
                    var numChunks = 0

                    windowsDone = 0
                    emitProgress("window 0/$expectedWindows")

                    var prefetchStarted = false
                    var chunkStart = 0  // position cursor in seconds

                    while (chunkStart < trackDurationS || chunkStart == 0) {
                        ensureActive()
                        val chunkS = computeChunkDurationS()
                        val isLastChunk = chunkStart + chunkS >= trackDurationS

                        // First chunk: use prefetch if available
                        val audio24k = if (chunkStart == 0 && i == prefetchIndex && prefetchJob != null) {
                            emitProgress("awaiting prefetch")
                            val result = prefetchJob!!.await()
                            prefetchJob = null
                            prefetchIndex = -1
                            result
                        } else {
                            if (chunkStart == 0) emitProgress("decoding")
                            audioDecoder.decode(audioFile, MertInference.SAMPLE_RATE,
                                maxDurationS = chunkS, startTimeS = chunkStart)
                        }

                        if (audio24k == null || audio24k.samples.isEmpty()) break
                        totalDecodeMs += audio24k.decodeMs
                        totalResampleMs += audio24k.resampleMs
                        numChunks++

                        val chunkWindows = mert.windowCount(audio24k.durationS)
                        if (chunkWindows == 0) { chunkStart += chunkS; continue }

                        // Prefetch next track's first chunk during last chunk of this track
                        if (isLastChunk && !prefetchStarted) {
                            startPrefetch(i + 1, chunkS)
                            prefetchStarted = true
                        }

                        // Stream MERT features to disk
                        val mertStart = System.nanoTime()
                        if (windowsDone == 0) currentTrackMertStart = mertStart
                        val numExtracted = mert.extractFeaturesStreaming(
                            audio24k,
                            onFeatureExtracted = { feature ->
                                writeBuf.clear()
                                writeBuf.asFloatBuffer().put(feature)
                                writeBuf.rewind()
                                writeChannel.write(writeBuf)
                            },
                            onWindowDone = {
                                windowsCompleted++
                                windowsDone++
                                // Update EMA from within-track MERT windows. This gives
                                // an ETA even during the first track (after ~3 windows).
                                if (windowsDone >= 3) {
                                    val mertElapsed = (System.nanoTime() - currentTrackMertStart) / 1_000_000.0
                                    val measured = mertElapsed / windowsDone
                                    msPerWindowEma = if (msPerWindowEma <= 0) measured
                                                     else msPerWindowEma * (1 - EMA_ALPHA) + measured * EMA_ALPHA
                                }
                                emitProgress("window $windowsDone/$expectedWindows")
                            },
                        )
                        totalMertMs += (System.nanoTime() - mertStart) / 1_000_000
                        trackWindowsExtracted += numExtracted
                        chunkStart += chunkS
                    }

                    val trackMs = (System.nanoTime() - trackStart) / 1_000_000
                    Log.i(TAG, "TIMING: mert_track ${i+1}/${unindexed.size} " +
                        "\"${track.artist} - ${track.title}\" = ${trackMs}ms " +
                        "(decode=${totalDecodeMs}ms, resample=${totalResampleMs}ms, " +
                        "mert=${totalMertMs}ms, windows=$trackWindowsExtracted, chunks=$numChunks)")

                    if (trackWindowsExtracted > 0) {
                        windowCounts[i] = trackWindowsExtracted
                        tracksCompleted++
                        // Update EMA from full track wall-clock for cross-track accuracy
                        val measuredMsPerWindow = trackMs.toDouble() / trackWindowsExtracted
                        msPerWindowEma = if (tracksCompleted == 1) measuredMsPerWindow
                                         else msPerWindowEma * (1 - EMA_ALPHA) + measuredMsPerWindow * EMA_ALPHA
                    } else {
                        failed++
                    }
                }
                prefetchJob?.cancel()
                } finally {
                    writeChannel.close()
                }

                val actualTotalWindows = windowCounts.sum()
                val spillMB = actualTotalWindows.toLong() * windowBytes / (1024 * 1024)
                Log.i(TAG, "MERT phase done: $actualTotalWindows windows spilled to disk (${spillMB}MB)")

                // ── Phase transition: MERT → CLaMP3 audio encoder ────────
                mert.close()
                mertInference = null

                _state.value = IndexingState.Processing(
                    current = 0, total = unindexed.size,
                    trackName = "", passName = "CLaMP3 encode",
                    detail = "Loading CLaMP3 audio encoder (GPU)...",
                    progressFraction = fraction(),
                    estimatedRemainingMs = eta(),
                )
                updateNotification("Loading CLaMP3 audio encoder...")
                val clamp3LoadStart = System.nanoTime()
                clamp3Inference = try { Clamp3AudioInference(clamp3AudioFile) }
                catch (e: Exception) {
                    Log.e(TAG, "Failed to load CLaMP3 audio encoder", e)
                    _state.value = IndexingState.Error("Failed to load CLaMP3 model: ${e.message}")
                    db.close()
                    stopSelf()
                    return@launch
                }
                val clamp3 = clamp3Inference!!
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
                    val numSegments = clamp3.segmentCount(numWindows)
                    val encMsg = "CLaMP3 $encoded/$encodableCount ${track.title}"
                    // CLaMP3 phase fills the 0.95→1.0 range
                    val clamp3Frac = 0.95f + 0.05f * (encoded - 1).toFloat() / maxOf(encodableCount, 1)
                    _state.value = IndexingState.Processing(
                        current = encoded,
                        total = encodableCount,
                        trackName = "${track.artist} - ${track.title}",
                        passName = "CLaMP3 encode",
                        detail = "encoding ($numWindows windows, $numSegments segments)",
                        progressFraction = clamp3Frac,
                        estimatedRemainingMs = 0,  // CLaMP3 is too fast for meaningful ETA
                    )
                    updateNotification(encMsg, windowsCompleted, totalWindows)
                    val encStart = System.nanoTime()

                    // Stream MERT features from disk one window at a time
                    val embedding = clamp3.encodeStreaming(
                        numWindows = numWindows,
                        readNextWindow = {
                            readBuf.clear()
                            readChannel.read(readBuf)
                            readBuf.flip()
                            FloatArray(MertInference.FEATURE_DIM).also { arr ->
                                readBuf.asFloatBuffer().get(arr)
                            }
                        },
                        onSegmentDone = { },
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

                clamp3.close()
                clamp3Inference = null

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
                // Release GPU resources if cancelled before normal .close() path
                try { mertInference?.close() } catch (_: Exception) {}
                try { clamp3Inference?.close() } catch (_: Exception) {}
                activeJob = null
                releaseWakeLock()
            }
        }
    }

    /**
     * Compute optimal decode chunk duration based on available heap memory.
     * Larger chunks = fewer codec setups + better CPU prefetch overlap.
     *
     * Peak memory per second of audio during decode+resample (48kHz→24kHz):
     *   native mono float (48000×4) + resampled float (24000×4) coexist
     *   = 288,000 bytes/second of audio
     *
     * Uses 50% of available heap for the chunk, rest as headroom for
     * TFLite GPU buffers, GC pressure, notification bitmaps, etc.
     */
    private fun computeChunkDurationS(): Int {
        val runtime = Runtime.getRuntime()
        val usedBytes = runtime.totalMemory() - runtime.freeMemory()
        val availableBytes = runtime.maxMemory() - usedBytes
        val budgetBytes = availableBytes / 2
        // Assume 48kHz source (worst common case: Opus, high-bitrate FLAC)
        val peakBytesPerSec = (48000 + 24000) * 4L  // 288,000
        val chunkS = (budgetBytes / peakBytesPerSec).toInt()
            .coerceIn(30, 600)  // 30s min (6 windows), 10min max
        Log.i(TAG, "Chunk sizing: heap_max=${runtime.maxMemory()/1024/1024}MB, " +
            "available=${availableBytes/1024/1024}MB, budget=${budgetBytes/1024/1024}MB, " +
            "chunk=${chunkS}s")
        return chunkS
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
