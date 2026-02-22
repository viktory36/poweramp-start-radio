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
import kotlinx.coroutines.Deferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.async
import kotlinx.coroutines.cancel
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import java.io.File

/**
 * Foreground service for on-device embedding indexing.
 *
 * Runs MuQ-MuLan + Flamingo TFLite inference on new tracks detected in the Poweramp
 * library, writes embeddings to the database, and rebuilds indices.
 *
 * Follows the same foreground service pattern as RadioService.
 */
class IndexingService : Service() {

    enum class FusionTier { QUICK_UPDATE, RECLUSTER, FULL_REFUSION }

    data class FusionOptions(val tier: FusionTier, val buildKnnGraph: Boolean = false)

    companion object {
        private const val TAG = "IndexingService"
        private const val NOTIFICATION_ID = 2
        private const val CHANNEL_ID = "indexing_service"

        /** Model variant suffixes in preference order: FP16 (GPU-native), then FP32. */
        private val MODEL_VARIANTS = listOf("_fp16", "")

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

        /** Post-indexing fusion options. Null = extract indices only (no fusion). */
        @Volatile
        var pendingFusionOptions: FusionOptions? = null

        fun startIndexing(
            context: Context,
            selectedTracks: List<NewTrackDetector.UnindexedTrack>? = null,
            fusionOptions: FusionOptions? = null,
        ) {
            if (isActive) return
            pendingTracks = selectedTracks
            pendingFusionOptions = fusionOptions
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

    /**
     * State of the indexing operation.
     */
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

                // Open database in read-write mode
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

                val mulanFile = resolveModelFile(filesDir, "mulan_audio")
                val flamingoFile = resolveModelFile(filesDir, "flamingo_encoder")
                val projectorFile = resolveModelFile(filesDir, "flamingo_projector")

                val hasMulan = mulanFile.exists()
                val hasFlamingo = flamingoFile.exists()

                if (!hasMulan && !hasFlamingo) {
                    _state.value = IndexingState.Error(
                        "No TFLite models found. Transfer mulan_audio.tflite and/or " +
                        "flamingo_encoder.tflite to ${filesDir.absolutePath}"
                    )
                    db.close()
                    stopSelf()
                    return@launch
                }

                // Load projection matrices from DB metadata
                val rawDb = db.getRawDatabase()
                // flamingo_projection is stored as V_k [3584, 512] (row-major) from
                // the Python `reduce` command. On-device we need V_k^T [512, 3584]
                // so that multiplyVector(flamingoRaw_3584d) → reduced_512d.
                val flamingoProjection = EmbeddingProcessor.loadProjectionMatrix(
                    rawDb, "flamingo_projection", 3584, 512
                )?.transpose()
                val fusedProjection = EmbeddingProcessor.loadProjectionMatrix(
                    rawDb, "fused_projection", 512, 1024
                )
                val centroids = db.loadCentroids()
                val writer = EmbeddingWriter(db, centroids)

                var indexed = 0
                var failed = 0

                // Pre-compute total work steps for ETA.
                // Step weights calibrated from Snapdragon 8 Gen 3 benchmarks (14 tracks):
                // - MuLan clip inference: ~0.27s (base unit, weight=1)
                // - Audio decode: ~12.2s/track (weight=45 clip-equivalents)
                // - Resample 24→16kHz: ~4.4s/track (weight=16)
                // - Flamingo chunk inference: ~1.9s/chunk (weight=7)
                // - Flamingo project+fuse+write: ~2ms/track (weight=1, minimal but provides UI feedback)
                val DECODE_WEIGHT = 45
                val RESAMPLE_WEIGHT = 16
                val FLAMINGO_CHUNK_WEIGHT = 7
                val FLAMINGO_PROJECT_WEIGHT = 1
                var totalSteps = 0
                for (t in unindexed) {
                    val durS = t.durationMs / 1000f
                    if (hasMulan && durS >= 30f) {
                        totalSteps += DECODE_WEIGHT
                        if (hasFlamingo) totalSteps += RESAMPLE_WEIGHT
                        val mulanChunks = maxOf(1, minOf((durS / 60f).toInt(), 30))
                        totalSteps += mulanChunks * 3  // MuLan clips, weight=1 each
                    }
                    if (hasFlamingo && durS >= 3f) {
                        if (!hasMulan) totalSteps += DECODE_WEIGHT
                        val flamingoChunks = maxOf(1, minOf(kotlin.math.ceil(durS / 30.0).toInt(), 60))
                        totalSteps += flamingoChunks * FLAMINGO_CHUNK_WEIGHT
                        totalSteps += FLAMINGO_PROJECT_WEIGHT  // project+fuse+write per track
                    }
                }
                var completedSteps = 0
                // Floor prevents progress bar regression when cache misses
                // dynamically increase totalSteps (line ~540).
                var fractionFloor = 0f
                // ETA clock starts after model loading (not batchStartTime which
                // includes model load). Prevents inflated early ETAs from 4-7s
                // GPU JIT compilation being amortized across the first few steps.
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
                    // Smooth: ETA can drop freely but only increase by 20% per update
                    // to prevent jarring jumps when a slow track inflates the average.
                    lastEta = if (rawEta > lastEta && lastEta > 0L) {
                        minOf(rawEta, (lastEta * 1.2).toLong())
                    } else rawEta
                    return lastEta
                }

                // MuLan results held in memory until Flamingo pass completes.
                // Avoids writing partial entries to DB that would be invisible
                // to search (no fused embedding) but considered "indexed" by detector.
                // Each embedding is 512 floats = 2KB, negligible memory.
                val mulanResults = mutableMapOf<NewTrackDetector.UnindexedTrack, EmbeddingProcessor.EmbeddingResult>()

                // Decode-once optimization: cache 16kHz audio during MuLan pass
                // for Flamingo pass. Avoids redundant decode+resample (~50s/track).
                // Use a LinkedHashMap to maintain insertion order for LRU eviction.
                // Cap at 128MB to prevent OOM on large batches (heap limit ~368MB).
                val cachedAudio16k = if (hasMulan && hasFlamingo)
                    linkedMapOf<NewTrackDetector.UnindexedTrack, AudioDecoder.DecodedAudio>()
                else null
                var cachedAudio16kBytes = 0L
                val maxCacheBytes = 128L * 1024 * 1024  // 128MB
                val audioDecoder = AudioDecoder()

                // ── Pass 1: MuLan ──────────────────────────────────────
                // Sequential loading: load MuLan first, process all tracks,
                // close it before loading Flamingo. This avoids having multiple
                // large models in memory simultaneously.

                if (hasMulan) {
                    _state.value = IndexingState.Processing(
                        current = 0, total = unindexed.size,
                        trackName = "", passName = "MuQ-MuLan pass",
                        detail = "Loading MuQ-MuLan model (GPU)...",
                        progressFraction = 0f,
                        estimatedRemainingMs = 0,
                    )
                    updateNotification("Loading MuQ-MuLan model...")
                    val mulanLoadStart = System.nanoTime()
                    val mulanInference = try { MuLanInference(mulanFile) }
                    catch (e: Exception) {
                        Log.e(TAG, "Failed to load MuQ-MuLan", e)
                        null
                    }
                    Log.i(TAG, "TIMING: mulan_model_load = ${(System.nanoTime() - mulanLoadStart) / 1_000_000}ms")

                    if (mulanInference != null) {
                        val processor = EmbeddingProcessor(
                            mulanModel = mulanInference,
                            flamingoProjection = null,
                            fusedProjection = null,
                        )
                        etaStartTime = System.currentTimeMillis()

                        for ((i, track) in unindexed.withIndex()) {
                            ensureActive()
                            val trackProgress = "${i + 1}/${unindexed.size}"
                            var clipsDone = 0
                            fun emitProgress(detail: String, etaMs: Long = eta()) {
                                val msg = "MuQ-MuLan $trackProgress ${track.title} \u2013 $detail${formatEta(etaMs)}"
                                _state.value = IndexingState.Processing(
                                    current = i + 1,
                                    total = unindexed.size,
                                    trackName = "${track.artist} - ${track.title}",
                                    passName = "MuQ-MuLan pass",
                                    detail = detail,
                                    progressFraction = fraction(),
                                    estimatedRemainingMs = etaMs,
                                )
                                updateNotification(msg, completedSteps, totalSteps)
                            }
                            val durMin = track.durationMs / 60000
                            val durSec = (track.durationMs / 1000) % 60
                            val durStr = if (durMin > 0) "${durMin}m${durSec}s" else "${durSec}s"
                            emitProgress("decoding ($durStr)")
                            val trackStart = System.nanoTime()

                            val audioFile = resolveAudioFile(track)
                            if (audioFile == null) {
                                Log.w(TAG, "Cannot resolve audio file for: ${track.title}")
                                if (!hasFlamingo) failed++
                                continue
                            }

                            // Tighter pre-alloc: use actual duration instead of 900s cap.
                            // A 3-min track at 48kHz: 35MB vs 165MB with the blanket 900s cap.
                            val maxDurMulan = minOf((track.durationMs / 1000).toInt() + 10, 900)
                            val audio24k = audioDecoder.decode(
                                audioFile, 24000,
                                maxDurationS = maxDurMulan,
                            )
                            if (audio24k == null) {
                                Log.w(TAG, "Decode failed for: ${track.title}")
                                if (!hasFlamingo) failed++
                                continue
                            }
                            completedSteps += DECODE_WEIGHT

                            // Launch 16kHz resample on CPU in background while MuLan runs on GPU.
                            // Resample output is small (~11MB for 3-min track) so no OOM risk.
                            var resampleDeferred: Deferred<FloatArray?>? = null
                            if (cachedAudio16k != null) {
                                emitProgress("resampling")
                                resampleDeferred = async(Dispatchers.Default) {
                                    try {
                                        audioDecoder.resample(audio24k.samples, 24000, 16000)
                                    } catch (e: OutOfMemoryError) {
                                        Log.w(TAG, "OOM resampling for ${track.title} — skipping cache", e)
                                        System.gc()
                                        null
                                    } catch (e: Exception) {
                                        Log.w(TAG, "16kHz cache failed for ${track.title}: ${e.message}")
                                        null
                                    }
                                }
                            }

                            // Compute total clips for this track for progress detail
                            val durS = track.durationMs / 1000f
                            val mulanChunks = maxOf(1, minOf((durS / 60f).toInt(), 30))
                            val totalClips = mulanChunks * 3
                            clipsDone = 0

                            // MuLan inference (GPU) — runs concurrently with CPU resample
                            val mulanInferStart = System.nanoTime()
                            emitProgress("clip 0/$totalClips")
                            val embeddings = processor.processTrack(
                                audioFile, preDecodedAudio = audio24k,
                                onProgress = { /* ignore internal progress */ },
                                onChunkDone = {
                                    completedSteps++
                                    clipsDone++
                                    emitProgress("clip $clipsDone/$totalClips")
                                },
                            )
                            val mulanInferMs = (System.nanoTime() - mulanInferStart) / 1_000_000

                            // Collect resample result (should be done by now — inference takes longer)
                            if (resampleDeferred != null) {
                                val samples16k = resampleDeferred.await()
                                if (samples16k != null) {
                                    val entryBytes = samples16k.size.toLong() * 4
                                    // Evict oldest entries if cache would exceed budget
                                    while (cachedAudio16kBytes + entryBytes > maxCacheBytes
                                        && cachedAudio16k!!.isNotEmpty()) {
                                        val oldest = cachedAudio16k.entries.first()
                                        cachedAudio16kBytes -= oldest.value.samples.size.toLong() * 4
                                        cachedAudio16k.remove(oldest.key)
                                    }
                                    cachedAudio16k!![track] = AudioDecoder.DecodedAudio(
                                        samples16k, 16000, samples16k.size.toFloat() / 16000
                                    )
                                    cachedAudio16kBytes += entryBytes
                                }
                                completedSteps += RESAMPLE_WEIGHT
                            }

                            val trackMs = (System.nanoTime() - trackStart) / 1_000_000
                            Log.i(TAG, "TIMING: mulan_track ${i+1}/${unindexed.size} " +
                                "\"${track.artist} - ${track.title}\" = ${trackMs}ms " +
                                "(inference=${mulanInferMs}ms)")

                            if (embeddings == null) {
                                if (!hasFlamingo) failed++
                                continue
                            }

                            if (hasFlamingo) {
                                // Hold in memory — will write after Flamingo pass completes
                                mulanResults[track] = embeddings
                            } else {
                                // MuLan-only mode — write immediately
                                val trackId = writer.writeTrack(
                                    metadataKey = track.metadataKey,
                                    filenameKey = track.filenameKey,
                                    artist = track.artist.ifEmpty { null },
                                    album = track.album.ifEmpty { null },
                                    title = track.title.ifEmpty { null },
                                    durationMs = track.durationMs,
                                    filePath = track.path ?: audioFile.absolutePath,
                                    embeddings = embeddings,
                                    source = "phone",
                                )
                                if (trackId > 0) {
                                    indexed++
                                    Log.i(TAG, "Track ${i + 1}/${unindexed.size} indexed: ${track.artist} - ${track.title}")
                                } else {
                                    failed++
                                }
                            }
                        }

                        processor.close()
                        val mulanPassMs = (System.currentTimeMillis() - batchStartTime)
                        Log.i(TAG, "TIMING: mulan_pass_total = ${mulanPassMs}ms " +
                            "(${mulanResults.size} tracks, ${mulanPassMs / maxOf(mulanResults.size, 1)}ms/track avg)")
                    }
                }

                // ── Pass 2: Flamingo + Fusion (two-phase GPU) ────────
                // Phase 2a: Encoder — encode all tracks on GPU, spill hidden
                //           states to disk (each chunk = 3.84MB, 14 tracks ≈ 490MB
                //           which exceeds the ~368MB heap limit).
                // Phase 2b: Projector — close encoder, load projector on GPU,
                //           read hidden states back from disk, project, fuse, write.
                // Two CL environments can't coexist on Adreno GPUs, so we
                // must close the encoder before loading the projector.
                val flamingoPassStart = System.currentTimeMillis()
                if (hasFlamingo) {
                    // Use Processing state (not Detecting) to preserve progress bar during model load.
                    // The fraction stays at its current value while the detail shows loading status.
                    _state.value = IndexingState.Processing(
                        current = 0, total = unindexed.size,
                        trackName = "", passName = "Flamingo encode",
                        detail = "Loading Flamingo encoder (GPU)...",
                        progressFraction = fraction(),
                        estimatedRemainingMs = eta(),
                    )
                    updateNotification("Loading Flamingo encoder...")
                    val flamingoLoadStart = System.nanoTime()
                    val flamingoInference = try {
                        FlamingoInference(flamingoFile, projectorFile)
                    } catch (e: Exception) {
                        Log.e(TAG, "Failed to load Flamingo", e)
                        null
                    }
                    Log.i(TAG, "TIMING: flamingo_encoder_load = ${(System.nanoTime() - flamingoLoadStart) / 1_000_000}ms")

                    if (flamingoInference != null) {
                        if (etaStartTime == 0L) etaStartTime = System.currentTimeMillis()

                        // Adjust projection matrix dims if projector is absent
                        val actualFlamingoProjection = if (flamingoInference.outputDim != 3584) {
                            EmbeddingProcessor.loadProjectionMatrix(
                                rawDb, "flamingo_projection",
                                flamingoInference.outputDim, 512
                            )?.transpose()
                        } else flamingoProjection

                        // ── Phase 2a: Encode all tracks, spill to disk ──
                        // Hidden states are written to a temp file instead of accumulating
                        // in memory. Each chunk is 750×1280 floats = 3.84MB. For 14 tracks
                        // × ~9 chunks = ~490MB which would OOM the 368MB heap.
                        // Disk I/O is <0.5s total on UFS 4.0 — negligible overhead.
                        val hiddenFile = File(cacheDir, "flamingo_hiddens.tmp")
                        val chunkCounts = IntArray(unindexed.size)
                        val chunkBytes = FlamingoInference.HIDDEN_STATE_BYTES

                        try {  // outer try ensures hiddenFile is always cleaned up
                        val writeBuf = java.nio.ByteBuffer.allocate(chunkBytes)
                            .order(java.nio.ByteOrder.LITTLE_ENDIAN)
                        val writeChannel = java.io.FileOutputStream(hiddenFile).channel

                        try {
                        for ((i, track) in unindexed.withIndex()) {
                            ensureActive()
                            val trackProgress = "${i + 1}/${unindexed.size}"
                            var chunksDone = 0
                            fun emitProgress(detail: String, etaMs: Long = eta()) {
                                val msg = "Flamingo $trackProgress ${track.title} \u2013 $detail${formatEta(etaMs)}"
                                _state.value = IndexingState.Processing(
                                    current = i + 1,
                                    total = unindexed.size,
                                    trackName = "${track.artist} - ${track.title}",
                                    passName = "Flamingo encode",
                                    detail = detail,
                                    progressFraction = fraction(),
                                    estimatedRemainingMs = etaMs,
                                )
                                updateNotification(msg, completedSteps, totalSteps)
                            }

                            val flTrackStart = System.nanoTime()

                            val audioFile = resolveAudioFile(track)
                            if (audioFile == null) {
                                Log.w(TAG, "Cannot resolve audio file for: ${track.title}")
                                failed++
                                continue
                            }

                            // Use cached 16kHz audio from MuLan pass (decode-once optimization)
                            var audio16k = cachedAudio16k?.remove(track)
                            if (audio16k != null) {
                                cachedAudio16kBytes -= audio16k.samples.size.toLong() * 4
                            }

                            // Cache miss or no MuLan pass — decode externally for step tracking
                            if (audio16k == null) {
                                // Cache miss with both models: decode wasn't budgeted in totalSteps
                                if (hasMulan) totalSteps += DECODE_WEIGHT
                                val durMin = track.durationMs / 60000
                                val durSec = (track.durationMs / 1000) % 60
                                val durStr = if (durMin > 0) "${durMin}m${durSec}s" else "${durSec}s"
                                emitProgress("decoding ($durStr)")
                                val maxDurFlamingo = minOf((track.durationMs / 1000).toInt() + 10, 1800)
                                audio16k = audioDecoder.decode(audioFile, 16000, maxDurationS = maxDurFlamingo)
                                if (audio16k == null) {
                                    Log.w(TAG, "Decode failed for: ${track.title}")
                                    failed++
                                    continue
                                }
                                completedSteps += DECODE_WEIGHT
                            }

                            // Compute total Flamingo chunks for progress detail
                            val durS = track.durationMs / 1000f
                            val totalFChunks = maxOf(1, minOf(kotlin.math.ceil(durS / 30.0).toInt(), 60))
                            chunksDone = 0
                            emitProgress("chunk 0/$totalFChunks")

                            // Stream each encoded chunk directly to disk instead of
                            // accumulating all chunks in memory. This keeps per-track
                            // peak memory at ~3.84MB (one chunk) vs up to 230MB for
                            // a 30-min track (60 chunks × 3.84MB).
                            val encInferStart = System.nanoTime()
                            val numEncoded = flamingoInference.encodeTrackStreaming(
                                audio16k,
                                onChunkEncoded = { hidden ->
                                    writeBuf.clear()
                                    writeBuf.asFloatBuffer().put(hidden)
                                    writeBuf.rewind()
                                    writeChannel.write(writeBuf)
                                },
                                onChunkDone = {
                                    completedSteps += FLAMINGO_CHUNK_WEIGHT
                                    chunksDone++
                                    emitProgress("chunk $chunksDone/$totalFChunks")
                                },
                            )
                            val encInferMs = (System.nanoTime() - encInferStart) / 1_000_000

                            val trackMs = (System.nanoTime() - flTrackStart) / 1_000_000
                            Log.i(TAG, "TIMING: flamingo_encode ${i+1}/${unindexed.size} " +
                                "\"${track.artist} - ${track.title}\" = ${trackMs}ms " +
                                "(encode=${encInferMs}ms)")

                            if (numEncoded > 0) {
                                chunkCounts[i] = numEncoded
                            } else {
                                failed++
                            }
                        }
                        } finally {
                            writeChannel.close()
                        }

                        val totalChunks = chunkCounts.sum()
                        val spillMB = totalChunks.toLong() * chunkBytes / (1024 * 1024)
                        Log.i(TAG, "Flamingo encoder done: $totalChunks chunks spilled to disk (${spillMB}MB)")

                        // Free audio cache — no longer needed
                        cachedAudio16k?.clear()
                        cachedAudio16kBytes = 0L

                        // ── Phase transition: encoder → projector ──
                        flamingoInference.closeEncoder()
                        val projLoadStart = System.nanoTime()
                        _state.value = IndexingState.Processing(
                            current = 0, total = unindexed.size,
                            trackName = "", passName = "Flamingo project + fuse",
                            detail = "Loading Flamingo projector (GPU)...",
                            progressFraction = fraction(),
                            estimatedRemainingMs = eta(),
                        )
                        updateNotification("Loading Flamingo projector...")
                        flamingoInference.loadProjector()
                        Log.i(TAG, "TIMING: flamingo_projector_load = ${(System.nanoTime() - projLoadStart) / 1_000_000}ms")

                        // ── Phase 2b: Stream hidden states from disk, project + fuse + write ──
                        // Each chunk is read lazily via callback — only one 3.84MB
                        // chunk in memory at a time, vs 230MB for a 30-min track.
                        val readBuf = java.nio.ByteBuffer.allocate(chunkBytes)
                            .order(java.nio.ByteOrder.LITTLE_ENDIAN)
                        val readChannel = java.io.FileInputStream(hiddenFile).channel

                        try {
                        val projectableCount = chunkCounts.count { it > 0 }
                        var projected = 0
                        for ((i, track) in unindexed.withIndex()) {
                            val numChunks = chunkCounts[i]
                            if (numChunks == 0) continue  // failed during encode

                            ensureActive()
                            projected++
                            val projMsg = "Flamingo project $projected/$projectableCount ${track.title}"
                            _state.value = IndexingState.Processing(
                                current = projected,
                                total = projectableCount,
                                trackName = "${track.artist} - ${track.title}",
                                passName = "Flamingo project + fuse",
                                detail = "projecting + writing to DB",
                                progressFraction = fraction(),
                                estimatedRemainingMs = eta(),
                            )
                            updateNotification(projMsg, completedSteps, totalSteps)
                            val projStart = System.nanoTime()

                            // Stream hidden states from disk one chunk at a time
                            val flamingoRaw = flamingoInference.projectAndAverageStreaming(
                                numChunks = numChunks,
                                readNextChunk = {
                                    readBuf.clear()
                                    readChannel.read(readBuf)
                                    readBuf.flip()
                                    FloatArray(FlamingoInference.HIDDEN_STATE_FLOATS).also { arr ->
                                        readBuf.asFloatBuffer().get(arr)
                                    }
                                },
                            )
                            if (flamingoRaw == null) {
                                failed++
                                continue
                            }

                            // Apply flamingo_projection: 3584d → 512d
                            var flamingoReduced: FloatArray? = null
                            if (actualFlamingoProjection != null) {
                                flamingoReduced = actualFlamingoProjection.multiplyVector(flamingoRaw)
                                l2Normalize(flamingoReduced)
                            }

                            // Fuse with MuLan
                            val mulanEmbedding = mulanResults[track]?.mulanEmbedding
                            var fusedEmbedding: FloatArray? = null
                            if (mulanEmbedding != null && flamingoReduced != null && fusedProjection != null) {
                                val concat = FloatArray(1024)
                                mulanEmbedding.copyInto(concat, 0)
                                flamingoReduced.copyInto(concat, 512)
                                fusedEmbedding = fusedProjection.multiplyVector(concat)
                                l2Normalize(fusedEmbedding)
                            }

                            val merged = EmbeddingProcessor.EmbeddingResult(
                                mulanEmbedding = mulanEmbedding,
                                flamingoEmbedding = flamingoRaw,
                                flamingoReduced = flamingoReduced,
                                fusedEmbedding = fusedEmbedding,
                            )
                            val trackId = writer.writeTrack(
                                metadataKey = track.metadataKey,
                                filenameKey = track.filenameKey,
                                artist = track.artist.ifEmpty { null },
                                album = track.album.ifEmpty { null },
                                title = track.title.ifEmpty { null },
                                durationMs = track.durationMs,
                                filePath = track.path ?: "",
                                embeddings = merged,
                                source = "phone",
                            )
                            completedSteps += FLAMINGO_PROJECT_WEIGHT
                            val projMs = (System.nanoTime() - projStart) / 1_000_000
                            if (trackId > 0) {
                                indexed++
                                mulanResults.remove(track)  // Free memory
                            } else {
                                failed++
                            }
                            Log.i(TAG, "TIMING: flamingo_project " +
                                "\"${track.artist} - ${track.title}\" = ${projMs}ms")
                        }
                        } finally {
                            readChannel.close()
                        }

                        } finally {
                            hiddenFile.delete()
                        }

                        flamingoInference.close()
                        val flamingoPassMs = System.currentTimeMillis() - flamingoPassStart
                        Log.i(TAG, "TIMING: flamingo_pass_total = ${flamingoPassMs}ms " +
                            "($indexed tracks, ${flamingoPassMs / maxOf(indexed, 1)}ms/track avg)")
                    }
                }

                // Rebuild indices after indexing new tracks
                if (indexed > 0) {
                    val options = pendingFusionOptions.also { pendingFusionOptions = null }
                    val rebuildStart = System.currentTimeMillis()

                    if (options != null) {
                        val tierName = when (options.tier) {
                            FusionTier.QUICK_UPDATE -> "Quick update"
                            FusionTier.RECLUSTER -> "Re-cluster"
                            FusionTier.FULL_REFUSION -> "Full re-fusion"
                        }
                        _state.value = IndexingState.RebuildingIndices(
                            message = "Starting...", phaseName = tierName)
                        updateNotification("$tierName...")

                        val fusionEngine = FusionEngine(db, filesDir)
                        val fusionStart = System.currentTimeMillis()
                        val progressCb: (String, String, Float) -> Unit =
                            { phase, detail, fraction ->
                                val elapsed = System.currentTimeMillis() - fusionStart
                                val eta = if (fraction > 0.02f) {
                                    ((elapsed / fraction) * (1f - fraction)).toLong()
                                } else 0L
                                _state.value = IndexingState.RebuildingIndices(
                                    message = detail,
                                    phaseName = phase,
                                    progressFraction = fraction,
                                    estimatedRemainingMs = eta,
                                )
                                updateNotification("$phase: $detail")
                            }
                        when (options.tier) {
                            FusionTier.QUICK_UPDATE -> fusionEngine.quickUpdate(options.buildKnnGraph, progressCb)
                            FusionTier.RECLUSTER -> fusionEngine.recluster(options.buildKnnGraph, progressCb)
                            FusionTier.FULL_REFUSION -> fusionEngine.fullRefusion(options.buildKnnGraph, progressCb)
                        }
                        Log.i(TAG, "TIMING: fusion_${options.tier.name.lowercase()} = ${System.currentTimeMillis() - rebuildStart}ms")
                    } else {
                        // No fusion — just extract indices (indeterminate progress)
                        _state.value = IndexingState.RebuildingIndices(
                            message = "Rebuilding search indices...")
                        updateNotification("Rebuilding indices...")

                        val graphUpdater = GraphUpdater(db, filesDir)
                        graphUpdater.rebuildIndices { status ->
                            _state.value = IndexingState.RebuildingIndices(
                                message = status)
                            updateNotification(status)
                        }
                        Log.i(TAG, "TIMING: rebuild_indices = ${System.currentTimeMillis() - rebuildStart}ms")
                    }
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
     * Find the best available model variant.
     * Prefers weight-only INT8 (smaller, same quality), then FP32.
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
     *
     * Poweramp stores paths relative to its music folders. We try the path directly,
     * then common mount points.
     */
    private fun resolveAudioFile(track: NewTrackDetector.UnindexedTrack): File? {
        val path = track.path ?: return null

        // Try direct path
        val direct = File(path)
        if (direct.exists() && direct.canRead()) return direct

        // Try common Android storage prefixes
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
            Intent(this, com.powerampstartradio.indexing.IndexingActivity::class.java),
            PendingIntent.FLAG_IMMUTABLE,
        )
        val cancelIntent = PendingIntent.getService(
            this, 0,
            Intent(this, IndexingService::class.java).apply { action = ACTION_CANCEL },
            PendingIntent.FLAG_IMMUTABLE,
        )
        val title = "Indexing Tracks"
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle(title)
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
