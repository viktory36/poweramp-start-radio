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

        /** When true, run full SVD re-fusion instead of incremental graph update. */
        @Volatile
        var pendingRefusion: Boolean = false

        fun startIndexing(
            context: Context,
            selectedTracks: List<NewTrackDetector.UnindexedTrack>? = null,
            refusion: Boolean = false,
        ) {
            if (isActive) return
            pendingTracks = selectedTracks
            pendingRefusion = refusion
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
        data class RebuildingIndices(val message: String) : IndexingState()
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
                // Each track has: decode step + resample step (if caching) + inference chunks.
                // MuLan: 3 clips per chunk, chunks = max(1, min(floor(dur/60), 30))
                // Flamingo: 1 chunk per 30s, chunks = max(1, min(ceil(dur/30), 60))
                // Decode/resample steps are weighted to ~match one inference chunk's time.
                val DECODE_WEIGHT = 6  // decode takes roughly 6x one inference chunk
                val RESAMPLE_WEIGHT = 2  // resample takes roughly 2x one inference chunk
                var totalSteps = 0
                for (t in unindexed) {
                    val durS = t.durationMs / 1000f
                    if (hasMulan && durS >= 30f) {
                        totalSteps += DECODE_WEIGHT  // decode
                        if (hasFlamingo) totalSteps += RESAMPLE_WEIGHT  // resample for cache
                        val mulanChunks = maxOf(1, minOf((durS / 60f).toInt(), 30))
                        totalSteps += mulanChunks * 3  // 3 clips per chunk
                    }
                    if (hasFlamingo && durS >= 3f) {
                        if (!hasMulan) totalSteps += DECODE_WEIGHT  // decode only if not done in pass 1
                        totalSteps += maxOf(1, minOf(kotlin.math.ceil(durS / 30.0).toInt(), 60))
                    }
                }
                var completedSteps = 0

                fun eta(): Long {
                    if (completedSteps == 0) return 0L
                    val elapsed = System.currentTimeMillis() - batchStartTime
                    return elapsed * (totalSteps - completedSteps) / completedSteps
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
                    _state.value = IndexingState.Detecting("Loading MuQ-MuLan model (GPU)...")
                    updateNotification("Loading MuQ-MuLan model...")
                    val mulanInference = try { MuLanInference(mulanFile) }
                    catch (e: Exception) {
                        Log.e(TAG, "Failed to load MuQ-MuLan", e)
                        null
                    }

                    if (mulanInference != null) {
                        val processor = EmbeddingProcessor(
                            mulanModel = mulanInference,
                            flamingoModel = null,
                            flamingoProjection = null,
                            fusedProjection = null,
                        )

                        for ((i, track) in unindexed.withIndex()) {
                            ensureActive()
                            val trackProgress = "${i + 1}/${unindexed.size}"
                            var clipsDone = 0
                            fun emitProgress(detail: String, etaMs: Long = eta()) {
                                val msg = "MuLan $trackProgress ${track.title} \u2013 $detail${formatEta(etaMs)}"
                                _state.value = IndexingState.Processing(
                                    current = i + 1,
                                    total = unindexed.size,
                                    trackName = "${track.artist} - ${track.title}",
                                    passName = "MuLan pass",
                                    detail = detail,
                                    progressFraction = if (totalSteps > 0) completedSteps.toFloat() / totalSteps else 0f,
                                    estimatedRemainingMs = etaMs,
                                )
                                updateNotification(msg, completedSteps, totalSteps)
                            }
                            emitProgress("decoding")

                            val audioFile = resolveAudioFile(track)
                            if (audioFile == null) {
                                Log.w(TAG, "Cannot resolve audio file for: ${track.title}")
                                if (!hasFlamingo) failed++
                                continue
                            }

                            // Tighter pre-alloc: use actual duration instead of 900s cap.
                            // A 3-min track at 48kHz: 35MB vs 165MB with the blanket 900s cap.
                            val maxDurMulan = minOf((track.durationMs / 1000).toInt() + 10, 900)
                            val audio24k = audioDecoder.decode(audioFile, 24000, maxDurationS = maxDurMulan)
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
                        Log.i(TAG, "Pass 1 (MuLan) complete: ${mulanResults.size} tracks")
                    }
                }

                // ── Pass 2: Flamingo + Fusion ──────────────────────────
                if (hasFlamingo) {
                    _state.value = IndexingState.Detecting("Loading Flamingo model (GPU)...")
                    updateNotification("Loading Flamingo model...")
                    val flamingoInference = try {
                        FlamingoInference(flamingoFile, projectorFile)
                    } catch (e: Exception) {
                        Log.e(TAG, "Failed to load Flamingo", e)
                        null
                    }

                    if (flamingoInference != null) {
                        // Adjust projection matrix dims if projector is absent
                        val actualFlamingoProjection = if (flamingoInference.outputDim != 3584) {
                            EmbeddingProcessor.loadProjectionMatrix(
                                rawDb, "flamingo_projection",
                                flamingoInference.outputDim, 512
                            )?.transpose()
                        } else flamingoProjection

                        val processor = EmbeddingProcessor(
                            mulanModel = null,
                            flamingoModel = flamingoInference,
                            flamingoProjection = actualFlamingoProjection,
                            fusedProjection = fusedProjection,
                        )

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
                                    passName = "Flamingo pass",
                                    detail = detail,
                                    progressFraction = if (totalSteps > 0) completedSteps.toFloat() / totalSteps else 0f,
                                    estimatedRemainingMs = etaMs,
                                )
                                updateNotification(msg, completedSteps, totalSteps)
                            }

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
                                emitProgress("decoding")
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

                            // Use in-memory MuLan embedding from pass 1 for fusion
                            val mulanEmbedding = mulanResults[track]?.mulanEmbedding

                            val embeddings = processor.processTrack(
                                audioFile,
                                existingMulan = mulanEmbedding,
                                preDecodedAudio = audio16k,
                                onProgress = { /* ignore internal progress */ },
                                onChunkDone = {
                                    completedSteps++
                                    chunksDone++
                                    emitProgress("chunk $chunksDone/$totalFChunks")
                                },
                            )

                            if (embeddings == null) {
                                failed++
                                continue
                            }

                            // Merge MuLan embedding into the Flamingo result for a complete write
                            val merged = EmbeddingProcessor.EmbeddingResult(
                                mulanEmbedding = mulanEmbedding,
                                flamingoEmbedding = embeddings.flamingoEmbedding,
                                flamingoReduced = embeddings.flamingoReduced,
                                fusedEmbedding = embeddings.fusedEmbedding,
                            )
                            val trackId = writer.writeTrack(
                                metadataKey = track.metadataKey,
                                filenameKey = track.filenameKey,
                                artist = track.artist.ifEmpty { null },
                                album = track.album.ifEmpty { null },
                                title = track.title.ifEmpty { null },
                                durationMs = track.durationMs,
                                filePath = track.path ?: audioFile.absolutePath,
                                embeddings = merged,
                                source = "phone",
                            )
                            if (trackId > 0) {
                                indexed++
                                mulanResults.remove(track)  // Free memory
                            } else {
                                failed++
                            }
                            Log.i(TAG, "Track ${i + 1}/${unindexed.size} indexed: ${track.artist} - ${track.title}")
                        }

                        processor.close()
                        Log.i(TAG, "Pass 2 (Flamingo) complete")
                    }
                }

                // Rebuild indices after indexing new tracks
                if (indexed > 0) {
                    val refusion = pendingRefusion.also { pendingRefusion = false }

                    if (refusion) {
                        // Full re-fusion: recompute SVD, re-project ALL tracks,
                        // rebuild clusters and kNN graph from scratch.
                        // For users who index entirely on-device (no desktop DB).
                        _state.value = IndexingState.RebuildingIndices("Recomputing fusion...")
                        updateNotification("Recomputing fusion...")

                        val fusionEngine = FusionEngine(db, filesDir)
                        fusionEngine.recomputeFusion { status ->
                            _state.value = IndexingState.RebuildingIndices(status)
                            updateNotification(status)
                        }
                    } else {
                        // Incremental: rebuild .emb file and extract/build kNN graph.
                        // Uses the existing desktop SVD projection (adequate when
                        // adding a small number of tracks to a large desktop corpus).
                        _state.value = IndexingState.RebuildingIndices("Rebuilding search indices...")
                        updateNotification("Rebuilding indices...")

                        val graphUpdater = GraphUpdater(db, filesDir)
                        graphUpdater.rebuildIndices { status ->
                            _state.value = IndexingState.RebuildingIndices(status)
                            updateNotification(status)
                        }
                    }
                }

                db.close()

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
