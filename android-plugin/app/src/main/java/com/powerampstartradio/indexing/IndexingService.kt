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
            val elapsedMs: Long = 0,
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
                    Log.i(TAG, "Using ${it.size} pre-selected tracks")
                } ?: run {
                    _state.value = IndexingState.Detecting("Detecting new tracks...")
                    updateNotification("Detecting new tracks...")
                    val detector = NewTrackDetector(db)
                    detector.findUnindexedTracks(this@IndexingService)
                }

                if (unindexed.isEmpty()) {
                    _state.value = IndexingState.Complete(0, 0)
                    updateNotification("All tracks already indexed")
                    db.close()
                    stopSelfDelayed()
                    return@launch
                }

                Log.i(TAG, "Found ${unindexed.size} unindexed tracks")
                updateNotification("Found ${unindexed.size} new tracks")
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

                // Track IDs from pass 1 for pass 2's fusion
                val trackIds = mutableMapOf<NewTrackDetector.UnindexedTrack, Long>()

                // Decode-once optimization: cache 16kHz audio during MuLan pass
                // for Flamingo pass. Avoids redundant decode+resample (~50s/track).
                val cachedAudio16k = if (hasMulan && hasFlamingo)
                    mutableMapOf<NewTrackDetector.UnindexedTrack, AudioDecoder.DecodedAudio>()
                else null
                val audioDecoder = AudioDecoder()

                // ── Pass 1: MuLan ──────────────────────────────────────
                // Sequential loading: load MuLan first, process all tracks,
                // close it before loading Flamingo. This avoids having multiple
                // large models in memory simultaneously.

                if (hasMulan) {
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
                            val elapsed = System.currentTimeMillis() - batchStartTime
                            val eta = if (i > 0) elapsed * (unindexed.size - i) / i else 0L
                            _state.value = IndexingState.Processing(
                                current = i + 1,
                                total = unindexed.size,
                                trackName = "${track.artist} - ${track.title}",
                                elapsedMs = elapsed,
                                estimatedRemainingMs = eta,
                            )
                            val etaStr = formatEta(eta)
                            updateNotification("MuLan ${i + 1}/${unindexed.size}: ${track.title}$etaStr")

                            val audioFile = resolveAudioFile(track)
                            if (audioFile == null) {
                                Log.w(TAG, "Cannot resolve audio file for: ${track.title}")
                                if (!hasFlamingo) failed++
                                continue
                            }

                            // Decode at 24kHz for MuLan
                            val audio24k = audioDecoder.decode(audioFile, 24000, maxDurationS = 900)
                            if (audio24k == null) {
                                Log.w(TAG, "Decode failed for: ${track.title}")
                                if (!hasFlamingo) failed++
                                continue
                            }

                            // Cache 16kHz version for Flamingo pass (resample 24→16kHz via soxr)
                            if (cachedAudio16k != null) {
                                try {
                                    val samples16k = audioDecoder.resample(audio24k.samples, 24000, 16000)
                                    cachedAudio16k[track] = AudioDecoder.DecodedAudio(
                                        samples16k, 16000, samples16k.size.toFloat() / 16000
                                    )
                                } catch (e: Exception) {
                                    Log.w(TAG, "16kHz cache failed for ${track.title}: ${e.message}")
                                }
                            }

                            val embeddings = processor.processTrack(
                                audioFile, preDecodedAudio = audio24k
                            ) { status ->
                                updateNotification("MuLan ${i + 1}/${unindexed.size}: $status")
                            }

                            if (embeddings == null) {
                                if (!hasFlamingo) failed++
                                continue
                            }

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
                                trackIds[track] = trackId
                                if (!hasFlamingo) indexed++
                            } else {
                                if (!hasFlamingo) failed++
                            }
                        }

                        processor.close()
                        Log.i(TAG, "Pass 1 (MuLan) complete: ${trackIds.size} tracks")
                    }
                }

                // ── Pass 2: Flamingo + Fusion ──────────────────────────
                val pass2StartTime = System.currentTimeMillis()
                if (hasFlamingo) {
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
                            val elapsed = System.currentTimeMillis() - pass2StartTime
                            val eta = if (i > 0) elapsed * (unindexed.size - i) / i else 0L
                            _state.value = IndexingState.Processing(
                                current = i + 1,
                                total = unindexed.size,
                                trackName = "${track.artist} - ${track.title}",
                                elapsedMs = elapsed,
                                estimatedRemainingMs = eta,
                            )
                            val etaStr = formatEta(eta)
                            updateNotification("Flamingo ${i + 1}/${unindexed.size}: ${track.title}$etaStr")

                            val audioFile = resolveAudioFile(track)
                            if (audioFile == null) {
                                Log.w(TAG, "Cannot resolve audio file for: ${track.title}")
                                failed++
                                continue
                            }

                            // Use cached 16kHz audio from MuLan pass (decode-once optimization)
                            val cached16k = cachedAudio16k?.remove(track)

                            // Read MuLan embedding from pass 1 for fusion
                            val existingTrackId = trackIds[track]
                            val existingMulan = if (existingTrackId != null) {
                                db.getEmbeddingFromTable("embeddings_mulan", existingTrackId)
                            } else null

                            val embeddings = processor.processTrack(
                                audioFile,
                                existingMulan = existingMulan,
                                preDecodedAudio = cached16k,
                            ) { status ->
                                updateNotification("Flamingo ${i + 1}/${unindexed.size}: $status")
                            }

                            if (embeddings == null) {
                                failed++
                                continue
                            }

                            if (existingTrackId != null) {
                                // Track exists from pass 1 — add Flamingo + fused embeddings
                                val ok = writer.addEmbeddings(existingTrackId, embeddings)
                                if (ok) indexed++ else failed++
                            } else {
                                // No MuLan pass (only Flamingo available) — write full track
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
                                if (trackId > 0) indexed++ else failed++
                            }
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

    private fun createNotification(message: String): Notification {
        val pendingIntent = PendingIntent.getActivity(
            this, 0,
            Intent(this, MainActivity::class.java),
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
            .build()
    }

    private fun updateNotification(message: String) {
        getSystemService(NotificationManager::class.java)
            .notify(NOTIFICATION_ID, createNotification(message))
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
