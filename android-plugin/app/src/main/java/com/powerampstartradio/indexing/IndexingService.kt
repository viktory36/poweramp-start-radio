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
import com.google.ai.edge.litert.Accelerator
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

        /** Model variant suffixes in preference order (GPU-optimized first). */
        private val MODEL_VARIANTS = listOf("_wo_wi8", "_fc_only_w8a16", "_fc_conv_w8a16", "")

        const val ACTION_START_INDEXING = "com.powerampstartradio.START_INDEXING"
        const val ACTION_CANCEL = "com.powerampstartradio.CANCEL_INDEXING"

        private var activeJob: Job? = null
        val isActive: Boolean get() = activeJob?.isActive == true

        // Observable state for the UI
        private val _state = MutableStateFlow<IndexingState>(IndexingState.Idle)
        val state: StateFlow<IndexingState> = _state.asStateFlow()

        fun startIndexing(context: Context) {
            if (isActive) return
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
                _state.value = IndexingState.Detecting("Detecting new tracks...")
                updateNotification("Detecting new tracks...")

                val dbFile = File(filesDir, "embeddings.db")
                if (!dbFile.exists()) {
                    _state.value = IndexingState.Error("No embedding database found")
                    stopSelf()
                    return@launch
                }

                // Open database in read-write mode
                val db = EmbeddingDatabase.openReadWrite(dbFile)

                // Detect unindexed tracks
                val detector = NewTrackDetector(db)
                val unindexed = detector.findUnindexedTracks(this@IndexingService)

                if (unindexed.isEmpty()) {
                    _state.value = IndexingState.Complete(0, 0)
                    updateNotification("All tracks already indexed")
                    db.close()
                    stopSelfDelayed()
                    return@launch
                }

                Log.i(TAG, "Found ${unindexed.size} unindexed tracks")
                updateNotification("Found ${unindexed.size} new tracks")

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

                // ── Pass 1: MuLan ──────────────────────────────────────
                // Sequential loading: load MuLan first, process all tracks,
                // close it before loading Flamingo. This avoids having multiple
                // large models in memory simultaneously.
                val accelerator = Accelerator.GPU

                if (hasMulan) {
                    updateNotification("Loading MuQ-MuLan model...")
                    val mulanInference = try { MuLanInference(mulanFile, accelerator, this@IndexingService) }
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
                            _state.value = IndexingState.Processing(
                                current = i + 1,
                                total = unindexed.size,
                                trackName = "${track.artist} - ${track.title}",
                            )
                            updateNotification("MuLan ${i + 1}/${unindexed.size}: ${track.title}")

                            val audioFile = resolveAudioFile(track)
                            if (audioFile == null) {
                                Log.w(TAG, "Cannot resolve audio file for: ${track.title}")
                                if (!hasFlamingo) failed++
                                continue
                            }

                            val embeddings = processor.processTrack(audioFile) { status ->
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
                if (hasFlamingo) {
                    updateNotification("Loading Flamingo model...")
                    val flamingoInference = try {
                        FlamingoInference(flamingoFile, projectorFile, accelerator, this@IndexingService)
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
                            _state.value = IndexingState.Processing(
                                current = i + 1,
                                total = unindexed.size,
                                trackName = "${track.artist} - ${track.title}",
                            )
                            updateNotification("Flamingo ${i + 1}/${unindexed.size}: ${track.title}")

                            val audioFile = resolveAudioFile(track)
                            if (audioFile == null) {
                                Log.w(TAG, "Cannot resolve audio file for: ${track.title}")
                                failed++
                                continue
                            }

                            // Read MuLan embedding from pass 1 for fusion
                            val existingTrackId = trackIds[track]
                            val existingMulan = if (existingTrackId != null) {
                                db.getEmbeddingFromTable("embeddings_mulan", existingTrackId)
                            } else null

                            val embeddings = processor.processTrack(
                                audioFile, existingMulan = existingMulan
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
                                )
                                if (trackId > 0) indexed++ else failed++
                            }
                        }

                        processor.close()
                        Log.i(TAG, "Pass 2 (Flamingo) complete")
                    }
                }

                // Rebuild indices
                if (indexed > 0) {
                    _state.value = IndexingState.RebuildingIndices("Rebuilding search indices...")
                    updateNotification("Rebuilding indices...")

                    val graphUpdater = GraphUpdater(db, filesDir)
                    graphUpdater.rebuildIndices { status ->
                        updateNotification(status)
                    }
                }

                db.close()

                _state.value = IndexingState.Complete(indexed, failed)
                val message = "$indexed tracks indexed" +
                    if (failed > 0) " ($failed failed)" else ""
                updateNotification(message)
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
     * Prefers NPU-optimized FC+Conv W8A16, then weight-only INT8, then FP32.
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

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        super.onDestroy()
        serviceScope.cancel()
        releaseWakeLock()
    }
}
