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
 * Runs MuQ-MuLan + Flamingo ONNX inference on new tracks detected in the Poweramp
 * library, writes embeddings to the database, and rebuilds indices.
 *
 * Follows the same foreground service pattern as RadioService.
 */
class IndexingService : Service() {

    companion object {
        private const val TAG = "IndexingService"
        private const val NOTIFICATION_ID = 2
        private const val CHANNEL_ID = "indexing_service"

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

                // Load ONNX models
                val mulanFile = File(filesDir, "mulan_audio.onnx")
                val flamingoFile = File(filesDir, "flamingo_encoder.onnx")

                val mulanInference = if (mulanFile.exists()) {
                    updateNotification("Loading MuQ-MuLan model...")
                    try { MuLanInference(mulanFile) }
                    catch (e: Exception) {
                        Log.e(TAG, "Failed to load MuQ-MuLan", e)
                        null
                    }
                } else null

                val flamingoInference = if (flamingoFile.exists()) {
                    updateNotification("Loading Flamingo model...")
                    try { FlamingoInference(flamingoFile) }
                    catch (e: Exception) {
                        Log.e(TAG, "Failed to load Flamingo", e)
                        null
                    }
                } else null

                if (mulanInference == null && flamingoInference == null) {
                    _state.value = IndexingState.Error(
                        "No ONNX models found. Transfer mulan_audio.onnx and/or " +
                        "flamingo_encoder.onnx to ${filesDir.absolutePath}"
                    )
                    db.close()
                    stopSelf()
                    return@launch
                }

                // Load projection matrices from DB metadata
                val rawDb = db.getRawDatabase()
                val flamingoDim = flamingoInference?.outputDim ?: 3584
                val flamingoProjection = EmbeddingProcessor.loadProjectionMatrix(
                    rawDb, "flamingo_projection", flamingoDim, 512
                )
                val fusedProjection = EmbeddingProcessor.loadProjectionMatrix(
                    rawDb, "fused_projection", 512, 1024
                )

                // Load centroids for cluster assignment
                val centroids = db.loadCentroids()

                val processor = EmbeddingProcessor(
                    mulanModel = mulanInference,
                    flamingoModel = flamingoInference,
                    flamingoProjection = flamingoProjection,
                    fusedProjection = fusedProjection,
                )

                val writer = EmbeddingWriter(db, centroids)

                // Process each track
                var indexed = 0
                var failed = 0

                for ((i, track) in unindexed.withIndex()) {
                    _state.value = IndexingState.Processing(
                        current = i + 1,
                        total = unindexed.size,
                        trackName = "${track.artist} - ${track.title}",
                    )
                    updateNotification("${i + 1}/${unindexed.size}: ${track.title}")

                    // Resolve audio file path
                    val audioFile = resolveAudioFile(track)
                    if (audioFile == null) {
                        Log.w(TAG, "Cannot resolve audio file for: ${track.title}")
                        failed++
                        continue
                    }

                    val embeddings = processor.processTrack(audioFile) { status ->
                        updateNotification("${i + 1}/${unindexed.size}: $status")
                    }

                    if (embeddings == null) {
                        failed++
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

                    if (trackId > 0) indexed++ else failed++
                }

                processor.close()

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
