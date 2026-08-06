package com.powerampstartradio.services

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.net.Uri
import android.os.Build
import android.os.IBinder
import android.os.SystemClock
import android.provider.OpenableColumns
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.core.content.ContextCompat
import com.powerampstartradio.AudioLibraryPermission
import com.powerampstartradio.MainActivity
import com.powerampstartradio.R
import com.powerampstartradio.indexing.IndexingService
import com.powerampstartradio.indexing.IndexingViewModel
import com.powerampstartradio.indexing.v2.V2ActiveIndexingJobPointer
import com.powerampstartradio.indexing.v2.V2IndexGenerationReader
import com.powerampstartradio.indexing.v2.V2LibraryDatabaseResolver
import com.powerampstartradio.indexing.v2.V2ServerBundleMerger
import com.powerampstartradio.indexing.v2.V2ServerBundleRowDisposition
import com.powerampstartradio.indexing.v2.V2ServerMergeResult
import com.powerampstartradio.ui.MainViewModel
import com.powerampstartradio.ui.RadioSettingsStore
import java.util.Locale
import java.util.UUID
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

enum class ServerMergeProgressPhase {
    PREPARING,
    RELEASING_RECOMMENDATION_RESOURCES,
    WAITING_FOR_LIBRARY_INSPECTION,
    MERGING,
}

data class ServerMergeProgressState(
    val phase: ServerMergeProgressPhase,
    val detail: String,
    val completedUnits: Long? = null,
    val totalUnits: Long? = null,
    /** Stable merger-stage name when [phase] is [ServerMergeProgressPhase.MERGING]. */
    val mergeStage: String? = null,
)

internal data class ServerMergeCompletion(
    val operationId: Long,
    val activeGenerationChanged: Boolean,
)

internal data class ServerMergeServiceState(
    val operationId: Long = 0L,
    val running: Boolean = false,
    val progress: ServerMergeProgressState? = null,
    val resultText: String? = null,
    val errorText: String? = null,
    val completion: ServerMergeCompletion? = null,
)

internal data class ServerMergeSubmission(
    val accepted: Boolean,
    val errorText: String? = null,
)

/**
 * User-started foreground owner for one server-index merge.
 *
 * The merge remains deliberately non-durable: immutable staging and atomic generation publication
 * make a manual retry safe after process death. Foreground ownership exists only so leaving the
 * activity does not demote a merge that the user already started.
 */
class ServerMergeService : Service() {
    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val timedOut = AtomicBoolean(false)
    private var runnerJob: Job? = null
    private var latestStartId: Int = 0
    private var notificationStage: String? = null
    private var notificationPublishedAtElapsedMs: Long = 0L

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        latestStartId = maxOf(latestStartId, startId)
        if (runnerJob?.isActive == true) return START_NOT_STICKY
        val operationId = intent?.getLongExtra(EXTRA_OPERATION_ID, INVALID_OPERATION_ID)
            ?: INVALID_OPERATION_ID
        promoteToForeground(
            buildNotification(
                ServerMergeProgressState(
                    phase = ServerMergeProgressPhase.PREPARING,
                    detail = "Preparing the server merge",
                ),
            ),
        ) ?: run {
            claimPendingAdmission(operationId)?.release(applicationContext)
            return rejectStart(
                operationId = operationId,
                startId = startId,
                message = "Android could not keep the server merge running in the background.",
            )
        }

        val uri = intent?.takeIf { it.action == ACTION_MERGE }?.data
        val admission = claimPendingAdmission(operationId)
        if (uri == null || admission == null || timedOut.get() || runnerJob?.isActive == true) {
            admission?.release(applicationContext)
            return rejectStart(
                operationId = operationId,
                startId = startId,
                message = "The server merge request was no longer valid. Select the file again.",
            )
        }

        val job = serviceScope.launch {
            runMerge(
                operationId = operationId,
                uri = uri,
                admission = admission,
            )
        }
        runnerJob = job
        job.invokeOnCompletion {
            finishForeground()
            markOperationStopped(operationId)
        }
        return START_NOT_STICKY
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        runnerJob?.cancel()
        serviceScope.cancel()
        super.onDestroy()
    }

    override fun onTimeout(startId: Int, fgsType: Int) {
        timedOut.set(true)
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
        runnerJob?.cancel(
            CancellationException("Android's media-processing time limit ended the server merge"),
        )
    }

    private suspend fun runMerge(
        operationId: Long,
        uri: Uri,
        admission: PendingAdmission,
    ) {
        var selectedDocument = SelectedMusicIndexDocument(
            displayName = uri.lastPathSegment?.substringAfterLast('/')?.takeIf { it.isNotBlank() }
                ?: "selected file",
            byteCount = null,
        )
        var activeBefore: String? = null
        try {
            selectedDocument = selectedMusicIndexDocument(uri)
            activeBefore = activeGenerationIdOrNull()
            require(AudioLibraryPermission.isGranted(this)) {
                AudioLibraryPermission.DENIED_MESSAGE
            }
            require(V2LibraryDatabaseResolver.hasPublishedPointer(filesDir)) {
                "Import a music index before merging server embeddings"
            }
            indexingLibraryConflictReason()?.let { throw IllegalStateException(it) }

            val startedText = "Merge started · ${selectedDocument.summary}\n" +
                "Completion has not yet been recorded."
            persistResult(operationId, startedText)
            publishProgress(
                operationId,
                ServerMergeProgressState(
                    phase = ServerMergeProgressPhase.RELEASING_RECOMMENDATION_RESOURCES,
                    detail = "Releasing recommendation resources before changing the music index",
                ),
            )
            MainViewModel.releaseProcessRetrievalResourcesForIndexing()
            check(
                RadioService.suspendAndReleaseRecommendationResources(timeoutMs = 60_000L),
            ) {
                MainViewModel.MUSIC_INDEX_MUTATION_BUSY_MESSAGE
            }

            publishProgress(
                operationId,
                ServerMergeProgressState(
                    phase = ServerMergeProgressPhase.WAITING_FOR_LIBRARY_INSPECTION,
                    detail = "Waiting for the current Poweramp library comparison to finish",
                ),
            )
            val startedNs = System.nanoTime()
            val mergeContext = currentCoroutineContext()
            val result = V2ServerBundleMerger(
                context = this,
                cancellationCheck = { mergeContext.ensureActive() },
            ).merge(uri) { progress ->
                publishProgress(
                    operationId,
                    ServerMergeProgressState(
                        phase = ServerMergeProgressPhase.MERGING,
                        detail = progress.detail,
                        completedUnits = progress.completedUnits,
                        totalUnits = progress.totalUnits,
                        mergeStage = progress.stage.name,
                    ),
                )
            }
            currentCoroutineContext().ensureActive()
            if (!result.noOp) {
                invalidateUnindexedCountEvidence()
                IndexingViewModel.invalidateCache()
            }
            val elapsedMs = (System.nanoTime() - startedNs) / 1_000_000L
            val resultText = serverMergeResultText(
                result = result,
                selectedDocument = selectedDocument,
                elapsedMs = elapsedMs,
            )
            persistResult(operationId, resultText)
            completeOperation(
                operationId = operationId,
                activeGenerationChanged = !result.noOp,
            )
            logCompletion(result, elapsedMs)
        } catch (cancelled: CancellationException) {
            val activeChanged = activeGenerationIdOrNull()?.let { it != activeBefore } == true
            val reason = if (activeChanged) {
                "A new music index was published before interruption. Reopen Settings to refresh " +
                    "its details."
            } else {
                "Current music index unchanged."
            }
            val resultText = "Last merge was interrupted · ${selectedDocument.summary}\n$reason"
            persistResult(operationId, resultText)
            failOperation(operationId, resultText.substringAfter('\n'), activeChanged)
        } catch (error: Exception) {
            Log.e(TAG, "Server index merge failed", error)
            val activeChanged = activeGenerationIdOrNull()?.let { it != activeBefore } == true
            val errorText = if (activeChanged) {
                "The merged music index is active, but this screen could not refresh. " +
                    "Reopen Settings to refresh it."
            } else {
                val detail = error.message?.trim()?.takeIf(String::isNotEmpty)
                    ?: "The selected server bundle could not be merged"
                "$detail. The current music index was not changed."
            }
            val resultText = if (activeChanged) {
                "Last merge published a new index but refresh failed · " +
                    "${selectedDocument.summary}\n$errorText"
            } else {
                "Last merge failed · ${selectedDocument.summary}\n$errorText"
            }
            persistResult(operationId, resultText)
            failOperation(operationId, errorText, activeChanged)
        } finally {
            admission.release(applicationContext)
        }
    }

    private fun indexingLibraryConflictReason(): String? {
        if (IndexingService.state.value !is IndexingService.IndexingState.Idle) {
            return "Finish or discard the current indexing job before changing the music index."
        }
        val pointer = runCatching { V2ActiveIndexingJobPointer(filesDir).read() }
        if (pointer.isFailure) {
            return "The saved indexing job could not be read. Open On-device indexing before " +
                "changing the music index."
        }
        return pointer.getOrNull()?.let {
            "Finish or discard the saved indexing job before changing the music index."
        }
    }

    private fun selectedMusicIndexDocument(uri: Uri): SelectedMusicIndexDocument {
        var displayName: String? = null
        var byteCount: Long? = null
        runCatching {
            contentResolver.query(
                uri,
                arrayOf(OpenableColumns.DISPLAY_NAME, OpenableColumns.SIZE),
                null,
                null,
                null,
            )?.use { cursor ->
                if (cursor.moveToFirst()) {
                    val nameColumn = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                    val sizeColumn = cursor.getColumnIndex(OpenableColumns.SIZE)
                    if (nameColumn >= 0 && !cursor.isNull(nameColumn)) {
                        displayName = cursor.getString(nameColumn)
                    }
                    if (sizeColumn >= 0 && !cursor.isNull(sizeColumn)) {
                        byteCount = cursor.getLong(sizeColumn).takeIf { it >= 0L }
                    }
                }
            }
        }.onFailure { error ->
            Log.w(TAG, "Could not read selected server-bundle identity", error)
        }
        return SelectedMusicIndexDocument(
            displayName = displayName?.takeIf { it.isNotBlank() }
                ?: uri.lastPathSegment?.substringAfterLast('/')?.takeIf { it.isNotBlank() }
                ?: "selected file",
            byteCount = byteCount,
        )
    }

    private fun serverMergeResultText(
        result: V2ServerMergeResult,
        selectedDocument: SelectedMusicIndexDocument,
        elapsedMs: Long,
    ): String {
        val counts = result.rowOutcomes.groupingBy { it.disposition }.eachCount()
        val parts = mutableListOf("${result.addedTrackCount} added")
        counts[V2ServerBundleRowDisposition.ALREADY_INDEXED]
            ?.takeIf { it > 0 }
            ?.let { parts += "$it already indexed" }
        counts[V2ServerBundleRowDisposition.NOT_IN_POWERAMP_LIBRARY]
            ?.takeIf { it > 0 }
            ?.let { parts += "$it not in Poweramp" }
        counts[V2ServerBundleRowDisposition.CUE_OR_SHARED_SOURCE]
            ?.takeIf { it > 0 }
            ?.let { parts += "$it CUE or shared-source" }
        counts[V2ServerBundleRowDisposition.AMBIGUOUS_POWERAMP_PATH]
            ?.takeIf { it > 0 }
            ?.let { parts += "$it ambiguous" }
        counts[V2ServerBundleRowDisposition.SOURCE_FILE_UNAVAILABLE]
            ?.takeIf { it > 0 }
            ?.let { parts += "$it source files unavailable" }
        counts[V2ServerBundleRowDisposition.SOURCE_BYTES_MISMATCH]
            ?.takeIf { it > 0 }
            ?.let { parts += "$it phone files differ from server" }
        val bundleIdentity = result.sourceValidation.bundleId
            .removePrefix("server-bundle-v1-")
            .take(12)
        val activeTrackCountBefore = result.generation.manifest.trackCount - result.addedTrackCount
        val elapsed = if (elapsedMs < 60_000L) {
            "%.1f s".format(Locale.US, elapsedMs / 1_000.0)
        } else {
            "${elapsedMs / 60_000L} min ${(elapsedMs % 60_000L) / 1_000L} s"
        }
        return buildString {
            append(if (result.noOp) "Last merge made no changes" else "Last merge completed")
            append(" · ")
            append(selectedDocument.summary)
            append('\n')
            append(result.sourceValidation.tracks.size)
            append(" server embeddings · ")
            append(parts.joinToString(" · "))
            append('\n')
            append("Music index ")
            append(activeTrackCountBefore)
            append(" → ")
            append(result.generation.manifest.trackCount)
            append(" tracks · bundle ")
            append(bundleIdentity)
            append(" · ")
            append(elapsed)
        }
    }

    private fun logCompletion(result: V2ServerMergeResult, elapsedMs: Long) {
        val dispositionCounts = result.rowOutcomes
            .groupingBy { it.disposition }
            .eachCount()
            .entries
            .sortedBy { it.key.name }
            .joinToString(",") { "${it.key.name}:${it.value}" }
        val matchEvidenceCounts = result.rowOutcomes
            .mapNotNull { it.matchEvidence }
            .groupingBy { it }
            .eachCount()
            .entries
            .sortedBy { it.key.name }
            .joinToString(",") { "${it.key.name}:${it.value}" }
        Log.i(
            TAG,
            "Server index merge completed: added=${result.addedTrackCount} " +
                "noOp=${result.noOp} elapsedMs=$elapsedMs " +
                "bundle=${result.sourceValidation.bundleId} " +
                "dispositions=[$dispositionCounts] matchEvidence=[$matchEvidenceCounts]",
        )
    }

    private fun activeGenerationIdOrNull(): String? = runCatching {
        if (V2LibraryDatabaseResolver.hasPublishedPointer(filesDir)) {
            V2IndexGenerationReader.requireActivePointer(filesDir).generationId
        } else {
            null
        }
    }.getOrNull()

    private fun invalidateUnindexedCountEvidence() {
        getSharedPreferences(RadioSettingsStore.PREFERENCES_NAME, Context.MODE_PRIVATE)
            .edit()
            .remove("unindexed_count")
            .remove("unindexed_count_database_generation")
            .remove("unindexed_count_provider_generation")
            .remove("unindexed_count_exclusions_fingerprint")
            .remove("unindexed_count_attention_fingerprint")
            .remove("unindexed_count_detection_policy")
            .apply()
    }

    private fun publishProgress(operationId: Long, progress: ServerMergeProgressState) {
        if (!updateState(operationId) { it.copy(progress = progress) }) return
        updateNotification(progress)
    }

    private fun persistResult(operationId: Long, resultText: String) {
        getSharedPreferences(RadioSettingsStore.PREFERENCES_NAME, Context.MODE_PRIVATE)
            .edit()
            .putString(LAST_RESULT_PREF, resultText)
            .apply()
        updateState(operationId) { it.copy(resultText = resultText) }
    }

    private fun completeOperation(operationId: Long, activeGenerationChanged: Boolean) {
        updateState(operationId) {
            it.copy(
                progress = null,
                errorText = null,
                completion = ServerMergeCompletion(operationId, activeGenerationChanged),
            )
        }
    }

    private fun failOperation(
        operationId: Long,
        errorText: String,
        activeGenerationChanged: Boolean,
    ) {
        updateState(operationId) {
            it.copy(
                progress = null,
                errorText = errorText,
                completion = ServerMergeCompletion(operationId, activeGenerationChanged),
            )
        }
    }

    private fun updateNotification(progress: ServerMergeProgressState) {
        val now = SystemClock.elapsedRealtime()
        val stage = progress.mergeStage ?: progress.phase.name
        if (stage == notificationStage &&
            now - notificationPublishedAtElapsedMs < NOTIFICATION_UPDATE_INTERVAL_MS
        ) {
            return
        }
        notificationStage = stage
        notificationPublishedAtElapsedMs = now
        getSystemService(NotificationManager::class.java).notify(
            NOTIFICATION_ID,
            buildNotification(progress),
        )
    }

    private fun buildNotification(progress: ServerMergeProgressState): Notification {
        val contentIntent = PendingIntent.getActivity(
            this,
            0,
            Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        val completed = progress.completedUnits
        val total = progress.totalUnits
        val determinate = completed != null && total != null && total > 0L && completed in 0L..total
        return NotificationCompat.Builder(this, NOTIFICATION_CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_radio)
            .setContentTitle("Merging server index")
            .setContentText(progress.detail.substringBefore('\n'))
            .setStyle(NotificationCompat.BigTextStyle().bigText(progress.detail))
            .setContentIntent(contentIntent)
            .setCategory(NotificationCompat.CATEGORY_PROGRESS)
            .setVisibility(NotificationCompat.VISIBILITY_PUBLIC)
            .setOnlyAlertOnce(true)
            .setOngoing(true)
            .setProgress(
                if (determinate) 100 else 0,
                if (determinate) ((completed!! * 100L) / total!!).toInt() else 0,
                !determinate,
            )
            .build()
    }

    private fun promoteToForeground(notification: Notification): Notification? = try {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.VANILLA_ICE_CREAM) {
            startForeground(
                NOTIFICATION_ID,
                notification,
                ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROCESSING,
            )
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }
        notificationStage = ServerMergeProgressPhase.PREPARING.name
        notificationPublishedAtElapsedMs = SystemClock.elapsedRealtime()
        notification
    } catch (error: RuntimeException) {
        Log.w(TAG, "Android rejected server-merge foreground promotion", error)
        null
    }

    private fun rejectStart(operationId: Long, startId: Int, message: String): Int {
        if (operationId != INVALID_OPERATION_ID) {
            failOperation(operationId, message, activeGenerationChanged = false)
            markOperationStopped(operationId)
        }
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelfResult(startId)
        return START_NOT_STICKY
    }

    private fun markOperationStopped(operationId: Long) {
        updateState(operationId) { it.copy(running = false) }
    }

    private fun finishForeground() {
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelfResult(latestStartId)
    }

    private fun createNotificationChannel() {
        getSystemService(NotificationManager::class.java).createNotificationChannel(
            NotificationChannel(
                NOTIFICATION_CHANNEL_ID,
                "Server index merge",
                NotificationManager.IMPORTANCE_LOW,
            ).apply {
                description = "Progress while adding server-computed embeddings"
                setShowBadge(false)
            },
        )
    }

    private data class SelectedMusicIndexDocument(
        val displayName: String,
        val byteCount: Long?,
    ) {
        val summary: String
            get() = buildString {
                append(displayName)
                byteCount?.let { bytes ->
                    append(" · ")
                    append("%.1f MiB".format(Locale.US, bytes.toDouble() / BYTES_PER_MIB))
                }
            }
    }

    private data class PendingAdmission(
        val mutationOwner: MusicIndexMutationAdmission.Owner,
        val recommendationOwner: RecommendationWorkAdmission.ReservationOwner,
    ) {
        fun release(context: Context) {
            val globallyAvailable = RecommendationWorkAdmission.release(recommendationOwner)
            check(MusicIndexMutationAdmission.process.release(mutationOwner)) {
                "server merge lost music-index mutation admission"
            }
            if (globallyAvailable) RadioService.kickDeferredRecovery(context)
        }
    }

    companion object {
        private const val TAG = "ServerMergeService"
        private const val ACTION_MERGE = "com.powerampstartradio.action.MERGE_SERVER_INDEX"
        private const val EXTRA_OPERATION_ID = "operation_id"
        private const val INVALID_OPERATION_ID = -1L
        private const val NOTIFICATION_CHANNEL_ID = "server_index_merge"
        private const val NOTIFICATION_ID = 3
        private const val NOTIFICATION_UPDATE_INTERVAL_MS = 15_000L
        private const val LAST_RESULT_PREF = "last_server_merge_result_v2"
        private const val BYTES_PER_MIB = 1024L * 1024L

        private val stateLock = Any()
        private val operationIds = AtomicLong(0L)
        private val pendingAdmissions = mutableMapOf<Long, PendingAdmission>()
        private val _state = MutableStateFlow(ServerMergeServiceState())
        internal val state: StateFlow<ServerMergeServiceState> = _state.asStateFlow()

        internal fun submit(context: Context, uri: Uri): ServerMergeSubmission =
            synchronized(stateLock) {
                if (_state.value.running || pendingAdmissions.isNotEmpty()) {
                    return@synchronized ServerMergeSubmission(
                        accepted = false,
                        errorText = "A music-index update is already running.",
                    )
                }
                val operationId = operationIds.incrementAndGet()
                val mutationOwner = MusicIndexMutationAdmission.newOwner(
                    "server-merge:$operationId:${UUID.randomUUID()}",
                )
                if (!MusicIndexMutationAdmission.process.tryAcquire(mutationOwner)) {
                    return@synchronized ServerMergeSubmission(
                        accepted = false,
                        errorText = "A music-index update is already running.",
                    )
                }
                val recommendationOwner = RecommendationWorkAdmission.musicIndexMutationOwner(
                    "server-merge:$operationId:${UUID.randomUUID()}",
                )
                RecommendationWorkAdmission.reserve(recommendationOwner)
                pendingAdmissions[operationId] = PendingAdmission(
                    mutationOwner = mutationOwner,
                    recommendationOwner = recommendationOwner,
                )
                _state.value = ServerMergeServiceState(
                    operationId = operationId,
                    running = true,
                    progress = ServerMergeProgressState(
                        phase = ServerMergeProgressPhase.PREPARING,
                        detail = "Preparing the server merge",
                    ),
                )
                val intent = Intent(context, ServerMergeService::class.java).apply {
                    action = ACTION_MERGE
                    data = uri
                    putExtra(EXTRA_OPERATION_ID, operationId)
                    addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                }
                try {
                    ContextCompat.startForegroundService(context, intent)
                    ServerMergeSubmission(accepted = true)
                } catch (error: Throwable) {
                    pendingAdmissions.remove(operationId)?.release(context.applicationContext)
                    val message = "Android could not start the server merge in the background."
                    _state.value = _state.value.copy(
                        running = false,
                        progress = null,
                        errorText = message,
                        completion = ServerMergeCompletion(operationId, false),
                    )
                    Log.w(TAG, "Unable to launch server merge service", error)
                    ServerMergeSubmission(accepted = false, errorText = message)
                }
            }

        internal fun lastResult(context: Context): String? =
            context.getSharedPreferences(RadioSettingsStore.PREFERENCES_NAME, Context.MODE_PRIVATE)
                .getString(LAST_RESULT_PREF, null)

        private fun claimPendingAdmission(operationId: Long): PendingAdmission? =
            synchronized(stateLock) { pendingAdmissions.remove(operationId) }

        private fun updateState(
            operationId: Long,
            transform: (ServerMergeServiceState) -> ServerMergeServiceState,
        ): Boolean = synchronized(stateLock) {
            val current = _state.value
            if (current.operationId != operationId) return@synchronized false
            _state.value = transform(current)
            true
        }
    }
}
