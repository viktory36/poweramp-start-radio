package com.powerampstartradio.indexing

import android.Manifest
import android.app.Application
import android.content.Context
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.util.Log
import androidx.core.content.ContextCompat
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.powerampstartradio.data.EmbeddedTrack
import com.powerampstartradio.data.EmbeddingDatabase
import com.powerampstartradio.indexing.v2.FailureDisposition
import com.powerampstartradio.indexing.v2.RetryTrigger
import com.powerampstartradio.indexing.v2.TrackFailureCode
import com.powerampstartradio.indexing.v2.V2CurrentModelPolicyResolver
import com.powerampstartradio.indexing.v2.V2IndexingExecutionProfile
import com.powerampstartradio.indexing.v2.V2IndexingPreflightIntentFactory
import com.powerampstartradio.indexing.v2.V2IndexingJobRepository
import com.powerampstartradio.indexing.v2.V2IndexingPreflightSelectionFactory
import com.powerampstartradio.indexing.v2.V2LibraryMaintenancePublisher
import com.powerampstartradio.indexing.v2.V2LibraryDatabaseResolver
import com.powerampstartradio.indexing.v2.V2PowerampProviderSnapshotAcquirer
import com.powerampstartradio.indexing.v2.V2ProviderSpanReceiptReader
import com.powerampstartradio.poweramp.PowerampHelper
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Deferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.launch
import java.io.BufferedInputStream
import java.io.File
import java.io.FileInputStream
import java.io.BufferedOutputStream
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream
import java.util.UUID
import java.util.concurrent.atomic.AtomicBoolean

/**
 * ViewModel for the IndexingActivity track selection and indexing UI.
 *
 * Manages the list of unindexed tracks, user selection, dismissed tracks
 * ("never-index"), ignored tracks ("previously ignored"), and delegates
 * indexing to IndexingService.
 */
class IndexingViewModel(application: Application) : AndroidViewModel(application) {
    sealed class ExportState {
        data object Idle : ExportState()
        data object ChoosingDestination : ExportState()
        data class Exporting(val message: String) : ExportState()
        data class Complete(val filename: String) : ExportState()
        data class Error(val message: String) : ExportState()
    }

    sealed interface PlanningState {
        data object Idle : PlanningState
        data class Planning(val message: String) : PlanningState
        data class Failed(val message: String) : PlanningState
    }

    data class FailedTrackUi(
        val jobId: String,
        val workId: String,
        val powerampFileId: Long,
        val title: String,
        val artist: String,
        val code: TrackFailureCode,
        val diagnostic: String,
        val occurrences: Int,
        val retryTrigger: RetryTrigger,
        val disposition: FailureDisposition,
        val stableTrackSpanId: String,
        val providerPhysicalPath: String,
        val offsetMs: Long,
        val durationMs: Long,
        val jobState: com.powerampstartradio.indexing.v2.IndexingJobState,
    )

    data class SharedDetectionResult(
        val tracks: List<NewTrackDetector.UnindexedTrack>,
        val databaseGeneration: String,
        val providerGeneration: String,
    )

    companion object {
        private const val TAG = "IndexingViewModel"

        private data class CachedDetection(
            val tracks: List<NewTrackDetector.UnindexedTrack>,
            val databaseGeneration: String,
            val providerGeneration: String,
        )

        private data class CachedDatabaseOnlyDetection(
            val tracks: List<EmbeddedTrack>,
            val databaseGeneration: String,
            val providerGeneration: String,
        )

        private data class CompletedDetectionHandoff(
            val result: SharedDetectionResult,
            val completedAtElapsedMs: Long,
        )

        internal const val COMPLETED_DETECTION_HANDOFF_TTL_MS = 120_000L
        private val completedDetectionHandoffLock = Any()

        /**
         * Cached detection result shared across ViewModel instances.
         * Avoids repeating reconciliation when the exact immutable database generation and the
         * complete Poweramp provider generation are unchanged.
         */
        @Volatile private var cachedDetection: CachedDetection? = null
        @Volatile private var cachedDatabaseOnlyDetection: CachedDatabaseOnlyDetection? = null
        @Volatile private var completedDetectionHandoff: CompletedDetectionHandoff? = null
        private val _ownedDetectionResults = MutableSharedFlow<SharedDetectionResult>(
            extraBufferCapacity = 1,
            onBufferOverflow = BufferOverflow.DROP_OLDEST,
        )
        val ownedDetectionResults: SharedFlow<SharedDetectionResult> =
            _ownedDetectionResults.asSharedFlow()

        /**
         * A pending detection started by MainViewModel. IndexingViewModel will
         * await this instead of starting a duplicate scan.
         */
        @Volatile var pendingDetection: Deferred<SharedDetectionResult>? = null

        /** Live progress from any ongoing detection (observed by both ViewModels). */
        val detectionStatus = MutableStateFlow<String?>(null)

        /** Clear the cached detection result (e.g. after DB import). */
        fun invalidateCache() {
            cachedDetection = null
            cachedDatabaseOnlyDetection = null
            discardCompletedDetectionHandoff()
        }

        /** Store results from an external detection (e.g. MainViewModel's check). */
        fun cacheResults(
            tracks: List<NewTrackDetector.UnindexedTrack>,
            databaseGeneration: String,
            providerGeneration: String,
        ) {
            cachedDetection = CachedDetection(
                tracks = tracks,
                databaseGeneration = databaseGeneration,
                providerGeneration = providerGeneration,
            )
        }

        /** Reuse only a completed answer bound to both immutable library generations. */
        internal fun exactCachedResult(
            databaseGeneration: String,
            providerGeneration: String,
        ): SharedDetectionResult? = cachedDetection?.takeIf { cached ->
            databaseGeneration.isNotBlank() && providerGeneration.isNotBlank() &&
                cached.databaseGeneration == databaseGeneration &&
                cached.providerGeneration == providerGeneration
        }?.let { cached ->
            SharedDetectionResult(
                tracks = cached.tracks,
                databaseGeneration = cached.databaseGeneration,
                providerGeneration = cached.providerGeneration,
            )
        }

        /**
         * A Settings-owned detection is a complete provider snapshot. Manage Tracks may consume it
         * directly while the same immutable database generation is still active; job preflight
         * revalidates every selected provider row before any embedding work is published.
         */
        internal fun matchingPendingResult(
            databaseGeneration: String,
            result: SharedDetectionResult?,
        ): SharedDetectionResult? = result?.takeIf {
            databaseGeneration.isNotBlank() &&
                it.databaseGeneration == databaseGeneration &&
                it.providerGeneration.isNotBlank()
        }

        /**
         * Publishes one recent Settings result for the next Manage Tracks screen. This is separate
         * from the unbounded exact-generation cache because consuming a completed UI handoff must
         * not silently authorize stale provider state.
         */
        internal fun offerCompletedDetectionHandoff(
            result: SharedDetectionResult,
            completedAtElapsedMs: Long = monotonicElapsedMs(),
        ) {
            require(result.databaseGeneration.isNotBlank())
            require(result.providerGeneration.isNotBlank())
            cacheResults(
                tracks = result.tracks,
                databaseGeneration = result.databaseGeneration,
                providerGeneration = result.providerGeneration,
            )
            synchronized(completedDetectionHandoffLock) {
                completedDetectionHandoff = CompletedDetectionHandoff(
                    result = result,
                    completedAtElapsedMs = completedAtElapsedMs,
                )
            }
        }

        internal fun consumeCompletedDetectionHandoff(
            databaseGeneration: String,
            nowElapsedMs: Long = monotonicElapsedMs(),
        ): SharedDetectionResult? = synchronized(completedDetectionHandoffLock) {
            val handoff = completedDetectionHandoff ?: return@synchronized null
            completedDetectionHandoff = null
            val ageMs = nowElapsedMs - handoff.completedAtElapsedMs
            handoff.result.takeIf {
                databaseGeneration.isNotBlank() &&
                    it.databaseGeneration == databaseGeneration &&
                    it.providerGeneration.isNotBlank() &&
                    ageMs in 0L..COMPLETED_DETECTION_HANDOFF_TTL_MS
            }
        }

        internal fun discardCompletedDetectionHandoff() {
            synchronized(completedDetectionHandoffLock) {
                completedDetectionHandoff = null
            }
        }

        internal fun publishOwnedDetectionResult(result: SharedDetectionResult) {
            _ownedDetectionResults.tryEmit(result)
        }

        private fun monotonicElapsedMs(): Long = System.nanoTime() / 1_000_000L
    }

    private val prefs by lazy(LazyThreadSafetyMode.SYNCHRONIZED) {
        application.getSharedPreferences("indexing", Context.MODE_PRIVATE)
    }
    private val exclusionRepository by lazy(LazyThreadSafetyMode.SYNCHRONIZED) {
        V2TrackExclusionRepository(application)
    }
    private val preflightRetryChoiceRepository by lazy(LazyThreadSafetyMode.SYNCHRONIZED) {
        V2PreflightRetryChoiceRepository(application)
    }
    private val preflightRejectionHistorySource by lazy(LazyThreadSafetyMode.SYNCHRONIZED) {
        V2AtomicPreflightRejectionHistorySource(application.filesDir)
    }
    private val exclusionsLoaded = CompletableDeferred<Unit>()
    private val durableStateLoaded = CompletableDeferred<Unit>()
    private val appFileRefreshRequests = Channel<Unit>(Channel.CONFLATED)
    private val uiOperationAdmission = IndexingUiOperationAdmission()
    private var dismissedExclusions = emptyList<V2PersistedTrackExclusion>()
    private var ignoredExclusions = emptyList<V2PersistedTrackExclusion>()
    @Volatile private var retainedPreflightRejections = emptyList<V2PreflightRejectedSpan>()
    @Volatile private var preflightRetryChoiceSpans = emptySet<V2ProviderSpanLocator>()
    @Volatile private var retainedUnresolvedFailures = emptyList<FailedTrackUi>()
    private val preflightMetadataHydrationActive = AtomicBoolean(false)

    private val _unindexedTracks = MutableStateFlow<List<NewTrackDetector.UnindexedTrack>>(emptyList())
    val unindexedTracks: StateFlow<List<NewTrackDetector.UnindexedTrack>> = _unindexedTracks.asStateFlow()

    private val _selectedIds = MutableStateFlow<Set<Long>>(emptySet())
    val selectedIds: StateFlow<Set<Long>> = _selectedIds.asStateFlow()

    /** "Never-index" list: tracks hidden only by an explicit user action. */
    private val _dismissedIds = MutableStateFlow<Set<Long>>(emptySet())
    val dismissedIds: StateFlow<Set<Long>> = _dismissedIds.asStateFlow()

    /** Legacy V1 ignored list. V2 never adds tracks to it implicitly. */
    private val _ignoredIds = MutableStateFlow<Set<Long>>(emptySet())
    val ignoredIds: StateFlow<Set<Long>> = _ignoredIds.asStateFlow()

    private val _isDetecting = MutableStateFlow(false)
    val isDetecting: StateFlow<Boolean> = _isDetecting.asStateFlow()

    private val _detectingStatus = MutableStateFlow("")
    val detectingStatus: StateFlow<String> = _detectingStatus.asStateFlow()

    private val _detectionError = MutableStateFlow<String?>(null)
    val detectionError: StateFlow<String?> = _detectionError.asStateFlow()

    private val _hasModels = MutableStateFlow(false)
    val hasModels: StateFlow<Boolean> = _hasModels.asStateFlow()

    private val _isAppFilesChecking = MutableStateFlow(true)
    val isAppFilesChecking: StateFlow<Boolean> = _isAppFilesChecking.asStateFlow()

    private val _isInitializing = MutableStateFlow(true)
    val isInitializing: StateFlow<Boolean> = _isInitializing.asStateFlow()

    private val _hasAudioAccess = MutableStateFlow(false)
    val hasAudioAccess: StateFlow<Boolean> = _hasAudioAccess.asStateFlow()

    private val _hasDatabase = MutableStateFlow(false)
    val hasDatabase: StateFlow<Boolean> = _hasDatabase.asStateFlow()

    private val _databaseCleanupState =
        MutableStateFlow<DatabaseCleanupScanState>(DatabaseCleanupScanState.Idle)
    internal val databaseCleanupState: StateFlow<DatabaseCleanupScanState> =
        _databaseCleanupState.asStateFlow()

    private val _exportState = MutableStateFlow<ExportState>(ExportState.Idle)
    val exportState: StateFlow<ExportState> = _exportState.asStateFlow()

    private val _planningState = MutableStateFlow<PlanningState>(PlanningState.Idle)
    val planningState: StateFlow<PlanningState> = _planningState.asStateFlow()

    private val _executionProfile = MutableStateFlow(V2IndexingExecutionProfile.FULL)
    val executionProfile: StateFlow<V2IndexingExecutionProfile> = _executionProfile.asStateFlow()

    private val _failedTracks = MutableStateFlow<List<FailedTrackUi>>(emptyList())
    val failedTracks: StateFlow<List<FailedTrackUi>> = _failedTracks.asStateFlow()

    private val _preflightAttentionTracks =
        MutableStateFlow<List<V2PreflightAttentionTrack>>(emptyList())
    val preflightAttentionTracks: StateFlow<List<V2PreflightAttentionTrack>> =
        _preflightAttentionTracks.asStateFlow()

    private val _preflightHistoryError = MutableStateFlow<String?>(null)
    val preflightHistoryError: StateFlow<String?> = _preflightHistoryError.asStateFlow()

    private val _trackExclusionError = MutableStateFlow<String?>(null)
    val trackExclusionError: StateFlow<String?> = _trackExclusionError.asStateFlow()

    val indexingState: StateFlow<IndexingService.IndexingState> = IndexingService.state

    init {
        viewModelScope.launch(Dispatchers.IO) {
            try {
                val loaded = exclusionRepository.loadPersisted()
                dismissedExclusions = loaded.first
                ignoredExclusions = emptyList()
                preflightRetryChoiceSpans = preflightRetryChoiceRepository.load()
                _trackExclusionError.value = null
            } catch (error: Throwable) {
                Log.e(TAG, "Unable to load saved track exclusions", error)
                _trackExclusionError.value = trackExclusionUnavailableMessage()
                _selectedIds.value = emptySet()
            } finally {
                publishPreflightAttention()
                exclusionsLoaded.complete(Unit)
            }
        }
        viewModelScope.launch(Dispatchers.IO) {
            val repository = runCatching {
                V2IndexingJobRepository.get(application)
            }.onFailure { error ->
                Log.e(TAG, "Unable to open durable indexing state", error)
            }.getOrNull()
            if (repository != null) {
                runCatching {
                    repository.refresh()
                    IndexingService.attach(application)
                }.onFailure { error ->
                    Log.e(TAG, "Unable to refresh durable indexing state", error)
                }
                publishUnresolvedFailures(latestUnresolvedFailures(repository.jobs.value))
            }
            refreshPreflightIntentHistory()
            hydrateActivePreflightAttentionMetadataIfNeeded()
            durableStateLoaded.complete(Unit)
            if (repository != null) {
                repository.jobs.collect { ledgers ->
                    publishUnresolvedFailures(latestUnresolvedFailures(ledgers))
                }
            }
        }
        viewModelScope.launch(Dispatchers.IO) {
            var previousLifecycleKey: String? = null
            IndexingService.state.collect { current ->
                val lifecycleKey = when (current) {
                    IndexingService.IndexingState.Idle -> "idle"
                    is IndexingService.IndexingState.PreflightSnapshot ->
                        "preflight:${current.jobId}:${current.state}"
                    is IndexingService.IndexingState.JobSnapshot ->
                        "job:${current.jobId}:${current.jobState}"
                    is IndexingService.IndexingState.Error ->
                        "error:${current.jobId}:${current.message}"
                }
                if (lifecycleKey != previousLifecycleKey) {
                    previousLifecycleKey = lifecycleKey
                    refreshPreflightIntentHistory()
                    hydrateActivePreflightAttentionMetadataIfNeeded()
                }
            }
        }
        viewModelScope.launch(Dispatchers.IO) {
            for (ignored in appFileRefreshRequests) {
                _isAppFilesChecking.value = true
                do {
                    runCatching(::refreshAppFilesBlocking).onFailure { error ->
                        Log.e(TAG, "Unable to check app indexing files", error)
                    }
                } while (appFileRefreshRequests.tryReceive().isSuccess)
                _isAppFilesChecking.value = false
            }
        }
        viewModelScope.launch {
            exclusionsLoaded.await()
            durableStateLoaded.await()
            _isInitializing.value = false
        }
        refreshAppFiles()
    }

    fun detectUnindexed(forceRefresh: Boolean = false) {
        if (_isDetecting.value) return

        // Reset service state from Complete/Error so UI shows detecting spinner
        IndexingService.resetState()
        _detectionError.value = null
        _isDetecting.value = true
        _detectingStatus.value = "Reading Poweramp tracks and exact indexed source spans"

        viewModelScope.launch(Dispatchers.IO) {
            try {
                exclusionsLoaded.await()
                durableStateLoaded.await()
                if (savedIndexingStateError() != null) return@launch
                if (IndexingService.state.value !is IndexingService.IndexingState.Idle) {
                    return@launch
                }
                val app = getApplication<Application>()
                _detectingStatus.value = "Validating the published music-index files"
                val resolvedLibrary = V2LibraryDatabaseResolver.resolveOrNull(app.filesDir) {
                        progress ->
                    _detectingStatus.value = exactHashProgressText(
                        subject = "active music-index file ${progress.filename}",
                        completedBytes = progress.completedBytes,
                        totalBytes = progress.totalBytes,
                    )
                }
                val dbFile = resolvedLibrary?.databaseFile
                _hasDatabase.value = dbFile?.isFile == true
                val databaseGeneration = resolvedLibrary?.activeGeneration?.manifest?.generationId
                    .orEmpty()

                if (dbFile?.isFile != true) {
                    _unindexedTracks.value = emptyList()
                    invalidateCache()
                    _detectionError.value = "No music index is loaded."
                    return@launch
                }
                if (!PowerampHelper.canAccessData(app)) {
                    throw IllegalStateException(
                        "Poweramp library access is not granted. Grant access, then compare again."
                    )
                }

                // Consume a just-completed Settings-owned snapshot for this database generation.
                // Force refreshes, failed handoffs, and generation mismatches still scan afresh.
                val sharedResult = if (!forceRefresh) pendingDetection?.let { pending ->
                    _detectingStatus.value =
                        detectionStatus.value ?: "Comparing Poweramp with the music index..."
                    val statusJob = launch {
                        detectionStatus.collect { status ->
                            if (status != null) _detectingStatus.value = status
                        }
                    }
                    val result = try {
                        pending.await()
                    } catch (error: Exception) {
                        currentCoroutineContext().ensureActive()
                        Log.w(TAG, "Shared detection failed; continuing with an owned scan", error)
                        null
                    } finally {
                        statusJob.cancel()
                    }
                    result
                } else null

                val completedHandoff = consumeCompletedDetectionHandoff(databaseGeneration)
                val handedOff = if (forceRefresh) {
                    null
                } else {
                    completedHandoff ?: matchingPendingResult(databaseGeneration, sharedResult)
                }
                val (result, recomputed) = if (handedOff != null) {
                    Log.i(TAG, "Consuming Settings-owned unindexed detection")
                    cacheResults(
                        tracks = handedOff.tracks,
                        databaseGeneration = handedOff.databaseGeneration,
                        providerGeneration = handedOff.providerGeneration,
                    )
                    handedOff to false
                } else {
                    if (sharedResult != null) {
                        Log.i(TAG, "Ignored shared detection for a different database generation")
                    }
                    V2ProcessLibraryInspectionCoordinator.inspect {
                        _detectingStatus.value = "Opening the complete Poweramp library"
                        val providerSnapshot =
                            V2PowerampProviderSnapshotAcquirer(app).acquireBlocking {
                                    completedRows,
                                    totalRows,
                                ->
                                _detectingStatus.value = powerampLibraryReadProgressText(
                                    completedRows,
                                    totalRows,
                                )
                            }
                        val providerGeneration = requireNotNull(providerSnapshot.libraryGeneration) {
                            "Poweramp snapshot has no complete library generation"
                        }

                        // Reuse only an answer bound to both exact, complete library generations.
                        val cached = cachedDetection
                        if (!forceRefresh && databaseGeneration.isNotEmpty() &&
                            cached != null && dbFile.isFile &&
                            databaseGeneration == cached.databaseGeneration &&
                            providerGeneration == cached.providerGeneration
                        ) {
                            Log.i(TAG, "Reusing exact-generation unindexed cache")
                            return@inspect SharedDetectionResult(
                                tracks = cached.tracks,
                                databaseGeneration = cached.databaseGeneration,
                                providerGeneration = cached.providerGeneration,
                            ) to false
                        }

                        val activeCatalog = V2ActiveLibraryCatalogStore(app.filesDir)
                            .read(resolvedLibrary.activeGeneration)
                            ?.takeIf {
                                it.generationBinding.providerGenerationId == providerGeneration
                            }
                        val database = EmbeddingDatabase.open(dbFile)
                        val detected = try {
                            NewTrackDetector(database).findUnindexedTracks(
                                snapshot = providerSnapshot,
                                activeCatalog = activeCatalog,
                            ) { status -> _detectingStatus.value = status }
                        } finally {
                            database.close()
                        }
                        val sorted = detected.sortedByDescending { it.durationMs }
                        cachedDetection = CachedDetection(
                            tracks = sorted,
                            databaseGeneration = databaseGeneration,
                            providerGeneration = providerGeneration,
                        )
                        SharedDetectionResult(
                            tracks = sorted,
                            databaseGeneration = databaseGeneration,
                            providerGeneration = providerGeneration,
                        ) to true
                    }
                }
                if (handedOff == null) discardCompletedDetectionHandoff()
                val tracks = result.tracks
                _unindexedTracks.value = tracks
                autoSelect(tracks)
                if (recomputed) updateDismissedFingerprint()
                if (recomputed) publishOwnedDetectionResult(result)
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (e: Exception) {
                _unindexedTracks.value = emptyList()
                Log.e(TAG, "Unindexed scan failed", e)
                _detectionError.value = indexingListenerFailureText(
                    IndexingListenerFailureOperation.NEW_TRACK_SCAN,
                )
            } finally {
                _isDetecting.value = false
            }
        }
    }

    fun detectDatabaseOnlyTracks(forceRefresh: Boolean = false) {
        if (_databaseCleanupState.value is DatabaseCleanupScanState.Scanning) return

        _databaseCleanupState.value =
            DatabaseCleanupScanState.Scanning(
                "Reading exact indexed source spans and the current Poweramp library",
            )
        Log.i(TAG, "Starting clean-db scan (forceRefresh=$forceRefresh)")
        viewModelScope.launch(Dispatchers.IO) {
            var db: EmbeddingDatabase? = null
            try {
                exclusionsLoaded.await()
                val app = getApplication<Application>()
                val resolvedLibrary = V2LibraryDatabaseResolver.resolveOrNull(app.filesDir) {
                        progress ->
                    _databaseCleanupState.value = DatabaseCleanupScanState.Scanning(
                        exactHashProgressText(
                            subject = "active music-index file ${progress.filename}",
                            completedBytes = progress.completedBytes,
                            totalBytes = progress.totalBytes,
                        ),
                    )
                }
                val dbFile = resolvedLibrary?.databaseFile
                _hasDatabase.value = dbFile?.isFile == true
                val databaseGeneration = resolvedLibrary?.activeGeneration?.manifest?.generationId
                    .orEmpty()

                if (dbFile?.isFile != true) {
                    invalidateCache()
                    _databaseCleanupState.value = DatabaseCleanupScanState.Failed(
                        "No music index is loaded.",
                    )
                    Log.i(TAG, "Clean-db scan skipped: no embeddings.db")
                    return@launch
                }
                if (!PowerampHelper.canAccessData(app)) {
                    throw IllegalStateException(
                        "Poweramp library access is not granted. Grant access, then compare again."
                    )
                }

                _databaseCleanupState.value = DatabaseCleanupScanState.Scanning(
                    "Opening the complete Poweramp library",
                )
                val providerSnapshot = V2PowerampProviderSnapshotAcquirer(app).acquireBlocking {
                        completedRows,
                        totalRows,
                    ->
                    _databaseCleanupState.value = DatabaseCleanupScanState.Scanning(
                        powerampLibraryReadProgressText(completedRows, totalRows),
                    )
                }
                val providerGeneration = requireNotNull(providerSnapshot.libraryGeneration) {
                    "Poweramp snapshot has no complete library generation"
                }

                val cached = cachedDatabaseOnlyDetection
                if (!forceRefresh && databaseGeneration.isNotEmpty() && cached != null &&
                    databaseGeneration == cached.databaseGeneration &&
                    providerGeneration == cached.providerGeneration
                ) {
                    val tracks = cached.tracks
                    _databaseCleanupState.value = DatabaseCleanupScanState.Ready(
                        tracks = tracks,
                        message = "Found ${tracks.size} indexed tracks no longer in Poweramp",
                    )
                    Log.i(TAG, "Reusing exact-generation clean-db cache")
                    return@launch
                }

                db = EmbeddingDatabase.open(dbFile)
                val detector = NewTrackDetector(db)
                val tracks = detector.findDatabaseOnlyTracks(providerSnapshot) { status ->
                    _databaseCleanupState.value = DatabaseCleanupScanState.Scanning(status)
                }.sortedWith(
                    compareBy<EmbeddedTrack>({ it.artist.orEmpty().lowercase() },
                        { it.album.orEmpty().lowercase() },
                        { it.title.orEmpty().lowercase() })
                )

                cachedDatabaseOnlyDetection = CachedDatabaseOnlyDetection(
                    tracks = tracks,
                    databaseGeneration = databaseGeneration,
                    providerGeneration = providerGeneration,
                )
                updateDismissedFingerprint()
                _databaseCleanupState.value = DatabaseCleanupScanState.Ready(
                    tracks = tracks,
                    message = "Found ${tracks.size} indexed tracks no longer in Poweramp",
                )
                Log.i(TAG, "Clean-db scan complete: ${tracks.size} candidates")
            } catch (e: Exception) {
                Log.e(TAG, "Clean-db scan failed", e)
                _databaseCleanupState.value = DatabaseCleanupScanState.Failed(
                    indexingListenerFailureText(IndexingListenerFailureOperation.CLEANUP_SCAN),
                )
            } finally {
                db?.close()
            }
        }
    }

    fun deleteDatabaseOnlyTracks(ids: Set<Long>) {
        if (ids.isEmpty()) return
        if (_databaseCleanupState.value is DatabaseCleanupScanState.Scanning) return
        _databaseCleanupState.value = DatabaseCleanupScanState.Scanning(
            "Reading Poweramp again to confirm the selected indexed tracks are still absent",
        )
        viewModelScope.launch(Dispatchers.IO) {
            val app = getApplication<Application>()
            var refreshAfterPublication = false
            try {
                val resolution = V2LibraryDatabaseResolver.resolveOrNull(app.filesDir)
                    ?: error("No music index is loaded")
                val active = resolution.activeGeneration
                val database = EmbeddingDatabase.open(active.databaseFile)
                val eligibleIds = try {
                    NewTrackDetector(database).findDatabaseOnlyTracks(app) { status ->
                        _databaseCleanupState.value = DatabaseCleanupScanState.Scanning(status)
                    }.mapTo(mutableSetOf()) { it.id }
                } finally {
                    database.close()
                }
                require(ids.all(eligibleIds::contains)) {
                    "The Poweramp library changed; some selected tracks are no longer missing"
                }
                _databaseCleanupState.value = DatabaseCleanupScanState.Scanning(
                    "Updating the music index; the current version remains available...",
                )
                val result = V2LibraryMaintenancePublisher(app).removeTracksBlocking(ids) {
                    message ->
                    _databaseCleanupState.value = DatabaseCleanupScanState.Scanning(message)
                }
                if (result.noOp) {
                    refreshAfterPublication = true
                } else {
                    invalidateCache()
                    updateDismissedFingerprint()
                    refreshAfterPublication = true
                }
            } catch (error: Throwable) {
                Log.e(TAG, "Immutable clean-up publication failed", error)
                _databaseCleanupState.value = DatabaseCleanupScanState.Failed(
                    indexingListenerFailureText(IndexingListenerFailureOperation.CLEANUP_UPDATE),
                )
            } finally {
                refreshAppFiles()
            }
            if (refreshAfterPublication) {
                _databaseCleanupState.value = DatabaseCleanupScanState.Idle
                detectDatabaseOnlyTracks(forceRefresh = true)
            }
        }
    }

    fun beginExportSelection(): Boolean {
        if (_planningState.value !is PlanningState.Idle) return false
        if (!uiOperationAdmission.tryAcquire(IndexingUiOperation.EXPORT)) return false
        _exportState.value = ExportState.ChoosingDestination
        return true
    }

    fun cancelExportSelection() {
        if (_exportState.value !is ExportState.ChoosingDestination) return
        if (uiOperationAdmission.release(IndexingUiOperation.EXPORT)) {
            _exportState.value = ExportState.Idle
        }
    }

    fun exportInstance(uri: Uri) {
        if (_exportState.value !is ExportState.ChoosingDestination) return
        if (!uiOperationAdmission.isOwnedBy(IndexingUiOperation.EXPORT)) return
        _exportState.value = ExportState.Exporting("Preparing export...")
        viewModelScope.launch(Dispatchers.IO) {
            val app = getApplication<Application>()
            val filesDir = app.filesDir
            val terminalState = try {
                val resolution = V2LibraryDatabaseResolver.resolveOrNull(filesDir)
                    ?: error("No embeddings database found to export")
                val filesToZip = mutableListOf<Pair<String, File>>()
                filesToZip += "embeddings.db" to resolution.databaseFile
                val active = resolution.activeGeneration
                filesToZip += "clamp3.emb" to active.embeddingFile
                active.graphFile?.let { filesToZip += "graph.bin" to it }
                filesToZip += "manifest.json" to File(active.directory, "manifest.json")

                listOf(
                    "mert.tflite",
                    "clamp3_audio.tflite",
                    "clamp3_text.tflite",
                    "sentencepiece.bpe.model",
                ).forEach { name ->
                    val file = File(filesDir, name)
                    if (file.isFile) filesToZip += name to file
                }

                app.contentResolver.openOutputStream(uri)?.use { output ->
                    ZipOutputStream(BufferedOutputStream(output)).use { zip ->
                        for ((index, entry) in filesToZip.withIndex()) {
                            _exportState.value = ExportState.Exporting(
                                "Preparing library bundle \u00b7 file ${index + 1} of ${filesToZip.size}"
                            )
                            zip.putNextEntry(ZipEntry(entry.first))
                            BufferedInputStream(FileInputStream(entry.second)).use { input ->
                                input.copyTo(zip)
                            }
                            zip.closeEntry()
                        }
                    }
                } ?: throw IllegalArgumentException("Cannot open export destination")
                ExportState.Complete(File(uri.lastPathSegment ?: "export.zip").name)
            } catch (e: Throwable) {
                Log.e(TAG, "App-file export failed", e)
                ExportState.Error(
                    indexingListenerFailureText(IndexingListenerFailureOperation.EXPORT),
                )
            }
            check(uiOperationAdmission.release(IndexingUiOperation.EXPORT)) {
                "export completed without owning export admission"
            }
            _exportState.value = terminalState
        }
    }

    fun clearExportState() {
        if (_exportState.value is ExportState.Complete || _exportState.value is ExportState.Error) {
            _exportState.value = ExportState.Idle
        }
    }

    /** Auto-select only ready exact misses. Imported conflicts require safe supersession first. */
    private fun autoSelect(tracks: List<NewTrackDetector.UnindexedTrack>) {
        reconcileExclusions(tracks)
        if (savedIndexingStateError() != null) {
            _selectedIds.value = emptySet()
            return
        }
        val dismissed = _dismissedIds.value
        val ignored = _ignoredIds.value
        val allExcluded = dismissed + ignored +
            unresolvedFailureTrackIds(tracks) + preflightAttentionIds()
        _selectedIds.value = V2IndexingSelectionPolicy.readyTrackIds(tracks, allExcluded)
    }

    fun toggleSelection(id: Long) {
        if (savedIndexingStateError() != null) return
        val track = _unindexedTracks.value.firstOrNull { it.powerampFileId == id } ?: return
        if (!V2IndexingSelectionPolicy.isReadyTrack(track) ||
            id in _dismissedIds.value || id in _ignoredIds.value ||
            id in unresolvedFailureTrackIds(listOf(track)) || id in preflightAttentionIds()
        ) return
        _selectedIds.value = if (id in _selectedIds.value) {
            _selectedIds.value - id
        } else {
            _selectedIds.value + id
        }
    }

    fun selectAll() {
        if (savedIndexingStateError() != null) return
        val dismissed = _dismissedIds.value
        val ignored = _ignoredIds.value
        _selectedIds.value = V2IndexingSelectionPolicy.readyTrackIds(
            _unindexedTracks.value,
            dismissed + ignored + unresolvedFailureTrackIds() + preflightAttentionIds(),
        )
    }

    fun deselectAll() {
        _selectedIds.value = emptySet()
    }

    fun selectIds(ids: Set<Long>) {
        if (savedIndexingStateError() != null) return
        val unresolvedFailureIds = unresolvedFailureTrackIds()
        val selectable = _unindexedTracks.value.asSequence()
            .filter {
                it.powerampFileId in ids &&
                    V2IndexingSelectionPolicy.isReadyTrack(it) &&
                    it.powerampFileId !in _dismissedIds.value &&
                    it.powerampFileId !in _ignoredIds.value &&
                    it.powerampFileId !in unresolvedFailureIds &&
                    it.powerampFileId !in preflightAttentionIds()
            }
            .map { it.powerampFileId }
            .toSet()
        _selectedIds.value = V2IndexingSelectionPolicy.selectVisible(_selectedIds.value, selectable)
    }

    fun deselectIds(ids: Set<Long>) {
        _selectedIds.value = V2IndexingSelectionPolicy.deselectVisible(_selectedIds.value, ids)
    }

    fun neverIndexTracks(trackIds: Set<Long>) {
        if (_trackExclusionError.value != null) return
        val toDismiss = trackIds.toSet()
        if (toDismiss.isEmpty()) return
        val candidates = candidatesForIds(toDismiss)
        if (candidates.isEmpty()) {
            _planningState.value = PlanningState.Failed(
                "No selected track could be matched reliably enough to save a Never-index choice.",
            )
            return
        }
        dismissedExclusions = V2TrackExclusionPolicy.add(
            dismissedExclusions,
            candidates,
        )
        persistExclusions()
        reconcileExclusions(_unindexedTracks.value)
        _selectedIds.value = _selectedIds.value - candidates.mapTo(mutableSetOf()) {
            it.powerampFileId
        }
        if (candidates.size != toDismiss.size) {
            _planningState.value = PlanningState.Failed(
                "Some selected tracks could not be matched reliably and were not added to Never index.",
            )
        }
    }

    fun clearDismissed() {
        if (_trackExclusionError.value != null) return
        dismissedExclusions = emptyList()
        persistExclusions()
        reconcileExclusions(_unindexedTracks.value)
        // Re-select ready exact misses only; imported conflicts remain blocked.
        val ignored = _ignoredIds.value
        _selectedIds.value = V2IndexingSelectionPolicy.readyTrackIds(
            _unindexedTracks.value,
            ignored + unresolvedFailureTrackIds() + preflightAttentionIds(),
        )
    }

    /** Get dismissed track details from the full unindexedTracks list. */
    fun getDismissedTracks(): List<NewTrackDetector.UnindexedTrack> {
        return _unindexedTracks.value.filter { it.powerampFileId in _dismissedIds.value }
    }

    /** Restore specific tracks from dismissed back to visible (and auto-select them). */
    fun restoreFromDismissed(ids: Set<Long>) {
        if (_trackExclusionError.value != null) return
        dismissedExclusions = V2TrackExclusionPolicy.remove(
            dismissedExclusions,
            candidatesForIds(ids),
        )
        persistExclusions()
        reconcileExclusions(_unindexedTracks.value)
        selectIds(ids)
    }

    /** Get ignored track details from the full unindexedTracks list. */
    fun getIgnoredTracks(): List<NewTrackDetector.UnindexedTrack> {
        return _unindexedTracks.value.filter { it.powerampFileId in _ignoredIds.value }
    }

    /** Restore specific tracks from ignored back to visible (and auto-select them). */
    fun restoreFromIgnored(ids: Set<Long>) {
        if (_trackExclusionError.value != null) return
        ignoredExclusions = V2TrackExclusionPolicy.remove(
            ignoredExclusions,
            candidatesForIds(ids),
        )
        persistExclusions()
        reconcileExclusions(_unindexedTracks.value)
        selectIds(ids)
    }

    /** Move tracks from ignored to never-index (permanent exclusion). */
    fun moveIgnoredToNeverIndex(ids: Set<Long>) {
        if (_trackExclusionError.value != null) return
        val candidates = candidatesForIds(ids)
        ignoredExclusions = V2TrackExclusionPolicy.remove(ignoredExclusions, candidates)
        dismissedExclusions = V2TrackExclusionPolicy.add(dismissedExclusions, candidates)
        persistExclusions()
        reconcileExclusions(_unindexedTracks.value)
    }

    /** Clear all ignored tracks and re-run autoSelect. */
    fun clearIgnored() {
        if (_trackExclusionError.value != null) return
        ignoredExclusions = emptyList()
        _ignoredIds.value = emptySet()
        persistExclusions()
        autoSelect(_unindexedTracks.value)
    }

    /** Plans every explicitly selected row. Search and other visibility filters are irrelevant. */
    fun startIndexing(buildGraph: Boolean = false) {
        if (_planningState.value !is PlanningState.Idle) return
        savedIndexingStateError()?.let { error ->
            _planningState.value = PlanningState.Failed(error)
            return
        }
        val app = getApplication<Application>()
        val selected = V2IndexingSelectionPolicy.selectedForJob(_selectedIds.value)
        if (selected.isEmpty()) return
        val dismissed = _dismissedIds.value
        val ignored = _ignoredIds.value
        val unresolvedFailureIds = unresolvedFailureTrackIds()
        val tracks = _unindexedTracks.value.filter {
            V2IndexingSelectionPolicy.isReadyTrack(it) &&
                it.powerampFileId in selected &&
                it.powerampFileId !in dismissed &&
                it.powerampFileId !in ignored &&
                it.powerampFileId !in unresolvedFailureIds &&
                it.powerampFileId !in preflightAttentionIds()
        }
        if (tracks.isEmpty()) return
        if (!uiOperationAdmission.tryAcquire(IndexingUiOperation.JOB_PLANNING)) return

        // Close the rapid-double-tap window before validation and durable preflight begin.
        _planningState.value = PlanningState.Planning(
            "Confirming audio access and required indexing files",
        )
        viewModelScope.launch(Dispatchers.IO) {
            var requestIsDurable = false
            val terminalState: PlanningState = try {
                val hasAudio = hasAudioAccess(app)
                _hasAudioAccess.value = hasAudio
                if (!hasAudio) {
                    throw IllegalStateException(
                        "Music and audio access is required before an indexing job can be created.",
                    )
                }
                if (!PowerampHelper.canAccessData(app)) {
                    throw IllegalStateException(
                        "Poweramp library access is required before an indexing job can be created.",
                    )
                }
                val hasDatabase = V2LibraryDatabaseResolver.hasPublishedPointer(app.filesDir)
                _hasDatabase.value = hasDatabase
                if (!hasDatabase) {
                    throw IllegalStateException(
                        "A music index is required before an indexing job can be created.",
                    )
                }
                val hasModels = V2CurrentModelPolicyResolver.hasRequiredArtifacts(app.filesDir)
                _hasModels.value = hasModels
                if (!hasModels) {
                    throw IllegalStateException(
                        "The audio and text models required for indexing are not ready.",
                    )
                }

                _planningState.value = PlanningState.Planning(
                    "Saving a resumable ${tracks.size}-track indexing request",
                )
                val intent = V2IndexingPreflightIntentFactory.create(
                    jobId = UUID.randomUUID().toString(),
                    selected = V2IndexingPreflightSelectionFactory.fromTracks(tracks),
                    rebuildDerivedIndexes = buildGraph,
                    executionProfile = V2IndexingExecutionProfile.FULL,
                    nowEpochMs = System.currentTimeMillis(),
                )
                try {
                    IndexingService.submitPreflight(app, intent)
                } catch (error: Throwable) {
                    // Foreground launch can fail after the request and pointer are durable.
                    // Consume the choice whenever this exact immutable request now exists.
                    requestIsDurable = runCatching {
                        preflightRejectionHistorySource.inspect().intents.any {
                            it.jobId == intent.jobId
                        }
                    }.getOrDefault(false)
                    if (requestIsDurable) consumePreflightRetryChoices(tracks)
                    throw error
                }
                consumePreflightRetryChoices(tracks)
                PlanningState.Idle
            } catch (error: Throwable) {
                Log.e(TAG, "Unable to start V2 indexing request", error)
                PlanningState.Failed(
                    indexingListenerFailureText(
                        IndexingListenerFailureOperation.INDEXING_REQUEST,
                        indexingRequestIsDurable = requestIsDurable,
                    ),
                )
            }
            check(uiOperationAdmission.release(IndexingUiOperation.JOB_PLANNING)) {
                "indexing planning completed without owning planning admission"
            }
            _planningState.value = terminalState
        }
    }

    fun attachToJob(jobId: String?) {
        viewModelScope.launch(Dispatchers.IO) {
            IndexingService.attach(getApplication(), jobId)
        }
    }

    fun clearPlanningError() {
        if (_planningState.value is PlanningState.Failed) {
            _planningState.value = PlanningState.Idle
        }
    }

    fun pauseIndexing(jobId: String) = IndexingService.pause(getApplication(), jobId)

    fun resumeIndexing(jobId: String) = IndexingService.resume(getApplication(), jobId)

    fun cancelIndexing(jobId: String) = IndexingService.cancel(getApplication(), jobId)

    fun retryFailures(jobId: String) {
        val snapshot = indexingState.value as? IndexingService.IndexingState.JobSnapshot ?: return
        if (snapshot.jobId != jobId || !hasUserRetryEligibleFailure(
                snapshot.jobState,
                _failedTracks.value.asSequence()
                    .filter { it.jobId == jobId }
                    .map { it.retryTrigger }
                    .toList(),
            )
        ) return
        IndexingService.retry(getApplication(), jobId, RetryTrigger.USER_REQUEST)
    }

    fun retryFailure(failure: FailedTrackUi) {
        if (!canUserRetryIndexingFailure(failure.jobState, failure.retryTrigger)) return
        IndexingService.retryTrack(
            getApplication(),
            failure.jobId,
            failure.workId,
            RetryTrigger.USER_REQUEST,
        )
    }

    fun skipFailure(failure: FailedTrackUi) {
        if (!canOfferIndexingFailureActions(failure.jobState)) return
        IndexingService.skip(getApplication(), failure.jobId, failure.workId)
    }

    fun neverIndexFailure(failure: FailedTrackUi) {
        if (_trackExclusionError.value != null) return
        val canMutateCurrentJob = canOfferIndexingFailureActions(failure.jobState)
        if (!canNeverIndexFailure(failure.jobState)) return
        val matchingCurrentIds = unresolvedFailureCurrentTrackIds(
            failures = listOf(failure),
            tracks = _unindexedTracks.value,
        )
        val candidate = exclusionCandidate(
            powerampFileId = failure.powerampFileId,
            physicalPath = failure.providerPhysicalPath,
            offsetMs = failure.offsetMs,
            durationMs = failure.durationMs,
            stableTrackSpanId = failure.stableTrackSpanId,
        )
        if (candidate != null) {
            dismissedExclusions = V2TrackExclusionPolicy.add(
                dismissedExclusions,
                listOf(candidate),
            )
            persistExclusions()
            reconcileExclusions(_unindexedTracks.value)
        } else {
            _planningState.value = PlanningState.Failed(
                "This track no longer matches a current Poweramp item, so Never index was not saved.",
            )
            return
        }
        _selectedIds.value = _selectedIds.value - matchingCurrentIds
        if (canMutateCurrentJob) {
            IndexingService.skip(getApplication(), failure.jobId, failure.workId)
        } else {
            publishUnresolvedFailures(retainedUnresolvedFailures)
        }
    }

    fun selectFailureForNewRun(failure: FailedTrackUi) {
        val current = _failedTracks.value.firstOrNull {
            it.jobId == failure.jobId && it.workId == failure.workId
        } ?: return
        if (!canSelectIndexingFailureForNewRun(
                state = current.jobState,
                retryTrigger = current.retryTrigger,
                failureCode = current.code,
            )
        ) return
        val candidate = exclusionCandidate(
            powerampFileId = current.powerampFileId,
            physicalPath = current.providerPhysicalPath,
            offsetMs = current.offsetMs,
            durationMs = current.durationMs,
            stableTrackSpanId = current.stableTrackSpanId,
        ) ?: run {
            _planningState.value = PlanningState.Failed(
                "This failed track is no longer in the current Poweramp library.",
            )
            return
        }
        preflightRetryChoiceSpans = preflightRetryChoiceSpans + candidate.providerSpan
        preflightRetryChoiceRepository.persist(preflightRetryChoiceSpans)
        publishUnresolvedFailures(retainedUnresolvedFailures)
        selectIds(
            unresolvedFailureCurrentTrackIds(
                failures = listOf(current),
                tracks = _unindexedTracks.value,
            ),
        )
    }

    fun tryAgainPreflightRejection(attention: V2PreflightAttentionTrack) {
        val current = currentPreflightAttention(attention) ?: return
        if (!current.canTryAgain) return
        val span = current.rejection.providerSpan
        preflightRetryChoiceSpans = preflightRetryChoiceSpans + span
        preflightRetryChoiceRepository.persist(preflightRetryChoiceSpans)
        publishUnresolvedFailures(retainedUnresolvedFailures)
        selectIds(setOf(current.currentTrack.powerampFileId))
    }

    fun neverIndexPreflightRejection(attention: V2PreflightAttentionTrack) {
        if (_trackExclusionError.value != null) return
        val current = currentPreflightAttention(attention) ?: return
        val candidate = V2TrackExclusionCandidate(
            powerampFileId = current.currentTrack.powerampFileId,
            providerSpan = current.rejection.providerSpan,
        )
        dismissedExclusions = V2TrackExclusionPolicy.add(
            dismissedExclusions,
            listOf(candidate),
        )
        preflightRetryChoiceSpans = preflightRetryChoiceSpans - current.rejection.providerSpan
        preflightRetryChoiceRepository.persist(preflightRetryChoiceSpans)
        persistExclusions()
        reconcileExclusions(_unindexedTracks.value)
        _selectedIds.value = _selectedIds.value - current.currentTrack.powerampFileId
    }

    fun neverIndexAttentionTracks(
        failures: List<FailedTrackUi>,
        preflightAttention: List<V2PreflightAttentionTrack>,
        sourceAttention: List<NewTrackDetector.UnindexedTrack>,
    ) {
        val requestedCount = failures.size + preflightAttention.size + sourceAttention.size
        if (requestedCount == 0) return
        if (_trackExclusionError.value != null) {
            _planningState.value = PlanningState.Failed(
                "Never index is unavailable because saved choices could not be verified. " +
                    "No tracks were changed.",
            )
            return
        }
        if (indexingState.value !is IndexingService.IndexingState.Idle) {
            _planningState.value = PlanningState.Failed(
                "Indexing state changed before confirmation. No tracks were changed.",
            )
            return
        }

        val currentFailures = failures.mapNotNull { requested ->
            _failedTracks.value.firstOrNull { current ->
                current.jobId == requested.jobId &&
                    current.workId == requested.workId &&
                    current.stableTrackSpanId == requested.stableTrackSpanId
            }
        }
        val failureCandidates = currentFailures.mapNotNull { failure ->
            exclusionCandidate(
                powerampFileId = failure.powerampFileId,
                physicalPath = failure.providerPhysicalPath,
                offsetMs = failure.offsetMs,
                durationMs = failure.durationMs,
                stableTrackSpanId = failure.stableTrackSpanId,
            )?.let { candidate -> failure to candidate }
        }
        val matchedPreflightAttention = preflightAttention.mapNotNull(::currentPreflightAttention)
        val preflightCandidates = matchedPreflightAttention.mapNotNull { attention ->
            val span = attention.rejection.providerSpan
            exclusionCandidate(
                powerampFileId = attention.currentTrack.powerampFileId,
                physicalPath = span.normalizedPhysicalPath,
                offsetMs = span.offsetMs,
                durationMs = span.durationMs,
                stableTrackSpanId = null,
            )?.let { candidate -> attention to candidate }
        }
        val matchedSourceAttention = sourceAttention.mapNotNull { requested ->
            V2IndexingSelectionPolicy.currentNonReadySourceAttentionMatch(
                requested = requested,
                currentTracks = _unindexedTracks.value,
            )
        }
        val sourceAttentionCandidates = matchedSourceAttention.mapNotNull { track ->
            exclusionCandidate(track)?.let { candidate -> track to candidate }
        }
        val successfulCount =
            failureCandidates.size + preflightCandidates.size + sourceAttentionCandidates.size
        if (successfulCount == 0) {
            val trackNoun = if (requestedCount == 1) "track" else "tracks"
            _planningState.value = PlanningState.Failed(
                "None of the $requestedCount attention $trackNoun could be matched reliably. " +
                    "No Never-index choices were changed.",
            )
            return
        }

        dismissedExclusions = V2TrackExclusionPolicy.add(
            dismissedExclusions,
            failureCandidates.map { it.second } +
                preflightCandidates.map { it.second } +
                sourceAttentionCandidates.map { it.second },
        )
        val acceptedPreflightSpans = preflightCandidates.mapTo(linkedSetOf()) {
            it.first.rejection.providerSpan
        }
        if (acceptedPreflightSpans.isNotEmpty()) {
            preflightRetryChoiceSpans = preflightRetryChoiceSpans - acceptedPreflightSpans
            preflightRetryChoiceRepository.persist(preflightRetryChoiceSpans)
        }
        persistExclusions()
        reconcileExclusions(_unindexedTracks.value)

        val acceptedFailureTracks = failureCandidates.map { it.first }
        val attentionIds = unresolvedFailureCurrentTrackIds(
            failures = acceptedFailureTracks,
            tracks = _unindexedTracks.value,
        ) + preflightCandidates.mapTo(linkedSetOf()) {
            it.first.currentTrack.powerampFileId
        } + sourceAttentionCandidates.mapTo(linkedSetOf()) {
            it.first.powerampFileId
        }
        _selectedIds.value = _selectedIds.value - attentionIds
        acceptedFailureTracks.filter {
            canOfferIndexingFailureActions(it.jobState)
        }.forEach { failure ->
            IndexingService.skip(getApplication(), failure.jobId, failure.workId)
        }

        val unmatchedCount = requestedCount - successfulCount
        if (unmatchedCount > 0) {
            val trackNoun = if (unmatchedCount == 1) "track was" else "tracks were"
            _planningState.value = PlanningState.Failed(
                "Added $successfulCount of $requestedCount attention tracks to Never index. " +
                    "$unmatchedCount $trackNoun not matched reliably and remained unchanged.",
            )
        }
    }

    fun changeExecutionProfile(profile: V2IndexingExecutionProfile, jobId: String? = null) {
        _executionProfile.value = profile
        jobId?.let { IndexingService.changeProfile(getApplication(), it, profile) }
    }

    fun requestPowerampAccess() {
        PowerampHelper.requestDataPermission(getApplication())
    }

    fun refreshAppFiles() {
        appFileRefreshRequests.trySend(Unit)
    }

    fun refreshDurablePreflightHistory() {
        viewModelScope.launch(Dispatchers.IO) {
            try {
                val loaded = exclusionRepository.loadPersisted()
                dismissedExclusions = loaded.first
                ignoredExclusions = emptyList()
                preflightRetryChoiceSpans = preflightRetryChoiceRepository.load()
                _trackExclusionError.value = null
            } catch (error: Throwable) {
                Log.e(TAG, "Unable to reload saved track exclusions", error)
                _trackExclusionError.value = trackExclusionUnavailableMessage()
                _selectedIds.value = emptySet()
            }
            refreshPreflightIntentHistory()
            if (savedIndexingStateError() == null &&
                IndexingService.state.value is IndexingService.IndexingState.Idle
            ) {
                detectUnindexed(forceRefresh = true)
            }
        }
    }

    private fun refreshAppFilesBlocking() {
        val filesDir = getApplication<Application>().filesDir
        _hasAudioAccess.value = hasAudioAccess(getApplication())
        _hasDatabase.value = V2LibraryDatabaseResolver.hasPublishedPointer(filesDir)
        _hasModels.value = V2CurrentModelPolicyResolver.hasRequiredArtifacts(filesDir)
    }

    private fun hasAudioAccess(context: Context): Boolean {
        val permission = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            Manifest.permission.READ_MEDIA_AUDIO
        } else {
            Manifest.permission.READ_EXTERNAL_STORAGE
        }
        return ContextCompat.checkSelfPermission(context, permission) == PackageManager.PERMISSION_GRANTED
    }

    private fun latestUnresolvedFailures(
        ledgers: List<com.powerampstartradio.indexing.v2.IndexingJobLedger>,
    ): List<FailedTrackUi> {
        val outcomes = ledgers.asSequence()
            .flatMap { ledger ->
                val descriptorById = ledger.jobSpec.tracks.associateBy { it.workId }
                ledger.tracks.asSequence().mapNotNull { track ->
                    val descriptor = descriptorById[track.workId] ?: return@mapNotNull null
                    val occurrenceIdentity = V2UnresolvedFailureIdentityPolicy.identityOrNull(
                        stableTrackSpanId =
                            descriptor.stableTrackSpanIdentity.stableTrackSpanId,
                        powerampFileId = descriptor.powerampFileId,
                        providerPhysicalPath = descriptor.providerRow.providerPhysicalPath,
                        offsetMs = descriptor.providerOffsetMs,
                        durationMs = descriptor.providerDurationMs,
                    ) ?: return@mapNotNull null
                    val failure = track.activeFailureId?.let { activeId ->
                        track.failures.firstOrNull { it.failureId == activeId }
                    } ?: track.failures.maxByOrNull { it.lastOccurredAtEpochMs }
                    val isVisible = isVisibleUnresolvedIndexingFailure(
                        trackState = track.state,
                        hasFailureEvidence = failure != null,
                    )
                    V2UnresolvedFailureOccurrenceOutcome(
                        identity = occurrenceIdentity,
                        contentSupersessionIdentity =
                            V2UnresolvedFailureIdentityPolicy
                                .contentSupersessionIdentityOrNull(
                                    occurrenceIdentity = occurrenceIdentity,
                                    stableTrackSpanIdentity =
                                        descriptor.stableTrackSpanIdentity,
                                ),
                        jobCreatedAtEpochMs = ledger.jobSpec.createdAtEpochMs,
                        ledgerRevision = ledger.revision,
                        unresolvedValue = if (isVisible && failure != null) FailedTrackUi(
                            jobId = ledger.jobSpec.jobId,
                            workId = track.workId,
                            powerampFileId = descriptor.powerampFileId,
                            title = descriptor.displayMetadata.title,
                            artist = descriptor.displayMetadata.artist,
                            code = failure.code,
                            diagnostic = failure.diagnostic,
                            occurrences = failure.occurrences,
                            retryTrigger = failure.retryTrigger,
                            disposition = failure.disposition,
                            stableTrackSpanId =
                                descriptor.stableTrackSpanIdentity.stableTrackSpanId,
                            providerPhysicalPath = descriptor.providerRow.providerPhysicalPath,
                            offsetMs = descriptor.providerOffsetMs,
                            durationMs = descriptor.providerDurationMs,
                            jobState = ledger.state,
                        ) else null,
                    )
                }
            }
            .toList()

        return V2UnresolvedFailureIdentityPolicy.latestUnresolvedValues(outcomes)
            .sortedWith(compareBy(FailedTrackUi::artist, FailedTrackUi::title))
    }

    private fun publishUnresolvedFailures(failures: List<FailedTrackUi>) {
        retainedUnresolvedFailures = failures
        val suppressedSpans = preflightRetryChoiceSpans +
            dismissedExclusions.mapTo(linkedSetOf(), V2PersistedTrackExclusion::providerSpan) +
            ignoredExclusions.mapTo(linkedSetOf(), V2PersistedTrackExclusion::providerSpan)
        val visibleFailures = failures.filterNot { failure ->
            exclusionCandidate(
                powerampFileId = failure.powerampFileId,
                physicalPath = failure.providerPhysicalPath,
                offsetMs = failure.offsetMs,
                durationMs = failure.durationMs,
                stableTrackSpanId = failure.stableTrackSpanId,
            )?.providerSpan in suppressedSpans
        }
        if (_failedTracks.value != visibleFailures) {
            _failedTracks.value = visibleFailures
            _selectedIds.value = _selectedIds.value - unresolvedFailureTrackIds()
        }
        publishPreflightAttention()
    }

    private fun unresolvedFailureTrackIds(
        tracks: Collection<NewTrackDetector.UnindexedTrack> = _unindexedTracks.value,
    ): Set<Long> = unresolvedFailureCurrentTrackIds(_failedTracks.value, tracks)

    private fun preflightAttentionIds(): Set<Long> = _preflightAttentionTracks.value
        .mapTo(hashSetOf()) { it.currentTrack.powerampFileId }

    private fun refreshPreflightIntentHistory() {
        val inspection = runCatching {
            preflightRejectionHistorySource.inspect()
        }.onFailure { error ->
            Log.e(TAG, "Unable to read durable preflight rejection history", error)
        }.getOrNull()
        if (inspection == null) {
            _preflightHistoryError.value = preflightHistoryUnavailableMessage(null)
        } else {
            retainedPreflightRejections = V2PreflightRejectionPolicy.retainedRejections(
                inspection.intents,
            )
            _preflightHistoryError.value = if (inspection.issues.isEmpty()) {
                null
            } else {
                preflightHistoryUnavailableMessage(inspection.issues.size)
            }
        }
        if (_preflightHistoryError.value != null) {
            _selectedIds.value = emptySet()
        }
        publishUnresolvedFailures(retainedUnresolvedFailures)
    }

    private fun preflightHistoryUnavailableMessage(issueCount: Int?): String = buildString {
        append("Saved indexing history could not be read")
        if (issueCount != null) {
            append(" (")
            append(issueCount)
            append(if (issueCount == 1) " unreadable record" else " unreadable records")
            append(')')
        }
        append(". Indexing is paused to avoid retrying rejected tracks. ")
        append("Check again; no saved history was deleted.")
    }

    private fun trackExclusionUnavailableMessage(): String =
        "Saved Never-index choices could not be read. Indexing is paused to avoid selecting " +
            "excluded tracks. Check again; no saved choices were deleted."

    private fun savedIndexingStateError(): String? =
        _trackExclusionError.value ?: _preflightHistoryError.value

    private fun publishPreflightAttention() {
        val suppressedSpans = preflightRetryChoiceSpans +
            dismissedExclusions.mapTo(linkedSetOf(), V2PersistedTrackExclusion::providerSpan) +
            ignoredExclusions.mapTo(linkedSetOf(), V2PersistedTrackExclusion::providerSpan)
        _preflightAttentionTracks.value = V2PreflightRejectionPolicy.joinCurrentUnindexed(
            retained = retainedPreflightRejections,
            currentUnindexed = _unindexedTracks.value,
            suppressedSpans = suppressedSpans,
        ).filterNot {
            it.currentTrack.powerampFileId in unresolvedFailureTrackIds(
                listOf(it.currentTrack),
            )
        }
        _selectedIds.value = _selectedIds.value - preflightAttentionIds()
    }

    private fun currentPreflightAttention(
        requested: V2PreflightAttentionTrack,
    ): V2PreflightAttentionTrack? = _preflightAttentionTracks.value.firstOrNull { current ->
        current.rejection.jobId == requested.rejection.jobId &&
            current.rejection.providerSpan == requested.rejection.providerSpan &&
            current.currentTrack.powerampFileId == requested.currentTrack.powerampFileId
    }

    private fun consumePreflightRetryChoices(
        submittedTracks: Collection<NewTrackDetector.UnindexedTrack>,
    ) {
        val submittedSpans = submittedTracks.mapNotNullTo(linkedSetOf()) { track ->
            V2TrackExclusionRepository.candidate(track)?.providerSpan
        }
        val remaining = preflightRetryChoiceSpans - submittedSpans
        if (remaining != preflightRetryChoiceSpans) {
            preflightRetryChoiceSpans = remaining
            preflightRetryChoiceRepository.persist(remaining)
            refreshPreflightIntentHistory()
        }
    }

    private fun hydrateActivePreflightAttentionMetadataIfNeeded() {
        if (_unindexedTracks.value.isNotEmpty() ||
            !preflightMetadataHydrationActive.compareAndSet(false, true)
        ) return
        val activeJobId = when (val current = IndexingService.state.value) {
            is IndexingService.IndexingState.PreflightSnapshot -> current.jobId
            is IndexingService.IndexingState.JobSnapshot -> current.jobId
            else -> {
                preflightMetadataHydrationActive.set(false)
                return
            }
        }
        val retained = retainedPreflightRejections.filter { it.jobId == activeJobId }
        if (retained.isEmpty()) {
            preflightMetadataHydrationActive.set(false)
            return
        }

        try {
            val app = getApplication<Application>()
            if (!PowerampHelper.canAccessData(app)) return
            val databaseFile = V2LibraryDatabaseResolver.resolveOrNull(app.filesDir)
                ?.databaseFile
                ?.takeIf(File::isFile)
                ?: return
            val snapshot = V2PowerampProviderSnapshotAcquirer(app).acquireBlocking()
            val tracks = V2PreflightRejectionPolicy.currentUnindexedFromCompleteSnapshot(
                retained = retained,
                snapshot = snapshot,
                receipts = V2ProviderSpanReceiptReader.read(databaseFile).receipts,
            )
            if (_unindexedTracks.value.isEmpty() && tracks.isNotEmpty()) {
                _unindexedTracks.value = tracks
                reconcileExclusions(tracks)
            }
        } catch (error: Throwable) {
            Log.w(TAG, "Unable to hydrate current metadata for preflight rejections", error)
        } finally {
            preflightMetadataHydrationActive.set(false)
        }
    }

    /** Quick count of tracks in the Poweramp library (no full cursor scan). */
    private fun getPowerampTrackCount(context: Context): Int {
        return try {
            val filesUri = PowerampHelper.ROOT_URI.buildUpon()
                .appendEncodedPath("files").build()
            context.contentResolver.query(
                filesUri, arrayOf("COUNT(*)"), null, null, null
            )?.use { cursor ->
                if (cursor.moveToFirst()) cursor.getInt(0) else -1
            } ?: -1
        } catch (_: Exception) { -1 }
    }

    private fun reconcileExclusions(tracks: List<NewTrackDetector.UnindexedTrack>) {
        val resolved = try {
            exclusionRepository.resolveAndMigrate(tracks)
        } catch (error: Throwable) {
            Log.e(TAG, "Unable to verify saved track exclusions", error)
            _trackExclusionError.value = trackExclusionUnavailableMessage()
            _selectedIds.value = emptySet()
            return
        }
        _trackExclusionError.value = null
        dismissedExclusions = resolved.never
        ignoredExclusions = emptyList()
        _dismissedIds.value = resolved.neverIds
        _ignoredIds.value = emptySet()
        publishUnresolvedFailures(retainedUnresolvedFailures)
    }

    private fun candidatesForIds(ids: Set<Long>): List<V2TrackExclusionCandidate> =
        _unindexedTracks.value.asSequence()
            .filter { it.powerampFileId in ids }
            .mapNotNull(V2TrackExclusionRepository::candidate)
            .toList()

    private fun exclusionCandidate(
        track: NewTrackDetector.UnindexedTrack,
    ): V2TrackExclusionCandidate? = V2TrackExclusionRepository.candidate(track)

    private fun exclusionCandidate(
        powerampFileId: Long,
        physicalPath: String?,
        offsetMs: Long,
        durationMs: Long,
        stableTrackSpanId: String?,
    ): V2TrackExclusionCandidate? = V2TrackExclusionRepository.candidate(
        powerampFileId = powerampFileId,
        physicalPath = physicalPath,
        offsetMs = offsetMs,
        durationMs = durationMs,
        stableTrackSpanId = stableTrackSpanId,
    )

    private fun persistExclusions() {
        exclusionRepository.persist(dismissedExclusions, ignoredExclusions)
    }

    private fun databaseFingerprint(dbFile: File?): String {
        return if (dbFile?.isFile == true) {
            "${dbFile.length()}_${dbFile.lastModified()}"
        } else {
            ""
        }
    }

    /** Refresh persisted locators after library/database maintenance without trusting numeric IDs. */
    private fun updateDismissedFingerprint() {
        reconcileExclusions(_unindexedTracks.value)
    }

}
