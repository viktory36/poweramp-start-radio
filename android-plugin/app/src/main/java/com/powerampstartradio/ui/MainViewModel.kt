package com.powerampstartradio.ui

import android.app.Application
import android.content.Context
import android.net.Uri
import android.os.Debug
import android.provider.OpenableColumns
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.powerampstartradio.AudioLibraryPermission
import com.powerampstartradio.data.EmbeddedTrack
import com.powerampstartradio.data.EmbeddingDatabase
import com.powerampstartradio.data.EmbeddingIndex
import com.powerampstartradio.data.StableIdentityGenerationBinding
import com.powerampstartradio.data.StableTrackIdentityCatalog
import com.powerampstartradio.data.StableTrackIdentityResolution
import com.powerampstartradio.data.TextEmbeddingIndexGeneration
import com.powerampstartradio.data.TextEmbeddingIndexSnapshot
import com.powerampstartradio.indexing.V2ActiveLibraryCatalog
import com.powerampstartradio.indexing.V2ActiveLibraryCatalogLoader
import com.powerampstartradio.indexing.V2ActiveLibraryCatalogLoadProgress
import com.powerampstartradio.indexing.V2ActiveLibraryCatalogStore
import com.powerampstartradio.indexing.V2ActiveLibraryQuarantineReason
import com.powerampstartradio.similarity.ActiveRecommendationDomain
import com.powerampstartradio.similarity.FindMusicTextQueuePlanEvidence
import com.powerampstartradio.similarity.FindMusicAllOfQueuePlanEvidence
import com.powerampstartradio.similarity.FindMusicAllOfQueuePlanner
import com.powerampstartradio.similarity.FindMusicTextQueuePlanner
import com.powerampstartradio.similarity.SimilarTrack
import com.powerampstartradio.similarity.StableSimilarityTopK
import com.powerampstartradio.similarity.StableVisibleResultReducer
import com.powerampstartradio.similarity.StableVisibleReduction
import com.powerampstartradio.similarity.algorithms.ComposedRankingRow
import com.powerampstartradio.indexing.IndexingService
import com.powerampstartradio.indexing.IndexingViewModel
import com.powerampstartradio.indexing.Clamp3TextInference
import com.powerampstartradio.indexing.OfficialSentencePieceTokenizer
import com.powerampstartradio.indexing.NewTrackDetector
import com.powerampstartradio.indexing.V2TrackExclusionRepository
import com.powerampstartradio.indexing.V2IndexingSelectionPolicy
import com.powerampstartradio.indexing.V2IndexingAttentionHistorySource
import com.powerampstartradio.indexing.V2IndexingReadinessPolicy
import com.powerampstartradio.indexing.V2ProcessLibraryInspectionCoordinator
import com.powerampstartradio.indexing.V2UnindexedCountCacheIdentity
import com.powerampstartradio.indexing.V2UnindexedCountCachePolicy
import com.powerampstartradio.indexing.exactHashProgressText
import com.powerampstartradio.indexing.powerampLibraryReadProgressText
import com.powerampstartradio.indexing.v2.TextRetrievalSpecFingerprint
import com.powerampstartradio.indexing.v2.IndexingJobState
import com.powerampstartradio.indexing.v2.V2ActiveIndexingJobPointer
import com.powerampstartradio.indexing.v2.V2ActiveIndexingJobPointerInspection
import com.powerampstartradio.indexing.v2.V2IndexingPreflightIntentState
import com.powerampstartradio.indexing.v2.V2BootstrapGenerationImporter
import com.powerampstartradio.indexing.v2.V2CurrentModelPolicyResolver
import com.powerampstartradio.indexing.v2.V2EmbeddingCommitRepository
import com.powerampstartradio.indexing.v2.V2FileSha256
import com.powerampstartradio.indexing.v2.V2GenerationArtifactHashProgress
import com.powerampstartradio.indexing.v2.V2IndexGenerationReader
import com.powerampstartradio.indexing.v2.V2IndexingWorkPolicy
import com.powerampstartradio.indexing.v2.V2LibraryDatabaseResolver
import com.powerampstartradio.indexing.v2.V2PowerampProviderSnapshotAcquirer
import com.powerampstartradio.indexing.v2.V2ProviderPathGroupSnapshot
import com.powerampstartradio.indexing.v2.V2ResolvedActiveIndexGeneration
import com.powerampstartradio.indexing.v2.V2ResolvedLibraryDatabase
import com.powerampstartradio.indexing.v2.V2ServerBundleMerger
import com.powerampstartradio.indexing.v2.V2ServerBundleRowDisposition
import com.powerampstartradio.indexing.v2.V2ServerMergeResult
import com.powerampstartradio.poweramp.PowerampHelper
import com.powerampstartradio.poweramp.PowerampReceiver
import com.powerampstartradio.services.RadioService
import com.powerampstartradio.services.RadioRequestAdmission
import com.powerampstartradio.services.RecommendationWorkAdmission
import com.powerampstartradio.services.MusicIndexMutationAdmission
import com.powerampstartradio.services.StableTrackSpanReceiptReader
import com.powerampstartradio.similarity.RecommendationAssetFiles
import com.powerampstartradio.similarity.RecommendationEngine
import com.powerampstartradio.similarity.PreparedRecommendationEmbeddingIndex
import com.google.ai.edge.litert.Accelerator
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.async
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.drop
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.lang.ref.WeakReference
import java.util.Locale
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.abs
import kotlin.math.roundToInt

private fun activeMusicIndexHashStatus(progress: V2GenerationArtifactHashProgress): String =
    exactHashProgressText(
        subject = "active music-index file ${progress.filename}",
        completedBytes = progress.completedBytes,
        totalBytes = progress.totalBytes,
    )

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

internal fun activeLibraryCatalogProgressText(
    progress: V2ActiveLibraryCatalogLoadProgress,
): String = when (progress) {
    is V2ActiveLibraryCatalogLoadProgress.PowerampRows ->
        if (progress.completedRows == 0) {
            "Preparing to normalize ${"%,d".format(progress.totalRows)} Poweramp library rows"
        } else {
            "Normalized ${"%,d".format(progress.completedRows)} of " +
                "${"%,d".format(progress.totalRows)} Poweramp library rows"
        }

    is V2ActiveLibraryCatalogLoadProgress.IndexedRowRead ->
        if (progress.completedRows == null) {
            "Reading ${"%,d".format(progress.totalRows)} indexed track rows from SQLite"
        } else {
            "Read ${"%,d".format(progress.completedRows)} of " +
                "${"%,d".format(progress.totalRows)} indexed track rows from SQLite"
        }

    is V2ActiveLibraryCatalogLoadProgress.IndexedRows ->
        if (progress.completedRows == 0) {
            "Preparing to normalize ${"%,d".format(progress.totalRows)} indexed track rows"
        } else {
            "Normalized ${"%,d".format(progress.completedRows)} of " +
                "${"%,d".format(progress.totalRows)} indexed track rows"
        }

    is V2ActiveLibraryCatalogLoadProgress.SourceSpanReceipts ->
        progress.receiptCount?.let { count ->
            "Read ${"%,d".format(count)} exact source-span receipts for " +
                "${"%,d".format(progress.indexedRowCount)} indexed tracks"
        } ?: "Reading exact source-span receipts for " +
            "${"%,d".format(progress.indexedRowCount)} indexed tracks from SQLite"

    is V2ActiveLibraryCatalogLoadProgress.Bindings ->
        if (progress.activeBindingCount == null || progress.quarantinedRowCount == null) {
            "Reconciling ${"%,d".format(progress.powerampRowCount)} Poweramp rows with " +
                "${"%,d".format(progress.indexedRowCount)} indexed tracks using " +
                "${"%,d".format(progress.receiptCount)} exact source-span receipts"
        } else {
            "Reconciled ${"%,d".format(progress.indexedRowCount)} indexed tracks: " +
                "${"%,d".format(progress.activeBindingCount)} current Poweramp bindings, " +
                "${"%,d".format(progress.quarantinedRowCount)} without a current exact binding"
        }
}

/**
 * ViewModel for the main screen.
 */
class MainViewModel(application: Application) : AndroidViewModel(application) {
    private val indexingHandoffReservationOwner = RecommendationWorkAdmission.uiHandoffOwner(
        UUID.randomUUID().toString(),
    )
    private val musicIndexMutationOwner = RecommendationWorkAdmission.musicIndexMutationOwner(
        UUID.randomUUID().toString(),
    )
    private val libraryLifecycleMutationOwner = MusicIndexMutationAdmission.newOwner(
        "main-view-model:${UUID.randomUUID()}",
    )
    private val prefs = application.getSharedPreferences(
        RadioSettingsStore.PREFERENCES_NAME,
        Context.MODE_PRIVATE,
    )
    private val initialRadioConfig = RadioSettingsStore.from(application)
        .readSnapshot()
        .storedConfig
    private val databaseLifecycleBusy = AtomicBoolean(false)
    private val databaseRefreshGate = V2DatabaseRefreshRequestGate()
    private val modelCheckRunning = AtomicBoolean(false)

    private fun logMemoryPhase(phase: String) {
        val runtime = Runtime.getRuntime()
        val javaUsed = runtime.totalMemory() - runtime.freeMemory()
        val mib = 1024L * 1024L
        Log.i(
            "V2Memory",
            "phase=$phase javaUsedMiB=${javaUsed / mib} " +
                "javaCommittedMiB=${runtime.totalMemory() / mib} " +
                "javaMaxMiB=${runtime.maxMemory() / mib} " +
                "nativeAllocatedMiB=${Debug.getNativeHeapAllocatedSize() / mib}",
        )
    }

    // --- RadioConfig settings ---

    private val _numTracks = MutableStateFlow(initialRadioConfig.numTracks)
    val numTracks: StateFlow<Int> = _numTracks.asStateFlow()

    private val _libraryAddedDays = MutableStateFlow(initialRadioConfig.effectiveLibraryAddedDays)
    val libraryAddedDays: StateFlow<Int?> = _libraryAddedDays.asStateFlow()

    private val _selectionMode = MutableStateFlow(initialRadioConfig.selectionMode)
    val selectionMode: StateFlow<SelectionMode> = _selectionMode.asStateFlow()

    private val _driftEnabled = MutableStateFlow(initialRadioConfig.driftEnabled)
    val driftEnabled: StateFlow<Boolean> = _driftEnabled.asStateFlow()

    private val _driftMode = MutableStateFlow(initialRadioConfig.driftMode)
    val driftMode: StateFlow<DriftMode> = _driftMode.asStateFlow()

    private val _anchorStrength = MutableStateFlow(initialRadioConfig.anchorStrength)
    val anchorStrength: StateFlow<Float> = _anchorStrength.asStateFlow()

    private val _walkRestartAlpha = MutableStateFlow(initialRadioConfig.walkRestartAlpha)
    val walkRestartAlpha: StateFlow<Float> = _walkRestartAlpha.asStateFlow()

    private val _anchorDecay = MutableStateFlow(initialRadioConfig.anchorDecay)
    val anchorDecay: StateFlow<DecaySchedule> = _anchorDecay.asStateFlow()

    private val _anchorHalfLifeTracks = MutableStateFlow(
        initialRadioConfig.anchorHalfLifeTracks,
    )
    val anchorHalfLifeTracks: StateFlow<Float> = _anchorHalfLifeTracks.asStateFlow()

    private val _momentumBeta = MutableStateFlow(initialRadioConfig.momentumBeta)
    val momentumBeta: StateFlow<Float> = _momentumBeta.asStateFlow()

    private val _diversityLambda = MutableStateFlow(initialRadioConfig.diversityLambda)
    val diversityLambda: StateFlow<Float> = _diversityLambda.asStateFlow()

    private val _mmrCandidatePoolFraction = MutableStateFlow(
        initialRadioConfig.mmrCandidatePoolFraction,
    )
    val mmrCandidatePoolFraction: StateFlow<Float> = _mmrCandidatePoolFraction.asStateFlow()

    private val _dppFixedCandidatePoolFraction = MutableStateFlow(
        initialRadioConfig.dppFixedCandidatePoolFraction,
    )
    val dppFixedCandidatePoolFraction: StateFlow<Float> =
        _dppFixedCandidatePoolFraction.asStateFlow()

    private val _dppUsesCertifiedFullDomain = MutableStateFlow(
        initialRadioConfig.dppUsesCertifiedFullDomain,
    )
    val dppUsesCertifiedFullDomain: StateFlow<Boolean> =
        _dppUsesCertifiedFullDomain.asStateFlow()

    private val _dppQualityExponent = MutableStateFlow(initialRadioConfig.dppQualityExponent)
    val dppQualityExponent: StateFlow<Float> = _dppQualityExponent.asStateFlow()

    private val _shuffleSeed = MutableStateFlow(initialRadioConfig.shuffleSeed)
    val shuffleSeed: StateFlow<Long> = _shuffleSeed.asStateFlow()

    private val _artistLimitsEnabled = MutableStateFlow(initialRadioConfig.artistLimitsEnabled)
    val artistLimitsEnabled: StateFlow<Boolean> = _artistLimitsEnabled.asStateFlow()

    private val _maxPerArtist = MutableStateFlow(initialRadioConfig.maxPerArtist)
    val maxPerArtist: StateFlow<Int> = _maxPerArtist.asStateFlow()

    private val _minArtistSpacing = MutableStateFlow(initialRadioConfig.minArtistSpacing)
    val minArtistSpacing: StateFlow<Int> = _minArtistSpacing.asStateFlow()

    private val _findMusicOperator = MutableStateFlow(
        FindMusicOperator.fromWireName(prefs.getString("find_music_operator", null))
    )
    val findMusicOperator: StateFlow<FindMusicOperator> = _findMusicOperator.asStateFlow()

    private val _findMusicRefinePrimaryIngredientIndex = MutableStateFlow(0)
    val findMusicRefinePrimaryIngredientIndex: StateFlow<Int> =
        _findMusicRefinePrimaryIngredientIndex.asStateFlow()

    private val _findMusicRefineNeighborhood = MutableStateFlow(
        FindMusicRefineNeighborhood.entries.firstOrNull {
            it.wireName == prefs.getString("find_music_refine_neighborhood", null)
        } ?: FindMusicRefineNeighborhood.TOP_1_PERCENT,
    )
    val findMusicRefineNeighborhood: StateFlow<FindMusicRefineNeighborhood> =
        _findMusicRefineNeighborhood.asStateFlow()

    private val _findMusicTextResultPlanner = MutableStateFlow(
        FindMusicTextResultPlanner.entries.firstOrNull {
            it.wireName == prefs.getString("find_music_text_result_planner", null)
        } ?: FindMusicTextResultPlanner.CLOSEST,
    )
    val findMusicTextResultPlanner: StateFlow<FindMusicTextResultPlanner> =
        _findMusicTextResultPlanner.asStateFlow()

    // --- Database & permission state ---

    private val _databaseInfo = MutableStateFlow<DatabaseInfo?>(null)
    val databaseInfo: StateFlow<DatabaseInfo?> = _databaseInfo.asStateFlow()

    private val _databaseLoading = MutableStateFlow(true)
    val databaseLoading: StateFlow<Boolean> = _databaseLoading.asStateFlow()
    private val _databaseVerificationStatus = MutableStateFlow<String?>(null)
    val databaseVerificationStatus: StateFlow<String?> =
        _databaseVerificationStatus.asStateFlow()

    private val _hasPermission = MutableStateFlow(false)
    val hasPermission: StateFlow<Boolean> = _hasPermission.asStateFlow()

    private val _permissionLoading = MutableStateFlow(true)
    val permissionLoading: StateFlow<Boolean> = _permissionLoading.asStateFlow()

    // --- Indexing state ---
    // -3 = last check failed, -2 = never checked, -1 = checking now, 0+ = ready count
    // The persisted value is explicitly labeled as the last completed comparison in Settings.
    private val _unindexedCount = MutableStateFlow(
        prefs.getInt("unindexed_count", -2).takeIf { it >= 0 } ?: -2,
    )
    val unindexedCount: StateFlow<Int> = _unindexedCount.asStateFlow()

    private val _unindexedCheckStatus = MutableStateFlow<String?>(null)
    val unindexedCheckStatus: StateFlow<String?> = _unindexedCheckStatus.asStateFlow()

    private val _hasModels = MutableStateFlow(false)
    val hasModels: StateFlow<Boolean> = _hasModels.asStateFlow()

    private val _modelsLoading = MutableStateFlow(true)
    val modelsLoading: StateFlow<Boolean> = _modelsLoading.asStateFlow()

    private val _modelsLoadingStatus = MutableStateFlow<String?>(null)
    val modelsLoadingStatus: StateFlow<String?> = _modelsLoadingStatus.asStateFlow()

    private val _fileStatuses = MutableStateFlow<List<AppFileStatus>>(emptyList())
    val fileStatuses: StateFlow<List<AppFileStatus>> = _fileStatuses.asStateFlow()

    private val _importStatus = MutableStateFlow<String?>(null)
    val importStatus: StateFlow<String?> = _importStatus.asStateFlow()

    private val _serverMergeProgress = MutableStateFlow<ServerMergeProgressState?>(null)
    val serverMergeProgress: StateFlow<ServerMergeProgressState?> =
        _serverMergeProgress.asStateFlow()

    private val _importError = MutableStateFlow<String?>(null)
    val importError: StateFlow<String?> = _importError.asStateFlow()

    private val _musicIndexUpdateResult = MutableStateFlow(
        prefs.getString(LAST_SERVER_MERGE_RESULT_PREF, null),
    )
    val musicIndexUpdateResult: StateFlow<String?> = _musicIndexUpdateResult.asStateFlow()

    private val _libraryLifecycleBusy = MutableStateFlow(false)
    private val _libraryControlsBlockedReason = MutableStateFlow<String?>(null)
    val libraryControlsBlockedReason: StateFlow<String?> =
        _libraryControlsBlockedReason.asStateFlow()

    val indexingState: StateFlow<IndexingService.IndexingState> = IndexingService.state

    private val _previews = MutableStateFlow<Map<SelectionMode, SettingsPeekPreview>>(emptyMap())
    val previews: StateFlow<Map<SelectionMode, SettingsPeekPreview>> = _previews.asStateFlow()

    private val previewJobs = mutableMapOf<SelectionMode, Job>()
    private val activePreviewRunIds = ConcurrentHashMap<SelectionMode, String>()
    private val previewPlanningRevision = AtomicLong(0L)
    private val previewComparisonHistory = SettingsPeekComparisonHistory()
    private val previewStateLock = Any()

    private var rankIndexSnapshot: RankIndexSnapshot? = null

    val radioState: StateFlow<RadioUiState> = RadioService.uiState
    val sessionHistory: StateFlow<List<RadioResult>> = RadioService.sessionHistory

    private val replayEligibilityRevision = AtomicLong(0L)
    private val replayEligibilityRefreshRunning = AtomicBoolean(false)
    private val replayEligibilityRefreshRequested = AtomicBoolean(false)
    private val replayEligibilityJobLock = Any()
    private var replayEligibilityRefreshJob: Job? = null
    @Volatile private var verifiedReplayLibraryBinding: VerifiedReplayLibraryBinding? = null
    private val _sessionReplayEligibility =
        MutableStateFlow<Map<String, SessionReplayEligibility>>(emptyMap())
    val sessionReplayEligibility: StateFlow<Map<String, SessionReplayEligibility>> =
        _sessionReplayEligibility.asStateFlow()

    private val displayedQueueEligibilityRevision = AtomicLong(0L)
    private val _displayedQueueEligibility =
        MutableStateFlow(DisplayedFindMusicQueueEligibility.UNAVAILABLE)
    val displayedQueueEligibility: StateFlow<DisplayedFindMusicQueueEligibility> =
        _displayedQueueEligibility.asStateFlow()

    @Volatile
    private var lastObservedPowerampTrackId = PowerampReceiver.currentTrack?.realId

    private val trackChangeListener: (com.powerampstartradio.poweramp.PowerampTrack?) -> Unit = { track ->
        val previousTrackId = lastObservedPowerampTrackId
        lastObservedPowerampTrackId = track?.realId
        if (previousTrackId != track?.realId) {
            invalidateAllPreviewsForContextChange()
        }
        val state = RadioService.uiState.value
        if (state is RadioUiState.Success) {
            val result = state.result
            val knownIds = buildSet {
                addAll(result.queuedFileIds)
                add(result.seedTrack.realId)
                result.queueAnchorId?.let { add(it) }
            }
            if (track == null || track.realId !in knownIds) {
                RadioService.resetState()
            }
        }
    }

    override fun onCleared() {
        cancelFindMusicRequest()
        clearVerifiedFindMusicCatalogHandoff()
        releaseIndexingHandoffReservation()
        PowerampReceiver.removeTrackChangeListener(trackChangeListener)
        // LiteRT calls are not interruptible. This process-owned scope waits for any in-flight
        // call to release the mutex, then closes the model after viewModelScope is cancelled. Keep
        // this instance in the process registry until cleanup is complete so an indexing handoff
        // racing activity teardown can still find and await every recommendation resource owner.
        TEXT_INFERENCE_CLEANUP_SCOPE.launch {
            try {
                textInferenceMutex.withLock {
                    try {
                        textInference?.close()
                    } catch (e: Exception) {
                        Log.w("MainViewModel", "Failed to close text inference", e)
                    }
                    textInference = null
                    textInferenceIdentity = null
                    textEmbeddingCache.clear()
                }
                synchronized(TEXT_LIBRARY_BUILD_LOCK) {
                    synchronized(this@MainViewModel) {
                        rankIndexSnapshot = null
                        textIndex = null
                        textIndexSnapshot = null
                    }
                    invalidateTextLibrarySnapshot()
                }
            } finally {
                synchronized(MAIN_VIEW_MODEL_INSTANCES_LOCK) {
                    mainViewModelInstances.removeAll {
                        it.get() == null || it.get() === this@MainViewModel
                    }
                }
            }
        }
        super.onCleared()
    }

    /**
     * Build current RadioConfig from all settings.
     */
    fun buildConfig(): RadioConfig = buildStoredConfig().forActiveSelectionMode()

    /** Build the saved control state before applying the selected mode's capability mask. */
    private fun buildStoredConfig(): RadioConfig = RadioConfig(
        numTracks = _numTracks.value,
        libraryAddedDays = _libraryAddedDays.value,
        mmrCandidatePoolFraction = _mmrCandidatePoolFraction.value,
        dppFixedCandidatePoolFraction = _dppFixedCandidatePoolFraction.value,
        dppUsesCertifiedFullDomain = _dppUsesCertifiedFullDomain.value,
        selectionMode = _selectionMode.value,
        driftEnabled = _driftEnabled.value,
        driftMode = _driftMode.value,
        anchorStrength = _anchorStrength.value,
        anchorDecay = _anchorDecay.value,
        anchorHalfLifeTracks = _anchorHalfLifeTracks.value,
        walkRestartAlpha = _walkRestartAlpha.value,
        momentumBeta = _momentumBeta.value,
        diversityLambda = _diversityLambda.value,
        dppQualityExponent = _dppQualityExponent.value,
        shuffleSeed = _shuffleSeed.value,
        artistLimitsEnabled = _artistLimitsEnabled.value,
        maxPerArtist = _maxPerArtist.value,
        minArtistSpacing = _minArtistSpacing.value,
    )

    // --- Setters ---

    fun setNumTracks(count: Int) {
        val canonical = RadioSettingsCodec.canonicalRecommendationCount(count)
        _numTracks.value = canonical
        prefs.edit()
            .putInt("num_tracks", canonical)
            .remove("text_search_top_k")
            .apply()
    }

    fun setLibraryAddedDays(days: Int?) {
        require(days == null || days in 1..MAX_LIBRARY_ADDED_DAYS) {
            "Poweramp added-date day count must be 1..$MAX_LIBRARY_ADDED_DAYS"
        }
        if (_libraryAddedDays.value == days) return
        _libraryAddedDays.value = days
        val editor = prefs.edit().remove("library_added_range")
        if (days == null) {
            editor.remove("library_added_days")
        } else {
            editor.putInt("library_added_days", days)
        }
        editor.apply()
        clearFindMusicResults()
        invalidateAllPreviewsForContextChange()
    }

    fun setSelectionMode(mode: SelectionMode) {
        _selectionMode.value = mode
        prefs.edit().putString("selection_mode", mode.name).apply()
    }

    fun setDriftEnabled(enabled: Boolean) {
        _driftEnabled.value = enabled
        prefs.edit().putBoolean("drift_enabled", enabled).apply()
    }

    fun setDriftMode(mode: DriftMode) {
        _driftMode.value = mode
        prefs.edit().putString("drift_mode", mode.name).apply()
    }

    fun setAnchorStrength(value: Float) {
        val canonical = RadioSettingsCodec.canonicalAnchorStrength(
            value = value,
            decay = _anchorDecay.value,
        )
        _anchorStrength.value = canonical
        prefs.edit().putFloat("anchor_strength", canonical).apply()
    }

    fun setWalkRestartAlpha(value: Float) {
        val canonical = RadioSettingsCodec.canonicalGraphStopChance(value)
        _walkRestartAlpha.value = canonical
        prefs.edit().putFloat("walk_restart_alpha", canonical).apply()
    }

    fun setAnchorDecay(schedule: DecaySchedule) {
        _anchorDecay.value = schedule
        val canonicalAnchor = RadioSettingsCodec.canonicalAnchorStrength(
            value = _anchorStrength.value,
            decay = schedule,
        )
        _anchorStrength.value = canonicalAnchor
        prefs.edit()
            .putString("anchor_decay", schedule.name)
            .putFloat("anchor_strength", canonicalAnchor)
            .apply()
    }

    fun setAnchorHalfLifeTracks(value: Float) {
        val canonical = RadioSettingsCodec.canonicalFadeTiming(value)
        _anchorHalfLifeTracks.value = canonical
        prefs.edit().putFloat("anchor_half_life_tracks", canonical).apply()
    }

    fun setMomentumBeta(value: Float) {
        val canonical = RadioSettingsCodec.canonicalMomentum(value)
        _momentumBeta.value = canonical
        prefs.edit().putFloat("momentum_beta", canonical).apply()
    }

    fun setDiversityLambda(value: Float) {
        val canonical = RadioSettingsCodec.canonicalMmrRelevance(value)
        _diversityLambda.value = canonical
        prefs.edit().putFloat("diversity_lambda", canonical).apply()
    }

    fun setMmrCandidatePoolFraction(value: Float) {
        val canonical = RadioSettingsCodec.canonicalMmrReach(value)
        _mmrCandidatePoolFraction.value = canonical
        prefs.edit().putFloat("mmr_candidate_pool_fraction", canonical).apply()
    }

    fun setDppFixedCandidatePoolFraction(value: Float) {
        val canonical = RadioSettingsCodec.canonicalDppFixedReach(value)
        _dppFixedCandidatePoolFraction.value = canonical
        prefs.edit().putFloat("dpp_fixed_candidate_pool_fraction", canonical).apply()
    }

    fun setDppUsesCertifiedFullDomain(value: Boolean) {
        _dppUsesCertifiedFullDomain.value = value
        prefs.edit().putBoolean("dpp_uses_certified_full_domain", value).apply()
    }

    fun setDppQualityExponent(value: Float) {
        val canonical = RadioSettingsCodec.canonicalDppSeedPull(value)
        _dppQualityExponent.value = canonical
        prefs.edit().putFloat("dpp_quality_exponent", canonical).apply()
    }

    fun setShuffleSeed(value: Long) {
        val canonical = RadioSettingsCodec.canonicalShuffleSeed(value)
        _shuffleSeed.value = canonical
        prefs.edit().putLong("shuffle_seed", canonical).apply()
    }

    /** A deterministic next order: the same starting state and taps produce the same queues. */
    fun advanceShuffleSeed() {
        setShuffleSeed(
            com.powerampstartradio.similarity.algorithms.UniformShuffleSelector.nextSeed(
                _shuffleSeed.value,
            ),
        )
    }

    fun setArtistLimitsEnabled(enabled: Boolean) {
        _artistLimitsEnabled.value = enabled
        prefs.edit().putBoolean("artist_limits_enabled", enabled).apply()
    }

    fun setMaxPerArtist(value: Int) {
        val canonical = RadioSettingsCodec.canonicalMaxPerArtist(value)
        _maxPerArtist.value = canonical
        prefs.edit().putInt("max_per_artist", canonical).apply()
    }

    fun setMinArtistSpacing(value: Int) {
        val canonical = RadioSettingsCodec.canonicalMinArtistSpacing(value)
        _minArtistSpacing.value = canonical
        prefs.edit().putInt("min_artist_spacing", canonical).apply()
    }

    fun setFindMusicOperator(value: FindMusicOperator) {
        val effective = FindMusicEditorPolicy.effectiveOperator(
            requested = value,
            activeIngredientCount = activeIngredientCount(),
        )
        if (effective == FindMusicOperator.REFINE) normalizeRefinePrimaryIngredient()
        persistFindMusicOperator(effective)
        normalizeWeightState(releaseHolds = true)
    }

    fun setFindMusicRefinePrimaryIngredient(index: Int) {
        val signs = activeIngredientSigns()
        if (index !in signs.indices || signs[index]) return
        _findMusicRefinePrimaryIngredientIndex.value = index
    }

    fun setFindMusicRefineNeighborhood(value: FindMusicRefineNeighborhood) {
        _findMusicRefineNeighborhood.value = value
        prefs.edit().putString("find_music_refine_neighborhood", value.wireName).apply()
    }

    fun setFindMusicTextResultPlanner(value: FindMusicTextResultPlanner) {
        val changed = _findMusicTextResultPlanner.value != value
        persistFindMusicTextResultPlanner(value)
        if (!changed) return

        val displayed = currentDisplayedFindMusicResult()
        val displayedSpec = displayed?.querySpec
        if (displayed?.kind == FindMusicResultKind.TEXT &&
            displayedSpec?.isSimplePositiveTextOnly == true
        ) {
            val rerunSpec = displayedSpec.copy(
                textResultPlanner = value,
                libraryBinding = null,
            )
            launchFindMusicRequest(FindMusicSurface.TEXT) { revision ->
                executeTextSearch(revision, FindMusicSurface.TEXT, rerunSpec)
            }
        } else if (
            displayed?.kind == FindMusicResultKind.COMPOSED &&
            displayedSpec?.operator == FindMusicOperator.ALL_OF &&
            displayedSpec.activeIngredientCount >= 2 &&
            value in setOf(
                FindMusicTextResultPlanner.CLOSEST,
                FindMusicTextResultPlanner.VARIED_ALL_OF_DPP,
            )
        ) {
            val rerunSpec = displayedSpec.copy(
                textResultPlanner = value,
                libraryBinding = null,
            )
            launchFindMusicRequest(FindMusicSurface.COMPOSED) { revision ->
                executeComposedSearch(revision, rerunSpec)
            }
        }
    }

    private fun persistFindMusicOperator(value: FindMusicOperator) {
        _findMusicOperator.value = value
        prefs.edit().putString("find_music_operator", value.wireName).apply()
    }

    private fun persistFindMusicTextResultPlanner(value: FindMusicTextResultPlanner) {
        _findMusicTextResultPlanner.value = value
        prefs.edit().putString("find_music_text_result_planner", value.wireName).apply()
    }

    // --- Actions ---

    fun startRadio() {
        RadioService.startRadio(getApplication(), buildConfig())
    }

    fun cancelSearch() {
        RadioService.cancelSearch()
    }

    fun resetRadioState() {
        RadioService.resetState()
    }

    fun clearSessionHistory() {
        RadioService.clearHistory()
    }

    fun requeueSession(session: RadioResult, placement: DirectQueuePlacement) {
        val eligibility = replayEligibilityFor(session)
        if (!eligibility.eligible) {
            Log.w("MainViewModel", "Refusing ineligible history replay: ${eligibility.reason}")
            return
        }
        RadioService.replaySession(
            context = getApplication(),
            session = session,
            placement = placement,
        )
    }

    fun replayEligibilityFor(session: RadioResult): SessionReplayEligibility =
        _sessionReplayEligibility.value[session.replayEligibilityKey()]
            ?: SessionReplayEligibility.CHECKING

    private fun refreshSessionReplayEligibility() {
        val requestedSessions = sessionHistory.value.toList()
        replayEligibilityRevision.incrementAndGet()
        replayEligibilityRefreshRequested.set(true)
        _sessionReplayEligibility.value = requestedSessions.associate { session ->
            session.replayEligibilityKey() to SessionReplayEligibility.CHECKING
        }
        if (requestedSessions.isEmpty()) {
            replayEligibilityRefreshRequested.set(false)
            return
        }
        if (_databaseLoading.value || _libraryLifecycleBusy.value) {
            return
        }
        startPendingSessionReplayEligibilityRefresh()
    }

    private fun startPendingSessionReplayEligibilityRefresh() {
        if (_databaseLoading.value || _libraryLifecycleBusy.value ||
            !replayEligibilityRefreshRequested.get()
        ) {
            return
        }
        lateinit var job: Job
        job = viewModelScope.launch(
            context = Dispatchers.IO,
            start = CoroutineStart.LAZY,
        ) {
            try {
                do {
                    replayEligibilityRefreshRequested.set(false)
                    runSessionReplayEligibilityRefresh()
                } while (
                    replayEligibilityRefreshRequested.get() &&
                    !_databaseLoading.value &&
                    !_libraryLifecycleBusy.value
                )
            } finally {
                replayEligibilityRefreshRunning.set(false)
                if (replayEligibilityRefreshRequested.get() &&
                    !_databaseLoading.value &&
                    !_libraryLifecycleBusy.value
                ) {
                    startPendingSessionReplayEligibilityRefresh()
                }
                synchronized(replayEligibilityJobLock) {
                    if (replayEligibilityRefreshJob === job) {
                        replayEligibilityRefreshJob = null
                    }
                }
            }
        }
        trackRetrievalJobUntilCompletion(job)
        val admitted = synchronized(RETRIEVAL_RESOURCE_ADMISSION_LOCK) {
            if (recommendationResourcesBlocked() ||
                !replayEligibilityRefreshRunning.compareAndSet(false, true)
            ) {
                false
            } else {
                activeRetrievalResourceJobs.add(job)
                synchronized(replayEligibilityJobLock) {
                    replayEligibilityRefreshJob = job
                }
                true
            }
        }
        if (admitted) job.start() else job.cancel()
    }

    private fun runSessionReplayEligibilityRefresh() {
        val sessions = sessionHistory.value.toList()
        val revision = replayEligibilityRevision.get()
        if (sessions.isEmpty()) {
            if (replayEligibilityRevision.get() == revision) {
                _sessionReplayEligibility.value = emptyMap()
            }
            return
        }
        val startedNs = System.nanoTime()
        logMemoryPhase("replay_eligibility_start_$revision")
        val app = getApplication<Application>()
        val active = runCatching { V2IndexGenerationReader.requireActive(app.filesDir) }
            .getOrNull()
        val activeToken = active?.toRadioGenerationToken()
        val replayBinding = verifiedReplayLibraryBinding?.takeIf {
            it.generation == activeToken
        }
        val exactEmbeddedRowCandidateSessions = if (activeToken == null || replayBinding == null) {
            emptySet()
        } else {
            sessions.asSequence()
                .filter { session ->
                    session.generation == activeToken ||
                        (session.providerGenerationId != null &&
                            session.providerGenerationId == replayBinding.providerGenerationId)
                }
                .map(RadioResult::replayEligibilityKey)
                .toSet()
        }
        val relevantTrackIds = if (activeToken == null) {
            emptySet()
        } else {
            sessions.asSequence()
                .filter { it.replayEligibilityKey() in exactEmbeddedRowCandidateSessions }
                .flatMap { session ->
                    session.tracks.asSequence()
                        .filter { it.status == QueueStatus.QUEUED }
                        .map { it.track.id }
                }
                .toSet()
        }
        val crossGenerationStableIds = if (activeToken == null) {
            emptySet()
        } else {
            sessions.asSequence()
                .filter {
                    it.generation != null &&
                        it.generation != activeToken
                }
                .flatMap { session ->
                    session.tracks.asSequence()
                        .filter { it.status == QueueStatus.QUEUED }
                        .mapNotNull { it.stableTrackSpanId }
                }
                .toSet()
        }
        val currentIdentities = if (active == null || relevantTrackIds.isEmpty()) {
            emptyMap()
        } else {
            runCatching {
                val stableIds = StableTrackSpanReceiptReader.readMany(
                    databaseFile = active.databaseFile,
                    trackIds = relevantTrackIds,
                    embeddingSpecId = active.manifest.receiptEmbeddingSpec.specId,
                )
                val database = EmbeddingDatabase.open(active.databaseFile)
                try {
                    relevantTrackIds.associateWith { trackId ->
                        CurrentReplayTrackIdentity(
                            track = database.getTrackById(trackId),
                            stableTrackSpanId = stableIds[trackId],
                        )
                    }
                } finally {
                    database.close()
                }
            }.getOrDefault(emptyMap())
        }
        val crossGenerationTrackIdsByStableId: Map<String, List<Long>>? = when {
            active == null -> null
            crossGenerationStableIds.isEmpty() -> emptyMap()
            else -> runCatching {
                val index = getOrOpenRankIndex()
                    ?: error("Active generation embedding index is unavailable")
                val database = EmbeddingDatabase.open(active.databaseFile)
                try {
                    val catalog = StableTrackIdentityCatalog.load(
                        app.filesDir,
                        database,
                        index,
                    )
                    buildMap {
                        for (stableId in crossGenerationStableIds) {
                            val resolution = catalog.resolveStable(stableId)
                            if (resolution is StableTrackIdentityResolution.Resolved) {
                                put(stableId, resolution.allEquivalentTrackIds)
                            }
                        }
                    }
                } finally {
                    database.close()
                }
            }.getOrNull()
        }
        val availableFullContentStableIds = crossGenerationTrackIdsByStableId?.keys
        val resolvedRowsBySession = sessions.associate { session ->
            val queuedRows = session.tracks.filter { it.status == QueueStatus.QUEUED }
            val sameGeneration = session.generation == activeToken
            val sameProviderGeneration = session.providerGenerationId != null &&
                session.providerGenerationId == replayBinding?.providerGenerationId
            val resolutions = queuedRows.map { row ->
                val savedPowerampFileId = row.resolvedPowerampFileId
                    ?: return@map null to false
                val current = currentIdentities[row.track.id]
                val currentPowerampFileId = replayBinding?.powerampFileIdForTrack(row.track.id)
                val exactRowAuthenticated = ReplayExactRowAuthenticationPolicy.isAuthenticated(
                    sameGeneration = sameGeneration,
                    sameProviderGeneration = sameProviderGeneration,
                    savedTrack = row.track,
                    savedStableTrackSpanId = row.stableTrackSpanId,
                    savedPowerampFileId = savedPowerampFileId,
                    currentTrack = current?.track,
                    currentStableTrackSpanId = current?.stableTrackSpanId,
                    currentPowerampFileId = currentPowerampFileId,
                )
                if (exactRowAuthenticated) {
                    row.track.id to true
                } else {
                    val stableEquivalentTrackIds = if (sameGeneration) {
                        emptyList()
                    } else {
                        row.stableTrackSpanId?.let {
                            crossGenerationTrackIdsByStableId?.get(it)
                        }.orEmpty()
                    }
                    val stableTrackId = StableReplayTrackSelectionPolicy.select(
                        equivalentTrackIds = stableEquivalentTrackIds,
                        savedPowerampFileId = savedPowerampFileId,
                        preferSavedPowerampOccurrence = sameProviderGeneration,
                        currentPowerampFileId = { trackId ->
                            replayBinding?.powerampFileIdForTrack(trackId)
                        },
                    )
                    stableTrackId to false
                }
            }
            session.replayEligibilityKey() to Pair(
                resolutions.map { resolution ->
                    resolution.first?.let { replayBinding?.powerampFileIdForTrack(it) }
                },
                resolutions.map { it.second },
            )
        }
        val eligibility = sessions.associate { session ->
            session.replayEligibilityKey() to SessionReplayEligibilityPolicy.evaluate(
                session = session,
                activeGeneration = activeToken,
                resolvedCurrentPowerampFileIds =
                    resolvedRowsBySession[session.replayEligibilityKey()]?.first,
                resolvedByExactEmbeddedRows =
                    resolvedRowsBySession[session.replayEligibilityKey()]?.second,
                currentTrackIdentities = currentIdentities,
                activeProviderGenerationId = replayBinding?.providerGenerationId,
                availableFullContentStableTrackSpanIds = availableFullContentStableIds,
            )
        }
        if (replayEligibilityRevision.get() == revision) {
            _sessionReplayEligibility.value = eligibility
        }
        logMemoryPhase("replay_eligibility_end_$revision")
        Log.i(
            "MainViewModel",
            "Replay eligibility refresh $revision completed in " +
                "${(System.nanoTime() - startedNs) / 1_000_000L}ms; " +
                "sessions=${sessions.size}, exactBindings=${replayBinding?.activeTrackCount ?: 0}",
        )
    }

    private fun currentDisplayedFindMusicResult(): TextSearchResult? =
        _multiSeedResult.value ?: _textSearchResult.value

    private fun clearDisplayedQueueEligibility() {
        displayedQueueEligibilityRevision.incrementAndGet()
        _displayedQueueEligibility.value = DisplayedFindMusicQueueEligibility.UNAVAILABLE
    }

    private fun refreshDisplayedQueueEligibility() {
        val result = currentDisplayedFindMusicResult()
        val revision = displayedQueueEligibilityRevision.incrementAndGet()
        if (result == null || result.error != null || result.matches.isEmpty()) {
            _displayedQueueEligibility.value = DisplayedFindMusicQueueEligibility.UNAVAILABLE
            return
        }
        _displayedQueueEligibility.value = DisplayedFindMusicQueueEligibility.CHECKING
        if (_databaseLoading.value || _libraryLifecycleBusy.value) return
        viewModelScope.launch(Dispatchers.IO) {
            val eligibility = runCatching {
                val app = getApplication<Application>()
                val active = V2IndexGenerationReader.requireActive(app.filesDir)
                val activeToken = active.toRadioGenerationToken()
                val replayBinding = requireNotNull(verifiedReplayLibraryBinding?.takeIf {
                    it.generation == activeToken
                }) { "The verified active-library binding is not ready" }
                val trackIds = result.matches.map { it.track.id }.toSet()
                val stableIds = StableTrackSpanReceiptReader.readMany(
                    databaseFile = active.databaseFile,
                    trackIds = trackIds,
                    embeddingSpecId = active.manifest.receiptEmbeddingSpec.specId,
                )
                val database = EmbeddingDatabase.open(active.databaseFile)
                val currentTracks = try {
                    trackIds.associateWith { trackId ->
                        CurrentDisplayedFindMusicTrack(
                            existsInGeneration = database.getTrackById(trackId) != null,
                            stableTrackSpanId = stableIds[trackId],
                            activePowerampFileId = replayBinding.powerampFileIdForTrack(trackId),
                        )
                    }
                } finally {
                    database.close()
                }
                DisplayedFindMusicQueueEligibilityPolicy.evaluate(
                    result = result,
                    activeGeneration = activeToken,
                    activeProviderGenerationId = replayBinding.providerGenerationId,
                    orderedActiveTrackIdsSha256 =
                        replayBinding.orderedActiveTrackIdsSha256,
                    activeTrackCount = replayBinding.activeTrackCount,
                    currentTracks = currentTracks,
                )
            }.getOrElse { error ->
                Log.w("MainViewModel", "Displayed Find Music eligibility check failed", error)
                DisplayedFindMusicQueueEligibility(
                    eligible = false,
                    reason = "This ranking no longer matches the current library. Run Find Music again.",
                )
            }
            if (displayedQueueEligibilityRevision.get() == revision &&
                currentDisplayedFindMusicResult() === result
            ) {
                _displayedQueueEligibility.value = eligibility
            }
        }
    }

    private fun getOrOpenRankIndex(): EmbeddingIndex? {
        val filesDir = getApplication<Application>().filesDir
        val activeGeneration = runCatching {
            V2LibraryDatabaseResolver.requirePublished(filesDir)
        }.getOrNull() ?: return null
        return runCatching { getOrOpenRankIndex(activeGeneration) }.getOrNull()
    }

    @Synchronized
    private fun getOrOpenRankIndex(
        activeGeneration: V2ResolvedActiveIndexGeneration,
    ): EmbeddingIndex {
        check(!recommendationResourcesBlocked()) {
            "Recommendation resources are paused while on-device indexing is open"
        }
        val embeddingFile = activeGeneration.embeddingFile
        check(embeddingFile.isFile) { "Active V2 embedding file is missing" }
        val signature = RankIndexSignature(
            canonicalPath = embeddingFile.canonicalPath,
            byteLength = embeddingFile.length(),
            modifiedMs = embeddingFile.lastModified(),
            activationBindingId = activeGeneration.manifest.activationBindingId,
            embeddingSha256 = activeGeneration.manifest.embeddingSha256,
        )
        rankIndexSnapshot?.takeIf { it.signature == signature }?.let { return it.index }
        check(
            EmbeddingIndex.readHeaderTrackCount(embeddingFile) ==
                activeGeneration.manifest.trackCount,
        ) { "Active V2 embedding file disagrees with its generation manifest" }
        return EmbeddingIndex.mmap(embeddingFile).also { index ->
            check(index.numTracks == activeGeneration.manifest.trackCount) {
                "Active V2 embedding index changed while mapping"
            }
            rankIndexSnapshot = RankIndexSnapshot(signature, index)
        }
    }

    private data class RankIndexSnapshot(
        val signature: RankIndexSignature,
        val index: EmbeddingIndex,
    )

    private data class RankIndexSignature(
        val canonicalPath: String,
        val byteLength: Long,
        val modifiedMs: Long,
        val activationBindingId: String?,
        val embeddingSha256: String,
    )

    fun invalidatePreview(mode: SelectionMode) {
        val job = synchronized(previewStateLock) {
            activePreviewRunIds.remove(mode)
            _previews.value = _previews.value - mode
            previewJobs.remove(mode)
        }
        job?.cancel()
    }

    /** Resume only restores resources; the refreshed Poweramp track listener invalidates if needed. */
    fun onActivityResumed() {
        resumeRetrievalResourcesAfterIndexingIfIdle()
    }

    fun computePreview(mode: SelectionMode) {
        val planningRunId = "settings-peek-${previewPlanningRevision.incrementAndGet()}"
        lateinit var job: Job
        job = viewModelScope.launch(
            context = Dispatchers.IO,
            start = CoroutineStart.LAZY,
        ) {
            try {
                val result = runPreviewForMode(mode, planningRunId)
                currentCoroutineContext().ensureActive()
                val currentContext = if (result is SettingsPeekPreview.Ready) {
                    currentSettingsPeekPublicationContext(result.publicationContext)
                } else {
                    null
                }
                currentCoroutineContext().ensureActive()
                synchronized(previewStateLock) {
                    val activeRunId = activePreviewRunIds[mode]
                    if (activeRunId == planningRunId && previewJobs[mode] === job) {
                        val presented = if (result is SettingsPeekPreview.Ready) {
                            if (SettingsPeekPublicationPolicy.canPublish(
                                    planningRunId = planningRunId,
                                    activePlanningRunId = activeRunId,
                                    plannedContext = result.publicationContext,
                                    currentContext = currentContext,
                                )
                            ) {
                                val comparison =
                                    previewComparisonHistory.compareAndRemember(result.snapshot)
                                result.copy(
                                    comparisonLine = comparison?.messages?.joinToString(" "),
                                )
                            } else {
                                SettingsPeekPreview.Unavailable(
                                    SettingsPeekUnavailableReason
                                        .CONTEXT_CHANGED_DURING_PLANNING,
                                )
                            }
                        } else {
                            result
                        }
                        _previews.value = _previews.value + (mode to presented)
                    }
                }
            } finally {
                synchronized(previewStateLock) {
                    activePreviewRunIds.remove(mode, planningRunId)
                    if (previewJobs[mode] === job) previewJobs.remove(mode)
                }
            }
        }
        trackRetrievalJobUntilCompletion(job)
        var admitted = false
        val previous = synchronized(RETRIEVAL_RESOURCE_ADMISSION_LOCK) {
            if (recommendationResourcesBlocked()) {
                null
            } else {
                activeRetrievalResourceJobs.add(job)
                synchronized(previewStateLock) {
                    val old = previewJobs.put(mode, job)
                    activePreviewRunIds[mode] = planningRunId
                    _previews.value = _previews.value + (mode to SettingsPeekPreview.Loading)
                    admitted = true
                    old
                }
            }
        }
        if (!admitted) {
            job.cancel(CancellationException("On-device indexing owns recommendation resources"))
            return
        }
        previous?.cancel()
        job.start()
    }

    private fun invalidateAllPreviewsForContextChange(
        reason: SettingsPeekUnavailableReason =
            SettingsPeekUnavailableReason.CONTEXT_CHANGED_DURING_PLANNING,
    ) {
        val jobs = synchronized(previewStateLock) {
            markAllPreviewsContextChangedLocked(reason)
        }
        jobs.forEach { it.cancel() }
    }

    /** Caller owns [previewStateLock] so invalidation is atomic with context publication. */
    private fun markAllPreviewsContextChangedLocked(
        reason: SettingsPeekUnavailableReason =
            SettingsPeekUnavailableReason.CONTEXT_CHANGED_DURING_PLANNING,
    ): List<Job> {
        val oldJobs = previewJobs.values.toList()
        previewJobs.clear()
        activePreviewRunIds.clear()
        _previews.value = SettingsPeekContextInvalidationPolicy.invalidate(
            previews = _previews.value,
            reason = reason,
        )
        return oldJobs
    }

    private fun currentSettingsPeekPublicationContext(
        expected: SettingsPeekPublicationContext,
    ): SettingsPeekPublicationContext? {
        val app = getApplication<Application>()
        return runCatching {
            val active = V2IndexGenerationReader.requireActive(app.filesDir)
            val generation = active.toRadioGenerationToken()
            require(generation == expected.generation) {
                "Active embeddings generation changed during Peek publication"
            }
            val published = requireNotNull(_databaseInfo.value) {
                "The verified music index is not ready"
            }
            require(
                published.generationId == generation.generationId &&
                    published.providerGenerationId == expected.providerGenerationId
            ) {
                "The published music-library binding changed during Peek publication"
            }
            val currentTrack = requireNotNull(
                PowerampReceiver.requireProviderVerifiedCurrentTrack(app),
            ) { "Poweramp has no provider-verified current track" }
            SettingsPeekPublicationContext(
                generation = generation,
                providerGenerationId = expected.providerGenerationId,
                seedPowerampFileId = currentTrack.realId,
            )
        }.onFailure { error ->
            Log.i("MainViewModel", "Settings Peek publication context changed", error)
        }.getOrNull()
    }

    private suspend fun runPreviewForMode(
        mode: SelectionMode,
        planningRunId: String,
    ): SettingsPeekPreview {
        val currentTrack = try {
            PowerampReceiver.requireProviderVerifiedCurrentTrack(getApplication())
        } catch (cancelled: CancellationException) {
            throw cancelled
        } catch (error: Exception) {
            Log.w("MainViewModel", "Settings Peek has no provider-verified current track", error)
            return SettingsPeekPreview.Unavailable(
                SettingsPeekUnavailableReason.NO_PROVIDER_VERIFIED_CURRENT_TRACK,
            )
        } ?: return SettingsPeekPreview.Unavailable(
            SettingsPeekUnavailableReason.NO_PROVIDER_VERIFIED_CURRENT_TRACK,
        )
        val filesDir = getApplication<Application>().filesDir
        try {
            V2LibraryDatabaseResolver.requirePublished(filesDir)
        } catch (cancelled: CancellationException) {
            throw cancelled
        } catch (error: Exception) {
            Log.w("MainViewModel", "Settings Peek has no active validated library", error)
            return SettingsPeekPreview.Unavailable(
                SettingsPeekUnavailableReason.NO_ACTIVE_VALIDATED_LIBRARY,
            )
        }
        val library = try {
            getOrCreateTextLibrarySnapshot(filesDir)
        } catch (cancelled: CancellationException) {
            throw cancelled
        } catch (error: Exception) {
            Log.e("MainViewModel", "Settings Peek could not bind the active library", error)
            return SettingsPeekErrorPresentation.from(error)
        }
        if (!isCurrentPreviewLibrary(filesDir, library)) {
            return SettingsPeekPreview.Unavailable(
                SettingsPeekUnavailableReason.NO_ACTIVE_VALIDATED_LIBRARY,
            )
        }
        val seedId = library.activeDomain.trackIdForPowerampFile(currentTrack.realId)
            ?: return SettingsPeekPreview.Unavailable(
                SettingsPeekUnavailableReason.SEED_ABSENT_FROM_ACTIVE_DOMAIN,
            )

        return try {
            val db = EmbeddingDatabase.open(library.databaseFile)
            try {
                val pinnedAssets = RecommendationAssetFiles(
                    embeddingFile = library.embeddingFile,
                    graphFile = library.graphFile,
                    activationBindingId = library.activationBindingId,
                )
                val engine = RecommendationEngine(
                    database = db,
                    filesDir = filesDir,
                    pinnedAssets = pinnedAssets,
                    activeCatalog = library.activeCatalog,
                    preparedEmbeddingIndex = PreparedRecommendationEmbeddingIndex(
                        embeddingFile = library.embeddingFile,
                        activationBindingId = checkNotNull(library.activationBindingId) {
                            "Peek requires an exact active-generation binding"
                        },
                        index = library.embeddingIndex,
                    ),
                )
                engine.ensureIndices()

                val requestReferenceEpochSecond = System.currentTimeMillis() / 1_000L
                val storedConfig = buildStoredConfig().copy(selectionMode = mode)
                val candidateDomain = library.activeDomain.candidateDomain(
                    minimumLibraryAddedAtEpochSecond(
                        storedConfig.effectiveLibraryAddedDays,
                        requestReferenceEpochSecond,
                    ),
                )
                val eligibleCandidateIdentityCount =
                    candidateDomain.eligibleCandidateIdentityCount(seedId)
                val config = storedConfig
                    .forSelectionRequest(eligibleCandidateIdentityCount)
                var selectorRankedCandidateCount: Int? = null
                val tracks = engine.generatePlaylist(
                    seedTrackId = seedId,
                    config = config,
                    requestReferenceEpochSecond = requestReferenceEpochSecond,
                    onGraphExplorationEvidence = { evidence ->
                        selectorRankedCandidateCount = evidence.rankedCandidateCount
                    },
                    onUniformShuffleIdentityEvidence = { evidence ->
                        selectorRankedCandidateCount = evidence.rankedCandidateCount
                    },
                )
                if (!isCurrentPreviewLibrary(filesDir, library)) {
                    return SettingsPeekPreview.Unavailable(
                        SettingsPeekUnavailableReason.NO_ACTIVE_VALIDATED_LIBRARY,
                    )
                }
                SettingsPeekPreview.Ready(
                    snapshot = PlanSnapshot(
                        planningRunId = planningRunId,
                        materialization = PlanMaterialization.FRESH,
                        generation = library.generation,
                        providerGenerationId = library.activeDomain.binding.providerGenerationId,
                        seedIdentity = RadioSeedIdentity(
                            embeddedTrackId = seedId,
                            stableTrackSpanId =
                                library.identityCatalog.stableTrackSpanId(seedId),
                        ),
                        selectionMode = mode,
                        semanticControls = PlanSemanticControlPolicy.fromConfig(
                            source = config,
                            eligibleCandidateIdentityCount = eligibleCandidateIdentityCount,
                        ),
                        candidateCount = selectorRankedCandidateCount
                            ?: PlanSemanticControlPolicy.semanticCandidateCount(
                                config = config,
                                eligibleCandidateIdentityCount =
                                    eligibleCandidateIdentityCount,
                            ),
                        orderedTrackIds = tracks.map { it.track.id },
                    ),
                    publicationContext = SettingsPeekPublicationContext(
                        generation = library.generation,
                        providerGenerationId =
                            library.activeDomain.binding.providerGenerationId,
                        seedPowerampFileId = currentTrack.realId,
                    ),
                    firstDisplayLabels = tracks
                        .take(SettingsPeekPreview.DISPLAY_LIMIT)
                        .map { track ->
                            "${track.track.title ?: "?"} \u2013 ${track.track.artist ?: "?"}"
                        },
                )
            } finally {
                db.close()
            }
        } catch (cancelled: CancellationException) {
            throw cancelled
        } catch (error: Exception) {
            Log.e("MainViewModel", "Settings Peek planning failed for $mode", error)
            SettingsPeekErrorPresentation.from(error)
        }
    }

    // --- Text search state ---

    /** Current text search result. Null when idle, non-null after a search completes. */
    private val _textSearchResult = MutableStateFlow<TextSearchResult?>(null)
    val textSearchResult: StateFlow<TextSearchResult?> = _textSearchResult.asStateFlow()

    private val _textSearchLoading = MutableStateFlow(false)
    val textSearchLoading: StateFlow<Boolean> = _textSearchLoading.asStateFlow()

    private val _findMusicLoadingStatus = MutableStateFlow<String?>(null)
    val findMusicLoadingStatus: StateFlow<String?> = _findMusicLoadingStatus.asStateFlow()

    /** Recent searches (persisted across sessions, includes full multi-seed state). */
    private val _recentSearches = MutableStateFlow<List<RecentSearch>>(loadRecentSearches())
    val recentSearches: StateFlow<List<RecentSearch>> = _recentSearches.asStateFlow()

    private fun loadRecentSearches(): List<RecentSearch> {
        prefs.getString("find_music_recent_queries_v3", null)?.let { json ->
            try {
                val decoded = FindMusicQuerySpecCodec.fromJsonArray(json)
                val canonical = FindMusicQuerySpecCodec.toJsonArray(decoded)
                if (canonical != json) persistRecentSearches(decoded)
                return decoded
            } catch (e: Exception) {
                Log.w("MainViewModel", "Failed to parse find_music_recent_queries_v3", e)
            }
        }
        // Migrate the old structured format without changing its weights or signs.
        prefs.getString("recent_searches_v2", null)?.let { json ->
            try {
                val migrated = FindMusicQuerySpecCodec.fromJsonArray(json)
                persistRecentSearches(migrated)
                return migrated
            } catch (e: Exception) {
                Log.w("MainViewModel", "Failed to parse recent_searches_v2", e)
            }
        }
        // Preserve V1 text labels as visibly legacy; do not silently adopt rank-v3 semantics.
        prefs.getString("recent_searches", null)?.let { v1 ->
            val migrated = v1.split("\u0000").filter { it.isNotBlank() }
                .map { query ->
                    RecentSearch(
                        schemaVersion = 1,
                        rankingVersion = FindMusicQuerySpec.LEGACY_RANKING_VERSION,
                        textIngredients = listOf(
                            FindMusicTextIngredient(
                                query = query,
                                weight = 1f,
                                negative = false,
                            ),
                        ),
                    )
                }
            if (migrated.isNotEmpty()) {
                persistRecentSearches(migrated)
            }
            return migrated
        }
        return emptyList()
    }

    private fun persistRecentSearches(searches: List<RecentSearch>) {
        prefs.edit()
            .putString(
                "find_music_recent_queries_v3",
                FindMusicQuerySpecCodec.toJsonArray(searches),
            )
            .remove("recent_searches_v2")
            .remove("recent_searches")
            .apply()
    }

    private var textInference: Clamp3TextInference? = null
    private var textInferenceIdentity: TextEmbeddingRuntimeIdentity? = null
    private val textInferenceMutex = Mutex()
    private val textEmbeddingCache = ExactTextEmbeddingCache(maxEntries = 32)
    private var textIndex: EmbeddingIndex? = null
    private var textIndexSnapshot: TextEmbeddingIndexSnapshot? = null

    private data class FindMusicLibrarySnapshot(
        val databaseFile: File,
        val embeddingFile: File,
        val graphFile: File?,
        val embeddingIndex: EmbeddingIndex,
        val activationBindingId: String?,
        val identityCatalog: StableTrackIdentityCatalog,
        val activeCatalog: V2ActiveLibraryCatalog,
        val activeDomain: ActiveRecommendationDomain,
        val textAssets: TextRetrievalAssets,
        val generation: RadioGenerationToken,
    )

    private data class VerifiedFindMusicCatalogHandoff(
        val generation: RadioGenerationToken,
        val activeCatalog: V2ActiveLibraryCatalog,
    )

    private val findMusicCatalogHandoffLock = Any()
    private var verifiedFindMusicCatalogHandoff: VerifiedFindMusicCatalogHandoff? = null

    private data class TextRetrievalAssets(
        val modelFile: File,
        val tokenizerModelFile: File,
        val modelSha256: String,
        val tokenizerModelSha256: String,
        val identity: String,
    ) {
        fun runtimeIdentity() = TextEmbeddingRuntimeIdentity(
            retrievalSpecIdentity = identity,
            textModelSha256 = modelSha256,
            tokenizerModelSha256 = tokenizerModelSha256,
            tokenizerRuntimeContractSha256 =
                OfficialSentencePieceTokenizer.CONTRACT_SHA256,
            inferenceBackendPolicyId =
                V2IndexingWorkPolicy.TEXT_INFERENCE_BACKEND_POLICY_ID,
        )
    }

    private data class CachedAssetHash(
        val length: Long,
        val modifiedMs: Long,
        val sha256: String,
    )

    private val textAssetHashes = HashMap<String, CachedAssetHash>()

    private enum class FindMusicSurface { TEXT, COMPOSED }

    private fun FindMusicSurface.resultKind(): FindMusicResultKind = when (this) {
        FindMusicSurface.TEXT -> FindMusicResultKind.TEXT
        FindMusicSurface.COMPOSED -> FindMusicResultKind.COMPOSED
    }

    private val findMusicRequestGate = LatestFindMusicRequestGate()
    private val findMusicJobLock = Any()
    private var activeFindMusicJob: Job? = null

    /**
     * Start one immutable request. Cancelling a coroutine cannot interrupt a LiteRT call,
     * so every publication is additionally guarded by the monotonic revision.
     */
    private fun launchFindMusicRequest(
        surface: FindMusicSurface,
        block: suspend (revision: Long) -> Unit,
    ) {
        val revision = findMusicRequestGate.begin()
        val job = viewModelScope.launch(
            context = Dispatchers.IO,
            start = CoroutineStart.LAZY,
        ) {
            try {
                block(revision)
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                Log.e("MainViewModel", "Find Music request failed", e)
                publishFindMusicError(
                    revision = revision,
                    surface = surface,
                    query = "Find music",
                    message = "Find Music could not complete this search. Try again.",
                )
            } finally {
                synchronized(findMusicJobLock) {
                    if (
                        findMusicRequestGate.isCurrent(revision) &&
                        activeFindMusicJob === coroutineContext[Job]
                    ) {
                        _textSearchLoading.value = false
                        _multiSeedLoading.value = false
                        _findMusicLoadingStatus.value = null
                        activeFindMusicJob = null
                    }
                }
            }
        }

        trackRetrievalJobUntilCompletion(job)
        var admitted = false
        val previous = synchronized(RETRIEVAL_RESOURCE_ADMISSION_LOCK) {
            if (recommendationResourcesBlocked()) {
                null
            } else {
                activeRetrievalResourceJobs.add(job)
                synchronized(findMusicJobLock) {
                    activeFindMusicJob.also {
                        admitted = true
                        activeFindMusicJob = job
                        _textSearchLoading.value = surface == FindMusicSurface.TEXT
                        _multiSeedLoading.value = surface == FindMusicSurface.COMPOSED
                        _findMusicLoadingStatus.value =
                            "Opening the active music index for this search"
                        _textSearchResult.value = null
                        _multiSeedResult.value = null
                        clearDisplayedQueueEligibility()
                    }
                }
            }
        }
        if (!admitted) {
            job.cancel(CancellationException("On-device indexing owns recommendation resources"))
            return
        }
        previous?.cancel(CancellationException("Superseded by Find Music request $revision"))
        job.start()
    }

    private suspend fun ensureCurrentFindMusicRequest(revision: Long) {
        currentCoroutineContext().ensureActive()
        if (!findMusicRequestGate.isCurrent(revision)) {
            throw CancellationException("Find Music request $revision is no longer current")
        }
    }

    private fun cancelFindMusicRequest(): List<Job> {
        findMusicRequestGate.cancel()
        val job = synchronized(findMusicJobLock) {
            activeFindMusicJob.also {
                activeFindMusicJob = null
                _textSearchLoading.value = false
                _multiSeedLoading.value = false
                _findMusicLoadingStatus.value = null
            }
        }
        job?.cancel(CancellationException("Find Music screen cleared"))
        val songLookupJob = cancelSongSeedLookup()
        return listOfNotNull(job, songLookupJob)
    }

    private fun publishFindMusicResult(
        revision: Long,
        surface: FindMusicSurface,
        result: TextSearchResult,
    ) {
        synchronized(findMusicJobLock) {
            if (!findMusicRequestGate.isCurrent(revision)) return
            val typedResult = result.copy(kind = surface.resultKind())
            when (surface) {
                FindMusicSurface.TEXT -> _textSearchResult.value = typedResult
                FindMusicSurface.COMPOSED -> _multiSeedResult.value = typedResult
            }
        }
        refreshDisplayedQueueEligibility()
    }

    private fun publishFindMusicLoadingStatus(revision: Long, status: String) {
        if (findMusicRequestGate.isCurrent(revision)) {
            _findMusicLoadingStatus.value = status
        }
    }

    private fun publishFindMusicError(
        revision: Long,
        surface: FindMusicSurface,
        query: String,
        message: String,
        querySpec: FindMusicQuerySpec? = null,
        unresolvedAnchors: List<FindMusicSongAnchor> = emptyList(),
    ) {
        publishFindMusicResult(
            revision = revision,
            surface = surface,
            result = TextSearchResult(
                query = query,
                error = message,
                querySpec = querySpec,
                unresolvedAnchors = unresolvedAnchors,
            ),
        )
    }

    /**
     * Return an index from the same published generation as the current database.
     * Rebuilt indices are atomically replaced, so an older mmap remains a valid snapshot
     * until this method observes the new publication and remaps it.
     */
    private fun getOrCreateTextLibrarySnapshot(
        filesDir: File,
        onStatus: (String) -> Unit = {},
    ): FindMusicLibrarySnapshot =
        synchronized(TEXT_LIBRARY_BUILD_LOCK) {
            check(!recommendationResourcesBlocked()) {
                "Recommendation resources are paused while on-device indexing is open"
            }
            getOrCreateTextLibrarySnapshotWhileBuildLocked(filesDir, onStatus)
        }

    private fun getOrCreateTextLibrarySnapshotWhileBuildLocked(
        filesDir: File,
        onStatus: (String) -> Unit,
    ): FindMusicLibrarySnapshot {
        val startedNs = System.nanoTime()
        onStatus("Validating the active music-index files")
        val activeGeneration = V2LibraryDatabaseResolver.requirePublished(filesDir) { progress ->
            onStatus(activeMusicIndexHashStatus(progress))
        }
        val dbFile = activeGeneration.databaseFile
        val embFile = activeGeneration.embeddingFile
        val textAssets = resolveTextRetrievalAssets(
            filesDir = filesDir,
            expectedSpec = activeGeneration.manifest.textRetrievalSpec,
            onStatus = onStatus,
        )
        check(dbFile.isFile) { "No embedding database found" }
        onStatus("Reading the CLaMP3 embedding count from the music-index database")
        val db = EmbeddingDatabase.open(dbFile)
        val databaseCount = try {
            db.getEmbeddingCountForTable("embeddings_clamp3")
        } finally {
            db.close()
        }
        check(databaseCount > 0) { "No CLaMP3 embeddings in database" }

        val generation = TextEmbeddingIndexGeneration.current()
        val headerCount = EmbeddingIndex.readHeaderTrackCount(embFile)
        currentProcessTextLibrarySnapshot()?.let { snapshot ->
            if (
                snapshot.embeddingIndex.numTracks == databaseCount &&
                headerCount == databaseCount &&
                snapshot.matchesStaticLibrary(
                    activeGeneration = activeGeneration,
                    textAssets = textAssets,
                )
            ) {
                textIndex = snapshot.embeddingIndex
                textIndexSnapshot = TextEmbeddingIndexSnapshot.capture(
                    generation = generation,
                    trackCount = databaseCount,
                    file = embFile,
                )
                Log.i(
                    "FindMusicLibrary",
                    "library cache=hit tracks=$databaseCount active=${snapshot.activeDomain.activeTrackCount} " +
                        "totalMs=${elapsedMs(startedNs)}",
                )
                return snapshot
            }
            invalidateTextLibrarySnapshot(snapshot)
        }
        val cached = textIndex
        if (
            cached != null &&
            cached.numTracks == databaseCount &&
            headerCount == databaseCount &&
            textIndexSnapshot?.matches(generation, databaseCount, embFile) == true
        ) {
            return bindFindMusicLibrarySnapshot(
                filesDir = filesDir,
                activeGeneration = activeGeneration,
                embeddingIndex = cached,
                textAssets = textAssets,
                lookupStartedNs = startedNs,
                onStatus = onStatus,
            )
        }

        textIndex = null
        invalidateTextLibrarySnapshot()
        if (headerCount != databaseCount) {
            error("Active V2 embedding file disagrees with its database")
        }
        check(embFile.exists()) { "Active V2 embedding file is missing" }

        onStatus("Memory-mapping ${activeGeneration.manifest.trackCount} CLaMP3 embeddings")
        val mapped = getOrOpenRankIndex(activeGeneration)
        check(mapped.numTracks == databaseCount) {
            "Embedding index changed while loading; retry search"
        }
        textIndex = mapped
        val snapshot = TextEmbeddingIndexSnapshot.capture(
            generation = TextEmbeddingIndexGeneration.current(),
            trackCount = mapped.numTracks,
            file = embFile,
        )
        textIndexSnapshot = snapshot
        Log.i(
            "MainViewModel",
            "Text embedding index mapped: ${mapped.numTracks} tracks, " +
                "generation=${snapshot.generation}",
        )
        return bindFindMusicLibrarySnapshot(
            filesDir = filesDir,
            activeGeneration = activeGeneration,
            embeddingIndex = mapped,
            textAssets = textAssets,
            lookupStartedNs = startedNs,
            onStatus = onStatus,
        )
    }

    private fun bindFindMusicLibrarySnapshot(
        filesDir: File,
        activeGeneration: V2ResolvedActiveIndexGeneration,
        embeddingIndex: EmbeddingIndex,
        textAssets: TextRetrievalAssets,
        lookupStartedNs: Long,
        onStatus: (String) -> Unit,
    ): FindMusicLibrarySnapshot {
        val handedOffCatalog = takeVerifiedFindMusicCatalogHandoff(activeGeneration)
        val processCatalog = if (handedOffCatalog == null) {
            RadioService.processActiveLibraryCatalog(activeGeneration.toRadioGenerationToken())
        } else {
            null
        }
        val durableCatalog = if (handedOffCatalog == null && processCatalog == null) {
            V2ActiveLibraryCatalogStore(filesDir).read(activeGeneration)
        } else {
            null
        }
        val cachedCatalog = handedOffCatalog ?: processCatalog ?: durableCatalog
        var providerRows = -1
        var providerQueryMs = -1L
        var providerAssemblyMs = -1L
        var providerMs = 0L
        var activeCatalogMs = 0L
        val activeCatalogSource: String
        val activeCatalog = if (cachedCatalog != null) {
            activeCatalogSource = when {
                handedOffCatalog != null -> "database_refresh"
                processCatalog != null -> "process_cache"
                else -> "durable_cache"
            }
            cachedCatalog
        } else {
            activeCatalogSource = "provider_scan"
            val providerStartedNs = System.nanoTime()
            val providerSnapshot = V2PowerampProviderSnapshotAcquirer(
                getApplication<Application>(),
            ).acquireBlocking { completedRows, totalRows ->
                onStatus(powerampLibraryReadProgressText(completedRows, totalRows))
            }
            providerMs = elapsedMs(providerStartedNs)
            providerSnapshot.acquisitionEvidence?.let { evidence ->
                providerRows = evidence.rowCount
                providerQueryMs = evidence.queryAndCursorReadMs ?: -1L
                providerAssemblyMs = evidence.snapshotAssemblyMs ?: -1L
            }
            val activeCatalogStartedNs = System.nanoTime()
            V2ActiveLibraryCatalogLoader.load(
                activeGeneration = activeGeneration,
                providerSnapshot = providerSnapshot,
                onProgress = { progress ->
                    onStatus(activeLibraryCatalogProgressText(progress))
                },
            ).also {
                activeCatalogMs = elapsedMs(activeCatalogStartedNs)
                V2ActiveLibraryCatalogStore(filesDir).write(
                    activeGeneration = activeGeneration,
                    catalog = it,
                )
            }
        }
        RadioService.publishActiveLibrarySnapshot(
            activeGeneration.toRadioGenerationToken(),
            activeCatalog,
        )
        if (_databaseInfo.value?.generationId == activeGeneration.manifest.generationId &&
            _databaseInfo.value?.providerGenerationId !=
            activeCatalog.generationBinding.providerGenerationId
        ) {
            publishDatabaseInfo(
                readDatabaseInfo(
                    resolution = V2ResolvedLibraryDatabase(
                        databaseFile = activeGeneration.databaseFile,
                        activeGeneration = activeGeneration,
                    ),
                    activeCatalogOverride = activeCatalog,
                    reconcileProviderIfMissing = false,
                ),
            )
        }
        val identityStartedNs = System.nanoTime()
        onStatus("Loading stable recording identities for ${embeddingIndex.numTracks} embeddings")
        val database = EmbeddingDatabase.open(activeGeneration.databaseFile)
        val identityCatalog = try {
            StableTrackIdentityCatalog.load(filesDir, database, embeddingIndex)
        } finally {
            database.close()
        }
        val identityMs = elapsedMs(identityStartedNs)
        val activeDomainStartedNs = System.nanoTime()
        onStatus("Building the active recommendation domain from the current Poweramp library")
        val activeDomain = ActiveRecommendationDomain.create(
            activeCatalog = activeCatalog,
            identityCatalog = identityCatalog,
            embeddingIndex = embeddingIndex,
        )
        val activeDomainMs = elapsedMs(activeDomainStartedNs)
        require(activeDomain.activeTrackCount > 0) {
            "The current Poweramp library has no active indexed tracks"
        }
        return FindMusicLibrarySnapshot(
            databaseFile = activeGeneration.databaseFile,
            embeddingFile = activeGeneration.embeddingFile,
            graphFile = activeGeneration.graphFile,
            embeddingIndex = embeddingIndex,
            activationBindingId = activeGeneration.manifest.activationBindingId,
            identityCatalog = identityCatalog,
            activeCatalog = activeCatalog,
            activeDomain = activeDomain,
            textAssets = textAssets,
            generation = activeGeneration.toRadioGenerationToken(),
        ).also { snapshot ->
            cacheTextLibrarySnapshot(snapshot)
            Log.i(
                "FindMusicLibrary",
                "library cache=miss tracks=${embeddingIndex.numTracks} " +
                    "active=${activeDomain.activeTrackCount} catalogSource=$activeCatalogSource " +
                    "providerRows=$providerRows providerMs=$providerMs " +
                    "providerQueryMs=$providerQueryMs providerAssemblyMs=$providerAssemblyMs " +
                    "catalogMs=$activeCatalogMs identityMs=$identityMs " +
                    "domainMs=$activeDomainMs totalMs=${elapsedMs(lookupStartedNs)}",
            )
        }
    }

    private fun FindMusicLibrarySnapshot.matchesStaticLibrary(
        activeGeneration: V2ResolvedActiveIndexGeneration,
        textAssets: TextRetrievalAssets,
    ): Boolean =
        generation == activeGeneration.toRadioGenerationToken() &&
            activationBindingId == activeGeneration.manifest.activationBindingId &&
            databaseFile.canonicalFile == activeGeneration.databaseFile.canonicalFile &&
            embeddingFile.canonicalFile == activeGeneration.embeddingFile.canonicalFile &&
            this.textAssets.identity == textAssets.identity

    private fun V2ResolvedActiveIndexGeneration.toRadioGenerationToken(): RadioGenerationToken {
        val source = manifest
        return RadioGenerationToken(
            generationId = source.generationId,
            activationBindingId = source.activationBindingId,
            manifestSha256 = manifestSha256,
            embeddingSpecId = source.receiptEmbeddingSpec.specId,
            databaseContentSha256 = source.databaseContentSha256,
            orderedTrackSetSha256 = source.orderedTrackSetSha256,
            stableTrackUidMappingSha256 = source.stableTrackUidCoverage.mappingSha256,
        )
    }

    private fun resolveTextRetrievalAssets(
        filesDir: File,
        expectedSpec: TextRetrievalSpecFingerprint?,
        onStatus: (String) -> Unit = {},
    ): TextRetrievalAssets {
        val exactModelFile = File(filesDir, "clamp3_text.tflite")
        val modelCandidates = if (expectedSpec != null) {
            listOf(exactModelFile).filter(File::isFile)
        } else {
            listOf(
                exactModelFile,
                File(filesDir, "clamp3_text_fp32.tflite"),
                File(filesDir, "clamp3_text_fp16.tflite"),
            ).filter(File::isFile)
        }
        val tokenizerModelFile = File(filesDir, "sentencepiece.bpe.model")
        require(modelCandidates.isNotEmpty()) { "CLaMP3 text model not found" }
        require(tokenizerModelFile.isFile) { "SentencePiece tokenizer model not found" }

        if (expectedSpec != null) {
            require(
                expectedSpec.tokenizerPolicyId ==
                    V2IndexingWorkPolicy.TEXT_TOKENIZER_POLICY_ID &&
                    expectedSpec.tokenizerRuntimeContractSha256 ==
                    V2IndexingWorkPolicy.TEXT_TOKENIZER_RUNTIME_CONTRACT_SHA256
            ) {
                "Active V2 generation uses an unsupported SentencePiece runtime contract"
            }
        }

        val installedModelIdentity = if (expectedSpec != null) {
            V2CurrentModelPolicyResolver.resolveInstalled(filesDir) { progress ->
                onStatus(
                    exactHashProgressText(
                        subject = "changed indexing file ${progress.fileOrdinal} of " +
                            "${progress.fileCount}: ${progress.filename}",
                        completedBytes = progress.completedBytes,
                        totalBytes = progress.totalBytes,
                    ),
                )
            }
        } else {
            null
        }

        fun hashAsset(file: File, purpose: String): String =
            installedModelIdentity?.sha256ByName?.get(file.name) ?: textAssetSha256(file) {
                    completedBytes,
                    totalBytes,
                ->
                onStatus(exactHashProgressText("$purpose ${file.name}", completedBytes, totalBytes))
            }

        var selectedModelHash: String? = null
        val modelFile = if (expectedSpec == null) {
            modelCandidates.first()
        } else {
            modelCandidates.firstOrNull {
                val hash = hashAsset(it, "text-model file")
                (hash == expectedSpec.textModelSha256).also { matches ->
                    if (matches) selectedModelHash = hash
                }
            }
                ?: error("No installed CLaMP3 text model matches the active generation")
        }
        val modelSha256 = selectedModelHash ?: hashAsset(modelFile, "text-model file")
        val tokenizerModelSha256 = hashAsset(tokenizerModelFile, "tokenizer file")
        if (expectedSpec != null) {
            require(tokenizerModelSha256 == expectedSpec.tokenizerModelSha256) {
                "Installed SentencePiece model does not match the active generation"
            }
        }
        return TextRetrievalAssets(
            modelFile = modelFile,
            tokenizerModelFile = tokenizerModelFile,
            modelSha256 = modelSha256,
            tokenizerModelSha256 = tokenizerModelSha256,
            identity = expectedSpec?.specId ?: listOf(
                OfficialSentencePieceTokenizer.CONTRACT_SHA256,
                modelSha256,
                tokenizerModelSha256,
            ).joinToString(":"),
        )
    }

    @Synchronized
    private fun textAssetSha256(
        file: File,
        onProgress: (completedBytes: Long, totalBytes: Long) -> Unit = { _, _ -> },
    ): String {
        val canonical = file.canonicalFile
        val cached = textAssetHashes[canonical.path]
        if (cached != null && cached.length == canonical.length() &&
            cached.modifiedMs == canonical.lastModified()
        ) {
            onProgress(canonical.length(), canonical.length())
            return cached.sha256
        }
        return V2FileSha256.digest(canonical, onProgress).also { sha256 ->
            textAssetHashes[canonical.path] = CachedAssetHash(
                length = canonical.length(),
                modifiedMs = canonical.lastModified(),
                sha256 = sha256,
            )
        }
    }

    /** Peek is read-only; use the already-verified catalog and recheck only cheap bindings. */
    private fun isCurrentPreviewLibrary(
        filesDir: File,
        snapshot: FindMusicLibrarySnapshot,
    ): Boolean {
        val active = runCatching { V2IndexGenerationReader.requireActive(filesDir) }.getOrNull()
            ?: return false
        if (!snapshot.matchesActiveGeneration(active)) return false
        val published = _databaseInfo.value ?: return false
        return published.generationId == active.manifest.generationId &&
            published.providerGenerationId == snapshot.activeDomain.binding.providerGenerationId
    }

    /**
     * Read-only Find Music results are coherent against their immutable, verified snapshot.
     * Queue delivery separately revalidates every selected Poweramp row immediately before the
     * mutation, so publishing a result must not rescan and hash the complete provider library.
     */
    private fun isCurrentReadOnlyFindMusicLibrary(
        filesDir: File,
        snapshot: FindMusicLibrarySnapshot,
        phase: String,
    ): Boolean {
        val startedNs = System.nanoTime()
        val current = isCurrentPreviewLibrary(filesDir, snapshot)
        if (!current) invalidateTextLibrarySnapshot(snapshot)
        val reason = if (current) "immutable_snapshot_binding" else "static_binding"
        Log.i(
            "FindMusicLibrary",
            "freshness phase=$phase current=$current reason=$reason " +
                "validation=read_only totalMs=${elapsedMs(startedNs)}",
        )
        return current
    }

    private fun FindMusicLibrarySnapshot.matchesActiveGeneration(
        activeGeneration: V2ResolvedActiveIndexGeneration,
    ): Boolean =
        generation == activeGeneration.toRadioGenerationToken() &&
            activationBindingId == activeGeneration.manifest.activationBindingId &&
            identityCatalog.binding.activationBindingId ==
            activeGeneration.manifest.activationBindingId &&
            activeDomain.binding.databaseGenerationId == activeGeneration.manifest.generationId &&
            databaseFile.canonicalFile == activeGeneration.databaseFile.canonicalFile &&
            embeddingFile.canonicalFile == activeGeneration.embeddingFile.canonicalFile

    private fun elapsedMs(startedNs: Long): Long =
        (System.nanoTime() - startedNs) / 1_000_000L

    private fun noCandidateRecordingsMessage(days: Int?): String =
        if (days == null) {
            "No indexed recordings are available."
        } else {
            "No indexed recordings were added to Poweramp in the " +
                "${libraryAddedDaysLabel(days).lowercase()}."
        }

    @Synchronized
    private fun invalidateTextLibrarySnapshot(snapshot: FindMusicLibrarySnapshot? = null) {
        synchronized(TEXT_LIBRARY_CACHE_LOCK) {
            if (snapshot == null || processTextLibrarySnapshot === snapshot) {
                processTextLibrarySnapshot = null
            }
        }
    }

    private fun currentProcessTextLibrarySnapshot(): FindMusicLibrarySnapshot? =
        synchronized(TEXT_LIBRARY_CACHE_LOCK) { processTextLibrarySnapshot }

    private fun cacheTextLibrarySnapshot(snapshot: FindMusicLibrarySnapshot) {
        synchronized(TEXT_LIBRARY_CACHE_LOCK) {
            processTextLibrarySnapshot = snapshot
        }
    }

    private fun publishVerifiedFindMusicCatalogHandoff(next: VerifiedDatabaseInfo?) {
        val alreadyCached = currentProcessTextLibrarySnapshot() != null
        val runtime = Runtime.getRuntime()
        val usedHeapBytes = runtime.totalMemory() - runtime.freeMemory()
        val retainCatalog = FindMusicCatalogHandoffPolicy.shouldRetain(
            maxHeapBytes = runtime.maxMemory(),
            usedHeapBytes = usedHeapBytes,
            processSnapshotPresent = alreadyCached,
        )
        synchronized(findMusicCatalogHandoffLock) {
            verifiedFindMusicCatalogHandoff = if (retainCatalog) {
                val catalog = next?.activeCatalog
                val generation = next?.replayLibraryBinding?.generation
                if (catalog != null && generation != null &&
                    catalog.generationBinding.databaseGenerationId == generation.generationId
                ) {
                    VerifiedFindMusicCatalogHandoff(generation, catalog)
                } else {
                    null
                }
            } else {
                null
            }
        }
        Log.i(
            "FindMusicLibrary",
            "database catalog handoff retained=" +
                "${synchronized(findMusicCatalogHandoffLock) {
                    verifiedFindMusicCatalogHandoff != null
                }} usedHeapMiB=${usedHeapBytes / BYTES_PER_MIB} " +
                "maxHeapMiB=${runtime.maxMemory() / BYTES_PER_MIB}",
        )
    }

    private fun takeVerifiedFindMusicCatalogHandoff(
        activeGeneration: V2ResolvedActiveIndexGeneration,
    ): V2ActiveLibraryCatalog? = synchronized(findMusicCatalogHandoffLock) {
        val handoff = verifiedFindMusicCatalogHandoff
        verifiedFindMusicCatalogHandoff = null
        handoff?.takeIf { candidate ->
            candidate.generation == activeGeneration.toRadioGenerationToken() &&
                candidate.activeCatalog.generationBinding.databaseGenerationId ==
                    activeGeneration.manifest.generationId
        }?.activeCatalog
    }

    private fun clearVerifiedFindMusicCatalogHandoff() {
        synchronized(findMusicCatalogHandoffLock) {
            verifiedFindMusicCatalogHandoff = null
        }
    }

    @Synchronized
    private fun invalidateTextEmbeddingIndex() {
        textIndex = null
        textIndexSnapshot = null
        invalidateTextLibrarySnapshot()
        TextEmbeddingIndexGeneration.invalidate()
    }

    /**
     * Search for the best matching track by text query using CLaMP3 text embeddings.
     */
    fun performTextSearch(query: String) {
        val normalized = query.trim()
        if (normalized.isBlank()) return
        val spec = FindMusicQuerySpec(
            operator = _findMusicOperator.value,
            textIngredients = listOf(
                FindMusicTextIngredient(
                    query = normalized,
                    weight = 1f,
                    negative = false,
                ),
            ),
            resultLimit = _numTracks.value,
            textResultPlanner = _findMusicTextResultPlanner.value,
            libraryAddedDays = _libraryAddedDays.value,
        )
        launchFindMusicRequest(FindMusicSurface.TEXT) { revision ->
            executeTextSearch(revision, FindMusicSurface.TEXT, spec)
        }
    }

    private suspend fun executeTextSearch(
        revision: Long,
        surface: FindMusicSurface,
        spec: FindMusicQuerySpec,
    ) {
        val textQuery = spec.activeTextIngredients.singleOrNull()
            ?.takeUnless(FindMusicTextIngredient::negative)
            ?.query
        if (textQuery == null || spec.songSeeds.any { it.weight > 0f }) {
            Log.w(
                "MainViewModel",
                "Rejected raw cosine route because the request was not one positive text ingredient",
            )
            publishFindMusicError(
                revision,
                surface,
                spec.displayLabel,
                "Closest and Varied need one positive description.",
                spec,
            )
            return
        }
        val requestStartedNs = System.nanoTime()
        val requestReferenceEpochSecond = System.currentTimeMillis() / 1_000L
        logMemoryPhase("find_music_text_${spec.textResultPlanner.wireName}_start")
        validateQueryContract(spec)?.let { message ->
            publishFindMusicError(revision, surface, textQuery, message, spec)
            return
        }
        val filesDir = getApplication<Application>().filesDir
        val libraryStartedNs = System.nanoTime()
        val library = try {
            getOrCreateTextLibrarySnapshot(filesDir) { status ->
                publishFindMusicLoadingStatus(revision, status)
            }
        } catch (e: Exception) {
            Log.e("MainViewModel", "Find Music could not prepare the active library", e)
            publishFindMusicError(
                revision,
                surface,
                textQuery,
                "Find Music could not verify the active library. " +
                    "Check the music index and indexing files in Settings, then try again.",
                spec,
            )
            return
        }
        val libraryElapsedMs = elapsedMs(libraryStartedNs)
        val dbFile = library.databaseFile
        val index = library.embeddingIndex
        val activeDomain = library.activeDomain
        val candidateDomain = activeDomain.candidateDomain(
            minimumLibraryAddedAtEpochSecond(
                spec.effectiveLibraryAddedDays,
                requestReferenceEpochSecond,
            ),
        )
        if (candidateDomain.candidateIdentityCount == 0) {
            publishFindMusicError(
                revision,
                surface,
                textQuery,
                noCandidateRecordingsMessage(spec.effectiveLibraryAddedDays),
                spec,
            )
            return
        }

        val embeddingStartedNs = System.nanoTime()
        val embedding = try {
            generateTextEmbeddingWithRetry(
                revision,
                textQuery,
                filesDir,
                library.textAssets,
            ) { status -> publishFindMusicLoadingStatus(revision, status) }
        } catch (e: CancellationException) {
            throw e
        } catch (e: IllegalStateException) {
            Log.e("MainViewModel", "Find Music text inference failed", e)
            publishFindMusicError(
                revision,
                surface,
                textQuery,
                "Find Music could not use the text model. " +
                    "Check the music index and indexing files in Settings, then try again.",
                spec,
            )
            return
        }
        val embeddingElapsedMs = elapsedMs(embeddingStartedNs)
        ensureCurrentFindMusicRequest(revision)

        val db = EmbeddingDatabase.open(dbFile)
        val catalog: StableTrackIdentityCatalog
        val boundSpec: FindMusicQuerySpec
        val reduction: StableVisibleReduction<Triple<Long, Float, Int>>
        val matchedTracks: List<TextSearchMatch>
        var textQueuePlan: FindMusicTextQueuePlanEvidence? = null
        var similarityElapsedMs = 0L
        var rankingElapsedMs = 0L
        var planningElapsedMs = 0L
        var materializationElapsedMs = 0L
        try {
            catalog = library.identityCatalog
            boundSpec = spec.copy(
                schemaVersion = FindMusicQuerySpec.CURRENT_SCHEMA_VERSION,
                libraryBinding = catalog.binding,
            )
            val similarityStartedNs = System.nanoTime()
            publishFindMusicLoadingStatus(
                revision,
                "Comparing the description with ${index.numTracks} embeddings; " +
                    "${candidateDomain.candidateIdentityCount} candidate identities are eligible",
            )
            val fullSimilarities = index.computeAllSimilarities(embedding)
            val activeTrackIds = candidateDomain.orderedIdentityRepresentativeTrackIds()
            val activeSimilarities =
                candidateDomain.identityRepresentativeScoresFromFull(fullSimilarities)
            similarityElapsedMs = (System.nanoTime() - similarityStartedNs) / 1_000_000L
            ensureCurrentFindMusicRequest(revision)
            val requestJob = currentCoroutineContext()[Job]
            val cancellationCheck = {
                requestJob?.ensureActive()
                if (!findMusicRequestGate.isCurrent(revision)) {
                    throw CancellationException(
                        "Find Music request $revision is no longer current",
                    )
                }
            }
            val rankingStartedNs = System.nanoTime()
            publishFindMusicLoadingStatus(
                revision,
                when (spec.textResultPlanner) {
                    FindMusicTextResultPlanner.CLOSEST ->
                        "Selecting the ${spec.resultLimit} closest distinct recordings"
                    FindMusicTextResultPlanner.VARIED_DPP ->
                        "Planning ${spec.resultLimit} varied results from the complete selected " +
                            "candidate domain"
                    FindMusicTextResultPlanner.VARIED_ALL_OF_DPP ->
                        error("All-of Varied cannot use the simple text route")
                },
            )
            val rankedRows = when (spec.textResultPlanner) {
                FindMusicTextResultPlanner.CLOSEST -> {
                    val rows = StableSimilarityTopK.select(
                        orderedTrackIds = activeTrackIds,
                        similarities = activeSimilarities,
                        topK = candidateDomain.rankedRowsForVisibleCount(spec.resultLimit),
                        rankingTieKey = catalog::rankingTieKey,
                        cancellationCheck = cancellationCheck,
                    ).mapIndexed { rawRank, row ->
                        Triple(row.trackId, row.score, rawRank + 1)
                    }
                    rankingElapsedMs = (System.nanoTime() - rankingStartedNs) / 1_000_000L
                    rows
                }
                FindMusicTextResultPlanner.VARIED_DPP -> {
                    val completeRanking = StableSimilarityTopK.select(
                        orderedTrackIds = activeTrackIds,
                        similarities = activeSimilarities,
                        topK = activeTrackIds.size,
                        rankingTieKey = catalog::rankingTieKey,
                        cancellationCheck = cancellationCheck,
                    )
                    rankingElapsedMs = (System.nanoTime() - rankingStartedNs) / 1_000_000L
                    logMemoryPhase("find_music_text_varied_dpp_complete_ranking")
                    val planningStartedNs = System.nanoTime()
                    val plan = FindMusicTextQueuePlanner.plan(
                        planner = spec.textResultPlanner,
                        completeRelevanceRanking = completeRanking,
                        requestedResultCount = spec.resultLimit,
                        embeddingIndex = index,
                        cancellationCheck = cancellationCheck,
                    )
                    planningElapsedMs = (System.nanoTime() - planningStartedNs) / 1_000_000L
                    textQueuePlan = plan.evidence
                    logMemoryPhase("find_music_text_varied_dpp_planned")
                    plan.selections.map { selected ->
                        Triple(
                            selected.trackId,
                            selected.textSimilarity,
                            selected.originalTextObjectiveRank,
                        )
                    }
                }
                FindMusicTextResultPlanner.VARIED_ALL_OF_DPP ->
                    error("All-of Varied cannot use the simple text route")
            }
            val materializationStartedNs = System.nanoTime()
            publishFindMusicLoadingStatus(
                revision,
                "Reading display metadata for ${spec.resultLimit} selected recordings",
            )
            val tracksById = HashMap<Long, EmbeddedTrack?>()
            reduction = StableVisibleResultReducer.reduce(
                rankedItems = rankedRows,
                requestedVisibleCount = spec.resultLimit,
                identityOf = { (trackId, _, _) -> catalog.visibleResultIdentity(trackId) },
                isEligible = { (trackId, _, _) ->
                    tracksById.getOrPut(trackId) { db.getTrackById(trackId) } != null
                },
            )
            textQueuePlan?.let { plan ->
                require(
                    reduction.items.map { (trackId) -> trackId } ==
                        plan.orderedSelectedTrackIds &&
                        reduction.collapsedEquivalentCount == 0,
                ) {
                    "Varied text plan did not survive exact-copy and library-row verification " +
                        "unchanged; refusing to rewrite its complete-domain proof"
                }
            }
            matchedTracks = reduction.items.mapNotNull { (trackId, score, rawRank) ->
                (tracksById[trackId] ?: db.getTrackById(trackId))?.let { track ->
                    TextSearchMatch(
                        track = track,
                        similarity = score,
                        identity = RadioSeedIdentity(
                            embeddedTrackId = track.id,
                            stableTrackSpanId = catalog.stableTrackSpanId(track.id),
                        ),
                        objectiveRank = rawRank,
                    )
                }
            }
            require(matchedTracks.size == reduction.items.size) {
                "Verified Find Music rows changed while their display records were materialized"
            }
            materializationElapsedMs = elapsedMs(materializationStartedNs)
        } finally {
            db.close()
        }
        ensureCurrentFindMusicRequest(revision)
        if (matchedTracks.isEmpty()) {
            publishFindMusicError(revision, surface, textQuery, "No matches found", spec)
            return
        }

        val freshnessStartedNs = System.nanoTime()
        publishFindMusicLoadingStatus(
            revision,
            "Confirming the active music index did not change during this search",
        )
        if (!isCurrentReadOnlyFindMusicLibrary(
                filesDir,
                library,
                phase = "text_before_publish",
            )
        ) {
            publishFindMusicError(
                revision,
                surface,
                textQuery,
                "The active music library changed while this search was running. Run it again.",
                spec,
            )
            return
        }
        val freshnessElapsedMs = elapsedMs(freshnessStartedNs)
        ensureCurrentFindMusicRequest(revision)

        val publicationStartedNs = System.nanoTime()
        publishFindMusicResult(
            revision,
            surface,
            TextSearchResult(
                query = textQuery,
                matches = matchedTracks,
                querySpec = boundSpec,
                libraryBinding = catalog.binding,
                providerGenerationId = activeDomain.binding.providerGenerationId,
                orderedActiveTrackIdsSha256 = activeDomain.orderedActiveTrackIdsSha256,
                activeTrackCount = activeDomain.activeTrackCount,
                objectiveRankingDomainCount = candidateDomain.candidateIdentityCount,
                stableResultReduction = reduction.toStableResultReductionEvidence(),
                textQueuePlan = textQueuePlan,
            ),
        )
        saveRecentSearchIfCurrent(revision, boundSpec.copy(libraryBinding = null))
        val publicationElapsedMs = elapsedMs(publicationStartedNs)
        val proof = textQueuePlan?.dppSelection
        val totalElapsedMs = elapsedMs(requestStartedNs)
        val accountedElapsedMs = libraryElapsedMs + embeddingElapsedMs +
            similarityElapsedMs + rankingElapsedMs + planningElapsedMs +
            materializationElapsedMs + freshnessElapsedMs + publicationElapsedMs
        Log.i(
            "FindMusicPlanner",
                "query=${JSONObject.quote(textQuery)} " +
                "planner=${boundSpec.textResultPlanner.wireName} " +
                "addedDays=${boundSpec.effectiveLibraryAddedDays ?: "all"} " +
                "domain=${candidateDomain.candidateIdentityCount} " +
                "activeRows=${activeDomain.activeTrackCount} requested=${boundSpec.resultLimit} " +
                "displayed=${matchedTracks.size} libraryMs=$libraryElapsedMs " +
                "embeddingMs=$embeddingElapsedMs similarityMs=$similarityElapsedMs " +
                "rankingMs=$rankingElapsedMs planningMs=$planningElapsedMs " +
                "materializationMs=$materializationElapsedMs " +
                "freshnessMs=$freshnessElapsedMs publicationMs=$publicationElapsedMs " +
                "overheadMs=${(totalElapsedMs - accountedElapsedMs).coerceAtLeast(0L)} " +
                "totalMs=$totalElapsedMs " +
                "workingCandidates=${proof?.finalWorkingCandidateCount ?: 0} " +
                "attempts=${proof?.attemptedCandidateCounts?.joinToString(",") ?: "none"} " +
                "collapsedCopies=${reduction.collapsedEquivalentCount}",
        )
        logMemoryPhase("find_music_text_${boundSpec.textResultPlanner.wireName}_published")
    }

    /** Serialize model creation, inference, close and retry across every caller. */
    private suspend fun generateTextEmbeddingWithRetry(
        revision: Long?,
        query: String,
        filesDir: File,
        textAssets: TextRetrievalAssets,
        onStatus: (String) -> Unit = {},
    ): FloatArray = textInferenceMutex.withLock {
        check(!recommendationResourcesBlocked()) {
            "Text search is paused while on-device indexing is open"
        }
        if (revision != null) ensureCurrentFindMusicRequest(revision)
        val runtimeIdentity = textAssets.runtimeIdentity()
        val cacheKey = ExactTextEmbeddingCacheKey(
            query = query,
            runtimeIdentity = runtimeIdentity,
        )
        val debugDir = File(filesDir, "debug_embeddings")
        if (textInferenceIdentity != runtimeIdentity) {
            try {
                textInference?.close()
            } catch (e: Exception) {
                Log.w("MainViewModel", "Failed to close replaced text inference", e)
            }
            textInference = null
            textInferenceIdentity = null
            textEmbeddingCache.clear()
        }
        textEmbeddingCache.get(cacheKey)?.let { cached ->
            onStatus("Using the exact cached text embedding for this description")
            if (revision != null) ensureCurrentFindMusicRequest(revision)
            Log.i("FindMusicPlanner", "textEmbeddingCache=hit")
            return@withLock cached
        }
        val inferenceStartedNs = System.nanoTime()
        repeat(2) { attempt ->
            currentCoroutineContext().ensureActive()
            val inference = textInference ?: run {
                onStatus("Initializing the CLaMP3 text model on CPU")
                createTextInference(textAssets).also {
                    textInference = it
                    textInferenceIdentity = runtimeIdentity
                }
            }
            onStatus(
                if (attempt == 0) {
                    "Encoding the description with the CLaMP3 text model"
                } else {
                    "Retrying description encoding with a fresh CLaMP3 text-model runtime"
                },
            )
            val embedding = inference.generateEmbedding(query, debugDir)
            if (embedding != null) {
                // Cache model output before the request-specific staleness check. LiteRT inference
                // is not interruptible, so a superseding identical request can reuse this result.
                textEmbeddingCache.put(cacheKey, embedding)
                Log.i(
                    "FindMusicPlanner",
                    "textEmbeddingCache=miss inferenceMs=" +
                        (System.nanoTime() - inferenceStartedNs) / 1_000_000L,
                )
                if (revision != null) ensureCurrentFindMusicRequest(revision)
                return@withLock embedding
            }

            try { inference.close() } catch (_: Exception) {}
            if (textInference === inference) {
                textInference = null
                textInferenceIdentity = null
            }
            if (attempt == 0) {
                Log.w("MainViewModel", "Text inference returned no embedding; retrying once")
            }
        }
        throw IllegalStateException("Text inference failed after one fresh-model retry")
    }

    private fun createTextInference(textAssets: TextRetrievalAssets): Clamp3TextInference {
        return try {
            Clamp3TextInference(
                textAssets.modelFile,
                textAssets.tokenizerModelFile,
                accelerator = Accelerator.CPU,
                strictAccelerator = true,
            ).also {
                check(it.activeAccelerator == Accelerator.CPU)
                Log.i(
                    "MainViewModel",
                    "Text model loaded: ${textAssets.modelFile.name} on pinned CPU backend",
                )
            }
        } catch (e: Exception) {
            throw IllegalStateException(
                "Pinned CPU text backend failed for ${textAssets.modelFile.name}: ${e.message}",
                e,
            )
        }
    }

    /**
     * Queue a specific search result list directly into Poweramp.
     * The caller passes the exact result being displayed — no ambiguity.
     */
    fun queueDisplayedResults(
        result: TextSearchResult,
        placement: DirectQueuePlacement,
    ): Boolean {
        val tracks = result.matches.map { it.track }
        val identities = result.matches.map { it.identity }
        val binding = result.libraryBinding
        val querySpec = result.querySpec
        val reduction = result.stableResultReduction
        val membershipPlanMatchesDisplay = when {
            querySpec == null -> false
            querySpec.isSimplePositiveTextOnly &&
                querySpec.textResultPlanner == FindMusicTextResultPlanner.CLOSEST ->
                result.textQueuePlan == null
            querySpec.isSimplePositiveTextOnly -> result.textQueuePlan?.let { plan ->
                plan.orderedSelectedTrackIds == tracks.map { it.id } &&
                    plan.orderedOriginalTextObjectiveRanks ==
                    result.matches.mapNotNull { it.objectiveRank }
            } == true
            querySpec.textResultPlanner ==
                FindMusicTextResultPlanner.VARIED_ALL_OF_DPP ->
                result.allOfQueuePlan?.let { plan ->
                    plan.orderedSelectedTrackIds == tracks.map { it.id } &&
                        plan.orderedOriginalAllOfObjectiveRanks ==
                        result.matches.mapNotNull { it.objectiveRank }
                } == true
            else -> result.textQueuePlan == null && result.allOfQueuePlan == null
        }
        val eligibility = _displayedQueueEligibility.value
        if (currentDisplayedFindMusicResult() !== result || !eligibility.eligible ||
            tracks.isEmpty() || binding == null ||
            querySpec?.libraryBinding != binding ||
            reduction == null ||
            !membershipPlanMatchesDisplay ||
            !result.hasExactActiveDomainBinding() ||
            result.objectiveRankingDomainCount == null ||
            (!querySpec.isSimplePositiveTextOnly && result.ingredientRankingDomainCount == null) ||
            result.matches.any { it.objectiveRank == null }
        ) {
            Log.w(
                "MainViewModel",
                "Refusing a displayed queue without complete query, domain, reduction and rank evidence",
            )
            return false
        }
        val sessionEvidence = FindMusicSessionEvidence(
            querySpec = querySpec,
            orderedActiveTrackIdsSha256 = checkNotNull(result.orderedActiveTrackIdsSha256),
            activeTrackCount = checkNotNull(result.activeTrackCount),
            objectiveRankingDomainCount = checkNotNull(result.objectiveRankingDomainCount),
            ingredientRankingDomainCount = result.ingredientRankingDomainCount,
            stableResultReduction = reduction,
            textQueuePlan = result.textQueuePlan,
            allOfQueuePlan = result.allOfQueuePlan,
        )
        val trackEvidence = result.matches.mapIndexed { index, match ->
            FindMusicTrackEvidence(
                displayedRank = index + 1,
                objectiveRank = checkNotNull(match.objectiveRank),
                resultScore = match.similarity,
                rankingScore = match.rankingScore,
                ingredientPercentiles = match.anchorPercentiles.toList(),
            )
        }
        val origin = when (result.kind) {
            FindMusicResultKind.COMPOSED -> QueueOrigin.COMPOSED_RESULT_LIST
            FindMusicResultKind.TEXT -> QueueOrigin.TEXT_RESULT_LIST
        }
        Log.i(
            "MainViewModel",
            "QUEUE_DISPLAYED: requesting admission for ${tracks.size} displayed results " +
                "for '${result.query}'",
        )
        for ((i, t) in tracks.withIndex()) {
            Log.d(
                "MainViewModel",
                "QUEUE_DISPLAYED: [$i] ${t.artist} - ${t.title} (trackId=${t.id})",
            )
        }
        val admission = RadioService.queueDirectly(
            context = getApplication(),
            tracks = tracks,
            trackIdentities = identities,
            displayedBinding = binding,
            displayedProviderGenerationId = checkNotNull(result.providerGenerationId),
            displayedOrderedActiveTrackIdsSha256 =
                checkNotNull(result.orderedActiveTrackIdsSha256),
            displayedActiveTrackCount = checkNotNull(result.activeTrackCount),
            label = result.query,
            origin = origin,
            placement = placement,
            findMusicSessionEvidence = sessionEvidence,
            findMusicTrackEvidence = trackEvidence,
        )
        return when (admission) {
            RadioRequestAdmission.ACCEPTED -> {
                Log.i("MainViewModel", "QUEUE_DISPLAYED: request admitted")
                true
            }
            RadioRequestAdmission.BUSY -> {
                Log.w("MainViewModel", "QUEUE_DISPLAYED: request not admitted; radio is busy")
                false
            }
        }
    }

    fun clearFindMusicResults() {
        cancelFindMusicRequest()
        _textSearchResult.value = null
        _multiSeedResult.value = null
        clearDisplayedQueueEligibility()
    }

    private fun trackRetrievalJobUntilCompletion(job: Job) {
        job.invokeOnCompletion {
            synchronized(RETRIEVAL_RESOURCE_ADMISSION_LOCK) {
                activeRetrievalResourceJobs.remove(job)
            }
        }
    }

    private data class RetrievalAdmissionTransition(
        val becameAvailable: Boolean = false,
        val shouldQuiesce: Boolean = false,
    )

    private fun reconcileIndexingResourceOwnership(
        durableAttachCompletedNow: Boolean = false,
    ): RetrievalAdmissionTransition = synchronized(RETRIEVAL_RESOURCE_ADMISSION_LOCK) {
        if (durableAttachCompletedNow) durableIndexingAttachCompleted = true
        val previousOwnership = indexingResourceOwnership
        val wasBlocked = retrievalResourcesSuspended.get()
        val nextOwnership = indexingResourceOwnershipFor(
            state = IndexingService.state.value,
            durableAttachCompleted = durableIndexingAttachCompleted,
            filesDir = getApplication<Application>().filesDir,
        )
        indexingResourceOwnership = nextOwnership
        if (nextOwnership != IndexingResourceOwnership.FREE &&
            previousOwnership != nextOwnership &&
            indexingActivityHandoffOwners.isEmpty()
        ) {
            retrievalResourceReleaseRequired = true
        }
        updateRetrievalResourceAdmissionLocked()
        val isBlocked = retrievalResourcesSuspended.get()
        RetrievalAdmissionTransition(
            becameAvailable = wasBlocked && !isBlocked,
            shouldQuiesce = retrievalResourceReleaseRequired &&
                !retrievalResourceReleaseInProgress &&
                indexingActivityHandoffOwners.isEmpty(),
        )
    }

    private fun onRetrievalResourcesBecameAvailable() {
        Log.i("MainViewModel", "Recommendation resources may load again after indexing")
        if (databaseRefreshGate.hasDeferredRequest()) {
            refreshDatabaseInfo()
        }
        startPendingSessionReplayEligibilityRefresh()
        refreshDisplayedQueueEligibility()
    }

    private suspend fun releaseRetrievalResourcesForIndexing() {
        RETRIEVAL_RESOURCE_RELEASE_MUTEX.withLock {
            val startedNs = System.nanoTime()
            logMemoryPhase("indexing_admission_before_find_music_release")

            val jobsToJoin = synchronized(RETRIEVAL_RESOURCE_ADMISSION_LOCK) {
                retrievalResourceReleaseInProgress = true
                updateRetrievalResourceAdmissionLocked()
                cancelFindMusicRequest()
                synchronized(previewStateLock) {
                    markAllPreviewsContextChangedLocked(
                        SettingsPeekUnavailableReason.FOREGROUND_REVALIDATION_REQUIRED,
                    )
                }
                synchronized(replayEligibilityJobLock) {
                    replayEligibilityRefreshJob = null
                }
                replayEligibilityRevision.incrementAndGet()
                replayEligibilityRefreshRunning.set(false)
                replayEligibilityRefreshRequested.set(sessionHistory.value.isNotEmpty())
                activeRetrievalResourceJobs.toList()
            }

            var becameAvailable = false
            try {
                kotlinx.coroutines.withContext(kotlinx.coroutines.NonCancellable) {
                    jobsToJoin.forEach {
                        it.cancel(CancellationException("Opening on-device indexing"))
                    }
                    jobsToJoin.forEach { it.join() }

                    kotlinx.coroutines.withContext(Dispatchers.IO) {
                        textInferenceMutex.withLock {
                            val inference = textInference
                            textInference = null
                            textInferenceIdentity = null
                            textEmbeddingCache.clear()
                            try {
                                inference?.close()
                            } catch (error: Exception) {
                                Log.w(
                                    "MainViewModel",
                                    "Failed to close text inference before indexing",
                                    error,
                                )
                            }
                        }

                        synchronized(TEXT_LIBRARY_BUILD_LOCK) {
                            synchronized(this@MainViewModel) {
                                rankIndexSnapshot = null
                                textIndex = null
                                textIndexSnapshot = null
                            }
                            invalidateTextLibrarySnapshot()
                            clearVerifiedFindMusicCatalogHandoff()
                        }

                        // This is an explicit transition between two memory-heavy modes. Encourage
                        // prompt reclamation after every strong reference and worker has gone away.
                        System.gc()
                    }
                }
            } finally {
                becameAvailable = synchronized(RETRIEVAL_RESOURCE_ADMISSION_LOCK) {
                    activeRetrievalResourceJobs.removeAll { it.isCompleted }
                    val wasBlocked = retrievalResourcesSuspended.get()
                    retrievalResourceReleaseInProgress = false
                    retrievalResourceReleaseRequired = false
                    updateRetrievalResourceAdmissionLocked()
                    wasBlocked && !retrievalResourcesSuspended.get()
                }
            }

            logMemoryPhase("indexing_admission_after_find_music_release")
            Log.i(
                "MainViewModel",
                "Released Find Music resources before indexing in " +
                    "${(System.nanoTime() - startedNs) / 1_000_000L}ms; " +
                    "joinedJobs=${jobsToJoin.size}",
            )
            if (becameAvailable) onRetrievalResourcesBecameAvailable()
        }
    }

    /**
     * Indexing loads two large audio models and rebuilds the embedding library. Release the
     * process's text-search model and cached library before crossing that memory-heavy boundary.
     */
    suspend fun prepareForOnDeviceIndexing() {
        synchronized(RETRIEVAL_RESOURCE_ADMISSION_LOCK) {
            indexingActivityHandoffOwners += indexingHandoffReservationOwner
            retrievalResourceReleaseRequired = true
            updateRetrievalResourceAdmissionLocked()
        }
        RecommendationWorkAdmission.reserve(indexingHandoffReservationOwner)
        try {
            releaseProcessRetrievalResourcesForIndexing()
            check(
                RadioService.suspendAndReleaseRecommendationResources(
                    timeoutMs = 60_000L,
                ),
            ) {
                INDEXING_HANDOFF_BUSY_MESSAGE
            }
        } catch (error: Throwable) {
            releaseIndexingHandoffReservation()
            throw error
        }
    }

    fun resumeRetrievalResourcesAfterIndexingIfIdle() {
        releaseIndexingHandoffReservation()
        val transition = synchronized(RETRIEVAL_RESOURCE_ADMISSION_LOCK) {
            val wasBlocked = retrievalResourcesSuspended.get()
            indexingResourceOwnership = indexingResourceOwnershipFor(
                state = IndexingService.state.value,
                durableAttachCompleted = durableIndexingAttachCompleted,
                filesDir = getApplication<Application>().filesDir,
            )
            updateRetrievalResourceAdmissionLocked()
            RetrievalAdmissionTransition(
                becameAvailable = wasBlocked && !retrievalResourcesSuspended.get(),
                shouldQuiesce = retrievalResourceReleaseRequired &&
                    !retrievalResourceReleaseInProgress,
            )
        }
        if (transition.shouldQuiesce) {
            viewModelScope.launch(Dispatchers.IO) {
                releaseRetrievalResourcesForIndexing()
            }
        }
        if (transition.becameAvailable) onRetrievalResourcesBecameAvailable()
    }

    fun abortOnDeviceIndexingHandoff() {
        resumeRetrievalResourcesAfterIndexingIfIdle()
    }

    private fun releaseIndexingHandoffReservation() {
        synchronized(RETRIEVAL_RESOURCE_ADMISSION_LOCK) {
            indexingActivityHandoffOwners -= indexingHandoffReservationOwner
            updateRetrievalResourceAdmissionLocked()
        }
        if (RecommendationWorkAdmission.release(indexingHandoffReservationOwner)) {
            RadioService.kickDeferredRecovery(getApplication<Application>())
        }
    }

    private suspend fun prepareForMusicIndexMutation() {
        RecommendationWorkAdmission.reserve(musicIndexMutationOwner)
        synchronized(RETRIEVAL_RESOURCE_ADMISSION_LOCK) {
            retrievalResourceReleaseRequired = true
            updateRetrievalResourceAdmissionLocked()
        }
        try {
            releaseProcessRetrievalResourcesForIndexing()
            check(
                RadioService.suspendAndReleaseRecommendationResources(
                    timeoutMs = 60_000L,
                ),
            ) {
                MUSIC_INDEX_MUTATION_BUSY_MESSAGE
            }
        } catch (error: Throwable) {
            releaseMusicIndexMutationReservation()
            throw error
        }
    }

    private fun releaseMusicIndexMutationReservation() {
        val globallyAvailable = RecommendationWorkAdmission.release(musicIndexMutationOwner)
        val locallyAvailable = synchronized(RETRIEVAL_RESOURCE_ADMISSION_LOCK) {
            val wasBlocked = retrievalResourcesSuspended.get()
            updateRetrievalResourceAdmissionLocked()
            wasBlocked && !retrievalResourcesSuspended.get()
        }
        if (locallyAvailable) onRetrievalResourcesBecameAvailable()
        if (globallyAvailable) {
            RadioService.kickDeferredRecovery(getApplication<Application>())
        }
    }

    private fun saveRecentSearch(search: RecentSearch) {
        // Deduplicate by exact search state, not by human-facing label.
        val stateKey = search.stateKey
        val updated = (listOf(search) + _recentSearches.value.filter { it.stateKey != stateKey }).take(10)
        _recentSearches.value = updated
        persistRecentSearches(updated)
    }

    private fun saveRecentSearchIfCurrent(revision: Long, search: RecentSearch) {
        synchronized(findMusicJobLock) {
            if (findMusicRequestGate.isCurrent(revision)) saveRecentSearch(search)
        }
    }

    fun clearRecentSearches() {
        _recentSearches.value = emptyList()
        prefs.edit()
            .remove("find_music_recent_queries_v3")
            .remove("recent_searches_v2")
            .remove("recent_searches")
            .apply()
    }

    private fun restoredTextIngredientState(search: RecentSearch): List<TextIngredientState> =
        search.textIngredients.mapNotNull { text ->
            val active = text.query.isNotBlank() && text.weight > 0f
            if (!active) return@mapNotNull null
            TextIngredientState(
                query = text.query,
                weight = text.weight,
                negative = text.negative,
            )
        }

    private fun restoreRefineEditorState(search: FindMusicQuerySpec) {
        val refine = search.refineSpec
        _findMusicRefinePrimaryIngredientIndex.value = refine?.primaryIngredientIndex ?: 0
        refine?.neighborhood?.let(::setFindMusicRefineNeighborhood)
    }

    /** Replay restores the complete editor request before running it again. */
    fun replayRecentSearch(search: RecentSearch) {
        val isPlainText = search.isSimplePositiveTextOnly
        val surface = if (isPlainText) {
            FindMusicSurface.TEXT
        } else {
            FindMusicSurface.COMPOSED
        }
        // The replayed request becomes the editor's visible current state, so persist the same
        // values that the controls now show rather than reverting them after process recreation.
        persistFindMusicOperator(search.operator)
        persistFindMusicTextResultPlanner(search.textResultPlanner)
        search.refineSpec?.neighborhood?.let(::setFindMusicRefineNeighborhood)
        setLibraryAddedDays(search.effectiveLibraryAddedDays)
        launchFindMusicRequest(surface) { revision ->
            validateQueryContract(search)?.let { message ->
                kotlinx.coroutines.withContext(Dispatchers.Main) {
                    _songSeeds.value = search.songSeeds.filter { it.weight > 0f }.map { anchor ->
                        SongSeedState(
                            query = anchor.displayLabel,
                            confirmedTrack = null,
                            stableTrackSpanId = anchor.stableTrackSpanId,
                            libraryBinding = search.libraryBinding,
                            weight = 0f,
                            negative = false,
                        )
                    }
                    _textIngredients.value = restoredTextIngredientState(search)
                    _findMusicOperator.value = search.operator
                    restoreRefineEditorState(search)
                }
                publishFindMusicError(
                    revision = revision,
                    surface = surface,
                    query = search.displayLabel,
                    message = message,
                    querySpec = search,
                )
                return@launchFindMusicRequest
            }
            if (search.songSeeds.none { it.weight > 0f }) {
                kotlinx.coroutines.withContext(Dispatchers.Main) {
                    cancelSongSeedLookup()
                    _songSeeds.value = emptyList()
                    _textIngredients.value = restoredTextIngredientState(search)
                    _findMusicOperator.value = search.operator
                    restoreRefineEditorState(search)
                }
                if (isPlainText) {
                    executeTextSearch(revision, surface, search)
                } else {
                    executeComposedSearch(revision, search)
                }
                return@launchFindMusicRequest
            }

            val filesDir = getApplication<Application>().filesDir
            val library = try {
                getOrCreateTextLibrarySnapshot(filesDir)
            } catch (e: Exception) {
                Log.e("MainViewModel", "Find Music replay could not prepare the active library", e)
                publishFindMusicError(
                    revision,
                    surface,
                    search.displayLabel,
                    "Find Music could not verify the active library. " +
                        "Check the music index and indexing files in Settings, then try again.",
                    search,
                )
                return@launchFindMusicRequest
            }
            val dbFile = library.databaseFile
            val restoredSeeds = mutableListOf<SongSeedState>()
            var resolvedSearch: FindMusicQuerySpec? = null
            var resolutionFailure: FindMusicAnchorBindingResult.Failure? = null
            val db = EmbeddingDatabase.open(dbFile)
            try {
                val catalog = library.identityCatalog
                val generationResolution = FindMusicAnchorResolver.resolveReplay(search, catalog)
                val resolution = when (generationResolution) {
                    is FindMusicAnchorBindingResult.Failure -> generationResolution
                    is FindMusicAnchorBindingResult.Success -> bindAnchorsToActiveDomain(
                        resolution = generationResolution,
                        identityCatalog = catalog,
                        activeDomain = library.activeDomain,
                    )
                }
                when (resolution) {
                    is FindMusicAnchorBindingResult.Failure -> resolutionFailure = resolution
                    is FindMusicAnchorBindingResult.Success -> resolvedSearch = resolution.querySpec
                }
                for (s in resolvedSearch?.songSeeds.orEmpty()) {
                    if (s.weight <= 0f) {
                        continue
                    }
                    val track = db.getTrackById(s.trackId)
                    if (track != null) {
                        restoredSeeds.add(SongSeedState(
                            query = s.displayLabel,
                            confirmedTrack = track,
                            stableTrackSpanId = s.stableTrackSpanId,
                            libraryBinding = resolvedSearch?.libraryBinding,
                            weight = s.weight,
                            negative = s.negative,
                        ))
                    } else {
                        resolutionFailure = FindMusicAnchorBindingResult.Failure(
                            message = "This saved search no longer matches the current library. " +
                                "Recreate it and choose the recording again: ${s.displayLabel}.",
                            unresolvedAnchors = listOf(s),
                            diagnosticDetail = "Resolved embedding track ${s.trackId} has no database row",
                        )
                    }
                }
            } finally {
                db.close()
            }
            ensureCurrentFindMusicRequest(revision)
            if (!isCurrentReadOnlyFindMusicLibrary(
                    filesDir,
                    library,
                    phase = "composed_replay_before_execute",
                )
            ) {
                publishFindMusicError(
                    revision,
                    surface,
                    search.displayLabel,
                    "The active music library changed while this search was being restored. Run it again.",
                    search,
                )
                return@launchFindMusicRequest
            }

            resolutionFailure?.let { failure ->
                failure.diagnosticDetail?.let { detail ->
                    Log.w("MainViewModel", "Find Music replay anchor failure: $detail")
                }
                kotlinx.coroutines.withContext(Dispatchers.Main) {
                    _songSeeds.value = search.songSeeds.filter { it.weight > 0f }.map { anchor ->
                        SongSeedState(
                            query = anchor.displayLabel,
                            confirmedTrack = null,
                            stableTrackSpanId = anchor.stableTrackSpanId,
                            libraryBinding = search.libraryBinding,
                            weight = 0f,
                            negative = false,
                        )
                    }
                    _textIngredients.value = restoredTextIngredientState(search)
                    _findMusicOperator.value = search.operator
                    restoreRefineEditorState(search)
                }
                publishFindMusicError(
                    revision = revision,
                    surface = surface,
                    query = search.displayLabel,
                    message = failure.message,
                    querySpec = search,
                    unresolvedAnchors = failure.unresolvedAnchors,
                )
                return@launchFindMusicRequest
            }
            val exactSearch = checkNotNull(resolvedSearch)

            kotlinx.coroutines.withContext(Dispatchers.Main) {
                _songSeeds.value = restoredSeeds
                _textIngredients.value = restoredTextIngredientState(search)
                _findMusicOperator.value = search.operator
                restoreRefineEditorState(search)
            }

            executeComposedSearch(revision, exactSearch, bindCurrentAnchors = false)
        }
    }

    // --- Composed Find Music editor state ---

    private val _songSeeds = MutableStateFlow<List<SongSeedState>>(emptyList())
    val songSeeds: StateFlow<List<SongSeedState>> = _songSeeds.asStateFlow()

    private val _recordingLookupState =
        MutableStateFlow<RecordingLookupState>(RecordingLookupState.Idle)
    val recordingLookupState: StateFlow<RecordingLookupState> =
        _recordingLookupState.asStateFlow()

    private var songSeedSearchBinding: StableIdentityGenerationBinding? = null
    private var songSeedSearchStableIds: Map<Long, String?> = emptyMap()

    private val songSeedLookupGate = LatestFindMusicRequestGate()
    private val songSeedLookupJobLock = Any()
    private var songSeedLookupJob: Job? = null

    /** Composed search results (separate from single-text raw-cosine results). */
    private val _multiSeedResult = MutableStateFlow<TextSearchResult?>(null)
    val multiSeedResult: StateFlow<TextSearchResult?> = _multiSeedResult.asStateFlow()

    private val _multiSeedLoading = MutableStateFlow(false)
    val multiSeedLoading: StateFlow<Boolean> = _multiSeedLoading.asStateFlow()

    private val _textIngredients = MutableStateFlow(listOf(TextIngredientState()))
    val textIngredients: StateFlow<List<TextIngredientState>> = _textIngredients.asStateFlow()

    init {
        synchronized(MAIN_VIEW_MODEL_INSTANCES_LOCK) {
            mainViewModelInstances.removeAll { it.get() == null }
            mainViewModelInstances.add(WeakReference(this))
        }
        RadioService.initHistory(application.filesDir)
        refreshDatabaseInfo()
        checkPermission()
        PowerampReceiver.addTrackChangeListener(trackChangeListener)

        // IndexingService.state is process-local and begins at Idle. Keep recommendation
        // admission closed until attach has inspected the durable pointer and orphan preflights.
        viewModelScope.launch(Dispatchers.IO) {
            runCatching {
                IndexingService.attach(application)
            }.onSuccess {
                val transition = reconcileIndexingResourceOwnership(
                    durableAttachCompletedNow = true,
                )
                if (transition.shouldQuiesce) releaseRetrievalResourcesForIndexing()
                if (transition.becameAvailable) onRetrievalResourcesBecameAvailable()
            }.onFailure { error ->
                Log.e(
                    "MainViewModel",
                    "Durable indexing ownership could not be reconciled",
                    error,
                )
            }
        }

        viewModelScope.launch(Dispatchers.IO) {
            updateLibraryControlsBlockedReason()
        }

        viewModelScope.launch(Dispatchers.IO) {
            RecommendationWorkAdmission.indexingReserved.collect { reserved ->
                val transition = synchronized(RETRIEVAL_RESOURCE_ADMISSION_LOCK) {
                    val wasBlocked = retrievalResourcesSuspended.get()
                    if (reserved && indexingActivityHandoffOwners.isEmpty() &&
                        !retrievalResourceReleaseInProgress
                    ) {
                        retrievalResourceReleaseRequired = true
                    }
                    updateRetrievalResourceAdmissionLocked()
                    RetrievalAdmissionTransition(
                        becameAvailable = wasBlocked && !retrievalResourcesSuspended.get(),
                        shouldQuiesce = retrievalResourceReleaseRequired &&
                            !retrievalResourceReleaseInProgress &&
                            indexingActivityHandoffOwners.isEmpty(),
                    )
                }
                if (transition.shouldQuiesce) releaseRetrievalResourcesForIndexing()
                if (transition.becameAvailable) onRetrievalResourcesBecameAvailable()
            }
        }

        // A completed publication changes the immutable database generation, so the prior
        // generation's ready count is no longer evidence about the active index. Manage Tracks
        // publishes its normal post-job comparison below; Settings adopts that exact result rather
        // than opening the complete Poweramp provider a second time.
        viewModelScope.launch(Dispatchers.IO) {
            var wasComplete = false
            IndexingService.state.collect { state ->
                val admission = reconcileIndexingResourceOwnership()
                if (admission.shouldQuiesce) releaseRetrievalResourcesForIndexing()
                if (admission.becameAvailable) onRetrievalResourcesBecameAvailable()
                updateLibraryControlsBlockedReason()
                if (state is IndexingService.IndexingState.JobSnapshot &&
                    state.jobState == IndexingJobState.COMPLETE
                ) {
                    val firstObservation = !wasComplete
                    wasComplete = true
                    if (firstObservation) {
                        rankIndexSnapshot = null
                        invalidateTextEmbeddingIndex()
                        invalidateAllPreviewsForContextChange()
                        IndexingViewModel.invalidateCache()
                        invalidateUnindexedCountEvidence()
                    }
                }
                if (wasComplete && state is IndexingService.IndexingState.Idle) {
                    wasComplete = false
                }
            }
        }
        viewModelScope.launch(Dispatchers.IO) {
            IndexingViewModel.ownedDetectionResults.collect { result ->
                runCatching {
                    publishUnindexedTrackCount(application, result)
                }.onFailure { error ->
                    Log.w(
                        "MainViewModel",
                        "Ignored completed Manage Tracks detection for a superseded music index",
                        error,
                    )
                }
            }
        }
        viewModelScope.launch(Dispatchers.IO) {
            // Initial readiness publishes one exact replay binding for whichever history was
            // restored. Only subsequent history mutations need an independent refresh request.
            sessionHistory.drop(1).collect {
                adoptProcessCatalogIfAvailable()
                refreshSessionReplayEligibility()
            }
        }
    }

    fun updateTextIngredientQuery(index: Int, query: String) {
        val current = _textIngredients.value.toMutableList()
        if (index !in current.indices) return
        val wasComplete = current[index].query.isNotBlank()
        val isComplete = query.isNotBlank()
        current[index] = current[index].copy(
            query = query,
            weight = if (isComplete) current[index].weight else 0f,
            negative = if (isComplete) current[index].negative else false,
            locked = if (isComplete) current[index].locked else false,
        )
        _textIngredients.value = current
        when {
            !wasComplete && isComplete -> activateWeightSlot(index)
            wasComplete && !isComplete -> normalizeWeightState(releaseHolds = true)
        }
    }

    fun toggleTextIngredientSign(index: Int) {
        val current = _textIngredients.value.toMutableList()
        if (index !in current.indices || !current[index].isActive) return
        if (!current[index].negative && !FindMusicEditorPolicy.canSetTextToAvoid(
                textIngredients = current,
                songSeeds = _songSeeds.value,
                index = index,
            )
        ) return
        if (!current[index].negative &&
            _findMusicOperator.value == FindMusicOperator.REFINE &&
            current.take(index).count(TextIngredientState::isActive) ==
            _findMusicRefinePrimaryIngredientIndex.value
        ) return
        current[index] = current[index].copy(negative = !current[index].negative)
        _textIngredients.value = current
        normalizeRefinePrimaryIngredient()
    }

    fun toggleTextIngredientLock(index: Int) {
        val current = _textIngredients.value.toMutableList()
        if (index !in current.indices || !current[index].isActive ||
            activeIngredientCount() < 3
        ) return
        current[index] = current[index].copy(locked = !current[index].locked)
        _textIngredients.value = current
        normalizeWeightState()
    }

    companion object {
        internal const val INDEXING_HANDOFF_BUSY_MESSAGE =
            "A radio queue is still finishing. Try opening on-device indexing again after it completes."
        internal const val MUSIC_INDEX_MUTATION_BUSY_MESSAGE =
            "A radio queue is still finishing. Try merging the server index again after it completes."
        private const val LAST_SERVER_MERGE_RESULT_PREF = "last_server_merge_result_v2"
        private const val RECORDING_LOOKUP_DISPLAY_LIMIT = 50
        private const val BYTES_PER_MIB = 1024L * 1024L
        private val TEXT_INFERENCE_CLEANUP_SCOPE =
            CoroutineScope(SupervisorJob() + Dispatchers.IO)
        private val TEXT_LIBRARY_BUILD_LOCK = Any()
        private val TEXT_LIBRARY_CACHE_LOCK = Any()
        private val RETRIEVAL_RESOURCE_ADMISSION_LOCK = Any()
        private val RETRIEVAL_RESOURCE_RELEASE_MUTEX = Mutex()
        private val activeRetrievalResourceJobs = linkedSetOf<Job>()
        private val MAIN_VIEW_MODEL_INSTANCES_LOCK = Any()
        private val mainViewModelInstances = mutableListOf<WeakReference<MainViewModel>>()
        private val indexingActivityHandoffOwners =
            mutableSetOf<RecommendationWorkAdmission.ReservationOwner>()
        private val databaseProviderAcquisitions = AtomicLong(0L)
        private var retrievalResourceReleaseInProgress = false
        private var retrievalResourceReleaseRequired = false
        private var durableIndexingAttachCompleted = false
        private var indexingResourceOwnership = IndexingResourceOwnership.UNKNOWN
        // A cold process is fail-closed until durable indexing ownership has been inspected.
        private val retrievalResourcesSuspended = AtomicBoolean(true)
        @Volatile
        private var processTextLibrarySnapshot: FindMusicLibrarySnapshot? = null

        internal fun databaseProviderAcquisitionCount(): Long =
            databaseProviderAcquisitions.get()

        private fun updateRetrievalResourceAdmissionLocked() {
            retrievalResourcesSuspended.set(
                indexingActivityHandoffOwners.isNotEmpty() ||
                    retrievalResourceReleaseInProgress ||
                    retrievalResourceReleaseRequired ||
                    indexingResourceOwnership != IndexingResourceOwnership.FREE ||
                    RecommendationWorkAdmission.isIndexingReserved,
            )
        }

        private fun recommendationResourcesBlocked(): Boolean =
            retrievalResourcesSuspended.get() ||
                RecommendationWorkAdmission.isIndexingReserved

        private fun indexingResourceOwnershipFor(
            state: IndexingService.IndexingState,
            durableAttachCompleted: Boolean,
            filesDir: File,
        ): IndexingResourceOwnership {
            val stateOwnership = when (state) {
                IndexingService.IndexingState.Idle -> {
                    if (durableAttachCompleted) IndexingResourceOwnership.FREE
                    else IndexingResourceOwnership.UNKNOWN
                }
                is IndexingService.IndexingState.Error -> IndexingResourceOwnership.UNKNOWN
                // A valid snapshot proves durable ownership was read. Runtime model ownership is
                // represented independently by RecommendationWorkAdmission; paused or interrupted
                // ledgers must remain resumable without disabling recommendation work.
                is IndexingService.IndexingState.JobSnapshot -> IndexingResourceOwnership.FREE
                is IndexingService.IndexingState.PreflightSnapshot -> IndexingResourceOwnership.FREE
            }
            if (stateOwnership == IndexingResourceOwnership.FREE) return stateOwnership
            return when (V2ActiveIndexingJobPointer(filesDir).inspect()) {
                V2ActiveIndexingJobPointerInspection.Missing -> stateOwnership
                is V2ActiveIndexingJobPointerInspection.Readable ->
                    IndexingResourceOwnership.HELD
                is V2ActiveIndexingJobPointerInspection.Unreadable ->
                    IndexingResourceOwnership.UNKNOWN
            }
        }

        internal suspend fun releaseProcessRetrievalResourcesForIndexing() {
            val instances = synchronized(MAIN_VIEW_MODEL_INSTANCES_LOCK) {
                mainViewModelInstances.removeAll { it.get() == null }
                mainViewModelInstances.mapNotNull { it.get() }.distinct()
            }
            if (instances.isEmpty()) {
                synchronized(TEXT_LIBRARY_BUILD_LOCK) {
                    synchronized(TEXT_LIBRARY_CACHE_LOCK) {
                        processTextLibrarySnapshot = null
                    }
                }
                return
            }
            instances.forEach { instance ->
                synchronized(RETRIEVAL_RESOURCE_ADMISSION_LOCK) {
                    retrievalResourceReleaseRequired = true
                    updateRetrievalResourceAdmissionLocked()
                }
                instance.releaseRetrievalResourcesForIndexing()
            }
        }
    }

    private enum class IndexingResourceOwnership {
        UNKNOWN,
        FREE,
        HELD,
    }

    private fun currentEditorWeightSlots(): List<FindMusicEditorWeightSlot> {
        val texts = _textIngredients.value
        val songs = _songSeeds.value
        return buildList(texts.size + songs.size) {
            texts.forEach { text ->
                add(
                    FindMusicEditorWeightSlot(
                        weight = text.weight,
                        locked = text.locked,
                        completed = text.query.isNotBlank(),
                    ),
                )
            }
            songs.forEach { song ->
                add(
                    FindMusicEditorWeightSlot(
                        weight = song.weight,
                        locked = song.locked,
                        completed = song.confirmedTrack != null,
                    ),
                )
            }
        }
    }

    private fun applyWeightValues(weights: FloatArray) {
        val texts = _textIngredients.value
        val songs = _songSeeds.value
        require(weights.size == texts.size + songs.size)
        _textIngredients.value = texts.mapIndexed { index, text ->
            text.copy(weight = weights[index])
        }
        _songSeeds.value = songs.mapIndexed { index, song ->
            song.copy(weight = weights[index + texts.size])
        }
    }

    private fun normalizeWeightState(releaseHolds: Boolean = false) {
        if (releaseHolds) releaseAllIngredientHolds()
        clearIrrelevantLocks()
        applyWeightValues(
            FindMusicEditorWeightPolicy.normalize(
                slots = currentEditorWeightSlots(),
                minimumActiveWeight = currentMinimumActiveWeight(),
            ),
        )
        resetIrrelevantOperator()
        snapTwoIngredientAllOfPriority()
        normalizeRefinePrimaryIngredient()
    }

    private fun snapTwoIngredientAllOfPriority() {
        if (_findMusicOperator.value != FindMusicOperator.ALL_OF) return
        val slots = currentEditorWeightSlots()
        val activeIndices = slots.indices.filter { slots[it].completed }
        if (activeIndices.size != 2) return

        val firstIndex = activeIndices[0]
        val secondIndex = activeIndices[1]
        val firstWeight = ((slots[firstIndex].weight * 10f).roundToInt().coerceIn(1, 9)) / 10f
        applyWeightValues(
            FloatArray(slots.size).also { snapped ->
                snapped[firstIndex] = firstWeight
                snapped[secondIndex] = 1f - firstWeight
            },
        )
    }

    private fun releaseAllIngredientHolds() {
        _textIngredients.value = _textIngredients.value.map { it.copy(locked = false) }
        _songSeeds.value = _songSeeds.value.map { it.copy(locked = false) }
    }

    private fun clearIrrelevantLocks() {
        if (activeIngredientCount() >= 3) return
        _textIngredients.value = _textIngredients.value.map { it.copy(locked = false) }
        _songSeeds.value = _songSeeds.value.map { it.copy(locked = false) }
    }

    private fun activateWeightSlot(index: Int) {
        releaseAllIngredientHolds()
        applyWeightValues(
            FindMusicEditorWeightPolicy.activate(
                slots = currentEditorWeightSlots(),
                activatedIndex = index,
                minimumActiveWeight = currentMinimumActiveWeight(),
            ),
        )
        resetIrrelevantOperator()
    }

    private fun resetIrrelevantOperator() {
        if (activeIngredientCount() != 2 &&
            _findMusicOperator.value == FindMusicOperator.REFINE
        ) {
            persistFindMusicOperator(FindMusicOperator.ALL_OF)
        }
    }

    private fun hasActiveAvoidIngredient(): Boolean =
        _textIngredients.value.any { it.isActive && it.negative } ||
            _songSeeds.value.any { it.isActive && it.negative }

    private fun activeIngredientCount(): Int =
        _textIngredients.value.count(TextIngredientState::isActive) +
            _songSeeds.value.count(SongSeedState::isActive)

    private fun activeIngredientSigns(): List<Boolean> =
        _textIngredients.value.filter(TextIngredientState::isActive).map { it.negative } +
            _songSeeds.value.filter(SongSeedState::isActive).map { it.negative }

    private fun normalizeRefinePrimaryIngredient() {
        val signs = activeIngredientSigns()
        val current = _findMusicRefinePrimaryIngredientIndex.value
        if (current !in signs.indices || signs[current]) {
            _findMusicRefinePrimaryIngredientIndex.value =
                signs.indexOfFirst { negative -> !negative }.coerceAtLeast(0)
        }
    }

    private fun currentMinimumActiveWeight(): Float =
        FindMusicEditorWeightPolicy.minimumActiveWeight(
            operator = _findMusicOperator.value,
            resultLimit = _numTracks.value,
        )

    private fun currentWeightSlots(): List<FindMusicWeightSlot> =
        currentEditorWeightSlots().map { slot ->
            if (slot.completed) {
                FindMusicWeightSlot(
                    weight = slot.weight,
                    minimum = currentMinimumActiveWeight(),
                    locked = slot.locked,
                )
            } else {
                FindMusicWeightSlot(weight = 0f, minimum = 0f, locked = true)
            }
        }

    /**
     * Update one knob and deterministically redistribute the residual immediately.
     */
    fun updateTextIngredientWeight(index: Int, weight: Float) {
        if (index !in _textIngredients.value.indices ||
            !_textIngredients.value[index].isActive
        ) return
        applyWeightValues(
            FindMusicWeightAllocator.adjust(currentWeightSlots(), index, weight),
        )
    }

    /** Defensive idempotent normalization at gesture completion. */
    fun finalizeTextIngredientWeight(index: Int) {
        if (index !in _textIngredients.value.indices) return
        normalizeWeightState()
    }

    /**
     * Song index is translated to the shared text-plus-songs simplex.
     */
    fun updateSongSeedWeight(index: Int, weight: Float) {
        if (index !in _songSeeds.value.indices || !_songSeeds.value[index].isActive) return
        applyWeightValues(
            FindMusicWeightAllocator.adjust(
                currentWeightSlots(),
                index + _textIngredients.value.size,
                weight,
            ),
        )
    }

    /** Defensive idempotent normalization at gesture completion. */
    fun finalizeSongSeedWeight(index: Int) {
        if (index !in _songSeeds.value.indices) return
        normalizeWeightState()
    }

    fun clearSongSeeds() {
        cancelSongSeedLookup()
        val removedActiveIngredient = _songSeeds.value.any { it.confirmedTrack != null }
        _songSeeds.value = emptyList()
        if (_textIngredients.value.isEmpty()) {
            _textIngredients.value = listOf(TextIngredientState())
        }
        normalizeWeightState(releaseHolds = removedActiveIngredient)
    }

    fun addTextIngredient() {
        if (_textIngredients.value.size >= FindMusicQuerySpec.MAX_TEXT_INGREDIENTS) return
        _textIngredients.value = _textIngredients.value + TextIngredientState()
    }

    fun removeTextIngredient(index: Int) {
        val current = _textIngredients.value.toMutableList()
        if (index !in current.indices) return
        val removedActiveIngredient = current[index].query.isNotBlank()
        current.removeAt(index)
        if (current.isEmpty() && _songSeeds.value.isEmpty()) {
            current += TextIngredientState()
        }
        _textIngredients.value = current
        normalizeWeightState(releaseHolds = removedActiveIngredient)
    }

    fun addSongSeed() {
        if (_songSeeds.value.isEmpty() &&
            _textIngredients.value.singleOrNull()?.query?.isBlank() == true
        ) {
            _textIngredients.value = emptyList()
        }
        _songSeeds.value = _songSeeds.value + SongSeedState()
    }

    fun removeSongSeed(index: Int) {
        cancelSongSeedLookup()
        val current = _songSeeds.value.toMutableList()
        var removedActiveIngredient = false
        if (index in current.indices) {
            removedActiveIngredient = current[index].confirmedTrack != null
            current.removeAt(index)
            _songSeeds.value = current
        }
        if (current.isEmpty() && _textIngredients.value.isEmpty()) {
            _textIngredients.value = listOf(TextIngredientState())
        }
        if (current.isEmpty()) _multiSeedResult.value = null
        normalizeWeightState(releaseHolds = removedActiveIngredient)
    }

    fun updateSongSeedQuery(index: Int, query: String) {
        val current = _songSeeds.value.toMutableList()
        var editedSeedId: Long? = null
        if (index in current.indices) {
            editedSeedId = current[index].id
            val wasComplete = current[index].confirmedTrack != null
            current[index] = current[index].copy(
                query = query,
                confirmedTrack = null,
                stableTrackSpanId = null,
                libraryBinding = null,
                weight = 0f,
                negative = false,
                locked = false,
            )
            _songSeeds.value = current
            if (wasComplete) normalizeWeightState(releaseHolds = true)
        }
        if (editedSeedId != null &&
            _recordingLookupState.value.forSeed(editedSeedId) !is RecordingLookupState.Idle
        ) {
            cancelSongSeedLookup()
        }
    }

    fun confirmSongSeed(index: Int, track: EmbeddedTrack) {
        val seed = _songSeeds.value.getOrNull(index) ?: return
        val success = _recordingLookupState.value.forSeed(seed.id)
            as? RecordingLookupState.Success ?: return
        if (success.candidates.none { it == track }) return
        val confirmedStableId = songSeedSearchStableIds[track.id]
        val confirmedBinding = songSeedSearchBinding
        cancelSongSeedLookup()
        val current = _songSeeds.value.toMutableList()
        if (index in current.indices) {
            val wasComplete = current[index].confirmedTrack != null
            current[index] = current[index].copy(
                query = recordingDisplayLabel(
                    artist = track.artist,
                    title = track.title,
                    fallback = "Track ${track.id}",
                ),
                confirmedTrack = track,
                stableTrackSpanId = confirmedStableId,
                libraryBinding = confirmedBinding,
            )
            _songSeeds.value = current
            if (!wasComplete) activateWeightSlot(index + _textIngredients.value.size)
        }
    }

    fun toggleSongSeedLock(index: Int) {
        val current = _songSeeds.value.toMutableList()
        if (index in current.indices && current[index].isActive &&
            activeIngredientCount() >= 3
        ) {
            current[index] = current[index].copy(locked = !current[index].locked)
            _songSeeds.value = current
            normalizeWeightState()
        }
    }

    fun toggleSongSeedSign(index: Int) {
        val current = _songSeeds.value.toMutableList()
        if (index in current.indices && current[index].isActive) {
            if (!current[index].negative && !FindMusicEditorPolicy.canSetSongToAvoid(
                    textIngredients = _textIngredients.value,
                    songSeeds = current,
                    index = index,
                )
            ) return
            val activeOrdinal = _textIngredients.value.count(TextIngredientState::isActive) +
                current.take(index).count(SongSeedState::isActive)
            if (!current[index].negative &&
                _findMusicOperator.value == FindMusicOperator.REFINE &&
                activeOrdinal == _findMusicRefinePrimaryIngredientIndex.value
            ) return
            current[index] = current[index].copy(negative = !current[index].negative)
            _songSeeds.value = current
            normalizeRefinePrimaryIngredient()
        }
    }

    fun searchSongSeed(index: Int) {
        val seeds = _songSeeds.value
        if (index !in seeds.indices) return
        val query = seeds[index].query.trim()
        if (query.isBlank()) return

        val seedId = seeds[index].id
        val revision = songSeedLookupGate.begin()
        lateinit var job: Job
        job = viewModelScope.launch(
            context = Dispatchers.IO,
            start = CoroutineStart.LAZY,
        ) {
            try {
                val filesDir = getApplication<Application>().filesDir
                val library = getOrCreateTextLibrarySnapshot(filesDir) { status ->
                    if (songSeedLookupGate.isCurrent(revision)) {
                        _recordingLookupState.value = RecordingLookupStateReducer.progress(
                            current = _recordingLookupState.value,
                            seedId = seedId,
                            query = query,
                            message = status,
                        )
                    }
                }
                val dbFile = library.databaseFile
                _recordingLookupState.value = RecordingLookupStateReducer.progress(
                    current = _recordingLookupState.value,
                    seedId = seedId,
                    query = query,
                    message = "Searching indexed artist, album, title, and filename fields for " +
                        "\"$query\"",
                )
                val db = EmbeddingDatabase.open(dbFile)
                var binding: StableIdentityGenerationBinding? = null
                var stableIds: Map<Long, String?> = emptyMap()
                var hasMoreMatches = false
                val results = try {
                    val matchPage = db.searchTracksByTextPage(
                        query = query,
                        limit = RECORDING_LOOKUP_DISPLAY_LIMIT,
                        includeTrackId = library.activeDomain::containsActiveTrack,
                        canonicalTrackId =
                            library.activeDomain::activeIdentityRepresentativeTrackId,
                    )
                    val matches = matchPage.tracks
                    hasMoreMatches = matchPage.hasMore
                    val catalog = library.identityCatalog
                    binding = catalog.binding
                    stableIds = matches.associate { it.id to catalog.stableTrackSpanId(it.id) }
                    matches
                } finally {
                    db.close()
                }
                val currentSeed = _songSeeds.value.getOrNull(index)
                if (songSeedLookupGate.isCurrent(revision) &&
                    currentSeed?.id == seedId && currentSeed.query.trim() == query
                ) {
                    if (isCurrentReadOnlyFindMusicLibrary(
                            filesDir,
                            library,
                            phase = "recording_lookup_before_publish",
                        )
                    ) {
                        songSeedSearchBinding = binding
                        songSeedSearchStableIds = stableIds
                        _recordingLookupState.value = RecordingLookupStateReducer.succeed(
                            current = _recordingLookupState.value,
                            seedId = seedId,
                            query = query,
                            candidates = results,
                            hasMoreMatches = hasMoreMatches,
                        )
                    } else {
                        _recordingLookupState.value = RecordingLookupStateReducer.fail(
                            current = _recordingLookupState.value,
                            seedId = seedId,
                            query = query,
                            message = "The library changed during lookup. Search for the recording again.",
                        )
                    }
                }
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                Log.e("MainViewModel", "Song seed search failed", e)
                if (songSeedLookupGate.isCurrent(revision)) {
                    _recordingLookupState.value = RecordingLookupStateReducer.fail(
                        current = _recordingLookupState.value,
                        seedId = seedId,
                        query = query,
                        message = "Recording search could not read the active library. Try again.",
                    )
                }
            } finally {
                synchronized(songSeedLookupJobLock) {
                    if (songSeedLookupJob === job) songSeedLookupJob = null
                }
            }
        }
        trackRetrievalJobUntilCompletion(job)
        var previous: Job? = null
        val admitted = synchronized(RETRIEVAL_RESOURCE_ADMISSION_LOCK) {
            if (recommendationResourcesBlocked()) {
                false
            } else {
                activeRetrievalResourceJobs.add(job)
                synchronized(songSeedLookupJobLock) {
                    previous = songSeedLookupJob
                    songSeedLookupJob = job
                }
                true
            }
        }
        if (!admitted) {
            job.cancel(CancellationException("On-device indexing owns recommendation resources"))
            return
        }
        songSeedSearchBinding = null
        songSeedSearchStableIds = emptyMap()
        _recordingLookupState.value = RecordingLookupStateReducer.start(seedId, query)
        previous?.cancel(CancellationException("Superseded song lookup"))
        job.start()
    }

    fun dismissSongSeedSearch() {
        cancelSongSeedLookup()
    }

    private fun cancelSongSeedLookup(): Job? {
        songSeedLookupGate.cancel()
        val job = synchronized(songSeedLookupJobLock) {
            songSeedLookupJob.also { songSeedLookupJob = null }
        }
        job?.cancel(CancellationException("Song lookup dismissed"))
        _recordingLookupState.value = RecordingLookupStateReducer.clear()
        songSeedSearchBinding = null
        songSeedSearchStableIds = emptyMap()
        return job
    }

    /** Snapshot every visible ingredient, then run exactly that immutable request. */
    fun performFindMusicSearch() {
        val editorTexts = _textIngredients.value
        val editorSeeds = _songSeeds.value
        val searchRunning = _textSearchLoading.value || _multiSeedLoading.value
        val activeEditorTexts = editorTexts.filter(TextIngredientState::isActive)
        val activeEditorSeeds = editorSeeds.filter(SongSeedState::isActive)
        val activeIngredientCount = activeEditorTexts.size + activeEditorSeeds.size
        val effectiveOperator = FindMusicEditorPolicy.effectiveOperator(
            requested = _findMusicOperator.value,
            activeIngredientCount = activeIngredientCount,
        )
        val readiness = FindMusicEditorPolicy.readiness(
            textIngredients = editorTexts,
            songSeeds = editorSeeds,
            searchRunning = searchRunning,
            operator = effectiveOperator,
            resultLimit = _numTracks.value,
            refinePrimaryIngredientIndex = _findMusicRefinePrimaryIngredientIndex.value,
        )
        if (!readiness.canSearch) {
            if (!searchRunning) {
                launchFindMusicRequest(FindMusicSurface.COMPOSED) { revision ->
                    publishFindMusicError(
                        revision = revision,
                        surface = FindMusicSurface.COMPOSED,
                        query = "Find music",
                        message = checkNotNull(readiness.reason),
                    )
                }
            }
            return
        }
        val legacyBindings = activeEditorSeeds
            .filter { it.confirmedTrack != null && it.stableTrackSpanId == null }
            .mapNotNull { it.libraryBinding }
            .distinct()
        val unconfirmed = activeEditorSeeds.filter {
            it.confirmedTrack == null ||
                (it.stableTrackSpanId == null &&
                    (it.libraryBinding == null || legacyBindings.size > 1))
        }
        val queryBinding = legacyBindings.singleOrNull()
            ?: activeEditorSeeds.mapNotNull { it.libraryBinding }.distinct().singleOrNull()
        val canonicalRefineWeight = if (effectiveOperator == FindMusicOperator.REFINE) {
            1f / activeIngredientCount
        } else {
            null
        }
        val draftQuerySpec = FindMusicQuerySpec(
            operator = effectiveOperator,
            textIngredients = activeEditorTexts.map { text ->
                FindMusicTextIngredient(
                    query = text.query,
                    weight = canonicalRefineWeight ?: text.weight,
                    negative = text.negative,
                )
            },
            songSeeds = activeEditorSeeds.map { seed ->
                checkNotNull(seed.confirmedTrack).let { track ->
                    FindMusicSongAnchor(
                        trackId = track.id,
                        stableTrackSpanId = seed.stableTrackSpanId,
                        artist = track.artist,
                        title = track.title,
                        weight = canonicalRefineWeight ?: seed.weight,
                        negative = seed.negative,
                    )
                }
            },
            refineSpec = if (effectiveOperator == FindMusicOperator.REFINE) {
                FindMusicRefineSpec(
                    primaryIngredientIndex = _findMusicRefinePrimaryIngredientIndex.value,
                    neighborhood = _findMusicRefineNeighborhood.value,
                )
            } else {
                null
            },
            resultLimit = _numTracks.value,
            textResultPlanner = _findMusicTextResultPlanner.value,
            libraryAddedDays = _libraryAddedDays.value,
            libraryBinding = queryBinding,
        )
        val effectivePlanner = when {
            draftQuerySpec.isSimplePositiveTextOnly &&
                draftQuerySpec.textResultPlanner == FindMusicTextResultPlanner.VARIED_DPP ->
                FindMusicTextResultPlanner.VARIED_DPP
            draftQuerySpec.operator == FindMusicOperator.ALL_OF &&
                draftQuerySpec.activeIngredientCount >= 2 &&
                draftQuerySpec.textResultPlanner ==
                FindMusicTextResultPlanner.VARIED_ALL_OF_DPP ->
                FindMusicTextResultPlanner.VARIED_ALL_OF_DPP
            else -> FindMusicTextResultPlanner.CLOSEST
        }
        val querySpec = draftQuerySpec.copy(textResultPlanner = effectivePlanner)
        val surface = if (querySpec.isSimplePositiveTextOnly) {
            FindMusicSurface.TEXT
        } else {
            FindMusicSurface.COMPOSED
        }

        launchFindMusicRequest(surface) { revision ->
            val blankDescriptions = editorTexts.count {
                it.weight > 0f && it.query.isBlank()
            }
            if (blankDescriptions > 0) {
                publishFindMusicError(
                    revision = revision,
                    surface = surface,
                    query = querySpec.displayLabel,
                    message = "Fill or remove every active description before searching.",
                    querySpec = querySpec,
                )
                return@launchFindMusicRequest
            }
            if (unconfirmed.isNotEmpty()) {
                val labels = unconfirmed.joinToString(", ") {
                    it.query.trim().ifBlank { "unnamed song ingredient" }
                }
                publishFindMusicError(
                    revision = revision,
                    surface = surface,
                    query = querySpec.displayLabel,
                    message = "Choose an exact recording for ${if (unconfirmed.size == 1) "this song ingredient" else "these song ingredients"}: $labels. No first match was selected automatically.",
                    querySpec = querySpec,
                )
                return@launchFindMusicRequest
            }
            if (surface == FindMusicSurface.TEXT) {
                executeTextSearch(revision, surface, querySpec)
            } else {
                executeComposedSearch(revision, querySpec, bindCurrentAnchors = true)
            }
        }
    }

    private suspend fun executeComposedSearch(
        revision: Long,
        querySpec: FindMusicQuerySpec,
        bindCurrentAnchors: Boolean = false,
    ) {
        val requestReferenceEpochSecond = System.currentTimeMillis() / 1_000L
        validateComposedQuery(querySpec)?.let { message ->
            publishFindMusicError(
                revision,
                FindMusicSurface.COMPOSED,
                querySpec.displayLabel,
                message,
                querySpec,
            )
            return
        }

        val filesDir = getApplication<Application>().filesDir
        val library = try {
            getOrCreateTextLibrarySnapshot(filesDir) { status ->
                publishFindMusicLoadingStatus(revision, status)
            }
        } catch (e: Exception) {
            Log.e("MainViewModel", "All-of search could not prepare the active library", e)
            publishFindMusicError(
                revision,
                FindMusicSurface.COMPOSED,
                querySpec.displayLabel,
                "Find Music could not verify the active library. " +
                    "Check the music index and indexing files in Settings, then try again.",
                querySpec,
            )
            return
        }
        val dbFile = library.databaseFile
        val index = library.embeddingIndex
        val candidateDomain = library.activeDomain.candidateDomain(
            minimumLibraryAddedAtEpochSecond(
                querySpec.effectiveLibraryAddedDays,
                requestReferenceEpochSecond,
            ),
        )
        if (candidateDomain.candidateIdentityCount == 0) {
            publishFindMusicError(
                revision,
                FindMusicSurface.COMPOSED,
                querySpec.displayLabel,
                noCandidateRecordingsMessage(querySpec.effectiveLibraryAddedDays),
                querySpec,
            )
            return
        }

        val seedSpecs = mutableListOf<SeedSpec>()
        querySpec.activeTextIngredients.forEachIndexed { index, text ->
            val embedding = try {
                generateTextEmbeddingWithRetry(
                    revision,
                    text.query,
                    filesDir,
                    library.textAssets,
                ) { status ->
                    publishFindMusicLoadingStatus(
                        revision,
                        "Description ${index + 1} of " +
                            "${querySpec.activeTextIngredients.size} · $status",
                    )
                }
            } catch (e: CancellationException) {
                throw e
            } catch (e: IllegalStateException) {
                Log.e("MainViewModel", "All-of text inference failed", e)
                publishFindMusicError(
                    revision,
                    FindMusicSurface.COMPOSED,
                    querySpec.displayLabel,
                    "Find Music could not use the text model. " +
                        "Check the music index and indexing files in Settings, then try again.",
                    querySpec,
                )
                return
            }
            seedSpecs += SeedSpec(
                embedding = embedding,
                weight = if (text.negative) -text.weight else text.weight,
                label = text.query,
                type = SeedType.TEXT,
            )
        }
        ensureCurrentFindMusicRequest(revision)

        val missingEmbeddings = mutableListOf<FindMusicSongAnchor>()
        var effectiveQuerySpec = querySpec
        var equivalentSeedTrackIds: Set<Long> = emptySet()
        lateinit var identityCatalog: StableTrackIdentityCatalog
        val db = EmbeddingDatabase.open(dbFile)
        try {
            identityCatalog = library.identityCatalog
            val generationResolution = if (bindCurrentAnchors) {
                FindMusicAnchorResolver.bindCurrent(querySpec, identityCatalog)
            } else {
                FindMusicAnchorResolver.resolveReplay(querySpec, identityCatalog)
            }
            val resolution = when (generationResolution) {
                is FindMusicAnchorBindingResult.Failure -> generationResolution
                is FindMusicAnchorBindingResult.Success -> bindAnchorsToActiveDomain(
                    resolution = generationResolution,
                    identityCatalog = identityCatalog,
                    activeDomain = library.activeDomain,
                )
            }
            when (resolution) {
                is FindMusicAnchorBindingResult.Failure -> {
                    resolution.diagnosticDetail?.let { detail ->
                        Log.w("MainViewModel", "Find Music anchor failure: $detail")
                    }
                    publishFindMusicError(
                        revision = revision,
                        surface = FindMusicSurface.COMPOSED,
                        query = querySpec.displayLabel,
                        message = resolution.message,
                        querySpec = querySpec,
                        unresolvedAnchors = resolution.unresolvedAnchors,
                    )
                    return
                }
                is FindMusicAnchorBindingResult.Success -> {
                    effectiveQuerySpec = resolution.querySpec
                    equivalentSeedTrackIds = resolution.equivalentTrackIdsToExclude
                }
            }
            for (anchor in effectiveQuerySpec.songSeeds.filter { it.weight > 0f }) {
                val embedding = db.getEmbedding(anchor.trackId)
                if (embedding == null) {
                    missingEmbeddings += anchor
                } else {
                    seedSpecs += SeedSpec(
                        embedding = embedding,
                        weight = if (anchor.negative) -anchor.weight else anchor.weight,
                        label = anchor.displayLabel,
                        type = SeedType.SONG,
                        trackId = anchor.trackId,
                    )
                }
            }
        } finally {
            db.close()
        }
        ensureCurrentFindMusicRequest(revision)

        if (missingEmbeddings.isNotEmpty()) {
            publishFindMusicError(
                revision = revision,
                surface = FindMusicSurface.COMPOSED,
                query = effectiveQuerySpec.displayLabel,
                message = "Cannot run exactly. ${missingEmbeddings.joinToString(", ") { it.displayLabel }} " +
                    "${if (missingEmbeddings.size == 1) "has" else "have"} not been indexed for music " +
                    "matching. No ingredients were dropped.",
                querySpec = effectiveQuerySpec,
                unresolvedAnchors = missingEmbeddings,
            )
            return
        }

        val requestJob = currentCoroutineContext()[Job]
        publishFindMusicLoadingStatus(
            revision,
            if (effectiveQuerySpec.textResultPlanner ==
                FindMusicTextResultPlanner.VARIED_ALL_OF_DPP
            ) {
                "Ranking the complete All-of domain before planning " +
                    "${effectiveQuerySpec.resultLimit} varied recordings"
            } else {
                "Ranking ${candidateDomain.candidateIdentityCount} recording identities in " +
                "${libraryAddedDaysLabel(querySpec.effectiveLibraryAddedDays).lowercase()} against " +
                    "${seedSpecs.size} ingredients"
            },
        )
        val rankingSnapshot = com.powerampstartradio.similarity.algorithms.GeoMeanSelector
            .computeRankingDetailedSnapshot(
                index = index,
                seeds = seedSpecs.map { it.embedding to it.weight },
                operator = effectiveQuerySpec.operator,
                refineSpec = effectiveQuerySpec.refineSpec,
                topK = if (
                    effectiveQuerySpec.textResultPlanner ==
                    FindMusicTextResultPlanner.VARIED_ALL_OF_DPP
                ) {
                    candidateDomain.candidateIdentityCount
                } else {
                    candidateDomain.rankedRowsForVisibleCount(
                        effectiveQuerySpec.resultLimit,
                    )
                },
                excludeTrackIds = equivalentSeedTrackIds.ifEmpty {
                    seedSpecs.mapNotNull { it.trackId }.toSet()
                },
                identityCatalog = identityCatalog,
                includedTrackIds = candidateDomain.eligibleTrackIds(),
                cancellationCheck = {
                    requestJob?.ensureActive()
                    if (!findMusicRequestGate.isCurrent(revision)) {
                        throw CancellationException("Superseded composed query")
                    }
                },
            )
        val ranking = rankingSnapshot.rows
        ensureCurrentFindMusicRequest(revision)
        var allOfQueuePlan: FindMusicAllOfQueuePlanEvidence? = null
        val plannedRanking: List<RankedFindMusicRow> = when (
            effectiveQuerySpec.textResultPlanner
        ) {
            FindMusicTextResultPlanner.CLOSEST -> ranking.mapIndexed { indexInRanking, row ->
                RankedFindMusicRow(indexInRanking + 1, row)
            }
            FindMusicTextResultPlanner.VARIED_ALL_OF_DPP -> {
                publishFindMusicLoadingStatus(
                    revision,
                    "Balancing All-of match with difference across the selected set",
                )
                val plan = FindMusicAllOfQueuePlanner.plan(
                    completeObjectiveRanking = ranking,
                    requestedResultCount = effectiveQuerySpec.resultLimit,
                    embeddingIndex = index,
                    cancellationCheck = {
                        requestJob?.ensureActive()
                        if (!findMusicRequestGate.isCurrent(revision)) {
                            throw CancellationException("Superseded composed query")
                        }
                    },
                )
                allOfQueuePlan = plan.evidence
                plan.selections.map { selected ->
                    RankedFindMusicRow(
                        objectiveRank = selected.originalAllOfObjectiveRank,
                        row = selected.row,
                    )
                }
            }
            FindMusicTextResultPlanner.VARIED_DPP ->
                error("Text Varied cannot plan a composed All-of result")
        }
        ensureCurrentFindMusicRequest(revision)

        val resultDb = EmbeddingDatabase.open(dbFile)
        val reduction: StableVisibleReduction<RankedFindMusicRow>
        val matches: List<TextSearchMatch>
        try {
            publishFindMusicLoadingStatus(
                revision,
                "Reading display metadata for ${effectiveQuerySpec.resultLimit} selected recordings",
            )
            val tracksById = HashMap<Long, EmbeddedTrack?>()
            reduction = StableVisibleResultReducer.reduce(
                rankedItems = plannedRanking,
                requestedVisibleCount = effectiveQuerySpec.resultLimit,
                identityOf = { ranked ->
                    identityCatalog.visibleResultIdentity(ranked.row.trackId)
                },
                isEligible = { ranked ->
                    tracksById.getOrPut(ranked.row.trackId) {
                        resultDb.getTrackById(ranked.row.trackId)
                    } != null
                },
            )
            matches = reduction.items.mapNotNull { ranked ->
                val row = ranked.row
                (tracksById[row.trackId] ?: resultDb.getTrackById(row.trackId))?.let { track ->
                    TextSearchMatch(
                        track = track,
                        similarity = row.objectiveScore,
                        identity = RadioSeedIdentity(
                            embeddedTrackId = track.id,
                            stableTrackSpanId = identityCatalog.stableTrackSpanId(track.id),
                        ),
                        rankingScore = row.objectiveScore,
                        objectiveRank = ranked.objectiveRank,
                        anchorPercentiles = row.anchorPercentiles,
                    )
                }
            }
        } finally {
            resultDb.close()
        }
        ensureCurrentFindMusicRequest(revision)

        if (!isCurrentReadOnlyFindMusicLibrary(
                filesDir,
                library,
                phase = "composed_before_publish",
            )
        ) {
            publishFindMusicError(
                revision,
                FindMusicSurface.COMPOSED,
                effectiveQuerySpec.displayLabel,
                "The active music library changed while this search was running. Run it again.",
                effectiveQuerySpec,
            )
            return
        }
        ensureCurrentFindMusicRequest(revision)

        val result = TextSearchResult(
            query = effectiveQuerySpec.displayLabel,
            matches = matches,
            error = if (matches.isEmpty()) "No matches found" else null,
            querySpec = effectiveQuerySpec,
            libraryBinding = identityCatalog.binding,
            providerGenerationId = library.activeDomain.binding.providerGenerationId,
            orderedActiveTrackIdsSha256 = library.activeDomain.orderedActiveTrackIdsSha256,
            activeTrackCount = library.activeDomain.activeTrackCount,
            objectiveRankingDomainCount = rankingSnapshot.objectiveRankingDomainCount,
            ingredientRankingDomainCount = rankingSnapshot.ingredientRankingDomainCount,
            stableResultReduction = reduction.toStableResultReductionEvidence(),
            allOfQueuePlan = allOfQueuePlan,
            preparedSeeds = seedSpecs.map { seed ->
                seed.copy(embedding = seed.embedding.copyOf())
            },
        )
        publishFindMusicResult(revision, FindMusicSurface.COMPOSED, result)
        if (matches.isNotEmpty()) saveRecentSearchIfCurrent(revision, effectiveQuerySpec)

        Log.i("MultiSeed", "MULTISEED_QUERY: ${FindMusicQuerySpecCodec.toJson(effectiveQuerySpec)}")
        Log.i(
            "MultiSeed",
            "MULTISEED_RESULTS: " + JSONArray().apply {
                matches.forEachIndexed { rank, match ->
                    put(JSONObject().apply {
                        put("rank", rank + 1)
                        put("track_id", match.track.id)
                        put("objective_score", match.rankingScore.toDouble())
                        put("planner", effectiveQuerySpec.textResultPlanner.wireName)
                        put("anchor_percentiles", JSONArray(match.anchorPercentiles))
                    })
                }
            },
        )
    }

    private fun bindAnchorsToActiveDomain(
        resolution: FindMusicAnchorBindingResult.Success,
        identityCatalog: StableTrackIdentityCatalog,
        activeDomain: ActiveRecommendationDomain,
    ): FindMusicAnchorBindingResult {
        val unresolved = mutableListOf<FindMusicSongAnchor>()
        val activeEquivalentTrackIds = linkedSetOf<Long>()
        val activeAnchors = resolution.querySpec.songSeeds.map { anchor ->
            if (anchor.weight <= 0f) return@map anchor

            val equivalentIds = anchor.stableTrackSpanId?.let { stableId ->
                (identityCatalog.resolveStable(stableId) as? StableTrackIdentityResolution.Resolved)
                    ?.allEquivalentTrackIds
            }.orEmpty().ifEmpty { listOf(anchor.trackId) }
            val activeIds = equivalentIds.filter(activeDomain::containsActiveTrack)
            val activeTrackId = activeIds.firstOrNull()
            if (activeTrackId == null) {
                unresolved += anchor
                anchor
            } else {
                activeEquivalentTrackIds += identityCatalog
                    .equivalentVisibleTrackIds(activeTrackId)
                    .filter(activeDomain::containsActiveTrack)
                anchor.copy(
                    trackId = activeDomain.activeIdentityRepresentativeTrackId(activeTrackId),
                )
            }
        }
        if (unresolved.isNotEmpty()) {
            return FindMusicAnchorBindingResult.Failure(
                message = "${if (unresolved.size == 1) "This recording is" else "These recordings are"} " +
                    "not in the current Poweramp library. Choose " +
                    "${if (unresolved.size == 1) "it" else "them"} again: " +
                    unresolved.joinToString(", ") { it.displayLabel } + ".",
                unresolvedAnchors = unresolved,
                diagnosticDetail = "Resolved anchor identities are absent from the active " +
                    "recommendation domain: " + unresolved.joinToString(", ") { it.displayLabel },
            )
        }
        return FindMusicAnchorBindingResult.Success(
            querySpec = resolution.querySpec.copy(songSeeds = activeAnchors),
            equivalentTrackIdsToExclude = activeEquivalentTrackIds,
        )
    }

    private fun validateComposedQuery(querySpec: FindMusicQuerySpec): String? {
        validateQueryContract(querySpec)?.let { return it }
        val everyWeight = querySpec.textIngredients.map { it.weight } +
            querySpec.songSeeds.map { it.weight }
        if (everyWeight.any { !it.isFinite() || it < 0f || it > 1f }) {
            return "Every ingredient weight must be between 0% and 100%"
        }
        if (querySpec.textIngredients.any { it.query.isBlank() && it.weight > 0f }) {
            return "Blank descriptions cannot contribute to a query"
        }
        val activeWeights = buildList {
            querySpec.textIngredients.filter { it.query.isNotBlank() && it.weight > 0f }
                .forEach { add(it.weight) }
            querySpec.songSeeds.filter { it.weight > 0f }.forEach { add(it.weight) }
        }
        if (activeWeights.isEmpty() || activeWeights.all { it <= 0f }) {
            return "Add at least one active text or song ingredient"
        }
        val totalWeight = activeWeights.sum()
        if (abs(totalWeight - 1f) > 0.005f) {
            return "Active ingredient shares total ${(totalWeight * 100).toInt()}%. Adjust them to exactly 100% before searching."
        }
        val duplicateIds = querySpec.songSeeds.filter { it.weight > 0f }
            .groupBy { it.trackId }
            .filterValues { it.size > 1 }
        if (duplicateIds.isNotEmpty()) {
            val labels = duplicateIds.values.map { it.first().displayLabel }.joinToString(", ")
            return "The same song ingredient appears more than once: $labels. Keep one copy with the intended sign and weight."
        }
        return null
    }

    private fun validateQueryContract(querySpec: FindMusicQuerySpec): String? =
        validateFindMusicQueryContract(querySpec)

    // --- Indexing actions ---

    fun checkUnindexedTracks() {
        if (_unindexedCount.value == -1) return
        Log.i("MainViewModel", "Starting unindexed track check")
        _unindexedCount.value = -1 // signal "checking" to UI
        startUnindexedTrackCheck(providerSnapshot = null)
    }

    private fun startUnindexedTrackCheck(
        providerSnapshot: V2ProviderPathGroupSnapshot?,
    ) {
        val app = getApplication<Application>()

        val deferred = viewModelScope.async(Dispatchers.IO) {
            V2ProcessLibraryInspectionCoordinator.inspect {
                val active = V2LibraryDatabaseResolver.requirePublished(app.filesDir) { progress ->
                    val status = activeMusicIndexHashStatus(progress)
                    _unindexedCheckStatus.value = status
                    IndexingViewModel.detectionStatus.value = status
                }
                if (!PowerampHelper.canAccessData(app)) {
                    throw IllegalStateException("Poweramp library access is not granted")
                }
                val exactProviderSnapshot = providerSnapshot
                    ?: V2PowerampProviderSnapshotAcquirer(app).acquireBlocking {
                            completedRows,
                            totalRows,
                        ->
                        val status = powerampLibraryReadProgressText(completedRows, totalRows)
                        _unindexedCheckStatus.value = status
                        IndexingViewModel.detectionStatus.value = status
                    }
                performUnindexedTrackDetection(active, exactProviderSnapshot)
            }
        }
        // Expose so IndexingViewModel can await if user opens Manage Tracks mid-check
        IndexingViewModel.pendingDetection = deferred

        viewModelScope.launch(Dispatchers.IO) {
            try {
                val result = deferred.await()
                publishUnindexedTrackCount(app, result)
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (e: Exception) {
                _unindexedCount.value = -3
                Log.e("MainViewModel", "Unindexed track check failed", e)
            } finally {
                if (IndexingViewModel.pendingDetection === deferred) {
                    IndexingViewModel.pendingDetection = null
                }
                IndexingViewModel.detectionStatus.value = null
                _unindexedCheckStatus.value = null
            }
        }
    }

    private fun publishUnindexedTrackCount(
        app: Application,
        result: IndexingViewModel.SharedDetectionResult,
    ) {
        val currentActive = V2LibraryDatabaseResolver.requirePublished(app.filesDir)
        require(currentActive.manifest.generationId == result.databaseGeneration) {
            "Active embeddings generation changed during the unindexed check"
        }
        // Resolve the same path/span envelopes used by Manage Tracks. Numeric provider IDs are
        // transient locators and never filter V2's Settings count.
        val exclusions = V2TrackExclusionRepository(app).resolveAndMigrate(result.tracks)
        val attentionHistory = V2IndexingAttentionHistorySource(app).load()
        val visible = V2IndexingReadinessPolicy.readyTrackIds(
            tracks = result.tracks,
            exclusions = exclusions,
            attentionHistory = attentionHistory,
        ).size
        val identity = V2UnindexedCountCacheIdentity(
            databaseGeneration = result.databaseGeneration,
            providerGeneration = result.providerGeneration,
            exclusionsFingerprint = exclusions.persistedFingerprint,
            attentionFingerprint = attentionHistory.fingerprint,
            detectionPolicyId = V2UnindexedCountCachePolicy.DETECTION_POLICY_ID,
        )
        setUnindexedCount(visible, identity)
        Log.i("MainViewModel", "Unindexed track check complete: $visible visible tracks")
    }

    /** Caller owns the process-wide full-library inspection admission. */
    private fun performUnindexedTrackDetection(
        active: V2ResolvedActiveIndexGeneration,
        providerSnapshot: V2ProviderPathGroupSnapshot,
    ): IndexingViewModel.SharedDetectionResult {
        val databaseGeneration = active.manifest.generationId
        val providerGeneration = requireNotNull(providerSnapshot.libraryGeneration) {
            "Poweramp snapshot has no complete library generation"
        }
        val db = EmbeddingDatabase.open(active.databaseFile)
        val tracks = try {
            NewTrackDetector(db).findUnindexedTracks(providerSnapshot) { status ->
                _unindexedCheckStatus.value = status
                IndexingViewModel.detectionStatus.value = status
            }
        } finally {
            db.close()
        }
        val sorted = tracks.sortedByDescending { it.durationMs }
        return IndexingViewModel.SharedDetectionResult(
            tracks = sorted,
            databaseGeneration = databaseGeneration,
            providerGeneration = providerGeneration,
        ).also { IndexingViewModel.offerCompletedDetectionHandoff(it) }
    }

    private fun setUnindexedCount(count: Int, identity: V2UnindexedCountCacheIdentity) {
        _unindexedCount.value = count
        prefs.edit()
            .putInt("unindexed_count", count)
            .putString("unindexed_count_database_generation", identity.databaseGeneration)
            .putString("unindexed_count_provider_generation", identity.providerGeneration)
            .putString("unindexed_count_exclusions_fingerprint", identity.exclusionsFingerprint)
            .putString("unindexed_count_attention_fingerprint", identity.attentionFingerprint)
            .putString("unindexed_count_detection_policy", identity.detectionPolicyId)
            .remove("unindexed_last_checked_ms")
            .remove("unindexed_count_db_fingerprint")
            .apply()
    }

    fun checkModels() {
        if (!modelCheckRunning.compareAndSet(false, true)) return
        _modelsLoading.value = true
        _modelsLoadingStatus.value = "Opening the active music-index manifest"
        viewModelScope.launch(Dispatchers.IO) {
            try {
                checkModelsOnWorker()
            } finally {
                _modelsLoadingStatus.value = null
                _modelsLoading.value = false
                modelCheckRunning.set(false)
            }
        }
    }

    fun refreshSettingsStatus() {
        checkModels()
    }

    private fun checkModelsOnWorker() {
        val filesDir = getApplication<Application>().filesDir
        val resolutionResult = runCatching {
            V2LibraryDatabaseResolver.resolveOrNull(filesDir) { progress ->
                _modelsLoadingStatus.value = activeMusicIndexHashStatus(progress)
            }
        }
        resolutionResult.exceptionOrNull()?.let { error ->
            Log.e("MainViewModel", "Active library validation failed during model check", error)
        }
        val resolution = resolutionResult.getOrNull()
        val active = resolution?.activeGeneration
        val mertFile = File(filesDir, "mert.tflite")
        val clamp3AudioFile = File(filesDir, "clamp3_audio.tflite")
        val clamp3TextFile = File(filesDir, "clamp3_text.tflite")
        val tokenizerModelFile = File(filesDir, "sentencepiece.bpe.model")

        fun fileSizeMb(file: File?): String? = file?.takeIf(File::isFile)?.let {
            "%.1f MB".format(it.length() / (1024.0 * 1024.0))
        }

        val expectedAudio = active?.manifest?.receiptEmbeddingSpec?.modelArtifactSha256
        val expectedText = active?.manifest?.textRetrievalSpec
        val installedModels = runCatching {
            V2CurrentModelPolicyResolver.resolveInstalled(filesDir) { progress ->
                _modelsLoadingStatus.value = exactHashProgressText(
                    subject = "changed indexing file ${progress.fileOrdinal} of " +
                        "${progress.fileCount}: ${progress.filename}",
                    completedBytes = progress.completedBytes,
                    totalBytes = progress.totalBytes,
                )
            }
        }.onFailure { error ->
            Log.e("MainViewModel", "Installed model identity could not be read", error)
        }.getOrNull()
        val mertHash = installedModels?.sha256ByName?.get("mert.tflite")
        val clampAudioHash = installedModels?.sha256ByName?.get("clamp3_audio.tflite")
        val clampTextHash = installedModels?.sha256ByName?.get("clamp3_text.tflite")
        val tokenizerHash = installedModels?.sha256ByName?.get("sentencepiece.bpe.model")
        _modelsLoadingStatus.value = "Matching the saved model identity to the active music index"
        val expectedMertHash = expectedAudio?.get("mert")
        val expectedClampAudioHash = expectedAudio?.get("clamp3_audio")
        val expectedClampTextHash = expectedText?.textModelSha256
        val expectedTokenizerHash = expectedText?.tokenizerModelSha256
            ?: V2IndexingWorkPolicy.TEXT_TOKENIZER_MODEL_SHA256
        val mertReady = mertHash != null && expectedMertHash != null &&
            mertHash == expectedMertHash
        val clampAudioReady = clampAudioHash != null && expectedClampAudioHash != null &&
            clampAudioHash == expectedClampAudioHash
        val clampTextReady = clampTextHash != null && expectedClampTextHash != null &&
            clampTextHash == expectedClampTextHash
        val tokenizerReady = tokenizerHash == V2IndexingWorkPolicy.TEXT_TOKENIZER_MODEL_SHA256 &&
            tokenizerHash == expectedTokenizerHash
        _hasModels.value = mertReady && clampAudioReady && clampTextReady && tokenizerReady

        fun modelDetail(
            file: File,
            ready: Boolean,
            hash: String?,
            expectedHash: String?,
            purpose: String,
        ): String = when {
            !file.isFile -> "$purpose; exact file missing"
            hash == null -> "$purpose; SHA-256 could not be read"
            expectedHash == null -> "$purpose; active index has no model identity"
            ready -> "$purpose; SHA-256 matches the active music index"
            else -> "$purpose; SHA-256 does not match this music index"
        }

        val dbFile = resolution?.databaseFile
        val embFile = active?.embeddingFile
        val graphFile = active?.graphFile
        val dbDetail = when {
            resolutionResult.isFailure ->
                "The active music index could not be read. Import a music index."
            active != null ->
                "${String.format(
                    Locale.getDefault(),
                    "%,d",
                    active.manifest.embeddingCoverage.totalTrackCount,
                )} tracks in the current music index"
            else -> "A music index is required"
        }

        _fileStatuses.value = listOf(
            AppFileStatus("database", dbFile?.isFile == true, fileSizeMb(dbFile), dbDetail),
            AppFileStatus(
                "search index",
                embFile?.isFile == true,
                fileSizeMb(embFile),
                if (embFile?.isFile == true) "Fast search index matches this music index"
                else "No matching fast search index for this music index",
            ),
            AppFileStatus(
                "similarity graph",
                graphFile?.isFile == true,
                fileSizeMb(graphFile),
                if (graphFile?.isFile == true) "Similarity-path graph matches this music index"
                else "Graph Explorer is unavailable; its similarity graph will be rebuilt when needed",
            ),
            AppFileStatus("mert.tflite", mertReady, fileSizeMb(mertFile),
                modelDetail(
                    mertFile,
                    mertReady,
                    mertHash,
                    expectedMertHash,
                    "MERT audio feature model",
                )),
            AppFileStatus("clamp3_audio.tflite", clampAudioReady, fileSizeMb(clamp3AudioFile),
                modelDetail(
                    clamp3AudioFile,
                    clampAudioReady,
                    clampAudioHash,
                    expectedClampAudioHash,
                    "CLaMP3 audio encoder",
                )),
            AppFileStatus("clamp3_text.tflite", clampTextReady, fileSizeMb(clamp3TextFile),
                modelDetail(
                    clamp3TextFile,
                    clampTextReady,
                    clampTextHash,
                    expectedClampTextHash,
                    "CLaMP3 text encoder",
                )),
            AppFileStatus(
                "sentencepiece.bpe.model",
                tokenizerReady,
                fileSizeMb(tokenizerModelFile),
                modelDetail(
                    tokenizerModelFile,
                    tokenizerReady,
                    tokenizerHash,
                    expectedTokenizerHash,
                    "Official CLaMP3 SentencePiece tokenizer",
                ),
            ),
        )
    }

    fun resetToDefaults() {
        val defaults = RadioConfig()
        val findMusicDefaults = FindMusicEditorPolicy.reset(
            FindMusicEditorSnapshot(
                textIngredients = _textIngredients.value,
                songSeeds = _songSeeds.value,
                operator = _findMusicOperator.value,
                resultLimit = _numTracks.value,
            ),
        )
        clearFindMusicResults()
        _textIngredients.value = findMusicDefaults.textIngredients
        _songSeeds.value = findMusicDefaults.songSeeds
        setNumTracks(defaults.numTracks)
        setLibraryAddedDays(defaults.effectiveLibraryAddedDays)
        setSelectionMode(defaults.selectionMode)
        setDriftEnabled(defaults.driftEnabled)
        setDriftMode(defaults.driftMode)
        setAnchorStrength(defaults.anchorStrength)
        setWalkRestartAlpha(defaults.walkRestartAlpha)
        setAnchorDecay(defaults.anchorDecay)
        setAnchorHalfLifeTracks(defaults.anchorHalfLifeTracks)
        setMomentumBeta(defaults.momentumBeta)
        setDiversityLambda(defaults.diversityLambda)
        setMmrCandidatePoolFraction(defaults.mmrCandidatePoolFraction)
        setDppFixedCandidatePoolFraction(defaults.dppFixedCandidatePoolFraction)
        setDppUsesCertifiedFullDomain(defaults.dppUsesCertifiedFullDomain)
        setDppQualityExponent(defaults.dppQualityExponent)
        setShuffleSeed(defaults.shuffleSeed)
        setArtistLimitsEnabled(defaults.artistLimitsEnabled)
        setMaxPerArtist(defaults.maxPerArtist)
        setMinArtistSpacing(defaults.minArtistSpacing)
        setFindMusicOperator(findMusicDefaults.operator)
        _findMusicRefinePrimaryIngredientIndex.value = 0
        setFindMusicRefineNeighborhood(FindMusicRefineNeighborhood.TOP_1_PERCENT)
        persistFindMusicTextResultPlanner(FindMusicTextResultPlanner.CLOSEST)
    }

    fun checkPermission() {
        _permissionLoading.value = true
        viewModelScope.launch(Dispatchers.IO) {
            try {
                _hasPermission.value = PowerampHelper.canAccessData(getApplication())
            } finally {
                _permissionLoading.value = false
            }
        }
    }

    fun requestPermission() {
        PowerampHelper.requestDataPermission(getApplication())
    }

    private fun indexingLibraryConflictReason(): String? {
        if (IndexingService.state.value !is IndexingService.IndexingState.Idle) {
            return "Finish or discard the current indexing job before changing the music index."
        }
        val pointer = runCatching {
            V2ActiveIndexingJobPointer(getApplication<Application>().filesDir).read()
        }
        if (pointer.isFailure) {
            return "The saved indexing job could not be read. Open On-device indexing before changing the music index."
        }
        return pointer.getOrNull()?.let {
            "Finish or discard the saved indexing job before changing the music index."
        }
    }

    private fun updateLibraryControlsBlockedReason() {
        _libraryControlsBlockedReason.value = when {
            _libraryLifecycleBusy.value -> "A music-index update is already running."
            else -> indexingLibraryConflictReason()
        }
    }

    private fun beginLibraryLifecycle(onBlocked: (String) -> Unit): Boolean {
        if (!databaseLifecycleBusy.compareAndSet(false, true)) {
            onBlocked("A music-index update is already running.")
            updateLibraryControlsBlockedReason()
            return false
        }
        if (!MusicIndexMutationAdmission.process.tryAcquire(libraryLifecycleMutationOwner)) {
            databaseLifecycleBusy.set(false)
            onBlocked("A music-index update is already running.")
            updateLibraryControlsBlockedReason()
            return false
        }
        _libraryLifecycleBusy.value = true
        // Indexing cannot create an intent or claim the active pointer while this admission is
        // owned, so this check and the complete mutation lifecycle are one atomic start decision.
        val racedIndexingJob = indexingLibraryConflictReason()
        if (racedIndexingJob != null) {
            check(MusicIndexMutationAdmission.process.release(libraryLifecycleMutationOwner)) {
                "library lifecycle lost music-index mutation admission"
            }
            databaseLifecycleBusy.set(false)
            _libraryLifecycleBusy.value = false
            onBlocked(racedIndexingJob)
            updateLibraryControlsBlockedReason()
            return false
        }
        updateLibraryControlsBlockedReason()
        return true
    }

    private fun finishLibraryLifecycle() {
        check(MusicIndexMutationAdmission.process.release(libraryLifecycleMutationOwner)) {
            "library lifecycle completed without owning music-index mutation admission"
        }
        databaseLifecycleBusy.set(false)
        _libraryLifecycleBusy.value = false
        updateLibraryControlsBlockedReason()
        refreshSessionReplayEligibility()
        refreshDisplayedQueueEligibility()
    }

    /**
     * Import a database from the given URI asynchronously.
     * Shows progress in importStatus, then refreshes everything.
     */
    fun importDatabase(uri: Uri) {
        if (V2LibraryDatabaseResolver.hasPublishedPointer(
                getApplication<Application>().filesDir,
            )
        ) {
            _importError.value = "A music index is already loaded."
            return
        }
        if (!beginLibraryLifecycle { _importError.value = it }) return
        _importError.value = null
        _musicIndexUpdateResult.value = null
        _importStatus.value = "Opening the selected music-index file"
        viewModelScope.launch(Dispatchers.IO) {
            val activeBefore = activeGenerationIdOrNull()
            try {
                val t0 = System.nanoTime()
                val app = getApplication<Application>()
                Log.i("MainViewModel", "Starting immutable bootstrap import from $uri")
                _importStatus.value = "Hashing four required model and tokenizer files..."
                val result = V2BootstrapGenerationImporter(app).import(uri) { status ->
                    _importStatus.value = status
                }

                invalidateAllPreviewsForContextChange()
                rankIndexSnapshot = null
                invalidateTextEmbeddingIndex()
                IndexingViewModel.invalidateCache()
                publishDatabaseInfo(readDatabaseInfo(
                    V2ResolvedLibraryDatabase(
                        databaseFile = result.generation.databaseFile,
                        activeGeneration = result.generation,
                    ),
                ))
                checkModels()
                val totalMs = (System.nanoTime() - t0) / 1_000_000
                Log.i(
                    "MainViewModel",
                    "Immutable import activated ${result.generation.manifest.generationId} " +
                        "(${result.generation.manifest.trackCount} tracks, ${totalMs}ms)",
                )
                _importStatus.value = null
                checkUnindexedTracks()
            } catch (cancelled: CancellationException) {
                _importStatus.value = null
                throw cancelled
            } catch (e: Exception) {
                Log.e("MainViewModel", "Import failed", e)
                _importStatus.value = null
                val activeChanged = activeGenerationIdOrNull()?.let { it != activeBefore } == true
                _importError.value = if (activeChanged) {
                    "The imported music index is active, but this screen could not refresh. " +
                        "Reopen Settings to refresh it."
                } else {
                    "The selected music index could not be loaded. No music index was activated."
                }
                refreshDatabaseInfo()
            } finally {
                finishLibraryLifecycle()
            }
        }
    }

    fun canSelectServerMergeFile(): Boolean {
        if (AudioLibraryPermission.isGranted(getApplication())) return true
        _importError.value = AudioLibraryPermission.DENIED_MESSAGE
        return false
    }

    fun mergeServerDatabase(uri: Uri) {
        if (!canSelectServerMergeFile()) return
        if (!V2LibraryDatabaseResolver.hasPublishedPointer(
                getApplication<Application>().filesDir,
            )
        ) {
            _importError.value = "Import a music index before merging server embeddings."
            return
        }
        if (!beginLibraryLifecycle { _importError.value = it }) return
        _importError.value = null
        publishServerMergeProgress(
            ServerMergeProgressState(
                phase = ServerMergeProgressPhase.PREPARING,
                detail = "Preparing the server merge",
            ),
        )
        viewModelScope.launch(Dispatchers.IO) {
            val activeBefore = activeGenerationIdOrNull()
            val activeTrackCountBefore = runCatching {
                V2IndexGenerationReader.requireActive(
                    getApplication<Application>().filesDir,
                ).manifest.trackCount
            }.getOrDefault(0)
            val selectedDocument = selectedMusicIndexDocument(uri)
            var mutationReserved = false
            try {
                val startedNs = System.nanoTime()
                val app = getApplication<Application>()
                persistServerMergeResult(
                    "Merge started · ${selectedDocument.summary}\n" +
                        "Completion has not yet been recorded.",
                )
                publishServerMergeProgress(
                    ServerMergeProgressState(
                        phase = ServerMergeProgressPhase.RELEASING_RECOMMENDATION_RESOURCES,
                        detail = "Releasing recommendation resources before changing the music index",
                    ),
                )
                prepareForMusicIndexMutation()
                mutationReserved = true
                publishServerMergeProgress(
                    ServerMergeProgressState(
                        phase = ServerMergeProgressPhase.WAITING_FOR_LIBRARY_INSPECTION,
                        detail = "Waiting for the current Poweramp library comparison to finish",
                    ),
                )
                val result = V2ServerBundleMerger(app).merge(uri) { progress ->
                    publishServerMergeProgress(
                        ServerMergeProgressState(
                            phase = ServerMergeProgressPhase.MERGING,
                            detail = progress.detail,
                            completedUnits = progress.completedUnits,
                            totalUnits = progress.totalUnits,
                            mergeStage = progress.stage.name,
                        ),
                    )
                }
                if (!result.noOp) {
                    invalidateAllPreviewsForContextChange()
                    rankIndexSnapshot = null
                    invalidateTextEmbeddingIndex()
                    IndexingViewModel.invalidateCache()
                    invalidateUnindexedCountEvidence()
                }
                publishDatabaseInfo(
                    readDatabaseInfo(
                        resolution = V2ResolvedLibraryDatabase(
                            databaseFile = result.generation.databaseFile,
                            activeGeneration = result.generation,
                        ),
                        activeCatalogOverride = result.activeCatalog,
                    ),
                )
                val elapsedMs = (System.nanoTime() - startedNs) / 1_000_000L
                persistServerMergeResult(
                    serverMergeResultText(
                        result = result,
                        selectedDocument = selectedDocument,
                        activeTrackCountBefore = activeTrackCountBefore,
                        elapsedMs = elapsedMs,
                    ),
                )
                publishServerMergeProgress(null)
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
                    "MainViewModel",
                    "Server index merge completed: added=${result.addedTrackCount} " +
                        "noOp=${result.noOp} elapsedMs=$elapsedMs " +
                        "bundle=${result.sourceValidation.bundleId} " +
                        "dispositions=[$dispositionCounts] " +
                        "matchEvidence=[$matchEvidenceCounts]",
                )
            } catch (cancelled: CancellationException) {
                publishServerMergeProgress(null)
                val activeChanged = activeGenerationIdOrNull()?.let { it != activeBefore } == true
                persistServerMergeResult(
                    "Last merge was interrupted · ${selectedDocument.summary}\n" +
                        if (activeChanged) {
                            "A new music index was published before interruption. Reopen Settings " +
                                "to refresh its details."
                        } else {
                            "Current music index unchanged."
                        },
                )
                throw cancelled
            } catch (error: Exception) {
                Log.e("MainViewModel", "Server index merge failed", error)
                publishServerMergeProgress(null)
                val activeChanged = activeGenerationIdOrNull()?.let { it != activeBefore } == true
                _importError.value = if (activeChanged) {
                    "The merged music index is active, but this screen could not refresh. " +
                        "Reopen Settings to refresh it."
                } else {
                    val detail = error.message?.trim()?.takeIf { it.isNotEmpty() }
                        ?: "The selected server bundle could not be merged"
                    "$detail. The current music index was not changed."
                }
                persistServerMergeResult(
                    if (activeChanged) {
                        "Last merge published a new index but refresh failed · " +
                            "${selectedDocument.summary}\n${_importError.value}"
                    } else {
                        "Last merge failed · ${selectedDocument.summary}\n" +
                            _importError.value
                    },
                )
                refreshDatabaseInfo()
            } finally {
                if (mutationReserved ||
                    RecommendationWorkAdmission.isReservedBy(musicIndexMutationOwner)
                ) {
                    releaseMusicIndexMutationReservation()
                }
                finishLibraryLifecycle()
            }
        }
    }

    private fun publishServerMergeProgress(progress: ServerMergeProgressState?) {
        _serverMergeProgress.value = progress
        // Existing Settings collectors retain their String contract until the determinate UI is
        // wired to serverMergeProgress.
        _importStatus.value = progress?.detail
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

    private fun selectedMusicIndexDocument(uri: Uri): SelectedMusicIndexDocument {
        var displayName: String? = null
        var byteCount: Long? = null
        runCatching {
            getApplication<Application>().contentResolver.query(
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
            Log.w("MainViewModel", "Could not read selected server-bundle identity", error)
        }
        return SelectedMusicIndexDocument(
            displayName = displayName?.takeIf { it.isNotBlank() }
                ?: uri.lastPathSegment?.substringAfterLast('/')?.takeIf { it.isNotBlank() }
                ?: "selected file",
            byteCount = byteCount,
        )
    }

    private fun persistServerMergeResult(result: String) {
        _musicIndexUpdateResult.value = result
        prefs.edit().putString(LAST_SERVER_MERGE_RESULT_PREF, result).apply()
    }

    private fun invalidateUnindexedCountEvidence() {
        _unindexedCount.value = -2
        prefs.edit()
            .remove("unindexed_count")
            .remove("unindexed_count_database_generation")
            .remove("unindexed_count_provider_generation")
            .remove("unindexed_count_exclusions_fingerprint")
            .remove("unindexed_count_attention_fingerprint")
            .remove("unindexed_count_detection_policy")
            .apply()
    }

    private fun serverMergeResultText(
        result: V2ServerMergeResult,
        selectedDocument: SelectedMusicIndexDocument,
        activeTrackCountBefore: Int,
        elapsedMs: Long,
    ): String {
        val counts = result.rowOutcomes.groupingBy { it.disposition }.eachCount()
        val parts = mutableListOf<String>()
        parts += "${result.addedTrackCount} added"
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

    private fun activeGenerationIdOrNull(): String? = runCatching {
        if (V2LibraryDatabaseResolver.hasPublishedPointer(getApplication<Application>().filesDir)) {
            V2IndexGenerationReader.requireActive(getApplication<Application>().filesDir)
                .manifest.generationId
        } else {
            null
        }
    }.getOrNull()

    /**
     * A cold launch intentionally publishes database-only readiness. The first real radio request
     * may then establish the exact provider binding in RadioService; adopt that same-process
     * result when its history entry arrives so replay becomes available without another scan.
     */
    private fun adoptProcessCatalogIfAvailable() {
        val published = _databaseInfo.value ?: return
        val app = getApplication<Application>()
        val resolution = runCatching {
            V2LibraryDatabaseResolver.resolveOrNull(app.filesDir)
        }.onFailure { error ->
            Log.w("MainViewModel", "Could not resolve the music index for catalog handoff", error)
        }.getOrNull() ?: return
        val active = resolution.activeGeneration
        if (published.generationId != active.manifest.generationId) return
        val catalog = RadioService.processActiveLibraryCatalog(
            active.toRadioGenerationToken(),
        ) ?: return
        if (published.providerGenerationId == catalog.generationBinding.providerGenerationId &&
            verifiedReplayLibraryBinding?.generation == active.toRadioGenerationToken()
        ) {
            return
        }

        val verified = runCatching {
            readDatabaseInfo(
                resolution = resolution,
                activeCatalogOverride = catalog,
                reconcileProviderIfMissing = false,
            )
        }.onFailure { error ->
            Log.w("MainViewModel", "Could not adopt the radio library catalog", error)
        }.getOrNull() ?: return
        if (activeGenerationIdOrNull() != verified.info.generationId) return

        publishDatabaseInfo(verified)
        if (activeGenerationIdOrNull() != verified.info.generationId) {
            Log.i(
                "MainViewModel",
                "Music index changed during the radio catalog handoff; refreshing",
            )
            refreshDatabaseInfo()
            return
        }
        Log.i(
            "MainViewModel",
            "Adopted radio library catalog for ${catalog.activeTrackIds.size} active tracks",
        )
    }

    private data class VerifiedDatabaseInfo(
        val info: DatabaseInfo,
        val replayLibraryBinding: VerifiedReplayLibraryBinding?,
        val activeCatalog: V2ActiveLibraryCatalog?,
    )

    private fun hasTrustedCurrentDatabaseInfo(activeGeneration: String?): Boolean =
        V2DatabaseReadySnapshotPolicy.isCurrent(
            publishedGeneration = _databaseInfo.value?.generationId,
            activeGeneration = activeGeneration,
            hasExactLibraryBinding = verifiedReplayLibraryBinding?.generation?.generationId ==
                activeGeneration,
        )

    private fun readCachedReadySnapshot(
        resolution: V2ResolvedLibraryDatabase,
    ): VerifiedDatabaseInfo? {
        val activeGeneration = resolution.activeGeneration
        val activeToken = activeGeneration.toRadioGenerationToken()
        val processCatalog = RadioService.processActiveLibraryCatalog(activeToken)
        val catalog = processCatalog ?: V2ActiveLibraryCatalogStore(
            getApplication<Application>().filesDir,
        ).read(activeGeneration)?.also { restored ->
            Log.i(
                "MainViewModel",
                "Durable active-library cache=hit generation=" +
                    "${restored.generationBinding.databaseGenerationId} " +
                    "tracks=${restored.activeTrackIds.size}",
            )
        } ?: return null
        if (!V2DatabaseReadySnapshotPolicy.canBootstrap(
                cachedDatabaseGeneration = catalog.generationBinding.databaseGenerationId,
                activeGeneration = activeGeneration.manifest.generationId,
            )
        ) {
            return null
        }
        return readDatabaseInfo(
            resolution = resolution,
            activeCatalogOverride = catalog,
        )
    }

    fun refreshDatabaseInfo() {
        if (_importStatus.value != null) return
        when (databaseRefreshGate.admit(recommendationResourcesBlocked())) {
            V2DatabaseRefreshAdmission.START -> Unit
            V2DatabaseRefreshAdmission.JOIN_RUNNING -> {
                Log.d("MainViewModel", "Joining the active database refresh")
                return
            }
            V2DatabaseRefreshAdmission.DEFER_UNTIL_RESOURCES_ARE_FREE -> return
        }
        logMemoryPhase("database_refresh_start")
        viewModelScope.launch(Dispatchers.IO) {
            val refreshStartedNs = System.nanoTime()
            var refreshPass = 0
            var waitingForDeferredRefresh = false
            val activeAtAdmission = activeGenerationIdOrNull()
            val trustedAtAdmission = hasTrustedCurrentDatabaseInfo(activeAtAdmission)
            if (trustedAtAdmission) {
                _databaseLoading.value = false
                Log.i(
                    "MainViewModel",
                    "Keeping generation $activeAtAdmission ready during background library " +
                        "reconciliation",
                )
            } else {
                _databaseLoading.value = true
                verifiedReplayLibraryBinding = null
                refreshSessionReplayEligibility()
                refreshDisplayedQueueEligibility()
            }
            try {
                do {
                    refreshPass += 1
                    val passStartedNs = System.nanoTime()
                    try {
                        _databaseVerificationStatus.value =
                            "Opening the saved music-index manifest and file receipts"
                        val resolutionStartedNs = System.nanoTime()
                        val resolution = V2LibraryDatabaseResolver.resolveOrNull(
                            getApplication<Application>().filesDir,
                        ) { progress ->
                            _databaseVerificationStatus.value =
                                activeMusicIndexHashStatus(progress)
                        }
                        val resolutionMs = elapsedMs(resolutionStartedNs)
                        Log.i(
                            "MainViewModel",
                            "Active database generation resolved in ${resolutionMs}ms",
                        )
                        val resolvedGeneration = resolution?.activeGeneration?.manifest?.generationId
                        val activeAfterResolution = activeGenerationIdOrNull()
                        if (V2DatabaseRefreshGenerationPolicy.needsAnotherPass(
                                validatedGeneration = resolvedGeneration,
                                activeGeneration = activeAfterResolution,
                            )
                        ) {
                            if (recommendationResourcesBlocked()) {
                                databaseRefreshGate.deferUntilResourcesAreFree()
                                waitingForDeferredRefresh = true
                                break
                            }
                            continue
                        }

                        if (resolution != null) {
                            val bootstrapStartedNs = System.nanoTime()
                            val bootstrap = runCatching {
                                readCachedReadySnapshot(resolution)
                            }.onFailure { error ->
                                Log.w(
                                    "MainViewModel",
                                    "Cached active-library snapshot could not be published",
                                    error,
                                )
                            }.getOrNull()
                            if (bootstrap != null) {
                                val activeBeforeBootstrapPublication = activeGenerationIdOrNull()
                                if (V2DatabaseRefreshGenerationPolicy.needsAnotherPass(
                                        validatedGeneration = bootstrap.info.generationId,
                                        activeGeneration = activeBeforeBootstrapPublication,
                                    )
                                ) {
                                    if (recommendationResourcesBlocked()) {
                                        databaseRefreshGate.deferUntilResourcesAreFree()
                                        waitingForDeferredRefresh = true
                                        break
                                    }
                                    continue
                                }
                                publishDatabaseInfo(bootstrap)
                                val activeAfterBootstrapPublication = activeGenerationIdOrNull()
                                if (V2DatabaseRefreshGenerationPolicy.needsAnotherPass(
                                        validatedGeneration = bootstrap.info.generationId,
                                        activeGeneration = activeAfterBootstrapPublication,
                                    )
                                ) {
                                    _databaseLoading.value = true
                                    if (recommendationResourcesBlocked()) {
                                        databaseRefreshGate.deferUntilResourcesAreFree()
                                        waitingForDeferredRefresh = true
                                        break
                                    }
                                    continue
                                }
                                _databaseLoading.value = false
                                refreshSessionReplayEligibility()
                                refreshDisplayedQueueEligibility()
                                Log.i(
                                    "MainViewModel",
                                    "Cached active-library snapshot published ready in " +
                                        "${elapsedMs(bootstrapStartedNs)}ms; " +
                                        "generation=${bootstrap.info.generationId} " +
                                        "tracks=${bootstrap.info.activeTrackCount}",
                                )
                                break
                            }
                        }

                        val next = resolution?.let { resolved ->
                            readDatabaseInfo(
                                resolution = resolved,
                                reconcileProviderIfMissing = false,
                            ) { status ->
                                _databaseVerificationStatus.value = status
                            }
                        }
                        val activeGenerationNow = activeGenerationIdOrNull()
                        if (V2DatabaseRefreshGenerationPolicy.needsAnotherPass(
                                validatedGeneration = next?.info?.generationId,
                                activeGeneration = activeGenerationNow,
                            )
                        ) {
                            // An import or indexing activation won the race with this read. Loop
                            // against the newly published immutable generation.
                            if (recommendationResourcesBlocked()) {
                                databaseRefreshGate.deferUntilResourcesAreFree()
                                waitingForDeferredRefresh = true
                                break
                            }
                            continue
                        }
                        publishDatabaseInfo(next)
                        val activeGenerationAfterPublication = activeGenerationIdOrNull()
                        if (V2DatabaseRefreshGenerationPolicy.needsAnotherPass(
                                validatedGeneration = next?.info?.generationId,
                                activeGeneration = activeGenerationAfterPublication,
                            )
                        ) {
                            // Close the check-to-publication race as well. Import publishes its own
                            // result, while indexing defers this follow-up until release.
                            if (recommendationResourcesBlocked()) {
                                databaseRefreshGate.deferUntilResourcesAreFree()
                                waitingForDeferredRefresh = true
                                break
                            }
                            continue
                        }
                        Log.i(
                            "MainViewModel",
                            "Active database refresh pass $refreshPass completed in " +
                                "${(System.nanoTime() - passStartedNs) / 1_000_000L}ms",
                        )
                    } catch (e: Exception) {
                        Log.e("MainViewModel", "Active database validation failed", e)
                        val activeAfterFailure = activeGenerationIdOrNull()
                        if (hasTrustedCurrentDatabaseInfo(activeAfterFailure)) {
                            _databaseLoading.value = false
                            Log.w(
                                "MainViewModel",
                                "Background library reconciliation failed; retained ready " +
                                    "generation $activeAfterFailure",
                            )
                        } else {
                            publishDatabaseInfo(null)
                            _unindexedCount.value = -2
                        }
                    }
                    break
                } while (true)
            } finally {
                Log.i(
                    "MainViewModel",
                    "Active database refresh finished after $refreshPass pass(es) in " +
                        "${(System.nanoTime() - refreshStartedNs) / 1_000_000L}ms",
                )
                _databaseVerificationStatus.value = null
                _databaseLoading.value = waitingForDeferredRefresh
                val hasDeferredRefresh = databaseRefreshGate.completeRunningRequest()
                if (hasDeferredRefresh && !recommendationResourcesBlocked()) {
                    refreshDatabaseInfo()
                } else {
                    refreshSessionReplayEligibility()
                    refreshDisplayedQueueEligibility()
                }
            }
        }
    }

    /** Foreground resume should not rescan the entire Poweramp library when nothing was published. */
    fun refreshDatabaseInfoIfGenerationChanged() {
        viewModelScope.launch(Dispatchers.IO) {
            val activeGeneration = activeGenerationIdOrNull()
            if (V2DatabaseReadySnapshotPolicy.shouldRefreshOnResume(
                    publishedGeneration = _databaseInfo.value?.generationId,
                    activeGeneration = activeGeneration,
                )
            ) {
                refreshDatabaseInfo()
            } else {
                Log.d("MainViewModel", "Foreground resume kept the published music index")
            }
        }
    }

    private fun publishDatabaseInfo(next: VerifiedDatabaseInfo?) {
        var hadPublishedLibrary = false
        var libraryContextChanged = false
        val jobs = synchronized(previewStateLock) {
            val previous = _databaseInfo.value
            if (next?.activeCatalog == null &&
                previous?.generationId == next?.info?.generationId &&
                verifiedReplayLibraryBinding?.generation?.generationId ==
                    previous?.generationId
            ) {
                Log.d(
                    "MainViewModel",
                    "Kept the exact active-library binding over a same-generation " +
                        "database-only snapshot",
                )
                return
            }
            val nextInfo = next?.info
            hadPublishedLibrary = previous != null
            val contextChanged = previous?.generationId != nextInfo?.generationId ||
                previous?.providerGenerationId != nextInfo?.providerGenerationId ||
                previous?.activeTrackCount != nextInfo?.activeTrackCount ||
                previous?.eligibleCandidateIdentityCount !=
                    nextInfo?.eligibleCandidateIdentityCount
            libraryContextChanged = contextChanged
            val invalidated = if (contextChanged && _previews.value.isNotEmpty()) {
                markAllPreviewsContextChangedLocked()
            } else {
                emptyList()
            }
            verifiedReplayLibraryBinding = next?.replayLibraryBinding
            _databaseInfo.value = nextInfo
            invalidated
        }
        val cachedLibraryChanged = currentProcessTextLibrarySnapshot()?.let { snapshot ->
            val nextInfo = next?.info
            nextInfo == null ||
                snapshot.generation.generationId != nextInfo.generationId ||
                snapshot.activeDomain.binding.providerGenerationId !=
                    nextInfo.providerGenerationId ||
                snapshot.activeDomain.activeTrackCount != nextInfo.activeTrackCount
        } ?: false
        if ((hadPublishedLibrary && libraryContextChanged) || cachedLibraryChanged) {
            invalidateTextLibrarySnapshot()
        }
        RadioService.publishActiveLibrarySnapshot(
            generation = next?.replayLibraryBinding?.generation,
            catalog = next?.activeCatalog,
        )
        publishVerifiedFindMusicCatalogHandoff(next)
        jobs.forEach { it.cancel() }
    }

    private fun readDatabaseInfo(
        resolution: V2ResolvedLibraryDatabase,
        activeCatalogOverride: V2ActiveLibraryCatalog? = null,
        reconcileProviderIfMissing: Boolean = true,
        onStatus: (String) -> Unit = {},
    ): VerifiedDatabaseInfo {
        val totalStartedNs = System.nanoTime()
        logMemoryPhase("database_info_read_start")
        var providerSnapshotMs: Long? = null
        var catalogLoadMs: Long? = null
        val activeGeneration = resolution.activeGeneration
        val activeCatalogSource = if (activeCatalogOverride != null) {
            "cached_snapshot"
        } else if (!reconcileProviderIfMissing) {
            "database_only"
        } else {
            "provider_reconciliation"
        }
        val activeCatalog = activeCatalogOverride?.also { catalog ->
            require(
                catalog.generationBinding.databaseGenerationId ==
                    activeGeneration.manifest.generationId,
            ) { "Process active-library snapshot belongs to another database generation" }
        } ?: if (reconcileProviderIfMissing) {
            onStatus("Reading the Poweramp library...")
            val acquisition = databaseProviderAcquisitions.incrementAndGet()
            Log.i(
                "MainViewModel",
                "Active database provider acquisition=$acquisition " +
                    "reason=background_reconciliation",
            )
            val providerStartedNs = System.nanoTime()
            val providerSnapshot = V2PowerampProviderSnapshotAcquirer(
                getApplication<Application>(),
            ).acquireBlocking { completedRows, totalRows ->
                onStatus(powerampLibraryReadProgressText(completedRows, totalRows))
            }
            providerSnapshotMs = (System.nanoTime() - providerStartedNs) / 1_000_000L
            logMemoryPhase("database_provider_snapshot_ready")
            val catalogStartedNs = System.nanoTime()
            V2ActiveLibraryCatalogLoader.load(
                activeGeneration = activeGeneration,
                providerSnapshot = providerSnapshot,
                onProgress = { progress ->
                    onStatus(activeLibraryCatalogProgressText(progress))
                },
            ).also {
                catalogLoadMs = (System.nanoTime() - catalogStartedNs) / 1_000_000L
                logMemoryPhase("database_active_catalog_ready")
            }.also { catalog ->
                V2ActiveLibraryCatalogStore(getApplication<Application>().filesDir).write(
                    activeGeneration = activeGeneration,
                    catalog = catalog,
                )
            }
        } else {
            null
        }
        onStatus("Reading music-index track and embedding counts...")
        val databaseStartedNs = System.nanoTime()
        val db = EmbeddingDatabase.open(resolution.databaseFile)
        return try {
            val manifest = activeGeneration.manifest
            val eligibleCandidateIdentityCount = activeCatalog?.let { catalog ->
                onStatus("Counting exact duplicate recordings in the active Poweramp library...")
                val duplicateExcess = db.queryActiveFullContentDuplicateExcess(
                    receiptTable = V2EmbeddingCommitRepository.RECEIPT_TABLE,
                    receiptSchemaVersion = V2EmbeddingCommitRepository.RECEIPT_SCHEMA_VERSION,
                    expectedEmbeddingSpecId =
                        activeGeneration.manifest.receiptEmbeddingSpec.specId,
                    isActiveTrackId = catalog::containsActiveTrack,
                )
                val activeCandidateIdentityCount = catalog.activeTrackIds.size - duplicateExcess
                require(activeCandidateIdentityCount in 1..catalog.activeTrackIds.size) {
                    "Active identity count is inconsistent with the active provider catalog"
                }
                logMemoryPhase("database_identity_count_ready")
                (activeCandidateIdentityCount - 1).coerceAtLeast(0)
            }
            val info = DatabaseInfo(
                trackCount = db.getTrackCount(),
                embeddingCount = db.getEmbeddingCount(),
                embeddingDim = db.getEmbeddingDim(),
                version = db.getMetadata("version"),
                sizeKb = resolution.databaseFile.length() / 1024,
                hasGraph = activeGeneration.graphFile?.isFile == true,
                embeddingTable = db.embeddingTable,
                availableModels = db.getAvailableModels(),
                generationId = manifest.generationId,
                activeTrackCount = activeCatalog?.activeTrackIds?.size,
                unresolvedReceiptCount = activeCatalog?.quarantinedTracks?.count {
                    it.reason == V2ActiveLibraryQuarantineReason.UNRESOLVED_EXACT_RECEIPT
                } ?: 0,
                spanRebuildRequiredCount = activeCatalog?.quarantinedTracks?.count {
                    it.reason == V2ActiveLibraryQuarantineReason.SPAN_SPECIFIC_REBUILD_REQUIRED
                } ?: 0,
                pathTimingConflictCount = activeCatalog?.quarantinedTracks?.count {
                    it.reason == V2ActiveLibraryQuarantineReason.PATH_TIMING_CONFLICT
                } ?: 0,
                noExactProviderBindingCount = activeCatalog?.quarantinedTracks?.count {
                    it.reason == V2ActiveLibraryQuarantineReason.NO_CURRENT_PROVIDER_BINDING
                } ?: 0,
                eligibleCandidateIdentityCount = eligibleCandidateIdentityCount,
                providerGenerationId = activeCatalog?.generationBinding?.providerGenerationId,
                receiptBoundTrackCount = manifest.embeddingCoverage.receiptBoundTrackCount,
                compatibilityTrackCount =
                    manifest.embeddingCoverage.compatibilityBase?.trackCount ?: 0,
            ).also { info ->
                Log.i(
                    "MainViewModel",
                    "Active index verification: provider=${providerSnapshotMs ?: -1}ms, " +
                        "catalog=${catalogLoadMs ?: -1}ms, " +
                        "catalogSource=$activeCatalogSource, " +
                        "database=${(System.nanoTime() - databaseStartedNs) / 1_000_000L}ms, " +
                        "total=${(System.nanoTime() - totalStartedNs) / 1_000_000L}ms, " +
                        "active=${info.activeTrackCount ?: -1}, " +
                        "eligibleIdentities=${info.eligibleCandidateIdentityCount ?: -1}, " +
                        "stored=${info.trackCount}",
                )
            }
            val replayLibraryBinding = activeCatalog?.let { catalog ->
                VerifiedReplayLibraryBinding.from(
                    generation = activeGeneration.toRadioGenerationToken(),
                    catalog = catalog,
                )
            }
            logMemoryPhase("database_info_read_end")
            VerifiedDatabaseInfo(
                info = info,
                replayLibraryBinding = replayLibraryBinding,
                activeCatalog = activeCatalog,
            )
        } finally {
            db.close()
        }
    }
}

/**
 * Database info for display.
 */
data class DatabaseInfo(
    val trackCount: Int,
    val embeddingCount: Int,
    val embeddingDim: Int?,
    val version: String?,
    val sizeKb: Long,
    val hasGraph: Boolean = false,
    val embeddingTable: String = "embeddings_clamp3",
    val availableModels: List<Pair<String, Int>> = emptyList(),
    val generationId: String? = null,
    /** Exact provider-bound selector domain, or null when Poweramp access is unavailable. */
    val activeTrackCount: Int? = null,
    val unresolvedReceiptCount: Int = 0,
    val spanRebuildRequiredCount: Int = 0,
    val pathTimingConflictCount: Int = 0,
    val noExactProviderBindingCount: Int = 0,
    /** Exact distinct active candidate identities left after excluding any one active seed. */
    val eligibleCandidateIdentityCount: Int? = null,
    val providerGenerationId: String? = null,
    val receiptBoundTrackCount: Int = 0,
    val compatibilityTrackCount: Int = 0,
)

data class AppFileStatus(
    val name: String,
    val present: Boolean,
    val sizeMb: String? = null,
    val detail: String? = null,
)

data class TextSearchMatch(
    val track: EmbeddedTrack,
    val similarity: Float,
    val identity: RadioSeedIdentity,
    val rankingScore: Float = similarity,
    val objectiveRank: Int? = null,
    val anchorPercentiles: List<Float> = emptyList(),
)

enum class FindMusicResultKind { TEXT, COMPOSED }

data class TextSearchResult(
    val query: String,
    val matches: List<TextSearchMatch> = emptyList(),
    val error: String? = null,
    val querySpec: FindMusicQuerySpec? = null,
    val libraryBinding: StableIdentityGenerationBinding? = null,
    val providerGenerationId: String? = null,
    val orderedActiveTrackIdsSha256: String? = null,
    val activeTrackCount: Int? = null,
    /** Exact recording-identity domain of objectiveRank. */
    val objectiveRankingDomainCount: Int? = null,
    /** Exact identity domain used for composed per-ingredient percentiles. */
    val ingredientRankingDomainCount: Int? = null,
    val unresolvedAnchors: List<FindMusicSongAnchor> = emptyList(),
    val kind: FindMusicResultKind = FindMusicResultKind.TEXT,
    val stableResultReduction: StableResultReductionEvidence? = null,
    /** Exact complete-domain membership proof for a displayed Varied text result. */
    val textQueuePlan: FindMusicTextQueuePlanEvidence? = null,
    /** Exact complete-domain membership proof for a displayed Varied All-of result. */
    val allOfQueuePlan: FindMusicAllOfQueuePlanEvidence? = null,
    /** In-memory exact embeddings used for this displayed composed ranking. */
    val preparedSeeds: List<SeedSpec> = emptyList(),
)

private fun TextSearchResult.hasExactActiveDomainBinding(): Boolean =
    !providerGenerationId.isNullOrBlank() &&
        orderedActiveTrackIdsSha256?.matches(Regex("^[0-9a-f]{64}$")) == true &&
        (activeTrackCount ?: 0) > 0

private data class RankedFindMusicRow(
    val objectiveRank: Int,
    val row: ComposedRankingRow,
)

private fun StableVisibleReduction<*>.toStableResultReductionEvidence() =
    StableResultReductionEvidence(
        identityPolicyVersion = StableVisibleResultReducer.IDENTITY_POLICY_VERSION,
        requestedVisibleCount = requestedVisibleCount,
        scannedRowCount = scannedRowCount,
        collapsedEquivalentCount = collapsedEquivalentCount,
    )

/**
 * State for a single song seed in multi-seed search.
 */
data class SongSeedState(
    val id: Long = nextSeedId(),
    val query: String = "",
    val confirmedTrack: EmbeddedTrack? = null,
    val stableTrackSpanId: String? = null,
    val libraryBinding: StableIdentityGenerationBinding? = null,
    val weight: Float = 0f,
    val negative: Boolean = false,  // true = "less like this"
    val locked: Boolean = false,
) {
    val isActive: Boolean
        get() = confirmedTrack != null && weight > 0f
}

/** Editor state for one independent text embedding request. */
data class TextIngredientState(
    val id: Long = nextIngredientId(),
    val query: String = "",
    val weight: Float = 0f,
    val negative: Boolean = false,
    val locked: Boolean = false,
) {
    val isActive: Boolean
        get() = query.isNotBlank() && weight > 0f
}

private var seedIdCounter = 0L
fun nextSeedId(): Long = seedIdCounter++

private var ingredientIdCounter = 0L
private fun nextIngredientId(): Long = ingredientIdCounter++
