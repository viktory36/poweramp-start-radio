package com.powerampstartradio.ui

import android.app.Application
import android.content.Context
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.powerampstartradio.data.EmbeddingDatabase
import com.powerampstartradio.poweramp.PowerampHelper
import com.powerampstartradio.poweramp.PowerampReceiver
import com.powerampstartradio.poweramp.TrackMatcher
import com.powerampstartradio.services.RadioService
import com.powerampstartradio.similarity.RecommendationEngine
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import java.io.File

/**
 * ViewModel for the main screen.
 */
class MainViewModel(application: Application) : AndroidViewModel(application) {

    private val prefs = application.getSharedPreferences("settings", Context.MODE_PRIVATE)

    // --- RadioConfig settings ---

    private val _numTracks = MutableStateFlow(prefs.getInt("num_tracks", RadioService.DEFAULT_NUM_TRACKS))
    val numTracks: StateFlow<Int> = _numTracks.asStateFlow()

    private val _selectionMode = MutableStateFlow(
        try {
            val stored = prefs.getString("selection_mode", null)
            if (stored != null) SelectionMode.valueOf(stored) else SelectionMode.MMR
        } catch (e: IllegalArgumentException) { SelectionMode.MMR }
    )
    val selectionMode: StateFlow<SelectionMode> = _selectionMode.asStateFlow()

    private val _driftEnabled = MutableStateFlow(prefs.getBoolean("drift_enabled", true))
    val driftEnabled: StateFlow<Boolean> = _driftEnabled.asStateFlow()

    private val _driftMode = MutableStateFlow(
        try { DriftMode.valueOf(prefs.getString("drift_mode", DriftMode.SEED_INTERPOLATION.name)!!) }
        catch (e: IllegalArgumentException) { DriftMode.SEED_INTERPOLATION }
    )
    val driftMode: StateFlow<DriftMode> = _driftMode.asStateFlow()

    private val _anchorStrength = MutableStateFlow(prefs.getFloat("anchor_strength", 0.5f))
    val anchorStrength: StateFlow<Float> = _anchorStrength.asStateFlow()

    private val _anchorDecay = MutableStateFlow(
        try { DecaySchedule.valueOf(prefs.getString("anchor_decay", DecaySchedule.EXPONENTIAL.name)!!) }
        catch (e: IllegalArgumentException) { DecaySchedule.EXPONENTIAL }
    )
    val anchorDecay: StateFlow<DecaySchedule> = _anchorDecay.asStateFlow()

    private val _momentumBeta = MutableStateFlow(prefs.getFloat("momentum_beta", 0.7f))
    val momentumBeta: StateFlow<Float> = _momentumBeta.asStateFlow()

    private val _diversityLambda = MutableStateFlow(prefs.getFloat("diversity_lambda", 0.4f))
    val diversityLambda: StateFlow<Float> = _diversityLambda.asStateFlow()

    private val _temperature = MutableStateFlow(prefs.getFloat("temperature", 0.05f))
    val temperature: StateFlow<Float> = _temperature.asStateFlow()

    private val _maxPerArtist = MutableStateFlow(prefs.getInt("max_per_artist", 8))
    val maxPerArtist: StateFlow<Int> = _maxPerArtist.asStateFlow()

    private val _minArtistSpacing = MutableStateFlow(prefs.getInt("min_artist_spacing", 2))
    val minArtistSpacing: StateFlow<Int> = _minArtistSpacing.asStateFlow()

    private val _candidatePoolSize = MutableStateFlow(prefs.getInt("candidate_pool_size", 200))
    val candidatePoolSize: StateFlow<Int> = _candidatePoolSize.asStateFlow()

    // --- Database & permission state ---

    private val _databaseInfo = MutableStateFlow<DatabaseInfo?>(null)
    val databaseInfo: StateFlow<DatabaseInfo?> = _databaseInfo.asStateFlow()

    private val _hasPermission = MutableStateFlow(false)
    val hasPermission: StateFlow<Boolean> = _hasPermission.asStateFlow()

    private val _indexStatus = MutableStateFlow<String?>(null)
    val indexStatus: StateFlow<String?> = _indexStatus.asStateFlow()

    private val _previews = MutableStateFlow<Map<SelectionMode, List<String>>>(emptyMap())
    val previews: StateFlow<Map<SelectionMode, List<String>>> = _previews.asStateFlow()

    private val _previewsLoading = MutableStateFlow<Set<SelectionMode>>(emptySet())
    val previewsLoading: StateFlow<Set<SelectionMode>> = _previewsLoading.asStateFlow()

    private val previewJobs = mutableMapOf<SelectionMode, Job>()

    val radioState: StateFlow<RadioUiState> = RadioService.uiState
    val sessionHistory: StateFlow<List<RadioResult>> = RadioService.sessionHistory

    private val trackChangeListener: (com.powerampstartradio.poweramp.PowerampTrack?) -> Unit = { track ->
        val state = RadioService.uiState.value
        if (state is RadioUiState.Success) {
            val result = state.result
            val knownIds = result.queuedFileIds + result.seedTrack.realId
            if (track == null || track.realId !in knownIds) {
                RadioService.resetState()
            }
        }
    }

    init {
        refreshDatabaseInfo()
        checkPermission()
        prepareIndices()
        PowerampReceiver.addTrackChangeListener(trackChangeListener)
    }

    override fun onCleared() {
        super.onCleared()
        PowerampReceiver.removeTrackChangeListener(trackChangeListener)
    }

    /**
     * Build current RadioConfig from all settings.
     */
    fun buildConfig(): RadioConfig = RadioConfig(
        numTracks = _numTracks.value,
        candidatePoolSize = _candidatePoolSize.value,
        selectionMode = _selectionMode.value,
        driftEnabled = _driftEnabled.value,
        driftMode = _driftMode.value,
        anchorStrength = _anchorStrength.value,
        anchorDecay = _anchorDecay.value,
        momentumBeta = _momentumBeta.value,
        diversityLambda = _diversityLambda.value,
        temperature = _temperature.value,
        maxPerArtist = _maxPerArtist.value,
        minArtistSpacing = _minArtistSpacing.value,
    )

    // --- Setters ---

    fun setNumTracks(count: Int) {
        _numTracks.value = count
        prefs.edit().putInt("num_tracks", count).apply()
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
        _anchorStrength.value = value
        prefs.edit().putFloat("anchor_strength", value).apply()
    }

    fun setAnchorDecay(schedule: DecaySchedule) {
        _anchorDecay.value = schedule
        prefs.edit().putString("anchor_decay", schedule.name).apply()
    }

    fun setMomentumBeta(value: Float) {
        _momentumBeta.value = value
        prefs.edit().putFloat("momentum_beta", value).apply()
    }

    fun setDiversityLambda(value: Float) {
        _diversityLambda.value = value
        prefs.edit().putFloat("diversity_lambda", value).apply()
    }

    fun setTemperature(value: Float) {
        _temperature.value = value
        prefs.edit().putFloat("temperature", value).apply()
    }

    fun setMaxPerArtist(value: Int) {
        _maxPerArtist.value = value
        prefs.edit().putInt("max_per_artist", value).apply()
    }

    fun setMinArtistSpacing(value: Int) {
        _minArtistSpacing.value = value
        prefs.edit().putInt("min_artist_spacing", value).apply()
    }

    fun setCandidatePoolSize(value: Int) {
        _candidatePoolSize.value = value
        prefs.edit().putInt("candidate_pool_size", value).apply()
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

    fun clearPreview(mode: SelectionMode) {
        previewJobs[mode]?.cancel()
        _previews.value = _previews.value - mode
        _previewsLoading.value = _previewsLoading.value + mode
    }

    fun invalidatePreview(mode: SelectionMode) {
        previewJobs[mode]?.cancel()
        _previews.value = _previews.value - mode
        _previewsLoading.value = _previewsLoading.value - mode
    }

    fun computePreview(mode: SelectionMode) {
        previewJobs[mode]?.cancel()
        _previewsLoading.value = _previewsLoading.value + mode
        previewJobs[mode] = viewModelScope.launch(Dispatchers.IO) {
            val result = runPreviewForMode(mode)
            if (result != null) {
                _previews.value = _previews.value + (mode to result)
            }
            _previewsLoading.value = _previewsLoading.value - mode
        }
    }

    private suspend fun runPreviewForMode(mode: SelectionMode): List<String>? {
        val currentTrack = PowerampReceiver.currentTrack ?: return null
        val dbFile = File(getApplication<Application>().filesDir, "embeddings.db")
        if (!dbFile.exists()) return null

        var db: EmbeddingDatabase? = null
        return try {
            db = EmbeddingDatabase.open(dbFile)
            val matcher = TrackMatcher(db)
            val match = matcher.findMatch(currentTrack)
            if (match == null || match.matchType == TrackMatcher.MatchType.NOT_FOUND) {
                return null
            }
            val seedId = match.embeddedTrack.id

            val engine = RecommendationEngine(db, getApplication<Application>().filesDir)
            engine.ensureIndices()

            val config = buildConfig().copy(numTracks = 10, selectionMode = mode)
            val tracks = engine.generatePlaylist(seedId, config)
            tracks.map { t -> "${t.track.title ?: "?"} \u2013 ${t.track.artist ?: "?"}" }
        } catch (_: Exception) { null }
        finally { db?.close() }
    }

    fun resetToDefaults() {
        val defaults = RadioConfig()
        setNumTracks(defaults.numTracks)
        setCandidatePoolSize(defaults.candidatePoolSize)
        setSelectionMode(defaults.selectionMode)
        setDriftEnabled(defaults.driftEnabled)
        setDriftMode(defaults.driftMode)
        setAnchorStrength(defaults.anchorStrength)
        setAnchorDecay(defaults.anchorDecay)
        setMomentumBeta(defaults.momentumBeta)
        setDiversityLambda(defaults.diversityLambda)
        setTemperature(defaults.temperature)
        setMaxPerArtist(defaults.maxPerArtist)
        setMinArtistSpacing(defaults.minArtistSpacing)
    }

    fun prepareIndices() {
        viewModelScope.launch(Dispatchers.IO) {
            val dbFile = File(getApplication<Application>().filesDir, "embeddings.db")
            if (!dbFile.exists()) return@launch
            try {
                val db = EmbeddingDatabase.open(dbFile)
                val engine = RecommendationEngine(db, getApplication<Application>().filesDir)
                engine.ensureIndices { message ->
                    _indexStatus.value = message
                }
                _indexStatus.value = "Index ready"
                db.close()
            } catch (e: Exception) {
                _indexStatus.value = "Index error: ${e.message}"
            }
        }
    }

    fun checkPermission() {
        viewModelScope.launch {
            _hasPermission.value = PowerampHelper.canAccessData(getApplication())
        }
    }

    fun requestPermission() {
        PowerampHelper.requestDataPermission(getApplication())
    }

    fun refreshDatabaseInfo() {
        TrackMatcher.invalidateCache()
        viewModelScope.launch {
            val dbFile = File(getApplication<Application>().filesDir, "embeddings.db")
            if (dbFile.exists()) {
                try {
                    val db = EmbeddingDatabase.open(dbFile)
                    val info = DatabaseInfo(
                        trackCount = db.getTrackCount(),
                        embeddingCount = db.getEmbeddingCount(),
                        embeddingDim = db.getEmbeddingDim(),
                        version = db.getMetadata("version"),
                        sizeKb = dbFile.length() / 1024,
                        hasFused = db.hasFusedEmbeddings,
                        hasGraph = db.hasBinaryData("knn_graph"),
                        embeddingTable = db.embeddingTable,
                    )
                    db.close()
                    _databaseInfo.value = info
                    prepareIndices()
                } catch (e: Exception) {
                    _databaseInfo.value = null
                }
            } else {
                _databaseInfo.value = null
            }
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
    val hasFused: Boolean = false,
    val hasGraph: Boolean = false,
    val embeddingTable: String = "embeddings_fused",
)
