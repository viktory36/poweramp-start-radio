package com.powerampstartradio.ui

import android.app.Application
import android.content.Context
import android.net.Uri
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.powerampstartradio.data.EmbeddingDatabase
import com.powerampstartradio.data.EmbeddingIndex
import com.powerampstartradio.indexing.IndexingService
import com.powerampstartradio.indexing.IndexingViewModel
import com.powerampstartradio.indexing.NewTrackDetector
import com.powerampstartradio.poweramp.PowerampHelper
import com.powerampstartradio.poweramp.PowerampReceiver
import com.powerampstartradio.poweramp.TrackMatcher
import com.powerampstartradio.services.RadioService
import com.powerampstartradio.similarity.RecommendationEngine
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.async
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
            if (stored != null) SelectionMode.valueOf(stored)
            else {
                // Migrate from old search_strategy pref
                val old = prefs.getString("search_strategy", null)
                if (old != null) SelectionMode.MMR else SelectionMode.MMR
            }
        } catch (e: IllegalArgumentException) { SelectionMode.MMR }
    )
    val selectionMode: StateFlow<SelectionMode> = _selectionMode.asStateFlow()

    private val _driftEnabled = MutableStateFlow(prefs.getBoolean("drift_enabled", false))
    val driftEnabled: StateFlow<Boolean> = _driftEnabled.asStateFlow()

    private val _driftMode = MutableStateFlow(
        try { DriftMode.valueOf(prefs.getString("drift_mode", DriftMode.SEED_INTERPOLATION.name)!!) }
        catch (e: IllegalArgumentException) { DriftMode.SEED_INTERPOLATION }
    )
    val driftMode: StateFlow<DriftMode> = _driftMode.asStateFlow()

    private val _anchorStrength = MutableStateFlow(prefs.getFloat("anchor_strength", 0.5f))
    val anchorStrength: StateFlow<Float> = _anchorStrength.asStateFlow()

    private val _pageRankAlpha = MutableStateFlow(prefs.getFloat("pagerank_alpha", 0.5f))
    val pageRankAlpha: StateFlow<Float> = _pageRankAlpha.asStateFlow()

    private val _anchorDecay = MutableStateFlow(
        try { DecaySchedule.valueOf(prefs.getString("anchor_decay", DecaySchedule.EXPONENTIAL.name)!!) }
        catch (e: IllegalArgumentException) { DecaySchedule.EXPONENTIAL }
    )
    val anchorDecay: StateFlow<DecaySchedule> = _anchorDecay.asStateFlow()

    private val _momentumBeta = MutableStateFlow(prefs.getFloat("momentum_beta", 0.7f))
    val momentumBeta: StateFlow<Float> = _momentumBeta.asStateFlow()

    private val _diversityLambda = MutableStateFlow(prefs.getFloat("diversity_lambda", 0.4f))
    val diversityLambda: StateFlow<Float> = _diversityLambda.asStateFlow()

    private val _maxPerArtist = MutableStateFlow(prefs.getInt("max_per_artist", 8))
    val maxPerArtist: StateFlow<Int> = _maxPerArtist.asStateFlow()

    private val _minArtistSpacing = MutableStateFlow(prefs.getInt("min_artist_spacing", 3))
    val minArtistSpacing: StateFlow<Int> = _minArtistSpacing.asStateFlow()

    // --- Database & permission state ---

    private val _databaseInfo = MutableStateFlow<DatabaseInfo?>(null)
    val databaseInfo: StateFlow<DatabaseInfo?> = _databaseInfo.asStateFlow()

    private val _hasPermission = MutableStateFlow(false)
    val hasPermission: StateFlow<Boolean> = _hasPermission.asStateFlow()

    private val _indexStatus = MutableStateFlow<String?>(null)
    val indexStatus: StateFlow<String?> = _indexStatus.asStateFlow()

    // --- Indexing state ---
    // -2 = never checked, -1 = checking now, 0+ = actual count
    private val _unindexedCount = MutableStateFlow(prefs.getInt("unindexed_count", -2))
    val unindexedCount: StateFlow<Int> = _unindexedCount.asStateFlow()

    private val _unindexedCheckStatus = MutableStateFlow<String?>(null)
    val unindexedCheckStatus: StateFlow<String?> = _unindexedCheckStatus.asStateFlow()

    private val _hasModels = MutableStateFlow(false)
    val hasModels: StateFlow<Boolean> = _hasModels.asStateFlow()

    private val _fileStatuses = MutableStateFlow<List<AppFileStatus>>(emptyList())
    val fileStatuses: StateFlow<List<AppFileStatus>> = _fileStatuses.asStateFlow()

    private val _importStatus = MutableStateFlow<String?>(null)
    val importStatus: StateFlow<String?> = _importStatus.asStateFlow()

    val indexingState: StateFlow<IndexingService.IndexingState> = IndexingService.state

    private val _previews = MutableStateFlow<Map<SelectionMode, List<String>>>(emptyMap())
    val previews: StateFlow<Map<SelectionMode, List<String>>> = _previews.asStateFlow()

    private val _previewsLoading = MutableStateFlow<Set<SelectionMode>>(emptySet())
    val previewsLoading: StateFlow<Set<SelectionMode>> = _previewsLoading.asStateFlow()

    private val previewJobs = mutableMapOf<SelectionMode, Job>()

    // Lazy drift rank computation (on-expand)
    private var rankIndex: EmbeddingIndex? = null
    private val _driftRanks = MutableStateFlow<Map<Long, Int>>(emptyMap())
    val driftRanks: StateFlow<Map<Long, Int>> = _driftRanks.asStateFlow()

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
        RadioService.initHistory(application.filesDir)
        refreshDatabaseInfo()
        checkPermission()
        prepareIndices()
        checkModels()
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
        // candidatePoolSize auto-computed by RecommendationEngine
        selectionMode = _selectionMode.value,
        driftEnabled = _driftEnabled.value,
        driftMode = _driftMode.value,
        anchorStrength = _anchorStrength.value,
        anchorDecay = _anchorDecay.value,
        pageRankAlpha = _pageRankAlpha.value,
        momentumBeta = _momentumBeta.value,
        diversityLambda = _diversityLambda.value,
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

    fun setPageRankAlpha(value: Float) {
        _pageRankAlpha.value = value
        prefs.edit().putFloat("pagerank_alpha", value).apply()
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

    fun setMaxPerArtist(value: Int) {
        _maxPerArtist.value = value
        prefs.edit().putInt("max_per_artist", value).apply()
    }

    fun setMinArtistSpacing(value: Int) {
        _minArtistSpacing.value = value
        prefs.edit().putInt("min_artist_spacing", value).apply()
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
        _driftRanks.value = emptyMap()
    }

    fun requestDriftRank(trackId: Long) {
        if (_driftRanks.value.containsKey(trackId)) return
        val refEmb = RadioService.driftReferences.value[trackId] ?: return
        viewModelScope.launch(Dispatchers.IO) {
            val index = getOrOpenRankIndex() ?: return@launch
            val sims = index.computeAllSimilarities(refEmb)
            val rank = index.rankFromSimilarities(sims, trackId)
            _driftRanks.value = _driftRanks.value + (trackId to rank)
        }
    }

    private fun getOrOpenRankIndex(): EmbeddingIndex? {
        rankIndex?.let { return it }
        val embFile = File(getApplication<Application>().filesDir, "fused.emb")
        if (!embFile.exists()) return null
        return try {
            EmbeddingIndex.mmap(embFile).also { rankIndex = it }
        } catch (_: Exception) { null }
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

        return try {
            val db = EmbeddingDatabase.open(dbFile)
            val matcher = TrackMatcher(db)
            val match = matcher.findMatch(currentTrack)
            if (match == null || match.matchType == TrackMatcher.MatchType.NOT_FOUND) {
                db.close()
                return null
            }
            val seedId = match.embeddedTrack.id

            val engine = RecommendationEngine(db, getApplication<Application>().filesDir)
            engine.ensureIndices()

            val config = buildConfig().copy(numTracks = 10, selectionMode = mode)
            val tracks = engine.generatePlaylist(seedId, config)
            db.close()
            tracks.map { t -> "${t.track.title ?: "?"} \u2013 ${t.track.artist ?: "?"}" }
        } catch (_: Exception) { null }
    }

    // --- Indexing actions ---

    fun checkUnindexedTracks() {
        _unindexedCount.value = -1 // signal "checking" to UI
        val app = getApplication<Application>()
        val dbFile = File(app.filesDir, "embeddings.db")

        val deferred = viewModelScope.async(Dispatchers.IO) {
            if (!dbFile.exists()) return@async emptyList()
            val db = EmbeddingDatabase.open(dbFile)
            val detector = NewTrackDetector(db)
            val tracks = detector.findUnindexedTracks(app) { status ->
                _unindexedCheckStatus.value = status
                IndexingViewModel.detectionStatus.value = status
            }
            val sorted = tracks.sortedByDescending { it.durationMs }
            db.close()
            // Cache in IndexingViewModel so Manage Tracks can reuse
            val powerampCount = getPowerampTrackCount(app)
            IndexingViewModel.cacheResults(sorted, dbFile.lastModified(), powerampCount)
            sorted
        }
        // Expose so IndexingViewModel can await if user opens Manage Tracks mid-check
        IndexingViewModel.pendingDetection = deferred

        viewModelScope.launch {
            try {
                val result = deferred.await()
                // Exclude dismissed tracks from the count
                val dismissedJson = app.getSharedPreferences("indexing", Context.MODE_PRIVATE)
                    .getString("dismissed_track_ids", null)
                val dismissed = if (dismissedJson != null) {
                    try {
                        val arr = org.json.JSONArray(dismissedJson)
                        (0 until arr.length()).map { arr.getLong(it) }.toSet()
                    } catch (_: Exception) { emptySet() }
                } else emptySet<Long>()
                val visible = result.count { it.powerampFileId !in dismissed }
                setUnindexedCount(visible)
            } catch (_: Exception) {
                setUnindexedCount(0)
            } finally {
                IndexingViewModel.pendingDetection = null
                IndexingViewModel.detectionStatus.value = null
                _unindexedCheckStatus.value = null
            }
        }
    }

    private fun setUnindexedCount(count: Int) {
        _unindexedCount.value = count
        prefs.edit().putInt("unindexed_count", count).apply()
    }

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

    fun checkModels() {
        val filesDir = getApplication<Application>().filesDir
        val variants = listOf("_fp16", "")

        fun findModel(base: String): File? {
            for (suffix in variants) {
                val f = File(filesDir, "${base}${suffix}.tflite")
                if (f.exists()) return f
            }
            return null
        }

        val mulanFile = findModel("mulan_audio")
        val flamingoFile = findModel("flamingo_encoder")
        val projectorFile = findModel("flamingo_projector")
        _hasModels.value = mulanFile != null || flamingoFile != null

        fun fileSizeMb(f: File?): String? {
            if (f == null) return null
            val mb = f.length() / 1024 / 1024
            return "${mb} MB"
        }

        val dbFile = File(filesDir, "embeddings.db")
        val embFile = File(filesDir, "fused.emb")
        val graphFile = File(filesDir, "graph.bin")

        _fileStatuses.value = listOf(
            AppFileStatus("embeddings.db", dbFile.exists(), fileSizeMb(dbFile),
                "Embedding database (required)"),
            AppFileStatus("fused.emb", embFile.exists(), fileSizeMb(embFile),
                "Auto-generated from database"),
            AppFileStatus("graph.bin", graphFile.exists(), fileSizeMb(graphFile),
                "kNN graph for Random Walk"),
            AppFileStatus("mulan_audio", mulanFile != null, fileSizeMb(mulanFile),
                if (mulanFile != null) mulanFile.name else "MuQ-MuLan model"),
            AppFileStatus("flamingo_encoder", flamingoFile != null, fileSizeMb(flamingoFile),
                if (flamingoFile != null) flamingoFile.name else "Flamingo encoder model"),
            AppFileStatus("flamingo_projector", projectorFile != null, fileSizeMb(projectorFile),
                if (projectorFile != null) projectorFile.name else "Flamingo projector"),
        )
    }

    fun resetToDefaults() {
        val defaults = RadioConfig()
        setNumTracks(defaults.numTracks)
        setSelectionMode(defaults.selectionMode)
        setDriftEnabled(defaults.driftEnabled)
        setDriftMode(defaults.driftMode)
        setAnchorStrength(defaults.anchorStrength)
        setPageRankAlpha(defaults.pageRankAlpha)
        setAnchorDecay(defaults.anchorDecay)
        setMomentumBeta(defaults.momentumBeta)
        setDiversityLambda(defaults.diversityLambda)
        setMaxPerArtist(defaults.maxPerArtist)
        setMinArtistSpacing(defaults.minArtistSpacing)
    }

    fun prepareIndices() {
        viewModelScope.launch(Dispatchers.IO) {
            prepareIndicesWithProgress { message ->
                _indexStatus.value = message
            }
        }
    }

    private suspend fun prepareIndicesWithProgress(onProgress: (String) -> Unit) {
        val dbFile = File(getApplication<Application>().filesDir, "embeddings.db")
        if (!dbFile.exists()) return
        try {
            val db = EmbeddingDatabase.open(dbFile)
            val engine = RecommendationEngine(db, getApplication<Application>().filesDir)
            engine.ensureIndices { message ->
                onProgress(message)
                _indexStatus.value = message
            }
            _indexStatus.value = "Index ready"
            onProgress("Index ready")
            db.close()
        } catch (e: Exception) {
            _indexStatus.value = "Index error: ${e.message}"
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

    /**
     * Import a database from the given URI asynchronously.
     * Shows progress in importStatus, then refreshes everything.
     */
    fun importDatabase(uri: Uri) {
        viewModelScope.launch(Dispatchers.IO) {
            try {
                _importStatus.value = "Copying database..."
                val app = getApplication<Application>()
                val destFile = File(app.filesDir, "embeddings.db")
                val tempFile = File(app.filesDir, "embeddings.db.import")

                // Invalidate mmap'd index pointing at stale files
                rankIndex = null

                // Copy to temp file first — avoids corrupting the live DB mid-write
                // (refreshDatabaseInfo from onResume can race with us)
                app.contentResolver.openInputStream(uri)?.use { input ->
                    java.io.FileOutputStream(tempFile).use { output ->
                        input.copyTo(output)
                    }
                } ?: throw IllegalArgumentException("Cannot open URI: $uri")

                // Delete stale derived files so they're re-extracted from the new DB
                File(app.filesDir, "fused.emb").delete()
                File(app.filesDir, "graph.bin").delete()

                // Atomic replace: delete old, rename temp to final
                destFile.delete()
                if (!tempFile.renameTo(destFile)) {
                    throw java.io.IOException("Failed to move imported database")
                }

                IndexingViewModel.invalidateCache()

                val db = EmbeddingDatabase.open(destFile)

                _importStatus.value = "Reading database info..."
                val info = DatabaseInfo(
                    trackCount = db.getTrackCount(),
                    embeddingCount = db.getEmbeddingCount(),
                    embeddingDim = db.getEmbeddingDim(),
                    version = db.getMetadata("version"),
                    sizeKb = destFile.length() / 1024,
                    hasFused = db.hasFusedEmbeddings,
                    hasGraph = db.hasBinaryData("knn_graph"),
                    embeddingTable = db.embeddingTable,
                    availableModels = db.getAvailableModels(),
                )
                db.close()
                _databaseInfo.value = info

                // Extract indices with progress updates
                _importStatus.value = "Extracting search index..."
                prepareIndicesWithProgress { status ->
                    _importStatus.value = status
                }

                checkModels()
                _importStatus.value = null

                // Fire-and-forget: unindexed count updates independently
                checkUnindexedTracks()
            } catch (e: Exception) {
                Log.e("MainViewModel", "Import failed", e)
                _importStatus.value = "Import failed: ${e.message}"
                // Clear error after a few seconds so the import button reappears
                kotlinx.coroutines.delay(5000)
                _importStatus.value = null
            }
        }
    }

    fun refreshDatabaseInfo() {
        if (_importStatus.value != null) return // import in progress, skip
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
                        availableModels = db.getAvailableModels(),
                    )
                    db.close()
                    _databaseInfo.value = info
                    prepareIndices()
                    checkModels()
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
    val availableModels: List<Pair<String, Int>> = emptyList(),
)

data class AppFileStatus(
    val name: String,
    val present: Boolean,
    val sizeMb: String? = null,
    val detail: String? = null,
)
