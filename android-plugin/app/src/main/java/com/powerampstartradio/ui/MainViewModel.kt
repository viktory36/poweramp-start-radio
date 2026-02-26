package com.powerampstartradio.ui

import android.app.Application
import android.content.Context
import android.net.Uri
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.powerampstartradio.data.EmbeddedTrack
import com.powerampstartradio.data.EmbeddingDatabase
import com.powerampstartradio.data.EmbeddingIndex
import com.powerampstartradio.indexing.IndexingService
import com.powerampstartradio.indexing.IndexingViewModel
import com.powerampstartradio.indexing.Clamp3TextInference
import com.powerampstartradio.indexing.NewTrackDetector
import com.powerampstartradio.poweramp.PowerampHelper
import com.powerampstartradio.poweramp.PowerampReceiver
import com.powerampstartradio.poweramp.TrackMatcher
import com.powerampstartradio.services.RadioService
import com.powerampstartradio.similarity.RecommendationEngine
import com.google.ai.edge.litert.Accelerator
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
    private val _unindexedCount = MutableStateFlow(initUnindexedCount())
    val unindexedCount: StateFlow<Int> = _unindexedCount.asStateFlow()

    /** Load persisted count, but reset if DB has changed since last check. */
    private fun initUnindexedCount(): Int {
        val app = getApplication<Application>()
        val dbFile = File(app.filesDir, "embeddings.db")
        val currentFp = if (dbFile.exists()) "${dbFile.length()}_${dbFile.lastModified()}" else ""
        val savedFp = app.getSharedPreferences("indexing", Context.MODE_PRIVATE)
            .getString("dismissed_db_fingerprint", "") ?: ""
        if (currentFp != savedFp) {
            // DB changed — stale count, force re-check
            prefs.edit().remove("unindexed_count").apply()
            return -2
        }
        return prefs.getInt("unindexed_count", -2)
    }

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

    init {
        RadioService.initHistory(application.filesDir)
        refreshDatabaseInfo()
        checkPermission()
        prepareIndices()
        checkModels()
        PowerampReceiver.addTrackChangeListener(trackChangeListener)

        // Re-check unindexed count when indexing completes and user dismisses the result
        viewModelScope.launch {
            var wasComplete = false
            IndexingService.state.collect { state ->
                if (state is IndexingService.IndexingState.Complete) wasComplete = true
                if (wasComplete && state is IndexingService.IndexingState.Idle) {
                    wasComplete = false
                    checkUnindexedTracks()
                }
            }
        }
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
        val embFile = File(getApplication<Application>().filesDir, "clamp3.emb")
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

    // --- Text search state ---

    /** Current text search result. Null when idle, non-null after a search completes. */
    private val _textSearchResult = MutableStateFlow<TextSearchResult?>(null)
    val textSearchResult: StateFlow<TextSearchResult?> = _textSearchResult.asStateFlow()

    private val _textSearchLoading = MutableStateFlow(false)
    val textSearchLoading: StateFlow<Boolean> = _textSearchLoading.asStateFlow()

    /** Recent text search queries (persisted across sessions). */
    private val _recentSearches = MutableStateFlow<List<String>>(
        prefs.getString("recent_searches", null)?.split("\u0000")?.filter { it.isNotBlank() }
            ?: emptyList()
    )
    val recentSearches: StateFlow<List<String>> = _recentSearches.asStateFlow()

    private var textInference: Clamp3TextInference? = null
    private var textIndex: EmbeddingIndex? = null

    /**
     * Search for the best matching track by text query using CLaMP3 text embeddings.
     */
    fun performTextSearch(query: String) {
        if (_textSearchLoading.value) return
        _textSearchLoading.value = true
        _textSearchResult.value = null

        viewModelScope.launch(Dispatchers.IO) {
            try {
                val filesDir = getApplication<Application>().filesDir
                val dbFile = File(filesDir, "embeddings.db")
                if (!dbFile.exists()) {
                    _textSearchResult.value = TextSearchResult(query = query, error = "No embedding database found")
                    return@launch
                }

                // Lazy init text inference with fallback chain
                val inference = textInference ?: run {
                    val vocabFile = File(filesDir, "xlm_roberta_vocab.json")
                    if (!vocabFile.exists()) {
                        _textSearchResult.value = TextSearchResult(query = query, error = "Tokenizer vocab not found")
                        return@launch
                    }

                    // Try: FP16+GPU → FP16+CPU → FP32+GPU → FP32+CPU
                    val candidates = buildList {
                        val fp16 = File(filesDir, "clamp3_text_fp16.tflite")
                        val fp32 = File(filesDir, "clamp3_text.tflite")
                        if (fp16.exists()) {
                            add(fp16 to Accelerator.GPU)
                            add(fp16 to Accelerator.CPU)
                        }
                        if (fp32.exists()) {
                            add(fp32 to Accelerator.GPU)
                            add(fp32 to Accelerator.CPU)
                        }
                    }
                    if (candidates.isEmpty()) {
                        _textSearchResult.value = TextSearchResult(query = query, error = "CLaMP3 text model not found")
                        return@launch
                    }

                    var lastError: Exception? = null
                    var result: Clamp3TextInference? = null
                    for ((modelFile, accel) in candidates) {
                        try {
                            result = Clamp3TextInference(modelFile, vocabFile, accel)
                            Log.i("MainViewModel", "Text model loaded: ${modelFile.name} on $accel")
                            break
                        } catch (e: Exception) {
                            Log.w("MainViewModel", "Text model ${modelFile.name}+$accel failed: ${e.message}")
                            lastError = e
                        }
                    }
                    if (result == null) {
                        _textSearchResult.value = TextSearchResult(query = query, error = "Failed to load text model: ${lastError?.message}")
                        return@launch
                    }
                    result.also { textInference = it }
                }

                // Generate text embedding (save to debug dir for quality comparison)
                val debugDir = File(filesDir, "debug_embeddings")
                val embedding = inference.generateEmbedding(query, debugDir)
                if (embedding == null) {
                    _textSearchResult.value = TextSearchResult(query = query, error = "Text inference failed")
                    return@launch
                }

                // Lazy init CLaMP3 embedding index for text search
                // Text and audio embeddings share the same 768d space in CLaMP3
                val index = textIndex ?: run {
                    val embFile = File(filesDir, "clamp3.emb")
                    if (!embFile.exists()) {
                        val db = EmbeddingDatabase.open(dbFile)
                        val clamp3Count = db.getEmbeddingCountForTable("embeddings_clamp3")
                        if (clamp3Count == 0) {
                            db.close()
                            _textSearchResult.value = TextSearchResult(query = query, error = "No CLaMP3 embeddings in database")
                            return@launch
                        }
                        EmbeddingIndex.extractFromDatabase(db, embFile, table = "embeddings_clamp3")
                        db.close()
                    }
                    if (!embFile.exists()) {
                        _textSearchResult.value = TextSearchResult(query = query, error = "Failed to extract CLaMP3 index")
                        return@launch
                    }
                    EmbeddingIndex.mmap(embFile).also { textIndex = it }
                }

                // Find top matches
                val topMatches = index.findTopK(embedding, topK = 5)
                if (topMatches.isEmpty()) {
                    _textSearchResult.value = TextSearchResult(query = query, error = "No matches found")
                    return@launch
                }

                // Resolve track metadata
                val db = EmbeddingDatabase.open(dbFile)
                val matchedTracks = topMatches.mapNotNull { (trackId, score) ->
                    db.getTrackById(trackId)?.let { track -> TextSearchMatch(track, score) }
                }
                db.close()

                _textSearchResult.value = TextSearchResult(query = query, matches = matchedTracks)

                // Save to recent searches
                saveRecentSearch(query)

            } catch (e: Exception) {
                Log.e("MainViewModel", "Text search failed", e)
                _textSearchResult.value = TextSearchResult(query = query, error = "Search failed: ${e.message}")
            } finally {
                _textSearchLoading.value = false
            }
        }
    }

    /**
     * Start radio using a text search match as seed.
     */
    fun startRadioFromTextSearch(trackId: Long) {
        RadioService.startRadioFromSeed(getApplication(), trackId, buildConfig())
    }

    fun clearTextSearchResult() {
        _textSearchResult.value = null
    }

    private fun saveRecentSearch(query: String) {
        val updated = (listOf(query) + _recentSearches.value.filter { it != query }).take(10)
        _recentSearches.value = updated
        prefs.edit().putString("recent_searches", updated.joinToString("\u0000")).apply()
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
                // Check if DB changed — clear stale dismissed IDs if so
                val indexingPrefs = app.getSharedPreferences("indexing", Context.MODE_PRIVATE)
                val currentFingerprint = if (dbFile.exists())
                    "${dbFile.length()}_${dbFile.lastModified()}" else ""
                val savedFingerprint = indexingPrefs.getString("dismissed_db_fingerprint", "") ?: ""
                if (currentFingerprint != savedFingerprint) {
                    indexingPrefs.edit()
                        .remove("dismissed_track_ids")
                        .putString("dismissed_db_fingerprint", currentFingerprint)
                        .apply()
                }
                // Exclude dismissed tracks from the count
                val dismissedJson = indexingPrefs.getString("dismissed_track_ids", null)
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

        val mertFile = findModel("mert")
        val clamp3AudioFile = findModel("clamp3_audio")
        val clamp3TextFile = findModel("clamp3_text")
        _hasModels.value = mertFile != null && clamp3AudioFile != null

        fun fileSizeMb(f: File?): String? {
            if (f == null) return null
            val mb = f.length() / 1024 / 1024
            return "${mb} MB"
        }

        val dbFile = File(filesDir, "embeddings.db")
        val embFile = File(filesDir, "clamp3.emb")
        val graphFile = File(filesDir, "graph.bin")
        val vocabFile = File(filesDir, "xlm_roberta_vocab.json")

        _fileStatuses.value = listOf(
            AppFileStatus("embeddings.db", dbFile.exists(), fileSizeMb(dbFile),
                "Embedding database (required)"),
            AppFileStatus("clamp3.emb", embFile.exists(), fileSizeMb(embFile),
                "Auto-generated from database"),
            AppFileStatus("graph.bin", graphFile.exists(), fileSizeMb(graphFile),
                "kNN graph for Random Walk"),
            AppFileStatus("mert", mertFile != null, fileSizeMb(mertFile),
                if (mertFile != null) mertFile.name else "MERT audio feature model"),
            AppFileStatus("clamp3_audio", clamp3AudioFile != null, fileSizeMb(clamp3AudioFile),
                if (clamp3AudioFile != null) clamp3AudioFile.name else "CLaMP3 audio encoder"),
            AppFileStatus("clamp3_text", clamp3TextFile != null, fileSizeMb(clamp3TextFile),
                if (clamp3TextFile != null) clamp3TextFile.name else "CLaMP3 text encoder (for text search)"),
            AppFileStatus("vocab", vocabFile.exists(), fileSizeMb(vocabFile),
                "xlm_roberta_vocab.json (for text search)"),
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
                val t0 = System.nanoTime()
                Log.i("MainViewModel", "importDatabase: starting from URI $uri")
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
                File(app.filesDir, "clamp3.emb").delete()
                File(app.filesDir, "graph.bin").delete()

                // Atomic replace: delete old, rename temp to final
                val copyMs = (System.nanoTime() - t0) / 1_000_000
                val fileSizeMB = tempFile.length() / (1024 * 1024)
                Log.i("MainViewModel", "TIMING: DB copy ${fileSizeMB}MB in ${copyMs}ms")
                destFile.delete()
                if (!tempFile.renameTo(destFile)) {
                    throw java.io.IOException("Failed to move imported database")
                }

                IndexingViewModel.invalidateCache()

                val db = EmbeddingDatabase.open(destFile)

                _importStatus.value = "Reading database info..."
                val tInfo = System.nanoTime()
                val info = DatabaseInfo(
                    trackCount = db.getTrackCount(),
                    embeddingCount = db.getEmbeddingCount(),
                    embeddingDim = db.getEmbeddingDim(),
                    version = db.getMetadata("version"),
                    sizeKb = destFile.length() / 1024,
                    hasGraph = db.hasBinaryData("knn_graph"),
                    embeddingTable = db.embeddingTable,
                    availableModels = db.getAvailableModels(),
                )
                db.close()
                val infoMs = (System.nanoTime() - tInfo) / 1_000_000
                Log.i("MainViewModel", "DB info: ${info.trackCount} tracks, " +
                    "${info.embeddingCount} embeddings, dim=${info.embeddingDim}, " +
                    "hasGraph=${info.hasGraph}, models=${info.availableModels} (${infoMs}ms)")
                _databaseInfo.value = info

                // Extract indices with progress updates
                _importStatus.value = "Extracting search index..."
                prepareIndicesWithProgress { status ->
                    _importStatus.value = status
                }

                checkModels()
                val totalMs = (System.nanoTime() - t0) / 1_000_000
                Log.i("MainViewModel", "TIMING: importDatabase total = ${totalMs}ms")
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
    val hasGraph: Boolean = false,
    val embeddingTable: String = "embeddings_clamp3",
    val availableModels: List<Pair<String, Int>> = emptyList(),
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
)

data class TextSearchResult(
    val query: String,
    val matches: List<TextSearchMatch> = emptyList(),
    val error: String? = null,
)
