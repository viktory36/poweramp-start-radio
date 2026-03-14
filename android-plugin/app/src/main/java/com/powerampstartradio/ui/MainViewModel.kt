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
import com.powerampstartradio.similarity.SimilarTrack
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
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.roundToInt

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

    private val _walkRestartAlpha = MutableStateFlow(prefs.getFloat("walk_restart_alpha", 0.5f))
    val walkRestartAlpha: StateFlow<Float> = _walkRestartAlpha.asStateFlow()

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

    private val _textSearchTopK = MutableStateFlow(prefs.getInt("text_search_top_k", 20))
    val textSearchTopK: StateFlow<Int> = _textSearchTopK.asStateFlow()

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
        val currentFp = getDbFingerprint(dbFile)
        val savedFp = prefs.getString("unindexed_count_db_fingerprint", "") ?: ""
        if (currentFp != savedFp) {
            // DB changed — stale count, force re-check
            prefs.edit()
                .remove("unindexed_count")
                .remove("unindexed_last_checked_ms")
                .remove("unindexed_count_db_fingerprint")
                .apply()
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

    private val indicesPreparing = AtomicBoolean(false)

    init {
        RadioService.initHistory(application.filesDir)
        refreshDatabaseInfo()
        checkPermission()
        checkModels()
        PowerampReceiver.addTrackChangeListener(trackChangeListener)

        // Re-check unindexed count when indexing completes and user dismisses the result
        viewModelScope.launch {
            var wasComplete = false
            IndexingService.state.collect { state ->
                if (state is IndexingService.IndexingState.Complete) wasComplete = true
                if (wasComplete && state is IndexingService.IndexingState.Idle) {
                    wasComplete = false
                    syncIndexingDbFingerprint()
                    checkUnindexedTracks()
                }
            }
        }
    }

    override fun onCleared() {
        super.onCleared()
        PowerampReceiver.removeTrackChangeListener(trackChangeListener)
        try { textInference?.close() } catch (_: Exception) {}
        textInference = null
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
        walkRestartAlpha = _walkRestartAlpha.value,
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

    fun setWalkRestartAlpha(value: Float) {
        _walkRestartAlpha.value = value
        prefs.edit().putFloat("walk_restart_alpha", value).apply()
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

    fun setTextSearchTopK(value: Int) {
        _textSearchTopK.value = value
        prefs.edit().putInt("text_search_top_k", value).apply()
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
        val currentTrack = PowerampReceiver.getCurrentTrack(getApplication()) ?: return null
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

    /** Recent searches (persisted across sessions, includes full multi-seed state). */
    private val _recentSearches = MutableStateFlow<List<RecentSearch>>(loadRecentSearches())
    val recentSearches: StateFlow<List<RecentSearch>> = _recentSearches.asStateFlow()

    private fun loadRecentSearches(): List<RecentSearch> {
        // Try v2 JSON format first
        prefs.getString("recent_searches_v2", null)?.let { json ->
            try {
                return RecentSearch.fromJsonArray(json)
            } catch (e: Exception) {
                Log.w("MainViewModel", "Failed to parse recent_searches_v2", e)
            }
        }
        // Migrate from v1 (null-delimited text-only)
        prefs.getString("recent_searches", null)?.let { v1 ->
            val migrated = v1.split("\u0000").filter { it.isNotBlank() }
                .map { RecentSearch(textQuery = it) }
            if (migrated.isNotEmpty()) {
                prefs.edit()
                    .putString("recent_searches_v2", RecentSearch.toJsonArray(migrated))
                    .remove("recent_searches")
                    .apply()
            }
            return migrated
        }
        return emptyList()
    }

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

                    // Text model has INT64 ops → GPU always fails → CPU via XNNPACK.
                    // Try GPU first for future-proofing, fall back to CPU.
                    val candidates = buildList {
                        val fp32 = File(filesDir, "clamp3_text.tflite")
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

                // Generate text embedding. If inference fails on a cached model
                // (stale GPU context from cancelled indexing), destroy and retry once.
                val debugDir = File(filesDir, "debug_embeddings")
                var embedding = try {
                    inference.generateEmbedding(query, debugDir)
                } catch (e: Exception) {
                    Log.w("MainViewModel", "Text inference failed, will retry with fresh model", e)
                    null
                }
                if (embedding == null && textInference != null) {
                    Log.i("MainViewModel", "Destroying stale text model and retrying")
                    try { textInference?.close() } catch (_: Exception) {}
                    textInference = null
                    _textSearchResult.value = TextSearchResult(query = query, error = "Model error — retrying...")
                    // Retry: re-run performTextSearch which will lazy-init a fresh model
                    _textSearchLoading.value = false
                    performTextSearch(query)
                    return@launch
                }
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
                val topMatches = index.findTopK(embedding, topK = _textSearchTopK.value)
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
                saveRecentSearch(RecentSearch(textQuery = query))

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
        Log.i("MainViewModel", "START_RADIO_FROM_SONG: trackId=$trackId (single-seed radio)")
        RadioService.startRadioFromSeed(getApplication(), trackId, buildConfig())
    }

    /**
     * Queue a specific search result list directly into Poweramp.
     * The caller passes the exact result being displayed — no ambiguity.
     */
    fun queueDisplayedResults(result: TextSearchResult) {
        val tracks = result.matches.map { it.track }
        if (tracks.isEmpty()) return
        Log.i("MainViewModel", "QUEUE_DISPLAYED: queueing ${tracks.size} displayed results for '${result.query}'")
        for ((i, t) in tracks.withIndex()) {
            Log.d("MainViewModel", "QUEUE_DISPLAYED: [$i] ${t.artist} - ${t.title} (score=${result.matches[i].similarity})")
        }
        RadioService.queueDirectly(getApplication(), tracks, result.query)
    }

    fun clearTextSearchResult() {
        _textSearchResult.value = null
    }

    private fun saveRecentSearch(search: RecentSearch) {
        // Deduplicate by exact search state, not by human-facing label.
        val stateKey = search.stateKey
        val updated = (listOf(search) + _recentSearches.value.filter { it.stateKey != stateKey }).take(10)
        _recentSearches.value = updated
        prefs.edit().putString("recent_searches_v2", RecentSearch.toJsonArray(updated)).apply()
    }

    fun clearRecentSearches() {
        _recentSearches.value = emptyList()
        prefs.edit().remove("recent_searches_v2").remove("recent_searches").apply()
    }

    /**
     * Replay a recent search: restore full multi-seed state and re-run the search.
     * Returns the text query to put in the search field.
     */
    fun replayRecentSearch(search: RecentSearch) {
        if (search.songSeeds.isEmpty()) {
            // Text-only search — just run it
            if (search.textQuery.isNotBlank()) {
                performTextSearch(search.textQuery)
            }
            return
        }

        // Restore song seed state
        viewModelScope.launch(Dispatchers.IO) {
            val dbFile = File(getApplication<Application>().filesDir, "embeddings.db")
            val restoredSeeds = mutableListOf<SongSeedState>()
            if (dbFile.exists()) {
                val db = EmbeddingDatabase.open(dbFile)
                for (s in search.songSeeds) {
                    val track = db.getTrackById(s.trackId)
                    if (track != null) {
                        restoredSeeds.add(SongSeedState(
                            query = "${s.artist ?: ""} - ${s.title ?: ""}".trim(),
                            confirmedTrack = track,
                            weight = s.weight,
                            negative = s.negative,
                        ))
                    }
                }
                db.close()
            }

            if (restoredSeeds.isEmpty() && search.textQuery.isBlank()) return@launch

            // Renormalize weights if some tracks were lost
            if (restoredSeeds.size < search.songSeeds.size) {
                val totalWeight = (if (search.textQuery.isNotBlank()) search.textWeight else 0f) +
                    restoredSeeds.sumOf { it.weight.toDouble() }.toFloat()
                if (totalWeight > 0f) {
                    val scale = 1f / totalWeight
                    for (i in restoredSeeds.indices) {
                        restoredSeeds[i] = restoredSeeds[i].copy(weight = restoredSeeds[i].weight * scale)
                    }
                }
            }

            kotlinx.coroutines.withContext(Dispatchers.Main) {
                _songSeeds.value = restoredSeeds
                _textSeedWeight.value = when {
                    search.textQuery.isNotBlank() -> search.textWeight
                    restoredSeeds.isNotEmpty() -> 0f
                    else -> 1.0f
                }
                _textSeedLocked.value = false
                _textSeedNegative.value = search.textNegative
            }

            // Run the search
            _multiSeedLoading.value = true
            doMultiSeedSearch(search.textQuery, search.textQuery.isNotBlank(), restoredSeeds)
        }
    }

    // --- Multi-seed (song seed) state ---

    private val _songSeeds = MutableStateFlow<List<SongSeedState>>(emptyList())
    val songSeeds: StateFlow<List<SongSeedState>> = _songSeeds.asStateFlow()

    private val _songSeedSearchResults = MutableStateFlow<List<EmbeddedTrack>?>(null)
    val songSeedSearchResults: StateFlow<List<EmbeddedTrack>?> = _songSeedSearchResults.asStateFlow()

    /** Index of the song seed currently showing search results dropdown. */
    private val _activeSeedSearchIndex = MutableStateFlow(-1)
    val activeSeedSearchIndex: StateFlow<Int> = _activeSeedSearchIndex.asStateFlow()

    /** Multi-seed search results (separate from single text search results). */
    private val _multiSeedResult = MutableStateFlow<TextSearchResult?>(null)
    val multiSeedResult: StateFlow<TextSearchResult?> = _multiSeedResult.asStateFlow()

    private val _multiSeedLoading = MutableStateFlow(false)
    val multiSeedLoading: StateFlow<Boolean> = _multiSeedLoading.asStateFlow()

    private val _textSeedWeight = MutableStateFlow(1.0f)
    val textSeedWeight: StateFlow<Float> = _textSeedWeight.asStateFlow()

    private val _textSeedLocked = MutableStateFlow(false)
    val textSeedLocked: StateFlow<Boolean> = _textSeedLocked.asStateFlow()

    private val _textSeedNegative = MutableStateFlow(false)
    val textSeedNegative: StateFlow<Boolean> = _textSeedNegative.asStateFlow()

    fun toggleTextSeedSign() {
        _textSeedNegative.value = !_textSeedNegative.value
    }

    fun toggleTextSeedLock() {
        _textSeedLocked.value = !_textSeedLocked.value
    }

    companion object {
        /** Minimum weight per seed — prevents any seed from being zeroed out. */
        private const val MIN_WEIGHT = 0.01f
        private const val UNINDEXED_AUTO_REFRESH_MS = 24L * 60L * 60L * 1000L
    }

    /**
     * Budget available for unlocked seeds = 1.0 − sum(locked weights).
     * Each unlocked seed gets at least [MIN_WEIGHT].
     */
    private fun unlockedBudget(): Float {
        val lockedSum = lockedWeightSum()
        return (1f - lockedSum).coerceAtLeast(0f)
    }

    private fun lockedWeightSum(): Float {
        var sum = 0f
        if (_textSeedLocked.value) sum += _textSeedWeight.value
        for (s in _songSeeds.value) {
            if (s.locked) sum += s.weight
        }
        return sum
    }

    /**
     * Update text seed weight during drag. Only updates this knob — no redistribution.
     * Redistribution + auto-lock happen in [finalizeTextSeedWeight] on drag end.
     */
    fun updateTextSeedWeight(weight: Float) {
        val budget = unlockedBudget()
        val otherUnlockedCount = _songSeeds.value.count { !it.locked }
        // Text weight can go to 0 when song seeds exist (song-only mode)
        val minForText = if (_songSeeds.value.isNotEmpty()) 0f else MIN_WEIGHT
        val maxForThis = (budget - otherUnlockedCount * MIN_WEIGHT).coerceAtLeast(minForText)
        _textSeedWeight.value = weight.coerceIn(minForText, maxForThis)
    }

    /** Called on drag end: redistribute remaining budget among unlocked seeds. */
    fun finalizeTextSeedWeight() {
        redistributeUnlocked(changedIndex = -1)
    }

    /**
     * Update song seed weight during drag. Only updates this knob — no redistribution.
     * Redistribution + auto-lock happen in [finalizeSongSeedWeight] on drag end.
     */
    fun updateSongSeedWeight(index: Int, weight: Float) {
        val current = _songSeeds.value.toMutableList()
        if (index !in current.indices) return

        val budget = unlockedBudget()
        // Reserve MIN_WEIGHT for each other unlocked song seed.
        // Text seed doesn't need reservation — it can go to 0 when song seeds exist.
        var otherReserve = 0f
        for (i in current.indices) {
            if (i != index && !current[i].locked) otherReserve += MIN_WEIGHT
        }
        val maxForThis = (budget - otherReserve).coerceAtLeast(MIN_WEIGHT)
        current[index] = current[index].copy(weight = weight.coerceIn(MIN_WEIGHT, maxForThis))
        _songSeeds.value = current
    }

    /** Called on drag end: redistribute remaining budget and auto-lock the song seed. */
    fun finalizeSongSeedWeight(index: Int) {
        if (index !in _songSeeds.value.indices) return
        redistributeUnlocked(changedIndex = index)
    }

    /**
     * Redistribute unlocked budget among unlocked seeds (excluding changedIndex).
     * Locked seeds are never touched. changedIndex = -1 means text seed changed.
     * The changed seed keeps its new value; the remaining unlocked budget is split
     * proportionally among the other unlocked seeds.
     *
     * If no other unlocked seeds exist, the changed seed is snapped to the full
     * unlocked budget (it must consume 100% of what's left after locked seeds).
     */
    private fun redistributeUnlocked(changedIndex: Int) {
        val budget = unlockedBudget()

        // Collect other unlocked seeds and their current weights
        val others = mutableListOf<Pair<Int, Float>>() // index -> weight (-1 = text)
        if (!_textSeedLocked.value && changedIndex != -1) {
            others.add(-1 to _textSeedWeight.value)
        }
        val seeds = _songSeeds.value
        for (i in seeds.indices) {
            if (i != changedIndex && !seeds[i].locked) {
                others.add(i to seeds[i].weight)
            }
        }

        // If no other unlocked seeds, snap the changed seed to the full budget
        if (others.isEmpty()) {
            val newSeeds = seeds.toMutableList()
            when (changedIndex) {
                -1 -> {
                    // Text can be 0 when song seeds exist
                    val floor = if (seeds.isNotEmpty()) 0f else MIN_WEIGHT
                    _textSeedWeight.value = budget.coerceAtLeast(floor)
                }
                else -> {
                    newSeeds[changedIndex] = newSeeds[changedIndex].copy(
                        weight = budget.coerceAtLeast(MIN_WEIGHT)
                    )
                    _songSeeds.value = newSeeds
                }
            }
            return
        }

        // How much budget is left after the changed seed?
        // If the changed seed is locked (auto-lock on drag end), the budget already excludes it.
        // If it's unlocked (live drag), subtract it from the budget.
        val changedIsLocked = when (changedIndex) {
            -1 -> _textSeedLocked.value
            else -> seeds[changedIndex].locked
        }
        // Min reserve for others: text can be 0 when song seeds exist, songs need MIN_WEIGHT
        val othersMinReserve = others.sumOf { (idx, _) ->
            if (idx == -1 && seeds.isNotEmpty()) 0.0 else MIN_WEIGHT.toDouble()
        }.toFloat()
        val remaining = if (changedIsLocked) {
            budget.coerceAtLeast(othersMinReserve)
        } else {
            val changedWeight = when (changedIndex) {
                -1 -> _textSeedWeight.value
                else -> seeds[changedIndex].weight
            }
            (budget - changedWeight).coerceAtLeast(othersMinReserve)
        }

        // Distribute proportionally among others
        val totalOthers = others.sumOf { it.second.toDouble() }.toFloat()
        val newSeeds = seeds.toMutableList()

        if (totalOthers <= 0.001f) {
            // Equal split as fallback
            val each = remaining / others.size
            for ((idx, _) in others) {
                if (idx == -1) _textSeedWeight.value = each
                else newSeeds[idx] = newSeeds[idx].copy(weight = each)
            }
        } else {
            var assigned = 0f
            for (j in others.indices) {
                val (idx, w) = others[j]
                val floor = if (idx == -1 && seeds.isNotEmpty()) 0f else MIN_WEIGHT
                val share = if (j == others.lastIndex) {
                    // Last one gets remainder to avoid rounding drift
                    (remaining - assigned).coerceAtLeast(floor)
                } else {
                    (w / totalOthers * remaining).coerceAtLeast(floor)
                }
                assigned += share
                if (idx == -1) _textSeedWeight.value = share
                else newSeeds[idx] = newSeeds[idx].copy(weight = share)
            }
        }
        _songSeeds.value = newSeeds
    }

    /**
     * Renormalize only unlocked weights so all weights sum to 1.0.
     * Locked weights are never changed.
     */
    private fun renormalize() {
        val lockedSum = lockedWeightSum()
        val targetUnlocked = (1f - lockedSum).coerceAtLeast(0f)

        val seeds = _songSeeds.value
        var unlockSum = 0f
        if (!_textSeedLocked.value) unlockSum += _textSeedWeight.value
        for (s in seeds) {
            if (!s.locked) unlockSum += s.weight
        }

        if (unlockSum <= 0.001f || targetUnlocked <= 0.001f) return
        val scale = targetUnlocked / unlockSum

        if (!_textSeedLocked.value) {
            // Text can be 0 when song seeds exist (song-only mode)
            val textFloor = if (seeds.isNotEmpty()) 0f else MIN_WEIGHT
            _textSeedWeight.value = (_textSeedWeight.value * scale).coerceAtLeast(textFloor)
        }
        val newSeeds = seeds.map {
            if (!it.locked) it.copy(weight = (it.weight * scale).coerceAtLeast(MIN_WEIGHT))
            else it
        }
        _songSeeds.value = newSeeds
    }

    fun clearSongSeeds() {
        _songSeeds.value = emptyList()
        _songSeedSearchResults.value = null
        _activeSeedSearchIndex.value = -1
        _textSeedWeight.value = 1.0f
        _textSeedLocked.value = false
        _textSeedNegative.value = false
    }

    fun addSongSeed(hasTextSeed: Boolean) {
        val isFirst = _songSeeds.value.isEmpty()
        if (isFirst && !_textSeedLocked.value) {
            // First song seed:
            // - blank text box => song-only mode (0/100)
            // - non-blank text box => split evenly (50/50)
            _textSeedWeight.value = if (hasTextSeed) 0.5f else 0f
        }
        val budget = unlockedBudget()
        // Don't count text at 0% in budget split — it doesn't participate
        val textParticipates = !_textSeedLocked.value && _textSeedWeight.value > 0f
        var currentUnlockedCount = if (textParticipates) 1 else 0
        for (s in _songSeeds.value) { if (!s.locked) currentUnlockedCount++ }
        val newTotalUnlocked = currentUnlockedCount + 1
        val shareForNew = (budget / newTotalUnlocked).coerceAtLeast(MIN_WEIGHT)
        _songSeeds.value = _songSeeds.value + SongSeedState(weight = shareForNew)
        renormalize()
    }

    private fun countUnlocked(): Int {
        var n = 0
        if (!_textSeedLocked.value) n++
        for (s in _songSeeds.value) {
            if (!s.locked) n++
        }
        return n
    }

    fun removeSongSeed(index: Int) {
        val current = _songSeeds.value.toMutableList()
        if (index in current.indices) {
            current.removeAt(index)
            _songSeeds.value = current
        }
        if (_activeSeedSearchIndex.value == index) {
            _songSeedSearchResults.value = null
            _activeSeedSearchIndex.value = -1
        }
        if (current.isEmpty()) {
            // Removed last seed — clear multi-seed results and reset text weight
            _multiSeedResult.value = null
            _textSeedWeight.value = 1.0f
            _textSeedLocked.value = false
            _textSeedNegative.value = false
        } else {
            renormalize()
        }
    }

    fun updateSongSeedQuery(index: Int, query: String) {
        val current = _songSeeds.value.toMutableList()
        if (index in current.indices) {
            current[index] = current[index].copy(query = query, confirmedTrack = null)
            _songSeeds.value = current
        }
    }

    fun confirmSongSeed(index: Int, track: EmbeddedTrack) {
        val current = _songSeeds.value.toMutableList()
        if (index in current.indices) {
            current[index] = current[index].copy(
                query = "${track.artist ?: ""} - ${track.title ?: ""}".trim(),
                confirmedTrack = track,
            )
            _songSeeds.value = current
        }
        _songSeedSearchResults.value = null
        _activeSeedSearchIndex.value = -1
    }

    fun toggleSongSeedLock(index: Int) {
        val current = _songSeeds.value.toMutableList()
        if (index in current.indices) {
            current[index] = current[index].copy(locked = !current[index].locked)
            _songSeeds.value = current
        }
    }

    fun toggleSongSeedSign(index: Int) {
        val current = _songSeeds.value.toMutableList()
        if (index in current.indices) {
            current[index] = current[index].copy(negative = !current[index].negative)
            _songSeeds.value = current
        }
    }

    fun searchSongSeed(index: Int) {
        val seeds = _songSeeds.value
        if (index !in seeds.indices) return
        val query = seeds[index].query.trim()
        if (query.isBlank()) return

        _activeSeedSearchIndex.value = index
        viewModelScope.launch(Dispatchers.IO) {
            try {
                val dbFile = File(getApplication<Application>().filesDir, "embeddings.db")
                if (!dbFile.exists()) return@launch
                val db = EmbeddingDatabase.open(dbFile)
                val results = db.searchTracksByText(query)
                db.close()
                _songSeedSearchResults.value = results
            } catch (e: Exception) {
                Log.e("MainViewModel", "Song seed search failed", e)
                _songSeedSearchResults.value = emptyList()
            }
        }
    }

    fun dismissSongSeedSearch() {
        _songSeedSearchResults.value = null
        _activeSeedSearchIndex.value = -1
    }

    /**
     * Perform a multi-seed search: text query + song seeds → geo mean of percentiles.
     * If only text query with no song seeds, degenerates to regular text search.
     */
    fun performMultiSeedSearch(textQuery: String) {
        val hasText = textQuery.isNotBlank()

        // Auto-confirm unconfirmed seeds that have query text
        _multiSeedLoading.value = true
        viewModelScope.launch(Dispatchers.IO) {
            val seeds = _songSeeds.value.toMutableList()
            var changed = false
            val dbFile = File(getApplication<Application>().filesDir, "embeddings.db")
            if (dbFile.exists()) {
                val lookupDb = EmbeddingDatabase.open(dbFile)
                for (i in seeds.indices) {
                    if (seeds[i].confirmedTrack == null && seeds[i].query.isNotBlank()) {
                        val matches = lookupDb.searchTracksByText(seeds[i].query)
                        if (matches.isNotEmpty()) {
                            val track = matches.first()
                            seeds[i] = seeds[i].copy(
                                query = "${track.artist ?: ""} - ${track.title ?: ""}".trim(),
                                confirmedTrack = track,
                            )
                            changed = true
                            Log.d("MultiSeed", "Auto-confirmed seed[$i] '${seeds[i].query}' → ${track.artist} - ${track.title}")
                        }
                    }
                }
                lookupDb.close()
            }
            if (changed) {
                _songSeeds.value = seeds
            }
            // Now proceed on IO thread
            doMultiSeedSearch(textQuery, hasText, seeds)
        }
    }

    private suspend fun doMultiSeedSearch(textQuery: String, hasText: Boolean, seeds: List<SongSeedState>) {
        val confirmedSeeds = seeds.filter { it.confirmedTrack != null && it.weight != 0f }

        Log.d("MultiSeed", "performMultiSeedSearch: text='$textQuery', " +
            "songSeeds=${seeds.size}, confirmedSeeds=${confirmedSeeds.size}")
        for ((i, s) in confirmedSeeds.withIndex()) {
            Log.d("MultiSeed", "  seed[$i]: '${s.confirmedTrack?.title}' " +
                "weight=${s.weight} negative=${s.negative} → effective=${if (s.negative) -s.weight else s.weight}")
        }

        // If no song seeds, fall back to regular text search
        if (confirmedSeeds.isEmpty() && hasText) {
            Log.d("MultiSeed", "No confirmed song seeds, falling back to text-only search")
            kotlinx.coroutines.withContext(Dispatchers.Main) { performTextSearch(textQuery) }
            _multiSeedLoading.value = false
            return
        }

        if (confirmedSeeds.isEmpty() && !hasText) {
            _multiSeedLoading.value = false
            return
        }

        _multiSeedResult.value = null

        try {
            val filesDir = getApplication<Application>().filesDir
            val dbFile = File(filesDir, "embeddings.db")
            if (!dbFile.exists()) {
                _multiSeedResult.value = TextSearchResult(query = textQuery, error = "No embedding database found")
                return
            }

            val seedSpecs = mutableListOf<SeedSpec>()

            // Text seed (skip if weight is 0 — no influence)
            val textWeight = _textSeedWeight.value
            val textNegative = _textSeedNegative.value
            if (hasText && textWeight != 0f) {
                val inference = getOrInitTextInference(textQuery) ?: return
                val debugDir = File(filesDir, "debug_embeddings")
                val embedding = try {
                    inference.generateEmbedding(textQuery.trim(), debugDir)
                } catch (e: Exception) {
                    Log.w("MainViewModel", "Text inference failed in multi-seed", e)
                    null
                }
                if (embedding == null) {
                    _multiSeedResult.value = TextSearchResult(query = textQuery, error = "Text inference failed")
                    return
                }
                val effectiveTextWeight = if (textNegative) -textWeight else textWeight
                seedSpecs.add(SeedSpec(
                    embedding = embedding,
                    weight = effectiveTextWeight,
                    label = textQuery.trim(),
                    type = SeedType.TEXT,
                ))
            }

            // Song seeds
            val db = EmbeddingDatabase.open(dbFile)
            for (seed in confirmedSeeds) {
                val track = seed.confirmedTrack!!
                val embedding = db.getEmbedding(track.id)
                if (embedding != null) {
                    val effectiveWeight = if (seed.negative) -seed.weight else seed.weight
                    seedSpecs.add(SeedSpec(
                        embedding = embedding,
                        weight = effectiveWeight,
                        label = "${track.artist ?: "?"} - ${track.title ?: "?"}",
                        type = SeedType.SONG,
                        trackId = track.id,
                    ))
                }
            }
            db.close()

            if (seedSpecs.isEmpty()) {
                _multiSeedResult.value = TextSearchResult(query = textQuery, error = "No valid seeds")
                return
            }

            // Use embedding index for geo-mean ranking
            val index = textIndex ?: run {
                val embFile = File(filesDir, "clamp3.emb")
                if (!embFile.exists()) {
                    val db2 = EmbeddingDatabase.open(dbFile)
                    EmbeddingIndex.extractFromDatabase(db2, embFile, table = "embeddings_clamp3")
                    db2.close()
                }
                if (!embFile.exists()) {
                    _multiSeedResult.value = TextSearchResult(query = textQuery, error = "Failed to extract index")
                    return
                }
                EmbeddingIndex.mmap(embFile).also { textIndex = it }
            }

            val excludeIds = seedSpecs.mapNotNull { it.trackId }.toSet()
            val topK = _textSearchTopK.value

            Log.d("MultiSeed", "Calling GeoMeanSelector with ${seedSpecs.size} specs:")
            for ((i, spec) in seedSpecs.withIndex()) {
                Log.d("MultiSeed", "  spec[$i]: type=${spec.type} label='${spec.label}' weight=${spec.weight}")
            }

            val seedVectors = seedSpecs.map { it.embedding to it.weight }
            val ranking = com.powerampstartradio.similarity.algorithms.GeoMeanSelector.computeRanking(
                index,
                seedVectors,
                topK,
                excludeIds,
            )
            val displaySims = com.powerampstartradio.similarity.algorithms.GeoMeanSelector
                .computeDisplaySimilarities(index, seedVectors)

            // Resolve track metadata
            val db3 = EmbeddingDatabase.open(dbFile)
            val matches = ranking.mapNotNull { (trackId, score) ->
                db3.getTrackById(trackId)?.let { track ->
                    TextSearchMatch(
                        track = track,
                        similarity = index.getSimFromPrecomputed(displaySims, trackId),
                        rankingScore = score,
                    )
                }
            }
            db3.close()

            Log.d("MultiSeed", "GeoMeanSelector returned ${ranking.size} results")
            if (ranking.isNotEmpty()) {
                Log.d("MultiSeed", "  top score: ${ranking.first().second}")
            }

            // Structured verification logging (replay on desktop with multi_seed_search.py)
            val queryJson = JSONObject().apply {
                if (hasText) {
                    put("text", textQuery)
                    put("text_weight", textWeight.toDouble())
                    if (textNegative) put("text_negative", true)
                }
                val seedArr = JSONArray()
                for (s in confirmedSeeds) {
                    seedArr.put(JSONObject().apply {
                        put("track_id", s.confirmedTrack!!.id)
                        s.confirmedTrack.artist?.let { put("artist", it) }
                        s.confirmedTrack.title?.let { put("title", it) }
                        val effectiveWeight = if (s.negative) -s.weight else s.weight
                        put("weight", effectiveWeight.toDouble())
                    })
                }
                put("seeds", seedArr)
            }
            Log.i("MultiSeed", "MULTISEED_QUERY: $queryJson")

            val resultsJson = JSONArray()
            for ((i, m) in matches.withIndex()) {
                resultsJson.put(JSONObject().apply {
                    put("rank", i)
                    put("track_id", m.track.id)
                    m.track.artist?.let { put("artist", it) }
                    m.track.title?.let { put("title", it) }
                    put("score", m.rankingScore.toDouble())
                    put("display_similarity", m.similarity.toDouble())
                })
            }
            Log.i("MultiSeed", "MULTISEED_RESULTS: $resultsJson")

            val queryLabel = formatSearchLabel(
                textQuery = textQuery.takeIf { hasText }?.let {
                    val label = "$it (${(textWeight * 100).roundToInt()}%)"
                    if (textNegative) "- $label" else label
                },
                songSeeds = confirmedSeeds.map { s ->
                    SearchLabelPart(
                        text = "${s.confirmedTrack!!.title} (${(s.weight * 100).roundToInt()}%)",
                        negative = s.negative
                    )
                }
            )

            _multiSeedResult.value = TextSearchResult(query = queryLabel, matches = matches)

            // Save full multi-seed state to recent searches
            saveRecentSearch(RecentSearch(
                textQuery = if (hasText) textQuery else "",
                textWeight = if (hasText) textWeight else 0f,
                textNegative = if (hasText) textNegative else false,
                songSeeds = confirmedSeeds.map { s ->
                    RecentSongSeed(
                        trackId = s.confirmedTrack!!.id,
                        artist = s.confirmedTrack.artist,
                        title = s.confirmedTrack.title,
                        weight = s.weight,
                        negative = s.negative,
                    )
                },
            ))

        } catch (e: Exception) {
            Log.e("MainViewModel", "Multi-seed search failed", e)
            _multiSeedResult.value = TextSearchResult(query = textQuery, error = "Search failed: ${e.message}")
        } finally {
            _multiSeedLoading.value = false
        }
    }

    /** Helper: get or init text inference, returning null (with error set) on failure. */
    private fun getOrInitTextInference(queryForError: String): com.powerampstartradio.indexing.Clamp3TextInference? {
        textInference?.let { return it }
        val filesDir = getApplication<Application>().filesDir
        val vocabFile = File(filesDir, "xlm_roberta_vocab.json")
        if (!vocabFile.exists()) {
            _multiSeedResult.value = TextSearchResult(query = queryForError, error = "Tokenizer vocab not found")
            return null
        }
        val candidates = buildList {
            val fp32 = File(filesDir, "clamp3_text.tflite")
            if (fp32.exists()) {
                add(fp32 to com.google.ai.edge.litert.Accelerator.GPU)
                add(fp32 to com.google.ai.edge.litert.Accelerator.CPU)
            }
        }
        if (candidates.isEmpty()) {
            _multiSeedResult.value = TextSearchResult(query = queryForError, error = "CLaMP3 text model not found")
            return null
        }
        var lastError: Exception? = null
        for ((modelFile, accel) in candidates) {
            try {
                return com.powerampstartradio.indexing.Clamp3TextInference(modelFile, vocabFile, accel)
                    .also { textInference = it }
            } catch (e: Exception) {
                lastError = e
            }
        }
        _multiSeedResult.value = TextSearchResult(query = queryForError, error = "Failed to load text model: ${lastError?.message}")
        return null
    }

    fun clearMultiSeedResult() {
        _multiSeedResult.value = null
    }

    /**
     * Start radio from multi-seed search results.
     * Builds SeedSpec list from current state and sends to RadioService.
     */
    fun startRadioFromMultiSeed(textQuery: String) {
        val seeds = _songSeeds.value
        val confirmedSeeds = seeds.filter { it.confirmedTrack != null && it.weight != 0f }

        viewModelScope.launch(Dispatchers.IO) {
            try {
                val filesDir = getApplication<Application>().filesDir
                val dbFile = File(filesDir, "embeddings.db")
                if (!dbFile.exists()) return@launch

                val seedSpecs = mutableListOf<SeedSpec>()

                // Text seed (skip if weight is 0)
                val textWeight = _textSeedWeight.value
                val textNeg = _textSeedNegative.value
                if (textQuery.isNotBlank() && textWeight != 0f) {
                    val inference = textInference ?: return@launch
                    val debugDir = File(filesDir, "debug_embeddings")
                    val embedding = inference.generateEmbedding(textQuery.trim(), debugDir) ?: return@launch
                    val effTextWeight = if (textNeg) -textWeight else textWeight
                    seedSpecs.add(SeedSpec(
                        embedding = embedding,
                        weight = effTextWeight,
                        label = textQuery.trim(),
                        type = SeedType.TEXT,
                    ))
                }

                // Song seeds
                val db = EmbeddingDatabase.open(dbFile)
                for (seed in confirmedSeeds) {
                    val track = seed.confirmedTrack!!
                    val embedding = db.getEmbedding(track.id)
                    if (embedding != null) {
                        val effectiveWeight = if (seed.negative) -seed.weight else seed.weight
                        seedSpecs.add(SeedSpec(
                            embedding = embedding,
                            weight = effectiveWeight,
                            label = "${track.artist ?: "?"} - ${track.title ?: "?"}",
                            type = SeedType.SONG,
                            trackId = track.id,
                        ))
                    }
                }
                db.close()

                if (seedSpecs.isEmpty()) return@launch

                val config = buildConfig()
                RadioService.startRadioFromMultiSeed(getApplication(), seedSpecs, config)
            } catch (e: Exception) {
                Log.e("MainViewModel", "startRadioFromMultiSeed failed", e)
            }
        }
    }

    // --- Indexing actions ---

    fun checkUnindexedTracks() {
        if (_unindexedCount.value == -1) return
        Log.i("MainViewModel", "Starting unindexed track check")
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
            IndexingViewModel.cacheResults(sorted, getDbFingerprint(dbFile), powerampCount)
            sorted
        }
        // Expose so IndexingViewModel can await if user opens Manage Tracks mid-check
        IndexingViewModel.pendingDetection = deferred

        viewModelScope.launch {
            try {
                val result = deferred.await()
                // DB fingerprint changes should not wipe ignored / never-index choices.
                // Those are Poweramp-library choices, not DB-specific state.
                val indexingPrefs = app.getSharedPreferences("indexing", Context.MODE_PRIVATE)
                val currentFingerprint = if (dbFile.exists())
                    "${dbFile.length()}_${dbFile.lastModified()}" else ""
                val savedFingerprint = indexingPrefs.getString("dismissed_db_fingerprint", "") ?: ""
                if (currentFingerprint != savedFingerprint) {
                    indexingPrefs.edit()
                        .putString("dismissed_db_fingerprint", currentFingerprint)
                        .apply()
                    Log.i("MainViewModel", "DB fingerprint changed during unindexed recount; preserved hidden track choices")
                }
                // Exclude dismissed and ignored tracks from the count
                val dismissedJson = indexingPrefs.getString("dismissed_track_ids", null)
                val dismissed = if (dismissedJson != null) {
                    try {
                        val arr = org.json.JSONArray(dismissedJson)
                        (0 until arr.length()).map { arr.getLong(it) }.toSet()
                    } catch (_: Exception) { emptySet() }
                } else emptySet<Long>()
                val ignoredJson = indexingPrefs.getString("ignored_track_ids", null)
                val ignored = if (ignoredJson != null) {
                    try {
                        val arr = org.json.JSONArray(ignoredJson)
                        (0 until arr.length()).map { arr.getLong(it) }.toSet()
                    } catch (_: Exception) { emptySet() }
                } else emptySet<Long>()
                val visible = result.count {
                    it.durationMs > 0 &&
                        it.powerampFileId !in dismissed &&
                        it.powerampFileId !in ignored
                }
                setUnindexedCount(visible)
                Log.i("MainViewModel", "Unindexed track check complete: $visible visible tracks")
            } catch (_: Exception) {
                setUnindexedCount(0)
            } finally {
                IndexingViewModel.pendingDetection = null
                IndexingViewModel.detectionStatus.value = null
                _unindexedCheckStatus.value = null
            }
        }
    }

    fun maybeRefreshUnindexedTracks() {
        val app = getApplication<Application>()
        val dbFile = File(app.filesDir, "embeddings.db")
        if (!dbFile.exists() || !_hasPermission.value) return
        if (IndexingService.state.value !is IndexingService.IndexingState.Idle) return

        val currentFp = getDbFingerprint(dbFile)
        val savedFp = prefs.getString("unindexed_count_db_fingerprint", "") ?: ""
        val lastCheckedMs = prefs.getLong("unindexed_last_checked_ms", 0L)
        val now = System.currentTimeMillis()

        val refreshReason = when {
            _unindexedCount.value == -1 -> null
            _unindexedCount.value == -2 -> "count not checked yet"
            currentFp != savedFp -> "database fingerprint changed"
            lastCheckedMs <= 0L -> "missing freshness timestamp"
            now - lastCheckedMs >= UNINDEXED_AUTO_REFRESH_MS -> "stale after ${UNINDEXED_AUTO_REFRESH_MS / (60L * 60L * 1000L)}h"
            else -> null
        }

        if (refreshReason != null) {
            Log.i("MainViewModel", "Auto-refreshing unindexed count: $refreshReason")
            checkUnindexedTracks()
        }
    }

    private fun setUnindexedCount(count: Int) {
        _unindexedCount.value = count
        val dbFile = File(getApplication<Application>().filesDir, "embeddings.db")
        prefs.edit()
            .putInt("unindexed_count", count)
            .putLong("unindexed_last_checked_ms", System.currentTimeMillis())
            .putString("unindexed_count_db_fingerprint", getDbFingerprint(dbFile))
            .apply()
    }

    private fun syncIndexingDbFingerprint() {
        val app = getApplication<Application>()
        val indexingPrefs = app.getSharedPreferences("indexing", Context.MODE_PRIVATE)
        val dbFile = File(app.filesDir, "embeddings.db")
        val fingerprint = getDbFingerprint(dbFile)
        if (fingerprint.isNotEmpty()) {
            indexingPrefs.edit()
                .putString("dismissed_db_fingerprint", fingerprint)
                .apply()
            Log.i("MainViewModel", "Synced indexing DB fingerprint after app-owned DB update")
        }
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

    private fun getDbFingerprint(dbFile: File): String =
        if (dbFile.exists()) "${dbFile.length()}_${dbFile.lastModified()}" else ""

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
        setWalkRestartAlpha(defaults.walkRestartAlpha)
        setAnchorDecay(defaults.anchorDecay)
        setMomentumBeta(defaults.momentumBeta)
        setDiversityLambda(defaults.diversityLambda)
        setMaxPerArtist(defaults.maxPerArtist)
        setMinArtistSpacing(defaults.minArtistSpacing)
        setTextSearchTopK(20)
    }

    fun prepareIndices() {
        if (!indicesPreparing.compareAndSet(false, true)) return
        viewModelScope.launch(Dispatchers.IO) {
            try {
                prepareIndicesWithProgress { message ->
                    _indexStatus.value = message
                }
            } finally {
                indicesPreparing.set(false)
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
    val rankingScore: Float = similarity,
)

data class TextSearchResult(
    val query: String,
    val matches: List<TextSearchMatch> = emptyList(),
    val error: String? = null,
)

/**
 * State for a single song seed in multi-seed search.
 */
data class SongSeedState(
    val id: Long = nextSeedId(),
    val query: String = "",
    val confirmedTrack: EmbeddedTrack? = null,
    val weight: Float = 1.0f,
    val negative: Boolean = false,  // true = "less like this"
    val locked: Boolean = false,
)

private var seedIdCounter = 0L
fun nextSeedId(): Long = seedIdCounter++

/**
 * A saved song seed for recent search replay.
 */
data class RecentSongSeed(
    val trackId: Long,
    val artist: String?,
    val title: String?,
    val weight: Float,
    val negative: Boolean,
)

/**
 * A recent search entry with full multi-seed state.
 */
data class RecentSearch(
    val textQuery: String,
    val textWeight: Float = 1.0f,
    val textNegative: Boolean = false,
    val songSeeds: List<RecentSongSeed> = emptyList(),
) {
    /** Composite display label: "sufi music + Ragini - Ektaal" */
    val displayLabel: String
        get() = formatSearchLabel(
            textQuery = textQuery.takeIf { it.isNotBlank() }?.let {
                if (textNegative) "- $it" else it
            },
            songSeeds = songSeeds.map { s ->
                SearchLabelPart(
                    text = s.title ?: "?",
                    negative = s.negative,
                )
            }
        )

    /** Stable identity for deduping exact recent-search state. */
    val stateKey: String
        get() = buildString {
            append(textQuery)
            append('|')
            append(textWeight)
            append('|')
            append(textNegative)
            for (s in songSeeds) {
                append('|')
                append(s.trackId)
                append(':')
                append(s.weight)
                append(':')
                append(s.negative)
            }
        }

    companion object {
        fun toJsonArray(list: List<RecentSearch>): String {
            val arr = JSONArray()
            for (search in list) {
                val obj = JSONObject()
                obj.put("text", search.textQuery)
                obj.put("text_weight", search.textWeight.toDouble())
                obj.put("text_negative", search.textNegative)
                if (search.songSeeds.isNotEmpty()) {
                    val seedArr = JSONArray()
                    for (s in search.songSeeds) {
                        val seedObj = JSONObject()
                        seedObj.put("id", s.trackId)
                        s.artist?.let { seedObj.put("artist", it) }
                        s.title?.let { seedObj.put("title", it) }
                        seedObj.put("weight", s.weight.toDouble())
                        seedObj.put("negative", s.negative)
                        seedArr.put(seedObj)
                    }
                    obj.put("seeds", seedArr)
                }
                arr.put(obj)
            }
            return arr.toString()
        }

        fun fromJsonArray(json: String): List<RecentSearch> {
            val arr = JSONArray(json)
            val result = mutableListOf<RecentSearch>()
            for (i in 0 until arr.length()) {
                val obj = arr.getJSONObject(i)
                val seeds = if (obj.has("seeds")) {
                    val seedArr = obj.getJSONArray("seeds")
                    (0 until seedArr.length()).map { j ->
                        val s = seedArr.getJSONObject(j)
                        RecentSongSeed(
                            trackId = s.getLong("id"),
                            artist = if (s.has("artist")) s.getString("artist") else null,
                            title = if (s.has("title")) s.getString("title") else null,
                            weight = s.getDouble("weight").toFloat(),
                            negative = s.optBoolean("negative", false),
                        )
                    }
                } else emptyList()
                result.add(RecentSearch(
                    textQuery = obj.optString("text", ""),
                    textWeight = when {
                        obj.has("text_weight") -> obj.getDouble("text_weight").toFloat()
                        obj.optString("text", "").isBlank() -> 0f
                        else -> 1.0f
                    },
                    textNegative = obj.optBoolean("text_negative", false),
                    songSeeds = seeds,
                ))
            }
            return result
        }
    }
}

private data class SearchLabelPart(
    val text: String,
    val negative: Boolean,
)

private fun formatSearchLabel(
    textQuery: String?,
    songSeeds: List<SearchLabelPart>,
): String {
    val parts = mutableListOf<String>()

    textQuery?.takeIf { it.isNotBlank() }?.let { parts += it }

    for ((index, seed) in songSeeds.withIndex()) {
        val isFirstVisiblePart = parts.isEmpty()
        val seedText = when {
            seed.negative -> "- ${seed.text}"
            isFirstVisiblePart && index == 0 -> seed.text
            else -> "+ ${seed.text}"
        }
        parts += seedText
    }

    if (parts.isEmpty()) return ""

    return buildString {
        append(parts.first())
        for (i in 1 until parts.size) {
            val part = parts[i]
            if (part.startsWith("- ")) {
                append(" - ")
                append(part.removePrefix("- "))
            } else if (part.startsWith("+ ")) {
                append(" + ")
                append(part.removePrefix("+ "))
            } else {
                append(" ")
                append(part)
            }
        }
    }
}
