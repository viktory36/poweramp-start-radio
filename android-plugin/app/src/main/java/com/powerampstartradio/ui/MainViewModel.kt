package com.powerampstartradio.ui

import android.app.Application
import android.content.Context
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.powerampstartradio.data.EmbeddingDatabase
import com.powerampstartradio.data.EmbeddingModel
import com.powerampstartradio.poweramp.PowerampHelper
import com.powerampstartradio.services.RadioService
import com.powerampstartradio.similarity.AnchorExpandConfig
import com.powerampstartradio.similarity.SearchStrategy
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

    // Track count setting
    private val _numTracks = MutableStateFlow(prefs.getInt("num_tracks", RadioService.DEFAULT_NUM_TRACKS))
    val numTracks: StateFlow<Int> = _numTracks.asStateFlow()

    // Search strategy settings
    private val _searchStrategy = MutableStateFlow(
        try {
            val stored = prefs.getString("search_strategy", SearchStrategy.ANCHOR_EXPAND.name)!!
            // Migrate from old name
            if (stored == "FEED_FORWARD") SearchStrategy.ANCHOR_EXPAND
            else SearchStrategy.valueOf(stored)
        } catch (e: IllegalArgumentException) {
            SearchStrategy.ANCHOR_EXPAND
        }
    )
    val searchStrategy: StateFlow<SearchStrategy> = _searchStrategy.asStateFlow()

    private val _anchorExpandPrimary = MutableStateFlow(
        try {
            val stored = prefs.getString("anchor_expand_primary", null)
                ?: prefs.getString("feed_forward_primary", EmbeddingModel.MULAN.name)
            EmbeddingModel.valueOf(stored!!)
        } catch (e: IllegalArgumentException) {
            EmbeddingModel.MULAN
        }
    )
    val anchorExpandPrimary: StateFlow<EmbeddingModel> = _anchorExpandPrimary.asStateFlow()

    private val _anchorExpandExpansion = MutableStateFlow(
        prefs.getInt("anchor_expand_expansion",
            prefs.getInt("feed_forward_expansion", 3))
    )
    val anchorExpandExpansion: StateFlow<Int> = _anchorExpandExpansion.asStateFlow()

    // Drift mode: each result seeds the next search
    private val _drift = MutableStateFlow(prefs.getBoolean("drift", false))
    val drift: StateFlow<Boolean> = _drift.asStateFlow()

    // Database info
    private val _databaseInfo = MutableStateFlow<DatabaseInfo?>(null)
    val databaseInfo: StateFlow<DatabaseInfo?> = _databaseInfo.asStateFlow()

    // Poweramp permission state
    private val _hasPermission = MutableStateFlow(false)
    val hasPermission: StateFlow<Boolean> = _hasPermission.asStateFlow()

    // Radio state from service
    val radioState: StateFlow<RadioUiState> = RadioService.uiState

    // Session history
    val sessionHistory: StateFlow<List<RadioResult>> = RadioService.sessionHistory

    init {
        refreshDatabaseInfo()
        checkPermission()
    }

    fun setNumTracks(count: Int) {
        _numTracks.value = count
        prefs.edit().putInt("num_tracks", count).apply()
    }

    fun setSearchStrategy(strategy: SearchStrategy) {
        _searchStrategy.value = strategy
        prefs.edit().putString("search_strategy", strategy.name).apply()
    }

    fun setAnchorExpandPrimary(model: EmbeddingModel) {
        _anchorExpandPrimary.value = model
        prefs.edit().putString("anchor_expand_primary", model.name).apply()
    }

    fun setAnchorExpandExpansion(n: Int) {
        _anchorExpandExpansion.value = n
        prefs.edit().putInt("anchor_expand_expansion", n).apply()
    }

    fun setDrift(enabled: Boolean) {
        _drift.value = enabled
        prefs.edit().putBoolean("drift", enabled).apply()
    }

    fun startRadio() {
        val strategy = _searchStrategy.value
        val aeConfig = if (strategy == SearchStrategy.ANCHOR_EXPAND) {
            AnchorExpandConfig(_anchorExpandPrimary.value, _anchorExpandExpansion.value)
        } else null
        RadioService.startRadio(
            getApplication(), _numTracks.value, strategy, aeConfig, _drift.value
        )
    }

    fun resetRadioState() {
        RadioService.resetState()
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
        viewModelScope.launch {
            val dbFile = File(getApplication<Application>().filesDir, "embeddings.db")
            if (dbFile.exists()) {
                try {
                    val db = EmbeddingDatabase.open(dbFile)
                    val availableModels = db.getAvailableModels()
                    val info = DatabaseInfo(
                        trackCount = db.getTrackCount(),
                        version = db.getMetadata("version"),
                        sizeKb = dbFile.length() / 1024,
                        availableModels = availableModels
                    )
                    db.close()
                    _databaseInfo.value = info
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
    val version: String?,
    val sizeKb: Long,
    val availableModels: Set<EmbeddingModel> = emptySet()
)
