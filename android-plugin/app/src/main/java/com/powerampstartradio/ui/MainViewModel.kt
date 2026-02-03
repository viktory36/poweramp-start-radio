package com.powerampstartradio.ui

import android.app.Application
import android.content.Context
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.powerampstartradio.data.EmbeddingDatabase
import com.powerampstartradio.data.EmbeddingModel
import com.powerampstartradio.poweramp.PowerampHelper
import com.powerampstartradio.services.RadioService
import com.powerampstartradio.similarity.FeedForwardConfig
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
            SearchStrategy.valueOf(prefs.getString("search_strategy", SearchStrategy.FEED_FORWARD.name)!!)
        } catch (e: IllegalArgumentException) {
            SearchStrategy.FEED_FORWARD
        }
    )
    val searchStrategy: StateFlow<SearchStrategy> = _searchStrategy.asStateFlow()

    private val _feedForwardPrimary = MutableStateFlow(
        try {
            EmbeddingModel.valueOf(prefs.getString("feed_forward_primary", EmbeddingModel.MULAN.name)!!)
        } catch (e: IllegalArgumentException) {
            EmbeddingModel.MULAN
        }
    )
    val feedForwardPrimary: StateFlow<EmbeddingModel> = _feedForwardPrimary.asStateFlow()

    private val _feedForwardExpansion = MutableStateFlow(
        prefs.getInt("feed_forward_expansion", 3)
    )
    val feedForwardExpansion: StateFlow<Int> = _feedForwardExpansion.asStateFlow()

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

    fun setFeedForwardPrimary(model: EmbeddingModel) {
        _feedForwardPrimary.value = model
        prefs.edit().putString("feed_forward_primary", model.name).apply()
    }

    fun setFeedForwardExpansion(n: Int) {
        _feedForwardExpansion.value = n
        prefs.edit().putInt("feed_forward_expansion", n).apply()
    }

    fun startRadio() {
        val strategy = _searchStrategy.value
        val ffConfig = if (strategy == SearchStrategy.FEED_FORWARD) {
            FeedForwardConfig(_feedForwardPrimary.value, _feedForwardExpansion.value)
        } else null
        RadioService.startRadio(getApplication(), _numTracks.value, strategy, ffConfig)
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
