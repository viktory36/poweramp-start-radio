package com.powerampstartradio.ui

import android.app.Application
import android.content.Context
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.powerampstartradio.data.EmbeddingDatabase
import com.powerampstartradio.data.EmbeddingModel
import com.powerampstartradio.poweramp.PowerampHelper
import com.powerampstartradio.services.RadioService
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

    // Database info
    private val _databaseInfo = MutableStateFlow<DatabaseInfo?>(null)
    val databaseInfo: StateFlow<DatabaseInfo?> = _databaseInfo.asStateFlow()

    // Poweramp permission state
    private val _hasPermission = MutableStateFlow(false)
    val hasPermission: StateFlow<Boolean> = _hasPermission.asStateFlow()

    // Radio state from service
    val radioState: StateFlow<RadioUiState> = RadioService.uiState

    init {
        refreshDatabaseInfo()
        checkPermission()
    }

    fun setNumTracks(count: Int) {
        _numTracks.value = count
        prefs.edit().putInt("num_tracks", count).apply()
    }

    fun startRadio() {
        RadioService.startRadio(getApplication(), _numTracks.value)
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
