package com.powerampstartradio.indexing

import android.app.Application
import android.content.Context
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.powerampstartradio.data.EmbeddingDatabase
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import org.json.JSONArray
import java.io.File

/**
 * ViewModel for the IndexingActivity track selection and indexing UI.
 *
 * Manages the list of unindexed tracks, user selection, dismissed tracks,
 * and delegates indexing to IndexingService.
 */
class IndexingViewModel(application: Application) : AndroidViewModel(application) {

    private val prefs = application.getSharedPreferences("indexing", Context.MODE_PRIVATE)

    private val _unindexedTracks = MutableStateFlow<List<NewTrackDetector.UnindexedTrack>>(emptyList())
    val unindexedTracks: StateFlow<List<NewTrackDetector.UnindexedTrack>> = _unindexedTracks.asStateFlow()

    private val _selectedIds = MutableStateFlow<Set<Long>>(emptySet())
    val selectedIds: StateFlow<Set<Long>> = _selectedIds.asStateFlow()

    private val _dismissedIds = MutableStateFlow<Set<Long>>(loadDismissedIds())
    val dismissedIds: StateFlow<Set<Long>> = _dismissedIds.asStateFlow()

    private val _isDetecting = MutableStateFlow(false)
    val isDetecting: StateFlow<Boolean> = _isDetecting.asStateFlow()

    val indexingState: StateFlow<IndexingService.IndexingState> = IndexingService.state

    /** Visible tracks = unindexed minus dismissed. */
    val visibleTracks: List<NewTrackDetector.UnindexedTrack>
        get() {
            val dismissed = _dismissedIds.value
            return _unindexedTracks.value.filter { it.powerampFileId !in dismissed }
        }

    init {
        detectUnindexed()
    }

    fun detectUnindexed() {
        if (_isDetecting.value) return
        viewModelScope.launch(Dispatchers.IO) {
            _isDetecting.value = true
            var db: EmbeddingDatabase? = null
            try {
                val dbFile = File(getApplication<Application>().filesDir, "embeddings.db")
                if (!dbFile.exists()) {
                    _unindexedTracks.value = emptyList()
                    return@launch
                }
                db = EmbeddingDatabase.open(dbFile)
                val detector = NewTrackDetector(db)
                val tracks = detector.findUnindexedTracks(getApplication())
                _unindexedTracks.value = tracks
                // Auto-select all visible tracks
                val dismissed = _dismissedIds.value
                _selectedIds.value = tracks
                    .filter { it.powerampFileId !in dismissed }
                    .map { it.powerampFileId }
                    .toSet()
            } catch (e: Exception) {
                _unindexedTracks.value = emptyList()
            } finally {
                db?.close()
                _isDetecting.value = false
            }
        }
    }

    fun toggleSelection(id: Long) {
        _selectedIds.value = if (id in _selectedIds.value) {
            _selectedIds.value - id
        } else {
            _selectedIds.value + id
        }
    }

    fun selectAll() {
        val dismissed = _dismissedIds.value
        _selectedIds.value = _unindexedTracks.value
            .filter { it.powerampFileId !in dismissed }
            .map { it.powerampFileId }
            .toSet()
    }

    fun deselectAll() {
        _selectedIds.value = emptySet()
    }

    fun dismissSelected() {
        val toDismiss = _selectedIds.value
        if (toDismiss.isEmpty()) return
        val newDismissed = _dismissedIds.value + toDismiss
        _dismissedIds.value = newDismissed
        _selectedIds.value = emptySet()
        saveDismissedIds(newDismissed)
    }

    fun clearDismissed() {
        _dismissedIds.value = emptySet()
        saveDismissedIds(emptySet())
        // Re-select all (including previously dismissed)
        _selectedIds.value = _unindexedTracks.value
            .map { it.powerampFileId }
            .toSet()
    }

    fun startIndexing(refusion: Boolean = false) {
        val selected = _selectedIds.value
        if (selected.isEmpty()) return
        val dismissed = _dismissedIds.value
        val tracks = _unindexedTracks.value.filter {
            it.powerampFileId in selected && it.powerampFileId !in dismissed
        }
        if (tracks.isEmpty()) return
        IndexingService.startIndexing(getApplication(), tracks, refusion = refusion)
    }

    fun cancelIndexing() {
        IndexingService.cancelIndexing()
    }

    private fun loadDismissedIds(): Set<Long> {
        val json = prefs.getString("dismissed_track_ids", null) ?: return emptySet()
        return try {
            val arr = JSONArray(json)
            val set = mutableSetOf<Long>()
            for (i in 0 until arr.length()) {
                set.add(arr.getLong(i))
            }
            set
        } catch (_: Exception) {
            emptySet()
        }
    }

    private fun saveDismissedIds(ids: Set<Long>) {
        val arr = JSONArray()
        ids.forEach { arr.put(it) }
        prefs.edit().putString("dismissed_track_ids", arr.toString()).apply()
    }
}
