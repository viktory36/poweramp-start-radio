package com.powerampstartradio.indexing

import android.app.Application
import android.content.Context
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.powerampstartradio.data.EmbeddingDatabase
import com.powerampstartradio.poweramp.PowerampHelper
import kotlinx.coroutines.Deferred
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

    companion object {
        /**
         * Cached detection result shared across ViewModel instances.
         * Avoids re-scanning 74K tracks when the user navigates away and back.
         * Invalidated when the DB file changes, Poweramp library changes,
         * or on explicit "Check Again".
         */
        private var cachedTracks: List<NewTrackDetector.UnindexedTrack>? = null
        private var cachedDbLastModified: Long = 0L
        private var cachedPowerampTrackCount: Int = -1

        /**
         * A pending detection started by MainViewModel. IndexingViewModel will
         * await this instead of starting a duplicate scan.
         */
        @Volatile var pendingDetection: Deferred<List<NewTrackDetector.UnindexedTrack>>? = null

        /** Live progress from any ongoing detection (observed by both ViewModels). */
        val detectionStatus = MutableStateFlow<String?>(null)

        /** Clear the cached detection result (e.g. after DB import). */
        fun invalidateCache() {
            cachedTracks = null
            cachedDbLastModified = 0L
            cachedPowerampTrackCount = -1
        }

        /** Store results from an external detection (e.g. MainViewModel's check). */
        fun cacheResults(tracks: List<NewTrackDetector.UnindexedTrack>, dbMod: Long, paCount: Int) {
            cachedTracks = tracks
            cachedDbLastModified = dbMod
            cachedPowerampTrackCount = paCount
        }
    }

    private val prefs = application.getSharedPreferences("indexing", Context.MODE_PRIVATE)

    private val _unindexedTracks = MutableStateFlow<List<NewTrackDetector.UnindexedTrack>>(emptyList())
    val unindexedTracks: StateFlow<List<NewTrackDetector.UnindexedTrack>> = _unindexedTracks.asStateFlow()

    private val _selectedIds = MutableStateFlow<Set<Long>>(emptySet())
    val selectedIds: StateFlow<Set<Long>> = _selectedIds.asStateFlow()

    private val _dismissedIds = MutableStateFlow<Set<Long>>(loadDismissedIdsWithDbCheck(application))
    val dismissedIds: StateFlow<Set<Long>> = _dismissedIds.asStateFlow()

    private val _isDetecting = MutableStateFlow(false)
    val isDetecting: StateFlow<Boolean> = _isDetecting.asStateFlow()

    private val _detectingStatus = MutableStateFlow("")
    val detectingStatus: StateFlow<String> = _detectingStatus.asStateFlow()

    val indexingState: StateFlow<IndexingService.IndexingState> = IndexingService.state

    init {
        detectUnindexed()
    }

    fun detectUnindexed(forceRefresh: Boolean = false) {
        if (_isDetecting.value) return

        // Reset service state from Complete/Error so UI shows detecting spinner
        IndexingService.resetState()

        val app = getApplication<Application>()
        val dbFile = File(app.filesDir, "embeddings.db")
        val powerampCount = getPowerampTrackCount(app)

        // Check if DB was replaced since we last saved dismissed IDs
        val currentFingerprint = if (dbFile.exists())
            "${dbFile.length()}_${dbFile.lastModified()}" else ""
        val savedFingerprint = prefs.getString("dismissed_db_fingerprint", "") ?: ""
        if (currentFingerprint != savedFingerprint) {
            // DB changed — clear stale dismissed IDs
            _dismissedIds.value = emptySet()
            prefs.edit()
                .remove("dismissed_track_ids")
                .putString("dismissed_db_fingerprint", currentFingerprint)
                .apply()
            invalidateCache()
        }

        // Use cached result if neither the DB nor the Poweramp library has changed
        if (!forceRefresh && cachedTracks != null && dbFile.exists()
            && dbFile.lastModified() == cachedDbLastModified
            && powerampCount == cachedPowerampTrackCount) {
            val tracks = cachedTracks!!
            _unindexedTracks.value = tracks
            autoSelect(tracks)
            return
        }

        // If MainViewModel is already running a detection, await its results
        val pending = pendingDetection
        if (pending != null && !forceRefresh) {
            viewModelScope.launch(Dispatchers.IO) {
                _isDetecting.value = true
                try {
                    // Mirror shared progress into our own status flow
                    val statusJob = launch {
                        detectionStatus.collect { status ->
                            if (status != null) _detectingStatus.value = status
                        }
                    }
                    val tracks = pending.await()
                    statusJob.cancel()
                    _unindexedTracks.value = tracks
                    autoSelect(tracks)
                } catch (_: Exception) {
                    _unindexedTracks.value = emptyList()
                } finally {
                    _isDetecting.value = false
                }
            }
            return
        }

        viewModelScope.launch(Dispatchers.IO) {
            _isDetecting.value = true
            var db: EmbeddingDatabase? = null
            try {
                if (!dbFile.exists()) {
                    _unindexedTracks.value = emptyList()
                    cachedTracks = emptyList()
                    cachedDbLastModified = 0L
                    return@launch
                }
                db = EmbeddingDatabase.open(dbFile)
                val detector = NewTrackDetector(db)
                val tracks = detector.findUnindexedTracks(getApplication()) { status ->
                    _detectingStatus.value = status
                }
                // Sort by duration descending (longest first)
                val sorted = tracks.sortedByDescending { it.durationMs }
                _unindexedTracks.value = sorted
                autoSelect(sorted)
                // Cache for reuse
                cachedTracks = sorted
                cachedDbLastModified = dbFile.lastModified()
                cachedPowerampTrackCount = powerampCount
            } catch (e: Exception) {
                _unindexedTracks.value = emptyList()
            } finally {
                db?.close()
                _isDetecting.value = false
            }
        }
    }

    /** Auto-select visible tracks, excluding 0-duration and dismissed. */
    private fun autoSelect(tracks: List<NewTrackDetector.UnindexedTrack>) {
        val dismissed = _dismissedIds.value
        _selectedIds.value = tracks
            .filter { it.powerampFileId !in dismissed && it.durationMs > 0 }
            .map { it.powerampFileId }
            .toSet()
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
        // Re-select all (including previously dismissed), excluding 0-duration
        _selectedIds.value = _unindexedTracks.value
            .filter { it.durationMs > 0 }
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

        // Auto-dismiss unchecked visible tracks so they don't count as unindexed
        val unchecked = _unindexedTracks.value
            .filter { it.powerampFileId !in selected && it.powerampFileId !in dismissed }
            .map { it.powerampFileId }
            .toSet()
        if (unchecked.isNotEmpty()) {
            val newDismissed = dismissed + unchecked
            _dismissedIds.value = newDismissed
            saveDismissedIds(newDismissed)
        }

        // Cache auto-invalidates when DB modification time changes after indexing.
        // Don't invalidate eagerly — if the user cancels, the cache is still valid.
        IndexingService.startIndexing(getApplication(), tracks, refusion = refusion)
    }

    fun cancelIndexing() {
        IndexingService.cancelIndexing()
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

    /**
     * Load dismissed IDs, but clear them if the database has changed
     * (e.g. user replaced embeddings.db with a fresh copy).
     */
    private fun loadDismissedIdsWithDbCheck(app: Application): Set<Long> {
        val dbFile = File(app.filesDir, "embeddings.db")
        val currentFingerprint = if (dbFile.exists())
            "${dbFile.length()}_${dbFile.lastModified()}" else ""
        val savedFingerprint = prefs.getString("dismissed_db_fingerprint", "") ?: ""

        if (currentFingerprint != savedFingerprint) {
            // DB has changed — clear stale dismissed IDs and detection cache
            prefs.edit()
                .remove("dismissed_track_ids")
                .putString("dismissed_db_fingerprint", currentFingerprint)
                .apply()
            invalidateCache()
            return emptySet()
        }
        return loadDismissedIds()
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
        prefs.edit()
            .putString("dismissed_track_ids", arr.toString())
            .putString("dismissed_db_fingerprint", getDbFingerprint())
            .apply()
    }

    private fun getDbFingerprint(): String {
        val dbFile = File(getApplication<Application>().filesDir, "embeddings.db")
        return if (dbFile.exists()) "${dbFile.length()}_${dbFile.lastModified()}" else ""
    }
}
