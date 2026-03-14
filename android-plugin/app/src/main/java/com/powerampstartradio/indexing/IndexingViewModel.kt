package com.powerampstartradio.indexing

import android.app.Application
import android.content.Context
import android.net.Uri
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.powerampstartradio.data.EmbeddedTrack
import com.powerampstartradio.data.EmbeddingDatabase
import com.powerampstartradio.indexing.GraphUpdater
import com.powerampstartradio.poweramp.PowerampHelper
import kotlinx.coroutines.Deferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import org.json.JSONArray
import java.io.BufferedInputStream
import java.io.File
import java.io.FileInputStream
import java.io.BufferedOutputStream
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

/**
 * ViewModel for the IndexingActivity track selection and indexing UI.
 *
 * Manages the list of unindexed tracks, user selection, dismissed tracks
 * ("never-index"), ignored tracks ("previously ignored"), and delegates
 * indexing to IndexingService.
 */
class IndexingViewModel(application: Application) : AndroidViewModel(application) {
    sealed class ExportState {
        data object Idle : ExportState()
        data class Exporting(val message: String) : ExportState()
        data class Complete(val filename: String) : ExportState()
        data class Error(val message: String) : ExportState()
    }

    companion object {
        private const val TAG = "IndexingViewModel"
        private const val RECENT_DETECTION_CACHE_MS = 5L * 60L * 1000L

        /**
         * Cached detection result shared across ViewModel instances.
         * Avoids re-scanning 74K tracks when the user navigates away and back.
         * Invalidated when the DB file changes, Poweramp library changes,
         * or on explicit "Check Again".
         */
        private var cachedTracks: List<NewTrackDetector.UnindexedTrack>? = null
        private var cachedDbFingerprint: String = ""
        private var cachedPowerampTrackCount: Int = -1
        private var cachedTracksCheckedAtMs: Long = 0L
        private var cachedDatabaseOnlyTracks: List<EmbeddedTrack>? = null
        private var cachedDatabaseOnlyDbFingerprint: String = ""
        private var cachedDatabaseOnlyPowerampTrackCount: Int = -1
        private var cachedDatabaseOnlyCheckedAtMs: Long = 0L

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
            cachedDbFingerprint = ""
            cachedPowerampTrackCount = -1
            cachedTracksCheckedAtMs = 0L
            cachedDatabaseOnlyTracks = null
            cachedDatabaseOnlyDbFingerprint = ""
            cachedDatabaseOnlyPowerampTrackCount = -1
            cachedDatabaseOnlyCheckedAtMs = 0L
        }

        /** Store results from an external detection (e.g. MainViewModel's check). */
        fun cacheResults(
            tracks: List<NewTrackDetector.UnindexedTrack>,
            dbFingerprint: String,
            paCount: Int,
        ) {
            cachedTracks = tracks
            cachedDbFingerprint = dbFingerprint
            cachedPowerampTrackCount = paCount
            cachedTracksCheckedAtMs = System.currentTimeMillis()
        }
    }

    private val prefs = application.getSharedPreferences("indexing", Context.MODE_PRIVATE)

    private val _unindexedTracks = MutableStateFlow<List<NewTrackDetector.UnindexedTrack>>(emptyList())
    val unindexedTracks: StateFlow<List<NewTrackDetector.UnindexedTrack>> = _unindexedTracks.asStateFlow()

    private val _selectedIds = MutableStateFlow<Set<Long>>(emptySet())
    val selectedIds: StateFlow<Set<Long>> = _selectedIds.asStateFlow()

    /** "Never-index" list: explicit user action via overflow menu + auto-moved 0:00 tracks. */
    private val _dismissedIds = MutableStateFlow<Set<Long>>(loadDismissedIdsWithDbCheck(application))
    val dismissedIds: StateFlow<Set<Long>> = _dismissedIds.asStateFlow()

    /** "Previously ignored" list: auto-moved when starting indexing without search filter. */
    private val _ignoredIds = MutableStateFlow<Set<Long>>(loadIgnoredIdsWithDbCheck(application))
    val ignoredIds: StateFlow<Set<Long>> = _ignoredIds.asStateFlow()

    private val _isDetecting = MutableStateFlow(false)
    val isDetecting: StateFlow<Boolean> = _isDetecting.asStateFlow()

    private val _detectingStatus = MutableStateFlow("")
    val detectingStatus: StateFlow<String> = _detectingStatus.asStateFlow()

    private val _hasModels = MutableStateFlow(false)
    val hasModels: StateFlow<Boolean> = _hasModels.asStateFlow()

    private val _hasDatabase = MutableStateFlow(File(application.filesDir, "embeddings.db").exists())
    val hasDatabase: StateFlow<Boolean> = _hasDatabase.asStateFlow()

    private val _databaseOnlyTracks = MutableStateFlow<List<EmbeddedTrack>>(emptyList())
    val databaseOnlyTracks: StateFlow<List<EmbeddedTrack>> = _databaseOnlyTracks.asStateFlow()

    private val _isDetectingDatabaseOnly = MutableStateFlow(false)
    val isDetectingDatabaseOnly: StateFlow<Boolean> = _isDetectingDatabaseOnly.asStateFlow()

    private val _databaseOnlyStatus = MutableStateFlow("")
    val databaseOnlyStatus: StateFlow<String> = _databaseOnlyStatus.asStateFlow()

    private val _exportState = MutableStateFlow<ExportState>(ExportState.Idle)
    val exportState: StateFlow<ExportState> = _exportState.asStateFlow()

    val indexingState: StateFlow<IndexingService.IndexingState> = IndexingService.state

    init {
        refreshAppFiles()
    }

    fun detectUnindexed(forceRefresh: Boolean = false) {
        if (_isDetecting.value) return

        // Reset service state from Complete/Error so UI shows detecting spinner
        IndexingService.resetState()
        refreshAppFiles()

        val app = getApplication<Application>()
        val dbFile = File(app.filesDir, "embeddings.db")
        val dbFingerprint = getDbFingerprint()
        val powerampCount = getPowerampTrackCount(app)

        // DB fingerprint changes should invalidate detection cache, but they should not
        // wipe the user's ignored / never-index choices. Those choices are keyed to the
        // Poweramp library, not to a specific embeddings.db mtime.
        if (cachedTracks != null) {
            val currentFingerprint = if (dbFile.exists())
                "${dbFile.length()}_${dbFile.lastModified()}" else ""
            val savedFingerprint = prefs.getString("dismissed_db_fingerprint", "") ?: ""
            if (currentFingerprint != savedFingerprint) {
                prefs.edit()
                    .putString("dismissed_db_fingerprint", currentFingerprint)
                    .apply()
                invalidateCache()
                Log.i(TAG, "DB fingerprint changed; preserved hidden track choices and invalidated detection cache")
            }
        }

        // If MainViewModel is already running a detection, await its results
        val pending = pendingDetection
        if (pending != null && !forceRefresh) {
            _isDetecting.value = true
            _detectingStatus.value = detectionStatus.value ?: "Checking which tracks are unindexed..."
            viewModelScope.launch(Dispatchers.IO) {
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

        // Use cached result if neither the DB nor the Poweramp library has changed.
        // This must come after pendingDetection so a settings-side refresh can hand off
        // its in-flight job to Manage Tracks instead of showing the previous cached result.
        if (!forceRefresh && cachedTracks != null && dbFile.exists()
            && dbFingerprint == cachedDbFingerprint
            && powerampCount == cachedPowerampTrackCount
            && System.currentTimeMillis() - cachedTracksCheckedAtMs <= RECENT_DETECTION_CACHE_MS) {
            val tracks = cachedTracks!!
            Log.i(
                TAG,
                "Reusing recent unindexed cache (${System.currentTimeMillis() - cachedTracksCheckedAtMs}ms old)"
            )
            _unindexedTracks.value = tracks
            autoSelect(tracks)
            return
        }

        viewModelScope.launch(Dispatchers.IO) {
            _isDetecting.value = true
            var db: EmbeddingDatabase? = null
            try {
                if (!dbFile.exists()) {
                    _unindexedTracks.value = emptyList()
                    cachedTracks = emptyList()
                    cachedDbFingerprint = ""
                    cachedTracksCheckedAtMs = 0L
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
                cachedDbFingerprint = getDbFingerprint()
                cachedPowerampTrackCount = powerampCount
                cachedTracksCheckedAtMs = System.currentTimeMillis()
                // Sync dismissed fingerprint with current DB state
                updateDismissedFingerprint()
            } catch (e: Exception) {
                _unindexedTracks.value = emptyList()
            } finally {
                db?.close()
                _isDetecting.value = false
            }
        }
    }

    fun detectDatabaseOnlyTracks(forceRefresh: Boolean = false) {
        if (_isDetectingDatabaseOnly.value) return

        refreshAppFiles()
        val app = getApplication<Application>()
        val dbFile = File(app.filesDir, "embeddings.db")
        val dbFingerprint = getDbFingerprint()
        val powerampCount = getPowerampTrackCount(app)

        if (!forceRefresh && cachedDatabaseOnlyTracks != null && dbFile.exists()
            && dbFingerprint == cachedDatabaseOnlyDbFingerprint
            && powerampCount == cachedDatabaseOnlyPowerampTrackCount
            && System.currentTimeMillis() - cachedDatabaseOnlyCheckedAtMs <= RECENT_DETECTION_CACHE_MS) {
            _databaseOnlyTracks.value = cachedDatabaseOnlyTracks!!
            _databaseOnlyStatus.value = "Found ${cachedDatabaseOnlyTracks!!.size} clean-up candidates"
            Log.i(
                TAG,
                "Reusing recent clean-db cache (${System.currentTimeMillis() - cachedDatabaseOnlyCheckedAtMs}ms old, ${cachedDatabaseOnlyTracks!!.size} candidates)"
            )
            return
        }

        _isDetectingDatabaseOnly.value = true
        _databaseOnlyStatus.value = "Checking database..."
        Log.i(TAG, "Starting clean-db scan (forceRefresh=$forceRefresh)")
        viewModelScope.launch(Dispatchers.IO) {
            var db: EmbeddingDatabase? = null
            try {
                if (!dbFile.exists()) {
                    _databaseOnlyTracks.value = emptyList()
                    cachedDatabaseOnlyTracks = emptyList()
                    cachedDatabaseOnlyDbFingerprint = ""
                    cachedDatabaseOnlyCheckedAtMs = 0L
                    _databaseOnlyStatus.value = "No database loaded."
                    Log.i(TAG, "Clean-db scan skipped: no embeddings.db")
                    return@launch
                }

                db = EmbeddingDatabase.open(dbFile)
                val detector = NewTrackDetector(db)
                val tracks = detector.findDatabaseOnlyTracks(app) { status ->
                    _databaseOnlyStatus.value = status
                }.sortedWith(
                    compareBy<EmbeddedTrack>({ it.artist.orEmpty().lowercase() },
                        { it.album.orEmpty().lowercase() },
                        { it.title.orEmpty().lowercase() })
                )

                _databaseOnlyTracks.value = tracks
                cachedDatabaseOnlyTracks = tracks
                cachedDatabaseOnlyDbFingerprint = getDbFingerprint()
                cachedDatabaseOnlyPowerampTrackCount = powerampCount
                cachedDatabaseOnlyCheckedAtMs = System.currentTimeMillis()
                updateDismissedFingerprint()
                Log.i(TAG, "Clean-db scan complete: ${tracks.size} candidates")
            } catch (e: Exception) {
                _databaseOnlyTracks.value = emptyList()
                _databaseOnlyStatus.value = "Clean-up scan failed."
                Log.e(TAG, "Clean-db scan failed", e)
            } finally {
                db?.close()
                _isDetectingDatabaseOnly.value = false
            }
        }
    }

    fun deleteDatabaseOnlyTracks(ids: Set<Long>) {
        if (ids.isEmpty()) return

        viewModelScope.launch(Dispatchers.IO) {
            val app = getApplication<Application>()
            val dbFile = File(app.filesDir, "embeddings.db")
            if (!dbFile.exists()) return@launch

            var db: EmbeddingDatabase? = null
            var deletedTracks = 0
            try {
                _isDetectingDatabaseOnly.value = true
                _databaseOnlyStatus.value = "Deleting ${ids.size} tracks from the database..."
                Log.i(TAG, "Deleting ${ids.size} clean-db track ids")
                db = EmbeddingDatabase.openReadWrite(dbFile)
                deletedTracks = db.deleteTracks(ids)
                Log.i(TAG, "Clean-db delete removed $deletedTracks track rows")
                if (deletedTracks != ids.size) {
                    Log.w(TAG, "Clean-db requested ${ids.size} deletions but removed $deletedTracks rows")
                }
                if (deletedTracks > 0) {
                    db.deleteBinaryData("knn_graph")
                    File(app.filesDir, "clamp3.emb").delete()
                    File(app.filesDir, "graph.bin").delete()
                    invalidateCache()
                    Log.i(TAG, "Rebuilding derived search files after clean-db delete")
                    _databaseOnlyStatus.value = "Rebuilding search files..."
                    GraphUpdater(db, app.filesDir).rebuildIndices { status ->
                        _databaseOnlyStatus.value = status
                    }
                    Log.i(TAG, "Finished rebuilding search files after clean-db delete")
                } else {
                    _databaseOnlyStatus.value = "No database rows were deleted."
                }
            } catch (e: Exception) {
                Log.e(TAG, "Clean-db delete failed", e)
                _databaseOnlyStatus.value = "Delete failed. Please try again."
            } finally {
                db?.close()
                refreshAppFiles()
                updateDismissedFingerprint()
                _isDetectingDatabaseOnly.value = false
            }

            detectDatabaseOnlyTracks(forceRefresh = true)
        }
    }

    fun exportInstance(uri: Uri) {
        viewModelScope.launch(Dispatchers.IO) {
            val app = getApplication<Application>()
            val filesDir = app.filesDir
            val dbFile = File(filesDir, "embeddings.db")
            if (!dbFile.exists()) {
                _exportState.value = ExportState.Error("No embeddings.db found to export")
                return@launch
            }

            val tempGraph = File(app.cacheDir, "export-graph.bin")
            tempGraph.delete()

            try {
                _exportState.value = ExportState.Exporting("Preparing export...")

                val filesToZip = mutableListOf<Pair<String, File>>()
                filesToZip += "embeddings.db" to dbFile

                listOf(
                    "clamp3.emb",
                    "graph.bin",
                    "mert.tflite",
                    "mert_fp16.tflite",
                    "clamp3_audio.tflite",
                    "clamp3_audio_fp16.tflite",
                    "clamp3_text.tflite",
                    "clamp3_text_fp16.tflite",
                    "xlm_roberta_vocab.json",
                ).forEach { name ->
                    val file = File(filesDir, name)
                    if (file.exists()) filesToZip += name to file
                }

                if (filesToZip.none { it.first == "graph.bin" }) {
                    val db = EmbeddingDatabase.open(dbFile)
                    try {
                        if (db.hasBinaryData("knn_graph") && db.extractBinaryToFile("knn_graph", tempGraph)) {
                            filesToZip += "graph.bin" to tempGraph
                        }
                    } finally {
                        db.close()
                    }
                }

                app.contentResolver.openOutputStream(uri)?.use { output ->
                    ZipOutputStream(BufferedOutputStream(output)).use { zip ->
                        for ((index, entry) in filesToZip.withIndex()) {
                            _exportState.value = ExportState.Exporting(
                                "Packing ${index + 1}/${filesToZip.size}: ${entry.first}"
                            )
                            zip.putNextEntry(ZipEntry(entry.first))
                            BufferedInputStream(FileInputStream(entry.second)).use { input ->
                                input.copyTo(zip)
                            }
                            zip.closeEntry()
                        }
                    }
                } ?: throw IllegalArgumentException("Cannot open export destination")

                _exportState.value = ExportState.Complete(File(uri.lastPathSegment ?: "export.zip").name)
            } catch (e: Exception) {
                _exportState.value = ExportState.Error(e.message ?: "Export failed")
            } finally {
                tempGraph.delete()
            }
        }
    }

    fun clearExportState() {
        _exportState.value = ExportState.Idle
    }

    /** Auto-select visible tracks, excluding 0-duration, dismissed, and ignored. */
    private fun autoSelect(tracks: List<NewTrackDetector.UnindexedTrack>) {
        val dismissed = _dismissedIds.value
        val ignored = _ignoredIds.value

        // Auto-move 0:00 tracks to never-index
        val zeroDuration = tracks.filter { it.durationMs == 0 && it.powerampFileId !in dismissed }
            .map { it.powerampFileId }.toSet()
        if (zeroDuration.isNotEmpty()) {
            val newDismissed = dismissed + zeroDuration
            _dismissedIds.value = newDismissed
            saveDismissedIds(newDismissed)
        }

        val allExcluded = (dismissed + zeroDuration) + ignored
        _selectedIds.value = tracks
            .filter { it.powerampFileId !in allExcluded && it.durationMs > 0 }
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
        val ignored = _ignoredIds.value
        _selectedIds.value = _unindexedTracks.value
            .filter { it.powerampFileId !in dismissed && it.powerampFileId !in ignored }
            .map { it.powerampFileId }
            .toSet()
    }

    fun deselectAll() {
        _selectedIds.value = emptySet()
    }

    fun selectIds(ids: Set<Long>) {
        _selectedIds.value = _selectedIds.value + ids
    }

    fun deselectIds(ids: Set<Long>) {
        _selectedIds.value = _selectedIds.value - ids
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
        // Re-select all (including previously dismissed), excluding 0-duration and ignored
        val ignored = _ignoredIds.value
        _selectedIds.value = _unindexedTracks.value
            .filter { it.durationMs > 0 && it.powerampFileId !in ignored }
            .map { it.powerampFileId }
            .toSet()
    }

    /** Get dismissed track details from the full unindexedTracks list. */
    fun getDismissedTracks(): List<NewTrackDetector.UnindexedTrack> {
        return _unindexedTracks.value.filter { it.powerampFileId in _dismissedIds.value }
    }

    /** Restore specific tracks from dismissed back to visible (and auto-select them). */
    fun restoreFromDismissed(ids: Set<Long>) {
        val newDismissed = _dismissedIds.value - ids
        _dismissedIds.value = newDismissed
        saveDismissedIds(newDismissed)
        _selectedIds.value = _selectedIds.value + ids
    }

    /** Get ignored track details from the full unindexedTracks list. */
    fun getIgnoredTracks(): List<NewTrackDetector.UnindexedTrack> {
        return _unindexedTracks.value.filter { it.powerampFileId in _ignoredIds.value }
    }

    /** Restore specific tracks from ignored back to visible (and auto-select them). */
    fun restoreFromIgnored(ids: Set<Long>) {
        val newIgnored = _ignoredIds.value - ids
        _ignoredIds.value = newIgnored
        saveIgnoredIds(newIgnored)
        _selectedIds.value = _selectedIds.value + ids
    }

    /** Move tracks from ignored to never-index (permanent exclusion). */
    fun moveIgnoredToNeverIndex(ids: Set<Long>) {
        val newIgnored = _ignoredIds.value - ids
        _ignoredIds.value = newIgnored
        saveIgnoredIds(newIgnored)
        val newDismissed = _dismissedIds.value + ids
        _dismissedIds.value = newDismissed
        saveDismissedIds(newDismissed)
    }

    /** Clear all ignored tracks and re-run autoSelect. */
    fun clearIgnored() {
        _ignoredIds.value = emptySet()
        saveIgnoredIds(emptySet())
        autoSelect(_unindexedTracks.value)
    }

    /**
     * @param autoDismissUnselected When true, unselected visible tracks are auto-ignored.
     *   Should be true when the user sees the full list (no search filter) — their choice
     *   to not select something is moved to "previously ignored". Should be false when a
     *   search filter is active — the user is only focused on the search results.
     */
    /**
     * @param onlyIds When non-null, restrict indexing to these track IDs (intersection
     *   with selected). Used when a search filter is active — only index visible tracks.
     */
    fun startIndexing(buildGraph: Boolean = false, autoDismissUnselected: Boolean = true,
                      onlyIds: Set<Long>? = null) {
        refreshAppFiles()
        if (!_hasModels.value) return
        val selected = if (onlyIds != null) _selectedIds.value.intersect(onlyIds) else _selectedIds.value
        if (selected.isEmpty()) return
        val dismissed = _dismissedIds.value
        val ignored = _ignoredIds.value
        val tracks = _unindexedTracks.value.filter {
            it.powerampFileId in selected && it.powerampFileId !in dismissed
        }
        if (tracks.isEmpty()) return

        if (autoDismissUnselected) {
            val unselectedVisible = _unindexedTracks.value.filter {
                it.powerampFileId !in selected && it.powerampFileId !in dismissed && it.powerampFileId !in ignored
            }.map { it.powerampFileId }.toSet()
            if (unselectedVisible.isNotEmpty()) {
                val newIgnored = ignored + unselectedVisible
                _ignoredIds.value = newIgnored
                saveIgnoredIds(newIgnored)
            }
        }

        // Invalidate cache so the next detectUnindexed() re-scans from the DB.
        // Can't rely on dbFile.lastModified() — SQLite WAL may not flush to the main file.
        invalidateCache()
        IndexingService.startIndexing(getApplication(), tracks, buildGraph = buildGraph)
    }

    fun cancelIndexing() {
        IndexingService.cancelIndexing()
    }

    fun refreshAppFiles() {
        val filesDir = getApplication<Application>().filesDir
        _hasDatabase.value = File(filesDir, "embeddings.db").exists()
        _hasModels.value = File(filesDir, "mert.tflite").exists() &&
            File(filesDir, "clamp3_audio.tflite").exists()
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

    private fun loadDismissedIdsWithDbCheck(app: Application): Set<Long> {
        val dbFile = File(app.filesDir, "embeddings.db")
        val currentFingerprint = if (dbFile.exists())
            "${dbFile.length()}_${dbFile.lastModified()}" else ""
        val savedFingerprint = prefs.getString("dismissed_db_fingerprint", "") ?: ""

        if (currentFingerprint != savedFingerprint) {
            prefs.edit()
                .putString("dismissed_db_fingerprint", currentFingerprint)
                .apply()
            invalidateCache()
            Log.i(TAG, "DB fingerprint changed during load; preserved hidden track choices")
        }
        return loadDismissedIds()
    }

    private fun loadIgnoredIdsWithDbCheck(app: Application): Set<Long> {
        val dbFile = File(app.filesDir, "embeddings.db")
        val currentFingerprint = if (dbFile.exists())
            "${dbFile.length()}_${dbFile.lastModified()}" else ""
        val savedFingerprint = prefs.getString("dismissed_db_fingerprint", "") ?: ""

        if (currentFingerprint != savedFingerprint) {
            prefs.edit().putString("dismissed_db_fingerprint", currentFingerprint).apply()
        }
        return loadIgnoredIds()
    }

    private fun loadDismissedIds(): Set<Long> {
        val json = prefs.getString("dismissed_track_ids", null) ?: return emptySet()
        return parseIdJson(json)
    }

    private fun loadIgnoredIds(): Set<Long> {
        val json = prefs.getString("ignored_track_ids", null) ?: return emptySet()
        return parseIdJson(json)
    }

    private fun parseIdJson(json: String): Set<Long> {
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

    private fun saveIgnoredIds(ids: Set<Long>) {
        val arr = JSONArray()
        ids.forEach { arr.put(it) }
        prefs.edit()
            .putString("ignored_track_ids", arr.toString())
            .putString("dismissed_db_fingerprint", getDbFingerprint())
            .apply()
    }

    private fun getDbFingerprint(): String {
        val dbFile = File(getApplication<Application>().filesDir, "embeddings.db")
        return if (dbFile.exists()) "${dbFile.length()}_${dbFile.lastModified()}" else ""
    }

    /** Update saved fingerprint so dismissed IDs survive DB changes from our own indexing. */
    private fun updateDismissedFingerprint() {
        val fp = getDbFingerprint()
        if (fp.isNotEmpty()) {
            prefs.edit().putString("dismissed_db_fingerprint", fp).apply()
        }
    }

}
