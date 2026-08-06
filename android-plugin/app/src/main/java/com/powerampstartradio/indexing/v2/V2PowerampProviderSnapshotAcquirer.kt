package com.powerampstartradio.indexing.v2

import android.content.Context
import android.database.Cursor
import android.net.Uri
import android.util.Log
import com.powerampstartradio.poweramp.PowerampHelper
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.util.concurrent.CancellationException

fun interface V2PowerampProviderQuery {
    fun query(context: Context, filesUri: Uri, projection: Array<String>): Cursor?
}

object V2AndroidPowerampProviderQuery : V2PowerampProviderQuery {
    override fun query(context: Context, filesUri: Uri, projection: Array<String>): Cursor? =
        context.contentResolver.query(filesUri, projection, null, null, null)
}

fun interface V2PowerampProviderSelectionQuery {
    fun query(
        context: Context,
        filesUri: Uri,
        projection: Array<String>,
        selection: String,
        selectionArgs: Array<String>,
    ): Cursor?
}

object V2AndroidPowerampProviderSelectionQuery : V2PowerampProviderSelectionQuery {
    override fun query(
        context: Context,
        filesUri: Uri,
        projection: Array<String>,
        selection: String,
        selectionArgs: Array<String>,
    ): Cursor? = context.contentResolver.query(
        filesUri,
        projection,
        selection,
        selectionArgs,
        null,
    )
}

enum class V2PowerampProviderAcquisitionScope {
    SELECTED_PATH_GROUPS,
    COMPLETE_LIBRARY,
}

data class V2PowerampProviderAcquisitionResult(
    val snapshot: V2ProviderPathGroupSnapshot,
    val scope: V2PowerampProviderAcquisitionScope,
    val requestedRowCount: Int,
    val selectedProbeMs: Long,
    val totalMs: Long,
)

internal object V2SelectedProviderSnapshotPolicy {
    fun requiresCompleteLibrarySnapshot(
        pathGroupSnapshot: V2ProviderPathGroupSnapshot,
        selectedIds: Set<Long>,
    ): Boolean {
        val selectedGroups = pathGroupSnapshot.groups.filter { group ->
            group.rows.any { it.powerampFileId in selectedIds }
        }
        val foundIds = selectedGroups.flatMap { it.rows }
            .mapTo(hashSetOf()) { it.powerampFileId }
            .intersect(selectedIds)
        if (foundIds != selectedIds) return true
        return selectedGroups.any { group ->
            group.rows.any { row ->
                row.offsetMs > 0L || row.cueSourceImageFolderId != null
            }
        }
    }
}

/** Strict V2-only Poweramp provider reader. V1's permissive helper remains unchanged. */
class V2PowerampProviderSnapshotAcquirer(
    context: Context,
    private val assembler: V2PowerampProviderSnapshotAssembler =
        V2PowerampProviderSnapshotAssembler(),
    private val providerQuery: V2PowerampProviderQuery = V2AndroidPowerampProviderQuery,
    private val providerSelectionQuery: V2PowerampProviderSelectionQuery =
        V2AndroidPowerampProviderSelectionQuery,
) {
    companion object {
        private const val TAG = "V2PowerampSnapshot"
        private const val PROGRESS_ROW_INTERVAL = 512
        private const val MAX_BOUNDED_PATH_GROUP_FILENAMES = 500
        val REQUIRED_PROJECTION = listOf(
            "folder_files._id",
            "artist",
            "album",
            "title_tag",
            "folder_files.duration",
            "path",
            "folder_files.name",
            "folder_files.offset_ms",
            "cue_folder_id",
            "folder_files.created_at",
        )
    }

    private val appContext = context.applicationContext
    private val filesUri = PowerampHelper.ROOT_URI.buildUpon()
        .appendEncodedPath("files")
        .build()

    suspend fun acquire(): V2ProviderPathGroupSnapshot = withContext(Dispatchers.IO) {
        acquireBlocking()
    }

    /**
     * Reads the selected IDs, then every provider row with their physical filenames. The second
     * bounded cursor proves complete lexical path groups, including a zero-offset first CUE row.
     * Imported CUE work still falls back to the complete library required by its base-row proof.
     */
    fun acquireSelectedWithCueFallbackBlocking(
        fileIds: Collection<Long>,
        requireCompletePathGroups: Boolean = false,
        onRowProgress: ((completedRows: Int, totalRows: Int) -> Unit)? = null,
    ): V2PowerampProviderAcquisitionResult {
        val requestedIds = fileIds.toList()
        require(requestedIds.isNotEmpty()) { "Selected Poweramp row query is empty" }
        require(requestedIds.all { it > 0L }) { "Poweramp file IDs must be positive" }
        require(requestedIds.distinct().size == requestedIds.size) {
            "Selected Poweramp row query contains duplicate IDs"
        }
        val startedNs = System.nanoTime()
        if (requireCompletePathGroups ||
            requestedIds.size > MAX_BOUNDED_PATH_GROUP_FILENAMES
        ) {
            Log.i(
                TAG,
                "Complete provider snapshot required: " + if (requireCompletePathGroups) {
                    "logical CUE authorization"
                } else {
                    "${requestedIds.size} selected rows exceed the bounded query limit"
                },
            )
            val snapshot = acquireBlocking(onRowProgress)
            return acquisitionResult(
                snapshot = snapshot,
                requestedRowCount = requestedIds.size,
                selectedProbeMs = 0L,
                startedNs = startedNs,
            )
        }

        val probeStartedNs = System.nanoTime()
        onRowProgress?.invoke(0, requestedIds.size)
        val entries = try {
            PowerampHelper.requireFileEntriesByIds(appContext, requestedIds)
        } catch (cancelled: CancellationException) {
            throw cancelled
        } catch (error: Throwable) {
            val causes = generateSequence(error) { it.cause }.toList()
            causes.filterIsInstance<CancellationException>().firstOrNull()?.let { throw it }
            if (causes.any { it is SecurityException }) {
                fail(
                    code = V2PowerampProviderSnapshotFailureCode.POWERAMP_PERMISSION_DENIED,
                    message = "Poweramp denied selected-row access",
                    cause = error,
                )
            }
            val selectedProbeMs = elapsedMs(probeStartedNs)
            Log.w(TAG, "Selected Poweramp row read failed; using complete snapshot", error)
            val snapshot = acquireBlocking(onRowProgress)
            return acquisitionResult(
                snapshot = snapshot,
                requestedRowCount = requestedIds.size,
                selectedProbeMs = selectedProbeMs,
                startedNs = startedNs,
            )
        }
        onRowProgress?.invoke(entries.size, requestedIds.size)
        val selectedProbeMs = elapsedMs(probeStartedNs)
        val selectedEntries = requestedIds.mapNotNull(entries::get)
        if (selectedEntries.size != requestedIds.size) {
            val snapshot = acquireBlocking(onRowProgress)
            return acquisitionResult(
                snapshot = snapshot,
                requestedRowCount = requestedIds.size,
                selectedProbeMs = selectedProbeMs,
                startedNs = startedNs,
            )
        }
        val selectedFilenames = selectedEntries.map { it.providerFileName }
        val filenames = selectedFilenames.filterNotNull()
            .filter(String::isNotBlank)
            .distinct()
        if (filenames.size != selectedFilenames.distinct().size ||
            filenames.size > MAX_BOUNDED_PATH_GROUP_FILENAMES
        ) {
            Log.i(TAG, "Complete provider snapshot required: selected filename evidence is unsafe")
            val snapshot = acquireBlocking(onRowProgress)
            return acquisitionResult(
                snapshot = snapshot,
                requestedRowCount = requestedIds.size,
                selectedProbeMs = selectedProbeMs,
                startedNs = startedNs,
            )
        }
        val pathGroupSnapshot = try {
            acquirePathGroupsByFilenamesBlocking(filenames)
        } catch (cancelled: CancellationException) {
            throw cancelled
        } catch (error: Throwable) {
            Log.w(TAG, "Bounded Poweramp path-group read failed; using complete snapshot", error)
            val snapshot = acquireBlocking(onRowProgress)
            return acquisitionResult(
                snapshot = snapshot,
                requestedRowCount = requestedIds.size,
                selectedProbeMs = selectedProbeMs,
                startedNs = startedNs,
            )
        }
        if (V2SelectedProviderSnapshotPolicy.requiresCompleteLibrarySnapshot(
                pathGroupSnapshot = pathGroupSnapshot,
                selectedIds = requestedIds.toSet(),
            )
        ) {
            Log.i(TAG, "Complete provider snapshot required: selected path group is CUE-shaped")
            val snapshot = acquireBlocking(onRowProgress)
            return acquisitionResult(
                snapshot = snapshot,
                requestedRowCount = requestedIds.size,
                selectedProbeMs = selectedProbeMs,
                startedNs = startedNs,
            )
        }

        val result = V2PowerampProviderAcquisitionResult(
            snapshot = pathGroupSnapshot,
            scope = V2PowerampProviderAcquisitionScope.SELECTED_PATH_GROUPS,
            requestedRowCount = requestedIds.size,
            selectedProbeMs = selectedProbeMs,
            totalMs = elapsedMs(startedNs),
        )
        logAcquisition(result)
        return result
    }

    /** Intended for an existing IO executor or instrumentation diagnostics. */
    fun acquireBlocking(
        onRowProgress: ((completedRows: Int, totalRows: Int) -> Unit)? = null,
    ): V2ProviderPathGroupSnapshot {
        val queryStartedNs = System.nanoTime()
        val cursor = try {
            providerQuery.query(
                context = appContext,
                filesUri = filesUri,
                projection = REQUIRED_PROJECTION.toTypedArray(),
            )
        } catch (error: SecurityException) {
            fail(
                V2PowerampProviderSnapshotFailureCode.POWERAMP_PERMISSION_DENIED,
                "Poweramp denied provider access",
                error,
            )
        } catch (cancelled: CancellationException) {
            throw cancelled
        } catch (error: Throwable) {
            fail(
                V2PowerampProviderSnapshotFailureCode.PROVIDER_QUERY_FAILED,
                "Poweramp files query failed: ${error.message}",
                error,
            )
        } ?: fail(
            V2PowerampProviderSnapshotFailureCode.PROVIDER_RETURNED_NULL_CURSOR,
            "Poweramp files query returned no cursor",
        )

        val rows = mutableListOf<V2RawPowerampProviderRow>()
        var returnedColumns = emptyList<String>()
        var exhaustedNormally = false
        try {
            cursor.use { openCursor ->
                returnedColumns = openCursor.columnNames.toList()
                val columns = resolveRequiredColumns(openCursor)
                val totalRows = openCursor.count.coerceAtLeast(0)
                if (totalRows > 0) onRowProgress?.invoke(0, totalRows)
                while (true) {
                    if (!openCursor.moveToNext()) {
                        exhaustedNormally = true
                        break
                    }
                    rows += readRow(openCursor, columns)
                    val completedRows = rows.size
                    if (completedRows == totalRows ||
                        completedRows % PROGRESS_ROW_INTERVAL == 0
                    ) {
                        onRowProgress?.invoke(completedRows, totalRows)
                    }
                }
            }
        } catch (error: V2PowerampProviderSnapshotException) {
            throw error
        } catch (cancelled: CancellationException) {
            throw cancelled
        } catch (error: SecurityException) {
            fail(
                V2PowerampProviderSnapshotFailureCode.POWERAMP_PERMISSION_DENIED,
                "Poweramp revoked provider access while reading the snapshot",
                error,
            )
        } catch (error: Throwable) {
            fail(
                V2PowerampProviderSnapshotFailureCode.PROVIDER_CURSOR_FAILED,
                "Poweramp cursor failed before complete exhaustion: ${error.message}",
                error,
            )
        }

        val queryAndCursorReadMs = elapsedMs(queryStartedNs)
        val assemblyStartedNs = System.nanoTime()
        val snapshot = assembler.assembleAfterSuccessfulExhaustion(
            rows = rows,
            acquisitionEvidence = V2ProviderSnapshotAcquisitionEvidence(
                queryUri = filesUri.toString(),
                requestedColumns = REQUIRED_PROJECTION,
                returnedColumns = returnedColumns,
                rowCount = rows.size,
                cursorExhaustedNormally = exhaustedNormally,
                queryAndCursorReadMs = queryAndCursorReadMs,
            ),
        )
        val assemblyMs = elapsedMs(assemblyStartedNs)
        return snapshot.copy(
            acquisitionEvidence = snapshot.acquisitionEvidence?.copy(
                snapshotAssemblyMs = assemblyMs,
            ),
        )
    }

    private fun acquirePathGroupsByFilenamesBlocking(
        filenames: List<String>,
    ): V2ProviderPathGroupSnapshot {
        require(filenames.isNotEmpty()) { "Selected Poweramp filenames are empty" }
        require(filenames.size <= MAX_BOUNDED_PATH_GROUP_FILENAMES) {
            "Selected Poweramp filename query is too large"
        }
        val queryStartedNs = System.nanoTime()
        val cursor = try {
            providerSelectionQuery.query(
                context = appContext,
                filesUri = filesUri,
                projection = REQUIRED_PROJECTION.toTypedArray(),
                selection = "folder_files.name IN (${filenames.joinToString(",") { "?" }})",
                selectionArgs = filenames.toTypedArray(),
            )
        } catch (error: SecurityException) {
            fail(
                V2PowerampProviderSnapshotFailureCode.POWERAMP_PERMISSION_DENIED,
                "Poweramp denied selected path-group access",
                error,
            )
        } catch (cancelled: CancellationException) {
            throw cancelled
        } catch (error: Throwable) {
            fail(
                V2PowerampProviderSnapshotFailureCode.PROVIDER_QUERY_FAILED,
                "Poweramp path-group query failed: ${error.message}",
                error,
            )
        } ?: fail(
            V2PowerampProviderSnapshotFailureCode.PROVIDER_RETURNED_NULL_CURSOR,
            "Poweramp path-group query returned no cursor",
        )

        val rows = mutableListOf<V2RawPowerampProviderRow>()
        var returnedColumns = emptyList<String>()
        var exhaustedNormally = false
        try {
            cursor.use { openCursor ->
                returnedColumns = openCursor.columnNames.toList()
                val columns = resolveRequiredColumns(openCursor)
                while (true) {
                    if (!openCursor.moveToNext()) {
                        exhaustedNormally = true
                        break
                    }
                    rows += readRow(openCursor, columns)
                }
            }
        } catch (error: V2PowerampProviderSnapshotException) {
            throw error
        } catch (cancelled: CancellationException) {
            throw cancelled
        } catch (error: SecurityException) {
            fail(
                V2PowerampProviderSnapshotFailureCode.POWERAMP_PERMISSION_DENIED,
                "Poweramp revoked provider access while reading selected path groups",
                error,
            )
        } catch (error: Throwable) {
            fail(
                V2PowerampProviderSnapshotFailureCode.PROVIDER_CURSOR_FAILED,
                "Poweramp path-group cursor failed before exhaustion: ${error.message}",
                error,
            )
        }

        val queryAndCursorReadMs = elapsedMs(queryStartedNs)
        val assemblyStartedNs = System.nanoTime()
        val acquisition = V2ProviderSnapshotAcquisitionEvidence(
            queryUri = filesUri.toString(),
            requestedColumns = REQUIRED_PROJECTION,
            returnedColumns = returnedColumns,
            rowCount = rows.size,
            cursorExhaustedNormally = exhaustedNormally,
            queryAndCursorReadMs = queryAndCursorReadMs,
        )
        val snapshot = assembler.assembleAfterSuccessfulExhaustion(rows, acquisition)
        return snapshot.copy(
            acquisitionEvidence = snapshot.acquisitionEvidence?.copy(
                snapshotAssemblyMs = elapsedMs(assemblyStartedNs),
            ),
        )
    }

    private fun acquisitionResult(
        snapshot: V2ProviderPathGroupSnapshot,
        requestedRowCount: Int,
        selectedProbeMs: Long,
        startedNs: Long,
    ): V2PowerampProviderAcquisitionResult = V2PowerampProviderAcquisitionResult(
        snapshot = snapshot,
        scope = V2PowerampProviderAcquisitionScope.COMPLETE_LIBRARY,
        requestedRowCount = requestedRowCount,
        selectedProbeMs = selectedProbeMs,
        totalMs = elapsedMs(startedNs),
    ).also(::logAcquisition)

    private fun logAcquisition(result: V2PowerampProviderAcquisitionResult) {
        val evidence = result.snapshot.acquisitionEvidence
        Log.i(
            TAG,
            "Provider evidence scope=${result.scope} requested=${result.requestedRowCount} " +
                "read=${evidence?.rowCount ?: 0} selectedProbeMs=${result.selectedProbeMs} " +
                "snapshotQueryMs=${evidence?.queryAndCursorReadMs ?: -1L} " +
                "assemblyMs=${evidence?.snapshotAssemblyMs ?: -1L} totalMs=${result.totalMs}",
        )
    }

    private data class RequiredColumns(
        val id: Int,
        val artist: Int,
        val album: Int,
        val title: Int,
        val duration: Int,
        val folderPath: Int,
        val fileName: Int,
        val offset: Int,
        val cueFolderId: Int,
        val createdAt: Int,
    )

    private fun resolveRequiredColumns(cursor: Cursor): RequiredColumns = RequiredColumns(
        id = requiredColumn(cursor, "Poweramp file ID", "_id", "folder_files._id"),
        artist = requiredColumn(cursor, "artist", "artist"),
        album = requiredColumn(cursor, "album", "album"),
        title = requiredColumn(cursor, "title", "title_tag"),
        duration = requiredColumn(cursor, "duration", "duration", "folder_files.duration"),
        folderPath = requiredColumn(cursor, "folder path", "path"),
        fileName = requiredColumn(cursor, "file name", "name", "folder_files.name"),
        offset = requiredColumn(cursor, "logical offset", "offset_ms", "folder_files.offset_ms"),
        cueFolderId = requiredColumn(cursor, "CUE source evidence", "cue_folder_id"),
        createdAt = requiredColumn(
            cursor,
            "Poweramp first-seen time",
            "created_at",
            "folder_files.created_at",
        ),
    )

    private fun requiredColumn(cursor: Cursor, label: String, vararg candidates: String): Int {
        val returnedColumns = cursor.columnNames
        for (candidate in candidates) {
            val exactIndex = returnedColumns.indexOf(candidate)
            if (exactIndex >= 0) return exactIndex
        }
        for (candidate in candidates) {
            val index = cursor.getColumnIndex(candidate)
            if (index >= 0) return index
        }
        fail(
            V2PowerampProviderSnapshotFailureCode.PROVIDER_SCHEMA_MISMATCH,
            "Poweramp cursor is missing required $label column; returned " +
                cursor.columnNames.joinToString(),
        )
    }

    private fun readRow(cursor: Cursor, columns: RequiredColumns): V2RawPowerampProviderRow {
        if (cursor.isNull(columns.id)) {
            fail(
                V2PowerampProviderSnapshotFailureCode.INVALID_PROVIDER_ROW,
                "Poweramp row ${cursor.position} has null ID",
            )
        }
        val id = cursor.getLong(columns.id)
        val createdAtEpochSecond = if (cursor.isNull(columns.createdAt)) {
            0L
        } else {
            cursor.getLong(columns.createdAt).coerceAtLeast(0L)
        }
        val folderPath = cursor.getString(columns.folderPath)
        val fileName = cursor.getString(columns.fileName)
        if (folderPath.isNullOrBlank() || fileName.isNullOrBlank()) {
            fail(
                V2PowerampProviderSnapshotFailureCode.INVALID_PROVIDER_ROW,
                "Poweramp row $id has no complete physical path",
                powerampFileId = id,
            )
        }
        val offsetWasNull = cursor.isNull(columns.offset)
        return V2RawPowerampProviderRow(
            powerampFileId = id,
            artist = cursor.getString(columns.artist),
            album = cursor.getString(columns.album),
            title = cursor.getString(columns.title),
            durationMs = if (cursor.isNull(columns.duration)) 0L else {
                cursor.getLong(columns.duration)
            },
            folderPath = folderPath,
            fileName = fileName,
            offsetMs = if (offsetWasNull) 0L else cursor.getLong(columns.offset),
            offsetWasNull = offsetWasNull,
            cueSourceImageFolderId = if (cursor.isNull(columns.cueFolderId)) {
                null
            } else {
                cursor.getLong(columns.cueFolderId)
            },
            createdAtEpochSecond = createdAtEpochSecond,
        )
    }

    private fun fail(
        code: V2PowerampProviderSnapshotFailureCode,
        message: String,
        cause: Throwable? = null,
        powerampFileId: Long? = null,
    ): Nothing = throw V2PowerampProviderSnapshotException(
        code = code,
        powerampFileId = powerampFileId,
        message = message,
        cause = cause,
    )

    private fun elapsedMs(startedNs: Long): Long =
        (System.nanoTime() - startedNs) / 1_000_000L

}
