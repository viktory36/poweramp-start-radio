package com.powerampstartradio.poweramp

import android.content.ComponentName
import android.content.ContentProviderOperation
import android.content.ContentValues
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.database.Cursor
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.util.Log
import java.io.File
import java.net.URI

enum class QueueMutationKind {
    REPLACE,
    APPEND,
}

/**
 * Observable result of one Poweramp queue mutation.
 *
 * [verifiedRequestIndices] is occurrence-based, not a set of file IDs: duplicate file IDs in
 * a ranked list remain independently accountable. An index is present only when a subsequent
 * provider read exposed that requested occurrence in queue order.
 */
data class QueueMutationResult(
    val kind: QueueMutationKind,
    val requestedFileIds: List<Long>,
    val verifiedRequestIndices: Set<Int>,
    val verifiedFileIds: List<Long>,
    val providerReportedInsertCount: Int,
    val beforeCount: Int?,
    val afterCount: Int?,
    val preservedAnchorFileId: Long?,
    /** Exact queue row kept alive across a replacement, not a folder_files ID. */
    val preservedAnchorQueueId: Long? = null,
    /** Exact read-back queue row for each verified request occurrence. */
    val verifiedQueueEntryIdsByRequestIndex: Map<Int, Long> = emptyMap(),
    val unexpectedObservedCount: Int,
    val fallbackUsed: Boolean,
    val mutationError: String? = null,
    val verificationError: String? = null,
    val rollbackAttempted: Boolean = false,
    /** Ordered file/sort content restored; a replacement may regenerate deleted non-anchor IDs. */
    val rollbackVerified: Boolean = false,
    val rollbackError: String? = null,
) {
    val requestedCount: Int get() = requestedFileIds.size
    val verifiedCount: Int get() = verifiedRequestIndices.size
    val failedCount: Int get() = requestedCount - verifiedCount
    val verificationSucceeded: Boolean get() = verificationError == null
    val fullyVerified: Boolean
        get() = verificationSucceeded &&
            mutationError == null &&
            !rollbackAttempted &&
            unexpectedObservedCount == 0 &&
            verifiedCount == requestedCount &&
            verifiedQueueEntryIdsByRequestIndex.keys == verifiedRequestIndices &&
            verifiedQueueEntryIdsByRequestIndex.values.toSet().size == verifiedCount

    fun isRequestVerified(index: Int): Boolean = index in verifiedRequestIndices

    fun verifiedQueueEntryId(index: Int): Long? =
        verifiedQueueEntryIdsByRequestIndex[index]
}

internal data class QueueEntry(
    val queueId: Long,
    val fileId: Long,
    val sort: Int,
)

internal data class PlannedQueueInsertion(
    val requestIndex: Int,
    val fileId: Long,
    val sort: Int,
)

internal data class PartialQueueInsertionPlan(
    val safeToContinue: Boolean,
    val appliedRequestIndices: Set<Int>,
    val missing: List<PlannedQueueInsertion>,
    val error: String? = null,
)

/** Reconciles a possibly partially committed applyBatch before any individual fallback. */
internal object PartialQueueInsertionReconciler {
    fun reconcile(
        baseline: List<QueueEntry>,
        observed: List<QueueEntry>,
        intended: List<PlannedQueueInsertion>,
    ): PartialQueueInsertionPlan {
        val baselineById = baseline.associateBy(QueueEntry::queueId)
        if (baselineById.size != baseline.size ||
            baseline.any { expected -> observed.none { it == expected } }
        ) {
            return rejected("Queue baseline changed during insertion")
        }
        val intendedByIdentity = intended.associateBy { it.fileId to it.sort }
        if (intendedByIdentity.size != intended.size ||
            intended.map { it.requestIndex }.toSet().size != intended.size
        ) {
            return rejected("Intended queue occurrences are not uniquely addressable")
        }
        val applied = linkedSetOf<Int>()
        for (entry in observed) {
            if (entry.queueId in baselineById) continue
            val insertion = intendedByIdentity[entry.fileId to entry.sort]
                ?: return rejected("Unexpected queue occurrence appeared during insertion")
            if (!applied.add(insertion.requestIndex)) {
                return rejected("A partially committed queue occurrence is duplicated")
            }
        }
        return PartialQueueInsertionPlan(
            safeToContinue = true,
            appliedRequestIndices = applied,
            missing = intended.filterNot { it.requestIndex in applied },
        )
    }

    private fun rejected(reason: String) = PartialQueueInsertionPlan(
        safeToContinue = false,
        appliedRequestIndices = emptySet(),
        missing = emptyList(),
        error = reason,
    )
}

/** Queue IDs may be regenerated, but order, file IDs, sorts, and a live anchor must be restored. */
internal object QueueRollbackVerifier {
    fun error(
        original: List<QueueEntry>,
        restored: List<QueueEntry>,
        preservedAnchorQueueId: Long?,
    ): String? {
        val originalContent = original.map { it.fileId to it.sort }
        val restoredContent = restored.map { it.fileId to it.sort }
        if (originalContent != restoredContent) return "Restored queue content differs from the snapshot"
        if (preservedAnchorQueueId != null) {
            val originalAnchor = original.singleOrNull { it.queueId == preservedAnchorQueueId }
                ?: return "Original queue snapshot omitted its live anchor"
            if (restored.none { it == originalAnchor }) {
                return "Restored queue did not preserve the live anchor occurrence"
            }
        }
        return null
    }
}

internal object QueueMutationReconciler {
    data class ObservedOccurrence(
        val queueId: Long,
        val fileId: Long,
    )

    data class Match(
        val requestIndices: Set<Int>,
        val fileIds: List<Long>,
        val unexpectedObservedCount: Int,
        /** Request index to the exact matching index in the observed occurrence list. */
        val observedIndicesByRequestIndex: Map<Int, Int> = emptyMap(),
    )

    /**
     * Ordered occurrence reconciliation with bounded memory.
     *
     * Small mutations use exact LCS. Large direct queues use a conservative linear matcher:
     * it can under-report a pathological crossing match, but never reports an occurrence whose
     * order was not observed. Exact successful delivery is handled by the fast equality path.
     */
    fun reconcile(requested: List<Long>, observed: List<Long>): Match {
        if (requested == observed) {
            return Match(
                requested.indices.toSet(),
                requested.toList(),
                0,
                requested.indices.associateWith { it },
            )
        }
        val cells = (requested.size.toLong() + 1L) * (observed.size.toLong() + 1L)
        if (cells > MAX_EXACT_LCS_CELLS) {
            val left = greedyRequestedIntoObserved(requested, observed)
            val right = greedyObservedIntoRequested(requested, observed)
            return if (left.requestIndices.size >= right.requestIndices.size) left else right
        }

        val requestedSize = requested.size
        val observedSize = observed.size
        val lengths = Array(requestedSize + 1) { IntArray(observedSize + 1) }

        for (requestIndex in requestedSize - 1 downTo 0) {
            for (observedIndex in observedSize - 1 downTo 0) {
                lengths[requestIndex][observedIndex] = if (
                    requested[requestIndex] == observed[observedIndex]
                ) {
                    1 + lengths[requestIndex + 1][observedIndex + 1]
                } else {
                    maxOf(
                        lengths[requestIndex + 1][observedIndex],
                        lengths[requestIndex][observedIndex + 1],
                    )
                }
            }
        }

        val matchedIndices = linkedSetOf<Int>()
        val matchedFileIds = mutableListOf<Long>()
        val matchedObservedIndices = linkedMapOf<Int, Int>()
        var requestIndex = 0
        var observedIndex = 0
        while (requestIndex < requestedSize && observedIndex < observedSize) {
            if (
                requested[requestIndex] == observed[observedIndex] &&
                lengths[requestIndex][observedIndex] ==
                    1 + lengths[requestIndex + 1][observedIndex + 1]
            ) {
                matchedIndices += requestIndex
                matchedFileIds += requested[requestIndex]
                matchedObservedIndices[requestIndex] = observedIndex
                requestIndex++
                observedIndex++
            } else if (
                lengths[requestIndex + 1][observedIndex] >=
                lengths[requestIndex][observedIndex + 1]
            ) {
                requestIndex++
            } else {
                observedIndex++
            }
        }

        return Match(
            requestIndices = matchedIndices,
            fileIds = matchedFileIds,
            unexpectedObservedCount = observedSize - matchedFileIds.size,
            observedIndicesByRequestIndex = matchedObservedIndices,
        )
    }

    /** Verify the same physical queue rows observed by an earlier mutation readback. */
    fun reconcileExactOccurrences(
        requestedFileIds: List<Long>,
        expectedQueueEntryIdsByRequestIndex: Map<Int, Long>,
        observedOccurrences: List<ObservedOccurrence>,
        countUnmatchedObserved: Boolean,
    ): Match {
        val observedIndexByQueueId = observedOccurrences.withIndex()
            .associate { (index, entry) -> entry.queueId to index }
        val matchedRequestIndices = linkedSetOf<Int>()
        val matchedFileIds = mutableListOf<Long>()
        val observedIndices = linkedMapOf<Int, Int>()
        var previousObservedIndex = -1
        for (requestIndex in requestedFileIds.indices) {
            val queueId = expectedQueueEntryIdsByRequestIndex[requestIndex] ?: continue
            val observedIndex = observedIndexByQueueId[queueId] ?: continue
            if (observedOccurrences[observedIndex].fileId != requestedFileIds[requestIndex]) continue
            if (previousObservedIndex >= 0 && observedIndex != previousObservedIndex + 1) continue
            matchedRequestIndices += requestIndex
            matchedFileIds += requestedFileIds[requestIndex]
            observedIndices[requestIndex] = observedIndex
            previousObservedIndex = observedIndex
        }
        return Match(
            requestIndices = matchedRequestIndices,
            fileIds = matchedFileIds,
            unexpectedObservedCount = if (countUnmatchedObserved) {
                observedOccurrences.size - matchedRequestIndices.size
            } else {
                0
            },
            observedIndicesByRequestIndex = observedIndices,
        )
    }

    private fun greedyRequestedIntoObserved(requested: List<Long>, observed: List<Long>): Match {
        val matched = linkedSetOf<Int>()
        val matchedObservedIndices = linkedMapOf<Int, Int>()
        var observedIndex = 0
        for (requestIndex in requested.indices) {
            while (observedIndex < observed.size &&
                observed[observedIndex] != requested[requestIndex]
            ) {
                observedIndex++
            }
            if (observedIndex < observed.size) {
                matched += requestIndex
                matchedObservedIndices[requestIndex] = observedIndex
                observedIndex++
            }
        }
        return Match(
            requestIndices = matched,
            fileIds = matched.map(requested::get),
            unexpectedObservedCount = observed.size - matched.size,
            observedIndicesByRequestIndex = matchedObservedIndices,
        )
    }

    private fun greedyObservedIntoRequested(requested: List<Long>, observed: List<Long>): Match {
        val matched = linkedSetOf<Int>()
        val matchedObservedIndices = linkedMapOf<Int, Int>()
        var requestIndex = 0
        for ((observedIndex, fileId) in observed.withIndex()) {
            while (requestIndex < requested.size && requested[requestIndex] != fileId) {
                requestIndex++
            }
            if (requestIndex < requested.size) {
                matched += requestIndex
                matchedObservedIndices[requestIndex] = observedIndex
                requestIndex++
            }
        }
        return Match(
            requestIndices = matched,
            fileIds = matched.map(requested::get),
            unexpectedObservedCount = observed.size - matched.size,
            observedIndicesByRequestIndex = matchedObservedIndices,
        )
    }

    private const val MAX_EXACT_LCS_CELLS = 2_000_000L
}

/**
 * Helper for interacting with Poweramp via its public API.
 *
 * References:
 * - powerampapi/poweramp_api_lib/src/main/java/com/maxmpz/poweramp/player/PowerampAPI.java
 * - powerampapi/poweramp_api_example/src/main/java/com/maxmpz/poweramp/apiexample/MainActivity.java
 */
object PowerampHelper {
    private const val TAG = "PowerampHelper"

    // Poweramp package and component names
    const val POWERAMP_PACKAGE = "com.maxmpz.audioplayer"
    private const val API_ACTIVITY = "com.maxmpz.audioplayer.apiactivity.ApiActivity"

    // Poweramp content provider authority
    private const val AUTHORITY = "com.maxmpz.audioplayer.data"
    val ROOT_URI: Uri = Uri.parse("content://$AUTHORITY")

    // Actions
    const val ACTION_TRACK_CHANGED = "com.maxmpz.audioplayer.TRACK_CHANGED"
    const val ACTION_TRACK_CHANGED_EXPLICIT = "com.maxmpz.audioplayer.TRACK_CHANGED_EXPLICIT"
    const val ACTION_STATUS_CHANGED = "com.maxmpz.audioplayer.STATUS_CHANGED"
    const val ACTION_STATUS_CHANGED_EXPLICIT = "com.maxmpz.audioplayer.STATUS_CHANGED_EXPLICIT"
    const val ACTION_RELOAD_DATA = "com.maxmpz.audioplayer.ACTION_RELOAD_DATA"
    const val ACTION_ASK_FOR_DATA_PERMISSION = "com.maxmpz.audioplayer.ACTION_ASK_FOR_DATA_PERMISSION"

    // Extras
    const val EXTRA_TRACK = "track"
    const val EXTRA_PACKAGE = "pak"
    const val EXTRA_TABLE = "table"
    const val EXTRA_STATE = "state"
    const val EXTRA_TIMESTAMP = "ts"

    const val STATE_STOPPED = 0
    const val STATE_PLAYING = 1
    const val STATE_PAUSED = 2

    // Track extras
    const val TRACK_ID = "id"
    const val TRACK_REAL_ID = "realId"
    const val TRACK_CAT_URI = "catUri"
    const val TRACK_POS_IN_LIST = "posInList"
    const val TRACK_TITLE = "title"
    const val TRACK_ARTIST = "artist"
    const val TRACK_ALBUM = "album"
    const val TRACK_DURATION = "dur"
    const val TRACK_DURATION_MS = "durMs"
    const val TRACK_PATH = "path"

    // Table names
    const val TABLE_QUEUE = "queue"
    const val TABLE_FILES = "folder_files"

    // Queue columns
    const val QUEUE_FOLDER_FILE_ID = "folder_file_id"
    const val QUEUE_SORT = "sort"

    private data class QueueSnapshot(
        val entries: List<QueueEntry> = emptyList(),
        val error: String? = null,
    )

    private data class InsertAttempt(
        val reportedCount: Int,
        val fallbackUsed: Boolean,
        val error: String? = null,
    )

    private data class RollbackAttempt(
        val snapshot: QueueSnapshot,
        val verified: Boolean,
        val error: String?,
    )

    /**
     * Send a command intent to Poweramp via its API Activity.
     */
    fun sendIntent(context: Context, intent: Intent) {
        intent.setComponent(ComponentName(POWERAMP_PACKAGE, API_ACTIVITY))
        intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        try {
            context.startActivity(intent)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to send intent to Poweramp", e)
        }
    }

    /**
     * Request permission from Poweramp to access its content provider.
     * On Android 8+ this must be called before querying Poweramp's database.
     * Poweramp will show a dialog to the user to grant permission.
     */
    fun requestDataPermission(context: Context) {
        Log.d(TAG, "Requesting data permission from Poweramp")
        val intent = Intent(ACTION_ASK_FOR_DATA_PERMISSION).apply {
            setPackage(POWERAMP_PACKAGE)
            putExtra(EXTRA_PACKAGE, context.packageName)
            if (context !is android.app.Activity) {
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            }
        }
        // Use implicit intent (don't set explicit component)
        try {
            context.startActivity(intent)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to request data permission", e)
        }
    }

    /**
     * Check if we can access Poweramp's content provider.
     */
    fun canAccessData(context: Context): Boolean {
        return try {
            // Use URI parameter for limit instead of sortOrder
            val filesUri = ROOT_URI.buildUpon()
                .appendEncodedPath("files")
                .appendQueryParameter("lim", "1")
                .build()
            val cursor = context.contentResolver.query(
                filesUri,
                arrayOf("folder_files._id"),
                null,
                null,
                null
            )
            cursor?.close()
            cursor != null
        } catch (e: Exception) {
            Log.d(TAG, "Cannot access Poweramp data: ${e.message}")
            false
        }
    }

    /**
     * Get the current track info from a track changed intent.
     */
    fun getCurrentTrackFromIntent(intent: Intent): PowerampTrack? = runCatching {
        val trackBundle = intent.getBundleExtra(EXTRA_TRACK)
            ?: intent.extras?.takeIf { it.containsKey(TRACK_REAL_ID) }
            ?: return@runCatching null
        getCurrentTrackFromBundle(trackBundle)
    }.onFailure { error ->
        Log.w(TAG, "Ignoring malformed Poweramp track event", error)
    }.getOrNull()

    private fun getCurrentTrackFromBundle(trackBundle: Bundle): PowerampTrack? {
        val realId = trackBundle.getLong(TRACK_REAL_ID, -1L)
        if (realId <= 0L) return null
        val durationMs = when {
            trackBundle.containsKey(TRACK_DURATION_MS) ->
                trackBundle.getInt(TRACK_DURATION_MS, 0).coerceAtLeast(0)
            else -> trackBundle.getInt(TRACK_DURATION, 0).coerceAtLeast(0)
                .toLong()
                .times(1_000L)
                .coerceAtMost(Int.MAX_VALUE.toLong())
                .toInt()
        }
        return PowerampTrack(
            realId = realId,
            title = trackBundle.getString(TRACK_TITLE) ?: "",
            artist = trackBundle.getString(TRACK_ARTIST),
            album = trackBundle.getString(TRACK_ALBUM),
            durationMs = durationMs,
            path = trackBundle.getString(TRACK_PATH),
            trackId = trackBundle.getLong(TRACK_ID, -1L),
            categoryUri = readTrackCategoryUri(trackBundle),
            positionInList = trackBundle
                .takeIf { it.containsKey(TRACK_POS_IN_LIST) }
                ?.getInt(TRACK_POS_IN_LIST),
        )
    }

    @Suppress("DEPRECATION")
    private fun readTrackCategoryUri(trackBundle: android.os.Bundle): String? =
        (trackBundle.getParcelable(TRACK_CAT_URI) as? Uri)?.toString()

    fun getStickyCurrentTrack(context: Context): PowerampTrack? {
        val filter = IntentFilter(ACTION_TRACK_CHANGED)
        val sticky = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            context.registerReceiver(null, filter, Context.RECEIVER_EXPORTED)
        } else {
            @Suppress("DEPRECATION")
            context.registerReceiver(null, filter)
        }
        return sticky?.let(::getCurrentTrackFromIntent)
    }

    /** Return Poweramp's sticky playback state while its service is alive. */
    fun getStickyPlaybackState(context: Context): Int? {
        val filter = IntentFilter(ACTION_STATUS_CHANGED)
        val sticky = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            context.registerReceiver(null, filter, Context.RECEIVER_EXPORTED)
        } else {
            @Suppress("DEPRECATION")
            context.registerReceiver(null, filter)
        }
        return sticky?.takeIf { it.hasExtra(EXTRA_STATE) }?.getIntExtra(EXTRA_STATE, STATE_STOPPED)
    }

    /**
     * Query Poweramp's files table to find a file ID by metadata.
     */
    fun findFileIdByMetadata(
        context: Context,
        artist: String?,
        album: String?,
        title: String,
        durationMs: Int
    ): Long? {
        val filesUri = ROOT_URI.buildUpon().appendEncodedPath("files").build()

        // Build selection query
        val selection = StringBuilder()
        val selectionArgs = mutableListOf<String>()

        selection.append("title_tag LIKE ?")
        selectionArgs.add(title)

        if (!artist.isNullOrEmpty()) {
            selection.append(" AND (artist LIKE ? OR album_artist LIKE ?)")
            selectionArgs.add(artist)
            selectionArgs.add(artist)
        }

        if (!album.isNullOrEmpty()) {
            selection.append(" AND album LIKE ?")
            selectionArgs.add(album)
        }

        // Duration check with tolerance (within 5 seconds)
        val durationSec = durationMs / 1000
        selection.append(" AND duration BETWEEN ? AND ?")
        selectionArgs.add((durationSec - 5).toString())
        selectionArgs.add((durationSec + 5).toString())

        try {
            val cursor: Cursor? = context.contentResolver.query(
                filesUri,
                arrayOf("folder_files._id"),
                selection.toString(),
                selectionArgs.toTypedArray(),
                null
            )

            cursor?.use {
                if (it.moveToFirst()) {
                    return it.getLong(0)
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error querying Poweramp files", e)
        }

        return null
    }

    /**
     * Get all file IDs from Poweramp's library.
     * Returns a map of metadata key to file ID.
     */
    fun getAllFileIds(context: Context): Map<String, Long> {
        val filesUri = ROOT_URI.buildUpon().appendEncodedPath("files").build()
        val result = mutableMapOf<String, Long>()

        try {
            val cursor = context.contentResolver.query(
                filesUri,
                arrayOf("folder_files._id", "artist", "album", "title_tag", "folder_files.duration"),
                null,
                null,
                null
            )

            cursor?.use {
                val idIdx = it.getColumnIndex("_id")
                val artistIdx = it.getColumnIndex("artist")
                val albumIdx = it.getColumnIndex("album")
                val titleIdx = it.getColumnIndex("title_tag")
                val durationIdx = it.getColumnIndex("duration")

                while (it.moveToNext()) {
                    val id = it.getLong(idIdx)
                    val artist = (it.getString(artistIdx) ?: "").lowercase().trim()
                    val album = (it.getString(albumIdx) ?: "").lowercase().trim()
                    val title = (it.getString(titleIdx) ?: "").lowercase().trim()
                    val durationMs = it.getInt(durationIdx)

                    // Create metadata key matching desktop indexer format (rounds to 100ms)
                    val durationRounded = (durationMs / 100) * 100
                    val key = "$artist|$album|$title|$durationRounded"
                    result[key] = id
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error getting all file IDs", e)
        }

        return result
    }

    /**
     * Get all file entries from Poweramp's library with NFC-normalized individual fields.
     * Used by TrackMatcher for robust matching that doesn't depend on pipe-delimited keys.
     */
    fun getAllFileEntries(context: Context): List<PowerampFileEntry> =
        requireCompleteFileSnapshot(context).entries

    /** Read one exact provider-owned file identity without materializing the complete library. */
    fun requireFileEntryById(context: Context, fileId: Long): PowerampFileEntry =
        requireNotNull(requireFileEntriesByIds(context, listOf(fileId))[fileId]) {
            "Poweramp file $fileId is absent from the targeted provider result"
        }

    /** Read a small exact ID set in one provider query, failing on missing or duplicate rows. */
    fun requireFileEntriesByIds(
        context: Context,
        fileIds: Collection<Long>,
    ): Map<Long, PowerampFileEntry> {
        val requestedIds = fileIds.toList()
        require(requestedIds.all { it > 0L }) { "Poweramp file IDs must be positive" }
        require(requestedIds.distinct().size == requestedIds.size) {
            "Targeted Poweramp file query contains duplicate IDs"
        }
        if (requestedIds.isEmpty()) return emptyMap()
        val requestedIdSet = requestedIds.toSet()
        val filesUri = ROOT_URI.buildUpon().appendEncodedPath("files").build()
        val projection = arrayOf(
            "folder_files._id",
            "artist",
            "album",
            "title_tag",
            "folder_files.duration",
            "path",
            "folder_files.name",
            "folder_files.offset_ms",
            "cue_folder_id",
        )
        val rowsById = try {
            val cursor = context.contentResolver.query(
                filesUri,
                projection,
                "folder_files._id IN (${requestedIds.joinToString(",") { "?" }})",
                requestedIds.map(Long::toString).toTypedArray(),
                null,
            ) ?: throw PowerampProviderSnapshotException(
                "Poweramp file query returned no cursor",
            )
            cursor.use {
                val idIdx = it.getColumnIndexOrThrow("_id")
                val artistIdx = it.getColumnIndexOrThrow("artist")
                val albumIdx = it.getColumnIndexOrThrow("album")
                val titleIdx = it.getColumnIndexOrThrow("title_tag")
                val durationIdx = it.getColumnIndexOrThrow("duration")
                val pathIdx = it.getColumnIndexOrThrow("path")
                val nameIdx = it.getColumnIndexOrThrow("name")
                val offsetIdx = it.getColumnIndexOrThrow("offset_ms")
                val cueFolderIdx = it.getColumnIndexOrThrow("cue_folder_id")
                buildMap<Long, PowerampFileEntry> {
                    while (it.moveToNext()) {
                        val id = it.getLong(idIdx)
                        require(id in requestedIdSet) {
                            "Poweramp targeted query returned unexpected file $id"
                        }
                        val artist = TrackNormalization.normalizeArtist(it.getString(artistIdx))
                        val album = TrackNormalization.normalizeAlbum(it.getString(albumIdx))
                        val title = TrackNormalization.normalizeTitle(it.getString(titleIdx))
                        val durationMs = it.getInt(durationIdx).coerceAtLeast(0)
                        val fileName = it.getString(nameIdx)
                        val folder = it.getString(pathIdx).orEmpty()
                        val name = fileName.orEmpty()
                        val entry = PowerampFileEntry(
                                id = id,
                                artist = artist,
                                album = album,
                                title = title,
                                durationMs = durationMs,
                                path = TrackNormalization.normalizePath(
                                    if (name.isNotEmpty()) File(folder, name).path else null,
                                ),
                                offsetMs = if (!it.isNull(offsetIdx)) it.getLong(offsetIdx) else 0L,
                                offsetWasNull = it.isNull(offsetIdx),
                                cueFolderId = if (!it.isNull(cueFolderIdx)) {
                                    it.getLong(cueFolderIdx)
                                } else {
                                    null
                                },
                                metadataKey = TrackNormalization.buildMetadataKey(
                                    artist,
                                    album,
                                    title,
                                    durationMs,
                                ),
                                filenameKeys = TrackNormalization.buildFilenameKeys(
                                    artist,
                                    title,
                                    fileName?.substringBeforeLast('.', fileName),
                                ),
                                providerFileName = fileName,
                            )
                        require(put(id, entry) == null) {
                            "Poweramp targeted query returned duplicate file $id"
                        }
                    }
                }
            }
        } catch (failure: PowerampProviderSnapshotException) {
            throw failure
        } catch (failure: Exception) {
            throw PowerampProviderSnapshotException(
                "Poweramp file query failed before completion",
                failure,
            )
        }
        require(rowsById.keys == requestedIdSet) {
            val missing = requestedIdSet - rowsById.keys
            "Poweramp files are absent from the provider: ${missing.sorted().joinToString()}"
        }
        return requestedIds.associateWith { fileId -> checkNotNull(rowsById[fileId]) }
    }

    /**
     * Read one all-or-nothing Poweramp library snapshot.
     *
     * Rows are staged privately and published only after the cursor reaches its end. A null
     * cursor, missing projected column, or bad row aborts the snapshot instead of leaking a
     * partial library into V2 matching.
     */
    fun requireCompleteFileSnapshot(context: Context): PowerampLibrarySnapshot {
        val filesUri = ROOT_URI.buildUpon().appendEncodedPath("files").build()
        val result = mutableListOf<PowerampFileEntry>()
        val projection = arrayOf(
            "folder_files._id",
            "artist",
            "album",
            "title_tag",
            "folder_files.duration",
            "path",
            "folder_files.name",
            "folder_files.offset_ms",
            "cue_folder_id",
        )
        try {
            val cursor = context.contentResolver.query(
                filesUri,
                projection,
                null,
                null,
                null,
            ) ?: throw PowerampProviderSnapshotException(
                "Poweramp files query returned no cursor",
            )

            cursor.use {
                val idIdx = it.getColumnIndexOrThrow("_id")
                val artistIdx = it.getColumnIndexOrThrow("artist")
                val albumIdx = it.getColumnIndexOrThrow("album")
                val titleIdx = it.getColumnIndexOrThrow("title_tag")
                val durationIdx = it.getColumnIndexOrThrow("duration")
                val pathIdx = it.getColumnIndexOrThrow("path")
                val nameIdx = it.getColumnIndexOrThrow("name")
                val offsetIdx = it.getColumnIndexOrThrow("offset_ms")
                val cueFolderIdx = it.getColumnIndexOrThrow("cue_folder_id")

                while (it.moveToNext()) {
                    val id = it.getLong(idIdx)
                    require(id > 0L) { "Poweramp exposed a non-positive folder_files ID" }
                    val artist = TrackNormalization.normalizeArtist(it.getString(artistIdx))
                    val album = TrackNormalization.normalizeAlbum(it.getString(albumIdx))
                    val title = TrackNormalization.normalizeTitle(it.getString(titleIdx))
                    val durationMs = it.getInt(durationIdx).coerceAtLeast(0)
                    val fileName = it.getString(nameIdx)
                    val folder = it.getString(pathIdx).orEmpty()
                    val name = fileName.orEmpty()
                    val path = TrackNormalization.normalizePath(
                        if (name.isNotEmpty()) File(folder, name).path else null,
                    )
                    result.add(
                        PowerampFileEntry(
                            id = id,
                            artist = artist,
                            album = album,
                            title = title,
                            durationMs = durationMs,
                            path = path,
                            offsetMs = if (!it.isNull(offsetIdx)) it.getLong(offsetIdx) else 0L,
                            offsetWasNull = it.isNull(offsetIdx),
                            cueFolderId = if (!it.isNull(cueFolderIdx)) it.getLong(cueFolderIdx) else null,
                            metadataKey = TrackNormalization.buildMetadataKey(
                                artist,
                                album,
                                title,
                                durationMs,
                            ),
                            filenameKeys = TrackNormalization.buildFilenameKeys(
                                artist,
                                title,
                                fileName?.substringBeforeLast('.', fileName),
                            ),
                            providerFileName = fileName,
                        ),
                    )
                }
            }
            require(result.mapTo(HashSet(result.size)) { it.id }.size == result.size) {
                "Poweramp library snapshot contains duplicate folder_files IDs"
            }
        } catch (failure: PowerampProviderSnapshotException) {
            throw failure
        } catch (failure: Exception) {
            throw PowerampProviderSnapshotException(
                "Poweramp library snapshot failed before completion",
                failure,
            )
        }

        return PowerampLibrarySnapshot(entries = result.toList())
    }
    /**
     * Clear the Poweramp queue.
     */
    fun clearQueue(context: Context) {
        val queueUri = ROOT_URI.buildUpon().appendEncodedPath("queue").build()
        try {
            context.contentResolver.delete(queueUri, null, null)
        } catch (e: Exception) {
            Log.e(TAG, "Error clearing queue", e)
        }
    }

    /**
     * Append tracks and verify the newly observable queue entries by provider readback.
     */
    fun addTracksToQueue(context: Context, fileIds: List<Long>): QueueMutationResult {
        val before = readQueueSnapshot(context)
        if (fileIds.isEmpty()) {
            return QueueMutationResult(
                kind = QueueMutationKind.APPEND,
                requestedFileIds = emptyList(),
                verifiedRequestIndices = emptySet(),
                verifiedFileIds = emptyList(),
                providerReportedInsertCount = 0,
                beforeCount = before.entries.size.takeIf { before.error == null },
                afterCount = before.entries.size.takeIf { before.error == null },
                preservedAnchorFileId = null,
                unexpectedObservedCount = 0,
                fallbackUsed = false,
                verificationError = before.error,
            )
        }
        if (before.error != null) {
            return QueueMutationResult(
                kind = QueueMutationKind.APPEND,
                requestedFileIds = fileIds.toList(),
                verifiedRequestIndices = emptySet(),
                verifiedFileIds = emptyList(),
                providerReportedInsertCount = 0,
                beforeCount = null,
                afterCount = null,
                preservedAnchorFileId = null,
                unexpectedObservedCount = 0,
                fallbackUsed = false,
                mutationError = before.error,
                verificationError = before.error,
            )
        }

        val maxSort = before.entries.maxOfOrNull { it.sort } ?: 0
        val intended = try {
            planQueueInsertions(fileIds, maxSort)
        } catch (_: ArithmeticException) {
            return QueueMutationResult(
                kind = QueueMutationKind.APPEND,
                requestedFileIds = fileIds.toList(),
                verifiedRequestIndices = emptySet(),
                verifiedFileIds = emptyList(),
                providerReportedInsertCount = 0,
                beforeCount = before.entries.size,
                afterCount = before.entries.size,
                preservedAnchorFileId = null,
                unexpectedObservedCount = 0,
                fallbackUsed = false,
                mutationError = "Poweramp queue sort value overflowed",
                verificationError = "Poweramp queue sort value overflowed",
            )
        }
        val insertion = insertQueueEntries(context, intended, before)
        val after = readQueueSnapshot(context)
        val exactPlan = if (after.error == null) {
            PartialQueueInsertionReconciler.reconcile(before.entries, after.entries, intended)
        } else {
            PartialQueueInsertionPlan(false, emptySet(), emptyList(), after.error)
        }
        val transactionError = when {
            insertion.error != null -> insertion.error
            !exactPlan.safeToContinue -> exactPlan.error
            exactPlan.missing.isNotEmpty() ->
                "Poweramp queue omitted ${exactPlan.missing.size} requested occurrences"
            after.entries.size != before.entries.size + fileIds.size ->
                "Poweramp queue contains unexpected occurrences after append"
            else -> null
        }
        if (transactionError != null) {
            val rollback = rollbackQueueSnapshot(context, before, preservedAnchorQueueId = null)
            return failedQueueMutation(
                kind = QueueMutationKind.APPEND,
                requestedFileIds = fileIds,
                before = before,
                after = rollback.snapshot,
                providerReportedInsertCount = insertion.reportedCount,
                fallbackUsed = insertion.fallbackUsed,
                preservedAnchor = null,
                failure = transactionError,
                rollback = rollback,
            )
        }
        val previousIds = before.entries.mapTo(HashSet(before.entries.size)) { it.queueId }
        val observedEntries = after.entries.filterNot { it.queueId in previousIds }
        val match = QueueMutationReconciler.reconcile(fileIds, observedEntries.map { it.fileId })

        return QueueMutationResult(
            kind = QueueMutationKind.APPEND,
            requestedFileIds = fileIds.toList(),
            verifiedRequestIndices = match.requestIndices,
            verifiedFileIds = match.fileIds,
            providerReportedInsertCount = insertion.reportedCount,
            beforeCount = before.entries.size,
            afterCount = after.entries.size.takeIf { after.error == null },
            preservedAnchorFileId = null,
            verifiedQueueEntryIdsByRequestIndex = queueIdsForMatch(match, observedEntries),
            unexpectedObservedCount = match.unexpectedObservedCount,
            fallbackUsed = insertion.fallbackUsed,
        )
    }

    private fun queueIdsForMatch(
        match: QueueMutationReconciler.Match,
        observedEntries: List<QueueEntry>,
    ): Map<Int, Long> = buildMap(match.observedIndicesByRequestIndex.size) {
        for ((requestIndex, observedIndex) in match.observedIndicesByRequestIndex) {
            val queueId = observedEntries.getOrNull(observedIndex)?.queueId ?: continue
            if (queueId > 0L) put(requestIndex, queueId)
        }
    }

    private fun insertQueueEntries(
        context: Context,
        intended: List<PlannedQueueInsertion>,
        baseline: QueueSnapshot,
    ): InsertAttempt {
        val queueUri = ROOT_URI.buildUpon().appendEncodedPath("queue").build()

        // Build batch insert operations
        val operations = ArrayList<ContentProviderOperation>(intended.size)
        for (insertion in intended) {
            operations.add(
                ContentProviderOperation.newInsert(queueUri)
                    .withValue(QUEUE_FOLDER_FILE_ID, insertion.fileId)
                    .withValue(QUEUE_SORT, insertion.sort)
                    .build()
            )
        }

        try {
            val results = context.contentResolver.applyBatch(AUTHORITY, operations)
            val reported = results.count { it.uri != null || (it.count ?: 0) > 0 }
            if (reported == intended.size) return InsertAttempt(reported, false)
            return completePartialQueueInsertion(
                context = context,
                baseline = baseline,
                intended = intended,
                reportedBeforeFallback = reported,
                reason = "Poweramp batch reported $reported of ${intended.size} inserts",
            )
        } catch (e: Exception) {
            Log.e(TAG, "Batch queue insert failed; reconciling before fallback", e)
            return completePartialQueueInsertion(
                context = context,
                baseline = baseline,
                intended = intended,
                reportedBeforeFallback = 0,
                reason = e.message ?: e.javaClass.simpleName,
            )
        }
    }

    private fun planQueueInsertions(
        fileIds: List<Long>,
        maxSort: Int,
    ): List<PlannedQueueInsertion> = fileIds.mapIndexed { index, fileId ->
        PlannedQueueInsertion(
            requestIndex = index,
            fileId = fileId,
            sort = Math.addExact(maxSort, index + 1),
        )
    }

    private fun completePartialQueueInsertion(
        context: Context,
        baseline: QueueSnapshot,
        intended: List<PlannedQueueInsertion>,
        reportedBeforeFallback: Int,
        reason: String,
    ): InsertAttempt {
        val observed = readQueueSnapshot(context)
        if (observed.error != null) {
            return InsertAttempt(
                reportedBeforeFallback,
                false,
                "$reason; partial batch state could not be read: ${observed.error}",
            )
        }
        val plan = PartialQueueInsertionReconciler.reconcile(
            baseline = baseline.entries,
            observed = observed.entries,
            intended = intended,
        )
        if (!plan.safeToContinue) {
            return InsertAttempt(
                plan.appliedRequestIndices.size,
                false,
                "$reason; ${plan.error}",
            )
        }
        if (plan.missing.isEmpty()) {
            return InsertAttempt(plan.appliedRequestIndices.size, false)
        }

        val queueUri = ROOT_URI.buildUpon().appendEncodedPath("queue").build()
        var added = 0
        var failed = 0
        for (insertion in plan.missing) {
            val values = ContentValues().apply {
                put(QUEUE_FOLDER_FILE_ID, insertion.fileId)
                put(QUEUE_SORT, insertion.sort)
            }
            try {
                if (context.contentResolver.insert(queueUri, values) != null) added++ else failed++
            } catch (error: Exception) {
                failed++
                Log.w(TAG, "Failed fallback queue occurrence ${insertion.requestIndex}", error)
            }
        }
        return InsertAttempt(
            reportedCount = plan.appliedRequestIndices.size + added,
            fallbackUsed = true,
            error = if (failed > 0) "$failed missing queue occurrences failed to insert" else null,
        )
    }

    private fun readQueueSnapshot(context: Context): QueueSnapshot {
        val queueUri = ROOT_URI.buildUpon().appendEncodedPath("queue").build()
        return try {
            val cursor = context.contentResolver.query(
                queueUri,
                arrayOf("queue._id", "queue.folder_file_id", "queue.sort"),
                null,
                null,
                null,
            ) ?: return QueueSnapshot(error = "Poweramp queue query returned no cursor")

            val entries = cursor.use {
                val queueIdIndex = it.getColumnIndexOrThrow("_id")
                val fileIdIndex = it.getColumnIndexOrThrow(QUEUE_FOLDER_FILE_ID)
                val sortIndex = it.getColumnIndexOrThrow(QUEUE_SORT)
                buildList {
                    while (it.moveToNext()) {
                        val queueId = it.getLong(queueIdIndex)
                        val fileId = it.getLong(fileIdIndex)
                        require(queueId > 0L && fileId > 0L) {
                            "Poweramp queue exposed a non-positive occurrence or file ID"
                        }
                        add(
                            QueueEntry(
                                queueId = queueId,
                                fileId = fileId,
                                sort = it.getInt(sortIndex),
                            )
                        )
                    }
                }
            }.sortedWith(compareBy(QueueEntry::sort, QueueEntry::queueId))
            require(entries.mapTo(HashSet(entries.size)) { it.queueId }.size == entries.size) {
                "Poweramp queue snapshot contains duplicate occurrence IDs"
            }
            QueueSnapshot(entries = entries)
        } catch (e: Exception) {
            Log.e(TAG, "Could not read Poweramp queue for verification", e)
            QueueSnapshot(error = e.message ?: e.javaClass.simpleName)
        }
    }

    /** A command may use queue occurrence identity only after one complete provider read. */
    internal fun requireCompleteQueueSnapshot(context: Context): List<QueueEntry> {
        val snapshot = readQueueSnapshot(context)
        if (snapshot.error != null) {
            throw PowerampProviderSnapshotException(
                "Poweramp queue snapshot failed before completion: ${snapshot.error}",
            )
        }
        return snapshot.entries.toList()
    }

    private fun rollbackQueueSnapshot(
        context: Context,
        original: QueueSnapshot,
        preservedAnchorQueueId: Long?,
    ): RollbackAttempt {
        require(original.error == null) { "cannot restore an unreadable queue snapshot" }
        val queueUri = ROOT_URI.buildUpon().appendEncodedPath("queue").build()
        val errors = mutableListOf<String>()

        // Append failures normally need only remove occurrences created by this transaction. This
        // preserves every original physical queue row and is attempted before destructive repair.
        val current = readQueueSnapshot(context)
        if (current.error == null && original.entries.all { expected ->
                current.entries.any { it == expected }
            }
        ) {
            val originalIds = original.entries.mapTo(HashSet()) { it.queueId }
            for (extra in current.entries.filterNot { it.queueId in originalIds }) {
                try {
                    if (context.contentResolver.delete(
                            queueUri,
                            "queue._id = ?",
                            arrayOf(extra.queueId.toString()),
                        ) != 1
                    ) {
                        errors += "Poweramp did not report removing queue occurrence ${extra.queueId}"
                    }
                } catch (error: Exception) {
                    errors += "Could not remove queue occurrence ${extra.queueId}: ${error.message}"
                }
            }
            val selectivelyRestored = readQueueSnapshot(context)
            val selectiveError = selectivelyRestored.error
                ?: QueueRollbackVerifier.error(
                    original.entries,
                    selectivelyRestored.entries,
                    preservedAnchorQueueId,
                )
            if (selectiveError == null) {
                return RollbackAttempt(
                    selectivelyRestored,
                    true,
                    errors.distinct().takeIf { it.isNotEmpty() }?.joinToString("; "),
                )
            }
            selectiveError?.let(errors::add)
        } else {
            current.error?.let(errors::add)
        }

        val originalAnchor = preservedAnchorQueueId?.let { queueId ->
            original.entries.singleOrNull { it.queueId == queueId }
        }
        val currentBeforeRepair = readQueueSnapshot(context)
        val canPreserveAnchor = originalAnchor != null && currentBeforeRepair.error == null &&
            currentBeforeRepair.entries.any { it == originalAnchor }
        try {
            if (canPreserveAnchor) {
                context.contentResolver.delete(
                    queueUri,
                    "queue._id != ?",
                    arrayOf(preservedAnchorQueueId.toString()),
                )
            } else {
                context.contentResolver.delete(queueUri, null, null)
            }
        } catch (error: Exception) {
            errors += "Could not clear failed queue transaction: ${error.message}"
        }

        for (entry in original.entries) {
            if (canPreserveAnchor && entry.queueId == preservedAnchorQueueId) continue
            val values = ContentValues().apply {
                put(QUEUE_FOLDER_FILE_ID, entry.fileId)
                put(QUEUE_SORT, entry.sort)
            }
            try {
                if (context.contentResolver.insert(queueUri, values) == null) {
                    errors += "Poweramp rejected restoration of file ${entry.fileId}"
                }
            } catch (error: Exception) {
                errors += "Could not restore file ${entry.fileId}: ${error.message}"
            }
        }
        val restored = readQueueSnapshot(context)
        val verificationError = restored.error ?: QueueRollbackVerifier.error(
            original.entries,
            restored.entries,
            preservedAnchorQueueId,
        )
        verificationError?.let(errors::add)
        return RollbackAttempt(
            snapshot = restored,
            verified = verificationError == null,
            error = errors.distinct().takeIf { it.isNotEmpty() }?.joinToString("; "),
        )
    }

    private fun failedQueueMutation(
        kind: QueueMutationKind,
        requestedFileIds: List<Long>,
        before: QueueSnapshot,
        after: QueueSnapshot,
        providerReportedInsertCount: Int,
        fallbackUsed: Boolean,
        preservedAnchor: QueueEntry?,
        failure: String,
        rollback: RollbackAttempt,
    ): QueueMutationResult {
        val rollbackStatus = if (rollback.verified) {
            "Queue mutation failed; the original ordered queue content was restored"
        } else {
            buildString {
                append("Queue mutation failed and the original queue snapshot could not be verified")
                rollback.error?.let {
                    append(": ")
                    append(it)
                }
            }
        }
        return QueueMutationResult(
            kind = kind,
            requestedFileIds = requestedFileIds.toList(),
            verifiedRequestIndices = emptySet(),
            verifiedFileIds = emptyList(),
            providerReportedInsertCount = providerReportedInsertCount,
            beforeCount = before.entries.size.takeIf { before.error == null },
            afterCount = after.entries.size.takeIf { after.error == null },
            preservedAnchorFileId = preservedAnchor?.fileId,
            preservedAnchorQueueId = preservedAnchor?.queueId,
            unexpectedObservedCount = 0,
            fallbackUsed = fallbackUsed,
            mutationError = failure,
            verificationError = rollbackStatus,
            rollbackAttempted = true,
            rollbackVerified = rollback.verified,
            rollbackError = rollback.error,
        )
    }

    /** Final whole-plan readback used before a durable radio request may commit. */
    fun verifyCurrentQueuePlan(
        context: Context,
        kind: QueueMutationKind,
        preservedAnchorQueueId: Long?,
        expectedFileIds: List<Long>,
        expectedQueueEntryIdsByRequestIndex: Map<Int, Long>? = null,
    ): QueueMutationResult {
        val snapshot = readQueueSnapshot(context)
        val anchor = preservedAnchorQueueId?.let { queueId ->
            snapshot.entries.firstOrNull { it.queueId == queueId }
        }
        val verificationErrors = buildList {
            snapshot.error?.let(::add)
            if (snapshot.error == null && preservedAnchorQueueId != null && anchor == null) {
                add("Poweramp queue no longer contains the exact preserved occurrence")
            }
            if (expectedQueueEntryIdsByRequestIndex != null) {
                if (expectedQueueEntryIdsByRequestIndex.keys.any { it !in expectedFileIds.indices }) {
                    add("Expected queue occurrence evidence contains an invalid request index")
                }
                if (expectedQueueEntryIdsByRequestIndex.values.any { it <= 0L } ||
                    expectedQueueEntryIdsByRequestIndex.values.toSet().size !=
                    expectedQueueEntryIdsByRequestIndex.size
                ) {
                    add("Expected queue occurrence evidence is invalid or duplicated")
                }
            }
        }
        val eligibleEntries = if (verificationErrors.isNotEmpty()) {
            emptyList<QueueEntry>()
        } else {
            when (kind) {
                QueueMutationKind.REPLACE -> snapshot.entries
                    .filterNot { it.queueId == anchor?.queueId }
                QueueMutationKind.APPEND -> if (expectedQueueEntryIdsByRequestIndex == null) {
                    snapshot.entries.takeLast(expectedFileIds.size.coerceAtMost(snapshot.entries.size))
                } else {
                    snapshot.entries
                }
            }
        }
        val match = if (verificationErrors.isNotEmpty()) {
            QueueMutationReconciler.Match(emptySet(), emptyList(), 0)
        } else if (expectedQueueEntryIdsByRequestIndex != null) {
            QueueMutationReconciler.reconcileExactOccurrences(
                requestedFileIds = expectedFileIds,
                expectedQueueEntryIdsByRequestIndex = expectedQueueEntryIdsByRequestIndex,
                observedOccurrences = eligibleEntries.map {
                    QueueMutationReconciler.ObservedOccurrence(it.queueId, it.fileId)
                },
                countUnmatchedObserved = kind == QueueMutationKind.REPLACE,
            )
        } else {
            QueueMutationReconciler.reconcile(expectedFileIds, eligibleEntries.map { it.fileId })
        }
        return QueueMutationResult(
            kind = kind,
            requestedFileIds = expectedFileIds.toList(),
            verifiedRequestIndices = match.requestIndices,
            verifiedFileIds = match.fileIds,
            providerReportedInsertCount = 0,
            beforeCount = snapshot.entries.size.takeIf { snapshot.error == null },
            afterCount = snapshot.entries.size.takeIf { snapshot.error == null },
            preservedAnchorFileId = anchor?.fileId,
            preservedAnchorQueueId = anchor?.queueId,
            verifiedQueueEntryIdsByRequestIndex = queueIdsForMatch(match, eligibleEntries),
            unexpectedObservedCount = match.unexpectedObservedCount,
            fallbackUsed = false,
            verificationError = verificationErrors.takeIf { it.isNotEmpty() }?.joinToString("; "),
        )
    }

    /**
     * Check if a file is currently in the Poweramp queue.
     */
    fun isInQueue(context: Context, fileId: Long): Boolean {
        val snapshot = readQueueSnapshot(context)
        return snapshot.error == null && snapshot.entries.any { it.fileId == fileId }
    }

    /**
     * Replace upcoming queue contents with V1's Poweramp behavior.
     *
     * If the current file already occurs in Queue, retain its first occurrence, delete every
     * other occurrence, and append the new plan after it. Otherwise replace Queue with the plan.
     * V2 adds provider readback and rollback around that same observable mutation.
     */
    fun replaceQueue(
        context: Context,
        currentTrack: PowerampTrack?,
        newFileIds: List<Long>,
    ): QueueMutationResult {
        val queueUri = ROOT_URI.buildUpon().appendEncodedPath("queue").build()
        val before = readQueueSnapshot(context)
        if (newFileIds.isEmpty()) {
            return QueueMutationResult(
                kind = QueueMutationKind.REPLACE,
                requestedFileIds = emptyList(),
                verifiedRequestIndices = emptySet(),
                verifiedFileIds = emptyList(),
                providerReportedInsertCount = 0,
                beforeCount = before.entries.size.takeIf { before.error == null },
                afterCount = before.entries.size.takeIf { before.error == null },
                preservedAnchorFileId = null,
                unexpectedObservedCount = 0,
                fallbackUsed = false,
                verificationError = before.error,
            )
        }

        if (before.error != null) {
            return rejectedQueueReplacement(
                newFileIds = newFileIds,
                before = before,
                reason = before.error,
            )
        }

        val anchor = currentTrack?.realId?.let { currentFileId ->
            before.entries.firstOrNull { it.fileId == currentFileId }
        }
        val anchorQueueId = anchor?.queueId
        val maxSort = anchor?.sort ?: 0
        val intended = try {
            planQueueInsertions(newFileIds, maxSort)
        } catch (_: ArithmeticException) {
            return rejectedQueueReplacement(
                newFileIds = newFileIds,
                before = before,
                reason = "Poweramp queue sort value overflowed",
                preservedAnchor = anchor,
            )
        }

        val preparationErrors = mutableListOf<String>()
        val deletedCount = try {
            if (anchorQueueId != null) {
                context.contentResolver.delete(
                    queueUri,
                    "queue._id != ?",
                    arrayOf(anchorQueueId.toString()),
                )
            } else {
                context.contentResolver.delete(queueUri, null, null)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error preparing Poweramp queue replacement", e)
            preparationErrors += e.message ?: e.javaClass.simpleName
            null
        }

        if (deletedCount != null) {
            val expectedDeletes = before.entries.count { it.queueId != anchorQueueId }
            if (deletedCount != expectedDeletes) {
                preparationErrors +=
                    "Poweramp reported deleting $deletedCount of $expectedDeletes old queue entries"
            }
        }

        val prepared = readQueueSnapshot(context)
        prepared.error?.let(preparationErrors::add)
        val expectedPreparedEntries = listOfNotNull(anchor)
        if (prepared.error == null && prepared.entries != expectedPreparedEntries) {
            preparationErrors += "Poweramp queue preparation did not produce the exact expected baseline"
        }
        if (deletedCount == null || preparationErrors.isNotEmpty()) {
            val failure = preparationErrors.distinct().joinToString("; ")
                .ifBlank { "Poweramp queue preparation failed" }
            val rollback = rollbackQueueSnapshot(context, before, anchorQueueId)
            return failedQueueMutation(
                kind = QueueMutationKind.REPLACE,
                requestedFileIds = newFileIds,
                before = before,
                after = rollback.snapshot,
                providerReportedInsertCount = 0,
                fallbackUsed = false,
                preservedAnchor = anchor,
                failure = failure,
                rollback = rollback,
            )
        }

        val insertion = insertQueueEntries(context, intended, prepared)
        val after = readQueueSnapshot(context)

        val exactPlan = if (after.error == null) {
            PartialQueueInsertionReconciler.reconcile(prepared.entries, after.entries, intended)
        } else {
            PartialQueueInsertionPlan(false, emptySet(), emptyList(), after.error)
        }
        val transactionError = when {
            insertion.error != null -> insertion.error
            after.error != null -> after.error
            anchor != null && after.entries.none { it == anchor } ->
                "Poweramp did not preserve the exact active queue anchor"
            !exactPlan.safeToContinue -> exactPlan.error
            exactPlan.missing.isNotEmpty() ->
                "Poweramp queue omitted ${exactPlan.missing.size} requested occurrences"
            after.entries.size != prepared.entries.size + newFileIds.size ->
                "Poweramp queue contains unexpected occurrences after replacement"
            else -> null
        }
        if (transactionError != null) {
            val rollback = rollbackQueueSnapshot(context, before, anchorQueueId)
            return failedQueueMutation(
                kind = QueueMutationKind.REPLACE,
                requestedFileIds = newFileIds,
                before = before,
                after = rollback.snapshot,
                providerReportedInsertCount = insertion.reportedCount,
                fallbackUsed = insertion.fallbackUsed,
                preservedAnchor = anchor,
                failure = transactionError,
                rollback = rollback,
            )
        }

        val preparedIds = prepared.entries.mapTo(HashSet(prepared.entries.size)) { it.queueId }
        val observedEntries = after.entries.filterNot { it.queueId in preparedIds }
        val match = QueueMutationReconciler.reconcile(newFileIds, observedEntries.map { it.fileId })

        return QueueMutationResult(
            kind = QueueMutationKind.REPLACE,
            requestedFileIds = newFileIds.toList(),
            verifiedRequestIndices = match.requestIndices,
            verifiedFileIds = match.fileIds,
            providerReportedInsertCount = insertion.reportedCount,
            beforeCount = before.entries.size,
            afterCount = after.entries.size.takeIf { after.error == null },
            preservedAnchorFileId = anchor?.fileId,
            preservedAnchorQueueId = anchor?.queueId,
            verifiedQueueEntryIdsByRequestIndex = queueIdsForMatch(match, observedEntries),
            unexpectedObservedCount = match.unexpectedObservedCount,
            fallbackUsed = insertion.fallbackUsed,
        )
    }

    private fun rejectedQueueReplacement(
        newFileIds: List<Long>,
        before: QueueSnapshot,
        reason: String?,
        preservedAnchor: QueueEntry? = null,
    ): QueueMutationResult = QueueMutationResult(
        kind = QueueMutationKind.REPLACE,
        requestedFileIds = newFileIds.toList(),
        verifiedRequestIndices = emptySet(),
        verifiedFileIds = emptyList(),
        providerReportedInsertCount = 0,
        beforeCount = before.entries.size.takeIf { before.error == null },
        afterCount = before.entries.size.takeIf { before.error == null },
        preservedAnchorFileId = preservedAnchor?.fileId,
        preservedAnchorQueueId = preservedAnchor?.queueId,
        unexpectedObservedCount = 0,
        fallbackUsed = false,
        mutationError = reason ?: "Poweramp queue replacement was rejected",
        verificationError = reason ?: "Poweramp queue replacement was rejected",
    )

    /**
     * Tell Poweramp to reload its data (after modifying queue).
     */
    fun reloadData(context: Context, table: String = TABLE_QUEUE) {
        val intent = Intent(ACTION_RELOAD_DATA).apply {
            setPackage(POWERAMP_PACKAGE)
            putExtra(EXTRA_PACKAGE, context.packageName)
            putExtra(EXTRA_TABLE, table)
        }
        // Send as broadcast, not activity
        try {
            context.sendBroadcast(intent)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to send reload broadcast", e)
        }
    }

}

/**
 * Represents a track from Poweramp.
 */
data class PowerampTrack(
    val realId: Long,
    val title: String,
    val artist: String?,
    val album: String?,
    val durationMs: Int,
    val path: String?,
    /** Poweramp Track.ID: the current category row, which is not generally REAL_ID. */
    val trackId: Long = -1L,
    /** Poweramp Track.CAT_URI serialized for durable session evidence. */
    val categoryUri: String? = null,
    /** Poweramp Track.POS_IN_LIST, when the broadcast includes it. */
    val positionInList: Int? = null,
) {
    /** Exact queue row only when the live category URI proves this is Queue playback. */
    val queueOccurrenceId: Long?
        get() = PowerampQueueOccurrencePolicy.queueOccurrenceId(
            trackId = trackId,
            categoryUri = categoryUri,
        )

    /**
     * Create a metadata key for matching with embedded tracks.
     */
    val metadataKey: String
        get() {
            return TrackNormalization.buildMetadataKey(
                TrackNormalization.normalizeArtist(artist),
                TrackNormalization.normalizeAlbum(album),
                TrackNormalization.normalizeTitle(title),
                durationMs,
            )
    }
}

/** Pure interpretation of the official Poweramp Track.ID/CAT_URI contract. */
internal object PowerampQueueOccurrencePolicy {
    fun queueOccurrenceId(trackId: Long, categoryUri: String?): Long? =
        trackId.takeIf { it > 0L && isQueueCategory(categoryUri) }

    fun isQueueCategory(categoryUri: String?): Boolean {
        if (categoryUri.isNullOrBlank()) return false
        val parsed = runCatching { URI(categoryUri) }.getOrNull() ?: return false
        return parsed.scheme == "content" &&
            parsed.authority == "com.maxmpz.audioplayer.data" &&
            parsed.path?.trimEnd('/') == "/queue"
    }
}

/** One complete provider read. A snapshot object is never constructed from partial rows. */
data class PowerampLibrarySnapshot(
    val entries: List<PowerampFileEntry>,
)

class PowerampProviderSnapshotException(
    message: String,
    cause: Throwable? = null,
) : IllegalStateException(message, cause)

/**
 * A Poweramp file entry with NFC-normalized individual fields.
 * Used by TrackMatcher for robust matching without pipe-delimited key splitting.
 */
data class PowerampFileEntry(
    val id: Long,
    val artist: String,
    val album: String,
    val title: String,
    val durationMs: Int,
    val path: String?,
    /** Start of this logical row within its physical source file. */
    val offsetMs: Long = 0L,
    /** Distinguishes an absent provider offset from an explicit zero offset. */
    val offsetWasNull: Boolean = false,
    /** Non-null only when the provider exposes an uncut CUE source-image row. */
    val cueFolderId: Long? = null,
    val metadataKey: String,
    val filenameKeys: Set<String>,
    /** Exact provider filename retained for bounded path-group queries. */
    val providerFileName: String? = path?.let { File(it).name },
)
