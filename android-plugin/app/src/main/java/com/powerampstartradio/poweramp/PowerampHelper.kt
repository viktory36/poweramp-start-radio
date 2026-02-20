package com.powerampstartradio.poweramp

import android.content.ComponentName
import android.content.ContentProviderOperation
import android.content.ContentValues
import android.content.Context
import android.content.Intent
import android.database.Cursor
import android.net.Uri
import android.util.Log
import java.text.Normalizer

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
    private const val API_RECEIVER = "com.maxmpz.audioplayer.apiactivity.ApiReceiver"

    // Poweramp content provider authority
    private const val AUTHORITY = "com.maxmpz.audioplayer.data"
    val ROOT_URI: Uri = Uri.parse("content://$AUTHORITY")

    // Actions
    const val ACTION_TRACK_CHANGED = "com.maxmpz.audioplayer.TRACK_CHANGED"
    const val ACTION_STATUS_CHANGED = "com.maxmpz.audioplayer.STATUS_CHANGED"
    const val ACTION_RELOAD_DATA = "com.maxmpz.audioplayer.ACTION_RELOAD_DATA"
    const val ACTION_ASK_FOR_DATA_PERMISSION = "com.maxmpz.audioplayer.ACTION_ASK_FOR_DATA_PERMISSION"

    // Extras
    const val EXTRA_TRACK = "track"
    const val EXTRA_PACKAGE = "pak"
    const val EXTRA_TABLE = "table"
    const val EXTRA_STATE = "state"

    // Track extras
    const val TRACK_REAL_ID = "realId"
    const val TRACK_TITLE = "title"
    const val TRACK_ARTIST = "artist"
    const val TRACK_ALBUM = "album"
    const val TRACK_DURATION = "dur"
    const val TRACK_PATH = "path"

    // Table names
    const val TABLE_QUEUE = "queue"
    const val TABLE_FILES = "folder_files"

    // Queue columns
    const val QUEUE_FOLDER_FILE_ID = "folder_file_id"
    const val QUEUE_SORT = "sort"

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
    fun getCurrentTrackFromIntent(intent: Intent): PowerampTrack? {
        val trackBundle = intent.getBundleExtra(EXTRA_TRACK) ?: return null

        return PowerampTrack(
            realId = trackBundle.getLong(TRACK_REAL_ID, -1),
            title = trackBundle.getString(TRACK_TITLE) ?: "",
            artist = trackBundle.getString(TRACK_ARTIST),
            album = trackBundle.getString(TRACK_ALBUM),
            durationMs = trackBundle.getInt(TRACK_DURATION, 0) * 1000, // Poweramp sends seconds
            path = trackBundle.getString(TRACK_PATH)
        )
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
    fun getAllFileEntries(context: Context): List<PowerampFileEntry> {
        val filesUri = ROOT_URI.buildUpon().appendEncodedPath("files").build()
        val result = mutableListOf<PowerampFileEntry>()

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
                    val rawArtist = (it.getString(artistIdx) ?: "").lowercase().trim()
                    val rawTitle = (it.getString(titleIdx) ?: "").lowercase().trim()
                    result.add(PowerampFileEntry(
                        id = it.getLong(idIdx),
                        artist = normalizeNfc(normalizePowerampArtist(rawArtist)),
                        album = normalizeNfc((it.getString(albumIdx) ?: "").lowercase().trim()),
                        title = normalizeNfc(stripAudioExtension(rawTitle)),
                        durationMs = it.getInt(durationIdx)
                    ))
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error getting all file entries", e)
        }

        return result
    }

    private fun normalizeNfc(s: String): String {
        return Normalizer.normalize(s, Normalizer.Form.NFC)
    }

    /** Poweramp uses "unknown artist" for untagged files; normalize to empty. */
    private fun normalizePowerampArtist(artist: String): String {
        return if (artist == "unknown artist") "" else artist
    }

    /** Poweramp sometimes includes the file extension in title_tag for untagged files. */
    private fun stripAudioExtension(title: String): String {
        val idx = title.lastIndexOf('.')
        if (idx > 0) {
            val ext = title.substring(idx)
            if (ext in AUDIO_EXTENSIONS) return title.substring(0, idx)
        }
        return title
    }

    private val AUDIO_EXTENSIONS = setOf(
        ".mp3", ".flac", ".opus", ".ogg", ".m4a", ".aac", ".wav",
        ".wma", ".ape", ".wv", ".alac", ".aiff", ".aif"
    )

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
     * Add tracks to the Poweramp queue using batch operations.
     */
    fun addTracksToQueue(context: Context, fileIds: List<Long>): Int {
        val queueUri = ROOT_URI.buildUpon().appendEncodedPath("queue").build()

        // Get current max sort value
        var maxSort = 0
        try {
            val cursor = context.contentResolver.query(
                queueUri,
                arrayOf("MAX(queue.sort)"),
                null,
                null,
                null
            )
            cursor?.use {
                if (it.moveToFirst()) {
                    maxSort = it.getInt(0)
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "Could not get max sort", e)
        }

        // Build batch insert operations
        val operations = ArrayList<ContentProviderOperation>(fileIds.size)
        for ((index, fileId) in fileIds.withIndex()) {
            operations.add(
                ContentProviderOperation.newInsert(queueUri)
                    .withValue(QUEUE_FOLDER_FILE_ID, fileId)
                    .withValue(QUEUE_SORT, maxSort + index + 1)
                    .build()
            )
        }

        return try {
            val results = context.contentResolver.applyBatch(AUTHORITY, operations)
            results.count { it.uri != null }
        } catch (e: Exception) {
            Log.e(TAG, "Batch queue insert failed, falling back to individual inserts", e)
            // Fallback to individual inserts
            var added = 0
            for ((index, fileId) in fileIds.withIndex()) {
                val values = ContentValues().apply {
                    put(QUEUE_FOLDER_FILE_ID, fileId)
                    put(QUEUE_SORT, maxSort + index + 1)
                }
                try {
                    val uri = context.contentResolver.insert(queueUri, values)
                    if (uri != null) added++
                } catch (e2: Exception) {
                    Log.w(TAG, "Failed to add track $fileId to queue", e2)
                }
            }
            added
        }
    }

    /**
     * Replace queue contents, preserving the currently playing entry if it's in the queue.
     *
     * If [currentFileId] is found in the queue, all other entries are deleted and new tracks
     * are added after it. This keeps Poweramp's internal position pointer valid — when the
     * current track finishes, Poweramp advances to the first new track.
     *
     * If [currentFileId] is not in the queue (or is null), the queue is cleared entirely
     * before adding new tracks.
     */
    fun replaceQueue(context: Context, currentFileId: Long?, newFileIds: List<Long>): Int {
        val queueUri = ROOT_URI.buildUpon().appendEncodedPath("queue").build()

        if (currentFileId != null) {
            val currentQueueId = findQueueEntryByFileId(context, currentFileId)

            if (currentQueueId != null) {
                // Playing from queue — delete all entries except the current one
                try {
                    context.contentResolver.delete(
                        queueUri,
                        "queue._id != ?",
                        arrayOf(currentQueueId.toString())
                    )
                } catch (e: Exception) {
                    Log.e(TAG, "Error deleting non-current queue entries", e)
                }
                return addTracksToQueue(context, newFileIds)
            }
        }

        // Not playing from queue — clear entirely, then add new tracks
        clearQueue(context)
        return addTracksToQueue(context, newFileIds)
    }

    /**
     * Find a queue entry's _id by its folder_file_id.
     * Returns null if the file is not in the queue.
     */
    private fun findQueueEntryByFileId(context: Context, fileId: Long): Long? {
        val queueUri = ROOT_URI.buildUpon().appendEncodedPath("queue").build()
        return try {
            val cursor = context.contentResolver.query(
                queueUri,
                arrayOf("queue._id"),
                "folder_file_id = ?",
                arrayOf(fileId.toString()),
                null
            )
            cursor?.use {
                if (it.moveToFirst()) it.getLong(0) else null
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error finding queue entry for file $fileId", e)
            null
        }
    }

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
    val path: String?
) {
    /**
     * Create a metadata key for matching with embedded tracks.
     */
    val metadataKey: String
        get() {
            val a = (artist ?: "").lowercase().trim()
            val al = (album ?: "").lowercase().trim()
            val t = title.lowercase().trim()
            val d = (durationMs / 100) * 100
            return "$a|$al|$t|$d"
        }
}

/**
 * A Poweramp file entry with NFC-normalized individual fields.
 * Used by TrackMatcher for robust matching without pipe-delimited key splitting.
 */
data class PowerampFileEntry(
    val id: Long,
    val artist: String,
    val album: String,
    val title: String,
    val durationMs: Int
)
