package com.powerampstartradio.indexing

import android.content.Context
import android.util.Log
import com.powerampstartradio.data.EmbeddingDatabase
import com.powerampstartradio.poweramp.PowerampHelper
import java.text.Normalizer

/**
 * Detects Poweramp tracks that are not yet in the embedding database.
 *
 * Compares the Poweramp library (via content provider) against the embedding DB
 * using the same matching strategies as TrackMatcher (metadata key, artist+title, filename).
 */
class NewTrackDetector(
    private val embeddingDb: EmbeddingDatabase,
) {
    companion object {
        private const val TAG = "NewTrackDetector"
    }

    /**
     * A Poweramp track that needs to be indexed.
     */
    data class UnindexedTrack(
        val powerampFileId: Long,
        val artist: String,
        val album: String,
        val title: String,
        val durationMs: Int,
        val path: String?,
    ) {
        /** Desktop-format metadata key for insertion into the embedding DB. */
        val metadataKey: String
            get() {
                val durationRounded = (durationMs / 100) * 100
                return "$artist|$album|$title|$durationRounded"
            }

        /** Filename-based key for fallback matching. */
        val filenameKey: String
            get() = normalizeAsFilename(
                if (artist.isNotEmpty()) "$artist - $title" else title
            )

        private fun normalizeAsFilename(s: String): String {
            return Normalizer.normalize(
                s.lowercase()
                    .replace(Regex("\\s*[\\(\\[].*?[\\)\\]]"), "")
                    .replace(Regex("^\\d+[.\\-\\s]+"), "")
                    .replace(Regex("\\s+"), " ")
                    .trim(),
                Normalizer.Form.NFC
            )
        }
    }

    /**
     * Find all Poweramp tracks that aren't in the embedding database.
     *
     * @param context Android context for querying Poweramp content provider
     * @return List of unindexed tracks with their Poweramp metadata
     */
    fun findUnindexedTracks(context: Context): List<UnindexedTrack> {
        // Get all Poweramp entries with paths
        val powerampEntries = getAllPowerampEntriesWithPaths(context)
        if (powerampEntries.isEmpty()) {
            Log.w(TAG, "No entries from Poweramp library")
            return emptyList()
        }

        // Get all embedded track metadata keys and file paths
        val embeddedKeys = embeddingDb.getAllMetadataKeys()
        val embeddedPaths = embeddingDb.getAllFilePaths()

        Log.d(TAG, "Poweramp: ${powerampEntries.size} tracks, " +
            "Embedded: ${embeddedKeys.size} metadata keys, ${embeddedPaths.size} paths")

        // Find unmatched entries
        val unindexed = mutableListOf<UnindexedTrack>()

        for (entry in powerampEntries) {
            val metadataKey = entry.metadataKey

            // Check metadata key match
            if (metadataKey in embeddedKeys) continue

            // Check partial metadata key (artist|*|title|*)
            val artistTitlePrefix = "${entry.artist}|"
            val titlePart = "|${entry.title}|"
            val hasPartialMatch = embeddedKeys.any { key ->
                key.startsWith(artistTitlePrefix) && key.contains(titlePart)
            }
            if (hasPartialMatch) continue

            // Check file path match (Poweramp path may match embedded file_path)
            if (entry.path != null && entry.path in embeddedPaths) continue

            unindexed.add(UnindexedTrack(
                powerampFileId = entry.id,
                artist = entry.artist,
                album = entry.album,
                title = entry.title,
                durationMs = entry.durationMs,
                path = entry.path,
            ))
        }

        Log.i(TAG, "Found ${unindexed.size} unindexed tracks")
        return unindexed
    }

    /**
     * Get all Poweramp file entries including file paths.
     * Extends PowerampHelper.getAllFileEntries with the path column.
     */
    private fun getAllPowerampEntriesWithPaths(context: Context): List<PowerampEntryWithPath> {
        val filesUri = PowerampHelper.ROOT_URI.buildUpon()
            .appendEncodedPath("files").build()
        val result = mutableListOf<PowerampEntryWithPath>()

        try {
            val cursor = context.contentResolver.query(
                filesUri,
                arrayOf(
                    "folder_files._id",
                    "artist",
                    "album",
                    "title_tag",
                    "folder_files.duration",
                    "folder_files.path"
                ),
                null, null, null
            )

            cursor?.use {
                val idIdx = it.getColumnIndex("_id")
                val artistIdx = it.getColumnIndex("artist")
                val albumIdx = it.getColumnIndex("album")
                val titleIdx = it.getColumnIndex("title_tag")
                val durationIdx = it.getColumnIndex("duration")
                val pathIdx = it.getColumnIndex("path")

                while (it.moveToNext()) {
                    val rawArtist = (it.getString(artistIdx) ?: "").lowercase().trim()
                    val rawTitle = (it.getString(titleIdx) ?: "").lowercase().trim()
                    val artist = normalizeNfc(normalizePowerampArtist(rawArtist))
                    val title = normalizeNfc(stripAudioExtension(rawTitle))

                    result.add(PowerampEntryWithPath(
                        id = it.getLong(idIdx),
                        artist = artist,
                        album = normalizeNfc((it.getString(albumIdx) ?: "").lowercase().trim()),
                        title = title,
                        durationMs = it.getInt(durationIdx) * 1000,
                        path = if (pathIdx >= 0) it.getString(pathIdx) else null,
                        metadataKey = run {
                            val durationRounded = (it.getInt(durationIdx) * 1000 / 100) * 100
                            "$artist|${normalizeNfc((it.getString(albumIdx) ?: "").lowercase().trim())}|$title|$durationRounded"
                        }
                    ))
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error querying Poweramp files", e)
        }

        return result
    }

    private data class PowerampEntryWithPath(
        val id: Long,
        val artist: String,
        val album: String,
        val title: String,
        val durationMs: Int,
        val path: String?,
        val metadataKey: String,
    )

    private fun normalizeNfc(s: String): String =
        Normalizer.normalize(s, Normalizer.Form.NFC)

    private fun normalizePowerampArtist(artist: String): String =
        if (artist == "unknown artist") "" else artist

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
}
