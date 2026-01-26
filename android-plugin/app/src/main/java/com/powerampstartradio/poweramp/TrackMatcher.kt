package com.powerampstartradio.poweramp

import android.content.Context
import android.util.Log
import com.powerampstartradio.data.EmbeddedTrack
import com.powerampstartradio.data.EmbeddingDatabase

/**
 * Matches Poweramp tracks to embedded tracks in the database.
 *
 * Uses multiple strategies:
 * 1. Primary: Exact metadata key match (artist|album|title|duration)
 * 2. Fallback: Filename key match
 * 3. Fuzzy: Artist + title only (for compilations)
 */
class TrackMatcher(
    private val embeddingDb: EmbeddingDatabase
) {
    companion object {
        private const val TAG = "TrackMatcher"
    }

    /**
     * Result of matching a Poweramp track to an embedded track.
     */
    data class MatchResult(
        val embeddedTrack: EmbeddedTrack,
        val matchType: MatchType
    )

    enum class MatchType {
        METADATA_EXACT,   // Full metadata key match
        FILENAME,         // Filename-based match
        ARTIST_TITLE,     // Artist + title only (fuzzy)
        NOT_FOUND
    }

    /**
     * Find the best matching embedded track for a Poweramp track.
     */
    fun findMatch(powerampTrack: PowerampTrack): MatchResult? {
        Log.d(TAG, "Finding match for: ${powerampTrack.title} by ${powerampTrack.artist}")

        // Try exact metadata key match first
        val metadataKey = powerampTrack.metadataKey
        Log.d(TAG, "Trying metadata key: $metadataKey")

        embeddingDb.findTrackByMetadataKey(metadataKey)?.let { track ->
            Log.d(TAG, "Found metadata match: ${track.title}")
            return MatchResult(track, MatchType.METADATA_EXACT)
        }

        // Try filename-based match
        powerampTrack.path?.let { path ->
            val filename = path.substringAfterLast("/").substringBeforeLast(".")
                .lowercase()
                .replace(Regex("\\s*[\\(\\[].*?[\\)\\]]"), "") // Remove parentheticals
                .replace(Regex("^\\d+[.\\-\\s]+"), "") // Remove track numbers
                .trim()

            Log.d(TAG, "Trying filename key: $filename")

            embeddingDb.findTrackByFilenameKey(filename)?.let { track ->
                Log.d(TAG, "Found filename match: ${track.title}")
                return MatchResult(track, MatchType.FILENAME)
            }
        }

        // Try fuzzy match with just artist + title
        val artist = powerampTrack.artist
        val title = powerampTrack.title

        if (!artist.isNullOrEmpty() && title.isNotEmpty()) {
            Log.d(TAG, "Trying artist+title match: $artist - $title")

            val matches = embeddingDb.findTracksByArtistAndTitle(artist, title)
            if (matches.isNotEmpty()) {
                // If multiple matches, prefer one with similar duration
                val bestMatch = matches.minByOrNull { track ->
                    kotlin.math.abs(track.durationMs - powerampTrack.durationMs)
                }
                bestMatch?.let { track ->
                    Log.d(TAG, "Found artist+title match: ${track.title}")
                    return MatchResult(track, MatchType.ARTIST_TITLE)
                }
            }
        }

        Log.d(TAG, "No match found")
        return null
    }

    /**
     * Map embedded track IDs to Poweramp file IDs.
     * This is needed because the embedding database uses different IDs than Poweramp.
     */
    fun mapEmbeddedTracksToFileIds(
        context: Context,
        embeddedTracks: List<EmbeddedTrack>
    ): List<Long> {
        // Get all Poweramp file IDs indexed by metadata key
        val powerampFiles = PowerampHelper.getAllFileIds(context)
        Log.d(TAG, "Loaded ${powerampFiles.size} Poweramp file IDs")

        // Also create indexes for fallback matching
        val byArtistAlbumTitle = mutableMapOf<String, Long>()
        val byArtistTitle = mutableMapOf<String, Long>()
        for ((key, id) in powerampFiles) {
            val parts = key.split("|")
            if (parts.size >= 3) {
                val artist = parts[0]
                val album = parts[1]
                val title = parts[2]
                byArtistAlbumTitle["$artist|$album|$title"] = id
                byArtistTitle["$artist|$title"] = id
            }
        }

        val fileIds = mutableListOf<Long>()

        for (track in embeddedTracks) {
            // Try exact metadata key first
            var fileId = powerampFiles[track.metadataKey]

            if (fileId == null) {
                // Try artist|album|title (ignore duration)
                val parts = track.metadataKey.split("|")
                if (parts.size >= 3) {
                    val keyNoDuration = "${parts[0]}|${parts[1]}|${parts[2]}"
                    fileId = byArtistAlbumTitle[keyNoDuration]
                }
            }

            if (fileId == null) {
                // Try artist|title only (ignore album and duration)
                val parts = track.metadataKey.split("|")
                if (parts.size >= 3) {
                    val keyArtistTitle = "${parts[0]}|${parts[2]}"
                    fileId = byArtistTitle[keyArtistTitle]
                }
            }

            if (fileId != null) {
                fileIds.add(fileId)
            } else {
                Log.d(TAG, "Could not find Poweramp file ID for: ${track.title}")
                Log.d(TAG, "  Embedded key: ${track.metadataKey}")
            }
        }

        Log.d(TAG, "Mapped ${fileIds.size} of ${embeddedTracks.size} tracks to Poweramp file IDs")
        return fileIds
    }
}
