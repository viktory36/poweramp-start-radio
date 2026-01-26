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

        // Try metadata match (artist + title)
        val metadataKey = powerampTrack.metadataKey
        Log.d(TAG, "Looking for: ${powerampTrack.artist} - ${powerampTrack.title}")

        embeddingDb.findTrackByMetadataKey(metadataKey)?.let { track ->
            Log.d(TAG, "Found match: ${track.title}")
            return MatchResult(track, MatchType.METADATA_EXACT)
        }

        // Try filename-based match (include artist for better precision)
        powerampTrack.path?.let { path ->
            val filename = path.substringAfterLast("/").substringBeforeLast(".")
                .lowercase()
                .replace(Regex("\\s*[\\(\\[].*?[\\)\\]]"), "") // Remove parentheticals
                .replace(Regex("^\\d+[.\\-\\s]+"), "") // Remove track numbers
                .trim()

            // Try with artist prefix first for more precise matching
            val artist = powerampTrack.artist?.lowercase()?.trim()
            if (!artist.isNullOrEmpty()) {
                val artistFilename = "$artist - ${powerampTrack.title.lowercase().trim()}"
                Log.d(TAG, "Trying filename key with artist: $artistFilename")
                embeddingDb.findTrackByFilenameKey(artistFilename)?.let { track ->
                    Log.d(TAG, "Found filename match with artist: ${track.title}")
                    return MatchResult(track, MatchType.FILENAME)
                }
            }

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
     * Uses title + fuzzy artist matching to handle "Artist1; Artist2" cases.
     */
    fun mapEmbeddedTracksToFileIds(
        context: Context,
        embeddedTracks: List<EmbeddedTrack>
    ): List<Long> {
        // Build indexes from Poweramp library
        val powerampFiles = PowerampHelper.getAllFileIds(context)
        val byArtistTitle = mutableMapOf<String, Long>()
        val byTitle = mutableMapOf<String, MutableList<Pair<String, Long>>>() // title -> [(artist, id)]

        for ((key, id) in powerampFiles) {
            val parts = key.split("|")
            if (parts.size >= 3) {
                val artist = parts[0]
                val title = parts[2]
                byArtistTitle["$artist|$title"] = id
                byTitle.getOrPut(title) { mutableListOf() }.add(artist to id)
            }
        }
        Log.d(TAG, "Indexed ${powerampFiles.size} Poweramp tracks")

        val fileIds = mutableListOf<Long>()
        for (track in embeddedTracks) {
            val parts = track.metadataKey.split("|")
            if (parts.size >= 3) {
                val embeddedArtist = parts[0]
                val embeddedTitle = parts[2]

                // Try exact artist|title match first
                var fileId = byArtistTitle["$embeddedArtist|$embeddedTitle"]

                // Fuzzy: find by title, check artist substring
                if (fileId == null) {
                    byTitle[embeddedTitle]?.find { (powerampArtist, _) ->
                        embeddedArtist.isNotEmpty() && (
                            powerampArtist.contains(embeddedArtist) ||
                            embeddedArtist.contains(powerampArtist)
                        )
                    }?.let { (_, id) -> fileId = id }
                }

                if (fileId != null) {
                    fileIds.add(fileId!!)
                } else {
                    Log.d(TAG, "No Poweramp match for: ${track.artist} - ${track.title}")
                }
            }
        }

        Log.d(TAG, "Mapped ${fileIds.size} of ${embeddedTracks.size} tracks")
        return fileIds
    }
}
