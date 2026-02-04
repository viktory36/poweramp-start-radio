package com.powerampstartradio.poweramp

import android.content.Context
import android.util.Log
import com.powerampstartradio.data.EmbeddedTrack
import com.powerampstartradio.data.EmbeddingDatabase
import com.powerampstartradio.similarity.SimilarTrack

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

        // Cache the Poweramp file ID map across invocations (same app session)
        private var cachedFileIds: Map<String, Long>? = null

        // Secondary lookup indexes (built from cachedFileIds)
        private var cachedByArtistAlbumTitle: Map<String, Long>? = null
        private var cachedByArtistTitle: Map<String, Long>? = null
        private var cachedByTitle: Map<String, Map<String, Long>>? = null

        fun invalidateCache() {
            cachedFileIds = null
            cachedByArtistAlbumTitle = null
            cachedByArtistTitle = null
            cachedByTitle = null
        }
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
     * Result of mapping a similar track to Poweramp.
     */
    data class MappedTrack(
        val similarTrack: SimilarTrack,
        val fileId: Long?   // null if not found in Poweramp library
    )

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
     * Ensure all caches (file IDs + secondary lookup indexes) are built.
     * No-op if already cached.
     */
    private fun ensureCache(context: Context) {
        val powerampFiles = cachedFileIds ?: PowerampHelper.getAllFileIds(context).also {
            cachedFileIds = it
        }

        if (cachedByArtistAlbumTitle == null) {
            val byArtistAlbumTitle = mutableMapOf<String, Long>()
            val byArtistTitle = mutableMapOf<String, Long>()
            val byTitle = mutableMapOf<String, MutableMap<String, Long>>()

            for ((key, id) in powerampFiles) {
                val parts = key.split("|")
                if (parts.size >= 3) {
                    val artist = parts[0]
                    val album = parts[1]
                    val title = parts[2]
                    byArtistAlbumTitle["$artist|$album|$title"] = id
                    byArtistTitle["$artist|$title"] = id
                    byTitle.getOrPut(title) { mutableMapOf() }[artist] = id
                }
            }
            Log.d(TAG, "Indexed ${powerampFiles.size} Poweramp tracks")

            cachedByArtistAlbumTitle = byArtistAlbumTitle
            cachedByArtistTitle = byArtistTitle
            cachedByTitle = byTitle
        }
    }

    /**
     * Map a single similar track to its Poweramp file ID.
     *
     * Used by the streaming path to map tracks one at a time as they arrive.
     * The [seen] set is managed by the caller for cross-track dedup.
     *
     * @return the file ID, or null if not found or duplicate
     */
    fun mapSingleTrackToFileId(
        context: Context,
        similarTrack: SimilarTrack,
        seen: MutableSet<Long>
    ): Long? {
        ensureCache(context)
        val byArtistAlbumTitle = cachedByArtistAlbumTitle!!
        val byArtistTitle = cachedByArtistTitle!!
        val byTitle = cachedByTitle!!

        val track = similarTrack.track
        val parts = track.metadataKey.split("|")

        if (parts.size < 3) {
            Log.w(TAG, "MISS: bad metadata_key='${track.metadataKey}' (not enough parts)")
            return null
        }

        val embeddedArtist = parts[0]
        val embeddedAlbum = parts[1]
        val embeddedTitle = parts[2]

        // 1. Try exact artist|album|title
        var fileId = byArtistAlbumTitle["$embeddedArtist|$embeddedAlbum|$embeddedTitle"]

        // 2. Try artist|title (any album)
        if (fileId == null) {
            fileId = byArtistTitle["$embeddedArtist|$embeddedTitle"]
        }

        // 3. Fuzzy: find by title, check artist substring
        if (fileId == null) {
            byTitle[embeddedTitle]?.entries?.find { (powerampArtist, _) ->
                embeddedArtist.isNotEmpty() && (
                    powerampArtist.contains(embeddedArtist) ||
                    embeddedArtist.contains(powerampArtist)
                )
            }?.let { (_, id) -> fileId = id }
        }

        if (fileId == null) {
            Log.w(TAG, "MISS: embedded='$embeddedArtist|$embeddedAlbum|$embeddedTitle' — no Poweramp match")
            return null
        }

        if (fileId in seen) {
            Log.d(TAG, "DUPE: embedded='$embeddedArtist|$embeddedTitle' → fileId=$fileId already queued, skipping")
            return null
        }

        seen.add(fileId!!)
        return fileId
    }

    /**
     * Map similar tracks to Poweramp file IDs, preserving similarity scores.
     * Returns all tracks with their mapping status (fileId is null if not found).
     */
    fun mapSimilarTracksToFileIds(
        context: Context,
        similarTracks: List<SimilarTrack>
    ): List<MappedTrack> {
        ensureCache(context)

        val seen = mutableSetOf<Long>()
        val result = mutableListOf<MappedTrack>()

        for (similarTrack in similarTracks) {
            val fileId = mapSingleTrackToFileId(context, similarTrack, seen)
            // mapSingleTrackToFileId returns null for both not-found and dupes;
            // for batch mapping we still want to include not-found tracks in results
            // but skip dupes. Re-check to distinguish.
            if (fileId != null) {
                result.add(MappedTrack(similarTrack, fileId))
            } else {
                // Check if it was a dupe (fileId was found but already seen)
                val track = similarTrack.track
                val parts = track.metadataKey.split("|")
                if (parts.size >= 3) {
                    val byArtistAlbumTitle = cachedByArtistAlbumTitle!!
                    val byArtistTitle = cachedByArtistTitle!!
                    val byTitle = cachedByTitle!!
                    val embeddedArtist = parts[0]
                    val embeddedAlbum = parts[1]
                    val embeddedTitle = parts[2]
                    val resolvedId = byArtistAlbumTitle["$embeddedArtist|$embeddedAlbum|$embeddedTitle"]
                        ?: byArtistTitle["$embeddedArtist|$embeddedTitle"]
                        ?: byTitle[embeddedTitle]?.entries?.find { (pa, _) ->
                            embeddedArtist.isNotEmpty() && (pa.contains(embeddedArtist) || embeddedArtist.contains(pa))
                        }?.value
                    if (resolvedId != null && resolvedId in seen) {
                        // Dupe — skip entirely (matches old behavior: `continue`)
                        continue
                    }
                }
                result.add(MappedTrack(similarTrack, null))
            }
        }

        val mapped = result.count { it.fileId != null }
        Log.d(TAG, "Mapped $mapped of ${similarTracks.size} similar tracks")
        return result
    }
}
