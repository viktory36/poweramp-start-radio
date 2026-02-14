package com.powerampstartradio.poweramp

import android.content.Context
import android.util.Log
import com.powerampstartradio.data.EmbeddedTrack
import com.powerampstartradio.data.EmbeddingDatabase
import com.powerampstartradio.similarity.SimilarTrack
import java.text.Normalizer

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

        // Cache the Poweramp lookup indexes across invocations (same app session)
        private var cachedByArtistAlbumTitle: Map<String, Long>? = null
        private var cachedByArtistTitle: Map<String, Long>? = null
        private var cachedByTitle: Map<String, Map<String, Long>>? = null
        private var cachedByFilenameKey: Map<String, Long>? = null
        private var cachedEntryCount: Int = 0

        fun invalidateCache() {
            cachedByArtistAlbumTitle = null
            cachedByArtistTitle = null
            cachedByTitle = null
            cachedByFilenameKey = null
            cachedEntryCount = 0
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
     * Ensure all Poweramp lookup indexes are built. No-op if already cached.
     *
     * Builds indexes from individual Poweramp fields (not pipe-delimited keys),
     * so pipes in artist/album/title don't corrupt matching.
     */
    private fun ensureCache(context: Context) {
        if (cachedByArtistAlbumTitle != null) return

        val entries = PowerampHelper.getAllFileEntries(context)

        val byArtistAlbumTitle = HashMap<String, Long>(entries.size)
        val byArtistTitle = HashMap<String, Long>(entries.size)
        val byTitle = HashMap<String, MutableMap<String, Long>>()
        val byFilenameKey = HashMap<String, Long>(entries.size * 2)

        for (entry in entries) {
            // Indexes keyed on individual NFC-normalized fields — immune to pipe corruption
            byArtistAlbumTitle["${entry.artist}\u0000${entry.album}\u0000${entry.title}"] = entry.id
            byArtistTitle["${entry.artist}\u0000${entry.title}"] = entry.id
            byTitle.getOrPut(entry.title) { mutableMapOf() }[entry.artist] = entry.id

            // Filename-key index: normalize Poweramp title as a filename for fallback matching.
            // Catches cases where desktop used filename-as-title but Poweramp has the same text.
            val normalizedTitle = normalizeAsFilename(entry.title)
            if (normalizedTitle.isNotEmpty()) {
                byFilenameKey[normalizedTitle] = entry.id
            }
            // Also index "artist - title" combined, for when the desktop filename was
            // "Artist - Title.ext" but Poweramp split it into separate fields.
            if (entry.artist.isNotEmpty()) {
                val combined = normalizeAsFilename("${entry.artist} - ${entry.title}")
                if (combined.isNotEmpty()) {
                    byFilenameKey[combined] = entry.id
                }
            }
        }

        Log.d(TAG, "Indexed ${entries.size} Poweramp tracks " +
            "(${byFilenameKey.size} filename keys)")

        cachedByArtistAlbumTitle = byArtistAlbumTitle
        cachedByArtistTitle = byArtistTitle
        cachedByTitle = byTitle
        cachedByFilenameKey = byFilenameKey
        cachedEntryCount = entries.size
    }

    /**
     * Resolve an embedded track to its Poweramp file ID using all matching strategies.
     * Returns null if no match found.
     */
    private fun resolveFileId(track: EmbeddedTrack): Long? {
        val byArtistAlbumTitle = cachedByArtistAlbumTitle!!
        val byArtistTitle = cachedByArtistTitle!!
        val byTitle = cachedByTitle!!
        val byFilenameKey = cachedByFilenameKey!!

        // Use individual fields (NFC normalized) — immune to pipes in metadata
        val embeddedArtist = normalizeNfc((track.artist ?: "").lowercase().trim())
        val embeddedAlbum = normalizeNfc((track.album ?: "").lowercase().trim())
        val embeddedTitle = normalizeNfc((track.title ?: "").lowercase().trim())

        // 1. Exact artist + album + title
        var fileId = byArtistAlbumTitle["$embeddedArtist\u0000$embeddedAlbum\u0000$embeddedTitle"]

        // 2. artist + title (any album)
        if (fileId == null) {
            fileId = byArtistTitle["$embeddedArtist\u0000$embeddedTitle"]
        }

        // 3. Fuzzy: find by title, check artist overlap
        if (fileId == null) {
            val candidates = byTitle[embeddedTitle]
            if (candidates != null) {
                if (embeddedArtist.isNotEmpty()) {
                    candidates.entries.find { (powerampArtist, _) ->
                        powerampArtist.contains(embeddedArtist) ||
                            embeddedArtist.contains(powerampArtist)
                    }?.let { (_, id) -> fileId = id }
                } else {
                    // No artist info — accept any match for this title
                    candidates.values.firstOrNull()?.let { id -> fileId = id }
                }
            }
        }

        // 4. Filename key fallback: match embedded filenameKey against normalized
        //    Poweramp titles (and combined "artist - title" keys).
        if (fileId == null && track.filenameKey.isNotEmpty()) {
            val normalizedFnKey = normalizeNfc(track.filenameKey)
            fileId = byFilenameKey[normalizedFnKey]
        }

        return fileId
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

        val fileId = resolveFileId(similarTrack.track)

        if (fileId == null) {
            Log.w(TAG, "MISS: '${similarTrack.track.artist ?: ""}' - '${similarTrack.track.title ?: ""}' " +
                "(fnKey='${similarTrack.track.filenameKey}')")
            return null
        }

        if (fileId in seen) {
            Log.d(TAG, "DUPE: '${similarTrack.track.artist ?: ""}' - '${similarTrack.track.title ?: ""}' " +
                "→ fileId=$fileId already queued, skipping")
            return null
        }

        seen.add(fileId)
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
            if (fileId != null) {
                result.add(MappedTrack(similarTrack, fileId))
            } else {
                // Distinguish not-found from dupe: resolve again without dedup
                val resolvedId = resolveFileId(similarTrack.track)
                if (resolvedId != null && resolvedId in seen) {
                    // Dupe — skip entirely
                    continue
                }
                result.add(MappedTrack(similarTrack, null))
            }
        }

        val mapped = result.count { it.fileId != null }
        Log.d(TAG, "Mapped $mapped of ${similarTracks.size} similar tracks")
        return result
    }

    /** NFC-normalize a string for consistent matching across platforms. */
    private fun normalizeNfc(s: String): String {
        return Normalizer.normalize(s, Normalizer.Form.NFC)
    }

    /**
     * Normalize a string the same way the desktop indexer normalizes filenames:
     * lowercase, strip parentheticals, strip track numbers, collapse whitespace.
     */
    private fun normalizeAsFilename(s: String): String {
        return normalizeNfc(
            s.lowercase()
                .replace(Regex("\\s*[\\(\\[].*?[\\)\\]]"), "")
                .replace(Regex("^\\d+[.\\-\\s]+"), "")
                .replace(Regex("\\s+"), " ")
                .trim()
        )
    }
}
