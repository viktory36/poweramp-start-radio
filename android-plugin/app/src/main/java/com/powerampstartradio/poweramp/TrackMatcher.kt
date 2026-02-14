package com.powerampstartradio.poweramp

import android.content.Context
import android.util.Log
import com.powerampstartradio.data.EmbeddedTrack
import com.powerampstartradio.data.EmbeddingDatabase
import com.powerampstartradio.similarity.SimilarTrack
import java.text.Normalizer
import kotlin.math.abs

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

        private val PARENTHETICAL_REGEX = Regex("\\s*[\\(\\[].*?[\\)\\]]")
        private val TRACK_PREFIX_REGEX = Regex("^\\d+[.\\-\\s]+")
        private val NON_ALNUM_REGEX = Regex("[^a-z0-9]+")
        private val WHITESPACE_REGEX = Regex("\\s+")
        private val FEAT_REGEX = Regex("\\b(feat\\.?|ft\\.?|featuring)\\b")

        // Cache Poweramp rows and lookup indexes across invocations (same app session)
        private var cachedRecordsById: Map<Long, PowerampFileRecord>? = null
        private var cachedByArtistAlbumTitle: Map<String, List<Long>>? = null
        private var cachedByArtistTitle: Map<String, List<Long>>? = null
        private var cachedByTitle: Map<String, List<Long>>? = null
        private var cachedByFilenameKey: Map<String, List<Long>>? = null
        private var cachedByFilenameCanonical: Map<String, List<Long>>? = null
        private var cachedByCanonicalTitle: Map<String, List<Long>>? = null
        private var cachedByTitleToken: Map<String, List<Long>>? = null

        fun invalidateCache() {
            cachedRecordsById = null
            cachedByArtistAlbumTitle = null
            cachedByArtistTitle = null
            cachedByTitle = null
            cachedByFilenameKey = null
            cachedByFilenameCanonical = null
            cachedByCanonicalTitle = null
            cachedByTitleToken = null
        }

        private fun normalizeText(value: String?): String {
            if (value.isNullOrBlank()) return ""
            val normalized = Normalizer.normalize(value, Normalizer.Form.NFKD)
                .replace(Regex("\\p{M}+"), "")
            return normalized.lowercase().trim()
        }

        private fun normalizeFilenameKey(value: String): String {
            return normalizeText(value)
                .replace(PARENTHETICAL_REGEX, "")
                .replace(TRACK_PREFIX_REGEX, "")
                .replace(WHITESPACE_REGEX, " ")
                .trim()
        }

        private fun canonicalizeFilenameKey(value: String): String {
            return normalizeFilenameKey(value)
                .replace(NON_ALNUM_REGEX, " ")
                .replace(WHITESPACE_REGEX, " ")
                .trim()
        }

        private fun canonicalizeTitle(value: String): String {
            return normalizeText(value)
                .replace(PARENTHETICAL_REGEX, " ")
                .replace(FEAT_REGEX, " ")
                .replace(NON_ALNUM_REGEX, " ")
                .replace(WHITESPACE_REGEX, " ")
                .trim()
        }

        private fun extensionFromPath(path: String?): String {
            if (path.isNullOrBlank()) return ""
            val dot = path.lastIndexOf('.')
            if (dot < 0 || dot == path.length - 1) return ""
            return path.substring(dot + 1).lowercase().trim()
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
        val fileId: Long?,   // null if not found in Poweramp library
        val matchOutcome: MatchOutcome
    )

    enum class MatchReason {
        METADATA_EXACT,
        ARTIST_TITLE,
        TITLE_WITH_ARTIST_FUZZY,
        FILENAME_KEY,
        TITLE_NORMALIZED,
        TITLE_TOKEN_FUZZY,
        BAD_METADATA_KEY,
        NOT_FOUND,
        DUPLICATE_ALREADY_QUEUED,
    }

    data class MatchOutcome(
        val fileId: Long?,
        val reason: MatchReason,
        val score: Float = 0f,
        val candidatesTried: Int = 0,
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
        if (cachedRecordsById != null) return

        val records = PowerampHelper.getAllFileRecords(context)
        val recordsById = HashMap<Long, PowerampFileRecord>(records.size)
        val byArtistAlbumTitle = mutableMapOf<String, MutableList<Long>>()
        val byArtistTitle = mutableMapOf<String, MutableList<Long>>()
        val byTitle = mutableMapOf<String, MutableList<Long>>()
        val byFilenameKey = mutableMapOf<String, MutableList<Long>>()
        val byFilenameCanonical = mutableMapOf<String, MutableList<Long>>()
        val byCanonicalTitle = mutableMapOf<String, MutableList<Long>>()
        val byTitleToken = mutableMapOf<String, MutableList<Long>>()

        fun addToIndex(index: MutableMap<String, MutableList<Long>>, key: String, fileId: Long) {
            if (key.isBlank()) return
            index.getOrPut(key) { mutableListOf() }.add(fileId)
        }

        for (record in records) {
            recordsById[record.fileId] = record

            val artist = normalizeText(record.artist)
            val album = normalizeText(record.album)
            val title = normalizeText(record.title)

            addToIndex(byArtistAlbumTitle, "$artist|$album|$title", record.fileId)
            addToIndex(byArtistTitle, "$artist|$title", record.fileId)
            addToIndex(byTitle, title, record.fileId)

            val titleFilename = normalizeFilenameKey(record.title)
            val titleFilenameCanonical = canonicalizeFilenameKey(record.title)
            addToIndex(byFilenameKey, titleFilename, record.fileId)
            addToIndex(byFilenameCanonical, titleFilenameCanonical, record.fileId)

            if (artist.isNotBlank()) {
                val artistTitle = normalizeFilenameKey("$artist - ${record.title}")
                val artistTitleCanonical = canonicalizeFilenameKey("$artist - ${record.title}")
                addToIndex(byFilenameKey, artistTitle, record.fileId)
                addToIndex(byFilenameCanonical, artistTitleCanonical, record.fileId)
            }

            val canonicalTitle = canonicalizeTitle(record.title)
            addToIndex(byCanonicalTitle, canonicalTitle, record.fileId)
            canonicalTitle.split(' ')
                .asSequence()
                .map { it.trim() }
                .filter { it.length >= 3 }
                .forEach { token ->
                    addToIndex(byTitleToken, token, record.fileId)
                }
        }

        cachedRecordsById = recordsById
        cachedByArtistAlbumTitle = byArtistAlbumTitle.mapValues { it.value.distinct() }
        cachedByArtistTitle = byArtistTitle.mapValues { it.value.distinct() }
        cachedByTitle = byTitle.mapValues { it.value.distinct() }
        cachedByFilenameKey = byFilenameKey.mapValues { it.value.distinct() }
        cachedByFilenameCanonical = byFilenameCanonical.mapValues { it.value.distinct() }
        cachedByCanonicalTitle = byCanonicalTitle.mapValues { it.value.distinct() }
        cachedByTitleToken = byTitleToken.mapValues { it.value.distinct() }

        Log.d(TAG, "Indexed ${recordsById.size} Poweramp tracks")
    }

    private fun selectBestCandidate(
        candidateIds: Set<Long>,
        track: EmbeddedTrack,
        reason: MatchReason
    ): MatchOutcome? {
        if (candidateIds.isEmpty()) return null

        val recordsById = cachedRecordsById ?: return null
        val embeddedArtist = normalizeText(track.artist)
        val embeddedAlbum = normalizeText(track.album)
        val embeddedTitle = normalizeText(track.title)
        val embeddedTitleCanonical = canonicalizeTitle(track.title ?: "")
        val embeddedFilename = normalizeFilenameKey(track.filenameKey)
        val embeddedFilenameCanonical = canonicalizeFilenameKey(track.filenameKey)
        val embeddedDuration = track.durationMs
        val embeddedExt = extensionFromPath(track.filePath)

        var bestId: Long? = null
        var bestScore = Float.NEGATIVE_INFINITY
        var bestDurationDelta = Int.MAX_VALUE

        for (fileId in candidateIds) {
            val record = recordsById[fileId] ?: continue
            val powerampArtist = normalizeText(record.artist)
            val powerampAlbum = normalizeText(record.album)
            val powerampTitle = normalizeText(record.title)
            val powerampTitleCanonical = canonicalizeTitle(record.title)
            val powerampFilename = normalizeFilenameKey(record.title)
            val powerampFilenameCanonical = canonicalizeFilenameKey(record.title)
            val powerampArtistTitleFilename = if (powerampArtist.isNotBlank()) {
                normalizeFilenameKey("$powerampArtist - ${record.title}")
            } else ""
            val powerampArtistTitleFilenameCanonical = if (powerampArtist.isNotBlank()) {
                canonicalizeFilenameKey("$powerampArtist - ${record.title}")
            } else ""

            var score = 0f

            if (embeddedTitle.isNotBlank() && embeddedTitle == powerampTitle) score += 3f
            if (embeddedTitleCanonical.isNotBlank() && embeddedTitleCanonical == powerampTitleCanonical) score += 1.5f

            if (embeddedArtist.isNotBlank()) {
                if (embeddedArtist == powerampArtist) {
                    score += 2f
                } else if (
                    powerampArtist.isNotBlank() &&
                    (embeddedArtist.contains(powerampArtist) || powerampArtist.contains(embeddedArtist))
                ) {
                    score += 0.5f
                }
            }
            if (embeddedAlbum.isNotBlank() && embeddedAlbum == powerampAlbum) score += 1f

            if (embeddedFilename.isNotBlank()) {
                if (embeddedFilename == powerampFilename) score += 2f
                if (embeddedFilename == powerampArtistTitleFilename) score += 3f
            }
            if (embeddedFilenameCanonical.isNotBlank()) {
                if (embeddedFilenameCanonical == powerampFilenameCanonical) score += 1.2f
                if (embeddedFilenameCanonical == powerampArtistTitleFilenameCanonical) score += 1.6f
            }

            val durationDelta = if (embeddedDuration > 0 && record.durationMs > 0) {
                abs(embeddedDuration - record.durationMs)
            } else Int.MAX_VALUE

            if (durationDelta != Int.MAX_VALUE) {
                when {
                    durationDelta <= 1000 -> score += 1.5f
                    durationDelta <= 5000 -> score += 1.2f
                    durationDelta <= 15000 -> score += 0.8f
                    durationDelta <= 30000 -> score += 0.4f
                    else -> score -= 0.5f
                }
            }

            val powerampExt = extensionFromPath(record.path)
            if (embeddedExt.isNotBlank() && powerampExt.isNotBlank() && embeddedExt == powerampExt) {
                score += 0.4f
            }

            val better = score > bestScore || (score == bestScore && durationDelta < bestDurationDelta) ||
                (score == bestScore && durationDelta == bestDurationDelta && (bestId == null || fileId < bestId))
            if (better) {
                bestId = fileId
                bestScore = score
                bestDurationDelta = durationDelta
            }
        }

        return if (bestId != null) {
            MatchOutcome(bestId, reason, score = bestScore, candidatesTried = candidateIds.size)
        } else null
    }

    private fun resolveTrackToFile(track: EmbeddedTrack): MatchOutcome {
        val byArtistAlbumTitle = cachedByArtistAlbumTitle ?: emptyMap()
        val byArtistTitle = cachedByArtistTitle ?: emptyMap()
        val byTitle = cachedByTitle ?: emptyMap()
        val byFilenameKey = cachedByFilenameKey ?: emptyMap()
        val byFilenameCanonical = cachedByFilenameCanonical ?: emptyMap()
        val byCanonicalTitle = cachedByCanonicalTitle ?: emptyMap()
        val byTitleToken = cachedByTitleToken ?: emptyMap()

        val parts = track.metadataKey.split("|")
        if (parts.size < 3) {
            Log.w(TAG, "MISS: bad metadata_key='${track.metadataKey}' (not enough parts)")
            return MatchOutcome(null, MatchReason.BAD_METADATA_KEY)
        }

        val embeddedArtist = normalizeText(parts[0])
        val embeddedAlbum = normalizeText(parts[1])
        val embeddedTitle = normalizeText(parts[2])

        selectBestCandidate(
            byArtistAlbumTitle["$embeddedArtist|$embeddedAlbum|$embeddedTitle"]?.toSet() ?: emptySet(),
            track,
            MatchReason.METADATA_EXACT
        )?.let { return it }

        selectBestCandidate(
            byArtistTitle["$embeddedArtist|$embeddedTitle"]?.toSet() ?: emptySet(),
            track,
            MatchReason.ARTIST_TITLE
        )?.let { return it }

        if (embeddedArtist.isNotBlank()) {
            val titleMatches = byTitle[embeddedTitle].orEmpty().toSet()
            val artistFiltered = titleMatches.filter { fileId ->
                val rec = cachedRecordsById?.get(fileId) ?: return@filter false
                val pArtist = normalizeText(rec.artist)
                pArtist.isNotBlank() && (pArtist.contains(embeddedArtist) || embeddedArtist.contains(pArtist))
            }.toSet()
            selectBestCandidate(
                artistFiltered,
                track,
                MatchReason.TITLE_WITH_ARTIST_FUZZY
            )?.let { return it }
        }

        val filenameKey = normalizeFilenameKey(track.filenameKey)
        selectBestCandidate(
            byFilenameKey[filenameKey]?.toSet() ?: emptySet(),
            track,
            MatchReason.FILENAME_KEY
        )?.let { return it }

        val filenameCanonical = canonicalizeFilenameKey(track.filenameKey)
        selectBestCandidate(
            byFilenameCanonical[filenameCanonical]?.toSet() ?: emptySet(),
            track,
            MatchReason.FILENAME_KEY
        )?.let { return it }

        val canonicalTitle = canonicalizeTitle(track.title ?: embeddedTitle)
        selectBestCandidate(
            byCanonicalTitle[canonicalTitle]?.toSet() ?: emptySet(),
            track,
            MatchReason.TITLE_NORMALIZED
        )?.let { return it }

        // Last-resort fuzzy token match for punctuation/wording variants.
        val tokenCandidates = canonicalTitle.split(" ")
            .asSequence()
            .map { it.trim() }
            .filter { it.length >= 4 }
            .flatMap { token -> byTitleToken[token].orEmpty().asSequence() }
            .toSet()
        selectBestCandidate(tokenCandidates, track, MatchReason.TITLE_TOKEN_FUZZY)
            ?.takeIf { it.score >= 1.5f }
            ?.let { return it }

        Log.w(
            TAG,
            "MISS: embedded='${embeddedArtist}|${embeddedAlbum}|${embeddedTitle}' " +
                "filename='${track.filenameKey}'"
        )
        return MatchOutcome(null, MatchReason.NOT_FOUND)
    }

    /**
     * Map a single similar track to a Poweramp file ID with reason metadata.
     */
    fun mapSingleTrack(
        context: Context,
        similarTrack: SimilarTrack,
        seen: MutableSet<Long>
    ): MatchOutcome {
        ensureCache(context)
        val outcome = resolveTrackToFile(similarTrack.track)
        val fileId = outcome.fileId ?: return outcome

        if (fileId in seen) {
            Log.d(TAG, "DUPE: track='${similarTrack.track.title}' -> fileId=$fileId already queued")
            return MatchOutcome(null, MatchReason.DUPLICATE_ALREADY_QUEUED, score = outcome.score)
        }

        seen.add(fileId)
        return outcome
    }

    /**
     * Legacy wrapper used by older call sites.
     */
    fun mapSingleTrackToFileId(
        context: Context,
        similarTrack: SimilarTrack,
        seen: MutableSet<Long>
    ): Long? {
        return mapSingleTrack(context, similarTrack, seen).fileId
    }

    /**
     * Map similar tracks to Poweramp file IDs, preserving similarity scores.
     * Returns all tracks with their mapping status (fileId is null if not found).
     */
    fun mapSimilarTracksToFileIds(
        context: Context,
        similarTracks: List<SimilarTrack>
    ): List<MappedTrack> {
        val seen = mutableSetOf<Long>()
        val result = mutableListOf<MappedTrack>()
        var duplicatesSkipped = 0

        for (similarTrack in similarTracks) {
            val outcome = mapSingleTrack(context, similarTrack, seen)
            when {
                outcome.reason == MatchReason.DUPLICATE_ALREADY_QUEUED -> {
                    duplicatesSkipped++
                }
                else -> {
                    result.add(MappedTrack(similarTrack, outcome.fileId, outcome))
                }
            }
        }

        val mapped = result.count { it.fileId != null }
        val misses = result.count { it.fileId == null }
        val reasonCounts = result.groupingBy { it.matchOutcome.reason }.eachCount()
            .toMutableMap()
        if (duplicatesSkipped > 0) {
            reasonCounts[MatchReason.DUPLICATE_ALREADY_QUEUED] = duplicatesSkipped
        }
        Log.i(
            TAG,
            "Mapping summary: mapped=$mapped/${similarTracks.size}, misses=$misses, " +
                "duplicatesSkipped=$duplicatesSkipped, reasons=$reasonCounts"
        )
        return result
    }
}
