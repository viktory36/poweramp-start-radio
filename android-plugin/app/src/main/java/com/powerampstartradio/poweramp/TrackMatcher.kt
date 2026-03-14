package com.powerampstartradio.poweramp

import android.content.Context
import android.util.Log
import com.powerampstartradio.data.EmbeddedTrack
import com.powerampstartradio.data.EmbeddingDatabase
import com.powerampstartradio.similarity.SimilarTrack

/**
 * Matches Poweramp tracks to embedded tracks and resolves embedded tracks back to
 * Poweramp file IDs.
 *
 * The same normalization and duration rules are used across both directions so the
 * app has one consistent view of library state.
 */
class TrackMatcher(
    private val embeddingDb: EmbeddingDatabase
) {
    companion object {
        private const val TAG = "TrackMatcher"

        private var cachedEntries: List<PowerampFileEntry>? = null
        private var cachedByPath: Map<String, List<PowerampFileEntry>>? = null
        private var cachedByMetadataKey: Map<String, List<PowerampFileEntry>>? = null
        private var cachedByArtistAlbumTitle: Map<String, List<PowerampFileEntry>>? = null
        private var cachedByArtistTitle: Map<String, List<PowerampFileEntry>>? = null
        private var cachedByTitle: Map<String, List<PowerampFileEntry>>? = null
        private var cachedByFilenameKey: Map<String, List<PowerampFileEntry>>? = null
        private var cachedEntryCount: Int = 0

        fun invalidateCache() {
            cachedEntries = null
            cachedByPath = null
            cachedByMetadataKey = null
            cachedByArtistAlbumTitle = null
            cachedByArtistTitle = null
            cachedByTitle = null
            cachedByFilenameKey = null
            cachedEntryCount = 0
        }
    }

    data class MatchResult(
        val embeddedTrack: EmbeddedTrack,
        val matchType: MatchType
    )

    enum class MatchType {
        PATH_EXACT,
        METADATA_EXACT,
        ARTIST_ALBUM_TITLE,
        FILENAME,
        ARTIST_TITLE,
        ARTIST_TITLE_FUZZY,
        NOT_FOUND
    }

    data class MappedTrack(
        val similarTrack: SimilarTrack,
        val fileId: Long?
    )

    data class QueueAudit(
        val totalTracks: Int,
        val matchedTracks: Int,
        val unmatchedTracks: Int,
        val matchCounts: Map<String, Int>,
        val unmatchedSample: List<String>,
    )

    private data class EmbeddedCandidate(
        val track: EmbeddedTrack,
        val artist: String,
        val album: String,
        val title: String,
        val path: String?,
        val metadataKey: String,
        val metadataPrefix: String,
        val filenameKeys: Set<String>,
    )

    private data class MatchScore(
        val pathPenalty: Int,
        val metadataPenalty: Int,
        val artistPenalty: Int,
        val titlePenalty: Int,
        val albumPenalty: Int,
        val durationPenaltyMs: Int,
    ) : Comparable<MatchScore> {
        override fun compareTo(other: MatchScore): Int {
            return compareValuesBy(
                this,
                other,
                MatchScore::pathPenalty,
                MatchScore::metadataPenalty,
                MatchScore::artistPenalty,
                MatchScore::titlePenalty,
                MatchScore::albumPenalty,
                MatchScore::durationPenaltyMs,
            )
        }
    }

    fun findMatch(powerampTrack: PowerampTrack): MatchResult? {
        val lookup = powerampTrack.asLookup()
        Log.d(TAG, "Finding match for: ${lookup.artist} - ${lookup.title}")

        lookup.path?.let { path ->
            chooseBestEmbedded(lookup, embeddingDb.findTracksByPath(path).map { it.asLookup() })?.let {
                return MatchResult(it.track, MatchType.PATH_EXACT)
            }
        }

        chooseBestEmbedded(lookup, embeddingDb.findTracksByMetadataPrefix(lookup.metadataPrefix).map { it.asLookup() })?.let {
            return MatchResult(
                it.track,
                if (it.metadataKey == lookup.metadataKey) MatchType.METADATA_EXACT else MatchType.ARTIST_ALBUM_TITLE,
            )
        }

        val filenameMatches = lookup.filenameKeys
            .flatMap { key -> embeddingDb.findTracksByFilenameKey(key) }
            .distinctBy { it.id }
            .map { it.asLookup() }
        chooseBestEmbedded(lookup, filenameMatches)?.let {
            return MatchResult(it.track, MatchType.FILENAME)
        }

        if (lookup.artist.isNotEmpty() && lookup.title.isNotEmpty()) {
            chooseBestEmbedded(lookup, embeddingDb.findTracksByArtistAndTitle(lookup.artist, lookup.title).map { it.asLookup() })?.let {
                return MatchResult(it.track, MatchType.ARTIST_TITLE)
            }
        }

        val fuzzyCandidates = embeddingDb.findTracksByTitle(lookup.title)
            .map { it.asLookup() }
            .filter { candidate ->
                artistOverlaps(lookup.artist, candidate.artist) &&
                    TrackNormalization.durationCompatible(lookup.track.durationMs, candidate.track.durationMs)
            }
        chooseBestEmbedded(lookup, fuzzyCandidates)?.let {
            return MatchResult(it.track, MatchType.ARTIST_TITLE_FUZZY)
        }

        Log.d(TAG, "No match found for ${lookup.metadataKey}")
        return null
    }

    private fun ensureCache(context: Context) {
        if (cachedEntries != null) return

        val entries = PowerampHelper.getAllFileEntries(context)
        cachedEntries = entries
        cachedByPath = entries.groupByTo(HashMap()) { it.path ?: "" }.filterKeys { it.isNotBlank() }
        cachedByMetadataKey = entries.groupByTo(HashMap()) { it.metadataKey }
        cachedByArtistAlbumTitle = entries.groupByTo(HashMap()) { "${it.artist}\u0000${it.album}\u0000${it.title}" }
        cachedByArtistTitle = entries.groupByTo(HashMap()) { "${it.artist}\u0000${it.title}" }
        cachedByTitle = entries.groupByTo(HashMap()) { it.title }

        val byFilenameKey = HashMap<String, MutableList<PowerampFileEntry>>(entries.size * 2)
        for (entry in entries) {
            for (key in entry.filenameKeys) {
                byFilenameKey.getOrPut(key) { mutableListOf() }.add(entry)
            }
        }
        cachedByFilenameKey = byFilenameKey
        cachedEntryCount = entries.size

        Log.d(TAG, "Indexed ${entries.size} Poweramp tracks (${byFilenameKey.size} filename keys)")
    }

    fun findFileId(context: Context, track: EmbeddedTrack): Long? {
        ensureCache(context)
        return resolveEntry(track)?.id
    }

    fun mapSingleTrackToFileId(
        context: Context,
        similarTrack: SimilarTrack,
        seen: MutableSet<Long>
    ): Long? {
        ensureCache(context)

        val entry = resolveEntry(similarTrack.track)
        if (entry == null) {
            Log.w(TAG, "MISS: '${similarTrack.track.artist ?: ""}' - '${similarTrack.track.title ?: ""}' (fnKey='${similarTrack.track.filenameKey}')")
            return null
        }
        if (!seen.add(entry.id)) {
            Log.d(TAG, "DUPE: '${similarTrack.track.artist ?: ""}' - '${similarTrack.track.title ?: ""}' → fileId=${entry.id} already queued, skipping")
            return null
        }
        return entry.id
    }

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
                val resolved = resolveEntry(similarTrack.track)
                if (resolved != null && resolved.id in seen) continue
                result.add(MappedTrack(similarTrack, null))
            }
        }

        val mapped = result.count { it.fileId != null }
        Log.d(TAG, "Mapped $mapped of ${similarTracks.size} similar tracks")
        return result
    }

    fun auditQueueResolution(context: Context, tracks: List<EmbeddedTrack>): QueueAudit {
        ensureCache(context)

        val counts = linkedMapOf<MatchType, Int>()
        val misses = mutableListOf<String>()
        var matched = 0

        for (track in tracks) {
            val resolution = resolveWithType(track)
            if (resolution == null) {
                if (misses.size < 50) misses += track.metadataKey
                continue
            }
            matched++
            counts[resolution.second] = (counts[resolution.second] ?: 0) + 1
        }

        return QueueAudit(
            totalTracks = tracks.size,
            matchedTracks = matched,
            unmatchedTracks = tracks.size - matched,
            matchCounts = counts.mapKeys { it.key.name },
            unmatchedSample = misses,
        )
    }

    private fun resolveWithType(track: EmbeddedTrack): Pair<PowerampFileEntry, MatchType>? {
        val lookup = track.asLookup()
        val byPath = cachedByPath!!
        val byMetadataKey = cachedByMetadataKey!!
        val byArtistAlbumTitle = cachedByArtistAlbumTitle!!
        val byArtistTitle = cachedByArtistTitle!!
        val byTitle = cachedByTitle!!
        val byFilenameKey = cachedByFilenameKey!!

        lookup.path?.let { path ->
            chooseBestPoweramp(lookup, byPath[path].orEmpty())?.let {
                return it to MatchType.PATH_EXACT
            }
        }

        chooseBestPoweramp(lookup, byMetadataKey[lookup.metadataKey].orEmpty())?.let {
            return it to MatchType.METADATA_EXACT
        }

        chooseBestPoweramp(lookup, byArtistAlbumTitle["${lookup.artist}\u0000${lookup.album}\u0000${lookup.title}"].orEmpty())?.let {
            return it to MatchType.ARTIST_ALBUM_TITLE
        }

        val filenameCandidates = lookup.filenameKeys
            .flatMap { key -> byFilenameKey[key].orEmpty() }
            .distinctBy { it.id }
        chooseBestPoweramp(lookup, filenameCandidates)?.let {
            return it to MatchType.FILENAME
        }

        chooseBestPoweramp(lookup, byArtistTitle["${lookup.artist}\u0000${lookup.title}"].orEmpty())?.let {
            return it to MatchType.ARTIST_TITLE
        }

        val fuzzyCandidates = byTitle[lookup.title].orEmpty().filter { candidate ->
            artistOverlaps(lookup.artist, candidate.artist) &&
                TrackNormalization.durationCompatible(lookup.track.durationMs, candidate.durationMs)
        }
        chooseBestPoweramp(lookup, fuzzyCandidates)?.let {
            return it to MatchType.ARTIST_TITLE_FUZZY
        }

        logMissDiagnostics(lookup)
        return null
    }

    private fun resolveEntry(track: EmbeddedTrack): PowerampFileEntry? =
        resolveWithType(track)?.first

    private fun chooseBestEmbedded(
        lookup: EmbeddedCandidate,
        candidates: List<EmbeddedCandidate>,
    ): EmbeddedCandidate? {
        if (candidates.isEmpty()) return null
        val liveCandidates = candidates.filter {
            TrackNormalization.durationCompatible(lookup.track.durationMs, it.track.durationMs)
        }
        if (liveCandidates.isEmpty()) return null
        val scored = liveCandidates.map { candidate ->
            candidate to embeddedScore(lookup, candidate)
        }.sortedWith(compareBy<Pair<EmbeddedCandidate, MatchScore>> { it.second })
        return scored.first().first
    }

    private fun chooseBestPoweramp(
        lookup: EmbeddedCandidate,
        candidates: List<PowerampFileEntry>,
    ): PowerampFileEntry? {
        if (candidates.isEmpty()) return null
        val liveCandidates = candidates.filter {
            TrackNormalization.durationCompatible(lookup.track.durationMs, it.durationMs)
        }
        if (liveCandidates.isEmpty()) return null
        val scored = liveCandidates.map { candidate ->
            candidate to powerampScore(lookup, candidate)
        }.sortedWith(compareBy<Pair<PowerampFileEntry, MatchScore>> { it.second })
        return scored.first().first
    }

    private fun embeddedScore(lookup: EmbeddedCandidate, candidate: EmbeddedCandidate): MatchScore {
        return MatchScore(
            pathPenalty = if (lookup.path != null && lookup.path == candidate.path) 0 else 1,
            metadataPenalty = if (lookup.metadataKey == candidate.metadataKey) 0 else 1,
            artistPenalty = if (lookup.artist == candidate.artist) 0 else 1,
            titlePenalty = if (lookup.title == candidate.title) 0 else 1,
            albumPenalty = if (lookup.album == candidate.album) 0 else 1,
            durationPenaltyMs = TrackNormalization.durationPenalty(lookup.track.durationMs, candidate.track.durationMs),
        )
    }

    private fun powerampScore(lookup: EmbeddedCandidate, candidate: PowerampFileEntry): MatchScore {
        return MatchScore(
            pathPenalty = if (lookup.path != null && lookup.path == candidate.path) 0 else 1,
            metadataPenalty = if (lookup.metadataKey == candidate.metadataKey) 0 else 1,
            artistPenalty = if (lookup.artist == candidate.artist) 0 else 1,
            titlePenalty = if (lookup.title == candidate.title) 0 else 1,
            albumPenalty = if (lookup.album == candidate.album) 0 else 1,
            durationPenaltyMs = TrackNormalization.durationPenalty(lookup.track.durationMs, candidate.durationMs),
        )
    }

    private fun logMissDiagnostics(lookup: EmbeddedCandidate) {
        val byTitle = cachedByTitle!!
        val byFilenameKey = cachedByFilenameKey!!
        val words = lookup.title.split(Regex("\\s+"))
            .filter { it.length >= 4 }
            .take(3)
        val fnWords = lookup.filenameKeys
            .flatMap { it.split(Regex("\\s+")) }
            .filter { it.length >= 4 }
            .take(3)

        val nearTitles = byTitle.keys
            .filter { title -> words.isNotEmpty() && words.all { word -> title.contains(word) } }
            .take(5)
        val nearFnKeys = byFilenameKey.keys
            .filter { key -> fnWords.isNotEmpty() && fnWords.all { word -> key.contains(word) } }
            .take(5)

        Log.w(TAG, "MISS DIAGNOSTICS for track ${lookup.track.id}:")
        Log.w(TAG, "  embedded: artist='${lookup.artist}' title='${lookup.title}' album='${lookup.album}'")
        Log.w(TAG, "  fnKeys='${lookup.filenameKeys.joinToString()} '")
        if (nearTitles.isNotEmpty()) {
            for (title in nearTitles) {
                val artists = byTitle[title]?.joinToString(", ") { it.artist } ?: "?"
                Log.w(TAG, "  ~title: '$title' (artists: $artists)")
            }
        }
        if (nearFnKeys.isNotEmpty()) {
            for (key in nearFnKeys) {
                val ids = byFilenameKey[key]?.joinToString(", ") { it.id.toString() } ?: "?"
                Log.w(TAG, "  ~fnKey: '$key' → ids=$ids")
            }
        }
    }

    private fun PowerampTrack.asLookup(): EmbeddedCandidate {
        val artist = TrackNormalization.normalizeArtist(artist)
        val album = TrackNormalization.normalizeAlbum(album)
        val title = TrackNormalization.normalizeTitle(title)
        val path = TrackNormalization.normalizePath(path)
        val metadataKey = TrackNormalization.buildMetadataKey(artist, album, title, durationMs)
        return EmbeddedCandidate(
            track = EmbeddedTrack(
                id = realId,
                metadataKey = metadataKey,
                filenameKey = path
                    ?.substringAfterLast('/')
                    ?.substringBeforeLast('.', missingDelimiterValue = title)
                    ?.let(TrackNormalization::normalizeAsFilename)
                    ?: TrackNormalization.normalizeAsFilename(title),
                artist = artist,
                album = album,
                title = title,
                durationMs = durationMs,
                filePath = path.orEmpty(),
                source = "poweramp",
            ),
            artist = artist,
            album = album,
            title = title,
            path = path,
            metadataKey = metadataKey,
            metadataPrefix = metadataPrefix(metadataKey),
            filenameKeys = TrackNormalization.buildFilenameKeys(
                artist,
                title,
                path?.substringAfterLast('/')?.substringBeforeLast('.', missingDelimiterValue = title),
            ),
        )
    }

    private fun EmbeddedTrack.asLookup(): EmbeddedCandidate {
        val artist = TrackNormalization.normalizeArtist(artist)
        val album = TrackNormalization.normalizeAlbum(album)
        val title = TrackNormalization.normalizeTitle(title)
        val path = TrackNormalization.normalizePath(filePath)
        val metadataKey = TrackNormalization.buildMetadataKey(artist, album, title, durationMs)
        return EmbeddedCandidate(
            track = this,
            artist = artist,
            album = album,
            title = title,
            path = path,
            metadataKey = metadataKey,
            metadataPrefix = metadataPrefix(metadataKey),
            filenameKeys = TrackNormalization.buildFilenameKeys(artist, title, filenameKey),
        )
    }

    private fun metadataPrefix(metadataKey: String): String =
        metadataKey.substringBeforeLast('|') + "|"

    private fun artistOverlaps(a: String, b: String): Boolean {
        if (a.isBlank() || b.isBlank()) return a.isBlank() && b.isBlank()
        if (a == b) return true
        return a.contains(b) || b.contains(a)
    }
}
