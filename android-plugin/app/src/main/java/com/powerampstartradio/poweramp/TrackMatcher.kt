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
    }

    private val providerIndex = RequestScopedSnapshot<PowerampProviderIndex>()

    data class MatchResult(
        val embeddedTrack: EmbeddedTrack,
        val matchType: MatchType
    )

    enum class MatchType {
        ACTIVE_CATALOG_EXACT,
        PATH_EXACT,
        METADATA_EXACT,
        ARTIST_ALBUM_TITLE,
        FILENAME,
        ARTIST_TITLE,
        ARTIST_TITLE_FUZZY,
        COMPOSED_QUERY,
        NOT_APPLICABLE,
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

    private data class PowerampProviderIndex(
        val entries: List<PowerampFileEntry>,
        val byPath: Map<String, List<PowerampFileEntry>>,
        val byMetadataKey: Map<String, List<PowerampFileEntry>>,
        val byArtistAlbumTitle: Map<String, List<PowerampFileEntry>>,
        val byArtistTitle: Map<String, List<PowerampFileEntry>>,
        val byTitle: Map<String, List<PowerampFileEntry>>,
        val byFilenameKey: Map<String, List<PowerampFileEntry>>,
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

    /**
     * One matcher is created for one durable request. It publishes one complete provider index
     * atomically and never shares it process-wide, so every later request observes library
     * additions/removals while every operation within this request sees one coherent snapshot.
     */
    private fun requireProviderIndex(context: Context): PowerampProviderIndex =
        providerIndex.require {
            buildProviderIndex(PowerampHelper.requireCompleteFileSnapshot(context))
        }

    private fun buildProviderIndex(snapshot: PowerampLibrarySnapshot): PowerampProviderIndex {
        val entries = snapshot.entries
        val byFilenameKey = HashMap<String, MutableList<PowerampFileEntry>>(entries.size * 2)
        for (entry in entries) {
            for (key in entry.filenameKeys) {
                byFilenameKey.getOrPut(key) { mutableListOf() }.add(entry)
            }
        }
        return PowerampProviderIndex(
            entries = entries,
            byPath = entries.groupByTo(HashMap()) { it.path ?: "" }
                .filterKeys { it.isNotBlank() },
            byMetadataKey = entries.groupByTo(HashMap()) { it.metadataKey },
            byArtistAlbumTitle = entries.groupByTo(HashMap()) {
                "${it.artist}\u0000${it.album}\u0000${it.title}"
            },
            byArtistTitle = entries.groupByTo(HashMap()) { "${it.artist}\u0000${it.title}" },
            byTitle = entries.groupByTo(HashMap()) { it.title },
            byFilenameKey = byFilenameKey,
        ).also {
            Log.d(
                TAG,
                "Request snapshot indexed ${entries.size} Poweramp tracks " +
                    "(${byFilenameKey.size} filename keys)",
            )
        }
    }

    fun findFileId(context: Context, track: EmbeddedTrack): Long? {
        return resolveEntry(track, requireProviderIndex(context))?.id
    }

    /** Resolve against a caller-owned complete snapshot so a multi-track plan cannot mix reads. */
    fun findFileId(snapshot: PowerampLibrarySnapshot, track: EmbeddedTrack): Long? =
        resolveEntry(track, buildProviderIndex(snapshot))?.id

    /** Keeps occurrence order and intentionally does not collapse repeated requested recordings. */
    fun findFileIds(
        snapshot: PowerampLibrarySnapshot,
        tracks: List<EmbeddedTrack>,
    ): List<Long?> {
        val index = buildProviderIndex(snapshot)
        val resolved = tracks.map { track -> resolveEntry(track, index)?.id }
        return IdentityConsistentFileResolutionPolicy.rejectAliasedIdentities(
            trackIds = tracks.map(EmbeddedTrack::id),
            fileIds = resolved,
        )
    }

    fun mapSingleTrackToFileId(
        context: Context,
        similarTrack: SimilarTrack,
        seen: MutableSet<Long>
    ): Long? {
        val index = requireProviderIndex(context)

        val entry = resolveEntry(similarTrack.track, index)
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
        val index = requireProviderIndex(context)

        val seen = mutableSetOf<Long>()
        val result = mutableListOf<MappedTrack>()

        for (similarTrack in similarTracks) {
            val entry = resolveEntry(similarTrack.track, index)
            val fileId = entry?.id?.takeIf(seen::add)
            if (fileId != null) {
                result.add(MappedTrack(similarTrack, fileId))
            } else {
                if (entry != null && entry.id in seen) continue
                result.add(MappedTrack(similarTrack, null))
            }
        }

        val mapped = result.count { it.fileId != null }
        Log.d(TAG, "Mapped $mapped of ${similarTracks.size} similar tracks")
        return result
    }

    fun auditQueueResolution(context: Context, tracks: List<EmbeddedTrack>): QueueAudit {
        val index = requireProviderIndex(context)

        val counts = linkedMapOf<MatchType, Int>()
        val misses = mutableListOf<String>()
        var matched = 0

        for (track in tracks) {
            val resolution = resolveWithType(track, index)
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

    private fun resolveWithType(
        track: EmbeddedTrack,
        index: PowerampProviderIndex,
    ): Pair<PowerampFileEntry, MatchType>? {
        val lookup = track.asLookup()
        val byPath = index.byPath
        val byMetadataKey = index.byMetadataKey
        val byArtistAlbumTitle = index.byArtistAlbumTitle
        val byArtistTitle = index.byArtistTitle
        val byTitle = index.byTitle
        val byFilenameKey = index.byFilenameKey

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

        logMissDiagnostics(lookup, index)
        return null
    }

    private fun resolveEntry(
        track: EmbeddedTrack,
        index: PowerampProviderIndex,
    ): PowerampFileEntry? = resolveWithType(track, index)?.first

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
        }
        return UniqueBestMatchPolicy.choose(scored)
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
        }
        return UniqueBestMatchPolicy.choose(scored)
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

    private fun logMissDiagnostics(
        lookup: EmbeddedCandidate,
        index: PowerampProviderIndex,
    ) {
        val byTitle = index.byTitle
        val byFilenameKey = index.byFilenameKey
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

/** Equal best evidence is ambiguous; provider/cursor order is never a tie-break. */
internal object UniqueBestMatchPolicy {
    fun <T, S : Comparable<S>> choose(scored: List<Pair<T, S>>): T? {
        if (scored.isEmpty()) return null
        var best: Pair<T, S>? = null
        var bestCount = 0
        for (candidate in scored) {
            val comparison = best?.let { candidate.second.compareTo(it.second) } ?: -1
            when {
                best == null || comparison < 0 -> {
                    best = candidate
                    bestCount = 1
                }
                comparison == 0 -> bestCount++
            }
        }
        return best?.first?.takeIf { bestCount == 1 }
    }
}

/** One current Poweramp row cannot prove two different active embedding identities. */
internal object IdentityConsistentFileResolutionPolicy {
    fun rejectAliasedIdentities(
        trackIds: List<Long>,
        fileIds: List<Long?>,
    ): List<Long?> {
        require(trackIds.size == fileIds.size) { "Track and Poweramp resolutions are misaligned" }
        val identitiesByFileId = HashMap<Long, MutableSet<Long>>()
        fileIds.forEachIndexed { index, fileId ->
            if (fileId != null) {
                identitiesByFileId.getOrPut(fileId) { mutableSetOf() }.add(trackIds[index])
            }
        }
        val ambiguousFileIds = identitiesByFileId
            .filterValues { trackIdentities -> trackIdentities.size > 1 }
            .keys
        return fileIds.map { fileId -> fileId?.takeUnless { it in ambiguousFileIds } }
    }
}

/** Caches one success or failure, preventing a request from mixing provider snapshots. */
internal class RequestScopedSnapshot<T> {
    private var attempted = false
    private var value: T? = null
    private var failure: Exception? = null

    @Synchronized
    fun require(loader: () -> T): T {
        if (!attempted) {
            try {
                value = loader()
            } catch (caught: Exception) {
                failure = caught
            } finally {
                attempted = true
            }
        }
        failure?.let { throw it }
        @Suppress("UNCHECKED_CAST")
        return value as T
    }
}
