package com.powerampstartradio.indexing

import android.content.Context
import android.util.Log
import com.powerampstartradio.data.EmbeddedTrack
import com.powerampstartradio.data.EmbeddingDatabase
import com.powerampstartradio.poweramp.PowerampHelper
import com.powerampstartradio.poweramp.PowerampFileEntry
import com.powerampstartradio.poweramp.TrackNormalization
import java.io.File
import java.text.Normalizer

private typealias PowerampEntryWithPath = PowerampFileEntry

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
        private val TRACK_NUMBER_PREFIX = Regex("^\\d+[.\\-\\s]+")
        private val NON_ALPHANUM = Regex("[^a-z0-9 ()']")
        private val MULTI_SPACE = Regex("\\s+")
    }

    /** Categorized reason why a track didn't match any embedded key. */
    enum class FailureReason {
        DURATION_ONLY,       // Same artist+album+title, duration differs
        ARTIST_MISMATCH,     // Artist differs (after normalization)
        TITLE_MISMATCH,      // Same artist but title differs
        NO_SIMILAR_KEY,      // No embedded key shares artist prefix at all
    }

    /** Diagnostic info for a single unmatched track. */
    data class UnmatchedDetail(
        val artist: String,
        val album: String,
        val title: String,
        val durationMs: Int,
        val powerampKey: String,
        val closestEmbeddedKey: String?,
        val failureReason: FailureReason,
        val path: String?,
    )

    /** Full diagnostic result from matching analysis. */
    data class DiagnosticResult(
        val powerampCount: Int,
        val embeddedTrackCount: Int,
        val embeddedKeyCount: Int,
        val embeddedPathCount: Int,
        val embeddedPathsSample: List<String>,
        val exactKeyMatches: Int,
        val partialMatches: Int,
        val pathMatches: Int,
        val unmatchedCount: Int,
        val unmatchedSample: List<UnmatchedDetail>,
        val failureCategories: Map<FailureReason, Int>,
        val dbOnlyCount: Int,
        val dbOnlyOnDeviceCount: Int,
        val dbOnlyMissingCount: Int,
        val dbOnlySample: List<EmbeddedUnmatchedDetail>,
        val matchPassCounts: Map<String, Int>,
    )

    data class EmbeddedUnmatchedDetail(
        val trackId: Long,
        val artist: String?,
        val album: String?,
        val title: String?,
        val durationMs: Int,
        val metadataKey: String,
        val path: String?,
        val source: String,
    )

    private enum class MatchPass {
        PATH_EXACT,
        METADATA_EXACT,
        FILENAME_EXACT,
        ARTIST_ALBUM_TITLE,
        ARTIST_TITLE,
        ARTIST_TITLE_FUZZY,
    }

    private data class MatchPair(
        val powerampId: Long,
        val trackId: Long,
        val pass: MatchPass,
    )

    private data class ComparableEmbeddedTrack(
        val track: EmbeddedTrack,
        val artist: String,
        val album: String,
        val title: String,
        val durationMs: Int,
        val path: String?,
        val metadataKey: String,
        val filenameKeys: Set<String>,
    )

    private data class LibrarySyncReport(
        val powerampEntries: List<PowerampEntryWithPath>,
        val embeddedTracks: List<ComparableEmbeddedTrack>,
        val matchedPairs: List<MatchPair>,
        val passCounts: Map<MatchPass, Int>,
        val unmatchedPoweramp: List<PowerampEntryWithPath>,
        val unmatchedEmbedded: List<ComparableEmbeddedTrack>,
    )

    private data class RepresentationIndex(
        val byPath: Map<String, List<ComparableEmbeddedTrack>>,
        val byMetadataKey: Map<String, List<ComparableEmbeddedTrack>>,
        val byArtistAlbumTitle: Map<String, List<ComparableEmbeddedTrack>>,
        val byArtistTitle: Map<String, List<ComparableEmbeddedTrack>>,
        val byFilenameKey: Map<String, List<ComparableEmbeddedTrack>>,
        val byArtist: Map<String, List<ComparableEmbeddedTrack>>,
    )

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
    fun findUnindexedTracks(
        context: Context,
        onProgress: (String) -> Unit = {},
    ): List<UnindexedTrack> {
        onProgress("Querying Poweramp library...")
        val powerampEntries = getAllPowerampEntriesWithPaths(context)
        onProgress("Reading embedded tracks...")
        val embeddedTracks = embeddingDb.getAllTracks().map { it.asComparableEntry() }
        val representationIndex = buildRepresentationIndex(embeddedTracks)

        onProgress("Checking which tracks are already represented...")
        val unmatched = powerampEntries.filterNot { isRepresentedInDatabase(it, representationIndex) }

        // Collapse duplicate Poweramp rows that describe the same track representation so
        // Manage Tracks does not surface the same song multiple times.
        val unindexed = unmatched
            .distinctBy { it.metadataKey }
            .map { entry ->
            UnindexedTrack(
                powerampFileId = entry.id,
                artist = entry.artist,
                album = entry.album,
                title = entry.title,
                durationMs = entry.durationMs,
                path = entry.path,
            )
        }
        val collapsed = unmatched.size - unindexed.size
        onProgress(
            buildString {
                append("Found ${unindexed.size} unindexed tracks")
                if (collapsed > 0) append(" ($collapsed duplicate rows collapsed)")
            }
        )
        Log.i(TAG, "Found ${unindexed.size} unindexed tracks")
        return unindexed
    }

    /**
     * Find tracks that exist in the embedding DB but are no longer present in Poweramp.
     *
     * Uses the same matching rules as on-device indexing detection so "clean db" reflects
     * the app's actual notion of which records are still represented in the library.
     */
    fun findDatabaseOnlyTracks(
        context: Context,
        onProgress: (String) -> Unit = {},
    ): List<EmbeddedTrack> {
        val report = buildLibrarySyncReport(context, onProgress)
        val musicRoots = collectMusicRoots(report.powerampEntries)
        val (stillPresentOnDevice, missingFromDevice) = report.unmatchedEmbedded.partition { entry ->
            existsOnDeviceStorage(entry.track, musicRoots)
        }
        onProgress(
            buildString {
                append("Found ${missingFromDevice.size} clean-up candidates")
                if (stillPresentOnDevice.isNotEmpty()) {
                    append(" (${stillPresentOnDevice.size} more still exist on device)")
                }
            }
        )
        return missingFromDevice.map { it.track }
    }

    /**
     * Run matching with full diagnostics — categorizes every failure reason.
     * No inference, just detection + analysis.
     *
     * @param context Android context for querying Poweramp content provider
     * @param onProgress Callback for progress updates
     * @return Full diagnostic result with per-strategy counters and failure details
     */
    fun diagnoseMatching(
        context: Context,
        onProgress: (String) -> Unit = {},
    ): DiagnosticResult {
        val report = buildLibrarySyncReport(context, onProgress)
        val unmatchedKeys = report.unmatchedEmbedded.mapTo(HashSet(report.unmatchedEmbedded.size)) { it.metadataKey }
        val keysByArtist = buildKeysByArtist(unmatchedKeys)
        val musicRoots = collectMusicRoots(report.powerampEntries)
        val failureCounts = mutableMapOf<FailureReason, Int>()

        val unmatchedSample = report.unmatchedPoweramp.take(200).map { entry ->
            val reason = categorizeFailure(entry, keysByArtist)
            failureCounts[reason] = (failureCounts[reason] ?: 0) + 1
            UnmatchedDetail(
                artist = entry.artist,
                album = entry.album,
                title = entry.title,
                durationMs = entry.durationMs,
                powerampKey = entry.metadataKey,
                closestEmbeddedKey = findClosestKey(entry, keysByArtist),
                failureReason = reason,
                path = entry.path,
            )
        }

        val dbOnlySample = report.unmatchedEmbedded.take(200).map { entry ->
            EmbeddedUnmatchedDetail(
                trackId = entry.track.id,
                artist = entry.track.artist,
                album = entry.track.album,
                title = entry.track.title,
                durationMs = entry.track.durationMs,
                metadataKey = entry.track.metadataKey,
                path = entry.track.filePath,
                source = entry.track.source,
            )
        }

        val result = DiagnosticResult(
            powerampCount = report.powerampEntries.size,
            embeddedTrackCount = report.embeddedTracks.size,
            embeddedKeyCount = report.embeddedTracks.mapTo(HashSet()) { it.metadataKey }.size,
            embeddedPathCount = report.embeddedTracks.count { it.path != null },
            embeddedPathsSample = report.embeddedTracks.mapNotNull { it.path }.take(5),
            exactKeyMatches = report.passCounts[MatchPass.METADATA_EXACT] ?: 0,
            partialMatches = (report.passCounts[MatchPass.FILENAME_EXACT] ?: 0) +
                (report.passCounts[MatchPass.ARTIST_ALBUM_TITLE] ?: 0) +
                (report.passCounts[MatchPass.ARTIST_TITLE] ?: 0) +
                (report.passCounts[MatchPass.ARTIST_TITLE_FUZZY] ?: 0),
            pathMatches = report.passCounts[MatchPass.PATH_EXACT] ?: 0,
            unmatchedCount = report.unmatchedPoweramp.size,
            unmatchedSample = unmatchedSample,
            failureCategories = failureCounts,
            dbOnlyCount = report.unmatchedEmbedded.size,
            dbOnlyOnDeviceCount = report.unmatchedEmbedded.count { existsOnDeviceStorage(it.track, musicRoots) },
            dbOnlyMissingCount = report.unmatchedEmbedded.count { !existsOnDeviceStorage(it.track, musicRoots) },
            dbOnlySample = dbOnlySample,
            matchPassCounts = report.passCounts.mapKeys { it.key.name },
        )

        Log.i(TAG, "Diagnostic passes: ${result.matchPassCounts}")
        Log.i(TAG, "Diagnostic leftovers: powerampOnly=${result.unmatchedCount}, dbOnly=${result.dbOnlyCount}")
        Log.i(TAG, "Failure categories: $failureCounts")

        return result
    }

    private fun buildLibrarySyncReport(
        context: Context,
        onProgress: (String) -> Unit = {},
    ): LibrarySyncReport {
        onProgress("Querying Poweramp library...")
        val powerampEntries = getAllPowerampEntriesWithPaths(context)
        if (powerampEntries.isEmpty()) {
            Log.w(TAG, "No entries from Poweramp library")
            return LibrarySyncReport(emptyList(), emptyList(), emptyList(), emptyMap(), emptyList(), emptyList())
        }

        onProgress("Reading embedded tracks...")
        val embeddedTracks = embeddingDb.getAllTracks().map { it.asComparableEntry() }

        val remainingPoweramp = LinkedHashMap<Long, PowerampEntryWithPath>(powerampEntries.size)
        powerampEntries.forEach { remainingPoweramp[it.id] = it }
        val remainingEmbedded = LinkedHashMap<Long, ComparableEmbeddedTrack>(embeddedTracks.size)
        embeddedTracks.forEach { remainingEmbedded[it.track.id] = it }
        val pairs = mutableListOf<MatchPair>()
        val passCounts = linkedMapOf<MatchPass, Int>()

        onProgress("Matching exact file paths...")
        pairByExactKey(
            remainingPoweramp,
            remainingEmbedded,
            MatchPass.PATH_EXACT,
            passCounts,
            powerampKey = { it.path?.let(::normalizeNfc)?.takeIf { key -> key.isNotBlank() } },
            embeddedKey = { it.path?.let(::normalizeNfc)?.takeIf { key -> key.isNotBlank() } },
            sink = pairs,
        )

        onProgress("Matching exact tags...")
        pairByExactKey(
            remainingPoweramp,
            remainingEmbedded,
            MatchPass.METADATA_EXACT,
            passCounts,
            powerampKey = { it.metadataKey },
            embeddedKey = { it.metadataKey },
            sink = pairs,
        )

        onProgress("Matching filename fallbacks...")
        pairByFilenameKeys(
            remainingPoweramp,
            remainingEmbedded,
            passCounts,
            pairs,
        )

        onProgress("Matching tags without duration...")
        pairByGroupedDuration(
            remainingPoweramp,
            remainingEmbedded,
            MatchPass.ARTIST_ALBUM_TITLE,
            passCounts,
            sink = pairs,
            powerampKey = { "${it.artist}\u0000${it.album}\u0000${it.title}" },
            embeddedKey = { "${it.artist}\u0000${it.album}\u0000${it.title}" },
        )

        onProgress("Matching artist and title...")
        pairByGroupedDuration(
            remainingPoweramp,
            remainingEmbedded,
            MatchPass.ARTIST_TITLE,
            passCounts,
            sink = pairs,
            powerampKey = { "${it.artist}\u0000${it.title}" },
            embeddedKey = { "${it.artist}\u0000${it.title}" },
        )

        onProgress("Matching fuzzy title leftovers...")
        pairByFuzzyArtistTitle(
            remainingPoweramp,
            remainingEmbedded,
            passCounts,
            pairs,
        )

        val unmatchedPoweramp = remainingPoweramp.values.toList()
        val unmatchedEmbedded = remainingEmbedded.values.toList()
        onProgress("Library sync: matched ${pairs.size}, poweramp-only ${unmatchedPoweramp.size}, db-only ${unmatchedEmbedded.size}")

        return LibrarySyncReport(
            powerampEntries = powerampEntries,
            embeddedTracks = embeddedTracks,
            matchedPairs = pairs,
            passCounts = passCounts,
            unmatchedPoweramp = unmatchedPoweramp,
            unmatchedEmbedded = unmatchedEmbedded,
        )
    }

    private fun buildRepresentationIndex(embeddedTracks: List<ComparableEmbeddedTrack>): RepresentationIndex {
        val byPath = embeddedTracks.mapNotNull { track ->
            track.path?.takeIf { it.isNotBlank() }?.let { it to track }
        }.groupBy({ it.first }, { it.second })

        val byMetadataKey = embeddedTracks.groupBy { it.metadataKey }
        val byArtistAlbumTitle = embeddedTracks.groupBy { "${it.artist}\u0000${it.album}\u0000${it.title}" }
        val byArtistTitle = embeddedTracks.groupBy { "${it.artist}\u0000${it.title}" }
        val byArtist = embeddedTracks.groupBy { it.artist }

        val byFilenameKey = HashMap<String, MutableList<ComparableEmbeddedTrack>>(embeddedTracks.size * 2)
        for (track in embeddedTracks) {
            for (key in track.filenameKeys) {
                byFilenameKey.getOrPut(key) { mutableListOf() }.add(track)
            }
        }

        return RepresentationIndex(
            byPath = byPath,
            byMetadataKey = byMetadataKey,
            byArtistAlbumTitle = byArtistAlbumTitle,
            byArtistTitle = byArtistTitle,
            byFilenameKey = byFilenameKey,
            byArtist = byArtist,
        )
    }

    private fun isRepresentedInDatabase(
        entry: PowerampEntryWithPath,
        index: RepresentationIndex,
    ): Boolean {
        entry.path?.let { path ->
            if (index.byPath[path].orEmpty().any { durationCompatible(it.durationMs, entry.durationMs) }) {
                return true
            }
        }

        if (index.byMetadataKey[entry.metadataKey].orEmpty().isNotEmpty()) return true

        val artistAlbumTitleKey = "${entry.artist}\u0000${entry.album}\u0000${entry.title}"
        if (index.byArtistAlbumTitle[artistAlbumTitleKey].orEmpty()
                .any { durationCompatible(it.durationMs, entry.durationMs) }
        ) {
            return true
        }

        val artistTitleKey = "${entry.artist}\u0000${entry.title}"
        if (index.byArtistTitle[artistTitleKey].orEmpty()
                .any { durationCompatible(it.durationMs, entry.durationMs) }
        ) {
            return true
        }

        val filenameMatches = entry.filenameKeys
            .flatMap { key -> index.byFilenameKey[key].orEmpty() }
            .distinctBy { it.track.id }
        if (filenameMatches.any { durationCompatible(it.durationMs, entry.durationMs) }) {
            return true
        }

        return resolveArtistTrackCandidates(entry.artist, index.byArtist)
            .any { candidate ->
                fuzzyTitlePair(entry.title, candidate.title) &&
                    durationCompatible(entry.durationMs, candidate.durationMs)
            }
    }

    private fun pairByExactKey(
        remainingPoweramp: MutableMap<Long, PowerampEntryWithPath>,
        remainingEmbedded: MutableMap<Long, ComparableEmbeddedTrack>,
        pass: MatchPass,
        passCounts: MutableMap<MatchPass, Int>,
        powerampKey: (PowerampEntryWithPath) -> String?,
        embeddedKey: (ComparableEmbeddedTrack) -> String?,
        sink: MutableList<MatchPair>,
    ) {
        val powerampGroups = HashMap<String, ArrayDeque<Long>>()
        for ((id, entry) in remainingPoweramp) {
            val key = powerampKey(entry) ?: continue
            powerampGroups.getOrPut(key) { ArrayDeque() }.add(id)
        }

        val embeddedGroups = HashMap<String, ArrayDeque<Long>>()
        for ((id, entry) in remainingEmbedded) {
            val key = embeddedKey(entry) ?: continue
            embeddedGroups.getOrPut(key) { ArrayDeque() }.add(id)
        }

        for ((key, powerampIds) in powerampGroups) {
            val embeddedIds = embeddedGroups[key] ?: continue
            while (powerampIds.isNotEmpty() && embeddedIds.isNotEmpty()) {
                val powerampId = powerampIds.removeFirst()
                val trackId = embeddedIds.removeFirst()
                if (!remainingPoweramp.containsKey(powerampId) || !remainingEmbedded.containsKey(trackId)) continue
                remainingPoweramp.remove(powerampId)
                remainingEmbedded.remove(trackId)
                sink.add(MatchPair(powerampId, trackId, pass))
                passCounts[pass] = (passCounts[pass] ?: 0) + 1
            }
        }
    }

    private fun pairByGroupedDuration(
        remainingPoweramp: MutableMap<Long, PowerampEntryWithPath>,
        remainingEmbedded: MutableMap<Long, ComparableEmbeddedTrack>,
        pass: MatchPass,
        passCounts: MutableMap<MatchPass, Int>,
        sink: MutableList<MatchPair>,
        powerampKey: (PowerampEntryWithPath) -> String,
        embeddedKey: (ComparableEmbeddedTrack) -> String,
    ) {
        val powerampGroups = HashMap<String, MutableList<Long>>()
        for ((id, entry) in remainingPoweramp) {
            powerampGroups.getOrPut(powerampKey(entry)) { mutableListOf() }.add(id)
        }
        val embeddedGroups = HashMap<String, MutableList<Long>>()
        for ((id, entry) in remainingEmbedded) {
            embeddedGroups.getOrPut(embeddedKey(entry)) { mutableListOf() }.add(id)
        }

        for ((key, powerampIds) in powerampGroups) {
            val embeddedIds = embeddedGroups[key] ?: continue
            val livePoweramp = powerampIds.filter { remainingPoweramp.containsKey(it) }.toMutableList()
            val liveEmbedded = embeddedIds.filter { remainingEmbedded.containsKey(it) }.toMutableList()

            while (livePoweramp.isNotEmpty() && liveEmbedded.isNotEmpty()) {
                var bestPowerampIndex = -1
                var bestEmbeddedIndex = -1
                var bestPenalty = Int.MAX_VALUE

                for (pi in livePoweramp.indices) {
                    val poweramp = remainingPoweramp[livePoweramp[pi]] ?: continue
                    for (ei in liveEmbedded.indices) {
                        val embedded = remainingEmbedded[liveEmbedded[ei]] ?: continue
                        if (!durationCompatible(poweramp.durationMs, embedded.durationMs)) continue
                        val penalty = durationPenalty(poweramp.durationMs, embedded.durationMs)
                        if (penalty < bestPenalty) {
                            bestPenalty = penalty
                            bestPowerampIndex = pi
                            bestEmbeddedIndex = ei
                        }
                    }
                }

                if (bestPowerampIndex < 0 || bestEmbeddedIndex < 0) break

                val powerampId = livePoweramp.removeAt(bestPowerampIndex)
                val trackId = liveEmbedded.removeAt(bestEmbeddedIndex)
                if (!remainingPoweramp.containsKey(powerampId) || !remainingEmbedded.containsKey(trackId)) continue
                remainingPoweramp.remove(powerampId)
                remainingEmbedded.remove(trackId)
                sink.add(MatchPair(powerampId, trackId, pass))
                passCounts[pass] = (passCounts[pass] ?: 0) + 1
            }
        }
    }

    private fun pairByFilenameKeys(
        remainingPoweramp: MutableMap<Long, PowerampEntryWithPath>,
        remainingEmbedded: MutableMap<Long, ComparableEmbeddedTrack>,
        passCounts: MutableMap<MatchPass, Int>,
        sink: MutableList<MatchPair>,
    ) {
        val powerampByFilename = HashMap<String, MutableSet<Long>>()
        for ((id, entry) in remainingPoweramp) {
            for (key in entry.filenameKeys) {
                powerampByFilename.getOrPut(key) { linkedSetOf() }.add(id)
            }
        }

        val candidatesByTrack = remainingEmbedded.values.mapNotNull { track ->
            val candidates = track.filenameKeys
                .flatMap { powerampByFilename[it].orEmpty() }
                .distinct()
            if (candidates.isEmpty()) null else track.track.id to candidates
        }.sortedBy { it.second.size }

        for ((trackId, candidateIds) in candidatesByTrack) {
            val track = remainingEmbedded[trackId] ?: continue
            val liveCandidates = candidateIds.mapNotNull { remainingPoweramp[it] }
                .filter { durationCompatible(it.durationMs, track.durationMs) }
            val best = selectUniqueBestPoweramp(track, liveCandidates) ?: continue

            remainingPoweramp.remove(best.id)
            remainingEmbedded.remove(trackId)
            sink.add(MatchPair(best.id, trackId, MatchPass.FILENAME_EXACT))
            passCounts[MatchPass.FILENAME_EXACT] = (passCounts[MatchPass.FILENAME_EXACT] ?: 0) + 1
        }
    }

    private fun pairByFuzzyArtistTitle(
        remainingPoweramp: MutableMap<Long, PowerampEntryWithPath>,
        remainingEmbedded: MutableMap<Long, ComparableEmbeddedTrack>,
        passCounts: MutableMap<MatchPass, Int>,
        sink: MutableList<MatchPair>,
    ) {
        val tracksByArtist = HashMap<String, MutableList<ComparableEmbeddedTrack>>()
        for (entry in remainingEmbedded.values) {
            tracksByArtist.getOrPut(entry.artist) { mutableListOf() }.add(entry)
        }

        val powerampOrder = remainingPoweramp.values.sortedBy { it.title.length }
        for (entry in powerampOrder) {
            if (!remainingPoweramp.containsKey(entry.id)) continue

            val candidates = resolveArtistTrackCandidates(entry.artist, tracksByArtist)
                .filter { candidate ->
                    remainingEmbedded.containsKey(candidate.track.id) &&
                        fuzzyTitlePair(entry.title, candidate.title) &&
                        durationCompatible(entry.durationMs, candidate.durationMs)
                }

            val best = selectUniqueBestEmbedded(entry, candidates) ?: continue
            remainingPoweramp.remove(entry.id)
            remainingEmbedded.remove(best.track.id)
            sink.add(MatchPair(entry.id, best.track.id, MatchPass.ARTIST_TITLE_FUZZY))
            passCounts[MatchPass.ARTIST_TITLE_FUZZY] = (passCounts[MatchPass.ARTIST_TITLE_FUZZY] ?: 0) + 1
        }
    }

    private fun selectUniqueBestPoweramp(
        track: ComparableEmbeddedTrack,
        candidates: List<PowerampEntryWithPath>,
    ): PowerampEntryWithPath? {
        if (candidates.isEmpty()) return null
        val scored = candidates.map { candidate ->
            candidate to scoreAgainstEmbedded(track, candidate)
        }.sortedWith(compareBy<Pair<PowerampEntryWithPath, MatchScore>> { it.second })
        if (scored.size > 1 && scored[0].second == scored[1].second) return null
        return scored.first().first
    }

    private fun selectUniqueBestEmbedded(
        entry: PowerampEntryWithPath,
        candidates: List<ComparableEmbeddedTrack>,
    ): ComparableEmbeddedTrack? {
        if (candidates.isEmpty()) return null
        val scored = candidates.map { candidate ->
            candidate to scoreAgainstPoweramp(entry, candidate)
        }.sortedWith(compareBy<Pair<ComparableEmbeddedTrack, MatchScore>> { it.second })
        if (scored.size > 1 && scored[0].second == scored[1].second) return null
        return scored.first().first
    }

    private data class MatchScore(
        val artistPenalty: Int,
        val titlePenalty: Int,
        val albumPenalty: Int,
        val durationPenaltyMs: Int,
    ) : Comparable<MatchScore> {
        override fun compareTo(other: MatchScore): Int {
            return compareValuesBy(
                this,
                other,
                MatchScore::artistPenalty,
                MatchScore::titlePenalty,
                MatchScore::albumPenalty,
                MatchScore::durationPenaltyMs,
            )
        }
    }

    private fun scoreAgainstEmbedded(
        track: ComparableEmbeddedTrack,
        candidate: PowerampEntryWithPath,
    ): MatchScore {
        return MatchScore(
            artistPenalty = if (candidate.artist == track.artist) 0 else 1,
            titlePenalty = if (candidate.title == track.title) 0 else 1,
            albumPenalty = if (candidate.album == track.album) 0 else 1,
            durationPenaltyMs = durationPenalty(candidate.durationMs, track.durationMs),
        )
    }

    private fun scoreAgainstPoweramp(
        entry: PowerampEntryWithPath,
        candidate: ComparableEmbeddedTrack,
    ): MatchScore {
        return MatchScore(
            artistPenalty = if (candidate.artist == entry.artist) 0 else 1,
            titlePenalty = if (candidate.title == entry.title) 0 else 1,
            albumPenalty = if (candidate.album == entry.album) 0 else 1,
            durationPenaltyMs = durationPenalty(candidate.durationMs, entry.durationMs),
        )
    }

    private fun resolveArtistTrackCandidates(
        artist: String,
        tracksByArtist: Map<String, List<ComparableEmbeddedTrack>>,
    ): List<ComparableEmbeddedTrack> {
        val result = mutableListOf<ComparableEmbeddedTrack>()
        val seen = mutableSetOf<Long>()

        fun tryArtist(a: String) {
            tracksByArtist[a]?.forEach { track ->
                if (seen.add(track.track.id)) result.add(track)
            }
        }

        tryArtist(artist)
        if (';' in artist) artist.split(';').forEach { tryArtist(it.trim()) }
        if (',' in artist) tryArtist(artist.substringBefore(',').trim())
        if (" / " in artist) artist.split(" / ").forEach { tryArtist(it.trim()) }
        if ('.' in artist && artist.length > 1) {
            tryArtist(artist.trimEnd('.'))
            tryArtist(artist.replace(".", ""))
        } else if (artist.length in 1..5) {
            tryArtist("$artist.")
        }
        if (" & " in artist) tryArtist(artist.replace(" & ", " and "))
        else if (" and " in artist) tryArtist(artist.replace(" and ", " & "))
        if (artist.isEmpty()) tryArtist("unknown artist")

        if (artist.length >= 25) {
            for ((candidateArtist, tracks) in tracksByArtist) {
                if (candidateArtist.length < 25) continue
                val shorter = if (artist.length <= candidateArtist.length) artist else candidateArtist
                val longer = if (artist.length > candidateArtist.length) artist else candidateArtist
                if (shorter.length in 25..30 && longer.startsWith(shorter)) {
                    tracks.forEach { track ->
                        if (seen.add(track.track.id)) result.add(track)
                    }
                }
            }
        }

        return result
    }

    private fun fuzzyTitlePair(a: String, b: String): Boolean {
        val strippedA = a.replace(TRACK_NUMBER_PREFIX, "")
        val strippedB = b.replace(TRACK_NUMBER_PREFIX, "")
        if (a == b || strippedA == b || strippedB == a || strippedA == strippedB) return true
        if (a.length in 25..30 && b.length > a.length && b.startsWith(a)) return true
        if (b.length in 25..30 && a.length > b.length && a.startsWith(b)) return true
        val aNoExt = stripAudioExtension(a)
        if (aNoExt != a && aNoExt == b) return true
        val bNoExt = stripAudioExtension(b)
        if (bNoExt != b && bNoExt == a) return true
        return normalizeTitle(a) == normalizeTitle(b)
    }

    private fun durationCompatible(aMs: Int, bMs: Int, toleranceMs: Int = 5_000): Boolean {
        if (aMs <= 0 || bMs <= 0) return true
        return kotlin.math.abs(aMs - bMs) <= toleranceMs
    }

    private fun durationPenalty(aMs: Int, bMs: Int): Int {
        if (aMs <= 0 || bMs <= 0) return 0
        return kotlin.math.abs(aMs - bMs)
    }

    /** Categorize why a Poweramp entry didn't match any embedded key. */
    private fun categorizeFailure(
        entry: PowerampEntryWithPath,
        keysByArtist: Map<String, List<String>>,
    ): FailureReason {
        // Find keys with same artist (also check semicolon-split and unknown artist)
        val sameArtist = resolveArtistKeys(entry.artist, keysByArtist)
        if (sameArtist.isEmpty()) {
            return FailureReason.NO_SIMILAR_KEY
        }

        // Check if any key has same artist+title but different duration
        val titlePart = "|${entry.title}|"
        val hasTitleMatch = sameArtist.any { it.contains(titlePart) }
        if (hasTitleMatch) {
            return FailureReason.DURATION_ONLY
        }

        // Same artist exists but title doesn't match
        return FailureReason.TITLE_MISMATCH
    }

    /** Find the closest embedded key for an unmatched entry (for diagnostic detail). */
    private fun findClosestKey(
        entry: PowerampEntryWithPath,
        keysByArtist: Map<String, List<String>>,
    ): String? {
        // Try exact artist match first
        val sameArtist = keysByArtist[entry.artist]
        if (sameArtist != null) {
            // Find key with most similar title
            val titlePart = "|${entry.title}|"
            // Prefer keys containing the title
            val titleMatch = sameArtist.firstOrNull { it.contains(titlePart) }
            if (titleMatch != null) return titleMatch
            // Just return first key with same artist
            return sameArtist.firstOrNull()
        }

        // Try pipe-replaced artist (desktop replaces | with /)
        val pipeReplaced = entry.artist.replace("|", "/")
        if (pipeReplaced != entry.artist) {
            val pipeKeys = keysByArtist[pipeReplaced]
            if (pipeKeys != null) return pipeKeys.firstOrNull()
        }

        return null
    }

    /** Build artist → keys index from embedded keys. */
    private fun buildKeysByArtist(embeddedKeys: Set<String>): Map<String, List<String>> {
        val map = HashMap<String, MutableList<String>>(embeddedKeys.size / 10)
        for (key in embeddedKeys) {
            val pipeIdx = key.indexOf('|')
            val artist = if (pipeIdx >= 0) key.substring(0, pipeIdx) else key
            map.getOrPut(artist) { mutableListOf() }.add(key)
        }
        return map
    }

    /**
     * Resolve all DB keys that could belong to the same artist, accounting for:
     * - Exact artist match
     * - Semicolon/comma/slash-split primary artist
     *   (Poweramp: "a; b" or "a, b" or "a / b", DB: "a")
     * - Period normalization (PA: "o.c", DB: "o.c." — common in hip-hop)
     * - & ↔ and equivalence ("lq & the crew" ↔ "lq and the crew")
     * - Empty ↔ "unknown artist" equivalence
     * - ID3v1 30-char artist truncation (DB: "wanderwelle & bandhagens musik",
     *   phone: "wanderwelle & bandhagens musikförening")
     */
    private fun resolveArtistKeys(
        artist: String,
        keysByArtist: Map<String, List<String>>,
    ): List<String> {
        val result = mutableListOf<String>()
        val tried = mutableSetOf<String>()

        fun tryArtist(a: String) {
            if (a in tried) return
            tried.add(a)
            keysByArtist[a]?.let { result.addAll(it) }
        }

        // Exact match (including empty-artist tracks in the DB)
        tryArtist(artist)

        // Semicolon split (Poweramp: "artist1; artist2", DB: "artist1")
        if (';' in artist) {
            artist.split(';').forEach { tryArtist(it.trim()) }
        }

        // Comma split (Poweramp splits by "," too)
        if (',' in artist) {
            tryArtist(artist.substringBefore(',').trim())
        }

        // Slash split ("joe henderson / alice coltrane" → "joe henderson")
        if (" / " in artist) {
            artist.split(" / ").forEach { tryArtist(it.trim()) }
        }

        // Period normalization: "o.c" ↔ "o.c." — try with/without trailing period,
        // and with all periods stripped
        if ('.' in artist && artist.length > 1) {
            tryArtist(artist.trimEnd('.'))
            tryArtist(artist.replace(".", ""))
        } else if (artist.length in 1..5) {
            // Short name without period — try with trailing period (DB might have "o.c.")
            tryArtist("$artist.")
        }

        // & ↔ and normalization
        if (" & " in artist) {
            tryArtist(artist.replace(" & ", " and "))
        } else if (" and " in artist) {
            tryArtist(artist.replace(" and ", " & "))
        }

        // Empty ↔ "unknown artist"
        if (artist.isEmpty()) {
            tryArtist("unknown artist")
        }

        // ID3v1 artist truncation: prefix match when one side is 25-30 chars
        // (30-byte ID3v1 field, trailing spaces stripped by mutagen → 25-30 chars)
        if (artist.length >= 25) {
            for ((dbArtist, keys) in keysByArtist) {
                if (dbArtist in tried || dbArtist.length < 25) continue
                val shorter = if (artist.length <= dbArtist.length) artist else dbArtist
                val longer = if (artist.length > dbArtist.length) artist else dbArtist
                if (shorter.length in 25..30 && longer.startsWith(shorter)) {
                    result.addAll(keys)
                    tried.add(dbArtist)
                }
            }
        }
        return result
    }

    /**
     * Fuzzy title matching for edge cases that exact/partial strategies miss:
     * - Track number prefix (DB: "18 song", phone: "song" — from filename-as-title)
     * - ID3v1 truncation (DB title exactly 30 chars is a prefix of phone title)
     * - Audio extension in DB title (DB: "welcome.wav", phone: "welcome")
     * - Empty ↔ "unknown artist" equivalence
     *
     * Only runs for tracks that fail all faster strategies, so iterating same-artist
     * keys (typically a few dozen) is acceptable.
     */
    private fun fuzzyMatchesAny(
        entry: PowerampEntryWithPath,
        keysByArtist: Map<String, List<String>>,
    ): Boolean {
        val sameArtistKeys = resolveArtistKeys(entry.artist, keysByArtist)

        val phoneTitle = entry.title
        val phoneTitleStripped = phoneTitle.replace(TRACK_NUMBER_PREFIX, "")

        if (sameArtistKeys.isNotEmpty() && fuzzyTitleMatch(phoneTitle, phoneTitleStripped, sameArtistKeys)) {
            return true
        }

        // Empty artist on phone but DB has a real artist: search by title+album across all keys.
        // Only for non-generic titles (>5 chars) to avoid false positives on "intro", "untitled", etc.
        if (entry.artist.isEmpty() && phoneTitle.length > 5 && entry.album != "unknown album") {
            val albumPart = "|${entry.album}|"
            val titlePart = "|$phoneTitle|"
            for ((_, keys) in keysByArtist) {
                for (key in keys) {
                    if (key.contains(albumPart) && key.contains(titlePart)) return true
                }
            }
        }

        return false
    }

    /** Check if any key in the list has a title that fuzzy-matches the phone title. */
    private fun fuzzyTitleMatch(
        phoneTitle: String,
        phoneTitleStripped: String,
        keys: List<String>,
    ): Boolean {
        // Pre-compute normalized phone title for the expensive comparisons
        val phoneNorm = normalizeTitle(phoneTitle)

        for (key in keys) {
            val dbTitle = extractTitleFromKey(key)

            // Track number prefix: DB has "18 title", phone has "title" (or vice versa)
            val dbTitleStripped = dbTitle.replace(TRACK_NUMBER_PREFIX, "")
            if (dbTitleStripped == phoneTitle) return true
            if (phoneTitleStripped == dbTitle) return true
            if (dbTitleStripped == phoneTitleStripped && dbTitleStripped != dbTitle) return true

            // ID3v1 truncation: field in 25-30 chars is prefix of the other
            // (30-byte ID3v1 field, trailing spaces stripped by mutagen → 25-30 chars)
            if (dbTitle.length in 25..30 && phoneTitle.length > dbTitle.length && phoneTitle.startsWith(dbTitle)) return true
            if (phoneTitle.length in 25..30 && dbTitle.length > phoneTitle.length && dbTitle.startsWith(phoneTitle)) return true

            // Audio extension in DB title: DB has "welcome.wav", phone has "welcome"
            val dbTitleNoExt = stripAudioExtension(dbTitle)
            if (dbTitleNoExt != dbTitle && dbTitleNoExt == phoneTitle) return true

            // Normalized comparison: & ↔ and, strip special chars, collapse whitespace.
            // Catches "turiya & ramakrishna" = "turiya and ramakrishna",
            // "$kurrency$" = "kurrency", "ghost land" ≈ "ghostland", etc.
            // Only compare if titles are similar length (within 20%) to avoid false matches.
            val dbNorm = normalizeTitle(dbTitle)
            if (phoneNorm == dbNorm && phoneNorm.length >= 3) return true
        }
        return false
    }

    /**
     * Normalize title for fuzzy comparison: & → and, strip non-alphanumeric
     * (except spaces), collapse whitespace. Preserves parenthetical content
     * to avoid matching "instrumental" vs "vocal" as the same track.
     */
    private fun normalizeTitle(title: String): String {
        return title
            .replace(" & ", " and ")
            .replace("\u2018", "'")   // left smart quote → ASCII
            .replace("\u2019", "'")   // right smart quote → ASCII
            .replace(NON_ALPHANUM, "")
            .replace(MULTI_SPACE, " ")
            .trim()
    }

    /** Extract the title field (3rd pipe-delimited segment) from a metadata key. */
    private fun extractTitleFromKey(key: String): String {
        var start = -1
        var count = 0
        for (i in key.indices) {
            if (key[i] == '|') {
                count++
                if (count == 2) start = i + 1
                if (count == 3) return key.substring(start, i)
            }
        }
        return ""
    }

    /**
     * Get all Poweramp file entries including file paths.
     * Extends PowerampHelper.getAllFileEntries with the path column.
     */
    private fun getAllPowerampEntriesWithPaths(context: Context): List<PowerampEntryWithPath> {
        return PowerampHelper.getAllFileEntries(context)
    }

    private fun EmbeddedTrack.asComparableEntry(): ComparableEmbeddedTrack {
        val normalizedArtist = TrackNormalization.normalizeArtist(artist)
        val normalizedAlbum = TrackNormalization.normalizeAlbum(album)
        val normalizedTitle = TrackNormalization.normalizeTitle(title)
        val filenameKeys = TrackNormalization.buildFilenameKeys(normalizedArtist, normalizedTitle, filenameKey)

        return ComparableEmbeddedTrack(
            track = this,
            artist = normalizedArtist,
            album = normalizedAlbum,
            title = normalizedTitle,
            durationMs = durationMs,
            path = TrackNormalization.normalizePath(filePath),
            metadataKey = TrackNormalization.buildMetadataKey(normalizedArtist, normalizedAlbum, normalizedTitle, durationMs),
            filenameKeys = filenameKeys,
        )
    }

    private fun collectMusicRoots(entries: List<PowerampEntryWithPath>): List<String> {
        val roots = linkedSetOf<String>()
        for (entry in entries) {
            val path = entry.path ?: continue
            val normalized = path.replace('\\', '/')
            val idx = normalized.lowercase().indexOf("/music/")
            if (idx >= 0) {
                roots += normalized.substring(0, idx + "/music".length)
            }
        }
        return roots.toList()
    }

    private fun existsOnDeviceStorage(track: EmbeddedTrack, musicRoots: List<String>): Boolean {
        val exactPath = TrackNormalization.normalizePath(track.filePath)
        if (!exactPath.isNullOrBlank() && exactPath.startsWith("/storage/") && File(exactPath).exists()) {
            return true
        }

        val relative = extractMusicRelativePath(track.filePath) ?: return false
        for (root in musicRoots) {
            val candidate = "$root/$relative"
            if (File(candidate).exists()) return true
        }
        return false
    }

    private fun extractMusicRelativePath(path: String?): String? {
        val raw = path?.takeIf { it.isNotBlank() } ?: return null
        val normalized = raw.replace('\\', '/').trim()
        val lower = normalized.lowercase()
        val idx = lower.indexOf("/music/")
        if (idx >= 0) {
            return normalized.substring(idx + "/music/".length)
        }
        return normalized.substringAfterLast("/").takeIf { it.isNotBlank() }
    }

    private fun normalizeAsFilename(s: String): String {
        return TrackNormalization.normalizeAsFilename(s)
    }

    private fun normalizeNfc(s: String): String =
        TrackNormalization.normalizeNfc(s)

    /** Replace pipe with / to match desktop indexer key format. */
    private fun sanitizePipe(s: String): String =
        TrackNormalization.sanitizePipe(s)

    private fun normalizePowerampArtist(artist: String): String =
        TrackNormalization.normalizeArtist(artist)

    private fun stripAudioExtension(title: String): String {
        return TrackNormalization.stripAudioExtension(title)
    }

    private val AUDIO_EXTENSIONS = TrackNormalization.audioExtensions
}
