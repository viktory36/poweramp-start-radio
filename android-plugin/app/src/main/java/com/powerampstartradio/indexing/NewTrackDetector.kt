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
        private val TRACK_NUMBER_PREFIX = Regex("^\\d+[.\\-\\s]+")
    }

    /** Categorized reason why a track didn't match any embedded key. */
    enum class FailureReason {
        DURATION_ONLY,       // Same artist+album+title, duration differs
        PIPE_IN_METADATA,    // Track has | in artist/album/title (desktop replaces with /)
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
        val embeddedKeyCount: Int,
        val embeddedPathCount: Int,
        val embeddedPathsSample: List<String>,
        val exactKeyMatches: Int,
        val partialMatches: Int,
        val pathMatches: Int,
        val unmatchedCount: Int,
        val unmatchedSample: List<UnmatchedDetail>,
        val failureCategories: Map<FailureReason, Int>,
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
    fun findUnindexedTracks(context: Context): List<UnindexedTrack> {
        // Get all Poweramp entries with paths
        val powerampEntries = getAllPowerampEntriesWithPaths(context)
        if (powerampEntries.isEmpty()) {
            Log.w(TAG, "No entries from Poweramp library")
            return emptyList()
        }

        // Get all embedded track metadata keys and file paths
        // NFC-normalize loaded keys — some DB entries have NFD despite desktop claiming NFC
        val rawKeys = embeddingDb.getAllMetadataKeys()
        val embeddedKeys = rawKeys.mapTo(HashSet<String>(rawKeys.size)) { normalizeNfc(it) }
        val embeddedPaths = embeddingDb.getAllFilePaths()

        Log.d(TAG, "Poweramp: ${powerampEntries.size} tracks, " +
            "Embedded: ${embeddedKeys.size} metadata keys, ${embeddedPaths.size} paths")

        // Build artist → keys index for fuzzy matching
        val keysByArtist = buildKeysByArtist(embeddedKeys)

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

            // Check semicolon artist split (Poweramp: "artist1; artist2", DB: "artist1")
            if (';' in entry.artist) {
                val primaryArtist = entry.artist.substringBefore(';').trim()
                val primaryPrefix = "$primaryArtist|"
                val hasPrimaryMatch = embeddedKeys.any { key ->
                    key.startsWith(primaryPrefix) && key.contains(titlePart)
                }
                if (hasPrimaryMatch) continue
            }

            // Check fuzzy title matching (track number prefix, ID3v1 truncation, extensions,
            // empty/"unknown artist" equivalence)
            if (fuzzyMatchesAny(entry, keysByArtist)) continue

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
        onProgress("Querying Poweramp library...")
        val powerampEntries = getAllPowerampEntriesWithPaths(context)
        if (powerampEntries.isEmpty()) {
            Log.w(TAG, "No entries from Poweramp library")
            return DiagnosticResult(0, 0, 0, emptyList(), 0, 0, 0, 0, emptyList(), emptyMap())
        }

        onProgress("Loading embedded keys (${powerampEntries.size} Poweramp tracks)...")
        // NFC-normalize loaded keys — some DB entries have NFD despite desktop claiming NFC
        val rawKeys = embeddingDb.getAllMetadataKeys()
        val embeddedKeys = rawKeys.mapTo(HashSet<String>(rawKeys.size)) { normalizeNfc(it) }
        val embeddedPaths = embeddingDb.getAllFilePaths()

        Log.d(TAG, "Poweramp: ${powerampEntries.size}, Embedded: ${embeddedKeys.size} keys, ${embeddedPaths.size} paths")
        onProgress("Matching ${powerampEntries.size} Poweramp tracks against ${embeddedKeys.size} embedded keys...")

        // Build index for faster closest-key lookup: artist prefix → list of keys
        val keysByArtist = HashMap<String, MutableList<String>>(embeddedKeys.size / 10)
        for (key in embeddedKeys) {
            val pipeIdx = key.indexOf('|')
            val artist = if (pipeIdx >= 0) key.substring(0, pipeIdx) else key
            keysByArtist.getOrPut(artist) { mutableListOf() }.add(key)
        }

        var exactKeyMatches = 0
        var partialMatches = 0
        var pathMatches = 0
        val unmatched = mutableListOf<UnmatchedDetail>()
        val failureCounts = mutableMapOf<FailureReason, Int>()

        for ((i, entry) in powerampEntries.withIndex()) {
            if (i % 10000 == 0 && i > 0) {
                onProgress("Processing $i / ${powerampEntries.size}...")
            }

            // Strategy 1: Exact metadata key match
            if (entry.metadataKey in embeddedKeys) {
                exactKeyMatches++
                continue
            }

            // Strategy 2: Partial match (artist+title, ignoring album+duration)
            val artistTitlePrefix = "${entry.artist}|"
            val titlePart = "|${entry.title}|"
            val hasPartialMatch = embeddedKeys.any { key ->
                key.startsWith(artistTitlePrefix) && key.contains(titlePart)
            }
            if (hasPartialMatch) {
                partialMatches++
                continue
            }

            // Strategy 2b: Semicolon artist split (Poweramp: "artist1; artist2", DB: "artist1")
            if (';' in entry.artist) {
                val primaryArtist = entry.artist.substringBefore(';').trim()
                val primaryPrefix = "$primaryArtist|"
                val hasPrimaryMatch = embeddedKeys.any { key ->
                    key.startsWith(primaryPrefix) && key.contains(titlePart)
                }
                if (hasPrimaryMatch) {
                    partialMatches++
                    continue
                }
            }

            // Strategy 3: Fuzzy title matching (track number prefix, ID3v1 truncation,
            // extensions, empty/"unknown artist" equivalence)
            if (fuzzyMatchesAny(entry, keysByArtist)) {
                partialMatches++
                continue
            }

            // Strategy 4: File path match
            if (entry.path != null && entry.path in embeddedPaths) {
                pathMatches++
                continue
            }

            // Unmatched — categorize failure
            val reason = categorizeFailure(entry, keysByArtist)
            failureCounts[reason] = (failureCounts[reason] ?: 0) + 1

            // Keep all unmatched for analysis (closest key only for first 200)
            val closest = if (unmatched.size < 200) findClosestKey(entry, keysByArtist) else null
            unmatched.add(UnmatchedDetail(
                artist = entry.artist,
                album = entry.album,
                title = entry.title,
                durationMs = entry.durationMs,
                powerampKey = entry.metadataKey,
                closestEmbeddedKey = closest,
                failureReason = reason,
                path = entry.path,
            ))
        }

        val result = DiagnosticResult(
            powerampCount = powerampEntries.size,
            embeddedKeyCount = embeddedKeys.size,
            embeddedPathCount = embeddedPaths.size,
            embeddedPathsSample = embeddedPaths.take(5).toList(),
            exactKeyMatches = exactKeyMatches,
            partialMatches = partialMatches,
            pathMatches = pathMatches,
            unmatchedCount = failureCounts.values.sum(),
            unmatchedSample = unmatched,
            failureCategories = failureCounts,
        )

        Log.i(TAG, "Diagnostic: exact=$exactKeyMatches, partial=$partialMatches, " +
            "path=$pathMatches, unmatched=${failureCounts.values.sum()}")
        Log.i(TAG, "Failure categories: $failureCounts")

        return result
    }

    /** Categorize why a Poweramp entry didn't match any embedded key. */
    private fun categorizeFailure(
        entry: PowerampEntryWithPath,
        keysByArtist: Map<String, List<String>>,
    ): FailureReason {
        // Check for pipe in raw metadata (desktop replaces | with /)
        if ('|' in (entry.rawArtist ?: "") || '|' in (entry.rawAlbum ?: "") || '|' in (entry.rawTitle ?: "")) {
            return FailureReason.PIPE_IN_METADATA
        }

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
     * - Semicolon-split primary artist (Poweramp: "a; b", DB: "a")
     * - Empty ↔ "unknown artist" equivalence
     * - ID3v1 30-char artist truncation (DB: "wanderwelle & bandhagens musik",
     *   phone: "wanderwelle & bandhagens musikförening")
     */
    private fun resolveArtistKeys(
        artist: String,
        keysByArtist: Map<String, List<String>>,
    ): List<String> {
        val result = mutableListOf<String>()
        keysByArtist[artist]?.let { result.addAll(it) }
        // Semicolon split
        if (';' in artist) {
            val primary = artist.substringBefore(';').trim()
            if (primary != artist) keysByArtist[primary]?.let { result.addAll(it) }
        }
        // Empty ↔ "unknown artist"
        if (artist.isEmpty()) {
            keysByArtist["unknown artist"]?.let { result.addAll(it) }
        }
        // ID3v1 30-char artist truncation: phone has full name, DB has truncated
        if (artist.length > 30) {
            val truncated = artist.substring(0, 30)
            keysByArtist[truncated]?.let { result.addAll(it) }
        }
        // Reverse: DB has full name, phone has truncated artist
        if (artist.length == 30) {
            for ((dbArtist, keys) in keysByArtist) {
                if (dbArtist.length > 30 && dbArtist.startsWith(artist)) {
                    result.addAll(keys)
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
        for (key in keys) {
            val dbTitle = extractTitleFromKey(key)

            // Track number prefix: DB has "18 title", phone has "title" (or vice versa)
            val dbTitleStripped = dbTitle.replace(TRACK_NUMBER_PREFIX, "")
            if (dbTitleStripped == phoneTitle) return true
            if (phoneTitleStripped == dbTitle) return true
            if (dbTitleStripped == phoneTitleStripped && dbTitleStripped != dbTitle) return true

            // ID3v1 truncation: DB field exactly 30 chars is prefix of phone title
            if (dbTitle.length == 30 && phoneTitle.length > 30 && phoneTitle.startsWith(dbTitle)) return true
            if (phoneTitle.length == 30 && dbTitle.length > 30 && dbTitle.startsWith(phoneTitle)) return true

            // Audio extension in DB title: DB has "welcome.wav", phone has "welcome"
            val dbTitleNoExt = stripAudioExtension(dbTitle)
            if (dbTitleNoExt != dbTitle && dbTitleNoExt == phoneTitle) return true
        }
        return false
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
        val filesUri = PowerampHelper.ROOT_URI.buildUpon()
            .appendEncodedPath("files").build()
        val result = mutableListOf<PowerampEntryWithPath>()

        // "path" is from the joined folders table (has trailing /),
        // "folder_files.name" is the short filename.
        // Note: "folder_path"/"file_name" are PlaylistEntries columns, not folder_files!
        val cursor = try {
            context.contentResolver.query(
                filesUri,
                arrayOf(
                    "folder_files._id", "artist", "album", "title_tag",
                    "folder_files.duration", "path", "folder_files.name"
                ),
                null, null, null
            )
        } catch (e: Exception) {
            Log.w(TAG, "path+name columns not available, querying without paths")
            context.contentResolver.query(
                filesUri,
                arrayOf(
                    "folder_files._id", "artist", "album", "title_tag",
                    "folder_files.duration"
                ),
                null, null, null
            )
        }

        try {
            cursor?.use {
                val idIdx = it.getColumnIndex("_id")
                val artistIdx = it.getColumnIndex("artist")
                val albumIdx = it.getColumnIndex("album")
                val titleIdx = it.getColumnIndex("title_tag")
                val durationIdx = it.getColumnIndex("duration")
                val pathIdx = it.getColumnIndex("path")       // folders.path (trailing /)
                val nameIdx = it.getColumnIndex("name")       // folder_files.name

                while (it.moveToNext()) {
                    val lcArtist = (it.getString(artistIdx) ?: "").lowercase().trim()
                    val lcTitle = (it.getString(titleIdx) ?: "").lowercase().trim()
                    val lcAlbum = (it.getString(albumIdx) ?: "").lowercase().trim()
                    val artist = normalizeNfc(normalizePowerampArtist(lcArtist))
                    val album = normalizeNfc(lcAlbum)
                    val title = normalizeNfc(stripAudioExtension(lcTitle))

                    val path = when {
                        pathIdx >= 0 && nameIdx >= 0 -> {
                            val folder = it.getString(pathIdx) ?: ""  // already has trailing /
                            val name = it.getString(nameIdx) ?: ""
                            if (name.isNotEmpty()) "$folder$name" else null
                        }
                        else -> null
                    }

                    val durationRounded = (it.getInt(durationIdx) / 100) * 100

                    result.add(PowerampEntryWithPath(
                        id = it.getLong(idIdx),
                        artist = artist,
                        album = album,
                        title = title,
                        durationMs = it.getInt(durationIdx),
                        path = path,
                        metadataKey = "$artist|$album|$title|$durationRounded",
                        rawArtist = lcArtist,
                        rawAlbum = lcAlbum,
                        rawTitle = lcTitle,
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
        val rawArtist: String? = null,
        val rawAlbum: String? = null,
        val rawTitle: String? = null,
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
