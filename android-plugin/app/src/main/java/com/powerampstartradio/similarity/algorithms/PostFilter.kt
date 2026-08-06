package com.powerampstartradio.similarity.algorithms

import com.powerampstartradio.data.EmbeddedTrack
import com.powerampstartradio.data.ArtistCreditCatalog
import com.powerampstartradio.data.UNKNOWN_ARTIST_CREDIT_ID
import com.powerampstartradio.data.normalizeArtistCredit
import com.powerampstartradio.similarity.SimilarTrack

/**
 * Artist-credit constraint helpers.
 *
 * Enforces:
 * - Maximum N tracks per artist in the final queue
 * - Minimum M track spacing between songs by the same artist
 */
object PostFilter {

    /** Prepare the same artist constraint for repeated objective scans without metadata I/O. */
    internal fun prepare(
        catalog: ArtistCreditCatalog,
        maxPerArtist: Int,
        minSpacing: Int,
    ): PreparedArtistConstraint = PreparedArtistConstraint(
        catalog = catalog,
        maxPerArtist = maxPerArtist,
        minSpacing = minSpacing,
    )

    /**
     * Check if adding a track would violate artist constraints.
     *
     * @param track Candidate track to add
     * @param currentQueue Tracks already in the queue (in order)
     * @param maxPerArtist Maximum tracks per artist
     * @param minSpacing Minimum positions between same-artist tracks
     * @return true if the track can be added without violating constraints
     */
    fun canAdd(
        track: EmbeddedTrack,
        currentQueue: List<EmbeddedTrack>,
        maxPerArtist: Int,
        minSpacing: Int
    ): Boolean {
        val artist = normalizeArtistCredit(track.artist) ?: return true

        // Check total count for this artist
        val artistCount = currentQueue.count { normalizeArtistCredit(it.artist) == artist }
        if (artistCount >= maxPerArtist) return false

        // Check spacing: look at the last minSpacing entries
        if (minSpacing > 0 && currentQueue.isNotEmpty()) {
            val recentWindow = currentQueue.takeLast(minSpacing)
            if (recentWindow.any { normalizeArtistCredit(it.artist) == artist }) return false
        }

        return true
    }

    /**
     * Filter a batch of selected tracks to enforce artist constraints.
     * Preserves order, dropping tracks that violate constraints.
     *
     * @param tracks Ordered list of SimilarTrack
     * @param maxPerArtist Maximum tracks per artist
     * @param minSpacing Minimum positions between same-artist tracks
     * @return Filtered list preserving order
     */
    fun enforceBatch(
        tracks: List<SimilarTrack>,
        maxPerArtist: Int,
        minSpacing: Int
    ): List<SimilarTrack> {
        val result = mutableListOf<SimilarTrack>()
        val artistCounts = mutableMapOf<String, Int>()

        for (st in tracks) {
            val artist = normalizeArtistCredit(st.track.artist)

            // Check max per artist
            if (artist != null) {
                val count = artistCounts.getOrDefault(artist, 0)
                if (count >= maxPerArtist) continue
            }

            // Check spacing
            if (artist != null && minSpacing > 0 && result.isNotEmpty()) {
                val recentWindow = result.takeLast(minSpacing)
                if (recentWindow.any { normalizeArtistCredit(it.track.artist) == artist }) continue
            }

            result.add(st)
            if (artist != null) {
                artistCounts[artist] = (artistCounts[artist] ?: 0) + 1
            }
        }

        return result
    }

}

/** Mutable state scoped to one selector invocation; selected lists are append-only per selector. */
internal class PreparedArtistConstraint(
    private val catalog: ArtistCreditCatalog,
    private val maxPerArtist: Int,
    private val minSpacing: Int,
) {
    private val counts = IntArray(catalog.distinctCreditCount)
    private val lastPosition = IntArray(catalog.distinctCreditCount) { NO_POSITION }
    private var selectedReference: List<Long>? = null
    private val selectedSnapshot = ArrayList<Long>()

    fun canAdd(candidateTrackId: Long, selectedTrackIds: List<Long>): Boolean {
        synchronize(selectedTrackIds)
        val creditId = catalog.creditId(candidateTrackId)
        if (creditId == UNKNOWN_ARTIST_CREDIT_ID) return true
        if (counts[creditId] >= maxPerArtist) return false
        val previousPosition = lastPosition[creditId]
        return minSpacing <= 0 || previousPosition == NO_POSITION ||
            selectedTrackIds.size - previousPosition > minSpacing
    }

    private fun synchronize(selectedTrackIds: List<Long>) {
        if (selectedTrackIds === selectedReference &&
            selectedTrackIds.size == selectedSnapshot.size
        ) {
            return
        }
        val sameAppendOnlyList = selectedTrackIds === selectedReference &&
            selectedTrackIds.size >= selectedSnapshot.size &&
            selectedSnapshot.indices.all { position ->
                selectedSnapshot[position] == selectedTrackIds[position]
            }
        if (!sameAppendOnlyList) {
            counts.fill(0)
            lastPosition.fill(NO_POSITION)
            selectedSnapshot.clear()
            selectedReference = selectedTrackIds
        }

        for (position in selectedSnapshot.size until selectedTrackIds.size) {
            val trackId = selectedTrackIds[position]
            selectedSnapshot += trackId
            val creditId = catalog.creditId(trackId)
            if (creditId != UNKNOWN_ARTIST_CREDIT_ID) {
                counts[creditId]++
                lastPosition[creditId] = position
            }
        }
    }

    private companion object {
        const val NO_POSITION = -1
    }
}
