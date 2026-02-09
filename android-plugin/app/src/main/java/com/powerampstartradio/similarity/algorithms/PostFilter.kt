package com.powerampstartradio.similarity.algorithms

import com.powerampstartradio.data.EmbeddedTrack
import com.powerampstartradio.similarity.SimilarTrack

/**
 * Post-selection filter for artist/album diversity constraints.
 *
 * Enforces:
 * - Maximum N tracks per artist in the final queue
 * - Minimum M track spacing between songs by the same artist
 */
object PostFilter {

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
        val artist = track.artist?.lowercase() ?: return true  // Unknown artist: always OK

        // Check total count for this artist
        val artistCount = currentQueue.count { it.artist?.lowercase() == artist }
        if (artistCount >= maxPerArtist) return false

        // Check spacing: look at the last minSpacing entries
        if (minSpacing > 0 && currentQueue.isNotEmpty()) {
            val recentWindow = currentQueue.takeLast(minSpacing)
            if (recentWindow.any { it.artist?.lowercase() == artist }) return false
        }

        return true
    }

    /**
     * Result of batch filtering, including statistics about what was dropped.
     */
    data class FilterResult(
        val tracks: List<SimilarTrack>,
        val dropCount: Int,
        val dropReasons: Map<String, Int>,  // artist -> count dropped
    )

    /**
     * Filter a batch of selected tracks to enforce artist constraints.
     * Preserves order, dropping tracks that violate constraints.
     * Returns both filtered list and drop statistics.
     *
     * @param tracks Ordered list of SimilarTrack
     * @param maxPerArtist Maximum tracks per artist
     * @param minSpacing Minimum positions between same-artist tracks
     * @return FilterResult with filtered list and drop stats
     */
    fun enforceBatch(
        tracks: List<SimilarTrack>,
        maxPerArtist: Int,
        minSpacing: Int
    ): FilterResult {
        val result = mutableListOf<SimilarTrack>()
        val artistCounts = mutableMapOf<String, Int>()
        val dropReasons = mutableMapOf<String, Int>()
        var dropCount = 0

        for (st in tracks) {
            val artist = st.track.artist?.lowercase()

            // Check max per artist
            if (artist != null) {
                val count = artistCounts.getOrDefault(artist, 0)
                if (count >= maxPerArtist) {
                    dropCount++
                    dropReasons[artist] = (dropReasons[artist] ?: 0) + 1
                    continue
                }
            }

            // Check spacing
            if (artist != null && minSpacing > 0 && result.isNotEmpty()) {
                val recentWindow = result.takeLast(minSpacing)
                if (recentWindow.any { it.track.artist?.lowercase() == artist }) {
                    dropCount++
                    dropReasons[artist] = (dropReasons[artist] ?: 0) + 1
                    continue
                }
            }

            result.add(st)
            if (artist != null) {
                artistCounts[artist] = (artistCounts[artist] ?: 0) + 1
            }
        }

        return FilterResult(result, dropCount, dropReasons)
    }
}
