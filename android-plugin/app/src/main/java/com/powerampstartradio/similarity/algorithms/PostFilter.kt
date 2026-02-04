package com.powerampstartradio.similarity.algorithms

import com.powerampstartradio.data.EmbeddedTrack

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
     * Filter a batch of selected tracks to enforce artist constraints.
     * Preserves order, dropping tracks that violate constraints.
     *
     * @param tracks Ordered list of (track, similarity) pairs
     * @param maxPerArtist Maximum tracks per artist
     * @param minSpacing Minimum positions between same-artist tracks
     * @return Filtered list preserving order
     */
    fun enforceBatch(
        tracks: List<Pair<EmbeddedTrack, Float>>,
        maxPerArtist: Int,
        minSpacing: Int
    ): List<Pair<EmbeddedTrack, Float>> {
        val result = mutableListOf<Pair<EmbeddedTrack, Float>>()
        val artistCounts = mutableMapOf<String, Int>()

        for ((track, score) in tracks) {
            val artist = track.artist?.lowercase()

            // Check max per artist
            if (artist != null) {
                val count = artistCounts.getOrDefault(artist, 0)
                if (count >= maxPerArtist) continue
            }

            // Check spacing
            if (artist != null && minSpacing > 0 && result.isNotEmpty()) {
                val recentWindow = result.takeLast(minSpacing)
                if (recentWindow.any { it.first.artist?.lowercase() == artist }) continue
            }

            result.add(track to score)
            if (artist != null) {
                artistCounts[artist] = (artistCounts[artist] ?: 0) + 1
            }
        }

        return result
    }
}
