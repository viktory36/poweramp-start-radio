package com.powerampstartradio.data

internal const val UNKNOWN_ARTIST_CREDIT_ID = -1

/** The exact artist-credit equality used by the user-facing queue constraint. */
internal fun normalizeArtistCredit(value: String?): String? =
    value?.trim()?.takeIf { it.isNotEmpty() }?.lowercase()

/**
 * Compact artist-credit equality IDs aligned to the immutable embedding generation.
 *
 * The numeric IDs have no ranking meaning. They replace repeated metadata reads and retained
 * [EmbeddedTrack] objects while preserving the existing artist equality contract exactly.
 */
internal class ArtistCreditCatalog(
    private val orderedTrackIds: LongArray,
    private val creditIdByTrackPosition: IntArray,
    val distinctCreditCount: Int,
) {
    init {
        require(orderedTrackIds.size == creditIdByTrackPosition.size) {
            "Artist-credit rows are not aligned to track IDs"
        }
        require(orderedTrackIds.indices.drop(1).all { position ->
            orderedTrackIds[position] > orderedTrackIds[position - 1]
        }) { "Artist-credit track IDs must be strictly increasing" }
        require(creditIdByTrackPosition.all { it == UNKNOWN_ARTIST_CREDIT_ID || it in 0 until distinctCreditCount }) {
            "Artist-credit ID is outside the catalog"
        }
    }

    fun creditId(trackId: Long): Int {
        val position = orderedTrackIds.binarySearch(trackId)
        require(position >= 0) { "Track $trackId is absent from the artist-credit catalog" }
        return creditIdByTrackPosition[position]
    }
}
