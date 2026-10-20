package com.powerampstartradio.ui

import com.powerampstartradio.data.EmbeddedTrack

data class FavoriteMatch(
    val track: EmbeddedTrack,
    val rank: Int,
    val isNew: Boolean,
)

internal fun indexAddedAgeLabel(addedAtEpochSecond: Long, nowEpochSecond: Long): String? {
    if (addedAtEpochSecond <= 0L) return null
    val days = (nowEpochSecond - addedAtEpochSecond).coerceAtLeast(0L) / 86_400L
    return if (days == 0L) "added today" else "added ${days}d ago"
}

data class FavoriteMatchesState(
    val seed: FavoriteSeed? = null,
    val matches: List<FavoriteMatch> = emptyList(),
    val generationId: String? = null,
    val loading: Boolean = false,
    val status: String? = null,
    val error: String? = null,
    val resolvedSeedTrackId: Long? = null,
)
