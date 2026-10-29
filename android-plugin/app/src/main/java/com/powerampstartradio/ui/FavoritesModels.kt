package com.powerampstartradio.ui

import com.powerampstartradio.data.EmbeddedTrack
import com.powerampstartradio.data.StableIdentityGenerationBinding

data class FavoriteMatch(
    val track: EmbeddedTrack,
    val rank: Int,
    val isNew: Boolean,
    val stableTrackSpanId: String? = null,
)

enum class FavoriteMatchesSort(val label: String) {
    CLOSEST_FIRST("Nearest first"), FARTHEST_FIRST("Farthest first"),
    NEWEST_FIRST("Newest first"), OLDEST_FIRST("Oldest first");

    val byAddedDate: Boolean get() = this == NEWEST_FIRST || this == OLDEST_FIRST
    val ascending: Boolean get() = this == CLOSEST_FIRST || this == OLDEST_FIRST

    fun toggleCloseness(): FavoriteMatchesSort =
        if (this == CLOSEST_FIRST) FARTHEST_FIRST else CLOSEST_FIRST

    fun toggleAddedDate(): FavoriteMatchesSort =
        if (this == NEWEST_FIRST) OLDEST_FIRST else NEWEST_FIRST
}

internal fun sortFavoriteMatches(
    matches: List<FavoriteMatch>,
    sort: FavoriteMatchesSort,
): List<FavoriteMatch> {
    val byRank = compareBy<FavoriteMatch> { it.rank }.thenBy { it.track.id }
    val comparator = when (sort) {
        FavoriteMatchesSort.CLOSEST_FIRST -> byRank
        FavoriteMatchesSort.FARTHEST_FIRST -> compareByDescending<FavoriteMatch> { it.rank }
            .thenBy { it.track.id }
        FavoriteMatchesSort.NEWEST_FIRST -> compareBy<FavoriteMatch> { it.track.indexAddedAtEpochSecond <= 0L }
            .thenByDescending { it.track.indexAddedAtEpochSecond }.then(byRank)
        FavoriteMatchesSort.OLDEST_FIRST -> compareBy<FavoriteMatch> { it.track.indexAddedAtEpochSecond <= 0L }
            .thenBy { it.track.indexAddedAtEpochSecond }.then(byRank)
    }
    return matches.sortedWith(comparator)
}

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
    val resolvedSeedTrack: EmbeddedTrack? = null,
    val resolvedSeedStableTrackSpanId: String? = null,
    val binding: FavoriteResultsBinding? = null,
)

data class FavoriteResultsBinding(
    val library: StableIdentityGenerationBinding,
    val providerGenerationId: String,
    val orderedActiveTrackIdsSha256: String,
    val activeTrackCount: Int,
    val rankingIdentityCount: Int,
)

/** Original cosine ranks remain aligned with the displayed order; the prepended seed has no rank. */
data class FavoriteQueueEvidence(
    val seed: FavoriteSeed,
    val sort: FavoriteMatchesSort,
    val seedIncluded: Boolean,
    val orderedClosestRanks: List<Int?>,
    val rankingIdentityCount: Int,
)

internal data class FavoriteDisplayedQueue(
    val tracks: List<EmbeddedTrack>,
    val identities: List<RadioSeedIdentity>,
    val evidence: FavoriteQueueEvidence,
)

internal fun favoriteDisplayedQueue(state: FavoriteMatchesState, sort: FavoriteMatchesSort): FavoriteDisplayedQueue {
    check(!state.loading && state.error == null) { "Wait for the favorite results to finish loading." }
    val binding = checkNotNull(state.binding) { "Open the favorite again for fresh results." }
    val seed = checkNotNull(state.seed)
    val sorted = sortFavoriteMatches(state.matches, sort)
    val seedTrack = state.resolvedSeedTrack
    val canonicalSeed = seedTrack?.let {
        FavoriteSeed("song:${it.id}", trackId = it.id,
            stableTrackSpanId = state.resolvedSeedStableTrackSpanId, title = it.title, artist = it.artist)
    } ?: seed
    val tracks = listOfNotNull(seedTrack) + sorted.map { it.track }
    check(tracks.isNotEmpty()) { "There are no recordings to queue." }
    check(tracks.map { it.id }.distinct().size == tracks.size) { "The displayed list repeats a recording." }
    return FavoriteDisplayedQueue(tracks,
        listOfNotNull(seedTrack?.let { RadioSeedIdentity(it.id, state.resolvedSeedStableTrackSpanId) }) +
            sorted.map { RadioSeedIdentity(it.track.id, it.stableTrackSpanId) },
        FavoriteQueueEvidence(canonicalSeed, sort, seedTrack != null,
            (if (seedTrack != null) listOf(null) else emptyList()) + sorted.map { it.rank },
            binding.rankingIdentityCount))
}
