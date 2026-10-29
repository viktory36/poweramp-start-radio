package com.powerampstartradio.ui

import com.powerampstartradio.data.EmbeddedTrack
import org.junit.Assert.*
import org.junit.Test

class FavoriteMatchesSortTest {
    @Test
    fun `sorting retains the nearest set and its ranks and badges`() {
        val matches = listOf(match(1, 100), match(2, 300), match(3, 300), match(4, 0))
        val expected = mapOf(
            FavoriteMatchesSort.CLOSEST_FIRST to listOf(1, 2, 3, 4),
            FavoriteMatchesSort.FARTHEST_FIRST to listOf(4, 3, 2, 1),
            FavoriteMatchesSort.NEWEST_FIRST to listOf(2, 3, 1, 4),
            FavoriteMatchesSort.OLDEST_FIRST to listOf(1, 2, 3, 4),
        )
        expected.forEach { (sort, ranks) ->
            val result = sortFavoriteMatches(matches.reversed(), sort)
            assertEquals(ranks, result.map { it.rank })
            assertEquals(matches.toSet(), result.toSet())
            result.forEach { assertSame(matches[it.rank - 1], it) }
            assertEquals(result, sortFavoriteMatches(result, sort))
        }
        assertEquals(listOf(1, 2, 3, 4), matches.map { it.rank })
        assertTrue(sortFavoriteMatches(emptyList(), FavoriteMatchesSort.NEWEST_FIRST).isEmpty())
    }

    @Test
    fun `column clicks select natural order then reverse it`() {
        assertEquals(FavoriteMatchesSort.NEWEST_FIRST, FavoriteMatchesSort.CLOSEST_FIRST.toggleAddedDate())
        assertEquals(FavoriteMatchesSort.OLDEST_FIRST, FavoriteMatchesSort.NEWEST_FIRST.toggleAddedDate())
        assertEquals(FavoriteMatchesSort.NEWEST_FIRST, FavoriteMatchesSort.OLDEST_FIRST.toggleAddedDate())
        assertEquals(FavoriteMatchesSort.CLOSEST_FIRST, FavoriteMatchesSort.NEWEST_FIRST.toggleCloseness())
        assertEquals(FavoriteMatchesSort.FARTHEST_FIRST, FavoriteMatchesSort.CLOSEST_FIRST.toggleCloseness())
        assertEquals(FavoriteMatchesSort.CLOSEST_FIRST, FavoriteMatchesSort.FARTHEST_FIRST.toggleCloseness())
    }

    private fun match(rank: Int, added: Long) = FavoriteMatch(
        EmbeddedTrack(rank.toLong(), "track-$rank", "track-$rank", "Artist", "Album", "Track $rank",
            180_000, "/test/$rank.flac", indexAddedAtEpochSecond = added),
        rank, rank == 2,
    )
}
