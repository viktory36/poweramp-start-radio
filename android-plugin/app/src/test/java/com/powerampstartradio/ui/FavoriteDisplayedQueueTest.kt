package com.powerampstartradio.ui

import com.powerampstartradio.data.EmbeddedTrack
import com.powerampstartradio.data.StableIdentityGenerationBinding
import org.junit.Assert.*
import org.junit.Test

class FavoriteDisplayedQueueTest {
    private val seed = track(10, 1)
    private val matches = listOf(FavoriteMatch(track(11, 100), 1, false),
        FavoriteMatch(track(12, 300), 2, true), FavoriteMatch(track(13, 200), 3, false))
    private val state = FavoriteMatchesState(seed = FavoriteSeed("song:10", trackId = 10,
        title = seed.title, artist = seed.artist), matches = matches, generationId = "generation",
        resolvedSeedTrackId = 10, resolvedSeedTrack = seed,
        binding = FavoriteResultsBinding(StableIdentityGenerationBinding("binding", "generation", "activation",
            "a".repeat(64), "b".repeat(64)), "provider", "c".repeat(64), 100, 99))

    @Test
    fun `seed stays first for all displayed sort orders without reranking`() {
        val orders = mapOf(FavoriteMatchesSort.CLOSEST_FIRST to listOf(11L, 12L, 13L),
            FavoriteMatchesSort.FARTHEST_FIRST to listOf(13L, 12L, 11L),
            FavoriteMatchesSort.NEWEST_FIRST to listOf(12L, 13L, 11L),
            FavoriteMatchesSort.OLDEST_FIRST to listOf(11L, 13L, 12L))
        orders.forEach { (sort, ids) ->
            val plan = favoriteDisplayedQueue(state, sort)
            assertEquals(listOf(10L) + ids, plan.tracks.map { it.id })
            assertEquals(plan.tracks.map { it.id }, plan.identities.map { it.embeddedTrackId })
            assertEquals(listOf(null) + ids.map { id -> matches.single { it.track.id == id }.rank },
                plan.evidence.orderedClosestRanks)
            assertTrue(plan.evidence.seedIncluded)
            assertEquals(sort, plan.evidence.sort)
        }
        assertEquals(listOf(1, 2, 3), state.matches.map { it.rank })
    }

    @Test
    fun `text has no playable seed and incomplete displays cannot be queued`() {
        val text = state.copy(seed = FavoriteSeed("text:sleep", text = "sleep"),
            resolvedSeedTrackId = null, resolvedSeedTrack = null)
        val plan = favoriteDisplayedQueue(text, FavoriteMatchesSort.NEWEST_FIRST)
        assertEquals(listOf(12L, 13L, 11L), plan.tracks.map { it.id })
        assertFalse(plan.evidence.seedIncluded)
        assertEquals("sleep", plan.evidence.seed.text)
        assertThrows(IllegalStateException::class.java) { favoriteDisplayedQueue(state.copy(loading = true), FavoriteMatchesSort.CLOSEST_FIRST) }
        assertThrows(IllegalStateException::class.java) { favoriteDisplayedQueue(state.copy(error = "failed"), FavoriteMatchesSort.CLOSEST_FIRST) }
    }

    private fun track(id: Long, added: Long) = EmbeddedTrack(id, "key:$id", "track$id", "Artist", "Album",
        "Song $id", 180_000, "/music/$id.flac", indexAddedAtEpochSecond = added)
}
