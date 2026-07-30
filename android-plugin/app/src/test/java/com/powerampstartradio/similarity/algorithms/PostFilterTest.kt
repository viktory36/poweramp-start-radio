package com.powerampstartradio.similarity.algorithms

import com.powerampstartradio.data.EmbeddedTrack
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class PostFilterTest {
    @Test
    fun matchesTrimmedArtistCreditsWithoutParsingPerformers() {
        val queue = listOf(track(1, "Artist A & Artist B"))

        assertFalse(PostFilter.canAdd(track(2, " artist a & artist b "), queue, 8, 1))
        assertTrue(PostFilter.canAdd(track(3, "Artist A"), queue, 8, 1))
    }

    @Test
    fun blankCreditsAreExplicitlyUnconstrained() {
        val queue = listOf(track(1, null), track(2, "  "))

        assertTrue(PostFilter.canAdd(track(3, null), queue, 1, 20))
        assertTrue(PostFilter.canAdd(track(4, ""), queue, 1, 20))
    }

    @Test
    fun maximumOneMakesEverySpacingValueEligibilityEquivalent() {
        val queue = listOf(track(1, "Artist A"), track(2, "Artist B"))
        val candidate = track(3, "Artist A")

        assertFalse(PostFilter.canAdd(candidate, queue, 1, 0))
        assertFalse(PostFilter.canAdd(candidate, queue, 1, 20))
    }

    @Test
    fun spacingAtOrBeyondQueueHorizonHasTheSameNoRepeatReach() {
        val queueWithCredit = listOf(track(1, "Artist A")) +
            (2L..9L).map { track(it, "Artist $it") }
        val queueWithoutCredit = (1L..9L).map { track(it, "Artist $it") }
        val candidate = track(10, "Artist A")

        assertFalse(PostFilter.canAdd(candidate, queueWithCredit, 8, 9))
        assertFalse(PostFilter.canAdd(candidate, queueWithCredit, 8, 20))
        assertTrue(PostFilter.canAdd(candidate, queueWithoutCredit, 8, 9))
        assertTrue(PostFilter.canAdd(candidate, queueWithoutCredit, 8, 20))
    }

    private fun track(id: Long, artist: String?) = EmbeddedTrack(
        id = id,
        metadataKey = "metadata-$id",
        filenameKey = "filename-$id",
        artist = artist,
        album = null,
        title = "Track $id",
        durationMs = 1_000,
        filePath = "/music/$id.flac",
    )
}
