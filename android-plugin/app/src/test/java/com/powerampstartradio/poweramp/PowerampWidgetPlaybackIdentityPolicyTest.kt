package com.powerampstartradio.poweramp

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class PowerampWidgetPlaybackIdentityPolicyTest {
    @Test
    fun `api 34 widget can use provider verified sticky track without authenticated state`() {
        val sticky = track(realId = 10L, queueRowId = 100L)
        val candidate = PowerampLegacyStickyCandidatePolicy.fromSticky(
            stickyTrack = sticky,
            stickyPlaybackState = PowerampHelper.STATE_PLAYING,
        )

        val verified = PowerampCommandTrackPolicy.requireLegacyProviderBacked(
            candidate = candidate,
            providerEntries = listOf(
                PowerampFileEntry(
                    id = 10L,
                    title = "track",
                    artist = "artist",
                    album = "album",
                    durationMs = 180_000,
                    path = "/music/10.flac",
                    metadataKey = "artist|album|track|180000",
                    filenameKeys = setOf("track"),
                ),
            ),
            queueEntries = listOf(QueueEntry(queueId = 100L, fileId = 10L, sort = 7)),
        )

        requireNotNull(verified)
        assertEquals(10L, verified.realId)
        assertEquals(100L, verified.queueOccurrenceId)
    }

    @Test
    fun `live authenticated evidence is ready only while sticky identity still matches`() {
        val track = track(realId = 10L, queueRowId = 100L)
        val authenticated = PowerampAuthenticatedState(
            track = track,
            playbackState = PowerampHelper.STATE_PLAYING,
            lastEventTimestampMs = 1_000L,
            origin = PowerampAuthenticatedStateOrigin.LIVE_EXPLICIT,
        )

        assertTrue(
            PowerampCurrentTrackIdentityPolicy.authenticatedStateMatchesSticky(
                authenticated,
                track.copy(title = "Display case may differ"),
                PowerampHelper.STATE_PLAYING,
            ),
        )
        assertFalse(
            PowerampCurrentTrackIdentityPolicy.authenticatedStateMatchesSticky(
                authenticated,
                track.copy(realId = 11L),
                PowerampHelper.STATE_PLAYING,
            ),
        )
        assertFalse(
            PowerampCurrentTrackIdentityPolicy.authenticatedStateMatchesSticky(
                authenticated,
                track.copy(trackId = 101L),
                PowerampHelper.STATE_PLAYING,
            ),
        )
    }

    @Test
    fun `playback state drift makes authenticated widget evidence unready`() {
        val authenticated = PowerampAuthenticatedState(
            track = track(10L, 100L),
            playbackState = PowerampHelper.STATE_PLAYING,
            lastEventTimestampMs = 1_000L,
            origin = PowerampAuthenticatedStateOrigin.LIVE_EXPLICIT,
        )

        assertFalse(
            PowerampCurrentTrackIdentityPolicy.authenticatedStateMatchesSticky(
                authenticated,
                authenticated.track,
                PowerampHelper.STATE_PAUSED,
            ),
        )
    }

    private fun track(realId: Long, queueRowId: Long) = PowerampTrack(
        realId = realId,
        title = "Track",
        artist = "Artist",
        album = "Album",
        durationMs = 180_000,
        path = "/music/$realId.flac",
        trackId = queueRowId,
        categoryUri = "content://com.maxmpz.audioplayer.data/queue",
    )
}
