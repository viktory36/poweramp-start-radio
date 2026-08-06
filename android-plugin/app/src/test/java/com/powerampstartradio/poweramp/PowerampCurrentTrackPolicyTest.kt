package com.powerampstartradio.poweramp

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class PowerampCurrentTrackPolicyTest {
    @Test
    fun `activity resume requires sticky revalidation of live authenticated state`() {
        val live = authenticated(
            track = track(),
            state = PowerampHelper.STATE_PLAYING,
        )

        val resumed = PowerampActivityResumePolicy.requireStickyRevalidation(live)

        assertEquals(
            live.copy(origin = PowerampAuthenticatedStateOrigin.PERSISTED_EXPLICIT),
            resumed,
        )
        assertEquals(
            resumed,
            PowerampActivityResumePolicy.requireStickyRevalidation(resumed),
        )
        assertNull(PowerampActivityResumePolicy.requireStickyRevalidation(null))
    }

    @Test
    fun `process restart admits only matching sticky identity and state`() {
        val live = authenticated(track = track(), state = PowerampHelper.STATE_PLAYING)
        val persisted = PowerampAuthenticatedStateCodec.decode(
            PowerampAuthenticatedStateCodec.encode(live),
        )

        assertEquals(PowerampAuthenticatedStateOrigin.PERSISTED_EXPLICIT, persisted.origin)
        assertTrue(
            PowerampCurrentTrackIdentityPolicy.persistedStateMatchesSticky(
                persisted,
                stickyTrack = track(),
                stickyPlaybackState = PowerampHelper.STATE_PLAYING,
            ),
        )
        assertFalse(
            PowerampCurrentTrackIdentityPolicy.persistedStateMatchesSticky(
                persisted,
                stickyTrack = track(realId = 8L),
                stickyPlaybackState = PowerampHelper.STATE_PLAYING,
            ),
        )
        assertFalse(
            PowerampCurrentTrackIdentityPolicy.persistedStateMatchesSticky(
                persisted,
                stickyTrack = track(),
                stickyPlaybackState = PowerampHelper.STATE_PAUSED,
            ),
        )
    }

    @Test
    fun `persisted identity cannot be replaced by a spoofed sticky path or queue occurrence`() {
        val persisted = PowerampAuthenticatedStateCodec.decode(
            PowerampAuthenticatedStateCodec.encode(
                authenticated(track = track(queueId = 44L), state = PowerampHelper.STATE_PLAYING),
            ),
        )

        assertFalse(
            PowerampCurrentTrackIdentityPolicy.persistedStateMatchesSticky(
                persisted,
                stickyTrack = track(path = "/music/attacker.flac", queueId = 44L),
                stickyPlaybackState = PowerampHelper.STATE_PLAYING,
            ),
        )
        assertFalse(
            PowerampCurrentTrackIdentityPolicy.persistedStateMatchesSticky(
                persisted,
                stickyTrack = track(queueId = 45L),
                stickyPlaybackState = PowerampHelper.STATE_PLAYING,
            ),
        )
    }

    @Test
    fun `provider owns command metadata and exact queue occurrence is revalidated`() {
        val resolved = PowerampCommandTrackPolicy.requireProviderBacked(
            authenticated = authenticated(
                track = track(queueId = 44L),
                state = PowerampHelper.STATE_PLAYING,
            ),
            providerEntries = listOf(provider()),
            queueEntries = listOf(QueueEntry(queueId = 44L, fileId = 7L, sort = 12)),
        )

        requireNotNull(resolved)
        assertEquals("Song", resolved.title)
        assertEquals(44L, resolved.queueOccurrenceId)
        assertEquals(7L, resolved.realId)
    }

    @Test
    fun `provider id reuse fails closed even when numeric id is unchanged`() {
        val evidence = authenticated(track = track(), state = PowerampHelper.STATE_PAUSED)

        assertThrows(IllegalArgumentException::class.java) {
            PowerampCommandTrackPolicy.requireProviderBacked(
                authenticated = evidence,
                providerEntries = listOf(provider(path = "/music/replaced.flac")),
                queueEntries = null,
            )
        }
        assertThrows(IllegalArgumentException::class.java) {
            PowerampCommandTrackPolicy.requireProviderBacked(
                authenticated = evidence,
                providerEntries = listOf(provider(title = "Different recording")),
                queueEntries = null,
            )
        }
    }

    @Test
    fun `legacy sticky candidate fails closed on provider id or path reuse`() {
        val legacy = PowerampLegacyStickyCandidatePolicy.fromSticky(
            stickyTrack = track(),
            stickyPlaybackState = PowerampHelper.STATE_PLAYING,
        )

        assertThrows(IllegalArgumentException::class.java) {
            PowerampCommandTrackPolicy.requireLegacyProviderBacked(
                candidate = legacy,
                providerEntries = listOf(provider(path = "/music/reused.flac")),
                queueEntries = null,
            )
        }
        assertThrows(IllegalStateException::class.java) {
            PowerampCommandTrackPolicy.requireLegacyProviderBacked(
                candidate = legacy,
                providerEntries = listOf(provider().copy(id = 8L)),
                queueEntries = null,
            )
        }
    }

    @Test
    fun `legacy queue candidate drops stale or ambiguous occurrence but remains a valid seed`() {
        val legacy = PowerampLegacyStickyCandidatePolicy.fromSticky(
            stickyTrack = track(queueId = 44L),
            stickyPlaybackState = PowerampHelper.STATE_PLAYING,
        )

        assertProviderSeedWithoutQueue(
            PowerampCommandTrackPolicy.requireLegacyProviderBacked(
                candidate = legacy,
                providerEntries = listOf(provider()),
                queueEntries = listOf(QueueEntry(queueId = 44L, fileId = 8L, sort = 12)),
            ),
        )
        assertProviderSeedWithoutQueue(
            PowerampCommandTrackPolicy.requireLegacyProviderBacked(
                candidate = legacy,
                providerEntries = listOf(provider()),
                queueEntries = emptyList(),
            ),
        )
        assertProviderSeedWithoutQueue(
            PowerampCommandTrackPolicy.requireLegacyProviderBacked(
                candidate = legacy,
                providerEntries = listOf(provider()),
                queueEntries = listOf(
                    QueueEntry(queueId = 44L, fileId = 7L, sort = 12),
                    QueueEntry(queueId = 44L, fileId = 7L, sort = 13),
                ),
            ),
        )
    }

    @Test
    fun `legacy stopped presentation retains provider verifiable selection and races fail`() {
        val stopped = PowerampLegacyStickyCandidatePolicy.fromSticky(
            stickyTrack = track(),
            stickyPlaybackState = PowerampHelper.STATE_STOPPED,
        )
        assertEquals(track(), stopped.track)
        assertEquals(
            7L,
            PowerampCommandTrackPolicy.requireLegacyProviderBacked(
                candidate = stopped,
                providerEntries = listOf(provider()),
                queueEntries = null,
            )?.realId,
        )

        val stoppedWithoutSelection = PowerampLegacyStickyCandidatePolicy.fromSticky(
            stickyTrack = null,
            stickyPlaybackState = PowerampHelper.STATE_STOPPED,
        )
        assertNull(stoppedWithoutSelection.track)
        assertNull(
            PowerampCommandTrackPolicy.requireLegacyProviderBacked(
                candidate = stoppedWithoutSelection,
                providerEntries = emptyList(),
                queueEntries = null,
            ),
        )

        val stoppedWithOnlyStaleFallback = PowerampLegacyStickyCandidatePolicy.fromSticky(
            stickyTrack = null,
            stickyPlaybackState = PowerampHelper.STATE_STOPPED,
            fallbackTrack = track(),
        )
        assertNull(stoppedWithOnlyStaleFallback.track)

        val pausedWithFallback = PowerampLegacyStickyCandidatePolicy.fromSticky(
            stickyTrack = null,
            stickyPlaybackState = PowerampHelper.STATE_PAUSED,
            fallbackTrack = track(),
        )
        assertEquals(track(), pausedWithFallback.track)

        val before = PowerampLegacyStickyCandidatePolicy.fromSticky(
            stickyTrack = track(),
            stickyPlaybackState = PowerampHelper.STATE_PLAYING,
        )
        val after = PowerampLegacyStickyCandidatePolicy.fromSticky(
            stickyTrack = track(path = "/music/next.flac"),
            stickyPlaybackState = PowerampHelper.STATE_PLAYING,
        )
        assertFalse(PowerampLegacyStickyCandidatePolicy.unchanged(before, after))
    }

    @Test
    fun `missing reused or ambiguous queue row drops anchor but keeps exact provider seed`() {
        val evidence = authenticated(
            track = track(queueId = 44L),
            state = PowerampHelper.STATE_PLAYING,
        )

        assertProviderSeedWithoutQueue(
            PowerampCommandTrackPolicy.requireProviderBacked(
                authenticated = evidence,
                providerEntries = listOf(provider()),
                queueEntries = listOf(QueueEntry(queueId = 44L, fileId = 999L, sort = 12)),
            ),
        )
        assertProviderSeedWithoutQueue(
            PowerampCommandTrackPolicy.requireProviderBacked(
                authenticated = evidence,
                providerEntries = listOf(provider()),
                queueEntries = listOf(
                    QueueEntry(queueId = 44L, fileId = 7L, sort = 12),
                    QueueEntry(queueId = 44L, fileId = 7L, sort = 13),
                ),
            ),
        )
    }

    @Test
    fun `authenticated stopped state survives restart without a track`() {
        val stopped = authenticated(track = null, state = PowerampHelper.STATE_STOPPED)
        val persisted = PowerampAuthenticatedStateCodec.decode(
            PowerampAuthenticatedStateCodec.encode(stopped),
        )

        assertTrue(
            PowerampCurrentTrackIdentityPolicy.persistedStateMatchesSticky(
                persisted,
                stickyTrack = null,
                stickyPlaybackState = PowerampHelper.STATE_STOPPED,
            ),
        )
        assertFalse(
            PowerampCurrentTrackIdentityPolicy.persistedStateMatchesSticky(
                persisted,
                stickyTrack = track(),
                stickyPlaybackState = PowerampHelper.STATE_STOPPED,
            ),
        )
        assertNull(
            PowerampCommandTrackPolicy.requireProviderBacked(
                authenticated = persisted,
                providerEntries = emptyList(),
                queueEntries = null,
            ),
        )
    }

    @Test
    fun `older explicit delivery cannot roll current state backward`() {
        val current = authenticated(
            track = track(realId = 8L, path = "/music/new.flac"),
            state = PowerampHelper.STATE_PLAYING,
            timestampMs = 200L,
        )

        assertEquals(
            current,
            PowerampExplicitEventStateMachine.trackChanged(
                current,
                track(realId = 7L),
                eventTimestampMs = 199L,
            ),
        )
        val stopped = PowerampExplicitEventStateMachine.statusChanged(
            current,
            playbackState = PowerampHelper.STATE_STOPPED,
            eventTrack = track(realId = 7L),
            eventTimestampMs = 201L,
        )
        assertNull(stopped.track)
        assertEquals(PowerampHelper.STATE_STOPPED, stopped.playbackState)
    }

    @Test
    fun `status-only event cannot promote a process-restart track to live evidence`() {
        val persisted = PowerampAuthenticatedStateCodec.decode(
            PowerampAuthenticatedStateCodec.encode(
                authenticated(track = track(), state = PowerampHelper.STATE_PLAYING),
            ),
        )

        val paused = PowerampExplicitEventStateMachine.statusChanged(
            previous = persisted,
            playbackState = PowerampHelper.STATE_PAUSED,
            eventTrack = null,
            eventTimestampMs = 101L,
        )

        assertEquals(PowerampAuthenticatedStateOrigin.PERSISTED_EXPLICIT, paused.origin)
        assertEquals(track(), paused.track)
        assertTrue(
            PowerampCurrentTrackIdentityPolicy.persistedStateMatchesSticky(
                persisted = paused,
                stickyTrack = track(),
                stickyPlaybackState = PowerampHelper.STATE_PAUSED,
            ),
        )
        assertEquals(
            PowerampAuthenticatedStateOrigin.PERSISTED_EXPLICIT,
            PowerampAuthenticatedStateCodec.decode(
                PowerampAuthenticatedStateCodec.encode(paused),
            ).origin,
        )

        val stopped = PowerampExplicitEventStateMachine.statusChanged(
            previous = persisted,
            playbackState = PowerampHelper.STATE_STOPPED,
            eventTrack = null,
            eventTimestampMs = 102L,
        )
        assertEquals(PowerampAuthenticatedStateOrigin.LIVE_EXPLICIT, stopped.origin)
        assertNull(stopped.track)
    }

    @Test
    fun `corrupt persisted evidence is rejected`() {
        assertThrows(RuntimeException::class.java) {
            PowerampAuthenticatedStateCodec.decode("{\"schemaVersion\":999}")
        }
        assertThrows(RuntimeException::class.java) {
            PowerampAuthenticatedStateCodec.decode(
                "{\"schemaVersion\":1,\"playbackState\":0," +
                    "\"track\":{\"realId\":7,\"title\":\"Injected\"}}",
            )
        }
    }

    private fun authenticated(
        track: PowerampTrack?,
        state: Int?,
        timestampMs: Long = 100L,
    ) = PowerampAuthenticatedState(
        track = track,
        playbackState = state,
        lastEventTimestampMs = timestampMs,
        origin = PowerampAuthenticatedStateOrigin.LIVE_EXPLICIT,
    )

    private fun track(
        realId: Long = 7L,
        path: String = "/music/artist/song.flac",
        queueId: Long? = null,
    ) = PowerampTrack(
        realId = realId,
        title = "Song",
        artist = "Artist",
        album = "Album",
        durationMs = 123_000,
        path = path,
        trackId = queueId ?: 91L,
        categoryUri = if (queueId == null) {
            "content://com.maxmpz.audioplayer.data/folders/3/files"
        } else {
            "content://com.maxmpz.audioplayer.data/queue?shs=2"
        },
        positionInList = 2,
    )

    private fun provider(
        path: String = "/music/artist/song.flac",
        title: String = "song",
    ) = PowerampFileEntry(
        id = 7L,
        artist = "artist",
        album = "album",
        title = title,
        durationMs = 123_000,
        path = path,
        metadataKey = "artist|album|song|123000",
        filenameKeys = setOf("song"),
    )

    private fun assertProviderSeedWithoutQueue(track: PowerampTrack?) {
        val resolved = requireNotNull(track)
        assertEquals(7L, resolved.realId)
        assertEquals("Song", resolved.title)
        assertNull(resolved.queueOccurrenceId)
        assertNull(resolved.categoryUri)
    }
}
