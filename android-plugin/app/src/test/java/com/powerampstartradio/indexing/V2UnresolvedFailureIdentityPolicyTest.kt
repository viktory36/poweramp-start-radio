package com.powerampstartradio.indexing

import com.powerampstartradio.indexing.v2.StableTrackSpanIdentity
import com.powerampstartradio.indexing.v2.StableTrackSpanIdentityStrength
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Test

class V2UnresolvedFailureIdentityPolicyTest {
    @Test
    fun `reused numeric id cannot hide a different provider span`() {
        val failure = requireNotNull(identity(7L, "/music/old.flac", 0L, 10_000L))
        val reusedId = track(7L, "/music/new.flac", 0L, 10_000)
        val remappedSameSpan = track(91L, "/music/old.flac", 0L, 10_000)

        assertEquals(
            setOf(91L),
            V2UnresolvedFailureIdentityPolicy.currentTrackIds(
                failures = listOf(failure),
                tracks = listOf(reusedId, remappedSameSpan),
            ),
        )
    }

    @Test
    fun `same path with changed duration is a different current occurrence`() {
        val failure = requireNotNull(identity(7L, "/music/a.flac", 0L, 10_000L))

        assertEquals(
            emptySet<Long>(),
            V2UnresolvedFailureIdentityPolicy.currentTrackIds(
                failures = listOf(failure),
                tracks = listOf(track(7L, "/music/a.flac", 0L, 10_001)),
            ),
        )
    }

    @Test
    fun `provider paths are normalized before current matching`() {
        val failure = requireNotNull(identity(7L, "/music/album/../a.flac", 0L, 10_000L))

        assertEquals(
            setOf(91L),
            V2UnresolvedFailureIdentityPolicy.currentTrackIds(
                failures = listOf(failure),
                tracks = listOf(track(91L, "/music/a.flac", 0L, 10_000)),
            ),
        )
    }

    @Test
    fun `invalid failure identity cannot fall back to numeric id`() {
        val invalid = identity(7L, "relative.flac", 0L, 10_000L)

        assertEquals(null, invalid)
        assertNotNull(track(7L, "/music/current.flac", 0L, 10_000))
    }

    @Test
    fun `byte-identical files at different provider spans remain separate failures`() {
        val first = requireNotNull(identity(7L, "/music/first.flac", 0L, 10_000L))
        val second = requireNotNull(identity(8L, "/music/second.flac", 0L, 10_000L))

        assertEquals(
            listOf("first failure", "second failure"),
            V2UnresolvedFailureIdentityPolicy.latestUnresolvedValues(
                listOf(
                    outcome(first, createdAt = 1L, revision = 1L, value = "first failure"),
                    outcome(second, createdAt = 1L, revision = 1L, value = "second failure"),
                ),
            ),
        )
    }

    @Test
    fun `new ledger outcome supersedes only the exact provider occurrence`() {
        val first = requireNotNull(identity(7L, "/music/first.flac", 0L, 10_000L))
        val second = requireNotNull(identity(8L, "/music/second.flac", 0L, 10_000L))

        assertEquals(
            listOf("second failure"),
            V2UnresolvedFailureIdentityPolicy.latestUnresolvedValues(
                listOf(
                    outcome(first, createdAt = 1L, revision = 1L, value = "first failure"),
                    outcome(second, createdAt = 1L, revision = 1L, value = "second failure"),
                    outcome<String>(first, createdAt = 2L, revision = 1L, value = null),
                ),
            ),
        )
    }

    @Test
    fun `full content proof bridges provisional and finalized EOS stable ids`() {
        val provisional = requireNotNull(
            identity(7L, "/music/moby.mp3", 0L, 209_110L, stableSuffix = "a"),
        )
        val finalized = requireNotNull(
            identity(7L, "/music/moby.mp3", 0L, 209_110L, stableSuffix = "b"),
        )
        val content = fullContentIdentity(finalized)

        assertEquals(
            emptyList<String>(),
            V2UnresolvedFailureIdentityPolicy.latestUnresolvedValues(
                listOf(
                    outcome(
                        provisional,
                        createdAt = 1L,
                        revision = 1L,
                        value = "provisional decode failure",
                        contentSupersessionIdentity = fullContentIdentity(provisional),
                    ),
                    outcome<String>(
                        finalized,
                        createdAt = 2L,
                        revision = 1L,
                        value = null,
                        contentSupersessionIdentity = content,
                    ),
                ),
            ),
        )
    }

    @Test
    fun `sampled content cannot bridge changed stable span ids`() {
        val provisional = requireNotNull(
            identity(7L, "/music/moby.mp3", 0L, 209_110L, stableSuffix = "a"),
        )
        val finalized = requireNotNull(
            identity(7L, "/music/moby.mp3", 0L, 209_110L, stableSuffix = "b"),
        )

        assertEquals(
            listOf("provisional decode failure"),
            V2UnresolvedFailureIdentityPolicy.latestUnresolvedValues(
                listOf(
                    outcome(provisional, 1L, 1L, "provisional decode failure"),
                    outcome<String>(finalized, 2L, 1L, null),
                ),
            ),
        )
        assertNull(
            V2UnresolvedFailureIdentityPolicy.contentSupersessionIdentityOrNull(
                provisional,
                stableIdentity(StableTrackSpanIdentityStrength.VERSIONED_SAMPLED_CONTENT_SHA256),
            ),
        )
    }

    private fun identity(
        id: Long,
        path: String,
        offsetMs: Long,
        durationMs: Long,
        stableSuffix: String = "a",
    ) = V2UnresolvedFailureIdentityPolicy.identityOrNull(
        stableTrackSpanId = "stable-track-span-v1-${stableSuffix.repeat(64)}",
        powerampFileId = id,
        providerPhysicalPath = path,
        offsetMs = offsetMs,
        durationMs = durationMs,
    )

    private fun track(
        id: Long,
        path: String,
        offsetMs: Long,
        durationMs: Int,
    ) = NewTrackDetector.UnindexedTrack(
        powerampFileId = id,
        artist = "Artist",
        album = "Album",
        title = "Track",
        durationMs = durationMs,
        path = path,
        offsetMs = offsetMs,
    )

    private fun <T : Any> outcome(
        identity: V2UnresolvedFailureOccurrenceIdentity,
        createdAt: Long,
        revision: Long,
        value: T?,
        contentSupersessionIdentity: V2UnresolvedFailureContentSupersessionIdentity? = null,
    ) = V2UnresolvedFailureOccurrenceOutcome(
        identity = identity,
        contentSupersessionIdentity = contentSupersessionIdentity,
        jobCreatedAtEpochMs = createdAt,
        ledgerRevision = revision,
        unresolvedValue = value,
    )

    private fun fullContentIdentity(
        occurrence: V2UnresolvedFailureOccurrenceIdentity,
    ) = requireNotNull(
        V2UnresolvedFailureIdentityPolicy.contentSupersessionIdentityOrNull(
            occurrence,
            stableIdentity(StableTrackSpanIdentityStrength.FULL_CONTENT_SHA256),
        ),
    )

    private fun stableIdentity(
        strength: StableTrackSpanIdentityStrength,
    ) = StableTrackSpanIdentity(
        identitySpecId = "stable-track-span-v1:test",
        stableTrackSpanId = "stable-track-span-v1-${"c".repeat(64)}",
        strength = strength,
        contentFingerprintSpecId = if (
            strength == StableTrackSpanIdentityStrength.FULL_CONTENT_SHA256
        ) "full-content-sha256-v1" else "sampled-v1",
        contentSha256 = "d".repeat(64),
        sourceSizeBytes = 7_216_894L,
        sourceSampleRateHz = 44_100,
        startSourceSample = 0L,
        endSourceSampleExclusive = 9_219_840L,
    )
}
