package com.powerampstartradio.indexing

import org.junit.Assert.assertEquals
import org.junit.Test

class V2TrackExclusionPolicyTest {
    @Test
    fun `provider identity survives id remap but rejects reused numeric id`() {
        val original = candidate(7L, "/music/a.flac", offsetMs = 0L, durationMs = 10_000L)
        val exclusions = V2TrackExclusionPolicy.add(emptyList(), listOf(original))
        val remappedSameTrack = original.copy(powerampFileId = 91L)
        val reusedOldId = candidate(7L, "/music/new.flac", offsetMs = 0L, durationMs = 8_000L)

        assertEquals(
            setOf(91L),
            V2TrackExclusionPolicy.resolve(exclusions, listOf(remappedSameTrack, reusedOldId)),
        )
        assertEquals(
            91L,
            V2TrackExclusionPolicy.refreshLocators(exclusions, listOf(remappedSameTrack))
                .single().lastKnownPowerampFileId,
        )
    }

    @Test
    fun `logical spans in one source remain distinct`() {
        val first = candidate(1L, "/music/mix.flac", offsetMs = 0L, durationMs = 30_000L)
        val second = candidate(2L, "/music/mix.flac", offsetMs = 30_000L, durationMs = 40_000L)
        val exclusions = V2TrackExclusionPolicy.add(emptyList(), listOf(first))

        assertEquals(setOf(1L), V2TrackExclusionPolicy.resolve(exclusions, listOf(first, second)))
    }

    @Test
    fun `same path with changed duration is not the excluded occurrence`() {
        val original = candidate(7L, "/music/a.flac", offsetMs = 0L, durationMs = 10_000L)
        val changed = candidate(7L, "/music/a.flac", offsetMs = 0L, durationMs = 10_001L)

        assertEquals(
            emptySet<Long>(),
            V2TrackExclusionPolicy.resolve(
                V2TrackExclusionPolicy.add(emptyList(), listOf(original)),
                listOf(changed),
            ),
        )
    }

    @Test
    fun `non-positive duration sentinels share exclusion identity but positive duration does not`() {
        val negativeSentinel = candidate(7L, "/music/unknown.opus", 0L, -1L)
        val zeroSentinel = candidate(91L, "/music/unknown.opus", 0L, 0L)
        val laterKnown = candidate(92L, "/music/unknown.opus", 0L, 10_000L)
        val exclusions = V2TrackExclusionPolicy.add(emptyList(), listOf(negativeSentinel))

        assertEquals(0L, exclusions.single().providerSpan.durationMs)
        assertEquals(
            setOf(91L),
            V2TrackExclusionPolicy.resolve(exclusions, listOf(zeroSentinel, laterKnown)),
        )
        assertEquals(
            0L,
            V2TrackExclusionPolicy.requireValid(
                V2TrackExclusionEnvelope(
                    entries = listOf(
                        exclusions.single().copy(
                            providerSpan = exclusions.single().providerSpan.copy(durationMs = -99L),
                        ),
                    ),
                ),
            ).entries.single().providerSpan.durationMs,
        )
    }

    @Test
    fun `verified stable span survives provider path migration`() {
        val stableId = "stable-track-span-v1-" + "a".repeat(64)
        val original = candidate(1L, "/old/a.flac", 0L, 10_000L)
            .copy(stableTrackSpanId = stableId)
        val moved = candidate(88L, "/new/a.flac", 0L, 10_000L)
            .copy(stableTrackSpanId = stableId)

        val exclusions = V2TrackExclusionPolicy.add(emptyList(), listOf(original))
        assertEquals(setOf(88L), V2TrackExclusionPolicy.resolve(exclusions, listOf(moved)))
    }

    private fun candidate(
        id: Long,
        path: String,
        offsetMs: Long,
        durationMs: Long,
    ) = V2TrackExclusionCandidate(
        powerampFileId = id,
        providerSpan = V2ProviderSpanLocator(path, offsetMs, durationMs),
    )
}
