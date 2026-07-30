package com.powerampstartradio.services

import com.powerampstartradio.data.EmbeddedTrack
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class HistoryReplayIdentityPolicyTest {
    @Test
    fun `authenticated embedded rows may replay legacy and sampled tracks by exact ID`() {
        val saved = listOf(
            SavedHistoryReplayIdentity(track(7L), null, 70L),
            SavedHistoryReplayIdentity(track(8L), stableId('a'), 80L),
        )

        val resolved = HistoryReplayIdentityPolicy.resolve(
            savedRows = saved,
            resolveAuthenticatedExactRow = { it.embeddedTrackId },
            resolveFullContentStableSpan = { error("not used") },
        )

        assertEquals(listOf(7L, 8L), resolved.map { it.embeddedTrackId })
    }

    @Test
    fun `cross generation remaps only catalog proven full content spans`() {
        val stable = stableId('b')
        val resolved = HistoryReplayIdentityPolicy.resolve(
            savedRows = listOf(SavedHistoryReplayIdentity(track(7L), stable, 70L)),
            resolveAuthenticatedExactRow = { null },
            resolveFullContentStableSpan = {
                if (it.stableTrackSpanId == stable) 70L else null
            },
        )

        assertEquals(70L, resolved.single().embeddedTrackId)
        assertEquals(stable, resolved.single().stableTrackSpanId)
    }

    @Test
    fun `cross generation rejects legacy or unresolvable identity`() {
        val missingIdentity = assertThrows(IllegalArgumentException::class.java) {
            HistoryReplayIdentityPolicy.resolve(
                savedRows = listOf(SavedHistoryReplayIdentity(track(7L), null, 70L)),
                resolveAuthenticatedExactRow = { null },
                resolveFullContentStableSpan = { 70L },
            )
        }
        assertTrue(
            missingIdentity.message?.contains("byte-identical indexed source-span identity") == true,
        )

        val missingSpan = assertThrows(IllegalArgumentException::class.java) {
            HistoryReplayIdentityPolicy.resolve(
                savedRows = listOf(
                    SavedHistoryReplayIdentity(track(7L), stableId('c'), 70L),
                ),
                resolveAuthenticatedExactRow = { null },
                resolveFullContentStableSpan = { null },
            )
        }
        assertTrue(
            missingSpan.message?.contains("byte-identical indexed source span") == true,
        )
    }

    @Test
    fun `mixed replay resolves exact legacy row then stable fallback row independently`() {
        val stable = stableId('d')
        val resolved = HistoryReplayIdentityPolicy.resolve(
            savedRows = listOf(
                SavedHistoryReplayIdentity(track(7L), null, 70L),
                SavedHistoryReplayIdentity(track(8L), stable, 80L),
            ),
            resolveAuthenticatedExactRow = { saved ->
                saved.embeddedTrackId.takeIf { it == 7L }
            },
            resolveFullContentStableSpan = { saved ->
                800L.takeIf { saved.stableTrackSpanId == stable }
            },
        )

        assertEquals(listOf(7L, 800L), resolved.map { it.embeddedTrackId })
        assertEquals(listOf(null, stable), resolved.map { it.stableTrackSpanId })
    }

    private fun stableId(value: Char) = "stable-track-span-v1-${value.toString().repeat(64)}"

    private fun track(id: Long) = EmbeddedTrack(
        id = id,
        metadataKey = "metadata-$id",
        filenameKey = "file-$id",
        artist = null,
        album = null,
        title = "Track $id",
        durationMs = 1,
        filePath = "/track/$id",
    )
}
