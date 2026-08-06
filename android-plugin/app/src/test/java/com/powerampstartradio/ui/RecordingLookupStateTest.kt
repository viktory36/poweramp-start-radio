package com.powerampstartradio.ui

import com.powerampstartradio.data.EmbeddedTrack
import org.junit.Assert.assertEquals
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Test

class RecordingLookupStateTest {
    @Test
    fun `idle starts loading and matching completion publishes all candidates`() {
        val loading = RecordingLookupStateReducer.start(7L, "pink floyd echoes")
        val candidates = listOf(track(1L), track(2L), track(3L))
        val success = RecordingLookupStateReducer.succeed(
            loading,
            seedId = 7L,
            query = "pink floyd echoes",
            candidates = candidates,
        )

        assertTrue(loading is RecordingLookupState.Loading)
        assertTrue(success is RecordingLookupState.Success)
        assertEquals(candidates, (success as RecordingLookupState.Success).candidates)
        assertEquals(false, success.hasMoreMatches)
    }

    @Test
    fun `bounded completion preserves honest more matches signal`() {
        val loading = RecordingLookupStateReducer.start(7L, "pink floyd")
        val success = RecordingLookupStateReducer.succeed(
            current = loading,
            seedId = 7L,
            query = "pink floyd",
            candidates = listOf(track(1L), track(2L)),
            hasMoreMatches = true,
        ) as RecordingLookupState.Success

        assertEquals(2, success.candidates.size)
        assertTrue(success.hasMoreMatches)
    }

    @Test
    fun `successful empty result is distinct from failure`() {
        val loading = RecordingLookupStateReducer.start(7L, "missing")
        val empty = RecordingLookupStateReducer.succeed(loading, 7L, "missing", emptyList())
        val failure = RecordingLookupStateReducer.fail(
            loading,
            7L,
            "missing",
            "Recording lookup failed: database unavailable",
        )

        assertTrue(empty is RecordingLookupState.Success)
        assertTrue((empty as RecordingLookupState.Success).candidates.isEmpty())
        assertTrue(failure is RecordingLookupState.Failure)
        assertTrue((failure as RecordingLookupState.Failure).message.contains("failed"))
    }

    @Test
    fun `stale completion cannot replace a newer lookup`() {
        val current = RecordingLookupStateReducer.start(8L, "new query")

        assertSame(
            current,
            RecordingLookupStateReducer.succeed(current, 7L, "old query", listOf(track(1L))),
        )
        assertSame(
            current,
            RecordingLookupStateReducer.fail(current, 7L, "old query", "old failure"),
        )
    }

    @Test
    fun `dismiss always returns to explicit idle`() {
        assertEquals(RecordingLookupState.Idle, RecordingLookupStateReducer.clear())
    }

    @Test
    fun `candidate evidence includes album duration and filename`() {
        val evidence = RecordingCandidateEvidenceFormatter.format(track(1L))

        assertEquals("Artist \u00b7 Live at Pompeii", evidence.artistAndAlbum)
        assertEquals("4:05 \u00b7 echoes-live.flac", evidence.durationAndFile)
    }

    @Test
    fun `candidate evidence extracts filename from desktop database path on Android`() {
        val track = track(1L).copy(filePath = "C:\\Music\\Fixture Artist\\Example.flac")

        assertEquals(
            "4:05 \u00b7 Example.flac",
            RecordingCandidateEvidenceFormatter.format(track).durationAndFile,
        )
    }

    private fun track(id: Long) = EmbeddedTrack(
        id = id,
        metadataKey = "metadata-$id",
        filenameKey = "file-$id",
        artist = "Artist",
        album = "Live at Pompeii",
        title = "Echoes (Live)",
        durationMs = 245_000,
        filePath = "/music/echoes-live.flac",
    )
}
