package com.powerampstartradio.services

import com.powerampstartradio.data.EmbeddedTrack
import com.powerampstartradio.indexing.V2ActiveLibraryBinding
import com.powerampstartradio.indexing.V2ActiveLibraryBindingEvidence
import com.powerampstartradio.indexing.v2.V2CommittedProviderSpan
import com.powerampstartradio.indexing.v2.V2ProviderSpanReceipt
import com.powerampstartradio.poweramp.PowerampFileEntry
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class CurrentPowerampBindingPolicyTest {
    @Test
    fun exactReceiptSpanAcceptsTheCurrentRecording() {
        assertTrue(
            CurrentPowerampBindingPolicy.matches(
                binding = binding(V2ActiveLibraryBindingEvidence.EXACT_V2_RECEIPT_SPAN),
                indexed = track(),
                current = provider(),
                receipt = receipt(),
            ),
        )
    }

    @Test
    fun reusedPowerampIdCannotChangeTheCachedSeedRecording() {
        assertFalse(
            CurrentPowerampBindingPolicy.matches(
                binding = binding(V2ActiveLibraryBindingEvidence.EXACT_V2_RECEIPT_SPAN),
                indexed = track(),
                current = provider(path = "/music/replacement.flac"),
                receipt = receipt(),
            ),
        )
    }

    @Test
    fun receiptForAnotherTrackCannotAuthenticateTheBinding() {
        assertFalse(
            CurrentPowerampBindingPolicy.matches(
                binding = binding(V2ActiveLibraryBindingEvidence.EXACT_V2_RECEIPT_SPAN),
                indexed = track(),
                current = provider(),
                receipt = receipt(trackId = 12L),
            ),
        )
    }

    @Test
    fun exactLegacyPathStillRequiresCompatibleRecordingEvidence() {
        val binding = binding(V2ActiveLibraryBindingEvidence.LEGACY_EXACT_ABSOLUTE_PATH)
        assertTrue(
            CurrentPowerampBindingPolicy.matches(binding, track(), provider(), receipt = null),
        )
        assertFalse(
            CurrentPowerampBindingPolicy.matches(
                binding,
                track(),
                provider(durationMs = 30_000, artist = "someone else", title = "other"),
                receipt = null,
            ),
        )
    }

    private fun binding(evidence: V2ActiveLibraryBindingEvidence) = V2ActiveLibraryBinding(
        trackId = 11L,
        powerampFileId = 22L,
        evidence = evidence,
    )

    private fun track() = EmbeddedTrack(
        id = 11L,
        metadataKey = "artist|album|song|120000",
        filenameKey = "artist|song",
        title = "Song",
        artist = "Artist",
        album = "Album",
        durationMs = 120_000,
        filePath = "/music/song.flac",
    )

    private fun provider(
        path: String = "/music/song.flac",
        durationMs: Int = 120_000,
        artist: String = "artist",
        title: String = "song",
    ) = PowerampFileEntry(
        id = 22L,
        artist = artist,
        album = "album",
        title = title,
        durationMs = durationMs,
        path = path,
        metadataKey = "$artist|album|$title|$durationMs",
        filenameKeys = setOf("song"),
    )

    private fun receipt(trackId: Long = 11L) = V2ProviderSpanReceipt(
        trackId = trackId,
        providerSpan = V2CommittedProviderSpan(
            normalizedPhysicalPath = "/music/song.flac",
            offsetMs = 0L,
            durationMs = 120_000L,
        ),
    )
}
