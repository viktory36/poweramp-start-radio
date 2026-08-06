package com.powerampstartradio.services

import com.powerampstartradio.indexing.v2.V2ProviderPathRowEvidence
import com.powerampstartradio.poweramp.TrackNormalization
import com.powerampstartradio.widget.WidgetRadioSeedReference
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class RequiredSeedProviderBindingTest {
    @Test
    fun `provider display casing is preserved while match identity remains canonical`() {
        val providerPath =
            "/music/L. Subramaniam - Subramaniam in Moscow/01. Miss Melodia.flac"
        val binding = RequiredSeedProviderBinding.from(
            V2ProviderPathRowEvidence(
                powerampFileId = 13_319L,
                physicalPath = providerPath,
                providerPhysicalPath = providerPath,
                artist = "  L. Subramaniam  ",
                album = " Subramaniam in Moscow ",
                title = " Miss Melodia ",
                offsetMs = 0L,
                durationMs = 412_500L,
                cueSourceImageFolderId = null,
            ),
        )
        val reference = WidgetRadioSeedReference(
            powerampFileId = 13_319L,
            normalizedPath = TrackNormalization.normalizePath(providerPath),
            normalizedTitle = TrackNormalization.normalizeTitle("Miss Melodia"),
            displayTitle = "Miss Melodia",
        )

        assertEquals("l. subramaniam", binding.matchingEntry.artist)
        assertEquals("subramaniam in moscow", binding.matchingEntry.album)
        assertEquals("miss melodia", binding.matchingEntry.title)
        assertEquals(
            "l. subramaniam|subramaniam in moscow|miss melodia|412500",
            binding.matchingEntry.metadataKey,
        )
        assertTrue(reference.matchesProvider(binding.matchingEntry))

        val displayTrack = binding.toDisplayTrack(reference)
        assertEquals("L. Subramaniam", displayTrack.artist)
        assertEquals("Subramaniam in Moscow", displayTrack.album)
        assertEquals("Miss Melodia", displayTrack.title)
    }
}
