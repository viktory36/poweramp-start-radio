package com.powerampstartradio.ui

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class ArtistConstraintControlPolicyTest {
    @Test
    fun `spacing ceiling is exact across divisible and partial queues`() {
        assertEquals(1, ArtistConstraintControlPolicy.maximumOccurrencesFromSpacing(1, 0))
        assertEquals(10, ArtistConstraintControlPolicy.maximumOccurrencesFromSpacing(10, 0))
        assertEquals(5, ArtistConstraintControlPolicy.maximumOccurrencesFromSpacing(10, 1))
        assertEquals(4, ArtistConstraintControlPolicy.maximumOccurrencesFromSpacing(10, 2))
        assertEquals(3, ArtistConstraintControlPolicy.maximumOccurrencesFromSpacing(10, 3))
        assertEquals(1, ArtistConstraintControlPolicy.maximumOccurrencesFromSpacing(10, 9))
        assertEquals(1, ArtistConstraintControlPolicy.maximumOccurrencesFromSpacing(10, 20))
    }

    @Test
    fun `single recommendation has no subordinate artist control`() {
        val controls = ArtistConstraintControlPolicy.forRequest(
            recommendationCount = 1,
            maxPerArtist = 8,
            minArtistSpacing = 3,
        )

        assertFalse(controls.showMaximum)
        assertFalse(controls.showSpacing)
        assertEquals(
            "One recommendation cannot repeat an artist credit.",
            controls.evidenceLine,
        )
    }

    @Test
    fun `maximum one hides only the spacing control and preserves a way out`() {
        val controls = ArtistConstraintControlPolicy.forRequest(
            recommendationCount = 10,
            maxPerArtist = 1,
            minArtistSpacing = 20,
        )

        assertTrue(controls.showMaximum)
        assertFalse(controls.showSpacing)
        assertEquals(listOf(1, 2), controls.maximumOptions)
        assertTrue(controls.spacingOptions.isEmpty())
        assertEquals(
            "Maximum 1 already prevents the same artist credit appearing twice.",
            controls.evidenceLine,
        )
    }

    @Test
    fun `no-repeat spacing hides only the redundant maximum and retains its exact value`() {
        val controls = ArtistConstraintControlPolicy.forRequest(
            recommendationCount = 10,
            maxPerArtist = 8,
            minArtistSpacing = 20,
        )

        assertFalse(controls.showMaximum)
        assertTrue(controls.showSpacing)
        assertTrue(controls.maximumOptions.isEmpty())
        assertEquals((0..8).toList() + 20, controls.spacingOptions)
        assertEquals(
            "For 10 recommendations, spacing 20 permits at most one track with the same artist credit.",
            controls.evidenceLine,
        )
    }

    @Test
    fun `redundant maximum tail has one exact current representative`() {
        val controls = ArtistConstraintControlPolicy.forRequest(
            recommendationCount = 10,
            maxPerArtist = 8,
            minArtistSpacing = 3,
        )

        assertTrue(controls.showMaximum)
        assertTrue(controls.showSpacing)
        assertEquals(3, controls.spacingOccurrenceCeiling)
        assertEquals(listOf(1, 2, 8), controls.maximumOptions)
        assertEquals(
            "Together, these settings allow at most 3 tracks with the same artist credit.",
            controls.evidenceLine,
        )
    }

    @Test
    fun `default fifty-track request retains every supported maximum stop`() {
        val controls = ArtistConstraintControlPolicy.forRequest(
            recommendationCount = 50,
            maxPerArtist = 8,
            minArtistSpacing = 3,
        )

        assertEquals(13, controls.spacingOccurrenceCeiling)
        assertEquals((1..10).toList(), controls.maximumOptions)
        assertEquals((0..20).toList(), controls.spacingOptions)
        assertNull(controls.evidenceLine)
    }

    @Test
    fun `equal spacing and maximum ceilings do not repeat the visible maximum`() {
        val controls = ArtistConstraintControlPolicy.forRequest(
            recommendationCount = 30,
            maxPerArtist = 8,
            minArtistSpacing = 3,
        )

        assertEquals(8, controls.spacingOccurrenceCeiling)
        assertNull(controls.evidenceLine)
    }

    @Test
    fun `first redundant maximum is the only no-cap endpoint when current cap binds`() {
        val controls = ArtistConstraintControlPolicy.forRequest(
            recommendationCount = 10,
            maxPerArtist = 2,
            minArtistSpacing = 1,
        )

        assertEquals(5, controls.spacingOccurrenceCeiling)
        assertEquals(listOf(1, 2, 3, 4, 5), controls.maximumOptions)
        assertNull(controls.evidenceLine)
    }
}
