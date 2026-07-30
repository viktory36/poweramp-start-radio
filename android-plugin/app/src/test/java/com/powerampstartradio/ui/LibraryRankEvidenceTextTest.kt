package com.powerampstartradio.ui

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class LibraryRankEvidenceTextTest {
    @Test
    fun `rank text uses exact denominator and adaptive top fraction`() {
        assertEquals(
            "#8 of 80,000 \u00b7 top 0.01%",
            LibraryRankEvidenceText.rankWithTopFraction(8, 80_000),
        )
        assertEquals("#615 of 80,323 \u00b7 top 0.8%", LibraryRankEvidenceText.rankWithTopFraction(615, 80_323))
        assertNull(LibraryRankEvidenceText.rankWithTopFraction(12, null))
        assertNull(LibraryRankEvidenceText.rankWithTopFraction(12, 10))
        assertNull(LibraryRankEvidenceText.rank(0))
    }

    @Test
    fun `compact radio row keeps only exact nearest rank`() {
        assertEquals(
            "#615 nearest",
            LibraryRankEvidenceText.compactNearestRank(615, 80_323),
        )
        assertEquals(
            "#615 \u00b7 top 0.8%",
            LibraryRankEvidenceText.compactRankAndTopFraction(615, 80_323),
        )
        assertEquals(
            "#80,323 nearest",
            LibraryRankEvidenceText.compactNearestRank(80_323, 80_323),
        )
        assertNull(LibraryRankEvidenceText.compactNearestRank(12, null))
        assertNull(LibraryRankEvidenceText.compactNearestRank(12, 10))
    }

    @Test
    fun `upper cdf percentile becomes the first exact rank in its tie group`() {
        assertEquals(51, LibraryRankEvidenceText.rankFromUpperCdfPercentile(0.9f, 500))
        assertEquals("top 10.2%", LibraryRankEvidenceText.ingredientTopFraction(0.9f, 500))
        assertEquals(
            "#51 of 500 \u00b7 top 10.2%",
            LibraryRankEvidenceText.ingredientRankWithTopFraction(0.9f, 500),
        )
        assertEquals(
            "#51 \u00b7 top 10.2%",
            LibraryRankEvidenceText.compactIngredientRankAndTopFraction(0.9f, 500),
        )
        assertEquals(1, LibraryRankEvidenceText.rankFromUpperCdfPercentile(1f, 80_000))
        assertNull(LibraryRankEvidenceText.rankFromUpperCdfPercentile(0.90001f, 500))
        assertNull(LibraryRankEvidenceText.rankFromUpperCdfPercentile(Float.NaN, 80_000))
    }

    @Test
    fun `top fraction is a containing bound and never rounds below the exact rank`() {
        assertEquals("top 41.2%", LibraryRankEvidenceText.topFraction(32_935, 80_000))
        assertEquals("top 7.8%", LibraryRankEvidenceText.topFraction(6_206, 80_335))
        assertEquals("top <0.01%", LibraryRankEvidenceText.topFraction(1, 80_000))
        assertEquals("top 100%", LibraryRankEvidenceText.topFraction(80_000, 80_000))
    }
}
