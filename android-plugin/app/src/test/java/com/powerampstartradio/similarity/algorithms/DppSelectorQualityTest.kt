package com.powerampstartradio.similarity.algorithms

import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

class DppSelectorQualityTest {
    @Test
    fun negativeCosineCannotBecomeNanForFractionalExponent() {
        assertEquals(0.0, DppSelector.qualityScore(-0.4f, 0.5f), 0.0)
    }

    @Test
    fun zeroExponentRemovesRelevancePullWithinRetrievedNeighborhood() {
        assertEquals(1.0, DppSelector.qualityScore(0.8f, 0f), 0.0)
        assertEquals(1.0, DppSelector.qualityScore(0f, 0f), 0.0)
    }

    @Test
    fun exponentChangesQualityMonotonicallyForUnitRangeRelevance() {
        val relevance = 0.7f
        val exponentOne = DppSelector.qualityScore(relevance, 1f)
        val exponentTwo = DppSelector.qualityScore(relevance, 2f)
        assertEquals(0.7, exponentOne, 1e-7)
        assertEquals(0.49, exponentTwo, 1e-6)
    }

    @Test
    fun completeDomainMaximumNormalizesQualityBeforeExponentiation() {
        assertEquals(
            1.0,
            DppSelector.qualityScore(
                relevance = 0.8f,
                exponent = 64f,
                completeDomainMaxRelevance = 0.8f.toDouble(),
            ),
            0.0,
        )
        assertEquals(
            Math.pow(0.5, 64.0),
            DppSelector.qualityScore(
                relevance = 0.4f,
                exponent = 64f,
                completeDomainMaxRelevance = 0.8f.toDouble(),
            ),
            1e-30,
        )
    }

    @Test
    fun invalidExponentFailsClosed() {
        assertThrows(IllegalArgumentException::class.java) {
            DppSelector.qualityScore(0.5f, Float.NaN)
        }
        assertThrows(IllegalArgumentException::class.java) {
            DppSelector.qualityScore(0.5f, -1f)
        }
    }
}
