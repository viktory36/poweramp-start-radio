package com.powerampstartradio.similarity

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertThrows
import org.junit.Test

class RecommendationDomainEvidenceTest {
    @Test
    fun `bounded pool is measured inside the seed excluded identity domain`() {
        val evidence = RecommendationDomainEvidence(
            seedExcludedCandidateIdentityCount = 80_322,
            seedExcludedActiveIdentityCount = 100_000,
            resolvedCandidatePoolSize = 1_606,
        )

        assertEquals(80_322, evidence.seedExcludedCandidateIdentityCount)
        assertEquals(100_000, evidence.seedExcludedActiveIdentityCount)
        assertEquals(1_606, evidence.resolvedCandidatePoolSize)
    }

    @Test
    fun `full domain modes have no bounded pool`() {
        val evidence = RecommendationDomainEvidence(
            seedExcludedCandidateIdentityCount = 80_322,
            seedExcludedActiveIdentityCount = 80_322,
            resolvedCandidatePoolSize = null,
        )

        assertNull(evidence.resolvedCandidatePoolSize)
    }

    @Test
    fun `bounded pool cannot exceed its semantic identity domain`() {
        assertThrows(IllegalArgumentException::class.java) {
            RecommendationDomainEvidence(
                seedExcludedCandidateIdentityCount = 10,
                seedExcludedActiveIdentityCount = 10,
                resolvedCandidatePoolSize = 11,
            )
        }
    }

    @Test
    fun `added-date candidate domain cannot exceed full active rank domain`() {
        assertThrows(IllegalArgumentException::class.java) {
            RecommendationDomainEvidence(
                seedExcludedCandidateIdentityCount = 11,
                seedExcludedActiveIdentityCount = 10,
                resolvedCandidatePoolSize = null,
            )
        }
    }
}
