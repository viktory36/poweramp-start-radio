package com.powerampstartradio.poweramp

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class UniqueBestMatchPolicyTest {
    @Test
    fun equalBestCandidatesAreRejectedIndependentOfCursorOrder() {
        val candidates = listOf(
            "first" to 0,
            "second" to 0,
            "worse" to 1,
        )

        assertNull(UniqueBestMatchPolicy.choose(candidates))
        assertNull(UniqueBestMatchPolicy.choose(candidates.reversed()))
    }

    @Test
    fun uniquelyStrongerIdentityEvidenceWinsIndependentOfCursorOrder() {
        val candidates = listOf(
            "metadata-only" to 2,
            "exact-path" to 0,
            "duration-near" to 1,
        )

        assertEquals("exact-path", UniqueBestMatchPolicy.choose(candidates))
        assertEquals("exact-path", UniqueBestMatchPolicy.choose(candidates.reversed()))
    }

    @Test
    fun onePowerampRowCannotStandInForTwoActiveTrackIdentities() {
        assertEquals(
            listOf(null, null, 800L),
            IdentityConsistentFileResolutionPolicy.rejectAliasedIdentities(
                trackIds = listOf(7L, 8L, 9L),
                fileIds = listOf(700L, 700L, 800L),
            ),
        )
    }

    @Test
    fun repeatedOccurrencesOfOneActiveTrackMayReuseItsPowerampFileId() {
        assertEquals(
            listOf(700L, 700L),
            IdentityConsistentFileResolutionPolicy.rejectAliasedIdentities(
                trackIds = listOf(7L, 7L),
                fileIds = listOf(700L, 700L),
            ),
        )
    }
}
