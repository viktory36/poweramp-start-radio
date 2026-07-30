package com.powerampstartradio.similarity

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class StableTrackSelectionPolicyTest {
    private val identities = mapOf(
        1L to "recording-a",
        2L to "recording-a",
        3L to "recording-b",
        4L to "recording-b",
    )
    private val equivalents = identities.entries
        .groupBy({ it.value }, { it.key })
    private val policy = StableTrackSelectionPolicy(
        identityForTrack = { trackId ->
            com.powerampstartradio.data.StableVisibleResultIdentity(
                identityToken = identities[trackId] ?: "legacy-$trackId",
                isCollapsibleRecording = identities.containsKey(trackId),
            )
        },
        equivalentTrackIds = { trackId ->
            identities[trackId]?.let { equivalents[it] }.orEmpty().ifEmpty { listOf(trackId) }
        },
    )

    @Test
    fun `seed exclusion expands to its complete recording equivalence class`() {
        assertEquals(setOf(1L, 2L, 9L), policy.exclusionClosure(listOf(1L, 9L)))
    }

    @Test
    fun `second row for an already selected recording is ineligible`() {
        assertFalse(policy.canSelect(2L, selectedIds = listOf(1L)))
        assertTrue(policy.canSelect(3L, selectedIds = listOf(1L)))
    }

    @Test
    fun `legacy rows remain distinct by generation scoped row id`() {
        assertFalse(policy.canSelect(9L, selectedIds = listOf(9L)))
        assertTrue(policy.canSelect(10L, selectedIds = listOf(9L)))
    }
}
