package com.powerampstartradio.indexing

import org.junit.Assert.assertEquals
import org.junit.Test

class V2IndexingSelectionPolicyTest {
    @Test
    fun `ready count admits ordinary unknown duration but not CUE-shaped attention`() {
        val ready = track(1L, V2UnindexedDetectionKind.DEFINITELY_UNINDEXED, 10_000)
        val unknownOrdinary = track(3L, V2UnindexedDetectionKind.SOURCE_ATTENTION, 0)
        val hidden = track(4L, V2UnindexedDetectionKind.DEFINITELY_UNINDEXED, 10_000)
        val unknownCue = track(7L, V2UnindexedDetectionKind.SOURCE_ATTENTION, 0).copy(
            sourceHasLogicalOffsets = true,
        )
        val missingPath = track(8L, V2UnindexedDetectionKind.SOURCE_ATTENTION, 0).copy(
            path = null,
        )
        val importedTimingUnavailable = track(
            9L,
            V2UnindexedDetectionKind.LEGACY_PATH_TIMING_UNAVAILABLE,
            0,
        )

        assertEquals(
            setOf(1L, 3L),
            V2IndexingSelectionPolicy.readyTrackIds(
                listOf(
                    ready,
                    unknownOrdinary,
                    hidden,
                    unknownCue,
                    missingPath,
                    importedTimingUnavailable,
                ),
                hiddenIds = setOf(4L),
            ),
        )
    }

    @Test
    fun `search visibility cannot silently narrow the persisted job selection`() {
        val selected = setOf(10L, 20L, 30L)
        val searchResults = setOf(20L)

        assertEquals(selected, V2IndexingSelectionPolicy.selectedForJob(selected))
        assertEquals(
            setOf(10L, 30L),
            V2IndexingSelectionPolicy.deselectVisible(selected, searchResults),
        )
        assertEquals(
            selected,
            V2IndexingSelectionPolicy.selectVisible(setOf(10L, 30L), searchResults),
        )
    }

    @Test
    fun `never-index restore can select a nonready row without making it indexable`() {
        val nonReady = track(41L, V2UnindexedDetectionKind.SOURCE_ATTENTION, 0).copy(
            sourceHasLogicalOffsets = true,
        )

        assertEquals(false, V2IndexingSelectionPolicy.isReadyTrack(nonReady))
        assertEquals(
            false,
            V2IndexingSelectionPolicy.canToggleTrackRow(
                nonReady,
                allowNonReadySelection = false,
            ),
        )
        assertEquals(
            true,
            V2IndexingSelectionPolicy.canToggleTrackRow(
                nonReady,
                allowNonReadySelection = true,
            ),
        )
    }

    @Test
    fun `source attention confirmation is revalidated by current id and provider span`() {
        val requested = track(51L, V2UnindexedDetectionKind.SOURCE_ATTENTION, 0).copy(
            sourceHasLogicalOffsets = true,
        )
        val sameOccurrence = requested.copy(title = "Renamed metadata")

        assertEquals(
            sameOccurrence,
            V2IndexingSelectionPolicy.currentNonReadySourceAttentionMatch(
                requested,
                listOf(sameOccurrence),
            ),
        )
        assertEquals(
            null,
            V2IndexingSelectionPolicy.currentNonReadySourceAttentionMatch(
                requested,
                listOf(sameOccurrence.copy(powerampFileId = 52L)),
            ),
        )
        assertEquals(
            null,
            V2IndexingSelectionPolicy.currentNonReadySourceAttentionMatch(
                requested,
                listOf(sameOccurrence.copy(path = "/music/replaced.flac")),
            ),
        )
    }

    @Test
    fun `clean confirmation freezes the exact reviewed selection`() {
        val mutableSelection = mutableSetOf(3L, 5L, 8L)
        val confirmation = V2CleanDatabaseConfirmation.create(mutableSelection)
        mutableSelection.clear()

        assertEquals(3, confirmation.exactCount)
        assertEquals(setOf(3L, 5L, 8L), confirmation.trackIds)
    }

    @Test
    fun `never-index confirmation freezes the exact reviewed selection`() {
        val mutableSelection = mutableSetOf(13L, 21L, 34L)
        val confirmation = V2NeverIndexConfirmation.create(mutableSelection)
        mutableSelection.clear()

        assertEquals(3, confirmation.exactCount)
        assertEquals(setOf(13L, 21L, 34L), confirmation.trackIds)
    }

    @Test
    fun `attention confirmation freezes nonready source rows`() {
        val sourceAttention = mutableListOf(
            track(55L, V2UnindexedDetectionKind.SOURCE_ATTENTION, 0).copy(
                sourceHasLogicalOffsets = true,
            ),
        )
        val confirmation = V2NeverIndexAttentionConfirmation.create(
            failures = emptyList(),
            preflightAttention = emptyList(),
            sourceAttention = sourceAttention,
        )
        sourceAttention.clear()

        assertEquals(1, confirmation.exactCount)
        assertEquals(55L, confirmation.sourceAttention.single().powerampFileId)
    }

    @Test(expected = IllegalArgumentException::class)
    fun `never-index confirmation rejects an empty selection`() {
        V2NeverIndexConfirmation.create(emptySet())
    }

    private fun track(
        id: Long,
        kind: V2UnindexedDetectionKind,
        durationMs: Int,
    ) = NewTrackDetector.UnindexedTrack(
        powerampFileId = id,
        artist = "Artist",
        album = "Album",
        title = "Track $id",
        durationMs = durationMs,
        path = "/music/$id.flac",
        detectionKind = kind,
    )
}
