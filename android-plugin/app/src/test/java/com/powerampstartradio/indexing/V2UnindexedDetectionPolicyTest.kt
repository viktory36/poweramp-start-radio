package com.powerampstartradio.indexing

import com.powerampstartradio.indexing.v2.V2CommittedProviderSpan
import com.powerampstartradio.indexing.v2.V2ProviderSpanReceipt
import org.junit.Assert.assertEquals
import org.junit.Test

class V2UnindexedDetectionPolicyTest {
    @Test
    fun `exact receipt hides only its one occurrence and never the same-tag remaster`() {
        val original = occurrence(1, "/music/album/original.flac", 240_000)
        val remaster = occurrence(2, "/music/album/remaster.flac", 240_000)

        val result = V2UnindexedDetectionPolicy.classify(
            providerOccurrences = listOf(original, remaster),
            receipts = listOf(V2ProviderSpanReceipt(100, original.providerSpan)),
        )

        assertEquals(
            listOf(V2UnindexedOccurrence(2, V2UnindexedDetectionKind.DEFINITELY_UNINDEXED)),
            result,
        )
    }

    @Test
    fun `unknown duration receipt hides zero occurrence but not later positive evidence`() {
        val unknown = occurrence(1, "/music/unknown.opus", 0)
        val receipt = V2ProviderSpanReceipt(100, unknown.providerSpan)

        assertEquals(
            emptyList<V2UnindexedOccurrence>(),
            V2UnindexedDetectionPolicy.classify(listOf(unknown), listOf(receipt)),
        )
        assertEquals(
            listOf(V2UnindexedOccurrence(2, V2UnindexedDetectionKind.DEFINITELY_UNINDEXED)),
            V2UnindexedDetectionPolicy.classify(
                listOf(unknown.copy(powerampFileId = 2, providerSpan = unknown.providerSpan.copy(durationMs = 90_000))),
                listOf(receipt),
            ),
        )
    }

    @Test
    fun `one legacy row cannot silently consume either of two duplicate files`() {
        val first = occurrence(10, "/music/a/copy.flac", 180_000)
        val second = occurrence(11, "/music/b/copy.flac", 180_000)

        assertEquals(
            listOf(
                V2UnindexedOccurrence(10, V2UnindexedDetectionKind.DEFINITELY_UNINDEXED),
                V2UnindexedOccurrence(11, V2UnindexedDetectionKind.DEFINITELY_UNINDEXED),
            ),
            V2UnindexedDetectionPolicy.classify(
                providerOccurrences = listOf(first, second),
                receipts = emptyList(),
            ),
        )
    }

    @Test
    fun `unmatched occurrence is ready while duplicate provider rows collapse by exact span`() {
        val firstProviderRow = occurrence(20, "/music/new.flac", 90_000)
        val duplicateProviderRow = firstProviderRow.copy(powerampFileId = 21)

        assertEquals(
            listOf(V2UnindexedOccurrence(20, V2UnindexedDetectionKind.DEFINITELY_UNINDEXED)),
            V2UnindexedDetectionPolicy.classify(
                providerOccurrences = listOf(firstProviderRow, duplicateProviderRow),
                receipts = emptyList(),
            ),
        )
    }

    @Test
    fun `playable row wins when a raw cue source row describes the same acoustic span`() {
        val rawCueRow = occurrence(19, "/music/image.flac", 90_000)
            .copy(isRawCueSourceImage = true)
        val playableRow = rawCueRow.copy(powerampFileId = 22, isRawCueSourceImage = false)

        assertEquals(
            listOf(V2UnindexedOccurrence(22, V2UnindexedDetectionKind.DEFINITELY_UNINDEXED)),
            V2UnindexedDetectionPolicy.classify(
                providerOccurrences = listOf(rawCueRow, playableRow),
                receipts = emptyList(),
            ),
        )
    }

    @Test
    fun `strong compatibility coverage hides a row without fabricating a receipt`() {
        val imported = occurrence(25, "/music/imported.flac", 90_000)

        assertEquals(
            emptyList<V2UnindexedOccurrence>(),
            V2UnindexedDetectionPolicy.classify(
                providerOccurrences = listOf(imported),
                receipts = emptyList(),
                compatibilityCoveredIds = setOf(imported.powerampFileId),
            ),
        )
    }

    @Test
    fun `standalone raw CUE source image is never offered as a track`() {
        val rawCueRow = occurrence(26, "/music/image.flac", 900_000)
            .copy(isRawCueSourceImage = true)

        assertEquals(
            emptyList<V2UnindexedOccurrence>(),
            V2UnindexedDetectionPolicy.classify(
                providerOccurrences = listOf(rawCueRow),
                receipts = emptyList(),
            ),
        )
    }

    @Test
    fun `cleanup returns only receipted rows with an absent exact span`() {
        val present = occurrence(30, "/music/present.flac", 100_000).providerSpan
        val absent = occurrence(31, "/music/absent.flac", 120_000).providerSpan

        assertEquals(
            setOf(301L),
            V2UnindexedDetectionPolicy.provablyAbsentTrackIds(
                receipts = listOf(
                    V2ProviderSpanReceipt(300, present),
                    V2ProviderSpanReceipt(301, absent),
                ),
                currentProviderSpans = setOf(present),
            ),
        )
    }

    private fun occurrence(id: Long, path: String, durationMs: Long) = V2ProviderOccurrence(
        powerampFileId = id,
        providerSpan = V2CommittedProviderSpan(path, 0L, durationMs),
    )
}
