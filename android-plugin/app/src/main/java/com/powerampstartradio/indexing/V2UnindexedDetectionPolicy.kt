package com.powerampstartradio.indexing

import com.powerampstartradio.indexing.v2.V2CommittedProviderSpan
import com.powerampstartradio.indexing.v2.V2ProviderSpanReceipt

enum class V2UnindexedDetectionKind {
    /** No exact receipt or imported-library conflict. Safe to select by default. */
    DEFINITELY_UNINDEXED,

    /** Poweramp cannot currently provide enough source information to plan this row. */
    SOURCE_ATTENTION,

    /** An imported row has the same reciprocal path, but Poweramp exposes no current timing. */
    LEGACY_PATH_TIMING_UNAVAILABLE,

}

data class V2ProviderOccurrence(
    val powerampFileId: Long,
    val providerSpan: V2CommittedProviderSpan,
    val isRawCueSourceImage: Boolean = false,
)

data class V2UnindexedOccurrence(
    val powerampFileId: Long,
    val kind: V2UnindexedDetectionKind,
)

/**
 * Pure occurrence policy. Exact receipts and separately proven one-to-one imported compatibility
 * may hide a provider row. Metadata resemblance never changes an unrepresented row into an
 * indexed one.
 */
internal object V2UnindexedDetectionPolicy {
    fun classify(
        providerOccurrences: Collection<V2ProviderOccurrence>,
        receipts: Collection<V2ProviderSpanReceipt>,
        compatibilityCoveredIds: Set<Long> = emptySet(),
        providerTimingUnavailableIds: Set<Long> = emptySet(),
    ): List<V2UnindexedOccurrence> {
        val representedSpans = receipts.mapTo(hashSetOf()) { it.providerSpan }
        return providerOccurrences
            .groupBy(V2ProviderOccurrence::providerSpan)
            .values
            .map { duplicates ->
                // A raw CUE source-image row is not indexable; prefer a playable logical row when
                // Poweramp exposes both for the exact same acoustic span.
                duplicates.minWith(
                    compareBy<V2ProviderOccurrence> { it.isRawCueSourceImage }
                        .thenBy { it.powerampFileId },
                ) to duplicates
            }
            .sortedBy { it.first.powerampFileId }
            .filterNot { (occurrence, _) -> occurrence.providerSpan in representedSpans }
            .filterNot { (occurrence, _) -> occurrence.isRawCueSourceImage }
            .filterNot { (_, duplicates) ->
                duplicates.any { it.powerampFileId in compatibilityCoveredIds }
            }
            .map { (occurrence, duplicates) ->
                val providerTimingUnavailable = duplicates.any {
                    it.powerampFileId in providerTimingUnavailableIds
                }
                V2UnindexedOccurrence(
                    powerampFileId = occurrence.powerampFileId,
                    kind = if (providerTimingUnavailable) {
                        V2UnindexedDetectionKind.LEGACY_PATH_TIMING_UNAVAILABLE
                    } else {
                        V2UnindexedDetectionKind.DEFINITELY_UNINDEXED
                    },
                )
            }
    }

    /** Cleanup can name only receipted rows whose exact occurrence is absent from a complete scan. */
    fun provablyAbsentTrackIds(
        receipts: Collection<V2ProviderSpanReceipt>,
        currentProviderSpans: Set<V2CommittedProviderSpan>,
    ): Set<Long> = receipts.asSequence()
        .filter { it.providerSpan !in currentProviderSpans }
        .mapTo(linkedSetOf()) { it.trackId }
}
