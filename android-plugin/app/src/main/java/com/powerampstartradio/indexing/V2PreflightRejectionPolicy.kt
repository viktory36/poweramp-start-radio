package com.powerampstartradio.indexing

import com.powerampstartradio.indexing.v2.FailureDisposition
import com.powerampstartradio.indexing.v2.RetryTrigger
import com.powerampstartradio.indexing.v2.V2IndexingPreflightFailureCode
import com.powerampstartradio.indexing.v2.V2IndexingPreflightIntent
import com.powerampstartradio.indexing.v2.V2IndexingPreflightIntentState
import com.powerampstartradio.indexing.v2.V2IndexingPreflightRejectedRow
import com.powerampstartradio.indexing.v2.V2ProviderPathGroupSnapshot
import com.powerampstartradio.indexing.v2.V2ProviderSpanReceipt
import com.powerampstartradio.indexing.v2.V2StableProviderLexicalPathNormalizer
import com.powerampstartradio.poweramp.TrackNormalization

data class V2PreflightRejectedSpan(
    val jobId: String,
    val attemptCreatedAtEpochMs: Long,
    val originalPowerampFileId: Long,
    val providerSpan: V2ProviderSpanLocator,
    val code: V2IndexingPreflightFailureCode,
    val disposition: FailureDisposition,
    val retryTrigger: RetryTrigger,
    val diagnostic: String,
)

data class V2PreflightAttentionTrack(
    val rejection: V2PreflightRejectedSpan,
    val currentTrack: NewTrackDetector.UnindexedTrack,
) {
    val canTryAgain: Boolean
        get() = V2IndexingSelectionPolicy.isReadyTrack(currentTrack)
}

enum class V2PreflightAttentionAction {
    TRY_AGAIN,
    NEVER_INDEX,
}

/** Durable preflight history is reduced by provider span; numeric provider IDs are locators only. */
internal object V2PreflightRejectionPolicy {
    fun retainedRejections(
        intents: Collection<V2IndexingPreflightIntent>,
    ): List<V2PreflightRejectedSpan> {
        val latestBySpan = linkedMapOf<V2ProviderSpanLocator, V2PreflightRejectedSpan?>()
        intents.asSequence()
            .filter { intent ->
                intent.state == V2IndexingPreflightIntentState.MATERIALIZED ||
                    intent.state ==
                    V2IndexingPreflightIntentState.RESOLVED_WITHOUT_EXECUTABLE_ROWS
            }
            .sortedWith(
                compareBy<V2IndexingPreflightIntent> { it.createdAtEpochMs }
                    .thenBy { it.updatedAtEpochMs }
                    .thenBy { it.revision }
                    .thenBy { it.jobId },
            )
            .forEach { intent ->
                intent.rejected
                    .mapNotNull { row -> row.toRejectedSpan(intent) }
                    .groupBy(V2PreflightRejectedSpan::providerSpan)
                    .mapValues { (_, rows) -> rows.minBy(V2PreflightRejectedSpan::originalPowerampFileId) }
                    .toSortedMap(PROVIDER_SPAN_COMPARATOR)
                    .forEach { (span, rejected) -> latestBySpan[span] = rejected }

                // If one immutable materialized plan can execute this span, that newer outcome
                // supersedes an older rejection for the same physical occurrence.
                if (intent.state == V2IndexingPreflightIntentState.MATERIALIZED) {
                    intent.planned.mapNotNull(::providerSpanOrNull)
                        .distinct()
                        .sortedWith(PROVIDER_SPAN_COMPARATOR)
                        .forEach { span -> latestBySpan[span] = null }
                }
            }
        return latestBySpan.values.filterNotNull().sortedWith(
            compareBy<V2PreflightRejectedSpan> { it.providerSpan.normalizedPhysicalPath }
                .thenBy { it.providerSpan.offsetMs }
                .thenBy { it.providerSpan.durationMs }
                .thenBy { it.jobId },
        )
    }

    fun joinCurrentUnindexed(
        retained: Collection<V2PreflightRejectedSpan>,
        currentUnindexed: Collection<NewTrackDetector.UnindexedTrack>,
        suppressedSpans: Set<V2ProviderSpanLocator> = emptySet(),
    ): List<V2PreflightAttentionTrack> {
        val visibleRetained = retained.filter { it.providerSpan !in suppressedSpans }
        if (visibleRetained.isEmpty()) return emptyList()
        val targetSpans = visibleRetained.mapTo(hashSetOf(), V2PreflightRejectedSpan::providerSpan)
        val currentBySpan = currentUnindexed.asSequence()
            .mapNotNull { track ->
                V2TrackExclusionRepository.candidate(track)?.providerSpan?.let { it to track }
            }
            .filter { (span, _) -> span in targetSpans }
            .groupBy({ it.first }, { it.second })
            .mapValues { (_, tracks) -> tracks.minBy(NewTrackDetector.UnindexedTrack::powerampFileId) }
        return visibleRetained.asSequence()
            .mapNotNull { rejected ->
                currentBySpan[rejected.providerSpan]?.let { current ->
                    V2PreflightAttentionTrack(rejected, current)
                }
            }
            .sortedWith(
                compareBy<V2PreflightAttentionTrack> {
                    it.currentTrack.artist.lowercase()
                }.thenBy { it.currentTrack.album.lowercase() }
                    .thenBy { it.currentTrack.title.lowercase() }
                    .thenBy { it.rejection.providerSpan.normalizedPhysicalPath }
                    .thenBy { it.rejection.providerSpan.offsetMs }
                    .thenBy { it.rejection.providerSpan.durationMs },
            )
            .toList()
    }

    fun actionsFor(track: V2PreflightAttentionTrack): Set<V2PreflightAttentionAction> =
        buildSet {
            if (track.canTryAgain) add(V2PreflightAttentionAction.TRY_AGAIN)
            add(V2PreflightAttentionAction.NEVER_INDEX)
        }

    /** Targeted restart hydration for a running job; avoids a full imported-library scan. */
    fun currentUnindexedFromCompleteSnapshot(
        retained: Collection<V2PreflightRejectedSpan>,
        snapshot: V2ProviderPathGroupSnapshot,
        receipts: Collection<V2ProviderSpanReceipt>,
    ): List<NewTrackDetector.UnindexedTrack> {
        val acquisition = requireNotNull(snapshot.acquisitionEvidence) {
            "Poweramp snapshot has no cursor-completion evidence"
        }
        require(
            acquisition.cursorExhaustedNormally &&
                acquisition.rowCount == snapshot.groups.sumOf { it.rows.size },
        ) { "Poweramp snapshot is not complete" }
        val targetSpans = retained.mapTo(hashSetOf(), V2PreflightRejectedSpan::providerSpan)
        if (targetSpans.isEmpty()) return emptyList()
        val representedSpans = receipts.mapTo(hashSetOf()) { receipt ->
            V2ProviderSpanLocatorPolicy.create(
                receipt.providerSpan.normalizedPhysicalPath,
                receipt.providerSpan.offsetMs,
                receipt.providerSpan.durationMs,
            )
        }
        return snapshot.groups.asSequence().flatMap { group ->
            val referenceCount = maxOf(group.rows.size, 1)
            val hasLogicalOffsets = group.rows.any { it.offsetMs > 0L }
            val hasCueImageRow = group.rows.any { it.cueSourceImageFolderId != null }
            group.rows.asSequence().mapNotNull { row ->
                val span = V2ProviderSpanLocatorPolicy.create(
                    normalizedPhysicalPath = row.physicalPath,
                    offsetMs = row.offsetMs,
                    durationMs = row.durationMs,
                )
                val canonicalDurationMs = span.durationMs
                if (span !in targetSpans || span in representedSpans ||
                    row.cueSourceImageFolderId != null ||
                    canonicalDurationMs > Int.MAX_VALUE.toLong()
                ) return@mapNotNull null
                NewTrackDetector.UnindexedTrack(
                    powerampFileId = row.powerampFileId,
                    artist = TrackNormalization.normalizeArtist(row.artist),
                    album = TrackNormalization.normalizeAlbum(row.album),
                    title = TrackNormalization.normalizeTitle(row.title),
                    durationMs = canonicalDurationMs.toInt(),
                    path = row.physicalPath,
                    detectionKind = if (canonicalDurationMs == 0L) {
                        V2UnindexedDetectionKind.SOURCE_ATTENTION
                    } else {
                        V2UnindexedDetectionKind.DEFINITELY_UNINDEXED
                    },
                    offsetMs = row.offsetMs,
                    cueFolderId = row.cueSourceImageFolderId,
                    sourceReferenceCount = referenceCount,
                    sourceHasLogicalOffsets = hasLogicalOffsets,
                    sourceHasCueImageRow = hasCueImageRow,
                )
            }
        }.sortedBy(NewTrackDetector.UnindexedTrack::powerampFileId).toList()
    }

    private fun V2IndexingPreflightRejectedRow.toRejectedSpan(
        intent: V2IndexingPreflightIntent,
    ): V2PreflightRejectedSpan? {
        val span = providerSpanOrNull(selected) ?: return null
        return V2PreflightRejectedSpan(
            jobId = intent.jobId,
            attemptCreatedAtEpochMs = intent.createdAtEpochMs,
            originalPowerampFileId = selected.powerampFileId,
            providerSpan = span,
            code = code,
            disposition = disposition,
            retryTrigger = retryTrigger,
            diagnostic = diagnostic,
        )
    }

    private fun providerSpanOrNull(
        selected: com.powerampstartradio.indexing.v2.V2IndexingPreflightSelection,
    ): V2ProviderSpanLocator? {
        if (selected.offsetMs < 0L) return null
        val path = runCatching {
            V2StableProviderLexicalPathNormalizer.normalizeAbsolute(selected.providerPhysicalPath)
        }.getOrNull() ?: return null
        return V2ProviderSpanLocatorPolicy.create(path, selected.offsetMs, selected.durationMs)
    }

    private val PROVIDER_SPAN_COMPARATOR =
        compareBy<V2ProviderSpanLocator> { it.normalizedPhysicalPath }
            .thenBy { it.offsetMs }
            .thenBy { it.durationMs }
}
