package com.powerampstartradio.indexing

import android.content.Context
import com.powerampstartradio.indexing.v2.V2IndexingJobRepository
import java.nio.ByteBuffer
import java.security.MessageDigest

internal data class V2IndexingAttentionHistory(
    val unresolvedFailures: Set<V2UnresolvedFailureOccurrenceIdentity>,
    val retainedPreflightRejections: List<V2PreflightRejectedSpan>,
    val preflightRetryChoices: Set<V2ProviderSpanLocator>,
    val fingerprint: String,
)

/** Reads only durable indexing decisions needed to make Settings agree with Manage Tracks. */
internal class V2IndexingAttentionHistorySource(context: Context) {
    private val app = context.applicationContext

    fun load(): V2IndexingAttentionHistory {
        val unresolved = V2UnresolvedFailureIdentityPolicy.latestUnresolvedOccurrences(
            V2IndexingJobRepository.get(app).refresh(),
        )
        val retainedPreflight = V2PreflightRejectionPolicy.retainedRejections(
            V2AtomicPreflightRejectionHistorySource(app.filesDir).inspect().requireComplete(),
        )
        val retryChoices = V2PreflightRetryChoiceRepository(app).load()
        return V2IndexingAttentionHistory(
            unresolvedFailures = unresolved,
            retainedPreflightRejections = retainedPreflight,
            preflightRetryChoices = retryChoices,
            fingerprint = fingerprint(unresolved, retainedPreflight, retryChoices),
        )
    }

    private fun fingerprint(
        unresolved: Set<V2UnresolvedFailureOccurrenceIdentity>,
        retainedPreflight: List<V2PreflightRejectedSpan>,
        retryChoices: Set<V2ProviderSpanLocator>,
    ): String {
        val digest = MessageDigest.getInstance("SHA-256")
        update(digest, "indexing-attention-history-v1")
        unresolved.sortedWith(compareBy<V2UnresolvedFailureOccurrenceIdentity> {
            it.providerSpan.normalizedPhysicalPath
        }.thenBy { it.providerSpan.offsetMs }.thenBy { it.providerSpan.durationMs }
            .thenBy { it.stableTrackSpanId }).forEach { identity ->
            update(digest, "failure")
            update(digest, identity.stableTrackSpanId)
            update(digest, identity.providerSpan)
        }
        retainedPreflight.map(V2PreflightRejectedSpan::providerSpan)
            .distinct()
            .sortedWith(PROVIDER_SPAN_COMPARATOR)
            .forEach { span ->
                update(digest, "preflight")
                update(digest, span)
            }
        retryChoices.sortedWith(PROVIDER_SPAN_COMPARATOR).forEach { span ->
            update(digest, "retry")
            update(digest, span)
        }
        return digest.digest().joinToString("") { byte -> "%02x".format(byte) }
    }

    private fun update(digest: MessageDigest, span: V2ProviderSpanLocator) {
        update(digest, span.normalizedPhysicalPath)
        update(digest, span.offsetMs.toString())
        update(digest, span.durationMs.toString())
    }

    private fun update(digest: MessageDigest, value: String) {
        val bytes = value.toByteArray(Charsets.UTF_8)
        digest.update(ByteBuffer.allocate(Int.SIZE_BYTES).putInt(bytes.size).array())
        digest.update(bytes)
    }

    private companion object {
        val PROVIDER_SPAN_COMPARATOR = compareBy<V2ProviderSpanLocator> {
            it.normalizedPhysicalPath
        }.thenBy { it.offsetMs }.thenBy { it.durationMs }
    }
}

internal object V2IndexingReadinessPolicy {
    fun readyTrackIds(
        tracks: Collection<NewTrackDetector.UnindexedTrack>,
        exclusions: V2ResolvedTrackExclusions,
        attentionHistory: V2IndexingAttentionHistory,
    ): Set<Long> {
        // V2 deliberately discarded V1's implicit ignored list. Only an explicit Never-index
        // choice remains a hidden track in Manage Tracks.
        val hiddenIds = exclusions.neverIds
        val suppressedAttentionSpans = attentionHistory.preflightRetryChoices +
            exclusions.never.mapTo(linkedSetOf(), V2PersistedTrackExclusion::providerSpan)
        val unresolvedFailureIds = V2UnresolvedFailureIdentityPolicy.currentTrackIds(
            failures = attentionHistory.unresolvedFailures.filterNot {
                it.providerSpan in suppressedAttentionSpans
            },
            tracks = tracks,
        )
        val preflightAttentionIds = V2PreflightRejectionPolicy.joinCurrentUnindexed(
            retained = attentionHistory.retainedPreflightRejections,
            currentUnindexed = tracks,
            suppressedSpans = suppressedAttentionSpans,
        ).asSequence()
            .map { it.currentTrack.powerampFileId }
            .filterNot { it in unresolvedFailureIds }
            .toSet()
        return V2IndexingSelectionPolicy.readyTrackIds(tracks, hiddenIds) -
            unresolvedFailureIds - preflightAttentionIds
    }
}
