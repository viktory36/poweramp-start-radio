package com.powerampstartradio.indexing

import com.powerampstartradio.indexing.v2.StableTrackSpanIdentity
import com.powerampstartradio.indexing.v2.StableTrackSpanIdentityStrength
import com.powerampstartradio.indexing.v2.IndexingJobLedger

/** Durable failure identity. A Poweramp numeric ID is deliberately not part of the key. */
internal data class V2UnresolvedFailureOccurrenceIdentity(
    val stableTrackSpanId: String,
    val providerSpan: V2ProviderSpanLocator,
)

/** Proof that two stable-span IDs still describe the same provider occurrence and source bytes. */
internal data class V2UnresolvedFailureContentSupersessionIdentity(
    val providerSpan: V2ProviderSpanLocator,
    val contentFingerprintSpecId: String,
    val fullContentSha256: String,
    val sourceSizeBytes: Long,
)

/** One ledger's latest outcome for an exact durable track occurrence. */
internal data class V2UnresolvedFailureOccurrenceOutcome<T : Any>(
    val identity: V2UnresolvedFailureOccurrenceIdentity,
    val contentSupersessionIdentity: V2UnresolvedFailureContentSupersessionIdentity? = null,
    val jobCreatedAtEpochMs: Long,
    val ledgerRevision: Long,
    val unresolvedValue: T?,
)

internal object V2UnresolvedFailureIdentityPolicy {
    fun identityOrNull(
        stableTrackSpanId: String,
        powerampFileId: Long,
        providerPhysicalPath: String?,
        offsetMs: Long,
        durationMs: Long,
    ): V2UnresolvedFailureOccurrenceIdentity? {
        if (stableTrackSpanId.isBlank()) return null
        val candidate = V2TrackExclusionRepository.candidate(
            powerampFileId = powerampFileId,
            physicalPath = providerPhysicalPath,
            offsetMs = offsetMs,
            durationMs = durationMs,
            stableTrackSpanId = stableTrackSpanId,
        ) ?: return null
        return V2UnresolvedFailureOccurrenceIdentity(
            stableTrackSpanId = stableTrackSpanId,
            providerSpan = candidate.providerSpan,
        )
    }

    /**
     * Whole-file EOS finalization legitimately changes the stable span ID. Only a full source hash
     * may bridge that change; a sampled fingerprint remains scoped to the exact stable span ID.
     */
    fun contentSupersessionIdentityOrNull(
        occurrenceIdentity: V2UnresolvedFailureOccurrenceIdentity,
        stableTrackSpanIdentity: StableTrackSpanIdentity,
    ): V2UnresolvedFailureContentSupersessionIdentity? {
        if (stableTrackSpanIdentity.strength !=
            StableTrackSpanIdentityStrength.FULL_CONTENT_SHA256
        ) return null
        val sha256 = stableTrackSpanIdentity.contentSha256.lowercase()
        if (!sha256.matches(Regex("^[0-9a-f]{64}$")) ||
            stableTrackSpanIdentity.contentFingerprintSpecId.isBlank() ||
            stableTrackSpanIdentity.sourceSizeBytes < 0L
        ) return null
        return V2UnresolvedFailureContentSupersessionIdentity(
            providerSpan = occurrenceIdentity.providerSpan,
            contentFingerprintSpecId = stableTrackSpanIdentity.contentFingerprintSpecId,
            fullContentSha256 = sha256,
            sourceSizeBytes = stableTrackSpanIdentity.sourceSizeBytes,
        )
    }

    /**
     * Current unindexed rows do not yet have content-derived stable IDs. Exact provider span is the
     * strongest identity available; a reused numeric ID can never hide an unrelated occurrence.
     */
    fun currentTrackIds(
        failures: Collection<V2UnresolvedFailureOccurrenceIdentity>,
        tracks: Collection<NewTrackDetector.UnindexedTrack>,
    ): Set<Long> {
        if (failures.isEmpty() || tracks.isEmpty()) return emptySet()
        val failedSpans = failures.mapTo(hashSetOf()) { it.providerSpan }
        return tracks.asSequence()
            .mapNotNull { track ->
                V2TrackExclusionRepository.candidate(track)
                    ?.takeIf { it.providerSpan in failedSpans }
                    ?.powerampFileId
            }
            .toCollection(linkedSetOf())
    }

    /**
     * A newer ledger outcome replaces history only for the same acoustic identity and provider
     * span. A null value is a resolved outcome and deliberately suppresses an older failure.
     */
    fun <T : Any> latestUnresolvedValues(
        outcomes: Iterable<V2UnresolvedFailureOccurrenceOutcome<T>>,
    ): List<T> = outcomes
        .groupBy { outcome ->
            outcome.contentSupersessionIdentity ?: outcome.identity
        }
        .values
        .mapNotNull { occurrenceOutcomes ->
            occurrenceOutcomes.maxWithOrNull(
                compareBy<V2UnresolvedFailureOccurrenceOutcome<T>>(
                    V2UnresolvedFailureOccurrenceOutcome<T>::jobCreatedAtEpochMs,
                    V2UnresolvedFailureOccurrenceOutcome<T>::ledgerRevision,
                ),
            )?.unresolvedValue
        }

    fun latestUnresolvedOccurrences(
        ledgers: Collection<IndexingJobLedger>,
    ): Set<V2UnresolvedFailureOccurrenceIdentity> {
        val outcomes = ledgers.asSequence().flatMap { ledger ->
            val descriptorById = ledger.jobSpec.tracks.associateBy { it.workId }
            ledger.tracks.asSequence().mapNotNull { track ->
                val descriptor = descriptorById[track.workId] ?: return@mapNotNull null
                val identity = identityOrNull(
                    stableTrackSpanId = descriptor.stableTrackSpanIdentity.stableTrackSpanId,
                    powerampFileId = descriptor.powerampFileId,
                    providerPhysicalPath = descriptor.providerRow.providerPhysicalPath,
                    offsetMs = descriptor.providerOffsetMs,
                    durationMs = descriptor.providerDurationMs,
                ) ?: return@mapNotNull null
                val failure = track.activeFailureId?.let { activeId ->
                    track.failures.firstOrNull { it.failureId == activeId }
                } ?: track.failures.maxByOrNull { it.lastOccurredAtEpochMs }
                V2UnresolvedFailureOccurrenceOutcome(
                    identity = identity,
                    contentSupersessionIdentity = contentSupersessionIdentityOrNull(
                        occurrenceIdentity = identity,
                        stableTrackSpanIdentity = descriptor.stableTrackSpanIdentity,
                    ),
                    jobCreatedAtEpochMs = ledger.jobSpec.createdAtEpochMs,
                    ledgerRevision = ledger.revision,
                    unresolvedValue = identity.takeIf {
                        isVisibleUnresolvedIndexingFailure(
                            trackState = track.state,
                            hasFailureEvidence = failure != null,
                        )
                    },
                )
            }
        }.toList()
        return latestUnresolvedValues(outcomes).toSet()
    }
}

internal fun unresolvedFailureCurrentTrackIds(
    failures: Collection<IndexingViewModel.FailedTrackUi>,
    tracks: Collection<NewTrackDetector.UnindexedTrack>,
): Set<Long> = V2UnresolvedFailureIdentityPolicy.currentTrackIds(
    failures = failures.mapNotNull { failure ->
        V2UnresolvedFailureIdentityPolicy.identityOrNull(
            stableTrackSpanId = failure.stableTrackSpanId,
            powerampFileId = failure.powerampFileId,
            providerPhysicalPath = failure.providerPhysicalPath,
            offsetMs = failure.offsetMs,
            durationMs = failure.durationMs,
        )
    },
    tracks = tracks,
)
