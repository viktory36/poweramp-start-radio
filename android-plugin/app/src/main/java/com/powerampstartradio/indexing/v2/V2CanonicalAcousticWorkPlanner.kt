package com.powerampstartradio.indexing.v2

data class V2CanonicalAcousticWorkKey(
    val stableTrackSpanId: String,
    val embeddingSpecId: String,
)

data class V2CanonicalAcousticWorkGroup(
    val key: V2CanonicalAcousticWorkKey,
    val canonical: SelectedTrackDescriptor,
    val members: List<SelectedTrackDescriptor>,
) {
    val aliases: List<SelectedTrackDescriptor> get() = members.drop(1)
}

/**
 * Deterministic acoustic deduplication only; every Poweramp locator remains a separate DB row.
 * Sampled fingerprints are mutation detectors, not equality proofs, so they never alias work.
 */
object V2CanonicalAcousticWorkPlanner {
    fun groups(spec: IndexingJobSpec): List<V2CanonicalAcousticWorkGroup> = spec.tracks
        .groupBy { descriptor ->
            V2CanonicalAcousticWorkKey(
                stableTrackSpanId = if (
                    descriptor.stableTrackSpanIdentity.strength ==
                    StableTrackSpanIdentityStrength.FULL_CONTENT_SHA256
                ) {
                    descriptor.stableTrackSpanIdentity.stableTrackSpanId
                } else {
                    "unverified-${descriptor.workId}"
                },
                embeddingSpecId = spec.embeddingSpec.specId,
            )
        }
        .values
        .map { descriptors -> descriptors.sortedBy(SelectedTrackDescriptor::ordinal) }
        .sortedBy { it.first().ordinal }
        .map { descriptors ->
            V2CanonicalAcousticWorkGroup(
                key = V2CanonicalAcousticWorkKey(
                    stableTrackSpanId =
                        descriptors.first().stableTrackSpanIdentity.stableTrackSpanId,
                    embeddingSpecId = spec.embeddingSpec.specId,
                ),
                canonical = descriptors.first(),
                members = descriptors,
            )
        }
}

/** Chooses physical work from durable group state without permanently privileging one locator. */
object V2CanonicalAcousticWorkExecutionPolicy {
    fun artifactDonor(
        group: V2CanonicalAcousticWorkGroup,
        trackByWorkId: Map<String, IndexingTrackLedger>,
        artifactKind: VerifiedArtifactKind,
    ): SelectedTrackDescriptor? = group.members.firstOrNull { descriptor ->
        trackByWorkId.getValue(descriptor.workId).verifiedArtifacts.any {
            it.kind == artifactKind
        }
    }

    fun leadersInState(
        ledger: IndexingJobLedger,
        state: IndexingTrackState,
        completedArtifactKind: VerifiedArtifactKind,
    ): List<SelectedTrackDescriptor> {
        val trackByWorkId = ledger.tracks.associateBy(IndexingTrackLedger::workId)
        return V2CanonicalAcousticWorkPlanner.groups(ledger.jobSpec).mapNotNull { group ->
            if (artifactDonor(group, trackByWorkId, completedArtifactKind) != null) {
                null
            } else {
                group.members.firstOrNull { descriptor ->
                    trackByWorkId.getValue(descriptor.workId).state == state
                }
            }
        }
    }
}
