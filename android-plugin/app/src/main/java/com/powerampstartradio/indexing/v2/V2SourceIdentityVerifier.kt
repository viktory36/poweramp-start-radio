package com.powerampstartradio.indexing.v2

import java.io.File
import java.util.concurrent.CancellationException

/**
 * Revalidates an immutable source binding without treating unstable filesystem metadata as audio
 * identity. A full-content SHA-256 is the authority whenever stat evidence is inconclusive.
 */
internal class V2SourceIdentityVerifier(
    private val fingerprinter: V2SourceFingerprintProvider,
) {
    fun requireVerified(
        providerPhysicalPath: String,
        canonicalPath: String,
        powerampFileId: Long,
        planned: SourceFingerprint,
        exactContent: Boolean,
        onHashProgress: (completedBytes: Long, totalBytes: Long) -> Unit = { _, _ -> },
    ): File {
        val providerTarget = try {
            File(providerPhysicalPath).canonicalFile
        } catch (error: Exception) {
            throw V2SourceIdentityChangedException(
                "unable to resolve source target for row $powerampFileId: ${error.message}",
            )
        }
        if (V2IndexingLedgerIds.canonicalPath(providerTarget.path) != canonicalPath ||
            !providerTarget.isFile || !providerTarget.canRead()
        ) {
            throw V2SourceIdentityChangedException(
                "source target changed for Poweramp row $powerampFileId",
            )
        }
        if (providerTarget.length() != planned.sizeBytes) {
            throw V2SourceIdentityChangedException(
                "source size changed for Poweramp row $powerampFileId",
            )
        }

        val plannedModified = planned.lastModifiedEpochMs
        val statEvidenceMatches = plannedModified != null &&
            providerTarget.lastModified() == plannedModified
        if (exactContent || !statEvidenceMatches) {
            val current = try {
                fingerprinter.fingerprint(providerTarget, onHashProgress)
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (error: Exception) {
                throw V2SourceIdentityChangedException(
                    "unable to re-fingerprint ${providerTarget.path}: ${error.message}",
                )
            }
            if (!V2ExactSourceContentIdentity.matches(planned, current)) {
                throw V2SourceIdentityChangedException(
                    "source bytes changed for Poweramp row $powerampFileId",
                )
            }
        }
        return providerTarget
    }
}

/** `fileKey` and mtime are observations, not cross-process content identity. */
internal object V2ExactSourceContentIdentity {
    fun matches(planned: SourceFingerprint, current: SourceFingerprint): Boolean {
        val plannedSha256 = planned.fullContentSha256 ?: return false
        val currentSha256 = current.fullContentSha256 ?: return false
        return planned.sizeBytes == current.sizeBytes && plannedSha256 == currentSha256
    }
}

/** Only results present in the private generation need final source revalidation. */
internal object V2ActivationSourceSelection {
    fun committedDescriptors(ledger: IndexingJobLedger): List<SelectedTrackDescriptor> {
        val descriptorsByWorkId = ledger.jobSpec.tracks.associateBy { it.workId }
        require(descriptorsByWorkId.size == ledger.jobSpec.tracks.size) {
            "immutable indexing plan repeats work IDs"
        }
        return ledger.tracks
            .asSequence()
            .filter { it.state == IndexingTrackState.COMMITTED }
            .map { track ->
                descriptorsByWorkId[track.workId]
                    ?: throw InvalidIndexingLedgerException(
                        "committed work ${track.workId} has no immutable descriptor",
                    )
            }
            .toList()
    }
}
