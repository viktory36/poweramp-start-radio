package com.powerampstartradio.indexing.v2

import android.util.AtomicFile
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.nio.ByteBuffer

/**
 * Publishes complete intermediate artifacts with AtomicFile. A ledger artifact is returned only
 * after the renamed file has been reopened and its exact length and SHA-256 have been verified.
 */
class AtomicV2ArtifactStore {
    fun publishMertFeatures(
        target: File,
        storageKey: String,
        features: Sequence<FloatArray>,
        expectedWindows: Int,
        finalizedAudioSpan: FinalizedAudioSpanEvidence,
        executionBoundary: VerifiedExecutionBoundaryEvidence,
        embeddingSpecId: String,
        sourceFingerprint: SourceFingerprint,
        verifiedAtEpochMs: Long,
    ): VerifiedArtifact {
        require(expectedWindows > 0) { "expectedWindows must be positive" }
        require(finalizedAudioSpan.authority !=
            V2AudioSpanAuthority.PROVISIONAL_END_OF_STREAM
        ) { "MERT features cannot be published from a provisional EOS span" }
        require(expectedWindows == finalizedAudioSpan.expectedWork.mertWindows) {
            "expectedWindows does not match the finalized acoustic span"
        }
        V2ArtifactIO.requireExecutionBoundaryMatches(finalizedAudioSpan, executionBoundary)
        require(storageKey.isNotBlank()) { "storageKey must not be blank" }
        require(embeddingSpecId.isNotBlank()) { "embeddingSpecId must not be blank" }
        require(verifiedAtEpochMs >= 0L) { "verifiedAtEpochMs must not be negative" }
        prepareParent(target)

        publishAtomically(target) { output ->
            var writtenWindows = 0
            for (feature in features) {
                require(writtenWindows < expectedWindows) {
                    "received more than $expectedWindows MERT windows"
                }
                val bytes = V2ArtifactIO.encodeMertWindow(feature)
                V2ExactChannelIO.writeFully(output.channel, ByteBuffer.wrap(bytes))
                writtenWindows++
            }
            require(writtenWindows == expectedWindows) {
                "received $writtenWindows MERT windows, expected $expectedWindows"
            }
        }

        val artifact = VerifiedArtifact(
            kind = VerifiedArtifactKind.MERT_FEATURES,
            storageKey = storageKey,
            byteLength = V2ArtifactIO.expectedMertByteLength(expectedWindows),
            sha256 = V2ArtifactIO.sha256(target),
            completedUnits = expectedWindows,
            plannedUnits = expectedWindows,
            embeddingSpecId = embeddingSpecId,
            sourceFingerprint = sourceFingerprint,
            verifiedAtEpochMs = verifiedAtEpochMs,
            executionBoundary = executionBoundary,
        )
        V2ArtifactIO.requireVerifiedFile(
            file = target,
            artifact = artifact,
            expectedKind = VerifiedArtifactKind.MERT_FEATURES,
            expectedStorageKey = storageKey,
            expectedEmbeddingSpecId = embeddingSpecId,
            expectedSourceFingerprint = sourceFingerprint,
            expectedPlannedUnits = expectedWindows,
        )
        return artifact
    }

    /** Streaming variant used by the real executor so a long track never retains all features. */
    fun publishMertFeaturesStreaming(
        target: File,
        storageKey: String,
        expectedWindows: Int,
        finalizedAudioSpan: FinalizedAudioSpanEvidence,
        executionBoundary: VerifiedExecutionBoundaryEvidence,
        embeddingSpecId: String,
        sourceFingerprint: SourceFingerprint,
        verificationCompletedAtEpochMs: () -> Long,
        produce: (writeFeature: (FloatArray) -> Unit) -> Int,
    ): VerifiedArtifact {
        require(expectedWindows > 0) { "expectedWindows must be positive" }
        require(finalizedAudioSpan.authority !=
            V2AudioSpanAuthority.PROVISIONAL_END_OF_STREAM
        ) { "MERT features cannot be published from a provisional EOS span" }
        require(expectedWindows == finalizedAudioSpan.expectedWork.mertWindows) {
            "expectedWindows does not match the finalized acoustic span"
        }
        V2ArtifactIO.requireExecutionBoundaryMatches(finalizedAudioSpan, executionBoundary)
        require(storageKey.isNotBlank()) { "storageKey must not be blank" }
        require(embeddingSpecId.isNotBlank()) { "embeddingSpecId must not be blank" }
        prepareParent(target)

        publishAtomically(target) { output ->
            var writtenWindows = 0
            val produced = produce { feature ->
                require(writtenWindows < expectedWindows) {
                    "received more than $expectedWindows MERT windows"
                }
                val bytes = V2ArtifactIO.encodeMertWindow(feature)
                V2ExactChannelIO.writeFully(output.channel, ByteBuffer.wrap(bytes))
                writtenWindows++
            }
            require(produced == writtenWindows) {
                "producer reported $produced MERT windows but wrote $writtenWindows"
            }
            require(writtenWindows == expectedWindows) {
                "received $writtenWindows MERT windows, expected $expectedWindows"
            }
        }

        val verificationCandidate = VerifiedArtifact(
            kind = VerifiedArtifactKind.MERT_FEATURES,
            storageKey = storageKey,
            byteLength = V2ArtifactIO.expectedMertByteLength(expectedWindows),
            sha256 = V2ArtifactIO.sha256(target),
            completedUnits = expectedWindows,
            plannedUnits = expectedWindows,
            embeddingSpecId = embeddingSpecId,
            sourceFingerprint = sourceFingerprint,
            // File verification accepts this placeholder; the ledger receives only the final copy.
            verifiedAtEpochMs = 0L,
            executionBoundary = executionBoundary,
        )
        return V2ArtifactVerificationCompletion.stamp(
            verify = {
                V2ArtifactIO.requireVerifiedFile(
                    file = target,
                    artifact = verificationCandidate,
                    expectedKind = VerifiedArtifactKind.MERT_FEATURES,
                    expectedStorageKey = storageKey,
                    expectedEmbeddingSpecId = embeddingSpecId,
                    expectedSourceFingerprint = sourceFingerprint,
                    expectedPlannedUnits = expectedWindows,
                )
                verificationCandidate
            },
            nowEpochMs = verificationCompletedAtEpochMs,
        )
    }

    fun publishClampVector(
        target: File,
        storageKey: String,
        vector: FloatArray,
        completedClampSegments: Int,
        embeddingSpecId: String,
        sourceFingerprint: SourceFingerprint,
        verifiedAtEpochMs: Long,
    ): VerifiedArtifact {
        require(completedClampSegments > 0) { "completedClampSegments must be positive" }
        require(storageKey.isNotBlank()) { "storageKey must not be blank" }
        require(embeddingSpecId.isNotBlank()) { "embeddingSpecId must not be blank" }
        require(verifiedAtEpochMs >= 0L) { "verifiedAtEpochMs must not be negative" }
        prepareParent(target)
        val bytes = V2Clamp3VectorCodec.encode(vector)

        publishAtomically(target) { output ->
            V2ExactChannelIO.writeFully(output.channel, ByteBuffer.wrap(bytes))
        }

        val artifact = VerifiedArtifact(
            kind = VerifiedArtifactKind.CLAMP_VECTOR,
            storageKey = storageKey,
            byteLength = V2_CLAMP3_BLOB_BYTES.toLong(),
            sha256 = V2ArtifactDigests.sha256(bytes),
            completedUnits = completedClampSegments,
            plannedUnits = completedClampSegments,
            embeddingSpecId = embeddingSpecId,
            sourceFingerprint = sourceFingerprint,
            verifiedAtEpochMs = verifiedAtEpochMs,
        )
        V2ArtifactIO.requireVerifiedFile(
            file = target,
            artifact = artifact,
            expectedKind = VerifiedArtifactKind.CLAMP_VECTOR,
            expectedStorageKey = storageKey,
            expectedEmbeddingSpecId = embeddingSpecId,
            expectedSourceFingerprint = sourceFingerprint,
            expectedPlannedUnits = completedClampSegments,
        )
        return artifact
    }

    fun publishMertAlias(
        source: File,
        sourceArtifact: VerifiedArtifact,
        target: File,
        targetStorageKey: String,
        targetSpan: FinalizedAudioSpanEvidence,
        targetSourceFingerprint: SourceFingerprint,
        verifiedAtEpochMs: Long,
    ): VerifiedArtifact {
        val boundary = requireNotNull(sourceArtifact.executionBoundary) {
            "canonical MERT artifact has no execution-boundary evidence"
        }
        V2ArtifactIO.requireVerifiedFile(
            file = source,
            artifact = sourceArtifact,
            expectedKind = VerifiedArtifactKind.MERT_FEATURES,
            expectedStorageKey = sourceArtifact.storageKey,
            expectedEmbeddingSpecId = sourceArtifact.embeddingSpecId,
            expectedSourceFingerprint = sourceArtifact.sourceFingerprint,
            expectedPlannedUnits = sourceArtifact.plannedUnits,
        )
        require(sourceArtifact.plannedUnits == targetSpan.expectedWork.mertWindows) {
            "canonical MERT artifact has a different acoustic work plan"
        }
        V2ArtifactIO.requireExecutionBoundaryMatches(targetSpan, boundary)
        copyAtomically(source, target)
        return sourceArtifact.copy(
            storageKey = targetStorageKey,
            sourceFingerprint = targetSourceFingerprint,
            verifiedAtEpochMs = verifiedAtEpochMs,
        ).also { alias ->
            V2ArtifactIO.requireVerifiedFile(
                file = target,
                artifact = alias,
                expectedKind = VerifiedArtifactKind.MERT_FEATURES,
                expectedStorageKey = targetStorageKey,
                expectedEmbeddingSpecId = alias.embeddingSpecId,
                expectedSourceFingerprint = targetSourceFingerprint,
                expectedPlannedUnits = targetSpan.expectedWork.mertWindows,
            )
        }
    }

    fun publishClampAlias(
        source: File,
        sourceArtifact: VerifiedArtifact,
        target: File,
        targetStorageKey: String,
        targetSourceFingerprint: SourceFingerprint,
        expectedClampSegments: Int,
        verifiedAtEpochMs: Long,
    ): VerifiedArtifact {
        V2ArtifactIO.requireVerifiedFile(
            file = source,
            artifact = sourceArtifact,
            expectedKind = VerifiedArtifactKind.CLAMP_VECTOR,
            expectedStorageKey = sourceArtifact.storageKey,
            expectedEmbeddingSpecId = sourceArtifact.embeddingSpecId,
            expectedSourceFingerprint = sourceArtifact.sourceFingerprint,
            expectedPlannedUnits = expectedClampSegments,
        )
        copyAtomically(source, target)
        return sourceArtifact.copy(
            storageKey = targetStorageKey,
            sourceFingerprint = targetSourceFingerprint,
            verifiedAtEpochMs = verifiedAtEpochMs,
        ).also { alias ->
            V2ArtifactIO.requireVerifiedFile(
                file = target,
                artifact = alias,
                expectedKind = VerifiedArtifactKind.CLAMP_VECTOR,
                expectedStorageKey = targetStorageKey,
                expectedEmbeddingSpecId = alias.embeddingSpecId,
                expectedSourceFingerprint = targetSourceFingerprint,
                expectedPlannedUnits = expectedClampSegments,
            )
        }
    }

    private fun copyAtomically(source: File, target: File) {
        require(source.isFile) { "canonical artifact is missing: $source" }
        prepareParent(target)
        publishAtomically(target) { output ->
            FileInputStream(source).channel.use { input ->
                val buffer = ByteBuffer.allocateDirect(64 * 1024)
                while (true) {
                    buffer.clear()
                    val read = input.read(buffer)
                    if (read < 0) break
                    if (read == 0) continue
                    buffer.flip()
                    V2ExactChannelIO.writeFully(output.channel, buffer)
                }
            }
        }
    }

    private fun publishAtomically(
        target: File,
        write: (FileOutputStream) -> Unit,
    ) {
        val atomicFile = AtomicFile(target)
        var output: FileOutputStream? = null
        try {
            output = atomicFile.startWrite()
            write(output)
            output.channel.force(true)
            atomicFile.finishWrite(output)
            output = null
        } catch (failure: Throwable) {
            output?.let { openOutput ->
                try {
                    atomicFile.failWrite(openOutput)
                } catch (rollbackFailure: Throwable) {
                    failure.addSuppressed(rollbackFailure)
                }
            }
            throw failure
        }
    }

    private fun prepareParent(target: File) {
        require(!target.exists() || target.isFile) { "artifact target is not a file: $target" }
        val parent = target.absoluteFile.parentFile
            ?: throw IllegalArgumentException("artifact target has no parent: $target")
        require(parent.isDirectory || parent.mkdirs()) {
            "could not create artifact directory: $parent"
        }
    }
}
