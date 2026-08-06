package com.powerampstartradio.indexing.v2

import org.junit.Assert.assertThrows
import org.junit.Test

class V2LibraryMaintenancePublicationPolicyTest {
    @Test
    fun `successful exact subset remains successful after base generation is pruned`() {
        val base = manifest(
            origin = V2IndexGenerationOrigin.INDEXING_JOB,
            generationId = generationId('a'),
            baseGenerationId = null,
            trackCount = 100,
        )
        val published = manifest(
            origin = V2IndexGenerationOrigin.LIBRARY_MAINTENANCE,
            generationId = generationId('b'),
            baseGenerationId = base.generationId,
            trackCount = 97,
        )

        V2LibraryMaintenancePublicationPolicy.requireExactRetainedSubset(
            base = base,
            published = published,
            removedTrackCount = 3,
        )
    }

    @Test
    fun `postcondition rejects the wrong origin base or row delta`() {
        val base = manifest(
            origin = V2IndexGenerationOrigin.INDEXING_JOB,
            generationId = generationId('a'),
            baseGenerationId = null,
            trackCount = 100,
        )
        val exact = manifest(
            origin = V2IndexGenerationOrigin.LIBRARY_MAINTENANCE,
            generationId = generationId('b'),
            baseGenerationId = base.generationId,
            trackCount = 97,
        )

        listOf(
            exact.copy(origin = V2IndexGenerationOrigin.SERVER_MERGE),
            exact.copy(baseGenerationId = generationId('c')),
            exact.copy(trackCount = 98),
        ).forEach { invalid ->
            assertThrows(IllegalArgumentException::class.java) {
                V2LibraryMaintenancePublicationPolicy.requireExactRetainedSubset(
                    base = base,
                    published = invalid,
                    removedTrackCount = 3,
                )
            }
        }
    }

    private fun manifest(
        origin: V2IndexGenerationOrigin,
        generationId: String,
        baseGenerationId: String?,
        trackCount: Int,
    ): V2IndexGenerationManifest = V2IndexGenerationManifest(
        schemaVersion = 3,
        origin = origin,
        generationId = generationId,
        activationBindingId = "activation-binding-v3-" + "d".repeat(64),
        jobId = "job",
        jobSpecId = "job-spec-v5-" + "e".repeat(64),
        receiptEmbeddingSpec = embeddingSpec(),
        textRetrievalSpec = textRetrievalSpec(),
        baseGenerationId = baseGenerationId,
        rebuildDerivedIndexes = false,
        graphPolicy = V2IndexGenerationGraphPolicy.ABSENT,
        createdAtEpochMs = 0L,
        databaseRelativePath = "library.db",
        databaseByteLength = 1L,
        databaseSha256 = "1".repeat(64),
        databaseContentSha256 = "2".repeat(64),
        orderedTrackSetSha256 = "3".repeat(64),
        stableTrackUidCoverage = V2StableTrackUidCoverageBinding(
            coveredTrackCount = 0,
            uncoveredTrackCount = trackCount,
            uniqueStableTrackSpanCount = 0,
            fullContentIdentityCount = 0,
            sampledContentIdentityCount = 0,
            mappingSha256 = "4".repeat(64),
        ),
        embeddingCoverage = V2EmbeddingSpecCoverageBinding(
            totalTrackCount = trackCount,
            receiptBoundTrackCount = 0,
            receiptSpecTrackCounts = emptyMap(),
            compatibilityBase = V2CompatibilityBaseEmbeddingCoverageBinding(
                provenancePolicyId =
                    V2EmbeddingSpecCoverage.COMPATIBILITY_BASE_PROVENANCE_POLICY_ID,
                trackCount = trackCount,
                orderedContentSha256 = "5".repeat(64),
            ),
            mappingSha256 = "6".repeat(64),
        ),
        trackCount = trackCount,
        embeddingDimension = V2_CLAMP3_DIMENSION,
        embeddingRelativePath = "clamp3.emb",
        embeddingByteLength = 1L,
        embeddingSha256 = "7".repeat(64),
        graph = null,
    )

    private fun embeddingSpec() = EmbeddingSpecFingerprint(
        specId = "embedding-spec-v2-" + "d".repeat(64),
        preprocessingSpecId = "preprocessing-spec-v2-" + "8".repeat(64),
        decoderPolicyId = "decoder-policy-v2-" + "9".repeat(64),
        inferenceBackendPolicyId = "inference-backend-policy-v2-" + "a".repeat(64),
        outputDimension = V2_CLAMP3_DIMENSION,
        modelArtifactSha256 = mapOf(
            "mert" to "b".repeat(64),
            "clamp3_audio" to "c".repeat(64),
        ),
    )

    private fun textRetrievalSpec() = TextRetrievalSpecFingerprint(
        specId = "text-retrieval-spec-v1-" + "3".repeat(64),
        compatibleAudioEmbeddingSpecId = "embedding-spec-v2-" + "d".repeat(64),
        outputDimension = V2_CLAMP3_DIMENSION,
        textModelSha256 = "e".repeat(64),
        tokenizerModelSha256 = "f".repeat(64),
        tokenizerPolicyId = "tokenizer-policy-v1-" + "1".repeat(64),
        tokenizerRuntimeContractSha256 = "2".repeat(64),
        outputSpaceId = "clamp3-shared-space",
        inferenceBackendPolicyId = "text-inference-backend-v1",
    )

    private fun generationId(character: Char): String =
        "index-generation-v2-" + character.toString().repeat(64)
}
