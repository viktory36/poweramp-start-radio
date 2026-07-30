package com.powerampstartradio.indexing.v2

import java.io.File
import org.junit.Assert.assertFalse
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class V2IndexingStoragePolicyTest {
    @get:Rule
    val temporaryFolder = TemporaryFolder()

    @Test
    fun `estimate includes full publication staging artifacts and graph`() {
        val withoutGraph = V2IndexingStoragePolicy.estimate(
            active = active(),
            spans = listOf(span(mertWindows = 10, samples24k = 240_000L)),
            rebuildGraph = false,
            availableBytes = Long.MAX_VALUE,
        )
        val withGraph = V2IndexingStoragePolicy.estimate(
            active = active(),
            spans = listOf(span(mertWindows = 10, samples24k = 240_000L)),
            rebuildGraph = true,
            availableBytes = Long.MAX_VALUE,
        )

        assertTrue(withoutGraph.jobDatabaseBytes >= 380_000_000L)
        assertTrue(withoutGraph.publicationBytes > withoutGraph.jobDatabaseBytes)
        assertTrue(withoutGraph.durableArtifactBytes >= 11L * 3_072L)
        assertTrue(withGraph.publicationBytes > withoutGraph.publicationBytes)
    }

    @Test
    fun `insufficient capacity fails before execution`() {
        val estimate = V2IndexingStoragePolicy.estimate(
            active = active(),
            spans = listOf(span(mertWindows = 1_000, samples24k = 24_000_000L)),
            rebuildGraph = true,
            availableBytes = 1L,
        )
        assertFalse(estimate.hasCapacity)
        assertThrows(V2IndexingPreflightException::class.java) {
            V2IndexingStoragePolicy.requireCapacity(estimate)
        }
    }

    @Test
    fun `unknown duration storage is an explicit lower bound rather than zero exact work`() {
        val known = span(mertWindows = 10, samples24k = 240_000L)
        val knownOnly = V2IndexingStoragePolicy.estimate(
            active = active(),
            spans = listOf(known),
            rebuildGraph = false,
            availableBytes = Long.MAX_VALUE,
        )
        val withUnknown = V2IndexingStoragePolicy.estimate(
            active = active(),
            spans = listOf(
                known,
                V2PlannedStorageSpan(
                    mertWindows = null,
                    exactSampleCount24k = null,
                    sourceSampleCount = null,
                ),
            ),
            rebuildGraph = false,
            availableBytes = Long.MAX_VALUE,
        )

        assertTrue(knownOnly.isExact)
        assertFalse(withUnknown.isExact)
        assertEquals(1, withUnknown.unresolvedAudioSpanCount)
        assertEquals(knownOnly.peakPcmBytes, withUnknown.peakPcmBytes)
        assertTrue(withUnknown.requiredAdditionalBytes > knownOnly.requiredAdditionalBytes)
    }

    @Test
    fun `pcm admission includes native scratch and one track ahead overlap`() {
        val firstTargetSamples = 240_000L
        val secondTargetSamples = 480_000L
        val secondSourceSamples = 882_000L
        val estimate = V2IndexingStoragePolicy.estimate(
            active = active(),
            spans = listOf(
                span(mertWindows = 10, samples24k = firstTargetSamples),
                span(
                    mertWindows = 20,
                    samples24k = secondTargetSamples,
                    sourceSamples = secondSourceSamples,
                ),
            ),
            rebuildGraph = false,
            availableBytes = Long.MAX_VALUE,
        )

        assertEquals(
            (firstTargetSamples + secondTargetSamples + secondSourceSamples) * Float.SIZE_BYTES,
            estimate.peakPcmBytes,
        )
    }

    @Test
    fun `bootstrap admission reserves every immutable copy before reading`() {
        val estimate = V2GenerationMutationStoragePolicy.estimateBootstrapAdmission(
            sourceLength = 380_000_000L,
            availableBytes = Long.MAX_VALUE,
        )

        assertTrue(estimate.requiredAdditionalBytes > 4L * 380_000_000L)
    }

    @Test
    fun `bootstrap publication accounts exact database pemb and graph`() {
        val withoutGraph = V2GenerationMutationStoragePolicy.estimateBootstrapPublication(
            databaseBytes = 380_000_000L,
            trackCount = 80_000,
            embeddingDimension = 768,
            graphBytes = 0L,
            availableBytes = Long.MAX_VALUE,
        )
        val withGraph = V2GenerationMutationStoragePolicy.estimateBootstrapPublication(
            databaseBytes = 380_000_000L,
            trackCount = 80_000,
            embeddingDimension = 768,
            graphBytes = 4_000_000L,
            availableBytes = Long.MAX_VALUE,
        )

        assertTrue(withGraph.requiredAdditionalBytes > withoutGraph.requiredAdditionalBytes)
    }

    @Test
    fun `maintenance reserves two databases one pemb and two graph copies`() {
        val estimate = V2GenerationMutationStoragePolicy.estimateMaintenance(
            active = active(withGraph = true),
            availableBytes = Long.MAX_VALUE,
        )
        val rawMinimum = 2L * 380_000_000L + 245_760_016L + 2L * 4_000_000L

        assertTrue(estimate.requiredAdditionalBytes > rawMinimum)
    }

    @Test
    fun `mutation capacity error names the operation`() {
        val error = assertThrows(V2StorageCapacityException::class.java) {
            V2GenerationMutationStoragePolicy.requireCapacity(
                V2GenerationMutationStorageEstimate(10L, 9L),
                operation = "Library clean-up",
            )
        }

        assertTrue(error.message.orEmpty().contains("Library clean-up"))
        assertEquals(false, V2GenerationMutationStorageEstimate(10L, 9L).hasCapacity)
    }

    private fun active(withGraph: Boolean = false): V2ResolvedActiveIndexGeneration {
        val directory = temporaryFolder.root
        val database = File(directory, "library.db").apply { writeBytes(ByteArray(1)) }
        val embedding = File(directory, "clamp3.emb").apply { writeBytes(ByteArray(1)) }
        val graph = if (withGraph) {
            File(directory, "graph.bin").apply { writeBytes(ByteArray(1)) }
        } else {
            null
        }
        val manifest = V2IndexGenerationManifest(
            schemaVersion = 3,
            origin = V2IndexGenerationOrigin.BOOTSTRAP_COMPATIBILITY,
            generationId = "index-generation-v2-${"1".repeat(64)}",
            activationBindingId = "activation-binding-v3-${"2".repeat(64)}",
            jobId = V2IndexGenerationManifestPolicy.BOOTSTRAP_JOB_ID,
            jobSpecId = "job-spec-v5-${"3".repeat(64)}",
            receiptEmbeddingSpec = embeddingSpec(),
            textRetrievalSpec = textSpec(),
            baseGenerationId = null,
            rebuildDerivedIndexes = false,
            graphPolicy = V2IndexGenerationGraphPolicy.ABSENT,
            createdAtEpochMs = 0L,
            databaseRelativePath = "library.db",
            databaseByteLength = 380_000_000L,
            databaseSha256 = "4".repeat(64),
            databaseContentSha256 = "5".repeat(64),
            orderedTrackSetSha256 = "6".repeat(64),
            stableTrackUidCoverage = V2StableTrackUidCoverageBinding(0, 80_000, 0, 0, 0, "7".repeat(64)),
            embeddingCoverage = V2EmbeddingSpecCoverageBinding(
                totalTrackCount = 80_000,
                receiptBoundTrackCount = 0,
                receiptSpecTrackCounts = emptyMap(),
                compatibilityBase = V2CompatibilityBaseEmbeddingCoverageBinding(
                    V2EmbeddingSpecCoverage.COMPATIBILITY_BASE_PROVENANCE_POLICY_ID,
                    80_000,
                    "8".repeat(64),
                ),
                mappingSha256 = "9".repeat(64),
            ),
            trackCount = 80_000,
            embeddingDimension = 768,
            embeddingRelativePath = "clamp3.emb",
            embeddingByteLength = 245_760_016L,
            embeddingSha256 = "a".repeat(64),
            graph = graph?.let {
                V2IndexGenerationGraphBinding(
                    relativePath = "graph.bin",
                    byteLength = 4_000_000L,
                    sha256 = "1".repeat(64),
                    nodeCount = 80_000,
                    neighborsPerNode = 5,
                    orderedTrackSetSha256 = "6".repeat(64),
                )
            },
        )
        return V2ResolvedActiveIndexGeneration(
            manifest,
            "b".repeat(64),
            directory,
            database,
            embedding,
            graph,
        )
    }

    private fun embeddingSpec() = V2IndexingLedgerPlanner.createEmbeddingSpec(
        EmbeddingSpecInput(
            preprocessingSpecId = V2IndexingWorkPolicy.PREPROCESSING_SPEC_ID,
            decoderPolicyId = V2IndexingWorkPolicy.DECODER_POLICY_ID,
            inferenceBackendPolicyId = V2IndexingWorkPolicy.INFERENCE_BACKEND_POLICY_ID,
            outputDimension = 768,
            modelArtifactSha256 = mapOf("mert" to "c".repeat(64), "clamp3_audio" to "d".repeat(64)),
        ),
    )

    private fun textSpec(): TextRetrievalSpecFingerprint {
        val audio = embeddingSpec()
        return V2IndexingLedgerPlanner.createTextRetrievalSpec(
            TextRetrievalSpecInput(
                compatibleAudioEmbeddingSpecId = audio.specId,
                textModelSha256 = "e".repeat(64),
                tokenizerModelSha256 = V2IndexingWorkPolicy.TEXT_TOKENIZER_MODEL_SHA256,
                tokenizerPolicyId = V2IndexingWorkPolicy.TEXT_TOKENIZER_POLICY_ID,
                tokenizerRuntimeContractSha256 =
                    V2IndexingWorkPolicy.TEXT_TOKENIZER_RUNTIME_CONTRACT_SHA256,
                outputSpaceId = V2IndexingWorkPolicy.TEXT_OUTPUT_SPACE_ID,
                outputDimension = 768,
                inferenceBackendPolicyId =
                    V2IndexingWorkPolicy.TEXT_INFERENCE_BACKEND_POLICY_ID,
            ),
        )
    }

    private fun span(
        mertWindows: Int,
        samples24k: Long,
        sourceSamples: Long = samples24k,
    ) =
        V2PlannedStorageSpan(
            mertWindows = mertWindows,
            exactSampleCount24k = samples24k,
            sourceSampleCount = sourceSamples,
        )
}
