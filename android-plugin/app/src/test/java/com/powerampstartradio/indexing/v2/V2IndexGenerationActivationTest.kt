package com.powerampstartradio.indexing.v2

import com.google.gson.Gson
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

class V2IndexGenerationActivationTest {
    @get:Rule
    val temporaryFolder = TemporaryFolder()

    @Test
    fun `PEMB binds exact ordered ids and vector bytes`() {
        val first = temporaryFolder.newFile("first.emb")
        val second = temporaryFolder.newFile("second.emb")
        val firstBinding = V2EmbeddingGenerationFile.write(
            source(10L to unitVector(0), 20L to unitVector(1)),
            first,
        )
        val secondBinding = V2EmbeddingGenerationFile.write(
            source(10L to unitVector(0), 20L to unitVector(2)),
            second,
        )

        assertEquals(firstBinding, V2EmbeddingGenerationFile.inspect(first))
        assertEquals(firstBinding.orderedTrackSetSha256, secondBinding.orderedTrackSetSha256)
        assertNotEquals(firstBinding.databaseContentSha256, secondBinding.databaseContentSha256)
        assertNotEquals(firstBinding.fileSha256, secondBinding.fileSha256)
        assertEquals(
            V2DatabaseEmbeddingBinding(
                trackCount = firstBinding.trackCount,
                dimension = firstBinding.dimension,
                orderedTrackSetSha256 = firstBinding.orderedTrackSetSha256,
                databaseContentSha256 = firstBinding.databaseContentSha256,
            ),
            V2EmbeddingGenerationFile.digest(
                source(10L to unitVector(0), 20L to unitVector(1)),
            ),
        )
    }

    @Test
    fun `PEMB rejects duplicate or unordered ids and malformed vectors`() {
        assertThrows(IllegalArgumentException::class.java) {
            V2EmbeddingGenerationFile.write(
                source(20L to unitVector(0), 10L to unitVector(1)),
                temporaryFolder.newFile("unordered.emb"),
            )
        }
        assertThrows(IllegalArgumentException::class.java) {
            V2EmbeddingGenerationFile.write(
                source(10L to FloatArray(V2_CLAMP3_DIMENSION)),
                temporaryFolder.newFile("zero.emb"),
            )
        }
    }

    @Test
    fun `graph binding requires the identical ordered track set`() {
        val matching = temporaryFolder.newFile("matching.graph")
        writeGraph(matching, longArrayOf(10L, 20L), neighbors = 1)
        val graph = V2GraphGenerationFile.inspect(matching)
        val embedding = V2EmbeddingGenerationFile.write(
            source(10L to unitVector(0), 20L to unitVector(1)),
            temporaryFolder.newFile("graph.emb"),
        )
        assertEquals(embedding.orderedTrackSetSha256, graph.orderedTrackSetSha256)

        val mismatched = temporaryFolder.newFile("mismatched.graph")
        writeGraph(mismatched, longArrayOf(10L, 30L), neighbors = 1)
        assertNotEquals(
            embedding.orderedTrackSetSha256,
            V2GraphGenerationFile.inspect(mismatched).orderedTrackSetSha256,
        )

        val corruptNeighbor = temporaryFolder.newFile("corrupt-neighbor.graph")
        writeGraph(corruptNeighbor, longArrayOf(10L, 20L), neighbors = 1, neighborIndex = 2)
        assertThrows(IllegalArgumentException::class.java) {
            V2GraphGenerationFile.inspect(corruptNeighbor)
        }
    }

    @Test
    fun `manifest identity binds optional graph and ignores display timestamp`() {
        val binding = V2EmbeddingGenerationFile.write(
            source(10L to unitVector(0)),
            temporaryFolder.newFile("manifest.emb"),
        )
        val provisional = manifest(binding)
        val withoutGraph = V2IndexGenerationIdentity.generationId(provisional)
        assertEquals(
            withoutGraph,
            V2IndexGenerationIdentity.generationId(provisional.copy(createdAtEpochMs = 999L)),
        )
        val graph = V2IndexGenerationGraphBinding(
            relativePath = "graph.bin",
            byteLength = 24L,
            sha256 = "d".repeat(64),
            nodeCount = 1,
            neighborsPerNode = 1,
            orderedTrackSetSha256 = binding.orderedTrackSetSha256,
        )
        assertNotEquals(
            withoutGraph,
            V2IndexGenerationIdentity.generationId(
                provisional.copy(
                    rebuildDerivedIndexes = true,
                    graphPolicy = V2IndexGenerationGraphPolicy.EXPLICIT_REBUILD,
                    graph = graph,
                ),
            ),
        )
        assertNull(provisional.graph)
    }

    @Test
    fun `bootstrap origin binds identity and requires entirely unclaimed compatibility coverage`() {
        val binding = V2EmbeddingGenerationFile.write(
            source(10L to unitVector(0), 20L to unitVector(1)),
            temporaryFolder.newFile("bootstrap-policy.emb"),
        )
        val indexing = manifest(binding)
        val compatibility = V2CompatibilityBaseEmbeddingCoverageBinding(
            provenancePolicyId =
                V2EmbeddingSpecCoverage.COMPATIBILITY_BASE_PROVENANCE_POLICY_ID,
            trackCount = binding.trackCount,
            orderedContentSha256 = "8".repeat(64),
        )
        val bootstrapDraft = indexing.copy(
            origin = V2IndexGenerationOrigin.BOOTSTRAP_COMPATIBILITY,
            generationId = "",
            activationBindingId = "activation-binding-v3-" + "0".repeat(64),
            jobId = V2IndexGenerationManifestPolicy.BOOTSTRAP_JOB_ID,
            jobSpecId = V2IndexGenerationIdentity.bootstrapSpecId(
                indexing.receiptEmbeddingSpec,
                indexing.textRetrievalSpec,
            ),
            createdAtEpochMs = 0L,
            stableTrackUidCoverage = indexing.stableTrackUidCoverage.copy(
                coveredTrackCount = 0,
                uncoveredTrackCount = binding.trackCount,
                uniqueStableTrackSpanCount = 0,
                fullContentIdentityCount = 0,
                sampledContentIdentityCount = 0,
            ),
            embeddingCoverage = indexing.embeddingCoverage.copy(
                receiptBoundTrackCount = 0,
                receiptSpecTrackCounts = emptyMap(),
                compatibilityBase = compatibility,
            ),
        )
        V2IndexGenerationManifestPolicy.requireValidProvenance(bootstrapDraft)
        V2IndexGenerationManifestPolicy.requireValidCoverage(bootstrapDraft)
        assertNotEquals(
            V2IndexGenerationIdentity.activationBindingId(indexing),
            V2IndexGenerationIdentity.activationBindingId(bootstrapDraft),
        )

        assertThrows(IllegalArgumentException::class.java) {
            V2IndexGenerationManifestPolicy.requireValidCoverage(
                bootstrapDraft.copy(embeddingCoverage = indexing.embeddingCoverage),
            )
        }
        assertThrows(IllegalArgumentException::class.java) {
            V2IndexGenerationManifestPolicy.requireValidCoverage(
                indexing.copy(embeddingCoverage = bootstrapDraft.embeddingCoverage),
            )
        }
    }

    @Test
    fun `maintenance policy accepts exact receipt-bound and mixed retained coverage`() {
        val binding = V2EmbeddingGenerationFile.write(
            source(10L to unitVector(0), 20L to unitVector(1)),
            temporaryFolder.newFile("maintenance-policy.emb"),
        )
        val indexing = manifest(binding)
        val baseId = "index-generation-v2-" + "4".repeat(64)
        val mixed = indexing.copy(
            origin = V2IndexGenerationOrigin.LIBRARY_MAINTENANCE,
            generationId = "",
            jobId = V2IndexGenerationManifestPolicy.MAINTENANCE_JOB_ID,
            jobSpecId = V2IndexGenerationIdentity.maintenanceSpecId(
                baseId,
                indexing.receiptEmbeddingSpec,
                indexing.textRetrievalSpec,
            ),
            baseGenerationId = baseId,
            createdAtEpochMs = 0L,
            stableTrackUidCoverage = indexing.stableTrackUidCoverage.copy(
                coveredTrackCount = 1,
                uncoveredTrackCount = 1,
                uniqueStableTrackSpanCount = 1,
                fullContentIdentityCount = 0,
                sampledContentIdentityCount = 1,
            ),
            embeddingCoverage = indexing.embeddingCoverage.copy(
                receiptBoundTrackCount = 1,
                receiptSpecTrackCounts = mapOf(indexing.receiptEmbeddingSpec.specId to 1),
                compatibilityBase = V2CompatibilityBaseEmbeddingCoverageBinding(
                    provenancePolicyId =
                        V2EmbeddingSpecCoverage.COMPATIBILITY_BASE_PROVENANCE_POLICY_ID,
                    trackCount = 1,
                    orderedContentSha256 = "5".repeat(64),
                ),
            ),
        )
        V2IndexGenerationManifestPolicy.requireValidProvenance(mixed)
        V2IndexGenerationManifestPolicy.requireValidCoverage(mixed)

        val repaired = mixed.copy(
            graphPolicy = V2IndexGenerationGraphPolicy.BASE_BOUND_DELETION_REPAIR,
            graph = V2IndexGenerationGraphBinding(
                relativePath = V2GraphGenerationFile.GRAPH_FILE,
                byteLength = 64L,
                sha256 = "8".repeat(64),
                nodeCount = mixed.trackCount,
                neighborsPerNode = 1,
                orderedTrackSetSha256 = mixed.orderedTrackSetSha256,
            ),
        )
        V2IndexGenerationManifestPolicy.requireValidProvenance(repaired)
        assertThrows(IllegalArgumentException::class.java) {
            V2IndexGenerationManifestPolicy.requireValidProvenance(
                repaired.copy(graph = null),
            )
        }

        val receiptBound = mixed.copy(
            stableTrackUidCoverage = indexing.stableTrackUidCoverage,
            embeddingCoverage = indexing.embeddingCoverage,
        )
        V2IndexGenerationManifestPolicy.requireValidCoverage(receiptBound)

        assertThrows(IllegalArgumentException::class.java) {
            V2IndexGenerationManifestPolicy.requireValidCoverage(
                mixed.copy(
                    embeddingCoverage = mixed.embeddingCoverage.copy(
                        receiptSpecTrackCounts = mapOf("embedding-spec-v2-" + "9".repeat(64) to 1),
                    ),
                ),
            )
        }
    }

    @Test
    fun `server merge provenance requires a base-bound addition graph update`() {
        val binding = V2EmbeddingGenerationFile.write(
            source(10L to unitVector(0), 20L to unitVector(1)),
            temporaryFolder.newFile("server-merge-policy.emb"),
        )
        val indexing = manifest(binding)
        val baseId = "index-generation-v2-" + "6".repeat(64)
        val graph = V2IndexGenerationGraphBinding(
            relativePath = V2GraphGenerationFile.GRAPH_FILE,
            byteLength = 64L,
            sha256 = "7".repeat(64),
            nodeCount = binding.trackCount,
            neighborsPerNode = 1,
            orderedTrackSetSha256 = binding.orderedTrackSetSha256,
        )
        val serverMerge = indexing.copy(
            origin = V2IndexGenerationOrigin.SERVER_MERGE,
            jobId = V2IndexGenerationManifestPolicy.SERVER_MERGE_JOB_ID,
            jobSpecId = V2IndexGenerationIdentity.serverMergeSpecId(
                baseGenerationId = baseId,
                bundleDatabaseSha256 = "8".repeat(64),
                receiptEmbeddingSpec = indexing.receiptEmbeddingSpec,
                textRetrievalSpec = indexing.textRetrievalSpec,
            ),
            baseGenerationId = baseId,
            rebuildDerivedIndexes = false,
            graphPolicy = V2IndexGenerationGraphPolicy.BASE_BOUND_ADDITION_UPDATE,
            graph = graph,
            createdAtEpochMs = 0L,
        )
        V2IndexGenerationManifestPolicy.requireValidProvenance(serverMerge)
        V2IndexGenerationManifestPolicy.requireValidCoverage(serverMerge)

        assertThrows(IllegalArgumentException::class.java) {
            V2IndexGenerationManifestPolicy.requireValidProvenance(
                serverMerge.copy(
                    graphPolicy = V2IndexGenerationGraphPolicy.EXPLICIT_REBUILD,
                    rebuildDerivedIndexes = true,
                ),
            )
        }
        assertThrows(IllegalArgumentException::class.java) {
            V2IndexGenerationManifestPolicy.requireValidProvenance(serverMerge.copy(graph = null))
        }
    }

    @Test
    fun `reader rejects same-count PEMB replacement with different vectors`() {
        val root = temporaryFolder.newFolder("generation-root")
        val binding = V2EmbeddingGenerationFile.write(
            source(10L to unitVector(0), 20L to unitVector(1)),
            File(root, "clamp3.emb"),
        )
        val database = File(root, "library.db").apply {
            writeBytes(ByteArray(64) { index -> index.toByte() })
        }
        val provisional = manifest(binding, database)
        val complete = provisional.copy(
            generationId = V2IndexGenerationIdentity.generationId(provisional),
        )
        val directory = File(root.parentFile, complete.generationId)
        assertTrue(root.renameTo(directory))
        File(directory, "manifest.json").writeText(Gson().toJson(complete))
        V2IndexGenerationReader.requireGenerationDirectory(directory)

        V2EmbeddingGenerationFile.write(
            source(10L to unitVector(0), 20L to unitVector(2)),
            File(directory, "clamp3.emb"),
        )
        assertThrows(IllegalArgumentException::class.java) {
            V2IndexGenerationReader.requireGenerationDirectory(
                directory,
                validation = V2GenerationArtifactValidation.PUBLISHED_BYTE_EXACT,
            )
        }
    }

    @Test
    fun `reader rejects same-length database replacement by exact byte hash`() {
        val root = temporaryFolder.newFolder("database-byte-binding-root")
        val binding = V2EmbeddingGenerationFile.write(
            source(10L to unitVector(0), 20L to unitVector(1)),
            File(root, "clamp3.emb"),
        )
        val database = File(root, "library.db").apply {
            writeBytes(ByteArray(64) { index -> index.toByte() })
        }
        val provisional = manifest(binding, database)
        val complete = provisional.copy(
            generationId = V2IndexGenerationIdentity.generationId(provisional),
        )
        val directory = File(root.parentFile, complete.generationId)
        assertTrue(root.renameTo(directory))
        File(directory, "manifest.json").writeText(Gson().toJson(complete))
        V2IndexGenerationReader.requireGenerationDirectory(directory)

        File(directory, "library.db").writeBytes(
            ByteArray(64) { index -> (index xor 0x55).toByte() },
        )
        assertThrows(IllegalArgumentException::class.java) {
            V2IndexGenerationReader.requireGenerationDirectory(
                directory,
                validation = V2GenerationArtifactValidation.PUBLISHED_BYTE_EXACT,
            )
        }
    }

    @Test
    fun `fresh installation resolves stat binding without exact revalidation`() {
        val fixture = installedGenerationFixture(
            stagingName = "fresh-install-staging",
            vectorIndex = 0,
            databaseSalt = 0x11,
        )
        var freshResolutionCount = 0
        var exactValidationCount = 0

        val resolved = V2InstalledGenerationResolutionPolicy.resolve(
            installedByThisCall = true,
            freshlyInstalled = {
                freshResolutionCount++
                V2FreshlyInstalledGenerationBindingResolver.requireResolved(
                    directory = fixture.directory,
                    manifest = fixture.manifest,
                    manifestSha256 = fixture.manifestSha256,
                    manifestByteLength = fixture.manifestByteLength,
                )
            },
            preexisting = {
                exactValidationCount++
                error("fresh publication must not rescan published artifacts")
            },
        )

        assertEquals(1, freshResolutionCount)
        assertEquals(0, exactValidationCount)
        assertEquals(fixture.directory.canonicalFile, resolved.directory)
        assertEquals(File(fixture.directory, "library.db").canonicalFile, resolved.databaseFile)
        assertEquals(File(fixture.directory, "clamp3.emb").canonicalFile, resolved.embeddingFile)
        assertNull(resolved.graphFile)
    }

    @Test
    fun `preexisting installation invokes exact validator`() {
        val fixture = installedGenerationFixture(
            stagingName = "preexisting-install-staging",
            vectorIndex = 1,
            databaseSalt = 0x22,
        )
        val exactResult = V2FreshlyInstalledGenerationBindingResolver.requireResolved(
            directory = fixture.directory,
            manifest = fixture.manifest,
            manifestSha256 = fixture.manifestSha256,
            manifestByteLength = fixture.manifestByteLength,
        )
        var freshResolutionCount = 0
        var exactValidationCount = 0

        val resolved = V2InstalledGenerationResolutionPolicy.resolve(
            installedByThisCall = false,
            freshlyInstalled = {
                freshResolutionCount++
                error("pre-existing publication must not trust a prior install")
            },
            preexisting = {
                exactValidationCount++
                exactResult
            },
        )

        assertEquals(0, freshResolutionCount)
        assertEquals(1, exactValidationCount)
        assertEquals(exactResult, resolved)
    }

    @Test
    fun `fresh installation rejects changed artifact stat binding`() {
        val fixture = installedGenerationFixture(
            stagingName = "changed-stat-staging",
            vectorIndex = 2,
            databaseSalt = 0x33,
        )
        File(fixture.directory, "library.db").appendBytes(byteArrayOf(0x44))

        assertThrows(IllegalArgumentException::class.java) {
            V2FreshlyInstalledGenerationBindingResolver.requireResolved(
                directory = fixture.directory,
                manifest = fixture.manifest,
                manifestSha256 = fixture.manifestSha256,
                manifestByteLength = fixture.manifestByteLength,
            )
        }
    }

    @Test
    fun `publication CAS reads only the durable pointer`() {
        val pointer = pointer('4')
        var pointerReadCount = 0
        var exactValidationCount = 0

        val resolved = V2GenerationPointerReadPolicy.resolve(
            purpose = V2GenerationPointerReadPurpose.PUBLICATION_CAS,
            pointerOnly = {
                pointerReadCount++
                pointer
            },
            exactGeneration = {
                exactValidationCount++
                error("CAS must not rescan the active generation")
            },
        )

        assertEquals(pointer, resolved)
        assertEquals(1, pointerReadCount)
        assertEquals(0, exactValidationCount)
    }

    @Test
    fun `crash recovery validates the exact active generation`() {
        val pointer = pointer('5')
        var pointerReadCount = 0
        var exactValidationCount = 0

        val resolved = V2GenerationPointerReadPolicy.resolve(
            purpose = V2GenerationPointerReadPurpose.CRASH_RECOVERY,
            pointerOnly = {
                pointerReadCount++
                error("recovery must validate the active generation")
            },
            exactGeneration = {
                exactValidationCount++
                pointer
            },
        )

        assertEquals(pointer, resolved)
        assertEquals(0, pointerReadCount)
        assertEquals(1, exactValidationCount)
    }

    @Test
    fun `publication policy rejects stale pointer and keeps only the new active generation`() {
        val first = pointer('1')
        val second = pointer('2')

        assertThrows(V2GenerationPublicationConflictException::class.java) {
            V2GenerationPublicationPolicy.pointerForCommit(
                expected = first,
                current = second,
                generationId = generationId('3'),
                manifestSha256 = "3".repeat(64),
            )
        }
        assertEquals(
            V2ActiveGenerationPointer(
                schemaVersion = 2,
                generationId = generationId('3'),
                manifestSha256 = "3".repeat(64),
            ),
            V2GenerationPublicationPolicy.pointerForCommit(
                expected = second,
                current = second,
                generationId = generationId('3'),
                manifestSha256 = "3".repeat(64),
            ),
        )
    }

    @Test
    fun `publication policy replay keeps the one active pointer unchanged`() {
        val current = pointer('2')
        assertEquals(
            current,
            V2GenerationPublicationPolicy.pointerForCommit(
                expected = current,
                current = current,
                generationId = current.generationId,
                manifestSha256 = current.manifestSha256,
            ),
        )
    }

    @Test
    fun `orphan retention fails closed and protects only owned indexing generations`() {
        val binding = V2EmbeddingGenerationFile.write(
            source(10L to unitVector(0)),
            temporaryFolder.newFile("orphan-retention.emb"),
        )
        val indexing = manifest(binding)

        assertTrue(
            V2GenerationOrphanRetentionPolicy.retainUnreferenced(
                manifest = null,
                protectedNonterminalJobIds = emptySet(),
            ),
        )
        assertTrue(
            V2GenerationOrphanRetentionPolicy.retainUnreferenced(
                manifest = indexing,
                protectedNonterminalJobIds = setOf(indexing.jobId),
            ),
        )
        assertFalse(
            V2GenerationOrphanRetentionPolicy.retainUnreferenced(
                manifest = indexing,
                protectedNonterminalJobIds = emptySet(),
            ),
        )
        assertFalse(
            V2GenerationOrphanRetentionPolicy.retainUnreferenced(
                manifest = indexing.copy(origin = V2IndexGenerationOrigin.BOOTSTRAP_COMPATIBILITY),
                protectedNonterminalJobIds = setOf(indexing.jobId),
            ),
        )
    }

    private data class InstalledGenerationFixture(
        val directory: File,
        val manifest: V2IndexGenerationManifest,
        val manifestSha256: String,
        val manifestByteLength: Long,
    )

    private fun installedGenerationFixture(
        stagingName: String,
        vectorIndex: Int,
        databaseSalt: Int,
    ): InstalledGenerationFixture {
        val staging = temporaryFolder.newFolder(stagingName)
        val binding = V2EmbeddingGenerationFile.write(
            source(10L to unitVector(vectorIndex)),
            File(staging, "clamp3.emb"),
        )
        val database = File(staging, "library.db").apply {
            writeBytes(ByteArray(64) { index -> (index xor databaseSalt).toByte() })
        }
        val provisional = manifest(binding, database)
        val complete = provisional.copy(
            generationId = V2IndexGenerationIdentity.generationId(provisional),
        )
        val directory = File(staging.parentFile, complete.generationId)
        assertTrue(staging.renameTo(directory))
        val manifestFile = File(directory, "manifest.json").apply {
            writeText(Gson().toJson(complete))
        }
        return InstalledGenerationFixture(
            directory = directory,
            manifest = complete,
            manifestSha256 = V2FileSha256.digest(manifestFile),
            manifestByteLength = manifestFile.length(),
        )
    }

    private fun manifest(
        binding: V2OrderedEmbeddingBinding,
        database: File? = null,
    ): V2IndexGenerationManifest {
        val audioSpec = embeddingSpec()
        val provisional = V2IndexGenerationManifest(
            schemaVersion = 3,
            origin = V2IndexGenerationOrigin.INDEXING_JOB,
            generationId = "",
            activationBindingId = "",
            jobId = "job-1",
            jobSpecId = "job-spec-v5-" + "b".repeat(64),
            receiptEmbeddingSpec = audioSpec,
            textRetrievalSpec = textRetrievalSpec(audioSpec),
            baseGenerationId = null,
            rebuildDerivedIndexes = false,
            graphPolicy = V2IndexGenerationGraphPolicy.ABSENT,
            createdAtEpochMs = 1L,
            databaseRelativePath = "library.db",
            databaseByteLength = database?.length() ?: 64L,
            databaseSha256 = database?.let(V2FileSha256::digest) ?: "c".repeat(64),
            databaseContentSha256 = binding.databaseContentSha256,
            orderedTrackSetSha256 = binding.orderedTrackSetSha256,
            stableTrackUidCoverage = V2StableTrackUidCoverageBinding(
                coveredTrackCount = binding.trackCount,
                uncoveredTrackCount = 0,
                uniqueStableTrackSpanCount = binding.trackCount,
                fullContentIdentityCount = 0,
                sampledContentIdentityCount = binding.trackCount,
                mappingSha256 = "d".repeat(64),
            ),
            embeddingCoverage = V2EmbeddingSpecCoverageBinding(
                totalTrackCount = binding.trackCount,
                receiptBoundTrackCount = binding.trackCount,
                receiptSpecTrackCounts = mapOf(audioSpec.specId to binding.trackCount),
                compatibilityBase = null,
                mappingSha256 = "a".repeat(64),
            ),
            trackCount = binding.trackCount,
            embeddingDimension = binding.dimension,
            embeddingRelativePath = "clamp3.emb",
            embeddingByteLength = binding.byteLength,
            embeddingSha256 = binding.fileSha256,
            graph = null,
        )
        return provisional.copy(
            activationBindingId = V2IndexGenerationIdentity.activationBindingId(provisional),
        )
    }

    private fun embeddingSpec() = V2IndexingLedgerPlanner.createEmbeddingSpec(
        EmbeddingSpecInput(
            preprocessingSpecId = V2IndexingWorkPolicy.PREPROCESSING_SPEC_ID,
            decoderPolicyId = V2IndexingWorkPolicy.DECODER_POLICY_ID,
            inferenceBackendPolicyId = V2IndexingWorkPolicy.INFERENCE_BACKEND_POLICY_ID,
            outputDimension = V2_CLAMP3_DIMENSION,
            modelArtifactSha256 = mapOf(
                "mert" to "e".repeat(64),
                "clamp3_audio" to "f".repeat(64),
            ),
        ),
    )

    private fun textRetrievalSpec(
        audioSpec: EmbeddingSpecFingerprint,
    ) = V2IndexingLedgerPlanner.createTextRetrievalSpec(
        TextRetrievalSpecInput(
            compatibleAudioEmbeddingSpecId = audioSpec.specId,
            textModelSha256 = "1".repeat(64),
            tokenizerModelSha256 = V2IndexingWorkPolicy.TEXT_TOKENIZER_MODEL_SHA256,
            tokenizerPolicyId = V2IndexingWorkPolicy.TEXT_TOKENIZER_POLICY_ID,
            tokenizerRuntimeContractSha256 =
                V2IndexingWorkPolicy.TEXT_TOKENIZER_RUNTIME_CONTRACT_SHA256,
            outputSpaceId = V2IndexingWorkPolicy.TEXT_OUTPUT_SPACE_ID,
            outputDimension = audioSpec.outputDimension,
            inferenceBackendPolicyId = V2IndexingWorkPolicy.TEXT_INFERENCE_BACKEND_POLICY_ID,
        ),
    )

    private fun source(
        vararg rows: Pair<Long, FloatArray>,
    ): V2OrderedEmbeddingSource = object : V2OrderedEmbeddingSource {
        override val trackCount: Int = rows.size
        override val dimension: Int = V2_CLAMP3_DIMENSION
        override fun forEachOrdered(consumer: V2OrderedEmbeddingConsumer) {
            rows.forEach { (id, vector) -> consumer.accept(id, V2Clamp3VectorCodec.encode(vector)) }
        }
    }

    private fun unitVector(index: Int): FloatArray = FloatArray(V2_CLAMP3_DIMENSION).also {
        it[index] = 1f
    }

    private fun pointer(value: Char) = V2ActiveGenerationPointer(
        schemaVersion = 2,
        generationId = generationId(value),
        manifestSha256 = value.toString().repeat(64),
    )

    private fun generationId(value: Char): String =
        "index-generation-v2-${value.toString().repeat(64)}"

    private fun writeGraph(
        file: File,
        ids: LongArray,
        neighbors: Int,
        neighborIndex: Int = 0,
    ) {
        val bytes = ByteBuffer.allocate(8 + ids.size * 8 + ids.size * neighbors * 8)
            .order(ByteOrder.LITTLE_ENDIAN)
            .putInt(ids.size)
            .putInt(neighbors)
        ids.forEach(bytes::putLong)
        repeat(ids.size * neighbors) {
            bytes.putInt(neighborIndex)
            bytes.putFloat(1f)
        }
        file.writeBytes(bytes.array())
    }
}
