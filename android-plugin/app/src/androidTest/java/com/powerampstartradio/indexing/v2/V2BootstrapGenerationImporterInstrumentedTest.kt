package com.powerampstartradio.indexing.v2

import android.database.sqlite.SQLiteDatabase
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.UUID
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class V2BootstrapGenerationImporterInstrumentedTest {
    @Test
    fun completePrePointerCrashOrphanIsReconciledWithoutChangingTheActiveGeneration() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val root = File(context.cacheDir, "generation-orphan-reconcile-${UUID.randomUUID()}")
        assertTrue(root.mkdirs())
        val policy = futurePolicy()
        try {
            val activeSource = File(root, "active.db")
            createDatabase(activeSource, listOf(1L to unitVectorBlob(0)))
            val active = V2BootstrapGenerationImporter(
                context = context,
                filesDir = root,
                publisher = V2IndexGenerationPublisher(root),
                modelPolicyResolver = { policy },
            ).importFileBlocking(activeSource, policy).generation

            val orphanId = AtomicReference<String?>()
            val orphanSource = File(root, "orphan.db")
            createDatabase(orphanSource, listOf(2L to unitVectorBlob(1)))
            val failingImporter = V2BootstrapGenerationImporter(
                context = context,
                filesDir = root,
                publisher = V2IndexGenerationPublisher(
                    filesDir = root,
                    afterInstallBeforePointerPublication = { manifest ->
                        orphanId.set(manifest.generationId)
                        throw SimulatedCrash()
                    },
                ),
                modelPolicyResolver = { policy },
            )
            assertThrows(SimulatedCrash::class.java) {
                failingImporter.importFileBlocking(orphanSource, policy)
            }
            val installedOrphanId = requireNotNull(orphanId.get())
            val orphanDirectory = File(root, "indexing_v2/generations/$installedOrphanId")
            assertTrue(orphanDirectory.isDirectory)
            assertEquals(
                active.manifest.generationId,
                V2IndexGenerationReader.requireActive(root).manifest.generationId,
            )

            val result = V2GenerationPublicationCoordinator.reconcileCrashOrphans(
                filesDir = root,
                protectedNonterminalJobIds = emptySet(),
            )
            assertEquals(listOf(installedOrphanId), result.deletedGenerationIds)
            assertFalse(orphanDirectory.exists())
            assertEquals(
                active.manifest.generationId,
                V2IndexGenerationReader.requireActive(root).manifest.generationId,
            )
        } finally {
            root.deleteRecursively()
        }
    }

    @Test
    fun orphanReconciliationDeletesNothingWhenTheActivePointerIsUnreadable() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val root = File(context.cacheDir, "generation-orphan-fail-closed-${UUID.randomUUID()}")
        assertTrue(root.mkdirs())
        val policy = futurePolicy()
        try {
            val activeSource = File(root, "active.db")
            createDatabase(activeSource, listOf(1L to unitVectorBlob(0)))
            V2BootstrapGenerationImporter(
                context = context,
                filesDir = root,
                publisher = V2IndexGenerationPublisher(root),
                modelPolicyResolver = { policy },
            ).importFileBlocking(activeSource, policy)

            val orphanId = AtomicReference<String?>()
            val orphanSource = File(root, "orphan.db")
            createDatabase(orphanSource, listOf(2L to unitVectorBlob(1)))
            val failingImporter = V2BootstrapGenerationImporter(
                context = context,
                filesDir = root,
                publisher = V2IndexGenerationPublisher(
                    filesDir = root,
                    afterInstallBeforePointerPublication = { manifest ->
                        orphanId.set(manifest.generationId)
                        throw SimulatedCrash()
                    },
                ),
                modelPolicyResolver = { policy },
            )
            assertThrows(SimulatedCrash::class.java) {
                failingImporter.importFileBlocking(orphanSource, policy)
            }
            val orphanDirectory = File(
                root,
                "indexing_v2/generations/${requireNotNull(orphanId.get())}",
            )
            assertTrue(orphanDirectory.isDirectory)
            File(root, "indexing_v2/generations/active-generation.json").writeText("{")

            val result = V2GenerationPublicationCoordinator.reconcileCrashOrphans(
                filesDir = root,
                protectedNonterminalJobIds = emptySet(),
            )
            assertEquals(
                V2GenerationOrphanReconciliationSkipReason.ACTIVE_GENERATION_UNREADABLE,
                result.skipReason,
            )
            assertTrue(orphanDirectory.isDirectory)
            assertTrue(result.deletedGenerationIds.isEmpty())
        } finally {
            root.deleteRecursively()
        }
    }

    @Test
    fun concurrentPublishersUseExactPointerCompareAndSwapAndBoundRetention() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val root = File(context.cacheDir, "generation-cas-${UUID.randomUUID()}")
        assertTrue(root.mkdirs())
        val policy = futurePolicy()
        val firstReady = CountDownLatch(1)
        val releaseFirst = CountDownLatch(1)
        val firstFailure = AtomicReference<Throwable?>()
        val executor = Executors.newSingleThreadExecutor()
        try {
            val firstSource = File(root, "cas-first.db")
            createDatabase(firstSource, listOf(1L to unitVectorBlob(0)))
            val blockedImporter = V2BootstrapGenerationImporter(
                context = context,
                filesDir = root,
                publisher = V2IndexGenerationPublisher(
                    filesDir = root,
                    beforePointerPublication = {
                        firstReady.countDown()
                        assertTrue(releaseFirst.await(20, TimeUnit.SECONDS))
                    },
                ),
                modelPolicyResolver = { policy },
            )
            val future = executor.submit {
                try {
                    blockedImporter.importFileBlocking(firstSource, policy)
                } catch (error: Throwable) {
                    firstFailure.set(error)
                }
            }
            assertTrue(firstReady.await(20, TimeUnit.SECONDS))

            val winningSource = File(root, "cas-winner.db")
            createDatabase(winningSource, listOf(2L to unitVectorBlob(1)))
            val importer = V2BootstrapGenerationImporter(
                context = context,
                filesDir = root,
                publisher = V2IndexGenerationPublisher(root),
                modelPolicyResolver = { policy },
            )
            val winner = importer.importFileBlocking(winningSource, policy).generation
            releaseFirst.countDown()
            future.get(20, TimeUnit.SECONDS)
            assertTrue(firstFailure.get() is V2GenerationPublicationConflictException)
            assertEquals(
                winner.manifest.generationId,
                V2IndexGenerationReader.requireActive(root).manifest.generationId,
            )

            assertEquals(1, generationDirectories(root).size)
        } finally {
            releaseFirst.countDown()
            executor.shutdownNow()
            root.deleteRecursively()
        }
    }

    @Test
    fun malformedWrongDimensionNonFiniteAndTruncatedSourcesAreRejected() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val root = File(context.cacheDir, "bootstrap-invalid-${UUID.randomUUID()}")
        assertTrue(root.mkdirs())
        try {
            val malformed = File(root, "malformed.db").apply { writeText("not sqlite") }
            assertThrows(Exception::class.java) {
                V2BootstrapDatabaseValidator.validate(malformed)
            }

            val wrongDimension = File(root, "wrong-dimension.db")
            createDatabase(wrongDimension, listOf(1L to floatBlob(FloatArray(767).apply {
                this[0] = 1f
            })))
            assertThrows(IllegalArgumentException::class.java) {
                V2BootstrapDatabaseValidator.validate(wrongDimension)
            }

            val nonFinite = File(root, "non-finite.db")
            createDatabase(nonFinite, listOf(1L to floatBlob(FloatArray(768).apply {
                this[0] = Float.NaN
            })))
            assertThrows(IllegalArgumentException::class.java) {
                V2BootstrapDatabaseValidator.validate(nonFinite)
            }

            val valid = File(root, "valid.db")
            createDatabase(valid, listOf(1L to unitVectorBlob(0)))
            val truncated = File(root, "truncated.db").apply {
                writeBytes(valid.readBytes().copyOf(valid.length().toInt() / 2))
            }
            assertThrows(Exception::class.java) {
                V2BootstrapDatabaseValidator.validate(truncated)
            }
        } finally {
            root.deleteRecursively()
        }
    }

    @Test
    fun initialImportPublishesOnceAndRefusesReplacement() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val root = File(context.cacheDir, "bootstrap-lifecycle-${UUID.randomUUID()}")
        assertTrue(root.mkdirs())
        val policy = futurePolicy()
        try {
            val firstSource = File(root, "first-source.db")
            createDatabase(firstSource, listOf(1L to unitVectorBlob(0), 2L to unitVectorBlob(1)))
            val importer = V2BootstrapGenerationImporter(
                context = context,
                filesDir = root,
                publisher = V2IndexGenerationPublisher(root),
                modelPolicyResolver = { policy },
            )
            val first = importer.importFileBlocking(firstSource, policy)
            assertPureCompatibility(first.generation, expectedTracks = 2)

            assertThrows(IllegalArgumentException::class.java) {
                importer.importFileBlocking(firstSource, policy)
            }

            val secondSource = File(root, "second-source.db")
            createDatabase(secondSource, listOf(1L to unitVectorBlob(2), 2L to unitVectorBlob(3)))
            assertThrows(IllegalArgumentException::class.java) {
                importer.importFileBlocking(secondSource, policy)
            }
            val stillActive = V2IndexGenerationReader.requireActive(root)
            assertEquals(first.generation.manifest.generationId, stillActive.manifest.generationId)
            assertTrue(first.generation.directory.isDirectory)
            assertEquals(1, generationDirectories(root).size)
        } finally {
            root.deleteRecursively()
        }
    }

    @Test
    fun maintenancePublishesExactCompatibilitySubsetSupportsNoOpAndRollsBack() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val root = File(context.cacheDir, "maintenance-lifecycle-${UUID.randomUUID()}")
        assertTrue(root.mkdirs())
        val policy = futurePolicy()
        try {
            val source = File(root, "source.db")
            createDatabase(
                source,
                listOf(
                    1L to unitVectorBlob(0),
                    2L to unitVectorBlob(1),
                    3L to unitVectorBlob(2),
                ),
            )
            val base = V2BootstrapGenerationImporter(
                context = context,
                filesDir = root,
                publisher = V2IndexGenerationPublisher(root),
                modelPolicyResolver = { policy },
            ).importFileBlocking(source, policy).generation

            val maintenance = V2LibraryMaintenancePublisher(
                context = context,
                filesDir = root,
                publisher = V2IndexGenerationPublisher(root),
            )
            val cleaned = maintenance.removeTracksBlocking(setOf(2L))
            assertEquals(1, cleaned.removedTrackCount)
            assertEquals(false, cleaned.noOp)
            assertEquals(V2IndexGenerationOrigin.LIBRARY_MAINTENANCE,
                cleaned.generation.manifest.origin)
            assertEquals(base.manifest.generationId,
                cleaned.generation.manifest.baseGenerationId)
            assertEquals(2, cleaned.generation.manifest.trackCount)
            assertEquals(0, cleaned.generation.manifest.embeddingCoverage.receiptBoundTrackCount)
            assertEquals(2,
                cleaned.generation.manifest.embeddingCoverage.compatibilityBase?.trackCount)
            assertEquals(
                V2IndexGenerationGraphPolicy.ABSENT,
                cleaned.generation.manifest.graphPolicy,
            )
            assertNull(cleaned.generation.graphFile)
            assertFalse(base.directory.exists())
            assertEquals(
                cleaned.generation.manifest.generationId,
                V2IndexGenerationReader.requireActive(root).manifest.generationId,
            )

            val noOp = maintenance.removeTracksBlocking(setOf(2L))
            assertTrue(noOp.noOp)
            assertEquals(cleaned.generation.manifest.generationId,
                noOp.generation.manifest.generationId)

            val failing = V2LibraryMaintenancePublisher(
                context = context,
                filesDir = root,
                publisher = V2IndexGenerationPublisher(
                    filesDir = root,
                    beforePointerPublication = { throw SimulatedCrash() },
                ),
            )
            assertThrows(SimulatedCrash::class.java) {
                failing.removeTracksBlocking(setOf(3L))
            }
            assertEquals(
                cleaned.generation.manifest.generationId,
                V2IndexGenerationReader.requireActive(root).manifest.generationId,
            )
            assertFalse(base.directory.exists())
            assertTrue(cleaned.generation.directory.isDirectory)
        } finally {
            root.deleteRecursively()
        }
    }

    @Test
    fun maintenancePublishesBaseBoundDeletionRepairWhenTheActiveGraphIsUsable() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val root = File(context.cacheDir, "maintenance-graph-${UUID.randomUUID()}")
        assertTrue(root.mkdirs())
        val policy = futurePolicy()
        try {
            val source = File(root, "source-with-graph.db")
            createDatabase(
                source,
                listOf(
                    1L to positiveVectorBlob(1.00f, 0.10f),
                    2L to positiveVectorBlob(0.98f, 0.20f),
                    3L to positiveVectorBlob(0.90f, 0.40f),
                    4L to positiveVectorBlob(0.80f, 0.60f),
                ),
                graphIds = longArrayOf(1L, 2L, 3L, 4L),
            )
            val base = V2BootstrapGenerationImporter(
                context = context,
                filesDir = root,
                publisher = V2IndexGenerationPublisher(root),
                modelPolicyResolver = { policy },
            ).importFileBlocking(source, policy).generation
            assertEquals(
                V2IndexGenerationGraphPolicy.VALIDATED_COMPATIBILITY_IMPORT,
                base.manifest.graphPolicy,
            )

            val cleaned = V2LibraryMaintenancePublisher(
                context = context,
                filesDir = root,
                publisher = V2IndexGenerationPublisher(root),
            ).removeTracksBlocking(setOf(2L))
            assertEquals(
                V2IndexGenerationGraphPolicy.BASE_BOUND_DELETION_REPAIR,
                cleaned.generation.manifest.graphPolicy,
            )
            val binding = V2GraphGenerationFile.inspect(
                requireNotNull(cleaned.generation.graphFile),
            )
            assertEquals(3, binding.nodeCount)
            assertEquals(
                cleaned.generation.manifest.orderedTrackSetSha256,
                binding.orderedTrackSetSha256,
            )
            assertTrue(base.directory.isDirectory)
        } finally {
            root.deleteRecursively()
        }
    }

    @Test
    fun bootstrapPreservesOnlyAFormatValidGraphWithExactOrderedTrackIds() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val root = File(context.cacheDir, "bootstrap-graph-${UUID.randomUUID()}")
        assertTrue(root.mkdirs())
        val policy = futurePolicy()
        try {
            val exactSource = File(root, "exact.db")
            createDatabase(
                exactSource,
                listOf(1L to unitVectorBlob(0), 2L to unitVectorBlob(1)),
                graphIds = longArrayOf(1L, 2L),
            )
            val exact = V2BootstrapGenerationImporter(
                context = context,
                filesDir = root,
                publisher = V2IndexGenerationPublisher(root),
                modelPolicyResolver = { policy },
            ).importFileBlocking(exactSource, policy).generation
            assertEquals(
                V2IndexGenerationGraphPolicy.VALIDATED_COMPATIBILITY_IMPORT,
                exact.manifest.graphPolicy,
            )
            assertTrue(exact.graphFile?.isFile == true)

            val staleSource = File(root, "stale.db")
            createDatabase(
                staleSource,
                listOf(10L to unitVectorBlob(2), 20L to unitVectorBlob(3)),
                graphIds = longArrayOf(10L, 30L),
            )
            val stale = V2BootstrapGenerationImporter(
                context = context,
                filesDir = root,
                publisher = V2IndexGenerationPublisher(root),
                modelPolicyResolver = { policy },
            ).importFileBlocking(staleSource, policy).generation
            assertEquals(V2IndexGenerationGraphPolicy.ABSENT, stale.manifest.graphPolicy)
            assertNull(stale.graphFile)
            assertTrue(exact.directory.isDirectory)
        } finally {
            root.deleteRecursively()
        }
    }

    private fun assertPureCompatibility(
        generation: V2ResolvedActiveIndexGeneration,
        expectedTracks: Int,
    ) {
        val manifest = generation.manifest
        assertEquals(V2IndexGenerationOrigin.BOOTSTRAP_COMPATIBILITY, manifest.origin)
        assertEquals(expectedTracks, manifest.trackCount)
        assertEquals(0, manifest.embeddingCoverage.receiptBoundTrackCount)
        assertTrue(manifest.embeddingCoverage.receiptSpecTrackCounts.isEmpty())
        assertEquals(expectedTracks, manifest.embeddingCoverage.compatibilityBase?.trackCount)
        assertEquals(0, manifest.stableTrackUidCoverage.coveredTrackCount)
        assertEquals(expectedTracks, manifest.stableTrackUidCoverage.uncoveredTrackCount)
        assertEquals(V2IndexGenerationGraphPolicy.ABSENT, manifest.graphPolicy)
        assertEquals(null, manifest.graph)
    }

    private fun generationDirectories(root: File): List<File> =
        File(root, "indexing_v2/generations").listFiles().orEmpty()
            .filter { it.isDirectory && it.name.startsWith("index-generation-v2-") }

    private fun createDatabase(
        file: File,
        rows: List<Pair<Long, ByteArray>>,
        graphIds: LongArray? = null,
    ) {
        SQLiteDatabase.openOrCreateDatabase(file, null).use { database ->
            database.execSQL(
                """
                CREATE TABLE tracks (
                    id INTEGER PRIMARY KEY,
                    metadata_key TEXT NOT NULL,
                    filename_key TEXT NOT NULL,
                    artist TEXT,
                    album TEXT,
                    title TEXT,
                    duration_ms INTEGER,
                    file_path TEXT NOT NULL,
                    source TEXT DEFAULT 'desktop'
                )
                """.trimIndent(),
            )
            database.execSQL(
                """
                CREATE TABLE embeddings_clamp3 (
                    track_id INTEGER PRIMARY KEY,
                    embedding BLOB NOT NULL,
                    FOREIGN KEY(track_id) REFERENCES tracks(id)
                )
                """.trimIndent(),
            )
            rows.forEach { (id, blob) ->
                database.execSQL(
                    """
                    INSERT INTO tracks(id, metadata_key, filename_key, artist, album, title,
                                       duration_ms, file_path, source)
                    VALUES (?, ?, ?, 'Artist', 'Album', ?, 1000, ?, 'desktop')
                    """.trimIndent(),
                    arrayOf<Any>(
                        id,
                        "artist|album|title$id|1000",
                        "track$id.flac",
                        "Title $id",
                        "/track$id.flac",
                    ),
                )
                database.execSQL(
                    "INSERT INTO embeddings_clamp3(track_id, embedding) VALUES (?, ?)",
                    arrayOf(id, blob),
                )
            }
            graphIds?.let { ids ->
                database.execSQL(
                    "CREATE TABLE binary_data(key TEXT PRIMARY KEY, data BLOB NOT NULL)",
                )
                database.execSQL(
                    "INSERT INTO binary_data(key, data) VALUES ('knn_graph', ?)",
                    arrayOf(graphBlob(ids)),
                )
            }
        }
    }

    private fun graphBlob(ids: LongArray): ByteArray {
        val buffer = ByteBuffer.allocate(8 + ids.size * 8 + ids.size * 8)
            .order(ByteOrder.LITTLE_ENDIAN)
            .putInt(ids.size)
            .putInt(1)
        ids.forEach(buffer::putLong)
        ids.indices.forEach { index ->
            buffer.putInt((index + 1) % ids.size)
            buffer.putFloat(1f)
        }
        return buffer.array()
    }

    private fun futurePolicy(): V2FutureModelPolicy {
        val embedding = V2IndexingLedgerPlanner.createEmbeddingSpec(
            EmbeddingSpecInput(
                preprocessingSpecId = V2IndexingWorkPolicy.PREPROCESSING_SPEC_ID,
                decoderPolicyId = V2IndexingWorkPolicy.DECODER_POLICY_ID,
                inferenceBackendPolicyId = V2IndexingWorkPolicy.INFERENCE_BACKEND_POLICY_ID,
                outputDimension = V2_CLAMP3_DIMENSION,
                modelArtifactSha256 = mapOf(
                    "mert" to "a".repeat(64),
                    "clamp3_audio" to "b".repeat(64),
                ),
            ),
        )
        return V2FutureModelPolicy(
            receiptEmbeddingSpec = embedding,
            textRetrievalSpec = V2IndexingLedgerPlanner.createTextRetrievalSpec(
                TextRetrievalSpecInput(
                    compatibleAudioEmbeddingSpecId = embedding.specId,
                    textModelSha256 = "c".repeat(64),
                    tokenizerModelSha256 = V2IndexingWorkPolicy.TEXT_TOKENIZER_MODEL_SHA256,
                    tokenizerPolicyId = V2IndexingWorkPolicy.TEXT_TOKENIZER_POLICY_ID,
                    tokenizerRuntimeContractSha256 =
                        V2IndexingWorkPolicy.TEXT_TOKENIZER_RUNTIME_CONTRACT_SHA256,
                    outputSpaceId = V2IndexingWorkPolicy.TEXT_OUTPUT_SPACE_ID,
                    outputDimension = V2_CLAMP3_DIMENSION,
                    inferenceBackendPolicyId =
                        V2IndexingWorkPolicy.TEXT_INFERENCE_BACKEND_POLICY_ID,
                ),
            ),
        )
    }

    private fun unitVectorBlob(index: Int): ByteArray =
        V2Clamp3VectorCodec.encode(FloatArray(V2_CLAMP3_DIMENSION).apply { this[index] = 1f })

    private fun positiveVectorBlob(first: Float, second: Float): ByteArray {
        val norm = kotlin.math.sqrt(first * first + second * second)
        return V2Clamp3VectorCodec.encode(FloatArray(V2_CLAMP3_DIMENSION).apply {
            this[0] = first / norm
            this[1] = second / norm
        })
    }

    private fun floatBlob(values: FloatArray): ByteArray =
        ByteBuffer.allocate(values.size * Float.SIZE_BYTES).order(ByteOrder.LITTLE_ENDIAN).apply {
            values.forEach(::putFloat)
        }.array()

    private class SimulatedCrash : RuntimeException("simulated pre-pointer crash")
}
