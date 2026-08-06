package com.powerampstartradio.indexing.v2

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

/** Opt-in: root stages the frozen DB and exact models into the V2 app's private files directory. */
@RunWith(AndroidJUnit4::class)
class V2FrozenDatabaseImportAcceptanceTest {
    @Test
    fun importFrozenDatabaseIsExactGraphBoundIdempotentAndRollbackSafe() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val filesDir = context.filesDir
        val source = File(filesDir, "device_acceptance/embeddings.db")
        assumeTrue("Opt-in frozen database is not staged", source.isFile)
        REQUIRED_MODELS.forEach { name ->
            assumeTrue("Opt-in exact model is not staged: $name", File(filesDir, name).isFile)
        }
        assertEquals(EXPECTED_SOURCE_SHA256, V2FileSha256.digest(source))

        val previous = runCatching { V2IndexGenerationReader.requireActive(filesDir) }.getOrNull()
        val importer = V2BootstrapGenerationImporter(context)
        val first = importer.importFileBlocking(source)
        val manifest = first.generation.manifest
        assertEquals(EXPECTED_TRACK_COUNT, first.sourceValidation.trackCount)
        assertEquals(EXPECTED_SOURCE_SHA256, first.sourceValidation.sourceSha256)
        assertEquals(V2IndexGenerationOrigin.BOOTSTRAP_COMPATIBILITY, manifest.origin)
        assertEquals(EXPECTED_TRACK_COUNT, manifest.trackCount)
        assertEquals(V2_CLAMP3_DIMENSION, manifest.embeddingDimension)
        assertEquals(0, manifest.embeddingCoverage.receiptBoundTrackCount)
        assertTrue(manifest.embeddingCoverage.receiptSpecTrackCounts.isEmpty())
        assertEquals(EXPECTED_TRACK_COUNT, manifest.embeddingCoverage.compatibilityBase?.trackCount)
        assertEquals(0, manifest.stableTrackUidCoverage.coveredTrackCount)
        assertEquals(EXPECTED_TRACK_COUNT, manifest.stableTrackUidCoverage.uncoveredTrackCount)
        assertEquals(
            V2IndexGenerationGraphPolicy.VALIDATED_COMPATIBILITY_IMPORT,
            manifest.graphPolicy,
        )
        assertEquals(EXPECTED_GRAPH_SHA256, manifest.graph?.sha256)
        assertNotNull(first.generation.graphFile)
        assertEquals(
            manifest.orderedTrackSetSha256,
            requireNotNull(manifest.graph).orderedTrackSetSha256,
        )
        assertEquals(
            V2OrderedEmbeddingBinding(
                trackCount = manifest.trackCount,
                dimension = manifest.embeddingDimension,
                byteLength = manifest.embeddingByteLength,
                fileSha256 = manifest.embeddingSha256,
                orderedTrackSetSha256 = manifest.orderedTrackSetSha256,
                databaseContentSha256 = manifest.databaseContentSha256,
            ),
            V2EmbeddingGenerationFile.inspect(first.generation.embeddingFile),
        )
        V2IndexGenerationReader.requireActive(filesDir)
        previous?.let { assertTrue("Previous generation was deleted", it.directory.isDirectory) }

        val replay = importer.importFileBlocking(source)
        assertEquals(manifest.generationId, replay.generation.manifest.generationId)
        assertEquals(manifest.manifestIdentity(), replay.generation.manifest.manifestIdentity())

        val policy = V2CurrentModelPolicyResolver.resolve(filesDir)
        val failingPublisher = V2IndexGenerationPublisher(
            filesDir = filesDir,
            beforePointerPublication = { throw SimulatedCrash() },
        )
        try {
            failingPublisher.publishBootstrapCompatibility(
                privateStagingDatabase = source,
                futureReceiptEmbeddingSpec = policy.receiptEmbeddingSpec,
                textRetrievalSpec = policy.textRetrievalSpec,
            )
            throw AssertionError("Simulated pre-pointer failure did not abort publication")
        } catch (_: SimulatedCrash) {
            // Expected. The assertion below is the rollback proof.
        }
        assertEquals(
            manifest.generationId,
            V2IndexGenerationReader.requireActive(filesDir).manifest.generationId,
        )
    }

    private fun V2IndexGenerationManifest.manifestIdentity(): List<String> = listOf(
        generationId,
        activationBindingId,
        databaseSha256,
        databaseContentSha256,
        orderedTrackSetSha256,
        embeddingSha256,
        graph?.sha256.orEmpty(),
    )

    private class SimulatedCrash : RuntimeException("simulated pre-pointer failure")

    private companion object {
        const val EXPECTED_TRACK_COUNT = 80_421
        const val EXPECTED_SOURCE_SHA256 =
            "08dfcec60f7c2e9de4bc6b923d601bd824f80b6251769f6c7bcd8062ce6aa504"
        const val EXPECTED_GRAPH_SHA256 =
            "65dafdae5e713f3913e6d6f082612813f6859d37d35ec9e2cdcc43f06c077656"
        val REQUIRED_MODELS = listOf(
            "mert.tflite",
            "clamp3_audio.tflite",
            "clamp3_text.tflite",
            "sentencepiece.bpe.model",
        )
    }
}
