package com.powerampstartradio.indexing.v2

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import java.util.UUID
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class AtomicV2ArtifactStoreInstrumentedTest {
    @Test
    fun incompletePublicationRestoresOldFileAndCompleteArtifactsVerify() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val directory = File(context.cacheDir, "v2-artifacts-${UUID.randomUUID()}")
        val mertFile = File(directory, "mert.bin")
        val clampFile = File(directory, "clamp.bin")
        check(directory.mkdirs())
        val original = "previous-complete-artifact".toByteArray()
        mertFile.writeBytes(original)
        val feature = FloatArray(V2_CLAMP3_DIMENSION) { index -> index / 10_000f }
        val fingerprint = fingerprint()
        val finalizedSpan = finalizedSpan()
        val boundary = executionBoundary()
        val store = AtomicV2ArtifactStore()

        try {
            assertThrows(IllegalArgumentException::class.java) {
                store.publishMertFeatures(
                    target = mertFile,
                    storageKey = "mert/instrumented.bin",
                    features = sequenceOf(feature, feature.copyOf()),
                    expectedWindows = 2,
                    finalizedAudioSpan = finalizedSpan.copy(
                        authority = V2AudioSpanAuthority.PROVISIONAL_END_OF_STREAM,
                    ),
                    executionBoundary = boundary,
                    embeddingSpecId = SPEC_ID,
                    sourceFingerprint = fingerprint,
                    verifiedAtEpochMs = 900L,
                )
            }
            assertArrayEquals(original, mertFile.readBytes())

            assertThrows(IllegalArgumentException::class.java) {
                store.publishMertFeatures(
                    target = mertFile,
                    storageKey = "mert/instrumented.bin",
                    features = sequenceOf(feature),
                    expectedWindows = 2,
                    finalizedAudioSpan = finalizedSpan,
                    executionBoundary = boundary,
                    embeddingSpecId = SPEC_ID,
                    sourceFingerprint = fingerprint,
                    verifiedAtEpochMs = 1_000L,
                )
            }
            assertArrayEquals(original, mertFile.readBytes())

            val mertArtifact = store.publishMertFeatures(
                target = mertFile,
                storageKey = "mert/instrumented.bin",
                features = sequenceOf(feature, feature.copyOf()),
                expectedWindows = 2,
                finalizedAudioSpan = finalizedSpan,
                executionBoundary = boundary,
                embeddingSpecId = SPEC_ID,
                sourceFingerprint = fingerprint,
                verifiedAtEpochMs = 2_000L,
            )
            assertEquals(2L * V2_CLAMP3_BLOB_BYTES, mertFile.length())
            assertEquals(V2ArtifactIO.sha256(mertFile), mertArtifact.sha256)

            var windows = 0
            V2ArtifactIO.forEachVerifiedMertWindow(
                file = mertFile,
                artifact = mertArtifact,
                expectedStorageKey = "mert/instrumented.bin",
                expectedEmbeddingSpecId = SPEC_ID,
                expectedSourceFingerprint = fingerprint,
                expectedWindows = 2,
                expectedFinalizedAudioSpan = finalizedSpan,
            ) { _, observed ->
                assertArrayEquals(feature, observed, 0f)
                windows++
            }
            assertEquals(2, windows)

            val vector = FloatArray(V2_CLAMP3_DIMENSION).apply { this[17] = 1f }
            val clampArtifact = store.publishClampVector(
                target = clampFile,
                storageKey = "clamp/instrumented.bin",
                vector = vector,
                completedClampSegments = 4,
                embeddingSpecId = SPEC_ID,
                sourceFingerprint = fingerprint,
                verifiedAtEpochMs = 3_000L,
            )
            assertArrayEquals(
                V2Clamp3VectorCodec.encode(vector),
                V2ArtifactIO.readVerifiedClampBlob(
                    file = clampFile,
                    artifact = clampArtifact,
                    expectedStorageKey = "clamp/instrumented.bin",
                    expectedEmbeddingSpecId = SPEC_ID,
                    expectedSourceFingerprint = fingerprint,
                    expectedClampSegments = 4,
                ),
            )

            val aliasFingerprint = fingerprint().copy(fileKey = "second-poweramp-locator")
            val mertAliasFile = File(directory, "mert-alias.bin")
            val mertAlias = store.publishMertAlias(
                source = mertFile,
                sourceArtifact = mertArtifact,
                target = mertAliasFile,
                targetStorageKey = "mert/alias.bin",
                targetSpan = finalizedSpan(),
                targetSourceFingerprint = aliasFingerprint,
                verifiedAtEpochMs = 4_000L,
            )
            assertArrayEquals(mertFile.readBytes(), mertAliasFile.readBytes())
            assertEquals(mertArtifact.sha256, mertAlias.sha256)
            assertEquals(aliasFingerprint, mertAlias.sourceFingerprint)

            val clampAliasFile = File(directory, "clamp-alias.bin")
            val clampAlias = store.publishClampAlias(
                source = clampFile,
                sourceArtifact = clampArtifact,
                target = clampAliasFile,
                targetStorageKey = "clamp/alias.bin",
                targetSourceFingerprint = aliasFingerprint,
                expectedClampSegments = 4,
                verifiedAtEpochMs = 5_000L,
            )
            assertArrayEquals(clampFile.readBytes(), clampAliasFile.readBytes())
            assertEquals(clampArtifact.sha256, clampAlias.sha256)
            assertEquals(aliasFingerprint, clampAlias.sourceFingerprint)
        } finally {
            directory.deleteRecursively()
        }
    }

    private fun fingerprint() = SourceFingerprint(
        fingerprintSpecId = "instrumented-artifact-source-v1",
        sizeBytes = 123_456L,
        lastModifiedEpochMs = 500L,
        fileKey = "instrumented-artifact-file",
        sampledContentSha256 =
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        fullContentSha256 = null,
    )

    private fun finalizedSpan() = FinalizedAudioSpanEvidence(
        kind = V2ResolvedAudioSpanKind.WHOLE_FILE,
        authority = V2AudioSpanAuthority.DECODED_END_OF_STREAM,
        executionBoundaryRequirement =
            V2ExecutionBoundaryRequirement.VERIFY_END_OF_STREAM_AND_RECONCILE,
        providerSpan = V2ProviderSpanEvidence(0L, 6_000_000L, 6_000_000L),
        cueClassification = V2CueClassificationEvidence(1, 1, emptyList(), emptyList()),
        container = V2AudioContainerEvidence(
            physicalPath = "/instrumented/audio.flac",
            audioTrackIndex = 0,
            durationUsEstimate = 6_000_000L,
            sampleRateHz = 24_000,
            channelCount = 2,
            mime = "audio/flac",
        ),
        startUs = 0L,
        endExclusiveUs = 6_000_000L,
        startSourceSample = 0L,
        endSourceSampleExclusive = 144_000L,
        sourceSampleCount = 144_000L,
        exactSampleCount24k = 144_000L,
        expectedWork = ExpectedTrackWork(2, 1),
    )

    private fun executionBoundary() = VerifiedExecutionBoundaryEvidence(
        requirement = V2ExecutionBoundaryRequirement.VERIFY_END_OF_STREAM_AND_RECONCILE,
        observedStartSourceSample = 0L,
        observedEndSourceSampleExclusive = 144_000L,
        observedSourceSampleCount = 144_000L,
        exactSampleCount24k = 144_000L,
        endOfStreamReached = true,
        providerBoundaryEnforced = false,
    )

    companion object {
        private const val SPEC_ID = "embedding-spec-v1-instrumented"
    }
}
