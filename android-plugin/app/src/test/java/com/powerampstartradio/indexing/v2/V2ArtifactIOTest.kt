package com.powerampstartradio.indexing.v2

import java.io.ByteArrayOutputStream
import java.io.EOFException
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.channels.ReadableByteChannel
import java.nio.channels.WritableByteChannel
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

class V2ArtifactIOTest {
    @Test
    fun `exact channel helpers handle partial reads and writes`() {
        val expected = ByteArray(97) { it.toByte() }
        val sink = ChunkedWritableChannel(maxChunk = 3)
        V2ExactChannelIO.writeFully(sink, ByteBuffer.wrap(expected))
        assertArrayEquals(expected, sink.bytes())

        val source = ChunkedReadableChannel(expected, maxChunk = 5)
        assertArrayEquals(expected, V2ExactChannelIO.readExactly(source, expected.size))
    }

    @Test
    fun `exact reads reject truncation and trailing bytes`() {
        assertThrows(EOFException::class.java) {
            V2ExactChannelIO.readExactly(
                ChunkedReadableChannel(byteArrayOf(1, 2), maxChunk = 1),
                3,
            )
        }
        assertThrows(IOException::class.java) {
            V2ExactChannelIO.readExactly(
                ChunkedReadableChannel(byteArrayOf(1, 2, 3, 4), maxChunk = 2),
                3,
            )
        }
    }

    @Test
    fun `verified MERT artifact streams exact finite windows`() {
        val first = FloatArray(V2_CLAMP3_DIMENSION) { index -> index / 1000f }
        val second = FloatArray(V2_CLAMP3_DIMENSION) { index -> -index / 2000f }
        val file = tempFile().also { target ->
            FileOutputStream(target).channel.use { channel ->
                listOf(first, second).forEach { feature ->
                    V2ExactChannelIO.writeFully(
                        channel,
                        ByteBuffer.wrap(V2ArtifactIO.encodeMertWindow(feature)),
                    )
                }
            }
        }
        val artifact = artifact(
            kind = VerifiedArtifactKind.MERT_FEATURES,
            storageKey = "mert/test.bin",
            byteLength = 2L * V2_CLAMP3_BLOB_BYTES,
            sha256 = V2ArtifactIO.sha256(file),
            units = 2,
        )

        val observed = mutableListOf<FloatArray>()
        V2ArtifactIO.forEachVerifiedMertWindow(
            file = file,
            artifact = artifact,
            expectedStorageKey = "mert/test.bin",
            expectedEmbeddingSpecId = SPEC_ID,
            expectedSourceFingerprint = fingerprint,
            expectedWindows = 2,
            expectedFinalizedAudioSpan = finalizedSpan,
        ) { index, feature ->
            assertEquals(observed.size, index)
            observed += feature
        }

        assertArrayEquals(first, observed[0], 0f)
        assertArrayEquals(second, observed[1], 0f)
        file.delete()
    }

    @Test
    fun `verified CLaMP artifact rejects same-length mutation`() {
        val vector = FloatArray(V2_CLAMP3_DIMENSION).apply { this[7] = 1f }
        val bytes = V2Clamp3VectorCodec.encode(vector)
        val file = tempFile().also { it.writeBytes(bytes) }
        val artifact = artifact(
            kind = VerifiedArtifactKind.CLAMP_VECTOR,
            storageKey = "clamp/test.bin",
            byteLength = V2_CLAMP3_BLOB_BYTES.toLong(),
            sha256 = V2ArtifactIO.sha256(file),
            units = 3,
        )

        assertArrayEquals(
            bytes,
            V2ArtifactIO.readVerifiedClampBlob(
                file = file,
                artifact = artifact,
                expectedStorageKey = "clamp/test.bin",
                expectedEmbeddingSpecId = SPEC_ID,
                expectedSourceFingerprint = fingerprint,
                expectedClampSegments = 3,
            ),
        )

        RandomAccessFile(file, "rw").use { random ->
            random.seek(10)
            val original = random.read()
            random.seek(10)
            random.write(original.xor(0x01))
        }
        assertThrows(IllegalArgumentException::class.java) {
            V2ArtifactIO.requireVerifiedFile(
                file = file,
                artifact = artifact,
                expectedKind = VerifiedArtifactKind.CLAMP_VECTOR,
                expectedStorageKey = "clamp/test.bin",
                expectedEmbeddingSpecId = SPEC_ID,
                expectedSourceFingerprint = fingerprint,
                expectedPlannedUnits = 3,
            )
        }
        file.delete()
    }

    @Test
    fun `ledger field mismatches fail before an artifact can be consumed`() {
        val vector = FloatArray(V2_CLAMP3_DIMENSION).apply { this[0] = 1f }
        val file = tempFile().also { it.writeBytes(V2Clamp3VectorCodec.encode(vector)) }
        val artifact = artifact(
            kind = VerifiedArtifactKind.CLAMP_VECTOR,
            storageKey = "clamp/test.bin",
            byteLength = V2_CLAMP3_BLOB_BYTES.toLong(),
            sha256 = V2ArtifactIO.sha256(file),
            units = 2,
        )

        assertThrows(IllegalArgumentException::class.java) {
            V2ArtifactIO.requireVerifiedFile(
                file,
                artifact,
                VerifiedArtifactKind.CLAMP_VECTOR,
                "different-key",
                SPEC_ID,
                fingerprint,
                2,
            )
        }
        assertThrows(IllegalArgumentException::class.java) {
            V2ArtifactIO.requireVerifiedFile(
                file,
                artifact,
                VerifiedArtifactKind.CLAMP_VECTOR,
                "clamp/test.bin",
                "different-spec",
                fingerprint,
                2,
            )
        }
        assertThrows(IllegalArgumentException::class.java) {
            V2ArtifactIO.requireVerifiedFile(
                file,
                artifact,
                VerifiedArtifactKind.CLAMP_VECTOR,
                "clamp/test.bin",
                SPEC_ID,
                fingerprint,
                1,
            )
        }
        file.delete()
    }

    private fun artifact(
        kind: VerifiedArtifactKind,
        storageKey: String,
        byteLength: Long,
        sha256: String,
        units: Int,
    ) = VerifiedArtifact(
        kind = kind,
        storageKey = storageKey,
        byteLength = byteLength,
        sha256 = sha256,
        completedUnits = units,
        plannedUnits = units,
        embeddingSpecId = SPEC_ID,
        sourceFingerprint = fingerprint,
        verifiedAtEpochMs = 1_000L,
        executionBoundary = if (kind == VerifiedArtifactKind.MERT_FEATURES) {
            executionBoundary
        } else {
            null
        },
    )

    private fun tempFile(): File = File.createTempFile("v2-artifact-", ".bin")

    private val fingerprint = SourceFingerprint(
        fingerprintSpecId = "source-test-v1",
        sizeBytes = 123_456L,
        lastModifiedEpochMs = 900L,
        fileKey = "test-file",
        sampledContentSha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        fullContentSha256 = null,
    )
    private val finalizedSpan = FinalizedAudioSpanEvidence(
        kind = V2ResolvedAudioSpanKind.WHOLE_FILE,
        authority = V2AudioSpanAuthority.PROVISIONAL_END_OF_STREAM,
        executionBoundaryRequirement =
            V2ExecutionBoundaryRequirement.VERIFY_END_OF_STREAM_AND_RECONCILE,
        providerSpan = V2ProviderSpanEvidence(0L, 6_000_000L, 6_000_000L),
        cueClassification = V2CueClassificationEvidence(1, 1, emptyList(), emptyList()),
        container = V2AudioContainerEvidence(
            physicalPath = "/test/audio.flac",
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
    private val executionBoundary = VerifiedExecutionBoundaryEvidence(
        requirement = V2ExecutionBoundaryRequirement.VERIFY_END_OF_STREAM_AND_RECONCILE,
        observedStartSourceSample = 0L,
        observedEndSourceSampleExclusive = 144_000L,
        observedSourceSampleCount = 144_000L,
        exactSampleCount24k = 144_000L,
        endOfStreamReached = true,
        providerBoundaryEnforced = false,
    )

    private class ChunkedReadableChannel(
        private val bytes: ByteArray,
        private val maxChunk: Int,
    ) : ReadableByteChannel {
        private var offset = 0
        private var open = true

        override fun read(destination: ByteBuffer): Int {
            check(open)
            if (offset == bytes.size) return -1
            val count = minOf(maxChunk, destination.remaining(), bytes.size - offset)
            destination.put(bytes, offset, count)
            offset += count
            return count
        }

        override fun isOpen(): Boolean = open

        override fun close() {
            open = false
        }
    }

    private class ChunkedWritableChannel(
        private val maxChunk: Int,
    ) : WritableByteChannel {
        private val output = ByteArrayOutputStream()
        private var open = true

        override fun write(source: ByteBuffer): Int {
            check(open)
            val count = minOf(maxChunk, source.remaining())
            val chunk = ByteArray(count)
            source.get(chunk)
            output.write(chunk)
            return count
        }

        fun bytes(): ByteArray = output.toByteArray()

        override fun isOpen(): Boolean = open

        override fun close() {
            open = false
        }
    }

    companion object {
        private const val SPEC_ID = "embedding-spec-v1-test"
    }
}
