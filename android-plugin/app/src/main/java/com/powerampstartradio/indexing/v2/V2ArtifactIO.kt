package com.powerampstartradio.indexing.v2

import java.io.EOFException
import java.io.File
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.ReadableByteChannel
import java.nio.channels.WritableByteChannel
import java.security.MessageDigest

/** Channel primitives that never assume one read/write call transfers the entire buffer. */
object V2ExactChannelIO {
    private const val MAX_ZERO_PROGRESS_CALLS = 1_024

    fun writeFully(channel: WritableByteChannel, buffer: ByteBuffer) {
        var zeroProgressCalls = 0
        while (buffer.hasRemaining()) {
            val written = channel.write(buffer)
            when {
                written < 0 -> throw IOException("channel closed before the exact write completed")
                written == 0 -> {
                    zeroProgressCalls++
                    if (zeroProgressCalls > MAX_ZERO_PROGRESS_CALLS) {
                        throw IOException("channel made no write progress")
                    }
                }
                else -> zeroProgressCalls = 0
            }
        }
    }

    fun readFully(channel: ReadableByteChannel, buffer: ByteBuffer) {
        var zeroProgressCalls = 0
        while (buffer.hasRemaining()) {
            val read = channel.read(buffer)
            when {
                read < 0 -> throw EOFException(
                    "channel ended with ${buffer.remaining()} expected bytes missing",
                )
                read == 0 -> {
                    zeroProgressCalls++
                    if (zeroProgressCalls > MAX_ZERO_PROGRESS_CALLS) {
                        throw IOException("channel made no read progress")
                    }
                }
                else -> zeroProgressCalls = 0
            }
        }
    }

    fun readExactly(channel: ReadableByteChannel, byteLength: Int): ByteArray {
        require(byteLength >= 0) { "byteLength must not be negative" }
        val bytes = ByteArray(byteLength)
        readFully(channel, ByteBuffer.wrap(bytes))
        requireExhausted(channel)
        return bytes
    }

    fun requireExhausted(channel: ReadableByteChannel) {
        val extra = ByteBuffer.allocate(1)
        var zeroProgressCalls = 0
        while (true) {
            when (val read = channel.read(extra)) {
                -1 -> return
                0 -> {
                    zeroProgressCalls++
                    if (zeroProgressCalls > MAX_ZERO_PROGRESS_CALLS) {
                        throw IOException("channel made no progress while checking EOF")
                    }
                }
                else -> throw IOException("channel contains $read or more trailing bytes")
            }
        }
    }
}

/** Exact file formats and ledger-evidence verification for intermediate V2 artifacts. */
object V2ArtifactIO {
    const val FLOATS_PER_RECORD = V2_CLAMP3_DIMENSION
    const val BYTES_PER_RECORD = V2_CLAMP3_BLOB_BYTES

    fun expectedMertByteLength(windowCount: Int): Long {
        require(windowCount > 0) { "windowCount must be positive" }
        return Math.multiplyExact(windowCount.toLong(), BYTES_PER_RECORD.toLong())
    }

    fun encodeMertWindow(feature: FloatArray): ByteArray {
        requireFiniteRecord(feature, "MERT feature")
        return encodeFloat32Record(feature)
    }

    fun decodeMertWindow(bytes: ByteArray): FloatArray {
        val result = decodeFloat32Record(bytes)
        requireFiniteRecord(result, "MERT feature")
        return result
    }

    fun sha256(file: File): String {
        require(file.isFile) { "artifact does not exist or is not a file: $file" }
        val digest = MessageDigest.getInstance("SHA-256")
        FileInputStream(file).channel.use { channel ->
            val buffer = ByteBuffer.allocateDirect(64 * 1024)
            var zeroProgressCalls = 0
            while (true) {
                buffer.clear()
                val read = channel.read(buffer)
                if (read < 0) break
                if (read == 0) {
                    zeroProgressCalls++
                    if (zeroProgressCalls > 1_024) {
                        throw IOException("artifact channel made no hash progress")
                    }
                    continue
                }
                zeroProgressCalls = 0
                buffer.flip()
                digest.update(buffer)
            }
        }
        return digest.digest().toV2CommitHex()
    }

    fun requireVerifiedFile(
        file: File,
        artifact: VerifiedArtifact,
        expectedKind: VerifiedArtifactKind,
        expectedStorageKey: String,
        expectedEmbeddingSpecId: String,
        expectedSourceFingerprint: SourceFingerprint,
        expectedPlannedUnits: Int,
    ) {
        require(expectedKind != VerifiedArtifactKind.DATABASE_COMMIT) {
            "DATABASE_COMMIT is SQLite evidence, not a file artifact"
        }
        require(artifact.kind == expectedKind) {
            "artifact kind mismatch: expected $expectedKind, got ${artifact.kind}"
        }
        require(artifact.storageKey == expectedStorageKey && expectedStorageKey.isNotBlank()) {
            "artifact storage key mismatch"
        }
        require(artifact.embeddingSpecId == expectedEmbeddingSpecId) {
            "artifact embedding spec mismatch"
        }
        require(artifact.sourceFingerprint == expectedSourceFingerprint) {
            "artifact source fingerprint mismatch"
        }
        require(expectedPlannedUnits > 0 &&
            artifact.plannedUnits == expectedPlannedUnits &&
            artifact.completedUnits == expectedPlannedUnits
        ) {
            "artifact unit evidence is incomplete or mismatched"
        }
        requireV2Sha256(artifact.sha256, "artifact sha256")
        require(artifact.verifiedAtEpochMs >= 0L) { "artifact verification time is negative" }
        if (expectedKind == VerifiedArtifactKind.MERT_FEATURES) {
            require(artifact.executionBoundary != null) {
                "MERT artifact has no verified execution boundary"
            }
        } else {
            require(artifact.executionBoundary == null) {
                "non-MERT artifact carries execution-boundary evidence"
            }
        }

        val formatLength = when (expectedKind) {
            VerifiedArtifactKind.MERT_FEATURES -> expectedMertByteLength(expectedPlannedUnits)
            VerifiedArtifactKind.CLAMP_VECTOR -> BYTES_PER_RECORD.toLong()
            VerifiedArtifactKind.DATABASE_COMMIT -> error("unreachable")
        }
        require(artifact.byteLength == formatLength) {
            "artifact length does not match its format: ${artifact.byteLength} != $formatLength"
        }
        require(file.length() == artifact.byteLength) {
            "artifact file length mismatch: ${file.length()} != ${artifact.byteLength}"
        }
        val actualSha256 = sha256(file)
        require(actualSha256 == artifact.sha256) {
            "artifact SHA-256 mismatch: $actualSha256 != ${artifact.sha256}"
        }
    }

    fun readVerifiedClampBlob(
        file: File,
        artifact: VerifiedArtifact,
        expectedStorageKey: String,
        expectedEmbeddingSpecId: String,
        expectedSourceFingerprint: SourceFingerprint,
        expectedClampSegments: Int,
    ): ByteArray {
        requireVerifiedFile(
            file = file,
            artifact = artifact,
            expectedKind = VerifiedArtifactKind.CLAMP_VECTOR,
            expectedStorageKey = expectedStorageKey,
            expectedEmbeddingSpecId = expectedEmbeddingSpecId,
            expectedSourceFingerprint = expectedSourceFingerprint,
            expectedPlannedUnits = expectedClampSegments,
        )
        val bytes = FileInputStream(file).channel.use { channel ->
            V2ExactChannelIO.readExactly(channel, BYTES_PER_RECORD)
        }
        V2Clamp3VectorCodec.requireValidBlob(bytes)
        return bytes
    }

    fun forEachVerifiedMertWindow(
        file: File,
        artifact: VerifiedArtifact,
        expectedStorageKey: String,
        expectedEmbeddingSpecId: String,
        expectedSourceFingerprint: SourceFingerprint,
        expectedWindows: Int,
        expectedFinalizedAudioSpan: FinalizedAudioSpanEvidence,
        block: (windowIndex: Int, feature: FloatArray) -> Unit,
    ) {
        requireVerifiedFile(
            file = file,
            artifact = artifact,
            expectedKind = VerifiedArtifactKind.MERT_FEATURES,
            expectedStorageKey = expectedStorageKey,
            expectedEmbeddingSpecId = expectedEmbeddingSpecId,
            expectedSourceFingerprint = expectedSourceFingerprint,
            expectedPlannedUnits = expectedWindows,
        )
        requireExecutionBoundaryMatches(
            expectedFinalizedAudioSpan,
            requireNotNull(artifact.executionBoundary),
        )
        FileInputStream(file).channel.use { channel ->
            repeat(expectedWindows) { index ->
                val bytes = ByteArray(BYTES_PER_RECORD)
                V2ExactChannelIO.readFully(channel, ByteBuffer.wrap(bytes))
                block(index, decodeMertWindow(bytes))
            }
            V2ExactChannelIO.requireExhausted(channel)
        }
    }

    fun requireExecutionBoundaryMatches(
        span: FinalizedAudioSpanEvidence,
        boundary: VerifiedExecutionBoundaryEvidence,
    ) {
        require(boundary.requirement == span.executionBoundaryRequirement) {
            "execution-boundary requirement mismatch"
        }
        require(boundary.observedStartSourceSample == span.startSourceSample &&
            boundary.observedEndSourceSampleExclusive == span.endSourceSampleExclusive &&
            boundary.observedSourceSampleCount == span.sourceSampleCount &&
            boundary.exactSampleCount24k == span.exactSampleCount24k
        ) {
            "observed decoder boundary disagrees with finalized acoustic span"
        }
        when (span.executionBoundaryRequirement) {
            V2ExecutionBoundaryRequirement.VERIFY_END_OF_STREAM_AND_RECONCILE -> {
                require(boundary.endOfStreamReached && !boundary.providerBoundaryEnforced) {
                    "ordinary span has not been reconciled against decoder EOS"
                }
            }
            V2ExecutionBoundaryRequirement.ENFORCE_PROVIDER_HALF_OPEN_SPAN -> {
                require(boundary.providerBoundaryEnforced) {
                    "CUE provider half-open boundary was not enforced"
                }
            }
        }
    }

    private fun requireFiniteRecord(values: FloatArray, label: String) {
        require(values.size == FLOATS_PER_RECORD) {
            "$label must have $FLOATS_PER_RECORD values, got ${values.size}"
        }
        values.forEachIndexed { index, value ->
            require(value.isFinite()) { "$label contains a non-finite value at $index" }
        }
    }

    private fun encodeFloat32Record(values: FloatArray): ByteArray =
        ByteBuffer.allocate(BYTES_PER_RECORD)
            .order(ByteOrder.LITTLE_ENDIAN)
            .also { buffer -> values.forEach(buffer::putFloat) }
            .array()

    private fun decodeFloat32Record(bytes: ByteArray): FloatArray {
        require(bytes.size == BYTES_PER_RECORD) {
            "float32 record must be exactly $BYTES_PER_RECORD bytes, got ${bytes.size}"
        }
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        return FloatArray(FLOATS_PER_RECORD) { buffer.float }
    }
}
