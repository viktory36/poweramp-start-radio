package com.powerampstartradio.indexing.v2

import java.nio.charset.StandardCharsets
import java.security.MessageDigest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class V2ServerBundleContractTest {
    @get:Rule
    val temporaryFolder = TemporaryFolder()

    @Test
    fun serverRelativePathEncodingMatchesPythonRfc3986Quote() {
        assertEquals(
            "Albums/Don%27t%20%28Live%29%21/%E5%A4%9C.flac",
            V2ServerBundlePathPolicy.encodeServerRelativePath(
                "Albums/Don't (Live)!/夜.flac",
            ),
        )
    }

    @Test
    fun canonicalEmbeddingSpecHasItsPinnedHashAndClaimsCompleteAudio() {
        val digest = MessageDigest.getInstance("SHA-256")
            .digest(V2ServerBundleContract.EMBEDDING_SPEC_JSON.toByteArray(StandardCharsets.UTF_8))
            .joinToString("") { byte ->
                (byte.toInt() and 0xff).toString(16).padStart(2, '0')
            }
        assertEquals(V2ServerBundleContract.EMBEDDING_SPEC_SHA256, digest)
        assertTrue(
            V2ServerBundleContract.EMBEDDING_SPEC_JSON.contains(
                "\"audio_span\":\"complete-physical-file\"",
            ),
        )
        assertTrue(
            V2ServerBundleContract.EMBEDDING_SPEC_JSON.contains(
                "\"maximum_duration_seconds\":null",
            ),
        )
    }

    @Test
    fun logicalBundleIdentityMatchesPythonCanonicalJson() {
        val record = V2ServerBundleValidator.logicalIdRecord(
            album = "Album 夜",
            artist = "Don't!",
            durationMs = 123_456L,
            embeddingSha256 = "1".repeat(64),
            filenameKey = "don't - live",
            filePath = "server://test/Albums/Don%27t%20%28Live%29%21/%E5%A4%9C.flac",
            metadataKey = "don't!|album 夜|live|123400",
            relativePath = "Albums/Don't (Live)!/夜.flac",
            rootId = "test",
            sourceSampleCount = 987_654L,
            sourceSampleRateHz = 48_000,
            sourceSha256 = "2".repeat(64),
            sourceSizeBytes = 12_345_678L,
            title = "Live (夜)",
        )
        assertEquals(
            "{\"album\":\"Album \\u591c\",\"artist\":\"Don't!\"," +
                "\"duration_ms\":123456,\"embedding_sha256\":\"${"1".repeat(64)}\"," +
                "\"file_path\":\"server://test/Albums/Don%27t%20%28Live%29%21/" +
                "%E5%A4%9C.flac\",\"filename_key\":\"don't - live\"," +
                "\"metadata_key\":\"don't!|album \\u591c|live|123400\"," +
                "\"relative_path\":\"Albums/Don't (Live)!/\\u591c.flac\"," +
                "\"root_id\":\"test\",\"source_sample_count\":987654," +
                "\"source_sample_rate_hz\":48000,\"source_sha256\":\"${"2".repeat(64)}\"," +
                "\"source_size_bytes\":12345678,\"title\":\"Live (\\u591c)\"}",
            record,
        )
        assertEquals(
            "server-bundle-v1-72a7ddb32c362a684dfd2c994677f480ebcfb8f5e5a787654281ec107443d2b0",
            V2ServerBundleValidator.logicalBundleId(listOf(record)),
        )
    }

    @Test
    fun exactPathAssignmentsReserveEachProviderOnce() {
        val suffix = V2ServerBundleReciprocalAssignmentPolicy.reserveUnique(
            edges = mapOf(
                10L to emptySet(),
                20L to setOf(200L),
            ),
        )
        assertEquals(mapOf(20L to 200L), suffix)
    }

    @Test
    fun reciprocalAssignmentRejectsContendedExactBytes() {
        assertTrue(
            V2ServerBundleReciprocalAssignmentPolicy.reserveUnique(
                edges = mapOf(
                    10L to setOf(200L),
                    20L to setOf(200L),
                ),
            ).isEmpty(),
        )
    }

    @Test
    fun sameLengthWrongBytesCannotAuthorizeASyncedSource() {
        val phoneBytes = byteArrayOf(1, 2, 3, 4)
        val serverBytes = byteArrayOf(4, 3, 2, 1)
        val source = temporaryFolder.newFile("wrong-bytes.flac").apply {
            writeBytes(phoneBytes)
        }
        var fingerprintReads = 0

        val match = V2ServerBundleSourceMatchPolicy.matchExactSource(
            sourceFile = source,
            bundle = bundle(source.length(), sha256(serverBytes)),
        ) { exactFile ->
            fingerprintReads += 1
            V2ExactSourceFingerprinter().fingerprint(exactFile)
        }

        assertNull(match)
        assertEquals(1, fingerprintReads)
    }

    @Test
    fun byteLengthMismatchIsRejectedBeforeReadingContent() {
        val source = temporaryFolder.newFile("short.flac").apply {
            writeBytes(byteArrayOf(1, 2, 3))
        }
        var fingerprintReads = 0

        assertNull(
            V2ServerBundleSourceMatchPolicy.matchExactSource(
                sourceFile = source,
                bundle = bundle(source.length() + 1L, "b".repeat(64)),
            ) {
                fingerprintReads += 1
                error("A wrong-length candidate must not be read")
            },
        )
        assertEquals(0, fingerprintReads)
    }

    @Test
    fun exactServerBytesAuthorizeTheSourceWithFingerprintEvidence() {
        val serverBytes = byteArrayOf(4, 3, 2, 1)
        val source = temporaryFolder.newFile("exact.flac").apply {
            writeBytes(serverBytes)
        }
        var fingerprintReads = 0

        val match = V2ServerBundleSourceMatchPolicy.matchExactSource(
            sourceFile = source,
            bundle = bundle(source.length(), sha256(serverBytes)),
        ) { exactFile ->
            fingerprintReads += 1
            V2ExactSourceFingerprinter().fingerprint(exactFile)
        }

        checkNotNull(match)
        assertEquals(1, fingerprintReads)
        assertEquals(source.canonicalFile, match.canonicalFile)
        assertEquals(source.length(), match.observedSizeBytes)
        assertEquals(sha256(serverBytes), match.exactFingerprint.fullContentSha256)
        assertEquals(V2ServerBundleMatchEvidence.FULL_CONTENT_SHA256, match.evidence)
    }

    private fun sha256(bytes: ByteArray): String =
        MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { byte ->
            (byte.toInt() and 0xff).toString(16).padStart(2, '0')
        }

    private fun bundle(sizeBytes: Long, sourceSha256: String) = V2ServerBundleTrack(
        trackId = 1L,
        rootId = "musicnew",
        relativePath = "Albums/夜.flac",
        sourceSha256 = sourceSha256,
        sourceSizeBytes = sizeBytes,
        sourceSampleRateHz = 48_000,
        sourceSampleCount = 48_000L,
        spanStartSample = 0L,
        spanEndSampleExclusive = 48_000L,
        embeddingSha256 = "d".repeat(64),
    )
}
