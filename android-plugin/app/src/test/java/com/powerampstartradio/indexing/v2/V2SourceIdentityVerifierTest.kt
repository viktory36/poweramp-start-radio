package com.powerampstartradio.indexing.v2

import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File

class V2SourceIdentityVerifierTest {
    @get:Rule
    val temporaryFolder = TemporaryFolder()

    @Test
    fun `exact verification ignores file key changes after process restart`() {
        val source = sourceFile("restart.flac", "same exact bytes")
        val planned = fingerprint(source).copy(fileKey = "external-storage-inode-before")
        val afterRestart = planned.copy(fileKey = "external-storage-inode-after")

        val before = verifierReturning(planned).requireVerified(
            source.path,
            source.canonicalPath,
            41L,
            planned,
            exactContent = true,
        )
        val resumed = verifierReturning(afterRestart).requireVerified(
            source.path,
            source.canonicalPath,
            41L,
            planned,
            exactContent = true,
        )

        assertEquals(source.canonicalFile, before)
        assertEquals(source.canonicalFile, resumed)
    }

    @Test
    fun `preflight content revalidation ignores mutable stat evidence`() {
        val source = sourceFile("revalidation.flac", "same exact bytes")
        val planned = fingerprint(source).copy(
            lastModifiedEpochMs = 100L,
            fileKey = "external-storage-inode-before",
        )
        val revalidated = planned.copy(
            lastModifiedEpochMs = 200L,
            fileKey = "external-storage-inode-after",
        )

        assertEquals(true, V2ExactSourceContentIdentity.matches(planned, revalidated))
        assertEquals(
            false,
            V2ExactSourceContentIdentity.matches(
                planned,
                revalidated.copy(fullContentSha256 = "f".repeat(64)),
            ),
        )
    }

    @Test
    fun `mtime only change falls back to exact content and remains valid`() {
        val source = sourceFile("retimestamped.flac", "same exact bytes")
        val planned = fingerprint(source)
        val changedModified = planned.lastModifiedEpochMs!! + 20_000L
        require(source.setLastModified(changedModified))
        var fingerprintCalls = 0
        val verifier = V2SourceIdentityVerifier { current ->
            fingerprintCalls++
            fingerprint(current)
        }

        val verified = verifier.requireVerified(
            source.path,
            source.canonicalPath,
            42L,
            planned,
            exactContent = false,
        )

        assertEquals(source.canonicalFile, verified)
        assertEquals(1, fingerprintCalls)
    }

    @Test
    fun `quick verification with unchanged stat defers hashing to exact boundary`() {
        val source = sourceFile("quick.flac", "same exact bytes")
        val planned = fingerprint(source)
        val verifier = V2SourceIdentityVerifier {
            throw AssertionError("unchanged quick stat must not perform another full file read")
        }

        val verified = verifier.requireVerified(
            source.path,
            source.canonicalPath,
            45L,
            planned,
            exactContent = false,
        )

        assertEquals(source.canonicalFile, verified)
    }

    @Test
    fun `same size changed bytes are rejected even when stat evidence is unchanged`() {
        val source = sourceFile("mutated.flac", "original")
        val planned = fingerprint(source)
        source.writeText("mutation")
        require(source.setLastModified(planned.lastModifiedEpochMs!!))

        assertThrows(V2SourceIdentityChangedException::class.java) {
            V2SourceIdentityVerifier(V2ExactSourceFingerprinter()).requireVerified(
                source.path,
                source.canonicalPath,
                43L,
                planned,
                exactContent = true,
            )
        }
    }

    @Test
    fun `size change fails before an equal hash claim can be accepted`() {
        val source = sourceFile("resized.flac", "original")
        val planned = fingerprint(source)
        source.appendText(" more")
        var fingerprintCalls = 0

        assertThrows(V2SourceIdentityChangedException::class.java) {
            V2SourceIdentityVerifier {
                fingerprintCalls++
                planned.copy(sizeBytes = source.length())
            }.requireVerified(
                source.path,
                source.canonicalPath,
                46L,
                planned,
                exactContent = true,
            )
        }
        assertEquals(0, fingerprintCalls)
    }

    @Test
    fun `exact verification fails closed without full content hashes`() {
        val source = sourceFile("missing-hash.flac", "same exact bytes")
        val exact = fingerprint(source)

        assertThrows(V2SourceIdentityChangedException::class.java) {
            verifierReturning(exact).requireVerified(
                source.path,
                source.canonicalPath,
                47L,
                exact.copy(fullContentSha256 = null, sampledContentSha256 = "a".repeat(64)),
                exactContent = true,
            )
        }
        assertThrows(V2SourceIdentityChangedException::class.java) {
            verifierReturning(exact.copy(fullContentSha256 = null)).requireVerified(
                source.path,
                source.canonicalPath,
                47L,
                exact,
                exactContent = true,
            )
        }
    }

    @Test
    fun `canonical provider target remains part of the immutable binding`() {
        val source = sourceFile("bound.flac", "same exact bytes")
        val other = sourceFile("other.flac", "same exact bytes")
        val planned = fingerprint(source)

        assertThrows(V2SourceIdentityChangedException::class.java) {
            verifierReturning(planned).requireVerified(
                other.path,
                source.canonicalPath,
                44L,
                planned,
                exactContent = true,
            )
        }
    }

    @Test
    fun `canonical binding uses the same unicode normalization as planning`() {
        val decomposedName = "caf\u0065\u0301.flac"
        val source = sourceFile(decomposedName, "same exact bytes")
        val planned = fingerprint(source)
        val plannedCanonicalPath = V2IndexingLedgerIds.canonicalPath(source.canonicalPath)

        val verified = verifierReturning(planned).requireVerified(
            source.path,
            plannedCanonicalPath,
            48L,
            planned,
            exactContent = true,
        )

        assertEquals(source.canonicalFile, verified)
    }

    private fun verifierReturning(current: SourceFingerprint) =
        V2SourceIdentityVerifier(V2SourceFingerprintProvider { current })

    private fun sourceFile(name: String, bytes: String): File =
        temporaryFolder.newFile(name).apply { writeText(bytes) }

    private fun fingerprint(file: File): SourceFingerprint = V2ExactSourceFingerprinter()
        .fingerprint(file)
}
