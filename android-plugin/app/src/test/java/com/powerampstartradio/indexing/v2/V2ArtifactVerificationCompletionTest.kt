package com.powerampstartradio.indexing.v2

import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

class V2ArtifactVerificationCompletionTest {
    @Test
    fun `stamps the time observed after verification`() {
        val calls = mutableListOf<String>()
        val candidate = artifact(verifiedAtEpochMs = 0L)

        val completed = V2ArtifactVerificationCompletion.stamp(
            verify = {
                calls += "verified"
                candidate
            },
            nowEpochMs = {
                calls += "clock"
                2_000L
            },
        )

        assertEquals(listOf("verified", "clock"), calls)
        assertEquals(2_000L, completed.verifiedAtEpochMs)
        assertEquals(candidate.copy(verifiedAtEpochMs = 2_000L), completed)
    }

    @Test
    fun `rejects an impossible completion timestamp`() {
        assertThrows(IllegalArgumentException::class.java) {
            V2ArtifactVerificationCompletion.stamp(
                verify = { artifact(verifiedAtEpochMs = 0L) },
                nowEpochMs = { -1L },
            )
        }
    }

    @Test
    fun `does not claim a time when verification fails`() {
        var clockReads = 0

        assertThrows(IllegalStateException::class.java) {
            V2ArtifactVerificationCompletion.stamp(
                verify = { error("verification failed") },
                nowEpochMs = {
                    clockReads++
                    2_000L
                },
            )
        }

        assertEquals(0, clockReads)
    }

    private fun artifact(verifiedAtEpochMs: Long) = VerifiedArtifact(
        kind = VerifiedArtifactKind.MERT_FEATURES,
        storageKey = "mert.bin",
        byteLength = V2_CLAMP3_BLOB_BYTES.toLong(),
        sha256 = "a".repeat(64),
        completedUnits = 1,
        plannedUnits = 1,
        embeddingSpecId = "embedding-spec",
        sourceFingerprint = SourceFingerprint(
            fingerprintSpecId = "source-fingerprint-test-v1",
            sizeBytes = 1L,
            lastModifiedEpochMs = 1L,
            fileKey = "test-file",
            sampledContentSha256 = "b".repeat(64),
            fullContentSha256 = null,
        ),
        verifiedAtEpochMs = verifiedAtEpochMs,
    )
}
