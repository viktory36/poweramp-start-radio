package com.powerampstartradio.indexing.v2

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertThrows
import org.junit.Test

class V2StableTrackSpanIdentityTest {
    @Test
    fun `sampled identity ignores mutable stat evidence but records sampled strength`() {
        val first = V2IndexingLedgerIds.stableTrackSpanIdentity(fingerprint(), span())
        val restatted = V2IndexingLedgerIds.stableTrackSpanIdentity(
            fingerprint().copy(lastModifiedEpochMs = 999L, fileKey = "different-file-key"),
            span(),
        )

        assertEquals(first, restatted)
        assertEquals(
            StableTrackSpanIdentityStrength.VERSIONED_SAMPLED_CONTENT_SHA256,
            first.strength,
        )
        assertEquals(V2FixedRegionSampling.SPEC_ID, first.contentFingerprintSpecId)
    }

    @Test
    fun `full content hash takes precedence over sampled fingerprint details`() {
        val first = V2IndexingLedgerIds.stableTrackSpanIdentity(
            fingerprint().copy(fullContentSha256 = "f".repeat(64)),
            span(),
        )
        val changedSample = V2IndexingLedgerIds.stableTrackSpanIdentity(
            fingerprint().copy(
                fingerprintSpecId = "different-sampling-v9",
                sampledContentSha256 = "b".repeat(64),
                fullContentSha256 = "f".repeat(64),
            ),
            span(),
        )

        assertEquals(first, changedSample)
        assertEquals(StableTrackSpanIdentityStrength.FULL_CONTENT_SHA256, first.strength)
        assertEquals(V2IndexingLedgerIds.FULL_CONTENT_FINGERPRINT_SPEC_ID, first.contentFingerprintSpecId)
    }

    @Test
    fun `content and exact native span are identity boundaries`() {
        val baseline = V2IndexingLedgerIds.stableTrackSpanIdentity(fingerprint(), span())

        assertNotEquals(
            baseline.stableTrackSpanId,
            V2IndexingLedgerIds.stableTrackSpanIdentity(
                fingerprint().copy(sampledContentSha256 = "b".repeat(64)),
                span(),
            ).stableTrackSpanId,
        )
        assertNotEquals(
            baseline.stableTrackSpanId,
            V2IndexingLedgerIds.stableTrackSpanIdentity(
                fingerprint(),
                span().copy(endSourceSampleExclusive = 480_001L, sourceSampleCount = 480_001L),
            ).stableTrackSpanId,
        )
        assertNotEquals(
            baseline.stableTrackSpanId,
            V2IndexingLedgerIds.stableTrackSpanIdentity(
                fingerprint(),
                span().copy(container = span().container.copy(sampleRateHz = 44_100)),
            ).stableTrackSpanId,
        )
    }

    @Test
    fun `identity refuses a source without content evidence`() {
        assertThrows(InvalidIndexingLedgerException::class.java) {
            V2IndexingLedgerIds.stableTrackSpanIdentity(
                fingerprint().copy(sampledContentSha256 = null, fullContentSha256 = null),
                span(),
            )
        }
    }

    private fun fingerprint() = SourceFingerprint(
        fingerprintSpecId = V2FixedRegionSampling.SPEC_ID,
        sizeBytes = 123_456L,
        lastModifiedEpochMs = 100L,
        fileKey = "dev=1;ino=2",
        sampledContentSha256 = "a".repeat(64),
        fullContentSha256 = null,
    )

    private fun span() = FinalizedAudioSpanEvidence(
        kind = V2ResolvedAudioSpanKind.WHOLE_FILE,
        authority = V2AudioSpanAuthority.PROVISIONAL_END_OF_STREAM,
        executionBoundaryRequirement =
            V2ExecutionBoundaryRequirement.VERIFY_END_OF_STREAM_AND_RECONCILE,
        providerSpan = V2ProviderSpanEvidence(0L, 10_000_000L, 10_000_000L),
        cueClassification = V2CueClassificationEvidence(
            providerGroupRowCount = 1,
            logicalRowCount = 1,
            nonZeroOffsetRowIds = emptyList(),
            rawSourceImageRowIds = emptyList(),
        ),
        container = V2AudioContainerEvidence(
            physicalPath = "/music/example.flac",
            audioTrackIndex = 0,
            durationUsEstimate = 10_000_000L,
            sampleRateHz = 48_000,
            channelCount = 2,
            mime = "audio/flac",
        ),
        startUs = 0L,
        endExclusiveUs = 10_000_000L,
        startSourceSample = 0L,
        endSourceSampleExclusive = 480_000L,
        sourceSampleCount = 480_000L,
        exactSampleCount24k = 240_000L,
        expectedWork = ExpectedTrackWork(mertWindows = 2, clampSegments = 1),
    )
}
