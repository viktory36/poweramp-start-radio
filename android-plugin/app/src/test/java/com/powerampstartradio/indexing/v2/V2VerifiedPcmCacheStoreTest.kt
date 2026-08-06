package com.powerampstartradio.indexing.v2

import java.io.File
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class V2VerifiedPcmCacheStoreTest {
    @get:Rule
    val temporary = TemporaryFolder()

    @Test
    fun `PCM without receipt is deleted before verification`() {
        val root = temporary.newFolder("pcm-only")
        val store = V2VerifiedPcmCacheStore()
        val pcm = store.pcmFile(root, POWERAMP_FILE_ID)
        val receiptSidecar = File(receiptFile(root).path + ".new")
        pcm.writeBytes(byteArrayOf(1, 2, 3, 4))
        receiptSidecar.writeText("partial")

        assertNull(store.loadVerified(root, JOB_ID, descriptor()))

        assertFalse(pcm.exists())
        assertFalse(receiptSidecar.exists())
    }

    @Test
    fun `receipt without PCM is deleted before verification`() {
        val root = temporary.newFolder("receipt-only")
        val store = V2VerifiedPcmCacheStore()
        val pcmSidecar = File(store.pcmFile(root, POWERAMP_FILE_ID).path + ".bak")
        val receipt = receiptFile(root)
        pcmSidecar.writeText("stale")
        receipt.writeText("{}")

        assertNull(store.loadVerified(root, JOB_ID, descriptor()))

        assertFalse(pcmSidecar.exists())
        assertFalse(receipt.exists())
    }

    private fun receiptFile(root: File): File =
        File(root, "pcm-cache-v1/$POWERAMP_FILE_ID.receipt.json")

    private fun descriptor(): SelectedTrackDescriptor {
        val path = "/storage/emulated/0/Music/test.flac"
        val samples = 48_000L
        val fingerprint = SourceFingerprint(
            fingerprintSpecId = "source-test-v1",
            sizeBytes = 1_024L,
            lastModifiedEpochMs = 1_000L,
            fileKey = "test-file",
            sampledContentSha256 = SHA,
            fullContentSha256 = null,
        )
        return SelectedTrackDescriptor(
            workId = "work-v4-$SHA",
            stableTrackSpanIdentity = StableTrackSpanIdentity(
                identitySpecId = "stable-track-span-v1",
                stableTrackSpanId = "stable-track-span-v1-$SHA",
                strength = StableTrackSpanIdentityStrength.VERSIONED_SAMPLED_CONTENT_SHA256,
                contentFingerprintSpecId = fingerprint.fingerprintSpecId,
                contentSha256 = SHA,
                sourceSizeBytes = fingerprint.sizeBytes,
                sourceSampleRateHz = 24_000,
                startSourceSample = 0L,
                endSourceSampleExclusive = samples,
            ),
            ordinal = 0,
            powerampFileId = POWERAMP_FILE_ID,
            providerSnapshotGeneration = "provider-generation",
            providerRow = V2ProviderPathRowEvidence(
                powerampFileId = POWERAMP_FILE_ID,
                physicalPath = path,
                providerPhysicalPath = path,
                artist = "Artist",
                album = "Album",
                title = "Title",
                offsetMs = 0L,
                durationMs = 2_000L,
                cueSourceImageFolderId = null,
            ),
            displayMetadata = DisplayTrackMetadata("Artist", "Album", "Title"),
            normalizedMetadata = NormalizedTrackMetadata(
                normalizationSpecId = "track-normalization-v2",
                artist = "artist",
                album = "album",
                title = "title",
                metadataKey = "artist|album|title|2000",
            ),
            physicalPath = path,
            canonicalPath = path,
            sourceFingerprint = fingerprint,
            finalizedAudioSpan = FinalizedAudioSpanEvidence(
                kind = V2ResolvedAudioSpanKind.WHOLE_FILE,
                authority = V2AudioSpanAuthority.DECODED_END_OF_STREAM,
                executionBoundaryRequirement =
                    V2ExecutionBoundaryRequirement.VERIFY_END_OF_STREAM_AND_RECONCILE,
                providerSpan = V2ProviderSpanEvidence(0L, 2_000_000L, 2_000_000L),
                cueClassification = V2CueClassificationEvidence(1, 1, emptyList(), emptyList()),
                container = V2AudioContainerEvidence(
                    physicalPath = path,
                    audioTrackIndex = 0,
                    durationUsEstimate = 2_000_000L,
                    sampleRateHz = 24_000,
                    channelCount = 2,
                    mime = "audio/flac",
                ),
                startUs = 0L,
                endExclusiveUs = 2_000_000L,
                startSourceSample = 0L,
                endSourceSampleExclusive = samples,
                sourceSampleCount = samples,
                exactSampleCount24k = samples,
                expectedWork = V2AudioSpanMath.expectedWorkFor24kSamples(samples),
            ),
        )
    }

    private companion object {
        const val JOB_ID = "pcm-cache-test"
        const val POWERAMP_FILE_ID = 99L
        const val SHA = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    }
}
