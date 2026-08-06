package com.powerampstartradio.indexing.v2

import com.powerampstartradio.indexing.NewTrackDetector
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertThrows
import org.junit.Test

class V2IndexingTrackPlanFactoryTest {
    @Test
    fun `logical CUE row preserves resolved half-open sample identity`() {
        val track = track(durationMs = 1_000, offsetMs = 123_456L)
        val row = providerRow(track)
        val resolved = resolvedSpan(track, row, V2ResolvedAudioSpanKind.LOGICAL_CUE)
        val input = V2IndexingTrackPlanFactory.create(resolved, row, fingerprint())

        assertEquals(55L, input.powerampFileId)
        assertEquals(123_456L, input.providerRow.offsetMs)
        assertEquals(1_000L, input.providerRow.durationMs)
        assertNull(input.providerRow.cueSourceImageFolderId)
        assertEquals(ExpectedTrackWork(1, 1), input.expectedWork)
        assertEquals(123_456_000L, input.finalizedAudioSpan.startUs)
        assertEquals(124_456_000L, input.finalizedAudioSpan.endExclusiveUs)
        assertEquals(44_100L, input.finalizedAudioSpan.sourceSampleCount)
        assertEquals(24_000L, input.finalizedAudioSpan.exactSampleCount24k)
        assertEquals(
            V2ExecutionBoundaryRequirement.ENFORCE_PROVIDER_HALF_OPEN_SPAN,
            input.finalizedAudioSpan.executionBoundaryRequirement,
        )
        assertEquals("/provider/path/album-image.flac", input.providerRow.physicalPath)
        assertEquals("/storage/music/album-image.flac", input.physicalPath)
        assertEquals("/storage/music/album-image.flac", input.finalizedAudioSpan.container.physicalPath)
        assertEquals(track.metadataKey, input.normalizedMetadata.metadataKey)
    }

    @Test
    fun `wrong provider duration stays diagnostic for ordinary acoustic work`() {
        val track = track(durationMs = 20_414_388, offsetMs = 0L)
        val row = providerRow(track)
        val resolved = resolvedSpan(
            track,
            row,
            V2ResolvedAudioSpanKind.WHOLE_FILE,
            containerDurationUs = 3_827_697_625L,
        )
        val input = V2IndexingTrackPlanFactory.create(resolved, row, fingerprint())

        assertEquals(20_414_388L, input.providerRow.durationMs)
        assertEquals(3_827_697_625L, input.finalizedAudioSpan.endExclusiveUs)
        assertEquals(ExpectedTrackWork(766, 6), input.expectedWork)
        assertEquals(
            V2ExecutionBoundaryRequirement.VERIFY_END_OF_STREAM_AND_RECONCILE,
            input.finalizedAudioSpan.executionBoundaryRequirement,
        )
    }

    @Test
    fun `unknown ordinary duration persists explicit unresolved work without a denominator`() {
        val track = track(durationMs = 0, offsetMs = 0L)
        val row = providerRow(track)
        val resolved = resolvedSpan(
            track = track,
            row = row,
            kind = V2ResolvedAudioSpanKind.WHOLE_FILE,
            containerDurationUs = 0L,
        )

        val input = V2IndexingTrackPlanFactory.create(resolved, row, fingerprint())

        assertEquals(V2DurationEstimateSource.UNAVAILABLE,
            input.finalizedAudioSpan.container.durationEstimateSource)
        assertEquals(0L, input.finalizedAudioSpan.endExclusiveUs)
        assertEquals(0L, input.finalizedAudioSpan.exactSampleCount24k)
        assertEquals(ExpectedTrackWork(0, 0), input.expectedWork)
    }

    @Test
    fun `raw CUE source image remains a typed preflight exclusion`() {
        val track = track(durationMs = 300_000, offsetMs = 0L, cueFolderId = 777L)
        val row = providerRow(track)
        val error = assertThrows(V2IndexingPreflightException::class.java) {
            V2IndexingTrackPlanFactory.create(
                resolvedSpan(track, row, V2ResolvedAudioSpanKind.WHOLE_FILE),
                row,
                fingerprint(),
            )
        }
        assertEquals(V2IndexingPreflightFailureCode.CUE_SOURCE_IMAGE, error.code)
        assertEquals(55L, error.powerampFileId)
    }

    @Test
    fun `sub-second resolved PCM and inconsistent row identity fail distinctly`() {
        val shortTrack = track(durationMs = 999, offsetMs = 0L)
        val shortRow = providerRow(shortTrack)
        val tooShort = assertThrows(V2IndexingPreflightException::class.java) {
            V2IndexingTrackPlanFactory.create(
                resolvedSpan(
                    shortTrack,
                    shortRow,
                    V2ResolvedAudioSpanKind.WHOLE_FILE,
                    containerDurationUs = 999_000L,
                ),
                shortRow,
                fingerprint(),
            )
        }
        assertEquals(V2IndexingPreflightFailureCode.AUDIO_TOO_SHORT, tooShort.code)

        val validTrack = track(durationMs = 1_000, offsetMs = 0L)
        val validRow = providerRow(validTrack)
        val invalid = assertThrows(V2IndexingPreflightException::class.java) {
            V2IndexingTrackPlanFactory.create(
                resolvedSpan(validTrack, validRow, V2ResolvedAudioSpanKind.WHOLE_FILE),
                validRow.copy(powerampFileId = 99L),
                fingerprint(),
            )
        }
        assertEquals(V2IndexingPreflightFailureCode.INVALID_LOGICAL_SPAN, invalid.code)
    }

    private fun resolvedSpan(
        track: NewTrackDetector.UnindexedTrack,
        row: V2ProviderPathRowEvidence,
        kind: V2ResolvedAudioSpanKind,
        containerDurationUs: Long = if (kind == V2ResolvedAudioSpanKind.LOGICAL_CUE) {
            300_000_000L
        } else {
            track.durationMs.toLong() * 1_000L
        },
        canonicalContainerPath: String = "/storage/music/album-image.flac",
    ): V2ResolvedAudioSpan {
        val providerSpan = V2ProviderSpanEvidence(
            offsetUs = row.offsetMs * 1_000L,
            durationUs = row.durationMs * 1_000L,
            endExclusiveUs = (row.offsetMs + row.durationMs) * 1_000L,
        )
        val startUs = if (kind == V2ResolvedAudioSpanKind.LOGICAL_CUE) {
            providerSpan.offsetUs
        } else {
            0L
        }
        val endUs = if (kind == V2ResolvedAudioSpanKind.LOGICAL_CUE) {
            providerSpan.endExclusiveUs
        } else {
            containerDurationUs
        }
        val startSample = V2AudioSpanMath.sampleAtOrAfter(startUs, 44_100)
        val endSample = V2AudioSpanMath.sampleAtOrAfter(endUs, 44_100)
        val sourceSamples = endSample - startSample
        val samples24k = V2AudioSpanMath.resampledLength(sourceSamples, 44_100, 24_000)
        return V2ResolvedAudioSpan(
            selectedTrack = track,
            libraryGeneration = GENERATION,
            kind = kind,
            authority = if (kind == V2ResolvedAudioSpanKind.LOGICAL_CUE) {
                V2AudioSpanAuthority.PROVIDER_CUE_HALF_OPEN_SPAN
            } else {
                V2AudioSpanAuthority.PROVISIONAL_END_OF_STREAM
            },
            executionBoundaryRequirement = if (kind == V2ResolvedAudioSpanKind.LOGICAL_CUE) {
                V2ExecutionBoundaryRequirement.ENFORCE_PROVIDER_HALF_OPEN_SPAN
            } else {
                V2ExecutionBoundaryRequirement.VERIFY_END_OF_STREAM_AND_RECONCILE
            },
            providerEvidence = providerSpan,
            cueClassificationEvidence = V2CueClassificationEvidence(
                providerGroupRowCount = 1,
                logicalRowCount = 1,
                nonZeroOffsetRowIds = if (kind == V2ResolvedAudioSpanKind.LOGICAL_CUE) {
                    listOf(track.powerampFileId)
                } else {
                    emptyList()
                },
                rawSourceImageRowIds = emptyList(),
            ),
            containerEvidence = V2AudioContainerEvidence(
                physicalPath = canonicalContainerPath,
                audioTrackIndex = 0,
                durationUsEstimate = containerDurationUs,
                durationEstimateSource = if (containerDurationUs > 0L) {
                    V2DurationEstimateSource.CONTAINER_METADATA
                } else {
                    V2DurationEstimateSource.UNAVAILABLE
                },
                sampleRateHz = 44_100,
                channelCount = 2,
                mime = "audio/flac",
            ),
            startUs = startUs,
            endExclusiveUs = endUs,
            startSourceSample = startSample,
            endSourceSampleExclusive = endSample,
            sourceSampleCount = sourceSamples,
            exactSampleCount24k = samples24k,
            expectedWork = V2AudioSpanMath.expectedWorkFor24kSamples(samples24k),
        )
    }

    private fun track(
        durationMs: Int,
        offsetMs: Long,
        cueFolderId: Long? = null,
    ) = NewTrackDetector.UnindexedTrack(
        powerampFileId = 55L,
        artist = "artist",
        album = "album",
        title = "title",
        durationMs = durationMs,
        path = "/provider/path/album-image.flac",
        offsetMs = offsetMs,
        cueFolderId = cueFolderId,
    )

    private fun providerRow(track: NewTrackDetector.UnindexedTrack) =
        V2ProviderPathRowEvidence(
            powerampFileId = track.powerampFileId,
            physicalPath = requireNotNull(track.path),
            providerPhysicalPath = requireNotNull(track.path),
            artist = track.artist,
            album = track.album,
            title = track.title,
            offsetMs = track.offsetMs,
            durationMs = track.durationMs.toLong(),
            cueSourceImageFolderId = track.cueFolderId,
        )

    private fun fingerprint() = SourceFingerprint(
        fingerprintSpecId = V2FixedRegionSampling.SPEC_ID,
        sizeBytes = 100_000L,
        lastModifiedEpochMs = 123L,
        fileKey = null,
        sampledContentSha256 = "a".repeat(64),
        fullContentSha256 = null,
    )

    companion object {
        private const val GENERATION =
            "poweramp-provider-snapshot-v3-sha256:" +
                "7777777777777777777777777777777777777777777777777777777777777777"
    }
}
