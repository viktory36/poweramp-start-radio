package com.powerampstartradio.indexing.v2

import com.powerampstartradio.indexing.NewTrackDetector
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class V2AudioSpanResolverTest {
    @Test
    fun `sample boundaries round toward the first sample at or after time`() {
        assertEquals(0L, V2AudioSpanMath.sampleAtOrAfter(0L, 44_100))
        assertEquals(1L, V2AudioSpanMath.sampleAtOrAfter(1L, 44_100))
        assertEquals(44_100L, V2AudioSpanMath.sampleAtOrAfter(1_000_000L, 44_100))
        assertEquals(44_101L, V2AudioSpanMath.sampleAtOrAfter(1_000_001L, 44_100))

        assertEquals(24_000L, V2AudioSpanMath.resampledLength(44_100L, 44_100, 24_000))
        assertEquals(24_001L, V2AudioSpanMath.resampledLength(44_101L, 44_100, 24_000))
    }

    @Test
    fun `work counts come from exact 24k samples including tail and bookends`() {
        val cases = linkedMapOf(
            23_999L to ExpectedTrackWork(0, 0),
            24_000L to ExpectedTrackWork(1, 1),
            119_999L to ExpectedTrackWork(1, 1),
            120_000L to ExpectedTrackWork(1, 1),
            143_999L to ExpectedTrackWork(1, 1),
            144_000L to ExpectedTrackWork(2, 1),
            15_120_000L to ExpectedTrackWork(126, 1),
            15_240_000L to ExpectedTrackWork(127, 2),
        )
        cases.forEach { (samples, expected) ->
            assertEquals(
                "samples=$samples",
                expected,
                V2AudioSpanMath.expectedWorkFor24kSamples(samples),
            )
        }
    }

    @Test
    fun `Kitson ordinary file ignores wildly wrong Poweramp duration`() {
        val path = "/music/Daniel Kitson On Triple R - Show 2 - Part 1.mp3"
        val selected = track(
            id = 75_815L,
            path = path,
            durationMs = 20_414_388,
        )
        val inspector = RecordingInspector(
            evidence(path, durationUs = 3_827_697_625L, sampleRateHz = 44_100),
        )

        val result = V2AudioSpanResolver(inspector).resolve(
            selectedTracks = listOf(selected),
            providerSnapshot = snapshot(group(path, row(selected))),
        ).resolved.single()

        assertEquals(V2ResolvedAudioSpanKind.WHOLE_FILE, result.kind)
        assertEquals(V2AudioSpanAuthority.PROVISIONAL_END_OF_STREAM, result.authority)
        assertEquals(20_414_388_000L, result.providerEvidence.durationUs)
        assertEquals(3_827_697_625L, result.plannedDurationUs)
        assertEquals(ExpectedTrackWork(mertWindows = 766, clampSegments = 6), result.expectedWork)
        assertTrue(result.mustVerifyEndOfStream)
        assertEquals(1, inspector.inspectedPaths.size)
    }

    @Test
    fun `ordinary file remains resolvable when Poweramp duration is zero`() {
        val path = "/music/provider-duration-missing.opus"
        val selected = track(id = 75_817L, path = path, durationMs = 0)

        val result = V2AudioSpanResolver(
            RecordingInspector(evidence(path, durationUs = 12_500_000L, sampleRateHz = 48_000)),
        ).resolve(
            selectedTracks = listOf(selected),
            providerSnapshot = snapshot(group(path, row(selected))),
        ).resolved.single()

        assertEquals(0L, result.providerEvidence.durationUs)
        assertEquals(12_500_000L, result.plannedDurationUs)
        assertEquals(ExpectedTrackWork(3, 1), result.expectedWork)
    }

    @Test
    fun `first zero-offset CUE row is classified from its complete path group`() {
        val path = "/music/dj-shadow-album-image.flac"
        val first = track(id = 84_702L, path = path, durationMs = 190_000, offsetMs = 0L)
        val second = track(id = 84_703L, path = path, durationMs = 68_000, offsetMs = 190_000L)
        val third = track(id = 84_704L, path = path, durationMs = 82_000, offsetMs = 258_000L)

        val result = V2AudioSpanResolver(
            RecordingInspector(evidence(path, durationUs = 4_616_933_333L)),
        ).resolve(
            selectedTracks = listOf(first),
            providerSnapshot = snapshot(
                group(path, row(first), row(second), row(third)),
            ),
        ).resolved.single()

        assertEquals(V2ResolvedAudioSpanKind.LOGICAL_CUE, result.kind)
        assertEquals(V2AudioSpanAuthority.PROVIDER_CUE_HALF_OPEN_SPAN, result.authority)
        assertEquals(0L, result.startUs)
        assertEquals(190_000_000L, result.endExclusiveUs)
        assertEquals(listOf(84_703L, 84_704L), result.cueClassificationEvidence.nonZeroOffsetRowIds)
        assertFalse(result.mustVerifyEndOfStream)
    }

    @Test
    fun `duplicate all-zero ordinary rows are not guessed to be CUE`() {
        val path = "/music/ordinary.flac"
        val selected = track(id = 1L, path = path, durationMs = 180_000)
        val duplicate = track(id = 2L, path = path, durationMs = 180_000)

        val result = V2AudioSpanResolver(
            RecordingInspector(evidence(path, durationUs = 180_123_456L)),
        ).resolve(
            listOf(selected),
            snapshot(group(path, row(selected), row(duplicate))),
        ).resolved.single()

        assertEquals(V2ResolvedAudioSpanKind.WHOLE_FILE, result.kind)
        assertTrue(result.cueClassificationEvidence.nonZeroOffsetRowIds.isEmpty())
        assertTrue(result.cueClassificationEvidence.rawSourceImageRowIds.isEmpty())
    }

    @Test
    fun `raw CUE source images are excluded without container inspection`() {
        val path = "/music/cue-source.flac"
        val source = track(
            id = 10L,
            path = path,
            durationMs = 600_000,
            cueFolderId = 99L,
        )
        val inspector = RecordingInspector(evidence(path, durationUs = 600_000_000L))

        val batch = V2AudioSpanResolver(inspector).resolve(
            listOf(source),
            snapshot(group(path, row(source))),
        )

        assertTrue(batch.resolved.isEmpty())
        assertEquals(V2AudioSpanExclusionReason.RAW_CUE_SOURCE_IMAGE, batch.excluded.single().reason)
        assertTrue(inspector.inspectedPaths.isEmpty())
    }

    @Test
    fun `CUE provider half-open span remains authoritative past a short container estimate`() {
        val path = "/music/short-image.flac"
        val first = track(id = 20L, path = path, durationMs = 600_001)
        val sibling = track(id = 21L, path = path, durationMs = 1_000, offsetMs = 600_001L)

        val result = V2AudioSpanResolver(
            RecordingInspector(evidence(path, durationUs = 600_000_000L)),
        ).resolve(
            listOf(first),
            snapshot(group(path, row(first), row(sibling))),
        ).resolved.single()

        assertEquals(V2AudioSpanAuthority.PROVIDER_CUE_HALF_OPEN_SPAN, result.authority)
        assertEquals(600_001_000L, result.endExclusiveUs)
        assertEquals(600_000_000L, result.containerEvidence.durationUsEstimate)
    }

    @Test
    fun `CUE provider half-open span remains authoritative below a long container estimate`() {
        val path = "/music/long-image.flac"
        val first = track(id = 22L, path = path, durationMs = 599_999)
        val sibling = track(id = 23L, path = path, durationMs = 1_000, offsetMs = 599_999L)

        val result = V2AudioSpanResolver(
            RecordingInspector(evidence(path, durationUs = 600_000_000L)),
        ).resolve(
            listOf(first),
            snapshot(group(path, row(first), row(sibling))),
        ).resolved.single()

        assertEquals(V2AudioSpanAuthority.PROVIDER_CUE_HALF_OPEN_SPAN, result.authority)
        assertEquals(599_999_000L, result.endExclusiveUs)
        assertEquals(600_000_000L, result.containerEvidence.durationUsEstimate)
    }

    @Test
    fun `missing whole-file container duration records provider fallback for scheduling`() {
        val path = "/music/missing-container-duration.opus"
        val selected = track(id = 24L, path = path, durationMs = 12_345)
        val unavailable = evidence(path, durationUs = 0L).copy(
            durationEstimateSource = V2DurationEstimateSource.UNAVAILABLE,
        )

        val result = V2AudioSpanResolver(RecordingInspector(unavailable)).resolve(
            listOf(selected),
            snapshot(group(path, row(selected))),
        ).resolved.single()

        assertEquals(V2DurationEstimateSource.PROVIDER_SPAN_FALLBACK,
            result.containerEvidence.durationEstimateSource)
        assertEquals(12_345_000L, result.endExclusiveUs)
        assertEquals(V2AudioSpanAuthority.PROVISIONAL_END_OF_STREAM, result.authority)
    }

    @Test
    fun `fully unknown ordinary duration remains unresolved until physical EOS`() {
        val path = "/music/unknown-duration.opus"
        val selected = track(id = 27L, path = path, durationMs = 0)
        val unavailable = evidence(path, durationUs = 0L).copy(
            durationEstimateSource = V2DurationEstimateSource.UNAVAILABLE,
        )

        val batch = V2AudioSpanResolver(RecordingInspector(unavailable)).resolve(
            listOf(selected),
            snapshot(group(path, row(selected))),
        )
        val result = batch.resolved.single()

        assertTrue(batch.rejected.isEmpty())
        assertTrue(V2UnknownDurationOrdinarySpanPolicy.isUnresolved(result))
        assertEquals(V2DurationEstimateSource.UNAVAILABLE,
            result.containerEvidence.durationEstimateSource)
        assertEquals(0L, result.endExclusiveUs)
        assertEquals(0L, result.exactSampleCount24k)
        assertEquals(ExpectedTrackWork(0, 0), result.expectedWork)
    }

    @Test
    fun `unknown duration CUE row remains a row-local rejection`() {
        val path = "/music/unknown-cue-duration.flac"
        val selected = track(id = 28L, path = path, durationMs = 0)
        val sibling = track(id = 29L, path = path, durationMs = 10_000, offsetMs = 1_000L)
        val unavailable = evidence(path, durationUs = 0L).copy(
            durationEstimateSource = V2DurationEstimateSource.UNAVAILABLE,
        )

        val batch = V2AudioSpanResolver(RecordingInspector(unavailable)).resolve(
            listOf(selected),
            snapshot(group(path, row(selected), row(sibling))),
        )

        assertTrue(batch.resolved.isEmpty())
        assertEquals(
            V2AudioSpanResolutionFailureCode.INVALID_PROVIDER_SPAN,
            batch.rejected.single().code,
        )
    }

    @Test
    fun `missing CUE container duration records provider half-open fallback`() {
        val path = "/music/missing-cue-container-duration.flac"
        val first = track(id = 25L, path = path, durationMs = 20_000)
        val sibling = track(id = 26L, path = path, durationMs = 10_000, offsetMs = 20_000L)
        val unavailable = evidence(path, durationUs = 0L).copy(
            durationEstimateSource = V2DurationEstimateSource.UNAVAILABLE,
        )

        val result = V2AudioSpanResolver(RecordingInspector(unavailable)).resolve(
            listOf(first),
            snapshot(group(path, row(first), row(sibling))),
        ).resolved.single()

        assertEquals(V2DurationEstimateSource.PROVIDER_SPAN_FALLBACK,
            result.containerEvidence.durationEstimateSource)
        assertEquals(20_000_000L, result.endExclusiveUs)
        assertEquals(V2AudioSpanAuthority.PROVIDER_CUE_HALF_OPEN_SPAN, result.authority)
    }

    @Test
    fun `partial provider path groups are rejected before classification`() {
        val path = "/music/unknown-shape.flac"
        val selected = track(id = 30L, path = path, durationMs = 100_000)
        val partial = V2ProviderPathGroupEvidence(
            physicalPath = path,
            rows = listOf(row(selected)),
            completeness = V2ProviderPathGroupCompleteness.PARTIAL,
        )

        val error = assertThrows(V2AudioSpanResolutionException::class.java) {
            V2AudioSpanResolver(
                RecordingInspector(evidence(path, durationUs = 100_000_000L)),
            ).resolve(listOf(selected), snapshot(partial))
        }
        assertEquals(V2AudioSpanResolutionFailureCode.INCOMPLETE_PATH_GROUP, error.code)
    }

    @Test
    fun `container inspection uses raw provider path rather than normalized identity`() {
        val normalizedPath = "/music/Café/Eté.flac"
        val rawPath = "/music/Cafe\u0301/Ete\u0301.flac"
        val selected = track(id = 31L, path = normalizedPath, durationMs = 100_000)
        val providerRow = row(selected).copy(
            physicalPath = normalizedPath,
            providerPhysicalPath = rawPath,
        )
        val inspector = RecordingInspector(evidence(rawPath, durationUs = 100_000_000L))

        val resolved = V2AudioSpanResolver(inspector).resolve(
            listOf(selected),
            snapshot(group(normalizedPath, providerRow)),
        ).resolved.single()

        assertEquals(listOf(rawPath), inspector.inspectedPaths)
        assertEquals(rawPath, resolved.containerEvidence.physicalPath)
    }

    @Test
    fun `one bad physical source rejects every selected occurrence sharing it and keeps good rows`() {
        val goodPath = "/music/good.flac"
        val badPath = "/music/not-audio.bin"
        val good = track(id = 40L, path = goodPath, durationMs = 100_000)
        val badFirst = track(id = 41L, path = badPath, durationMs = 100_000)
        val badSecond = track(id = 42L, path = badPath, durationMs = 100_000)
        val inspector = V2AudioContainerInspector { path ->
            if (path == badPath) {
                throw V2AudioContainerInspectionException(
                    V2AudioContainerInspectionFailureCode.NO_AUDIO_STREAM,
                    "No audio stream in $path",
                )
            }
            evidence(path, durationUs = 100_000_000L)
        }

        val result = V2AudioSpanResolver(inspector).resolve(
            selectedTracks = listOf(good, badFirst, badSecond),
            providerSnapshot = snapshot(
                group(goodPath, row(good)),
                group(badPath, row(badFirst), row(badSecond)),
            ),
        )

        assertEquals(listOf(40L), result.resolved.map { it.selectedTrack.powerampFileId })
        assertEquals(listOf(41L, 42L), result.rejected.map {
            it.selectedTrack.powerampFileId
        })
        assertTrue(result.rejected.all {
            it.code == V2AudioSpanResolutionFailureCode.NO_AUDIO_STREAM
        })
    }

    @Test
    fun `invalid CUE row is a local outcome while another selected row resolves`() {
        val cuePath = "/music/broken-cue.flac"
        val broken = track(id = 50L, path = cuePath, durationMs = -1)
        val cueSibling = track(id = 51L, path = cuePath, durationMs = 1_000, offsetMs = 1L)
        val goodPath = "/music/other.flac"
        val good = track(id = 52L, path = goodPath, durationMs = 100_000)

        val result = V2AudioSpanResolver { path ->
            evidence(path, durationUs = 100_000_000L)
        }.resolve(
            selectedTracks = listOf(broken, good),
            providerSnapshot = snapshot(
                group(cuePath, row(broken), row(cueSibling)),
                group(goodPath, row(good)),
            ),
        )

        assertEquals(listOf(52L), result.resolved.map { it.selectedTrack.powerampFileId })
        assertEquals(50L, result.rejected.single().selectedTrack.powerampFileId)
        assertEquals(
            V2AudioSpanResolutionFailureCode.INVALID_PROVIDER_SPAN,
            result.rejected.single().code,
        )
    }

    @Test
    fun `selection or provider ambiguity still aborts the whole batch`() {
        val path = "/music/changed.flac"
        val selected = track(id = 60L, path = path, durationMs = 100_000)
        val changedRow = row(selected).copy(durationMs = 99_000L)

        val error = assertThrows(V2AudioSpanResolutionException::class.java) {
            V2AudioSpanResolver(RecordingInspector(evidence(path, 100_000_000L))).resolve(
                selectedTracks = listOf(selected),
                providerSnapshot = snapshot(group(path, changedRow)),
            )
        }

        assertEquals(V2AudioSpanResolutionFailureCode.SELECTED_ROW_CHANGED, error.code)
        assertEquals(
            V2AudioSpanResolutionFailureScope.GLOBAL_SNAPSHOT_OR_SELECTION,
            V2AudioSpanResolutionFailurePolicy.scope(error.code),
        )
    }

    @Test
    fun `every audio resolution failure has one explicit scope`() {
        val scopes = V2AudioSpanResolutionFailureCode.entries.associateWith(
            V2AudioSpanResolutionFailurePolicy::scope,
        )

        assertEquals(V2AudioSpanResolutionFailureCode.entries.size, scopes.size)
        assertEquals(
            setOf(
                V2AudioSpanResolutionFailureCode.CONTAINER_INSPECTION_FAILED,
                V2AudioSpanResolutionFailureCode.INVALID_CONTAINER_EVIDENCE,
                V2AudioSpanResolutionFailureCode.INVALID_PROVIDER_SPAN,
                V2AudioSpanResolutionFailureCode.CUE_SPAN_OUT_OF_BOUNDS,
                V2AudioSpanResolutionFailureCode.SAMPLE_COORDINATE_OVERFLOW,
                V2AudioSpanResolutionFailureCode.SOURCE_UNREADABLE,
                V2AudioSpanResolutionFailureCode.NO_AUDIO_STREAM,
                V2AudioSpanResolutionFailureCode.UNSUPPORTED_OR_INVALID_CONTAINER,
            ),
            scopes.filterValues {
                it == V2AudioSpanResolutionFailureScope.SELECTED_SOURCE_OR_OCCURRENCE
            }.keys,
        )
    }

    private class RecordingInspector(
        private vararg val evidence: V2AudioContainerEvidence,
    ) : V2AudioContainerInspector {
        val inspectedPaths = mutableListOf<String>()

        override fun inspect(physicalPath: String): V2AudioContainerEvidence {
            inspectedPaths += physicalPath
            return evidence.single { it.physicalPath == physicalPath }
        }
    }

    private fun snapshot(
        vararg groups: V2ProviderPathGroupEvidence,
    ) = V2ProviderPathGroupSnapshot(
        libraryGeneration = "provider-snapshot-7",
        groups = groups.toList(),
        acquisitionEvidence = V2ProviderSnapshotAcquisitionEvidence(
            queryUri = "content://poweramp/files",
            requestedColumns = listOf("required"),
            returnedColumns = listOf("required"),
            rowCount = groups.sumOf { it.rows.size },
            cursorExhaustedNormally = true,
        ),
    )

    private fun group(
        path: String,
        vararg rows: V2ProviderPathRowEvidence,
    ) = V2ProviderPathGroupEvidence(
        physicalPath = path,
        rows = rows.toList(),
        completeness = V2ProviderPathGroupCompleteness.COMPLETE,
    )

    private fun row(track: NewTrackDetector.UnindexedTrack) = V2ProviderPathRowEvidence(
        powerampFileId = track.powerampFileId,
        physicalPath = requireNotNull(track.path),
        artist = track.artist,
        album = track.album,
        title = track.title,
        offsetMs = track.offsetMs,
        durationMs = track.durationMs.toLong(),
        cueSourceImageFolderId = track.cueFolderId,
    )

    private fun evidence(
        path: String,
        durationUs: Long,
        sampleRateHz: Int = 44_100,
    ) = V2AudioContainerEvidence(
        physicalPath = path,
        audioTrackIndex = 0,
        durationUsEstimate = durationUs,
        sampleRateHz = sampleRateHz,
        channelCount = 2,
        mime = "audio/flac",
    )

    private fun track(
        id: Long,
        path: String,
        durationMs: Int,
        offsetMs: Long = 0L,
        cueFolderId: Long? = null,
    ) = NewTrackDetector.UnindexedTrack(
        powerampFileId = id,
        artist = "artist",
        album = "album",
        title = "track-$id",
        durationMs = durationMs,
        path = path,
        offsetMs = offsetMs,
        cueFolderId = cueFolderId,
    )
}
