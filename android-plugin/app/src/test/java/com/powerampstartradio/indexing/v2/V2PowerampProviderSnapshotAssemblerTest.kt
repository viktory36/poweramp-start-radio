package com.powerampstartradio.indexing.v2

import com.powerampstartradio.indexing.NewTrackDetector
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class V2PowerampProviderSnapshotAssemblerTest {
    private val assembler = V2PowerampProviderSnapshotAssembler()

    @Test
    fun `successful exhaustion groups lexical aliases and preserves provider evidence`() {
        val rows = cueRows()
        val snapshot = assembler.assembleAfterSuccessfulExhaustion(rows, evidence(rows.size))

        assertTrue(requireNotNull(snapshot.libraryGeneration).startsWith(
            "poweramp-provider-snapshot-v3-sha256:",
        ))
        assertEquals(3, snapshot.acquisitionEvidence?.rowCount)
        assertTrue(snapshot.acquisitionEvidence?.cursorExhaustedNormally == true)
        val group = snapshot.groups.single()
        assertEquals("/music/album-image.flac", group.physicalPath)
        assertEquals(V2ProviderPathGroupCompleteness.COMPLETE, group.completeness)
        assertEquals(listOf(100L, 101L, 102L), group.rows.map { it.powerampFileId })

        val first = group.rows.first()
        assertEquals("/music/set/../album-image.flac", first.providerPhysicalPath)
        assertEquals("DJ Shadow", first.artist)
        assertEquals("Live", first.album)
        assertEquals("Opening", first.title)
        assertEquals(190_000L, first.durationMs)
        assertEquals(0L, first.offsetMs)
        assertTrue(first.offsetWasNull)
        assertEquals(null, first.cueSourceImageFolderId)
        assertEquals(1L, first.createdAtEpochSecond)

        val second = group.rows[1]
        assertEquals(190_000L, second.offsetMs)
        assertEquals(null, second.cueSourceImageFolderId)
        val sourceImage = group.rows.last()
        assertEquals(0L, sourceImage.offsetMs)
        assertEquals(777L, sourceImage.cueSourceImageFolderId)
    }

    @Test
    fun `snapshot identity and group ordering do not depend on cursor row order`() {
        val rows = cueRows()
        val forward = assembler.assembleAfterSuccessfulExhaustion(rows, evidence(rows.size))
        val reverse = assembler.assembleAfterSuccessfulExhaustion(
            rows.reversed(),
            evidence(rows.size),
        )

        assertEquals(forward.libraryGeneration, reverse.libraryGeneration)
        assertEquals(forward.groups, reverse.groups)

        val metadataChanged = rows.toMutableList().also {
            it[0] = it[0].copy(title = "Different title")
        }
        val changed = assembler.assembleAfterSuccessfulExhaustion(
            metadataChanged,
            evidence(metadataChanged.size),
        )
        assertNotEquals(forward.libraryGeneration, changed.libraryGeneration)
    }

    @Test
    fun `snapshot preserves raw decomposed path while grouping by normalized identity`() {
        val decomposedAlbum = "Cafe\u0301"
        val row = cueRows().first().copy(
            folderPath = "/music/$decomposedAlbum",
            fileName = "Ete\u0301.flac",
        )

        val snapshot = assembler.assembleAfterSuccessfulExhaustion(listOf(row), evidence(1))
        val providerRow = snapshot.groups.single().rows.single()

        assertEquals("/music/Café/Eté.flac", snapshot.groups.single().physicalPath)
        assertEquals("/music/$decomposedAlbum/Ete\u0301.flac", providerRow.providerPhysicalPath)
    }

    @Test
    fun `same row count changes every provider identity signal`() {
        val rows = cueRows()
        val baseline = assembler.assembleAfterSuccessfulExhaustion(
            rows,
            evidence(rows.size),
        ).libraryGeneration
        val first = rows.first()
        val variants = listOf(
            first.copy(powerampFileId = 999L),
            first.copy(folderPath = "/music/other"),
            first.copy(fileName = "replacement.flac"),
            first.copy(durationMs = first.durationMs + 1L),
            first.copy(offsetMs = 1L),
            first.copy(offsetWasNull = false),
            first.copy(cueSourceImageFolderId = 888L),
            first.copy(createdAtEpochSecond = 2L),
        )

        for (variant in variants) {
            val changedRows = rows.toMutableList().also { it[0] = variant }
            val changed = assembler.assembleAfterSuccessfulExhaustion(
                changedRows,
                evidence(changedRows.size),
            )
            assertNotEquals(baseline, changed.libraryGeneration)
        }
    }

    @Test
    fun `missing first-seen evidence does not invalidate the complete library snapshot`() {
        val rows = listOf(cueRows().first().copy(createdAtEpochSecond = 0L))

        val snapshot = assembler.assembleAfterSuccessfulExhaustion(rows, evidence(rows.size))

        assertEquals(0L, snapshot.groups.single().rows.single().createdAtEpochSecond)
    }

    @Test
    fun `nonpositive provider duration sentinels canonicalize to one snapshot identity`() {
        val negativeRows = listOf(cueRows().first().copy(durationMs = -1L))
        val zeroRows = listOf(negativeRows.single().copy(durationMs = 0L))
        val laterKnownRows = listOf(negativeRows.single().copy(durationMs = 190_000L))

        val negative = assembler.assembleAfterSuccessfulExhaustion(
            negativeRows,
            evidence(negativeRows.size),
        )
        val zero = assembler.assembleAfterSuccessfulExhaustion(zeroRows, evidence(zeroRows.size))
        val laterKnown = assembler.assembleAfterSuccessfulExhaustion(
            laterKnownRows,
            evidence(laterKnownRows.size),
        )

        assertEquals(0L, negative.groups.single().rows.single().durationMs)
        assertEquals(negative.libraryGeneration, zero.libraryGeneration)
        assertNotEquals(negative.libraryGeneration, laterKnown.libraryGeneration)
    }

    @Test
    fun `snapshot cannot become complete before cursor exhaustion`() {
        val rows = cueRows()
        val error = assertThrows(V2PowerampProviderSnapshotException::class.java) {
            assembler.assembleAfterSuccessfulExhaustion(
                rows,
                evidence(rows.size).copy(cursorExhaustedNormally = false),
            )
        }
        assertEquals(V2PowerampProviderSnapshotFailureCode.CURSOR_NOT_EXHAUSTED, error.code)
    }

    @Test
    fun `empty provider result is a typed failure rather than empty success`() {
        val error = assertThrows(V2PowerampProviderSnapshotException::class.java) {
            assembler.assembleAfterSuccessfulExhaustion(emptyList(), evidence(0))
        }
        assertEquals(V2PowerampProviderSnapshotFailureCode.EMPTY_PROVIDER_RESULT, error.code)
    }

    @Test
    fun `cursor row-count disagreement is a typed failure`() {
        val rows = cueRows()
        val error = assertThrows(V2PowerampProviderSnapshotException::class.java) {
            assembler.assembleAfterSuccessfulExhaustion(rows, evidence(rows.size + 1))
        }
        assertEquals(
            V2PowerampProviderSnapshotFailureCode.CURSOR_EVIDENCE_MISMATCH,
            error.code,
        )
    }

    @Test
    fun `duplicate Poweramp IDs fail closed`() {
        val first = cueRows().first()
        val rows = listOf(first, first.copy(title = "Duplicate"))
        val error = assertThrows(V2PowerampProviderSnapshotException::class.java) {
            assembler.assembleAfterSuccessfulExhaustion(rows, evidence(rows.size))
        }
        assertEquals(
            V2PowerampProviderSnapshotFailureCode.DUPLICATE_POWERAMP_FILE_ID,
            error.code,
        )
        assertEquals(first.powerampFileId, error.powerampFileId)
    }

    @Test
    fun `normalization failure identifies the provider row`() {
        val row = cueRows().first()
        val rejecting = V2PowerampProviderSnapshotAssembler(
            V2ProviderLexicalPathNormalizer { throw IllegalStateException("invalid path") },
        )
        val error = assertThrows(V2PowerampProviderSnapshotException::class.java) {
            rejecting.assembleAfterSuccessfulExhaustion(listOf(row), evidence(1))
        }
        assertEquals(
            V2PowerampProviderSnapshotFailureCode.PATH_NORMALIZATION_FAILED,
            error.code,
        )
        assertEquals(row.powerampFileId, error.powerampFileId)
    }

    @Test
    fun `resolver binds selected raw path by ID and inspects the raw filesystem path`() {
        val rows = cueRows().map { it.copy(cueSourceImageFolderId = null) }
        val snapshot = assembler.assembleAfterSuccessfulExhaustion(rows, evidence(rows.size))
        val selected = NewTrackDetector.UnindexedTrack(
            powerampFileId = 100L,
            artist = "dj shadow",
            album = "live",
            title = "opening",
            durationMs = 190_000,
            path = "/music/set/../album-image.flac",
            offsetMs = 0L,
            cueFolderId = null,
        )
        val inspected = mutableListOf<String>()
        val result = V2AudioSpanResolver { path ->
            inspected += path
            V2AudioContainerEvidence(
                physicalPath = path,
                audioTrackIndex = 0,
                durationUsEstimate = 400_000_000L,
                sampleRateHz = 44_100,
                channelCount = 2,
                mime = "audio/flac",
            )
        }.resolve(listOf(selected), snapshot).resolved.single()

        assertEquals(V2ResolvedAudioSpanKind.LOGICAL_CUE, result.kind)
        assertEquals("/music/set/../album-image.flac", result.containerEvidence.physicalPath)
        assertEquals(listOf("/music/set/../album-image.flac"), inspected)
        assertEquals(listOf(101L), result.cueClassificationEvidence.nonZeroOffsetRowIds)
    }

    private fun cueRows() = listOf(
        V2RawPowerampProviderRow(
            powerampFileId = 100L,
            artist = "DJ Shadow",
            album = "Live",
            title = "Opening",
            durationMs = 190_000L,
            folderPath = "/music/set/..",
            fileName = "album-image.flac",
            offsetMs = 0L,
            offsetWasNull = true,
            cueSourceImageFolderId = null,
        ),
        V2RawPowerampProviderRow(
            powerampFileId = 101L,
            artist = "DJ Shadow",
            album = "Live",
            title = "Second",
            durationMs = 68_000L,
            folderPath = "/music",
            fileName = "album-image.flac",
            offsetMs = 190_000L,
            offsetWasNull = false,
            cueSourceImageFolderId = null,
        ),
        V2RawPowerampProviderRow(
            powerampFileId = 102L,
            artist = "DJ Shadow",
            album = "Live",
            title = "Album source image",
            durationMs = 400_000L,
            folderPath = "/music",
            fileName = "album-image.flac",
            offsetMs = 0L,
            offsetWasNull = false,
            cueSourceImageFolderId = 777L,
        ),
    )

    private fun evidence(rowCount: Int) = V2ProviderSnapshotAcquisitionEvidence(
        queryUri = "content://com.maxmpz.audioplayer.data/files",
        requestedColumns = V2PowerampProviderSnapshotAcquirer.REQUIRED_PROJECTION.toList(),
        returnedColumns = listOf(
            "_id",
            "artist",
            "album",
            "title_tag",
            "duration",
            "path",
            "name",
            "offset_ms",
            "cue_folder_id",
            "created_at",
        ),
        rowCount = rowCount,
        cursorExhaustedNormally = true,
    )
}
