package com.powerampstartradio.indexing.v2

import org.junit.Assert.assertFalse
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class V2SelectedProviderSnapshotPolicyTest {
    @Test
    fun `ordinary selected rows remain a bounded provider read`() {
        assertFalse(
            V2SelectedProviderSnapshotPolicy.requiresCompleteLibrarySnapshot(
                pathGroupSnapshot = snapshot(
                    row(id = 1L, offsetMs = 0L, offsetWasNull = false, cueFolderId = null),
                ),
                selectedIds = setOf(1L),
            ),
        )
    }

    @Test
    fun `complete selected group detects a zero-offset first CUE row`() {
        val cueGroup = snapshot(
            row(id = 1L, offsetMs = 0L, offsetWasNull = false, cueFolderId = null),
            row(id = 2L, offsetMs = 120_000L, offsetWasNull = false, cueFolderId = null),
        )

        assertTrue(
            V2SelectedProviderSnapshotPolicy.requiresCompleteLibrarySnapshot(
                pathGroupSnapshot = cueGroup,
                selectedIds = setOf(1L),
            ),
        )
    }

    @Test
    fun `raw CUE source row requires complete library authorization`() {
        assertTrue(
            V2SelectedProviderSnapshotPolicy.requiresCompleteLibrarySnapshot(
                pathGroupSnapshot = snapshot(
                    row(id = 1L, offsetMs = 0L, offsetWasNull = false, cueFolderId = null),
                    row(id = 2L, offsetMs = 0L, offsetWasNull = false, cueFolderId = 42L),
                ),
                selectedIds = setOf(1L),
            ),
        )
    }

    @Test
    fun `missing selected row fails over to the complete library`() {
        assertTrue(
            V2SelectedProviderSnapshotPolicy.requiresCompleteLibrarySnapshot(
                pathGroupSnapshot = snapshot(
                    row(id = 2L, offsetMs = 0L, offsetWasNull = false, cueFolderId = null),
                ),
                selectedIds = setOf(1L),
            ),
        )
    }

    @Test
    fun `CUE rows sharing a filename elsewhere do not widen an ordinary selection`() {
        val ordinary = group(
            "/music/ordinary/track.flac",
            row(
                id = 1L,
                path = "/music/ordinary/track.flac",
                offsetMs = 0L,
                offsetWasNull = false,
                cueFolderId = null,
            ),
        )
        val unrelatedCue = group(
            "/music/cue/track.flac",
            row(
                id = 2L,
                path = "/music/cue/track.flac",
                offsetMs = 0L,
                offsetWasNull = false,
                cueFolderId = null,
            ),
            row(
                id = 3L,
                path = "/music/cue/track.flac",
                offsetMs = 120_000L,
                offsetWasNull = false,
                cueFolderId = null,
            ),
        )

        assertFalse(
            V2SelectedProviderSnapshotPolicy.requiresCompleteLibrarySnapshot(
                pathGroupSnapshot = V2ProviderPathGroupSnapshot(
                    libraryGeneration = "test-generation",
                    groups = listOf(ordinary, unrelatedCue),
                ),
                selectedIds = setOf(1L),
            ),
        )
    }

    @Test
    fun `bounded ordinary path group has the same selected source evidence as a full snapshot`() {
        val selectedGroup = group(
            "/music/selected/track.flac",
            row(
                id = 1L,
                path = "/music/selected/track.flac",
                offsetMs = 0L,
                offsetWasNull = false,
                cueFolderId = null,
            ),
        )
        val unrelatedGroup = group(
            "/music/unrelated/other.flac",
            row(
                id = 2L,
                path = "/music/unrelated/other.flac",
                offsetMs = 0L,
                offsetWasNull = false,
                cueFolderId = null,
            ),
        )
        val fullSnapshot = V2ProviderPathGroupSnapshot(
            libraryGeneration = "full-generation",
            groups = listOf(selectedGroup, unrelatedGroup),
        )
        val boundedSnapshot = V2ProviderPathGroupSnapshot(
            libraryGeneration = "bounded-generation",
            groups = listOf(selectedGroup),
        )

        val fullSelectedGroup = fullSnapshot.groups.single { group ->
            group.rows.any { it.powerampFileId == 1L }
        }
        val boundedSelectedGroup = boundedSnapshot.groups.single { group ->
            group.rows.any { it.powerampFileId == 1L }
        }
        assertEquals(fullSelectedGroup.rows.single(), boundedSelectedGroup.rows.single())
        assertEquals(
            V2CueClassificationEvidenceFactory.from(fullSelectedGroup),
            V2CueClassificationEvidenceFactory.from(boundedSelectedGroup),
        )
    }

    private fun row(
        id: Long,
        path: String = "/music/track.flac",
        offsetMs: Long,
        offsetWasNull: Boolean,
        cueFolderId: Long?,
    ) = V2ProviderPathRowEvidence(
        powerampFileId = id,
        physicalPath = path,
        artist = "artist",
        album = "album",
        title = "title",
        offsetMs = offsetMs,
        offsetWasNull = offsetWasNull,
        durationMs = 180_000L,
        cueSourceImageFolderId = cueFolderId,
    )

    private fun snapshot(
        vararg rows: V2ProviderPathRowEvidence,
    ) = V2ProviderPathGroupSnapshot(
        libraryGeneration = "test-generation",
        groups = listOf(group("/music/track.flac", *rows)),
    )

    private fun group(
        path: String,
        vararg rows: V2ProviderPathRowEvidence,
    ) = V2ProviderPathGroupEvidence(
        physicalPath = path,
        rows = rows.toList(),
        completeness = V2ProviderPathGroupCompleteness.COMPLETE,
    )
}
