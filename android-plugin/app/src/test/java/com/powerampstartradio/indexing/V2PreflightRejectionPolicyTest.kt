package com.powerampstartradio.indexing

import com.powerampstartradio.indexing.v2.FailureDisposition
import com.powerampstartradio.indexing.v2.RetryTrigger
import com.powerampstartradio.indexing.v2.V2CommittedProviderSpan
import com.powerampstartradio.indexing.v2.V2IndexingExecutionProfile
import com.powerampstartradio.indexing.v2.V2IndexingPreflightFailureCode
import com.powerampstartradio.indexing.v2.V2IndexingPreflightIntent
import com.powerampstartradio.indexing.v2.V2IndexingPreflightIntentFactory
import com.powerampstartradio.indexing.v2.V2IndexingPreflightIntentStateMachine
import com.powerampstartradio.indexing.v2.V2IndexingPreflightPhase
import com.powerampstartradio.indexing.v2.V2IndexingPreflightProgress
import com.powerampstartradio.indexing.v2.V2IndexingPreflightRejectedRow
import com.powerampstartradio.indexing.v2.V2IndexingPreflightSelection
import com.powerampstartradio.indexing.v2.V2PowerampProviderSnapshotAssembler
import com.powerampstartradio.indexing.v2.V2ProviderSnapshotAcquisitionEvidence
import com.powerampstartradio.indexing.v2.V2ProviderSpanReceipt
import com.powerampstartradio.indexing.v2.V2RawPowerampProviderRow
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class V2PreflightRejectionPolicyTest {
    @Test
    fun `only retained terminal result states publish rejection attention`() {
        val row = selection(1L, "/music/a.flac", 0L, 10_000L)
        val planning = planning("planning", 10L, listOf(row))
        val transient = V2IndexingPreflightIntentStateMachine.resolveWithExecutableRows(
            current = planning,
            planned = listOf(row),
            rejected = emptyList(),
            specId = "spec-transient",
            nowEpochMs = 12L,
        )
        val terminal = allRejected("rejected", 20L, row)

        assertEquals(
            listOf("rejected"),
            V2PreflightRejectionPolicy.retainedRejections(listOf(planning, transient, terminal))
                .map(V2PreflightRejectedSpan::jobId),
        )
    }

    @Test
    fun `new materialized plan supersedes old rejection and later rejection restores attention`() {
        val row = selection(1L, "/music/a.flac", 0L, 10_000L)
        val firstRejection = allRejected("first", 10L, row)
        val materialized = materialized("second", 20L, planned = listOf(row))
        val laterRejection = allRejected("third", 30L, row)

        assertTrue(
            V2PreflightRejectionPolicy.retainedRejections(
                listOf(firstRejection, materialized),
            ).isEmpty(),
        )
        assertEquals(
            "third",
            V2PreflightRejectionPolicy.retainedRejections(
                listOf(laterRejection, firstRejection, materialized),
            ).single().jobId,
        )
    }

    @Test
    fun `logical versions in one file remain distinct provider spans`() {
        val first = selection(1L, "/music/album.flac", 0L, 10_000L)
        val second = selection(2L, "/music/album.flac", 10_000L, 20_000L)
        val firstRejected = allRejected("first", 10L, first)
        val secondRejected = allRejected("second", 20L, second)
        val firstPlanned = materialized("third", 30L, planned = listOf(first))

        val retained = V2PreflightRejectionPolicy.retainedRejections(
            listOf(firstRejected, secondRejected, firstPlanned),
        )
        assertEquals(1, retained.size)
        assertEquals(10_000L, retained.single().providerSpan.offsetMs)
        assertEquals(20_000L, retained.single().providerSpan.durationMs)
    }

    @Test
    fun `mixed materialized job publishes only its rejected partition`() {
        val executable = selection(1L, "/music/good.flac", 0L, 10_000L)
        val rejected = selection(2L, "/music/bad.flac", 0L, 20_000L)
        val mixed = materialized(
            jobId = "mixed",
            createdAt = 10L,
            planned = listOf(executable),
            rejected = listOf(rejection(rejected)),
        )

        val retained = V2PreflightRejectionPolicy.retainedRejections(listOf(mixed))
        assertEquals(listOf(2L), retained.map { it.originalPowerampFileId })
        assertEquals("mixed", retained.single().jobId)
    }

    @Test
    fun `current library metadata joins by span instead of stale numeric id`() {
        val rejected = V2PreflightRejectionPolicy.retainedRejections(
            listOf(allRejected("attempt", 10L, selection(7L, "/music/a.flac", 0L, 10_000L))),
        )
        val reusedOldId = track(7L, "/music/different.flac", 0L, 10_000)
        val remappedCurrent = track(91L, "/music/a.flac", 0L, 10_000, title = "Current title")

        val joined = V2PreflightRejectionPolicy.joinCurrentUnindexed(
            rejected,
            listOf(reusedOldId, remappedCurrent),
        ).single()
        assertEquals(91L, joined.currentTrack.powerampFileId)
        assertEquals("Current title", joined.currentTrack.title)
    }

    @Test
    fun `same path with changed duration is a new occurrence rather than retained attention`() {
        val rejected = V2PreflightRejectionPolicy.retainedRejections(
            listOf(allRejected("attempt", 10L, selection(7L, "/music/a.flac", 0L, 10_000L))),
        )

        assertTrue(
            V2PreflightRejectionPolicy.joinCurrentUnindexed(
                rejected,
                listOf(track(7L, "/music/a.flac", 0L, 10_001)),
            ).isEmpty(),
        )
    }

    @Test
    fun `unknown duration rejection is superseded only by the same zero occurrence`() {
        val unknown = selection(7L, "/music/unknown.opus", 0L, 0L)
        val laterKnown = unknown.copy(durationMs = 10_000L)
        val rejected = allRejected("rejected", 10L, unknown)
        val plannedKnown = materialized("known", 20L, planned = listOf(laterKnown))

        assertEquals(
            0L,
            V2PreflightRejectionPolicy.retainedRejections(listOf(rejected, plannedKnown))
                .single().providerSpan.durationMs,
        )
        assertTrue(
            V2PreflightRejectionPolicy.retainedRejections(
                listOf(rejected, plannedKnown, materialized("unknown", 30L, listOf(unknown))),
            ).isEmpty(),
        )
    }

    @Test
    fun `explicit retry suppresses attention and selection policy can make row ready`() {
        val current = track(91L, "/music/a.flac", 0L, 10_000)
        val rejected = V2PreflightRejectionPolicy.retainedRejections(
            listOf(allRejected("attempt", 10L, selection(7L, "/music/a.flac", 0L, 10_000L))),
        )
        val span = rejected.single().providerSpan

        assertTrue(
            V2PreflightRejectionPolicy.joinCurrentUnindexed(
                rejected,
                listOf(current),
                suppressedSpans = setOf(span),
            ).isEmpty(),
        )
        assertEquals(
            setOf(91L),
            V2IndexingSelectionPolicy.readyTrackIds(listOf(current), hiddenIds = emptySet()),
        )
    }

    @Test
    fun `preflight attention offers try again and never index but no skip action`() {
        val current = track(91L, "/music/a.flac", 0L, 10_000)
        val rejected = V2PreflightRejectionPolicy.retainedRejections(
            listOf(allRejected("attempt", 10L, selection(7L, "/music/a.flac", 0L, 10_000L))),
        )
        val attention = V2PreflightRejectionPolicy.joinCurrentUnindexed(
            rejected,
            listOf(current),
        ).single()

        val actions = V2PreflightRejectionPolicy.actionsFor(attention)
        assertEquals(
            setOf(
                V2PreflightAttentionAction.TRY_AGAIN,
                V2PreflightAttentionAction.NEVER_INDEX,
            ),
            actions,
        )
        assertFalse(actions.any { it.name.contains("SKIP") })
        assertEquals(null, preflightTryAgainUnavailableReason(attention))

    }

    @Test
    fun `ordinary unknown duration attention remains retryable and excludable`() {
        val unknown = selection(7L, "/music/unknown.opus", 0L, 0L)
        val rejected = V2PreflightRejectionPolicy.retainedRejections(
            listOf(allRejected("attempt", 10L, unknown)),
        )
        val current = track(91L, "/music/unknown.opus", 0L, 0).copy(
            detectionKind = V2UnindexedDetectionKind.SOURCE_ATTENTION,
        )

        val attention = V2PreflightRejectionPolicy.joinCurrentUnindexed(
            rejected,
            listOf(current),
        ).single()

        assertTrue(attention.canTryAgain)
        assertEquals(
            setOf(
                V2PreflightAttentionAction.TRY_AGAIN,
                V2PreflightAttentionAction.NEVER_INDEX,
            ),
            V2PreflightRejectionPolicy.actionsFor(attention),
        )
    }

    @Test
    fun `restart hydration uses complete current metadata and omits receipted spans`() {
        val first = allRejected(
            "attempt-1",
            10L,
            selection(1L, "/music/album.flac", 0L, 10_000L),
        )
        val second = allRejected(
            "attempt-2",
            20L,
            selection(2L, "/music/album.flac", 10_000L, 20_000L),
        )
        val retained = V2PreflightRejectionPolicy.retainedRejections(listOf(first, second))
        val rows = listOf(
            rawRow(101L, "First now", 0L, 10_000L),
            rawRow(202L, "Second now", 10_000L, 20_000L),
        )
        val snapshot = V2PowerampProviderSnapshotAssembler().assembleAfterSuccessfulExhaustion(
            rows = rows,
            acquisitionEvidence = V2ProviderSnapshotAcquisitionEvidence(
                queryUri = "content://poweramp/files",
                requestedColumns = emptyList(),
                returnedColumns = emptyList(),
                rowCount = rows.size,
                cursorExhaustedNormally = true,
            ),
        )
        val firstReceipt = V2ProviderSpanReceipt(
            trackId = 44L,
            providerSpan = V2CommittedProviderSpan("/music/album.flac", 0L, 10_000L),
        )

        val current = V2PreflightRejectionPolicy.currentUnindexedFromCompleteSnapshot(
            retained,
            snapshot,
            receipts = listOf(firstReceipt),
        )
        assertEquals(listOf(202L), current.map { it.powerampFileId })
        assertEquals("second now", current.single().title)
        assertEquals(10_000L, current.single().offsetMs)
    }

    @Test
    fun `restart hydration distinguishes ordinary unknown duration from CUE shaped rows`() {
        val ordinarySelection = selection(1L, "/music/unknown.opus", 0L, 0L)
        val cueSelection = selection(2L, "/music/album.flac", 0L, 0L)
        val retained = V2PreflightRejectionPolicy.retainedRejections(
            listOf(
                allRejected("ordinary", 10L, ordinarySelection),
                allRejected("cue", 20L, cueSelection),
            ),
        )
        val rows = listOf(
            rawRow(101L, "Unknown ordinary", 0L, -1L).copy(fileName = "unknown.opus"),
            rawRow(202L, "Logical CUE row", 0L, 0L),
            rawRow(203L, "Raw CUE image", 0L, 0L).copy(cueSourceImageFolderId = 77L),
        )
        val snapshot = V2PowerampProviderSnapshotAssembler().assembleAfterSuccessfulExhaustion(
            rows = rows,
            acquisitionEvidence = V2ProviderSnapshotAcquisitionEvidence(
                queryUri = "content://poweramp/files",
                requestedColumns = emptyList(),
                returnedColumns = emptyList(),
                rowCount = rows.size,
                cursorExhaustedNormally = true,
            ),
        )

        val current = V2PreflightRejectionPolicy.currentUnindexedFromCompleteSnapshot(
            retained,
            snapshot,
            receipts = emptyList(),
        )
        val ordinary = current.single { it.powerampFileId == 101L }
        val cue = current.single { it.powerampFileId == 202L }
        assertEquals(0, ordinary.durationMs)
        assertEquals(V2UnindexedDetectionKind.SOURCE_ATTENTION, ordinary.detectionKind)
        assertFalse(ordinary.sourceHasLogicalOffsets)
        assertFalse(ordinary.sourceHasCueImageRow)
        assertTrue(V2IndexingSelectionPolicy.isReadyTrack(ordinary))
        assertEquals(V2UnindexedDetectionKind.SOURCE_ATTENTION, cue.detectionKind)
        assertTrue(cue.sourceHasCueImageRow)
        assertFalse(V2IndexingSelectionPolicy.isReadyTrack(cue))
        assertFalse(current.any { it.powerampFileId == 203L })

        val afterOrdinaryReceipt = V2PreflightRejectionPolicy
            .currentUnindexedFromCompleteSnapshot(
                retained,
                snapshot,
                receipts = listOf(
                    V2ProviderSpanReceipt(
                        trackId = 44L,
                        providerSpan = V2CommittedProviderSpan(
                            "/music/unknown.opus",
                            0L,
                            0L,
                        ),
                    ),
                ),
            )
        assertEquals(listOf(202L), afterOrdinaryReceipt.map { it.powerampFileId })
    }

    private fun allRejected(
        jobId: String,
        createdAt: Long,
        row: V2IndexingPreflightSelection,
    ): V2IndexingPreflightIntent = V2IndexingPreflightIntentStateMachine
        .resolveWithoutExecutableRows(
            current = planning(jobId, createdAt, listOf(row)),
            rejected = listOf(rejection(row)),
            nowEpochMs = createdAt + 2L,
        )

    private fun materialized(
        jobId: String,
        createdAt: Long,
        planned: List<V2IndexingPreflightSelection>,
        rejected: List<V2IndexingPreflightRejectedRow> = emptyList(),
    ): V2IndexingPreflightIntent {
        val selected = planned + rejected.map(V2IndexingPreflightRejectedRow::selected)
        val resolved = V2IndexingPreflightIntentStateMachine.resolveWithExecutableRows(
            current = planning(jobId, createdAt, selected),
            planned = planned,
            rejected = rejected,
            specId = "spec-$jobId",
            nowEpochMs = createdAt + 2L,
        )
        return V2IndexingPreflightIntentStateMachine.materializeResolved(
            resolved,
            nowEpochMs = createdAt + 3L,
        )
    }

    private fun planning(
        jobId: String,
        createdAt: Long,
        selected: List<V2IndexingPreflightSelection>,
    ): V2IndexingPreflightIntent = V2IndexingPreflightIntentStateMachine.beginOrResumePlanning(
        current = V2IndexingPreflightIntentFactory.create(
            jobId = jobId,
            selected = selected,
            rebuildDerivedIndexes = true,
            executionProfile = V2IndexingExecutionProfile.FULL,
            nowEpochMs = createdAt,
        ),
        baseGenerationId = "generation",
        progress = V2IndexingPreflightProgress(
            V2IndexingPreflightPhase.AUDIO_SPANS,
            "Inspecting selected audio spans",
        ),
        nowEpochMs = createdAt + 1L,
    )

    private fun rejection(
        selected: V2IndexingPreflightSelection,
    ) = V2IndexingPreflightRejectedRow(
        selected = selected,
        code = V2IndexingPreflightFailureCode.UNSUPPORTED_OR_INVALID_AUDIO_CONTAINER,
        disposition = FailureDisposition.BLOCKED,
        retryTrigger = RetryTrigger.USER_REQUEST,
        diagnostic = "The selected file has no supported audio container.",
    )

    private fun selection(
        id: Long,
        path: String,
        offsetMs: Long,
        durationMs: Long,
    ) = V2IndexingPreflightSelection(
        powerampFileId = id,
        providerPhysicalPath = path,
        durationMs = durationMs,
        offsetMs = offsetMs,
        cueSourceImageFolderId = null,
    )

    private fun track(
        id: Long,
        path: String,
        offsetMs: Long,
        durationMs: Int,
        title: String = "Track",
    ) = NewTrackDetector.UnindexedTrack(
        powerampFileId = id,
        artist = "Artist",
        album = "Album",
        title = title,
        durationMs = durationMs,
        path = path,
        offsetMs = offsetMs,
    )

    private fun rawRow(
        id: Long,
        title: String,
        offsetMs: Long,
        durationMs: Long,
    ) = V2RawPowerampProviderRow(
        powerampFileId = id,
        artist = "Artist",
        album = "Album",
        title = title,
        durationMs = durationMs,
        folderPath = "/music",
        fileName = "album.flac",
        offsetMs = offsetMs,
        offsetWasNull = false,
        cueSourceImageFolderId = null,
    )
}
