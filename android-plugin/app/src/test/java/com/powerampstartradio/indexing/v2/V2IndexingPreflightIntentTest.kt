package com.powerampstartradio.indexing.v2

import com.powerampstartradio.indexing.NewTrackDetector
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertThrows
import org.junit.Test

class V2IndexingPreflightIntentTest {
    @Test
    fun `request is durable before a base generation or expensive evidence exists`() {
        val intent = requested()

        assertEquals(2, intent.schemaVersion)
        assertEquals(V2IndexingPreflightIntentState.REQUESTED, intent.state)
        assertEquals(V2IndexingPreflightPhase.QUEUED, intent.progress.phase)
        assertNull(intent.baseGenerationId)
        assertEquals(listOf(7L, 11L), intent.selected.map { it.powerampFileId })
    }

    @Test
    fun `interrupted preflight resumes only against its originally bound generation`() {
        val planning = V2IndexingPreflightIntentStateMachine.beginOrResumePlanning(
            current = requested(),
            baseGenerationId = "generation-a",
            progress = progress(V2IndexingPreflightPhase.POWERAMP_SNAPSHOT),
            nowEpochMs = 101L,
        )
        val interrupted = V2IndexingPreflightIntentStateMachine.interrupt(
            planning,
            "Android media-processing quota paused this preflight",
            102L,
        )
        val resumed = V2IndexingPreflightIntentStateMachine.beginOrResumePlanning(
            interrupted,
            "generation-a",
            progress(V2IndexingPreflightPhase.POWERAMP_SNAPSHOT),
            103L,
        )

        assertEquals(V2IndexingPreflightIntentState.INTERRUPTED, interrupted.state)
        assertEquals(V2IndexingPreflightIntentState.PLANNING, resumed.state)
        assertEquals("generation-a", resumed.baseGenerationId)
        assertThrows(IllegalArgumentException::class.java) {
            V2IndexingPreflightIntentStateMachine.beginOrResumePlanning(
                interrupted,
                "generation-b",
                progress(V2IndexingPreflightPhase.POWERAMP_SNAPSHOT),
                103L,
            )
        }
    }

    @Test
    fun `cancel request wins over late planning progress`() {
        val planning = V2IndexingPreflightIntentStateMachine.beginOrResumePlanning(
            requested(),
            "generation-a",
            progress(V2IndexingPreflightPhase.SOURCE_FINGERPRINTS),
            101L,
        )
        val cancelling = V2IndexingPreflightIntentStateMachine.requestCancel(planning, 102L)
        val lateProgress = V2IndexingPreflightIntentStateMachine.updateProgress(
            cancelling,
            V2IndexingPreflightProgress(
                V2IndexingPreflightPhase.SOURCE_FINGERPRINTS,
                "Hashing selected source identities",
                1L,
                2L,
            ),
            103L,
        )
        val cancelled = V2IndexingPreflightIntentStateMachine.finishCancellation(
            lateProgress,
            104L,
        )

        assertEquals(cancelling, lateProgress)
        assertEquals(V2IndexingPreflightIntentState.CANCELLED, cancelled.state)
    }

    @Test
    fun `materialization records the immutable ledger spec`() {
        val planning = V2IndexingPreflightIntentStateMachine.beginOrResumePlanning(
            requested(),
            "generation-a",
            progress(V2IndexingPreflightPhase.PERSISTING_LEDGER),
            101L,
        )
        val rejected = rejection(planning.selected[1])
        val resolved = V2IndexingPreflightIntentStateMachine.resolveWithExecutableRows(
            planning,
            planned = listOf(planning.selected[0]),
            rejected = listOf(rejected),
            specId = "job-spec-sha256:abc",
            nowEpochMs = 102L,
        )
        val materialized = V2IndexingPreflightIntentStateMachine.materializeResolved(
            resolved,
            103L,
        )

        assertEquals(V2IndexingPreflightIntentState.RESOLVED_WITH_EXECUTABLE_ROWS, resolved.state)
        assertEquals(V2IndexingPreflightIntentState.MATERIALIZED, materialized.state)
        assertEquals("job-spec-sha256:abc", materialized.materializedSpecId)
        assertEquals(listOf(planning.selected[0]), materialized.planned)
        assertEquals(listOf(rejected), materialized.rejected)
    }

    @Test
    fun `timeout cannot erase a resolved partition while its ledger is being published`() {
        val planning = V2IndexingPreflightIntentStateMachine.beginOrResumePlanning(
            requested(),
            "generation-a",
            progress(V2IndexingPreflightPhase.PERSISTING_LEDGER),
            101L,
        )
        val rejected = rejection(planning.selected[1])
        val resolved = V2IndexingPreflightIntentStateMachine.resolveWithExecutableRows(
            planning,
            planned = listOf(planning.selected[0]),
            rejected = listOf(rejected),
            specId = "job-spec-sha256:abc",
            nowEpochMs = 102L,
        )

        val afterTimeout = V2IndexingPreflightIntentStateMachine.interrupt(
            resolved,
            "Android media-processing timeout",
            103L,
        )

        assertEquals(resolved, afterTimeout)
        assertEquals(listOf(planning.selected[0]), afterTimeout.planned)
        assertEquals(listOf(rejected), afterTimeout.rejected)
        assertEquals("job-spec-sha256:abc", afterTimeout.resolvedSpecId)
    }

    @Test
    fun `all rejected selection is terminal without pretending an empty ledger exists`() {
        val planning = V2IndexingPreflightIntentStateMachine.beginOrResumePlanning(
            requested(),
            "generation-a",
            progress(V2IndexingPreflightPhase.AUDIO_SPANS),
            101L,
        )
        val rejected = planning.selected.map(::rejection)

        val terminal = V2IndexingPreflightIntentStateMachine.resolveWithoutExecutableRows(
            planning,
            rejected,
            102L,
        )

        assertEquals(
            V2IndexingPreflightIntentState.RESOLVED_WITHOUT_EXECUTABLE_ROWS,
            terminal.state,
        )
        assertEquals(emptyList<V2IndexingPreflightSelection>(), terminal.planned)
        assertEquals(rejected, terminal.rejected)
        assertNull(terminal.resolvedSpecId)
        assertNull(terminal.materializedSpecId)
    }

    @Test
    fun `durable result must exactly partition immutable selection in selection order`() {
        val planning = V2IndexingPreflightIntentStateMachine.beginOrResumePlanning(
            requested(),
            "generation-a",
            progress(V2IndexingPreflightPhase.AUDIO_SPANS),
            101L,
        )

        assertThrows(InvalidV2IndexingPreflightIntentException::class.java) {
            V2IndexingPreflightIntentStateMachine.resolveWithExecutableRows(
                planning,
                planned = listOf(planning.selected[0]),
                rejected = emptyList(),
                specId = "job-spec-sha256:abc",
                nowEpochMs = 102L,
            )
        }
    }

    @Test
    fun `FAILED is reserved for typed global request failures`() {
        val planning = V2IndexingPreflightIntentStateMachine.beginOrResumePlanning(
            requested(),
            "generation-a",
            progress(V2IndexingPreflightPhase.POWERAMP_SNAPSHOT),
            101L,
        )

        assertThrows(IllegalArgumentException::class.java) {
            V2IndexingPreflightIntentStateMachine.fail(
                planning,
                V2IndexingPreflightFailureCode.AUDIO_TOO_SHORT,
                "one row is short",
                102L,
            )
        }
        val failed = V2IndexingPreflightIntentStateMachine.fail(
            planning,
            V2IndexingPreflightFailureCode.PROVIDER_SNAPSHOT_INVALID,
            "snapshot incomplete",
            102L,
        )
        assertEquals(V2IndexingPreflightIntentState.FAILED, failed.state)
        assertEquals(V2IndexingPreflightFailureCode.PROVIDER_SNAPSHOT_INVALID, failed.failureCode)
    }

    @Test
    fun `durable selection retains canonical unknown duration but rejects negative evidence`() {
        val unknown = V2IndexingPreflightSelectionFactory.fromTracks(
            listOf(
                NewTrackDetector.UnindexedTrack(
                    powerampFileId = 13L,
                    artist = "Artist",
                    album = "Album",
                    title = "Unknown",
                    durationMs = -1,
                    path = "/music/unknown.opus",
                ),
            ),
        ).single()
        val intent = V2IndexingPreflightIntentFactory.create(
            jobId = "unknown-duration-job",
            selected = listOf(unknown),
            rebuildDerivedIndexes = true,
            executionProfile = V2IndexingExecutionProfile.FULL,
            nowEpochMs = 100L,
        )

        assertEquals(0L, intent.selected.single().durationMs)
        assertThrows(InvalidV2IndexingPreflightIntentException::class.java) {
            V2IndexingPreflightIntentFactory.create(
                jobId = "negative-duration-job",
                selected = listOf(unknown.copy(durationMs = -1L)),
                rebuildDerivedIndexes = true,
                executionProfile = V2IndexingExecutionProfile.FULL,
                nowEpochMs = 100L,
            )
        }
    }

    private fun requested(): V2IndexingPreflightIntent =
        V2IndexingPreflightIntentFactory.create(
            jobId = "job-1",
            selected = listOf(selection(11L), selection(7L)),
            rebuildDerivedIndexes = true,
            executionProfile = V2IndexingExecutionProfile.BACKGROUND,
            nowEpochMs = 100L,
        )

    private fun selection(id: Long) = V2IndexingPreflightSelection(
        powerampFileId = id,
        providerPhysicalPath = "/music/$id.flac",
        durationMs = 180_000L,
        offsetMs = 0L,
        cueSourceImageFolderId = null,
    )

    private fun progress(phase: V2IndexingPreflightPhase) = V2IndexingPreflightProgress(
        phase,
        "Testing $phase",
    )

    private fun rejection(
        selected: V2IndexingPreflightSelection,
    ) = V2IndexingPreflightRejectedRow(
        selected = selected,
        code = V2IndexingPreflightFailureCode.AUDIO_TOO_SHORT,
        disposition = FailureDisposition.BLOCKED,
        retryTrigger = RetryTrigger.SOURCE_OR_LIBRARY_CHANGED,
        diagnostic = "Audio is too short",
    )
}
