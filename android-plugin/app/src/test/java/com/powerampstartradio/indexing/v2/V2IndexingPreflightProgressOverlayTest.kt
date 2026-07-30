package com.powerampstartradio.indexing.v2

import com.google.gson.GsonBuilder
import java.io.StringReader
import java.io.StringWriter
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Test

class V2IndexingPreflightProgressOverlayTest {
    private val gson = GsonBuilder().disableHtmlEscaping().create()

    @Test
    fun `current overlay projects progress without changing state-machine evidence`() {
        val planning = planningIntent(selectionCount = 2)
        val progress = progress(completed = 1L, total = 2L)
        val overlay = V2IndexingPreflightProgressOverlayPolicy.create(planning, progress, 103L)

        val projected = V2IndexingPreflightProgressOverlayPolicy.applyIfCurrent(planning, overlay)

        assertEquals(progress, projected.progress)
        assertEquals(planning.revision, projected.revision)
        assertEquals(planning.updatedAtEpochMs, projected.updatedAtEpochMs)
        assertEquals(planning.state, projected.state)
        assertEquals(planning.selected, projected.selected)
    }

    @Test
    fun `overlay left by a crash cannot override a later main-file transition`() {
        val planning = planningIntent(selectionCount = 2)
        val overlay = V2IndexingPreflightProgressOverlayPolicy.create(
            planning,
            progress(completed = 1L, total = 2L),
            103L,
        )
        val interrupted = V2IndexingPreflightIntentStateMachine.interrupt(
            planning,
            "Android stopped preflight",
            104L,
        )

        val afterRestart = V2IndexingPreflightProgressOverlayPolicy.applyIfCurrent(
            interrupted,
            overlay,
        )

        assertSame(interrupted, afterRestart)
        assertEquals(V2IndexingPreflightIntentState.INTERRUPTED, afterRestart.state)
        assertEquals(V2IndexingPreflightPhase.COMPLETE, afterRestart.progress.phase)
    }

    @Test
    fun `same revision overlay cannot cross immutable request identity`() {
        val planning = planningIntent(selectionCount = 2)
        val overlay = V2IndexingPreflightProgressOverlayPolicy.create(
            planning,
            progress(completed = 1L, total = 2L),
            103L,
        )
        val differentRequest = planning.copy(
            selected = planning.selected.mapIndexed { index, row ->
                if (index == 0) row.copy(providerPhysicalPath = "/different/${row.powerampFileId}")
                else row
            },
        )

        assertNotEquals(
            V2IndexingPreflightRequestFingerprint.compute(planning),
            V2IndexingPreflightRequestFingerprint.compute(differentRequest),
        )
        assertSame(
            differentRequest,
            V2IndexingPreflightProgressOverlayPolicy.applyIfCurrent(differentRequest, overlay),
        )
    }

    @Test
    fun `corrupt or unsupported overlay envelope is ignored`() {
        assertNull(
            V2IndexingPreflightProgressOverlayCodec.readOrNull(
                gson,
                StringReader("{not-json"),
            ),
        )
        assertNull(
            V2IndexingPreflightProgressOverlayCodec.readOrNull(
                gson,
                StringReader(
                    """{"format":"wrong","schemaVersion":1,"overlay":{}}""",
                ),
            ),
        )
    }

    @Test
    fun `periodic write payload is bounded independently of a 75000-row selection`() {
        val small = planningIntent(selectionCount = 1)
        val productionScale = planningIntent(selectionCount = 75_000)
        val progress = progress(completed = 37_500L, total = 75_000L)

        val smallBytes = encode(
            V2IndexingPreflightProgressOverlayPolicy.create(small, progress, 103L),
        ).toByteArray().size
        val productionBytes = encode(
            V2IndexingPreflightProgressOverlayPolicy.create(productionScale, progress, 103L),
        ).toByteArray().size
        val monolithicBytes = gson.toJson(productionScale).toByteArray().size

        assertEquals(smallBytes, productionBytes)
        assertTrue("overlay was $productionBytes bytes", productionBytes < 1_024)
        assertTrue(
            "main=$monolithicBytes overlay=$productionBytes",
            monolithicBytes > productionBytes * 5_000,
        )
        println(
            "preflight_progress_write_size selected=75000 " +
                "main_bytes=$monolithicBytes overlay_bytes=$productionBytes",
        )
    }

    private fun encode(overlay: V2IndexingPreflightProgressOverlay): String =
        StringWriter().also { writer ->
            V2IndexingPreflightProgressOverlayCodec.write(gson, overlay, writer)
        }.toString()

    private fun planningIntent(selectionCount: Int): V2IndexingPreflightIntent {
        val requested = V2IndexingPreflightIntentFactory.create(
            jobId = "job-progress-proof",
            selected = List(selectionCount) { index -> selection(index + 1L) },
            rebuildDerivedIndexes = true,
            executionProfile = V2IndexingExecutionProfile.BACKGROUND,
            nowEpochMs = 100L,
        )
        return V2IndexingPreflightIntentStateMachine.beginOrResumePlanning(
            requested,
            baseGenerationId = "generation-a",
            progress = progress(completed = 0L, total = selectionCount.toLong()),
            nowEpochMs = 102L,
        )
    }

    private fun selection(id: Long) = V2IndexingPreflightSelection(
        powerampFileId = id,
        providerPhysicalPath = "/music/library/album-$id/track-$id.flac",
        durationMs = 180_000L,
        offsetMs = 0L,
        cueSourceImageFolderId = null,
    )

    private fun progress(completed: Long, total: Long) = V2IndexingPreflightProgress(
        phase = V2IndexingPreflightPhase.SOURCE_FINGERPRINTS,
        message = "Hashing exact selected source identities",
        completedUnits = completed,
        totalUnits = total,
    )
}
