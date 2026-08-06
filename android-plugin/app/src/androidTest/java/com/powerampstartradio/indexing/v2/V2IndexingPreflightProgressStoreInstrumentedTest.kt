package com.powerampstartradio.indexing.v2

import android.content.Context
import android.util.AtomicFile
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.google.gson.GsonBuilder
import java.io.File
import java.io.OutputStreamWriter
import java.nio.charset.StandardCharsets
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class V2IndexingPreflightProgressStoreInstrumentedTest {
    private val context = ApplicationProvider.getApplicationContext<Context>()
    private val gson = GsonBuilder().disableHtmlEscaping().create()

    @Test
    fun interruptedOverlayWriteRestoresLastCompleteProgress() = withStore { root, store ->
        val planning = createPlanning(store, "job-crash")
        val first = progress(1L)
        store.persistProgressOverlay(planning.jobId, first, 103L)

        val progressFile = File(root, "progress/${planning.jobId}.json")
        AtomicFile(progressFile).startWrite().let { unfinished ->
            unfinished.write("{truncated".toByteArray(StandardCharsets.UTF_8))
            unfinished.flush()
            unfinished.close()
        }

        assertEquals(first, store.require(planning.jobId).progress)
    }

    @Test
    fun staleAndCorruptOverlaysCannotOverrideCommittedMainState() = withStore { root, store ->
        val planning = createPlanning(store, "job-stale")
        val overlay = V2IndexingPreflightProgressOverlayPolicy.create(
            planning,
            progress(1L),
            103L,
        )
        val interrupted = store.updateLatest(planning.jobId) { current ->
            V2IndexingPreflightIntentStateMachine.interrupt(
                current,
                "Android stopped preflight",
                104L,
            )
        }
        writeOverlay(File(root, "progress/${planning.jobId}.json"), overlay)

        assertEquals(interrupted, store.require(planning.jobId))

        val progressFile = File(root, "progress/${planning.jobId}.json")
        val atomic = AtomicFile(progressFile)
        val stream = atomic.startWrite()
        stream.write("not-json".toByteArray(StandardCharsets.UTF_8))
        atomic.finishWrite(stream)
        assertEquals(interrupted, store.require(planning.jobId))
    }

    @Test
    fun processLockMakesConflictCheckAndCreateAtomicAcrossStoreInstances() {
        val root = freshRoot()
        try {
            val stores = List(2) { AtomicV2IndexingPreflightIntentStore(root) }
            val ready = CountDownLatch(2)
            val start = CountDownLatch(1)
            val pool = Executors.newFixedThreadPool(2)
            val results = stores.mapIndexed { index, store ->
                pool.submit<V2IndexingPreflightIntent?> {
                    ready.countDown()
                    assertTrue(start.await(5, TimeUnit.SECONDS))
                    store.createIfNoConflict(requested("job-admission-$index")) { existing ->
                        existing.state != V2IndexingPreflightIntentState.CANCELLED
                    }
                }
            }
            assertTrue(ready.await(5, TimeUnit.SECONDS))
            start.countDown()
            val outcomes = results.map { it.get(10, TimeUnit.SECONDS) }
            pool.shutdownNow()

            assertEquals(1, outcomes.count { it == null })
            assertEquals(1, outcomes.count { it != null })
            assertEquals(1, stores.first().list().size)
        } finally {
            root.deleteRecursively()
        }
    }

    @Test
    fun inspectionPreservesReadableHistoryAndReportsCorruptIntent() = withStore { root, store ->
        store.create(requested("job-readable"))
        val corruptFile = File(root, "job-corrupt.json")
        val corruptAtomic = AtomicFile(corruptFile)
        val stream = corruptAtomic.startWrite()
        stream.write("not-json".toByteArray(StandardCharsets.UTF_8))
        corruptAtomic.finishWrite(stream)

        val inspection = store.inspect()

        assertFalse(inspection.isComplete)
        assertEquals(listOf("job-readable"), inspection.intents.map { it.jobId })
        assertEquals("job-corrupt", inspection.issues.single().jobId)
        assertThrows(V2IndexingPreflightIntentInspectionException::class.java) {
            store.list()
        }
        assertThrows(V2IndexingPreflightIntentInspectionException::class.java) {
            store.createIfNoConflict(requested("job-new")) { false }
        }
    }

    @Test
    fun deleteUsesAtomicFileForMainAndProgressArtifacts() = withStore { root, store ->
        val planning = createPlanning(store, "job-delete")
        store.persistProgressOverlay(planning.jobId, progress(1L), 103L)
        val main = File(root, "${planning.jobId}.json")
        val progress = File(root, "progress/${planning.jobId}.json")
        File(main.path + ".new").writeText("unfinished")
        File(main.path + ".bak").writeText("legacy")
        File(progress.path + ".new").writeText("unfinished")
        File(progress.path + ".bak").writeText("legacy")

        assertTrue(store.delete(planning.jobId))
        listOf(main, progress).forEach { file ->
            assertFalse(file.exists())
            assertFalse(File(file.path + ".new").exists())
            assertFalse(File(file.path + ".bak").exists())
        }
    }

    private fun createPlanning(
        store: AtomicV2IndexingPreflightIntentStore,
        jobId: String,
    ): V2IndexingPreflightIntent {
        store.create(requested(jobId))
        return store.updateLatest(jobId) { current ->
            V2IndexingPreflightIntentStateMachine.beginOrResumePlanning(
                current,
                "generation-a",
                progress(0L),
                102L,
            )
        }
    }

    private fun requested(jobId: String) = V2IndexingPreflightIntentFactory.create(
        jobId = jobId,
        selected = listOf(
            V2IndexingPreflightSelection(
                powerampFileId = 1L,
                providerPhysicalPath = "/music/one.flac",
                durationMs = 180_000L,
                offsetMs = 0L,
                cueSourceImageFolderId = null,
            ),
        ),
        rebuildDerivedIndexes = false,
        executionProfile = V2IndexingExecutionProfile.FULL,
        nowEpochMs = 100L,
    )

    private fun progress(completed: Long) = V2IndexingPreflightProgress(
        phase = V2IndexingPreflightPhase.SOURCE_FINGERPRINTS,
        message = "Hashing source",
        completedUnits = completed,
        totalUnits = 2L,
    )

    private fun writeOverlay(file: File, overlay: V2IndexingPreflightProgressOverlay) {
        val atomic = AtomicFile(file)
        val stream = atomic.startWrite()
        try {
            val writer = OutputStreamWriter(stream, StandardCharsets.UTF_8)
            V2IndexingPreflightProgressOverlayCodec.write(gson, overlay, writer)
            writer.flush()
            atomic.finishWrite(stream)
        } catch (error: Throwable) {
            atomic.failWrite(stream)
            throw error
        }
    }

    private fun withStore(
        block: (File, AtomicV2IndexingPreflightIntentStore) -> Unit,
    ) {
        val root = freshRoot()
        try {
            block(root, AtomicV2IndexingPreflightIntentStore(root))
        } finally {
            root.deleteRecursively()
        }
    }

    private fun freshRoot(): File =
        File(context.cacheDir, "preflight-progress-${System.nanoTime()}").also { root ->
            assertTrue(root.mkdirs())
            assertNotNull(root.canonicalFile)
        }
}
