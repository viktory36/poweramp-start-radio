package com.powerampstartradio.indexing

import android.app.Notification
import android.app.NotificationManager
import android.os.SystemClock
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.powerampstartradio.indexing.v2.IndexingJobState
import com.powerampstartradio.indexing.v2.V2AtomicExecutorLeasePersistence
import com.powerampstartradio.indexing.v2.V2IndexingJobRepository
import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class V2MediaProcessingTimeoutAcceptanceTest {
    @Test
    fun realQuotaTimeoutCannotRestartWorkAfterStoppingTheService() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val jobId = InstrumentationRegistry.getArguments().getString(ARG_JOB_ID)
        assumeTrue("Opt-in production job ID was not supplied", !jobId.isNullOrBlank())
        val resolvedJobId = requireNotNull(jobId)
        val context = instrumentation.targetContext
        val repository = V2IndexingJobRepository.get(context)
        repository.reconcileStartup()

        val before = repository.require(resolvedJobId)
        assertTrue(
            "Fixture must already be safely resumable",
            before.state == IndexingJobState.INTERRUPTED || before.state == IndexingJobState.PAUSED,
        )
        val beforeAttempts = before.tracks.map { it.attemptCount }
        val beforeArtifacts = before.tracks.map { track ->
            track.verifiedArtifacts.map { artifact ->
                Triple(artifact.kind, artifact.sha256, artifact.byteLength)
            }
        }
        val activeGenerationFile = File(
            context.filesDir,
            "indexing_v2/generations/active-generation.json",
        )
        val activeGenerationBefore = activeGenerationFile.readBytes()
        val notificationManager = context.getSystemService(NotificationManager::class.java)
        notificationManager.cancel(INDEXING_NOTIFICATION_ID)

        IndexingService.resume(context, resolvedJobId)

        assertTrue(
            "Android did not deliver the expected exhausted media-processing quota callback",
            waitUntil(TIMEOUT_MS) {
                notificationManager.activeNotifications.any { active ->
                    active.id == INDEXING_NOTIFICATION_ID &&
                        active.notification.extras
                            .getCharSequence(Notification.EXTRA_TEXT)
                            ?.contains("media-processing time limit was reached") == true
                }
            },
        )
        assertTrue(
            "Timed-out executor did not settle durably",
            waitUntil(TIMEOUT_MS) {
                val ledger = repository.require(resolvedJobId)
                val lease = V2AtomicExecutorLeasePersistence(context.filesDir).read().active
                ledger.state in setOf(IndexingJobState.INTERRUPTED, IndexingJobState.PAUSED) &&
                    lease == null
            },
        )

        val after = repository.require(resolvedJobId)
        assertTrue(after.state == IndexingJobState.INTERRUPTED || after.state == IndexingJobState.PAUSED)
        assertEquals(beforeAttempts, after.tracks.map { it.attemptCount })
        assertEquals(
            beforeArtifacts,
            after.tracks.map { track ->
                track.verifiedArtifacts.map { artifact ->
                    Triple(artifact.kind, artifact.sha256, artifact.byteLength)
                }
            },
        )
        assertEquals(activeGenerationBefore.toList(), activeGenerationFile.readBytes().toList())
        assertNull(V2AtomicExecutorLeasePersistence(context.filesDir).read().active)
    }

    private fun waitUntil(timeoutMs: Long, condition: () -> Boolean): Boolean {
        val deadline = SystemClock.elapsedRealtime() + timeoutMs
        while (SystemClock.elapsedRealtime() < deadline) {
            if (condition()) return true
            SystemClock.sleep(POLL_MS)
        }
        return condition()
    }

    private companion object {
        const val ARG_JOB_ID = "indexing_job_id"
        const val INDEXING_NOTIFICATION_ID = 2
        const val TIMEOUT_MS = 10_000L
        const val POLL_MS = 100L
    }
}
