package com.powerampstartradio.indexing.v2

import android.content.Context
import android.content.ContextWrapper
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import java.io.File
import java.util.UUID
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class V2IndexingPreflightIntentStoreInstrumentedTest {
    @Test
    fun atomicIntentSurvivesReloadMaterializesProfileAndCancelWinsOverProgress() {
        val context = ApplicationProvider.getApplicationContext<android.content.Context>()
        val directory = File(context.cacheDir, "preflight-store-${UUID.randomUUID()}")
        try {
            val store = AtomicV2IndexingPreflightIntentStore(directory)
            val requested = V2IndexingPreflightIntentFactory.create(
                jobId = "job-${UUID.randomUUID()}",
                selected = listOf(
                    V2IndexingPreflightSelection(
                        powerampFileId = 7L,
                        providerPhysicalPath = "/storage/emulated/0/Music/test.flac",
                        durationMs = 180_000L,
                        offsetMs = 0L,
                        cueSourceImageFolderId = null,
                    ),
                ),
                rebuildDerivedIndexes = true,
                executionProfile = V2IndexingExecutionProfile.BALANCED,
                nowEpochMs = 100L,
            )
            store.create(requested)
            val planning = store.updateLatest(requested.jobId) {
                V2IndexingPreflightIntentStateMachine.beginOrResumePlanning(
                    it,
                    "generation-a",
                    V2IndexingPreflightProgress(
                        V2IndexingPreflightPhase.SOURCE_FINGERPRINTS,
                        "Hashing selected source identities",
                        0L,
                        1L,
                    ),
                    101L,
                )
            }
            assertEquals(planning, AtomicV2IndexingPreflightIntentStore(directory).require(requested.jobId))
            val isolatedFiles = File(directory, "files").also { check(it.mkdirs()) }
            listOf(
                "mert.tflite",
                "clamp3_audio.tflite",
                "clamp3_text.tflite",
                "sentencepiece.bpe.model",
            ).forEach { name -> File(isolatedFiles, name).writeBytes(byteArrayOf(1)) }
            val request = V2IndexingPreflightRequestMaterializer(
                IsolatedFilesContext(context, isolatedFiles),
            ).materialize(
                planning,
                V2ProviderPathGroupSnapshot(
                    libraryGeneration = "provider-a",
                    groups = listOf(
                        V2ProviderPathGroupEvidence(
                            physicalPath = requested.selected.single().providerPhysicalPath,
                            rows = listOf(
                                V2ProviderPathRowEvidence(
                                    powerampFileId = 7L,
                                    physicalPath = requested.selected.single().providerPhysicalPath,
                                    offsetMs = 0L,
                                    durationMs = 180_000L,
                                    cueSourceImageFolderId = null,
                                ),
                            ),
                            completeness = V2ProviderPathGroupCompleteness.COMPLETE,
                        ),
                    ),
                    acquisitionEvidence = V2ProviderSnapshotAcquisitionEvidence(
                        queryUri = "fixture://profile-propagation",
                        requestedColumns = emptyList(),
                        returnedColumns = emptyList(),
                        rowCount = 1,
                        cursorExhaustedNormally = true,
                    ),
                ),
            )
            assertEquals(V2IndexingExecutionProfile.BALANCED, request.executionProfile)

            val cancelling = store.updateLatest(requested.jobId) {
                V2IndexingPreflightIntentStateMachine.requestCancel(it, 102L)
            }
            val unchanged = store.updateLatest(requested.jobId) {
                V2IndexingPreflightIntentStateMachine.updateProgress(
                    it,
                    V2IndexingPreflightProgress(
                        V2IndexingPreflightPhase.SOURCE_FINGERPRINTS,
                        "Hashing selected source identities",
                        1L,
                        1L,
                    ),
                    103L,
                )
            }
            assertEquals(cancelling, unchanged)
        } finally {
            directory.deleteRecursively()
        }
    }

    private class IsolatedFilesContext(base: Context, private val root: File) :
        ContextWrapper(base) {
        override fun getApplicationContext(): Context = this
        override fun getFilesDir(): File = root
    }
}
