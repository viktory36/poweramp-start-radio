package com.powerampstartradio

import android.app.Application
import android.util.Log
import com.powerampstartradio.indexing.IndexingService
import com.powerampstartradio.indexing.v2.V2ActiveIndexingJobPointer
import com.powerampstartradio.indexing.v2.V2ActiveIndexingJobPointerInspection
import com.powerampstartradio.indexing.v2.V2ActiveIndexingJobPointerUnreadableReason
import com.powerampstartradio.indexing.v2.V2GenerationPublicationCoordinator
import com.powerampstartradio.indexing.v2.V2IndexingJobRepository
import com.powerampstartradio.indexing.v2.isTerminal
import com.powerampstartradio.services.RecommendationWorkAdmission
import com.powerampstartradio.services.RadioService
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch

class PowerampStartRadioApp : Application() {
    private val applicationScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    override fun onCreate() {
        super.onCreate()
        RecommendationWorkAdmission.beginColdReconciliation()
        val coldPointer = runCatching { V2ActiveIndexingJobPointer(filesDir).inspect() }
            .getOrElse { error ->
                Log.e(TAG, "Unable to inspect active indexing ownership; reserving fail-closed", error)
                V2ActiveIndexingJobPointerInspection.Unreadable(
                    V2ActiveIndexingJobPointerUnreadableReason.IO_FAILURE,
                )
            }
        if (coldPointer is V2ActiveIndexingJobPointerInspection.Unreadable) {
            Log.w(
                TAG,
                "Active indexing ownership is unreadable (${coldPointer.reason}); " +
                    "recommendation work remains reserved",
            )
        }
        runCatching {
            V2GenerationPublicationCoordinator
                .reconcileAbandonedStagingAtColdProcessStart(filesDir)
        }.onSuccess { result ->
            if (result.deletedDirectoryCount > 0 || result.failedDirectoryCount > 0) {
                Log.i(
                    TAG,
                    "Cold-start generation staging cleanup: " +
                        "deleted=${result.deletedDirectoryCount}, failed=${result.failedDirectoryCount}",
                )
            }
        }.onFailure { error ->
            Log.e(TAG, "Unable to reconcile abandoned generation staging", error)
        }
        // Widget actions can be the first entry point into a cold process. History must
        // therefore be ready independently of MainActivity/MainViewModel creation.
        RadioService.initHistory(filesDir)
        // Cold process entry may be a foreground-service launch. Opening a large job ledger here
        // would consume the platform's short startForeground deadline, so reconciliation is
        // intentionally asynchronous; boot or the visible indexing UI may still resume it.
        applicationScope.launch {
            var coldReconciliationSettled = false
            runCatching {
                val repository = V2IndexingJobRepository.get(this@PowerampStartRadioApp)
                repository.reconcileStartup()
                IndexingService.attach(this@PowerampStartRadioApp)
                val recommendationWorkAvailable =
                    RecommendationWorkAdmission.finishColdReconciliation(
                        success = IndexingService.state.value !is IndexingService.IndexingState.Error,
                    )
                coldReconciliationSettled = true
                if (recommendationWorkAvailable) {
                    RadioService.kickDeferredRecovery(this@PowerampStartRadioApp)
                }
                val protectedJobs = repository.jobs.value
                    .filterNot { it.state.isTerminal() }
                    .mapTo(mutableSetOf()) { it.jobSpec.jobId }
                V2GenerationPublicationCoordinator.reconcileCrashOrphans(
                    filesDir = filesDir,
                    protectedNonterminalJobIds = protectedJobs,
                ).also { result ->
                    if (result.deletedGenerationIds.isNotEmpty() ||
                        result.retainedJobGenerationIds.isNotEmpty() ||
                        result.retainedUnverifiedGenerationIds.isNotEmpty() ||
                        result.failedDeletionGenerationIds.isNotEmpty() || result.skipReason != null
                    ) {
                        Log.i(
                            TAG,
                            "Generation orphan reconciliation: deleted=" +
                                "${result.deletedGenerationIds.size}, protected=" +
                                "${result.retainedJobGenerationIds.size}, unverified=" +
                                "${result.retainedUnverifiedGenerationIds.size}, failed=" +
                                "${result.failedDeletionGenerationIds.size}, " +
                                "skipped=${result.skipReason}",
                        )
                    }
                }
            }.onFailure { error ->
                if (!coldReconciliationSettled) {
                    RecommendationWorkAdmission.finishColdReconciliation(success = false)
                }
                Log.e(TAG, "Unable to reconcile indexing state during cold start", error)
            }
        }
    }

    private companion object {
        const val TAG = "PowerampStartRadioApp"
    }
}
