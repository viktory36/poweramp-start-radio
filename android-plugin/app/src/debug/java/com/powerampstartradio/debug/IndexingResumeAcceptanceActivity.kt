package com.powerampstartradio.debug

import android.app.Activity
import android.os.Bundle
import androidx.core.content.ContextCompat
import com.powerampstartradio.indexing.IndexingService
import com.powerampstartradio.indexing.V2IndexingServiceIntents

/** Debug-only bridge for exercising real indexing resume/recovery commands from adb. */
class IndexingResumeAcceptanceActivity : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        showOverLockScreenForAcceptance()

        val jobId = requireNotNull(intent.getStringExtra(EXTRA_JOB_ID)) {
            "missing $EXTRA_JOB_ID"
        }
        when (intent.getStringExtra(EXTRA_COMMAND) ?: COMMAND_RESUME) {
            COMMAND_RESUME -> IndexingService.resume(this, jobId)
            COMMAND_RECOVER -> ContextCompat.startForegroundService(
                this,
                V2IndexingServiceIntents.recover(this, jobId),
            )
            else -> error("unsupported $EXTRA_COMMAND")
        }
        finishAndRemoveTask()
    }

    companion object {
        const val EXTRA_JOB_ID = "indexing_job_id"
        const val EXTRA_COMMAND = "indexing_command"
        const val COMMAND_RESUME = "resume"
        const val COMMAND_RECOVER = "recover"
    }
}
