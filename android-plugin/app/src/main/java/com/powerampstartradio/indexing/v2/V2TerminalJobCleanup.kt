package com.powerampstartradio.indexing.v2

import java.io.File
import java.io.IOException

/** Removes only job-private work after the durable ledger says it is disposable. */
internal class V2TerminalJobCleanup(
    private val artifactDirectory: (String) -> File,
    private val cleanupStagingDatabase: (String) -> Unit,
    private val deleteArtifactTree: (File) -> Boolean = File::deleteRecursively,
) {
    fun cleanup(jobId: String, state: IndexingJobState) {
        require(jobId.matches(SAFE_JOB_ID)) { "unsafe job id" }
        require(state == IndexingJobState.COMPLETE || state == IndexingJobState.CANCELLED) {
            "job-private work is not disposable while the job is $state"
        }

        val failures = mutableListOf<Throwable>()
        runCatching {
            val directory = artifactDirectory(jobId)
            if (directory.exists()) {
                val reportedDeleted = deleteArtifactTree(directory)
                if (!reportedDeleted || directory.exists()) {
                    throw IOException("unable to remove private artifacts for job $jobId")
                }
            }
        }.exceptionOrNull()?.let(failures::add)

        runCatching { cleanupStagingDatabase(jobId) }
            .exceptionOrNull()
            ?.let(failures::add)

        if (failures.isNotEmpty()) {
            throw IOException(
                "temporary indexing files remain for terminal job $jobId",
                failures.first(),
            ).apply {
                failures.drop(1).forEach(::addSuppressed)
            }
        }
    }

    private companion object {
        val SAFE_JOB_ID = Regex("^[A-Za-z0-9._-]{1,128}$")
    }
}
