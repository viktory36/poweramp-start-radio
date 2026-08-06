package com.powerampstartradio.indexing.v2

import java.io.File
import java.io.IOException
import java.nio.file.Files
import org.junit.Assert.assertFalse
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class V2TerminalJobCleanupTest {
    @Test
    fun `complete ledger recovery removes crash residue and is idempotent`() = withRoot { root ->
        val artifactDirectory = artifactDirectory(root).apply {
            mkdirs()
            File(this, "mert/features.bin").apply {
                parentFile?.mkdirs()
                writeText("verified intermediate work")
            }
        }
        val stagingFiles = createStagingResidue(root)
        val publishedGeneration = File(
            root,
            "indexing_v2/generations/generation-live/embeddings.db",
        ).apply {
            parentFile?.mkdirs()
            writeText("published generation")
        }
        val cleanup = cleaner(root)

        cleanup.cleanup(JOB_ID, IndexingJobState.COMPLETE)
        cleanup.cleanup(JOB_ID, IndexingJobState.COMPLETE)

        assertFalse(artifactDirectory.exists())
        assertTrue(stagingFiles.none(File::exists))
        assertTrue(publishedGeneration.isFile)
        assertTrue(publishedGeneration.readText() == "published generation")
    }

    @Test
    fun `cleanup failure remains visible and a later terminal recovery can retry`() =
        withRoot { root ->
            val artifactDirectory = artifactDirectory(root).apply {
                mkdirs()
                File(this, "remaining.bin").writeText("residue")
            }
            val stagingFiles = createStagingResidue(root)
            var deletionMayFinish = false
            val cleanup = V2TerminalJobCleanup(
                artifactDirectory = { artifactDirectory },
                cleanupStagingDatabase = V2JobPrivateDatabaseStore(root)::cleanup,
                deleteArtifactTree = { directory ->
                    deletionMayFinish && directory.deleteRecursively()
                },
            )

            assertThrows(IOException::class.java) {
                cleanup.cleanup(JOB_ID, IndexingJobState.COMPLETE)
            }
            assertTrue(artifactDirectory.isDirectory)
            assertTrue(stagingFiles.none(File::exists))

            deletionMayFinish = true
            cleanup.cleanup(JOB_ID, IndexingJobState.COMPLETE)
            assertFalse(artifactDirectory.exists())
        }

    @Test
    fun `nonterminal work is never disposable`() = withRoot { root ->
        val artifactDirectory = artifactDirectory(root).apply {
            mkdirs()
            File(this, "keep.bin").writeText("in flight")
        }
        val stagingFiles = createStagingResidue(root)

        assertThrows(IllegalArgumentException::class.java) {
            cleaner(root).cleanup(JOB_ID, IndexingJobState.RUNNING)
        }

        assertTrue(artifactDirectory.isDirectory)
        assertTrue(stagingFiles.all(File::exists))
    }

    private fun cleaner(root: File) = V2TerminalJobCleanup(
        artifactDirectory = { artifactDirectory(root) },
        cleanupStagingDatabase = V2JobPrivateDatabaseStore(root)::cleanup,
    )

    private fun artifactDirectory(root: File): File =
        File(root, "indexing_v2/artifacts/$JOB_ID")

    private fun createStagingResidue(root: File): List<File> {
        val database = File(root, "indexing_v2/job-databases/$JOB_ID.db")
        return listOf(
            database,
            File(database.path + "-wal"),
            File(database.path + "-shm"),
            File(database.path + "-journal"),
            File(database.parentFile, "$JOB_ID.binding.json"),
            File(database.parentFile, "$JOB_ID.binding.json.new"),
            File(database.parentFile, "$JOB_ID.binding.json.bak"),
        ).onEach { file ->
            file.parentFile?.mkdirs()
            file.writeText("private staging residue")
        }
    }

    private inline fun withRoot(test: (File) -> Unit) {
        val root = Files.createTempDirectory("v2-terminal-cleanup").toFile()
        try {
            test(root)
        } finally {
            root.deleteRecursively()
        }
    }

    private companion object {
        const val JOB_ID = "completed-job"
    }
}
