package com.powerampstartradio.services

import com.powerampstartradio.ui.RadioConfig
import com.powerampstartradio.ui.SelectionMode
import com.powerampstartradio.ui.forSelectionRequest
import com.powerampstartradio.widget.WidgetRadioSeedReference
import java.io.File
import java.io.FileNotFoundException
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class WidgetRadioIngressStoreTest {
    @Test
    fun `pending ingress is published only after precommit succeeds`() = withTempDir { root ->
        val store = store(root)
        val commandId = "00000000-0000-1000-8000-000000000001"
        var precommitRan = false

        val admission = store.admitAfterPrecommit(
            commandId = commandId,
            expectedSeed = seed(),
            config = config(),
        ) { pending ->
            assertEquals(commandId, pending.commandId)
            assertNull(store.read(commandId))
            assertTrue(store.pendingRecords().isEmpty())
            precommitRan = true
        }

        assertTrue(precommitRan)
        assertTrue(admission is WidgetRadioIngressAdmission.Accepted)
        assertEquals(commandId, store.read(commandId)?.commandId)
        assertEquals(17, store.read(commandId)?.config?.libraryAddedDays)
        assertEquals(listOf(commandId), store.pendingRecords().map { it.commandId })
    }

    @Test
    fun `failed precommit leaves no executable ingress`() = withTempDir { root ->
        val store = store(root)
        val commandId = "00000000-0000-1000-8000-000000000002"

        assertThrows(IllegalStateException::class.java) {
            store.admitAfterPrecommit(
                commandId = commandId,
                expectedSeed = seed(),
                config = config(),
            ) {
                throw IllegalStateException("foreground owner unavailable")
            }
        }

        assertNull(store.read(commandId))
        assertTrue(store.pendingRecords().isEmpty())
    }

    @Test
    fun `existing ingress remains authoritative without rerunning precommit`() =
        withTempDir { root ->
            val store = store(root)
            val commandId = "00000000-0000-1000-8000-000000000003"
            store.admit(commandId, seed(), config())
            var precommitRan = false

            val admission = store.admitAfterPrecommit(
                commandId = commandId,
                expectedSeed = seed(),
                config = config().copy(numTracks = 10),
            ) {
                precommitRan = true
            }

            assertFalse(precommitRan)
            admission as WidgetRadioIngressAdmission.Accepted
            assertFalse(admission.newlyPersisted)
            assertEquals(config(), admission.record.config)
        }

    @Test
    fun `materialization failure terminalizes ingress and always schedules bounded stop`() =
        withTempDir { root ->
            val store = store(root)
            val commandId = "00000000-0000-1000-8000-000000000004"
            store.admit(commandId, seed(), config())
            var publishedSeed: WidgetRadioSeedReference? = null
            var stopScheduled = false

            val recovery = recoverFailedWidgetIngress(
                store = store,
                commandId = commandId,
                detail = "could not materialize",
                publishFailure = { publishedSeed = it },
                scheduleBoundedStop = { stopScheduled = true },
            )

            assertTrue(recovery.terminalized)
            assertTrue(recovery.statusPublished)
            assertNull(recovery.terminalizationFailure)
            assertNull(recovery.statusFailure)
            assertEquals(seed(), publishedSeed)
            assertTrue(stopScheduled)
            assertTrue(store.pendingRecords().isEmpty())
            assertEquals(WidgetRadioIngressState.FAILED, store.read(commandId)?.state)
        }

    @Test
    fun `failed terminal and status writes retain retry ingress but still schedule stop`() =
        withTempDir { root ->
            var failWrites = false
            val store = store(
                root,
                atomicWriter = fileWriter { failWrites },
            )
            val commandId = "00000000-0000-1000-8000-000000000005"
            store.admit(commandId, seed(), config())
            failWrites = true
            var stopScheduled = false

            val recovery = recoverFailedWidgetIngress(
                store = store,
                commandId = commandId,
                detail = "could not materialize",
                publishFailure = { throw IllegalStateException("status unavailable") },
                scheduleBoundedStop = { stopScheduled = true },
            )

            assertFalse(recovery.terminalized)
            assertFalse(recovery.statusPublished)
            assertTrue(recovery.terminalizationFailure is IllegalStateException)
            assertTrue(recovery.statusFailure is IllegalStateException)
            assertTrue(stopScheduled)
            assertEquals(listOf(commandId), store.pendingRecords().map { it.commandId })
        }

    private fun seed() = WidgetRadioSeedReference(
        powerampFileId = 42L,
        normalizedPath = "/music/seed.flac",
        normalizedTitle = "seed",
        displayTitle = "Seed",
        queueOccurrenceId = 7L,
    )

    private fun config() = RadioConfig(
        numTracks = 30,
        libraryAddedDays = 17,
        selectionMode = SelectionMode.MMR,
    ).forSelectionRequest()

    private fun store(
        root: File,
        atomicWriter: RadioRequestAtomicWriter = fileWriter(),
    ) = WidgetRadioIngressStore(
        rootDir = root,
        clock = { 1_000L },
        atomicWriter = atomicWriter,
        atomicReader = RadioRequestAtomicReader { file ->
            try {
                file.readBytes()
            } catch (_: FileNotFoundException) {
                null
            }
        },
        atomicDeleter = RadioRequestAtomicDeleter { file ->
            file.delete()
            !file.exists()
        },
    )

    private fun fileWriter(
        shouldFail: () -> Boolean = { false },
    ) = RadioRequestAtomicWriter { file, bytes ->
        if (shouldFail()) throw IllegalStateException("journal unavailable")
        file.parentFile?.mkdirs()
        val temporary = File(file.parentFile, "${file.name}.tmp")
        temporary.outputStream().use { output ->
            output.write(bytes)
            output.flush()
        }
        Files.move(
            temporary.toPath(),
            file.toPath(),
            StandardCopyOption.REPLACE_EXISTING,
        )
    }

    private fun withTempDir(block: (File) -> Unit) {
        val root = Files.createTempDirectory("widget-radio-ingress-test").toFile()
        try {
            block(root)
        } finally {
            root.deleteRecursively()
        }
    }
}
