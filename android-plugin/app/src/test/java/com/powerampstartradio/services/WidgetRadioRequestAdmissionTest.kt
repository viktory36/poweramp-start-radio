package com.powerampstartradio.services

import com.powerampstartradio.poweramp.PowerampTrack
import com.powerampstartradio.poweramp.TrackMatcher
import com.powerampstartradio.ui.QueueOrigin
import com.powerampstartradio.ui.RadioConfig
import com.powerampstartradio.ui.RadioGenerationToken
import com.powerampstartradio.ui.RadioSeedIdentity
import com.powerampstartradio.ui.SelectionMode
import java.io.File
import java.io.FileNotFoundException
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class WidgetRadioRequestAdmissionTest {
    @Test
    fun `widget busy status requires a durable owner request`() {
        assertEquals(
            "reserved-request",
            WidgetBusyStatusOwnerPolicy.resolve(
                reservedRequestId = "reserved-request",
                activeDurableRequestId = "active-request",
            ),
        )
        assertEquals(
            "active-request",
            WidgetBusyStatusOwnerPolicy.resolve(
                reservedRequestId = null,
                activeDurableRequestId = "active-request",
            ),
        )
        assertNull(
            WidgetBusyStatusOwnerPolicy.resolve(
                reservedRequestId = null,
                activeDurableRequestId = null,
            ),
        )
    }

    @Test
    fun `duplicate widget command ID republishes only an identical immutable request`() =
        withTempDir { root ->
            val store = store(root, "receiver-process")
            val request = request("00000000-0000-1000-8000-000000000001")

            assertEquals(request.requestId, store.persistIdempotently(request))
            assertEquals(request.requestId, store.persistIdempotently(request))
            assertEquals(request, store.readRequest(request.requestId))
            assertTrue(store.hasRecord(request.requestId))

            val changed = request.copy(
                radio = request.radio!!.copy(config = request.radio.config.copy(numTracks = 2)),
            )
            assertThrows(IllegalArgumentException::class.java) {
                store.persistIdempotently(changed)
            }
        }

    @Test
    fun `foreign process sees claimed widget command as the existing single flight`() =
        withTempDir { root ->
            val request = request("00000000-0000-1000-8000-000000000002")
            val service = store(root, "service-process-a")
            service.persist(request)
            assertTrue(service.claim(request.requestId) is RadioRequestClaim.Claimed)

            val coldReceiver = store(root, "receiver-process-b")
            assertEquals(listOf(request.requestId), coldReceiver.recoverableRequestIds())
            assertEquals(request, coldReceiver.readRequest(request.requestId))
        }

    private fun request(requestId: String) = DurableRadioRequest.radio(
        generation = RadioGenerationToken(
            generationId = "index-generation-v2-${"1".repeat(64)}",
            activationBindingId = "activation-binding-v3-${"2".repeat(64)}",
            manifestSha256 = "3".repeat(64),
            embeddingSpecId = "clamp3-audio-v2",
            databaseContentSha256 = "4".repeat(64),
            orderedTrackSetSha256 = "5".repeat(64),
            stableTrackUidMappingSha256 = "6".repeat(64),
        ),
        providerGenerationId = "poweramp-provider-snapshot-v3-sha256:${"7".repeat(64)}",
        config = RadioConfig(
            numTracks = 1,
            selectionMode = SelectionMode.UNIFORM_SHUFFLE,
            shuffleSeed = 123L,
        ),
        seed = PinnedRadioSeed(
            identity = RadioSeedIdentity(
                embeddedTrackId = 12L,
                stableTrackSpanId = "stable-track-span-v1-${"8".repeat(64)}",
            ),
            displayTrack = PowerampTrack(
                realId = 120L,
                title = "Frozen tap seed",
                artist = "Artist",
                album = "Album",
                durationMs = 180_000,
                path = "/music/frozen.flac",
            ),
            matchType = TrackMatcher.MatchType.ACTIVE_CATALOG_EXACT,
        ),
        showToasts = true,
        origin = QueueOrigin.WIDGET_RADIO,
        requestId = requestId,
        createdAtEpochMs = 1_000L,
    )

    private fun store(root: File, owner: String) = RadioRequestStore(
        rootDir = root,
        ownerToken = owner,
        clock = { 2_000L },
        atomicWriter = RadioRequestAtomicWriter { file, bytes ->
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
        },
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

    private fun withTempDir(block: (File) -> Unit) {
        val root = Files.createTempDirectory("widget-radio-request-test").toFile()
        try {
            block(root)
        } finally {
            root.deleteRecursively()
        }
    }
}
