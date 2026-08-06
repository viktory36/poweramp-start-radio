package com.powerampstartradio.widget

import com.powerampstartradio.poweramp.PowerampHelper
import com.powerampstartradio.poweramp.PowerampTrack
import com.powerampstartradio.ui.RadioSeedIdentity
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

class WidgetRadioStatusStoreTest {
    @Test
    fun `listener status text is typed and never renders persisted diagnostics`() {
        val diagnostic = "internal score=0.4123 / java.lang.IllegalStateException"
        val status = status("request-a", track(10), WidgetRadioRequestState.FAILED)
            .copy(message = diagnostic)

        assertEquals(
            "Radio failed \u00b7 Open app",
            WidgetRadioPresentationPolicy.listenerStatusText(status.state),
        )
        assertFalse(WidgetRadioPresentationPolicy.listenerStatusText(status.state).contains("0.4123"))
    }

    @Test
    fun `widget title uses only the current sticky track`() {
        val cached = track(10)
        val sticky = track(11)

        assertEquals(
            sticky,
            WidgetPlaybackTrackFreshnessPolicy.select(
                stickyTrack = sticky,
                cachedTrack = cached,
            ),
        )
        assertNull(
            WidgetPlaybackTrackFreshnessPolicy.select(
                stickyTrack = null,
                cachedTrack = cached,
            ),
        )
    }

    @Test
    fun `widget starts radio only from a provider-ready concrete track`() {
        assertEquals(
            WidgetPrimaryAction.START_RADIO,
            WidgetRadioPresentationPolicy.primaryAction(playback(track(7))),
        )
        assertEquals(
            WidgetPrimaryAction.OPEN_POWERAMP,
            WidgetRadioPresentationPolicy.primaryAction(
                playback(track(7)).copy(readiness = WidgetPlaybackReadiness.REFRESH_POWERAMP),
            ),
        )
        assertEquals(
            WidgetPrimaryAction.OPEN_POWERAMP,
            WidgetRadioPresentationPolicy.primaryAction(
                playback(track(7)).copy(track = null, readiness = WidgetPlaybackReadiness.NO_TRACK),
            ),
        )
    }

    @Test
    fun `provider-ready paused track remains actionable and owns its listener status`() {
        val pausedTrack = track(7)
        val pausedPlayback = playback(pausedTrack).copy(
            playbackState = PowerampHelper.STATE_PAUSED,
        )
        val starting = status(
            "request-paused",
            pausedTrack,
            WidgetRadioRequestState.STARTING,
        )

        assertEquals(
            WidgetPrimaryAction.START_RADIO,
            WidgetRadioPresentationPolicy.primaryAction(pausedPlayback),
        )
        assertEquals(
            starting,
            WidgetRadioPresentationPolicy.visibleStatus(
                playback = pausedPlayback,
                status = starting,
                nowEpochMs = 2_000L,
            ),
        )
    }

    @Test
    fun `structured request and exact seed survive atomic round trip`() = withTempDir { root ->
        val store = store(root)
        val expected = status("request-a", track(10), WidgetRadioRequestState.STARTING)

        store.write(expected)

        assertEquals(expected, store(root).read())
    }

    @Test
    fun `only the owning request can update status`() = withTempDir { root ->
        val store = store(root)
        val starting = status("request-a", track(10), WidgetRadioRequestState.STARTING)
        store.write(starting)

        assertFalse(
            store.updateMatchingRequest(
                requestId = "request-b",
                state = WidgetRadioRequestState.SUCCEEDED,
                message = "wrong owner",
                updatedAtEpochMs = 2_000L,
            ),
        )
        assertEquals(starting, store.read())

        assertTrue(
            store.updateMatchingRequest(
                requestId = "request-a",
                state = WidgetRadioRequestState.SUCCEEDED,
                message = "50 tracks ready",
                updatedAtEpochMs = 2_000L,
            ),
        )
        assertEquals(WidgetRadioRequestState.SUCCEEDED, store.read()?.state)
        assertEquals(starting.seed, store.read()?.seed)
    }

    @Test
    fun `terminal redelivery preserves richer partial and cancelled evidence`() =
        withTempDir { root ->
            listOf(
                WidgetRadioRequestState.PARTIAL_FAILED,
                WidgetRadioRequestState.CANCELLED,
            ).forEach { existingState ->
                val store = store(root)
                val existing = status("request-a", track(10), existingState)
                store.write(existing)

                assertTrue(
                    store.updateMatchingRequest(
                        requestId = "request-a",
                        state = WidgetRadioRequestState.FAILED,
                        message = "generic lifecycle failure",
                        updatedAtEpochMs = 2_000L,
                        preserveCurrentStates = setOf(
                            WidgetRadioRequestState.PARTIAL_FAILED,
                            WidgetRadioRequestState.CANCELLED,
                        ),
                    ),
                )
                assertEquals(existing, store.read())
            }
        }

    @Test
    fun `status is hidden as soon as authoritative playback identity changes`() {
        val original = track(10)
        val status = status("request-a", original, WidgetRadioRequestState.BUSY)

        assertEquals(
            status,
            WidgetRadioPresentationPolicy.visibleStatus(
                playback = playback(original),
                status = status,
                nowEpochMs = 2_000L,
            ),
        )
        assertNull(
            WidgetRadioPresentationPolicy.visibleStatus(
                playback = playback(track(11)),
                status = status,
                nowEpochMs = 2_000L,
            ),
        )
        assertNull(
            WidgetRadioPresentationPolicy.visibleStatus(
                playback = playback(original).copy(
                    readiness = WidgetPlaybackReadiness.REFRESH_POWERAMP,
                ),
                status = status,
                nowEpochMs = 2_000L,
            ),
        )
        assertNull(
            WidgetRadioPresentationPolicy.visibleStatus(
                playback = playback(original),
                status = status,
                nowEpochMs = 1_000L + WidgetRadioPresentationPolicy.STATUS_VISIBLE_MS + 1L,
            ),
        )
    }

    @Test
    fun `successful enqueue is acknowledged briefly while actionable states remain visible`() {
        val track = track(10)
        val succeeded = status("request-a", track, WidgetRadioRequestState.SUCCEEDED)

        assertEquals(
            succeeded,
            WidgetRadioPresentationPolicy.visibleStatus(
                playback = playback(track),
                status = succeeded,
                nowEpochMs = 2_000L,
            ),
        )
        assertNull(
            WidgetRadioPresentationPolicy.visibleStatus(
                playback = playback(track),
                status = succeeded,
                nowEpochMs =
                    1_000L + WidgetRadioPresentationPolicy.SUCCESS_VISIBLE_MS + 1L,
            ),
        )
        listOf(
            WidgetRadioRequestState.STARTING,
            WidgetRadioRequestState.BUSY,
            WidgetRadioRequestState.WAITING_FOR_INDEXING,
            WidgetRadioRequestState.PARTIAL_FAILED,
            WidgetRadioRequestState.CANCELLED,
            WidgetRadioRequestState.FAILED,
        ).forEach { state ->
            val status = status("request-$state", track, state)
            assertEquals(
                status,
                WidgetRadioPresentationPolicy.visibleStatus(
                    playback = playback(track),
                    status = status,
                    nowEpochMs = 2_000L,
                ),
            )
        }
    }

    @Test
    fun `active status cannot be persisted without an exact seed`() = withTempDir { root ->
        val invalid = WidgetRadioStatus(
            requestId = "request-a",
            seed = null,
            state = WidgetRadioRequestState.BUSY,
            message = "Starting",
            updatedAtEpochMs = 1_000L,
        )

        assertThrows(IllegalArgumentException::class.java) { store(root).write(invalid) }
    }

    @Test
    fun `duplicate file in a different queue occurrence does not inherit status`() {
        val firstOccurrence = track(10).copy(
            trackId = 100L,
            categoryUri = "content://com.maxmpz.audioplayer.data/queue?shs=2",
        )
        val status = status("request-a", firstOccurrence, WidgetRadioRequestState.SUCCEEDED)

        assertNull(
            WidgetRadioPresentationPolicy.visibleStatus(
                playback = playback(firstOccurrence.copy(trackId = 101L)),
                status = status,
                nowEpochMs = 2_000L,
            ),
        )
    }

    private fun status(
        requestId: String,
        track: PowerampTrack,
        state: WidgetRadioRequestState,
    ) = WidgetRadioStatus(
        requestId = requestId,
        seed = WidgetRadioSeedReference.from(
            track,
            RadioSeedIdentity(track.realId, "stable-track-span-v1-${"a".repeat(64)}"),
        ),
        state = state,
        message = "Starting radio for ${track.title}",
        updatedAtEpochMs = 1_000L,
    )

    private fun playback(track: PowerampTrack) = WidgetPlaybackSnapshot(
        track = track,
        playbackState = 1,
        readiness = WidgetPlaybackReadiness.READY,
    )

    private fun track(id: Long) = PowerampTrack(
        realId = id,
        title = "Track $id",
        artist = "Artist",
        album = "Album",
        durationMs = 180_000,
        path = "/music/$id.flac",
    )

    private fun store(root: File) = WidgetRadioStatusStore(
        rootDir = root,
        atomicWriter = WidgetStatusAtomicWriter { file, bytes ->
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
        atomicReader = WidgetStatusAtomicReader { file ->
            try {
                file.readBytes()
            } catch (_: FileNotFoundException) {
                null
            }
        },
    )

    private fun withTempDir(block: (File) -> Unit) {
        val root = Files.createTempDirectory("widget-radio-status-test").toFile()
        try {
            block(root)
        } finally {
            root.deleteRecursively()
        }
    }
}
