package com.powerampstartradio.poweramp

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class PowerampQueueOccurrencePolicyTest {
    @Test
    fun `queue category turns Track ID into exact queue occurrence`() {
        assertEquals(
            44L,
            PowerampQueueOccurrencePolicy.queueOccurrenceId(
                trackId = 44L,
                categoryUri = "content://com.maxmpz.audioplayer.data/queue?shs=2",
            ),
        )
        assertTrue(
            PowerampQueueOccurrencePolicy.isQueueCategory(
                "content://com.maxmpz.audioplayer.data/queue/",
            ),
        )
    }

    @Test
    fun `file or playlist identity is never reinterpreted as queue row`() {
        assertNull(
            PowerampQueueOccurrencePolicy.queueOccurrenceId(
                trackId = 44L,
                categoryUri = "content://com.maxmpz.audioplayer.data/files/12",
            ),
        )
        assertNull(
            PowerampQueueOccurrencePolicy.queueOccurrenceId(
                trackId = 44L,
                categoryUri = "content://com.maxmpz.audioplayer.data/playlists/7/files",
            ),
        )
        assertFalse(
            PowerampQueueOccurrencePolicy.isQueueCategory(
                "content://untrusted.provider/queue",
            ),
        )
        assertNull(
            PowerampQueueOccurrencePolicy.queueOccurrenceId(
                trackId = -1L,
                categoryUri = "content://com.maxmpz.audioplayer.data/queue",
            ),
        )
    }
}
