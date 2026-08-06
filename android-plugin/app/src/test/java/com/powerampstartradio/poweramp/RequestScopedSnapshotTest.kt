package com.powerampstartradio.poweramp

import org.junit.Assert.assertEquals
import org.junit.Assert.assertSame
import org.junit.Assert.assertThrows
import org.junit.Test

class RequestScopedSnapshotTest {
    @Test
    fun `one request publishes one snapshot while a new request refreshes`() {
        var loads = 0
        val firstRequest = RequestScopedSnapshot<List<Long>>()

        assertEquals(listOf(1L), firstRequest.require { listOf((++loads).toLong()) })
        assertEquals(listOf(1L), firstRequest.require { listOf((++loads).toLong()) })
        assertEquals(1, loads)

        val nextRequest = RequestScopedSnapshot<List<Long>>()
        assertEquals(listOf(2L), nextRequest.require { listOf((++loads).toLong()) })
        assertEquals(2, loads)
    }

    @Test
    fun `failed snapshot is cached and cannot become a mixed retry`() {
        var loads = 0
        val request = RequestScopedSnapshot<List<Long>>()
        val expected = IllegalStateException("provider cursor failed")

        val first = assertThrows(IllegalStateException::class.java) {
            request.require {
                loads++
                throw expected
            }
        }
        val second = assertThrows(IllegalStateException::class.java) {
            request.require {
                loads++
                listOf(2L)
            }
        }

        assertSame(expected, first)
        assertSame(expected, second)
        assertEquals(1, loads)
    }
}
