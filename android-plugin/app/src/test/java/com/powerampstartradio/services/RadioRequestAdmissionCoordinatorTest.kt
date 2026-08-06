package com.powerampstartradio.services

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class RadioRequestAdmissionCoordinatorTest {
    @Test
    fun `busy admission does not schedule work`() {
        var scheduled = false

        val result = RadioRequestAdmissionCoordinator.admit(
            reserve = { null },
            release = { error("Nothing was reserved") },
            schedule = { scheduled = true },
        )

        assertEquals(RadioRequestAdmission.BUSY, result)
        assertFalse(scheduled)
    }

    @Test
    fun `accepted admission schedules with the owned token`() {
        var scheduledToken: String? = null

        val result = RadioRequestAdmissionCoordinator.admit(
            reserve = { "submission-a" },
            release = { error("Accepted scheduling must retain its reservation") },
            schedule = { scheduledToken = it },
        )

        assertEquals(RadioRequestAdmission.ACCEPTED, result)
        assertEquals("submission-a", scheduledToken)
    }

    @Test
    fun `synchronous scheduling failure releases its reservation`() {
        var releasedToken: String? = null
        var failed = false

        try {
            RadioRequestAdmissionCoordinator.admit(
                reserve = { "submission-a" },
                release = { releasedToken = it },
                schedule = { throw IllegalStateException("scheduler unavailable") },
            )
        } catch (expected: IllegalStateException) {
            failed = true
            assertEquals("scheduler unavailable", expected.message)
        }

        assertTrue(failed)
        assertEquals("submission-a", releasedToken)
    }
}
