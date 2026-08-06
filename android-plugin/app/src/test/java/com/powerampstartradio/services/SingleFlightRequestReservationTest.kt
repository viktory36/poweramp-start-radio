package com.powerampstartradio.services

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class SingleFlightRequestReservationTest {
    @Test
    fun `one acquisition excludes later submissions until release`() {
        val reservation = SingleFlightRequestReservation()

        assertTrue(reservation.tryAcquire("submission-a"))
        assertTrue(reservation.isReserved)
        assertFalse(reservation.tryAcquire("submission-b"))

        assertTrue(reservation.failSubmission("submission-a"))
        assertFalse(reservation.isReserved)
        assertTrue(reservation.tryAcquire("submission-b"))
    }

    @Test
    fun `only the bound request dispatch can complete the reservation`() {
        val reservation = SingleFlightRequestReservation()
        assertTrue(reservation.tryAcquire("submission-a"))
        assertTrue(reservation.bindRequest("submission-a", "request-a"))

        assertFalse(reservation.completeDispatch("request-b"))
        assertTrue(reservation.isReserved)
        assertFalse(reservation.tryAcquire("submission-b"))

        assertTrue(reservation.completeDispatch("request-a"))
        assertFalse(reservation.isReserved)
    }

    @Test
    fun `mismatched binding and failures cannot release another owner`() {
        val reservation = SingleFlightRequestReservation()
        assertTrue(reservation.tryAcquire("submission-a"))

        assertFalse(reservation.bindRequest("submission-b", "request-b"))
        assertFalse(reservation.failSubmission("submission-b"))
        assertTrue(reservation.isReserved)

        assertTrue(reservation.bindRequest("submission-a", "request-a"))
        assertFalse(reservation.failRequest("request-b"))
        assertTrue(reservation.isReserved)
        assertTrue(reservation.failRequest("request-a"))
        assertFalse(reservation.isReserved)
    }

    @Test
    fun `cold widget ingress can reserve its final request ID before persistence`() {
        val reservation = SingleFlightRequestReservation()

        assertTrue(reservation.tryAcquire("submission-a", "request-a"))
        assertTrue(reservation.isReserved)
        assertTrue(reservation.activeRequestId == "request-a")
        assertTrue(reservation.bindRequest("submission-a", "request-a"))
        assertFalse(reservation.bindRequest("submission-a", "request-b"))
        assertTrue(reservation.completeDispatch("request-a"))
    }
}
