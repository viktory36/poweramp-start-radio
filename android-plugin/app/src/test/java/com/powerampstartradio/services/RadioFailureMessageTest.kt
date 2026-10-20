package com.powerampstartradio.services

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class RadioFailureMessageTest {
    @Test
    fun missingSeedExplainsIndexingOrMerge() {
        val message = RadioFailureMessage.from(RadioSeedNotIndexedException())
        assertTrue(message.contains("No indexed match"))
        assertTrue(message.contains("On-device indexing"))
        assertTrue(message.contains("server index"))
    }

    @Test
    fun unrelatedWidgetFailuresKeepTheirExistingMessages() {
        assertEquals(
            "Poweramp access is unavailable. Open Start Radio and grant access.",
            RadioFailureMessage.from(SecurityException("Permission denied")),
        )
        assertEquals(
            "Radio could not start. Open Start Radio, then try again.",
            RadioFailureMessage.from(IllegalStateException("Unexpected failure")),
        )
    }
}
