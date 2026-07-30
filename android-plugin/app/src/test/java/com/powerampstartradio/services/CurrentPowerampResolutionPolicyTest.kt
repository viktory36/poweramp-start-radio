package com.powerampstartradio.services

import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

class CurrentPowerampResolutionPolicyTest {
    @Test
    fun unchangedCurrentResolutionIsAcceptedOccurrenceForOccurrence() {
        val current = listOf(701L, 701L, null, 900L)

        assertEquals(
            current,
            CurrentPowerampResolutionPolicy.requireUnchanged(current, current),
        )
    }

    @Test
    fun reusedOrNowAmbiguousPowerampIdIsRejectedBeforeQueueMutation() {
        assertThrows(IllegalArgumentException::class.java) {
            CurrentPowerampResolutionPolicy.requireUnchanged(
                pinnedFileIds = listOf(700L),
                currentFileIds = listOf(701L),
            )
        }
        assertThrows(IllegalArgumentException::class.java) {
            CurrentPowerampResolutionPolicy.requireUnchanged(
                pinnedFileIds = listOf(700L),
                currentFileIds = listOf(null),
            )
        }
    }
}
