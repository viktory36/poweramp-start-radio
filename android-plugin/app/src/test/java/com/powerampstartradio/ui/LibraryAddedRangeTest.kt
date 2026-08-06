package com.powerampstartradio.ui

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertThrows
import org.junit.Test

class LibraryAddedRangeTest {
    @Test
    fun `rolling ranges resolve exact request-scoped cutoffs`() {
        val reference = 2_000_000_000L

        assertEquals(
            reference - 7L * 86_400L,
            LibraryAddedRange.LAST_7_DAYS.minimumCreatedAtEpochSecond(reference),
        )
        assertEquals(
            reference - 30L * 86_400L,
            LibraryAddedRange.LAST_30_DAYS.minimumCreatedAtEpochSecond(reference),
        )
        assertEquals(
            reference - 365L * 86_400L,
            LibraryAddedRange.LAST_365_DAYS.minimumCreatedAtEpochSecond(reference),
        )
        assertNull(LibraryAddedRange.ALL_DATES.minimumCreatedAtEpochSecond(reference))
    }

    @Test
    fun `exact rolling days resolve labels and request-scoped cutoffs`() {
        val reference = 2_000_000_000L

        assertEquals("All dates", libraryAddedDaysLabel(null))
        assertEquals("Last 1 day", libraryAddedDaysLabel(1))
        assertEquals("Last 17 days", libraryAddedDaysLabel(17))
        assertNull(minimumLibraryAddedAtEpochSecond(null, reference))
        assertEquals(
            reference - 17L * 86_400L,
            minimumLibraryAddedAtEpochSecond(17, reference),
        )
    }

    @Test
    fun `large finite range saturates above unknown first-seen sentinel`() {
        assertEquals(
            1L,
            minimumLibraryAddedAtEpochSecond(MAX_LIBRARY_ADDED_DAYS, 2_000_000_000L),
        )
    }

    @Test
    fun `invalid exact day counts fail instead of changing meaning`() {
        assertThrows(IllegalArgumentException::class.java) {
            minimumLibraryAddedAtEpochSecond(0, 2_000_000_000L)
        }
        assertThrows(IllegalArgumentException::class.java) {
            libraryAddedDaysLabel(MAX_LIBRARY_ADDED_DAYS + 1)
        }
    }

    @Test
    fun `effective days preserve legacy preset records and prefer exact values`() {
        assertEquals(
            30,
            RadioConfig(libraryAddedRange = LibraryAddedRange.LAST_30_DAYS)
                .effectiveLibraryAddedDays,
        )
        assertEquals(
            17,
            RadioConfig(
                libraryAddedRange = LibraryAddedRange.LAST_30_DAYS,
                libraryAddedDays = 17,
            ).effectiveLibraryAddedDays,
        )
        assertNull(RadioConfig().effectiveLibraryAddedDays)
    }
}
