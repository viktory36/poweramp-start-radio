package com.powerampstartradio.ui

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class IndexAddedAgeLabelTest {
    @Test
    fun `age uses whole elapsed days and hides unknown dates`() {
        val now = 2_000_000_000L
        assertNull(indexAddedAgeLabel(0L, now))
        assertEquals("added today", indexAddedAgeLabel(now, now))
        assertEquals("added today", indexAddedAgeLabel(now - 86_399L, now))
        assertEquals("added 1d ago", indexAddedAgeLabel(now - 86_400L, now))
        assertEquals("added 8d ago", indexAddedAgeLabel(now - 8L * 86_400L, now))
        assertEquals("added today", indexAddedAgeLabel(now + 60L, now))
    }
}
