package com.powerampstartradio.similarity

import com.powerampstartradio.data.StableVisibleResultIdentity
import org.junit.Assert.assertEquals
import org.junit.Test

class StableVisibleResultReducerTest {
    @Test
    fun `verified duplicate collapses and ranked scan refills visible count`() {
        val rows = listOf(
            Row(1, stable("a")),
            Row(2, stable("a")),
            Row(3, stable("b")),
            Row(4, stable("c")),
        )

        val result = StableVisibleResultReducer.reduce(
            rankedItems = rows,
            requestedVisibleCount = 3,
            identityOf = Row::identity,
        )

        assertEquals(listOf(1, 3, 4), result.items.map(Row::id))
        assertEquals(4, result.scannedRowCount)
        assertEquals(1, result.collapsedEquivalentCount)
    }

    @Test
    fun `legacy rows never collapse even if a caller repeats a token`() {
        val rows = listOf(
            Row(1, legacy("same-token")),
            Row(2, legacy("same-token")),
        )

        val result = StableVisibleResultReducer.reduce(
            rankedItems = rows,
            requestedVisibleCount = 2,
            identityOf = Row::identity,
        )

        assertEquals(listOf(1, 2), result.items.map(Row::id))
        assertEquals(0, result.collapsedEquivalentCount)
    }

    @Test
    fun `rejected occurrence does not consume its verified identity`() {
        val rows = listOf(
            Row(1, stable("a")),
            Row(2, stable("a")),
            Row(3, stable("b")),
        )

        val result = StableVisibleResultReducer.reduce(
            rankedItems = rows,
            requestedVisibleCount = 2,
            identityOf = Row::identity,
            isEligible = { it.id != 1 },
        )

        assertEquals(listOf(2, 3), result.items.map(Row::id))
        assertEquals(3, result.scannedRowCount)
        assertEquals(0, result.collapsedEquivalentCount)
    }

    private data class Row(val id: Int, val identity: StableVisibleResultIdentity)

    private fun stable(token: String) = StableVisibleResultIdentity(token, true)
    private fun legacy(token: String) = StableVisibleResultIdentity(token, false)
}
