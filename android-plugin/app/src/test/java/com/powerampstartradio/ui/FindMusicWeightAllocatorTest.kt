package com.powerampstartradio.ui

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

class FindMusicWeightAllocatorTest {
    @Test
    fun `sequential drags keep one exact bounded budget`() {
        var slots = listOf(
            slot(0.34f, minimum = 0f),
            slot(0.33f),
            slot(0.33f),
        )

        repeat(20) { step ->
            val changed = step % slots.size
            val weights = FindMusicWeightAllocator.adjust(
                slots,
                changedIndex = changed,
                requestedWeight = if (step % 2 == 0) 0.93f else 0.017f,
            )
            slots = slots.mapIndexed { index, slot -> slot.copy(weight = weights[index]) }
            assertValid(slots)
        }
    }

    @Test
    fun `locks add and remove preserve fixed shares and residual`() {
        var slots = listOf(
            slot(0.5f, minimum = 0f),
            slot(0.3f, locked = true),
            slot(0.2f),
        )

        var weights = FindMusicWeightAllocator.adjust(slots, 0, 0.68f)
        slots = slots.mapIndexed { index, slot -> slot.copy(weight = weights[index]) }
        assertEquals(0.3f, slots[1].weight, 0f)
        assertValid(slots)

        slots = slots + slot(0.1f)
        weights = FindMusicWeightAllocator.normalize(slots)
        slots = slots.mapIndexed { index, slot -> slot.copy(weight = weights[index]) }
        assertEquals(0.3f, slots[1].weight, 0f)
        assertValid(slots)

        slots = slots.filterIndexed { index, _ -> index != 2 }
        weights = FindMusicWeightAllocator.normalize(slots)
        slots = slots.mapIndexed { index, slot -> slot.copy(weight = weights[index]) }
        assertEquals(0.3f, slots[1].weight, 0f)
        assertValid(slots)
    }

    @Test
    fun `independent floors never create an overfull simplex`() {
        val weights = FindMusicWeightAllocator.normalize(
            listOf(
                slot(0.98f, minimum = 0f),
                slot(0.01f),
                slot(0.01f),
                slot(0.01f),
            ),
        )

        assertTrue(weights.drop(1).all { it >= 0.01f })
        assertEquals(1f, weights.sum(), 1e-6f)
    }

    @Test
    fun `incomplete rows stay zero and do not dilute the first completed ingredient`() {
        val weights = FindMusicEditorWeightPolicy.activate(
            slots = listOf(
                editorSlot(weight = 0f, completed = true),
                editorSlot(weight = 0f, completed = false),
                editorSlot(weight = 0f, completed = false),
            ),
            activatedIndex = 0,
        )

        assertArrayEquals(floatArrayOf(1f, 0f, 0f), weights)
    }

    @Test
    fun `completing a second ingredient allocates one deterministic half share`() {
        val weights = FindMusicEditorWeightPolicy.activate(
            slots = listOf(
                editorSlot(weight = 1f, completed = true),
                editorSlot(weight = 0f, completed = true),
                editorSlot(weight = 0f, completed = false),
            ),
            activatedIndex = 1,
        )

        assertArrayEquals(floatArrayOf(0.5f, 0.5f, 0f), weights)
    }

    @Test
    fun `deactivating an ingredient restores the lone completed ingredient to full share`() {
        val weights = FindMusicEditorWeightPolicy.normalize(
            listOf(
                editorSlot(weight = 0.5f, completed = true, locked = true),
                editorSlot(weight = 0f, completed = false),
            ),
        )

        assertArrayEquals(floatArrayOf(1f, 0f), weights)
    }

    @Test
    fun `activation never rescales locked shares when they consume the budget`() {
        val weights = FindMusicEditorWeightPolicy.activate(
            slots = listOf(
                editorSlot(weight = 0.6f, completed = true, locked = true),
                editorSlot(weight = 0.4f, completed = true, locked = true),
                editorSlot(weight = 0f, completed = true),
            ),
            activatedIndex = 2,
        )

        assertArrayEquals(floatArrayOf(0.6f, 0.4f, 0f), weights)
    }

    @Test
    fun `activation uses exact residual without changing feasible locks`() {
        val weights = FindMusicEditorWeightPolicy.activate(
            slots = listOf(
                editorSlot(weight = 0.6f, completed = true, locked = true),
                editorSlot(weight = 0.39f, completed = true, locked = true),
                editorSlot(weight = 0f, completed = true),
            ),
            activatedIndex = 2,
        )

        assertArrayEquals(floatArrayOf(0.6f, 0.39f, 0.01f), weights)
    }

    @Test
    fun `only shares with a real redistribution degree of freedom are adjustable`() {
        val oneFreeShare = listOf(
            editorSlot(weight = 0.4f, completed = true, locked = true),
            editorSlot(weight = 0.3f, completed = true, locked = true),
            editorSlot(weight = 0.3f, completed = true),
        )
        assertFalse(FindMusicEditorWeightPolicy.canAdjust(oneFreeShare, 0))
        assertFalse(FindMusicEditorWeightPolicy.canAdjust(oneFreeShare, 2))

        val twoFreeShares = listOf(
            editorSlot(weight = 0.4f, completed = true, locked = true),
            editorSlot(weight = 0.3f, completed = true),
            editorSlot(weight = 0.3f, completed = true),
        )
        assertTrue(FindMusicEditorWeightPolicy.canAdjust(twoFreeShares, 1))
        assertTrue(FindMusicEditorWeightPolicy.canAdjust(twoFreeShares, 2))

        val allFreeBudgetAtFloor = listOf(
            editorSlot(weight = 0.98f, completed = true, locked = true),
            editorSlot(weight = 0.01f, completed = true),
            editorSlot(weight = 0.01f, completed = true),
        )
        assertFalse(FindMusicEditorWeightPolicy.canAdjust(allFreeBudgetAtFloor, 1))
        assertFalse(FindMusicEditorWeightPolicy.canAdjust(allFreeBudgetAtFloor, 2))
    }

    @Test
    fun `All-of and Refine keep the same one-percent ingredient floor`() {
        val allOfMinimum = FindMusicEditorWeightPolicy.minimumActiveWeight(
            operator = FindMusicOperator.ALL_OF,
            resultLimit = 10,
        )
        val refineMinimum = FindMusicEditorWeightPolicy.minimumActiveWeight(
            operator = FindMusicOperator.REFINE,
            resultLimit = 1_000,
        )
        val weights = FindMusicEditorWeightPolicy.normalize(
            slots = listOf(
                editorSlot(weight = 0.98f, completed = true),
                editorSlot(weight = 0.01f, completed = true),
                editorSlot(weight = 0.01f, completed = true),
            ),
            minimumActiveWeight = refineMinimum,
        )

        assertEquals(0.01f, allOfMinimum, 0f)
        assertEquals(allOfMinimum, refineMinimum, 0f)
        assertTrue(weights.all { it >= 0.01f })
        assertEquals(1f, weights.sum(), 1e-6f)
    }

    private fun slot(
        weight: Float,
        minimum: Float = 0.01f,
        locked: Boolean = false,
    ) = FindMusicWeightSlot(weight, minimum, locked)

    private fun editorSlot(
        weight: Float,
        completed: Boolean,
        locked: Boolean = false,
    ) = FindMusicEditorWeightSlot(weight, locked, completed)

    private fun assertArrayEquals(expected: FloatArray, actual: FloatArray) {
        assertEquals(expected.size, actual.size)
        expected.indices.forEach { index ->
            assertEquals("weight[$index]", expected[index], actual[index], 1e-6f)
        }
    }

    private fun assertValid(slots: List<FindMusicWeightSlot>) {
        assertTrue(abs(slots.sumOf { it.weight.toDouble() } - 1.0) <= 1e-6)
        assertTrue(slots.all { it.weight + 1e-7f >= it.minimum })
    }
}
