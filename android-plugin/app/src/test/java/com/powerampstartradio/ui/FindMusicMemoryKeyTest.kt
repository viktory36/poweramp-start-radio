package com.powerampstartradio.ui

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class FindMusicMemoryKeyTest {
    private val query = FindMusicQuerySpec(textIngredients = listOf(
        FindMusicTextIngredient("slow", .4f, false),
        FindMusicTextIngredient("psychedelic", .6f, false),
    ))

    @Test
    fun executionControlsAndIngredientOrderDoNotResetMemory() {
        val other = query.copy(
            textIngredients = query.textIngredients.reversed(),
            resultLimit = 70,
            libraryAddedDays = 14,
            textResultPlanner = FindMusicTextResultPlanner.VARIED_ALL_OF_DPP,
            rememberPrevious = true,
        )
        assertEquals(FindMusicMemoryKey.of(query), FindMusicMemoryKey.of(other))
        assertNotEquals(query.stateKey, other.stateKey)
    }

    @Test
    fun IngredientsDirectionsAndWeightsDefineDifferentRequests() {
        for (ingredient in listOf(
            query.textIngredients[0].copy(query = "guitar"),
            query.textIngredients[0].copy(negative = true),
            query.textIngredients[0].copy(weight = .5f),
        )) {
            assertNotEquals(
                FindMusicMemoryKey.of(query),
                FindMusicMemoryKey.of(query.copy(textIngredients = listOf(ingredient, query.textIngredients[1]))),
            )
        }
    }

    @Test
    fun refineKeepsRolesButIgnoresWidthAndUnusedWeights() {
        val refine = query.copy(
            operator = FindMusicOperator.REFINE,
            refineSpec = FindMusicRefineSpec(0, FindMusicRefineNeighborhood.TOP_1_PERCENT),
        )
        assertEquals(FindMusicMemoryKey.of(refine), FindMusicMemoryKey.of(refine.copy(
            textIngredients = refine.textIngredients.map { it.copy(weight = .5f) },
            refineSpec = FindMusicRefineSpec(0, FindMusicRefineNeighborhood.TOP_2_PERCENT),
        )))
        assertEquals(FindMusicMemoryKey.of(refine), FindMusicMemoryKey.of(refine.copy(
            textIngredients = refine.textIngredients.reversed(),
            refineSpec = FindMusicRefineSpec(1, FindMusicRefineNeighborhood.TOP_1_PERCENT),
        )))
        assertNotEquals(FindMusicMemoryKey.of(refine), FindMusicMemoryKey.of(refine.copy(
            refineSpec = FindMusicRefineSpec(1, FindMusicRefineNeighborhood.TOP_1_PERCENT),
        )))
    }

    @Test
    fun savedAndCurrentControlsCaptureMemoryWithoutChangingRecipe() {
        val remembered = query.copy(rememberPrevious = true)
        assertEquals(remembered, FindMusicQuerySpecCodec.fromJsonArray(
            FindMusicQuerySpecCodec.toJsonArray(listOf(remembered)),
        ).single())
        val updated = query.withCurrentExecutionControls(
            30, null, FindMusicTextResultPlanner.CLOSEST, FindMusicRefineNeighborhood.DEFAULT,
            rememberPrevious = true,
        )
        assertTrue(updated.rememberPrevious)
        assertNotEquals(query.hasSameExecutionControlsAs(updated), true)
        assertEquals(FindMusicMemoryKey.of(query), FindMusicMemoryKey.of(updated))
    }
}
