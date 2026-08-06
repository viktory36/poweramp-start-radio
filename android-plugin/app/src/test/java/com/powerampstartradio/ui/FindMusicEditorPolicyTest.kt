package com.powerampstartradio.ui

import com.powerampstartradio.data.EmbeddedTrack
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class FindMusicEditorPolicyTest {
    @Test
    fun `weight controls expose one degree of freedom without inverse sliders`() {
        assertFalse(FindMusicEditorPolicy.usesSharedBalance(1))
        assertTrue(FindMusicEditorPolicy.usesSharedBalance(2))
        assertFalse(FindMusicEditorPolicy.usesPerIngredientWeightControls(2))
        assertTrue(FindMusicEditorPolicy.usesPerIngredientWeightControls(3))
    }

    @Test
    fun `Refine is offered only for two ingredients and falls back outside that shape`() {
        assertFalse(
            FindMusicEditorPolicy.shouldShowOperatorControl(
                activeIngredientCount = 1,
            ),
        )
        assertTrue(FindMusicEditorPolicy.shouldShowOperatorControl(activeIngredientCount = 2))
        assertFalse(FindMusicEditorPolicy.shouldShowOperatorControl(activeIngredientCount = 3))
        assertEquals(
            FindMusicOperator.REFINE,
            FindMusicEditorPolicy.effectiveOperator(
                requested = FindMusicOperator.REFINE,
                activeIngredientCount = 2,
            ),
        )
        assertEquals(
            FindMusicOperator.ALL_OF,
            FindMusicEditorPolicy.effectiveOperator(
                requested = FindMusicOperator.REFINE,
                activeIngredientCount = 3,
            ),
        )
        assertEquals(
            FindMusicOperator.ALL_OF,
            FindMusicEditorPolicy.effectiveOperator(
                requested = FindMusicOperator.ALL_OF,
                activeIngredientCount = 2,
            ),
        )
    }

    @Test
    fun `typed unconfirmed recording blocks a valid text request`() {
        val readiness = FindMusicEditorPolicy.readiness(
            textIngredients = listOf(text("slow ambient")),
            songSeeds = listOf(SongSeedState(query = "Bonobo Drift")),
            searchRunning = false,
        )

        assertFalse(readiness.canSearch)
        assertTrue(readiness.reason?.contains("exact recording") == true)
    }

    @Test
    fun `blank recording row is not silently promoted into an ingredient`() {
        val readiness = FindMusicEditorPolicy.readiness(
            textIngredients = listOf(text("sleep")),
            songSeeds = listOf(SongSeedState()),
            searchRunning = false,
        )

        assertTrue(readiness.canSearch)
        assertEquals(null, readiness.reason)
    }

    @Test
    fun `confirmed recording and positive description are searchable`() {
        assertTrue(
            FindMusicEditorPolicy.readiness(
                textIngredients = listOf(text("guitar")),
                songSeeds = listOf(song("Pink Floyd")),
                searchRunning = false,
            ).canSearch,
        )
    }

    @Test
    fun `completed zero-share ingredient blocks rather than disappearing from the request`() {
        val readiness = FindMusicEditorPolicy.readiness(
            textIngredients = listOf(
                text("ambient"),
                TextIngredientState(query = "guitar", weight = 0f),
            ),
            songSeeds = emptyList(),
            searchRunning = false,
        )

        assertFalse(readiness.canSearch)
        assertEquals(
            "Release a held share so every chosen ingredient can receive at least 1%.",
            readiness.reason,
        )
    }

    @Test
    fun `Refine requires exactly two ingredients and a Like primary`() {
        val valid = FindMusicEditorPolicy.readiness(
            textIngredients = listOf(
                TextIngredientState(query = "ambient", weight = 0.7f),
                TextIngredientState(query = "harsh", weight = 0.3f, negative = true),
            ),
            songSeeds = emptyList(),
            searchRunning = false,
            operator = FindMusicOperator.REFINE,
            refinePrimaryIngredientIndex = 0,
        )
        val wrongCount = FindMusicEditorPolicy.readiness(
            textIngredients = listOf(text("ambient")),
            songSeeds = emptyList(),
            searchRunning = false,
            operator = FindMusicOperator.REFINE,
        )
        val avoidPrimary = FindMusicEditorPolicy.readiness(
            textIngredients = listOf(
                TextIngredientState(query = "ambient", weight = 0.7f),
                TextIngredientState(query = "harsh", weight = 0.3f, negative = true),
            ),
            songSeeds = emptyList(),
            searchRunning = false,
            operator = FindMusicOperator.REFINE,
            refinePrimaryIngredientIndex = 1,
        )

        assertTrue(valid.canSearch)
        assertFalse(wrongCount.canSearch)
        assertTrue(wrongCount.reason?.contains("exactly two") == true)
        assertFalse(avoidPrimary.canSearch)
        assertTrue(avoidPrimary.reason?.contains("Like ingredient") == true)
    }

    @Test
    fun `empty avoid-only and running requests are disabled with distinct reasons`() {
        val empty = FindMusicEditorPolicy.readiness(
            textIngredients = listOf(TextIngredientState()),
            songSeeds = emptyList(),
            searchRunning = false,
        )
        val avoidOnly = FindMusicEditorPolicy.readiness(
            textIngredients = listOf(text("harsh", negative = true)),
            songSeeds = emptyList(),
            searchRunning = false,
        )
        val running = FindMusicEditorPolicy.readiness(
            textIngredients = listOf(text("sleep")),
            songSeeds = emptyList(),
            searchRunning = true,
        )

        assertFalse(empty.canSearch)
        assertTrue(empty.reason?.contains("Add") == true)
        assertFalse(avoidOnly.canSearch)
        assertTrue(avoidOnly.reason?.contains("Like") == true)
        assertFalse(running.canSearch)
        assertTrue(running.reason?.contains("current") == true)
    }

    @Test
    fun `last positive text or recording cannot become Avoid`() {
        val onlyText = listOf(text("sleep"))
        assertFalse(FindMusicEditorPolicy.canSetTextToAvoid(onlyText, emptyList(), 0))

        val onlySong = listOf(song("Drift"))
        assertFalse(FindMusicEditorPolicy.canSetSongToAvoid(emptyList(), onlySong, 0))

        val twoPositive = listOf(text("sleep"), text("ambient"))
        assertTrue(FindMusicEditorPolicy.canSetTextToAvoid(twoPositive, emptyList(), 0))

        val mixed = listOf(text("sleep"))
        assertTrue(FindMusicEditorPolicy.canSetSongToAvoid(mixed, onlySong, 0))
    }

    @Test
    fun `sign selector appears only when it offers a real choice`() {
        assertFalse(
            FindMusicEditorPolicy.shouldShowSignControl(
                negative = false,
                canSetAvoid = false,
                operator = FindMusicOperator.ALL_OF,
            ),
        )
        assertFalse(
            FindMusicEditorPolicy.shouldShowSignControl(
                negative = false,
                canSetAvoid = false,
                operator = FindMusicOperator.REFINE,
            ),
        )
        assertTrue(
            FindMusicEditorPolicy.shouldShowSignControl(
                negative = false,
                canSetAvoid = true,
                operator = FindMusicOperator.ALL_OF,
            ),
        )
        assertTrue(
            FindMusicEditorPolicy.shouldShowSignControl(
                negative = false,
                canSetAvoid = true,
                operator = FindMusicOperator.REFINE,
            ),
        )
        assertTrue(
            FindMusicEditorPolicy.shouldShowSignControl(
                negative = true,
                canSetAvoid = false,
                operator = FindMusicOperator.REFINE,
            ),
        )
    }

    @Test
    fun `reset snapshot clears every live Find Music control`() {
        val defaults = FindMusicEditorPolicy.reset(
            FindMusicEditorSnapshot(
                textIngredients = listOf(text("busy drums", negative = true)),
                songSeeds = listOf(song("Echoes")),
                operator = FindMusicOperator.REFINE,
                resultLimit = 95,
            ),
        )

        assertEquals(1, defaults.textIngredients.size)
        assertEquals("", defaults.textIngredients.single().query)
        assertEquals(0f, defaults.textIngredients.single().weight)
        assertTrue(defaults.songSeeds.isEmpty())
        assertEquals(FindMusicOperator.ALL_OF, defaults.operator)
        assertEquals(FindMusicQuerySpec.DEFAULT_RESULT_LIMIT, defaults.resultLimit)
    }

    private fun text(query: String, negative: Boolean = false) = TextIngredientState(
        query = query,
        weight = 1f,
        negative = negative,
    )

    private fun song(query: String) = SongSeedState(
        query = query,
        confirmedTrack = track(),
        weight = 1f,
    )

    private fun track() = EmbeddedTrack(
        id = 7L,
        metadataKey = "metadata",
        filenameKey = "file",
        artist = "Artist",
        album = "Album",
        title = "Title",
        durationMs = 245_000,
        filePath = "/music/title.flac",
    )
}
