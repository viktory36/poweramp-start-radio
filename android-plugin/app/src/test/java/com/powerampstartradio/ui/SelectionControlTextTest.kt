package com.powerampstartradio.ui

import org.junit.Assert.assertEquals
import org.junit.Test

class SelectionControlTextTest {
    @Test
    fun `mode labels are compact and retain exact algorithm names`() {
        assertEquals("Closest", SelectionControlText.modeLabel(SelectionMode.CLOSEST))
        assertEquals("MMR", SelectionControlText.modeLabel(SelectionMode.MMR))
        assertEquals("DPP", SelectionControlText.modeLabel(SelectionMode.DPP))
        assertEquals("Graph Explorer", SelectionControlText.modeLabel(SelectionMode.RANDOM_WALK))
        assertEquals(
            "Uniform shuffle",
            SelectionControlText.modeLabel(SelectionMode.UNIFORM_SHUFFLE),
        )
    }

    @Test
    fun `mode differentiators make each compact choice meaningful`() {
        assertEquals(
            "Direct cosine similarity to the seed",
            SelectionControlText.modeDifferentiator(SelectionMode.CLOSEST),
        )
        assertEquals(
            "Relevance minus resemblance to earlier picks",
            SelectionControlText.modeDifferentiator(SelectionMode.MMR),
        )
        assertEquals(
            "Greedy set-wide quality and diversity",
            SelectionControlText.modeDifferentiator(SelectionMode.DPP),
        )
        assertEquals(
            "Terminal probability across similarity-graph paths",
            SelectionControlText.modeDifferentiator(SelectionMode.RANDOM_WALK),
        )
        assertEquals(
            "Uniform order without similarity ranking",
            SelectionControlText.modeDifferentiator(SelectionMode.UNIFORM_SHUFFLE),
        )
    }

    @Test
    fun `MMR balance names both sides without exposing a score formula`() {
        assertEquals(
            "Seed relevance 60% \u00b7 variety 40%",
            SelectionControlText.mmrBalanceTitle(0.6f, driftEnabled = false),
        )
        assertEquals(
            "Current-direction relevance 80% \u00b7 variety 20%",
            SelectionControlText.mmrBalanceTitle(0.8f, driftEnabled = true),
        )
        assertEquals(
            "Nearest first \u00b7 then variety only",
            SelectionControlText.mmrBalanceTitle(0f, driftEnabled = false),
        )
    }

    @Test
    fun `mode descriptions remain true at zero and maximum control endpoints`() {
        assertEquals(
            "Ranks available distinct recordings by cosine similarity to the seed.",
            SelectionControlText.modeDescription(SelectionMode.CLOSEST),
        )
        assertEquals(
            "At each pick, subtracts the weighted maximum similarity to any earlier pick from " +
                "weighted seed relevance.",
            SelectionControlText.modeDescription(SelectionMode.MMR),
        )
        assertEquals(
            "Greedily maximizes determinant gain over a quality-weighted cosine-similarity " +
                "kernel, rewarding strong, internally diverse sets.",
            SelectionControlText.modeDescription(SelectionMode.DPP),
        )
        assertEquals(
            "Ranks terminal probabilities from deterministic uniform non-backtracking walks " +
                "over the similarity graph.",
            SelectionControlText.modeDescription(SelectionMode.RANDOM_WALK),
        )
        assertEquals(
            "Creates a reproducible uniform permutation of the other available distinct " +
                "recordings.",
            SelectionControlText.modeDescription(SelectionMode.UNIFORM_SHUFFLE),
        )
    }

    @Test
    fun `DPP seed pull uses directional labels without exposing its exponent`() {
        val expected = listOf(
            0f to "Nearest first, then set variety",
            0.25f to "Very light seed pull",
            0.5f to "Light seed pull",
            1f to "Standard seed pull",
            2f to "Strong seed pull",
            3f to "Very strong seed pull",
            4f to "Strongest seed pull",
        )

        expected.forEach { (value, text) ->
            assertEquals(text, SelectionControlText.dppSeedPullLabel(value))
        }
    }

    @Test
    fun `drift modes describe the audible direction and retain momentum name`() {
        assertEquals(
            "Seed + last pick",
            SelectionControlText.driftModeLabel(DriftMode.SEED_INTERPOLATION),
        )
        assertEquals(
            "Rolling direction (momentum)",
            SelectionControlText.driftModeLabel(DriftMode.MOMENTUM),
        )
    }
}
