package com.powerampstartradio.ui

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class PlanResponseComparatorTest {
    @Test
    fun `candidate-domain change is not confused with a queue change`() {
        val before = snapshot(
            runId = "run-a",
            reach = control("0.02", "2%"),
            candidateCount = 1_606,
            tracks = listOf(11, 22, 33),
        )
        val after = snapshot(
            runId = "run-b",
            reach = control("0.05", "5%"),
            candidateCount = 4_016,
            tracks = listOf(11, 22, 33),
        )

        val response = compared(before, after)

        assertEquals("reach", response.changedControl.id)
        assertEquals(2_410, response.candidates.delta)
        assertEquals(QueuePlanResponseKind.EXACTLY_UNCHANGED, response.queue.kind)
        assertTrue(response.queue.exactSetUnchanged)
        assertTrue(response.queue.exactOrderUnchanged)
        assertEquals(
            listOf(
                "Neighborhood reach changed from 2% to 5%.",
                "Tracks considered: 1,606 to 4,016.",
                "Queue unchanged: same 3 tracks in the same order.",
            ),
            response.messages,
        )
    }

    @Test
    fun `same generation-scoped set reports an order-only response`() {
        val response = compared(
            snapshot("run-a", control("0.4", "40%"), 1_606, listOf(11, 22, 33)),
            snapshot("run-b", control("0.6", "60%"), 1_606, listOf(22, 11, 33)),
        )

        assertEquals(QueuePlanResponseKind.SAME_SET_REORDERED, response.queue.kind)
        assertTrue(response.queue.exactSetUnchanged)
        assertFalse(response.queue.exactOrderUnchanged)
        assertEquals(1, response.queue.samePositionCount)
        assertEquals("Tracks considered: 1,606 (unchanged).", response.candidates.message)
        assertEquals(
            listOf(
                "Neighborhood reach changed from 40% to 60%.",
                "Queue reordered: same 3 tracks; 1 of 3 positions is unchanged.",
            ),
            response.messages,
        )
        assertEquals(
            "Queue reordered: same 3 tracks; 1 of 3 positions is unchanged.",
            response.queue.message,
        )
    }

    @Test
    fun `set response preserves exact removed and added plan order`() {
        val response = compared(
            snapshot("run-a", control("1", "1"), 1_606, listOf(11, 22, 33, 44)),
            snapshot("run-b", control("2", "2"), 1_606, listOf(11, 55, 44, 66)),
        )

        assertEquals(QueuePlanResponseKind.SET_CHANGED, response.queue.kind)
        assertEquals(2, response.queue.retainedTrackCount)
        assertEquals(listOf(22L, 33L), response.queue.removedTrackIds)
        assertEquals(listOf(55L, 66L), response.queue.addedTrackIds)
        assertEquals(1, response.queue.samePositionCount)
        assertEquals(
            "4-track queue changed: 2 retained, 2 replaced; " +
                "1 track stayed in the same position.",
            response.queue.message,
        )
    }

    @Test
    fun `queue size change is stated as prose`() {
        val response = compared(
            snapshot("run-a", control("1", "1"), 1_606, listOf(11, 22, 33)),
            snapshot("run-b", control("2", "2"), 1_606, listOf(11, 44, 55, 66)),
        )

        assertEquals(
            "Queue changed from 3 to 4 tracks; 1 retained, 2 removed, 3 added; " +
                "1 track stayed in the same position.",
            response.queue.message,
        )
    }

    @Test
    fun `comparison requires two distinct fresh planning runs`() {
        val before = snapshot("run-a", control("0.02", "2%"), 1_606, listOf(11))

        assertRejected(
            PlanComparisonRejection.PLAN_NOT_FRESH,
            before,
            before.copy(
                planningRunId = "run-b",
                materialization = PlanMaterialization.REUSED,
                semanticControls = listOf(control("0.05", "5%")),
            ),
        )
        assertRejected(
            PlanComparisonRejection.SAME_PLANNING_RUN,
            before,
            before.copy(semanticControls = listOf(control("0.05", "5%"))),
        )
    }

    @Test
    fun `generation provider seed and mode must each remain exact`() {
        val before = snapshot("run-a", control("0.02", "2%"), 1_606, listOf(11))
        val changed = before.copy(
            planningRunId = "run-b",
            semanticControls = listOf(control("0.05", "5%")),
        )

        assertRejected(
            PlanComparisonRejection.GENERATION_MISMATCH,
            before,
            changed.copy(generation = generation('b')),
        )
        assertRejected(
            PlanComparisonRejection.PROVIDER_GENERATION_MISMATCH,
            before,
            changed.copy(providerGenerationId = "provider-b"),
        )
        assertRejected(
            PlanComparisonRejection.SEED_MISMATCH,
            before,
            changed.copy(seedIdentity = RadioSeedIdentity(42L, "stable-b")),
        )
        assertRejected(
            PlanComparisonRejection.MODE_MISMATCH,
            before,
            changed.copy(selectionMode = SelectionMode.DPP),
        )
    }

    @Test
    fun `exactly one effective semantic control must change`() {
        val before = snapshot("run-a", control("0.02", "2%"), 1_606, listOf(11)).copy(
            semanticControls = listOf(
                control("0.02", "2%"),
                PlanSemanticControl("lambda", "0.4", "Relevance", "40%"),
            ),
        )

        assertRejected(
            PlanComparisonRejection.NO_SEMANTIC_CONTROL_CHANGE,
            before,
            before.copy(planningRunId = "run-b"),
        )
        assertRejected(
            PlanComparisonRejection.MULTIPLE_SEMANTIC_CONTROLS_CHANGED,
            before,
            before.copy(
                planningRunId = "run-b",
                semanticControls = listOf(
                    control("0.05", "5%"),
                    PlanSemanticControl("lambda", "0.6", "Relevance", "60%"),
                ),
            ),
        )
        assertRejected(
            PlanComparisonRejection.CONTROL_SET_MISMATCH,
            before,
            before.copy(
                planningRunId = "run-b",
                semanticControls = listOf(control("0.05", "5%")),
            ),
        )
    }

    @Test
    fun `snapshot refuses duplicate row occurrences and duplicate control identities`() {
        assertThrows(IllegalArgumentException::class.java) {
            snapshot("run-a", control("0.02", "2%"), 1_606, listOf(11, 11))
        }
        assertThrows(IllegalArgumentException::class.java) {
            snapshot("run-a", control("0.02", "2%"), 1_606, listOf(11)).copy(
                semanticControls = listOf(
                    control("0.02", "2%"),
                    control("0.05", "5%"),
                ),
            )
        }
    }

    private fun compared(before: PlanSnapshot, after: PlanSnapshot): PlanControlResponse {
        val result = PlanResponseComparator.compare(before, after)
        assertTrue("Expected comparison, got $result", result is PlanComparisonResult.Compared)
        return (result as PlanComparisonResult.Compared).response
    }

    private fun assertRejected(
        reason: PlanComparisonRejection,
        before: PlanSnapshot,
        after: PlanSnapshot,
    ) {
        val result = PlanResponseComparator.compare(before, after)
        assertTrue("Expected rejection, got $result", result is PlanComparisonResult.Rejected)
        assertEquals(reason, (result as PlanComparisonResult.Rejected).reason)
    }

    private fun snapshot(
        runId: String,
        reach: PlanSemanticControl,
        candidateCount: Int,
        tracks: List<Int>,
    ) = PlanSnapshot(
        planningRunId = runId,
        materialization = PlanMaterialization.FRESH,
        generation = generation('a'),
        providerGenerationId = "provider-a",
        seedIdentity = RadioSeedIdentity(41L, "stable-a"),
        selectionMode = SelectionMode.MMR,
        semanticControls = listOf(reach),
        candidateCount = candidateCount,
        orderedTrackIds = tracks.map(Int::toLong),
    )

    private fun control(valueKey: String, displayValue: String) = PlanSemanticControl(
        id = "reach",
        valueKey = valueKey,
        displayName = "Neighborhood reach",
        displayValue = displayValue,
    )

    private fun generation(hash: Char) = RadioGenerationToken(
        generationId = "generation-$hash",
        activationBindingId = "activation-$hash",
        manifestSha256 = hash.toString().repeat(64),
        embeddingSpecId = "clamp3-audio-v1",
        databaseContentSha256 = hash.toString().repeat(64),
        orderedTrackSetSha256 = hash.toString().repeat(64),
        stableTrackUidMappingSha256 = hash.toString().repeat(64),
    )
}
