package com.powerampstartradio.indexing

import org.junit.Assert.assertEquals
import org.junit.Test

class V2GraphWorkPlannerTest {
    @Test
    fun `normal growth plans every exact incremental dot product and byte transfer`() {
        val plan = V2GraphWorkPlanner.plan(
            targetNodes = 80_500,
            embeddingDimension = 768,
            neighborsPerNode = 5,
            existingGraphNodes = 80_421,
            existingGraphNeighborsPerNode = 5,
            existingGraphByteLength = 3_860_216L,
        )

        assertEquals(V2GraphUpdateStrategy.INCREMENTAL, plan.strategy)
        assertEquals(79, plan.newNodes)
        assertEquals(80_500L, plan.embeddingRows)
        assertEquals(6_761_605L, plan.similarityDotProducts)
        assertEquals(3_860_216L, plan.graphBinaryInputBytes)
        assertEquals(7_728_016L, plan.graphBinaryOutputBytes)
    }

    @Test
    fun `matching graph is extracted and reused without similarity or output work`() {
        val bytes = V2GraphWorkPlanner.graphFileBytes(80_421, 5)
        val plan = V2GraphWorkPlanner.plan(
            targetNodes = 80_421,
            embeddingDimension = 768,
            existingGraphNodes = 80_421,
            existingGraphNeighborsPerNode = 5,
            existingGraphByteLength = bytes,
        )

        assertEquals(V2GraphUpdateStrategy.REUSE, plan.strategy)
        assertEquals(0L, plan.similarityDotProducts)
        assertEquals(bytes, plan.graphBinaryBytes)
        assertEquals(0L, plan.graphBinaryOutputBytes)
    }

    @Test
    fun `missing or incompatible graph has an explicit quadratic full rebuild`() {
        val missing = V2GraphWorkPlanner.plan(
            targetNodes = 1_000,
            embeddingDimension = 768,
        )
        assertEquals(V2GraphUpdateStrategy.FULL_REBUILD, missing.strategy)
        assertEquals(999_000L, missing.similarityDotProducts)

        val wrongK = V2GraphWorkPlanner.plan(
            targetNodes = 1_010,
            embeddingDimension = 768,
            neighborsPerNode = 5,
            existingGraphNodes = 1_000,
            existingGraphNeighborsPerNode = 10,
            existingGraphByteLength = V2GraphWorkPlanner.graphFileBytes(1_000, 10),
        )
        assertEquals(V2GraphUpdateStrategy.FULL_REBUILD, wrongK.strategy)
        assertEquals(1_019_090L, wrongK.similarityDotProducts)
        assertEquals(V2GraphWorkPlanner.graphFileBytes(1_000, 10), wrongK.graphBinaryInputBytes)
    }

    @Test
    fun `runtime can bind an exact non-default count of old neighbor comparisons`() {
        val plan = V2GraphWorkPlanner.plan(
            targetNodes = 12,
            embeddingDimension = 3,
            neighborsPerNode = 2,
            existingGraphNodes = 10,
            existingGraphNeighborsPerNode = 2,
            validOldNeighborDotProducts = 17L,
        )

        assertEquals(41L, plan.similarityDotProducts)
    }

    @Test
    fun `mixed deletion and append plans only affected rescans plus one scan per new row`() {
        val plan = V2GraphWorkPlanner.plan(
            targetNodes = 80_463,
            embeddingDimension = 768,
            neighborsPerNode = 5,
            existingGraphNodes = 80_453,
            existingGraphNeighborsPerNode = 5,
            removedBaseNodes = 7,
            rescannedBaseNodes = 35,
            validOldNeighborDotProducts = (80_446L - 35L) * 5L,
        )

        assertEquals(V2GraphUpdateStrategy.INCREMENTAL, plan.strategy)
        assertEquals(17, plan.newNodes)
        assertEquals(7, plan.removedBaseNodes)
        assertEquals(35, plan.rescannedBaseNodes)
        assertEquals(80_446, plan.retainedBaseNodes)
        assertEquals(
            17L * 80_463L + 35L * 80_462L + (80_446L - 35L) * 5L,
            plan.similarityDotProducts,
        )
    }
}
