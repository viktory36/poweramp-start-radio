package com.powerampstartradio.indexing

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class V2GraphTopKOrderingTest {
    @Test
    fun `incremental equal-score fixture exactly matches full rebuild ordering`() {
        // Index order deliberately differs from track-ID order.
        val ids = longArrayOf(30L, 20L, 10L, 40L)
        val neighbors = intArrayOf(1, 0) // IDs 20, 30
        val scores = floatArrayOf(0.8f, 0.5f)

        assertTrue(
            V2GraphTopKOrdering.tryInsert(
                neighbors,
                scores,
                candidateIndex = 2,
                candidateScore = 0.5f,
                trackIdsByIndex = ids,
                k = 2,
            ),
        )
        V2GraphTopKOrdering.sortBestFirst(neighbors, scores, ids, 2)

        val incremental = neighbors.map { ids[it] }.toLongArray()
        val full = listOf(
            1 to 0.8f,
            0 to 0.5f,
            2 to 0.5f,
        ).sortedWith(
            compareByDescending<Pair<Int, Float>> { it.second }
                .thenBy { ids[it.first] },
        ).take(2).map { ids[it.first] }.toLongArray()
        assertArrayEquals(full, incremental)
    }

    @Test
    fun `negative and NaN candidates use native top-k ordering before clamping`() {
        val ids = longArrayOf(40L, 30L, 20L, 10L)
        val neighbors = intArrayOf(0, 1)
        val scores = floatArrayOf(-0.3f, Float.NaN)

        assertTrue(V2GraphTopKOrdering.tryInsert(neighbors, scores, 2, -0.4f, ids, 2))
        assertTrue(V2GraphTopKOrdering.tryInsert(neighbors, scores, 3, -0.3f, ids, 2))
        V2GraphTopKOrdering.sortBestFirst(neighbors, scores, ids, 2)

        assertArrayEquals(longArrayOf(10L, 40L), neighbors.map { ids[it] }.toLongArray())
    }

    @Test
    fun `nonpositive cosine row keeps only real nearest slots uniformly traversable`() {
        val scores = floatArrayOf(-0.2f, 0f, -0.8f, 99f)

        V2GraphWeightPolicy.normalizeNonnegativeInPlace(scores, 3)

        assertArrayEquals(floatArrayOf(1f / 3f, 1f / 3f, 1f / 3f, 99f), scores, 0f)
    }

    @Test
    fun `positive cosine row is clamped and normalized without changing rank`() {
        val scores = floatArrayOf(0.6f, 0.3f, -0.1f)

        V2GraphWeightPolicy.normalizeNonnegativeInPlace(scores, 3)

        assertArrayEquals(floatArrayOf(2f / 3f, 1f / 3f, 0f), scores, 0.000001f)
        assertEquals(1f, scores.sum(), 0.000001f)
    }

    @Test
    fun `nonfinite graph cosine is rejected before publication`() {
        assertThrows(IllegalArgumentException::class.java) {
            V2GraphWeightPolicy.normalizeNonnegativeInPlace(floatArrayOf(Float.NaN), 1)
        }
    }
}
