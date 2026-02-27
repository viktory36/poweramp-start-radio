package com.powerampstartradio.similarity.algorithms

import com.powerampstartradio.data.GraphIndex
import java.util.Random

/**
 * Monte Carlo random walks on the precomputed kNN graph.
 *
 * Runs [NUM_WALKS] independent walks from the seed. Each walk follows ONE
 * uniformly-random edge per step, then restarts with probability [alpha].
 * Nodes are ranked by how often they are the **terminal position** of a walk —
 * not every node visited along the way. This prevents hop-1 neighbors from
 * drowning out deeper exploration (every walk passes through hop-1 on step 0).
 *
 * Alpha controls exploration depth (expected walk length = 1/alpha):
 * - High alpha (0.95): ~1 step → terminals at hop 1 → tight
 * - Medium alpha (0.50): ~2 steps → terminals at hop 1-3
 * - Low alpha (0.05): ~20 steps → terminals deep in graph → serendipitous
 *
 * Stochastic by nature — each run gives slightly different results,
 * fitting the "Explorer" mode identity.
 */
object RandomWalkSelector {

    private const val NUM_WALKS = 10000
    private const val MAX_STEPS = 100  // safety cap per walk

    /**
     * Compute terminal-position ranking via Monte Carlo random walks.
     *
     * @param graph The kNN graph
     * @param seedTrackId Starting node for each walk
     * @param alpha Restart probability (0..1). Higher = stays closer to seed.
     * @param iterations Unused (kept for API compatibility)
     * @param additionalSeeds Unused (kept for API compatibility)
     * @return List of (trackId, normalizedScore) sorted by terminal count descending
     */
    fun computeRanking(
        graph: GraphIndex,
        seedTrackId: Long,
        alpha: Float = 0.5f,
        iterations: Int = 30,
        additionalSeeds: List<Long> = emptyList()
    ): List<Pair<Long, Float>> {
        val rand = Random()
        val terminalCounts = HashMap<Long, Int>(512)

        repeat(NUM_WALKS) {
            var prev = -1L
            var current = seedTrackId
            for (step in 0 until MAX_STEPS) {
                val next = graph.sampleNeighbor(current, rand, excludeId = prev)
                if (next == -1L) break
                prev = current
                current = next
                if (rand.nextFloat() < alpha) break  // restart
            }
            // Only count where the walk ended, not every node along the way
            if (current != seedTrackId) {
                terminalCounts[current] = (terminalCounts[current] ?: 0) + 1
            }
        }

        if (terminalCounts.isEmpty()) return emptyList()

        val maxCount = terminalCounts.values.max()
        return terminalCounts.entries
            .sortedByDescending { it.value }
            .map { it.key to it.value.toFloat() / maxCount }
    }
}
