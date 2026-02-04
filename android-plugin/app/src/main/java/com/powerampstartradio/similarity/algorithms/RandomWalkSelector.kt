package com.powerampstartradio.similarity.algorithms

import com.powerampstartradio.data.GraphIndex

/**
 * Personalized PageRank on the precomputed kNN graph.
 *
 * Power iteration: pi = (1-alpha) * W^T * pi + alpha * restart
 *
 * Alpha (restart probability) controls exploration radius:
 * - High alpha (0.8+): stays near seed, tight recommendations
 * - Low alpha (0.2-): wanders far, discovers transitive connections
 *
 * Discovers tracks connected through intermediate links: if A→B→C,
 * C appears in A's results even without direct embedding similarity.
 */
object RandomWalkSelector {

    /**
     * Compute personalized PageRank scores for all nodes.
     *
     * @param graph The kNN graph
     * @param seedTrackId Starting node for the random walk
     * @param alpha Restart probability (0..1). Higher = stays closer to seed.
     * @param iterations Number of power iteration steps
     * @param additionalSeeds Optional additional seed tracks for multi-seed restart
     * @return Map of trackId to PageRank score, sorted by score descending
     */
    fun computeRanking(
        graph: GraphIndex,
        seedTrackId: Long,
        alpha: Float = 0.5f,
        iterations: Int = 30,
        additionalSeeds: List<Long> = emptyList()
    ): List<Pair<Long, Float>> {
        val n = graph.numNodes

        // Build restart vector
        val allSeeds = listOf(seedTrackId) + additionalSeeds.filter { graph.hasTrack(it) }

        // We operate on a dense array indexed by graph node index.
        // Build track ID -> node index mapping by scanning graph.
        // GraphIndex stores track IDs internally; we need index-based iteration.

        // Initialize pi uniformly, restart vector concentrated on seeds
        // Since GraphIndex only exposes getNeighbors(trackId), we work in trackId space
        // using sparse representation for efficiency.

        // Sparse PageRank: only track nodes with non-zero probability
        var pi = mutableMapOf<Long, Float>()
        val restartWeight = 1f / allSeeds.size
        val restart = mutableMapOf<Long, Float>()
        for (seed in allSeeds) {
            restart[seed] = restartWeight
            pi[seed] = restartWeight
        }

        // Power iteration
        for (iter in 0 until iterations) {
            val newPi = mutableMapOf<Long, Float>()

            // Transition: for each node with probability, distribute to its neighbors
            for ((nodeId, prob) in pi) {
                val neighbors = graph.getNeighbors(nodeId)
                for ((neighborId, weight) in neighbors) {
                    newPi[neighborId] = (newPi[neighborId] ?: 0f) + (1f - alpha) * prob * weight
                }
            }

            // Add restart
            for ((seedId, weight) in restart) {
                newPi[seedId] = (newPi[seedId] ?: 0f) + alpha * weight
            }

            pi = newPi

            // Prune very small probabilities to keep sparse
            if (pi.size > n / 2) {
                val threshold = 1e-8f
                pi = pi.filterValues { it > threshold }.toMutableMap()
            }
        }

        // Sort by score, exclude seeds
        val seedSet = allSeeds.toSet()
        return pi.entries
            .filter { it.key !in seedSet }
            .sortedByDescending { it.value }
            .map { it.key to it.value }
    }
}
