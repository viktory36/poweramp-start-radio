package com.powerampstartradio.similarity.algorithms

import com.powerampstartradio.data.GraphIndex

/**
 * Compatibility facade for call sites still named after V1's sampled Random Walk mode.
 *
 * V2 no longer samples 10,000 walks. This delegates to [GraphExplorerSelector]'s exact,
 * deterministic terminal distribution and preserves the legacy display-score convention
 * where the first result is normalized to `1f`. New integrations should consume
 * [GraphExplorerResult] directly so terminal probability and expected route length remain
 * available as honest evidence.
 */
object RandomWalkSelector {
    @Suppress("UNUSED_PARAMETER")
    fun computeRanking(
        graph: GraphIndex,
        seedTrackId: Long,
        alpha: Float = 0.5f,
        iterations: Int = 30,
        additionalSeeds: List<Long> = emptyList(),
        cancellationCheck: () -> Unit = {},
    ): List<Pair<Long, Float>> {
        val result = GraphExplorerSelector.compute(
            graph = graph,
            seedTrackId = seedTrackId,
            stopProbability = alpha,
            cancellationCheck = cancellationCheck,
        )
        val maximum = result.ranking.firstOrNull()?.terminalProbability ?: return emptyList()
        return result.ranking.map { score ->
            score.trackId to (score.terminalProbability / maximum).toFloat()
        }
    }
}
