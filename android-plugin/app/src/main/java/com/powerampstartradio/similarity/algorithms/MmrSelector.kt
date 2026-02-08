package com.powerampstartradio.similarity.algorithms

import com.powerampstartradio.data.EmbeddingIndex
import com.powerampstartradio.similarity.SelectedTrack

/**
 * Maximal Marginal Relevance (MMR) selector.
 *
 * Picks tracks that balance relevance to the query with diversity from
 * already-selected tracks. At each step:
 *   score(c) = lambda * sim(c, query) - (1-lambda) * max_sim(c, selected)
 *
 * Lambda controls the tradeoff: 1.0 = pure relevance, 0.0 = pure diversity.
 */
object MmrSelector {

    /**
     * Select one track from candidates using MMR.
     *
     * @param candidates List of (trackId, relevanceScore) sorted by relevance
     * @param selectedEmbeddings Embeddings of already-selected tracks
     * @param index Embedding index for looking up candidate embeddings
     * @param lambda Relevance-diversity tradeoff (0..1)
     * @return SelectedTrack of the best candidate, or null
     */
    fun selectOne(
        candidates: List<Pair<Long, Float>>,
        selectedEmbeddings: List<FloatArray>,
        index: EmbeddingIndex,
        lambda: Float
    ): SelectedTrack? {
        if (candidates.isEmpty()) return null
        if (selectedEmbeddings.isEmpty()) {
            val (id, score) = candidates.first()
            return SelectedTrack(id, score, candidateRank = 1)
        }

        var bestIdx = -1
        var bestId = -1L
        var bestRelevance = 0f
        var bestMmrScore = Float.NEGATIVE_INFINITY

        for (i in candidates.indices) {
            val (trackId, relevance) = candidates[i]
            val emb = index.getEmbeddingByTrackId(trackId) ?: continue

            // Max similarity to any already-selected track
            var maxSimToSelected = Float.NEGATIVE_INFINITY
            for (sel in selectedEmbeddings) {
                val sim = dotProduct(emb, sel)
                if (sim > maxSimToSelected) maxSimToSelected = sim
            }

            val mmrScore = lambda * relevance - (1f - lambda) * maxSimToSelected

            if (mmrScore > bestMmrScore) {
                bestMmrScore = mmrScore
                bestIdx = i
                bestRelevance = relevance
                bestId = trackId
            }
        }

        return if (bestId >= 0) SelectedTrack(bestId, bestRelevance, candidateRank = bestIdx + 1) else null
    }

    /**
     * Select a batch of tracks using iterative MMR.
     *
     * @param candidates List of (trackId, relevanceScore)
     * @param numSelect How many to select
     * @param index Embedding index
     * @param lambda Relevance-diversity tradeoff
     * @return Selected tracks in selection order
     */
    fun selectBatch(
        candidates: List<Pair<Long, Float>>,
        numSelect: Int,
        index: EmbeddingIndex,
        lambda: Float
    ): List<SelectedTrack> {
        if (candidates.isEmpty()) return emptyList()

        val result = mutableListOf<SelectedTrack>()
        val selectedEmbeddings = mutableListOf<FloatArray>()
        val remaining = candidates.toMutableList()

        // Pre-load all candidate embeddings
        val embCache = HashMap<Long, FloatArray>(candidates.size)
        for ((trackId, _) in candidates) {
            index.getEmbeddingByTrackId(trackId)?.let { embCache[trackId] = it }
        }

        // Track max-sim-to-selected for each candidate (incrementally updated)
        val maxSimToSelected = FloatArray(candidates.size) { Float.NEGATIVE_INFINITY }
        // Pre-build trackId -> original index map for O(1) lookup
        val tidToOrigIdx = HashMap<Long, Int>(candidates.size)
        for (i in candidates.indices) tidToOrigIdx[candidates[i].first] = i

        for (step in 0 until numSelect) {
            if (remaining.isEmpty()) break

            var bestIdx = -1
            var bestScore = Float.NEGATIVE_INFINITY

            for (i in remaining.indices) {
                val (trackId, relevance) = remaining[i]
                val emb = embCache[trackId] ?: continue

                val origIdx = tidToOrigIdx[trackId] ?: continue

                // Update max sim with the most recently selected track
                if (selectedEmbeddings.isNotEmpty()) {
                    val lastSelected = selectedEmbeddings.last()
                    val sim = dotProduct(emb, lastSelected)
                    if (sim > maxSimToSelected[origIdx]) {
                        maxSimToSelected[origIdx] = sim
                    }
                }

                val penalty = if (selectedEmbeddings.isEmpty()) 0f else maxSimToSelected[origIdx]
                val mmrScore = lambda * relevance - (1f - lambda) * penalty

                if (mmrScore > bestScore) {
                    bestScore = mmrScore
                    bestIdx = i
                }
            }

            if (bestIdx < 0) break

            val (selectedId, selectedSim) = remaining.removeAt(bestIdx)
            val selectedEmb = embCache[selectedId] ?: continue
            val origIdx = tidToOrigIdx[selectedId] ?: continue
            result.add(SelectedTrack(selectedId, selectedSim, candidateRank = origIdx + 1))
            selectedEmbeddings.add(selectedEmb)
        }

        return result
    }

    private fun dotProduct(a: FloatArray, b: FloatArray): Float {
        var sum = 0f
        for (i in a.indices) sum += a[i] * b[i]
        return sum
    }
}
