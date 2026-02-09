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
        selectedTrackIds: List<Long> = emptyList(),
        index: EmbeddingIndex,
        lambda: Float
    ): SelectedTrack? {
        if (candidates.isEmpty()) return null
        if (selectedEmbeddings.isEmpty()) {
            val (id, score) = candidates.first()
            return SelectedTrack(id, score, candidateRank = 1, algorithmScore = score)
        }

        var bestIdx = -1
        var bestId = -1L
        var bestRelevance = 0f
        var bestMmrScore = Float.NEGATIVE_INFINITY
        var bestMaxSim = 0f
        var bestNearestId = -1L

        for (i in candidates.indices) {
            val (trackId, relevance) = candidates[i]
            val emb = index.getEmbeddingByTrackId(trackId) ?: continue

            // Max similarity to any already-selected track
            var maxSimToSelected = Float.NEGATIVE_INFINITY
            var nearestSelIdx = -1
            for (j in selectedEmbeddings.indices) {
                val sim = dotProduct(emb, selectedEmbeddings[j])
                if (sim > maxSimToSelected) {
                    maxSimToSelected = sim
                    nearestSelIdx = j
                }
            }

            val mmrScore = lambda * relevance - (1f - lambda) * maxSimToSelected

            if (mmrScore > bestMmrScore) {
                bestMmrScore = mmrScore
                bestIdx = i
                bestRelevance = relevance
                bestId = trackId
                bestMaxSim = maxSimToSelected
                bestNearestId = if (nearestSelIdx >= 0 && nearestSelIdx < selectedTrackIds.size)
                    selectedTrackIds[nearestSelIdx] else -1L
            }
        }

        if (bestId < 0) return null

        // Count how many candidates with higher relevance were bypassed
        val bypassed = candidates.count { (_, rel) -> rel > bestRelevance }

        return SelectedTrack(
            bestId, bestRelevance, candidateRank = bestIdx + 1,
            algorithmScore = bestMmrScore,
            redundancyPenalty = bestMaxSim,
            nearestSelectedId = if (bestNearestId >= 0) bestNearestId else null,
            bypassed = bypassed
        )
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

        // Track which selected track is nearest for each candidate
        val nearestSelectedIdx = IntArray(candidates.size) { -1 }
        val selectedTrackIds = mutableListOf<Long>()

        for (step in 0 until numSelect) {
            if (remaining.isEmpty()) break

            var bestIdx = -1
            var bestScore = Float.NEGATIVE_INFINITY
            var bestPenalty = 0f
            var bestNearestSelIdx = -1

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
                        nearestSelectedIdx[origIdx] = selectedEmbeddings.size - 1
                    }
                }

                val penalty = if (selectedEmbeddings.isEmpty()) 0f else maxSimToSelected[origIdx]
                val mmrScore = lambda * relevance - (1f - lambda) * penalty

                if (mmrScore > bestScore) {
                    bestScore = mmrScore
                    bestIdx = i
                    bestPenalty = penalty
                    bestNearestSelIdx = nearestSelectedIdx[origIdx]
                }
            }

            if (bestIdx < 0) break

            val (selectedId, selectedSim) = remaining.removeAt(bestIdx)
            val selectedEmb = embCache[selectedId] ?: continue
            val origIdx = tidToOrigIdx[selectedId] ?: continue

            // Count bypassed: candidates with higher relevance that weren't picked
            val bypassed = remaining.count { (_, rel) -> rel > selectedSim }

            val nearestId = if (bestNearestSelIdx >= 0 && bestNearestSelIdx < selectedTrackIds.size)
                selectedTrackIds[bestNearestSelIdx] else null

            result.add(SelectedTrack(
                selectedId, selectedSim, candidateRank = origIdx + 1,
                algorithmScore = bestScore,
                redundancyPenalty = bestPenalty,
                nearestSelectedId = nearestId,
                bypassed = bypassed
            ))
            selectedEmbeddings.add(selectedEmb)
            selectedTrackIds.add(selectedId)
        }

        return result
    }

    private fun dotProduct(a: FloatArray, b: FloatArray): Float {
        var sum = 0f
        for (i in a.indices) sum += a[i] * b[i]
        return sum
    }
}
