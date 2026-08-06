package com.powerampstartradio.similarity.algorithms

import com.powerampstartradio.similarity.SelectedTrack

/** Fixed-query cosine ranking with no secondary musical objective. */
object ClosestSelector {
    fun select(
        candidates: List<Pair<Long, Float>>,
        numSelect: Int,
        isEligible: (Long, List<Long>) -> Boolean = { _, _ -> true },
    ): List<SelectedTrack> {
        require(numSelect >= 0) { "numSelect must be non-negative" }
        val selected = ArrayList<SelectedTrack>(minOf(numSelect, candidates.size))
        val selectedIds = ArrayList<Long>(minOf(numSelect, candidates.size))
        val alreadySelected = BooleanArray(candidates.size)
        while (selected.size < numSelect) {
            val candidateIndex = candidates.indices.firstOrNull { index ->
                !alreadySelected[index] && isEligible(candidates[index].first, selectedIds)
            } ?: break
            val candidate = candidates[candidateIndex]
            alreadySelected[candidateIndex] = true
            selected += SelectedTrack(
                trackId = candidate.first,
                score = candidate.second,
                candidateRank = candidateIndex + 1,
            )
            selectedIds += candidate.first
        }
        return selected
    }
}
