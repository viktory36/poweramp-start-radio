package com.powerampstartradio.indexing

/** Matches NativeMath top-K: numeric score descending, exact ties by ascending track ID. */
internal object V2GraphTopKOrdering {
    fun tryInsert(
        neighborIndices: IntArray,
        rawScores: FloatArray,
        candidateIndex: Int,
        candidateScore: Float,
        trackIdsByIndex: LongArray,
        k: Int,
    ): Boolean {
        require(k in 1..neighborIndices.size && k <= rawScores.size)
        var worst = 0
        for (position in 1 until k) {
            if (isWorse(
                    rawScores[position],
                    trackIdsByIndex[neighborIndices[position]],
                    rawScores[worst],
                    trackIdsByIndex[neighborIndices[worst]],
                )
            ) {
                worst = position
            }
        }
        val candidateId = trackIdsByIndex[candidateIndex]
        val worstId = trackIdsByIndex[neighborIndices[worst]]
        if (!isBetter(candidateScore, candidateId, rawScores[worst], worstId)) return false
        neighborIndices[worst] = candidateIndex
        rawScores[worst] = candidateScore
        return true
    }

    fun sortBestFirst(
        neighborIndices: IntArray,
        rawScores: FloatArray,
        trackIdsByIndex: LongArray,
        k: Int,
    ) {
        require(k in 0..neighborIndices.size && k <= rawScores.size)
        for (position in 0 until k - 1) {
            var best = position
            for (candidate in position + 1 until k) {
                if (isBetter(
                        rawScores[candidate],
                        trackIdsByIndex[neighborIndices[candidate]],
                        rawScores[best],
                        trackIdsByIndex[neighborIndices[best]],
                    )
                ) {
                    best = candidate
                }
            }
            if (best != position) {
                val index = neighborIndices[position]
                neighborIndices[position] = neighborIndices[best]
                neighborIndices[best] = index
                val score = rawScores[position]
                rawScores[position] = rawScores[best]
                rawScores[best] = score
            }
        }
    }

    private fun isBetter(score: Float, trackId: Long, other: Float, otherId: Long): Boolean {
        if (score.isNaN()) return other.isNaN() && trackId < otherId
        if (other.isNaN()) return true
        return score > other || (score == other && trackId < otherId)
    }

    private fun isWorse(score: Float, trackId: Long, other: Float, otherId: Long): Boolean {
        if (score.isNaN()) return !other.isNaN() || trackId > otherId
        if (other.isNaN()) return false
        return score < other || (score == other && trackId > otherId)
    }
}

/** Converts nearest-neighbor cosine scores into a complete stochastic graph row. */
internal object V2GraphWeightPolicy {
    fun normalizeNonnegativeInPlace(rawScores: FloatArray, activeScores: Int) {
        require(activeScores in 1..rawScores.size)
        var sum = 0.0
        for (position in 0 until activeScores) {
            val score = rawScores[position]
            require(score.isFinite()) { "graph cosine score must be finite" }
            val nonnegative = maxOf(score, 0f)
            rawScores[position] = nonnegative
            sum += nonnegative.toDouble()
        }
        if (sum > 0.0) {
            for (position in 0 until activeScores) {
                rawScores[position] = (rawScores[position].toDouble() / sum).toFloat()
            }
            return
        }

        // Cosine can validly be non-positive for every nearest neighbor. The topology is still
        // meaningful, and Graph Explorer traverses its K nearest slots uniformly.
        val uniform = 1f / activeScores.toFloat()
        for (position in 0 until activeScores) rawScores[position] = uniform
    }
}
