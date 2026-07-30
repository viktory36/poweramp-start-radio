package com.powerampstartradio.similarity.algorithms

import com.powerampstartradio.data.EmbeddingIndex
import com.powerampstartradio.similarity.SelectedTrack

data class MmrSelectionEvidence(
    val step: Int,
    val trackId: Long,
    val relevance: Float,
    val maximumSelectedSimilarity: Float,
    /** Earlier selected track which produced the maximum overlap, null for the first pick. */
    val maximumSelectedTrackId: Long?,
    val objective: Float,
    val candidateRank: Int,
)

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
     * Incremental exact MMR state for a single queue plan.
     *
     * Overlap state is keyed by mmap row, so a changing drift candidate set can reuse every
     * previously computed candidate/selection score. A row first encountered later catches up
     * against each earlier selection in selection order. This is the same max-over-selected MMR
     * objective as a fresh scan, without retaining one [FloatArray] embedding per candidate.
     */
    class IncrementalState(
        private val index: EmbeddingIndex,
        private val cancellationCheck: (() -> Unit)? = null,
    ) {
        private val maximumSelectedSimilarity =
            FloatArray(index.numTracks) { Float.NEGATIVE_INFINITY }
        private val maximumSelectedIndex = IntArray(index.numTracks) { NO_SELECTION }
        private val appliedSelectionCount = IntArray(index.numTracks)

        private val selectedTrackIds = mutableListOf<Long>()
        private var selectedRows = IntArray(INITIAL_SELECTION_CAPACITY)

        private val pairLeftRows = IntArray(PAIR_CHUNK_SIZE)
        private val pairRightRows = IntArray(PAIR_CHUNK_SIZE)
        private val pairSelectionIndices = IntArray(PAIR_CHUNK_SIZE)
        private val pairScores = FloatArray(PAIR_CHUNK_SIZE)

        private var sequentialSimilarities: FloatArray? = null
        private var selectedEmbedding: FloatArray? = null

        val selectionCount: Int
            get() = selectedTrackIds.size

        /** Record a returned pick only after its caller has accepted it into the queue. */
        fun recordSelection(trackId: Long) {
            val row = index.findTrackIndex(trackId)
                ?: throw IllegalArgumentException("Selected track $trackId is absent from the embedding index")
            ensureSelectedCapacity(selectionCount + 1)
            selectedRows[selectionCount] = row
            selectedTrackIds.add(trackId)
        }

        /**
         * Select one candidate while preserving the supplied order as the deterministic tie-break.
         */
        fun selectOne(
            candidates: List<Pair<Long, Float>>,
            lambda: Float,
            isEligible: (Long) -> Boolean = { true },
        ): SelectedTrack? = selectPrepared(
            prepared = prepare(candidates),
            lambda = lambda,
            isEligibleAtPosition = { position -> isEligible(candidates[position].first) },
        )

        internal fun prepare(candidates: List<Pair<Long, Float>>): PreparedCandidates =
            PreparedCandidates(
                candidates = candidates,
                rows = IntArray(candidates.size) { position ->
                    index.findTrackIndex(candidates[position].first) ?: MISSING_ROW
                },
            )

        internal fun selectPrepared(
            prepared: PreparedCandidates,
            lambda: Float,
            isEligibleAtPosition: (Int) -> Boolean,
        ): SelectedTrack? {
            require(lambda in 0f..1f) { "lambda must be between 0 and 1" }
            if (prepared.candidates.isEmpty()) return null

            refreshOverlapState(prepared.rows)

            var bestPosition = -1
            var bestScore = Float.NEGATIVE_INFINITY
            var bestPenalty = 0f
            var bestMaximumSelectionIndex = NO_SELECTION

            for (position in prepared.candidates.indices) {
                if ((position and CANCELLATION_MASK) == 0) cancellationCheck?.invoke()
                val row = prepared.rows[position]
                if (row == MISSING_ROW || !isEligibleAtPosition(position)) continue

                val relevance = prepared.candidates[position].second
                val penalty = if (selectionCount == 0) {
                    0f
                } else {
                    check(appliedSelectionCount[row] == selectionCount) {
                        "MMR overlap state is stale for embedding row $row"
                    }
                    maximumSelectedSimilarity[row]
                }
                val objective = lambda * relevance - (1f - lambda) * penalty

                // Strict comparison deliberately retains candidate order as the tie-break.
                if (objective > bestScore) {
                    bestScore = objective
                    bestPosition = position
                    bestPenalty = penalty
                    bestMaximumSelectionIndex = maximumSelectedIndex[row]
                }
            }
            cancellationCheck?.invoke()

            if (bestPosition < 0) return null
            val (trackId, relevance) = prepared.candidates[bestPosition]
            val evidence = MmrSelectionEvidence(
                step = selectionCount + 1,
                trackId = trackId,
                relevance = relevance,
                maximumSelectedSimilarity = bestPenalty,
                maximumSelectedTrackId = bestMaximumSelectionIndex
                    .takeUnless { it == NO_SELECTION }
                    ?.let(selectedTrackIds::get),
                objective = bestScore,
                candidateRank = bestPosition + 1,
            )
            return SelectedTrack(
                trackId = trackId,
                score = relevance,
                candidateRank = bestPosition + 1,
                mmrSelectionEvidence = evidence,
            )
        }

        /** Bring every present candidate row current with selections it has not seen before. */
        private fun refreshOverlapState(candidateRows: IntArray) {
            if (selectionCount == 0 || candidateRows.isEmpty()) return

            val lastSelectionIndex = selectionCount - 1
            var validCount = 0
            var allNeedOnlyLastSelection = true
            for (row in candidateRows) {
                if (row == MISSING_ROW) continue
                validCount++
                if (appliedSelectionCount[row] != lastSelectionIndex) {
                    allNeedOnlyLastSelection = false
                }
            }

            // Above the measured gather/sequential crossover, one ordered mmap scan has much
            // better locality than visiting a relevance-sorted permutation of nearly every row.
            if (allNeedOnlyLastSelection &&
                validCount.toLong() * SEQUENTIAL_SCAN_DENOMINATOR >=
                index.numTracks.toLong() * SEQUENTIAL_SCAN_NUMERATOR
            ) {
                refreshLastSelectionSequentially(candidateRows, lastSelectionIndex)
                return
            }

            refreshMissingPairs(candidateRows)
        }

        private fun refreshLastSelectionSequentially(
            candidateRows: IntArray,
            selectionIndex: Int,
        ) {
            val similarities = sequentialSimilarities
                ?: FloatArray(index.numTracks).also { sequentialSimilarities = it }
            val embedding = selectedEmbedding
                ?: FloatArray(index.dim).also { selectedEmbedding = it }
            index.copyEmbedding(selectedRows[selectionIndex], embedding)
            index.computeAllSimilaritiesInto(
                reference = embedding,
                outSimilarities = similarities,
                cancellationCheck = cancellationCheck,
            )

            for (position in candidateRows.indices) {
                if ((position and CANCELLATION_MASK) == 0) cancellationCheck?.invoke()
                val row = candidateRows[position]
                if (row == MISSING_ROW) continue
                applyScore(row, selectionIndex, similarities[row])
            }
        }

        private fun refreshMissingPairs(candidateRows: IntArray) {
            var pairCount = 0

            fun flush() {
                if (pairCount == 0) return
                index.computePairSimilaritiesInto(
                    leftIndices = pairLeftRows,
                    rightIndices = pairRightRows,
                    outScores = pairScores,
                    pairCount = pairCount,
                    cancellationCheck = cancellationCheck,
                )
                for (pair in 0 until pairCount) {
                    applyScore(
                        row = pairLeftRows[pair],
                        selectionIndex = pairSelectionIndices[pair],
                        score = pairScores[pair],
                    )
                }
                pairCount = 0
            }

            for (candidatePosition in candidateRows.indices) {
                if ((candidatePosition and CANCELLATION_MASK) == 0) cancellationCheck?.invoke()
                val row = candidateRows[candidatePosition]
                if (row == MISSING_ROW) continue

                val firstMissingSelection = appliedSelectionCount[row]
                for (selectionIndex in firstMissingSelection until selectionCount) {
                    pairLeftRows[pairCount] = row
                    pairRightRows[pairCount] = selectedRows[selectionIndex]
                    pairSelectionIndices[pairCount] = selectionIndex
                    pairCount++
                    if (pairCount == PAIR_CHUNK_SIZE) flush()
                }
            }
            flush()
        }

        private fun applyScore(row: Int, selectionIndex: Int, score: Float) {
            check(appliedSelectionCount[row] == selectionIndex) {
                "MMR selections must be applied to a row in order"
            }
            if (score > maximumSelectedSimilarity[row]) {
                maximumSelectedSimilarity[row] = score
                maximumSelectedIndex[row] = selectionIndex
            }
            appliedSelectionCount[row] = selectionIndex + 1
        }

        private fun ensureSelectedCapacity(requiredSize: Int) {
            if (requiredSize <= selectedRows.size) return
            selectedRows = selectedRows.copyOf(maxOf(requiredSize, selectedRows.size * 2))
        }
    }

    internal data class PreparedCandidates(
        val candidates: List<Pair<Long, Float>>,
        val rows: IntArray,
    )

    /**
     * Select one track from candidates using MMR.
     *
     * This compatibility entry point accepts arbitrary selected vectors. It reads candidate rows
     * directly from the mmap and therefore creates no per-candidate embedding arrays. Queue and
     * drift planning use [IncrementalState] so each overlap is computed only once.
     */
    fun selectOne(
        candidates: List<Pair<Long, Float>>,
        selectedTrackIds: List<Long>,
        selectedEmbeddings: List<FloatArray>,
        index: EmbeddingIndex,
        lambda: Float,
        isEligible: (Long) -> Boolean = { true },
        cancellationCheck: (() -> Unit)? = null,
    ): SelectedTrack? {
        require(selectedTrackIds.size == selectedEmbeddings.size) {
            "Selected track IDs and embeddings must stay aligned"
        }
        require(lambda in 0f..1f) { "lambda must be between 0 and 1" }
        if (candidates.isEmpty()) return null
        if (selectedEmbeddings.isEmpty()) {
            val firstEligible = candidates.indexOfFirst { (trackId, _) ->
                index.findTrackIndex(trackId) != null && isEligible(trackId)
            }
            if (firstEligible < 0) return null
            val (id, score) = candidates[firstEligible]
            val evidence = MmrSelectionEvidence(
                step = 1,
                trackId = id,
                relevance = score,
                maximumSelectedSimilarity = 0f,
                maximumSelectedTrackId = null,
                objective = lambda * score,
                candidateRank = firstEligible + 1,
            )
            return SelectedTrack(
                id,
                score,
                candidateRank = firstEligible + 1,
                mmrSelectionEvidence = evidence,
            )
        }

        var bestPosition = -1
        var bestRelevance = 0f
        var bestMmrScore = Float.NEGATIVE_INFINITY
        var bestMaximumSelectedSimilarity = 0f
        var bestMaximumSelectedTrackId: Long? = null

        for (position in candidates.indices) {
            if ((position and CANCELLATION_MASK) == 0) cancellationCheck?.invoke()
            val (trackId, relevance) = candidates[position]
            if (!isEligible(trackId)) continue
            val row = index.findTrackIndex(trackId) ?: continue

            var maxSimilarity = Float.NEGATIVE_INFINITY
            var maxTrackId: Long? = null
            for (selectionIndex in selectedEmbeddings.indices) {
                val similarity = index.dotProduct(selectedEmbeddings[selectionIndex], row)
                if (similarity > maxSimilarity) {
                    maxSimilarity = similarity
                    maxTrackId = selectedTrackIds[selectionIndex]
                }
            }

            val objective = lambda * relevance - (1f - lambda) * maxSimilarity
            if (objective > bestMmrScore) {
                bestMmrScore = objective
                bestPosition = position
                bestRelevance = relevance
                bestMaximumSelectedSimilarity = maxSimilarity
                bestMaximumSelectedTrackId = maxTrackId
            }
        }
        cancellationCheck?.invoke()

        if (bestPosition < 0) return null
        val bestId = candidates[bestPosition].first
        val evidence = MmrSelectionEvidence(
            step = selectedEmbeddings.size + 1,
            trackId = bestId,
            relevance = bestRelevance,
            maximumSelectedSimilarity = bestMaximumSelectedSimilarity,
            maximumSelectedTrackId = bestMaximumSelectedTrackId,
            objective = bestMmrScore,
            candidateRank = bestPosition + 1,
        )
        return SelectedTrack(
            bestId,
            bestRelevance,
            candidateRank = bestPosition + 1,
            mmrSelectionEvidence = evidence,
        )
    }

    /** Select a fixed-query batch using one prepared row table and incremental overlap state. */
    fun selectBatch(
        candidates: List<Pair<Long, Float>>,
        numSelect: Int,
        index: EmbeddingIndex,
        lambda: Float,
        isEligible: (Long, List<Long>) -> Boolean = { _, _ -> true },
        onSelection: ((MmrSelectionEvidence) -> Unit)? = null,
        cancellationCheck: (() -> Unit)? = null,
    ): List<SelectedTrack> {
        if (candidates.isEmpty() || numSelect <= 0) return emptyList()

        val state = IncrementalState(index, cancellationCheck)
        val prepared = state.prepare(candidates)
        val selectedPositions = BooleanArray(candidates.size)
        val selectedIds = mutableListOf<Long>()
        val result = ArrayList<SelectedTrack>(minOf(numSelect, candidates.size))

        for (step in 0 until numSelect) {
            val selected = state.selectPrepared(
                prepared = prepared,
                lambda = lambda,
                isEligibleAtPosition = { position ->
                    !selectedPositions[position] &&
                        isEligible(candidates[position].first, selectedIds)
                },
            ) ?: break

            val selectedPosition = selected.candidateRank - 1
            selectedPositions[selectedPosition] = true
            selectedIds.add(selected.trackId)
            state.recordSelection(selected.trackId)
            result.add(selected)
            selected.mmrSelectionEvidence?.let { onSelection?.invoke(it) }
        }

        return result
    }

    private const val MISSING_ROW = -1
    private const val NO_SELECTION = -1
    private const val INITIAL_SELECTION_CAPACITY = 32
    private const val PAIR_CHUNK_SIZE = 16_384
    private const val CANCELLATION_MASK = 1023
    // XQ-EC72 corpus evidence crosses over between 35% gather and 40% sequential.
    private const val SEQUENTIAL_SCAN_NUMERATOR = 3L
    private const val SEQUENTIAL_SCAN_DENOMINATOR = 8L
}
