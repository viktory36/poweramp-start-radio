package com.powerampstartradio.similarity.algorithms

import com.powerampstartradio.data.EmbeddingIndex
import com.powerampstartradio.similarity.SelectedTrack
import kotlin.math.ceil
import kotlin.math.sqrt

/**
 * Determinantal Point Process (DPP) greedy MAP selector.
 *
 * Maximizes both quality and diversity simultaneously using a DPP kernel.
 * The kernel L[i][j] = q[i] * q[j] * dot(emb_i, emb_j) captures:
 * - Quality: q[i] = (relevance / complete-domain maximum)^qualityExponent
 * - Diversity: dot(emb_i, emb_j) = pairwise similarity
 *
 * DPP assigns higher probability to subsets where items are both
 * high-quality and dissimilar. It optimizes a different set objective from MMR.
 * Complete-domain normalization is one global positive scaling of the kernel, so it preserves
 * fixed-cardinality greedy ordering while avoiding score-scale-dependent numerical underflow.
 *
 * Uses fast greedy MAP with incremental Cholesky decomposition
 * (Chen et al., 2018). O(K²×N) for a K-item selection.
 */
object DppSelector {

    data class CertificationEvidence(
        val totalCandidateCount: Int,
        val initialCandidateCount: Int,
        val attemptedCandidateCounts: List<Int>,
        val finalCandidateCount: Int,
        val selectedMarginalGains: List<Double>,
        val finalUnseenGainUpperBound: Double?,
        val usedFullDomain: Boolean,
    )

    data class CertifiedSelection(
        val tracks: List<SelectedTrack>,
        val evidence: CertificationEvidence,
    )

    /**
     * Select a batch of tracks using greedy DPP MAP inference.
     *
     * @param candidates List of (trackId, relevanceScore)
     * @param numSelect How many to select
     * @param index Embedding index for looking up embeddings
     * @param qualityExponent Exponent for quality scores (higher = prefer more relevant)
     * @return Selected tracks in selection order
     */
    fun selectBatch(
        candidates: List<Pair<Long, Float>>,
        numSelect: Int,
        index: EmbeddingIndex,
        qualityExponent: Float = 1.0f,
        cancellationCheck: (() -> Unit)? = null,
        isEligible: (Long, List<Long>) -> Boolean = { _, _ -> true },
    ): List<SelectedTrack> {
        require(qualityExponent.isFinite() && qualityExponent >= 0f) {
            "qualityExponent must be finite and non-negative"
        }
        require(candidates.all { (_, relevance) -> relevance.isFinite() }) {
            "candidate relevance must be finite"
        }
        if (candidates.isEmpty() || numSelect <= 0) return emptyList()

        val completeDomainMaxRelevance = completeDomainMaxRelevance(candidates)
        return greedySelect(
            candidates = candidates,
            numSelect = numSelect,
            index = index,
            qualityExponent = qualityExponent,
            completeDomainMaxRelevance = completeDomainMaxRelevance,
            cancellationCheck = cancellationCheck,
            isEligible = isEligible,
        ).toSelectedTracks(candidates)
    }

    /**
     * Reproduces greedy DPP over the complete candidate domain while usually loading only a
     * prefix of its embeddings.
     *
     * A candidate's remaining Cholesky diagonal can only decrease from its initial `q²`.
     * Therefore, a prefix choice is globally certified when its marginal gain is strictly
     * greater than the largest initial `q²` outside the prefix at every selected step. If a
     * step cannot be certified, the prefix grows and the greedy sequence is recomputed.
     *
     * Candidate order determines stable tie handling and `candidateRank`; callers should pass
     * the complete relevance-ranked domain, not a previously truncated neighborhood.
     * `isEligible` must be a deterministic function of its arguments because an uncertified
     * prefix is recomputed after expansion.
     */
    fun selectBatchCertified(
        candidates: List<Pair<Long, Float>>,
        numSelect: Int,
        index: EmbeddingIndex,
        initialCandidateCount: Int,
        qualityExponent: Float = 1.0f,
        growthFactor: Float = 2f,
        cancellationCheck: (() -> Unit)? = null,
        isEligible: (Long, List<Long>) -> Boolean = { _, _ -> true },
    ): CertifiedSelection {
        require(qualityExponent.isFinite() && qualityExponent >= 0f) {
            "qualityExponent must be finite and non-negative"
        }
        require(growthFactor.isFinite() && growthFactor > 1f) {
            "growthFactor must be finite and greater than one"
        }
        require(candidates.all { (_, relevance) -> relevance.isFinite() }) {
            "candidate relevance must be finite"
        }
        if (candidates.isEmpty() || numSelect <= 0) {
            return CertifiedSelection(
                tracks = emptyList(),
                evidence = CertificationEvidence(
                    totalCandidateCount = candidates.size,
                    initialCandidateCount = 0,
                    attemptedCandidateCounts = emptyList(),
                    finalCandidateCount = 0,
                    selectedMarginalGains = emptyList(),
                    finalUnseenGainUpperBound = null,
                    usedFullDomain = false,
                ),
            )
        }

        val totalCount = candidates.size
        val firstCount = initialCandidateCount.coerceIn(1, totalCount)
        val completeDomainMaxRelevance = completeDomainMaxRelevance(candidates)
        // q=1 for every candidate at exponent zero, so a proper prefix cannot pass the strict
        // certificate. At exponent 0.5, production-corpus device evidence found geometric
        // retries 2.08x slower than this same bounded full result. Both use full greedy semantics.
        if (qualityExponent <= DIRECT_FULL_MAX_EXPONENT) {
            val run = greedySelect(
                candidates = candidates,
                numSelect = numSelect,
                index = index,
                qualityExponent = qualityExponent,
                completeDomainMaxRelevance = completeDomainMaxRelevance,
                cancellationCheck = cancellationCheck,
                isEligible = isEligible,
            )
            return CertifiedSelection(
                tracks = run.toSelectedTracks(candidates),
                evidence = CertificationEvidence(
                    totalCandidateCount = totalCount,
                    initialCandidateCount = totalCount,
                    attemptedCandidateCounts = listOf(totalCount),
                    finalCandidateCount = totalCount,
                    selectedMarginalGains = run.selectedMarginalGains,
                    finalUnseenGainUpperBound = null,
                    usedFullDomain = true,
                ),
            )
        }
        val suffixInitialGainBound = DoubleArray(totalCount + 1)
        for (indexInCandidates in candidates.lastIndex downTo 0) {
            if ((indexInCandidates and 1023) == 0) cancellationCheck?.invoke()
            suffixInitialGainBound[indexInCandidates] = maxOf(
                initialGainUpperBound(
                    relevance = candidates[indexInCandidates].second,
                    exponent = qualityExponent,
                    completeDomainMaxRelevance = completeDomainMaxRelevance,
                ),
                suffixInitialGainBound[indexInCandidates + 1],
            )
        }

        val attemptedCounts = mutableListOf<Int>()
        var candidateCount = firstCount
        while (true) {
            attemptedCounts += candidateCount
            val run = greedySelect(
                candidates = candidates.subList(0, candidateCount),
                numSelect = numSelect,
                index = index,
                qualityExponent = qualityExponent,
                completeDomainMaxRelevance = completeDomainMaxRelevance,
                cancellationCheck = cancellationCheck,
                isEligible = isEligible,
            )
            val unseenBound = suffixInitialGainBound[candidateCount].takeIf {
                candidateCount < totalCount
            }
            val targetCount = minOf(numSelect, totalCount)
            val selectedEnough = run.selectedIndices.size == targetCount
            val everyStepCertified = unseenBound == null ||
                run.selectedMarginalGains.all { gain -> gain > unseenBound }

            if (unseenBound == null || (selectedEnough && everyStepCertified)) {
                return CertifiedSelection(
                    tracks = run.toSelectedTracks(candidates),
                    evidence = CertificationEvidence(
                        totalCandidateCount = totalCount,
                        initialCandidateCount = firstCount,
                        attemptedCandidateCounts = attemptedCounts.toList(),
                        finalCandidateCount = candidateCount,
                        selectedMarginalGains = run.selectedMarginalGains.toList(),
                        finalUnseenGainUpperBound = unseenBound,
                        usedFullDomain = candidateCount == totalCount,
                    ),
                )
            }

            val geometricNextCount = minOf(
                totalCount,
                maxOf(candidateCount + 1, ceil(candidateCount * growthFactor).toInt()),
            )
            val certificateGuidedNextCount = if (selectedEnough) {
                firstPrefixBelowSelectedGain(
                    currentCandidateCount = candidateCount,
                    suffixInitialGainBound = suffixInitialGainBound,
                    selectedMarginalGains = run.selectedMarginalGains,
                )
            } else {
                null
            }
            // This bound only chooses the next workspace; the expanded run must still pass the
            // same strict certificate above. Never grow more slowly than the geometric fallback.
            candidateCount = maxOf(
                geometricNextCount,
                certificateGuidedNextCount ?: geometricNextCount,
            )
        }
    }

    /**
     * Find the first larger prefix whose unseen initial-gain bound is below the weakest gain in
     * the current run. New candidates can change that run, so this is a safe jump hint, not proof.
     */
    private fun firstPrefixBelowSelectedGain(
        currentCandidateCount: Int,
        suffixInitialGainBound: DoubleArray,
        selectedMarginalGains: List<Double>,
    ): Int? {
        val weakestSelectedGain = selectedMarginalGains.minOrNull()
            ?.takeIf { it.isFinite() && it > 0.0 }
            ?: return null
        val totalCount = suffixInitialGainBound.lastIndex
        if (currentCandidateCount >= totalCount ||
            suffixInitialGainBound[totalCount] >= weakestSelectedGain
        ) {
            return null
        }

        var low = currentCandidateCount + 1
        var high = totalCount
        while (low < high) {
            val middle = low + (high - low) / 2
            if (suffixInitialGainBound[middle] < weakestSelectedGain) {
                high = middle
            } else {
                low = middle + 1
            }
        }
        return low
    }

    private data class GreedyRun(
        val selectedIndices: List<Int>,
        val selectedMarginalGains: List<Double>,
    ) {
        fun toSelectedTracks(candidates: List<Pair<Long, Float>>): List<SelectedTrack> =
            selectedIndices.map { index ->
                val (trackId, relevance) = candidates[index]
                SelectedTrack(trackId, relevance, candidateRank = index + 1)
            }
    }

    private fun greedySelect(
        candidates: List<Pair<Long, Float>>,
        numSelect: Int,
        index: EmbeddingIndex,
        qualityExponent: Float,
        completeDomainMaxRelevance: Double,
        cancellationCheck: (() -> Unit)?,
        isEligible: (Long, List<Long>) -> Boolean,
    ): GreedyRun {
        val n = candidates.size
        val k = minOf(numSelect, n)

        // Resolve mmap rows once. Embeddings stay in the mapped file instead of being copied to
        // an O(N * dimension) Java-heap matrix.
        val rowByCandidate = index.findTrackIndices(
            LongArray(n) { position -> candidates[position].first },
        )
        val quality = DoubleArray(n)
        val validMask = BooleanArray(n)
        var validCount = 0

        for (i in 0 until n) {
            if (rowByCandidate[i] >= 0) {
                quality[i] = qualityScore(
                    relevance = candidates[i].second,
                    exponent = qualityExponent,
                    completeDomainMaxRelevance = completeDomainMaxRelevance,
                )
                validMask[i] = true
                validCount++
            }
        }

        // Greedy DPP MAP with incremental Cholesky
        // L[i][j] = q[i] * q[j] * dot(emb_i, emb_j)
        // We maintain a partial Cholesky factor to incrementally compute
        // the marginal gain of adding each candidate.

        val selected = mutableListOf<Int>()
        val selectedMarginalGains = mutableListOf<Double>()
        val selectedIds = mutableListOf<Long>()
        val selectedMask = BooleanArray(n)
        if (validCount == 0) {
            return GreedyRun(emptyList(), emptyList())
        }

        // Primitive, candidate-major c[i][j] storage avoids N DoubleArray objects and is the
        // dominant bounded workspace: N * K * 8 bytes.
        val choleskyFactors = DoubleArray(Math.multiplyExact(n, k))

        // Pair rows and scores are kept in candidate order, compacted only around missing rows.
        // They are reused at every greedy step.
        val validCandidatePositions = IntArray(validCount)
        val candidateRows = IntArray(validCount)
        var validPosition = 0
        for (candidatePosition in 0 until n) {
            val row = rowByCandidate[candidatePosition]
            if (row >= 0) {
                validCandidatePositions[validPosition] = candidatePosition
                candidateRows[validPosition] = row
                validPosition++
            }
        }
        val selectedRows = IntArray(validCount)
        val pairSimilarities = FloatArray(validCount)
        val useSequentialSimilarityColumns =
            validCount.toLong() * SEQUENTIAL_SCAN_DENOMINATOR >=
                index.numTracks.toLong() * SEQUENTIAL_SCAN_NUMERATOR
        val selectedEmbedding = if (useSequentialSimilarityColumns) FloatArray(index.dim) else null
        val allSimilarities =
            if (useSequentialSimilarityColumns) FloatArray(index.numTracks) else null

        // d[i] = remaining diagonal term for candidate i
        // Initially d[i] = L[i][i] = q[i]² * dot(emb_i, emb_i) = q[i]²
        // (embeddings are L2-normalized, so dot(emb_i, emb_i) = 1)
        val diagRemaining = DoubleArray(n) { i ->
            if (validMask[i]) quality[i] * quality[i] else 0.0
        }

        for (step in 0 until k) {
            cancellationCheck?.invoke()
            // Find candidate with maximum marginal gain (= d[i])
            var bestIdx = -1
            var bestGain = -1.0

            for (i in 0 until n) {
                if ((i and 1023) == 0) cancellationCheck?.invoke()
                if (!validMask[i] || selectedMask[i]) continue
                if (!isEligible(candidates[i].first, selectedIds)) continue
                if (diagRemaining[i] > bestGain) {
                    bestGain = diagRemaining[i]
                    bestIdx = i
                }
            }

            if (bestIdx < 0 || bestGain <= 0.0) break

            selected.add(bestIdx)
            selectedMarginalGains.add(bestGain)
            selectedMask[bestIdx] = true
            selectedIds.add(candidates[bestIdx].first)

            // Update Cholesky factors for remaining candidates
            val sqrtGain = sqrt(bestGain)
            if (useSequentialSimilarityColumns) {
                index.copyEmbedding(rowByCandidate[bestIdx], selectedEmbedding!!)
                index.computeAllSimilaritiesInto(
                    reference = selectedEmbedding,
                    outSimilarities = allSimilarities!!,
                    cancellationCheck = cancellationCheck,
                )
                for (pairPosition in 0 until validCount) {
                    if ((pairPosition and 1023) == 0) cancellationCheck?.invoke()
                    pairSimilarities[pairPosition] = allSimilarities[candidateRows[pairPosition]]
                }
            } else {
                selectedRows.fill(rowByCandidate[bestIdx])
                index.computePairSimilaritiesInto(
                    leftIndices = candidateRows,
                    rightIndices = selectedRows,
                    outScores = pairSimilarities,
                    cancellationCheck = cancellationCheck,
                )
            }
            val bestFactorOffset = bestIdx * k

            for (pairPosition in 0 until validCount) {
                if ((pairPosition and 1023) == 0) cancellationCheck?.invoke()
                val i = validCandidatePositions[pairPosition]
                if (selectedMask[i]) continue

                // L[i][bestIdx] = q[i] * q[bestIdx] * dot(emb_i, emb_bestIdx)
                val kernelVal =
                    quality[i] * quality[bestIdx] * pairSimilarities[pairPosition].toDouble()

                // Subtract contributions from previous Cholesky entries
                var subtracted = kernelVal
                val factorOffset = i * k
                for (j in 0 until step) {
                    subtracted -= choleskyFactors[factorOffset + j] *
                        choleskyFactors[bestFactorOffset + j]
                }

                choleskyFactors[factorOffset + step] = subtracted / sqrtGain

                // Update remaining diagonal
                diagRemaining[i] -= choleskyFactors[factorOffset + step] *
                    choleskyFactors[factorOffset + step]
                if (diagRemaining[i] < 0.0) diagRemaining[i] = 0.0
            }

            // Also update the selected item's own Cholesky factor
            choleskyFactors[bestFactorOffset + step] = sqrtGain
        }

        return GreedyRun(selected, selectedMarginalGains)
    }

    private fun initialGainUpperBound(
        relevance: Float,
        exponent: Float,
        completeDomainMaxRelevance: Double,
    ): Double {
        val quality = qualityScore(relevance, exponent, completeDomainMaxRelevance)
        val gain = quality * quality
        return gain.takeIf(Double::isFinite) ?: Double.POSITIVE_INFINITY
    }

    internal fun qualityScore(
        relevance: Float,
        exponent: Float,
        completeDomainMaxRelevance: Double = 1.0,
    ): Double {
        require(exponent.isFinite() && exponent >= 0f) {
            "exponent must be finite and non-negative"
        }
        require(relevance.isFinite()) { "relevance must be finite" }
        require(completeDomainMaxRelevance.isFinite() && completeDomainMaxRelevance >= 0.0) {
            "completeDomainMaxRelevance must be finite and non-negative"
        }
        if (exponent == 0f) return 1.0
        if (completeDomainMaxRelevance == 0.0) return 0.0
        val normalizedRelevance =
            (relevance.coerceAtLeast(0f).toDouble() / completeDomainMaxRelevance)
                .coerceIn(0.0, 1.0)
        return Math.pow(normalizedRelevance, exponent.toDouble())
    }

    private fun completeDomainMaxRelevance(candidates: List<Pair<Long, Float>>): Double {
        var maximum = 0.0
        candidates.forEach { (_, relevance) ->
            maximum = maxOf(maximum, relevance.coerceAtLeast(0f).toDouble())
        }
        return maximum
    }

    // XQ-EC72 corpus evidence crosses over between 35% gather and 40% sequential.
    private const val SEQUENTIAL_SCAN_NUMERATOR = 3L
    private const val SEQUENTIAL_SCAN_DENOMINATOR = 8L
    private const val DIRECT_FULL_MAX_EXPONENT = 0.5f
}
