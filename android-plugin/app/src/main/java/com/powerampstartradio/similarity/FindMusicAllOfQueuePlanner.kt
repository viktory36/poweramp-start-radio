package com.powerampstartradio.similarity

import com.powerampstartradio.data.EmbeddingIndex
import com.powerampstartradio.similarity.algorithms.ComposedRankingRow
import com.powerampstartradio.similarity.algorithms.DppSelector
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest

/** One Varied All-of selection with its unchanged rank and evidence in the base objective. */
data class FindMusicAllOfQueueSelection(
    val row: ComposedRankingRow,
    val originalAllOfObjectiveRank: Int,
)

/** Complete objective identity and full-domain proof behind one displayed Varied All-of set. */
data class FindMusicAllOfQueuePlanEvidence(
    val plannerVersion: Int,
    val completeAllOfRankingSha256: String,
    val completeCandidateDomainCount: Int,
    val requestedResultCount: Int,
    val orderedSelectedTrackIds: List<Long>,
    val orderedOriginalAllOfObjectiveRanks: List<Int>,
    val dppSelection: DppSelectionEvidence,
) {
    init {
        requireValid()
    }

    fun requireValid() {
        require(plannerVersion == FindMusicAllOfQueuePlanner.PLANNER_VERSION)
        require(completeAllOfRankingSha256.matches(SHA256)) {
            "Complete All-of ranking fingerprint is invalid"
        }
        require(completeCandidateDomainCount > 0)
        require(requestedResultCount > 0)
        require(orderedSelectedTrackIds.size == orderedOriginalAllOfObjectiveRanks.size)
        require(orderedSelectedTrackIds.size <= requestedResultCount)
        require(orderedSelectedTrackIds.toSet().size == orderedSelectedTrackIds.size)
        require(
            orderedOriginalAllOfObjectiveRanks.all { it in 1..completeCandidateDomainCount } &&
                orderedOriginalAllOfObjectiveRanks.toSet().size ==
                orderedOriginalAllOfObjectiveRanks.size,
        )
        require(dppSelection.completeCandidateDomainCount == completeCandidateDomainCount) {
            "Varied All-of DPP proof does not cover the complete objective domain"
        }
        dppSelection.requireValid(orderedSelectedTrackIds.size)
    }

    private companion object {
        val SHA256 = Regex("^[0-9a-f]{64}$")
    }
}

data class FindMusicAllOfQueuePlan(
    val selections: List<FindMusicAllOfQueueSelection>,
    val evidence: FindMusicAllOfQueuePlanEvidence,
) {
    init {
        require(
            selections.map { it.row.trackId } == evidence.orderedSelectedTrackIds,
        )
        require(
            selections.map(FindMusicAllOfQueueSelection::originalAllOfObjectiveRank) ==
                evidence.orderedOriginalAllOfObjectiveRanks,
        )
    }
}

/**
 * Selects a relevance-preserving varied membership from the exact complete All-of order.
 *
 * The fixed exponent is a versioned product contract, not a visible calibration control.
 * Complete-domain score normalization changes only the kernel's global scale, so it preserves
 * fixed-cardinality ordering while keeping the calculation numerically stable. Prefix growth is
 * only an execution optimization: every result carries the same unseen-gain certificate as text
 * Varied.
 */
object FindMusicAllOfQueuePlanner {
    const val PLANNER_VERSION = 2
    const val QUALITY_EXPONENT = 64f

    fun plan(
        completeObjectiveRanking: List<ComposedRankingRow>,
        requestedResultCount: Int,
        embeddingIndex: EmbeddingIndex,
        cancellationCheck: (() -> Unit)? = null,
        isEligible: (Long, List<Long>) -> Boolean = { _, _ -> true },
    ): FindMusicAllOfQueuePlan {
        require(completeObjectiveRanking.isNotEmpty())
        require(requestedResultCount > 0)
        require(completeObjectiveRanking.all { it.objectiveScore.isFinite() })
        require(
            completeObjectiveRanking.map(ComposedRankingRow::trackId).toSet().size ==
                completeObjectiveRanking.size,
        )
        require(
            completeObjectiveRanking.zipWithNext().all { (left, right) ->
                left.objectiveScore >= right.objectiveScore
            },
        )
        cancellationCheck?.invoke()

        val candidates = completeObjectiveRanking.map { it.trackId to it.objectiveScore }
        val initialWorkingCount = initialWorkingCandidateCount(
            completeDomainCount = candidates.size,
            requestedResultCount = requestedResultCount,
        )
        val certified = DppSelector.selectBatchCertified(
            candidates = candidates,
            numSelect = requestedResultCount,
            index = embeddingIndex,
            initialCandidateCount = initialWorkingCount,
            qualityExponent = QUALITY_EXPONENT,
            cancellationCheck = cancellationCheck,
            isEligible = isEligible,
        )
        val selections = certified.tracks.map { selected ->
            FindMusicAllOfQueueSelection(
                row = completeObjectiveRanking[selected.candidateRank - 1],
                originalAllOfObjectiveRank = selected.candidateRank,
            )
        }
        val dppEvidence = DppSelectionEvidence(
            completeCandidateDomainCount = certified.evidence.totalCandidateCount,
            initialWorkingCandidateCount = certified.evidence.initialCandidateCount,
            attemptedCandidateCounts = certified.evidence.attemptedCandidateCounts,
            finalWorkingCandidateCount = certified.evidence.finalCandidateCount,
            selectedMarginalGains = certified.evidence.selectedMarginalGains,
            finalUnseenInitialGainUpperBound = certified.evidence.finalUnseenGainUpperBound,
            usedCompleteCandidateDomain = certified.evidence.usedFullDomain,
            reproducedFullDomainGreedySequence = true,
        )
        cancellationCheck?.invoke()
        return FindMusicAllOfQueuePlan(
            selections = selections,
            evidence = FindMusicAllOfQueuePlanEvidence(
                plannerVersion = PLANNER_VERSION,
                completeAllOfRankingSha256 = rankingFingerprint(completeObjectiveRanking),
                completeCandidateDomainCount = completeObjectiveRanking.size,
                requestedResultCount = requestedResultCount,
                orderedSelectedTrackIds = selections.map { it.row.trackId },
                orderedOriginalAllOfObjectiveRanks = selections.map(
                    FindMusicAllOfQueueSelection::originalAllOfObjectiveRank,
                ),
                dppSelection = dppEvidence,
            ),
        )
    }

    private fun initialWorkingCandidateCount(
        completeDomainCount: Int,
        requestedResultCount: Int,
    ): Int {
        val twoPercentCeiling = Math.addExact(completeDomainCount, 49) / 50
        return maxOf(
            MIN_INITIAL_WORKING_CANDIDATES,
            requestedResultCount,
            twoPercentCeiling,
        ).coerceAtMost(completeDomainCount)
    }

    private fun rankingFingerprint(ranking: List<ComposedRankingRow>): String {
        val digest = MessageDigest.getInstance("SHA-256")
        digest.update(RANKING_FINGERPRINT_DOMAIN)
        val buffer = ByteBuffer.allocate(Long.SIZE_BYTES + Int.SIZE_BYTES)
            .order(ByteOrder.BIG_ENDIAN)
        ranking.forEach { row ->
            buffer.clear()
            buffer.putLong(row.trackId)
            buffer.putInt(row.objectiveScore.toRawBits())
            digest.update(buffer.array())
        }
        return digest.digest().joinToString("") { byte ->
            "%02x".format(byte.toInt() and 0xff)
        }
    }

    private const val MIN_INITIAL_WORKING_CANDIDATES = 50
    private val RANKING_FINGERPRINT_DOMAIN =
        "find-music-complete-all-of-ranking-v1\u0000".toByteArray(Charsets.US_ASCII)
}
