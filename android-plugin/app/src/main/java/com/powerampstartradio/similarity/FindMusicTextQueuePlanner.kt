package com.powerampstartradio.similarity

import com.powerampstartradio.data.EmbeddingIndex
import com.powerampstartradio.similarity.algorithms.DppSelector
import com.powerampstartradio.ui.FindMusicTextResultPlanner
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest

/** One selected row with its unchanged rank in the complete text-cosine objective. */
data class FindMusicTextQueueSelection(
    val trackId: Long,
    val textSimilarity: Float,
    val originalTextObjectiveRank: Int,
)

/** Exact planner input/output and any full-domain DPP proof retained with a displayed queue. */
data class FindMusicTextQueuePlanEvidence(
    val planner: FindMusicTextResultPlanner,
    val plannerVersion: Int,
    val completeTextRankingSha256: String,
    val completeCandidateDomainCount: Int,
    val requestedResultCount: Int,
    val orderedSelectedTrackIds: List<Long>,
    val orderedOriginalTextObjectiveRanks: List<Int>,
    val dppSelection: DppSelectionEvidence? = null,
) {
    init {
        requireValid()
    }

    fun requireValid() {
        require(plannerVersion == planner.currentVersion) {
            "Find Music text planner version does not match its planner"
        }
        require(completeTextRankingSha256.matches(SHA256)) {
            "Complete text-ranking fingerprint is invalid"
        }
        require(completeCandidateDomainCount > 0) {
            "Complete text candidate domain must be non-empty"
        }
        require(requestedResultCount > 0) { "Requested text result count must be positive" }
        require(orderedSelectedTrackIds.size == orderedOriginalTextObjectiveRanks.size) {
            "Selected text track IDs and original ranks do not align"
        }
        require(orderedSelectedTrackIds.size <= requestedResultCount) {
            "Selected text result count exceeds the request"
        }
        require(orderedSelectedTrackIds.toSet().size == orderedSelectedTrackIds.size) {
            "Selected text track IDs are not unique"
        }
        require(
            orderedOriginalTextObjectiveRanks.all { it in 1..completeCandidateDomainCount } &&
                orderedOriginalTextObjectiveRanks.toSet().size ==
                orderedOriginalTextObjectiveRanks.size,
        ) { "Original text objective ranks are invalid or duplicated" }
        when (planner) {
            FindMusicTextResultPlanner.CLOSEST -> {
                require(dppSelection == null) { "Closest text results cannot carry a DPP proof" }
                require(orderedOriginalTextObjectiveRanks.zipWithNext().all { (a, b) -> a < b }) {
                    "Closest text objective ranks must remain ordered"
                }
            }
            FindMusicTextResultPlanner.VARIED_DPP -> {
                val proof = requireNotNull(dppSelection) {
                    "Varied text results require a full-domain DPP proof"
                }
                require(proof.completeCandidateDomainCount == completeCandidateDomainCount) {
                    "Varied text DPP proof does not cover the complete candidate domain"
                }
                proof.requireValid(orderedSelectedTrackIds.size)
            }
            FindMusicTextResultPlanner.VARIED_ALL_OF_DPP ->
                error("All-of Varied evidence cannot use the text queue plan")
        }
    }

    private companion object {
        val SHA256 = Regex("^[0-9a-f]{64}$")
    }
}

data class FindMusicTextQueuePlan(
    val selections: List<FindMusicTextQueueSelection>,
    val evidence: FindMusicTextQueuePlanEvidence,
) {
    init {
        require(selections.map(FindMusicTextQueueSelection::trackId) ==
            evidence.orderedSelectedTrackIds
        ) { "Text queue selections disagree with their ordered evidence" }
        require(selections.map(FindMusicTextQueueSelection::originalTextObjectiveRank) ==
            evidence.orderedOriginalTextObjectiveRanks
        ) { "Text queue objective ranks disagree with their ordered evidence" }
    }
}

/**
 * Plans simple-positive-text result membership from the complete active-domain cosine ranking.
 *
 * Varied is canonical greedy DPP over that complete domain. Its initial workspace is only an
 * exact-search optimization: every returned sequence carries a strict unseen-gain certificate
 * proving it is the same sequence a materialized full-domain run would produce.
 */
object FindMusicTextQueuePlanner {
    fun plan(
        planner: FindMusicTextResultPlanner,
        completeRelevanceRanking: List<RankedSimilarity>,
        requestedResultCount: Int,
        embeddingIndex: EmbeddingIndex,
        cancellationCheck: (() -> Unit)? = null,
    ): FindMusicTextQueuePlan {
        require(completeRelevanceRanking.isNotEmpty()) {
            "Complete text relevance ranking must be non-empty"
        }
        require(requestedResultCount > 0) { "Requested text result count must be positive" }
        require(completeRelevanceRanking.all { it.score.isFinite() }) {
            "Complete text relevance ranking contains a non-finite score"
        }
        require(completeRelevanceRanking.map(RankedSimilarity::trackId).toSet().size ==
            completeRelevanceRanking.size
        ) { "Complete text relevance ranking contains duplicate track IDs" }
        require(completeRelevanceRanking.zipWithNext().all { (a, b) -> a.score >= b.score }) {
            "Complete text relevance ranking is not ordered by descending cosine"
        }
        require(planner != FindMusicTextResultPlanner.VARIED_ALL_OF_DPP) {
            "All-of Varied must use FindMusicAllOfQueuePlanner"
        }
        cancellationCheck?.invoke()

        val selections: List<FindMusicTextQueueSelection>
        val dppEvidence: DppSelectionEvidence?
        when (planner) {
            FindMusicTextResultPlanner.CLOSEST -> {
                selections = completeRelevanceRanking
                    .take(requestedResultCount)
                    .mapIndexed { index, row ->
                        FindMusicTextQueueSelection(
                            trackId = row.trackId,
                            textSimilarity = row.score,
                            originalTextObjectiveRank = index + 1,
                        )
                    }
                dppEvidence = null
            }
            FindMusicTextResultPlanner.VARIED_DPP -> {
                val candidates = completeRelevanceRanking.map { it.trackId to it.score }
                val initialWorkingCount = initialWorkingCandidateCount(
                    completeDomainCount = candidates.size,
                    requestedResultCount = requestedResultCount,
                )
                val certified = DppSelector.selectBatchCertified(
                    candidates = candidates,
                    numSelect = requestedResultCount,
                    index = embeddingIndex,
                    initialCandidateCount = initialWorkingCount,
                    qualityExponent = VARIED_QUALITY_EXPONENT,
                    cancellationCheck = cancellationCheck,
                )
                selections = certified.tracks.map { selected ->
                    FindMusicTextQueueSelection(
                        trackId = selected.trackId,
                        textSimilarity = selected.score,
                        originalTextObjectiveRank = selected.candidateRank,
                    )
                }
                dppEvidence = DppSelectionEvidence(
                    completeCandidateDomainCount = certified.evidence.totalCandidateCount,
                    initialWorkingCandidateCount = certified.evidence.initialCandidateCount,
                    attemptedCandidateCounts = certified.evidence.attemptedCandidateCounts,
                    finalWorkingCandidateCount = certified.evidence.finalCandidateCount,
                    selectedMarginalGains = certified.evidence.selectedMarginalGains,
                    finalUnseenInitialGainUpperBound =
                        certified.evidence.finalUnseenGainUpperBound,
                    usedCompleteCandidateDomain = certified.evidence.usedFullDomain,
                    reproducedFullDomainGreedySequence = true,
                )
            }
            FindMusicTextResultPlanner.VARIED_ALL_OF_DPP ->
                error("All-of Varied must use FindMusicAllOfQueuePlanner")
        }
        cancellationCheck?.invoke()

        val evidence = FindMusicTextQueuePlanEvidence(
            planner = planner,
            plannerVersion = planner.currentVersion,
            completeTextRankingSha256 = rankingFingerprint(completeRelevanceRanking),
            completeCandidateDomainCount = completeRelevanceRanking.size,
            requestedResultCount = requestedResultCount,
            orderedSelectedTrackIds = selections.map(FindMusicTextQueueSelection::trackId),
            orderedOriginalTextObjectiveRanks = selections.map(
                FindMusicTextQueueSelection::originalTextObjectiveRank,
            ),
            dppSelection = dppEvidence,
        )
        return FindMusicTextQueuePlan(selections, evidence)
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

    private fun rankingFingerprint(ranking: List<RankedSimilarity>): String {
        val digest = MessageDigest.getInstance("SHA-256")
        digest.update(RANKING_FINGERPRINT_DOMAIN)
        val buffer = ByteBuffer.allocate(Long.SIZE_BYTES + Int.SIZE_BYTES)
            .order(ByteOrder.BIG_ENDIAN)
        ranking.forEach { row ->
            buffer.clear()
            buffer.putLong(row.trackId)
            buffer.putInt(row.score.toRawBits())
            digest.update(buffer.array())
        }
        return digest.digest().joinToString("") { byte -> "%02x".format(byte.toInt() and 0xff) }
    }

    private const val VARIED_QUALITY_EXPONENT = 4f
    private const val MIN_INITIAL_WORKING_CANDIDATES = 50
    private val RANKING_FINGERPRINT_DOMAIN =
        "find-music-complete-text-ranking-v1\u0000".toByteArray(Charsets.US_ASCII)
}
