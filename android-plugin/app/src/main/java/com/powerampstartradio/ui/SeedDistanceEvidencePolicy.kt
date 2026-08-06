package com.powerampstartradio.ui

import com.powerampstartradio.poweramp.TrackMatcher

data class SeedDistanceEvidence(
    val seedRank: Int,
    val rankingIdentityCount: Int,
)

/** Fail-closed presentation of exact single-song seed-relative evidence. */
object SeedDistanceEvidencePolicy {
    fun evidenceOrNull(
        session: RadioResult?,
        track: QueuedTrackResult,
    ): SeedDistanceEvidence? {
        if (session == null || session.isDirectQueue || session.composedQuerySpec != null ||
            session.composedContract != null ||
            session.matchType == TrackMatcher.MatchType.COMPOSED_QUERY ||
            session.matchType == TrackMatcher.MatchType.NOT_APPLICABLE ||
            track.seedRank == null || track.findMusicEvidence != null ||
            track.composedEvidence != null
        ) {
            return null
        }
        val seedRank = track.seedRank
        val rankingIdentityCount = (
            session.seedRankingIdentityCount
                ?: session.legacyAllDatesRankingIdentityCount()
            )
            ?.takeIf { seedRank in 1..it }
            ?: return null
        val rawCosine = track.similarityToSeed
        if (!rawCosine.isFinite() || rawCosine !in -COSINE_TOLERANCE..COSINE_TOLERANCE) {
            return null
        }
        return SeedDistanceEvidence(
            seedRank = seedRank,
            rankingIdentityCount = rankingIdentityCount,
        )
    }

    private const val COSINE_TOLERANCE = 1.001f

    /**
     * Before separate rank evidence existed, the candidate count had the same meaning only when no
     * added-date filter existed. A missing legacy enum is also an all-dates record from before that
     * control was introduced.
     */
    private fun RadioResult.legacyAllDatesRankingIdentityCount(): Int? {
        if (config.libraryAddedDays != null) return null
        val legacyRange: LibraryAddedRange? = config.libraryAddedRange
        if (legacyRange != null && legacyRange != LibraryAddedRange.ALL_DATES) return null
        return eligibleCandidateIdentityCount
    }
}
