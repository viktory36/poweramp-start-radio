package com.powerampstartradio.ui

import com.powerampstartradio.data.EmbeddingDatabase
import com.powerampstartradio.data.EmbeddingIndex
import com.powerampstartradio.data.StableTrackIdentityCatalog
import com.powerampstartradio.similarity.ActiveRecommendationDomain
import com.powerampstartradio.similarity.StableSimilarityTopK
import com.powerampstartradio.similarity.RankedSimilarity

internal object FavoritesRanking {
    fun nearest(
        embedding: FloatArray,
        seedTrackId: Long?,
        limit: Int,
        index: EmbeddingIndex,
        domain: ActiveRecommendationDomain,
        identities: StableTrackIdentityCatalog,
        database: EmbeddingDatabase,
        addedTrackIds: Set<Long>,
    ): List<FavoriteMatch> = rank(embedding, seedTrackId, limit, index, domain, identities)
        .mapIndexed { position, row ->
            FavoriteMatch(
                track = checkNotNull(database.getTrackById(row.trackId)),
                rank = position + 1,
                isNew = isNewRecording(row.trackId, identities, addedTrackIds),
                stableTrackSpanId = identities.stableTrackSpanId(row.trackId),
            )
        }

    fun hasNewNearest(
        embedding: FloatArray,
        seedTrackId: Long?,
        limit: Int,
        index: EmbeddingIndex,
        domain: ActiveRecommendationDomain,
        identities: StableTrackIdentityCatalog,
        addedTrackIds: Set<Long>,
    ): Boolean = addedTrackIds.isNotEmpty() &&
        rank(embedding, seedTrackId, limit, index, domain, identities)
            .any { isNewRecording(it.trackId, identities, addedTrackIds) }

    private fun isNewRecording(
        trackId: Long,
        identities: StableTrackIdentityCatalog,
        addedTrackIds: Set<Long>,
    ): Boolean = identities.equivalentVisibleTrackIds(trackId).all { it in addedTrackIds }

    private fun rank(
        embedding: FloatArray,
        seedTrackId: Long?,
        limit: Int,
        index: EmbeddingIndex,
        domain: ActiveRecommendationDomain,
        identities: StableTrackIdentityCatalog,
    ): List<RankedSimilarity> {
        require(limit in 1..1_000)
        val candidates = domain.candidateDomain(null)
        val seedCopies = seedTrackId?.let(identities::equivalentVisibleTrackIds).orEmpty().toSet()
        val allIds = candidates.orderedIdentityRepresentativeTrackIds()
        val scores = candidates.identityRepresentativeScoresFromFull(index.computeAllSimilarities(embedding))
        return StableSimilarityTopK.select(
            orderedTrackIds = allIds,
            similarities = scores,
            topK = limit,
            rankingTieKey = identities::rankingTieKey,
            excludeIds = seedCopies,
        )
    }
}
