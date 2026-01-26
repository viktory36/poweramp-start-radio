package com.powerampstartradio.similarity

import com.powerampstartradio.data.EmbeddedTrack
import com.powerampstartradio.data.EmbeddingDatabase
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Result of a similarity search.
 */
data class SimilarTrack(
    val track: EmbeddedTrack,
    val similarity: Float
)

/**
 * Engine for computing audio similarity using pre-computed embeddings.
 *
 * Since embeddings are L2-normalized, cosine similarity is just the dot product.
 */
class SimilarityEngine(
    private val database: EmbeddingDatabase
) {
    // Cache of all embeddings for fast similarity computation
    private var embeddingsCache: Map<Long, FloatArray>? = null

    /**
     * Compute cosine similarity between two L2-normalized vectors.
     * Since they're normalized, this is just the dot product.
     */
    private fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
        var dot = 0f
        for (i in a.indices) {
            dot += a[i] * b[i]
        }
        return dot
    }

    /**
     * Load all embeddings into memory for fast similarity computation.
     * Call this once before performing searches.
     */
    suspend fun loadEmbeddings() = withContext(Dispatchers.IO) {
        if (embeddingsCache == null) {
            embeddingsCache = database.getAllEmbeddings()
        }
    }

    /**
     * Find the most similar tracks to a seed track.
     *
     * @param seedTrackId The ID of the seed track in the embedding database
     * @param topN Number of similar tracks to return
     * @param excludeSeed Whether to exclude the seed track from results
     * @return List of similar tracks sorted by similarity (highest first)
     */
    suspend fun findSimilarTracks(
        seedTrackId: Long,
        topN: Int = 50,
        excludeSeed: Boolean = true
    ): List<SimilarTrack> = withContext(Dispatchers.Default) {
        // Ensure embeddings are loaded
        if (embeddingsCache == null) {
            loadEmbeddings()
        }

        val embeddings = embeddingsCache ?: return@withContext emptyList()
        val seedEmbedding = embeddings[seedTrackId] ?: return@withContext emptyList()

        // Compute similarities
        val similarities = mutableListOf<Pair<Long, Float>>()
        for ((trackId, embedding) in embeddings) {
            if (excludeSeed && trackId == seedTrackId) continue
            val similarity = cosineSimilarity(seedEmbedding, embedding)
            similarities.add(trackId to similarity)
        }

        // Sort by similarity and take top N
        similarities.sortByDescending { it.second }
        val topSimilar = similarities.take(topN)

        // Convert to SimilarTrack objects
        topSimilar.mapNotNull { (trackId, similarity) ->
            database.getTrackById(trackId)?.let { track ->
                SimilarTrack(track, similarity)
            }
        }
    }

    /**
     * Find similar tracks to a given embedding vector.
     * Useful when the seed track isn't in the database.
     */
    suspend fun findSimilarToEmbedding(
        embedding: FloatArray,
        topN: Int = 50
    ): List<SimilarTrack> = withContext(Dispatchers.Default) {
        if (embeddingsCache == null) {
            loadEmbeddings()
        }

        val embeddings = embeddingsCache ?: return@withContext emptyList()

        val similarities = mutableListOf<Pair<Long, Float>>()
        for ((trackId, trackEmbedding) in embeddings) {
            val similarity = cosineSimilarity(embedding, trackEmbedding)
            similarities.add(trackId to similarity)
        }

        similarities.sortByDescending { it.second }
        val topSimilar = similarities.take(topN)

        topSimilar.mapNotNull { (trackId, similarity) ->
            database.getTrackById(trackId)?.let { track ->
                SimilarTrack(track, similarity)
            }
        }
    }

    /**
     * Clear the embeddings cache to free memory.
     */
    fun clearCache() {
        embeddingsCache = null
    }

    /**
     * Get the number of tracks in the cache.
     */
    fun getCacheSize(): Int = embeddingsCache?.size ?: 0
}
