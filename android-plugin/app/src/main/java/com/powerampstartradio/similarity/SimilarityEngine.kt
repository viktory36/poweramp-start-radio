package com.powerampstartradio.similarity

import com.powerampstartradio.data.EmbeddedTrack
import com.powerampstartradio.data.EmbeddingDatabase
import com.powerampstartradio.data.EmbeddingModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Result of a similarity search.
 */
data class SimilarTrack(
    val track: EmbeddedTrack,
    val similarity: Float,
    val model: EmbeddingModel = EmbeddingModel.MUQ
)

/**
 * Engine for computing audio similarity using pre-computed embeddings.
 *
 * Since embeddings are L2-normalized, cosine similarity is just the dot product.
 * Supports multiple embedding models with per-model caches and interleaved results.
 */
class SimilarityEngine(
    private val database: EmbeddingDatabase
) {
    // Per-model cache of embeddings
    private val caches = mutableMapOf<EmbeddingModel, Map<Long, FloatArray>>()

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
     * Load embeddings for a specific model into memory.
     */
    suspend fun loadEmbeddings(model: EmbeddingModel = EmbeddingModel.MUQ) = withContext(Dispatchers.IO) {
        if (model !in caches) {
            caches[model] = database.getAllEmbeddings(model)
        }
    }

    /**
     * Find the most similar tracks to a seed track using a specific model.
     *
     * @param seedTrackId The ID of the seed track in the embedding database
     * @param topN Number of similar tracks to return
     * @param excludeSeed Whether to exclude the seed track from results
     * @param model Which embedding model to use
     * @return List of similar tracks sorted by similarity (highest first)
     */
    suspend fun findSimilarTracks(
        seedTrackId: Long,
        topN: Int = 50,
        excludeSeed: Boolean = true,
        model: EmbeddingModel = EmbeddingModel.MUQ
    ): List<SimilarTrack> = withContext(Dispatchers.Default) {
        // Ensure embeddings are loaded
        if (model !in caches) {
            loadEmbeddings(model)
        }

        val embeddings = caches[model] ?: return@withContext emptyList()
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
                SimilarTrack(track, similarity, model)
            }
        }
    }

    /**
     * Find similar tracks using both models and interleave results.
     *
     * Fetches top N from each available model, then interleaves:
     * MuQ#1, MuLan#1, MuQ#2, MuLan#2, ...
     * Skips duplicates (same track from different models).
     * Falls back to single-model if seed has no embedding for one model.
     *
     * @param seedTrackId The ID of the seed track
     * @param requestedCount Total number of tracks desired in the result
     * @return Interleaved list of similar tracks from both models
     */
    suspend fun findSimilarTracksDual(
        seedTrackId: Long,
        requestedCount: Int = 50
    ): List<SimilarTrack> = withContext(Dispatchers.Default) {
        val availableModels = database.getAvailableModels()

        // Load caches for all available models
        for (model in availableModels) {
            if (model !in caches) {
                loadEmbeddings(model)
            }
        }

        // Get results from each model that has the seed track
        val perModelResults = mutableMapOf<EmbeddingModel, List<SimilarTrack>>()
        for (model in availableModels) {
            val cache = caches[model] ?: continue
            if (seedTrackId !in cache) continue

            val results = findSimilarTracks(
                seedTrackId = seedTrackId,
                topN = requestedCount,
                excludeSeed = true,
                model = model
            )
            if (results.isNotEmpty()) {
                perModelResults[model] = results
            }
        }

        if (perModelResults.isEmpty()) {
            return@withContext emptyList()
        }

        // Single model fallback
        if (perModelResults.size == 1) {
            return@withContext perModelResults.values.first()
        }

        // Interleave results from all models
        val interleaved = mutableListOf<SimilarTrack>()
        val seenTrackIds = mutableSetOf<Long>()
        val iterators = perModelResults.values.map { it.iterator() }.toMutableList()

        while (interleaved.size < requestedCount && iterators.any { it.hasNext() }) {
            for (iter in iterators) {
                if (interleaved.size >= requestedCount) break
                // Take the next unseen track from this model
                while (iter.hasNext()) {
                    val candidate = iter.next()
                    if (candidate.track.id !in seenTrackIds) {
                        seenTrackIds.add(candidate.track.id)
                        interleaved.add(candidate)
                        break
                    }
                }
            }
        }

        interleaved
    }

    /**
     * Find similar tracks to a given embedding vector.
     * Useful when the seed track isn't in the database.
     */
    suspend fun findSimilarToEmbedding(
        embedding: FloatArray,
        topN: Int = 50,
        model: EmbeddingModel = EmbeddingModel.MUQ
    ): List<SimilarTrack> = withContext(Dispatchers.Default) {
        if (model !in caches) {
            loadEmbeddings(model)
        }

        val embeddings = caches[model] ?: return@withContext emptyList()

        val similarities = mutableListOf<Pair<Long, Float>>()
        for ((trackId, trackEmbedding) in embeddings) {
            val similarity = cosineSimilarity(embedding, trackEmbedding)
            similarities.add(trackId to similarity)
        }

        similarities.sortByDescending { it.second }
        val topSimilar = similarities.take(topN)

        topSimilar.mapNotNull { (trackId, similarity) ->
            database.getTrackById(trackId)?.let { track ->
                SimilarTrack(track, similarity, model)
            }
        }
    }

    /**
     * Clear all embeddings caches to free memory.
     */
    fun clearCache() {
        caches.clear()
    }

    /**
     * Get the total number of embeddings across all cached models.
     */
    fun getCacheSize(): Int = caches.values.sumOf { it.size }
}
