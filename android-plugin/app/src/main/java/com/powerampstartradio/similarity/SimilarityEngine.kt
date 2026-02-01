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
 * Supports multiple embedding models. Only one model's embeddings are held in
 * memory at a time to stay within Android heap limits (~300-400 MB).
 */
class SimilarityEngine(
    private val database: EmbeddingDatabase
) {
    // Only one model's cache in memory at a time
    private var cachedModel: EmbeddingModel? = null
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
     * Load embeddings for a specific model into memory.
     * Evicts any previously loaded model's cache first.
     */
    suspend fun loadEmbeddings(model: EmbeddingModel = EmbeddingModel.MUQ) = withContext(Dispatchers.IO) {
        if (cachedModel != model) {
            // Evict previous cache to free memory before loading new one
            embeddingsCache = null
            cachedModel = null
            embeddingsCache = database.getAllEmbeddings(model)
            cachedModel = model
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
        // Ensure correct model is loaded
        loadEmbeddings(model)

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
                SimilarTrack(track, similarity, model)
            }
        }
    }

    /**
     * Find similar tracks using both models and interleave results.
     *
     * Processes one model at a time to stay within heap limits:
     * 1. Load MuQ cache, compute top-N, free cache
     * 2. Load MuLan cache, compute top-N, free cache
     * 3. Interleave: MuQ#1, MuLan#1, MuQ#2, MuLan#2, ...
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

        // Get results from each model sequentially (one cache at a time)
        val perModelResults = mutableMapOf<EmbeddingModel, List<SimilarTrack>>()
        for (model in availableModels) {
            val results = findSimilarTracks(
                seedTrackId = seedTrackId,
                topN = requestedCount,
                excludeSeed = true,
                model = model
            )
            if (results.isNotEmpty()) {
                perModelResults[model] = results
            }
            // findSimilarTracks calls loadEmbeddings which evicts the previous cache,
            // so only one model's data is in memory at any time.
        }

        // Free the last model's cache now that we have our result lists
        clearCache()

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
     * Clear the embeddings cache to free memory.
     */
    fun clearCache() {
        embeddingsCache = null
        cachedModel = null
    }

    /**
     * Get the number of embeddings in the current cache.
     */
    fun getCacheSize(): Int = embeddingsCache?.size ?: 0
}
