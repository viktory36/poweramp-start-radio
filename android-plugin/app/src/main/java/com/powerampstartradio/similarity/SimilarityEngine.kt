package com.powerampstartradio.similarity

import android.util.Log
import com.powerampstartradio.data.EmbeddedTrack
import com.powerampstartradio.data.EmbeddingDatabase
import com.powerampstartradio.data.EmbeddingIndex
import com.powerampstartradio.data.EmbeddingModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

/**
 * Result of a similarity search.
 */
data class SimilarTrack(
    val track: EmbeddedTrack,
    val similarity: Float,
    val model: EmbeddingModel = EmbeddingModel.MULAN
)

/**
 * User-selectable search strategy.
 */
enum class SearchStrategy {
    MULAN_ONLY,
    FLAMINGO_ONLY,
    INTERLEAVE,
    FEED_FORWARD
}

/**
 * Configuration for the feed-forward strategy.
 */
data class FeedForwardConfig(
    val primaryModel: EmbeddingModel,   // Which model finds anchors
    val expansionCount: Int = 3         // How many secondary results per anchor
) {
    val secondaryModel: EmbeddingModel
        get() = when (primaryModel) {
            EmbeddingModel.MULAN -> EmbeddingModel.FLAMINGO
            EmbeddingModel.FLAMINGO -> EmbeddingModel.MULAN
            else -> EmbeddingModel.FLAMINGO
        }
}

/**
 * Engine for computing audio similarity using mmap'd embedding indices.
 *
 * Embeddings are stored in .emb binary files (extracted from SQLite on first use)
 * and memory-mapped so the OS pages data on demand. This allows both MuLan (~150 MB)
 * and Flamingo (~1 GB) indices to be "loaded" simultaneously without heap pressure.
 */
class SimilarityEngine(
    private val database: EmbeddingDatabase,
    private val filesDir: File
) {
    companion object {
        private const val TAG = "SimilarityEngine"
    }

    private var mulanIndex: EmbeddingIndex? = null
    private var flamingoIndex: EmbeddingIndex? = null

    /**
     * Ensure mmap'd indices are ready. Extracts from SQLite if needed.
     */
    suspend fun ensureIndices() = withContext(Dispatchers.IO) {
        val availableModels = database.getAvailableModels()
        val dbFile = File(filesDir, "embeddings.db")
        val dbModified = dbFile.lastModified()

        if (EmbeddingModel.MULAN in availableModels) {
            val embFile = File(filesDir, "mulan.emb")
            if (!embFile.exists() || embFile.lastModified() < dbModified) {
                Log.d(TAG, "Extracting MuLan index...")
                EmbeddingIndex.extractFromDatabase(database, EmbeddingModel.MULAN, embFile)
            }
            if (mulanIndex == null) {
                mulanIndex = EmbeddingIndex.mmap(embFile)
                Log.d(TAG, "MuLan index: ${mulanIndex!!.numTracks} tracks, dim=${mulanIndex!!.dim}")
            }
        }

        if (EmbeddingModel.FLAMINGO in availableModels) {
            val embFile = File(filesDir, "flamingo.emb")
            if (!embFile.exists() || embFile.lastModified() < dbModified) {
                Log.d(TAG, "Extracting Flamingo index...")
                EmbeddingIndex.extractFromDatabase(database, EmbeddingModel.FLAMINGO, embFile)
            }
            if (flamingoIndex == null) {
                flamingoIndex = EmbeddingIndex.mmap(embFile)
                Log.d(TAG, "Flamingo index: ${flamingoIndex!!.numTracks} tracks, dim=${flamingoIndex!!.dim}")
            }
        }
    }

    private fun getIndex(model: EmbeddingModel): EmbeddingIndex? {
        return when (model) {
            EmbeddingModel.MULAN -> mulanIndex
            EmbeddingModel.FLAMINGO -> flamingoIndex
            else -> null
        }
    }

    /**
     * Find similar tracks using the specified strategy.
     */
    suspend fun findSimilarTracks(
        seedTrackId: Long,
        numTracks: Int,
        strategy: SearchStrategy,
        feedForwardConfig: FeedForwardConfig? = null
    ): List<SimilarTrack> = withContext(Dispatchers.Default) {
        ensureIndices()

        when (strategy) {
            SearchStrategy.MULAN_ONLY ->
                singleModelSearch(seedTrackId, numTracks, EmbeddingModel.MULAN)
            SearchStrategy.FLAMINGO_ONLY ->
                singleModelSearch(seedTrackId, numTracks, EmbeddingModel.FLAMINGO)
            SearchStrategy.INTERLEAVE ->
                interleaveSearch(seedTrackId, numTracks)
            SearchStrategy.FEED_FORWARD ->
                feedForwardSearch(seedTrackId, numTracks, feedForwardConfig!!)
        }
    }

    /**
     * Single model: findTopK on one index.
     */
    private fun singleModelSearch(
        seedTrackId: Long,
        numTracks: Int,
        model: EmbeddingModel
    ): List<SimilarTrack> {
        val index = getIndex(model) ?: run {
            Log.e(TAG, "No index available for ${model.name}")
            return emptyList()
        }

        val seedEmbedding = index.getEmbeddingByTrackId(seedTrackId) ?: run {
            Log.e(TAG, "Seed track $seedTrackId has no ${model.name} embedding")
            return emptyList()
        }

        val topK = index.findTopK(seedEmbedding, numTracks, excludeIds = setOf(seedTrackId))

        return topK.mapNotNull { (trackId, score) ->
            database.getTrackById(trackId)?.let { track ->
                SimilarTrack(track, score, model)
            }
        }
    }

    /**
     * Interleave: findTopK on both models, round-robin merge with dedup.
     */
    private fun interleaveSearch(
        seedTrackId: Long,
        numTracks: Int
    ): List<SimilarTrack> {
        val mulanResults = mulanIndex?.let { idx ->
            idx.getEmbeddingByTrackId(seedTrackId)?.let { seed ->
                idx.findTopK(seed, numTracks, excludeIds = setOf(seedTrackId))
                    .mapNotNull { (id, score) ->
                        database.getTrackById(id)?.let { SimilarTrack(it, score, EmbeddingModel.MULAN) }
                    }
            }
        } ?: emptyList()

        val flamingoResults = flamingoIndex?.let { idx ->
            idx.getEmbeddingByTrackId(seedTrackId)?.let { seed ->
                idx.findTopK(seed, numTracks, excludeIds = setOf(seedTrackId))
                    .mapNotNull { (id, score) ->
                        database.getTrackById(id)?.let { SimilarTrack(it, score, EmbeddingModel.FLAMINGO) }
                    }
            }
        } ?: emptyList()

        // Single model fallback
        if (mulanResults.isEmpty()) return flamingoResults.take(numTracks)
        if (flamingoResults.isEmpty()) return mulanResults.take(numTracks)

        // Round-robin interleave with dedup
        val interleaved = mutableListOf<SimilarTrack>()
        val seenTrackIds = mutableSetOf<Long>()
        val mulanIter = mulanResults.iterator()
        val flamingoIter = flamingoResults.iterator()

        while (interleaved.size < numTracks && (mulanIter.hasNext() || flamingoIter.hasNext())) {
            // MuLan turn
            while (mulanIter.hasNext() && interleaved.size < numTracks) {
                val candidate = mulanIter.next()
                if (candidate.track.id !in seenTrackIds) {
                    seenTrackIds.add(candidate.track.id)
                    interleaved.add(candidate)
                    break
                }
            }
            // Flamingo turn
            while (flamingoIter.hasNext() && interleaved.size < numTracks) {
                val candidate = flamingoIter.next()
                if (candidate.track.id !in seenTrackIds) {
                    seenTrackIds.add(candidate.track.id)
                    interleaved.add(candidate)
                    break
                }
            }
        }

        return interleaved
    }

    /**
     * Feed-forward: primary model finds anchors, secondary model expands each.
     *
     * Algorithm:
     * 1. Primary index: findTopK(seedEmbedding, numGroups) → anchor tracks
     * 2. For each anchor: look up its embedding in secondary index
     * 3. Secondary index: findTopKMulti(anchorEmbeddings, expansionCount) — single corpus pass
     * 4. Assemble: [anchor1, expansion1a, expansion1b, ..., anchor2, ...]
     */
    private fun feedForwardSearch(
        seedTrackId: Long,
        numTracks: Int,
        config: FeedForwardConfig
    ): List<SimilarTrack> {
        val primaryIndex = getIndex(config.primaryModel) ?: run {
            Log.e(TAG, "No index for primary model ${config.primaryModel.name}")
            return emptyList()
        }
        val secondaryIndex = getIndex(config.secondaryModel) ?: run {
            Log.e(TAG, "No index for secondary model ${config.secondaryModel.name}")
            // Fall back to single-model search with primary
            return singleModelSearch(seedTrackId, numTracks, config.primaryModel)
        }

        val seedEmbedding = primaryIndex.getEmbeddingByTrackId(seedTrackId) ?: run {
            Log.e(TAG, "Seed track $seedTrackId has no ${config.primaryModel.name} embedding")
            return emptyList()
        }

        // Calculate how many anchor groups we need
        val groupSize = 1 + config.expansionCount  // anchor + expansions
        val numGroups = (numTracks + groupSize - 1) / groupSize

        // Step 1: Find anchor tracks with primary model
        val anchors = primaryIndex.findTopK(
            seedEmbedding, numGroups, excludeIds = setOf(seedTrackId)
        )

        if (anchors.isEmpty()) return emptyList()

        // Step 2: Collect secondary embeddings for each anchor
        val anchorQueries = mutableMapOf<Long, FloatArray>()
        for ((anchorId, _) in anchors) {
            secondaryIndex.getEmbeddingByTrackId(anchorId)?.let { emb ->
                anchorQueries[anchorId] = emb
            }
        }

        // Step 3: Single corpus pass — find expansions for all anchors at once
        val seen = mutableSetOf(seedTrackId)
        seen.addAll(anchors.map { it.first })

        val expansionResults = if (anchorQueries.isNotEmpty()) {
            secondaryIndex.findTopKMulti(anchorQueries, config.expansionCount, excludeIds = seen)
        } else {
            emptyMap()
        }

        // Step 4: Assemble results in group order
        val result = mutableListOf<SimilarTrack>()

        for ((anchorId, anchorScore) in anchors) {
            if (result.size >= numTracks) break

            // Add anchor
            database.getTrackById(anchorId)?.let { track ->
                result.add(SimilarTrack(track, anchorScore, config.primaryModel))
            }

            // Add expansions
            val expansions = expansionResults[anchorId] ?: emptyList()
            for ((expansionId, expansionScore) in expansions) {
                if (result.size >= numTracks) break
                if (expansionId in seen) continue
                seen.add(expansionId)

                database.getTrackById(expansionId)?.let { track ->
                    result.add(SimilarTrack(track, expansionScore, config.secondaryModel))
                }
            }
        }

        return result
    }
}
