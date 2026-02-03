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
    ANCHOR_EXPAND
}

/**
 * Configuration for the Anchor & Expand strategy.
 */
data class AnchorExpandConfig(
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
     * Ensure mmap'd indices are ready. Extracts from SQLite if needed (one-time).
     *
     * @param onProgress called with a human-readable status message during extraction
     */
    suspend fun ensureIndices(
        onProgress: ((message: String) -> Unit)? = null
    ) = withContext(Dispatchers.IO) {
        val availableModels = database.getAvailableModels()
        val dbFile = File(filesDir, "embeddings.db")
        val dbModified = dbFile.lastModified()

        if (EmbeddingModel.MULAN in availableModels) {
            val embFile = File(filesDir, "mulan.emb")
            if (!embFile.exists() || embFile.lastModified() < dbModified) {
                Log.i(TAG, "Extracting MuLan index (one-time)...")
                onProgress?.invoke("Extracting MuLan index...")
                EmbeddingIndex.extractFromDatabase(database, EmbeddingModel.MULAN, embFile) { cur, total ->
                    onProgress?.invoke("Extracting MuLan: $cur / $total")
                }
            }
            if (mulanIndex == null) {
                mulanIndex = EmbeddingIndex.mmap(embFile)
                Log.i(TAG, "MuLan index: ${mulanIndex!!.numTracks} tracks, dim=${mulanIndex!!.dim}")
            }
        }

        if (EmbeddingModel.FLAMINGO in availableModels) {
            val embFile = File(filesDir, "flamingo.emb")
            if (!embFile.exists() || embFile.lastModified() < dbModified) {
                Log.i(TAG, "Extracting Flamingo index (one-time)...")
                onProgress?.invoke("Extracting Flamingo index...")
                EmbeddingIndex.extractFromDatabase(database, EmbeddingModel.FLAMINGO, embFile) { cur, total ->
                    onProgress?.invoke("Extracting Flamingo: $cur / $total")
                }
            }
            if (flamingoIndex == null) {
                flamingoIndex = EmbeddingIndex.mmap(embFile)
                Log.i(TAG, "Flamingo index: ${flamingoIndex!!.numTracks} tracks, dim=${flamingoIndex!!.dim}")
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
     *
     * @param drift When true, each result seeds the next search instead of
     *              always searching relative to the original seed. The queue
     *              gradually drifts through embedding space.
     */
    suspend fun findSimilarTracks(
        seedTrackId: Long,
        numTracks: Int,
        strategy: SearchStrategy,
        anchorExpandConfig: AnchorExpandConfig? = null,
        drift: Boolean = false,
        onProgress: ((String) -> Unit)? = null
    ): List<SimilarTrack> = withContext(Dispatchers.Default) {
        // Caller should have called ensureIndices() first for progress reporting.
        // Call it here as a safety net (no-op if already done).
        ensureIndices()

        when (strategy) {
            SearchStrategy.MULAN_ONLY ->
                if (drift) singleModelDrift(seedTrackId, numTracks, EmbeddingModel.MULAN, onProgress)
                else singleModelSearch(seedTrackId, numTracks, EmbeddingModel.MULAN)
            SearchStrategy.FLAMINGO_ONLY ->
                if (drift) singleModelDrift(seedTrackId, numTracks, EmbeddingModel.FLAMINGO, onProgress)
                else singleModelSearch(seedTrackId, numTracks, EmbeddingModel.FLAMINGO)
            SearchStrategy.INTERLEAVE ->
                if (drift) interleaveDrift(seedTrackId, numTracks, onProgress)
                else interleaveSearch(seedTrackId, numTracks)
            SearchStrategy.ANCHOR_EXPAND ->
                if (drift) anchorExpandDrift(seedTrackId, numTracks, anchorExpandConfig!!, onProgress)
                else anchorExpandSearch(seedTrackId, numTracks, anchorExpandConfig!!, onProgress)
        }
    }

    // ---- Non-drift strategies ----

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
     * Anchor & Expand: primary model finds anchors, secondary model expands each.
     *
     * 1. Primary index: findTopK(seedEmbedding, numGroups) -> anchor tracks
     * 2. For each anchor: look up its embedding in secondary index
     * 3. Secondary index: findTopKMulti(anchorEmbeddings, expansionCount) -- single corpus pass
     * 4. Assemble: [anchor1, expansion1a, expansion1b, ..., anchor2, ...]
     */
    private fun anchorExpandSearch(
        seedTrackId: Long,
        numTracks: Int,
        config: AnchorExpandConfig,
        onProgress: ((String) -> Unit)? = null
    ): List<SimilarTrack> {
        val primaryIndex = getIndex(config.primaryModel) ?: run {
            Log.e(TAG, "No index for primary model ${config.primaryModel.name}")
            return emptyList()
        }
        val secondaryIndex = getIndex(config.secondaryModel) ?: run {
            Log.e(TAG, "No index for secondary model ${config.secondaryModel.name}")
            return singleModelSearch(seedTrackId, numTracks, config.primaryModel)
        }

        val seedEmbedding = primaryIndex.getEmbeddingByTrackId(seedTrackId) ?: run {
            Log.e(TAG, "Seed track $seedTrackId has no ${config.primaryModel.name} embedding")
            return emptyList()
        }

        val groupSize = 1 + config.expansionCount
        val numGroups = (numTracks + groupSize - 1) / groupSize

        // Step 1: Find anchor tracks with primary model
        onProgress?.invoke("Finding anchors...")
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

        // Step 3: Single corpus pass -- find expansions for all anchors at once
        val seen = mutableSetOf(seedTrackId)
        seen.addAll(anchors.map { it.first })

        onProgress?.invoke("Expanding anchors...")
        val expansionResults = if (anchorQueries.isNotEmpty()) {
            secondaryIndex.findTopKMulti(anchorQueries, config.expansionCount, excludeIds = seen)
        } else {
            emptyMap()
        }

        // Step 4: Assemble results in group order
        val result = mutableListOf<SimilarTrack>()

        for ((anchorId, anchorScore) in anchors) {
            if (result.size >= numTracks) break

            database.getTrackById(anchorId)?.let { track ->
                result.add(SimilarTrack(track, anchorScore, config.primaryModel))
            }

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

    // ---- Drift strategies ----
    // Each result seeds the next search, so the queue gradually moves
    // through embedding space away from the original seed.

    /**
     * Single model drift: find top-1, use it as next seed, repeat.
     */
    private fun singleModelDrift(
        seedTrackId: Long,
        numTracks: Int,
        model: EmbeddingModel,
        onProgress: ((String) -> Unit)? = null
    ): List<SimilarTrack> {
        val index = getIndex(model) ?: return emptyList()
        var currentEmb = index.getEmbeddingByTrackId(seedTrackId) ?: return emptyList()

        val result = mutableListOf<SimilarTrack>()
        val seen = mutableSetOf(seedTrackId)

        for (i in 0 until numTracks) {
            onProgress?.invoke("Drifting ahead: ${i + 1}/$numTracks...")
            val top = index.findTopK(currentEmb, 1, excludeIds = seen)
            if (top.isEmpty()) break

            val (trackId, score) = top[0]
            seen.add(trackId)

            database.getTrackById(trackId)?.let { track ->
                result.add(SimilarTrack(track, score, model))
            }

            currentEmb = index.getEmbeddingByTrackId(trackId) ?: break
        }

        Log.d(TAG, "Drift (${model.name}): ${result.size} tracks")
        return result
    }

    /**
     * Interleave drift: models take turns, last queued track seeds the next pick.
     *
     * Each model looks up the current track in its own embedding space,
     * so they stay coherent even though the embedding spaces differ.
     */
    private fun interleaveDrift(
        seedTrackId: Long,
        numTracks: Int,
        onProgress: ((String) -> Unit)? = null
    ): List<SimilarTrack> {
        val mulanIdx = mulanIndex
        val flamingoIdx = flamingoIndex

        if (mulanIdx == null && flamingoIdx == null) return emptyList()
        if (mulanIdx == null) return singleModelDrift(seedTrackId, numTracks, EmbeddingModel.FLAMINGO)
        if (flamingoIdx == null) return singleModelDrift(seedTrackId, numTracks, EmbeddingModel.MULAN)

        val result = mutableListOf<SimilarTrack>()
        val seen = mutableSetOf(seedTrackId)
        var currentTrackId = seedTrackId

        while (result.size < numTracks) {
            var advanced = false
            onProgress?.invoke("Drifting ahead: ${result.size}/$numTracks...")

            // MuLan turn
            mulanIdx.getEmbeddingByTrackId(currentTrackId)?.let { emb ->
                val top = mulanIdx.findTopK(emb, 1, excludeIds = seen)
                if (top.isNotEmpty()) {
                    val (id, score) = top[0]
                    seen.add(id)
                    database.getTrackById(id)?.let {
                        result.add(SimilarTrack(it, score, EmbeddingModel.MULAN))
                        currentTrackId = id
                        advanced = true
                    }
                }
            }

            if (result.size >= numTracks) break

            // Flamingo turn
            flamingoIdx.getEmbeddingByTrackId(currentTrackId)?.let { emb ->
                val top = flamingoIdx.findTopK(emb, 1, excludeIds = seen)
                if (top.isNotEmpty()) {
                    val (id, score) = top[0]
                    seen.add(id)
                    database.getTrackById(id)?.let {
                        result.add(SimilarTrack(it, score, EmbeddingModel.FLAMINGO))
                        currentTrackId = id
                        advanced = true
                    }
                }
            }

            if (!advanced) break
        }

        Log.d(TAG, "Drift (interleave): ${result.size} tracks")
        return result
    }

    /**
     * Anchor & Expand drift: each anchor is found relative to the previous anchor
     * (not the original seed), so groups gradually move through embedding space.
     */
    private fun anchorExpandDrift(
        seedTrackId: Long,
        numTracks: Int,
        config: AnchorExpandConfig,
        onProgress: ((String) -> Unit)? = null
    ): List<SimilarTrack> {
        val primaryIndex = getIndex(config.primaryModel) ?: return emptyList()
        val secondaryIndex = getIndex(config.secondaryModel) ?: run {
            return singleModelDrift(seedTrackId, numTracks, config.primaryModel)
        }

        var currentEmb = primaryIndex.getEmbeddingByTrackId(seedTrackId) ?: return emptyList()

        val result = mutableListOf<SimilarTrack>()
        val seen = mutableSetOf(seedTrackId)
        val groupSize = 1 + config.expansionCount
        val numGroups = (numTracks + groupSize - 1) / groupSize

        for (g in 0 until numGroups) {
            if (result.size >= numTracks) break
            onProgress?.invoke("Drifting ahead: group ${g + 1}/$numGroups...")

            // Find anchor relative to current seed (drifted from previous anchor)
            val anchors = primaryIndex.findTopK(currentEmb, 1, excludeIds = seen)
            if (anchors.isEmpty()) break

            val (anchorId, anchorScore) = anchors[0]
            seen.add(anchorId)

            database.getTrackById(anchorId)?.let { track ->
                result.add(SimilarTrack(track, anchorScore, config.primaryModel))
            }

            // Expand anchor with secondary model
            secondaryIndex.getEmbeddingByTrackId(anchorId)?.let { anchorEmb ->
                val expansions = secondaryIndex.findTopK(anchorEmb, config.expansionCount, excludeIds = seen)
                for ((expId, expScore) in expansions) {
                    if (result.size >= numTracks) break
                    seen.add(expId)
                    database.getTrackById(expId)?.let { track ->
                        result.add(SimilarTrack(track, expScore, config.secondaryModel))
                    }
                }
            }

            // Drift: next anchor is found relative to this anchor
            currentEmb = primaryIndex.getEmbeddingByTrackId(anchorId) ?: break
        }

        Log.d(TAG, "Drift (anchor & expand): ${result.size} tracks in ${result.size / maxOf(groupSize, 1)} groups")
        return result
    }
}
