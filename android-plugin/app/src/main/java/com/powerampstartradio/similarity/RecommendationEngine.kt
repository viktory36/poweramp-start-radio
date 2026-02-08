package com.powerampstartradio.similarity

import android.util.Log
import com.powerampstartradio.data.EmbeddedTrack
import com.powerampstartradio.data.EmbeddingDatabase
import com.powerampstartradio.data.EmbeddingIndex
import com.powerampstartradio.data.GraphIndex
import com.powerampstartradio.similarity.algorithms.DppSelector
import com.powerampstartradio.similarity.algorithms.DriftEngine
import com.powerampstartradio.similarity.algorithms.MmrSelector
import com.powerampstartradio.similarity.algorithms.PostFilter
import com.powerampstartradio.similarity.algorithms.RandomWalkSelector
import com.powerampstartradio.similarity.algorithms.TemperatureSelector
import com.powerampstartradio.ui.DriftMode
import com.powerampstartradio.ui.Influence
import com.powerampstartradio.ui.RadioConfig
import com.powerampstartradio.ui.SelectionMode
import com.powerampstartradio.ui.TrackProvenance
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import java.io.File
import kotlin.coroutines.coroutineContext

/**
 * Result of a similarity search.
 */
data class SimilarTrack(
    val track: EmbeddedTrack,
    val similarity: Float,
    val provenance: TrackProvenance = TrackProvenance(),
)

/**
 * Unified recommendation engine using fused embeddings.
 *
 * Two-stage architecture:
 * 1. RETRIEVE: brute-force top-N candidates from fused index
 * 2. SELECT: user-configured algorithm (MMR / DPP / Random Walk / Temperature)
 * Optional: DRIFT modifies query per step (seed interpolation or EMA momentum)
 * Post-filter: artist/album caps
 */
class RecommendationEngine(
    private val database: EmbeddingDatabase,
    private val filesDir: File
) {
    companion object {
        private const val TAG = "RecommendationEngine"
    }

    private var embeddingIndex: EmbeddingIndex? = null
    private var graphIndex: GraphIndex? = null

    /**
     * Ensure mmap'd indices are ready. Extracts from SQLite if needed (one-time).
     */
    suspend fun ensureIndices(
        onProgress: ((message: String) -> Unit)? = null
    ) = withContext(Dispatchers.IO) {
        val dbFile = File(filesDir, "embeddings.db")
        val dbModified = dbFile.lastModified()

        // Embedding index
        val embFile = File(filesDir, "fused.emb")
        if (!embFile.exists() || embFile.lastModified() < dbModified) {
            Log.i(TAG, "Extracting embedding index (one-time)...")
            onProgress?.invoke("Extracting embedding index...")
            EmbeddingIndex.extractFromDatabase(database, embFile) { cur, total ->
                onProgress?.invoke("Extracting: $cur / $total")
            }
        }
        if (embeddingIndex == null) {
            embeddingIndex = EmbeddingIndex.mmap(embFile)
            Log.i(TAG, "Index: ${embeddingIndex!!.numTracks} tracks, dim=${embeddingIndex!!.dim}")
        }

        // Graph index (optional, only for Random Walk)
        val graphFile = File(filesDir, "graph.bin")
        if (!graphFile.exists() || graphFile.lastModified() < dbModified) {
            onProgress?.invoke("Extracting kNN graph...")
            GraphIndex.extractFromDatabase(database, graphFile)
        }
        if (graphFile.exists() && graphIndex == null) {
            try {
                graphIndex = GraphIndex.mmap(graphFile)
            } catch (e: Exception) {
                Log.w(TAG, "Failed to load graph.bin: ${e.message}")
            }
        }
    }

    /**
     * Generate a playlist using the configured algorithm.
     *
     * @param seedTrackId Track ID to start from
     * @param config Algorithm configuration
     * @param onProgress Status message callback
     * @param onResult Per-track streaming callback (for drift mode)
     * @return Complete list of similar tracks
     */
    suspend fun generatePlaylist(
        seedTrackId: Long,
        config: RadioConfig,
        onProgress: ((String) -> Unit)? = null,
        onResult: (suspend (SimilarTrack) -> Unit)? = null
    ): List<SimilarTrack> = withContext(Dispatchers.Default) {
        ensureIndices()

        val index = embeddingIndex ?: return@withContext emptyList()
        val cancellationCheck: () -> Unit = { coroutineContext.ensureActive() }

        val seedEmb = index.getEmbeddingByTrackId(seedTrackId)
        if (seedEmb == null) {
            Log.e(TAG, "Seed track $seedTrackId has no embedding")
            return@withContext emptyList()
        }

        // Random Walk mode uses graph, not embedding scan
        if (config.selectionMode == SelectionMode.RANDOM_WALK) {
            return@withContext randomWalkPlaylist(seedTrackId, config, onProgress, onResult)
        }

        // DPP in drift mode is degenerate (k=1 selection = pure greedy max similarity,
        // identical to MMR lambda=1). Force batch mode for DPP.
        val effectiveConfig = if (config.selectionMode == SelectionMode.DPP && config.driftEnabled) {
            Log.w(TAG, "DPP+drift is degenerate; forcing batch mode")
            config.copy(driftEnabled = false)
        } else config

        if (effectiveConfig.driftEnabled) {
            driftPlaylist(seedTrackId, seedEmb, index, effectiveConfig, onProgress, onResult, cancellationCheck)
        } else {
            batchPlaylist(seedTrackId, seedEmb, index, effectiveConfig, onProgress, cancellationCheck)
        }
    }

    /**
     * Compute nearest-neighbor diversity provenance for a batch of tracks.
     *
     * For each track, finds the closest sibling in the selection. The strip
     * then shows: primary = uniqueness (1 - nn_sim), colored = redundancy with
     * nearest neighbor (nn_sim). Wider primary → more diverse pick.
     */
    private fun nnProvenance(
        filtered: List<Pair<com.powerampstartradio.data.EmbeddedTrack, Float>>,
        index: EmbeddingIndex
    ): List<TrackProvenance> {
        if (filtered.size <= 1) return filtered.map { TrackProvenance() }

        val embs = filtered.map { (track, _) -> index.getEmbeddingByTrackId(track.id) }

        return filtered.indices.map { i ->
            val a = embs[i]
            if (a == null) return@map TrackProvenance()
            var maxSim = -1f
            var nnIdx = 0
            for (j in filtered.indices) {
                if (j == i) continue
                val b = embs[j] ?: continue
                var dot = 0f
                for (d in a.indices) dot += a[d] * b[d]
                if (dot > maxSim) { maxSim = dot; nnIdx = j }
            }
            if (maxSim >= 0f) {
                TrackProvenance(listOf(
                    Influence(-1, 1f - maxSim),
                    Influence(nnIdx, maxSim)
                ))
            } else TrackProvenance()
        }
    }

    /**
     * Compute provenance for a track based on the query that produced it.
     *
     * For seed interpolation: the query was `alpha * seed + (1-alpha) * prev_track`,
     * so the track has exactly 2 influences — seed (weight=seedWeight) and
     * previous track (weight=1-seedWeight).
     *
     * For EMA momentum: the query is a weighted sum of all predecessors,
     * so every prior track + seed has an influence with geometrically decaying weights.
     *
     * @param resultIndex 0-based index of this track in the result list
     * @param seedWeight Exact seed weight from the DriftResult that produced the current query
     *                   (1.0 for the first track, since query = pure seed)
     * @param config Radio configuration
     */
    private fun computeProvenance(
        resultIndex: Int,
        seedWeight: Float,
        config: RadioConfig
    ): TrackProvenance {
        if (resultIndex == 0) return TrackProvenance()  // 100% seed

        return when (config.driftMode) {
            DriftMode.SEED_INTERPOLATION -> {
                // Query was: seedWeight * seed + (1-seedWeight) * track_{i-1}
                TrackProvenance(listOf(
                    Influence(-1, seedWeight),
                    Influence(resultIndex - 1, 1f - seedWeight)
                ))
            }
            DriftMode.MOMENTUM -> {
                val beta = config.momentumBeta
                val influences = mutableListOf<Influence>()
                // Seed contribution: beta^(resultIndex)
                val sw = Math.pow(beta.toDouble(), resultIndex.toDouble()).toFloat()
                if (sw > 0.01f) influences.add(Influence(-1, sw))
                // Track j contributes beta^(resultIndex - j - 1) * (1 - beta)
                for (j in 0 until resultIndex) {
                    val w = Math.pow(beta.toDouble(), (resultIndex - j - 1).toDouble()).toFloat() * (1f - beta)
                    if (w > 0.01f) influences.add(Influence(j, w))
                }
                TrackProvenance(influences)
            }
        }
    }

    /**
     * Drift mode: per-step selection with evolving query.
     */
    private suspend fun driftPlaylist(
        seedTrackId: Long,
        seedEmb: FloatArray,
        index: EmbeddingIndex,
        config: RadioConfig,
        onProgress: ((String) -> Unit)?,
        onResult: (suspend (SimilarTrack) -> Unit)?,
        cancellationCheck: () -> Unit
    ): List<SimilarTrack> {
        val result = mutableListOf<SimilarTrack>()
        val selectedTracks = mutableListOf<EmbeddedTrack>()
        val selectedEmbeddings = mutableListOf<FloatArray>()
        val seen = mutableSetOf(seedTrackId)
        var query = seedEmb
        var emaState: FloatArray? = null
        // Track the seed weight of the current query for provenance.
        // Initial query = pure seed, so seedWeight = 1.0
        var currentSeedWeight = 1f

        for (step in 0 until config.numTracks) {
            coroutineContext.ensureActive()
            onProgress?.invoke("Finding track ${step + 1}/${config.numTracks}...")

            // Retrieve candidates for current query
            val poolSize = minOf(config.candidatePoolSize, 50)  // Smaller pool per drift step
            val candidates = index.findTopK(query, poolSize, excludeIds = seen, cancellationCheck = cancellationCheck)

            if (candidates.isEmpty()) break

            // Select one using configured algorithm
            val selected = selectOneFromCandidates(
                candidates, selectedEmbeddings, index, config
            ) ?: break

            val (trackId, score) = selected
            seen.add(trackId)

            val track = database.getTrackById(trackId) ?: continue
            val provenance = computeProvenance(result.size, currentSeedWeight, config)

            // Check artist constraints
            if (!PostFilter.canAdd(track, selectedTracks, config.maxPerArtist, config.minArtistSpacing)) {
                // Try next best candidates
                val fallback = candidates
                    .filter { it.first != trackId && it.first !in seen }
                    .firstOrNull { (altId, _) ->
                        val altTrack = database.getTrackById(altId)
                        altTrack != null && PostFilter.canAdd(altTrack, selectedTracks, config.maxPerArtist, config.minArtistSpacing)
                    }

                if (fallback != null) {
                    val (fbId, fbScore) = fallback
                    seen.add(fbId)
                    val fbTrack = database.getTrackById(fbId) ?: continue
                    val similarTrack = SimilarTrack(fbTrack, fbScore, provenance)
                    result.add(similarTrack)
                    selectedTracks.add(fbTrack)
                    index.getEmbeddingByTrackId(fbId)?.let { selectedEmbeddings.add(it) }
                    onResult?.invoke(similarTrack)

                    val currentEmb = index.getEmbeddingByTrackId(fbId) ?: continue
                    val driftResult = DriftEngine.updateQuery(
                        seedEmb, currentEmb, emaState, step, config.numTracks, config
                    )
                    query = driftResult.query
                    emaState = driftResult.emaState
                    currentSeedWeight = driftResult.seedWeight
                    continue
                }
                // No valid fallback — skip this step
                continue
            }

            val similarTrack = SimilarTrack(track, score, provenance)
            result.add(similarTrack)
            selectedTracks.add(track)
            onResult?.invoke(similarTrack)

            // Update query for next step
            val currentEmb = index.getEmbeddingByTrackId(trackId)
            if (currentEmb != null) {
                selectedEmbeddings.add(currentEmb)
                val driftResult = DriftEngine.updateQuery(
                    seedEmb, currentEmb, emaState, step, config.numTracks, config
                )
                query = driftResult.query
                emaState = driftResult.emaState
                currentSeedWeight = driftResult.seedWeight
            }
        }

        Log.d(TAG, "Drift: ${result.size} tracks")
        return result
    }

    /**
     * Batch mode: retrieve large pool, apply algorithm, post-filter.
     */
    private fun batchPlaylist(
        seedTrackId: Long,
        seedEmb: FloatArray,
        index: EmbeddingIndex,
        config: RadioConfig,
        onProgress: ((String) -> Unit)?,
        cancellationCheck: () -> Unit
    ): List<SimilarTrack> {
        onProgress?.invoke("Searching...")

        // Stage 1: Retrieve candidates
        val candidates = index.findTopK(
            seedEmb, config.candidatePoolSize,
            excludeIds = setOf(seedTrackId),
            cancellationCheck = cancellationCheck
        )

        if (candidates.isEmpty()) return emptyList()

        // Stage 2: Select using algorithm
        onProgress?.invoke("Selecting tracks...")
        val selected = when (config.selectionMode) {
            SelectionMode.MMR -> MmrSelector.selectBatch(
                candidates, config.numTracks, index, config.diversityLambda
            )
            SelectionMode.DPP -> DppSelector.selectBatch(
                candidates, config.numTracks, index
            )
            SelectionMode.TEMPERATURE -> TemperatureSelector.selectBatch(
                candidates, config.numTracks, config.temperature
            )
            SelectionMode.RANDOM_WALK -> candidates.take(config.numTracks) // Shouldn't reach here
        }

        // Resolve track metadata
        val tracks = selected.mapNotNull { (trackId, score) ->
            database.getTrackById(trackId)?.let { it to score }
        }

        // Stage 3: Post-filter artist constraints
        val filtered = PostFilter.enforceBatch(
            tracks.map { (track, score) -> track to score },
            config.maxPerArtist,
            config.minArtistSpacing
        )

        val provenance = nnProvenance(filtered, index)
        return filtered.mapIndexed { i, (track, score) ->
            SimilarTrack(track, score, provenance[i])
        }
    }

    /**
     * Random Walk playlist using precomputed kNN graph.
     */
    private suspend fun randomWalkPlaylist(
        seedTrackId: Long,
        config: RadioConfig,
        onProgress: ((String) -> Unit)?,
        onResult: (suspend (SimilarTrack) -> Unit)?
    ): List<SimilarTrack> {
        val graph = graphIndex
        if (graph == null) {
            Log.w(TAG, "No graph.bin available, falling back to embedding search")
            val index = embeddingIndex ?: return emptyList()
            val seedEmb = index.getEmbeddingByTrackId(seedTrackId) ?: return emptyList()
            val ctx = coroutineContext
            val cancellationCheck: () -> Unit = { ctx.ensureActive() }
            return batchPlaylist(seedTrackId, seedEmb, index,
                config.copy(selectionMode = SelectionMode.MMR), onProgress, cancellationCheck)
        }

        onProgress?.invoke("Computing random walk...")
        val alpha = config.anchorStrength  // Reuse anchor strength as restart probability
        val ranking = RandomWalkSelector.computeRanking(graph, seedTrackId, alpha)

        // Resolve tracks and post-filter
        val tracks = ranking.take(config.candidatePoolSize).mapNotNull { (trackId, score) ->
            database.getTrackById(trackId)?.let { it to score }
        }

        val filtered = PostFilter.enforceBatch(
            tracks,
            config.maxPerArtist,
            config.minArtistSpacing
        ).take(config.numTracks)

        val index = embeddingIndex
        val provenance = if (index != null) nnProvenance(filtered, index)
            else filtered.map { TrackProvenance() }
        val result = filtered.mapIndexed { i, (track, score) ->
            SimilarTrack(track, score, provenance[i])
        }

        // Stream results if callback provided
        onResult?.let { callback ->
            for (track in result) {
                callback(track)
            }
        }

        Log.d(TAG, "Random walk: ${result.size} tracks")
        return result
    }

    /**
     * Select one track from candidates using the configured algorithm.
     * Used in drift mode for per-step selection.
     */
    private fun selectOneFromCandidates(
        candidates: List<Pair<Long, Float>>,
        selectedEmbeddings: List<FloatArray>,
        index: EmbeddingIndex,
        config: RadioConfig
    ): Pair<Long, Float>? {
        return when (config.selectionMode) {
            SelectionMode.MMR -> MmrSelector.selectOne(
                candidates, selectedEmbeddings, index, config.diversityLambda
            )
            SelectionMode.DPP -> {
                // DPP in single-select mode: use batch of 1
                DppSelector.selectBatch(candidates, 1, index).firstOrNull()
            }
            SelectionMode.TEMPERATURE -> TemperatureSelector.selectOne(
                candidates, config.temperature
            )
            SelectionMode.RANDOM_WALK -> candidates.firstOrNull()
        }
    }
}
