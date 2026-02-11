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
import com.powerampstartradio.ui.QueueMetrics
import com.powerampstartradio.ui.RadioConfig
import com.powerampstartradio.ui.SelectionMode
import com.powerampstartradio.ui.TrackProvenance
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import java.io.File
import kotlin.coroutines.coroutineContext

/**
 * Result of a selector: which track was picked and where it ranked in the candidate pool.
 */
data class SelectedTrack(
    val trackId: Long,
    val score: Float,
    val candidateRank: Int,  // 1-based position in sorted candidate list
)

/**
 * Result of a similarity search.
 */
data class SimilarTrack(
    val track: EmbeddedTrack,
    val similarity: Float,
    val similarityToSeed: Float,
    val candidateRank: Int? = null,
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

        // Auto-compute pool size: 2% of library, floor 100
        val poolConfig = if (config.candidatePoolSize <= 0) {
            val autoPool = (index.numTracks * 0.02f).toInt().coerceAtLeast(100)
            Log.d(TAG, "Auto pool size: $autoPool (${index.numTracks} tracks)")
            config.copy(candidatePoolSize = autoPool)
        } else config

        val seedEmb = index.getEmbeddingByTrackId(seedTrackId)
        if (seedEmb == null) {
            Log.e(TAG, "Seed track $seedTrackId has no embedding")
            return@withContext emptyList()
        }

        // Random Walk mode uses graph, not embedding scan
        if (poolConfig.selectionMode == SelectionMode.RANDOM_WALK) {
            return@withContext randomWalkPlaylist(seedTrackId, poolConfig, onProgress, onResult)
        }

        // DPP in drift mode is degenerate (k=1 selection = pure greedy max similarity,
        // identical to MMR lambda=1). Force batch mode for DPP.
        val effectiveConfig = if (poolConfig.selectionMode == SelectionMode.DPP && poolConfig.driftEnabled) {
            Log.w(TAG, "DPP+drift is degenerate; forcing batch mode")
            poolConfig.copy(driftEnabled = false)
        } else poolConfig

        if (effectiveConfig.driftEnabled) {
            driftPlaylist(seedTrackId, seedEmb, index, effectiveConfig, onProgress, onResult, cancellationCheck)
        } else {
            batchPlaylist(seedTrackId, seedEmb, index, effectiveConfig, onProgress, cancellationCheck)
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
            val candidates = index.findTopK(query, config.candidatePoolSize, excludeIds = seen, cancellationCheck = cancellationCheck)

            if (candidates.isEmpty()) break

            // Select one using configured algorithm
            val selected = selectOneFromCandidates(
                candidates, selectedEmbeddings, index, config
            ) ?: break

            val trackId = selected.trackId
            val score = selected.score
            seen.add(trackId)

            val track = database.getTrackById(trackId) ?: continue
            val provenance = computeProvenance(result.size, currentSeedWeight, config)

            // Compute similarityToSeed via dot product with original seed
            val trackEmb = index.getEmbeddingByTrackId(trackId)
            val simToSeed = if (trackEmb != null) dotProduct(seedEmb, trackEmb) else score

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
                    val fbEmb = index.getEmbeddingByTrackId(fbId)
                    val fbSimToSeed = if (fbEmb != null) dotProduct(seedEmb, fbEmb) else fbScore
                    val similarTrack = SimilarTrack(fbTrack, fbScore, fbSimToSeed, provenance = provenance)
                    result.add(similarTrack)
                    selectedTracks.add(fbTrack)
                    fbEmb?.let { selectedEmbeddings.add(it) }
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

            val similarTrack = SimilarTrack(track, score, simToSeed, selected.candidateRank, provenance)
            result.add(similarTrack)
            selectedTracks.add(track)
            onResult?.invoke(similarTrack)

            // Update query for next step
            if (trackEmb != null) {
                selectedEmbeddings.add(trackEmb)
                val driftResult = DriftEngine.updateQuery(
                    seedEmb, trackEmb, emaState, step, config.numTracks, config
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
        val selected: List<SelectedTrack> = when (config.selectionMode) {
            SelectionMode.MMR -> MmrSelector.selectBatch(
                candidates, config.numTracks, index, config.diversityLambda
            )
            SelectionMode.DPP -> DppSelector.selectBatch(
                candidates, config.numTracks, index
            )
            SelectionMode.TEMPERATURE -> TemperatureSelector.selectBatch(
                candidates, config.numTracks, config.temperature
            )
            SelectionMode.RANDOM_WALK -> candidates.take(config.numTracks).mapIndexed { i, (id, score) ->
                SelectedTrack(id, score, candidateRank = i + 1)
            }
        }

        // Resolve track metadata and build SimilarTrack with new fields
        // In batch mode query IS seed, so similarityToSeed = score
        val tracks = selected.mapNotNull { sel ->
            database.getTrackById(sel.trackId)?.let { track ->
                SimilarTrack(
                    track = track,
                    similarity = sel.score,
                    similarityToSeed = sel.score,
                    candidateRank = sel.candidateRank,
                )
            }
        }

        // Stage 3: Post-filter artist constraints
        return PostFilter.enforceBatch(tracks, config.maxPerArtist, config.minArtistSpacing)
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
        val index = embeddingIndex
        if (graph == null) {
            Log.w(TAG, "No graph.bin available, falling back to embedding search")
            if (index == null) return emptyList()
            val seedEmb = index.getEmbeddingByTrackId(seedTrackId) ?: return emptyList()
            val ctx = coroutineContext
            val cancellationCheck: () -> Unit = { ctx.ensureActive() }
            return batchPlaylist(seedTrackId, seedEmb, index,
                config.copy(selectionMode = SelectionMode.MMR), onProgress, cancellationCheck)
        }

        onProgress?.invoke("Computing random walk...")
        val alpha = config.anchorStrength  // Reuse anchor strength as restart probability
        val ranking = RandomWalkSelector.computeRanking(graph, seedTrackId, alpha)

        val seedEmb = index?.getEmbeddingByTrackId(seedTrackId)

        // Resolve tracks, compute similarityToSeed, and post-filter
        val ranked = ranking.take(config.candidatePoolSize)
        val tracks = ranked.indices.mapNotNull { i ->
            val (trackId, score) = ranked[i]
            database.getTrackById(trackId)?.let { track ->
                val simToSeed = if (seedEmb != null && index != null) {
                    val emb = index.getEmbeddingByTrackId(trackId)
                    if (emb != null) dotProduct(seedEmb, emb) else 0f
                } else 0f
                SimilarTrack(
                    track = track,
                    similarity = score,
                    similarityToSeed = simToSeed,
                    candidateRank = i + 1,
                )
            }
        }

        val filtered = PostFilter.enforceBatch(
            tracks,
            config.maxPerArtist,
            config.minArtistSpacing
        ).take(config.numTracks)

        // Stream results if callback provided
        onResult?.let { callback ->
            for (track in filtered) {
                callback(track)
            }
        }

        Log.d(TAG, "Random walk: ${filtered.size} tracks")
        return filtered
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
    ): SelectedTrack? {
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
            SelectionMode.RANDOM_WALK -> candidates.firstOrNull()?.let {
                SelectedTrack(it.first, it.second, candidateRank = 1)
            }
        }
    }

    /**
     * Compute quality metrics for a completed queue.
     *
     * @param tracks The queued tracks with similarity scores
     * @return QueueMetrics with artist count, cluster spread, and sim range
     */
    fun computeQueueMetrics(tracks: List<SimilarTrack>): QueueMetrics {
        if (tracks.isEmpty()) return QueueMetrics(0, 0, 0 to 0)

        // Unique artists
        val artists = tracks.mapNotNull { it.track.artist?.lowercase() }.toSet()

        // Cluster spread
        val clusterAssignments = database.loadClusterAssignments()
        val clusters = tracks.mapNotNull { clusterAssignments[it.track.id] }.toSet()

        // Similarity range (as percentage)
        val sims = tracks.map { it.similarityToSeed }
        val minSim = (sims.min() * 100).toInt()
        val maxSim = (sims.max() * 100).toInt()

        return QueueMetrics(
            uniqueArtists = artists.size,
            clusterSpread = clusters.size,
            simRange = minSim to maxSim,
        )
    }

    private fun dotProduct(a: FloatArray, b: FloatArray): Float {
        var sum = 0f
        for (i in a.indices) sum += a[i] * b[i]
        return sum
    }
}
