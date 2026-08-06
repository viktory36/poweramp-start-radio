package com.powerampstartradio.similarity

import android.util.Log
import com.powerampstartradio.data.EmbeddedTrack
import com.powerampstartradio.data.ActiveDomainGraphTopology
import com.powerampstartradio.data.ActiveDomainGraphTopologyBuilder
import com.powerampstartradio.data.ArtistCreditCatalog
import com.powerampstartradio.data.EmbeddingDatabase
import com.powerampstartradio.data.EmbeddingIndex
import com.powerampstartradio.data.GraphIndex
import com.powerampstartradio.data.StableIdentityGenerationBinding
import com.powerampstartradio.data.StableTrackIdentityCatalog
import com.powerampstartradio.indexing.V2ActiveLibraryCatalog
import com.powerampstartradio.similarity.algorithms.DppSelector
import com.powerampstartradio.similarity.algorithms.ClosestSelector
import com.powerampstartradio.similarity.algorithms.DriftEngine
import com.powerampstartradio.similarity.algorithms.GeoMeanSelector
import com.powerampstartradio.similarity.algorithms.MmrSelector
import com.powerampstartradio.similarity.algorithms.MmrSelectionEvidence
import com.powerampstartradio.similarity.algorithms.PostFilter
import com.powerampstartradio.similarity.algorithms.GraphExplorerSelector
import com.powerampstartradio.similarity.algorithms.GraphExplorerTrackScore
import com.powerampstartradio.similarity.algorithms.UniformShuffleSelector
import com.powerampstartradio.ui.DriftMode
import com.powerampstartradio.ui.ComposedTrackEvidence
import com.powerampstartradio.ui.FindMusicOperator
import com.powerampstartradio.ui.FindMusicQuerySpec
import com.powerampstartradio.ui.FindMusicTextResultPlanner
import com.powerampstartradio.ui.Influence
import com.powerampstartradio.ui.MmrTrackEvidence
import com.powerampstartradio.ui.QueueMetrics
import com.powerampstartradio.ui.RadioConfig
import com.powerampstartradio.ui.SeedSpec
import com.powerampstartradio.ui.SelectionControlText
import com.powerampstartradio.ui.SelectionMode
import com.powerampstartradio.ui.StableResultReductionEvidence
import com.powerampstartradio.ui.TrackProvenance
import com.powerampstartradio.ui.effectiveLibraryAddedDays
import com.powerampstartradio.ui.minimumLibraryAddedAtEpochSecond
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
    val mmrSelectionEvidence: MmrSelectionEvidence? = null,
)

/**
 * Result of a similarity search.
 */
data class SimilarTrack(
    val track: EmbeddedTrack,
    val similarity: Float,
    val similarityToSeed: Float,
    val candidateRank: Int? = null,
    /** Cosine rank from the original seed over full active identities, excluding the seed. */
    val seedRank: Int? = null,
    /** Cosine rank from the evolving query over the same full active identity domain. */
    val driftRank: Int? = null,
    val graphTerminalProbability: Double? = null,
    val graphExpectedRouteLinks: Double? = null,
    val graphHops: Int? = null,
    val provenance: TrackProvenance = TrackProvenance(),
    val driftReferenceEmb: FloatArray? = null,
    val composedEvidence: ComposedTrackEvidence? = null,
    val mmrEvidence: MmrTrackEvidence? = null,
)

data class ComposedPlaylistResult(
    val tracks: List<SimilarTrack>,
    val stableResultReduction: StableResultReductionEvidence,
    /** Exact remaining representative domain over which objectiveRank was assigned. */
    val objectiveRankingDomainCount: Int,
)

/** Session-wide proof emitted by exact Graph Explorer before metadata filtering. */
data class GraphExplorationEvidence(
    val expectedRouteLinks: Double,
    val excludedSeedProbability: Double,
    val totalTerminalProbability: Double,
    val numericMassError: Double,
    val evaluatedLinks: Int,
    /** Positive-probability queue identities after removing the seed identity closure. */
    val rankedCandidateCount: Int,
    /** Populated when the generation activation layer exposes a durable identity. */
    val graphGenerationId: String? = null,
)

/** Identity coverage behind one deterministic Uniform Shuffle result. */
data class UniformShuffleIdentityEvidence(
    val generationBinding: StableIdentityGenerationBinding,
    val libraryTrackCount: Int,
    val stableLibraryTrackCount: Int,
    val legacyLibraryTrackCount: Int,
    val selectedStableTrackCount: Int,
    val selectedLegacyTrackCount: Int,
    /** Distinct identity places ranked by the permutation after excluding the seed identity. */
    val rankedCandidateCount: Int,
)

/** Exact work certificate emitted by automatic full-domain greedy DPP. */
data class DppSelectionEvidence(
    val completeCandidateDomainCount: Int,
    val initialWorkingCandidateCount: Int,
    val attemptedCandidateCounts: List<Int>,
    val finalWorkingCandidateCount: Int,
    /** Greedy Cholesky diagonal selected at each step, in returned-track order. */
    val selectedMarginalGains: List<Double>,
    val finalUnseenInitialGainUpperBound: Double?,
    val usedCompleteCandidateDomain: Boolean,
    val reproducedFullDomainGreedySequence: Boolean,
) {
    init {
        requireValid()
    }

    fun requireValid(expectedSelectedCount: Int? = null) {
        require(completeCandidateDomainCount > 0) {
            "Complete DPP candidate domain must be non-empty"
        }
        require(initialWorkingCandidateCount in 1..completeCandidateDomainCount) {
            "Initial DPP working domain is invalid"
        }
        require(attemptedCandidateCounts.isNotEmpty()) {
            "DPP proof must retain at least one attempted working domain"
        }
        require(attemptedCandidateCounts.first() == initialWorkingCandidateCount) {
            "Initial DPP attempt disagrees with its evidence"
        }
        require(attemptedCandidateCounts.last() == finalWorkingCandidateCount) {
            "Final DPP attempt disagrees with its evidence"
        }
        require(attemptedCandidateCounts.all { it in 1..completeCandidateDomainCount }) {
            "DPP attempted working domain is invalid"
        }
        require(attemptedCandidateCounts.zipWithNext().all { (left, right) -> left < right }) {
            "DPP attempted working domains must grow strictly"
        }
        require(usedCompleteCandidateDomain ==
            (finalWorkingCandidateCount == completeCandidateDomainCount)
        ) { "DPP complete-domain flag disagrees with its final attempt" }
        require((finalUnseenInitialGainUpperBound == null) == usedCompleteCandidateDomain) {
            "DPP unseen-gain bound disagrees with complete-domain use"
        }
        finalUnseenInitialGainUpperBound?.let { unseenBound ->
            require(unseenBound.isFinite() && unseenBound >= 0.0) {
                "DPP unseen-gain bound is invalid"
            }
            require(selectedMarginalGains.all { gain -> gain > unseenBound }) {
                "DPP selected gains do not strictly exceed the unseen bound"
            }
        }
        require(selectedMarginalGains.all { gain -> gain.isFinite() && gain > 0.0 }) {
            "DPP selected marginal gains are invalid"
        }
        require(reproducedFullDomainGreedySequence) {
            "DPP result was not certified against the complete candidate domain"
        }
        expectedSelectedCount?.let { count ->
            require(count >= 0 && selectedMarginalGains.size == count) {
                "DPP selected marginal gains do not align with selected tracks"
            }
        }
    }
}

/** Exact seed-excluded semantic domain and any bounded pool resolved from it. */
data class RecommendationDomainEvidence(
    /** Exact added-date candidate identity domain after excluding the seed identity. */
    val seedExcludedCandidateIdentityCount: Int,
    /** Exact full active identity domain used by seedRank and driftRank. */
    val seedExcludedActiveIdentityCount: Int,
    val resolvedCandidatePoolSize: Int?,
) {
    init {
        require(seedExcludedCandidateIdentityCount >= 0)
        require(seedExcludedActiveIdentityCount >= seedExcludedCandidateIdentityCount)
        require(resolvedCandidatePoolSize == null ||
            resolvedCandidatePoolSize in 0..seedExcludedCandidateIdentityCount
        )
    }
}

/** Immutable files belonging to one validated active index generation. */
data class RecommendationAssetFiles(
    val embeddingFile: File,
    val graphFile: File?,
    val activationBindingId: String? = null,
)

/** One already-mapped PEMB whose source generation was fully validated before planning. */
internal data class PreparedRecommendationEmbeddingIndex(
    val embeddingFile: File,
    val activationBindingId: String,
    val index: EmbeddingIndex,
)

/** Fail-closed gate for avoiding a second mmap only inside the exact pinned generation. */
internal object PreparedRecommendationEmbeddingIndexPolicy {
    fun requireReusable(
        prepared: PreparedRecommendationEmbeddingIndex,
        pinnedAssets: RecommendationAssetFiles?,
        databaseTrackCount: Int,
        headerTrackCount: Int,
    ): EmbeddingIndex {
        val pinned = requireNotNull(pinnedAssets) {
            "A prepared embedding index requires pinned generation assets"
        }
        require(pinned.activationBindingId != null &&
            pinned.activationBindingId == prepared.activationBindingId
        ) { "Prepared embedding index belongs to a different activation binding" }
        require(
            pinned.embeddingFile.canonicalFile == prepared.embeddingFile.canonicalFile,
        ) { "Prepared embedding index belongs to a different PEMB file" }
        require(databaseTrackCount > 0 && headerTrackCount == databaseTrackCount) {
            "Pinned embedding asset does not match its active database"
        }
        require(prepared.index.numTracks == databaseTrackCount) {
            "Prepared embedding index does not match its active database"
        }
        return prepared.index
    }
}

/** One shared seed/selection exclusion rule for queue-visible recording copies. */
internal class StableTrackSelectionPolicy(
    private val identityForTrack: (Long) -> com.powerampstartradio.data.StableVisibleResultIdentity,
    private val equivalentTrackIds: (Long) -> List<Long>,
) {
    fun exclusionClosure(trackIds: Collection<Long>): Set<Long> = buildSet {
        trackIds.forEach { trackId ->
            addAll(equivalentTrackIds(trackId))
        }
    }

    fun canSelect(candidateId: Long, selectedIds: Collection<Long>): Boolean {
        val identityToken = identityForTrack(candidateId).identityToken
        return selectedIds.none { selectedId ->
            identityForTrack(selectedId).identityToken == identityToken
        }
    }
}

/**
 * Unified recommendation engine using CLaMP3 embeddings.
 *
 * Two-stage architecture:
 * 1. RETRIEVE: brute-force top-N candidates from embedding index
 * 2. SELECT: user-configured algorithm (Closest / MMR / DPP / Graph Explorer / Shuffle)
 * Optional: DRIFT modifies query per step (seed interpolation or EMA momentum)
 * Post-filter: artist/album caps
 */
internal class RecommendationEngine(
    private val database: EmbeddingDatabase,
    private val filesDir: File,
    private val pinnedAssets: RecommendationAssetFiles? = null,
    private val activeCatalog: V2ActiveLibraryCatalog? = null,
    private val preparedEmbeddingIndex: PreparedRecommendationEmbeddingIndex? = null,
) {
    companion object {
        private const val TAG = "RecommendationEngine"
    }

    private var embeddingIndex: EmbeddingIndex? = null
    private var graphIndex: GraphIndex? = null
    private var activeDomainSnapshot: ActiveDomainSnapshot? = null
    private var activeGraphTopologySnapshot: ActiveGraphTopologySnapshot? = null

    private fun requireActiveDomain(index: EmbeddingIndex): ActiveDomainSnapshot {
        val sourceCatalog = requireNotNull(activeCatalog) {
            "Recommendation ranking requires an exact active Poweramp library catalog"
        }
        return synchronized(this) {
            activeDomainSnapshot?.takeIf { snapshot ->
                snapshot.embeddingIndex === index && snapshot.sourceCatalog === sourceCatalog
            } ?: run {
                val identityCatalog = StableTrackIdentityCatalog.load(filesDir, database, index)
                val domain = ActiveRecommendationDomain.create(
                    activeCatalog = sourceCatalog,
                    identityCatalog = identityCatalog,
                    embeddingIndex = index,
                )
                require(domain.activeTrackCount > 0) {
                    "The active Poweramp library has no recommendation-eligible tracks"
                }
                val orderedActiveTrackIds = domain.orderedActiveTrackIds()
                val orderedActiveIdentityRepresentativeTrackIds =
                    domain.orderedActiveIdentityRepresentativeTrackIds()
                val orderedDatabaseTrackIds = identityCatalog.orderedTrackIds()
                ActiveDomainSnapshot(
                    embeddingIndex = index,
                    sourceCatalog = sourceCatalog,
                    identityCatalog = identityCatalog,
                    domain = domain,
                    orderedDatabaseTrackIds = orderedDatabaseTrackIds,
                    orderedActiveTrackIds = orderedActiveTrackIds,
                    orderedActiveIdentityRepresentativeTrackIds =
                        orderedActiveIdentityRepresentativeTrackIds,
                    activeTrackIds = orderedActiveTrackIds.toSet(),
                    artistCreditCatalog = database.loadArtistCreditCatalog(orderedDatabaseTrackIds),
                ).also { snapshot ->
                    activeDomainSnapshot = snapshot
                    activeGraphTopologySnapshot = null
                }
            }
        }
    }

    private fun requireActiveGraphTopology(
        index: EmbeddingIndex,
        active: ActiveDomainSnapshot,
        cancellationCheck: () -> Unit,
    ): ActiveDomainGraphTopology {
        val graph = requireNotNull(graphIndex) {
            "Graph Explorer requires a valid graph for the active embedding generation"
        }
        return synchronized(this) {
            activeGraphTopologySnapshot?.takeIf { snapshot ->
                snapshot.graphIndex === graph &&
                    snapshot.embeddingIndex === index &&
                    snapshot.activeDomainBinding == active.domain.binding
            }?.topology ?: ActiveDomainGraphTopologyBuilder.build(
                graph = graph,
                embeddings = index,
                // Graph Explorer is defined over queue-visible recording identities. Keeping
                // duplicate occurrence nodes would let copies absorb terminal probability before
                // the result is collapsed, changing the walk and sometimes shortening the queue.
                orderedActiveTrackIds = active.orderedActiveIdentityRepresentativeTrackIds,
                cancellationCheck = cancellationCheck,
            ).also { topology ->
                check(topology.evidence.nodeCount == active.domain.activeCandidateIdentityCount) {
                    "Active graph node count does not match the active identity domain"
                }
                activeGraphTopologySnapshot = ActiveGraphTopologySnapshot(
                    graphIndex = graph,
                    embeddingIndex = index,
                    activeDomainBinding = active.domain.binding,
                    topology = topology,
                )
            }
        }
    }

    /**
     * Ensure mmap'd indices are ready. Extracts from SQLite if needed (one-time).
     */
    suspend fun ensureIndices(
        onProgress: ((message: String) -> Unit)? = null,
        requireGraph: Boolean = false,
    ) = withContext(Dispatchers.IO) {
        // Embedding index — content-based staleness: compare .emb header track count
        // against actual DB row count. Immune to WAL-recovery mtime bumps.
        val embFile = pinnedAssets?.embeddingFile ?: File(filesDir, "clamp3.emb")
        val dbEmbCount = database.getEmbeddingCountForTable(database.embeddingTable)
        val embHeaderCount = EmbeddingIndex.readHeaderTrackCount(embFile)
        if (pinnedAssets != null) {
            require(dbEmbCount > 0 && embHeaderCount == dbEmbCount) {
                "Pinned embedding asset does not match its active database"
            }
        }
        if (pinnedAssets == null && (embHeaderCount != dbEmbCount || dbEmbCount == 0)) {
            if (dbEmbCount == 0) {
                Log.w(TAG, "No embeddings in DB, skipping extraction")
            } else {
                Log.i(TAG, "Embedding index stale (file=$embHeaderCount, db=$dbEmbCount), re-extracting...")
                onProgress?.invoke("Writing $dbEmbCount embeddings into the memory-mapped search index...")
                EmbeddingIndex.extractFromDatabase(database, embFile) { cur, total ->
                    onProgress?.invoke("Writing search index · $cur of $total embeddings")
                }
                // Invalidate cached mmap since file changed
                embeddingIndex = null
            }
        }
        if (embFile.exists() && embeddingIndex == null) {
            embeddingIndex = if (preparedEmbeddingIndex != null) {
                PreparedRecommendationEmbeddingIndexPolicy.requireReusable(
                    prepared = preparedEmbeddingIndex,
                    pinnedAssets = pinnedAssets,
                    databaseTrackCount = dbEmbCount,
                    headerTrackCount = embHeaderCount,
                ).also {
                    Log.i(TAG, "Reusing generation-bound embedding index: ${it.numTracks} tracks")
                }
            } else {
                onProgress?.invoke("Memory-mapping $dbEmbCount embeddings for similarity search...")
                EmbeddingIndex.mmap(embFile)
            }
            Log.i(TAG, "Index: ${embeddingIndex!!.numTracks} tracks, dim=${embeddingIndex!!.dim}")
        }

        if (!requireGraph) return@withContext

        // Graph index is required only for Graph Explorer.
        // Content-based: compare graph.bin header N against .emb track count
        val graphFile = if (pinnedAssets != null) {
            pinnedAssets.graphFile
        } else {
            File(filesDir, "graph.bin")
        }
        if (pinnedAssets != null && graphFile == null) {
            throw IllegalStateException(
                "Graph Explorer is unavailable because the active generation has no graph"
            )
        }
        val requiredGraphFile = checkNotNull(graphFile)
        val embTrackCount = embeddingIndex?.numTracks ?: embHeaderCount
        val graphNodeCount = GraphIndex.readHeaderNodeCount(requiredGraphFile)
        if (graphNodeCount != embTrackCount) {
            if (pinnedAssets != null) {
                throw IllegalStateException(
                    "Pinned graph asset does not match its active embedding generation"
                )
            }
            onProgress?.invoke("Extracting the $embTrackCount-node similarity graph from the music index...")
            val extracted = GraphIndex.extractFromDatabase(database, requiredGraphFile)
            if (!extracted) {
                graphIndex = null
                throw IllegalStateException(
                    "Graph Explorer is unavailable because this library has no prepared similarity graph"
                )
            }
            // Invalidate cached mmap since file changed
            graphIndex = null
        }
        if (requiredGraphFile.exists() && graphIndex == null) {
            try {
                val loadedGraph = GraphIndex.mmap(requiredGraphFile)
                val loadedEmbeddings = embeddingIndex
                    ?: throw IllegalStateException("Graph Explorer requires the embedding index")
                require(loadedGraph.hasSameOrderedTrackIds(loadedEmbeddings)) {
                    "Graph Explorer graph does not belong to the active embedding index"
                }
                graphIndex = loadedGraph
            } catch (e: Exception) {
                Log.w(TAG, "Failed to load graph.bin: ${e.message}")
                graphIndex = null
                throw IllegalStateException(
                    "Graph Explorer graph is invalid for the active library: ${e.message}",
                    e,
                )
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
        requestReferenceEpochSecond: Long = System.currentTimeMillis() / 1_000L,
        onProgress: ((String) -> Unit)? = null,
        onResult: (suspend (SimilarTrack) -> Unit)? = null,
        onGraphExplorationEvidence: ((GraphExplorationEvidence) -> Unit)? = null,
        onUniformShuffleIdentityEvidence: ((UniformShuffleIdentityEvidence) -> Unit)? = null,
        onMmrSelectionEvidence: ((MmrSelectionEvidence) -> Unit)? = null,
        onDppSelectionEvidence: ((DppSelectionEvidence) -> Unit)? = null,
        onRecommendationDomainEvidence: ((RecommendationDomainEvidence) -> Unit)? = null,
    ): List<SimilarTrack> = withContext(Dispatchers.Default) {
        val t0 = System.nanoTime()
        ensureIndices(requireGraph = config.selectionMode == SelectionMode.RANDOM_WALK)
        val indicesMs = (System.nanoTime() - t0) / 1_000_000

        val index = requireNotNull(embeddingIndex) {
            "Recommendation ranking requires a prepared embedding index"
        }
        val callerContext = coroutineContext
        val cancellationCheck: () -> Unit = { callerContext.ensureActive() }
        val active = requireActiveDomain(index)
        require(active.domain.containsActiveTrack(seedTrackId)) {
            "Seed track $seedTrackId is not in the active Poweramp library"
        }
        val candidateDomain = active.domain.candidateDomain(
            minimumLibraryAddedAtEpochSecond(
                config.effectiveLibraryAddedDays,
                requestReferenceEpochSecond,
            ),
        )
        val identityCatalog = active.identityCatalog
        val identityPolicy = selectionPolicy(identityCatalog)
        Log.i(TAG, "generatePlaylist: seed=$seedTrackId, mode=${config.selectionMode.name}, " +
            "drift=${config.driftEnabled}, numTracks=${config.numTracks}, " +
            "lambda=${config.diversityLambda}, candidates=" +
            "${candidateDomain.candidateIdentityCount}, active=${active.domain.activeTrackCount}/" +
            "${index.numTracks} tracks (indices=${indicesMs}ms)")

        // MMR always reranks an explicit retrieved neighborhood. Fixed DPP does the same;
        // automatic DPP uses this count only as its first proof prefix and widens as needed.
        // Closest ranks the complete library so filtering can refill without a hidden cutoff.
        val usesCandidatePool = config.selectionMode == SelectionMode.MMR ||
            config.selectionMode == SelectionMode.DPP
        val boundedIdentityDomain = config.selectionMode == SelectionMode.MMR ||
            (config.selectionMode == SelectionMode.DPP && !config.dppUsesCertifiedFullDomain)
        val eligibleCandidateIdentityCount =
            candidateDomain.eligibleCandidateIdentityCount(seedTrackId)
        val poolConfig = if (usesCandidatePool) {
            val poolDomainCount = if (boundedIdentityDomain) {
                eligibleCandidateIdentityCount
            } else {
                // Automatic DPP's prefix is an implementation detail over this request's
                // complete first-seen-filtered candidate domain.
                candidateDomain.candidateIdentityCount
            }
            val activePool = config.resolveCandidatePoolSize(poolDomainCount)
            Log.d(
                TAG,
                "Active-domain pool size: $activePool/$poolDomainCount " +
                    "(fraction=${config.effectiveWorkingCandidatePoolFraction}, " +
                    "distinctIdentities=$boundedIdentityDomain, " +
                    "certifiedDpp=${config.selectionMode == SelectionMode.DPP && config.dppUsesCertifiedFullDomain})",
            )
            config.copy(candidatePoolSize = activePool)
        } else config
        onRecommendationDomainEvidence?.invoke(
            RecommendationDomainEvidence(
                seedExcludedCandidateIdentityCount = eligibleCandidateIdentityCount,
                seedExcludedActiveIdentityCount =
                    active.domain.eligibleCandidateIdentityCount(seedTrackId),
                resolvedCandidatePoolSize = poolConfig.candidatePoolSize.takeIf {
                    boundedIdentityDomain
                },
            ),
        )
        if (eligibleCandidateIdentityCount == 0) {
            return@withContext emptyList()
        }

        val seedEmb = requireNotNull(index.getEmbeddingByTrackId(seedTrackId)) {
            "Active seed track $seedTrackId has no embedding"
        }

        // Graph Explorer uses the graph, not an embedding candidate scan.
        if (poolConfig.selectionMode == SelectionMode.RANDOM_WALK) {
            return@withContext graphExplorerPlaylist(
                seedTrackId,
                poolConfig,
                onProgress,
                onResult,
                onGraphExplorationEvidence,
                identityPolicy,
                active,
                candidateDomain,
            )
        }

        if (poolConfig.selectionMode == SelectionMode.UNIFORM_SHUFFLE) {
            return@withContext uniformShufflePlaylist(
                seedTrackId = seedTrackId,
                seedEmb = seedEmb,
                index = index,
                config = poolConfig,
                onProgress = onProgress,
                cancellationCheck = cancellationCheck,
                onIdentityEvidence = onUniformShuffleIdentityEvidence,
                identityCatalog = identityCatalog,
                identityPolicy = identityPolicy,
                active = active,
                candidateDomain = candidateDomain,
            )
        }

        // Only MMR has a defined evolving-query contract. Closest is deliberately a fixed-seed
        // baseline, while DPP and graph exploration are set-level selectors.
        val effectiveConfig = if (poolConfig.selectionMode != SelectionMode.MMR && poolConfig.driftEnabled) {
            Log.w(TAG, "${poolConfig.selectionMode}+drift is undefined; forcing batch mode")
            poolConfig.copy(driftEnabled = false)
        } else poolConfig

        val result = if (effectiveConfig.driftEnabled) {
            driftPlaylist(
                seedTrackId,
                seedEmb,
                index,
                effectiveConfig,
                onProgress,
                onResult,
                cancellationCheck,
                identityPolicy,
                active,
                candidateDomain,
            )
        } else {
            batchPlaylist(
                seedTrackId,
                seedEmb,
                index,
                effectiveConfig,
                onProgress,
                cancellationCheck,
                identityPolicy,
                identityCatalog,
                active,
                candidateDomain,
                onMmrSelectionEvidence,
                onDppSelectionEvidence,
            )
        }
        val totalMs = (System.nanoTime() - t0) / 1_000_000
        Log.i(TAG, "TIMING: generatePlaylist ${result.size} tracks in ${totalMs}ms " +
            "(mode=${effectiveConfig.selectionMode.name}, drift=${effectiveConfig.driftEnabled})")
        result
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
        cancellationCheck: () -> Unit,
        identityPolicy: StableTrackSelectionPolicy,
        active: ActiveDomainSnapshot,
        candidateDomain: RecommendationCandidateDomain,
    ): List<SimilarTrack> {
        val result = mutableListOf<SimilarTrack>()
        val selectedTrackIds = mutableListOf<Long>()
        val mmrState = MmrSelector.IncrementalState(index, cancellationCheck)
        val artistConstraint = if (config.artistLimitsEnabled) {
            PostFilter.prepare(
                active.artistCreditCatalog,
                config.maxPerArtist,
                config.minArtistSpacing,
            )
        } else {
            null
        }
        val trackCache = HashMap<Long, EmbeddedTrack?>()
        fun resolveTrack(trackId: Long): EmbeddedTrack? {
            if (!trackCache.containsKey(trackId)) {
                trackCache[trackId] = database.getTrackById(trackId)
            }
            return trackCache[trackId]
        }
        val seen = active.domain.inactiveTrackIds.toMutableSet().apply {
            addAll(identityPolicy.exclusionClosure(listOf(seedTrackId)))
        }
        var query = seedEmb
        var emaState: FloatArray? = null
        // Track the seed weight of the current query for provenance.
        // Initial query = pure seed, so seedWeight = 1.0
        var currentSeedWeight = 1f

        // Precompute seed similarities once — cheap rank lookups for all steps
        val seedSims = index.computeAllSimilarities(seedEmb)
        val candidateTrackIds = candidateDomain.orderedIdentityRepresentativeTrackIds()

        for (step in 0 until config.numTracks) {
            coroutineContext.ensureActive()
            onProgress?.invoke("Selecting track ${step + 1} of ${config.numTracks}...")

            // Rank only the request's exact Poweramp first-seen candidate domain.
            val selectionQuerySimilarities = if (step == 0) {
                seedSims
            } else {
                index.computeAllSimilarities(query)
            }
            cancellationCheck()
            val candidates = StableSimilarityTopK.select(
                orderedTrackIds = candidateTrackIds,
                similarities = candidateDomain.identityRepresentativeScoresFromFull(
                    selectionQuerySimilarities,
                ),
                topK = config.candidatePoolSize,
                rankingTieKey = active.identityCatalog::rankingTieKey,
                excludeIds = seen,
                cancellationCheck = cancellationCheck,
            ).map { it.trackId to it.score }

            if (candidates.isEmpty()) break

            val selected = mmrState.selectOne(
                candidates = candidates,
                lambda = config.diversityLambda,
                isEligible = { candidateId ->
                    if (!candidateDomain.containsIdentityRepresentative(candidateId) ||
                        !identityPolicy.canSelect(candidateId, selectedTrackIds)
                    ) {
                        false
                    } else artistConstraint?.canAdd(candidateId, selectedTrackIds) ?: true
                },
            ) ?: break

            val trackId = selected.trackId
            val score = selected.score
            check(active.domain.containsActiveTrack(trackId)) {
                "Drift selector returned inactive track $trackId"
            }
            val selectionQueryRank = active.domain.rankEligibleIdentityFromFullSimilarities(
                similarities = selectionQuerySimilarities,
                targetTrackId = trackId,
                seedTrackId = seedTrackId,
                cancellationCheck = cancellationCheck,
            )
            seen.addAll(identityPolicy.exclusionClosure(listOf(trackId)))

            val track = resolveTrack(trackId) ?: continue
            val provenance = computeProvenance(result.size, currentSeedWeight, config)

            // Compute similarityToSeed via dot product with original seed
            val trackEmb = index.getEmbeddingByTrackId(trackId)
            val simToSeed = if (trackEmb != null) dotProduct(seedEmb, trackEmb) else score

            val seedRank = active.domain.rankEligibleIdentityFromFullSimilarities(
                similarities = seedSims,
                targetTrackId = trackId,
                seedTrackId = seedTrackId,
                cancellationCheck = cancellationCheck,
            )
            val similarTrack = SimilarTrack(
                track = track,
                similarity = score,
                similarityToSeed = simToSeed,
                candidateRank = selected.candidateRank,
                seedRank = seedRank,
                driftRank = selectionQueryRank,
                provenance = provenance,
                mmrEvidence = selected.mmrSelectionEvidence?.toTrackEvidence(),
            )
            result.add(similarTrack)
            selectedTrackIds.add(trackId)
            onResult?.invoke(similarTrack)

            // Update query for next step
            if (trackEmb != null) {
                mmrState.recordSelection(trackId)
                val driftResult = DriftEngine.updateQuery(
                    seedEmb, trackEmb, emaState, step, config
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
        cancellationCheck: () -> Unit,
        identityPolicy: StableTrackSelectionPolicy,
        identityCatalog: StableTrackIdentityCatalog,
        active: ActiveDomainSnapshot,
        candidateDomain: RecommendationCandidateDomain,
        onMmrSelectionEvidence: ((MmrSelectionEvidence) -> Unit)?,
        onDppSelectionEvidence: ((DppSelectionEvidence) -> Unit)?,
    ): List<SimilarTrack> {
        onProgress?.invoke(
            "Computing seed similarity across ${index.numTracks} indexed embeddings and " +
                "ranking ${candidateDomain.candidateIdentityCount} selected candidate identities...",
        )

        // Stage 1: Retrieve candidates
        val t1 = System.nanoTime()
        val seedSimilarities = index.computeAllSimilarities(seedEmb)
        cancellationCheck()
        val excludedRows = identityPolicy.exclusionClosure(listOf(seedTrackId))
        val certifiedDpp = config.selectionMode == SelectionMode.DPP &&
            config.dppUsesCertifiedFullDomain
        val retrievalCount = if (
            config.selectionMode == SelectionMode.CLOSEST || certifiedDpp
        ) {
            candidateDomain.candidateIdentityCount
        } else {
            config.candidatePoolSize
        }
        val rankedCandidates = StableSimilarityTopK.select(
            orderedTrackIds = candidateDomain.orderedIdentityRepresentativeTrackIds(),
            similarities = candidateDomain.identityRepresentativeScoresFromFull(
                seedSimilarities,
            ),
            topK = retrievalCount,
            rankingTieKey = identityCatalog::rankingTieKey,
            excludeIds = excludedRows,
            cancellationCheck = cancellationCheck,
        )
        val candidates = rankedCandidates.map { it.trackId to it.score }
        val retrieveMs = (System.nanoTime() - t1) / 1_000_000
        Log.d(TAG, "Batch retrieve: ${candidates.size} candidates in ${retrieveMs}ms " +
            when {
                config.selectionMode == SelectionMode.CLOSEST ->
                    "(complete selected candidate domain)"
                certifiedDpp -> "(full domain for adaptive DPP proof)"
                else -> "(pool=${config.candidatePoolSize})"
            })

        if (candidates.isEmpty()) return emptyList()

        // Stage 2: Select using algorithm
        onProgress?.invoke(
            "Selecting up to ${config.numTracks} tracks with " +
                "${SelectionControlText.modeLabel(config.selectionMode)} from " +
                "${candidates.size} ranked candidates...",
        )
        val t2 = System.nanoTime()
        val trackCache = HashMap<Long, EmbeddedTrack?>()
        fun resolveTrack(trackId: Long): EmbeddedTrack? {
            if (!trackCache.containsKey(trackId)) {
                trackCache[trackId] = database.getTrackById(trackId)
            }
            return trackCache[trackId]
        }
        val artistConstraint = if (config.artistLimitsEnabled) {
            PostFilter.prepare(
                active.artistCreditCatalog,
                config.maxPerArtist,
                config.minArtistSpacing,
            )
        } else {
            null
        }
        val isEligible: (Long, List<Long>) -> Boolean = { candidateId, selectedIds ->
            if (!candidateDomain.containsIdentityRepresentative(candidateId) ||
                !identityPolicy.canSelect(candidateId, selectedIds)
            ) {
                false
            } else artistConstraint?.canAdd(candidateId, selectedIds) ?: true
        }
        val selected: List<SelectedTrack> = when (config.selectionMode) {
            SelectionMode.CLOSEST -> {
                ClosestSelector.select(candidates, config.numTracks, isEligible)
            }
            SelectionMode.MMR -> MmrSelector.selectBatch(
                candidates,
                config.numTracks,
                index,
                config.diversityLambda,
                isEligible,
                onMmrSelectionEvidence,
                cancellationCheck,
            )
            SelectionMode.DPP -> if (config.dppUsesCertifiedFullDomain) {
                val certified = DppSelector.selectBatchCertified(
                    candidates = candidates,
                    numSelect = config.numTracks,
                    index = index,
                    initialCandidateCount = config.candidatePoolSize.coerceAtMost(candidates.size),
                    qualityExponent = config.effectiveDppQualityExponent,
                    cancellationCheck = cancellationCheck,
                    isEligible = isEligible,
                )
                Log.i(
                    TAG,
                    "Certified DPP: attempts=${certified.evidence.attemptedCandidateCounts}, " +
                        "final=${certified.evidence.finalCandidateCount}/" +
                        "${certified.evidence.totalCandidateCount}, " +
                        "unseenBound=${certified.evidence.finalUnseenGainUpperBound}, " +
                        "full=${certified.evidence.usedFullDomain}",
                )
                onDppSelectionEvidence?.invoke(
                    DppSelectionEvidence(
                        completeCandidateDomainCount = certified.evidence.totalCandidateCount,
                        initialWorkingCandidateCount = certified.evidence.initialCandidateCount,
                        attemptedCandidateCounts = certified.evidence.attemptedCandidateCounts,
                        finalWorkingCandidateCount = certified.evidence.finalCandidateCount,
                        selectedMarginalGains = certified.evidence.selectedMarginalGains,
                        finalUnseenInitialGainUpperBound =
                            certified.evidence.finalUnseenGainUpperBound,
                        usedCompleteCandidateDomain = certified.evidence.usedFullDomain,
                        reproducedFullDomainGreedySequence = true,
                    ),
                )
                certified.tracks
            } else {
                DppSelector.selectBatch(
                    candidates,
                    config.numTracks,
                    index,
                    qualityExponent = config.effectiveDppQualityExponent,
                    cancellationCheck = cancellationCheck,
                    isEligible = isEligible,
                )
            }
            // RANDOM_WALK dispatches at generatePlaylist() before reaching batchPlaylist
            else -> error("Unreachable: ${config.selectionMode}")
        }
        val selectMs = (System.nanoTime() - t2) / 1_000_000
        Log.d(TAG, "Batch select: ${selected.size} tracks via ${config.selectionMode.name} in ${selectMs}ms")

        // Selector score and seed relevance are distinct for MMR and DPP.
        val tracks = selected.mapNotNull { sel ->
            check(active.domain.containsActiveTrack(sel.trackId)) {
                "${config.selectionMode} returned inactive track ${sel.trackId}"
            }
            resolveTrack(sel.trackId)?.let { track ->
                val similarityToSeed = index.getSimFromPrecomputed(
                    seedSimilarities,
                    sel.trackId,
                )
                SimilarTrack(
                    track = track,
                    similarity = sel.score,
                    similarityToSeed = similarityToSeed,
                    candidateRank = sel.candidateRank,
                    seedRank = active.domain.rankEligibleIdentityFromFullSimilarities(
                        similarities = seedSimilarities,
                        targetTrackId = sel.trackId,
                        seedTrackId = seedTrackId,
                        cancellationCheck = cancellationCheck,
                    ),
                    mmrEvidence = sel.mmrSelectionEvidence?.toTrackEvidence(),
                )
            }
        }

        return tracks
    }

    /**
     * Equal-opportunity library shuffle with an explicit replay seed.
     *
     * Embeddings do not decide membership; they are read only to preserve the useful
     * distance-from-seed evidence shown for every queued track.
     */
    private fun uniformShufflePlaylist(
        seedTrackId: Long,
        seedEmb: FloatArray,
        index: EmbeddingIndex,
        config: RadioConfig,
        onProgress: ((String) -> Unit)?,
        cancellationCheck: () -> Unit,
        onIdentityEvidence: ((UniformShuffleIdentityEvidence) -> Unit)?,
        identityCatalog: StableTrackIdentityCatalog,
        identityPolicy: StableTrackSelectionPolicy,
        active: ActiveDomainSnapshot,
        candidateDomain: RecommendationCandidateDomain,
    ): List<SimilarTrack> {
        onProgress?.invoke("Shuffling indexed library...")
        cancellationCheck()

        val trackIds = candidateDomain.orderedIdentityRepresentativeTrackIds()
        val seedSimilarities = index.computeAllSimilarities(seedEmb)
        val trackCache = HashMap<Long, EmbeddedTrack?>()
        fun resolveTrack(trackId: Long): EmbeddedTrack? {
            if (!trackCache.containsKey(trackId)) {
                trackCache[trackId] = database.getTrackById(trackId)
            }
            return trackCache[trackId]
        }
        val isEligible: (Long, List<Long>) -> Boolean = { candidateId, selectedIds ->
            cancellationCheck()
            if (!candidateDomain.containsIdentityRepresentative(candidateId)) {
                false
            } else if (!config.artistLimitsEnabled) {
                resolveTrack(candidateId) != null
            } else {
                val candidate = resolveTrack(candidateId)
                val selectedTracks = selectedIds.mapNotNull(::resolveTrack)
                candidate != null && PostFilter.canAdd(
                    candidate,
                    selectedTracks,
                    config.maxPerArtist,
                    config.minArtistSpacing,
                )
            }
        }

        val picks = UniformShuffleSelector.select(
            trackIds = trackIds,
            numSelect = config.numTracks,
            seed = config.effectiveShuffleSeed,
            identityKeyForTrack = identityCatalog::shuffleIdentityKey,
            excludeIds = identityPolicy.exclusionClosure(listOf(seedTrackId)),
            isEligible = isEligible,
        )
        val result = picks.mapNotNull { pick ->
            check(active.domain.containsActiveTrack(pick.trackId)) {
                "Uniform shuffle returned inactive track ${pick.trackId}"
            }
            resolveTrack(pick.trackId)?.let { track ->
                val similarity = index.getSimFromPrecomputed(seedSimilarities, pick.trackId)
                SimilarTrack(
                    track = track,
                    similarity = similarity,
                    similarityToSeed = similarity,
                    candidateRank = pick.shuffleRank,
                    seedRank = active.domain.rankEligibleIdentityFromFullSimilarities(
                        similarities = seedSimilarities,
                        targetTrackId = pick.trackId,
                        seedTrackId = seedTrackId,
                        cancellationCheck = cancellationCheck,
                    ),
                )
            }
        }
        val seedIdentityTokens = identityPolicy.exclusionClosure(listOf(seedTrackId))
            .mapTo(HashSet()) { trackId ->
                identityCatalog.shuffleIdentityKey(trackId).identityToken
            }
        val rankedIdentityTokens = HashSet<String>(trackIds.size)
        var activeStable = 0
        trackIds.forEach { trackId ->
            val identity = identityCatalog.shuffleIdentityKey(trackId)
            if (identityCatalog.stableTrackSpanId(trackId) != null) activeStable++
            if (identity.identityToken !in seedIdentityTokens) {
                rankedIdentityTokens += identity.identityToken
            }
        }
        val selectedStable = result.count { track ->
            identityCatalog.stableTrackSpanId(track.track.id) != null
        }
        val identityEvidence = UniformShuffleIdentityEvidence(
            generationBinding = identityCatalog.binding,
            libraryTrackCount = trackIds.size,
            stableLibraryTrackCount = activeStable,
            legacyLibraryTrackCount = trackIds.size - activeStable,
            selectedStableTrackCount = selectedStable,
            selectedLegacyTrackCount = result.size - selectedStable,
            rankedCandidateCount = rankedIdentityTokens.size,
        )
        onIdentityEvidence?.invoke(identityEvidence)
        Log.i(
            TAG,
            "Uniform shuffle selected ${result.size}/${config.numTracks} tracks " +
                "from ${trackIds.size} selected candidate identities " +
                "with seed=${config.effectiveShuffleSeed}; " +
                "stableIdentity=${identityEvidence.selectedStableTrackCount}/${result.size}, " +
                "generation=${identityCatalog.binding.generationId}",
        )
        return result
    }

    /** Exact deterministic graph-terminal ranking using the precomputed kNN graph. */
    private suspend fun graphExplorerPlaylist(
        seedTrackId: Long,
        config: RadioConfig,
        onProgress: ((String) -> Unit)?,
        onResult: (suspend (SimilarTrack) -> Unit)?,
        onGraphExplorationEvidence: ((GraphExplorationEvidence) -> Unit)?,
        identityPolicy: StableTrackSelectionPolicy,
        active: ActiveDomainSnapshot,
        candidateDomain: RecommendationCandidateDomain,
    ): List<SimilarTrack> {
        val index = requireNotNull(embeddingIndex) {
            "Graph Explorer requires a prepared embedding index"
        }
        val callerContext = coroutineContext
        val cancellationCheck: () -> Unit = { callerContext.ensureActive() }
        onProgress?.invoke(
            "Aligning the similarity graph with " +
                "${active.domain.activeCandidateIdentityCount} active recording identities...",
        )
        val activeTopology = requireActiveGraphTopology(index, active, cancellationCheck)

        onProgress?.invoke("Ranking tracks across the similarity graph...")
        val tWalk = System.nanoTime()
        val alpha = config.walkRestartAlpha
        val graphSeedTrackId = active.domain.activeIdentityRepresentativeTrackId(seedTrackId)
        val exploration = GraphExplorerSelector.computeCancellable(
            activeTopology.topology,
            graphSeedTrackId,
            alpha,
        )
        val walkMs = (System.nanoTime() - tWalk) / 1_000_000
        val seedExclusions = identityPolicy.exclusionClosure(listOf(seedTrackId))
        val seedIdentityToken = active.identityCatalog
            .visibleResultIdentity(seedTrackId)
            .identityToken
        val rankedIdentityTokens = HashSet<String>(candidateDomain.candidateIdentityCount)
        val identityRanking = ArrayList<GraphExplorerTrackScore>(
            exploration.ranking.size,
        )
        exploration.ranking.forEach { score ->
            val identityToken = active.identityCatalog
                .visibleResultIdentity(score.trackId)
                .identityToken
            check(identityToken != seedIdentityToken) {
                "Identity graph returned the excluded seed recording"
            }
            val candidateRepresentative =
                candidateDomain.representativeForVisibleIdentity(score.trackId)
                ?: return@forEach
            check(rankedIdentityTokens.add(identityToken)) {
                "Identity graph repeats queue-visible recording $identityToken"
            }
            identityRanking += score.copy(trackId = candidateRepresentative)
        }
        onGraphExplorationEvidence?.invoke(
            GraphExplorationEvidence(
                expectedRouteLinks = exploration.expectedRouteLinks,
                excludedSeedProbability = exploration.excludedSeedProbability,
                totalTerminalProbability = exploration.totalTerminalProbability,
                numericMassError = exploration.numericMassError,
                evaluatedLinks = exploration.evaluatedLinks,
                rankedCandidateCount = identityRanking.size,
            )
        )
        Log.d(
            TAG,
            "Graph Explorer: ${exploration.ranking.size} ranked nodes in ${walkMs}ms " +
                "(stop=$alpha, expectedLinks=${exploration.expectedRouteLinks}, " +
                "massError=${exploration.numericMassError}, " +
                "activeNodes=${activeTopology.evidence.nodeCount}, " +
                "candidateIdentities=${identityRanking.size}, " +
                "repairedRows=${activeTopology.evidence.affectedRowCount})",
        )

        val seedEmb = requireNotNull(index.getEmbeddingByTrackId(seedTrackId)) {
            "Active Graph Explorer seed $seedTrackId has no embedding"
        }

        // Single scan serves both simToSeed and seedRank
        val seedSims = index.computeAllSimilarities(seedEmb)

        // Graph Explorer uses full support because its discovery power comes from transitive
        // connections. Resolve metadata only until the requested queue is full; exact support
        // can include most of the library, and one SQLite lookup per reachable node is wasteful.
        val ranked = identityRanking
        val tracks = ArrayList<SimilarTrack>(config.numTracks)
        val selectedMetadata = ArrayList<EmbeddedTrack>(config.numTracks)
        val selectedIds = ArrayList<Long>(config.numTracks)
        for (i in ranked.indices) {
            coroutineContext.ensureActive()
            val score = ranked[i]
            val trackId = score.trackId
            check(candidateDomain.containsIdentityRepresentative(trackId)) {
                "Graph Explorer returned a track outside the selected added-date domain: $trackId"
            }
            if (trackId in seedExclusions || !identityPolicy.canSelect(trackId, selectedIds)) {
                continue
            }
            val track = database.getTrackById(trackId) ?: continue
            if (
                config.artistLimitsEnabled &&
                !PostFilter.canAdd(
                    track,
                    selectedMetadata,
                    config.maxPerArtist,
                    config.minArtistSpacing,
                )
            ) {
                continue
            }
            val simToSeed = index.getSimFromPrecomputed(seedSims, trackId)
            tracks += SimilarTrack(
                track = track,
                similarity = score.terminalProbability.toFloat(),
                similarityToSeed = simToSeed,
                candidateRank = i + 1,
                graphTerminalProbability = score.terminalProbability,
                graphExpectedRouteLinks = score.expectedRouteLinks,
            )
            selectedMetadata += track
            selectedIds += trackId
            if (tracks.size == config.numTracks) break
        }

        // Compute seedRank only for the ~50 filtered tracks (O(N) per call)
        val withRanks = tracks.map { track ->
            track.copy(
                seedRank = active.domain.rankEligibleIdentityFromFullSimilarities(
                    similarities = seedSims,
                    targetTrackId = track.track.id,
                    seedTrackId = seedTrackId,
                    cancellationCheck = cancellationCheck,
                ),
            )
        }

        // Stream results if callback provided
        onResult?.let { callback ->
            for (track in withRanks) {
                callback(track)
            }
        }

        Log.d(TAG, "Graph Explorer: ${withRanks.size} tracks")
        return withRanks
    }

    private fun MmrSelectionEvidence.toTrackEvidence(): MmrTrackEvidence = MmrTrackEvidence(
        selectionStep = step,
        queryRelevance = relevance,
        greatestSelectedOverlap = maximumSelectedSimilarity,
        greatestOverlapTrackId = maximumSelectedTrackId,
        objective = objective,
        candidateRank = candidateRank,
    )

    /** Generate one truthful All-of radio from the exact displayed Find Music request. */
    suspend fun generateComposedAllOfPlaylist(
        seeds: List<SeedSpec>,
        querySpec: FindMusicQuerySpec,
        config: RadioConfig,
        requestReferenceEpochSecond: Long = System.currentTimeMillis() / 1_000L,
        onProgress: ((String) -> Unit)? = null,
    ): ComposedPlaylistResult = withContext(Dispatchers.Default) {
        require(querySpec.operator == FindMusicOperator.ALL_OF) {
            "Radio is defined only for the All-of composed objective"
        }
        val t0 = System.nanoTime()
        ensureIndices(onProgress)

        val index = requireNotNull(embeddingIndex) {
            "Composed recommendation ranking requires a prepared embedding index"
        }
        val active = requireActiveDomain(index)
        val candidateDomain = active.domain.candidateDomain(
            minimumLibraryAddedAtEpochSecond(
                querySpec.effectiveLibraryAddedDays,
                requestReferenceEpochSecond,
            ),
        )

        val validSeeds = seeds.filter { it.weight != 0f }
        if (validSeeds.isEmpty()) return@withContext emptyComposedPlaylist(config.numTracks)
        validSeeds.mapNotNull(SeedSpec::trackId).forEach { trackId ->
            require(active.domain.containsActiveTrack(trackId)) {
                "Composed song ingredient $trackId is not in the active Poweramp library"
            }
        }
        if (candidateDomain.candidateIdentityCount == 0) {
            return@withContext emptyComposedPlaylist(config.numTracks)
        }
        val identityCatalog = active.identityCatalog
        val identityPolicy = selectionPolicy(identityCatalog)

        Log.i(TAG, "generateComposedAllOfPlaylist: ${validSeeds.size} seeds, " +
            "numTracks=${config.numTracks}")

        onProgress?.invoke(
            "Comparing ${validSeeds.size} All-of ingredients with ${index.numTracks} indexed " +
                "embeddings and ranking ${candidateDomain.candidateIdentityCount} selected " +
                "candidate identities...",
        )

        val excludeIds = identityPolicy.exclusionClosure(validSeeds.mapNotNull { it.trackId })
        val seedVectors = validSeeds.map { it.embedding to it.weight }
        // Rank once, then scan the deterministic objective order until verified identity and
        // artist constraints have filled N or the corpus is exhausted. This has no 3x refill
        // heuristic and never repeats text/audio similarity work.
        val ranking = GeoMeanSelector.computeAllOfRankingSnapshot(
            index = index,
            seeds = seedVectors,
            excludeTrackIds = excludeIds,
            identityCatalog = identityCatalog,
            includedTrackIds = candidateDomain.eligibleTrackIds(),
            cancellationCheck = { coroutineContext.ensureActive() },
        )
        val artistConstraint = if (config.artistLimitsEnabled) {
            PostFilter.prepare(
                active.artistCreditCatalog,
                config.maxPerArtist,
                config.minArtistSpacing,
            )
        } else {
            null
        }
        val plannedRanking:
            Iterable<com.powerampstartradio.similarity.algorithms.RankedComposedRow> =
            when (querySpec.textResultPlanner) {
            FindMusicTextResultPlanner.CLOSEST -> ranking
            FindMusicTextResultPlanner.VARIED_ALL_OF_DPP -> {
                val plan = FindMusicAllOfQueuePlanner.plan(
                    completeObjectiveRanking = ranking.map { it.row }.toList(),
                    requestedResultCount = config.numTracks,
                    embeddingIndex = index,
                    cancellationCheck = { coroutineContext.ensureActive() },
                    isEligible = { candidateId, selectedIds ->
                        identityPolicy.canSelect(candidateId, selectedIds) &&
                            (artistConstraint?.canAdd(candidateId, selectedIds) ?: true)
                    },
                )
                plan.selections.map { selected ->
                    com.powerampstartradio.similarity.algorithms.RankedComposedRow(
                        objectiveRank = selected.originalAllOfObjectiveRank,
                        row = selected.row,
                    )
                }
            }
            FindMusicTextResultPlanner.VARIED_DPP ->
                error("Text Varied cannot generate an All-of playlist")
        }
        val metadataByTrackId = HashMap<Long, EmbeddedTrack?>()
        val selectedMetadata = ArrayList<EmbeddedTrack>(config.numTracks)
        val reduction = StableVisibleResultReducer.reduce(
            rankedItems = plannedRanking,
            requestedVisibleCount = config.numTracks,
            identityOf = { candidate ->
                identityCatalog.visibleResultIdentity(candidate.row.trackId)
            },
            isEligible = { candidate ->
                coroutineContext.ensureActive()
                check(active.domain.containsActiveTrack(candidate.row.trackId)) {
                    "All-of ranking returned inactive track ${candidate.row.trackId}"
                }
                val track = metadataByTrackId.getOrPut(candidate.row.trackId) {
                    database.getTrackById(candidate.row.trackId)
                } ?: return@reduce false
                val accepted =
                    querySpec.textResultPlanner ==
                    FindMusicTextResultPlanner.VARIED_ALL_OF_DPP ||
                        !config.artistLimitsEnabled ||
                        PostFilter.canAdd(
                            track,
                            selectedMetadata,
                            config.maxPerArtist,
                            config.minArtistSpacing,
                        )
                if (accepted) selectedMetadata += track
                accepted
            },
        )
        val tracks = reduction.items.mapNotNull { candidate ->
            check(active.domain.containsActiveTrack(candidate.row.trackId)) {
                "All-of reduction returned inactive track ${candidate.row.trackId}"
            }
            val track = metadataByTrackId[candidate.row.trackId]
                ?: database.getTrackById(candidate.row.trackId)
            track?.let {
                val evidence = ComposedTrackEvidence(
                    objectiveRank = candidate.objectiveRank,
                    objectiveScore = candidate.row.objectiveScore,
                    ingredientPercentiles = candidate.row.anchorPercentiles,
                )
                SimilarTrack(
                    track = it,
                    similarity = candidate.row.objectiveScore,
                    similarityToSeed = candidate.row.objectiveScore,
                    composedEvidence = evidence,
                )
            }
        }

        val totalMs = (System.nanoTime() - t0) / 1_000_000
        Log.i(
            TAG,
            "TIMING: generateComposedAllOfPlaylist ${tracks.size} tracks from " +
                "${reduction.scannedRowCount} ranked rows in ${totalMs}ms",
        )

        ComposedPlaylistResult(
            tracks = tracks,
            stableResultReduction = StableResultReductionEvidence(
                identityPolicyVersion = StableVisibleResultReducer.IDENTITY_POLICY_VERSION,
                requestedVisibleCount = reduction.requestedVisibleCount,
                scannedRowCount = reduction.scannedRowCount,
                collapsedEquivalentCount = reduction.collapsedEquivalentCount,
            ),
            objectiveRankingDomainCount = ranking.size,
        )
    }

    private fun emptyComposedPlaylist(requestedVisibleCount: Int) = ComposedPlaylistResult(
        tracks = emptyList(),
        stableResultReduction = StableResultReductionEvidence(
            identityPolicyVersion = StableVisibleResultReducer.IDENTITY_POLICY_VERSION,
            requestedVisibleCount = requestedVisibleCount,
            scannedRowCount = 0,
            collapsedEquivalentCount = 0,
        ),
        objectiveRankingDomainCount = 0,
    )

    private fun selectionPolicy(
        catalog: StableTrackIdentityCatalog,
    ): StableTrackSelectionPolicy = StableTrackSelectionPolicy(
        identityForTrack = catalog::visibleResultIdentity,
        equivalentTrackIds = catalog::equivalentVisibleTrackIds,
    )

    private data class ActiveDomainSnapshot(
        val embeddingIndex: EmbeddingIndex,
        val sourceCatalog: V2ActiveLibraryCatalog,
        val identityCatalog: StableTrackIdentityCatalog,
        val domain: ActiveRecommendationDomain,
        val orderedDatabaseTrackIds: LongArray,
        val orderedActiveTrackIds: LongArray,
        val orderedActiveIdentityRepresentativeTrackIds: LongArray,
        val activeTrackIds: Set<Long>,
        val artistCreditCatalog: ArtistCreditCatalog,
    )

    private data class ActiveGraphTopologySnapshot(
        val graphIndex: GraphIndex,
        val embeddingIndex: EmbeddingIndex,
        val activeDomainBinding: ActiveRecommendationDomainBinding,
        val topology: ActiveDomainGraphTopology,
    )

    /**
     * Compute quality metrics for a completed queue.
     *
     * @param tracks The queued tracks with similarity scores
     * @return QueueMetrics with artist count, cluster spread, and sim range
     */
    fun computeQueueMetrics(tracks: List<SimilarTrack>): QueueMetrics {
        if (tracks.isEmpty()) return QueueMetrics(0, 0, 0, 0)

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
            simMin = minSim,
            simMax = maxSim,
        )
    }

    private fun dotProduct(a: FloatArray, b: FloatArray): Float {
        var sum = 0f
        for (i in a.indices) sum += a[i] * b[i]
        return sum
    }
}
