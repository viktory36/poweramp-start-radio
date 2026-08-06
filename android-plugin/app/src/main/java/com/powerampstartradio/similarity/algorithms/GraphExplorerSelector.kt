package com.powerampstartradio.similarity.algorithms

import com.powerampstartradio.data.GraphIndex
import com.powerampstartradio.data.UniformGraphSnapshot
import kotlinx.coroutines.ensureActive
import kotlin.coroutines.coroutineContext
import kotlin.math.abs

/** One track's exact terminal-position evidence in deterministic Graph Explorer. */
data class GraphExplorerTrackScore(
    val trackId: Long,
    val terminalProbability: Double,
    /** Expected followed links, conditioned on the walk terminating at this track. */
    val expectedRouteLinks: Double,
)

/** Exact integration result, including probability hidden by the seed-exclusion rule. */
data class GraphExplorerResult(
    /** Seed-excluding results ordered by probability, then canonical ascending track ID. */
    val ranking: List<GraphExplorerTrackScore>,
    /** Terminal probability before excluding the seed. Numerically equal to one. */
    val totalTerminalProbability: Double,
    /** Probability that terminated back at the seed and is deliberately not returned. */
    val excludedSeedProbability: Double,
    /** Expected number of followed links over all walks, including seed terminals. */
    val expectedRouteLinks: Double,
    /** Largest link depth that carried probability (zero only for a dead-end seed). */
    val evaluatedLinks: Int,
    val numericMassError: Double,
)

/**
 * Exact terminal-position distribution for the released non-backtracking walk semantics.
 *
 * Probability is propagated over directed `(previous, current)` edge states. Every walk:
 *
 * 1. starts at the seed and uniformly follows one valid outgoing graph slot;
 * 2. stops after that link with probability [stopProbability];
 * 3. otherwise follows a uniformly chosen outgoing slot excluding only `previous`;
 * 4. assigns all remaining probability at a dead end or after [MAX_LINKS] links.
 *
 * The result is deterministic. Stored graph weights are intentionally ignored because the
 * released selector sampled valid slots uniformly. Duplicate slots therefore retain their
 * multiplicity. All arithmetic is Float64 and total mass must remain within
 * [MASS_TOLERANCE] of one.
 */
object GraphExplorerSelector {
    const val MAX_LINKS = 100
    const val MASS_TOLERANCE = 5e-12
    private const val CANCELLATION_CHECK_MASK = 0x0fff

    fun compute(
        graph: GraphIndex,
        seedTrackId: Long,
        stopProbability: Float = 0.5f,
        cancellationCheck: () -> Unit = {},
    ): GraphExplorerResult = computeNative(
        topology = graph.uniformTopology(),
        seedTrackId = seedTrackId,
        stopProbability = stopProbability,
        cancellationCheck = cancellationCheck,
    )

    /** Pure Float64 reference solver used by host tests and parity checks. */
    fun compute(
        topology: UniformGraphSnapshot,
        seedTrackId: Long,
        stopProbability: Float = 0.5f,
        cancellationCheck: () -> Unit = {},
    ): GraphExplorerResult {
        require(stopProbability.isFinite() && stopProbability in 0f..1f) {
            "stopProbability must be finite and within [0, 1]"
        }
        val seed = topology.indexOfTrackId(seedTrackId)
        require(seed >= 0) { "seed track $seedTrackId is absent from the graph" }
        cancellationCheck()

        val propagation = propagateReference(topology, seed, stopProbability, cancellationCheck)
        return buildResult(topology, seed, propagation, cancellationCheck)
    }

    /** Native production kernel; semantically identical to [compute]'s pure reference. */
    fun computeNative(
        topology: UniformGraphSnapshot,
        seedTrackId: Long,
        stopProbability: Float = 0.5f,
        cancellationCheck: () -> Unit = {},
    ): GraphExplorerResult {
        require(stopProbability.isFinite() && stopProbability in 0f..1f) {
            "stopProbability must be finite and within [0, 1]"
        }
        val seed = topology.indexOfTrackId(seedTrackId)
        require(seed >= 0) { "seed track $seedTrackId is absent from the graph" }
        cancellationCheck()

        val propagation = NativeGraphExplorer.propagate(
            topology = topology,
            seedIndex = seed,
            stopProbability = stopProbability,
            maxLinks = MAX_LINKS,
            cancellationCheck = cancellationCheck,
        )
        return buildResult(topology, seed, propagation, cancellationCheck)
    }

    private fun propagateReference(
        topology: UniformGraphSnapshot,
        seed: Int,
        stopProbability: Float,
        cancellationCheck: () -> Unit,
    ): GraphExplorerPropagation {

        val n = topology.nodeCount
        val k = topology.neighborsPerNode
        val edgeCount = topology.neighbors.size
        val terminal = DoubleArray(n)
        val terminalLinkMass = DoubleArray(n)

        var currentProbability = DoubleArray(edgeCount)
        var nextProbability = DoubleArray(edgeCount)
        var currentStates = IntArray(edgeCount)
        var nextStates = IntArray(edgeCount)
        var currentCount = 0

        val seedRow = seed * k
        var initialChoiceCount = 0
        for (slot in 0 until k) {
            if (topology.neighbors[seedRow + slot] >= 0) initialChoiceCount++
        }

        if (initialChoiceCount == 0) {
            terminal[seed] = 1.0
        } else {
            val initialProbability = 1.0 / initialChoiceCount.toDouble()
            for (slot in 0 until k) {
                val state = seedRow + slot
                if (topology.neighbors[state] < 0) continue
                currentProbability[state] = initialProbability
                currentStates[currentCount++] = state
            }
        }

        val alpha = stopProbability.toDouble()
        val continuationScale = 1.0 - alpha
        var evaluatedLinks = 0

        for (linkCount in 1..MAX_LINKS) {
            if (currentCount == 0) break
            cancellationCheck()
            evaluatedLinks = linkCount
            var nextCount = 0

            for (activeOffset in 0 until currentCount) {
                if ((activeOffset and CANCELLATION_CHECK_MASK) == 0) cancellationCheck()

                val state = currentStates[activeOffset]
                val probability = currentProbability[state]
                currentProbability[state] = 0.0
                if (probability == 0.0) continue

                val current = topology.neighbors[state]
                check(current >= 0) { "active graph state has no destination" }
                val choiceCount = topology.nonBacktrackingChoiceCounts[state].toInt() and 0xff
                val mustTerminate = linkCount == MAX_LINKS || choiceCount == 0
                val stoppedMass = if (mustTerminate) probability else alpha * probability
                terminal[current] += stoppedMass
                terminalLinkMass[current] += linkCount.toDouble() * stoppedMass

                if (mustTerminate || continuationScale == 0.0) continue

                val contribution = probability * continuationScale / choiceCount.toDouble()
                if (contribution == 0.0) continue
                val previous = state / k
                val nextRow = current * k
                for (slot in 0 until k) {
                    val nextState = nextRow + slot
                    val following = topology.neighbors[nextState]
                    if (following < 0 || following == previous) continue
                    if (nextProbability[nextState] == 0.0) {
                        nextStates[nextCount++] = nextState
                    }
                    nextProbability[nextState] += contribution
                }
            }

            val probabilitySwap = currentProbability
            currentProbability = nextProbability
            nextProbability = probabilitySwap
            val stateSwap = currentStates
            currentStates = nextStates
            nextStates = stateSwap
            currentCount = nextCount
        }

        cancellationCheck()
        return GraphExplorerPropagation(
            terminalProbability = terminal,
            terminalLinkMass = terminalLinkMass,
            evaluatedLinks = evaluatedLinks,
        )
    }

    internal fun buildResult(
        topology: UniformGraphSnapshot,
        seed: Int,
        propagation: GraphExplorerPropagation,
        cancellationCheck: () -> Unit = {},
    ): GraphExplorerResult {
        val terminal = propagation.terminalProbability
        val terminalLinkMass = propagation.terminalLinkMass
        require(terminal.size == topology.nodeCount) { "terminal probability size mismatch" }
        require(terminalLinkMass.size == topology.nodeCount) { "terminal route mass size mismatch" }

        var totalMass = 0.0
        var totalLinkMass = 0.0
        for (node in 0 until topology.nodeCount) {
            if ((node and CANCELLATION_CHECK_MASK) == 0) cancellationCheck()
            totalMass += terminal[node]
            totalLinkMass += terminalLinkMass[node]
        }
        val massError = abs(1.0 - totalMass)
        check(massError <= MASS_TOLERANCE) {
            "Graph Explorer probability mass error $massError exceeds $MASS_TOLERANCE"
        }

        val ranking = ArrayList<GraphExplorerTrackScore>()
        for (node in 0 until topology.nodeCount) {
            if ((node and CANCELLATION_CHECK_MASK) == 0) cancellationCheck()
            if (node == seed) continue
            val probability = terminal[node]
            if (probability <= 0.0) continue
            ranking += GraphExplorerTrackScore(
                trackId = topology.trackIdAt(node),
                terminalProbability = probability,
                expectedRouteLinks = terminalLinkMass[node] / probability,
            )
        }
        cancellationCheck()
        ranking.sortWith(
            compareByDescending<GraphExplorerTrackScore> { it.terminalProbability }
                .thenBy { it.trackId }
        )
        cancellationCheck()

        return GraphExplorerResult(
            ranking = ranking,
            totalTerminalProbability = totalMass,
            excludedSeedProbability = terminal[seed],
            expectedRouteLinks = totalLinkMass / totalMass,
            evaluatedLinks = propagation.evaluatedLinks,
            numericMassError = massError,
        )
    }

    /** Coroutine-aware wrapper; native propagation observes cancellation after every link. */
    suspend fun computeCancellable(
        graph: GraphIndex,
        seedTrackId: Long,
        stopProbability: Float = 0.5f,
    ): GraphExplorerResult {
        val callerContext = coroutineContext
        return compute(graph, seedTrackId, stopProbability) {
            callerContext.ensureActive()
        }
    }

    /** Coroutine-aware production kernel over an already bound active-domain topology. */
    suspend fun computeCancellable(
        topology: UniformGraphSnapshot,
        seedTrackId: Long,
        stopProbability: Float = 0.5f,
    ): GraphExplorerResult {
        val callerContext = coroutineContext
        return computeNative(topology, seedTrackId, stopProbability) {
            callerContext.ensureActive()
        }
    }
}

internal data class GraphExplorerPropagation(
    val terminalProbability: DoubleArray,
    val terminalLinkMass: DoubleArray,
    val evaluatedLinks: Int,
)
