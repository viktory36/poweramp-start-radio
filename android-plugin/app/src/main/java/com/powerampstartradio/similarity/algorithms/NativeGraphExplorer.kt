package com.powerampstartradio.similarity.algorithms

import com.powerampstartradio.data.UniformGraphSnapshot

/** JNI boundary for the exact Graph Explorer propagation hot loop. */
internal object NativeGraphExplorer {
    init {
        System.loadLibrary("math-jni")
    }

    fun propagate(
        topology: UniformGraphSnapshot,
        seedIndex: Int,
        stopProbability: Float,
        maxLinks: Int,
        cancellationCheck: () -> Unit,
    ): GraphExplorerPropagation {
        val terminal = DoubleArray(topology.nodeCount)
        val terminalLinkMass = DoubleArray(topology.nodeCount)
        val evaluatedLinks = nativePropagate(
            neighbors = topology.neighbors,
            nonBacktrackingChoiceCounts = topology.nonBacktrackingChoiceCounts,
            nodeCount = topology.nodeCount,
            neighborsPerNode = topology.neighborsPerNode,
            seedIndex = seedIndex,
            stopProbability = stopProbability,
            maxLinks = maxLinks,
            terminalProbability = terminal,
            terminalLinkMass = terminalLinkMass,
            cancellationCheck = cancellationCheck,
        )
        check(evaluatedLinks >= 0) { "native Graph Explorer propagation failed" }
        return GraphExplorerPropagation(terminal, terminalLinkMass, evaluatedLinks)
    }

    @JvmStatic
    private external fun nativePropagate(
        neighbors: IntArray,
        nonBacktrackingChoiceCounts: ByteArray,
        nodeCount: Int,
        neighborsPerNode: Int,
        seedIndex: Int,
        stopProbability: Float,
        maxLinks: Int,
        terminalProbability: DoubleArray,
        terminalLinkMass: DoubleArray,
        cancellationCheck: () -> Unit,
    ): Int
}
