package com.powerampstartradio.data

/**
 * Immutable, allocation-free topology used by deterministic uniform graph traversal.
 *
 * [neighborIndices] is row-major `[nodeCount * neighborsPerNode]`. An entry is `-1`
 * when the corresponding graph slot is invalid. Duplicate slots remain duplicates:
 * the released explorer sampled slots uniformly, so removing duplicates would change
 * its transition probabilities.
 */
class UniformGraphSnapshot internal constructor(
    internal val trackIds: LongArray,
    internal val neighborIndices: IntArray,
    val neighborsPerNode: Int,
    private val trackIdToIndex: Map<Long, Int>,
    copyArrays: Boolean,
) {
    val nodeCount: Int = trackIds.size

    internal val ids: LongArray = if (copyArrays) trackIds.copyOf() else trackIds
    internal val neighbors: IntArray =
        if (copyArrays) neighborIndices.copyOf() else neighborIndices

    /** Number of valid continuation slots for every `(previous, current)` edge state. */
    internal val nonBacktrackingChoiceCounts: ByteArray

    init {
        require(neighborsPerNode > 0) { "neighborsPerNode must be positive" }
        require(neighborIndices.size.toLong() == trackIds.size.toLong() * neighborsPerNode) {
            "neighborIndices size does not match nodeCount * neighborsPerNode"
        }
        require(trackIdToIndex.size == trackIds.size) { "track IDs must be unique" }
        for (neighbor in neighbors) {
            require(neighbor == -1 || neighbor in ids.indices) {
                "neighbor index $neighbor is outside the graph"
            }
        }

        nonBacktrackingChoiceCounts = ByteArray(neighbors.size)
        for (state in neighbors.indices) {
            val previous = state / neighborsPerNode
            val current = neighbors[state]
            if (current < 0) continue

            var count = 0
            val rowStart = current * neighborsPerNode
            for (slot in 0 until neighborsPerNode) {
                val next = neighbors[rowStart + slot]
                if (next >= 0 && next != previous) count++
            }
            require(count <= 255) { "graph degree exceeds byte-backed choice count" }
            nonBacktrackingChoiceCounts[state] = count.toByte()
        }
    }

    fun indexOfTrackId(trackId: Long): Int = trackIdToIndex[trackId] ?: -1

    fun trackIdAt(index: Int): Long = ids[index]

    companion object {
        /** Pure constructor for tests, benchmarks, and non-Android graph tooling. */
        fun fromRaw(
            trackIds: LongArray,
            neighborIndices: IntArray,
            neighborsPerNode: Int,
        ): UniformGraphSnapshot {
            val idToIndex = HashMap<Long, Int>(trackIds.size * 2)
            trackIds.forEachIndexed { index, trackId ->
                require(idToIndex.put(trackId, index) == null) {
                    "duplicate track ID $trackId"
                }
            }
            return UniformGraphSnapshot(
                trackIds = trackIds,
                neighborIndices = neighborIndices,
                neighborsPerNode = neighborsPerNode,
                trackIdToIndex = idToIndex,
                copyArrays = true,
            )
        }
    }
}
