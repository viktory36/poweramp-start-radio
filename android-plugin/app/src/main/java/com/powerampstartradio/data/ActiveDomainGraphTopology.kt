package com.powerampstartradio.data

data class ActiveDomainGraphRepairEvidence(
    val nodeCount: Int,
    val neighborsPerNode: Int,
    val affectedRowCount: Int,
    val preservedRowCount: Int,
    val invalidatedSlotCount: Int,
)

data class ActiveDomainGraphTopology(
    val topology: UniformGraphSnapshot,
    val evidence: ActiveDomainGraphRepairEvidence,
)

/** Builds the exact KNN graph induced by a caller-provided active track subset. */
object ActiveDomainGraphTopologyBuilder {
    fun build(
        graph: GraphIndex,
        embeddings: EmbeddingIndex,
        /** Sorted active graph node IDs. */
        orderedActiveTrackIds: LongArray,
        cancellationCheck: () -> Unit = {},
    ): ActiveDomainGraphTopology {
        require(graph.hasSameOrderedTrackIds(embeddings)) {
            "Graph and embedding index do not share one ordered generation"
        }
        val base = graph.uniformTopology()
        return buildFromNodeSubset(
            base = base,
            orderedNodeTrackIds = orderedActiveTrackIds,
            exactTopK = { trackId, k, exclusions ->
                val query = requireNotNull(embeddings.getEmbeddingByTrackId(trackId)) {
                    "Active graph track $trackId has no embedding"
                }
                embeddings.findTopK(
                    query = query,
                    topK = k,
                    excludeIds = exclusions,
                    cancellationCheck = cancellationCheck,
                ).map { (neighborId, _) -> neighborId }
            },
            cancellationCheck = cancellationCheck,
        )
    }

    /** Pure subset planner used by host tests and the mmap-backed production wrapper. */
    internal fun buildFromNodeSubset(
        base: UniformGraphSnapshot,
        orderedNodeTrackIds: LongArray,
        exactTopK: (trackId: Long, k: Int, exclusions: Set<Long>) -> List<Long>,
        cancellationCheck: () -> Unit = {},
    ): ActiveDomainGraphTopology {
        require(orderedNodeTrackIds.isNotEmpty()) { "Active graph domain is empty" }
        require(orderedNodeTrackIds.isStrictlyIncreasing()) {
            "Active graph track IDs must be strictly increasing"
        }

        val k = base.neighborsPerNode
        require(orderedNodeTrackIds.size > k) {
            "Active graph domain must contain more tracks than its neighbor count"
        }

        val nodeIndexByTrackId = HashMap<Long, Int>(orderedNodeTrackIds.size * 2)
        orderedNodeTrackIds.forEachIndexed { index, trackId ->
            require(nodeIndexByTrackId.put(trackId, index) == null) {
                "Active graph domain repeats track $trackId"
            }
        }

        val oldToNode = IntArray(base.nodeCount) { -1 }
        val nodeToOld = IntArray(orderedNodeTrackIds.size) { -1 }
        for (oldIndex in 0 until base.nodeCount) {
            if ((oldIndex and CANCELLATION_CHECK_MASK) == 0) cancellationCheck()
            val nodeIndex = nodeIndexByTrackId[base.ids[oldIndex]] ?: continue
            oldToNode[oldIndex] = nodeIndex
            nodeToOld[nodeIndex] = oldIndex
        }
        require(nodeToOld.all { it >= 0 }) {
            "Active graph domain contains a track outside the graph generation"
        }

        val excludedTrackIds = HashSet<Long>(base.nodeCount - orderedNodeTrackIds.size)
        for (oldIndex in 0 until base.nodeCount) {
            if (oldToNode[oldIndex] < 0) excludedTrackIds += base.ids[oldIndex]
        }

        val affected = BooleanArray(orderedNodeTrackIds.size)
        var affectedRowCount = 0
        var invalidatedSlotCount = 0
        for (nodeRow in orderedNodeTrackIds.indices) {
            if ((nodeRow and CANCELLATION_CHECK_MASK) == 0) cancellationCheck()
            val oldRow = nodeToOld[nodeRow]
            val oldRowStart = oldRow * k
            val seenNeighbors = IntArray(k)
            var seenCount = 0
            for (slot in 0 until k) {
                val oldNeighbor = base.neighbors[oldRowStart + slot]
                val nodeNeighbor = if (oldNeighbor >= 0) oldToNode[oldNeighbor] else -1
                var invalid = nodeNeighbor < 0 || nodeNeighbor == nodeRow
                if (!invalid) {
                    for (seenOffset in 0 until seenCount) {
                        if (seenNeighbors[seenOffset] == nodeNeighbor) {
                            invalid = true
                            break
                        }
                    }
                }
                if (invalid) {
                    invalidatedSlotCount++
                    if (!affected[nodeRow]) {
                        affected[nodeRow] = true
                        affectedRowCount++
                    }
                } else {
                    seenNeighbors[seenCount++] = nodeNeighbor
                }
            }
        }

        val repairedNeighbors = IntArray(orderedNodeTrackIds.size * k) { -1 }
        for (nodeRow in orderedNodeTrackIds.indices) {
            if ((nodeRow and CANCELLATION_CHECK_MASK) == 0) cancellationCheck()
            val destinationStart = nodeRow * k
            val oldRow = nodeToOld[nodeRow]
            if (!affected[nodeRow]) {
                val oldRowStart = oldRow * k
                for (slot in 0 until k) {
                    repairedNeighbors[destinationStart + slot] =
                        oldToNode[base.neighbors[oldRowStart + slot]]
                }
                continue
            }

            val trackId = orderedNodeTrackIds[nodeRow]
            val exclusions = HashSet<Long>(excludedTrackIds.size + 1).apply {
                addAll(excludedTrackIds)
                add(trackId)
            }
            val exact = exactTopK(trackId, k, exclusions)
            require(exact.size == k) {
                "Active graph row $trackId produced ${exact.size}/$k exact neighbors"
            }
            val distinctNeighborIds = HashSet<Long>(k * 2)
            exact.forEachIndexed { slot, neighborId ->
                require(neighborId != trackId && distinctNeighborIds.add(neighborId)) {
                    "Active graph repair returned a repeated or self neighbor $neighborId for $trackId"
                }
                repairedNeighbors[destinationStart + slot] =
                    requireNotNull(nodeIndexByTrackId[neighborId]) {
                        "Active graph repair returned excluded track $neighborId"
                    }
            }
        }
        cancellationCheck()

        return ActiveDomainGraphTopology(
            topology = UniformGraphSnapshot.fromRaw(
                trackIds = orderedNodeTrackIds,
                neighborIndices = repairedNeighbors,
                neighborsPerNode = k,
            ),
            evidence = ActiveDomainGraphRepairEvidence(
                nodeCount = orderedNodeTrackIds.size,
                neighborsPerNode = k,
                affectedRowCount = affectedRowCount,
                preservedRowCount = orderedNodeTrackIds.size - affectedRowCount,
                invalidatedSlotCount = invalidatedSlotCount,
            ),
        )
    }

    private fun LongArray.isStrictlyIncreasing(): Boolean {
        var previous = Long.MIN_VALUE
        for (value in this) {
            if (value <= previous) return false
            previous = value
        }
        return true
    }

    private const val CANCELLATION_CHECK_MASK = 0x3ff
}
