package com.powerampstartradio.indexing.v2

import com.powerampstartradio.indexing.Clamp3AudioInference

data class V2IndexingStorageEstimate(
    /** Exact requirement when [unresolvedAudioSpanCount] is zero; otherwise a known lower bound. */
    val requiredAdditionalBytes: Long,
    val availableBytes: Long,
    val jobDatabaseBytes: Long,
    val publicationBytes: Long,
    val durableArtifactBytes: Long,
    val peakPcmBytes: Long,
    val unresolvedAudioSpanCount: Int,
) {
    val hasCapacity: Boolean get() = availableBytes >= requiredAdditionalBytes
    val isExact: Boolean get() = unresolvedAudioSpanCount == 0
}

data class V2PlannedStorageSpan(
    val mertWindows: Int?,
    val exactSampleCount24k: Long?,
    val sourceSampleCount: Long?,
) {
    init {
        require(
            listOf(mertWindows, exactSampleCount24k, sourceSampleCount)
                .map { it == null }
                .distinct()
                .size == 1,
        ) {
            "storage work must be wholly exact or wholly unresolved"
        }
        if (mertWindows != null) require(mertWindows > 0)
        if (exactSampleCount24k != null) require(exactSampleCount24k > 0L)
        if (sourceSampleCount != null) require(sourceSampleCount > 0L)
    }

    val isExact: Boolean get() = mertWindows != null
}

data class V2GenerationMutationStorageEstimate(
    val requiredAdditionalBytes: Long,
    val availableBytes: Long,
) {
    val hasCapacity: Boolean get() = availableBytes >= requiredAdditionalBytes
}

class V2StorageCapacityException(message: String) : IllegalStateException(message)

/** Conservative peak-space accounting before a durable job is admitted. */
object V2IndexingStoragePolicy {
    private const val DATABASE_GROWTH_PER_TRACK = 8L * 1024L
    private const val SQLITE_AND_ATOMIC_HEADROOM = 64L * 1024L * 1024L
    private const val SAFETY_NUMERATOR = 6L
    private const val SAFETY_DENOMINATOR = 5L
    private const val FLOAT_BYTES = 4L

    fun estimate(
        active: V2ResolvedActiveIndexGeneration,
        spans: List<V2PlannedStorageSpan>,
        rebuildGraph: Boolean,
        availableBytes: Long,
    ): V2IndexingStorageEstimate {
        require(spans.isNotEmpty()) { "storage estimate requires selected tracks" }
        require(availableBytes >= 0L) { "available storage cannot be negative" }
        val newTracks = spans.size.toLong()
        val databaseGrowth = multiply(newTracks, DATABASE_GROWTH_PER_TRACK)
        val nextDatabase = add(active.manifest.databaseByteLength, databaseGrowth)
        val jobDatabase = nextDatabase
        val nextEmbedding = add(
            active.manifest.embeddingByteLength,
            multiply(newTracks, Long.SIZE_BYTES + V2_CLAMP3_BLOB_BYTES.toLong()),
        )
        val nextGraph = if (rebuildGraph) {
            val nodes = add(active.manifest.trackCount.toLong(), newTracks)
            add(8L, multiply(nodes, Long.SIZE_BYTES + 5L * 2L * Int.SIZE_BYTES))
        } else {
            0L
        }
        val durableArtifacts = spans.fold(0L) { total, span ->
            val mert = span.mertWindows?.let { windows ->
                multiply(
                    windows.toLong(),
                    Clamp3AudioInference.WINDOW_BYTES.toLong(),
                )
            } ?: 0L
            add(total, add(mert, V2_CLAMP3_BLOB_BYTES.toLong()))
        }
        val pcmFootprints = spans.map { span ->
            val targetPcm = span.exactSampleCount24k
                ?.let { sampleCount -> multiply(sampleCount, FLOAT_BYTES) }
                ?: 0L
            val nativeScratch = span.sourceSampleCount
                ?.let { sampleCount -> multiply(sampleCount, FLOAT_BYTES) }
                ?: 0L
            targetPcm to nativeScratch
        }
        val peakPcm = pcmFootprints.indices.maxOfOrNull { index ->
            val (targetPcm, nativeScratch) = pcmFootprints[index]
            val previousTargetPcm = if (index > 0) pcmFootprints[index - 1].first else 0L
            // TrackPcmCache retains native scratch while publishing target PCM. One-ahead
            // execution can additionally retain the previous track's target PCM for MERT.
            add(add(targetPcm, nativeScratch), previousTargetPcm)
        } ?: 0L
        val unresolvedAudioSpanCount = spans.count { !it.isExact }
        val publication = add(add(nextDatabase, nextEmbedding), nextGraph)
        val raw = add(
            add(jobDatabase, publication),
            add(add(durableArtifacts, peakPcm), SQLITE_AND_ATOMIC_HEADROOM),
        )
        val required = divideRoundUp(multiply(raw, SAFETY_NUMERATOR), SAFETY_DENOMINATOR)
        return V2IndexingStorageEstimate(
            requiredAdditionalBytes = required,
            availableBytes = availableBytes,
            jobDatabaseBytes = jobDatabase,
            publicationBytes = publication,
            durableArtifactBytes = durableArtifacts,
            peakPcmBytes = peakPcm,
            unresolvedAudioSpanCount = unresolvedAudioSpanCount,
        )
    }

    fun requireCapacity(estimate: V2IndexingStorageEstimate) {
        if (!estimate.hasCapacity) {
            throw V2IndexingPreflightException(
                code = V2IndexingPreflightFailureCode.INSUFFICIENT_STORAGE,
                message = "Indexing needs about ${formatBytes(estimate.requiredAdditionalBytes)} " +
                    "of free space to finish safely; ${formatBytes(estimate.availableBytes)} is available.",
            )
        }
    }

    private fun add(left: Long, right: Long): Long = try {
        Math.addExact(left, right)
    } catch (_: ArithmeticException) {
        Long.MAX_VALUE
    }

    private fun multiply(left: Long, right: Long): Long = try {
        Math.multiplyExact(left, right)
    } catch (_: ArithmeticException) {
        Long.MAX_VALUE
    }

    private fun divideRoundUp(value: Long, divisor: Long): Long =
        if (value == Long.MAX_VALUE) value else add(value, divisor - 1L) / divisor

    private fun formatBytes(bytes: Long): String {
        val gib = bytes.toDouble() / (1024.0 * 1024.0 * 1024.0)
        return if (gib >= 1.0) "%.1f GiB".format(gib) else "%.0f MiB".format(gib * 1024.0)
    }
}

/** Peak-space admission for bootstrap import and immutable library maintenance. */
object V2GenerationMutationStoragePolicy {
    private const val DATABASE_GROWTH_PER_TRACK = 8L * 1024L
    private const val SQLITE_AND_ATOMIC_HEADROOM = 64L * 1024L * 1024L
    private const val SAFETY_NUMERATOR = 6L
    private const val SAFETY_DENOMINATOR = 5L

    /**
     * Before an import is copied, its row count is unknown. Four source lengths cover the private
     * source copy, published SQLite copy, PEMB extraction, and any embedded graph with headroom.
     */
    fun estimateBootstrapAdmission(
        sourceLength: Long,
        availableBytes: Long,
    ): V2GenerationMutationStorageEstimate {
        require(sourceLength > 0L) { "import source length must be positive" }
        return estimate(add(multiply(sourceLength, 4L), SQLITE_AND_ATOMIC_HEADROOM), availableBytes)
    }

    /** Remaining peak allocation after the source DB and optional graph are already staged. */
    fun estimateBootstrapPublication(
        databaseBytes: Long,
        trackCount: Int,
        embeddingDimension: Int,
        graphBytes: Long,
        availableBytes: Long,
    ): V2GenerationMutationStorageEstimate {
        require(databaseBytes > 0L)
        require(trackCount > 0)
        require(embeddingDimension > 0)
        require(graphBytes >= 0L)
        val embeddingBytes = add(
            16L,
            multiply(
                trackCount.toLong(),
                add(Long.SIZE_BYTES.toLong(), multiply(embeddingDimension.toLong(), Float.SIZE_BYTES.toLong())),
            ),
        )
        return estimate(
            add(add(add(databaseBytes, embeddingBytes), graphBytes), SQLITE_AND_ATOMIC_HEADROOM),
            availableBytes,
        )
    }

    /**
     * Maintenance keeps its retained DB while publication snapshots it again. Graph repair also
     * retains one repaired graph while the publisher copies it into the new generation.
     */
    fun estimateMaintenance(
        active: V2ResolvedActiveIndexGeneration,
        availableBytes: Long,
    ): V2GenerationMutationStorageEstimate {
        val databaseBytes = active.manifest.databaseByteLength
        val embeddingBytes = active.manifest.embeddingByteLength
        val graphBytes = active.manifest.graph?.byteLength ?: 0L
        val peak = add(
            add(multiply(databaseBytes, 2L), embeddingBytes),
            add(multiply(graphBytes, 2L), SQLITE_AND_ATOMIC_HEADROOM),
        )
        return estimate(peak, availableBytes)
    }

    /** Remaining peak after the selected bundle is staged; prepared DB and graph ownership moves. */
    fun estimateServerMerge(
        active: V2ResolvedActiveIndexGeneration,
        addedTrackCount: Int,
        availableBytes: Long,
    ): V2GenerationMutationStorageEstimate {
        require(addedTrackCount > 0) { "server merge must add at least one track" }
        val addedTracks = addedTrackCount.toLong()
        val databaseBytes = add(
            active.manifest.databaseByteLength,
            multiply(addedTracks, DATABASE_GROWTH_PER_TRACK),
        )
        val embeddingBytes = add(
            active.manifest.embeddingByteLength,
            multiply(addedTracks, Long.SIZE_BYTES + V2_CLAMP3_BLOB_BYTES.toLong()),
        )
        val graph = requireNotNull(active.manifest.graph) {
            "server merge requires an active similarity graph"
        }
        val graphBytes = add(
            graph.byteLength,
            multiply(
                addedTracks,
                Long.SIZE_BYTES + graph.neighborsPerNode.toLong() * 2L * Int.SIZE_BYTES,
            ),
        )
        val peak = add(
            add(databaseBytes, embeddingBytes),
            add(graphBytes, SQLITE_AND_ATOMIC_HEADROOM),
        )
        return estimate(peak, availableBytes)
    }

    fun requireCapacity(
        estimate: V2GenerationMutationStorageEstimate,
        operation: String,
    ) {
        if (!estimate.hasCapacity) {
            throw V2StorageCapacityException(
                "$operation needs about ${formatBytes(estimate.requiredAdditionalBytes)} of free " +
                    "space to publish safely; ${formatBytes(estimate.availableBytes)} is available.",
            )
        }
    }

    private fun estimate(
        rawRequiredBytes: Long,
        availableBytes: Long,
    ): V2GenerationMutationStorageEstimate {
        require(availableBytes >= 0L)
        val required = divideRoundUp(
            multiply(rawRequiredBytes, SAFETY_NUMERATOR),
            SAFETY_DENOMINATOR,
        )
        return V2GenerationMutationStorageEstimate(required, availableBytes)
    }

    private fun add(left: Long, right: Long): Long = try {
        Math.addExact(left, right)
    } catch (_: ArithmeticException) {
        Long.MAX_VALUE
    }

    private fun multiply(left: Long, right: Long): Long = try {
        Math.multiplyExact(left, right)
    } catch (_: ArithmeticException) {
        Long.MAX_VALUE
    }

    private fun divideRoundUp(value: Long, divisor: Long): Long =
        if (value == Long.MAX_VALUE) value else add(value, divisor - 1L) / divisor

    private fun formatBytes(bytes: Long): String {
        val gib = bytes.toDouble() / (1024.0 * 1024.0 * 1024.0)
        return if (gib >= 1.0) "%.1f GiB".format(gib) else "%.0f MiB".format(gib * 1024.0)
    }
}
