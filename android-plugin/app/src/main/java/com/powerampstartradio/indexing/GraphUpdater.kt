package com.powerampstartradio.indexing

import android.util.Log
import com.powerampstartradio.data.EmbeddingDatabase
import com.powerampstartradio.data.EmbeddingIndex
import com.powerampstartradio.indexing.v2.V2EmbeddingGenerationFile
import com.powerampstartradio.indexing.v2.V2FileSha256
import com.powerampstartradio.indexing.v2.V2GraphGenerationFile
import com.powerampstartradio.indexing.v2.V2IndexGenerationGraphBinding
import com.powerampstartradio.indexing.v2.V2OrderedEmbeddingBinding
import com.powerampstartradio.indexing.v2.V2OrderedEmbeddingConsumer
import com.powerampstartradio.indexing.v2.V2OrderedEmbeddingSource
import com.powerampstartradio.indexing.v2.V2ResolvedActiveIndexGeneration
import java.io.File
import java.io.FileOutputStream
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.charset.StandardCharsets
import java.security.MessageDigest
import java.util.UUID

enum class V2GraphUpdateStrategy {
    REUSE,
    INCREMENTAL,
    FULL_REBUILD,
}

enum class V2GraphUpdaterStage {
    EMBEDDING_ROWS,
    SIMILARITY_DOT_PRODUCTS,
    GRAPH_BINARY_BYTES,
}

/** Exact work contract for one graph build against a fixed embedding row set. */
data class V2GraphWorkPlan(
    val strategy: V2GraphUpdateStrategy,
    val targetNodes: Int,
    val baseGraphNodes: Int,
    val newNodes: Int,
    val neighborsPerNode: Int,
    val embeddingDimension: Int,
    val embeddingRows: Long,
    val similarityDotProducts: Long,
    /** Bytes read from the manifest-bound exact base graph, when one is eligible. */
    val graphBinaryInputBytes: Long,
    /** Bytes durably written to SQLite and graph.bin. */
    val graphBinaryOutputBytes: Long,
    /** Base graph nodes absent from the target generation. */
    val removedBaseNodes: Int = 0,
    /** Retained base rows whose prior top-K contained at least one removed node. */
    val rescannedBaseNodes: Int = 0,
    /** Base graph nodes retained unchanged in the target embedding row set. */
    val retainedBaseNodes: Int = baseGraphNodes - removedBaseNodes,
) {
    val graphBinaryBytes: Long
        get() = Math.addExact(graphBinaryInputBytes, graphBinaryOutputBytes)
}

data class V2GraphUpdaterProgress(
    val plan: V2GraphWorkPlan,
    val stage: V2GraphUpdaterStage,
    val completedUnits: Long,
    val totalUnits: Long,
    val detail: String,
)

internal data class V2PreparedGraphUpdate(
    val plan: V2GraphWorkPlan,
    val databaseFile: File,
    val embeddingFile: File,
    val embeddingBinding: V2OrderedEmbeddingBinding,
    val graphFile: File,
    val graphBinding: V2IndexGenerationGraphBinding,
)

interface V2GraphUpdaterControl {
    fun onProgress(progress: V2GraphUpdaterProgress)

    /** Must be cheap. It is called inside long loops so pause/cancel latency stays bounded. */
    fun onControlPoint(stage: V2GraphUpdaterStage, completedUnits: Long)
}

/** Manifest-validated exact graph/embedding pair eligible to seed an exact delta update. */
class V2ExactGraphIncrementalBase internal constructor(
    val generationId: String,
    val embeddingFile: File,
    val graphFile: File,
    val trackCount: Int,
    val embeddingDimension: Int,
    val embeddingByteLength: Long,
    val graphByteLength: Long,
    val embeddingSha256: String,
    val graphSha256: String,
) {
    init {
        require(SHA256.matches(embeddingSha256) && SHA256.matches(graphSha256)) {
            "exact base asset hashes must be lowercase SHA-256"
        }
    }

    internal fun requireManifestBoundAssets() {
        requireManifestBoundLengths()
        require(V2FileSha256.digest(embeddingFile) == embeddingSha256) {
            "exact base embedding changed after manifest validation"
        }
        require(V2FileSha256.digest(graphFile) == graphSha256) {
            "exact base graph changed after manifest validation"
        }
    }

    internal fun requireManifestBoundLengths() {
        require(embeddingFile.isFile &&
            embeddingFile.length() == embeddingByteLength &&
            graphFile.isFile &&
            graphFile.length() == graphByteLength
        ) { "exact base assets changed after manifest validation" }
    }

    companion object {
        private val SHA256 = Regex("^[0-9a-f]{64}$")

        fun fromActiveGeneration(
            active: V2ResolvedActiveIndexGeneration,
        ): V2ExactGraphIncrementalBase? {
            val graph = active.manifest.graph ?: return null
            val graphFile = active.graphFile ?: return null
            require(graph.nodeCount == active.manifest.trackCount &&
                graph.neighborsPerNode > 0 &&
                graph.orderedTrackSetSha256 == active.manifest.orderedTrackSetSha256
            ) { "exact base graph binding is inconsistent" }
            val database = EmbeddingDatabase.open(active.databaseFile)
            val exactProof = try {
                database.getSmallBinaryData(V2GraphExactProof.DATABASE_KEY)
            } finally {
                database.close()
            }
            if (!V2GraphExactProof.matches(
                    bytes = exactProof,
                    graphSha256 = graph.sha256,
                    embeddingSha256 = active.manifest.embeddingSha256,
                )
            ) {
                return null
            }
            return V2ExactGraphIncrementalBase(
                generationId = active.manifest.generationId,
                embeddingFile = active.embeddingFile,
                graphFile = graphFile,
                trackCount = active.manifest.trackCount,
                embeddingDimension = active.manifest.embeddingDimension,
                embeddingByteLength = active.manifest.embeddingByteLength,
                graphByteLength = graph.byteLength,
                embeddingSha256 = active.manifest.embeddingSha256,
                graphSha256 = graph.sha256,
            )
        }
    }
}

/** Durable induction base: native full builds and exact descendants alone write this receipt. */
internal object V2GraphExactProof {
    const val DATABASE_KEY = "knn_graph_android_exact_v1"
    private const val MAGIC = 0x58474B4E // NKGX
    private const val VERSION = 1
    private const val SHA256_ASCII_BYTES = 64
    private const val BYTE_LENGTH = 8 + SHA256_ASCII_BYTES * 2
    private val SHA256 = Regex("^[0-9a-f]{64}$")

    fun create(graphFile: File, embeddingFile: File): ByteArray = encode(
        graphSha256 = V2FileSha256.digest(graphFile),
        embeddingSha256 = V2FileSha256.digest(embeddingFile),
    )

    /** Encodes hashes already produced by exact generation-file validation without rereading them. */
    fun createBoundHashes(graphSha256: String, embeddingSha256: String): ByteArray = encode(
        graphSha256 = graphSha256,
        embeddingSha256 = embeddingSha256,
    )

    fun matchesFiles(
        database: EmbeddingDatabase,
        graphFile: File,
        embeddingFile: File,
    ): Boolean = graphFile.isFile && embeddingFile.isFile && matches(
        bytes = database.getSmallBinaryData(DATABASE_KEY),
        graphSha256 = V2FileSha256.digest(graphFile),
        embeddingSha256 = V2FileSha256.digest(embeddingFile),
    )

    fun matches(bytes: ByteArray?, graphSha256: String, embeddingSha256: String): Boolean {
        if (bytes == null || bytes.size != BYTE_LENGTH ||
            !SHA256.matches(graphSha256) || !SHA256.matches(embeddingSha256)
        ) {
            return false
        }
        return runCatching {
            val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
            require(buffer.int == MAGIC && buffer.int == VERSION)
            val graph = ByteArray(SHA256_ASCII_BYTES).also(buffer::get)
            val embedding = ByteArray(SHA256_ASCII_BYTES).also(buffer::get)
            String(graph, StandardCharsets.US_ASCII) == graphSha256 &&
                String(embedding, StandardCharsets.US_ASCII) == embeddingSha256
        }.getOrDefault(false)
    }

    private fun encode(graphSha256: String, embeddingSha256: String): ByteArray {
        require(SHA256.matches(graphSha256) && SHA256.matches(embeddingSha256))
        return ByteBuffer.allocate(BYTE_LENGTH).order(ByteOrder.LITTLE_ENDIAN)
            .putInt(MAGIC)
            .putInt(VERSION)
            .put(graphSha256.toByteArray(StandardCharsets.US_ASCII))
            .put(embeddingSha256.toByteArray(StandardCharsets.US_ASCII))
            .array()
    }
}

/** Pure count math used by both the service's up-front ETA and the runtime builder. */
object V2GraphWorkPlanner {
    fun plan(
        targetNodes: Int,
        embeddingDimension: Int,
        neighborsPerNode: Int = 5,
        storeGraphInDatabase: Boolean = true,
        existingGraphNodes: Int? = null,
        existingGraphNeighborsPerNode: Int? = null,
        existingGraphByteLength: Long? = null,
        validOldNeighborDotProducts: Long? = null,
        removedBaseNodes: Int = 0,
        rescannedBaseNodes: Int = 0,
    ): V2GraphWorkPlan {
        require(targetNodes > 0) { "graph target must contain at least one node" }
        require(embeddingDimension > 0) { "embedding dimension must be positive" }
        require(neighborsPerNode > 0) { "neighborsPerNode must be positive" }
        val oldN = existingGraphNodes?.takeIf { it > 0 } ?: 0
        require(removedBaseNodes in 0..oldN) { "removed base-node count is invalid" }
        val retainedN = oldN - removedBaseNodes
        require(rescannedBaseNodes in 0..retainedN) { "rescanned base-node count is invalid" }
        val compatibleOld = oldN > 0 &&
            existingGraphNeighborsPerNode == neighborsPerNode &&
            retainedN <= targetNodes
        val strategy = when {
            compatibleOld && removedBaseNodes == 0 && oldN == targetNodes ->
                V2GraphUpdateStrategy.REUSE
            compatibleOld -> V2GraphUpdateStrategy.INCREMENTAL
            else -> V2GraphUpdateStrategy.FULL_REBUILD
        }
        val newN = when (strategy) {
            V2GraphUpdateStrategy.INCREMENTAL -> targetNodes - retainedN
            V2GraphUpdateStrategy.REUSE -> 0
            V2GraphUpdateStrategy.FULL_REBUILD -> targetNodes
        }
        val dots = when (strategy) {
            V2GraphUpdateStrategy.REUSE -> 0L
            V2GraphUpdateStrategy.FULL_REBUILD ->
                multiply(targetNodes, targetNodes - 1)
            V2GraphUpdateStrategy.INCREMENTAL -> {
                val oldNeighborDots = validOldNeighborDotProducts
                    ?: multiply(retainedN - rescannedBaseNodes, neighborsPerNode)
                Math.addExact(Math.addExact(
                    multiply(newN, targetNodes),
                    multiply(rescannedBaseNodes, targetNodes - 1),
                ), oldNeighborDots)
            }
        }
        val inputBytes = existingGraphByteLength
            ?.takeIf { it > 0L }
            ?: if (oldN > 0) graphFileBytes(oldN, existingGraphNeighborsPerNode ?: neighborsPerNode)
            else 0L
        val outputBytes = if (strategy == V2GraphUpdateStrategy.REUSE) {
            0L
        } else {
            Math.multiplyExact(
                if (storeGraphInDatabase) 2L else 1L,
                graphFileBytes(targetNodes, neighborsPerNode),
            )
        }
        return V2GraphWorkPlan(
            strategy = strategy,
            targetNodes = targetNodes,
            baseGraphNodes = if (compatibleOld) oldN else 0,
            newNodes = newN,
            neighborsPerNode = neighborsPerNode,
            embeddingDimension = embeddingDimension,
            embeddingRows = targetNodes.toLong(),
            similarityDotProducts = dots,
            graphBinaryInputBytes = inputBytes,
            graphBinaryOutputBytes = outputBytes,
            removedBaseNodes = if (compatibleOld) removedBaseNodes else 0,
            rescannedBaseNodes = if (compatibleOld) rescannedBaseNodes else 0,
            retainedBaseNodes = if (compatibleOld) retainedN else 0,
        )
    }

    fun graphFileBytes(nodes: Int, neighborsPerNode: Int): Long {
        require(nodes >= 0 && neighborsPerNode > 0)
        return Math.addExact(
            8L,
            Math.addExact(
                Math.multiplyExact(nodes.toLong(), 8L),
                Math.multiplyExact(Math.multiplyExact(nodes.toLong(), neighborsPerNode.toLong()), 8L),
            ),
        )
    }

    private fun multiply(left: Int, right: Int): Long =
        Math.multiplyExact(left.toLong(), right.toLong())
}

/**
 * Updates the exact K-nearest-neighbor graph after the target track set is committed.
 *
 * Incremental math is identical to a full rebuild. New rows scan every target embedding. Retained
 * rows whose old top-K lost a node are rescanned completely; other retained rows compare their
 * exact old top-K with every new node. Checkpoints contain raw pre-normalization scores, so resuming
 * cannot change ordering or output bytes.
 */
class GraphUpdater(
    private val db: EmbeddingDatabase,
    private val filesDir: File,
    private val knnK: Int = 5,
    private val checkpointsEnabled: Boolean = true,
    private val storeGraphInDatabase: Boolean = true,
) {
    companion object {
        private const val TAG = "GraphUpdater"
        private const val CHECKPOINT_MAGIC = 0x47524350 // GRCP
        private const val CHECKPOINT_VERSION = 2
        private const val PHASE_FULL_ROWS = 1
        private const val PHASE_INCREMENTAL_NEW_ROWS = 2
        private const val PHASE_INCREMENTAL_RESCANS = 3
        private const val CHECKPOINT_TARGET_DOTS = 50_000_000L
        private const val PROGRESS_TARGET_DOTS = 10_000_000L
        private const val CONTROL_ROW_INTERVAL = 1_024
        private const val DIGEST_BYTES = 32
        private val NO_CONTROL = object : V2GraphUpdaterControl {
            override fun onProgress(progress: V2GraphUpdaterProgress) = Unit
            override fun onControlPoint(stage: V2GraphUpdaterStage, completedUnits: Long) = Unit
        }
    }

    fun rebuildIndices(
        control: V2GraphUpdaterControl = NO_CONTROL,
        exactBase: V2ExactGraphIncrementalBase? = null,
    ): V2GraphWorkPlan = rebuildIndicesInternal(
        control = control,
        exactBase = exactBase,
        prepareForPublication = false,
        retainedEmbeddingsAreByteExactBase = false,
    ).plan

    internal fun rebuildAfterByteExactAppendForPublication(
        control: V2GraphUpdaterControl = NO_CONTROL,
        exactBase: V2ExactGraphIncrementalBase,
    ): V2PreparedGraphUpdate {
        val result = rebuildIndicesInternal(
            control = control,
            exactBase = exactBase,
            prepareForPublication = true,
            retainedEmbeddingsAreByteExactBase = true,
        )
        return V2PreparedGraphUpdate(
            plan = result.plan,
            databaseFile = db.databaseFile,
            embeddingFile = result.embeddingFile,
            embeddingBinding = checkNotNull(result.embeddingBinding),
            graphFile = result.graphFile,
            graphBinding = checkNotNull(result.graphBinding),
        )
    }

    private fun rebuildIndicesInternal(
        control: V2GraphUpdaterControl,
        exactBase: V2ExactGraphIncrementalBase?,
        prepareForPublication: Boolean,
        retainedEmbeddingsAreByteExactBase: Boolean,
    ): RebuildResult {
        require(!retainedEmbeddingsAreByteExactBase ||
            (prepareForPublication && exactBase != null)
        ) { "byte-exact append trust requires a publication build and exact base" }
        val t0 = System.nanoTime()
        val targetNodes = db.getEmbeddingCount()
        val dimension = requireNotNull(db.getEmbeddingDim()) { "embedding dimension is unavailable" }
        require(targetNodes > 0) { "cannot build a graph for an empty embedding database" }

        val embFile = File(filesDir, "clamp3.emb")
        val provisional = V2GraphWorkPlanner.plan(
            targetNodes = targetNodes,
            embeddingDimension = dimension,
            neighborsPerNode = knnK,
            storeGraphInDatabase = storeGraphInDatabase,
        )
        emit(control, provisional, V2GraphUpdaterStage.EMBEDDING_ROWS, 0L, targetNodes.toLong(),
            "Exporting the immutable embedding row set")
        val embeddingBinding = if (prepareForPublication) {
            V2EmbeddingGenerationFile.write(
                source = DatabaseEmbeddingSource(db, targetNodes, dimension),
                target = embFile,
                onRowProgress = { current, total ->
                    emit(
                        control,
                        provisional,
                        V2GraphUpdaterStage.EMBEDDING_ROWS,
                        current.toLong(),
                        total.toLong(),
                        "Exported $current/$total embedding rows",
                    )
                    control.onControlPoint(V2GraphUpdaterStage.EMBEDDING_ROWS, current.toLong())
                },
            )
        } else {
            EmbeddingIndex.extractFromDatabase(db, embFile) { current, total ->
                emit(
                    control,
                    provisional,
                    V2GraphUpdaterStage.EMBEDDING_ROWS,
                    current.toLong(),
                    total.toLong(),
                    "Exported $current/$total embedding rows",
                )
                control.onControlPoint(V2GraphUpdaterStage.EMBEDDING_ROWS, current.toLong())
            }
            null
        }
        val index = EmbeddingIndex.mmap(embFile)
        require(index.numTracks == targetNodes && index.dim == dimension) {
            "exported embedding index disagrees with its database plan"
        }

        val currentTrackIds = HashSet<Long>(targetNodes * 2)
        for (position in 0 until targetNodes) currentTrackIds += index.getTrackId(position)
        val graphFile = File(filesDir, "graph.bin")
        val oldGraph = exactBase?.let { base ->
            if (retainedEmbeddingsAreByteExactBase) {
                prepareExactBase(
                    base = base,
                    targetIndex = index,
                    localGraphFile = graphFile,
                    verifyRetainedEmbeddings = false,
                )
            } else {
                runCatching {
                    prepareExactBase(
                        base = base,
                        targetIndex = index,
                        localGraphFile = graphFile,
                        verifyRetainedEmbeddings = true,
                    )
                }
                    .onFailure { error ->
                        Log.w(TAG, "Exact base graph rejected; using full rebuild: ${error.message}")
                    }
                    .getOrNull()
            }
        }
        val hasGraph = oldGraph != null
        val structurallyUsableOld = oldGraph?.takeIf { old ->
            old.neighborsPerNode == knnK &&
                old.trackIds.toSet().size == old.trackIds.size &&
                old.validForExactUpdate &&
                targetNodes > knnK
        }
        val exactReuse = structurallyUsableOld?.takeIf { old ->
            old.trackIds.size == targetNodes &&
                old.trackIds.indices.all { position ->
                    old.trackIds[position] == index.getTrackId(position)
                }
        }
        val mutation = structurallyUsableOld
            ?.takeUnless { it === exactReuse }
            ?.let { describeMutation(index, it, currentTrackIds) }
            ?.takeIf {
                it.retainedBaseNodes > 0 &&
                    (it.newTrackIndices.isNotEmpty() || it.removedBaseNodes > 0)
            }
        val inputBytes = if (hasGraph) {
            exactBase?.graphFile?.length() ?: graphFile.takeIf(File::isFile)?.length() ?: 0L
        } else {
            0L
        }
        val plan = when {
            exactReuse != null -> V2GraphWorkPlanner.plan(
                targetNodes = targetNodes,
                embeddingDimension = dimension,
                neighborsPerNode = knnK,
                storeGraphInDatabase = storeGraphInDatabase,
                existingGraphNodes = targetNodes,
                existingGraphNeighborsPerNode = knnK,
                existingGraphByteLength = inputBytes,
            )
            mutation != null -> V2GraphWorkPlanner.plan(
                targetNodes = targetNodes,
                embeddingDimension = dimension,
                neighborsPerNode = knnK,
                storeGraphInDatabase = storeGraphInDatabase,
                existingGraphNodes = mutation.oldGraph.trackIds.size,
                existingGraphNeighborsPerNode = mutation.oldGraph.neighborsPerNode,
                existingGraphByteLength = inputBytes,
                validOldNeighborDotProducts = Math.multiplyExact(
                    mutation.unaffectedBasePositions.size.toLong(),
                    knnK.toLong(),
                ),
                removedBaseNodes = mutation.removedBaseNodes,
                rescannedBaseNodes = mutation.affectedBasePositions.size,
            )
            else -> V2GraphWorkPlanner.plan(
                targetNodes = targetNodes,
                embeddingDimension = dimension,
                neighborsPerNode = knnK,
                storeGraphInDatabase = storeGraphInDatabase,
                existingGraphByteLength = inputBytes.takeIf { it > 0L },
            )
        }

        if (inputBytes > 0L) {
            emit(
                control,
                plan,
                V2GraphUpdaterStage.GRAPH_BINARY_BYTES,
                inputBytes,
                plan.graphBinaryBytes,
                "Validated ${structurallyUsableOld?.trackIds?.size ?: 0} prior graph rows",
            )
            control.onControlPoint(V2GraphUpdaterStage.GRAPH_BINARY_BYTES, inputBytes)
        }

        when (plan.strategy) {
            V2GraphUpdateStrategy.REUSE -> {
                if (checkpointsEnabled) checkpointFile().delete()
                Log.i(TAG, "Reused exact graph: $targetNodes nodes, K=$knnK")
            }
            V2GraphUpdateStrategy.INCREMENTAL -> incrementalUpdate(
                index = index,
                graphFile = graphFile,
                mutation = checkNotNull(mutation),
                plan = plan,
                control = control,
                embeddingSha256 = embeddingBinding?.fileSha256,
                oldGraphSha256 = exactBase?.graphSha256,
            )
            V2GraphUpdateStrategy.FULL_REBUILD -> buildKnnGraph(
                index = index,
                graphFile = graphFile,
                plan = plan,
                control = control,
                embeddingSha256 = embeddingBinding?.fileSha256,
            )
        }
        val graphBinding = if (prepareForPublication) {
            V2GraphGenerationFile.inspect(graphFile)
        } else {
            null
        }
        db.setBinaryData(
            V2GraphExactProof.DATABASE_KEY,
            if (embeddingBinding != null && graphBinding != null) {
                V2GraphExactProof.createBoundHashes(
                    graphSha256 = graphBinding.sha256,
                    embeddingSha256 = embeddingBinding.fileSha256,
                )
            } else {
                V2GraphExactProof.create(graphFile = graphFile, embeddingFile = embFile)
            },
        )

        val totalMs = (System.nanoTime() - t0) / 1_000_000
        Log.i(TAG, "TIMING: graph tail ${plan.strategy} completed in ${totalMs}ms")
        return RebuildResult(
            plan = plan,
            embeddingFile = embFile,
            embeddingBinding = embeddingBinding,
            graphFile = graphFile,
            graphBinding = graphBinding,
        )
    }

    private data class RebuildResult(
        val plan: V2GraphWorkPlan,
        val embeddingFile: File,
        val embeddingBinding: V2OrderedEmbeddingBinding?,
        val graphFile: File,
        val graphBinding: V2IndexGenerationGraphBinding?,
    )

    private class DatabaseEmbeddingSource(
        private val database: EmbeddingDatabase,
        override val trackCount: Int,
        override val dimension: Int,
    ) : V2OrderedEmbeddingSource {
        override fun forEachOrdered(consumer: V2OrderedEmbeddingConsumer) {
            database.forEachEmbeddingRaw { trackId, embedding ->
                consumer.accept(trackId, embedding)
            }
        }
    }

    private fun prepareExactBase(
        base: V2ExactGraphIncrementalBase,
        targetIndex: EmbeddingIndex,
        localGraphFile: File,
        verifyRetainedEmbeddings: Boolean,
    ): OldGraph {
        if (verifyRetainedEmbeddings) {
            base.requireManifestBoundAssets()
        } else {
            base.requireManifestBoundLengths()
        }
        val baseIndex = EmbeddingIndex.mmap(base.embeddingFile, preload = false)
        require(baseIndex.numTracks == base.trackCount &&
            baseIndex.dim == base.embeddingDimension &&
            targetIndex.dim == base.embeddingDimension
        ) { "exact base embedding shape changed" }

        val graphToParse = if (verifyRetainedEmbeddings) {
            val temporary = File(
                localGraphFile.parentFile,
                ".${localGraphFile.name}.base-${UUID.randomUUID()}",
            )
            try {
                val graphDigest = MessageDigest.getInstance("SHA-256")
                base.graphFile.inputStream().use { input ->
                    FileOutputStream(temporary).use { output ->
                        val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
                        while (true) {
                            val read = input.read(buffer)
                            if (read < 0) break
                            if (read == 0) continue
                            output.write(buffer, 0, read)
                            graphDigest.update(buffer, 0, read)
                        }
                        output.fd.sync()
                    }
                }
                require(graphDigest.digest().joinToString("") { byte -> "%02x".format(byte) } ==
                    base.graphSha256
                ) {
                    "exact base graph copy changed"
                }
                EmbeddingIndex.replaceAtomically(temporary, localGraphFile)
            } finally {
                temporary.delete()
            }
            localGraphFile
        } else {
            base.graphFile
        }

        val graph = requireNotNull(parseOldGraph(
            graphFile = graphToParse,
            expectedSha256 = if (verifyRetainedEmbeddings) null else base.graphSha256,
        )) {
            "manifest-bound base graph is unreadable"
        }
        require(graph.trackIds.size == baseIndex.numTracks &&
            graph.neighborsPerNode == knnK &&
            graph.validForExactUpdate &&
            graph.trackIds.indices.all { position ->
                graph.trackIds[position] == baseIndex.getTrackId(position)
            }
        ) { "manifest-bound base graph and embedding rows disagree" }

        val targetIndexById = HashMap<Long, Int>(targetIndex.numTracks * 2)
        for (position in 0 until targetIndex.numTracks) {
            targetIndexById[targetIndex.getTrackId(position)] = position
        }
        if (verifyRetainedEmbeddings) {
            graph.trackIds.indices.forEach { basePosition ->
                val targetPosition = targetIndexById[graph.trackIds[basePosition]]
                    ?: return@forEach
                require(baseIndex.hasBitIdenticalEmbedding(
                    index = basePosition,
                    other = targetIndex,
                    otherIndex = targetPosition,
                )) { "retained embedding ${graph.trackIds[basePosition]} changed from the exact base" }
            }
        }
        return graph
    }

    private fun describeMutation(
        index: EmbeddingIndex,
        oldGraph: OldGraph,
        targetTrackIds: Set<Long>,
    ): GraphMutation {
        val oldTrackIds = oldGraph.trackIds.toHashSet()
        val unaffected = ArrayList<Int>(oldGraph.trackIds.size)
        val affected = ArrayList<Int>()
        var retained = 0
        oldGraph.trackIds.indices.forEach { oldPosition ->
            if (oldGraph.trackIds[oldPosition] !in targetTrackIds) return@forEach
            retained++
            if (oldGraph.neighborTrackIds[oldPosition].any { it !in targetTrackIds }) {
                affected += oldPosition
            } else {
                unaffected += oldPosition
            }
        }
        val newTrackIndices = IntArray(index.numTracks - retained)
        var newCursor = 0
        for (targetPosition in 0 until index.numTracks) {
            if (index.getTrackId(targetPosition) !in oldTrackIds) {
                newTrackIndices[newCursor++] = targetPosition
            }
        }
        require(newCursor == newTrackIndices.size) { "graph mutation ID accounting changed" }
        return GraphMutation(
            oldGraph = oldGraph,
            newTrackIndices = newTrackIndices,
            unaffectedBasePositions = unaffected.toIntArray(),
            affectedBasePositions = affected.toIntArray(),
            removedBaseNodes = oldGraph.trackIds.size - retained,
        )
    }

    private fun incrementalUpdate(
        index: EmbeddingIndex,
        graphFile: File,
        mutation: GraphMutation,
        plan: V2GraphWorkPlan,
        control: V2GraphUpdaterControl,
        embeddingSha256: String?,
        oldGraphSha256: String?,
    ) {
        val totalN = index.numTracks
        val k = knnK
        val oldGraph = mutation.oldGraph
        val idToIdx = HashMap<Long, Int>(totalN * 2)
        val trackIdsByIndex = LongArray(totalN)
        for (i in 0 until totalN) {
            index.getTrackId(i).also { id ->
                trackIdsByIndex[i] = id
                idToIdx[id] = i
            }
        }
        val newTrackIndices = mutation.newTrackIndices
        val unaffectedBasePositions = mutation.unaffectedBasePositions
        val affectedBasePositions = mutation.affectedBasePositions
        val retainedBaseNodes = unaffectedBasePositions.size + affectedBasePositions.size
        require(newTrackIndices.size == plan.newNodes &&
            mutation.removedBaseNodes == plan.removedBaseNodes &&
            affectedBasePositions.size == plan.rescannedBaseNodes &&
            retainedBaseNodes == plan.retainedBaseNodes
        ) { "incremental graph mutation plan changed" }
        val targetIndexForBasePosition = IntArray(oldGraph.trackIds.size) { oldPosition ->
            idToIdx[oldGraph.trackIds[oldPosition]] ?: -1
        }
        val newEmbs = Array(newTrackIndices.size) { position ->
            index.getEmbedding(newTrackIndices[position])
        }
        val fingerprint = if (checkpointsEnabled) {
            checkpointFingerprint(
                plan = plan,
                embeddingFile = File(filesDir, "clamp3.emb"),
                oldGraphFile = graphFile,
                embeddingSha256 = embeddingSha256,
                oldGraphSha256 = oldGraphSha256,
            )
        } else {
            null
        }
        val restored = fingerprint?.let { readCheckpoint(it, totalN, k) }
        val neighbors = restored?.neighbors ?: Array(totalN) { IntArray(k) }
        val weights = restored?.weights ?: Array(totalN) { FloatArray(k) }
        var phase = restored?.phase ?: PHASE_INCREMENTAL_RESCANS
        var cursor = restored?.cursor ?: 0
        if (phase !in setOf(PHASE_INCREMENTAL_RESCANS, PHASE_INCREMENTAL_NEW_ROWS)) {
            phase = PHASE_INCREMENTAL_RESCANS
            cursor = 0
            clearRows(neighbors, weights)
        }

        val checkpointEvery = checkpointInterval(totalN)
        val preservedNeighborDots = Math.multiplyExact(unaffectedBasePositions.size, k)
        if (restored == null) {
            val leftRows = IntArray(preservedNeighborDots)
            val rightRows = IntArray(preservedNeighborDots)
            var pair = 0
            for (oldPosition in unaffectedBasePositions) {
                val row = targetIndexForBasePosition[oldPosition]
                require(row >= 0) { "preserved graph row disappeared from the target" }
                for (neighborId in oldGraph.neighborTrackIds[oldPosition]) {
                    leftRows[pair] = row
                    rightRows[pair] = requireNotNull(idToIdx[neighborId])
                    pair++
                }
            }
            require(pair == preservedNeighborDots)
            val oldScores = index.computePairSimilarities(leftRows, rightRows) {
                control.onControlPoint(
                    V2GraphUpdaterStage.SIMILARITY_DOT_PRODUCTS,
                    pair.toLong(),
                )
            }
            pair = 0
            for (oldPosition in unaffectedBasePositions) {
                val row = targetIndexForBasePosition[oldPosition]
                for (slot in 0 until k) {
                    neighbors[row][slot] = rightRows[pair]
                    weights[row][slot] = oldScores[pair]
                    pair++
                }
                V2GraphTopKOrdering.sortBestFirst(neighbors[row], weights[row], trackIdsByIndex, k)
                if (oldPosition % CONTROL_ROW_INTERVAL == 0) {
                    control.onControlPoint(
                        V2GraphUpdaterStage.SIMILARITY_DOT_PRODUCTS,
                        pair.toLong(),
                    )
                }
            }
            require(pair == preservedNeighborDots) {
                "preserved graph candidate count changed during incremental preparation"
            }
            emitDots(
                control,
                plan,
                preservedNeighborDots.toLong(),
                "Prepared ${unaffectedBasePositions.size} preserved graph rows",
            )
            fingerprint?.let {
                writeCheckpoint(
                    it,
                    PHASE_INCREMENTAL_RESCANS,
                    0,
                    neighbors,
                    weights,
                )
            }
        }

        if (phase == PHASE_INCREMENTAL_RESCANS) {
            cursor = cursor.coerceIn(0, affectedBasePositions.size)
            var completedDots = Math.addExact(
                preservedNeighborDots.toLong(),
                Math.multiplyExact(cursor.toLong(), (totalN - 1).toLong()),
            )
            emitDots(
                control,
                plan,
                completedDots,
                if (cursor == 0) "Repairing graph rows affected by removed tracks"
                else "Resuming graph-row repair after removed tracks",
            )
            for (position in cursor until affectedBasePositions.size) {
                val oldPosition = affectedBasePositions[position]
                val row = targetIndexForBasePosition[oldPosition]
                require(row >= 0) { "affected graph row disappeared from the target" }
                val trackId = trackIdsByIndex[row]
                val topK = index.findTopK(
                    index.getEmbedding(row),
                    k,
                    excludeIds = setOf(trackId),
                    cancellationCheck = {
                        control.onControlPoint(
                            V2GraphUpdaterStage.SIMILARITY_DOT_PRODUCTS,
                            completedDots,
                        )
                    },
                )
                writeTopK(neighbors[row], weights[row], topK, idToIdx, k)
                completedDots = Math.addExact(completedDots, (totalN - 1).toLong())
                emitDots(
                    control,
                    plan,
                    completedDots,
                    "Repaired ${position + 1}/${affectedBasePositions.size} affected graph rows",
                )
                controlAfterCompletedUnit(
                    control,
                    fingerprint,
                    PHASE_INCREMENTAL_RESCANS,
                    position + 1,
                    neighbors,
                    weights,
                    completedDots,
                )
                if (fingerprint != null && ((position + 1) % checkpointEvery == 0 ||
                    position == affectedBasePositions.lastIndex
                )) {
                    writeCheckpoint(
                        fingerprint,
                        PHASE_INCREMENTAL_RESCANS,
                        position + 1,
                        neighbors,
                        weights,
                    )
                }
            }
            fingerprint?.let {
                writeCheckpoint(
                    it,
                    PHASE_INCREMENTAL_NEW_ROWS,
                    0,
                    neighbors,
                    weights,
                )
            }
            phase = PHASE_INCREMENTAL_NEW_ROWS
            cursor = 0
        }

        if (phase == PHASE_INCREMENTAL_NEW_ROWS) {
            cursor = cursor.coerceIn(0, newTrackIndices.size)
            val fixedDots = Math.addExact(
                preservedNeighborDots.toLong(),
                Math.multiplyExact(
                    affectedBasePositions.size.toLong(),
                    (totalN - 1).toLong(),
                ),
            )
            var completedDots = Math.addExact(
                fixedDots,
                Math.multiplyExact(cursor.toLong(), totalN.toLong()),
            )
            emitDots(
                control,
                plan,
                completedDots,
                if (cursor == 0) "Comparing new tracks with the complete library"
                else "Resuming new-track graph comparisons",
            )
            val similarities = FloatArray(totalN)
            for (position in cursor until newTrackIndices.size) {
                val newIndex = newTrackIndices[position]
                index.computeAllSimilaritiesInto(
                    reference = newEmbs[position],
                    outSimilarities = similarities,
                    cancellationCheck = {
                        control.onControlPoint(
                            V2GraphUpdaterStage.SIMILARITY_DOT_PRODUCTS,
                            completedDots,
                        )
                    },
                )
                writeTopKFromSimilarities(
                    row = newIndex,
                    similarities = similarities,
                    neighborsRow = neighbors[newIndex],
                    weightsRow = weights[newIndex],
                    trackIdsByIndex = trackIdsByIndex,
                    k = k,
                )
                for (oldPosition in unaffectedBasePositions) {
                    val row = targetIndexForBasePosition[oldPosition]
                    V2GraphTopKOrdering.tryInsert(
                        neighbors[row],
                        weights[row],
                        newIndex,
                        similarities[row],
                        trackIdsByIndex,
                        k,
                    )
                    if (oldPosition % CONTROL_ROW_INTERVAL == 0) {
                        control.onControlPoint(
                            V2GraphUpdaterStage.SIMILARITY_DOT_PRODUCTS,
                            completedDots,
                        )
                    }
                }
                completedDots = Math.addExact(completedDots, totalN.toLong())
                emitDots(
                    control,
                    plan,
                    completedDots,
                    "Compared ${position + 1}/${newTrackIndices.size} new graph rows",
                )
                controlAfterCompletedUnit(
                    control,
                    fingerprint,
                    PHASE_INCREMENTAL_NEW_ROWS,
                    position + 1,
                    neighbors,
                    weights,
                    completedDots,
                )
                if (fingerprint != null &&
                    ((position + 1) % checkpointEvery == 0 ||
                        position == newTrackIndices.lastIndex)
                ) {
                    writeCheckpoint(
                        fingerprint,
                        PHASE_INCREMENTAL_NEW_ROWS,
                        position + 1,
                        neighbors,
                        weights,
                    )
                }
            }
            require(completedDots == plan.similarityDotProducts) {
                "incremental graph dot-product count disagrees with its plan"
            }
        }

        for (oldPosition in unaffectedBasePositions) {
            val row = targetIndexForBasePosition[oldPosition]
            V2GraphTopKOrdering.sortBestFirst(neighbors[row], weights[row], trackIdsByIndex, k)
            V2GraphWeightPolicy.normalizeNonnegativeInPlace(weights[row], k)
            if (oldPosition % CONTROL_ROW_INTERVAL == 0) {
                control.onControlPoint(
                    V2GraphUpdaterStage.SIMILARITY_DOT_PRODUCTS,
                    plan.similarityDotProducts,
                )
            }
        }
        publishGraph(index, graphFile, neighbors, weights, plan, control)
        if (checkpointsEnabled) checkpointFile().delete()
    }

    private fun buildKnnGraph(
        index: EmbeddingIndex,
        graphFile: File,
        plan: V2GraphWorkPlan,
        control: V2GraphUpdaterControl,
        embeddingSha256: String?,
    ) {
        val n = index.numTracks
        val k = knnK
        val idToIndex = HashMap<Long, Int>(n * 2)
        for (i in 0 until n) idToIndex[index.getTrackId(i)] = i
        val fingerprint = if (checkpointsEnabled) {
            checkpointFingerprint(
                plan = plan,
                embeddingFile = File(filesDir, "clamp3.emb"),
                oldGraphFile = null,
                embeddingSha256 = embeddingSha256,
                oldGraphSha256 = null,
            )
        } else {
            null
        }
        val restored = fingerprint?.let { readCheckpoint(it, n, k) }
            ?.takeIf { it.phase == PHASE_FULL_ROWS }
        val neighbors = restored?.neighbors ?: Array(n) { IntArray(k) }
        val weights = restored?.weights ?: Array(n) { FloatArray(k) }
        val startRow = restored?.cursor?.coerceIn(0, n) ?: 0
        var completedDots = Math.multiplyExact(startRow.toLong(), (n - 1).toLong())
        emitDots(
            control,
            plan,
            completedDots,
            if (completedDots == 0L) "Starting full graph comparisons"
            else "Resuming full graph comparisons",
        )
        val checkpointEvery = checkpointInterval(n)
        val progressEvery = progressInterval(n)
        for (row in startRow until n) {
            val trackId = index.getTrackId(row)
            val embedding = index.getEmbedding(row)
            val topK = index.findTopK(
                embedding,
                k,
                excludeIds = setOf(trackId),
                cancellationCheck = {
                    control.onControlPoint(
                        V2GraphUpdaterStage.SIMILARITY_DOT_PRODUCTS,
                        completedDots,
                    )
                },
            )
            writeTopK(neighbors[row], weights[row], topK, idToIndex, k)
            completedDots = Math.addExact(completedDots, (n - 1).toLong())
            if ((row + 1) % progressEvery == 0 || row == n - 1) {
                emitDots(control, plan, completedDots, "Built ${row + 1}/$n graph rows")
            }
            controlAfterCompletedUnit(
                control,
                fingerprint,
                PHASE_FULL_ROWS,
                row + 1,
                neighbors,
                weights,
                completedDots,
            )
            if (fingerprint != null && ((row + 1) % checkpointEvery == 0 || row == n - 1)) {
                writeCheckpoint(
                    fingerprint,
                    PHASE_FULL_ROWS,
                    row + 1,
                    neighbors,
                    weights,
                )
            }
        }
        require(completedDots == plan.similarityDotProducts) {
            "full graph dot-product count disagrees with its plan"
        }
        publishGraph(index, graphFile, neighbors, weights, plan, control)
        if (checkpointsEnabled) checkpointFile().delete()
    }

    private fun publishGraph(
        index: EmbeddingIndex,
        graphFile: File,
        neighbors: Array<IntArray>,
        weights: Array<FloatArray>,
        plan: V2GraphWorkPlan,
        control: V2GraphUpdaterControl,
    ) {
        val blob = buildGraphBinary(index, neighbors, weights, knnK, control, plan)
        val graphBytes = blob.size.toLong()
        require(plan.graphBinaryOutputBytes ==
            graphBytes * if (storeGraphInDatabase) 2L else 1L
        )
        var completed = plan.graphBinaryInputBytes
        if (db.isReadWrite) {
            if (storeGraphInDatabase) {
                db.setBinaryData("knn_graph", blob)
                completed += graphBytes
                emit(control, plan, V2GraphUpdaterStage.GRAPH_BINARY_BYTES, completed,
                    plan.graphBinaryBytes, "Committed graph bytes to the private database")
                control.onControlPoint(V2GraphUpdaterStage.GRAPH_BINARY_BYTES, completed)
            } else {
                db.deleteBinaryData("knn_graph")
            }
        } else {
            throw IllegalStateException("graph publication requires a writable private database")
        }

        val temporary = File(graphFile.parentFile, ".${graphFile.name}.incomplete-${UUID.randomUUID()}")
        try {
            FileOutputStream(temporary).use { stream ->
                stream.write(blob)
                stream.fd.sync()
            }
            EmbeddingIndex.replaceAtomically(temporary, graphFile)
        } finally {
            temporary.delete()
        }
        completed += graphBytes
        emit(control, plan, V2GraphUpdaterStage.GRAPH_BINARY_BYTES, completed,
            plan.graphBinaryBytes, "Published ${plan.targetNodes} graph rows atomically")
        control.onControlPoint(V2GraphUpdaterStage.GRAPH_BINARY_BYTES, completed)
    }

    private fun parseOldGraph(graphFile: File, expectedSha256: String? = null): OldGraph? {
        if (!graphFile.isFile || graphFile.length() < 8L) return null
        return try {
            RandomAccessFile(graphFile, "r").use { raf ->
                val headerBytes = ByteArray(8)
                raf.readFully(headerBytes)
                val fileDigest = expectedSha256?.let { MessageDigest.getInstance("SHA-256") }
                    ?.apply { update(headerBytes) }
                val header = ByteBuffer.wrap(headerBytes).order(ByteOrder.LITTLE_ENDIAN)
                val n = header.int
                val k = header.int
                require(n > 0 && k > 0)
                require(raf.length() == V2GraphWorkPlanner.graphFileBytes(n, k))
                val idBytes = ByteArray(Math.multiplyExact(n, 8))
                raf.readFully(idBytes)
                fileDigest?.update(idBytes)
                val ids = ByteBuffer.wrap(idBytes).order(ByteOrder.LITTLE_ENDIAN)
                val trackIds = LongArray(n) { ids.long }
                val graphBytes = ByteArray(Math.multiplyExact(Math.multiplyExact(n, k), 8))
                raf.readFully(graphBytes)
                fileDigest?.let { digest ->
                    digest.update(graphBytes)
                    require(digest.digest().joinToString("") { byte -> "%02x".format(byte) } ==
                        expectedSha256
                    ) { "manifest-bound base graph hash changed" }
                }
                val graph = ByteBuffer.wrap(graphBytes).order(ByteOrder.LITTLE_ENDIAN)
                var validNeighbors = 0
                var validForExactUpdate = true
                val neighborTrackIds = Array(n) {
                    val node = it
                    var rowWeight = 0.0
                    val rowNeighbors = HashSet<Int>(k * 2)
                    LongArray(k) {
                        val neighborIndex = graph.int
                        val weight = graph.float
                        if (!weight.isFinite() || weight < 0f) validForExactUpdate = false
                        rowWeight += weight.toDouble()
                        if (!rowNeighbors.add(neighborIndex) || neighborIndex == node) {
                            validForExactUpdate = false
                        }
                        if (neighborIndex in 0 until n) {
                            validNeighbors++
                            trackIds[neighborIndex]
                        } else {
                            -1L
                        }
                    }.also {
                        if (kotlin.math.abs(rowWeight - 1.0) > 0.005) {
                            validForExactUpdate = false
                        }
                    }
                }
                OldGraph(
                    trackIds,
                    neighborTrackIds,
                    k,
                    validNeighbors,
                    validForExactUpdate && validNeighbors == n * k,
                )
            }
        } catch (error: Exception) {
            Log.w(TAG, "Rejected old graph: ${error.message}")
            null
        }
    }

    private fun writeTopK(
        neighborsRow: IntArray,
        weightsRow: FloatArray,
        topK: List<Pair<Long, Float>>,
        idToIdx: Map<Long, Int>,
        k: Int,
    ) {
        require(topK.isNotEmpty()) { "graph row has no non-self neighbor" }
        for (position in topK.indices) {
            neighborsRow[position] = idToIdx[topK[position].first] ?: 0
            weightsRow[position] = maxOf(topK[position].second, 0f)
        }
        for (position in topK.size until k) {
            neighborsRow[position] = 0
            weightsRow[position] = 0f
        }
        V2GraphWeightPolicy.normalizeNonnegativeInPlace(weightsRow, topK.size)
    }

    /** Exact NativeMath ordering over one already-computed all-library score vector. */
    private fun writeTopKFromSimilarities(
        row: Int,
        similarities: FloatArray,
        neighborsRow: IntArray,
        weightsRow: FloatArray,
        trackIdsByIndex: LongArray,
        k: Int,
    ) {
        require(similarities.size == trackIdsByIndex.size && row in similarities.indices)
        require(similarities.size - 1 >= k) { "graph row has fewer than K non-self candidates" }
        var selected = 0
        for (candidate in similarities.indices) {
            if (candidate == row) continue
            if (selected < k) {
                neighborsRow[selected] = candidate
                weightsRow[selected] = similarities[candidate]
                selected++
            } else {
                V2GraphTopKOrdering.tryInsert(
                    neighborsRow,
                    weightsRow,
                    candidate,
                    similarities[candidate],
                    trackIdsByIndex,
                    k,
                )
            }
        }
        require(selected == k)
        V2GraphTopKOrdering.sortBestFirst(neighborsRow, weightsRow, trackIdsByIndex, k)
        V2GraphWeightPolicy.normalizeNonnegativeInPlace(weightsRow, k)
    }

    private fun buildGraphBinary(
        index: EmbeddingIndex,
        neighbors: Array<IntArray>,
        weights: Array<FloatArray>,
        k: Int,
        control: V2GraphUpdaterControl,
        plan: V2GraphWorkPlan,
    ): ByteArray {
        val n = index.numTracks
        val size = Math.toIntExact(V2GraphWorkPlanner.graphFileBytes(n, k))
        val buffer = ByteBuffer.allocate(size).order(ByteOrder.LITTLE_ENDIAN)
        buffer.putInt(n)
        buffer.putInt(k)
        for (row in 0 until n) buffer.putLong(index.getTrackId(row))
        for (row in 0 until n) {
            for (slot in 0 until k) {
                buffer.putInt(neighbors[row][slot])
                buffer.putFloat(weights[row][slot])
            }
            if (row % CONTROL_ROW_INTERVAL == 0) {
                control.onControlPoint(
                    V2GraphUpdaterStage.SIMILARITY_DOT_PRODUCTS,
                    plan.similarityDotProducts,
                )
            }
        }
        return buffer.array()
    }

    private fun emitDots(
        control: V2GraphUpdaterControl,
        plan: V2GraphWorkPlan,
        completed: Long,
        detail: String,
    ) = emit(
        control,
        plan,
        V2GraphUpdaterStage.SIMILARITY_DOT_PRODUCTS,
        completed,
        plan.similarityDotProducts,
        detail,
    )

    private fun emit(
        control: V2GraphUpdaterControl,
        plan: V2GraphWorkPlan,
        stage: V2GraphUpdaterStage,
        completed: Long,
        total: Long,
        detail: String,
    ) {
        require(completed in 0L..total) { "$stage progress $completed is outside 0..$total" }
        control.onProgress(V2GraphUpdaterProgress(plan, stage, completed, total, detail))
    }

    private fun controlAfterCompletedUnit(
        control: V2GraphUpdaterControl,
        fingerprint: String?,
        phase: Int,
        cursor: Int,
        neighbors: Array<IntArray>,
        weights: Array<FloatArray>,
        completedDots: Long,
    ) {
        try {
            control.onControlPoint(V2GraphUpdaterStage.SIMILARITY_DOT_PRODUCTS, completedDots)
        } catch (error: Throwable) {
            fingerprint?.let { writeCheckpoint(it, phase, cursor, neighbors, weights) }
            throw error
        }
    }

    private fun checkpointInterval(nodes: Int): Int =
        ((CHECKPOINT_TARGET_DOTS + nodes - 1L) / nodes.toLong())
            .coerceIn(1L, Int.MAX_VALUE.toLong())
            .toInt()

    private fun progressInterval(nodes: Int): Int =
        ((PROGRESS_TARGET_DOTS + nodes - 1L) / nodes.toLong())
            .coerceIn(1L, Int.MAX_VALUE.toLong())
            .toInt()

    private fun checkpointFingerprint(
        plan: V2GraphWorkPlan,
        embeddingFile: File,
        oldGraphFile: File?,
        embeddingSha256: String? = null,
        oldGraphSha256: String? = null,
    ): String {
        val digest = MessageDigest.getInstance("SHA-256")
        digest.update("v2-exact-graph-checkpoint-v2".toByteArray(StandardCharsets.UTF_8))
        digest.update(plan.strategy.name.toByteArray(StandardCharsets.UTF_8))
        listOf(
            plan.targetNodes,
            plan.baseGraphNodes,
            plan.newNodes,
            plan.removedBaseNodes,
            plan.rescannedBaseNodes,
            plan.retainedBaseNodes,
            plan.neighborsPerNode,
            plan.embeddingDimension,
        ).forEach { value ->
            digest.update(ByteBuffer.allocate(4).order(ByteOrder.BIG_ENDIAN).putInt(value).array())
        }
        digest.update(
            (embeddingSha256 ?: V2FileSha256.digest(embeddingFile))
                .toByteArray(StandardCharsets.US_ASCII),
        )
        if (oldGraphFile != null && oldGraphFile.isFile) {
            digest.update(
                (oldGraphSha256 ?: V2FileSha256.digest(oldGraphFile))
                    .toByteArray(StandardCharsets.US_ASCII),
            )
        }
        return digest.digest().joinToString("") { byte -> "%02x".format(byte) }
    }

    private fun checkpointFile(): File = File(filesDir, ".graph-work-v2/checkpoint.bin")

    private fun writeCheckpoint(
        fingerprint: String,
        phase: Int,
        cursor: Int,
        neighbors: Array<IntArray>,
        weights: Array<FloatArray>,
    ) {
        val n = neighbors.size
        val k = neighbors.firstOrNull()?.size ?: knnK
        require(weights.size == n && weights.all { it.size == k })
        val fingerprintBytes = fingerprint.toByteArray(StandardCharsets.US_ASCII)
        require(fingerprintBytes.size == 64)
        val bodySize = Math.addExact(
            6 * 4 + fingerprintBytes.size,
            Math.multiplyExact(Math.multiplyExact(n, k), 8),
        )
        val bytes = ByteBuffer.allocate(Math.addExact(bodySize, DIGEST_BYTES))
            .order(ByteOrder.LITTLE_ENDIAN)
        bytes.putInt(CHECKPOINT_MAGIC)
        bytes.putInt(CHECKPOINT_VERSION)
        bytes.putInt(phase)
        bytes.putInt(cursor)
        bytes.putInt(n)
        bytes.putInt(k)
        bytes.put(fingerprintBytes)
        for (row in 0 until n) {
            for (slot in 0 until k) {
                bytes.putInt(neighbors[row][slot])
                bytes.putFloat(weights[row][slot])
            }
        }
        val body = bytes.array().copyOf(bodySize)
        bytes.put(MessageDigest.getInstance("SHA-256").digest(body))
        val target = checkpointFile()
        require(target.parentFile?.isDirectory == true || target.parentFile?.mkdirs() == true)
        val temporary = File(target.parentFile, ".${target.name}.incomplete-${UUID.randomUUID()}")
        try {
            FileOutputStream(temporary).use { stream ->
                stream.write(bytes.array())
                stream.fd.sync()
            }
            EmbeddingIndex.replaceAtomically(temporary, target)
        } finally {
            temporary.delete()
        }
    }

    private fun readCheckpoint(
        fingerprint: String,
        expectedNodes: Int,
        expectedK: Int,
    ): GraphCheckpoint? {
        val file = checkpointFile()
        if (!file.isFile) return null
        return try {
            val bytes = file.readBytes()
            val bodySize = bytes.size - DIGEST_BYTES
            require(bodySize > 0)
            val actualDigest = MessageDigest.getInstance("SHA-256")
                .digest(bytes.copyOf(bodySize))
            require(actualDigest.contentEquals(bytes.copyOfRange(bodySize, bytes.size)))
            val buffer = ByteBuffer.wrap(bytes, 0, bodySize).order(ByteOrder.LITTLE_ENDIAN)
            require(buffer.int == CHECKPOINT_MAGIC)
            require(buffer.int == CHECKPOINT_VERSION)
            val phase = buffer.int
            val cursor = buffer.int
            val n = buffer.int
            val k = buffer.int
            require(n == expectedNodes && k == expectedK && cursor >= 0)
            val fingerprintBytes = ByteArray(64).also(buffer::get)
            require(String(fingerprintBytes, StandardCharsets.US_ASCII) == fingerprint)
            val expectedBodySize = Math.addExact(
                6 * 4 + 64,
                Math.multiplyExact(Math.multiplyExact(n, k), 8),
            )
            require(bodySize == expectedBodySize)
            val neighbors = Array(n) { IntArray(k) }
            val weights = Array(n) { FloatArray(k) }
            for (row in 0 until n) {
                for (slot in 0 until k) {
                    val neighbor = buffer.int
                    require(neighbor in 0 until n)
                    neighbors[row][slot] = neighbor
                    weights[row][slot] = buffer.float
                }
            }
            GraphCheckpoint(phase, cursor, neighbors, weights)
        } catch (error: Exception) {
            Log.w(TAG, "Discarded invalid graph checkpoint: ${error.message}")
            file.delete()
            null
        }
    }

    private fun clearRows(neighbors: Array<IntArray>, weights: Array<FloatArray>) {
        neighbors.forEach { it.fill(0) }
        weights.forEach { it.fill(0f) }
    }

    private data class OldGraph(
        val trackIds: LongArray,
        val neighborTrackIds: Array<LongArray>,
        val neighborsPerNode: Int,
        val validNeighborCount: Int,
        val validForExactUpdate: Boolean,
    )

    private data class GraphMutation(
        val oldGraph: OldGraph,
        val newTrackIndices: IntArray,
        val unaffectedBasePositions: IntArray,
        val affectedBasePositions: IntArray,
        val removedBaseNodes: Int,
    ) {
        val retainedBaseNodes: Int
            get() = unaffectedBasePositions.size + affectedBasePositions.size
    }

    private data class GraphCheckpoint(
        val phase: Int,
        val cursor: Int,
        val neighbors: Array<IntArray>,
        val weights: Array<FloatArray>,
    )
}
