package com.powerampstartradio.indexing

import android.util.Log
import com.powerampstartradio.data.EmbeddingDatabase
import com.powerampstartradio.data.EmbeddingIndex
import com.powerampstartradio.data.GraphIndex
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Updates the kNN graph after new tracks are added to the database.
 *
 * Two strategies:
 * - **Full rebuild**: When no prior graph exists. N × findTopK = N² dot products.
 * - **Incremental**: When graph exists but is stale (M new tracks added).
 *   New tracks get full findTopK (M × N). Existing tracks only check M new
 *   candidates against their current K neighbors (N × (K+M) dot products).
 *   For M=2, N=75K, K=5: ~525K vs 5.6B dot products. Mathematically identical.
 */
class GraphUpdater(
    private val db: EmbeddingDatabase,
    private val filesDir: File,
    private val knnK: Int = 5,
) {
    companion object {
        private const val TAG = "GraphUpdater"
    }

    /**
     * Rebuild the embedding index and kNN graph after new tracks have been inserted.
     *
     * @param onProgress Status callback
     */
    fun rebuildIndices(onProgress: ((String) -> Unit)? = null) {
        val t0 = System.nanoTime()

        // Step 1: Regenerate .emb file from database
        onProgress?.invoke("Rebuilding embedding index...")
        val embFile = File(filesDir, "clamp3.emb")
        val tEmb = System.nanoTime()
        EmbeddingIndex.extractFromDatabase(db, embFile) { current, total ->
            onProgress?.invoke("Extracting embeddings: $current/$total")
        }
        val embMs = (System.nanoTime() - tEmb) / 1_000_000
        Log.i(TAG, "TIMING: extract embeddings = ${embMs}ms (${embFile.length() / 1024}KB)")

        // Step 2: Regenerate graph from database (if graph exists in DB)
        onProgress?.invoke("Extracting graph...")
        val graphFile = File(filesDir, "graph.bin")
        val tGraph = System.nanoTime()
        val hasGraph = GraphIndex.extractFromDatabase(db, graphFile)
        val graphExtractMs = (System.nanoTime() - tGraph) / 1_000_000
        Log.d(TAG, "TIMING: extract graph = ${graphExtractMs}ms (hasGraph=$hasGraph)")

        if (!hasGraph) {
            onProgress?.invoke("No graph in database, building from scratch...")
            buildKnnGraph(embFile, onProgress)
        } else {
            val embTrackCount = db.getEmbeddingCount()
            val graphNodeCount = GraphIndex.readHeaderNodeCount(graphFile)
            if (graphNodeCount in 1 until embTrackCount) {
                val newCount = embTrackCount - graphNodeCount
                Log.i(TAG, "Graph stale: $graphNodeCount nodes vs $embTrackCount embeddings ($newCount new)")
                onProgress?.invoke("Graph stale, updating $newCount new tracks...")
                incrementalUpdate(embFile, graphFile, onProgress)
            }
        }

        val totalMs = (System.nanoTime() - t0) / 1_000_000
        Log.i(TAG, "TIMING: rebuildIndices total = ${totalMs}ms")
        onProgress?.invoke("Indices rebuilt")
    }

    /**
     * Incrementally update the kNN graph for M new tracks.
     *
     * For each new track: full findTopK (M × N dot products).
     * For each existing track: recompute similarities to current K neighbors,
     * check if any new track beats the weakest, take top-K from K+M candidates.
     * Row-normalize to transition probabilities.
     *
     * Falls back to full rebuild if the old graph can't be parsed.
     */
    private fun incrementalUpdate(
        embFile: File,
        graphFile: File,
        onProgress: ((String) -> Unit)? = null
    ) {
        val tBuild = System.nanoTime()

        // Parse old graph to get existing neighbor relationships as track IDs
        val oldGraph = parseOldGraph(graphFile)
        if (oldGraph == null) {
            Log.w(TAG, "Cannot parse old graph, falling back to full rebuild")
            buildKnnGraph(embFile, onProgress)
            return
        }

        val index = EmbeddingIndex.mmap(embFile)
        val totalN = index.numTracks
        val k = knnK

        // Map trackId → index in new .emb file
        val idToIdx = HashMap<Long, Int>(totalN * 2)
        for (i in 0 until totalN) idToIdx[index.getTrackId(i)] = i

        // Identify new tracks (in new .emb but not in old graph)
        val oldIdSet = HashSet<Long>(oldGraph.trackIds.size * 2)
        for (id in oldGraph.trackIds) oldIdSet.add(id)

        val newTrackIndices = ArrayList<Int>()
        for (i in 0 until totalN) {
            if (index.getTrackId(i) !in oldIdSet) newTrackIndices.add(i)
        }
        val m = newTrackIndices.size

        Log.i(TAG, "Incremental kNN: $m new + ${oldGraph.trackIds.size} existing = $totalN, K=$k")

        if (m == 0) {
            Log.w(TAG, "No new tracks found despite stale graph count")
            return
        }

        onProgress?.invoke("Incremental kNN: $m new tracks...")

        // Preload new track embeddings for reuse
        val newEmbs = Array(m) { j ->
            index.getEmbedding(newTrackIndices[j])
        }

        val neighbors = Array(totalN) { IntArray(k) }
        val weights = Array(totalN) { FloatArray(k) }

        // Step 1: New tracks — full findTopK
        for (j in 0 until m) {
            val ni = newTrackIndices[j]
            val trackId = index.getTrackId(ni)
            val topK = index.findTopK(newEmbs[j], k, excludeIds = setOf(trackId))
            writeTopK(neighbors[ni], weights[ni], topK, idToIdx, k)
        }

        // Step 2a: Existing tracks — seed each row with the old graph's K neighbors.
        val oldN = oldGraph.trackIds.size
        val oldIndices = IntArray(oldN) { oi -> idToIdx[oldGraph.trackIds[oi]] ?: -1 }
        val updatedByNew = BooleanArray(oldN)
        val reuseEmb = FloatArray(index.dim)

        for (oi in 0 until oldN) {
            if (oi % 5000 == 0 && oi > 0) {
                onProgress?.invoke("Preparing existing: $oi/$oldN")
            }

            val idx = oldIndices[oi]
            if (idx < 0) continue

            index.copyEmbedding(idx, reuseEmb)
            var writePos = 0
            for (nid in oldGraph.neighborTrackIds[oi]) {
                if (nid < 0L) continue
                val nIdx = idToIdx[nid] ?: continue
                neighbors[idx][writePos] = nIdx
                weights[idx][writePos] = maxOf(index.dotProduct(reuseEmb, nIdx), 0f)
                writePos++
                if (writePos == k) break
            }
            for (ki in writePos until k) {
                neighbors[idx][ki] = 0
                weights[idx][ki] = 0f
            }
            sortRowDescending(neighbors[idx], weights[idx], k)
        }

        // Step 2b: Stream each new track through the native allSimilarities() path.
        val newProgressInterval = maxOf(m / 20, 1)
        for (j in 0 until m) {
            if ((j + 1) % newProgressInterval == 0 || j == 0 || j == m - 1) {
                onProgress?.invoke("Checking new tracks: ${j + 1}/$m")
            }

            val newIdx = newTrackIndices[j]
            val sims = index.computeAllSimilarities(newEmbs[j])
            for (oi in 0 until oldN) {
                val idx = oldIndices[oi]
                if (idx < 0) continue

                val sim = sims[idx]
                if (sim <= 0f) continue
                if (maybeInsertCandidate(neighbors[idx], weights[idx], newIdx, sim, k)) {
                    updatedByNew[oi] = true
                }
            }
        }

        var updatedCount = 0
        for (oi in 0 until oldN) {
            val idx = oldIndices[oi]
            if (idx < 0) continue
            sortRowDescending(neighbors[idx], weights[idx], k)
            normalizeRow(weights[idx], k)
            if (updatedByNew[oi]) updatedCount++
        }

        Log.i(TAG, "$updatedCount of $oldN existing nodes gained new neighbors")

        onProgress?.invoke("Writing graph...")
        val blob = buildGraphBinary(index, neighbors, weights, k)
        if (db.isReadWrite) db.setBinaryData("knn_graph", blob)
        File(filesDir, "graph.bin").writeBytes(blob)

        val ms = (System.nanoTime() - tBuild) / 1_000_000
        val sizeMB = blob.size / 1024 / 1024
        onProgress?.invoke("Graph updated: $totalN nodes ($m new) in ${ms}ms")
        Log.i(TAG, "TIMING: incremental_knn $totalN nodes ($m new, $updatedCount updated) " +
            "K=$k ${sizeMB}MB in ${ms}ms")
    }

    /**
     * Parse an existing graph binary to extract neighbor relationships as track IDs.
     * Returns null if the file is missing or corrupt.
     */
    private fun parseOldGraph(graphFile: File): OldGraph? {
        if (!graphFile.exists() || graphFile.length() < 8) return null
        return try {
            RandomAccessFile(graphFile, "r").use { raf ->
                val hdrBuf = ByteArray(8)
                raf.readFully(hdrBuf)
                val hdr = ByteBuffer.wrap(hdrBuf).order(ByteOrder.LITTLE_ENDIAN)
                val n = hdr.getInt(0)
                val k = hdr.getInt(4)

                val idBytes = ByteArray(n * 8)
                raf.readFully(idBytes)
                val idBuf = ByteBuffer.wrap(idBytes).order(ByteOrder.LITTLE_ENDIAN)
                val trackIds = LongArray(n) { idBuf.getLong() }

                val graphBytes = ByteArray(n * k * 8)
                raf.readFully(graphBytes)
                val gBuf = ByteBuffer.wrap(graphBytes).order(ByteOrder.LITTLE_ENDIAN)

                // Convert neighbor indices to track IDs (self-contained within old graph)
                val neighborTrackIds = Array(n) {
                    LongArray(k) {
                        val neighborIdx = gBuf.getInt()
                        gBuf.getFloat()  // skip weight (row-normalized, not useful)
                        if (neighborIdx in 0 until n) trackIds[neighborIdx] else -1L
                    }
                }

                OldGraph(trackIds, neighborTrackIds)
            }
        } catch (e: Exception) {
            Log.w(TAG, "Failed to parse old graph: ${e.message}")
            null
        }
    }

    private class OldGraph(
        val trackIds: LongArray,
        val neighborTrackIds: Array<LongArray>,
    )

    /**
     * Build a full kNN graph from the embedding index.
     *
     * Used when no graph exists in the DB (e.g., after first on-device indexing).
     */
    private fun buildKnnGraph(embFile: File, onProgress: ((String) -> Unit)? = null) {
        val tBuild = System.nanoTime()
        val index = EmbeddingIndex.mmap(embFile)
        val n = index.numTracks
        val k = knnK

        Log.i(TAG, "Building kNN graph: $n nodes, K=$k")
        onProgress?.invoke("Building kNN graph ($n nodes, K=$k)...")

        val neighbors = Array(n) { IntArray(k) }
        val weights = Array(n) { FloatArray(k) }

        val idToIndex = HashMap<Long, Int>(n * 2)
        for (i in 0 until n) {
            idToIndex[index.getTrackId(i)] = i
        }

        for (i in 0 until n) {
            if (i % 1000 == 0) {
                onProgress?.invoke("kNN: $i/$n")
            }

            val trackId = index.getTrackId(i)
            val embedding = index.getEmbedding(i)
            val topK = index.findTopK(embedding, k, excludeIds = setOf(trackId))
            writeTopK(neighbors[i], weights[i], topK, idToIndex, k)
        }

        onProgress?.invoke("Writing graph binary...")
        val graphBlob = buildGraphBinary(index, neighbors, weights, k)

        if (db.isReadWrite) {
            db.setBinaryData("knn_graph", graphBlob)
        } else {
            Log.d(TAG, "DB is read-only, skipping binary_data write (graph.bin file only)")
        }

        val graphFile = File(filesDir, "graph.bin")
        graphFile.writeBytes(graphBlob)

        val buildMs = (System.nanoTime() - tBuild) / 1_000_000
        val sizeMB = graphBlob.size / 1024 / 1024
        onProgress?.invoke("Graph built: $n nodes, K=$k, ${sizeMB} MB")
        Log.i(TAG, "TIMING: graph_build $n nodes, K=$k, ${sizeMB}MB in ${buildMs}ms " +
            "(${buildMs / maxOf(n, 1)}ms/node)")
    }

    /** Write findTopK results into neighbor/weight arrays and row-normalize. */
    private fun writeTopK(
        neighborsRow: IntArray,
        weightsRow: FloatArray,
        topK: List<Pair<Long, Float>>,
        idToIdx: Map<Long, Int>,
        k: Int
    ) {
        for (j in topK.indices) {
            neighborsRow[j] = idToIdx[topK[j].first] ?: 0
            weightsRow[j] = maxOf(topK[j].second, 0f)
        }
        for (j in topK.size until k) {
            neighborsRow[j] = 0
            weightsRow[j] = 0f
        }
        normalizeRow(weightsRow, k)
    }

    /** Row-normalize weights to transition probabilities (sum to 1). */
    private fun normalizeRow(row: FloatArray, k: Int) {
        var sum = 0f
        for (j in 0 until k) sum += row[j]
        if (sum > 0f) for (j in 0 until k) row[j] /= sum
    }

    /** Replace the weakest candidate when a better score arrives. */
    private fun maybeInsertCandidate(
        neighborsRow: IntArray,
        weightsRow: FloatArray,
        neighborIdx: Int,
        score: Float,
        k: Int
    ): Boolean {
        var minPos = 0
        var minScore = weightsRow[0]
        for (j in 1 until k) {
            if (weightsRow[j] < minScore) {
                minScore = weightsRow[j]
                minPos = j
            }
        }
        if (score <= minScore) return false
        neighborsRow[minPos] = neighborIdx
        weightsRow[minPos] = score
        return true
    }

    /** K is tiny, so a simple in-place sort keeps rows ordered by descending score. */
    private fun sortRowDescending(neighborsRow: IntArray, weightsRow: FloatArray, k: Int) {
        for (i in 0 until k - 1) {
            var best = i
            for (j in i + 1 until k) {
                if (weightsRow[j] > weightsRow[best]) best = j
            }
            if (best == i) continue

            val tmpNeighbor = neighborsRow[i]
            neighborsRow[i] = neighborsRow[best]
            neighborsRow[best] = tmpNeighbor

            val tmpWeight = weightsRow[i]
            weightsRow[i] = weightsRow[best]
            weightsRow[best] = tmpWeight
        }
    }

    /**
     * Build graph binary blob matching the desktop format.
     *
     * Format: Header (N uint32, K uint32) + ID map (N × int64) + Graph (N × K × (uint32 + float32))
     */
    private fun buildGraphBinary(
        index: EmbeddingIndex,
        neighbors: Array<IntArray>,
        weights: Array<FloatArray>,
        k: Int
    ): ByteArray {
        val n = index.numTracks
        val size = 8 + n * 8 + n * k * 8  // header + id_map + graph
        val buffer = ByteBuffer.allocate(size).order(ByteOrder.LITTLE_ENDIAN)

        // Header
        buffer.putInt(n)
        buffer.putInt(k)

        // ID map
        for (i in 0 until n) {
            buffer.putLong(index.getTrackId(i))
        }

        // Graph data
        for (i in 0 until n) {
            for (j in 0 until k) {
                buffer.putInt(neighbors[i][j])
                buffer.putFloat(weights[i][j])
            }
        }

        return buffer.array()
    }
}
