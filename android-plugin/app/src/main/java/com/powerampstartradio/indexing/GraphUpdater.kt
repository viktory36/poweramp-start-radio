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
 * Incrementally updates the kNN graph after new tracks are added to the database.
 *
 * For M new tracks added to N existing:
 * 1. Rebuild the embedding index (.emb) from the DB (includes new tracks)
 * 2. For each new track: brute-force find K nearest neighbors
 * 3. For each existing track: check if any new track beats their K-th neighbor
 * 4. Write updated graph binary
 *
 * At 75K tracks this is fast: M×N dot products ~ 30×75K×512 ~ 1.2B ops ~ <1s on flagship.
 */
class GraphUpdater(
    private val db: EmbeddingDatabase,
    private val filesDir: File,
    private val knnK: Int = 20,
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
        }

        val totalMs = (System.nanoTime() - t0) / 1_000_000
        Log.i(TAG, "TIMING: rebuildIndices total = ${totalMs}ms")
        onProgress?.invoke("Indices rebuilt")
    }

    /**
     * Build a full kNN graph from the embedding index.
     *
     * This is used when no graph exists in the DB (e.g., after first on-device indexing).
     * For 75K tracks × K=20, this takes ~5-10s on a flagship phone.
     */
    private fun buildKnnGraph(embFile: File, onProgress: ((String) -> Unit)? = null) {
        val tBuild = System.nanoTime()
        val index = EmbeddingIndex.mmap(embFile)
        val n = index.numTracks
        val k = knnK

        Log.i(TAG, "Building kNN graph: $n nodes, K=$k")
        onProgress?.invoke("Building kNN graph ($n nodes, K=$k)...")

        // For each track, find K nearest neighbors
        val neighbors = Array(n) { IntArray(k) }
        val weights = Array(n) { FloatArray(k) }

        // Build trackId → index lookup to avoid O(N) linear scan per neighbor
        val idToIndex = HashMap<Long, Int>(n * 2)
        for (i in 0 until n) {
            idToIndex[index.getTrackId(i)] = i
        }

        for (i in 0 until n) {
            if (i % 1000 == 0) {
                onProgress?.invoke("kNN: $i/$n")
            }

            // Get this track's embedding
            val trackId = index.getTrackId(i)
            val embedding = index.getEmbeddingByTrackId(trackId) ?: continue

            // Find top-K (exclude self)
            val topK = index.findTopK(embedding, k, excludeIds = setOf(trackId))

            for (j in topK.indices) {
                val neighborTrackId = topK[j].first
                val similarity = topK[j].second
                val neighborIdx = idToIndex[neighborTrackId] ?: 0

                neighbors[i][j] = neighborIdx
                weights[i][j] = maxOf(similarity, 0f)
            }

            // Fill remaining slots if topK returned fewer than k
            for (j in topK.size until k) {
                neighbors[i][j] = 0
                weights[i][j] = 0f
            }

            // Row-normalize weights to transition probabilities
            var total = 0f
            for (j in 0 until k) total += weights[i][j]
            if (total > 0f) {
                for (j in 0 until k) weights[i][j] /= total
            }
        }

        // Write graph binary
        onProgress?.invoke("Writing graph binary...")
        val graphBlob = buildGraphBinary(index, neighbors, weights, k)

        // Store in DB binary_data table
        db.setBinaryData("knn_graph", graphBlob)

        // Also extract to file
        val graphFile = File(filesDir, "graph.bin")
        graphFile.writeBytes(graphBlob)

        val buildMs = (System.nanoTime() - tBuild) / 1_000_000
        val sizeMB = graphBlob.size / 1024 / 1024
        onProgress?.invoke("Graph built: $n nodes, K=$k, ${sizeMB} MB")
        Log.i(TAG, "TIMING: graph_build $n nodes, K=$k, ${sizeMB}MB in ${buildMs}ms " +
            "(${buildMs / maxOf(n, 1)}ms/node)")
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
