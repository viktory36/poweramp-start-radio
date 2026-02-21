package com.powerampstartradio.indexing

import android.util.Log
import com.powerampstartradio.data.EmbeddingDatabase
import com.powerampstartradio.data.EmbeddingIndex
import com.powerampstartradio.data.GraphIndex
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * On-device re-fusion engine matching the desktop `poweramp-indexer fuse` command.
 *
 * Recomputes everything from scratch:
 * 1. SVD projection (via covariance matrix eigendecomposition)
 * 2. Fused embeddings for ALL tracks (re-projected through new SVD)
 * 3. k-means clustering
 * 4. kNN graph
 *
 * Uses NEON-accelerated native math (NativeMath JNI) for dot products in
 * k-means and kNN, giving ~6-10x speedup over scalar Kotlin loops.
 *
 * Memory-efficient: streams embeddings twice (covariance + projection) rather than
 * holding the full N×1024 matrix. Peak memory: ~10 MB for the 1024×1024 covariance
 * matrix + eigenvector matrix, plus ~150 MB for k-means (75K × 512d flat array).
 *
 * Computation time on Snapdragon 8 Gen 3 (~75K tracks):
 * - Covariance matrix: ~4 min (streaming 75K×1024 with 150K SQLite queries)
 * - Eigendecomposition: ~4 min (Jacobi, 1024×1024)
 * - Projection pass: ~1.5 min (streaming 75K×1024 → 512)
 * - k-means: ~3 min (75K × 200 clusters × 512d, NEON-accelerated)
 * - kNN graph: ~3 min (cluster-accelerated, NEON dot products)
 * Total: ~15 minutes
 */
class FusionEngine(
    private val db: EmbeddingDatabase,
    private val filesDir: File,
    private val targetDim: Int = 512,
    private val nClusters: Int = 200,
    private val knnK: Int = 20,
) {
    companion object {
        private const val TAG = "FusionEngine"
    }

    /**
     * Full re-fusion: SVD + project + cluster + kNN graph.
     *
     * @param onProgress Status callback for UI updates
     */
    fun recomputeFusion(onProgress: ((String) -> Unit)? = null) {
        val fusionStart = System.currentTimeMillis()
        fun progress(msg: String) {
            Log.i(TAG, msg)
            onProgress?.invoke(msg)
        }

        // --- Step 1: Collect track IDs and compute covariance matrix ---
        progress("Loading embeddings for fusion...")

        val rawDb = db.getRawDatabase()
        val sourceDim = detectSourceDim()
        val concatDim = sourceDim * 2

        if (targetDim > concatDim) {
            throw IllegalStateException("Target dim ($targetDim) > concat dim ($concatDim)")
        }

        // Get all track IDs that have at least one embedding
        val mulanTrackIds = getTrackIdsForTable("embeddings_mulan")
        val flamingoTrackIds = getTrackIdsForTable("embeddings_flamingo")
        val allTrackIds = (mulanTrackIds + flamingoTrackIds).sorted().distinct().toLongArray()
        val nTracks = allTrackIds.size
        val bothCount = mulanTrackIds.intersect(flamingoTrackIds.toSet()).size

        progress("Tracks: $nTracks total ($bothCount with both models, " +
            "${mulanTrackIds.size - bothCount} MuLan-only, " +
            "${flamingoTrackIds.size - bothCount} Flamingo-only)")

        if (nTracks == 0) {
            progress("No embeddings to fuse")
            return
        }

        // Compute covariance matrix C = X^T X by streaming (memory: ~8 MB for 1024×1024 doubles)
        val covStart = System.currentTimeMillis()
        progress("Computing covariance matrix ($nTracks tracks, ${concatDim}d)...")
        val covariance = DoubleArray(concatDim * concatDim)
        val mulanSet = mulanTrackIds.toHashSet()
        val flamingoSet = flamingoTrackIds.toHashSet()

        // Batch covariance accumulation for native acceleration
        val covBatchSize = 500
        var covBatch = FloatArray(covBatchSize * concatDim)
        var covBatchIdx = 0

        for ((idx, trackId) in allTrackIds.withIndex()) {
            if (idx % 5000 == 0 && idx > 0) {
                progress("Covariance: $idx/$nTracks")
            }

            val concat = getConcatenatedEmbedding(trackId, sourceDim, mulanSet, flamingoSet)
            concat.copyInto(covBatch, covBatchIdx * concatDim)
            covBatchIdx++

            if (covBatchIdx == covBatchSize || idx == nTracks - 1) {
                NativeMath.covarianceAccum(covariance, covBatch, covBatchIdx, concatDim)
                covBatchIdx = 0
            }
        }
        @Suppress("UNUSED_VALUE")
        covBatch = FloatArray(0) // free

        // Fill lower triangle from upper
        for (i in 0 until concatDim) {
            for (j in 0 until i) {
                covariance[i * concatDim + j] = covariance[j * concatDim + i]
            }
        }

        Log.i(TAG, "TIMING: covariance = ${System.currentTimeMillis() - covStart}ms")

        // --- Step 2: Eigendecomposition ---
        val eigenStart = System.currentTimeMillis()
        progress("Eigendecomposition ($concatDim x $concatDim)...")
        val (eigenvalues, eigenvectors) = jacobiEigen(covariance, concatDim, onProgress)

        // eigenvalues are sorted descending; eigenvectors columns correspond to them.
        // Projection matrix = top targetDim eigenvectors as rows (Vt convention):
        // Vt[targetDim, concatDim] where Vt[i] = eigenvectors[:, i]^T
        val projectionData = FloatArray(targetDim * concatDim)
        for (i in 0 until targetDim) {
            for (j in 0 until concatDim) {
                projectionData[i * concatDim + j] = eigenvectors[j * concatDim + i].toFloat()
            }
        }

        // Compute variance retained
        val totalVar = eigenvalues.sum()
        val retainedVar = eigenvalues.take(targetDim).sum() / totalVar
        progress("Variance retained: ${"%.2f".format(retainedVar * 100)}%")

        Log.i(TAG, "TIMING: eigendecomposition = ${System.currentTimeMillis() - eigenStart}ms")

        // --- Step 3: Re-project all tracks ---
        val projStart = System.currentTimeMillis()
        progress("Projecting $nTracks tracks to ${targetDim}d...")

        // Ensure tables exist
        rawDb.execSQL("""
            CREATE TABLE IF NOT EXISTS embeddings_fused (
                track_id INTEGER PRIMARY KEY,
                embedding BLOB NOT NULL,
                FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE CASCADE
            )
        """)
        rawDb.execSQL("""
            CREATE TABLE IF NOT EXISTS binary_data (
                key TEXT PRIMARY KEY,
                data BLOB NOT NULL
            )
        """)

        rawDb.beginTransaction()
        try {
            // Clear stale fused embeddings from previous runs (e.g., deleted tracks)
            rawDb.execSQL("DELETE FROM embeddings_fused")

            for ((idx, trackId) in allTrackIds.withIndex()) {
                if (idx % 5000 == 0 && idx > 0) {
                    progress("Projecting: $idx/$nTracks")
                }

                val concat = getConcatenatedEmbedding(trackId, sourceDim, mulanSet, flamingoSet)
                // Use native mat-vec multiply for projection
                val fused = NativeMath.matVecMul(projectionData, targetDim, concatDim, concat)
                    ?: FloatMatrix(projectionData, targetDim, concatDim).multiplyVector(concat)
                l2Normalize(fused)
                db.insertEmbedding("embeddings_fused", trackId, fused)
            }
            rawDb.setTransactionSuccessful()
        } finally {
            rawDb.endTransaction()
        }
        progress("Projected $nTracks fused embeddings")

        // Store projection matrix in metadata
        val projBlob = ByteBuffer.allocate(projectionData.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        for (f in projectionData) projBlob.putFloat(f)
        rawDb.execSQL(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            arrayOf("fused_projection", projBlob.array())
        )
        rawDb.execSQL(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            arrayOf("fused_dim", targetDim.toString())
        )
        rawDb.execSQL(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            arrayOf("fused_source_dim", concatDim.toString())
        )
        rawDb.execSQL(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            arrayOf("fused_variance_retained", "%.6f".format(retainedVar))
        )

        Log.i(TAG, "TIMING: projection = ${System.currentTimeMillis() - projStart}ms")

        // --- Step 4: k-means clustering ---
        val kmeansStart = System.currentTimeMillis()
        progress("Loading fused embeddings for clustering...")
        val flatEmbeddings = loadAllFusedEmbeddingsFlat(allTrackIds)

        val actualClusters = minOf(nClusters, nTracks)
        progress("k-means clustering (K=$actualClusters)...")
        val (labels, centroids) = kmeans(flatEmbeddings, nTracks, targetDim,
            actualClusters, onProgress = { progress(it) })

        // Store cluster assignments and centroids
        progress("Writing clusters...")
        rawDb.execSQL("""
            CREATE TABLE IF NOT EXISTS clusters (
                cluster_id INTEGER PRIMARY KEY,
                embedding BLOB NOT NULL
            )
        """)
        // Add cluster_id column if not exists
        try {
            rawDb.execSQL("ALTER TABLE tracks ADD COLUMN cluster_id INTEGER")
        } catch (_: Exception) { }

        rawDb.beginTransaction()
        try {
            for (i in allTrackIds.indices) {
                rawDb.execSQL(
                    "UPDATE tracks SET cluster_id = ? WHERE id = ?",
                    arrayOf<Any>(labels[i], allTrackIds[i])
                )
            }
            rawDb.execSQL("DELETE FROM clusters")
            for (k in 0 until actualClusters) {
                val blob = EmbeddingDatabase.floatArrayToBlob(centroids[k])
                rawDb.execSQL(
                    "INSERT INTO clusters (cluster_id, embedding) VALUES (?, ?)",
                    arrayOf<Any>(k, blob)
                )
            }
            rawDb.setTransactionSuccessful()
        } finally {
            rawDb.endTransaction()
        }

        Log.i(TAG, "TIMING: kmeans = ${System.currentTimeMillis() - kmeansStart}ms")

        // --- Step 5: kNN graph ---
        val knnStart = System.currentTimeMillis()
        progress("Building kNN graph (K=$knnK)...")
        buildKnnGraph(flatEmbeddings, nTracks, targetDim,
            allTrackIds, labels, centroids,
            onProgress = { progress(it) })

        Log.i(TAG, "TIMING: knn_graph = ${System.currentTimeMillis() - knnStart}ms")

        // --- Step 6: Extract .emb and graph.bin files ---
        val extractStart = System.currentTimeMillis()
        progress("Extracting index files...")
        EmbeddingIndex.extractFromDatabase(db, File(filesDir, "fused.emb")) { cur, total ->
            if (cur % 10000 == 0) progress("Extracting embeddings: $cur/$total")
        }
        GraphIndex.extractFromDatabase(db, File(filesDir, "graph.bin"))

        Log.i(TAG, "TIMING: extract_indices = ${System.currentTimeMillis() - extractStart}ms")
        Log.i(TAG, "TIMING: fusion_total = ${System.currentTimeMillis() - fusionStart}ms")

        progress("Re-fusion complete: $nTracks tracks, ${targetDim}d, " +
            "${"%.1f".format(retainedVar * 100)}% variance")
    }

    // --- Helpers ---

    private fun detectSourceDim(): Int {
        val mulanDim = try {
            rawDb().rawQuery("SELECT length(embedding)/4 FROM embeddings_mulan LIMIT 1", null).use {
                if (it.moveToFirst()) it.getInt(0) else null
            }
        } catch (_: Exception) { null }

        val flamingoDim = try {
            rawDb().rawQuery("SELECT length(embedding)/4 FROM embeddings_flamingo LIMIT 1", null).use {
                if (it.moveToFirst()) it.getInt(0) else null
            }
        } catch (_: Exception) { null }

        val dim = mulanDim ?: flamingoDim ?: throw IllegalStateException("No embeddings found")
        if (mulanDim != null && flamingoDim != null && mulanDim != flamingoDim) {
            throw IllegalStateException(
                "MuLan dim ($mulanDim) != Flamingo dim ($flamingoDim). Run reduce first."
            )
        }
        return dim
    }

    private fun rawDb() = db.getRawDatabase()

    private fun getTrackIdsForTable(table: String): List<Long> {
        val ids = mutableListOf<Long>()
        try {
            rawDb().rawQuery("SELECT track_id FROM [$table]", null).use { cursor ->
                while (cursor.moveToNext()) {
                    ids.add(cursor.getLong(0))
                }
            }
        } catch (_: Exception) { }
        return ids
    }

    /**
     * Get concatenated [mulan | flamingo] embedding for a track.
     * Zero-pads if one model is missing (matches desktop behavior).
     */
    private fun getConcatenatedEmbedding(
        trackId: Long,
        sourceDim: Int,
        mulanSet: Set<Long>,
        flamingoSet: Set<Long>,
    ): FloatArray {
        val concat = FloatArray(sourceDim * 2)

        if (trackId in mulanSet) {
            val emb = db.getEmbeddingFromTable("embeddings_mulan", trackId)
            emb?.copyInto(concat, 0, 0, minOf(emb.size, sourceDim))
        }
        if (trackId in flamingoSet) {
            val emb = db.getEmbeddingFromTable("embeddings_flamingo", trackId)
            emb?.copyInto(concat, sourceDim, 0, minOf(emb.size, sourceDim))
        }

        return concat
    }

    /** Load all fused embeddings into a contiguous flat array [n × d] for native math. */
    private fun loadAllFusedEmbeddingsFlat(trackIds: LongArray): FloatArray {
        val d = targetDim
        val flat = FloatArray(trackIds.size * d)
        for (i in trackIds.indices) {
            val emb = db.getEmbeddingFromTable("embeddings_fused", trackIds[i])
            if (emb != null) {
                emb.copyInto(flat, i * d, 0, minOf(emb.size, d))
            }
        }
        return flat
    }

    // --- Jacobi eigenvalue decomposition ---

    /**
     * Jacobi cyclic eigendecomposition for a symmetric matrix.
     *
     * Returns eigenvalues (sorted descending by magnitude) and the corresponding
     * eigenvector matrix (columns = eigenvectors, stored in row-major as DoubleArray).
     *
     * Uses cyclic-by-row sweep ordering with Schur rotations for numerical stability.
     * For a 1024×1024 matrix, converges in ~10-15 sweeps (~5-15s on flagship phone).
     */
    private fun jacobiEigen(
        matrix: DoubleArray,
        n: Int,
        onProgress: ((String) -> Unit)? = null,
    ): Pair<DoubleArray, DoubleArray> {
        // Work on a copy
        val a = matrix.copyOf()

        // Eigenvector matrix, starts as identity
        val v = DoubleArray(n * n)
        for (i in 0 until n) v[i * n + i] = 1.0

        val maxSweeps = 50
        val eps = 1e-10

        for (sweep in 0 until maxSweeps) {
            // Compute sum of squared off-diagonal elements
            var offDiagSum = 0.0
            for (i in 0 until n) {
                for (j in i + 1 until n) {
                    offDiagSum += a[i * n + j] * a[i * n + j]
                }
            }

            if (offDiagSum < eps) {
                onProgress?.invoke("Jacobi converged after $sweep sweeps")
                break
            }

            // Threshold: decreasing with sweeps for faster convergence
            val threshold = if (sweep < 3) 0.2 * offDiagSum / (n * n) else 0.0

            // Sweep through all upper-triangle pairs
            for (p in 0 until n - 1) {
                for (q in p + 1 until n) {
                    val apq = a[p * n + q]
                    if (abs(apq) <= threshold) continue

                    val app = a[p * n + p]
                    val aqq = a[q * n + q]
                    val diff = aqq - app

                    val t: Double = if (abs(apq) < eps * abs(diff)) {
                        apq / diff
                    } else {
                        val phi = diff / (2.0 * apq)
                        val signPhi = if (phi >= 0.0) 1.0 else -1.0
                        signPhi / (abs(phi) + sqrt(1.0 + phi * phi))
                    }

                    val c = 1.0 / sqrt(1.0 + t * t)
                    val s = t * c
                    val tau = s / (1.0 + c)

                    // Update matrix A
                    a[p * n + p] -= t * apq
                    a[q * n + q] += t * apq
                    a[p * n + q] = 0.0
                    a[q * n + p] = 0.0

                    // Update off-diagonal elements
                    for (r in 0 until n) {
                        if (r == p || r == q) continue
                        val arp = a[r * n + p]
                        val arq = a[r * n + q]
                        a[r * n + p] = arp - s * (arq + tau * arp)
                        a[p * n + r] = a[r * n + p]
                        a[r * n + q] = arq + s * (arp - tau * arq)
                        a[q * n + r] = a[r * n + q]
                    }

                    // Accumulate eigenvectors
                    for (r in 0 until n) {
                        val vrp = v[r * n + p]
                        val vrq = v[r * n + q]
                        v[r * n + p] = vrp - s * (vrq + tau * vrp)
                        v[r * n + q] = vrq + s * (vrp - tau * vrq)
                    }
                }
            }

            if ((sweep + 1) % 5 == 0) {
                onProgress?.invoke("Jacobi sweep ${sweep + 1}, off-diag=${"%.2e".format(sqrt(offDiagSum))}")
            }
        }

        // Extract eigenvalues from diagonal
        val eigenvalues = DoubleArray(n) { a[it * n + it] }

        // Sort by descending eigenvalue (and reorder eigenvectors)
        val indices = (0 until n).sortedByDescending { eigenvalues[it] }.toIntArray()
        val sortedEigenvalues = DoubleArray(n) { eigenvalues[indices[it]] }
        val sortedEigenvectors = DoubleArray(n * n)
        for (col in 0 until n) {
            val srcCol = indices[col]
            for (row in 0 until n) {
                sortedEigenvectors[row * n + col] = v[row * n + srcCol]
            }
        }

        return Pair(sortedEigenvalues, sortedEigenvectors)
    }

    // --- k-means clustering ---

    /**
     * Cosine-distance k-means with centroid re-normalization.
     * Uses NEON-accelerated native math for the assignment step.
     * Matches the desktop implementation in fusion.py.
     *
     * @param flatEmbeddings Contiguous [n × d] float array, row-major
     */
    private fun kmeans(
        flatEmbeddings: FloatArray,
        n: Int,
        d: Int,
        k: Int,
        maxIter: Int = 100,
        onProgress: ((String) -> Unit)? = null,
    ): Pair<IntArray, Array<FloatArray>> {

        // k-means++ initialization with incremental min-distance tracking.
        // Uses native batchDot for distance computation.
        val rng = java.util.Random(42)
        val centroids = Array(k) { FloatArray(d) }

        // First centroid: random
        val firstIdx = rng.nextInt(n)
        System.arraycopy(flatEmbeddings, firstIdx * d, centroids[0], 0, d)

        // Track minimum squared distance from each point to its nearest centroid
        val minDistSq = FloatArray(n) { Float.MAX_VALUE }

        for (ci in 1 until k) {
            if (ci % 50 == 0) onProgress?.invoke("k-means++ init: $ci/$k centroids")

            // Update minDistSq with distance to the just-added centroid (ci-1)
            val prev = centroids[ci - 1]
            val sims = NativeMath.batchDot(prev, flatEmbeddings, n, d)
            if (sims != null) {
                for (i in 0 until n) {
                    val distSq = maxOf(1f - sims[i], 0f).let { it * it }
                    if (distSq < minDistSq[i]) minDistSq[i] = distSq
                }
            }

            // Weighted random selection using accumulated minDistSq
            var total = 0.0
            for (i in 0 until n) total += minDistSq[i]
            if (total > 0) {
                var r = rng.nextDouble() * total
                var selected = n - 1
                for (i in 0 until n) {
                    r -= minDistSq[i]
                    if (r <= 0) { selected = i; break }
                }
                System.arraycopy(flatEmbeddings, selected * d, centroids[ci], 0, d)
            } else {
                val idx = rng.nextInt(n)
                System.arraycopy(flatEmbeddings, idx * d, centroids[ci], 0, d)
            }
        }
        onProgress?.invoke("k-means++ init complete")

        // Flatten centroids for native assignment call
        val flatCentroids = FloatArray(k * d)

        // Iterate with early stopping when < 0.1% of points change
        var labels = IntArray(n)
        val convergeThreshold = maxOf(n / 1000, 1)

        for (iter in 0 until maxIter) {
            // Flatten current centroids
            for (j in 0 until k) {
                centroids[j].copyInto(flatCentroids, j * d)
            }

            // NEON-accelerated assignment: n × k dot products in native code
            val newLabels = NativeMath.kmeansAssign(flatEmbeddings, n, flatCentroids, k, d)

            var changed = 0
            if (newLabels != null) {
                for (i in 0 until n) {
                    if (labels[i] != newLabels[i]) changed++
                }
                labels = newLabels
            } else {
                // Fallback to Kotlin (shouldn't happen)
                for (i in 0 until n) {
                    val emb = flatEmbeddings
                    val offset = i * d
                    var bestK = 0
                    var bestSim = Float.NEGATIVE_INFINITY
                    for (j in 0 until k) {
                        var sim = 0f
                        val cOff = j * d
                        for (di in 0 until d) sim += emb[offset + di] * flatCentroids[cOff + di]
                        if (sim > bestSim) { bestSim = sim; bestK = j }
                    }
                    if (labels[i] != bestK) { labels[i] = bestK; changed++ }
                }
            }

            if (iter % 10 == 0 || changed <= convergeThreshold) {
                onProgress?.invoke("k-means iter $iter: $changed reassignments" +
                    if (changed <= convergeThreshold) " (converged)" else "")
            }
            if (changed <= convergeThreshold) break

            // Update centroids: mean of assigned points, then re-normalize
            for (j in 0 until k) {
                centroids[j].fill(0f)
            }
            val counts = IntArray(k)
            for (i in 0 until n) {
                val c = labels[i]
                counts[c]++
                val offset = i * d
                for (di in 0 until d) {
                    centroids[c][di] += flatEmbeddings[offset + di]
                }
            }
            for (j in 0 until k) {
                if (counts[j] > 0) {
                    for (di in 0 until d) {
                        centroids[j][di] /= counts[j]
                    }
                    l2Normalize(centroids[j])
                }
            }
        }

        return Pair(labels, centroids)
    }

    // --- kNN graph ---

    /**
     * Build kNN graph using cluster-accelerated search with NEON dot products.
     *
     * For each track, only searches tracks in the top-C nearest clusters instead of
     * all N tracks. With 200 clusters and C=10, this searches ~5% of the corpus per
     * query, giving ~20x speedup over brute force.
     *
     * Matches the graph format from GraphUpdater / desktop fusion.py.
     *
     * @param flatEmbeddings Contiguous [n × d] float array, row-major
     */
    private fun buildKnnGraph(
        flatEmbeddings: FloatArray,
        n: Int,
        d: Int,
        trackIds: LongArray,
        labels: IntArray,
        centroids: Array<FloatArray>,
        onProgress: ((String) -> Unit)? = null,
    ) {
        val k = knnK
        val numClusters = centroids.size
        val searchClusters = minOf(10, numClusters)
        val neighbors = Array(n) { IntArray(k) }
        val weights = Array(n) { FloatArray(k) }

        // Build cluster → member indices map
        val clusterMembers = Array(numClusters) { mutableListOf<Int>() }
        for (i in 0 until n) clusterMembers[labels[i]].add(i)

        // Convert cluster members to IntArrays and build flat candidate arrays per cluster
        // for native batch dot product calls
        val clusterMemberArrays = Array(numClusters) { c -> clusterMembers[c].toIntArray() }
        val clusterFlatEmbeddings = Array(numClusters) { c ->
            val members = clusterMemberArrays[c]
            val flat = FloatArray(members.size * d)
            for ((idx, memberIdx) in members.withIndex()) {
                System.arraycopy(flatEmbeddings, memberIdx * d, flat, idx * d, d)
            }
            flat
        }

        // Flatten centroids for batch dot
        val flatCentroids = FloatArray(numClusters * d)
        for (c in 0 until numClusters) {
            centroids[c].copyInto(flatCentroids, c * d)
        }

        val queryBuf = FloatArray(d)

        for (i in 0 until n) {
            if (i % 1000 == 0) {
                onProgress?.invoke("kNN: $i/$n")
            }

            // Extract query embedding
            System.arraycopy(flatEmbeddings, i * d, queryBuf, 0, d)

            // Find nearest clusters using native batch dot
            val clusterSims = NativeMath.batchDot(queryBuf, flatCentroids, numClusters, d)
                ?: FloatArray(numClusters) { c -> dotProduct(queryBuf, centroids[c]) }
            val topClusters = clusterSims.indices
                .sortedByDescending { clusterSims[it] }
                .take(searchClusters)

            // Search candidates from top clusters using min-heap of size K
            val heap = java.util.PriorityQueue<IntArray>(k + 1,
                compareBy { java.lang.Float.intBitsToFloat(it[1]) })

            for (c in topClusters) {
                val members = clusterMemberArrays[c]
                val clusterFlat = clusterFlatEmbeddings[c]
                // Native batch dot: query vs all members of this cluster
                val sims = NativeMath.batchDot(queryBuf, clusterFlat, members.size, d)

                if (sims != null) {
                    for (idx in members.indices) {
                        val j = members[idx]
                        if (j == i) continue
                        val sim = sims[idx]
                        if (heap.size < k) {
                            heap.add(intArrayOf(j, java.lang.Float.floatToRawIntBits(sim)))
                        } else {
                            val minSim = java.lang.Float.intBitsToFloat(heap.peek()!![1])
                            if (sim > minSim) {
                                heap.poll()
                                heap.add(intArrayOf(j, java.lang.Float.floatToRawIntBits(sim)))
                            }
                        }
                    }
                }
            }

            // Extract sorted results
            val sorted = heap.sortedByDescending { java.lang.Float.intBitsToFloat(it[1]) }
            for (j in sorted.indices) {
                neighbors[i][j] = sorted[j][0]
                weights[i][j] = maxOf(java.lang.Float.intBitsToFloat(sorted[j][1]), 0f)
            }
            for (j in sorted.size until k) {
                neighbors[i][j] = 0
                weights[i][j] = 0f
            }

            // Row-normalize to transition probabilities
            var total = 0f
            for (j in 0 until k) total += weights[i][j]
            if (total > 0f) {
                for (j in 0 until k) weights[i][j] /= total
            }
        }

        // Free per-cluster flat embedding copies (~153MB for 75K tracks) before
        // allocating the graph binary ByteBuffer. Without this, OOM is inevitable
        // since both flatEmbeddings (153MB) and clusterFlatEmbeddings (153MB) are alive.
        for (i in clusterFlatEmbeddings.indices) {
            clusterFlatEmbeddings[i] = FloatArray(0)
        }
        System.gc()

        // Build binary blob
        onProgress?.invoke("Writing graph binary...")
        val size = 8 + n * 8 + n * k * 8
        val buffer = ByteBuffer.allocate(size).order(ByteOrder.LITTLE_ENDIAN)
        buffer.putInt(n)
        buffer.putInt(k)
        for (tid in trackIds) buffer.putLong(tid)
        for (i in 0 until n) {
            for (j in 0 until k) {
                buffer.putInt(neighbors[i][j])
                buffer.putFloat(weights[i][j])
            }
        }
        val graphBlob = buffer.array()

        // Store in DB
        db.setBinaryData("knn_graph", graphBlob)

        // Also write to file
        File(filesDir, "graph.bin").writeBytes(graphBlob)

        val sizeMB = graphBlob.size / 1024 / 1024
        onProgress?.invoke("Graph built: $n nodes, K=$k, $sizeMB MB")
    }

    private fun dotProduct(a: FloatArray, b: FloatArray): Float {
        var sum = 0f
        for (i in a.indices) sum += a[i] * b[i]
        return sum
    }
}
