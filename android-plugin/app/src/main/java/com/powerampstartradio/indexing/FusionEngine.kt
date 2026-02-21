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
 * Memory-efficient: streams embeddings twice (covariance + projection) rather than
 * holding the full N×1024 matrix. Peak memory: ~10 MB for the 1024×1024 covariance
 * matrix + eigenvector matrix, plus ~150 MB for k-means (75K × 512d).
 *
 * Computation time on Snapdragon 8 Gen 3 (~75K tracks):
 * - Covariance matrix: ~10-20s (streaming 75K×1024)
 * - Eigendecomposition: ~5-15s (Jacobi, 1024×1024)
 * - Projection pass: ~5-10s (streaming 75K×1024 → 512)
 * - k-means: ~30-60s (75K × 200 clusters × 512d, ~15 iterations)
 * - kNN graph: ~5-10s (already implemented)
 * Total: ~1-2 minutes
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
        progress("Computing covariance matrix ($nTracks tracks, ${concatDim}d)...")
        val covariance = DoubleArray(concatDim * concatDim)
        val mulanSet = mulanTrackIds.toHashSet()
        val flamingoSet = flamingoTrackIds.toHashSet()

        for ((idx, trackId) in allTrackIds.withIndex()) {
            if (idx % 5000 == 0 && idx > 0) {
                progress("Covariance: $idx/$nTracks")
            }

            val concat = getConcatenatedEmbedding(trackId, sourceDim, mulanSet, flamingoSet)

            // Accumulate outer product: C += x * x^T (upper triangle only, exploit symmetry)
            for (i in 0 until concatDim) {
                val xi = concat[i].toDouble()
                if (xi == 0.0) continue
                val rowOffset = i * concatDim
                for (j in i until concatDim) {
                    covariance[rowOffset + j] += xi * concat[j].toDouble()
                }
            }
        }

        // Fill lower triangle from upper
        for (i in 0 until concatDim) {
            for (j in 0 until i) {
                covariance[i * concatDim + j] = covariance[j * concatDim + i]
            }
        }

        // --- Step 2: Eigendecomposition ---
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
        val projection = FloatMatrix(projectionData, targetDim, concatDim)

        // Compute variance retained
        val totalVar = eigenvalues.sum()
        val retainedVar = eigenvalues.take(targetDim).sum() / totalVar
        progress("Variance retained: ${"%.2f".format(retainedVar * 100)}%")

        // --- Step 3: Re-project all tracks ---
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
                val fused = projection.multiplyVector(concat)
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

        // --- Step 4: k-means clustering ---
        progress("Loading fused embeddings for clustering...")
        val fusedEmbeddings = loadAllFusedEmbeddings(allTrackIds)

        val actualClusters = minOf(nClusters, nTracks)
        progress("k-means clustering (K=$actualClusters)...")
        val (labels, centroids) = kmeans(fusedEmbeddings, actualClusters, onProgress = { progress(it) })

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

        // --- Step 5: kNN graph ---
        progress("Building kNN graph (K=$knnK)...")
        buildKnnGraph(fusedEmbeddings, allTrackIds, labels, centroids,
            onProgress = { progress(it) })

        // --- Step 6: Extract .emb and graph.bin files ---
        progress("Extracting index files...")
        EmbeddingIndex.extractFromDatabase(db, File(filesDir, "fused.emb")) { cur, total ->
            if (cur % 10000 == 0) progress("Extracting embeddings: $cur/$total")
        }
        GraphIndex.extractFromDatabase(db, File(filesDir, "graph.bin"))

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

    private fun loadAllFusedEmbeddings(trackIds: LongArray): Array<FloatArray> {
        return Array(trackIds.size) { i ->
            db.getEmbeddingFromTable("embeddings_fused", trackIds[i])
                ?: FloatArray(targetDim)
        }
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
     * Matches the desktop implementation in fusion.py.
     */
    private fun kmeans(
        embeddings: Array<FloatArray>,
        k: Int,
        maxIter: Int = 100,
        onProgress: ((String) -> Unit)? = null,
    ): Pair<IntArray, Array<FloatArray>> {
        val n = embeddings.size
        val d = embeddings[0].size

        // k-means++ initialization
        val rng = java.util.Random(42)
        val centroids = Array(k) { FloatArray(d) }

        // First centroid: random
        embeddings[rng.nextInt(n)].copyInto(centroids[0])

        // Remaining: probabilistic furthest-first
        for (ci in 1 until k) {
            val dists = FloatArray(n)
            for (i in 0 until n) {
                var maxSim = Float.NEGATIVE_INFINITY
                for (j in 0 until ci) {
                    val sim = dotProduct(embeddings[i], centroids[j])
                    if (sim > maxSim) maxSim = sim
                }
                val dist = maxOf(1f - maxSim, 0f)
                dists[i] = dist * dist
            }
            // Weighted random selection
            val total = dists.sum()
            if (total > 0) {
                var r = rng.nextFloat() * total
                var selected = n - 1  // fallback for floating-point rounding
                for (i in 0 until n) {
                    r -= dists[i]
                    if (r <= 0) { selected = i; break }
                }
                embeddings[selected].copyInto(centroids[ci])
            } else {
                embeddings[rng.nextInt(n)].copyInto(centroids[ci])
            }
        }

        // Iterate
        val labels = IntArray(n)
        for (iter in 0 until maxIter) {
            // Assign each point to nearest centroid
            var changed = 0
            for (i in 0 until n) {
                var bestK = 0
                var bestSim = Float.NEGATIVE_INFINITY
                for (j in 0 until k) {
                    val sim = dotProduct(embeddings[i], centroids[j])
                    if (sim > bestSim) {
                        bestSim = sim
                        bestK = j
                    }
                }
                if (labels[i] != bestK) {
                    labels[i] = bestK
                    changed++
                }
            }

            if (iter % 10 == 0 || changed == 0) {
                onProgress?.invoke("k-means iter $iter: $changed reassignments")
            }
            if (changed == 0) break

            // Update centroids: mean of assigned points, then re-normalize
            for (j in 0 until k) {
                centroids[j].fill(0f)
            }
            val counts = IntArray(k)
            for (i in 0 until n) {
                val c = labels[i]
                counts[c]++
                for (dim in 0 until d) {
                    centroids[c][dim] += embeddings[i][dim]
                }
            }
            for (j in 0 until k) {
                if (counts[j] > 0) {
                    for (dim in 0 until d) {
                        centroids[j][dim] /= counts[j]
                    }
                    l2Normalize(centroids[j])
                }
            }
        }

        return Pair(labels, centroids)
    }

    // --- kNN graph ---

    /**
     * Build kNN graph using cluster-accelerated search.
     *
     * For each track, only searches tracks in the top-C nearest clusters instead of
     * all N tracks. With 200 clusters and C=10, this searches ~5% of the corpus per
     * query, giving ~20x speedup over brute force (from ~24 min to ~1 min at 75K tracks).
     *
     * Matches the graph format from GraphUpdater / desktop fusion.py.
     */
    private fun buildKnnGraph(
        embeddings: Array<FloatArray>,
        trackIds: LongArray,
        labels: IntArray,
        centroids: Array<FloatArray>,
        onProgress: ((String) -> Unit)? = null,
    ) {
        val n = embeddings.size
        val k = knnK
        val numClusters = centroids.size
        val searchClusters = minOf(10, numClusters)
        val neighbors = Array(n) { IntArray(k) }
        val weights = Array(n) { FloatArray(k) }

        // Build cluster → member indices map
        val clusterMembers = Array(numClusters) { mutableListOf<Int>() }
        for (i in 0 until n) clusterMembers[labels[i]].add(i)

        for (i in 0 until n) {
            if (i % 1000 == 0) {
                onProgress?.invoke("kNN: $i/$n")
            }

            // Find nearest clusters to this track's embedding
            val clusterSims = FloatArray(numClusters) { c ->
                dotProduct(embeddings[i], centroids[c])
            }
            val topClusters = clusterSims.indices
                .sortedByDescending { clusterSims[it] }
                .take(searchClusters)

            // Search candidates from top clusters using min-heap of size K
            val heap = java.util.PriorityQueue<IntArray>(k + 1,
                compareBy { java.lang.Float.intBitsToFloat(it[1]) })

            for (c in topClusters) {
                for (j in clusterMembers[c]) {
                    if (j == i) continue
                    val sim = dotProduct(embeddings[i], embeddings[j])
                    // Pack index + float bits to avoid boxing
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
