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
 * Supports three tiers of post-indexing update:
 *
 * - **quickUpdate**: Load existing SVD, project new+old tracks, assign to existing clusters.
 *   ~30s on Snapdragon 8 Gen 3 (~75K tracks).
 *
 * - **recluster**: Load existing SVD, project all tracks, full k-means from scratch.
 *   ~2 min.
 *
 * - **fullRefusion**: Recompute SVD from covariance matrix, project, cluster, rebuild.
 *   ~4 min (without kNN).
 *
 * Each tier can optionally build the kNN graph (+4 min), required only for Random Walk mode.
 *
 * Progress callback: `(phase: String, detail: String, overallFraction: Float) -> Unit`
 * - phase: Human-readable name of current sub-step (e.g. "Projecting tracks")
 * - detail: Specific status (e.g. "5000/74726")
 * - overallFraction: [0, 1] across the entire operation, for progress bar + ETA
 *
 * Uses NEON-accelerated native math (NativeMath JNI) for dot products in
 * k-means and kNN, giving ~6-10x speedup over scalar Kotlin loops.
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

        // Phase weights based on real timing (Snapdragon 8 Gen 3, 75K tracks).
        // Values are approximate seconds; only their ratios matter.
        private const val W_COVARIANCE = 22f
        private const val W_EIGEN = 85f
        private const val W_PROJECTION = 18f
        private const val W_KMEANS_FULL = 72f
        private const val W_KMEANS_ASSIGN = 3f
        private const val W_KNN = 225f
        private const val W_EXTRACT = 8f
    }

    // Shared state populated by setup()
    private lateinit var allTrackIds: LongArray
    private lateinit var mulanSet: HashSet<Long>
    private lateinit var flamingoSet: HashSet<Long>
    private var sourceDim: Int = 0
    private var concatDim: Int = 0

    /**
     * Tracks progress across sequential phases with relative weights.
     * Each phase reports sub-fraction [0,1]; this maps it to an overall fraction.
     */
    private class PhaseTracker(
        private val phases: List<Pair<String, Float>>,
        private val onProgress: ((String, String, Float) -> Unit)?,
    ) {
        private val totalWeight = phases.sumOf { it.second.toDouble() }.toFloat()
        private var phaseIdx = 0
        private var completedWeight = 0f

        /** Get a callback for the current phase. */
        fun current(): (String, Float) -> Unit = { detail, sub ->
            val weight = if (phaseIdx < phases.size) phases[phaseIdx].second else 0f
            val frac = (completedWeight + weight * sub.coerceIn(0f, 1f)) / totalWeight
            val name = if (phaseIdx < phases.size) phases[phaseIdx].first else ""
            onProgress?.invoke(name, detail, frac.coerceIn(0f, 1f))
        }

        /** Mark current phase complete and advance to the next. */
        fun advance() {
            if (phaseIdx < phases.size) {
                completedWeight += phases[phaseIdx].second
                phaseIdx++
            }
        }
    }

    // ── Orchestrators ────────────────────────────────────────────────

    /**
     * Quick update: project all tracks through existing SVD, assign to existing clusters.
     * Fastest option (~30s) — suitable when adding a few tracks to a large desktop-fused corpus.
     */
    fun quickUpdate(
        buildGraph: Boolean,
        onProgress: ((phase: String, detail: String, fraction: Float) -> Unit)? = null,
    ) {
        val start = System.currentTimeMillis()
        setup()

        val projection = loadExistingSvd()
            ?: throw IllegalStateException("No existing SVD matrix — use fullRefusion instead")

        val phases = buildList {
            add("Projecting tracks" to W_PROJECTION)
            add("Assigning clusters" to W_KMEANS_ASSIGN)
            if (buildGraph) add("Building kNN graph" to W_KNN)
            add("Extracting indices" to W_EXTRACT)
        }
        val tracker = PhaseTracker(phases, onProgress)

        projectAllTracks(projection, tracker.current())
        tracker.advance()

        clusterAssignOnly(tracker.current())
        tracker.advance()

        if (buildGraph) {
            buildKnnGraphPhase(tracker.current())
            tracker.advance()
        }

        extractIndices(buildGraph, tracker.current())
        tracker.advance()

        Log.i(TAG, "TIMING: quick_update_total = ${System.currentTimeMillis() - start}ms")
    }

    /**
     * Re-cluster: project all tracks through existing SVD, full k-means from scratch.
     * ~2 min — use when cluster assignments have drifted after many incremental updates.
     */
    fun recluster(
        buildGraph: Boolean,
        onProgress: ((phase: String, detail: String, fraction: Float) -> Unit)? = null,
    ) {
        val start = System.currentTimeMillis()
        setup()

        val projection = loadExistingSvd()
            ?: throw IllegalStateException("No existing SVD matrix — use fullRefusion instead")

        val phases = buildList {
            add("Projecting tracks" to W_PROJECTION)
            add("Clustering" to W_KMEANS_FULL)
            if (buildGraph) add("Building kNN graph" to W_KNN)
            add("Extracting indices" to W_EXTRACT)
        }
        val tracker = PhaseTracker(phases, onProgress)

        projectAllTracks(projection, tracker.current())
        tracker.advance()

        clusterFull(tracker.current())
        tracker.advance()

        if (buildGraph) {
            buildKnnGraphPhase(tracker.current())
            tracker.advance()
        }

        extractIndices(buildGraph, tracker.current())
        tracker.advance()

        Log.i(TAG, "TIMING: recluster_total = ${System.currentTimeMillis() - start}ms")
    }

    /**
     * Full re-fusion: SVD + project + cluster + optional kNN graph.
     * ~4 min without kNN — use for first-time on-device fusion or when corpus changes significantly.
     */
    fun fullRefusion(
        buildGraph: Boolean,
        onProgress: ((phase: String, detail: String, fraction: Float) -> Unit)? = null,
    ) {
        val start = System.currentTimeMillis()
        setup()

        val phases = buildList {
            add("Computing covariance" to W_COVARIANCE)
            add("Eigendecomposition" to W_EIGEN)
            add("Projecting tracks" to W_PROJECTION)
            add("Clustering" to W_KMEANS_FULL)
            if (buildGraph) add("Building kNN graph" to W_KNN)
            add("Extracting indices" to W_EXTRACT)
        }
        val tracker = PhaseTracker(phases, onProgress)

        val covariance = computeCovariance(tracker.current())
        tracker.advance()

        val projection = computeEigen(covariance, tracker.current())
        tracker.advance()

        projectAllTracks(projection, tracker.current())
        tracker.advance()

        clusterFull(tracker.current())
        tracker.advance()

        if (buildGraph) {
            buildKnnGraphPhase(tracker.current())
            tracker.advance()
        }

        extractIndices(buildGraph, tracker.current())
        tracker.advance()

        Log.i(TAG, "TIMING: fusion_total = ${System.currentTimeMillis() - start}ms")
    }

    /**
     * Legacy entry point — equivalent to fullRefusion(buildGraph = true).
     */
    fun recomputeFusion(onProgress: ((String) -> Unit)? = null) {
        fullRefusion(buildGraph = true) { _, detail, _ ->
            onProgress?.invoke(detail)
        }
    }

    // ── Phase methods ────────────────────────────────────────────────
    // Each phase method accepts (detail: String, subFraction: Float) -> Unit
    // where subFraction is [0, 1] within this phase only.

    /**
     * Load track IDs, detect dimensions, populate shared state.
     */
    private fun setup() {
        Log.i(TAG, "Loading embeddings for fusion...")
        sourceDim = detectSourceDim()
        concatDim = sourceDim * 2

        if (targetDim > concatDim) {
            throw IllegalStateException("Target dim ($targetDim) > concat dim ($concatDim)")
        }

        val mulanTrackIds = getTrackIdsForTable("embeddings_mulan")
        val flamingoTrackIds = getTrackIdsForTable("embeddings_flamingo")
        allTrackIds = (mulanTrackIds + flamingoTrackIds).sorted().distinct().toLongArray()
        mulanSet = mulanTrackIds.toHashSet()
        flamingoSet = flamingoTrackIds.toHashSet()

        val nTracks = allTrackIds.size
        val bothCount = mulanSet.intersect(flamingoSet).size

        Log.i(TAG, "Tracks: $nTracks total ($bothCount with both models, " +
            "${mulanTrackIds.size - bothCount} MuLan-only, " +
            "${flamingoTrackIds.size - bothCount} Flamingo-only)")

        if (nTracks == 0) {
            throw IllegalStateException("No embeddings to fuse")
        }
    }

    /**
     * Compute covariance matrix C = X^T X by streaming embeddings.
     * Sub-progress: linear with tracks processed.
     */
    private fun computeCovariance(
        onProgress: ((String, Float) -> Unit)? = null,
    ): DoubleArray {
        val nTracks = allTrackIds.size
        val covStart = System.currentTimeMillis()
        onProgress?.invoke("$nTracks tracks, ${concatDim}d...", 0f)

        val covariance = DoubleArray(concatDim * concatDim)

        val covBatchSize = 500
        var covBatch = FloatArray(covBatchSize * concatDim)
        var covBatchIdx = 0

        for ((idx, trackId) in allTrackIds.withIndex()) {
            if (idx % 5000 == 0 && idx > 0) {
                onProgress?.invoke("$idx / $nTracks tracks", idx.toFloat() / nTracks)
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
        covBatch = FloatArray(0)

        // Fill lower triangle from upper
        for (i in 0 until concatDim) {
            for (j in 0 until i) {
                covariance[i * concatDim + j] = covariance[j * concatDim + i]
            }
        }

        Log.i(TAG, "TIMING: covariance = ${System.currentTimeMillis() - covStart}ms")
        onProgress?.invoke("Covariance complete", 1f)
        return covariance
    }

    /**
     * Eigendecomposition + build projection matrix + store in DB.
     * Sub-progress: 0 at start, 1 when complete (native Jacobi is blocking).
     */
    private fun computeEigen(
        covariance: DoubleArray,
        onProgress: ((String, Float) -> Unit)? = null,
    ): FloatArray {
        val rawDb = db.getRawDatabase()
        val eigenStart = System.currentTimeMillis()
        onProgress?.invoke("${concatDim}x$concatDim matrix...", 0f)

        val nativeResult = nativeJacobiEigen(covariance, concatDim)
        val (eigenvalues, eigenvectors) = if (nativeResult != null) {
            Log.i(TAG, "Native Jacobi completed in ${System.currentTimeMillis() - eigenStart}ms")
            nativeResult
        } else {
            Log.w(TAG, "Native Jacobi failed, falling back to Kotlin")
            jacobiEigen(covariance, concatDim) { detail, sub ->
                onProgress?.invoke(detail, sub)
            }
        }

        // Projection matrix = top targetDim eigenvectors as rows (Vt convention)
        val projectionData = FloatArray(targetDim * concatDim)
        for (i in 0 until targetDim) {
            for (j in 0 until concatDim) {
                projectionData[i * concatDim + j] = eigenvectors[j * concatDim + i].toFloat()
            }
        }

        // Compute variance retained
        val totalVar = eigenvalues.sum()
        val retainedVar = eigenvalues.take(targetDim).sum() / totalVar

        Log.i(TAG, "TIMING: eigendecomposition = ${System.currentTimeMillis() - eigenStart}ms")
        Log.i(TAG, "Variance retained: ${"%.2f".format(retainedVar * 100)}%")

        // Store projection matrix and metadata
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

        onProgress?.invoke("${"%.1f".format(retainedVar * 100)}% variance retained", 1f)
        return projectionData
    }

    /**
     * Load existing SVD projection matrix from DB metadata.
     * Returns FloatArray[targetDim * concatDim] or null if not found.
     */
    private fun loadExistingSvd(): FloatArray? {
        val rawDb = db.getRawDatabase()
        val matrix = EmbeddingProcessor.loadProjectionMatrix(
            rawDb, "fused_projection", targetDim, concatDim
        ) ?: return null
        return matrix.data
    }

    /**
     * Project all tracks through SVD and write fused embeddings to DB.
     * Sub-progress: linear with tracks processed.
     */
    private fun projectAllTracks(
        projectionData: FloatArray,
        onProgress: ((String, Float) -> Unit)? = null,
    ) {
        val rawDb = db.getRawDatabase()
        val nTracks = allTrackIds.size

        val projStart = System.currentTimeMillis()
        onProgress?.invoke("$nTracks tracks to ${targetDim}d...", 0f)

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
            rawDb.execSQL("DELETE FROM embeddings_fused")

            for ((idx, trackId) in allTrackIds.withIndex()) {
                if (idx % 5000 == 0 && idx > 0) {
                    onProgress?.invoke("$idx / $nTracks", idx.toFloat() / nTracks)
                }

                val concat = getConcatenatedEmbedding(trackId, sourceDim, mulanSet, flamingoSet)
                val fused = NativeMath.matVecMul(projectionData, targetDim, concatDim, concat)
                    ?: FloatMatrix(projectionData, targetDim, concatDim).multiplyVector(concat)
                l2Normalize(fused)
                db.insertEmbedding("embeddings_fused", trackId, fused)
            }
            rawDb.setTransactionSuccessful()
        } finally {
            rawDb.endTransaction()
        }

        Log.i(TAG, "TIMING: projection = ${System.currentTimeMillis() - projStart}ms")
        onProgress?.invoke("$nTracks tracks projected", 1f)
    }

    /**
     * Full k-means clustering from scratch.
     * Sub-progress: delegates to kmeans (init 0-0.2, iterations 0.2-1.0).
     */
    private fun clusterFull(onProgress: ((String, Float) -> Unit)? = null) {
        val rawDb = db.getRawDatabase()
        val nTracks = allTrackIds.size

        val kmeansStart = System.currentTimeMillis()
        onProgress?.invoke("Loading embeddings...", 0f)
        val flatEmbeddings = loadAllFusedEmbeddingsFlat(allTrackIds)

        val actualClusters = minOf(nClusters, nTracks)
        val (labels, centroids) = kmeans(flatEmbeddings, nTracks, targetDim,
            actualClusters, onProgress = onProgress)

        onProgress?.invoke("Writing clusters...", 0.98f)
        writeClusters(rawDb, labels, centroids, actualClusters)

        Log.i(TAG, "TIMING: kmeans = ${System.currentTimeMillis() - kmeansStart}ms")
        onProgress?.invoke("$actualClusters clusters", 1f)
    }

    /**
     * Assign-only clustering: load existing centroids, single assignment pass.
     * Falls back to full k-means if no centroids exist.
     * Sub-progress: 0 → 1 (near-instant, ~3s).
     */
    private fun clusterAssignOnly(onProgress: ((String, Float) -> Unit)? = null) {
        val rawDb = db.getRawDatabase()
        val nTracks = allTrackIds.size

        val existingCentroids = db.loadCentroids()
        if (existingCentroids.isEmpty()) {
            Log.i(TAG, "No existing centroids — falling back to full k-means")
            clusterFull(onProgress)
            return
        }

        val assignStart = System.currentTimeMillis()
        val k = existingCentroids.size
        val d = targetDim
        onProgress?.invoke("$nTracks tracks to $k clusters...", 0f)

        val flatEmbeddings = loadAllFusedEmbeddingsFlat(allTrackIds)

        // Flatten centroids for native assignment
        val flatCentroids = FloatArray(k * d)
        for ((clusterId, centroid) in existingCentroids) {
            if (clusterId < k) {
                centroid.copyInto(flatCentroids, clusterId * d, 0, minOf(centroid.size, d))
            }
        }

        val labels = NativeMath.kmeansAssign(flatEmbeddings, nTracks, flatCentroids, k, d)
            ?: IntArray(nTracks) // fallback: all cluster 0

        // Reconstruct centroids array from the map (ordered by cluster_id)
        val centroidsArray = Array(k) { cid ->
            existingCentroids[cid] ?: FloatArray(d)
        }

        writeClusters(rawDb, labels, centroidsArray, k)

        Log.i(TAG, "TIMING: cluster_assign = ${System.currentTimeMillis() - assignStart}ms")
        onProgress?.invoke("$nTracks tracks assigned", 1f)
    }

    /**
     * Build kNN graph from fused embeddings + cluster assignments.
     * Sub-progress: linear with tracks processed (i/n).
     */
    private fun buildKnnGraphPhase(onProgress: ((String, Float) -> Unit)? = null) {
        val nTracks = allTrackIds.size

        val knnStart = System.currentTimeMillis()
        onProgress?.invoke("Loading embeddings...", 0f)

        val flatEmbeddings = loadAllFusedEmbeddingsFlat(allTrackIds)

        // Load cluster data for graph build
        val rawDb = db.getRawDatabase()
        val labels = IntArray(nTracks)
        rawDb.rawQuery(
            "SELECT id, cluster_id FROM tracks WHERE id IN (${allTrackIds.joinToString(",")})",
            null
        ).use { cursor ->
            val idToIndex = HashMap<Long, Int>(nTracks)
            for (i in allTrackIds.indices) idToIndex[allTrackIds[i]] = i
            while (cursor.moveToNext()) {
                val trackId = cursor.getLong(0)
                val clusterId = if (cursor.isNull(1)) 0 else cursor.getInt(1)
                idToIndex[trackId]?.let { labels[it] = clusterId }
            }
        }

        val centroidsMap = db.loadCentroids()
        val numClusters = centroidsMap.size
        val centroids = Array(numClusters) { cid ->
            centroidsMap[cid] ?: FloatArray(targetDim)
        }

        buildKnnGraph(flatEmbeddings, nTracks, targetDim,
            allTrackIds, labels, centroids, onProgress)

        Log.i(TAG, "TIMING: knn_graph = ${System.currentTimeMillis() - knnStart}ms")
    }

    /**
     * Extract .emb index and optionally graph.bin from database.
     * Sub-progress: linear with embeddings extracted.
     */
    private fun extractIndices(
        includeGraph: Boolean,
        onProgress: ((String, Float) -> Unit)? = null,
    ) {
        val extractStart = System.currentTimeMillis()
        onProgress?.invoke("Extracting index files...", 0f)

        var totalTracks = 0
        EmbeddingIndex.extractFromDatabase(db, File(filesDir, "fused.emb")) { cur, total ->
            totalTracks = total
            if (cur % 10000 == 0) {
                onProgress?.invoke("$cur / $total embeddings", cur.toFloat() / total)
            }
        }
        if (includeGraph) {
            GraphIndex.extractFromDatabase(db, File(filesDir, "graph.bin"))
        }

        Log.i(TAG, "TIMING: extract_indices = ${System.currentTimeMillis() - extractStart}ms")
        onProgress?.invoke("$totalTracks embeddings extracted", 1f)
    }

    // ── Helpers ──────────────────────────────────────────────────────

    /**
     * Write cluster labels and centroids to DB.
     */
    private fun writeClusters(
        rawDb: android.database.sqlite.SQLiteDatabase,
        labels: IntArray,
        centroids: Array<FloatArray>,
        k: Int,
    ) {
        rawDb.execSQL("""
            CREATE TABLE IF NOT EXISTS clusters (
                cluster_id INTEGER PRIMARY KEY,
                embedding BLOB NOT NULL
            )
        """)
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
            for (cid in 0 until k) {
                val blob = EmbeddingDatabase.floatArrayToBlob(centroids[cid])
                rawDb.execSQL(
                    "INSERT INTO clusters (cluster_id, embedding) VALUES (?, ?)",
                    arrayOf<Any>(cid, blob)
                )
            }
            rawDb.setTransactionSuccessful()
        } finally {
            rawDb.endTransaction()
        }
    }

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

    private fun nativeJacobiEigen(
        matrix: DoubleArray, n: Int,
    ): Pair<DoubleArray, DoubleArray>? {
        val result = NativeMath.jacobiEigen(matrix, n) ?: return null
        if (result.size != n + n * n) return null
        val eigenvalues = result.copyOfRange(0, n)
        val eigenvectors = result.copyOfRange(n, n + n * n)
        return Pair(eigenvalues, eigenvectors)
    }

    // --- Jacobi eigenvalue decomposition ---

    private fun jacobiEigen(
        matrix: DoubleArray,
        n: Int,
        onProgress: ((String, Float) -> Unit)? = null,
    ): Pair<DoubleArray, DoubleArray> {
        val a = matrix.copyOf()
        val v = DoubleArray(n * n)
        for (i in 0 until n) v[i * n + i] = 1.0

        val maxSweeps = 50
        val eps = 1e-10

        for (sweep in 0 until maxSweeps) {
            var offDiagSum = 0.0
            for (i in 0 until n) {
                for (j in i + 1 until n) {
                    offDiagSum += a[i * n + j] * a[i * n + j]
                }
            }

            if (offDiagSum < eps) {
                onProgress?.invoke("Converged after $sweep sweeps",
                    (sweep.toFloat() / maxSweeps).coerceAtMost(1f))
                break
            }

            val threshold = if (sweep < 3) 0.2 * offDiagSum / (n * n) else 0.0

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

                    a[p * n + p] -= t * apq
                    a[q * n + q] += t * apq
                    a[p * n + q] = 0.0
                    a[q * n + p] = 0.0

                    for (r in 0 until n) {
                        if (r == p || r == q) continue
                        val arp = a[r * n + p]
                        val arq = a[r * n + q]
                        a[r * n + p] = arp - s * (arq + tau * arp)
                        a[p * n + r] = a[r * n + p]
                        a[r * n + q] = arq + s * (arp - tau * arq)
                        a[q * n + r] = a[r * n + q]
                    }

                    for (r in 0 until n) {
                        val vrp = v[r * n + p]
                        val vrq = v[r * n + q]
                        v[r * n + p] = vrp - s * (vrq + tau * vrp)
                        v[r * n + q] = vrq + s * (vrp - tau * vrq)
                    }
                }
            }

            if ((sweep + 1) % 5 == 0) {
                onProgress?.invoke("Sweep ${sweep + 1}, off-diag=${"%.2e".format(sqrt(offDiagSum))}",
                    (sweep + 1).toFloat() / maxSweeps)
            }
        }

        val eigenvalues = DoubleArray(n) { a[it * n + it] }
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
     * Sub-progress: init 0→0.2, iterations 0.2→1.0.
     */
    private fun kmeans(
        flatEmbeddings: FloatArray,
        n: Int,
        d: Int,
        k: Int,
        maxIter: Int = 100,
        onProgress: ((String, Float) -> Unit)? = null,
    ): Pair<IntArray, Array<FloatArray>> {

        val rng = java.util.Random(42)
        val centroids = Array(k) { FloatArray(d) }

        val firstIdx = rng.nextInt(n)
        System.arraycopy(flatEmbeddings, firstIdx * d, centroids[0], 0, d)

        val minDistSq = FloatArray(n) { Float.MAX_VALUE }

        for (ci in 1 until k) {
            if (ci % 50 == 0) {
                onProgress?.invoke("k-means++ init: $ci/$k centroids",
                    ci.toFloat() / k * 0.2f)
            }

            val prev = centroids[ci - 1]
            val sims = NativeMath.batchDot(prev, flatEmbeddings, n, d)
            if (sims != null) {
                for (i in 0 until n) {
                    val distSq = maxOf(1f - sims[i], 0f).let { it * it }
                    if (distSq < minDistSq[i]) minDistSq[i] = distSq
                }
            }

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
        onProgress?.invoke("k-means++ init complete", 0.2f)

        val flatCentroids = FloatArray(k * d)

        var labels = IntArray(n)
        val convergeThreshold = maxOf(n / 200, 1)
        // Estimate ~40 iterations for ETA (typical convergence at ~34 iters)
        val estimatedIters = 40f

        for (iter in 0 until maxIter) {
            for (j in 0 until k) {
                centroids[j].copyInto(flatCentroids, j * d)
            }

            val newLabels = NativeMath.kmeansAssign(flatEmbeddings, n, flatCentroids, k, d)

            var changed = 0
            if (newLabels != null) {
                for (i in 0 until n) {
                    if (labels[i] != newLabels[i]) changed++
                }
                labels = newLabels
            } else {
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

            val iterFrac = 0.2f + ((iter + 1).toFloat() / estimatedIters).coerceAtMost(1f) * 0.8f

            if (iter % 10 == 0 || changed <= convergeThreshold) {
                val status = "Iter $iter: $changed reassignments" +
                    if (changed <= convergeThreshold) " (converged)" else ""
                onProgress?.invoke(status, if (changed <= convergeThreshold) 1f else iterFrac)
            }
            if (changed <= convergeThreshold) break

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
     * Sub-progress: linear with tracks processed (i/n).
     */
    private fun buildKnnGraph(
        flatEmbeddings: FloatArray,
        n: Int,
        d: Int,
        trackIds: LongArray,
        labels: IntArray,
        centroids: Array<FloatArray>,
        onProgress: ((String, Float) -> Unit)? = null,
    ) {
        val k = knnK
        val numClusters = centroids.size
        val searchClusters = minOf(10, numClusters)
        val neighbors = Array(n) { IntArray(k) }
        val weights = Array(n) { FloatArray(k) }

        val clusterMembers = Array(numClusters) { mutableListOf<Int>() }
        for (i in 0 until n) clusterMembers[labels[i]].add(i)

        val clusterMemberArrays = Array(numClusters) { c -> clusterMembers[c].toIntArray() }

        val maxClusterSize = clusterMemberArrays.maxOf { it.size }
        var clusterBuf = FloatArray(maxClusterSize * d)
        Log.i(TAG, "kNN: max cluster size = $maxClusterSize, " +
            "buffer = ${clusterBuf.size * 4 / 1024}KB")

        val flatCentroids = FloatArray(numClusters * d)
        for (c in 0 until numClusters) {
            centroids[c].copyInto(flatCentroids, c * d)
        }

        val queryBuf = FloatArray(d)

        for (i in 0 until n) {
            if (i % 1000 == 0) {
                onProgress?.invoke("$i / $n nodes", i.toFloat() / n)
            }

            System.arraycopy(flatEmbeddings, i * d, queryBuf, 0, d)

            val clusterSims = NativeMath.batchDot(queryBuf, flatCentroids, numClusters, d)
                ?: FloatArray(numClusters) { c -> dotProduct(queryBuf, centroids[c]) }
            val topClusters = clusterSims.indices
                .sortedByDescending { clusterSims[it] }
                .take(searchClusters)

            val heap = java.util.PriorityQueue<IntArray>(k + 1,
                compareBy { java.lang.Float.intBitsToFloat(it[1]) })

            for (c in topClusters) {
                val members = clusterMemberArrays[c]
                for ((idx, memberIdx) in members.withIndex()) {
                    System.arraycopy(flatEmbeddings, memberIdx * d, clusterBuf, idx * d, d)
                }
                val sims = NativeMath.batchDot(queryBuf, clusterBuf, members.size, d)

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

            val sorted = heap.sortedByDescending { java.lang.Float.intBitsToFloat(it[1]) }
            for (j in sorted.indices) {
                neighbors[i][j] = sorted[j][0]
                weights[i][j] = maxOf(java.lang.Float.intBitsToFloat(sorted[j][1]), 0f)
            }
            for (j in sorted.size until k) {
                neighbors[i][j] = 0
                weights[i][j] = 0f
            }

            var total = 0f
            for (j in 0 until k) total += weights[i][j]
            if (total > 0f) {
                for (j in 0 until k) weights[i][j] /= total
            }
        }

        @Suppress("UNUSED_VALUE")
        clusterBuf = FloatArray(0)

        onProgress?.invoke("Writing graph binary...", 0.98f)
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

        db.setBinaryData("knn_graph", graphBlob)
        File(filesDir, "graph.bin").writeBytes(graphBlob)

        val sizeMB = graphBlob.size / 1024 / 1024
        onProgress?.invoke("$n nodes, K=$k, $sizeMB MB", 1f)
    }

    private fun dotProduct(a: FloatArray, b: FloatArray): Float {
        var sum = 0f
        for (i in a.indices) sum += a[i] * b[i]
        return sum
    }
}
