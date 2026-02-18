package com.powerampstartradio.indexing

import android.util.Log
import com.powerampstartradio.data.EmbeddingDatabase
import java.io.File

/**
 * Writes new track embeddings to the database, applying projection matrices
 * and assigning clusters.
 *
 * Handles the full write pipeline:
 * 1. Insert track metadata
 * 2. Insert MuQ-MuLan embedding (512d)
 * 3. Insert Flamingo embedding (3584d raw, or 512d reduced)
 * 4. Insert fused embedding (512d)
 * 5. Assign nearest cluster
 */
class EmbeddingWriter(
    private val db: EmbeddingDatabase,
    private val centroids: Map<Int, FloatArray>?,
) {
    companion object {
        private const val TAG = "EmbeddingWriter"
    }

    /**
     * Write all embeddings for a single track.
     *
     * @param metadataKey Desktop-format metadata key "artist|album|title|duration_rounded"
     * @param filenameKey Filename-based key for fallback matching
     * @param artist Track artist
     * @param album Track album
     * @param title Track title
     * @param durationMs Track duration in milliseconds
     * @param filePath Canonical file path
     * @param embeddings Computed embeddings from EmbeddingProcessor
     * @return The new track ID, or -1 on failure
     */
    fun writeTrack(
        metadataKey: String,
        filenameKey: String,
        artist: String?,
        album: String?,
        title: String?,
        durationMs: Int,
        filePath: String,
        embeddings: EmbeddingProcessor.EmbeddingResult,
    ): Long {
        return try {
            val rawDb = db.getRawDatabase()
            rawDb.beginTransaction()
            try {
                // Insert track
                val trackId = db.insertTrack(
                    metadataKey, filenameKey, artist, album, title, durationMs, filePath
                )

                // Insert MuQ-MuLan embedding
                embeddings.mulanEmbedding?.let {
                    db.insertEmbedding("embeddings_mulan", trackId, it)
                }

                // Insert Flamingo embedding (store the reduced 512d version)
                embeddings.flamingoReduced?.let {
                    db.insertEmbedding("embeddings_flamingo", trackId, it)
                }

                // Insert fused embedding
                embeddings.fusedEmbedding?.let {
                    db.insertEmbedding("embeddings_fused", trackId, it)
                }

                // Assign cluster (nearest centroid by dot product)
                if (embeddings.fusedEmbedding != null && centroids != null && centroids.isNotEmpty()) {
                    val clusterId = findNearestCluster(embeddings.fusedEmbedding)
                    if (clusterId >= 0) {
                        db.updateClusterId(trackId, clusterId)
                    }
                }

                rawDb.setTransactionSuccessful()
                Log.d(TAG, "Wrote track $trackId: $artist - $title")
                trackId
            } finally {
                rawDb.endTransaction()
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to write track: $artist - $title", e)
            -1L
        }
    }

    /**
     * Add Flamingo + fused embeddings to an existing track (pass 2 of sequential loading).
     *
     * Used when MuLan was written in pass 1 and Flamingo inference runs in pass 2.
     *
     * @param trackId Existing track ID from pass 1
     * @param embeddings Computed embeddings (only flamingoReduced and fusedEmbedding are written)
     * @return true on success
     */
    fun addEmbeddings(trackId: Long, embeddings: EmbeddingProcessor.EmbeddingResult): Boolean {
        return try {
            val rawDb = db.getRawDatabase()
            rawDb.beginTransaction()
            try {
                embeddings.flamingoReduced?.let {
                    db.insertEmbedding("embeddings_flamingo", trackId, it)
                }

                embeddings.fusedEmbedding?.let {
                    db.insertEmbedding("embeddings_fused", trackId, it)
                }

                if (embeddings.fusedEmbedding != null && centroids != null && centroids.isNotEmpty()) {
                    val clusterId = findNearestCluster(embeddings.fusedEmbedding)
                    if (clusterId >= 0) {
                        db.updateClusterId(trackId, clusterId)
                    }
                }

                rawDb.setTransactionSuccessful()
                true
            } finally {
                rawDb.endTransaction()
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to add embeddings for track $trackId", e)
            false
        }
    }

    /**
     * Find nearest cluster centroid by dot product (cosine similarity for unit vectors).
     */
    private fun findNearestCluster(embedding: FloatArray): Int {
        var bestClusterId = -1
        var bestSim = Float.NEGATIVE_INFINITY

        centroids?.forEach { (clusterId, centroid) ->
            var sim = 0f
            for (i in embedding.indices) {
                sim += embedding[i] * centroid[i]
            }
            if (sim > bestSim) {
                bestSim = sim
                bestClusterId = clusterId
            }
        }

        return bestClusterId
    }
}
