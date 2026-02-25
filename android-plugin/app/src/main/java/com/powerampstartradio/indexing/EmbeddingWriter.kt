package com.powerampstartradio.indexing

import android.util.Log
import com.powerampstartradio.data.EmbeddingDatabase

/**
 * Writes CLaMP3 track embeddings to the database.
 *
 * Pipeline:
 * 1. Insert track metadata
 * 2. Insert CLaMP3 embedding (768d) into embeddings_clamp3 table
 */
class EmbeddingWriter(
    private val db: EmbeddingDatabase,
) {
    companion object {
        private const val TAG = "EmbeddingWriter"
    }

    /**
     * Write a track's CLaMP3 embedding to the database.
     *
     * @param metadataKey Desktop-format metadata key "artist|album|title|duration_rounded"
     * @param filenameKey Filename-based key for fallback matching
     * @param artist Track artist
     * @param album Track album
     * @param title Track title
     * @param durationMs Track duration in milliseconds
     * @param filePath Canonical file path
     * @param embedding 768d CLaMP3 embedding (L2-normalized)
     * @param source Origin of the embedding ("phone" for on-device, "desktop" for imported)
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
        embedding: FloatArray,
        source: String = "phone",
    ): Long {
        return try {
            val rawDb = db.getRawDatabase()
            rawDb.beginTransaction()
            try {
                val trackId = db.insertTrack(
                    metadataKey, filenameKey, artist, album, title, durationMs, filePath, source
                )

                db.insertEmbedding("embeddings_clamp3", trackId, embedding)

                rawDb.setTransactionSuccessful()
                Log.d(TAG, "Wrote track $trackId: $artist - $title (768d CLaMP3)")
                trackId
            } finally {
                rawDb.endTransaction()
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to write track: $artist - $title", e)
            -1L
        }
    }
}
