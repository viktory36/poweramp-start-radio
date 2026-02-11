package com.powerampstartradio.data

import android.content.Context
import android.database.Cursor
import android.database.sqlite.SQLiteDatabase
import android.net.Uri
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Data class representing a track from the embedding database.
 */
data class EmbeddedTrack(
    val id: Long,
    val metadataKey: String,
    val filenameKey: String,
    val artist: String?,
    val album: String?,
    val title: String?,
    val durationMs: Int,
    val filePath: String
)

/**
 * Wrapper for reading the embeddings SQLite database created by the desktop indexer.
 *
 * Expects a fused embedding database with:
 * - embeddings_fused table (primary, for similarity search)
 * - clusters table (centroids)
 * - cluster_id column on tracks
 *
 * Falls back to single-model tables if fused is not available.
 */
class EmbeddingDatabase private constructor(
    private val db: SQLiteDatabase
) {
    companion object {
        /**
         * Open the database from a file path.
         */
        fun open(dbFile: File): EmbeddingDatabase {
            val db = SQLiteDatabase.openDatabase(
                dbFile.absolutePath,
                null,
                SQLiteDatabase.OPEN_READONLY
            )
            return EmbeddingDatabase(db)
        }

        /**
         * Import database from a content URI (e.g., from document picker).
         */
        fun importFrom(context: Context, uri: Uri, destFile: File): EmbeddingDatabase {
            context.contentResolver.openInputStream(uri)?.use { input ->
                FileOutputStream(destFile).use { output ->
                    input.copyTo(output)
                }
            } ?: throw IllegalArgumentException("Cannot open URI: $uri")

            return open(destFile)
        }

        /**
         * Convert a BLOB to a FloatArray.
         */
        fun blobToFloatArray(blob: ByteArray): FloatArray {
            val buffer = ByteBuffer.wrap(blob).order(ByteOrder.LITTLE_ENDIAN)
            val floats = FloatArray(blob.size / 4)
            for (i in floats.indices) {
                floats[i] = buffer.getFloat()
            }
            return floats
        }
    }

    /**
     * Whether this database has fused embeddings.
     */
    val hasFusedEmbeddings: Boolean by lazy {
        tableHasRows("embeddings_fused")
    }

    /**
     * The embedding table to use for similarity search.
     * Prefers fused, falls back to mulan, then flamingo.
     */
    val embeddingTable: String by lazy {
        val candidates = listOf("embeddings_fused", "embeddings_mulan", "embeddings_flamingo")
        candidates.firstOrNull { tableHasRows(it) } ?: "embeddings_fused"
    }

    private fun getTableNames(): Set<String> {
        val names = mutableSetOf<String>()
        db.rawQuery(
            "SELECT name FROM sqlite_master WHERE type='table'", null
        ).use { cursor ->
            while (cursor.moveToNext()) {
                names.add(cursor.getString(0))
            }
        }
        return names
    }

    private fun tableHasRows(tableName: String): Boolean {
        return try {
            db.rawQuery("SELECT 1 FROM [$tableName] LIMIT 1", null).use { it.moveToFirst() }
        } catch (e: Exception) {
            false
        }
    }

    /**
     * Get total number of tracks in the database.
     */
    fun getTrackCount(): Int {
        val cursor = db.rawQuery("SELECT COUNT(*) FROM tracks", null)
        return cursor.use {
            if (it.moveToFirst()) it.getInt(0) else 0
        }
    }

    /**
     * Get the count of embeddings in the active table.
     */
    fun getEmbeddingCount(): Int {
        return try {
            val cursor = db.rawQuery("SELECT COUNT(*) FROM [${embeddingTable}]", null)
            cursor.use { if (it.moveToFirst()) it.getInt(0) else 0 }
        } catch (e: Exception) {
            0
        }
    }

    /**
     * Detect the actual embedding dimension by probing the first row.
     */
    fun getEmbeddingDim(): Int? {
        return try {
            db.rawQuery("SELECT length(embedding) FROM [${embeddingTable}] LIMIT 1", null).use {
                if (it.moveToFirst()) it.getInt(0) / 4 else null
            }
        } catch (e: Exception) {
            null
        }
    }

    /**
     * Get available embedding models and their row counts.
     * Returns list of (model_name, count) for embedding tables that have data.
     */
    fun getAvailableModels(): List<Pair<String, Int>> {
        val models = mutableListOf<Pair<String, Int>>()
        val tables = getTableNames()
        for (table in tables) {
            if (!table.startsWith("embeddings_")) continue
            val model = table.removePrefix("embeddings_")
            try {
                db.rawQuery("SELECT COUNT(*) FROM [$table]", null).use { cursor ->
                    if (cursor.moveToFirst()) {
                        val count = cursor.getInt(0)
                        if (count > 0) models.add(model to count)
                    }
                }
            } catch (_: Exception) { }
        }
        return models
    }

    /**
     * Get a track by its metadata key (primary matching method).
     * Tries in order:
     * 1. Exact artist|album|title match
     * 2. Artist|title match (ignores album)
     * 3. Fuzzy artist match
     */
    fun findTrackByMetadataKey(key: String): EmbeddedTrack? {
        val parts = key.split("|")
        if (parts.size >= 3) {
            val artist = parts[0]
            val album = parts[1]
            val title = parts[2]

            // 1. Try exact artist|album|title match
            val exactPattern = "$artist|$album|$title|%"
            db.rawQuery(
                "SELECT id, metadata_key, filename_key, artist, album, title, duration_ms, file_path FROM tracks WHERE metadata_key LIKE ?",
                arrayOf(exactPattern)
            ).use { cursorToTrack(it)?.let { return it } }

            // 2. Try artist|title (any album)
            val artistTitlePattern = "$artist|%|$title|%"
            db.rawQuery(
                "SELECT id, metadata_key, filename_key, artist, album, title, duration_ms, file_path FROM tracks WHERE metadata_key LIKE ?",
                arrayOf(artistTitlePattern)
            ).use { cursorToTrack(it)?.let { return it } }

            // 3. Fuzzy: find by title, check artist substring overlap
            val titlePattern = "%|%|$title|%"
            db.rawQuery(
                "SELECT id, metadata_key, filename_key, artist, album, title, duration_ms, file_path FROM tracks WHERE metadata_key LIKE ?",
                arrayOf(titlePattern)
            ).use {
                val matches = cursorToTrackList(it)
                return matches.find { track ->
                    val embeddedArtist = track.artist?.lowercase() ?: ""
                    artist.isNotEmpty() && (
                        embeddedArtist.contains(artist) || artist.contains(embeddedArtist)
                    )
                }
            }
        }
        return null
    }

    /**
     * Get a track by its filename key (fallback matching).
     */
    fun findTrackByFilenameKey(key: String): EmbeddedTrack? {
        val cursor = db.rawQuery(
            "SELECT id, metadata_key, filename_key, artist, album, title, duration_ms, file_path FROM tracks WHERE filename_key = ?",
            arrayOf(key)
        )
        return cursor.use { cursorToTrack(it) }
    }

    /**
     * Find tracks by artist and title only (fuzzy fallback).
     */
    fun findTracksByArtistAndTitle(artist: String, title: String): List<EmbeddedTrack> {
        val key = "${artist.lowercase()}|%|${title.lowercase()}|%"
        val cursor = db.rawQuery(
            "SELECT id, metadata_key, filename_key, artist, album, title, duration_ms, file_path FROM tracks WHERE metadata_key LIKE ?",
            arrayOf(key)
        )
        return cursor.use { cursorToTrackList(it) }
    }

    /**
     * Get the embedding for a track by its ID from the active embedding table.
     */
    fun getEmbedding(trackId: Long): FloatArray? {
        val cursor = db.rawQuery(
            "SELECT embedding FROM [${embeddingTable}] WHERE track_id = ?",
            arrayOf(trackId.toString())
        )
        return cursor.use {
            if (it.moveToFirst()) {
                val blob = it.getBlob(0)
                blobToFloatArray(blob)
            } else null
        }
    }

    /**
     * Stream embeddings one row at a time without holding them all in memory.
     */
    fun forEachEmbeddingRaw(block: (trackId: Long, blob: ByteArray) -> Unit) {
        db.rawQuery("SELECT track_id, embedding FROM [${embeddingTable}]", null).use { cursor ->
            while (cursor.moveToNext()) {
                val trackId = cursor.getLong(0)
                val blob = cursor.getBlob(1)
                block(trackId, blob)
            }
        }
    }

    /**
     * Load cluster assignments: track_id -> cluster_id.
     */
    fun loadClusterAssignments(): Map<Long, Int> {
        val result = mutableMapOf<Long, Int>()
        try {
            db.rawQuery("SELECT id, cluster_id FROM tracks WHERE cluster_id IS NOT NULL", null).use { cursor ->
                while (cursor.moveToNext()) {
                    result[cursor.getLong(0)] = cursor.getInt(1)
                }
            }
        } catch (e: Exception) {
            // cluster_id column may not exist in older databases
        }
        return result
    }

    /**
     * Load cluster centroids: cluster_id -> embedding.
     */
    fun loadCentroids(): Map<Int, FloatArray> {
        val result = mutableMapOf<Int, FloatArray>()
        try {
            db.rawQuery("SELECT cluster_id, embedding FROM clusters", null).use { cursor ->
                while (cursor.moveToNext()) {
                    result[cursor.getInt(0)] = blobToFloatArray(cursor.getBlob(1))
                }
            }
        } catch (e: Exception) {
            // clusters table may not exist in older databases
        }
        return result
    }

    /**
     * Get a track by its ID.
     */
    fun getTrackById(id: Long): EmbeddedTrack? {
        val cursor = db.rawQuery(
            "SELECT id, metadata_key, filename_key, artist, album, title, duration_ms, file_path FROM tracks WHERE id = ?",
            arrayOf(id.toString())
        )
        return cursor.use { cursorToTrack(it) }
    }

    /**
     * Get database metadata value.
     */
    fun getMetadata(key: String): String? {
        val cursor = db.rawQuery(
            "SELECT value FROM metadata WHERE key = ?",
            arrayOf(key)
        )
        return cursor.use {
            if (it.moveToFirst()) it.getString(0) else null
        }
    }

    /**
     * Check if a binary data key exists in the binary_data table.
     * Does NOT read the blob — safe for large entries.
     */
    fun hasBinaryData(key: String): Boolean {
        return try {
            db.rawQuery(
                "SELECT 1 FROM binary_data WHERE key = ?",
                arrayOf(key)
            ).use { it.moveToFirst() }
        } catch (e: Exception) {
            false
        }
    }

    /**
     * Extract a binary data blob to a file, reading in chunks to avoid
     * Android's ~2 MB CursorWindow limit.
     */
    fun extractBinaryToFile(key: String, outFile: File): Boolean {
        return try {
            val length = db.rawQuery(
                "SELECT length(data) FROM binary_data WHERE key = ?",
                arrayOf(key)
            ).use {
                if (it.moveToFirst()) it.getLong(0) else return false
            }

            val chunkSize = 1_000_000 // 1 MB chunks
            FileOutputStream(outFile).use { fos ->
                var offset = 1 // SQL substr is 1-indexed
                while (offset <= length) {
                    val chunk = db.rawQuery(
                        "SELECT substr(data, ?, ?) FROM binary_data WHERE key = ?",
                        arrayOf(offset.toString(), chunkSize.toString(), key)
                    ).use { cursor ->
                        if (cursor.moveToFirst()) cursor.getBlob(0) else null
                    } ?: break
                    fos.write(chunk)
                    offset += chunkSize
                }
            }
            true
        } catch (e: Exception) {
            false
        }
    }

    private fun cursorToTrack(cursor: Cursor): EmbeddedTrack? {
        return if (cursor.moveToFirst()) {
            EmbeddedTrack(
                id = cursor.getLong(0),
                metadataKey = cursor.getString(1),
                filenameKey = cursor.getString(2),
                artist = cursor.getString(3),
                album = cursor.getString(4),
                title = cursor.getString(5),
                durationMs = cursor.getInt(6),
                filePath = cursor.getString(7)
            )
        } else null
    }

    private fun cursorToTrackList(cursor: Cursor): List<EmbeddedTrack> {
        val result = mutableListOf<EmbeddedTrack>()
        while (cursor.moveToNext()) {
            result.add(
                EmbeddedTrack(
                    id = cursor.getLong(0),
                    metadataKey = cursor.getString(1),
                    filenameKey = cursor.getString(2),
                    artist = cursor.getString(3),
                    album = cursor.getString(4),
                    title = cursor.getString(5),
                    durationMs = cursor.getInt(6),
                    filePath = cursor.getString(7)
                )
            )
        }
        return result
    }

    fun close() {
        db.close()
    }
}
