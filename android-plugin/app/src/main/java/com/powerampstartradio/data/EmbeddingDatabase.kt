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
 */
class EmbeddingDatabase private constructor(
    private val db: SQLiteDatabase
) {
    companion object {
        private const val EMBEDDING_DIM = 512

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
     * Get total number of tracks in the database.
     */
    fun getTrackCount(): Int {
        val cursor = db.rawQuery("SELECT COUNT(*) FROM tracks", null)
        return cursor.use {
            if (it.moveToFirst()) it.getInt(0) else 0
        }
    }

    /**
     * Get a track by its metadata key (primary matching method).
     * Tries in order:
     * 1. Exact artist|album|title match (finds specific rendition)
     * 2. Artist|title match (ignores album)
     * 3. Fuzzy artist match (handles "Artist1; Artist2" split tags)
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
     * Get the embedding for a track by its ID.
     */
    fun getEmbedding(trackId: Long): FloatArray? {
        val cursor = db.rawQuery(
            "SELECT embedding FROM embeddings WHERE track_id = ?",
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
     * Get all embeddings for similarity computation.
     * Returns a map of track ID to embedding.
     */
    fun getAllEmbeddings(): Map<Long, FloatArray> {
        val cursor = db.rawQuery("SELECT track_id, embedding FROM embeddings", null)
        val result = mutableMapOf<Long, FloatArray>()
        cursor.use {
            while (it.moveToNext()) {
                val trackId = it.getLong(0)
                val blob = it.getBlob(1)
                result[trackId] = blobToFloatArray(blob)
            }
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
     * Get all tracks.
     */
    fun getAllTracks(): List<EmbeddedTrack> {
        val cursor = db.rawQuery(
            "SELECT id, metadata_key, filename_key, artist, album, title, duration_ms, file_path FROM tracks",
            null
        )
        return cursor.use { cursorToTrackList(it) }
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
