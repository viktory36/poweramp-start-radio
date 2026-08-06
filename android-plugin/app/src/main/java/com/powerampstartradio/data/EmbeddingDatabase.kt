package com.powerampstartradio.data

import android.content.Context
import android.database.Cursor
import android.database.sqlite.SQLiteDatabase
import android.net.Uri
import android.util.Log
import com.powerampstartradio.indexing.v2.StableTrackSpanIdentityStrength
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Locale

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
    val filePath: String,
    val source: String = "desktop",
)

data class EmbeddedTrackTextSearchResult(
    val tracks: List<EmbeddedTrack>,
    val hasMore: Boolean,
)

/**
 * Wrapper for reading the embeddings SQLite database created by the desktop indexer.
 *
 * Primary table: embeddings_clamp3 (768d CLaMP3 audio embeddings).
 */
class EmbeddingDatabase private constructor(
    private val db: SQLiteDatabase
) {
    /** Whether this database was opened in read-write mode. */
    val isReadWrite: Boolean get() = !db.isReadOnly

    /** Canonical backing file used to bind consumers to one exact database generation. */
    val databaseFile: File get() = File(db.path).canonicalFile

    companion object {
        private const val TAG = "EmbeddingDatabase"
        /**
         * Open the database in read-only mode.
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
         * Open the database in read-write mode for inserting new tracks.
         */
        fun openReadWrite(dbFile: File): EmbeddingDatabase {
            val db = SQLiteDatabase.openDatabase(
                dbFile.absolutePath,
                null,
                SQLiteDatabase.OPEN_READWRITE
            )
            // Migration: add source column if not present
            try {
                db.execSQL("ALTER TABLE tracks ADD COLUMN source TEXT DEFAULT 'desktop'")
                Log.d(TAG, "Migration: added 'source' column to tracks")
            } catch (_: Exception) {
                // Column already exists
            }
            // Migration: create embeddings_clamp3 table if not present
            try {
                db.execSQL("""
                    CREATE TABLE IF NOT EXISTS embeddings_clamp3 (
                        track_id INTEGER PRIMARY KEY REFERENCES tracks(id),
                        embedding BLOB NOT NULL
                    )
                """)
            } catch (_: Exception) {
                // Table already exists
            }
            ensureBinaryDataSchema(db)
            return EmbeddingDatabase(db)
        }

        /** Bootstrap-compatible legacy databases may not have stored a graph yet. */
        private fun ensureBinaryDataSchema(db: SQLiteDatabase) {
            db.execSQL(
                """
                CREATE TABLE IF NOT EXISTS binary_data (
                    key TEXT PRIMARY KEY,
                    data BLOB NOT NULL
                )
                """.trimIndent(),
            )
            data class Column(val type: String, val notNull: Boolean, val primaryKey: Boolean)
            val columns = linkedMapOf<String, Column>()
            db.rawQuery("PRAGMA table_info(binary_data)", null).use { cursor ->
                val name = cursor.getColumnIndexOrThrow("name")
                val type = cursor.getColumnIndexOrThrow("type")
                val notNull = cursor.getColumnIndexOrThrow("notnull")
                val primaryKey = cursor.getColumnIndexOrThrow("pk")
                while (cursor.moveToNext()) {
                    columns[cursor.getString(name)] = Column(
                        type = cursor.getString(type).uppercase(),
                        notNull = cursor.getInt(notNull) != 0,
                        primaryKey = cursor.getInt(primaryKey) != 0,
                    )
                }
            }
            require(columns["key"]?.let { it.type == "TEXT" && it.primaryKey } == true &&
                columns["data"]?.let { it.type == "BLOB" && it.notNull } == true
            ) { "binary_data has an incompatible schema: $columns" }
        }

        /**
         * Import database from a content URI (e.g., from document picker).
         */
        fun importFrom(context: Context, uri: Uri, destFile: File): EmbeddingDatabase {
            val t0 = System.nanoTime()
            context.contentResolver.openInputStream(uri)?.use { input ->
                FileOutputStream(destFile).use { output ->
                    input.copyTo(output)
                }
            } ?: throw IllegalArgumentException("Cannot open URI: $uri")

            val copyMs = (System.nanoTime() - t0) / 1_000_000
            Log.i(TAG, "Imported DB: ${destFile.length() / 1024}KB in ${copyMs}ms from $uri")
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

        /**
         * Convert a FloatArray to a BLOB (little-endian float32).
         */
        fun floatArrayToBlob(floats: FloatArray): ByteArray {
            val buffer = ByteBuffer.allocate(floats.size * 4).order(ByteOrder.LITTLE_ENDIAN)
            for (f in floats) buffer.putFloat(f)
            return buffer.array()
        }
    }

    /**
     * The embedding table to use for similarity search.
     * Uses embeddings_clamp3 (768d CLaMP3 audio embeddings).
     * Computed each time to avoid stale caches after on-device indexing.
     */
    val embeddingTable: String
        get() = "embeddings_clamp3"

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
            Log.w(TAG, "getEmbeddingCount failed (table=$embeddingTable): ${e.message}")
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
            Log.w(TAG, "getEmbeddingDim failed (table=$embeddingTable): ${e.message}")
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
            } catch (e: Exception) {
                Log.w(TAG, "getAvailableModels: failed to query $table: ${e.message}")
            }
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
        // 1. Try exact metadata_key match (uses idx_tracks_metadata_key index)
        db.rawQuery(
            "SELECT id, metadata_key, filename_key, artist, album, title, duration_ms, file_path FROM tracks WHERE metadata_key = ?",
            arrayOf(key)
        ).use { cursorToTrack(it)?.let { return it } }

        // 2. Prefix match: same artist|album|title, any duration (index-friendly prefix scan)
        val lastPipe = key.lastIndexOf('|')
        if (lastPipe > 0) {
            val prefix = key.substring(0, lastPipe + 1)  // "artist|album|title|"
            db.rawQuery(
                "SELECT id, metadata_key, filename_key, artist, album, title, duration_ms, file_path FROM tracks WHERE metadata_key >= ? AND metadata_key < ?",
                arrayOf(prefix, prefix + "\uffff")
            ).use { cursorToTrack(it)?.let { return it } }
        }

        // 3. Individual column match: artist + title (any album/duration)
        val parts = key.split("|")
        if (parts.size >= 3) {
            val artist = parts[0]
            val title = parts[2]

            db.rawQuery(
                "SELECT id, metadata_key, filename_key, artist, album, title, duration_ms, file_path FROM tracks WHERE LOWER(artist) = ? AND LOWER(title) = ?",
                arrayOf(artist, title)
            ).use { cursorToTrack(it)?.let { return it } }

            // 4. Fuzzy: find by title, check artist substring overlap
            db.rawQuery(
                "SELECT id, metadata_key, filename_key, artist, album, title, duration_ms, file_path FROM tracks WHERE LOWER(title) = ?",
                arrayOf(title)
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

    fun findTracksByPath(path: String): List<EmbeddedTrack> {
        val cursor = db.rawQuery(
            "SELECT id, metadata_key, filename_key, artist, album, title, duration_ms, file_path FROM tracks WHERE file_path = ?",
            arrayOf(path)
        )
        return cursor.use { cursorToTrackList(it) }
    }

    fun findTracksByMetadataPrefix(prefix: String): List<EmbeddedTrack> {
        val cursor = db.rawQuery(
            "SELECT id, metadata_key, filename_key, artist, album, title, duration_ms, file_path FROM tracks WHERE metadata_key >= ? AND metadata_key < ?",
            arrayOf(prefix, prefix + "\uffff")
        )
        return cursor.use { cursorToTrackList(it) }
    }

    fun findTracksByArtistAlbumTitle(artist: String, album: String, title: String): List<EmbeddedTrack> {
        val cursor = db.rawQuery(
            "SELECT id, metadata_key, filename_key, artist, album, title, duration_ms, file_path FROM tracks WHERE LOWER(artist) = ? AND LOWER(album) = ? AND LOWER(title) = ?",
            arrayOf(artist.lowercase(), album.lowercase(), title.lowercase())
        )
        return cursor.use { cursorToTrackList(it) }
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

    fun findTracksByFilenameKey(key: String): List<EmbeddedTrack> {
        val cursor = db.rawQuery(
            "SELECT id, metadata_key, filename_key, artist, album, title, duration_ms, file_path FROM tracks WHERE filename_key = ?",
            arrayOf(key)
        )
        return cursor.use { cursorToTrackList(it) }
    }

    /**
     * Find tracks by artist and title only (fuzzy fallback).
     */
    fun findTracksByArtistAndTitle(artist: String, title: String): List<EmbeddedTrack> {
        val cursor = db.rawQuery(
            "SELECT id, metadata_key, filename_key, artist, album, title, duration_ms, file_path FROM tracks WHERE LOWER(artist) = ? AND LOWER(title) = ?",
            arrayOf(artist.lowercase(), title.lowercase())
        )
        return cursor.use { cursorToTrackList(it) }
    }

    fun findTracksByTitle(title: String): List<EmbeddedTrack> {
        val cursor = db.rawQuery(
            "SELECT id, metadata_key, filename_key, artist, album, title, duration_ms, file_path FROM tracks WHERE LOWER(title) = ?",
            arrayOf(title.lowercase())
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
     *
     * @param table The embedding table to read from. Defaults to the auto-detected best table.
     */
    fun forEachEmbeddingRaw(
        table: String = embeddingTable,
        block: (trackId: Long, blob: ByteArray) -> Unit,
    ) {
        db.rawQuery("SELECT track_id, embedding FROM [$table] ORDER BY track_id", null).use { cursor ->
            while (cursor.moveToNext()) {
                val trackId = cursor.getLong(0)
                val blob = cursor.getBlob(1)
                block(trackId, blob)
            }
        }
    }

    /**
     * Get the count of embeddings in a specific table.
     */
    fun getEmbeddingCountForTable(table: String): Int {
        return try {
            db.rawQuery("SELECT COUNT(*) FROM [$table]", null).use {
                if (it.moveToFirst()) it.getInt(0) else 0
            }
        } catch (e: Exception) {
            Log.w(TAG, "getEmbeddingCountForTable($table) failed: ${e.message}")
            0
        }
    }

    /**
     * Detect the embedding dimension for a specific table.
     */
    fun getEmbeddingDimForTable(table: String): Int? {
        return try {
            db.rawQuery("SELECT length(embedding) FROM [$table] LIMIT 1", null).use {
                if (it.moveToFirst()) it.getInt(0) / 4 else null
            }
        } catch (e: Exception) {
            Log.w(TAG, "getEmbeddingDimForTable($table) failed: ${e.message}")
            null
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
            Log.d(TAG, "loadClusterAssignments: cluster_id column not available: ${e.message}")
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
     * Read artist equality once and encode it in the exact immutable embedding-row order.
     * Unknown/blank credits remain unconstrained, matching [PostFilter.canAdd] semantics.
     */
    internal fun loadArtistCreditCatalog(orderedTrackIds: LongArray): ArtistCreditCatalog {
        require(orderedTrackIds.indices.drop(1).all { position ->
            orderedTrackIds[position] > orderedTrackIds[position - 1]
        }) { "Artist-credit input track IDs must be strictly increasing" }

        val creditIds = IntArray(orderedTrackIds.size) { UNKNOWN_ARTIST_CREDIT_ID }
        val internedCredits = HashMap<String, Int>()
        var requestedPosition = 0
        db.rawQuery("SELECT id, artist FROM tracks ORDER BY id", null).use { cursor ->
            while (requestedPosition < orderedTrackIds.size && cursor.moveToNext()) {
                val databaseTrackId = cursor.getLong(0)
                val requestedTrackId = orderedTrackIds[requestedPosition]
                if (databaseTrackId < requestedTrackId) continue
                require(databaseTrackId == requestedTrackId) {
                    "Embedding track $requestedTrackId is absent from the metadata database"
                }
                normalizeArtistCredit(cursor.getString(1))?.let { credit ->
                    creditIds[requestedPosition] = internedCredits.getOrPut(credit) {
                        internedCredits.size
                    }
                }
                requestedPosition++
            }
        }
        require(requestedPosition == orderedTrackIds.size) {
            "Only $requestedPosition/${orderedTrackIds.size} embedding tracks have metadata rows"
        }
        return ArtistCreditCatalog(
            orderedTrackIds = orderedTrackIds.copyOf(),
            creditIdByTrackPosition = creditIds,
            distinctCreditCount = internedCredits.size,
        )
    }

    /**
     * Reads identity receipts in the same order as the PEMB embedding index.
     *
     * A receipt is exposed only for the exact active embedding spec. Missing or legacy receipts
     * stay null rather than being inferred from mutable track metadata.
     */
    internal fun queryStableTrackIdentityRows(
        receiptTable: String,
        receiptSchemaVersion: Int,
        expectedEmbeddingSpecId: String?,
    ): List<StableTrackIdentityRow> {
        require(receiptTable.matches(Regex("[A-Za-z_][A-Za-z0-9_]*"))) {
            "Unsafe receipt table name"
        }
        val hasReceiptTable = expectedEmbeddingSpecId != null && db.rawQuery(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1",
            arrayOf(receiptTable),
        ).use(Cursor::moveToFirst)
        val query: String
        val args: Array<String>?
        if (hasReceiptTable) {
            query = """
                SELECT e.track_id, r.stable_track_span_id, r.stable_identity_spec_id,
                       r.stable_identity_strength, r.embedding_spec_id, r.embedding_sha256,
                       t.filename_key, t.artist, t.album, t.title, t.duration_ms
                FROM [$embeddingTable] e
                JOIN tracks t ON t.id = e.track_id
                LEFT JOIN [$receiptTable] r
                  ON r.track_id = e.track_id
                 AND r.receipt_schema_version = ?
                 AND r.embedding_spec_id = ?
                ORDER BY e.track_id
            """.trimIndent()
            args = arrayOf(receiptSchemaVersion.toString(), checkNotNull(expectedEmbeddingSpecId))
        } else {
            query = """
                SELECT e.track_id, NULL, NULL, NULL, NULL, NULL,
                       t.filename_key, t.artist, t.album, t.title, t.duration_ms
                FROM [$embeddingTable] e
                JOIN tracks t ON t.id = e.track_id
                ORDER BY e.track_id
            """.trimIndent()
            args = null
        }

        return buildList {
            db.rawQuery(query, args).use { cursor ->
                while (cursor.moveToNext()) {
                    add(
                        StableTrackIdentityRow(
                            trackId = cursor.getLong(0),
                            stableTrackSpanId = cursor.getNullableString(1),
                            stableIdentitySpecId = cursor.getNullableString(2),
                            stableIdentityStrength = cursor.getNullableString(3)?.let { value ->
                                runCatching { StableTrackSpanIdentityStrength.valueOf(value) }
                                    .getOrElse {
                                        throw IllegalStateException(
                                            "Unsupported stable identity strength for track ${cursor.getLong(0)}",
                                            it,
                                        )
                                    }
                            },
                            embeddingSpecId = cursor.getNullableString(4),
                            embeddingSha256 = cursor.getNullableString(5),
                            filenameKey = cursor.getNullableString(6),
                            artist = cursor.getNullableString(7),
                            album = cursor.getNullableString(8),
                            title = cursor.getNullableString(9),
                            durationMs = cursor.getInt(10),
                        ),
                    )
                }
            }
        }
    }

    /**
     * Count only proven full-content duplicate excess for a caller-defined active domain.
     *
     * This streams receipt rows and retains one stable-ID set; it deliberately avoids building
     * the full process-wide identity catalog when a readiness surface needs only the exact count.
     */
    internal fun queryActiveFullContentDuplicateExcess(
        receiptTable: String,
        receiptSchemaVersion: Int,
        expectedEmbeddingSpecId: String?,
        isActiveTrackId: (Long) -> Boolean,
    ): Int {
        require(receiptTable.matches(Regex("[A-Za-z_][A-Za-z0-9_]*"))) {
            "Unsafe receipt table name"
        }
        if (expectedEmbeddingSpecId == null) return 0
        val hasReceiptTable = db.rawQuery(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1",
            arrayOf(receiptTable),
        ).use(Cursor::moveToFirst)
        if (!hasReceiptTable) return 0

        val uniqueActiveStableIds = HashSet<String>()
        var activeFullContentRows = 0
        db.rawQuery(
            """
            SELECT track_id, stable_track_span_id
            FROM [$receiptTable]
            WHERE receipt_schema_version = ?
              AND embedding_spec_id = ?
              AND stable_identity_strength = ?
            ORDER BY track_id
            """.trimIndent(),
            arrayOf(
                receiptSchemaVersion.toString(),
                expectedEmbeddingSpecId,
                StableTrackSpanIdentityStrength.FULL_CONTENT_SHA256.name,
            ),
        ).use { cursor ->
            var previousTrackId = Long.MIN_VALUE
            while (cursor.moveToNext()) {
                val trackId = cursor.getLong(0)
                require(trackId > previousTrackId) {
                    "Full-content receipt rows are not strictly ordered"
                }
                previousTrackId = trackId
                val stableTrackSpanId = cursor.getNullableString(1)
                    ?: error("Full-content receipt $trackId has no stable track-span ID")
                if (isActiveTrackId(trackId)) {
                    activeFullContentRows++
                    uniqueActiveStableIds += stableTrackSpanId
                }
            }
        }
        return activeFullContentRows - uniqueActiveStableIds.size
    }

    internal fun requireByteEqualStableIdentityEmbeddings(
        equivalentTrackIdGroups: Collection<List<Long>>,
    ) {
        equivalentTrackIdGroups.forEach { trackIds ->
            if (trackIds.size <= 1) return@forEach
            val referenceId = trackIds.first()
            val reference = getRawEmbeddingBlob(referenceId)
                ?: error("Stable identity track $referenceId has no embedding")
            trackIds.drop(1).forEach { trackId ->
                val candidate = getRawEmbeddingBlob(trackId)
                    ?: error("Stable identity track $trackId has no embedding")
                require(reference.contentEquals(candidate)) {
                    "Stable identity rows $referenceId and $trackId have different embedding bytes"
                }
            }
        }
    }

    private fun getRawEmbeddingBlob(trackId: Long): ByteArray? = db.rawQuery(
        "SELECT embedding FROM [$embeddingTable] WHERE track_id = ?",
        arrayOf(trackId.toString()),
    ).use { cursor ->
        if (cursor.moveToFirst()) cursor.getBlob(0) else null
    }

    /**
     * Get all tracks in the database.
     */
    fun getAllTracks(): List<EmbeddedTrack> {
        val cursor = db.rawQuery(
            "SELECT id, metadata_key, filename_key, artist, album, title, duration_ms, file_path " +
                "FROM tracks ORDER BY LOWER(COALESCE(artist, '')), LOWER(COALESCE(album, '')), LOWER(COALESCE(title, ''))",
            null
        )
        return cursor.use { cursorToTrackList(it) }
    }

    /**
     * Stream the catalog-reconciliation columns through [transform].
     *
     * Unlike [getAllTracks], this never retains an intermediate [EmbeddedTrack] list. The returned
     * list therefore contains only the caller's compact projection while the cursor advances in
     * the same deterministic order used by [getAllTracks].
     */
    internal fun <T> mapAllTrackCatalogRows(
        onRowCount: (Int) -> Unit = {},
        transform: (
            trackId: Long,
            artist: String?,
            album: String?,
            title: String?,
            durationMs: Int,
            filePath: String,
        ) -> T,
    ): List<T> = db.rawQuery(
        "SELECT id, artist, album, title, duration_ms, file_path " +
            "FROM tracks ORDER BY LOWER(COALESCE(artist, '')), " +
            "LOWER(COALESCE(album, '')), LOWER(COALESCE(title, ''))",
        null,
    ).use { cursor ->
        val rowCount = cursor.count
        onRowCount(rowCount)
        ArrayList<T>(rowCount).also { result ->
            while (cursor.moveToNext()) {
                result += transform(
                    cursor.getLong(0),
                    cursor.getString(1),
                    cursor.getString(2),
                    cursor.getString(3),
                    cursor.getInt(4),
                    cursor.getString(5),
                )
            }
            require(result.size == rowCount) {
                "Track catalog cursor count changed while projecting rows"
            }
        }
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
            Log.d(TAG, "hasBinaryData($key): table missing or error: ${e.message}")
            false
        }
    }

    /** Read a deliberately small binary contract without exposing large graph blobs to CursorWindow. */
    fun getSmallBinaryData(key: String, maxBytes: Int = 4_096): ByteArray? {
        require(maxBytes > 0)
        return try {
            db.rawQuery(
                "SELECT length(data), data FROM binary_data WHERE key = ?",
                arrayOf(key),
            ).use { cursor ->
                if (!cursor.moveToFirst()) return@use null
                val length = cursor.getInt(0)
                require(length in 1..maxBytes) { "binary value $key is unexpectedly large: $length" }
                cursor.getBlob(1).also { require(it.size == length) }
            }
        } catch (error: Exception) {
            Log.d(TAG, "getSmallBinaryData($key) failed: ${error.message}")
            null
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
            Log.e(TAG, "extractBinaryToFile($key) failed: ${e.message}")
            false
        }
    }

    private fun cursorToTrack(cursor: Cursor): EmbeddedTrack? {
        return if (cursor.moveToFirst()) {
            cursorToTrackAtCurrentPosition(cursor)
        } else null
    }

    private fun cursorToTrackList(cursor: Cursor): List<EmbeddedTrack> {
        val result = mutableListOf<EmbeddedTrack>()
        while (cursor.moveToNext()) {
            result.add(cursorToTrackAtCurrentPosition(cursor))
        }
        return result
    }

    private fun cursorToTrackAtCurrentPosition(cursor: Cursor): EmbeddedTrack {
        val sourceIdx = cursor.getColumnIndex("source")
        return EmbeddedTrack(
            id = cursor.getLong(0),
            metadataKey = cursor.getString(1),
            filenameKey = cursor.getString(2),
            artist = cursor.getString(3),
            album = cursor.getString(4),
            title = cursor.getString(5),
            durationMs = cursor.getInt(6),
            filePath = cursor.getString(7),
            source = if (sourceIdx >= 0) cursor.getString(sourceIdx) ?: "desktop" else "desktop",
        )
    }

    /**
     * Search tracks by free-text query against artist + title.
     * Splits query into tokens and requires all tokens to match (case-insensitive).
     *
     * @param query Free text like "time pachanga boys"
     * @param limit Maximum results to return
     * @return Matching tracks, best matches first
     */
    fun searchTracksByText(query: String, limit: Int = 10): List<EmbeddedTrack> {
        return searchTracksByTextPage(query = query, limit = limit).tracks
    }

    /**
     * Deterministic recording lookup with eligibility filtering before the result limit.
     *
     * The extra-row signal lets callers tell the user that the display is a bounded chooser,
     * without materializing every broad metadata match in a large library.
     */
    fun searchTracksByTextPage(
        query: String,
        limit: Int = 50,
        includeTrackId: (Long) -> Boolean = { true },
        canonicalTrackId: (Long) -> Long = { it },
    ): EmbeddedTrackTextSearchResult {
        require(limit > 0) { "Recording lookup limit must be positive" }
        val tokens = Regex("[\\p{L}\\p{M}\\p{N}]+")
            .findAll(query.lowercase(Locale.ROOT))
            .map { it.value }
            .toList()
        if (tokens.isEmpty()) {
            return EmbeddedTrackTextSearchResult(tracks = emptyList(), hasMore = false)
        }

        val searchableText = "TRIM(COALESCE(artist,'') || ' ' || COALESCE(title,''))"
        val whereClauses = tokens.map {
            "LOWER($searchableText) LIKE ? ESCAPE '!'"
        }
        val whereArgs = tokens.map { "%${escapeLikePattern(it)}%" }
        val normalizedQuery = tokens.joinToString(" ")
        val sql = """
            SELECT id, metadata_key, filename_key, artist, album, title, duration_ms, file_path
            FROM tracks
            WHERE ${whereClauses.joinToString(" AND ")}
            ORDER BY
                CASE
                    WHEN LOWER(TRIM(COALESCE(title,''))) = ? THEN 0
                    WHEN LOWER($searchableText) = ? THEN 1
                    WHEN LOWER(TRIM(COALESCE(title,''))) LIKE ? ESCAPE '!' THEN 2
                    ELSE 3
                END,
                LENGTH($searchableText),
                LOWER(COALESCE(artist,'')),
                LOWER(COALESCE(title,'')),
                id
        """.trimIndent()
        val allArgs = (whereArgs + listOf(
            normalizedQuery,
            normalizedQuery,
            "${escapeLikePattern(normalizedQuery)}%",
        )).toTypedArray()

        val tracks = ArrayList<EmbeddedTrack>(limit)
        val seenCanonicalTrackIds = HashSet<Long>(limit)
        var hasMore = false
        db.rawQuery(sql, allArgs).use { cursor ->
            while (cursor.moveToNext()) {
                val trackId = cursor.getLong(0)
                if (!includeTrackId(trackId)) continue
                if (!seenCanonicalTrackIds.add(canonicalTrackId(trackId))) continue
                if (tracks.size == limit) {
                    hasMore = true
                    break
                }
                tracks.add(cursorToTrackAtCurrentPosition(cursor))
            }
        }
        return EmbeddedTrackTextSearchResult(tracks = tracks, hasMore = hasMore)
    }

    private fun escapeLikePattern(value: String): String = value
        .replace("!", "!!")
        .replace("%", "!%")
        .replace("_", "!_")

    // --- Write operations (require openReadWrite) ---

    /**
     * Insert a new track and return its ID.
     */
    fun insertTrack(
        metadataKey: String,
        filenameKey: String,
        artist: String?,
        album: String?,
        title: String?,
        durationMs: Int,
        filePath: String,
        source: String = "desktop",
    ): Long {
        val values = android.content.ContentValues().apply {
            put("metadata_key", metadataKey)
            put("filename_key", filenameKey)
            put("artist", artist)
            put("album", album)
            put("title", title)
            put("duration_ms", durationMs)
            put("file_path", filePath)
            put("source", source)
        }
        return db.insertOrThrow("tracks", null, values)
    }

    /**
     * Insert an embedding for a track in the specified model table.
     */
    fun insertEmbedding(tableName: String, trackId: Long, embedding: FloatArray) {
        val blob = floatArrayToBlob(embedding)
        val values = android.content.ContentValues().apply {
            put("track_id", trackId)
            put("embedding", blob)
        }
        db.insertWithOnConflict(tableName, null, values,
            android.database.sqlite.SQLiteDatabase.CONFLICT_REPLACE)
    }

    /**
     * Store a binary blob in the binary_data table.
     */
    fun setBinaryData(key: String, data: ByteArray) {
        val values = android.content.ContentValues().apply {
            put("key", key)
            put("data", data)
        }
        db.insertWithOnConflict("binary_data", null, values,
            android.database.sqlite.SQLiteDatabase.CONFLICT_REPLACE)
    }

    /**
     * Delete a binary blob from the binary_data table.
     */
    fun deleteBinaryData(key: String) {
        db.delete("binary_data", "key = ?", arrayOf(key))
    }

    /**
     * Delete tracks and any embeddings attached to them.
     *
     * Returns the number of rows removed from the tracks table.
     */
    fun deleteTracks(trackIds: Set<Long>): Int {
        if (trackIds.isEmpty()) return 0
        val idArgs = trackIds.map(Long::toString).toTypedArray()
        val placeholders = trackIds.joinToString(",") { "?" }
        val embeddingTables = getTableNames().filter { it.startsWith("embeddings_") }
        var deletedTracks = 0

        db.beginTransaction()
        try {
            for (table in embeddingTables) {
                db.delete(table, "track_id IN ($placeholders)", idArgs)
            }
            deletedTracks = db.delete("tracks", "id IN ($placeholders)", idArgs)
            db.setTransactionSuccessful()
        } finally {
            db.endTransaction()
        }

        return deletedTracks
    }

    /**
     * Get all metadata keys from the tracks table (for detecting unindexed tracks).
     */
    fun getAllMetadataKeys(): Set<String> {
        val keys = mutableSetOf<String>()
        db.rawQuery("SELECT metadata_key FROM tracks", null).use { cursor ->
            while (cursor.moveToNext()) {
                keys.add(cursor.getString(0))
            }
        }
        return keys
    }

    /**
     * Get all file paths from the tracks table.
     */
    fun getAllFilePaths(): Set<String> {
        val paths = mutableSetOf<String>()
        db.rawQuery("SELECT file_path FROM tracks", null).use { cursor ->
            while (cursor.moveToNext()) {
                paths.add(cursor.getString(0))
            }
        }
        return paths
    }

    /**
     * Get the raw SQLiteDatabase for direct access (e.g., loading projection matrices).
     */
    fun getRawDatabase(): SQLiteDatabase = db

    fun close() {
        db.close()
    }
}

private fun Cursor.getNullableString(columnIndex: Int): String? =
    if (isNull(columnIndex)) null else getString(columnIndex)
