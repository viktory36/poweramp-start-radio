package com.powerampstartradio.indexing.v2

import android.database.sqlite.SQLiteDatabase
import java.io.File

data class V2ProviderSpanReceipt(
    val trackId: Long,
    val providerSpan: V2CommittedProviderSpan,
)

data class V2ProviderSpanReceiptSnapshot(
    val compatibleSchema: Boolean,
    val receipts: List<V2ProviderSpanReceipt>,
    val invalidReceiptCount: Int,
)

/** Read-only projection used by library detection; it never infers identity from track metadata. */
object V2ProviderSpanReceiptReader {
    private val requiredColumns = setOf(
        "receipt_schema_version",
        "track_id",
        "provider_physical_path",
        "provider_offset_ms",
        "provider_duration_ms",
    )

    fun read(databaseFile: File): V2ProviderSpanReceiptSnapshot =
        SQLiteDatabase.openDatabase(
            databaseFile.canonicalPath,
            null,
            SQLiteDatabase.OPEN_READONLY,
        ).use(::read)

    internal fun read(database: SQLiteDatabase): V2ProviderSpanReceiptSnapshot {
        val table = V2EmbeddingCommitRepository.RECEIPT_TABLE
        if (!hasTable(database, table)) {
            return V2ProviderSpanReceiptSnapshot(false, emptyList(), 0)
        }
        val columns = database.rawQuery("PRAGMA table_info('$table')", null).use { cursor ->
            buildSet {
                val nameColumn = cursor.getColumnIndexOrThrow("name")
                while (cursor.moveToNext()) add(cursor.getString(nameColumn))
            }
        }
        if (!columns.containsAll(requiredColumns)) {
            return V2ProviderSpanReceiptSnapshot(false, emptyList(), countRows(database, table))
        }

        val receipts = mutableListOf<V2ProviderSpanReceipt>()
        var invalid = 0
        database.rawQuery(
            """
            SELECT r.receipt_schema_version, r.track_id, r.provider_physical_path,
                   r.provider_offset_ms, r.provider_duration_ms,
                   t.id, e.track_id
            FROM $table r
            LEFT JOIN ${V2EmbeddingCommitRepository.TRACK_TABLE} t ON t.id = r.track_id
            LEFT JOIN ${V2EmbeddingCommitRepository.EMBEDDING_TABLE} e ON e.track_id = r.track_id
            ORDER BY r.track_id
            """.trimIndent(),
            null,
        ).use { cursor ->
            while (cursor.moveToNext()) {
                val schema = cursor.getInt(0)
                val trackId = cursor.getLong(1)
                val rawPath = if (cursor.isNull(2)) "" else cursor.getString(2)
                val offsetMs = cursor.getLong(3)
                val durationMs = cursor.getLong(4)
                val normalizedPath = runCatching {
                    V2StableProviderLexicalPathNormalizer.normalizeAbsolute(rawPath)
                }.getOrNull()
                val valid = schema == V2_EMBEDDING_COMMIT_RECEIPT_SCHEMA_VERSION &&
                    trackId > 0L &&
                    normalizedPath == rawPath &&
                    offsetMs >= 0L &&
                    durationMs >= 0L &&
                    !cursor.isNull(5) &&
                    !cursor.isNull(6)
                if (!valid) {
                    invalid++
                    continue
                }
                receipts += V2ProviderSpanReceipt(
                    trackId = trackId,
                    providerSpan = V2CommittedProviderSpan(rawPath, offsetMs, durationMs),
                )
            }
        }
        return V2ProviderSpanReceiptSnapshot(
            compatibleSchema = true,
            receipts = receipts,
            invalidReceiptCount = invalid,
        )
    }

    private fun hasTable(database: SQLiteDatabase, table: String): Boolean = database.rawQuery(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1",
        arrayOf(table),
    ).use { it.moveToFirst() }

    private fun countRows(database: SQLiteDatabase, table: String): Int = database.rawQuery(
        "SELECT COUNT(*) FROM [$table]",
        null,
    ).use { cursor ->
        check(cursor.moveToFirst())
        cursor.getInt(0)
    }
}
