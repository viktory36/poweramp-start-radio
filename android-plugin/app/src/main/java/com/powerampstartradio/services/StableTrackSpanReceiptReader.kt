package com.powerampstartradio.services

import android.database.sqlite.SQLiteDatabase
import com.powerampstartradio.indexing.v2.V2EmbeddingCommitRepository
import java.io.File

internal object StableTrackSpanReceiptReader {
    fun read(databaseFile: File, trackId: Long, embeddingSpecId: String): String? =
        readMany(databaseFile, setOf(trackId), embeddingSpecId)[trackId]

    fun readMany(
        databaseFile: File,
        trackIds: Set<Long>,
        embeddingSpecId: String,
    ): Map<Long, String?> {
        if (trackIds.isEmpty()) return emptyMap()
        val database = SQLiteDatabase.openDatabase(
            databaseFile.absolutePath,
            null,
            SQLiteDatabase.OPEN_READONLY,
        )
        return database.use { db ->
            val receiptTable = V2EmbeddingCommitRepository.RECEIPT_TABLE
            val hasReceiptTable = db.rawQuery(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1",
                arrayOf(receiptTable),
            ).use { cursor -> cursor.moveToFirst() }
            if (!hasReceiptTable) {
                return@use trackIds.associateWith { null }
            }
            trackIds.associateWith { trackId ->
                db.rawQuery(
                    """
                    SELECT stable_track_span_id
                    FROM $receiptTable
                    WHERE track_id = ? AND receipt_schema_version = ? AND embedding_spec_id = ?
                    """.trimIndent(),
                    arrayOf(
                        trackId.toString(),
                        V2EmbeddingCommitRepository.RECEIPT_SCHEMA_VERSION.toString(),
                        embeddingSpecId,
                    ),
                ).use { cursor ->
                    if (cursor.moveToFirst() && !cursor.isNull(0)) cursor.getString(0) else null
                }
            }
        }
    }
}
