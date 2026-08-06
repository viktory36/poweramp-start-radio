package com.powerampstartradio.data

import android.content.ContentValues
import android.database.sqlite.SQLiteDatabase
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class EmbeddingDatabaseCatalogProjectionInstrumentedTest {
    @Test
    fun catalogProjectionPreservesTrackOrderAndRawValuesWithoutAnIntermediateTrackList() =
        withDatabase(
            listOf(
                Row(30L, "Beta", "A", "Zulu", 30_300, "/Music/beta.flac"),
                Row(10L, null, null, "Second", 10_100, "/Music/second.flac"),
                Row(20L, null, "", "First", 20_200, "/Music/first.flac"),
                Row(40L, "alpha", "Z", null, 40_400, "/Music/alpha.flac"),
            ),
        ) { database ->
            val expected = database.getAllTracks().map { track ->
                Projection(
                    track.id,
                    track.artist,
                    track.album,
                    track.title,
                    track.durationMs,
                    track.filePath,
                )
            }
            val reportedCounts = mutableListOf<Int>()

            val projected = database.mapAllTrackCatalogRows(
                onRowCount = reportedCounts::add,
            ) { trackId, artist, album, title, durationMs, filePath ->
                Projection(trackId, artist, album, title, durationMs, filePath)
            }

            assertEquals(listOf(20L, 10L, 40L, 30L), projected.map(Projection::trackId))
            assertEquals(expected, projected)
            assertEquals(listOf(projected.size), reportedCounts)
        }

    @Test
    fun emptyCatalogProjectionReportsZeroAndReturnsNoRows() = withDatabase(emptyList()) { database ->
        val reportedCounts = mutableListOf<Int>()

        val projected = database.mapAllTrackCatalogRows(
            onRowCount = reportedCounts::add,
        ) { trackId, artist, album, title, durationMs, filePath ->
            Projection(trackId, artist, album, title, durationMs, filePath)
        }

        assertEquals(emptyList<Projection>(), projected)
        assertEquals(listOf(0), reportedCounts)
    }

    private fun withDatabase(rows: List<Row>, block: (EmbeddingDatabase) -> Unit) {
        val context = ApplicationProvider.getApplicationContext<android.content.Context>()
        val file = File(context.cacheDir, "catalog-projection-${System.nanoTime()}.db")
        try {
            SQLiteDatabase.openOrCreateDatabase(file, null).use { database ->
                database.execSQL(
                    """
                    CREATE TABLE tracks (
                        id INTEGER PRIMARY KEY,
                        metadata_key TEXT NOT NULL,
                        filename_key TEXT NOT NULL,
                        artist TEXT,
                        album TEXT,
                        title TEXT,
                        duration_ms INTEGER NOT NULL,
                        file_path TEXT NOT NULL
                    )
                    """.trimIndent(),
                )
                rows.forEach { row ->
                    database.insertOrThrow(
                        "tracks",
                        null,
                        ContentValues().apply {
                            put("id", row.id)
                            put("metadata_key", "metadata-${row.id}")
                            put("filename_key", "filename-${row.id}")
                            if (row.artist == null) putNull("artist") else put("artist", row.artist)
                            if (row.album == null) putNull("album") else put("album", row.album)
                            if (row.title == null) putNull("title") else put("title", row.title)
                            put("duration_ms", row.durationMs)
                            put("file_path", row.filePath)
                        },
                    )
                }
            }
            val database = EmbeddingDatabase.open(file)
            try {
                block(database)
            } finally {
                database.close()
            }
        } finally {
            file.delete()
        }
    }

    private data class Row(
        val id: Long,
        val artist: String?,
        val album: String?,
        val title: String?,
        val durationMs: Int,
        val filePath: String,
    )

    private data class Projection(
        val trackId: Long,
        val artist: String?,
        val album: String?,
        val title: String?,
        val durationMs: Int,
        val filePath: String,
    )
}
