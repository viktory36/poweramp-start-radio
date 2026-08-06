package com.powerampstartradio.data

import android.content.ContentValues
import android.database.sqlite.SQLiteDatabase
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

@RunWith(AndroidJUnit4::class)
class EmbeddingDatabaseTextSearchInstrumentedTest {
    @Test
    fun punctuationIsASeparatorAndExactArtistTitleComesFirst() = withDatabase(
        listOf(
            Row(10L, "Pink Floyd", "Echoes"),
            Row(20L, "Pink Floyd", "Echoes (Remastered)"),
            Row(30L, "Echoes Ensemble", "Pink Floyd Tribute"),
        ),
    ) { database ->
        val result = database.searchTracksByTextPage(
            query = "Pink Floyd - Echoes",
            limit = 10,
        )

        assertEquals(listOf(10L, 20L, 30L), result.tracks.map(EmbeddedTrack::id))
        assertFalse(result.hasMore)
    }

    @Test
    fun eligibilityAndIdentityCollapseHappenBeforeTheDisplayBound() = withDatabase(
        listOf(
            Row(1L, "", "Ambient"),
            Row(2L, "B", "Ambient A"),
            Row(3L, "C", "Ambient B"),
            Row(4L, "D", "Ambient C"),
            Row(5L, "E", "Ambient D"),
        ),
    ) { database ->
        val result = database.searchTracksByTextPage(
            query = "ambient",
            limit = 2,
            includeTrackId = { it != 1L },
            canonicalTrackId = { if (it == 2L || it == 3L) 200L else it },
        )

        assertEquals(listOf(2L, 4L), result.tracks.map(EmbeddedTrack::id))
        assertTrue(result.hasMore)
    }

    @Test
    fun combiningMarksRemainPartOfNonEnglishRecordingNames() = withDatabase(
        listOf(
            Row(
                1L,
                "\u0915\u0948\u0932\u093e\u0936 \u0916\u0947\u0930",
                "\u0924\u0947\u0930\u0940 \u0926\u0940\u0935\u093e\u0928\u0940",
            ),
        ),
    ) { database ->
        val result = database.searchTracksByTextPage(
            query = "\u0915\u0948\u0932\u093e\u0936 - \u0926\u0940\u0935\u093e\u0928\u0940",
            limit = 10,
        )

        assertEquals(listOf(1L), result.tracks.map(EmbeddedTrack::id))
    }

    private fun withDatabase(rows: List<Row>, block: (EmbeddingDatabase) -> Unit) {
        val context = ApplicationProvider.getApplicationContext<android.content.Context>()
        val file = File(context.cacheDir, "recording-text-search-${System.nanoTime()}.db")
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
                            put("filename_key", "file-${row.id}")
                            put("artist", row.artist)
                            put("album", "Album")
                            put("title", row.title)
                            put("duration_ms", 180_000)
                            put("file_path", "C:\\Music\\${row.id}.flac")
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
        val artist: String,
        val title: String,
    )
}
