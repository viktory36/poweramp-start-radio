package com.powerampstartradio.ui

import android.content.ContentValues
import android.database.sqlite.SQLiteDatabase
import com.powerampstartradio.data.EmbeddedTrack
import java.io.File

/** Uncapped confirmed-enqueue memory, independent of the bounded session-history display. */
class RadioSelectionMemory(private val filesDir: File) {
    fun initializeFromHistory(sessions: List<RadioResult>) = withDatabase { database ->
        database.beginTransaction()
        try {
            val initialized = database.rawQuery("SELECT key FROM state", null).use { cursor ->
                buildSet { while (cursor.moveToNext()) add(cursor.getString(0)) }
            }
            if ("initialized" !in initialized || "find_music_initialized" !in initialized) {
                sessions.forEach { session ->
                    val marker = if (session.findMusicSessionEvidence != null) {
                        "find_music_initialized"
                    } else "initialized"
                    if (marker !in initialized) recordSession(database, session)
                }
                markInitialized(database)
            }
            database.setTransactionSuccessful()
        } finally {
            database.endTransaction()
        }
    }

    fun recordSession(session: RadioResult) = withDatabase { database ->
        database.beginTransaction()
        try {
            recordSession(database, session)
            database.setTransactionSuccessful()
        } finally {
            database.endTransaction()
        }
    }

    fun rememberedTrackIds(
        seedTrackId: Long,
        stableSeedId: String?,
        currentGeneration: String,
        resolveTrack: (Long) -> EmbeddedTrack?,
    ): List<Long> = withDatabase { database ->
        val remembered = sortedSetOf<Long>()
        val currentSeed by lazy { resolveTrack(seedTrackId) }
        database.rawQuery(
            "SELECT seed_key, track_id, stable_id, generation_id, artist, title FROM seeds " +
                "WHERE track_id = ? OR stable_id = ?",
            arrayOf(seedTrackId.toString(), stableSeedId.orEmpty()),
        ).use { seeds ->
            while (seeds.moveToNext()) {
                val storedStableId = seeds.getString(2)
                val sameSeed = if (stableSeedId != null && storedStableId != null) {
                    stableSeedId == storedStableId
                } else {
                    seeds.getLong(1) == seedTrackId && (
                        seeds.getString(3) == currentGeneration || currentSeed?.let { current ->
                            current.artist.equals(seeds.getString(4), ignoreCase = true) &&
                                current.title.equals(seeds.getString(5), ignoreCase = true)
                        } == true
                    )
                }
                if (!sameSeed) continue
                remembered += readTracks(database, seeds.getString(0), currentGeneration, resolveTrack)
            }
        }
        remembered.toList()
    }

    fun rememberedSearchTrackIds(
        query: FindMusicQuerySpec,
        currentGeneration: String,
        resolveTrack: (Long) -> EmbeddedTrack?,
    ): List<Long> = withDatabase { database ->
        readTracks(database, FindMusicMemoryKey.of(query), currentGeneration, resolveTrack)
    }

    private fun readTracks(
        database: SQLiteDatabase,
        key: String,
        currentGeneration: String,
        resolveTrack: (Long) -> EmbeddedTrack?,
    ): List<Long> = database.rawQuery(
        "SELECT track_id, generation_id, metadata_key, filename_key, duration_ms " +
            "FROM tracks WHERE seed_key = ? ORDER BY track_id",
        arrayOf(key),
    ).use { tracks ->
        buildList {
            while (tracks.moveToNext()) {
                val trackId = tracks.getLong(0)
                if (tracks.getString(1) != currentGeneration) {
                    val current = resolveTrack(trackId) ?: continue
                    if (current.metadataKey != tracks.getString(2) ||
                        current.filenameKey != tracks.getString(3) ||
                        current.durationMs != tracks.getInt(4)
                    ) continue
                }
                add(trackId)
            }
        }
    }

    /** Clear every seed and search, retaining markers so history cannot refill memory. */
    fun clear() = withDatabase { database ->
        database.beginTransaction()
        try {
            database.delete("tracks", null, null)
            database.delete("seeds", null, null)
            markInitialized(database)
            database.setTransactionSuccessful()
        } finally {
            database.endTransaction()
        }
    }

    private fun recordSession(database: SQLiteDatabase, session: RadioResult) {
        session.findMusicSessionEvidence?.querySpec?.let { query ->
            val generation = session.generation?.generationId ?: return
            recordTracks(database, FindMusicMemoryKey.of(query), generation, session.tracks)
            return
        }
        if (session.isDirectQueue || session.composedQuerySpec != null) return
        val seed = session.seedIdentity ?: return
        val generation = session.generation?.generationId ?: return
        val queued = session.tracks.filter { it.status == QueueStatus.QUEUED }
        if (queued.isEmpty()) return
        val key = seed.stableTrackSpanId?.let { "stable:$it" }
            ?: "legacy:$generation:${seed.embeddedTrackId}"
        database.insertWithOnConflict("seeds", null, ContentValues().apply {
            put("seed_key", key)
            put("track_id", seed.embeddedTrackId)
            put("stable_id", seed.stableTrackSpanId)
            put("generation_id", generation)
            put("artist", session.seedTrack.artist)
            put("title", session.seedTrack.title)
        }, SQLiteDatabase.CONFLICT_REPLACE).also { check(it != -1L) { "Could not remember the seed" } }
        recordTracks(database, key, generation, queued)
    }

    private fun recordTracks(
        database: SQLiteDatabase,
        key: String,
        generation: String,
        rows: List<QueuedTrackResult>,
    ) {
        for (row in rows.filter { it.status == QueueStatus.QUEUED }) {
            database.insertWithOnConflict("tracks", null, ContentValues().apply {
                put("seed_key", key)
                put("track_id", row.track.id)
                put("generation_id", generation)
                put("metadata_key", row.track.metadataKey)
                put("filename_key", row.track.filenameKey)
                put("duration_ms", row.track.durationMs)
            }, SQLiteDatabase.CONFLICT_REPLACE).also {
                check(it != -1L) { "Could not remember an enqueued recording" }
            }
        }
    }

    private fun markInitialized(database: SQLiteDatabase) {
        database.execSQL("INSERT OR IGNORE INTO state(key) VALUES ('initialized')")
        database.execSQL("INSERT OR IGNORE INTO state(key) VALUES ('find_music_initialized')")
    }

    private fun <T> withDatabase(block: (SQLiteDatabase) -> T): T = synchronized(LOCK) {
        SQLiteDatabase.openOrCreateDatabase(File(filesDir, FILE_NAME), null).use { database ->
            if (database.version == 0) {
                database.beginTransaction()
                try {
                    database.execSQL("CREATE TABLE state (key TEXT PRIMARY KEY)")
                    database.execSQL(
                        "CREATE TABLE seeds (seed_key TEXT PRIMARY KEY, track_id INTEGER NOT NULL, " +
                            "stable_id TEXT, generation_id TEXT NOT NULL, artist TEXT, title TEXT)",
                    )
                    database.execSQL("CREATE INDEX seed_track_id ON seeds(track_id)")
                    database.execSQL("CREATE INDEX seed_stable_id ON seeds(stable_id)")
                    database.execSQL(
                        "CREATE TABLE tracks (seed_key TEXT NOT NULL, track_id INTEGER NOT NULL, " +
                            "generation_id TEXT NOT NULL, metadata_key TEXT NOT NULL, " +
                            "filename_key TEXT NOT NULL, duration_ms INTEGER NOT NULL, " +
                            "PRIMARY KEY(seed_key, track_id))",
                    )
                    database.version = 1
                    database.setTransactionSuccessful()
                } finally {
                    database.endTransaction()
                }
            }
            block(database)
        }
    }

    companion object {
        private val LOCK = Any()
        const val FILE_NAME = "radio-selection-memory.db"
    }
}
