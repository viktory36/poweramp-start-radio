package com.powerampstartradio.ui

import android.content.ContentValues
import android.database.sqlite.SQLiteDatabase
import com.google.gson.Gson
import java.io.File

data class FavoriteSeed(
    val key: String,
    val text: String? = null,
    val trackId: Long? = null,
    val stableTrackSpanId: String? = null,
    val title: String? = null,
    val artist: String? = null,
    /** Transient Now Playing lookup, resolved to an embedding identity before enqueueing. */
    val powerampFileId: Long? = null,
) {
    val label: String get() = text ?: title ?: "Track $trackId"
}

data class FavoriteEntry(val seed: FavoriteSeed, val count: Int, val lastRequestedAt: Long)

/** Request IDs make history import and service recovery idempotent. History deletion is separate. */
class FavoritesStore(private val filesDir: File) {
    fun initializeFromHistory(history: List<RadioResult>) = synchronized(LOCK) {
        open().use { db ->
            val initialized = db.rawQuery("SELECT value FROM state WHERE name='history_imported'", null)
                .use { it.moveToFirst() }
            if (initialized) return@synchronized
            db.beginTransaction()
            try {
                history.forEach { insertSession(db, it) }
                db.execSQL("INSERT INTO state(name,value) VALUES ('history_imported',1)")
                db.setTransactionSuccessful()
            } finally { db.endTransaction() }
        }
    }

    fun recordSession(session: RadioResult) = synchronized(LOCK) {
        open().use { db ->
            db.beginTransaction()
            try {
                insertSession(db, session)
                db.setTransactionSuccessful()
            } finally {
                db.endTransaction()
            }
        }
    }

    private fun insertSession(db: SQLiteDatabase, session: RadioResult) {
        if (session.tracks.none { it.status == QueueStatus.QUEUED }) return
        val seeds = seedsFrom(session)
        val requestId = session.requestId ?: "history:${session.timestamp}:${seeds.map { it.key }}"
        seeds.distinctBy { it.key }.forEach { seed ->
            db.insertWithOnConflict("seeds", null, ContentValues().apply {
                put("seed_key", seed.key)
                put("seed_json", gson.toJson(seed))
            }, SQLiteDatabase.CONFLICT_REPLACE)
            db.insertWithOnConflict("requests", null, ContentValues().apply {
                put("request_id", requestId)
                put("seed_key", seed.key)
                put("requested_at", session.timestamp)
            }, SQLiteDatabase.CONFLICT_IGNORE)
        }
    }

    fun entries(): List<FavoriteEntry> = synchronized(LOCK) {
        open().use { db ->
            db.rawQuery(
                "SELECT s.seed_json, COUNT(*), MAX(r.requested_at) FROM seeds s " +
                    "JOIN requests r ON r.seed_key=s.seed_key GROUP BY s.seed_key " +
                    "ORDER BY COUNT(*) DESC, MAX(r.requested_at) DESC, s.seed_key ASC",
                null,
            ).use { cursor ->
                buildList {
                    while (cursor.moveToNext()) add(FavoriteEntry(
                        gson.fromJson(cursor.getString(0), FavoriteSeed::class.java),
                        cursor.getInt(1), cursor.getLong(2),
                    ))
                }
            }
        }
    }

    fun newMatchFlags(contextKey: String): Map<String, Boolean> = synchronized(LOCK) {
        open().use { db ->
            db.rawQuery("SELECT seed_key, has_new FROM new_matches WHERE context_key=?",
                arrayOf(contextKey)).use { cursor ->
                buildMap { while (cursor.moveToNext()) put(cursor.getString(0), cursor.getInt(1) != 0) }
            }
        }
    }

    fun saveNewMatchFlags(contextKey: String, flags: Map<String, Boolean>) = synchronized(LOCK) {
        if (flags.isEmpty()) return@synchronized
        open().use { db ->
            db.beginTransaction()
            try {
                flags.forEach { (seedKey, hasNew) ->
                    db.insertWithOnConflict("new_matches", null, ContentValues().apply {
                        put("seed_key", seedKey)
                        put("context_key", contextKey)
                        put("has_new", if (hasNew) 1 else 0)
                    }, SQLiteDatabase.CONFLICT_REPLACE)
                }
                db.setTransactionSuccessful()
            } finally { db.endTransaction() }
        }
    }

    private fun open(): SQLiteDatabase {
        filesDir.mkdirs()
        return SQLiteDatabase.openOrCreateDatabase(File(filesDir, "favorites.db"), null).also { db ->
            db.execSQL("CREATE TABLE IF NOT EXISTS seeds (seed_key TEXT PRIMARY KEY, seed_json TEXT NOT NULL)")
            db.execSQL("CREATE TABLE IF NOT EXISTS requests (request_id TEXT NOT NULL, " +
                "seed_key TEXT NOT NULL, requested_at INTEGER NOT NULL, PRIMARY KEY(request_id,seed_key))")
            db.execSQL("CREATE TABLE IF NOT EXISTS state (name TEXT PRIMARY KEY, value INTEGER NOT NULL)")
            db.execSQL("CREATE TABLE IF NOT EXISTS new_matches (seed_key TEXT PRIMARY KEY, " +
                "context_key TEXT NOT NULL, has_new INTEGER NOT NULL)")
        }
    }

    companion object {
        private val LOCK = Any()
        private val gson = Gson()

        internal fun seedsFrom(session: RadioResult): List<FavoriteSeed> {
            session.favoriteQueueEvidence?.let { return listOf(it.seed) }
            val query = session.findMusicSessionEvidence?.querySpec ?: session.composedQuerySpec
            if (query != null) return buildList {
                query.activeTextIngredients.forEach {
                    val text = it.query.trim()
                    add(FavoriteSeed(key = "text:$text", text = text))
                }
                query.songSeeds.filter { it.weight > 0f }.forEach {
                    add(FavoriteSeed("song:${it.trackId}", trackId = it.trackId,
                        stableTrackSpanId = it.stableTrackSpanId, title = it.title, artist = it.artist))
                }
            }
            if (session.isDirectQueue) return emptyList()
            val identity = session.seedIdentity ?: return emptyList()
            return listOf(FavoriteSeed("song:${identity.embeddedTrackId}",
                trackId = identity.embeddedTrackId, stableTrackSpanId = identity.stableTrackSpanId,
                title = session.seedTrack.title, artist = session.seedTrack.artist))
        }
    }
}
