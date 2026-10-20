package com.powerampstartradio.indexing.v2

import android.database.sqlite.SQLiteDatabase
import android.util.AtomicFile
import com.google.gson.Gson
import java.io.File

/** Small generation-local addition list. Reads only row IDs, never music files or vector blobs. */
object FavoritesIndexUpdate {
    private const val FILE = "favorites-additions.json"
    private val gson = Gson()

    fun prepare(filesDir: File, previousId: String?, nextId: String) {
        val root = File(filesDir, "indexing_v2/generations")
        val destination = File(root, nextId)
        val output = AtomicFile(File(destination, FILE))
        // A retried pointer publication must preserve the original update boundary.
        if (output.baseFile.exists()) return
        val previous = previousId?.let { File(File(root, it), "library.db") }
        val added = if (previous?.isFile == true) {
            SQLiteDatabase.openDatabase(File(destination, "library.db").path, null,
                SQLiteDatabase.OPEN_READONLY).use { db ->
                db.execSQL("ATTACH DATABASE ? AS previous_index", arrayOf(previous.path))
                db.rawQuery(
                    "SELECT n.track_id FROM embeddings_clamp3 n " +
                        "LEFT JOIN previous_index.embeddings_clamp3 p ON p.track_id=n.track_id " +
                        "WHERE p.track_id IS NULL ORDER BY n.track_id", null,
                ).use { cursor -> buildList { while (cursor.moveToNext()) add(cursor.getLong(0)) } }
            }
        } else emptyList()
        val stream = output.startWrite()
        try {
            stream.write(gson.toJson(added).toByteArray(Charsets.UTF_8))
            output.finishWrite(stream)
        } catch (failure: Throwable) {
            output.failWrite(stream)
            throw failure
        }
    }

    fun read(generationDirectory: File): Set<Long> {
        val file = AtomicFile(File(generationDirectory, FILE))
        return try {
            file.openRead().bufferedReader(Charsets.UTF_8).use {
                gson.fromJson(it, LongArray::class.java).toSet()
            }
        } catch (_: java.io.FileNotFoundException) { emptySet() }
    }
}
