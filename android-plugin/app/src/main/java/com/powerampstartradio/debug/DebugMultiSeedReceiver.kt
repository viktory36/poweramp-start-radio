package com.powerampstartradio.debug

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.util.Log
import com.powerampstartradio.data.EmbeddingDatabase
import com.powerampstartradio.data.EmbeddingIndex
import com.powerampstartradio.similarity.algorithms.GeoMeanSelector
import java.io.File
import kotlin.concurrent.thread

/**
 * Debug receiver for automated multi-seed search testing via ADB.
 *
 * Songs are specified as "artist title" strings. Weights can be negative (repel).
 *
 * Usage (up to 4 song seeds):
 *   adb shell am broadcast -a com.powerampstartradio.DEBUG_MULTI_SEED \
 *     -n com.powerampstartradio/.debug.DebugMultiSeedReceiver \
 *     --es song1 "vandou tumblack" --ef weight1 1.0 \
 *     --es song2 "time pachanga boys" --ef weight2 -0.5 \
 *     --ei top_k 10
 */
class DebugMultiSeedReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        val topK = intent.getIntExtra("top_k", 10)

        // Collect song seeds
        data class SongSeed(val query: String, val weight: Float)
        val seeds = mutableListOf<SongSeed>()
        for (i in 1..4) {
            val song = intent.getStringExtra("song$i") ?: continue
            val weight = intent.getFloatExtra("weight$i", 1.0f)
            seeds.add(SongSeed(song, weight))
        }

        // Search-only mode: look up track names without running multi-seed
        val lookupQuery = intent.getStringExtra("lookup")
        if (lookupQuery != null) {
            thread(name = "debug-lookup") {
                val dbFile = File(context.filesDir, "embeddings.db")
                if (!dbFile.exists()) { Log.e("DebugMultiSeed", "No DB"); return@thread }
                val db = EmbeddingDatabase.open(dbFile)
                val results = db.searchTracksByText(lookupQuery, limit = 10)
                db.close()
                Log.i("DebugMultiSeed", "Lookup '$lookupQuery': ${results.size} matches")
                results.forEach { Log.i("DebugMultiSeed", "  ${it.artist} - ${it.title} (id=${it.id})") }
            }
            return
        }

        if (seeds.isEmpty()) {
            Log.e("DebugMultiSeed", "No song seeds provided (use --es song1 \"query\" --ef weight1 1.0)")
            return
        }

        Log.i("DebugMultiSeed", "=== Multi-seed search: ${seeds.size} seeds, topK=$topK ===")
        seeds.forEachIndexed { i, s ->
            Log.i("DebugMultiSeed", "  seed[$i]: '${s.query}' weight=${s.weight}")
        }

        thread(name = "debug-multiseed") {
            try {
                val filesDir = context.filesDir
                val dbFile = File(filesDir, "embeddings.db")
                if (!dbFile.exists()) {
                    Log.e("DebugMultiSeed", "embeddings.db not found")
                    return@thread
                }

                val db = EmbeddingDatabase.open(dbFile)

                // Look up each song seed
                val seedPairs = mutableListOf<Pair<FloatArray, Float>>()
                val excludeIds = mutableSetOf<Long>()
                val resolvedNames = mutableListOf<String>()

                for ((i, seed) in seeds.withIndex()) {
                    val matches = db.searchTracksByText(seed.query, limit = 3)
                    if (matches.isEmpty()) {
                        Log.w("DebugMultiSeed", "  seed[$i] '${seed.query}': NO MATCH FOUND")
                        continue
                    }
                    val track = matches.first()
                    val embedding = db.getEmbedding(track.id)
                    if (embedding == null) {
                        Log.w("DebugMultiSeed", "  seed[$i] '${seed.query}': no embedding for track ${track.id}")
                        continue
                    }
                    Log.i("DebugMultiSeed", "  seed[$i] resolved: '${track.artist} - ${track.title}' (id=${track.id})")
                    seedPairs.add(embedding to seed.weight)
                    excludeIds.add(track.id)
                    resolvedNames.add("${track.artist} - ${track.title}")
                }
                db.close()

                if (seedPairs.isEmpty()) {
                    Log.e("DebugMultiSeed", "No valid seeds resolved")
                    return@thread
                }

                // Load embedding index
                val embFile = File(filesDir, "clamp3.emb")
                if (!embFile.exists()) {
                    val db2 = EmbeddingDatabase.open(dbFile)
                    EmbeddingIndex.extractFromDatabase(db2, embFile, table = "embeddings_clamp3")
                    db2.close()
                }
                val index = EmbeddingIndex.mmap(embFile)

                // Run GeoMeanSelector
                val t0 = System.currentTimeMillis()
                val ranking = GeoMeanSelector.computeRanking(index, seedPairs, topK, excludeIds)
                val elapsed = System.currentTimeMillis() - t0

                // Resolve result metadata
                val db3 = EmbeddingDatabase.open(dbFile)
                Log.i("DebugMultiSeed", "--- Results (${elapsed}ms) ---")
                for ((rank, pair) in ranking.withIndex()) {
                    val (trackId, score) = pair
                    val track = db3.getTrackById(trackId)
                    val name = if (track != null) "${track.artist} - ${track.title}" else "id=$trackId"
                    Log.i("DebugMultiSeed", "  #${rank + 1}: $name  score=%.4f".format(score))
                }
                db3.close()

                Log.i("DebugMultiSeed", "=== Done ===")
            } catch (e: Exception) {
                Log.e("DebugMultiSeed", "Search failed", e)
            }
        }
    }
}
