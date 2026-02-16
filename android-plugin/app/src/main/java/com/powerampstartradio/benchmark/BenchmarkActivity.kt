package com.powerampstartradio.benchmark

import android.content.Context
import android.os.Build
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.google.gson.GsonBuilder
import com.powerampstartradio.indexing.*
import com.powerampstartradio.poweramp.PowerampHelper
import kotlinx.coroutines.*
import java.io.File

/**
 * Standalone benchmark activity for testing ONNX inference on-device.
 *
 * Runs MuLan and Flamingo on a few Poweramp tracks, reports timing and
 * execution provider, and saves full embeddings as JSON for desktop comparison.
 *
 * Launch via:
 *   adb shell am start -n com.powerampstartradio/.benchmark.BenchmarkActivity
 *
 * Pull results via:
 *   adb pull /data/data/com.powerampstartradio/files/benchmark_results.json
 */
class BenchmarkActivity : ComponentActivity() {

    companion object {
        private const val TAG = "EmbeddingBenchmark"
        private const val MAX_TRACKS = 5
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContent {
            MaterialTheme {
                BenchmarkScreen()
            }
        }
    }

    @Composable
    private fun BenchmarkScreen() {
        var status by remember { mutableStateOf("Ready. Tap 'Run Benchmark' to start.") }
        var running by remember { mutableStateOf(false) }
        val scope = rememberCoroutineScope()

        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp)
        ) {
            Text("ONNX Embedding Benchmark", style = MaterialTheme.typography.headlineSmall)
            Spacer(Modifier.height(8.dp))

            Text(
                "Device: ${Build.MANUFACTURER} ${Build.MODEL}\n" +
                "SOC: ${Build.SOC_MODEL}\n" +
                "Android: ${Build.VERSION.RELEASE} (SDK ${Build.VERSION.SDK_INT})",
                fontSize = 12.sp,
                fontFamily = FontFamily.Monospace,
            )
            Spacer(Modifier.height(16.dp))

            Button(
                onClick = {
                    running = true
                    status = "Starting benchmark..."
                    scope.launch(Dispatchers.IO) {
                        try {
                            runBenchmark { msg ->
                                status = msg
                            }
                        } catch (e: Exception) {
                            status = "ERROR: ${e.message}\n\n${e.stackTraceToString()}"
                        } finally {
                            running = false
                        }
                    }
                },
                enabled = !running,
            ) {
                Text(if (running) "Running..." else "Run Benchmark")
            }

            Spacer(Modifier.height(16.dp))

            // Scrollable monospace output
            Text(
                text = status,
                fontSize = 11.sp,
                fontFamily = FontFamily.Monospace,
                lineHeight = 15.sp,
                modifier = Modifier
                    .fillMaxSize()
                    .verticalScroll(rememberScrollState())
                    .horizontalScroll(rememberScrollState()),
            )
        }
    }

    /**
     * Query Poweramp for tracks with file paths.
     */
    private fun queryTracksWithPaths(): List<TestTrack> {
        val filesUri = PowerampHelper.ROOT_URI.buildUpon()
            .appendEncodedPath("files").build()
        val result = mutableListOf<TestTrack>()

        // Try with path column first, fall back to folder_path + file_name
        val columnsWithPath = arrayOf(
            "folder_files._id", "artist", "album", "title_tag",
            "folder_files.duration", "path"
        )
        val columnsWithoutPath = arrayOf(
            "folder_files._id", "artist", "album", "title_tag",
            "folder_files.duration", "folder_path", "file_name"
        )

        try {
            // Try with 'path' column
            var cursor = try {
                contentResolver.query(filesUri, columnsWithPath, null, null, null)
            } catch (e: Exception) {
                Log.w(TAG, "path column not available, trying folder_path + file_name")
                null
            }

            val usePath = cursor != null
            if (cursor == null) {
                cursor = try {
                    contentResolver.query(filesUri, columnsWithoutPath, null, null, null)
                } catch (e: Exception) {
                    Log.w(TAG, "folder_path + file_name not available either, trying minimal")
                    // Last resort: no path columns at all
                    contentResolver.query(
                        filesUri,
                        arrayOf("folder_files._id", "artist", "title_tag", "folder_files.duration"),
                        null, null, null
                    )
                }
            }

            cursor?.use {
                val idIdx = it.getColumnIndex("_id")
                val artistIdx = it.getColumnIndex("artist")
                val titleIdx = it.getColumnIndex("title_tag")
                val pathIdx = it.getColumnIndex("path")
                val folderPathIdx = it.getColumnIndex("folder_path")
                val fileNameIdx = it.getColumnIndex("file_name")

                while (it.moveToNext()) {
                    val path = when {
                        usePath && pathIdx >= 0 -> it.getString(pathIdx)
                        folderPathIdx >= 0 && fileNameIdx >= 0 -> {
                            val folder = it.getString(folderPathIdx) ?: ""
                            val file = it.getString(fileNameIdx) ?: ""
                            if (file.isNotEmpty()) "$folder/$file" else null
                        }
                        else -> null
                    }
                    if (path != null) {
                        result.add(TestTrack(
                            id = it.getLong(idIdx),
                            artist = it.getString(artistIdx) ?: "",
                            title = it.getString(titleIdx) ?: "",
                            path = path,
                        ))
                    }
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error querying Poweramp", e)
        }

        return result
    }

    private suspend fun runBenchmark(onStatus: (String) -> Unit) {
        val sb = StringBuilder()
        fun log(msg: String) {
            sb.appendLine(msg)
            Log.i(TAG, msg)
            onStatus(sb.toString())
        }

        log("=== ONNX Embedding Benchmark ===")
        log("Device: ${Build.MANUFACTURER} ${Build.MODEL}")
        log("SOC: ${Build.SOC_MODEL}")
        log("")

        // Discover tracks from Poweramp
        log("Querying Poweramp library...")
        val allTracks = withContext(Dispatchers.Main) { queryTracksWithPaths() }
        if (allTracks.isEmpty()) {
            log("ERROR: No tracks found in Poweramp library.")
            log("Make sure Poweramp is installed and has scanned your library.")
            return
        }
        log("Found ${allTracks.size} tracks in Poweramp")

        // Pick random tracks that are resolvable
        val testTracks = allTracks.shuffled().filter { resolveFile(it.path) != null }.take(MAX_TRACKS)
        if (testTracks.isEmpty()) {
            log("ERROR: Could not resolve any audio file paths.")
            log("Sample paths: ${allTracks.take(3).map { it.path }}")
            return
        }
        log("Selected ${testTracks.size} random tracks for benchmark\n")

        val mulanFile = File(filesDir, "mulan_audio.onnx")
        val flamingoFile = File(filesDir, "flamingo_encoder.onnx")
        val projectorFile = File(filesDir, "flamingo_projector.onnx")

        val results = mutableListOf<TrackResult>()
        // Pre-populate results list so both passes write to the same entries
        for (track in testTracks) {
            results.add(TrackResult(
                path = track.path,
                artist = track.artist,
                title = track.title,
            ))
        }

        val decoder = AudioDecoder()

        // ── MuLan Pass ──
        if (mulanFile.exists()) {
            log("Loading MuLan model...")
            val loadStart = System.nanoTime()
            var mulanInference: MuLanInference? = null
            try {
                mulanInference = MuLanInference(mulanFile, filesDir)
                val loadMs = (System.nanoTime() - loadStart) / 1_000_000
                log("  MuLan loaded in ${loadMs}ms, EP: ${mulanInference.executionProvider}")
            } catch (e: Exception) {
                log("  MuLan load FAILED: ${e.message}")
            }

            if (mulanInference != null) {
                log("")
                for ((i, track) in testTracks.withIndex()) {
                    log("MuLan [${i + 1}/${testTracks.size}] ${track.artist} - ${track.title}")

                    val audioFile = resolveFile(track.path)!!

                    val audio = decoder.decode(audioFile, 24000)
                    if (audio == null) {
                        log("  Decode failed")
                        continue
                    }
                    log("  Audio: ${audio.durationS}s @ ${audio.sampleRate}Hz")

                    val start = System.nanoTime()
                    val embedding = mulanInference.generateEmbedding(audio)
                    val inferMs = (System.nanoTime() - start) / 1_000_000

                    if (embedding != null) {
                        log("  Embedding: ${embedding.size}d in ${inferMs}ms")
                        log("  First 5: ${embedding.take(5).map { "%.4f".format(it) }}")
                        results[i].mulan = EmbeddingResult(
                            ep = mulanInference.executionProvider,
                            timingMs = inferMs,
                            dim = embedding.size,
                            embedding = embedding.toList(),
                        )
                    } else {
                        log("  Inference FAILED")
                    }
                }
                mulanInference.close()
                log("\nMuLan session closed.")
            }
        } else {
            log("MuLan model not found at ${mulanFile.absolutePath}")
        }

        // ── Flamingo Pass ──
        if (flamingoFile.exists()) {
            log("\nLoading Flamingo model...")
            val loadStart = System.nanoTime()
            var flamingoInference: FlamingoInference? = null
            try {
                flamingoInference = FlamingoInference(flamingoFile, projectorFile, filesDir)
                val loadMs = (System.nanoTime() - loadStart) / 1_000_000
                log("  Flamingo loaded in ${loadMs}ms, EP: ${flamingoInference.executionProvider}")
                log("  Output dim: ${flamingoInference.outputDim}")
            } catch (e: Exception) {
                log("  Flamingo load FAILED: ${e.message}")
            }

            if (flamingoInference != null) {
                log("")
                for ((i, track) in testTracks.withIndex()) {
                    log("Flamingo [${i + 1}/${testTracks.size}] ${track.artist} - ${track.title}")

                    val audioFile = resolveFile(track.path)!!

                    val audio = decoder.decode(audioFile, 16000)
                    if (audio == null) {
                        log("  Decode failed")
                        continue
                    }
                    log("  Audio: ${audio.durationS}s @ ${audio.sampleRate}Hz")

                    val start = System.nanoTime()
                    val embedding = flamingoInference.generateEmbedding(audio)
                    val inferMs = (System.nanoTime() - start) / 1_000_000

                    if (embedding != null) {
                        log("  Embedding: ${embedding.size}d in ${inferMs}ms")
                        log("  First 5: ${embedding.take(5).map { "%.4f".format(it) }}")
                        results[i].flamingo = EmbeddingResult(
                            ep = flamingoInference.executionProvider,
                            timingMs = inferMs,
                            dim = embedding.size,
                            embedding = embedding.toList(),
                        )
                    } else {
                        log("  Inference FAILED")
                    }
                }
                flamingoInference.close()
                log("\nFlamingo session closed.")
            }
        } else {
            log("\nFlamingo model not found at ${flamingoFile.absolutePath}")
        }

        // ── Save results as JSON ──
        val output = BenchmarkOutput(
            device = "${Build.MANUFACTURER} ${Build.MODEL}",
            soc = Build.SOC_MODEL,
            androidVersion = "${Build.VERSION.RELEASE} (SDK ${Build.VERSION.SDK_INT})",
            ortVersion = "1.23.2",
            tracks = results,
        )

        val gson = GsonBuilder().setPrettyPrinting().create()
        val json = gson.toJson(output)
        val outputFile = File(filesDir, "benchmark_results.json")
        outputFile.writeText(json)

        log("\n=== Results saved ===")
        log("File: ${outputFile.absolutePath}")
        log("Pull via: adb pull ${outputFile.absolutePath}")
        log("\n=== Summary ===")
        for (r in results) {
            log("${r.artist} - ${r.title}")
            r.mulan?.let { log("  MuLan: ${it.dim}d, ${it.timingMs}ms, EP=${it.ep}") }
            r.flamingo?.let { log("  Flamingo: ${it.dim}d, ${it.timingMs}ms, EP=${it.ep}") }
        }
        log("\nBenchmark complete.")
    }

    private fun resolveFile(path: String): File? {
        val direct = File(path)
        if (direct.exists() && direct.canRead()) return direct
        for (prefix in listOf("/storage/emulated/0/", "/sdcard/", "/storage/sdcard0/")) {
            val f = File(prefix + path)
            if (f.exists() && f.canRead()) return f
        }
        return null
    }

    // Data classes
    private data class TestTrack(
        val id: Long,
        val artist: String,
        val title: String,
        val path: String,
    )

    // JSON output data classes
    data class BenchmarkOutput(
        val device: String,
        val soc: String,
        val androidVersion: String,
        val ortVersion: String,
        val tracks: List<TrackResult>,
    )

    data class TrackResult(
        val path: String,
        val artist: String,
        val title: String,
        var mulan: EmbeddingResult? = null,
        var flamingo: EmbeddingResult? = null,
    )

    data class EmbeddingResult(
        val ep: String,
        val timingMs: Long,
        val dim: Int,
        val embedding: List<Float>,
    )
}
