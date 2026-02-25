package com.powerampstartradio.benchmark

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
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
import androidx.core.content.ContextCompat
import com.google.gson.GsonBuilder
import com.google.ai.edge.litert.Accelerator
import com.powerampstartradio.data.EmbeddingDatabase
import com.powerampstartradio.indexing.*
import com.powerampstartradio.poweramp.PowerampHelper
import kotlinx.coroutines.*
import java.io.File

/**
 * Standalone benchmark activity for testing CLaMP3 TFLite inference on-device.
 *
 * Runs MERT + CLaMP3 audio encoder on a few Poweramp tracks, reports timing,
 * and saves full embeddings as JSON for desktop comparison.
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
        private const val MAX_RESOLVE_ATTEMPTS = 100
    }

    private val audioPermission: String
        get() = if (Build.VERSION.SDK_INT >= 33)
            Manifest.permission.READ_MEDIA_AUDIO
        else
            Manifest.permission.READ_EXTERNAL_STORAGE

    private var onPermissionResult: ((Boolean) -> Unit)? = null

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        onPermissionResult?.invoke(granted)
        onPermissionResult = null
    }

    private fun hasAudioPermission(): Boolean =
        ContextCompat.checkSelfPermission(this, audioPermission) == PackageManager.PERMISSION_GRANTED

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
        var status by remember { mutableStateOf("Ready. Select accelerator and tap 'Run Benchmark'.") }
        var running by remember { mutableStateOf(false) }
        var selectedAccelerator by remember { mutableStateOf(Accelerator.GPU) }
        val scope = rememberCoroutineScope()

        fun startBenchmark() {
            running = true
            status = "Starting benchmark with $selectedAccelerator..."
            scope.launch(Dispatchers.IO) {
                try {
                    runBenchmark(selectedAccelerator) { msg -> status = msg }
                } catch (e: Exception) {
                    status = "ERROR: ${e.message}\n\n${e.stackTraceToString()}"
                } finally {
                    running = false
                }
            }
        }

        fun startDiagnostics() {
            running = true
            status = "Starting matching diagnostics..."
            scope.launch(Dispatchers.IO) {
                try {
                    runDiagnostics { msg -> status = msg }
                } catch (e: Exception) {
                    status = "ERROR: ${e.message}\n\n${e.stackTraceToString()}"
                } finally {
                    running = false
                }
            }
        }

        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp)
        ) {
            Text("CLaMP3 Embedding Benchmark", style = MaterialTheme.typography.headlineSmall)
            Spacer(Modifier.height(8.dp))

            Text(
                "Device: ${Build.MANUFACTURER} ${Build.MODEL}\n" +
                "SOC: ${Build.SOC_MODEL}\n" +
                "Android: ${Build.VERSION.RELEASE} (SDK ${Build.VERSION.SDK_INT})",
                fontSize = 12.sp,
                fontFamily = FontFamily.Monospace,
            )
            Spacer(Modifier.height(12.dp))

            // Accelerator selector
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                for (accel in listOf(Accelerator.CPU, Accelerator.GPU)) {
                    FilterChip(
                        selected = selectedAccelerator == accel,
                        onClick = { if (!running) selectedAccelerator = accel },
                        label = { Text(accel.name) },
                    )
                }
            }
            Spacer(Modifier.height(12.dp))

            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button(
                    onClick = {
                        if (hasAudioPermission()) {
                            startBenchmark()
                        } else {
                            status = "Requesting audio file permission..."
                            onPermissionResult = { granted ->
                                if (granted) {
                                    startBenchmark()
                                } else {
                                    status = "Permission denied. Cannot read audio files."
                                }
                            }
                            permissionLauncher.launch(audioPermission)
                        }
                    },
                    enabled = !running,
                ) {
                    Text(if (running) "Running..." else "Benchmark ($selectedAccelerator)")
                }

                OutlinedButton(
                    onClick = { startDiagnostics() },
                    enabled = !running,
                ) {
                    Text("Diagnose Matching")
                }
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

        data class ColumnSet(val name: String, val columns: Array<String>)
        val columnSets = listOf(
            ColumnSet("path+name", arrayOf(
                "folder_files._id", "artist", "album", "title_tag", "folder_files.duration",
                "path", "folder_files.name"
            )),
            ColumnSet("minimal", arrayOf(
                "folder_files._id", "artist", "album", "title_tag", "folder_files.duration"
            )),
        )

        try {
            var cursor: android.database.Cursor? = null
            var usedSet = "minimal"
            for (cs in columnSets) {
                cursor = try {
                    contentResolver.query(filesUri, cs.columns, null, null, null)
                } catch (e: Exception) {
                    Log.w(TAG, "Column set '${cs.name}' failed: ${e.message}")
                    null
                }
                if (cursor != null) {
                    usedSet = cs.name
                    Log.i(TAG, "Using column set: ${cs.name}")
                    break
                }
            }

            cursor?.use {
                val idIdx = it.getColumnIndex("_id")
                val artistIdx = it.getColumnIndex("artist")
                val albumIdx = it.getColumnIndex("album")
                val titleIdx = it.getColumnIndex("title_tag")
                val durationIdx = it.getColumnIndex("duration")
                val pathIdx = it.getColumnIndex("path")
                val nameIdx = it.getColumnIndex("name")

                if (it.moveToFirst()) {
                    Log.i(TAG, "Using column set '$usedSet', columns: ${it.columnNames.toList()}")
                    it.moveToPosition(-1)
                }

                while (it.moveToNext()) {
                    val path = when {
                        pathIdx >= 0 && nameIdx >= 0 -> {
                            val folder = it.getString(pathIdx) ?: ""
                            val name = it.getString(nameIdx) ?: ""
                            if (name.isNotEmpty()) "$folder$name" else null
                        }
                        else -> null
                    }
                    if (path != null) {
                        result.add(TestTrack(
                            id = it.getLong(idIdx),
                            artist = it.getString(artistIdx) ?: "",
                            album = if (albumIdx >= 0) it.getString(albumIdx) ?: "" else "",
                            title = it.getString(titleIdx) ?: "",
                            durationMs = if (durationIdx >= 0) it.getLong(durationIdx) else 0L,
                            path = path,
                        ))
                    }
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error querying Poweramp", e)
        }

        Log.i(TAG, "Query returned ${result.size} tracks with paths")
        return result
    }

    private suspend fun runBenchmark(accelerator: Accelerator, onStatus: (String) -> Unit) {
        val sb = StringBuilder()
        fun log(msg: String) {
            sb.appendLine(msg)
            Log.i(TAG, msg)
            onStatus(sb.toString())
        }

        log("=== CLaMP3 Embedding Benchmark ===")
        log("Device: ${Build.MANUFACTURER} ${Build.MODEL}")
        log("SOC: ${Build.SOC_MODEL}")
        log("Requested accelerator: $accelerator")
        log("")

        // Discover tracks from Poweramp
        log("Querying Poweramp library...")
        val allTracks = withContext(Dispatchers.Main) { queryTracksWithPaths() }
        if (allTracks.isEmpty()) {
            log("ERROR: No tracks found in Poweramp library.")
            return
        }
        log("Found ${allTracks.size} tracks in Poweramp")

        // Pick random tracks that are readable
        val testTracks = mutableListOf<TestTrack>()
        var resolveAttempts = 0
        for (track in allTracks.shuffled()) {
            if (testTracks.size >= MAX_TRACKS) break
            if (resolveAttempts >= MAX_RESOLVE_ATTEMPTS) break
            resolveAttempts++
            if (resolveFile(track.path) != null) {
                testTracks.add(track)
            }
        }
        if (testTracks.isEmpty()) {
            log("ERROR: Could not resolve any audio file paths (tried $resolveAttempts).")
            return
        }
        log("Selected ${testTracks.size} tracks (resolved $resolveAttempts attempts)\n")

        val mertFile = resolveModelFile(filesDir, "mert")
        val clamp3AudioFile = resolveModelFile(filesDir, "clamp3_audio")

        if (!mertFile.exists() || !clamp3AudioFile.exists()) {
            log("ERROR: CLaMP3 models not found.")
            log("  MERT: ${mertFile.absolutePath} (exists=${mertFile.exists()})")
            log("  CLaMP3 audio: ${clamp3AudioFile.absolutePath} (exists=${clamp3AudioFile.exists()})")
            log("Transfer mert.tflite and clamp3_audio.tflite to ${filesDir.absolutePath}")
            return
        }

        val results = mutableListOf<TrackResult>()
        for (track in testTracks) {
            results.add(TrackResult(
                path = track.path,
                artist = track.artist,
                album = track.album,
                title = track.title,
                durationMs = track.durationMs,
            ))
        }

        val decoder = AudioDecoder()

        // ── Phase 1: MERT Feature Extraction ──
        log("Loading MERT model (requesting $accelerator)...")
        val mertLoadStart = System.nanoTime()
        val mertInference = try { MertInference(mertFile) }
        catch (e: Exception) {
            log("MERT load FAILED: ${e.message}")
            return
        }
        val mertLoadMs = (System.nanoTime() - mertLoadStart) / 1_000_000
        log("  MERT loaded in ${mertLoadMs}ms")
        log("")

        // Extract features for all tracks
        data class TrackFeatures(val features: List<FloatArray>, val decodeMsMs: Long, val mertMs: Long)
        val allFeatures = arrayOfNulls<TrackFeatures>(testTracks.size)

        for ((i, track) in testTracks.withIndex()) {
            log("MERT [${i + 1}/${testTracks.size}] ${track.artist} - ${track.title}")

            val audioFile = resolveFile(track.path)!!
            try {
                val decodeStart = System.nanoTime()
                val audio = decoder.decode(audioFile, MertInference.SAMPLE_RATE, maxDurationS = 900)
                val decodeMs = (System.nanoTime() - decodeStart) / 1_000_000
                if (audio == null) { log("  Decode failed"); continue }
                log("  Audio: ${audio.durationS}s @ ${audio.sampleRate}Hz (decode: ${decodeMs}ms)")

                val inferStart = System.nanoTime()
                val features = mutableListOf<FloatArray>()
                val numWindows = mertInference.extractFeaturesStreaming(
                    audio,
                    onFeatureExtracted = { features.add(it.copyOf()) },
                    onWindowDone = {},
                )
                val mertMs = (System.nanoTime() - inferStart) / 1_000_000

                log("  Extracted $numWindows windows in ${mertMs}ms")
                allFeatures[i] = TrackFeatures(features, decodeMs, mertMs)
            } catch (e: Throwable) {
                log("  ERROR: ${e.javaClass.simpleName}: ${e.message}")
            }
        }

        mertInference.close()
        log("\nMERT session closed.")

        // ── Phase 2: CLaMP3 Audio Encoding ──
        log("\nLoading CLaMP3 audio encoder (requesting $accelerator)...")
        val clamp3LoadStart = System.nanoTime()
        val clamp3Inference = try { Clamp3AudioInference(clamp3AudioFile) }
        catch (e: Exception) {
            log("CLaMP3 audio load FAILED: ${e.message}")
            return
        }
        val clamp3LoadMs = (System.nanoTime() - clamp3LoadStart) / 1_000_000
        log("  CLaMP3 audio encoder loaded in ${clamp3LoadMs}ms")
        log("")

        var ep: String? = null

        for ((i, track) in testTracks.withIndex()) {
            val tf = allFeatures[i] ?: continue
            log("CLaMP3 [${i + 1}/${testTracks.size}] ${track.artist} - ${track.title}")

            try {
                val encStart = System.nanoTime()
                val embedding = clamp3Inference.encode(tf.features, tf.features.size)
                val encMs = (System.nanoTime() - encStart) / 1_000_000
                val totalMs = tf.decodeMsMs + tf.mertMs + encMs

                if (embedding != null) {
                    ep = "LiteRT GPU"
                    log("  Embedding: ${embedding.size}d (decode=${tf.decodeMsMs}ms, mert=${tf.mertMs}ms, clamp3=${encMs}ms, total=${totalMs}ms)")
                    log("  First 5: ${embedding.take(5).map { "%.4f".format(it) }}")
                    results[i].clamp3 = EmbeddingResult(
                        ep = ep,
                        timingMs = totalMs,
                        dim = embedding.size,
                        embedding = embedding.toList(),
                    )
                } else {
                    log("  Encode FAILED")
                }
            } catch (e: Throwable) {
                log("  ERROR: ${e.javaClass.simpleName}: ${e.message}")
            }
        }

        clamp3Inference.close()
        log("\nCLaMP3 session closed.")

        // ── Save results as JSON ──
        val output = BenchmarkOutput(
            device = "${Build.MANUFACTURER} ${Build.MODEL}",
            soc = Build.SOC_MODEL,
            androidVersion = "${Build.VERSION.RELEASE} (SDK ${Build.VERSION.SDK_INT})",
            runtime = "LiteRT",
            clamp3Ep = ep,
            tracks = results,
        )

        val gson = GsonBuilder().setPrettyPrinting().create()
        val json = gson.toJson(output)
        val outputFile = File(filesDir, "benchmark_results.json")
        outputFile.writeText(json)

        log("\n=== Results saved ===")
        log("File: ${outputFile.absolutePath}")
        log("Pull via: adb pull ${outputFile.absolutePath}")
        log("CLaMP3 EP: ${ep ?: "not loaded"}")

        log("\n=== Summary ===")
        for (r in results) {
            log("${r.artist} - ${r.title} [${r.album}] (${r.durationMs}ms)")
            r.clamp3?.let { log("  CLaMP3: ${it.dim}d, ${it.timingMs}ms, EP=${it.ep}") }
        }
        log("\nBenchmark complete.")
    }

    /**
     * Run matching diagnostics: compare Poweramp library against embedding DB.
     */
    private suspend fun runDiagnostics(onStatus: (String) -> Unit) {
        val sb = StringBuilder()
        fun log(msg: String) {
            sb.appendLine(msg)
            Log.i(TAG, msg)
            onStatus(sb.toString())
        }

        log("=== Matching Diagnostics ===")
        log("Device: ${Build.MANUFACTURER} ${Build.MODEL}")
        log("")

        val dbFile = File(filesDir, "embeddings.db")
        if (!dbFile.exists()) {
            log("ERROR: embeddings.db not found at ${dbFile.absolutePath}")
            return
        }

        log("Opening embedding DB: ${dbFile.name} (${dbFile.length() / 1024 / 1024}MB)")
        val embeddingDb = EmbeddingDatabase.open(dbFile)
        try {
            val detector = NewTrackDetector(embeddingDb)

            log("Running diagnostic matching...")
            val startTime = System.nanoTime()
            val result = detector.diagnoseMatching(this@BenchmarkActivity) { progress ->
                log(progress)
            }
            val elapsedMs = (System.nanoTime() - startTime) / 1_000_000

            log("")
            log("=== Results (${elapsedMs}ms) ===")
            log("Poweramp tracks:  ${result.powerampCount}")
            log("Embedded keys:    ${result.embeddedKeyCount}")
            log("Embedded paths:   ${result.embeddedPathCount}")
            log("")
            log("Exact key match:  ${result.exactKeyMatches}")
            log("Partial match:    ${result.partialMatches}")
            log("Path match:       ${result.pathMatches}")
            log("UNMATCHED:        ${result.unmatchedCount}")
            log("")
            log("--- Failure categories ---")
            for ((reason, count) in result.failureCategories.entries.sortedByDescending { it.value }) {
                log("  $reason: $count")
            }

            if (result.unmatchedSample.isNotEmpty()) {
                log("")
                log("--- Unmatched samples (first ${result.unmatchedSample.size}) ---")
                for (u in result.unmatchedSample.take(10)) {
                    log("  [${u.failureReason}] ${u.powerampKey}")
                    if (u.closestEmbeddedKey != null) {
                        log("    closest: ${u.closestEmbeddedKey}")
                    }
                }
            }

            val gson = GsonBuilder().setPrettyPrinting().create()
            val json = gson.toJson(result)
            val outputDir = getExternalFilesDir(null) ?: filesDir
            val outputFile = File(outputDir, "matching_diagnostics.json")
            outputFile.writeText(json)

            log("")
            log("=== JSON saved ===")
            log("File: ${outputFile.absolutePath}")
            log("Pull: adb pull ${outputFile.absolutePath}")
        } finally {
            embeddingDb.close()
        }

        log("\nDiagnostics complete.")
    }

    /** Prefer FP16 models (GPU-native, half size) over FP32 originals. */
    private fun resolveModelFile(dir: File, baseName: String): File {
        val variants = listOf("_fp16", "")
        for (suffix in variants) {
            val f = File(dir, "${baseName}${suffix}.tflite")
            if (f.exists()) {
                Log.i(TAG, "Model resolved: ${f.name}")
                return f
            }
        }
        return File(dir, "${baseName}.tflite")
    }

    private fun resolveFile(path: String): File? {
        val f = File(path)
        if (f.isFile && f.canRead()) return f
        return null
    }

    // Data classes
    private data class TestTrack(
        val id: Long,
        val artist: String,
        val album: String,
        val title: String,
        val durationMs: Long,
        val path: String,
    )

    data class BenchmarkOutput(
        val device: String,
        val soc: String,
        val androidVersion: String,
        val runtime: String,
        val clamp3Ep: String? = null,
        val tracks: List<TrackResult>,
    )

    data class TrackResult(
        val path: String,
        val artist: String,
        val album: String,
        val title: String,
        val durationMs: Long,
        var clamp3: EmbeddingResult? = null,
    )

    data class EmbeddingResult(
        val ep: String,
        val timingMs: Long,
        val dim: Int,
        val embedding: List<Float>,
    )
}
