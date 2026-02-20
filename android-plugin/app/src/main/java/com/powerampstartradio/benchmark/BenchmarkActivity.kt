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
 * Standalone benchmark activity for testing TFLite inference on-device.
 *
 * Runs MuLan and Flamingo on a few Poweramp tracks, reports timing,
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
            Text("LiteRT Embedding Benchmark", style = MaterialTheme.typography.headlineSmall)
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

        // Try column sets in order until one works.
        // Poweramp's content provider joins folder_files with folders.
        // The correct columns are: "path" (from folders, has trailing /)
        // and "folder_files.name" (short filename).
        // Note: "folder_path"/"file_name" are PlaylistEntries columns, not folder_files!
        data class ColumnSet(val name: String, val columns: Array<String>)
        val columnSets = listOf(
            ColumnSet("path+name", arrayOf(
                "folder_files._id", "artist", "title_tag", "folder_files.duration",
                "path", "folder_files.name"
            )),
            ColumnSet("minimal", arrayOf(
                "folder_files._id", "artist", "title_tag", "folder_files.duration"
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
                val titleIdx = it.getColumnIndex("title_tag")
                val pathIdx = it.getColumnIndex("path")       // folders.path (trailing /)
                val nameIdx = it.getColumnIndex("name")       // folder_files.name

                // Log first row's available columns for debugging
                if (it.moveToFirst()) {
                    Log.i(TAG, "Using column set '$usedSet', columns: ${it.columnNames.toList()}")
                    val samplePath = if (pathIdx >= 0) it.getString(pathIdx) else null
                    val sampleName = if (nameIdx >= 0) it.getString(nameIdx) else null
                    Log.i(TAG, "Sample row: path=$samplePath, name=$sampleName")
                    // Reset to process from first row
                    it.moveToPosition(-1)
                }

                while (it.moveToNext()) {
                    val path = when {
                        pathIdx >= 0 && nameIdx >= 0 -> {
                            val folder = it.getString(pathIdx) ?: ""  // already has trailing /
                            val name = it.getString(nameIdx) ?: ""
                            if (name.isNotEmpty()) "$folder$name" else null
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

        Log.i(TAG, "Query returned ${result.size} tracks with paths")
        if (result.isNotEmpty()) {
            Log.i(TAG, "Sample paths: ${result.take(3).map { it.path }}")
        }

        return result
    }

    private suspend fun runBenchmark(accelerator: Accelerator, onStatus: (String) -> Unit) {
        val sb = StringBuilder()
        fun log(msg: String) {
            sb.appendLine(msg)
            Log.i(TAG, msg)
            onStatus(sb.toString())
        }

        log("=== LiteRT Embedding Benchmark ===")
        log("Device: ${Build.MANUFACTURER} ${Build.MODEL}")
        log("SOC: ${Build.SOC_MODEL}")
        log("Requested accelerator: $accelerator")
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

        // Pick random tracks that are readable (cap attempts to avoid scanning all 74K)
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
            log("Sample paths: ${allTracks.take(3).map { it.path }}")
            log("Has audio permission: ${hasAudioPermission()}")
            return
        }
        log("Selected ${testTracks.size} tracks (resolved $resolveAttempts attempts)\n")

        val mulanFile = resolveModelFile(filesDir, "mulan_audio")
        val flamingoFile = resolveModelFile(filesDir, "flamingo_encoder")
        val projectorFile = resolveModelFile(filesDir, "flamingo_projector")

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
        var mulanEp: String? = null
        var flamingoEp: String? = null
        val hasFlamingo = flamingoFile.exists()

        // Cached 16kHz audio from MuLan pass for Flamingo (decode-once optimization)
        val cachedAudio16k = arrayOfNulls<AudioDecoder.DecodedAudio>(testTracks.size)

        log("Native mel: ${if (NativeMel.isAvailable) "YES" else "NO (Kotlin fallback)"}")
        log("")

        // ── MuLan Pass ──
        if (mulanFile.exists()) {
            log("Loading MuLan model (requesting $accelerator)...")
            val loadStart = System.nanoTime()
            var mulanInference: MuLanInference? = null
            try {
                mulanInference = MuLanInference(mulanFile, accelerator)
                val loadMs = (System.nanoTime() - loadStart) / 1_000_000
                log("  MuLan loaded in ${loadMs}ms (${mulanInference.activeAccelerator})")
            } catch (e: Exception) {
                log("  MuLan load FAILED: ${e.message}")
            }

            if (mulanInference != null) {
                mulanEp = "LiteRT ${mulanInference.activeAccelerator}"
                log("")
                for ((i, track) in testTracks.withIndex()) {
                    log("MuLan [${i + 1}/${testTracks.size}] ${track.artist} - ${track.title}")

                    val audioFile = resolveFile(track.path)!!

                    try {
                        val decodeStart = System.nanoTime()
                        val audio = decoder.decode(audioFile, 24000, maxDurationS = 900)
                        val decodeMs = (System.nanoTime() - decodeStart) / 1_000_000
                        if (audio == null) {
                            log("  Decode failed")
                            continue
                        }
                        log("  Audio: ${audio.durationS}s @ ${audio.sampleRate}Hz (decode: ${decodeMs}ms)")

                        // Cache 16kHz version for Flamingo pass (resample 24→16kHz via soxr)
                        if (hasFlamingo) {
                            try {
                                val resampleStart = System.nanoTime()
                                val samples16k = decoder.resample(audio.samples, 24000, 16000)
                                val resampleMs = (System.nanoTime() - resampleStart) / 1_000_000
                                cachedAudio16k[i] = AudioDecoder.DecodedAudio(
                                    samples16k, 16000, samples16k.size.toFloat() / 16000
                                )
                                log("  Cached 16kHz: ${samples16k.size} samples (resample: ${resampleMs}ms)")
                            } catch (e: Exception) {
                                log("  16kHz cache failed: ${e.message}")
                            }
                        }

                        val inferStart = System.nanoTime()
                        val embedding = mulanInference.generateEmbedding(audio)
                        val inferMs = (System.nanoTime() - inferStart) / 1_000_000

                        if (embedding != null) {
                            log("  Embedding: ${embedding.size}d in ${inferMs}ms")
                            log("  First 5: ${embedding.take(5).map { "%.4f".format(it) }}")
                            results[i].mulan = EmbeddingResult(
                                ep = mulanEp!!,
                                timingMs = inferMs,
                                dim = embedding.size,
                                embedding = embedding.toList(),
                            )
                        } else {
                            log("  Inference FAILED")
                        }
                    } catch (e: Throwable) {
                        log("  ERROR: ${e.javaClass.simpleName}: ${e.message}")
                    }
                }
                mulanInference.close()
                log("\nMuLan session closed.")
            }
        } else {
            log("MuLan model not found at ${mulanFile.absolutePath}")
        }

        // ── Flamingo Pass ──
        if (hasFlamingo) {
            log("\nLoading Flamingo model (requesting $accelerator)...")
            val loadStart = System.nanoTime()
            var flamingoInference: FlamingoInference? = null
            try {
                flamingoInference = FlamingoInference(flamingoFile, projectorFile, accelerator)
                val loadMs = (System.nanoTime() - loadStart) / 1_000_000
                log("  Flamingo loaded in ${loadMs}ms (${flamingoInference.activeAccelerator})")
                log("  Output dim: ${flamingoInference.outputDim}")
            } catch (e: Exception) {
                log("  Flamingo load FAILED: ${e.message}")
            }

            if (flamingoInference != null) {
                flamingoEp = "LiteRT ${flamingoInference.activeAccelerator}"
                log("")
                for ((i, track) in testTracks.withIndex()) {
                    log("Flamingo [${i + 1}/${testTracks.size}] ${track.artist} - ${track.title}")

                    val audioFile = resolveFile(track.path)!!

                    try {
                        // Use cached 16kHz audio from MuLan pass (decode-once optimization)
                        val cached = cachedAudio16k[i]
                        cachedAudio16k[i] = null  // free memory as we go

                        val decodeStart = System.nanoTime()
                        val audio = cached ?: decoder.decode(audioFile, 16000, maxDurationS = 1800)
                        val decodeMs = (System.nanoTime() - decodeStart) / 1_000_000
                        if (audio == null) {
                            log("  Decode failed")
                            continue
                        }
                        val source = if (cached != null) "cached" else "decoded"
                        log("  Audio: ${audio.durationS}s @ ${audio.sampleRate}Hz ($source: ${decodeMs}ms)")

                        val inferStart = System.nanoTime()
                        val embedding = flamingoInference.generateEmbedding(audio)
                        val inferMs = (System.nanoTime() - inferStart) / 1_000_000

                        if (embedding != null) {
                            log("  Embedding: ${embedding.size}d in ${inferMs}ms")
                            log("  First 5: ${embedding.take(5).map { "%.4f".format(it) }}")
                            results[i].flamingo = EmbeddingResult(
                                ep = flamingoEp!!,
                                timingMs = inferMs,
                                dim = embedding.size,
                                embedding = embedding.toList(),
                            )
                        } else {
                            log("  Inference FAILED")
                        }
                    } catch (e: Throwable) {
                        log("  ERROR: ${e.javaClass.simpleName}: ${e.message}")
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
            runtime = "LiteRT 2.1.1",
            mulanEp = mulanEp,
            flamingoEp = flamingoEp,
            tracks = results,
        )

        val gson = GsonBuilder().setPrettyPrinting().create()
        val json = gson.toJson(output)
        val outputFile = File(filesDir, "benchmark_results.json")
        outputFile.writeText(json)

        log("\n=== Results saved ===")
        log("File: ${outputFile.absolutePath}")
        log("Pull via: adb pull ${outputFile.absolutePath}")
        log("\nMuLan EP: ${mulanEp ?: "not loaded"}")
        log("Flamingo EP: ${flamingoEp ?: "not loaded"}")

        log("\n=== Summary ===")
        for (r in results) {
            log("${r.artist} - ${r.title}")
            r.mulan?.let { log("  MuLan: ${it.dim}d, ${it.timingMs}ms, EP=${it.ep}") }
            r.flamingo?.let { log("  Flamingo: ${it.dim}d, ${it.timingMs}ms, EP=${it.ep}") }
        }
        log("\nBenchmark complete.")
    }

    /**
     * Run matching diagnostics: compare Poweramp library against embedding DB,
     * categorize all match failures, and save detailed JSON for analysis.
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

        // Open embedding database
        val dbFile = File(filesDir, "embeddings.db")
        if (!dbFile.exists()) {
            log("ERROR: embeddings.db not found at ${dbFile.absolutePath}")
            log("Copy your fused.db to ${dbFile.absolutePath} first.")
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

            if (result.embeddedPathsSample.isNotEmpty()) {
                log("")
                log("--- Embedded path samples ---")
                for (p in result.embeddedPathsSample) {
                    log("  $p")
                }
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

            // Save full JSON to external files dir (accessible via adb pull without root)
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
        Log.w(TAG, "Could not resolve file: $path")
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
        val runtime: String,
        val mulanEp: String? = null,
        val flamingoEp: String? = null,
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
