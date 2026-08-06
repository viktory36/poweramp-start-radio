package com.powerampstartradio.benchmark

import android.Manifest
import android.content.pm.PackageManager
import android.media.MediaExtractor
import android.media.MediaFormat
import android.os.Build
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
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
import com.powerampstartradio.data.EmbeddingIndex
import com.powerampstartradio.indexing.*
import com.powerampstartradio.indexing.v2.V2LibraryDatabaseResolver
import com.powerampstartradio.poweramp.PowerampHelper
import com.powerampstartradio.poweramp.TrackMatcher
import com.powerampstartradio.ui.theme.PowerampStartRadioTheme
import kotlinx.coroutines.*
import java.io.File
import java.util.Locale

/**
 * Standalone benchmark activity for testing CLaMP3 TFLite inference on-device.
 *
 * Runs MERT + CLaMP3 audio encoder on a few Poweramp tracks, reports timing,
 * and saves embeddings as JSON for desktop comparison. `max_duration_s=0`
 * means full-track mode, which uses the same chunk-stitched extraction path
 * as on-device indexing instead of the older truncated benchmark path.
 *
 * Launch via:
 *   adb shell am start -n com.powerampstartradio.v2/.benchmark.BenchmarkActivity
 *
 * Auto-start via adb (no UI interaction needed):
 *   adb shell am start -n com.powerampstartradio.v2/.benchmark.BenchmarkActivity --ez auto_start true
 *
 * Full-track auto-start:
 *   adb shell am start -n com.powerampstartradio.v2/.benchmark.BenchmarkActivity --ez auto_start true --ei max_duration_s 0
 *
 * Pull results via:
 *   adb shell run-as com.powerampstartradio.v2 cat files/benchmark_results.json
 */
class BenchmarkActivity : ComponentActivity() {

    companion object {
        private const val TAG = "EmbeddingBenchmark"
        private const val MAX_TRACKS = 5
        private const val MAX_RESOLVE_ATTEMPTS = 100
        private const val TORCHAUDIO_HANN_V1_MAX_ABS_TOLERANCE = 5e-6f
        private const val TORCHAUDIO_HANN_V1_RMSE_TOLERANCE = 5e-7
        private const val TORCHAUDIO_HANN_V1_COSINE_TOLERANCE = 0.999999999

        private fun deviceSocLabel(): String {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                Build.SOC_MODEL.trim().takeUnless { it.isEmpty() || it.equals("unknown", ignoreCase = true) }
                    ?.let { return it }
            }

            val hardware = Build.HARDWARE.trim().ifEmpty { "unknown" }
            val board = Build.BOARD.trim().ifEmpty { "unknown" }
            return "hardware=$hardware, board=$board"
        }
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
        enableEdgeToEdge()

        setContent {
            PowerampStartRadioTheme {
                BenchmarkScreen()
            }
        }
    }

    @Composable
    private fun BenchmarkScreen() {
        val autoStartRequested = remember {
            intent.getBooleanExtra("auto_start", false)
        }
        var status by remember {
            mutableStateOf(
                if (autoStartRequested) {
                    "Preparing requested benchmark..."
                } else {
                    "Ready. Select accelerator and tap 'Run Benchmark'."
                },
            )
        }
        var running by remember { mutableStateOf(autoStartRequested) }
        var selectedAccelerator by remember { mutableStateOf(Accelerator.GPU) }
        val scope = rememberCoroutineScope()

        val resampleQuality = when (intent.getStringExtra("resample_quality")) {
            "mq" -> NativeResampler.QUALITY_MQ
            "vhq" -> NativeResampler.QUALITY_VHQ
            else -> NativeResampler.QUALITY_HQ
        }

        fun startBenchmark() {
            running = true
            status = "Starting benchmark with $selectedAccelerator..."
            scope.launch(Dispatchers.IO) {
                try {
                    runBenchmark(selectedAccelerator, resampleQuality) { msg -> status = msg }
                } catch (e: Throwable) {
                    Log.e(TAG, "Benchmark failed", e)
                    status = "ERROR: ${e.message}\n\n${e.stackTraceToString()}"
                } finally {
                    running = false
                }
            }
        }

        fun startTextBenchmark() {
            running = true
            status = "Starting text search benchmark..."
            scope.launch(Dispatchers.IO) {
                try {
                    runTextBenchmark { msg -> status = msg }
                } catch (e: Throwable) {
                    Log.e(TAG, "Text benchmark failed", e)
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

        fun startSpanContinuityBenchmark() {
            running = true
            status = "Starting exact-span continuity benchmark..."
            scope.launch(Dispatchers.IO) {
                try {
                    runSpanContinuityBenchmark { msg -> status = msg }
                } catch (e: Throwable) {
                    Log.e(TAG, "Span continuity benchmark failed", e)
                    status = "ERROR: ${e.message}\n\n${e.stackTraceToString()}"
                } finally {
                    running = false
                }
            }
        }

        fun startPcmCacheParityBenchmark() {
            running = true
            status = "Starting production PCM-cache parity benchmark..."
            scope.launch(Dispatchers.IO) {
                try {
                    runPcmCacheParityBenchmark { msg -> status = msg }
                } catch (e: Throwable) {
                    Log.e(TAG, "PCM-cache parity benchmark failed", e)
                    status = "ERROR: ${e.message}\n\n${e.stackTraceToString()}"
                } finally {
                    running = false
                }
            }
        }

        fun startTorchAudioHannV1Benchmark() {
            running = true
            status = "Starting TorchAudio Hann V1 native parity benchmark..."
            scope.launch(Dispatchers.IO) {
                try {
                    runTorchAudioHannV1Benchmark { msg -> status = msg }
                } catch (e: Throwable) {
                    Log.e(TAG, "TorchAudio Hann V1 parity benchmark failed", e)
                    status = "ERROR: ${e.message}\n\n${e.stackTraceToString()}"
                } finally {
                    running = false
                }
            }
        }

        // Auto-start via intent extra: --ez auto_start true [--es benchmark_type text|audio|diagnose]
        LaunchedEffect(autoStartRequested) {
            if (autoStartRequested) {
                // Let the truthful preparing state reach the screen before a requested
                // benchmark starts producing progress updates.
                withFrameNanos { }
                val type = intent.getStringExtra("benchmark_type") ?: "audio"
                when (type) {
                    "text" -> startTextBenchmark()
                    "diagnose" -> startDiagnostics()
                    "span" -> startSpanContinuityBenchmark()
                    "pcm_cache" -> startPcmCacheParityBenchmark()
                    "hann_v1" -> startTorchAudioHannV1Benchmark()
                    else -> {
                        if (hasAudioPermission()) {
                            startBenchmark()
                        } else {
                            running = false
                            status = "Audio permission is required. Tap Benchmark to grant it."
                        }
                    }
                }
            }
        }

        Column(
            modifier = Modifier
                .fillMaxSize()
                .windowInsetsPadding(WindowInsets.safeDrawing)
                .padding(16.dp)
        ) {
            Text(
                "CLaMP3 Embedding Benchmark — ${if (running) "RUNNING" else "idle"}",
                style = MaterialTheme.typography.headlineSmall,
            )
            Spacer(Modifier.height(8.dp))

            Text(
                "Device: ${Build.MANUFACTURER} ${Build.MODEL}\n" +
                "SOC: ${deviceSocLabel()}\n" +
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
                    onClick = { startTextBenchmark() },
                    enabled = !running,
                ) {
                    Text("Text Search")
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

    private suspend fun runBenchmark(
        accelerator: Accelerator,
        resampleQuality: Int = NativeResampler.QUALITY_HQ,
        onStatus: (String) -> Unit,
    ) {
        val qualityName = when (resampleQuality) {
            NativeResampler.QUALITY_MQ -> "mq"
            NativeResampler.QUALITY_VHQ -> "vhq"
            else -> "hq"
        }
        val sb = StringBuilder()
        fun log(msg: String) {
            sb.appendLine(msg)
            Log.i(TAG, msg)
            onStatus(sb.toString())
        }

        log("=== CLaMP3 Embedding Benchmark ===")
        log("Device: ${Build.MANUFACTURER} ${Build.MODEL}")
        log("SOC: ${deviceSocLabel()}")
        log("Requested accelerator: $accelerator")
        log("Resample quality: $qualityName")
        val benchmarkMaxDurationS = intent.getIntExtra("max_duration_s", 0)
        log("Max duration: ${if (benchmarkMaxDurationS > 0) "${benchmarkMaxDurationS}s cap" else "full track"}")
        log("")

        // Discover tracks from Poweramp
        log("Querying Poweramp library...")
        // Poweramp can return tens of thousands of rows. A provider query has no UI-thread
        // requirement and must not stall the first frame of an auto-started benchmark.
        val allTracks = withContext(Dispatchers.IO) { queryTracksWithPaths() }
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
        val mertAccel = mertInference.activeAccelerator.name
        log("  MERT loaded in ${mertLoadMs}ms (accelerator: $mertAccel)")
        log("  Model: ${mertFile.name} (${mertFile.length() / 1024 / 1024}MB)")
        log("")

        // Extract features for all tracks
        data class TrackFeatures(
            val features: List<FloatArray>,
            val decodeMs: Long,
            val resampleMs: Long,
            val mertMs: Long,
            val perWindowMs: List<Long>,
            val audioDurationS: Float,
        )
        val allFeatures = arrayOfNulls<TrackFeatures>(testTracks.size)

        for ((i, track) in testTracks.withIndex()) {
            log("MERT [${i + 1}/${testTracks.size}] ${track.artist} - ${track.title}")

            val audioFile = resolveFile(track.path)!!
            try {
                val features = mutableListOf<FloatArray>()
                val perWindowMs = mutableListOf<Long>()
                var lastWindowEndNs = System.nanoTime()
                var mertNanWindows = 0
                var carrySamples = FloatArray(0)
                var totalDecodeMs = 0L
                var totalResampleMs = 0L
                var totalMertMs = 0L
                var chunkCount = 0

                val trackDurationS = (track.durationMs / 1000L).toInt()
                val targetDurationS = if (benchmarkMaxDurationS > 0) {
                    minOf(trackDurationS, benchmarkMaxDurationS)
                } else {
                    trackDurationS
                }
                if (targetDurationS <= 0) {
                    log("  Audio too short")
                    continue
                }

                var chunkStartS = 0
                while (chunkStartS < targetDurationS || chunkStartS == 0) {
                    val remainingS = targetDurationS - chunkStartS
                    val chunkDurationS = minOf(MertInference.CHUNK_DURATION_S, remainingS)
                    val isLastChunk = chunkStartS + chunkDurationS >= targetDurationS

                    val audio = decoder.decode(
                        audioFile,
                        MertInference.SAMPLE_RATE,
                        maxDurationS = chunkDurationS,
                        startTimeS = chunkStartS,
                        resampleQuality = resampleQuality,
                    )
                    if (audio == null || audio.samples.isEmpty()) {
                        log("  Decode failed at chunk ${chunkCount + 1}")
                        break
                    }

                    totalDecodeMs += audio.decodeMs
                    totalResampleMs += audio.resampleMs
                    chunkCount++

                    val inferStart = System.nanoTime()
                    // Keep benchmark windowing identical to indexing so quality
                    // validation catches chunk-boundary regressions too.
                    val extraction = mertInference.extractFeaturesStreaming(
                        audio,
                        carrySamples = carrySamples,
                        flushTail = isLastChunk,
                        onFeatureExtracted = { feat ->
                            features.add(feat.copyOf())
                            val nanCount = feat.count { it.isNaN() }
                            val infCount = feat.count { it.isInfinite() }
                            if (nanCount > 0 || infCount > 0) {
                                mertNanWindows++
                                if (mertNanWindows <= 3) {
                                    Log.w(TAG, "MERT window ${features.size}: $nanCount NaN, $infCount Inf")
                                }
                            }
                        },
                        onWindowDone = {
                            val now = System.nanoTime()
                            perWindowMs.add((now - lastWindowEndNs) / 1_000_000)
                            lastWindowEndNs = now
                        },
                    )
                    totalMertMs += (System.nanoTime() - inferStart) / 1_000_000
                    carrySamples = extraction.carrySamples
                    chunkStartS += chunkDurationS
                }

                val numWindows = features.size
                val mertMs = totalMertMs
                val avgMs = if (numWindows > 0) mertMs / numWindows else 0

                log("  Audio: ${targetDurationS}s @ ${MertInference.SAMPLE_RATE}Hz (${chunkCount} chunk(s), decode=${totalDecodeMs}ms, resample=${totalResampleMs}ms)")
                log("  $numWindows windows: total=${mertMs}ms, avg=${avgMs}ms/win")
                if (perWindowMs.isNotEmpty()) {
                    log("  per-window: min=${perWindowMs.min()}ms, max=${perWindowMs.max()}ms")
                }
                if (mertNanWindows > 0) {
                    log("  WARNING: $mertNanWindows/$numWindows MERT windows contain NaN!")
                } else if (features.isNotEmpty()) {
                    // Log MERT feature stats for first window
                    val f0 = features[0]
                    val fMin = f0.min()
                    val fMax = f0.max()
                    val fAbsMax = f0.maxOf { kotlin.math.abs(it) }
                    log("  MERT features OK: min=${"%.4f".format(fMin)}, max=${"%.4f".format(fMax)}, absmax=${"%.4f".format(fAbsMax)}")
                }
                allFeatures[i] = TrackFeatures(
                    features,
                    totalDecodeMs,
                    totalResampleMs,
                    mertMs,
                    perWindowMs.toList(),
                    targetDurationS.toFloat(),
                )
                results[i].durationS = targetDurationS.toFloat()
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
        val clamp3Accel = clamp3Inference.activeAccelerator.name
        log("  CLaMP3 audio loaded in ${clamp3LoadMs}ms (accelerator: $clamp3Accel)")
        log("  Model: ${clamp3AudioFile.name} (${clamp3AudioFile.length() / 1024 / 1024}MB)")
        log("")

        for ((i, track) in testTracks.withIndex()) {
            val tf = allFeatures[i] ?: continue
            log("CLaMP3 [${i + 1}/${testTracks.size}] ${track.artist} - ${track.title}")

            try {
                val numSegments = clamp3Inference.segmentCount(tf.features.size)
                val encStart = System.nanoTime()
                val embedding = clamp3Inference.encode(tf.features, tf.features.size)
                val encMs = (System.nanoTime() - encStart) / 1_000_000
                val totalMs = tf.decodeMs + tf.resampleMs + tf.mertMs + encMs
                val realtimeFactor = if (tf.audioDurationS > 0) totalMs / (tf.audioDurationS * 1000f) else 0f

                if (embedding != null) {
                    val embNanCount = embedding.count { it.isNaN() }
                    val embInfCount = embedding.count { it.isInfinite() }
                    if (embNanCount > 0 || embInfCount > 0) {
                        log("  CLaMP3 output: $embNanCount NaN, $embInfCount Inf out of ${embedding.size}")
                    } else {
                        val norm = kotlin.math.sqrt(embedding.sumOf { (it * it).toDouble() }).toFloat()
                        log("  CLaMP3 output OK: norm=${"%.4f".format(norm)}, absmax=${"%.4f".format(embedding.maxOf { kotlin.math.abs(it) })}")
                    }
                    log("  ${embedding.size}d, $numSegments seg (decode=${tf.decodeMs}ms, resample=${tf.resampleMs}ms, mert=${tf.mertMs}ms, clamp3=${encMs}ms, total=${totalMs}ms)")
                    log("  Realtime factor: ${"%.2f".format(realtimeFactor)}x (${tf.audioDurationS}s audio in ${totalMs / 1000f}s)")
                    results[i].timing = TrackTiming(
                        decodeMs = tf.decodeMs,
                        resampleMs = tf.resampleMs,
                        resampleQuality = qualityName,
                        mertTotalMs = tf.mertMs,
                        mertWindows = tf.features.size,
                        mertPerWindowMs = tf.perWindowMs,
                        mertAvgWindowMs = if (tf.features.isNotEmpty()) tf.mertMs / tf.features.size else 0,
                        clamp3Segments = numSegments,
                        clamp3TotalMs = encMs,
                        totalMs = totalMs,
                        realtimeFactor = realtimeFactor,
                    )
                    results[i].clamp3 = EmbeddingResult(
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
        Log.i(TAG, "Building output JSON...")

        // ── Save results as JSON ──
        val output = BenchmarkOutput(
            device = "${Build.MANUFACTURER} ${Build.MODEL}",
            soc = deviceSocLabel(),
            androidVersion = "${Build.VERSION.RELEASE} (SDK ${Build.VERSION.SDK_INT})",
            runtime = "LiteRT",
            mertModel = mertFile.name,
            mertAccelerator = mertAccel,
            mertLoadMs = mertLoadMs,
            clamp3Model = clamp3AudioFile.name,
            clamp3Accelerator = clamp3Accel,
            clamp3LoadMs = clamp3LoadMs,
            tracks = results,
        )

        // Check for NaN embeddings (GPU numerical issues)
        for (r in results) {
            val emb = r.clamp3?.embedding
            if (emb != null && emb.any { it.isNaN() || it.isInfinite() }) {
                val nanCount = emb.count { it.isNaN() }
                val infCount = emb.count { it.isInfinite() }
                Log.w(TAG, "NaN/Inf in embedding for ${r.artist} - ${r.title}: " +
                    "$nanCount NaN, $infCount Inf out of ${emb.size}")
                // Replace NaN/Inf with 0 so JSON serialization doesn't fail
                r.clamp3 = EmbeddingResult(
                    dim = emb.size,
                    embedding = emb.map { if (it.isNaN() || it.isInfinite()) 0f else it },
                )
            }
        }

        Log.i(TAG, "BenchmarkOutput built, ${results.size} tracks, serializing...")
        val gson = GsonBuilder().setPrettyPrinting().create()
        val json = gson.toJson(output)
        Log.i(TAG, "JSON serialized: ${json.length} chars")
        val outputFile = File(filesDir, "benchmark_results.json")
        outputFile.writeText(json)
        Log.i(TAG, "Written to ${outputFile.absolutePath}")

        log("\n=== Results saved ===")
        log("File: ${outputFile.absolutePath}")
        log("Pull via: adb pull ${outputFile.absolutePath}")

        // ── Timing Summary ──
        log("\n=== Timing Summary ===")
        log("Models: MERT=${mertFile.name} ($mertAccel, load=${mertLoadMs}ms)")
        log("        CLaMP3=${clamp3AudioFile.name} ($clamp3Accel, load=${clamp3LoadMs}ms)")
        log("")
        log(String.format(Locale.ROOT, "%-30s %7s %7s %6s %6s %5s %7s %6s",
            "Track", "Dec", "Resamp", "MERT", "CL3", "Win", "Total", "RT-x"))
        log("-".repeat(105))

        val allTimings = results.mapNotNull { it.timing }
        for (r in results) {
            val t = r.timing ?: continue
            log(String.format(Locale.ROOT, "%-30s %6dms %6dms %5dms %5dms %4dw %6dms %5.2fx",
                "${r.artist} - ${r.title}".take(30),
                t.decodeMs, t.resampleMs, t.mertTotalMs, t.clamp3TotalMs, t.mertWindows, t.totalMs, t.realtimeFactor))
        }

        if (allTimings.isNotEmpty()) {
            log("-".repeat(105))
            val avgDecode = allTimings.map { it.decodeMs }.average().toLong()
            val avgResample = allTimings.map { it.resampleMs }.average().toLong()
            val avgMert = allTimings.map { it.mertTotalMs }.average().toLong()
            val avgClamp3 = allTimings.map { it.clamp3TotalMs }.average().toLong()
            val avgWindows = allTimings.map { it.mertWindows }.average().toInt()
            val avgTotal = allTimings.map { it.totalMs }.average().toLong()
            val avgRt = allTimings.map { it.realtimeFactor.toDouble() }.average().toFloat()
            val avgPerWindow = allTimings.flatMap { it.mertPerWindowMs }.let { if (it.isNotEmpty()) it.average().toLong() else 0 }
            log(String.format(Locale.ROOT, "%-30s %6dms %6dms %5dms %5dms %4dw %6dms %5.2fx",
                "AVERAGE", avgDecode, avgResample, avgMert, avgClamp3, avgWindows, avgTotal, avgRt))
            log("")
            log("MERT per-window: avg=${avgPerWindow}ms")
            val allWindowMs = allTimings.flatMap { it.mertPerWindowMs }
            if (allWindowMs.isNotEmpty()) {
                log("  min=${allWindowMs.min()}ms, max=${allWindowMs.max()}ms, " +
                    "p50=${allWindowMs.sorted()[allWindowMs.size / 2]}ms")
            }
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

        val generation = try {
            V2LibraryDatabaseResolver.requirePublished(filesDir)
        } catch (e: Exception) {
            log("ERROR: no valid published V2 generation: ${e.message}")
            return
        }
        val dbFile = generation.databaseFile

        log(
            "Opening generation ${generation.manifest.generationId}: " +
                "${dbFile.name} (${dbFile.length() / 1024 / 1024}MB)",
        )
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
            log("Embedded tracks:  ${result.embeddedTrackCount}")
            log("Embedded keys:    ${result.embeddedKeyCount}")
            log("Embedded paths:   ${result.embeddedPathCount}")
            log("")
            log("--- Match passes ---")
            for ((pass, count) in result.matchPassCounts.entries.sortedByDescending { it.value }) {
                log("  $pass: $count")
            }
            log("")
            log("Exact key match:  ${result.exactKeyMatches}")
            log("Partial match:    ${result.partialMatches}")
            log("Path match:       ${result.pathMatches}")
            log("Poweramp only:    ${result.unmatchedCount}")
            log("DB only:          ${result.dbOnlyCount}")
            log("  still on device: ${result.dbOnlyOnDeviceCount}")
            log("  missing on device: ${result.dbOnlyMissingCount}")
            log("")
            log("--- Failure categories ---")
            for ((reason, count) in result.failureCategories.entries.sortedByDescending { it.value }) {
                log("  $reason: $count")
            }

            val queueAudit = TrackMatcher(embeddingDb).auditQueueResolution(
                this@BenchmarkActivity,
                embeddingDb.getAllTracks(),
            )
            log("")
            log("--- Queue resolution audit ---")
            log("Matched:          ${queueAudit.matchedTracks}/${queueAudit.totalTracks}")
            log("Unmatched:        ${queueAudit.unmatchedTracks}")
            for ((matchType, count) in queueAudit.matchCounts.entries.sortedByDescending { it.value }) {
                log("  $matchType: $count")
            }
            if (queueAudit.unmatchedSample.isNotEmpty()) {
                log("  sample misses:")
                queueAudit.unmatchedSample.take(10).forEach { log("    $it") }
            }

            val unindexed = detector.findUnindexedTracks(this@BenchmarkActivity)
            log("")
            log("--- Manage Tracks audit ---")
            log("Unindexed shown:  ${unindexed.size}")
            unindexed.take(10).forEach { track ->
                log("  ${track.metadataKey}")
            }
            log("  battery:")
            val shownKeys = unindexed.map { it.metadataKey }.toSet()
            listOf(
                "tipper cloaked title track" to ("tipper|cloaked|cloaked|212800" to true),
                "soundgarden superunknown" to ("soundgarden|superunknown|superunknown|306500" to true),
                "asha puthli space talk" to ("asha puthli|the essential asha puthli|space talk|326000" to false),
                "shamoon na toon" to ("shamoon ismail; haider mustehsan; mooroo|cookie|na toon|208000" to false),
                "shamoon tuntuna" to ("shamoon ismail|tuntuna|tuntuna|266300" to false),
                "lsd thunderclouds" to ("lsd; sia; diplo; labrinth|labrinth, sia & diplo present... lsd (feat. sia, diplo & labrinth)|thunderclouds (feat. sia, diplo & labrinth)|187000" to false),
            ).forEach { (label, expectation) ->
                val (key, shouldBePresent) = expectation
                val actual = key in shownKeys
                log("    ${if (actual == shouldBePresent) "PASS" else "FAIL"}: $label (shown=$actual)")
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

            if (result.dbOnlySample.isNotEmpty()) {
                log("")
                log("--- DB-only samples (first ${result.dbOnlySample.size}) ---")
                for (entry in result.dbOnlySample.take(10)) {
                    log("  [${entry.source}] ${entry.metadataKey}")
                }
            }

            // Preserve raw provider fields which the production track model currently
            // drops. In particular, offset_ms is the start of a logical CUE track and
            // cue_folder_id identifies an uncut CUE source row. Without these fields,
            // different logical tracks backed by one file are all decoded from t=0.
            val providerAudit = queryProviderAudit(
                targetIds = unindexed.mapTo(linkedSetOf()) { it.powerampFileId },
                probeDecode = intent.getBooleanExtra("decode_probe", false),
                probeCueSpans = intent.getBooleanExtra("span_probe", false),
            ) { message -> log(message) }

            val gson = GsonBuilder().setPrettyPrinting().create()
            val payload = linkedMapOf<String, Any>(
                "schemaVersion" to 2,
                "generatedAtEpochMs" to System.currentTimeMillis(),
                "packageName" to packageName,
                "matching" to result,
                "queueResolution" to queueAudit,
                // V1 logged only the first ten Manage Tracks candidates. Retain the
                // complete identity/path set so decode failures, hidden choices, and
                // matcher defects can be diagnosed without starting an index run.
                "unindexed" to unindexed,
                "providerAudit" to providerAudit,
            )
            val json = gson.toJson(payload)
            val outputDir = getExternalFilesDir(null) ?: filesDir
            val outputFile = File(outputDir, "matching_diagnostics_v2.json")
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

    /** Non-mutating native parity proof against frozen TorchAudio-generated debug assets. */
    private fun runTorchAudioHannV1Benchmark(onStatus: (String) -> Unit) {
        val lines = StringBuilder()
        fun log(message: String) {
            lines.appendLine(message)
            Log.i(TAG, message)
            onStatus(lines.toString())
        }

        log("=== TorchAudio Hann V1 Native Parity ===")
        val gson = GsonBuilder().setPrettyPrinting().create()
        val assetRoot = "resampler_hann_v1"
        val manifestBytes = assets.open("$assetRoot/manifest.json").use { it.readBytes() }
        val manifest = gson.fromJson(
            manifestBytes.toString(Charsets.UTF_8),
            TorchAudioHannV1FixtureManifest::class.java,
        )
        require(manifest.specId == NativeMath.TORCHAUDIO_HANN_V1_SPEC_ID) {
            "Fixture spec ${manifest.specId} does not match ${NativeMath.TORCHAUDIO_HANN_V1_SPEC_ID}"
        }

        val fixtureResults = manifest.fixtures.map { fixture ->
            log("Fixture ${fixture.name}: ${fixture.inputSamples} @ ${fixture.fromRate}Hz")
            val inputBytes = assets.open("$assetRoot/${fixture.inputFile}").use { it.readBytes() }
            val expectedBytes = assets.open("$assetRoot/${fixture.expectedFile}").use { it.readBytes() }
            require(byteSha256(inputBytes) == fixture.inputSha256) {
                "Input fixture hash mismatch for ${fixture.name}"
            }
            require(byteSha256(expectedBytes) == fixture.expectedSha256) {
                "Expected fixture hash mismatch for ${fixture.name}"
            }
            val input = EmbeddingDatabase.blobToFloatArray(inputBytes)
            val expected = EmbeddingDatabase.blobToFloatArray(expectedBytes)
            require(input.size == fixture.inputSamples)
            require(expected.size == fixture.expectedSamples)

            val policyLength = NativeMath.torchAudioHannV1ResampledLength(
                input.size.toLong(), fixture.fromRate, fixture.toRate,
            )
            require(policyLength == expected.size.toLong()) {
                "Policy length $policyLength != fixture ${expected.size} for ${fixture.name}"
            }

            val wholeStarted = System.nanoTime()
            val whole = NativeMath.resampleTorchAudioHannV1(
                input, fixture.fromRate, fixture.toRate,
            ) ?: error("Whole native Hann resampling failed for ${fixture.name}")
            val wholeMs = (System.nanoTime() - wholeStarted) / 1_000_000L
            require(whole.size == expected.size) {
                "Whole native length ${whole.size} != expected ${expected.size}"
            }

            val chunked = FloatArray(whole.size)
            var outputStart = 0
            var scheduleIndex = 0
            var chunks = 0
            while (outputStart < whole.size) {
                val requested = fixture.chunkSchedule[
                    scheduleIndex % fixture.chunkSchedule.size
                ]
                val outputCount = minOf(requested, whole.size - outputStart)
                val required = TorchAudioHannV1Policy.requiredInputRange(
                    totalInputSamples = input.size.toLong(),
                    fromRate = fixture.fromRate,
                    toRate = fixture.toRate,
                    outputStartSample = outputStart.toLong(),
                    outputSampleCount = outputCount,
                )
                val inputStart = required.start.toInt()
                val inputEnd = required.endExclusive.toInt()
                val sourceSlice = input.copyOfRange(inputStart, inputEnd)
                val outputSlice = NativeMath.resampleTorchAudioHannV1Aligned(
                    samples = sourceSlice,
                    fromRate = fixture.fromRate,
                    toRate = fixture.toRate,
                    inputStartSample = required.start,
                    totalInputSamples = input.size.toLong(),
                    outputStartSample = outputStart.toLong(),
                    outputSampleCount = outputCount,
                ) ?: error("Aligned native Hann resampling failed for ${fixture.name} at $outputStart")
                require(outputSlice.size == outputCount)
                outputSlice.copyInto(chunked, outputStart)
                outputStart += outputCount
                scheduleIndex++
                chunks++
            }

            val contextProbeStart = minOf(500, whole.lastIndex).coerceAtLeast(0)
            val contextProbeCount = minOf(97, whole.size - contextProbeStart)
            val contextRange = TorchAudioHannV1Policy.requiredInputRange(
                totalInputSamples = input.size.toLong(),
                fromRate = fixture.fromRate,
                toRate = fixture.toRate,
                outputStartSample = contextProbeStart.toLong(),
                outputSampleCount = contextProbeCount,
            )
            val insufficientContextRejected = if (
                contextRange.start > 0L &&
                contextRange.endExclusive - contextRange.start > 1L
            ) {
                val missingStart = contextRange.start + 1L
                val missingSlice = input.copyOfRange(
                    missingStart.toInt(), contextRange.endExclusive.toInt(),
                )
                NativeMath.resampleTorchAudioHannV1Aligned(
                    samples = missingSlice,
                    fromRate = fixture.fromRate,
                    toRate = fixture.toRate,
                    inputStartSample = missingStart,
                    totalInputSamples = input.size.toLong(),
                    outputStartSample = contextProbeStart.toLong(),
                    outputSampleCount = contextProbeCount,
                ) == null
            } else null

            val comparison = compareFloatArrays(whole, expected)
            val chunkExact = whole.contentEquals(chunked)
            val withinTolerance =
                comparison.maxAbsError <= TORCHAUDIO_HANN_V1_MAX_ABS_TOLERANCE &&
                    comparison.rmse <= TORCHAUDIO_HANN_V1_RMSE_TOLERANCE &&
                    comparison.cosine >= TORCHAUDIO_HANN_V1_COSINE_TOLERANCE
            log(
                "  len=${whole.size}, chunks=$chunks exactChunks=$chunkExact " +
                    "rmse=${"%.9g".format(comparison.rmse)} " +
                    "max=${"%.9g".format(comparison.maxAbsError)} " +
                    "cos=${"%.12f".format(comparison.cosine)} pass=$withinTolerance"
            )
            TorchAudioHannV1FixtureResult(
                name = fixture.name,
                fromRate = fixture.fromRate,
                toRate = fixture.toRate,
                inputSamples = input.size,
                expectedSamples = expected.size,
                nativeSamples = whole.size,
                chunks = chunks,
                wholeMs = wholeMs,
                expectedSha256 = fixture.expectedSha256,
                nativeSha256 = floatSha256(whole),
                chunkedSha256 = floatSha256(chunked),
                exactWholeToChunked = chunkExact,
                insufficientContextRejected = insufficientContextRejected,
                rmseToTorchAudio = comparison.rmse,
                maxAbsErrorToTorchAudio = comparison.maxAbsError,
                cosineToTorchAudio = comparison.cosine,
                withinTolerance = withinTolerance,
            )
        }

        val lengthResults = listOf(
            TorchAudioHannV1LengthCase(19_556_776L, 44_100, 10_643_143L),
            TorchAudioHannV1LengthCase(19_724_591L, 44_100, 10_734_471L),
            TorchAudioHannV1LengthCase(10_452_143L, 44_100, 5_688_241L),
            TorchAudioHannV1LengthCase(16_212_626L, 44_100, 8_823_198L),
            TorchAudioHannV1LengthCase(6_300_000L, 48_000, 3_150_000L),
        ).map { expectedCase ->
            expectedCase.copy(
                actualOutputSamples = NativeMath.torchAudioHannV1ResampledLength(
                    expectedCase.inputSamples,
                    expectedCase.fromRate,
                    MertInference.SAMPLE_RATE,
                ),
            )
        }
        require(lengthResults.all { it.actualOutputSamples == it.expectedOutputSamples }) {
            "TorchAudio Hann V1 length fixture failed"
        }
        require(fixtureResults.all { it.exactWholeToChunked && it.withinTolerance }) {
            "TorchAudio Hann V1 numeric or chunk parity failed"
        }
        require(fixtureResults.all { it.insufficientContextRejected != false }) {
            "TorchAudio Hann V1 accepted a source slice with missing context"
        }

        val outputPayload = TorchAudioHannV1BenchmarkOutput(
            generatedAtEpochMs = System.currentTimeMillis(),
            device = "${Build.MANUFACTURER} ${Build.MODEL}",
            soc = deviceSocLabel(),
            androidVersion = "${Build.VERSION.RELEASE} (SDK ${Build.VERSION.SDK_INT})",
            specId = NativeMath.TORCHAUDIO_HANN_V1_SPEC_ID,
            fixtureManifestSha256 = byteSha256(manifestBytes),
            tolerances = TorchAudioHannV1Tolerances(
                maxAbsError = TORCHAUDIO_HANN_V1_MAX_ABS_TOLERANCE,
                rmse = TORCHAUDIO_HANN_V1_RMSE_TOLERANCE,
                cosine = TORCHAUDIO_HANN_V1_COSINE_TOLERANCE,
            ),
            fixtures = fixtureResults,
            lengthCases = lengthResults,
        )
        val output = File(filesDir, "torchaudio_hann_v1_parity.json")
        output.writeText(gson.toJson(outputPayload))
        log("PASS: saved ${output.absolutePath}")
    }

    private fun compareFloatArrays(
        actual: FloatArray,
        expected: FloatArray,
    ): FloatComparison {
        require(actual.size == expected.size)
        var squaredError = 0.0
        var maxAbsError = 0f
        var dot = 0.0
        var actualNorm = 0.0
        var expectedNorm = 0.0
        for (index in actual.indices) {
            val difference = actual[index] - expected[index]
            squaredError += difference * difference
            maxAbsError = maxOf(maxAbsError, kotlin.math.abs(difference))
            dot += actual[index] * expected[index]
            actualNorm += actual[index] * actual[index]
            expectedNorm += expected[index] * expected[index]
        }
        return FloatComparison(
            rmse = kotlin.math.sqrt(squaredError / actual.size),
            maxAbsError = maxAbsError,
            cosine = dot / kotlin.math.sqrt(actualNorm * expectedNorm),
        )
    }

    private fun byteSha256(bytes: ByteArray): String =
        java.security.MessageDigest.getInstance("SHA-256")
            .digest(bytes)
            .joinToString("") { "%02x".format(it) }

    /** Compare exact PCM for one logical CUE span decoded whole versus in chunks. */
    private suspend fun runSpanContinuityBenchmark(onStatus: (String) -> Unit) {
        val lines = StringBuilder()
        fun log(message: String) {
            lines.appendLine(message)
            Log.i(TAG, message)
            onStatus(lines.toString())
        }

        log("=== Exact Span Continuity ===")
        val audit = queryProviderAudit(
            targetIds = emptySet(),
            probeDecode = false,
            probeCueSpans = false,
        ) { message -> log(message) }
        val track = audit.tracks.firstOrNull {
            it.path?.contains("10-31-09") == true &&
                (it.offsetMs ?: 0L) > 0L && (it.offsetMs ?: 0L) % 1000L != 0L
        } ?: error("No sub-second CUE offset found")
        val file = track.path?.let(::File)?.takeIf { it.isFile && it.canRead() }
            ?: error("CUE source is not readable")
        val startMs = track.offsetMs ?: error("CUE offset missing")
        val durationMs = 10_000L
        val splitMs = 4_000L
        val decoder = AudioDecoder()
        val results = mutableListOf<SpanContinuityResult>()

        for (sampleRate in listOf(44_100, MertInference.SAMPLE_RATE)) {
            log("Decoding ${track.title} at ${sampleRate}Hz, start=${startMs}ms...")
            val whole = decoder.decode(
                file,
                targetSampleRate = sampleRate,
                startTimeMs = startMs,
                maxDurationMs = durationMs,
            ) ?: error("Whole-span decode failed at ${sampleRate}Hz")
            val first = decoder.decode(
                file,
                targetSampleRate = sampleRate,
                startTimeMs = startMs,
                maxDurationMs = splitMs,
            ) ?: error("First chunk decode failed at ${sampleRate}Hz")
            val second = decoder.decode(
                file,
                targetSampleRate = sampleRate,
                startTimeMs = startMs + splitMs,
                maxDurationMs = durationMs - splitMs,
            ) ?: error("Second chunk decode failed at ${sampleRate}Hz")
            val chunked = FloatArray(first.samples.size + second.samples.size).also {
                first.samples.copyInto(it, 0)
                second.samples.copyInto(it, first.samples.size)
            }
            val n = minOf(whole.samples.size, chunked.size)
            var sumSq = 0.0
            var maxAbs = 0f
            var dot = 0.0
            var normA = 0.0
            var normB = 0.0
            for (i in 0 until n) {
                val a = whole.samples[i]
                val b = chunked[i]
                val delta = a - b
                sumSq += delta * delta
                maxAbs = maxOf(maxAbs, kotlin.math.abs(delta))
                dot += a * b
                normA += a * a
                normB += b * b
            }
            val rmse = if (n > 0) kotlin.math.sqrt(sumSq / n) else Double.NaN
            val cosine = if (normA > 0.0 && normB > 0.0) dot / kotlin.math.sqrt(normA * normB) else Double.NaN
            val result = SpanContinuityResult(
                sampleRate = sampleRate,
                wholeSamples = whole.samples.size,
                chunkedSamples = chunked.size,
                exact = whole.samples.contentEquals(chunked),
                wholeSha256 = floatSha256(whole.samples),
                chunkedSha256 = floatSha256(chunked),
                rmse = rmse,
                maxAbsError = maxAbs,
                cosine = cosine,
            )
            results += result
            log("  exact=${result.exact}, samples=${result.wholeSamples}/${result.chunkedSamples}, " +
                "rmse=${"%.9g".format(result.rmse)}, max=${"%.9g".format(result.maxAbsError)}, " +
                "cos=${"%.9f".format(result.cosine)}")
        }

        log("Resampling one native-rate decode as globally aligned 4s + 6s slices...")
        val source = decoder.decode(
            file,
            targetSampleRate = 0,
            startTimeMs = startMs,
            maxDurationMs = durationMs,
        ) ?: error("Native-rate whole-span decode failed")
        val targetRate = MertInference.SAMPLE_RATE
        val expectedOutputSamples = (durationMs * targetRate / 1000L).toInt()
        val splitOutputSamples = (splitMs * targetRate / 1000L).toInt()
        val wholeResampled = if (source.sampleRate == targetRate) {
            source.samples
        } else {
            NativeMath.resamplePolyphase(source.samples, source.sampleRate, targetRate)
                ?: error("Whole-span native polyphase resample failed")
        }
        require(wholeResampled.size >= expectedOutputSamples) {
            "Expected $expectedOutputSamples output samples, got ${wholeResampled.size}"
        }
        val whole24k = wholeResampled.copyOfRange(0, expectedOutputSamples)

        // One second of overlap is deliberately generous relative to this FIR's support.
        // The native API independently validates the precise context required per range.
        val sourceBoundary =
            (splitOutputSamples.toLong() * source.sampleRate / targetRate).toInt()
        val contextSamples = source.sampleRate
        val firstInputEnd = (sourceBoundary + contextSamples).coerceAtMost(source.samples.size)
        val secondInputStart = (sourceBoundary - contextSamples).coerceAtLeast(0)
        val firstInput = source.samples.copyOfRange(0, firstInputEnd)
        val secondInput = source.samples.copyOfRange(secondInputStart, source.samples.size)
        val firstAligned = NativeMath.resamplePolyphaseAligned(
            samples = firstInput,
            fromRate = source.sampleRate,
            toRate = targetRate,
            inputStartSample = 0L,
            totalInputSamples = source.samples.size.toLong(),
            outputStartSample = 0L,
            outputSampleCount = splitOutputSamples,
        ) ?: error("First aligned polyphase slice rejected valid context")
        val secondAligned = NativeMath.resamplePolyphaseAligned(
            samples = secondInput,
            fromRate = source.sampleRate,
            toRate = targetRate,
            inputStartSample = secondInputStart.toLong(),
            totalInputSamples = source.samples.size.toLong(),
            outputStartSample = splitOutputSamples.toLong(),
            outputSampleCount = expectedOutputSamples - splitOutputSamples,
        ) ?: error("Second aligned polyphase slice rejected valid context")
        val aligned24k = FloatArray(firstAligned.size + secondAligned.size).also {
            firstAligned.copyInto(it, 0)
            secondAligned.copyInto(it, firstAligned.size)
        }

        // Starting the second slice exactly at its nominal boundary omits the FIR's
        // left context. A rejection demonstrates that an accidental zero-padded seam
        // cannot silently enter production.
        val noContextSecond = source.samples.copyOfRange(sourceBoundary, source.samples.size)
        val insufficientContextRejected = NativeMath.resamplePolyphaseAligned(
            samples = noContextSecond,
            fromRate = source.sampleRate,
            toRate = targetRate,
            inputStartSample = sourceBoundary.toLong(),
            totalInputSamples = source.samples.size.toLong(),
            outputStartSample = splitOutputSamples.toLong(),
            outputSampleCount = expectedOutputSamples - splitOutputSamples,
        ) == null

        var alignedSumSq = 0.0
        var alignedMaxAbs = 0f
        var alignedDot = 0.0
        var alignedNormWhole = 0.0
        var alignedNormSlices = 0.0
        for (i in whole24k.indices) {
            val a = whole24k[i]
            val b = aligned24k[i]
            val delta = a - b
            alignedSumSq += delta * delta
            alignedMaxAbs = maxOf(alignedMaxAbs, kotlin.math.abs(delta))
            alignedDot += a * b
            alignedNormWhole += a * a
            alignedNormSlices += b * b
        }
        val alignedRmse = kotlin.math.sqrt(alignedSumSq / whole24k.size)
        val alignedCosine = alignedDot / kotlin.math.sqrt(alignedNormWhole * alignedNormSlices)
        val alignedPolyphase = AlignedPolyphaseResult(
            sourceSampleRate = source.sampleRate,
            targetSampleRate = targetRate,
            totalInputSamples = source.samples.size,
            firstInputStartSample = 0,
            firstInputSamples = firstInput.size,
            secondInputStartSample = secondInputStart,
            secondInputSamples = secondInput.size,
            firstOutputSamples = firstAligned.size,
            secondOutputSamples = secondAligned.size,
            wholeSamples = whole24k.size,
            alignedSamples = aligned24k.size,
            insufficientContextRejected = insufficientContextRejected,
            exact = whole24k.contentEquals(aligned24k),
            wholeSha256 = floatSha256(whole24k),
            alignedSha256 = floatSha256(aligned24k),
            rmse = alignedRmse,
            maxAbsError = alignedMaxAbs,
            cosine = alignedCosine,
        )
        log("  aligned exact=${alignedPolyphase.exact}, " +
            "shaEqual=${alignedPolyphase.wholeSha256 == alignedPolyphase.alignedSha256}, " +
            "contextRejected=${alignedPolyphase.insufficientContextRejected}, " +
            "samples=${alignedPolyphase.firstOutputSamples}+${alignedPolyphase.secondOutputSamples}, " +
            "rmse=${"%.9g".format(alignedPolyphase.rmse)}, " +
            "max=${"%.9g".format(alignedPolyphase.maxAbsError)}, " +
            "cos=${"%.9f".format(alignedPolyphase.cosine)}")

        val payload = SpanContinuityOutput(
            generatedAtEpochMs = System.currentTimeMillis(),
            track = track,
            startMs = startMs,
            durationMs = durationMs,
            splitMs = splitMs,
            results = results,
            alignedPolyphase = alignedPolyphase,
        )
        val output = File(filesDir, "span_continuity_results.json")
        output.writeText(GsonBuilder().setPrettyPrinting().create().toJson(payload))
        log("Saved ${output.absolutePath}")
    }

    /**
     * Prove that the memory-bounded production cache is invariant to chunk size and
     * produces the same MERT features and final CLaMP3 embedding as one-pass PCM.
     */
    private suspend fun runPcmCacheParityBenchmark(onStatus: (String) -> Unit) {
        val lines = StringBuilder()
        fun log(message: String) {
            lines.appendLine(message)
            Log.i(TAG, message)
            onStatus(lines.toString())
        }

        log("=== Production PCM Cache Parity ===")
        val audit = queryProviderAudit(
            targetIds = emptySet(),
            probeDecode = false,
            probeCueSpans = false,
        ) { message -> log(message) }
        val track = audit.tracks.firstOrNull {
            it.path?.contains("10-31-09") == true &&
                (it.offsetMs ?: 0L) > 0L && (it.offsetMs ?: 0L) % 1000L != 0L
        } ?: error("No sub-second CUE offset found")
        val sourceFile = track.path?.let(::File)?.takeIf { it.isFile && it.canRead() }
            ?: error("CUE source is not readable")
        val startMs = track.offsetMs ?: error("CUE offset missing")
        val durationMs = if (intent.getBooleanExtra("full_span", false)) {
            track.durationMs
        } else {
            10_000L
        }
        val workDir = File(cacheDir, "pcm_cache_parity").apply {
            deleteRecursively()
            mkdirs()
        }

        log("Building one-pass 24 kHz reference for ${track.title}...")
        val decodedReference = AudioDecoder().decode(
            file = sourceFile,
            targetSampleRate = MertInference.SAMPLE_RATE,
            startTimeMs = startMs,
            maxDurationMs = durationMs,
        ) ?: error("Reference decode failed")
        val referencePcm = decodedReference.samples
        val referenceFile = File(workDir, "reference.f32le")
        referenceFile.writeBytes(EmbeddingDatabase.floatArrayToBlob(referencePcm))
        val referenceNormalization = normalizationOf(referencePcm)
        val referenceSha = floatSha256(referencePcm)

        val cacheBuilder = TrackPcmCache()
        val cacheRuns = mutableListOf<PcmCacheRun>()
        val cacheFeatures = mutableListOf<List<FloatArray>>()
        val chunkDurationsMs = if (durationMs > 60_000L) {
            listOf(30_000L, 67_000L, durationMs)
        } else {
            listOf(4_000L, 7_000L, durationMs)
        }
        for (chunkDurationMs in chunkDurationsMs) {
            log("Building cache with ${chunkDurationMs}ms chunks...")
            val cacheFile = File(workDir, "chunk_${chunkDurationMs}.f32le")
            val result = cacheBuilder.build(
                sourceFile = sourceFile,
                logicalStartUs = Math.multiplyExact(startMs, 1000L),
                logicalDurationUs = Math.multiplyExact(durationMs, 1000L),
                chunkDurationMs = chunkDurationMs,
                outputFile = cacheFile,
            )
            val pcm = EmbeddingDatabase.blobToFloatArray(cacheFile.readBytes())
            val exact = pcm.contentEquals(referencePcm)
            val normalizationExact =
                result.normalization == referenceNormalization
            cacheRuns += PcmCacheRun(
                chunkDurationMs = chunkDurationMs,
                chunks = result.chunks,
                sourceSampleRate = result.sourceSampleRate,
                decoderName = result.decoderName,
                sourceChannelCount = result.sourceChannelCount,
                sourcePcmEncoding = result.sourcePcmEncoding,
                samples = pcm.size,
                pcmSha256 = floatSha256(pcm),
                exactPcmToReference = exact,
                normalization = result.normalization,
                exactNormalizationToReference = normalizationExact,
                decodeMs = result.decodeMs,
                resampleMs = result.resampleMs,
            )
            require(exact) { "${chunkDurationMs}ms cache PCM differs from one-pass reference" }
            require(normalizationExact) {
                "${chunkDurationMs}ms cache normalization differs from reference"
            }
            log("  exact PCM=$exact, exact normalization=$normalizationExact, " +
                "sha=${cacheRuns.last().pcmSha256.take(12)}..., chunks=${result.chunks}")
        }

        val mertFile = resolveModelFile(filesDir, "mert")
        require(mertFile.isFile) { "MERT model not found" }
        log("Extracting MERT features from reference and every cache...")
        val referenceFeatures: List<FloatArray>
        val mert = MertInference(mertFile)
        try {
            referenceFeatures = extractVerifiedFeatures(
                mert,
                referenceFile,
                referenceNormalization,
            )
            for ((index, chunkDurationMs) in chunkDurationsMs.withIndex()) {
                val cacheFile = File(workDir, "chunk_${chunkDurationMs}.f32le")
                val features = extractVerifiedFeatures(
                    mert,
                    cacheFile,
                    cacheRuns[index].normalization,
                )
                require(features.size == referenceFeatures.size) {
                    "MERT window count differs for ${chunkDurationMs}ms chunks"
                }
                require(features.indices.all { features[it].contentEquals(referenceFeatures[it]) }) {
                    "MERT features differ for ${chunkDurationMs}ms chunks"
                }
                cacheFeatures += features
            }
        } finally {
            mert.close()
        }
        val referenceFeatureSha = featureSha256(referenceFeatures)

        val clamp3File = resolveModelFile(filesDir, "clamp3_audio")
        require(clamp3File.isFile) { "CLaMP3 audio model not found" }
        log("Encoding reference and cache features with CLaMP3...")
        val referenceEmbedding: FloatArray
        val cacheEmbeddings = mutableListOf<FloatArray>()
        val clamp3 = Clamp3AudioInference(clamp3File)
        try {
            referenceEmbedding = clamp3.encode(referenceFeatures, referenceFeatures.size)
                ?: error("Reference CLaMP3 encoding failed")
            for ((index, features) in cacheFeatures.withIndex()) {
                val embedding = clamp3.encode(features, features.size)
                    ?: error("CLaMP3 encoding failed for ${chunkDurationsMs[index]}ms chunks")
                require(embedding.contentEquals(referenceEmbedding)) {
                    "Embedding differs for ${chunkDurationsMs[index]}ms chunks"
                }
                cacheEmbeddings += embedding
            }
        } finally {
            clamp3.close()
        }

        val completedRuns = cacheRuns.mapIndexed { index, run ->
            run.copy(
                featureWindows = cacheFeatures[index].size,
                featureSha256 = featureSha256(cacheFeatures[index]),
                exactFeaturesToReference = true,
                embeddingSha256 = floatSha256(cacheEmbeddings[index]),
                exactEmbeddingToReference = true,
                embeddingCosineToReference = cosine(
                    cacheEmbeddings[index],
                    referenceEmbedding,
                ),
            )
        }
        val payload = PcmCacheParityOutput(
            generatedAtEpochMs = System.currentTimeMillis(),
            track = track,
            startMs = startMs,
            durationMs = durationMs,
            modelFiles = listOf(mertFile.name, clamp3File.name),
            referencePcmSha256 = referenceSha,
            referenceNormalization = referenceNormalization,
            referenceFeatureWindows = referenceFeatures.size,
            referenceFeatureSha256 = referenceFeatureSha,
            referenceEmbeddingSha256 = floatSha256(referenceEmbedding),
            runs = completedRuns,
        )
        val output = File(
            getExternalFilesDir(null) ?: filesDir,
            "pcm_cache_parity_results.json",
        )
        output.writeText(GsonBuilder().setPrettyPrinting().create().toJson(payload))
        log("PASS: PCM, normalization, MERT features, and embeddings are chunk invariant.")
        log("Saved ${output.absolutePath}")
    }

    private fun extractVerifiedFeatures(
        mert: MertInference,
        pcmFile: File,
        normalization: MertInference.WholeTrackNormalization,
    ): List<FloatArray> = buildList {
        val extracted = mert.extractFeaturesFromPcmFile(
            pcmFile = pcmFile,
            normalization = normalization,
            onFeatureExtracted = { add(it.copyOf()) },
        )
        require(extracted == size) { "MERT reported $extracted windows but emitted $size" }
    }

    private fun normalizationOf(samples: FloatArray): MertInference.WholeTrackNormalization {
        require(samples.isNotEmpty())
        var sum = 0.0
        for (sample in samples) sum += sample
        val mean = (sum / samples.size).toFloat()
        var squaredDeviationSum = 0.0
        for (sample in samples) {
            val difference = sample - mean
            squaredDeviationSum += difference * difference
        }
        val variance = squaredDeviationSum / samples.size
        return MertInference.WholeTrackNormalization(
            sampleCount = samples.size.toLong(),
            mean = mean,
            standardDeviation = kotlin.math.sqrt(variance.toFloat() + 1e-7f),
        )
    }

    private fun featureSha256(features: List<FloatArray>): String {
        val digest = java.security.MessageDigest.getInstance("SHA-256")
        for (feature in features) digest.update(EmbeddingDatabase.floatArrayToBlob(feature))
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    private fun cosine(a: FloatArray, b: FloatArray): Double {
        require(a.size == b.size)
        var dot = 0.0
        var normA = 0.0
        var normB = 0.0
        for (index in a.indices) {
            dot += a[index] * b[index]
            normA += a[index] * a[index]
            normB += b[index] * b[index]
        }
        return dot / kotlin.math.sqrt(normA * normB)
    }

    private fun floatSha256(samples: FloatArray): String =
        java.security.MessageDigest.getInstance("SHA-256")
            .digest(EmbeddingDatabase.floatArrayToBlob(samples))
            .joinToString("") { "%02x".format(it) }

    /**
     * Query provider-native identity/span fields and optionally decode one second from
     * every unmatched logical track. This is deliberately diagnostic-only: it gathers
     * evidence without mutating the DB, hidden-track preferences, or Poweramp queue.
     */
    private fun queryProviderAudit(
        targetIds: Set<Long>,
        probeDecode: Boolean,
        probeCueSpans: Boolean,
        onStatus: (String) -> Unit,
    ): ProviderAudit {
        val filesUri = PowerampHelper.ROOT_URI.buildUpon()
            .appendEncodedPath("files")
            .build()
        val requestedColumns = arrayOf(
            "folder_files._id",
            "artist",
            "album",
            "title_tag",
            "folder_files.duration",
            "path",
            "folder_files.name",
            "folder_files.offset_ms",
            "cue_folder_id",
            "file_type",
            "tag_status",
        )
        val details = mutableListOf<ProviderTrackAudit>()
        var totalRows = 0
        var nonzeroOffsetRows = 0
        var cueSourceRows = 0
        var queryError: String? = null
        var returnedColumns = emptyList<String>()

        onStatus("Querying Poweramp span/container fields...")
        try {
            contentResolver.query(filesUri, requestedColumns, null, null, null)?.use { cursor ->
                returnedColumns = cursor.columnNames.toList()
                val idIdx = cursor.getColumnIndex("_id")
                val artistIdx = cursor.getColumnIndex("artist")
                val albumIdx = cursor.getColumnIndex("album")
                val titleIdx = cursor.getColumnIndex("title_tag")
                val durationIdx = cursor.getColumnIndex("duration")
                val pathIdx = cursor.getColumnIndex("path")
                val nameIdx = cursor.getColumnIndex("name")
                val offsetIdx = cursor.getColumnIndex("offset_ms")
                val cueFolderIdx = cursor.getColumnIndex("cue_folder_id")
                val fileTypeIdx = cursor.getColumnIndex("file_type")
                val tagStatusIdx = cursor.getColumnIndex("tag_status")

                while (cursor.moveToNext()) {
                    totalRows++
                    val id = cursor.getLong(idIdx)
                    val artist = cursor.stringOrNull(artistIdx)
                    val album = cursor.stringOrNull(albumIdx)
                    val title = cursor.stringOrNull(titleIdx)
                    val durationMs = cursor.longOrNull(durationIdx) ?: 0L
                    val offsetMs = cursor.longOrNull(offsetIdx)
                    val cueFolderId = cursor.longOrNull(cueFolderIdx)
                    if ((offsetMs ?: 0L) > 0L) nonzeroOffsetRows++
                    if (cueFolderId != null) cueSourceRows++

                    val isDjShadowEvidence = listOf(artist, album, title)
                        .any { it?.contains("dj shadow", ignoreCase = true) == true }
                    if (id !in targetIds && !isDjShadowEvidence) continue

                    val folder = cursor.stringOrNull(pathIdx).orEmpty()
                    val name = cursor.stringOrNull(nameIdx).orEmpty()
                    val fullPath = if (name.isNotEmpty()) "$folder$name" else null
                    val file = fullPath?.let(::File)
                    val base = ProviderTrackAudit(
                        powerampFileId = id,
                        artist = artist,
                        album = album,
                        title = title,
                        durationMs = durationMs,
                        path = fullPath,
                        offsetMs = offsetMs,
                        cueFolderId = cueFolderId,
                        fileType = cursor.intOrNull(fileTypeIdx),
                        tagStatus = cursor.intOrNull(tagStatusIdx),
                        fileExists = file?.isFile == true,
                        fileReadable = file?.canRead() == true,
                        fileSizeBytes = file?.takeIf { it.isFile }?.length(),
                    )
                    details += inspectContainerAndDecode(
                        base,
                        file,
                        (probeDecode && id in targetIds) || (probeCueSpans && isDjShadowEvidence),
                    )
                }
            }
        } catch (t: Throwable) {
            queryError = "${t.javaClass.simpleName}: ${t.message}"
            Log.e(TAG, "Provider span audit failed", t)
        }

        onStatus(
            "Provider audit: $totalRows rows, $nonzeroOffsetRows logical offsets, " +
                "$cueSourceRows CUE sources, ${details.count { it.powerampFileId in targetIds }} unmatched details"
        )
        return ProviderAudit(
            totalProviderRows = totalRows,
            requestedColumns = requestedColumns.toList(),
            returnedColumns = returnedColumns,
            nonzeroOffsetRows = nonzeroOffsetRows,
            cueSourceRows = cueSourceRows,
            decodeProbeEnabled = probeDecode,
            queryError = queryError,
            tracks = details,
        )
    }

    private fun inspectContainerAndDecode(
        base: ProviderTrackAudit,
        file: File?,
        probeDecode: Boolean,
    ): ProviderTrackAudit {
        if (file?.isFile != true || !file.canRead()) return base

        var mime: String? = null
        var sampleRate: Int? = null
        var channelCount: Int? = null
        var containerDurationUs: Long? = null
        var extractorError: String? = null
        val extractor = MediaExtractor()
        try {
            extractor.setDataSource(file.absolutePath)
            for (i in 0 until extractor.trackCount) {
                val format = extractor.getTrackFormat(i)
                val candidateMime = format.getString(MediaFormat.KEY_MIME)
                if (candidateMime?.startsWith("audio/") != true) continue
                mime = candidateMime
                if (format.containsKey(MediaFormat.KEY_SAMPLE_RATE)) {
                    sampleRate = format.getInteger(MediaFormat.KEY_SAMPLE_RATE)
                }
                if (format.containsKey(MediaFormat.KEY_CHANNEL_COUNT)) {
                    channelCount = format.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
                }
                if (format.containsKey(MediaFormat.KEY_DURATION)) {
                    containerDurationUs = format.getLong(MediaFormat.KEY_DURATION)
                }
                break
            }
            if (mime == null) extractorError = "no audio track"
        } catch (t: Throwable) {
            extractorError = "${t.javaClass.simpleName}: ${t.message}"
        } finally {
            extractor.release()
        }

        var decodeSucceeded: Boolean? = null
        var decodeError: String? = null
        var decodeMs: Long? = null
        var decodeSamples: Int? = null
        var decodeSha256: String? = null
        if (probeDecode && extractorError == null) {
            val startedNs = System.nanoTime()
            try {
                // Start at the logical Poweramp CUE offset. This exposes whether the
                // production t=0 behavior is embedding a different section of the file.
                val audio = AudioDecoder().decode(
                    file = file,
                    targetSampleRate = MertInference.SAMPLE_RATE,
                    maxDurationS = 1,
                    startTimeS = ((base.offsetMs ?: 0L) / 1000L).toInt(),
                    startTimeMs = base.offsetMs ?: 0L,
                    maxDurationMs = 1000L,
                )
                decodeSucceeded = audio?.samples?.isNotEmpty() == true
                if (audio != null) {
                    decodeSamples = audio.samples.size
                    decodeSha256 = java.security.MessageDigest.getInstance("SHA-256")
                        .digest(EmbeddingDatabase.floatArrayToBlob(audio.samples))
                        .joinToString("") { "%02x".format(it) }
                }
                if (decodeSucceeded == false) decodeError = "decoder returned no samples"
            } catch (t: Throwable) {
                decodeSucceeded = false
                decodeError = "${t.javaClass.simpleName}: ${t.message}"
            } finally {
                decodeMs = (System.nanoTime() - startedNs) / 1_000_000L
            }
        }

        return base.copy(
            extractorMime = mime,
            extractorSampleRate = sampleRate,
            extractorChannelCount = channelCount,
            containerDurationUs = containerDurationUs,
            extractorError = extractorError,
            decodeProbeSucceeded = decodeSucceeded,
            decodeProbeMs = decodeMs,
            decodeProbeSamples = decodeSamples,
            decodeProbeSampleSha256 = decodeSha256,
            decodeProbeError = decodeError,
        )
    }

    private fun android.database.Cursor.stringOrNull(index: Int): String? =
        if (index < 0 || isNull(index)) null else getString(index)

    private fun android.database.Cursor.longOrNull(index: Int): Long? =
        if (index < 0 || isNull(index)) null else getLong(index)

    private fun android.database.Cursor.intOrNull(index: Int): Int? =
        if (index < 0 || isNull(index)) null else getInt(index)

    // ── Text Search Benchmark ─────────────────────────────────────────────────

    private val textTestQueries = listOf(
        "ethereal ambient",
        "heavy metal guitar",
        "jazz piano trio",
        "electronic dance music",
        "sufi devotional music",
        "lo-fi hip hop beats",
        "orchestral film score",
        "acoustic folk guitar",
    )

    private suspend fun runTextBenchmark(onStatus: (String) -> Unit) {
        val sb = StringBuilder()
        fun log(msg: String) {
            sb.appendLine(msg)
            Log.i(TAG, msg)
            onStatus(sb.toString())
        }

        log("=== CLaMP3 Text Search Benchmark ===")
        log("Device: ${Build.MANUFACTURER} ${Build.MODEL}")
        log("SOC: ${deviceSocLabel()}")
        log("")

        // Resolve text model + its authoritative SentencePiece model.
        val textModelFile = resolveModelFile(filesDir, "clamp3_text")
        val tokenizerModelFile = File(filesDir, "sentencepiece.bpe.model")

        if (!textModelFile.exists()) {
            log("ERROR: CLaMP3 text model not found at ${textModelFile.absolutePath}")
            log("Push clamp3_text.tflite or clamp3_text_fp16.tflite to ${filesDir.absolutePath}")
            return
        }
        if (!tokenizerModelFile.exists()) {
            log("ERROR: SentencePiece model not found at ${tokenizerModelFile.absolutePath}")
            log("Push sentencepiece.bpe.model to ${filesDir.absolutePath}")
            return
        }

        // Load text model
        log("Loading CLaMP3 text model...")
        val loadStart = System.nanoTime()
        val textInference = try {
            Clamp3TextInference(textModelFile, tokenizerModelFile)
        } catch (e: Exception) {
            log("Text model load FAILED: ${e.message}")
            log(e.stackTraceToString())
            return
        }
        val loadMs = (System.nanoTime() - loadStart) / 1_000_000
        val textAccel = textInference.activeAccelerator.name
        log("  Loaded in ${loadMs}ms (accelerator: $textAccel)")
        log("  Model: ${textModelFile.name} (${textModelFile.length() / 1024 / 1024}MB)")
        log("  Tokenizer: ${tokenizerModelFile.name} (${tokenizerModelFile.length() / 1024}KB)")
        log("")

        // Resolve DB and PEMB from one validated immutable generation.
        val generation = try {
            V2LibraryDatabaseResolver.requirePublished(filesDir)
        } catch (e: Exception) {
            log("ERROR: no valid published V2 generation: ${e.message}")
            textInference.close()
            return
        }
        val embFile = generation.embeddingFile
        val dbFile = generation.databaseFile
        var embIndex: EmbeddingIndex? = null

        log(
            "Loading generation ${generation.manifest.generationId} embedding index: " +
                "${embFile.name} (${embFile.length() / 1024 / 1024}MB)",
        )
        try {
            embIndex = EmbeddingIndex.mmap(embFile)
            log("  ${embIndex.numTracks} tracks, ${embIndex.dim}d")
        } catch (e: Exception) {
            log("  WARNING: Failed to load embedding index: ${e.message}")
        }

        // Open DB for track metadata lookups (getTrackById for each search hit)
        var metaDb: EmbeddingDatabase? = null
        if (embIndex != null && dbFile.exists()) {
            try {
                metaDb = EmbeddingDatabase.open(dbFile)
                log("  DB opened for metadata (${metaDb.getTrackCount()} tracks)")
            } catch (e: Exception) {
                log("  WARNING: Failed to open DB for metadata: ${e.message}")
            }
        }
        log("")

        // Run test queries
        val queryResults = mutableListOf<TextQueryResult>()
        val debugDir = File(filesDir, "text_benchmark")

        for ((qi, query) in textTestQueries.withIndex()) {
            log("Query [${qi + 1}/${textTestQueries.size}]: \"$query\"")

            val t0 = System.nanoTime()
            val embedding = textInference.generateEmbedding(query, debugDir = debugDir)
            val totalMs = (System.nanoTime() - t0) / 1_000_000

            if (embedding == null) {
                log("  FAILED: null embedding")
                queryResults.add(TextQueryResult(query = query, error = "null embedding"))
                continue
            }

            // Check embedding quality
            val nanCount = embedding.count { it.isNaN() }
            val infCount = embedding.count { it.isInfinite() }
            val norm = kotlin.math.sqrt(embedding.sumOf { (it * it).toDouble() }).toFloat()
            val absMax = embedding.maxOf { kotlin.math.abs(it) }

            if (nanCount > 0 || infCount > 0) {
                log("  WARNING: $nanCount NaN, $infCount Inf in ${embedding.size}d embedding")
            }
            log("  ${embedding.size}d, norm=${"%.4f".format(norm)}, absmax=${"%.4f".format(absMax)}, time=${totalMs}ms")

            // Search against audio embeddings
            var searchResults: List<TextSearchHit>? = null
            if (embIndex != null) {
                val searchStart = System.nanoTime()
                val topK = embIndex.findTopK(embedding, 10)
                val searchMs = (System.nanoTime() - searchStart) / 1_000_000
                log("  Search: ${embIndex.numTracks} tracks in ${searchMs}ms")

                searchResults = topK.map { (trackId, sim) ->
                    val track = metaDb?.getTrackById(trackId)
                    val label = if (track != null) {
                        "${track.artist} - ${track.title}"
                    } else "trackId=$trackId"
                    TextSearchHit(trackId = trackId, similarity = sim, label = label)
                }

                for ((rank, hit) in searchResults.withIndex()) {
                    log("    ${rank + 1}. ${"%.4f".format(hit.similarity)}  ${hit.label}")
                }
            }

            queryResults.add(TextQueryResult(
                query = query,
                dim = embedding.size,
                norm = norm,
                absMax = absMax,
                totalMs = totalMs,
                embedding = embedding.toList(),
                topMatches = searchResults,
            ))
            log("")
        }

        textInference.close()
        metaDb?.close()
        log("Text model closed.")

        // Save results JSON
        val output = TextBenchmarkOutput(
            device = "${Build.MANUFACTURER} ${Build.MODEL}",
            soc = deviceSocLabel(),
            androidVersion = "${Build.VERSION.RELEASE} (SDK ${Build.VERSION.SDK_INT})",
            textModel = textModelFile.name,
            textAccelerator = textAccel,
            textLoadMs = loadMs,
            numAudioTracks = embIndex?.numTracks ?: 0,
            queries = queryResults,
        )

        val gson = GsonBuilder().setPrettyPrinting().create()
        val json = gson.toJson(output)
        val outputFile = File(filesDir, "text_benchmark_results.json")
        outputFile.writeText(json)

        log("\n=== Results saved ===")
        log("File: ${outputFile.absolutePath}")
        log("Pull: adb shell run-as com.powerampstartradio.v2 cat files/text_benchmark_results.json")

        // Timing summary
        log("\n=== Timing Summary ===")
        log("Model load: ${loadMs}ms ($textAccel)")
        log("")
        log(String.format(Locale.ROOT, "%-30s %8s %8s %8s", "Query", "Total", "Norm", "AbsMax"))
        log("-".repeat(60))
        for (r in queryResults) {
            if (r.error != null) {
                log(String.format(Locale.ROOT, "%-30s %8s", r.query.take(30), "FAILED"))
            } else {
                log(String.format(Locale.ROOT, "%-30s %7dms %7.4f %7.4f",
                    r.query.take(30), r.totalMs, r.norm, r.absMax))
            }
        }

        val successfulQueries = queryResults.filter { it.error == null }
        if (successfulQueries.isNotEmpty()) {
            val avgMs = successfulQueries.map { it.totalMs }.average().toLong()
            val avgNorm = successfulQueries.map { it.norm.toDouble() }.average()
            log("-".repeat(60))
            log(String.format(Locale.ROOT, "%-30s %7dms %7.4f",
                "AVERAGE (${successfulQueries.size})", avgMs, avgNorm))
        }

        log("\nText benchmark complete.")
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

    data class ProviderAudit(
        val totalProviderRows: Int,
        val requestedColumns: List<String>,
        val returnedColumns: List<String>,
        val nonzeroOffsetRows: Int,
        val cueSourceRows: Int,
        val decodeProbeEnabled: Boolean,
        val queryError: String?,
        val tracks: List<ProviderTrackAudit>,
    )

    data class ProviderTrackAudit(
        val powerampFileId: Long,
        val artist: String?,
        val album: String?,
        val title: String?,
        val durationMs: Long,
        val path: String?,
        val offsetMs: Long?,
        val cueFolderId: Long?,
        val fileType: Int?,
        val tagStatus: Int?,
        val fileExists: Boolean,
        val fileReadable: Boolean,
        val fileSizeBytes: Long?,
        val extractorMime: String? = null,
        val extractorSampleRate: Int? = null,
        val extractorChannelCount: Int? = null,
        val containerDurationUs: Long? = null,
        val extractorError: String? = null,
        val decodeProbeSucceeded: Boolean? = null,
        val decodeProbeMs: Long? = null,
        val decodeProbeSamples: Int? = null,
        val decodeProbeSampleSha256: String? = null,
        val decodeProbeError: String? = null,
    )

    data class TorchAudioHannV1FixtureManifest(
        val specId: String,
        val torchVersion: String,
        val torchaudioVersion: String,
        val resamplingMethod: String,
        val lowpassFilterWidth: Int,
        val rolloff: Double,
        val targetLength: String,
        val fixtures: List<TorchAudioHannV1Fixture>,
    )

    data class TorchAudioHannV1Fixture(
        val name: String,
        val fromRate: Int,
        val toRate: Int,
        val inputFile: String,
        val inputSamples: Int,
        val inputSha256: String,
        val expectedFile: String,
        val expectedSamples: Int,
        val expectedSha256: String,
        val chunkSchedule: List<Int>,
    )

    data class TorchAudioHannV1BenchmarkOutput(
        val generatedAtEpochMs: Long,
        val device: String,
        val soc: String,
        val androidVersion: String,
        val specId: String,
        val fixtureManifestSha256: String,
        val tolerances: TorchAudioHannV1Tolerances,
        val fixtures: List<TorchAudioHannV1FixtureResult>,
        val lengthCases: List<TorchAudioHannV1LengthCase>,
    )

    data class TorchAudioHannV1Tolerances(
        val maxAbsError: Float,
        val rmse: Double,
        val cosine: Double,
    )

    data class TorchAudioHannV1FixtureResult(
        val name: String,
        val fromRate: Int,
        val toRate: Int,
        val inputSamples: Int,
        val expectedSamples: Int,
        val nativeSamples: Int,
        val chunks: Int,
        val wholeMs: Long,
        val expectedSha256: String,
        val nativeSha256: String,
        val chunkedSha256: String,
        val exactWholeToChunked: Boolean,
        val insufficientContextRejected: Boolean?,
        val rmseToTorchAudio: Double,
        val maxAbsErrorToTorchAudio: Float,
        val cosineToTorchAudio: Double,
        val withinTolerance: Boolean,
    )

    data class TorchAudioHannV1LengthCase(
        val inputSamples: Long,
        val fromRate: Int,
        val expectedOutputSamples: Long,
        val actualOutputSamples: Long = -1L,
    )

    data class FloatComparison(
        val rmse: Double,
        val maxAbsError: Float,
        val cosine: Double,
    )

    data class SpanContinuityOutput(
        val generatedAtEpochMs: Long,
        val track: ProviderTrackAudit,
        val startMs: Long,
        val durationMs: Long,
        val splitMs: Long,
        val results: List<SpanContinuityResult>,
        val alignedPolyphase: AlignedPolyphaseResult,
    )

    data class SpanContinuityResult(
        val sampleRate: Int,
        val wholeSamples: Int,
        val chunkedSamples: Int,
        val exact: Boolean,
        val wholeSha256: String,
        val chunkedSha256: String,
        val rmse: Double,
        val maxAbsError: Float,
        val cosine: Double,
    )

    data class AlignedPolyphaseResult(
        val sourceSampleRate: Int,
        val targetSampleRate: Int,
        val totalInputSamples: Int,
        val firstInputStartSample: Int,
        val firstInputSamples: Int,
        val secondInputStartSample: Int,
        val secondInputSamples: Int,
        val firstOutputSamples: Int,
        val secondOutputSamples: Int,
        val wholeSamples: Int,
        val alignedSamples: Int,
        val insufficientContextRejected: Boolean,
        val exact: Boolean,
        val wholeSha256: String,
        val alignedSha256: String,
        val rmse: Double,
        val maxAbsError: Float,
        val cosine: Double,
    )

    data class PcmCacheParityOutput(
        val generatedAtEpochMs: Long,
        val track: ProviderTrackAudit,
        val startMs: Long,
        val durationMs: Long,
        val modelFiles: List<String>,
        val referencePcmSha256: String,
        val referenceNormalization: MertInference.WholeTrackNormalization,
        val referenceFeatureWindows: Int,
        val referenceFeatureSha256: String,
        val referenceEmbeddingSha256: String,
        val runs: List<PcmCacheRun>,
    )

    data class PcmCacheRun(
        val chunkDurationMs: Long,
        val chunks: Int,
        val sourceSampleRate: Int,
        val decoderName: String,
        val sourceChannelCount: Int,
        val sourcePcmEncoding: Int,
        val samples: Int,
        val pcmSha256: String,
        val exactPcmToReference: Boolean,
        val normalization: MertInference.WholeTrackNormalization,
        val exactNormalizationToReference: Boolean,
        val decodeMs: Long,
        val resampleMs: Long,
        val featureWindows: Int = 0,
        val featureSha256: String = "",
        val exactFeaturesToReference: Boolean = false,
        val embeddingSha256: String = "",
        val exactEmbeddingToReference: Boolean = false,
        val embeddingCosineToReference: Double = Double.NaN,
    )

    data class BenchmarkOutput(
        val device: String,
        val soc: String,
        val androidVersion: String,
        val runtime: String,
        val mertModel: String? = null,
        val mertAccelerator: String? = null,
        val mertLoadMs: Long = 0,
        val clamp3Model: String? = null,
        val clamp3Accelerator: String? = null,
        val clamp3LoadMs: Long = 0,
        val tracks: List<TrackResult>,
    )

    data class TrackResult(
        val path: String,
        val artist: String,
        val album: String,
        val title: String,
        val durationMs: Long,
        var durationS: Float = 0f,
        var timing: TrackTiming? = null,
        var clamp3: EmbeddingResult? = null,
    )

    data class TrackTiming(
        val decodeMs: Long,
        val resampleMs: Long = 0,
        val resampleQuality: String = "hq",
        val mertTotalMs: Long,
        val mertWindows: Int,
        val mertPerWindowMs: List<Long>,
        val mertAvgWindowMs: Long,
        val clamp3Segments: Int,
        val clamp3TotalMs: Long,
        val totalMs: Long,
        val realtimeFactor: Float,
    )

    data class EmbeddingResult(
        val dim: Int,
        val embedding: List<Float>,
    )

    // Text benchmark data classes
    data class TextBenchmarkOutput(
        val device: String,
        val soc: String,
        val androidVersion: String,
        val textModel: String,
        val textAccelerator: String,
        val textLoadMs: Long,
        val numAudioTracks: Int,
        val queries: List<TextQueryResult>,
    )

    data class TextQueryResult(
        val query: String,
        val dim: Int = 0,
        val norm: Float = 0f,
        val absMax: Float = 0f,
        val totalMs: Long = 0,
        val embedding: List<Float>? = null,
        val topMatches: List<TextSearchHit>? = null,
        val error: String? = null,
    )

    data class TextSearchHit(
        val trackId: Long,
        val similarity: Float,
        val label: String,
    )
}
