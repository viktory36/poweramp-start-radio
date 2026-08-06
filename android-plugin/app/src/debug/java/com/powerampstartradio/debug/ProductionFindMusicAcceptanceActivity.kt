package com.powerampstartradio.debug

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.SharedPreferences
import android.os.BatteryManager
import android.os.Build
import android.os.Bundle
import android.os.Debug
import android.os.PowerManager
import android.os.SystemClock
import android.util.AtomicFile
import android.util.Base64
import android.util.Log
import android.view.Gravity
import android.view.WindowManager
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.lifecycleScope
import com.google.gson.GsonBuilder
import com.powerampstartradio.similarity.FindMusicTextQueuePlanEvidence
import com.powerampstartradio.ui.DatabaseInfo
import com.powerampstartradio.ui.DisplayedFindMusicQueueEligibility
import com.powerampstartradio.ui.FindMusicResultKind
import com.powerampstartradio.ui.FindMusicQuerySpec
import com.powerampstartradio.ui.FindMusicTextResultPlanner
import com.powerampstartradio.ui.MainViewModel
import com.powerampstartradio.ui.RadioSettingsStore
import com.powerampstartradio.ui.StableResultReductionEvidence
import com.powerampstartradio.ui.TextSearchResult
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.filterNotNull
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout
import kotlinx.coroutines.yield
import java.io.File
import java.io.FileNotFoundException
import java.io.FileOutputStream
import java.io.OutputStreamWriter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import java.time.Instant
import java.util.TreeMap
import java.util.UUID
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.locks.LockSupport

private val TEXT_ACCEPTANCE_PLANNERS = listOf(
    FindMusicTextResultPlanner.CLOSEST,
    FindMusicTextResultPlanner.VARIED_DPP,
)

/**
 * Debug-only empirical acceptance for the exact production simple-text Find Music path.
 *
 * The activity drives [MainViewModel]'s public controls and result flow. It never imports a
 * Poweramp queue API. The host is expected to compare the Poweramp queue before and after the run.
 */
class ProductionFindMusicAcceptanceActivity : ComponentActivity() {
    private lateinit var statusView: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        showOverLockScreenForAcceptance()
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        window.attributes = window.attributes.apply { screenBrightness = 0.01f }
        statusView = TextView(this).apply {
            gravity = Gravity.CENTER
            textSize = 16f
            setPadding(48, 48, 48, 48)
            text = "Preparing production Find Music acceptance..."
        }
        setContentView(statusView)

        if (savedInstanceState == null) {
            val viewModel = ViewModelProvider(this)[MainViewModel::class.java]
            lifecycleScope.launch {
                val request = ProductionFindMusicAcceptanceRequest.from(intent.extras)
                val wakeLock = getSystemService(PowerManager::class.java).newWakeLock(
                    PowerManager.PARTIAL_WAKE_LOCK,
                    "$packageName:production-find-music-acceptance",
                )
                wakeLock.acquire()
                val finalStatus = try {
                    ProductionFindMusicAcceptanceRunner(
                        context = applicationContext,
                        filesDir = filesDir,
                        viewModel = viewModel,
                    ).run(request) { message -> statusView.text = message }
                } finally {
                    if (wakeLock.isHeld) wakeLock.release()
                }
                statusView.text = finalStatus
                finishAndRemoveTask()
            }
        }
    }
}

private data class ProductionFindMusicAcceptanceRequest(
    val runId: String,
    val queries: List<String>,
    val resultCounts: List<Int>,
    val repeatCount: Int,
    val timeoutPerCaseSeconds: Int,
    val memorySampleIntervalMs: Int,
    val includeCancellation: Boolean,
) {
    val plannedSearchRunCount: Int
        get() = Math.multiplyExact(
            Math.multiplyExact(queries.size, resultCounts.size),
            Math.multiplyExact(repeatCount, TEXT_ACCEPTANCE_PLANNERS.size),
        )

    companion object {
        fun from(extras: Bundle?): ProductionFindMusicAcceptanceRequest {
            val encodedQueries = extras?.getString("queries_b64")?.let { encoded ->
                runCatching {
                    String(Base64.decode(encoded, Base64.NO_WRAP), Charsets.UTF_8)
                }.getOrNull()
            }
            val queries = (encodedQueries ?: extras?.getString("queries"))
                ?.split('|')
                ?.map(String::trim)
                ?.filter(String::isNotBlank)
                ?.distinct()
                ?.take(20)
                ?.takeIf(List<String>::isNotEmpty)
                ?: DEFAULT_QUERIES
            val resultCounts = extras?.getString("result_counts")
                ?.split(',')
                ?.mapNotNull { it.trim().toIntOrNull() }
                ?.filter { it in 10..100 }
                ?.distinct()
                ?.sorted()
                ?.takeIf(List<Int>::isNotEmpty)
                ?: listOf(20, 30, 50)
            val requestedRunId = extras?.getString("run_id")?.trim()
            val runId = requestedRunId?.takeIf { it.matches(RUN_ID) }
                ?: UUID.randomUUID().toString()
            return ProductionFindMusicAcceptanceRequest(
                runId = runId,
                queries = queries,
                resultCounts = resultCounts,
                repeatCount = (extras?.getInt("repeat_count", 3) ?: 3).coerceIn(1, 5),
                timeoutPerCaseSeconds =
                    (extras?.getInt("timeout_per_case_seconds", 180) ?: 180)
                        .coerceIn(30, 600),
                memorySampleIntervalMs =
                    (extras?.getInt("memory_sample_interval_ms", 3) ?: 3)
                        .coerceIn(2, 5),
                includeCancellation = extras?.getBoolean("include_cancellation", true) ?: true,
            )
        }

        private val RUN_ID = Regex("^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
        private val DEFAULT_QUERIES = listOf(
            "ambient",
            "sleep",
            "relaxing",
            "slow",
            "psychedelic",
            "guitar",
            "late night downtempo",
            "organic electronic",
            "spacey jazz",
        )
    }
}

private data class ProductionFindMusicAcceptanceMemory(
    val sampleIntervalMs: Int,
    val sampleCount: Long,
    val javaUsedBytesAtStart: Long,
    val javaUsedBytesAtEnd: Long,
    val peakJavaUsedBytes: Long,
    val javaCommittedBytesAtEnd: Long,
    val javaMaximumBytes: Long,
    val nativeAllocatedBytesAtStart: Long,
    val nativeAllocatedBytesAtEnd: Long,
    val peakNativeAllocatedBytes: Long,
    val pssKbAtStart: Long,
    val pssKbAtEnd: Long,
    val processVmPeakKbAtEnd: Long?,
    val processVmHwmKbAtEnd: Long?,
    val processVmRssKbAtEnd: Long?,
)

private data class ProductionFindMusicDeviceCondition(
    val capturedAtElapsedRealtimeMs: Long,
    val thermalStatus: Int?,
    val thermalStatusName: String,
    val batteryTemperatureTenthsCelsius: Int?,
    val batteryTemperatureCelsius: Float?,
)

private data class ProductionFindMusicAcceptanceTrack(
    val displayedPosition: Int,
    val embeddedTrackId: Long,
    val stableTrackSpanId: String?,
    val objectiveRank: Int,
    val artist: String?,
    val album: String?,
    val title: String?,
    val durationMs: Int,
    val filePath: String,
    val textSimilarity: Float,
    val rankingScore: Float,
)

private data class ProductionFindMusicAcceptanceSearchRun(
    val query: String,
    val publishedQuery: String?,
    val publishedQuerySpec: FindMusicQuerySpec?,
    val resultCount: Int,
    val planner: FindMusicTextResultPlanner,
    val plannerVersion: Int,
    val repeat: Int,
    val elapsedToResultMs: Long,
    val elapsedToQueueReadyMs: Long,
    val resultFingerprint: String?,
    val resultKind: FindMusicResultKind?,
    val error: String?,
    val libraryGenerationId: String?,
    val activationBindingId: String?,
    val providerGenerationId: String?,
    val orderedActiveTrackIdsSha256: String?,
    val activeTrackCount: Int?,
    val objectiveRankingDomainCount: Int?,
    val stableResultReduction: StableResultReductionEvidence?,
    val textQueuePlan: FindMusicTextQueuePlanEvidence?,
    val queueEligibility: DisplayedFindMusicQueueEligibility,
    val deviceConditionAtStart: ProductionFindMusicDeviceCondition,
    val deviceConditionAtEnd: ProductionFindMusicDeviceCondition,
    val timingInstrumentedByMemorySampler: Boolean,
    val memory: ProductionFindMusicAcceptanceMemory?,
    val tracks: List<ProductionFindMusicAcceptanceTrack>,
)

private data class ProductionFindMusicCancellationEvidence(
    val firstQuery: String,
    val secondQuery: String,
    val supersedeDelayMs: Long,
    val elapsedAfterSupersedeMs: Long,
    val conflatedStateFlowQueriesObserved: List<String>,
    val resultVisibleAtSupersedeBoundary: String?,
    val finalQuery: String?,
    val finalPlanner: FindMusicTextResultPlanner?,
    val latestResultSmokePassed: Boolean,
    val nonConflatingPublicationAuditAvailable: Boolean = false,
    val supersededRequestNonPublicationProven: Boolean = false,
    val evidenceLimitation: String =
        "StateFlow is conflated; this case checks the boundary and final state, not publication history.",
    val finalResultFingerprint: String?,
    val finalQueueEligibility: DisplayedFindMusicQueueEligibility,
    val deviceConditionAtStart: ProductionFindMusicDeviceCondition,
    val deviceConditionAtEnd: ProductionFindMusicDeviceCondition,
    val memory: ProductionFindMusicAcceptanceMemory,
    val error: String? = null,
)

private data class ProductionFindMusicDeterminismEvaluation(
    val observationCount: Int,
    val evaluated: Boolean,
    val deterministic: Boolean?,
)

private data class ProductionFindMusicAcceptanceStatus(
    val schemaVersion: Int = 1,
    val state: String,
    val runId: String,
    val startedAt: String,
    val updatedAt: String,
    val plannedSearchRunCount: Int,
    val completedSearchRunCount: Int,
    val currentCase: String?,
    val resultsJsonlFile: String,
    val fatalError: String? = null,
)

private data class ProductionFindMusicAcceptanceReport(
    val schemaVersion: Int = 2,
    val state: String,
    val runId: String,
    val startedAt: String,
    val completedAt: String,
    val request: ProductionFindMusicAcceptanceRequest,
    val startupReadyMs: Long?,
    val databaseInfo: DatabaseInfo?,
    val queueMutationApisCalled: Int = 0,
    val settingsSnapshotSha256Before: String,
    val settingsSnapshotSha256AfterRestore: String?,
    val recoveredSettingsFromPriorInterruptedRun: Boolean,
    val settingsRestoredExactly: Boolean,
    val searchRuns: List<ProductionFindMusicAcceptanceSearchRun>,
    val repeatDeterminism: Map<String, ProductionFindMusicDeterminismEvaluation>,
    val crossCountPrefixDeterminism: Map<String, ProductionFindMusicDeterminismEvaluation>,
    val cancellation: ProductionFindMusicCancellationEvidence?,
    val fatalError: String? = null,
)

private class ProductionFindMusicAcceptanceRunner(
    context: Context,
    private val filesDir: File,
    private val viewModel: MainViewModel,
) {
    companion object {
        private const val TAG = "ProductionFindMusicAcceptance"
        private val gson = GsonBuilder().setPrettyPrinting().disableHtmlEscaping().create()
        private val compactGson = GsonBuilder().disableHtmlEscaping().create()
        private val SHA256 = Regex("^[0-9a-f]{64}$")
        private const val CANCELLATION_SUPERSEDE_DELAY_MS = 50L
    }

    private val preferences = context.getSharedPreferences(
        RadioSettingsStore.PREFERENCES_NAME,
        Context.MODE_PRIVATE,
    )
    private val appContext = context.applicationContext
    private val outputDir = File(filesDir, "production_find_music_acceptance")

    suspend fun run(
        request: ProductionFindMusicAcceptanceRequest,
        onStatus: (String) -> Unit,
    ): String {
        require(outputDir.isDirectory || outputDir.mkdirs()) {
            "Cannot create ${outputDir.absolutePath}"
        }
        val startedAt = Instant.now().toString()
        val statusFile = AtomicFile(File(outputDir, "status.json"))
        val resultsJsonl = File(outputDir, "${request.runId}.runs.jsonl")
        val finalFile = File(outputDir, "${request.runId}.json")
        val settingsRecoveryFile = AtomicFile(File(outputDir, "settings-recovery.json"))
        val recoveredSettings = readAtomicIfExists(settingsRecoveryFile)?.let { encoded ->
            gson.fromJson(encoded, PreferenceSnapshot::class.java).also { snapshot ->
                require(snapshot.restore(preferences) && PreferenceSnapshot.capture(preferences) == snapshot) {
                    "Could not restore settings left by an interrupted acceptance run"
                }
            }
        }
        if (recoveredSettings != null) settingsRecoveryFile.delete()
        require(!resultsJsonl.exists() && !finalFile.exists()) {
            "Acceptance run ${request.runId} already exists"
        }
        val settingsBefore = PreferenceSnapshot.capture(preferences)
        val settingsBeforeSha = settingsBefore.sha256()
        writeAtomic(settingsRecoveryFile, gson.toJson(settingsBefore))
        val runs = mutableListOf<ProductionFindMusicAcceptanceSearchRun>()
        var startupReadyMs: Long? = null
        var databaseInfo: DatabaseInfo? = null
        var cancellationEvidence: ProductionFindMusicCancellationEvidence? = null
        var fatalError: Throwable? = null

        fun checkpoint(state: String, currentCase: String?, failure: Throwable? = null) {
            writeAtomic(
                statusFile,
                gson.toJson(
                    ProductionFindMusicAcceptanceStatus(
                        state = state,
                        runId = request.runId,
                        startedAt = startedAt,
                        updatedAt = Instant.now().toString(),
                        plannedSearchRunCount = request.plannedSearchRunCount,
                        completedSearchRunCount = runs.size,
                        currentCase = currentCase,
                        resultsJsonlFile = resultsJsonl.name,
                        fatalError = failure?.stackTraceToString(),
                    ),
                ),
            )
        }

        checkpoint("RUNNING", "waiting_for_verified_library")
        try {
            val startupStarted = SystemClock.elapsedRealtimeNanos()
            onStatus("Waiting for the verified active library...")
            databaseInfo = withTimeout(request.timeoutPerCaseSeconds * 1_000L) {
                combine(viewModel.databaseLoading, viewModel.databaseInfo) { loading, info ->
                    if (!loading) info else null
                }.filterNotNull().first()
            }
            startupReadyMs = elapsedMs(startupStarted)
            require(databaseInfo?.generationId != null && databaseInfo?.activeTrackCount != null) {
                "Production library readiness did not publish an exact active generation"
            }

            for (query in request.queries) {
                for (resultCount in request.resultCounts) {
                    for (planner in TEXT_ACCEPTANCE_PLANNERS) {
                        for (repeat in 1..request.repeatCount) {
                            val case = "$query | $resultCount | ${planner.wireName} | $repeat"
                            onStatus(
                                "$case\n${runs.size}/${request.plannedSearchRunCount} complete",
                            )
                            checkpoint("RUNNING", case)
                            val run = executeSearch(
                                request = request,
                                settingsBaseline = settingsBefore,
                                query = query,
                                resultCount = resultCount,
                                planner = planner,
                                repeat = repeat,
                            )
                            require(settingsBefore.restore(preferences)) {
                                "Could not restore settings after a Find Music case"
                            }
                            validateSearchRun(run, requireNotNull(databaseInfo))
                            runs += run
                            appendJsonLine(resultsJsonl, run)
                            checkpoint("RUNNING", null)
                        }
                    }
                }
            }

            if (request.includeCancellation) {
                onStatus("Superseding one in-flight Varied request...")
                checkpoint("RUNNING", "latest_request_cancellation")
                cancellationEvidence = executeCancellation(request, settingsBefore)
                require(cancellationEvidence.latestResultSmokePassed) {
                    "Latest-request cancellation smoke check failed"
                }
            }
        } catch (failure: Throwable) {
            fatalError = failure
            Log.e(TAG, "Acceptance run ${request.runId} failed", failure)
        } finally {
            viewModel.clearFindMusicResults()
        }

        val settingsRestored = settingsBefore.restore(preferences)
        // MainViewModel uses apply(). Its in-memory mutations happened synchronously, while this
        // committed restore is ordered after them; allow any older queued disk write to drain
        // before proving the final semantic map and returning control to a fresh ViewModel.
        withContext(NonCancellable) { delay(250L) }
        val settingsAfter = PreferenceSnapshot.capture(preferences)
        val settingsAfterSha = settingsAfter.sha256()
        val restoredExactly = settingsRestored && settingsAfter == settingsBefore
        if (restoredExactly) settingsRecoveryFile.delete()
        if (!restoredExactly && fatalError == null) {
            fatalError = IllegalStateException("V2 settings were not restored exactly")
        }
        val completedAt = Instant.now().toString()
        val repeatDeterminism = repeatDeterminism(runs)
        val prefixDeterminism = crossCountPrefixDeterminism(runs)
        if ((repeatDeterminism.values.any { it.evaluated && it.deterministic != true } ||
                prefixDeterminism.values.any { it.evaluated && it.deterministic != true }) &&
            fatalError == null
        ) {
            fatalError = IllegalStateException("Find Music result determinism check failed")
        }
        val finalState = if (fatalError == null) "COMPLETE" else "FAILED"
        val report = ProductionFindMusicAcceptanceReport(
            state = finalState,
            runId = request.runId,
            startedAt = startedAt,
            completedAt = completedAt,
            request = request,
            startupReadyMs = startupReadyMs,
            databaseInfo = databaseInfo,
            settingsSnapshotSha256Before = settingsBeforeSha,
            settingsSnapshotSha256AfterRestore = settingsAfterSha,
            recoveredSettingsFromPriorInterruptedRun = recoveredSettings != null,
            settingsRestoredExactly = restoredExactly,
            searchRuns = runs,
            repeatDeterminism = repeatDeterminism,
            crossCountPrefixDeterminism = prefixDeterminism,
            cancellation = cancellationEvidence,
            fatalError = fatalError?.stackTraceToString(),
        )
        writeAtomic(AtomicFile(finalFile), gson.toJson(report))
        checkpoint(finalState, null, fatalError)
        return if (fatalError == null) {
            "Complete. ${runs.size} exact production searches recorded."
        } else {
            "FAILED: ${fatalError?.message ?: fatalError?.javaClass?.simpleName}"
        }
    }

    private suspend fun executeSearch(
        request: ProductionFindMusicAcceptanceRequest,
        settingsBaseline: PreferenceSnapshot,
        query: String,
        resultCount: Int,
        planner: FindMusicTextResultPlanner,
        repeat: Int,
    ): ProductionFindMusicAcceptanceSearchRun {
        viewModel.clearFindMusicResults()
        viewModel.setNumTracks(resultCount)
        viewModel.setFindMusicTextResultPlanner(planner)
        prepareSingleTextEditor(query)
        require(settingsBaseline.restore(preferences)) {
            "Could not restore persistent settings before a Find Music case"
        }
        // Repeat 1 captures aggressive memory telemetry. Later repeats are deliberately free of
        // that 200-500Hz sampler and are the runs suitable for latency comparisons.
        val sampler = if (repeat == 1) {
            AcceptanceMemorySampler(request.memorySampleIntervalMs)
        } else {
            null
        }
        val deviceConditionAtStart = captureDeviceCondition()
        return try {
            sampler?.start()
            val started = SystemClock.elapsedRealtimeNanos()
            Log.i(
                TAG,
                "case_start run=${request.runId} query_sha256=${textSha256(query)} " +
                    "count=$resultCount planner=${planner.wireName} repeat=$repeat",
            )
            viewModel.performFindMusicSearch()
            val result = withTimeout(request.timeoutPerCaseSeconds * 1_000L) {
                combine(viewModel.textSearchLoading, viewModel.textSearchResult) { loading, value ->
                    if (!loading) value else null
                }.filterNotNull().first()
            }
            val elapsedToResult = elapsedMs(started)
            val queueEligibility = awaitTerminalQueueEligibility(request.timeoutPerCaseSeconds)
            val elapsedToQueueReady = elapsedMs(started)
            val deviceConditionAtEnd = captureDeviceCondition()
            val tracks = result.toAcceptanceTracks()
            val run = ProductionFindMusicAcceptanceSearchRun(
                query = query,
                publishedQuery = result.query,
                publishedQuerySpec = result.querySpec,
                resultCount = resultCount,
                planner = planner,
                plannerVersion = planner.currentVersion,
                repeat = repeat,
                elapsedToResultMs = elapsedToResult,
                elapsedToQueueReadyMs = elapsedToQueueReady,
                resultFingerprint = resultFingerprint(result),
                resultKind = result.kind,
                error = result.error,
                libraryGenerationId = result.libraryBinding?.generationId,
                activationBindingId = result.libraryBinding?.activationBindingId,
                providerGenerationId = result.providerGenerationId,
                orderedActiveTrackIdsSha256 = result.orderedActiveTrackIdsSha256,
                activeTrackCount = result.activeTrackCount,
                objectiveRankingDomainCount = result.objectiveRankingDomainCount,
                stableResultReduction = result.stableResultReduction,
                textQueuePlan = result.textQueuePlan,
                queueEligibility = queueEligibility,
                deviceConditionAtStart = deviceConditionAtStart,
                deviceConditionAtEnd = deviceConditionAtEnd,
                timingInstrumentedByMemorySampler = sampler != null,
                memory = sampler?.finish(),
                tracks = tracks,
            )
            Log.i(
                TAG,
                "case_end run=${request.runId} query_sha256=${textSha256(query)} " +
                    "count=$resultCount planner=${planner.wireName} repeat=$repeat " +
                    "result_ms=${run.elapsedToResultMs} ready_ms=${run.elapsedToQueueReadyMs} " +
                    "fingerprint=${run.resultFingerprint} " +
                    "memory_sampled=${run.timingInstrumentedByMemorySampler} " +
                    "peak_java_bytes=${run.memory?.peakJavaUsedBytes} " +
                    "peak_native_bytes=${run.memory?.peakNativeAllocatedBytes}",
            )
            run
        } catch (failure: Throwable) {
            sampler?.finish()
            throw failure
        }
    }

    private suspend fun executeCancellation(
        request: ProductionFindMusicAcceptanceRequest,
        settingsBaseline: PreferenceSnapshot,
    ): ProductionFindMusicCancellationEvidence {
        val firstQuery = request.queries.firstOrNull { it != "sleep" } ?: "ambient"
        val secondQuery = request.queries.firstOrNull { it != firstQuery } ?: "sleep"
        val resultCount = request.resultCounts.firstOrNull { it == 30 }
            ?: request.resultCounts.first()
        viewModel.clearFindMusicResults()
        viewModel.setNumTracks(resultCount)
        viewModel.setFindMusicTextResultPlanner(FindMusicTextResultPlanner.VARIED_DPP)
        prepareSingleTextEditor(firstQuery)
        require(settingsBaseline.restore(preferences)) {
            "Could not restore persistent settings before cancellation acceptance"
        }
        val published = mutableListOf<String>()
        val collector: Job = CoroutineScope(currentCoroutineContext()).launch {
            viewModel.textSearchResult.filterNotNull().collect { result ->
                published += result.query
            }
        }
        val sampler = AcceptanceMemorySampler(request.memorySampleIntervalMs)
        val deviceConditionAtStart = captureDeviceCondition()
        return try {
            sampler.start()
            yield()
            // The visible Search action is deliberately disabled while a request is running.
            // Exercise the lower-level text entry point here so this debug-only probe actually
            // submits two immutable requests and tests the latest-request publication guard.
            viewModel.performTextSearch(firstQuery)
            val (resultVisibleAtSupersedeBoundary, supersededAt) =
                withContext(Dispatchers.Default) {
                    // Do not schedule this boundary on the Activity's UI thread: the whole point
                    // is to prove supersession while a compute-heavy request is still running.
                    delay(CANCELLATION_SUPERSEDE_DELAY_MS)
                    val visible = viewModel.textSearchResult.value?.query
                    val at = SystemClock.elapsedRealtimeNanos()
                    viewModel.performTextSearch(secondQuery)
                    visible to at
                }
            prepareSingleTextEditor(secondQuery)
            val finalResult = withTimeout(request.timeoutPerCaseSeconds * 1_000L) {
                combine(viewModel.textSearchLoading, viewModel.textSearchResult) { loading, value ->
                    if (!loading) value else null
                }.filterNotNull().first { result -> result.query == secondQuery }
            }
            val queueEligibility = awaitTerminalQueueEligibility(request.timeoutPerCaseSeconds)
            val deviceConditionAtEnd = captureDeviceCondition()
            yield()
            collector.cancelAndJoin()
            val publishedSnapshot = published.toList()
            ProductionFindMusicCancellationEvidence(
                firstQuery = firstQuery,
                secondQuery = secondQuery,
                supersedeDelayMs = CANCELLATION_SUPERSEDE_DELAY_MS,
                elapsedAfterSupersedeMs = elapsedMs(supersededAt),
                conflatedStateFlowQueriesObserved = publishedSnapshot,
                resultVisibleAtSupersedeBoundary = resultVisibleAtSupersedeBoundary,
                finalQuery = finalResult.query,
                finalPlanner = finalResult.querySpec?.textResultPlanner,
                latestResultSmokePassed = resultVisibleAtSupersedeBoundary == null &&
                    finalResult.query == secondQuery &&
                    finalResult.querySpec?.textResultPlanner ==
                    FindMusicTextResultPlanner.VARIED_DPP &&
                    finalResult.error == null && queueEligibility.eligible,
                finalResultFingerprint = resultFingerprint(finalResult),
                finalQueueEligibility = queueEligibility,
                deviceConditionAtStart = deviceConditionAtStart,
                deviceConditionAtEnd = deviceConditionAtEnd,
                memory = sampler.finish(),
            )
        } catch (failure: Throwable) {
            collector.cancelAndJoin()
            ProductionFindMusicCancellationEvidence(
                firstQuery = firstQuery,
                secondQuery = secondQuery,
                supersedeDelayMs = CANCELLATION_SUPERSEDE_DELAY_MS,
                elapsedAfterSupersedeMs = 0L,
                conflatedStateFlowQueriesObserved = published.toList(),
                resultVisibleAtSupersedeBoundary = viewModel.textSearchResult.value?.query,
                finalQuery = viewModel.textSearchResult.value?.query,
                finalPlanner = viewModel.textSearchResult.value?.querySpec?.textResultPlanner,
                latestResultSmokePassed = false,
                finalResultFingerprint = viewModel.textSearchResult.value?.let(::resultFingerprint),
                finalQueueEligibility = viewModel.displayedQueueEligibility.value,
                deviceConditionAtStart = deviceConditionAtStart,
                deviceConditionAtEnd = captureDeviceCondition(),
                memory = sampler.finish(),
                error = failure.stackTraceToString(),
            )
        }
    }

    private fun validateSearchRun(
        run: ProductionFindMusicAcceptanceSearchRun,
        expectedLibrary: DatabaseInfo,
    ) {
        require(run.publishedQuery == run.query) {
            "Published query ${run.publishedQuery} differs from requested query ${run.query}"
        }
        val publishedSpec = requireNotNull(run.publishedQuerySpec) {
            "Find Music result has no published query specification"
        }
        val publishedText = publishedSpec.activeTextIngredients.singleOrNull()
        require(publishedSpec.isSimplePositiveTextOnly &&
            publishedText?.query == run.query &&
            publishedText.weight == 1f &&
            !publishedText.negative &&
            publishedSpec.songSeeds.none { it.weight > 0f } &&
            publishedSpec.resultLimit == run.resultCount &&
            publishedSpec.textResultPlanner == run.planner
        ) { "Published query specification differs from the requested production UI case" }
        require(run.error == null) { run.error ?: "Find Music returned an error" }
        require(run.resultKind == FindMusicResultKind.TEXT) { "Find Music result was not text" }
        require(run.tracks.size == run.resultCount) {
            "Find Music returned ${run.tracks.size}/${run.resultCount} tracks"
        }
        require(run.libraryGenerationId != null && run.activationBindingId != null &&
            run.providerGenerationId != null &&
            run.orderedActiveTrackIdsSha256?.matches(SHA256) == true &&
            run.activeTrackCount != null && run.activeTrackCount > 0 &&
            run.objectiveRankingDomainCount != null &&
            run.objectiveRankingDomainCount in 1..run.activeTrackCount
        ) { "Find Music result has incomplete active-library binding" }
        require(run.libraryGenerationId == expectedLibrary.generationId &&
            run.providerGenerationId == expectedLibrary.providerGenerationId &&
            run.activeTrackCount == expectedLibrary.activeTrackCount
        ) { "The active library changed during production Find Music acceptance" }
        require(run.tracks.map { it.embeddedTrackId }.toSet().size == run.tracks.size) {
            "Find Music returned duplicate embedded track IDs"
        }
        require(run.tracks.map { it.objectiveRank }.toSet().size == run.tracks.size) {
            "Find Music returned duplicate text-objective ranks"
        }
        require(run.queueEligibility.eligible) {
            run.queueEligibility.reason ?: "Find Music result is not queue-ready"
        }
        val reduction = requireNotNull(run.stableResultReduction) {
            "Find Music result has no stable-copy reduction evidence"
        }
        require(reduction.requestedVisibleCount == run.resultCount) {
            "Find Music reduction requested count changed"
        }
        when (run.planner) {
            FindMusicTextResultPlanner.CLOSEST -> {
                require(run.textQueuePlan == null) {
                    "Closest result unexpectedly carried a Varied plan"
                }
                require(run.tracks.map { it.objectiveRank }.zipWithNext().all { (a, b) -> a < b }) {
                    "Closest text-objective ranks are not ordered"
                }
            }
            FindMusicTextResultPlanner.VARIED_DPP -> {
                val plan = requireNotNull(run.textQueuePlan) {
                    "Varied result has no complete-domain DPP plan"
                }
                plan.requireValid()
                val proof = requireNotNull(plan.dppSelection) {
                    "Varied result has no persisted DPP selection-step proof"
                }
                require(plan.planner == run.planner &&
                    plan.completeCandidateDomainCount == run.objectiveRankingDomainCount &&
                    plan.requestedResultCount == run.resultCount &&
                    plan.orderedSelectedTrackIds == run.tracks.map { it.embeddedTrackId } &&
                    plan.orderedOriginalTextObjectiveRanks == run.tracks.map { it.objectiveRank } &&
                    proof.selectedMarginalGains.size == run.tracks.size &&
                    reduction.collapsedEquivalentCount == 0
                ) { "Displayed Varied result differs from its complete-domain proof" }
            }
            FindMusicTextResultPlanner.VARIED_ALL_OF_DPP ->
                error("Text acceptance cannot run the All-of Varied planner")
        }
    }

    private suspend fun awaitTerminalQueueEligibility(
        timeoutSeconds: Int,
    ): DisplayedFindMusicQueueEligibility = withTimeout(timeoutSeconds * 1_000L) {
        viewModel.displayedQueueEligibility.first { eligibility ->
            eligibility != DisplayedFindMusicQueueEligibility.CHECKING &&
                eligibility != DisplayedFindMusicQueueEligibility.UNAVAILABLE
        }
    }

    private fun prepareSingleTextEditor(query: String) {
        require(viewModel.songSeeds.value.isEmpty()) {
            "Debug acceptance ViewModel unexpectedly contains song ingredients"
        }
        require(viewModel.textIngredients.value.size == 1) {
            "Debug acceptance ViewModel unexpectedly contains multiple text ingredients"
        }
        viewModel.updateTextIngredientQuery(0, query)
        val ingredient = viewModel.textIngredients.value.single()
        require(ingredient.query == query && ingredient.weight == 1f && !ingredient.negative) {
            "Could not prepare the exact one-positive-text UI editor state"
        }
    }

    private fun captureDeviceCondition(): ProductionFindMusicDeviceCondition {
        val thermalStatus = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            appContext.getSystemService(PowerManager::class.java)?.currentThermalStatus
        } else {
            null
        }
        val batteryIntent: Intent? = appContext.registerReceiver(
            null,
            IntentFilter(Intent.ACTION_BATTERY_CHANGED),
        )
        val temperatureTenthsCelsius = batteryIntent
            ?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, Int.MIN_VALUE)
            ?.takeUnless { it == Int.MIN_VALUE }
        return ProductionFindMusicDeviceCondition(
            capturedAtElapsedRealtimeMs = SystemClock.elapsedRealtime(),
            thermalStatus = thermalStatus,
            thermalStatusName = when (thermalStatus) {
                PowerManager.THERMAL_STATUS_NONE -> "NONE"
                PowerManager.THERMAL_STATUS_LIGHT -> "LIGHT"
                PowerManager.THERMAL_STATUS_MODERATE -> "MODERATE"
                PowerManager.THERMAL_STATUS_SEVERE -> "SEVERE"
                PowerManager.THERMAL_STATUS_CRITICAL -> "CRITICAL"
                PowerManager.THERMAL_STATUS_EMERGENCY -> "EMERGENCY"
                PowerManager.THERMAL_STATUS_SHUTDOWN -> "SHUTDOWN"
                null -> "UNAVAILABLE"
                else -> "UNKNOWN_$thermalStatus"
            },
            batteryTemperatureTenthsCelsius = temperatureTenthsCelsius,
            batteryTemperatureCelsius = temperatureTenthsCelsius?.div(10f),
        )
    }

    private fun repeatDeterminism(
        runs: List<ProductionFindMusicAcceptanceSearchRun>,
    ): Map<String, ProductionFindMusicDeterminismEvaluation> =
        TreeMap<String, ProductionFindMusicDeterminismEvaluation>().apply {
        runs.groupBy { "${it.query}|${it.resultCount}|${it.planner.wireName}" }
            .forEach { (key, group) ->
                val evaluated = group.size >= 2
                val deterministic = group.map { run ->
                    listOf(
                        run.resultFingerprint,
                        run.textQueuePlan?.completeTextRankingSha256,
                        run.textQueuePlan?.dppSelection,
                    )
                }.distinct().size == 1
                this[key] = ProductionFindMusicDeterminismEvaluation(
                    observationCount = group.size,
                    evaluated = evaluated,
                    deterministic = deterministic.takeIf { evaluated },
                )
            }
    }

    private fun crossCountPrefixDeterminism(
        runs: List<ProductionFindMusicAcceptanceSearchRun>,
    ): Map<String, ProductionFindMusicDeterminismEvaluation> =
        TreeMap<String, ProductionFindMusicDeterminismEvaluation>().apply {
        runs.groupBy { "${it.query}|${it.planner.wireName}|${it.repeat}" }
            .forEach { (key, group) ->
                val ordered = group.sortedBy { it.resultCount }
                val evaluated = ordered.size >= 2
                val deterministic = ordered.zipWithNext().all { (shorter, longer) ->
                    val shorterIds = shorter.tracks.map { it.embeddedTrackId }
                    val longerIds = longer.tracks.map { it.embeddedTrackId }
                    shorterIds == longerIds.take(shorterIds.size) &&
                        shorter.tracks.map { it.objectiveRank } ==
                        longer.tracks.take(shorterIds.size).map { it.objectiveRank }
                }
                this[key] = ProductionFindMusicDeterminismEvaluation(
                    observationCount = ordered.size,
                    evaluated = evaluated,
                    deterministic = deterministic.takeIf { evaluated },
                )
            }
    }

    private fun TextSearchResult.toAcceptanceTracks(): List<ProductionFindMusicAcceptanceTrack> =
        matches.mapIndexed { index, match ->
            ProductionFindMusicAcceptanceTrack(
                displayedPosition = index + 1,
                embeddedTrackId = match.track.id,
                stableTrackSpanId = match.identity.stableTrackSpanId,
                objectiveRank = requireNotNull(match.objectiveRank) {
                    "Displayed Find Music track has no objective rank"
                },
                artist = match.track.artist,
                album = match.track.album,
                title = match.track.title,
                durationMs = match.track.durationMs,
                filePath = match.track.filePath,
                textSimilarity = match.similarity,
                rankingScore = match.rankingScore,
            )
        }

    private fun resultFingerprint(result: TextSearchResult): String? {
        if (result.error != null || result.matches.isEmpty()) return null
        val digest = MessageDigest.getInstance("SHA-256")
        digest.update("production-find-music-result-v1\u0000".toByteArray(Charsets.US_ASCII))
        val numeric = ByteBuffer.allocate(Long.SIZE_BYTES + Int.SIZE_BYTES * 3)
            .order(ByteOrder.BIG_ENDIAN)
        result.matches.forEachIndexed { index, match ->
            numeric.clear()
            numeric.putLong(match.track.id)
            numeric.putInt(requireNotNull(match.objectiveRank))
            numeric.putInt(match.similarity.toRawBits())
            numeric.putInt(index + 1)
            digest.update(numeric.array())
            val stableId = match.identity.stableTrackSpanId.orEmpty()
            digest.update(stableId.toByteArray(Charsets.US_ASCII))
            digest.update(0)
        }
        return digest.digest().toHex()
    }

    private fun textSha256(value: String): String = MessageDigest.getInstance("SHA-256")
        .digest(value.toByteArray(Charsets.UTF_8))
        .toHex()

    private fun appendJsonLine(file: File, value: Any) {
        FileOutputStream(file, true).use { output ->
            OutputStreamWriter(output, Charsets.UTF_8).use { writer ->
                writer.append(compactGson.toJson(value)).append('\n')
                writer.flush()
                output.fd.sync()
            }
        }
    }

    private fun writeAtomic(file: AtomicFile, value: String) {
        val bytes = value.toByteArray(Charsets.UTF_8)
        var output: FileOutputStream? = null
        try {
            output = file.startWrite()
            output.write(bytes)
            output.fd.sync()
            file.finishWrite(output)
        } catch (failure: Throwable) {
            output?.let(file::failWrite)
            throw failure
        }
    }

    private fun readAtomicIfExists(file: AtomicFile): String? = try {
        file.openRead().bufferedReader(Charsets.UTF_8).use { it.readText() }
    } catch (_: FileNotFoundException) {
        null
    }

    private fun elapsedMs(startedNanos: Long): Long =
        (SystemClock.elapsedRealtimeNanos() - startedNanos) / 1_000_000L

    private fun ByteArray.toHex(): String = joinToString("") { "%02x".format(it) }

}

private class AcceptanceMemorySampler(
    private val intervalMs: Int,
) {
    private val running = AtomicBoolean(false)
    private val peakJava = AtomicLong(0L)
    private val peakNative = AtomicLong(0L)
    private val sampleCount = AtomicLong(0L)
    private var thread: Thread? = null
    private var start: MemoryPoint? = null
    private var pssStartKb: Long = 0L
    private var finished: ProductionFindMusicAcceptanceMemory? = null

    fun start() {
        check(running.compareAndSet(false, true)) { "Memory sampler already started" }
        start = sample().also(::record)
        pssStartKb = Debug.getPss()
        thread = Thread({
            while (running.get()) {
                record(sample())
                LockSupport.parkNanos(intervalMs * 1_000_000L)
            }
            record(sample())
        }, "find-music-acceptance-memory").apply {
            isDaemon = true
            start()
        }
    }

    fun finish(): ProductionFindMusicAcceptanceMemory {
        finished?.let { return it }
        running.set(false)
        thread?.join(2_000L)
        val end = sample().also(::record)
        val startPoint = start ?: end
        return ProductionFindMusicAcceptanceMemory(
            sampleIntervalMs = intervalMs,
            sampleCount = sampleCount.get(),
            javaUsedBytesAtStart = startPoint.javaUsed,
            javaUsedBytesAtEnd = end.javaUsed,
            peakJavaUsedBytes = peakJava.get(),
            javaCommittedBytesAtEnd = end.javaCommitted,
            javaMaximumBytes = end.javaMaximum,
            nativeAllocatedBytesAtStart = startPoint.nativeAllocated,
            nativeAllocatedBytesAtEnd = end.nativeAllocated,
            peakNativeAllocatedBytes = peakNative.get(),
            pssKbAtStart = pssStartKb,
            pssKbAtEnd = Debug.getPss(),
            processVmPeakKbAtEnd = processStatusKb("VmPeak"),
            processVmHwmKbAtEnd = processStatusKb("VmHWM"),
            processVmRssKbAtEnd = processStatusKb("VmRSS"),
        ).also { finished = it }
    }

    private fun sample(): MemoryPoint {
        val runtime = Runtime.getRuntime()
        return MemoryPoint(
            javaUsed = runtime.totalMemory() - runtime.freeMemory(),
            javaCommitted = runtime.totalMemory(),
            javaMaximum = runtime.maxMemory(),
            nativeAllocated = Debug.getNativeHeapAllocatedSize(),
        )
    }

    private fun record(point: MemoryPoint) {
        peakJava.accumulateAndGet(point.javaUsed) { current, candidate ->
            maxOf(current, candidate)
        }
        peakNative.accumulateAndGet(point.nativeAllocated) { current, candidate ->
            maxOf(current, candidate)
        }
        sampleCount.incrementAndGet()
    }

    private fun processStatusKb(label: String): Long? = runCatching {
        File("/proc/self/status").useLines { lines ->
            lines.firstOrNull { it.startsWith("$label:") }
                ?.substringAfter(':')
                ?.trim()
                ?.substringBefore(' ')
                ?.toLongOrNull()
        }
    }.getOrNull()

    private data class MemoryPoint(
        val javaUsed: Long,
        val javaCommitted: Long,
        val javaMaximum: Long,
        val nativeAllocated: Long,
    )
}

private data class PreferenceSnapshot(
    val values: Map<String, PreferenceValue>,
) {
    fun sha256(): String {
        val digest = MessageDigest.getInstance("SHA-256")
        digest.update("production-find-music-settings-v1\u0000".toByteArray(Charsets.US_ASCII))
        values.toSortedMap().forEach { (key, value) ->
            digest.update(key.toByteArray(Charsets.UTF_8))
            digest.update(0)
            digest.update(value.type.toByteArray(Charsets.US_ASCII))
            digest.update(0)
            value.values.forEach { item ->
                digest.update(item.toByteArray(Charsets.UTF_8))
                digest.update(0)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    fun restore(preferences: SharedPreferences): Boolean {
        val editor = preferences.edit().clear()
        values.forEach { (key, value) -> value.put(editor, key) }
        return editor.commit()
    }

    companion object {
        fun capture(preferences: SharedPreferences): PreferenceSnapshot = PreferenceSnapshot(
            values = preferences.all.mapValues { (_, raw) -> PreferenceValue.from(raw) },
        )
    }
}

private data class PreferenceValue(
    val type: String,
    val values: List<String>,
) {
    fun put(editor: SharedPreferences.Editor, key: String) {
        when (type) {
            "boolean" -> editor.putBoolean(key, values.single().toBooleanStrict())
            "float" -> editor.putFloat(key, Float.fromBits(values.single().toInt()))
            "int" -> editor.putInt(key, values.single().toInt())
            "long" -> editor.putLong(key, values.single().toLong())
            "string" -> editor.putString(key, values.single())
            "string_set" -> editor.putStringSet(key, values.toSet())
            else -> error("Unsupported preference type $type")
        }
    }

    companion object {
        fun from(raw: Any?): PreferenceValue = when (raw) {
            is Boolean -> PreferenceValue("boolean", listOf(raw.toString()))
            is Float -> PreferenceValue("float", listOf(raw.toRawBits().toString()))
            is Int -> PreferenceValue("int", listOf(raw.toString()))
            is Long -> PreferenceValue("long", listOf(raw.toString()))
            is String -> PreferenceValue("string", listOf(raw))
            is Set<*> -> PreferenceValue(
                "string_set",
                raw.map { requireNotNull(it as? String) }.sorted(),
            )
            else -> error("Unsupported preference value ${raw?.javaClass?.name}")
        }
    }
}
