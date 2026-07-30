package com.powerampstartradio.debug

import android.content.Context
import android.content.SharedPreferences
import android.os.Bundle
import android.os.PowerManager
import android.os.SystemClock
import android.util.AtomicFile
import android.util.Log
import android.view.Gravity
import android.view.WindowManager
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.lifecycleScope
import com.google.gson.GsonBuilder
import com.powerampstartradio.data.EmbeddedTrack
import com.powerampstartradio.ui.DatabaseInfo
import com.powerampstartradio.ui.DisplayedFindMusicQueueEligibility
import com.powerampstartradio.ui.FindMusicOperator
import com.powerampstartradio.ui.FindMusicQuerySpec
import com.powerampstartradio.ui.FindMusicRefineNeighborhood
import com.powerampstartradio.ui.FindMusicRefineSpec
import com.powerampstartradio.ui.FindMusicResultKind
import com.powerampstartradio.ui.FindMusicTextResultPlanner
import com.powerampstartradio.ui.LibraryRankEvidenceText
import com.powerampstartradio.ui.MainViewModel
import com.powerampstartradio.ui.RadioSettingsStore
import com.powerampstartradio.ui.RecordingLookupState
import com.powerampstartradio.ui.TextSearchResult
import com.powerampstartradio.ui.effectiveLibraryAddedDays
import com.powerampstartradio.similarity.FindMusicAllOfQueuePlanEvidence
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.filterNotNull
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout
import java.io.File
import java.io.FileNotFoundException
import java.io.FileOutputStream
import java.io.OutputStreamWriter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import java.time.Instant
import java.util.UUID
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln

/** Read-only acceptance for the exact production composed Find Music editor and ranking path. */
class ProductionComposedFindMusicAcceptanceActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        showOverLockScreenForAcceptance()
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        window.attributes = window.attributes.apply { screenBrightness = 0.01f }
        val status = TextView(this).apply {
            gravity = Gravity.CENTER
            textSize = 16f
            setPadding(48, 48, 48, 48)
            text = "Preparing composed Find Music acceptance..."
        }
        setContentView(status)

        if (savedInstanceState == null) {
            val viewModel = ViewModelProvider(this)[MainViewModel::class.java]
            lifecycleScope.launch {
                val request = ComposedAcceptanceRequest.from(intent.extras)
                val wakeLock = getSystemService(PowerManager::class.java).newWakeLock(
                    PowerManager.PARTIAL_WAKE_LOCK,
                    "$packageName:composed-find-music-acceptance",
                )
                wakeLock.acquire()
                val finalStatus = try {
                    ComposedAcceptanceRunner(
                        context = applicationContext,
                        filesDir = filesDir,
                        viewModel = viewModel,
                    ).run(request) { message -> status.text = message }
                } finally {
                    if (wakeLock.isHeld) wakeLock.release()
                }
                status.text = finalStatus
                finishAndRemoveTask()
            }
        }
    }
}

private data class ComposedAcceptanceRequest(
    val runId: String,
    val repeatCount: Int,
    val timeoutPerCaseSeconds: Int,
    val recording: ComposedRecordingInput,
) {
    companion object {
        private val RUN_ID = Regex("^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

        fun from(extras: Bundle?): ComposedAcceptanceRequest {
            val requestExtras = requireNotNull(extras) {
                "recording acceptance arguments are required"
            }
            val requestedRunId = requestExtras.getString("run_id")?.trim()
            val recordingQuery = requireNotNull(
                requestExtras.getString("recording_query")?.trim()?.takeIf(String::isNotEmpty),
            ) { "recording_query is required" }
            val recordingTrackId = requestExtras.getLong("recording_track_id", -1L)
            require(recordingTrackId > 0L) { "recording_track_id must be positive" }
            val recordingArtist = requireNotNull(
                requestExtras.getString("recording_artist")?.trim()?.takeIf(String::isNotEmpty),
            ) { "recording_artist is required" }
            val recordingTitle = requireNotNull(
                requestExtras.getString("recording_title")?.trim()?.takeIf(String::isNotEmpty),
            ) { "recording_title is required" }
            return ComposedAcceptanceRequest(
                runId = requestedRunId?.takeIf { it.matches(RUN_ID) }
                    ?: UUID.randomUUID().toString(),
                repeatCount = requestExtras.getInt("repeat_count", 2).coerceIn(1, 4),
                timeoutPerCaseSeconds =
                    requestExtras.getInt("timeout_per_case_seconds", 180)
                        .coerceIn(30, 600),
                recording = ComposedRecordingInput(
                    lookupQuery = recordingQuery,
                    expectedTrackId = recordingTrackId,
                    expectedArtist = recordingArtist,
                    expectedTitle = recordingTitle,
                    weight = 0.7f,
                ),
            )
        }
    }
}

private data class ComposedTextInput(
    val query: String,
    val weight: Float,
    val negative: Boolean = false,
)

private data class ComposedRecordingInput(
    val lookupQuery: String,
    val expectedTrackId: Long,
    val expectedArtist: String,
    val expectedTitle: String,
    val weight: Float,
    val negative: Boolean = false,
)

private data class ComposedAcceptanceCase(
    val id: String,
    val operator: FindMusicOperator,
    val texts: List<ComposedTextInput>,
    val recording: ComposedRecordingInput? = null,
    val resultCount: Int = 30,
    val libraryAddedDays: Int? = null,
    val refineSpec: FindMusicRefineSpec? = null,
    val planner: FindMusicTextResultPlanner = FindMusicTextResultPlanner.CLOSEST,
) {
    val activeIngredientCount: Int
        get() = texts.size + if (recording == null) 0 else 1

    init {
        require((operator == FindMusicOperator.REFINE) == (refineSpec != null)) {
            "Only Refine cases may carry a Refine specification"
        }
        require(operator != FindMusicOperator.REFINE || activeIngredientCount == 2) {
            "Refine acceptance needs exactly two active ingredients"
        }
        require(
            planner == FindMusicTextResultPlanner.CLOSEST ||
                operator == FindMusicOperator.ALL_OF,
        ) { "Only All-of acceptance may use a varied composed planner" }
    }
}

private data class ComposedIngredientEvidence(
    val label: String,
    /** Debug-only exact-rank source. Never rendered as a user-facing quality score. */
    val percentile: Float,
    val exactRank: Int,
    val rankedTrackCount: Int,
    val topFraction: String,
)

private data class ComposedAcceptanceTrack(
    val displayedPosition: Int,
    val embeddedTrackId: Long,
    val stableTrackSpanId: String?,
    val artist: String?,
    val album: String?,
    val title: String?,
    val durationMs: Int,
    val filePath: String,
    val objectiveRank: Int,
    /** Debug-only formula evidence. The user surface renders ranks and top fractions instead. */
    val rankingScore: Float,
    val overallEvidence: String?,
    val ingredientEvidence: List<ComposedIngredientEvidence>,
)

private data class ComposedAcceptanceRun(
    val caseId: String,
    val repeat: Int,
    val requestedOperator: FindMusicOperator,
    val requestedRefineSpec: FindMusicRefineSpec?,
    val requestedTexts: List<ComposedTextInput>,
    val requestedRecording: ComposedRecordingInput?,
    val publishedQuerySpec: FindMusicQuerySpec,
    val elapsedToResultMs: Long,
    val elapsedToQueueReadyMs: Long,
    val resultFingerprint: String,
    val libraryGenerationId: String,
    val activationBindingId: String,
    val providerGenerationId: String,
    val orderedActiveTrackIdsSha256: String,
    val activeTrackCount: Int,
    val objectiveRankingDomainCount: Int,
    val ingredientRankingDomainCount: Int,
    val stableReductionIdentityPolicyVersion: Int,
    val stableReductionRequestedVisibleCount: Int,
    val stableReductionScannedRowCount: Int,
    val collapsedEquivalentCount: Int,
    val refineAllowedCandidateCounts: List<Int>?,
    val confirmedRecording: ComposedConfirmedRecording?,
    val allOfQueuePlan: FindMusicAllOfQueuePlanEvidence?,
    val tracks: List<ComposedAcceptanceTrack>,
)

private data class ComposedPairComparison(
    val changedControl: String,
    val leftCaseId: String,
    val rightCaseId: String,
    val sharedTrackCount: Int,
    val changedTrackCountPerSide: Int,
    val sharedTracksThatMoved: Int,
    val sameOrderedResult: Boolean,
    val enteredRight: List<String>,
    val exitedRight: List<String>,
)

private data class ComposedConfirmedRecording(
    val embeddedTrackId: Long,
    val stableTrackSpanId: String?,
    val artist: String?,
    val album: String?,
    val title: String?,
    val durationMs: Int,
    val filePath: String,
)

private data class ComposedDeterminismEvidence(
    val observationCount: Int,
    val deterministic: Boolean?,
)

private data class ComposedAcceptanceStatus(
    val state: String,
    val runId: String,
    val startedAt: String,
    val updatedAt: String,
    val completedRunCount: Int,
    val plannedRunCount: Int,
    val currentCase: String?,
    val fatalError: String? = null,
)

private data class ComposedAcceptanceReport(
    val schemaVersion: Int = 3,
    val state: String,
    val runId: String,
    val startedAt: String,
    val completedAt: String,
    val request: ComposedAcceptanceRequest,
    val databaseInfo: DatabaseInfo?,
    val queueMutationApisCalled: Int = 0,
    val settingsSnapshotSha256Before: String,
    val settingsSnapshotSha256AfterRestore: String?,
    val recoveredSettingsFromPriorInterruptedRun: Boolean,
    val settingsRestoredExactly: Boolean,
    val runs: List<ComposedAcceptanceRun>,
    val repeatDeterminism: Map<String, ComposedDeterminismEvidence>,
    val pairComparisons: List<ComposedPairComparison>,
    val fatalError: String? = null,
)

private class ComposedAcceptanceRunner(
    context: Context,
    private val filesDir: File,
    private val viewModel: MainViewModel,
) {
    companion object {
        private const val TAG = "ComposedFindMusicAcceptance"
        private val SHA256 = Regex("^[0-9a-f]{64}$")
        private val gson = GsonBuilder().setPrettyPrinting().disableHtmlEscaping().create()
        private val compactGson = GsonBuilder().disableHtmlEscaping().create()

        private fun cases(recording: ComposedRecordingInput) = listOf(
            ComposedAcceptanceCase(
                id = "all_recording70_sleep30",
                operator = FindMusicOperator.ALL_OF,
                texts = listOf(ComposedTextInput("sleep", 0.3f)),
                recording = recording,
            ),
            ComposedAcceptanceCase(
                id = "all_recording70_sleep30_varied",
                operator = FindMusicOperator.ALL_OF,
                texts = listOf(ComposedTextInput("sleep", 0.3f)),
                recording = recording,
                planner = FindMusicTextResultPlanner.VARIED_ALL_OF_DPP,
            ),
            ComposedAcceptanceCase(
                id = "all_recording30_sleep70",
                operator = FindMusicOperator.ALL_OF,
                texts = listOf(ComposedTextInput("sleep", 0.7f)),
                recording = recording.copy(weight = 0.3f),
            ),
            ComposedAcceptanceCase(
                id = "all_recording70_avoid_sleep30",
                operator = FindMusicOperator.ALL_OF,
                texts = listOf(ComposedTextInput("sleep", 0.3f, negative = true)),
                recording = recording,
            ),
            ComposedAcceptanceCase(
                id = "all_recording70_avoid_sleep30_varied",
                operator = FindMusicOperator.ALL_OF,
                texts = listOf(ComposedTextInput("sleep", 0.3f, negative = true)),
                recording = recording,
                planner = FindMusicTextResultPlanner.VARIED_ALL_OF_DPP,
            ),
            ComposedAcceptanceCase(
                id = "all_recent_14d_four_texts_varied",
                operator = FindMusicOperator.ALL_OF,
                texts = listOf(
                    ComposedTextInput("ambient", 0.25f),
                    ComposedTextInput("sleep", 0.25f),
                    ComposedTextInput("guitar", 0.25f),
                    ComposedTextInput("psychedelic", 0.25f),
                ),
                libraryAddedDays = 14,
                planner = FindMusicTextResultPlanner.VARIED_ALL_OF_DPP,
            ),
            ComposedAcceptanceCase(
                id = "refine_recording_by_sleep_default",
                operator = FindMusicOperator.REFINE,
                texts = listOf(ComposedTextInput("sleep", 0.5f)),
                recording = recording.copy(weight = 0.5f),
                refineSpec = FindMusicRefineSpec(
                    primaryIngredientIndex = 1,
                    neighborhood = FindMusicRefineNeighborhood.DEFAULT,
                ),
            ),
        )
    }

    private val appContext = context.applicationContext
    private val preferences = context.getSharedPreferences(
        RadioSettingsStore.PREFERENCES_NAME,
        Context.MODE_PRIVATE,
    )
    private val outputDir = File(filesDir, "production_composed_find_music_acceptance")

    suspend fun run(
        request: ComposedAcceptanceRequest,
        onStatus: (String) -> Unit,
    ): String {
        require(outputDir.isDirectory || outputDir.mkdirs()) {
            "Cannot create ${outputDir.absolutePath}"
        }
        val startedAt = Instant.now().toString()
        val statusFile = AtomicFile(File(outputDir, "status.json"))
        val jsonlFile = File(outputDir, "${request.runId}.runs.jsonl")
        val finalFile = AtomicFile(File(outputDir, "${request.runId}.json"))
        val settingsRecoveryFile = AtomicFile(File(outputDir, "settings-recovery.json"))
        require(!jsonlFile.exists() && !File(outputDir, "${request.runId}.json").exists()) {
            "Acceptance run ${request.runId} already exists"
        }
        val recoveredSettings = readAtomicIfExists(settingsRecoveryFile)?.let { encoded ->
            gson.fromJson(encoded, ComposedPreferenceSnapshot::class.java).also { snapshot ->
                require(snapshot.restore(preferences) &&
                    ComposedPreferenceSnapshot.capture(preferences) == snapshot
                ) { "Could not restore settings left by an interrupted composed acceptance run" }
            }
        }
        if (recoveredSettings != null) settingsRecoveryFile.delete()
        val settingsBefore = ComposedPreferenceSnapshot.capture(preferences)
        val settingsBeforeSha = settingsBefore.sha256()
        writeAtomic(settingsRecoveryFile, gson.toJson(settingsBefore))
        val runs = mutableListOf<ComposedAcceptanceRun>()
        val acceptanceCases = cases(request.recording)
        val planned = acceptanceCases.size * request.repeatCount
        var databaseInfo: DatabaseInfo? = null
        var fatalError: Throwable? = null

        fun checkpoint(state: String, currentCase: String?, failure: Throwable? = null) {
            writeAtomic(
                statusFile,
                gson.toJson(
                    ComposedAcceptanceStatus(
                        state = state,
                        runId = request.runId,
                        startedAt = startedAt,
                        updatedAt = Instant.now().toString(),
                        completedRunCount = runs.size,
                        plannedRunCount = planned,
                        currentCase = currentCase,
                        fatalError = failure?.stackTraceToString(),
                    ),
                ),
            )
        }

        checkpoint("RUNNING", "waiting_for_verified_library")
        try {
            onStatus("Waiting for the verified active library...")
            databaseInfo = withTimeout(request.timeoutPerCaseSeconds * 1_000L) {
                combine(viewModel.databaseLoading, viewModel.databaseInfo) { loading, info ->
                    if (!loading) info else null
                }.filterNotNull().first()
            }
            require(databaseInfo.generationId != null && databaseInfo.providerGenerationId != null &&
                databaseInfo.activeTrackCount != null
            ) { "Production library readiness did not publish an exact active generation" }

            for (case in acceptanceCases) {
                for (repeat in 1..request.repeatCount) {
                    val label = "${case.id} | repeat $repeat"
                    onStatus("$label\n${runs.size}/$planned complete")
                    checkpoint("RUNNING", label)
                    require(settingsBefore.restore(preferences)) {
                        "Could not restore settings before $label"
                    }
                    val prepared = prepareEditor(case, request.timeoutPerCaseSeconds)
                    val run = executeCase(
                        case = case,
                        repeat = repeat,
                        timeoutSeconds = request.timeoutPerCaseSeconds,
                        confirmedRecording = prepared,
                    )
                    validateRun(run, case, databaseInfo)
                    runs += run
                    appendJsonLine(jsonlFile, run)
                    viewModel.clearFindMusicResults()
                    delay(100L)
                    require(settingsBefore.restore(preferences)) {
                        "Could not restore settings after $label"
                    }
                    checkpoint("RUNNING", null)
                }
            }
        } catch (failure: Throwable) {
            fatalError = failure
            Log.e(TAG, "Acceptance run ${request.runId} failed", failure)
        } finally {
            viewModel.clearFindMusicResults()
            clearEditor()
        }

        withContext(NonCancellable) {
            delay(500L)
            settingsBefore.restore(preferences)
            delay(250L)
        }
        val settingsAfter = ComposedPreferenceSnapshot.capture(preferences)
        val settingsAfterSha = settingsAfter.sha256()
        val restoredExactly = settingsAfter == settingsBefore
        if (!restoredExactly && fatalError == null) {
            fatalError = IllegalStateException("V2 settings were not restored exactly")
        }
        val determinism = runs.groupBy(ComposedAcceptanceRun::caseId).toSortedMap()
            .mapValues { (_, values) ->
                val evaluated = values.size >= 2
                ComposedDeterminismEvidence(
                    observationCount = values.size,
                    deterministic = values.map(ComposedAcceptanceRun::resultFingerprint)
                        .distinct().size.let { it == 1 }.takeIf { evaluated },
                )
            }
        if (determinism.values.any { it.deterministic == false } && fatalError == null) {
            fatalError = IllegalStateException("Composed Find Music determinism check failed")
        }
        val pairComparisons = buildPairComparisons(runs)
        if (pairComparisons.size == 4 &&
            pairComparisons.any(ComposedPairComparison::sameOrderedResult) &&
            fatalError == null
        ) {
            fatalError = IllegalStateException(
                "A composed Find Music weight, sign, or operator control was inert",
            )
        }
        if (restoredExactly) settingsRecoveryFile.delete()
        val state = if (fatalError == null) "COMPLETE" else "FAILED"
        val report = ComposedAcceptanceReport(
            state = state,
            runId = request.runId,
            startedAt = startedAt,
            completedAt = Instant.now().toString(),
            request = request,
            databaseInfo = databaseInfo,
            settingsSnapshotSha256Before = settingsBeforeSha,
            settingsSnapshotSha256AfterRestore = settingsAfterSha,
            recoveredSettingsFromPriorInterruptedRun = recoveredSettings != null,
            settingsRestoredExactly = restoredExactly,
            runs = runs,
            repeatDeterminism = determinism,
            pairComparisons = pairComparisons,
            fatalError = fatalError?.stackTraceToString(),
        )
        writeAtomic(finalFile, gson.toJson(report))
        checkpoint(state, null, fatalError)
        return if (fatalError == null) {
            "Complete. ${runs.size} composed production searches recorded."
        } else {
            "FAILED: ${fatalError.message ?: fatalError.javaClass.simpleName}"
        }
    }

    private suspend fun prepareEditor(
        case: ComposedAcceptanceCase,
        timeoutSeconds: Int,
    ): ComposedConfirmedRecording? {
        clearEditor()
        viewModel.setLibraryAddedDays(case.libraryAddedDays)
        viewModel.setNumTracks(case.resultCount)
        viewModel.setFindMusicTextResultPlanner(case.planner)
        case.texts.forEachIndexed { index, text ->
            if (index > 0) viewModel.addTextIngredient()
            viewModel.updateTextIngredientQuery(index, text.query)
        }

        var confirmed: ComposedConfirmedRecording? = null
        case.recording?.let { recording ->
            viewModel.addSongSeed()
            val songIndex = viewModel.songSeeds.value.lastIndex
            viewModel.updateSongSeedQuery(songIndex, recording.lookupQuery)
            val seedId = viewModel.songSeeds.value[songIndex].id
            viewModel.searchSongSeed(songIndex)
            val terminal = withTimeout(timeoutSeconds * 1_000L) {
                viewModel.recordingLookupState.first { state ->
                    (state is RecordingLookupState.Success && state.seedId == seedId) ||
                        (state is RecordingLookupState.Failure && state.seedId == seedId)
                }
            }
            val success = terminal as? RecordingLookupState.Success
                ?: error((terminal as RecordingLookupState.Failure).message)
            val candidate = success.candidates.singleOrNull { track ->
                track.id == recording.expectedTrackId &&
                    track.artist.equals(recording.expectedArtist, ignoreCase = true) &&
                    track.title.equals(recording.expectedTitle, ignoreCase = true)
            } ?: error(
                "Recording lookup did not resolve exactly one " +
                    "${recording.expectedArtist} - ${recording.expectedTitle}",
            )
            viewModel.confirmSongSeed(songIndex, candidate)
            val confirmedState = viewModel.songSeeds.value[songIndex]
            require(confirmedState.confirmedTrack == candidate &&
                confirmedState.libraryBinding != null
            ) {
                "Production recording confirmation did not bind the selected track"
            }
            confirmed = candidate.toConfirmedRecording(confirmedState.stableTrackSpanId)
        }

        viewModel.setFindMusicOperator(case.operator)
        case.refineSpec?.let { refine ->
            viewModel.setFindMusicRefinePrimaryIngredient(refine.primaryIngredientIndex)
            viewModel.setFindMusicRefineNeighborhood(refine.neighborhood)
        }
        val firstWeight = case.texts.firstOrNull()?.weight
            ?: error("Every acceptance case needs a leading text ingredient")
        viewModel.updateTextIngredientWeight(0, firstWeight)
        viewModel.finalizeTextIngredientWeight(0)
        case.texts.forEachIndexed { index, text ->
            if (text.negative) viewModel.toggleTextIngredientSign(index)
        }
        require(viewModel.findMusicOperator.value == case.operator) {
            "Could not prepare ${case.operator} in the production editor"
        }
        case.refineSpec?.let { refine ->
            require(viewModel.findMusicRefinePrimaryIngredientIndex.value ==
                refine.primaryIngredientIndex &&
                viewModel.findMusicRefineNeighborhood.value == refine.neighborhood
            ) { "Production Refine controls differ from ${case.id}" }
        }
        val actualTexts = viewModel.textIngredients.value.filter { it.isActive }
        require(actualTexts.size == case.texts.size && actualTexts.zip(case.texts).all { (a, e) ->
            a.query == e.query && a.negative == e.negative && abs(a.weight - e.weight) <= 0.005f
        }) { "Production text editor state differs from ${case.id}" }
        val actualSongs = viewModel.songSeeds.value.filter { it.isActive }
        require(actualSongs.size == if (case.recording == null) 0 else 1) {
            "Production recording editor state differs from ${case.id}"
        }
        case.recording?.let { expected ->
            val actual = actualSongs.single()
            require(actual.negative == expected.negative &&
                abs(actual.weight - expected.weight) <= 0.005f
            ) { "Production recording weight/sign differs from ${case.id}" }
        }
        return confirmed
    }

    private fun clearEditor() {
        viewModel.clearFindMusicResults()
        viewModel.clearSongSeeds()
        while (viewModel.textIngredients.value.size > 1) {
            viewModel.removeTextIngredient(viewModel.textIngredients.value.lastIndex)
        }
        viewModel.updateTextIngredientQuery(0, "")
        viewModel.setFindMusicOperator(FindMusicOperator.ALL_OF)
    }

    private suspend fun executeCase(
        case: ComposedAcceptanceCase,
        repeat: Int,
        timeoutSeconds: Int,
        confirmedRecording: ComposedConfirmedRecording?,
    ): ComposedAcceptanceRun {
        val started = SystemClock.elapsedRealtimeNanos()
        viewModel.performFindMusicSearch()
        val result = withTimeout(timeoutSeconds * 1_000L) {
            combine(viewModel.multiSeedLoading, viewModel.multiSeedResult) { loading, value ->
                if (!loading) value else null
            }.filterNotNull().first()
        }
        val elapsedToResult = elapsedMs(started)
        val eligibility = withTimeout(timeoutSeconds * 1_000L) {
            viewModel.displayedQueueEligibility.first { value ->
                value != DisplayedFindMusicQueueEligibility.CHECKING &&
                    value != DisplayedFindMusicQueueEligibility.UNAVAILABLE
            }
        }
        val elapsedToReady = elapsedMs(started)
        require(eligibility.eligible) {
            eligibility.reason ?: "Composed result is not queue-ready"
        }
        val spec = requireNotNull(result.querySpec)
        val ingredientLabels = spec.activeEvidenceLabels
        val ingredientCount = requireNotNull(result.ingredientRankingDomainCount)
        val objectiveCount = requireNotNull(result.objectiveRankingDomainCount)
        val tracks = result.matches.mapIndexed { index, match ->
            val objectiveRank = requireNotNull(match.objectiveRank)
            val ingredients = match.anchorPercentiles.mapIndexed { ingredientIndex, percentile ->
                val exactRank = requireNotNull(
                    LibraryRankEvidenceText.rankFromUpperCdfPercentile(percentile, ingredientCount),
                ) { "Ingredient percentile is not an exact library rank" }
                ComposedIngredientEvidence(
                    label = ingredientLabels[ingredientIndex],
                    percentile = percentile,
                    exactRank = exactRank,
                    rankedTrackCount = ingredientCount,
                    topFraction = requireNotNull(
                        LibraryRankEvidenceText.topFraction(exactRank, ingredientCount),
                    ),
                )
            }
            ComposedAcceptanceTrack(
                displayedPosition = index + 1,
                embeddedTrackId = match.track.id,
                stableTrackSpanId = match.identity.stableTrackSpanId,
                artist = match.track.artist,
                album = match.track.album,
                title = match.track.title,
                durationMs = match.track.durationMs,
                filePath = match.track.filePath,
                objectiveRank = objectiveRank,
                rankingScore = match.rankingScore,
                overallEvidence = if (case.operator == FindMusicOperator.ALL_OF) {
                    LibraryRankEvidenceText.rankWithTopFraction(objectiveRank, objectiveCount)
                } else {
                    null
                },
                ingredientEvidence = ingredients,
            )
        }
        val possibleAvailableIdentityCounts = if (case.recording == null) {
            listOf(ingredientCount)
        } else {
            listOf(ingredientCount, ingredientCount - 1).filter { it > 0 }.distinct()
        }
        val allowedRefineCandidateCounts = if (case.operator == FindMusicOperator.REFINE) {
            val refine = requireNotNull(spec.refineSpec)
            possibleAvailableIdentityCounts
                .map(refine.neighborhood::candidateCount)
                .distinct()
                .sorted()
        } else {
            null
        }
        val binding = requireNotNull(result.libraryBinding)
        val reduction = requireNotNull(result.stableResultReduction)
        return ComposedAcceptanceRun(
            caseId = case.id,
            repeat = repeat,
            requestedOperator = case.operator,
            requestedRefineSpec = case.refineSpec,
            requestedTexts = case.texts,
            requestedRecording = case.recording,
            publishedQuerySpec = spec,
            elapsedToResultMs = elapsedToResult,
            elapsedToQueueReadyMs = elapsedToReady,
            resultFingerprint = resultFingerprint(result),
            libraryGenerationId = binding.generationId,
            activationBindingId = binding.activationBindingId,
            providerGenerationId = requireNotNull(result.providerGenerationId),
            orderedActiveTrackIdsSha256 = requireNotNull(result.orderedActiveTrackIdsSha256),
            activeTrackCount = requireNotNull(result.activeTrackCount),
            objectiveRankingDomainCount = objectiveCount,
            ingredientRankingDomainCount = ingredientCount,
            stableReductionIdentityPolicyVersion = reduction.identityPolicyVersion,
            stableReductionRequestedVisibleCount = reduction.requestedVisibleCount,
            stableReductionScannedRowCount = reduction.scannedRowCount,
            collapsedEquivalentCount = reduction.collapsedEquivalentCount,
            refineAllowedCandidateCounts = allowedRefineCandidateCounts,
            confirmedRecording = confirmedRecording,
            allOfQueuePlan = result.allOfQueuePlan,
            tracks = tracks,
        )
    }

    private fun validateRun(
        run: ComposedAcceptanceRun,
        case: ComposedAcceptanceCase,
        expectedLibrary: DatabaseInfo,
    ) {
        val spec = run.publishedQuerySpec
        require(spec.schemaVersion == FindMusicQuerySpec.CURRENT_SCHEMA_VERSION &&
            spec.rankingVersion == FindMusicQuerySpec.CURRENT_RANKING_VERSION &&
            spec.embeddingSpace == FindMusicQuerySpec.EMBEDDING_SPACE_CLAMP3 &&
            spec.textResultPlanner == case.planner &&
            spec.operator == case.operator && spec.resultLimit == case.resultCount &&
            spec.refineSpec == case.refineSpec &&
            spec.effectiveLibraryAddedDays == case.libraryAddedDays
        ) {
            "Published operator, Refine contract, result count, or date filter differs " +
                "from ${case.id}"
        }
        require(spec.activeTextIngredients.size == case.texts.size &&
            spec.activeTextIngredients.zip(case.texts).all { (actual, expected) ->
                actual.query == expected.query && actual.negative == expected.negative &&
                    abs(actual.weight - expected.weight) <= 0.005f
            }
        ) { "Published text ingredients differ from ${case.id}" }
        val publishedSongs = spec.songSeeds.filter { it.weight > 0f }
        require(publishedSongs.size == if (case.recording == null) 0 else 1) {
            "Published recording ingredients differ from ${case.id}"
        }
        case.recording?.let { expected ->
            val published = publishedSongs.single()
            val confirmed = requireNotNull(run.confirmedRecording)
            require(published.trackId == expected.expectedTrackId &&
                published.trackId == confirmed.embeddedTrackId &&
                published.stableTrackSpanId == confirmed.stableTrackSpanId &&
                published.artist.equals(expected.expectedArtist, ignoreCase = true) &&
                published.title.equals(expected.expectedTitle, ignoreCase = true) &&
                published.negative == expected.negative &&
                abs(published.weight - expected.weight) <= 0.005f
            ) { "Published recording identity, sign, or weight differs from ${case.id}" }
        }
        val queryBinding = requireNotNull(spec.libraryBinding) {
            "Published query has no exact library binding"
        }
        require(run.libraryGenerationId == expectedLibrary.generationId &&
            run.providerGenerationId == expectedLibrary.providerGenerationId &&
            run.activeTrackCount == expectedLibrary.activeTrackCount &&
            run.orderedActiveTrackIdsSha256.matches(SHA256) &&
            run.activationBindingId.isNotBlank() &&
            queryBinding.generationId == run.libraryGenerationId &&
            queryBinding.activationBindingId == run.activationBindingId
        ) { "The active library changed during ${case.id}" }
        val maximumExcludedRecordingIdentityCount = if (case.recording == null) 0 else 1
        val possibleAvailableIdentityCounts =
            (0..maximumExcludedRecordingIdentityCount).map { excludedCount ->
                run.ingredientRankingDomainCount - excludedCount
            }.filter { it > 0 }
        val allowedObjectiveDomainCounts = when (case.operator) {
            FindMusicOperator.ALL_OF -> possibleAvailableIdentityCounts
            FindMusicOperator.REFINE -> requireNotNull(case.refineSpec).neighborhood.let { neighborhood ->
                possibleAvailableIdentityCounts.map(neighborhood::candidateCount)
            }
        }
        require(run.ingredientRankingDomainCount > 0 &&
            run.objectiveRankingDomainCount in allowedObjectiveDomainCounts
        ) { "Composed ranking domain differs from the exact ${case.operator} contract" }
        require(run.tracks.size == case.resultCount &&
            run.tracks.map { it.embeddedTrackId }.toSet().size == run.tracks.size &&
            run.tracks.map { it.objectiveRank }.toSet().size == run.tracks.size
        ) { "${case.id} did not return a complete unique visible result set" }
        val stableResultIds = run.tracks.mapNotNull { it.stableTrackSpanId }
        require(stableResultIds.toSet().size == stableResultIds.size &&
            run.stableReductionRequestedVisibleCount == case.resultCount &&
            run.stableReductionScannedRowCount == case.resultCount &&
            run.collapsedEquivalentCount == 0
        ) { "${case.id} did not preserve the verified stable-identity result prefix" }
        require(run.tracks.all { it.ingredientEvidence.size == case.activeIngredientCount }) {
            "${case.id} omitted ingredient evidence"
        }
        val publishedWeights = spec.activeTextIngredients.map { it.weight } +
            publishedSongs.map { it.weight }
        if (case.operator == FindMusicOperator.ALL_OF) {
            require(run.tracks.all { it.overallEvidence != null }) {
                "All-of evidence is incomplete for ${case.id}"
            }
            require(
                case.planner != FindMusicTextResultPlanner.CLOSEST ||
                    run.tracks.map { it.objectiveRank }
                        .zipWithNext().all { (a, b) -> a < b },
            ) {
                "Ranked All-of objective ranks are not ordered for ${case.id}"
            }
            if (case.planner == FindMusicTextResultPlanner.VARIED_ALL_OF_DPP) {
                val plan = requireNotNull(run.allOfQueuePlan) {
                    "Varied All-of result omitted its complete-domain plan"
                }
                plan.requireValid()
                require(
                    plan.completeCandidateDomainCount == run.objectiveRankingDomainCount &&
                        plan.requestedResultCount == case.resultCount &&
                        plan.orderedSelectedTrackIds ==
                        run.tracks.map(ComposedAcceptanceTrack::embeddedTrackId) &&
                        plan.orderedOriginalAllOfObjectiveRanks ==
                        run.tracks.map(ComposedAcceptanceTrack::objectiveRank),
                ) { "Varied All-of result differs from its persisted selection proof" }
            } else {
                require(run.allOfQueuePlan == null) {
                    "Ranked All-of result unexpectedly carried a Varied plan"
                }
            }
            run.tracks.forEach { track ->
                val totalWeight = publishedWeights.sumOf { abs(it.toDouble()) }
                val percentileFloor = 1.0 / run.ingredientRankingDomainCount.toDouble()
                val expectedScore = exp(
                    track.ingredientEvidence.zip(publishedWeights).sumOf { (ingredient, weight) ->
                        abs(weight.toDouble()) / totalWeight *
                            ln(ingredient.percentile.toDouble().coerceIn(percentileFloor, 1.0))
                    },
                ).toFloat()
                require(abs(expectedScore - track.rankingScore) <= 1e-7f) {
                    "All-of row score does not equal its weighted geometric mean"
                }
            }
        } else {
            val refine = requireNotNull(spec.refineSpec)
            val primaryIndex = refine.primaryIngredientIndex
            val secondaryIndex = 1 - primaryIndex
            require(run.requestedRefineSpec == refine &&
                run.refineAllowedCandidateCounts == allowedObjectiveDomainCounts.distinct().sorted() &&
                run.tracks.all { it.overallEvidence == null }
            ) {
                "Refine request or neighborhood evidence is incomplete for ${case.id}"
            }
            run.tracks.forEach { track ->
                require(track.ingredientEvidence[primaryIndex].exactRank <=
                    run.objectiveRankingDomainCount + maximumExcludedRecordingIdentityCount &&
                    track.rankingScore.toRawBits() ==
                    track.ingredientEvidence[secondaryIndex].percentile.toRawBits()
                ) { "Refine row is outside the primary neighborhood or secondary ordering" }
            }
            require(run.tracks.zipWithNext().all { (left, right) ->
                val leftSecondary = left.ingredientEvidence[secondaryIndex].percentile
                val rightSecondary = right.ingredientEvidence[secondaryIndex].percentile
                val leftPrimary = left.ingredientEvidence[primaryIndex].percentile
                val rightPrimary = right.ingredientEvidence[primaryIndex].percentile
                leftSecondary > rightSecondary ||
                    (leftSecondary == rightSecondary && leftPrimary >= rightPrimary)
            }) { "Refine results are not ordered by refiner then primary percentile" }
        }
        run.confirmedRecording?.let { recording ->
            require(run.tracks.none {
                it.embeddedTrackId == recording.embeddedTrackId ||
                    (recording.stableTrackSpanId != null &&
                        it.stableTrackSpanId == recording.stableTrackSpanId)
            }) {
                "The recording anchor or a byte-identical equivalent was returned as a result"
            }
        }
    }

    private fun resultFingerprint(result: TextSearchResult): String {
        require(result.kind == FindMusicResultKind.COMPOSED && result.error == null)
        val digest = MessageDigest.getInstance("SHA-256")
        digest.update("production-composed-find-music-result-v3\u0000".toByteArray(Charsets.US_ASCII))
        fun update(value: String?) {
            digest.update(value.orEmpty().toByteArray(Charsets.UTF_8))
            digest.update(0)
        }
        val spec = requireNotNull(result.querySpec)
        val binding = requireNotNull(result.libraryBinding)
        val reduction = requireNotNull(result.stableResultReduction)
        update(spec.stateKey)
        update(binding.generationId)
        update(binding.activationBindingId)
        update(binding.databaseContentSha256)
        update(binding.orderedTrackSetSha256)
        update(result.providerGenerationId)
        update(result.orderedActiveTrackIdsSha256)
        digest.update(
            ByteBuffer.allocate(Int.SIZE_BYTES * 7).order(ByteOrder.BIG_ENDIAN).apply {
                putInt(requireNotNull(result.activeTrackCount))
                putInt(requireNotNull(result.objectiveRankingDomainCount))
                putInt(requireNotNull(result.ingredientRankingDomainCount))
                putInt(reduction.identityPolicyVersion)
                putInt(reduction.requestedVisibleCount)
                putInt(reduction.scannedRowCount)
                putInt(reduction.collapsedEquivalentCount)
            }.array(),
        )
        result.matches.forEachIndexed { position, match ->
            val numeric = ByteBuffer.allocate(
                Long.SIZE_BYTES + Int.SIZE_BYTES * (3 + match.anchorPercentiles.size),
            ).order(ByteOrder.BIG_ENDIAN)
            numeric.putLong(match.track.id)
            numeric.putInt(requireNotNull(match.objectiveRank))
            numeric.putInt(position + 1)
            numeric.putInt(match.rankingScore.toRawBits())
            match.anchorPercentiles.forEach { numeric.putInt(it.toRawBits()) }
            digest.update(numeric.array())
            digest.update(match.identity.stableTrackSpanId.orEmpty().toByteArray(Charsets.US_ASCII))
            digest.update(0)
        }
        return digest.digest().toHex()
    }

    private fun buildPairComparisons(
        runs: List<ComposedAcceptanceRun>,
    ): List<ComposedPairComparison> {
        val firstRuns = runs.filter { it.repeat == 1 }.associateBy { it.caseId }
        return listOf(
            Triple("weight", "all_recording70_sleep30", "all_recording30_sleep70"),
            Triple(
                "Like versus Avoid",
                "all_recording70_sleep30",
                "all_recording70_avoid_sleep30",
            ),
            Triple(
                "Ranked versus Varied",
                "all_recording70_sleep30",
                "all_recording70_sleep30_varied",
            ),
            Triple(
                "All of versus Refine",
                "all_recording70_sleep30",
                "refine_recording_by_sleep_default",
            ),
        ).mapNotNull { (control, leftId, rightId) ->
            val left = firstRuns[leftId] ?: return@mapNotNull null
            val right = firstRuns[rightId] ?: return@mapNotNull null
            val leftById = left.tracks.associateBy { it.embeddedTrackId }
            val rightById = right.tracks.associateBy { it.embeddedTrackId }
            val shared = leftById.keys intersect rightById.keys
            fun ComposedAcceptanceTrack.label(): String =
                "${artist ?: "Unknown artist"} - ${title ?: "Unknown title"} " +
                    "(track $embeddedTrackId)"
            ComposedPairComparison(
                changedControl = control,
                leftCaseId = leftId,
                rightCaseId = rightId,
                sharedTrackCount = shared.size,
                changedTrackCountPerSide = left.tracks.size - shared.size,
                sharedTracksThatMoved = shared.count { id ->
                    leftById.getValue(id).displayedPosition !=
                        rightById.getValue(id).displayedPosition
                },
                sameOrderedResult = left.tracks.map { it.embeddedTrackId } ==
                    right.tracks.map { it.embeddedTrackId },
                enteredRight = right.tracks.filter { it.embeddedTrackId !in leftById }
                    .map { it.label() },
                exitedRight = left.tracks.filter { it.embeddedTrackId !in rightById }
                    .map { it.label() },
            )
        }
    }

    private fun EmbeddedTrack.toConfirmedRecording(
        stableTrackSpanId: String?,
    ): ComposedConfirmedRecording = ComposedConfirmedRecording(
        embeddedTrackId = id,
        stableTrackSpanId = stableTrackSpanId,
        artist = artist,
        album = album,
        title = title,
        durationMs = durationMs,
        filePath = filePath,
    )

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

private data class ComposedPreferenceSnapshot(
    val values: Map<String, ComposedPreferenceValue>,
) {
    fun sha256(): String {
        val digest = MessageDigest.getInstance("SHA-256")
        digest.update("composed-find-music-settings-v1\u0000".toByteArray(Charsets.US_ASCII))
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
        return digest.digest().toHexString()
    }

    fun restore(preferences: SharedPreferences): Boolean {
        val editor = preferences.edit().clear()
        values.forEach { (key, value) -> value.put(editor, key) }
        return editor.commit()
    }

    companion object {
        fun capture(preferences: SharedPreferences): ComposedPreferenceSnapshot =
            ComposedPreferenceSnapshot(
                preferences.all.mapValues { (_, raw) -> ComposedPreferenceValue.from(raw) },
            )
    }
}

private data class ComposedPreferenceValue(
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
        fun from(raw: Any?): ComposedPreferenceValue = when (raw) {
            is Boolean -> ComposedPreferenceValue("boolean", listOf(raw.toString()))
            is Float -> ComposedPreferenceValue("float", listOf(raw.toRawBits().toString()))
            is Int -> ComposedPreferenceValue("int", listOf(raw.toString()))
            is Long -> ComposedPreferenceValue("long", listOf(raw.toString()))
            is String -> ComposedPreferenceValue("string", listOf(raw))
            is Set<*> -> ComposedPreferenceValue(
                "string_set",
                raw.map { requireNotNull(it as? String) }.sorted(),
            )
            else -> error("Unsupported preference value ${raw?.javaClass?.name}")
        }
    }
}

private fun ByteArray.toHexString(): String = joinToString("") { "%02x".format(it) }
