package com.powerampstartradio.debug

import android.content.Intent
import android.content.Context
import android.os.Bundle
import android.os.Process
import android.os.SystemClock
import android.util.AtomicFile
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.lifecycle.lifecycleScope
import com.google.gson.Gson
import com.google.gson.GsonBuilder
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import com.powerampstartradio.data.EmbeddingDatabase
import com.powerampstartradio.indexing.IndexingService
import com.powerampstartradio.indexing.NewTrackDetector
import com.powerampstartradio.indexing.V2IndexingAttentionHistorySource
import com.powerampstartradio.indexing.V2IndexingReadinessPolicy
import com.powerampstartradio.indexing.V2IndexingSelectionPolicy
import com.powerampstartradio.indexing.V2ResolvedTrackExclusions
import com.powerampstartradio.indexing.V2TrackExclusionPolicy
import com.powerampstartradio.indexing.V2TrackExclusionRepository
import com.powerampstartradio.indexing.indexingNotificationEvidence
import com.powerampstartradio.indexing.v2.AtomicV2IndexingPreflightIntentStore
import com.powerampstartradio.indexing.v2.V2ActiveIndexingJobPointer
import com.powerampstartradio.indexing.v2.V2ActiveIndexingJobPointerInspection
import com.powerampstartradio.indexing.v2.V2CurrentModelPolicyResolver
import com.powerampstartradio.indexing.v2.V2IndexGenerationReader
import com.powerampstartradio.indexing.v2.V2IndexingExecutionProfile
import com.powerampstartradio.indexing.v2.V2IndexingJobRepository
import com.powerampstartradio.indexing.v2.V2IndexingPreflightIntent
import com.powerampstartradio.indexing.v2.V2IndexingPreflightIntentFactory
import com.powerampstartradio.indexing.v2.V2IndexingPreflightRequestFingerprint
import com.powerampstartradio.indexing.v2.V2IndexingPreflightSelection
import com.powerampstartradio.indexing.v2.V2LibraryDatabaseResolver
import com.powerampstartradio.indexing.v2.V2MediaExtractorAudioInspector
import com.powerampstartradio.indexing.v2.V2PowerampProviderSnapshotAcquirer
import com.powerampstartradio.indexing.v2.V2ProviderDurationEvidencePolicy
import com.powerampstartradio.indexing.v2.V2ProviderPathGroupEvidence
import com.powerampstartradio.indexing.v2.V2ProviderPathRowEvidence
import com.powerampstartradio.poweramp.TrackNormalization
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileInputStream
import java.io.OutputStreamWriter
import java.nio.charset.StandardCharsets
import java.security.MessageDigest

/**
 * Debug-only adb bridge for connected acceptance of the real production indexing service.
 *
 * REPORT is read-only. START accepts only the exact REPORT artifact and replays every provider,
 * generation, model, container, and source-byte check before submitting the normal immutable
 * preflight intent. SNAPSHOT copies process-local ETA evidence and the durable ledger without
 * sending playback or Poweramp queue commands.
 */
class ProductionIndexingAcceptanceActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        if (savedInstanceState != null) return

        lifecycleScope.launch {
            val command = intent.getStringExtra(EXTRA_COMMAND)?.trim()?.lowercase()
            val inputRelativePath = intent.getStringExtra(EXTRA_INPUT_RELATIVE_PATH)
            val expectedInputSha256 = intent.getStringExtra(EXTRA_INPUT_SHA256)
            val outputRelativePath = intent.getStringExtra(EXTRA_OUTPUT_RELATIVE_PATH)
            val output = runCatching {
                withContext(Dispatchers.IO) {
                    val paths = AcceptancePrivatePaths(
                        filesDir = filesDir,
                        inputRelativePath = requireNotNull(inputRelativePath) {
                            "missing $EXTRA_INPUT_RELATIVE_PATH"
                        },
                        outputRelativePath = requireNotNull(outputRelativePath) {
                            "missing $EXTRA_OUTPUT_RELATIVE_PATH"
                        },
                    )
                    val input = paths.readVerifiedInput(
                        requireNotNull(expectedInputSha256) { "missing $EXTRA_INPUT_SHA256" },
                    )
                    val runner = ProductionIndexingAcceptanceRunner(this@ProductionIndexingAcceptanceActivity)
                    val result = when (command) {
                        COMMAND_REPORT -> runner.report(input)
                        COMMAND_START -> runner.start(input)
                        COMMAND_SNAPSHOT -> runner.snapshot(input)
                        else -> error("unsupported acceptance command: $command")
                    }
                    paths.writeOutput(result)
                    result
                }
            }.getOrElse { error ->
                Log.e(TAG, "Production indexing acceptance command failed", error)
                val failure = GSON.toJson(
                    AcceptanceCommandFailure(
                        command = command,
                        capturedAtEpochMs = System.currentTimeMillis(),
                        errorType = error::class.java.simpleName,
                        message = (error.message ?: "unknown acceptance failure").take(2_000),
                    ),
                )
                runCatching {
                    withContext(Dispatchers.IO) {
                        if (inputRelativePath != null && outputRelativePath != null) {
                            AcceptancePrivatePaths(filesDir, inputRelativePath, outputRelativePath)
                                .writeOutput(failure)
                        }
                    }
                }
                failure
            }
            setResult(RESULT_OK, Intent().putExtra(EXTRA_RESULT_JSON, output))
            finishAndRemoveTask()
        }
    }

    companion object {
        const val EXTRA_COMMAND = "acceptance_command"
        const val EXTRA_INPUT_RELATIVE_PATH = "acceptance_input_relative_path"
        const val EXTRA_INPUT_SHA256 = "acceptance_input_sha256"
        const val EXTRA_OUTPUT_RELATIVE_PATH = "acceptance_output_relative_path"
        const val EXTRA_RESULT_JSON = "acceptance_result_json"
        const val COMMAND_REPORT = "report"
        const val COMMAND_START = "start"
        const val COMMAND_SNAPSHOT = "snapshot"

        private const val TAG = "ProdIndexAcceptance"
        private val GSON = GsonBuilder()
            .disableHtmlEscaping()
            .serializeNulls()
            .create()
    }
}

private class ProductionIndexingAcceptanceRunner(
    activity: ComponentActivity,
) {
    private val app = activity.applicationContext
    private val filesDir = activity.filesDir

    fun report(inputJson: String): String {
        val request = AcceptanceJson.parseRequest(inputJson)
        require(V2ActiveIndexingJobPointer(filesDir).inspect() ==
            V2ActiveIndexingJobPointerInspection.Missing
        ) { "an active or unreadable indexing pointer already exists" }
        val manifest = buildCandidateManifest(request, capturedAtEpochMs = System.currentTimeMillis())
        return GSON.toJson(manifest)
    }

    fun start(inputJson: String): String {
        val pinned = AcceptanceJson.parseManifest(inputJson)
        AcceptanceValidation.requireManifest(pinned)
        require(pinned.tracks.isNotEmpty()) { "candidate manifest has no tracks" }
        require(pinned.tracks.all { it.decision == AcceptanceCandidateDecision.READY }) {
            "candidate manifest contains a blocked track"
        }
        require(pinned.tracks.none(AcceptanceCandidateTrack::isCueGroup)) {
            "CUE rows remain blocked until atomic imported-row supersession is implemented"
        }

        val intent = pinned.toPreflightIntent()
        val existing = AtomicV2IndexingPreflightIntentStore(
            File(filesDir, "indexing_v2/preflight-intents"),
        ).load(pinned.jobId)
        if (existing != null) {
            requireSameRequest(existing, intent)
            requirePointerMatches(pinned.jobId)
            return GSON.toJson(
                AcceptanceStartResult(
                    runId = pinned.runId,
                    jobId = pinned.jobId,
                    result = AcceptanceStartDisposition.ALREADY_SUBMITTED,
                    capturedAtEpochMs = System.currentTimeMillis(),
                    requestFingerprint = V2IndexingPreflightRequestFingerprint.compute(existing),
                    durableState = existing.state.name,
                    message = "The exact immutable request is already durable; no duplicate start was sent.",
                ),
            )
        }

        require(V2ActiveIndexingJobPointer(filesDir).inspect() ==
            V2ActiveIndexingJobPointerInspection.Missing
        ) { "another active or unreadable indexing pointer exists" }
        val current = buildCandidateManifest(
            pinned.toRequest(),
            capturedAtEpochMs = pinned.capturedAtEpochMs,
        )
        require(current == pinned) {
            "candidate evidence changed since REPORT; generate and review a new manifest"
        }

        val disposition = try {
            IndexingService.submitPreflight(app, intent)
            AcceptanceStartDisposition.SUBMITTED
        } catch (launchError: Throwable) {
            val durable = AtomicV2IndexingPreflightIntentStore(
                File(filesDir, "indexing_v2/preflight-intents"),
            ).load(pinned.jobId)
            if (durable == null) throw launchError
            requireSameRequest(durable, intent)
            requirePointerMatches(pinned.jobId)
            AcceptanceStartDisposition.SUBMITTED_LAUNCH_DEFERRED
        }
        val durable = AtomicV2IndexingPreflightIntentStore(
            File(filesDir, "indexing_v2/preflight-intents"),
        ).require(pinned.jobId)
        return GSON.toJson(
            AcceptanceStartResult(
                runId = pinned.runId,
                jobId = pinned.jobId,
                result = disposition,
                capturedAtEpochMs = System.currentTimeMillis(),
                requestFingerprint = V2IndexingPreflightRequestFingerprint.compute(durable),
                durableState = durable.state.name,
                message = if (disposition == AcceptanceStartDisposition.SUBMITTED) {
                    "The exact manifest was submitted through production preflight."
                } else {
                    "The request is durable, but Android deferred the foreground launch."
                },
            ),
        )
    }

    fun snapshot(inputJson: String): String {
        val pinned = AcceptanceJson.parseManifest(inputJson)
        AcceptanceValidation.requireManifest(pinned)
        val pointer = V2ActiveIndexingJobPointer(filesDir).inspect()
        val pointerJobId = (pointer as? V2ActiveIndexingJobPointerInspection.Readable)?.jobId
        require(pointerJobId == null || pointerJobId == pinned.jobId) {
            "active indexing pointer belongs to a different job"
        }
        val intent = AtomicV2IndexingPreflightIntentStore(
            File(filesDir, "indexing_v2/preflight-intents"),
        ).load(pinned.jobId)
        val ledger = runCatching {
            V2IndexingJobRepository.get(app).require(pinned.jobId)
        }.getOrNull()
        val state = IndexingService.state.value
        val stateJobId = when (state) {
            is IndexingService.IndexingState.JobSnapshot -> state.jobId
            is IndexingService.IndexingState.PreflightSnapshot -> state.jobId
            is IndexingService.IndexingState.Error -> state.jobId
            IndexingService.IndexingState.Idle -> null
        }
        val notification = (state as? IndexingService.IndexingState.JobSnapshot)?.let {
            indexingNotificationEvidence(it.jobState, it.event)
        }
        return GSON.toJson(
            AcceptanceRuntimeSnapshot(
                runId = pinned.runId,
                jobId = pinned.jobId,
                capturedAtEpochMs = System.currentTimeMillis(),
                elapsedRealtimeMs = SystemClock.elapsedRealtime(),
                processId = Process.myPid(),
                activePointer = pointer.toString(),
                activePointerJobId = pointerJobId,
                processStateKind = state::class.java.simpleName,
                processStateJobId = stateJobId,
                serviceState = state,
                notificationEvidence = notification,
                preflightIntent = intent,
                ledger = ledger,
            ),
        )
    }

    private fun buildCandidateManifest(
        request: AcceptanceCandidateRequest,
        capturedAtEpochMs: Long,
    ): AcceptanceCandidateManifest {
        AcceptanceValidation.requireRequest(request)
        require(filesDir.usableSpace >= request.minimumUsableBytes) {
            "private storage has ${filesDir.usableSpace} usable bytes; " +
                "manifest requires ${request.minimumUsableBytes}"
        }
        val active = V2LibraryDatabaseResolver.requirePublished(filesDir)
        val provider = V2PowerampProviderSnapshotAcquirer(app).acquireBlocking()
        val providerGenerationId = requireNotNull(provider.libraryGeneration) {
            "complete Poweramp provider snapshot has no generation identity"
        }
        val groups = provider.groups
        val rowsById = groups.flatMap(V2ProviderPathGroupEvidence::rows)
            .associateBy(V2ProviderPathRowEvidence::powerampFileId)
        require(rowsById.size == provider.groups.sumOf { it.rows.size }) {
            "provider snapshot contains duplicate Poweramp IDs"
        }

        val database = EmbeddingDatabase.open(active.databaseFile)
        val unindexedTracks = try {
            NewTrackDetector(database).findUnindexedTracks(provider)
        } finally {
            database.close()
        }
        val unindexedById = unindexedTracks.associateBy(
            NewTrackDetector.UnindexedTrack::powerampFileId,
        )
        val resolvedSelection = resolveCandidateSelection(request, unindexedTracks)
        val groupByPath = groups.associateBy(V2ProviderPathGroupEvidence::physicalPath)
        val inspector = V2MediaExtractorAudioInspector()
        val sourceEvidence = mutableMapOf<String, AcceptanceSourceEvidence>()
        val candidates = resolvedSelection.selectedPowerampFileIds.map { id ->
            val row = rowsById[id]
            val group = row?.let { groupByPath[it.physicalPath] }
            val unindexed = unindexedById[id]
            candidateTrack(id, row, group, unindexed) {
                sourceEvidence.getOrPut(requireNotNull(row).physicalPath) {
                    val file = File(row.providerPhysicalPath)
                    require(file.isFile && file.canRead()) {
                        "selected source is not a readable regular file: ${row.providerPhysicalPath}"
                    }
                    val container = inspector.inspect(row.providerPhysicalPath)
                    val byteLength = file.length()
                    val modifiedEpochMs = file.lastModified()
                    val sourceSha256 = sha256(file)
                    require(file.isFile && file.canRead() &&
                        file.length() == byteLength && file.lastModified() == modifiedEpochMs
                    ) { "selected source changed while it was hashed: ${row.providerPhysicalPath}" }
                    AcceptanceSourceEvidence(
                        byteLength = byteLength,
                        modifiedEpochMs = modifiedEpochMs,
                        sha256 = sourceSha256,
                        containerMime = container.mime,
                        containerDurationUsEstimate = container.durationUsEstimate,
                        containerSampleRateHz = container.sampleRateHz,
                        containerChannelCount = container.channelCount,
                    )
                }
            }
        }

        val modelPolicy = V2CurrentModelPolicyResolver.resolveFresh(filesDir)
        val providerAfter = V2PowerampProviderSnapshotAcquirer(app).acquireBlocking()
        require(providerAfter.libraryGeneration == provider.libraryGeneration) {
            "Poweramp provider changed while candidate sources were inspected"
        }
        val activeAfter = V2IndexGenerationReader.requireActive(filesDir)
        require(activeAfter.manifest.generationId == active.manifest.generationId &&
            activeAfter.manifestSha256 == active.manifestSha256
        ) { "active index generation changed while candidate sources were inspected" }
        val resolvedSelectionAfter = resolveCandidateSelection(request, unindexedTracks)
        require(resolvedSelectionAfter == resolvedSelection) {
            "production-ready selection changed while candidate sources were inspected"
        }
        return AcceptanceCandidateManifest(
            runId = request.runId,
            purpose = request.purpose,
            jobId = request.jobId,
            jobCreatedAtEpochMs = request.jobCreatedAtEpochMs,
            capturedAtEpochMs = capturedAtEpochMs,
            applicationId = app.packageName,
            installedApkSha256 = sha256(File(app.applicationInfo.sourceDir)),
            activeGenerationId = active.manifest.generationId,
            activeManifestSha256 = active.manifestSha256,
            providerGenerationId = providerGenerationId,
            receiptEmbeddingSpecId = modelPolicy.receiptEmbeddingSpec.specId,
            textRetrievalSpecId = modelPolicy.textRetrievalSpec.specId,
            selectionMode = request.selectionMode,
            readyCap = request.readyCap,
            selectionPolicy = resolvedSelection.selectionPolicy,
            discoveredReadyPowerampFileIds = resolvedSelection.discoveredReadyPowerampFileIds,
            discoveredReadyFingerprint = resolvedSelection.discoveredReadyFingerprint,
            executionProfile = request.executionProfile,
            rebuildDerivedIndexes = request.rebuildDerivedIndexes,
            minimumUsableBytes = request.minimumUsableBytes,
            cuePolicy = CUE_POLICY,
            missingSourcePolicy = MISSING_SOURCE_POLICY,
            tracks = candidates,
        ).also(AcceptanceValidation::requireManifest)
    }

    private fun resolveCandidateSelection(
        request: AcceptanceCandidateRequest,
        tracks: List<NewTrackDetector.UnindexedTrack>,
    ): AcceptanceResolvedSelection = when (
        AcceptanceSelectionMode.valueOf(request.selectionMode)
    ) {
        AcceptanceSelectionMode.EXPLICIT_IDS -> AcceptanceResolvedSelection(
            selectedPowerampFileIds = request.powerampFileIds.sorted(),
            discoveredReadyPowerampFileIds = null,
            discoveredReadyFingerprint = null,
            selectionPolicy = EXPLICIT_SELECTION_POLICY,
        )

        AcceptanceSelectionMode.ALL_READY,
        AcceptanceSelectionMode.READY_CAP,
        -> {
            val exclusions = loadTrackExclusionsReadOnly(tracks)
            val attentionHistory = V2IndexingAttentionHistorySource(app).load()
            val readyIds = (V2IndexingReadinessPolicy.readyTrackIds(
                tracks = tracks,
                exclusions = exclusions,
                attentionHistory = attentionHistory,
            ) - exclusions.ignoredIds).sorted()
            require(readyIds.size <= MAX_READY_UNIVERSE) {
                "ready universe exceeds the acceptance reporting limit"
            }
            if (request.selectionMode == AcceptanceSelectionMode.ALL_READY.name) {
                require(readyIds.size <= MAX_TRACKS) {
                    "all-ready contains ${readyIds.size} tracks; use a capped report"
                }
            }
            val selectedIds = when (AcceptanceSelectionMode.valueOf(request.selectionMode)) {
                AcceptanceSelectionMode.ALL_READY -> readyIds
                AcceptanceSelectionMode.READY_CAP -> readyIds.take(requireNotNull(request.readyCap))
                AcceptanceSelectionMode.EXPLICIT_IDS -> error("unreachable explicit selection")
            }
            AcceptanceResolvedSelection(
                selectedPowerampFileIds = selectedIds,
                discoveredReadyPowerampFileIds = readyIds,
                discoveredReadyFingerprint = readyUniverseFingerprint(
                    readyIds = readyIds,
                    exclusionFingerprint = exclusions.persistedFingerprint,
                    attentionFingerprint = attentionHistory.fingerprint,
                ),
                selectionPolicy = AUTOMATIC_READY_SELECTION_POLICY,
            )
        }
    }

    /** Resolves current V2 exclusion envelopes without migration or preference writes. */
    private fun loadTrackExclusionsReadOnly(
        tracks: List<NewTrackDetector.UnindexedTrack>,
    ): V2ResolvedTrackExclusions {
        val preferences = app.getSharedPreferences(
            V2TrackExclusionRepository.PREFERENCES_NAME,
            Context.MODE_PRIVATE,
        )
        val hasPendingLegacyNever =
            !preferences.contains(V2TrackExclusionRepository.NEVER_EXCLUSIONS_KEY) &&
                preferences.contains("dismissed_track_ids")
        val hasPendingLegacyIgnored =
            !preferences.contains(V2TrackExclusionRepository.IGNORED_EXCLUSIONS_KEY) &&
                preferences.contains("ignored_track_ids")
        require(!hasPendingLegacyNever && !hasPendingLegacyIgnored) {
            "saved V1 indexing choices still require production migration; " +
                "open Manage tracks once, then generate a fresh candidate report"
        }
        val repository = V2TrackExclusionRepository(app)
        val (never, ignored) = repository.loadPersisted()
        val candidates = tracks.mapNotNull(V2TrackExclusionRepository::candidate)
        return V2ResolvedTrackExclusions(
            never = never,
            ignored = ignored,
            neverIds = V2TrackExclusionPolicy.resolve(never, candidates),
            ignoredIds = V2TrackExclusionPolicy.resolve(ignored, candidates),
            persistedFingerprint = repository.persistedFingerprint(),
        )
    }

    private fun candidateTrack(
        requestedPowerampFileId: Long,
        row: V2ProviderPathRowEvidence?,
        group: V2ProviderPathGroupEvidence?,
        unindexed: NewTrackDetector.UnindexedTrack?,
        source: () -> AcceptanceSourceEvidence,
    ): AcceptanceCandidateTrack {
        if (row == null || group == null) {
            return AcceptanceCandidateTrack.missingProvider(requestedPowerampFileId)
        }
        val groupHasOffsets = group.rows.any { it.offsetMs > 0L }
        val groupHasCueImage = group.rows.any { it.cueSourceImageFolderId != null }
        val base = AcceptanceCandidateTrack(
            powerampFileId = row.powerampFileId,
            artist = TrackNormalization.normalizeArtist(row.artist),
            album = TrackNormalization.normalizeAlbum(row.album),
            title = TrackNormalization.normalizeTitle(row.title),
            providerPhysicalPath = row.providerPhysicalPath,
            durationMs = V2ProviderDurationEvidencePolicy.canonicalMs(row.durationMs),
            offsetMs = row.offsetMs,
            cueSourceImageFolderId = row.cueSourceImageFolderId,
            sourceReferenceCount = group.rows.size,
            sourceHasLogicalOffsets = groupHasOffsets,
            sourceHasCueImageRow = groupHasCueImage,
            detectionKind = unindexed?.detectionKind?.name,
            decision = AcceptanceCandidateDecision.BLOCKED_NOT_READY,
            blocker = null,
            sourceByteLength = null,
            sourceModifiedEpochMs = null,
            sourceSha256 = null,
            containerMime = null,
            containerDurationUsEstimate = null,
            containerSampleRateHz = null,
            containerChannelCount = null,
        )
        if (base.isCueGroup()) {
            return base.copy(
                decision = AcceptanceCandidateDecision.BLOCKED_CUE_SUPERSESSION,
                blocker = CUE_POLICY,
            )
        }
        if (unindexed == null) {
            return base.copy(
                decision = AcceptanceCandidateDecision.BLOCKED_ALREADY_REPRESENTED,
                blocker = "The current active index does not classify this provider occurrence as unindexed.",
            )
        }
        if (!V2IndexingSelectionPolicy.isReadyTrack(unindexed)) {
            return base.copy(
                decision = AcceptanceCandidateDecision.BLOCKED_NOT_READY,
                blocker = "Detection kind ${unindexed.detectionKind} is not production-ready.",
            )
        }
        val evidence = try {
            source()
        } catch (error: Throwable) {
            return base.copy(
                decision = AcceptanceCandidateDecision.BLOCKED_SOURCE_OR_CONTAINER,
                blocker = (error.message ?: error::class.java.simpleName).take(1_000),
            )
        }
        return base.copy(
            decision = AcceptanceCandidateDecision.READY,
            sourceByteLength = evidence.byteLength,
            sourceModifiedEpochMs = evidence.modifiedEpochMs,
            sourceSha256 = evidence.sha256,
            containerMime = evidence.containerMime,
            containerDurationUsEstimate = evidence.containerDurationUsEstimate,
            containerSampleRateHz = evidence.containerSampleRateHz,
            containerChannelCount = evidence.containerChannelCount,
        )
    }

    private fun AcceptanceCandidateManifest.toRequest() = AcceptanceCandidateRequest(
        runId = runId,
        purpose = purpose,
        jobId = jobId,
        jobCreatedAtEpochMs = jobCreatedAtEpochMs,
        powerampFileIds = if (selectionMode == AcceptanceSelectionMode.EXPLICIT_IDS.name) {
            tracks.map(AcceptanceCandidateTrack::powerampFileId)
        } else {
            emptyList()
        },
        selectionMode = selectionMode,
        readyCap = readyCap,
        executionProfile = executionProfile,
        rebuildDerivedIndexes = rebuildDerivedIndexes,
        minimumUsableBytes = minimumUsableBytes,
    )

    private fun AcceptanceCandidateManifest.toPreflightIntent(): V2IndexingPreflightIntent =
        V2IndexingPreflightIntentFactory.create(
            jobId = jobId,
            selected = tracks.map { track ->
                V2IndexingPreflightSelection(
                    powerampFileId = track.powerampFileId,
                    providerPhysicalPath = requireNotNull(track.providerPhysicalPath),
                    durationMs = requireNotNull(track.durationMs),
                    offsetMs = requireNotNull(track.offsetMs),
                    cueSourceImageFolderId = track.cueSourceImageFolderId,
                )
            },
            rebuildDerivedIndexes = rebuildDerivedIndexes,
            executionProfile = V2IndexingExecutionProfile.valueOf(executionProfile),
            nowEpochMs = jobCreatedAtEpochMs,
        )

    private fun requireSameRequest(
        existing: V2IndexingPreflightIntent,
        expected: V2IndexingPreflightIntent,
    ) {
        require(V2IndexingPreflightRequestFingerprint.compute(existing) ==
            V2IndexingPreflightRequestFingerprint.compute(expected)
        ) { "job ID already exists with different immutable request evidence" }
    }

    private fun requirePointerMatches(jobId: String) {
        val pointer = V2ActiveIndexingJobPointer(filesDir).inspect()
        require(pointer is V2ActiveIndexingJobPointerInspection.Readable &&
            pointer.jobId == jobId
        ) { "durable request exists without its exact active-job pointer" }
    }
}

private class AcceptancePrivatePaths(
    filesDir: File,
    inputRelativePath: String,
    outputRelativePath: String,
) {
    private val root = File(filesDir, ACCEPTANCE_ROOT).canonicalFile
    private val input = resolve(inputRelativePath)
    private val output = resolve(outputRelativePath)

    init {
        require(input != output) { "acceptance input and output paths must differ" }
    }

    fun readVerifiedInput(expectedSha256: String): String {
        require(SHA256.matches(expectedSha256)) { "invalid expected input SHA-256" }
        require(input.isFile && input.canRead() && input.length() in 1..MAX_INPUT_BYTES) {
            "acceptance input is missing, unreadable, empty, or too large"
        }
        require(sha256(input) == expectedSha256) { "acceptance input SHA-256 mismatch" }
        return input.readText(StandardCharsets.UTF_8)
    }

    fun writeOutput(json: String) {
        require(json.toByteArray(StandardCharsets.UTF_8).size <= MAX_OUTPUT_BYTES) {
            "acceptance output is too large"
        }
        output.parentFile?.let { parent ->
            require(parent.isDirectory || parent.mkdirs()) { "cannot create acceptance output directory" }
        }
        val atomic = AtomicFile(output)
        val stream = atomic.startWrite()
        try {
            val writer = OutputStreamWriter(stream, StandardCharsets.UTF_8)
            writer.write(json)
            writer.write("\n")
            writer.flush()
            atomic.finishWrite(stream)
        } catch (error: Throwable) {
            atomic.failWrite(stream)
            throw error
        }
    }

    private fun resolve(relativePath: String): File {
        require(SAFE_RELATIVE_PATH.matches(relativePath) && ".." !in relativePath.split('/')) {
            "unsafe acceptance relative path"
        }
        val resolved = File(root, relativePath).canonicalFile
        require(resolved.path.startsWith(root.path + File.separator)) {
            "acceptance path escapes its private root"
        }
        return resolved
    }
}

private object AcceptanceJson {
    fun parseRequest(json: String): AcceptanceCandidateRequest {
        val root = parseObject(json)
        requireExactKeys(root, REQUEST_KEYS, "candidate request")
        return GSON.fromJson(root, AcceptanceCandidateRequest::class.java)
            ?: error("candidate request is empty")
    }

    fun parseManifest(json: String): AcceptanceCandidateManifest {
        val root = parseObject(json)
        requireExactKeys(root, MANIFEST_KEYS, "candidate manifest")
        val tracks = root.getAsJsonArray("tracks") ?: error("candidate manifest has no tracks")
        tracks.forEachIndexed { index, element ->
            require(element.isJsonObject) { "candidate track $index is not an object" }
            requireExactKeys(element.asJsonObject, TRACK_KEYS, "candidate track $index")
        }
        return GSON.fromJson(root, AcceptanceCandidateManifest::class.java)
            ?: error("candidate manifest is empty")
    }

    private fun parseObject(json: String): JsonObject {
        val element = JsonParser.parseString(json)
        require(element.isJsonObject) { "acceptance input must be one JSON object" }
        return element.asJsonObject
    }

    private fun requireExactKeys(value: JsonObject, expected: Set<String>, label: String) {
        val actual = value.keySet()
        require(actual == expected) {
            "$label keys differ: missing=${expected - actual}, extra=${actual - expected}"
        }
    }
}

private object AcceptanceValidation {
    fun requireRequest(request: AcceptanceCandidateRequest) {
        require(request.format == REQUEST_FORMAT && request.schemaVersion == SCHEMA_VERSION) {
            "unsupported candidate request schema"
        }
        require(SAFE_RUN_ID.matches(request.runId)) { "unsafe run ID" }
        require(SAFE_JOB_ID.matches(request.jobId)) { "unsafe job ID" }
        require(request.jobCreatedAtEpochMs > 0L) { "invalid job creation time" }
        require(request.purpose in ACCEPTED_PURPOSES) { "unsupported acceptance purpose" }
        val selectionMode = runCatching {
            AcceptanceSelectionMode.valueOf(request.selectionMode)
        }.getOrNull() ?: error("unsupported candidate selection mode")
        require(request.powerampFileIds.all { it > 0L } &&
            request.powerampFileIds.distinct().size == request.powerampFileIds.size
        ) { "candidate IDs must be unique positive IDs" }
        when (selectionMode) {
            AcceptanceSelectionMode.EXPLICIT_IDS -> require(
                request.powerampFileIds.size in 1..MAX_TRACKS && request.readyCap == null,
            ) { "explicit selection requires 1..$MAX_TRACKS IDs and no ready cap" }

            AcceptanceSelectionMode.ALL_READY -> require(
                request.powerampFileIds.isEmpty() && request.readyCap == null,
            ) { "all-ready selection cannot carry explicit IDs or a cap" }

            AcceptanceSelectionMode.READY_CAP -> require(
                request.powerampFileIds.isEmpty() &&
                    (request.readyCap ?: 0) in 1..MAX_TRACKS,
            ) { "capped-ready selection requires a 1..$MAX_TRACKS cap and no explicit IDs" }
        }
        require(runCatching {
            V2IndexingExecutionProfile.valueOf(request.executionProfile)
        }.isSuccess) { "unsupported execution profile" }
        require(request.rebuildDerivedIndexes) {
            "production acceptance requires rebuilding derived indexes"
        }
        require(request.minimumUsableBytes >= MINIMUM_STORAGE_FLOOR_BYTES) {
            "minimum usable storage is below the acceptance safety floor"
        }
    }

    fun requireManifest(manifest: AcceptanceCandidateManifest) {
        require(manifest.format == MANIFEST_FORMAT && manifest.schemaVersion == SCHEMA_VERSION) {
            "unsupported candidate manifest schema"
        }
        requireRequest(
            AcceptanceCandidateRequest(
                runId = manifest.runId,
                purpose = manifest.purpose,
                jobId = manifest.jobId,
                jobCreatedAtEpochMs = manifest.jobCreatedAtEpochMs,
                powerampFileIds = if (
                    manifest.selectionMode == AcceptanceSelectionMode.EXPLICIT_IDS.name
                ) {
                    manifest.tracks.map(AcceptanceCandidateTrack::powerampFileId)
                } else {
                    emptyList()
                },
                selectionMode = manifest.selectionMode,
                readyCap = manifest.readyCap,
                executionProfile = manifest.executionProfile,
                rebuildDerivedIndexes = manifest.rebuildDerivedIndexes,
                minimumUsableBytes = manifest.minimumUsableBytes,
            ),
        )
        require(manifest.capturedAtEpochMs >= manifest.jobCreatedAtEpochMs) {
            "manifest capture predates its job"
        }
        require(manifest.applicationId == "com.powerampstartradio.v2") {
            "candidate manifest was not produced by V2"
        }
        listOf(
            manifest.installedApkSha256,
            manifest.activeManifestSha256,
        ).forEach { require(SHA256.matches(it)) { "manifest contains an invalid SHA-256" } }
        require(manifest.activeGenerationId.isNotBlank() &&
            manifest.providerGenerationId.isNotBlank() &&
            manifest.receiptEmbeddingSpecId.isNotBlank() &&
            manifest.textRetrievalSpecId.isNotBlank()
        ) { "manifest identity evidence is incomplete" }
        require(manifest.cuePolicy == CUE_POLICY &&
            manifest.missingSourcePolicy == MISSING_SOURCE_POLICY
        ) { "manifest safety policies differ" }
        val trackIds = manifest.tracks.map(AcceptanceCandidateTrack::powerampFileId)
        require(trackIds.distinct().size == trackIds.size && trackIds == trackIds.sorted()) {
            "candidate manifest track IDs must be unique and ascending"
        }
        when (AcceptanceSelectionMode.valueOf(manifest.selectionMode)) {
            AcceptanceSelectionMode.EXPLICIT_IDS -> require(
                manifest.selectionPolicy == EXPLICIT_SELECTION_POLICY &&
                    manifest.discoveredReadyPowerampFileIds == null &&
                    manifest.discoveredReadyFingerprint == null,
            ) { "explicit manifest contains automatic readiness evidence" }

            AcceptanceSelectionMode.ALL_READY,
            AcceptanceSelectionMode.READY_CAP,
            -> {
                val readyIds = requireNotNull(manifest.discoveredReadyPowerampFileIds) {
                    "automatic manifest omitted the discovered ready universe"
                }
                require(readyIds.size <= MAX_READY_UNIVERSE && readyIds.all { it > 0L } &&
                    readyIds.distinct().size == readyIds.size && readyIds == readyIds.sorted() &&
                    manifest.selectionPolicy == AUTOMATIC_READY_SELECTION_POLICY &&
                    SHA256.matches(requireNotNull(manifest.discoveredReadyFingerprint))
                ) { "automatic readiness evidence is invalid" }
                val expectedSelected = when (
                    AcceptanceSelectionMode.valueOf(manifest.selectionMode)
                ) {
                    AcceptanceSelectionMode.ALL_READY -> readyIds
                    AcceptanceSelectionMode.READY_CAP -> readyIds.take(
                        requireNotNull(manifest.readyCap),
                    )
                    AcceptanceSelectionMode.EXPLICIT_IDS -> error("unreachable explicit mode")
                }
                require(trackIds == expectedSelected) {
                    "manifest tracks differ from the frozen automatic selection"
                }
                require(trackIds.size <= MAX_TRACKS) {
                    "automatic selected cohort exceeds the execution limit"
                }
            }
        }
        manifest.tracks.forEach { track ->
            require(track.powerampFileId > 0L) { "invalid candidate Poweramp ID" }
            if (track.decision == AcceptanceCandidateDecision.READY) {
                require(!track.isCueGroup()) { "CUE candidate cannot be ready" }
                require(!track.providerPhysicalPath.isNullOrBlank() &&
                    requireNotNull(track.providerPhysicalPath).startsWith('/') &&
                    requireNotNull(track.durationMs) >= 0L &&
                    requireNotNull(track.offsetMs) == 0L &&
                    requireNotNull(track.sourceByteLength) > 0L &&
                    requireNotNull(track.sourceModifiedEpochMs) >= 0L &&
                    SHA256.matches(requireNotNull(track.sourceSha256)) &&
                    !track.containerMime.isNullOrBlank() &&
                    requireNotNull(track.containerSampleRateHz) > 0 &&
                    requireNotNull(track.containerChannelCount) > 0
                ) { "ready candidate ${track.powerampFileId} lacks pinned source evidence" }
            }
        }
    }
}

private data class AcceptanceCandidateRequest(
    val format: String = REQUEST_FORMAT,
    val schemaVersion: Int = SCHEMA_VERSION,
    val runId: String,
    val purpose: String,
    val jobId: String,
    val jobCreatedAtEpochMs: Long,
    val powerampFileIds: List<Long>,
    val selectionMode: String,
    val readyCap: Int?,
    val executionProfile: String,
    val rebuildDerivedIndexes: Boolean,
    val minimumUsableBytes: Long,
)

private data class AcceptanceCandidateManifest(
    val format: String = MANIFEST_FORMAT,
    val schemaVersion: Int = SCHEMA_VERSION,
    val runId: String,
    val purpose: String,
    val jobId: String,
    val jobCreatedAtEpochMs: Long,
    val capturedAtEpochMs: Long,
    val applicationId: String,
    val installedApkSha256: String,
    val activeGenerationId: String,
    val activeManifestSha256: String,
    val providerGenerationId: String,
    val receiptEmbeddingSpecId: String,
    val textRetrievalSpecId: String,
    val selectionMode: String,
    val readyCap: Int?,
    val selectionPolicy: String,
    val discoveredReadyPowerampFileIds: List<Long>?,
    val discoveredReadyFingerprint: String?,
    val executionProfile: String,
    val rebuildDerivedIndexes: Boolean,
    val minimumUsableBytes: Long,
    val cuePolicy: String,
    val missingSourcePolicy: String,
    val tracks: List<AcceptanceCandidateTrack>,
)

private data class AcceptanceResolvedSelection(
    val selectedPowerampFileIds: List<Long>,
    val discoveredReadyPowerampFileIds: List<Long>?,
    val discoveredReadyFingerprint: String?,
    val selectionPolicy: String,
)

private enum class AcceptanceSelectionMode {
    EXPLICIT_IDS,
    ALL_READY,
    READY_CAP,
}

private data class AcceptanceCandidateTrack(
    val powerampFileId: Long,
    val artist: String?,
    val album: String?,
    val title: String?,
    val providerPhysicalPath: String?,
    val durationMs: Long?,
    val offsetMs: Long?,
    val cueSourceImageFolderId: Long?,
    val sourceReferenceCount: Int?,
    val sourceHasLogicalOffsets: Boolean?,
    val sourceHasCueImageRow: Boolean?,
    val detectionKind: String?,
    val decision: AcceptanceCandidateDecision,
    val blocker: String?,
    val sourceByteLength: Long?,
    val sourceModifiedEpochMs: Long?,
    val sourceSha256: String?,
    val containerMime: String?,
    val containerDurationUsEstimate: Long?,
    val containerSampleRateHz: Int?,
    val containerChannelCount: Int?,
) {
    fun isCueGroup(): Boolean = offsetMs != null && offsetMs > 0L ||
        cueSourceImageFolderId != null ||
        sourceHasLogicalOffsets == true ||
        sourceHasCueImageRow == true

    companion object {
        fun missingProvider(powerampFileId: Long) = AcceptanceCandidateTrack(
            powerampFileId = powerampFileId,
            artist = null,
            album = null,
            title = null,
            providerPhysicalPath = null,
            durationMs = null,
            offsetMs = null,
            cueSourceImageFolderId = null,
            sourceReferenceCount = null,
            sourceHasLogicalOffsets = null,
            sourceHasCueImageRow = null,
            detectionKind = null,
            decision = AcceptanceCandidateDecision.BLOCKED_PROVIDER_ROW_MISSING,
            blocker = "Poweramp provider does not contain the requested ID.",
            sourceByteLength = null,
            sourceModifiedEpochMs = null,
            sourceSha256 = null,
            containerMime = null,
            containerDurationUsEstimate = null,
            containerSampleRateHz = null,
            containerChannelCount = null,
        )
    }
}

private enum class AcceptanceCandidateDecision {
    READY,
    BLOCKED_PROVIDER_ROW_MISSING,
    BLOCKED_CUE_SUPERSESSION,
    BLOCKED_ALREADY_REPRESENTED,
    BLOCKED_NOT_READY,
    BLOCKED_SOURCE_OR_CONTAINER,
}

private data class AcceptanceSourceEvidence(
    val byteLength: Long,
    val modifiedEpochMs: Long,
    val sha256: String,
    val containerMime: String,
    val containerDurationUsEstimate: Long,
    val containerSampleRateHz: Int,
    val containerChannelCount: Int,
)

private enum class AcceptanceStartDisposition {
    SUBMITTED,
    SUBMITTED_LAUNCH_DEFERRED,
    ALREADY_SUBMITTED,
}

private data class AcceptanceStartResult(
    val format: String = START_RESULT_FORMAT,
    val schemaVersion: Int = SCHEMA_VERSION,
    val runId: String,
    val jobId: String,
    val result: AcceptanceStartDisposition,
    val capturedAtEpochMs: Long,
    val requestFingerprint: String,
    val durableState: String,
    val message: String,
)

private data class AcceptanceRuntimeSnapshot(
    val format: String = RUNTIME_SNAPSHOT_FORMAT,
    val schemaVersion: Int = SCHEMA_VERSION,
    val runId: String,
    val jobId: String,
    val capturedAtEpochMs: Long,
    val elapsedRealtimeMs: Long,
    val processId: Int,
    val activePointer: String,
    val activePointerJobId: String?,
    val processStateKind: String,
    val processStateJobId: String?,
    val serviceState: IndexingService.IndexingState,
    val notificationEvidence: Any?,
    val preflightIntent: V2IndexingPreflightIntent?,
    val ledger: Any?,
)

private data class AcceptanceCommandFailure(
    val format: String = FAILURE_FORMAT,
    val schemaVersion: Int = SCHEMA_VERSION,
    val command: String?,
    val capturedAtEpochMs: Long,
    val errorType: String,
    val message: String,
)

private fun sha256(file: File): String {
    val digest = MessageDigest.getInstance("SHA-256")
    FileInputStream(file).use { input ->
        val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
        while (true) {
            val read = input.read(buffer)
            if (read < 0) break
            if (read > 0) digest.update(buffer, 0, read)
        }
    }
    return digest.digest().joinToString("") { byte -> "%02x".format(byte.toInt() and 0xff) }
}

private fun readyUniverseFingerprint(
    readyIds: List<Long>,
    exclusionFingerprint: String,
    attentionFingerprint: String,
): String {
    require(readyIds == readyIds.sorted() && readyIds.distinct().size == readyIds.size)
    val digest = MessageDigest.getInstance("SHA-256")
    fun putString(value: String) {
        val bytes = value.toByteArray(StandardCharsets.UTF_8)
        digest.update(byteArrayOf(
            (bytes.size ushr 24).toByte(),
            (bytes.size ushr 16).toByte(),
            (bytes.size ushr 8).toByte(),
            bytes.size.toByte(),
        ))
        digest.update(bytes)
    }
    fun putLong(value: Long) {
        for (shift in 56 downTo 0 step 8) digest.update((value ushr shift).toByte())
    }
    putString(AUTOMATIC_READY_SELECTION_POLICY)
    putString(exclusionFingerprint)
    putString(attentionFingerprint)
    putLong(readyIds.size.toLong())
    readyIds.forEach(::putLong)
    return digest.digest().joinToString("") { byte -> "%02x".format(byte.toInt() and 0xff) }
}

private const val SCHEMA_VERSION = 2
private const val REQUEST_FORMAT = "poweramp-start-radio-v2-production-indexing-acceptance-request"
private const val MANIFEST_FORMAT = "poweramp-start-radio-v2-production-indexing-candidate-manifest"
private const val START_RESULT_FORMAT = "poweramp-start-radio-v2-production-indexing-start-result"
private const val RUNTIME_SNAPSHOT_FORMAT = "poweramp-start-radio-v2-production-indexing-runtime-snapshot"
private const val FAILURE_FORMAT = "poweramp-start-radio-v2-production-indexing-acceptance-failure"
private const val ACCEPTANCE_ROOT = "device_acceptance/indexing"
private const val MAX_INPUT_BYTES = 4L * 1024L * 1024L
private const val MAX_OUTPUT_BYTES = 32L * 1024L * 1024L
private const val MAX_TRACKS = 5_000
private const val MAX_READY_UNIVERSE = 100_000
private const val MINIMUM_STORAGE_FLOOR_BYTES = 4L * 1024L * 1024L * 1024L
private const val CUE_POLICY =
    "BLOCK_UNTIL_ATOMIC_IMPORTED_ROW_SUPERSESSION_IS_WIRED_TO_PRODUCTION_COMMIT"
private const val MISSING_SOURCE_POLICY =
    "NATURAL_PROVIDER_SOURCE_STATE_ONLY; NO_MISSING_SOURCE_INJECTION"
private const val EXPLICIT_SELECTION_POLICY = "EXACT_REQUESTED_POWERAMP_IDS_ASCENDING_V1"
private const val AUTOMATIC_READY_SELECTION_POLICY =
    "PRODUCTION_READY_MINUS_NEVER_IGNORED_FAILURE_AND_PREFLIGHT_ATTENTION;POWERAMP_ID_ASC_V1"
private val GSON: Gson = GsonBuilder().disableHtmlEscaping().serializeNulls().create()
private val SHA256 = Regex("^[0-9a-f]{64}$")
private val SAFE_RUN_ID = Regex("^[A-Za-z0-9._-]{1,80}$")
private val SAFE_JOB_ID = Regex("^[A-Za-z0-9._-]{1,128}$")
private val SAFE_RELATIVE_PATH = Regex("^[A-Za-z0-9._/-]{1,240}$")
private val ACCEPTED_PURPOSES = setOf("REBOOT", "OVERNIGHT")
private val REQUEST_KEYS = setOf(
    "format",
    "schemaVersion",
    "runId",
    "purpose",
    "jobId",
    "jobCreatedAtEpochMs",
    "powerampFileIds",
    "selectionMode",
    "readyCap",
    "executionProfile",
    "rebuildDerivedIndexes",
    "minimumUsableBytes",
)
private val MANIFEST_KEYS = setOf(
    "format",
    "schemaVersion",
    "runId",
    "purpose",
    "jobId",
    "jobCreatedAtEpochMs",
    "capturedAtEpochMs",
    "applicationId",
    "installedApkSha256",
    "activeGenerationId",
    "activeManifestSha256",
    "providerGenerationId",
    "receiptEmbeddingSpecId",
    "textRetrievalSpecId",
    "selectionMode",
    "readyCap",
    "selectionPolicy",
    "discoveredReadyPowerampFileIds",
    "discoveredReadyFingerprint",
    "executionProfile",
    "rebuildDerivedIndexes",
    "minimumUsableBytes",
    "cuePolicy",
    "missingSourcePolicy",
    "tracks",
)
private val TRACK_KEYS = setOf(
    "powerampFileId",
    "artist",
    "album",
    "title",
    "providerPhysicalPath",
    "durationMs",
    "offsetMs",
    "cueSourceImageFolderId",
    "sourceReferenceCount",
    "sourceHasLogicalOffsets",
    "sourceHasCueImageRow",
    "detectionKind",
    "decision",
    "blocker",
    "sourceByteLength",
    "sourceModifiedEpochMs",
    "sourceSha256",
    "containerMime",
    "containerDurationUsEstimate",
    "containerSampleRateHz",
    "containerChannelCount",
)
