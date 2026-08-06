package com.powerampstartradio.debug

import android.app.ActivityManager
import android.content.Context
import android.os.Build
import android.os.Bundle
import android.os.PowerManager
import android.os.SystemClock
import android.util.AtomicFile
import android.util.Base64
import android.util.Log
import android.view.Gravity
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.lifecycle.lifecycleScope
import com.google.ai.edge.litert.Accelerator
import com.google.gson.GsonBuilder
import com.powerampstartradio.data.EmbeddedTrack
import com.powerampstartradio.data.EmbeddingDatabase
import com.powerampstartradio.data.EmbeddingIndex
import com.powerampstartradio.data.StableTrackIdentityCatalog
import com.powerampstartradio.indexing.Clamp3TextInference
import com.powerampstartradio.indexing.V2ActiveLibraryCatalog
import com.powerampstartradio.indexing.V2ActiveLibraryCatalogLoader
import com.powerampstartradio.indexing.v2.V2IndexingModelResolver
import com.powerampstartradio.indexing.v2.V2LibraryDatabaseResolver
import com.powerampstartradio.indexing.v2.V2PowerampProviderSnapshotAcquirer
import com.powerampstartradio.poweramp.PowerampFileEntry
import com.powerampstartradio.poweramp.PowerampLibrarySnapshot
import com.powerampstartradio.poweramp.TrackMatcher
import com.powerampstartradio.poweramp.TrackNormalization
import com.powerampstartradio.similarity.GraphExplorationEvidence
import com.powerampstartradio.similarity.DppSelectionEvidence
import com.powerampstartradio.similarity.RecommendationAssetFiles
import com.powerampstartradio.similarity.RecommendationEngine
import com.powerampstartradio.similarity.SimilarTrack
import com.powerampstartradio.similarity.StableSimilarityTopK
import com.powerampstartradio.similarity.StableVisibleResultReducer
import com.powerampstartradio.similarity.UniformShuffleIdentityEvidence
import com.powerampstartradio.similarity.algorithms.MmrSelectionEvidence
import com.powerampstartradio.ui.RadioConfig
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.io.OutputStreamWriter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import java.time.Instant
import java.util.UUID
import kotlin.math.sqrt

/**
 * Debug-only, read-only execution surface for empirical feature acceptance on the real phone.
 *
 * It calls the production retrieval/selection components and resolves their results against one
 * complete Poweramp provider snapshot. It deliberately imports no queue API and never starts the
 * radio service. The host wrapper independently proves that the queue stayed byte-for-byte equal.
 */
class FeatureAcceptanceActivity : ComponentActivity() {
    private lateinit var statusView: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        statusView = TextView(this).apply {
            gravity = Gravity.CENTER
            textSize = 16f
            setPadding(48, 48, 48, 48)
            text = "Preparing read-only feature acceptance..."
        }
        setContentView(statusView)

        if (savedInstanceState == null) {
            lifecycleScope.launch {
                val request = AcceptanceRequest.from(intent.extras)
                val finalStatus = withContext(Dispatchers.IO) {
                    FeatureAcceptanceRunner(applicationContext, filesDir).run(request) { message ->
                        runOnUiThread { statusView.text = message }
                    }
                }
                statusView.text = finalStatus
            }
        }
    }
}

private data class AcceptanceRequest(
    val runId: String,
    val seedTrackIds: List<Long>,
    val queries: List<String>,
    val numTracks: Int,
    val repeatCount: Int,
    val includeSelection: Boolean,
    val includeText: Boolean,
    val selectionMatrixId: String,
) {
    companion object {
        fun from(extras: Bundle?): AcceptanceRequest {
            val seedIds = extras?.getString("seed_track_ids")
                ?.split(',')
                ?.mapNotNull { it.trim().toLongOrNull()?.takeIf { id -> id > 0L } }
                ?.distinct()
                ?.takeIf(List<Long>::isNotEmpty)
                ?: error("seed_track_ids must contain at least one positive track ID")
            val encodedQueries = extras?.getString("queries_b64")
                ?.let { encoded ->
                    runCatching {
                        String(Base64.decode(encoded, Base64.NO_WRAP), Charsets.UTF_8)
                    }.getOrNull()
                }
            val queries = (encodedQueries ?: extras?.getString("queries"))
                ?.split('|')
                ?.map(String::trim)
                ?.filter(String::isNotBlank)
                ?.distinct()
                ?.takeIf(List<String>::isNotEmpty)
                ?: listOf("ambient", "sleep", "guitar", "psychedelic")
            val suite = extras?.getString("suite")?.trim()?.lowercase() ?: "all"
            return AcceptanceRequest(
                runId = extras?.getString("run_id")?.trim()?.takeIf(String::isNotBlank)
                    ?: UUID.randomUUID().toString(),
                seedTrackIds = seedIds,
                queries = queries,
                numTracks = (extras?.getInt("num_tracks", 30) ?: 30).coerceIn(1, 200),
                repeatCount = (extras?.getInt("repeat_count", 2) ?: 2).coerceIn(1, 5),
                includeSelection = suite == "all" || suite == "selection",
                includeText = suite == "all" || suite == "text",
                selectionMatrixId = extras?.getString("selection_matrix")
                    ?.trim()
                    ?.lowercase()
                    ?.takeIf(String::isNotBlank)
                    ?: CurrentV2SelectionAcceptanceMatrix.DEFAULTS_EXTREMES_ID,
            )
        }
    }
}

private data class AcceptanceGeneration(
    val generationId: String,
    val activationBindingId: String,
    val manifestSha256: String,
    val databaseSha256: String,
    val databaseContentSha256: String,
    val orderedTrackSetSha256: String,
    val embeddingSha256: String,
    val graphSha256: String?,
    val trackCount: Int,
    val embeddingDimension: Int,
)

private data class AcceptanceDevice(
    val manufacturer: String,
    val model: String,
    val socModel: String,
    val androidRelease: String,
    val sdk: Int,
    val processors: Int,
    val totalMemoryBytes: Long,
    val thermalStatusAtStart: Int,
)

private data class AcceptanceProviderSnapshot(
    val generationId: String,
    val rowCount: Int,
    val physicalSourceCount: Int,
    val queryAndCursorReadMs: Long?,
    val assemblyMs: Long?,
)

private data class AcceptanceActiveCatalog(
    val buildMs: Long,
    val activeTrackCount: Int,
    val quarantinedTrackCount: Int,
    val unboundProviderCount: Int,
    val bindingEvidenceCounts: Map<String, Int>,
    val quarantineReasonCounts: Map<String, Int>,
    val quarantinedTrackIds: List<Long>,
    val unboundPowerampFileIds: List<Long>,
    val completeTsvFile: String,
    val completeTsvSha256: String,
)

private data class AcceptanceTrack(
    val rank: Int,
    val trackId: Long,
    val artist: String?,
    val album: String?,
    val title: String?,
    val durationMs: Int,
    val filePath: String,
    val score: Float,
    val similarityToSeed: Float,
    val candidateRank: Int?,
    val seedRank: Int?,
    val driftRank: Int?,
    val graphTerminalProbability: Double?,
    val graphExpectedRouteLinks: Double?,
    /** Exact generation/provider catalog resolution; this is the truthful V2 queue target. */
    val catalogPowerampFileId: Long?,
    /** What the current production fuzzy/path matcher would resolve for comparison. */
    val matcherPowerampFileId: Long?,
    val activeInCurrentLibrary: Boolean,
    val quarantineReason: String?,
    val matcherAgreesWithCatalog: Boolean,
)

private data class AcceptanceSelectionRun(
    val caseId: String,
    val repeat: Int,
    val seedTrackId: Long,
    val seed: EmbeddedTrack?,
    val config: RadioConfig,
    val elapsedMs: Long,
    val progress: List<String>,
    val resultFingerprint: String?,
    val resolvedCount: Int,
    val matcherResolvedCount: Int,
    val inactiveResultCount: Int,
    val matcherDisagreementCount: Int,
    val duplicatePowerampResolutionCount: Int,
    val graphEvidence: GraphExplorationEvidence?,
    val dppSelectionEvidence: DppSelectionEvidence?,
    val shuffleIdentityEvidence: UniformShuffleIdentityEvidence?,
    val mmrSelectionEvidence: List<MmrSelectionEvidence>,
    val tracks: List<AcceptanceTrack>,
    val error: String? = null,
)

private data class AcceptanceTextRun(
    val query: String,
    val repeat: Int,
    val inferenceMs: Long,
    val rankingMs: Long,
    val embeddingDimension: Int,
    val embeddingNorm: Float,
    val embeddingSha256: String?,
    val embedding: List<Float>,
    val resultFingerprint: String?,
    val resolvedCount: Int,
    val matcherResolvedCount: Int,
    val inactiveResultCount: Int,
    val matcherDisagreementCount: Int,
    val tracks: List<AcceptanceTrack>,
    val error: String? = null,
)

private data class FeatureAcceptanceReport(
    val schemaVersion: Int = 1,
    val state: String,
    val runId: String,
    val startedAt: String,
    val updatedAt: String,
    val completedAt: String? = null,
    val request: AcceptanceRequest,
    val queueMutationApisCalled: Int = 0,
    val device: AcceptanceDevice? = null,
    val generation: AcceptanceGeneration? = null,
    val providerSnapshot: AcceptanceProviderSnapshot? = null,
    val activeCatalog: AcceptanceActiveCatalog? = null,
    val indexPreparationMs: Long? = null,
    val plannedSelectionCases: List<FeatureSelectionCase> = emptyList(),
    val plannedSelectionRunCount: Int = 0,
    val selectionRuns: List<AcceptanceSelectionRun> = emptyList(),
    val textRuns: List<AcceptanceTextRun> = emptyList(),
    val fatalError: String? = null,
)

private class FeatureAcceptanceRunner(
    private val context: Context,
    private val filesDir: File,
) {
    companion object {
        private const val TAG = "FeatureAcceptance"
        private val gson = GsonBuilder().setPrettyPrinting().disableHtmlEscaping().create()
    }

    private val outputDir = File(filesDir, "feature_acceptance")
    private val statusFile = AtomicFile(File(outputDir, "status.json"))

    suspend fun run(request: AcceptanceRequest, onStatus: (String) -> Unit): String {
        require(outputDir.isDirectory || outputDir.mkdirs()) {
            "Cannot create ${outputDir.absolutePath}"
        }
        val startedAt = Instant.now().toString()
        var report = FeatureAcceptanceReport(
            state = "RUNNING",
            runId = request.runId,
            startedAt = startedAt,
            updatedAt = startedAt,
            request = request,
        )
        writeCheckpoint(report)

        try {
            val selectionCases = if (request.includeSelection) {
                CurrentV2SelectionAcceptanceMatrix.cases(
                    matrixId = request.selectionMatrixId,
                    numTracks = request.numTracks,
                )
            } else {
                emptyList()
            }
            report = report.copy(
                updatedAt = Instant.now().toString(),
                plannedSelectionCases = selectionCases,
                plannedSelectionRunCount = selectionCases.size *
                    request.seedTrackIds.size * request.repeatCount,
            )
            writeCheckpoint(report)

            onStatus("Reading the published index and complete Poweramp library...")
            val device = captureDevice()
            val active = V2LibraryDatabaseResolver.requirePublished(filesDir)
            val provider = V2PowerampProviderSnapshotAcquirer(context)
                .acquireBlocking()
            val providerRows = provider.groups.flatMap { it.rows }
            val powerampSnapshot = provider.toMatcherSnapshot()
            val catalogStarted = SystemClock.elapsedRealtimeNanos()
            val activeCatalog = V2ActiveLibraryCatalogLoader.load(active, provider)
            val catalogBuildMs = elapsedMs(catalogStarted)
            val catalogDryRun = activeCatalog.dryRunReport()
            val catalogTsv = catalogDryRun.renderDeterministicTsv()
            val catalogTsvName = "${request.runId}-active-catalog.tsv"
            writeAtomic(File(outputDir, catalogTsvName), catalogTsv)
            val catalogEvidence = AcceptanceActiveCatalog(
                buildMs = catalogBuildMs,
                activeTrackCount = catalogDryRun.activeTrackCount,
                quarantinedTrackCount = catalogDryRun.quarantinedTrackCount,
                unboundProviderCount = catalogDryRun.unboundProviderCount,
                bindingEvidenceCounts = catalogDryRun.bindingEvidenceCounts
                    .mapKeys { it.key.name },
                quarantineReasonCounts = catalogDryRun.quarantinedTracks
                    .groupingBy { it.reason.name }
                    .eachCount(),
                quarantinedTrackIds = catalogDryRun.quarantinedTracks.map { it.trackId },
                unboundPowerampFileIds = catalogDryRun.unboundPowerampFileIds,
                completeTsvFile = catalogTsvName,
                completeTsvSha256 = textSha256(catalogTsv),
            )
            val generation = AcceptanceGeneration(
                generationId = active.manifest.generationId,
                activationBindingId = active.manifest.activationBindingId,
                manifestSha256 = active.manifestSha256,
                databaseSha256 = active.manifest.databaseSha256,
                databaseContentSha256 = active.manifest.databaseContentSha256,
                orderedTrackSetSha256 = active.manifest.orderedTrackSetSha256,
                embeddingSha256 = active.manifest.embeddingSha256,
                graphSha256 = active.manifest.graph?.sha256,
                trackCount = active.manifest.trackCount,
                embeddingDimension = active.manifest.embeddingDimension,
            )
            val providerEvidence = AcceptanceProviderSnapshot(
                generationId = requireNotNull(provider.libraryGeneration),
                rowCount = providerRows.size,
                physicalSourceCount = provider.groups.size,
                queryAndCursorReadMs = provider.acquisitionEvidence?.queryAndCursorReadMs,
                assemblyMs = provider.acquisitionEvidence?.snapshotAssemblyMs,
            )
            report = report.copy(
                updatedAt = Instant.now().toString(),
                device = device,
                generation = generation,
                providerSnapshot = providerEvidence,
                activeCatalog = catalogEvidence,
            )
            writeCheckpoint(report)

            val database = EmbeddingDatabase.open(active.databaseFile)
            try {
                val index = EmbeddingIndex.mmap(active.embeddingFile)
                val identityCatalog = StableTrackIdentityCatalog.load(filesDir, database, index)
                val matcher = TrackMatcher(database)
                val engine = RecommendationEngine(
                    database = database,
                    filesDir = filesDir,
                    pinnedAssets = RecommendationAssetFiles(active.embeddingFile, active.graphFile),
                    activeCatalog = activeCatalog,
                )
                val prepareStarted = SystemClock.elapsedRealtimeNanos()
                engine.ensureIndices()
                val preparationMs = elapsedMs(prepareStarted)
                report = report.copy(
                    updatedAt = Instant.now().toString(),
                    indexPreparationMs = preparationMs,
                )
                writeCheckpoint(report)

                if (request.includeSelection) {
                    for (seedTrackId in request.seedTrackIds) {
                        for (selectionCase in selectionCases) {
                            for (repeat in 1..request.repeatCount) {
                                onStatus(
                                    "Selection ${selectionCase.id}, seed $seedTrackId, " +
                                        "repeat $repeat/${request.repeatCount}",
                                )
                                val run = runSelection(
                                    selectionCase,
                                    repeat,
                                    seedTrackId,
                                    database,
                                    engine,
                                    matcher,
                                    powerampSnapshot,
                                    activeCatalog,
                                )
                                report = report.copy(
                                    updatedAt = Instant.now().toString(),
                                    selectionRuns = report.selectionRuns + run,
                                )
                                writeCheckpoint(report)
                            }
                        }
                    }
                }

                if (request.includeText) {
                    val textAssets = V2IndexingModelResolver(filesDir)
                        .resolveText(active.manifest.textRetrievalSpec)
                    val inference = Clamp3TextInference(
                        modelFile = textAssets.clamp3Text,
                        tokenizerModelFile = textAssets.sentencePieceModel,
                        accelerator = Accelerator.CPU,
                        strictAccelerator = true,
                    )
                    try {
                        for (query in request.queries) {
                            for (repeat in 1..request.repeatCount) {
                                onStatus(
                                    "Text: $query, repeat $repeat/${request.repeatCount}",
                                )
                                val run = runText(
                                    query,
                                    repeat,
                                    request.numTracks,
                                    inference,
                                    index,
                                    database,
                                    identityCatalog,
                                    matcher,
                                    powerampSnapshot,
                                    activeCatalog,
                                )
                                report = report.copy(
                                    updatedAt = Instant.now().toString(),
                                    textRuns = report.textRuns + run,
                                )
                                writeCheckpoint(report)
                            }
                        }
                    } finally {
                        inference.close()
                    }
                }
            } finally {
                database.close()
            }

            val completedAt = Instant.now().toString()
            report = report.copy(
                state = "COMPLETE",
                updatedAt = completedAt,
                completedAt = completedAt,
            )
            writeCheckpoint(report)
            writeAtomic(File(outputDir, "${request.runId}.json"), gson.toJson(report))
            return "Complete. ${report.selectionRuns.size} selection runs and " +
                "${report.textRuns.size} text runs recorded."
        } catch (cancelled: CancellationException) {
            throw cancelled
        } catch (failure: Throwable) {
            Log.e(TAG, "Acceptance run ${request.runId} failed", failure)
            val failedAt = Instant.now().toString()
            report = report.copy(
                state = "FAILED",
                updatedAt = failedAt,
                completedAt = failedAt,
                fatalError = failure.stackTraceToString(),
            )
            writeCheckpoint(report)
            writeAtomic(File(outputDir, "${request.runId}.json"), gson.toJson(report))
            return "FAILED: ${failure.message ?: failure.javaClass.simpleName}"
        }
    }

    private suspend fun runSelection(
        selectionCase: FeatureSelectionCase,
        repeat: Int,
        seedTrackId: Long,
        database: EmbeddingDatabase,
        engine: RecommendationEngine,
        matcher: TrackMatcher,
        powerampSnapshot: PowerampLibrarySnapshot,
        activeCatalog: V2ActiveLibraryCatalog,
    ): AcceptanceSelectionRun {
        val progress = mutableListOf<String>()
        var graphEvidence: GraphExplorationEvidence? = null
        var dppEvidence: DppSelectionEvidence? = null
        var shuffleEvidence: UniformShuffleIdentityEvidence? = null
        val mmrEvidence = mutableListOf<MmrSelectionEvidence>()
        val started = SystemClock.elapsedRealtimeNanos()
        return try {
            val similar = engine.generatePlaylist(
                seedTrackId = seedTrackId,
                config = selectionCase.config,
                onProgress = progress::add,
                onGraphExplorationEvidence = { graphEvidence = it },
                onDppSelectionEvidence = { dppEvidence = it },
                onUniformShuffleIdentityEvidence = { shuffleEvidence = it },
                onMmrSelectionEvidence = mmrEvidence::add,
            )
            val elapsed = elapsedMs(started)
            val matcherFileIds = matcher.findFileIds(
                powerampSnapshot,
                similar.map { it.track },
            )
            val tracks = similar.toAcceptanceTracks(matcherFileIds, activeCatalog)
            AcceptanceSelectionRun(
                caseId = selectionCase.id,
                repeat = repeat,
                seedTrackId = seedTrackId,
                seed = database.getTrackById(seedTrackId),
                config = selectionCase.config,
                elapsedMs = elapsed,
                progress = progress,
                resultFingerprint = fingerprint(tracks),
                resolvedCount = tracks.count { it.catalogPowerampFileId != null },
                matcherResolvedCount = tracks.count { it.matcherPowerampFileId != null },
                inactiveResultCount = tracks.count { !it.activeInCurrentLibrary },
                matcherDisagreementCount = tracks.count { !it.matcherAgreesWithCatalog },
                duplicatePowerampResolutionCount = tracks.mapNotNull {
                    it.catalogPowerampFileId
                }.let { it.size - it.distinct().size },
                graphEvidence = graphEvidence,
                dppSelectionEvidence = dppEvidence,
                shuffleIdentityEvidence = shuffleEvidence,
                mmrSelectionEvidence = mmrEvidence,
                tracks = tracks,
            )
        } catch (failure: Throwable) {
            AcceptanceSelectionRun(
                caseId = selectionCase.id,
                repeat = repeat,
                seedTrackId = seedTrackId,
                seed = database.getTrackById(seedTrackId),
                config = selectionCase.config,
                elapsedMs = elapsedMs(started),
                progress = progress,
                resultFingerprint = null,
                resolvedCount = 0,
                matcherResolvedCount = 0,
                inactiveResultCount = 0,
                matcherDisagreementCount = 0,
                duplicatePowerampResolutionCount = 0,
                graphEvidence = graphEvidence,
                dppSelectionEvidence = dppEvidence,
                shuffleIdentityEvidence = shuffleEvidence,
                mmrSelectionEvidence = mmrEvidence,
                tracks = emptyList(),
                error = failure.stackTraceToString(),
            )
        }
    }

    private fun runText(
        query: String,
        repeat: Int,
        resultLimit: Int,
        inference: Clamp3TextInference,
        index: EmbeddingIndex,
        database: EmbeddingDatabase,
        identityCatalog: StableTrackIdentityCatalog,
        matcher: TrackMatcher,
        powerampSnapshot: PowerampLibrarySnapshot,
        activeCatalog: V2ActiveLibraryCatalog,
    ): AcceptanceTextRun {
        val inferenceStarted = SystemClock.elapsedRealtimeNanos()
        return try {
            val embedding = requireNotNull(
                inference.generateEmbedding(query, File(outputDir, "text_embeddings")),
            ) { "Text inference returned no embedding" }
            val inferenceMs = elapsedMs(inferenceStarted)
            val rankingStarted = SystemClock.elapsedRealtimeNanos()
            val similarities = index.computeAllSimilarities(embedding)
            val rankedRows = StableSimilarityTopK.select(
                orderedTrackIds = identityCatalog.orderedTrackIds(),
                similarities = similarities,
                topK = identityCatalog.rankedRowsForVisibleCount(resultLimit),
                rankingTieKey = identityCatalog::rankingTieKey,
            ).map { it.trackId to it.score }
            val tracksById = HashMap<Long, EmbeddedTrack?>()
            val reduction = StableVisibleResultReducer.reduce(
                rankedItems = rankedRows,
                requestedVisibleCount = resultLimit,
                identityOf = { (trackId, _) -> identityCatalog.visibleResultIdentity(trackId) },
                isEligible = { (trackId, _) ->
                    tracksById.getOrPut(trackId) { database.getTrackById(trackId) } != null
                },
            )
            val similar = reduction.items.mapNotNull { (trackId, score) ->
                (tracksById[trackId] ?: database.getTrackById(trackId))?.let { track ->
                    SimilarTrack(track, score, score)
                }
            }
            val rankingMs = elapsedMs(rankingStarted)
            val matcherFileIds = matcher.findFileIds(
                powerampSnapshot,
                similar.map { it.track },
            )
            val tracks = similar.toAcceptanceTracks(matcherFileIds, activeCatalog)
            AcceptanceTextRun(
                query = query,
                repeat = repeat,
                inferenceMs = inferenceMs,
                rankingMs = rankingMs,
                embeddingDimension = embedding.size,
                embeddingNorm = sqrt(embedding.sumOf { value ->
                    (value * value).toDouble()
                }).toFloat(),
                embeddingSha256 = floatArraySha256(embedding),
                embedding = embedding.toList(),
                resultFingerprint = fingerprint(tracks),
                resolvedCount = tracks.count { it.catalogPowerampFileId != null },
                matcherResolvedCount = tracks.count { it.matcherPowerampFileId != null },
                inactiveResultCount = tracks.count { !it.activeInCurrentLibrary },
                matcherDisagreementCount = tracks.count { !it.matcherAgreesWithCatalog },
                tracks = tracks,
            )
        } catch (failure: Throwable) {
            AcceptanceTextRun(
                query = query,
                repeat = repeat,
                inferenceMs = elapsedMs(inferenceStarted),
                rankingMs = 0,
                embeddingDimension = 0,
                embeddingNorm = 0f,
                embeddingSha256 = null,
                embedding = emptyList(),
                resultFingerprint = null,
                resolvedCount = 0,
                matcherResolvedCount = 0,
                inactiveResultCount = 0,
                matcherDisagreementCount = 0,
                tracks = emptyList(),
                error = failure.stackTraceToString(),
            )
        }
    }

    private fun List<SimilarTrack>.toAcceptanceTracks(
        matcherFileIds: List<Long?>,
        activeCatalog: V2ActiveLibraryCatalog,
    ): List<AcceptanceTrack> {
        require(size == matcherFileIds.size)
        val quarantineReasons = activeCatalog.quarantinedTracks.associate {
            it.trackId to it.reason.name
        }
        return mapIndexed { index, result ->
            val catalogFileId = activeCatalog.powerampFileIdForTrack(result.track.id)
            val matcherFileId = matcherFileIds[index]
            AcceptanceTrack(
                rank = index + 1,
                trackId = result.track.id,
                artist = result.track.artist,
                album = result.track.album,
                title = result.track.title,
                durationMs = result.track.durationMs,
                filePath = result.track.filePath,
                score = result.similarity,
                similarityToSeed = result.similarityToSeed,
                candidateRank = result.candidateRank,
                seedRank = result.seedRank,
                driftRank = result.driftRank,
                graphTerminalProbability = result.graphTerminalProbability,
                graphExpectedRouteLinks = result.graphExpectedRouteLinks,
                catalogPowerampFileId = catalogFileId,
                matcherPowerampFileId = matcherFileId,
                activeInCurrentLibrary = activeCatalog.containsActiveTrack(result.track.id),
                quarantineReason = quarantineReasons[result.track.id],
                matcherAgreesWithCatalog = matcherFileId == catalogFileId,
            )
        }
    }

    private fun captureDevice(): AcceptanceDevice {
        val memory = ActivityManager.MemoryInfo()
        context.getSystemService(ActivityManager::class.java).getMemoryInfo(memory)
        val thermal = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            context.getSystemService(PowerManager::class.java).currentThermalStatus
        } else {
            PowerManager.THERMAL_STATUS_NONE
        }
        return AcceptanceDevice(
            manufacturer = Build.MANUFACTURER,
            model = Build.MODEL,
            socModel = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) Build.SOC_MODEL else "",
            androidRelease = Build.VERSION.RELEASE,
            sdk = Build.VERSION.SDK_INT,
            processors = Runtime.getRuntime().availableProcessors(),
            totalMemoryBytes = memory.totalMem,
            thermalStatusAtStart = thermal,
        )
    }

    private fun com.powerampstartradio.indexing.v2.V2ProviderPathGroupSnapshot
        .toMatcherSnapshot(): PowerampLibrarySnapshot = PowerampLibrarySnapshot(
        entries = groups.flatMap { group ->
            group.rows.map { row ->
                val artist = TrackNormalization.normalizeArtist(row.artist)
                val album = TrackNormalization.normalizeAlbum(row.album)
                val title = TrackNormalization.normalizeTitle(row.title)
                val filename = File(row.providerPhysicalPath).name
                val durationMs = row.durationMs.coerceIn(0L, Int.MAX_VALUE.toLong()).toInt()
                PowerampFileEntry(
                    id = row.powerampFileId,
                    artist = artist,
                    album = album,
                    title = title,
                    durationMs = durationMs,
                    path = TrackNormalization.normalizePath(row.providerPhysicalPath),
                    offsetMs = row.offsetMs,
                    cueFolderId = row.cueSourceImageFolderId,
                    metadataKey = TrackNormalization.buildMetadataKey(
                        artist,
                        album,
                        title,
                        durationMs,
                    ),
                    filenameKeys = TrackNormalization.buildFilenameKeys(
                        artist,
                        title,
                        filename.substringBeforeLast('.', filename),
                    ),
                )
            }
        },
    )

    private fun writeCheckpoint(report: FeatureAcceptanceReport) {
        writeAtomic(statusFile, gson.toJson(report))
    }

    private fun writeAtomic(file: File, text: String) = writeAtomic(AtomicFile(file), text)

    private fun writeAtomic(file: AtomicFile, text: String) {
        var output: FileOutputStream? = null
        try {
            output = file.startWrite()
            val writer = OutputStreamWriter(output, Charsets.UTF_8)
            writer.write(text)
            writer.flush()
            file.finishWrite(output)
        } catch (failure: Throwable) {
            if (output != null) file.failWrite(output)
            throw failure
        }
    }

    private fun fingerprint(tracks: List<AcceptanceTrack>): String {
        val digest = MessageDigest.getInstance("SHA-256")
        tracks.forEach { track ->
            digest.update(longBytes(track.trackId))
            digest.update(intBytes(track.score.toRawBits()))
            digest.update(intBytes(track.similarityToSeed.toRawBits()))
            digest.update(longBytes(track.catalogPowerampFileId ?: -1L))
            digest.update(longBytes(track.matcherPowerampFileId ?: -1L))
        }
        return digest.digest().toHex()
    }

    private fun floatArraySha256(values: FloatArray): String {
        val buffer = ByteBuffer.allocate(values.size * Float.SIZE_BYTES)
            .order(ByteOrder.LITTLE_ENDIAN)
        values.forEach(buffer::putFloat)
        return MessageDigest.getInstance("SHA-256").digest(buffer.array()).toHex()
    }

    private fun textSha256(value: String): String = MessageDigest.getInstance("SHA-256")
        .digest(value.toByteArray(Charsets.UTF_8))
        .toHex()

    private fun intBytes(value: Int): ByteArray = ByteBuffer.allocate(Int.SIZE_BYTES)
        .order(ByteOrder.LITTLE_ENDIAN)
        .putInt(value)
        .array()

    private fun longBytes(value: Long): ByteArray = ByteBuffer.allocate(Long.SIZE_BYTES)
        .order(ByteOrder.LITTLE_ENDIAN)
        .putLong(value)
        .array()

    private fun ByteArray.toHex(): String = joinToString("") { "%02x".format(it) }

    private fun elapsedMs(startedNanos: Long): Long =
        (SystemClock.elapsedRealtimeNanos() - startedNanos) / 1_000_000L
}
