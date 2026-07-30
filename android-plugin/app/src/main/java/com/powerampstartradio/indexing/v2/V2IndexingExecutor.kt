package com.powerampstartradio.indexing.v2

import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.os.Process
import android.os.SystemClock
import com.google.ai.edge.litert.Accelerator
import com.powerampstartradio.data.EmbeddingDatabase
import com.powerampstartradio.indexing.AudioDecoder
import com.powerampstartradio.indexing.Clamp3AudioInference
import com.powerampstartradio.indexing.GraphUpdater
import com.powerampstartradio.indexing.MertInference
import com.powerampstartradio.indexing.TrackPcmCache
import com.powerampstartradio.indexing.V2ExactGraphIncrementalBase
import com.powerampstartradio.indexing.V2GraphUpdaterControl
import com.powerampstartradio.indexing.V2GraphExactProof
import com.powerampstartradio.indexing.V2GraphUpdaterProgress
import com.powerampstartradio.indexing.V2GraphUpdaterStage
import com.powerampstartradio.indexing.V2GraphWorkPlan
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.text.Normalizer
import java.util.Locale
import java.util.concurrent.Callable
import java.util.concurrent.CancellationException
import java.util.concurrent.ExecutionException
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.Future
import java.util.concurrent.TimeUnit

private const val MERT_PROGRESS_EVENT_INTERVAL_MS = 5_000L
private const val BYTES_PER_MEBIBYTE = 1024.0 * 1024.0

private fun formatIndexingBytes(bytes: Long): String =
    String.format(Locale.US, "%.1f MiB", bytes / BYTES_PER_MEBIBYTE)

private fun formatIndexingByteProgress(
    action: String,
    completedBytes: Long,
    totalBytes: Long,
    initialAction: String = action,
): String = if (completedBytes == 0L) {
    "$initialAction · ${formatIndexingBytes(totalBytes)} total to read"
} else {
    "$action · ${formatIndexingBytes(completedBytes)} of ${formatIndexingBytes(totalBytes)}"
}

enum class V2IndexingExecutorOutcome {
    COMPLETE,
    WAITING_FOR_INPUT,
}

data class V2IndexingExecutorEvent(
    val jobId: String,
    val workId: String?,
    val trackOrdinal: Int?,
    val trackTitle: String?,
    val stage: V2MeasuredWorkStage,
    val completedUnits: Long?,
    val totalUnits: Long?,
    val detail: String,
    val scope: V2IndexingEventScope? = null,
    val graphWorkPlan: V2GraphWorkPlan? = null,
    val pcmRateMeasurement: V2PcmRateMeasurement? = null,
)

enum class V2IndexingEventScope {
    CURRENT_TRACK,
    INDEXING_RUN,
    WHOLE_LIBRARY,
}

enum class V2PcmRateMeasurementPoint {
    MATERIALIZATION_STARTED,
    MATERIALIZATION_PROGRESS,
    MATERIALIZATION_COMPLETED_EXACT,
    VERIFIED_CACHE_REUSED_EXACT,
}

/** Explicit rate-learning evidence; provisional PCM progress totals are presentation only. */
data class V2PcmRateMeasurement(
    val powerampFileId: Long,
    val point: V2PcmRateMeasurementPoint,
    val exactSampleCount24k: Long? = null,
) {
    init {
        when (point) {
            V2PcmRateMeasurementPoint.MATERIALIZATION_STARTED,
            V2PcmRateMeasurementPoint.MATERIALIZATION_PROGRESS,
            -> require(exactSampleCount24k == null) {
                "$point cannot claim an exact sample count before EOS finalization"
            }

            V2PcmRateMeasurementPoint.MATERIALIZATION_COMPLETED_EXACT,
            V2PcmRateMeasurementPoint.VERIFIED_CACHE_REUSED_EXACT,
            -> require(exactSampleCount24k != null && exactSampleCount24k > 0L) {
                "$point requires the positive decoded 24 kHz sample count"
            }
        }
    }
}

interface V2IndexingExecutorControl {
    fun throwIfStopped()
    fun executionProfile(): V2IndexingExecutionProfile
    fun onProgress(event: V2IndexingExecutorEvent)
}

data class V2ResolvedExecutorModels(
    val mert: File,
    val clamp3Audio: File,
)

data class V2ResolvedTextRetrievalAssets(
    val clamp3Text: File,
    val sentencePieceModel: File,
)

data class V2ResolvedIndexingAssets(
    val audio: V2ResolvedExecutorModels,
    val text: V2ResolvedTextRetrievalAssets,
)

private data class V2PreparedPcmCache(
    val verified: V2VerifiedPcmCache,
    val materializedNow: Boolean,
)

private data class V2PendingPcmPrefetch(
    val powerampFileId: Long,
    val future: Future<V2PreparedPcmCache>,
)

private data class V2ProviderRowActivationVerification(
    val changed: List<SelectedTrackDescriptor>,
    /** Reused by CUE authorization so activation never repeats the same complete provider read. */
    val completeSnapshot: V2ProviderPathGroupSnapshot?,
)

data class V2IndexingArtifactHashProgress(
    val filename: String,
    val fileOrdinal: Int,
    val fileCount: Int,
    val completedBytes: Long,
    val totalBytes: Long,
)

/** Resolves installed model bytes by the immutable hashes recorded during preflight. */
class V2IndexingModelResolver(private val filesDir: File) {
    @Synchronized
    fun resolveAll(
        embeddingSpec: EmbeddingSpecFingerprint,
        textSpec: TextRetrievalSpecFingerprint,
        onHashProgress: (V2IndexingArtifactHashProgress) -> Unit,
    ): V2ResolvedIndexingAssets {
        V2IndexingExecutionPolicies.requirePinnedByteContract(embeddingSpec)
        V2IndexingLedgerValidator.requireValidTextRetrievalSpec(textSpec)
        require(textSpec.compatibleAudioEmbeddingSpecId == embeddingSpec.specId) {
            "text and audio model specifications are incompatible"
        }
        val resolved = resolveInstalled(onHashProgress)
        requirePolicyMatches(resolved, embeddingSpec, textSpec)
        return V2ResolvedIndexingAssets(
            audio = V2ResolvedExecutorModels(
                mert = resolved.filesByName.getValue("mert.tflite"),
                clamp3Audio = resolved.filesByName.getValue("clamp3_audio.tflite"),
            ),
            text = V2ResolvedTextRetrievalAssets(
                clamp3Text = resolved.filesByName.getValue("clamp3_text.tflite"),
                sentencePieceModel = resolved.filesByName.getValue("sentencepiece.bpe.model"),
            ),
        )
    }

    @Synchronized
    fun resolveTextWithProgress(
        spec: TextRetrievalSpecFingerprint,
        onHashProgress: (V2IndexingArtifactHashProgress) -> Unit,
    ): V2ResolvedTextRetrievalAssets {
        V2IndexingLedgerValidator.requireValidTextRetrievalSpec(spec)
        val resolved = resolveInstalled(onHashProgress)
        requireExpectedHash(resolved, "clamp3_text.tflite", spec.textModelSha256)
        requireExpectedHash(resolved, "sentencepiece.bpe.model", spec.tokenizerModelSha256)
        return V2ResolvedTextRetrievalAssets(
            clamp3Text = resolved.filesByName.getValue("clamp3_text.tflite"),
            sentencePieceModel = resolved.filesByName.getValue("sentencepiece.bpe.model"),
        )
    }

    @Synchronized
    fun resolve(spec: EmbeddingSpecFingerprint): V2ResolvedExecutorModels {
        V2IndexingExecutionPolicies.requirePinnedByteContract(spec)
        val resolved = resolveInstalled()
        requireExpectedHash(
            resolved,
            "mert.tflite",
            spec.modelArtifactSha256.getValue("mert"),
        )
        requireExpectedHash(
            resolved,
            "clamp3_audio.tflite",
            spec.modelArtifactSha256.getValue("clamp3_audio"),
        )
        return V2ResolvedExecutorModels(
            mert = resolved.filesByName.getValue("mert.tflite"),
            clamp3Audio = resolved.filesByName.getValue("clamp3_audio.tflite"),
        )
    }

    @Synchronized
    fun resolveText(spec: TextRetrievalSpecFingerprint): V2ResolvedTextRetrievalAssets {
        V2IndexingLedgerValidator.requireValidTextRetrievalSpec(spec)
        val resolved = resolveInstalled()
        requireExpectedHash(resolved, "clamp3_text.tflite", spec.textModelSha256)
        requireExpectedHash(resolved, "sentencepiece.bpe.model", spec.tokenizerModelSha256)
        return V2ResolvedTextRetrievalAssets(
            clamp3Text = resolved.filesByName.getValue("clamp3_text.tflite"),
            sentencePieceModel = resolved.filesByName.getValue("sentencepiece.bpe.model"),
        )
    }

    private fun resolveInstalled(
        onHashProgress: (V2IndexingArtifactHashProgress) -> Unit = {},
    ): V2ResolvedInstalledModelPolicy = try {
        V2CurrentModelPolicyResolver.resolveInstalled(filesDir) { progress ->
            onHashProgress(
                V2IndexingArtifactHashProgress(
                    filename = progress.filename,
                    fileOrdinal = progress.fileOrdinal,
                    fileCount = progress.fileCount,
                    completedBytes = progress.completedBytes,
                    totalBytes = progress.totalBytes,
                ),
            )
        }
    } catch (error: V2ModelLoadException) {
        throw error
    } catch (error: Exception) {
        throw V2ModelLoadException("installed model identity could not be verified", error)
    }

    private fun requirePolicyMatches(
        installed: V2ResolvedInstalledModelPolicy,
        embeddingSpec: EmbeddingSpecFingerprint,
        textSpec: TextRetrievalSpecFingerprint,
    ) {
        requireExpectedHash(
            installed,
            "mert.tflite",
            embeddingSpec.modelArtifactSha256.getValue("mert"),
        )
        requireExpectedHash(
            installed,
            "clamp3_audio.tflite",
            embeddingSpec.modelArtifactSha256.getValue("clamp3_audio"),
        )
        requireExpectedHash(installed, "clamp3_text.tflite", textSpec.textModelSha256)
        requireExpectedHash(installed, "sentencepiece.bpe.model", textSpec.tokenizerModelSha256)
    }

    private fun requireExpectedHash(
        installed: V2ResolvedInstalledModelPolicy,
        filename: String,
        expectedSha256: String,
    ) {
        val actual = installed.sha256ByName[filename]
        if (actual != expectedSha256) {
            throw V2ModelLoadException(
                "installed $filename does not match immutable SHA-256 $expectedSha256",
            )
        }
    }
}

/**
 * Exact V2 indexing pipeline. It writes only job-private artifacts until generation activation's
 * atomic pointer publication, and every ledger mutation is guarded by an executor lease epoch.
 */
class V2IndexingExecutor(
    context: Context,
    private val repository: V2IndexingJobRepository = V2IndexingJobRepository.get(context),
    private val databaseStore: V2JobPrivateDatabaseStore =
        V2JobPrivateDatabaseStore(context.filesDir),
    private val providerSnapshots: V2PowerampProviderSnapshotAcquirer =
        V2PowerampProviderSnapshotAcquirer(context),
    private val sourceFingerprinter: V2SourceFingerprintProvider = V2ExactSourceFingerprinter(),
    private val modelResolver: V2IndexingModelResolver =
        V2IndexingModelResolver(context.filesDir),
    private val artifacts: AtomicV2ArtifactStore = AtomicV2ArtifactStore(),
    private val pcmCache: TrackPcmCache = TrackPcmCache(),
    private val verifiedPcmCaches: V2VerifiedPcmCacheStore = V2VerifiedPcmCacheStore(),
    private val generationPublisher: V2IndexGenerationPublisher =
        V2IndexGenerationPublisher(context.filesDir),
    private val importedRowAuthorizationStore: V2ImportedRowSupersessionAuthorizationStore =
        V2ImportedRowSupersessionAuthorizationStore(File(context.filesDir, "indexing_v2/jobs")),
    private val nowEpochMs: () -> Long = System::currentTimeMillis,
    private val pcmPrefetchEnabled: Boolean = true,
) {
    private val filesDir = context.filesDir
    private val terminalJobCleanup = V2TerminalJobCleanup(
        artifactDirectory = repository::artifactDirectory,
        cleanupStagingDatabase = databaseStore::cleanup,
    )
    private val sourceIdentityVerifier = V2SourceIdentityVerifier(sourceFingerprinter)
    @Volatile private var eventJobId: String? = null

    fun run(
        token: V2ExecutorLeaseToken,
        control: V2IndexingExecutorControl,
    ): V2IndexingExecutorOutcome {
        eventJobId = token.jobId
        var ledger = repository.require(token.jobId)
        V2IndexingExecutionPolicies.requirePinnedByteContract(ledger.jobSpec.embeddingSpec)
        if (ledger.state == IndexingJobState.ACTIVATING) {
            return activate(token, control)
        }
        require(ledger.state == IndexingJobState.RUNNING) {
            "executor requires RUNNING or ACTIVATING, got ${ledger.state}"
        }
        control.throwIfStopped()
        admitPlannedTracks(token, control)
        if (waitForInputWhenNothingCanReachCommit(
                token,
                productiveStates = setOf(
                    IndexingTrackState.PREFLIGHTED,
                    IndexingTrackState.MERT_COMPLETE,
                    IndexingTrackState.CLAMP_COMPLETE,
                    IndexingTrackState.COMMITTED,
                ),
            )
        ) {
            return V2IndexingExecutorOutcome.WAITING_FOR_INPUT
        }

        ledger = repository.require(token.jobId)
        val models = try {
            if (ledger.jobSpec.textRetrievalSpec.compatibleAudioEmbeddingSpecId !=
                ledger.jobSpec.embeddingSpec.specId
            ) {
                throw V2ModelLoadException(
                    "text retrieval contract targets a different audio embedding spec",
                )
            }
            modelResolver.resolveAll(
                embeddingSpec = ledger.jobSpec.embeddingSpec,
                textSpec = ledger.jobSpec.textRetrievalSpec,
            ) { progress ->
                control.throwIfStopped()
                emit(
                    control = control,
                    descriptor = null,
                    stage = V2MeasuredWorkStage.INDEXING_MODEL_FILES,
                    completed = progress.completedBytes,
                    total = progress.totalBytes,
                    detail = formatIndexingByteProgress(
                        action = "Verifying changed installed indexing file " +
                            "${progress.fileOrdinal} of " +
                            "${progress.fileCount}: ${progress.filename}",
                        completedBytes = progress.completedBytes,
                        totalBytes = progress.totalBytes,
                        initialAction = "Opening changed installed indexing file " +
                            "${progress.fileOrdinal} of ${progress.fileCount}: " +
                            "${progress.filename} for SHA-256 verification",
                    ),
                    scope = V2IndexingEventScope.INDEXING_RUN,
                )
            }.audio
        } catch (error: Throwable) {
            V2ExecutorFailureBoundary.rethrowControlFlow(error)
            exhaustGlobalAutomaticRetries(token, error)
            repository.waitForInput(token)
            return V2IndexingExecutorOutcome.WAITING_FOR_INPUT
        }

        runMertPhase(token, models.mert, control)
        ledger = repository.require(token.jobId)
        if (ledger.tracks.any { it.state == IndexingTrackState.RETRYABLE_FAILURE }) {
            repository.waitForInput(token)
            return V2IndexingExecutorOutcome.WAITING_FOR_INPUT
        }
        if (waitForInputWhenNothingCanReachCommit(
                token,
                productiveStates = setOf(
                    IndexingTrackState.MERT_COMPLETE,
                    IndexingTrackState.CLAMP_COMPLETE,
                    IndexingTrackState.COMMITTED,
                ),
            )
        ) {
            return V2IndexingExecutorOutcome.WAITING_FOR_INPUT
        }
        val stagingAndAuthorization = try {
            V2IndexingPlanFinalizationPolicy.requireStagingDatabaseReady(ledger)
            val database = databaseStore.prepare(ledger) { progress ->
                control.throwIfStopped()
                emit(
                    control = control,
                    descriptor = null,
                    stage = V2MeasuredWorkStage.PRIVATE_INDEX_COPY,
                    completed = progress.completedBytes,
                    total = progress.totalBytes,
                    detail = progress.detail + if (
                        progress.completedBytes != null && progress.totalBytes != null
                    ) {
                        " · ${formatIndexingBytes(progress.completedBytes)} of " +
                            formatIndexingBytes(progress.totalBytes)
                    } else {
                        ""
                    },
                    scope = V2IndexingEventScope.INDEXING_RUN,
                )
            }
            database to requireImportedRowAuthorization(ledger)
        } catch (error: Throwable) {
            V2ExecutorFailureBoundary.rethrowControlFlow(error)
            recordGlobalFailureAtCurrentCheckpoints(token, error)
            repository.waitForInput(token)
            return V2IndexingExecutorOutcome.WAITING_FOR_INPUT
        }
        runClampAndCommitPhase(
            token,
            models.clamp3Audio,
            stagingAndAuthorization.first,
            stagingAndAuthorization.second,
            control,
        )

        ledger = repository.require(token.jobId)
        if (ledger.tracks.any { it.state == IndexingTrackState.RETRYABLE_FAILURE }) {
            repository.waitForInput(token)
            return V2IndexingExecutorOutcome.WAITING_FOR_INPUT
        }
        if (waitForInputWhenNothingCanReachCommit(
                token,
                productiveStates = setOf(IndexingTrackState.COMMITTED),
            )
        ) {
            return V2IndexingExecutorOutcome.WAITING_FOR_INPUT
        }
        if (ledger.tracks.any { !it.state.isResolvedForActivation() }) {
            throw InvalidIndexingLedgerException("executor stopped with unresolved runnable tracks")
        }
        return activate(token, control)
    }

    fun cleanupCancelledJob(token: V2ExecutorLeaseToken) {
        val jobId = token.jobId
        repository.artifactDirectory(jobId).deleteRecursively()
        databaseStore.cleanup(jobId)
        if (repository.artifactDirectory(jobId).exists()) {
            throw IllegalStateException("unable to remove private artifacts for cancelled job $jobId")
        }
    }

    fun cleanupTerminalJob(ledger: IndexingJobLedger) {
        terminalJobCleanup.cleanup(ledger.jobSpec.jobId, ledger.state)
    }

    private fun waitForInputWhenNothingCanReachCommit(
        token: V2ExecutorLeaseToken,
        productiveStates: Set<IndexingTrackState>,
    ): Boolean {
        val ledger = repository.require(token.jobId)
        val hasFailure = ledger.tracks.any {
            it.state == IndexingTrackState.RETRYABLE_FAILURE ||
                it.state == IndexingTrackState.BLOCKED_FAILURE
        }
        if (!hasFailure || ledger.tracks.any { it.state in productiveStates }) return false
        repository.waitForInput(token)
        return true
    }

    /**
     * Planning already pinned every selected source by exact content. Promote those immutable
     * descriptors in one ledger write; each source is verified again immediately before its own
     * decode, so there is no reason to gate track one behind an availability sweep of the job.
     */
    private fun admitPlannedTracks(
        token: V2ExecutorLeaseToken,
        control: V2IndexingExecutorControl,
    ) {
        if (repository.require(token.jobId).tracks.none {
                it.state == IndexingTrackState.QUEUED
            }
        ) return
        control.throwIfStopped()
        try {
            repository.executorUpdate(token) { current ->
                V2IndexingLedgerStateMachine.admitPlannedTracksForExecution(
                    current,
                    nowEpochMs(),
                )
            }
        } catch (error: Throwable) {
            control.throwIfStopped()
            throw error
        }
    }

    private fun runMertPhase(
        token: V2ExecutorLeaseToken,
        modelFile: File,
        control: V2IndexingExecutorControl,
    ) {
        while (true) {
            materializeMertAliases(token, control)
            val candidates = acousticWorkLeadersInState(
                token.jobId,
                IndexingTrackState.PREFLIGHTED,
                VerifiedArtifactKind.MERT_FEATURES,
            )
            if (candidates.isEmpty()) return
            emit(
                control = control,
                descriptor = null,
                stage = V2MeasuredWorkStage.INDEXING_MODEL_FILES,
                completed = null,
                total = null,
                detail = "Initializing the MERT GPU runtime from " +
                    "${modelFile.name} (${formatIndexingBytes(modelFile.length())})",
                scope = V2IndexingEventScope.INDEXING_RUN,
            )
            val inference = try {
                MertInference(modelFile).also { loaded ->
                    if (loaded.activeAccelerator != Accelerator.GPU) {
                        loaded.close()
                        throw V2ModelLoadException("MERT did not activate the pinned GPU backend")
                    }
                }
            } catch (error: Throwable) {
                V2ExecutorFailureBoundary.rethrowControlFlow(error)
                exhaustGlobalAutomaticRetries(
                    token,
                    V2ModelLoadException("MERT load failed", error),
                    setOf(IndexingTrackState.PREFLIGHTED),
                )
                return
            }
            val prefetchExecutor = if (pcmPrefetchEnabled && candidates.size > 1) {
                newPcmPrefetchExecutor()
            } else {
                null
            }
            var pendingPrefetch: V2PendingPcmPrefetch? = null
            try {
                for ((candidateIndex, candidate) in candidates.withIndex()) {
                    control.throwIfStopped()
                    if (repository.require(token.jobId).tracks.single {
                            it.workId == candidate.workId
                        }.state != IndexingTrackState.PREFLIGHTED
                    ) {
                        if (pendingPrefetch?.powerampFileId == candidate.powerampFileId) {
                            pendingPrefetch?.future?.cancel(true)
                            pendingPrefetch = null
                        }
                        continue
                    }
                    var descriptor = candidate
                    beginStage(token, descriptor)
                    val artifactRoot = repository.artifactDirectory(token.jobId)
                    try {
                        val prefetched = pendingPrefetch
                            ?.takeIf { it.powerampFileId == descriptor.powerampFileId }
                            ?.also { pendingPrefetch = null }
                            ?.let { awaitPcmPrefetch(it, control) }
                        val source = verifySource(
                            descriptor = descriptor,
                            exactContent = false,
                        ) { completedBytes, totalBytes ->
                            emit(
                                control = control,
                                descriptor = descriptor,
                                stage = V2MeasuredWorkStage.SOURCE_AUDIO_HASH,
                                completed = completedBytes,
                                total = totalBytes,
                                detail = formatIndexingByteProgress(
                                    action = "Hashing this source audio against its saved fingerprint",
                                    completedBytes = completedBytes,
                                    totalBytes = totalBytes,
                                    initialAction = "Opening this source audio for exact fingerprint " +
                                        "comparison",
                                ),
                                scope = V2IndexingEventScope.CURRENT_TRACK,
                            )
                        }
                        val schedule = applyProfile(control.executionProfile())
                        val cached = prefetched?.verified ?: verifiedPcmCaches.loadVerified(
                            artifactRoot,
                            token.jobId,
                            descriptor,
                        ) { completedBytes, totalBytes ->
                            emit(
                                control = control,
                                descriptor = descriptor,
                                stage = V2MeasuredWorkStage.PCM_CACHE_BYTES,
                                completed = completedBytes,
                                total = totalBytes,
                                detail = formatIndexingByteProgress(
                                    action = "Verifying saved decoded audio",
                                    completedBytes = completedBytes,
                                    totalBytes = totalBytes,
                                    initialAction = "Opening saved decoded audio for receipt verification",
                                ),
                            )
                        }
                        val materializedNow = prefetched?.materializedNow ?: (cached == null)
                        val provisionalPcmUnits = descriptor.finalizedAudioSpan
                            .exactSampleCount24k
                            .takeIf { it > 0L }
                        val verifiedCache = if (cached != null) {
                            cached
                        } else {
                            emit(
                                control,
                                descriptor,
                                V2MeasuredWorkStage.PCM_24K_SAMPLES,
                                null,
                                null,
                                "Decoding source audio before resampling",
                                pcmRateMeasurement = pcmRateMeasurement(
                                    descriptor,
                                    V2PcmRateMeasurementPoint.MATERIALIZATION_STARTED,
                                ),
                            )
                            val built = pcmCache.build(
                                sourceFile = source,
                                logicalStartUs = descriptor.finalizedAudioSpan.startUs,
                                logicalDurationUs = descriptor.finalizedAudioSpan.endExclusiveUs -
                                    descriptor.finalizedAudioSpan.startUs,
                                chunkDurationMs = schedule.pcmChunkDurationMs,
                                outputFile = verifiedPcmCaches.pcmFile(
                                    artifactRoot,
                                    descriptor.powerampFileId,
                                ),
                                boundaryMode = pcmBoundaryMode(descriptor),
                                resamplerPolicy = TrackPcmCache.ResamplerPolicy.TORCHAUDIO_HANN_V1,
                                onChunkDone = { completed, total ->
                                    control.throwIfStopped()
                                    emit(
                                        control,
                                        descriptor,
                                        V2MeasuredWorkStage.PCM_24K_SAMPLES,
                                        provisionalPcmUnits?.let { units ->
                                            units * completed / total
                                        },
                                        provisionalPcmUnits,
                                        "Resampling decoded audio · chunk $completed of $total",
                                        pcmRateMeasurement = pcmRateMeasurement(
                                            descriptor,
                                            V2PcmRateMeasurementPoint.MATERIALIZATION_PROGRESS,
                                        ),
                                    )
                                },
                                onNormalizationProgress = { completed, total ->
                                    control.throwIfStopped()
                                    emit(
                                        control = control,
                                        descriptor = descriptor,
                                        stage = V2MeasuredWorkStage.PCM_NORMALIZATION_SAMPLES,
                                        completed = completed,
                                        total = total,
                                        detail = if (completed == 0L) {
                                            "Preparing to calculate normalization across $total audio samples"
                                        } else {
                                            "Calculated normalization across $completed of $total audio samples"
                                        },
                                    )
                                },
                                isCancelled = {
                                    control.throwIfStopped()
                                    false
                                },
                            )
                            verifySource(
                                descriptor = descriptor,
                                exactContent = false,
                            ) { completedBytes, totalBytes ->
                                emit(
                                    control = control,
                                    descriptor = descriptor,
                                    stage = V2MeasuredWorkStage.SOURCE_AUDIO_HASH,
                                    completed = completedBytes,
                                    total = totalBytes,
                                    detail = formatIndexingByteProgress(
                                        action = "Source metadata changed during decode; comparing " +
                                            "its exact fingerprint",
                                        completedBytes = completedBytes,
                                        totalBytes = totalBytes,
                                        initialAction = "Source metadata changed during decode; " +
                                            "opening it for exact fingerprint comparison",
                                    ),
                                    scope = V2IndexingEventScope.CURRENT_TRACK,
                                )
                            }
                            verifiedPcmCaches.publish(
                                artifactRoot,
                                token.jobId,
                                descriptor,
                                built,
                                nowEpochMs(),
                            ) { completedBytes, totalBytes ->
                                control.throwIfStopped()
                                emit(
                                    control = control,
                                    descriptor = descriptor,
                                    stage = V2MeasuredWorkStage.PCM_CACHE_BYTES,
                                    completed = completedBytes,
                                    total = totalBytes,
                                    detail = formatIndexingByteProgress(
                                        action = "Hashing decoded audio for its crash-safe receipt",
                                        completedBytes = completedBytes,
                                        totalBytes = totalBytes,
                                        initialAction = "Opening decoded audio for crash-safe receipt " +
                                            "hashing",
                                    ),
                                )
                            }
                        }
                        val pcm = verifiedCache.result
                        descriptor = finalizeDecodedEos(token, descriptor, pcm)
                        val verifiedPcm = verifiedPcmCaches.requireUnchanged(
                            jobArtifactDirectory = artifactRoot,
                            jobId = token.jobId,
                            descriptor = descriptor,
                            verified = verifiedCache,
                        ).result
                        V2IndexingPlanFinalizationPolicy.requireMertReady(descriptor)
                        emit(
                            control,
                            descriptor,
                            V2MeasuredWorkStage.PCM_24K_SAMPLES,
                            descriptor.finalizedAudioSpan.exactSampleCount24k,
                            descriptor.finalizedAudioSpan.exactSampleCount24k,
                            "Physical boundary finalized; ETA revised to exact work",
                            pcmRateMeasurement = pcmRateMeasurement(
                                descriptor,
                                if (materializedNow) {
                                    V2PcmRateMeasurementPoint.MATERIALIZATION_COMPLETED_EXACT
                                } else {
                                    V2PcmRateMeasurementPoint.VERIFIED_CACHE_REUSED_EXACT
                                },
                                descriptor.finalizedAudioSpan.exactSampleCount24k,
                            ),
                        )
                        control.throwIfStopped()
                        val workFiles = workFiles(token.jobId, descriptor.workId)
                        val boundary = requirePcmMatchesPlan(descriptor, verifiedPcm)
                        if (prefetchExecutor != null && pendingPrefetch == null) {
                            val next = candidates.drop(candidateIndex + 1).firstOrNull { queued ->
                                stateFor(token.jobId, queued.workId) == IndexingTrackState.PREFLIGHTED
                            }
                            if (next != null) {
                                pendingPrefetch = startPcmPrefetch(
                                    executor = prefetchExecutor,
                                    jobId = token.jobId,
                                    artifactRoot = artifactRoot,
                                    descriptor = next,
                                    schedule = schedule,
                                    control = control,
                                )
                            }
                        }
                        var windows = 0
                        val totalWindows = descriptor.expectedWork.mertWindows.toLong()
                        val progressCadence = V2IndexingProgressEventCadence(
                            MERT_PROGRESS_EVENT_INTERVAL_MS,
                        )
                        if (progressCadence.shouldEmit(
                                0L,
                                totalWindows,
                                SystemClock.elapsedRealtime(),
                            )
                        ) {
                            emit(
                                control,
                                descriptor,
                                V2MeasuredWorkStage.MERT_WINDOWS,
                                null,
                                null,
                                "Starting MERT audio features · window 1 of " +
                                    "${descriptor.expectedWork.mertWindows}",
                            )
                        }
                        val artifact = artifacts.publishMertFeaturesStreaming(
                            target = workFiles.mert,
                            storageKey = workFiles.mertStorageKey,
                            expectedWindows = descriptor.expectedWork.mertWindows,
                            finalizedAudioSpan = descriptor.finalizedAudioSpan,
                            executionBoundary = boundary,
                            embeddingSpecId = repository.require(token.jobId).jobSpec.embeddingSpec.specId,
                            sourceFingerprint = descriptor.sourceFingerprint,
                            verificationCompletedAtEpochMs = nowEpochMs,
                        ) { writeFeature ->
                            inference.extractFeaturesFromPcmFile(
                                pcmFile = verifiedPcm.file,
                                normalization = verifiedPcm.normalization,
                                onFeatureExtracted = writeFeature,
                                onWindowDone = {
                                    windows++
                                    control.throwIfStopped()
                                    if (progressCadence.shouldEmit(
                                            windows.toLong(),
                                            totalWindows,
                                            SystemClock.elapsedRealtime(),
                                        )
                                    ) {
                                        emit(
                                            control,
                                            descriptor,
                                            V2MeasuredWorkStage.MERT_WINDOWS,
                                            windows.toLong(),
                                            totalWindows,
                                            "Computing MERT audio features · window $windows of " +
                                                "${descriptor.expectedWork.mertWindows}",
                                        )
                                    }
                                    yieldForProfile(schedule)
                                },
                            )
                        }
                        completeStage(token, descriptor, artifact)
                        materializeMertAliasesFromDonor(token, descriptor, control)
                        verifiedPcmCaches.delete(artifactRoot, descriptor.powerampFileId)
                    } catch (error: Throwable) {
                        recordTrackFailure(token, descriptor, error, IndexingStage.DECODE_AND_MERT)
                    }
                }
            } finally {
                shutdownPcmPrefetch(prefetchExecutor, pendingPrefetch)
                inference.close()
                Process.setThreadPriority(Process.THREAD_PRIORITY_DEFAULT)
            }
            val hasMertWork = acousticWorkLeadersInState(
                token.jobId,
                IndexingTrackState.PREFLIGHTED,
                VerifiedArtifactKind.MERT_FEATURES,
            ).isNotEmpty()
            if (hasMertWork) continue
            if (!retryImmediate(token)) return
        }
    }

    private fun runClampAndCommitPhase(
        token: V2ExecutorLeaseToken,
        modelFile: File,
        stagingDatabase: File,
        importedRowAuthorization: V2ImportedRowSupersessionAuthorization?,
        control: V2IndexingExecutorControl,
    ) {
        while (true) {
            materializeClampAliases(token, control)
            val clampCandidates = acousticWorkLeadersInState(
                token.jobId,
                IndexingTrackState.MERT_COMPLETE,
                VerifiedArtifactKind.CLAMP_VECTOR,
            )
            if (clampCandidates.isNotEmpty()) {
                emit(
                    control = control,
                    descriptor = null,
                    stage = V2MeasuredWorkStage.INDEXING_MODEL_FILES,
                    completed = null,
                    total = null,
                    detail = "Initializing the CLaMP3 GPU runtime from " +
                        "${modelFile.name} (${formatIndexingBytes(modelFile.length())})",
                    scope = V2IndexingEventScope.INDEXING_RUN,
                )
                val inference = try {
                    Clamp3AudioInference(modelFile).also { loaded ->
                        if (loaded.activeAccelerator != Accelerator.GPU) {
                            loaded.close()
                            throw V2ModelLoadException(
                                "CLaMP3 audio did not activate the pinned GPU backend",
                            )
                        }
                    }
                } catch (error: Throwable) {
                    V2ExecutorFailureBoundary.rethrowControlFlow(error)
                    exhaustGlobalAutomaticRetries(
                        token,
                        V2ModelLoadException("CLaMP3 audio load failed", error),
                        setOf(IndexingTrackState.MERT_COMPLETE),
                    )
                    continue
                }
                try {
                    for (descriptor in clampCandidates) {
                        control.throwIfStopped()
                        if (stateFor(token.jobId, descriptor.workId) !=
                            IndexingTrackState.MERT_COMPLETE
                        ) continue
                        val failure = encodeClamp(token, descriptor, inference, control)
                        if (failure == null) {
                            materializeClampAliasesFromDonor(token, descriptor, control)
                        }
                    }
                } finally {
                    inference.close()
                    Process.setThreadPriority(Process.THREAD_PRIORITY_DEFAULT)
                }
            }

            val commitCandidates = descriptorsInState(token.jobId, IndexingTrackState.CLAMP_COMPLETE)
            if (commitCandidates.isNotEmpty()) {
                SQLiteDatabase.openDatabase(
                    stagingDatabase.path,
                    null,
                    SQLiteDatabase.OPEN_READWRITE,
                ).use { database ->
                    val commits = V2EmbeddingCommitRepository(database)
                    for (descriptor in commitCandidates) {
                        control.throwIfStopped()
                        if (stateFor(token.jobId, descriptor.workId) !=
                            IndexingTrackState.CLAMP_COMPLETE
                        ) continue
                        commitClamp(
                            token,
                            descriptor,
                            commits,
                            importedRowAuthorization,
                            control,
                        )
                    }
                }
            }

            val hasClampWork = descriptorsInState(
                token.jobId,
                IndexingTrackState.MERT_COMPLETE,
            ).isNotEmpty() || descriptorsInState(
                token.jobId,
                IndexingTrackState.CLAMP_COMPLETE,
            ).isNotEmpty()
            if (hasClampWork) continue
            if (!retryImmediate(token)) return
        }
    }

    private fun encodeClamp(
        token: V2ExecutorLeaseToken,
        descriptor: SelectedTrackDescriptor,
        inference: Clamp3AudioInference,
        control: V2IndexingExecutorControl,
    ): Throwable? {
        beginStage(token, descriptor)
        val files = workFiles(token.jobId, descriptor.workId)
        try {
            V2DecodedEosPublicationPolicy.requirePublishable(descriptor.finalizedAudioSpan)
            verifySource(descriptor, exactContent = false)
            val ledger = repository.require(token.jobId)
            val mertArtifact = artifactFor(
                ledger,
                descriptor.workId,
                VerifiedArtifactKind.MERT_FEATURES,
            )
            V2ArtifactIO.requireVerifiedFile(
                file = files.mert,
                artifact = mertArtifact,
                expectedKind = VerifiedArtifactKind.MERT_FEATURES,
                expectedStorageKey = files.mertStorageKey,
                expectedEmbeddingSpecId = ledger.jobSpec.embeddingSpec.specId,
                expectedSourceFingerprint = descriptor.sourceFingerprint,
                expectedPlannedUnits = descriptor.expectedWork.mertWindows,
            )
            V2ArtifactIO.requireExecutionBoundaryMatches(
                descriptor.finalizedAudioSpan,
                requireNotNull(mertArtifact.executionBoundary),
            )
            val expectedSegments = descriptor.expectedWork.clampSegments
            if (inference.segmentCount(descriptor.expectedWork.mertWindows) != expectedSegments) {
                throw V2InvalidModelOutputException("CLaMP segment count disagrees with plan")
            }
            val schedule = applyProfile(control.executionProfile())
            var segments = 0
            emit(
                control,
                descriptor,
                V2MeasuredWorkStage.CLAMP_SEGMENTS,
                null,
                null,
                "Starting CLaMP3 embedding · segment 1 of $expectedSegments",
            )
            val vector = FileInputStream(files.mert).channel.use { channel ->
                var windowsRead = 0
                val result = inference.encodeStreaming(
                    numWindows = descriptor.expectedWork.mertWindows,
                    readNextWindow = {
                        control.throwIfStopped()
                        val bytes = ByteArray(V2ArtifactIO.BYTES_PER_RECORD)
                        V2ExactChannelIO.readFully(channel, ByteBuffer.wrap(bytes))
                        windowsRead++
                        V2ArtifactIO.decodeMertWindow(bytes)
                    },
                    onSegmentDone = {
                        segments++
                        control.throwIfStopped()
                        emit(
                            control,
                            descriptor,
                            V2MeasuredWorkStage.CLAMP_SEGMENTS,
                            segments.toLong(),
                            expectedSegments.toLong(),
                            "Computing CLaMP3 embedding · segment $segments of $expectedSegments",
                        )
                        yieldForProfile(schedule)
                    },
                ) ?: throw V2InvalidModelOutputException("CLaMP3 produced no finite embedding")
                if (windowsRead != descriptor.expectedWork.mertWindows) {
                    throw V2ArtifactIntegrityException("MERT reader consumed $windowsRead windows")
                }
                V2ExactChannelIO.requireExhausted(channel)
                result
            }
            if (segments != expectedSegments) {
                throw V2InvalidModelOutputException(
                    "CLaMP3 completed $segments of $expectedSegments segments",
                )
            }
            val artifact = artifacts.publishClampVector(
                target = files.clamp,
                storageKey = files.clampStorageKey,
                vector = vector,
                completedClampSegments = expectedSegments,
                embeddingSpecId = ledger.jobSpec.embeddingSpec.specId,
                sourceFingerprint = descriptor.sourceFingerprint,
                verifiedAtEpochMs = nowEpochMs(),
            )
            completeStage(token, descriptor, artifact)
            return null
        } catch (error: Throwable) {
            recordTrackFailure(token, descriptor, error, IndexingStage.CLAMP3)
            return error
        }
    }

    private fun commitClamp(
        token: V2ExecutorLeaseToken,
        descriptor: SelectedTrackDescriptor,
        commits: V2EmbeddingCommitRepository,
        importedRowAuthorization: V2ImportedRowSupersessionAuthorization?,
        control: V2IndexingExecutorControl,
    ) {
        beginStage(token, descriptor)
        val files = workFiles(token.jobId, descriptor.workId)
        try {
            V2DecodedEosPublicationPolicy.requirePublishable(descriptor.finalizedAudioSpan)
            val ledger = repository.require(token.jobId)
            val clampArtifact = artifactFor(
                ledger,
                descriptor.workId,
                VerifiedArtifactKind.CLAMP_VECTOR,
            )
            val blob = V2ArtifactIO.readVerifiedClampBlob(
                file = files.clamp,
                artifact = clampArtifact,
                expectedStorageKey = files.clampStorageKey,
                expectedEmbeddingSpecId = ledger.jobSpec.embeddingSpec.specId,
                expectedSourceFingerprint = descriptor.sourceFingerprint,
                expectedClampSegments = descriptor.expectedWork.clampSegments,
            )
            emit(
                control,
                descriptor,
                V2MeasuredWorkStage.DATABASE_COMMITS,
                0L,
                1L,
                "Saving this track's embedding in the new music index",
            )
            val result = commits.commit(
                request = V2EmbeddingCommitRequest(
                    workId = descriptor.workId,
                    stableTrackSpanIdentity = descriptor.stableTrackSpanIdentity,
                    embeddingSpecId = ledger.jobSpec.embeddingSpec.specId,
                    providerSpan = V2CommittedProviderSpan(
                        normalizedPhysicalPath = descriptor.providerRow.physicalPath,
                        offsetMs = descriptor.providerOffsetMs,
                        durationMs = V2ProviderDurationEvidencePolicy.canonicalMs(
                            descriptor.providerDurationMs,
                        ),
                    ),
                    metadata = commitMetadata(descriptor),
                    sourceFingerprint = descriptor.sourceFingerprint,
                    clampVectorArtifact = clampArtifact,
                    verifiedAtEpochMs = nowEpochMs(),
                    importedRowSupersession = when (
                        descriptor.finalizedAudioSpan.kind
                    ) {
                        V2ResolvedAudioSpanKind.WHOLE_FILE -> null
                        V2ResolvedAudioSpanKind.LOGICAL_CUE ->
                            requireNotNull(importedRowAuthorization) {
                                "Logical CUE work has no imported-row authorization"
                            }.commitAuthorizationFor(descriptor.workId, ledger)
                    },
                ),
                clampVectorBlob = blob,
            )
            completeStage(token, descriptor, result.databaseArtifact)
            emit(
                control,
                descriptor,
                V2MeasuredWorkStage.DATABASE_COMMITS,
                1L,
                1L,
                "Saved this track's embedding in the new music index",
            )
        } catch (error: Throwable) {
            recordTrackFailure(token, descriptor, error, IndexingStage.DATABASE_COMMIT)
        }
    }

    private fun activate(
        token: V2ExecutorLeaseToken,
        control: V2IndexingExecutorControl,
    ): V2IndexingExecutorOutcome {
        control.throwIfStopped()
        var ledger = repository.require(token.jobId)
        emit(
            control = control,
            descriptor = null,
            stage = V2MeasuredWorkStage.STAGING_INDEX_INSPECTION,
            completed = null,
            total = null,
            detail = "Opening the staged music index and counting its embeddings",
            scope = V2IndexingEventScope.INDEXING_RUN,
        )
        val database = databaseStore.requirePrepared(ledger)
        val activationTrackCount = EmbeddingDatabase.open(database).let { activationDatabase ->
            try {
                activationDatabase.getEmbeddingCount().also { count ->
                    require(count > 0) { "activation database has no embedding rows" }
                }
            } finally {
                activationDatabase.close()
            }
        }
        val graph = if (ledger.jobSpec.rebuildDerivedIndexes) {
            emit(
                control = control,
                descriptor = null,
                stage = V2MeasuredWorkStage.SIMILARITY_GRAPH_SETUP,
                completed = null,
                total = null,
                detail = "Reading the existing graph and planning the exact similarity update",
                scope = V2IndexingEventScope.WHOLE_LIBRARY,
            )
            val exactGraphBase = try {
                V2IndexGenerationReader.requireActive(filesDir) { progress ->
                    control.throwIfStopped()
                    emit(
                        control = control,
                        descriptor = null,
                        stage = V2MeasuredWorkStage.SIMILARITY_GRAPH_SETUP,
                        completed = progress.completedBytes,
                        total = progress.totalBytes,
                        detail = formatIndexingByteProgress(
                            action = "Hashing graph-base music-index file ${progress.filename}",
                            completedBytes = progress.completedBytes,
                            totalBytes = progress.totalBytes,
                            initialAction = "Opening graph-base music-index file " +
                                "${progress.filename} for SHA-256 verification",
                        ),
                        scope = V2IndexingEventScope.WHOLE_LIBRARY,
                    )
                }.also { active ->
                    require(active.manifest.generationId == ledger.jobSpec.baseGenerationId) {
                        "active graph base changed after indexing preflight"
                    }
                }.let(V2ExactGraphIncrementalBase::fromActiveGeneration)
            } catch (error: Throwable) {
                V2ExecutorFailureBoundary.rethrowControlFlow(error)
                null
            }
            val directory = File(repository.artifactDirectory(token.jobId), "derived")
            val file = File(directory, V2GraphGenerationFile.GRAPH_FILE)
            val embeddingFile = File(directory, "clamp3.emb")
            val validExisting = runCatching { V2GraphGenerationFile.inspect(file) }.isSuccess &&
                embeddingFile.isFile && EmbeddingDatabase.open(database).let { privateDatabase ->
                    try {
                        V2GraphExactProof.matchesFiles(privateDatabase, file, embeddingFile)
                    } finally {
                        privateDatabase.close()
                    }
                }
            if (!validExisting) {
                directory.mkdirs()
                val db = EmbeddingDatabase.openReadWrite(database)
                try {
                    val lastYieldUnits = mutableMapOf<V2GraphUpdaterStage, Long>()
                    applyProfile(control.executionProfile())
                    GraphUpdater(db, directory).rebuildIndices(
                        control = object : V2GraphUpdaterControl {
                            override fun onProgress(progress: V2GraphUpdaterProgress) {
                                control.throwIfStopped()
                                emit(
                                    control = control,
                                    descriptor = null,
                                    stage = progress.stage.measuredStage(),
                                    completed = progress.completedUnits,
                                    total = progress.totalUnits,
                                    detail = progress.detail,
                                    graphWorkPlan = progress.plan.takeUnless {
                                        progress.stage == V2GraphUpdaterStage.EMBEDDING_ROWS
                                    },
                                )
                            }

                            override fun onControlPoint(
                                stage: V2GraphUpdaterStage,
                                completedUnits: Long,
                            ) {
                                control.throwIfStopped()
                                val schedule = applyProfile(control.executionProfile())
                                val previous = lastYieldUnits[stage] ?: 0L
                                if (completedUnits - previous >= stage.profileYieldInterval()) {
                                    yieldForProfile(schedule)
                                    lastYieldUnits[stage] = completedUnits
                                    control.throwIfStopped()
                                }
                            }
                        },
                        exactBase = exactGraphBase,
                    )
                } finally {
                    db.close()
                    Process.setThreadPriority(Process.THREAD_PRIORITY_DEFAULT)
                }
            }
            file
        } else null

        if (ledger.state == IndexingJobState.RUNNING) {
            ledger = repository.executorUpdate(token) { current ->
                V2IndexingLedgerStateMachine.beginActivation(current, nowEpochMs())
            }
        }
        control.throwIfStopped()
        var activationProviderSnapshot: V2ProviderPathGroupSnapshot? = null
        if (!generationAlreadyPublishedByThisJob(ledger, control)) {
            val committed = V2ActivationSourceSelection.committedDescriptors(ledger)
            val sourceFailures = verifyCommittedSourcesForActivation(committed, control)
            if (sourceFailures.isNotEmpty()) {
                return blockCommittedActivationForNewJob(token, sourceFailures)
            }
            control.throwIfStopped()
            emit(
                control = control,
                descriptor = null,
                stage = V2MeasuredWorkStage.POWERAMP_LIBRARY_ROWS,
                completed = null,
                total = null,
                detail = "Reading selected Poweramp rows before publishing the new music index",
                scope = V2IndexingEventScope.INDEXING_RUN,
            )
            val providerVerification = changedProviderRows(
                committed,
                onSnapshotProgress = { completedRows, totalRows ->
                    val selectedRead = totalRows == committed.size && committed.none {
                        it.finalizedAudioSpan.kind == V2ResolvedAudioSpanKind.LOGICAL_CUE
                    }
                    emit(
                        control = control,
                        descriptor = null,
                        stage = V2MeasuredWorkStage.POWERAMP_LIBRARY_ROWS,
                        completed = completedRows.toLong(),
                        total = totalRows.toLong(),
                        detail = if (selectedRead && completedRows == 0) {
                            "Reading ${committed.size} selected Poweramp rows before publication"
                        } else if (selectedRead) {
                            "Read $completedRows of ${committed.size} selected Poweramp rows " +
                                "before publication"
                        } else if (completedRows == 0) {
                            "Poweramp reports $totalRows library rows; beginning the final read " +
                                "before publication"
                        } else {
                            "Read $completedRows of $totalRows Poweramp library rows before " +
                                "publication"
                        },
                        scope = V2IndexingEventScope.INDEXING_RUN,
                    )
                },
            )
            activationProviderSnapshot = providerVerification.completeSnapshot
            val providerFailures = providerVerification.changed.map { descriptor ->
                V2ActivationTrackFailure(
                    workId = descriptor.workId,
                    code = TrackFailureCode.PROVIDER_SNAPSHOT_CHANGED,
                    diagnostic = "Poweramp row ${descriptor.powerampFileId} changed since preflight",
                )
            }
            if (providerFailures.isNotEmpty()) {
                return blockCommittedActivationForNewJob(token, providerFailures)
            }
            control.throwIfStopped()
        }
        val importedRowAuthorization = try {
            requireImportedRowAuthorization(ledger, activationProviderSnapshot)
        } catch (error: V2ImportedRowAuthorizationException) {
            return blockImportedRowActivation(token, error)
        }
        control.throwIfStopped()
        val resolved = try {
            generationPublisher.publish(
                ledger = ledger,
                jobPrivateStagingDatabase = database,
                explicitGraphFile = graph,
                importedRowAuthorization = importedRowAuthorization,
            ) { progress ->
                control.throwIfStopped()
                val detail = progress.detail + if (
                    progress.unit == V2GenerationPublicationUnit.BYTES &&
                    progress.completedUnits != null && progress.totalUnits != null
                ) {
                    " · ${formatIndexingBytes(progress.completedUnits)} of " +
                        formatIndexingBytes(progress.totalUnits)
                } else {
                    ""
                }
                emit(
                    control = control,
                    descriptor = null,
                    stage = V2MeasuredWorkStage.MUSIC_INDEX_PUBLICATION,
                    completed = progress.completedUnits,
                    total = progress.totalUnits,
                    detail = detail,
                    scope = V2IndexingEventScope.WHOLE_LIBRARY,
                )
            }
        } catch (error: V2ImportedRowAuthorizationException) {
            return blockImportedRowActivation(token, error)
        }
        val complete = repository.executorUpdate(token) { current ->
            V2IndexingLedgerStateMachine.completeJob(
                current,
                resolved.activatedEvidence(nowEpochMs()),
                nowEpochMs(),
            )
        }
        cleanupTerminalJob(complete)
        emit(
            control,
            null,
            V2MeasuredWorkStage.ACTIVATION_TRACKS,
            activationTrackCount.toLong(),
            activationTrackCount.toLong(),
            "New embeddings are ready",
        )
        return V2IndexingExecutorOutcome.COMPLETE
    }

    private fun blockImportedRowActivation(
        token: V2ExecutorLeaseToken,
        error: V2ImportedRowAuthorizationException,
    ): V2IndexingExecutorOutcome {
        val current = repository.require(token.jobId)
        val logicalCue = current.jobSpec.tracks.firstOrNull {
            it.finalizedAudioSpan.kind == V2ResolvedAudioSpanKind.LOGICAL_CUE
        } ?: throw InvalidIndexingLedgerException(
            "imported-row authorization failed for a job with no logical CUE work",
        )
        repository.executorUpdate(token) { ledger ->
            V2IndexingLedgerStateMachine.blockImportedRowActivationForNewJob(
                ledger = ledger,
                workId = logicalCue.workId,
                diagnostic = error.message ?: "Imported CUE activation proof changed",
                nowEpochMs = nowEpochMs(),
            )
        }
        return V2IndexingExecutorOutcome.WAITING_FOR_INPUT
    }

    private fun blockCommittedActivationForNewJob(
        token: V2ExecutorLeaseToken,
        failures: List<V2ActivationTrackFailure>,
    ): V2IndexingExecutorOutcome {
        repository.executorUpdate(token) { ledger ->
            V2IndexingLedgerStateMachine.blockCommittedActivationForNewJob(
                ledger = ledger,
                failures = failures,
                nowEpochMs = nowEpochMs(),
            )
        }
        return V2IndexingExecutorOutcome.WAITING_FOR_INPUT
    }

    private fun verifyCommittedSourcesForActivation(
        committed: List<SelectedTrackDescriptor>,
        control: V2IndexingExecutorControl,
    ): List<V2ActivationTrackFailure> {
        val verifier = V2SourceIdentityVerifier(
            V2DeduplicatingSourceFingerprintProvider(sourceFingerprinter),
        )
        val totalBytes = committed.sumOf { it.sourceFingerprint.sizeBytes }
        var completedBeforeSource = 0L
        return buildList {
            committed.forEachIndexed { index, descriptor ->
                control.throwIfStopped()
                val title = descriptor.displayMetadata.let { metadata ->
                    listOf(metadata.artist, metadata.title)
                        .filter(String::isNotBlank)
                        .joinToString(" - ")
                }
                try {
                    verifier.requireVerified(
                        providerPhysicalPath = descriptor.providerRow.providerPhysicalPath,
                        canonicalPath = descriptor.canonicalPath,
                        powerampFileId = descriptor.powerampFileId,
                        planned = descriptor.sourceFingerprint,
                        exactContent = true,
                    ) { completedInSource, _ ->
                        val completedBytes = completedBeforeSource + completedInSource
                        emit(
                            control = control,
                            descriptor = null,
                            stage = V2MeasuredWorkStage.SOURCE_AUDIO_HASH,
                            completed = completedBytes,
                            total = totalBytes,
                            detail = "Hashing source ${index + 1} of ${committed.size} before " +
                                "publication · ${formatIndexingBytes(completedBytes)} of " +
                                "${formatIndexingBytes(totalBytes)}" +
                                title.takeIf(String::isNotBlank)
                                    ?.let { " · $it" }
                                    .orEmpty(),
                            scope = V2IndexingEventScope.INDEXING_RUN,
                        )
                    }
                } catch (error: V2SourceIdentityChangedException) {
                    add(
                        V2ActivationTrackFailure(
                            workId = descriptor.workId,
                            code = TrackFailureCode.SOURCE_FINGERPRINT_CHANGED,
                            diagnostic = error.message ?: "Source bytes changed before activation",
                        ),
                    )
                }
                completedBeforeSource += descriptor.sourceFingerprint.sizeBytes
            }
        }
    }

    private fun generationAlreadyPublishedByThisJob(
        ledger: IndexingJobLedger,
        control: V2IndexingExecutorControl,
    ): Boolean {
        val active = try {
            V2IndexGenerationReader.requireActive(filesDir) { progress ->
                control.throwIfStopped()
                emit(
                    control = control,
                    descriptor = null,
                    stage = V2MeasuredWorkStage.MUSIC_INDEX_PUBLICATION,
                    completed = progress.completedBytes,
                    total = progress.totalBytes,
                    detail = formatIndexingByteProgress(
                        action = "Hashing active publication file ${progress.filename}",
                        completedBytes = progress.completedBytes,
                        totalBytes = progress.totalBytes,
                        initialAction = "Opening active publication file ${progress.filename} for " +
                            "SHA-256 verification",
                    ),
                    scope = V2IndexingEventScope.WHOLE_LIBRARY,
                )
            }
        } catch (error: Throwable) {
            V2ExecutorFailureBoundary.rethrowControlFlow(error)
            return false
        }
        return active.manifest.origin == V2IndexGenerationOrigin.INDEXING_JOB &&
            active.manifest.jobId == ledger.jobSpec.jobId &&
            active.manifest.jobSpecId == ledger.jobSpec.specId
    }

    private fun changedProviderRows(
        descriptors: List<SelectedTrackDescriptor>,
        onSnapshotProgress: ((completedRows: Int, totalRows: Int) -> Unit)? = null,
    ): V2ProviderRowActivationVerification {
        val selectedIds = descriptors
            .mapTo(hashSetOf()) { it.powerampFileId }
        require(selectedIds.size == descriptors.size) {
            "immutable indexing plan repeats a Poweramp row"
        }
        val acquisition = providerSnapshots.acquireSelectedWithCueFallbackBlocking(
            fileIds = descriptors.map { it.powerampFileId },
            requireCompletePathGroups = descriptors.any {
                it.finalizedAudioSpan.kind == V2ResolvedAudioSpanKind.LOGICAL_CUE
            },
            onRowProgress = onSnapshotProgress,
        )
        val currentSelectedGroups =
            HashMap<Long, V2ProviderPathGroupEvidence>(selectedIds.size)
        acquisition.snapshot.groups.forEach { group ->
            group.rows.forEach { row ->
                if (row.powerampFileId in selectedIds) {
                    check(currentSelectedGroups.put(row.powerampFileId, group) == null) {
                        "Poweramp snapshot repeats selected row ${row.powerampFileId}"
                    }
                }
            }
        }
        return V2ProviderRowActivationVerification(
            changed = descriptors.filter { descriptor ->
                val group = currentSelectedGroups[descriptor.powerampFileId]
                    ?: return@filter true
                group.rows.singleOrNull {
                    it.powerampFileId == descriptor.powerampFileId
                } != descriptor.providerRow ||
                    V2CueClassificationEvidenceFactory.from(group) !=
                    descriptor.finalizedAudioSpan.cueClassification
            },
            completeSnapshot = acquisition.snapshot.takeIf {
                acquisition.scope == V2PowerampProviderAcquisitionScope.COMPLETE_LIBRARY
            },
        )
    }

    private fun requireImportedRowAuthorization(
        ledger: IndexingJobLedger,
        activationProviderSnapshot: V2ProviderPathGroupSnapshot? = null,
    ): V2ImportedRowSupersessionAuthorization? {
        val hasLogicalCue = ledger.jobSpec.tracks.any {
            it.finalizedAudioSpan.kind == V2ResolvedAudioSpanKind.LOGICAL_CUE
        }
        if (!hasLogicalCue) return null
        val persisted = importedRowAuthorizationStore.requireFor(ledger)
        val active = try {
            V2IndexGenerationReader.requireActive(filesDir)
        } catch (error: Exception) {
            throw V2ImportedRowAuthorizationException(
                "Unable to verify the imported CUE base generation; start a new preflight",
                error,
            )
        }
        val publishedByThisJob = active.manifest.origin ==
            V2IndexGenerationOrigin.INDEXING_JOB &&
            active.manifest.jobId == ledger.jobSpec.jobId &&
            active.manifest.jobSpecId == ledger.jobSpec.specId &&
            active.manifest.baseGenerationId == persisted.baseGenerationId
        if (!publishedByThisJob) {
            val snapshot = activationProviderSnapshot ?: providerSnapshots.acquireBlocking()
            val recomputed = V2ImportedRowSupersessionAuthorizer.authorize(
                ledger,
                active,
                snapshot,
            ) ?: throw V2ImportedRowAuthorizationException(
                "Imported CUE authorization disappeared; start a new preflight",
            )
            if (persisted != recomputed ||
                persisted.baseManifestSha256 != active.manifestSha256 ||
                persisted.baseDatabaseByteLength != active.manifest.databaseByteLength ||
                persisted.baseDatabaseSha256 != active.manifest.databaseSha256 ||
                persisted.baseDatabaseContentSha256 != active.manifest.databaseContentSha256
            ) {
                throw V2ImportedRowAuthorizationException(
                    "Imported CUE evidence changed; start a new indexing preflight",
                )
            }
        }
        val privateDatabase = databaseStore.requirePrepared(ledger)
        val privateBinding = databaseStore.requirePreparedBinding(ledger)
        val executionPrivateBaseBindingId = persisted.executionPrivateBaseBindingId(ledger)
        if (privateBinding.schemaVersion < 2 ||
            privateBinding.bindingId != executionPrivateBaseBindingId ||
            privateBinding.sourceDatabaseByteLength != persisted.baseDatabaseByteLength ||
            privateBinding.sourceDatabaseSha256 != persisted.baseDatabaseSha256 ||
            privateBinding.baseManifestSha256 != persisted.baseManifestSha256 ||
            privateBinding.baseDatabaseContentSha256 != persisted.baseDatabaseContentSha256
        ) {
            throw V2ImportedRowAuthorizationException(
                "The private CUE base copy is not authorized; start a new indexing preflight",
            )
        }
        val hasDurableCommit = ledger.tracks.any { track ->
            track.checkpoint == TrackCheckpoint.COMMITTED ||
                track.verifiedArtifacts.any { it.kind == VerifiedArtifactKind.DATABASE_COMMIT }
        }
        if (!hasDurableCommit) {
            val privateContent = SQLiteDatabase.openDatabase(
                privateDatabase.path,
                null,
                SQLiteDatabase.OPEN_READONLY,
            ).use { database ->
                V2EmbeddingGenerationFile.digest(V2SqliteOrderedEmbeddingSource(database))
            }
            if (privateContent.trackCount != active.manifest.trackCount ||
                privateContent.orderedTrackSetSha256 != active.manifest.orderedTrackSetSha256 ||
                privateContent.databaseContentSha256 != persisted.baseDatabaseContentSha256
            ) {
                throw V2ImportedRowAuthorizationException(
                    "The private CUE base copy content changed; start a new indexing preflight",
                )
            }
        }
        return persisted
    }

    private fun verifySource(
        descriptor: SelectedTrackDescriptor,
        exactContent: Boolean,
        onHashProgress: (completedBytes: Long, totalBytes: Long) -> Unit = { _, _ -> },
    ): File = sourceIdentityVerifier.requireVerified(
        providerPhysicalPath = descriptor.providerRow.providerPhysicalPath,
        canonicalPath = descriptor.canonicalPath,
        powerampFileId = descriptor.powerampFileId,
        planned = descriptor.sourceFingerprint,
        exactContent = exactContent,
        onHashProgress = onHashProgress,
    )

    private fun newPcmPrefetchExecutor(): ExecutorService =
        Executors.newSingleThreadExecutor { task ->
            Thread(task, "v2-pcm-prefetch").apply { isDaemon = true }
        }

    private fun startPcmPrefetch(
        executor: ExecutorService,
        jobId: String,
        artifactRoot: File,
        descriptor: SelectedTrackDescriptor,
        schedule: V2IndexingExecutionSchedule,
        control: V2IndexingExecutorControl,
    ): V2PendingPcmPrefetch = V2PendingPcmPrefetch(
        powerampFileId = descriptor.powerampFileId,
        future = executor.submit(
            Callable {
                Process.setThreadPriority(schedule.threadPriority)
                try {
                    preparePcmPrefetch(
                        jobId = jobId,
                        artifactRoot = artifactRoot,
                        descriptor = descriptor,
                        schedule = schedule,
                        control = control,
                    )
                } finally {
                    Process.setThreadPriority(Process.THREAD_PRIORITY_DEFAULT)
                }
            },
        ),
    )

    private fun preparePcmPrefetch(
        jobId: String,
        artifactRoot: File,
        descriptor: SelectedTrackDescriptor,
        schedule: V2IndexingExecutionSchedule,
        control: V2IndexingExecutorControl,
    ): V2PreparedPcmCache {
        requirePcmPrefetchActive(control)
        val source = verifySource(
            descriptor = descriptor,
            exactContent = false,
        ) { _, _ -> requirePcmPrefetchActive(control) }
        val cached = verifiedPcmCaches.loadVerified(
            jobArtifactDirectory = artifactRoot,
            jobId = jobId,
            descriptor = descriptor,
        ) { _, _ -> requirePcmPrefetchActive(control) }
        if (cached != null) return V2PreparedPcmCache(cached, materializedNow = false)

        val built = pcmCache.build(
            sourceFile = source,
            logicalStartUs = descriptor.finalizedAudioSpan.startUs,
            logicalDurationUs = descriptor.finalizedAudioSpan.endExclusiveUs -
                descriptor.finalizedAudioSpan.startUs,
            chunkDurationMs = schedule.pcmChunkDurationMs,
            outputFile = verifiedPcmCaches.pcmFile(
                artifactRoot,
                descriptor.powerampFileId,
            ),
            boundaryMode = pcmBoundaryMode(descriptor),
            resamplerPolicy = TrackPcmCache.ResamplerPolicy.TORCHAUDIO_HANN_V1,
            onChunkDone = { _, _ -> requirePcmPrefetchActive(control) },
            onNormalizationProgress = { _, _ -> requirePcmPrefetchActive(control) },
            isCancelled = {
                control.throwIfStopped()
                Thread.currentThread().isInterrupted
            },
        )
        verifySource(
            descriptor = descriptor,
            exactContent = false,
        ) { _, _ -> requirePcmPrefetchActive(control) }
        val verified = verifiedPcmCaches.publish(
            jobArtifactDirectory = artifactRoot,
            jobId = jobId,
            descriptor = descriptor,
            result = built,
            verifiedAtEpochMs = nowEpochMs(),
        ) { _, _ -> requirePcmPrefetchActive(control) }
        return V2PreparedPcmCache(verified, materializedNow = true)
    }

    private fun awaitPcmPrefetch(
        pending: V2PendingPcmPrefetch,
        control: V2IndexingExecutorControl,
    ): V2PreparedPcmCache = try {
        pending.future.get()
    } catch (error: InterruptedException) {
        Thread.currentThread().interrupt()
        control.throwIfStopped()
        throw AudioDecoder.AudioDecodeCancelledException()
    } catch (error: CancellationException) {
        control.throwIfStopped()
        throw AudioDecoder.AudioDecodeCancelledException()
    } catch (error: ExecutionException) {
        throw (error.cause ?: error)
    }

    private fun shutdownPcmPrefetch(
        executor: ExecutorService?,
        pending: V2PendingPcmPrefetch?,
    ) {
        pending?.future?.cancel(true)
        if (executor == null) return
        executor.shutdownNow()
        var interrupted = false
        while (!executor.isTerminated) {
            try {
                executor.awaitTermination(250L, TimeUnit.MILLISECONDS)
            } catch (_: InterruptedException) {
                interrupted = true
            }
        }
        if (interrupted) Thread.currentThread().interrupt()
    }

    private fun requirePcmPrefetchActive(control: V2IndexingExecutorControl) {
        control.throwIfStopped()
        if (Thread.currentThread().isInterrupted) {
            throw AudioDecoder.AudioDecodeCancelledException()
        }
    }

    private fun pcmBoundaryMode(
        descriptor: SelectedTrackDescriptor,
    ): TrackPcmCache.BoundaryMode = when (
        descriptor.finalizedAudioSpan.executionBoundaryRequirement
    ) {
        V2ExecutionBoundaryRequirement.VERIFY_END_OF_STREAM_AND_RECONCILE ->
            TrackPcmCache.BoundaryMode.REQUIRE_PHYSICAL_END_OF_STREAM
        V2ExecutionBoundaryRequirement.ENFORCE_PROVIDER_HALF_OPEN_SPAN ->
            TrackPcmCache.BoundaryMode.ENFORCE_LOGICAL_HALF_OPEN_SPAN
    }

    private fun finalizeDecodedEos(
        token: V2ExecutorLeaseToken,
        descriptor: SelectedTrackDescriptor,
        pcm: TrackPcmCache.Result,
    ): SelectedTrackDescriptor {
        if (descriptor.finalizedAudioSpan.kind != V2ResolvedAudioSpanKind.WHOLE_FILE) {
            return descriptor
        }
        val evidence = V2DecodedEosEvidence(
            sourceSampleRateHz = pcm.sourceSampleRate,
            observedStartSourceSample = pcm.sourceStartSample,
            observedEndSourceSampleExclusive = pcm.sourceEndSampleExclusive,
            observedSourceSampleCount = pcm.sourceSampleCount,
            exactSampleCount24k = pcm.exactSampleCount24k,
            endOfStreamReached = pcm.endOfStreamReached,
        )
        if (descriptor.finalizedAudioSpan.authority ==
            V2AudioSpanAuthority.PROVISIONAL_END_OF_STREAM
        ) {
            repository.executorUpdate(token) { current ->
                V2DecodedEosPlanFinalizer.finalizeCanonicalGroup(
                    ledger = current,
                    canonicalWorkId = descriptor.workId,
                    evidence = evidence,
                    nowEpochMs = nowEpochMs(),
                ).ledger
            }
        } else {
            V2DecodedEosSpanFinalizer.finalize(descriptor.finalizedAudioSpan, evidence)
        }
        return repository.require(token.jobId).jobSpec.tracks.single {
            it.powerampFileId == descriptor.powerampFileId
        }
    }

    private fun requirePcmMatchesPlan(
        descriptor: SelectedTrackDescriptor,
        pcm: TrackPcmCache.Result,
    ): VerifiedExecutionBoundaryEvidence {
        val span = descriptor.finalizedAudioSpan
        V2IndexingPlanFinalizationPolicy.requireMertReady(descriptor)
        if (pcm.sourceSampleRate != span.container.sampleRateHz ||
            pcm.sourceStartSample != span.startSourceSample ||
            pcm.sourceEndSampleExclusive != span.endSourceSampleExclusive ||
            pcm.sourceSampleCount != span.sourceSampleCount ||
            pcm.exactSampleCount24k != span.exactSampleCount24k ||
            pcm.preprocessingSpecId != com.powerampstartradio.indexing.NativeMath
                .TORCHAUDIO_HANN_V1_SPEC_ID
        ) {
            throw TrackPcmCache.PcmContractException(
                TrackPcmCache.PcmContractFailure.SOURCE_PLAN_MISMATCH,
                "decoded PCM evidence disagrees with immutable track span",
            )
        }
        return VerifiedExecutionBoundaryEvidence(
            requirement = span.executionBoundaryRequirement,
            observedStartSourceSample = pcm.sourceStartSample,
            observedEndSourceSampleExclusive = pcm.sourceEndSampleExclusive,
            observedSourceSampleCount = pcm.sourceSampleCount,
            exactSampleCount24k = pcm.exactSampleCount24k,
            endOfStreamReached = pcm.endOfStreamReached,
            providerBoundaryEnforced = pcm.logicalBoundaryEnforced,
        ).also { V2ArtifactIO.requireExecutionBoundaryMatches(span, it) }
    }

    private fun recordGlobalFailureAtCurrentCheckpoints(
        token: V2ExecutorLeaseToken,
        error: Throwable,
        allowedStates: Set<IndexingTrackState>? = null,
    ) {
        val stageByState = listOf(
            IndexingTrackState.QUEUED to IndexingStage.PREFLIGHT,
            IndexingTrackState.PREFLIGHTED to IndexingStage.DECODE_AND_MERT,
            IndexingTrackState.MERT_COMPLETE to IndexingStage.CLAMP3,
            IndexingTrackState.CLAMP_COMPLETE to IndexingStage.DATABASE_COMMIT,
        )
        stageByState.forEach { (state, stage) ->
            if (allowedStates != null && state !in allowedStates) return@forEach
            descriptorsInState(token.jobId, state).forEach { descriptor ->
                beginStage(token, descriptor)
                recordTrackFailure(token, descriptor, error, stage)
            }
        }
    }

    private fun exhaustGlobalAutomaticRetries(
        token: V2ExecutorLeaseToken,
        error: Throwable,
        allowedStates: Set<IndexingTrackState>? = null,
    ) {
        do {
            recordGlobalFailureAtCurrentCheckpoints(token, error, allowedStates)
        } while (retryImmediate(token))
    }

    private fun recordTrackFailure(
        token: V2ExecutorLeaseToken,
        descriptor: SelectedTrackDescriptor,
        error: Throwable,
        stage: IndexingStage,
    ) {
        val classified = V2ExecutorFailureBoundary.classifyOrRethrow(
            error,
            stage,
            descriptor.finalizedAudioSpan,
        )
        repository.executorUpdate(token) { current ->
            V2IndexingLedgerStateMachine.recordTrackFailure(
                current,
                descriptor.workId,
                classified.code,
                stage,
                classified.diagnostic,
                nowEpochMs(),
            )
        }
    }

    private fun retryImmediate(token: V2ExecutorLeaseToken): Boolean =
        repository.retryFailed(token.jobId, RetryTrigger.IMMEDIATE).retried > 0

    private fun beginStage(token: V2ExecutorLeaseToken, descriptor: SelectedTrackDescriptor) {
        repository.executorUpdate(token) { current ->
            V2IndexingLedgerStateMachine.beginNextTrackStage(
                current,
                descriptor.workId,
                nowEpochMs(),
            )
        }
    }

    private fun completeStage(
        token: V2ExecutorLeaseToken,
        descriptor: SelectedTrackDescriptor,
        artifact: VerifiedArtifact?,
    ) {
        repository.executorUpdate(token) { current ->
            V2IndexingLedgerStateMachine.completeActiveTrackStage(
                current,
                descriptor.workId,
                artifact,
                nowEpochMs(),
            )
        }
    }

    private fun acousticWorkLeadersInState(
        jobId: String,
        state: IndexingTrackState,
        completedArtifactKind: VerifiedArtifactKind,
    ): List<SelectedTrackDescriptor> = V2CanonicalAcousticWorkExecutionPolicy.leadersInState(
        ledger = repository.require(jobId),
        state = state,
        completedArtifactKind = completedArtifactKind,
    )

    private fun materializeMertAliases(
        token: V2ExecutorLeaseToken,
        control: V2IndexingExecutorControl,
    ) {
        val ledger = repository.require(token.jobId)
        val trackByWorkId = ledger.tracks.associateBy(IndexingTrackLedger::workId)
        V2CanonicalAcousticWorkPlanner.groups(ledger.jobSpec).forEach { group ->
            V2CanonicalAcousticWorkExecutionPolicy.artifactDonor(
                group,
                trackByWorkId,
                VerifiedArtifactKind.MERT_FEATURES,
            )?.let { donor -> materializeMertAliasesFromDonor(token, donor, control) }
        }
    }

    private fun materializeMertAliasesFromDonor(
        token: V2ExecutorLeaseToken,
        donor: SelectedTrackDescriptor,
        control: V2IndexingExecutorControl,
    ) {
        val group = acousticGroup(token.jobId, donor.workId)
        if (group.members.size == 1) return
        val ledger = repository.require(token.jobId)
        val donorArtifact = artifactFor(
            ledger,
            donor.workId,
            VerifiedArtifactKind.MERT_FEATURES,
        )
        val donorFiles = workFiles(token.jobId, donor.workId)
        group.members.filter { it.workId != donor.workId }.forEach { alias ->
            if (stateFor(token.jobId, alias.workId) != IndexingTrackState.PREFLIGHTED) {
                return@forEach
            }
            control.throwIfStopped()
            beginStage(token, alias)
            try {
                verifySource(alias, exactContent = false) { completedBytes, totalBytes ->
                    emit(
                        control = control,
                        descriptor = alias,
                        stage = V2MeasuredWorkStage.SOURCE_AUDIO_HASH,
                        completed = completedBytes,
                        total = totalBytes,
                        detail = formatIndexingByteProgress(
                            action = "Hashing this duplicate source before reusing analyzed features",
                            completedBytes = completedBytes,
                            totalBytes = totalBytes,
                            initialAction = "Opening this duplicate source before verifying feature reuse",
                        ),
                        scope = V2IndexingEventScope.CURRENT_TRACK,
                    )
                }
                require(alias.expectedWork == donor.expectedWork) {
                    "stable acoustic aliases have different MERT work plans"
                }
                val aliasFiles = workFiles(token.jobId, alias.workId)
                val artifact = artifacts.publishMertAlias(
                    source = donorFiles.mert,
                    sourceArtifact = donorArtifact,
                    target = aliasFiles.mert,
                    targetStorageKey = aliasFiles.mertStorageKey,
                    targetSpan = alias.finalizedAudioSpan,
                    targetSourceFingerprint = alias.sourceFingerprint,
                    verifiedAtEpochMs = nowEpochMs(),
                )
                completeStage(token, alias, artifact)
                emit(
                    control,
                    alias,
                    V2MeasuredWorkStage.MERT_WINDOWS,
                    alias.expectedWork.mertWindows.toLong(),
                    alias.expectedWork.mertWindows.toLong(),
                    "Reused byte-identical MERT features",
                )
            } catch (error: Throwable) {
                recordTrackFailure(token, alias, error, IndexingStage.DECODE_AND_MERT)
            }
        }
    }

    private fun materializeClampAliases(
        token: V2ExecutorLeaseToken,
        control: V2IndexingExecutorControl,
    ) {
        val ledger = repository.require(token.jobId)
        val trackByWorkId = ledger.tracks.associateBy(IndexingTrackLedger::workId)
        V2CanonicalAcousticWorkPlanner.groups(ledger.jobSpec).forEach { group ->
            V2CanonicalAcousticWorkExecutionPolicy.artifactDonor(
                group,
                trackByWorkId,
                VerifiedArtifactKind.CLAMP_VECTOR,
            )?.let { donor -> materializeClampAliasesFromDonor(token, donor, control) }
        }
    }

    private fun materializeClampAliasesFromDonor(
        token: V2ExecutorLeaseToken,
        donor: SelectedTrackDescriptor,
        control: V2IndexingExecutorControl,
    ) {
        val group = acousticGroup(token.jobId, donor.workId)
        if (group.members.size == 1) return
        val ledger = repository.require(token.jobId)
        val donorArtifact = artifactFor(
            ledger,
            donor.workId,
            VerifiedArtifactKind.CLAMP_VECTOR,
        )
        val donorFiles = workFiles(token.jobId, donor.workId)
        group.members.filter { it.workId != donor.workId }.forEach { alias ->
            if (stateFor(token.jobId, alias.workId) != IndexingTrackState.MERT_COMPLETE) {
                return@forEach
            }
            control.throwIfStopped()
            beginStage(token, alias)
            try {
                verifySource(alias, exactContent = false)
                require(alias.expectedWork == donor.expectedWork) {
                    "stable acoustic aliases have different CLaMP work plans"
                }
                val aliasFiles = workFiles(token.jobId, alias.workId)
                val artifact = artifacts.publishClampAlias(
                    source = donorFiles.clamp,
                    sourceArtifact = donorArtifact,
                    target = aliasFiles.clamp,
                    targetStorageKey = aliasFiles.clampStorageKey,
                    targetSourceFingerprint = alias.sourceFingerprint,
                    expectedClampSegments = alias.expectedWork.clampSegments,
                    verifiedAtEpochMs = nowEpochMs(),
                )
                completeStage(token, alias, artifact)
                emit(
                    control,
                    alias,
                    V2MeasuredWorkStage.CLAMP_SEGMENTS,
                    alias.expectedWork.clampSegments.toLong(),
                    alias.expectedWork.clampSegments.toLong(),
                    "Reused byte-identical CLaMP embedding",
                )
            } catch (error: Throwable) {
                recordTrackFailure(token, alias, error, IndexingStage.CLAMP3)
            }
        }
    }

    private fun acousticGroup(jobId: String, memberWorkId: String): V2CanonicalAcousticWorkGroup =
        V2CanonicalAcousticWorkPlanner.groups(repository.require(jobId).jobSpec)
            .single { group -> group.members.any { it.workId == memberWorkId } }

    private fun descriptorsInState(
        jobId: String,
        state: IndexingTrackState,
    ): List<SelectedTrackDescriptor> {
        val ledger = repository.require(jobId)
        val states = ledger.tracks.associate { it.workId to it.state }
        return ledger.jobSpec.tracks.filter { states[it.workId] == state }
    }

    private fun stateFor(jobId: String, workId: String): IndexingTrackState =
        repository.require(jobId).tracks.single { it.workId == workId }.state

    private fun artifactFor(
        ledger: IndexingJobLedger,
        workId: String,
        kind: VerifiedArtifactKind,
    ): VerifiedArtifact = ledger.tracks.single { it.workId == workId }
        .verifiedArtifacts.singleOrNull { it.kind == kind }
        ?: throw V2ArtifactIntegrityException("ledger omitted $kind artifact for $workId")

    private fun commitMetadata(descriptor: SelectedTrackDescriptor): V2CommitTrackMetadata {
        val display = descriptor.displayMetadata
        val filenameSource = if (display.artist.isNotEmpty()) {
            "${display.artist} - ${display.title}"
        } else display.title
        val filenameKey = Normalizer.normalize(
            filenameSource.lowercase()
                .replace(Regex("\\s*[\\(\\[].*?[\\)\\]]"), "")
                .replace(Regex("^\\d+[.\\-\\s]+"), "")
                .replace(Regex("\\s+"), " ")
                .trim(),
            Normalizer.Form.NFC,
        )
        return V2CommitTrackMetadata(
            metadataKey = descriptor.normalizedMetadata.metadataKey,
            filenameKey = filenameKey,
            artist = display.artist.ifBlank { null },
            album = display.album.ifBlank { null },
            title = display.title.ifBlank { null },
            durationMs = Math.toIntExact(
                V2ProviderDurationEvidencePolicy.canonicalMs(descriptor.providerDurationMs),
            ),
            filePath = descriptor.providerRow.providerPhysicalPath,
            source = "phone-v2",
        )
    }

    private fun applyProfile(
        @Suppress("UNUSED_PARAMETER") profile: V2IndexingExecutionProfile,
    ): V2IndexingExecutionSchedule =
        V2IndexingExecutionPolicies.schedule(V2IndexingExecutionProfile.FULL).also { schedule ->
            Process.setThreadPriority(schedule.threadPriority)
        }

    private fun yieldForProfile(schedule: V2IndexingExecutionSchedule) {
        if (schedule.yieldAfterCompletedUnitMs > 0L) {
            Thread.sleep(schedule.yieldAfterCompletedUnitMs)
        }
    }

    private fun emit(
        control: V2IndexingExecutorControl,
        descriptor: SelectedTrackDescriptor?,
        stage: V2MeasuredWorkStage,
        completed: Long?,
        total: Long?,
        detail: String,
        scope: V2IndexingEventScope? = null,
        graphWorkPlan: V2GraphWorkPlan? = null,
        pcmRateMeasurement: V2PcmRateMeasurement? = null,
    ) {
        control.onProgress(
            V2IndexingExecutorEvent(
                jobId = checkNotNull(eventJobId) { "executor event has no active job" },
                workId = descriptor?.workId,
                trackOrdinal = descriptor?.ordinal,
                trackTitle = descriptor?.displayMetadata?.let { metadata ->
                    listOf(metadata.artist, metadata.title)
                        .filter(String::isNotBlank)
                        .joinToString(" - ")
                },
                stage = stage,
                completedUnits = completed,
                totalUnits = total,
                detail = detail,
                scope = scope,
                graphWorkPlan = graphWorkPlan,
                pcmRateMeasurement = pcmRateMeasurement,
            ),
        )
    }

    private fun pcmRateMeasurement(
        descriptor: SelectedTrackDescriptor,
        point: V2PcmRateMeasurementPoint,
        exactSampleCount24k: Long? = null,
    ) = V2PcmRateMeasurement(
        powerampFileId = descriptor.powerampFileId,
        point = point,
        exactSampleCount24k = exactSampleCount24k,
    )

    private fun V2GraphUpdaterStage.measuredStage(): V2MeasuredWorkStage = when (this) {
        V2GraphUpdaterStage.EMBEDDING_ROWS -> V2MeasuredWorkStage.DERIVED_EMBEDDING_ROWS
        V2GraphUpdaterStage.SIMILARITY_DOT_PRODUCTS ->
            V2MeasuredWorkStage.GRAPH_SIMILARITY_DOT_PRODUCTS
        V2GraphUpdaterStage.GRAPH_BINARY_BYTES -> V2MeasuredWorkStage.GRAPH_BINARY_BYTES
    }

    private fun V2GraphUpdaterStage.profileYieldInterval(): Long = when (this) {
        V2GraphUpdaterStage.EMBEDDING_ROWS -> 4_096L
        V2GraphUpdaterStage.SIMILARITY_DOT_PRODUCTS -> 1_000_000L
        V2GraphUpdaterStage.GRAPH_BINARY_BYTES -> 1L shl 20
    }

    private data class WorkFiles(
        val pcm: File,
        val mert: File,
        val clamp: File,
        val mertStorageKey: String,
        val clampStorageKey: String,
    )

    private fun workFiles(jobId: String, workId: String): WorkFiles {
        val directory = File(repository.artifactDirectory(jobId), "tracks/$workId")
        return WorkFiles(
            pcm = File(directory, "pcm-24k-f32.bin"),
            mert = File(directory, "mert-features-f32.bin"),
            clamp = File(directory, "clamp3-vector-f32.bin"),
            mertStorageKey = "job:$jobId:track:$workId:mert-features-f32-v1",
            clampStorageKey = "job:$jobId:track:$workId:clamp3-vector-f32-v1",
        )
    }
}
