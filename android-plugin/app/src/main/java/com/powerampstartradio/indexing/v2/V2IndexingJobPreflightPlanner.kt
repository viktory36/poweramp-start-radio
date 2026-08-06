package com.powerampstartradio.indexing.v2

import android.content.Context
import android.content.pm.PackageInfo
import android.os.Build
import com.powerampstartradio.indexing.NewTrackDetector
import com.powerampstartradio.poweramp.TrackNormalization
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.IOException
import java.util.Locale
import java.util.UUID

private fun formatPreflightBytes(bytes: Long): String =
    String.format(Locale.US, "%.1f MiB", bytes / (1024.0 * 1024.0))

private fun formatPreflightByteProgress(
    action: String,
    completedBytes: Long,
    totalBytes: Long,
    initialAction: String = action,
): String = if (completedBytes == 0L) {
    "$initialAction · ${formatPreflightBytes(totalBytes)} total to read"
} else {
    "$action · ${formatPreflightBytes(completedBytes)} of ${formatPreflightBytes(totalBytes)}"
}

data class V2ResolvedTrackSource(
    val track: NewTrackDetector.UnindexedTrack,
    val sourceFile: File,
)

data class V2ResolvedIndexingModels(
    val mertModelFile: File,
    val clamp3AudioModelFile: File,
    val clamp3TextModelFile: File,
    val sentencePieceModelFile: File,
)

data class V2IndexingPreflightRequest(
    val selectedTracks: List<V2ResolvedTrackSource>,
    val models: V2ResolvedIndexingModels,
    val providerSnapshot: V2ProviderPathGroupSnapshot,
    val baseGenerationId: String? = null,
    val rebuildDerivedIndexes: Boolean,
    val executionProfile: V2IndexingExecutionProfile = V2IndexingExecutionProfile.FULL,
    val jobId: String = UUID.randomUUID().toString(),
    /** Stable across resumed durable preflight attempts. */
    val createdAtEpochMs: Long? = null,
    /** Exact durable request occurrences when this plan comes from a persisted intent. */
    val selectedOccurrences: List<V2IndexingPreflightSelection>? = null,
)

data class V2PersistedIndexingJob(
    val jobId: String,
    val specId: String,
    val ledgerDirectory: File,
    val ledgerFile: File,
    val trackCount: Int,
    val createdAtEpochMs: Long,
)

data class V2PreparedIndexingJob(
    val jobId: String,
    val specId: String,
    val planned: List<V2IndexingPreflightSelection>,
    val rejected: List<V2IndexingPreflightRejectedRow>,
)

sealed interface V2IndexingPreflightResolution {
    val jobId: String
    val planned: List<V2IndexingPreflightSelection>
    val rejected: List<V2IndexingPreflightRejectedRow>

    data class Materialized(
        val job: V2PersistedIndexingJob,
        override val planned: List<V2IndexingPreflightSelection>,
        override val rejected: List<V2IndexingPreflightRejectedRow>,
    ) : V2IndexingPreflightResolution {
        override val jobId: String get() = job.jobId
    }

    data class WithoutExecutableRows(
        override val jobId: String,
        val resolvedAtEpochMs: Long,
        override val rejected: List<V2IndexingPreflightRejectedRow>,
    ) : V2IndexingPreflightResolution {
        override val planned: List<V2IndexingPreflightSelection> = emptyList()
    }
}

/**
 * Resolves all semantic identity before returning a service-launchable job ID. Heavy hashing is
 * dispatched to IO; callers launch the foreground executor only after this method succeeds.
 */
class V2IndexingJobPreflightPlanner(
    context: Context,
    private val ledgerDirectory: File = File(context.filesDir, "indexing_v2/jobs"),
    sourceFingerprinter: V2SourceFingerprintProvider = V2ExactSourceFingerprinter(),
    private val audioSpanResolver: V2AudioSpanResolver =
        V2AudioSpanResolver(V2MediaExtractorAudioInspector()),
    private val selectedSourceBinder: V2SelectedSourceBinder = V2SelectedSourceBinder(),
    private val nowEpochMs: () -> Long = System::currentTimeMillis,
    private val observer: V2IndexingPreflightObserver = V2IndexingPreflightObserver.None,
    private val importedRowAuthorizationStore: V2ImportedRowSupersessionAuthorizationStore =
        V2ImportedRowSupersessionAuthorizationStore(ledgerDirectory),
) {
    private val appContext = context.applicationContext
    private val sourceFingerprintProvider = sourceFingerprinter

    suspend fun planAndPersist(
        request: V2IndexingPreflightRequest,
    ): V2PersistedIndexingJob = withContext(Dispatchers.IO) {
        planAndPersistBlocking(request)
    }

    /** Intended for an existing IO executor or deterministic integration tests. */
    fun planAndPersistBlocking(
        request: V2IndexingPreflightRequest,
    ): V2PersistedIndexingJob = when (val result = resolveAndPersistBlocking(request)) {
        is V2IndexingPreflightResolution.Materialized -> result.job
        is V2IndexingPreflightResolution.WithoutExecutableRows ->
            throw IllegalStateException("Preflight resolved without executable rows")
    }

    /**
     * [onPrepared] runs after the immutable ledger is built but before it is published. A durable
     * caller uses this boundary to persist the exact planned/rejected partition and expected spec
     * ID, closing the process-death window between the intent result and ledger publication.
     */
    fun resolveAndPersistBlocking(
        request: V2IndexingPreflightRequest,
        onPrepared: (V2PreparedIndexingJob) -> Unit = {},
    ): V2IndexingPreflightResolution {
        observer.throwIfCancelled()
        validateSelection(request)
        val selectedOccurrences = request.selectedOccurrences ?: request.selectedTracks.map(
            ::selectedOccurrence,
        )
        requireSelectedOccurrencesMatch(request.selectedTracks, selectedOccurrences)
        val rejections = LocalRejectionAccumulator(selectedOccurrences)
        val createdAt = request.createdAtEpochMs ?: nowEpochMs()
        require(createdAt >= 0L) { "preflight creation time is negative" }
        report(
            V2IndexingPreflightPhase.AUDIO_SPANS,
            "Reading audio details and exact track spans",
        )
        val spanResolution = resolveAudioSpans(request)
        spanResolution.excluded.forEach { excluded ->
            rejections.reject(
                excluded.selectedTrack.powerampFileId,
                V2IndexingPreflightFailureCode.CUE_SOURCE_IMAGE,
                "Poweramp row ${excluded.selectedTrack.powerampFileId} is a raw CUE source image",
            )
        }
        spanResolution.rejected.forEach { rejected ->
            rejections.reject(
                rejected.selectedTrack.powerampFileId,
                rejected.code.toPreflightFailureCode(),
                rejected.diagnostic,
            )
        }
        val spansById = spanResolution.resolved.associateBy {
            it.selectedTrack.powerampFileId
        }
        report(
            V2IndexingPreflightPhase.ACTIVE_GENERATION,
            "Reading the active music index identity",
        )
        val activeGeneration = V2IndexGenerationReader.requireActive(appContext.filesDir) { progress ->
            report(
                V2IndexingPreflightPhase.ACTIVE_GENERATION,
                formatPreflightByteProgress(
                    action = "Hashing active music-index file: ${progress.filename}",
                    completedBytes = progress.completedBytes,
                    totalBytes = progress.totalBytes,
                    initialAction = "Opening active music-index file: ${progress.filename} for " +
                        "SHA-256 verification",
                ),
                completedUnits = progress.completedBytes,
                totalUnits = progress.totalBytes,
                unit = V2IndexingPreflightProgressUnit.BYTES,
            )
        }
        if (activeGeneration.manifest.generationId != request.baseGenerationId) {
            throw V2IndexingPreflightException(
                V2IndexingPreflightFailureCode.SOURCE_CHANGED,
                message = "The active embedding generation changed before storage planning",
            )
        }
        val providerSnapshot = resolvedSnapshotEvidence(request.providerSnapshot, spanResolution)
        val providerRowsById = request.providerSnapshot.groups
            .flatMap { it.rows }
            .associateBy { it.powerampFileId }
        val sourceCandidates = request.selectedTracks
            .filterNot { rejections.contains(it.track.powerampFileId) }
            .map { resolved ->
            val providerRow = providerRowsById[resolved.track.powerampFileId]
                ?: throw V2IndexingPreflightException(
                    V2IndexingPreflightFailureCode.PROVIDER_SNAPSHOT_INVALID,
                    powerampFileId = resolved.track.powerampFileId,
                    message = "Provider snapshot omitted selected Poweramp row " +
                        resolved.track.powerampFileId,
                )
            V2SelectedSourceCandidate(
                powerampFileId = resolved.track.powerampFileId,
                providerPathKey = providerRow.physicalPath,
                providerPhysicalPath = providerRow.providerPhysicalPath,
                suppliedSourceFile = resolved.sourceFile,
            )
        }
        report(
            V2IndexingPreflightPhase.SOURCE_BINDINGS,
            "Locating the selected audio files",
        )
        val sourceBindings = bindSourcesWithLocalFailures(
            sourceCandidates,
            rejections,
        )
        val runFingerprinter = V2DeduplicatingSourceFingerprintProvider(sourceFingerprintProvider)
        val selectedInputsById = linkedMapOf<Long, SelectedTrackInput>()
        request.selectedTracks.forEachIndexed { index, resolved ->
            val trackLabel = listOf(resolved.track.artist, resolved.track.title)
                .filter(String::isNotBlank)
                .joinToString(" - ")
            report(
                V2IndexingPreflightPhase.SOURCE_FINGERPRINTS,
                "Binding selected track ${index + 1} of ${request.selectedTracks.size} to its " +
                    "exact source fingerprint" +
                    trackLabel.takeIf(String::isNotBlank)?.let { " · $it" }.orEmpty(),
                completedUnits = index.toLong(),
                totalUnits = request.selectedTracks.size.toLong(),
            )
            val id = resolved.track.powerampFileId
            if (!rejections.contains(id)) {
                val resolvedSpan = spansById[id] ?: throw V2IndexingPreflightException(
                    V2IndexingPreflightFailureCode.INVALID_PLAN,
                    powerampFileId = id,
                    message = "Audio span resolver omitted selected Poweramp row $id",
                )
                val providerRow = providerRowsById[id] ?: throw V2IndexingPreflightException(
                    V2IndexingPreflightFailureCode.PROVIDER_SNAPSHOT_INVALID,
                    powerampFileId = id,
                    message = "Provider snapshot omitted selected Poweramp row $id",
                )
                try {
                    selectedInputsById[id] = resolveTrackInput(
                        resolved = resolved,
                        resolvedSpan = resolvedSpan,
                        providerRow = providerRow,
                        sourceBinding = sourceBindings.getValue(id),
                        runFingerprinter = runFingerprinter,
                        onSourceHashProgress = { completedBytes, totalBytes ->
                            report(
                                V2IndexingPreflightPhase.SOURCE_FINGERPRINTS,
                                formatPreflightByteProgress(
                                    action = "Hashing source for selected track ${index + 1} of " +
                                        request.selectedTracks.size,
                                    completedBytes = completedBytes,
                                    totalBytes = totalBytes,
                                    initialAction = "Opening source for selected track ${index + 1} of " +
                                        "${request.selectedTracks.size} for exact fingerprinting",
                                ) + trackLabel.takeIf(String::isNotBlank)
                                    ?.let { " · $it" }
                                    .orEmpty(),
                                completedUnits = completedBytes,
                                totalUnits = totalBytes,
                                unit = V2IndexingPreflightProgressUnit.BYTES,
                            )
                        },
                    )
                } catch (error: V2IndexingPreflightException) {
                    val semantics = V2IndexingPreflightFailurePolicy.semantics(error.code)
                    if (semantics.scope == V2IndexingPreflightFailureScope.GLOBAL_REQUEST) {
                        throw error
                    }
                    if (error.code == V2IndexingPreflightFailureCode.SOURCE_UNREADABLE) {
                        val sourcePath = providerRow.physicalPath
                        sourceCandidates.filter { it.providerPathKey == sourcePath }.forEach {
                            rejections.reject(
                                it.powerampFileId,
                                error.code,
                                error.message ?: "Selected source is unreadable",
                            )
                            selectedInputsById.remove(it.powerampFileId)
                        }
                    } else {
                        rejections.reject(
                            id,
                            error.code,
                            error.message ?: "Selected occurrence was rejected",
                        )
                    }
                }
            }
            report(
                V2IndexingPreflightPhase.SOURCE_FINGERPRINTS,
                "Binding selected tracks to exact source hashes",
                completedUnits = (index + 1L),
                totalUnits = request.selectedTracks.size.toLong(),
            )
        }
        val selectedInputs = request.selectedTracks.mapNotNull {
            selectedInputsById[it.track.powerampFileId]
                ?.takeUnless { _ -> rejections.contains(it.track.powerampFileId) }
        }
        val plannedIds = selectedInputs.map { it.powerampFileId }.toSet()
        val planned = rejections.planned(plannedIds)
        if (selectedInputs.isEmpty()) {
            rejections.requireCompletePartition(emptySet())
            return V2IndexingPreflightResolution.WithoutExecutableRows(
                jobId = request.jobId,
                resolvedAtEpochMs = nowEpochMs(),
                rejected = rejections.ordered(),
            )
        }

        V2IndexingStoragePolicy.requireCapacity(
            V2IndexingStoragePolicy.estimate(
                active = activeGeneration,
                spans = selectedInputs.map { input ->
                    if (V2UnknownDurationOrdinarySpanPolicy.isUnresolved(
                            input.finalizedAudioSpan,
                        )
                    ) {
                        V2PlannedStorageSpan(
                            mertWindows = null,
                            exactSampleCount24k = null,
                            sourceSampleCount = null,
                        )
                    } else {
                        V2PlannedStorageSpan(
                            mertWindows = input.expectedWork.mertWindows,
                            exactSampleCount24k = input.finalizedAudioSpan.exactSampleCount24k,
                            sourceSampleCount = input.finalizedAudioSpan.sourceSampleCount,
                        )
                    }
                },
                rebuildGraph = request.rebuildDerivedIndexes,
                availableBytes = appContext.filesDir.usableSpace,
            ),
        )
        val modelPolicy = resolveModelPolicy(request.models)
        val embeddingSpec = modelPolicy.receiptEmbeddingSpec
        val textRetrievalSpec = modelPolicy.textRetrievalSpec
        val runtimeFingerprint = resolveRuntimeFingerprint()

        val ledger = try {
            V2IndexingLedgerPlanner.planJob(
                providerSnapshot = providerSnapshot,
                embeddingSpec = embeddingSpec,
                textRetrievalSpec = textRetrievalSpec,
                runtimeFingerprint = runtimeFingerprint,
                selectedTracks = selectedInputs,
                rebuildDerivedIndexes = request.rebuildDerivedIndexes,
                executionProfile = request.executionProfile,
                baseGenerationId = request.baseGenerationId,
                createdAtEpochMs = createdAt,
                jobId = request.jobId,
            )
        } catch (error: Exception) {
            throw V2IndexingPreflightException(
                code = V2IndexingPreflightFailureCode.INVALID_PLAN,
                message = "Unable to construct immutable indexing plan: ${error.message}",
                cause = error,
            )
        }

        val importedRowAuthorization = try {
            V2ImportedRowSupersessionAuthorizer.authorize(
                ledger = ledger,
                activeBase = activeGeneration,
                providerSnapshot = request.providerSnapshot,
            )
        } catch (error: V2ImportedRowAuthorizationException) {
            throw V2IndexingPreflightException(
                code = V2IndexingPreflightFailureCode.SOURCE_CHANGED,
                message = error.message ?: "Imported CUE repair evidence changed",
                cause = error,
            )
        }
        // Publish destructive authorization first. A crash may orphan this sidecar, but can never
        // leave a service-launchable logical-CUE ledger without its immutable authorization.
        try {
            if (importedRowAuthorization != null) {
                importedRowAuthorizationStore.createOrRequireExact(importedRowAuthorization)
            } else {
                importedRowAuthorizationStore.requireAbsent(ledger.jobSpec.jobId)
            }
        } catch (error: V2ImportedRowAuthorizationException) {
            throw V2IndexingPreflightException(
                code = V2IndexingPreflightFailureCode.SOURCE_CHANGED,
                message = error.message ?: "Imported CUE repair authorization changed",
                cause = error,
            )
        } catch (error: IOException) {
            throw V2IndexingPreflightException(
                code = V2IndexingPreflightFailureCode.PERSISTENCE_FAILED,
                message = "Unable to persist imported CUE repair authorization",
                cause = error,
            )
        }

        val ledgerFile = File(ledgerDirectory, "${request.jobId}.json")
        report(
            V2IndexingPreflightPhase.PERSISTING_LEDGER,
            "Saving the indexing job",
        )
        val rejected = rejections.ordered()
        rejections.requireCompletePartition(plannedIds)
        onPrepared(V2PreparedIndexingJob(request.jobId, ledger.jobSpec.specId, planned, rejected))
        observer.throwIfCancelled()
        try {
            val store = AtomicV2IndexingLedgerStore(ledgerDirectory)
            try {
                store.create(ledger)
            } catch (conflict: IndexingLedgerConflictException) {
                val existing = store.require(request.jobId)
                val sameUnstartedPlan = existing.state == IndexingJobState.PLANNED &&
                    existing.revision == 0L &&
                    existing.jobSpec.specId == ledger.jobSpec.specId &&
                    existing.jobSpec.jobId == ledger.jobSpec.jobId &&
                    existing.tracks.map { it.workId } == ledger.tracks.map { it.workId } &&
                    existing.executionProfile == ledger.executionProfile
                if (!sameUnstartedPlan) throw conflict
            }
            check(ledgerFile.isFile) { "Atomic ledger write did not publish ${ledgerFile.name}" }
        } catch (error: Exception) {
            throw V2IndexingPreflightException(
                code = V2IndexingPreflightFailureCode.PERSISTENCE_FAILED,
                message = "Unable to persist indexing job ${request.jobId}: ${error.message}",
                cause = error,
            )
        }

        return V2IndexingPreflightResolution.Materialized(
            job = V2PersistedIndexingJob(
                jobId = request.jobId,
                specId = ledger.jobSpec.specId,
                ledgerDirectory = ledgerDirectory,
                ledgerFile = ledgerFile,
                trackCount = ledger.tracks.size,
                createdAtEpochMs = createdAt,
            ),
            planned = planned,
            rejected = rejected,
        )
    }

    private fun validateSelection(request: V2IndexingPreflightRequest) {
        if (request.selectedTracks.isEmpty()) {
            throw V2IndexingPreflightException(
                V2IndexingPreflightFailureCode.EMPTY_SELECTION,
                message = "At least one track must be selected",
            )
        }
        val duplicateId = request.selectedTracks
            .groupingBy { it.track.powerampFileId }
            .eachCount()
            .entries
            .firstOrNull { it.value > 1 }
            ?.key
        if (duplicateId != null) {
            throw V2IndexingPreflightException(
                V2IndexingPreflightFailureCode.DUPLICATE_POWERAMP_ROW,
                powerampFileId = duplicateId,
                message = "Poweramp row $duplicateId was selected more than once",
            )
        }
    }

    private fun selectedOccurrence(
        resolved: V2ResolvedTrackSource,
    ): V2IndexingPreflightSelection {
        val track = resolved.track
        val path = try {
            requireNotNull(TrackNormalization.normalizePath(track.path)) {
                "selected path is missing"
            }
        } catch (error: Exception) {
            throw V2IndexingPreflightException(
                V2IndexingPreflightFailureCode.SOURCE_CHANGED,
                powerampFileId = track.powerampFileId,
                message = "Selected Poweramp row ${track.powerampFileId} has invalid path evidence",
                cause = error,
            )
        }
        return V2IndexingPreflightSelection(
            powerampFileId = track.powerampFileId,
            providerPhysicalPath = path,
            durationMs = V2ProviderDurationEvidencePolicy.canonicalMs(
                track.durationMs.toLong(),
            ),
            offsetMs = track.offsetMs,
            cueSourceImageFolderId = track.cueFolderId,
        )
    }

    private fun requireSelectedOccurrencesMatch(
        tracks: List<V2ResolvedTrackSource>,
        occurrences: List<V2IndexingPreflightSelection>,
    ) {
        val matches = tracks.size == occurrences.size && tracks.zip(occurrences).all {
                (resolved, selected) ->
            val track = resolved.track
            selected.powerampFileId == track.powerampFileId &&
                selected.providerPhysicalPath == TrackNormalization.normalizePath(track.path) &&
                selected.durationMs == V2ProviderDurationEvidencePolicy.canonicalMs(
                    track.durationMs.toLong(),
                ) &&
                selected.offsetMs == track.offsetMs &&
                selected.cueSourceImageFolderId == track.cueFolderId
        }
        if (!matches) {
            throw V2IndexingPreflightException(
                V2IndexingPreflightFailureCode.INVALID_PLAN,
                message = "Durable selected occurrences do not match planner input",
            )
        }
    }

    private fun bindSourcesWithLocalFailures(
        candidates: List<V2SelectedSourceCandidate>,
        rejections: LocalRejectionAccumulator,
    ): Map<Long, V2SelectedSourceBinding> {
        var remaining = candidates
        while (remaining.isNotEmpty()) {
            try {
                return selectedSourceBinder.bind(remaining)
            } catch (error: V2IndexingPreflightException) {
                val semantics = V2IndexingPreflightFailurePolicy.semantics(error.code)
                if (semantics.scope == V2IndexingPreflightFailureScope.GLOBAL_REQUEST) throw error
                val failedId = error.powerampFileId ?: throw V2IndexingPreflightException(
                    V2IndexingPreflightFailureCode.INVALID_PLAN,
                    message = "Local source-binding failure omitted its selected occurrence",
                    cause = error,
                )
                val failed = remaining.firstOrNull { it.powerampFileId == failedId }
                    ?: throw V2IndexingPreflightException(
                        V2IndexingPreflightFailureCode.INVALID_PLAN,
                        message = "Source-binding failure referred to unselected row $failedId",
                        cause = error,
                    )
                val rejectedGroup = remaining.filter {
                    it.providerPathKey == failed.providerPathKey
                }
                rejectedGroup.forEach { candidate ->
                    rejections.reject(
                        candidate.powerampFileId,
                        error.code,
                        error.message ?: "Selected source cannot be bound",
                    )
                }
                val rejectedIds = rejectedGroup.mapTo(hashSetOf()) { it.powerampFileId }
                remaining = remaining.filterNot { it.powerampFileId in rejectedIds }
            }
        }
        return emptyMap()
    }

    private fun resolveAudioSpans(
        request: V2IndexingPreflightRequest,
    ): V2AudioSpanResolutionBatch = try {
        audioSpanResolver.resolve(
            selectedTracks = request.selectedTracks.map { it.track },
            providerSnapshot = request.providerSnapshot,
            onSourceInspection = { completedSources, totalSources, currentPath ->
                report(
                    phase = V2IndexingPreflightPhase.AUDIO_SPANS,
                    message = currentPath?.let { path ->
                        "Reading audio container ${completedSources + 1} of $totalSources: " +
                            File(path).name
                    } ?: "Read audio details for $completedSources of $totalSources physical sources",
                    completedUnits = completedSources.toLong(),
                    totalUnits = totalSources.toLong(),
                )
            },
        )
    } catch (error: V2AudioSpanResolutionException) {
        val preflightCode = error.code.toPreflightFailureCode()
        if (V2IndexingPreflightFailurePolicy.semantics(preflightCode).scope ==
            V2IndexingPreflightFailureScope.SELECTED_OCCURRENCE
        ) {
            throw V2IndexingPreflightException(
                code = V2IndexingPreflightFailureCode.INVALID_PLAN,
                powerampFileId = error.powerampFileId,
                message = "Audio resolver leaked a local failure instead of returning it: " +
                    error.message,
                cause = error,
            )
        }
        throw V2IndexingPreflightException(
            code = preflightCode,
            powerampFileId = error.powerampFileId,
            message = "Unable to resolve immutable acoustic span: ${error.message}",
            cause = error,
        )
    }

    private fun resolvedSnapshotEvidence(
        snapshot: V2ProviderPathGroupSnapshot,
        resolution: V2AudioSpanResolutionBatch,
    ): PowerampProviderSnapshotEvidence {
        val generation = snapshot.libraryGeneration
        val acquisition = snapshot.acquisitionEvidence
        if (generation.isNullOrBlank() || resolution.libraryGeneration != generation ||
            acquisition == null || !acquisition.cursorExhaustedNormally
        ) {
            throw V2IndexingPreflightException(
                V2IndexingPreflightFailureCode.PROVIDER_SNAPSHOT_INVALID,
                message = "Resolved tracks do not share one verified provider snapshot",
            )
        }
        return PowerampProviderSnapshotEvidence(
            libraryGeneration = generation,
            acquisition = acquisition,
        )
    }

    private fun V2AudioSpanResolutionFailureCode.toPreflightFailureCode():
        V2IndexingPreflightFailureCode = when (this) {
        V2AudioSpanResolutionFailureCode.INVALID_SNAPSHOT_ACQUISITION_EVIDENCE,
        V2AudioSpanResolutionFailureCode.DUPLICATE_PATH_GROUP,
        V2AudioSpanResolutionFailureCode.INCOMPLETE_PATH_GROUP,
        V2AudioSpanResolutionFailureCode.INVALID_PATH_GROUP,
        V2AudioSpanResolutionFailureCode.SELECTED_ROW_NOT_IN_SNAPSHOT,
        -> V2IndexingPreflightFailureCode.PROVIDER_SNAPSHOT_INVALID

        V2AudioSpanResolutionFailureCode.SELECTED_ROW_CHANGED,
        V2AudioSpanResolutionFailureCode.SELECTED_PATH_MISSING,
        -> V2IndexingPreflightFailureCode.SOURCE_CHANGED

        V2AudioSpanResolutionFailureCode.DUPLICATE_SELECTED_ROW,
        -> V2IndexingPreflightFailureCode.DUPLICATE_POWERAMP_ROW

        V2AudioSpanResolutionFailureCode.DURATION_UNAVAILABLE_FOR_PLANNING ->
            V2IndexingPreflightFailureCode.INVALID_SELECTION_EVIDENCE

        V2AudioSpanResolutionFailureCode.INVALID_PROVIDER_SPAN,
        V2AudioSpanResolutionFailureCode.CUE_SPAN_OUT_OF_BOUNDS,
        V2AudioSpanResolutionFailureCode.SAMPLE_COORDINATE_OVERFLOW,
        -> V2IndexingPreflightFailureCode.INVALID_LOGICAL_SPAN

        V2AudioSpanResolutionFailureCode.SOURCE_UNREADABLE ->
            V2IndexingPreflightFailureCode.SOURCE_UNREADABLE
        V2AudioSpanResolutionFailureCode.NO_AUDIO_STREAM ->
            V2IndexingPreflightFailureCode.NO_AUDIO_STREAM
        V2AudioSpanResolutionFailureCode.CONTAINER_INSPECTION_FAILED,
        V2AudioSpanResolutionFailureCode.INVALID_CONTAINER_EVIDENCE,
        V2AudioSpanResolutionFailureCode.UNSUPPORTED_OR_INVALID_CONTAINER,
        -> V2IndexingPreflightFailureCode.UNSUPPORTED_OR_INVALID_AUDIO_CONTAINER
    }

    private class LocalRejectionAccumulator(
        private val selected: List<V2IndexingPreflightSelection>,
    ) {
        private val byId = linkedMapOf<Long, V2IndexingPreflightRejectedRow>()
        private val selectedById = selected.associateBy(V2IndexingPreflightSelection::powerampFileId)

        fun contains(powerampFileId: Long): Boolean = byId.containsKey(powerampFileId)

        fun reject(
            powerampFileId: Long,
            code: V2IndexingPreflightFailureCode,
            diagnostic: String,
        ) {
            val occurrence = selectedById[powerampFileId]
                ?: throw V2IndexingPreflightException(
                    V2IndexingPreflightFailureCode.INVALID_PLAN,
                    message = "Local rejection referred to unselected row $powerampFileId",
                )
            val semantics = V2IndexingPreflightFailurePolicy.requireLocal(code)
            val result = V2IndexingPreflightRejectedRow(
                selected = occurrence,
                code = code,
                disposition = requireNotNull(semantics.disposition),
                retryTrigger = requireNotNull(semantics.retryTrigger),
                diagnostic = diagnostic.trim().take(MAX_REJECTION_DIAGNOSTIC_CHARS)
                    .ifBlank { "Selected occurrence could not be indexed" },
            )
            val previous = byId.putIfAbsent(powerampFileId, result)
            if (previous != null && previous != result) {
                throw V2IndexingPreflightException(
                    V2IndexingPreflightFailureCode.INVALID_PLAN,
                    powerampFileId = powerampFileId,
                    message = "Selected occurrence received conflicting local rejections",
                )
            }
        }

        fun ordered(): List<V2IndexingPreflightRejectedRow> = selected.mapNotNull {
            byId[it.powerampFileId]
        }

        fun planned(plannedIds: Set<Long>): List<V2IndexingPreflightSelection> = selected.filter {
            it.powerampFileId in plannedIds
        }

        fun requireCompletePartition(plannedIds: Set<Long>) {
            val selectedIds = selected.mapTo(linkedSetOf()) { it.powerampFileId }
            val rejectedIds = byId.keys
            if (plannedIds.intersect(rejectedIds).isNotEmpty() ||
                plannedIds + rejectedIds != selectedIds
            ) {
                throw V2IndexingPreflightException(
                    V2IndexingPreflightFailureCode.INVALID_PLAN,
                    message = "Planned and rejected rows do not partition the immutable selection",
                )
            }
        }

        private companion object {
            const val MAX_REJECTION_DIAGNOSTIC_CHARS = 2_048
        }
    }

    private fun resolveTrackInput(
        resolved: V2ResolvedTrackSource,
        resolvedSpan: V2ResolvedAudioSpan,
        providerRow: V2ProviderPathRowEvidence,
        sourceBinding: V2SelectedSourceBinding,
        runFingerprinter: V2SourceFingerprintProvider,
        onSourceHashProgress: (completedBytes: Long, totalBytes: Long) -> Unit,
    ): SelectedTrackInput {
        val track = resolved.track
        val sourceFile = sourceBinding.canonicalSourceFile
        if (!sourceFile.isFile || !sourceFile.canRead() || sourceFile.length() <= 0L) {
            throw sourceFailure(track, sourceFile, null)
        }
        if (sourceBinding.powerampFileId != track.powerampFileId ||
            sourceBinding.providerPathKey != providerRow.physicalPath ||
            resolvedSpan.containerEvidence.physicalPath != providerRow.providerPhysicalPath
        ) {
            throw V2IndexingPreflightException(
                V2IndexingPreflightFailureCode.SOURCE_CHANGED,
                powerampFileId = track.powerampFileId,
                message = "Selected source path disagrees with resolved provider/container evidence",
            )
        }
        val fingerprint = try {
            runFingerprinter.fingerprint(sourceFile, onSourceHashProgress)
        } catch (cancelled: V2IndexingPreflightCancelledException) {
            throw cancelled
        } catch (error: Exception) {
            throw sourceFailure(track, sourceFile, error)
        }

        val canonicalResolvedSpan = resolvedSpan.copy(
            containerEvidence = resolvedSpan.containerEvidence.copy(
                physicalPath = sourceFile.path,
            ),
        )

        return V2IndexingTrackPlanFactory.create(
            resolvedSpan = canonicalResolvedSpan,
            providerRow = providerRow,
            sourceFingerprint = fingerprint,
        )
    }

    private fun resolveModelPolicy(models: V2ResolvedIndexingModels): V2FutureModelPolicy {
        report(
            V2IndexingPreflightPhase.MODEL_FINGERPRINTS,
            "Reading the installed model identity",
        )
        val expectedFiles = mapOf(
            "mert.tflite" to models.mertModelFile,
            "clamp3_audio.tflite" to models.clamp3AudioModelFile,
            "clamp3_text.tflite" to models.clamp3TextModelFile,
            "sentencepiece.bpe.model" to models.sentencePieceModelFile,
        )
        val modelRoot = expectedFiles.values.map { file ->
            runCatching { file.canonicalFile.parentFile }.getOrElse { error ->
                throw modelFailure(file.name, file, error)
            }
        }.distinct().singleOrNull() ?: throw V2IndexingPreflightException(
            V2IndexingPreflightFailureCode.MODEL_UNREADABLE,
            message = "Installed model artifacts do not share one private directory",
        )
        val resolved = try {
            V2CurrentModelPolicyResolver.resolveInstalled(modelRoot) { progress ->
                report(
                    V2IndexingPreflightPhase.MODEL_FINGERPRINTS,
                    formatPreflightByteProgress(
                        action = "Verifying changed installed file ${progress.fileOrdinal} of " +
                            "${progress.fileCount}: ${progress.filename}",
                        completedBytes = progress.completedBytes,
                        totalBytes = progress.totalBytes,
                        initialAction = "Opening changed installed file ${progress.fileOrdinal} of " +
                            "${progress.fileCount}: ${progress.filename} for SHA-256 verification",
                    ),
                    completedUnits = progress.completedBytes,
                    totalUnits = progress.totalBytes,
                    unit = V2IndexingPreflightProgressUnit.BYTES,
                )
            }
        } catch (cancelled: V2IndexingPreflightCancelledException) {
            throw cancelled
        } catch (error: Exception) {
            throw V2IndexingPreflightException(
                V2IndexingPreflightFailureCode.MODEL_UNREADABLE,
                message = "Unable to resolve installed model identity: ${error.message}",
                cause = error,
            )
        }
        expectedFiles.forEach { (filename, requested) ->
            val requestedCanonical = runCatching { requested.canonicalFile }.getOrElse { error ->
                throw modelFailure(filename, requested, error)
            }
            if (resolved.filesByName.getValue(filename) != requestedCanonical) {
                throw modelFailure(filename, requestedCanonical, null)
            }
        }
        return resolved.policy
    }

    private fun resolveRuntimeFingerprint(): IndexingRuntimeFingerprint {
        report(
            V2IndexingPreflightPhase.RUNTIME_FINGERPRINT,
            "Reading the installed app version and decoder identity for crash-safe resume",
        )
        val packageInfo = packageInfo(appContext)
        val appArtifactEvidence = try {
            buildList {
                add(File(appContext.applicationInfo.sourceDir))
                appContext.applicationInfo.splitSourceDirs.orEmpty().mapTo(this, ::File)
            }.map(File::getCanonicalFile)
                .sortedBy(File::getPath)
                .joinToString("\n") { artifact ->
                    require(artifact.isFile && artifact.canRead() && artifact.length() > 0L) {
                        "installed app artifact is unreadable: $artifact"
                    }
                    "${artifact.path}:${artifact.length()}:${artifact.lastModified()}"
                }
        } catch (error: Exception) {
            throw V2IndexingPreflightException(
                V2IndexingPreflightFailureCode.APP_ARTIFACT_UNREADABLE,
                message = "Unable to read installed app file metadata: ${error.message}",
                cause = error,
            )
        }
        val versionCode = packageInfo.longVersionCodeCompat()
        val appIdentityEvidence = listOf(
            "package-runtime-stat-v2",
            appContext.packageName,
            versionCode.toString(),
            packageInfo.lastUpdateTime.toString(),
            appArtifactEvidence,
        ).joinToString("|")
        val platform = Build.FINGERPRINT.ifBlank {
            "android-${Build.VERSION.SDK_INT}-${Build.MANUFACTURER}-${Build.MODEL}"
        }
        val decoderEvidence = listOf(
            "android-mediacodec-runtime-v1",
            Build.VERSION.SDK_INT.toString(),
            platform,
            Build.HARDWARE,
            Build.SUPPORTED_ABIS.joinToString(","),
        ).joinToString("|")
        return IndexingRuntimeFingerprint(
            appVersionCode = versionCode,
            appBuildId = "package-runtime-stat-v2:" +
                V2FileSha256.digestText(appIdentityEvidence),
            decoderRuntimeId = "decoder-runtime-sha256-v1:${V2FileSha256.digestText(decoderEvidence)}",
            platformFingerprint = platform,
        )
    }

    private fun report(
        phase: V2IndexingPreflightPhase,
        message: String,
        completedUnits: Long? = null,
        totalUnits: Long? = null,
        unit: V2IndexingPreflightProgressUnit? = null,
    ) {
        observer.throwIfCancelled()
        observer.onProgress(
            V2IndexingPreflightProgress(
                phase = phase,
                message = message,
                completedUnits = completedUnits,
                totalUnits = totalUnits,
                unit = unit,
            ),
        )
        observer.throwIfCancelled()
    }

    private fun sourceFailure(
        track: NewTrackDetector.UnindexedTrack,
        file: File,
        cause: Throwable?,
    ) = sourceFailure(track.powerampFileId, file, cause)

    private fun sourceFailure(
        powerampFileId: Long,
        file: File,
        cause: Throwable?,
    ) = V2IndexingPreflightException(
        V2IndexingPreflightFailureCode.SOURCE_UNREADABLE,
        powerampFileId = powerampFileId,
        message = "Source is not readable for Poweramp row $powerampFileId: $file",
        cause = cause,
    )

    private fun modelFailure(name: String, file: File, cause: Throwable?) =
        V2IndexingPreflightException(
            V2IndexingPreflightFailureCode.MODEL_UNREADABLE,
            message = "Resolved $name model is not readable: $file",
            cause = cause,
        )

    @Suppress("DEPRECATION")
    private fun packageInfo(context: Context): PackageInfo =
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            context.packageManager.getPackageInfo(
                context.packageName,
                android.content.pm.PackageManager.PackageInfoFlags.of(0L),
            )
        } else {
            context.packageManager.getPackageInfo(context.packageName, 0)
        }

    @Suppress("DEPRECATION")
    private fun PackageInfo.longVersionCodeCompat(): Long =
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) longVersionCode
        else versionCode.toLong()
}
