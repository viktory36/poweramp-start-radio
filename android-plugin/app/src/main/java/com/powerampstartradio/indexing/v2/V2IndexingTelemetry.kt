package com.powerampstartradio.indexing.v2

import com.powerampstartradio.indexing.V2GraphWorkPlan
import java.io.File

/** Only decode throughput depends on the source codec/container key. */
object V2MeasuredWorkKeyPolicy {
    fun codecClass(stage: V2MeasuredWorkStage, sourceMime: String?): String? =
        sourceMime
            ?.trim()
            ?.lowercase()
            ?.takeIf(String::isNotEmpty)
            ?.takeIf { stage == V2MeasuredWorkStage.PCM_24K_SAMPLES }

    fun hasEtaRate(stage: V2MeasuredWorkStage): Boolean = when (stage) {
        // Input graph extraction has no start callback, so its bytes cannot share a truthful rate
        // with the later database/file outputs yet.
        V2MeasuredWorkStage.GRAPH_BINARY_BYTES,
        V2MeasuredWorkStage.ACTIVATION_TRACKS,
        V2MeasuredWorkStage.GRAPH_NODES,
        V2MeasuredWorkStage.PCM_NORMALIZATION_SAMPLES,
        V2MeasuredWorkStage.PCM_CACHE_BYTES,
        V2MeasuredWorkStage.POWERAMP_LIBRARY_ROWS,
        V2MeasuredWorkStage.INDEXING_MODEL_FILES,
        V2MeasuredWorkStage.SOURCE_AUDIO_HASH,
        V2MeasuredWorkStage.PRIVATE_INDEX_COPY,
        V2MeasuredWorkStage.SIMILARITY_GRAPH_SETUP,
        V2MeasuredWorkStage.MUSIC_INDEX_PUBLICATION,
        V2MeasuredWorkStage.SAVED_ARTIFACT_INSPECTION,
        V2MeasuredWorkStage.RECOMMENDATION_RESOURCE_HANDOFF,
        V2MeasuredWorkStage.STAGING_INDEX_INSPECTION,
        -> false

        else -> true
    }
}

/**
 * Turns the executor's real start/progress/end events into estimator samples. A terminal event
 * without a matching start is deliberately ignored, so cached and aliased work cannot be learned
 * as if inference had run.
 */
class V2IndexingEventRateRecorder(
    private val estimator: V2StageAwareWorkEstimator,
) {
    private data class EventKey(
        val jobId: String,
        val workId: String?,
        val stage: V2MeasuredWorkStage,
    )

    private data class Observation(
        val profile: V2IndexingExecutionProfile,
        val codecClass: String?,
        val completedUnits: Long,
        val observedAtElapsedMs: Long,
    )

    private data class PcmObservation(
        val jobId: String,
        val powerampFileId: Long,
        val profile: V2IndexingExecutionProfile,
        val codecClass: String?,
        val observedAtElapsedMs: Long,
    )

    private val observations = mutableMapOf<EventKey, Observation>()
    private var activeKey: EventKey? = null
    private var pcmObservation: PcmObservation? = null

    @Synchronized
    fun onEvent(
        event: V2IndexingExecutorEvent,
        profile: V2IndexingExecutionProfile,
        sourceMime: String?,
        observedAtElapsedMs: Long,
    ): Boolean {
        if (event.stage == V2MeasuredWorkStage.PCM_24K_SAMPLES) {
            observations.clear()
            activeKey = null
            return onPcmEvent(event, profile, sourceMime, observedAtElapsedMs)
        }
        if (event.stage == V2MeasuredWorkStage.PCM_NORMALIZATION_SAMPLES ||
            event.stage == V2MeasuredWorkStage.PCM_CACHE_BYTES
        ) {
            // These are visible substages of the same materialization measurement. Their time is
            // intentionally included in the exact PCM completion sample.
            observations.clear()
            activeKey = null
            return false
        }
        pcmObservation = null
        val completed = event.completedUnits ?: return false
        val total = event.totalUnits ?: return false
        val key = EventKey(event.jobId, event.workId, event.stage)
        if (activeKey != key) {
            // Time spent in another track or stage cannot be charged to a stale interval. This is
            // important for graph binary I/O, whose input and output events bracket dot products.
            observations.clear()
            activeKey = key
        }
        if (completed !in 0L..total ||
            total <= 0L ||
            !V2MeasuredWorkKeyPolicy.hasEtaRate(event.stage)
        ) {
            observations.remove(key)
            activeKey = null
            return false
        }

        val codecClass = V2MeasuredWorkKeyPolicy.codecClass(event.stage, sourceMime)
        val previous = observations[key]
        var recorded = false
        if (previous != null &&
            previous.profile == profile &&
            previous.codecClass == codecClass &&
            completed > previous.completedUnits &&
            observedAtElapsedMs > previous.observedAtElapsedMs
        ) {
            estimator.recordCompleted(
                stage = event.stage,
                profile = profile,
                completedUnits = completed - previous.completedUnits,
                activeDurationMs = observedAtElapsedMs - previous.observedAtElapsedMs,
                codecClass = codecClass,
            )
            recorded = true
        }

        if (completed == total) {
            observations.remove(key)
            activeKey = null
        } else {
            observations[key] = Observation(
                profile = profile,
                codecClass = codecClass,
                completedUnits = completed,
                observedAtElapsedMs = observedAtElapsedMs,
            )
        }
        return recorded
    }

    private fun onPcmEvent(
        event: V2IndexingExecutorEvent,
        profile: V2IndexingExecutionProfile,
        sourceMime: String?,
        observedAtElapsedMs: Long,
    ): Boolean {
        val measurement = event.pcmRateMeasurement
        if (measurement == null) {
            pcmObservation = null
            return false
        }
        val codecClass = V2MeasuredWorkKeyPolicy.codecClass(event.stage, sourceMime)
        val previous = pcmObservation
        val sameMeasurement = previous?.jobId == event.jobId &&
            previous.powerampFileId == measurement.powerampFileId
        return when (measurement.point) {
            V2PcmRateMeasurementPoint.MATERIALIZATION_STARTED -> {
                pcmObservation = if (
                    event.completedUnits == 0L ||
                    (event.completedUnits == null && event.totalUnits == null)
                ) {
                    PcmObservation(
                        jobId = event.jobId,
                        powerampFileId = measurement.powerampFileId,
                        profile = profile,
                        codecClass = codecClass,
                        observedAtElapsedMs = observedAtElapsedMs,
                    )
                } else {
                    null
                }
                false
            }

            V2PcmRateMeasurementPoint.MATERIALIZATION_PROGRESS -> {
                if (!sameMeasurement) pcmObservation = null
                false
            }

            V2PcmRateMeasurementPoint.MATERIALIZATION_COMPLETED_EXACT -> {
                pcmObservation = null
                val exactUnits = checkNotNull(measurement.exactSampleCount24k)
                if (sameMeasurement &&
                    previous.profile == profile &&
                    previous.codecClass == codecClass &&
                    event.completedUnits == exactUnits &&
                    event.totalUnits == exactUnits &&
                    observedAtElapsedMs > previous.observedAtElapsedMs
                ) {
                    estimator.recordCompleted(
                        stage = event.stage,
                        profile = profile,
                        completedUnits = exactUnits,
                        activeDurationMs = observedAtElapsedMs - previous.observedAtElapsedMs,
                        codecClass = codecClass,
                    )
                    true
                } else {
                    false
                }
            }

            V2PcmRateMeasurementPoint.VERIFIED_CACHE_REUSED_EXACT -> {
                // Revalidation is correctness work, not a second decode/resample observation.
                pcmObservation = null
                false
            }
        }
    }

    @Synchronized
    fun clear() {
        observations.clear()
        activeKey = null
        pcmObservation = null
    }

}

fun interface V2VerifiedPcmReceiptVerifier {
    fun isVerified(
        jobArtifactDirectory: File,
        jobId: String,
        descriptor: SelectedTrackDescriptor,
    ): Boolean
}

/**
 * Resolves durable PCM completion for event-free snapshots. Verification is cached only while the
 * descriptor and both receipt files retain the same identity-relevant file stamps.
 */
class V2VerifiedPcmCompletionResolver(
    private val verifier: V2VerifiedPcmReceiptVerifier = defaultVerifier(),
) {
    private data class FileStamp(
        val isFile: Boolean,
        val byteLength: Long,
        val lastModifiedEpochMs: Long,
    )

    private data class VerificationKey(
        val artifactDirectory: String,
        val jobId: String,
        val descriptor: SelectedTrackDescriptor,
        val pcm: FileStamp,
        val receipt: FileStamp,
    )

    private val cached = mutableMapOf<VerificationKey, Boolean>()

    @Synchronized
    fun verifiedWorkIds(
        ledger: IndexingJobLedger,
        jobArtifactDirectory: File,
    ): Set<String> {
        val currentKeys = linkedSetOf<VerificationKey>()
        val verified = linkedSetOf<String>()
        val pcmDirectory = File(jobArtifactDirectory, PCM_CACHE_DIRECTORY)
        val cachedPowerampFileIds = pcmDirectory.listFiles()
            ?.asSequence()
            ?.mapNotNull { file ->
                RECEIPT_FILENAME.matchEntire(file.name)?.groupValues?.get(1)?.toLongOrNull()
            }
            ?.toSet()
            .orEmpty()
        if (cachedPowerampFileIds.isEmpty()) {
            cached.clear()
            return emptySet()
        }
        val trackByWorkId = ledger.tracks.associateBy(IndexingTrackLedger::workId)
        V2CanonicalAcousticWorkPlanner.groups(ledger.jobSpec).forEach { group ->
            group.members.forEach memberLoop@ { descriptor ->
                if (descriptor.powerampFileId !in cachedPowerampFileIds) return@memberLoop
                val track = trackByWorkId.getValue(descriptor.workId)
                if (track.verifiedArtifacts.any {
                        it.kind == VerifiedArtifactKind.MERT_FEATURES
                    }
                ) return@memberLoop
                val pcm = File(pcmDirectory, "${descriptor.powerampFileId}.pcm-24k-f32.bin")
                val receipt = File(pcmDirectory, "${descriptor.powerampFileId}.receipt.json")
                val key = VerificationKey(
                    artifactDirectory = jobArtifactDirectory.canonicalPath,
                    jobId = ledger.jobSpec.jobId,
                    descriptor = descriptor,
                    pcm = pcm.stamp(),
                    receipt = receipt.stamp(),
                )
                currentKeys += key
                val isVerified = cached.getOrPut(key) {
                    key.pcm.isFile && key.receipt.isFile && verifier.isVerified(
                        jobArtifactDirectory,
                        ledger.jobSpec.jobId,
                        descriptor,
                    )
                }
                if (isVerified) verified += descriptor.workId
            }
        }
        cached.keys.retainAll(currentKeys)
        return verified
    }

    @Synchronized
    fun clear() = cached.clear()

    private fun File.stamp() = FileStamp(
        isFile = isFile,
        byteLength = if (isFile) length() else 0L,
        lastModifiedEpochMs = if (exists()) lastModified() else 0L,
    )

    private companion object {
        const val PCM_CACHE_DIRECTORY = "pcm-cache-v1"
        val RECEIPT_FILENAME = Regex("^(-?\\d+)\\.receipt\\.json$")

        fun defaultVerifier(): V2VerifiedPcmReceiptVerifier {
            val store = V2VerifiedPcmCacheStore()
            return V2VerifiedPcmReceiptVerifier { root, jobId, descriptor ->
                store.loadVerified(root, jobId, descriptor) != null
            }
        }
    }
}

/**
 * Keeps the receipt evidence resolved once at runner startup available to hot progress snapshots.
 * New exact PCM completions are tracked from the executor's verified-completion event, while a
 * durable MERT artifact makes the process-local receipt evidence redundant.
 */
class V2VerifiedPcmProgressTracker(initialVerifiedWorkIds: Set<String>) {
    private val verifiedWorkIds = initialVerifiedWorkIds.toMutableSet()
    private var lastPrunedLedgerRevision: Long? = null

    @Synchronized
    fun snapshot(
        ledger: IndexingJobLedger,
        event: V2IndexingExecutorEvent? = null,
    ): Set<String> {
        if (lastPrunedLedgerRevision != ledger.revision) {
            if (verifiedWorkIds.isNotEmpty()) {
                val tracks = ledger.tracks.associateBy(IndexingTrackLedger::workId)
                verifiedWorkIds.removeAll { workId ->
                    val track = tracks[workId] ?: return@removeAll true
                    track.verifiedArtifacts.any { it.kind == VerifiedArtifactKind.MERT_FEATURES }
                }
            }
            lastPrunedLedgerRevision = ledger.revision
        }
        if (event?.stage == V2MeasuredWorkStage.PCM_24K_SAMPLES) {
            reconcilePcmEvent(event)
        }
        return verifiedWorkIds.toSet()
    }

    private fun reconcilePcmEvent(event: V2IndexingExecutorEvent) {
        val workId = event.workId ?: return
        val measurement = event.pcmRateMeasurement ?: return
        when (measurement.point) {
            V2PcmRateMeasurementPoint.MATERIALIZATION_STARTED -> verifiedWorkIds.remove(workId)

            V2PcmRateMeasurementPoint.MATERIALIZATION_COMPLETED_EXACT,
            V2PcmRateMeasurementPoint.VERIFIED_CACHE_REUSED_EXACT,
            -> {
                val exact = measurement.exactSampleCount24k ?: return
                if (exact > 0L && event.completedUnits == exact && event.totalUnits == exact) {
                    verifiedWorkIds += workId
                }
            }

            V2PcmRateMeasurementPoint.MATERIALIZATION_PROGRESS -> Unit
        }
    }
}

enum class V2IndexingEtaScope {
    WHOLE_JOB,
    MEASURED_STAGES_ONLY,
}

/** Work that has no truthful physical-unit callbacks yet and is therefore excluded from ETA. */
enum class V2UnmeasuredIndexingWork {
    UNKNOWN_DURATION_AUDIO_WORK,
    VALIDATION_AND_FINAL_PUBLICATION,
    DERIVED_GRAPH_WITHOUT_STRUCTURED_PLAN,
    DERIVED_GRAPH_BINARY_IO_WITHOUT_COMPLETE_BOUNDARIES,
}

data class V2OverallStageWorkSnapshot(
    val stage: V2MeasuredWorkStage,
    val codecClass: String?,
    val completedUnits: Long,
    val remainingUnits: Long,
    val abandonedUnits: Long,
) {
    init {
        require(completedUnits >= 0L)
        require(remainingUnits >= 0L)
        require(abandonedUnits >= 0L)
        require(
            codecClass == V2MeasuredWorkKeyPolicy.codecClass(stage, codecClass),
        ) { "only PCM work may carry a normalized codec class" }
    }

    val plannedUnits: Long
        get() = Math.addExact(Math.addExact(completedUnits, remainingUnits), abandonedUnits)

    val resolvedUnits: Long
        get() = Math.addExact(completedUnits, abandonedUnits)
}

data class V2IndexingEtaCoverageSnapshot(
    val scope: V2IndexingEtaScope,
    val measuredRemainingStages: Set<V2MeasuredWorkStage>,
    val omittedRemainingWork: Set<V2UnmeasuredIndexingWork>,
) {
    val coversWholeJob: Boolean
        get() = scope == V2IndexingEtaScope.WHOLE_JOB && omittedRemainingWork.isEmpty()
}

/** Immutable job-level work vector; unlike physical units are never collapsed into one scalar. */
data class V2IndexingOverallWorkSnapshot(
    val resolvedTracks: Int,
    val totalTracks: Int,
    val stages: List<V2OverallStageWorkSnapshot>,
    val omittedRemainingWork: Set<V2UnmeasuredIndexingWork>,
) {
    init {
        require(resolvedTracks in 0..totalTracks)
        require(stages.distinctBy { it.stage to it.codecClass }.size == stages.size)
    }

    val etaCoverage: V2IndexingEtaCoverageSnapshot
        get() = V2IndexingEtaCoverageSnapshot(
            scope = if (omittedRemainingWork.isEmpty()) {
                V2IndexingEtaScope.WHOLE_JOB
            } else {
                V2IndexingEtaScope.MEASURED_STAGES_ONLY
            },
            measuredRemainingStages = stages
                .asSequence()
                .filter {
                    it.remainingUnits > 0L && V2MeasuredWorkKeyPolicy.hasEtaRate(it.stage)
                }
                .mapTo(linkedSetOf()) { it.stage },
            omittedRemainingWork = omittedRemainingWork,
        )

    fun remainingMeasuredWork(): List<V2RemainingStageWork> = stages.mapNotNull { work ->
        work.remainingUnits.takeIf {
            it > 0L && V2MeasuredWorkKeyPolicy.hasEtaRate(work.stage)
        }?.let { remaining ->
            V2RemainingStageWork(work.stage, remaining, work.codecClass)
        }
    }
}

/** Builds one truthful physical-work snapshot from the durable ledger plus the current event. */
object V2IndexingOverallWorkPlanner {
    private data class StageKey(
        val stage: V2MeasuredWorkStage,
        val codecClass: String?,
    )

    private data class MutableStageWork(
        var completed: Long = 0L,
        var remaining: Long = 0L,
        var abandoned: Long = 0L,
    )

    fun snapshot(
        ledger: IndexingJobLedger,
        event: V2IndexingExecutorEvent?,
        graphPlan: V2GraphWorkPlan?,
        verifiedPcmWorkIds: Set<String> = emptySet(),
    ): V2IndexingOverallWorkSnapshot {
        val trackByWorkId = ledger.tracks.associateBy(IndexingTrackLedger::workId)
        val stages = linkedMapOf<StageKey, MutableStageWork>()
        val jobCancelled = ledger.state == IndexingJobState.CANCELLED
        var hasRunnableUnknownDurationWork = false

        V2CanonicalAcousticWorkPlanner.groups(ledger.jobSpec).forEach { group ->
            val descriptor = group.canonical
            val memberIds = group.members.mapTo(linkedSetOf()) { it.workId }
            val groupTracks = group.members.map { trackByWorkId.getValue(it.workId) }
            val artifacts = groupTracks.flatMapTo(mutableSetOf()) { track ->
                track.verifiedArtifacts.map { it.kind }
            }
            val abandoned = jobCancelled || groupTracks.all { track ->
                track.state == IndexingTrackState.BLOCKED_FAILURE ||
                    track.state == IndexingTrackState.SKIPPED_BY_USER
            }
            if (V2UnknownDurationOrdinarySpanPolicy.isUnresolved(
                    descriptor.finalizedAudioSpan,
                ) && !abandoned
            ) {
                hasRunnableUnknownDurationWork = true
            }
            val current = event.takeIf { it?.workId in memberIds }
            val pcmPlanned = descriptor.finalizedAudioSpan.exactSampleCount24k
            val mertPlanned = descriptor.expectedWork.mertWindows.toLong()
            val clampPlanned = descriptor.expectedWork.clampSegments.toLong()
            val mertDurable = VerifiedArtifactKind.MERT_FEATURES in artifacts
            val clampDurable = VerifiedArtifactKind.CLAMP_VECTOR in artifacts
            val pcmDurable = mertDurable || memberIds.any { it in verifiedPcmWorkIds }

            val pcmLive = when (current?.stage) {
                V2MeasuredWorkStage.PCM_24K_SAMPLES -> current.validPartial(pcmPlanned)
                V2MeasuredWorkStage.MERT_WINDOWS,
                V2MeasuredWorkStage.CLAMP_SEGMENTS,
                V2MeasuredWorkStage.DATABASE_COMMITS,
                V2MeasuredWorkStage.PCM_NORMALIZATION_SAMPLES,
                V2MeasuredWorkStage.PCM_CACHE_BYTES,
                -> pcmPlanned
                else -> 0L
            }
            addTrackWork(
                stages = stages,
                stage = V2MeasuredWorkStage.PCM_24K_SAMPLES,
                codecClass = V2MeasuredWorkKeyPolicy.codecClass(
                    V2MeasuredWorkStage.PCM_24K_SAMPLES,
                    descriptor.finalizedAudioSpan.container.mime,
                ),
                planned = pcmPlanned,
                completed = if (pcmDurable) pcmPlanned else pcmLive,
                abandoned = abandoned && !pcmDurable,
            )

            val mertLive = when (current?.stage) {
                V2MeasuredWorkStage.MERT_WINDOWS -> current.validPartial(mertPlanned)
                V2MeasuredWorkStage.CLAMP_SEGMENTS,
                V2MeasuredWorkStage.DATABASE_COMMITS,
                -> mertPlanned
                else -> 0L
            }
            addTrackWork(
                stages = stages,
                stage = V2MeasuredWorkStage.MERT_WINDOWS,
                codecClass = null,
                planned = mertPlanned,
                completed = if (mertDurable) mertPlanned else mertLive,
                abandoned = abandoned && !mertDurable,
            )

            val clampLive = when (current?.stage) {
                V2MeasuredWorkStage.CLAMP_SEGMENTS -> current.validPartial(clampPlanned)
                V2MeasuredWorkStage.DATABASE_COMMITS -> clampPlanned
                else -> 0L
            }
            addTrackWork(
                stages = stages,
                stage = V2MeasuredWorkStage.CLAMP_SEGMENTS,
                codecClass = null,
                planned = clampPlanned,
                completed = if (clampDurable) clampPlanned else clampLive,
                abandoned = abandoned && !clampDurable,
            )
        }

        ledger.jobSpec.tracks.forEach { descriptor ->
            val track = trackByWorkId.getValue(descriptor.workId)
            val commitDurable = track.verifiedArtifacts.any {
                it.kind == VerifiedArtifactKind.DATABASE_COMMIT
            }
            val abandoned = !commitDurable && (
                jobCancelled ||
                    track.state == IndexingTrackState.BLOCKED_FAILURE ||
                    track.state == IndexingTrackState.SKIPPED_BY_USER
                )
            val live = event
                .takeIf {
                    it?.workId == descriptor.workId &&
                        it.stage == V2MeasuredWorkStage.DATABASE_COMMITS
                }
                ?.validPartial(1L)
                ?: 0L
            addTrackWork(
                stages = stages,
                stage = V2MeasuredWorkStage.DATABASE_COMMITS,
                codecClass = null,
                planned = 1L,
                completed = if (commitDurable) 1L else live,
                abandoned = abandoned,
            )
        }

        graphPlan?.let { plan -> addGraphWork(stages, ledger, event, plan) }

        val progress = V2IndexingProgress.from(ledger)
        val stageSnapshots = stages.entries
            .map { (key, value) ->
                V2OverallStageWorkSnapshot(
                    stage = key.stage,
                    codecClass = key.codecClass,
                    completedUnits = value.completed,
                    remainingUnits = value.remaining,
                    abandonedUnits = value.abandoned,
                )
            }
            .sortedWith(compareBy({ it.stage.ordinal }, { it.codecClass ?: "" }))
        val terminal = ledger.state == IndexingJobState.COMPLETE ||
            ledger.state == IndexingJobState.CANCELLED
        val omissions = if (terminal) {
            emptySet()
        } else {
            buildSet {
                if (hasRunnableUnknownDurationWork) {
                    add(V2UnmeasuredIndexingWork.UNKNOWN_DURATION_AUDIO_WORK)
                }
                add(V2UnmeasuredIndexingWork.VALIDATION_AND_FINAL_PUBLICATION)
                if (ledger.jobSpec.rebuildDerivedIndexes &&
                    graphPlan == null &&
                    ledger.state != IndexingJobState.ACTIVATING
                ) {
                    add(V2UnmeasuredIndexingWork.DERIVED_GRAPH_WITHOUT_STRUCTURED_PLAN)
                }
                if (stageSnapshots.any {
                        it.stage == V2MeasuredWorkStage.GRAPH_BINARY_BYTES &&
                            it.remainingUnits > 0L
                    }
                ) {
                    add(
                        V2UnmeasuredIndexingWork
                            .DERIVED_GRAPH_BINARY_IO_WITHOUT_COMPLETE_BOUNDARIES,
                    )
                }
            }
        }
        return V2IndexingOverallWorkSnapshot(
            resolvedTracks = progress.resolvedTracks,
            totalTracks = progress.totalTracks,
            stages = stageSnapshots,
            omittedRemainingWork = omissions,
        )
    }

    private fun addTrackWork(
        stages: MutableMap<StageKey, MutableStageWork>,
        stage: V2MeasuredWorkStage,
        codecClass: String?,
        planned: Long,
        completed: Long,
        abandoned: Boolean,
    ) {
        require(planned >= 0L)
        val boundedCompleted = completed.coerceIn(0L, planned)
        val abandonedUnits = if (abandoned) planned - boundedCompleted else 0L
        val remaining = planned - boundedCompleted - abandonedUnits
        add(stages, stage, codecClass, boundedCompleted, remaining, abandonedUnits)
    }

    private fun addGraphWork(
        stages: MutableMap<StageKey, MutableStageWork>,
        ledger: IndexingJobLedger,
        event: V2IndexingExecutorEvent?,
        plan: V2GraphWorkPlan,
    ) {
        val complete = ledger.state == IndexingJobState.ACTIVATING ||
            ledger.state == IndexingJobState.COMPLETE
        val abandoned = ledger.state == IndexingJobState.CANCELLED
        val currentStage = event?.stage
        val currentPartial = event?.completedUnits

        val rowsCompleted = when {
            complete -> plan.embeddingRows
            currentStage == V2MeasuredWorkStage.DERIVED_EMBEDDING_ROWS ->
                event.validPartial(plan.embeddingRows)
            currentStage in setOf(
                V2MeasuredWorkStage.GRAPH_SIMILARITY_DOT_PRODUCTS,
                V2MeasuredWorkStage.GRAPH_BINARY_BYTES,
                V2MeasuredWorkStage.ACTIVATION_TRACKS,
            ) -> plan.embeddingRows
            else -> 0L
        }
        addTrackWork(
            stages,
            V2MeasuredWorkStage.DERIVED_EMBEDDING_ROWS,
            null,
            plan.embeddingRows,
            rowsCompleted,
            abandoned,
        )

        val dotsCompleted = when {
            complete -> plan.similarityDotProducts
            currentStage == V2MeasuredWorkStage.GRAPH_SIMILARITY_DOT_PRODUCTS ->
                event.validPartial(plan.similarityDotProducts)
            currentStage == V2MeasuredWorkStage.GRAPH_BINARY_BYTES &&
                currentPartial != null && currentPartial > plan.graphBinaryInputBytes ->
                plan.similarityDotProducts
            currentStage == V2MeasuredWorkStage.ACTIVATION_TRACKS -> plan.similarityDotProducts
            else -> 0L
        }
        addTrackWork(
            stages,
            V2MeasuredWorkStage.GRAPH_SIMILARITY_DOT_PRODUCTS,
            null,
            plan.similarityDotProducts,
            dotsCompleted,
            abandoned,
        )

        val bytesCompleted = when {
            complete -> plan.graphBinaryBytes
            currentStage == V2MeasuredWorkStage.GRAPH_BINARY_BYTES ->
                event.validPartial(plan.graphBinaryBytes)
            currentStage == V2MeasuredWorkStage.GRAPH_SIMILARITY_DOT_PRODUCTS ->
                plan.graphBinaryInputBytes
            currentStage == V2MeasuredWorkStage.ACTIVATION_TRACKS -> plan.graphBinaryBytes
            else -> 0L
        }
        addTrackWork(
            stages,
            V2MeasuredWorkStage.GRAPH_BINARY_BYTES,
            null,
            plan.graphBinaryBytes,
            bytesCompleted,
            abandoned,
        )
    }

    private fun add(
        stages: MutableMap<StageKey, MutableStageWork>,
        stage: V2MeasuredWorkStage,
        codecClass: String?,
        completed: Long,
        remaining: Long,
        abandoned: Long,
    ) {
        val key = StageKey(stage, V2MeasuredWorkKeyPolicy.codecClass(stage, codecClass))
        val value = stages.getOrPut(key, ::MutableStageWork)
        value.completed = Math.addExact(value.completed, completed)
        value.remaining = Math.addExact(value.remaining, remaining)
        value.abandoned = Math.addExact(value.abandoned, abandoned)
    }

    private fun V2IndexingExecutorEvent?.validPartial(planned: Long): Long {
        if (this == null || planned <= 0L) return 0L
        val completed = completedUnits ?: return 0L
        val total = totalUnits ?: return 0L
        if (completed < 0L || total <= 0L || completed > total) return 0L
        return completed.coerceAtMost(planned)
    }
}
