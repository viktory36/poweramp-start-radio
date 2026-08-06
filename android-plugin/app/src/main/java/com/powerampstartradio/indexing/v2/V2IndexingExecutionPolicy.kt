package com.powerampstartradio.indexing.v2

import kotlin.math.ceil
import kotlin.math.floor

/** Scheduling may change latency and resource pressure, never preprocessing or output bytes. */
enum class V2IndexingExecutionProfile {
    FULL,
    BALANCED,
    BACKGROUND,
}

data class V2IndexingExecutionSchedule(
    /** Identical across profiles so lossy decoder seek/preroll cannot change PCM bytes. */
    val pcmChunkDurationMs: Long,
    val yieldAfterCompletedUnitMs: Long,
    val threadPriority: Int,
)

object V2IndexingExecutionPolicies {
    const val PINNED_PREPROCESSING_SPEC_ID = V2IndexingWorkPolicy.PREPROCESSING_SPEC_ID
    const val BYTE_STABLE_PCM_CHUNK_DURATION_MS = 60_000L

    fun schedule(profile: V2IndexingExecutionProfile): V2IndexingExecutionSchedule = when (profile) {
        V2IndexingExecutionProfile.FULL -> V2IndexingExecutionSchedule(
            pcmChunkDurationMs = BYTE_STABLE_PCM_CHUNK_DURATION_MS,
            yieldAfterCompletedUnitMs = 0L,
            threadPriority = -2,
        )
        V2IndexingExecutionProfile.BALANCED -> V2IndexingExecutionSchedule(
            pcmChunkDurationMs = BYTE_STABLE_PCM_CHUNK_DURATION_MS,
            yieldAfterCompletedUnitMs = 8L,
            threadPriority = 0,
        )
        V2IndexingExecutionProfile.BACKGROUND -> V2IndexingExecutionSchedule(
            pcmChunkDurationMs = BYTE_STABLE_PCM_CHUNK_DURATION_MS,
            yieldAfterCompletedUnitMs = 40L,
            threadPriority = 10,
        )
    }

    fun requirePinnedByteContract(spec: EmbeddingSpecFingerprint) {
        require(spec.preprocessingSpecId == PINNED_PREPROCESSING_SPEC_ID) {
            "V2 executor requires $PINNED_PREPROCESSING_SPEC_ID"
        }
        require(spec.decoderPolicyId == V2IndexingWorkPolicy.DECODER_POLICY_ID) {
            "V2 executor requires ${V2IndexingWorkPolicy.DECODER_POLICY_ID}"
        }
        require(spec.inferenceBackendPolicyId == V2IndexingWorkPolicy.INFERENCE_BACKEND_POLICY_ID) {
            "V2 executor requires ${V2IndexingWorkPolicy.INFERENCE_BACKEND_POLICY_ID}"
        }
        require(V2IndexingExecutionProfile.entries.map(::schedule)
            .map(V2IndexingExecutionSchedule::pcmChunkDurationMs).distinct().size == 1
        ) { "execution profiles changed the PCM partition contract" }
    }
}

data class V2DurableStageCounter(
    val completedUnits: Long,
    val remainingUnits: Long,
    val abandonedUnits: Long,
) {
    val plannedUnits: Long get() = completedUnits + remainingUnits + abandonedUnits
}

data class V2IndexingProgressSnapshot(
    val resolvedTracks: Int,
    val succeededTracks: Int,
    val blockedTracks: Int,
    val skippedTracks: Int,
    val totalTracks: Int,
    val tracksWithMertFeatures: Int,
    val tracksWithClampVectors: Int,
    val mertWindows: V2DurableStageCounter,
    val clampSegments: V2DurableStageCounter,
    val databaseCommits: V2DurableStageCounter,
    val activation: V2DurableStageCounter,
    val activeTrackOrdinal: Int?,
    val activeStage: IndexingStage?,
) {
    val resolvedFraction: Double
        get() = if (totalTracks == 0) 1.0 else resolvedTracks.toDouble() / totalTracks
}

/** No scalar mixes unlike units; every counter has one physical meaning. */
object V2IndexingProgress {
    fun from(ledger: IndexingJobLedger): V2IndexingProgressSnapshot {
        val descriptorByWorkId = ledger.jobSpec.tracks.associateBy { it.workId }
        var mertCompleted = 0L
        var mertRemaining = 0L
        var mertAbandoned = 0L
        var clampCompleted = 0L
        var clampRemaining = 0L
        var clampAbandoned = 0L
        var commitCompleted = 0L
        var commitRemaining = 0L
        var commitAbandoned = 0L
        var tracksWithMertFeatures = 0
        var tracksWithClampVectors = 0
        var activeTrackOrdinal: Int? = null
        var activeStage: IndexingStage? = null

        ledger.tracks.forEach { track ->
            val descriptor = requireNotNull(descriptorByWorkId[track.workId])
            val terminalWithoutCommit = track.state == IndexingTrackState.BLOCKED_FAILURE ||
                track.state == IndexingTrackState.SKIPPED_BY_USER
            val kinds = track.verifiedArtifacts.mapTo(mutableSetOf()) { it.kind }
            val mertDone = VerifiedArtifactKind.MERT_FEATURES in kinds
            val clampDone = VerifiedArtifactKind.CLAMP_VECTOR in kinds
            val commitDone = VerifiedArtifactKind.DATABASE_COMMIT in kinds

            if (mertDone) tracksWithMertFeatures++
            if (clampDone) tracksWithClampVectors++
            mertCompleted += if (mertDone) descriptor.expectedWork.mertWindows.toLong() else 0L
            clampCompleted += if (clampDone) descriptor.expectedWork.clampSegments.toLong() else 0L
            commitCompleted += if (commitDone) 1L else 0L
            if (!mertDone) {
                if (terminalWithoutCommit) mertAbandoned += descriptor.expectedWork.mertWindows
                else mertRemaining += descriptor.expectedWork.mertWindows
            }
            if (!clampDone) {
                if (terminalWithoutCommit) clampAbandoned += descriptor.expectedWork.clampSegments
                else clampRemaining += descriptor.expectedWork.clampSegments
            }
            if (!commitDone) {
                if (terminalWithoutCommit) commitAbandoned++ else commitRemaining++
            }
            if (track.state.isActiveStage()) {
                activeTrackOrdinal = descriptor.ordinal
                activeStage = track.state.stageOrNull()
            }
        }

        val succeeded = ledger.tracks.count { it.state == IndexingTrackState.COMMITTED }
        val blocked = ledger.tracks.count { it.state == IndexingTrackState.BLOCKED_FAILURE }
        val skipped = ledger.tracks.count { it.state == IndexingTrackState.SKIPPED_BY_USER }
        val activationComplete = ledger.state == IndexingJobState.COMPLETE
        val activationAbandoned = ledger.state == IndexingJobState.CANCELLED
        return V2IndexingProgressSnapshot(
            resolvedTracks = succeeded + blocked + skipped,
            succeededTracks = succeeded,
            blockedTracks = blocked,
            skippedTracks = skipped,
            totalTracks = ledger.tracks.size,
            tracksWithMertFeatures = tracksWithMertFeatures,
            tracksWithClampVectors = tracksWithClampVectors,
            mertWindows = V2DurableStageCounter(mertCompleted, mertRemaining, mertAbandoned),
            clampSegments = V2DurableStageCounter(clampCompleted, clampRemaining, clampAbandoned),
            databaseCommits = V2DurableStageCounter(
                commitCompleted,
                commitRemaining,
                commitAbandoned,
            ),
            activation = V2DurableStageCounter(
                completedUnits = if (activationComplete) 1L else 0L,
                remainingUnits = if (!activationComplete && !activationAbandoned) 1L else 0L,
                abandonedUnits = if (activationAbandoned) 1L else 0L,
            ),
            activeTrackOrdinal = activeTrackOrdinal,
            activeStage = activeStage,
        )
    }
}

enum class V2MeasuredWorkStage {
    PCM_24K_SAMPLES,
    MERT_WINDOWS,
    CLAMP_SEGMENTS,
    DATABASE_COMMITS,
    DERIVED_EMBEDDING_ROWS,
    GRAPH_SIMILARITY_DOT_PRODUCTS,
    GRAPH_BINARY_BYTES,
    /** Retained only so an old local ETA sample can be ignored without failing JSON restore. */
    GRAPH_NODES,
    ACTIVATION_TRACKS,
    /** Presentation-only operations with no fabricated ETA rate. */
    PCM_NORMALIZATION_SAMPLES,
    PCM_CACHE_BYTES,
    POWERAMP_LIBRARY_ROWS,
    INDEXING_MODEL_FILES,
    SOURCE_AUDIO_HASH,
    PRIVATE_INDEX_COPY,
    SIMILARITY_GRAPH_SETUP,
    MUSIC_INDEX_PUBLICATION,
    SAVED_ARTIFACT_INSPECTION,
    RECOMMENDATION_RESOURCE_HANDOFF,
    STAGING_INDEX_INSPECTION,
}

data class V2RemainingStageWork(
    val stage: V2MeasuredWorkStage,
    val units: Long,
    val codecClass: String? = null,
)

data class V2StageAwareEtaEstimate(
    val remainingMs: Long?,
    val lowerBoundMs: Long?,
    val upperBoundMs: Long?,
    val calibratingStages: Set<V2MeasuredWorkStage>,
)

data class V2PersistedStageRateSample(
    val stage: V2MeasuredWorkStage,
    val profile: V2IndexingExecutionProfile,
    val codecClass: String?,
    val activeMsPerUnit: Double,
)

data class V2PersistedStageRateSnapshot(
    val schemaVersion: Int = 1,
    val samples: List<V2PersistedStageRateSample>,
)

/**
 * Rates are isolated by physical stage, execution profile, and codec when applicable. Callers
 * pass active work time, so pause/background-idle time can never inflate the estimate.
 */
class V2StageAwareWorkEstimator(
    private val minimumSamples: Int = 2,
    private val maximumSamples: Int = 24,
    restoredSnapshot: V2PersistedStageRateSnapshot? = null,
) {
    private data class RateKey(
        val stage: V2MeasuredWorkStage,
        val profile: V2IndexingExecutionProfile,
        val codecClass: String?,
    )

    private data class RateBounds(
        val center: Double,
        val lower: Double,
        val upper: Double,
    )

    private data class CrossProfilePrior(
        val bounds: RateBounds,
    )

    private val samples = mutableMapOf<RateKey, ArrayDeque<Double>>()

    init {
        require(minimumSamples > 0) { "minimumSamples must be positive" }
        require(maximumSamples >= minimumSamples) {
            "maximumSamples must be at least minimumSamples"
        }
        restoredSnapshot?.let(::restore)
    }

    @Synchronized
    fun recordCompleted(
        stage: V2MeasuredWorkStage,
        profile: V2IndexingExecutionProfile,
        completedUnits: Long,
        activeDurationMs: Long,
        codecClass: String? = null,
    ) {
        require(completedUnits > 0L) { "completedUnits must be positive" }
        require(activeDurationMs > 0L) { "activeDurationMs must be positive" }
        val rate = activeDurationMs.toDouble() / completedUnits
        if (!rate.isFinite() || rate <= 0.0) return
        val key = RateKey(stage, profile, codecClass.normalizedCodec())
        val values = samples.getOrPut(key) { ArrayDeque() }
        values.addLast(rate)
        while (values.size > maximumSamples) values.removeFirst()
    }

    @Synchronized
    fun estimate(
        remaining: List<V2RemainingStageWork>,
        profile: V2IndexingExecutionProfile,
    ): V2StageAwareEtaEstimate {
        val pending = remaining.filter { it.units > 0L }
        if (pending.isEmpty()) return V2StageAwareEtaEstimate(0L, 0L, 0L, emptySet())
        var center = 0.0
        var lower = 0.0
        var upper = 0.0
        val calibrating = linkedSetOf<V2MeasuredWorkStage>()
        var missingRate = false
        pending.forEach { work ->
            val candidates = rateCandidates(work, profile)
            val sufficientExact = candidates.firstOrNull { it.size >= minimumSamples }
            val bounds = if (sufficientExact != null) {
                empiricalBounds(sufficientExact)
            } else {
                calibrating += work.stage
                val prior = crossProfilePrior(work, profile)
                if (prior == null) {
                    missingRate = true
                    null
                } else {
                    val partialExact = candidates.firstOrNull(ArrayDeque<Double>::isNotEmpty)
                    partialExact?.let { refinePriorWithExactSample(prior.bounds, it) }
                        ?: prior.bounds
                }
            }
            if (bounds != null) {
                center += bounds.center * work.units
                lower += bounds.lower * work.units
                upper += bounds.upper * work.units
            }
        }
        if (missingRate) {
            return V2StageAwareEtaEstimate(null, null, null, calibrating)
        }
        return V2StageAwareEtaEstimate(
            remainingMs = durationMs(center),
            lowerBoundMs = durationMs(lower),
            upperBoundMs = durationMs(upper),
            calibratingStages = calibrating,
        )
    }

    @Synchronized
    fun snapshot(): V2PersistedStageRateSnapshot = V2PersistedStageRateSnapshot(
        samples = samples.entries
            .sortedWith(
                compareBy<Map.Entry<RateKey, ArrayDeque<Double>>>(
                    { it.key.stage.ordinal },
                    { it.key.profile.ordinal },
                    { it.key.codecClass ?: "" },
                ),
            )
            .flatMap { (key, values) ->
                values.map { rate ->
                    V2PersistedStageRateSample(
                        stage = key.stage,
                        profile = key.profile,
                        codecClass = key.codecClass,
                        activeMsPerUnit = rate,
                    )
                }
            },
    )

    @Synchronized
    fun restore(snapshot: V2PersistedStageRateSnapshot) {
        if (snapshot.schemaVersion != PERSISTED_SCHEMA_VERSION) return
        snapshot.samples.forEach { sample ->
            val rate = sample.activeMsPerUnit
            if (!rate.isFinite() || rate <= 0.0) return@forEach
            val key = RateKey(
                sample.stage,
                sample.profile,
                sample.codecClass.normalizedCodec(),
            )
            val values = samples.getOrPut(key) { ArrayDeque() }
            values.addLast(rate)
            while (values.size > maximumSamples) values.removeFirst()
        }
    }

    private fun rateCandidates(
        work: V2RemainingStageWork,
        profile: V2IndexingExecutionProfile,
    ): List<ArrayDeque<Double>> {
        val codec = work.codecClass.normalizedCodec()
        return listOfNotNull(samples[RateKey(work.stage, profile, codec)])
    }

    private fun crossProfilePrior(
        work: V2RemainingStageWork,
        targetProfile: V2IndexingExecutionProfile,
    ): CrossProfilePrior? {
        val targetYield = V2IndexingExecutionPolicies.schedule(targetProfile)
            .yieldAfterCompletedUnitMs
        return V2IndexingExecutionProfile.entries
            .asSequence()
            .filter { it != targetProfile }
            .sortedWith(
                compareBy<V2IndexingExecutionProfile> {
                    kotlin.math.abs(
                        V2IndexingExecutionPolicies.schedule(it).yieldAfterCompletedUnitMs -
                            targetYield,
                    )
                }.thenBy { it.ordinal },
            )
            .mapNotNull { sourceProfile ->
                val rates = rateCandidates(work, sourceProfile)
                    .firstOrNull { it.size >= minimumSamples }
                    ?: return@mapNotNull null
                val empirical = empiricalBounds(rates)
                val factors = crossProfileFactors(sourceProfile, targetProfile)
                CrossProfilePrior(
                    bounds = RateBounds(
                        center = empirical.center * factors.center,
                        lower = empirical.lower * factors.lower,
                        upper = empirical.upper * factors.upper,
                    ),
                )
            }
            .firstOrNull()
    }

    private fun crossProfileFactors(
        sourceProfile: V2IndexingExecutionProfile,
        targetProfile: V2IndexingExecutionProfile,
    ): RateBounds {
        val maxYield = V2IndexingExecutionPolicies.schedule(V2IndexingExecutionProfile.BACKGROUND)
            .yieldAfterCompletedUnitMs
            .coerceAtLeast(1L)
            .toDouble()
        val sourcePosition = V2IndexingExecutionPolicies.schedule(sourceProfile)
            .yieldAfterCompletedUnitMs / maxYield
        val targetPosition = V2IndexingExecutionPolicies.schedule(targetProfile)
            .yieldAfterCompletedUnitMs / maxYield
        val delta = targetPosition - sourcePosition
        if (delta >= 0.0) {
            return RateBounds(
                center = 1.0 + delta * (CROSS_PROFILE_CENTER_FACTOR - 1.0),
                lower = 1.0,
                upper = 1.0 + delta * (CROSS_PROFILE_UPPER_FACTOR - 1.0),
            )
        }
        val magnitude = -delta
        return RateBounds(
            center = 1.0 / (1.0 + magnitude * (CROSS_PROFILE_CENTER_FACTOR - 1.0)),
            lower = 1.0 / (1.0 + magnitude * (CROSS_PROFILE_UPPER_FACTOR - 1.0)),
            upper = 1.0,
        )
    }

    private fun empiricalBounds(rates: Collection<Double>): RateBounds {
        val sorted = rates.sorted()
        return RateBounds(
            center = percentile(sorted, 0.50),
            lower = percentile(sorted, 0.20),
            upper = percentile(sorted, 0.80),
        )
    }

    private fun refinePriorWithExactSample(
        prior: RateBounds,
        exactRates: Collection<Double>,
    ): RateBounds {
        val exact = empiricalBounds(exactRates)
        return RateBounds(
            center = exact.center,
            lower = minOf(prior.lower, exact.lower * PARTIAL_EXACT_LOWER_FACTOR),
            upper = maxOf(prior.upper, exact.upper * PARTIAL_EXACT_UPPER_FACTOR),
        )
    }

    private fun percentile(sorted: List<Double>, quantile: Double): Double {
        val position = (sorted.size - 1) * quantile
        val lowerIndex = floor(position).toInt().coerceIn(sorted.indices)
        val upperIndex = ceil(position).toInt().coerceIn(sorted.indices)
        if (lowerIndex == upperIndex) return sorted[lowerIndex]
        val fraction = position - lowerIndex
        return sorted[lowerIndex] + (sorted[upperIndex] - sorted[lowerIndex]) * fraction
    }

    private fun durationMs(value: Double): Long {
        val rounded = ceil(value)
        return when {
            !rounded.isFinite() || rounded >= Long.MAX_VALUE.toDouble() -> Long.MAX_VALUE
            rounded <= 0.0 -> 0L
            else -> rounded.toLong()
        }
    }

    private fun String?.normalizedCodec(): String? = this?.trim()?.lowercase()?.takeIf(String::isNotEmpty)

    private companion object {
        const val PERSISTED_SCHEMA_VERSION = 1

        // On this phone, isolated whole-track Background runs were 1.0-1.19x Full, while live
        // stage rates reached 2.3-3.1x. The 1x-4x envelope covers both observations and the
        // declared 0/8/40 ms yield schedule without presenting a borrowed rate as calibrated.
        const val CROSS_PROFILE_CENTER_FACTOR = 2.5
        const val CROSS_PROFILE_UPPER_FACTOR = 4.0
        const val PARTIAL_EXACT_LOWER_FACTOR = 0.75
        const val PARTIAL_EXACT_UPPER_FACTOR = 1.5
    }
}
