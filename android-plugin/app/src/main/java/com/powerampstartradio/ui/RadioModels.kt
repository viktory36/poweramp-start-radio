package com.powerampstartradio.ui

import com.powerampstartradio.data.EmbeddedTrack
import com.powerampstartradio.poweramp.PowerampTrack
import com.powerampstartradio.poweramp.TrackMatcher
import com.powerampstartradio.similarity.GraphExplorationEvidence
import com.powerampstartradio.similarity.DppSelectionEvidence
import com.powerampstartradio.similarity.FindMusicTextQueuePlanEvidence
import com.powerampstartradio.similarity.FindMusicAllOfQueuePlanEvidence
import com.powerampstartradio.similarity.UniformShuffleIdentityEvidence

/**
 * Type of seed for multi-seed search.
 */
enum class SeedType { TEXT, SONG }

/**
 * A single seed for multi-seed search.
 *
 * @param embedding 768d CLaMP3 embedding (text or audio)
 * @param weight -1.0 to 1.0. Positive = "more like", negative = "less like". 0 = ignored.
 * @param label Display label (e.g. "90s boombap" or "Time - Pachanga Boys")
 * @param type TEXT or SONG
 * @param trackId For SONG seeds: the embedding DB track ID (for exclusion from results)
 */
data class SeedSpec(
    val embedding: FloatArray,
    val weight: Float,
    val label: String,
    val type: SeedType,
    val trackId: Long? = null,
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is SeedSpec) return false
        return label == other.label && weight == other.weight && type == other.type && trackId == other.trackId
    }
    override fun hashCode(): Int = label.hashCode() * 31 + type.hashCode()
}

/**
 * User-selectable recommendation algorithm.
 */
enum class SelectionMode {
    CLOSEST,
    MMR,
    DPP,
    RANDOM_WALK,
    UNIFORM_SHUFFLE,
}

/** Rolling Poweramp first-seen window used to define recommendation candidates. */
enum class LibraryAddedRange(
    val dayCount: Long?,
    val label: String,
) {
    LAST_7_DAYS(7L, "Last 7 days"),
    LAST_30_DAYS(30L, "Last 30 days"),
    LAST_365_DAYS(365L, "Last 365 days"),
    ALL_DATES(null, "All dates");

    fun minimumCreatedAtEpochSecond(referenceEpochSecond: Long): Long? =
        minimumLibraryAddedAtEpochSecond(dayCount?.toInt(), referenceEpochSecond)
}

const val DEFAULT_LIBRARY_ADDED_DAYS = 30
const val MAX_LIBRARY_ADDED_DAYS = 36_500

/** Exact rolling Poweramp first-seen window. Null keeps every dated and undated recording. */
fun libraryAddedDaysLabel(days: Int?): String = when (days) {
    null -> "All dates"
    1 -> "Last 1 day"
    else -> {
        require(days in 1..MAX_LIBRARY_ADDED_DAYS) {
            "Poweramp added-date day count must be 1..$MAX_LIBRARY_ADDED_DAYS"
        }
        "Last $days days"
    }
}

/**
 * Resolve a rolling N-by-24-hour window at one durable request timestamp.
 *
 * A very large finite window saturates at the first valid Unix timestamp. It remains distinct
 * from All dates because active rows with unknown first-seen time (zero) stay excluded.
 */
fun minimumLibraryAddedAtEpochSecond(
    days: Int?,
    referenceEpochSecond: Long,
): Long? {
    if (days == null) return null
    require(days in 1..MAX_LIBRARY_ADDED_DAYS) {
        "Poweramp added-date day count must be 1..$MAX_LIBRARY_ADDED_DAYS"
    }
    require(referenceEpochSecond > 0L) { "Poweramp added-date reference time must be positive" }
    val durationSeconds = Math.multiplyExact(days.toLong(), 86_400L)
    return (referenceEpochSecond - durationSeconds).coerceAtLeast(1L)
}

/**
 * How the query evolves across drift steps.
 */
enum class DriftMode {
    SEED_INTERPOLATION,
    MOMENTUM
}

/**
 * How anchor strength decays over time in seed interpolation.
 */
enum class DecaySchedule {
    NONE,
    LINEAR,
    EXPONENTIAL,
    STEP
}

/**
 * Full configuration for a radio session.
 */
data class RadioConfig(
    val configSchemaVersion: Int = CURRENT_CONFIG_SCHEMA_VERSION,
    val numTracks: Int = 50,
    /** Legacy preset field retained so existing V2 Gson records remain readable. */
    val libraryAddedRange: LibraryAddedRange = LibraryAddedRange.ALL_DATES,
    /** Exact rolling day count for new requests; null falls back to the legacy preset field. */
    val libraryAddedDays: Int? = null,
    val candidatePoolSize: Int = 0,  // Runtime value; 0 derives from the selected mode's reach.
    val mmrCandidatePoolFraction: Float = DEFAULT_MMR_CANDIDATE_POOL_FRACTION,
    val dppFixedCandidatePoolFraction: Float = DEFAULT_DPP_FIXED_CANDIDATE_POOL_FRACTION,
    /** True runs adaptive DPP until every greedy choice is certified against all unseen rows. */
    val dppUsesCertifiedFullDomain: Boolean = true,
    val selectionMode: SelectionMode = SelectionMode.MMR,
    val driftEnabled: Boolean = false,
    val driftMode: DriftMode = DriftMode.SEED_INTERPOLATION,
    val anchorStrength: Float = 0.5f,
    val anchorDecay: DecaySchedule = DecaySchedule.EXPONENTIAL,
    val anchorHalfLifeTracks: Float = DEFAULT_ANCHOR_HALF_LIFE_TRACKS,
    val walkRestartAlpha: Float = 0.5f,
    val momentumBeta: Float = DEFAULT_MOMENTUM_BETA,
    val diversityLambda: Float = DEFAULT_MMR_RELEVANCE_WEIGHT,
    val dppQualityExponent: Float = DEFAULT_DPP_QUALITY_EXPONENT,
    val shuffleSeed: Long = DEFAULT_SHUFFLE_SEED,
    val artistLimitsEnabled: Boolean = true,
    val maxPerArtist: Int = 8,
    val minArtistSpacing: Int = 3,          // Audit: spacing=3 vs 5 (5 drops queue to ~44)
) {
    /** Gson V1 history records deserialize missing primitive fields as zero. */
    val effectiveAnchorHalfLifeTracks: Float
        get() = anchorHalfLifeTracks.takeIf { it > 0f && it.isFinite() }
            ?: DEFAULT_ANCHOR_HALF_LIFE_TRACKS

    val effectiveMmrCandidatePoolFraction: Float
        get() = mmrCandidatePoolFraction.takeIf { it > 0f && it <= 1f && it.isFinite() }
            ?: DEFAULT_MMR_CANDIDATE_POOL_FRACTION

    val effectiveDppFixedCandidatePoolFraction: Float
        get() = dppFixedCandidatePoolFraction.takeIf {
            it > 0f && it <= 1f && it.isFinite()
        } ?: DEFAULT_DPP_FIXED_CANDIDATE_POOL_FRACTION

    /** MMR's semantic domain, or automatic DPP's output-neutral first proof prefix. */
    val effectiveWorkingCandidatePoolFraction: Float
        get() = when (selectionMode) {
            SelectionMode.MMR -> effectiveMmrCandidatePoolFraction
            SelectionMode.DPP -> if (dppUsesCertifiedFullDomain) {
                DEFAULT_DPP_CERTIFICATE_INITIAL_FRACTION
            } else {
                effectiveDppFixedCandidatePoolFraction
            }
            else -> error("${selectionMode.name} has no candidate working set")
        }

    fun resolveCandidatePoolSize(activeTrackCount: Int): Int {
        require(activeTrackCount >= 0) { "activeTrackCount must be non-negative" }
        require(selectionMode == SelectionMode.MMR || selectionMode == SelectionMode.DPP) {
            "${selectionMode.name} has no candidate working set"
        }
        if (activeTrackCount == 0) return 0
        if (candidatePoolSize > 0) return candidatePoolSize.coerceAtMost(activeTrackCount)
        return (activeTrackCount * effectiveWorkingCandidatePoolFraction).toInt()
            .coerceAtLeast(maxOf(100, numTracks))
            .coerceAtMost(activeTrackCount)
    }

    val effectiveDppQualityExponent: Float
        get() = dppQualityExponent.takeIf {
            configSchemaVersion >= DPP_CONFIG_SCHEMA_VERSION &&
                it >= 0f && it <= 8f && it.isFinite()
        }
            ?: DEFAULT_DPP_QUALITY_EXPONENT

    /** Gson records which predate explicit shuffle seeds deserialize this field as zero. */
    val effectiveShuffleSeed: Long
        get() = shuffleSeed.takeIf { it != 0L } ?: DEFAULT_SHUFFLE_SEED

    companion object {
        const val DEFAULT_ANCHOR_HALF_LIFE_TRACKS = 7f
        const val DEFAULT_MMR_RELEVANCE_WEIGHT = 0.5f
        const val DEFAULT_MMR_CANDIDATE_POOL_FRACTION = 0.02f
        const val DEFAULT_DPP_FIXED_CANDIDATE_POOL_FRACTION = 0.02f
        const val DEFAULT_DPP_CERTIFICATE_INITIAL_FRACTION = 0.02f
        const val DEFAULT_DPP_QUALITY_EXPONENT = 1f
        const val DEFAULT_MOMENTUM_BETA = 0.9f
        const val DEFAULT_SHUFFLE_SEED = 0x5053525632534855L
        const val DPP_CONFIG_SCHEMA_VERSION = 2
        const val CURRENT_CONFIG_SCHEMA_VERSION = 5
    }
}

/** Canonical added-date semantics shared by new exact values and legacy preset records. */
val RadioConfig.effectiveLibraryAddedDays: Int?
    get() = libraryAddedDays ?: libraryAddedRange.dayCount?.toInt()

/** Apply the cheapest UI capability mask without erasing saved controls for another mode. */
fun RadioConfig.forActiveSelectionMode(): RadioConfig =
    if (selectionMode != SelectionMode.MMR && driftEnabled) copy(driftEnabled = false) else this

/**
 * Project saved controls onto the exact semantic inputs of one new radio request.
 *
 * Saved preferences deliberately retain dormant values so switching modes restores the user's
 * choices. Requests do not: irrelevant values are canonicalized so the journal, widget, Peek,
 * and service all describe the behavior that can actually affect the queue.
 */
fun RadioConfig.forSelectionRequest(
    eligibleCandidateIdentityCount: Int? = null,
): RadioConfig {
    val defaults = RadioConfig()
    val effectiveDppFullDomain = DppDomainControlPolicy.effectiveUsesCertifiedFullDomain(
        storedUsesCertifiedFullDomain = dppUsesCertifiedFullDomain,
        eligibleCandidateIdentityCount = eligibleCandidateIdentityCount,
        numTracks = numTracks,
    )
    val common = copy(
        configSchemaVersion = RadioConfig.CURRENT_CONFIG_SCHEMA_VERSION,
        candidatePoolSize = 0,
        dppUsesCertifiedFullDomain = effectiveDppFullDomain,
        maxPerArtist = if (artistLimitsEnabled) maxPerArtist else defaults.maxPerArtist,
        minArtistSpacing = if (artistLimitsEnabled) minArtistSpacing else defaults.minArtistSpacing,
    )
    val withoutDrift = common.copy(
        driftEnabled = false,
        driftMode = defaults.driftMode,
        anchorStrength = defaults.anchorStrength,
        anchorDecay = defaults.anchorDecay,
        anchorHalfLifeTracks = defaults.anchorHalfLifeTracks,
        momentumBeta = defaults.momentumBeta,
    )

    return when (selectionMode) {
        SelectionMode.CLOSEST -> withoutDrift.copy(
            mmrCandidatePoolFraction = defaults.mmrCandidatePoolFraction,
            dppFixedCandidatePoolFraction = defaults.dppFixedCandidatePoolFraction,
            dppUsesCertifiedFullDomain = defaults.dppUsesCertifiedFullDomain,
            walkRestartAlpha = defaults.walkRestartAlpha,
            diversityLambda = defaults.diversityLambda,
            dppQualityExponent = defaults.dppQualityExponent,
            shuffleSeed = defaults.shuffleSeed,
        )
        SelectionMode.MMR -> {
            val drift = if (!common.driftEnabled) {
                withoutDrift
            } else when (common.driftMode) {
                DriftMode.SEED_INTERPOLATION -> common.copy(
                    momentumBeta = defaults.momentumBeta,
                    anchorHalfLifeTracks = when (common.anchorDecay) {
                        DecaySchedule.NONE -> defaults.anchorHalfLifeTracks
                        DecaySchedule.STEP ->
                        DriftControlPolicy.canonicalFadeTiming(
                            decay = common.anchorDecay,
                            timingTracks = common.anchorHalfLifeTracks,
                            recommendationCount = common.numTracks,
                        )
                        else -> common.anchorHalfLifeTracks
                    },
                )
                DriftMode.MOMENTUM -> common.copy(
                    anchorStrength = defaults.anchorStrength,
                    anchorDecay = defaults.anchorDecay,
                    anchorHalfLifeTracks = defaults.anchorHalfLifeTracks,
                )
            }
            drift.copy(
                dppFixedCandidatePoolFraction = defaults.dppFixedCandidatePoolFraction,
                dppUsesCertifiedFullDomain = defaults.dppUsesCertifiedFullDomain,
                walkRestartAlpha = defaults.walkRestartAlpha,
                dppQualityExponent = defaults.dppQualityExponent,
                shuffleSeed = defaults.shuffleSeed,
            )
        }
        SelectionMode.DPP -> withoutDrift.copy(
            mmrCandidatePoolFraction = defaults.mmrCandidatePoolFraction,
            dppFixedCandidatePoolFraction = if (common.dppUsesCertifiedFullDomain) {
                defaults.dppFixedCandidatePoolFraction
            } else {
                common.dppFixedCandidatePoolFraction
            },
            walkRestartAlpha = defaults.walkRestartAlpha,
            diversityLambda = defaults.diversityLambda,
            shuffleSeed = defaults.shuffleSeed,
        )
        SelectionMode.RANDOM_WALK -> withoutDrift.copy(
            mmrCandidatePoolFraction = defaults.mmrCandidatePoolFraction,
            dppFixedCandidatePoolFraction = defaults.dppFixedCandidatePoolFraction,
            dppUsesCertifiedFullDomain = defaults.dppUsesCertifiedFullDomain,
            diversityLambda = defaults.diversityLambda,
            dppQualityExponent = defaults.dppQualityExponent,
            shuffleSeed = defaults.shuffleSeed,
        )
        SelectionMode.UNIFORM_SHUFFLE -> withoutDrift.copy(
            mmrCandidatePoolFraction = defaults.mmrCandidatePoolFraction,
            dppFixedCandidatePoolFraction = defaults.dppFixedCandidatePoolFraction,
            dppUsesCertifiedFullDomain = defaults.dppUsesCertifiedFullDomain,
            walkRestartAlpha = defaults.walkRestartAlpha,
            diversityLambda = defaults.diversityLambda,
            dppQualityExponent = defaults.dppQualityExponent,
        )
    }
}

object NeighborhoodReachPolicy {
    val MMR_OPTIONS = listOf(0.0025f, 0.005f, 0.01f, 0.02f, 0.05f, 0.10f, 0.25f, 0.50f, 1f)
    val DPP_FIXED_OPTIONS = MMR_OPTIONS.dropLast(1)

    fun candidateCount(fraction: Float, librarySize: Int, numTracks: Int): Int {
        require(fraction > 0f && fraction <= 1f && fraction.isFinite())
        require(librarySize >= 0)
        require(numTracks >= 0)
        if (librarySize == 0) return 0
        return (librarySize * fraction).toInt()
            .coerceAtLeast(maxOf(100, numTracks))
            .coerceAtMost(librarySize)
    }

    /**
     * Keep one visible stop for each result domain the current library can actually produce.
     *
     * When the saved semantic fraction belongs to a collapsed group, it represents that group.
     * This keeps the thumb and label aligned without replacing the user's percentage with a
     * library-size-dependent alias which would mean something different after the library grows.
     */
    fun distinctOptions(
        options: List<Float>,
        librarySize: Int,
        numTracks: Int,
        preferredFraction: Float? = null,
    ): List<Float> {
        require(options.isNotEmpty())
        require(librarySize >= 0)
        val preferredOption = preferredFraction?.let { preferred ->
            options.firstOrNull { it == preferred }
        }
        return options
            .groupBy { candidateCount(it, librarySize, numTracks) }
            .values
            .map { sameDomain ->
                preferredOption?.takeIf(sameDomain::contains) ?: sameDomain.last()
            }
    }

    /** Unknown library context preserves the control; an exact context may prove it has no choice. */
    fun hasMultipleDistinctDomains(
        options: List<Float>,
        eligibleCandidateIdentityCount: Int?,
        numTracks: Int,
        preferredFraction: Float? = null,
    ): Boolean = eligibleCandidateIdentityCount?.let { candidateCount ->
        distinctOptions(
            options = options,
            librarySize = candidateCount,
            numTracks = numTracks,
            preferredFraction = preferredFraction,
        ).size > 1
    } ?: true
}

/** A DPP domain choice exists only when at least one measured fixed stop is a proper subset. */
object DppDomainControlPolicy {
    fun hasProperFixedSubset(
        eligibleCandidateIdentityCount: Int?,
        numTracks: Int,
    ): Boolean = eligibleCandidateIdentityCount?.let { candidateCount ->
        NeighborhoodReachPolicy.DPP_FIXED_OPTIONS.any { fraction ->
            NeighborhoodReachPolicy.candidateCount(
                fraction = fraction,
                librarySize = candidateCount,
                numTracks = numTracks,
            ) < candidateCount
        }
    } ?: true

    fun effectiveUsesCertifiedFullDomain(
        storedUsesCertifiedFullDomain: Boolean,
        eligibleCandidateIdentityCount: Int?,
        numTracks: Int,
    ): Boolean = storedUsesCertifiedFullDomain || !hasProperFixedSubset(
        eligibleCandidateIdentityCount = eligibleCandidateIdentityCount,
        numTracks = numTracks,
    )

    fun shouldExposeDomainControl(
        eligibleCandidateIdentityCount: Int?,
        numTracks: Int,
    ): Boolean = hasProperFixedSubset(
        eligibleCandidateIdentityCount = eligibleCandidateIdentityCount,
        numTracks = numTracks,
    )
}

/** Reach only matters when MMR can prefer something other than raw relevance order. */
object MmrControlPolicy {
    fun reachCanAffectOutput(
        relevanceWeight: Float,
        artistLimitsEnabled: Boolean,
        recommendationCount: Int,
        maxPerArtist: Int,
        minArtistSpacing: Int,
    ): Boolean = relevanceWeight != 1f || artistConstraintsCanReject(
        artistLimitsEnabled = artistLimitsEnabled,
        recommendationCount = recommendationCount,
        maxPerArtist = maxPerArtist,
        minArtistSpacing = minArtistSpacing,
    )

    /** Structural truth only: no artist metadata is inspected or inferred. */
    fun artistConstraintsCanReject(
        artistLimitsEnabled: Boolean,
        recommendationCount: Int,
        maxPerArtist: Int,
        minArtistSpacing: Int,
    ): Boolean = artistLimitsEnabled && recommendationCount > 0 &&
        (maxPerArtist < recommendationCount ||
            (recommendationCount > 1 && minArtistSpacing > 0))

    fun reachCanAffectOutput(config: RadioConfig): Boolean = reachCanAffectOutput(
        relevanceWeight = config.diversityLambda,
        artistLimitsEnabled = config.artistLimitsEnabled,
        recommendationCount = config.numTracks,
        maxPerArtist = config.maxPerArtist,
        minArtistSpacing = config.minArtistSpacing,
    )

    fun shouldExposeReach(
        config: RadioConfig,
        eligibleCandidateIdentityCount: Int?,
    ): Boolean = reachCanAffectOutput(config) &&
        NeighborhoodReachPolicy.hasMultipleDistinctDomains(
        options = NeighborhoodReachPolicy.MMR_OPTIONS,
        eligibleCandidateIdentityCount = eligibleCandidateIdentityCount,
        numTracks = config.numTracks,
        preferredFraction = config.effectiveMmrCandidatePoolFraction,
    )
}

object SelectionKnobPolicy {
    val MMR_RELEVANCE_OPTIONS = listOf(0f, 0.25f, 0.4f, 0.5f, 0.55f, 0.6f, 0.8f, 0.9f, 1f)
    val DPP_SEED_PULL_OPTIONS = listOf(0f, 0.25f, 0.5f, 1f, 2f, 3f, 4f)
    val GRAPH_STOP_OPTIONS = listOf(0.05f, 0.10f, 0.25f, 0.50f, 0.75f, 0.90f)

    fun nearestIndex(options: List<Float>, value: Float): Int {
        require(options.isNotEmpty())
        return options.indices.minBy { index -> kotlin.math.abs(options[index] - value) }
    }

    fun nearestValue(options: List<Float>, value: Float): Float =
        options[nearestIndex(options, value)]
}

/** Distinct, measured controls for MMR's evolving-query variants. */
object DriftControlPolicy {
    val SEED_PULL_OPTIONS = listOf(0f, 0.25f, 0.5f, 0.75f, 0.85f, 1f)
    val MOMENTUM_OPTIONS = listOf(
        0.25f,
        0.5f,
        0.75f,
        RadioConfig.DEFAULT_MOMENTUM_BETA,
        0.95f,
        1f,
    )
    val FADE_TIMING_OPTIONS = listOf(
        1f,
        3f,
        RadioConfig.DEFAULT_ANCHOR_HALF_LIFE_TRACKS,
        10f,
        15f,
        30f,
    )

    /**
     * A Step update made after the final useful query cannot affect an emitted track.
     * Other schedules begin changing immediately, so their absolute timing remains meaningful
     * even when their named half-life lies beyond the requested queue.
     */
    fun fadeTimingOptions(
        decay: DecaySchedule,
        recommendationCount: Int,
    ): List<Float> = if (decay == DecaySchedule.STEP) {
        FADE_TIMING_OPTIONS.filter { stepTimingCanAffectQueue(it, recommendationCount) }
            .ifEmpty { listOf(FADE_TIMING_OPTIONS.first()) }
    } else {
        FADE_TIMING_OPTIONS
    }

    fun canonicalFadeTiming(
        decay: DecaySchedule,
        timingTracks: Float,
        recommendationCount: Int,
    ): Float {
        val options = fadeTimingOptions(decay, recommendationCount)
        val finiteTiming = timingTracks.takeIf { it.isFinite() && it > 0f }
            ?: RadioConfig.DEFAULT_ANCHOR_HALF_LIFE_TRACKS
        return SelectionKnobPolicy.nearestValue(options, finiteTiming)
    }

    fun stepTimingCanAffectQueue(
        timingTracks: Float,
        recommendationCount: Int,
    ): Boolean = timingTracks.isFinite() && timingTracks > 0f &&
        timingTracks <= recommendationCount - 2f

    /** The Step threshold is applied after this many emitted picks have completed. */
    fun stepDropAfterPickCount(timingTracks: Float): Int {
        require(timingTracks.isFinite() && timingTracks > 0f)
        return kotlin.math.ceil(timingTracks.toDouble()).toInt() + 1
    }

    /** Holding 100% seed duplicates Momentum 100%'s fixed-query drift path. */
    fun seedPullOptions(decay: DecaySchedule): List<Float> =
        if (decay == DecaySchedule.NONE) SEED_PULL_OPTIONS.dropLast(1) else SEED_PULL_OPTIONS

    /** At 100% starting seed pull, Hold duplicates Momentum 100%. */
    fun decaySchedules(anchorStrength: Float): List<DecaySchedule> =
        DecaySchedule.entries.filterNot {
            it == DecaySchedule.NONE && anchorStrength >= 1f
        }

    /** With zero seed pull, every fade schedule and timing produces the same query. */
    fun seedFadeApplies(anchorStrength: Float): Boolean = anchorStrength > 0f

    fun usesFixedSeedQuery(
        mode: DriftMode,
        anchorStrength: Float,
        anchorDecay: DecaySchedule,
        momentumBeta: Float,
    ): Boolean = when (mode) {
        DriftMode.SEED_INTERPOLATION ->
            anchorDecay == DecaySchedule.NONE && anchorStrength >= 1f
        DriftMode.MOMENTUM -> momentumBeta >= 1f
    }

    fun isFollowLastPick(
        mode: DriftMode,
        anchorStrength: Float,
        momentumBeta: Float,
    ): Boolean = when (mode) {
        DriftMode.SEED_INTERPOLATION -> anchorStrength <= 0f
        DriftMode.MOMENTUM -> momentumBeta <= 0f
    }
}

/** Immutable identity of the exact embedding generation used by a radio request. */
data class RadioGenerationToken(
    val schemaVersion: Int = CURRENT_SCHEMA_VERSION,
    val generationId: String,
    val activationBindingId: String,
    val manifestSha256: String,
    val embeddingSpecId: String,
    val databaseContentSha256: String,
    val orderedTrackSetSha256: String,
    val stableTrackUidMappingSha256: String,
) {
    companion object {
        const val CURRENT_SCHEMA_VERSION = 1
    }
}

/** Generation-bound seed locator with a path-independent identity where indexing supplied one. */
data class RadioSeedIdentity(
    val embeddedTrackId: Long,
    val stableTrackSpanId: String?,
)

enum class ComposedRadioOperator {
    ALL_OF_GEOMETRIC_MEAN_PERCENTILES,
}

/** Explicit contract for composed radio; ordinary selection-mode knobs do not describe it. */
data class ComposedRadioContract(
    val schemaVersion: Int = CURRENT_SCHEMA_VERSION,
    val operator: ComposedRadioOperator = ComposedRadioOperator.ALL_OF_GEOMETRIC_MEAN_PERCENTILES,
    val rankingVersion: Int = CURRENT_RANKING_VERSION,
) {
    companion object {
        const val CURRENT_SCHEMA_VERSION = 1
        const val CURRENT_RANKING_VERSION = FindMusicQuerySpec.CURRENT_RANKING_VERSION
    }
}

/** Durable end state of a user-requested queue operation. */
enum class RadioSessionOutcome {
    SUCCEEDED,
    PARTIAL_FAILED,
    CANCELLED,
}

enum class DirectQueuePlacement {
    REPLACE_UPCOMING,
    APPEND,
}

/**
 * Status of a single track in the queue operation.
 */
enum class QueueStatus {
    PENDING,
    QUEUED,
    NOT_IN_LIBRARY,
    QUEUE_FAILED
}

/** The user-visible path which produced a queue mutation. */
enum class QueueOrigin(val displayLabel: String, val isDirectList: Boolean = false) {
    APP_RADIO("Started in app"),
    WIDGET_RADIO("Started from widget"),
    QUEUE_TRACK_RADIO("Started from queued track"),
    TEXT_RESULT_RADIO("Started from Find Music"),
    COMPOSED_RADIO("Started from Find Music"),
    TEXT_RESULT_LIST("Find Music queue", isDirectList = true),
    COMPOSED_RESULT_LIST("Find Music queue", isDirectList = true),
    HISTORY_REQUEUE("Queued from history", isDirectList = true),
    DEBUG_RADIO("Debug radio"),
    LEGACY_UNKNOWN("Legacy session"),
}

/** Persisted queue-delivery facts; none of these counts is inferred from an insert return. */
data class QueueDeliverySummary(
    val origin: QueueOrigin,
    val requestedCount: Int,
    val rankedCount: Int,
    val resolvedCount: Int,
    val verifiedCount: Int,
    val notInLibraryCount: Int,
    val queueFailedCount: Int,
    val verificationComplete: Boolean,
    val mutationCount: Int = 1,
    val unexpectedObservedCount: Int = 0,
) {
    companion object {
        fun fromTracks(
            origin: QueueOrigin,
            requestedCount: Int,
            rankedCount: Int,
            resolvedCount: Int,
            tracks: List<QueuedTrackResult>,
            verificationComplete: Boolean,
            mutationCount: Int = 1,
            unexpectedObservedCount: Int = 0,
        ): QueueDeliverySummary {
            return QueueDeliverySummary(
                origin = origin,
                requestedCount = requestedCount,
                rankedCount = rankedCount,
                resolvedCount = resolvedCount,
                verifiedCount = tracks.count { it.status == QueueStatus.QUEUED },
                notInLibraryCount = tracks.count { it.status == QueueStatus.NOT_IN_LIBRARY },
                queueFailedCount = tracks.count { it.status == QueueStatus.QUEUE_FAILED },
                verificationComplete = verificationComplete,
                mutationCount = mutationCount,
                unexpectedObservedCount = unexpectedObservedCount,
            )
        }
    }
}

/**
 * A single influence on a track's selection.
 *
 * @param sourceIndex -1 = seed track, 0..N-1 = index in result list
 * @param weight Exact mathematical influence weight (sums to ~1.0 per track)
 */
data class Influence(
    val sourceIndex: Int,
    val weight: Float,
)

/**
 * Full provenance for a queued track — every influence that shaped its selection.
 *
 * Batch modes: seed only. Seed interp: seed + previous track. EMA: all predecessors.
 */
data class TrackProvenance(
    val influences: List<Influence> = listOf(Influence(-1, 1f))
)

/** Truthful ranking evidence for one All-of composed-radio row. */
data class ComposedTrackEvidence(
    val objectiveRank: Int,
    val objectiveScore: Float,
    val ingredientPercentiles: List<Float>,
)

/** Exact MMR objective terms for one selected track. */
data class MmrTrackEvidence(
    val selectionStep: Int,
    val queryRelevance: Float,
    val greatestSelectedOverlap: Float,
    val greatestOverlapTrackId: Long?,
    val objective: Float,
    val candidateRank: Int,
)

/** Proof of how a visible top-N prefix was filled from objective-ranked database rows. */
data class StableResultReductionEvidence(
    val identityPolicyVersion: Int,
    val requestedVisibleCount: Int,
    val scannedRowCount: Int,
    val collapsedEquivalentCount: Int,
)

/** Exact immutable context for a Find Music result list queued as it was displayed. */
data class FindMusicSessionEvidence(
    val querySpec: FindMusicQuerySpec,
    val orderedActiveTrackIdsSha256: String,
    /** Binding count of active embedding rows; not necessarily a composed ranking denominator. */
    val activeTrackCount: Int,
    /** Exact domain of FindMusicTrackEvidence.objectiveRank. */
    val objectiveRankingDomainCount: Int? = null,
    /** Exact composed identity domain used to compute ingredient percentiles. */
    val ingredientRankingDomainCount: Int? = null,
    val stableResultReduction: StableResultReductionEvidence,
    /** Exact complete-domain text membership plan; required for Varied text results. */
    val textQueuePlan: FindMusicTextQueuePlanEvidence? = null,
    /** Exact complete-domain All-of membership plan; required for Varied All-of results. */
    val allOfQueuePlan: FindMusicAllOfQueuePlanEvidence? = null,
)

/** Exact ranking facts shown for one row in a queued Find Music result list. */
data class FindMusicTrackEvidence(
    val displayedRank: Int,
    val objectiveRank: Int,
    val resultScore: Float,
    val rankingScore: Float,
    val ingredientPercentiles: List<Float> = emptyList(),
)

/**
 * Result of attempting to queue a single similar track.
 */
data class QueuedTrackResult(
    val track: EmbeddedTrack,
    val similarity: Float,
    val similarityToSeed: Float,
    val candidateRank: Int? = null,
    /** Cosine rank from the original seed over full active identities, excluding the seed. */
    val seedRank: Int? = null,
    /** Cosine rank from the evolving query over the same full active identity domain. */
    val driftRank: Int? = null,
    val graphTerminalProbability: Double? = null,
    val graphExpectedRouteLinks: Double? = null,
    /** V1-only shortest-path evidence; it did not describe the sampled walk. */
    val graphHops: Int? = null,
    val status: QueueStatus,
    val provenance: TrackProvenance = TrackProvenance(),
    /** Exact Poweramp folder_files row resolved for this track, never a similarity input. */
    val resolvedPowerampFileId: Long? = null,
    /** Exact queue._id observed after delivery; distinct even for duplicate file IDs. */
    val resolvedPowerampQueueId: Long? = null,
    /** Path-independent full-content source-span identity when the generation proves one. */
    val stableTrackSpanId: String? = null,
    /** Present only for rows ranked by the composed All-of objective. */
    val composedEvidence: ComposedTrackEvidence? = null,
    /** Present for both fixed-query and evolving-query MMR selections. */
    val mmrEvidence: MmrTrackEvidence? = null,
    /** Present only when this exact row came from a displayed Find Music result list. */
    val findMusicEvidence: FindMusicTrackEvidence? = null,
)

/**
 * Queue quality metrics computed from embeddings after playlist generation.
 *
 * @param uniqueArtists Number of distinct artists in the queue
 * @param clusterSpread Number of distinct style clusters represented
 * @param simMin Minimum similarity to seed, as percentage
 * @param simMax Maximum similarity to seed, as percentage
 */
data class QueueMetrics(
    val uniqueArtists: Int,
    val clusterSpread: Int,
    val simMin: Int,
    val simMax: Int,
)

/**
 * Complete result of a "Start Radio" operation.
 */
data class RadioResult(
    val seedTrack: PowerampTrack,
    val matchType: TrackMatcher.MatchType,
    val tracks: List<QueuedTrackResult>,
    val config: RadioConfig = RadioConfig(),
    val timestamp: Long = System.currentTimeMillis(),
    val queuedFileIds: Set<Long> = emptySet(),
    /** Poweramp file ID anchored at queue pos 0 for text search (may differ from seed). */
    val queueAnchorId: Long? = null,
    /** Exact queue._id preserved for the active item, when playback came from Queue. */
    val queueAnchorOccurrenceId: Long? = null,
    val isComplete: Boolean = true,
    val totalExpected: Int = 0,
    val metrics: QueueMetrics? = null,
    /** True for direct-queue sessions (text/multi-seed search results). No meaningful single-seed distance. */
    val isDirectQueue: Boolean = false,
    /** Nullable so Gson can load V1 history records which predate structured delivery facts. */
    val delivery: QueueDeliverySummary? = null,
    /** Exact session-wide graph proof; null for non-graph and legacy sessions. */
    val graphExploration: GraphExplorationEvidence? = null,
    /** Exact automatic-DPP prefix proof; null for fixed-neighborhood and non-DPP sessions. */
    val dppSelection: DppSelectionEvidence? = null,
    /** Durable V2 request identity; null for sessions created before the request journal. */
    val requestId: String? = null,
    /** Explicit V2 outcome. Null only for history written before the durable request protocol. */
    val outcome: RadioSessionOutcome? = null,
    val failureDetail: String? = null,
    val generation: RadioGenerationToken? = null,
    /** Exact Poweramp provider snapshot used for ranking and queue occurrence binding. */
    val providerGenerationId: String? = null,
    val seedIdentity: RadioSeedIdentity? = null,
    val composedContract: ComposedRadioContract? = null,
    /** Exact displayed Find Music request which produced a composed radio. */
    val composedQuerySpec: FindMusicQuerySpec? = null,
    val stableResultReduction: StableResultReductionEvidence? = null,
    /** Exact All-of objective domain; null for non-composed and legacy sessions. */
    val composedObjectiveRankingDomainCount: Int? = null,
    val uniformShuffleIdentity: UniformShuffleIdentityEvidence? = null,
    /** Exact added-date, seed-excluded identity domain used by selection and reach controls. */
    val eligibleCandidateIdentityCount: Int? = null,
    /** Exact full-active, seed-excluded identity domain used by seedRank and driftRank. */
    val seedRankingIdentityCount: Int? = null,
    val directQueuePlacement: DirectQueuePlacement? = null,
    /** Exact query and active-domain facts for a directly queued Find Music result list. */
    val findMusicSessionEvidence: FindMusicSessionEvidence? = null,
) {
    val effectiveOutcome: RadioSessionOutcome
        get() = outcome ?: if (isComplete) {
            RadioSessionOutcome.SUCCEEDED
        } else {
            RadioSessionOutcome.PARTIAL_FAILED
        }
    val origin: QueueOrigin
        get() = delivery?.origin ?: QueueOrigin.LEGACY_UNKNOWN
    val queuedCount: Int
        get() = delivery?.verifiedCount ?: tracks.count { it.status == QueueStatus.QUEUED }
    val requestedCount: Int
        get() = delivery?.requestedCount ?: tracks.size
    val rankedCount: Int
        get() = delivery?.rankedCount ?: tracks.size
    val resolvedCount: Int
        get() = delivery?.resolvedCount ?: tracks.count { it.status != QueueStatus.NOT_IN_LIBRARY }
    val notInLibraryCount: Int
        get() = delivery?.notInLibraryCount ?: tracks.count { it.status == QueueStatus.NOT_IN_LIBRARY }
    val queueFailedCount: Int
        get() = delivery?.queueFailedCount ?: tracks.count { it.status == QueueStatus.QUEUE_FAILED }
    val failedCount: Int get() = (requestedCount - queuedCount).coerceAtLeast(0)
}

/**
 * UI state for the main screen.
 */
sealed class RadioUiState {
    object Idle : RadioUiState()
    data class Loading(
        val message: String = "Saving a durable radio request for the current Poweramp track",
    ) : RadioUiState()
    data class Searching(val message: String) : RadioUiState()
    data class Streaming(val result: RadioResult) : RadioUiState()
    data class Success(val result: RadioResult) : RadioUiState()
    data class Error(val message: String) : RadioUiState()
}
