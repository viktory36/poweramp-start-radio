package com.powerampstartradio.ui

import android.content.Context
import android.content.SharedPreferences

/** One immutable, canonical reading of the user-facing radio preferences. */
data class RadioSettingsSnapshot(
    val storedConfig: RadioConfig,
) {
    /** Saved dormant state is retained, but only effective controls reach a request. */
    val requestConfig: RadioConfig = storedConfig.forSelectionRequest()
}

class RadioSettingsStore private constructor(
    private val preferences: SharedPreferences,
) {
    fun readSnapshot(): RadioSettingsSnapshot = RadioSettingsCodec.decode(preferences.all)

    companion object {
        const val PREFERENCES_NAME = "settings"

        fun from(context: Context): RadioSettingsStore = RadioSettingsStore(
            context.getSharedPreferences(PREFERENCES_NAME, Context.MODE_PRIVATE),
        )
    }
}

/**
 * Canonical decoder for preferences which may predate the current discrete control surface.
 * Session-history configs deliberately do not pass through this codec.
 */
object RadioSettingsCodec {
    val RECOMMENDATION_COUNT_OPTIONS: List<Int> = (10..100 step 10).toList()

    private const val MAX_VALID_ARTIST_CONSTRAINT = 1_000

    fun decode(values: Map<String, *>): RadioSettingsSnapshot {
        val defaults = RadioConfig()
        val numTracks = canonicalRecommendationCount(
            values.int("num_tracks") ?: defaults.numTracks,
        )
        val anchorDecay = values.enumValue("anchor_decay", defaults.anchorDecay)
        val savedDriftMode = values.enumValue("drift_mode", defaults.driftMode)
        val savedAnchorStrength = values.float("anchor_strength") ?: defaults.anchorStrength
        val migrateFixedSeedAlias = savedDriftMode == DriftMode.SEED_INTERPOLATION &&
            anchorDecay == DecaySchedule.NONE && savedAnchorStrength == 1f
        val driftMode = if (migrateFixedSeedAlias) DriftMode.MOMENTUM else savedDriftMode
        val legacyLibraryAddedRange = values.enumValue(
            "library_added_range",
            defaults.libraryAddedRange,
        )
        val libraryAddedDays = if (values.containsKey("library_added_days")) {
            values.int("library_added_days")?.takeIf {
                it in 1..MAX_LIBRARY_ADDED_DAYS
            }
        } else {
            legacyLibraryAddedRange.dayCount?.toInt()
        }
        val stored = RadioConfig(
            numTracks = numTracks,
            libraryAddedRange = LibraryAddedRange.ALL_DATES,
            libraryAddedDays = libraryAddedDays,
            selectionMode = values.enumValue("selection_mode", defaults.selectionMode),
            driftEnabled = values.boolean("drift_enabled") ?: defaults.driftEnabled,
            driftMode = driftMode,
            anchorStrength = canonicalAnchorStrength(
                savedAnchorStrength,
                anchorDecay,
            ),
            walkRestartAlpha = canonicalGraphStopChance(
                values.float("walk_restart_alpha") ?: defaults.walkRestartAlpha,
            ),
            anchorDecay = anchorDecay,
            anchorHalfLifeTracks = canonicalFadeTiming(
                values.float("anchor_half_life_tracks") ?: defaults.anchorHalfLifeTracks,
            ),
            momentumBeta = if (migrateFixedSeedAlias) {
                1f
            } else {
                canonicalMomentum(values.float("momentum_beta") ?: defaults.momentumBeta)
            },
            diversityLambda = canonicalMmrRelevance(
                values.float("diversity_lambda") ?: defaults.diversityLambda,
            ),
            mmrCandidatePoolFraction = canonicalMmrReach(
                values.float("mmr_candidate_pool_fraction")
                    ?: defaults.mmrCandidatePoolFraction,
            ),
            dppFixedCandidatePoolFraction = canonicalDppFixedReach(
                values.float("dpp_fixed_candidate_pool_fraction")
                    ?: defaults.dppFixedCandidatePoolFraction,
            ),
            dppUsesCertifiedFullDomain = values.boolean("dpp_uses_certified_full_domain")
                ?: defaults.dppUsesCertifiedFullDomain,
            dppQualityExponent = canonicalDppSeedPull(
                values.float("dpp_quality_exponent") ?: defaults.dppQualityExponent,
            ),
            shuffleSeed = canonicalShuffleSeed(
                values.long("shuffle_seed") ?: defaults.shuffleSeed,
            ),
            artistLimitsEnabled = values.boolean("artist_limits_enabled")
                ?: defaults.artistLimitsEnabled,
            maxPerArtist = canonicalMaxPerArtist(
                values.int("max_per_artist") ?: defaults.maxPerArtist,
            ),
            minArtistSpacing = canonicalMinArtistSpacing(
                values.int("min_artist_spacing") ?: defaults.minArtistSpacing,
            ),
        )
        return RadioSettingsSnapshot(storedConfig = stored)
    }

    fun canonicalRecommendationCount(value: Int): Int {
        val defaults = RadioConfig()
        if (value !in RECOMMENDATION_COUNT_OPTIONS.first()..RECOMMENDATION_COUNT_OPTIONS.last()) {
            return defaults.numTracks
        }
        return RECOMMENDATION_COUNT_OPTIONS.minBy { option ->
            kotlin.math.abs(option.toLong() - value.toLong())
        }
    }

    fun canonicalAnchorStrength(value: Float, decay: DecaySchedule): Float = canonicalOption(
        options = DriftControlPolicy.seedPullOptions(decay),
        value = value,
        default = RadioConfig().anchorStrength,
    )

    fun canonicalGraphStopChance(value: Float): Float = canonicalOption(
        options = SelectionKnobPolicy.GRAPH_STOP_OPTIONS,
        value = value,
        default = RadioConfig().walkRestartAlpha,
    )

    fun canonicalFadeTiming(value: Float): Float = canonicalOption(
        options = DriftControlPolicy.FADE_TIMING_OPTIONS,
        value = value,
        default = RadioConfig.DEFAULT_ANCHOR_HALF_LIFE_TRACKS,
    )

    fun canonicalMomentum(value: Float): Float = canonicalOption(
        options = DriftControlPolicy.MOMENTUM_OPTIONS,
        value = value,
        default = RadioConfig.DEFAULT_MOMENTUM_BETA,
    )

    fun canonicalMmrRelevance(value: Float): Float = canonicalOption(
        options = SelectionKnobPolicy.MMR_RELEVANCE_OPTIONS,
        value = value,
        default = RadioConfig().diversityLambda,
    )

    fun canonicalMmrReach(value: Float): Float = canonicalOption(
        options = NeighborhoodReachPolicy.MMR_OPTIONS,
        value = value,
        default = RadioConfig.DEFAULT_MMR_CANDIDATE_POOL_FRACTION,
    )

    fun canonicalDppFixedReach(value: Float): Float = canonicalOption(
        options = NeighborhoodReachPolicy.DPP_FIXED_OPTIONS,
        value = value,
        default = RadioConfig.DEFAULT_DPP_FIXED_CANDIDATE_POOL_FRACTION,
    )

    fun canonicalDppSeedPull(value: Float): Float = canonicalOption(
        options = SelectionKnobPolicy.DPP_SEED_PULL_OPTIONS,
        value = value,
        default = RadioConfig.DEFAULT_DPP_QUALITY_EXPONENT,
    )

    fun canonicalShuffleSeed(value: Long): Long =
        value.takeIf { it != 0L } ?: RadioConfig.DEFAULT_SHUFFLE_SEED

    fun canonicalMaxPerArtist(value: Int): Int = value.takeIf {
        it in 1..MAX_VALID_ARTIST_CONSTRAINT
    } ?: RadioConfig().maxPerArtist

    fun canonicalMinArtistSpacing(value: Int): Int = value.takeIf {
        it in 0..MAX_VALID_ARTIST_CONSTRAINT
    } ?: RadioConfig().minArtistSpacing

    private fun canonicalOption(options: List<Float>, value: Float, default: Float): Float {
        val inRange = value.takeIf {
            it.isFinite() && it >= options.first() && it <= options.last()
        } ?: return default
        return SelectionKnobPolicy.nearestValue(options, inRange)
    }

    private fun Map<String, *>.int(key: String): Int? = this[key] as? Int

    private fun Map<String, *>.long(key: String): Long? = this[key] as? Long

    private fun Map<String, *>.float(key: String): Float? = this[key] as? Float

    private fun Map<String, *>.boolean(key: String): Boolean? = this[key] as? Boolean

    private inline fun <reified T : Enum<T>> Map<String, *>.enumValue(
        key: String,
        default: T,
    ): T = (this[key] as? String)
        ?.let { raw -> enumValues<T>().firstOrNull { it.name == raw } }
        ?: default
}
