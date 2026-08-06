package com.powerampstartradio.ui

import java.util.Locale

/** Exhaustive state of one explicit Settings Peek planning run. */
sealed interface SettingsPeekPreview {
    data object Loading : SettingsPeekPreview

    data class Ready(
        val snapshot: PlanSnapshot,
        val publicationContext: SettingsPeekPublicationContext,
        val firstDisplayLabels: List<String>,
        val comparisonLine: String? = null,
    ) : SettingsPeekPreview {
        init {
            require(snapshot.generation == publicationContext.generation)
            require(snapshot.providerGenerationId == publicationContext.providerGenerationId)
            require(firstDisplayLabels.size <= DISPLAY_LIMIT)
            require(firstDisplayLabels.size <= snapshot.orderedTrackIds.size)
        }

        /** Exactly one quiet evidence line is shown above the track labels. */
        val resultLine: String
            get() {
                comparisonLine?.let { return it }
                val planned = String.format(Locale.US, "%,d", snapshot.orderedTrackIds.size)
                val requested = snapshot.semanticControls
                    .firstOrNull { it.id == "queue_size" }
                    ?.valueKey
                    ?.removePrefix("i32:")
                    ?.toIntOrNull()
                    ?: snapshot.orderedTrackIds.size
                val requestedText = String.format(Locale.US, "%,d", requested)
                val shown = firstDisplayLabels.size
                return buildString {
                    if (snapshot.orderedTrackIds.size == requested) {
                        append("$planned-track queue preview")
                    } else {
                        append("$planned of $requestedText requested tracks in queue preview")
                    }
                    if (shown in 1 until snapshot.orderedTrackIds.size) {
                        append(" \u00b7 first $shown shown")
                    }
                    append('.')
                }
            }
    }

    data class Unavailable(
        val reason: SettingsPeekUnavailableReason,
    ) : SettingsPeekPreview {
        val resultLine: String get() = reason.userAction
    }

    data class Error(
        val resultLine: String,
    ) : SettingsPeekPreview {
        init {
            require(resultLine.isNotBlank())
        }
    }

    companion object {
        const val DISPLAY_LIMIT = 10
    }
}

enum class SettingsPeekUnavailableReason(val userAction: String) {
    NO_PROVIDER_VERIFIED_CURRENT_TRACK(
        "Current Poweramp track could not be read. Return to Poweramp, then preview again.",
    ),
    NO_ACTIVE_VALIDATED_LIBRARY(
        "No music index is ready. Import one or finish indexing.",
    ),
    SEED_ABSENT_FROM_ACTIVE_DOMAIN(
        "The current track is not in the music index. Index it, then preview again.",
    ),
    CONTEXT_CHANGED_DURING_PLANNING(
        "The current track or music index changed. Preview again for a fresh result.",
    ),
    FOREGROUND_REVALIDATION_REQUIRED(
        "Preview again for a fresh result.",
    ),
}

/** Exact external state against which one completed Peek plan may be published. */
data class SettingsPeekPublicationContext(
    val generation: RadioGenerationToken,
    val providerGenerationId: String,
    val seedPowerampFileId: Long,
) {
    init {
        require(providerGenerationId.isNotBlank())
        require(seedPowerampFileId > 0L)
    }
}

/** Pure final gate; invalidation and publication synchronize around this decision in the VM. */
internal object SettingsPeekPublicationPolicy {
    fun canPublish(
        planningRunId: String,
        activePlanningRunId: String?,
        plannedContext: SettingsPeekPublicationContext,
        currentContext: SettingsPeekPublicationContext?,
    ): Boolean = planningRunId == activePlanningRunId && plannedContext == currentContext
}

/** Accepts a verified seed only when complete provider evidence stayed stable around it. */
internal object SettingsPeekPublicationContextBracket {
    fun coherentProviderGenerationOrNull(
        providerGenerationBefore: String?,
        providerGenerationAfter: String?,
        verifiedSeedPowerampFileId: Long,
        providerAfterContainsVerifiedSeed: Boolean,
    ): String? {
        if (verifiedSeedPowerampFileId <= 0L || !providerAfterContainsVerifiedSeed) return null
        return providerGenerationBefore?.takeIf {
            it.isNotBlank() && it == providerGenerationAfter
        }
    }
}

internal object SettingsPeekContextInvalidationPolicy {
    fun invalidate(
        previews: Map<SelectionMode, SettingsPeekPreview>,
        reason: SettingsPeekUnavailableReason =
            SettingsPeekUnavailableReason.CONTEXT_CHANGED_DURING_PLANNING,
    ): Map<SelectionMode, SettingsPeekPreview> = previews.keys.associateWith {
        SettingsPeekPreview.Unavailable(reason)
    }
}

internal object SettingsPeekInteractionPolicy {
    /** Opening Peek is an explicit fresh run; hiding it never starts work. */
    fun requestsFreshPlan(
        expanded: Boolean,
        preview: SettingsPeekPreview?,
    ): Boolean = !expanded && preview !is SettingsPeekPreview.Loading
}

internal object SettingsPeekErrorPresentation {
    fun from(@Suppress("UNUSED_PARAMETER") error: Throwable): SettingsPeekPreview.Error =
        SettingsPeekPreview.Error(
            "Queue preview could not be verified. Close it and try again.",
        )
}

/** Retains only the last fresh snapshot for each mode; UI invalidation does not clear it. */
internal class SettingsPeekComparisonHistory {
    private val lastFreshByMode = mutableMapOf<SelectionMode, PlanSnapshot>()

    @Synchronized
    fun compareAndRemember(snapshot: PlanSnapshot): PlanControlResponse? {
        if (snapshot.materialization != PlanMaterialization.FRESH) return null
        val previous = lastFreshByMode.put(snapshot.selectionMode, snapshot) ?: return null
        return (PlanResponseComparator.compare(previous, snapshot) as? PlanComparisonResult.Compared)
            ?.response
    }
}

/** Builds only controls that are effective for this mode and configuration. */
object PlanSemanticControlPolicy {
    fun fromConfig(
        source: RadioConfig,
        eligibleCandidateIdentityCount: Int? = null,
    ): List<PlanSemanticControl> {
        val config = source.forExactCandidateDomain(eligibleCandidateIdentityCount)
        return buildList {
            add(integer("queue_size", "Queue size", config.numTracks, "${config.numTracks} tracks"))
            add(choice(
                "library_added_days",
                "Added to Poweramp",
                config.effectiveLibraryAddedDays?.toString() ?: "all",
                libraryAddedDaysLabel(config.effectiveLibraryAddedDays),
            ))
            add(toggle("artist_limits", "Artist limits", config.artistLimitsEnabled))
            if (config.artistLimitsEnabled) {
                val controls = ArtistConstraintControlPolicy.forRequest(
                    recommendationCount = config.numTracks,
                    maxPerArtist = config.maxPerArtist,
                    minArtistSpacing = config.minArtistSpacing,
                )
                if (controls.showMaximum) {
                    add(integer(
                        "max_per_artist",
                        "Maximum tracks per artist credit",
                        config.maxPerArtist,
                        config.maxPerArtist.toString(),
                    ))
                }
                if (controls.showSpacing) {
                    add(integer(
                        "artist_spacing",
                        "Tracks between the same artist credit",
                        config.minArtistSpacing,
                        "${config.minArtistSpacing} tracks",
                    ))
                }
            }

            when (config.selectionMode) {
                SelectionMode.CLOSEST -> Unit
                SelectionMode.MMR -> {
                    add(float(
                        "mmr_relevance",
                        if (config.driftEnabled) {
                            "Current-direction relevance"
                        } else {
                            "Seed relevance"
                        },
                        config.diversityLambda,
                        percent(config.diversityLambda),
                    ))
                    if (MmrControlPolicy.shouldExposeReach(
                            config = config,
                            eligibleCandidateIdentityCount = eligibleCandidateIdentityCount,
                        )) {
                        add(float(
                            "mmr_reach",
                            "Neighborhood reach",
                            config.effectiveMmrCandidatePoolFraction,
                            percent(config.effectiveMmrCandidatePoolFraction),
                        ))
                    }
                    add(toggle("drift_enabled", "Drift", config.driftEnabled))
                    if (config.driftEnabled) addDriftControls(config)
                }
                SelectionMode.DPP -> {
                    add(float(
                        "dpp_seed_pull",
                        "DPP seed pull",
                        config.effectiveDppQualityExponent,
                        SelectionControlText.dppSeedPullLabel(
                            config.effectiveDppQualityExponent,
                        ),
                    ))
                    add(choice(
                        "dpp_domain",
                        "DPP search range",
                        if (config.dppUsesCertifiedFullDomain) "full" else "fixed",
                        if (config.dppUsesCertifiedFullDomain) {
                            "All eligible recordings"
                        } else {
                            "Nearest subset"
                        },
                    ))
                    if (!config.dppUsesCertifiedFullDomain &&
                        NeighborhoodReachPolicy.hasMultipleDistinctDomains(
                            options = NeighborhoodReachPolicy.DPP_FIXED_OPTIONS,
                            eligibleCandidateIdentityCount = eligibleCandidateIdentityCount,
                            numTracks = config.numTracks,
                            preferredFraction = config.effectiveDppFixedCandidatePoolFraction,
                        )
                    ) {
                        add(float(
                            "dpp_fixed_reach",
                            "Search neighborhood",
                            config.effectiveDppFixedCandidatePoolFraction,
                            percent(config.effectiveDppFixedCandidatePoolFraction),
                        ))
                    }
                }
                SelectionMode.RANDOM_WALK -> add(float(
                    "graph_stop_chance",
                    "Stop chance after each track-to-track move",
                    config.walkRestartAlpha,
                    percent(config.walkRestartAlpha),
                ))
                SelectionMode.UNIFORM_SHUFFLE -> add(choice(
                    "shuffle_seed",
                    "Reproducible order",
                    java.lang.Long.toUnsignedString(config.effectiveShuffleSeed, 16),
                    "Stored order",
                ))
            }
        }
    }

    fun semanticCandidateCount(
        config: RadioConfig,
        eligibleCandidateIdentityCount: Int,
    ): Int {
        require(eligibleCandidateIdentityCount >= 0)
        val effectiveConfig = config.forExactCandidateDomain(eligibleCandidateIdentityCount)
        return when {
            effectiveConfig.selectionMode == SelectionMode.MMR &&
                MmrControlPolicy.reachCanAffectOutput(effectiveConfig) ->
                effectiveConfig.resolveCandidatePoolSize(eligibleCandidateIdentityCount)
            effectiveConfig.selectionMode == SelectionMode.DPP &&
                !effectiveConfig.dppUsesCertifiedFullDomain ->
                effectiveConfig.resolveCandidatePoolSize(eligibleCandidateIdentityCount)
            else -> eligibleCandidateIdentityCount
        }
    }

    private fun RadioConfig.forExactCandidateDomain(
        eligibleCandidateIdentityCount: Int?,
    ): RadioConfig {
        val active = forActiveSelectionMode()
        if (active.selectionMode != SelectionMode.DPP) return active
        val useFullDomain = DppDomainControlPolicy.effectiveUsesCertifiedFullDomain(
            storedUsesCertifiedFullDomain = active.dppUsesCertifiedFullDomain,
            eligibleCandidateIdentityCount = eligibleCandidateIdentityCount,
            numTracks = active.numTracks,
        )
        return active.copy(
            dppUsesCertifiedFullDomain = useFullDomain,
            dppFixedCandidatePoolFraction = if (useFullDomain) {
                RadioConfig.DEFAULT_DPP_FIXED_CANDIDATE_POOL_FRACTION
            } else {
                active.dppFixedCandidatePoolFraction
            },
        )
    }

    private fun MutableList<PlanSemanticControl>.addDriftControls(config: RadioConfig) {
        add(choice(
            "drift_mode",
            "Drift mode",
            config.driftMode.name,
            SelectionControlText.driftModeLabel(config.driftMode),
        ))
        when (config.driftMode) {
            DriftMode.MOMENTUM -> add(float(
                "momentum",
                "Prior-direction memory",
                config.momentumBeta,
                percent(config.momentumBeta),
            ))
            DriftMode.SEED_INTERPOLATION -> {
                add(float(
                    "starting_seed_pull",
                    "Starting seed pull",
                    config.anchorStrength,
                    percent(config.anchorStrength),
                ))
                if (DriftControlPolicy.seedFadeApplies(config.anchorStrength)) {
                    add(choice(
                        "seed_pull_fade",
                        "Seed pull fade",
                        config.anchorDecay.name,
                        when (config.anchorDecay) {
                            DecaySchedule.NONE -> "Hold"
                            DecaySchedule.LINEAR -> "Linear"
                            DecaySchedule.EXPONENTIAL -> "Exponential"
                            DecaySchedule.STEP -> "Step"
                        },
                    ))
                    if (config.anchorDecay != DecaySchedule.NONE) {
                        add(float(
                            "seed_pull_fade_timing",
                            when (config.anchorDecay) {
                                DecaySchedule.LINEAR -> "Half-strength point"
                                DecaySchedule.EXPONENTIAL -> "Half-life"
                                DecaySchedule.STEP -> "Drop point"
                                DecaySchedule.NONE -> error("Hold has no fade-timing control")
                            },
                            config.effectiveAnchorHalfLifeTracks,
                            if (config.anchorDecay == DecaySchedule.STEP) {
                                "After ${DriftControlPolicy.stepDropAfterPickCount(config.effectiveAnchorHalfLifeTracks)} picks"
                            } else {
                                "${concise(config.effectiveAnchorHalfLifeTracks)} picks"
                            },
                        ))
                    }
                }
            }
        }
    }

    private fun toggle(id: String, name: String, value: Boolean) = choice(
        id = id,
        name = name,
        key = value.toString(),
        display = if (value) "On" else "Off",
    )

    private fun integer(
        id: String,
        name: String,
        value: Int,
        display: String,
    ) = PlanSemanticControl(id, "i32:$value", name, display)

    private fun float(
        id: String,
        name: String,
        value: Float,
        display: String,
    ) = PlanSemanticControl(
        id,
        "f32:" + Integer.toUnsignedString(value.toRawBits(), 16).padStart(8, '0'),
        name,
        display,
    )

    private fun choice(
        id: String,
        name: String,
        key: String,
        display: String,
    ) = PlanSemanticControl(id, "choice:$key", name, display)

    private fun percent(value: Float): String {
        val scaled = value * 100f
        val formatted = String.format(
            Locale.ROOT,
            if (kotlin.math.abs(scaled) < 1f) "%.2f" else "%.1f",
            scaled,
        ).trimEnd('0').trimEnd('.')
        return "$formatted%"
    }

    private fun concise(value: Float): String =
        String.format(Locale.ROOT, "%.4f", value).trimEnd('0').trimEnd('.')
}
