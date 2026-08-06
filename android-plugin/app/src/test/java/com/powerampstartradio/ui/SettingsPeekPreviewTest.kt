package com.powerampstartradio.ui

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class SettingsPeekPreviewTest {
    @Test
    fun `automatic DPP exposes its full semantic domain and no proof prefix control`() {
        val config = RadioConfig(
            selectionMode = SelectionMode.DPP,
            dppUsesCertifiedFullDomain = true,
            dppFixedCandidatePoolFraction = 0.5f,
            driftEnabled = true,
        )

        val controls = PlanSemanticControlPolicy.fromConfig(config)

        assertEquals(80_322, PlanSemanticControlPolicy.semanticCandidateCount(config, 80_322))
        assertEquals(
            listOf(
                "queue_size",
                "library_added_days",
                "artist_limits",
                "max_per_artist",
                "artist_spacing",
                "dpp_seed_pull",
                "dpp_domain",
            ),
            controls.map { it.id },
        )
        assertFalse(controls.any { it.id.contains("drift") || it.id.contains("reach") })
    }

    @Test
    fun `added-date semantic control carries exact days rather than a preset`() {
        val control = PlanSemanticControlPolicy.fromConfig(
            RadioConfig(libraryAddedDays = 17),
        ).single { it.id == "library_added_days" }

        assertEquals("choice:17", control.valueKey)
        assertEquals("Last 17 days", control.displayValue)
    }

    @Test
    fun `fixed selectors expose their resolved neighborhood while full modes use active domain`() {
        val mmr = RadioConfig(
            selectionMode = SelectionMode.MMR,
            mmrCandidatePoolFraction = 0.02f,
        )
        val fixedDpp = RadioConfig(
            selectionMode = SelectionMode.DPP,
            dppUsesCertifiedFullDomain = false,
            dppFixedCandidatePoolFraction = 0.02f,
        )

        assertEquals(1_606, PlanSemanticControlPolicy.semanticCandidateCount(mmr, 80_322))
        assertEquals(1_606, PlanSemanticControlPolicy.semanticCandidateCount(fixedDpp, 80_322))
        assertEquals(
            80_322,
            PlanSemanticControlPolicy.semanticCandidateCount(
                RadioConfig(selectionMode = SelectionMode.RANDOM_WALK),
                80_322,
            ),
        )
    }

    @Test
    fun `candidate domain consumes the exact seed-excluded identity count`() {
        assertEquals(
            97,
            PlanSemanticControlPolicy.semanticCandidateCount(
                config = RadioConfig(selectionMode = SelectionMode.CLOSEST),
                eligibleCandidateIdentityCount = 97,
            ),
        )
        assertEquals(
            97,
            PlanSemanticControlPolicy.semanticCandidateCount(
                config = RadioConfig(
                    selectionMode = SelectionMode.MMR,
                    mmrCandidatePoolFraction = 1f,
                ),
                eligibleCandidateIdentityCount = 97,
            ),
        )
    }

    @Test
    fun `disabled policies omit saved subordinate settings`() {
        val driftOffA = RadioConfig(
            selectionMode = SelectionMode.MMR,
            driftEnabled = false,
            driftMode = DriftMode.SEED_INTERPOLATION,
            anchorStrength = 0.25f,
            anchorHalfLifeTracks = 3f,
        )
        val driftOffB = driftOffA.copy(
            driftMode = DriftMode.MOMENTUM,
            anchorStrength = 1f,
            anchorHalfLifeTracks = 30f,
            momentumBeta = 0.25f,
        )
        val artistOffA = driftOffA.copy(
            artistLimitsEnabled = false,
            maxPerArtist = 1,
            minArtistSpacing = 1,
        )
        val artistOffB = artistOffA.copy(maxPerArtist = 10, minArtistSpacing = 20)

        assertEquals(
            PlanSemanticControlPolicy.fromConfig(driftOffA),
            PlanSemanticControlPolicy.fromConfig(driftOffB),
        )
        assertEquals(
            PlanSemanticControlPolicy.fromConfig(artistOffA),
            PlanSemanticControlPolicy.fromConfig(artistOffB),
        )
    }

    @Test
    fun `pure relevance MMR omits reach unless artist limits can require refill`() {
        val unfiltered = RadioConfig(
            selectionMode = SelectionMode.MMR,
            diversityLambda = 1f,
            artistLimitsEnabled = false,
        )
        val filtered = unfiltered.copy(artistLimitsEnabled = true)
        val ineffectiveFilter = filtered.copy(
            maxPerArtist = filtered.numTracks,
            minArtistSpacing = 0,
        )

        assertFalse(
            PlanSemanticControlPolicy.fromConfig(unfiltered).any { it.id == "mmr_reach" },
        )
        assertTrue(
            PlanSemanticControlPolicy.fromConfig(filtered).any { it.id == "mmr_reach" },
        )
        assertFalse(
            PlanSemanticControlPolicy.fromConfig(ineffectiveFilter)
                .any { it.id == "mmr_reach" },
        )
        assertEquals(
            500,
            PlanSemanticControlPolicy.semanticCandidateCount(
                config = ineffectiveFilter,
                eligibleCandidateIdentityCount = 500,
            ),
        )
    }

    @Test
    fun `one exact reach domain is not a semantic control`() {
        val mmr = RadioConfig(
            numTracks = 10,
            selectionMode = SelectionMode.MMR,
            diversityLambda = 0.4f,
        )
        val fixedDpp = mmr.copy(
            selectionMode = SelectionMode.DPP,
            dppUsesCertifiedFullDomain = false,
        )

        assertFalse(
            PlanSemanticControlPolicy.fromConfig(
                source = mmr,
                eligibleCandidateIdentityCount = 100,
            ).any { it.id == "mmr_reach" },
        )
        assertFalse(
            PlanSemanticControlPolicy.fromConfig(
                source = fixedDpp,
                eligibleCandidateIdentityCount = 100,
            ).any { it.id == "dpp_fixed_reach" },
        )
        assertEquals(
            "choice:full",
            PlanSemanticControlPolicy.fromConfig(
                source = fixedDpp,
                eligibleCandidateIdentityCount = 100,
            ).single { it.id == "dpp_domain" }.valueKey,
        )
        assertTrue(
            PlanSemanticControlPolicy.fromConfig(
                source = mmr,
                eligibleCandidateIdentityCount = 101,
            ).any { it.id == "mmr_reach" },
        )
    }

    @Test
    fun `artist comparison schema omits only structurally inactive controls`() {
        val maximumOne = RadioConfig(
            numTracks = 10,
            artistLimitsEnabled = true,
            maxPerArtist = 1,
            minArtistSpacing = 3,
        )
        val noRepeatSpacing = maximumOne.copy(
            maxPerArtist = 8,
            minArtistSpacing = 9,
        )
        val redundantButAdjustableMaximum = noRepeatSpacing.copy(minArtistSpacing = 3)

        assertFalse(
            PlanSemanticControlPolicy.fromConfig(maximumOne)
                .any { it.id == "artist_spacing" },
        )
        assertFalse(
            PlanSemanticControlPolicy.fromConfig(noRepeatSpacing)
                .any { it.id == "max_per_artist" },
        )
        assertTrue(
            PlanSemanticControlPolicy.fromConfig(redundantButAdjustableMaximum)
                .any { it.id == "max_per_artist" },
        )
    }

    @Test
    fun `zero starting seed pull omits meaningless fade controls`() {
        val controls = PlanSemanticControlPolicy.fromConfig(
            RadioConfig(
                selectionMode = SelectionMode.MMR,
                driftEnabled = true,
                driftMode = DriftMode.SEED_INTERPOLATION,
                anchorStrength = 0f,
                anchorDecay = DecaySchedule.EXPONENTIAL,
                anchorHalfLifeTracks = 30f,
            ),
        )

        assertTrue(controls.any { it.id == "starting_seed_pull" })
        assertFalse(controls.any { it.id == "seed_pull_fade" })
        assertFalse(controls.any { it.id == "seed_pull_fade_timing" })
    }

    @Test
    fun `short queue Peek states the effective Step outcome rather than dormant timing`() {
        val stored = RadioConfig(
            numTracks = 10,
            selectionMode = SelectionMode.MMR,
            driftEnabled = true,
            driftMode = DriftMode.SEED_INTERPOLATION,
            anchorStrength = 0.75f,
            anchorDecay = DecaySchedule.STEP,
            anchorHalfLifeTracks = 30f,
        )
        val request = stored.forSelectionRequest()

        val timing = PlanSemanticControlPolicy.fromConfig(request)
            .single { it.id == "seed_pull_fade_timing" }
        assertEquals(30f, stored.anchorHalfLifeTracks, 0f)
        assertEquals(7f, request.anchorHalfLifeTracks, 0f)
        assertEquals("Drop point", timing.displayName)
        assertEquals("After 8 picks", timing.displayValue)
    }

    @Test
    fun `comparison copy uses control language and bounded percentage precision`() {
        val fixed = PlanSemanticControlPolicy.fromConfig(
            RadioConfig(
                selectionMode = SelectionMode.MMR,
                diversityLambda = 0.123456f,
                mmrCandidatePoolFraction = 0.0025f,
                driftEnabled = false,
            ),
        )
        assertEquals(
            "Seed relevance",
            fixed.single { it.id == "mmr_relevance" }.displayName,
        )
        assertEquals(
            "12.3%",
            fixed.single { it.id == "mmr_relevance" }.displayValue,
        )
        assertEquals(
            "0.25%",
            fixed.single { it.id == "mmr_reach" }.displayValue,
        )

        val drifting = PlanSemanticControlPolicy.fromConfig(
            RadioConfig(
                selectionMode = SelectionMode.MMR,
                diversityLambda = 0.5f,
                driftEnabled = true,
            ),
        )
        assertEquals(
            "Current-direction relevance",
            drifting.single { it.id == "mmr_relevance" }.displayName,
        )
    }

    @Test
    fun `comparison labels match the visible setting concepts`() {
        val artistControls = PlanSemanticControlPolicy.fromConfig(
            RadioConfig(
                selectionMode = SelectionMode.CLOSEST,
                artistLimitsEnabled = true,
                maxPerArtist = 3,
                minArtistSpacing = 2,
            ),
        ).associateBy { it.id }
        assertEquals(
            "Maximum tracks per artist credit",
            artistControls.getValue("max_per_artist").displayName,
        )
        assertEquals(
            "Tracks between the same artist credit",
            artistControls.getValue("artist_spacing").displayName,
        )

        val graphControl = PlanSemanticControlPolicy.fromConfig(
            RadioConfig(selectionMode = SelectionMode.RANDOM_WALK),
        ).single { it.id == "graph_stop_chance" }
        assertEquals("Stop chance after each track-to-track move", graphControl.displayName)

        val momentumControl = PlanSemanticControlPolicy.fromConfig(
            RadioConfig(
                selectionMode = SelectionMode.MMR,
                driftEnabled = true,
                driftMode = DriftMode.MOMENTUM,
            ),
        ).single { it.id == "momentum" }
        assertEquals("Prior-direction memory", momentumControl.displayName)

        val exponentialTiming = PlanSemanticControlPolicy.fromConfig(
            RadioConfig(
                selectionMode = SelectionMode.MMR,
                driftEnabled = true,
                driftMode = DriftMode.SEED_INTERPOLATION,
                anchorStrength = 0.75f,
                anchorDecay = DecaySchedule.EXPONENTIAL,
            ),
        ).single { it.id == "seed_pull_fade_timing" }
        assertEquals("Half-life", exponentialTiming.displayName)
    }

    @Test
    fun `history compares the next fresh peek without a UI invalidation callback`() {
        val history = SettingsPeekComparisonHistory()
        val before = snapshot(
            runId = "peek-1",
            config = RadioConfig(
                selectionMode = SelectionMode.MMR,
                mmrCandidatePoolFraction = 0.02f,
            ),
            candidates = 1_606,
        )
        val after = snapshot(
            runId = "peek-2",
            config = beforeConfig().copy(mmrCandidatePoolFraction = 0.05f),
            candidates = 4_016,
        )

        assertNull(history.compareAndRemember(before))
        val response = history.compareAndRemember(after)

        assertEquals("mmr_reach", response?.changedControl?.id)
        assertEquals(2_410, response?.candidates?.delta)
        assertEquals(QueuePlanResponseKind.EXACTLY_UNCHANGED, response?.queue?.kind)
    }

    @Test
    fun `structured preview retains full order but renders at most ten labels`() {
        val snapshot = snapshot(
            runId = "peek-1",
            config = beforeConfig(),
            candidates = 1_606,
            trackIds = (1L..30L).toList(),
        )
        val preview = SettingsPeekPreview.Ready(
            snapshot = snapshot,
            publicationContext = publicationContext(),
            firstDisplayLabels = (1..10).map { "Track $it" },
        )

        assertEquals(30, preview.snapshot.orderedTrackIds.size)
        assertEquals(10, preview.firstDisplayLabels.size)
        assertEquals(
            "30 of 50 requested tracks in queue preview \u00b7 first 10 shown.",
            preview.resultLine,
        )
    }

    @Test
    fun `pure relevance MMR does not present its hidden retrieval prefix as a domain`() {
        val config = RadioConfig(
            numTracks = 10,
            selectionMode = SelectionMode.MMR,
            diversityLambda = 1f,
            artistLimitsEnabled = true,
            maxPerArtist = 10,
            minArtistSpacing = 0,
        )
        val preview = SettingsPeekPreview.Ready(
            snapshot = snapshot(
                runId = "peek-pure-relevance",
                config = config,
                candidates = 500,
            ),
            publicationContext = publicationContext(),
            firstDisplayLabels = emptyList(),
        )

        assertEquals("3 of 10 requested tracks in queue preview.", preview.resultLine)
    }

    @Test
    fun `preview leads with queue outcome instead of retrieval internals`() {
        val config = beforeConfig().copy(driftEnabled = true)
        val preview = SettingsPeekPreview.Ready(
            snapshot = snapshot(
                runId = "peek-drift-frontier",
                config = config,
                candidates = 1_606,
            ),
            publicationContext = publicationContext(),
            firstDisplayLabels = emptyList(),
        )

        assertEquals(
            "3 of 50 requested tracks in queue preview.",
            preview.resultLine,
        )
    }

    @Test
    fun `every unavailable outcome names the missing prerequisite and an action`() {
        val lines = SettingsPeekUnavailableReason.entries.map { reason ->
            SettingsPeekPreview.Unavailable(reason).resultLine
        }

        assertEquals(5, lines.size)
        assertTrue(lines.any { "Current Poweramp track could not be read" in it })
        assertTrue(lines.any { "No music index is ready" in it })
        assertTrue(lines.any { "not in the music index" in it })
        assertTrue(lines.any { "music index changed" in it })
        assertTrue(lines.all { line ->
            "preview" in line.lowercase() || "Import one or finish indexing" in line
        })
    }

    @Test
    fun `opening a terminal or empty Peek explicitly reruns but hiding and loading do not`() {
        val unavailable = SettingsPeekPreview.Unavailable(
            SettingsPeekUnavailableReason.NO_PROVIDER_VERIFIED_CURRENT_TRACK,
        )
        val failed = SettingsPeekPreview.Error("Peek planning failed: test failure.")
        val ready = SettingsPeekPreview.Ready(
            snapshot("peek-ready", beforeConfig(), candidates = 10),
            publicationContext = publicationContext(),
            firstDisplayLabels = emptyList(),
        )

        assertTrue(SettingsPeekInteractionPolicy.requestsFreshPlan(false, null))
        assertTrue(SettingsPeekInteractionPolicy.requestsFreshPlan(false, ready))
        assertTrue(SettingsPeekInteractionPolicy.requestsFreshPlan(false, unavailable))
        assertTrue(SettingsPeekInteractionPolicy.requestsFreshPlan(false, failed))
        assertFalse(
            SettingsPeekInteractionPolicy.requestsFreshPlan(
                false,
                SettingsPeekPreview.Loading,
            ),
        )
        assertFalse(SettingsPeekInteractionPolicy.requestsFreshPlan(true, unavailable))
    }

    @Test
    fun `planning errors never expose exception details`() {
        val error = IllegalStateException(
            "outer planning wrapper",
            IllegalArgumentException("graph row 18\nwas out of range"),
        )

        val presented = SettingsPeekErrorPresentation.from(error)

        assertEquals(
            "Queue preview could not be verified. Close it and try again.",
            presented.resultLine,
        )
        assertFalse("graph row" in presented.resultLine)
        assertFalse("IllegalArgumentException" in presented.resultLine)
    }

    @Test
    fun `publication requires the active run and exact live library and seed context`() {
        val planned = publicationContext()

        assertTrue(
            SettingsPeekPublicationPolicy.canPublish(
                planningRunId = "peek-7",
                activePlanningRunId = "peek-7",
                plannedContext = planned,
                currentContext = planned,
            ),
        )
        assertFalse(
            SettingsPeekPublicationPolicy.canPublish(
                "peek-7",
                "peek-8",
                planned,
                planned,
            ),
        )
        assertFalse(
            SettingsPeekPublicationPolicy.canPublish(
                "peek-7",
                "peek-7",
                planned,
                planned.copy(providerGenerationId = "provider-b"),
            ),
        )
        assertFalse(
            SettingsPeekPublicationPolicy.canPublish(
                "peek-7",
                "peek-7",
                planned,
                planned.copy(seedPowerampFileId = 99L),
            ),
        )
        assertFalse(
            SettingsPeekPublicationPolicy.canPublish(
                "peek-7",
                "peek-7",
                planned,
                planned.copy(generation = generation().copy(generationId = "generation-b")),
            ),
        )
        assertFalse(
            SettingsPeekPublicationPolicy.canPublish(
                "peek-7",
                "peek-7",
                planned,
                null,
            ),
        )
    }

    @Test
    fun `publication context requires stable provider evidence around verified seed`() {
        assertEquals(
            "provider-a",
            SettingsPeekPublicationContextBracket.coherentProviderGenerationOrNull(
                providerGenerationBefore = "provider-a",
                providerGenerationAfter = "provider-a",
                verifiedSeedPowerampFileId = 7L,
                providerAfterContainsVerifiedSeed = true,
            ),
        )
        assertNull(
            SettingsPeekPublicationContextBracket.coherentProviderGenerationOrNull(
                providerGenerationBefore = "provider-a",
                providerGenerationAfter = "provider-b",
                verifiedSeedPowerampFileId = 7L,
                providerAfterContainsVerifiedSeed = true,
            ),
        )
        assertNull(
            SettingsPeekPublicationContextBracket.coherentProviderGenerationOrNull(
                providerGenerationBefore = "provider-a",
                providerGenerationAfter = "provider-a",
                verifiedSeedPowerampFileId = 7L,
                providerAfterContainsVerifiedSeed = false,
            ),
        )
        assertNull(
            SettingsPeekPublicationContextBracket.coherentProviderGenerationOrNull(
                providerGenerationBefore = null,
                providerGenerationAfter = null,
                verifiedSeedPowerampFileId = 7L,
                providerAfterContainsVerifiedSeed = true,
            ),
        )
    }

    @Test
    fun `activity resume invalidation retires ready and in-flight Peek state`() {
        val ready = SettingsPeekPreview.Ready(
            snapshot("peek-ready", beforeConfig(), candidates = 10),
            publicationContext = publicationContext(),
            firstDisplayLabels = emptyList(),
        )
        val invalidated = SettingsPeekContextInvalidationPolicy.invalidate(
            previews = mapOf(
                SelectionMode.MMR to ready,
                SelectionMode.DPP to SettingsPeekPreview.Loading,
            ),
            reason = SettingsPeekUnavailableReason.FOREGROUND_REVALIDATION_REQUIRED,
        )

        assertEquals(setOf(SelectionMode.MMR, SelectionMode.DPP), invalidated.keys)
        assertTrue(invalidated.values.all { preview ->
            preview == SettingsPeekPreview.Unavailable(
                SettingsPeekUnavailableReason.FOREGROUND_REVALIDATION_REQUIRED,
            )
        })
        assertTrue(SettingsPeekContextInvalidationPolicy.invalidate(emptyMap()).isEmpty())
    }

    private fun beforeConfig() = RadioConfig(
        selectionMode = SelectionMode.MMR,
        mmrCandidatePoolFraction = 0.02f,
    )

    private fun snapshot(
        runId: String,
        config: RadioConfig,
        candidates: Int,
        trackIds: List<Long> = listOf(10L, 20L, 30L),
    ) = PlanSnapshot(
        planningRunId = runId,
        materialization = PlanMaterialization.FRESH,
        generation = generation(),
        providerGenerationId = "provider-a",
        seedIdentity = RadioSeedIdentity(41L, "stable-a"),
        selectionMode = config.selectionMode,
        semanticControls = PlanSemanticControlPolicy.fromConfig(config),
        candidateCount = candidates,
        orderedTrackIds = trackIds,
    )

    private fun generation() = RadioGenerationToken(
        generationId = "generation-a",
        activationBindingId = "activation-a",
        manifestSha256 = "a".repeat(64),
        embeddingSpecId = "clamp3-audio-v1",
        databaseContentSha256 = "b".repeat(64),
        orderedTrackSetSha256 = "c".repeat(64),
        stableTrackUidMappingSha256 = "d".repeat(64),
    )

    private fun publicationContext() = SettingsPeekPublicationContext(
        generation = generation(),
        providerGenerationId = "provider-a",
        seedPowerampFileId = 7L,
    )
}
