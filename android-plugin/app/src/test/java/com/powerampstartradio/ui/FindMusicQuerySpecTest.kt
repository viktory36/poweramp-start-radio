package com.powerampstartradio.ui

import com.powerampstartradio.data.StableIdentityGenerationBinding
import com.powerampstartradio.data.StableTrackIdentityCatalog
import com.powerampstartradio.data.StableTrackIdentityRow
import com.powerampstartradio.indexing.v2.StableTrackSpanIdentityStrength
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class FindMusicQuerySpecTest {
    @Test
    fun currentQuerySpecRoundTripsEveryRankingInput() {
        val original = FindMusicQuerySpec(
            operator = FindMusicOperator.REFINE,
            textIngredients = listOf(
                FindMusicTextIngredient("broken beat", 0.35f, negative = false),
            ),
            songSeeds = listOf(
                FindMusicSongAnchor(
                    trackId = 42L,
                    stableTrackSpanId = stableId(42),
                    artist = "Artist",
                    title = "Song",
                    weight = 0.65f,
                    negative = false,
                ),
            ),
            resultLimit = 75,
            refineSpec = FindMusicRefineSpec(
                primaryIngredientIndex = 1,
                neighborhood = FindMusicRefineNeighborhood.TOP_0_5_PERCENT,
            ),
            libraryAddedDays = 17,
            libraryBinding = binding("generation-a"),
        )

        val encoded = FindMusicQuerySpecCodec.toJsonArray(listOf(original))
        val decoded = FindMusicQuerySpecCodec.fromJsonArray(encoded)

        assertEquals(listOf(original), decoded)
        assertEquals(original.stateKey, decoded.single().stateKey)
        assertEquals(null, validateFindMusicQueryContract(decoded.single()))
        assertTrue(encoded.contains("\"operator\":\"refine\""))
        assertTrue(encoded.contains("\"primary_ingredient_index\":1"))
        assertTrue(encoded.contains("\"neighborhood\":\"top_0_5_percent\""))
    }

    @Test
    fun priorRankV3SingletonTextRemainsReadableButFailsTheCurrentRankingContract() {
        val prior = """[{"schema_version":2,"ranking_version":3,"operator":"all_of","result_limit":35,"text":{"query":"dark ambient","weight":0.3,"negative":true,"locked":true},"song_anchors":[{"track_id":7,"artist":"A","title":"B","weight":0.7,"negative":false,"locked":false}]}]"""

        val decoded = FindMusicQuerySpecCodec.fromJsonArray(prior).single()

        assertEquals(2, decoded.schemaVersion)
        assertEquals(3, decoded.rankingVersion)
        assertEquals(
            listOf(FindMusicTextIngredient("dark ambient", 0.3f, negative = true)),
            decoded.textIngredients,
        )
        assertEquals(35, decoded.resultLimit)
        assertEquals(LibraryAddedRange.ALL_DATES, decoded.libraryAddedRange)
        assertEquals(null, decoded.effectiveLibraryAddedDays)
        assertEquals(
            INCOMPATIBLE_SAVED_FIND_MUSIC_QUERY_MESSAGE,
            validateFindMusicQueryContract(decoded),
        )
    }

    @Test
    fun legacyAddedRangeAndExactDaysBothRoundTripWithTheSameEffectiveContract() {
        val legacy = FindMusicQuerySpec(
            textIngredients = listOf(
                FindMusicTextIngredient("ambient", 1f, negative = false),
            ),
            libraryAddedRange = LibraryAddedRange.LAST_30_DAYS,
        )
        val exact = legacy.copy(
            libraryAddedRange = LibraryAddedRange.ALL_DATES,
            libraryAddedDays = 30,
        )

        val decodedLegacy = FindMusicQuerySpecCodec.fromJsonArray(
            FindMusicQuerySpecCodec.toJsonArray(listOf(legacy)),
        ).single()
        val decodedExact = FindMusicQuerySpecCodec.fromJsonArray(
            FindMusicQuerySpecCodec.toJsonArray(listOf(exact)),
        ).single()

        assertEquals(30, decodedLegacy.effectiveLibraryAddedDays)
        assertEquals(30, decodedExact.effectiveLibraryAddedDays)
        assertFalse(legacy.stateKey == exact.stateKey)
    }

    @Test
    fun invalidExactAddedDaysAreRejectedWhileMissingMeansAllDates() {
        val allDates = FindMusicQuerySpec(
            textIngredients = listOf(
                FindMusicTextIngredient("ambient", 1f, negative = false),
            ),
        )
        val validJson = FindMusicQuerySpecCodec.toJson(allDates)
        val invalidJson = validJson.replace(
            "\"library_added_range\"",
            "\"library_added_days\":0,\"library_added_range\"",
        )

        assertEquals(
            null,
            FindMusicQuerySpecCodec.fromJsonArray("[$validJson]")
                .single()
                .effectiveLibraryAddedDays,
        )
        assertThrows(IllegalArgumentException::class.java) {
            FindMusicQuerySpecCodec.fromJsonArray("[$invalidJson]")
        }
    }

    @Test
    fun legacyStructuredSearchMigratesWithoutChangingWeightsOrSigns() {
        val legacy = """[{"text":"dark ambient","text_weight":0.3,"text_negative":true,"seeds":[{"id":7,"artist":"A","title":"B","weight":0.7,"negative":false}]}]"""

        val decoded = FindMusicQuerySpecCodec.fromJsonArray(legacy).single()

        assertEquals(FindMusicOperator.ALL_OF, decoded.operator)
        assertEquals("dark ambient", decoded.textIngredients.single().query)
        assertEquals(0.3f, decoded.textIngredients.single().weight)
        assertTrue(decoded.textIngredients.single().negative)
        assertEquals(7L, decoded.songSeeds.single().trackId)
        assertEquals(0.7f, decoded.songSeeds.single().weight)
        assertEquals(FindMusicQuerySpec.DEFAULT_RESULT_LIMIT, decoded.resultLimit)
        assertEquals(1, decoded.schemaVersion)
        assertEquals(FindMusicQuerySpec.LEGACY_RANKING_VERSION, decoded.rankingVersion)
        assertEquals(null, decoded.libraryBinding)
        assertEquals(
            INCOMPATIBLE_SAVED_FIND_MUSIC_QUERY_MESSAGE,
            validateFindMusicQueryContract(decoded),
        )
    }

    @Test
    fun legacyEditorLocksAreIgnoredBySemanticIdentity() {
        val legacy = """[{"schema_version":3,"ranking_version":4,"operator":"all_of","result_limit":30,"text_ingredients":[{"query":"sleep","weight":0.4,"negative":false,"locked":true}],"song_anchors":[{"track_id":7,"artist":"A","title":"B","weight":0.6,"negative":false,"locked":true}]}]"""
        val expected = FindMusicQuerySpec(
            textIngredients = listOf(
                FindMusicTextIngredient("sleep", 0.4f, negative = false),
            ),
            songSeeds = listOf(
                FindMusicSongAnchor(
                    trackId = 7,
                    artist = "A",
                    title = "B",
                    weight = 0.6f,
                    negative = false,
                ),
            ),
        )

        val decoded = FindMusicQuerySpecCodec.fromJsonArray(legacy).single()

        assertEquals(expected, decoded)
        assertEquals(expected.stateKey, decoded.stateKey)
        assertFalse(FindMusicQuerySpecCodec.toJson(decoded).contains("locked"))
    }

    @Test
    fun replayResolvesAStableRecordingAfterDatabaseRowReplacement() {
        val saved = FindMusicQuerySpec(
            songSeeds = listOf(anchor(trackId = 7, stableId = stableId(5))),
            libraryBinding = binding("old-generation"),
        )
        val current = catalog(
            binding("new-generation"),
            stableRow(trackId = 70, stableId = stableId(5)),
        )

        val result = FindMusicAnchorResolver.resolveReplay(saved, current)

        val resolved = result as FindMusicAnchorBindingResult.Success
        assertEquals(70L, resolved.querySpec.songSeeds.single().trackId)
        assertEquals(current.binding, resolved.querySpec.libraryBinding)
    }

    @Test
    fun replayUsesCanonicalLocatorAndExcludesEveryEquivalentRow() {
        val id = stableId(8)
        val saved = FindMusicQuerySpec(
            songSeeds = listOf(anchor(trackId = 7, stableId = id)),
            libraryBinding = binding("old-generation"),
        )
        val current = catalog(
            binding("new-generation"),
            stableRow(trackId = 80, stableId = id),
            stableRow(trackId = 81, stableId = id),
        )

        val result = FindMusicAnchorResolver.resolveReplay(saved, current)

        val success = result as FindMusicAnchorBindingResult.Success
        assertEquals(80L, success.querySpec.songSeeds.single().trackId)
        assertEquals(setOf(80L, 81L), success.equivalentTrackIdsToExclude)
    }

    @Test
    fun duplicateStableSongIngredientsUseListenerLanguageAndRetainDiagnostics() {
        val id = stableId(8)
        val saved = FindMusicQuerySpec(
            songSeeds = listOf(
                anchor(trackId = 80, stableId = id).copy(weight = 0.5f),
                anchor(trackId = 81, stableId = id).copy(weight = 0.5f),
            ),
            libraryBinding = binding("current"),
        )
        val current = catalog(
            binding("current"),
            stableRow(trackId = 80, stableId = id),
            stableRow(trackId = 81, stableId = id),
        )

        val result = FindMusicAnchorResolver.bindCurrent(saved, current)

        val failure = result as FindMusicAnchorBindingResult.Failure
        assertTrue(failure.message.contains("same indexed audio"))
        assertFalse(failure.message.contains("source span"))
        assertTrue(failure.diagnosticDetail.orEmpty().contains("stable track-span"))
    }

    @Test
    fun legacyReplayFailsClosedWithoutItsOriginalGenerationBinding() {
        val saved = FindMusicQuerySpec(
            schemaVersion = 1,
            songSeeds = listOf(anchor(trackId = 7, stableId = null)),
            libraryBinding = null,
        )
        val current = catalog(binding("current"), legacyRow(7))

        val result = FindMusicAnchorResolver.resolveReplay(saved, current)

        assertTrue(result is FindMusicAnchorBindingResult.Failure)
        val failure = result as FindMusicAnchorBindingResult.Failure
        assertTrue(failure.message.contains("saved search no longer matches"))
        assertFalse(failure.message.contains("generation"))
        assertTrue(failure.diagnosticDetail.orEmpty().contains("predates"))
    }

    @Test
    fun zeroWeightIngredientIsPersistedButNotResolvedLabeledOrExcluded() {
        val inactive = anchor(trackId = 999, stableId = null).copy(weight = 0f)
        val saved = FindMusicQuerySpec(
            textIngredients = listOf(
                FindMusicTextIngredient("active text", 1f, negative = false),
            ),
            songSeeds = listOf(inactive),
            libraryBinding = null,
        )
        val current = catalog(binding("current"), legacyRow(7))

        val result = FindMusicAnchorResolver.resolveReplay(saved, current)

        val success = result as FindMusicAnchorBindingResult.Success
        assertEquals(listOf(inactive), success.querySpec.songSeeds)
        assertTrue(success.equivalentTrackIdsToExclude.isEmpty())
        assertEquals("active text", success.querySpec.displayLabel)
    }

    @Test
    fun displayAndEvidenceAnchorOrderKeepsTextsBeforeRecordings() {
        val spec = FindMusicQuerySpec(
            textIngredients = listOf(
                FindMusicTextIngredient("warm piano", 0.4f, negative = false),
                FindMusicTextIngredient("busy drums", 0.2f, negative = true),
            ),
            songSeeds = listOf(anchor(trackId = 7, stableId = stableId(7)).copy(weight = 0.4f)),
        )

        assertEquals("warm piano \u00b7 less like busy drums \u00b7 Song", spec.displayLabel)
        assertEquals(listOf("warm piano", "busy drums"), spec.activeTextIngredients.map { it.query })
        assertEquals(
            listOf("warm piano", "Less like: busy drums", "Artist - Song"),
            spec.activeEvidenceLabels,
        )
        assertEquals(3, spec.activeIngredientCount)
    }

    @Test
    fun recordingLabelDoesNotRepeatAnArtistAlreadyPresentInTheTitle() {
        assertEquals(
            "Bonobo - Drift (official video)",
            FindMusicSongAnchor(
                trackId = 80_437,
                artist = "bonobo",
                title = "Bonobo - Drift (official video)",
                weight = 1f,
                negative = false,
            ).displayLabel,
        )
        assertEquals(
            "Bonobo - Cirrus",
            FindMusicSongAnchor(
                trackId = 1,
                artist = "Bonobo",
                title = "Cirrus",
                weight = 1f,
                negative = false,
            ).displayLabel,
        )
    }

    @Test
    fun refineRequiresALikePrimaryButDoesNotParseNaturalLanguage() {
        val explicitAvoidPrimary = FindMusicQuerySpec(
            operator = FindMusicOperator.REFINE,
            textIngredients = listOf(
                FindMusicTextIngredient("harsh", 0.5f, negative = true),
                FindMusicTextIngredient("bright", 0.5f, negative = false),
            ),
            refineSpec = FindMusicRefineSpec(primaryIngredientIndex = 0),
        )
        val ordinaryWords = explicitAvoidPrimary.copy(
            textIngredients = listOf(
                FindMusicTextIngredient("not too bright", 0.5f, negative = false),
                FindMusicTextIngredient("without harsh vocals", 0.5f, negative = false),
            ),
        )

        assertTrue(
            validateFindMusicQueryContract(explicitAvoidPrimary)
                ?.contains("primary ingredient must be Like") == true,
        )
        assertEquals(null, validateFindMusicQueryContract(ordinaryWords))
    }

    @Test
    fun refineContractRejectsMissingInvalidAndMisplacedSpecifications() {
        val valid = FindMusicQuerySpec(
            operator = FindMusicOperator.REFINE,
            textIngredients = listOf(
                FindMusicTextIngredient("ambient", 0.5f, negative = false),
                FindMusicTextIngredient("guitar", 0.5f, negative = false),
            ),
            refineSpec = FindMusicRefineSpec(
                primaryIngredientIndex = 0,
                neighborhood = FindMusicRefineNeighborhood.TOP_2_PERCENT,
            ),
        )

        assertEquals(null, validateFindMusicQueryContract(valid))
        assertTrue(
            validateFindMusicQueryContract(valid.copy(refineSpec = null))
                ?.contains("primary ingredient and neighborhood") == true,
        )
        assertTrue(
            validateFindMusicQueryContract(
                valid.copy(refineSpec = FindMusicRefineSpec(primaryIngredientIndex = 2)),
            )?.contains("outside the active request") == true,
        )
        assertTrue(
            validateFindMusicQueryContract(
                valid.copy(
                    textIngredients = valid.textIngredients +
                        FindMusicTextIngredient("piano", 0.25f, negative = false),
                ),
            )?.contains("exactly two active ingredients") == true,
        )
        assertTrue(
            validateFindMusicQueryContract(valid.copy(operator = FindMusicOperator.ALL_OF))
                ?.contains("cannot carry a Refine neighborhood") == true,
        )
    }

    @Test
    fun refineNeighborhoodUsesAnExactCeilingWithoutWideningToTheResultCount() {
        assertEquals(0, FindMusicRefineNeighborhood.TOP_0_25_PERCENT.candidateCount(0))
        assertEquals(1, FindMusicRefineNeighborhood.TOP_0_25_PERCENT.candidateCount(1))
        assertEquals(3, FindMusicRefineNeighborhood.TOP_0_5_PERCENT.candidateCount(500))
        assertEquals(856, FindMusicRefineNeighborhood.TOP_1_PERCENT.candidateCount(85_567))
        assertEquals(1_712, FindMusicRefineNeighborhood.TOP_2_PERCENT.candidateCount(85_567))
    }

    @Test
    fun persistedRefineFailsClosedWhenItsSpecificationIsIncompleteOrUnknown() {
        assertThrows(IllegalArgumentException::class.java) {
            FindMusicQuerySpecCodec.fromJsonArray(
                """[{"operator":"refine","refine":{"primary_ingredient_index":0}}]""",
            )
        }
        assertThrows(IllegalArgumentException::class.java) {
            FindMusicQuerySpecCodec.fromJsonArray(
                """[{"operator":"refine","refine":{"neighborhood":"top_1_percent"}}]""",
            )
        }
        assertThrows(IllegalArgumentException::class.java) {
            FindMusicQuerySpecCodec.fromJsonArray(
                """[{"operator":"refine","refine":{"primary_ingredient_index":0,"neighborhood":"future"}}]""",
            )
        }
    }

    @Test
    fun avoidNeedsAPositiveMusicalAnchor() {
        val avoidOnly = FindMusicQuerySpec(
            textIngredients = listOf(
                FindMusicTextIngredient("harsh vocals", 1f, negative = true),
            ),
        )
        val anchored = avoidOnly.copy(
            textIngredients = listOf(
                FindMusicTextIngredient("slow ambient", 0.7f, negative = false),
                FindMusicTextIngredient("harsh vocals", 0.3f, negative = true),
            ),
        )

        assertTrue(validateFindMusicQueryContract(avoidOnly)?.contains("Like") == true)
        assertEquals(null, validateFindMusicQueryContract(anchored))
    }

    @Test
    fun defaultFindMusicQueueContainsThirtyResults() {
        assertEquals(30, FindMusicQuerySpec.DEFAULT_RESULT_LIMIT)
        assertEquals(30, FindMusicQuerySpec().resultLimit)
        assertEquals(
            FindMusicTextResultPlanner.CLOSEST,
            FindMusicQuerySpec().textResultPlanner,
        )
    }

    @Test
    fun variedTextPlannerRoundTripsWithItsVersionAndChangesRequestIdentity() {
        val closest = FindMusicQuerySpec(
            textIngredients = listOf(
                FindMusicTextIngredient("spacey jazz", 1f, negative = false),
            ),
        )
        val varied = closest.copy(textResultPlanner = FindMusicTextResultPlanner.VARIED_DPP)

        val json = FindMusicQuerySpecCodec.toJson(varied)
        val decoded = FindMusicQuerySpecCodec.fromJsonArray("[$json]").single()

        assertEquals(varied, decoded)
        assertTrue(json.contains("\"text_result_planner\":\"varied_dpp\""))
        assertTrue(json.contains("\"text_result_planner_version\":2"))
        assertFalse(closest.stateKey == varied.stateKey)
        assertEquals(null, validateFindMusicQueryContract(varied))
    }

    @Test
    fun variedAllOfPlannerRoundTripsAndIsRejectedOutsideMultiIngredientAllOf() {
        val allOf = FindMusicQuerySpec(
            operator = FindMusicOperator.ALL_OF,
            textIngredients = listOf(
                FindMusicTextIngredient("ambient", 0.5f, negative = false),
                FindMusicTextIngredient("techno", 0.5f, negative = false),
            ),
            textResultPlanner = FindMusicTextResultPlanner.VARIED_ALL_OF_DPP,
        )

        val json = FindMusicQuerySpecCodec.toJson(allOf)
        assertEquals(allOf, FindMusicQuerySpecCodec.fromJsonArray("[$json]").single())
        assertTrue(json.contains("\"text_result_planner\":\"varied_all_of_dpp\""))
        assertEquals(null, validateFindMusicQueryContract(allOf))
        assertTrue(
            validateFindMusicQueryContract(
                allOf.copy(
                    textIngredients = listOf(
                        FindMusicTextIngredient("ambient", 1f, negative = false),
                    ),
                ),
            )?.contains("two") == true,
        )
        assertTrue(
            validateFindMusicQueryContract(
                allOf.copy(
                    operator = FindMusicOperator.REFINE,
                    refineSpec = FindMusicRefineSpec(
                        primaryIngredientIndex = 0,
                    ),
                ),
            )?.contains("All-of Varied") == true,
        )
    }

    @Test
    fun persistedTextPlannerFailsClosedWhenItsIdentityIsIncompleteOrUnknown() {
        assertThrows(IllegalArgumentException::class.java) {
            FindMusicQuerySpecCodec.fromJsonArray(
                """[{"text_result_planner":"varied_dpp"}]""",
            )
        }
        assertThrows(IllegalArgumentException::class.java) {
            FindMusicQuerySpecCodec.fromJsonArray(
                """[{"text_result_planner":"varied_dpp","text_result_planner_version":1}]""",
            )
        }
        assertThrows(IllegalArgumentException::class.java) {
            FindMusicQuerySpecCodec.fromJsonArray(
                """[{"text_result_planner":"future","text_result_planner_version":1}]""",
            )
        }
    }

    @Test
    fun `incompatible saved search contracts share one actionable listener message`() {
        val current = FindMusicQuerySpec(
            textIngredients = listOf(
                FindMusicTextIngredient("slow ambient", 1f, negative = false),
            ),
        )
        val incompatible = listOf(
            current.copy(schemaVersion = 1),
            current.copy(schemaVersion = FindMusicQuerySpec.CURRENT_SCHEMA_VERSION + 1),
            current.copy(embeddingSpace = "another-space"),
            current.copy(rankingVersion = FindMusicQuerySpec.LEGACY_RANKING_VERSION),
            current.copy(rankingVersion = FindMusicQuerySpec.CURRENT_RANKING_VERSION + 1),
        )

        incompatible.forEach { query ->
            assertEquals(
                INCOMPATIBLE_SAVED_FIND_MUSIC_QUERY_MESSAGE,
                validateFindMusicQueryContract(query),
            )
        }
        assertFalse(INCOMPATIBLE_SAVED_FIND_MUSIC_QUERY_MESSAGE.contains("schema"))
        assertFalse(INCOMPATIBLE_SAVED_FIND_MUSIC_QUERY_MESSAGE.contains("embedding"))
    }

    @Test
    fun rawCosineRouteIsReservedForOnePositiveTextIngredient() {
        val simple = FindMusicQuerySpec(
            textIngredients = listOf(
                FindMusicTextIngredient("late-night jazz", 1f, negative = false),
            ),
        )

        assertTrue(simple.isSimplePositiveTextOnly)
        assertFalse(
            simple.copy(
                textIngredients = simple.textIngredients +
                    FindMusicTextIngredient("soft drums", 0.5f, negative = false),
            ).isSimplePositiveTextOnly,
        )
        assertFalse(
            simple.copy(
                textIngredients = listOf(simple.textIngredients.single().copy(negative = true)),
            ).isSimplePositiveTextOnly,
        )
        assertFalse(
            simple.copy(songSeeds = listOf(anchor(7, stableId(7)))).isSimplePositiveTextOnly,
        )
    }

    @Test
    fun latestRequestGateRejectsSupersededAndCancelledRevisions() {
        val gate = LatestFindMusicRequestGate()
        val first = gate.begin()
        assertTrue(gate.isCurrent(first))

        val second = gate.begin()
        assertFalse(gate.isCurrent(first))
        assertTrue(gate.isCurrent(second))

        gate.cancel()
        assertFalse(gate.isCurrent(second))
    }

    @Test
    fun unknownPersistedOperatorIsNotSilentlyChangedToAllOf() {
        assertThrows(IllegalArgumentException::class.java) {
            FindMusicQuerySpecCodec.fromJsonArray(
                """[{"operator":"future_operator","text":{"query":"sleep"}}]""",
            )
        }
    }

    private fun anchor(trackId: Long, stableId: String?) = FindMusicSongAnchor(
        trackId = trackId,
        stableTrackSpanId = stableId,
        artist = "Artist",
        title = "Song",
        weight = 1f,
        negative = false,
    )

    private fun stableRow(trackId: Long, stableId: String) = StableTrackIdentityRow(
        trackId = trackId,
        stableTrackSpanId = stableId,
        stableIdentitySpecId = STABLE_SPEC,
        stableIdentityStrength = StableTrackSpanIdentityStrength.FULL_CONTENT_SHA256,
        embeddingSpecId = "embedding-spec-test",
        embeddingSha256 = stableId.removePrefix("stable-track-span-v1-"),
    )

    private fun legacyRow(trackId: Long) = StableTrackIdentityRow(
        trackId = trackId,
        stableTrackSpanId = null,
        stableIdentitySpecId = null,
        stableIdentityStrength = null,
        embeddingSpecId = null,
        embeddingSha256 = null,
    )

    private fun catalog(
        binding: StableIdentityGenerationBinding,
        vararg rows: StableTrackIdentityRow,
    ) = StableTrackIdentityCatalog.fromOrderedRows(
        binding = binding,
        orderedEmbeddingTrackIds = rows.map { it.trackId }.toLongArray(),
        rows = rows.toList(),
    )

    private fun binding(id: String) = StableIdentityGenerationBinding(
        bindingSpecId = "test-binding-v1",
        generationId = id,
        activationBindingId = id,
        databaseContentSha256 = id.padEnd(64, '0'),
        orderedTrackSetSha256 = id.padEnd(64, '1'),
    )

    private fun stableId(number: Int): String =
        "stable-track-span-v1-${number.toString(16).padStart(64, '0')}"

    companion object {
        private const val STABLE_SPEC =
            "stable-track-span-v1:content-sha256:native-half-open-sample-span"
    }
}
