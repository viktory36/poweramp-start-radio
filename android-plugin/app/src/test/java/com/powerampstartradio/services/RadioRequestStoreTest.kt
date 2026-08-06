package com.powerampstartradio.services

import com.powerampstartradio.data.EmbeddedTrack
import com.powerampstartradio.data.StableIdentityGenerationBinding
import com.powerampstartradio.poweramp.PowerampTrack
import com.powerampstartradio.poweramp.TrackMatcher
import com.powerampstartradio.similarity.DppSelectionEvidence
import com.powerampstartradio.similarity.FindMusicAllOfQueuePlanner
import com.powerampstartradio.similarity.FindMusicAllOfQueuePlanEvidence
import com.powerampstartradio.similarity.FindMusicTextQueuePlanEvidence
import com.powerampstartradio.ui.ComposedRadioContract
import com.powerampstartradio.ui.ComposedTrackEvidence
import com.powerampstartradio.ui.DirectQueuePlacement
import com.powerampstartradio.ui.FindMusicOperator
import com.powerampstartradio.ui.FindMusicQuerySpec
import com.powerampstartradio.ui.FindMusicRefineNeighborhood
import com.powerampstartradio.ui.FindMusicRefineSpec
import com.powerampstartradio.ui.FindMusicSessionEvidence
import com.powerampstartradio.ui.FindMusicSongAnchor
import com.powerampstartradio.ui.FindMusicTextIngredient
import com.powerampstartradio.ui.FindMusicTextResultPlanner
import com.powerampstartradio.ui.FindMusicTrackEvidence
import com.powerampstartradio.ui.LibraryAddedRange
import com.powerampstartradio.ui.MAX_LIBRARY_ADDED_DAYS
import com.powerampstartradio.ui.QueueDeliverySummary
import com.powerampstartradio.ui.QueueOrigin
import com.powerampstartradio.ui.QueueStatus
import com.powerampstartradio.ui.QueuedTrackResult
import com.powerampstartradio.ui.RadioConfig
import com.powerampstartradio.ui.RadioGenerationToken
import com.powerampstartradio.ui.RadioResult
import com.powerampstartradio.ui.RadioSeedIdentity
import com.powerampstartradio.ui.RadioSessionOutcome
import com.powerampstartradio.ui.SeedSpec
import com.powerampstartradio.ui.SeedType
import com.powerampstartradio.ui.SelectionMode
import com.powerampstartradio.ui.StableResultReductionEvidence
import java.io.File
import java.io.FileOutputStream
import java.nio.file.AtomicMoveNotSupportedException
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.util.UUID
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class RadioRequestStoreTest {
    @Test
    fun `schema v5 provider binding query config identities and contract survive round trip`() =
        withTempDir { root ->
            val config = RadioConfig(
                numTracks = 137,
                libraryAddedDays = 17,
                candidatePoolSize = 2_345,
                mmrCandidatePoolFraction = 0.17f,
                dppFixedCandidatePoolFraction = 0.23f,
                dppUsesCertifiedFullDomain = false,
                selectionMode = SelectionMode.MMR,
                driftEnabled = true,
                anchorStrength = 0.63f,
                anchorHalfLifeTracks = 11.5f,
                walkRestartAlpha = 0.27f,
                momentumBeta = 0.81f,
                diversityLambda = 0.44f,
                dppQualityExponent = 1.75f,
                shuffleSeed = 0x1234_5678_9abc_def0L,
                artistLimitsEnabled = false,
                maxPerArtist = 12,
                minArtistSpacing = 7,
            )
            val querySpec = composedQuerySpec(libraryAddedDays = 17)
            val request = DurableRadioRequest.multiSeed(
                generation = generation(),
                providerGenerationId = providerGeneration(),
                seeds = listOf(
                    SeedSpec(FloatArray(768) { it / 768f }, 0.4f, "liquid", SeedType.TEXT),
                    SeedSpec(FloatArray(768) { (it + 1) / 768f }, 0.35f, "night drive", SeedType.TEXT),
                    SeedSpec(FloatArray(768) { -it / 768f }, -0.25f, "avoid", SeedType.SONG, 42L),
                ),
                seedIdentities = listOf(null, null, identity(42)),
                querySpec = querySpec,
                config = config,
                composedContract = ComposedRadioContract(),
                showToasts = true,
                origin = QueueOrigin.COMPOSED_RADIO,
                requestId = uuid(1),
                createdAtEpochMs = 1_000L,
            )
            val store = store(root, owner = "process-a")

            store.persist(request)
            val claimed = store.claim(request.requestId) as RadioRequestClaim.Claimed

            assertEquals(config, claimed.request.multiSeed?.config)
            assertEquals(generation(), claimed.request.generation)
            assertEquals(providerGeneration(), claimed.request.providerGenerationId)
            assertEquals(identity(42), claimed.request.multiSeed?.seedIdentities?.get(2))
            assertEquals(ComposedRadioContract(), claimed.request.multiSeed?.composedContract)
            assertEquals(querySpec, claimed.request.multiSeed?.querySpec)
        }

    @Test
    fun `request rejects a malformed Poweramp provider generation`() = withTempDir { root ->
        val store = store(root, owner = "process-a")
        val request = radioRequest(uuid(2)).copy(providerGenerationId = "mutable-provider")

        assertThrows(IllegalArgumentException::class.java) { store.persist(request) }
        assertThrows(IllegalArgumentException::class.java) { store.decode(com.google.gson.Gson().toJson(request).toByteArray()) }
    }

    @Test
    fun `composed radio rejects Refine and seed drift from displayed query`() = withTempDir { root ->
        val refine = composedRequest(uuid(10)).let { request ->
            val payload = requireNotNull(request.multiSeed)
            request.copy(
                multiSeed = payload.copy(
                    seeds = payload.seeds.take(2).map { it.copy(weight = 0.5f) },
                    seedIdentities = payload.seedIdentities.take(2),
                    querySpec = payload.querySpec.copy(
                        operator = FindMusicOperator.REFINE,
                        textIngredients = payload.querySpec.textIngredients.map {
                            it.copy(weight = 0.5f)
                        },
                        songSeeds = emptyList(),
                        refineSpec = FindMusicRefineSpec(
                            primaryIngredientIndex = 0,
                            neighborhood = FindMusicRefineNeighborhood.DEFAULT,
                        ),
                    ),
                ),
            )
        }
        assertThrows(IllegalArgumentException::class.java) {
            store(root, owner = "process-a").persist(refine)
        }

        val changedLabel = composedRequest(uuid(11)).let { request ->
            request.copy(
                multiSeed = request.multiSeed!!.copy(
                    seeds = request.multiSeed.seeds.mapIndexed { index, seed ->
                        if (index == 0) seed.copy(label = "different text") else seed
                    },
                ),
            )
        }
        assertThrows(IllegalArgumentException::class.java) {
            store(root, owner = "process-a").persist(changedLabel)
        }

        val changedAddedDays = composedRequest(uuid(12)).let { request ->
            request.copy(
                multiSeed = request.multiSeed!!.copy(
                    querySpec = request.multiSeed.querySpec.copy(libraryAddedDays = 17),
                ),
            )
        }
        assertThrows(IllegalArgumentException::class.java) {
            store(root, owner = "process-a").persist(changedAddedDays)
        }
    }

    @Test
    fun `legacy preset and equivalent exact days share one durable contract`() =
        withTempDir { root ->
            val request = composedRequest(uuid(13)).let { source ->
                source.copy(
                    multiSeed = source.multiSeed!!.copy(
                        config = source.multiSeed.config.copy(
                            libraryAddedRange = LibraryAddedRange.LAST_30_DAYS,
                        ),
                        querySpec = source.multiSeed.querySpec.copy(libraryAddedDays = 30),
                    ),
                )
            }

            store(root, owner = "process-a").persist(request)
        }

    @Test
    fun `composed receipt requires exact query and objective evidence`() = withTempDir { root ->
        val request = composedRequest(uuid(12))
        val store = store(root, owner = "process-a")
        store.persist(request)
        assertTrue(store.claim(request.requestId) is RadioRequestClaim.Claimed)

        val queued = QueuedTrackResult(
            track = track(13),
            similarity = 0.83f,
            similarityToSeed = 0.83f,
            status = QueueStatus.QUEUED,
            resolvedPowerampFileId = 130L,
            resolvedPowerampQueueId = 1_130L,
            stableTrackSpanId = stableId(13),
            composedEvidence = ComposedTrackEvidence(
                objectiveRank = 2,
                objectiveScore = 0.83f,
                ingredientPercentiles = listOf(0.9f, 0.84f, 0.77f),
            ),
        )
        val result = RadioResult(
            seedTrack = PowerampTrack(-1, "All of: liquid + night drive - avoid", null, null, 0, ""),
            matchType = TrackMatcher.MatchType.COMPOSED_QUERY,
            tracks = listOf(queued),
            config = request.multiSeed!!.config,
            delivery = QueueDeliverySummary.fromTracks(
                origin = QueueOrigin.COMPOSED_RADIO,
                requestedCount = 1,
                rankedCount = 1,
                resolvedCount = 1,
                tracks = listOf(queued),
                verificationComplete = true,
            ),
            requestId = request.requestId,
            outcome = RadioSessionOutcome.SUCCEEDED,
            generation = request.generation,
            providerGenerationId = request.providerGenerationId,
            composedContract = request.multiSeed.composedContract,
            composedQuerySpec = request.multiSeed.querySpec,
            stableResultReduction = StableResultReductionEvidence(
                identityPolicyVersion =
                    com.powerampstartradio.similarity.StableVisibleResultReducer
                        .IDENTITY_POLICY_VERSION,
                requestedVisibleCount = 1,
                scannedRowCount = 2,
                collapsedEquivalentCount = 1,
            ),
        )

        store.persistResultReceipt(request.requestId, result)
        assertThrows(IllegalArgumentException::class.java) {
            store.persistResultReceipt(
                request.requestId,
                result.copy(
                    composedQuerySpec = result.composedQuerySpec!!.copy(
                        resultLimit = result.composedQuerySpec.resultLimit + 1,
                    ),
                ),
            )
        }
        assertThrows(IllegalArgumentException::class.java) {
            store.persistResultReceipt(
                request.requestId,
                result.copy(tracks = listOf(queued.copy(composedEvidence = null))),
            )
        }
    }

    @Test
    fun `completion requires durable fully verified receipt and remains idempotent`() =
        withTempDir { root ->
            val request = radioRequest(uuid(2))
            val store = store(root, owner = "process-a")

            store.persist(request)
            assertTrue(store.claim(request.requestId) is RadioRequestClaim.Claimed)
            assertThrows(IllegalArgumentException::class.java) {
                store.markCompleted(request.requestId)
            }
            val result = successfulResult(request)
            val firstReceipt = store.persistResultReceipt(request.requestId, result)
            val secondReceipt = store.persistResultReceipt(request.requestId, result)
            assertEquals(firstReceipt, secondReceipt)

            store.markCompleted(request.requestId)
            assertEquals(
                RadioRequestClaim.AlreadyTerminal(RadioRequestStateKind.COMPLETED),
                store.claim(request.requestId),
            )
            assertFalse(payloadFile(root, request.requestId).exists())
        }

    @Test
    fun `seed rank domain survives receipts and validates only evidence that is present`() =
        withTempDir { root ->
            val request = radioRequest(uuid(200))
            val store = store(root, owner = "process-a")
            store.persist(request)
            assertTrue(store.claim(request.requestId) is RadioRequestClaim.Claimed)

            val ranked = successfulResult(request).let { result ->
                result.copy(
                    tracks = result.tracks.map { it.copy(seedRank = 90, driftRank = 75) },
                    eligibleCandidateIdentityCount = 5,
                    seedRankingIdentityCount = 100,
                )
            }
            val receipt = store.persistResultReceipt(request.requestId, ranked)
            assertEquals(5, receipt.result.eligibleCandidateIdentityCount)
            assertEquals(100, receipt.result.seedRankingIdentityCount)

            assertThrows(IllegalArgumentException::class.java) {
                store.persistResultReceipt(
                    request.requestId,
                    ranked.copy(eligibleCandidateIdentityCount = 101),
                )
            }
            assertThrows(IllegalArgumentException::class.java) {
                store.persistResultReceipt(
                    request.requestId,
                    ranked.copy(
                        tracks = ranked.tracks.map { it.copy(seedRank = 101) },
                    ),
                )
            }

            val legacyRequest = radioRequest(uuid(201))
            store.persist(legacyRequest)
            assertTrue(store.claim(legacyRequest.requestId) is RadioRequestClaim.Claimed)
            val legacy = successfulResult(legacyRequest).let { result ->
                result.copy(
                    tracks = result.tracks.map { it.copy(seedRank = 1) },
                    eligibleCandidateIdentityCount = 1,
                    seedRankingIdentityCount = null,
                )
            }
            assertEquals(
                legacy,
                store.persistResultReceipt(legacyRequest.requestId, legacy).result,
            )
        }

    @Test
    fun `foreign owner recovers receipt without replaying queue mutation`() = withTempDir { root ->
        val request = radioRequest(uuid(3))
        val processA = store(root, owner = "process-a")
        processA.persist(request)
        assertTrue(processA.claim(request.requestId) is RadioRequestClaim.Claimed)
        val expected = partialResult(request, "Poweramp readback failed")
        processA.persistResultReceipt(request.requestId, expected, expected.failureDetail)

        val processB = store(root, owner = "process-b")
        val recovery = processB.claim(request.requestId) as RadioRequestClaim.ResultReady
        assertEquals(expected, recovery.receipt.result)
        processB.finalizeRecoveredResult(recovery.receipt)

        assertEquals(
            RadioRequestClaim.AlreadyTerminal(RadioRequestStateKind.FAILED),
            store(root, owner = "process-c").claim(request.requestId),
        )
        assertTrue(payloadFile(root, request.requestId).exists())
    }

    @Test
    fun `claimed request without receipt fails closed after owner change`() = withTempDir { root ->
        val request = radioRequest(uuid(4))
        val processA = store(root, owner = "process-a")
        processA.persist(request)
        assertTrue(processA.claim(request.requestId) is RadioRequestClaim.Claimed)

        val processB = store(root, owner = "process-b")
        assertEquals(
            RadioRequestClaim.AlreadyTerminal(RadioRequestStateKind.INTERRUPTED_NEEDS_RETRY),
            processB.claim(request.requestId),
        )
        assertTrue(payloadFile(root, request.requestId).exists())
        assertTrue(processB.recoverableRequestIds().isEmpty())
    }

    @Test
    fun `payload published before torn state write remains recoverable`() = withTempDir { root ->
        val request = radioRequest(uuid(5))
        var writes = 0
        val tornWriter = RadioRequestAtomicWriter { file, bytes ->
            writes++
            if (writes == 2) throw IllegalStateException("simulated state-write crash")
            atomicJvmWrite(file, bytes)
        }

        assertEquals(
            request.requestId,
            RadioRequestStore(
                rootDir = root,
                ownerToken = "process-a",
                clock = { 2_000L },
                atomicWriter = tornWriter,
                atomicReader = JVM_READER,
                atomicDeleter = JVM_DELETER,
            ).persist(request),
        )

        val recovered = store(root, owner = "process-b")
        assertEquals(listOf(request.requestId), recovered.recoverableRequestIds())
        assertTrue(recovered.claim(request.requestId) is RadioRequestClaim.Claimed)
    }

    @Test
    fun `corrupt payload and unreadable state fail terminally`() = withTempDir { root ->
        val corruptPayload = radioRequest(uuid(6))
        val store = store(root, owner = "process-a")
        store.persist(corruptPayload)
        payloadFile(root, corruptPayload.requestId).appendText("corrupt")
        assertThrows(IllegalArgumentException::class.java) { store.claim(corruptPayload.requestId) }
        store.markFailed(corruptPayload.requestId, "payload digest mismatch")

        val corruptState = radioRequest(uuid(7))
        store.persist(corruptState)
        stateFile(root, corruptState.requestId).writeText("not-json")
        val restarted = store(root, owner = "process-b")
        assertEquals(
            RadioRequestClaim.AlreadyTerminal(RadioRequestStateKind.INTERRUPTED_NEEDS_RETRY),
            restarted.claim(corruptState.requestId),
        )
    }

    @Test
    fun `invalid selector values are rejected on persist and decode`() = withTempDir { root ->
        val store = store(root, owner = "process-a")
        listOf(
            RadioConfig(configSchemaVersion = 2),
            RadioConfig(numTracks = 0),
            RadioConfig(numTracks = 1_001),
            RadioConfig(libraryAddedDays = 0),
            RadioConfig(libraryAddedDays = MAX_LIBRARY_ADDED_DAYS + 1),
            RadioConfig(numTracks = 50, candidatePoolSize = 49),
            RadioConfig(mmrCandidatePoolFraction = Float.NaN),
            RadioConfig(dppFixedCandidatePoolFraction = Float.NaN),
            RadioConfig(diversityLambda = 1.1f),
            RadioConfig(anchorHalfLifeTracks = Float.POSITIVE_INFINITY),
            RadioConfig(selectionMode = SelectionMode.DPP, driftEnabled = true),
            RadioConfig(selectionMode = SelectionMode.RANDOM_WALK, walkRestartAlpha = 0f),
            RadioConfig(selectionMode = SelectionMode.RANDOM_WALK, walkRestartAlpha = 1f),
        ).forEachIndexed { index, config ->
            val request = radioRequest(uuid(20 + index), config)
            assertThrows(IllegalArgumentException::class.java) { store.persist(request) }
            val uncheckedJson = com.google.gson.GsonBuilder()
                .serializeSpecialFloatingPointValues()
                .create()
                .toJson(request)
                .toByteArray()
            assertThrows(IllegalArgumentException::class.java) { store.decode(uncheckedJson) }
        }
    }

    @Test
    fun `failed tombstones do not exhaust outstanding request quota`() = withTempDir { root ->
        val store = store(root, owner = "process-a")
        repeat(40) { index ->
            val request = radioRequest(uuid(100 + index))
            store.persist(request)
            assertTrue(store.claim(request.requestId) is RadioRequestClaim.Claimed)
            store.markFailed(request.requestId, "test failure")
        }

        val finalRequest = radioRequest(uuid(999))
        store.persist(finalRequest)
        assertTrue(finalRequest.requestId in store.recoverableRequestIds())
    }

    @Test
    fun `direct queue preserves occurrence order exact Poweramp IDs and placement`() =
        withTempDir { root ->
            val tracks = listOf(track(8), track(3), track(8))
            val identities = listOf(identity(8), identity(3), identity(8))
            val request = DurableRadioRequest.directQueue(
                generation = generation(),
                providerGenerationId = providerGeneration(),
                tracks = tracks,
                trackIdentities = identities,
                resolvedPowerampFileIds = listOf(80L, 30L, 81L),
                label = "Replay: night drive",
                origin = QueueOrigin.HISTORY_REQUEUE,
                placement = DirectQueuePlacement.APPEND,
                requestId = uuid(8),
                createdAtEpochMs = 1_000L,
            )
            val store = store(root, owner = "process-a")

            store.persist(request)
            val payload = (store.claim(request.requestId) as RadioRequestClaim.Claimed)
                .request.directQueue!!

            assertEquals(listOf(8L, 3L, 8L), payload.tracks.map { it.id })
            assertEquals(listOf(80L, 30L, 81L), payload.resolvedPowerampFileIds)
            assertEquals(DirectQueuePlacement.APPEND, payload.placement)
        }

    @Test
    fun `displayed Find Music evidence survives the direct request and terminal session`() =
        withTempDir { root ->
            val query = FindMusicQuerySpec(
                textIngredients = listOf(
                    FindMusicTextIngredient("sleep", 1f, negative = false),
                ),
                resultLimit = 30,
                libraryBinding = StableIdentityGenerationBinding(
                    bindingSpecId = "v2-active-index-generation-binding-v1",
                    generationId = generation().generationId,
                    activationBindingId = generation().activationBindingId,
                    databaseContentSha256 = generation().databaseContentSha256,
                    orderedTrackSetSha256 = generation().orderedTrackSetSha256,
                ),
            )
            val sessionEvidence = FindMusicSessionEvidence(
                querySpec = query,
                orderedActiveTrackIdsSha256 = "8".repeat(64),
                activeTrackCount = 100,
                stableResultReduction = StableResultReductionEvidence(
                    identityPolicyVersion =
                        com.powerampstartradio.similarity.StableVisibleResultReducer
                            .IDENTITY_POLICY_VERSION,
                    requestedVisibleCount = 30,
                    scannedRowCount = 31,
                    collapsedEquivalentCount = 1,
                ),
            )
            val rankingEvidence = FindMusicTrackEvidence(
                displayedRank = 1,
                objectiveRank = 4,
                resultScore = 0.42f,
                rankingScore = 0.42f,
            )
            val request = DurableRadioRequest.directQueue(
                generation = generation(),
                providerGenerationId = providerGeneration(),
                tracks = listOf(track(13)),
                trackIdentities = listOf(identity(13)),
                resolvedPowerampFileIds = listOf(130L),
                label = "sleep",
                origin = QueueOrigin.TEXT_RESULT_LIST,
                placement = DirectQueuePlacement.REPLACE_UPCOMING,
                findMusicSessionEvidence = sessionEvidence,
                findMusicTrackEvidence = listOf(rankingEvidence),
                requestId = uuid(80),
                createdAtEpochMs = 1_000L,
            )
            val store = store(root, owner = "process-a")

            store.persist(request)
            val claimed = store.claim(request.requestId) as RadioRequestClaim.Claimed
            assertEquals(sessionEvidence, claimed.request.directQueue?.findMusicSessionEvidence)
            assertEquals(
                listOf(rankingEvidence),
                claimed.request.directQueue?.findMusicTrackEvidence,
            )

            val queued = QueuedTrackResult(
                track = track(13),
                similarity = 0.42f,
                similarityToSeed = 0f,
                status = QueueStatus.QUEUED,
                resolvedPowerampFileId = 130L,
                resolvedPowerampQueueId = 1_130L,
                stableTrackSpanId = stableId(13),
                findMusicEvidence = rankingEvidence,
            )
            val result = RadioResult(
                seedTrack = PowerampTrack(-1, "sleep", null, null, 0, null),
                matchType = TrackMatcher.MatchType.NOT_APPLICABLE,
                tracks = listOf(queued),
                isDirectQueue = true,
                delivery = QueueDeliverySummary.fromTracks(
                    origin = QueueOrigin.TEXT_RESULT_LIST,
                    requestedCount = 1,
                    rankedCount = 1,
                    resolvedCount = 1,
                    tracks = listOf(queued),
                    verificationComplete = true,
                ),
                requestId = request.requestId,
                outcome = RadioSessionOutcome.SUCCEEDED,
                generation = request.generation,
                providerGenerationId = request.providerGenerationId,
                directQueuePlacement = DirectQueuePlacement.REPLACE_UPCOMING,
                findMusicSessionEvidence = sessionEvidence,
            )

            val receipt = store.persistResultReceipt(request.requestId, result)
            assertEquals(
                sessionEvidence,
                receipt.result.findMusicSessionEvidence,
            )
        }

    @Test
    fun `varied text request preserves selection order and original nonmonotonic ranks`() =
        withTempDir { root ->
            val generation = generation()
            val query = FindMusicQuerySpec(
                textIngredients = listOf(
                    FindMusicTextIngredient("spacey jazz", 1f, negative = false),
                ),
                resultLimit = 30,
                textResultPlanner = FindMusicTextResultPlanner.VARIED_DPP,
                libraryBinding = StableIdentityGenerationBinding(
                    bindingSpecId = "v2-active-index-generation-binding-v1",
                    generationId = generation.generationId,
                    activationBindingId = generation.activationBindingId,
                    databaseContentSha256 = generation.databaseContentSha256,
                    orderedTrackSetSha256 = generation.orderedTrackSetSha256,
                ),
            )
            val selectedIds = listOf(13L, 14L, 15L)
            val originalRanks = listOf(1, 4, 2)
            val textPlan = FindMusicTextQueuePlanEvidence(
                planner = FindMusicTextResultPlanner.VARIED_DPP,
                plannerVersion = FindMusicTextResultPlanner.VARIED_DPP.currentVersion,
                completeTextRankingSha256 = "a".repeat(64),
                completeCandidateDomainCount = 100,
                requestedResultCount = 30,
                orderedSelectedTrackIds = selectedIds,
                orderedOriginalTextObjectiveRanks = originalRanks,
                dppSelection = DppSelectionEvidence(
                    completeCandidateDomainCount = 100,
                    initialWorkingCandidateCount = 50,
                    attemptedCandidateCounts = listOf(50),
                    finalWorkingCandidateCount = 50,
                    selectedMarginalGains = listOf(0.5, 0.25, 0.125),
                    finalUnseenInitialGainUpperBound = 0.01,
                    usedCompleteCandidateDomain = false,
                    reproducedFullDomainGreedySequence = true,
                ),
            )
            val sessionEvidence = FindMusicSessionEvidence(
                querySpec = query,
                orderedActiveTrackIdsSha256 = "8".repeat(64),
                activeTrackCount = 102,
                objectiveRankingDomainCount = 100,
                stableResultReduction = StableResultReductionEvidence(
                    identityPolicyVersion =
                        com.powerampstartradio.similarity.StableVisibleResultReducer
                            .IDENTITY_POLICY_VERSION,
                    requestedVisibleCount = 30,
                    scannedRowCount = 3,
                    collapsedEquivalentCount = 0,
                ),
                textQueuePlan = textPlan,
            )
            val rowEvidence = originalRanks.mapIndexed { index, rank ->
                val score = listOf(0.5f, 0.4f, 0.45f)[index]
                FindMusicTrackEvidence(
                    displayedRank = index + 1,
                    objectiveRank = rank,
                    resultScore = score,
                    rankingScore = score,
                )
            }
            val request = DurableRadioRequest.directQueue(
                generation = generation,
                providerGenerationId = providerGeneration(),
                tracks = selectedIds.map(::track),
                trackIdentities = selectedIds.map(::identity),
                resolvedPowerampFileIds = selectedIds.map { it * 10L },
                label = "spacey jazz",
                origin = QueueOrigin.TEXT_RESULT_LIST,
                placement = DirectQueuePlacement.REPLACE_UPCOMING,
                findMusicSessionEvidence = sessionEvidence,
                findMusicTrackEvidence = rowEvidence,
                requestId = uuid(81),
                createdAtEpochMs = 1_000L,
            )
            val store = store(root, owner = "process-a")

            store.persist(request)
            val claimed = store.claim(request.requestId) as RadioRequestClaim.Claimed

            assertEquals(
                originalRanks,
                claimed.request.directQueue?.findMusicTrackEvidence?.map { it.objectiveRank },
            )
            assertEquals(
                textPlan,
                claimed.request.directQueue?.findMusicSessionEvidence?.textQueuePlan,
            )
            assertEquals(
                listOf(0.5, 0.25, 0.125),
                claimed.request.directQueue
                    ?.findMusicSessionEvidence
                    ?.textQueuePlan
                    ?.dppSelection
                    ?.selectedMarginalGains,
            )
        }

    @Test
    fun `varied All-of request preserves selection proof and original objective ranks`() =
        withTempDir { root ->
            val generation = generation()
            val query = FindMusicQuerySpec(
                operator = FindMusicOperator.ALL_OF,
                textIngredients = listOf(
                    FindMusicTextIngredient("ambient", 0.5f, negative = false),
                    FindMusicTextIngredient("sleep", 0.5f, negative = false),
                ),
                resultLimit = 30,
                textResultPlanner = FindMusicTextResultPlanner.VARIED_ALL_OF_DPP,
                libraryBinding = StableIdentityGenerationBinding(
                    bindingSpecId = "v2-active-index-generation-binding-v1",
                    generationId = generation.generationId,
                    activationBindingId = generation.activationBindingId,
                    databaseContentSha256 = generation.databaseContentSha256,
                    orderedTrackSetSha256 = generation.orderedTrackSetSha256,
                ),
            )
            val selectedIds = listOf(13L, 14L, 15L)
            val originalRanks = listOf(1, 7, 3)
            val allOfPlan = FindMusicAllOfQueuePlanEvidence(
                plannerVersion = FindMusicAllOfQueuePlanner.PLANNER_VERSION,
                completeAllOfRankingSha256 = "b".repeat(64),
                completeCandidateDomainCount = 100,
                requestedResultCount = 30,
                orderedSelectedTrackIds = selectedIds,
                orderedOriginalAllOfObjectiveRanks = originalRanks,
                dppSelection = DppSelectionEvidence(
                    completeCandidateDomainCount = 100,
                    initialWorkingCandidateCount = 50,
                    attemptedCandidateCounts = listOf(50),
                    finalWorkingCandidateCount = 50,
                    selectedMarginalGains = listOf(0.5, 0.25, 0.125),
                    finalUnseenInitialGainUpperBound = 0.01,
                    usedCompleteCandidateDomain = false,
                    reproducedFullDomainGreedySequence = true,
                ),
            )
            val sessionEvidence = FindMusicSessionEvidence(
                querySpec = query,
                orderedActiveTrackIdsSha256 = "8".repeat(64),
                activeTrackCount = 102,
                objectiveRankingDomainCount = 100,
                ingredientRankingDomainCount = 101,
                stableResultReduction = StableResultReductionEvidence(
                    identityPolicyVersion =
                        com.powerampstartradio.similarity.StableVisibleResultReducer
                            .IDENTITY_POLICY_VERSION,
                    requestedVisibleCount = 30,
                    scannedRowCount = 3,
                    collapsedEquivalentCount = 0,
                ),
                allOfQueuePlan = allOfPlan,
            )
            val rowEvidence = originalRanks.mapIndexed { index, rank ->
                val score = listOf(0.9f, 0.8f, 0.85f)[index]
                FindMusicTrackEvidence(
                    displayedRank = index + 1,
                    objectiveRank = rank,
                    resultScore = score,
                    rankingScore = score,
                    ingredientPercentiles = listOf(score, score),
                )
            }
            val request = DurableRadioRequest.directQueue(
                generation = generation,
                providerGenerationId = providerGeneration(),
                tracks = selectedIds.map(::track),
                trackIdentities = selectedIds.map(::identity),
                resolvedPowerampFileIds = selectedIds.map { it * 10L },
                label = query.displayLabel,
                origin = QueueOrigin.COMPOSED_RESULT_LIST,
                placement = DirectQueuePlacement.REPLACE_UPCOMING,
                findMusicSessionEvidence = sessionEvidence,
                findMusicTrackEvidence = rowEvidence,
                requestId = uuid(82),
                createdAtEpochMs = 1_000L,
            )
            val store = store(root, owner = "process-a")

            store.persist(request)
            val claimed = store.claim(request.requestId) as RadioRequestClaim.Claimed

            assertEquals(
                originalRanks,
                claimed.request.directQueue?.findMusicTrackEvidence?.map { it.objectiveRank },
            )
            assertEquals(
                allOfPlan,
                claimed.request.directQueue?.findMusicSessionEvidence?.allOfQueuePlan,
            )
        }

    private fun store(root: File, owner: String) = RadioRequestStore(
        rootDir = root,
        ownerToken = owner,
        clock = { 2_000L },
        atomicWriter = RadioRequestAtomicWriter(::atomicJvmWrite),
        atomicReader = JVM_READER,
        atomicDeleter = JVM_DELETER,
    )

    private fun radioRequest(
        requestId: String,
        config: RadioConfig = RadioConfig(
            numTracks = 1,
            selectionMode = SelectionMode.UNIFORM_SHUFFLE,
            shuffleSeed = 123_456L,
        ),
    ) = DurableRadioRequest.radio(
        generation = generation(),
        providerGenerationId = providerGeneration(),
        config = config,
        seed = PinnedRadioSeed(
            identity = identity(12),
            displayTrack = PowerampTrack(
                realId = 120L,
                title = "Seed",
                artist = "Artist",
                album = "Album",
                durationMs = 180_000,
                path = "/music/seed.flac",
            ),
            matchType = TrackMatcher.MatchType.METADATA_EXACT,
        ),
        showToasts = true,
        origin = QueueOrigin.TEXT_RESULT_RADIO,
        requestId = requestId,
        createdAtEpochMs = 1_000L,
    )

    private fun successfulResult(request: DurableRadioRequest): RadioResult {
        val queued = QueuedTrackResult(
            track = track(13),
            similarity = 0.9f,
            similarityToSeed = 0.9f,
            status = QueueStatus.QUEUED,
            resolvedPowerampFileId = 130L,
            resolvedPowerampQueueId = 1_130L,
            stableTrackSpanId = stableId(13),
        )
        return RadioResult(
            seedTrack = request.radio!!.seed.displayTrack,
            matchType = request.radio.seed.matchType,
            tracks = listOf(queued),
            delivery = QueueDeliverySummary.fromTracks(
                origin = request.radio.origin,
                requestedCount = request.radio.config.numTracks,
                rankedCount = 1,
                resolvedCount = 1,
                tracks = listOf(queued),
                verificationComplete = true,
            ),
            requestId = request.requestId,
            outcome = RadioSessionOutcome.SUCCEEDED,
            generation = request.generation,
            providerGenerationId = request.providerGenerationId,
            seedIdentity = request.radio.seed.identity,
        )
    }

    private fun partialResult(request: DurableRadioRequest, detail: String): RadioResult {
        val failed = QueuedTrackResult(
            track = track(13),
            similarity = 0.9f,
            similarityToSeed = 0.9f,
            status = QueueStatus.QUEUE_FAILED,
            resolvedPowerampFileId = 130L,
            stableTrackSpanId = stableId(13),
        )
        return RadioResult(
            seedTrack = request.radio!!.seed.displayTrack,
            matchType = request.radio.seed.matchType,
            tracks = listOf(failed),
            delivery = QueueDeliverySummary.fromTracks(
                origin = request.radio.origin,
                requestedCount = request.radio.config.numTracks,
                rankedCount = 1,
                resolvedCount = 1,
                tracks = listOf(failed),
                verificationComplete = false,
            ),
            requestId = request.requestId,
            outcome = RadioSessionOutcome.PARTIAL_FAILED,
            failureDetail = detail,
            generation = request.generation,
            providerGenerationId = request.providerGenerationId,
            seedIdentity = request.radio.seed.identity,
        )
    }

    private fun generation() = RadioGenerationToken(
        generationId = "index-generation-v2-${"1".repeat(64)}",
        activationBindingId = "activation-binding-v3-${"2".repeat(64)}",
        manifestSha256 = "3".repeat(64),
        embeddingSpecId = "clamp3-audio-v2",
        databaseContentSha256 = "4".repeat(64),
        orderedTrackSetSha256 = "5".repeat(64),
        stableTrackUidMappingSha256 = "6".repeat(64),
    )

    private fun providerGeneration() =
        "poweramp-provider-snapshot-v3-sha256:${"7".repeat(64)}"

    private fun composedRequest(requestId: String): DurableRadioRequest = DurableRadioRequest.multiSeed(
        generation = generation(),
        providerGenerationId = providerGeneration(),
        seeds = listOf(
            SeedSpec(FloatArray(768) { it / 768f }, 0.4f, "liquid", SeedType.TEXT),
            SeedSpec(FloatArray(768) { (it + 1) / 768f }, 0.35f, "night drive", SeedType.TEXT),
            SeedSpec(FloatArray(768) { -it / 768f }, -0.25f, "avoid", SeedType.SONG, 42L),
        ),
        seedIdentities = listOf(null, null, identity(42)),
        querySpec = composedQuerySpec(),
        config = RadioConfig(
            numTracks = 1,
            selectionMode = SelectionMode.CLOSEST,
            driftEnabled = false,
        ),
        composedContract = ComposedRadioContract(),
        showToasts = true,
        origin = QueueOrigin.COMPOSED_RADIO,
        requestId = requestId,
        createdAtEpochMs = 1_000L,
    )

    private fun composedQuerySpec(
        libraryAddedDays: Int? = null,
    ) = FindMusicQuerySpec(
        operator = FindMusicOperator.ALL_OF,
        textIngredients = listOf(
            FindMusicTextIngredient("liquid", 0.4f, negative = false),
            FindMusicTextIngredient("night drive", 0.35f, negative = false),
        ),
        songSeeds = listOf(
            FindMusicSongAnchor(
                trackId = 42L,
                stableTrackSpanId = stableId(42),
                artist = null,
                title = "avoid",
                weight = 0.25f,
                negative = true,
            ),
        ),
        resultLimit = 20,
        libraryAddedDays = libraryAddedDays,
        libraryBinding = StableIdentityGenerationBinding(
            bindingSpecId = "v2-active-index-generation-binding-v1",
            generationId = generation().generationId,
            activationBindingId = generation().activationBindingId,
            databaseContentSha256 = generation().databaseContentSha256,
            orderedTrackSetSha256 = generation().orderedTrackSetSha256,
        ),
    )

    private fun identity(id: Long) = RadioSeedIdentity(id, stableId(id))

    private fun stableId(id: Long) = "stable-track-span-v1-${id.toString(16).padStart(64, '0')}"

    private fun track(id: Long) = EmbeddedTrack(
        id = id,
        metadataKey = "artist|album|track-$id|180000",
        filenameKey = "track-$id.flac",
        artist = "Artist $id",
        album = "Album",
        title = "Track $id",
        durationMs = 180_000,
        filePath = "/music/track-$id.flac",
    )

    private fun payloadFile(root: File, requestId: String) =
        File(root, "radio_requests_v2/payloads/$requestId.json")

    private fun stateFile(root: File, requestId: String) =
        File(root, "radio_requests_v2/states/$requestId.state")

    private fun uuid(value: Int): String =
        UUID.nameUUIDFromBytes("request-$value".toByteArray()).toString()

    private fun atomicJvmWrite(file: File, bytes: ByteArray) {
        file.parentFile?.mkdirs()
        val temporary = File(file.parentFile, ".${file.name}.${UUID.randomUUID()}.tmp")
        FileOutputStream(temporary).use { output ->
            output.write(bytes)
            output.fd.sync()
        }
        try {
            Files.move(
                temporary.toPath(),
                file.toPath(),
                StandardCopyOption.ATOMIC_MOVE,
                StandardCopyOption.REPLACE_EXISTING,
            )
        } catch (_: AtomicMoveNotSupportedException) {
            Files.move(temporary.toPath(), file.toPath(), StandardCopyOption.REPLACE_EXISTING)
        }
    }

    private inline fun withTempDir(block: (File) -> Unit) {
        val root = Files.createTempDirectory("radio-request-store-").toFile()
        try {
            block(root)
        } finally {
            root.deleteRecursively()
        }
    }

    companion object {
        private val JVM_READER = RadioRequestAtomicReader { file ->
            file.takeIf(File::isFile)?.readBytes()
        }
        private val JVM_DELETER = RadioRequestAtomicDeleter { file ->
            !file.exists() || file.delete()
        }
    }
}
