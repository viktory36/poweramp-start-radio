package com.powerampstartradio.similarity

import android.content.Context
import android.util.Log
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import com.powerampstartradio.data.EmbeddingDatabase
import com.powerampstartradio.data.EmbeddingIndex
import com.powerampstartradio.data.StableTrackIdentityCatalog
import com.powerampstartradio.indexing.Clamp3TextInference
import com.powerampstartradio.indexing.V2ActiveLibraryCatalogCodec
import com.powerampstartradio.similarity.algorithms.DppSelector
import com.powerampstartradio.similarity.algorithms.GeoMeanSelector
import com.powerampstartradio.ui.*
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.util.UUID
import kotlin.system.measureTimeMillis

/** Production native algorithms and the real library; no Poweramp or owner-memory writes. */
@RunWith(AndroidJUnit4::class)
class FindMusicMemoryInstrumentedTest {
    @Test
    fun realSearchesRemainRankedDeterministicAndUncapped() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val generations = File(context.filesDir, "indexing_v2/generations")
        val active = JSONObject(File(generations, "active-generation.json").readText())
        val generationId = active.getString("generationId")
        val directory = File(generations, generationId)
        val index = EmbeddingIndex.mmap(File(directory, "clamp3.emb"))
        val cacheFile = File(context.filesDir, "indexing_v2/active-library-catalog-v1.bin")
        val header = cacheFile.inputStream().use(V2ActiveLibraryCatalogCodec::readHeader)
        val catalog = cacheFile.inputStream().use {
            V2ActiveLibraryCatalogCodec.read(
                it, generationId, header.activationBindingId,
                active.getString("manifestSha256"), index.numTracks,
            )
        }
        val database = EmbeddingDatabase.open(File(directory, "library.db"))
        val temporary = File(context.cacheDir, "find-memory-${UUID.randomUUID()}").apply { mkdirs() }
        try {
            val identities = StableTrackIdentityCatalog.load(context.filesDir, database, index)
            val domain = ActiveRecommendationDomain.create(catalog, identities, index,
                database.loadIndexAddedAtEpochSeconds(identities.orderedTrackIds()))
            val candidates = domain.candidateDomain(null)
            val ids = candidates.orderedIdentityRepresentativeTrackIds()
            val historyType = object : TypeToken<List<RadioResult>>() {}.type
            val history: List<RadioResult> = Gson().fromJson(
                File(context.filesDir, "session_history.json").readText(), historyType,
            )
            val template = history.last { it.generation?.generationId == generationId }
            val memory = RadioSelectionMemory(temporary)
            memory.initializeFromHistory(emptyList())

            fun remembered(query: FindMusicQuerySpec): List<Long> =
                RadioSelectionMemory(temporary).rememberedSearchTrackIds(
                    query, generationId, database::getTrackById,
                ).map(domain::activeIdentityRepresentativeTrackId).distinct().sorted()

            fun session(query: FindMusicQuerySpec, selected: List<Long>): RadioResult = template.copy(
                isDirectQueue = true,
                composedQuerySpec = null,
                tracks = selected.map { id ->
                    QueuedTrackResult(
                        requireNotNull(database.getTrackById(id)), 0f, 0f, status = QueueStatus.QUEUED,
                    )
                },
                findMusicSessionEvidence = FindMusicSessionEvidence(
                    querySpec = query,
                    orderedActiveTrackIdsSha256 = domain.orderedActiveTrackIdsSha256,
                    activeTrackCount = domain.activeTrackCount,
                    stableResultReduction = StableResultReductionEvidence(1, selected.size, selected.size, 0),
                ),
            )

            fun logRows(label: String, selected: List<Long>, ranks: List<Int>, elapsed: Long) {
                val rows = selected.take(8).mapIndexed { position, id ->
                    val track = requireNotNull(database.getTrackById(id))
                    "#${ranks[position]} ${track.artist} - ${track.title}"
                }
                Log.i(TAG, "$label count=${selected.size} ms=$elapsed ${rows.joinToString(" | ")}")
            }

            val vectors = mutableListOf<FloatArray>()
            val textModel = Clamp3TextInference(
                File(context.filesDir, "clamp3_text.tflite"),
                File(context.filesDir, "sentencepiece.bpe.model"),
            )
            try {
                for (description in listOf("ambient", "guitar")) {
                    val ms = measureTimeMillis {
                        vectors += requireNotNull(textModel.generateEmbedding(description))
                    }
                    Log.i(TAG, "text='$description' inferenceMs=$ms")
                }
            } finally {
                textModel.close()
            }
            val scores = index.computeAllSimilarities(vectors[0])
            val textRanking = StableSimilarityTopK.select(
                orderedTrackIds = ids,
                similarities = candidates.identityRepresentativeScoresFromFull(scores),
                topK = ids.size,
                rankingTieKey = identities::rankingTieKey,
            )
            val textQuery = FindMusicQuerySpec(
                textIngredients = listOf(FindMusicTextIngredient("ambient", 1f, false)),
                rememberPrevious = true,
            )
            val first = FindMusicTextQueuePlanner.plan(
                FindMusicTextResultPlanner.CLOSEST, textRanking, 30, index,
            )
            assertTrue(remembered(textQuery).isEmpty()) // Viewing alone does not record results.
            memory.recordSession(session(textQuery, first.selections.map { it.trackId }))
            val controlsChanged = textQuery.copy(resultLimit = 40, libraryAddedDays = 14)
            assertEquals(first.selections.map { it.trackId }.sorted(), remembered(controlsChanged))
            val closest = FindMusicTextQueuePlanner.plan(
                FindMusicTextResultPlanner.CLOSEST, textRanking, 30, index,
                previouslyQueuedTrackIds = remembered(textQuery),
            )
            assertEquals((31..60).toList(), closest.selections.map { it.originalTextObjectiveRank })

            val variedQuery = textQuery.copy(textResultPlanner = FindMusicTextResultPlanner.VARIED_DPP)
            var previous = remembered(variedQuery)
            repeat(3) { iteration ->
                lateinit var plan: FindMusicTextQueuePlan
                val elapsed = measureTimeMillis {
                    plan = FindMusicTextQueuePlanner.plan(
                        FindMusicTextResultPlanner.VARIED_DPP, textRanking, 30, index,
                        previouslyQueuedTrackIds = previous,
                    )
                }
                val selected = plan.selections.map { it.trackId }
                assertEquals(30, selected.size)
                assertTrue(selected.none { it in previous })
                assertEquals(plan, FindMusicTextQueuePlanner.plan(
                    FindMusicTextResultPlanner.VARIED_DPP, textRanking, 30, index,
                    previouslyQueuedTrackIds = previous,
                ))
                assertEquals(selected, DppSelector.selectBatch(
                    textRanking.map { it.trackId to it.score }, 30, index,
                    qualityExponent = 4f, initiallySelectedTrackIds = previous,
                ).map { it.trackId })
                plan.selections.forEach { assertEquals(textRanking[it.originalTextObjectiveRank - 1].trackId, it.trackId) }
                logRows("ambient varied iteration=${iteration + 1} memory=${previous.size}", selected,
                    plan.selections.map { it.originalTextObjectiveRank }, elapsed)
                memory.recordSession(session(variedQuery, selected))
                previous = remembered(variedQuery)
            }
            assertEquals(120, previous.size)
            // One more confirmed queue proves storage does not cap the accumulated recipe at 120.
            memory.recordSession(session(variedQuery, closest.selections.map { it.trackId }))
            val extra = textRanking.take(200).map { it.trackId }
            memory.recordSession(session(variedQuery, extra))
            assertTrue(remembered(variedQuery).size >= 200)

            val mixedQuery = FindMusicQuerySpec(
                textIngredients = listOf(
                    FindMusicTextIngredient("ambient", .5f, false),
                    FindMusicTextIngredient("guitar", .5f, false),
                ), rememberPrevious = true,
            )
            val allOf = GeoMeanSelector.computeRankingDetailedSnapshot(
                index, vectors.map { it to .5f }, FindMusicOperator.ALL_OF, ids.size,
                identityCatalog = identities, includedTrackIds = domain.activeTrackIds,
            )
            val rankedIds = allOf.rows.take(30).map { it.trackId }
            memory.recordSession(session(mixedQuery, rankedIds))
            val mixedPrevious = remembered(mixedQuery)
            lateinit var mixed: FindMusicAllOfQueuePlan
            val mixedMs = measureTimeMillis {
                mixed = FindMusicAllOfQueuePlanner.plan(
                    allOf.rows, 30, index, previouslyQueuedTrackIds = mixedPrevious,
                )
            }
            val mixedIds = mixed.selections.map { it.row.trackId }
            assertEquals(30, mixedIds.size)
            assertTrue(mixedIds.none { it in mixedPrevious })
            assertEquals(mixedIds, DppSelector.selectBatch(
                allOf.rows.map { it.trackId to it.objectiveScore }, 30, index,
                qualityExponent = FindMusicAllOfQueuePlanner.QUALITY_EXPONENT,
                initiallySelectedTrackIds = mixedPrevious,
            ).map { it.trackId })
            assertEquals(mixed, FindMusicAllOfQueuePlanner.plan(
                allOf.rows, 30, index, previouslyQueuedTrackIds = mixedPrevious,
            ))
            mixed.selections.forEach { assertEquals(allOf.rows[it.originalAllOfObjectiveRank - 1], it.row) }
            logRows("ambient + guitar varied memory=${mixedPrevious.size}", mixedIds,
                mixed.selections.map { it.originalAllOfObjectiveRank }, mixedMs)

            val refineQuery = mixedQuery.copy(
                operator = FindMusicOperator.REFINE,
                refineSpec = FindMusicRefineSpec(0, FindMusicRefineNeighborhood.TOP_0_25_PERCENT),
            )
            val refined = GeoMeanSelector.computeRankingDetailedSnapshot(
                index, vectors.map { it to .5f }, FindMusicOperator.REFINE, ids.size,
                refineSpec = refineQuery.refineSpec,
                identityCatalog = identities, includedTrackIds = domain.activeTrackIds,
            )
            assertEquals(refineQuery.refineSpec!!.neighborhood.candidateCount(ids.size), refined.rows.size)
            val refineSelected = refined.rows.take(refined.rows.size - 5).map { it.trackId }
            memory.recordSession(session(refineQuery, refineSelected))
            val refineMemory = remembered(refineQuery).toHashSet()
            val refineRemainder = refined.rows.withIndex().filter { it.value.trackId !in refineMemory }.take(30)
            assertEquals(5, refineRemainder.size)
            assertEquals((refined.rows.size - 4..refined.rows.size).toList(), refineRemainder.map { it.index + 1 })
            memory.recordSession(session(refineQuery, refined.rows.map { it.trackId }))
            val exhaustedRefineMemory = remembered(refineQuery).toHashSet()
            assertTrue(refined.rows.none { it.trackId !in exhaustedRefineMemory })
            val small = textRanking.take(5)
            for (planner in listOf(FindMusicTextResultPlanner.CLOSEST, FindMusicTextResultPlanner.VARIED_DPP)) {
                val short = FindMusicTextQueuePlanner.plan(
                    planner, small, 30, index, previouslyQueuedTrackIds = small.take(4).map { it.trackId },
                )
                assertEquals(listOf(small.last().trackId), short.selections.map { it.trackId })
                assertTrue(FindMusicTextQueuePlanner.plan(
                    planner, small, 30, index, previouslyQueuedTrackIds = small.map { it.trackId },
                ).selections.isEmpty())
            }
            val accumulated = remembered(variedQuery)
            val reopened = RadioSelectionMemory(temporary)
            reopened.initializeFromHistory(emptyList())
            assertEquals(accumulated, remembered(variedQuery))
            reopened.recordSession(session(variedQuery, textRanking.takeLast(3).map { it.trackId }).let { saved ->
                saved.copy(tracks = saved.tracks.map { it.copy(status = QueueStatus.QUEUE_FAILED) })
            })
            assertEquals(accumulated, remembered(variedQuery))
            reopened.clear()
            reopened.initializeFromHistory(listOf(session(variedQuery, extra), session(refineQuery, refineSelected)))
            assertTrue(remembered(variedQuery).isEmpty())
            assertTrue(remembered(refineQuery).isEmpty())
            Log.i(TAG, "PASS rows=${index.numTracks} activeIdentities=${ids.size} uncappedMemory=${accumulated.size} " +
                "refineDomain=${refined.rows.size} ranks/determinism/fullDpp/persistence/clear/failedEnqueues/smallDomains verified")
        } finally {
            database.close()
            temporary.deleteRecursively()
        }
    }

    private companion object { const val TAG = "FindMusicMemoryCheck" }
}
