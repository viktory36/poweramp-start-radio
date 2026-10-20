package com.powerampstartradio.similarity.algorithms

import android.util.Log
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import com.powerampstartradio.data.EmbeddingDatabase
import com.powerampstartradio.data.EmbeddingIndex
import com.powerampstartradio.data.StableTrackIdentityCatalog
import com.powerampstartradio.indexing.V2ActiveLibraryCatalogCodec
import com.powerampstartradio.similarity.ActiveRecommendationDomain
import com.powerampstartradio.similarity.RecommendationAssetFiles
import com.powerampstartradio.similarity.RecommendationEngine
import com.powerampstartradio.ui.RadioConfig
import com.powerampstartradio.ui.RadioResult
import com.powerampstartradio.ui.RadioSelectionMemory
import com.powerampstartradio.ui.QueueStatus
import com.powerampstartradio.ui.SelectionMode
import kotlinx.coroutines.runBlocking
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.util.UUID
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.system.measureTimeMillis

/** Read-only measurements against the phone's actual embeddings and native dot-product path. */
@RunWith(AndroidJUnit4::class)
class RememberedRadioInstrumentedTest {
    @Test
    fun nativeConditioningBeyond120StopsAtMathematicalExhaustion() {
        val context = ApplicationProvider.getApplicationContext<android.content.Context>()
        val file = File.createTempFile("remembered-space-", ".emb", context.cacheDir)
        try {
            val dim = 128
            val count = dim + 1
            val bytes = ByteBuffer.allocate(16 + count * 8 + count * dim * 4)
                .order(ByteOrder.LITTLE_ENDIAN)
            bytes.putInt(0x424D4550).putInt(1).putInt(count).putInt(dim)
            for (id in 1..count) bytes.putLong(id.toLong())
            for (row in 0 until count) {
                for (column in 0 until dim) bytes.putFloat(
                    if (row == dim) (1.0 / kotlin.math.sqrt(dim.toDouble())).toFloat()
                    else if (row == column) 1f else 0f,
                )
            }
            file.writeBytes(bytes.array())
            val index = EmbeddingIndex.mmap(file)
            val almostFullHistory = (1L..127L).toList()
            val candidates = listOf(128L to 1f, 129L to .9f)
            val partial = DppSelector.selectBatchCertified(
                candidates, 3, index, initialCandidateCount = 1,
                initiallySelectedTrackIds = almostFullHistory,
            )
            assertEquals(listOf(128L), partial.tracks.map { it.trackId })
            val fullHistory = almostFullHistory + 128L
            val exhausted = DppSelector.selectBatchCertified(
                listOf(129L to .9f), 3, index, initialCandidateCount = 1,
                initiallySelectedTrackIds = fullHistory,
            )
            assertTrue(exhausted.tracks.isEmpty())
            assertEquals(listOf(129L), MmrSelector.selectBatch(
                listOf(129L to .9f), 3, index, .9f, initiallySelectedTrackIds = fullHistory,
            ).map { it.trackId })
            Log.i("RememberedRadio", "128 remembered native directions: DPP partial=1 " +
                "exhausted=0, MMR remaining=1; no arbitrary cap or fill")
        } finally {
            file.delete()
        }
    }

    @Test
    fun realEngineRemembersHistoryAndKeepsGlobalRanks() = runBlocking<Unit> {
        val context = ApplicationProvider.getApplicationContext<android.content.Context>()
        val generations = File(context.filesDir, "indexing_v2/generations")
        val active = JSONObject(File(generations, "active-generation.json").readText())
        val generationId = active.getString("generationId")
        val directory = File(generations, generationId)
        val embeddingFile = File(directory, "clamp3.emb")
        val index = EmbeddingIndex.mmap(embeddingFile)
        val cacheFile = File(context.filesDir, "indexing_v2/active-library-catalog-v1.bin")
        val header = cacheFile.inputStream().use(V2ActiveLibraryCatalogCodec::readHeader)
        val catalog = cacheFile.inputStream().use {
            V2ActiveLibraryCatalogCodec.read(
                it, generationId, header.activationBindingId,
                active.getString("manifestSha256"), index.numTracks,
            )
        }
        val database = EmbeddingDatabase.open(File(directory, "library.db"))
        val memoryDirectory = File(context.cacheDir, "memory-check-${UUID.randomUUID()}").apply { mkdirs() }
        try {
            val identities = StableTrackIdentityCatalog.load(context.filesDir, database, index)
            val domain = ActiveRecommendationDomain.create(catalog, identities, index,
                database.loadIndexAddedAtEpochSeconds(identities.orderedTrackIds()))
            val seedId = 59474L
            val historyType = object : TypeToken<List<RadioResult>>() {}.type
            val history: List<RadioResult> = Gson().fromJson(
                File(context.filesDir, "session_history.json").readText(), historyType,
            )
            val memory = RadioSelectionMemory(memoryDirectory)
            memory.initializeFromHistory(history)
            val remembered = memory.rememberedTrackIds(
                seedId, identities.stableTrackSpanId(seedId), generationId,
                database::getTrackById,
            )
            val expectedCurrentGenerationIds = history.filter {
                it.seedIdentity?.embeddedTrackId == seedId &&
                    it.generation?.generationId == generationId &&
                    !it.isDirectQueue && it.composedQuerySpec == null
            }.flatMap { session ->
                session.tracks.filter { it.status == QueueStatus.QUEUED }.map { it.track.id }
            }.toSet()
            assertTrue(remembered.containsAll(expectedCurrentGenerationIds))
            assertTrue("Real history must exercise more than the removed 120-track cap", remembered.size > 120)
            val reopened = RadioSelectionMemory(memoryDirectory)
            // History trimming or clearing must not discard independent selection memory.
            reopened.initializeFromHistory(emptyList())
            assertEquals(remembered, reopened.rememberedTrackIds(
                seedId, identities.stableTrackSpanId(seedId), generationId, database::getTrackById,
            ))
            val engine = RecommendationEngine(
                database, context.filesDir,
                RecommendationAssetFiles(embeddingFile, null, header.activationBindingId),
                activeCatalog = catalog,
            )
            val similarities = index.computeAllSimilarities(requireNotNull(index.getEmbeddingByTrackId(seedId)))
            val rememberedIdentities = remembered.flatMap(identities::equivalentVisibleTrackIds).toSet()
            for (mode in listOf(SelectionMode.DPP, SelectionMode.MMR)) {
                val config = RadioConfig(
                    numTracks = 30, selectionMode = mode, rememberPrevious = true,
                    diversityLambda = .9f, artistLimitsEnabled = false,
                )
                val start = System.nanoTime()
                val first = engine.generatePlaylist(seedId, config, previouslyQueuedTrackIds = remembered)
                val elapsed = (System.nanoTime() - start) / 1_000_000
                val second = engine.generatePlaylist(seedId, config, previouslyQueuedTrackIds = remembered)
                assertEquals(30, first.size)
                assertEquals(first.map { it.track.id }, second.map { it.track.id })
                assertTrue(first.none { it.track.id in rememberedIdentities })
                for (track in first) {
                    assertEquals(
                        domain.rankEligibleIdentityFromFullSimilarities(similarities, track.track.id, seedId),
                        track.seedRank,
                    )
                }
                Log.i("RememberedRadio", "engine=$mode memory=${remembered.size} " +
                    "active=${domain.activeCandidateIdentityCount} ms=$elapsed globalRanks=verified")
            }
            val allOf = GeoMeanSelector.computeAllOfRankingSnapshot(
                index, listOf(requireNotNull(index.getEmbeddingByTrackId(seedId)) to 1f),
                excludeTrackIds = identities.equivalentVisibleTrackIds(seedId).toSet(),
                identityCatalog = identities, includedTrackIds = domain.activeTrackIds,
            )
            val firstRows = allOf.take(100)
            firstRows.forEach { assertEquals(it.objectiveRank, allOf.rankOf(it.row.trackId)) }
            assertEquals(null, allOf.rankOf(seedId))
            Log.i("RememberedRadio", "global Find Music rank lookup: 100 rows verified")

            val saved = history.last { it.seedIdentity?.embeddedTrackId == seedId && !it.isDirectQueue }
            reopened.recordSession(saved)
            assertEquals(remembered, reopened.rememberedTrackIds(
                seedId, identities.stableTrackSpanId(seedId), generationId, database::getTrackById,
            ))
            reopened.clear()
            val cleared = RadioSelectionMemory(memoryDirectory)
            cleared.initializeFromHistory(history)
            assertTrue(cleared.rememberedTrackIds(
                seedId, identities.stableTrackSpanId(seedId), generationId, database::getTrackById,
            ).isEmpty())
            cleared.recordSession(saved.copy(tracks = saved.tracks.map { it.copy(status = QueueStatus.QUEUE_FAILED) }))
            assertTrue(cleared.rememberedTrackIds(
                seedId, identities.stableTrackSpanId(seedId), generationId, database::getTrackById,
            ).isEmpty())
            cleared.recordSession(saved)
            assertTrue(cleared.rememberedTrackIds(
                seedId, identities.stableTrackSpanId(seedId), generationId, database::getTrackById,
            ).isNotEmpty())
            Log.i("RememberedRadio", "uncapped memory persistence, clear, no history refill, failed " +
                "enqueue exclusion, and new enqueues after clear: verified")
        } finally {
            database.close()
            memoryDirectory.deleteRecursively()
        }
    }

    @Test
    fun conditionedDppCertificateMatchesCompleteLibraryAndMmrRespectsMemory() {
        val context = ApplicationProvider.getApplicationContext<android.content.Context>()
        val generations = File(context.filesDir, "indexing_v2/generations")
        val active = JSONObject(File(generations, "active-generation.json").readText())
        val index = EmbeddingIndex.mmap(File(generations, "${active.getString("generationId")}/clamp3.emb"))
        val seedId = 59474L
        val embedding = requireNotNull(index.getEmbeddingByTrackId(seedId))
        val candidates = index.findTopK(embedding, index.numTracks, setOf(seedId))
        val earlier = DppSelector.selectBatch(candidates, 30, index).map { it.trackId }
        var complete = emptyList<com.powerampstartradio.similarity.SelectedTrack>()
        val fullMs = measureTimeMillis {
            complete = DppSelector.selectBatch(
                candidates, 30, index, initiallySelectedTrackIds = earlier,
            )
        }
        lateinit var certified: DppSelector.CertifiedSelection
        val certifiedMs = measureTimeMillis {
            certified = DppSelector.selectBatchCertified(
                candidates, 30, index, initialCandidateCount = 1900,
                initiallySelectedTrackIds = earlier,
            )
        }
        assertEquals(complete.map { it.trackId }, certified.tracks.map { it.trackId })
        assertEquals(30, complete.size)
        assertTrue(complete.none { it.trackId in earlier })
        val repeat = DppSelector.selectBatchCertified(
            candidates, 30, index, initialCandidateCount = 1900,
            initiallySelectedTrackIds = earlier,
        )
        assertEquals(certified.tracks, repeat.tracks)
        val mmr = MmrSelector.selectBatch(
            candidates.take(1900), 30, index, .9f, initiallySelectedTrackIds = earlier,
        )
        assertEquals(30, mmr.size)
        assertTrue(mmr.none { it.trackId in earlier })
        Log.i("RememberedRadio", "rows=${index.numTracks} seed=$seedId fullMs=$fullMs " +
            "certifiedMs=$certifiedMs attempts=${certified.evidence.attemptedCandidateCounts} " +
            "dpp=${complete.map { it.trackId }} mmr=${mmr.map { it.trackId }}")
    }
}
