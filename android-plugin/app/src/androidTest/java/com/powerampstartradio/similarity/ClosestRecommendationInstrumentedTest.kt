package com.powerampstartradio.similarity

import android.util.Log
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.powerampstartradio.data.EmbeddingDatabase
import com.powerampstartradio.data.EmbeddingIndex
import com.powerampstartradio.data.StableTrackIdentityCatalog
import com.powerampstartradio.indexing.V2ActiveLibraryCatalogLoader
import com.powerampstartradio.indexing.v2.V2PowerampProviderSnapshotAcquirer
import com.powerampstartradio.indexing.v2.V2IndexGenerationReader
import com.powerampstartradio.similarity.algorithms.ClosestSelector
import com.powerampstartradio.ui.RadioConfig
import com.powerampstartradio.ui.SelectionMode
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class ClosestRecommendationInstrumentedTest {
    @Test
    fun closestMatchesNativeCosineRankingAndRepeatsExactly() = runBlocking {
        val context = ApplicationProvider.getApplicationContext<android.content.Context>()
        val active = V2IndexGenerationReader.requireActive(context.filesDir)
        val dbFile = active.databaseFile
        assertTrue("Active V2 database is missing", dbFile.isFile)

        val provider = V2PowerampProviderSnapshotAcquirer(context).acquireBlocking()
        val activeCatalog = V2ActiveLibraryCatalogLoader.load(active, provider)
        val seedId = checkNotNull(activeCatalog.activeTrackIds.minOrNull()) {
            "Frozen active provider catalog contains no tracks"
        }

        val database = EmbeddingDatabase.open(dbFile)
        try {
            val engine = RecommendationEngine(
                database = database,
                filesDir = context.filesDir,
                pinnedAssets = RecommendationAssetFiles(
                    embeddingFile = active.embeddingFile,
                    graphFile = active.graphFile,
                ),
                activeCatalog = activeCatalog,
            )
            val startedAt = System.nanoTime()
            engine.ensureIndices()

            val index = EmbeddingIndex.mmap(active.embeddingFile)
            val seedEmbedding = checkNotNull(index.getEmbeddingByTrackId(seedId))
            val identities = StableTrackIdentityCatalog.load(context.filesDir, database, index)
            val identityPolicy = StableTrackSelectionPolicy(
                identityForTrack = identities::visibleResultIdentity,
                equivalentTrackIds = identities::equivalentVisibleTrackIds,
            )
            val excluded = activeCatalog.quarantinedTrackIds.toMutableSet().apply {
                addAll(identityPolicy.exclusionClosure(listOf(seedId)))
            }
            val similarities = index.computeAllSimilarities(seedEmbedding)
            val candidates = StableSimilarityTopK.select(
                orderedTrackIds = identities.orderedTrackIds(),
                similarities = similarities,
                topK = activeCatalog.activeTrackIds.size,
                rankingTieKey = identities::rankingTieKey,
                excludeIds = excluded,
            ).map { it.trackId to it.score }
            val expected = ClosestSelector.select(
                candidates = candidates,
                numSelect = QUEUE_SIZE,
                isEligible = identityPolicy::canSelect,
            )

            val config = RadioConfig(
                numTracks = QUEUE_SIZE,
                selectionMode = SelectionMode.CLOSEST,
                driftEnabled = true,
                artistLimitsEnabled = false,
            )
            val first = engine.generatePlaylist(seedId, config)
            val second = engine.generatePlaylist(seedId, config)

            assertEquals(expected.map { it.trackId }, first.map { it.track.id })
            assertEquals(expected.map { it.score }, first.map { it.similarity })
            assertTrue(first.all { it.track.id in activeCatalog.activeTrackIds })
            assertEquals(first.map { it.track.id }, second.map { it.track.id })
            assertEquals(first.map { it.similarity }, second.map { it.similarity })
            assertEquals(expected.map { it.candidateRank }, first.map { it.candidateRank })

            val elapsedMs = (System.nanoTime() - startedAt) / 1_000_000
            val metric = "seed=$seedId tracks=${index.numTracks} queue=$QUEUE_SIZE " +
                "active=${activeCatalog.activeTrackIds.size} totalMs=$elapsedMs"
            Log.i(TAG, metric)
            println("PASR_METRIC $TAG $metric")
        } finally {
            database.close()
        }
    }

    private companion object {
        const val TAG = "ClosestV2Test"
        const val QUEUE_SIZE = 30
    }
}
