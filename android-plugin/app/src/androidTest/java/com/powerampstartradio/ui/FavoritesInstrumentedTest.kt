package com.powerampstartradio.ui

import android.content.Context
import android.database.sqlite.SQLiteDatabase
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
import com.powerampstartradio.indexing.v2.FavoritesIndexUpdate
import com.powerampstartradio.services.RadioService
import com.powerampstartradio.similarity.ActiveRecommendationDomain
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.util.UUID
import java.time.LocalDateTime
import java.time.ZoneOffset
import kotlin.system.measureTimeMillis

/** Real production ranking/binding, isolated count/update stores, no Poweramp queue mutation. */
@RunWith(AndroidJUnit4::class)
class FavoritesInstrumentedTest {
    @Test
    fun countsFreshCosineAndUpdateBoundaries() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val generations = File(context.filesDir, "indexing_v2/generations")
        val active = JSONObject(File(generations, "active-generation.json").readText())
        val generationId = active.getString("generationId")
        val directory = File(generations, generationId)
        val index = EmbeddingIndex.mmap(File(directory, "clamp3.emb"))
        val cache = File(context.filesDir, "indexing_v2/active-library-catalog-v1.bin")
        val header = cache.inputStream().use(V2ActiveLibraryCatalogCodec::readHeader)
        val catalog = cache.inputStream().use {
            V2ActiveLibraryCatalogCodec.read(it, generationId, header.activationBindingId,
                active.getString("manifestSha256"), index.numTracks)
        }
        val db = EmbeddingDatabase.open(File(directory, "library.db"))
        val temporary = File(context.cacheDir, "favorites-${UUID.randomUUID()}").apply { mkdirs() }
        try {
            val identities = StableTrackIdentityCatalog.load(context.filesDir, db, index)
            val addedDates = db.loadIndexAddedAtEpochSeconds(identities.orderedTrackIds())
            val domain = ActiveRecommendationDomain.create(catalog, identities, index, addedDates)
            val referenceDates = HashMap<Long, Long>()
            SQLiteDatabase.openDatabase(db.databaseFile.path, null, SQLiteDatabase.OPEN_READONLY).use { sql ->
                sql.rawQuery("SELECT id, created_at FROM tracks ORDER BY id", null).use { cursor ->
                    while (cursor.moveToNext()) {
                        referenceDates[cursor.getLong(0)] = cursor.getString(1)?.let {
                            runCatching { LocalDateTime.parse(it.replace(' ', 'T'))
                                .toEpochSecond(ZoneOffset.UTC).coerceAtLeast(0L) }.getOrDefault(0L)
                        } ?: 0L
                    }
                }
            }
            identities.orderedTrackIds().forEachIndexed { position, id ->
                assertEquals("Index date $id", referenceDates.getValue(id), addedDates[position])
            }
            val referenceNow = System.currentTimeMillis() / 1_000L
            listOf(1, 7, 30, 90, 365).forEach { days ->
                val cutoff = minimumLibraryAddedAtEpochSecond(days, referenceNow)!!
                val expected = domain.orderedActiveTrackIds().filter {
                    referenceDates.getValue(it) > 0L && referenceDates.getValue(it) >= cutoff
                }
                val actual = domain.candidateDomain(cutoff).orderedEligibleTrackIds().toList()
                assertEquals("Index window $days days", expected, actual)
                val providerCount = domain.orderedActiveTrackIds().count {
                    (catalog.createdAtEpochSecondForTrack(it) ?: 0L) >= cutoff
                }
                Log.i(TAG, "indexWindowDays=$days eligibleRows=${actual.size} oldPowerampRows=$providerCount exact=pass")
            }
            Log.i(TAG, "indexDates=pass rows=${addedDates.size} independentUtcParsing=pass")
            val candidates = domain.candidateDomain(null)
            val ids = candidates.orderedIdentityRepresentativeTrackIds()
            val seedId = ids.first()
            val seed = checkNotNull(db.getTrackById(seedId))
            val history: List<RadioResult> = Gson().fromJson(
                File(context.filesDir, "session_history.json").readText(),
                object : TypeToken<List<RadioResult>>() {}.type,
            )
            val template = history.last { it.generation?.generationId == generationId }
            val row = QueuedTrackResult(seed, 0f, 0f, status = QueueStatus.QUEUED)
            val radio = template.copy(
                requestId = "favorites-radio", tracks = listOf(row), isDirectQueue = false,
                findMusicSessionEvidence = null, composedQuerySpec = null,
                seedIdentity = RadioSeedIdentity(seedId, identities.stableTrackSpanId(seedId)),
                seedTrack = template.seedTrack.copy(title = seed.title.orEmpty(), artist = seed.artist),
            )
            val store = FavoritesStore(temporary)
            store.initializeFromHistory(listOf(radio, radio))
            store.recordSession(radio)
            assertEquals(1, store.entries().single().count)
            val query = FindMusicQuerySpec(
                textIngredients = listOf(
                    FindMusicTextIngredient("sleep", 1f, false),
                    FindMusicTextIngredient("sleep", 1f, false),
                    FindMusicTextIngredient("ambient", 1f, false),
                    FindMusicTextIngredient("inactive", 0f, false),
                ),
                songSeeds = listOf(FindMusicSongAnchor(seedId,
                    identities.stableTrackSpanId(seedId), seed.artist, seed.title, 1f, false)),
            )
            val composed = radio.copy(requestId = "favorites-multi", isDirectQueue = true,
                findMusicSessionEvidence = FindMusicSessionEvidence(querySpec = query,
                    orderedActiveTrackIdsSha256 = domain.orderedActiveTrackIdsSha256,
                    activeTrackCount = domain.activeTrackCount,
                    stableResultReduction = StableResultReductionEvidence(1, 1, 1, 0)))
            store.recordSession(composed)
            store.recordSession(composed)
            store.recordSession(composed.copy(requestId = "failed", tracks = listOf(row.copy(status = QueueStatus.QUEUE_FAILED))))
            assertEquals(listOf(2, 1, 1), store.entries().map { it.count })
            store.initializeFromHistory(listOf(radio.copy(requestId = "should-not-reimport")))
            assertEquals(listOf(2, 1, 1), FavoritesStore(temporary).entries().map { it.count })
            Log.i(TAG, "counts=pass single+multi+text individual/idempotent/reopen/failed/history-independent")

            val vectors = mutableListOf(index.getEmbeddingByTrackId(seedId)!!)
            val inference = Clamp3TextInference(File(context.filesDir, "clamp3_text.tflite"),
                File(context.filesDir, "sentencepiece.bpe.model"),
                accelerator = com.google.ai.edge.litert.Accelerator.CPU, strictAccelerator = true)
            try {
                vectors += checkNotNull(inference.generateEmbedding("sleep"))
            } finally { inference.close() }
            vectors.forEachIndexed { kind, vector ->
                val exclude = if (kind == 0) identities.equivalentVisibleTrackIds(seedId).toSet() else emptySet()
                val scores = candidates.identityRepresentativeScoresFromFull(index.computeAllSimilarities(vector))
                val expected = ids.indices.filter { ids[it] !in exclude }.sortedWith(
                    compareByDescending<Int> { scores[it] }.thenBy { identities.rankingTieKey(ids[it]) },
                ).map { ids[it] }
                var matches = emptyList<FavoriteMatch>()
                val elapsed = measureTimeMillis {
                    matches = FavoritesRanking.nearest(vector, seedId.takeIf { kind == 0 }, 100,
                        index, domain, identities, db, emptySet())
                }
                assertEquals(expected.take(100), matches.map { it.track.id })
                assertEquals((1..100).toList(), matches.map { it.rank })
                matches.forEach { assertEquals(referenceDates.getValue(it.track.id), it.track.indexAddedAtEpochSecond) }
                assertTrue(matches.none { it.isNew })
                assertEquals(matches, FavoritesRanking.nearest(vector, seedId.takeIf { kind == 0 }, 100,
                    index, domain, identities, db, emptySet()))
                assertEquals(expected.take(7), FavoritesRanking.nearest(vector, seedId.takeIf { kind == 0 }, 7,
                    index, domain, identities, db, emptySet()).map { it.track.id })
                val additions = matches.take(3).flatMap { identities.equivalentVisibleTrackIds(it.track.id) }.toSet()
                val tagged = FavoritesRanking.nearest(vector, seedId.takeIf { kind == 0 }, 100,
                    index, domain, identities, db, additions)
                assertEquals(matches.map { it.track.id }, tagged.map { it.track.id })
                assertEquals(matches.take(3).map { it.track.id }, tagged.filter { it.isNew }.map { it.track.id })
                assertEquals(tagged.any { it.isNew }, FavoritesRanking.hasNewNearest(vector,
                    seedId.takeIf { kind == 0 }, 100, index, domain, identities, additions))
                val boundaryAddition = identities.equivalentVisibleTrackIds(matches.last().track.id).toSet()
                assertFalse(FavoritesRanking.hasNewNearest(vector, seedId.takeIf { kind == 0 },
                    99, index, domain, identities, boundaryAddition))
                assertTrue(FavoritesRanking.hasNewNearest(vector, seedId.takeIf { kind == 0 },
                    100, index, domain, identities, boundaryAddition))
                val duplicateId = ids.firstOrNull { identities.equivalentVisibleTrackIds(it).size > 1 }
                if (duplicateId != null) {
                    val copyIds = identities.equivalentVisibleTrackIds(duplicateId)
                    val duplicateVector = index.getEmbeddingByTrackId(duplicateId)!!
                    val duplicateRows = FavoritesRanking.nearest(duplicateVector, null, 100,
                        index, domain, identities, db, setOf(copyIds.last()))
                    assertFalse(duplicateRows.first { it.track.id == duplicateId }.isNew)
                }
                Log.i(TAG, "kind=$kind active=${ids.size} top100Ms=$elapsed exactCosine=pass deterministic=pass " +
                    "newAndDuplicateBadges=pass rows=${matches.take(3).map { it.track.artist + " - " + it.track.title }}")
            }

            val ownerFavorites = FavoritesStore(context.filesDir).entries()
            val badgeSeed = ownerFavorites.first { it.seed.trackId != null }.seed
            val badgeSeedResolution = badgeSeed.stableTrackSpanId?.let(identities::resolveStable)
                as? com.powerampstartradio.data.StableTrackIdentityResolution.Resolved
            val badgeSeedId = badgeSeedResolution?.trackId ?: badgeSeed.trackId!!
            val newNeighbor = FavoritesRanking.nearest(index.getEmbeddingByTrackId(badgeSeedId)!!,
                badgeSeedId, 100, index, domain, identities, db, emptySet()).first().track.id
            val simulatedAdditions = identities.equivalentVisibleTrackIds(newNeighbor).toSet()
            val flags = LinkedHashMap<String, Boolean>()
            val broadMs = measureTimeMillis {
                ownerFavorites.filter { it.seed.trackId != null }.forEach { entry ->
                    val stable = entry.seed.stableTrackSpanId?.let(identities::resolveStable)
                        as? com.powerampstartradio.data.StableTrackIdentityResolution.Resolved
                    val id = stable?.trackId ?: entry.seed.trackId!!
                    val vector = index.getEmbeddingByTrackId(id) ?: return@forEach
                    val hasNew = FavoritesRanking.hasNewNearest(vector, id, 100, index, domain,
                        identities, simulatedAdditions)
                    val expected = FavoritesRanking.nearest(vector, id, 100, index, domain,
                        identities, db, simulatedAdditions).any { it.isNew }
                    assertEquals(entry.seed.label, expected, hasNew)
                    flags[entry.seed.key] = hasNew
                }
            }
            assertTrue(flags.size >= 20)
            assertTrue("Expected at least one positive New badge", flags.values.any { it })
            assertTrue("Expected at least one unmarked favorite", flags.values.any { !it })
            val cacheContext = "$generationId|active-domain|100"
            store.saveNewMatchFlags(cacheContext, flags)
            assertEquals(flags, FavoritesStore(temporary).newMatchFlags(cacheContext))
            assertTrue(store.newMatchFlags("$generationId|active-domain|99").isEmpty())
            assertTrue(store.newMatchFlags("next-generation|active-domain|100").isEmpty())
            assertTrue(store.newMatchFlags("$generationId|changed-active-domain|100").isEmpty())
            Log.i(TAG, "favoriteBadges=pass songSeeds=${flags.size} simulatedNew=${flags.values.count { it }} " +
                "rankAndReferenceMs=$broadMs exactTopNBoundary=pass persistentCacheAndInvalidation=pass")

            val pinned = RadioService.buildFavoriteRadioRequest(context, seedId, generationId, RadioConfig())
            assertEquals(seedId, pinned.radio!!.seed.identity.embeddedTrackId)
            assertEquals(QueueOrigin.FAVORITE_RADIO, pinned.radio!!.origin)
            assertEquals(catalog.powerampFileIdForTrack(seedId), pinned.radio!!.seed.displayTrack.realId)
            assertEquals(RadioConfig().forSelectionRequest(
                candidates.eligibleCandidateIdentityCount(seedId)), pinned.radio!!.config)
            assertTrue(runCatching {
                RadioService.buildFavoriteRadioRequest(context, seedId, "old-generation", RadioConfig())
            }.isFailure)
            Log.i(TAG, "selectedRadio=pass pinned exact Poweramp file, configured settings, stale rejection; queue untouched")
            checkUpdateFiles(temporary)
        } finally {
            db.close()
            temporary.deleteRecursively()
        }
    }

    private fun checkUpdateFiles(temporary: File) {
        val root = File(temporary, "indexing_v2/generations")
        fun generation(name: String, ids: List<Long>): File {
            val directory = File(root, name).apply { mkdirs() }
            SQLiteDatabase.openOrCreateDatabase(File(directory, "library.db"), null).use { db ->
                db.execSQL("CREATE TABLE embeddings_clamp3(track_id INTEGER PRIMARY KEY)")
                ids.forEach { db.execSQL("INSERT INTO embeddings_clamp3 VALUES (?)", arrayOf(it)) }
            }
            return directory
        }
        val first = generation("first", listOf(1L, 2L))
        val second = generation("second", listOf(1L, 2L, 3L, 4L))
        val third = generation("third", listOf(1L, 2L, 3L, 4L, 5L))
        FavoritesIndexUpdate.prepare(temporary, null, "first")
        assertTrue(FavoritesIndexUpdate.read(first).isEmpty())
        FavoritesIndexUpdate.prepare(temporary, "first", "second")
        first.deleteRecursively()
        assertEquals(setOf(3L, 4L), FavoritesIndexUpdate.read(second))
        FavoritesIndexUpdate.prepare(temporary, "first", "second")
        assertEquals(setOf(3L, 4L), FavoritesIndexUpdate.read(second))
        FavoritesIndexUpdate.prepare(temporary, "second", "third")
        assertEquals(setOf(5L), FavoritesIndexUpdate.read(third))
        val fourth = generation("fourth", listOf(1L, 2L, 3L, 4L, 5L))
        FavoritesIndexUpdate.prepare(temporary, "third", "fourth")
        assertTrue(FavoritesIndexUpdate.read(fourth).isEmpty())
        Log.i(TAG, "indexUpdates=pass added-only, survives previous pruning/retry, next update replaces badges")
    }

    companion object { private const val TAG = "FavoritesCheck" }
}
