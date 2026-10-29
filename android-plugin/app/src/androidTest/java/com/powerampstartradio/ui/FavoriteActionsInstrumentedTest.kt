package com.powerampstartradio.ui

import android.app.Application
import android.net.Uri
import android.os.SystemClock
import android.util.Log
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.ViewModelStore
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.powerampstartradio.data.EmbeddingDatabase
import com.powerampstartradio.data.EmbeddingIndex
import com.powerampstartradio.data.StableTrackIdentityCatalog
import com.powerampstartradio.indexing.V2ActiveLibraryCatalogStore
import com.powerampstartradio.indexing.v2.V2LibraryDatabaseResolver
import com.powerampstartradio.poweramp.PowerampReceiver
import com.powerampstartradio.services.RadioRequestClaim
import com.powerampstartradio.services.RadioRequestStore
import com.powerampstartradio.services.RadioService
import com.powerampstartradio.similarity.ActiveRecommendationDomain
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.util.UUID

/** Uses real library/model/UI-controller code without changing playback or sending a queue. */
@RunWith(AndroidJUnit4::class)
class FavoriteActionsInstrumentedTest {
    @Test
    fun browseNeighborsAndRetraceWithoutQueueing() {
        val app = ApplicationProvider.getApplicationContext<Application>()
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val viewModels = ViewModelStore()
        lateinit var vm: MainViewModel
        instrumentation.runOnMainSync {
            vm = ViewModelProvider(viewModels, ViewModelProvider.AndroidViewModelFactory(app))[MainViewModel::class.java]
        }
        fun queuedFileIds(): List<Long> = checkNotNull(app.contentResolver.query(
            Uri.parse("content://com.maxmpz.audioplayer.data/queue"),
            arrayOf("folder_files._id"), null, null, "queue.sort",
        )).use { cursor -> buildList { while (cursor.moveToNext()) add(cursor.getLong(0)) } }
        try {
            val queueBefore = queuedFileIds()
            val countsBefore = FavoritesStore(app.filesDir).entries()
            val current = checkNotNull(PowerampReceiver.refreshCurrentTrackAfterActivityResume(app))
            instrumentation.runOnMainSync { vm.openNowPlayingFavorite(current) }
            await { !vm.favoriteMatches.value.loading && vm.favoriteMatches.value.seed != null }
            assertNull(vm.favoriteMatches.value.error)
            val trail = mutableListOf(vm.favoriteMatches.value)
            repeat(3) { hop ->
                val match = vm.favoriteMatches.value.matches.first { candidate ->
                    trail.none { it.resolvedSeedTrackId == candidate.track.id }
                }
                val started = SystemClock.elapsedRealtime()
                instrumentation.runOnMainSync { vm.openFavoriteMatch(match) }
                await { !vm.favoriteMatches.value.loading }
                val state = vm.favoriteMatches.value
                assertNull(state.error)
                assertEquals(match.track.id, state.resolvedSeedTrackId)
                assertEquals(match.stableTrackSpanId, state.resolvedSeedStableTrackSpanId)
                assertEquals(vm.favoriteResultCount.value, state.matches.size)
                assertFalse(state.matches.any { it.track.id == match.track.id })
                assertEquals(match.track.id, favoriteDisplayedQueue(state,
                    FavoriteMatchesSort.NEWEST_FIRST).tracks.first().id)
                trail.add(state)
                Log.i(TAG, "browseHop=${hop + 1} pass seed=${match.track.id} ms=${SystemClock.elapsedRealtime() - started}")
            }
            trail.dropLast(1).asReversed().forEach { previous ->
                instrumentation.runOnMainSync { vm.backFromFavorite() }
                await { !vm.favoriteMatches.value.loading }
                assertSame(previous, vm.favoriteMatches.value)
            }
            instrumentation.runOnMainSync { vm.backFromFavorite() }
            assertNull(vm.favoriteMatches.value.seed)
            assertEquals(queueBefore, queuedFileIds())
            assertEquals(countsBefore, FavoritesStore(app.filesDir).entries())
            Log.i(TAG, "browseTrail=pass three hops and back restore exact snapshots; queue and request counts unchanged")
        } finally {
            instrumentation.runOnMainSync { vm.closeFavorite(); viewModels.clear() }
        }
    }

    @Test
    fun nowPlayingLibrarySearchAndEveryDisplayedOrder() {
        val app = ApplicationProvider.getApplicationContext<Application>()
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val viewModels = ViewModelStore()
        lateinit var vm: MainViewModel
        instrumentation.runOnMainSync {
            vm = ViewModelProvider(viewModels, ViewModelProvider.AndroidViewModelFactory(app))[MainViewModel::class.java]
        }
        val temporary = File(app.cacheDir, "favorite-actions-${UUID.randomUUID()}").apply { mkdirs() }
        try {
            val current = checkNotNull(PowerampReceiver.refreshCurrentTrackAfterActivityResume(app)) {
                "Keep a Poweramp song selected, playing or paused, for this device check"
            }
            val started = SystemClock.elapsedRealtime()
            instrumentation.runOnMainSync { vm.openNowPlayingFavorite(current) }
            await { !vm.favoriteMatches.value.loading && vm.favoriteMatches.value.seed != null }
            val state = vm.favoriteMatches.value
            assertNull(state.error)
            assertNotNull(state.resolvedSeedTrack)
            val binding = checkNotNull(state.binding)
            val active = V2LibraryDatabaseResolver.requirePublished(app.filesDir)
            val catalog = checkNotNull(V2ActiveLibraryCatalogStore(app.filesDir).read(active))
            assertEquals(catalog.trackIdForPowerampFile(current.realId), state.resolvedSeedTrackId)
            assertEquals(vm.favoriteResultCount.value, state.matches.size)
            Log.i(TAG, "nowPlaying=pass paused=${!PowerampReceiver.isPlaying} count=${state.matches.size} ms=${SystemClock.elapsedRealtime() - started}")

            val database = EmbeddingDatabase.open(active.databaseFile)
            try {
                val index = EmbeddingIndex.mmap(active.embeddingFile)
                val identities = StableTrackIdentityCatalog.load(app.filesDir, database, index)
                val domain = ActiveRecommendationDomain.create(catalog, identities, index,
                    database.loadIndexAddedAtEpochSeconds(identities.orderedTrackIds()))
                val seedId = checkNotNull(state.resolvedSeedTrackId)
                val scores = index.computeAllSimilarities(checkNotNull(index.getEmbeddingByTrackId(seedId)))
                val candidates = domain.candidateDomain(null)
                val ids = candidates.orderedIdentityRepresentativeTrackIds()
                val identityScores = candidates.identityRepresentativeScoresFromFull(scores)
                val copies = identities.equivalentVisibleTrackIds(seedId).toSet()
                val expected = ids.indices.filter { ids[it] !in copies }.sortedWith(
                    compareByDescending<Int> { identityScores[it] }.thenBy { identities.rankingTieKey(ids[it]) }
                ).take(state.matches.size).map { ids[it] }
                assertEquals(expected, state.matches.map { it.track.id })
                val added = identities.equivalentVisibleTrackIds(state.matches.first().track.id).toSet()
                assertTrue(FavoritesRanking.hasNewNearest(index.getEmbeddingByTrackId(seedId)!!, seedId,
                    state.matches.size, index, domain, identities, added))
                val withNew = FavoritesRanking.nearest(index.getEmbeddingByTrackId(seedId)!!, seedId,
                    state.matches.size, index, domain, identities, database, added)
                assertTrue(withNew.first().isNew)
                assertEquals(expected, withNew.map { it.track.id })
                Log.i(TAG, "currentSongNew=pass simulated addition uses the same nearest list; active files untouched")

                val journal = RadioRequestStore(temporary)
                FavoriteMatchesSort.entries.forEach { sort ->
                    val plan = favoriteDisplayedQueue(state, sort)
                    val ordered = when (sort) {
                        FavoriteMatchesSort.CLOSEST_FIRST -> state.matches.sortedBy { it.rank }
                        FavoriteMatchesSort.FARTHEST_FIRST -> state.matches.sortedByDescending { it.rank }
                        FavoriteMatchesSort.NEWEST_FIRST -> state.matches.sortedWith(
                            compareByDescending<FavoriteMatch> { it.track.indexAddedAtEpochSecond }.thenBy { it.rank })
                        FavoriteMatchesSort.OLDEST_FIRST -> state.matches.sortedWith(
                            compareBy<FavoriteMatch> { it.track.indexAddedAtEpochSecond }.thenBy { it.rank })
                    }
                    assertEquals(listOf(seedId) + ordered.map { it.track.id }, plan.tracks.map { it.id })
                    val request = RadioService.buildPinnedDirectQueueRequest(app, plan.tracks, plan.identities,
                        binding.library, RadioService.Companion.DisplayedRecommendationDomain(
                            binding.providerGenerationId, binding.orderedActiveTrackIdsSha256, binding.activeTrackCount),
                        plan.evidence.seed.label, QueueOrigin.FAVORITE_RESULT_LIST, DirectQueuePlacement.REPLACE_UPCOMING,
                        null, null, plan.evidence)
                    journal.persist(request)
                    val recovered = (journal.claim(request.requestId) as RadioRequestClaim.Claimed).request.directQueue!!
                    assertEquals(plan.tracks, recovered.tracks)
                    assertEquals(plan.tracks.map { catalog.powerampFileIdForTrack(it.id) }, recovered.resolvedPowerampFileIds)
                    assertEquals(plan.evidence, recovered.favoriteQueueEvidence)
                    Log.i(TAG, "displayedOrder=$sort pass count=${plan.tracks.size} seedFirst=true exactPowerampIds=true durableRoundTrip=true")
                }
            } finally { database.close() }

            val favoriteIds = FavoritesStore(app.filesDir).entries().mapNotNull { it.seed.trackId }.toSet()
            instrumentation.runOnMainSync { vm.searchFavoriteSongs("bonobo") }
            await { vm.favoriteSongLookup.value !is RecordingLookupState.Loading }
            val search = vm.favoriteSongLookup.value as? RecordingLookupState.Success
                ?: error("Library lookup failed: ${vm.favoriteSongLookup.value}")
            val outsideFavorites = search.candidates.first { it.id !in favoriteIds }
            instrumentation.runOnMainSync { assertTrue(vm.openFavoriteSongSearchResult(outsideFavorites)) }
            await { !vm.favoriteMatches.value.loading }
            assertNull(vm.favoriteMatches.value.error)
            assertEquals(outsideFavorites.id, vm.favoriteMatches.value.resolvedSeedTrackId)
            Log.i(TAG, "librarySearch=pass candidates=${search.candidates.size} selectedNonFavorite=${outsideFavorites.id} closest=${vm.favoriteMatches.value.matches.size}")

            instrumentation.runOnMainSync {
                vm.searchFavoriteSongs("bonobo")
                vm.searchFavoriteSongs("rjd2")
                vm.searchFavoriteSongs("pink floyd")
            }
            await { vm.favoriteSongLookup.value !is RecordingLookupState.Loading }
            assertEquals("pink floyd", (vm.favoriteSongLookup.value as RecordingLookupState.Success).query)
            Log.i(TAG, "rapidQueries=pass latest request wins; playback and real queue untouched")
        } finally {
            instrumentation.runOnMainSync { vm.closeFavorite(); vm.dismissFavoriteSongSearch(); viewModels.clear() }
            temporary.deleteRecursively()
        }
    }

    private fun await(condition: () -> Boolean) {
        val deadline = SystemClock.elapsedRealtime() + 45_000L
        while (!condition()) {
            check(SystemClock.elapsedRealtime() < deadline) { "Timed out waiting for the Favorites workflow" }
            SystemClock.sleep(50L)
        }
    }

    companion object { private const val TAG = "FavoriteActionsCheck" }
}
