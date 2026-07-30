package com.powerampstartradio.indexing

import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.async
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Before
import org.junit.Test

class IndexingViewModelDetectionCacheTest {
    @Before
    fun clearBefore() {
        IndexingViewModel.invalidateCache()
    }

    @After
    fun clearAfter() {
        IndexingViewModel.invalidateCache()
    }

    @Test
    fun resultIsReusableOnlyForBothExactGenerations() {
        val tracks = listOf(
            NewTrackDetector.UnindexedTrack(
                powerampFileId = 42L,
                artist = "Artist",
                album = "Album",
                title = "Title",
                durationMs = 123_000,
                path = "/music/artist/title.flac",
            ),
        )
        IndexingViewModel.cacheResults(
            tracks = tracks,
            databaseGeneration = "database-a",
            providerGeneration = "provider-a",
        )

        val exact = IndexingViewModel.exactCachedResult("database-a", "provider-a")
        assertEquals(tracks, exact?.tracks)
        assertEquals("database-a", exact?.databaseGeneration)
        assertEquals("provider-a", exact?.providerGeneration)
        assertNull(IndexingViewModel.exactCachedResult("database-b", "provider-a"))
        assertNull(IndexingViewModel.exactCachedResult("database-a", "provider-b"))
        assertNull(IndexingViewModel.exactCachedResult("", "provider-a"))
        assertNull(IndexingViewModel.exactCachedResult("database-a", ""))
    }

    @Test
    fun pendingResultIsConsumedOnlyForTheCurrentDatabaseGeneration() {
        val tracks = listOf(
            NewTrackDetector.UnindexedTrack(
                powerampFileId = 42L,
                artist = "Artist",
                album = "Album",
                title = "Title",
                durationMs = 123_000,
                path = "/music/artist/title.flac",
            ),
        )
        val result = IndexingViewModel.SharedDetectionResult(
            tracks = tracks,
            databaseGeneration = "database-a",
            providerGeneration = "provider-a",
        )

        assertEquals(result, IndexingViewModel.matchingPendingResult("database-a", result))
        assertNull(IndexingViewModel.matchingPendingResult("database-b", result))
        assertNull(IndexingViewModel.matchingPendingResult("", result))
        assertNull(
            IndexingViewModel.matchingPendingResult(
                "database-a",
                result.copy(providerGeneration = ""),
            ),
        )
    }

    @Test
    fun completedSettingsResultCanBeConsumedOnceAfterPendingOwnerClears() {
        val result = sharedResult()
        IndexingViewModel.offerCompletedDetectionHandoff(
            result = result,
            completedAtElapsedMs = 10_000L,
        )
        IndexingViewModel.pendingDetection = null

        assertEquals(
            result,
            IndexingViewModel.consumeCompletedDetectionHandoff(
                databaseGeneration = "database-a",
                nowElapsedMs = 10_001L,
            ),
        )
        assertNull(
            IndexingViewModel.consumeCompletedDetectionHandoff(
                databaseGeneration = "database-a",
                nowElapsedMs = 10_002L,
            ),
        )
    }

    @Test
    fun completedSettingsHandoffExpiresAndRejectsAnotherDatabaseGeneration() {
        val result = sharedResult()
        IndexingViewModel.offerCompletedDetectionHandoff(
            result = result,
            completedAtElapsedMs = 10_000L,
        )
        assertNull(
            IndexingViewModel.consumeCompletedDetectionHandoff(
                databaseGeneration = "database-b",
                nowElapsedMs = 10_001L,
            ),
        )

        IndexingViewModel.offerCompletedDetectionHandoff(
            result = result,
            completedAtElapsedMs = 10_000L,
        )
        assertNull(
            IndexingViewModel.consumeCompletedDetectionHandoff(
                databaseGeneration = "database-a",
                nowElapsedMs = 10_001L +
                    IndexingViewModel.COMPLETED_DETECTION_HANDOFF_TTL_MS,
            ),
        )
    }

    @Test
    fun ownedDetectionIsPublishedToAnActiveSettingsSubscriber() = runBlocking {
        val result = sharedResult()
        val received = async(start = CoroutineStart.UNDISPATCHED) {
            IndexingViewModel.ownedDetectionResults.first()
        }

        IndexingViewModel.publishOwnedDetectionResult(result)

        assertEquals(result, received.await())
    }

    private fun sharedResult(): IndexingViewModel.SharedDetectionResult =
        IndexingViewModel.SharedDetectionResult(
            tracks = listOf(
                NewTrackDetector.UnindexedTrack(
                    powerampFileId = 42L,
                    artist = "Artist",
                    album = "Album",
                    title = "Title",
                    durationMs = 123_000,
                    path = "/music/artist/title.flac",
                ),
            ),
            databaseGeneration = "database-a",
            providerGeneration = "provider-a",
        )
}
