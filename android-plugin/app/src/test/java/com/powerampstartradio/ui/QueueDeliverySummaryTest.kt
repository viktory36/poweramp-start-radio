package com.powerampstartradio.ui

import com.google.gson.Gson
import com.google.gson.JsonParser
import com.powerampstartradio.data.EmbeddedTrack
import com.powerampstartradio.poweramp.PowerampTrack
import com.powerampstartradio.poweramp.TrackMatcher
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Test

class QueueDeliverySummaryTest {
    @Test
    fun summarySeparatesResolutionFailureFromQueueFailure() {
        val tracks = listOf(
            result(1L, QueueStatus.QUEUED),
            result(2L, QueueStatus.QUEUE_FAILED),
            result(3L, QueueStatus.NOT_IN_LIBRARY),
        )

        val summary = QueueDeliverySummary.fromTracks(
            origin = QueueOrigin.COMPOSED_RESULT_LIST,
            requestedCount = 5,
            rankedCount = 3,
            resolvedCount = 2,
            tracks = tracks,
            verificationComplete = true,
        )

        assertEquals(5, summary.requestedCount)
        assertEquals(3, summary.rankedCount)
        assertEquals(2, summary.resolvedCount)
        assertEquals(1, summary.verifiedCount)
        assertEquals(1, summary.queueFailedCount)
        assertEquals(1, summary.notInLibraryCount)
        assertEquals(QueueOrigin.COMPOSED_RESULT_LIST, summary.origin)
    }

    @Test
    fun radioResultUsesPersistedDeliveryFactsInsteadOfOptimisticRows() {
        val optimisticLegacyRows = listOf(
            result(1L, QueueStatus.QUEUED),
            result(2L, QueueStatus.QUEUED),
        )
        val delivery = QueueDeliverySummary(
            origin = QueueOrigin.WIDGET_RADIO,
            requestedCount = 50,
            rankedCount = 44,
            resolvedCount = 40,
            verifiedCount = 37,
            notInLibraryCount = 4,
            queueFailedCount = 3,
            verificationComplete = false,
        )
        val result = RadioResult(
            seedTrack = PowerampTrack(10L, "Seed", "Artist", null, 1000, null),
            matchType = TrackMatcher.MatchType.PATH_EXACT,
            tracks = optimisticLegacyRows,
            delivery = delivery,
        )

        assertEquals(QueueOrigin.WIDGET_RADIO, result.origin)
        assertEquals(50, result.requestedCount)
        assertEquals(44, result.rankedCount)
        assertEquals(40, result.resolvedCount)
        assertEquals(37, result.queuedCount)
        assertEquals(3, result.queueFailedCount)
        assertEquals(4, result.notInLibraryCount)
        assertFalse(result.delivery!!.verificationComplete)
    }

    @Test
    fun historyRoundTripPreservesOriginAndLoadsLegacyRowsWithoutDelivery() {
        val result = RadioResult(
            seedTrack = PowerampTrack(10L, "Seed", "Artist", null, 1000, null),
            matchType = TrackMatcher.MatchType.PATH_EXACT,
            tracks = listOf(result(1L, QueueStatus.QUEUED)),
            delivery = QueueDeliverySummary(
                origin = QueueOrigin.TEXT_RESULT_LIST,
                requestedCount = 3,
                rankedCount = 3,
                resolvedCount = 2,
                verifiedCount = 1,
                notInLibraryCount = 1,
                queueFailedCount = 1,
                verificationComplete = true,
            ),
        )
        val gson = Gson()

        val restored = gson.fromJson(gson.toJson(result), RadioResult::class.java)
        assertEquals(QueueOrigin.TEXT_RESULT_LIST, restored.origin)
        assertEquals(3, restored.requestedCount)
        assertEquals(1, restored.queuedCount)

        val legacyJson = JsonParser.parseString(gson.toJson(result)).asJsonObject.apply {
            remove("delivery")
        }
        val legacy = gson.fromJson(legacyJson, RadioResult::class.java)
        assertEquals(null, legacy.delivery)
        assertEquals(QueueOrigin.LEGACY_UNKNOWN, legacy.origin)
        assertEquals(legacy.tracks.size, legacy.requestedCount)
    }

    private fun result(id: Long, status: QueueStatus): QueuedTrackResult {
        return QueuedTrackResult(
            track = EmbeddedTrack(
                id = id,
                metadataKey = "key-$id",
                filenameKey = "file-$id",
                artist = "Artist",
                album = "Album",
                title = "Track $id",
                durationMs = 1000,
                filePath = "/music/$id.flac",
            ),
            similarity = 0.5f,
            similarityToSeed = 0.5f,
            status = status,
        )
    }
}
