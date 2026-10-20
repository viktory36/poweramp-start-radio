package com.powerampstartradio.services

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.powerampstartradio.data.EmbeddingDatabase
import com.powerampstartradio.indexing.NewTrackDetector
import com.powerampstartradio.indexing.V2ActiveLibraryCatalogLoader
import com.powerampstartradio.indexing.v2.V2LibraryDatabaseResolver
import com.powerampstartradio.indexing.v2.V2AudioSpanResolver
import com.powerampstartradio.indexing.v2.V2MediaExtractorAudioInspector
import com.powerampstartradio.indexing.v2.V2PowerampProviderSnapshotAcquirer
import com.powerampstartradio.poweramp.PowerampReceiver
import com.powerampstartradio.ui.QueueOrigin
import com.powerampstartradio.ui.RadioConfig
import com.powerampstartradio.widget.WidgetRadioSeedReference
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/** Reads the actual seed, catalog and source; never enqueues, indexes or edits exclusions. */
@RunWith(AndroidJUnit4::class)
class UnindexedRadioSeedInstrumentedTest {
    @Test
    fun currentUnindexedSeedHasNoEmbeddingButItsOriginalSourceOpens() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val current = requireNotNull(PowerampReceiver.requireProviderVerifiedCurrentTrack(context)) {
            "Poweramp has no current track for this diagnostic"
        }
        val snapshot = V2PowerampProviderSnapshotAcquirer(context).acquireBlocking()
        val row = snapshot.groups.flatMap { it.rows }.single {
            it.powerampFileId == current.realId
        }
        val active = V2LibraryDatabaseResolver.requirePublished(context.filesDir)
        val catalog = V2ActiveLibraryCatalogLoader.load(active, snapshot)
        assertEquals(null, catalog.trackIdForPowerampFile(current.realId))

        val database = EmbeddingDatabase.open(active.databaseFile)
        try {
            val full = NewTrackDetector(database).findUnindexedTracks(snapshot)
            val cached = NewTrackDetector(database).findUnindexedTracks(snapshot, catalog)
            assertEquals(full, cached)
            val candidate = full.single { it.powerampFileId == current.realId }
            assertEquals(row.physicalPath, candidate.path)
            val rawSource = File(row.providerPhysicalPath)
            assertTrue(rawSource.isFile && rawSource.canRead())
            assertFalse(row.providerPhysicalPath == row.physicalPath)
            assertFalse(File(row.physicalPath).exists())
            val spans = V2AudioSpanResolver(V2MediaExtractorAudioInspector())
                .resolve(listOf(candidate), snapshot)
            assertTrue(spans.rejected.isEmpty())
            val container = spans.resolved.single().containerEvidence
            assertEquals(row.providerPhysicalPath, container.physicalPath)
            assertTrue(container.durationUsEstimate > 0L)
            Log.i(
                "UnindexedSeedCheck",
                "fileId=${row.powerampFileId} absentFromIndex=true detectedAsUnindexed=true " +
                    "rawSourceReadable=true normalizedSourceExists=false " +
                    "mime=${container.mime} sampleRate=${container.sampleRateHz} " +
                    "durationUs=${container.durationUsEstimate}",
            )
        } finally {
            database.close()
        }

        var failure: RadioSeedNotIndexedException? = null
        try {
            RadioService.buildPinnedRadioRequest(
                context = context,
                config = RadioConfig(),
                currentSeedReference = WidgetRadioSeedReference.from(current),
                showToasts = false,
                origin = QueueOrigin.DEBUG_RADIO,
            )
        } catch (expected: RadioSeedNotIndexedException) {
            failure = expected
        }
        assertNotNull(failure)
        val message = RadioFailureMessage.from(requireNotNull(failure))
        assertTrue(message.contains("No indexed match"))
        Log.i("UnindexedSeedCheck", "Production radio rejection: $message")
    }
}
