package com.powerampstartradio.indexing

import android.os.SystemClock
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.powerampstartradio.data.EmbeddingDatabase
import com.powerampstartradio.indexing.v2.V2LibraryDatabaseResolver
import com.powerampstartradio.indexing.v2.V2PowerampProviderSnapshotAcquirer
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith

/** Real-device proof that the generation-bound catalog is an exact detection optimization. */
@RunWith(AndroidJUnit4::class)
class NewTrackDetectorCatalogInstrumentedTest {
    @Test
    fun unchangedCatalogProducesTheSameUnindexedTracksAsFullReconciliation() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val active = V2LibraryDatabaseResolver.requirePublished(context.filesDir)
        val catalog = requireNotNull(
            V2ActiveLibraryCatalogStore(context.filesDir).read(active),
        ) { "active library catalog is absent" }
        val snapshot = V2PowerampProviderSnapshotAcquirer(context).acquireBlocking()
        assertEquals(
            catalog.generationBinding.providerGenerationId,
            snapshot.libraryGeneration,
        )

        val database = EmbeddingDatabase.open(active.databaseFile)
        try {
            val detector = NewTrackDetector(database)
            val fullStarted = SystemClock.elapsedRealtime()
            val full = detector.findUnindexedTracks(snapshot)
            val fullMs = SystemClock.elapsedRealtime() - fullStarted

            val catalogStarted = SystemClock.elapsedRealtime()
            val fromCatalog = detector.findUnindexedTracks(snapshot, catalog)
            val catalogMs = SystemClock.elapsedRealtime() - catalogStarted

            assertEquals(full, fromCatalog)
            Log.i(
                TAG,
                "rows=${snapshot.acquisitionEvidence?.rowCount} " +
                    "unindexed=${full.size} fullMs=$fullMs catalogMs=$catalogMs",
            )
        } finally {
            database.close()
        }
    }

    private companion object {
        const val TAG = "DetectionCatalogProof"
    }
}
