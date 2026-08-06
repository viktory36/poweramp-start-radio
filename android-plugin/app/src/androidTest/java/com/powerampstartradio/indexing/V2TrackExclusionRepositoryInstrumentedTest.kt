package com.powerampstartradio.indexing

import android.content.Context
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test
import org.junit.runner.RunWith
import java.util.UUID

@RunWith(AndroidJUnit4::class)
class V2TrackExclusionRepositoryInstrumentedTest {
    @Test
    fun corruptNeverIndexChoicesFailClosedWithoutOverwritingEvidence() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val preferencesName = "v2_track_exclusion_test_${UUID.randomUUID()}"
        val prefs = context.getSharedPreferences(
            preferencesName,
            Context.MODE_PRIVATE,
        )
        val neverKey = V2TrackExclusionRepository.NEVER_EXCLUSIONS_KEY
        val ignoredKey = V2TrackExclusionRepository.IGNORED_EXCLUSIONS_KEY
        val corrupt = "{not-valid-json"
        try {
            check(prefs.edit().putString(neverKey, corrupt).remove(ignoredKey).commit())

            assertThrows(V2TrackExclusionReadException::class.java) {
                V2TrackExclusionRepository(context, preferencesName)
                    .resolveAndMigrate(emptyList())
            }
            assertEquals(corrupt, prefs.getString(neverKey, null))
        } finally {
            check(prefs.edit().clear().commit())
            context.deleteSharedPreferences(preferencesName)
        }
    }
}
