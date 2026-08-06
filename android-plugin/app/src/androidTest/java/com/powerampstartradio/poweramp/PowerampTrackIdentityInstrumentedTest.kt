package com.powerampstartradio.poweramp

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class PowerampTrackIdentityInstrumentedTest {
    @Test
    fun officialTrackIdentityAndQueueContextAreParsedSeparately() {
        val intent = Intent(PowerampHelper.ACTION_TRACK_CHANGED).putExtra(
            PowerampHelper.EXTRA_TRACK,
            Bundle().apply {
                putLong(PowerampHelper.TRACK_ID, 901L)
                putLong(PowerampHelper.TRACK_REAL_ID, 77L)
                putParcelable(
                    PowerampHelper.TRACK_CAT_URI,
                    Uri.parse("content://com.maxmpz.audioplayer.data/queue?shs=1"),
                )
                putInt(PowerampHelper.TRACK_POS_IN_LIST, 4)
                putString(PowerampHelper.TRACK_TITLE, "Duplicate recording")
                putInt(PowerampHelper.TRACK_DURATION, 120)
            },
        )

        val track = requireNotNull(PowerampHelper.getCurrentTrackFromIntent(intent))

        assertEquals(901L, track.trackId)
        assertEquals(77L, track.realId)
        assertEquals(4, track.positionInList)
        assertEquals(901L, track.queueOccurrenceId)
        assertEquals(120_000, track.durationMs)
    }
}
