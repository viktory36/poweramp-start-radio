package com.powerampstartradio.ui

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class FindMusicCatalogHandoffPolicyTest {
    @Test
    fun `retains only when catalog replaces a missing cache with measured headroom`() {
        val mib = 1024L * 1024L
        assertTrue(
            FindMusicCatalogHandoffPolicy.shouldRetain(
                maxHeapBytes = 368L * mib,
                usedHeapBytes = 165L * mib,
                processSnapshotPresent = false,
            ),
        )
        assertFalse(
            FindMusicCatalogHandoffPolicy.shouldRetain(
                maxHeapBytes = 368L * mib,
                usedHeapBytes = 300L * mib,
                processSnapshotPresent = false,
            ),
        )
        assertFalse(
            FindMusicCatalogHandoffPolicy.shouldRetain(
                maxHeapBytes = 368L * mib,
                usedHeapBytes = 165L * mib,
                processSnapshotPresent = true,
            ),
        )
    }
}
