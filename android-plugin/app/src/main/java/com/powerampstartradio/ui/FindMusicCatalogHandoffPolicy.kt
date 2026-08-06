package com.powerampstartradio.ui

/** Keeps one catalog only when it replaces, rather than overlaps, a complete Find Music cache. */
internal object FindMusicCatalogHandoffPolicy {
    const val MIN_HEADROOM_BYTES = 96L * 1024L * 1024L

    fun shouldRetain(
        maxHeapBytes: Long,
        usedHeapBytes: Long,
        processSnapshotPresent: Boolean,
    ): Boolean = !processSnapshotPresent &&
        maxHeapBytes >= usedHeapBytes &&
        maxHeapBytes - usedHeapBytes >= MIN_HEADROOM_BYTES
}
