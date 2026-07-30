package com.powerampstartradio.indexing

/** Canonical full-window plus >=1-second tail policy used by MERT and job planning. */
internal object MertWindowPolicy {
    const val SAMPLE_RATE = 24_000
    const val WINDOW_SAMPLES = 120_000
    const val MINIMUM_TAIL_SAMPLES = SAMPLE_RATE

    fun windowCount(sampleCount: Long): Int {
        require(sampleCount >= 0L) { "sampleCount must be non-negative" }
        val fullWindows = sampleCount / WINDOW_SAMPLES
        val tailSamples = sampleCount % WINDOW_SAMPLES
        val total = fullWindows + if (tailSamples >= MINIMUM_TAIL_SAMPLES) 1L else 0L
        require(total <= Int.MAX_VALUE) { "sampleCount has too many MERT windows" }
        return total.toInt()
    }
}
