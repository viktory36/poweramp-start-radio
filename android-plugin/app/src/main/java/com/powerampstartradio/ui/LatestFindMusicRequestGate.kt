package com.powerampstartradio.ui

import java.util.concurrent.atomic.AtomicLong

/** Monotonic publication gate: only the newest request may change visible results. */
internal class LatestFindMusicRequestGate {
    private val revision = AtomicLong(0L)

    fun begin(): Long = revision.incrementAndGet()

    fun cancel(): Long = revision.incrementAndGet()

    fun isCurrent(candidate: Long): Boolean = revision.get() == candidate
}
