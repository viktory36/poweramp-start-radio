package com.powerampstartradio.indexing.v2

/**
 * Coalesces process-local progress presentation without changing processing or checkpoint cadence.
 * Stage boundaries are always observable, even when they fall inside the intermediate interval.
 */
internal class V2IndexingProgressEventCadence(
    private val intermediateIntervalMs: Long,
) {
    private var lastEmittedAtElapsedMs: Long? = null

    init {
        require(intermediateIntervalMs > 0L) { "intermediate interval must be positive" }
    }

    fun shouldEmit(
        completedUnits: Long,
        totalUnits: Long,
        observedAtElapsedMs: Long,
    ): Boolean {
        require(totalUnits > 0L) { "total units must be positive" }
        require(completedUnits in 0L..totalUnits) { "completed units are outside the stage" }
        require(observedAtElapsedMs >= 0L) { "elapsed time must not be negative" }

        val previous = lastEmittedAtElapsedMs
        val boundary = completedUnits == 0L || completedUnits == totalUnits
        val due = previous == null || observedAtElapsedMs - previous >= intermediateIntervalMs
        return (boundary || due).also { emit ->
            if (emit) lastEmittedAtElapsedMs = observedAtElapsedMs
        }
    }
}
