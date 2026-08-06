package com.powerampstartradio.indexing

import kotlinx.coroutines.sync.Mutex

internal fun interface V2LibraryInspectionHeapReclaimer {
    fun reclaim()
}

internal fun <T> V2LibraryInspectionHeapReclaimer.reclaimAfterInspection(
    block: () -> T,
): T = try {
    block()
} finally {
    reclaim()
}

/**
 * Serializes complete provider/database reconciliations inside this process.
 *
 * Each inspection temporarily materializes the full Poweramp library and much of the embedding
 * catalog. Running two at once can exceed Android's app heap even though either inspection fits by
 * itself. Reclaim after the compact result has been retained but before the mutex is released, so
 * the next inspection or audio-model load cannot inherit the previous pass's unreachable heap.
 * Waiting is cancellation-aware so an obsolete Activity/ViewModel does not run later.
 */
internal class V2LibraryInspectionCoordinator(
    private val heapReclaimer: V2LibraryInspectionHeapReclaimer =
        V2LibraryInspectionHeapReclaimer(System::gc),
) {
    private val mutex = Mutex()

    suspend fun <T> inspect(block: suspend () -> T): T {
        mutex.lock()
        return try {
            block()
        } finally {
            try {
                heapReclaimer.reclaim()
            } finally {
                mutex.unlock()
            }
        }
    }
}

/** Shared by the Main and Manage Tracks ViewModels in the same app process. */
internal object V2ProcessLibraryInspectionCoordinator {
    private val coordinator = V2LibraryInspectionCoordinator()

    suspend fun <T> inspect(block: suspend () -> T): T = coordinator.inspect(block)
}
