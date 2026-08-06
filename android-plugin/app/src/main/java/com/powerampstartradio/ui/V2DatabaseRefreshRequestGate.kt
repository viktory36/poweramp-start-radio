package com.powerampstartradio.ui

import java.util.concurrent.atomic.AtomicBoolean

internal enum class V2DatabaseRefreshAdmission {
    START,
    JOIN_RUNNING,
    DEFER_UNTIL_RESOURCES_ARE_FREE,
}

/**
 * Single-flight admission for the full active-library refresh.
 *
 * Ordinary lifecycle requests join a running pass. A request made while indexing owns the shared
 * recommendation resources is different: it is retained until those resources are free because
 * indexing may publish a new immutable database generation before releasing them.
 */
internal class V2DatabaseRefreshRequestGate {
    private val running = AtomicBoolean(false)
    private val deferred = AtomicBoolean(false)

    fun admit(resourcesBlocked: Boolean): V2DatabaseRefreshAdmission {
        if (resourcesBlocked) {
            deferred.set(true)
            return V2DatabaseRefreshAdmission.DEFER_UNTIL_RESOURCES_ARE_FREE
        }
        if (!running.compareAndSet(false, true)) {
            return V2DatabaseRefreshAdmission.JOIN_RUNNING
        }
        deferred.set(false)
        return V2DatabaseRefreshAdmission.START
    }

    /** Releases single-flight ownership and reports whether blocked work still needs admission. */
    fun completeRunningRequest(): Boolean {
        running.set(false)
        return deferred.get()
    }

    fun deferUntilResourcesAreFree() {
        deferred.set(true)
    }

    fun hasDeferredRequest(): Boolean = deferred.get()
}

/** A successful pass is current only while its immutable generation remains the active pointer. */
internal object V2DatabaseRefreshGenerationPolicy {
    fun needsAnotherPass(
        validatedGeneration: String?,
        activeGeneration: String?,
    ): Boolean = validatedGeneration != activeGeneration
}

/** Keeps a verified current generation usable while its provider catalog is reconciled. */
internal object V2DatabaseReadySnapshotPolicy {
    fun isCurrent(
        publishedGeneration: String?,
        activeGeneration: String?,
        hasExactLibraryBinding: Boolean,
    ): Boolean = publishedGeneration != null &&
        publishedGeneration == activeGeneration &&
        hasExactLibraryBinding

    fun shouldRefreshOnResume(
        publishedGeneration: String?,
        activeGeneration: String?,
    ): Boolean = publishedGeneration != activeGeneration

    fun canBootstrap(
        cachedDatabaseGeneration: String?,
        activeGeneration: String?,
    ): Boolean = cachedDatabaseGeneration != null &&
        cachedDatabaseGeneration == activeGeneration
}
