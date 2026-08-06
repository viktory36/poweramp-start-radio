package com.powerampstartradio.services

/**
 * Process-wide exclusive admission for operations that can create or replace music-index state.
 *
 * A library import or server merge owns the admission for its complete lifecycle. A new on-device
 * indexing request owns it only until both its immutable preflight intent and active-job pointer
 * are durable. The pointer then prevents later library mutations while the indexing job exists.
 */
internal class MusicIndexMutationAdmission {
    internal class Owner internal constructor(label: String) {
        init {
            require(label.isNotBlank()) { "Music-index mutation owner label must not be blank" }
        }
    }

    private var activeOwner: Owner? = null

    @Synchronized
    fun tryAcquire(owner: Owner): Boolean {
        if (activeOwner != null) return false
        activeOwner = owner
        return true
    }

    @Synchronized
    fun release(owner: Owner): Boolean {
        if (activeOwner !== owner) return false
        activeOwner = null
        return true
    }

    @Synchronized
    fun isOwnedBy(owner: Owner): Boolean = activeOwner === owner

    companion object {
        val process = MusicIndexMutationAdmission()

        fun newOwner(label: String): Owner = Owner(label)
    }
}

internal class MusicIndexMutationBusyException : IllegalStateException(
    "A music-index update is already running.",
)
