package com.powerampstartradio.services

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * Process-wide ownership boundary between recommendation work and on-device indexing.
 *
 * Indexing reserves first, then drains already-admitted recommendation work. New work must use
 * [runIfRecommendationAllowed] for its irreversible admission step so it cannot race that
 * reservation. The atomic is the synchronous source of truth; the flow is for UI/lifecycle
 * observation.
 */
internal object RecommendationWorkAdmission {
    private val admissionLock = Any()
    private val reservationOwners = linkedSetOf<ReservationOwner>()
    @Volatile private var indexingReservedSnapshot = false
    private val _indexingReserved = MutableStateFlow(false)
    private val _coldReconciliationState =
        MutableStateFlow(ColdReconciliationState.RUNNING)

    enum class ColdReconciliationState {
        RUNNING,
        SUCCEEDED,
        FAILED,
    }

    data class ReservationOwner internal constructor(internal val key: String)

    val coldReconciliationOwner = ReservationOwner("cold-reconciliation")

    fun uiHandoffOwner(identity: String): ReservationOwner =
        ReservationOwner("ui-handoff:$identity")

    fun musicIndexMutationOwner(identity: String): ReservationOwner =
        ReservationOwner("music-index-mutation:$identity")

    fun indexingLaunchOwner(jobId: String): ReservationOwner =
        ReservationOwner("indexing-launch:$jobId")

    fun indexingServiceOwner(identity: String): ReservationOwner =
        ReservationOwner("indexing-service:$identity")

    val indexingReserved: StateFlow<Boolean> = _indexingReserved.asStateFlow()
    val coldReconciliationState: StateFlow<ColdReconciliationState> =
        _coldReconciliationState.asStateFlow()

    val isIndexingReserved: Boolean
        get() = indexingReservedSnapshot

    /** Begin the one cold-start ownership check for this Application process. */
    fun beginColdReconciliation() {
        synchronized(admissionLock) {
            _coldReconciliationState.value = ColdReconciliationState.RUNNING
            reservationOwners += coldReconciliationOwner
            publishLocked()
        }
    }

    /**
     * Settle the cold-start check and wake any foreground service preserving a widget tap.
     * Failure retains the reservation so recommendation work remains fail-closed.
     *
     * Returns true only when successful settlement made recommendation work available.
     */
    fun finishColdReconciliation(success: Boolean): Boolean {
        synchronized(admissionLock) {
            if (_coldReconciliationState.value != ColdReconciliationState.RUNNING) return false
            val wasReserved = reservationOwners.isNotEmpty()
            if (success) reservationOwners -= coldReconciliationOwner
            publishLocked()
            _coldReconciliationState.value = if (success) {
                ColdReconciliationState.SUCCEEDED
            } else {
                ColdReconciliationState.FAILED
            }
            return success && wasReserved && reservationOwners.isEmpty()
        }
    }

    /** Adds one independent reason to reject new recommendation work. */
    fun reserve(owner: ReservationOwner): Boolean {
        synchronized(admissionLock) {
            val wasReserved = reservationOwners.isNotEmpty()
            reservationOwners += owner
            publishLocked()
            return !wasReserved && reservationOwners.isNotEmpty()
        }
    }

    /** Releases only the caller's reason. Returns true when recommendation work became available. */
    fun release(owner: ReservationOwner): Boolean {
        synchronized(admissionLock) {
            val wasReserved = reservationOwners.isNotEmpty()
            reservationOwners -= owner
            publishLocked()
            return wasReserved && reservationOwners.isEmpty()
        }
    }

    fun isReservedBy(owner: ReservationOwner): Boolean = synchronized(admissionLock) {
        owner in reservationOwners
    }

    /**
     * Persist a bounded cold widget command without admitting recommendation work. The caller may
     * run only while startup reconciliation is the sole reservation owner; any real indexing
     * owner keeps the command outside the process boundary.
     */
    fun <T : Any> runIfOnlyColdReconciliationReserved(block: () -> T): T? =
        synchronized(admissionLock) {
            if (_coldReconciliationState.value == ColdReconciliationState.RUNNING &&
                reservationOwners.size == 1 && coldReconciliationOwner in reservationOwners
            ) {
                block()
            } else {
                null
            }
        }

    private fun publishLocked() {
        val reserved = reservationOwners.isNotEmpty()
        indexingReservedSnapshot = reserved
        _indexingReserved.value = reserved
    }

    /**
     * Run one bounded admission/claim step only when indexing has not reserved the process.
     * A null return means indexing won the boundary before [block] began.
     */
    fun <T : Any> runIfRecommendationAllowed(block: () -> T): T? =
        synchronized(admissionLock) {
            if (reservationOwners.isNotEmpty()) null else block()
        }
}
