package com.powerampstartradio.services

/**
 * Result of the synchronous boundary before asynchronous request construction begins.
 *
 * [ACCEPTED] means this process owns the single-flight reservation and scheduled the request
 * builder. It does not claim that later library revalidation or Poweramp mutation succeeded.
 */
internal enum class RadioRequestAdmission {
    ACCEPTED,
    BUSY,
}

/** Keeps user-visible admission aligned with ownership of the single-flight reservation. */
internal object RadioRequestAdmissionCoordinator {
    fun admit(
        reserve: () -> String?,
        release: (String) -> Unit,
        schedule: (String) -> Unit,
    ): RadioRequestAdmission {
        val submissionToken = reserve() ?: return RadioRequestAdmission.BUSY
        try {
            schedule(submissionToken)
        } catch (failure: Throwable) {
            release(submissionToken)
            throw failure
        }
        return RadioRequestAdmission.ACCEPTED
    }
}
