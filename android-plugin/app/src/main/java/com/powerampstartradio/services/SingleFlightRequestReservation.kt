package com.powerampstartradio.services

/**
 * Owns the gap between accepting a user action and dispatching its durable request.
 *
 * A reservation starts before asynchronous request construction. Once persistence assigns a
 * request ID, only that request's dispatch (or a matching failure) can release it.
 */
internal class SingleFlightRequestReservation {
    private data class ActiveReservation(
        val submissionToken: String,
        val requestId: String? = null,
    )

    private var active: ActiveReservation? = null

    @Synchronized
    fun tryAcquire(submissionToken: String, requestId: String? = null): Boolean {
        require(submissionToken.isNotBlank()) { "Submission token must not be blank" }
        requestId?.let { require(it.isNotBlank()) { "Request ID must not be blank" } }
        if (active != null) return false
        active = ActiveReservation(submissionToken, requestId)
        return true
    }

    @Synchronized
    fun bindRequest(submissionToken: String, requestId: String): Boolean {
        require(requestId.isNotBlank()) { "Request ID must not be blank" }
        val current = active ?: return false
        if (current.submissionToken != submissionToken) return false
        if (current.requestId != null && current.requestId != requestId) return false
        active = current.copy(requestId = requestId)
        return true
    }

    /** Release after the matching request is synchronously dispatched into its active job. */
    @Synchronized
    fun completeDispatch(requestId: String): Boolean {
        val current = active ?: return false
        if (current.requestId != requestId) return false
        active = null
        return true
    }

    /** Release before dispatch when asynchronous submission itself fails. */
    @Synchronized
    fun failSubmission(submissionToken: String): Boolean {
        val current = active ?: return false
        if (current.submissionToken != submissionToken) return false
        active = null
        return true
    }

    /** Release a bound request that terminally fails before active-job dispatch. */
    @Synchronized
    fun failRequest(requestId: String): Boolean {
        val current = active ?: return false
        if (current.requestId != requestId) return false
        active = null
        return true
    }

    @get:Synchronized
    val isReserved: Boolean
        get() = active != null

    @get:Synchronized
    val activeRequestId: String?
        get() = active?.requestId
}

/** Widget BUSY state must be owned by a durable request that can eventually terminalize it. */
internal object WidgetBusyStatusOwnerPolicy {
    fun resolve(
        reservedRequestId: String?,
        activeDurableRequestId: String?,
    ): String? = reservedRequestId ?: activeDurableRequestId
}
