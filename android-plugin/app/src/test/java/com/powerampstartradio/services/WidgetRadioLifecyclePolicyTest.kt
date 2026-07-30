package com.powerampstartradio.services

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class WidgetRadioLifecyclePolicyTest {
    @Test
    fun `failed cold reconciliation gives truthful terminal guidance`() {
        assertEquals(
            "Saved indexing work could not be verified. " +
                "Open On-device indexing, then restart the app after resolving it.",
            RecommendationAdmissionPresentation.unavailableMessage(
                coldOwnerReserved = true,
                coldState = RecommendationWorkAdmission.ColdReconciliationState.FAILED,
            ),
        )
    }

    @Test
    fun `running cold check and real indexing remain distinct`() {
        assertEquals(
            "Reading saved indexing-job ownership. Try again when that finishes.",
            RecommendationAdmissionPresentation.unavailableMessage(
                coldOwnerReserved = true,
                coldState = RecommendationWorkAdmission.ColdReconciliationState.RUNNING,
            ),
        )
        assertEquals(
            "On-device indexing is using the music model. " +
                "Try Start Radio after indexing finishes.",
            RecommendationAdmissionPresentation.unavailableMessage(
                coldOwnerReserved = false,
                coldState = RecommendationWorkAdmission.ColdReconciliationState.SUCCEEDED,
            ),
        )
    }

    @Test
    fun `durable request stopped for indexing publishes waiting state`() {
        assertTrue(
            WidgetDeferredStartPolicy.shouldPublishWaitingStatus(
                durableExecutionAction = true,
                explicitRequestId = "request-a",
                existingWorkMustDrain = false,
            ),
        )
        assertFalse(
            WidgetDeferredStartPolicy.shouldPublishWaitingStatus(
                durableExecutionAction = true,
                explicitRequestId = null,
                existingWorkMustDrain = false,
            ),
        )
        assertFalse(
            WidgetDeferredStartPolicy.shouldPublishWaitingStatus(
                durableExecutionAction = true,
                explicitRequestId = "request-a",
                existingWorkMustDrain = true,
            ),
        )
    }
}
