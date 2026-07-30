package com.powerampstartradio.ui

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class V2DatabaseRefreshRequestGateTest {
    @Test
    fun ordinaryRequestJoinsRunningPassWithoutQueuingAnotherPass() {
        val gate = V2DatabaseRefreshRequestGate()

        assertEquals(V2DatabaseRefreshAdmission.START, gate.admit(resourcesBlocked = false))
        assertEquals(
            V2DatabaseRefreshAdmission.JOIN_RUNNING,
            gate.admit(resourcesBlocked = false),
        )
        assertFalse(gate.completeRunningRequest())

        assertEquals(V2DatabaseRefreshAdmission.START, gate.admit(resourcesBlocked = false))
    }

    @Test
    fun blockedRequestSurvivesUntilResourcesAreFree() {
        val gate = V2DatabaseRefreshRequestGate()

        assertEquals(
            V2DatabaseRefreshAdmission.DEFER_UNTIL_RESOURCES_ARE_FREE,
            gate.admit(resourcesBlocked = true),
        )
        assertTrue(gate.hasDeferredRequest())
        assertEquals(V2DatabaseRefreshAdmission.START, gate.admit(resourcesBlocked = false))
        assertFalse(gate.hasDeferredRequest())
        assertFalse(gate.completeRunningRequest())
    }

    @Test
    fun blockedRequestDuringRunningPassRequiresLaterAdmission() {
        val gate = V2DatabaseRefreshRequestGate()

        assertEquals(V2DatabaseRefreshAdmission.START, gate.admit(resourcesBlocked = false))
        assertEquals(
            V2DatabaseRefreshAdmission.DEFER_UNTIL_RESOURCES_ARE_FREE,
            gate.admit(resourcesBlocked = true),
        )
        assertTrue(gate.completeRunningRequest())
        assertTrue(gate.hasDeferredRequest())
        assertEquals(V2DatabaseRefreshAdmission.START, gate.admit(resourcesBlocked = false))
    }

    @Test
    fun generationMismatchCanDeferTheOwnedPassUntilIndexingReleasesResources() {
        val gate = V2DatabaseRefreshRequestGate()

        assertEquals(V2DatabaseRefreshAdmission.START, gate.admit(resourcesBlocked = false))
        gate.deferUntilResourcesAreFree()
        assertTrue(gate.completeRunningRequest())
        assertEquals(
            V2DatabaseRefreshAdmission.DEFER_UNTIL_RESOURCES_ARE_FREE,
            gate.admit(resourcesBlocked = true),
        )
        assertEquals(V2DatabaseRefreshAdmission.START, gate.admit(resourcesBlocked = false))
    }

    @Test
    fun generationTransitionsRequireAnotherPassButIdenticalGenerationDoesNot() {
        assertFalse(
            V2DatabaseRefreshGenerationPolicy.needsAnotherPass(
                "generation-a",
                "generation-a",
            ),
        )
        assertFalse(V2DatabaseRefreshGenerationPolicy.needsAnotherPass(null, null))
        assertTrue(
            V2DatabaseRefreshGenerationPolicy.needsAnotherPass(
                "generation-a",
                "generation-b",
            ),
        )
        assertTrue(V2DatabaseRefreshGenerationPolicy.needsAnotherPass("generation-a", null))
        assertTrue(V2DatabaseRefreshGenerationPolicy.needsAnotherPass(null, "generation-a"))
    }

    @Test
    fun sameGenerationResumeDoesNotRequestAProviderRefresh() {
        assertFalse(
            V2DatabaseReadySnapshotPolicy.shouldRefreshOnResume(
                publishedGeneration = "generation-a",
                activeGeneration = "generation-a",
            ),
        )
        assertTrue(
            V2DatabaseReadySnapshotPolicy.shouldRefreshOnResume(
                publishedGeneration = null,
                activeGeneration = "generation-a",
            ),
        )
        assertTrue(
            V2DatabaseReadySnapshotPolicy.shouldRefreshOnResume(
                publishedGeneration = "generation-a",
                activeGeneration = "generation-b",
            ),
        )
    }

    @Test
    fun onlyAnExactlyBoundCurrentSnapshotCanStayReadyDuringReconciliation() {
        assertTrue(
            V2DatabaseReadySnapshotPolicy.isCurrent(
                publishedGeneration = "generation-a",
                activeGeneration = "generation-a",
                hasExactLibraryBinding = true,
            ),
        )
        assertFalse(
            V2DatabaseReadySnapshotPolicy.isCurrent(
                publishedGeneration = "generation-a",
                activeGeneration = "generation-a",
                hasExactLibraryBinding = false,
            ),
        )
        assertFalse(
            V2DatabaseReadySnapshotPolicy.isCurrent(
                publishedGeneration = "generation-a",
                activeGeneration = "generation-b",
                hasExactLibraryBinding = true,
            ),
        )
    }

    @Test
    fun processSnapshotBootstrapRequiresTheActiveImmutableGeneration() {
        assertTrue(
            V2DatabaseReadySnapshotPolicy.canBootstrap(
                cachedDatabaseGeneration = "generation-a",
                activeGeneration = "generation-a",
            ),
        )
        assertFalse(
            V2DatabaseReadySnapshotPolicy.canBootstrap(
                cachedDatabaseGeneration = "generation-a",
                activeGeneration = "generation-b",
            ),
        )
        assertFalse(
            V2DatabaseReadySnapshotPolicy.canBootstrap(
                cachedDatabaseGeneration = null,
                activeGeneration = "generation-a",
            ),
        )
    }
}
