package com.powerampstartradio.indexing

import com.powerampstartradio.indexing.v2.IndexingJobState
import com.powerampstartradio.indexing.v2.V2ActiveIndexingJobClaimPolicy
import com.powerampstartradio.indexing.v2.V2IndexingEtaScope
import com.powerampstartradio.indexing.v2.V2IndexingOverallWorkSnapshot
import com.powerampstartradio.indexing.v2.V2IndexingPreflightIntentState
import com.powerampstartradio.indexing.v2.V2IndexingPreflightIntentFactory
import com.powerampstartradio.indexing.v2.V2IndexingPreflightIntentStateMachine
import com.powerampstartradio.indexing.v2.V2IndexingPreflightPhase
import com.powerampstartradio.indexing.v2.V2IndexingPreflightProgress
import com.powerampstartradio.indexing.v2.V2IndexingPreflightSelection
import com.powerampstartradio.indexing.v2.V2IndexingExecutionProfile
import com.powerampstartradio.indexing.v2.V2UnmeasuredIndexingWork
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assert.assertThrows
import org.junit.Test

class V2IndexingServicePoliciesTest {
    @Test
    fun `one durable indexing slot rejects a different nonterminal job`() {
        assertTrue(V2ActiveIndexingJobClaimPolicy.canClaim("job-a", null, null))
        assertTrue(
            V2ActiveIndexingJobClaimPolicy.canClaim(
                "job-a",
                "job-a",
                IndexingJobState.RUNNING,
            ),
        )
        assertFalse(
            V2ActiveIndexingJobClaimPolicy.canClaim(
                "job-b",
                "job-a",
                IndexingJobState.PAUSED,
            ),
        )
        assertTrue(
            V2ActiveIndexingJobClaimPolicy.canClaim(
                "job-b",
                "job-a",
                IndexingJobState.COMPLETE,
            ),
        )
        assertFalse(
            V2ActiveIndexingJobClaimPolicy.canClaim(
                "job-b",
                "missing-job",
                null,
            ),
        )
    }

    @Test
    fun `omitted physical work makes ETA explicitly phase limited`() {
        val work = V2IndexingOverallWorkSnapshot(
            resolvedTracks = 0,
            totalTracks = 1,
            stages = emptyList(),
            omittedRemainingWork = setOf(
                V2UnmeasuredIndexingWork.VALIDATION_AND_FINAL_PUBLICATION,
            ),
        )

        assertEquals(V2IndexingEtaScope.MEASURED_STAGES_ONLY, work.etaCoverage.scope)
        assertFalse(work.etaCoverage.coversWholeJob)
        assertEquals(
            setOf(V2UnmeasuredIndexingWork.VALIDATION_AND_FINAL_PUBLICATION),
            work.etaCoverage.omittedRemainingWork,
        )
    }

    @Test
    fun `background start restrictions defer recovery without hiding programming errors`() {
        assertTrue(V2IndexingRecoveryStartPolicy.mayDefer(IllegalStateException("background")))
        assertTrue(V2IndexingRecoveryStartPolicy.mayDefer(SecurityException("denied")))
        assertFalse(V2IndexingRecoveryStartPolicy.mayDefer(IllegalArgumentException("bad intent")))
    }

    @Test
    fun `retained terminal pointer restarts service to finish private cleanup`() {
        assertTrue(V2IndexingRecoveryServicePolicy.shouldStart(IndexingJobState.COMPLETE))
        assertTrue(V2IndexingRecoveryServicePolicy.shouldStart(IndexingJobState.CANCELLED))
        assertTrue(V2IndexingRecoveryServicePolicy.shouldStart(IndexingJobState.INTERRUPTED))
        assertFalse(V2IndexingRecoveryServicePolicy.shouldStart(IndexingJobState.PAUSED))
        assertFalse(V2IndexingRecoveryServicePolicy.shouldStart(IndexingJobState.WAITING_FOR_INPUT))
    }

    @Test
    fun `complete cleanup crash preserves ownership and later recovery settles terminal job`() {
        val ledgerState = IndexingJobState.COMPLETE
        var pointerPresent = true
        var foregroundRunning = true
        var cleanupAttempts = 0
        var listenerMessage: String? = null
        var terminalPublished = false

        val first = V2IndexingTerminalCleanupRecoveryPolicy.settle(
            state = ledgerState,
            cleanup = {
                cleanupAttempts++
                throw java.io.IOException("simulated residue")
            },
            onRetryRequired = { _, message ->
                listenerMessage = message
                foregroundRunning = false
            },
            onClean = {
                pointerPresent = false
                terminalPublished = true
                foregroundRunning = false
            },
        )

        assertFalse(first)
        assertEquals(IndexingJobState.COMPLETE, ledgerState)
        assertTrue(pointerPresent)
        assertFalse(foregroundRunning)
        assertFalse(terminalPublished)
        assertTrue(listenerMessage.orEmpty().contains("music index is ready"))
        assertTrue(listenerMessage.orEmpty().contains("retry cleanup"))

        foregroundRunning = true
        val recovered = V2IndexingTerminalCleanupRecoveryPolicy.settle(
            state = ledgerState,
            cleanup = { cleanupAttempts++ },
            onRetryRequired = { _, _ -> throw AssertionError("recovery should be clean") },
            onClean = {
                pointerPresent = false
                terminalPublished = true
                foregroundRunning = false
            },
        )

        assertTrue(recovered)
        assertEquals(2, cleanupAttempts)
        assertEquals(IndexingJobState.COMPLETE, ledgerState)
        assertFalse(pointerPresent)
        assertFalse(foregroundRunning)
        assertTrue(terminalPublished)
    }

    @Test
    fun `cancelled terminal cleanup preserves cancellation semantics`() {
        var pointerPresent = true
        var terminalPublished = false

        assertTrue(
            V2IndexingTerminalCleanupRecoveryPolicy.settle(
                state = IndexingJobState.CANCELLED,
                cleanup = {},
                onRetryRequired = { _, _ -> throw AssertionError("cleanup should succeed") },
                onClean = {
                    pointerPresent = false
                    terminalPublished = true
                },
            ),
        )
        assertFalse(pointerPresent)
        assertTrue(terminalPublished)
    }

    @Test
    fun `resume arriving while old runner drains is replayed after lease release`() {
        assertTrue(V2IndexingRunnerDrainPolicy.shouldReplayStart(IndexingJobState.PAUSE_REQUESTED))
        assertTrue(V2IndexingRunnerDrainPolicy.shouldReplayStart(IndexingJobState.PAUSED))
        assertTrue(V2IndexingRunnerDrainPolicy.shouldReplayStart(IndexingJobState.INTERRUPTED))
        assertFalse(V2IndexingRunnerDrainPolicy.shouldReplayStart(IndexingJobState.RUNNING))
        assertFalse(V2IndexingRunnerDrainPolicy.shouldReplayStart(IndexingJobState.COMPLETE))
    }

    @Test
    fun `older runner cannot stop foreground service owned by newer command`() {
        assertFalse(
            V2IndexingServiceStartOwnershipPolicy.mayStopForeground(
                finishingStartId = 41,
                latestStartId = 42,
            ),
        )
        assertTrue(
            V2IndexingServiceStartOwnershipPolicy.mayStopForeground(
                finishingStartId = 42,
                latestStartId = 42,
            ),
        )
    }

    @Test
    fun `media processing timeout bypasses stale start ownership`() {
        assertFalse(
            V2IndexingServiceStartOwnershipPolicy.mayStopForeground(
                finishingStartId = 41,
                latestStartId = 42,
            ),
        )
        var serviceStopped = false

        val result = V2IndexingMediaProcessingTimeoutPolicy.stopThenCheckpoint(
            stopServiceImmediately = { serviceStopped = true },
            checkpoint = { "paused" },
        )

        assertTrue(serviceStopped)
        assertEquals("paused", result.getOrThrow())
    }

    @Test
    fun `media processing timeout stops before a slow checkpoint`() {
        val events = mutableListOf<String>()

        val result = V2IndexingMediaProcessingTimeoutPolicy.stopThenCheckpoint(
            stopServiceImmediately = { events += "stop" },
        ) {
            assertEquals(listOf("stop"), events)
            Thread.sleep(25L)
            events += "checkpoint"
        }

        assertTrue(result.isSuccess)
        assertEquals(listOf("stop", "checkpoint"), events)
    }

    @Test
    fun `checkpoint failure never prevents timeout shutdown`() {
        var serviceStopped = false

        val result = V2IndexingMediaProcessingTimeoutPolicy.stopThenCheckpoint(
            stopServiceImmediately = { serviceStopped = true },
        ) {
            throw IllegalStateException("disk unavailable")
        }

        assertTrue(serviceStopped)
        assertTrue(result.isFailure)
    }

    @Test
    fun `media processing timeout permanently rejects late work in that service instance`() {
        val gate = V2IndexingServiceLifetimeGate()
        assertTrue(gate.mayStartWork())

        gate.closeForMediaProcessingTimeout()
        gate.closeForMediaProcessingTimeout()

        assertFalse(gate.mayStartWork())
    }

    @Test
    fun `active notification progress is deduplicated and limited to fifteen seconds`() {
        listOf(IndexingJobState.RUNNING, IndexingJobState.ACTIVATING).forEach { state ->
            val policy = V2IndexingNotificationUpdatePolicy()
            val initial = notificationPresentation(state, "Work - 0%")
            val changed = initial.copy(text = "Work - 25%", progressPercent = 25)

            assertTrue(policy.shouldPublish(initial, observedAtElapsedMs = 1_000L))
            assertFalse(policy.shouldPublish(changed, observedAtElapsedMs = 15_999L))
            assertTrue(policy.shouldPublish(changed, observedAtElapsedMs = 16_000L))
            assertFalse(policy.shouldPublish(changed, observedAtElapsedMs = 40_000L))
        }
    }

    @Test
    fun `every ledger state and action transition publishes immediately`() {
        val policy = V2IndexingNotificationUpdatePolicy()
        val transitions = listOf(
            IndexingJobState.RUNNING to V2IndexingNotificationActionSet.PAUSE,
            IndexingJobState.PAUSE_REQUESTED to V2IndexingNotificationActionSet.NONE,
            IndexingJobState.PAUSED to V2IndexingNotificationActionSet.RESUME,
            IndexingJobState.INTERRUPTED to V2IndexingNotificationActionSet.RESUME,
            IndexingJobState.READY_TO_RESUME to V2IndexingNotificationActionSet.RESUME,
            IndexingJobState.WAITING_FOR_INPUT to V2IndexingNotificationActionSet.NONE,
            IndexingJobState.CANCELLING to V2IndexingNotificationActionSet.NONE,
            IndexingJobState.CANCELLED to V2IndexingNotificationActionSet.NONE,
            IndexingJobState.ACTIVATING to V2IndexingNotificationActionSet.NONE,
            IndexingJobState.COMPLETE to V2IndexingNotificationActionSet.NONE,
        )

        transitions.forEachIndexed { index, (state, actions) ->
            assertTrue(
                "$state transition was throttled",
                policy.shouldPublish(
                    notificationPresentation(state, state.name, actions),
                    observedAtElapsedMs = 1_000L + index,
                ),
            )
        }

        val actionPolicy = V2IndexingNotificationUpdatePolicy()
        assertTrue(
            actionPolicy.shouldPublish(
                notificationPresentation(
                    IndexingJobState.WAITING_FOR_INPUT,
                    IndexingJobState.WAITING_FOR_INPUT.name,
                    V2IndexingNotificationActionSet.NONE,
                ),
                observedAtElapsedMs = 1_100L,
            ),
        )
        assertTrue(
            actionPolicy.shouldPublish(
                notificationPresentation(
                    IndexingJobState.WAITING_FOR_INPUT,
                    IndexingJobState.WAITING_FOR_INPUT.name,
                    V2IndexingNotificationActionSet.RETRY,
                ),
                observedAtElapsedMs = 1_101L,
            ),
        )
    }

    @Test
    fun `settling replaces stale running content once before foreground detach`() {
        val policy = V2IndexingNotificationUpdatePolicy()
        val running = notificationPresentation(IndexingJobState.RUNNING, "Analyze audio - 10%")
        val latest = running.copy(text = "Analyze audio - 20%", progressPercent = 20)

        assertTrue(policy.shouldPublish(running, observedAtElapsedMs = 1_000L))
        assertFalse(policy.shouldPublish(latest, observedAtElapsedMs = 1_001L))
        assertTrue(
            policy.shouldPublish(
                latest,
                observedAtElapsedMs = 1_002L,
                settling = true,
            ),
        )
        assertFalse(
            policy.shouldPublish(
                latest,
                observedAtElapsedMs = 1_003L,
                settling = true,
            ),
        )
    }

    @Test
    fun `restart and explicit force republish even identical ledger content`() {
        val policy = V2IndexingNotificationUpdatePolicy()
        val running = notificationPresentation(IndexingJobState.RUNNING, "Analyze audio")

        assertTrue(policy.shouldPublish(running, observedAtElapsedMs = 1_000L))
        assertTrue(
            policy.shouldPublish(
                running,
                observedAtElapsedMs = 1_001L,
                force = true,
            ),
        )
        policy.invalidate()
        assertTrue(policy.shouldPublish(running, observedAtElapsedMs = 1_002L))
    }

    @Test
    fun `Android quota interruption is presented for explicit resume not failure auto replay`() {
        assertTrue(
            V2IndexingPreflightControlPolicy.isExplicitlyResumable(
                V2IndexingPreflightIntentState.INTERRUPTED,
            ),
        )
        assertFalse(
            V2IndexingPreflightControlPolicy.shouldAutoRecover(
                V2IndexingPreflightIntentState.INTERRUPTED,
            ),
        )
        assertTrue(
            V2IndexingPreflightControlPolicy.shouldAutoRecover(
                V2IndexingPreflightIntentState.PLANNING,
            ),
        )
    }

    @Test
    fun `normal service teardown checkpoints only unfinished planning as interruption`() {
        assertTrue(
            V2IndexingPreflightControlPolicy.shouldCheckpointInterruptionOnServiceTeardown(
                V2IndexingPreflightIntentState.REQUESTED,
            ),
        )
        assertTrue(
            V2IndexingPreflightControlPolicy.shouldCheckpointInterruptionOnServiceTeardown(
                V2IndexingPreflightIntentState.PLANNING,
            ),
        )
        assertFalse(
            V2IndexingPreflightControlPolicy.shouldCheckpointInterruptionOnServiceTeardown(
                V2IndexingPreflightIntentState.RESOLVED_WITH_EXECUTABLE_ROWS,
            ),
        )
        assertFalse(
            V2IndexingPreflightControlPolicy.shouldCheckpointInterruptionOnServiceTeardown(
                V2IndexingPreflightIntentState.MATERIALIZED,
            ),
        )
    }

    @Test
    fun `ledger published before final intent transition is promoted exactly once`() {
        val planning = V2IndexingPreflightIntentStateMachine.beginOrResumePlanning(
            requestedPreflight(),
            "generation-a",
            V2IndexingPreflightProgress(
                V2IndexingPreflightPhase.PERSISTING_LEDGER,
                "Persisting ledger",
            ),
            101L,
        )
        val resolved = V2IndexingPreflightIntentStateMachine.resolveWithExecutableRows(
            current = planning,
            planned = planning.selected,
            rejected = emptyList(),
            specId = "job-spec-a",
            nowEpochMs = 102L,
        )
        val materialized = V2IndexingPreflightIntentStateMachine.materializeResolved(
            resolved,
            103L,
        )

        assertEquals(
            V2IndexingPreflightLedgerLinkAction.PROMOTE_RESOLVED_INTENT,
            V2IndexingPreflightLedgerLinkPolicy.action(resolved, "job-spec-a"),
        )
        assertEquals(
            V2IndexingPreflightLedgerLinkAction.USE_MATERIALIZED_INTENT,
            V2IndexingPreflightLedgerLinkPolicy.action(materialized, "job-spec-a"),
        )
    }

    @Test
    fun `materialized handoff is verified again before ledger command dispatch`() {
        val planning = V2IndexingPreflightIntentStateMachine.beginOrResumePlanning(
            requestedPreflight(),
            "generation-a",
            V2IndexingPreflightProgress(
                V2IndexingPreflightPhase.PERSISTING_LEDGER,
                "Persisting ledger",
            ),
            101L,
        )
        val resolved = V2IndexingPreflightIntentStateMachine.resolveWithExecutableRows(
            current = planning,
            planned = planning.selected,
            rejected = emptyList(),
            specId = "job-spec-a",
            nowEpochMs = 102L,
        )
        val materialized = V2IndexingPreflightIntentStateMachine.materializeResolved(
            resolved,
            103L,
        )

        V2IndexingPreflightLedgerLinkPolicy.requireMaterializedLink(
            materialized,
            "job-spec-a",
        )
        assertThrows(IllegalArgumentException::class.java) {
            V2IndexingPreflightLedgerLinkPolicy.requireMaterializedLink(
                materialized,
                "job-spec-b",
            )
        }
        assertThrows(IllegalArgumentException::class.java) {
            V2IndexingPreflightLedgerLinkPolicy.requireMaterializedLink(
                resolved,
                "job-spec-a",
            )
        }
    }

    @Test
    fun `timeout during ledger publication preserves a reconcilable spec link`() {
        val planning = V2IndexingPreflightIntentStateMachine.beginOrResumePlanning(
            requestedPreflight(),
            "generation-a",
            V2IndexingPreflightProgress(
                V2IndexingPreflightPhase.PERSISTING_LEDGER,
                "Persisting ledger",
            ),
            101L,
        )
        val resolved = V2IndexingPreflightIntentStateMachine.resolveWithExecutableRows(
            current = planning,
            planned = planning.selected,
            rejected = emptyList(),
            specId = "job-spec-a",
            nowEpochMs = 102L,
        )

        val timedOut = V2IndexingPreflightIntentStateMachine.interrupt(
            resolved,
            "Android media-processing timeout",
            103L,
        )

        assertEquals(resolved, timedOut)
        assertEquals(
            V2IndexingPreflightLedgerLinkAction.PROMOTE_RESOLVED_INTENT,
            V2IndexingPreflightLedgerLinkPolicy.action(timedOut, "job-spec-a"),
        )
    }

    @Test
    fun `cancelled ledger finishes its interrupted intent transition idempotently`() {
        assertEquals(
            V2IndexingTerminalCancellationReconciliationAction.FINISH_CANCELLED_INTENT,
            V2IndexingPreflightLedgerLinkPolicy.terminalCancellationAction(
                intentState = V2IndexingPreflightIntentState.CANCEL_REQUESTED,
                ledgerState = IndexingJobState.CANCELLED,
            ),
        )
        assertEquals(
            V2IndexingTerminalCancellationReconciliationAction.USE_CANCELLED_PAIR,
            V2IndexingPreflightLedgerLinkPolicy.terminalCancellationAction(
                intentState = V2IndexingPreflightIntentState.CANCELLED,
                ledgerState = IndexingJobState.CANCELLED,
            ),
        )
        assertEquals(
            null,
            V2IndexingPreflightLedgerLinkPolicy.terminalCancellationAction(
                intentState = V2IndexingPreflightIntentState.CANCEL_REQUESTED,
                ledgerState = IndexingJobState.CANCELLING,
            ),
        )
    }

    @Test
    fun `ledger cancel settles an active preflight publication owner first`() {
        assertTrue(
            V2IndexingPreflightOwnerPolicy.shouldSettleBeforeLedgerCommand(
                commandType = V2IndexingServiceCommandType.CANCEL,
                preflightOwnerActive = true,
            ),
        )
        assertFalse(
            V2IndexingPreflightOwnerPolicy.shouldSettleBeforeLedgerCommand(
                commandType = V2IndexingServiceCommandType.RESUME,
                preflightOwnerActive = true,
            ),
        )
        assertFalse(
            V2IndexingPreflightOwnerPolicy.shouldSettleBeforeLedgerCommand(
                commandType = V2IndexingServiceCommandType.CANCEL,
                preflightOwnerActive = false,
            ),
        )
    }

    @Test
    fun `ledger handoff fails closed on missing result or spec mismatch`() {
        val planning = V2IndexingPreflightIntentStateMachine.beginOrResumePlanning(
            requestedPreflight(),
            "generation-a",
            V2IndexingPreflightProgress(
                V2IndexingPreflightPhase.PERSISTING_LEDGER,
                "Persisting ledger",
            ),
            101L,
        )
        val resolved = V2IndexingPreflightIntentStateMachine.resolveWithExecutableRows(
            current = planning,
            planned = planning.selected,
            rejected = emptyList(),
            specId = "job-spec-a",
            nowEpochMs = 102L,
        )

        assertThrows(IllegalStateException::class.java) {
            V2IndexingPreflightLedgerLinkPolicy.action(planning, "job-spec-a")
        }
        assertThrows(IllegalArgumentException::class.java) {
            V2IndexingPreflightLedgerLinkPolicy.action(resolved, "job-spec-b")
        }
    }

    @Test
    fun `resolved-with-ledger is auto recovered while terminal request results are not`() {
        assertTrue(
            V2IndexingPreflightControlPolicy.shouldAutoRecover(
                V2IndexingPreflightIntentState.RESOLVED_WITH_EXECUTABLE_ROWS,
            ),
        )
        assertFalse(
            V2IndexingPreflightControlPolicy.shouldAutoRecover(
                V2IndexingPreflightIntentState.RESOLVED_WITHOUT_EXECUTABLE_ROWS,
            ),
        )
        assertTrue(
            V2IndexingPreflightControlPolicy.isTerminalRequest(
                V2IndexingPreflightIntentState.RESOLVED_WITHOUT_EXECUTABLE_ROWS,
            ),
        )
        assertTrue(
            V2IndexingPreflightControlPolicy.isTerminalRequest(
                V2IndexingPreflightIntentState.MATERIALIZED,
            ),
        )
    }

    @Test
    fun `crash handoff restores only the untouched initial execution profile`() {
        assertTrue(
            V2IndexingPreflightLedgerLinkPolicy.shouldRestoreInitialProfile(
                ledgerState = IndexingJobState.PLANNED,
                ledgerRevision = 0L,
                ledgerProfile = V2IndexingExecutionProfile.FULL,
                requestedProfile = V2IndexingExecutionProfile.BACKGROUND,
            ),
        )
        assertFalse(
            V2IndexingPreflightLedgerLinkPolicy.shouldRestoreInitialProfile(
                ledgerState = IndexingJobState.PLANNED,
                ledgerRevision = 1L,
                ledgerProfile = V2IndexingExecutionProfile.FULL,
                requestedProfile = V2IndexingExecutionProfile.BACKGROUND,
            ),
        )
        assertFalse(
            V2IndexingPreflightLedgerLinkPolicy.shouldRestoreInitialProfile(
                ledgerState = IndexingJobState.RUNNING,
                ledgerRevision = 4L,
                ledgerProfile = V2IndexingExecutionProfile.FULL,
                requestedProfile = V2IndexingExecutionProfile.BACKGROUND,
            ),
        )
    }

    private fun requestedPreflight() = V2IndexingPreflightIntentFactory.create(
        jobId = "job-link-policy",
        selected = listOf(
            V2IndexingPreflightSelection(
                powerampFileId = 7L,
                providerPhysicalPath = "/music/7.flac",
                durationMs = 180_000L,
                offsetMs = 0L,
                cueSourceImageFolderId = null,
            ),
        ),
        rebuildDerivedIndexes = true,
        executionProfile = V2IndexingExecutionProfile.FULL,
        nowEpochMs = 100L,
    )

    private fun notificationPresentation(
        state: IndexingJobState,
        text: String,
        actionSet: V2IndexingNotificationActionSet =
            if (state == IndexingJobState.RUNNING) {
                V2IndexingNotificationActionSet.PAUSE
            } else {
                V2IndexingNotificationActionSet.NONE
            },
    ) = V2IndexingNotificationPresentation(
        jobId = "job-notification-policy",
        state = state,
        title = "On-device indexing",
        text = text,
        progressPercent = 0,
        indeterminateProgress = false,
        ongoing = state in setOf(
            IndexingJobState.RUNNING,
            IndexingJobState.PAUSE_REQUESTED,
            IndexingJobState.ACTIVATING,
            IndexingJobState.CANCELLING,
        ),
        actionSet = actionSet,
    )
}
