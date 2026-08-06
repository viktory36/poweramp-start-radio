package com.powerampstartradio.poweramp

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class QueueMutationReconcilerTest {
    @Test
    fun duplicateOccurrencesRemainIndependentlyVerified() {
        val match = QueueMutationReconciler.reconcile(
            requested = listOf(11L, 22L, 11L, 33L),
            observed = listOf(11L, 22L, 11L, 33L),
        )

        assertEquals(setOf(0, 1, 2, 3), match.requestIndices)
        assertEquals(listOf(11L, 22L, 11L, 33L), match.fileIds)
        assertEquals(mapOf(0 to 0, 1 to 1, 2 to 2, 3 to 3), match.observedIndicesByRequestIndex)
        assertEquals(0, match.unexpectedObservedCount)
    }

    @Test
    fun partialAcceptanceMarksTheActualOrderedOccurrences() {
        val match = QueueMutationReconciler.reconcile(
            requested = listOf(11L, 22L, 11L, 33L),
            observed = listOf(22L, 11L),
        )

        assertEquals(setOf(1, 2), match.requestIndices)
        assertEquals(listOf(22L, 11L), match.fileIds)
        assertEquals(0, match.unexpectedObservedCount)
    }

    @Test
    fun unrelatedObservedRowsAreNeverClaimedAsAccepted() {
        val match = QueueMutationReconciler.reconcile(
            requested = listOf(11L, 22L),
            observed = listOf(99L, 11L, 77L),
        )

        assertEquals(setOf(0), match.requestIndices)
        assertEquals(listOf(11L), match.fileIds)
        assertEquals(2, match.unexpectedObservedCount)
    }

    @Test
    fun resultNeverEquatesProviderReportWithVerification() {
        val result = QueueMutationResult(
            kind = QueueMutationKind.REPLACE,
            requestedFileIds = listOf(11L, 22L),
            verifiedRequestIndices = setOf(1),
            verifiedFileIds = listOf(22L),
            providerReportedInsertCount = 2,
            beforeCount = 4,
            afterCount = 1,
            preservedAnchorFileId = null,
            unexpectedObservedCount = 0,
            fallbackUsed = false,
        )

        assertFalse(result.isRequestVerified(0))
        assertTrue(result.isRequestVerified(1))
        assertEquals(1, result.verifiedCount)
        assertEquals(1, result.failedCount)
        assertFalse(result.fullyVerified)
    }

    @Test
    fun readbackFailureVerifiesNothing() {
        val result = QueueMutationResult(
            kind = QueueMutationKind.APPEND,
            requestedFileIds = listOf(11L),
            verifiedRequestIndices = emptySet(),
            verifiedFileIds = emptyList(),
            providerReportedInsertCount = 1,
            beforeCount = 2,
            afterCount = null,
            preservedAnchorFileId = null,
            unexpectedObservedCount = 0,
            fallbackUsed = false,
            verificationError = "provider unavailable",
        )

        assertFalse(result.verificationSucceeded)
        assertEquals(0, result.verifiedCount)
        assertFalse(result.fullyVerified)
    }

    @Test
    fun fullyVerifiedRequiresDistinctExactQueueRowsForEveryOccurrence() {
        val result = QueueMutationResult(
            kind = QueueMutationKind.REPLACE,
            requestedFileIds = listOf(11L, 11L),
            verifiedRequestIndices = setOf(0, 1),
            verifiedFileIds = listOf(11L, 11L),
            providerReportedInsertCount = 2,
            beforeCount = 1,
            afterCount = 2,
            preservedAnchorFileId = null,
            verifiedQueueEntryIdsByRequestIndex = mapOf(0 to 101L, 1 to 102L),
            unexpectedObservedCount = 0,
            fallbackUsed = false,
        )

        assertTrue(result.fullyVerified)
        assertEquals(101L, result.verifiedQueueEntryId(0))
        assertEquals(102L, result.verifiedQueueEntryId(1))
        assertFalse(
            result.copy(
                verifiedQueueEntryIdsByRequestIndex = mapOf(0 to 101L, 1 to 101L),
            ).fullyVerified,
        )
    }

    @Test
    fun exactReadbackTracksPhysicalRowsAcrossDuplicateFileIds() {
        val match = QueueMutationReconciler.reconcileExactOccurrences(
            requestedFileIds = listOf(11L, 11L, 22L),
            expectedQueueEntryIdsByRequestIndex = mapOf(
                0 to 101L,
                1 to 102L,
                2 to 103L,
            ),
            observedOccurrences = listOf(
                QueueMutationReconciler.ObservedOccurrence(101L, 11L),
                QueueMutationReconciler.ObservedOccurrence(103L, 22L),
                QueueMutationReconciler.ObservedOccurrence(104L, 11L),
            ),
            countUnmatchedObserved = true,
        )

        assertEquals(setOf(0, 2), match.requestIndices)
        assertEquals(mapOf(0 to 0, 2 to 1), match.observedIndicesByRequestIndex)
        assertEquals(1, match.unexpectedObservedCount)
    }

    @Test
    fun exactReadbackDoesNotVerifyReorderedPhysicalRows() {
        val match = QueueMutationReconciler.reconcileExactOccurrences(
            requestedFileIds = listOf(11L, 22L),
            expectedQueueEntryIdsByRequestIndex = mapOf(0 to 101L, 1 to 102L),
            observedOccurrences = listOf(
                QueueMutationReconciler.ObservedOccurrence(102L, 22L),
                QueueMutationReconciler.ObservedOccurrence(101L, 11L),
            ),
            countUnmatchedObserved = true,
        )

        assertFalse(match.requestIndices == setOf(0, 1))
        assertEquals(1, match.unexpectedObservedCount)
    }

    @Test
    fun largeExactQueueWithDuplicateOccurrencesUsesBoundedFastPath() {
        val requested = List(10_000) { index -> (index % 97).toLong() }

        val match = QueueMutationReconciler.reconcile(requested, requested.toList())

        assertEquals(10_000, match.requestIndices.size)
        assertEquals(requested, match.fileIds)
        assertEquals(0, match.unexpectedObservedCount)
    }

    @Test
    fun largePartialQueueConservativelyMarksOnlyObservedOrderedOccurrences() {
        val requested = List(10_000) { it.toLong() }
        val observed = requested.filterIndexed { index, _ -> index % 2 == 0 }

        val match = QueueMutationReconciler.reconcile(requested, observed)

        assertEquals(5_000, match.requestIndices.size)
        assertEquals(observed, match.fileIds)
        assertEquals(0, match.unexpectedObservedCount)
        assertTrue(match.requestIndices.all { it % 2 == 0 })
    }

    @Test
    fun partialBatchRetriesOnlyMissingOccurrencesWhenFileIdsRepeat() {
        val baseline = listOf(QueueEntry(queueId = 10L, fileId = 90L, sort = 4))
        val intended = listOf(
            PlannedQueueInsertion(requestIndex = 0, fileId = 11L, sort = 5),
            PlannedQueueInsertion(requestIndex = 1, fileId = 11L, sort = 6),
            PlannedQueueInsertion(requestIndex = 2, fileId = 22L, sort = 7),
        )

        val plan = PartialQueueInsertionReconciler.reconcile(
            baseline = baseline,
            observed = baseline + QueueEntry(queueId = 12L, fileId = 11L, sort = 6),
            intended = intended,
        )

        assertTrue(plan.safeToContinue)
        assertEquals(setOf(1), plan.appliedRequestIndices)
        assertEquals(listOf(intended[0], intended[2]), plan.missing)
    }

    @Test
    fun partialBatchFailsClosedWhenBaselineChangedOrUnexpectedRowAppears() {
        val baseline = listOf(QueueEntry(queueId = 10L, fileId = 90L, sort = 4))
        val intended = listOf(PlannedQueueInsertion(0, 11L, 5))

        val missingBaseline = PartialQueueInsertionReconciler.reconcile(
            baseline = baseline,
            observed = listOf(QueueEntry(queueId = 11L, fileId = 11L, sort = 5)),
            intended = intended,
        )
        val unexpected = PartialQueueInsertionReconciler.reconcile(
            baseline = baseline,
            observed = baseline + QueueEntry(queueId = 12L, fileId = 99L, sort = 5),
            intended = intended,
        )

        assertFalse(missingBaseline.safeToContinue)
        assertFalse(unexpected.safeToContinue)
    }

    @Test
    fun partialBatchFailsClosedWhenOneIntendedOccurrenceWasDuplicated() {
        val baseline = listOf(QueueEntry(queueId = 10L, fileId = 90L, sort = 4))
        val intended = listOf(PlannedQueueInsertion(0, 11L, 5))

        val plan = PartialQueueInsertionReconciler.reconcile(
            baseline = baseline,
            observed = baseline + listOf(
                QueueEntry(queueId = 11L, fileId = 11L, sort = 5),
                QueueEntry(queueId = 12L, fileId = 11L, sort = 5),
            ),
            intended = intended,
        )

        assertFalse(plan.safeToContinue)
    }

    @Test
    fun rollbackAllowsRegeneratedNonAnchorRowsButRequiresExactOrderedContent() {
        val original = listOf(
            QueueEntry(queueId = 10L, fileId = 90L, sort = 4),
            QueueEntry(queueId = 20L, fileId = 91L, sort = 5),
        )
        val restored = listOf(
            original[0],
            QueueEntry(queueId = 200L, fileId = 91L, sort = 5),
        )

        assertEquals(null, QueueRollbackVerifier.error(original, restored, 10L))
        assertTrue(
            QueueRollbackVerifier.error(original, restored.reversed(), 10L) != null,
        )
    }

    @Test
    fun rollbackFailsClosedWhenLiveAnchorWasRegenerated() {
        val original = listOf(
            QueueEntry(queueId = 10L, fileId = 90L, sort = 4),
            QueueEntry(queueId = 20L, fileId = 91L, sort = 5),
        )
        val restored = listOf(
            QueueEntry(queueId = 100L, fileId = 90L, sort = 4),
            QueueEntry(queueId = 200L, fileId = 91L, sort = 5),
        )

        assertTrue(QueueRollbackVerifier.error(original, restored, 10L) != null)
        assertEquals(null, QueueRollbackVerifier.error(original, restored, null))
    }

    @Test
    fun rollbackAttemptCanNeverBeReportedAsSuccessfulDelivery() {
        val result = QueueMutationResult(
            kind = QueueMutationKind.APPEND,
            requestedFileIds = listOf(11L),
            verifiedRequestIndices = setOf(0),
            verifiedFileIds = listOf(11L),
            providerReportedInsertCount = 1,
            beforeCount = 1,
            afterCount = 1,
            preservedAnchorFileId = null,
            verifiedQueueEntryIdsByRequestIndex = mapOf(0 to 101L),
            unexpectedObservedCount = 0,
            fallbackUsed = false,
            rollbackAttempted = true,
            rollbackVerified = true,
        )

        assertFalse(result.fullyVerified)
    }
}
