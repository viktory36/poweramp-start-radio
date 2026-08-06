package com.powerampstartradio.indexing

import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test

class V2LibraryInspectionCoordinatorTest {
    @Test
    fun concurrentInspectionsNeverOverlap() = runBlocking {
        val coordinator = coordinatorWithoutCollection()
        val firstEntered = CountDownLatch(1)
        val releaseFirst = CountDownLatch(1)
        val active = AtomicInteger(0)
        val peak = AtomicInteger(0)

        val first = async(Dispatchers.Default) {
            coordinator.inspect {
                peak.accumulateAndGet(active.incrementAndGet(), ::maxOf)
                firstEntered.countDown()
                assertTrue(releaseFirst.await(5, TimeUnit.SECONDS))
                active.decrementAndGet()
            }
        }
        assertTrue(firstEntered.await(5, TimeUnit.SECONDS))

        val secondEntered = CountDownLatch(1)
        val second = async(Dispatchers.Default) {
            coordinator.inspect {
                peak.accumulateAndGet(active.incrementAndGet(), ::maxOf)
                secondEntered.countDown()
                active.decrementAndGet()
            }
        }
        assertFalse(secondEntered.await(100, TimeUnit.MILLISECONDS))
        releaseFirst.countDown()
        first.await()
        second.await()

        assertEquals(1, peak.get())
        assertEquals(0, active.get())
    }

    @Test
    fun cancelledWaiterDoesNotRunAfterOwnerFinishes() = runBlocking {
        val coordinator = coordinatorWithoutCollection()
        val firstEntered = CountDownLatch(1)
        val releaseFirst = CountDownLatch(1)
        val waiterRan = AtomicInteger(0)

        val owner = async(Dispatchers.Default) {
            coordinator.inspect {
                firstEntered.countDown()
                assertTrue(releaseFirst.await(5, TimeUnit.SECONDS))
            }
        }
        assertTrue(firstEntered.await(5, TimeUnit.SECONDS))
        val waiter = async(Dispatchers.Default) {
            coordinator.inspect { waiterRan.incrementAndGet() }
        }

        waiter.cancel(CancellationException("obsolete screen"))
        waiter.cancelAndJoin()
        releaseFirst.countDown()
        owner.await()

        assertEquals(0, waiterRan.get())
    }

    @Test
    fun reclaimsBeforeReleasingNextInspectionAndPreservesResult() = runBlocking {
        val events = mutableListOf<String>()
        val coordinator = V2LibraryInspectionCoordinator(
            V2LibraryInspectionHeapReclaimer { events += "reclaim" },
        )
        val retainedResult = Any()

        val result = coordinator.inspect {
            events += "first inspection"
            retainedResult
        }
        coordinator.inspect { events += "second inspection" }

        assertSame(retainedResult, result)
        assertEquals(
            listOf(
                "first inspection",
                "reclaim",
                "second inspection",
                "reclaim",
            ),
            events,
        )
    }

    @Test
    fun failedInspectionStillReclaimsBeforePropagatingFailure() = runBlocking {
        val events = mutableListOf<String>()
        val coordinator = V2LibraryInspectionCoordinator(
            V2LibraryInspectionHeapReclaimer { events += "reclaim" },
        )

        val failure = runCatching {
            coordinator.inspect {
                events += "inspection"
                error("failed")
            }
        }.exceptionOrNull()

        assertEquals("failed", failure?.message)
        assertEquals(listOf("inspection", "reclaim"), events)
    }

    @Test
    fun failedReclaimerDoesNotLeaveCoordinatorLocked() = runBlocking {
        var reclaimCount = 0
        val coordinator = V2LibraryInspectionCoordinator(
            V2LibraryInspectionHeapReclaimer {
                reclaimCount++
                if (reclaimCount == 1) error("reclaimer failed")
            },
        )

        try {
            coordinator.inspect { "first" }
            fail("expected reclaimer failure")
        } catch (expected: IllegalStateException) {
            assertEquals("reclaimer failed", expected.message)
        }

        assertEquals("second", coordinator.inspect { "second" })
        assertEquals(2, reclaimCount)
    }

    private fun coordinatorWithoutCollection() = V2LibraryInspectionCoordinator(
        V2LibraryInspectionHeapReclaimer {},
    )
}
