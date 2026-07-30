package com.powerampstartradio.indexing.v2

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertThrows
import org.junit.Test

class V2ExecutorLeaseCoordinatorTest {
    @Test
    fun `one epoch owns the executor and stale callbacks fail closed`() {
        val persistence = MemoryPersistence()
        val firstProcess = V2ExecutorLeaseCoordinator(persistence, TickClock())
        val first = firstProcess.claim("job-a", "process-a")

        assertThrows(V2ExecutorLeaseConflictException::class.java) {
            firstProcess.claim("job-a", "process-a")
        }
        assertThrows(V2ExecutorLeaseConflictException::class.java) {
            firstProcess.claim("job-b", "process-a")
        }

        firstProcess.release(first)
        val second = firstProcess.claim("job-b", "process-a")
        assertEquals(first.epoch + 1L, second.epoch)
        assertThrows(V2ExecutorLeaseConflictException::class.java) {
            firstProcess.requireCurrent(first)
        }
    }

    @Test
    fun `new process can retire lease only after explicit startup reconciliation`() {
        val persistence = MemoryPersistence()
        val oldProcess = V2ExecutorLeaseCoordinator(persistence, TickClock())
        val old = oldProcess.claim("job-a", "process-a")
        val newProcess = V2ExecutorLeaseCoordinator(persistence, TickClock())

        assertThrows(V2ExecutorLeaseConflictException::class.java) {
            newProcess.claim("job-a", "process-b")
        }
        assertEquals(
            old,
            newProcess.retirePreviousProcessLease("process-b")?.token(),
        )
        assertNull(newProcess.activeLease())

        val recovered = newProcess.claim("job-a", "process-b")
        assertEquals(old.epoch + 1L, recovered.epoch)
        assertThrows(V2ExecutorLeaseConflictException::class.java) {
            oldProcess.heartbeat(old)
        }
    }

    private class MemoryPersistence : V2ExecutorLeasePersistence {
        private var state = V2ExecutorLeaseState(
            schemaVersion = V2ExecutorLeaseCoordinator.SCHEMA_VERSION,
            lastIssuedEpoch = 0L,
            active = null,
        )

        override fun read(): V2ExecutorLeaseState = state

        override fun write(state: V2ExecutorLeaseState) {
            this.state = state
        }
    }

    private class TickClock : () -> Long {
        private var value = 100L
        override fun invoke(): Long = ++value
    }
}
