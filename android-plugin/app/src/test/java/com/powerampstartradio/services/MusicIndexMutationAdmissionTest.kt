package com.powerampstartradio.services

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class MusicIndexMutationAdmissionTest {
    @Test
    fun `merge ownership refuses indexing before durable preflight work begins`() {
        val admission = MusicIndexMutationAdmission()
        val merge = MusicIndexMutationAdmission.newOwner("merge")
        val indexing = MusicIndexMutationAdmission.newOwner("indexing")
        var createdPreflight = false
        var claimedPointer = false

        assertTrue(admission.tryAcquire(merge))
        if (admission.tryAcquire(indexing)) {
            createdPreflight = true
            claimedPointer = true
        }

        assertFalse(createdPreflight)
        assertFalse(claimedPointer)
        assertTrue(admission.isOwnedBy(merge))
        assertTrue(admission.release(merge))
    }

    @Test
    fun `indexing ownership excludes merge through intent creation and pointer claim`() {
        val admission = MusicIndexMutationAdmission()
        val merge = MusicIndexMutationAdmission.newOwner("merge")
        val indexing = MusicIndexMutationAdmission.newOwner("indexing")

        assertTrue(admission.tryAcquire(indexing))
        val createdPreflight = true
        assertFalse(admission.tryAcquire(merge))
        val claimedPointer = true
        assertTrue(admission.release(indexing))

        assertTrue(createdPreflight)
        assertTrue(claimedPointer)
        assertTrue(admission.tryAcquire(merge))
        assertTrue(admission.release(merge))
    }

    @Test
    fun `only the owning operation can release the process boundary`() {
        val admission = MusicIndexMutationAdmission()
        val first = MusicIndexMutationAdmission.newOwner("first")
        val second = MusicIndexMutationAdmission.newOwner("second")

        assertTrue(admission.tryAcquire(first))
        assertFalse(admission.release(second))
        assertTrue(admission.isOwnedBy(first))
        assertFalse(admission.tryAcquire(second))
        assertTrue(admission.release(first))
        assertTrue(admission.tryAcquire(second))
        assertTrue(admission.release(second))
    }
}
