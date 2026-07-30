package com.powerampstartradio.indexing.v2

import org.junit.Assert.assertFalse
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class V2DerivedGraphExactProofPolicyTest {
    @Test
    fun `maintenance inherits exactness only through a proven base and repaired graph`() {
        assertTrue(
            V2DerivedGraphExactProofPolicy.shouldInstall(
                origin = V2IndexGenerationOrigin.LIBRARY_MAINTENANCE,
                graphPresent = true,
                baseHasExactProof = true,
            ),
        )
        assertFalse(
            V2DerivedGraphExactProofPolicy.shouldInstall(
                origin = V2IndexGenerationOrigin.LIBRARY_MAINTENANCE,
                graphPresent = true,
                baseHasExactProof = false,
            ),
        )
        assertFalse(
            V2DerivedGraphExactProofPolicy.shouldInstall(
                origin = V2IndexGenerationOrigin.LIBRARY_MAINTENANCE,
                graphPresent = false,
                baseHasExactProof = true,
            ),
        )
    }

    @Test
    fun `bootstrap graph remains an untrusted compatibility import`() {
        assertFalse(
            V2DerivedGraphExactProofPolicy.shouldInstall(
                origin = V2IndexGenerationOrigin.BOOTSTRAP_COMPATIBILITY,
                graphPresent = true,
                baseHasExactProof = true,
            ),
        )
    }

    @Test
    fun `server merge requires both exact base and updated graph`() {
        assertTrue(
            V2DerivedGraphExactProofPolicy.shouldInstall(
                origin = V2IndexGenerationOrigin.SERVER_MERGE,
                graphPresent = true,
                baseHasExactProof = true,
            ),
        )
        assertThrows(IllegalArgumentException::class.java) {
            V2DerivedGraphExactProofPolicy.shouldInstall(
                origin = V2IndexGenerationOrigin.SERVER_MERGE,
                graphPresent = true,
                baseHasExactProof = false,
            )
        }
        assertThrows(IllegalArgumentException::class.java) {
            V2DerivedGraphExactProofPolicy.shouldInstall(
                origin = V2IndexGenerationOrigin.SERVER_MERGE,
                graphPresent = false,
                baseHasExactProof = true,
            )
        }
    }
}
