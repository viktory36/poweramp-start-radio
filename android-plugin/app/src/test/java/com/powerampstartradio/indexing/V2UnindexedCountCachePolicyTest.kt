package com.powerampstartradio.indexing

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class V2UnindexedCountCachePolicyTest {
    private val current = V2UnindexedCountCacheIdentity(
        databaseGeneration = "generation-a",
        providerGeneration = "provider-a",
        exclusionsFingerprint = "exclusions-a",
        attentionFingerprint = "attention-a",
        detectionPolicyId = V2UnindexedCountCachePolicy.DETECTION_POLICY_ID,
    )

    @Test
    fun `reuses only the identical complete identity`() {
        assertTrue(V2UnindexedCountCachePolicy.canReuse(current, current))
        assertFalse(V2UnindexedCountCachePolicy.canReuse(null, current))
        assertFalse(
            V2UnindexedCountCachePolicy.canReuse(
                current.copy(providerGeneration = "provider-b"),
                current,
            ),
        )
        assertFalse(
            V2UnindexedCountCachePolicy.canReuse(
                current.copy(databaseGeneration = "generation-b"),
                current,
            ),
        )
        assertFalse(
            V2UnindexedCountCachePolicy.canReuse(
                current.copy(exclusionsFingerprint = "exclusions-b"),
                current,
            ),
        )
        assertFalse(
            V2UnindexedCountCachePolicy.canReuse(
                current.copy(attentionFingerprint = "attention-b"),
                current,
            ),
        )
        assertFalse(
            V2UnindexedCountCachePolicy.canReuse(
                current.copy(detectionPolicyId = "old-policy"),
                current,
            ),
        )
    }

    @Test
    fun `never reuses an incomplete current identity`() {
        assertFalse(
            V2UnindexedCountCachePolicy.canReuse(
                current.copy(databaseGeneration = ""),
                current.copy(databaseGeneration = ""),
            ),
        )
        assertFalse(
            V2UnindexedCountCachePolicy.canReuse(
                current.copy(providerGeneration = ""),
                current.copy(providerGeneration = ""),
            ),
        )
        assertFalse(
            V2UnindexedCountCachePolicy.canReuse(
                current.copy(exclusionsFingerprint = ""),
                current.copy(exclusionsFingerprint = ""),
            ),
        )
        assertFalse(
            V2UnindexedCountCachePolicy.canReuse(
                current.copy(attentionFingerprint = ""),
                current.copy(attentionFingerprint = ""),
            ),
        )
    }
}
