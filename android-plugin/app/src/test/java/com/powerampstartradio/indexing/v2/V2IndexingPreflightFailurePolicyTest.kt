package com.powerampstartradio.indexing.v2

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class V2IndexingPreflightFailurePolicyTest {
    @Test
    fun `every preflight code has exactly one explicit scope`() {
        val semantics = V2IndexingPreflightFailureCode.entries.associateWith(
            V2IndexingPreflightFailurePolicy::semantics,
        )

        assertEquals(V2IndexingPreflightFailureCode.entries.size, semantics.size)
        assertEquals(
            setOf(
                V2IndexingPreflightFailureCode.INVALID_LOGICAL_SPAN,
                V2IndexingPreflightFailureCode.CUE_SOURCE_IMAGE,
                V2IndexingPreflightFailureCode.AUDIO_TOO_SHORT,
                V2IndexingPreflightFailureCode.SOURCE_UNREADABLE,
                V2IndexingPreflightFailureCode.NO_AUDIO_STREAM,
                V2IndexingPreflightFailureCode.UNSUPPORTED_OR_INVALID_AUDIO_CONTAINER,
            ),
            semantics.filterValues {
                it.scope == V2IndexingPreflightFailureScope.SELECTED_OCCURRENCE
            }.keys,
        )
        semantics.forEach { (_, value) ->
            if (value.scope == V2IndexingPreflightFailureScope.SELECTED_OCCURRENCE) {
                assertTrue(value.disposition != null)
                assertTrue(value.retryTrigger != null)
            } else {
                assertNull(value.disposition)
                assertNull(value.retryTrigger)
            }
        }
    }

    @Test
    fun `local retry semantics describe the condition that can change the result`() {
        assertEquals(
            RetryTrigger.SOURCE_AVAILABLE,
            V2IndexingPreflightFailurePolicy.semantics(
                V2IndexingPreflightFailureCode.SOURCE_UNREADABLE,
            ).retryTrigger,
        )
        assertEquals(
            RetryTrigger.USER_REQUEST,
            V2IndexingPreflightFailurePolicy.semantics(
                V2IndexingPreflightFailureCode.UNSUPPORTED_OR_INVALID_AUDIO_CONTAINER,
            ).retryTrigger,
        )
        assertEquals(
            RetryTrigger.SOURCE_OR_LIBRARY_CHANGED,
            V2IndexingPreflightFailurePolicy.semantics(
                V2IndexingPreflightFailureCode.CUE_SOURCE_IMAGE,
            ).retryTrigger,
        )
    }
}
