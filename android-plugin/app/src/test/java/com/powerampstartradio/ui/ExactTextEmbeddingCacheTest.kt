package com.powerampstartradio.ui

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertThrows
import org.junit.Test

class ExactTextEmbeddingCacheTest {
    @Test
    fun cachePreservesExactFloatBitsAndOwnsItsStorage() {
        val cache = ExactTextEmbeddingCache(maxEntries = 2)
        val key = key("sleep")
        val source = floatArrayOf(
            Float.fromBits(0x3f800001),
            -0.0f,
            Float.fromBits(0xbf000001.toInt()),
        )
        val expectedBits = source.rawBits()

        cache.put(key, source)
        source[0] = 7f
        val firstRead = checkNotNull(cache.get(key))
        assertEquals(expectedBits, firstRead.rawBits())

        firstRead[1] = 9f
        assertEquals(expectedBits, checkNotNull(cache.get(key)).rawBits())
    }

    @Test
    fun keyDoesNotConflateCaseOrWhitespace() {
        val cache = ExactTextEmbeddingCache(maxEntries = 4)
        cache.put(key("ambient"), floatArrayOf(1f))

        assertNull(cache.get(key("Ambient")))
        assertNull(cache.get(key(" ambient")))
        assertNull(cache.get(key("ambient ")))
        assertEquals(listOf(1f.toRawBits()), checkNotNull(cache.get(key("ambient"))).rawBits())
    }

    @Test
    fun everyRuntimeIdentityFieldParticipatesInCacheIdentity() {
        val cache = ExactTextEmbeddingCache(maxEntries = 8)
        val baseline = identity()
        cache.put(
            ExactTextEmbeddingCacheKey("guitar", baseline),
            floatArrayOf(0.25f),
        )

        val alternatives = listOf(
            baseline.copy(retrievalSpecIdentity = "spec-b"),
            baseline.copy(textModelSha256 = "model-b"),
            baseline.copy(tokenizerModelSha256 = "tokenizer-b"),
            baseline.copy(tokenizerRuntimeContractSha256 = "runtime-b"),
            baseline.copy(inferenceBackendPolicyId = "backend-b"),
        )
        alternatives.forEach { changed ->
            assertNull(cache.get(ExactTextEmbeddingCacheKey("guitar", changed)))
        }
        assertEquals(
            listOf(0.25f.toRawBits()),
            checkNotNull(cache.get(ExactTextEmbeddingCacheKey("guitar", baseline))).rawBits(),
        )
    }

    @Test
    fun leastRecentlyUsedEntryIsEvictedAtTheBound() {
        val cache = ExactTextEmbeddingCache(maxEntries = 2)
        val first = key("first")
        val second = key("second")
        val third = key("third")
        cache.put(first, floatArrayOf(1f))
        cache.put(second, floatArrayOf(2f))

        checkNotNull(cache.get(first))
        cache.put(third, floatArrayOf(3f))

        assertEquals(2, cache.size())
        checkNotNull(cache.get(first))
        assertNull(cache.get(second))
        checkNotNull(cache.get(third))
    }

    @Test
    fun clearInvalidatesAllEntries() {
        val cache = ExactTextEmbeddingCache(maxEntries = 2)
        cache.put(key("sleep"), floatArrayOf(1f))
        cache.put(key("relaxing"), floatArrayOf(2f))

        cache.clear()

        assertEquals(0, cache.size())
        assertNull(cache.get(key("sleep")))
        assertNull(cache.get(key("relaxing")))
    }

    @Test
    fun invalidConfigurationAndEmptyEmbeddingsAreRejected() {
        assertThrows(IllegalArgumentException::class.java) {
            ExactTextEmbeddingCache(maxEntries = 0)
        }
        val cache = ExactTextEmbeddingCache(maxEntries = 1)
        assertThrows(IllegalArgumentException::class.java) {
            cache.put(key("sleep"), floatArrayOf())
        }
        assertFalse(cache.size() > 0)
    }

    private fun key(query: String) = ExactTextEmbeddingCacheKey(query, identity())

    private fun identity() = TextEmbeddingRuntimeIdentity(
        retrievalSpecIdentity = "spec-a",
        textModelSha256 = "model-a",
        tokenizerModelSha256 = "tokenizer-a",
        tokenizerRuntimeContractSha256 = "runtime-a",
        inferenceBackendPolicyId = "backend-a",
    )

    private fun FloatArray.rawBits(): List<Int> = map(Float::toRawBits)
}
