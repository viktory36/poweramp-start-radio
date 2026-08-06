package com.powerampstartradio.ui

import java.util.LinkedHashMap

/** Every artifact and runtime contract that can change a text embedding's float values. */
internal data class TextEmbeddingRuntimeIdentity(
    val retrievalSpecIdentity: String,
    val textModelSha256: String,
    val tokenizerModelSha256: String,
    val tokenizerRuntimeContractSha256: String,
    val inferenceBackendPolicyId: String,
)

/**
 * Exact model input plus its complete runtime identity.
 *
 * Query text is intentionally not case-folded, whitespace-folded or otherwise rewritten here:
 * cache reuse must never change the tokens that would have reached the model.
 */
internal data class ExactTextEmbeddingCacheKey(
    val query: String,
    val runtimeIdentity: TextEmbeddingRuntimeIdentity,
)

/** Small process-local LRU. Values are copied at both boundaries to prevent cache corruption. */
internal class ExactTextEmbeddingCache(
    private val maxEntries: Int,
) {
    init {
        require(maxEntries > 0) { "maxEntries must be positive" }
    }

    private val entries = object : LinkedHashMap<ExactTextEmbeddingCacheKey, FloatArray>(
        maxEntries + 1,
        0.75f,
        true,
    ) {
        override fun removeEldestEntry(
            eldest: MutableMap.MutableEntry<ExactTextEmbeddingCacheKey, FloatArray>?,
        ): Boolean = size > maxEntries
    }

    @Synchronized
    fun get(key: ExactTextEmbeddingCacheKey): FloatArray? = entries[key]?.copyOf()

    @Synchronized
    fun put(key: ExactTextEmbeddingCacheKey, embedding: FloatArray) {
        require(embedding.isNotEmpty()) { "Text embedding must not be empty" }
        entries[key] = embedding.copyOf()
    }

    @Synchronized
    fun clear() {
        entries.clear()
    }

    @Synchronized
    internal fun size(): Int = entries.size
}
