package com.powerampstartradio.data

/** First ordered identity disagreement between a graph and its embedding index. */
data class OrderedTrackIdMismatch(
    val index: Int,
    val graphTrackId: Long?,
    val embeddingTrackId: Long?,
)

/** Pure generation-alignment check shared by mmap code and host tests. */
object GraphEmbeddingIdAlignment {
    fun firstMismatch(
        graphTrackIds: LongArray,
        embeddingTrackIds: LongArray,
    ): OrderedTrackIdMismatch? = firstMismatch(
        graphCount = graphTrackIds.size,
        graphTrackIdAt = graphTrackIds::get,
        embeddingCount = embeddingTrackIds.size,
        embeddingTrackIdAt = embeddingTrackIds::get,
    )

    internal fun firstMismatch(
        graphCount: Int,
        graphTrackIdAt: (Int) -> Long,
        embeddingCount: Int,
        embeddingTrackIdAt: (Int) -> Long,
    ): OrderedTrackIdMismatch? {
        require(graphCount >= 0) { "graphCount must be non-negative" }
        require(embeddingCount >= 0) { "embeddingCount must be non-negative" }

        val commonCount = minOf(graphCount, embeddingCount)
        for (index in 0 until commonCount) {
            val graphTrackId = graphTrackIdAt(index)
            val embeddingTrackId = embeddingTrackIdAt(index)
            if (graphTrackId != embeddingTrackId) {
                return OrderedTrackIdMismatch(index, graphTrackId, embeddingTrackId)
            }
        }
        if (graphCount == embeddingCount) return null

        return OrderedTrackIdMismatch(
            index = commonCount,
            graphTrackId = if (commonCount < graphCount) graphTrackIdAt(commonCount) else null,
            embeddingTrackId =
                if (commonCount < embeddingCount) embeddingTrackIdAt(commonCount) else null,
        )
    }
}
