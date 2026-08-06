package com.powerampstartradio.similarity

import com.powerampstartradio.data.EmbeddingIndex
import org.junit.Assert.assertSame
import org.junit.Assert.assertThrows
import org.junit.Test
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

class PreparedRecommendationEmbeddingIndexPolicyTest {
    @Test
    fun `exact pinned generation reuses the already mapped index`() {
        val file = writeIndex(longArrayOf(10L, 20L))
        val index = EmbeddingIndex.mmap(file)
        val prepared = PreparedRecommendationEmbeddingIndex(
            embeddingFile = file,
            activationBindingId = "activation-a",
            index = index,
        )

        val reused = PreparedRecommendationEmbeddingIndexPolicy.requireReusable(
            prepared = prepared,
            pinnedAssets = RecommendationAssetFiles(
                embeddingFile = file,
                graphFile = null,
                activationBindingId = "activation-a",
            ),
            databaseTrackCount = 2,
            headerTrackCount = 2,
        )

        assertSame(index, reused)
    }

    @Test
    fun `reuse rejects an unbound mismatched or differently pathed generation`() {
        val file = writeIndex(longArrayOf(10L, 20L))
        val otherFile = writeIndex(longArrayOf(10L, 20L))
        val prepared = PreparedRecommendationEmbeddingIndex(
            embeddingFile = file,
            activationBindingId = "activation-a",
            index = EmbeddingIndex.mmap(file),
        )

        assertThrows(IllegalArgumentException::class.java) {
            reuse(prepared, RecommendationAssetFiles(file, null))
        }
        assertThrows(IllegalArgumentException::class.java) {
            reuse(
                prepared,
                RecommendationAssetFiles(file, null, activationBindingId = "activation-b"),
            )
        }
        assertThrows(IllegalArgumentException::class.java) {
            reuse(
                prepared,
                RecommendationAssetFiles(otherFile, null, activationBindingId = "activation-a"),
            )
        }
    }

    @Test
    fun `reuse rejects every track count disagreement`() {
        val file = writeIndex(longArrayOf(10L, 20L))
        val prepared = PreparedRecommendationEmbeddingIndex(
            embeddingFile = file,
            activationBindingId = "activation-a",
            index = EmbeddingIndex.mmap(file),
        )
        val assets = RecommendationAssetFiles(file, null, activationBindingId = "activation-a")

        assertThrows(IllegalArgumentException::class.java) {
            reuse(prepared, assets, databaseTrackCount = 3, headerTrackCount = 3)
        }
        assertThrows(IllegalArgumentException::class.java) {
            reuse(prepared, assets, databaseTrackCount = 2, headerTrackCount = 3)
        }
    }

    private fun reuse(
        prepared: PreparedRecommendationEmbeddingIndex,
        assets: RecommendationAssetFiles,
        databaseTrackCount: Int = 2,
        headerTrackCount: Int = 2,
    ): EmbeddingIndex = PreparedRecommendationEmbeddingIndexPolicy.requireReusable(
        prepared = prepared,
        pinnedAssets = assets,
        databaseTrackCount = databaseTrackCount,
        headerTrackCount = headerTrackCount,
    )

    private fun writeIndex(trackIds: LongArray): File {
        val dimension = 2
        val bytes = ByteBuffer.allocate(16 + trackIds.size * 8 + trackIds.size * dimension * 4)
            .order(ByteOrder.LITTLE_ENDIAN)
        bytes.putInt(0x424D4550)
        bytes.putInt(1)
        bytes.putInt(trackIds.size)
        bytes.putInt(dimension)
        trackIds.forEach(bytes::putLong)
        trackIds.indices.forEach { position ->
            bytes.putFloat(if (position == 0) 1f else 0f)
            bytes.putFloat(if (position == 0) 0f else 1f)
        }
        return File.createTempFile("prepared-recommendation-", ".emb").apply {
            deleteOnExit()
            writeBytes(bytes.array())
        }
    }
}
