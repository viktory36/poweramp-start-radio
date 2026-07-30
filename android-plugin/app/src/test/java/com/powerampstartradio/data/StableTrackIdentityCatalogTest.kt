package com.powerampstartradio.data

import com.powerampstartradio.indexing.v2.StableTrackSpanIdentityStrength
import com.powerampstartradio.similarity.algorithms.UniformShuffleSelector
import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertThrows
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class StableTrackIdentityCatalogTest {
    @get:Rule
    val temporaryFolder = TemporaryFolder()

    @Test
    fun stableIdentitySurvivesRowReplacementAndReordering() {
        val first = catalog(
            binding("generation-a"),
            stableRow(10, 1),
            stableRow(20, 2),
            stableRow(30, 3),
        )
        val replacement = catalog(
            binding("generation-b"),
            stableRow(101, 3),
            stableRow(102, 1),
            stableRow(103, 2),
        )

        val firstOrder = shuffledStableIds(first)
        val replacementOrder = shuffledStableIds(replacement)

        assertEquals(firstOrder, replacementOrder)
        assertEquals(
            StableTrackIdentityResolution.Resolved(102, listOf(102L)),
            replacement.resolveStable(stableId(1)),
        )
    }

    @Test
    fun duplicateStableIdentityIsOneEquivalentAcousticMember() {
        val catalog = catalog(
            binding("duplicates"),
            stableRow(1, 9),
            stableRow(2, 9),
        )

        assertEquals(
            StableTrackIdentityResolution.Resolved(1, listOf(1L, 2L)),
            catalog.resolveStable(stableId(9)),
        )
        assertEquals(1, catalog.duplicateExcessCount)
        assertEquals(2, catalog.rankedRowsForVisibleCount(1))
        assertEquals(true, catalog.visibleResultIdentity(1).isCollapsibleRecording)
        val shuffled = UniformShuffleSelector.select(
            trackIds = catalog.orderedTrackIds(),
            numSelect = 2,
            seed = 12,
            identityKeyForTrack = catalog::shuffleIdentityKey,
        )
        assertEquals(listOf(1L), shuffled.map { it.trackId })
    }

    @Test
    fun legacyResolutionRequiresTheExactGenerationBinding() {
        val expectedBinding = binding("legacy-a")
        val catalog = catalog(expectedBinding, legacyRow(41))

        assertEquals(
            StableTrackIdentityResolution.LegacyBindingRequired,
            catalog.resolveLegacy(41, null),
        )
        assertEquals(
            StableTrackIdentityResolution.LegacyBindingMismatch,
            catalog.resolveLegacy(41, binding("legacy-b")),
        )
        assertEquals(
            StableTrackIdentityResolution.Resolved(41, listOf(41L)),
            catalog.resolveLegacy(41, expectedBinding),
        )
    }

    @Test
    fun legacyShuffleKeysAreScopedToTheGeneration() {
        val first = catalog(binding("legacy-a"), legacyRow(7))
        val second = catalog(binding("legacy-b"), legacyRow(7))

        assertNotEquals(first.shuffleIdentityKey(7), second.shuffleIdentityKey(7))
        assertEquals(false, first.shuffleIdentityKey(7).isStableAcrossGenerations)
    }

    @Test
    fun confirmedGenerationLocalCopiesCollapseWithoutBecomingDurableIdentity() {
        val rows = arrayOf(legacyRow(7), legacyRow(8), legacyRow(9))
        val catalog = StableTrackIdentityCatalog.fromOrderedRows(
            binding = binding("confirmed-copies"),
            orderedEmbeddingTrackIds = rows.map(StableTrackIdentityRow::trackId).toLongArray(),
            rows = rows.toList(),
            queueDuplicatePairs = listOf(7L to 8L, 7L to 9L, 8L to 9L),
        )

        assertEquals(2, catalog.duplicateExcessCount)
        assertEquals(listOf(7L, 8L, 9L), catalog.equivalentVisibleTrackIds(7))
        assertEquals(catalog.visibleResultIdentity(7), catalog.visibleResultIdentity(9))
        assertEquals(false, catalog.shuffleIdentityKey(7).isStableAcrossGenerations)
        assertEquals(StableTrackIdentityResolution.Missing, catalog.resolveStable(stableId(7)))
    }

    @Test
    fun measuredAceDuplicateDeltaIsInsideFinalLegacyBound() {
        val rows = listOf(duplicateLegacyRow(1), duplicateLegacyRow(2))

        assertEquals(
            listOf(1L to 2L),
            generatedQueueDuplicatePairs(
                rows,
                arrayOf(floatArrayOf(0f), floatArrayOf(0.0000615492463f)),
            ),
        )
        assertEquals(
            emptyList<Pair<Long, Long>>(),
            generatedQueueDuplicatePairs(
                rows,
                arrayOf(floatArrayOf(0f), floatArrayOf(0.0000621f)),
            ),
        )
    }

    @Test
    fun v2ReceiptedRowsNeverUseApproximateQueueIdentity() {
        val rows = listOf(
            duplicateStableRow(1, 1),
            duplicateStableRow(2, 2),
        )

        assertEquals(
            emptyList<Pair<Long, Long>>(),
            generatedQueueDuplicatePairs(
                rows,
                arrayOf(floatArrayOf(0f), floatArrayOf(0f)),
            ),
        )
    }

    @Test
    fun incompleteThreeRowThresholdBridgeStaysFullyVisible() {
        val rows = listOf(
            duplicateLegacyRow(1),
            duplicateLegacyRow(2),
            duplicateLegacyRow(3),
        )

        assertEquals(
            emptyList<Pair<Long, Long>>(),
            generatedQueueDuplicatePairs(
                rows,
                arrayOf(
                    floatArrayOf(0f),
                    floatArrayOf(0.00004f),
                    floatArrayOf(0.00008f),
                ),
            ),
        )
        assertThrows(IllegalArgumentException::class.java) {
            StableTrackIdentityCatalog.fromOrderedRows(
                binding = binding("incomplete-component"),
                orderedEmbeddingTrackIds = longArrayOf(1, 2, 3),
                rows = rows,
                queueDuplicatePairs = listOf(1L to 2L, 2L to 3L),
            )
        }
    }

    @Test
    fun generationLocalCopyOutsideActiveRowsFailsClosed() {
        assertThrows(IllegalArgumentException::class.java) {
            StableTrackIdentityCatalog.fromOrderedRows(
                binding = binding("bad-copy"),
                orderedEmbeddingTrackIds = longArrayOf(7),
                rows = listOf(legacyRow(7)),
                queueDuplicatePairs = listOf(7L to 8L),
            )
        }
    }

    @Test
    fun sampledReceiptIsNotCollapsedOrResolvedAcrossGenerations() {
        val first = stableRow(1, 7).copy(
            stableIdentityStrength = StableTrackSpanIdentityStrength.VERSIONED_SAMPLED_CONTENT_SHA256,
        )
        val second = first.copy(trackId = 2)
        val catalog = catalog(binding("sampled"), first, second)

        assertEquals(0, catalog.stableTrackCount)
        assertEquals(2, catalog.legacyTrackCount)
        assertEquals(0, catalog.duplicateExcessCount)
        assertEquals(StableTrackIdentityResolution.Missing, catalog.resolveStable(stableId(7)))
        assertEquals(false, catalog.visibleResultIdentity(1).isCollapsibleRecording)
        assertNotEquals(catalog.shuffleIdentityKey(1), catalog.shuffleIdentityKey(2))
    }

    @Test
    fun pembAndDatabaseOrderMustAlignExactly() {
        assertThrows(IllegalArgumentException::class.java) {
            StableTrackIdentityCatalog.fromOrderedRows(
                binding = binding("misaligned"),
                orderedEmbeddingTrackIds = longArrayOf(2, 1),
                rows = listOf(legacyRow(1), legacyRow(2)),
            )
        }
    }

    @Test
    fun equivalentIdentityWithDifferentEmbeddingReceiptFailsClosed() {
        assertThrows(IllegalArgumentException::class.java) {
            catalog(
                binding("bad-equivalence"),
                stableRow(1, 4),
                stableRow(2, 4).copy(embeddingSha256 = "f".repeat(64)),
            )
        }
    }

    private fun shuffledStableIds(catalog: StableTrackIdentityCatalog): List<String?> =
        UniformShuffleSelector.select(
            trackIds = catalog.orderedTrackIds(),
            numSelect = catalog.trackCount,
            seed = 0x5eed,
            identityKeyForTrack = catalog::shuffleIdentityKey,
        ).map { catalog.stableTrackSpanId(it.trackId) }

    private fun catalog(
        binding: StableIdentityGenerationBinding,
        vararg rows: StableTrackIdentityRow,
    ): StableTrackIdentityCatalog = StableTrackIdentityCatalog.fromOrderedRows(
        binding = binding,
        orderedEmbeddingTrackIds = rows.map { it.trackId }.toLongArray(),
        rows = rows.toList(),
    )

    private fun stableRow(trackId: Long, identityNumber: Int) = StableTrackIdentityRow(
        trackId = trackId,
        stableTrackSpanId = stableId(identityNumber),
        stableIdentitySpecId = STABLE_SPEC,
        stableIdentityStrength = StableTrackSpanIdentityStrength.FULL_CONTENT_SHA256,
        embeddingSpecId = "embedding-spec-test",
        embeddingSha256 = identityNumber.toString(16).padStart(64, '0'),
    )

    private fun duplicateStableRow(trackId: Long, identityNumber: Int) =
        stableRow(trackId, identityNumber).copy(
            artist = "Lsdream",
            album = "Renegades of Light",
            title = "Ace of Cups",
            durationMs = 179_349,
        )

    private fun legacyRow(trackId: Long) = StableTrackIdentityRow(
        trackId = trackId,
        stableTrackSpanId = null,
        stableIdentitySpecId = null,
        stableIdentityStrength = null,
        embeddingSpecId = null,
        embeddingSha256 = null,
    )

    private fun duplicateLegacyRow(trackId: Long) = legacyRow(trackId).copy(
        artist = "Lsdream",
        album = "Renegades of Light",
        title = "Ace of Cups",
        durationMs = 179_349,
    )

    private fun generatedQueueDuplicatePairs(
        rows: List<StableTrackIdentityRow>,
        embeddings: Array<FloatArray>,
    ): List<Pair<Long, Long>> {
        val index = EmbeddingIndex.mmap(
            writeIndex(rows.map(StableTrackIdentityRow::trackId).toLongArray(), embeddings),
        )
        return StableTrackIdentityCatalog.generationQueueDuplicatePairs(rows, index)
    }

    private fun writeIndex(ids: LongArray, embeddings: Array<FloatArray>): java.io.File {
        require(ids.size == embeddings.size && embeddings.isNotEmpty())
        val dimension = embeddings.first().size
        require(embeddings.all { it.size == dimension })
        val bytes = ByteBuffer.allocate(
            16 + ids.size * Long.SIZE_BYTES +
                ids.size * dimension * Float.SIZE_BYTES,
        ).order(ByteOrder.LITTLE_ENDIAN)
        bytes.putInt(0x424D4550)
        bytes.putInt(1)
        bytes.putInt(ids.size)
        bytes.putInt(dimension)
        ids.forEach(bytes::putLong)
        embeddings.forEach { row -> row.forEach(bytes::putFloat) }
        return temporaryFolder.newFile().apply { writeBytes(bytes.array()) }
    }

    private fun stableId(number: Int): String =
        "stable-track-span-v1-${number.toString(16).padStart(64, '0')}"

    private fun binding(id: String) = StableIdentityGenerationBinding(
        bindingSpecId = "test-binding-v1",
        generationId = id,
        activationBindingId = id,
        databaseContentSha256 = id.padEnd(64, '0'),
        orderedTrackSetSha256 = id.padEnd(64, '1'),
    )

    companion object {
        private const val STABLE_SPEC =
            "stable-track-span-v1:content-sha256:native-half-open-sample-span"
    }
}
