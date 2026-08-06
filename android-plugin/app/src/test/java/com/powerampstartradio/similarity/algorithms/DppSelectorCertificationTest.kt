package com.powerampstartradio.similarity.algorithms

import com.powerampstartradio.data.EmbeddingIndex
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.CancellationException

class DppSelectorCertificationTest {
    @get:Rule
    val temporaryFolder = TemporaryFolder()

    @Test
    fun certifiedPrefixMatchesCompleteDomainGreedySequence() {
        val ids = longArrayOf(1L, 2L, 3L, 4L, 5L, 6L)
        val index = EmbeddingIndex.mmap(
            writeIndex(
                ids = ids,
                embeddings = arrayOf(
                    floatArrayOf(1f, 0f, 0f, 0f),
                    floatArrayOf(1f, 0f, 0f, 0f),
                    floatArrayOf(0f, 1f, 0f, 0f),
                    floatArrayOf(0f, 0f, 1f, 0f),
                    floatArrayOf(0f, 0f, 0f, 1f),
                    floatArrayOf(0.5f, 0.5f, 0.5f, 0.5f),
                ),
            ),
        )
        val candidates = listOf(
            1L to 1f,
            2L to 0.95f,
            3L to 0.8f,
            4L to 0.7f,
            5L to 0.1f,
            6L to 0.05f,
        )

        val complete = DppSelector.selectBatch(candidates, 3, index)
        val certified = DppSelector.selectBatchCertified(
            candidates = candidates,
            numSelect = 3,
            index = index,
            initialCandidateCount = 2,
        )

        assertEquals(listOf(1L, 3L, 4L), complete.map { it.trackId })
        assertEquals(complete, certified.tracks)
        assertEquals(listOf(2, 4), certified.evidence.attemptedCandidateCounts)
        assertEquals(4, certified.evidence.finalCandidateCount)
        assertEquals(0.01, certified.evidence.finalUnseenGainUpperBound!!, 1e-7)
        assertFalse(certified.evidence.usedFullDomain)
    }

    @Test
    fun highExponentNormalizesCompleteDomainAndFillsRequestedCardinality() {
        val ids = longArrayOf(1L, 2L, 3L, 4L)
        val index = EmbeddingIndex.mmap(
            writeIndex(
                ids = ids,
                embeddings = Array(ids.size) { row ->
                    FloatArray(ids.size).also { it[row] = 1f }
                },
            ),
        )
        val candidates = listOf(
            1L to 0.80f,
            2L to 0.79f,
            3L to 0.78f,
            4L to 0.77f,
        )

        val certified = DppSelector.selectBatchCertified(
            candidates = candidates,
            numSelect = 3,
            index = index,
            initialCandidateCount = 2,
            qualityExponent = 64f,
        )

        assertEquals(3, certified.tracks.size)
        assertEquals(
            DppSelector.selectBatch(
                candidates = candidates,
                numSelect = 3,
                index = index,
                qualityExponent = 64f,
            ),
            certified.tracks,
        )
        assertTrue(certified.evidence.selectedMarginalGains.all { it > 0.0 })
    }

    @Test
    fun tinyUnseenGainCannotCertifyAnUnderfilledPrefix() {
        val ids = longArrayOf(1L, 2L, 3L, 4L)
        val index = EmbeddingIndex.mmap(
            writeIndex(
                ids = ids,
                embeddings = arrayOf(
                    floatArrayOf(1f, 0f, 0f),
                    floatArrayOf(1f, 0f, 0f),
                    floatArrayOf(0f, 1f, 0f),
                    floatArrayOf(0f, 0f, 1f),
                ),
            ),
        )
        val candidates = listOf(
            1L to 1f,
            2L to 1f,
            3L to 0.5f,
            4L to 0.5f,
        )

        val certified = DppSelector.selectBatchCertified(
            candidates = candidates,
            numSelect = 3,
            index = index,
            initialCandidateCount = 2,
            qualityExponent = 64f,
        )

        assertEquals(listOf(2, 4), certified.evidence.attemptedCandidateCounts)
        assertTrue(certified.evidence.usedFullDomain)
        assertEquals(listOf(1L, 3L, 4L), certified.tracks.map { it.trackId })
        assertEquals(3, certified.evidence.selectedMarginalGains.size)
        assertTrue(certified.evidence.selectedMarginalGains.all { it > 0.0 })
    }

    @Test
    fun equalUnseenUpperBoundForcesExpansionForStableTieHandling() {
        val ids = longArrayOf(1L, 2L, 3L, 4L)
        val index = EmbeddingIndex.mmap(
            writeIndex(
                ids = ids,
                embeddings = arrayOf(
                    floatArrayOf(1f, 0f, 0f, 0f),
                    floatArrayOf(0f, 1f, 0f, 0f),
                    floatArrayOf(0f, 0f, 1f, 0f),
                    floatArrayOf(0f, 0f, 0f, 1f),
                ),
            ),
        )
        val candidates = ids.map { it to 0.8f }

        val certified = DppSelector.selectBatchCertified(
            candidates = candidates,
            numSelect = 2,
            index = index,
            initialCandidateCount = 2,
        )

        assertEquals(listOf(2, 4), certified.evidence.attemptedCandidateCounts)
        assertTrue(certified.evidence.usedFullDomain)
        assertEquals(listOf(1L, 2L), certified.tracks.map { it.trackId })
    }

    @Test
    fun zeroExponentFallsBackToCompleteDomain() {
        val ids = longArrayOf(1L, 2L, 3L, 4L, 5L)
        val index = EmbeddingIndex.mmap(
            writeIndex(
                ids = ids,
                embeddings = Array(ids.size) { row ->
                    FloatArray(ids.size).also { it[row] = 1f }
                },
            ),
        )
        val candidates = ids.mapIndexed { position, id ->
            id to (1f - position * 0.1f)
        }

        val certified = DppSelector.selectBatchCertified(
            candidates = candidates,
            numSelect = 3,
            index = index,
            initialCandidateCount = 2,
            qualityExponent = 0f,
        )

        assertEquals(listOf(5), certified.evidence.attemptedCandidateCounts)
        assertTrue(certified.evidence.usedFullDomain)
        assertEquals(listOf(1L, 2L, 3L), certified.tracks.map { it.trackId })
    }

    @Test
    fun missingEmbeddingForcesConservativeExpansionAndPreservesFullRank() {
        val index = EmbeddingIndex.mmap(
            writeIndex(
                ids = longArrayOf(1L, 2L, 3L),
                embeddings = arrayOf(
                    floatArrayOf(1f, 0f, 0f),
                    floatArrayOf(0f, 1f, 0f),
                    floatArrayOf(0f, 0f, 1f),
                ),
            ),
        )
        val candidates = listOf(1L to 1f, 99L to 0.9f, 2L to 0.8f, 3L to 0.1f)

        val certified = DppSelector.selectBatchCertified(
            candidates = candidates,
            numSelect = 2,
            index = index,
            initialCandidateCount = 1,
        )

        assertEquals(
            DppSelector.selectBatch(candidates, 2, index),
            certified.tracks,
        )
        assertEquals(listOf(1L, 2L), certified.tracks.map { it.trackId })
        assertEquals(listOf(1, 3), certified.tracks.map { it.candidateRank })
        assertEquals(listOf(1, 2, 4), certified.evidence.attemptedCandidateCounts)
    }

    @Test
    fun spacingEligibilityCanReenableCandidateAcrossCertifiedReruns() {
        val ids = longArrayOf(1L, 2L, 3L, 4L)
        val index = EmbeddingIndex.mmap(
            writeIndex(
                ids = ids,
                embeddings = Array(ids.size) { row ->
                    FloatArray(ids.size).also { it[row] = 1f }
                },
            ),
        )
        val candidates = listOf(1L to 1f, 2L to 0.9f, 3L to 0.8f, 4L to 0.1f)
        val artist = mapOf(1L to "A", 2L to "B", 3L to "A", 4L to "C")
        val eligibility: (Long, List<Long>) -> Boolean = { candidateId, selectedIds ->
            selectedIds.takeLast(1).none { selectedId ->
                artist.getValue(selectedId) == artist.getValue(candidateId)
            }
        }

        val certified = DppSelector.selectBatchCertified(
            candidates = candidates,
            numSelect = 3,
            index = index,
            initialCandidateCount = 2,
            isEligible = eligibility,
        )

        assertEquals(
            DppSelector.selectBatch(candidates, 3, index, isEligible = eligibility),
            certified.tracks,
        )
        assertEquals(listOf(1L, 2L, 3L), certified.tracks.map { it.trackId })
    }

    @Test
    fun emptyRequestDoesNotClaimToHaveScannedFullDomain() {
        val index = EmbeddingIndex.mmap(
            writeIndex(
                ids = longArrayOf(1L),
                embeddings = arrayOf(floatArrayOf(1f)),
            ),
        )

        val certified = DppSelector.selectBatchCertified(
            candidates = listOf(1L to 1f),
            numSelect = 0,
            index = index,
            initialCandidateCount = 1,
        )

        assertTrue(certified.tracks.isEmpty())
        assertFalse(certified.evidence.usedFullDomain)
        assertEquals(0, certified.evidence.finalCandidateCount)
    }

    @Test
    fun cancellationPropagatesDuringCandidateScan() {
        val ids = LongArray(2_048) { position -> position + 1L }
        val index = EmbeddingIndex.mmap(
            writeIndex(
                ids = ids,
                embeddings = Array(ids.size) { floatArrayOf(1f) },
            ),
        )
        var checks = 0

        assertThrows(CancellationException::class.java) {
            DppSelector.selectBatch(
                candidates = ids.map { it to 1f },
                numSelect = 30,
                index = index,
                cancellationCheck = {
                    checks++
                    if (checks == 3) throw CancellationException("test cancellation")
                },
            )
        }
        assertEquals(3, checks)
    }

    private fun writeIndex(ids: LongArray, embeddings: Array<FloatArray>): File {
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
        return temporaryFolder.newFile("certified-dpp.emb").apply {
            writeBytes(bytes.array())
        }
    }
}
