package com.powerampstartradio.similarity

import com.powerampstartradio.data.EmbeddingIndex
import com.powerampstartradio.ui.FindMusicTextResultPlanner
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertThrows
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class FindMusicTextQueuePlannerTest {
    @get:Rule
    val temporaryFolder = TemporaryFolder()

    @Test
    fun variedUsesCertifiedFullDomainDppAndPreservesOriginalTextRanks() {
        val index = testIndex()
        val ranking = ranking()

        val first = FindMusicTextQueuePlanner.plan(
            planner = FindMusicTextResultPlanner.VARIED_DPP,
            completeRelevanceRanking = ranking,
            requestedResultCount = 2,
            embeddingIndex = index,
        )
        val repeated = FindMusicTextQueuePlanner.plan(
            planner = FindMusicTextResultPlanner.VARIED_DPP,
            completeRelevanceRanking = ranking,
            requestedResultCount = 2,
            embeddingIndex = index,
        )

        assertEquals(listOf(1L, 3L), first.selections.map { it.trackId })
        assertEquals(listOf(1, 3), first.selections.map { it.originalTextObjectiveRank })
        assertEquals(first, repeated)
        assertEquals(4, first.evidence.completeCandidateDomainCount)
        assertEquals(FindMusicTextResultPlanner.VARIED_DPP, first.evidence.planner)
        assertEquals(FindMusicTextResultPlanner.VARIED_DPP.currentVersion,
            first.evidence.plannerVersion
        )
        assertEquals(listOf(4), first.evidence.dppSelection?.attemptedCandidateCounts)
        assertEquals(4, first.evidence.dppSelection?.initialWorkingCandidateCount)
        assertEquals(4, first.evidence.dppSelection?.finalWorkingCandidateCount)
        assertEquals(true, first.evidence.dppSelection?.usedCompleteCandidateDomain)
        assertEquals(true, first.evidence.dppSelection?.reproducedFullDomainGreedySequence)
        assertEquals(
            first.selections.size,
            first.evidence.dppSelection?.selectedMarginalGains?.size,
        )
        assertEquals(
            true,
            first.evidence.dppSelection?.selectedMarginalGains?.all { gain ->
                gain.isFinite() && gain > 0.0
            },
        )
        assertNotNull(first.evidence.dppSelection)
    }

    @Test
    fun closestRemainsTheDefaultObjectiveOrderWithoutDppEvidence() {
        val plan = FindMusicTextQueuePlanner.plan(
            planner = FindMusicTextResultPlanner.CLOSEST,
            completeRelevanceRanking = ranking(),
            requestedResultCount = 2,
            embeddingIndex = testIndex(),
        )

        assertEquals(listOf(1L, 2L), plan.selections.map { it.trackId })
        assertEquals(listOf(1, 2), plan.selections.map { it.originalTextObjectiveRank })
        assertEquals(null, plan.evidence.dppSelection)
        assertEquals(64, plan.evidence.completeTextRankingSha256.length)
        assertFalse(plan.evidence.completeTextRankingSha256.any { it !in '0'..'9' && it !in 'a'..'f' })
    }

    @Test
    fun productionSizedInitialWorkspaceCertifiesAgainstTheCompleteDomain() {
        val count = 80_323
        val ids = LongArray(count) { index -> index + 1L }
        val index = EmbeddingIndex.mmap(
            writeIndex(ids, Array(count) { floatArrayOf(1f) }),
        )
        val ranking = ids.mapIndexed { position, id ->
            RankedSimilarity(id, 1f - position.toFloat() / (count * 2f))
        }

        val plan = FindMusicTextQueuePlanner.plan(
            planner = FindMusicTextResultPlanner.VARIED_DPP,
            completeRelevanceRanking = ranking,
            requestedResultCount = 1,
            embeddingIndex = index,
        )

        val proof = requireNotNull(plan.evidence.dppSelection)
        assertEquals(80_323, proof.completeCandidateDomainCount)
        assertEquals(1_607, proof.initialWorkingCandidateCount)
        assertEquals(listOf(1_607), proof.attemptedCandidateCounts)
        assertEquals(1_607, proof.finalWorkingCandidateCount)
        assertFalse(proof.usedCompleteCandidateDomain)
        assertEquals(true, proof.reproducedFullDomainGreedySequence)
        assertEquals(1, proof.selectedMarginalGains.size)
        assertEquals(
            true,
            proof.selectedMarginalGains.all { gain ->
                gain > requireNotNull(proof.finalUnseenInitialGainUpperBound)
            },
        )
    }

    @Test
    fun uncertifiedInitialWorkspaceExpandsUntilFullDomainTieIsResolved() {
        val count = 100
        val ids = LongArray(count) { index -> index + 1L }
        val embeddings = Array(count) { row ->
            FloatArray(count).also { it[row] = 1f }
        }
        val index = EmbeddingIndex.mmap(writeIndex(ids, embeddings))
        val ranking = ids.map { id -> RankedSimilarity(id, 0.8f) }

        val plan = FindMusicTextQueuePlanner.plan(
            planner = FindMusicTextResultPlanner.VARIED_DPP,
            completeRelevanceRanking = ranking,
            requestedResultCount = 2,
            embeddingIndex = index,
        )

        val proof = requireNotNull(plan.evidence.dppSelection)
        assertEquals(listOf(50, 100), proof.attemptedCandidateCounts)
        assertEquals(100, proof.finalWorkingCandidateCount)
        assertEquals(true, proof.usedCompleteCandidateDomain)
        assertEquals(2, proof.selectedMarginalGains.size)
        assertEquals(listOf(1L, 2L), plan.selections.map { it.trackId })
    }

    @Test
    fun variedEvidenceRejectsIncompleteNonFiniteNonPositiveOrUncertifiedGains() {
        val valid = FindMusicTextQueuePlanner.plan(
            planner = FindMusicTextResultPlanner.VARIED_DPP,
            completeRelevanceRanking = productionSizedRanking(100),
            requestedResultCount = 1,
            embeddingIndex = constantIndex(100),
        ).evidence
        val proof = requireNotNull(valid.dppSelection)
        val unseenBound = requireNotNull(proof.finalUnseenInitialGainUpperBound)

        val invalidGainLists: List<List<Double>> = listOf(
            emptyList(),
            listOf(Double.NaN),
            listOf(Double.POSITIVE_INFINITY),
            listOf(0.0),
            listOf(unseenBound),
        )
        invalidGainLists.forEach { invalidGains ->
            assertThrows(IllegalArgumentException::class.java) {
                valid.copy(
                    dppSelection = proof.copy(selectedMarginalGains = invalidGains),
                )
            }
        }
    }

    private fun ranking() = listOf(
        RankedSimilarity(1L, 1f),
        RankedSimilarity(2L, 0.95f),
        RankedSimilarity(3L, 0.8f),
        RankedSimilarity(4L, 0.1f),
    )

    private fun productionSizedRanking(count: Int): List<RankedSimilarity> =
        (1..count).map { position ->
            RankedSimilarity(position.toLong(), 1f - position.toFloat() / (count * 2f))
        }

    private fun constantIndex(count: Int): EmbeddingIndex = EmbeddingIndex.mmap(
        writeIndex(
            ids = LongArray(count) { index -> index + 1L },
            embeddings = Array(count) { floatArrayOf(1f) },
        ),
    )

    private fun testIndex(): EmbeddingIndex = EmbeddingIndex.mmap(
        writeIndex(
            ids = longArrayOf(1L, 2L, 3L, 4L),
            embeddings = arrayOf(
                floatArrayOf(1f, 0f),
                floatArrayOf(1f, 0f),
                floatArrayOf(0f, 1f),
                floatArrayOf(0.70710677f, 0.70710677f),
            ),
        ),
    )

    private fun writeIndex(ids: LongArray, embeddings: Array<FloatArray>): File {
        val dimension = embeddings.first().size
        val bytes = ByteBuffer.allocate(
            16 + ids.size * Long.SIZE_BYTES + ids.size * dimension * Float.SIZE_BYTES,
        ).order(ByteOrder.LITTLE_ENDIAN)
        bytes.putInt(0x424D4550)
        bytes.putInt(1)
        bytes.putInt(ids.size)
        bytes.putInt(dimension)
        ids.forEach(bytes::putLong)
        embeddings.forEach { row -> row.forEach(bytes::putFloat) }
        return temporaryFolder.newFile("find-music-text-planner.emb").apply {
            writeBytes(bytes.array())
        }
    }
}
