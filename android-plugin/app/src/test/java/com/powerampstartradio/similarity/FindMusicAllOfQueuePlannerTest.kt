package com.powerampstartradio.similarity

import com.powerampstartradio.data.EmbeddingIndex
import com.powerampstartradio.similarity.algorithms.ComposedRankingRow
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

class FindMusicAllOfQueuePlannerTest {
    @get:Rule
    val temporaryFolder = TemporaryFolder()

    @Test
    fun plannerAndPersistedQueryVersionsStayAligned() {
        assertEquals(
            FindMusicTextResultPlanner.VARIED_ALL_OF_DPP.currentVersion,
            FindMusicAllOfQueuePlanner.PLANNER_VERSION,
        )
    }

    @Test
    fun variedAllOfIsDeterministicAndPreservesOriginalObjectiveRanks() {
        val ranking = listOf(
            row(1L, 1f),
            row(2L, 0.995f),
            row(3L, 0.99f),
            row(4L, 0.9f),
        )
        val index = testIndex()

        val first = FindMusicAllOfQueuePlanner.plan(ranking, 2, index)
        val repeated = FindMusicAllOfQueuePlanner.plan(ranking, 2, index)

        assertEquals(first, repeated)
        assertEquals(listOf(1L, 3L), first.selections.map { it.row.trackId })
        assertEquals(
            listOf(1, 3),
            first.selections.map { it.originalAllOfObjectiveRank },
        )
        assertEquals(FindMusicAllOfQueuePlanner.PLANNER_VERSION, first.evidence.plannerVersion)
        assertEquals(4, first.evidence.completeCandidateDomainCount)
        assertEquals(
            first.selections.size,
            first.evidence.dppSelection.selectedMarginalGains.size,
        )
        assertEquals(true, first.evidence.dppSelection.reproducedFullDomainGreedySequence)
        assertNotNull(first.evidence.dppSelection)
        assertEquals(64, first.evidence.completeAllOfRankingSha256.length)
        assertFalse(
            first.evidence.completeAllOfRankingSha256.any {
                it !in '0'..'9' && it !in 'a'..'f'
            },
        )
    }

    @Test
    fun productionExponentFillsLowAbsoluteScoreQueueAfterDomainNormalization() {
        val ranking = listOf(
            row(1L, 0.814697f),
            row(2L, 0.81f),
            row(3L, 0.79f),
            row(4L, 0.76f),
        )

        val plan = FindMusicAllOfQueuePlanner.plan(
            completeObjectiveRanking = ranking,
            requestedResultCount = 3,
            embeddingIndex = testIndex(),
        )

        assertEquals(3, plan.selections.size)
        assertEquals(listOf(1L, 3L, 4L), plan.selections.map { it.row.trackId })
        assertEquals(3, plan.evidence.dppSelection.selectedMarginalGains.size)
        assertEquals(
            true,
            plan.evidence.dppSelection.selectedMarginalGains.all { it > 0.0 },
        )
    }

    @Test
    fun variedAllOfEvidenceRejectsInconsistentWorkingDomainProof() {
        val valid = FindMusicAllOfQueuePlanner.plan(
            completeObjectiveRanking = listOf(
                row(1L, 1f),
                row(2L, 0.995f),
                row(3L, 0.99f),
                row(4L, 0.9f),
            ),
            requestedResultCount = 2,
            embeddingIndex = testIndex(),
        ).evidence.dppSelection

        assertThrows(IllegalArgumentException::class.java) {
            valid.copy(
                finalWorkingCandidateCount = valid.completeCandidateDomainCount - 1,
            )
        }
        assertThrows(IllegalArgumentException::class.java) {
            valid.copy(reproducedFullDomainGreedySequence = false)
        }
    }

    private fun row(id: Long, score: Float) = ComposedRankingRow(
        trackId = id,
        objectiveScore = score,
        anchorPercentiles = listOf(score, score),
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
        return temporaryFolder.newFile("find-music-all-of-planner.emb").apply {
            writeBytes(bytes.array())
        }
    }
}
