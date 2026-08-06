package com.powerampstartradio.similarity.algorithms

import com.powerampstartradio.data.EmbeddingIndex
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Random
import kotlin.math.sqrt

class MmrSelectorTest {
    @get:Rule
    val temporaryFolder = TemporaryFolder()

    @Test
    fun temporarilyIneligibleArtistStillAccumulatesDiversityPenalty() {
        val ids = longArrayOf(1L, 2L, 3L, 4L, 5L, 6L)
        val embeddings = arrayOf(
            floatArrayOf(1f, 0f, 0f, 0f, 0f),
            floatArrayOf(0f, 1f, 0f, 0f, 0f),
            floatArrayOf(0f, 0f, 1f, 0f, 0f),
            floatArrayOf(0f, 0f, 0f, 1f, 0f),
            floatArrayOf(0f, 1f, 0f, 0f, 0f),
            floatArrayOf(0.2f, 0.2f, 0.2f, 0.2f, sqrt(0.84f)),
        )
        val index = EmbeddingIndex.mmap(writeIndex(ids, embeddings))
        val candidates = listOf(
            1L to 1.00f,
            2L to 0.95f,
            3L to 0.90f,
            4L to 0.85f,
            5L to 0.80f,
            6L to 0.60f,
        )
        val artist = mapOf(
            1L to "A",
            2L to "B",
            3L to "C",
            4L to "D",
            5L to "B",
            6L to "E",
        )

        val selected = MmrSelector.selectBatch(
            candidates = candidates,
            numSelect = 5,
            index = index,
            lambda = 0.5f,
            isEligible = { candidateId, selectedIds ->
                selectedIds.takeLast(2).none { selectedId ->
                    artist.getValue(selectedId) == artist.getValue(candidateId)
                }
            },
        )

        // Track 5 must retain its similarity of 1.0 to track 2 while artist B is blocked.
        assertEquals(listOf(1L, 2L, 3L, 4L, 6L), selected.map { it.trackId })
        assertNull(selected.first().mmrSelectionEvidence?.maximumSelectedTrackId)
        assertEquals(
            1L,
            selected[1].mmrSelectionEvidence?.maximumSelectedTrackId,
        )
        assertEquals(
            0f,
            selected[1].mmrSelectionEvidence?.maximumSelectedSimilarity ?: Float.NaN,
            0f,
        )
        selected.forEachIndexed { indexInQueue, pick ->
            val evidence = requireNotNull(pick.mmrSelectionEvidence)
            assertEquals(pick.trackId, evidence.trackId)
            assertEquals(indexInQueue + 1, evidence.step)
            assertEquals(
                0.5f * evidence.relevance -
                    0.5f * evidence.maximumSelectedSimilarity,
                evidence.objective,
                1e-6f,
            )
            evidence.maximumSelectedTrackId?.let { strongestPriorPick ->
                assertEquals(
                    true,
                    strongestPriorPick in selected.take(indexInQueue).map { it.trackId },
                )
            }
        }
    }

    @Test
    fun `pure relevance is invariant to wider distinct reach without eligibility filters`() {
        val ids = longArrayOf(1L, 2L, 3L, 4L, 5L, 6L)
        val embeddings = Array(ids.size) { index ->
            FloatArray(ids.size) { dimension -> if (index == dimension) 1f else 0f }
        }
        val index = EmbeddingIndex.mmap(writeIndex(ids, embeddings))
        val candidates = listOf(
            1L to 0.99f,
            2L to 0.95f,
            3L to 0.90f,
            4L to 0.85f,
            5L to 0.80f,
        )

        val narrowBatch = MmrSelector.selectBatch(
            candidates = candidates.take(3),
            numSelect = 3,
            index = index,
            lambda = 1f,
        )
        val wideBatch = MmrSelector.selectBatch(
            candidates = candidates,
            numSelect = 3,
            index = index,
            lambda = 1f,
        )
        assertEquals(listOf(1L, 2L, 3L), narrowBatch.map { it.trackId })
        assertEquals(narrowBatch.map { it.trackId }, wideBatch.map { it.trackId })

        val narrowStep = MmrSelector.selectOne(
            candidates = candidates.take(2),
            selectedTrackIds = listOf(6L),
            selectedEmbeddings = listOf(embeddings[5]),
            index = index,
            lambda = 1f,
        )
        val wideStep = MmrSelector.selectOne(
            candidates = candidates,
            selectedTrackIds = listOf(6L),
            selectedEmbeddings = listOf(embeddings[5]),
            index = index,
            lambda = 1f,
        )
        assertEquals(1L, narrowStep?.trackId)
        assertEquals(narrowStep?.trackId, wideStep?.trackId)
    }

    @Test
    fun `mapped incremental batch is exactly equivalent to allocation heavy reference`() {
        val random = Random(0x4d4d52L)
        val ids = LongArray(240) { it + 1L }
        val embeddings = Array(ids.size) { randomUnitVector(random, 37) }
        val index = EmbeddingIndex.mmap(writeIndex(ids, embeddings))
        val candidates = ids
            .mapIndexed { position, trackId ->
                trackId to (1f - position / ids.size.toFloat())
            }
            .take(180)
        val eligibility: (Long, List<Long>) -> Boolean = { candidateId, selectedIds ->
            selectedIds.takeLast(3).none { selectedId ->
                selectedId % 13L == candidateId % 13L
            }
        }

        val expected = legacyBatch(
            candidates = candidates,
            numSelect = 35,
            embeddingsById = ids.indices.associate { ids[it] to embeddings[it] },
            lambda = 0.43f,
            isEligible = eligibility,
        )
        val actual = MmrSelector.selectBatch(
            candidates = candidates,
            numSelect = 35,
            index = index,
            lambda = 0.43f,
            isEligible = eligibility,
        )

        assertSelectionsExactlyEqual(expected, actual)
    }

    @Test
    fun `incremental drift state exactly catches up changing candidate frontiers`() {
        val random = Random(0x4452494654L)
        val ids = LongArray(320) { it + 1L }
        val embeddings = Array(ids.size) { randomUnitVector(random, 29) }
        val embeddingsById = ids.indices.associate { ids[it] to embeddings[it] }
        val index = EmbeddingIndex.mmap(writeIndex(ids, embeddings))
        val state = MmrSelector.IncrementalState(index)
        val selectedIds = mutableListOf<Long>()
        val selectedEmbeddings = mutableListOf<FloatArray>()

        repeat(24) { step ->
            // The modular permutation makes rows leave and later re-enter the frontier. New rows
            // must be compared with every selection made while they were absent.
            val candidates = ids.indices
                .asSequence()
                .map { source -> (source * 73 + step * 41) % ids.size }
                .distinct()
                .map { row ->
                    val relevance = 1f - row / ids.size.toFloat() + step * 0.00001f
                    ids[row] to relevance
                }
                .filterNot { (trackId, _) -> trackId in selectedIds }
                .take(84)
                .toList()
            val eligible: (Long) -> Boolean = { candidateId ->
                selectedIds.takeLast(2).none { selectedId ->
                    selectedId % 11L == candidateId % 11L
                }
            }

            val expected = legacyOne(
                candidates = candidates,
                selectedTrackIds = selectedIds,
                selectedEmbeddings = selectedEmbeddings,
                embeddingsById = embeddingsById,
                lambda = 0.37f,
                isEligible = eligible,
            )
            val actual = state.selectOne(
                candidates = candidates,
                lambda = 0.37f,
                isEligible = eligible,
            )
            assertSelectionsExactlyEqual(listOfNotNull(expected), listOfNotNull(actual))

            val selected = requireNotNull(actual)
            selectedIds.add(selected.trackId)
            selectedEmbeddings.add(embeddingsById.getValue(selected.trackId))
            state.recordSelection(selected.trackId)
        }
    }

    private fun legacyBatch(
        candidates: List<Pair<Long, Float>>,
        numSelect: Int,
        embeddingsById: Map<Long, FloatArray>,
        lambda: Float,
        isEligible: (Long, List<Long>) -> Boolean,
    ): List<com.powerampstartradio.similarity.SelectedTrack> {
        val selectedIds = mutableListOf<Long>()
        val selectedEmbeddings = mutableListOf<FloatArray>()
        val remaining = candidates.toMutableList()
        val originalIndex = candidates.withIndex().associate { (index, candidate) ->
            candidate.first to index
        }
        val maximumSimilarity = FloatArray(candidates.size) { Float.NEGATIVE_INFINITY }
        val maximumTrackId = LongArray(candidates.size) { Long.MIN_VALUE }
        val result = mutableListOf<com.powerampstartradio.similarity.SelectedTrack>()

        repeat(numSelect) { step ->
            var bestRemainingIndex = -1
            var bestObjective = Float.NEGATIVE_INFINITY
            var bestPenalty = 0f

            for (remainingIndex in remaining.indices) {
                val (trackId, relevance) = remaining[remainingIndex]
                val embedding = embeddingsById[trackId] ?: continue
                val candidateIndex = originalIndex.getValue(trackId)
                if (selectedEmbeddings.isNotEmpty()) {
                    val similarity = scalarDot(embedding, selectedEmbeddings.last())
                    if (similarity > maximumSimilarity[candidateIndex]) {
                        maximumSimilarity[candidateIndex] = similarity
                        maximumTrackId[candidateIndex] = selectedIds.last()
                    }
                }
                if (!isEligible(trackId, selectedIds)) continue

                val penalty = if (selectedEmbeddings.isEmpty()) {
                    0f
                } else {
                    maximumSimilarity[candidateIndex]
                }
                val objective = lambda * relevance - (1f - lambda) * penalty
                if (objective > bestObjective) {
                    bestObjective = objective
                    bestRemainingIndex = remainingIndex
                    bestPenalty = penalty
                }
            }
            if (bestRemainingIndex < 0) return result

            val (trackId, relevance) = remaining.removeAt(bestRemainingIndex)
            val candidateIndex = originalIndex.getValue(trackId)
            val evidence = MmrSelectionEvidence(
                step = step + 1,
                trackId = trackId,
                relevance = relevance,
                maximumSelectedSimilarity = bestPenalty,
                maximumSelectedTrackId = maximumTrackId[candidateIndex]
                    .takeUnless { it == Long.MIN_VALUE },
                objective = bestObjective,
                candidateRank = candidateIndex + 1,
            )
            result.add(
                com.powerampstartradio.similarity.SelectedTrack(
                    trackId = trackId,
                    score = relevance,
                    candidateRank = candidateIndex + 1,
                    mmrSelectionEvidence = evidence,
                ),
            )
            selectedIds.add(trackId)
            selectedEmbeddings.add(embeddingsById.getValue(trackId))
        }
        return result
    }

    private fun legacyOne(
        candidates: List<Pair<Long, Float>>,
        selectedTrackIds: List<Long>,
        selectedEmbeddings: List<FloatArray>,
        embeddingsById: Map<Long, FloatArray>,
        lambda: Float,
        isEligible: (Long) -> Boolean,
    ): com.powerampstartradio.similarity.SelectedTrack? {
        if (selectedEmbeddings.isEmpty()) {
            val position = candidates.indexOfFirst { (trackId, _) ->
                trackId in embeddingsById && isEligible(trackId)
            }
            if (position < 0) return null
            val (trackId, relevance) = candidates[position]
            return com.powerampstartradio.similarity.SelectedTrack(
                trackId = trackId,
                score = relevance,
                candidateRank = position + 1,
                mmrSelectionEvidence = MmrSelectionEvidence(
                    step = 1,
                    trackId = trackId,
                    relevance = relevance,
                    maximumSelectedSimilarity = 0f,
                    maximumSelectedTrackId = null,
                    objective = lambda * relevance,
                    candidateRank = position + 1,
                ),
            )
        }

        var bestPosition = -1
        var bestObjective = Float.NEGATIVE_INFINITY
        var bestPenalty = 0f
        var bestMaximumTrackId: Long? = null
        for (position in candidates.indices) {
            val (trackId, relevance) = candidates[position]
            val embedding = embeddingsById[trackId] ?: continue
            if (!isEligible(trackId)) continue
            var maximumSimilarity = Float.NEGATIVE_INFINITY
            var maximumTrackId: Long? = null
            for (selectionIndex in selectedEmbeddings.indices) {
                val similarity = scalarDot(embedding, selectedEmbeddings[selectionIndex])
                if (similarity > maximumSimilarity) {
                    maximumSimilarity = similarity
                    maximumTrackId = selectedTrackIds[selectionIndex]
                }
            }
            val objective = lambda * relevance - (1f - lambda) * maximumSimilarity
            if (objective > bestObjective) {
                bestObjective = objective
                bestPosition = position
                bestPenalty = maximumSimilarity
                bestMaximumTrackId = maximumTrackId
            }
        }
        if (bestPosition < 0) return null
        val (trackId, relevance) = candidates[bestPosition]
        return com.powerampstartradio.similarity.SelectedTrack(
            trackId = trackId,
            score = relevance,
            candidateRank = bestPosition + 1,
            mmrSelectionEvidence = MmrSelectionEvidence(
                step = selectedEmbeddings.size + 1,
                trackId = trackId,
                relevance = relevance,
                maximumSelectedSimilarity = bestPenalty,
                maximumSelectedTrackId = bestMaximumTrackId,
                objective = bestObjective,
                candidateRank = bestPosition + 1,
            ),
        )
    }

    private fun assertSelectionsExactlyEqual(
        expected: List<com.powerampstartradio.similarity.SelectedTrack>,
        actual: List<com.powerampstartradio.similarity.SelectedTrack>,
    ) {
        assertEquals(expected.map { it.trackId }, actual.map { it.trackId })
        assertEquals(expected.map { it.candidateRank }, actual.map { it.candidateRank })
        expected.zip(actual).forEach { (expectedTrack, actualTrack) ->
            assertEquals(expectedTrack.score, actualTrack.score, 0f)
            val expectedEvidence = requireNotNull(expectedTrack.mmrSelectionEvidence)
            val actualEvidence = requireNotNull(actualTrack.mmrSelectionEvidence)
            assertEquals(expectedEvidence.step, actualEvidence.step)
            assertEquals(expectedEvidence.trackId, actualEvidence.trackId)
            assertEquals(expectedEvidence.relevance, actualEvidence.relevance, 0f)
            assertEquals(
                expectedEvidence.maximumSelectedSimilarity,
                actualEvidence.maximumSelectedSimilarity,
                0f,
            )
            assertEquals(
                expectedEvidence.maximumSelectedTrackId,
                actualEvidence.maximumSelectedTrackId,
            )
            assertEquals(expectedEvidence.objective, actualEvidence.objective, 0f)
            assertEquals(expectedEvidence.candidateRank, actualEvidence.candidateRank)
        }
    }

    private fun randomUnitVector(random: Random, dimension: Int): FloatArray {
        val vector = FloatArray(dimension) { (random.nextFloat() * 2f) - 1f }
        var squaredNorm = 0f
        for (value in vector) squaredNorm += value * value
        val norm = sqrt(squaredNorm)
        for (index in vector.indices) vector[index] /= norm
        return vector
    }

    private fun scalarDot(left: FloatArray, right: FloatArray): Float {
        var result = 0f
        for (dimension in left.indices) result += left[dimension] * right[dimension]
        return result
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
        return temporaryFolder.newFile("mmr.emb").apply { writeBytes(bytes.array()) }
    }
}
