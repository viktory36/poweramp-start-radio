package com.powerampstartradio.data

import android.util.Log
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertEquals
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import kotlin.system.measureNanoTime

/** Corpus-gated device evidence for choosing DPP's mmap similarity-column strategy. */
@RunWith(AndroidJUnit4::class)
class EmbeddingIndexSimilarityColumnBenchmarkInstrumentedTest {
    @Test
    fun benchmarkGatherAgainstSequentialScan() {
        val context = ApplicationProvider.getApplicationContext<android.content.Context>()
        val file = File(context.filesDir, ARTIFACT_NAME)
        assumeTrue("$ARTIFACT_NAME is not staged", file.isFile && file.length() == ARTIFACT_BYTES)

        val index = EmbeddingIndex.mmap(file)
        assertEquals(80_421, index.numTracks)
        assertEquals(768, index.dim)
        val referenceRow = 40_210
        val reference = index.getEmbedding(referenceRow)
        val fullSimilarities = FloatArray(index.numTracks)

        for (candidateCount in CANDIDATE_COUNTS) {
            val candidateRows = IntArray(candidateCount) { position ->
                ((position.toLong() * PERMUTATION_STEP) % index.numTracks).toInt()
            }
            val selectedRows = IntArray(candidateCount) { referenceRow }
            val gatheredScores = FloatArray(candidateCount)
            val sequentialScores = FloatArray(candidateCount)

            index.computePairSimilaritiesInto(candidateRows, selectedRows, gatheredScores)
            index.computeAllSimilaritiesInto(reference, fullSimilarities)
            for (position in candidateRows.indices) {
                sequentialScores[position] = fullSimilarities[candidateRows[position]]
            }
            assertEquals(true, gatheredScores.contentEquals(sequentialScores))

            val gatherNs = benchmark {
                index.computePairSimilaritiesInto(candidateRows, selectedRows, gatheredScores)
            }
            val sequentialNs = benchmark {
                index.computeAllSimilaritiesInto(reference, fullSimilarities)
                for (position in candidateRows.indices) {
                    sequentialScores[position] = fullSimilarities[candidateRows[position]]
                }
            }
            Log.i(
                TAG,
                "similarity_column candidates=$candidateCount " +
                    "fraction=${"%.6f".format(candidateCount.toDouble() / index.numTracks)} " +
                    "gather_ms=${gatherNs.summary()} sequential_ms=${sequentialNs.summary()}",
            )
        }
    }

    private fun benchmark(block: () -> Unit): LongArray =
        LongArray(ITERATIONS) { measureNanoTime(block) }.apply { sort() }

    private fun LongArray.summary(): String =
        "p50=${this[size / 2].ms()},min=${first().ms()},max=${last().ms()}"

    private fun Long.ms(): String = "%.3f".format(this / 1_000_000.0)

    private companion object {
        const val TAG = "DppColumnBenchmark"
        const val ARTIFACT_NAME = "retrieval-benchmark.emb"
        const val ARTIFACT_BYTES = 247_696_696L
        const val PERMUTATION_STEP = 7_919L
        const val ITERATIONS = 7
        val CANDIDATE_COUNTS = intArrayOf(
            1_606,
            12_848,
            20_106,
            24_126,
            28_147,
            32_168,
            40_210,
            51_392,
            80_322,
        )
    }
}
