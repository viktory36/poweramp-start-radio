package com.powerampstartradio.data

import android.os.Debug
import android.util.Log
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertEquals
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import kotlin.system.measureNanoTime

/** Manual, corpus-gated evidence for the exact production retrieval path. */
@RunWith(AndroidJUnit4::class)
class EmbeddingIndexProductionBenchmarkInstrumentedTest {
    @Test
    fun benchmarkExact80421TrackIndex() {
        val context = ApplicationProvider.getApplicationContext<android.content.Context>()
        val file = File(context.filesDir, ARTIFACT_NAME)
        assumeTrue("$ARTIFACT_NAME is not staged", file.isFile && file.length() == ARTIFACT_BYTES)

        val before = memorySnapshot()
        lateinit var index: EmbeddingIndex
        val mapNs = measureNanoTime { index = EmbeddingIndex.mmap(file) }
        assertEquals(80_421, index.numTracks)
        assertEquals(768, index.dim)
        val afterMap = memorySnapshot()

        val query = index.getEmbedding(40_210)
        val firstTop100Ns = measureNanoTime {
            assertEquals(100, index.findTopK(query, 100).size)
        }
        val afterFirstScan = memorySnapshot()

        repeat(2) { index.findTopK(query, 100) }
        val warmTop1 = benchmark(9) { index.findTopK(query, 1) }
        val warmTop100 = benchmark(9) { index.findTopK(query, 100) }
        val warmTop1000 = benchmark(9) { index.findTopK(query, 1_000) }
        val warmTop1608 = benchmark(9) { index.findTopK(query, 1_608) }
        var cancellationChecks = 0
        val warmTop1608Cancellable = benchmark(9) {
            index.findTopK(query, 1_608) { cancellationChecks++ }
        }

        val excluded = buildSet {
            for (i in 0 until 1_000) add(index.getTrackId(i))
        }
        val warmTop100Exclude1000 = benchmark(5) {
            index.findTopK(query, 100, excluded)
        }

        lateinit var similarities: FloatArray
        val allSimilarities = benchmark(9) {
            similarities = index.computeAllSimilarities(query)
        }
        val targetId = index.getTrackId(60_000)
        val beforeIdLookup = memorySnapshot()
        var rank = -1
        val firstRankNs = measureNanoTime {
            rank = index.rankFromSimilarities(similarities, targetId)
        }
        val afterIdLookup = memorySnapshot()
        check(rank > 0)
        val warmRank = benchmark(9) {
            index.rankFromSimilarities(similarities, targetId)
        }

        Log.i(
            TAG,
            buildString {
                append("production_retrieval ")
                append("map_ms=${mapNs.ms()} first_top100_ms=${firstTop100Ns.ms()} ")
                append("warm_top1_ms=${warmTop1.summary()} ")
                append("warm_top100_ms=${warmTop100.summary()} ")
                append("warm_top1000_ms=${warmTop1000.summary()} ")
                append("warm_top1608_ms=${warmTop1608.summary()} ")
                append("warm_top1608_cancellable_ms=${warmTop1608Cancellable.summary()} ")
                append("cancellation_checks=$cancellationChecks ")
                append("warm_top100_exclude1000_ms=${warmTop100Exclude1000.summary()} ")
                append("all_similarities_ms=${allSimilarities.summary()} ")
                append("first_rank_ms=${firstRankNs.ms()} warm_rank_ms=${warmRank.summary()} ")
                append("memory_before=$before memory_after_map=$afterMap ")
                append("memory_after_first_scan=$afterFirstScan ")
                append("memory_before_id_lookup=$beforeIdLookup memory_after_id_lookup=$afterIdLookup")
            },
        )
    }

    private fun benchmark(iterations: Int, block: () -> Unit): LongArray {
        return LongArray(iterations) { measureNanoTime(block) }.apply { sort() }
    }

    private fun LongArray.summary(): String =
        "p50=${this[size / 2].ms()},min=${first().ms()},max=${last().ms()}"

    private fun Long.ms(): String = "%.3f".format(this / 1_000_000.0)

    private fun memorySnapshot(): String {
        val runtime = Runtime.getRuntime()
        val javaUsed = runtime.totalMemory() - runtime.freeMemory()
        return "pss_kb=${Debug.getPss()},java_used_kb=${javaUsed / 1024}"
    }

    private companion object {
        const val TAG = "EmbeddingIndexBench"
        const val ARTIFACT_NAME = "retrieval-benchmark.emb"
        const val ARTIFACT_BYTES = 247_696_696L
    }
}
