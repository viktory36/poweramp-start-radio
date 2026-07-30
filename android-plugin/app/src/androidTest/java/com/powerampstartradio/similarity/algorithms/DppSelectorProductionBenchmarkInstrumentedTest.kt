package com.powerampstartradio.similarity.algorithms

import android.os.Debug
import android.util.Log
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.powerampstartradio.data.EmbeddingIndex
import com.powerampstartradio.similarity.SelectedTrack
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import kotlin.math.abs
import kotlin.math.sqrt
import kotlin.system.measureNanoTime

/** Read-only, corpus-gated evidence for the production-sized bounded DPP path. */
@RunWith(AndroidJUnit4::class)
class DppSelectorProductionBenchmarkInstrumentedTest {
    @Test
    fun benchmarkAdaptiveAgainstOneFullDomainRun() {
        val index = productionIndex()
        val seedRow = 40_210
        val seedId = index.getTrackId(seedRow)
        val candidates = index.findTopK(
            query = index.getEmbedding(seedRow),
            topK = index.numTracks,
            excludeIds = setOf(seedId),
        )
        assertEquals(index.numTracks - 1, candidates.size)

        for (exponent in floatArrayOf(0f, 0.5f, 1f, 2f)) {
            fun fullRun(): List<SelectedTrack> = DppSelector.selectBatch(
                candidates = candidates,
                numSelect = QUEUE_SIZE,
                index = index,
                qualityExponent = exponent,
            )
            fun adaptiveRun(): DppSelector.CertifiedSelection =
                DppSelector.selectBatchCertified(
                    candidates = candidates,
                    numSelect = QUEUE_SIZE,
                    index = index,
                    initialCandidateCount = INITIAL_CANDIDATE_COUNT,
                    qualityExponent = exponent,
                )

            val warmFull = fullRun()
            val warmAdaptive = adaptiveRun()
            assertEquals(warmFull, warmAdaptive.tracks)
            val fullMs = DoubleArray(POLICY_REPETITIONS)
            val adaptiveMs = DoubleArray(POLICY_REPETITIONS)
            lateinit var evidence: DppSelector.CertificationEvidence
            repeat(POLICY_REPETITIONS) { iteration ->
                val full: Timed<List<SelectedTrack>>
                val adaptive: Timed<DppSelector.CertifiedSelection>
                if (iteration % 2 == 0) {
                    full = timedSelection("full_q${exponent}_run$iteration") { fullRun() }
                    adaptive = timedSelection("adaptive_q${exponent}_run$iteration") {
                        adaptiveRun()
                    }
                } else {
                    adaptive = timedSelection("adaptive_q${exponent}_run$iteration") {
                        adaptiveRun()
                    }
                    full = timedSelection("full_q${exponent}_run$iteration") { fullRun() }
                }
                assertEquals(full.result, adaptive.result.tracks)
                fullMs[iteration] = full.elapsedMs
                adaptiveMs[iteration] = adaptive.elapsedMs
                evidence = adaptive.result.evidence
            }
            fullMs.sort()
            adaptiveMs.sort()
            Log.i(
                TAG,
                "dpp_compare exponent=$exponent full_ms=${fullMs.summary()} " +
                    "adaptive_ms=${adaptiveMs.summary()} " +
                    "adaptive_over_full_p50=${adaptiveMs[POLICY_REPETITIONS / 2] / fullMs[POLICY_REPETITIONS / 2]} " +
                    "attempts=${evidence.attemptedCandidateCounts} " +
                    "final=${evidence.finalCandidateCount}/${candidates.size}",
            )
        }
    }

    private fun DoubleArray.summary(): String =
        "p50=${this[size / 2]},min=${first()},max=${last()}"

    @Test
    fun quantifyCanonicalNativeReductionAgainstLegacyScalar() {
        val index = productionIndex()
        val seedRow = 40_210
        val seedId = index.getTrackId(seedRow)
        val candidates = index.findTopK(
            query = index.getEmbedding(seedRow),
            topK = SCALAR_CANDIDATE_COUNT + 1,
            excludeIds = setOf(seedId),
        ).take(SCALAR_CANDIDATE_COUNT)

        val native = DppSelector.selectBatch(candidates, QUEUE_SIZE, index)
        val scalar = legacyScalarSelect(candidates, QUEUE_SIZE, index)
        val mismatchedQueuePositions = native.indices.filter { position ->
            native[position].trackId != scalar.tracks[position].trackId
        }

        val rows = index.findTrackIndices(candidates.map { it.first }.toLongArray())
        val left = IntArray(SCALAR_PAIR_COUNT) { position -> rows[position] }
        val right = IntArray(SCALAR_PAIR_COUNT) { position ->
            rows[(position * 37 + 101) % rows.size]
        }
        val nativeScores = index.computePairSimilarities(left, right)
        var maxAbsoluteDifference = 0f
        var bitDifferentCount = 0
        for (position in nativeScores.indices) {
            val scalarScore = index.dotProduct(index.getEmbedding(left[position]), right[position])
            if (nativeScores[position].toRawBits() != scalarScore.toRawBits()) bitDifferentCount++
            maxAbsoluteDifference = maxOf(
                maxAbsoluteDifference,
                abs(nativeScores[position] - scalarScore),
            )
        }

        Log.i(
            TAG,
            "native_scalar candidates=${candidates.size} queue_mismatches=" +
                "${mismatchedQueuePositions.size} mismatch_positions=$mismatchedQueuePositions " +
                "pair_bit_differences=$bitDifferentCount/$SCALAR_PAIR_COUNT " +
                "max_pair_abs_difference=$maxAbsoluteDifference " +
                "native_gain_count=${native.size} scalar_gains=${scalar.gains}",
        )
        assertEquals(QUEUE_SIZE, native.size)
        assertEquals(QUEUE_SIZE, scalar.tracks.size)
        assertTrue(maxAbsoluteDifference < 1e-5f)
    }

    private fun productionIndex(): EmbeddingIndex {
        val context = ApplicationProvider.getApplicationContext<android.content.Context>()
        val file = File(context.filesDir, ARTIFACT_NAME)
        assumeTrue("$ARTIFACT_NAME is not staged", file.isFile && file.length() == ARTIFACT_BYTES)
        return EmbeddingIndex.mmap(file).also { index ->
            assertEquals(80_421, index.numTracks)
            assertEquals(768, index.dim)
            assertEquals("Dalvik", System.getProperty("java.vm.name"))
        }
    }

    private fun <T : Any> timedSelection(label: String, block: () -> T): Timed<T> {
        Runtime.getRuntime().gc()
        Thread.sleep(50)
        val before = memorySnapshot()
        lateinit var result: T
        val elapsedNs = measureNanoTime { result = block() }
        val after = memorySnapshot()
        Log.i(TAG, "dpp_timing label=$label elapsed_ms=${elapsedNs / 1_000_000.0} before=$before after=$after")
        return Timed(result, elapsedNs / 1_000_000.0)
    }

    private fun memorySnapshot(): String {
        val runtime = Runtime.getRuntime()
        val used = runtime.totalMemory() - runtime.freeMemory()
        return "java_used_kb=${used / 1024},java_committed_kb=${runtime.totalMemory() / 1024}," +
            "java_max_kb=${runtime.maxMemory() / 1024},pss_kb=${Debug.getPss()}"
    }

    private fun legacyScalarSelect(
        candidates: List<Pair<Long, Float>>,
        numSelect: Int,
        index: EmbeddingIndex,
    ): LegacyRun {
        val count = candidates.size
        val limit = minOf(numSelect, count)
        val embeddings = Array(count) { position ->
            requireNotNull(index.getEmbeddingByTrackId(candidates[position].first))
        }
        val quality = FloatArray(count) { position -> candidates[position].second.coerceAtLeast(0f) }
        val factors = Array(count) { FloatArray(limit) }
        val residual = FloatArray(count) { position -> quality[position] * quality[position] }
        val selected = BooleanArray(count)
        val selectedPositions = mutableListOf<Int>()
        val gains = mutableListOf<Float>()

        for (step in 0 until limit) {
            var best = -1
            var bestGain = -1f
            for (position in 0 until count) {
                if (!selected[position] && residual[position] > bestGain) {
                    best = position
                    bestGain = residual[position]
                }
            }
            if (best < 0 || bestGain <= MIN_MARGINAL_GAIN) break
            selected[best] = true
            selectedPositions += best
            gains += bestGain
            val root = sqrt(bestGain)

            for (position in 0 until count) {
                if (selected[position]) continue
                var dot = 0f
                for (column in embeddings[position].indices) {
                    dot += embeddings[position][column] * embeddings[best][column]
                }
                var value = quality[position] * quality[best] * dot
                for (prior in 0 until step) {
                    value -= factors[position][prior] * factors[best][prior]
                }
                factors[position][step] = value / root
                residual[position] -= factors[position][step] * factors[position][step]
                if (residual[position] < 0f) residual[position] = 0f
            }
            factors[best][step] = root
        }

        return LegacyRun(
            tracks = selectedPositions.map { position ->
                val (trackId, relevance) = candidates[position]
                SelectedTrack(trackId, relevance, candidateRank = position + 1)
            },
            gains = gains,
        )
    }

    private data class Timed<T>(val result: T, val elapsedMs: Double)
    private data class LegacyRun(val tracks: List<SelectedTrack>, val gains: List<Float>)

    private companion object {
        const val TAG = "DppProductionBenchmark"
        const val ARTIFACT_NAME = "retrieval-benchmark.emb"
        const val ARTIFACT_BYTES = 247_696_696L
        const val QUEUE_SIZE = 30
        const val INITIAL_CANDIDATE_COUNT = 1_608
        const val SCALAR_CANDIDATE_COUNT = 1_608
        const val SCALAR_PAIR_COUNT = 1_024
        const val POLICY_REPETITIONS = 5
        const val MIN_MARGINAL_GAIN = 1e-10f
    }
}
