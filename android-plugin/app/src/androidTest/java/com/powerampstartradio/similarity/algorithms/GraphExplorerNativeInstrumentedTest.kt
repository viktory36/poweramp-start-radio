package com.powerampstartradio.similarity.algorithms

import android.util.Log
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.powerampstartradio.data.GraphIndex
import com.powerampstartradio.data.UniformGraphSnapshot
import kotlinx.coroutines.CancellationException
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import kotlin.system.measureNanoTime

@RunWith(AndroidJUnit4::class)
class GraphExplorerNativeInstrumentedTest {
    @Test
    fun nativeKernelMatchesPureReferenceAcrossAlphaEndpointsAndInterior() {
        val graph = graphOf(
            100L to longArrayOf(20L, 40L),
            20L to longArrayOf(100L, 30L, 40L),
            40L to longArrayOf(100L, 30L, 20L),
            30L to longArrayOf(20L, 40L, 50L),
            50L to longArrayOf(),
        )

        for (alpha in listOf(0f, 0.05f, 0.5f, 1f)) {
            val reference = GraphExplorerSelector.compute(graph, 100L, alpha)
            val native = GraphExplorerSelector.computeNative(graph, 100L, alpha)

            assertEquals(reference.ranking.map { it.trackId }, native.ranking.map { it.trackId })
            assertEquals(reference.evaluatedLinks, native.evaluatedLinks)
            assertEquals(reference.totalTerminalProbability, native.totalTerminalProbability, 1e-14)
            assertEquals(reference.excludedSeedProbability, native.excludedSeedProbability, 1e-14)
            assertEquals(reference.expectedRouteLinks, native.expectedRouteLinks, 1e-13)
            reference.ranking.zip(native.ranking).forEach { (expected, actual) ->
                assertEquals(expected.terminalProbability, actual.terminalProbability, 1e-14)
                assertEquals(expected.expectedRouteLinks, actual.expectedRouteLinks, 1e-13)
            }
        }
    }

    @Test
    fun nativeKernelPropagatesCallerCancellation() {
        val graph = graphOf(
            1L to longArrayOf(2L),
            2L to longArrayOf(3L),
            3L to longArrayOf(1L),
        )
        var checks = 0

        assertThrows(CancellationException::class.java) {
            GraphExplorerSelector.computeNative(graph, 1L, 0f) {
                checks++
                if (checks == 4) throw CancellationException("native cancellation test")
            }
        }
        assertEquals(4, checks)
    }

    /**
     * Opt-in, non-mutating benchmark. Before this test, copy the frozen graph to
     * `<target files>/graph-explorer-benchmark.bin`; the test never opens Poweramp.
     */
    @Test
    fun benchmarkNativeKernelOnStagedRealGraph() {
        val context = ApplicationProvider.getApplicationContext<android.content.Context>()
        val file = File(context.filesDir, "graph-explorer-benchmark.bin")
        assumeTrue("real graph was not staged", file.isFile)

        val topology = GraphIndex.mmap(file).uniformTopology()
        val seedId = topology.trackIdAt(topology.nodeCount / 2)
        GraphExplorerSelector.computeNative(topology, seedId, 0.5f)

        val vmHwmBeforeKb = readVmHwmKb()
        val pssBeforeKb = android.os.Debug.getPss()
        val benchmarkLines = mutableListOf<String>()
        for (alpha in listOf(0.05f, 0.5f, 0.95f)) {
            val timings = LongArray(3)
            val hashes = mutableSetOf<String>()
            lateinit var result: GraphExplorerResult
            repeat(timings.size) { run ->
                timings[run] = measureNanoTime {
                    result = GraphExplorerSelector.computeNative(topology, seedId, alpha)
                }
                hashes += resultHash(result)
            }
            timings.sort()

            assertEquals(1, hashes.size)
            assertEquals(
                1.0,
                result.totalTerminalProbability,
                GraphExplorerSelector.MASS_TOLERANCE,
            )
            benchmarkLines += "alpha=$alpha min_ms=${timings.first() / 1_000_000.0} " +
                "median_ms=${timings[1] / 1_000_000.0} support=${result.ranking.size} " +
                "expected_links=${result.expectedRouteLinks} mass_error=${result.numericMassError} " +
                "result_sha256=${hashes.single()}"
        }

        val callbackTimes = mutableListOf<Long>()
        try {
            GraphExplorerSelector.computeNative(topology, seedId, 0.05f) {
                callbackTimes += System.nanoTime()
                if (callbackTimes.size == 35) {
                    throw CancellationException("real-graph cancellation probe")
                }
            }
        } catch (_: CancellationException) {
            // Expected: callback spacing is the native cancellation polling bound.
        }
        assertEquals(35, callbackTimes.size)
        val cancellationPollMaxMs = callbackTimes.zipWithNext()
            .maxOf { (before, after) -> (after - before) / 1_000_000.0 }

        val edgeCount = topology.nodeCount.toLong() * topology.neighborsPerNode
        val exactWorkspaceBytes = edgeCount * (2L * 8L + 2L * 4L) +
            topology.nodeCount.toLong() * 2L * 8L
        val vmHwmAfterKb = readVmHwmKb()
        val pssAfterKb = android.os.Debug.getPss()
        val header = "nodes=${topology.nodeCount} k=${topology.neighborsPerNode} " +
            "workspace_bytes=$exactWorkspaceBytes vm_hwm_before_kb=$vmHwmBeforeKb " +
            "vm_hwm_after_kb=$vmHwmAfterKb pss_before_kb=$pssBeforeKb " +
            "pss_after_kb=$pssAfterKb cancellation_poll_max_ms=$cancellationPollMaxMs"
        val message = (listOf(header) + benchmarkLines).joinToString(" | ")
        Log.i("GraphExplorerBenchmark", message)
        println("GRAPH_EXPLORER_NATIVE_BENCHMARK $message")
    }

    private fun resultHash(result: GraphExplorerResult): String {
        val digest = MessageDigest.getInstance("SHA-256")
        val record = ByteBuffer.allocate(24).order(ByteOrder.LITTLE_ENDIAN)
        result.ranking.forEach { score ->
            record.clear()
            record.putLong(score.trackId)
            record.putLong(score.terminalProbability.toBits())
            record.putLong(score.expectedRouteLinks.toBits())
            digest.update(record.array())
        }
        return digest.digest().joinToString("") { byte -> "%02x".format(byte) }
    }

    private fun readVmHwmKb(): Long {
        val line = File("/proc/self/status").useLines { lines ->
            lines.firstOrNull { it.startsWith("VmHWM:") }
        } ?: return -1L
        return line.split(Regex("\\s+")).getOrNull(1)?.toLongOrNull() ?: -1L
    }

    private fun graphOf(vararg rows: Pair<Long, LongArray>): UniformGraphSnapshot {
        val ids = LongArray(rows.size) { rows[it].first }
        val indexById = ids.withIndex().associate { it.value to it.index }
        val k = rows.maxOfOrNull { it.second.size }?.coerceAtLeast(1) ?: 1
        val neighbors = IntArray(ids.size * k) { -1 }
        rows.forEachIndexed { node, row ->
            row.second.forEachIndexed { slot, neighborId ->
                neighbors[node * k + slot] = requireNotNull(indexById[neighborId])
            }
        }
        return UniformGraphSnapshot.fromRaw(ids, neighbors, k)
    }
}
