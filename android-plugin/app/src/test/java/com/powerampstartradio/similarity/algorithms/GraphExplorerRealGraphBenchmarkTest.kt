package com.powerampstartradio.similarity.algorithms

import com.powerampstartradio.data.UniformGraphSnapshot
import org.junit.Assert.assertEquals
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import kotlin.system.measureNanoTime

/** Opt-in host benchmark: set GRAPH_EXPLORER_BENCHMARK_FILE to an extracted graph.bin. */
class GraphExplorerRealGraphBenchmarkTest {
    @Test
    fun benchmarkExactDeepExplorerOnRealGraph() {
        val path = System.getenv("GRAPH_EXPLORER_BENCHMARK_FILE")
        assumeTrue("GRAPH_EXPLORER_BENCHMARK_FILE is not set", !path.isNullOrBlank())
        val topology = readGraph(File(requireNotNull(path)))
        val seedId = topology.trackIdAt(topology.nodeCount / 2)

        lateinit var result: GraphExplorerResult
        val elapsedNanos = measureNanoTime {
            result = GraphExplorerSelector.compute(topology, seedId, stopProbability = 0.05f)
        }

        assertEquals(1.0, result.totalTerminalProbability, GraphExplorerSelector.MASS_TOLERANCE)
        println(
            "GRAPH_EXPLORER_BENCHMARK nodes=${topology.nodeCount} " +
                "k=${topology.neighborsPerNode} alpha=0.05 " +
                "elapsed_ms=${elapsedNanos / 1_000_000.0} " +
                "support=${result.ranking.size} expected_links=${result.expectedRouteLinks} " +
                "mass_error=${result.numericMassError} result_sha256=${resultHash(result)}"
        )
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

    private fun readGraph(file: File): UniformGraphSnapshot {
        val buffer = ByteBuffer.wrap(file.readBytes()).order(ByteOrder.LITTLE_ENDIAN)
        val n = buffer.int
        val k = buffer.int
        require(n > 0 && k > 0)
        val expectedSize = 8L + n.toLong() * 8L + n.toLong() * k.toLong() * 8L
        require(file.length() == expectedSize) {
            "graph size mismatch: expected $expectedSize, got ${file.length()}"
        }

        val trackIds = LongArray(n) { buffer.long }
        val neighbors = IntArray(n * k)
        for (edge in neighbors.indices) {
            val neighbor = buffer.int
            val weight = buffer.float
            neighbors[edge] = if (neighbor in 0 until n && weight > 0f) neighbor else -1
        }
        return UniformGraphSnapshot.fromRaw(trackIds, neighbors, k)
    }
}
