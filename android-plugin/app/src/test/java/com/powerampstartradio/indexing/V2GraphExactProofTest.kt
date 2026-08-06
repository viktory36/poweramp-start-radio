package com.powerampstartradio.indexing

import org.junit.Assert.assertFalse
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class V2GraphExactProofTest {
    @get:Rule
    val temporary = TemporaryFolder()

    @Test
    fun `proof binds the exact graph and embedding bytes`() {
        val graph = temporary.newFile("graph.bin").apply { writeBytes(byteArrayOf(1, 2, 3)) }
        val embedding = temporary.newFile("clamp3.emb").apply {
            writeBytes(byteArrayOf(4, 5, 6))
        }
        val proof = V2GraphExactProof.create(graph, embedding)

        assertTrue(
            V2GraphExactProof.matches(
                bytes = proof,
                graphSha256 = com.powerampstartradio.indexing.v2.V2FileSha256.digest(graph),
                embeddingSha256 = com.powerampstartradio.indexing.v2.V2FileSha256.digest(embedding),
            ),
        )

        graph.appendBytes(byteArrayOf(7))
        assertFalse(
            V2GraphExactProof.matches(
                bytes = proof,
                graphSha256 = com.powerampstartradio.indexing.v2.V2FileSha256.digest(graph),
                embeddingSha256 = com.powerampstartradio.indexing.v2.V2FileSha256.digest(embedding),
            ),
        )
    }

    @Test
    fun `proof can bind already validated generation hashes without rereading assets`() {
        val graphSha = "a".repeat(64)
        val embeddingSha = "b".repeat(64)
        val proof = V2GraphExactProof.createBoundHashes(graphSha, embeddingSha)

        assertTrue(V2GraphExactProof.matches(proof, graphSha, embeddingSha))
        assertFalse(V2GraphExactProof.matches(proof, "c".repeat(64), embeddingSha))
        assertFalse(V2GraphExactProof.matches(proof, graphSha, "d".repeat(64)))
    }

    @Test
    fun `malformed or foreign proof fails closed`() {
        val graphSha = "a".repeat(64)
        val embeddingSha = "b".repeat(64)

        assertFalse(V2GraphExactProof.matches(null, graphSha, embeddingSha))
        assertFalse(V2GraphExactProof.matches(ByteArray(3), graphSha, embeddingSha))
        assertFalse(V2GraphExactProof.matches(ByteArray(136), graphSha, embeddingSha))
        assertFalse(V2GraphExactProof.matches(ByteArray(136), "A".repeat(64), embeddingSha))
    }

    @Test
    fun `same-length graph mutation invalidates constructed exact base`() {
        val base = exactBase()
        base.requireManifestBoundAssets()

        mutateLastByte(base.graphFile)

        assertThrows(IllegalArgumentException::class.java) {
            base.requireManifestBoundAssets()
        }
    }

    @Test
    fun `same-length embedding mutation invalidates constructed exact base`() {
        val base = exactBase()
        base.requireManifestBoundAssets()

        mutateLastByte(base.embeddingFile)

        assertThrows(IllegalArgumentException::class.java) {
            base.requireManifestBoundAssets()
        }
    }

    private fun exactBase(): V2ExactGraphIncrementalBase {
        val embedding = temporary.newFile("base-${System.nanoTime()}.emb").apply {
            writeBytes(byteArrayOf(1, 2, 3, 4))
        }
        val graph = temporary.newFile("base-${System.nanoTime()}.graph").apply {
            writeBytes(byteArrayOf(5, 6, 7, 8))
        }
        return V2ExactGraphIncrementalBase(
            generationId = "test-generation",
            embeddingFile = embedding,
            graphFile = graph,
            trackCount = 1,
            embeddingDimension = 1,
            embeddingByteLength = embedding.length(),
            graphByteLength = graph.length(),
            embeddingSha256 = com.powerampstartradio.indexing.v2.V2FileSha256.digest(embedding),
            graphSha256 = com.powerampstartradio.indexing.v2.V2FileSha256.digest(graph),
        )
    }

    private fun mutateLastByte(file: java.io.File) {
        val expectedLength = file.length()
        val bytes = file.readBytes()
        bytes[bytes.lastIndex] = (bytes.last().toInt() xor 0x01).toByte()
        file.writeBytes(bytes)
        assertTrue(file.length() == expectedLength)
    }
}
