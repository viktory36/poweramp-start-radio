package com.powerampstartradio.data

import org.junit.Assert.assertEquals
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder

class EmbeddingIndexPublicationTest {
    @get:Rule
    val temporaryFolder = TemporaryFolder()

    @Test
    fun atomicPublicationReplacesExistingDestination() {
        val source = temporaryFolder.newFile("clamp3.emb.next").apply {
            writeText("new generation")
        }
        val destination = temporaryFolder.newFile("clamp3.emb").apply {
            writeText("old generation")
        }

        val oldGeneration = RandomAccessFile(destination, "r")
        EmbeddingIndex.replaceAtomically(source, destination)

        assertFalse(source.exists())
        assertTrue(destination.exists())
        assertEquals("new generation", destination.readText())
        assertEquals("old generation", oldGeneration.use { it.readLine() })
    }

    @Test
    fun invalidationAdvancesProcessGeneration() {
        val before = TextEmbeddingIndexGeneration.current()
        val published = TextEmbeddingIndexGeneration.invalidate()
        assertTrue(published > before)
        assertEquals(published, TextEmbeddingIndexGeneration.current())
    }

    @Test
    fun snapshotRejectsEveryStaleGenerationSignal() {
        val indexFile = temporaryFolder.newFile("snapshot.emb").apply {
            writeText("first")
            setLastModified(1_000L)
        }
        val snapshot = TextEmbeddingIndexSnapshot.capture(
            generation = 7L,
            trackCount = 80,
            file = indexFile,
        )

        assertTrue(snapshot.matches(7L, 80, indexFile))
        assertFalse(snapshot.matches(8L, 80, indexFile))
        assertFalse(snapshot.matches(7L, 81, indexFile))

        indexFile.appendText("changed")
        assertFalse(snapshot.matches(7L, 80, indexFile))
    }

    @Test
    fun sortedAndLegacyUnsortedIdsResolveWithoutChangingVectorBytes() {
        val sorted = writeIndex(
            "sorted.emb",
            longArrayOf(10L, 20L, 30L),
            arrayOf(floatArrayOf(1f, 2f), floatArrayOf(3f, 4f), floatArrayOf(5f, 6f)),
        )
        val unsorted = writeIndex(
            "unsorted.emb",
            longArrayOf(30L, 10L, 20L),
            arrayOf(floatArrayOf(5f, 6f), floatArrayOf(1f, 2f), floatArrayOf(3f, 4f)),
        )

        assertArrayEquals(
            floatArrayOf(3f, 4f),
            EmbeddingIndex.mmap(sorted).getEmbeddingByTrackId(20L)!!,
            0f,
        )
        assertArrayEquals(
            floatArrayOf(3f, 4f),
            EmbeddingIndex.mmap(unsorted).getEmbeddingByTrackId(20L)!!,
            0f,
        )
        assertArrayEquals(
            floatArrayOf(5f, 6f),
            EmbeddingIndex.mmap(unsorted).getEmbeddingByTrackId(30L)!!,
            0f,
        )
    }

    @Test
    fun headerProbeRejectsUnsupportedVersionAndImpossibleShape() {
        val unsupported = temporaryFolder.newFile("unsupported.emb").apply {
            writeBytes(header(version = 2, tracks = 0, dim = 1))
        }
        val negativeTracks = temporaryFolder.newFile("negative.emb").apply {
            writeBytes(header(version = 1, tracks = -1, dim = 768))
        }

        assertEquals(-1, EmbeddingIndex.readHeaderTrackCount(unsupported))
        assertEquals(-1, EmbeddingIndex.readHeaderTrackCount(negativeTracks))
    }

    @Test
    fun similarityRankRejectsAStaleArrayFromAnotherGeneration() {
        val file = writeIndex(
            "rank-shape.emb",
            longArrayOf(10L, 20L),
            arrayOf(floatArrayOf(1f), floatArrayOf(0f)),
        )
        val index = EmbeddingIndex.mmap(file)
        var rejected = false
        try {
            index.rankFromSimilarities(floatArrayOf(1f), 10L)
        } catch (_: IllegalArgumentException) {
            rejected = true
        }
        assertTrue("cross-generation similarity arrays must fail closed", rejected)
    }

    private fun writeIndex(
        name: String,
        ids: LongArray,
        embeddings: Array<FloatArray>,
    ): java.io.File {
        require(ids.size == embeddings.size && embeddings.isNotEmpty())
        val dim = embeddings.first().size
        require(embeddings.all { it.size == dim })
        val file = temporaryFolder.newFile(name)
        val buffer = ByteBuffer.allocate(
            16 + ids.size * Long.SIZE_BYTES + ids.size * dim * Float.SIZE_BYTES,
        ).order(ByteOrder.LITTLE_ENDIAN)
        buffer.put(header(version = 1, tracks = ids.size, dim = dim))
        ids.forEach(buffer::putLong)
        embeddings.forEach { row -> row.forEach(buffer::putFloat) }
        file.writeBytes(buffer.array())
        return file
    }

    private fun header(version: Int, tracks: Int, dim: Int): ByteArray =
        ByteBuffer.allocate(16)
            .order(ByteOrder.LITTLE_ENDIAN)
            .putInt(0x424D4550)
            .putInt(version)
            .putInt(tracks)
            .putInt(dim)
            .array()
}
