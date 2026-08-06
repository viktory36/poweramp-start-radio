package com.powerampstartradio.data

import android.content.ContentValues
import android.database.sqlite.SQLiteDatabase
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.powerampstartradio.indexing.GraphUpdater
import com.powerampstartradio.indexing.V2ExactGraphIncrementalBase
import com.powerampstartradio.indexing.V2GraphUpdateStrategy
import com.powerampstartradio.indexing.v2.V2FileSha256
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.CancellationException

@RunWith(AndroidJUnit4::class)
class EmbeddingIndexNativeInstrumentedTest {
    @Test
    fun nativeTopKUsesAscendingTrackIdForExactScoreTiesAndCutoff() {
        val context = ApplicationProvider.getApplicationContext<android.content.Context>()
        val file = File(context.cacheDir, "native-top-k-ties.emb")
        val ids = longArrayOf(50L, 10L, 40L, 20L, 30L)
        writeIndex(file, ids)

        try {
            val index = EmbeddingIndex.mmap(file)
            val result = index.findTopK(floatArrayOf(1f, 0f), topK = 3)
            assertEquals(listOf(10L, 20L, 30L), result.map { it.first })
            assertEquals(listOf(1f, 1f, 1f), result.map { it.second })
            assertEquals(emptyList<Pair<Long, Float>>(), index.findTopK(floatArrayOf(1f, 0f), 0))
        } finally {
            file.delete()
        }
    }

    @Test
    fun nativeTopKHandlesLargeUnorderedExclusionSetWithoutChangingRankOrder() {
        val context = ApplicationProvider.getApplicationContext<android.content.Context>()
        val file = File(context.cacheDir, "native-top-k-exclusions.emb")
        val ids = LongArray(32) { (it + 1).toLong() }
        writeIndex(file, ids)

        try {
            val index = EmbeddingIndex.mmap(file)
            val excluded = linkedSetOf(7L, 3L, 11L, 1L, 9L, 5L, 12L, 2L, 8L, 4L, 10L, 6L)
            val result = index.findTopK(floatArrayOf(1f, 0f), topK = 5, excludeIds = excluded)
            assertEquals(listOf(13L, 14L, 15L, 16L, 17L), result.map { it.first })
        } finally {
            file.delete()
        }
    }

    @Test
    fun nativeTopKMatchesExactScalarOrderingAcrossKValues() {
        val context = ApplicationProvider.getApplicationContext<android.content.Context>()
        val file = File(context.cacheDir, "native-top-k-reference.emb")
        val ids = LongArray(97) { ((it * 37) % 97 + 1).toLong() }
        val embeddings = Array(ids.size) { row ->
            FloatArray(8) { column -> (((row * 11 + column * 7) % 9) - 4) / 8f }
        }
        val query = floatArrayOf(0.5f, -0.25f, 0.125f, 0.375f, -0.5f, 0.25f, 0.0f, -0.125f)
        val excluded = setOf(2L, 7L, 13L, 19L, 23L, 31L, 43L, 59L, 71L, 83L)
        writeIndex(file, ids, embeddings)

        try {
            val index = EmbeddingIndex.mmap(file)
            val expected = ids.indices
                .asSequence()
                .filter { ids[it] !in excluded }
                .map { row ->
                    val score = query.indices.sumOf { column ->
                        (query[column] * embeddings[row][column]).toDouble()
                    }.toFloat()
                    ids[row] to score
                }
                .sortedWith(compareByDescending<Pair<Long, Float>> { it.second }.thenBy { it.first })
                .toList()

            for (k in listOf(1, 5, 17, 64, 97)) {
                val actual = index.findTopK(query, k, excluded)
                assertEquals(expected.take(k), actual)
            }
        } finally {
            file.delete()
        }
    }

    @Test
    fun nativeRankUsesAscendingTrackIdForExactScoreTies() {
        val context = ApplicationProvider.getApplicationContext<android.content.Context>()
        val file = File(context.cacheDir, "native-rank-ties.emb")
        writeIndex(file, longArrayOf(30L, 10L, 20L))

        try {
            val index = EmbeddingIndex.mmap(file)
            val similarities = floatArrayOf(0.5f, 0.75f, 0.75f)
            assertEquals(1, index.rankFromSimilarities(similarities, 10L))
            assertEquals(2, index.rankFromSimilarities(similarities, 20L))
            assertEquals(3, index.rankFromSimilarities(similarities, 30L))
        } finally {
            file.delete()
        }
    }

    @Test
    fun nativeTopKPropagatesCancellationDuringScan() {
        val context = ApplicationProvider.getApplicationContext<android.content.Context>()
        val file = File(context.cacheDir, "native-top-k-cancellation.emb")
        writeIndex(file, LongArray(10_000) { (it + 1).toLong() })

        try {
            val index = EmbeddingIndex.mmap(file)
            var checks = 0
            var cancelled = false
            try {
                index.findTopK(floatArrayOf(1f, 0f), topK = 10) {
                    checks++
                    if (checks == 2) throw CancellationException("test cancellation")
                }
            } catch (_: CancellationException) {
                cancelled = true
            }
            assertTrue("scan must propagate callback cancellation", cancelled)
            assertEquals(2, checks)
        } finally {
            file.delete()
        }
    }

    @Test
    fun nativeTopKRejectsWrongQueryDimensionBeforeNativeMemoryAccess() {
        val context = ApplicationProvider.getApplicationContext<android.content.Context>()
        val file = File(context.cacheDir, "native-top-k-query-shape.emb")
        writeIndex(file, longArrayOf(1L, 2L))

        try {
            val index = EmbeddingIndex.mmap(file)
            var rejected = false
            try {
                index.findTopK(floatArrayOf(1f), topK = 1)
            } catch (_: IllegalArgumentException) {
                rejected = true
            }
            assertTrue("wrong query dimensions must fail closed", rejected)
            assertFalse(index.findTopK(floatArrayOf(1f, 0f), topK = 0).iterator().hasNext())
        } finally {
            file.delete()
        }
    }

    @Test
    fun nativePairScoresUseTheSameReductionAsFullSimilarityScans() {
        val context = ApplicationProvider.getApplicationContext<android.content.Context>()
        val file = File(context.cacheDir, "native-pair-scores.emb")
        val ids = LongArray(19) { (it + 1).toLong() }
        val embeddings = Array(ids.size) { row ->
            FloatArray(32) { column -> (((row * 17 + column * 13) % 31) - 15) / 16f }
        }
        writeIndex(file, ids, embeddings)

        try {
            val index = EmbeddingIndex.mmap(file)
            val left = intArrayOf(0, 3, 7, 11, 18)
            val right = intArrayOf(18, 9, 2, 11, 0)
            val pairs = index.computePairSimilarities(left, right)
            for (position in left.indices) {
                val fullScan = index.computeAllSimilarities(index.getEmbedding(left[position]))
                assertEquals(fullScan[right[position]], pairs[position], 0f)
            }
        } finally {
            file.delete()
        }
    }

    @Test
    fun nativePairScoresPropagateCancellationDuringLargeBatch() {
        val context = ApplicationProvider.getApplicationContext<android.content.Context>()
        val file = File(context.cacheDir, "native-pair-cancellation.emb")
        writeIndex(file, LongArray(4_096) { (it + 1).toLong() })

        try {
            val index = EmbeddingIndex.mmap(file)
            val left = IntArray(4_096) { it }
            val right = IntArray(4_096) { 4_095 - it }
            var checks = 0
            var cancelled = false
            try {
                index.computePairSimilarities(left, right) {
                    checks++
                    if (checks == 2) throw CancellationException("pair cancellation")
                }
            } catch (_: CancellationException) {
                cancelled = true
            }
            assertTrue("pair batch must propagate callback cancellation", cancelled)
            assertEquals(2, checks)
        } finally {
            file.delete()
        }
    }

    @Test
    fun incrementalGraphIsByteIdenticalToNativeFullRebuild() {
        val context = ApplicationProvider.getApplicationContext<android.content.Context>()
        val incrementalDbFile = File(context.cacheDir, "graph-incremental.db")
        val fullDbFile = File(context.cacheDir, "graph-full.db")
        val incrementalDir = File(context.cacheDir, "graph-incremental-work")
        val baseDir = File(context.cacheDir, "graph-incremental-base")
        val fullDir = File(context.cacheDir, "graph-full-work")
        val rows = (1L..9L).map { id ->
            id to FloatArray(16) { column ->
                0.1f + (((id.toInt() * 7 + column * 11) % 13) / 32f)
            }
        }
        createEmbeddingDatabase(incrementalDbFile, rows.take(7))
        createEmbeddingDatabase(fullDbFile, rows)

        try {
            val incrementalDb = EmbeddingDatabase.openReadWrite(incrementalDbFile)
            try {
                val initial = GraphUpdater(incrementalDb, incrementalDir).rebuildIndices()
                assertEquals(V2GraphUpdateStrategy.FULL_REBUILD, initial.strategy)
                val exactBase = copyExactBase(incrementalDir, baseDir, rows.take(7))
                rows.drop(7).forEach { (id, embedding) ->
                    incrementalDb.insertEmbedding("embeddings_clamp3", id, embedding)
                }
                val incremental = GraphUpdater(incrementalDb, incrementalDir).rebuildIndices(
                    exactBase = exactBase,
                )
                assertEquals(V2GraphUpdateStrategy.INCREMENTAL, incremental.strategy)
            } finally {
                incrementalDb.close()
            }

            val fullDb = EmbeddingDatabase.openReadWrite(fullDbFile)
            try {
                val full = GraphUpdater(fullDb, fullDir).rebuildIndices()
                assertEquals(V2GraphUpdateStrategy.FULL_REBUILD, full.strategy)
            } finally {
                fullDb.close()
            }
            assertTrue(
                "incremental graph bytes must equal the native full-build oracle",
                File(incrementalDir, "graph.bin").readBytes()
                    .contentEquals(File(fullDir, "graph.bin").readBytes()),
            )
        } finally {
            incrementalDbFile.delete()
            fullDbFile.delete()
            incrementalDir.deleteRecursively()
            baseDir.deleteRecursively()
            fullDir.deleteRecursively()
        }
    }

    @Test
    fun deletionAndAppendGraphIsByteIdenticalToNativeFullRebuild() {
        val context = ApplicationProvider.getApplicationContext<android.content.Context>()
        val incrementalDbFile = File(context.cacheDir, "graph-mixed-incremental.db")
        val fullDbFile = File(context.cacheDir, "graph-mixed-full.db")
        val incrementalDir = File(context.cacheDir, "graph-mixed-incremental-work")
        val baseDir = File(context.cacheDir, "graph-mixed-base")
        val fullDir = File(context.cacheDir, "graph-mixed-full-work")
        val allRows = (1L..14L).associateWith { id ->
            FloatArray(24) { column ->
                0.05f + (((id.toInt() * 17 + column * 19) % 23) / 40f)
            }
        }
        val baseRows = (1L..10L).map { it to requireNotNull(allRows[it]) }
        val targetIds = listOf(1L, 2L, 4L, 5L, 6L, 7L, 9L, 10L, 11L, 12L, 13L, 14L)
        val targetRows = targetIds.map { it to requireNotNull(allRows[it]) }
        createEmbeddingDatabase(incrementalDbFile, baseRows)
        createEmbeddingDatabase(fullDbFile, targetRows)

        try {
            val initialDb = EmbeddingDatabase.openReadWrite(incrementalDbFile)
            val exactBase = try {
                val initial = GraphUpdater(initialDb, incrementalDir).rebuildIndices()
                assertEquals(V2GraphUpdateStrategy.FULL_REBUILD, initial.strategy)
                copyExactBase(incrementalDir, baseDir, baseRows)
            } finally {
                initialDb.close()
            }
            SQLiteDatabase.openDatabase(
                incrementalDbFile.path,
                null,
                SQLiteDatabase.OPEN_READWRITE,
            ).use { database ->
                database.delete("embeddings_clamp3", "track_id IN (?, ?)", arrayOf("3", "8"))
                targetRows.filter { it.first >= 11L }.forEach { (trackId, embedding) ->
                    database.insertOrThrow(
                        "embeddings_clamp3",
                        null,
                        ContentValues().apply {
                            put("track_id", trackId)
                            put("embedding", EmbeddingDatabase.floatArrayToBlob(embedding))
                        },
                    )
                }
            }

            val incrementalDb = EmbeddingDatabase.openReadWrite(incrementalDbFile)
            try {
                val incremental = GraphUpdater(incrementalDb, incrementalDir).rebuildIndices(
                    exactBase = exactBase,
                )
                assertEquals(V2GraphUpdateStrategy.INCREMENTAL, incremental.strategy)
                assertEquals(2, incremental.removedBaseNodes)
                assertEquals(4, incremental.newNodes)
                assertTrue(incremental.rescannedBaseNodes > 0)
            } finally {
                incrementalDb.close()
            }
            val fullDb = EmbeddingDatabase.openReadWrite(fullDbFile)
            try {
                assertEquals(
                    V2GraphUpdateStrategy.FULL_REBUILD,
                    GraphUpdater(fullDb, fullDir).rebuildIndices().strategy,
                )
            } finally {
                fullDb.close()
            }
            assertTrue(
                "mixed delta graph bytes must equal the native full-build oracle",
                File(incrementalDir, "graph.bin").readBytes()
                    .contentEquals(File(fullDir, "graph.bin").readBytes()),
            )
        } finally {
            incrementalDbFile.delete()
            fullDbFile.delete()
            incrementalDir.deleteRecursively()
            baseDir.deleteRecursively()
            fullDir.deleteRecursively()
        }
    }

    @Test
    fun changedRetainedEmbeddingRejectsIncrementalBase() {
        val context = ApplicationProvider.getApplicationContext<android.content.Context>()
        val dbFile = File(context.cacheDir, "graph-changed-retained.db")
        val workDir = File(context.cacheDir, "graph-changed-retained-work")
        val baseDir = File(context.cacheDir, "graph-changed-retained-base")
        val baseRows = (1L..7L).map { id ->
            id to FloatArray(16) { column ->
                0.1f + (((id.toInt() * 7 + column * 11) % 13) / 32f)
            }
        }
        createEmbeddingDatabase(dbFile, baseRows)

        try {
            val initialDb = EmbeddingDatabase.openReadWrite(dbFile)
            val exactBase = try {
                assertEquals(
                    V2GraphUpdateStrategy.FULL_REBUILD,
                    GraphUpdater(initialDb, workDir).rebuildIndices().strategy,
                )
                copyExactBase(workDir, baseDir, baseRows)
            } finally {
                initialDb.close()
            }
            SQLiteDatabase.openDatabase(
                dbFile.path,
                null,
                SQLiteDatabase.OPEN_READWRITE,
            ).use { database ->
                database.update(
                    "embeddings_clamp3",
                    ContentValues().apply {
                        put(
                            "embedding",
                            EmbeddingDatabase.floatArrayToBlob(
                                FloatArray(16) { column -> 0.9f - column / 64f },
                            ),
                        )
                    },
                    "track_id = ?",
                    arrayOf("4"),
                )
                database.insertOrThrow(
                    "embeddings_clamp3",
                    null,
                    ContentValues().apply {
                        put("track_id", 8L)
                        put(
                            "embedding",
                            EmbeddingDatabase.floatArrayToBlob(
                                FloatArray(16) { column -> 0.2f + column / 128f },
                            ),
                        )
                    },
                )
            }

            val targetDb = EmbeddingDatabase.openReadWrite(dbFile)
            try {
                val result = GraphUpdater(targetDb, workDir).rebuildIndices(
                    exactBase = exactBase,
                )
                assertEquals(V2GraphUpdateStrategy.FULL_REBUILD, result.strategy)
            } finally {
                targetDb.close()
            }
        } finally {
            dbFile.delete()
            workDir.deleteRecursively()
            baseDir.deleteRecursively()
        }
    }

    @Test
    fun sameLengthGraphMutationAfterBaseConstructionFallsBackToFullRebuild() {
        assertSameLengthExactBaseMutationFallsBack("graph.bin")
    }

    @Test
    fun sameLengthEmbeddingMutationAfterBaseConstructionFallsBackToFullRebuild() {
        assertSameLengthExactBaseMutationFallsBack("clamp3.emb")
    }

    @Test
    fun extractionRejectsMixedDimensionsWithoutReplacingPublishedFile() {
        val context = ApplicationProvider.getApplicationContext<android.content.Context>()
        val dbFile = File(context.cacheDir, "mixed-embedding-dimensions.db")
        val destination = File(context.cacheDir, "published.emb").apply {
            writeText("previous generation")
        }
        createEmbeddingDatabase(
            dbFile,
            listOf(1L to floatArrayOf(1f, 0f), 2L to floatArrayOf(1f)),
        )

        val database = EmbeddingDatabase.open(dbFile)
        try {
            var rejected = false
            try {
                EmbeddingIndex.extractFromDatabase(database, destination)
            } catch (_: IllegalStateException) {
                rejected = true
            }
            assertTrue("mixed dimensions must fail extraction", rejected)
            assertEquals("previous generation", destination.readText())
        } finally {
            database.close()
            dbFile.delete()
            destination.delete()
        }
    }

    private fun writeIndex(
        file: File,
        ids: LongArray,
        embeddings: Array<FloatArray> = Array(ids.size) { floatArrayOf(1f, 0f) },
    ) {
        require(ids.size == embeddings.size && embeddings.isNotEmpty())
        val dim = embeddings.first().size
        require(embeddings.all { it.size == dim })
        val size = 16 + ids.size * Long.SIZE_BYTES + ids.size * dim * Float.SIZE_BYTES
        val buffer = ByteBuffer.allocate(size).order(ByteOrder.LITTLE_ENDIAN)
        buffer.putInt(0x424D4550)
        buffer.putInt(1)
        buffer.putInt(ids.size)
        buffer.putInt(dim)
        ids.forEach(buffer::putLong)
        embeddings.forEach { row -> row.forEach(buffer::putFloat) }
        file.writeBytes(buffer.array())
    }

    private fun createEmbeddingDatabase(
        file: File,
        rows: List<Pair<Long, FloatArray>>,
    ) {
        file.delete()
        SQLiteDatabase.openOrCreateDatabase(file, null).use { database ->
            database.execSQL(
                "CREATE TABLE embeddings_clamp3 (track_id INTEGER PRIMARY KEY, embedding BLOB NOT NULL)"
            )
            database.execSQL(
                "CREATE TABLE binary_data (key TEXT PRIMARY KEY, data BLOB NOT NULL)"
            )
            for ((trackId, embedding) in rows) {
                database.insertOrThrow(
                    "embeddings_clamp3",
                    null,
                    ContentValues().apply {
                        put("track_id", trackId)
                        put("embedding", EmbeddingDatabase.floatArrayToBlob(embedding))
                    },
                )
            }
        }
    }

    private fun copyExactBase(
        sourceDirectory: File,
        baseDirectory: File,
        rows: List<Pair<Long, FloatArray>>,
    ): V2ExactGraphIncrementalBase {
        baseDirectory.deleteRecursively()
        assertTrue(baseDirectory.mkdirs())
        val embedding = File(sourceDirectory, "clamp3.emb")
            .copyTo(File(baseDirectory, "clamp3.emb"), overwrite = true)
        val graph = File(sourceDirectory, "graph.bin")
            .copyTo(File(baseDirectory, "graph.bin"), overwrite = true)
        return V2ExactGraphIncrementalBase(
            generationId = "instrumented-exact-base",
            embeddingFile = embedding,
            graphFile = graph,
            trackCount = rows.size,
            embeddingDimension = rows.first().second.size,
            embeddingByteLength = embedding.length(),
            graphByteLength = graph.length(),
            embeddingSha256 = V2FileSha256.digest(embedding),
            graphSha256 = V2FileSha256.digest(graph),
        )
    }

    private fun assertSameLengthExactBaseMutationFallsBack(assetName: String) {
        val context = ApplicationProvider.getApplicationContext<android.content.Context>()
        val suffix = assetName.substringBefore('.')
        val dbFile = File(context.cacheDir, "graph-base-$suffix-mutation.db")
        val workDir = File(context.cacheDir, "graph-base-$suffix-mutation-work")
        val baseDir = File(context.cacheDir, "graph-base-$suffix-mutation-base")
        val baseRows = (1L..7L).map { id ->
            id to FloatArray(16) { column ->
                0.1f + (((id.toInt() * 13 + column * 17) % 19) / 40f)
            }
        }
        createEmbeddingDatabase(dbFile, baseRows)

        try {
            val initialDb = EmbeddingDatabase.openReadWrite(dbFile)
            val exactBase = try {
                assertEquals(
                    V2GraphUpdateStrategy.FULL_REBUILD,
                    GraphUpdater(initialDb, workDir).rebuildIndices().strategy,
                )
                copyExactBase(workDir, baseDir, baseRows)
            } finally {
                initialDb.close()
            }
            val mutated = File(baseDir, assetName)
            val originalLength = mutated.length()
            val bytes = mutated.readBytes()
            bytes[bytes.lastIndex] = (bytes.last().toInt() xor 0x01).toByte()
            mutated.writeBytes(bytes)
            assertEquals(originalLength, mutated.length())
            if (assetName == "graph.bin") {
                assertFalse(exactBase.graphSha256 == V2FileSha256.digest(mutated))
            } else {
                assertFalse(exactBase.embeddingSha256 == V2FileSha256.digest(mutated))
            }

            val targetDb = EmbeddingDatabase.openReadWrite(dbFile)
            try {
                targetDb.insertEmbedding(
                    "embeddings_clamp3",
                    8L,
                    FloatArray(16) { column -> 0.2f + column / 128f },
                )
                val result = GraphUpdater(targetDb, workDir).rebuildIndices(
                    exactBase = exactBase,
                )
                assertEquals(V2GraphUpdateStrategy.FULL_REBUILD, result.strategy)
            } finally {
                targetDb.close()
            }
        } finally {
            dbFile.delete()
            workDir.deleteRecursively()
            baseDir.deleteRecursively()
        }
    }
}
