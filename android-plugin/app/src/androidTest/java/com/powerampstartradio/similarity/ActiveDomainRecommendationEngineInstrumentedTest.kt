package com.powerampstartradio.similarity

import android.content.Context
import android.database.sqlite.SQLiteDatabase
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.powerampstartradio.data.EmbeddingDatabase
import com.powerampstartradio.data.EmbeddingIndex
import com.powerampstartradio.data.StableTrackIdentityCatalog
import com.powerampstartradio.indexing.V2ActiveLibraryBinding
import com.powerampstartradio.indexing.V2ActiveLibraryBindingEvidence
import com.powerampstartradio.indexing.V2ActiveLibraryCatalog
import com.powerampstartradio.indexing.V2ActiveLibraryGenerationBinding
import com.powerampstartradio.indexing.V2ActiveLibraryQuarantineReason
import com.powerampstartradio.indexing.V2ActiveLibraryQuarantinedTrack
import com.powerampstartradio.ui.DecaySchedule
import com.powerampstartradio.ui.DriftMode
import com.powerampstartradio.ui.FindMusicOperator
import com.powerampstartradio.ui.FindMusicQuerySpec
import com.powerampstartradio.ui.FindMusicSongAnchor
import com.powerampstartradio.ui.RadioConfig
import com.powerampstartradio.ui.SeedSpec
import com.powerampstartradio.ui.SeedType
import com.powerampstartradio.ui.SelectionMode
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.Closeable
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.UUID
import kotlin.math.cos
import kotlin.math.sin

@RunWith(AndroidJUnit4::class)
class ActiveDomainRecommendationEngineInstrumentedTest {
    @Test
    fun everySingleSeedModeUsesOnlyTheActivePoolAndRepeatsExactly() = withFixture { fixture ->
        runBlocking {
            val seedSimilarities = fixture.index.computeAllSimilarities(fixture.embedding(SEED_ID))
            assertTrue(
                "fixture must make quarantined rows outrank the best active candidate",
                fixture.score(seedSimilarities, POISON_IDS.first()) >
                    fixture.score(seedSimilarities, ACTIVE_CANDIDATE_IDS.first()),
            )

            val closest = fixture.repeat(
                config(SelectionMode.CLOSEST),
            )
            assertEquals(ACTIVE_CANDIDATE_IDS.take(QUEUE_SIZE), closest.ids)
            assertEquals(listOf(1, 2, 3), closest.candidateRanks)
            assertEquals(listOf(1, 2, 3), closest.seedRanks)

            val mmr = fixture.repeat(
                config(SelectionMode.MMR).copy(diversityLambda = 1f),
            )
            assertEquals(ACTIVE_CANDIDATE_IDS.take(QUEUE_SIZE), mmr.ids)
            assertEquals(listOf(1, 2, 3), mmr.candidateRanks)

            val dpp = fixture.repeat(config(SelectionMode.DPP))
            val referenceDppIds = referenceGreedyDppIds(
                fixture = fixture,
                candidateIds = ACTIVE_CANDIDATE_IDS,
                numSelect = QUEUE_SIZE,
            )
            assertEquals(listOf(40L, 60L, 70L), referenceDppIds)
            assertEquals(referenceDppIds, dpp.ids)
            assertEquals(listOf(1, 3, 4), dpp.candidateRanks)
            lateinit var dppEvidence: DppSelectionEvidence
            lateinit var domainEvidence: RecommendationDomainEvidence
            val evidencedDpp = fixture.engine.generatePlaylist(
                seedTrackId = SEED_ID,
                config = config(SelectionMode.DPP),
                onDppSelectionEvidence = { dppEvidence = it },
                onRecommendationDomainEvidence = { domainEvidence = it },
            )
            assertEquals(referenceDppIds, evidencedDpp.map { it.track.id })
            assertEquals(ACTIVE_CANDIDATE_IDS.size, dppEvidence.completeCandidateDomainCount)
            assertEquals(
                ACTIVE_CANDIDATE_IDS.size,
                domainEvidence.seedExcludedCandidateIdentityCount,
            )
            assertTrue(dppEvidence.reproducedFullDomainGreedySequence)

            for (driftMode in DriftMode.entries) {
                val drift = fixture.repeat(
                    config(SelectionMode.MMR).copy(
                        driftEnabled = true,
                        driftMode = driftMode,
                        diversityLambda = 1f,
                        anchorStrength = 0.6f,
                        anchorDecay = DecaySchedule.NONE,
                        momentumBeta = 0.7f,
                    ),
                )
                assertEquals(ACTIVE_CANDIDATE_IDS.first(), drift.ids.first())
                assertEquals(1, drift.candidateRanks.first())
                assertEquals(1, drift.seedRanks.first())
                assertTrue(drift.candidateRanks.all { it in 1..CANDIDATE_POOL_SIZE })
                assertTrue(drift.driftRanks.all { it in 1..ACTIVE_CANDIDATE_IDS.size })
            }

            lateinit var firstShuffleEvidence: UniformShuffleIdentityEvidence
            lateinit var secondShuffleEvidence: UniformShuffleIdentityEvidence
            val shuffleConfig = config(SelectionMode.UNIFORM_SHUFFLE).copy(
                shuffleSeed = 0x1234_5678_9abc_defL,
            )
            val firstShuffle = fixture.engine.generatePlaylist(
                seedTrackId = SEED_ID,
                config = shuffleConfig,
                onUniformShuffleIdentityEvidence = { firstShuffleEvidence = it },
            )
            val secondShuffle = fixture.engine.generatePlaylist(
                seedTrackId = SEED_ID,
                config = shuffleConfig,
                onUniformShuffleIdentityEvidence = { secondShuffleEvidence = it },
            )
            assertSameResult(firstShuffle, secondShuffle)
            assertOnlyActive(firstShuffle)
            assertEquals(ACTIVE_IDS.size, firstShuffleEvidence.libraryTrackCount)
            assertEquals(0, firstShuffleEvidence.stableLibraryTrackCount)
            assertEquals(ACTIVE_IDS.size, firstShuffleEvidence.legacyLibraryTrackCount)
            assertEquals(QUEUE_SIZE, firstShuffleEvidence.selectedLegacyTrackCount)
            assertEquals(firstShuffleEvidence, secondShuffleEvidence)
        }
    }

    @Test
    fun addedDateSelectionStaysFilteredWhileEverySeedDistanceRankUsesTheFullActiveDomain() =
        withFixture { fixture ->
            runBlocking {
                val filteredConfig = { mode: SelectionMode ->
                    config(mode).copy(libraryAddedDays = 1)
                }
                val modes = listOf(
                    filteredConfig(SelectionMode.CLOSEST),
                    filteredConfig(SelectionMode.MMR).copy(diversityLambda = 1f),
                    filteredConfig(SelectionMode.DPP),
                    filteredConfig(SelectionMode.RANDOM_WALK),
                    filteredConfig(SelectionMode.UNIFORM_SHUFFLE).copy(
                        shuffleSeed = 0x1234_5678_9abc_defL,
                    ),
                )

                modes.forEach { modeConfig ->
                    lateinit var domainEvidence: RecommendationDomainEvidence
                    val tracks = fixture.engine.generatePlaylist(
                        seedTrackId = SEED_ID,
                        config = modeConfig,
                        requestReferenceEpochSecond = REQUEST_REFERENCE_EPOCH_SECOND,
                        onRecommendationDomainEvidence = { domainEvidence = it },
                    )

                    assertTrue(
                        "${modeConfig.selectionMode} returned no eligible tracks",
                        tracks.isNotEmpty(),
                    )
                    assertTrue(
                        "${modeConfig.selectionMode} left the added-date candidate domain",
                        tracks.all { it.track.id in RECENT_CANDIDATE_IDS },
                    )
                    tracks.forEach { track ->
                        assertEquals(
                            EXPECTED_GLOBAL_SEED_RANK.getValue(track.track.id),
                            track.seedRank,
                        )
                    }
                    assertEquals(
                        RECENT_CANDIDATE_IDS.size,
                        domainEvidence.seedExcludedCandidateIdentityCount,
                    )
                    assertEquals(
                        ACTIVE_CANDIDATE_IDS.size,
                        domainEvidence.seedExcludedActiveIdentityCount,
                    )
                }

                val drift = fixture.engine.generatePlaylist(
                    seedTrackId = SEED_ID,
                    config = filteredConfig(SelectionMode.MMR).copy(
                        driftEnabled = true,
                        diversityLambda = 1f,
                        anchorStrength = 0.6f,
                        anchorDecay = DecaySchedule.NONE,
                    ),
                    requestReferenceEpochSecond = REQUEST_REFERENCE_EPOCH_SECOND,
                )
                assertEquals(RECENT_CANDIDATE_IDS.toSet(), drift.map { it.track.id }.toSet())
                assertEquals(4, drift.first().seedRank)
                assertEquals(
                    "The first evolving query is the original seed, so both global ranks agree",
                    drift.first().seedRank,
                    drift.first().driftRank,
                )
            }
        }

    @Test
    fun allOfPercentilesAndRanksAreComputedInsideTheActiveCorpus() = withFixture { fixture ->
        runBlocking {
            val seed = SeedSpec(
                embedding = fixture.embedding(SEED_ID),
                weight = 1f,
                label = "active seed",
                type = SeedType.SONG,
                trackId = SEED_ID,
            )
            val query = FindMusicQuerySpec(
                operator = FindMusicOperator.ALL_OF,
                songSeeds = listOf(
                    FindMusicSongAnchor(
                        trackId = SEED_ID,
                        artist = "Artist $SEED_ID",
                        title = "Track $SEED_ID",
                        weight = 1f,
                        negative = false,
                    ),
                ),
                resultLimit = QUEUE_SIZE,
            )
            val config = config(SelectionMode.MMR)

            val first = fixture.engine.generateComposedAllOfPlaylist(
                seeds = listOf(seed),
                querySpec = query,
                config = config,
            )
            val second = fixture.engine.generateComposedAllOfPlaylist(
                seeds = listOf(seed),
                querySpec = query,
                config = config,
            )

            assertEquals(ACTIVE_CANDIDATE_IDS.take(QUEUE_SIZE), first.tracks.map { it.track.id })
            assertSameResult(first.tracks, second.tracks, requireSeedRanks = false)
            assertEquals(
                listOf(6f / 7f, 5f / 7f, 4f / 7f),
                first.tracks.map { track ->
                    requireNotNull(track.composedEvidence).ingredientPercentiles.single()
                },
            )
            assertEquals(
                listOf(1, 2, 3),
                first.tracks.map { requireNotNull(it.composedEvidence).objectiveRank },
            )
            assertEquals(QUEUE_SIZE, first.stableResultReduction.scannedRowCount)
            assertEquals(first.stableResultReduction, second.stableResultReduction)
        }
    }

    @Test
    fun missingCatalogAndInactiveSeedFailBeforeReturningAnyRanking() = withFixture { fixture ->
        val missingCatalog = RecommendationEngine(
            database = fixture.database,
            filesDir = fixture.workDir,
            pinnedAssets = fixture.assets,
        )
        val missingCatalogError = assertThrows(IllegalArgumentException::class.java) {
            runBlocking {
                missingCatalog.generatePlaylist(SEED_ID, config(SelectionMode.CLOSEST))
            }
        }
        assertTrue(missingCatalogError.message.orEmpty().contains("active Poweramp library catalog"))

        val inactiveSeedError = assertThrows(IllegalArgumentException::class.java) {
            runBlocking {
                fixture.engine.generatePlaylist(
                    POISON_IDS.first(),
                    config(SelectionMode.CLOSEST),
                )
            }
        }
        assertTrue(inactiveSeedError.message.orEmpty().contains("not in the active Poweramp library"))
    }

    private fun config(mode: SelectionMode) = RadioConfig(
        numTracks = QUEUE_SIZE,
        candidatePoolSize = CANDIDATE_POOL_SIZE,
        selectionMode = mode,
        artistLimitsEnabled = false,
    )

    private fun assertOnlyActive(
        tracks: List<SimilarTrack>,
        requireSeedRanks: Boolean = true,
    ) {
        assertEquals(QUEUE_SIZE, tracks.size)
        assertTrue(tracks.all { it.track.id in ACTIVE_IDS })
        assertTrue(tracks.none { it.track.id in POISON_IDS })
        if (requireSeedRanks) {
            assertTrue(
                tracks.all {
                    requireNotNull(it.seedRank) in 1..ACTIVE_CANDIDATE_IDS.size
                },
            )
        }
    }

    private fun assertSameResult(
        first: List<SimilarTrack>,
        second: List<SimilarTrack>,
        requireSeedRanks: Boolean = true,
    ) {
        assertEquals(first.map { it.track.id }, second.map { it.track.id })
        assertEquals(first.map { it.similarity }, second.map { it.similarity })
        assertEquals(first.map { it.similarityToSeed }, second.map { it.similarityToSeed })
        assertEquals(first.map { it.candidateRank }, second.map { it.candidateRank })
        assertEquals(first.map { it.seedRank }, second.map { it.seedRank })
        assertEquals(first.map { it.driftRank }, second.map { it.driftRank })
        assertOnlyActive(first, requireSeedRanks)
        assertOnlyActive(second, requireSeedRanks)
    }

    /** Independent small-matrix determinant oracle for the fixture's greedy DPP sequence. */
    private fun referenceGreedyDppIds(
        fixture: Fixture,
        candidateIds: List<Long>,
        numSelect: Int,
    ): List<Long> {
        val seed = fixture.embedding(SEED_ID)
        val embeddings = candidateIds.associateWith(fixture::embedding)
        val quality = candidateIds.associateWith { trackId ->
            dot(seed, embeddings.getValue(trackId)).coerceAtLeast(0.0)
        }
        val selected = mutableListOf<Long>()
        repeat(minOf(numSelect, candidateIds.size)) {
            var bestId: Long? = null
            var bestDeterminant = Double.NEGATIVE_INFINITY
            candidateIds.filterNot(selected::contains).forEach { candidateId ->
                val trial = selected + candidateId
                val determinant = determinant(
                    Array(trial.size) { row ->
                        DoubleArray(trial.size) { column ->
                            val left = trial[row]
                            val right = trial[column]
                            quality.getValue(left) * quality.getValue(right) *
                                dot(
                                    embeddings.getValue(left),
                                    embeddings.getValue(right),
                                )
                        }
                    },
                )
                if (determinant > bestDeterminant) {
                    bestDeterminant = determinant
                    bestId = candidateId
                }
            }
            selected += requireNotNull(bestId)
        }
        return selected
    }

    private fun dot(left: FloatArray, right: FloatArray): Double {
        require(left.size == right.size)
        return left.indices.sumOf { index -> left[index].toDouble() * right[index] }
    }

    private fun determinant(source: Array<DoubleArray>): Double {
        val matrix = Array(source.size) { source[it].copyOf() }
        var result = 1.0
        for (column in matrix.indices) {
            var pivot = column
            for (row in column + 1 until matrix.size) {
                if (kotlin.math.abs(matrix[row][column]) >
                    kotlin.math.abs(matrix[pivot][column])
                ) pivot = row
            }
            if (kotlin.math.abs(matrix[pivot][column]) <= 1e-12) return 0.0
            if (pivot != column) {
                val swap = matrix[pivot]
                matrix[pivot] = matrix[column]
                matrix[column] = swap
                result = -result
            }
            val diagonal = matrix[column][column]
            result *= diagonal
            for (row in column + 1 until matrix.size) {
                val scale = matrix[row][column] / diagonal
                for (entry in column + 1 until matrix.size) {
                    matrix[row][entry] -= scale * matrix[column][entry]
                }
            }
        }
        return result
    }

    private fun <T> withFixture(block: (Fixture) -> T): T {
        val context = ApplicationProvider.getApplicationContext<Context>()
        return Fixture.create(context).use(block)
    }

    private data class RepeatedResult(
        val ids: List<Long>,
        val candidateRanks: List<Int>,
        val seedRanks: List<Int>,
        val driftRanks: List<Int>,
    )

    private class Fixture private constructor(
        val workDir: File,
        val database: EmbeddingDatabase,
        val index: EmbeddingIndex,
        val assets: RecommendationAssetFiles,
        val engine: RecommendationEngine,
        private val ids: LongArray,
        private val embeddings: Array<FloatArray>,
    ) : Closeable {
        fun embedding(trackId: Long): FloatArray =
            embeddings[ids.indexOf(trackId)].copyOf()

        fun score(similarities: FloatArray, trackId: Long): Float =
            similarities[ids.indexOf(trackId)]

        suspend fun repeat(config: RadioConfig): RepeatedResult {
            val first = engine.generatePlaylist(SEED_ID, config)
            val second = engine.generatePlaylist(SEED_ID, config)
            assertEquals(first.map { it.track.id }, second.map { it.track.id })
            assertEquals(first.map { it.similarity }, second.map { it.similarity })
            assertEquals(first.map { it.similarityToSeed }, second.map { it.similarityToSeed })
            assertEquals(first.map { it.candidateRank }, second.map { it.candidateRank })
            assertEquals(first.map { it.seedRank }, second.map { it.seedRank })
            assertEquals(first.map { it.driftRank }, second.map { it.driftRank })
            assertEquals(QUEUE_SIZE, first.size)
            assertTrue(first.all { it.track.id in ACTIVE_IDS })
            assertTrue(first.none { it.track.id in POISON_IDS })
            assertTrue(
                first.all {
                    requireNotNull(it.seedRank) in 1..ACTIVE_CANDIDATE_IDS.size
                },
            )
            return RepeatedResult(
                ids = first.map { it.track.id },
                candidateRanks = first.map { requireNotNull(it.candidateRank) },
                seedRanks = first.map { requireNotNull(it.seedRank) },
                driftRanks = if (config.driftEnabled) {
                    first.map { requireNotNull(it.driftRank) }
                } else {
                    emptyList()
                },
            )
        }

        override fun close() {
            database.close()
            workDir.deleteRecursively()
        }

        companion object {
            fun create(context: Context): Fixture {
                val workDir = File(
                    context.cacheDir,
                    "active-engine-fixture-${UUID.randomUUID()}",
                ).apply { check(mkdirs()) }
                val ids = longArrayOf(10L, 20L, 30L, 40L, 50L, 60L, 70L, 80L, 90L)
                val angles = doubleArrayOf(0.0, 0.02, 0.04, 0.15, 0.35, 0.65, 1.0, 1.4, 2.0)
                val embeddings = Array(ids.size) { row ->
                    FloatArray(ids.size).apply {
                        this[0] = cos(angles[row]).toFloat()
                        if (row > 0) this[row] = sin(angles[row]).toFloat()
                    }
                }
                val databaseFile = File(workDir, "fixture.db")
                val embeddingFile = File(workDir, "fixture.emb")
                val graphFile = File(workDir, "fixture.graph")
                writeDatabase(databaseFile, ids, embeddings)
                writeEmbeddingIndex(embeddingFile, ids, embeddings)
                writeDenseGraph(graphFile, ids)

                val database = EmbeddingDatabase.open(databaseFile)
                val index = EmbeddingIndex.mmap(embeddingFile)
                val identities = StableTrackIdentityCatalog.load(workDir, database, index)
                val activeCatalog = V2ActiveLibraryCatalog(
                    generationBinding = V2ActiveLibraryGenerationBinding(
                        databaseGenerationId = identities.binding.generationId,
                        providerGenerationId = "synthetic-provider-generation-v1",
                    ),
                    bindings = ACTIVE_IDS.map { trackId ->
                        V2ActiveLibraryBinding(
                            trackId = trackId,
                            powerampFileId = 1_000L + trackId,
                            evidence = V2ActiveLibraryBindingEvidence.EXACT_V2_RECEIPT_SPAN,
                            createdAtEpochSecond = if (trackId in RECENT_CANDIDATE_IDS) {
                                REQUEST_REFERENCE_EPOCH_SECOND - 60L
                            } else {
                                REQUEST_REFERENCE_EPOCH_SECOND - 10L * 86_400L
                            },
                        )
                    },
                    quarantinedTracks = POISON_IDS.map { trackId ->
                        V2ActiveLibraryQuarantinedTrack(
                            trackId = trackId,
                            reason = V2ActiveLibraryQuarantineReason.NO_CURRENT_PROVIDER_BINDING,
                        )
                    },
                    unboundPowerampFileIds = emptyList(),
                )
                val assets = RecommendationAssetFiles(embeddingFile, graphFile = graphFile)
                return Fixture(
                    workDir = workDir,
                    database = database,
                    index = index,
                    assets = assets,
                    engine = RecommendationEngine(
                        database = database,
                        filesDir = workDir,
                        pinnedAssets = assets,
                        activeCatalog = activeCatalog,
                    ),
                    ids = ids,
                    embeddings = embeddings,
                )
            }

            private fun writeDatabase(
                file: File,
                ids: LongArray,
                embeddings: Array<FloatArray>,
            ) {
                SQLiteDatabase.openOrCreateDatabase(file, null).use { database ->
                    database.execSQL(
                        """
                        CREATE TABLE tracks (
                            id INTEGER PRIMARY KEY,
                            metadata_key TEXT NOT NULL,
                            filename_key TEXT NOT NULL,
                            artist TEXT,
                            album TEXT,
                            title TEXT,
                            duration_ms INTEGER,
                            file_path TEXT NOT NULL,
                            source TEXT DEFAULT 'desktop'
                        )
                        """.trimIndent(),
                    )
                    database.execSQL(
                        """
                        CREATE TABLE embeddings_clamp3 (
                            track_id INTEGER PRIMARY KEY,
                            embedding BLOB NOT NULL
                        )
                        """.trimIndent(),
                    )
                    ids.forEachIndexed { index, trackId ->
                        database.execSQL(
                            """
                            INSERT INTO tracks(
                                id, metadata_key, filename_key, artist, album, title,
                                duration_ms, file_path, source
                            ) VALUES (?, ?, ?, ?, 'Fixture', ?, 180000, ?, 'test')
                            """.trimIndent(),
                            arrayOf<Any>(
                                trackId,
                                "fixture|$trackId",
                                "track-$trackId.flac",
                                "Artist $trackId",
                                "Track $trackId",
                                "/fixture/track-$trackId.flac",
                            ),
                        )
                        database.execSQL(
                            "INSERT INTO embeddings_clamp3(track_id, embedding) VALUES (?, ?)",
                            arrayOf(
                                trackId,
                                EmbeddingDatabase.floatArrayToBlob(embeddings[index]),
                            ),
                        )
                    }
                }
            }

            private fun writeEmbeddingIndex(
                file: File,
                ids: LongArray,
                embeddings: Array<FloatArray>,
            ) {
                val dim = embeddings.first().size
                val size = 16 + ids.size * Long.SIZE_BYTES +
                    ids.size * dim * Float.SIZE_BYTES
                val buffer = ByteBuffer.allocate(size).order(ByteOrder.LITTLE_ENDIAN)
                buffer.putInt(0x424D4550)
                buffer.putInt(1)
                buffer.putInt(ids.size)
                buffer.putInt(dim)
                ids.forEach(buffer::putLong)
                embeddings.forEach { row -> row.forEach(buffer::putFloat) }
                file.writeBytes(buffer.array())
            }

            private fun writeDenseGraph(file: File, ids: LongArray) {
                val neighborsPerNode = minOf(5, ids.size - 1)
                val size = 8 + ids.size * Long.SIZE_BYTES +
                    ids.size * neighborsPerNode * (Int.SIZE_BYTES + Float.SIZE_BYTES)
                val buffer = ByteBuffer.allocate(size).order(ByteOrder.LITTLE_ENDIAN)
                buffer.putInt(ids.size)
                buffer.putInt(neighborsPerNode)
                ids.forEach(buffer::putLong)
                ids.indices.forEach { source ->
                    (1..neighborsPerNode).forEach { offset ->
                        val neighbor = (source + offset) % ids.size
                        buffer.putInt(neighbor)
                        buffer.putFloat(1f / neighborsPerNode)
                    }
                }
                file.writeBytes(buffer.array())
            }
        }
    }

    private companion object {
        const val SEED_ID = 10L
        const val QUEUE_SIZE = 3
        const val CANDIDATE_POOL_SIZE = 3
        const val REQUEST_REFERENCE_EPOCH_SECOND = 2_000_000L
        val POISON_IDS = setOf(20L, 30L)
        val ACTIVE_CANDIDATE_IDS = listOf(40L, 50L, 60L, 70L, 80L, 90L)
        val RECENT_CANDIDATE_IDS = listOf(70L, 80L, 90L)
        val EXPECTED_GLOBAL_SEED_RANK = ACTIVE_CANDIDATE_IDS
            .withIndex()
            .associate { (index, trackId) -> trackId to index + 1 }
        val ACTIVE_IDS = setOf(SEED_ID) + ACTIVE_CANDIDATE_IDS
    }
}
