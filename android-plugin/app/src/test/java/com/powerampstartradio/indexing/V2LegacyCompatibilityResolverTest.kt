package com.powerampstartradio.indexing

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class V2LegacyCompatibilityResolverTest {
    @Test
    fun `music relative path reconciles different roots before weaker metadata`() {
        val provider = provider(
            id = 10,
            path = "/storage/1234/Music/albums/A/01 - Song.flac",
            metadataKey = "changed tags|album|song|100000",
        )
        val database = database(
            id = 20,
            path = "C:\\Music\\albums\\A\\01 - Song.flac",
            metadataKey = "old tags|album|song|100000",
        )

        val result = V2LegacyCompatibilityResolver.resolve(listOf(provider), listOf(database))

        assertEquals(
            listOf(
                V2LegacyCompatibilityBinding(
                    10,
                    20,
                    V2LegacyCompatibilityEvidence.EXACT_MUSIC_RELATIVE_PATH,
                ),
            ),
            result.bindings,
        )
    }

    @Test
    fun `one legacy row and two same-tag remasters remain unresolved`() {
        val provider = listOf(
            provider(1, "/music/a/remaster.flac", "artist|album|song|100000"),
            provider(2, "/music/b/remaster.flac", "artist|album|song|100000"),
        )
        val database = listOf(
            database(30, "C:\\old\\song.flac", "artist|album|song|100000"),
        )

        val result = V2LegacyCompatibilityResolver.resolve(provider, database)

        assertTrue(result.bindings.isEmpty())
        assertEquals(setOf(1L, 2L), result.unmatchedPowerampFileIds)
        assertEquals(setOf(30L), result.unmatchedTrackIds)
    }

    @Test
    fun `a uniquely solvable but non-reciprocal path graph remains unresolved`() {
        val path = "/storage/Music/ambiguous.flac"
        val metadata = "artist|album|ambiguous|100000"
        val result = V2LegacyCompatibilityResolver.resolve(
            provider = listOf(
                provider(1, path, metadata, durationMs = 100_000),
                provider(2, path, metadata, durationMs = 104_000),
            ),
            database = listOf(
                database(10, path, metadata, durationMs = 102_000),
                database(20, path, metadata, durationMs = 108_000),
            ),
        )

        // P1 -> D1 and P2 -> {D1, D2} has a unique perfect matching, but neither edge is
        // initially reciprocal-unique. Migration must not infer that matching.
        assertTrue(result.bindings.isEmpty())
    }

    @Test
    fun `same path collision binds only duration-separated reciprocal pairs deterministically`() {
        val path = "/storage/Music/collision.flac"
        val providers = listOf(
            provider(2, path, "artist|album|second|120000", durationMs = 120_000),
            provider(1, path, "artist|album|first|100000", durationMs = 100_000),
        )
        val database = listOf(
            database(20, path, "old|tags|second|118000", durationMs = 118_000),
            database(10, path, "old|tags|first|102000", durationMs = 102_000),
        )

        val forward = V2LegacyCompatibilityResolver.resolve(providers, database)
        val reversed = V2LegacyCompatibilityResolver.resolve(
            providers.reversed(),
            database.reversed(),
        )

        assertEquals(
            listOf(
                V2LegacyCompatibilityBinding(
                    1,
                    10,
                    V2LegacyCompatibilityEvidence.EXACT_ABSOLUTE_PATH,
                ),
                V2LegacyCompatibilityBinding(
                    2,
                    20,
                    V2LegacyCompatibilityEvidence.EXACT_ABSOLUTE_PATH,
                ),
            ),
            forward.bindings,
        )
        assertEquals(forward, reversed)
    }

    @Test
    fun `forced path binding does not let metadata consume the remaining pair`() {
        val sameMetadata = "artist|album|song|100000"
        val provider = listOf(
            provider(1, "/music/exact.flac", sameMetadata),
            provider(2, "/music/other.flac", sameMetadata),
        )
        val database = listOf(
            database(10, "/music/exact.flac", sameMetadata),
            database(20, "/legacy/unrelated.flac", sameMetadata),
        )

        val result = V2LegacyCompatibilityResolver.resolve(provider, database)

        assertEquals(
            listOf(
                V2LegacyCompatibilityBinding(
                    1,
                    10,
                    V2LegacyCompatibilityEvidence.EXACT_ABSOLUTE_PATH,
                ),
            ),
            result.bindings,
        )
        assertEquals(setOf(2L), result.unmatchedPowerampFileIds)
        assertEquals(setOf(20L), result.unmatchedTrackIds)
    }

    @Test
    fun `CUE logical rows never inherit imported path or metadata coverage`() {
        val provider = provider(
            id = 1,
            path = "/storage/Music/album-image.flac",
            metadataKey = "artist|album|cue song|100000",
            compatibilityEligible = false,
        )
        val database = database(
            id = 10,
            path = "C:\\Music\\album-image.flac",
            metadataKey = provider.metadataKey,
        )

        val result = V2LegacyCompatibilityResolver.resolve(listOf(provider), listOf(database))

        assertTrue(result.bindings.isEmpty())
        assertEquals(setOf(1L), result.unmatchedPowerampFileIds)
    }

    @Test
    fun `CUE counterparts become repairs while unmatched logical rows stay new`() {
        val path = "/storage/Music/Fixture Artist/album-image.flac"
        val providers = listOf(
            provider(
                1,
                path,
                "fixture artist|album|first|100000",
                requiresSpanSpecificRebuild = true,
            ),
            provider(
                2,
                path,
                "fixture artist|album|second|120000",
                durationMs = 120_000,
                requiresSpanSpecificRebuild = true,
            ),
            provider(
                3,
                path,
                "fixture artist|album|new cue|90000",
                durationMs = 90_000,
                requiresSpanSpecificRebuild = true,
            ),
        )
        val databases = listOf(
            database(
                10,
                "C:\\Music\\Fixture Artist\\album-image.flac",
                "fixture artist|album|first|100000",
            ),
            database(
                20,
                "C:\\Music\\Fixture Artist\\album-image.flac",
                "fixture artist|album|second|120000",
                120_000,
            ),
        )

        val result = V2LegacyCompatibilityResolver.resolve(providers, databases)

        assertTrue(result.bindings.isEmpty())
        assertEquals(setOf(1L, 2L), result.repairBindings.map { it.powerampFileId }.toSet())
        assertEquals(setOf(3L), result.unmatchedPowerampFileIds)
        assertTrue(result.unmatchedTrackIds.isEmpty())
    }

    @Test
    fun `unique same path and exact tags survive broken provider timing`() {
        val result = V2LegacyCompatibilityResolver.resolve(
            listOf(
                provider(
                    1,
                    "/storage/Music/show.flac",
                    "artist|show|part|200000",
                    durationMs = 200_000,
                ),
            ),
            listOf(
                database(
                    10,
                    "C:\\Music\\show.flac",
                    "artist|show|part|100000",
                    durationMs = 100_000,
                ),
            ),
        )

        assertEquals(
            listOf(
                V2LegacyCompatibilityBinding(
                    1,
                    10,
                    V2LegacyCompatibilityEvidence.EXACT_MUSIC_RELATIVE_PATH,
                ),
            ),
            result.bindings,
        )
        assertTrue(result.pathTimingConflictBindings.isEmpty())
        assertTrue(result.unmatchedPowerampFileIds.isEmpty())
        assertTrue(result.unmatchedTrackIds.isEmpty())
    }

    @Test
    fun `missing provider timing becomes reviewable reciprocal path attention`() {
        val result = V2LegacyCompatibilityResolver.resolve(
            listOf(
                provider(
                    1,
                    "/storage/Music/albums/Tool/10 - Blank.flac",
                    "||blank|0",
                    durationMs = 0,
                ),
            ),
            listOf(
                database(
                    10,
                    "C:\\backups\\Music\\albums\\Tool\\10 - Blank.flac",
                    "tool|undertow|blank|1000",
                    durationMs = 1_000,
                ),
            ),
        )

        assertTrue(result.bindings.isEmpty())
        assertEquals(
            listOf(
                V2LegacyCompatibilityBinding(
                    1,
                    10,
                    V2LegacyCompatibilityEvidence
                        .EXACT_MUSIC_RELATIVE_PATH_PROVIDER_TIMING_UNAVAILABLE,
                ),
            ),
            result.providerTimingUnavailableBindings,
        )
        assertTrue(result.pathTimingConflictBindings.isEmpty())
        assertTrue(result.unmatchedPowerampFileIds.isEmpty())
        assertTrue(result.unmatchedTrackIds.isEmpty())
    }

    @Test
    fun `positive incompatible timing remains a replacement candidate`() {
        val result = V2LegacyCompatibilityResolver.resolve(
            listOf(
                provider(
                    1,
                    "/storage/Music/show.flac",
                    "new artist|new album|new recording|200000",
                    durationMs = 200_000,
                ),
            ),
            listOf(
                database(
                    10,
                    "C:\\Music\\show.flac",
                    "old artist|old album|old recording|100000",
                    durationMs = 100_000,
                ),
            ),
        )

        assertTrue(result.bindings.isEmpty())
        assertTrue(result.providerTimingUnavailableBindings.isEmpty())
        assertEquals(
            listOf(
                V2LegacyCompatibilityBinding(
                    1,
                    10,
                    V2LegacyCompatibilityEvidence.EXACT_PATH_TIMING_CONFLICT,
                ),
            ),
            result.pathTimingConflictBindings,
        )
    }

    @Test
    fun `matching tags do not activate an exact path whose provider timing is unavailable`() {
        val result = V2LegacyCompatibilityResolver.resolve(
            listOf(
                provider(
                    1,
                    "/storage/Music/unknown.flac",
                    "artist|album|unknown|0",
                    durationMs = 0,
                ),
            ),
            listOf(
                database(
                    10,
                    "/storage/Music/unknown.flac",
                    "artist|album|unknown|100000",
                    durationMs = 100_000,
                ),
            ),
        )

        assertTrue(result.bindings.isEmpty())
        assertEquals(1, result.providerTimingUnavailableBindings.size)
    }

    @Test
    fun `duplicate paths cannot manufacture unavailable timing attention`() {
        val providers = listOf(
            provider(1, "/storage/Music/duplicate.flac", "||duplicate|0", durationMs = 0),
            provider(2, "/storage/Music/duplicate.flac", "||duplicate|0", durationMs = 0),
        )
        val databases = listOf(
            database(
                10,
                "C:\\Music\\duplicate.flac",
                "artist|album|duplicate|100000",
            ),
        )

        val result = V2LegacyCompatibilityResolver.resolve(providers, databases)

        assertTrue(result.providerTimingUnavailableBindings.isEmpty())
        assertEquals(setOf(1L, 2L), result.unmatchedPowerampFileIds)
        assertEquals(setOf(10L), result.unmatchedTrackIds)
    }

    @Test
    fun `CUE shaped missing timing never becomes ordinary path attention`() {
        val result = V2LegacyCompatibilityResolver.resolve(
            listOf(
                provider(
                    1,
                    "/storage/Music/image.flac",
                    "artist|album|part|0",
                    durationMs = 0,
                    compatibilityEligible = false,
                    requiresSpanSpecificRebuild = true,
                ),
            ),
            listOf(
                database(
                    10,
                    "C:\\Music\\image.flac",
                    "artist|album|part|100000",
                ),
            ),
        )

        assertTrue(result.providerTimingUnavailableBindings.isEmpty())
    }

    @Test
    fun `input order cannot change bindings`() {
        val providers = listOf(
            provider(3, "/storage/Music/c/3.flac", "c|c|c|100000"),
            provider(1, "/storage/Music/a/1.flac", "a|a|a|100000"),
            provider(2, "/storage/Music/b/2.flac", "b|b|b|100000"),
        )
        val database = listOf(
            database(30, "C:\\Music\\c\\3.flac", "c|c|c|100000"),
            database(10, "C:\\Music\\a\\1.flac", "a|a|a|100000"),
            database(20, "C:\\Music\\b\\2.flac", "b|b|b|100000"),
        )

        val forward = V2LegacyCompatibilityResolver.resolve(providers, database)
        val reversed = V2LegacyCompatibilityResolver.resolve(providers.reversed(), database.reversed())

        assertEquals(forward, reversed)
    }

    @Test
    fun `large duration disagreement is accepted only with exact path and tags`() {
        val provider = provider(1, "/storage/Music/show.mp3", "a|a|show|200000", 200_000)
        val database = database(10, "C:\\Music\\show.mp3", "a|a|show|100000", 100_000)

        val result = V2LegacyCompatibilityResolver.resolve(listOf(provider), listOf(database))

        assertEquals(1, result.bindings.size)
    }

    @Test
    fun `unknown duration never becomes compatibility coverage`() {
        val provider = provider(
            1,
            "/storage/Music/unknown.flac",
            "a|a|unknown|0",
            durationMs = 0,
            compatibilityEligible = false,
        )
        val database = database(10, "C:\\Music\\unknown.flac", "a|a|unknown|0", 0)

        val result = V2LegacyCompatibilityResolver.resolve(listOf(provider), listOf(database))

        assertTrue(result.bindings.isEmpty())
    }

    private fun provider(
        id: Long,
        path: String,
        metadataKey: String,
        durationMs: Int = 100_000,
        compatibilityEligible: Boolean = true,
        requiresSpanSpecificRebuild: Boolean = false,
    ) = V2LegacyProviderCandidate(
        powerampFileId = id,
        normalizedPhysicalPath = path,
        offsetMs = 0,
        durationMs = durationMs,
        metadataKey = metadataKey,
        compatibilityEligible = compatibilityEligible,
        requiresSpanSpecificRebuild = requiresSpanSpecificRebuild,
    )

    private fun database(
        id: Long,
        path: String,
        metadataKey: String,
        durationMs: Int = 100_000,
    ) = V2LegacyDatabaseCandidate(
        trackId = id,
        normalizedPath = path,
        durationMs = durationMs,
        metadataKey = metadataKey,
    )
}
