package com.powerampstartradio.ui

import com.powerampstartradio.data.EmbeddedTrack
import com.powerampstartradio.poweramp.PowerampTrack
import com.powerampstartradio.poweramp.TrackMatcher
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class SeedDistanceEvidencePolicyTest {
    @Test
    fun `single-song seed rank exposes only listener-meaningful rank evidence`() {
        val evidence = SeedDistanceEvidencePolicy.evidenceOrNull(
            session = session(),
            track = row(similarityToSeed = 0.8765f, seedRank = 9),
        )

        assertEquals(9, requireNotNull(evidence).seedRank)
        assertEquals(100, evidence.rankingIdentityCount)
    }

    @Test
    fun `rank denominator is the exact persisted full active seed excluded identity domain`() {
        val evidence = SeedDistanceEvidencePolicy.evidenceOrNull(
            session = session().copy(
                eligibleCandidateIdentityCount = 500,
                seedRankingIdentityCount = 70_000,
            ),
            track = row(similarityToSeed = 0.75f, seedRank = 69_000),
        )

        assertEquals(70_000, requireNotNull(evidence).rankingIdentityCount)
    }

    @Test
    fun `missing or inconsistent rank denominator fails closed`() {
        assertNull(
            SeedDistanceEvidencePolicy.evidenceOrNull(
                session = session().copy(
                    seedRankingIdentityCount = null,
                    eligibleCandidateIdentityCount = null,
                ),
                track = row(similarityToSeed = 0.75f, seedRank = 9),
            ),
        )
        assertNull(
            SeedDistanceEvidencePolicy.evidenceOrNull(
                session = session().copy(seedRankingIdentityCount = 8),
                track = row(similarityToSeed = 0.75f, seedRank = 9),
            ),
        )
    }

    @Test
    fun `legacy finite-date rank fails closed but all-dates rank can use the old denominator`() {
        val finiteLegacy = session().copy(
            config = RadioConfig(libraryAddedDays = 17),
            eligibleCandidateIdentityCount = 8,
            seedRankingIdentityCount = null,
        )
        assertNull(
            SeedDistanceEvidencePolicy.evidenceOrNull(
                session = finiteLegacy,
                track = row(similarityToSeed = 0.75f, seedRank = 9),
            ),
        )
        assertNull(
            SeedDistanceEvidencePolicy.evidenceOrNull(
                session = finiteLegacy.copy(
                    config = RadioConfig(
                        libraryAddedRange = LibraryAddedRange.LAST_30_DAYS,
                    ),
                    eligibleCandidateIdentityCount = 100,
                ),
                track = row(similarityToSeed = 0.75f, seedRank = 9),
            ),
        )

        val allDatesLegacy = finiteLegacy.copy(
            config = RadioConfig(),
            eligibleCandidateIdentityCount = 100,
        )
        val evidence = SeedDistanceEvidencePolicy.evidenceOrNull(
            session = allDatesLegacy,
            track = row(similarityToSeed = 0.75f, seedRank = 9),
        )
        assertEquals(100, requireNotNull(evidence).rankingIdentityCount)

        val preFilterConfig = com.google.gson.Gson().fromJson(
            "{}",
            RadioConfig::class.java,
        )
        val preFilterEvidence = SeedDistanceEvidencePolicy.evidenceOrNull(
            session = allDatesLegacy.copy(config = preFilterConfig),
            track = row(similarityToSeed = 0.75f, seedRank = 9),
        )
        assertEquals(100, requireNotNull(preFilterEvidence).rankingIdentityCount)
    }

    @Test
    fun `direct composed and unranked rows cannot masquerade as seed distance`() {
        assertNull(
            SeedDistanceEvidencePolicy.evidenceOrNull(
                session = session().copy(isDirectQueue = true),
                track = row(similarityToSeed = 0f, seedRank = 1),
            ),
        )
        assertNull(
            SeedDistanceEvidencePolicy.evidenceOrNull(
                session = session().copy(composedQuerySpec = FindMusicQuerySpec()),
                track = row(similarityToSeed = 0.9f, seedRank = 1),
            ),
        )
        assertNull(
            SeedDistanceEvidencePolicy.evidenceOrNull(
                session = session().copy(matchType = TrackMatcher.MatchType.COMPOSED_QUERY),
                track = row(similarityToSeed = 0.9f, seedRank = 1),
            ),
        )
        assertNull(
            SeedDistanceEvidencePolicy.evidenceOrNull(
                session = session().copy(matchType = TrackMatcher.MatchType.NOT_APPLICABLE),
                track = row(similarityToSeed = 0.9f, seedRank = 1),
            ),
        )
        assertNull(
            SeedDistanceEvidencePolicy.evidenceOrNull(
                session = session(),
                track = row(similarityToSeed = 0f, seedRank = null),
            ),
        )
    }

    @Test
    fun `numeric noise passes validation but invalid objective-like values fail closed`() {
        val tolerated = SeedDistanceEvidencePolicy.evidenceOrNull(
            session = session(),
            track = row(similarityToSeed = 1.0005f, seedRank = 1),
        )
        assertEquals(1, requireNotNull(tolerated).seedRank)
        assertNull(
            SeedDistanceEvidencePolicy.evidenceOrNull(
                session = session(),
                track = row(similarityToSeed = 1.01f, seedRank = 1),
            ),
        )
    }

    private fun session() = RadioResult(
        seedTrack = PowerampTrack(
            realId = 3L,
            title = "Seed",
            artist = "Artist",
            album = "Album",
            durationMs = 180_000,
            path = "/music/seed.flac",
        ),
        matchType = TrackMatcher.MatchType.METADATA_EXACT,
        tracks = emptyList(),
        eligibleCandidateIdentityCount = 100,
        seedRankingIdentityCount = 100,
    )

    private fun row(similarityToSeed: Float, seedRank: Int?) = QueuedTrackResult(
        track = EmbeddedTrack(
            id = 4L,
            metadataKey = "metadata",
            filenameKey = "file",
            artist = "Artist",
            album = "Album",
            title = "Result",
            durationMs = 170_000,
            filePath = "/music/result.flac",
        ),
        similarity = similarityToSeed,
        similarityToSeed = similarityToSeed,
        seedRank = seedRank,
        status = QueueStatus.QUEUED,
    )
}
