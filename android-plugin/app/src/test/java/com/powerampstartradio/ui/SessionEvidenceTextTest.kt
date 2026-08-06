package com.powerampstartradio.ui

import com.powerampstartradio.data.EmbeddedTrack
import com.powerampstartradio.poweramp.PowerampTrack
import com.powerampstartradio.poweramp.TrackMatcher
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test
import java.util.Calendar
import java.util.Locale
import java.util.TimeZone

class SessionEvidenceTextTest {
    @Test
    fun `replayed session title says replay once`() {
        val replay = radioSession().copy(
            seedTrack = radioSession().seedTrack.copy(title = "Replay: Miss Melodia"),
            delivery = QueueDeliverySummary(
                origin = QueueOrigin.HISTORY_REQUEUE,
                requestedCount = 0,
                rankedCount = 0,
                resolvedCount = 0,
                verifiedCount = 0,
                notInLibraryCount = 0,
                queueFailedCount = 0,
                verificationComplete = true,
            ),
        )

        assertEquals("Miss Melodia", SessionEvidenceText.seedTitle(replay))
        assertEquals("Seed", SessionEvidenceText.seedTitle(radioSession()))
    }

    @Test
    fun `seed identity uses every persisted human label and suppresses blanks`() {
        assertEquals(
            "Artist \u00b7 Album",
            SessionEvidenceText.seedIdentity(" Artist ", " Album "),
        )
        assertEquals("Artist", SessionEvidenceText.seedIdentity("Artist", "Artist"))
        assertEquals("Album", SessionEvidenceText.seedIdentity(null, "Album"))
        assertNull(SessionEvidenceText.seedIdentity(" ", null))
    }

    @Test
    fun `artist summary exposes the exact configured constraints`() {
        assertEquals(
            "Artist-credit limits off",
            SessionEvidenceText.artistConstraints(RadioConfig(artistLimitsEnabled = false)),
        )
        assertEquals(
            "Artist-credit limits \u00b7 max 4 tracks with the same credit \u00b7 " +
                "2 tracks between the same credit",
            SessionEvidenceText.artistConstraints(
                RadioConfig(
                    artistLimitsEnabled = true,
                    maxPerArtist = 4,
                    minArtistSpacing = 2,
                ),
            ),
        )
        assertEquals(
            "Artist-credit limits \u00b7 max 7 tracks with the same credit \u00b7 no spacing limit",
            SessionEvidenceText.artistConstraints(
                RadioConfig(
                    artistLimitsEnabled = true,
                    maxPerArtist = 7,
                    minArtistSpacing = 0,
                ),
            ),
        )
        assertEquals(
            "Artist-credit limits \u00b7 max 1 track with the same credit \u00b7 no spacing limit",
            SessionEvidenceText.artistConstraints(
                RadioConfig(
                    artistLimitsEnabled = true,
                    maxPerArtist = 1,
                    minArtistSpacing = 0,
                ),
            ),
        )
    }

    @Test
    fun `text result history exposes library rank in human scale`() {
        val session = evidence(
            FindMusicQuerySpec(
                textIngredients = listOf(
                    FindMusicTextIngredient("sleep", 1f, negative = false),
                ),
                resultLimit = 30,
            ),
            activeTrackCount = 80_000,
        )
        val track = FindMusicTrackEvidence(
            displayedRank = 3,
            objectiveRank = 8,
            resultScore = 0.31234f,
            rankingScore = 0.31234f,
        )

        assertEquals(
            "Text match #8",
            SessionEvidenceText.findMusicTrack(session, track),
        )
        assertEquals("Text \u00b7 Closest", SessionEvidenceText.findMusicMode(session))
        assertEquals(
            "Cosine-ranks every candidate recording against the text, strongest first. " +
                "\u00b7 Compared across 80,000 candidate recordings",
            SessionEvidenceText.findMusicQuery(session),
        )
    }

    @Test
    fun `Find Music history reports an exact Poweramp added-date window`() {
        val session = evidence(
            FindMusicQuerySpec(
                textIngredients = listOf(
                    FindMusicTextIngredient("sleep", 1f, negative = false),
                ),
                resultLimit = 30,
                libraryAddedDays = 17,
            ),
            activeTrackCount = 4_000,
        )

        assertEquals(
            "Cosine-ranks every candidate recording against the text, strongest first. " +
                "\u00b7 Compared across 4,000 candidate recordings\n" +
                "Candidates \u00b7 Last 17 days",
            SessionEvidenceText.findMusicQuery(session),
        )
    }

    @Test
    fun `composed result history exposes weights and per ingredient ranking evidence`() {
        val query = FindMusicQuerySpec(
            operator = FindMusicOperator.ALL_OF,
            textIngredients = listOf(
                FindMusicTextIngredient("ambient", 0.693f, negative = false),
                FindMusicTextIngredient("busy", 0.307f, negative = true),
            ),
            resultLimit = 20,
        )
        val session = evidence(query, activeTrackCount = 500)
        val track = FindMusicTrackEvidence(
            displayedRank = 1,
            objectiveRank = 2,
            resultScore = 0.81234f,
            rankingScore = 0.81234f,
            ingredientPercentiles = listOf(0.9f, 0.75f),
        )

        assertEquals(
            "All of \u00b7 ambient 69.3% priority \u00b7 Less like: busy 30.7% priority\n" +
                "Ranks compare 500 recordings in this library scope",
            SessionEvidenceText.findMusicQuery(session),
        )
        assertEquals(
            "Overall match \u00b7 #2 \u00b7 ambient match #51 \u00b7 Less like: busy match #126",
            SessionEvidenceText.findMusicTrack(session, track),
        )
    }

    @Test
    fun `ranking evidence is suppressed when its exact domain is absent`() {
        val query = FindMusicQuerySpec(
            operator = FindMusicOperator.ALL_OF,
            textIngredients = listOf(
                FindMusicTextIngredient("ambient", 0.7f, negative = false),
                FindMusicTextIngredient("busy", 0.3f, negative = true),
            ),
            resultLimit = 20,
        )
        val session = evidence(query, 500).copy(objectiveRankingDomainCount = null)
        val track = FindMusicTrackEvidence(
            displayedRank = 1,
            objectiveRank = 2,
            resultScore = 0.8f,
            rankingScore = 0.8f,
            ingredientPercentiles = listOf(0.9f, 0.75f),
        )

        assertNull(SessionEvidenceText.findMusicTrack(session, track))
        assertNull(SessionEvidenceText.findMusicTrack(
            session.copy(
                objectiveRankingDomainCount = 500,
                ingredientRankingDomainCount = null,
            ),
            track,
        ))
        assertNull(
            SessionEvidenceText.findMusicTrack(
                session.copy(
                    objectiveRankingDomainCount = 500,
                    ingredientRankingDomainCount = 500,
                ),
                track.copy(ingredientPercentiles = listOf(0.9f)),
            ),
        )
    }

    @Test
    fun `All-of names distinct overall and ingredient ranking domains`() {
        val query = FindMusicQuerySpec(
            operator = FindMusicOperator.ALL_OF,
            textIngredients = listOf(FindMusicTextIngredient("sleep", 0.5f, negative = false)),
            songSeeds = listOf(
                FindMusicSongAnchor(
                    trackId = 7L,
                    artist = "Bonobo",
                    title = "Drift",
                    weight = 0.5f,
                    negative = false,
                ),
            ),
            resultLimit = 20,
        )
        val session = evidence(query, 80_335).copy(objectiveRankingDomainCount = 80_334)

        assertEquals(
            "All of \u00b7 sleep 50% priority \u00b7 Bonobo - Drift 50% priority\n" +
                "Overall rank among 80,334 eligible recordings \u00b7 ingredient ranks across " +
                "80,335 recordings in this library scope",
            SessionEvidenceText.findMusicQuery(session),
        )
    }

    @Test
    fun `Refine session reports its asymmetric recipe domain and ingredient ranks`() {
        val query = FindMusicQuerySpec(
            operator = FindMusicOperator.REFINE,
            textIngredients = listOf(
                FindMusicTextIngredient("ambient", 0.5f, negative = false),
                FindMusicTextIngredient("guitar", 0.5f, negative = false),
            ),
            resultLimit = 10,
            refineSpec = FindMusicRefineSpec(
                primaryIngredientIndex = 0,
                neighborhood = FindMusicRefineNeighborhood.TOP_0_5_PERCENT,
            ),
        )
        val session = evidence(query, 500).copy(objectiveRankingDomainCount = 3)
        val track = FindMusicTrackEvidence(
            displayedRank = 1,
            objectiveRank = 1,
            resultScore = 0.9f,
            rankingScore = 0.9f,
            ingredientPercentiles = listOf(0.98f, 0.9f),
        )

        assertEquals(
            "Refine \u00b7 keep close to ambient \u00b7 rank by guitar \u00b7 nearest 0.5%\n" +
                "3 recordings in the primary neighborhood \u00b7 ingredient ranks across " +
                "500 recordings in this library scope",
            SessionEvidenceText.findMusicQuery(session),
        )
        assertEquals("Refine", SessionEvidenceText.findMusicMode(session))
        assertEquals(
            "Primary match #11 \u00b7 Secondary match #51",
            SessionEvidenceText.findMusicTrack(session, track),
        )
    }

    @Test
    fun `seed summary gives median range and exact candidate identity context`() {
        val session = radioSession(
            row(seedRank = 2),
            row(seedRank = 1_676),
            row(seedRank = 32_935),
        )

        assertEquals(
            "Typical distance from seed \u00b7 around #1,676 nearest\n" +
                "Range \u00b7 #2 to #32,935 nearest \u00b7 farthest in the closest 41.2% of 80,000",
            SessionEvidenceText.seedReach(session),
        )
    }

    @Test
    fun `session summary and drawer lead with mode signal before counts`() {
        val session = radioSession(row(seedRank = 2), row(seedRank = 500))
        val mode = "MMR query relevance 80% \u00b7 overlap penalty 20%"

        assertEquals(
            "$mode \u00b7 2 tracks",
            SessionEvidenceText.sessionHeaderSummary(session, mode),
        )
        assertEquals(
            "$mode \u00b7 2 tracks \u00b7 22:10",
            SessionEvidenceText.sessionDrawerSubtitle(session, mode, "22:10"),
        )
    }

    @Test
    fun `drawer exposes partial cancelled and incomplete outcomes`() {
        val base = radioSession(row(seedRank = 2), row(seedRank = 500))
        val partial = base.copy(
            delivery = QueueDeliverySummary(
                origin = QueueOrigin.APP_RADIO,
                requestedCount = 3,
                rankedCount = 2,
                resolvedCount = 1,
                verifiedCount = 1,
                notInLibraryCount = 1,
                queueFailedCount = 1,
                verificationComplete = false,
                unexpectedObservedCount = 1,
            ),
            outcome = RadioSessionOutcome.PARTIAL_FAILED,
        )
        assertEquals(
            "MMR \u00b7 1 of 3 queued \u00b7 partial \u00b7 final queue check incomplete \u00b7 2 selected \u00b7 " +
                "1 found in Poweramp \u00b7 1 extra Poweramp entries \u00b7 " +
                "Today \u00b7 22:10",
            SessionEvidenceText.sessionDrawerSubtitle(partial, "MMR", "Today \u00b7 22:10"),
        )

        val cancelled = base.copy(
            delivery = QueueDeliverySummary(
                origin = QueueOrigin.APP_RADIO,
                requestedCount = 2,
                rankedCount = 2,
                resolvedCount = 2,
                verifiedCount = 2,
                notInLibraryCount = 0,
                queueFailedCount = 0,
                verificationComplete = true,
            ),
            outcome = RadioSessionOutcome.CANCELLED,
        )
        assertEquals(
            "DPP \u00b7 2 tracks queued \u00b7 cancelled \u00b7 Yesterday \u00b7 09:15",
            SessionEvidenceText.sessionDrawerSubtitle(
                cancelled,
                "DPP",
                "Yesterday \u00b7 09:15",
            ),
        )

        val incomplete = base.copy(isComplete = false, totalExpected = 30)
        assertEquals(
            "Closest \u00b7 2 of 30 selected \u00b7 incomplete \u00b7 Today \u00b7 22:10",
            SessionEvidenceText.sessionDrawerSubtitle(
                incomplete,
                "Closest",
                "Today \u00b7 22:10",
            ),
        )

        val cancelledBeforeDelivery = base.copy(outcome = RadioSessionOutcome.CANCELLED)
        assertEquals(
            "MMR \u00b7 2 tracks \u00b7 cancelled \u00b7 Today \u00b7 22:10",
            SessionEvidenceText.sessionDrawerSubtitle(
                cancelledBeforeDelivery,
                "MMR",
                "Today \u00b7 22:10",
            ),
        )
    }

    @Test
    fun `drawer preserves a nondefault queue origin`() {
        val widget = radioSession(row(seedRank = 2)).copy(
            delivery = QueueDeliverySummary(
                origin = QueueOrigin.WIDGET_RADIO,
                requestedCount = 1,
                rankedCount = 1,
                resolvedCount = 1,
                verifiedCount = 1,
                notInLibraryCount = 0,
                queueFailedCount = 0,
                verificationComplete = true,
            ),
        )

        assertEquals(
            "MMR \u00b7 1 track queued \u00b7 Started from widget \u00b7 Today \u00b7 22:10",
            SessionEvidenceText.sessionDrawerSubtitle(widget, "MMR", "Today \u00b7 22:10"),
        )
    }

    @Test
    fun `replay drawer keeps placement and time without repeating its origin`() {
        val replay = radioSession(row(seedRank = 2), row(seedRank = 500)).copy(
            delivery = QueueDeliverySummary(
                origin = QueueOrigin.HISTORY_REQUEUE,
                requestedCount = 2,
                rankedCount = 2,
                resolvedCount = 2,
                verifiedCount = 2,
                notInLibraryCount = 0,
                queueFailedCount = 0,
                verificationComplete = true,
            ),
            directQueuePlacement = DirectQueuePlacement.APPEND,
        )

        assertEquals(
            "Replayed queue \u00b7 2 tracks queued \u00b7 Yesterday \u00b7 22:01",
            SessionEvidenceText.sessionDrawerSubtitle(
                replay,
                "Replayed queue",
                "Yesterday \u00b7 22:01",
            ),
        )
    }

    @Test
    fun `history timestamps expose their day without losing exact local time`() {
        val utc = TimeZone.getTimeZone("UTC")
        val now = timestamp(2026, Calendar.JULY, 16, 20, 15, utc)

        assertEquals(
            "Today \u00b7 08:05",
            SessionEvidenceText.historyTimestamp(
                timestamp(2026, Calendar.JULY, 16, 8, 5, utc),
                now,
                Locale.US,
                utc,
            ),
        )
        assertEquals(
            "Yesterday \u00b7 23:59",
            SessionEvidenceText.historyTimestamp(
                timestamp(2026, Calendar.JULY, 15, 23, 59, utc),
                now,
                Locale.US,
                utc,
            ),
        )
        assertEquals(
            "14 Jul \u00b7 08:05",
            SessionEvidenceText.historyTimestamp(
                timestamp(2026, Calendar.JULY, 14, 8, 5, utc),
                now,
                Locale.US,
                utc,
            ),
        )
        assertEquals(
            "14 Jul 2025 \u00b7 08:05",
            SessionEvidenceText.historyTimestamp(
                timestamp(2025, Calendar.JULY, 14, 8, 5, utc),
                now,
                Locale.US,
                utc,
            ),
        )
    }

    @Test
    fun `MMR prior-pick evidence is shown only when it says something`() {
        assertNull(SessionEvidenceText.mmrPriorPick(null))
        assertNull(SessionEvidenceText.mmrPriorPick(" "))
        assertEquals(
            "Most similar earlier pick: \"Oasis\"",
            SessionEvidenceText.mmrPriorPick("Oasis"),
        )
    }

    @Test
    fun `even seed summary reports one interpretable midpoint`() {
        val session = radioSession(
            row(seedRank = 2),
            row(seedRank = 500),
            row(seedRank = 1_676),
            row(seedRank = 32_935),
        )

        assertEquals(
            "Typical distance from seed \u00b7 around #1,088 nearest\n" +
                "Range \u00b7 #2 to #32,935 nearest \u00b7 farthest in the closest 41.2% of 80,000",
            SessionEvidenceText.seedReach(session),
        )
    }

    @Test
    fun `equal even middle ranks are reported once`() {
        val session = radioSession(
            row(seedRank = 2),
            row(seedRank = 500),
            row(seedRank = 500),
            row(seedRank = 32_935),
        )

        assertEquals(
            "Typical distance from seed \u00b7 around #500 nearest\n" +
                "Range \u00b7 #2 to #32,935 nearest \u00b7 farthest in the closest 41.2% of 80,000",
            SessionEvidenceText.seedReach(session),
        )
    }

    @Test
    fun `partial seed summary names its queued-only scope`() {
        val session = radioSession(
            row(seedRank = 2),
            row(seedRank = 500, status = QueueStatus.QUEUE_FAILED),
        )

        assertEquals(
            "Queued distance from seed \u00b7 #2 of 80,000 \u00b7 top <0.01%",
            SessionEvidenceText.seedReach(session),
        )
    }

    @Test
    fun `seed summary fails closed without one exact ranking domain`() {
        val session = radioSession(row(seedRank = 2)).copy(
            eligibleCandidateIdentityCount = null,
            seedRankingIdentityCount = null,
        )

        assertNull(SessionEvidenceText.seedReach(session))
    }

    @Test
    fun `finite added-date summary reports rank against the full active library`() {
        val session = radioSession(row(seedRank = 900)).copy(
            config = RadioConfig(libraryAddedDays = 17),
            eligibleCandidateIdentityCount = 100,
            seedRankingIdentityCount = 1_000,
        )

        assertEquals(
            "Distance from seed \u00b7 #900 of 1,000 \u00b7 top 90%",
            SessionEvidenceText.seedReach(session),
        )
    }

    @Test
    fun `seed summary fails closed when any queued row lacks exact rank evidence`() {
        val session = radioSession(
            row(seedRank = 2),
            row(seedRank = 500).copy(seedRank = null),
        )

        assertNull(SessionEvidenceText.seedReach(session))
    }

    @Test
    fun `text planner copy and reach describe distinct proven promises`() {
        assertEquals(
            "Varied (DPP)",
            SessionEvidenceText.textPlannerLabel(FindMusicTextResultPlanner.VARIED_DPP),
        )
        assertEquals(
            "Cosine-ranks every candidate recording against the text, strongest first.",
            SessionEvidenceText.textPlannerDescription(FindMusicTextResultPlanner.CLOSEST),
        )
        assertEquals(
            "Runs greedy DPP over the complete selected candidate domain, using text match as " +
                "quality while rewarding variety across the set.",
            SessionEvidenceText.textPlannerDescription(FindMusicTextResultPlanner.VARIED_DPP),
        )
        assertEquals(
            "Match range \u00b7 #1 to #296 of 80,323 eligible recordings \u00b7 farthest in top 0.4%",
            SessionEvidenceText.textMatchReach(
                objectiveRanks = listOf(23, 1, 296, 18),
                objectiveDomainCount = 80_323,
            ),
        )
        assertEquals(
            "Queued match range \u00b7 #23 of 80,323 eligible recordings \u00b7 top 0.03%",
            SessionEvidenceText.textMatchReach(
                objectiveRanks = listOf(23),
                objectiveDomainCount = 80_323,
                queuedOnly = true,
            ),
        )
        assertNull(
            SessionEvidenceText.textMatchReach(
                objectiveRanks = listOf(23),
                objectiveDomainCount = null,
            ),
        )
    }

    @Test
    fun `graph evidence reports a typical path instead of raw probability`() {
        assertEquals(
            "Graph Explorer \u00b7 typical path about 2 track-to-track moves \u00b7 " +
                "15% stop chance after each move",
            SessionEvidenceText.graphExploration(2.37, 0.15f),
        )
        assertEquals(
            "Graph Explorer \u00b7 typical path about 1 track-to-track move",
            SessionEvidenceText.graphExploration(1.2),
        )
        assertNull(SessionEvidenceText.graphExploration(Double.NaN))
    }

    private fun evidence(
        query: FindMusicQuerySpec,
        activeTrackCount: Int,
    ) = FindMusicSessionEvidence(
        querySpec = query,
        orderedActiveTrackIdsSha256 = "a".repeat(64),
        activeTrackCount = activeTrackCount,
        objectiveRankingDomainCount = activeTrackCount,
        ingredientRankingDomainCount = activeTrackCount.takeUnless {
            query.isSimplePositiveTextOnly
        },
        stableResultReduction = StableResultReductionEvidence(
            identityPolicyVersion = 1,
            requestedVisibleCount = query.resultLimit,
            scannedRowCount = query.resultLimit,
            collapsedEquivalentCount = 0,
        ),
    )

    private fun radioSession(vararg rows: QueuedTrackResult) = RadioResult(
        seedTrack = PowerampTrack(
            realId = 1L,
            title = "Seed",
            artist = "Artist",
            album = "Album",
            durationMs = 180_000,
            path = "/music/seed.flac",
        ),
        matchType = TrackMatcher.MatchType.ACTIVE_CATALOG_EXACT,
        tracks = rows.toList(),
        eligibleCandidateIdentityCount = 80_000,
        seedRankingIdentityCount = 80_000,
    )

    private fun row(
        seedRank: Int,
        status: QueueStatus = QueueStatus.QUEUED,
    ) = QueuedTrackResult(
        track = EmbeddedTrack(
            id = seedRank.toLong() + 1L,
            metadataKey = "metadata-$seedRank",
            filenameKey = "file-$seedRank",
            artist = "Artist",
            album = "Album",
            title = "Result $seedRank",
            durationMs = 170_000,
            filePath = "/music/result-$seedRank.flac",
        ),
        similarity = 0.8f,
        similarityToSeed = 0.8f,
        seedRank = seedRank,
        status = status,
    )

    private fun timestamp(
        year: Int,
        month: Int,
        day: Int,
        hour: Int,
        minute: Int,
        timeZone: TimeZone,
    ): Long = Calendar.getInstance(timeZone, Locale.US).run {
        clear()
        set(year, month, day, hour, minute, 0)
        timeInMillis
    }
}
