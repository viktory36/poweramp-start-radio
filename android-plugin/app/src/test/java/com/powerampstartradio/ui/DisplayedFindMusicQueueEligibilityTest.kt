package com.powerampstartradio.ui

import com.powerampstartradio.data.EmbeddedTrack
import com.powerampstartradio.data.StableIdentityGenerationBinding
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class DisplayedFindMusicQueueEligibilityTest {
    @Test
    fun `exact generation domain identity and occurrence are queueable`() {
        assertTrue(evaluate(result()).eligible)
    }

    @Test
    fun `generation or active provider domain change disables queueing`() {
        val generationChanged = evaluate(
            result(),
            activeGeneration = generation(generationId = "generation-b"),
        )
        val providerChanged = evaluate(
            result(),
            activeProviderGenerationId = "provider-b",
        )
        val membershipChanged = evaluate(
            result(),
            orderedActiveTrackIdsSha256 = hash('c'),
        )

        assertFalse(generationChanged.eligible)
        assertFalse(providerChanged.eligible)
        assertFalse(membershipChanged.eligible)
        assertTrue(generationChanged.reason?.contains("library changed") == true)
    }

    @Test
    fun `missing row changed stable identity or occurrence disables queueing`() {
        val missing = evaluate(
            result(),
            currentTracks = mapOf(7L to CurrentDisplayedFindMusicTrack(false, stableId(), 70L)),
        )
        val changedIdentity = evaluate(
            result(),
            currentTracks = mapOf(7L to CurrentDisplayedFindMusicTrack(true, "different", 70L)),
        )
        val noOccurrence = evaluate(
            result(),
            currentTracks = mapOf(7L to CurrentDisplayedFindMusicTrack(true, stableId(), null)),
        )

        assertFalse(missing.eligible)
        assertFalse(changedIdentity.eligible)
        assertFalse(noOccurrence.eligible)
    }

    @Test
    fun `incomplete displayed evidence fails closed before current library checks`() {
        val incomplete = result().copy(stableResultReduction = null)

        val eligibility = evaluate(incomplete)

        assertFalse(eligibility.eligible)
        assertTrue(eligibility.reason?.contains("Run Find Music again") == true)
        assertFalse(eligibility.reason?.contains("evidence") == true)
    }

    private fun evaluate(
        result: TextSearchResult,
        activeGeneration: RadioGenerationToken? = generation(),
        activeProviderGenerationId: String? = "provider-a",
        orderedActiveTrackIdsSha256: String? = hash('b'),
        activeTrackCount: Int? = 1,
        currentTracks: Map<Long, CurrentDisplayedFindMusicTrack> = mapOf(
            7L to CurrentDisplayedFindMusicTrack(true, stableId(), 70L),
        ),
    ) = DisplayedFindMusicQueueEligibilityPolicy.evaluate(
        result = result,
        activeGeneration = activeGeneration,
        activeProviderGenerationId = activeProviderGenerationId,
        orderedActiveTrackIdsSha256 = orderedActiveTrackIdsSha256,
        activeTrackCount = activeTrackCount,
        currentTracks = currentTracks,
    )

    private fun result(): TextSearchResult {
        val binding = binding()
        val track = EmbeddedTrack(
            id = 7L,
            metadataKey = "metadata",
            filenameKey = "file",
            artist = "Artist",
            album = "Album",
            title = "Title",
            durationMs = 200_000,
            filePath = "/music/title.flac",
        )
        return TextSearchResult(
            query = "sleep",
            matches = listOf(
                TextSearchMatch(
                    track = track,
                    similarity = 0.8f,
                    identity = RadioSeedIdentity(track.id, stableId()),
                    objectiveRank = 1,
                ),
            ),
            querySpec = FindMusicQuerySpec(
                textIngredients = listOf(FindMusicTextIngredient("sleep", 1f, false)),
                libraryBinding = binding,
            ),
            libraryBinding = binding,
            providerGenerationId = "provider-a",
            orderedActiveTrackIdsSha256 = hash('b'),
            activeTrackCount = 1,
            stableResultReduction = StableResultReductionEvidence(
                identityPolicyVersion = 1,
                requestedVisibleCount = 1,
                scannedRowCount = 1,
                collapsedEquivalentCount = 0,
            ),
        )
    }

    private fun binding() = StableIdentityGenerationBinding(
        bindingSpecId = "binding-v1",
        generationId = "generation-a",
        activationBindingId = "activation-a",
        databaseContentSha256 = hash('d'),
        orderedTrackSetSha256 = hash('e'),
    )

    private fun generation(generationId: String = "generation-a") = RadioGenerationToken(
        generationId = generationId,
        activationBindingId = "activation-a",
        manifestSha256 = hash('a'),
        embeddingSpecId = "embedding-v1",
        databaseContentSha256 = hash('d'),
        orderedTrackSetSha256 = hash('e'),
        stableTrackUidMappingSha256 = hash('f'),
    )

    private fun stableId() = "full-content-sha256:${hash('1')}"

    private fun hash(char: Char) = char.toString().repeat(64)
}
