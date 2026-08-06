package com.powerampstartradio.ui

import com.powerampstartradio.data.EmbeddedTrack
import com.powerampstartradio.indexing.V2ActiveLibraryBinding
import com.powerampstartradio.indexing.V2ActiveLibraryBindingEvidence
import com.powerampstartradio.indexing.V2ActiveLibraryCatalog
import com.powerampstartradio.indexing.V2ActiveLibraryGenerationBinding
import com.powerampstartradio.indexing.V2ActiveLibraryQuarantineReason
import com.powerampstartradio.indexing.V2ActiveLibraryQuarantinedTrack
import com.powerampstartradio.poweramp.PowerampTrack
import com.powerampstartradio.poweramp.TrackMatcher
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assert.assertThrows
import org.junit.Test

class SessionReplayEligibilityTest {
    @Test
    fun `legacy or generation-changed sessions are disabled`() {
        val legacy = session(generation = null)
        assertFalse(evaluate(legacy, generation()).eligible)

        val changed = session(generation = generation("generation-a"))
        assertFalse(evaluate(changed, generation("generation-b")).eligible)
    }

    @Test
    fun `missing occurrence or changed recording identity is disabled`() {
        val session = session(generation = generation())
        assertFalse(
            SessionReplayEligibilityPolicy.evaluate(
                session,
                generation(),
                resolvedCurrentPowerampFileIds = null,
                resolvedByExactEmbeddedRows = listOf(true),
                currentTrackIdentities = currentIdentities(),
            ).eligible,
        )
        assertFalse(
            SessionReplayEligibilityPolicy.evaluate(
                session,
                generation(),
                resolvedCurrentPowerampFileIds = listOf(701L),
                resolvedByExactEmbeddedRows = listOf(true),
                currentTrackIdentities = mapOf(
                    7L to CurrentReplayTrackIdentity(embeddedTrack(), stableId('b')),
                ),
            ).eligible,
        )
    }

    @Test
    fun `exact active generation occurrence and identity are replayable`() {
        assertTrue(evaluate(session(generation = generation()), generation()).eligible)
    }

    @Test
    fun `legacy queued row without physical delivery occurrence is not replayable`() {
        assertFalse(
            evaluate(
                session(generation = generation(), queueId = null),
                generation(),
            ).eligible,
        )
    }

    @Test
    fun `full content row resolves across generations without numeric track ID aliasing`() {
        val stable = stableId('a')
        val eligibility = SessionReplayEligibilityPolicy.evaluate(
            session = session(generation = generation("generation-a"), trackStableId = stable),
            activeGeneration = generation("generation-b"),
            // The saved ID was 700; the same recording now resolves uniquely to 701.
            resolvedCurrentPowerampFileIds = listOf(701L),
            resolvedByExactEmbeddedRows = listOf(false),
            currentTrackIdentities = mapOf(
                // A reused numeric ID is irrelevant across generations.
                7L to CurrentReplayTrackIdentity(embeddedTrack(), stableId('f')),
            ),
            availableFullContentStableTrackSpanIds = setOf(stable),
        )

        assertTrue(eligibility.eligible)
    }

    @Test
    fun `unchanged provider snapshot replays an exact legacy row across index generations`() {
        val eligibility = SessionReplayEligibilityPolicy.evaluate(
            session = session(
                generation = generation("generation-a"),
                trackStableId = null,
                providerGenerationId = "provider-a",
            ),
            activeGeneration = generation("generation-b"),
            resolvedCurrentPowerampFileIds = listOf(700L),
            resolvedByExactEmbeddedRows = listOf(true),
            currentTrackIdentities = mapOf(
                7L to CurrentReplayTrackIdentity(embeddedTrack(), null),
            ),
            activeProviderGenerationId = "provider-a",
        )

        assertTrue(eligibility.eligible)
    }

    @Test
    fun `unchanged provider snapshot still rejects changed row or Poweramp occurrence`() {
        val saved = session(
            generation = generation("generation-a"),
            trackStableId = null,
            providerGenerationId = "provider-a",
        )
        val changedRow = SessionReplayEligibilityPolicy.evaluate(
            session = saved,
            activeGeneration = generation("generation-b"),
            resolvedCurrentPowerampFileIds = listOf(700L),
            resolvedByExactEmbeddedRows = listOf(true),
            currentTrackIdentities = mapOf(
                7L to CurrentReplayTrackIdentity(embeddedTrack().copy(title = "Changed"), null),
            ),
            activeProviderGenerationId = "provider-a",
        )
        val changedOccurrence = SessionReplayEligibilityPolicy.evaluate(
            session = saved,
            activeGeneration = generation("generation-b"),
            resolvedCurrentPowerampFileIds = listOf(701L),
            resolvedByExactEmbeddedRows = listOf(true),
            currentTrackIdentities = mapOf(
                7L to CurrentReplayTrackIdentity(embeddedTrack(), null),
            ),
            activeProviderGenerationId = "provider-a",
        )

        assertFalse(changedRow.eligible)
        assertFalse(changedOccurrence.eligible)
    }

    @Test
    fun `cross generation replay rejects legacy or unresolved stable row`() {
        val active = generation("generation-b")
        val withoutStableIdentity = SessionReplayEligibilityPolicy.evaluate(
            session(generation("generation-a"), trackStableId = null),
            active,
            resolvedCurrentPowerampFileIds = listOf(701L),
            resolvedByExactEmbeddedRows = listOf(false),
            currentTrackIdentities = emptyMap(),
            availableFullContentStableTrackSpanIds = emptySet(),
        )
        assertFalse(withoutStableIdentity.eligible)
        assertEquals(
            "This older session cannot be queued again reliably. " +
                "Start a new radio from the same seed.",
            withoutStableIdentity.reason,
        )

        val absentStableSpan = SessionReplayEligibilityPolicy.evaluate(
            session(generation("generation-a"), trackStableId = stableId('a')),
            active,
            resolvedCurrentPowerampFileIds = listOf(701L),
            resolvedByExactEmbeddedRows = listOf(false),
            currentTrackIdentities = emptyMap(),
            availableFullContentStableTrackSpanIds = emptySet(),
        )
        assertFalse(absentStableSpan.eligible)
        assertEquals(
            "The library changed and some tracks in this session are no longer available. " +
                "Start a new radio from the same seed.",
            absentStableSpan.reason,
        )
    }

    @Test
    fun `unchanged provider snapshot falls back per stable row without weakening legacy rows`() {
        val stable = stableId('d')
        val legacy = session(
            generation = generation("generation-a"),
            trackStableId = null,
            providerGenerationId = "provider-a",
        ).tracks.single()
        val stableRow = legacy.copy(
            track = embeddedTrack().copy(id = 8L, title = "Renumbered stable recording"),
            resolvedPowerampFileId = 800L,
            resolvedPowerampQueueId = 1_800L,
            stableTrackSpanId = stable,
        )
        val mixed = session(
            generation = generation("generation-a"),
            trackStableId = null,
            providerGenerationId = "provider-a",
        ).copy(tracks = listOf(legacy, stableRow))

        val eligibility = SessionReplayEligibilityPolicy.evaluate(
            session = mixed,
            activeGeneration = generation("generation-b"),
            resolvedCurrentPowerampFileIds = listOf(700L, 800L),
            resolvedByExactEmbeddedRows = listOf(true, false),
            currentTrackIdentities = mapOf(
                7L to CurrentReplayTrackIdentity(embeddedTrack(), null),
            ),
            activeProviderGenerationId = "provider-a",
            availableFullContentStableTrackSpanIds = setOf(stable),
        )

        assertTrue(eligibility.eligible)
    }

    @Test
    fun `verified replay binding retains only exact active catalog mappings`() {
        val catalog = activeCatalog(
            generationId = "generation-a",
            bindings = listOf(11L to 901L, 7L to 701L),
            quarantinedTrackId = 9L,
        )

        val binding = VerifiedReplayLibraryBinding.from(generation(), catalog)

        assertEquals(2, binding.activeTrackCount)
        assertEquals(701L, binding.powerampFileIdForTrack(7L))
        assertEquals(901L, binding.powerampFileIdForTrack(11L))
        assertNull(binding.powerampFileIdForTrack(9L))
        assertNull(binding.powerampFileIdForTrack(12L))
    }

    @Test
    fun `verified replay binding rejects another database generation`() {
        val catalog = activeCatalog(
            generationId = "generation-b",
            bindings = listOf(7L to 701L),
        )

        assertThrows(IllegalArgumentException::class.java) {
            VerifiedReplayLibraryBinding.from(generation("generation-a"), catalog)
        }
    }

    @Test
    fun `stable fallback skips inactive representative and prefers saved active occurrence`() {
        val currentPowerampIds = mapOf(8L to 800L, 9L to 900L)

        assertEquals(
            9L,
            StableReplayTrackSelectionPolicy.select(
                equivalentTrackIds = listOf(7L, 8L, 9L),
                savedPowerampFileId = 900L,
                preferSavedPowerampOccurrence = true,
                currentPowerampFileId = currentPowerampIds::get,
            ),
        )
        assertEquals(
            8L,
            StableReplayTrackSelectionPolicy.select(
                equivalentTrackIds = listOf(7L, 8L, 9L),
                savedPowerampFileId = 900L,
                preferSavedPowerampOccurrence = false,
                currentPowerampFileId = currentPowerampIds::get,
            ),
        )
    }

    private fun evaluate(session: RadioResult, active: RadioGenerationToken) =
        SessionReplayEligibilityPolicy.evaluate(
            session,
            active,
            resolvedCurrentPowerampFileIds = listOf(701L),
            resolvedByExactEmbeddedRows = listOf(true),
            currentTrackIdentities = currentIdentities(),
        )

    private fun currentIdentities() = mapOf(
        7L to CurrentReplayTrackIdentity(embeddedTrack(), stableId('a')),
    )

    private fun session(
        generation: RadioGenerationToken?,
        queueId: Long? = 1_700L,
        trackStableId: String? = stableId('a'),
        providerGenerationId: String? = null,
    ) = RadioResult(
        seedTrack = PowerampTrack(1, "Seed", null, null, 1, null),
        matchType = TrackMatcher.MatchType.METADATA_EXACT,
        tracks = listOf(
            QueuedTrackResult(
                track = embeddedTrack(),
                similarity = 0.8f,
                similarityToSeed = 0.8f,
                status = QueueStatus.QUEUED,
                resolvedPowerampFileId = 700L,
                resolvedPowerampQueueId = queueId,
                stableTrackSpanId = trackStableId,
            ),
        ),
        generation = generation,
        providerGenerationId = providerGenerationId,
    )

    private fun embeddedTrack() =
        EmbeddedTrack(7, "m", "f", null, null, "Track", 1, "/track")

    private fun generation(id: String = "generation-a") = RadioGenerationToken(
        generationId = id,
        activationBindingId = "activation-$id",
        manifestSha256 = "a".repeat(64),
        embeddingSpecId = "clamp3-audio-v1",
        databaseContentSha256 = "b".repeat(64),
        orderedTrackSetSha256 = "c".repeat(64),
        stableTrackUidMappingSha256 = "d".repeat(64),
    )

    private fun stableId(hash: Char) = "stable-track-span-v1-${hash.toString().repeat(64)}"

    private fun activeCatalog(
        generationId: String,
        bindings: List<Pair<Long, Long>>,
        quarantinedTrackId: Long? = null,
    ) = V2ActiveLibraryCatalog(
        generationBinding = V2ActiveLibraryGenerationBinding(
            databaseGenerationId = generationId,
            providerGenerationId = "provider-generation",
        ),
        bindings = bindings.map { (trackId, powerampFileId) ->
            V2ActiveLibraryBinding(
                trackId = trackId,
                powerampFileId = powerampFileId,
                evidence = V2ActiveLibraryBindingEvidence.LEGACY_EXACT_ABSOLUTE_PATH,
            )
        },
        quarantinedTracks = quarantinedTrackId?.let { trackId ->
            listOf(
                V2ActiveLibraryQuarantinedTrack(
                    trackId = trackId,
                    reason = V2ActiveLibraryQuarantineReason.NO_CURRENT_PROVIDER_BINDING,
                ),
            )
        }.orEmpty(),
        unboundPowerampFileIds = emptyList(),
    )
}
