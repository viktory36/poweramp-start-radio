package com.powerampstartradio.services

import com.powerampstartradio.data.StableIdentityGenerationBinding
import com.powerampstartradio.ui.RadioGenerationToken
import com.powerampstartradio.ui.RadioSeedIdentity
import org.junit.Assert.assertThrows
import org.junit.Test

class DisplayedResultBindingGuardTest {
    @Test
    fun `activation race rejects a reused numeric row ID before it can resolve`() {
        val displayed = binding("generation-a", "activation-a", 'a')
        val nowActive = generation("generation-b", "activation-b", 'b')

        assertThrows(IllegalArgumentException::class.java) {
            DisplayedResultBindingGuard.requireExactGeneration(displayed, nowActive)
        }
    }

    @Test
    fun `same generation requires the exact stable row identity`() {
        val generation = generation("generation-a", "activation-a", 'a')
        DisplayedResultBindingGuard.requireExactGeneration(
            binding("generation-a", "activation-a", 'a'),
            generation,
        )

        assertThrows(IllegalArgumentException::class.java) {
            DisplayedResultBindingGuard.requireExactTrackIdentity(
                displayedTrackId = 42,
                displayedIdentity = RadioSeedIdentity(42, stableId('1')),
                activeStableTrackSpanId = stableId('2'),
            )
        }
    }

    private fun binding(
        generationId: String,
        activationBindingId: String,
        hash: Char,
    ) = StableIdentityGenerationBinding(
        bindingSpecId = "v2-active-index-generation-binding-v1",
        generationId = generationId,
        activationBindingId = activationBindingId,
        databaseContentSha256 = hash.toString().repeat(64),
        orderedTrackSetSha256 = hash.toString().repeat(64),
    )

    private fun generation(
        generationId: String,
        activationBindingId: String,
        hash: Char,
    ) = RadioGenerationToken(
        generationId = generationId,
        activationBindingId = activationBindingId,
        manifestSha256 = "f".repeat(64),
        embeddingSpecId = "clamp3-audio-v1",
        databaseContentSha256 = hash.toString().repeat(64),
        orderedTrackSetSha256 = hash.toString().repeat(64),
        stableTrackUidMappingSha256 = "e".repeat(64),
    )

    private fun stableId(hash: Char) = "stable-track-span-v1-${hash.toString().repeat(64)}"
}
