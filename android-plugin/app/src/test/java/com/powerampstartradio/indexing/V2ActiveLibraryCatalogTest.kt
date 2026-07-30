package com.powerampstartradio.indexing

import com.powerampstartradio.indexing.v2.V2CommittedProviderSpan
import com.powerampstartradio.indexing.v2.V2ProviderSpanReceipt
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class V2ActiveLibraryCatalogTest {
    @Test
    fun `stale highest similarity row is quarantined before a real ranking is consumed`() {
        val catalog = build(
            provider = listOf(provider(1, "/storage/Music/active.flac", "a|a|active|100000")),
            database = listOf(
                database(10, "C:\\Music\\active.flac", "a|a|active|100000"),
                database(99, "C:\\Music\\removed.flac", "a|a|removed|100000"),
            ),
        )
        val ranking = listOf(
            ScoredCandidate(99, 0.999f),
            ScoredCandidate(10, 0.800f),
        )

        val projection = catalog.projectCandidates(ranking, ScoredCandidate::trackId)
        val report = catalog.dryRunReport()

        assertEquals(listOf(10L), projection.activeCandidates.map { it.trackId })
        assertEquals(listOf(99L), projection.quarantinedCandidates.map { it.trackId })
        assertTrue(projection.unknownCandidates.isEmpty())
        assertEquals(listOf(10L), report.activeTrackIds)
        assertEquals(listOf(99L), report.quarantinedTracks.map { it.trackId })
        assertEquals(
            V2ActiveLibraryQuarantineReason.NO_CURRENT_PROVIDER_BINDING,
            report.quarantinedTracks.single().reason,
        )
    }

    @Test
    fun `exact receipt span binds before a conflicting legacy path candidate`() {
        val source = provider(7, "/storage/Music/song.flac", "artist|album|song|100000")
        val catalog = build(
            provider = listOf(source),
            database = listOf(
                database(10, "C:\\archive\\unrelated.flac", "old|tags|song|100000"),
                database(20, "/storage/Music/song.flac", source.metadataKey),
            ),
            receipts = listOf(receipt(10, source)),
        )

        assertEquals(7L, catalog.powerampFileIdForTrack(10))
        assertEquals(10L, catalog.trackIdForPowerampFile(7))
        assertNull(catalog.powerampFileIdForTrack(20))
        assertEquals(
            V2ActiveLibraryBindingEvidence.EXACT_V2_RECEIPT_SPAN,
            catalog.bindings.single().evidence,
        )
        assertEquals(setOf(20L), catalog.quarantinedTrackIds)
    }

    @Test
    fun `zero duration receipt binds the exact unknown duration provider occurrence`() {
        val unknown = provider(
            id = 7,
            path = "/storage/Music/unknown.opus",
            metadataKey = "artist|album|unknown|0",
            durationMs = 0,
        )
        val catalog = build(
            provider = listOf(unknown),
            database = listOf(
                database(
                    id = 10,
                    path = "/storage/Music/unknown.opus",
                    metadataKey = unknown.metadataKey,
                    durationMs = 0,
                ),
            ),
            receipts = listOf(receipt(10, unknown)),
        )

        assertEquals(7L, catalog.powerampFileIdForTrack(10))
        assertEquals(
            V2ActiveLibraryBindingEvidence.EXACT_V2_RECEIPT_SPAN,
            catalog.bindings.single().evidence,
        )
    }

    @Test
    fun `duplicate receipt span and duplicate legacy candidates never choose by input order`() {
        val receiptSourceA = provider(1, "/storage/Music/receipt.flac", "a|a|r|100000")
        val receiptSourceB = provider(2, receiptSourceA.normalizedPhysicalPath, receiptSourceA.metadataKey)
        val legacySource = provider(3, "/storage/Music/legacy.flac", "a|a|l|100000")
        val catalog = build(
            provider = listOf(receiptSourceA, receiptSourceB, legacySource),
            database = listOf(
                database(10, "C:\\Music\\receipt.flac", receiptSourceA.metadataKey),
                database(20, "C:\\Music\\legacy.flac", legacySource.metadataKey),
                database(30, "D:\\Music\\legacy.flac", legacySource.metadataKey),
            ),
            receipts = listOf(receipt(10, receiptSourceA)),
        )

        assertTrue(catalog.activeTrackIds.isEmpty())
        assertEquals(setOf(10L, 20L, 30L), catalog.quarantinedTrackIds)
        assertEquals(setOf(1L, 2L, 3L), catalog.unboundPowerampFileIds)
        assertEquals(
            V2ActiveLibraryQuarantineReason.UNRESOLVED_EXACT_RECEIPT,
            catalog.quarantinedTracks.single { it.trackId == 10L }.reason,
        )
    }

    @Test
    fun `CUE rows require exact receipts while unmatched database rows stay recoverable`() {
        val unreceiptedCue = provider(
            id = 1,
            path = "/storage/Music/cue.flac",
            metadataKey = "a|album|first|100000",
            offsetMs = 60_000,
            compatibilityEligible = false,
            requiresSpanSpecificRebuild = true,
        )
        val receiptedCue = provider(
            id = 2,
            path = "/storage/Music/cue.flac",
            metadataKey = "a|album|second|100000",
            offsetMs = 160_000,
            compatibilityEligible = false,
            requiresSpanSpecificRebuild = true,
        )
        val catalog = build(
            provider = listOf(unreceiptedCue, receiptedCue, provider(3, "/storage/Music/new.flac", "n|n|n|100000")),
            database = listOf(
                database(10, "C:\\Music\\cue.flac", unreceiptedCue.metadataKey),
                database(20, "C:\\Music\\cue.flac", receiptedCue.metadataKey),
                database(99, "C:\\Music\\gone.flac", "g|g|g|100000"),
            ),
            receipts = listOf(receipt(20, receiptedCue)),
        )

        assertEquals(setOf(20L), catalog.activeTrackIds)
        assertEquals(2L, catalog.powerampFileIdForTrack(20))
        assertEquals(setOf(10L, 99L), catalog.quarantinedTrackIds)
        assertEquals(
            V2ActiveLibraryQuarantineReason.SPAN_SPECIFIC_REBUILD_REQUIRED,
            catalog.quarantinedTracks.single { it.trackId == 10L }.reason,
        )
        assertEquals(setOf(1L, 3L), catalog.unboundPowerampFileIds)
    }

    @Test
    fun `catalog and dry run report are invariant to every input order`() {
        val receiptSource = provider(3, "/storage/Music/r.flac", "r|r|r|100000")
        val providers = listOf(
            provider(2, "/storage/Music/b.flac", "b|b|b|100000"),
            receiptSource,
            provider(1, "/storage/Music/a.flac", "a|a|a|100000"),
        )
        val database = listOf(
            database(30, "C:\\Music\\r.flac", receiptSource.metadataKey),
            database(20, "C:\\Music\\b.flac", "b|b|b|100000"),
            database(10, "C:\\Music\\a.flac", "a|a|a|100000"),
            database(90, "C:\\Music\\stale.flac", "s|s|s|100000"),
        )
        val receipts = listOf(receipt(30, receiptSource))

        val forward = build(providers, database, receipts).dryRunReport()
        val reversed = build(
            providers.reversed(),
            database.reversed(),
            receipts.reversed(),
        ).dryRunReport()

        assertEquals(forward, reversed)
        assertEquals(forward.renderDeterministicTsv(), reversed.renderDeterministicTsv())
        assertTrue(forward.renderDeterministicTsv().contains("QUARANTINED\t90\t"))
        assertEquals("db-generation", forward.generationBinding.databaseGenerationId)
        assertEquals("provider-generation", forward.generationBinding.providerGenerationId)
    }

    private fun build(
        provider: List<V2LegacyProviderCandidate>,
        database: List<V2LegacyDatabaseCandidate>,
        receipts: List<V2ProviderSpanReceipt> = emptyList(),
    ) = V2ActiveLibraryCatalogBuilder.build(
        databaseGenerationId = "db-generation",
        providerGenerationId = "provider-generation",
        provider = provider,
        database = database,
        receipts = receipts,
    )

    private fun provider(
        id: Long,
        path: String,
        metadataKey: String,
        durationMs: Int = 100_000,
        offsetMs: Long = 0,
        compatibilityEligible: Boolean = true,
        requiresSpanSpecificRebuild: Boolean = false,
    ) = V2LegacyProviderCandidate(
        powerampFileId = id,
        normalizedPhysicalPath = path,
        offsetMs = offsetMs,
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

    private fun receipt(
        trackId: Long,
        provider: V2LegacyProviderCandidate,
    ) = V2ProviderSpanReceipt(
        trackId = trackId,
        providerSpan = V2CommittedProviderSpan(
            normalizedPhysicalPath = provider.normalizedPhysicalPath,
            offsetMs = provider.offsetMs,
            durationMs = provider.durationMs.toLong(),
        ),
    )

    private data class ScoredCandidate(
        val trackId: Long,
        val score: Float,
    )
}
