package com.powerampstartradio.similarity

import com.powerampstartradio.data.EmbeddingIndex
import com.powerampstartradio.data.StableIdentityGenerationBinding
import com.powerampstartradio.data.StableTrackIdentityCatalog
import com.powerampstartradio.data.StableTrackIdentityRow
import com.powerampstartradio.indexing.V2ActiveLibraryBinding
import com.powerampstartradio.indexing.V2ActiveLibraryBindingEvidence
import com.powerampstartradio.indexing.V2ActiveLibraryCatalog
import com.powerampstartradio.indexing.V2ActiveLibraryGenerationBinding
import com.powerampstartradio.indexing.V2ActiveLibraryQuarantineReason
import com.powerampstartradio.indexing.V2ActiveLibraryQuarantinedTrack
import com.powerampstartradio.indexing.v2.StableTrackSpanIdentityStrength
import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class ActiveRecommendationDomainTest {
    @get:Rule
    val temporaryFolder = TemporaryFolder()

    @Test
    fun `domain is an immutable aligned active partition with exact delivery bindings`() {
        val index = index(longArrayOf(10, 20, 30, 40))
        val identities = identities(
            stableRow(10, 1),
            stableRow(20, 1),
            stableRow(30, 2),
            stableRow(40, 2),
        )
        val catalog = activeCatalog(
            activeTrackIds = longArrayOf(40, 10, 30),
            quarantinedTrackIds = longArrayOf(20),
        )

        val domain = ActiveRecommendationDomain.create(catalog, identities, index)

        assertEquals(3, domain.activeTrackCount)
        assertArrayEquals(longArrayOf(10, 30, 40), domain.orderedActiveTrackIds())
        assertArrayEquals(
            longArrayOf(10, 30),
            domain.orderedActiveIdentityRepresentativeTrackIds(),
        )
        assertArrayEquals(intArrayOf(0, 2, 3), domain.orderedActiveIndices())
        assertArrayEquals(longArrayOf(20), domain.orderedInactiveTrackIds())
        assertEquals(setOf(20L), domain.inactiveTrackIds)
        assertEquals(setOf(10L, 30L, 40L), domain.activeTrackIds)
        assertArrayEquals(
            floatArrayOf(0.1f, 0.3f, 0.4f),
            domain.activeScoresFromFull(floatArrayOf(0.1f, 0.2f, 0.3f, 0.4f)),
            0f,
        )
        assertTrue(domain.containsActiveTrack(30))
        assertFalse(domain.containsActiveTrack(20))
        assertEquals(1_030L, domain.powerampFileIdForTrack(30))
        assertEquals(30L, domain.trackIdForPowerampFile(1_030))
        assertNull(domain.powerampFileIdForTrack(20))
        assertEquals(1, domain.activeVisibleDuplicateExcessCount)
        assertEquals(2, domain.activeCandidateIdentityCount)
        assertEquals(10L, domain.activeIdentityRepresentativeTrackId(10))
        assertEquals(30L, domain.activeIdentityRepresentativeTrackId(30))
        assertEquals(30L, domain.activeIdentityRepresentativeTrackId(40))
        assertThrows(IllegalArgumentException::class.java) {
            domain.activeIdentityRepresentativeTrackId(20)
        }
        assertEquals(1, domain.eligibleCandidateIdentityCount(seedTrackId = 10))
        assertEquals(1, domain.eligibleCandidateIdentityCount(seedTrackId = 30))
        assertEquals(2, domain.rankedRowsForVisibleCount(1))
        assertEquals(3, domain.rankedRowsForVisibleCount(3))
        assertEquals(0, domain.rankedRowsForVisibleCount(0))

        assertEquals(ActiveRecommendationDomain.BINDING_SPEC_ID, domain.binding.bindingSpecId)
        assertEquals(DATABASE_GENERATION, domain.binding.databaseGenerationId)
        assertEquals(PROVIDER_GENERATION, domain.binding.providerGenerationId)
        assertEquals(4, domain.binding.databaseTrackCount)
        assertEquals(3, domain.binding.activeTrackCount)
        assertEquals(2, domain.binding.activeCandidateIdentityCount)
        assertEquals(domain.orderedActiveTrackIdsSha256, domain.binding.orderedActiveTrackIdsSha256)
        assertEquals(
            domain.orderedActiveIdentityRepresentativeTrackIdsSha256,
            domain.binding.orderedActiveIdentityRepresentativeTrackIdsSha256,
        )
        assertEquals(
            "7a8592d1da5e005e5171136583ddd0c278ef119d3bd09b9c1aa28b654bee1181",
            domain.orderedActiveTrackIdsSha256,
        )

        val leakedIds = domain.orderedActiveTrackIds()
        val leakedIdentityRepresentatives = domain.orderedActiveIdentityRepresentativeTrackIds()
        val leakedIndices = domain.orderedActiveIndices()
        val leakedInactive = domain.orderedInactiveTrackIds()
        leakedIds[0] = 999
        leakedIdentityRepresentatives[0] = 999
        leakedIndices[0] = 999
        leakedInactive[0] = 999
        assertArrayEquals(longArrayOf(10, 30, 40), domain.orderedActiveTrackIds())
        assertArrayEquals(
            longArrayOf(10, 30),
            domain.orderedActiveIdentityRepresentativeTrackIds(),
        )
        assertArrayEquals(intArrayOf(0, 2, 3), domain.orderedActiveIndices())
        assertArrayEquals(longArrayOf(20), domain.orderedInactiveTrackIds())
        assertThrows(UnsupportedOperationException::class.java) {
            @Suppress("UNCHECKED_CAST")
            (domain.inactiveTrackIds as MutableSet<Long>).add(999)
        }
    }

    @Test
    fun `only full-content proven copies share an active representative`() {
        val index = index(longArrayOf(10, 20, 30, 40, 50, 60))
        val domain = ActiveRecommendationDomain.create(
            activeCatalog(
                activeTrackIds = longArrayOf(10, 20, 30, 40, 50, 60),
                quarantinedTrackIds = longArrayOf(),
            ),
            identities(
                stableRow(10, 1),
                stableRow(20, 1),
                sampledRow(30, 2),
                sampledRow(40, 2),
                legacyRow(50),
                legacyRow(60),
            ),
            index,
        )

        assertArrayEquals(
            longArrayOf(10, 30, 40, 50, 60),
            domain.orderedActiveIdentityRepresentativeTrackIds(),
        )
        assertEquals(10L, domain.activeIdentityRepresentativeTrackId(20))
        assertEquals(30L, domain.activeIdentityRepresentativeTrackId(30))
        assertEquals(40L, domain.activeIdentityRepresentativeTrackId(40))
        assertEquals(50L, domain.activeIdentityRepresentativeTrackId(50))
        assertEquals(60L, domain.activeIdentityRepresentativeTrackId(60))
        assertEquals(5, domain.activeCandidateIdentityCount)
    }

    @Test
    fun `duplicate excess counts only active visible identities`() {
        val index = index(longArrayOf(10, 20, 30))
        val identities = identities(
            stableRow(10, 1),
            stableRow(20, 1),
            stableRow(30, 2),
        )
        assertEquals(1, identities.duplicateExcessCount)

        val domain = ActiveRecommendationDomain.create(
            activeCatalog(
                activeTrackIds = longArrayOf(10, 30),
                quarantinedTrackIds = longArrayOf(20),
            ),
            identities,
            index,
        )

        assertEquals(0, domain.activeVisibleDuplicateExcessCount)
        assertEquals(2, domain.activeCandidateIdentityCount)
        assertEquals(1, domain.eligibleCandidateIdentityCount(seedTrackId = 10))
        assertEquals(1, domain.rankedRowsForVisibleCount(1))
    }

    @Test
    fun `added-date domain filters occurrences before duplicate identity collapse`() {
        val index = index(longArrayOf(10, 20, 30, 40))
        val domain = ActiveRecommendationDomain.create(
            activeCatalog(
                activeTrackIds = longArrayOf(10, 20, 30, 40),
                quarantinedTrackIds = longArrayOf(),
                createdAtByTrackId = mapOf(
                    10L to 100L,
                    20L to 1_100L,
                    30L to 1_000L,
                    40L to 0L,
                ),
            ),
            identities(
                stableRow(10, 1),
                stableRow(20, 1),
                stableRow(30, 2),
                stableRow(40, 3),
            ),
            index,
        )

        val recent = domain.candidateDomain(minimumCreatedAtEpochSecond = 1_000L)
        val allDates = domain.candidateDomain(minimumCreatedAtEpochSecond = null)

        assertArrayEquals(longArrayOf(10, 20, 30, 40), allDates.orderedEligibleTrackIds())
        assertEquals(1_000L, recent.minimumCreatedAtEpochSecond)
        assertArrayEquals(longArrayOf(20, 30), recent.orderedEligibleTrackIds())
        assertArrayEquals(
            longArrayOf(20, 30),
            recent.orderedIdentityRepresentativeTrackIds(),
        )
        assertEquals(20L, recent.representativeForVisibleIdentity(10))
        assertEquals(2, recent.eligibleCandidateIdentityCount(seedTrackId = 40))
        assertEquals(1, recent.eligibleCandidateIdentityCount(seedTrackId = 10))
        assertArrayEquals(
            floatArrayOf(0.2f, 0.3f),
            recent.identityRepresentativeScoresFromFull(
                floatArrayOf(0.1f, 0.2f, 0.3f, 0.4f),
            ),
            0f,
        )
        assertThrows(IllegalArgumentException::class.java) {
            recent.identityRepresentativeScoresFromFull(floatArrayOf(0.1f))
        }
        assertEquals(
            1,
            recent.rankEligibleIdentityFromFullSimilarities(
                similarities = floatArrayOf(0.1f, 0.8f, 0.7f, 1f),
                targetTrackId = 20,
                seedTrackId = 40,
            ),
        )
    }

    @Test
    fun `full active seed rank remains global when selection uses an added-date subset`() {
        val index = index(longArrayOf(10, 20, 30, 40))
        val domain = ActiveRecommendationDomain.create(
            activeCatalog(
                activeTrackIds = longArrayOf(10, 20, 30, 40),
                quarantinedTrackIds = longArrayOf(),
                createdAtByTrackId = mapOf(
                    10L to 100L,
                    20L to 100L,
                    30L to 100L,
                    40L to 1_100L,
                ),
            ),
            identities(
                legacyRow(10),
                legacyRow(20),
                legacyRow(30),
                legacyRow(40),
            ),
            index,
        )
        val recent = domain.candidateDomain(minimumCreatedAtEpochSecond = 1_000L)
        val seedSimilarities = floatArrayOf(1f, 0.95f, 0.9f, 0.8f)

        assertEquals(1, recent.eligibleCandidateIdentityCount(seedTrackId = 10))
        assertEquals(3, domain.eligibleCandidateIdentityCount(seedTrackId = 10))
        assertEquals(
            1,
            recent.rankEligibleIdentityFromFullSimilarities(
                seedSimilarities,
                targetTrackId = 40,
                seedTrackId = 10,
            ),
        )
        assertEquals(
            3,
            domain.rankEligibleIdentityFromFullSimilarities(
                seedSimilarities,
                targetTrackId = 40,
                seedTrackId = 10,
            ),
        )
    }

    @Test
    fun `seed distance ranks fixed identity representatives and excludes the seed identity`() {
        val index = index(longArrayOf(10, 20, 30, 40, 50))
        val identities = identities(
            stableRow(10, 3),
            stableRow(20, 0),
            stableRow(30, 2),
            stableRow(40, 1),
            stableRow(50, 2),
        )
        val domain = ActiveRecommendationDomain.create(
            activeCatalog(
                activeTrackIds = longArrayOf(10, 30, 40, 50),
                quarantinedTrackIds = longArrayOf(20),
            ),
            identities,
            index,
        )
        // Track 50 is a verified copy of representative 30. Its deliberately larger row score
        // must not change either the ranking domain or the selected identity's rank.
        val scores = floatArrayOf(0.8f, 1.0f, 0.7f, 0.7f, 0.99f)

        assertEquals(
            1,
            domain.rankEligibleIdentityFromFullSimilarities(scores, 40, seedTrackId = 10),
        )
        assertEquals(
            2,
            domain.rankEligibleIdentityFromFullSimilarities(scores, 30, seedTrackId = 10),
        )
        assertEquals(
            2,
            domain.rankEligibleIdentityFromFullSimilarities(scores, 50, seedTrackId = 10),
        )
        assertEquals(
            1,
            domain.rankEligibleIdentityFromFullSimilarities(scores, 10, seedTrackId = 40),
        )
        assertEquals(
            2,
            domain.rankEligibleIdentityFromFullSimilarities(scores, 30, seedTrackId = 40),
        )

        val scoresWithNaN = floatArrayOf(0.8f, 1.0f, Float.NaN, 0.7f, 0.99f)
        assertEquals(
            1,
            domain.rankEligibleIdentityFromFullSimilarities(
                scoresWithNaN,
                40,
                seedTrackId = 10,
            ),
        )
        assertEquals(
            2,
            domain.rankEligibleIdentityFromFullSimilarities(
                scoresWithNaN,
                50,
                seedTrackId = 10,
            ),
        )
        assertThrows(IllegalArgumentException::class.java) {
            domain.rankEligibleIdentityFromFullSimilarities(
                floatArrayOf(1f),
                40,
                seedTrackId = 10,
            )
        }
        assertThrows(IllegalArgumentException::class.java) {
            domain.rankEligibleIdentityFromFullSimilarities(scores, 20, seedTrackId = 10)
        }
        assertThrows(IllegalArgumentException::class.java) {
            domain.rankEligibleIdentityFromFullSimilarities(scores, 40, seedTrackId = 20)
        }
        assertThrows(IllegalArgumentException::class.java) {
            domain.rankEligibleIdentityFromFullSimilarities(scores, 50, seedTrackId = 30)
        }
        assertThrows(IllegalArgumentException::class.java) {
            domain.activeScoresFromFull(floatArrayOf(1f))
        }
    }

    @Test
    fun `ordered active hash binds membership and order deterministically`() {
        val first = ActiveRecommendationDomain.create(
            activeCatalog(
                activeTrackIds = longArrayOf(30, 10),
                quarantinedTrackIds = longArrayOf(20),
            ),
            identities(legacyRow(10), legacyRow(20), legacyRow(30)),
            index(longArrayOf(10, 20, 30)),
        )
        val repeated = ActiveRecommendationDomain.create(
            activeCatalog(
                activeTrackIds = longArrayOf(10, 30),
                quarantinedTrackIds = longArrayOf(20),
            ),
            identities(legacyRow(10), legacyRow(20), legacyRow(30)),
            index(longArrayOf(10, 20, 30)),
        )
        val changed = ActiveRecommendationDomain.create(
            activeCatalog(
                activeTrackIds = longArrayOf(10, 20),
                quarantinedTrackIds = longArrayOf(30),
            ),
            identities(legacyRow(10), legacyRow(20), legacyRow(30)),
            index(longArrayOf(10, 20, 30)),
        )

        assertEquals(first.binding, repeated.binding)
        assertEquals(first.orderedActiveTrackIdsSha256, repeated.orderedActiveTrackIdsSha256)
        assertEquals(
            first.orderedActiveIdentityRepresentativeTrackIdsSha256,
            repeated.orderedActiveIdentityRepresentativeTrackIdsSha256,
        )
        assertFalse(first.orderedActiveTrackIdsSha256 == changed.orderedActiveTrackIdsSha256)
    }

    @Test
    fun `domain binding changes when the proven active identity partition changes`() {
        val index = index(longArrayOf(10, 20, 30))
        val catalog = activeCatalog(
            activeTrackIds = longArrayOf(10, 20, 30),
            quarantinedTrackIds = longArrayOf(),
        )
        val grouped = ActiveRecommendationDomain.create(
            catalog,
            identities(stableRow(10, 1), stableRow(20, 1), legacyRow(30)),
            index,
        )
        val separate = ActiveRecommendationDomain.create(
            catalog,
            identities(stableRow(10, 1), stableRow(20, 2), legacyRow(30)),
            index,
        )

        assertArrayEquals(
            longArrayOf(10, 30),
            grouped.orderedActiveIdentityRepresentativeTrackIds(),
        )
        assertArrayEquals(
            longArrayOf(10, 20, 30),
            separate.orderedActiveIdentityRepresentativeTrackIds(),
        )
        assertFalse(grouped.binding == separate.binding)
    }

    @Test
    fun `factory rejects incomplete extra or misaligned partitions`() {
        val index = index(longArrayOf(10, 20))
        val alignedIdentities = identities(legacyRow(10), legacyRow(20))

        assertThrows(IllegalArgumentException::class.java) {
            ActiveRecommendationDomain.create(
                activeCatalog(activeTrackIds = longArrayOf(10), quarantinedTrackIds = longArrayOf()),
                alignedIdentities,
                index,
            )
        }
        assertThrows(IllegalArgumentException::class.java) {
            ActiveRecommendationDomain.create(
                activeCatalog(activeTrackIds = longArrayOf(10), quarantinedTrackIds = longArrayOf(99)),
                alignedIdentities,
                index,
            )
        }
        assertThrows(IllegalArgumentException::class.java) {
            ActiveRecommendationDomain.create(
                activeCatalog(activeTrackIds = longArrayOf(10), quarantinedTrackIds = longArrayOf(20)),
                identities(legacyRow(10), legacyRow(30)),
                index,
            )
        }
    }

    @Test
    fun `factory rejects a provider catalog from another database generation`() {
        val index = index(longArrayOf(10))
        val wrongCatalog = activeCatalog(
            activeTrackIds = longArrayOf(10),
            quarantinedTrackIds = longArrayOf(),
            databaseGeneration = "another-generation",
        )

        assertThrows(IllegalArgumentException::class.java) {
            ActiveRecommendationDomain.create(
                wrongCatalog,
                identities(legacyRow(10)),
                index,
            )
        }
    }

    private fun activeCatalog(
        activeTrackIds: LongArray,
        quarantinedTrackIds: LongArray,
        databaseGeneration: String = DATABASE_GENERATION,
        createdAtByTrackId: Map<Long, Long> = emptyMap(),
    ) = V2ActiveLibraryCatalog(
        generationBinding = V2ActiveLibraryGenerationBinding(
            databaseGenerationId = databaseGeneration,
            providerGenerationId = PROVIDER_GENERATION,
        ),
        bindings = activeTrackIds.map { trackId ->
            V2ActiveLibraryBinding(
                trackId = trackId,
                powerampFileId = 1_000 + trackId,
                evidence = V2ActiveLibraryBindingEvidence.EXACT_V2_RECEIPT_SPAN,
                createdAtEpochSecond = createdAtByTrackId[trackId] ?: 1L,
            )
        },
        quarantinedTracks = quarantinedTrackIds.map { trackId ->
            V2ActiveLibraryQuarantinedTrack(
                trackId,
                V2ActiveLibraryQuarantineReason.NO_CURRENT_PROVIDER_BINDING,
            )
        },
        unboundPowerampFileIds = emptyList(),
    )

    private fun identities(vararg rows: StableTrackIdentityRow): StableTrackIdentityCatalog =
        StableTrackIdentityCatalog.fromOrderedRows(
            binding = StableIdentityGenerationBinding(
                bindingSpecId = "v2-active-index-generation-binding-v1",
                generationId = DATABASE_GENERATION,
                activationBindingId = "activation-binding",
                databaseContentSha256 = "a".repeat(64),
                orderedTrackSetSha256 = "b".repeat(64),
            ),
            orderedEmbeddingTrackIds = rows.map(StableTrackIdentityRow::trackId).toLongArray(),
            rows = rows.toList(),
        )

    private fun stableRow(trackId: Long, identityNumber: Int) = StableTrackIdentityRow(
        trackId = trackId,
        stableTrackSpanId = stableId(identityNumber),
        stableIdentitySpecId = STABLE_SPEC,
        stableIdentityStrength = StableTrackSpanIdentityStrength.FULL_CONTENT_SHA256,
        embeddingSpecId = "embedding-spec-test",
        embeddingSha256 = identityNumber.toString(16).padStart(64, '0'),
    )

    private fun sampledRow(trackId: Long, identityNumber: Int) =
        stableRow(trackId, identityNumber).copy(
            stableIdentityStrength =
                StableTrackSpanIdentityStrength.VERSIONED_SAMPLED_CONTENT_SHA256,
        )

    private fun legacyRow(trackId: Long) = StableTrackIdentityRow(
        trackId = trackId,
        stableTrackSpanId = null,
        stableIdentitySpecId = null,
        stableIdentityStrength = null,
        embeddingSpecId = null,
        embeddingSha256 = null,
    )

    private fun stableId(number: Int): String =
        "stable-track-span-v1-${number.toString(16).padStart(64, '0')}"

    private fun index(trackIds: LongArray): EmbeddingIndex {
        val dimension = 2
        val bytes = ByteBuffer.allocate(
            16 + trackIds.size * Long.SIZE_BYTES + trackIds.size * dimension * Float.SIZE_BYTES,
        ).order(ByteOrder.LITTLE_ENDIAN)
        bytes.putInt(0x424D4550)
        bytes.putInt(1)
        bytes.putInt(trackIds.size)
        bytes.putInt(dimension)
        trackIds.forEach(bytes::putLong)
        trackIds.indices.forEach { position ->
            bytes.putFloat(if (position and 1 == 0) 1f else 0f)
            bytes.putFloat(if (position and 1 == 0) 0f else 1f)
        }
        val file = temporaryFolder.newFile("domain-${nextFileId++}.emb")
        file.writeBytes(bytes.array())
        return EmbeddingIndex.mmap(file)
    }

    private var nextFileId = 0

    companion object {
        private const val DATABASE_GENERATION = "database-generation"
        private const val PROVIDER_GENERATION = "provider-generation"
        private const val STABLE_SPEC =
            "stable-track-span-v1:content-sha256:native-half-open-sample-span"
    }
}
