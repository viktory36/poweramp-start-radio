package com.powerampstartradio.similarity

import com.powerampstartradio.data.EmbeddingIndex
import com.powerampstartradio.data.StableTrackIdentityCatalog
import com.powerampstartradio.indexing.V2ActiveLibraryCatalog
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.charset.StandardCharsets
import java.security.MessageDigest

/** Immutable identity of one exact provider-visible recommendation domain. */
internal data class ActiveRecommendationDomainBinding(
    val bindingSpecId: String,
    val databaseGenerationId: String,
    val activationBindingId: String,
    val databaseContentSha256: String,
    val orderedDatabaseTrackSetSha256: String,
    val providerGenerationId: String,
    val orderedActiveTrackIdsSha256: String,
    val orderedActiveIdentityRepresentativeTrackIdsSha256: String,
    val databaseTrackCount: Int,
    val activeTrackCount: Int,
    val activeCandidateIdentityCount: Int,
)

/**
 * Request-scoped subset of the active library after applying Poweramp first-seen eligibility.
 *
 * Occurrences are filtered before visible identities are collapsed, so a recently added copy is
 * still eligible when another copy of the same recording was first seen earlier.
 */
internal class RecommendationCandidateDomain internal constructor(
    val minimumCreatedAtEpochSecond: Long?,
    private val databaseTrackCount: Int,
    private val identityCatalog: StableTrackIdentityCatalog,
    private val orderedEligibleTrackIds: LongArray,
    private val orderedIdentityRepresentativeTrackIds: LongArray,
    private val orderedIdentityRepresentativeIndexPositions: IntArray,
    private val representativeByVisibleIdentity: Map<String, Long>,
    val eligibleVisibleDuplicateExcessCount: Int,
) {
    val eligibleTrackCount: Int get() = orderedEligibleTrackIds.size
    val candidateIdentityCount: Int get() = orderedIdentityRepresentativeTrackIds.size

    init {
        require(orderedIdentityRepresentativeTrackIds.size ==
            orderedIdentityRepresentativeIndexPositions.size
        ) { "Candidate identity rows and embedding positions differ" }
        require(candidateIdentityCount + eligibleVisibleDuplicateExcessCount ==
            eligibleTrackCount
        ) { "Candidate identity collapse does not cover the eligible occurrence domain" }
    }

    fun orderedEligibleTrackIds(): LongArray = orderedEligibleTrackIds.copyOf()

    fun orderedIdentityRepresentativeTrackIds(): LongArray =
        orderedIdentityRepresentativeTrackIds.copyOf()

    fun eligibleTrackIds(): Set<Long> = orderedEligibleTrackIds.toSet()

    fun containsEligibleTrack(trackId: Long): Boolean =
        orderedEligibleTrackIds.binarySearch(trackId) >= 0

    fun containsIdentityRepresentative(trackId: Long): Boolean =
        orderedIdentityRepresentativeTrackIds.binarySearch(trackId) >= 0

    fun representativeForVisibleIdentity(trackId: Long): Long? =
        representativeByVisibleIdentity[
            identityCatalog.visibleResultIdentity(trackId).identityToken
        ]

    /** The seed is an anchor, so it need not itself belong to this candidate subset. */
    fun eligibleCandidateIdentityCount(seedTrackId: Long): Int =
        (candidateIdentityCount -
            if (representativeForVisibleIdentity(seedTrackId) == null) 0 else 1)
            .coerceAtLeast(0)

    fun identityRepresentativeScoresFromFull(scores: FloatArray): FloatArray {
        require(scores.size == databaseTrackCount) {
            "Score rows ${scores.size} != embedding rows $databaseTrackCount"
        }
        return FloatArray(orderedIdentityRepresentativeIndexPositions.size) { position ->
            scores[orderedIdentityRepresentativeIndexPositions[position]]
        }
    }

    fun rankedRowsForVisibleCount(requestedVisibleCount: Int): Int {
        if (requestedVisibleCount <= 0) return 0
        return requestedVisibleCount.coerceAtMost(candidateIdentityCount)
    }

    /** 1-based cosine rank inside this exact candidate domain, excluding the seed identity. */
    fun rankEligibleIdentityFromFullSimilarities(
        similarities: FloatArray,
        targetTrackId: Long,
        seedTrackId: Long,
        cancellationCheck: (() -> Unit)? = null,
    ): Int {
        require(similarities.size == databaseTrackCount) {
            "Similarity rows ${similarities.size} != embedding rows $databaseTrackCount"
        }
        val targetRepresentativeId = requireNotNull(
            representativeForVisibleIdentity(targetTrackId),
        ) { "Track $targetTrackId is outside the selected added-date candidate domain" }
        val seedRepresentativeId = representativeForVisibleIdentity(seedTrackId)
        require(targetRepresentativeId != seedRepresentativeId) {
            "Cannot rank the excluded seed identity"
        }
        val targetPosition = orderedIdentityRepresentativeTrackIds.binarySearch(
            targetRepresentativeId,
        )
        require(targetPosition >= 0) {
            "Track $targetTrackId has no candidate identity representative"
        }
        val targetIndex = orderedIdentityRepresentativeIndexPositions[targetPosition]
        require(targetIndex in similarities.indices) {
            "Candidate embedding position is outside the similarity vector"
        }
        val targetScore = similarities[targetIndex]
        val targetTieKey = identityCatalog.rankingTieKey(targetRepresentativeId)
        var rank = 1

        for (position in orderedIdentityRepresentativeTrackIds.indices) {
            if (position and CANCELLATION_CHECK_MASK == 0) cancellationCheck?.invoke()
            if (position == targetPosition) continue
            val candidateId = orderedIdentityRepresentativeTrackIds[position]
            if (candidateId == seedRepresentativeId) continue
            if (ranksBefore(
                    candidateScore = similarities[
                        orderedIdentityRepresentativeIndexPositions[position]
                    ],
                    candidateTieKey = identityCatalog.rankingTieKey(candidateId),
                    candidateId = candidateId,
                    targetScore = targetScore,
                    targetTieKey = targetTieKey,
                    targetId = targetRepresentativeId,
                )
            ) {
                rank++
            }
        }
        cancellationCheck?.invoke()
        return rank
    }

    private fun ranksBefore(
        candidateScore: Float,
        candidateTieKey: String,
        candidateId: Long,
        targetScore: Float,
        targetTieKey: String,
        targetId: Long,
    ): Boolean {
        val candidateNaN = candidateScore.isNaN()
        val targetNaN = targetScore.isNaN()
        if (candidateNaN != targetNaN) return !candidateNaN
        if (!candidateNaN) {
            if (candidateScore > targetScore) return true
            if (candidateScore < targetScore) return false
        }
        val tieOrder = candidateTieKey.compareTo(targetTieKey)
        return tieOrder < 0 || tieOrder == 0 && candidateId < targetId
    }

    companion object {
        private const val CANCELLATION_CHECK_MASK = 0x3ff
    }
}

/**
 * Exact intersection of one published embedding generation and one complete Poweramp catalog.
 *
 * Construction verifies that every embedding row is either active or explicitly quarantined.
 * Arrays are retained privately; copy accessors are for JNI or batch APIs, while indexed accessors
 * avoid allocating in ordinary ranking code.
 */
internal class ActiveRecommendationDomain private constructor(
    val binding: ActiveRecommendationDomainBinding,
    private val activeCatalog: V2ActiveLibraryCatalog,
    private val identityCatalog: StableTrackIdentityCatalog,
    private val orderedActiveIds: LongArray,
    private val orderedActiveIdentityRepresentativeIds: LongArray,
    private val orderedActiveIdentityRepresentativeIndexPositions: IntArray,
    private val activeRepresentativeByVisibleIdentity: Map<String, Long>,
    private val orderedActiveIndexPositions: IntArray,
    private val orderedInactiveIds: LongArray,
    val activeVisibleDuplicateExcessCount: Int,
) {
    val activeTrackCount: Int get() = orderedActiveIds.size
    /** Exact or embedding-confirmed copies share one candidate place. */
    val activeCandidateIdentityCount: Int
        get() = orderedActiveIdentityRepresentativeIds.size

    /** Immutable exact ranking membership, already aligned to this domain binding. */
    val activeTrackIds: Set<Long> get() = activeCatalog.activeTrackIds

    /** Complete generation rows excluded from recommendation objectives. */
    val inactiveTrackIds: Set<Long> get() = activeCatalog.quarantinedTrackIds

    val orderedActiveTrackIdsSha256: String get() = binding.orderedActiveTrackIdsSha256

    val orderedActiveIdentityRepresentativeTrackIdsSha256: String
        get() = binding.orderedActiveIdentityRepresentativeTrackIdsSha256

    fun containsActiveTrack(trackId: Long): Boolean = activeCatalog.containsActiveTrack(trackId)

    /** The deterministic active occurrence used as this identity's graph node. */
    fun activeIdentityRepresentativeTrackId(trackId: Long): Long {
        require(containsActiveTrack(trackId)) {
            "Track $trackId is not in the active recommendation domain"
        }
        val identity = identityCatalog.visibleResultIdentity(trackId)
        return requireNotNull(
            activeRepresentativeByVisibleIdentity[identity.identityToken],
        ) {
            "Queue-visible identity ${identity.identityToken} has no active representative"
        }
    }

    /** Distinct candidate identities remaining after the seed identity is excluded. */
    fun eligibleCandidateIdentityCount(seedTrackId: Long): Int {
        require(containsActiveTrack(seedTrackId)) {
            "Seed track $seedTrackId is not in the active recommendation domain"
        }
        return (activeCandidateIdentityCount - 1).coerceAtLeast(0)
    }

    fun powerampFileIdForTrack(trackId: Long): Long? =
        activeCatalog.powerampFileIdForTrack(trackId)

    fun trackIdForPowerampFile(powerampFileId: Long): Long? =
        activeCatalog.trackIdForPowerampFile(powerampFileId)

    fun activeTrackIdAt(position: Int): Long {
        require(position in orderedActiveIds.indices) { "Active position out of range: $position" }
        return orderedActiveIds[position]
    }

    fun activeEmbeddingIndexAt(position: Int): Int {
        require(position in orderedActiveIndexPositions.indices) {
            "Active position out of range: $position"
        }
        return orderedActiveIndexPositions[position]
    }

    fun orderedActiveTrackIds(): LongArray = orderedActiveIds.copyOf()

    /** One active row per proven identity; every sampled or legacy row remains present. */
    fun orderedActiveIdentityRepresentativeTrackIds(): LongArray =
        orderedActiveIdentityRepresentativeIds.copyOf()

    fun orderedActiveIndices(): IntArray = orderedActiveIndexPositions.copyOf()

    fun orderedInactiveTrackIds(): LongArray = orderedInactiveIds.copyOf()

    fun candidateDomain(
        minimumCreatedAtEpochSecond: Long?,
    ): RecommendationCandidateDomain {
        if (minimumCreatedAtEpochSecond == null) {
            return RecommendationCandidateDomain(
                minimumCreatedAtEpochSecond = null,
                databaseTrackCount = binding.databaseTrackCount,
                identityCatalog = identityCatalog,
                orderedEligibleTrackIds = orderedActiveIds,
                orderedIdentityRepresentativeTrackIds =
                    orderedActiveIdentityRepresentativeIds,
                orderedIdentityRepresentativeIndexPositions =
                    orderedActiveIdentityRepresentativeIndexPositions,
                representativeByVisibleIdentity = activeRepresentativeByVisibleIdentity,
                eligibleVisibleDuplicateExcessCount = activeVisibleDuplicateExcessCount,
            )
        }
        require(minimumCreatedAtEpochSecond > 0L) {
            "Poweramp first-seen cutoff must be positive"
        }

        val eligibleIds = ArrayList<Long>()
        val representativeIds = ArrayList<Long>()
        val representativeIndices = ArrayList<Int>()
        val representativeByIdentity = LinkedHashMap<String, Long>()
        orderedActiveIds.forEachIndexed { activePosition, trackId ->
            val createdAt = requireNotNull(
                activeCatalog.createdAtEpochSecondForTrack(trackId),
            ) { "Active track $trackId has no Poweramp first-seen time" }
            if (createdAt == 0L || createdAt < minimumCreatedAtEpochSecond) {
                return@forEachIndexed
            }
            eligibleIds += trackId
            val identityToken = identityCatalog.visibleResultIdentity(trackId).identityToken
            if (representativeByIdentity.putIfAbsent(identityToken, trackId) == null) {
                representativeIds += trackId
                representativeIndices += orderedActiveIndexPositions[activePosition]
            }
        }
        return RecommendationCandidateDomain(
            minimumCreatedAtEpochSecond = minimumCreatedAtEpochSecond,
            databaseTrackCount = binding.databaseTrackCount,
            identityCatalog = identityCatalog,
            orderedEligibleTrackIds = eligibleIds.toLongArray(),
            orderedIdentityRepresentativeTrackIds = representativeIds.toLongArray(),
            orderedIdentityRepresentativeIndexPositions = representativeIndices.toIntArray(),
            representativeByVisibleIdentity = representativeByIdentity.toMap(),
            eligibleVisibleDuplicateExcessCount = eligibleIds.size - representativeIds.size,
        )
    }

    /** Project one full-generation score vector into the exact active row order. */
    fun activeScoresFromFull(scores: FloatArray): FloatArray {
        require(scores.size == binding.databaseTrackCount) {
            "Score rows ${scores.size} != embedding rows ${binding.databaseTrackCount}"
        }
        return FloatArray(orderedActiveIndexPositions.size) { position ->
            scores[orderedActiveIndexPositions[position]]
        }
    }

    /** Project one full-generation score vector onto one row per queue-visible recording. */
    fun activeIdentityRepresentativeScoresFromFull(scores: FloatArray): FloatArray {
        require(scores.size == binding.databaseTrackCount) {
            "Score rows ${scores.size} != embedding rows ${binding.databaseTrackCount}"
        }
        return FloatArray(orderedActiveIdentityRepresentativeIndexPositions.size) { position ->
            scores[orderedActiveIdentityRepresentativeIndexPositions[position]]
        }
    }

    /** Enough active ranked rows to survive every possible verified-copy collapse. */
    fun rankedRowsForVisibleCount(requestedVisibleCount: Int): Int {
        if (requestedVisibleCount <= 0) return 0
        return (requestedVisibleCount.toLong() + activeVisibleDuplicateExcessCount)
            .coerceAtMost(activeTrackCount.toLong())
            .toInt()
    }

    /**
     * Exact 1-based rank in the fixed active representative domain after excluding the seed
     * identity. The selected occurrence is ranked at its deterministic identity representative.
     *
     * Numeric scores sort descending. NaN follows every numeric score, then the stable identity
     * key and representative track ID provide deterministic tie order.
     */
    fun rankEligibleIdentityFromFullSimilarities(
        similarities: FloatArray,
        targetTrackId: Long,
        seedTrackId: Long,
        cancellationCheck: (() -> Unit)? = null,
    ): Int {
        require(similarities.size == binding.databaseTrackCount) {
            "Similarity rows ${similarities.size} != embedding rows ${binding.databaseTrackCount}"
        }

        val targetRepresentativeId = activeIdentityRepresentativeTrackId(targetTrackId)
        val seedRepresentativeId = activeIdentityRepresentativeTrackId(seedTrackId)
        require(targetRepresentativeId != seedRepresentativeId) {
            "Cannot rank the excluded seed identity"
        }
        val targetPosition = orderedActiveIdentityRepresentativeIds.binarySearch(
            targetRepresentativeId,
        )
        require(targetPosition >= 0) {
            "Track $targetTrackId has no active identity representative"
        }
        val targetIndex = orderedActiveIdentityRepresentativeIndexPositions[targetPosition]
        val targetScore = similarities[targetIndex]
        val targetTieKey = identityCatalog.rankingTieKey(targetRepresentativeId)
        var rank = 1

        for (position in orderedActiveIdentityRepresentativeIds.indices) {
            if (position and CANCELLATION_CHECK_MASK == 0) cancellationCheck?.invoke()
            if (position == targetPosition) continue
            val candidateId = orderedActiveIdentityRepresentativeIds[position]
            if (candidateId == seedRepresentativeId) continue
            if (ranksBefore(
                    candidateScore = similarities[
                        orderedActiveIdentityRepresentativeIndexPositions[position]
                    ],
                    candidateTieKey = identityCatalog.rankingTieKey(candidateId),
                    candidateId = candidateId,
                    targetScore = targetScore,
                    targetTieKey = targetTieKey,
                    targetId = targetRepresentativeId,
                )
            ) {
                rank++
            }
        }
        cancellationCheck?.invoke()
        return rank
    }

    companion object {
        const val BINDING_SPEC_ID = "active-recommendation-domain-v3"
        private const val ACTIVE_IDS_DIGEST_DOMAIN =
            "active-recommendation-domain-ordered-track-ids-v1\u0000"
        private const val ACTIVE_IDENTITY_REPRESENTATIVE_IDS_DIGEST_DOMAIN =
            "active-recommendation-domain-ordered-identity-representative-track-ids-v2\u0000"
        private const val CANCELLATION_CHECK_MASK = 0x3ff

        fun create(
            activeCatalog: V2ActiveLibraryCatalog,
            identityCatalog: StableTrackIdentityCatalog,
            embeddingIndex: EmbeddingIndex,
        ): ActiveRecommendationDomain {
            val activeBinding = activeCatalog.generationBinding
            val identityBinding = identityCatalog.binding
            require(activeBinding.databaseGenerationId == identityBinding.generationId) {
                "Active provider catalog and stable identity catalog use different database generations"
            }
            require(identityBinding.activationBindingId.isNotBlank() &&
                identityBinding.databaseContentSha256.isNotBlank() &&
                identityBinding.orderedTrackSetSha256.isNotBlank()
            ) { "Stable identity catalog has an incomplete generation binding" }
            require(identityCatalog.trackCount == embeddingIndex.numTracks) {
                "Stable identity rows ${identityCatalog.trackCount} != embedding rows " +
                    embeddingIndex.numTracks
            }
            require(
                activeCatalog.activeTrackIds.size + activeCatalog.quarantinedTrackIds.size ==
                    embeddingIndex.numTracks,
            ) { "Active and quarantined rows do not partition the embedding generation" }

            val identityIds = identityCatalog.orderedTrackIds()
            val activeIds = LongArray(activeCatalog.activeTrackIds.size)
            val activeIndices = IntArray(activeIds.size)
            val inactiveIds = LongArray(activeCatalog.quarantinedTrackIds.size)
            var activeOffset = 0
            var inactiveOffset = 0
            var previousId = Long.MIN_VALUE

            for (index in 0 until embeddingIndex.numTracks) {
                val trackId = embeddingIndex.getTrackId(index)
                require(identityIds[index] == trackId) {
                    "Stable identity row ${identityIds[index]} != embedding row $trackId at $index"
                }
                require(index == 0 || trackId > previousId) {
                    "Embedding track IDs are not strictly increasing at row $index"
                }
                previousId = trackId

                val active = activeCatalog.containsActiveTrack(trackId)
                val quarantined = trackId in activeCatalog.quarantinedTrackIds
                require(active.xor(quarantined)) {
                    "Embedding track $trackId is not in exactly one active-catalog partition"
                }
                if (active) {
                    require(activeOffset < activeIds.size) {
                        "Active provider catalog contains a row outside the embedding generation"
                    }
                    activeIds[activeOffset] = trackId
                    activeIndices[activeOffset] = index
                    activeOffset++
                } else {
                    require(inactiveOffset < inactiveIds.size) {
                        "Quarantine contains a row outside the embedding generation"
                    }
                    inactiveIds[inactiveOffset++] = trackId
                }
            }
            require(activeOffset == activeIds.size && inactiveOffset == inactiveIds.size) {
                "Active-catalog partition contains rows outside the embedding generation"
            }
            require(activeCatalog.bindings.size == activeIds.size &&
                activeCatalog.bindings.all { binding ->
                    activeCatalog.powerampFileIdForTrack(binding.trackId) == binding.powerampFileId &&
                        activeCatalog.trackIdForPowerampFile(binding.powerampFileId) == binding.trackId
                }
            ) { "Active catalog does not provide one exact Poweramp binding per active row" }

            val activeRepresentativeByVisibleIdentity = LinkedHashMap<String, Long>()
            val identityRepresentatives = ArrayList<Long>(activeIds.size)
            val identityRepresentativeIndices = ArrayList<Int>(activeIds.size)
            var duplicateExcess = 0
            activeIds.forEachIndexed { activePosition, trackId ->
                val identityToken = identityCatalog.visibleResultIdentity(trackId).identityToken
                if (activeRepresentativeByVisibleIdentity.putIfAbsent(
                        identityToken,
                        trackId,
                    ) == null
                ) {
                    identityRepresentatives += trackId
                    identityRepresentativeIndices += activeIndices[activePosition]
                } else {
                    duplicateExcess++
                }
            }
            val orderedIdentityRepresentativeIds = identityRepresentatives.toLongArray()
            val orderedIdentityRepresentativeIndices = identityRepresentativeIndices.toIntArray()
            require(orderedIdentityRepresentativeIds.size == activeIds.size - duplicateExcess) {
                "Active identity representative count is inconsistent"
            }
            require(orderedIdentityRepresentativeIndices.size == orderedIdentityRepresentativeIds.size) {
                "Active identity representative indices are inconsistent"
            }
            val activeIdsSha256 = computeOrderedActiveIdsSha256(activeIds)
            val identityRepresentativeIdsSha256 =
                computeOrderedActiveIdentityRepresentativeIdsSha256(
                    orderedIdentityRepresentativeIds,
                )
            return ActiveRecommendationDomain(
                binding = ActiveRecommendationDomainBinding(
                    bindingSpecId = BINDING_SPEC_ID,
                    databaseGenerationId = activeBinding.databaseGenerationId,
                    activationBindingId = identityBinding.activationBindingId,
                    databaseContentSha256 = identityBinding.databaseContentSha256,
                    orderedDatabaseTrackSetSha256 = identityBinding.orderedTrackSetSha256,
                    providerGenerationId = activeBinding.providerGenerationId,
                    orderedActiveTrackIdsSha256 = activeIdsSha256,
                    orderedActiveIdentityRepresentativeTrackIdsSha256 =
                        identityRepresentativeIdsSha256,
                    databaseTrackCount = embeddingIndex.numTracks,
                    activeTrackCount = activeIds.size,
                    activeCandidateIdentityCount = orderedIdentityRepresentativeIds.size,
                ),
                activeCatalog = activeCatalog,
                identityCatalog = identityCatalog,
                orderedActiveIds = activeIds,
                orderedActiveIdentityRepresentativeIds = orderedIdentityRepresentativeIds,
                orderedActiveIdentityRepresentativeIndexPositions =
                    orderedIdentityRepresentativeIndices,
                activeRepresentativeByVisibleIdentity =
                    activeRepresentativeByVisibleIdentity.toMap(),
                orderedActiveIndexPositions = activeIndices,
                orderedInactiveIds = inactiveIds,
                activeVisibleDuplicateExcessCount = duplicateExcess,
            )
        }

        private fun ranksBefore(
            candidateScore: Float,
            candidateTieKey: String,
            candidateId: Long,
            targetScore: Float,
            targetTieKey: String,
            targetId: Long,
        ): Boolean {
            val candidateNaN = candidateScore.isNaN()
            val targetNaN = targetScore.isNaN()
            if (candidateNaN != targetNaN) return !candidateNaN
            if (!candidateNaN) {
                if (candidateScore > targetScore) return true
                if (candidateScore < targetScore) return false
            }
            val tieOrder = candidateTieKey.compareTo(targetTieKey)
            return tieOrder < 0 || tieOrder == 0 && candidateId < targetId
        }

        internal fun computeOrderedActiveIdsSha256(trackIds: LongArray): String {
            return computeOrderedIdsSha256(trackIds, ACTIVE_IDS_DIGEST_DOMAIN)
        }

        internal fun computeOrderedActiveIdentityRepresentativeIdsSha256(
            trackIds: LongArray,
        ): String = computeOrderedIdsSha256(
            trackIds,
            ACTIVE_IDENTITY_REPRESENTATIVE_IDS_DIGEST_DOMAIN,
        )

        private fun computeOrderedIdsSha256(trackIds: LongArray, domain: String): String {
            require(trackIds.indices.drop(1).all { trackIds[it] > trackIds[it - 1] }) {
                "Bound track IDs must be strictly increasing"
            }
            val digest = MessageDigest.getInstance("SHA-256")
            digest.update(domain.toByteArray(StandardCharsets.UTF_8))
            val encoded = ByteBuffer.allocate(Long.SIZE_BYTES).order(ByteOrder.BIG_ENDIAN)
            trackIds.forEach { trackId ->
                encoded.clear()
                encoded.putLong(trackId)
                digest.update(encoded.array())
            }
            return digest.digest().toHex()
        }

        private fun ByteArray.toHex(): String {
            val digits = "0123456789abcdef"
            return CharArray(size * 2).also { output ->
                forEachIndexed { index, byte ->
                    val value = byte.toInt() and 0xff
                    output[index * 2] = digits[value ushr 4]
                    output[index * 2 + 1] = digits[value and 0x0f]
                }
            }.concatToString()
        }
    }
}
