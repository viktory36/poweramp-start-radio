package com.powerampstartradio.indexing

import com.powerampstartradio.indexing.v2.V2CommittedProviderSpan
import com.powerampstartradio.indexing.v2.V2ProviderDurationEvidencePolicy
import com.powerampstartradio.indexing.v2.V2ProviderSpanReceipt
import java.util.Collections

internal data class V2ActiveLibraryGenerationBinding(
    val databaseGenerationId: String,
    val providerGenerationId: String,
)

internal enum class V2ActiveLibraryBindingEvidence {
    EXACT_V2_RECEIPT_SPAN,
    LEGACY_EXACT_ABSOLUTE_PATH,
    LEGACY_EXACT_MUSIC_RELATIVE_PATH,
}

internal data class V2ActiveLibraryBinding(
    val trackId: Long,
    val powerampFileId: Long,
    val evidence: V2ActiveLibraryBindingEvidence,
    /** Poweramp folder_files.created_at for the bound provider row. */
    val createdAtEpochSecond: Long = 1L,
)

internal enum class V2ActiveLibraryQuarantineReason {
    UNRESOLVED_EXACT_RECEIPT,
    SPAN_SPECIFIC_REBUILD_REQUIRED,
    PATH_TIMING_CONFLICT,
    NO_CURRENT_PROVIDER_BINDING,
}

internal data class V2ActiveLibraryQuarantinedTrack(
    val trackId: Long,
    val reason: V2ActiveLibraryQuarantineReason,
)

/** Serializable-shaped evidence for a real-library dry run without changing recommendation state. */
internal data class V2ActiveLibraryDryRunReport(
    val generationBinding: V2ActiveLibraryGenerationBinding,
    val databaseTrackCount: Int,
    val providerTrackCount: Int,
    val activeTrackCount: Int,
    val quarantinedTrackCount: Int,
    val unboundProviderCount: Int,
    val bindingEvidenceCounts: Map<V2ActiveLibraryBindingEvidence, Int>,
    val bindings: List<V2ActiveLibraryBinding>,
    val activeTrackIds: List<Long>,
    val quarantinedTracks: List<V2ActiveLibraryQuarantinedTrack>,
    val unboundPowerampFileIds: List<Long>,
) {
    /** Stable, complete artifact suitable for an adb pull and byte-for-byte repeat comparison. */
    fun renderDeterministicTsv(): String = buildString {
        append("v2-active-library-dry-run-v1\n")
        append("database_generation\t")
        append(generationBinding.databaseGenerationId.toTsvField())
        append('\n')
        append("provider_generation\t")
        append(generationBinding.providerGenerationId.toTsvField())
        append('\n')
        append("counts\tdatabase\t")
        append(databaseTrackCount)
        append("\tprovider\t")
        append(providerTrackCount)
        append("\tactive\t")
        append(activeTrackCount)
        append("\tquarantined\t")
        append(quarantinedTrackCount)
        append("\tunbound_provider\t")
        append(unboundProviderCount)
        append('\n')
        bindings.forEach { binding ->
            append("ACTIVE\t")
            append(binding.trackId)
            append('\t')
            append(binding.powerampFileId)
            append('\t')
            append(binding.evidence.name)
            append('\n')
        }
        quarantinedTracks.forEach { quarantined ->
            append("QUARANTINED\t")
            append(quarantined.trackId)
            append('\t')
            append(quarantined.reason.name)
            append('\n')
        }
        unboundPowerampFileIds.forEach { powerampFileId ->
            append("UNBOUND_POWERAMP\t")
            append(powerampFileId)
            append('\n')
        }
    }
}

/** Ordered dry-run projection of an actual recommendation result. */
internal data class V2ActiveLibraryCandidateProjection<T>(
    val activeCandidates: List<T>,
    val quarantinedCandidates: List<T>,
    val unknownCandidates: List<T>,
)

/**
 * Immutable, generation-bound view of embedding rows that have one current Poweramp occurrence.
 *
 * Quarantined rows remain in the source database. The catalog changes only the candidate domain;
 * it does not delete embeddings or infer recommendation signal from provider metadata.
 */
internal class V2ActiveLibraryCatalog internal constructor(
    val generationBinding: V2ActiveLibraryGenerationBinding,
    bindings: Collection<V2ActiveLibraryBinding>,
    quarantinedTracks: Collection<V2ActiveLibraryQuarantinedTrack>,
    unboundPowerampFileIds: Collection<Long>,
) {
    val bindings: List<V2ActiveLibraryBinding> = immutableList(
        bindings.sortedWith(
            compareBy(V2ActiveLibraryBinding::trackId)
                .thenBy(V2ActiveLibraryBinding::powerampFileId),
        ),
    )
    val quarantinedTracks: List<V2ActiveLibraryQuarantinedTrack> = immutableList(
        quarantinedTracks.sortedBy(V2ActiveLibraryQuarantinedTrack::trackId),
    )
    val unboundPowerampFileIds: Set<Long> = immutableSet(unboundPowerampFileIds.sorted())

    private val bindingByTrackId: Map<Long, V2ActiveLibraryBinding> = immutableMap(
        this.bindings.associateBy(V2ActiveLibraryBinding::trackId),
    )
    private val powerampFileIdByTrackId: Map<Long, Long> = immutableMap(
        bindingByTrackId.mapValues { it.value.powerampFileId },
    )
    private val trackIdByPowerampFileId: Map<Long, Long> = immutableMap(
        this.bindings.associate { it.powerampFileId to it.trackId },
    )
    private val createdAtEpochSecondByTrackId: Map<Long, Long> = immutableMap(
        this.bindings.associate { it.trackId to it.createdAtEpochSecond },
    )
    val activeTrackIds: Set<Long> = immutableSet(powerampFileIdByTrackId.keys.sorted())
    val quarantinedTrackIds: Set<Long> = immutableSet(
        this.quarantinedTracks.map(V2ActiveLibraryQuarantinedTrack::trackId),
    )

    init {
        require(generationBinding.databaseGenerationId.isNotBlank()) {
            "database generation is blank"
        }
        require(generationBinding.providerGenerationId.isNotBlank()) {
            "provider generation is blank"
        }
        require(this.bindings.all { it.trackId > 0L && it.powerampFileId > 0L }) {
            "active-library binding contains a non-positive ID"
        }
        require(this.bindings.all { it.createdAtEpochSecond >= 0L }) {
            "active-library binding contains an invalid Poweramp first-seen time"
        }
        require(powerampFileIdByTrackId.size == this.bindings.size) {
            "active-library catalog reuses a database track ID"
        }
        require(trackIdByPowerampFileId.size == this.bindings.size) {
            "active-library catalog reuses a Poweramp file ID"
        }
        require(activeTrackIds.intersect(quarantinedTrackIds).isEmpty()) {
            "active and quarantined track domains overlap"
        }
        require(this.quarantinedTracks.map { it.trackId }.distinct().size ==
            this.quarantinedTracks.size
        ) { "active-library catalog repeats a quarantined track ID" }
        require(this.quarantinedTracks.all { it.trackId > 0L }) {
            "active-library catalog contains a non-positive quarantined track ID"
        }
        require(this.unboundPowerampFileIds.all { it > 0L }) {
            "active-library catalog contains a non-positive unbound Poweramp ID"
        }
    }

    fun containsActiveTrack(trackId: Long): Boolean = trackId in activeTrackIds

    fun bindingForTrack(trackId: Long): V2ActiveLibraryBinding? = bindingByTrackId[trackId]

    fun powerampFileIdForTrack(trackId: Long): Long? = powerampFileIdByTrackId[trackId]

    fun trackIdForPowerampFile(powerampFileId: Long): Long? =
        trackIdByPowerampFileId[powerampFileId]

    fun createdAtEpochSecondForTrack(trackId: Long): Long? =
        createdAtEpochSecondByTrackId[trackId]

    /** Preserve the exact supplied ranking order while exposing every rejected occurrence. */
    fun <T> projectCandidates(
        candidates: Iterable<T>,
        trackIdOf: (T) -> Long,
    ): V2ActiveLibraryCandidateProjection<T> {
        val active = mutableListOf<T>()
        val quarantined = mutableListOf<T>()
        val unknown = mutableListOf<T>()
        candidates.forEach { candidate ->
            when (trackIdOf(candidate)) {
                in activeTrackIds -> active += candidate
                in quarantinedTrackIds -> quarantined += candidate
                else -> unknown += candidate
            }
        }
        return V2ActiveLibraryCandidateProjection(
            activeCandidates = active.toList(),
            quarantinedCandidates = quarantined.toList(),
            unknownCandidates = unknown.toList(),
        )
    }

    fun dryRunReport(): V2ActiveLibraryDryRunReport {
        val evidenceCounts = V2ActiveLibraryBindingEvidence.entries.associateWith { evidence ->
            bindings.count { it.evidence == evidence }
        }
        return V2ActiveLibraryDryRunReport(
            generationBinding = generationBinding,
            databaseTrackCount = activeTrackIds.size + quarantinedTrackIds.size,
            providerTrackCount = activeTrackIds.size + unboundPowerampFileIds.size,
            activeTrackCount = activeTrackIds.size,
            quarantinedTrackCount = quarantinedTrackIds.size,
            unboundProviderCount = unboundPowerampFileIds.size,
            bindingEvidenceCounts = immutableMap(evidenceCounts),
            bindings = bindings,
            activeTrackIds = activeTrackIds.sorted(),
            quarantinedTracks = quarantinedTracks,
            unboundPowerampFileIds = unboundPowerampFileIds.sorted(),
        )
    }
}

/** Pure reconciliation; callers own acquisition of the exact database and provider generations. */
internal object V2ActiveLibraryCatalogBuilder {
    fun build(
        databaseGenerationId: String,
        providerGenerationId: String,
        provider: Collection<V2LegacyProviderCandidate>,
        database: Collection<V2LegacyDatabaseCandidate>,
        receipts: Collection<V2ProviderSpanReceipt>,
    ): V2ActiveLibraryCatalog {
        require(databaseGenerationId.isNotBlank()) { "database generation is blank" }
        require(providerGenerationId.isNotBlank()) { "provider generation is blank" }
        require(provider.all { it.powerampFileId > 0L }) { "invalid provider candidate ID" }
        require(database.all { it.trackId > 0L }) { "invalid database candidate ID" }
        require(provider.map { it.powerampFileId }.distinct().size == provider.size) {
            "duplicate provider candidate ID"
        }
        require(database.map { it.trackId }.distinct().size == database.size) {
            "duplicate database candidate ID"
        }
        require(receipts.all { receipt ->
            receipt.trackId > 0L &&
                receipt.providerSpan.normalizedPhysicalPath.isNotBlank() &&
                receipt.providerSpan.offsetMs >= 0L &&
                receipt.providerSpan.durationMs >= 0L
        }) { "invalid exact V2 provider-span receipt" }

        val providerBySpan = provider.groupBy { it.asProviderSpan() }
        val providerById = provider.associateBy(V2LegacyProviderCandidate::powerampFileId)
        val receiptsBySpan = receipts.groupBy(V2ProviderSpanReceipt::providerSpan)
        val receiptsByTrackId = receipts.groupBy(V2ProviderSpanReceipt::trackId)
        val databaseIds = database.mapTo(hashSetOf()) { it.trackId }

        // Receipt claims are reserved even when ambiguous. Exact evidence must never be silently
        // downgraded to migration compatibility because one side of the receipt became unclear.
        val receiptOwnedTrackIds = receipts.asSequence()
            .map(V2ProviderSpanReceipt::trackId)
            .filter(databaseIds::contains)
            .toSet()
        val receiptClaimedProviderIds = receiptsBySpan.keys.asSequence()
            .flatMap { span -> providerBySpan[span].orEmpty().asSequence() }
            .mapTo(hashSetOf()) { it.powerampFileId }

        val exactReceiptBindings = receiptsBySpan.entries.mapNotNull { (span, spanReceipts) ->
            val receipt = spanReceipts.singleOrNull() ?: return@mapNotNull null
            val providerCandidate = providerBySpan[span].orEmpty().singleOrNull()
                ?: return@mapNotNull null
            if (receipt.trackId !in databaseIds ||
                receiptsByTrackId[receipt.trackId].orEmpty().size != 1
            ) {
                return@mapNotNull null
            }
            V2ActiveLibraryBinding(
                trackId = receipt.trackId,
                powerampFileId = providerCandidate.powerampFileId,
                evidence = V2ActiveLibraryBindingEvidence.EXACT_V2_RECEIPT_SPAN,
                createdAtEpochSecond = providerCandidate.createdAtEpochSecond,
            )
        }

        val remainingProvider = provider.filter {
            it.powerampFileId !in receiptClaimedProviderIds
        }
        val remainingDatabase = database.filter { it.trackId !in receiptOwnedTrackIds }
        val compatibility = V2LegacyCompatibilityResolver.resolve(
            provider = remainingProvider,
            database = remainingDatabase,
        )
        val compatibilityBindings = compatibility.bindings.map { binding ->
            V2ActiveLibraryBinding(
                trackId = binding.trackId,
                powerampFileId = binding.powerampFileId,
                evidence = binding.evidence.asActiveLibraryEvidence(),
                createdAtEpochSecond = checkNotNull(providerById[binding.powerampFileId])
                    .createdAtEpochSecond,
            )
        }
        val bindings = exactReceiptBindings + compatibilityBindings
        val activeTrackIds = bindings.mapTo(hashSetOf()) { it.trackId }

        val quarantineReasons = hashMapOf<Long, V2ActiveLibraryQuarantineReason>()
        receiptOwnedTrackIds.forEach { trackId ->
            if (trackId !in activeTrackIds) {
                quarantineReasons[trackId] =
                    V2ActiveLibraryQuarantineReason.UNRESOLVED_EXACT_RECEIPT
            }
        }
        compatibility.repairBindings.forEach { binding ->
            quarantineReasons.putIfAbsent(
                binding.trackId,
                V2ActiveLibraryQuarantineReason.SPAN_SPECIFIC_REBUILD_REQUIRED,
            )
        }
        compatibility.pathTimingConflictBindings.forEach { binding ->
            quarantineReasons.putIfAbsent(
                binding.trackId,
                V2ActiveLibraryQuarantineReason.PATH_TIMING_CONFLICT,
            )
        }
        database.forEach { candidate ->
            if (candidate.trackId !in activeTrackIds) {
                quarantineReasons.putIfAbsent(
                    candidate.trackId,
                    V2ActiveLibraryQuarantineReason.NO_CURRENT_PROVIDER_BINDING,
                )
            }
        }

        val activeProviderIds = bindings.mapTo(hashSetOf()) { it.powerampFileId }
        return V2ActiveLibraryCatalog(
            generationBinding = V2ActiveLibraryGenerationBinding(
                databaseGenerationId = databaseGenerationId,
                providerGenerationId = providerGenerationId,
            ),
            bindings = bindings,
            quarantinedTracks = quarantineReasons.map { (trackId, reason) ->
                V2ActiveLibraryQuarantinedTrack(trackId, reason)
            },
            unboundPowerampFileIds = provider.asSequence()
                .map(V2LegacyProviderCandidate::powerampFileId)
                .filterNot(activeProviderIds::contains)
                .toSet(),
        )
    }

    private fun V2LegacyProviderCandidate.asProviderSpan() = V2CommittedProviderSpan(
        normalizedPhysicalPath = normalizedPhysicalPath,
        offsetMs = offsetMs,
        durationMs = V2ProviderDurationEvidencePolicy.canonicalMs(durationMs.toLong()),
    )

    private fun V2LegacyCompatibilityEvidence.asActiveLibraryEvidence():
        V2ActiveLibraryBindingEvidence = when (this) {
        V2LegacyCompatibilityEvidence.EXACT_ABSOLUTE_PATH ->
            V2ActiveLibraryBindingEvidence.LEGACY_EXACT_ABSOLUTE_PATH
        V2LegacyCompatibilityEvidence.EXACT_MUSIC_RELATIVE_PATH ->
            V2ActiveLibraryBindingEvidence.LEGACY_EXACT_MUSIC_RELATIVE_PATH
        V2LegacyCompatibilityEvidence.CUE_LOGICAL_METADATA_REPAIR,
        V2LegacyCompatibilityEvidence.EXACT_PATH_TIMING_CONFLICT ->
            error("non-coverage legacy evidence cannot activate a database row")
    }
}

private fun <T> immutableList(values: Collection<T>): List<T> =
    Collections.unmodifiableList(values.toList())

private fun <T> immutableSet(values: Collection<T>): Set<T> =
    Collections.unmodifiableSet(LinkedHashSet(values))

private fun <K, V> immutableMap(values: Map<K, V>): Map<K, V> =
    Collections.unmodifiableMap(LinkedHashMap(values))

private fun String.toTsvField(): String = replace("\\", "\\\\")
    .replace("\t", "\\t")
    .replace("\r", "\\r")
    .replace("\n", "\\n")
