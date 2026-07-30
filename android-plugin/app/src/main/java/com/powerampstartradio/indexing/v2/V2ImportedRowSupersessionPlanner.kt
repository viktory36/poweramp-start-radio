package com.powerampstartradio.indexing.v2

import com.powerampstartradio.indexing.V2LegacyCompatibilityBinding
import com.powerampstartradio.indexing.V2LegacyCompatibilityEvidence

/** One selected V2 occurrence whose successful commit may replace an imported V1 row. */
internal data class V2SelectedImportedRowWork(
    val workId: String,
    val powerampFileId: Long,
    val providerSpan: V2CommittedProviderSpan,
)

internal data class V2ImportedRowSupersession(
    val selectedWork: V2SelectedImportedRowWork,
    val supersededTrackId: Long,
    val evidence: V2LegacyCompatibilityEvidence,
)

internal enum class V2ImportedRowMappingBlockReason {
    MULTIPLE_LEGACY_ROWS_FOR_SELECTED_WORK,
    LEGACY_ROW_CLAIMED_BY_MULTIPLE_SELECTED_WORKS,
}

internal data class V2BlockedImportedRowMapping(
    val selectedWork: V2SelectedImportedRowWork,
    val candidateLegacyTrackIds: List<Long>,
    val sharedLegacyTrackIds: List<Long>,
    val reasons: List<V2ImportedRowMappingBlockReason>,
)

/**
 * An exhaustive, disjoint migration plan for the supplied selected work.
 *
 * Additions have no repair binding. Supersessions have one unshared repair binding. Every other
 * bound work is blocked, so a caller can never infer destructive intent from ambiguous evidence.
 */
internal data class V2ImportedRowSupersessionPlan(
    val additions: List<V2SelectedImportedRowWork>,
    val supersessions: List<V2ImportedRowSupersession>,
    val supersededTrackIds: List<Long>,
    val blockedAmbiguousMappings: List<V2BlockedImportedRowMapping>,
)

/** Pure planning only: this object neither mutates SQLite nor grants acoustic compatibility. */
internal object V2ImportedRowSupersessionPlanner {
    fun plan(
        selectedWorks: Collection<V2SelectedImportedRowWork>,
        repairBindings: Collection<V2LegacyCompatibilityBinding>,
    ): V2ImportedRowSupersessionPlan {
        val selected = selectedWorks.toList()
        requireSelectedWorkIsValid(selected)

        val selectedByProviderId = selected.associateBy(V2SelectedImportedRowWork::powerampFileId)
        val bindings = repairBindings.toSet()
        bindings.forEach { binding ->
            require(binding.powerampFileId in selectedByProviderId) {
                "repair binding references unselected Poweramp row ${binding.powerampFileId}"
            }
            require(binding.trackId > 0L) { "legacy track ID must be positive" }
            require(binding.evidence == V2LegacyCompatibilityEvidence.CUE_LOGICAL_METADATA_REPAIR) {
                "supersession requires CUE logical repair evidence, not ${binding.evidence}"
            }
        }

        val candidatesByProviderId = bindings
            .groupBy(V2LegacyCompatibilityBinding::powerampFileId)
            .mapValues { (_, candidates) -> candidates.sortedBy(V2LegacyCompatibilityBinding::trackId) }
        val claimantsByLegacyTrackId = bindings
            .groupBy(V2LegacyCompatibilityBinding::trackId)
            .mapValues { (_, claimants) -> claimants.mapTo(sortedSetOf()) { it.powerampFileId } }

        val additions = mutableListOf<V2SelectedImportedRowWork>()
        val supersessions = mutableListOf<V2ImportedRowSupersession>()
        val blocked = mutableListOf<V2BlockedImportedRowMapping>()

        selected.sortedWith(selectedWorkComparator).forEach { work ->
            val candidates = candidatesByProviderId[work.powerampFileId].orEmpty()
            if (candidates.isEmpty()) {
                additions += work
                return@forEach
            }

            val sharedLegacyTrackIds = candidates
                .map(V2LegacyCompatibilityBinding::trackId)
                .filter { trackId -> claimantsByLegacyTrackId.getValue(trackId).size > 1 }
                .distinct()
                .sorted()
            val reasons = buildList {
                if (candidates.map(V2LegacyCompatibilityBinding::trackId).distinct().size > 1) {
                    add(V2ImportedRowMappingBlockReason.MULTIPLE_LEGACY_ROWS_FOR_SELECTED_WORK)
                }
                if (sharedLegacyTrackIds.isNotEmpty()) {
                    add(
                        V2ImportedRowMappingBlockReason
                            .LEGACY_ROW_CLAIMED_BY_MULTIPLE_SELECTED_WORKS,
                    )
                }
            }
            if (reasons.isNotEmpty()) {
                blocked += V2BlockedImportedRowMapping(
                    selectedWork = work,
                    candidateLegacyTrackIds = candidates
                        .map(V2LegacyCompatibilityBinding::trackId)
                        .distinct()
                        .sorted(),
                    sharedLegacyTrackIds = sharedLegacyTrackIds,
                    reasons = reasons,
                )
                return@forEach
            }

            val binding = candidates.single()
            supersessions += V2ImportedRowSupersession(
                selectedWork = work,
                supersededTrackId = binding.trackId,
                evidence = binding.evidence,
            )
        }

        return V2ImportedRowSupersessionPlan(
            additions = additions,
            supersessions = supersessions,
            supersededTrackIds = supersessions.map(V2ImportedRowSupersession::supersededTrackId)
                .sorted(),
            blockedAmbiguousMappings = blocked,
        )
    }

    private fun requireSelectedWorkIsValid(selected: List<V2SelectedImportedRowWork>) {
        require(selected.mapTo(hashSetOf()) { it.workId }.size == selected.size) {
            "duplicate selected work ID"
        }
        require(selected.mapTo(hashSetOf()) { it.powerampFileId }.size == selected.size) {
            "duplicate selected Poweramp row ID"
        }
        require(selected.mapTo(hashSetOf()) { it.providerSpan }.size == selected.size) {
            "duplicate selected provider span"
        }
        selected.forEach { work ->
            require(work.workId.isNotBlank()) { "selected work ID must not be blank" }
            require(work.powerampFileId > 0L) { "selected Poweramp row ID must be positive" }
            require(work.providerSpan.offsetMs >= 0L) {
                "selected provider span offset must not be negative"
            }
            require(work.providerSpan.durationMs >= 0L) {
                "selected provider span duration must not be negative"
            }
            val normalizedPath = V2StableProviderLexicalPathNormalizer.normalizeAbsolute(
                work.providerSpan.normalizedPhysicalPath,
            )
            require(normalizedPath == work.providerSpan.normalizedPhysicalPath) {
                "selected provider span path must already be normalized"
            }
        }
    }

    private val selectedWorkComparator =
        compareBy<V2SelectedImportedRowWork> { it.providerSpan.normalizedPhysicalPath }
            .thenBy { it.providerSpan.offsetMs }
            .thenBy { it.providerSpan.durationMs }
            .thenBy(V2SelectedImportedRowWork::workId)
            .thenBy(V2SelectedImportedRowWork::powerampFileId)
}
