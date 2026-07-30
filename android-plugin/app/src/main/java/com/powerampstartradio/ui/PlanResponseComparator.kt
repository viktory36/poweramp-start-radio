package com.powerampstartradio.ui

import java.util.Locale

/** Whether this snapshot came from a new selector run or from a previously materialized plan. */
enum class PlanMaterialization {
    FRESH,
    REUSED,
}

/**
 * One effective control that can change the selected mode's plan.
 *
 * Inapplicable saved settings must not be included. [valueKey] is the canonical semantic value;
 * the display fields are only the factual text shown in a comparison.
 */
data class PlanSemanticControl(
    val id: String,
    val valueKey: String,
    val displayName: String,
    val displayValue: String,
) {
    init {
        require(id.isNotBlank()) { "Control ID must not be blank" }
        require(valueKey.isNotBlank()) { "Control value key must not be blank" }
        require(displayName.isNotBlank()) { "Control display name must not be blank" }
        require(displayValue.isNotBlank()) { "Control display value must not be blank" }
    }
}

/**
 * Exact output of one complete planning run, before Poweramp delivery.
 *
 * [seedIdentity] is the exact generation row used as the seed. Track IDs are exact row identities
 * within [generation], so snapshots from different generations are never
 * compared. [semanticControls] contains effective controls only. [candidateCount] is the exact
 * semantic selector domain, never a temporary work buffer or an automatic-DPP proof prefix.
 */
data class PlanSnapshot(
    val planningRunId: String,
    val materialization: PlanMaterialization,
    val generation: RadioGenerationToken,
    val providerGenerationId: String,
    val seedIdentity: RadioSeedIdentity,
    val selectionMode: SelectionMode,
    val semanticControls: List<PlanSemanticControl>,
    val candidateCount: Int,
    val orderedTrackIds: List<Long>,
) {
    init {
        require(planningRunId.isNotBlank()) { "Planning run ID must not be blank" }
        require(providerGenerationId.isNotBlank()) { "Provider generation ID must not be blank" }
        require(seedIdentity.embeddedTrackId > 0L) { "Seed row ID must be positive" }
        require(candidateCount >= 0) { "Candidate count must be non-negative" }
        require(semanticControls.map { it.id }.distinct().size == semanticControls.size) {
            "Semantic control IDs must be unique"
        }
        require(orderedTrackIds.distinct().size == orderedTrackIds.size) {
            "A plan must not contain the same generation-scoped track row twice"
        }
    }
}

enum class PlanComparisonRejection {
    PLAN_NOT_FRESH,
    SAME_PLANNING_RUN,
    GENERATION_MISMATCH,
    PROVIDER_GENERATION_MISMATCH,
    SEED_MISMATCH,
    MODE_MISMATCH,
    CONTROL_SET_MISMATCH,
    NO_SEMANTIC_CONTROL_CHANGE,
    MULTIPLE_SEMANTIC_CONTROLS_CHANGED,
}

sealed interface PlanComparisonResult {
    data class Compared(val response: PlanControlResponse) : PlanComparisonResult

    data class Rejected(
        val reason: PlanComparisonRejection,
        val message: String,
    ) : PlanComparisonResult
}

data class PlanControlChange(
    val id: String,
    val displayName: String,
    val beforeValueKey: String,
    val afterValueKey: String,
    val beforeDisplayValue: String,
    val afterDisplayValue: String,
) {
    val message: String
        get() = if (id == "shuffle_seed") {
            "Reproducible order changed."
        } else {
            "$displayName changed from $beforeDisplayValue to $afterDisplayValue."
        }
}

data class CandidateCountResponse(
    val before: Int,
    val after: Int,
) {
    val delta: Int get() = after - before
    val changed: Boolean get() = before != after

    val message: String
        get() = if (!changed) {
            "Tracks considered: ${formatCount(before)} (unchanged)."
        } else {
            "Tracks considered: ${formatCount(before)} to ${formatCount(after)}."
        }

    private fun formatCount(value: Int): String = String.format(Locale.US, "%,d", value)
}

enum class QueuePlanResponseKind {
    EXACTLY_UNCHANGED,
    SAME_SET_REORDERED,
    SET_CHANGED,
}

data class QueuePlanResponse(
    val kind: QueuePlanResponseKind,
    val beforeCount: Int,
    val afterCount: Int,
    val retainedTrackCount: Int,
    /** Generation-scoped row IDs, in their original plan order. */
    val removedTrackIds: List<Long>,
    /** Generation-scoped row IDs, in their new plan order. */
    val addedTrackIds: List<Long>,
    val samePositionCount: Int,
) {
    val exactSetUnchanged: Boolean
        get() = removedTrackIds.isEmpty() && addedTrackIds.isEmpty()
    val exactOrderUnchanged: Boolean
        get() = kind == QueuePlanResponseKind.EXACTLY_UNCHANGED

    val message: String
        get() = when (kind) {
            QueuePlanResponseKind.EXACTLY_UNCHANGED ->
                "Queue unchanged: same $beforeCount tracks in the same order."
            QueuePlanResponseKind.SAME_SET_REORDERED ->
                "Queue reordered: same $beforeCount tracks; " +
                    positionResponse(samePositionCount, beforeCount)
            QueuePlanResponseKind.SET_CHANGED -> if (beforeCount == afterCount) {
                "$afterCount-track queue changed: $retainedTrackCount retained, " +
                    "${addedTrackIds.size} replaced; " +
                    sameTrackPositionResponse(samePositionCount)
            } else {
                "Queue changed from $beforeCount to $afterCount tracks; " +
                    "$retainedTrackCount retained, " +
                    "${removedTrackIds.size} removed, ${addedTrackIds.size} added; " +
                    sameTrackPositionResponse(samePositionCount)
            }
        }

    private fun positionResponse(unchanged: Int, total: Int): String =
        if (unchanged == 1) {
            "1 of $total positions is unchanged."
        } else {
            "$unchanged of $total positions are unchanged."
        }

    private fun sameTrackPositionResponse(count: Int): String =
        when (count) {
            0 -> "no track stayed in the same position."
            1 -> "1 track stayed in the same position."
            else -> "$count tracks stayed in the same position."
        }
}

data class PlanControlResponse(
    val changedControl: PlanControlChange,
    val candidates: CandidateCountResponse,
    val queue: QueuePlanResponse,
) {
    val messages: List<String>
        get() = listOfNotNull(
            changedControl.message,
            candidates.message.takeIf { candidates.changed },
            queue.message,
        )
}

/** Strict comparator for measuring the response to exactly one effective semantic control. */
object PlanResponseComparator {
    fun compare(before: PlanSnapshot, after: PlanSnapshot): PlanComparisonResult {
        if (before.materialization != PlanMaterialization.FRESH ||
            after.materialization != PlanMaterialization.FRESH
        ) {
            return rejected(
                PlanComparisonRejection.PLAN_NOT_FRESH,
                "Both results must come from fresh planning runs.",
            )
        }
        if (before.planningRunId == after.planningRunId) {
            return rejected(
                PlanComparisonRejection.SAME_PLANNING_RUN,
                "The results identify the same planning run.",
            )
        }
        if (before.generation != after.generation) {
            return rejected(
                PlanComparisonRejection.GENERATION_MISMATCH,
                "The embedding generation changed between plans.",
            )
        }
        if (before.providerGenerationId != after.providerGenerationId) {
            return rejected(
                PlanComparisonRejection.PROVIDER_GENERATION_MISMATCH,
                "The Poweramp library generation changed between plans.",
            )
        }
        if (before.seedIdentity != after.seedIdentity) {
            return rejected(
                PlanComparisonRejection.SEED_MISMATCH,
                "The seed or query changed between plans.",
            )
        }
        if (before.selectionMode != after.selectionMode) {
            return rejected(
                PlanComparisonRejection.MODE_MISMATCH,
                "The selection mode changed between plans.",
            )
        }

        val beforeControls = before.semanticControls.associateBy { it.id }
        val afterControls = after.semanticControls.associateBy { it.id }
        if (beforeControls.keys != afterControls.keys) {
            return rejected(
                PlanComparisonRejection.CONTROL_SET_MISMATCH,
                "The effective control set changed between plans.",
            )
        }
        val changedControlIds = beforeControls.keys
            .filter { id -> beforeControls.getValue(id).valueKey != afterControls.getValue(id).valueKey }
            .sorted()
        if (changedControlIds.isEmpty()) {
            return rejected(
                PlanComparisonRejection.NO_SEMANTIC_CONTROL_CHANGE,
                "No effective semantic control changed.",
            )
        }
        if (changedControlIds.size > 1) {
            return rejected(
                PlanComparisonRejection.MULTIPLE_SEMANTIC_CONTROLS_CHANGED,
                "${changedControlIds.size} effective semantic controls changed.",
            )
        }

        val changedId = changedControlIds.single()
        val oldControl = beforeControls.getValue(changedId)
        val newControl = afterControls.getValue(changedId)
        val beforeSet = before.orderedTrackIds.toHashSet()
        val afterSet = after.orderedTrackIds.toHashSet()
        val removed = before.orderedTrackIds.filterNot(afterSet::contains)
        val added = after.orderedTrackIds.filterNot(beforeSet::contains)
        val exactOrderUnchanged = before.orderedTrackIds == after.orderedTrackIds
        val kind = when {
            exactOrderUnchanged -> QueuePlanResponseKind.EXACTLY_UNCHANGED
            removed.isEmpty() && added.isEmpty() -> QueuePlanResponseKind.SAME_SET_REORDERED
            else -> QueuePlanResponseKind.SET_CHANGED
        }

        return PlanComparisonResult.Compared(
            PlanControlResponse(
                changedControl = PlanControlChange(
                    id = changedId,
                    displayName = newControl.displayName,
                    beforeValueKey = oldControl.valueKey,
                    afterValueKey = newControl.valueKey,
                    beforeDisplayValue = oldControl.displayValue,
                    afterDisplayValue = newControl.displayValue,
                ),
                candidates = CandidateCountResponse(
                    before = before.candidateCount,
                    after = after.candidateCount,
                ),
                queue = QueuePlanResponse(
                    kind = kind,
                    beforeCount = before.orderedTrackIds.size,
                    afterCount = after.orderedTrackIds.size,
                    retainedTrackCount = beforeSet.intersect(afterSet).size,
                    removedTrackIds = removed,
                    addedTrackIds = added,
                    samePositionCount = before.orderedTrackIds.zip(after.orderedTrackIds)
                        .count { (oldId, newId) -> oldId == newId },
                ),
            ),
        )
    }

    private fun rejected(
        reason: PlanComparisonRejection,
        message: String,
    ): PlanComparisonResult.Rejected = PlanComparisonResult.Rejected(reason, message)
}
