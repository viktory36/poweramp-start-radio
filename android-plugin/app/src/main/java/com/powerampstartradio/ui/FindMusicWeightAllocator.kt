package com.powerampstartradio.ui

import kotlin.math.max

internal data class FindMusicWeightSlot(
    val weight: Float,
    val minimum: Float,
    val locked: Boolean,
)

/** One editor row, including whether it currently represents real query evidence. */
internal data class FindMusicEditorWeightSlot(
    val weight: Float,
    val locked: Boolean,
    val completed: Boolean,
)

/** Deterministic lower-bounded simplex allocation for the Find Music ingredient budget. */
internal object FindMusicWeightAllocator {
    private const val TARGET = 1f
    private const val EPSILON = 1e-6f

    fun adjust(
        slots: List<FindMusicWeightSlot>,
        changedIndex: Int,
        requestedWeight: Float,
    ): FloatArray {
        validate(slots)
        if (changedIndex !in slots.indices || slots[changedIndex].locked) {
            return FloatArray(slots.size) { slots[it].weight }
        }

        val lockedSum = slots.indices
            .filter { slots[it].locked }
            .fold(0f) { sum, index -> sum + slots[index].weight }
        val otherUnlockedFloor = slots.indices
            .filter { it != changedIndex && !slots[it].locked }
            .fold(0f) { sum, index -> sum + slots[index].minimum }
        val changed = slots[changedIndex]
        val maximum = (TARGET - lockedSum - otherUnlockedFloor)
            .coerceAtLeast(changed.minimum)
        val desired = requestedWeight
            .takeIf(Float::isFinite)
            ?.coerceIn(changed.minimum, maximum)
            ?: changed.weight

        val preferred = slots.mapIndexed { index, slot ->
            if (index == changedIndex) slot.copy(weight = desired) else slot
        }
        return allocate(
            slots = preferred,
            fixedIndices = slots.indices.filterTo(HashSet()) {
                slots[it].locked || it == changedIndex
            },
        )
    }

    fun normalize(slots: List<FindMusicWeightSlot>): FloatArray {
        validate(slots)
        return allocate(
            slots = slots,
            fixedIndices = slots.indices.filterTo(HashSet()) { slots[it].locked },
        )
    }

    /** True only when this share has another free share and non-floor budget to trade with. */
    fun canAdjust(
        slots: List<FindMusicWeightSlot>,
        changedIndex: Int,
    ): Boolean {
        validate(slots)
        if (changedIndex !in slots.indices || slots[changedIndex].locked) return false
        val unlockedIndices = slots.indices.filter { !slots[it].locked }
        if (unlockedIndices.size < 2) return false
        val lockedSum = slots.indices
            .filter { slots[it].locked }
            .fold(0f) { sum, index -> sum + slots[index].weight }
        val unlockedFloor = unlockedIndices.fold(0f) { sum, index ->
            sum + slots[index].minimum
        }
        return TARGET - lockedSum - unlockedFloor > EPSILON
    }

    private fun allocate(
        slots: List<FindMusicWeightSlot>,
        fixedIndices: Set<Int>,
    ): FloatArray {
        val result = FloatArray(slots.size) { slots[it].weight }
        val mutableIndices = slots.indices.filter { it !in fixedIndices }
        if (mutableIndices.isEmpty()) return result

        val fixedSum = fixedIndices.fold(0f) { sum, index -> sum + slots[index].weight }
        val available = TARGET - fixedSum
        val floorSum = mutableIndices.fold(0f) { sum, index -> sum + slots[index].minimum }
        if (available + EPSILON < floorSum) return result

        val excessBudget = max(0f, available - floorSum)
        val preferences = FloatArray(mutableIndices.size) { position ->
            val index = mutableIndices[position]
            max(0f, slots[index].weight - slots[index].minimum)
        }
        val preferenceSum = preferences.fold(0f, Float::plus)
        val equalPreference = preferenceSum <= EPSILON

        var assigned = 0f
        mutableIndices.forEachIndexed { position, index ->
            var value = if (position == mutableIndices.lastIndex) {
                // One residual assignment prevents independent floor clamps from exceeding 100%.
                available - assigned
            } else {
                val share = if (equalPreference) {
                    excessBudget / mutableIndices.size
                } else {
                    excessBudget * preferences[position] / preferenceSum
                }
                slots[index].minimum + share
            }
            if (position == mutableIndices.lastIndex && value < slots[index].minimum) {
                var deficit = slots[index].minimum - value
                for (donorPosition in position - 1 downTo 0) {
                    val donorIndex = mutableIndices[donorPosition]
                    val room = result[donorIndex] - slots[donorIndex].minimum
                    val taken = minOf(room, deficit)
                    result[donorIndex] -= taken
                    deficit -= taken
                    if (deficit <= EPSILON) break
                }
                value = slots[index].minimum
            }
            result[index] = value
            assigned += result[index]
        }
        return result
    }

    private fun validate(slots: List<FindMusicWeightSlot>) {
        require(slots.isNotEmpty()) { "At least one weight slot is required" }
        require(slots.all {
            it.weight.isFinite() && it.weight >= 0f &&
                it.minimum.isFinite() && it.minimum >= 0f && it.minimum <= 1f
        }) { "Find Music weights and floors must be finite and non-negative" }
    }
}

/**
 * Structural weight transitions for the Find Music editor.
 *
 * Incomplete rows are placeholders, not zero-weight query ingredients. They stay pinned to zero
 * and never absorb residual weight. A held share is literal while tuning one fixed ingredient set;
 * structural editor changes release holds before invoking this policy. If a direct caller keeps
 * impossible holds, activation stays at zero and editor readiness explains how to proceed.
 */
internal object FindMusicEditorWeightPolicy {
    private const val TARGET = 1f
    const val MINIMUM_ACTIVE_WEIGHT = 0.01f

    fun minimumActiveWeight(
        @Suppress("UNUSED_PARAMETER") operator: FindMusicOperator,
        @Suppress("UNUSED_PARAMETER") resultLimit: Int,
    ): Float = MINIMUM_ACTIVE_WEIGHT

    fun normalize(
        slots: List<FindMusicEditorWeightSlot>,
        minimumActiveWeight: Float = MINIMUM_ACTIVE_WEIGHT,
    ): FloatArray {
        validateMinimum(minimumActiveWeight)
        if (slots.isEmpty()) return FloatArray(0)
        val activeIndices = slots.indices.filter { slots[it].completed }
        if (activeIndices.isEmpty()) return FloatArray(slots.size)
        if (activeIndices.size == 1) {
            return FloatArray(slots.size).also { it[activeIndices.single()] = TARGET }
        }
        return FindMusicWeightAllocator.normalize(slots.toAllocatorSlots(minimumActiveWeight))
    }

    fun activate(
        slots: List<FindMusicEditorWeightSlot>,
        activatedIndex: Int,
        minimumActiveWeight: Float = MINIMUM_ACTIVE_WEIGHT,
    ): FloatArray {
        validateMinimum(minimumActiveWeight)
        require(activatedIndex in slots.indices) { "Activated Find Music row is out of range" }
        require(slots[activatedIndex].completed) { "Activated Find Music row must be complete" }

        val activeIndices = slots.indices.filter { slots[it].completed }
        if (activeIndices.size == 1) {
            return FloatArray(slots.size).also { it[activatedIndex] = TARGET }
        }

        val existingIndices = activeIndices.filter { it != activatedIndex }
        val lockedIndices = existingIndices.filter { slots[it].locked }
        val unlockedIndices = existingIndices.filterNot { slots[it].locked }
        val lockedSum = lockedIndices.sumOf { slots[it].weight.toDouble() }.toFloat()
        val requiredUnlockedFloor = minimumActiveWeight * (unlockedIndices.size + 1)
        if (lockedSum + requiredUnlockedFloor > TARGET + EPSILON) {
            return FloatArray(slots.size) { slots[it].weight }
        }
        val adjustedSlots = slots.mapIndexed { index, slot ->
            when {
                !slot.completed -> slot.copy(weight = 0f, locked = true)
                index == activatedIndex -> slot.copy(weight = 0f, locked = false)
                else -> slot
            }
        }
        val adjustedLockedSum = lockedIndices
            .sumOf { adjustedSlots[it].weight.toDouble() }
            .toFloat()
        val newShare = ((TARGET - adjustedLockedSum) / (unlockedIndices.size + 1))
            .coerceAtLeast(minimumActiveWeight)
        val allocatorSlots = adjustedSlots.toAllocatorSlots(minimumActiveWeight)
            .mapIndexed { index, slot ->
            if (index == activatedIndex) slot.copy(weight = newShare) else slot
        }
        return FindMusicWeightAllocator.adjust(
            slots = allocatorSlots,
            changedIndex = activatedIndex,
            requestedWeight = newShare,
        )
    }

    fun canAdjust(
        slots: List<FindMusicEditorWeightSlot>,
        changedIndex: Int,
        minimumActiveWeight: Float = MINIMUM_ACTIVE_WEIGHT,
    ): Boolean = FindMusicWeightAllocator.canAdjust(
        slots = slots.toAllocatorSlots(minimumActiveWeight),
        changedIndex = changedIndex,
    )

    private fun List<FindMusicEditorWeightSlot>.toAllocatorSlots(
        minimumActiveWeight: Float,
    ): List<FindMusicWeightSlot> =
        map { slot ->
            if (slot.completed) {
                FindMusicWeightSlot(
                    weight = slot.weight,
                    minimum = minimumActiveWeight,
                    locked = slot.locked,
                )
            } else {
                FindMusicWeightSlot(weight = 0f, minimum = 0f, locked = true)
            }
        }

    private fun validateMinimum(minimumActiveWeight: Float) {
        require(minimumActiveWeight.isFinite() && minimumActiveWeight in 0f..1f)
    }

    private const val EPSILON = 1e-6f
}
