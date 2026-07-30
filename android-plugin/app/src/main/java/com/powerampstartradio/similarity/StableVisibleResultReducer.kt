package com.powerampstartradio.similarity

import com.powerampstartradio.data.StableVisibleResultIdentity

data class StableVisibleReduction<T>(
    val items: List<T>,
    val requestedVisibleCount: Int,
    val scannedRowCount: Int,
    val collapsedEquivalentCount: Int,
)

/** Pure ranked-prefix reduction shared by text retrieval and composed radio. */
object StableVisibleResultReducer {
    const val IDENTITY_POLICY_VERSION = 3

    fun <T> reduce(
        rankedItems: Iterable<T>,
        requestedVisibleCount: Int,
        identityOf: (T) -> StableVisibleResultIdentity,
        isEligible: (T) -> Boolean = { true },
    ): StableVisibleReduction<T> {
        require(requestedVisibleCount >= 0) { "Requested visible count cannot be negative" }
        if (requestedVisibleCount == 0) {
            return StableVisibleReduction(emptyList(), 0, 0, 0)
        }

        val selected = ArrayList<T>(requestedVisibleCount)
        val selectedStableIdentities = HashSet<String>()
        var scanned = 0
        var collapsed = 0
        for (item in rankedItems) {
            scanned++
            val identity = identityOf(item)
            if (identity.isCollapsibleRecording &&
                !selectedStableIdentities.add(identity.identityToken)
            ) {
                collapsed++
                continue
            }
            if (!isEligible(item)) {
                if (identity.isCollapsibleRecording) {
                    selectedStableIdentities.remove(identity.identityToken)
                }
                continue
            }
            selected += item
            if (selected.size == requestedVisibleCount) break
        }
        return StableVisibleReduction(
            items = selected,
            requestedVisibleCount = requestedVisibleCount,
            scannedRowCount = scanned,
            collapsedEquivalentCount = collapsed,
        )
    }
}
