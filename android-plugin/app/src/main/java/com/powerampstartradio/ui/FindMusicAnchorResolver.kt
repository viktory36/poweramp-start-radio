package com.powerampstartradio.ui

import com.powerampstartradio.data.StableTrackIdentityCatalog
import com.powerampstartradio.data.StableTrackIdentityResolution

internal sealed interface FindMusicAnchorBindingResult {
    data class Success(
        val querySpec: FindMusicQuerySpec,
        val equivalentTrackIdsToExclude: Set<Long>,
    ) : FindMusicAnchorBindingResult

    data class Failure(
        val message: String,
        val unresolvedAnchors: List<FindMusicSongAnchor>,
        val diagnosticDetail: String? = null,
    ) : FindMusicAnchorBindingResult
}

/** Resolves persisted song ingredients without consulting display metadata or paths. */
internal object FindMusicAnchorResolver {
    fun bindCurrent(
        querySpec: FindMusicQuerySpec,
        catalog: StableTrackIdentityCatalog,
    ): FindMusicAnchorBindingResult {
        val resolved = ArrayList<FindMusicSongAnchor>(querySpec.songSeeds.size)
        val equivalentTrackIds = LinkedHashSet<Long>()
        val failures = ArrayList<Pair<FindMusicSongAnchor, StableTrackIdentityResolution>>()
        querySpec.songSeeds.forEach { anchor ->
            if (anchor.weight <= 0f) {
                resolved += anchor
                return@forEach
            }
            val resolution = anchor.stableTrackSpanId?.let(catalog::resolveStable)
                ?: catalog.resolveLegacy(anchor.trackId, querySpec.libraryBinding)
            if (resolution is StableTrackIdentityResolution.Resolved) {
                resolved += anchor.copy(
                    trackId = resolution.trackId,
                    stableTrackSpanId = anchor.stableTrackSpanId
                        ?: catalog.stableTrackSpanId(resolution.trackId),
                )
                equivalentTrackIds += resolution.allEquivalentTrackIds
            } else {
                failures += anchor to resolution
            }
        }
        if (failures.isNotEmpty()) {
            val anchors = failures.map { it.first }
            return FindMusicAnchorBindingResult.Failure(
                message = missingMessage(
                    if (anchors.size == 1) {
                        "This recording is no longer in the current indexed library. Choose it again"
                    } else {
                        "These recordings are no longer in the current indexed library. Choose them again"
                    },
                    anchors,
                ),
                unresolvedAnchors = anchors,
                diagnosticDetail = "Current anchor binding failed: " +
                    diagnosticResolutions(failures),
            )
        }
        val boundSpec = querySpec.copy(
            schemaVersion = FindMusicQuerySpec.CURRENT_SCHEMA_VERSION,
            songSeeds = resolved,
            libraryBinding = catalog.binding,
        )
        val duplicateStableAnchors = duplicateStableAnchors(boundSpec)
        if (duplicateStableAnchors.isNotEmpty()) {
            val duplicates = duplicateStableAnchors.values.flatten()
            return FindMusicAnchorBindingResult.Failure(
                message = missingMessage(
                    "The same indexed audio was selected more than once. Keep one ingredient with the intended weight",
                    duplicates,
                ),
                unresolvedAnchors = duplicates,
                diagnosticDetail = "Duplicate stable track-span anchors: " +
                    duplicates.joinToString(", ") { it.stableTrackSpanId.orEmpty() },
            )
        }
        return FindMusicAnchorBindingResult.Success(
            querySpec = boundSpec,
            equivalentTrackIdsToExclude = equivalentTrackIds,
        )
    }

    fun resolveReplay(
        querySpec: FindMusicQuerySpec,
        catalog: StableTrackIdentityCatalog,
    ): FindMusicAnchorBindingResult {
        val resolved = ArrayList<FindMusicSongAnchor>(querySpec.songSeeds.size)
        val equivalentTrackIds = LinkedHashSet<Long>()
        val failures = ArrayList<Pair<FindMusicSongAnchor, StableTrackIdentityResolution>>()
        querySpec.songSeeds.forEach { anchor ->
            if (anchor.weight <= 0f) {
                resolved += anchor
                return@forEach
            }
            val resolution = anchor.stableTrackSpanId?.let(catalog::resolveStable)
                ?: catalog.resolveLegacy(anchor.trackId, querySpec.libraryBinding)
            when (resolution) {
                is StableTrackIdentityResolution.Resolved -> {
                    resolved += anchor.copy(trackId = resolution.trackId)
                    equivalentTrackIds += resolution.allEquivalentTrackIds
                }
                else -> failures += anchor to resolution
            }
        }
        if (failures.isNotEmpty()) {
            val anchors = failures.map { it.first }
            val diagnosticReason = when {
                failures.any { it.second == StableTrackIdentityResolution.LegacyBindingRequired } ->
                    "This search predates generation-bound source-span identity evidence"
                failures.any { it.second == StableTrackIdentityResolution.LegacyBindingMismatch } ->
                    "The legacy indexed row IDs belong to a different embedding database generation"
                else -> "The saved song ingredient is not indexed in this library generation"
            }
            return FindMusicAnchorBindingResult.Failure(
                message = missingMessage(
                    if (anchors.size == 1) {
                        "This saved search no longer matches the current library. Recreate it and choose the recording again"
                    } else {
                        "This saved search no longer matches the current library. Recreate it and choose the recordings again"
                    },
                    anchors,
                ),
                unresolvedAnchors = anchors,
                diagnosticDetail = "$diagnosticReason: ${diagnosticResolutions(failures)}",
            )
        }
        val resolvedSpec = querySpec.copy(
            schemaVersion = FindMusicQuerySpec.CURRENT_SCHEMA_VERSION,
            songSeeds = resolved,
            libraryBinding = catalog.binding,
        )
        val duplicateStableAnchors = duplicateStableAnchors(resolvedSpec)
        if (duplicateStableAnchors.isNotEmpty()) {
            val duplicates = duplicateStableAnchors.values.flatten()
            return FindMusicAnchorBindingResult.Failure(
                message = missingMessage(
                    "The same indexed audio was selected more than once. Keep one ingredient with the intended weight",
                    duplicates,
                ),
                unresolvedAnchors = duplicates,
                diagnosticDetail = "Duplicate stable track-span anchors: " +
                    duplicates.joinToString(", ") { it.stableTrackSpanId.orEmpty() },
            )
        }
        return FindMusicAnchorBindingResult.Success(
            querySpec = resolvedSpec,
            equivalentTrackIdsToExclude = equivalentTrackIds,
        )
    }

    private fun duplicateStableAnchors(
        querySpec: FindMusicQuerySpec,
    ): Map<String?, List<FindMusicSongAnchor>> = querySpec.songSeeds
        .filter { it.weight > 0f && it.stableTrackSpanId != null }
        .groupBy { it.stableTrackSpanId }
        .filterValues { it.size > 1 }

    private fun missingMessage(prefix: String, anchors: List<FindMusicSongAnchor>): String =
        "$prefix: ${anchors.joinToString(", ") { it.displayLabel }}."

    private fun diagnosticResolutions(
        failures: List<Pair<FindMusicSongAnchor, StableTrackIdentityResolution>>,
    ): String = failures.joinToString(", ") { (anchor, resolution) ->
        "${anchor.displayLabel}=$resolution"
    }
}
