package com.powerampstartradio.indexing

/** Search controls visibility only. Selection changes require an explicit selection command. */
internal object V2IndexingSelectionPolicy {
    fun readyTrackIds(
        tracks: Collection<NewTrackDetector.UnindexedTrack>,
        hiddenIds: Set<Long>,
    ): Set<Long> = tracks.asSequence()
        .filter { isReadyTrack(it) && it.powerampFileId !in hiddenIds }
        .mapTo(linkedSetOf()) { it.powerampFileId }

    fun isReadyTrack(track: NewTrackDetector.UnindexedTrack): Boolean =
        track.detectionKind == V2UnindexedDetectionKind.DEFINITELY_UNINDEXED ||
            (track.detectionKind == V2UnindexedDetectionKind.SOURCE_ATTENTION &&
                track.durationMs <= 0 &&
                !track.path.isNullOrBlank() &&
                track.cueFolderId == null &&
                !track.sourceHasLogicalOffsets &&
                !track.sourceHasCueImageRow)

    fun canToggleTrackRow(
        track: NewTrackDetector.UnindexedTrack,
        allowNonReadySelection: Boolean,
    ): Boolean = allowNonReadySelection || isReadyTrack(track)

    /** Revalidate a confirmation snapshot without trusting a potentially reused numeric ID alone. */
    fun currentNonReadySourceAttentionMatch(
        requested: NewTrackDetector.UnindexedTrack,
        currentTracks: Collection<NewTrackDetector.UnindexedTrack>,
    ): NewTrackDetector.UnindexedTrack? {
        if (requested.detectionKind != V2UnindexedDetectionKind.SOURCE_ATTENTION ||
            isReadyTrack(requested)
        ) return null
        val requestedCandidate = V2TrackExclusionRepository.candidate(requested) ?: return null
        return currentTracks.firstOrNull { current ->
            current.powerampFileId == requested.powerampFileId &&
                current.detectionKind == V2UnindexedDetectionKind.SOURCE_ATTENTION &&
                !isReadyTrack(current) &&
                V2TrackExclusionRepository.candidate(current)?.providerSpan ==
                requestedCandidate.providerSpan
        }
    }

    fun selectedForJob(selectedIds: Set<Long>): Set<Long> = selectedIds.toSet()

    fun selectVisible(current: Set<Long>, selectableVisible: Set<Long>): Set<Long> =
        current + selectableVisible

    fun deselectVisible(current: Set<Long>, visible: Set<Long>): Set<Long> = current - visible
}

internal class V2CleanDatabaseConfirmation private constructor(
    val trackIds: Set<Long>,
) {
    val exactCount: Int get() = trackIds.size

    companion object {
        fun create(selectedIds: Set<Long>): V2CleanDatabaseConfirmation {
            require(selectedIds.isNotEmpty()) { "clean confirmation requires selected tracks" }
            require(selectedIds.all { it > 0L }) { "clean confirmation contains an invalid ID" }
            return V2CleanDatabaseConfirmation(selectedIds.toSet())
        }
    }
}

internal class V2NeverIndexConfirmation private constructor(
    val trackIds: Set<Long>,
) {
    val exactCount: Int get() = trackIds.size

    companion object {
        fun create(selectedIds: Set<Long>): V2NeverIndexConfirmation {
            require(selectedIds.isNotEmpty()) {
                "never-index confirmation requires selected tracks"
            }
            require(selectedIds.all { it > 0L }) {
                "never-index confirmation contains an invalid ID"
            }
            return V2NeverIndexConfirmation(selectedIds.toSet())
        }
    }
}
