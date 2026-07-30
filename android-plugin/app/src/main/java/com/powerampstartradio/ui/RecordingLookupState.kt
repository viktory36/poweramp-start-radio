package com.powerampstartradio.ui

import com.powerampstartradio.data.EmbeddedTrack
import java.util.Locale

sealed interface RecordingLookupState {
    data object Idle : RecordingLookupState

    data class Loading(
        val seedId: Long,
        val query: String,
        val message: String,
    ) : RecordingLookupState

    data class Success(
        val seedId: Long,
        val query: String,
        val candidates: List<EmbeddedTrack>,
        val hasMoreMatches: Boolean,
    ) : RecordingLookupState

    data class Failure(
        val seedId: Long,
        val query: String,
        val message: String,
    ) : RecordingLookupState
}

internal object RecordingLookupStateReducer {
    fun start(seedId: Long, query: String): RecordingLookupState =
        RecordingLookupState.Loading(
            seedId,
            query,
            "Opening the active music index for this recording lookup",
        )

    fun progress(
        current: RecordingLookupState,
        seedId: Long,
        query: String,
        message: String,
    ): RecordingLookupState = if (current.matches(seedId, query)) {
        RecordingLookupState.Loading(seedId, query, message)
    } else {
        current
    }

    fun succeed(
        current: RecordingLookupState,
        seedId: Long,
        query: String,
        candidates: List<EmbeddedTrack>,
        hasMoreMatches: Boolean = false,
    ): RecordingLookupState = if (current.matches(seedId, query)) {
        RecordingLookupState.Success(
            seedId = seedId,
            query = query,
            candidates = candidates.toList(),
            hasMoreMatches = hasMoreMatches,
        )
    } else {
        current
    }

    fun fail(
        current: RecordingLookupState,
        seedId: Long,
        query: String,
        message: String,
    ): RecordingLookupState = if (current.matches(seedId, query)) {
        RecordingLookupState.Failure(seedId, query, message)
    } else {
        current
    }

    fun clear(): RecordingLookupState = RecordingLookupState.Idle

    private fun RecordingLookupState.matches(seedId: Long, query: String): Boolean =
        this is RecordingLookupState.Loading && this.seedId == seedId && this.query == query
}

internal fun RecordingLookupState.forSeed(seedId: Long): RecordingLookupState = when (this) {
    RecordingLookupState.Idle -> this
    is RecordingLookupState.Loading -> takeIf { it.seedId == seedId } ?: RecordingLookupState.Idle
    is RecordingLookupState.Success -> takeIf { it.seedId == seedId } ?: RecordingLookupState.Idle
    is RecordingLookupState.Failure -> takeIf { it.seedId == seedId } ?: RecordingLookupState.Idle
}

internal data class RecordingCandidateEvidence(
    val artistAndAlbum: String,
    val durationAndFile: String,
)

internal object RecordingCandidateEvidenceFormatter {
    fun format(track: EmbeddedTrack): RecordingCandidateEvidence {
        val artistAndAlbum = listOfNotNull(
            track.artist?.takeIf(String::isNotBlank),
            track.album?.takeIf(String::isNotBlank),
        ).joinToString(" \u00b7 ").ifBlank { "Unknown artist and album" }
        val duration = formatDuration(track.durationMs)
        val fileName = track.filePath
            .substringAfterLast('/')
            .substringAfterLast('\\')
            .takeIf(String::isNotBlank)
        return RecordingCandidateEvidence(
            artistAndAlbum = artistAndAlbum,
            durationAndFile = listOfNotNull(duration, fileName).joinToString(" \u00b7 ")
                .ifBlank { "Duration unavailable" },
        )
    }

    private fun formatDuration(durationMs: Int): String? {
        if (durationMs <= 0) return null
        val totalSeconds = durationMs / 1_000
        val hours = totalSeconds / 3_600
        val minutes = (totalSeconds % 3_600) / 60
        val seconds = totalSeconds % 60
        return if (hours > 0) {
            String.format(Locale.US, "%d:%02d:%02d", hours, minutes, seconds)
        } else {
            String.format(Locale.US, "%d:%02d", minutes, seconds)
        }
    }
}
