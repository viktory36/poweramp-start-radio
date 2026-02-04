package com.powerampstartradio.ui

import com.powerampstartradio.data.EmbeddedTrack
import com.powerampstartradio.poweramp.PowerampTrack
import com.powerampstartradio.poweramp.TrackMatcher

/**
 * User-selectable recommendation algorithm.
 */
enum class SelectionMode {
    MMR,
    DPP,
    RANDOM_WALK,
    TEMPERATURE
}

/**
 * How the query evolves across drift steps.
 */
enum class DriftMode {
    SEED_INTERPOLATION,
    MOMENTUM
}

/**
 * How anchor strength decays over time in seed interpolation.
 */
enum class DecaySchedule {
    NONE,
    LINEAR,
    EXPONENTIAL,
    STEP
}

/**
 * Full configuration for a radio session.
 */
data class RadioConfig(
    val numTracks: Int = 50,
    val candidatePoolSize: Int = 200,
    val selectionMode: SelectionMode = SelectionMode.MMR,
    val driftEnabled: Boolean = true,
    val driftMode: DriftMode = DriftMode.SEED_INTERPOLATION,
    val anchorStrength: Float = 0.5f,
    val anchorDecay: DecaySchedule = DecaySchedule.EXPONENTIAL,
    val momentumBeta: Float = 0.7f,
    val diversityLambda: Float = 0.6f,
    val temperature: Float = 0.1f,
    val maxPerArtist: Int = 3,
    val minArtistSpacing: Int = 5,
)

/**
 * Status of a single track in the queue operation.
 */
enum class QueueStatus {
    QUEUED,
    NOT_IN_LIBRARY,
    QUEUE_FAILED
}

/**
 * Result of attempting to queue a single similar track.
 */
data class QueuedTrackResult(
    val track: EmbeddedTrack,
    val similarity: Float,
    val status: QueueStatus,
)

/**
 * Complete result of a "Start Radio" operation.
 */
data class RadioResult(
    val seedTrack: PowerampTrack,
    val matchType: TrackMatcher.MatchType,
    val tracks: List<QueuedTrackResult>,
    val config: RadioConfig = RadioConfig(),
    val timestamp: Long = System.currentTimeMillis(),
    val queuedFileIds: Set<Long> = emptySet(),
    val isComplete: Boolean = true,
    val totalExpected: Int = 0
) {
    val queuedCount: Int get() = tracks.count { it.status == QueueStatus.QUEUED }
    val failedCount: Int get() = tracks.count { it.status != QueueStatus.QUEUED }
    val requestedCount: Int get() = tracks.size
}

/**
 * UI state for the main screen.
 */
sealed class RadioUiState {
    object Idle : RadioUiState()
    data class Loading(val message: String = "Starting radio...") : RadioUiState()
    data class Searching(val message: String) : RadioUiState()
    data class Streaming(val result: RadioResult) : RadioUiState()
    data class Success(val result: RadioResult) : RadioUiState()
    data class Error(val message: String) : RadioUiState()
}
