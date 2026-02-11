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
    val candidatePoolSize: Int = 0,  // 0 = auto (2% of library, floor 100)
    val selectionMode: SelectionMode = SelectionMode.MMR,
    val driftEnabled: Boolean = false,
    val driftMode: DriftMode = DriftMode.SEED_INTERPOLATION,
    val anchorStrength: Float = 0.5f,
    val anchorDecay: DecaySchedule = DecaySchedule.EXPONENTIAL,
    val momentumBeta: Float = 0.7f,
    val diversityLambda: Float = 0.4f,     // Audit: 0.4 = +4 artists, -0.01 sim vs 0.6
    val temperature: Float = 0.05f,         // Audit: 0.1+ all equivalent; 0.05 is the effective edge
    val maxPerArtist: Int = 8,
    val minArtistSpacing: Int = 3,          // Audit: spacing=3 vs 5 (5 drops queue to ~44)
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
 * A single influence on a track's selection.
 *
 * @param sourceIndex -1 = seed track, 0..N-1 = index in result list
 * @param weight Exact mathematical influence weight (sums to ~1.0 per track)
 */
data class Influence(
    val sourceIndex: Int,
    val weight: Float,
)

/**
 * Full provenance for a queued track — every influence that shaped its selection.
 *
 * Batch modes: seed only. Seed interp: seed + previous track. EMA: all predecessors.
 */
data class TrackProvenance(
    val influences: List<Influence> = listOf(Influence(-1, 1f))
)

/**
 * Result of attempting to queue a single similar track.
 */
data class QueuedTrackResult(
    val track: EmbeddedTrack,
    val similarity: Float,
    val similarityToSeed: Float,
    val candidateRank: Int? = null,
    val status: QueueStatus,
    val provenance: TrackProvenance = TrackProvenance(),
)

/**
 * Queue quality metrics computed from embeddings after playlist generation.
 *
 * @param uniqueArtists Number of distinct artists in the queue
 * @param clusterSpread Number of distinct style clusters represented
 * @param simRange Pair of (min, max) similarity to seed, as percentages
 */
data class QueueMetrics(
    val uniqueArtists: Int,
    val clusterSpread: Int,
    val simRange: Pair<Int, Int>,
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
    val totalExpected: Int = 0,
    val metrics: QueueMetrics? = null
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
