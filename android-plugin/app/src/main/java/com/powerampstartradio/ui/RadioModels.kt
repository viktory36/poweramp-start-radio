package com.powerampstartradio.ui

import com.powerampstartradio.data.EmbeddedTrack
import com.powerampstartradio.data.EmbeddingModel
import com.powerampstartradio.poweramp.PowerampTrack
import com.powerampstartradio.poweramp.TrackMatcher
import com.powerampstartradio.similarity.SearchStrategy

/**
 * Status of a single track in the queue operation.
 */
enum class QueueStatus {
    QUEUED,           // Successfully added to Poweramp queue
    NOT_IN_LIBRARY,   // Track not found in Poweramp library
    QUEUE_FAILED      // Found but failed to add to queue
}

/**
 * Result of attempting to queue a single similar track.
 */
data class QueuedTrackResult(
    val track: EmbeddedTrack,
    val similarity: Float,       // 0.0-1.0 raw cosine similarity
    val status: QueueStatus,
    val modelUsed: EmbeddingModel? = null
)

/**
 * Complete result of a "Start Radio" operation.
 */
data class RadioResult(
    val seedTrack: PowerampTrack,
    val matchType: TrackMatcher.MatchType,
    val tracks: List<QueuedTrackResult>,
    val availableModels: Set<EmbeddingModel> = emptySet(),
    val strategy: SearchStrategy = SearchStrategy.FEED_FORWARD,
    val timestamp: Long = System.currentTimeMillis()
) {
    val queuedCount: Int get() = tracks.count { it.status == QueueStatus.QUEUED }
    val failedCount: Int get() = tracks.count { it.status != QueueStatus.QUEUED }
    val requestedCount: Int get() = tracks.size
    val isMultiModel: Boolean get() = strategy == SearchStrategy.INTERLEAVE || strategy == SearchStrategy.FEED_FORWARD
}

/**
 * UI state for the main screen.
 */
sealed class RadioUiState {
    object Idle : RadioUiState()
    object Loading : RadioUiState()
    data class Success(val result: RadioResult) : RadioUiState()
    data class Error(val message: String) : RadioUiState()
}
