package com.powerampstartradio.widget

import android.util.AtomicFile
import com.google.gson.Gson
import com.powerampstartradio.poweramp.PowerampFileEntry
import com.powerampstartradio.poweramp.PowerampTrack
import com.powerampstartradio.poweramp.TrackNormalization
import com.powerampstartradio.ui.RadioSeedIdentity
import java.io.File
import java.io.FileNotFoundException
import java.util.concurrent.ConcurrentHashMap

internal enum class WidgetRadioRequestState {
    STARTING,
    BUSY,
    WAITING_FOR_INDEXING,
    SUCCEEDED,
    PARTIAL_FAILED,
    CANCELLED,
    FAILED,
}

/** Exact seed identity used to keep request status from leaking under a later track title. */
internal data class WidgetRadioSeedReference(
    val powerampFileId: Long,
    val normalizedPath: String?,
    val normalizedTitle: String,
    val displayTitle: String,
    val queueOccurrenceId: Long? = null,
    val embeddedTrackId: Long? = null,
    val stableTrackSpanId: String? = null,
) {
    fun matches(track: PowerampTrack): Boolean {
        if (powerampFileId <= 0L || track.realId != powerampFileId) return false
        val candidatePath = track.path
            ?.let(TrackNormalization::normalizePath)
            ?.takeIf(String::isNotBlank)
        if (normalizedPath != null && candidatePath != normalizedPath) return false
        val candidateQueueOccurrenceId = track.queueOccurrenceId
        if ((queueOccurrenceId != null || candidateQueueOccurrenceId != null) &&
            queueOccurrenceId != candidateQueueOccurrenceId
        ) {
            return false
        }
        return TrackNormalization.normalizeTitle(track.title) == normalizedTitle
    }

    /** Queue occurrence is playback context; provider identity is file/path/title owned. */
    fun matchesProvider(entry: PowerampFileEntry): Boolean {
        if (powerampFileId <= 0L || entry.id != powerampFileId) return false
        val providerPath = entry.path?.let(TrackNormalization::normalizePath)
            ?.takeIf(String::isNotBlank)
        if (normalizedPath != null && providerPath != normalizedPath) return false
        return entry.title == normalizedTitle
    }

    companion object {
        fun from(
            track: PowerampTrack,
            identity: RadioSeedIdentity? = null,
        ): WidgetRadioSeedReference = WidgetRadioSeedReference(
            powerampFileId = track.realId,
            normalizedPath = track.path
                ?.let(TrackNormalization::normalizePath)
                ?.takeIf(String::isNotBlank),
            normalizedTitle = TrackNormalization.normalizeTitle(track.title),
            displayTitle = track.title.takeIf(String::isNotBlank) ?: "current track",
            queueOccurrenceId = track.queueOccurrenceId,
            embeddedTrackId = identity?.embeddedTrackId,
            stableTrackSpanId = identity?.stableTrackSpanId,
        )
    }
}

internal data class WidgetRadioStatus(
    val schemaVersion: Int = CURRENT_SCHEMA_VERSION,
    val requestId: String,
    val seed: WidgetRadioSeedReference?,
    val state: WidgetRadioRequestState,
    val message: String,
    val updatedAtEpochMs: Long,
) {
    companion object {
        const val CURRENT_SCHEMA_VERSION = 1
    }
}

internal fun interface WidgetStatusAtomicWriter {
    fun write(file: File, bytes: ByteArray)
}

internal fun interface WidgetStatusAtomicReader {
    fun read(file: File): ByteArray?
}

internal object AndroidWidgetStatusAtomicWriter : WidgetStatusAtomicWriter {
    override fun write(file: File, bytes: ByteArray) {
        file.parentFile?.mkdirs()
        val atomic = AtomicFile(file)
        val output = atomic.startWrite()
        try {
            output.write(bytes)
            atomic.finishWrite(output)
        } catch (failure: Throwable) {
            atomic.failWrite(output)
            throw failure
        }
    }
}

internal object AndroidWidgetStatusAtomicReader : WidgetStatusAtomicReader {
    override fun read(file: File): ByteArray? = try {
        AtomicFile(file).openRead().use { input -> input.readBytes() }
    } catch (_: FileNotFoundException) {
        null
    }
}

/** One atomically replaced, request-bound widget status record. */
internal class WidgetRadioStatusStore(
    rootDir: File,
    private val atomicWriter: WidgetStatusAtomicWriter = AndroidWidgetStatusAtomicWriter,
    private val atomicReader: WidgetStatusAtomicReader = AndroidWidgetStatusAtomicReader,
    private val gson: Gson = Gson(),
) {
    private val file = File(rootDir, STATUS_FILE)
    private val lock = locks.computeIfAbsent(file.absoluteFile.normalize().path) { Any() }

    fun write(status: WidgetRadioStatus): WidgetRadioStatus = synchronized(lock) {
        validate(status)
        val bytes = gson.toJson(status).toByteArray(Charsets.UTF_8)
        require(bytes.size <= MAX_STATUS_BYTES) { "Widget status is too large" }
        atomicWriter.write(file, bytes)
        status
    }

    fun read(): WidgetRadioStatus? = synchronized(lock) {
        val bytes = atomicReader.read(file) ?: return@synchronized null
        require(bytes.isNotEmpty() && bytes.size <= MAX_STATUS_BYTES) {
            "Widget status has invalid size"
        }
        val status = gson.fromJson(bytes.toString(Charsets.UTF_8), WidgetRadioStatus::class.java)
            ?: throw IllegalArgumentException("Widget status is empty")
        validate(status)
        status
    }

    /** Ignore late service updates from a request that no longer owns the widget status. */
    fun updateMatchingRequest(
        requestId: String,
        state: WidgetRadioRequestState,
        message: String,
        seed: WidgetRadioSeedReference? = null,
        updatedAtEpochMs: Long = System.currentTimeMillis(),
        preserveCurrentStates: Set<WidgetRadioRequestState> = emptySet(),
    ): Boolean = synchronized(lock) {
        val current = read() ?: return@synchronized false
        if (current.requestId != requestId) return@synchronized false
        if (current.state in preserveCurrentStates) return@synchronized true
        write(
            current.copy(
                seed = seed ?: current.seed,
                state = state,
                message = message,
                updatedAtEpochMs = updatedAtEpochMs,
            ),
        )
        true
    }

    private fun validate(status: WidgetRadioStatus) {
        require(status.schemaVersion == WidgetRadioStatus.CURRENT_SCHEMA_VERSION) {
            "Unsupported widget status schema"
        }
        require(status.requestId.matches(REQUEST_ID_REGEX)) { "Invalid widget request ID" }
        require(status.message.isNotBlank() && status.message.length <= MAX_MESSAGE_CHARS) {
            "Invalid widget status message"
        }
        require(status.updatedAtEpochMs > 0L) { "Invalid widget status timestamp" }
        status.seed?.let { seed ->
            require(seed.powerampFileId > 0L) { "Invalid widget seed file ID" }
            require(seed.displayTitle.isNotBlank() && seed.displayTitle.length <= MAX_TITLE_CHARS) {
                "Invalid widget seed title"
            }
            require(seed.normalizedTitle.isNotBlank()) { "Invalid normalized widget seed title" }
            seed.normalizedPath?.let { path ->
                require(path.isNotBlank() && path.length <= MAX_PATH_CHARS) {
                    "Invalid widget seed path"
                }
            }
            seed.embeddedTrackId?.let { require(it >= 0L) { "Invalid embedded widget seed ID" } }
            seed.queueOccurrenceId?.let { require(it > 0L) { "Invalid widget queue occurrence ID" } }
        }
        if (status.state != WidgetRadioRequestState.FAILED) {
            requireNotNull(status.seed) { "Active and terminal widget requests require a seed" }
        }
    }

    private companion object {
        const val STATUS_FILE = "widget/start-radio-status-v1.json"
        const val MAX_STATUS_BYTES = 32 * 1024
        const val MAX_MESSAGE_CHARS = 512
        const val MAX_TITLE_CHARS = 512
        const val MAX_PATH_CHARS = 4_096
        val REQUEST_ID_REGEX = Regex("[A-Za-z0-9._-]{1,128}")
        val locks = ConcurrentHashMap<String, Any>()
    }
}

internal enum class WidgetPlaybackReadiness {
    READY,
    NO_TRACK,
    REFRESH_POWERAMP,
}

internal enum class WidgetPrimaryAction {
    START_RADIO,
    OPEN_POWERAMP,
}

internal data class WidgetPlaybackSnapshot(
    val track: PowerampTrack?,
    val playbackState: Int?,
    val readiness: WidgetPlaybackReadiness,
)

/** A cached track is never fresh enough to label a home-screen widget refresh. */
internal object WidgetPlaybackTrackFreshnessPolicy {
    @Suppress("UNUSED_PARAMETER")
    fun select(
        stickyTrack: PowerampTrack?,
        cachedTrack: PowerampTrack?,
    ): PowerampTrack? = stickyTrack
}

internal object WidgetRadioPresentationPolicy {
    const val STATUS_VISIBLE_MS = 30L * 60L * 1000L
    const val SUCCESS_VISIBLE_MS = 6_000L

    /** Persisted messages are audit evidence; the widget renders only this typed listener copy. */
    fun listenerStatusText(state: WidgetRadioRequestState): String = when (state) {
        WidgetRadioRequestState.STARTING -> "Starting radio..."
        WidgetRadioRequestState.BUSY -> "Radio already starting"
        WidgetRadioRequestState.WAITING_FOR_INDEXING -> "Waiting for indexing"
        WidgetRadioRequestState.SUCCEEDED -> "Radio queued"
        WidgetRadioRequestState.PARTIAL_FAILED -> "Queue incomplete \u00b7 Open app"
        WidgetRadioRequestState.CANCELLED -> "Radio cancelled"
        WidgetRadioRequestState.FAILED -> "Radio failed \u00b7 Open app"
    }

    fun primaryAction(playback: WidgetPlaybackSnapshot): WidgetPrimaryAction =
        if (playback.readiness == WidgetPlaybackReadiness.READY && playback.track != null) {
            WidgetPrimaryAction.START_RADIO
        } else {
            WidgetPrimaryAction.OPEN_POWERAMP
        }

    fun visibleStatus(
        playback: WidgetPlaybackSnapshot,
        status: WidgetRadioStatus?,
        nowEpochMs: Long,
    ): WidgetRadioStatus? {
        if (playback.readiness != WidgetPlaybackReadiness.READY) return null
        val track = playback.track ?: return null
        val candidate = status ?: return null
        val ageMs = nowEpochMs - candidate.updatedAtEpochMs
        val visibleMs = if (candidate.state == WidgetRadioRequestState.SUCCEEDED) {
            SUCCESS_VISIBLE_MS
        } else {
            STATUS_VISIBLE_MS
        }
        if (ageMs !in 0..visibleMs) return null
        return candidate.takeIf { it.seed?.matches(track) == true }
    }
}
