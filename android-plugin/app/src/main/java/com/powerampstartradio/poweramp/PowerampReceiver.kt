package com.powerampstartradio.poweramp

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.os.Build
import android.util.Log
import com.powerampstartradio.widget.StartRadioWidgetReceiver
import com.powerampstartradio.widget.WidgetPlaybackReadiness
import com.powerampstartradio.widget.WidgetPlaybackSnapshot
import com.powerampstartradio.widget.WidgetPlaybackTrackFreshnessPolicy
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.Executors

/**
 * Receives track change broadcasts from Poweramp.
 *
 * This is used to track the currently playing song so we know which track
 * to use as the seed when the user triggers "Start Radio".
 */
class PowerampReceiver : BroadcastReceiver() {

    companion object {
        private const val TAG = "PowerampReceiver"
        private const val PREFS_NAME = "poweramp_state"
        private const val KEY_REAL_ID = "current_track_real_id"
        private const val KEY_TITLE = "current_track_title"
        private const val KEY_ARTIST = "current_track_artist"
        private const val KEY_ALBUM = "current_track_album"
        private const val KEY_DURATION_MS = "current_track_duration_ms"
        private const val KEY_PATH = "current_track_path"
        private const val KEY_TRACK_ID = "current_track_category_row_id"
        private const val KEY_CATEGORY_URI = "current_track_category_uri"
        private const val KEY_POSITION_IN_LIST = "current_track_position_in_list"
        private const val AUTHENTICATED_PREFS_NAME = "poweramp_authenticated_state_v2"
        private const val AUTHENTICATED_STATE_KEY = "authenticated_explicit_state"

        // Singleton to hold current track state
        @Volatile
        var currentTrack: PowerampTrack? = null
            private set

        @Volatile
        var isPlaying: Boolean = false
            private set

        @Volatile
        var playbackState: Int? = null
            private set

        @Volatile
        private var authenticatedState: PowerampAuthenticatedState? = null

        @Volatile
        private var authenticatedStateLoaded = false

        private val authenticatedStateLock = Any()
        private val powerampRefreshExecutor = Executors.newSingleThreadExecutor { task ->
            Thread(task, "poweramp-event-refresh").apply { isDaemon = true }
        }

        // Broadcasts and UI lifecycle callbacks can arrive on different threads.
        private val trackChangeListeners = CopyOnWriteArrayList<(PowerampTrack?) -> Unit>()

        fun addTrackChangeListener(listener: (PowerampTrack?) -> Unit) {
            trackChangeListeners.add(listener)
        }

        fun removeTrackChangeListener(listener: (PowerampTrack?) -> Unit) {
            trackChangeListeners.remove(listener)
        }

        fun getCurrentTrack(context: Context): PowerampTrack? = refreshDisplayFromSticky(context)

        /** Refresh display state from the same sticky API used by V1. */
        fun refreshCurrentTrackAfterActivityResume(context: Context): PowerampTrack? =
            refreshDisplayFromSticky(context)

        /**
         * Read presentation state from Poweramp's sticky API on every widget refresh. Sticky
         * extras are display-only. On API 34+, command readiness comes either from a matching
         * sender-authenticated event or from an exact provider row (and Queue occurrence when
         * applicable) verified between stable sticky reads.
         */
        internal fun getWidgetPlaybackSnapshot(context: Context): WidgetPlaybackSnapshot {
            val authenticated = loadAuthenticatedState(context)
            return runCatching {
                val stickyState = PowerampHelper.getStickyPlaybackState(context)
                playbackState = stickyState
                isPlaying = stickyState == PowerampHelper.STATE_PLAYING

                val stickyTrack = PowerampHelper.getStickyCurrentTrack(context)
                if (stickyTrack != null) publishDisplayTrack(context, stickyTrack)
                val presentationTrack = WidgetPlaybackTrackFreshnessPolicy.select(
                    stickyTrack = stickyTrack,
                    cachedTrack = authenticated?.track ?: currentTrack,
                )

                val verifiedTrack = when {
                    stickyTrack == null -> null
                    Build.VERSION.SDK_INT < Build.VERSION_CODES.UPSIDE_DOWN_CAKE -> stickyTrack
                    authenticated?.origin == PowerampAuthenticatedStateOrigin.LIVE_EXPLICIT &&
                        PowerampCurrentTrackIdentityPolicy.authenticatedStateMatchesSticky(
                            authenticated = authenticated,
                            stickyTrack = stickyTrack,
                            stickyPlaybackState = stickyState,
                        ) -> stickyTrack
                    else -> runCatching {
                        requireTargetedProviderVerifiedStickyTrack(
                            context = context,
                            before = PowerampLegacyStickyCandidate(
                                stickyTrack,
                                stickyState ?: PowerampHelper.STATE_STOPPED,
                            ),
                        )
                    }.onFailure { error ->
                        Log.w(TAG, "Poweramp widget track could not be provider-verified", error)
                    }.getOrNull()
                }
                WidgetPlaybackSnapshot(
                    track = verifiedTrack ?: presentationTrack,
                    playbackState = stickyState,
                    readiness = if (verifiedTrack != null) {
                        WidgetPlaybackReadiness.READY
                    } else {
                        WidgetPlaybackReadiness.REFRESH_POWERAMP
                    },
                )
            }.onFailure { error ->
                Log.w(TAG, "Unable to read Poweramp widget presentation state", error)
            }.getOrElse {
                playbackState = null
                isPlaying = false
                WidgetPlaybackSnapshot(
                    track = WidgetPlaybackTrackFreshnessPolicy.select(
                        stickyTrack = null,
                        cachedTrack = currentTrack ?: authenticated?.track,
                    ),
                    playbackState = null,
                    readiness = WidgetPlaybackReadiness.REFRESH_POWERAMP,
                )
            }
        }

        /** Resolve the same V1-compatible track shown in the UI against its exact provider row. */
        fun requireProviderVerifiedCurrentTrack(context: Context): PowerampTrack? {
            val before = readLegacyStickyCandidate(context)
            val resolved = requireTargetedProviderVerifiedStickyTrack(context, before)
            if (resolved != null) publishDisplayTrack(context, resolved)
            return resolved
        }

        private fun requireTargetedProviderVerifiedStickyTrack(
            context: Context,
            before: PowerampLegacyStickyCandidate,
            authenticated: PowerampAuthenticatedState? = null,
        ): PowerampTrack? {
            val providerEntries = before.track?.let { track ->
                listOf(PowerampHelper.requireFileEntryById(context, track.realId))
            }.orEmpty()
            val queueEntries = before.track
                ?.takeIf { PowerampQueueOccurrencePolicy.isQueueCategory(it.categoryUri) }
                ?.let {
                    runCatching { PowerampHelper.requireCompleteQueueSnapshot(context) }
                        .onFailure { error ->
                            Log.w(
                                TAG,
                                "Current track is usable, but its Queue occurrence could not be verified",
                                error,
                            )
                        }
                        .getOrNull()
                }
            val resolved = PowerampCommandTrackPolicy.requireLegacyProviderBacked(
                candidate = before,
                providerEntries = providerEntries,
                queueEntries = queueEntries,
            )
            val after = readLegacyStickyCandidate(context)
            check(PowerampLegacyStickyCandidatePolicy.unchanged(before, after)) {
                "Poweramp playback changed while its widget track was being verified"
            }
            authenticated?.let { captured ->
                check(authenticatedState == captured) {
                    "Authenticated Poweramp playback changed while its track was being verified"
                }
            }
            return resolved
        }

        private fun requireStickyRevalidation(context: Context) {
            val captured = loadAuthenticatedState(context)
            synchronized(authenticatedStateLock) {
                if (authenticatedState == captured) {
                    authenticatedState = PowerampActivityResumePolicy
                        .requireStickyRevalidation(captured)
                }
            }
        }

        private fun readLegacyStickyCandidate(context: Context): PowerampLegacyStickyCandidate {
            val state = PowerampHelper.getStickyPlaybackState(context)
            return PowerampLegacyStickyCandidatePolicy.fromSticky(
                stickyTrack = PowerampHelper.getStickyCurrentTrack(context),
                stickyPlaybackState = state,
                fallbackTrack = currentTrack ?: loadPersistedDisplayTrack(context),
            )
        }

        private fun refreshDisplayFromSticky(context: Context): PowerampTrack? {
            return runCatching {
                val stickyState = PowerampHelper.getStickyPlaybackState(context)
                playbackState = stickyState
                isPlaying = stickyState == PowerampHelper.STATE_PLAYING
                val stickyTrack = PowerampHelper.getStickyCurrentTrack(context)
                val displayTrack = stickyTrack ?: currentTrack ?: loadPersistedDisplayTrack(context)
                if (displayTrack != null) publishDisplayTrack(context, displayTrack)
                displayTrack
            }.onFailure { error ->
                Log.w(TAG, "Unable to read display-only Poweramp sticky state", error)
                playbackState = null
                isPlaying = false
            }.getOrNull()
        }

        private fun loadPersistedDisplayTrack(context: Context): PowerampTrack? {
            val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            val title = prefs.getString(KEY_TITLE, null) ?: return null
            val realId = prefs.getLong(KEY_REAL_ID, -1L).takeIf { it > 0L } ?: return null
            return PowerampTrack(
                realId = realId,
                title = title,
                artist = prefs.getString(KEY_ARTIST, null),
                album = prefs.getString(KEY_ALBUM, null),
                durationMs = prefs.getInt(KEY_DURATION_MS, 0),
                path = prefs.getString(KEY_PATH, null),
                trackId = prefs.getLong(KEY_TRACK_ID, -1L),
                categoryUri = prefs.getString(KEY_CATEGORY_URI, null),
                positionInList = if (prefs.contains(KEY_POSITION_IN_LIST)) {
                    prefs.getInt(KEY_POSITION_IN_LIST, 0)
                } else {
                    null
                },
            )
        }

        private fun publishDisplayTrack(context: Context, track: PowerampTrack?) {
            val changed = currentTrack != track
            currentTrack = track
            persistCurrentTrack(context, track)
            if (changed) notifyTrackChanged(track)
        }

        private fun applyPlaybackPresentation(
            context: Context,
            state: PowerampAuthenticatedState,
        ) {
            playbackState = state.playbackState
            isPlaying = state.playbackState == PowerampHelper.STATE_PLAYING
            if (state.track != null) {
                publishDisplayTrack(context, state.track)
            } else {
                refreshDisplayFromSticky(context)
            }
        }

        private fun loadAuthenticatedState(context: Context): PowerampAuthenticatedState? {
            if (authenticatedStateLoaded) return authenticatedState
            return synchronized(authenticatedStateLock) {
                if (!authenticatedStateLoaded) {
                    val json = context.getSharedPreferences(
                        AUTHENTICATED_PREFS_NAME,
                        Context.MODE_PRIVATE,
                    ).getString(AUTHENTICATED_STATE_KEY, null)
                    authenticatedState = json?.let { stored ->
                        runCatching { PowerampAuthenticatedStateCodec.decode(stored) }
                            .onFailure { error ->
                                Log.w(TAG, "Discarding invalid authenticated Poweramp evidence", error)
                            }
                            .getOrNull()
                    }
                    authenticatedStateLoaded = true
                }
                authenticatedState
            }
        }

        private fun publishAuthenticatedState(
            context: Context,
            state: PowerampAuthenticatedState,
        ) {
            val encoded = PowerampAuthenticatedStateCodec.encode(state)
            synchronized(authenticatedStateLock) {
                authenticatedState = state
                authenticatedStateLoaded = true
                context.getSharedPreferences(AUTHENTICATED_PREFS_NAME, Context.MODE_PRIVATE)
                    .edit()
                    .putString(AUTHENTICATED_STATE_KEY, encoded)
                    .apply()
            }
            if (state.origin == PowerampAuthenticatedStateOrigin.LIVE_EXPLICIT) {
                applyPlaybackPresentation(context, state)
            } else {
                // A status event authenticated only playback state. Do not display the retained
                // process-restart track without reading current display state from Poweramp.
                refreshDisplayFromSticky(context)
            }
        }

        private fun persistCurrentTrack(context: Context, track: PowerampTrack?) {
            val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            prefs.edit().apply {
                if (track == null) {
                    remove(KEY_REAL_ID)
                    remove(KEY_TITLE)
                    remove(KEY_ARTIST)
                    remove(KEY_ALBUM)
                    remove(KEY_DURATION_MS)
                    remove(KEY_PATH)
                    remove(KEY_TRACK_ID)
                    remove(KEY_CATEGORY_URI)
                    remove(KEY_POSITION_IN_LIST)
                } else {
                    putLong(KEY_REAL_ID, track.realId)
                    putString(KEY_TITLE, track.title)
                    putString(KEY_ARTIST, track.artist)
                    putString(KEY_ALBUM, track.album)
                    putInt(KEY_DURATION_MS, track.durationMs)
                    putString(KEY_PATH, track.path)
                    putLong(KEY_TRACK_ID, track.trackId)
                    putString(KEY_CATEGORY_URI, track.categoryUri)
                    if (track.positionInList == null) {
                        remove(KEY_POSITION_IN_LIST)
                    } else {
                        putInt(KEY_POSITION_IN_LIST, track.positionInList)
                    }
                }
            }.apply()
        }

        private fun notifyTrackChanged(track: PowerampTrack?) {
            trackChangeListeners.forEach { it(track) }
        }
    }

    override fun onReceive(context: Context, intent: Intent) {
        when (senderDisposition(context)) {
            PowerampBroadcastDisposition.REJECT -> {
                val uid = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
                    sentFromUid
                } else {
                    -1
                }
                val packages = context.packageManager.getPackagesForUid(uid)
                    ?.joinToString(prefix = "[", postfix = "]")
                    ?: "[]"
                val senderPackage =
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
                        sentFromPackage
                    } else {
                        null
                    }
                Log.w(
                    TAG,
                    "Ignoring ${intent.action} from untrusted uid=$uid " +
                        "package=$senderPackage uidPackages=$packages",
                )
                return
            }
            PowerampBroadcastDisposition.REFRESH_HINT_ONLY -> {
                val pending = goAsync()
                powerampRefreshExecutor.execute {
                    try {
                        refreshDisplayFromSticky(context)
                        StartRadioWidgetReceiver.updateAllWidgets(context)
                    } finally {
                        pending.finish()
                    }
                }
                return
            }
            PowerampBroadcastDisposition.AUTHENTICATED_EXPLICIT -> Unit
        }

        val previous = loadAuthenticatedState(context)
        val timestamp = intent.takeIf { it.hasExtra(PowerampHelper.EXTRA_TIMESTAMP) }
            ?.getLongExtra(PowerampHelper.EXTRA_TIMESTAMP, -1L)
            ?.takeIf { it >= 0L }
        val next = runCatching {
            when (intent.action) {
                PowerampHelper.ACTION_TRACK_CHANGED_EXPLICIT -> {
                    val track = requireNotNull(PowerampHelper.getCurrentTrackFromIntent(intent)) {
                        "Authenticated Poweramp track event omitted its track identity"
                    }
                    PowerampExplicitEventStateMachine.trackChanged(previous, track, timestamp)
                }
                PowerampHelper.ACTION_STATUS_CHANGED_EXPLICIT -> {
                    require(intent.hasExtra(PowerampHelper.EXTRA_STATE)) {
                        "Authenticated Poweramp status event omitted playback state"
                    }
                    val state = intent.getIntExtra(PowerampHelper.EXTRA_STATE, Int.MIN_VALUE)
                    require(state in setOf(
                        PowerampHelper.STATE_STOPPED,
                        PowerampHelper.STATE_PLAYING,
                        PowerampHelper.STATE_PAUSED,
                    )) { "Authenticated Poweramp status event has an invalid playback state" }
                    PowerampExplicitEventStateMachine.statusChanged(
                        previous = previous,
                        playbackState = state,
                        eventTrack = PowerampHelper.getCurrentTrackFromIntent(intent),
                        eventTimestampMs = timestamp,
                    )
                }
                else -> return
            }
        }.onFailure { error ->
            Log.w(TAG, "Ignoring invalid authenticated Poweramp event", error)
        }.getOrNull()

        if (next != null) {
            runCatching { publishAuthenticatedState(context, next) }
                .onSuccess {
                    Log.d(
                        TAG,
                        "Authenticated Poweramp state: ${next.track?.title}, " +
                            "state=${next.playbackState}",
                    )
                }
                .onFailure { error ->
                    Log.w(TAG, "Authenticated Poweramp event failed validation", error)
                }
        }
        StartRadioWidgetReceiver.updateAllWidgets(context)
    }

    /**
     * Android 14 exposes broadcast sender identity. Older releases cannot authenticate this
     * unpermissioned third-party broadcast, so their receiver path treats it only as a refresh
     * hint and still reads the actual sticky Poweramp state instead of trusting its extras.
     */
    private fun senderDisposition(context: Context): PowerampBroadcastDisposition {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            return PowerampBroadcastDisposition.REFRESH_HINT_ONLY
        }
        val uid = sentFromUid
        val packages = context.packageManager.getPackagesForUid(uid)?.toSet().orEmpty()
        return PowerampBroadcastTrust.classify(
            sdkInt = Build.VERSION.SDK_INT,
            senderUid = uid,
            uidPackages = packages,
            senderPackage = sentFromPackage,
        )
    }
}

internal object PowerampBroadcastTrust {
    fun classify(
        sdkInt: Int,
        senderUid: Int,
        uidPackages: Set<String>,
        senderPackage: String?,
    ): PowerampBroadcastDisposition {
        if (sdkInt < Build.VERSION_CODES.UPSIDE_DOWN_CAKE || senderUid < 0) {
            return PowerampBroadcastDisposition.REFRESH_HINT_ONLY
        }
        if (senderUid < 0 || PowerampHelper.POWERAMP_PACKAGE !in uidPackages) {
            return PowerampBroadcastDisposition.REJECT
        }
        return if (senderPackage == null || senderPackage == PowerampHelper.POWERAMP_PACKAGE) {
            PowerampBroadcastDisposition.AUTHENTICATED_EXPLICIT
        } else {
            PowerampBroadcastDisposition.REJECT
        }
    }

    fun isTrusted(
        sdkInt: Int,
        senderUid: Int,
        uidPackages: Set<String>,
        senderPackage: String?,
    ): Boolean = classify(sdkInt, senderUid, uidPackages, senderPackage) !=
        PowerampBroadcastDisposition.REJECT
}
