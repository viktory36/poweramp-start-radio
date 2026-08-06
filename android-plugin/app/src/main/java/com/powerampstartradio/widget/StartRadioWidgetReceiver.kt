package com.powerampstartradio.widget

import android.app.PendingIntent
import android.appwidget.AppWidgetManager
import android.appwidget.AppWidgetProvider
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.os.Handler
import android.os.Looper
import android.view.View
import android.widget.RemoteViews
import com.powerampstartradio.MainActivity
import com.powerampstartradio.R
import com.powerampstartradio.poweramp.PowerampHelper
import com.powerampstartradio.poweramp.PowerampReceiver
import com.powerampstartradio.ui.RadioConfig
import com.powerampstartradio.ui.RadioSettingsStore
import java.util.UUID

class StartRadioWidgetReceiver : AppWidgetProvider() {

    companion object {
        private val statusExpiryHandler = Handler(Looper.getMainLooper())

        internal fun persistRadioStatus(context: Context, status: WidgetRadioStatus) {
            WidgetRadioStatusStore(context.filesDir).write(status)
            runCatching { updateAllWidgets(context) }
            scheduleSuccessExpiryRefresh(context, status.requestId, status.state)
        }

        internal fun updateRadioStatus(
            context: Context,
            requestId: String,
            state: WidgetRadioRequestState,
            message: String,
            seed: WidgetRadioSeedReference? = null,
            preserveCurrentStates: Set<WidgetRadioRequestState> = emptySet(),
        ): Boolean {
            val updated = runCatching {
                WidgetRadioStatusStore(context.filesDir).updateMatchingRequest(
                    requestId = requestId,
                    state = state,
                    message = message,
                    seed = seed,
                    preserveCurrentStates = preserveCurrentStates,
                )
            }.getOrDefault(false)
            if (updated) {
                runCatching { updateAllWidgets(context) }
                scheduleSuccessExpiryRefresh(context, requestId, state)
            }
            return updated
        }

        internal fun readRadioStatus(context: Context): WidgetRadioStatus? = runCatching {
            WidgetRadioStatusStore(context.filesDir).read()
        }.getOrNull()

        private fun scheduleSuccessExpiryRefresh(
            context: Context,
            requestId: String,
            state: WidgetRadioRequestState,
        ) {
            if (state != WidgetRadioRequestState.SUCCEEDED) return
            val appContext = context.applicationContext
            statusExpiryHandler.postDelayed(
                {
                    val current = readRadioStatus(appContext)
                    if (current?.requestId == requestId &&
                        current.state == WidgetRadioRequestState.SUCCEEDED
                    ) {
                        runCatching { updateAllWidgets(appContext) }
                    }
                },
                WidgetRadioPresentationPolicy.SUCCESS_VISIBLE_MS + 100L,
            )
        }

        internal fun persistRadioFailure(
            context: Context,
            requestId: String,
            message: String,
            seed: WidgetRadioSeedReference?,
        ) {
            persistRadioStatus(
                context,
                WidgetRadioStatus(
                    requestId = requestId,
                    seed = seed,
                    state = WidgetRadioRequestState.FAILED,
                    message = message,
                    updatedAtEpochMs = System.currentTimeMillis(),
                ),
            )
        }

        fun updateAllWidgets(context: Context) {
            val manager = AppWidgetManager.getInstance(context)
            val ids = manager.getAppWidgetIds(ComponentName(context, StartRadioWidgetReceiver::class.java))
            if (ids.isNotEmpty()) {
                updateWidgets(context, manager, ids)
            }
        }

        private fun updateWidgets(
            context: Context,
            manager: AppWidgetManager,
            appWidgetIds: IntArray
        ) {
            val playback = PowerampReceiver.getWidgetPlaybackSnapshot(context)
            val track = playback.track
            val title = track?.title?.takeIf { it.isNotBlank() } ?: when (playback.readiness) {
                WidgetPlaybackReadiness.REFRESH_POWERAMP ->
                    context.getString(R.string.widget_playback_unavailable)
                else -> context.getString(R.string.widget_no_track_playing)
            }
            val trackDetails = listOfNotNull(
                track?.artist?.takeIf { !it.isNullOrBlank() },
                track?.album?.takeIf { !it.isNullOrBlank() }
            ).joinToString(" · ")
            val playbackSubtitle = when (playback.readiness) {
                WidgetPlaybackReadiness.NO_TRACK ->
                    context.getString(R.string.widget_play_track_in_poweramp)
                WidgetPlaybackReadiness.REFRESH_POWERAMP ->
                    context.getString(R.string.widget_refresh_poweramp_playback)
                WidgetPlaybackReadiness.READY -> trackDetails
            }
            val visibleStatus = WidgetRadioPresentationPolicy.visibleStatus(
                playback = playback,
                status = readRadioStatus(context),
                nowEpochMs = System.currentTimeMillis(),
            )
            val subtitle = visibleStatus?.let {
                WidgetRadioPresentationPolicy.listenerStatusText(it.state)
            } ?: playbackSubtitle
            val commandReady = playback.readiness == WidgetPlaybackReadiness.READY && track != null
            val primaryAction = WidgetRadioPresentationPolicy.primaryAction(playback)
            val busy = visibleStatus?.state in setOf(
                WidgetRadioRequestState.STARTING,
                WidgetRadioRequestState.BUSY,
                WidgetRadioRequestState.WAITING_FOR_INDEXING,
            )

            val views = RemoteViews(context.packageName, R.layout.widget_start_radio).apply {
                setTextViewText(R.id.widget_track_title, title)
                setContentDescription(
                    R.id.widget_root,
                    if (track == null) {
                        context.getString(R.string.widget_open_app_action)
                    } else {
                        context.getString(R.string.widget_open_app_for_track_action, title)
                    },
                )
                setContentDescription(
                    R.id.widget_start_button,
                    if (commandReady) {
                        context.getString(R.string.widget_start_radio_for_track_action, title)
                    } else if (playback.readiness == WidgetPlaybackReadiness.REFRESH_POWERAMP) {
                        context.getString(R.string.widget_refresh_poweramp_action)
                    } else {
                        context.getString(R.string.widget_no_track_action)
                    },
                )
                setFloat(R.id.widget_start_button, "setAlpha", if (busy) 0.67f else 1f)
                setImageViewResource(
                    R.id.widget_action_icon,
                    if (primaryAction == WidgetPrimaryAction.START_RADIO) {
                        R.drawable.ic_radio_waves
                    } else {
                        android.R.drawable.ic_media_play
                    },
                )
                if (subtitle.isNotBlank()) {
                    setViewVisibility(R.id.widget_track_subtitle, View.VISIBLE)
                    setTextViewText(R.id.widget_track_subtitle, subtitle)
                } else {
                    setViewVisibility(R.id.widget_track_subtitle, View.GONE)
                    setTextViewText(R.id.widget_track_subtitle, "")
                }
                setOnClickPendingIntent(R.id.widget_root, openAppPendingIntent(context))
                setOnClickPendingIntent(
                    R.id.widget_start_button,
                    if (primaryAction == WidgetPrimaryAction.START_RADIO) {
                        val busySeed = visibleStatus?.seed.takeIf { busy }
                        startRadioPendingIntent(
                            context,
                            busySeed ?: WidgetRadioSeedReference.from(checkNotNull(track)),
                            visibleStatus?.requestId?.takeIf { busy }
                                ?: UUID.randomUUID().toString(),
                        )
                    } else {
                        openPowerampPendingIntent(context)
                    },
                )
            }

            appWidgetIds.forEach { manager.updateAppWidget(it, views) }
        }

        private fun startRadioPendingIntent(
            context: Context,
            expectedSeed: WidgetRadioSeedReference,
            commandId: String,
        ): PendingIntent {
            return PendingIntent.getBroadcast(
                context,
                0,
                StartRadioWidgetActionReceiver.intent(
                    context,
                    commandId,
                    expectedSeed,
                ),
                PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
            )
        }

        private fun openAppPendingIntent(context: Context): PendingIntent {
            val intent = Intent(context, MainActivity::class.java).apply {
                flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP
            }
            return PendingIntent.getActivity(
                context,
                1,
                intent,
                PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
            )
        }

        private fun openPowerampPendingIntent(context: Context): PendingIntent {
            val intent = (context.packageManager
                .getLaunchIntentForPackage(PowerampHelper.POWERAMP_PACKAGE)
                ?: Intent(context, MainActivity::class.java)).apply {
                flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP
            }
            return PendingIntent.getActivity(
                context,
                2,
                intent,
                PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
            )
        }

        internal fun buildRadioConfig(context: Context): RadioConfig {
            return RadioSettingsStore.from(context).readSnapshot().requestConfig
        }
    }

    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action == Intent.ACTION_MY_PACKAGE_REPLACED) {
            updateAllWidgets(context)
            return
        }
        super.onReceive(context, intent)
    }

    override fun onUpdate(context: Context, appWidgetManager: AppWidgetManager, appWidgetIds: IntArray) {
        updateWidgets(context, appWidgetManager, appWidgetIds)
    }

    override fun onEnabled(context: Context) {
        updateAllWidgets(context)
    }
}
