package com.powerampstartradio.widget

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.util.Log
import android.widget.Toast
import com.powerampstartradio.services.RadioService
import com.powerampstartradio.services.WidgetRadioStartDisposition
import com.powerampstartradio.services.WidgetRadioStartResult
import java.util.UUID

/** Private receiver behind the widget's creator-owned PendingIntent. */
class StartRadioWidgetActionReceiver : BroadcastReceiver() {
    companion object {
        private const val ACTION_START_RADIO =
            "com.powerampstartradio.widget.ACTION_START_RADIO"
        private const val EXTRA_COMMAND_ID = "widget_command_id"
        private const val EXTRA_EXPECTED_FILE_ID = "widget_expected_file_id"
        private const val EXTRA_EXPECTED_PATH = "widget_expected_path"
        private const val EXTRA_EXPECTED_TITLE = "widget_expected_title"
        private const val EXTRA_EXPECTED_DISPLAY_TITLE = "widget_expected_display_title"
        private const val EXTRA_EXPECTED_QUEUE_ID = "widget_expected_queue_id"

        internal fun intent(
            context: Context,
            commandId: String,
            expectedSeed: WidgetRadioSeedReference,
        ): Intent =
            Intent(context, StartRadioWidgetActionReceiver::class.java).apply {
                action = ACTION_START_RADIO
                data = Uri.parse("poweramp-start-radio-v2://widget-radio/$commandId")
                putExtra(EXTRA_COMMAND_ID, commandId)
                putExtra(EXTRA_EXPECTED_FILE_ID, expectedSeed.powerampFileId)
                putExtra(EXTRA_EXPECTED_PATH, expectedSeed.normalizedPath)
                putExtra(EXTRA_EXPECTED_TITLE, expectedSeed.normalizedTitle)
                putExtra(EXTRA_EXPECTED_DISPLAY_TITLE, expectedSeed.displayTitle)
                putExtra(EXTRA_EXPECTED_QUEUE_ID, expectedSeed.queueOccurrenceId ?: -1L)
            }

        private fun expectedSeed(intent: Intent): WidgetRadioSeedReference? {
            val fileId = intent.getLongExtra(EXTRA_EXPECTED_FILE_ID, -1L)
            val title = intent.getStringExtra(EXTRA_EXPECTED_TITLE).orEmpty()
            val displayTitle = intent.getStringExtra(EXTRA_EXPECTED_DISPLAY_TITLE).orEmpty()
            if (fileId <= 0L || title.isBlank() || displayTitle.isBlank()) return null
            return WidgetRadioSeedReference(
                powerampFileId = fileId,
                normalizedPath = intent.getStringExtra(EXTRA_EXPECTED_PATH)?.takeIf(String::isNotBlank),
                normalizedTitle = title,
                displayTitle = displayTitle,
                queueOccurrenceId = intent.getLongExtra(EXTRA_EXPECTED_QUEUE_ID, -1L)
                    .takeIf { it > 0L },
            )
        }
    }

    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action != ACTION_START_RADIO) return

        val commandId = intent.getStringExtra(EXTRA_COMMAND_ID)
            ?.takeIf { runCatching { UUID.fromString(it) }.isSuccess }
        if (commandId == null) {
            Toast.makeText(
                context,
                "Widget is out of date. Wait a moment, then try again.",
                Toast.LENGTH_LONG,
            )
                .show()
            StartRadioWidgetReceiver.updateAllWidgets(context)
            return
        }
        val expectedSeed = expectedSeed(intent)
        if (expectedSeed == null) {
            Toast.makeText(
                context,
                "Widget track is out of date. Wait a moment, then try again.",
                Toast.LENGTH_LONG,
            )
                .show()
            StartRadioWidgetReceiver.updateAllWidgets(context)
            return
        }
        val result = runCatching {
            RadioService.startRadioFromWidgetTap(
                context = context,
                commandId = commandId,
                config = StartRadioWidgetReceiver.buildRadioConfig(context),
                expectedDisplayedSeed = expectedSeed,
            )
        }.getOrElse { failure ->
            Log.e("StartRadioWidget", "Widget radio submission failed", failure)
            val message = "Radio could not start. Open Start Radio, then try again."
            WidgetRadioStartResult(WidgetRadioStartDisposition.FAILED, commandId, message)
        }
        if (result.disposition != WidgetRadioStartDisposition.STARTED) {
            Toast.makeText(
                context,
                result.message,
                Toast.LENGTH_LONG,
            ).show()
        }
    }
}
