package com.powerampstartradio.services

internal class RadioSeedNotIndexedException : IllegalStateException(
    "No indexed match for this song. Index it in On-device indexing or merge its server index.",
)

internal object RadioFailureMessage {
    fun from(failure: Exception): String {
        if (failure is RadioSeedNotIndexedException) return requireNotNull(failure.message)
        val detail = failure.message?.trim().orEmpty()
        return when {
            failure is SecurityException ||
                detail.contains("permission", ignoreCase = true) ||
                detail.contains("access", ignoreCase = true) ->
                "Poweramp access is unavailable. Open Start Radio and grant access."
            detail.contains("authenticated Poweramp", ignoreCase = true) ||
                detail.contains("playback no longer matches", ignoreCase = true) ->
                "Poweramp playback changed. Return to Poweramp, then tap Start Radio again."
            detail.contains("No track", ignoreCase = true) ->
                "No Poweramp track is available. Start a track, then try again."
            else -> "Radio could not start. Open Start Radio, then try again."
        }
    }
}
