package com.powerampstartradio.indexing

/** Stable listener copy for operation failures; exact exceptions belong in Logcat. */
internal enum class IndexingListenerFailureOperation {
    NEW_TRACK_SCAN,
    CLEANUP_SCAN,
    CLEANUP_UPDATE,
    EXPORT,
    INDEXING_REQUEST,
}

internal fun indexingListenerFailureText(
    operation: IndexingListenerFailureOperation,
    indexingRequestIsDurable: Boolean = false,
): String = when (operation) {
    IndexingListenerFailureOperation.NEW_TRACK_SCAN ->
        "Poweramp tracks could not be compared with indexed source spans. " +
            "No indexing status was changed."
    IndexingListenerFailureOperation.CLEANUP_SCAN ->
        "Indexed source spans could not be compared with the Poweramp library. Nothing was removed."
    IndexingListenerFailureOperation.CLEANUP_UPDATE ->
        "The selected tracks could not be removed from the music index. " +
            "No indexed tracks were changed."
    IndexingListenerFailureOperation.EXPORT ->
        "The app files could not be exported."
    IndexingListenerFailureOperation.INDEXING_REQUEST -> if (indexingRequestIsDurable) {
        "Indexing could not start. The saved request is available in On-device indexing."
    } else {
        "Indexing could not start, and no saved request could be confirmed. " +
            "Reopen On-device indexing to check the current state."
    }
}
