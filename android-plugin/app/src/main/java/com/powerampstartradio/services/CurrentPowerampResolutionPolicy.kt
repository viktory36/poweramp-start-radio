package com.powerampstartradio.services

/** Prevents a durable direct-queue request from trusting a reused Poweramp folder_files ID. */
internal object CurrentPowerampResolutionPolicy {
    fun requireUnchanged(
        pinnedFileIds: List<Long?>,
        currentFileIds: List<Long?>,
    ): List<Long?> {
        require(currentFileIds.size == pinnedFileIds.size) {
            "Current Poweramp resolution did not align with the requested recordings"
        }
        currentFileIds.forEachIndexed { index, current ->
            require(current == pinnedFileIds[index]) {
                "Poweramp recording ${index + 1} changed since this queue request was saved; retry it"
            }
        }
        return currentFileIds.toList()
    }
}
