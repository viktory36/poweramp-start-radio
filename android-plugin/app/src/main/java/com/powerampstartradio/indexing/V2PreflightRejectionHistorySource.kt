package com.powerampstartradio.indexing

import com.powerampstartradio.indexing.v2.AtomicV2IndexingPreflightIntentStore
import com.powerampstartradio.indexing.v2.V2IndexingPreflightIntent
import com.powerampstartradio.indexing.v2.V2IndexingPreflightIntentInspection
import java.io.File

/** Replaceable read boundary; the reducer does not depend on how retained evidence is compacted. */
internal interface V2PreflightRejectionHistorySource {
    fun inspect(): V2IndexingPreflightIntentInspection

    fun load(): List<V2IndexingPreflightIntent> = inspect().requireComplete()
}

internal class V2AtomicPreflightRejectionHistorySource(
    filesDir: File,
) : V2PreflightRejectionHistorySource {
    private val store = AtomicV2IndexingPreflightIntentStore(
        File(filesDir, "indexing_v2/preflight-intents"),
    )

    override fun inspect(): V2IndexingPreflightIntentInspection = store.inspect()
}
