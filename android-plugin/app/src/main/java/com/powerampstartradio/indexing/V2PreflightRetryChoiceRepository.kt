package com.powerampstartradio.indexing

import android.content.Context
import android.util.Log
import com.google.gson.GsonBuilder

data class V2PreflightRetryChoiceEnvelope(
    val schemaVersion: Int = V2PreflightRetryChoiceRepository.SCHEMA_VERSION,
    val spans: List<V2ProviderSpanLocator>,
)

/** Persists an explicit request to put a rejected provider span back in the ready selection. */
internal class V2PreflightRetryChoiceRepository(context: Context) {
    private val prefs = context.applicationContext.getSharedPreferences(
        V2TrackExclusionRepository.PREFERENCES_NAME,
        Context.MODE_PRIVATE,
    )
    private val gson = GsonBuilder().disableHtmlEscaping().create()

    fun load(): Set<V2ProviderSpanLocator> = synchronized(lock) {
        val json = prefs.getString(PREFERENCES_KEY, null) ?: return@synchronized emptySet()
        runCatching {
            val envelope = gson.fromJson(json, V2PreflightRetryChoiceEnvelope::class.java)
                ?: error("empty retry-choice envelope")
            require(envelope.schemaVersion == SCHEMA_VERSION)
            envelope.spans.map(::requireValid).toSet()
        }.onFailure { error ->
            Log.w(TAG, "Ignoring invalid saved preflight retry choices", error)
        }.getOrDefault(emptySet())
    }

    fun persist(spans: Collection<V2ProviderSpanLocator>) = synchronized(lock) {
        val normalized = spans.map(::requireValid).distinct().sortedWith(
            compareBy<V2ProviderSpanLocator> { it.normalizedPhysicalPath }
                .thenBy { it.offsetMs }
                .thenBy { it.durationMs },
        )
        check(
            prefs.edit().putString(
                PREFERENCES_KEY,
                gson.toJson(V2PreflightRetryChoiceEnvelope(spans = normalized)),
            ).commit(),
        ) { "Unable to persist preflight retry choices" }
    }

    private fun requireValid(span: V2ProviderSpanLocator): V2ProviderSpanLocator {
        require(span.normalizedPhysicalPath.startsWith('/'))
        require(span.offsetMs >= 0L)
        return V2ProviderSpanLocatorPolicy.canonicalize(span)
    }

    companion object {
        const val SCHEMA_VERSION = 1
        const val PREFERENCES_KEY = "v2_preflight_retry_choices"
        private const val TAG = "V2PreflightRetryChoices"
        private val lock = Any()
    }
}
