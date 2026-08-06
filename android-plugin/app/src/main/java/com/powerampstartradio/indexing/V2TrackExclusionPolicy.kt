package com.powerampstartradio.indexing

import com.powerampstartradio.indexing.v2.V2ProviderDurationEvidencePolicy

data class V2ProviderSpanLocator(
    val normalizedPhysicalPath: String,
    val offsetMs: Long,
    val durationMs: Long,
)

/** Poweramp may use any non-positive duration as the same unknown-duration sentinel. */
internal object V2ProviderSpanLocatorPolicy {
    fun create(normalizedPhysicalPath: String, offsetMs: Long, durationMs: Long) =
        V2ProviderSpanLocator(
            normalizedPhysicalPath = normalizedPhysicalPath,
            offsetMs = offsetMs,
            durationMs = canonicalDurationMs(durationMs),
        )

    fun canonicalize(span: V2ProviderSpanLocator): V2ProviderSpanLocator =
        if (span.durationMs >= 0L) span else span.copy(durationMs = 0L)

    fun canonicalDurationMs(durationMs: Long): Long =
        V2ProviderDurationEvidencePolicy.canonicalMs(durationMs)
}

data class V2TrackExclusionCandidate(
    val powerampFileId: Long,
    val providerSpan: V2ProviderSpanLocator,
    val stableTrackSpanId: String? = null,
)

data class V2PersistedTrackExclusion(
    val providerSpan: V2ProviderSpanLocator,
    val stableTrackSpanId: String?,
    val lastKnownPowerampFileId: Long,
)

data class V2TrackExclusionEnvelope(
    val schemaVersion: Int = V2TrackExclusionPolicy.SCHEMA_VERSION,
    val entries: List<V2PersistedTrackExclusion>,
)

/** Numeric Poweramp IDs are locators only and can never establish exclusion identity. */
internal object V2TrackExclusionPolicy {
    const val SCHEMA_VERSION = 1

    fun resolve(
        exclusions: Collection<V2PersistedTrackExclusion>,
        candidates: Collection<V2TrackExclusionCandidate>,
    ): Set<Long> = candidates.asSequence()
        .map(V2TrackExclusionPolicy::canonicalize)
        .filter { candidate -> exclusions.any { canonicalize(it).matches(candidate) } }
        .mapTo(linkedSetOf()) { it.powerampFileId }

    fun add(
        exclusions: Collection<V2PersistedTrackExclusion>,
        candidates: Collection<V2TrackExclusionCandidate>,
    ): List<V2PersistedTrackExclusion> = deduplicate(
        exclusions.map(::canonicalize) + candidates.map { candidate ->
            val canonical = canonicalize(candidate)
            V2PersistedTrackExclusion(
                providerSpan = canonical.providerSpan,
                stableTrackSpanId = canonical.stableTrackSpanId,
                lastKnownPowerampFileId = canonical.powerampFileId,
            )
        },
    )

    fun remove(
        exclusions: Collection<V2PersistedTrackExclusion>,
        candidates: Collection<V2TrackExclusionCandidate>,
    ): List<V2PersistedTrackExclusion> = exclusions.map(::canonicalize).filterNot { exclusion ->
        candidates.any { candidate -> exclusion.matches(canonicalize(candidate)) }
    }

    fun refreshLocators(
        exclusions: Collection<V2PersistedTrackExclusion>,
        candidates: Collection<V2TrackExclusionCandidate>,
    ): List<V2PersistedTrackExclusion> = deduplicate(exclusions.map { rawExclusion ->
        val exclusion = canonicalize(rawExclusion)
        val current = candidates.asSequence().map(::canonicalize)
            .firstOrNull { candidate -> exclusion.matches(candidate) }
        if (current == null) exclusion else exclusion.copy(
            stableTrackSpanId = exclusion.stableTrackSpanId ?: current.stableTrackSpanId,
            lastKnownPowerampFileId = current.powerampFileId,
        )
    })

    fun requireValid(envelope: V2TrackExclusionEnvelope): V2TrackExclusionEnvelope {
        require(envelope.schemaVersion == SCHEMA_VERSION) {
            "unsupported track exclusion schema ${envelope.schemaVersion}"
        }
        val canonicalEntries = envelope.entries.map(::canonicalize)
        canonicalEntries.forEach { entry ->
            require(entry.lastKnownPowerampFileId > 0L)
            require(entry.providerSpan.normalizedPhysicalPath.startsWith('/'))
            require(entry.providerSpan.offsetMs >= 0L)
            require(entry.providerSpan.durationMs >= 0L)
            entry.stableTrackSpanId?.let { stableId ->
                require(stableId.matches(STABLE_SPAN_ID)) { "invalid stable track span ID" }
            }
        }
        return envelope.copy(entries = deduplicate(canonicalEntries))
    }

    private fun V2PersistedTrackExclusion.matches(candidate: V2TrackExclusionCandidate): Boolean {
        if (stableTrackSpanId != null && candidate.stableTrackSpanId != null) {
            return stableTrackSpanId == candidate.stableTrackSpanId
        }
        return providerSpan == candidate.providerSpan
    }

    private fun canonicalize(value: V2TrackExclusionCandidate): V2TrackExclusionCandidate =
        value.copy(providerSpan = V2ProviderSpanLocatorPolicy.canonicalize(value.providerSpan))

    private fun canonicalize(value: V2PersistedTrackExclusion): V2PersistedTrackExclusion =
        value.copy(providerSpan = V2ProviderSpanLocatorPolicy.canonicalize(value.providerSpan))

    private fun deduplicate(
        values: Collection<V2PersistedTrackExclusion>,
    ): List<V2PersistedTrackExclusion> = values
        .groupBy<V2PersistedTrackExclusion, Any> { it.stableTrackSpanId ?: it.providerSpan }
        .values
        .map { records -> records.maxBy { it.lastKnownPowerampFileId } }
        .sortedWith(
            compareBy<V2PersistedTrackExclusion> { it.providerSpan.normalizedPhysicalPath }
                .thenBy { it.providerSpan.offsetMs }
                .thenBy { it.providerSpan.durationMs },
        )

    private val STABLE_SPAN_ID = Regex("^stable-track-span-v1-[0-9a-f]{64}$")
}
