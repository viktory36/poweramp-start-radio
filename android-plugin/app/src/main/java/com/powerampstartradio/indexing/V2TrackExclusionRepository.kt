package com.powerampstartradio.indexing

import android.content.Context
import com.google.gson.Gson
import com.google.gson.GsonBuilder
import com.powerampstartradio.indexing.v2.V2StableProviderLexicalPathNormalizer
import java.nio.ByteBuffer
import java.security.MessageDigest

data class V2ResolvedTrackExclusions(
    val never: List<V2PersistedTrackExclusion>,
    val ignored: List<V2PersistedTrackExclusion>,
    val neverIds: Set<Long>,
    val ignoredIds: Set<Long>,
    val persistedFingerprint: String = "",
)

/** One shared persistence and resolution contract for Settings and Manage Tracks. */
internal class V2TrackExclusionRepository(
    context: Context,
    preferencesName: String = PREFERENCES_NAME,
) {
    init {
        require(preferencesName.isNotBlank()) { "preferences name must not be blank" }
    }

    private val prefs = context.applicationContext
        .getSharedPreferences(preferencesName, Context.MODE_PRIVATE)
    private val gson = GsonBuilder().disableHtmlEscaping().create()

    fun loadPersisted(): Pair<List<V2PersistedTrackExclusion>, List<V2PersistedTrackExclusion>> =
        synchronized(lock) {
            loadEnvelope(NEVER_EXCLUSIONS_KEY) to loadEnvelope(IGNORED_EXCLUSIONS_KEY)
        }

    /** Stable cache key for Settings' ready count; changes whenever either saved envelope changes. */
    fun persistedFingerprint(): String = synchronized(lock) {
        persistedFingerprintLocked()
    }

    private fun persistedFingerprintLocked(): String {
        val digest = MessageDigest.getInstance("SHA-256")
        listOf(NEVER_EXCLUSIONS_KEY, IGNORED_EXCLUSIONS_KEY).forEach { key ->
            val bytes = prefs.getString(key, "").orEmpty().toByteArray(Charsets.UTF_8)
            digest.update(ByteBuffer.allocate(Int.SIZE_BYTES).putInt(bytes.size).array())
            digest.update(bytes)
        }
        return digest.digest().joinToString("") { byte -> "%02x".format(byte) }
    }

    fun resolveAndMigrate(
        tracks: Collection<NewTrackDetector.UnindexedTrack>,
    ): V2ResolvedTrackExclusions = synchronized(lock) {
        val candidates = tracks.mapNotNull(::candidate)
        var never = loadEnvelope(NEVER_EXCLUSIONS_KEY)
        var ignored = loadEnvelope(IGNORED_EXCLUSIONS_KEY)
        val legacyNever = if (!prefs.contains(NEVER_EXCLUSIONS_KEY)) {
            loadLegacyIds(LEGACY_NEVER_IDS_KEY)
        } else {
            emptySet()
        }
        val legacyIgnored = if (!prefs.contains(IGNORED_EXCLUSIONS_KEY)) {
            loadLegacyIds(LEGACY_IGNORED_IDS_KEY)
        } else {
            emptySet()
        }
        if (legacyNever.isNotEmpty()) {
            never = V2TrackExclusionPolicy.add(
                never,
                candidates.filter { it.powerampFileId in legacyNever },
            )
        }
        if (legacyIgnored.isNotEmpty()) {
            ignored = V2TrackExclusionPolicy.add(
                ignored,
                candidates.filter { it.powerampFileId in legacyIgnored },
            )
        }
        if (legacyNever.isNotEmpty() || legacyIgnored.isNotEmpty()) {
            prefs.edit().putString(
                LEGACY_EXCLUSION_ARCHIVE_KEY,
                "dismissed=${legacyNever.sorted()};ignored=${legacyIgnored.sorted()}",
            ).commit()
        }
        never = V2TrackExclusionPolicy.refreshLocators(never, candidates)
        ignored = V2TrackExclusionPolicy.refreshLocators(ignored, candidates)
        persistLocked(never, ignored)
        V2ResolvedTrackExclusions(
            never = never,
            ignored = ignored,
            neverIds = V2TrackExclusionPolicy.resolve(never, candidates),
            ignoredIds = V2TrackExclusionPolicy.resolve(ignored, candidates),
            persistedFingerprint = persistedFingerprintLocked(),
        )
    }

    fun persist(
        never: Collection<V2PersistedTrackExclusion>,
        ignored: Collection<V2PersistedTrackExclusion>,
    ) = synchronized(lock) {
        persistLocked(
            V2TrackExclusionPolicy.requireValid(
                V2TrackExclusionEnvelope(entries = never.toList()),
            ).entries,
            V2TrackExclusionPolicy.requireValid(
                V2TrackExclusionEnvelope(entries = ignored.toList()),
            ).entries,
        )
    }

    private fun loadEnvelope(key: String): List<V2PersistedTrackExclusion> {
        val json = prefs.getString(key, null) ?: return emptyList()
        return try {
            val envelope = gson.fromJson(json, V2TrackExclusionEnvelope::class.java)
                ?: error("empty exclusion envelope")
            V2TrackExclusionPolicy.requireValid(envelope).entries
        } catch (error: Exception) {
            throw V2TrackExclusionReadException(key, error)
        }
    }

    private fun loadLegacyIds(key: String): Set<Long> {
        val json = prefs.getString(key, null) ?: return emptySet()
        return try {
            val values = Gson().fromJson(json, LongArray::class.java) ?: LongArray(0)
            values.filterTo(linkedSetOf()) { it > 0L }
        } catch (error: Exception) {
            throw V2TrackExclusionReadException(key, error)
        }
    }

    private fun persistLocked(
        never: Collection<V2PersistedTrackExclusion>,
        ignored: Collection<V2PersistedTrackExclusion>,
    ) {
        check(
            prefs.edit()
                .putString(
                    NEVER_EXCLUSIONS_KEY,
                    gson.toJson(V2TrackExclusionEnvelope(entries = never.toList())),
                )
                .putString(
                    IGNORED_EXCLUSIONS_KEY,
                    gson.toJson(V2TrackExclusionEnvelope(entries = ignored.toList())),
                )
                .remove(LEGACY_NEVER_IDS_KEY)
                .remove(LEGACY_IGNORED_IDS_KEY)
                .remove("dismissed_db_fingerprint")
                .commit()
        ) { "Unable to persist V2 track exclusions" }
    }

    companion object {
        const val PREFERENCES_NAME = "indexing"
        const val NEVER_EXCLUSIONS_KEY = "v2_never_index_exclusions"
        const val IGNORED_EXCLUSIONS_KEY = "v2_ignored_track_exclusions"
        const val LEGACY_EXCLUSION_ARCHIVE_KEY = "v1_numeric_exclusions_archive"
        private const val LEGACY_NEVER_IDS_KEY = "dismissed_track_ids"
        private const val LEGACY_IGNORED_IDS_KEY = "ignored_track_ids"
        private val lock = Any()

        fun candidate(
            track: NewTrackDetector.UnindexedTrack,
        ): V2TrackExclusionCandidate? = candidate(
            powerampFileId = track.powerampFileId,
            physicalPath = track.path,
            offsetMs = track.offsetMs,
            durationMs = track.durationMs.toLong(),
            stableTrackSpanId = null,
        )

        fun candidate(
            powerampFileId: Long,
            physicalPath: String?,
            offsetMs: Long,
            durationMs: Long,
            stableTrackSpanId: String?,
        ): V2TrackExclusionCandidate? {
            if (powerampFileId <= 0L || physicalPath.isNullOrBlank() ||
                offsetMs < 0L
            ) return null
            val normalizedPath = runCatching {
                V2StableProviderLexicalPathNormalizer.normalizeAbsolute(physicalPath)
            }.getOrNull() ?: return null
            return V2TrackExclusionCandidate(
                powerampFileId = powerampFileId,
                providerSpan = V2ProviderSpanLocatorPolicy.create(
                    normalizedPath,
                    offsetMs,
                    durationMs,
                ),
                stableTrackSpanId = stableTrackSpanId,
            )
        }
    }
}

internal class V2TrackExclusionReadException(
    val preferenceKey: String,
    cause: Throwable,
) : IllegalStateException("Saved track exclusions could not be verified", cause)
