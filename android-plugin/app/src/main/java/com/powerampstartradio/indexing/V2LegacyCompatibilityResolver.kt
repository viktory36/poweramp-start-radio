package com.powerampstartradio.indexing

import java.text.Normalizer

internal enum class V2LegacyCompatibilityEvidence {
    EXACT_ABSOLUTE_PATH,
    EXACT_MUSIC_RELATIVE_PATH,
    CUE_LOGICAL_METADATA_REPAIR,
    EXACT_PATH_TIMING_CONFLICT,
}

internal data class V2LegacyProviderCandidate(
    val powerampFileId: Long,
    val normalizedPhysicalPath: String,
    val offsetMs: Long,
    val durationMs: Int,
    val metadataKey: String,
    /** False for CUE/path-group rows whose imported embedding has no trustworthy span identity. */
    val compatibilityEligible: Boolean,
    /** True for logical rows whose imported predecessor did not retain reliable span provenance. */
    val requiresSpanSpecificRebuild: Boolean = false,
    /** Poweramp's first-seen time for this exact provider row, in Unix epoch seconds. */
    val createdAtEpochSecond: Long = 1L,
)

internal data class V2LegacyDatabaseCandidate(
    val trackId: Long,
    val normalizedPath: String?,
    val durationMs: Int,
    val metadataKey: String,
)

internal data class V2LegacyCompatibilityBinding(
    val powerampFileId: Long,
    val trackId: Long,
    val evidence: V2LegacyCompatibilityEvidence,
)

internal data class V2LegacyCompatibilityResult(
    val bindings: List<V2LegacyCompatibilityBinding>,
    val repairBindings: List<V2LegacyCompatibilityBinding>,
    val pathTimingConflictBindings: List<V2LegacyCompatibilityBinding>,
    val unmatchedPowerampFileIds: Set<Long>,
    val unmatchedTrackIds: Set<Long>,
)

/**
 * Conservative migration-only reconciliation for imported embeddings.
 *
 * Exact V2 receipts remain the only acoustic provenance. This resolver merely establishes that
 * one imported DB row and one current provider row are reciprocally unique under strong identity
 * evidence. It never lets a filename/tag hint consume a remaster, duplicate, or edit.
 */
internal object V2LegacyCompatibilityResolver {
    fun resolve(
        provider: Collection<V2LegacyProviderCandidate>,
        database: Collection<V2LegacyDatabaseCandidate>,
    ): V2LegacyCompatibilityResult {
        require(provider.mapTo(hashSetOf()) { it.powerampFileId }.size == provider.size) {
            "duplicate provider candidate ID"
        }
        require(database.mapTo(hashSetOf()) { it.trackId }.size == database.size) {
            "duplicate legacy track ID"
        }

        val remainingProvider = provider.associateByTo(hashMapOf()) { it.powerampFileId }
        val remainingDatabase = database.associateByTo(hashMapOf()) { it.trackId }
        val bindings = mutableListOf<V2LegacyCompatibilityBinding>()
        val repairBindings = mutableListOf<V2LegacyCompatibilityBinding>()
        val pathTimingConflictBindings = mutableListOf<V2LegacyCompatibilityBinding>()

        bindReciprocalUnique(
            remainingProvider,
            remainingDatabase,
            V2LegacyCompatibilityEvidence.EXACT_ABSOLUTE_PATH,
            keyForProvider = { candidate ->
                candidate.normalizedPhysicalPath.takeIf {
                    candidate.compatibilityEligible && !candidate.requiresSpanSpecificRebuild &&
                        candidate.offsetMs == 0L && it.isNotBlank()
                }
            },
            keyForDatabase = { it.normalizedPath?.takeIf(String::isNotBlank) },
            compatible = ::pathEvidenceCompatible,
            sink = bindings,
        )
        bindReciprocalUnique(
            remainingProvider,
            remainingDatabase,
            V2LegacyCompatibilityEvidence.EXACT_MUSIC_RELATIVE_PATH,
            keyForProvider = { candidate ->
                candidate.normalizedPhysicalPath.takeIf {
                    candidate.compatibilityEligible && !candidate.requiresSpanSpecificRebuild &&
                        candidate.offsetMs == 0L
                }?.let(::strictMusicRelativePath)
            },
            keyForDatabase = { it.normalizedPath?.let(::strictMusicRelativePath) },
            compatible = ::pathEvidenceCompatible,
            sink = bindings,
        )
        // A legacy CUE row can identify which imported row it supersedes without claiming that
        // the imported audio was decoded from the correct logical span.
        bindReciprocalUnique(
            remainingProvider,
            remainingDatabase,
            V2LegacyCompatibilityEvidence.CUE_LOGICAL_METADATA_REPAIR,
            keyForProvider = { candidate ->
                if (!candidate.requiresSpanSpecificRebuild || candidate.durationMs <= 0) {
                    null
                } else {
                    val path = strictMusicRelativePath(candidate.normalizedPhysicalPath)
                    val metadata = metadataWithoutDuration(candidate.metadataKey)
                    if (path == null || metadata.isBlank()) null else "$path\u0000$metadata"
                }
            },
            keyForDatabase = { candidate ->
                val path = candidate.normalizedPath?.let(::strictMusicRelativePath)
                val metadata = metadataWithoutDuration(candidate.metadataKey)
                if (path == null || metadata.isBlank()) null else "$path\u0000$metadata"
            },
            compatible = ::durationCompatible,
            sink = repairBindings,
        )

        // A reciprocal path with incompatible timing is evidence of a conflict, not coverage and
        // not proof that the current bytes are new. Keep it out of automatic indexing.
        bindReciprocalUnique(
            remainingProvider,
            remainingDatabase,
            V2LegacyCompatibilityEvidence.EXACT_PATH_TIMING_CONFLICT,
            keyForProvider = { candidate ->
                candidate.normalizedPhysicalPath.takeIf {
                    !candidate.requiresSpanSpecificRebuild && candidate.offsetMs == 0L
                }?.let(::strictMusicRelativePath)
            },
            keyForDatabase = { it.normalizedPath?.let(::strictMusicRelativePath) },
            compatible = { providerCandidate, databaseCandidate ->
                !durationCompatible(providerCandidate, databaseCandidate)
            },
            sink = pathTimingConflictBindings,
        )

        return V2LegacyCompatibilityResult(
            bindings = bindings.sortedWith(
                compareBy(V2LegacyCompatibilityBinding::powerampFileId)
                    .thenBy(V2LegacyCompatibilityBinding::trackId),
            ),
            repairBindings = repairBindings.sortedWith(
                compareBy(V2LegacyCompatibilityBinding::powerampFileId)
                    .thenBy(V2LegacyCompatibilityBinding::trackId),
            ),
            pathTimingConflictBindings = pathTimingConflictBindings.sortedWith(
                compareBy(V2LegacyCompatibilityBinding::powerampFileId)
                    .thenBy(V2LegacyCompatibilityBinding::trackId),
            ),
            unmatchedPowerampFileIds = remainingProvider.keys.toSet(),
            unmatchedTrackIds = remainingDatabase.keys.toSet(),
        )
    }

    private fun bindReciprocalUnique(
        remainingProvider: MutableMap<Long, V2LegacyProviderCandidate>,
        remainingDatabase: MutableMap<Long, V2LegacyDatabaseCandidate>,
        evidence: V2LegacyCompatibilityEvidence,
        keyForProvider: (V2LegacyProviderCandidate) -> String?,
        keyForDatabase: (V2LegacyDatabaseCandidate) -> String?,
        compatible: (V2LegacyProviderCandidate, V2LegacyDatabaseCandidate) -> Boolean,
        sink: MutableList<V2LegacyCompatibilityBinding>,
    ) {
        val providersByKey = groupCandidatesByKey(remainingProvider.values, keyForProvider)
        val databaseByKey = groupCandidatesByKey(remainingDatabase.values, keyForDatabase)

        for ((key, providerBucket) in providersByKey) {
            val databaseBucket = databaseByKey[key] ?: continue
            val soleProvider = providerBucket.singleOrNull()
            val soleDatabase = databaseBucket.singleOrNull()
            if (soleProvider != null && soleDatabase != null) {
                if (compatible(soleProvider, soleDatabase)) {
                    bindCompatibilityPair(
                        remainingProvider,
                        remainingDatabase,
                        soleProvider,
                        soleDatabase,
                        evidence,
                        sink,
                    )
                }
                continue
            }

            val providers = providerBucket.toList()
            val databases = databaseBucket.toList()
            val providerDegree = IntArray(providers.size)
            val soleDatabaseIndex = IntArray(providers.size) { -1 }
            val databaseDegree = IntArray(databases.size)
            for (providerIndex in providers.indices) {
                val providerCandidate = providers[providerIndex]
                for (databaseIndex in databases.indices) {
                    if (!compatible(providerCandidate, databases[databaseIndex])) continue
                    providerDegree[providerIndex]++
                    soleDatabaseIndex[providerIndex] = databaseIndex
                    databaseDegree[databaseIndex]++
                }
            }
            for (providerIndex in providers.indices) {
                if (providerDegree[providerIndex] != 1) continue
                val databaseIndex = soleDatabaseIndex[providerIndex]
                if (databaseDegree[databaseIndex] != 1) continue
                bindCompatibilityPair(
                    remainingProvider,
                    remainingDatabase,
                    providers[providerIndex],
                    databases[databaseIndex],
                    evidence,
                    sink,
                )
            }
        }
    }

    private fun bindCompatibilityPair(
        remainingProvider: MutableMap<Long, V2LegacyProviderCandidate>,
        remainingDatabase: MutableMap<Long, V2LegacyDatabaseCandidate>,
        provider: V2LegacyProviderCandidate,
        database: V2LegacyDatabaseCandidate,
        evidence: V2LegacyCompatibilityEvidence,
        sink: MutableList<V2LegacyCompatibilityBinding>,
    ) {
        check(remainingProvider.remove(provider.powerampFileId) != null) {
            "reciprocal compatibility binding reused provider ${provider.powerampFileId}"
        }
        check(remainingDatabase.remove(database.trackId) != null) {
            "reciprocal compatibility binding reused legacy track ${database.trackId}"
        }
        sink += V2LegacyCompatibilityBinding(provider.powerampFileId, database.trackId, evidence)
    }

    private fun <T> groupCandidatesByKey(
        candidates: Collection<T>,
        keyForCandidate: (T) -> String?,
    ): Map<String, CandidateBucket<T>> {
        val result = HashMap<String, CandidateBucket<T>>()
        for (candidate in candidates) {
            val key = keyForCandidate(candidate) ?: continue
            val bucket = result[key]
            if (bucket == null) {
                result[key] = CandidateBucket(candidate)
            } else {
                bucket.add(candidate)
            }
        }
        return result
    }

    private class CandidateBucket<T>(private val first: T) {
        private var additional: ArrayList<T>? = null

        fun add(candidate: T) {
            val values = additional ?: ArrayList<T>().also { additional = it }
            values += candidate
        }

        fun singleOrNull(): T? = first.takeIf { additional == null }

        fun toList(): List<T> = ArrayList<T>(1 + (additional?.size ?: 0)).apply {
            add(this@CandidateBucket.first)
            additional?.let(::addAll)
        }
    }

    private fun durationCompatible(
        provider: V2LegacyProviderCandidate,
        database: V2LegacyDatabaseCandidate,
    ): Boolean {
        val left = provider.durationMs
        val right = database.durationMs
        if (left <= 0 || right <= 0) return false
        return kotlin.math.abs(left - right) <= DURATION_TOLERANCE_MS
    }

    /** Exact reciprocal paths tolerate broken provider timing only when all normalized tags agree. */
    internal fun pathEvidenceCompatible(
        provider: V2LegacyProviderCandidate,
        database: V2LegacyDatabaseCandidate,
    ): Boolean = durationCompatible(provider, database) ||
        metadataWithoutDuration(provider.metadataKey).takeIf(String::isNotBlank) ==
        metadataWithoutDuration(database.metadataKey).takeIf(String::isNotBlank)

    internal fun strictMusicRelativePath(path: String): String? {
        val normalized = Normalizer.normalize(path.replace('\\', '/').trim(), Normalizer.Form.NFC)
        val marker = "/music/"
        val index = normalized.lowercase().indexOf(marker)
        if (index < 0) return null
        return normalized.substring(index + marker.length).takeIf(String::isNotBlank)
    }

    internal fun metadataWithoutDuration(metadataKey: String): String =
        metadataKey.substringBeforeLast('|', missingDelimiterValue = "")

    private const val DURATION_TOLERANCE_MS = 5_000
}
