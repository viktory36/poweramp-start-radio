package com.powerampstartradio.data

import android.util.Log
import com.powerampstartradio.indexing.v2.StableTrackSpanIdentityStrength
import com.powerampstartradio.indexing.v2.V2EmbeddingCommitRepository
import com.powerampstartradio.indexing.v2.V2IndexGenerationReader
import com.powerampstartradio.indexing.v2.V2IndexingLedgerIds
import com.powerampstartradio.indexing.v2.V2ResolvedActiveIndexGeneration
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.charset.StandardCharsets
import java.security.MessageDigest
import java.text.Normalizer
import java.util.Locale

data class StableIdentityGenerationBinding(
    val bindingSpecId: String,
    val generationId: String,
    val activationBindingId: String,
    val databaseContentSha256: String,
    val orderedTrackSetSha256: String,
)

data class UniformShuffleIdentityKey(
    val high: Long,
    val low: Long,
    val identityToken: String,
    val isStableAcrossGenerations: Boolean,
)

/** One queue-visible recording identity; exact-content provenance remains separate. */
data class StableVisibleResultIdentity(
    val identityToken: String,
    val isCollapsibleRecording: Boolean,
)

internal data class StableTrackIdentityRow(
    val trackId: Long,
    val stableTrackSpanId: String?,
    val stableIdentitySpecId: String?,
    val stableIdentityStrength: StableTrackSpanIdentityStrength?,
    val embeddingSpecId: String?,
    val embeddingSha256: String?,
    val filenameKey: String? = null,
    val artist: String? = null,
    val album: String? = null,
    val title: String? = null,
    val durationMs: Int = 0,
)

sealed interface StableTrackIdentityResolution {
    data class Resolved(
        val trackId: Long,
        val allEquivalentTrackIds: List<Long>,
    ) : StableTrackIdentityResolution
    data object Missing : StableTrackIdentityResolution
    data object LegacyBindingRequired : StableTrackIdentityResolution
    data object LegacyBindingMismatch : StableTrackIdentityResolution
}

/**
 * Generation-scoped bridge from SQLite locators to path-independent acoustic-span identities.
 *
 * Full-content V2 IDs remain the only cross-generation provenance. Within one active library,
 * metadata-proposed copies may also share a generation-bound queue identity after their stored
 * model vectors meet a strict equivalence bound; that identity never participates in durable replay.
 */
class StableTrackIdentityCatalog private constructor(
    val binding: StableIdentityGenerationBinding,
    private val orderedRows: List<StableTrackIdentityRow>,
    private val keysByTrackId: Map<Long, UniformShuffleIdentityKey>,
    private val stableIdByTrackId: Map<Long, String>,
    private val trackIdsByStableId: Map<String, List<Long>>,
    private val trackIdsByVisibleIdentity: Map<String, List<Long>>,
) {
    val trackCount: Int get() = orderedRows.size
    /** Rows with full-content evidence that is safe to resolve across paths/generations. */
    val stableTrackCount: Int get() = stableIdByTrackId.size
    /** Legacy rows plus any receipt whose sampled identity is not an equality proof. */
    val legacyTrackCount: Int get() = trackCount - stableTrackCount
    val duplicateExcessCount: Int = trackIdsByVisibleIdentity.values.sumOf {
        (it.size - 1).coerceAtLeast(0)
    }

    fun containsTrackId(trackId: Long): Boolean = keysByTrackId.containsKey(trackId)

    fun stableTrackSpanId(trackId: Long): String? = stableIdByTrackId[trackId]

    fun visibleResultIdentity(trackId: Long): StableVisibleResultIdentity {
        val key = shuffleIdentityKey(trackId)
        return StableVisibleResultIdentity(
            identityToken = key.identityToken,
            isCollapsibleRecording =
                trackIdsByVisibleIdentity.getValue(key.identityToken).size > 1,
        )
    }

    /** Every DB occurrence which represents the same queue-visible recording. */
    fun equivalentVisibleTrackIds(trackId: Long): List<Long> {
        val token = visibleResultIdentity(trackId).identityToken
        return trackIdsByVisibleIdentity.getValue(token)
    }

    /** Stable tie key for objective rankings; numeric IDs are only a legacy fallback. */
    fun rankingTieKey(trackId: Long): String = shuffleIdentityKey(trackId).identityToken

    /** Enough objective-ranked rows to absorb every queue-visible duplicate collapse. */
    fun rankedRowsForVisibleCount(requestedVisibleCount: Int): Int {
        if (requestedVisibleCount <= 0) return 0
        return (requestedVisibleCount.toLong() + duplicateExcessCount)
            .coerceAtMost(trackCount.toLong())
            .toInt()
    }

    fun shuffleIdentityKey(trackId: Long): UniformShuffleIdentityKey =
        requireNotNull(keysByTrackId[trackId]) { "Track $trackId is not in the active embedding generation" }

    fun resolveStable(stableTrackSpanId: String): StableTrackIdentityResolution {
        val trackIds = trackIdsByStableId[stableTrackSpanId].orEmpty()
        return when (trackIds.size) {
            0 -> StableTrackIdentityResolution.Missing
            else -> StableTrackIdentityResolution.Resolved(trackIds.first(), trackIds)
        }
    }

    fun resolveLegacy(
        trackId: Long,
        savedBinding: StableIdentityGenerationBinding?,
    ): StableTrackIdentityResolution {
        if (savedBinding == null) return StableTrackIdentityResolution.LegacyBindingRequired
        if (savedBinding != binding) return StableTrackIdentityResolution.LegacyBindingMismatch
        return if (containsTrackId(trackId)) {
            StableTrackIdentityResolution.Resolved(trackId, listOf(trackId))
        } else {
            StableTrackIdentityResolution.Missing
        }
    }

    fun orderedTrackIds(): LongArray = LongArray(orderedRows.size) { orderedRows[it].trackId }

    companion object {
        private const val TAG = "StableTrackIdentity"
        private const val ACTIVE_BINDING_SPEC = "v2-active-index-generation-binding-v1"
        private const val LEGACY_BINDING_SPEC = "legacy-database-generation-binding-v1"
        private const val STABLE_ID_PREFIX = "stable-track-span-v1-"
        private const val QUEUE_DUPLICATE_MAX_COMPONENT_DELTA = 0.000062f
        private val STABLE_ID = Regex("^stable-track-span-v1-[0-9a-f]{64}$")
        private val SHA256 = Regex("^[0-9a-f]{64}$")
        private val METADATA_WHITESPACE = Regex("\\s+")

        @Volatile
        private var cached: CachedCatalog? = null

        @Synchronized
        fun load(
            filesDir: File,
            database: EmbeddingDatabase,
            embeddingIndex: EmbeddingIndex,
        ): StableTrackIdentityCatalog {
            val databaseFile = database.databaseFile
            val active = activeGenerationFor(filesDir, databaseFile)
            val orderedTrackSetSha256 = orderedTrackSetSha256(embeddingIndex)
            val signature = CatalogSignature.capture(
                databaseFile,
                embeddingIndex,
                orderedTrackSetSha256,
                active,
            )
            cached?.takeIf { it.signature == signature }?.let { return it.catalog }

            val binding = active?.let(::activeBinding)
                ?: legacyBinding(databaseFile, orderedTrackSetSha256)
            val expectedEmbeddingSpecId = active?.manifest?.receiptEmbeddingSpec?.specId
            val rows = database.getStableTrackIdentityRows(expectedEmbeddingSpecId)
            val queueDuplicatePairs = generationQueueDuplicatePairs(rows, embeddingIndex)
            val catalog = fromOrderedRows(
                binding = binding,
                orderedEmbeddingTrackIds = LongArray(embeddingIndex.numTracks, embeddingIndex::getTrackId),
                rows = rows,
                queueDuplicatePairs = queueDuplicatePairs,
            )
            Log.i(
                TAG,
                "Queue-visible identity: ${queueDuplicatePairs.size} strict legacy metadata/vector " +
                    "pairs, ${catalog.duplicateExcessCount} duplicate rows collapsed",
            )
            database.requireByteEqualStableIdentityEmbeddings(
                catalog.trackIdsByStableId.values.filter { it.size > 1 },
            )
            active?.manifest?.stableTrackUidCoverage?.let { coverage ->
                require(catalog.trackCount == active.manifest.trackCount &&
                    catalog.stableTrackCount == coverage.fullContentIdentityCount &&
                    catalog.legacyTrackCount ==
                    coverage.uncoveredTrackCount + coverage.sampledContentIdentityCount
                ) { "Stable identity catalog disagrees with the active generation manifest" }
            }
            if (active == null) {
                require(databaseSnapshotSha256(databaseFile) == binding.databaseContentSha256) {
                    "Legacy embedding database changed while its identity catalog was loading"
                }
            }
            val finalSignature = CatalogSignature.capture(
                databaseFile,
                embeddingIndex,
                orderedTrackSetSha256(embeddingIndex),
                active,
            )
            require(finalSignature == signature) {
                "Embedding generation changed while its identity catalog was loading"
            }
            cached = CachedCatalog(finalSignature, catalog)
            return catalog
        }

        internal fun fromOrderedRows(
            binding: StableIdentityGenerationBinding,
            orderedEmbeddingTrackIds: LongArray,
            rows: List<StableTrackIdentityRow>,
            queueDuplicatePairs: List<Pair<Long, Long>> = emptyList(),
        ): StableTrackIdentityCatalog {
            require(rows.size == orderedEmbeddingTrackIds.size) {
                "Stable identity rows ${rows.size} != embedding rows ${orderedEmbeddingTrackIds.size}"
            }
            val stableByTrack = HashMap<Long, String>()
            val tracksByStable = LinkedHashMap<String, MutableList<Long>>()
            val fallbackNamespace = digest128(
                "legacy-shuffle-fallback-v1\u0000${binding.activationBindingId}",
            )

            var previousTrackId = Long.MIN_VALUE
            rows.forEachIndexed { index, row ->
                require(row.trackId == orderedEmbeddingTrackIds[index]) {
                    "Stable identity row ${row.trackId} is not aligned with PEMB row " +
                        "${orderedEmbeddingTrackIds[index]} at index $index"
                }
                require(row.trackId > previousTrackId) { "Stable identity rows are not strictly ordered" }
                previousTrackId = row.trackId

                val stableId = row.stableTrackSpanId
                if (stableId != null) {
                    require(STABLE_ID.matches(stableId)) { "Invalid stable track-span ID for ${row.trackId}" }
                    require(
                        row.stableIdentitySpecId ==
                            V2IndexingLedgerIds.STABLE_TRACK_SPAN_IDENTITY_SPEC_ID &&
                            row.stableIdentityStrength != null && row.embeddingSpecId != null &&
                            row.embeddingSha256?.matches(SHA256) == true
                    ) { "Incomplete stable identity receipt for ${row.trackId}" }
                    if (row.stableIdentityStrength == StableTrackSpanIdentityStrength.FULL_CONTENT_SHA256) {
                        stableByTrack[row.trackId] = stableId
                        tracksByStable.getOrPut(stableId) { mutableListOf() } += row.trackId
                    }
                } else {
                    require(row.stableIdentitySpecId == null && row.stableIdentityStrength == null &&
                        row.embeddingSpecId == null && row.embeddingSha256 == null
                    ) { "Partial legacy identity receipt for ${row.trackId}" }
                }
            }

            val rowsByTrackId = rows.associateBy(StableTrackIdentityRow::trackId)
            tracksByStable.forEach { (stableId, trackIds) ->
                if (trackIds.size <= 1) return@forEach
                val equivalentRows = trackIds.map { trackId ->
                    requireNotNull(rowsByTrackId[trackId])
                }
                require(
                    equivalentRows.map { it.embeddingSpecId }.distinct().size == 1 &&
                        equivalentRows.map { it.embeddingSha256 }.distinct().size == 1 &&
                        equivalentRows.map { it.stableIdentityStrength }.distinct().size == 1
                ) { "Stable identity $stableId has non-equivalent embedding receipts" }
            }

            val rowPositionById = rows.indices.associateBy { rows[it].trackId }
            val components = DisjointTrackComponents(rows.size)
            tracksByStable.values.forEach { equivalentIds ->
                val first = rowPositionById.getValue(equivalentIds.first())
                equivalentIds.drop(1).forEach { trackId ->
                    components.union(first, rowPositionById.getValue(trackId))
                }
            }
            requireCompleteQueueDuplicateComponents(queueDuplicatePairs)
            queueDuplicatePairs.forEach { (leftTrackId, rightTrackId) ->
                require(leftTrackId != rightTrackId) {
                    "Queue-visible duplicate pair repeats track $leftTrackId"
                }
                val left = requireNotNull(rowPositionById[leftTrackId]) {
                    "Queue-visible duplicate track $leftTrackId is outside the active generation"
                }
                val right = requireNotNull(rowPositionById[rightTrackId]) {
                    "Queue-visible duplicate track $rightTrackId is outside the active generation"
                }
                components.union(left, right)
            }
            val componentRows = rows.indices.groupBy(components::find).values
            val keys = HashMap<Long, UniformShuffleIdentityKey>(rows.size)
            val tracksByVisibleIdentity = LinkedHashMap<String, List<Long>>(componentRows.size)
            componentRows.forEach { positions ->
                val componentTrackIds = positions.map { rows[it].trackId }.sorted()
                val componentStableIds = componentTrackIds.mapNotNull(stableByTrack::get).distinct()
                val key = if (componentTrackIds.size == 1) {
                    stableByTrack[componentTrackIds.single()]?.let(::keyFromStableId)
                        ?: fallbackKey(fallbackNamespace, componentTrackIds.single())
                } else if (
                    componentStableIds.size == 1 &&
                    componentTrackIds.all { stableByTrack[it] == componentStableIds.single() }
                ) {
                    keyFromStableId(componentStableIds.single())
                } else {
                    queueRecordingKey(binding, componentTrackIds)
                }
                require(tracksByVisibleIdentity.put(key.identityToken, componentTrackIds) == null) {
                    "Queue recording identity token collision"
                }
                componentTrackIds.forEach { trackId ->
                    require(keys.put(trackId, key) == null) { "Duplicate track ID $trackId" }
                }
            }

            return StableTrackIdentityCatalog(
                binding = binding,
                orderedRows = rows.toList(),
                keysByTrackId = keys,
                stableIdByTrackId = stableByTrack,
                trackIdsByStableId = tracksByStable.mapValues { it.value.toList() },
                trackIdsByVisibleIdentity = tracksByVisibleIdentity,
            )
        }

        /**
         * Confirm generation-local copies without pretending mutable tags are durable identity.
         * Metadata only proposes a pair; numerically indistinguishable stored model vectors must
         * confirm it. The resulting component is used only to avoid repeated queue entries.
         */
        internal fun generationQueueDuplicatePairs(
            rows: List<StableTrackIdentityRow>,
            embeddingIndex: EmbeddingIndex,
        ): List<Pair<Long, Long>> {
            if (rows.size < 2) return emptyList()

            val candidatesByMetadata = LinkedHashMap<QueueDuplicateMetadataKey, MutableList<Int>>()
            rows.forEachIndexed { position, row ->
                // V2 receipts already carry their own exact source/span identity. Never let a
                // metadata/vector heuristic broaden or join those proven identity components.
                if (row.stableTrackSpanId != null || row.stableIdentitySpecId != null ||
                    row.stableIdentityStrength != null || row.embeddingSpecId != null ||
                    row.embeddingSha256 != null
                ) {
                    return@forEachIndexed
                }
                if (row.durationMs <= 0) return@forEachIndexed
                val artist = row.artist.normalizedQueueDuplicateMetadata()
                val album = row.album.normalizedQueueDuplicateMetadata()
                val title = row.title.normalizedQueueDuplicateMetadata()
                if (artist.isEmpty() || album.isEmpty() || title.isEmpty()) return@forEachIndexed
                candidatesByMetadata
                    .getOrPut(
                        QueueDuplicateMetadataKey(artist, album, title, row.durationMs),
                    ) { mutableListOf() }
                    .add(position)
            }

            return buildList {
                candidatesByMetadata.values.forEach { positions ->
                    if (positions.size < 2) return@forEach
                    val passingPairs = ArrayList<Pair<Int, Int>>()
                    var complete = true
                    positions.forEachIndexed { leftOffset, leftPosition ->
                        for (rightOffset in leftOffset + 1 until positions.size) {
                            val rightPosition = positions[rightOffset]
                            if (embeddingIndex.hasEmbeddingWithinMaxAbsoluteDelta(
                                    leftPosition,
                                    rightPosition,
                                    QUEUE_DUPLICATE_MAX_COMPONENT_DELTA,
                                )
                            ) {
                                passingPairs += leftPosition to rightPosition
                            } else {
                                complete = false
                            }
                        }
                    }
                    // A threshold relation is not transitive. An incomplete A~B~C graph must
                    // therefore stay as three visible rows instead of becoming one component.
                    if (complete) {
                        passingPairs.forEach { (leftPosition, rightPosition) ->
                            add(rows[leftPosition].trackId to rows[rightPosition].trackId)
                        }
                    }
                }
            }
        }

        private fun requireCompleteQueueDuplicateComponents(
            pairs: List<Pair<Long, Long>>,
        ) {
            if (pairs.isEmpty()) return
            val adjacency = LinkedHashMap<Long, MutableSet<Long>>()
            pairs.forEach { (left, right) ->
                require(left != right) { "Queue-visible duplicate pair repeats track $left" }
                require(adjacency.getOrPut(left) { linkedSetOf() }.add(right) &&
                    adjacency.getOrPut(right) { linkedSetOf() }.add(left)
                ) { "Queue-visible duplicate pair $left/$right is repeated" }
            }
            val remaining = adjacency.keys.toMutableSet()
            while (remaining.isNotEmpty()) {
                val component = linkedSetOf<Long>()
                val pending = ArrayDeque<Long>()
                pending += remaining.first()
                while (pending.isNotEmpty()) {
                    val trackId = pending.removeFirst()
                    if (!component.add(trackId)) continue
                    adjacency.getValue(trackId).forEach(pending::addLast)
                }
                remaining.removeAll(component)
                require(component.all { trackId ->
                    adjacency.getValue(trackId).containsAll(component - trackId)
                }) {
                    "Queue-visible duplicate component is not pairwise complete: " +
                        component.sorted().joinToString(",")
                }
            }
        }

        private fun String?.normalizedQueueDuplicateMetadata(): String {
            if (this == null) return ""
            return METADATA_WHITESPACE.replace(
                Normalizer.normalize(this, Normalizer.Form.NFKC)
                    .trim()
                    .lowercase(Locale.ROOT),
                " ",
            )
        }

        private data class QueueDuplicateMetadataKey(
            val artist: String,
            val album: String,
            val title: String,
            val durationMs: Int,
        )

        private fun activeGenerationFor(
            filesDir: File,
            databaseFile: File,
        ): V2ResolvedActiveIndexGeneration? {
            val pointerFile = File(filesDir, "indexing_v2/generations/active-generation.json")
            if (!pointerFile.isFile) return null
            return V2IndexGenerationReader.requireActive(filesDir).also { generation ->
                require(generation.databaseFile.canonicalFile == databaseFile.canonicalFile) {
                    "Stable identity requires the exact database named by the active generation"
                }
            }
        }

        private fun activeBinding(active: V2ResolvedActiveIndexGeneration) =
            StableIdentityGenerationBinding(
                bindingSpecId = ACTIVE_BINDING_SPEC,
                generationId = active.manifest.generationId,
                activationBindingId = active.manifest.activationBindingId,
                databaseContentSha256 = active.manifest.databaseContentSha256,
                orderedTrackSetSha256 = active.manifest.orderedTrackSetSha256,
            )

        private fun legacyBinding(
            databaseFile: File,
            orderedTrackSetSha256: String,
        ): StableIdentityGenerationBinding {
            val databaseSha256 = databaseSnapshotSha256(databaseFile)
            val generationDigest = MessageDigest.getInstance("SHA-256").apply {
                update("legacy-index-generation-v1\u0000".toByteArray(StandardCharsets.UTF_8))
                update(databaseSha256.toByteArray(StandardCharsets.US_ASCII))
                update(orderedTrackSetSha256.toByteArray(StandardCharsets.US_ASCII))
            }.digest().toHex()
            val bindingId = "legacy-index-generation-v1-$generationDigest"
            return StableIdentityGenerationBinding(
                bindingSpecId = LEGACY_BINDING_SPEC,
                generationId = bindingId,
                activationBindingId = bindingId,
                databaseContentSha256 = databaseSha256,
                orderedTrackSetSha256 = orderedTrackSetSha256,
            )
        }

        private fun orderedTrackSetSha256(index: EmbeddingIndex): String {
            val digest = MessageDigest.getInstance("SHA-256")
            digest.update("stable-catalog-ordered-track-set-v1\u0000".toByteArray(StandardCharsets.UTF_8))
            val bytes = ByteBuffer.allocate(Long.SIZE_BYTES).order(ByteOrder.BIG_ENDIAN)
            for (position in 0 until index.numTracks) {
                bytes.clear()
                bytes.putLong(index.getTrackId(position))
                digest.update(bytes.array())
            }
            return digest.digest().toHex()
        }

        private fun keyFromStableId(stableId: String): UniformShuffleIdentityKey {
            val hex = stableId.removePrefix(STABLE_ID_PREFIX)
            return UniformShuffleIdentityKey(
                high = hex.substring(0, 16).toUnsignedLong(),
                low = hex.substring(16, 32).toUnsignedLong(),
                identityToken = stableId,
                isStableAcrossGenerations = true,
            )
        }

        private fun fallbackKey(namespace: Pair<Long, Long>, trackId: Long): UniformShuffleIdentityKey =
            UniformShuffleIdentityKey(
                high = mix64(namespace.first xor trackId),
                low = mix64(namespace.second + trackId * -7046029254386353131L),
                identityToken = "$LEGACY_BINDING_SPEC:${namespace.first}:${namespace.second}:$trackId",
                isStableAcrossGenerations = false,
            )

        private fun queueRecordingKey(
            binding: StableIdentityGenerationBinding,
            orderedTrackIds: List<Long>,
        ): UniformShuffleIdentityKey {
            val digest = MessageDigest.getInstance("SHA-256")
            digest.update(
                "queue-visible-recording-v2\u0000${binding.activationBindingId}\u0000"
                    .toByteArray(StandardCharsets.UTF_8),
            )
            val buffer = ByteBuffer.allocate(Long.SIZE_BYTES).order(ByteOrder.BIG_ENDIAN)
            orderedTrackIds.forEach { trackId ->
                buffer.clear()
                buffer.putLong(trackId)
                digest.update(buffer.array())
            }
            val bytes = digest.digest()
            val longs = ByteBuffer.wrap(bytes).order(ByteOrder.BIG_ENDIAN)
            val high = longs.long
            val low = longs.long
            return UniformShuffleIdentityKey(
                high = high,
                low = low,
                identityToken = "queue-visible-recording-v2:${high.toULong()}:${low.toULong()}",
                isStableAcrossGenerations = false,
            )
        }

        private fun digest128(value: String): Pair<Long, Long> {
            val bytes = MessageDigest.getInstance("SHA-256")
                .digest(value.toByteArray(StandardCharsets.UTF_8))
            val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.BIG_ENDIAN)
            return buffer.long to buffer.long
        }

        private fun mix64(input: Long): Long {
            var value = input
            value = (value xor (value ushr 30)) * -4658895280553007687L
            value = (value xor (value ushr 27)) * -7723592293110705685L
            return value xor (value ushr 31)
        }

        private fun String.toUnsignedLong(): Long = java.lang.Long.parseUnsignedLong(this, 16)

        private fun sha256(file: File): String {
            val digest = MessageDigest.getInstance("SHA-256")
            FileInputStream(file).use { input ->
                val buffer = ByteArray(1024 * 1024)
                while (true) {
                    val count = input.read(buffer)
                    if (count < 0) break
                    digest.update(buffer, 0, count)
                }
            }
            return digest.digest().toHex()
        }

        private fun databaseSnapshotSha256(databaseFile: File): String {
            val walFile = File(databaseFile.path + "-wal")
            val digest = MessageDigest.getInstance("SHA-256")
            digest.update("legacy-sqlite-snapshot-v1\u0000".toByteArray(StandardCharsets.UTF_8))
            listOf(databaseFile, walFile).forEach { file ->
                digest.update(if (file == databaseFile) 1.toByte() else 2.toByte())
                if (file.isFile) {
                    digest.update(1.toByte())
                    digest.update(sha256(file).toByteArray(StandardCharsets.US_ASCII))
                } else {
                    digest.update(0.toByte())
                }
            }
            return digest.digest().toHex()
        }

        private fun ByteArray.toHex(): String = joinToString("") {
            (it.toInt() and 0xff).toString(16).padStart(2, '0')
        }

        private data class CachedCatalog(
            val signature: CatalogSignature,
            val catalog: StableTrackIdentityCatalog,
        )

        private class DisjointTrackComponents(size: Int) {
            private val parent = IntArray(size) { it }
            private val rank = ByteArray(size)

            fun find(value: Int): Int {
                var root = value
                while (parent[root] != root) root = parent[root]
                var current = value
                while (parent[current] != current) {
                    val next = parent[current]
                    parent[current] = root
                    current = next
                }
                return root
            }

            fun union(left: Int, right: Int) {
                var leftRoot = find(left)
                var rightRoot = find(right)
                if (leftRoot == rightRoot) return
                if (rank[leftRoot] < rank[rightRoot]) {
                    val swap = leftRoot
                    leftRoot = rightRoot
                    rightRoot = swap
                }
                parent[rightRoot] = leftRoot
                if (rank[leftRoot] == rank[rightRoot]) rank[leftRoot]++
            }
        }

        private data class CatalogSignature(
            val databasePath: String,
            val databaseLength: Long,
            val databaseModifiedMs: Long,
            val databaseWalLength: Long,
            val databaseWalModifiedMs: Long,
            val embeddingTrackCount: Int,
            val orderedTrackSetSha256: String,
            val activeBindingId: String?,
        ) {
            companion object {
                fun capture(
                    databaseFile: File,
                    index: EmbeddingIndex,
                    orderedTrackSetSha256: String,
                    active: V2ResolvedActiveIndexGeneration?,
                ) = CatalogSignature(
                    databasePath = databaseFile.canonicalPath,
                    databaseLength = databaseFile.length(),
                    databaseModifiedMs = databaseFile.lastModified(),
                    databaseWalLength = File(databaseFile.path + "-wal").length(),
                    databaseWalModifiedMs = File(databaseFile.path + "-wal").lastModified(),
                    embeddingTrackCount = index.numTracks,
                    orderedTrackSetSha256 = orderedTrackSetSha256,
                    activeBindingId = active?.manifest?.activationBindingId,
                )
            }
        }
    }
}

private fun EmbeddingDatabase.getStableTrackIdentityRows(
    expectedEmbeddingSpecId: String?,
): List<StableTrackIdentityRow> = queryStableTrackIdentityRows(
    receiptTable = V2EmbeddingCommitRepository.RECEIPT_TABLE,
    receiptSchemaVersion = V2EmbeddingCommitRepository.RECEIPT_SCHEMA_VERSION,
    expectedEmbeddingSpecId = expectedEmbeddingSpecId,
)
