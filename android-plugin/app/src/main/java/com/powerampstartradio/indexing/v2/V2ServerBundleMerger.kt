package com.powerampstartradio.indexing.v2

import android.content.ContentValues
import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.net.Uri
import android.system.Os
import android.system.OsConstants
import com.powerampstartradio.AudioLibraryPermission
import com.powerampstartradio.data.EmbeddingDatabase
import com.powerampstartradio.indexing.GraphUpdater
import com.powerampstartradio.indexing.V2ActiveLibraryCatalog
import com.powerampstartradio.indexing.V2ActiveLibraryCatalogLoader
import com.powerampstartradio.indexing.V2ActiveLibraryCatalogStore
import com.powerampstartradio.indexing.V2ExactGraphIncrementalBase
import com.powerampstartradio.indexing.V2GraphUpdateStrategy
import com.powerampstartradio.indexing.V2GraphUpdaterControl
import com.powerampstartradio.indexing.V2GraphUpdaterProgress
import com.powerampstartradio.indexing.V2GraphUpdaterStage
import com.powerampstartradio.indexing.V2ProcessLibraryInspectionCoordinator
import com.powerampstartradio.poweramp.TrackNormalization
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.BufferedInputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.InputStream
import java.io.RandomAccessFile
import java.nio.charset.StandardCharsets
import java.security.MessageDigest
import java.util.UUID

/** Exact cross-platform contract admitted by the Android server-bundle merge boundary. */
internal object V2ServerBundleContract {
    const val BUNDLE_FORMAT = "poweramp-server-embedding-bundle"
    const val SCHEMA_VERSION = 1
    const val EMBEDDING_SPEC_ID = "poweramp-clamp3-server-v1"
    const val OUTPUT_SPACE_ID = "clamp3-joint-audio-text-l2-f32-v1"
    const val MERT_MODEL_SHA256 =
        "a2b8b747f72c06e0595aeae41ae5473f4364938c6b39b2c58be38c48e6bd3fcd"
    const val CLAMP3_AUDIO_MODEL_SHA256 =
        "5033f868e3977be3945ee416b5a1718d5589a173c7ba8982231d8c94a6441d80"
    const val EMBEDDING_SPEC_SHA256 =
        "d95d5373d2adb7b02f931edd355cc3fdee077ad9223f4bcd4bcb197cb6545412"
    const val EMBEDDING_SPEC_JSON =
        "{\"audio_span\":\"complete-physical-file\",\"clamp3_aggregation\":\"zero-bookends-" +
            "segment128-final-overlap-frame-weighted-l2\",\"clamp3_audio\":{" +
            "\"checkpoint_sha256\":\"$CLAMP3_AUDIO_MODEL_SHA256\"," +
            "\"repository\":\"sander-wood/clamp3\"," +
            "\"revision\":\"355625cc1c6f73726bbcd0eb9276ac7152d56426\"}," +
            "\"downmix\":\"arithmetic-channel-mean\"," +
            "\"embedding_spec_id\":\"$EMBEDDING_SPEC_ID\"," +
            "\"maximum_duration_seconds\":null,\"mert\":{" +
            "\"model_id\":\"m-a-p/MERT-v1-95M\"," +
            "\"pytorch_model_sha256\":\"$MERT_MODEL_SHA256\"," +
            "\"revision\":\"12af15fef9d0ac838c3f475bfbbf26d2060dd4f5\"}," +
            "\"mert_pooling\":\"mean-time-then-mean-layers\"," +
            "\"normalization\":\"wav2vec2-zero-mean-unit-variance-whole-span\"," +
            "\"output_dimension\":768,\"output_space_id\":\"$OUTPUT_SPACE_ID\"," +
            "\"precision\":\"fp32\"," +
            "\"resampler\":\"torchaudio-hann-width6-rolloff0.99-f32-target-length\"," +
            "\"schema_version\":1," +
            "\"tail_policy\":\"drop-below-1s-otherwise-zero-pad-to-5s\"," +
            "\"target_sample_rate_hz\":24000,\"window_hop_samples\":120000," +
            "\"window_samples\":120000}"

    const val TRACK_TABLE = "tracks"
    const val EMBEDDING_TABLE = "embeddings_clamp3"
    const val BUNDLE_TRACK_TABLE = "server_bundle_tracks"
    const val METADATA_TABLE = "metadata"

    const val METADATA_SCHEMA_VERSION = "server_bundle_schema_version"
    const val METADATA_BUNDLE_FORMAT = "bundle_format"
    const val METADATA_BUNDLE_ID = "server_bundle_id"
    const val METADATA_TRACK_COUNT = "server_bundle_track_count"
    const val METADATA_EMBEDDING_SPEC_ID = "server_bundle_embedding_spec_id"
    const val METADATA_OUTPUT_SPACE_ID = "server_bundle_output_space_id"
    const val METADATA_MERT_MODEL_SHA256 = "server_bundle_mert_model_sha256"
    const val METADATA_CLAMP3_AUDIO_MODEL_SHA256 =
        "server_bundle_clamp3_audio_model_sha256"
    const val METADATA_EMBEDDING_SPEC_JSON = "server_bundle_embedding_spec_json"
    const val METADATA_EMBEDDING_SPEC_SHA256 = "server_bundle_embedding_spec_sha256"
    const val METADATA_GRAPH_INCLUDED = "server_bundle_graph_included"
}

/** Durable per-row evidence for idempotent merges; separate from Android inference receipts. */
internal object V2ServerMergeReceiptContract {
    const val TABLE = "server_merge_receipts"
    const val SCHEMA_VERSION = 1

    val COLUMNS = listOf(
        "receipt_schema_version",
        "bundle_id",
        "root_id",
        "relative_path",
        "source_sha256",
        "source_size_bytes",
        "provider_file_id",
        "provider_physical_path",
        "track_id",
        "embedding_sha256",
        "embedding_spec_id",
        "output_space_id",
        "merged_at_epoch_ms",
    )
}

internal data class V2ServerBundleTrack(
    val trackId: Long,
    val rootId: String,
    val relativePath: String,
    val sourceSha256: String,
    val sourceSizeBytes: Long,
    val sourceSampleRateHz: Int,
    val sourceSampleCount: Long,
    val spanStartSample: Long,
    val spanEndSampleExclusive: Long,
    val embeddingSha256: String,
)

internal data class V2ServerBundleValidation(
    val sourceByteLength: Long,
    val sourceSha256: String,
    val bundleId: String,
    val tracks: List<V2ServerBundleTrack>,
)

internal enum class V2ServerBundleRowDisposition {
    ADDED,
    ALREADY_INDEXED,
    NOT_IN_POWERAMP_LIBRARY,
    AMBIGUOUS_POWERAMP_PATH,
    CUE_OR_SHARED_SOURCE,
    SOURCE_FILE_UNAVAILABLE,
    SOURCE_BYTES_MISMATCH,
}

internal enum class V2ServerBundleMatchEvidence {
    FULL_CONTENT_SHA256,
}

internal data class V2ServerBundleRowOutcome(
    val rootId: String,
    val relativePath: String,
    val disposition: V2ServerBundleRowDisposition,
    val detail: String,
    val matchEvidence: V2ServerBundleMatchEvidence? = null,
)

internal enum class V2ServerMergeStage {
    COPYING_BUNDLE,
    VALIDATING_BUNDLE,
    CLASSIFYING_EXISTING_ROWS,
    READING_POWERAMP_LIBRARY,
    MATCHING_TRACKS,
    VERIFYING_SOURCE_BYTES,
    COPYING_ACTIVE_INDEX,
    APPENDING_EMBEDDINGS,
    UPDATING_SIMILARITY_GRAPH,
    PUBLISHING_GENERATION,
    RECONCILING_LIBRARY,
}

internal data class V2ServerMergeProgress(
    val stage: V2ServerMergeStage,
    val detail: String,
    val completedUnits: Long? = null,
    val totalUnits: Long? = null,
)

internal data class V2ServerMergeResult(
    val generation: V2ResolvedActiveIndexGeneration,
    val sourceValidation: V2ServerBundleValidation,
    val rowOutcomes: List<V2ServerBundleRowOutcome>,
    val addedTrackCount: Int,
    val noOp: Boolean,
    val activeCatalog: V2ActiveLibraryCatalog,
)

/** Pure canonical-path rules, separated so suffix matching can be exercised without Android I/O. */
internal object V2ServerBundlePathPolicy {
    fun requireCanonicalRelativePath(value: String): String {
        require(value.isNotBlank() && value.length <= MAX_RELATIVE_PATH_CHARS) {
            "server relative path is blank or too long"
        }
        require(!value.startsWith('/') && '\\' !in value && '\u0000' !in value) {
            "server relative path is not POSIX-relative"
        }
        val normalized = TrackNormalization.normalizeNfc(value)
        require(normalized == value) { "server relative path is not NFC-normalized" }
        val segments = value.split('/')
        require(segments.all { segment ->
            segment.isNotEmpty() && segment != "." && segment != ".." &&
                segment.none(Char::isISOControl)
        }) { "server relative path contains an unsafe segment" }
        return value
    }

    fun physicalPathEndsWithRelativePath(physicalPath: String, relativePath: String): Boolean {
        val canonicalRelative = requireCanonicalRelativePath(relativePath)
        val normalizedPhysical = TrackNormalization.normalizeNfc(physicalPath)
        return normalizedPhysical == canonicalRelative ||
            normalizedPhysical.endsWith("/$canonicalRelative")
    }

    /** Python urllib.parse.quote(relativePath, safe="/") byte-for-byte. */
    fun encodeServerRelativePath(relativePath: String): String {
        val canonical = requireCanonicalRelativePath(relativePath)
        val bytes = canonical.toByteArray(StandardCharsets.UTF_8)
        return buildString(bytes.size) {
            bytes.forEach { raw ->
                val value = raw.toInt() and 0xff
                if (value in 'A'.code..'Z'.code ||
                    value in 'a'.code..'z'.code ||
                    value in '0'.code..'9'.code ||
                    value == '-'.code || value == '.'.code || value == '_'.code ||
                    value == '~'.code || value == '/'.code
                ) {
                    append(value.toChar())
                } else {
                    append('%')
                    append(HEX[value ushr 4])
                    append(HEX[value and 0x0f])
                }
            }
        }
    }

    private const val MAX_RELATIVE_PATH_CHARS = 8_192
    private const val HEX = "0123456789ABCDEF"
}

internal data class V2ServerBundleLocalSourceMatch(
    val canonicalFile: File,
    val observedSizeBytes: Long,
    val exactFingerprint: SourceFingerprint,
    val evidence: V2ServerBundleMatchEvidence,
)

/**
 * Relative path and byte length narrow the candidate set, but only exact content authorizes a
 * server embedding to bind to a Poweramp occurrence.
 */
internal object V2ServerBundleSourceMatchPolicy {
    fun matchExactSource(
        sourceFile: File,
        bundle: V2ServerBundleTrack,
        fingerprintSource: (File) -> SourceFingerprint?,
    ): V2ServerBundleLocalSourceMatch? {
        val canonicalFile = runCatching { sourceFile.canonicalFile }.getOrNull()
            ?: return null
        if (!canonicalFile.isFile || !canonicalFile.canRead() ||
            canonicalFile.length() != bundle.sourceSizeBytes
        ) {
            return null
        }
        val fingerprint = fingerprintSource(canonicalFile) ?: return null
        if (canonicalFile.length() != bundle.sourceSizeBytes ||
            fingerprint.sizeBytes != bundle.sourceSizeBytes ||
            fingerprint.lastModifiedEpochMs?.let { it != canonicalFile.lastModified() } == true ||
            fingerprint.fullContentSha256 != bundle.sourceSha256
        ) {
            return null
        }
        return V2ServerBundleLocalSourceMatch(
            canonicalFile = canonicalFile,
            observedSizeBytes = fingerprint.sizeBytes,
            exactFingerprint = fingerprint,
            evidence = V2ServerBundleMatchEvidence.FULL_CONTENT_SHA256,
        )
    }
}

internal object V2ServerBundleReciprocalAssignmentPolicy {
    fun reserveUnique(
        edges: Map<Long, Set<Long>>,
        alreadyReservedProviderIds: Set<Long> = emptySet(),
    ): Map<Long, Long> {
        val available = edges.mapValues { (_, providers) ->
            providers - alreadyReservedProviderIds
        }
        val providerDegree = available.values.flatten().groupingBy { it }.eachCount()
        return buildMap {
            available.forEach { (bundleTrackId, providers) ->
                val providerId = providers.singleOrNull() ?: return@forEach
                if (providerDegree[providerId] == 1) put(bundleTrackId, providerId)
            }
        }
    }
}

/** Validates every bundle claim before mutable staging or provider matching begins. */
internal object V2ServerBundleValidator {
    fun validate(
        databaseFile: File,
        onRowProgress: (completedRows: Int, totalRows: Int) -> Unit = { _, _ -> },
    ): V2ServerBundleValidation {
        require(databaseFile.isFile && databaseFile.length() > 0L) {
            "Selected server bundle is missing or empty"
        }
        val sourceSha256 = V2FileSha256.digest(databaseFile)
        val database = SQLiteDatabase.openDatabase(
            databaseFile.path,
            null,
            SQLiteDatabase.OPEN_READONLY,
        )
        try {
            requireIntegrity(database)
            requireColumns(database, V2ServerBundleContract.TRACK_TABLE, TRACK_COLUMNS)
            requireColumns(database, V2ServerBundleContract.EMBEDDING_TABLE, EMBEDDING_COLUMNS)
            requireColumns(database, V2ServerBundleContract.BUNDLE_TRACK_TABLE, BUNDLE_COLUMNS)
            requireColumns(database, V2ServerBundleContract.METADATA_TABLE, METADATA_COLUMNS)
            require(!hasTable(database, "clusters") && !hasTable(database, "binary_data")) {
                "Server bundle must not contain a similarity graph or cluster data"
            }

            val metadata = readMetadata(database)
            require(metadata.requireValue(V2ServerBundleContract.METADATA_BUNDLE_FORMAT) ==
                V2ServerBundleContract.BUNDLE_FORMAT
            ) { "Selected database is not a server embedding bundle" }
            require(metadata.requireValue(V2ServerBundleContract.METADATA_SCHEMA_VERSION).toInt() ==
                V2ServerBundleContract.SCHEMA_VERSION
            ) { "Server bundle schema version is unsupported" }
            val bundleId = metadata.requireValue(V2ServerBundleContract.METADATA_BUNDLE_ID)
            require(BUNDLE_ID.matches(bundleId)) { "Server bundle ID is invalid" }
            require(metadata.requireValue(V2ServerBundleContract.METADATA_EMBEDDING_SPEC_ID) ==
                V2ServerBundleContract.EMBEDDING_SPEC_ID
            ) { "Server bundle embedding policy is unsupported" }
            require(metadata.requireValue(V2ServerBundleContract.METADATA_OUTPUT_SPACE_ID) ==
                V2ServerBundleContract.OUTPUT_SPACE_ID
            ) { "Server bundle output space is incompatible with this library" }
            require(metadata.requireValue(V2ServerBundleContract.METADATA_MERT_MODEL_SHA256) ==
                V2ServerBundleContract.MERT_MODEL_SHA256
            ) { "Server bundle MERT artifact differs from the pinned model" }
            require(
                metadata.requireValue(
                    V2ServerBundleContract.METADATA_CLAMP3_AUDIO_MODEL_SHA256,
                ) == V2ServerBundleContract.CLAMP3_AUDIO_MODEL_SHA256,
            ) { "Server bundle CLaMP3 artifact differs from the pinned model" }
            val exactSpecJson = metadata.requireValue(
                V2ServerBundleContract.METADATA_EMBEDDING_SPEC_JSON,
            )
            require(exactSpecJson == V2ServerBundleContract.EMBEDDING_SPEC_JSON &&
                sha256(exactSpecJson.toByteArray(StandardCharsets.UTF_8)) ==
                V2ServerBundleContract.EMBEDDING_SPEC_SHA256 &&
                metadata.requireValue(
                    V2ServerBundleContract.METADATA_EMBEDDING_SPEC_SHA256,
                ) == V2ServerBundleContract.EMBEDDING_SPEC_SHA256
            ) { "Server bundle embedding spec JSON is not the exact canonical policy" }
            require(metadata.requireValue(V2ServerBundleContract.METADATA_GRAPH_INCLUDED) ==
                "false"
            ) { "Server transfer bundle must be graphless" }

            val trackCount = count(database, V2ServerBundleContract.TRACK_TABLE)
            val embeddingCount = count(database, V2ServerBundleContract.EMBEDDING_TABLE)
            val bundleTrackCount = count(database, V2ServerBundleContract.BUNDLE_TRACK_TABLE)
            val declaredTrackCount = metadata.requireValue(
                V2ServerBundleContract.METADATA_TRACK_COUNT,
            ).toInt()
            require(trackCount == embeddingCount && trackCount == bundleTrackCount &&
                trackCount == declaredTrackCount
            ) {
                "Server bundle row counts disagree: tracks=$trackCount, embeddings=" +
                    "$embeddingCount, provenance=$bundleTrackCount, declared=$declaredTrackCount"
            }
            require(trackCount >= 0) { "Server bundle track count is invalid" }
            require(scalarLong(
                database,
                """
                SELECT COUNT(*)
                FROM ${V2ServerBundleContract.BUNDLE_TRACK_TABLE} provenance
                LEFT JOIN ${V2ServerBundleContract.TRACK_TABLE} track
                  ON track.id = provenance.track_id
                LEFT JOIN ${V2ServerBundleContract.EMBEDDING_TABLE} embedding
                  ON embedding.track_id = provenance.track_id
                WHERE track.id IS NULL OR embedding.track_id IS NULL
                """.trimIndent(),
            ) == 0L) { "Server bundle contains orphaned track, provenance, or embedding rows" }

            val rows = ArrayList<V2ServerBundleTrack>(trackCount)
            val seenPaths = hashSetOf<Pair<String, String>>()
            val logicalIdDigest = MessageDigest.getInstance("SHA-256").apply {
                update("poweramp-server-bundle-v1\u0000".toByteArray(StandardCharsets.UTF_8))
                update(V2ServerBundleContract.EMBEDDING_SPEC_JSON.toByteArray(StandardCharsets.UTF_8))
                update(0.toByte())
            }
            onRowProgress(0, trackCount)
            database.rawQuery(
                """
                SELECT provenance.track_id, provenance.root_id, provenance.relative_path,
                       provenance.source_sha256, provenance.source_size_bytes,
                       provenance.source_sample_rate_hz, provenance.source_sample_count,
                       provenance.span_start_sample, provenance.span_end_sample_exclusive,
                       provenance.embedding_sha256, provenance.embedding_spec_id,
                       provenance.output_space_id, track.file_path, track.source,
                       track.metadata_key, track.filename_key, track.artist, track.album,
                       track.title, track.duration_ms, embedding.embedding
                FROM ${V2ServerBundleContract.BUNDLE_TRACK_TABLE} provenance
                JOIN ${V2ServerBundleContract.TRACK_TABLE} track
                  ON track.id = provenance.track_id
                JOIN ${V2ServerBundleContract.EMBEDDING_TABLE} embedding
                  ON embedding.track_id = provenance.track_id
                ORDER BY provenance.root_id, provenance.relative_path,
                         provenance.source_sha256, provenance.track_id
                """.trimIndent(),
                null,
            ).use { cursor ->
                while (cursor.moveToNext()) {
                    val trackId = cursor.getLong(0)
                    val rootId = cursor.getString(1)
                    val relativePath = V2ServerBundlePathPolicy.requireCanonicalRelativePath(
                        cursor.getString(2),
                    )
                    val sourceSha256Value = cursor.getString(3)
                    val sourceSizeBytes = cursor.getLong(4)
                    val sampleRate = cursor.getInt(5)
                    val sampleCount = cursor.getLong(6)
                    val spanStart = cursor.getLong(7)
                    val spanEnd = cursor.getLong(8)
                    val embeddingSha256 = cursor.getString(9)
                    val embeddingSpecId = cursor.getString(10)
                    val outputSpaceId = cursor.getString(11)
                    val serverFilePath = cursor.getString(12)
                    val source = cursor.getString(13)
                    val metadataKey = cursor.getString(14)
                    val filenameKey = cursor.getString(15)
                    val artist = cursor.getString(16)
                    val album = cursor.getString(17)
                    val title = cursor.getString(18)
                    val durationMs = cursor.getLong(19)
                    val embedding = cursor.getBlob(20)
                    require(trackId > 0L) { "Server bundle contains a non-positive track ID" }
                    require(ROOT_ID.matches(rootId)) { "Server root ID is invalid for track $trackId" }
                    require(seenPaths.add(rootId to relativePath)) {
                        "Server bundle repeats $rootId/$relativePath"
                    }
                    require(SHA256.matches(sourceSha256Value) && sourceSizeBytes > 0L) {
                        "Server source identity is invalid for $rootId/$relativePath"
                    }
                    require(sampleRate in 1..MAX_SAMPLE_RATE_HZ && sampleCount > 0L &&
                        spanStart == 0L && spanEnd == sampleCount
                    ) { "Server row is not a full ordinary-file span: $rootId/$relativePath" }
                    require(embeddingSpecId == V2ServerBundleContract.EMBEDDING_SPEC_ID &&
                        outputSpaceId == V2ServerBundleContract.OUTPUT_SPACE_ID
                    ) { "Server row policy differs from its bundle metadata" }
                    V2Clamp3VectorCodec.requireValidBlob(embedding)
                    require(SHA256.matches(embeddingSha256) &&
                        sha256(embedding) == embeddingSha256
                    ) { "Server embedding hash is invalid for $rootId/$relativePath" }
                    requireServerUri(serverFilePath, rootId, relativePath)
                    require(source == "server") { "Server track source claim is invalid" }
                    logicalIdDigest.update(
                        logicalIdRecord(
                            album = album,
                            artist = artist,
                            durationMs = durationMs,
                            embeddingSha256 = embeddingSha256,
                            filenameKey = filenameKey,
                            filePath = serverFilePath,
                            metadataKey = metadataKey,
                            relativePath = relativePath,
                            rootId = rootId,
                            sourceSampleCount = sampleCount,
                            sourceSampleRateHz = sampleRate,
                            sourceSha256 = sourceSha256Value,
                            sourceSizeBytes = sourceSizeBytes,
                            title = title,
                        ).toByteArray(StandardCharsets.UTF_8),
                    )
                    logicalIdDigest.update('\n'.code.toByte())
                    rows += V2ServerBundleTrack(
                        trackId = trackId,
                        rootId = rootId,
                        relativePath = relativePath,
                        sourceSha256 = sourceSha256Value,
                        sourceSizeBytes = sourceSizeBytes,
                        sourceSampleRateHz = sampleRate,
                        sourceSampleCount = sampleCount,
                        spanStartSample = spanStart,
                        spanEndSampleExclusive = spanEnd,
                        embeddingSha256 = embeddingSha256,
                    )
                    if (rows.size == trackCount || rows.size % ROW_PROGRESS_INTERVAL == 0) {
                        onRowProgress(rows.size, trackCount)
                    }
                }
            }
            require(rows.size == trackCount) { "Server bundle changed while it was validated" }
            val expectedBundleId = "server-bundle-v1-${logicalIdDigest.digest().toHex()}"
            require(bundleId == expectedBundleId) {
                "Server bundle logical ID does not match its ordered rows and canonical spec"
            }
            requireIntegrity(database)
            return V2ServerBundleValidation(
                sourceByteLength = databaseFile.length(),
                sourceSha256 = sourceSha256,
                bundleId = bundleId,
                tracks = rows,
            )
        } finally {
            database.close()
        }
    }

    private fun requireServerUri(value: String, rootId: String, relativePath: String) {
        val expected = "server://$rootId/" +
            V2ServerBundlePathPolicy.encodeServerRelativePath(relativePath)
        val uri = Uri.parse(value)
        require(uri.scheme == "server" && uri.authority == rootId &&
            uri.query == null && uri.fragment == null && uri.userInfo == null &&
            uri.path?.removePrefix("/") == relativePath &&
            value == expected
        ) { "Server file path is not bound to its root and relative path" }
    }

    internal fun logicalIdRecord(
        album: String?,
        artist: String?,
        durationMs: Long,
        embeddingSha256: String,
        filenameKey: String,
        filePath: String,
        metadataKey: String,
        relativePath: String,
        rootId: String,
        sourceSampleCount: Long,
        sourceSampleRateHz: Int,
        sourceSha256: String,
        sourceSizeBytes: Long,
        title: String?,
    ): String = buildString {
        append("{\"album\":"); appendJsonStringOrNull(album)
        append(",\"artist\":"); appendJsonStringOrNull(artist)
        append(",\"duration_ms\":"); append(durationMs)
        append(",\"embedding_sha256\":"); appendJsonString(embeddingSha256)
        append(",\"file_path\":"); appendJsonString(filePath)
        append(",\"filename_key\":"); appendJsonString(filenameKey)
        append(",\"metadata_key\":"); appendJsonString(metadataKey)
        append(",\"relative_path\":"); appendJsonString(relativePath)
        append(",\"root_id\":"); appendJsonString(rootId)
        append(",\"source_sample_count\":"); append(sourceSampleCount)
        append(",\"source_sample_rate_hz\":"); append(sourceSampleRateHz)
        append(",\"source_sha256\":"); appendJsonString(sourceSha256)
        append(",\"source_size_bytes\":"); append(sourceSizeBytes)
        append(",\"title\":"); appendJsonStringOrNull(title)
        append('}')
    }

    internal fun logicalBundleId(canonicalRecords: List<String>): String {
        val digest = MessageDigest.getInstance("SHA-256")
        digest.update("poweramp-server-bundle-v1\u0000".toByteArray(StandardCharsets.UTF_8))
        digest.update(V2ServerBundleContract.EMBEDDING_SPEC_JSON.toByteArray(StandardCharsets.UTF_8))
        digest.update(0.toByte())
        canonicalRecords.forEach { record ->
            digest.update(record.toByteArray(StandardCharsets.UTF_8))
            digest.update('\n'.code.toByte())
        }
        return "server-bundle-v1-${digest.digest().toHex()}"
    }

    private fun StringBuilder.appendJsonStringOrNull(value: String?) {
        if (value == null) append("null") else appendJsonString(value)
    }

    /** Matches Python json.dumps(..., ensure_ascii=True, separators=(",", ":")). */
    private fun StringBuilder.appendJsonString(value: String) {
        append('"')
        value.forEach { character ->
            when (character) {
                '"' -> append("\\\"")
                '\\' -> append("\\\\")
                '\b' -> append("\\b")
                '\t' -> append("\\t")
                '\n' -> append("\\n")
                '\u000c' -> append("\\f")
                '\r' -> append("\\r")
                else -> if (character.code < 0x20 || character.code > 0x7e) {
                    append("\\u")
                    append(character.code.toString(16).padStart(4, '0'))
                } else {
                    append(character)
                }
            }
        }
        append('"')
    }

    private fun readMetadata(database: SQLiteDatabase): Map<String, String> = buildMap {
        database.rawQuery(
            "SELECT key, value FROM ${V2ServerBundleContract.METADATA_TABLE}",
            null,
        ).use { cursor ->
            while (cursor.moveToNext()) {
                val key = cursor.getString(0)
                require(put(key, cursor.getString(1)) == null) {
                    "Server bundle repeats metadata key $key"
                }
            }
        }
    }

    private fun Map<String, String>.requireValue(key: String): String =
        get(key)?.takeIf(String::isNotBlank)
            ?: throw IllegalArgumentException("Server bundle metadata is missing $key")

    private fun requireColumns(
        database: SQLiteDatabase,
        table: String,
        requiredColumns: Map<String, String>,
    ) {
        require(database.rawQuery(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1",
            arrayOf(table),
        ).use { it.moveToFirst() }) { "Server bundle table is missing: $table" }
        val actual = linkedMapOf<String, String>()
        database.rawQuery("PRAGMA table_info([$table])", null).use { cursor ->
            while (cursor.moveToNext()) actual[cursor.getString(1)] = cursor.getString(2).uppercase()
        }
        requiredColumns.forEach { (name, affinity) ->
            require(actual[name]?.contains(affinity) == true) {
                "Server bundle column is missing or incompatible: $table.$name"
            }
        }
    }

    private fun hasTable(database: SQLiteDatabase, table: String): Boolean = database.rawQuery(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1",
        arrayOf(table),
    ).use { it.moveToFirst() }

    private fun count(database: SQLiteDatabase, table: String): Int =
        scalarLong(database, "SELECT COUNT(*) FROM [$table]").also { count ->
            require(count in 0..MAX_TRACK_COUNT.toLong()) { "Server bundle is too large" }
        }.toInt()

    private fun scalarLong(database: SQLiteDatabase, sql: String): Long =
        database.rawQuery(sql, null).use { cursor ->
            check(cursor.moveToFirst()) { "Server bundle scalar query returned no row" }
            cursor.getLong(0)
        }

    private fun requireIntegrity(database: SQLiteDatabase) {
        database.rawQuery("PRAGMA integrity_check", null).use { cursor ->
            require(cursor.moveToFirst() && cursor.getString(0) == "ok" && !cursor.moveToNext()) {
                "Server bundle SQLite integrity check failed"
            }
        }
    }

    private fun sha256(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256")
        .digest(bytes)
        .toHex()

    private fun ByteArray.toHex(): String = joinToString("") { byte ->
        (byte.toInt() and 0xff).toString(16).padStart(2, '0')
    }

    private val TRACK_COLUMNS = mapOf(
        "id" to "INTEGER",
        "metadata_key" to "TEXT",
        "filename_key" to "TEXT",
        "artist" to "TEXT",
        "album" to "TEXT",
        "title" to "TEXT",
        "duration_ms" to "INTEGER",
        "file_path" to "TEXT",
        "source" to "TEXT",
    )
    private val EMBEDDING_COLUMNS = mapOf("track_id" to "INTEGER", "embedding" to "BLOB")
    private val BUNDLE_COLUMNS = mapOf(
        "track_id" to "INTEGER",
        "root_id" to "TEXT",
        "relative_path" to "TEXT",
        "source_sha256" to "TEXT",
        "source_size_bytes" to "INTEGER",
        "source_sample_rate_hz" to "INTEGER",
        "source_sample_count" to "INTEGER",
        "span_start_sample" to "INTEGER",
        "span_end_sample_exclusive" to "INTEGER",
        "embedding_sha256" to "TEXT",
        "embedding_spec_id" to "TEXT",
        "output_space_id" to "TEXT",
    )
    private val METADATA_COLUMNS = mapOf("key" to "TEXT", "value" to "TEXT")
    private val SHA256 = Regex("^[0-9a-f]{64}$")
    private val ROOT_ID = Regex("^[A-Za-z0-9._-]{1,128}$")
    private val BUNDLE_ID = Regex("^[A-Za-z0-9._:-]{1,256}$")
    private const val MAX_TRACK_COUNT = 5_000_000
    private const val MAX_SAMPLE_RATE_HZ = 768_000
    private const val ROW_PROGRESS_INTERVAL = 512
}

/**
 * Merges server rows bound one-to-one to byte-identical ordinary Poweramp files. The synced-tree
 * path narrows the normal case to one source read; moved or ambiguous paths inspect only same-name,
 * same-size candidates. The active generation is immutable; pointer publication is the sole commit
 * point.
 */
internal class V2ServerBundleMerger(
    context: Context,
    private val filesDir: File = context.filesDir,
    private val publisher: V2IndexGenerationPublisher = V2IndexGenerationPublisher(filesDir),
    private val providerAcquirer: V2PowerampProviderSnapshotAcquirer =
        V2PowerampProviderSnapshotAcquirer(context),
    private val sourceFingerprinter: V2SourceFingerprintProvider = V2ExactSourceFingerprinter(),
) {
    private val appContext = context.applicationContext
    private val mergeRoot = File(filesDir, "indexing_v2/server-merges")
    private val executorLeases = V2ExecutorLeaseCoordinator(
        V2AtomicExecutorLeasePersistence(filesDir),
    )
    private val ownerInstanceId = "server-merge-${UUID.randomUUID()}"

    suspend fun merge(
        uri: Uri,
        onProgress: (V2ServerMergeProgress) -> Unit = {},
    ): V2ServerMergeResult {
        require(AudioLibraryPermission.isGranted(appContext)) {
            AudioLibraryPermission.DENIED_MESSAGE
        }
        return V2ProcessLibraryInspectionCoordinator.inspect {
            withContext(Dispatchers.IO) {
                val length = runCatching {
                    appContext.contentResolver.openAssetFileDescriptor(uri, "r")
                        ?.use { descriptor ->
                            descriptor.length.takeIf { it >= 0L }
                        }
                }.getOrNull()
                mergeBlocking(
                    sourceLength = length,
                    openSource = {
                        appContext.contentResolver.openInputStream(uri)
                            ?: throw IllegalArgumentException(
                                "Cannot open server bundle document",
                            )
                    },
                    onProgress = onProgress,
                )
            }
        }
    }

    internal fun mergeFileBlocking(
        source: File,
        onProgress: (V2ServerMergeProgress) -> Unit = {},
    ): V2ServerMergeResult = mergeBlocking(
        sourceLength = source.length(),
        openSource = { FileInputStream(source) },
        onProgress = onProgress,
    )

    private fun mergeBlocking(
        sourceLength: Long?,
        openSource: () -> InputStream,
        onProgress: (V2ServerMergeProgress) -> Unit,
    ): V2ServerMergeResult = synchronized(MERGE_PROCESS_LOCK) {
        val lease = executorLeases.claim(SERVER_MERGE_LEASE_JOB_ID, ownerInstanceId)
        try {
            mergeWithLeaseBlocking(sourceLength, openSource, onProgress)
        } finally {
            executorLeases.release(lease)
        }
    }

    private fun mergeWithLeaseBlocking(
        sourceLength: Long?,
        openSource: () -> InputStream,
        onProgress: (V2ServerMergeProgress) -> Unit,
    ): V2ServerMergeResult {
        val base = V2LibraryDatabaseResolver.requirePublished(filesDir)
        val expectedActive = V2GenerationPublicationCoordinator.capture(filesDir)
        require(expectedActive.pointer?.generationId == base.manifest.generationId &&
            expectedActive.pointer.manifestSha256 == base.manifestSha256
        ) { "The active music index changed before server merge began" }
        require(mergeRoot.isDirectory || mergeRoot.mkdirs()) {
            "Cannot create private server-merge staging"
        }
        removeAbandonedStaging()
        syncDirectory(mergeRoot.parentFile ?: filesDir)
        val staging = File(mergeRoot, ".staging-${UUID.randomUUID()}")
        require(staging.mkdir()) { "Cannot create private server-merge operation" }
        syncDirectory(mergeRoot)
        try {
            val bundleDatabase = File(staging, "server-bundle.db")
            publish(
                onProgress,
                V2ServerMergeStage.COPYING_BUNDLE,
                "Copying the selected server embedding bundle into private staging",
                0L,
                sourceLength,
            )
            copyAndSync(openSource, bundleDatabase, sourceLength) { copied, total ->
                publish(
                    onProgress,
                    V2ServerMergeStage.COPYING_BUNDLE,
                    "Copied $copied of ${total ?: copied} server-bundle bytes",
                    copied,
                    total,
                )
            }
            publish(
                onProgress,
                V2ServerMergeStage.VALIDATING_BUNDLE,
                "Validating server model, source-span, and embedding evidence",
            )
            val validation = V2ServerBundleValidator.validate(bundleDatabase) { completed, total ->
                publish(
                    onProgress,
                    V2ServerMergeStage.VALIDATING_BUNDLE,
                    "Validated $completed of $total server embedding rows",
                    completed.toLong(),
                    total.toLong(),
                )
            }

            val mergeReceipts = readServerMergeReceipts(base.databaseFile)
            val previouslyMerged = mutableListOf<V2ServerBundleTrack>()
            val unresolvedTracks = mutableListOf<V2ServerBundleTrack>()
            validation.tracks.forEach { bundle ->
                val receipt = stableReceiptFor(bundle, mergeReceipts)
                if (receipt == null) {
                    unresolvedTracks += bundle
                } else {
                    require(receipt.embeddingSha256 == bundle.embeddingSha256) {
                        "Pinned server policy produced different embeddings for the same source"
                    }
                    previouslyMerged += bundle
                }
            }
            val priorOutcomes = previouslyMerged.map { bundle ->
                bundle.outcome(
                    V2ServerBundleRowDisposition.ALREADY_INDEXED,
                    "This exact server embedding already has a durable merge receipt",
                )
            }
            publish(
                onProgress,
                V2ServerMergeStage.CLASSIFYING_EXISTING_ROWS,
                "Classified ${validation.tracks.size} server rows; " +
                    "${previouslyMerged.size} already have exact merge receipts",
                validation.tracks.size.toLong(),
                validation.tracks.size.toLong(),
            )
            if (unresolvedTracks.isEmpty()) {
                return V2ServerMergeResult(
                    generation = base,
                    sourceValidation = validation,
                    rowOutcomes = priorOutcomes.sortedByOutcome(),
                    addedTrackCount = 0,
                    noOp = true,
                    activeCatalog = requireCurrentCatalog(base, onProgress),
                )
            }

            publish(
                onProgress,
                V2ServerMergeStage.READING_POWERAMP_LIBRARY,
                "Reading the complete current Poweramp library",
            )
            val providerSnapshot = providerAcquirer.acquireBlocking { completed, total ->
                publish(
                    onProgress,
                    V2ServerMergeStage.READING_POWERAMP_LIBRARY,
                    "Read $completed of $total Poweramp rows",
                    completed.toLong(),
                    total.toLong(),
                )
            }
            val baseCatalog = V2ActiveLibraryCatalogLoader.load(base, providerSnapshot)
            publish(
                onProgress,
                V2ServerMergeStage.MATCHING_TRACKS,
                "Matching synced relative paths and verifying candidate bytes with SHA-256",
                0L,
                unresolvedTracks.size.toLong(),
            )
            val resolved = resolveRows(
                tracks = unresolvedTracks,
                providerSnapshot = providerSnapshot,
                baseCatalog = baseCatalog,
                indexedPaths = readIndexedPaths(base.databaseFile),
                mergeReceipts = mergeReceipts,
                onProgress = onProgress,
            )
            val prePublicationOutcomes = priorOutcomes + resolved.rejected
            if (resolved.accepted.isEmpty()) {
                return V2ServerMergeResult(
                    generation = base,
                    sourceValidation = validation,
                    rowOutcomes = prePublicationOutcomes.sortedByOutcome(),
                    addedTrackCount = 0,
                    noOp = true,
                    activeCatalog = baseCatalog,
                )
            }

            V2GenerationMutationStoragePolicy.requireCapacity(
                V2GenerationMutationStoragePolicy.estimateServerMerge(
                    active = base,
                    bundleBytes = bundleDatabase.length(),
                    availableBytes = filesDir.usableSpace,
                ),
                operation = "Server embedding merge",
            )
            val stagedDatabase = File(staging, "merged-library.db")
            publish(
                onProgress,
                V2ServerMergeStage.COPYING_ACTIVE_INDEX,
                "Copying the active music index into private staging",
            )
            snapshotDatabase(base, stagedDatabase) { copied, total ->
                publish(
                    onProgress,
                    V2ServerMergeStage.COPYING_ACTIVE_INDEX,
                    "Copied $copied of $total active-index bytes",
                    copied,
                    total,
                )
            }
            publish(
                onProgress,
                V2ServerMergeStage.APPENDING_EMBEDDINGS,
                "Appending ${resolved.accepted.size} exact server embeddings",
                0L,
                resolved.accepted.size.toLong(),
            )
            val stagedAdditions = appendRows(
                stagedDatabase = stagedDatabase,
                bundleDatabase = bundleDatabase,
                bundleId = validation.bundleId,
                rows = resolved.accepted,
                onProgress = onProgress,
            )

            val stagedCatalog = V2ActiveLibraryCatalogLoader.load(
                activeGeneration = base.copy(
                    manifest = base.manifest.copy(
                        generationId = "server-merge-staging",
                        trackCount = base.manifest.trackCount + stagedAdditions.size,
                    ),
                    databaseFile = stagedDatabase,
                ),
                providerSnapshot = providerSnapshot,
            )
            requireAddedBindings(stagedCatalog, stagedAdditions)
            requireInheritedBindingsPreserved(baseCatalog, stagedCatalog)
            val graphFile = updateExactGraph(
                base = base,
                stagedDatabase = stagedDatabase,
                staging = staging,
                onProgress = onProgress,
            )
            requireAcceptedSourcesStillCurrent(resolved.accepted, onProgress)
            publish(
                onProgress,
                V2ServerMergeStage.PUBLISHING_GENERATION,
                "Hashing and atomically publishing the merged music index",
            )
            val generation = publisher.publishServerMerge(
                privateStagingDatabase = stagedDatabase,
                baseGeneration = base,
                bundleDatabaseSha256 = validation.sourceSha256,
                addedTrackCount = resolved.accepted.size,
                exactUpdatedGraphFile = graphFile,
                expectedActive = expectedActive,
            )
            require(generation.manifest.origin == V2IndexGenerationOrigin.SERVER_MERGE &&
                generation.manifest.baseGenerationId == base.manifest.generationId &&
                generation.manifest.trackCount == base.manifest.trackCount + resolved.accepted.size
            ) { "Published generation is not the exact server append" }

            publish(
                onProgress,
                V2ServerMergeStage.RECONCILING_LIBRARY,
                "Binding the merged generation to the same Poweramp snapshot",
            )
            val catalog = V2ActiveLibraryCatalogLoader.load(generation, providerSnapshot)
            requireAddedBindings(catalog, stagedAdditions)
            V2ActiveLibraryCatalogStore(filesDir).write(generation, catalog)
            val added = resolved.accepted.map { accepted ->
                V2ServerBundleRowOutcome(
                    rootId = accepted.bundle.rootId,
                    relativePath = accepted.bundle.relativePath,
                    disposition = V2ServerBundleRowDisposition.ADDED,
                    detail = when (accepted.sourceMatch.evidence) {
                        V2ServerBundleMatchEvidence.FULL_CONTENT_SHA256 ->
                            "Full source SHA-256 matched Poweramp file " +
                                accepted.provider.powerampFileId
                    },
                    matchEvidence = accepted.sourceMatch.evidence,
                )
            }
            return V2ServerMergeResult(
                generation = generation,
                sourceValidation = validation,
                rowOutcomes = (prePublicationOutcomes + added).sortedByOutcome(),
                addedTrackCount = resolved.accepted.size,
                noOp = false,
                activeCatalog = catalog,
            )
        } finally {
            staging.deleteRecursively()
            syncDirectory(mergeRoot)
        }
    }

    private data class VerifiedCandidate(
        val bundle: V2ServerBundleTrack,
        val provider: V2ProviderPathRowEvidence,
        val sourceMatch: V2ServerBundleLocalSourceMatch,
    )

    private data class MatchedProvider(
        val providerFile: ProviderFile,
        val sourceMatch: V2ServerBundleLocalSourceMatch,
    )

    private data class StagedAddition(
        val accepted: VerifiedCandidate,
        val trackId: Long,
    )

    private data class ResolutionResult(
        val accepted: List<VerifiedCandidate>,
        val rejected: List<V2ServerBundleRowOutcome>,
    )

    private data class ProviderFile(
        val row: V2ProviderPathRowEvidence,
        val file: File,
    )

    private data class ExistingMergeReceipt(
        val rootId: String,
        val relativePath: String,
        val sourceSha256: String,
        val sourceSizeBytes: Long,
        val embeddingSha256: String,
        val embeddingSpecId: String,
        val outputSpaceId: String,
        val providerPhysicalPath: String,
    )

    private fun resolveRows(
        tracks: List<V2ServerBundleTrack>,
        providerSnapshot: V2ProviderPathGroupSnapshot,
        baseCatalog: V2ActiveLibraryCatalog,
        indexedPaths: Set<String>,
        mergeReceipts: Set<ExistingMergeReceipt>,
        onProgress: (V2ServerMergeProgress) -> Unit,
    ): ResolutionResult {
        val rejected = mutableListOf<V2ServerBundleRowOutcome>()
        val unresolvedTracks = tracks

        val eligibleRows = providerSnapshot.groups.filter { group ->
            group.completeness == V2ProviderPathGroupCompleteness.COMPLETE &&
                group.rows.size == 1 &&
                group.rows.single().offsetMs == 0L &&
                group.rows.single().cueSourceImageFolderId == null
        }.map { group ->
            val row = group.rows.single()
            ProviderFile(
                row = row,
                file = File(row.providerPhysicalPath),
            )
        }
        val groupsByFilename = providerSnapshot.groups.groupBy { group ->
            group.physicalPath.substringAfterLast('/')
        }
        val eligibleByFilename = eligibleRows.groupBy { candidate ->
            candidate.row.physicalPath.substringAfterLast('/')
        }
        val fingerprintCache = hashMapOf<String, SourceFingerprint?>()

        fun exactContentMatches(
            candidates: Iterable<ProviderFile>,
            bundle: V2ServerBundleTrack,
            trackIndex: Int,
        ): List<MatchedProvider> = candidates.mapNotNull { candidate ->
            val file = runCatching { candidate.file.canonicalFile }.getOrNull()
                ?: return@mapNotNull null
            val sourceMatch = V2ServerBundleSourceMatchPolicy.matchExactSource(
                sourceFile = file,
                bundle = bundle,
            ) { exactFile ->
                fingerprintCache.getOrPut(exactFile.path) {
                    runCatching {
                        sourceFingerprinter.fingerprint(
                            exactFile,
                        ) { completedBytes, totalBytes ->
                            publish(
                                onProgress,
                                V2ServerMergeStage.VERIFYING_SOURCE_BYTES,
                                "Verifying server track ${trackIndex + 1} of " +
                                    "${unresolvedTracks.size}: " +
                                    "$completedBytes of $totalBytes bytes\n" +
                                    candidate.row.providerPhysicalPath,
                                completedBytes,
                                totalBytes,
                            )
                        }
                    }.getOrNull()
                }
            } ?: return@mapNotNull null
            MatchedProvider(candidate, sourceMatch)
        }

        data class Prepared(
            val bundle: V2ServerBundleTrack,
            val suffixGroups: List<V2ProviderPathGroupEvidence>,
            val matches: List<MatchedProvider>,
        )

        val prepared = unresolvedTracks.mapIndexedNotNull { index, bundle ->
            publish(
                onProgress,
                V2ServerMergeStage.MATCHING_TRACKS,
                "Matching server track ${index + 1} of ${unresolvedTracks.size} " +
                    "by its synced relative path",
                index.toLong(),
                unresolvedTracks.size.toLong(),
            )
            val filename = bundle.relativePath.substringAfterLast('/')
            val suffixGroups = groupsByFilename[filename].orEmpty().filter { group ->
                V2ServerBundlePathPolicy.physicalPathEndsWithRelativePath(
                    group.physicalPath,
                    bundle.relativePath,
                )
            }
            val unsafeSuffix = suffixGroups.any { group ->
                group.completeness != V2ProviderPathGroupCompleteness.COMPLETE ||
                    group.rows.size != 1 ||
                    group.rows.single().offsetMs != 0L ||
                    group.rows.single().cueSourceImageFolderId != null
            }
            if (unsafeSuffix) {
                rejected += bundle.outcome(
                    V2ServerBundleRowDisposition.CUE_OR_SHARED_SOURCE,
                    "The relative path resolves to a shared or logical CUE source",
                )
                return@mapIndexedNotNull null
            }
            val uniqueOrdinarySuffix = suffixGroups.singleOrNull()?.takeIf { group ->
                group.completeness == V2ProviderPathGroupCompleteness.COMPLETE &&
                    group.rows.size == 1 && group.rows.single().offsetMs == 0L &&
                    group.rows.single().cueSourceImageFolderId == null
            }?.rows?.single()
            if (uniqueOrdinarySuffix != null) {
                val normalizedProviderPath = requireNotNull(
                    TrackNormalization.normalizePath(uniqueOrdinarySuffix.providerPhysicalPath),
                )
                val providerReceipt = providerReceiptFor(
                    bundle = bundle,
                    normalizedProviderPath = normalizedProviderPath,
                    receipts = mergeReceipts,
                )
                if (providerReceipt != null) {
                    require(providerReceipt.embeddingSha256 == bundle.embeddingSha256) {
                        "Pinned server policy produced different embeddings for the same source"
                    }
                    rejected += bundle.outcome(
                        V2ServerBundleRowDisposition.ALREADY_INDEXED,
                        "This exact source and Poweramp occurrence already have a merge receipt",
                    )
                    return@mapIndexedNotNull null
                }
                when {
                    baseCatalog.trackIdForPowerampFile(uniqueOrdinarySuffix.powerampFileId) != null -> {
                        rejected += bundle.outcome(
                            V2ServerBundleRowDisposition.ALREADY_INDEXED,
                            "The unique Poweramp occurrence is already bound to an embedding",
                        )
                        return@mapIndexedNotNull null
                    }
                    normalizedProviderPath in indexedPaths -> {
                        rejected += bundle.outcome(
                            V2ServerBundleRowDisposition.AMBIGUOUS_POWERAMP_PATH,
                            "An unbound indexed row already claims this exact Poweramp path",
                        )
                        return@mapIndexedNotNull null
                    }
                    uniqueOrdinarySuffix.powerampFileId !in baseCatalog.unboundPowerampFileIds -> {
                        rejected += bundle.outcome(
                            V2ServerBundleRowDisposition.AMBIGUOUS_POWERAMP_PATH,
                            "Poweramp file is reserved by unresolved active-library evidence",
                        )
                        return@mapIndexedNotNull null
                    }
                }
            }
            val sameNameCandidates = eligibleByFilename[filename].orEmpty()
            val suffixCandidates = sameNameCandidates.filter { candidate ->
                V2ServerBundlePathPolicy.physicalPathEndsWithRelativePath(
                    candidate.row.physicalPath,
                    bundle.relativePath,
                )
            }
            val exactSuffixMatches = suffixCandidates.singleOrNull()?.let { candidate ->
                exactContentMatches(
                    candidates = listOf(candidate),
                    bundle = bundle,
                    trackIndex = index,
                )
            }.orEmpty()
            val matches = if (exactSuffixMatches.isNotEmpty()) {
                exactSuffixMatches
            } else {
                // Try the unique synced path first. Only if its bytes differ or the path is
                // ambiguous do we inspect same-name, same-size files for a moved exact source.
                exactContentMatches(
                    candidates = sameNameCandidates,
                    bundle = bundle,
                    trackIndex = index,
                )
            }
            Prepared(
                bundle = bundle,
                suffixGroups = suffixGroups,
                matches = matches,
            )
        }

        val assignments = linkedMapOf<V2ServerBundleTrack, MatchedProvider>()
        val reservedProviderIds = hashSetOf<Long>()
        val providerMatchesById = prepared
            .flatMap(Prepared::matches)
            .associateBy { it.providerFile.row.powerampFileId }

        fun assignReciprocalUnique(edges: Map<V2ServerBundleTrack, List<MatchedProvider>>) {
            val byTrackId = edges.mapKeys { (bundle, _) -> bundle.trackId }
                .mapValues { (_, candidates) ->
                    candidates.mapTo(linkedSetOf()) { it.providerFile.row.powerampFileId }
                }
            val bundleByTrackId = edges.keys.associateBy(V2ServerBundleTrack::trackId)
            V2ServerBundleReciprocalAssignmentPolicy.reserveUnique(
                edges = byTrackId,
                alreadyReservedProviderIds = reservedProviderIds,
            ).forEach { (bundleTrackId, providerId) ->
                val bundle = checkNotNull(bundleByTrackId[bundleTrackId])
                assignments[bundle] = checkNotNull(providerMatchesById[providerId])
                check(reservedProviderIds.add(providerId))
            }
        }

        assignReciprocalUnique(prepared.associate { it.bundle to it.matches })
        val suffixProviderDegree = prepared
            .flatMap { item -> item.matches.map { it.providerFile.row.powerampFileId } }
            .groupingBy { it }
            .eachCount()

        val accepted = mutableListOf<VerifiedCandidate>()
        prepared.forEach { item ->
            val bundle = item.bundle
            val matchedProvider = assignments[bundle]
            if (matchedProvider == null) {
                val contendedExactPath = item.matches.any { candidate ->
                    suffixProviderDegree[candidate.providerFile.row.powerampFileId] != 1
                }
                rejected += bundle.outcome(
                    disposition = when {
                        item.matches.size > 1 || contendedExactPath ->
                            V2ServerBundleRowDisposition.AMBIGUOUS_POWERAMP_PATH
                        item.suffixGroups.isEmpty() ->
                            V2ServerBundleRowDisposition.NOT_IN_POWERAMP_LIBRARY
                        item.suffixGroups.any { group ->
                            group.rows.any { row ->
                                val file = runCatching {
                                    File(row.providerPhysicalPath).canonicalFile
                                }.getOrNull()
                                file == null || !file.isFile || !file.canRead()
                            }
                        } -> V2ServerBundleRowDisposition.SOURCE_FILE_UNAVAILABLE
                        else -> V2ServerBundleRowDisposition.SOURCE_BYTES_MISMATCH
                    },
                    detail = if (item.matches.size > 1 || contendedExactPath) {
                        "Source evidence does not form a reciprocal one-to-one Poweramp match"
                    } else if (item.suffixGroups.isEmpty()) {
                        "No same-name Poweramp file has the server source bytes"
                    } else {
                        "The Poweramp file at the synced relative path has a different byte length " +
                            "or source SHA-256"
                    },
                )
                return@forEach
            }
            val providerFile = matchedProvider.providerFile
            val provider = providerFile.row
            val normalizedProviderPath = requireNotNull(
                TrackNormalization.normalizePath(provider.providerPhysicalPath),
            )
            when {
                baseCatalog.trackIdForPowerampFile(provider.powerampFileId) != null ->
                    rejected += bundle.outcome(
                        V2ServerBundleRowDisposition.ALREADY_INDEXED,
                        "Poweramp file ${provider.powerampFileId} is already bound to an embedding",
                    )
                normalizedProviderPath in indexedPaths -> rejected += bundle.outcome(
                    V2ServerBundleRowDisposition.AMBIGUOUS_POWERAMP_PATH,
                    "An unbound indexed row already claims this exact Poweramp path",
                )
                provider.powerampFileId !in baseCatalog.unboundPowerampFileIds ->
                    rejected += bundle.outcome(
                        V2ServerBundleRowDisposition.AMBIGUOUS_POWERAMP_PATH,
                        "Poweramp file is reserved by unresolved active-library evidence",
                    )
                else -> {
                    accepted += VerifiedCandidate(
                        bundle = bundle,
                        provider = provider,
                        sourceMatch = matchedProvider.sourceMatch,
                    )
                }
            }
        }
        publish(
            onProgress,
            V2ServerMergeStage.MATCHING_TRACKS,
            "Resolved ${unresolvedTracks.size} of ${unresolvedTracks.size} server sources",
            unresolvedTracks.size.toLong(),
            unresolvedTracks.size.toLong(),
        )
        return ResolutionResult(accepted, rejected)
    }

    private fun appendRows(
        stagedDatabase: File,
        bundleDatabase: File,
        bundleId: String,
        rows: List<VerifiedCandidate>,
        onProgress: (V2ServerMergeProgress) -> Unit,
    ): List<StagedAddition> {
        val source = SQLiteDatabase.openDatabase(
            bundleDatabase.path,
            null,
            SQLiteDatabase.OPEN_READONLY,
        )
        val target = SQLiteDatabase.openDatabase(
            stagedDatabase.path,
            null,
            SQLiteDatabase.OPEN_READWRITE,
        )
        val additions = ArrayList<StagedAddition>(rows.size)
        try {
            target.disableWriteAheadLogging()
            target.beginTransaction()
            try {
                createServerMergeReceiptTable(target)
                rows.forEachIndexed { index, accepted ->
                    val blob = source.rawQuery(
                        "SELECT embedding FROM ${V2ServerBundleContract.EMBEDDING_TABLE} " +
                            "WHERE track_id = ?",
                        arrayOf(accepted.bundle.trackId.toString()),
                    ).use { cursor ->
                        require(cursor.moveToFirst()) { "Validated server embedding disappeared" }
                        cursor.getBlob(0).also {
                            require(!cursor.moveToNext()) { "Server embedding ID became ambiguous" }
                        }
                    }
                    V2Clamp3VectorCodec.requireValidBlob(blob)
                    require(MessageDigest.isEqual(
                        MessageDigest.getInstance("SHA-256").digest(blob),
                        accepted.bundle.embeddingSha256.hexBytes(),
                    )) { "Validated server embedding changed before append" }
                    val artist = TrackNormalization.normalizeArtist(accepted.provider.artist)
                    val album = TrackNormalization.normalizeAlbum(accepted.provider.album)
                    val title = TrackNormalization.normalizeTitle(accepted.provider.title)
                    val durationMs = Math.toIntExact(accepted.provider.durationMs)
                    val trackValues = ContentValues().apply {
                        put(
                            "metadata_key",
                            TrackNormalization.buildMetadataKey(artist, album, title, durationMs),
                        )
                        put(
                            "filename_key",
                            TrackNormalization.normalizeAsFilename(
                                if (artist.isNotBlank()) "$artist - $title" else title,
                            ),
                        )
                        put("artist", accepted.provider.artist?.ifBlank { null })
                        put("album", accepted.provider.album?.ifBlank { null })
                        put("title", accepted.provider.title?.ifBlank { null })
                        put("duration_ms", durationMs)
                        put("file_path", accepted.provider.providerPhysicalPath)
                        put("source", "server")
                    }
                    val trackId = target.insertOrThrow("tracks", null, trackValues)
                    require(trackId > 0L) { "SQLite returned an invalid merged track ID" }
                    val embeddingValues = ContentValues().apply {
                        put("track_id", trackId)
                        put("embedding", blob)
                    }
                    require(target.insertOrThrow(
                        V2ServerBundleContract.EMBEDDING_TABLE,
                        null,
                        embeddingValues,
                    ) > 0L) { "Unable to append a validated server embedding" }
                    val receiptValues = ContentValues().apply {
                        put("receipt_schema_version", V2ServerMergeReceiptContract.SCHEMA_VERSION)
                        put("bundle_id", bundleId)
                        put("root_id", accepted.bundle.rootId)
                        put("relative_path", accepted.bundle.relativePath)
                        put("source_sha256", accepted.bundle.sourceSha256)
                        put("source_size_bytes", accepted.bundle.sourceSizeBytes)
                        put("provider_file_id", accepted.provider.powerampFileId)
                        put("provider_physical_path", accepted.provider.providerPhysicalPath)
                        put("track_id", trackId)
                        put("embedding_sha256", accepted.bundle.embeddingSha256)
                        put("embedding_spec_id", V2ServerBundleContract.EMBEDDING_SPEC_ID)
                        put("output_space_id", V2ServerBundleContract.OUTPUT_SPACE_ID)
                        put("merged_at_epoch_ms", 0L)
                    }
                    require(target.insertOrThrow(
                        V2ServerMergeReceiptContract.TABLE,
                        null,
                        receiptValues,
                    ) > 0L) { "Unable to persist server-merge receipt" }
                    additions += StagedAddition(accepted, trackId)
                    publish(
                        onProgress,
                        V2ServerMergeStage.APPENDING_EMBEDDINGS,
                        "Appended ${index + 1} of ${rows.size} server embeddings",
                        (index + 1).toLong(),
                        rows.size.toLong(),
                    )
                }
                target.setTransactionSuccessful()
            } finally {
                target.endTransaction()
            }
            target.rawQuery("PRAGMA integrity_check", null).use { cursor ->
                require(cursor.moveToFirst() && cursor.getString(0) == "ok" &&
                    !cursor.moveToNext()
                ) { "Merged staging database failed SQLite integrity check" }
            }
        } finally {
            source.close()
            target.close()
        }
        syncFile(stagedDatabase)
        return additions
    }

    private fun createServerMergeReceiptTable(database: SQLiteDatabase) {
        database.execSQL(
            """
            CREATE TABLE IF NOT EXISTS ${V2ServerMergeReceiptContract.TABLE} (
                receipt_schema_version INTEGER NOT NULL,
                bundle_id TEXT NOT NULL,
                root_id TEXT NOT NULL,
                relative_path TEXT NOT NULL,
                source_sha256 TEXT NOT NULL,
                source_size_bytes INTEGER NOT NULL,
                provider_file_id INTEGER NOT NULL,
                provider_physical_path TEXT NOT NULL,
                track_id INTEGER PRIMARY KEY REFERENCES tracks(id) ON DELETE CASCADE,
                embedding_sha256 TEXT NOT NULL,
                embedding_spec_id TEXT NOT NULL,
                output_space_id TEXT NOT NULL,
                merged_at_epoch_ms INTEGER NOT NULL,
                UNIQUE(root_id, relative_path, source_sha256, source_size_bytes),
                UNIQUE(provider_physical_path)
            )
            """.trimIndent(),
        )
    }

    private fun updateExactGraph(
        base: V2ResolvedActiveIndexGeneration,
        stagedDatabase: File,
        staging: File,
        onProgress: (V2ServerMergeProgress) -> Unit,
    ): File {
        require(base.manifest.graph != null && base.graphFile != null) {
            "Server merge requires an active exact graph so Random Walk remains available"
        }
        val exactBase = requireNotNull(V2ExactGraphIncrementalBase.fromActiveGeneration(base)) {
            "The active similarity graph has no exact incremental-update proof"
        }
        val graphDirectory = File(staging, "graph-update")
        require(graphDirectory.mkdir()) { "Cannot create private graph-update staging" }
        publish(
            onProgress,
            V2ServerMergeStage.UPDATING_SIMILARITY_GRAPH,
            "Updating the exact similarity graph for appended embeddings",
        )
        val database = EmbeddingDatabase.openReadWrite(stagedDatabase)
        val plan = try {
            GraphUpdater(database, graphDirectory).rebuildIndices(
                control = object : V2GraphUpdaterControl {
                    override fun onProgress(progress: V2GraphUpdaterProgress) {
                        publish(
                            onProgress,
                            V2ServerMergeStage.UPDATING_SIMILARITY_GRAPH,
                            progress.detail,
                            progress.completedUnits,
                            progress.totalUnits,
                        )
                    }

                    override fun onControlPoint(
                        stage: V2GraphUpdaterStage,
                        completedUnits: Long,
                    ) = Unit
                },
                exactBase = exactBase,
            )
        } finally {
            database.close()
        }
        require(plan.strategy == V2GraphUpdateStrategy.INCREMENTAL) {
            "Server append did not produce an exact incremental graph update"
        }
        val graphFile = File(graphDirectory, V2GraphGenerationFile.GRAPH_FILE)
        require(graphFile.isFile && graphFile.length() > 0L) {
            "Exact server-merge graph output is missing"
        }
        syncFile(stagedDatabase)
        syncFile(graphFile)
        return graphFile
    }

    private fun requireAcceptedSourcesStillCurrent(
        rows: List<VerifiedCandidate>,
        onProgress: (V2ServerMergeProgress) -> Unit,
    ) {
        val verifier = V2SourceIdentityVerifier(sourceFingerprinter)
        rows.forEachIndexed { index, accepted ->
            val match = accepted.sourceMatch
            verifier.requireVerified(
                providerPhysicalPath = accepted.provider.providerPhysicalPath,
                canonicalPath = V2IndexingLedgerIds.canonicalPath(match.canonicalFile.path),
                powerampFileId = accepted.provider.powerampFileId,
                planned = match.exactFingerprint,
                exactContent = false,
            ) { completedBytes, totalBytes ->
                publish(
                    onProgress,
                    V2ServerMergeStage.VERIFYING_SOURCE_BYTES,
                    "Source metadata changed; confirming accepted track ${index + 1} of " +
                        "${rows.size}: $completedBytes of $totalBytes bytes",
                    completedBytes,
                    totalBytes,
                )
            }
        }
    }

    private fun requireAddedBindings(
        catalog: V2ActiveLibraryCatalog,
        additions: List<StagedAddition>,
    ) {
        require(additions.all { addition ->
            catalog.trackIdForPowerampFile(addition.accepted.provider.powerampFileId) ==
                addition.trackId &&
                catalog.powerampFileIdForTrack(addition.trackId) ==
                addition.accepted.provider.powerampFileId
        }) { "A staged server row does not bind one-to-one to its intended Poweramp file" }
    }

    private fun requireInheritedBindingsPreserved(
        base: V2ActiveLibraryCatalog,
        staged: V2ActiveLibraryCatalog,
    ) {
        require(base.bindings.all { binding ->
            staged.trackIdForPowerampFile(binding.powerampFileId) == binding.trackId &&
                staged.powerampFileIdForTrack(binding.trackId) == binding.powerampFileId
        }) { "Server append changed an inherited Poweramp binding" }
    }

    private fun readIndexedPaths(databaseFile: File): Set<String> =
        SQLiteDatabase.openDatabase(
            databaseFile.path,
            null,
            SQLiteDatabase.OPEN_READONLY,
        ).use { database ->
            buildSet {
                database.rawQuery("SELECT file_path FROM tracks", null).use { cursor ->
                    while (cursor.moveToNext()) {
                        TrackNormalization.normalizePath(cursor.getString(0))?.let(::add)
                    }
                }
            }
        }

    private fun readServerMergeReceipts(databaseFile: File): Set<ExistingMergeReceipt> =
        SQLiteDatabase.openDatabase(
            databaseFile.path,
            null,
            SQLiteDatabase.OPEN_READONLY,
        ).use { database ->
            val table = V2ServerMergeReceiptContract.TABLE
            val exists = database.rawQuery(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1",
                arrayOf(table),
            ).use { it.moveToFirst() }
            if (!exists) return@use emptySet()
            buildSet {
                database.rawQuery(
                    """
                    SELECT receipt_schema_version, root_id, relative_path, source_sha256,
                           source_size_bytes, embedding_sha256, embedding_spec_id,
                           output_space_id, provider_physical_path
                    FROM $table
                    """.trimIndent(),
                    null,
                ).use { cursor ->
                    while (cursor.moveToNext()) {
                        require(cursor.getInt(0) == V2ServerMergeReceiptContract.SCHEMA_VERSION) {
                            "Active library contains an unsupported server-merge receipt"
                        }
                        add(
                            ExistingMergeReceipt(
                                rootId = cursor.getString(1),
                                relativePath = V2ServerBundlePathPolicy
                                    .requireCanonicalRelativePath(cursor.getString(2)),
                                sourceSha256 = cursor.getString(3),
                                sourceSizeBytes = cursor.getLong(4),
                                embeddingSha256 = cursor.getString(5),
                                embeddingSpecId = cursor.getString(6),
                                outputSpaceId = cursor.getString(7),
                                providerPhysicalPath = requireNotNull(
                                    TrackNormalization.normalizePath(cursor.getString(8)),
                                ),
                            ),
                        )
                    }
                }
            }
        }

    private fun V2ServerBundleTrack.outcome(
        disposition: V2ServerBundleRowDisposition,
        detail: String,
    ) = V2ServerBundleRowOutcome(rootId, relativePath, disposition, detail)

    private fun stableReceiptFor(
        bundle: V2ServerBundleTrack,
        receipts: Set<ExistingMergeReceipt>,
    ): ExistingMergeReceipt? = receipts.singleOrNull { receipt ->
        receipt.rootId == bundle.rootId &&
            receipt.relativePath == bundle.relativePath &&
            receipt.sourceSha256 == bundle.sourceSha256 &&
            receipt.sourceSizeBytes == bundle.sourceSizeBytes &&
            receipt.embeddingSpecId == V2ServerBundleContract.EMBEDDING_SPEC_ID &&
            receipt.outputSpaceId == V2ServerBundleContract.OUTPUT_SPACE_ID
    }

    private fun providerReceiptFor(
        bundle: V2ServerBundleTrack,
        normalizedProviderPath: String,
        receipts: Set<ExistingMergeReceipt>,
    ): ExistingMergeReceipt? = receipts.singleOrNull { receipt ->
        receipt.sourceSha256 == bundle.sourceSha256 &&
            receipt.sourceSizeBytes == bundle.sourceSizeBytes &&
            receipt.providerPhysicalPath == normalizedProviderPath &&
            receipt.embeddingSpecId == V2ServerBundleContract.EMBEDDING_SPEC_ID &&
            receipt.outputSpaceId == V2ServerBundleContract.OUTPUT_SPACE_ID
    }

    private fun requireCurrentCatalog(
        base: V2ResolvedActiveIndexGeneration,
        onProgress: (V2ServerMergeProgress) -> Unit,
    ): V2ActiveLibraryCatalog {
        val store = V2ActiveLibraryCatalogStore(filesDir)
        store.read(base)?.let { return it }
        publish(
            onProgress,
            V2ServerMergeStage.READING_POWERAMP_LIBRARY,
            "The binding cache is absent; reading Poweramp to rebuild it",
        )
        val snapshot = providerAcquirer.acquireBlocking { completed, total ->
            publish(
                onProgress,
                V2ServerMergeStage.READING_POWERAMP_LIBRARY,
                "Read $completed of $total Poweramp rows",
                completed.toLong(),
                total.toLong(),
            )
        }
        return V2ActiveLibraryCatalogLoader.load(base, snapshot).also { catalog ->
            store.write(base, catalog)
        }
    }

    private fun List<V2ServerBundleRowOutcome>.sortedByOutcome(): List<V2ServerBundleRowOutcome> =
        sortedWith(
            compareBy(V2ServerBundleRowOutcome::rootId)
                .thenBy(V2ServerBundleRowOutcome::relativePath),
        )

    private fun String.hexBytes(): ByteArray {
        require(length == 64)
        return ByteArray(length / 2) { index ->
            substring(index * 2, index * 2 + 2).toInt(16).toByte()
        }
    }

    private fun ByteArray.toHex(): String = joinToString("") { byte ->
        (byte.toInt() and 0xff).toString(16).padStart(2, '0')
    }

    private fun snapshotDatabase(
        base: V2ResolvedActiveIndexGeneration,
        target: File,
        onProgress: (copiedBytes: Long, totalBytes: Long) -> Unit,
    ) {
        require(!target.exists()) { "Server-merge snapshot destination already exists" }
        val source = base.databaseFile
        require(!File(source.path + "-wal").exists() && !File(source.path + "-shm").exists()) {
            "Immutable active generation unexpectedly has SQLite sidecar files"
        }
        val digest = MessageDigest.getInstance("SHA-256")
        var copied = 0L
        FileInputStream(source).use { input ->
            FileOutputStream(target).use { output ->
                val buffer = ByteArray(COPY_BUFFER_BYTES)
                while (true) {
                    val read = input.read(buffer)
                    if (read < 0) break
                    if (read == 0) continue
                    output.write(buffer, 0, read)
                    digest.update(buffer, 0, read)
                    copied = Math.addExact(copied, read.toLong())
                    onProgress(copied, base.manifest.databaseByteLength)
                }
                output.flush()
                output.fd.sync()
            }
        }
        require(copied == base.manifest.databaseByteLength &&
            copied == source.length() &&
            digest.digest().toHex() == base.manifest.databaseSha256
        ) { "Active generation changed during byte-exact staging copy" }
        require(target.isFile && target.length() == copied) {
            "Byte-exact active-generation staging copy is incomplete"
        }
        syncFile(target)
    }

    private fun copyAndSync(
        openSource: () -> InputStream,
        target: File,
        expectedLength: Long?,
        onProgress: (Long, Long?) -> Unit,
    ) {
        require(!target.exists()) { "Server-bundle staging destination already exists" }
        var copied = 0L
        BufferedInputStream(openSource(), COPY_BUFFER_BYTES).use { input ->
            FileOutputStream(target).use { output ->
                val buffer = ByteArray(COPY_BUFFER_BYTES)
                while (true) {
                    val read = input.read(buffer)
                    if (read < 0) break
                    if (read == 0) continue
                    output.write(buffer, 0, read)
                    copied = Math.addExact(copied, read.toLong())
                    onProgress(copied, expectedLength)
                }
                output.flush()
                output.fd.sync()
            }
        }
        require(copied > 0L) { "Selected server bundle is empty" }
        expectedLength?.let { require(copied == it) { "Server bundle copy ended early" } }
    }

    private fun removeAbandonedStaging() {
        mergeRoot.listFiles().orEmpty()
            .filter { it.isDirectory && it.name.startsWith(".staging-") }
            .forEach(File::deleteRecursively)
    }

    private fun publish(
        callback: (V2ServerMergeProgress) -> Unit,
        stage: V2ServerMergeStage,
        detail: String,
        completed: Long? = null,
        total: Long? = null,
    ) = callback(V2ServerMergeProgress(stage, detail, completed, total))

    private fun syncFile(file: File) {
        RandomAccessFile(file, "rw").use { it.fd.sync() }
    }

    private fun syncDirectory(directory: File) {
        val descriptor = Os.open(directory.path, OsConstants.O_RDONLY, 0)
        try {
            Os.fsync(descriptor)
        } finally {
            Os.close(descriptor)
        }
    }

    private companion object {
        const val COPY_BUFFER_BYTES = 1024 * 1024
        const val SERVER_MERGE_LEASE_JOB_ID = "server-merge"
        val MERGE_PROCESS_LOCK = Any()
    }
}
