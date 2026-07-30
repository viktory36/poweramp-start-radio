package com.powerampstartradio.indexing.v2

import com.powerampstartradio.poweramp.TrackNormalization
import java.io.DataOutputStream
import java.io.File
import java.io.OutputStream
import java.nio.file.Paths
import java.security.DigestOutputStream
import java.security.MessageDigest

/** Proof retained with a snapshot assembled after one provider cursor reached exhaustion. */
data class V2ProviderSnapshotAcquisitionEvidence(
    val queryUri: String,
    val requestedColumns: List<String>,
    val returnedColumns: List<String>,
    val rowCount: Int,
    val cursorExhaustedNormally: Boolean,
    val queryAndCursorReadMs: Long? = null,
    val snapshotAssemblyMs: Long? = null,
) {
    /** Poweramp strips table qualifiers from some columns in the returned cursor. */
    fun returnedRequestedColumn(requestedColumn: String): Boolean {
        val providerAlias = requestedColumn.substringAfterLast('.')
        return requestedColumn in returnedColumns || providerAlias in returnedColumns
    }
}

/** Cursor-independent representation of one complete Poweramp provider row. */
data class V2RawPowerampProviderRow(
    val powerampFileId: Long,
    val artist: String?,
    val album: String?,
    val title: String?,
    val durationMs: Long,
    val folderPath: String,
    val fileName: String,
    val offsetMs: Long,
    val offsetWasNull: Boolean,
    val cueSourceImageFolderId: Long?,
    /** Poweramp first-seen Unix time, or zero when that optional row evidence is unavailable. */
    val createdAtEpochSecond: Long = 1L,
)

fun interface V2ProviderLexicalPathNormalizer {
    fun normalizeAbsolute(providerPhysicalPath: String): String
}

/**
 * Stable provider identity without filesystem access.
 *
 * Dot segments and duplicate separators are collapsed lexically. Symlinks are deliberately not
 * resolved here: a complete provider snapshot describes provider state, while strict preflight
 * binds only selected assets to verified canonical filesystem targets.
 */
object V2StableProviderLexicalPathNormalizer : V2ProviderLexicalPathNormalizer {
    override fun normalizeAbsolute(providerPhysicalPath: String): String {
        require(providerPhysicalPath.isNotBlank()) { "Provider physical path is blank" }
        require('\\' !in providerPhysicalPath) {
            "Provider physical path must use forward slashes"
        }
        val nfc = TrackNormalization.normalizeNfc(providerPhysicalPath)
        val path = Paths.get(nfc)
        require(path.isAbsolute) { "Provider physical path is not absolute: $providerPhysicalPath" }
        val normalized = TrackNormalization.normalizeNfc(path.normalize().toString())
        require(normalized.startsWith('/')) {
            "Normalized provider physical path is not absolute: $providerPhysicalPath"
        }
        return normalized
    }
}

enum class V2PowerampProviderSnapshotFailureCode {
    POWERAMP_PERMISSION_DENIED,
    PROVIDER_QUERY_FAILED,
    PROVIDER_RETURNED_NULL_CURSOR,
    PROVIDER_SCHEMA_MISMATCH,
    PROVIDER_CURSOR_FAILED,
    CURSOR_NOT_EXHAUSTED,
    CURSOR_EVIDENCE_MISMATCH,
    EMPTY_PROVIDER_RESULT,
    INVALID_PROVIDER_ROW,
    DUPLICATE_POWERAMP_FILE_ID,
    PATH_NORMALIZATION_FAILED,
}

class V2PowerampProviderSnapshotException(
    val code: V2PowerampProviderSnapshotFailureCode,
    val powerampFileId: Long? = null,
    message: String,
    cause: Throwable? = null,
) : IllegalStateException(message, cause)

/**
 * Pure completion boundary between cursor acquisition and span resolution.
 *
 * It cannot emit COMPLETE groups until the caller proves normal cursor exhaustion, and an empty
 * provider response is a typed failure rather than an apparently empty music library.
 */
class V2PowerampProviderSnapshotAssembler(
    private val pathNormalizer: V2ProviderLexicalPathNormalizer =
        V2StableProviderLexicalPathNormalizer,
) {
    fun assembleAfterSuccessfulExhaustion(
        rows: List<V2RawPowerampProviderRow>,
        acquisitionEvidence: V2ProviderSnapshotAcquisitionEvidence,
    ): V2ProviderPathGroupSnapshot {
        if (!acquisitionEvidence.cursorExhaustedNormally) {
            fail(
                V2PowerampProviderSnapshotFailureCode.CURSOR_NOT_EXHAUSTED,
                message = "Poweramp cursor did not reach normal exhaustion",
            )
        }
        if (acquisitionEvidence.rowCount != rows.size) {
            fail(
                V2PowerampProviderSnapshotFailureCode.CURSOR_EVIDENCE_MISMATCH,
                message = "Cursor evidence says ${acquisitionEvidence.rowCount} rows, " +
                    "but ${rows.size} were supplied",
            )
        }
        if (rows.isEmpty()) {
            fail(
                V2PowerampProviderSnapshotFailureCode.EMPTY_PROVIDER_RESULT,
                message = "Poweramp provider returned zero rows",
            )
        }

        val seenIds = mutableSetOf<Long>()
        val normalizedRows = rows.map { row ->
            if (row.powerampFileId <= 0L ||
                row.folderPath.isBlank() ||
                row.fileName.isBlank() ||
                row.createdAtEpochSecond < 0L
            ) {
                fail(
                    V2PowerampProviderSnapshotFailureCode.INVALID_PROVIDER_ROW,
                    row.powerampFileId,
                    "Invalid Poweramp provider row ${row.powerampFileId}",
                )
            }
            if (!seenIds.add(row.powerampFileId)) {
                fail(
                    V2PowerampProviderSnapshotFailureCode.DUPLICATE_POWERAMP_FILE_ID,
                    row.powerampFileId,
                    "Poweramp file ID ${row.powerampFileId} appears more than once",
                )
            }
            val providerPath = joinProviderPath(row)
            val pathKey = try {
                pathNormalizer.normalizeAbsolute(providerPath)
            } catch (error: Throwable) {
                fail(
                    V2PowerampProviderSnapshotFailureCode.PATH_NORMALIZATION_FAILED,
                    row.powerampFileId,
                    "Unable to normalize provider path $providerPath: ${error.message}",
                    error,
                )
            }
            V2ProviderPathRowEvidence(
                powerampFileId = row.powerampFileId,
                physicalPath = pathKey,
                providerPhysicalPath = providerPath,
                artist = row.artist,
                album = row.album,
                title = row.title,
                offsetMs = row.offsetMs,
                offsetWasNull = row.offsetWasNull,
                durationMs = V2ProviderDurationEvidencePolicy.canonicalMs(row.durationMs),
                cueSourceImageFolderId = row.cueSourceImageFolderId,
                createdAtEpochSecond = row.createdAtEpochSecond,
            )
        }

        val groups = normalizedRows
            .groupBy { it.physicalPath }
            .toSortedMap()
            .map { (providerPathKey, groupRows) ->
                V2ProviderPathGroupEvidence(
                    physicalPath = providerPathKey,
                    rows = groupRows.sortedBy { it.powerampFileId },
                    completeness = V2ProviderPathGroupCompleteness.COMPLETE,
                )
            }
        val generation = V2PowerampProviderSnapshotIdentity.generation(normalizedRows)
        return V2ProviderPathGroupSnapshot(
            libraryGeneration = generation,
            groups = groups,
            acquisitionEvidence = acquisitionEvidence,
        )
    }

    private fun joinProviderPath(row: V2RawPowerampProviderRow): String {
        if (File(row.fileName).isAbsolute || File(row.fileName).name != row.fileName ||
            row.fileName == "." || row.fileName == ".."
        ) {
            fail(
                V2PowerampProviderSnapshotFailureCode.INVALID_PROVIDER_ROW,
                row.powerampFileId,
                "Poweramp file name is unexpectedly absolute for row ${row.powerampFileId}",
            )
        }
        val joined = File(row.folderPath, row.fileName).path
        if (!File(joined).isAbsolute) {
            fail(
                V2PowerampProviderSnapshotFailureCode.INVALID_PROVIDER_ROW,
                row.powerampFileId,
                "Poweramp physical path is not absolute for row ${row.powerampFileId}: $joined",
            )
        }
        return joined
    }

    private fun fail(
        code: V2PowerampProviderSnapshotFailureCode,
        powerampFileId: Long? = null,
        message: String,
        cause: Throwable? = null,
    ): Nothing = throw V2PowerampProviderSnapshotException(
        code = code,
        powerampFileId = powerampFileId,
        message = message,
        cause = cause,
    )
}

/** Stable order-independent identity for the exact complete provider evidence used by a job. */
internal object V2PowerampProviderSnapshotIdentity {
    private const val SPEC_ID = "poweramp-provider-snapshot-v3-sha256"

    fun generation(rows: List<V2ProviderPathRowEvidence>): String {
        val digest = MessageDigest.getInstance("SHA-256")
        DataOutputStream(DigestOutputStream(DiscardingOutputStream, digest)).use { output ->
            output.writeLengthPrefixed(SPEC_ID)
            val ordered = rows.sortedWith(
                compareBy<V2ProviderPathRowEvidence> { it.powerampFileId }
                    .thenBy { it.physicalPath },
            )
            output.writeInt(ordered.size)
            for (row in ordered) {
                output.writeLong(row.powerampFileId)
                output.writeLengthPrefixed(row.physicalPath)
                output.writeLengthPrefixed(row.providerPhysicalPath)
                output.writeNullable(row.artist)
                output.writeNullable(row.album)
                output.writeNullable(row.title)
                output.writeLong(row.durationMs)
                output.writeLong(row.offsetMs)
                output.writeBoolean(row.offsetWasNull)
                output.writeNullableLong(row.cueSourceImageFolderId)
                output.writeLong(row.createdAtEpochSecond)
            }
        }
        return "$SPEC_ID:${digest.digest().joinToString("") { "%02x".format(it) }}"
    }

    private fun DataOutputStream.writeLengthPrefixed(value: String) {
        val bytes = value.toByteArray(Charsets.UTF_8)
        writeInt(bytes.size)
        write(bytes)
    }

    private fun DataOutputStream.writeNullable(value: String?) {
        writeBoolean(value != null)
        if (value != null) writeLengthPrefixed(value)
    }

    private fun DataOutputStream.writeNullableLong(value: Long?) {
        writeBoolean(value != null)
        if (value != null) writeLong(value)
    }

    private object DiscardingOutputStream : OutputStream() {
        override fun write(value: Int) = Unit
        override fun write(buffer: ByteArray, offset: Int, length: Int) = Unit
    }
}
