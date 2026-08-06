package com.powerampstartradio.indexing.v2

import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.net.Uri
import android.system.Os
import android.system.OsConstants
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.BufferedInputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.InputStream
import java.security.MessageDigest
import java.util.UUID

data class V2FutureModelPolicy(
    val receiptEmbeddingSpec: EmbeddingSpecFingerprint,
    val textRetrievalSpec: TextRetrievalSpecFingerprint,
)

data class V2BootstrapDatabaseValidation(
    val sourceByteLength: Long,
    val sourceSha256: String,
    val trackCount: Int,
    val embeddingDimension: Int,
    val orderedTrackSetSha256: String,
    val databaseContentSha256: String,
)

data class V2BootstrapImportResult(
    val generation: V2ResolvedActiveIndexGeneration,
    val sourceValidation: V2BootstrapDatabaseValidation,
)

data class V2LibraryMaintenanceResult(
    val generation: V2ResolvedActiveIndexGeneration,
    val removedTrackCount: Int,
    val noOp: Boolean,
)

data class V2ResolvedLibraryDatabase(
    val databaseFile: File,
    val activeGeneration: V2ResolvedActiveIndexGeneration,
) {
    val generationId: String get() = activeGeneration.manifest.generationId
}

/** Resolves exact current model artifacts into the policy bound by every new generation. */
object V2CurrentModelPolicyResolver {
    fun resolve(filesDir: File): V2FutureModelPolicy = resolveInstalled(filesDir).policy

    /** Compatibility entry point; stat changes, rather than every call, trigger exact rehashing. */
    fun resolveFresh(filesDir: File): V2FutureModelPolicy = resolve(filesDir)

    fun resolveInstalled(
        filesDir: File,
        onHashProgress: (V2InstalledModelHashProgress) -> Unit = {},
    ): V2ResolvedInstalledModelPolicy =
        V2InstalledModelReceiptStore.resolve(filesDir, onHashProgress)

    fun hasRequiredArtifacts(filesDir: File): Boolean = runCatching {
        listOf(
            "mert.tflite",
            "clamp3_audio.tflite",
            "clamp3_text.tflite",
            "sentencepiece.bpe.model",
        ).forEach { requireArtifact(filesDir, it) }
    }.isSuccess

    private fun requireArtifact(filesDir: File, name: String): File {
        val expectedParent = filesDir.canonicalFile
        val file = File(expectedParent, name).canonicalFile
        require(file.parentFile == expectedParent && file.isFile && file.canRead() &&
            file.length() > 0L
        ) { "Required exact model artifact is missing or unreadable: $name" }
        return file
    }

}

/** Resolves the app's one atomically published music index. */
object V2LibraryDatabaseResolver {
    private const val ACTIVE_POINTER_RELATIVE_PATH =
        "indexing_v2/generations/active-generation.json"

    fun hasPublishedPointer(filesDir: File): Boolean {
        val pointer = File(filesDir, ACTIVE_POINTER_RELATIVE_PATH)
        return pointer.isFile || File(pointer.path + ".bak").isFile
    }

    fun resolveOrNull(
        filesDir: File,
        onArtifactHashProgress: (V2GenerationArtifactHashProgress) -> Unit = {},
    ): V2ResolvedLibraryDatabase? {
        if (!hasPublishedPointer(filesDir)) return null
        val active = V2IndexGenerationReader.requireActive(
            filesDir,
            onArtifactHashProgress = onArtifactHashProgress,
        )
        return V2ResolvedLibraryDatabase(
            databaseFile = active.databaseFile,
            activeGeneration = active,
        )
    }

    fun requirePublished(
        filesDir: File,
        onArtifactHashProgress: (V2GenerationArtifactHashProgress) -> Unit = {},
    ): V2ResolvedActiveIndexGeneration = V2IndexGenerationReader.requireActive(
        filesDir,
        onArtifactHashProgress = onArtifactHashProgress,
    )
}

/**
 * Copies an untrusted document into private staging, validates every row, then publishes one
 * immutable compatibility generation. The active pointer is never touched before publication.
 */
class V2BootstrapGenerationImporter(
    context: Context,
    private val filesDir: File = context.filesDir,
    private val publisher: V2IndexGenerationPublisher =
        V2IndexGenerationPublisher(filesDir),
    private val modelPolicyResolver: (File) -> V2FutureModelPolicy =
        V2CurrentModelPolicyResolver::resolveFresh,
) {
    private val appContext = context.applicationContext
    private val importRoot = File(filesDir, "indexing_v2/imports")

    suspend fun import(
        uri: Uri,
        onProgress: (String) -> Unit = {},
    ): V2BootstrapImportResult = withContext(Dispatchers.IO) {
        val length = runCatching {
            appContext.contentResolver.openAssetFileDescriptor(uri, "r")?.use { descriptor ->
                descriptor.length.takeIf { it >= 0L }
            }
        }.getOrNull()
        importBlocking(
            sourceLength = length,
            openSource = {
                appContext.contentResolver.openInputStream(uri)
                    ?: throw IllegalArgumentException("Cannot open database document")
            },
            onProgress = onProgress,
        )
    }

    internal fun importFileBlocking(
        source: File,
        modelPolicy: V2FutureModelPolicy = modelPolicyResolver(filesDir),
        onProgress: (String) -> Unit = {},
    ): V2BootstrapImportResult = importBlocking(
        sourceLength = source.length(),
        openSource = { FileInputStream(source) },
        modelPolicy = modelPolicy,
        onProgress = onProgress,
    )

    internal fun importBlocking(
        sourceLength: Long?,
        openSource: () -> InputStream,
        modelPolicy: V2FutureModelPolicy = modelPolicyResolver(filesDir),
        onProgress: (String) -> Unit = {},
    ): V2BootstrapImportResult {
        val expectedActive = V2GenerationPublicationCoordinator.capture(filesDir)
        require(expectedActive.pointer == null) { "A music index is already loaded" }
        sourceLength?.takeIf { it > 0L }?.let { knownLength ->
            V2GenerationMutationStoragePolicy.requireCapacity(
                V2GenerationMutationStoragePolicy.estimateBootstrapAdmission(
                    sourceLength = knownLength,
                    availableBytes = filesDir.usableSpace,
                ),
                operation = "Database import",
            )
        }
        require(importRoot.isDirectory || importRoot.mkdirs()) {
            "Cannot create private import staging directory"
        }
        syncDirectory(importRoot.parentFile ?: filesDir)
        val staging = File(importRoot, ".staging-${UUID.randomUUID()}")
        require(staging.mkdir()) { "Cannot create private database staging directory" }
        syncDirectory(importRoot)
        try {
            val copied = File(staging, "source.db")
            onProgress(copyMessage(0L, sourceLength))
            copyAndSync(openSource, copied, sourceLength, onProgress)
            syncDirectory(staging)
            onProgress("Validating embedding rows in the imported database...")
            val validation = V2BootstrapDatabaseValidator.validate(copied) { rows, total ->
                onProgress(
                    "Validated ${formatCount(rows)} of ${formatCount(total)} embeddings",
                )
            }
            onProgress(
                "Reading the imported similarity graph and matching it to " +
                    "${formatCount(validation.trackCount)} tracks...",
            )
            val exactGraph = V2EmbeddedGraphImporter.extractIfExact(
                databaseFile = copied,
                target = File(staging, "validated-graph.bin"),
                expectedTrackCount = validation.trackCount,
                expectedOrderedTrackSetSha256 = validation.orderedTrackSetSha256,
            )
            V2GenerationMutationStoragePolicy.requireCapacity(
                V2GenerationMutationStoragePolicy.estimateBootstrapPublication(
                    databaseBytes = validation.sourceByteLength,
                    trackCount = validation.trackCount,
                    embeddingDimension = validation.embeddingDimension,
                    graphBytes = exactGraph?.length() ?: 0L,
                    availableBytes = filesDir.usableSpace,
                ),
                operation = "Database import publication",
            )
            onProgress(
                "Adding ${formatCount(validation.trackCount)} embeddings...",
            )
            val generation = publisher.publishBootstrapCompatibility(
                privateStagingDatabase = copied,
                futureReceiptEmbeddingSpec = modelPolicy.receiptEmbeddingSpec,
                textRetrievalSpec = modelPolicy.textRetrievalSpec,
                exactValidatedGraphFile = exactGraph,
                expectedActive = expectedActive,
            )
            require(generation.manifest.origin ==
                V2IndexGenerationOrigin.BOOTSTRAP_COMPATIBILITY &&
                generation.manifest.databaseContentSha256 == validation.databaseContentSha256 &&
                generation.manifest.orderedTrackSetSha256 == validation.orderedTrackSetSha256 &&
                generation.manifest.embeddingCoverage.receiptBoundTrackCount == 0
            ) { "Published bootstrap generation disagrees with validated source content" }
            val graphStatus = if (generation.graphFile != null) {
                "similarity graph preserved"
            } else {
                "similarity graph will be rebuilt when needed"
            }
            onProgress(
                "Imported ${formatCount(generation.manifest.trackCount)} embeddings; " +
                    "$graphStatus.",
            )
            return V2BootstrapImportResult(generation, validation)
        } finally {
            staging.deleteRecursively()
            syncDirectory(importRoot)
        }
    }

    private fun copyAndSync(
        openSource: () -> InputStream,
        target: File,
        expectedLength: Long?,
        onProgress: (String) -> Unit,
    ) {
        require(!target.exists()) { "Private staging destination already exists" }
        var copied = 0L
        var lastReported = 0L
        BufferedInputStream(openSource(), COPY_BUFFER_BYTES).use { input ->
            FileOutputStream(target).use { output ->
                val buffer = ByteArray(COPY_BUFFER_BYTES)
                while (true) {
                    val read = input.read(buffer)
                    if (read < 0) break
                    if (read == 0) continue
                    output.write(buffer, 0, read)
                    copied = Math.addExact(copied, read.toLong())
                    if (copied - lastReported >= PROGRESS_STEP_BYTES) {
                        onProgress(copyMessage(copied, expectedLength))
                        lastReported = copied
                    }
                }
                output.flush()
                output.fd.sync()
            }
        }
        require(copied > 0L) { "Selected database document is empty" }
        if (expectedLength != null) {
            require(copied == expectedLength) {
                "Database copy ended at $copied bytes; expected $expectedLength"
            }
        }
        require(target.length() == copied) { "Private database staging copy is truncated" }
        onProgress(copyMessage(copied, expectedLength ?: copied))
    }

    private fun copyMessage(copied: Long, total: Long?): String = if (total != null && total > 0L) {
        val percent = ((copied.coerceAtMost(total) * 100L) / total).toInt()
        "Copying database: $percent% (${formatBytes(copied)} / ${formatBytes(total)})"
    } else {
        "Copying database: ${formatBytes(copied)}"
    }

    private fun formatBytes(bytes: Long): String = when {
        bytes >= 1024L * 1024L -> "%.1f MB".format(bytes / (1024.0 * 1024.0))
        bytes >= 1024L -> "%.1f KB".format(bytes / 1024.0)
        else -> "$bytes B"
    }

    private fun formatCount(value: Int): String = "%,d".format(value)

    private companion object {
        const val COPY_BUFFER_BYTES = 1024 * 1024
        const val PROGRESS_STEP_BYTES = 8L * 1024L * 1024L
    }
}

/** Publishes an exact retained subset without mutating the complete base generation. */
class V2LibraryMaintenancePublisher(
    context: Context,
    private val filesDir: File = context.filesDir,
    private val publisher: V2IndexGenerationPublisher = V2IndexGenerationPublisher(filesDir),
) {
    private val maintenanceRoot = File(filesDir, "indexing_v2/maintenance")

    fun removeTracksBlocking(
        requestedTrackIds: Set<Long>,
        onProgress: (String) -> Unit = {},
    ): V2LibraryMaintenanceResult {
        require(requestedTrackIds.isNotEmpty()) { "No tracks were selected for clean-up" }
        require(requestedTrackIds.all { it > 0L }) { "Clean-up contains an invalid track ID" }
        val base = V2LibraryDatabaseResolver.requirePublished(filesDir)
        val expectedActive = V2GenerationPublicationCoordinator.capture(filesDir)
        require(expectedActive.pointer?.generationId == base.manifest.generationId &&
            expectedActive.pointer.manifestSha256 == base.manifestSha256
        ) { "The active library changed before clean-up could begin" }
        val existing = countRequestedRows(base.databaseFile, requestedTrackIds)
        if (existing == 0) {
            return V2LibraryMaintenanceResult(base, removedTrackCount = 0, noOp = true)
        }
        require(existing == requestedTrackIds.size) {
            "The active library changed: only $existing of ${requestedTrackIds.size} selected rows remain"
        }
        require(base.manifest.trackCount > requestedTrackIds.size) {
            "Clean-up cannot remove the entire embedding library"
        }
        V2GenerationMutationStoragePolicy.requireCapacity(
            V2GenerationMutationStoragePolicy.estimateMaintenance(
                active = base,
                availableBytes = filesDir.usableSpace,
            ),
            operation = "Library clean-up",
        )
        require(maintenanceRoot.isDirectory || maintenanceRoot.mkdirs()) {
            "Cannot create private maintenance staging"
        }
        syncDirectory(maintenanceRoot.parentFile ?: filesDir)
        val staging = File(maintenanceRoot, ".staging-${UUID.randomUUID()}")
        require(staging.mkdir()) { "Cannot create private maintenance operation" }
        syncDirectory(maintenanceRoot)
        try {
            val stagedDatabase = File(staging, "retained.db")
            onProgress("Copying the active music index into private staging...")
            snapshotDatabase(base.databaseFile, stagedDatabase)
            onProgress("Removing ${requestedTrackIds.size} selected tracks...")
            val removed = removeExactRows(stagedDatabase, requestedTrackIds)
            require(removed == requestedTrackIds.size) {
                "Staged clean-up removed $removed rows; expected ${requestedTrackIds.size}"
            }
            val repairedGraph = base.graphFile?.let { baseGraph ->
                onProgress("Updating the similarity graph...")
                val retainedEmbeddingFile = File(staging, "retained-for-graph.emb")
                try {
                    val retainedBinding = SQLiteDatabase.openDatabase(
                        stagedDatabase.path,
                        null,
                        SQLiteDatabase.OPEN_READONLY,
                    ).use { database ->
                        V2EmbeddingGenerationFile.write(
                            V2SqliteOrderedEmbeddingSource(database),
                            retainedEmbeddingFile,
                        )
                    }
                    V2GraphDeletionRepairer.repairOrNull(
                        baseGraphFile = baseGraph,
                        retainedEmbeddingFile = retainedEmbeddingFile,
                        retainedEmbeddingBinding = retainedBinding,
                        target = File(staging, "repaired-graph.bin"),
                        onProgress = onProgress,
                    )?.also { repair ->
                        onProgress(
                            "Similarity graph updated: ${repair.affectedRowCount} tracks rescanned, " +
                                "${repair.preservedRowCount} tracks retained.",
                        )
                    }?.graphFile
                } finally {
                    retainedEmbeddingFile.delete()
                    syncDirectory(staging)
                }
            }
            syncDirectory(staging)
            onProgress("Hashing and atomically publishing the updated music index...")
            val generation = publisher.publishLibraryMaintenance(
                privateStagingDatabase = stagedDatabase,
                baseGeneration = base,
                exactRepairedGraphFile = repairedGraph,
                expectedActive = expectedActive,
            )
            V2LibraryMaintenancePublicationPolicy.requireExactRetainedSubset(
                base = base.manifest,
                published = generation.manifest,
                removedTrackCount = removed,
            )
            onProgress(
                "Updated the music index to ${generation.manifest.trackCount} tracks.",
            )
            return V2LibraryMaintenanceResult(generation, removed, noOp = false)
        } finally {
            staging.deleteRecursively()
            syncDirectory(maintenanceRoot)
        }
    }

    private fun countRequestedRows(databaseFile: File, ids: Set<Long>): Int =
        SQLiteDatabase.openDatabase(
            databaseFile.path,
            null,
            SQLiteDatabase.OPEN_READONLY,
        ).use { database ->
            ids.sorted().chunked(READ_QUERY_ID_LIMIT).fold(0) { total, chunk ->
                val placeholders = List(chunk.size) { "?" }.joinToString(",")
                val count = database.rawQuery(
                    "SELECT COUNT(*) FROM tracks WHERE id IN ($placeholders)",
                    chunk.map(Long::toString).toTypedArray(),
                ).use { cursor ->
                    check(cursor.moveToFirst())
                    cursor.getInt(0)
                }
                Math.addExact(total, count)
            }
        }

    private fun removeExactRows(databaseFile: File, ids: Set<Long>): Int =
        SQLiteDatabase.openDatabase(
            databaseFile.path,
            null,
            SQLiteDatabase.OPEN_READWRITE,
        ).use { database ->
            database.disableWriteAheadLogging()
            installSelectedIds(database, ids)
            val receiptTable = V2EmbeddingCommitRepository.RECEIPT_TABLE
            database.beginTransaction()
            val removed: Int
            try {
                if (hasTable(database, receiptTable)) {
                    database.execSQL(
                        "DELETE FROM $receiptTable WHERE track_id IN (SELECT id FROM selected_ids)",
                    )
                }
                database.execSQL(
                    "DELETE FROM ${V2EmbeddingCommitRepository.EMBEDDING_TABLE} " +
                        "WHERE track_id IN (SELECT id FROM selected_ids)",
                )
                removed = database.delete(
                    "tracks",
                    "id IN (SELECT id FROM selected_ids)",
                    null,
                )
                if (hasTable(database, "binary_data")) {
                    database.delete("binary_data", "key = ?", arrayOf("knn_graph"))
                }
                database.setTransactionSuccessful()
            } finally {
                database.endTransaction()
            }
            database.rawQuery("PRAGMA integrity_check", null).use { cursor ->
                require(cursor.moveToFirst() && cursor.getString(0) == "ok" &&
                    !cursor.moveToNext()
                ) { "Cleaned staging database failed integrity validation" }
            }
            database.execSQL("DROP TABLE selected_ids")
            database.rawQuery("PRAGMA wal_checkpoint(TRUNCATE)", null).close()
            FileOutputStream(databaseFile, true).use { it.fd.sync() }
            removed
        }

    private fun installSelectedIds(database: SQLiteDatabase, ids: Set<Long>) {
        database.execSQL("DROP TABLE IF EXISTS temp.selected_ids")
        database.execSQL("CREATE TEMP TABLE selected_ids(id INTEGER PRIMARY KEY)")
        database.compileStatement("INSERT INTO selected_ids(id) VALUES (?)").use { statement ->
            ids.sorted().forEach { id ->
                statement.clearBindings()
                statement.bindLong(1, id)
                statement.executeInsert()
            }
        }
    }

    private fun snapshotDatabase(source: File, target: File) {
        require(!target.exists()) { "Maintenance snapshot already exists" }
        SQLiteDatabase.openDatabase(source.path, null, SQLiteDatabase.OPEN_READWRITE).use { database ->
            val escaped = target.canonicalPath.replace("'", "''")
            database.execSQL("VACUUM INTO '$escaped'")
        }
        require(target.isFile && target.length() > 0L) { "Maintenance snapshot was not created" }
        FileOutputStream(target, true).use { it.fd.sync() }
    }

    private fun hasTable(database: SQLiteDatabase, table: String): Boolean = database.rawQuery(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1",
        arrayOf(table),
    ).use { it.moveToFirst() }


    private companion object {
        const val READ_QUERY_ID_LIMIT = 500
    }
}

internal object V2LibraryMaintenancePublicationPolicy {
    fun requireExactRetainedSubset(
        base: V2IndexGenerationManifest,
        published: V2IndexGenerationManifest,
        removedTrackCount: Int,
    ) {
        require(removedTrackCount in 1 until base.trackCount) {
            "Maintenance removed-track count is invalid"
        }
        require(
            published.origin == V2IndexGenerationOrigin.LIBRARY_MAINTENANCE &&
                published.baseGenerationId == base.generationId &&
                published.trackCount == Math.subtractExact(base.trackCount, removedTrackCount),
        ) { "Published maintenance generation is not the exact retained subset" }
    }
}

object V2EmbeddedGraphImporter {
    fun extractIfExact(
        databaseFile: File,
        target: File,
        expectedTrackCount: Int,
        expectedOrderedTrackSetSha256: String,
    ): File? {
        target.delete()
        val database = SQLiteDatabase.openDatabase(
            databaseFile.path,
            null,
            SQLiteDatabase.OPEN_READONLY,
        )
        return try {
            val hasBinaryTable = database.rawQuery(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'binary_data' LIMIT 1",
                null,
            ).use { it.moveToFirst() }
            if (!hasBinaryTable) return null
            val byteLength = database.rawQuery(
                "SELECT length(data) FROM binary_data WHERE key = ?",
                arrayOf("knn_graph"),
            ).use { cursor ->
                if (!cursor.moveToFirst() || cursor.isNull(0)) return null
                cursor.getLong(0)
            }
            require(byteLength > 0L) { "Embedded graph is empty" }
            target.parentFile?.let { require(it.isDirectory || it.mkdirs()) }
            FileOutputStream(target).use { output ->
                var offset = 0L
                while (offset < byteLength) {
                    val amount = minOf(GRAPH_CHUNK_BYTES.toLong(), byteLength - offset).toInt()
                    val chunk = database.rawQuery(
                        "SELECT substr(data, ?, ?) FROM binary_data WHERE key = ?",
                        arrayOf((offset + 1L).toString(), amount.toString(), "knn_graph"),
                    ).use { cursor ->
                        require(cursor.moveToFirst() && !cursor.isNull(0)) {
                            "Embedded graph ended during chunked extraction"
                        }
                        cursor.getBlob(0)
                    }
                    require(chunk.size == amount) { "Embedded graph chunk is truncated" }
                    output.write(chunk)
                    offset += chunk.size
                }
                output.flush()
                output.fd.sync()
            }
            require(target.length() == byteLength) { "Extracted graph length mismatch" }
            val graph = V2GraphGenerationFile.inspect(target)
            if (graph.nodeCount == expectedTrackCount &&
                graph.orderedTrackSetSha256 == expectedOrderedTrackSetSha256
            ) {
                target
            } else {
                target.delete()
                null
            }
        } catch (_: Exception) {
            target.delete()
            null
        } finally {
            database.close()
        }
    }

    private const val GRAPH_CHUNK_BYTES = 512 * 1024
}

object V2BootstrapDatabaseValidator {
    fun validate(
        databaseFile: File,
        onEmbeddingProgress: (validatedRows: Int, totalRows: Int) -> Unit = { _, _ -> },
    ): V2BootstrapDatabaseValidation {
        require(databaseFile.isFile && databaseFile.length() > 0L) {
            "Imported database is missing or empty"
        }
        val sourceSha = V2FileSha256.digest(databaseFile)
        val database = SQLiteDatabase.openDatabase(
            databaseFile.path,
            null,
            SQLiteDatabase.OPEN_READONLY,
        )
        try {
            requireIntegrity(database)
            requireTable(database, "tracks", TRACK_COLUMNS)
            requireTable(database, V2EmbeddingCommitRepository.EMBEDDING_TABLE, EMBEDDING_COLUMNS)
            val trackCount = count(database, "tracks")
            val embeddingCount = count(database, V2EmbeddingCommitRepository.EMBEDDING_TABLE)
            require(trackCount > 0) { "Imported database contains no tracks" }
            require(trackCount == embeddingCount) {
                "Track/embedding count mismatch: $trackCount tracks, $embeddingCount embeddings"
            }
            val missingEmbeddings = scalarLong(
                database,
                """
                SELECT COUNT(*) FROM tracks t
                LEFT JOIN ${V2EmbeddingCommitRepository.EMBEDDING_TABLE} e ON e.track_id = t.id
                WHERE e.track_id IS NULL
                """.trimIndent(),
            )
            val orphanEmbeddings = scalarLong(
                database,
                """
                SELECT COUNT(*) FROM ${V2EmbeddingCommitRepository.EMBEDDING_TABLE} e
                LEFT JOIN tracks t ON t.id = e.track_id WHERE t.id IS NULL
                """.trimIndent(),
            )
            require(missingEmbeddings == 0L && orphanEmbeddings == 0L) {
                "Imported database has $missingEmbeddings tracks without vectors and " +
                    "$orphanEmbeddings orphan vectors"
            }
            require(scalarLong(database, "SELECT COUNT(*) FROM tracks WHERE id <= 0") == 0L) {
                "Imported database contains non-positive track IDs"
            }
            val receiptTable = V2EmbeddingCommitRepository.RECEIPT_TABLE
            if (hasTable(database, receiptTable)) {
                require(count(database, receiptTable) == 0) {
                    "Database contains V2 production receipts and cannot be imported as legacy"
                }
            }

            val binding = digestWithProgress(database, embeddingCount, onEmbeddingProgress)
            require(binding.trackCount == trackCount && binding.dimension == V2_CLAMP3_DIMENSION) {
                "Imported embedding shape changed during validation"
            }
            requireIntegrity(database)
            return V2BootstrapDatabaseValidation(
                sourceByteLength = databaseFile.length(),
                sourceSha256 = sourceSha,
                trackCount = binding.trackCount,
                embeddingDimension = binding.dimension,
                orderedTrackSetSha256 = binding.orderedTrackSetSha256,
                databaseContentSha256 = binding.databaseContentSha256,
            )
        } finally {
            database.close()
        }
    }

    private fun digestWithProgress(
        database: SQLiteDatabase,
        count: Int,
        onProgress: (Int, Int) -> Unit,
    ): V2DatabaseEmbeddingBinding {
        val idsDigest = MessageDigest.getInstance("SHA-256").apply {
            updateLengthPrefixed("v2-ordered-track-set-v1")
            updateInt(count)
        }
        val contentDigest = MessageDigest.getInstance("SHA-256").apply {
            updateLengthPrefixed("v2-ordered-clamp3-content-v1")
            updateInt(count)
            updateInt(V2_CLAMP3_DIMENSION)
        }
        var rows = 0
        var previousId = Long.MIN_VALUE
        database.rawQuery(
            "SELECT track_id, embedding FROM ${V2EmbeddingCommitRepository.EMBEDDING_TABLE} " +
                "ORDER BY track_id",
            null,
        ).use { cursor ->
            while (cursor.moveToNext()) {
                val id = cursor.getLong(0)
                val blob = cursor.getBlob(1)
                require(id > 0L && id > previousId) {
                    "Embedding IDs are duplicated or out of order at $id"
                }
                V2Clamp3VectorCodec.requireValidBlob(blob)
                idsDigest.updateLong(id)
                contentDigest.updateLong(id)
                contentDigest.updateInt(blob.size)
                contentDigest.update(blob)
                previousId = id
                rows++
                if (rows == count || rows % VALIDATION_PROGRESS_ROWS == 0) {
                    onProgress(rows, count)
                }
            }
        }
        require(rows == count) { "Embedding table changed during validation" }
        return V2DatabaseEmbeddingBinding(
            trackCount = count,
            dimension = V2_CLAMP3_DIMENSION,
            orderedTrackSetSha256 = idsDigest.digest().toHex(),
            databaseContentSha256 = contentDigest.digest().toHex(),
        )
    }

    private fun requireIntegrity(database: SQLiteDatabase) {
        database.rawQuery("PRAGMA integrity_check", null).use { cursor ->
            require(cursor.moveToFirst() && cursor.getString(0) == "ok" &&
                !cursor.moveToNext()
            ) { "SQLite integrity check failed" }
        }
    }

    private fun requireTable(
        database: SQLiteDatabase,
        table: String,
        required: Map<String, ColumnRequirement>,
    ) {
        require(hasTable(database, table)) { "Required table is missing: $table" }
        val columns = linkedMapOf<String, ColumnDescription>()
        database.rawQuery("PRAGMA table_info([$table])", null).use { cursor ->
            while (cursor.moveToNext()) {
                columns[cursor.getString(1)] = ColumnDescription(
                    affinity = cursor.getString(2).uppercase(),
                    notNull = cursor.getInt(3) == 1,
                    primaryKeyPosition = cursor.getInt(5),
                )
            }
        }
        required.forEach { (name, expected) ->
            val actual = columns[name]
                ?: throw IllegalArgumentException("Required column is missing: $table.$name")
            require(actual.affinity.contains(expected.affinity) &&
                (!expected.notNull || actual.notNull) &&
                (!expected.primaryKey || actual.primaryKeyPosition == 1)
            ) { "Column contract mismatch for $table.$name" }
        }
    }

    private fun hasTable(database: SQLiteDatabase, table: String): Boolean = database.rawQuery(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1",
        arrayOf(table),
    ).use { it.moveToFirst() }

    private fun count(database: SQLiteDatabase, table: String): Int =
        scalarLong(database, "SELECT COUNT(*) FROM [$table]").also {
            require(it in 0..Int.MAX_VALUE.toLong()) { "Table $table is too large" }
        }.toInt()

    private fun scalarLong(database: SQLiteDatabase, sql: String): Long =
        database.rawQuery(sql, null).use { cursor ->
            check(cursor.moveToFirst()) { "Scalar query returned no row" }
            cursor.getLong(0)
        }

    private data class ColumnDescription(
        val affinity: String,
        val notNull: Boolean,
        val primaryKeyPosition: Int,
    )

    private data class ColumnRequirement(
        val affinity: String,
        val notNull: Boolean = false,
        val primaryKey: Boolean = false,
    )

    private val TRACK_COLUMNS = mapOf(
        "id" to ColumnRequirement("INTEGER", primaryKey = true),
        "metadata_key" to ColumnRequirement("TEXT", notNull = true),
        "filename_key" to ColumnRequirement("TEXT", notNull = true),
        "artist" to ColumnRequirement("TEXT"),
        "album" to ColumnRequirement("TEXT"),
        "title" to ColumnRequirement("TEXT"),
        "duration_ms" to ColumnRequirement("INTEGER"),
        "file_path" to ColumnRequirement("TEXT", notNull = true),
        "source" to ColumnRequirement("TEXT"),
    )
    private val EMBEDDING_COLUMNS = mapOf(
        "track_id" to ColumnRequirement("INTEGER", primaryKey = true),
        "embedding" to ColumnRequirement("BLOB", notNull = true),
    )
    private const val VALIDATION_PROGRESS_ROWS = 1_000
}

private fun syncDirectory(directory: File) {
    if (!directory.isDirectory) return
    val descriptor = Os.open(directory.path, OsConstants.O_RDONLY, 0)
    try {
        Os.fsync(descriptor)
    } finally {
        Os.close(descriptor)
    }
}

private fun MessageDigest.updateLengthPrefixed(value: String) {
    val bytes = value.toByteArray(Charsets.UTF_8)
    updateInt(bytes.size)
    update(bytes)
}

private fun MessageDigest.updateInt(value: Int) {
    update((value ushr 24).toByte())
    update((value ushr 16).toByte())
    update((value ushr 8).toByte())
    update(value.toByte())
}

private fun MessageDigest.updateLong(value: Long) {
    update((value ushr 56).toByte())
    update((value ushr 48).toByte())
    update((value ushr 40).toByte())
    update((value ushr 32).toByte())
    update((value ushr 24).toByte())
    update((value ushr 16).toByte())
    update((value ushr 8).toByte())
    update(value.toByte())
}

private fun ByteArray.toHex(): String = joinToString("") { "%02x".format(it) }
