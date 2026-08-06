package com.powerampstartradio.indexing.v2

import android.database.sqlite.SQLiteDatabase
import android.util.AtomicFile
import com.google.gson.Gson
import com.google.gson.GsonBuilder
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.IOException
import java.io.OutputStreamWriter
import java.nio.charset.StandardCharsets
import java.nio.file.AtomicMoveNotSupportedException
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.util.UUID
import java.util.concurrent.CancellationException

enum class V2StagingDatabaseFailure {
    SOURCE_MISSING,
    BASE_GENERATION_CHANGED,
    PARTIAL_STATE,
    INTEGRITY_CHECK_FAILED,
}

class V2StagingDatabaseException(
    val reason: V2StagingDatabaseFailure,
    message: String,
    cause: Throwable? = null,
) : IllegalStateException(message, cause)

data class V2JobPrivateDatabaseBinding(
    val schemaVersion: Int,
    val jobId: String,
    val jobSpecId: String,
    val baseGenerationId: String?,
    val sourceDatabaseByteLength: Long,
    val sourceDatabaseSha256: String,
    /** Present in schema v2; v1 bindings remain valid for ordinary in-flight jobs. */
    val baseManifestSha256: String? = null,
    val baseDatabaseContentSha256: String? = null,
    val bindingId: String? = null,
)

data class V2PrivateDatabaseProgress(
    val detail: String,
    val completedBytes: Long? = null,
    val totalBytes: Long? = null,
)

/** Owns an isolated SQLite snapshot for one immutable indexing job. */
class V2JobPrivateDatabaseStore(
    private val filesDir: File,
    private val gson: Gson = GsonBuilder().disableHtmlEscaping().create(),
) {
    private val root = File(filesDir, "indexing_v2/job-databases")

    @Synchronized
    fun prepare(
        ledger: IndexingJobLedger,
        onProgress: (V2PrivateDatabaseProgress) -> Unit = {},
    ): File {
        V2IndexingLedgerValidator.requireValid(ledger)
        V2IndexingPlanFinalizationPolicy.requireStagingDatabaseReady(ledger)
        require(ledger.state != IndexingJobState.COMPLETE &&
            ledger.state != IndexingJobState.CANCELLED
        ) { "terminal jobs do not own staging databases" }
        require(root.isDirectory || root.mkdirs()) { "cannot create staging database root $root" }

        val database = databaseFile(ledger.jobSpec.jobId)
        val metadata = metadataFile(ledger.jobSpec.jobId)
        val hasDurableCommit = ledger.tracks.any {
            it.checkpoint == TrackCheckpoint.COMMITTED ||
                it.verifiedArtifacts.any { artifact ->
                    artifact.kind == VerifiedArtifactKind.DATABASE_COMMIT
                }
        }
        if (database.isFile && metadata.baseFile.exists()) {
            onProgress(V2PrivateDatabaseProgress("Opening the saved staging-database binding"))
            val binding = readBinding(metadata)
            requireBindingMatches(binding, ledger)
            onProgress(V2PrivateDatabaseProgress("Running SQLite quick_check on the saved staging database"))
            requireDatabaseIntegrity(database)
            return database
        }
        if (hasDurableCommit) {
            throw V2StagingDatabaseException(
                V2StagingDatabaseFailure.PARTIAL_STATE,
                "job ${ledger.jobSpec.jobId} has committed receipts but its private database " +
                    "or binding is missing",
            )
        }

        deleteDatabaseFamily(database)
        metadata.delete()
        onProgress(V2PrivateDatabaseProgress("Resolving the exact active music index used by this run"))
        val source = resolveSource(ledger, onProgress)
        val sourceLength = source.file.length()
        val sourceSha256 = source.databaseSha256
        val binding = V2JobPrivateDatabaseBinding(
            schemaVersion = BINDING_SCHEMA_VERSION,
            jobId = ledger.jobSpec.jobId,
            jobSpecId = ledger.jobSpec.specId,
            baseGenerationId = ledger.jobSpec.baseGenerationId,
            sourceDatabaseByteLength = sourceLength,
            sourceDatabaseSha256 = sourceSha256,
            baseManifestSha256 = source.baseManifestSha256,
            baseDatabaseContentSha256 = source.baseDatabaseContentSha256,
            bindingId = V2JobPrivateDatabaseBindingIdentity.compute(
                jobId = ledger.jobSpec.jobId,
                jobSpecId = ledger.jobSpec.specId,
                baseGenerationId = ledger.jobSpec.baseGenerationId,
                sourceDatabaseByteLength = sourceLength,
                sourceDatabaseSha256 = sourceSha256,
                baseManifestSha256 = source.baseManifestSha256,
                baseDatabaseContentSha256 = source.baseDatabaseContentSha256,
            ),
        )
        snapshot(source.file, database, onProgress)
        onProgress(V2PrivateDatabaseProgress("Running SQLite quick_check on the new staging database"))
        requireDatabaseIntegrity(database)
        onProgress(V2PrivateDatabaseProgress("Saving the staging database's exact base binding"))
        writeBinding(metadata, binding)
        return database
    }

    @Synchronized
    fun requirePrepared(ledger: IndexingJobLedger): File {
        V2IndexingPlanFinalizationPolicy.requireStagingDatabaseReady(ledger)
        val database = databaseFile(ledger.jobSpec.jobId)
        val metadata = metadataFile(ledger.jobSpec.jobId)
        if (!database.isFile || !metadata.baseFile.exists()) {
            throw V2StagingDatabaseException(
                V2StagingDatabaseFailure.PARTIAL_STATE,
                "job-private database is not durably prepared",
            )
        }
        requireBindingMatches(readBinding(metadata), ledger)
        requireDatabaseIntegrity(database)
        return database
    }

    /** Exact immutable-base binding used by destructive imported-row commit authorization. */
    @Synchronized
    fun requirePreparedBinding(ledger: IndexingJobLedger): V2JobPrivateDatabaseBinding {
        requirePrepared(ledger)
        return readBinding(metadataFile(ledger.jobSpec.jobId)).also { binding ->
            requireBindingMatches(binding, ledger)
        }
    }

    @Synchronized
    fun cleanup(jobId: String) {
        require(SAFE_JOB_ID.matches(jobId)) { "unsafe job id" }
        val database = databaseFile(jobId)
        deleteDatabaseFamily(database)
        val metadata = File(root, "$jobId.binding.json")
        val disposableFiles = listOf(
            database,
            File(database.path + "-wal"),
            File(database.path + "-shm"),
            File(database.path + "-journal"),
            metadata,
            File(metadata.path + ".new"),
            File(metadata.path + ".bak"),
        )
        disposableFiles.drop(4).forEach { file ->
            if (file.exists() && !file.delete()) throw IOException("unable to delete $file")
        }
        if (disposableFiles.any(File::exists)) {
            throw IOException("unable to remove private database for job $jobId")
        }
    }

    private fun resolveSource(
        ledger: IndexingJobLedger,
        onProgress: (V2PrivateDatabaseProgress) -> Unit,
    ): ResolvedSource {
        val baseGenerationId = requireNotNull(ledger.jobSpec.baseGenerationId) {
            "indexing requires an active published music index"
        }
        val active = try {
            V2IndexGenerationReader.requireActive(filesDir) { progress ->
                onProgress(
                    V2PrivateDatabaseProgress(
                        detail = "Hashing active music-index file ${progress.filename}",
                        completedBytes = progress.completedBytes,
                        totalBytes = progress.totalBytes,
                    ),
                )
            }
        } catch (cancelled: CancellationException) {
            throw cancelled
        } catch (error: Exception) {
            throw V2StagingDatabaseException(
                V2StagingDatabaseFailure.BASE_GENERATION_CHANGED,
                "unable to resolve planned base generation $baseGenerationId",
                error,
            )
        }
        if (active.manifest.generationId != baseGenerationId) {
            throw V2StagingDatabaseException(
                V2StagingDatabaseFailure.BASE_GENERATION_CHANGED,
                "planned base generation $baseGenerationId is no longer active",
            )
        }
        return ResolvedSource(
            file = active.databaseFile,
            databaseSha256 = active.manifest.databaseSha256,
            baseManifestSha256 = active.manifestSha256,
            baseDatabaseContentSha256 = active.manifest.databaseContentSha256,
        )
    }

    private fun snapshot(
        source: File,
        target: File,
        onProgress: (V2PrivateDatabaseProgress) -> Unit,
    ) {
        val temporary = File(root, ".${target.name}.incomplete-${UUID.randomUUID()}")
        try {
            val totalBytes = source.length()
            require(source.isFile && totalBytes > 0L) { "active generation database is missing" }
            onProgress(
                V2PrivateDatabaseProgress(
                    detail = "Copying the immutable active index into private staging",
                    completedBytes = 0L,
                    totalBytes = totalBytes,
                ),
            )
            FileInputStream(source).channel.use { input ->
                FileOutputStream(temporary).channel.use { output ->
                    var copiedBytes = 0L
                    while (copiedBytes < totalBytes) {
                        val copied = input.transferTo(
                            copiedBytes,
                            totalBytes - copiedBytes,
                            output,
                        )
                        require(copied > 0L) { "private database copy made no progress" }
                        copiedBytes += copied
                        onProgress(
                            V2PrivateDatabaseProgress(
                                detail = "Copying the immutable active index into private staging",
                                completedBytes = copiedBytes,
                                totalBytes = totalBytes,
                            ),
                        )
                    }
                    output.force(true)
                }
            }
            if (!temporary.isFile || temporary.length() != totalBytes) {
                throw V2StagingDatabaseException(
                    V2StagingDatabaseFailure.PARTIAL_STATE,
                    "The private database copy is incomplete",
                )
            }
            onProgress(V2PrivateDatabaseProgress("Publishing the complete staging database file"))
            moveAtomically(temporary, target)
        } finally {
            temporary.delete()
        }
    }

    private fun requireDatabaseIntegrity(databaseFile: File) {
        try {
            SQLiteDatabase.openDatabase(
                databaseFile.path,
                null,
                SQLiteDatabase.OPEN_READONLY,
            ).use { database ->
                database.rawQuery("PRAGMA quick_check(1)", null).use { cursor ->
                    if (!cursor.moveToFirst() || cursor.getString(0) != "ok") {
                        throw V2StagingDatabaseException(
                            V2StagingDatabaseFailure.INTEGRITY_CHECK_FAILED,
                            "job-private database integrity check failed",
                        )
                    }
                }
            }
        } catch (error: V2StagingDatabaseException) {
            throw error
        } catch (error: Exception) {
            throw V2StagingDatabaseException(
                V2StagingDatabaseFailure.INTEGRITY_CHECK_FAILED,
                "unable to verify job-private database",
                error,
            )
        }
    }

    private fun requireBindingMatches(
        binding: V2JobPrivateDatabaseBinding,
        ledger: IndexingJobLedger,
    ) {
        if (binding.schemaVersion !in SUPPORTED_BINDING_SCHEMA_VERSIONS ||
            binding.jobId != ledger.jobSpec.jobId ||
            binding.jobSpecId != ledger.jobSpec.specId ||
            binding.baseGenerationId != ledger.jobSpec.baseGenerationId ||
            binding.sourceDatabaseByteLength <= 0L ||
            !binding.sourceDatabaseSha256.matches(SHA256)
        ) {
            throw V2StagingDatabaseException(
                V2StagingDatabaseFailure.PARTIAL_STATE,
                "job-private database binding does not match the immutable job",
            )
        }
        if (binding.schemaVersion >= 2 &&
            (binding.bindingId == null || !binding.bindingId.matches(SHA256) ||
                binding.bindingId != V2JobPrivateDatabaseBindingIdentity.compute(
                    jobId = binding.jobId,
                    jobSpecId = binding.jobSpecId,
                    baseGenerationId = binding.baseGenerationId,
                    sourceDatabaseByteLength = binding.sourceDatabaseByteLength,
                    sourceDatabaseSha256 = binding.sourceDatabaseSha256,
                    baseManifestSha256 = binding.baseManifestSha256,
                    baseDatabaseContentSha256 = binding.baseDatabaseContentSha256,
                ) ||
                (binding.baseGenerationId != null &&
                    (binding.baseManifestSha256?.matches(SHA256) != true ||
                        binding.baseDatabaseContentSha256?.matches(SHA256) != true)))
        ) {
            throw V2StagingDatabaseException(
                V2StagingDatabaseFailure.PARTIAL_STATE,
                "job-private database v2 binding is incomplete or stale",
            )
        }
    }

    private fun readBinding(file: AtomicFile): V2JobPrivateDatabaseBinding = try {
        file.openRead().bufferedReader(StandardCharsets.UTF_8).use { reader ->
            gson.fromJson(reader, V2JobPrivateDatabaseBinding::class.java)
        } ?: throw IOException("empty private database binding")
    } catch (error: Exception) {
        throw V2StagingDatabaseException(
            V2StagingDatabaseFailure.PARTIAL_STATE,
            "unable to read job-private database binding",
            error,
        )
    }

    private fun writeBinding(file: AtomicFile, binding: V2JobPrivateDatabaseBinding) {
        val stream = file.startWrite()
        try {
            val writer = OutputStreamWriter(stream, StandardCharsets.UTF_8)
            gson.toJson(binding, writer)
            writer.flush()
            file.finishWrite(stream)
        } catch (error: Throwable) {
            file.failWrite(stream)
            throw IOException("unable to persist job-private database binding", error)
        }
    }

    private fun moveAtomically(source: File, destination: File) {
        try {
            Files.move(
                source.toPath(),
                destination.toPath(),
                StandardCopyOption.ATOMIC_MOVE,
            )
        } catch (_: AtomicMoveNotSupportedException) {
            Files.move(source.toPath(), destination.toPath())
        }
    }

    private fun deleteDatabaseFamily(database: File) {
        listOf(
            database,
            File(database.path + "-wal"),
            File(database.path + "-shm"),
            File(database.path + "-journal"),
        ).forEach { file ->
            if (file.exists() && !file.delete()) throw IOException("unable to delete $file")
        }
    }

    private fun databaseFile(jobId: String): File {
        require(SAFE_JOB_ID.matches(jobId)) { "unsafe job id" }
        return File(root, "$jobId.db")
    }

    private fun metadataFile(jobId: String): AtomicFile =
        AtomicFile(File(root, "$jobId.binding.json"))

    private companion object {
        const val BINDING_SCHEMA_VERSION = 2
        val SUPPORTED_BINDING_SCHEMA_VERSIONS = 1..BINDING_SCHEMA_VERSION
        val SAFE_JOB_ID = Regex("^[A-Za-z0-9._-]{1,128}$")
        val SHA256 = Regex("^[0-9a-f]{64}$")
    }

    private data class ResolvedSource(
        val file: File,
        val databaseSha256: String,
        val baseManifestSha256: String?,
        val baseDatabaseContentSha256: String?,
    )
}
