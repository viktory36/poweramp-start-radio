package com.powerampstartradio.indexing.v2

import android.content.ContentValues
import android.database.sqlite.SQLiteDatabase
import android.system.Os
import android.system.OsConstants
import android.util.AtomicFile
import android.util.Log
import com.google.gson.Gson
import com.google.gson.GsonBuilder
import com.powerampstartradio.data.EmbeddingDatabase
import com.powerampstartradio.indexing.V2ExactGraphIncrementalBase
import com.powerampstartradio.indexing.V2GraphExactProof
import com.powerampstartradio.indexing.V2GraphUpdateStrategy
import com.powerampstartradio.indexing.V2PreparedGraphUpdate
import com.powerampstartradio.poweramp.TrackNormalization
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.OutputStreamWriter
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.charset.StandardCharsets
import java.nio.file.AtomicMoveNotSupportedException
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.security.MessageDigest
import java.util.UUID

enum class V2IndexGenerationGraphPolicy {
    ABSENT,
    VALIDATED_COMPATIBILITY_IMPORT,
    BASE_BOUND_DELETION_REPAIR,
    BASE_BOUND_ADDITION_UPDATE,
    EXPLICIT_REBUILD,
}

enum class V2IndexGenerationOrigin {
    /** A normal immutable generation produced by a persisted V2 indexing job. */
    INDEXING_JOB,

    /** A validated legacy database whose existing vectors have no V2 production receipts. */
    BOOTSTRAP_COMPATIBILITY,

    /** A base-bound immutable subset created by an explicit library clean-up operation. */
    LIBRARY_MAINTENANCE,

    /** Exact server-produced CLaMP3 rows appended to one active generation. */
    SERVER_MERGE,
}

internal object V2DerivedGraphExactProofPolicy {
    fun shouldInstall(
        origin: V2IndexGenerationOrigin,
        graphPresent: Boolean,
        baseHasExactProof: Boolean,
    ): Boolean = when (origin) {
        V2IndexGenerationOrigin.BOOTSTRAP_COMPATIBILITY -> false
        V2IndexGenerationOrigin.LIBRARY_MAINTENANCE ->
            graphPresent && baseHasExactProof
        V2IndexGenerationOrigin.SERVER_MERGE -> {
            require(graphPresent) { "server merge must publish an exact updated graph" }
            require(baseHasExactProof) {
                "server merge requires an exact active graph induction base"
            }
            true
        }
        V2IndexGenerationOrigin.INDEXING_JOB ->
            error("indexing jobs do not use the derived-generation proof policy")
    }
}

data class V2IndexGenerationGraphBinding(
    val relativePath: String,
    val byteLength: Long,
    val sha256: String,
    val nodeCount: Int,
    val neighborsPerNode: Int,
    val orderedTrackSetSha256: String,
)

/** Exact coverage of stable V2 receipts; legacy rows are represented as uncovered. */
data class V2StableTrackUidCoverageBinding(
    val coveredTrackCount: Int,
    val uncoveredTrackCount: Int,
    val uniqueStableTrackSpanCount: Int,
    val fullContentIdentityCount: Int,
    val sampledContentIdentityCount: Int,
    val mappingSha256: String,
)

/** Inherited vectors with no V2 receipt. Their preprocessing/model provenance is unknown. */
data class V2CompatibilityBaseEmbeddingCoverageBinding(
    val provenancePolicyId: String,
    val trackCount: Int,
    val orderedContentSha256: String,
)

/** Exact per-row claim boundary: only receipt-bound rows carry a V2 embedding spec. */
data class V2EmbeddingSpecCoverageBinding(
    val totalTrackCount: Int,
    val receiptBoundTrackCount: Int,
    val receiptSpecTrackCounts: Map<String, Int>,
    val compatibilityBase: V2CompatibilityBaseEmbeddingCoverageBinding?,
    val mappingSha256: String,
)

data class V2IndexGenerationManifest(
    val schemaVersion: Int,
    val origin: V2IndexGenerationOrigin,
    val generationId: String,
    val activationBindingId: String,
    val jobId: String,
    val jobSpecId: String,
    val receiptEmbeddingSpec: EmbeddingSpecFingerprint,
    val textRetrievalSpec: TextRetrievalSpecFingerprint,
    val baseGenerationId: String?,
    val rebuildDerivedIndexes: Boolean,
    val graphPolicy: V2IndexGenerationGraphPolicy,
    /** Deterministic provenance timestamp; excluded from generation identity. */
    val createdAtEpochMs: Long,
    val databaseRelativePath: String,
    val databaseByteLength: Long,
    val databaseSha256: String,
    val databaseContentSha256: String,
    val orderedTrackSetSha256: String,
    val stableTrackUidCoverage: V2StableTrackUidCoverageBinding,
    val embeddingCoverage: V2EmbeddingSpecCoverageBinding,
    val trackCount: Int,
    val embeddingDimension: Int,
    val embeddingRelativePath: String,
    val embeddingByteLength: Long,
    val embeddingSha256: String,
    val graph: V2IndexGenerationGraphBinding?,
)

data class V2ActiveGenerationPointer(
    val schemaVersion: Int,
    val generationId: String,
    val manifestSha256: String,
)

data class V2ResolvedActiveIndexGeneration(
    val manifest: V2IndexGenerationManifest,
    val manifestSha256: String,
    val directory: File,
    val databaseFile: File,
    val embeddingFile: File,
    val graphFile: File?,
) {
    fun activatedEvidence(activatedAtEpochMs: Long): ActivatedGenerationEvidence =
        ActivatedGenerationEvidence(
            generationId = manifest.generationId,
            activationBindingId = manifest.activationBindingId,
            jobSpecId = manifest.jobSpecId,
            receiptEmbeddingSpecId = manifest.receiptEmbeddingSpec.specId,
            textRetrievalSpecId = manifest.textRetrievalSpec.specId,
            baseGenerationId = manifest.baseGenerationId,
            rebuildDerivedIndexes = manifest.rebuildDerivedIndexes,
            manifestSha256 = manifestSha256,
            databaseSha256 = manifest.databaseSha256,
            databaseContentSha256 = manifest.databaseContentSha256,
            orderedTrackSetSha256 = manifest.orderedTrackSetSha256,
            stableTrackUidMappingSha256 = manifest.stableTrackUidCoverage.mappingSha256,
            embeddingSha256 = manifest.embeddingSha256,
            graphSha256 = manifest.graph?.sha256,
            activatedAtEpochMs = activatedAtEpochMs,
        )
}

data class V2OrderedEmbeddingBinding(
    val trackCount: Int,
    val dimension: Int,
    val byteLength: Long,
    val fileSha256: String,
    val orderedTrackSetSha256: String,
    val databaseContentSha256: String,
)

data class V2DatabaseEmbeddingBinding(
    val trackCount: Int,
    val dimension: Int,
    val orderedTrackSetSha256: String,
    val databaseContentSha256: String,
)

fun interface V2OrderedEmbeddingConsumer {
    fun accept(trackId: Long, embedding: ByteArray)
}

interface V2OrderedEmbeddingSource {
    val trackCount: Int
    val dimension: Int
    fun forEachOrdered(consumer: V2OrderedEmbeddingConsumer)
}

class V2SqliteOrderedEmbeddingSource(
    private val database: SQLiteDatabase,
    private val table: String = V2EmbeddingCommitRepository.EMBEDDING_TABLE,
) : V2OrderedEmbeddingSource {
    init {
        require(SAFE_TABLE.matches(table)) { "unsafe embedding table" }
    }

    override val trackCount: Int
        get() = database.rawQuery("SELECT COUNT(*) FROM [$table]", null).use { cursor ->
            check(cursor.moveToFirst()) { "embedding count query returned no row" }
            cursor.getInt(0)
        }

    override val dimension: Int
        get() = database.rawQuery(
            "SELECT length(embedding) FROM [$table] ORDER BY track_id LIMIT 1",
            null,
        ).use { cursor ->
            check(cursor.moveToFirst() && !cursor.isNull(0)) { "embedding table is empty" }
            val bytes = cursor.getInt(0)
            check(bytes > 0 && bytes % Float.SIZE_BYTES == 0) {
                "invalid embedding byte length $bytes"
            }
            bytes / Float.SIZE_BYTES
        }

    override fun forEachOrdered(consumer: V2OrderedEmbeddingConsumer) {
        database.rawQuery(
            "SELECT track_id, embedding FROM [$table] ORDER BY track_id",
            null,
        ).use { cursor ->
            while (cursor.moveToNext()) {
                consumer.accept(cursor.getLong(0), cursor.getBlob(1))
            }
        }
    }

    private companion object {
        val SAFE_TABLE = Regex("^[A-Za-z0-9_]+$")
    }
}

/** Exact active-pointer value observed before a potentially long publication operation. */
class V2GenerationPublicationExpectation internal constructor(
    internal val pointer: V2ActiveGenerationPointer?,
)

class V2GenerationPublicationConflictException(message: String) :
    IllegalStateException(message)

enum class V2GenerationOrphanReconciliationSkipReason {
    NO_ACTIVE_GENERATION,
    ACTIVE_GENERATION_UNREADABLE,
    ACTIVE_INDEXING_JOB_UNREADABLE,
}

data class V2GenerationOrphanReconciliationResult(
    val deletedGenerationIds: List<String> = emptyList(),
    val retainedJobGenerationIds: List<String> = emptyList(),
    val retainedUnverifiedGenerationIds: List<String> = emptyList(),
    val failedDeletionGenerationIds: List<String> = emptyList(),
    val skipReason: V2GenerationOrphanReconciliationSkipReason? = null,
)

data class V2AbandonedGenerationStagingCleanupResult(
    val deletedDirectoryCount: Int,
    val failedDirectoryCount: Int,
)

/** Pure fail-closed ownership rule shared by commit pruning and cold-process reconciliation. */
object V2GenerationOrphanRetentionPolicy {
    fun retainUnreferenced(
        manifest: V2IndexGenerationManifest?,
        protectedNonterminalJobIds: Set<String>,
    ): Boolean = manifest == null ||
        (manifest.origin == V2IndexGenerationOrigin.INDEXING_JOB &&
            manifest.jobId in protectedNonterminalJobIds)
}

internal fun interface V2PreexistingGenerationValidator {
    fun requireValid(
        directory: File,
        expected: V2IndexGenerationManifest,
        expectedManifestSha256: String,
        gson: Gson,
        onArtifactHashProgress: (V2GenerationArtifactHashProgress) -> Unit,
    ): V2ResolvedActiveIndexGeneration
}

private object V2ExactPreexistingGenerationValidator : V2PreexistingGenerationValidator {
    override fun requireValid(
        directory: File,
        expected: V2IndexGenerationManifest,
        expectedManifestSha256: String,
        gson: Gson,
        onArtifactHashProgress: (V2GenerationArtifactHashProgress) -> Unit,
    ): V2ResolvedActiveIndexGeneration = V2IndexGenerationReader.requireGenerationDirectory(
        directory = directory,
        expected = expected,
        expectedManifestSha256 = expectedManifestSha256,
        gson = gson,
        onArtifactHashProgress = onArtifactHashProgress,
    ).also {
        V2IndexGenerationReader.requireDatabaseCoherence(directory, expected)
    }
}

internal object V2InstalledGenerationResolutionPolicy {
    fun <T> resolve(
        installedByThisCall: Boolean,
        freshlyInstalled: () -> T,
        preexisting: () -> T,
    ): T = if (installedByThisCall) freshlyInstalled() else preexisting()
}

internal enum class V2GenerationPointerReadPurpose {
    PUBLICATION_CAS,
    CRASH_RECOVERY,
}

internal object V2GenerationPointerReadPolicy {
    fun <T> resolve(
        purpose: V2GenerationPointerReadPurpose,
        pointerOnly: () -> T,
        exactGeneration: () -> T,
    ): T = when (purpose) {
        V2GenerationPointerReadPurpose.PUBLICATION_CAS -> pointerOnly()
        V2GenerationPointerReadPurpose.CRASH_RECOVERY -> exactGeneration()
    }
}

/** Pure compare-and-swap policy; the coordinator below supplies serialization and durable I/O. */
object V2GenerationPublicationPolicy {
    fun pointerForCommit(
        expected: V2ActiveGenerationPointer?,
        current: V2ActiveGenerationPointer?,
        generationId: String,
        manifestSha256: String,
    ): V2ActiveGenerationPointer {
        if (current != expected) {
            throw V2GenerationPublicationConflictException(
                "Active library changed while a new generation was being prepared",
            )
        }
        return V2ActiveGenerationPointer(
            schemaVersion = POINTER_SCHEMA_VERSION,
            generationId = generationId,
            manifestSha256 = manifestSha256,
        )
    }
}

/**
 * The sole active-generation commit point in this app process.
 *
 * Expensive preparation happens outside the lock in a dot-prefixed staging directory. The final
 * expectation check, content-addressed installation, destination binding, pointer replacement,
 * and pruning happen together. No exact generation directory is therefore visible to another
 * in-process commit or cleanup before its pointer decision.
 */
object V2GenerationPublicationCoordinator {
    private val commitLock = Any()
    private val publicationRootsTouched = mutableSetOf<String>()
    private val coldStartStagingRootsReconciled = mutableSetOf<String>()

    fun capture(
        filesDir: File,
        gson: Gson = GsonBuilder().disableHtmlEscaping().create(),
    ): V2GenerationPublicationExpectation = synchronized(commitLock) {
        markPublicationRootTouched(filesDir)
        V2GenerationPublicationExpectation(
            readCurrentPointer(
                filesDir,
                gson,
                V2GenerationPointerReadPurpose.PUBLICATION_CAS,
            ),
        )
    }

    /**
     * Must be called synchronously from Application.onCreate before any app work can publish.
     * Runtime cleanup deliberately ignores dot staging because preparation occurs outside the
     * publication lock.
     */
    fun reconcileAbandonedStagingAtColdProcessStart(
        filesDir: File,
    ): V2AbandonedGenerationStagingCleanupResult = synchronized(commitLock) {
        val rootPath = canonicalRootPath(filesDir)
        check(rootPath !in publicationRootsTouched) {
            "generation publication already started in this process"
        }
        if (!coldStartStagingRootsReconciled.add(rootPath)) {
            return@synchronized V2AbandonedGenerationStagingCleanupResult(0, 0)
        }
        val root = File(filesDir, GENERATIONS_DIRECTORY)
        var deleted = 0
        var failed = 0
        root.listFiles().orEmpty()
            .filter { it.isDirectory && it.name.startsWith(GENERATION_STAGING_PREFIX) }
            .forEach { directory ->
                if (runCatching { directory.deleteRecursively() }.getOrDefault(false)) {
                    deleted++
                } else {
                    failed++
                }
            }
        if (deleted > 0) runCatching { syncDirectory(root) }
        V2AbandonedGenerationStagingCleanupResult(deleted, failed)
    }

    /** Installs a fully prepared generation and publishes its pointer as one serialized action. */
    internal fun installAndCommit(
        filesDir: File,
        expected: V2GenerationPublicationExpectation,
        stagingDirectory: File,
        manifest: V2IndexGenerationManifest,
        manifestSha256: String,
        gson: Gson = GsonBuilder().disableHtmlEscaping().create(),
        afterInstallBeforePointerPublication: (V2IndexGenerationManifest) -> Unit = {},
        onArtifactHashProgress: (V2GenerationArtifactHashProgress) -> Unit = {},
        preexistingGenerationValidator: V2PreexistingGenerationValidator =
            V2ExactPreexistingGenerationValidator,
    ): V2ResolvedActiveIndexGeneration = synchronized(commitLock) {
        markPublicationRootTouched(filesDir)
        require(manifest.generationId.matches(GENERATION_ID) &&
            manifestSha256.matches(SHA256)
        ) { "invalid prepared generation pointer binding" }
        val root = File(filesDir, GENERATIONS_DIRECTORY).canonicalFile
        val staging = stagingDirectory.canonicalFile
        require(staging.parentFile == root &&
            staging.name.startsWith(GENERATION_STAGING_PREFIX) && staging.isDirectory
        ) { "prepared generation is not in private generation staging" }

        val current = readCurrentPointer(
            filesDir,
            gson,
            V2GenerationPointerReadPurpose.PUBLICATION_CAS,
        )
        val next = V2GenerationPublicationPolicy.pointerForCommit(
            expected = expected.pointer,
            current = current,
            generationId = manifest.generationId,
            manifestSha256 = manifestSha256,
        )
        val destination = File(root, manifest.generationId)
        val installedByThisCall = !destination.exists()
        val preparedManifestByteLength = File(staging, MANIFEST_FILE).let { preparedManifest ->
            require(preparedManifest.isFile && preparedManifest.length() > 0L) {
                "prepared generation manifest is missing"
            }
            preparedManifest.length()
        }
        if (installedByThisCall) {
            moveGenerationDirectory(staging, destination)
            syncDirectory(root)
        }
        val resolved = V2InstalledGenerationResolutionPolicy.resolve(
            installedByThisCall = installedByThisCall,
            freshlyInstalled = {
                V2FreshlyInstalledGenerationBindingResolver.requireResolved(
                    directory = destination,
                    manifest = manifest,
                    manifestSha256 = manifestSha256,
                    manifestByteLength = preparedManifestByteLength,
                    gson = gson,
                )
            },
            preexisting = {
                preexistingGenerationValidator.requireValid(
                    directory = destination,
                    expected = manifest,
                    expectedManifestSha256 = manifestSha256,
                    gson = gson,
                    onArtifactHashProgress = onArtifactHashProgress,
                )
            },
        )
        afterInstallBeforePointerPublication(manifest)
        if (next != current) writePointer(filesDir, next, gson)
        V2IndexGenerationReader.rememberFreshlyPublished(next, resolved)
        pruneUnreferencedGenerations(
            filesDir,
            next,
            protectedNonterminalJobIds(filesDir),
            gson,
        )
        resolved
    }

    fun commit(
        filesDir: File,
        expected: V2GenerationPublicationExpectation,
        generationId: String,
        manifestSha256: String,
        gson: Gson = GsonBuilder().disableHtmlEscaping().create(),
    ): V2ActiveGenerationPointer = synchronized(commitLock) {
        markPublicationRootTouched(filesDir)
        require(generationId.matches(GENERATION_ID) && manifestSha256.matches(SHA256)) {
            "invalid target generation pointer binding"
        }
        val current = readCurrentPointer(
            filesDir,
            gson,
            V2GenerationPointerReadPurpose.PUBLICATION_CAS,
        )
        val next = V2GenerationPublicationPolicy.pointerForCommit(
            expected = expected.pointer,
            current = current,
            generationId = generationId,
            manifestSha256 = manifestSha256,
        )
        if (next != current) writePointer(filesDir, next, gson)
        pruneUnreferencedGenerations(
            filesDir,
            next,
            protectedNonterminalJobIds(filesDir),
            gson,
        )
        next
    }

    /** Reclaims complete crash orphans while protecting the active library and in-flight jobs. */
    fun reconcileCrashOrphans(
        filesDir: File,
        protectedNonterminalJobIds: Set<String>,
        gson: Gson = GsonBuilder().disableHtmlEscaping().create(),
    ): V2GenerationOrphanReconciliationResult = synchronized(commitLock) {
        markPublicationRootTouched(filesDir)
        val pointer = try {
            readCurrentPointer(
                filesDir,
                gson,
                V2GenerationPointerReadPurpose.CRASH_RECOVERY,
            )
                ?: return@synchronized V2GenerationOrphanReconciliationResult(
                    skipReason = V2GenerationOrphanReconciliationSkipReason.NO_ACTIVE_GENERATION,
                )
        } catch (_: Throwable) {
            return@synchronized V2GenerationOrphanReconciliationResult(
                skipReason = V2GenerationOrphanReconciliationSkipReason.ACTIVE_GENERATION_UNREADABLE,
            )
        }
        val durableProtectedJobIds = when (
            val inspection = V2ActiveIndexingJobPointer(filesDir).inspect()
        ) {
            is V2ActiveIndexingJobPointerInspection.Unreadable ->
                return@synchronized V2GenerationOrphanReconciliationResult(
                    skipReason =
                        V2GenerationOrphanReconciliationSkipReason.ACTIVE_INDEXING_JOB_UNREADABLE,
                )
            is V2ActiveIndexingJobPointerInspection.Readable ->
                protectedNonterminalJobIds + inspection.jobId
            V2ActiveIndexingJobPointerInspection.Missing -> protectedNonterminalJobIds
        }

        val root = File(filesDir, GENERATIONS_DIRECTORY)
        val retainedByPointer = setOf(pointer.generationId)
        val deleted = mutableListOf<String>()
        val retainedJobs = mutableListOf<String>()
        val retainedUnverified = mutableListOf<String>()
        val failed = mutableListOf<String>()
        root.listFiles().orEmpty()
            .filter { it.isDirectory && GENERATION_ID.matches(it.name) && it.name !in retainedByPointer }
            .sortedBy { it.name }
            .forEach { directory ->
                val manifest = readManifestForOrphanClassification(directory, gson)
                if (V2GenerationOrphanRetentionPolicy.retainUnreferenced(
                        manifest,
                        durableProtectedJobIds,
                    )
                ) {
                    if (manifest == null) {
                        retainedUnverified += directory.name
                    } else {
                        val valid = runCatching {
                            V2IndexGenerationReader.requireGenerationDirectory(
                                directory = directory,
                                expected = manifest,
                                gson = gson,
                            ).also {
                                V2IndexGenerationReader.requireDatabaseCoherence(directory, manifest)
                            }
                        }.isSuccess
                        if (valid) retainedJobs += directory.name
                        else retainedUnverified += directory.name
                    }
                } else if (runCatching { directory.deleteRecursively() }.getOrDefault(false)) {
                    deleted += directory.name
                } else {
                    failed += directory.name
                }
            }
        if (deleted.isNotEmpty()) runCatching { syncDirectory(root) }
        V2GenerationOrphanReconciliationResult(
            deletedGenerationIds = deleted,
            retainedJobGenerationIds = retainedJobs,
            retainedUnverifiedGenerationIds = retainedUnverified,
            failedDeletionGenerationIds = failed,
        )
    }

    private fun readCurrentPointer(
        filesDir: File,
        gson: Gson,
        purpose: V2GenerationPointerReadPurpose,
    ): V2ActiveGenerationPointer? {
        val root = File(filesDir, GENERATIONS_DIRECTORY)
        val pointerFile = File(root, ACTIVE_POINTER_FILE)
        if (!pointerFile.isFile && !File(pointerFile.path + ".bak").isFile) return null
        val pointer = V2IndexGenerationReader.requireActivePointer(filesDir, gson)
        return V2GenerationPointerReadPolicy.resolve(
            purpose = purpose,
            pointerOnly = { pointer },
            exactGeneration = {
                val active = V2IndexGenerationReader.requireActive(filesDir, gson)
                require(active.manifest.generationId == pointer.generationId &&
                    active.manifestSha256 == pointer.manifestSha256
                ) { "current pointer does not resolve to its exact generation" }
                pointer
            },
        )
    }

    private fun writePointer(
        filesDir: File,
        value: V2ActiveGenerationPointer,
        gson: Gson,
    ) {
        val root = File(filesDir, GENERATIONS_DIRECTORY)
        require(root.isDirectory || root.mkdirs()) { "cannot create generation root $root" }
        val pointer = AtomicFile(File(root, ACTIVE_POINTER_FILE))
        val stream = pointer.startWrite()
        try {
            val writer = OutputStreamWriter(stream, StandardCharsets.UTF_8)
            gson.toJson(value, writer)
            writer.flush()
            pointer.finishWrite(stream)
            syncDirectory(root)
        } catch (error: Throwable) {
            pointer.failWrite(stream)
            throw error
        }
    }

    /** Pointer commit comes first; interrupted pruning can only leave harmless unreferenced data. */
    private fun pruneUnreferencedGenerations(
        filesDir: File,
        pointer: V2ActiveGenerationPointer,
        protectedNonterminalJobIds: Set<String>?,
        gson: Gson,
    ) {
        if (protectedNonterminalJobIds == null) return
        val root = File(filesDir, GENERATIONS_DIRECTORY)
        val retained = setOf(pointer.generationId)
        root.listFiles().orEmpty()
            .filter { directory ->
                directory.isDirectory && GENERATION_ID.matches(directory.name) &&
                    directory.name !in retained &&
                    !V2GenerationOrphanRetentionPolicy.retainUnreferenced(
                        readManifestForOrphanClassification(directory, gson),
                        protectedNonterminalJobIds,
                    )
            }
            .forEach { directory -> runCatching { directory.deleteRecursively() } }
        runCatching { syncDirectory(root) }
    }

    private fun protectedNonterminalJobIds(filesDir: File): Set<String>? =
        when (val inspection = V2ActiveIndexingJobPointer(filesDir).inspect()) {
            V2ActiveIndexingJobPointerInspection.Missing -> emptySet()
            is V2ActiveIndexingJobPointerInspection.Readable -> setOf(inspection.jobId)
            is V2ActiveIndexingJobPointerInspection.Unreadable -> null
        }

    private fun readManifestForOrphanClassification(
        directory: File,
        gson: Gson,
    ): V2IndexGenerationManifest? = runCatching {
        File(directory, MANIFEST_FILE).bufferedReader(StandardCharsets.UTF_8).use { reader ->
            gson.fromJson(reader, V2IndexGenerationManifest::class.java)
        }?.takeIf { manifest ->
            manifest.generationId == directory.name &&
                manifest.generationId == V2IndexGenerationIdentity.generationId(manifest)
        }
    }.getOrNull()

    private fun markPublicationRootTouched(filesDir: File) {
        publicationRootsTouched += canonicalRootPath(filesDir)
    }

    private fun canonicalRootPath(filesDir: File): String =
        File(filesDir, GENERATIONS_DIRECTORY).canonicalPath
}

/** Exact PEMB writer/inspector shared by activation and reader validation. */
object V2EmbeddingGenerationFile {
    private const val MAGIC = 0x424D4550
    private const val VERSION = 1
    private const val HEADER_BYTES = 16

    fun write(
        source: V2OrderedEmbeddingSource,
        target: File,
        onRowProgress: ((completedRows: Int, totalRows: Int) -> Unit)? = null,
        onHashProgress: ((completedBytes: Long, totalBytes: Long) -> Unit)? = null,
    ): V2OrderedEmbeddingBinding {
        val count = source.trackCount
        val dimension = source.dimension
        requireShape(count, dimension)
        val recordBytes = Math.multiplyExact(dimension, Float.SIZE_BYTES)
        val totalLength = Math.addExact(
            HEADER_BYTES.toLong(),
            Math.addExact(
                Math.multiplyExact(count.toLong(), Long.SIZE_BYTES.toLong()),
                Math.multiplyExact(count.toLong(), recordBytes.toLong()),
            ),
        )
        target.parentFile?.let { parent ->
            require(parent.isDirectory || parent.mkdirs()) { "cannot create $parent" }
        }
        val digests = OrderedEmbeddingDigests(count, dimension)
        onRowProgress?.invoke(0, count)
        RandomAccessFile(target, "rw").use { output ->
            output.setLength(totalLength)
            output.seek(0L)
            output.write(littleEndianHeader(count, dimension))
            val embeddingsStart = HEADER_BYTES.toLong() + count.toLong() * Long.SIZE_BYTES
            var index = 0
            var previousId = Long.MIN_VALUE
            source.forEachOrdered(V2OrderedEmbeddingConsumer { trackId, embedding ->
                require(index < count) { "source returned more than $count embeddings" }
                requireOrderedRow(trackId, previousId, embedding, recordBytes)
                output.seek(HEADER_BYTES.toLong() + index.toLong() * Long.SIZE_BYTES)
                output.writeLongLittleEndian(trackId)
                output.seek(embeddingsStart + index.toLong() * recordBytes.toLong())
                output.write(embedding)
                digests.update(trackId, embedding)
                previousId = trackId
                index++
                if (index == count || index % 4096 == 0) {
                    onRowProgress?.invoke(index, count)
                }
            })
            require(index == count) { "source returned $index embeddings, expected $count" }
            output.fd.sync()
        }
        require(target.length() == totalLength) { "PEMB length changed after publication" }
        val fileSha256 = if (onHashProgress == null) {
            V2FileSha256.digest(target)
        } else {
            V2FileSha256.digest(target, onHashProgress)
        }
        return digests.fileBinding(totalLength, fileSha256)
    }

    /** Streams semantic DB identity without writing a second PEMB file. */
    fun digest(source: V2OrderedEmbeddingSource): V2DatabaseEmbeddingBinding {
        val count = source.trackCount
        val dimension = source.dimension
        requireShape(count, dimension)
        val recordBytes = Math.multiplyExact(dimension, Float.SIZE_BYTES)
        val digests = OrderedEmbeddingDigests(count, dimension)
        var seen = 0
        var previousId = Long.MIN_VALUE
        source.forEachOrdered(V2OrderedEmbeddingConsumer { trackId, embedding ->
            require(seen < count) { "source returned more than $count embeddings" }
            requireOrderedRow(trackId, previousId, embedding, recordBytes)
            digests.update(trackId, embedding)
            previousId = trackId
            seen++
        })
        require(seen == count) { "source returned $seen embeddings, expected $count" }
        return digests.databaseBinding()
    }

    fun inspect(
        file: File,
        onRowProgress: ((completedRows: Int, totalRows: Int) -> Unit)? = null,
        onHashProgress: ((completedBytes: Long, totalBytes: Long) -> Unit)? = null,
    ): V2OrderedEmbeddingBinding {
        require(file.isFile && file.length() >= HEADER_BYTES) { "missing or short PEMB file" }
        RandomAccessFile(file, "r").use { input ->
            val header = ByteArray(HEADER_BYTES).also(input::readFully)
            val values = ByteBuffer.wrap(header).order(ByteOrder.LITTLE_ENDIAN)
            require(values.int == MAGIC) { "invalid PEMB magic" }
            require(values.int == VERSION) { "unsupported PEMB version" }
            val count = values.int
            val dimension = values.int
            requireShape(count, dimension)
            val recordBytes = dimension * Float.SIZE_BYTES
            val expectedLength = HEADER_BYTES.toLong() + count.toLong() * Long.SIZE_BYTES +
                count.toLong() * recordBytes.toLong()
            require(file.length() == expectedLength) {
                "PEMB length ${file.length()} != $expectedLength"
            }
            val fileDigest = MessageDigest.getInstance("SHA-256").apply { update(header) }
            var completedBytes = HEADER_BYTES.toLong()
            onHashProgress?.invoke(0L, expectedLength)
            val ids = LongArray(count)
            val idBytes = ByteArray(Long.SIZE_BYTES)
            val idValues = ByteBuffer.wrap(idBytes).order(ByteOrder.LITTLE_ENDIAN)
            repeat(count) { index ->
                input.readFully(idBytes)
                fileDigest.update(idBytes)
                ids[index] = idValues.getLong(0)
                completedBytes += idBytes.size
            }
            require(ids.all { it > 0L } &&
                (1 until ids.size).all { index -> ids[index - 1] < ids[index] }
            ) { "PEMB IDs are not strictly increasing" }
            val digests = OrderedEmbeddingDigests(count, dimension)
            val embedding = ByteArray(recordBytes)
            onRowProgress?.invoke(0, count)
            repeat(count) { index ->
                input.readFully(embedding)
                fileDigest.update(embedding)
                completedBytes += embedding.size
                V2Clamp3VectorCodec.requireValidBlob(embedding)
                digests.update(ids[index], embedding)
                val completedRows = index + 1
                if (completedRows == count || completedRows % 4096 == 0) {
                    onRowProgress?.invoke(completedRows, count)
                    onHashProgress?.invoke(completedBytes, expectedLength)
                }
            }
            require(completedBytes == expectedLength && input.filePointer == expectedLength) {
                "PEMB bytes changed during inspection"
            }
            return digests.fileBinding(expectedLength, fileDigest.digest().toV2CommitHex())
        }
    }

    private fun requireShape(count: Int, dimension: Int) {
        require(count > 0) { "cannot activate an empty embedding generation" }
        require(dimension == V2_CLAMP3_DIMENSION) {
            "expected $V2_CLAMP3_DIMENSION dimensions, got $dimension"
        }
    }

    private fun requireOrderedRow(
        trackId: Long,
        previousId: Long,
        embedding: ByteArray,
        recordBytes: Int,
    ) {
        require(trackId > 0L && trackId > previousId) {
            "embedding track IDs are not strictly increasing: $previousId then $trackId"
        }
        require(embedding.size == recordBytes) {
            "track $trackId has ${embedding.size} bytes, expected $recordBytes"
        }
        V2Clamp3VectorCodec.requireValidBlob(embedding)
    }

    private class OrderedEmbeddingDigests(count: Int, private val dimension: Int) {
        private val idsDigest = MessageDigest.getInstance("SHA-256").apply {
            updateLengthPrefixed("v2-ordered-track-set-v1")
            updateInt(count)
        }
        private val contentDigest = MessageDigest.getInstance("SHA-256").apply {
            updateLengthPrefixed("v2-ordered-clamp3-content-v1")
            updateInt(count)
            updateInt(dimension)
        }
        private val trackCount = count

        fun update(trackId: Long, embedding: ByteArray) {
            idsDigest.updateLong(trackId)
            contentDigest.updateLong(trackId)
            contentDigest.updateInt(embedding.size)
            contentDigest.update(embedding)
        }

        fun databaseBinding() = V2DatabaseEmbeddingBinding(
            trackCount = trackCount,
            dimension = dimension,
            orderedTrackSetSha256 = idsDigest.digest().toV2CommitHex(),
            databaseContentSha256 = contentDigest.digest().toV2CommitHex(),
        )

        fun fileBinding(byteLength: Long, sha256: String): V2OrderedEmbeddingBinding {
            val database = databaseBinding()
            return V2OrderedEmbeddingBinding(
                trackCount = database.trackCount,
                dimension = database.dimension,
                byteLength = byteLength,
                fileSha256 = sha256,
                orderedTrackSetSha256 = database.orderedTrackSetSha256,
                databaseContentSha256 = database.databaseContentSha256,
            )
        }
    }

    private fun littleEndianHeader(count: Int, dimension: Int): ByteArray =
        ByteBuffer.allocate(HEADER_BYTES).order(ByteOrder.LITTLE_ENDIAN)
            .putInt(MAGIC)
            .putInt(VERSION)
            .putInt(count)
            .putInt(dimension)
            .array()
}

object V2GraphGenerationFile {
    fun inspect(
        file: File,
        onRowProgress: ((completedRows: Int, totalRows: Int) -> Unit)? = null,
        onHashProgress: ((completedBytes: Long, totalBytes: Long) -> Unit)? = null,
    ): V2IndexGenerationGraphBinding {
        require(file.isFile && file.length() >= 8L) { "missing or short graph file" }
        RandomAccessFile(file, "r").use { input ->
            val header = ByteArray(8).also(input::readFully)
            val values = ByteBuffer.wrap(header).order(ByteOrder.LITTLE_ENDIAN)
            val nodes = values.int
            val neighbors = values.int
            require(nodes > 0 && neighbors > 0) { "invalid graph shape $nodes x $neighbors" }
            val expectedLength = 8L + nodes.toLong() * Long.SIZE_BYTES +
                nodes.toLong() * neighbors.toLong() * 8L
            require(file.length() == expectedLength) {
                "graph length ${file.length()} != $expectedLength"
            }
            val fileDigest = MessageDigest.getInstance("SHA-256").apply { update(header) }
            var completedBytes = header.size.toLong()
            onHashProgress?.invoke(0L, expectedLength)
            val digest = MessageDigest.getInstance("SHA-256").apply {
                updateLengthPrefixed("v2-ordered-track-set-v1")
                updateInt(nodes)
            }
            var previous = Long.MIN_VALUE
            onRowProgress?.invoke(0, nodes)
            val idBytes = ByteArray(Long.SIZE_BYTES)
            val idValues = ByteBuffer.wrap(idBytes).order(ByteOrder.LITTLE_ENDIAN)
            repeat(nodes) {
                input.readFully(idBytes)
                fileDigest.update(idBytes)
                completedBytes += idBytes.size
                val id = idValues.getLong(0)
                require(id > 0L && id > previous) { "graph IDs are not strictly increasing" }
                digest.updateLong(id)
                previous = id
            }
            val edgeBytes = ByteArray(Int.SIZE_BYTES + Float.SIZE_BYTES)
            val edgeValues = ByteBuffer.wrap(edgeBytes).order(ByteOrder.LITTLE_ENDIAN)
            repeat(nodes) { node ->
                var rowWeight = 0.0
                repeat(neighbors) { neighbor ->
                    input.readFully(edgeBytes)
                    fileDigest.update(edgeBytes)
                    completedBytes += edgeBytes.size
                    val neighborIndex = edgeValues.getInt(0)
                    val score = edgeValues.getFloat(Int.SIZE_BYTES)
                    require(neighborIndex in 0 until nodes) {
                        "graph edge $node:$neighbor has invalid neighbor index $neighborIndex"
                    }
                    require(score.isFinite() && score >= 0f) {
                        "graph edge $node:$neighbor has invalid weight $score"
                    }
                    rowWeight += score.toDouble()
                }
                require(kotlin.math.abs(rowWeight - 1.0) <= GRAPH_ROW_SUM_TOLERANCE) {
                    "graph row $node weights sum to $rowWeight, expected 1"
                }
                val completedRows = node + 1
                if (completedRows == nodes || completedRows % 4096 == 0) {
                    onRowProgress?.invoke(completedRows, nodes)
                    onHashProgress?.invoke(completedBytes, expectedLength)
                }
            }
            require(completedBytes == expectedLength && input.filePointer == expectedLength) {
                "graph bytes changed during inspection"
            }
            return V2IndexGenerationGraphBinding(
                relativePath = GRAPH_FILE,
                byteLength = file.length(),
                sha256 = fileDigest.digest().toV2CommitHex(),
                nodeCount = nodes,
                neighborsPerNode = neighbors,
                orderedTrackSetSha256 = digest.digest().toV2CommitHex(),
            )
        }
    }

    const val GRAPH_FILE = "graph.bin"
    private const val GRAPH_ROW_SUM_TOLERANCE = 0.005
}

object V2StableTrackUidCoverage {
    fun inspect(
        database: SQLiteDatabase,
        embeddingCount: Int,
        onRowProgress: ((completedRows: Int, totalRows: Int) -> Unit)? = null,
    ): V2StableTrackUidCoverageBinding {
        val receiptTable = V2EmbeddingCommitRepository.RECEIPT_TABLE
        val hasReceipts = database.rawQuery(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1",
            arrayOf(receiptTable),
        ).use { it.moveToFirst() }
        val digest = MessageDigest.getInstance("SHA-256").apply {
            updateLengthPrefixed("v2-stable-track-uid-coverage-v1")
            updateInt(embeddingCount)
        }
        var covered = 0
        var uncovered = 0
        var full = 0
        var sampled = 0
        val unique = mutableSetOf<String>()
        val query = if (hasReceipts) {
            """
            SELECT e.track_id, r.stable_track_span_id, r.stable_identity_spec_id,
                   r.stable_identity_strength
            FROM ${V2EmbeddingCommitRepository.EMBEDDING_TABLE} e
            LEFT JOIN $receiptTable r
              ON r.track_id = e.track_id
             AND r.receipt_schema_version = $V2_EMBEDDING_COMMIT_RECEIPT_SCHEMA_VERSION
            ORDER BY e.track_id
            """.trimIndent()
        } else {
            """
            SELECT track_id, NULL, NULL, NULL
            FROM ${V2EmbeddingCommitRepository.EMBEDDING_TABLE}
            ORDER BY track_id
            """.trimIndent()
        }
        onRowProgress?.invoke(0, embeddingCount)
        database.rawQuery(query, null).use { cursor ->
            var rows = 0
            var previousId = Long.MIN_VALUE
            while (cursor.moveToNext()) {
                val trackId = cursor.getLong(0)
                require(trackId > previousId) { "stable UID coverage rows are unordered" }
                digest.updateLong(trackId)
                if (cursor.isNull(1)) {
                    digest.update(0.toByte())
                    uncovered++
                } else {
                    val stableId = cursor.getString(1)
                    val specId = cursor.getString(2)
                    val strength = StableTrackSpanIdentityStrength.valueOf(cursor.getString(3))
                    require(stableId.matches(STABLE_ID)) { "invalid stable UID receipt for $trackId" }
                    require(specId == V2IndexingLedgerIds.STABLE_TRACK_SPAN_IDENTITY_SPEC_ID) {
                        "unsupported stable UID receipt spec for $trackId"
                    }
                    digest.update(1.toByte())
                    digest.updateLengthPrefixed(stableId)
                    digest.updateLengthPrefixed(specId)
                    digest.updateLengthPrefixed(strength.name)
                    unique += stableId
                    covered++
                    when (strength) {
                        StableTrackSpanIdentityStrength.FULL_CONTENT_SHA256 -> full++
                        StableTrackSpanIdentityStrength.VERSIONED_SAMPLED_CONTENT_SHA256 -> sampled++
                    }
                }
                previousId = trackId
                rows++
                if (rows == embeddingCount || rows % 4096 == 0) {
                    onRowProgress?.invoke(rows, embeddingCount)
                }
            }
            require(rows == embeddingCount) {
                "stable UID coverage saw $rows rows, expected $embeddingCount"
            }
        }
        if (hasReceipts) {
            val orphaned = database.rawQuery(
                """
                SELECT COUNT(*) FROM $receiptTable r
                LEFT JOIN ${V2EmbeddingCommitRepository.EMBEDDING_TABLE} e ON e.track_id = r.track_id
                WHERE e.track_id IS NULL
                """.trimIndent(),
                null,
            ).use { cursor ->
                check(cursor.moveToFirst())
                cursor.getInt(0)
            }
            require(orphaned == 0) { "activation database has $orphaned orphaned V2 receipts" }
        }
        return V2StableTrackUidCoverageBinding(
            coveredTrackCount = covered,
            uncoveredTrackCount = uncovered,
            uniqueStableTrackSpanCount = unique.size,
            fullContentIdentityCount = full,
            sampledContentIdentityCount = sampled,
            mappingSha256 = digest.digest().toV2CommitHex(),
        )
    }

    private val STABLE_ID = Regex("^stable-track-span-v1-[0-9a-f]{64}$")
}

object V2EmbeddingSpecCoverage {
    const val COMPATIBILITY_BASE_PROVENANCE_POLICY_ID =
        "unreceipted-clamp3-compatibility-base-v1:unknown-model-and-preprocessing:no-v2-claim"

    fun inspect(
        database: SQLiteDatabase,
        embeddingCount: Int,
        expectedReceiptSpecId: String,
        onRowProgress: ((completedRows: Int, totalRows: Int) -> Unit)? = null,
    ): V2EmbeddingSpecCoverageBinding {
        require(expectedReceiptSpecId.matches(EMBEDDING_SPEC_ID)) {
            "invalid expected receipt embedding spec ID"
        }
        val receiptTable = V2EmbeddingCommitRepository.RECEIPT_TABLE
        val hasReceipts = database.rawQuery(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1",
            arrayOf(receiptTable),
        ).use { it.moveToFirst() }
        val mappingDigest = MessageDigest.getInstance("SHA-256").apply {
            updateLengthPrefixed("v2-embedding-spec-coverage-v1")
            updateInt(embeddingCount)
        }
        val compatibilityDigest = MessageDigest.getInstance("SHA-256").apply {
            updateLengthPrefixed("v2-unreceipted-compatibility-content-v1")
        }
        val receiptCounts = sortedMapOf<String, Int>()
        var compatibilityCount = 0
        var rows = 0
        var previousId = Long.MIN_VALUE
        val query = if (hasReceipts) {
            """
            SELECT e.track_id, e.embedding, r.receipt_schema_version, r.embedding_spec_id
            FROM ${V2EmbeddingCommitRepository.EMBEDDING_TABLE} e
            LEFT JOIN $receiptTable r ON r.track_id = e.track_id
            ORDER BY e.track_id
            """.trimIndent()
        } else {
            """
            SELECT track_id, embedding, NULL, NULL
            FROM ${V2EmbeddingCommitRepository.EMBEDDING_TABLE}
            ORDER BY track_id
            """.trimIndent()
        }
        onRowProgress?.invoke(0, embeddingCount)
        database.rawQuery(query, null).use { cursor ->
            while (cursor.moveToNext()) {
                val trackId = cursor.getLong(0)
                val embedding = cursor.getBlob(1)
                require(trackId > 0L && trackId > previousId) {
                    "embedding coverage rows are duplicated or unordered"
                }
                V2Clamp3VectorCodec.requireValidBlob(embedding)
                mappingDigest.updateLong(trackId)
                if (cursor.isNull(2)) {
                    require(cursor.isNull(3)) { "partial V2 receipt for track $trackId" }
                    mappingDigest.update(0.toByte())
                    compatibilityDigest.updateLong(trackId)
                    compatibilityDigest.updateInt(embedding.size)
                    compatibilityDigest.update(embedding)
                    compatibilityCount++
                } else {
                    val receiptSchema = cursor.getInt(2)
                    require(!cursor.isNull(3)) { "partial V2 receipt for track $trackId" }
                    val receiptSpecId = cursor.getString(3)
                    require(receiptSchema == V2_EMBEDDING_COMMIT_RECEIPT_SCHEMA_VERSION) {
                        "unsupported V2 receipt schema $receiptSchema for track $trackId"
                    }
                    require(receiptSpecId.matches(EMBEDDING_SPEC_ID)) {
                        "invalid V2 receipt embedding spec for track $trackId"
                    }
                    require(receiptSpecId == expectedReceiptSpecId) {
                        "conflicting V2 receipt embedding spec $receiptSpecId for track $trackId"
                    }
                    mappingDigest.update(1.toByte())
                    mappingDigest.updateLengthPrefixed(receiptSpecId)
                    receiptCounts[receiptSpecId] = (receiptCounts[receiptSpecId] ?: 0) + 1
                }
                previousId = trackId
                rows++
                if (rows == embeddingCount || rows % 4096 == 0) {
                    onRowProgress?.invoke(rows, embeddingCount)
                }
            }
        }
        require(rows == embeddingCount) {
            "embedding coverage saw $rows rows, expected $embeddingCount"
        }
        if (hasReceipts) {
            val orphaned = database.rawQuery(
                """
                SELECT COUNT(*) FROM $receiptTable r
                LEFT JOIN ${V2EmbeddingCommitRepository.EMBEDDING_TABLE} e ON e.track_id = r.track_id
                WHERE e.track_id IS NULL
                """.trimIndent(),
                null,
            ).use { cursor ->
                check(cursor.moveToFirst())
                cursor.getInt(0)
            }
            require(orphaned == 0) { "activation database has $orphaned orphaned V2 receipts" }
        }
        val receiptBoundCount = receiptCounts.values.sum()
        require(receiptBoundCount + compatibilityCount == embeddingCount) {
            "embedding coverage does not cover the complete generation"
        }
        mappingDigest.updateInt(receiptCounts.size)
        receiptCounts.forEach { (specId, count) ->
            mappingDigest.updateLengthPrefixed(specId)
            mappingDigest.updateInt(count)
        }
        mappingDigest.updateInt(compatibilityCount)
        val compatibilityBase = if (compatibilityCount == 0) {
            null
        } else {
            compatibilityDigest.updateInt(compatibilityCount)
            V2CompatibilityBaseEmbeddingCoverageBinding(
                provenancePolicyId = COMPATIBILITY_BASE_PROVENANCE_POLICY_ID,
                trackCount = compatibilityCount,
                orderedContentSha256 = compatibilityDigest.digest().toV2CommitHex(),
            ).also { base ->
                mappingDigest.updateLengthPrefixed(base.provenancePolicyId)
                mappingDigest.updateLengthPrefixed(base.orderedContentSha256)
            }
        }
        return V2EmbeddingSpecCoverageBinding(
            totalTrackCount = embeddingCount,
            receiptBoundTrackCount = receiptBoundCount,
            receiptSpecTrackCounts = receiptCounts.toMap(),
            compatibilityBase = compatibilityBase,
            mappingSha256 = mappingDigest.digest().toV2CommitHex(),
        )
    }

    /** Expected imported compatibility base after removing only explicitly authorized row IDs. */
    internal fun inspectCompatibilityBase(
        database: SQLiteDatabase,
        excludedTrackIds: Set<Long>,
    ): V2CompatibilityBaseEmbeddingCoverageBinding? {
        val receiptTable = V2EmbeddingCommitRepository.RECEIPT_TABLE
        val hasReceipts = database.rawQuery(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1",
            arrayOf(receiptTable),
        ).use { it.moveToFirst() }
        val query = if (hasReceipts) {
            """
            SELECT e.track_id, e.embedding, r.track_id
            FROM ${V2EmbeddingCommitRepository.EMBEDDING_TABLE} e
            LEFT JOIN $receiptTable r ON r.track_id = e.track_id
            ORDER BY e.track_id
            """.trimIndent()
        } else {
            """
            SELECT track_id, embedding, NULL
            FROM ${V2EmbeddingCommitRepository.EMBEDDING_TABLE}
            ORDER BY track_id
            """.trimIndent()
        }
        val digest = MessageDigest.getInstance("SHA-256").apply {
            updateLengthPrefixed("v2-unreceipted-compatibility-content-v1")
        }
        var count = 0
        database.rawQuery(query, null).use { cursor ->
            while (cursor.moveToNext()) {
                val trackId = cursor.getLong(0)
                if (!cursor.isNull(2) || trackId in excludedTrackIds) continue
                val embedding = cursor.getBlob(1)
                V2Clamp3VectorCodec.requireValidBlob(embedding)
                digest.updateLong(trackId)
                digest.updateInt(embedding.size)
                digest.update(embedding)
                count++
            }
        }
        if (count == 0) return null
        digest.updateInt(count)
        return V2CompatibilityBaseEmbeddingCoverageBinding(
            provenancePolicyId = COMPATIBILITY_BASE_PROVENANCE_POLICY_ID,
            trackCount = count,
            orderedContentSha256 = digest.digest().toV2CommitHex(),
        )
    }

    private val EMBEDDING_SPEC_ID = Regex("^embedding-spec-v2-[0-9a-f]{64}$")
}

enum class V2GenerationPublicationUnit {
    ROWS,
    BYTES,
}

data class V2GenerationPublicationProgress(
    val detail: String,
    val completedUnits: Long? = null,
    val totalUnits: Long? = null,
    val unit: V2GenerationPublicationUnit? = null,
)

class V2IndexGenerationPublisher(
    private val filesDir: File,
    private val gson: Gson = GsonBuilder().disableHtmlEscaping().create(),
    private val beforePointerPublication: (V2IndexGenerationManifest) -> Unit = {},
    private val afterInstallBeforePointerPublication: (V2IndexGenerationManifest) -> Unit = {},
) {
    private val root = File(filesDir, GENERATIONS_DIRECTORY)

    /**
     * Activates only a job-private staging DB. The caller must close all writers before entry.
     * Pointer publication is the sole commit point; the prior generation remains readable until it.
     */
    fun publish(
        ledger: IndexingJobLedger,
        jobPrivateStagingDatabase: File,
        explicitGraphFile: File? = null,
        importedRowAuthorization: V2ImportedRowSupersessionAuthorization? = null,
        onProgress: (V2GenerationPublicationProgress) -> Unit = {},
    ): V2ResolvedActiveIndexGeneration {
        require(ledger.state == IndexingJobState.ACTIVATING) { "job is not ACTIVATING" }
        V2IndexingLedgerValidator.requireValid(ledger)
        val hasLogicalCue = ledger.jobSpec.tracks.any {
            it.finalizedAudioSpan.kind == V2ResolvedAudioSpanKind.LOGICAL_CUE
        }
        require(hasLogicalCue == (importedRowAuthorization != null)) {
            "logical CUE publication requires its exact imported-row authorization"
        }
        importedRowAuthorization?.let {
            V2ImportedRowSupersessionAuthorizationPolicy.requireValid(it, ledger)
        }
        require(jobPrivateStagingDatabase.isFile) { "job-private staging DB is missing" }
        require(root.isDirectory || root.mkdirs()) { "cannot create generation root $root" }
        require(jobPrivateStagingDatabase.canonicalFile.toPath().startsWith(root.canonicalFile.toPath()).not()) {
            "job-private DB must not already be inside the generation store"
        }
        val expectedGraphPolicy = if (ledger.jobSpec.rebuildDerivedIndexes) {
            V2IndexGenerationGraphPolicy.EXPLICIT_REBUILD
        } else {
            V2IndexGenerationGraphPolicy.ABSENT
        }
        require((explicitGraphFile != null) == ledger.jobSpec.rebuildDerivedIndexes) {
            "graph presence must exactly match rebuildDerivedIndexes"
        }
        onProgress(V2GenerationPublicationProgress("Reading the current active-generation pointer"))
        val expectedActive = V2GenerationPublicationCoordinator.capture(filesDir, gson)
        val activeBeforePublication = expectedActive.pointer?.let {
            V2IndexGenerationReader.requireActive(filesDir, gson) { progress ->
                onProgress(
                    V2GenerationPublicationProgress(
                        detail = "Hashing active music-index file ${progress.filename}",
                        completedUnits = progress.completedBytes,
                        totalUnits = progress.totalBytes,
                        unit = V2GenerationPublicationUnit.BYTES,
                    ),
                )
            }
        }
        if (activeBeforePublication?.manifest?.origin == V2IndexGenerationOrigin.INDEXING_JOB &&
            activeBeforePublication.manifest.jobId == ledger.jobSpec.jobId &&
            activeBeforePublication.manifest.jobSpecId == ledger.jobSpec.specId
        ) {
            if (importedRowAuthorization != null) {
                try {
                    SQLiteDatabase.openDatabase(
                        activeBeforePublication.databaseFile.path,
                        null,
                        SQLiteDatabase.OPEN_READONLY,
                    ).use { database ->
                        requireCoverageExtendsBase(
                            ledger = ledger,
                            current = activeBeforePublication.manifest.embeddingCoverage,
                            database = database,
                            importedRowAuthorization = importedRowAuthorization,
                        )
                    }
                } catch (error: Exception) {
                    throw V2ImportedRowAuthorizationException(
                        "Published CUE repair evidence changed; start a new indexing preflight",
                        error,
                    )
                }
            }
            V2GenerationPublicationCoordinator.commit(
                filesDir = filesDir,
                expected = expectedActive,
                generationId = activeBeforePublication.manifest.generationId,
                manifestSha256 = activeBeforePublication.manifestSha256,
                gson = gson,
            )
            return activeBeforePublication
        }
        require(expectedActive.pointer?.generationId == ledger.jobSpec.baseGenerationId) {
            "planned base generation is no longer active"
        }

        val staging = File(root, "$GENERATION_STAGING_PREFIX${UUID.randomUUID()}")
        require(staging.mkdir()) { "cannot create staging generation $staging" }
        try {
            val generationDatabase = File(staging, DATABASE_FILE)
            onProgress(
                V2GenerationPublicationProgress(
                    "SQLite is compacting the completed staging index into the new generation",
                ),
            )
            snapshotDatabase(jobPrivateStagingDatabase, generationDatabase)
            val embeddingFile = File(staging, EMBEDDING_FILE)
            val database = SQLiteDatabase.openDatabase(
                generationDatabase.path,
                null,
                SQLiteDatabase.OPEN_READWRITE,
            )
            try {
                database.disableWriteAheadLogging()
                var embeddingBindingResult: V2OrderedEmbeddingBinding? = null
                var stableCoverageResult: V2StableTrackUidCoverageBinding? = null
                var embeddingCoverageResult: V2EmbeddingSpecCoverageBinding? = null
                database.beginTransaction()
                try {
                    val source = V2SqliteOrderedEmbeddingSource(database)
                    embeddingBindingResult = V2EmbeddingGenerationFile.write(
                        source = source,
                        target = embeddingFile,
                        onRowProgress = { completedRows, totalRows ->
                            onProgress(
                                V2GenerationPublicationProgress(
                                    detail = if (completedRows == 0) {
                                        "Preparing $totalRows ordered embedding rows"
                                    } else {
                                        "Written $completedRows of $totalRows ordered embedding rows"
                                    },
                                    completedUnits = completedRows.toLong(),
                                    totalUnits = totalRows.toLong(),
                                    unit = V2GenerationPublicationUnit.ROWS,
                                ),
                            )
                        },
                        onHashProgress = { completedBytes, totalBytes ->
                            onProgress(
                                V2GenerationPublicationProgress(
                                    detail = "Hashing the ordered embedding file",
                                    completedUnits = completedBytes,
                                    totalUnits = totalBytes,
                                    unit = V2GenerationPublicationUnit.BYTES,
                                ),
                            )
                        },
                    )
                    onProgress(
                        V2GenerationPublicationProgress(
                            "Measuring stable source-identity coverage in the new index",
                        ),
                    )
                    stableCoverageResult = V2StableTrackUidCoverage.inspect(
                        database,
                        checkNotNull(embeddingBindingResult).trackCount,
                    ) { completedRows, totalRows ->
                        onProgress(
                            V2GenerationPublicationProgress(
                                detail = if (completedRows == 0) {
                                    "Preparing to read $totalRows stable source-identity rows"
                                } else {
                                    "Read $completedRows of $totalRows stable source-identity rows"
                                },
                                completedUnits = completedRows.toLong(),
                                totalUnits = totalRows.toLong(),
                                unit = V2GenerationPublicationUnit.ROWS,
                            ),
                        )
                    }
                    onProgress(
                        V2GenerationPublicationProgress(
                            "Measuring embedding-model receipt coverage in the new index",
                        ),
                    )
                    embeddingCoverageResult = V2EmbeddingSpecCoverage.inspect(
                        database = database,
                        embeddingCount = checkNotNull(embeddingBindingResult).trackCount,
                        expectedReceiptSpecId = ledger.jobSpec.embeddingSpec.specId,
                    ) { completedRows, totalRows ->
                        onProgress(
                            V2GenerationPublicationProgress(
                                detail = if (completedRows == 0) {
                                    "Preparing to read $totalRows embedding-model receipt rows"
                                } else {
                                    "Read $completedRows of $totalRows embedding-model receipt rows"
                                },
                                completedUnits = completedRows.toLong(),
                                totalUnits = totalRows.toLong(),
                                unit = V2GenerationPublicationUnit.ROWS,
                            ),
                        )
                    }
                    database.setTransactionSuccessful()
                } finally {
                    database.endTransaction()
                }
                val embeddingBinding = checkNotNull(embeddingBindingResult)
                val stableCoverage = checkNotNull(stableCoverageResult)
                val embeddingCoverage = checkNotNull(embeddingCoverageResult)
                require(
                    embeddingCoverage.receiptSpecTrackCounts[
                        ledger.jobSpec.embeddingSpec.specId
                    ]?.let { it > 0 } == true,
                ) { "activated job produced no receipt-bound V2 embeddings" }
                onProgress(
                    V2GenerationPublicationProgress(
                        "Comparing inherited tracks and embeddings with the active music index",
                    ),
                )
                try {
                    requireCoverageExtendsBase(
                        ledger,
                        embeddingCoverage,
                        database,
                        importedRowAuthorization,
                    )
                } catch (error: Exception) {
                    if (importedRowAuthorization != null) {
                        throw V2ImportedRowAuthorizationException(
                            "Imported CUE activation proof changed; start a new indexing preflight",
                            error,
                        )
                    }
                    throw error
                }

                val graphBinding = explicitGraphFile?.let { source ->
                    val target = File(staging, V2GraphGenerationFile.GRAPH_FILE)
                    copyAndSync(source, target) { completedBytes, totalBytes ->
                        onProgress(
                            V2GenerationPublicationProgress(
                                detail = "Copying the verified similarity graph",
                                completedUnits = completedBytes,
                                totalUnits = totalBytes,
                                unit = V2GenerationPublicationUnit.BYTES,
                            ),
                        )
                    }
                    V2GraphGenerationFile.inspect(
                        file = target,
                        onRowProgress = { completedRows, totalRows ->
                            onProgress(
                                V2GenerationPublicationProgress(
                                    detail = if (completedRows == 0) {
                                        "Preparing to inspect $totalRows similarity-graph rows"
                                    } else {
                                        "Inspected $completedRows of $totalRows similarity-graph rows"
                                    },
                                    completedUnits = completedRows.toLong(),
                                    totalUnits = totalRows.toLong(),
                                    unit = V2GenerationPublicationUnit.ROWS,
                                ),
                            )
                        },
                        onHashProgress = { completedBytes, totalBytes ->
                            onProgress(
                                V2GenerationPublicationProgress(
                                    detail = "Hashing the copied similarity graph",
                                    completedUnits = completedBytes,
                                    totalUnits = totalBytes,
                                    unit = V2GenerationPublicationUnit.BYTES,
                                ),
                            )
                        },
                    ).also { graph ->
                        require(graph.nodeCount == embeddingBinding.trackCount &&
                            graph.orderedTrackSetSha256 == embeddingBinding.orderedTrackSetSha256
                        ) { "graph is not bound to the exact PEMB ordered track set" }
                    }
                }
                val activationBindingId = V2IndexGenerationIdentity.activationBindingId(
                    ledger = ledger,
                    embeddings = embeddingBinding,
                    stableCoverage = stableCoverage,
                    embeddingCoverage = embeddingCoverage,
                    graphPolicy = expectedGraphPolicy,
                    graph = graphBinding,
                )
                onProgress(
                    V2GenerationPublicationProgress(
                        "Installing the generation receipt and invalidation triggers",
                    ),
                )
                installInvalidationReceipt(
                    database = database,
                    jobSpecId = ledger.jobSpec.specId,
                    receiptEmbeddingSpecId = ledger.jobSpec.embeddingSpec.specId,
                    textRetrievalSpecId = ledger.jobSpec.textRetrievalSpec.specId,
                    createdAtEpochMs = ledger.jobSpec.createdAtEpochMs,
                    activationBindingId = activationBindingId,
                    embeddings = embeddingBinding,
                    stableCoverage = stableCoverage,
                    embeddingCoverage = embeddingCoverage,
                    graph = graphBinding,
                )
                onProgress(V2GenerationPublicationProgress("Running SQLite integrity_check on the new generation"))
                database.rawQuery("PRAGMA integrity_check(1)", null).use { cursor ->
                    require(cursor.moveToFirst() && cursor.getString(0) == "ok") {
                        "generation database integrity check failed"
                    }
                }
                database.close()
                syncFile(generationDatabase)

                onProgress(V2GenerationPublicationProgress("Syncing and hashing the new generation database"))
                val generationDatabaseSha256 = V2FileSha256.digest(
                    generationDatabase,
                ) { completedBytes, totalBytes ->
                    onProgress(
                        V2GenerationPublicationProgress(
                            detail = "Hashing the new generation database",
                            completedUnits = completedBytes,
                            totalUnits = totalBytes,
                            unit = V2GenerationPublicationUnit.BYTES,
                        ),
                    )
                }
                val provisional = V2IndexGenerationManifest(
                    schemaVersion = MANIFEST_SCHEMA_VERSION,
                    origin = V2IndexGenerationOrigin.INDEXING_JOB,
                    generationId = "",
                    activationBindingId = activationBindingId,
                    jobId = ledger.jobSpec.jobId,
                    jobSpecId = ledger.jobSpec.specId,
                    receiptEmbeddingSpec = ledger.jobSpec.embeddingSpec,
                    textRetrievalSpec = ledger.jobSpec.textRetrievalSpec,
                    baseGenerationId = ledger.jobSpec.baseGenerationId,
                    rebuildDerivedIndexes = ledger.jobSpec.rebuildDerivedIndexes,
                    graphPolicy = expectedGraphPolicy,
                    createdAtEpochMs = ledger.jobSpec.createdAtEpochMs,
                    databaseRelativePath = DATABASE_FILE,
                    databaseByteLength = generationDatabase.length(),
                    databaseSha256 = generationDatabaseSha256,
                    databaseContentSha256 = embeddingBinding.databaseContentSha256,
                    orderedTrackSetSha256 = embeddingBinding.orderedTrackSetSha256,
                    stableTrackUidCoverage = stableCoverage,
                    embeddingCoverage = embeddingCoverage,
                    trackCount = embeddingBinding.trackCount,
                    embeddingDimension = embeddingBinding.dimension,
                    embeddingRelativePath = EMBEDDING_FILE,
                    embeddingByteLength = embeddingBinding.byteLength,
                    embeddingSha256 = embeddingBinding.fileSha256,
                    graph = graphBinding,
                )
                val manifest = provisional.copy(
                    generationId = V2IndexGenerationIdentity.generationId(provisional),
                )
                val manifestFile = File(staging, MANIFEST_FILE)
                onProgress(V2GenerationPublicationProgress("Writing and syncing the immutable generation manifest"))
                writeManifest(manifestFile, manifest)
                syncDirectory(staging)
                val manifestSha256 = V2FileSha256.digest(manifestFile)
                onProgress(V2GenerationPublicationProgress("Atomically selecting the completed music-index generation"))
                beforePointerPublication(manifest)
                return V2GenerationPublicationCoordinator.installAndCommit(
                    filesDir = filesDir,
                    expected = expectedActive,
                    stagingDirectory = staging,
                    manifest = manifest,
                    manifestSha256 = manifestSha256,
                    gson = gson,
                    afterInstallBeforePointerPublication =
                        afterInstallBeforePointerPublication,
                    onArtifactHashProgress = { progress ->
                        onProgress(
                            V2GenerationPublicationProgress(
                                detail = "Revalidating completed generation file " +
                                    progress.filename,
                                completedUnits = progress.completedBytes,
                                totalUnits = progress.totalBytes,
                                unit = V2GenerationPublicationUnit.BYTES,
                            ),
                        )
                    },
                )
            } finally {
                if (database.isOpen) database.close()
            }
        } finally {
            staging.deleteRecursively()
        }
    }

    /**
     * Publishes a validated legacy database without attributing any imported vector to V2.
     * The supplied specs are policy for future receipt-bound additions and text retrieval only.
     */
    fun publishBootstrapCompatibility(
        privateStagingDatabase: File,
        futureReceiptEmbeddingSpec: EmbeddingSpecFingerprint,
        textRetrievalSpec: TextRetrievalSpecFingerprint,
        exactValidatedGraphFile: File? = null,
        expectedActive: V2GenerationPublicationExpectation =
            V2GenerationPublicationCoordinator.capture(filesDir, gson),
    ): V2ResolvedActiveIndexGeneration = publishDerivedGeneration(
        privateStagingDatabase = privateStagingDatabase,
        origin = V2IndexGenerationOrigin.BOOTSTRAP_COMPATIBILITY,
        jobId = V2IndexGenerationManifestPolicy.BOOTSTRAP_JOB_ID,
        jobSpecId = V2IndexGenerationIdentity.bootstrapSpecId(
            futureReceiptEmbeddingSpec,
            textRetrievalSpec,
        ),
        futureReceiptEmbeddingSpec = futureReceiptEmbeddingSpec,
        textRetrievalSpec = textRetrievalSpec,
        baseGeneration = null,
        expectedActive = expectedActive,
        derivedGraphFile = exactValidatedGraphFile,
        validateCoverage = { embedding, stable, coverage ->
            require(coverage.receiptBoundTrackCount == 0 &&
                coverage.receiptSpecTrackCounts.isEmpty() &&
                coverage.compatibilityBase?.trackCount == embedding.trackCount
            ) { "bootstrap source contains V2 receipt claims" }
            require(stable.coveredTrackCount == 0 &&
                stable.uncoveredTrackCount == embedding.trackCount
            ) { "bootstrap source contains V2 stable-identity claims" }
        },
    )

    fun publishLibraryMaintenance(
        privateStagingDatabase: File,
        baseGeneration: V2ResolvedActiveIndexGeneration,
        exactRepairedGraphFile: File? = null,
        expectedActive: V2GenerationPublicationExpectation =
            V2GenerationPublicationCoordinator.capture(filesDir, gson),
    ): V2ResolvedActiveIndexGeneration = publishDerivedGeneration(
        privateStagingDatabase = privateStagingDatabase,
        origin = V2IndexGenerationOrigin.LIBRARY_MAINTENANCE,
        jobId = V2IndexGenerationManifestPolicy.MAINTENANCE_JOB_ID,
        jobSpecId = V2IndexGenerationIdentity.maintenanceSpecId(
            baseGeneration.manifest.generationId,
            baseGeneration.manifest.receiptEmbeddingSpec,
            baseGeneration.manifest.textRetrievalSpec,
        ),
        futureReceiptEmbeddingSpec = baseGeneration.manifest.receiptEmbeddingSpec,
        textRetrievalSpec = baseGeneration.manifest.textRetrievalSpec,
        baseGeneration = baseGeneration,
        expectedActive = expectedActive,
        derivedGraphFile = exactRepairedGraphFile,
        validateCoverage = { embedding, stable, coverage ->
            require(coverage.totalTrackCount == embedding.trackCount &&
                stable.coveredTrackCount == coverage.receiptBoundTrackCount
            ) { "maintenance source has inconsistent receipt or stable-identity coverage" }
        },
    )

    /**
     * Publishes a base-bound append of externally computed CLaMP3 vectors. Server rows remain
     * unreceipted compatibility embeddings: their host inference cannot claim Android V2 receipt
     * provenance even when the model artifacts and output space are byte-hash pinned.
     */
    internal fun publishServerMerge(
        privateStagingDatabase: File,
        baseGeneration: V2ResolvedActiveIndexGeneration,
        bundleDatabaseSha256: String,
        addedTrackCount: Int,
        preparedGraphUpdate: V2PreparedGraphUpdate,
        expectedActive: V2GenerationPublicationExpectation =
            V2GenerationPublicationCoordinator.capture(filesDir, gson),
        onProgress: (V2GenerationPublicationProgress) -> Unit = {},
    ): V2ResolvedActiveIndexGeneration {
        require(bundleDatabaseSha256.matches(SHA256)) {
            "server bundle database hash is invalid"
        }
        require(addedTrackCount > 0) { "server merge must append at least one track" }
        return publishDerivedGeneration(
            privateStagingDatabase = privateStagingDatabase,
            origin = V2IndexGenerationOrigin.SERVER_MERGE,
            jobId = V2IndexGenerationManifestPolicy.SERVER_MERGE_JOB_ID,
            jobSpecId = V2IndexGenerationIdentity.serverMergeSpecId(
                baseGenerationId = baseGeneration.manifest.generationId,
                bundleDatabaseSha256 = bundleDatabaseSha256,
                receiptEmbeddingSpec = baseGeneration.manifest.receiptEmbeddingSpec,
                textRetrievalSpec = baseGeneration.manifest.textRetrievalSpec,
            ),
            futureReceiptEmbeddingSpec = baseGeneration.manifest.receiptEmbeddingSpec,
            textRetrievalSpec = baseGeneration.manifest.textRetrievalSpec,
            baseGeneration = baseGeneration,
            expectedActive = expectedActive,
            derivedGraphFile = preparedGraphUpdate.graphFile,
            preparedGraphUpdate = preparedGraphUpdate,
            consumePreparedAssets = true,
            serverMergeAddedTrackCount = addedTrackCount,
            onProgress = onProgress,
            validateCoverage = { embedding, stable, coverage ->
                val inheritedStable = baseGeneration.manifest.stableTrackUidCoverage
                val inheritedCoverage = baseGeneration.manifest.embeddingCoverage
                require(embedding.trackCount == baseGeneration.manifest.trackCount + addedTrackCount &&
                    coverage.totalTrackCount == embedding.trackCount &&
                    coverage.receiptBoundTrackCount == inheritedCoverage.receiptBoundTrackCount &&
                    coverage.receiptSpecTrackCounts == inheritedCoverage.receiptSpecTrackCounts &&
                    stable.coveredTrackCount == inheritedStable.coveredTrackCount &&
                    stable.uncoveredTrackCount ==
                    inheritedStable.uncoveredTrackCount + addedTrackCount &&
                    stable.uniqueStableTrackSpanCount ==
                    inheritedStable.uniqueStableTrackSpanCount &&
                    stable.fullContentIdentityCount == inheritedStable.fullContentIdentityCount &&
                    stable.sampledContentIdentityCount ==
                    inheritedStable.sampledContentIdentityCount
                ) { "server merge changed inherited receipt or stable-identity coverage" }
            },
        )
    }

    private fun publishDerivedGeneration(
        privateStagingDatabase: File,
        origin: V2IndexGenerationOrigin,
        jobId: String,
        jobSpecId: String,
        futureReceiptEmbeddingSpec: EmbeddingSpecFingerprint,
        textRetrievalSpec: TextRetrievalSpecFingerprint,
        baseGeneration: V2ResolvedActiveIndexGeneration?,
        expectedActive: V2GenerationPublicationExpectation,
        derivedGraphFile: File?,
        preparedGraphUpdate: V2PreparedGraphUpdate? = null,
        consumePreparedAssets: Boolean = false,
        rebuildDerivedIndexes: Boolean = false,
        serverMergeAddedTrackCount: Int? = null,
        onProgress: (V2GenerationPublicationProgress) -> Unit = {},
        validateCoverage: (
            V2OrderedEmbeddingBinding,
            V2StableTrackUidCoverageBinding,
            V2EmbeddingSpecCoverageBinding,
        ) -> Unit,
    ): V2ResolvedActiveIndexGeneration {
        require(origin != V2IndexGenerationOrigin.INDEXING_JOB) {
            "derived publisher cannot impersonate an indexing job"
        }
        require(derivedGraphFile == null || origin in setOf(
            V2IndexGenerationOrigin.BOOTSTRAP_COMPATIBILITY,
            V2IndexGenerationOrigin.LIBRARY_MAINTENANCE,
            V2IndexGenerationOrigin.SERVER_MERGE,
        )) { "unsupported derived graph provenance" }
        require(!rebuildDerivedIndexes) {
            "base-bound graph updates must not claim a full derived-index rebuild"
        }
        require(origin != V2IndexGenerationOrigin.SERVER_MERGE || derivedGraphFile != null) {
            "server merge must preserve Random Walk with an exact updated graph"
        }
        require((origin == V2IndexGenerationOrigin.SERVER_MERGE) ==
            (preparedGraphUpdate != null)
        ) { "only a server merge may consume prepared graph-update artifacts" }
        require((origin == V2IndexGenerationOrigin.SERVER_MERGE) ==
            (serverMergeAddedTrackCount != null)
        ) { "server merge publication is missing its exact append count" }
        require(privateStagingDatabase.isFile) { "derived-generation staging DB is missing" }
        V2IndexingLedgerValidator.requireValidEmbeddingSpec(futureReceiptEmbeddingSpec)
        V2IndexingLedgerValidator.requireValidTextRetrievalSpec(textRetrievalSpec)
        require(
            textRetrievalSpec.compatibleAudioEmbeddingSpecId == futureReceiptEmbeddingSpec.specId &&
                textRetrievalSpec.outputDimension == futureReceiptEmbeddingSpec.outputDimension,
        ) { "derived-generation text policy is incompatible with the future audio policy" }
        baseGeneration?.let { base ->
            require(origin in setOf(
                V2IndexGenerationOrigin.LIBRARY_MAINTENANCE,
                V2IndexGenerationOrigin.SERVER_MERGE,
            )) {
                "only base-bound derived generations may bind an active base"
            }
            requireCurrentBase(base)
        }
        preparedGraphUpdate?.let { prepared ->
            requirePreparedGraphUpdate(privateStagingDatabase, prepared)
        }
        if (baseGeneration != null) {
            require(expectedActive.pointer?.generationId == baseGeneration.manifest.generationId &&
                expectedActive.pointer.manifestSha256 == baseGeneration.manifestSha256
            ) { "publication expectation does not match its maintenance base" }
        }
        val baseHasExactGraphProof = baseGeneration
            ?.let(V2ExactGraphIncrementalBase::fromActiveGeneration) != null
        val installExactGraphProof = V2DerivedGraphExactProofPolicy.shouldInstall(
            origin = origin,
            graphPresent = derivedGraphFile != null,
            baseHasExactProof = baseHasExactGraphProof,
        )
        require(root.isDirectory || root.mkdirs()) { "cannot create generation root $root" }
        require(
            privateStagingDatabase.canonicalFile.toPath()
                .startsWith(root.canonicalFile.toPath()).not(),
        ) { "derived-generation DB must not already be inside the generation store" }

        val staging = File(root, "$GENERATION_STAGING_PREFIX${UUID.randomUUID()}")
        require(staging.mkdir()) { "cannot create staging generation $staging" }
        try {
            val generationDatabase = File(staging, DATABASE_FILE)
            if (consumePreparedAssets) {
                require(!File(privateStagingDatabase.path + "-wal").exists() &&
                    !File(privateStagingDatabase.path + "-shm").exists()
                ) { "prepared server-merge database has SQLite sidecar files" }
                onProgress(V2GenerationPublicationProgress(
                    "Moving the prepared database into unpublished generation staging",
                ))
                movePreparedFile(privateStagingDatabase, generationDatabase)
            } else {
                onProgress(V2GenerationPublicationProgress(
                    "Compacting the prepared database into unpublished generation staging",
                ))
                snapshotDatabase(privateStagingDatabase, generationDatabase)
            }
            val embeddingFile = File(staging, EMBEDDING_FILE)
            val database = SQLiteDatabase.openDatabase(
                generationDatabase.path,
                null,
                SQLiteDatabase.OPEN_READWRITE,
            )
            try {
                database.disableWriteAheadLogging()
                val embeddingBinding = preparedGraphUpdate?.let { prepared ->
                    onProgress(V2GenerationPublicationProgress(
                        "Moving the prepared packed embeddings into generation staging",
                    ))
                    movePreparedFile(prepared.embeddingFile, embeddingFile)
                    require(embeddingFile.length() == prepared.embeddingBinding.byteLength) {
                        "prepared packed embeddings changed before publication"
                    }
                    prepared.embeddingBinding
                } ?: V2EmbeddingGenerationFile.write(
                        source = V2SqliteOrderedEmbeddingSource(database),
                        target = embeddingFile,
                        onRowProgress = { completedRows, totalRows ->
                            onProgress(V2GenerationPublicationProgress(
                                detail = "Packed $completedRows of $totalRows embeddings for retrieval",
                                completedUnits = completedRows.toLong(),
                                totalUnits = totalRows.toLong(),
                                unit = V2GenerationPublicationUnit.ROWS,
                            ))
                        },
                        onHashProgress = { completedBytes, totalBytes ->
                            onProgress(V2GenerationPublicationProgress(
                                detail = "Hashed $completedBytes of $totalBytes packed-embedding bytes",
                                completedUnits = completedBytes,
                                totalUnits = totalBytes,
                                unit = V2GenerationPublicationUnit.BYTES,
                            ))
                        },
                    )
                onProgress(V2GenerationPublicationProgress(
                    "Checking embedding provenance and stable-track coverage",
                ))
                val stableCoverage = V2StableTrackUidCoverage.inspect(
                    database,
                    embeddingBinding.trackCount,
                )
                val embeddingCoverage = V2EmbeddingSpecCoverage.inspect(
                    database = database,
                    embeddingCount = embeddingBinding.trackCount,
                    expectedReceiptSpecId = futureReceiptEmbeddingSpec.specId,
                )
                validateCoverage(embeddingBinding, stableCoverage, embeddingCoverage)
                if (baseGeneration != null) {
                    onProgress(V2GenerationPublicationProgress(
                        "Checking the prepared index against its active base",
                    ))
                    when (origin) {
                        V2IndexGenerationOrigin.LIBRARY_MAINTENANCE ->
                            requireMaintenanceSubset(
                                baseGeneration,
                                database,
                                embeddingBinding.trackCount,
                            )
                        V2IndexGenerationOrigin.SERVER_MERGE ->
                            requireServerMergeExtension(
                                base = baseGeneration,
                                database = database,
                                mergedTrackCount = embeddingBinding.trackCount,
                                addedTrackCount = checkNotNull(serverMergeAddedTrackCount),
                            )
                        else -> error("unsupported base-bound derived generation")
                    }
                }
                var installedGraphFile: File? = null
                val graphBinding = derivedGraphFile?.let { source ->
                    val target = File(staging, V2GraphGenerationFile.GRAPH_FILE)
                    if (preparedGraphUpdate != null) {
                        onProgress(V2GenerationPublicationProgress(
                            "Moving the prepared similarity graph into generation staging",
                        ))
                        movePreparedFile(source, target)
                        require(target.length() == preparedGraphUpdate.graphBinding.byteLength) {
                            "prepared similarity graph changed before publication"
                        }
                    } else {
                        copyAndSync(source, target) { completedBytes, totalBytes ->
                            onProgress(V2GenerationPublicationProgress(
                                detail = "Copied $completedBytes of $totalBytes graph bytes",
                                completedUnits = completedBytes,
                                totalUnits = totalBytes,
                                unit = V2GenerationPublicationUnit.BYTES,
                            ))
                        }
                    }
                    installedGraphFile = target
                    preparedGraphUpdate?.graphBinding ?: V2GraphGenerationFile.inspect(
                        file = target,
                        onRowProgress = { completedRows, totalRows ->
                            onProgress(V2GenerationPublicationProgress(
                                detail = "Checked $completedRows of $totalRows similarity-graph rows",
                                completedUnits = completedRows.toLong(),
                                totalUnits = totalRows.toLong(),
                                unit = V2GenerationPublicationUnit.ROWS,
                            ))
                        },
                        onHashProgress = { completedBytes, totalBytes ->
                            onProgress(V2GenerationPublicationProgress(
                                detail = "Hashed $completedBytes of $totalBytes similarity-graph bytes",
                                completedUnits = completedBytes,
                                totalUnits = totalBytes,
                                unit = V2GenerationPublicationUnit.BYTES,
                            ))
                        },
                    ).also { graph ->
                        require(graph.nodeCount == embeddingBinding.trackCount &&
                            graph.orderedTrackSetSha256 == embeddingBinding.orderedTrackSetSha256
                        ) { "imported graph is not bound to the exact PEMB track ordering" }
                    }
                }
                replaceDerivedGraphExactProof(
                    database = database,
                    graphFile = installedGraphFile,
                    embeddingFile = embeddingFile,
                    graphBinding = graphBinding,
                    embeddingBinding = embeddingBinding,
                    install = installExactGraphProof,
                )
                val graphPolicy = when {
                    graphBinding == null -> V2IndexGenerationGraphPolicy.ABSENT
                    origin == V2IndexGenerationOrigin.BOOTSTRAP_COMPATIBILITY ->
                        V2IndexGenerationGraphPolicy.VALIDATED_COMPATIBILITY_IMPORT
                    origin == V2IndexGenerationOrigin.LIBRARY_MAINTENANCE ->
                        V2IndexGenerationGraphPolicy.BASE_BOUND_DELETION_REPAIR
                    origin == V2IndexGenerationOrigin.SERVER_MERGE ->
                        V2IndexGenerationGraphPolicy.BASE_BOUND_ADDITION_UPDATE
                    else -> error("unsupported derived graph provenance")
                }
                val activationBindingId = V2IndexGenerationIdentity.activationBindingId(
                    origin = origin,
                    jobSpecId = jobSpecId,
                    receiptEmbeddingSpec = futureReceiptEmbeddingSpec,
                    textRetrievalSpec = textRetrievalSpec,
                    baseGenerationId = baseGeneration?.manifest?.generationId,
                    rebuildDerivedIndexes = rebuildDerivedIndexes,
                    embeddings = embeddingBinding,
                    stableCoverage = stableCoverage,
                    embeddingCoverage = embeddingCoverage,
                    graphPolicy = graphPolicy,
                    graph = graphBinding,
                )
                onProgress(V2GenerationPublicationProgress(
                    "Recording the index bindings used by the recommendation engine",
                ))
                installInvalidationReceipt(
                    database = database,
                    jobSpecId = jobSpecId,
                    receiptEmbeddingSpecId = futureReceiptEmbeddingSpec.specId,
                    textRetrievalSpecId = textRetrievalSpec.specId,
                    createdAtEpochMs = 0L,
                    activationBindingId = activationBindingId,
                    embeddings = embeddingBinding,
                    stableCoverage = stableCoverage,
                    embeddingCoverage = embeddingCoverage,
                    graph = graphBinding,
                )
                onProgress(V2GenerationPublicationProgress(
                    "Checking the completed SQLite index for corruption",
                ))
                database.rawQuery("PRAGMA integrity_check(1)", null).use { cursor ->
                    require(cursor.moveToFirst() && cursor.getString(0) == "ok") {
                        "derived generation database integrity check failed"
                    }
                }
                database.close()
                syncFile(generationDatabase)

                onProgress(V2GenerationPublicationProgress(
                    "Hashing the completed database for the immutable index manifest",
                ))
                val generationDatabaseSha256 = V2FileSha256.digest(
                    generationDatabase,
                ) { completedBytes, totalBytes ->
                    onProgress(V2GenerationPublicationProgress(
                        detail = "Hashed $completedBytes of $totalBytes database bytes",
                        completedUnits = completedBytes,
                        totalUnits = totalBytes,
                        unit = V2GenerationPublicationUnit.BYTES,
                    ))
                }
                val provisional = V2IndexGenerationManifest(
                    schemaVersion = MANIFEST_SCHEMA_VERSION,
                    origin = origin,
                    generationId = "",
                    activationBindingId = activationBindingId,
                    jobId = jobId,
                    jobSpecId = jobSpecId,
                    receiptEmbeddingSpec = futureReceiptEmbeddingSpec,
                    textRetrievalSpec = textRetrievalSpec,
                    baseGenerationId = baseGeneration?.manifest?.generationId,
                    rebuildDerivedIndexes = rebuildDerivedIndexes,
                    graphPolicy = graphPolicy,
                    createdAtEpochMs = 0L,
                    databaseRelativePath = DATABASE_FILE,
                    databaseByteLength = generationDatabase.length(),
                    databaseSha256 = generationDatabaseSha256,
                    databaseContentSha256 = embeddingBinding.databaseContentSha256,
                    orderedTrackSetSha256 = embeddingBinding.orderedTrackSetSha256,
                    stableTrackUidCoverage = stableCoverage,
                    embeddingCoverage = embeddingCoverage,
                    trackCount = embeddingBinding.trackCount,
                    embeddingDimension = embeddingBinding.dimension,
                    embeddingRelativePath = EMBEDDING_FILE,
                    embeddingByteLength = embeddingBinding.byteLength,
                    embeddingSha256 = embeddingBinding.fileSha256,
                    graph = graphBinding,
                )
                V2IndexGenerationManifestPolicy.requireValidProvenance(provisional)
                V2IndexGenerationManifestPolicy.requireValidCoverage(provisional)
                val manifest = provisional.copy(
                    generationId = V2IndexGenerationIdentity.generationId(provisional),
                )
                val manifestFile = File(staging, MANIFEST_FILE)
                onProgress(V2GenerationPublicationProgress(
                    "Writing the immutable index manifest",
                ))
                writeManifest(manifestFile, manifest)
                syncDirectory(staging)
                val manifestSha256 = V2FileSha256.digest(manifestFile)
                baseGeneration?.let(::requireCurrentBase)
                beforePointerPublication(manifest)
                onProgress(V2GenerationPublicationProgress(
                    "Activating the completed music index",
                ))
                return V2GenerationPublicationCoordinator.installAndCommit(
                    filesDir = filesDir,
                    expected = expectedActive,
                    stagingDirectory = staging,
                    manifest = manifest,
                    manifestSha256 = manifestSha256,
                    gson = gson,
                    afterInstallBeforePointerPublication =
                        afterInstallBeforePointerPublication,
                )
            } finally {
                if (database.isOpen) database.close()
            }
        } finally {
            staging.deleteRecursively()
        }
    }

    private fun snapshotDatabase(source: File, target: File) {
        require(!target.exists()) { "snapshot destination already exists" }
        SQLiteDatabase.openDatabase(source.path, null, SQLiteDatabase.OPEN_READWRITE).use { database ->
            require(!database.inTransaction()) { "staging DB has an active transaction" }
            val escaped = target.canonicalPath.replace("'", "''")
            database.execSQL("VACUUM INTO '$escaped'")
        }
        require(target.isFile && target.length() > 0L) { "VACUUM INTO did not publish a DB" }
        syncFile(target)
    }

    private fun requirePreparedGraphUpdate(
        privateStagingDatabase: File,
        prepared: V2PreparedGraphUpdate,
    ) {
        require(prepared.plan.strategy == V2GraphUpdateStrategy.INCREMENTAL) {
            "server merge requires an exact incremental graph update"
        }
        require(prepared.databaseFile.canonicalFile == privateStagingDatabase.canonicalFile) {
            "prepared graph update belongs to another database"
        }
        require(prepared.plan.targetNodes == prepared.embeddingBinding.trackCount &&
            prepared.plan.targetNodes == prepared.graphBinding.nodeCount &&
            prepared.plan.embeddingDimension == prepared.embeddingBinding.dimension &&
            prepared.plan.neighborsPerNode == prepared.graphBinding.neighborsPerNode &&
            prepared.embeddingBinding.orderedTrackSetSha256 ==
            prepared.graphBinding.orderedTrackSetSha256
        ) { "prepared graph-update bindings disagree" }
        require(prepared.embeddingFile.isFile &&
            prepared.embeddingFile.length() == prepared.embeddingBinding.byteLength &&
            prepared.graphFile.isFile &&
            prepared.graphFile.length() == prepared.graphBinding.byteLength
        ) { "prepared graph-update files changed before publication" }
        val database = EmbeddingDatabase.open(privateStagingDatabase)
        val proof = try {
            database.getSmallBinaryData(V2GraphExactProof.DATABASE_KEY)
        } finally {
            database.close()
        }
        require(V2GraphExactProof.matches(
            bytes = proof,
            graphSha256 = prepared.graphBinding.sha256,
            embeddingSha256 = prepared.embeddingBinding.fileSha256,
        )) { "prepared graph-update proof does not bind its exact files" }
    }

    private fun movePreparedFile(source: File, target: File) {
        require(source.isFile) { "prepared publication artifact is missing" }
        require(!target.exists()) { "publication destination already exists" }
        try {
            Files.move(source.toPath(), target.toPath(), StandardCopyOption.ATOMIC_MOVE)
        } catch (_: AtomicMoveNotSupportedException) {
            Files.move(source.toPath(), target.toPath())
        }
        require(target.isFile && !source.exists()) { "prepared publication artifact was not moved" }
    }

    private fun replaceDerivedGraphExactProof(
        database: SQLiteDatabase,
        graphFile: File?,
        embeddingFile: File,
        graphBinding: V2IndexGenerationGraphBinding?,
        embeddingBinding: V2OrderedEmbeddingBinding,
        install: Boolean,
    ) {
        val hasBinaryDataTable = database.rawQuery(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1",
            arrayOf("binary_data"),
        ).use { it.moveToFirst() }
        if (hasBinaryDataTable) {
            database.delete(
                "binary_data",
                "key = ?",
                arrayOf(V2GraphExactProof.DATABASE_KEY),
            )
        }
        if (!install) return

        val exactGraphFile = requireNotNull(graphFile) {
            "exact derived graph proof has no graph file"
        }
        val exactGraphBinding = requireNotNull(graphBinding) {
            "exact derived graph proof has no graph binding"
        }
        require(exactGraphFile.isFile && embeddingFile.isFile) {
            "exact derived graph proof assets are missing"
        }
        val proof = V2GraphExactProof.createBoundHashes(
            graphSha256 = exactGraphBinding.sha256,
            embeddingSha256 = embeddingBinding.fileSha256,
        )
        require(V2GraphExactProof.matches(
            bytes = proof,
            graphSha256 = exactGraphBinding.sha256,
            embeddingSha256 = embeddingBinding.fileSha256,
        )) { "derived graph or embedding changed while binding its exact proof" }
        if (!hasBinaryDataTable) {
            database.execSQL(
                """
                CREATE TABLE binary_data (
                    key TEXT PRIMARY KEY,
                    data BLOB NOT NULL
                )
                """.trimIndent(),
            )
        }
        val values = ContentValues().apply {
            put("key", V2GraphExactProof.DATABASE_KEY)
            put("data", proof)
        }
        require(database.insertWithOnConflict(
            "binary_data",
            null,
            values,
            SQLiteDatabase.CONFLICT_REPLACE,
        ) != -1L) { "unable to install exact derived graph proof" }
    }

    private fun requireCurrentBase(expected: V2ResolvedActiveIndexGeneration) {
        val current = V2IndexGenerationReader.requireActive(filesDir, gson)
        require(current.manifest.generationId == expected.manifest.generationId &&
            current.manifestSha256 == expected.manifestSha256 &&
            current.databaseFile.canonicalFile == expected.databaseFile.canonicalFile
        ) { "active base generation changed during maintenance publication" }
    }

    private fun requireMaintenanceSubset(
        base: V2ResolvedActiveIndexGeneration,
        database: SQLiteDatabase,
        retainedTrackCount: Int,
    ) {
        require(retainedTrackCount in 1 until base.manifest.trackCount) {
            "maintenance must retain a non-empty strict subset of the active generation"
        }
        val alias = "maintenance_base"
        val escaped = base.databaseFile.canonicalPath.replace("'", "''")
        database.execSQL("ATTACH DATABASE '$escaped' AS $alias")
        try {
            val embeddingMismatch = database.rawQuery(
                """
                SELECT COUNT(*)
                FROM ${V2EmbeddingCommitRepository.EMBEDDING_TABLE} current
                LEFT JOIN $alias.${V2EmbeddingCommitRepository.EMBEDDING_TABLE} base
                  ON base.track_id = current.track_id
                 AND base.embedding = current.embedding
                WHERE base.track_id IS NULL
                """.trimIndent(),
                null,
            ).use { cursor -> check(cursor.moveToFirst()); cursor.getLong(0) }
            require(embeddingMismatch == 0L) {
                "maintenance changed or introduced embedding rows"
            }
            val trackMismatch = database.rawQuery(
                """
                SELECT COUNT(*)
                FROM tracks current
                LEFT JOIN $alias.tracks base ON base.id = current.id
                WHERE base.id IS NULL OR NOT (
                    base.metadata_key IS current.metadata_key AND
                    base.filename_key IS current.filename_key AND
                    base.artist IS current.artist AND
                    base.album IS current.album AND
                    base.title IS current.title AND
                    base.duration_ms IS current.duration_ms AND
                    base.file_path IS current.file_path AND
                    base.source IS current.source
                )
                """.trimIndent(),
                null,
            ).use { cursor -> check(cursor.moveToFirst()); cursor.getLong(0) }
            require(trackMismatch == 0L) { "maintenance changed or introduced track rows" }
            requireMaintenanceReceiptsAreSubset(database, alias)
        } finally {
            database.execSQL("DETACH DATABASE $alias")
        }
    }

    private fun requireMaintenanceReceiptsAreSubset(
        database: SQLiteDatabase,
        baseAlias: String,
    ) {
        val receiptTable = V2EmbeddingCommitRepository.RECEIPT_TABLE
        fun tableExists(schema: String): Boolean = database.rawQuery(
            "SELECT 1 FROM $schema.sqlite_master WHERE type = 'table' AND name = ? LIMIT 1",
            arrayOf(receiptTable),
        ).use { it.moveToFirst() }

        val baseHasReceipts = tableExists(baseAlias)
        val currentHasReceipts = tableExists("main")
        if (!baseHasReceipts) {
            require(!currentHasReceipts || database.rawQuery(
                "SELECT COUNT(*) FROM $receiptTable",
                null,
            ).use { cursor -> check(cursor.moveToFirst()); cursor.getLong(0) } == 0L) {
                "maintenance introduced V2 receipts"
            }
            return
        }
        val expectedRetained = database.rawQuery(
            """
            SELECT COUNT(*) FROM $baseAlias.$receiptTable base
            JOIN ${V2EmbeddingCommitRepository.EMBEDDING_TABLE} current
              ON current.track_id = base.track_id
            """.trimIndent(),
            null,
        ).use { cursor -> check(cursor.moveToFirst()); cursor.getLong(0) }
        require(currentHasReceipts) { "maintenance removed the V2 receipt table" }
        val currentCount = database.rawQuery(
            "SELECT COUNT(*) FROM $receiptTable",
            null,
        ).use { cursor -> check(cursor.moveToFirst()); cursor.getLong(0) }
        require(currentCount == expectedRetained) {
            "maintenance removed or introduced receipt evidence"
        }
        val mismatched = database.rawQuery(
            """
            SELECT COUNT(*)
            FROM $receiptTable current
            LEFT JOIN $baseAlias.$receiptTable base ON
                base.receipt_schema_version = current.receipt_schema_version AND
                base.work_id = current.work_id AND
                base.stable_track_span_id = current.stable_track_span_id AND
                base.stable_identity_spec_id = current.stable_identity_spec_id AND
                base.stable_identity_strength = current.stable_identity_strength AND
                base.embedding_spec_id = current.embedding_spec_id AND
                base.provider_physical_path = current.provider_physical_path AND
                base.provider_offset_ms = current.provider_offset_ms AND
                base.provider_duration_ms = current.provider_duration_ms AND
                base.track_id = current.track_id AND
                base.metadata_sha256 = current.metadata_sha256 AND
                base.embedding_byte_length = current.embedding_byte_length AND
                base.embedding_sha256 = current.embedding_sha256 AND
                base.committed_at_epoch_ms = current.committed_at_epoch_ms
            WHERE base.track_id IS NULL
            """.trimIndent(),
            null,
        ).use { cursor -> check(cursor.moveToFirst()); cursor.getLong(0) }
        require(mismatched == 0L) { "maintenance changed V2 receipt evidence" }
    }

    private fun requireServerMergeExtension(
        base: V2ResolvedActiveIndexGeneration,
        database: SQLiteDatabase,
        mergedTrackCount: Int,
        addedTrackCount: Int,
    ) {
        require(addedTrackCount > 0 &&
            mergedTrackCount == base.manifest.trackCount + addedTrackCount
        ) { "server merge track delta differs from its publication claim" }
        val alias = "server_merge_base"
        val escaped = base.databaseFile.canonicalPath.replace("'", "''")
        database.execSQL("ATTACH DATABASE '$escaped' AS $alias")
        try {
            val invalidAdditionCount = database.rawQuery(
                """
                SELECT COUNT(*)
                FROM tracks current
                LEFT JOIN $alias.tracks base ON base.id = current.id
                LEFT JOIN ${V2EmbeddingCommitRepository.EMBEDDING_TABLE} embedding
                  ON embedding.track_id = current.id
                WHERE base.id IS NULL AND
                      (embedding.track_id IS NULL OR current.source != 'server')
                """.trimIndent(),
                null,
            ).use { cursor -> check(cursor.moveToFirst()); cursor.getLong(0) }
            require(invalidAdditionCount == 0L) {
                "server merge introduced a non-server track or a track without an embedding"
            }
            requireServerMergeAuditReceipts(
                database = database,
                baseAlias = alias,
                addedTrackCount = addedTrackCount,
            )
        } finally {
            database.execSQL("DETACH DATABASE $alias")
        }
    }

    private fun requireServerMergeAuditReceipts(
        database: SQLiteDatabase,
        baseAlias: String,
        addedTrackCount: Int,
    ) {
        val table = V2ServerMergeReceiptContract.TABLE
        fun tableExists(schema: String): Boolean = database.rawQuery(
            "SELECT 1 FROM $schema.sqlite_master WHERE type = 'table' AND name = ? LIMIT 1",
            arrayOf(table),
        ).use { it.moveToFirst() }
        val baseHasTable = tableExists(baseAlias)
        require(tableExists("main")) { "server merge omitted its row receipt table" }
        val baseCount = if (baseHasTable) {
            database.rawQuery("SELECT COUNT(*) FROM $baseAlias.$table", null).use { cursor ->
                check(cursor.moveToFirst())
                cursor.getLong(0)
            }
        } else {
            0L
        }
        val currentCount = database.rawQuery("SELECT COUNT(*) FROM $table", null).use { cursor ->
            check(cursor.moveToFirst())
            cursor.getLong(0)
        }
        require(currentCount == baseCount + addedTrackCount) {
            "server merge receipt delta differs from its appended track count"
        }
        val baseJoin = if (baseHasTable) {
            "LEFT JOIN $baseAlias.$table inherited ON inherited.track_id = receipt.track_id"
        } else {
            "LEFT JOIN $baseAlias.tracks inherited ON 0"
        }
        val invalidNew = database.rawQuery(
            """
            SELECT COUNT(*)
            FROM $table receipt
            $baseJoin
            JOIN tracks track ON track.id = receipt.track_id
            JOIN ${V2EmbeddingCommitRepository.EMBEDDING_TABLE} embedding
              ON embedding.track_id = receipt.track_id
            WHERE inherited.${if (baseHasTable) "track_id" else "id"} IS NULL AND NOT (
                receipt.receipt_schema_version = ${V2ServerMergeReceiptContract.SCHEMA_VERSION} AND
                receipt.bundle_id LIKE 'server-bundle-v1-%' AND
                length(receipt.source_sha256) = 64 AND
                receipt.source_size_bytes > 0 AND
                receipt.provider_file_id > 0 AND
                receipt.provider_physical_path = track.file_path AND
                length(receipt.embedding_sha256) = 64 AND
                receipt.embedding_spec_id = '${V2ServerBundleContract.EMBEDDING_SPEC_ID}' AND
                receipt.output_space_id = '${V2ServerBundleContract.OUTPUT_SPACE_ID}' AND
                receipt.merged_at_epoch_ms = 0 AND
                track.source = 'server'
            )
            """.trimIndent(),
            null,
        ).use { cursor -> check(cursor.moveToFirst()); cursor.getLong(0) }
        require(invalidNew == 0L) { "server merge contains an invalid per-row receipt" }
        var validatedNewReceipts = 0
        database.rawQuery(
            """
            SELECT receipt.bundle_id, receipt.root_id, receipt.relative_path,
                   receipt.source_sha256, receipt.source_size_bytes,
                   receipt.provider_physical_path, receipt.embedding_sha256,
                   receipt.embedding_spec_id, receipt.output_space_id, embedding.embedding
            FROM $table receipt
            JOIN ${V2EmbeddingCommitRepository.EMBEDDING_TABLE} embedding
              ON embedding.track_id = receipt.track_id
            LEFT JOIN $baseAlias.tracks base ON base.id = receipt.track_id
            WHERE base.id IS NULL
            ORDER BY receipt.track_id
            """.trimIndent(),
            null,
        ).use { cursor ->
            while (cursor.moveToNext()) {
                val bundleId = cursor.getString(0)
                val rootId = cursor.getString(1)
                val relativePath = cursor.getString(2)
                val sourceSha256 = cursor.getString(3)
                val sourceSizeBytes = cursor.getLong(4)
                val providerPath = cursor.getString(5)
                val embeddingSha256 = cursor.getString(6)
                val embeddingSpecId = cursor.getString(7)
                val outputSpaceId = cursor.getString(8)
                val embedding = cursor.getBlob(9)
                require(bundleId.matches(SERVER_BUNDLE_ID) &&
                    rootId.matches(SERVER_ROOT_ID) &&
                    V2ServerBundlePathPolicy.requireCanonicalRelativePath(relativePath) ==
                    relativePath &&
                    sourceSha256.matches(SHA256) && sourceSizeBytes > 0L &&
                    File(providerPath).isAbsolute &&
                    TrackNormalization.normalizePath(providerPath)?.isNotBlank() == true &&
                    embeddingSha256.matches(SHA256) &&
                    embeddingSpecId == V2ServerBundleContract.EMBEDDING_SPEC_ID &&
                    outputSpaceId == V2ServerBundleContract.OUTPUT_SPACE_ID
                ) { "server merge row receipt has invalid canonical evidence" }
                V2Clamp3VectorCodec.requireValidBlob(embedding)
                require(V2ArtifactDigests.sha256(embedding) == embeddingSha256) {
                    "server merge row receipt embedding SHA-256 does not match its vector"
                }
                validatedNewReceipts++
            }
        }
        require(validatedNewReceipts == addedTrackCount) {
            "server merge validated $validatedNewReceipts new receipts, expected $addedTrackCount"
        }
        val additionsWithoutReceipt = database.rawQuery(
            """
            SELECT COUNT(*)
            FROM tracks current
            LEFT JOIN $baseAlias.tracks base ON base.id = current.id
            LEFT JOIN $table receipt ON receipt.track_id = current.id
            WHERE base.id IS NULL AND receipt.track_id IS NULL
            """.trimIndent(),
            null,
        ).use { cursor -> check(cursor.moveToFirst()); cursor.getLong(0) }
        require(additionsWithoutReceipt == 0L) {
            "server merge appended a track without its exact row receipt"
        }
    }

    private fun requireCoverageExtendsBase(
        ledger: IndexingJobLedger,
        current: V2EmbeddingSpecCoverageBinding,
        database: SQLiteDatabase,
        importedRowAuthorization: V2ImportedRowSupersessionAuthorization?,
    ) {
        val baseGenerationId = ledger.jobSpec.baseGenerationId ?: return
        val base = requireActiveBaseGeneration(baseGenerationId)
        require(base.manifest.generationId == baseGenerationId) {
            "recorded base generation changed before coverage validation"
        }
        importedRowAuthorization?.let { authorization ->
            require(authorization.baseGenerationId == base.manifest.generationId &&
                authorization.baseManifestSha256 == base.manifestSha256 &&
                authorization.baseDatabaseByteLength == base.manifest.databaseByteLength &&
                authorization.baseDatabaseSha256 == base.manifest.databaseSha256 &&
                authorization.baseDatabaseContentSha256 ==
                    base.manifest.databaseContentSha256
            ) { "imported-row authorization is not bound to the recorded base generation" }
        }
        val inherited = base.manifest.embeddingCoverage
        val disposition = importedRowAuthorization?.let { authorization ->
            V2ImportedRowActivationPolicy.partition(ledger, authorization)
        } ?: V2ImportedRowActivationDisposition(emptyList(), emptyList())
        val expectedSupersessions = disposition.committedSupersessions
        val uncommittedSupersessions = disposition.uncommittedSupersessions
        val expectedPredecessorIds = expectedSupersessions.mapTo(linkedSetOf()) {
            requireNotNull(it.predecessor).trackId
        }
        val actualSupersessions = readCurrentJobSupersessions(database, ledger.jobSpec.specId)
        require(actualSupersessions.map { it.workId }.toSet() ==
            expectedSupersessions.map { it.workId }.toSet()
        ) { "generation supersession audit does not match preflight authorization" }

        require(current.totalTrackCount >= inherited.totalTrackCount) {
            "new generation removed inherited rows beyond authorized replacements"
        }
        val expectedCompatibility = if (importedRowAuthorization == null) {
            inherited.compatibilityBase
        } else {
            SQLiteDatabase.openDatabase(
                base.databaseFile.path,
                null,
                SQLiteDatabase.OPEN_READONLY,
            ).use { baseDatabase ->
                importedRowAuthorization.works
                    .filter { it.kind == V2ImportedRowCommitKind.SUPERSESSION }
                    .forEach { work ->
                        requireBasePredecessorMatches(
                            baseDatabase,
                            checkNotNull(work.predecessor),
                        )
                    }
                V2EmbeddingSpecCoverage.inspectCompatibilityBase(
                    baseDatabase,
                    expectedPredecessorIds,
                )
            }
        }
        require(current.compatibilityBase == expectedCompatibility) {
            "unreceipted compatibility base differs from the authorized predecessor removal"
        }
        inherited.receiptSpecTrackCounts.forEach { (specId, count) ->
            require((current.receiptSpecTrackCounts[specId] ?: 0) >= count) {
                "new generation removed inherited receipt coverage for $specId"
            }
        }
        uncommittedSupersessions.forEach { work ->
            requireCurrentPredecessorMatches(
                database,
                checkNotNull(work.predecessor),
            )
        }
        requireExactInheritedRows(
            database = database,
            base = base,
            ledger = ledger,
            authorization = importedRowAuthorization,
            expectedSupersessions = expectedSupersessions,
            actualSupersessions = actualSupersessions,
        )
    }

    private fun requireActiveBaseGeneration(
        generationId: String,
    ): V2ResolvedActiveIndexGeneration {
        val active = V2IndexGenerationReader.requireActive(filesDir)
        require(active.manifest.generationId == generationId) {
            "planned base generation is no longer active"
        }
        return active
    }

    private fun readCurrentJobSupersessions(
        database: SQLiteDatabase,
        jobSpecId: String,
    ): List<ActivationSupersessionEvidence> {
        val table = V2EmbeddingCommitRepository.IMPORTED_ROW_SUPERSESSION_TABLE
        val exists = database.rawQuery(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1",
            arrayOf(table),
        ).use { it.moveToFirst() }
        if (!exists) return emptyList()
        return database.rawQuery(
            """
            SELECT supersession_schema_version, work_id, embedding_spec_id, job_spec_id,
                   base_generation_id, base_manifest_sha256, base_database_sha256,
                   private_base_binding_id, provider_snapshot_generation,
                   predecessor_track_id, predecessor_metadata_sha256,
                   predecessor_embedding_byte_length, predecessor_embedding_sha256,
                   replacement_track_id, committed_at_epoch_ms
            FROM $table
            WHERE job_spec_id = ?
            ORDER BY work_id
            """.trimIndent(),
            arrayOf(jobSpecId),
        ).use { cursor ->
            buildList {
                while (cursor.moveToNext()) {
                    add(
                        ActivationSupersessionEvidence(
                            schemaVersion = cursor.getInt(0),
                            workId = cursor.getString(1),
                            embeddingSpecId = cursor.getString(2),
                            jobSpecId = cursor.getString(3),
                            baseGenerationId = cursor.getString(4),
                            baseManifestSha256 = cursor.getString(5),
                            baseDatabaseSha256 = cursor.getString(6),
                            privateBaseBindingId = cursor.getString(7),
                            providerSnapshotGeneration = cursor.getString(8),
                            predecessorTrackId = cursor.getLong(9),
                            predecessorMetadataSha256 = cursor.getString(10),
                            predecessorEmbeddingByteLength = cursor.getInt(11),
                            predecessorEmbeddingSha256 = cursor.getString(12),
                            replacementTrackId = cursor.getLong(13),
                            committedAtEpochMs = cursor.getLong(14),
                        ),
                    )
                }
            }
        }
    }

    private fun requireBasePredecessorMatches(
        baseDatabase: SQLiteDatabase,
        predecessor: V2ImportedPredecessorEvidence,
    ) = requirePredecessorMatches(
        database = baseDatabase,
        predecessor = predecessor,
        location = "base",
    )

    private fun requireCurrentPredecessorMatches(
        database: SQLiteDatabase,
        predecessor: V2ImportedPredecessorEvidence,
    ) = requirePredecessorMatches(
        database = database,
        predecessor = predecessor,
        location = "staged generation",
    )

    private fun requirePredecessorMatches(
        database: SQLiteDatabase,
        predecessor: V2ImportedPredecessorEvidence,
        location: String,
    ) {
        val receiptTable = V2EmbeddingCommitRepository.RECEIPT_TABLE
        val hasReceipts = database.rawQuery(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1",
            arrayOf(receiptTable),
        ).use { it.moveToFirst() }
        val receiptExpression = if (hasReceipts) {
            "(SELECT COUNT(*) FROM $receiptTable r WHERE r.track_id = t.id)"
        } else {
            "0"
        }
        database.rawQuery(
            """
            SELECT t.metadata_key, t.filename_key, t.artist, t.album, t.title,
                   t.duration_ms, t.file_path, t.source, e.embedding, $receiptExpression
            FROM ${V2EmbeddingCommitRepository.TRACK_TABLE} t
            JOIN ${V2EmbeddingCommitRepository.EMBEDDING_TABLE} e ON e.track_id = t.id
            WHERE t.id = ?
            """.trimIndent(),
            arrayOf(predecessor.trackId.toString()),
        ).use { cursor ->
            val observed = if (!cursor.moveToFirst()) {
                null
            } else {
                val metadata = V2CommitTrackMetadata(
                    metadataKey = cursor.getString(0),
                    filenameKey = cursor.getString(1),
                    artist = if (cursor.isNull(2)) null else cursor.getString(2),
                    album = if (cursor.isNull(3)) null else cursor.getString(3),
                    title = if (cursor.isNull(4)) null else cursor.getString(4),
                    durationMs = cursor.getInt(5),
                    filePath = cursor.getString(6),
                    source = cursor.getString(7),
                )
                val embedding = cursor.getBlob(8)
                V2ObservedImportedPredecessorEvidence(
                    metadata = metadata,
                    embeddingByteLength = embedding.size,
                    embeddingSha256 = V2ArtifactDigests.sha256(embedding),
                    receiptCount = cursor.getLong(9),
                ).also {
                    require(!cursor.moveToNext()) {
                        "authorized imported predecessor is ambiguous in $location"
                    }
                }
            }
            V2ImportedRowPredecessorPolicy.requireExactUnreceipted(
                expected = predecessor,
                observed = observed,
                location = location,
            )
        }
    }

    private fun requireExactInheritedRows(
        database: SQLiteDatabase,
        base: V2ResolvedActiveIndexGeneration,
        ledger: IndexingJobLedger,
        authorization: V2ImportedRowSupersessionAuthorization?,
        expectedSupersessions: List<V2ImportedRowWorkAuthorization>,
        actualSupersessions: List<ActivationSupersessionEvidence>,
    ) {
        val expectedByWork = expectedSupersessions.associateBy { it.workId }
        val expectedPrivateBaseBindingId = authorization?.executionPrivateBaseBindingId(ledger)
        require(actualSupersessions.all { actual ->
            val expected = expectedByWork[actual.workId]
            val predecessor = expected?.predecessor
            actual.schemaVersion ==
                V2EmbeddingCommitRepository.IMPORTED_ROW_SUPERSESSION_SCHEMA_VERSION &&
                actual.embeddingSpecId == ledger.jobSpec.embeddingSpec.specId &&
                actual.jobSpecId == ledger.jobSpec.specId &&
                actual.baseGenerationId == base.manifest.generationId &&
                actual.baseManifestSha256 == base.manifestSha256 &&
                actual.baseDatabaseSha256 == base.manifest.databaseSha256 &&
                actual.privateBaseBindingId == expectedPrivateBaseBindingId &&
                actual.providerSnapshotGeneration ==
                    ledger.jobSpec.providerSnapshot.libraryGeneration &&
                predecessor != null &&
                actual.predecessorTrackId == predecessor?.trackId &&
                actual.predecessorMetadataSha256 == predecessor?.metadataSha256 &&
                actual.predecessorEmbeddingByteLength == predecessor?.embeddingByteLength &&
                actual.predecessorEmbeddingSha256 == predecessor?.embeddingSha256 &&
                actual.replacementTrackId > 0L && actual.committedAtEpochMs >= 0L
        }) { "generation supersession audit is stale or does not match authorization" }

        val escaped = base.databaseFile.canonicalPath.replace("'", "''")
        val alias = "supersession_base"
        database.execSQL("ATTACH DATABASE '$escaped' AS $alias")
        try {
            val supersededIds = actualSupersessions.map { it.predecessorTrackId }
            val supersededPredicate = if (supersededIds.isEmpty()) {
                ""
            } else {
                "AND base.track_id NOT IN (${supersededIds.joinToString(",") { "?" }})"
            }
            val supersededArgs = supersededIds.map(Long::toString).toTypedArray()
            val inheritedEmbeddingMismatch = database.rawQuery(
                """
                SELECT COUNT(*)
                FROM $alias.${V2EmbeddingCommitRepository.EMBEDDING_TABLE} base
                LEFT JOIN ${V2EmbeddingCommitRepository.EMBEDDING_TABLE} current
                  ON current.track_id = base.track_id AND current.embedding = base.embedding
                WHERE current.track_id IS NULL
                  $supersededPredicate
                """.trimIndent(),
                supersededArgs,
            ).use { cursor -> check(cursor.moveToFirst()); cursor.getLong(0) }
            require(inheritedEmbeddingMismatch == 0L) {
                "generation changed or removed a non-superseded inherited embedding"
            }
            val inheritedTrackMismatch = database.rawQuery(
                """
                SELECT COUNT(*)
                FROM $alias.${V2EmbeddingCommitRepository.TRACK_TABLE} base
                LEFT JOIN ${V2EmbeddingCommitRepository.TRACK_TABLE} current
                  ON current.id = base.id
                 AND current.metadata_key IS base.metadata_key
                 AND current.filename_key IS base.filename_key
                 AND current.artist IS base.artist
                 AND current.album IS base.album
                 AND current.title IS base.title
                 AND current.duration_ms IS base.duration_ms
                 AND current.file_path IS base.file_path
                 AND current.source IS base.source
                WHERE current.id IS NULL
                  ${supersededPredicate.replace("base.track_id", "base.id")}
                """.trimIndent(),
                supersededArgs,
            ).use { cursor -> check(cursor.moveToFirst()); cursor.getLong(0) }
            require(inheritedTrackMismatch == 0L) {
                "generation changed or removed a non-superseded inherited track"
            }
            requireInheritedReceiptsUnchanged(database, alias)
            actualSupersessions.forEach { actual ->
                val predecessorPresent = database.rawQuery(
                    """
                    SELECT
                      (SELECT COUNT(*) FROM ${V2EmbeddingCommitRepository.TRACK_TABLE} WHERE id = ?) +
                      (SELECT COUNT(*) FROM ${V2EmbeddingCommitRepository.EMBEDDING_TABLE}
                       WHERE track_id = ?)
                    """.trimIndent(),
                    arrayOf(
                        actual.predecessorTrackId.toString(),
                        actual.predecessorTrackId.toString(),
                    ),
                ).use { cursor -> check(cursor.moveToFirst()); cursor.getLong(0) }
                val replacementReceipt = database.rawQuery(
                    """
                    SELECT COUNT(*) FROM ${V2EmbeddingCommitRepository.RECEIPT_TABLE}
                    WHERE track_id = ? AND work_id = ? AND embedding_spec_id = ?
                    """.trimIndent(),
                    arrayOf(
                        actual.replacementTrackId.toString(),
                        actual.workId,
                        actual.embeddingSpecId,
                    ),
                ).use { cursor -> check(cursor.moveToFirst()); cursor.getLong(0) }
                require(predecessorPresent == 0L && replacementReceipt == 1L) {
                    "replacement receipt or predecessor absence is not exact"
                }
            }
        } finally {
            database.execSQL("DETACH DATABASE $alias")
        }
    }

    private fun requireInheritedReceiptsUnchanged(database: SQLiteDatabase, baseAlias: String) {
        val table = V2EmbeddingCommitRepository.RECEIPT_TABLE
        val baseHasReceipts = database.rawQuery(
            "SELECT 1 FROM $baseAlias.sqlite_master WHERE type = 'table' AND name = ? LIMIT 1",
            arrayOf(table),
        ).use { it.moveToFirst() }
        if (!baseHasReceipts) return
        val currentHasReceipts = database.rawQuery(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1",
            arrayOf(table),
        ).use { it.moveToFirst() }
        require(currentHasReceipts) { "generation removed inherited V2 receipt table" }
        val missing = database.rawQuery(
            """
            SELECT COUNT(*) FROM $baseAlias.$table base
            LEFT JOIN $table current ON
                current.receipt_schema_version = base.receipt_schema_version AND
                current.work_id = base.work_id AND
                current.stable_track_span_id = base.stable_track_span_id AND
                current.stable_identity_spec_id = base.stable_identity_spec_id AND
                current.stable_identity_strength = base.stable_identity_strength AND
                current.embedding_spec_id = base.embedding_spec_id AND
                current.provider_physical_path = base.provider_physical_path AND
                current.provider_offset_ms = base.provider_offset_ms AND
                current.provider_duration_ms = base.provider_duration_ms AND
                current.track_id = base.track_id AND
                current.metadata_sha256 = base.metadata_sha256 AND
                current.embedding_byte_length = base.embedding_byte_length AND
                current.embedding_sha256 = base.embedding_sha256 AND
                current.committed_at_epoch_ms = base.committed_at_epoch_ms
            WHERE current.track_id IS NULL
            """.trimIndent(),
            null,
        ).use { cursor -> check(cursor.moveToFirst()); cursor.getLong(0) }
        require(missing == 0L) { "generation changed or removed inherited V2 receipt evidence" }
    }

    private data class ActivationSupersessionEvidence(
        val schemaVersion: Int,
        val workId: String,
        val embeddingSpecId: String,
        val jobSpecId: String,
        val baseGenerationId: String,
        val baseManifestSha256: String,
        val baseDatabaseSha256: String,
        val privateBaseBindingId: String,
        val providerSnapshotGeneration: String,
        val predecessorTrackId: Long,
        val predecessorMetadataSha256: String,
        val predecessorEmbeddingByteLength: Int,
        val predecessorEmbeddingSha256: String,
        val replacementTrackId: Long,
        val committedAtEpochMs: Long,
    )

    private fun installInvalidationReceipt(
        database: SQLiteDatabase,
        jobSpecId: String,
        receiptEmbeddingSpecId: String,
        textRetrievalSpecId: String,
        createdAtEpochMs: Long,
        activationBindingId: String,
        embeddings: V2OrderedEmbeddingBinding,
        stableCoverage: V2StableTrackUidCoverageBinding,
        embeddingCoverage: V2EmbeddingSpecCoverageBinding,
        graph: V2IndexGenerationGraphBinding?,
    ) {
        database.beginTransaction()
        try {
            database.execSQL("DROP TABLE IF EXISTS $ACTIVATION_RECEIPT_TABLE")
            database.execSQL(
                """
                CREATE TABLE IF NOT EXISTS $ACTIVATION_RECEIPT_TABLE (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    receipt_schema_version INTEGER NOT NULL,
                    is_valid INTEGER NOT NULL CHECK (is_valid IN (0, 1)),
                    activation_binding_id TEXT NOT NULL,
                    job_spec_id TEXT NOT NULL,
                    receipt_embedding_spec_id TEXT NOT NULL,
                    text_retrieval_spec_id TEXT NOT NULL,
                    embedding_coverage_sha256 TEXT NOT NULL,
                    compatibility_base_content_sha256 TEXT,
                    database_content_sha256 TEXT NOT NULL,
                    ordered_track_set_sha256 TEXT NOT NULL,
                    stable_uid_mapping_sha256 TEXT NOT NULL,
                    embedding_sha256 TEXT NOT NULL,
                    graph_sha256 TEXT,
                    created_at_epoch_ms INTEGER NOT NULL
                )
                """.trimIndent(),
            )
            database.delete(ACTIVATION_RECEIPT_TABLE, null, null)
            val values = ContentValues().apply {
                put("singleton", 1)
                put("receipt_schema_version", ACTIVATION_RECEIPT_SCHEMA_VERSION)
                put("is_valid", 1)
                put("activation_binding_id", activationBindingId)
                put("job_spec_id", jobSpecId)
                put("receipt_embedding_spec_id", receiptEmbeddingSpecId)
                put("text_retrieval_spec_id", textRetrievalSpecId)
                put("embedding_coverage_sha256", embeddingCoverage.mappingSha256)
                put(
                    "compatibility_base_content_sha256",
                    embeddingCoverage.compatibilityBase?.orderedContentSha256,
                )
                put("database_content_sha256", embeddings.databaseContentSha256)
                put("ordered_track_set_sha256", embeddings.orderedTrackSetSha256)
                put("stable_uid_mapping_sha256", stableCoverage.mappingSha256)
                put("embedding_sha256", embeddings.fileSha256)
                put("graph_sha256", graph?.sha256)
                put("created_at_epoch_ms", createdAtEpochMs)
            }
            require(database.insertOrThrow(ACTIVATION_RECEIPT_TABLE, null, values) > 0L) {
                "unable to install generation receipt"
            }
            installInvalidationTrigger(database, "embeddings_insert", "embeddings_clamp3", "INSERT")
            installInvalidationTrigger(database, "embeddings_update", "embeddings_clamp3", "UPDATE")
            installInvalidationTrigger(database, "embeddings_delete", "embeddings_clamp3", "DELETE")
            installInvalidationTrigger(database, "tracks_insert", "tracks", "INSERT")
            installInvalidationTrigger(database, "tracks_update", "tracks", "UPDATE")
            installInvalidationTrigger(database, "tracks_delete", "tracks", "DELETE")
            database.setTransactionSuccessful()
        } finally {
            database.endTransaction()
        }
    }

    private fun installInvalidationTrigger(
        database: SQLiteDatabase,
        suffix: String,
        table: String,
        operation: String,
    ) {
        val name = "v2_invalidate_generation_$suffix"
        database.execSQL("DROP TRIGGER IF EXISTS $name")
        database.execSQL(
            """
            CREATE TRIGGER $name AFTER $operation ON $table
            BEGIN
                UPDATE $ACTIVATION_RECEIPT_TABLE SET is_valid = 0 WHERE singleton = 1;
            END
            """.trimIndent(),
        )
    }

    private fun writeManifest(file: File, manifest: V2IndexGenerationManifest) {
        FileOutputStream(file).use { output ->
            OutputStreamWriter(output, StandardCharsets.UTF_8).use { writer ->
                gson.toJson(manifest, writer)
                writer.flush()
                output.fd.sync()
            }
        }
    }

    private fun copyAndSync(
        source: File,
        target: File,
        onProgress: ((completedBytes: Long, totalBytes: Long) -> Unit)? = null,
    ) {
        require(source.isFile) { "graph source is missing" }
        FileInputStream(source).channel.use { input ->
            FileOutputStream(target).channel.use { output ->
                var offset = 0L
                val totalBytes = input.size()
                onProgress?.invoke(offset, totalBytes)
                while (offset < totalBytes) {
                    val copied = input.transferTo(offset, totalBytes - offset, output)
                    require(copied > 0L) { "graph copy made no progress" }
                    offset += copied
                    onProgress?.invoke(offset, totalBytes)
                }
                output.force(true)
            }
        }
    }

}

/** Origin-specific rules kept separate so bootstrap data can never masquerade as V2 output. */
object V2IndexGenerationManifestPolicy {
    const val BOOTSTRAP_JOB_ID = "bootstrap-compatibility-import-v1"
    const val MAINTENANCE_JOB_ID = "library-maintenance-v1"
    const val SERVER_MERGE_JOB_ID = "server-index-merge-v1"

    fun requireValidProvenance(manifest: V2IndexGenerationManifest) {
        require(manifest.activationBindingId.matches(ACTIVATION_BINDING_ID) &&
            manifest.createdAtEpochMs >= 0L &&
            (manifest.baseGenerationId == null || manifest.baseGenerationId.matches(GENERATION_ID))
        ) { "generation manifest has invalid common provenance" }
        when (manifest.origin) {
            V2IndexGenerationOrigin.INDEXING_JOB -> require(
                manifest.jobId.matches(JOB_ID) && manifest.jobSpecId.matches(JOB_SPEC_ID) &&
                    manifest.graphPolicy in setOf(
                        V2IndexGenerationGraphPolicy.ABSENT,
                        V2IndexGenerationGraphPolicy.EXPLICIT_REBUILD,
                    ) &&
                    (manifest.graphPolicy == V2IndexGenerationGraphPolicy.ABSENT) ==
                    (manifest.graph == null),
            ) { "indexing generation has invalid job provenance" }

            V2IndexGenerationOrigin.BOOTSTRAP_COMPATIBILITY -> require(
                manifest.jobId == BOOTSTRAP_JOB_ID &&
                    manifest.jobSpecId == V2IndexGenerationIdentity.bootstrapSpecId(
                        manifest.receiptEmbeddingSpec,
                        manifest.textRetrievalSpec,
                    ) &&
                    manifest.baseGenerationId == null &&
                    !manifest.rebuildDerivedIndexes &&
                    manifest.graphPolicy in setOf(
                        V2IndexGenerationGraphPolicy.ABSENT,
                        V2IndexGenerationGraphPolicy.VALIDATED_COMPATIBILITY_IMPORT,
                    ) &&
                    (manifest.graphPolicy == V2IndexGenerationGraphPolicy.ABSENT) ==
                    (manifest.graph == null) &&
                    manifest.createdAtEpochMs == 0L,
            ) { "bootstrap generation has invalid compatibility provenance" }

            V2IndexGenerationOrigin.LIBRARY_MAINTENANCE -> require(
                manifest.jobId == MAINTENANCE_JOB_ID &&
                    manifest.baseGenerationId != null &&
                    manifest.jobSpecId == V2IndexGenerationIdentity.maintenanceSpecId(
                        manifest.baseGenerationId,
                        manifest.receiptEmbeddingSpec,
                        manifest.textRetrievalSpec,
                    ) &&
                    !manifest.rebuildDerivedIndexes &&
                    manifest.graphPolicy in setOf(
                        V2IndexGenerationGraphPolicy.ABSENT,
                        V2IndexGenerationGraphPolicy.BASE_BOUND_DELETION_REPAIR,
                    ) &&
                    (manifest.graphPolicy == V2IndexGenerationGraphPolicy.ABSENT) ==
                    (manifest.graph == null) &&
                manifest.createdAtEpochMs == 0L,
            ) { "maintenance generation has invalid base-bound provenance" }

            V2IndexGenerationOrigin.SERVER_MERGE -> require(
                manifest.jobId == SERVER_MERGE_JOB_ID &&
                    manifest.baseGenerationId != null &&
                    manifest.jobSpecId.matches(SERVER_MERGE_SPEC_ID) &&
                    !manifest.rebuildDerivedIndexes &&
                    manifest.graphPolicy ==
                    V2IndexGenerationGraphPolicy.BASE_BOUND_ADDITION_UPDATE &&
                    manifest.graph != null &&
                    manifest.createdAtEpochMs == 0L,
            ) { "server merge generation has invalid base-bound provenance" }
        }
    }

    fun requireValidCoverage(manifest: V2IndexGenerationManifest) {
        val coverage = manifest.embeddingCoverage
        val receiptSpecCount = coverage.receiptSpecTrackCounts[
            manifest.receiptEmbeddingSpec.specId
        ]
        require(
            coverage.totalTrackCount == manifest.trackCount &&
                coverage.receiptBoundTrackCount >= 0 &&
                coverage.receiptSpecTrackCounts.values.all { it > 0 } &&
                coverage.receiptSpecTrackCounts.values.sum() == coverage.receiptBoundTrackCount &&
                coverage.mappingSha256.matches(SHA256) &&
                manifest.stableTrackUidCoverage.coveredTrackCount ==
                coverage.receiptBoundTrackCount,
        ) { "generation embedding-spec coverage is inconsistent" }

        when (manifest.origin) {
            V2IndexGenerationOrigin.INDEXING_JOB -> require(
                coverage.receiptBoundTrackCount > 0 &&
                    coverage.receiptSpecTrackCounts.size == 1 &&
                    receiptSpecCount == coverage.receiptBoundTrackCount,
            ) { "indexing generation has no exact receipt-bound V2 coverage" }

            V2IndexGenerationOrigin.BOOTSTRAP_COMPATIBILITY -> require(
                coverage.receiptBoundTrackCount == 0 &&
                    coverage.receiptSpecTrackCounts.isEmpty() &&
                    manifest.stableTrackUidCoverage.coveredTrackCount == 0 &&
                    manifest.stableTrackUidCoverage.uncoveredTrackCount == manifest.trackCount &&
                    manifest.stableTrackUidCoverage.uniqueStableTrackSpanCount == 0 &&
                    manifest.stableTrackUidCoverage.fullContentIdentityCount == 0 &&
                    manifest.stableTrackUidCoverage.sampledContentIdentityCount == 0,
            ) { "bootstrap generation falsely claims V2 row or stable-identity coverage" }

            V2IndexGenerationOrigin.LIBRARY_MAINTENANCE,
            V2IndexGenerationOrigin.SERVER_MERGE -> require(
                coverage.receiptSpecTrackCounts.size <= 1 &&
                    (coverage.receiptBoundTrackCount == 0 ||
                        receiptSpecCount == coverage.receiptBoundTrackCount),
            ) { "base-bound generation has mixed or unsupported receipt coverage" }
        }

        val compatibilityCount = manifest.trackCount - coverage.receiptBoundTrackCount
        val compatibilityBase = coverage.compatibilityBase
        require(
            if (compatibilityCount == 0) {
                compatibilityBase == null
            } else {
                compatibilityBase != null &&
                    compatibilityBase.trackCount == compatibilityCount &&
                    compatibilityBase.provenancePolicyId ==
                    V2EmbeddingSpecCoverage.COMPATIBILITY_BASE_PROVENANCE_POLICY_ID &&
                    compatibilityBase.orderedContentSha256.matches(SHA256)
            },
        ) { "generation compatibility-base coverage is inconsistent" }
    }

    private val JOB_ID = Regex("^[A-Za-z0-9._-]{1,128}$")
    private val JOB_SPEC_ID = Regex("^job-spec-v5-[0-9a-f]{64}$")
    private val SERVER_MERGE_SPEC_ID = Regex("^server-merge-spec-v1-[0-9a-f]{64}$")
    private val ACTIVATION_BINDING_ID = Regex("^activation-binding-v3-[0-9a-f]{64}$")
}

data class V2GenerationArtifactHashProgress(
    val filename: String,
    val completedBytes: Long,
    val totalBytes: Long,
)

internal object V2FreshlyInstalledGenerationBindingResolver {
    fun requireResolved(
        directory: File,
        manifest: V2IndexGenerationManifest,
        manifestSha256: String,
        manifestByteLength: Long,
        gson: Gson = GsonBuilder().disableHtmlEscaping().create(),
    ): V2ResolvedActiveIndexGeneration {
        require(manifestSha256.matches(SHA256) && manifestByteLength > 0L) {
            "invalid freshly installed manifest binding"
        }
        val canonicalDirectory = directory.canonicalFile
        requireGenerationManifestContract(canonicalDirectory, manifest)
        val manifestFile = requireDirectGenerationFile(
            directory = canonicalDirectory,
            filename = MANIFEST_FILE,
            expectedByteLength = manifestByteLength,
            label = "manifest",
        )
        require(V2FileSha256.digest(manifestFile) == manifestSha256) {
            "freshly installed manifest SHA changed"
        }
        val installedManifest = manifestFile.bufferedReader(StandardCharsets.UTF_8).use { reader ->
            gson.fromJson(reader, V2IndexGenerationManifest::class.java)
        } ?: error("freshly installed manifest is empty")
        require(installedManifest == manifest) { "freshly installed manifest changed" }

        val database = requireDirectGenerationFile(
            directory = canonicalDirectory,
            filename = manifest.databaseRelativePath,
            expectedByteLength = manifest.databaseByteLength,
            label = "database",
        )
        val embedding = requireDirectGenerationFile(
            directory = canonicalDirectory,
            filename = manifest.embeddingRelativePath,
            expectedByteLength = manifest.embeddingByteLength,
            label = "PEMB",
        )
        val graph = manifest.graph?.let { binding ->
            requireDirectGenerationFile(
                directory = canonicalDirectory,
                filename = binding.relativePath,
                expectedByteLength = binding.byteLength,
                label = "graph",
            )
        }
        if (graph == null) {
            require(!File(canonicalDirectory, V2GraphGenerationFile.GRAPH_FILE).exists()) {
                "graph asset exists despite ABSENT graph policy"
            }
        }
        return V2ResolvedActiveIndexGeneration(
            manifest = manifest,
            manifestSha256 = manifestSha256,
            directory = canonicalDirectory,
            databaseFile = database,
            embeddingFile = embedding,
            graphFile = graph,
        )
    }

    private fun requireDirectGenerationFile(
        directory: File,
        filename: String,
        expectedByteLength: Long,
        label: String,
    ): File {
        require(expectedByteLength > 0L) { "generation $label has no bytes" }
        val file = File(directory, filename).canonicalFile
        require(file.parentFile == directory && file.isFile && file.length() == expectedByteLength) {
            "freshly installed generation $label stat/path binding mismatch"
        }
        return file
    }
}

private fun requireGenerationManifestContract(
    directory: File,
    manifest: V2IndexGenerationManifest,
) {
    require(directory.isDirectory) { "generation directory is missing" }
    require(manifest.schemaVersion == MANIFEST_SCHEMA_VERSION &&
        manifest.generationId == V2IndexGenerationIdentity.generationId(manifest)
    ) { "generation manifest identity mismatch" }
    V2IndexGenerationManifestPolicy.requireValidProvenance(manifest)
    require(
        manifest.activationBindingId == V2IndexGenerationIdentity.activationBindingId(manifest),
    ) { "generation activation binding identity mismatch" }
    require(directory.name == manifest.generationId) { "generation directory ID mismatch" }
    require(manifest.databaseRelativePath == DATABASE_FILE &&
        manifest.embeddingRelativePath == EMBEDDING_FILE
    ) { "generation contains unsupported asset paths" }
    require(manifest.receiptEmbeddingSpec.outputDimension == V2_CLAMP3_DIMENSION) {
        "generation embedding spec dimension mismatch"
    }
    V2IndexingLedgerValidator.requireValidEmbeddingSpec(manifest.receiptEmbeddingSpec)
    V2IndexingLedgerValidator.requireValidTextRetrievalSpec(manifest.textRetrievalSpec)
    require(
        manifest.textRetrievalSpec.compatibleAudioEmbeddingSpecId ==
            manifest.receiptEmbeddingSpec.specId &&
            manifest.textRetrievalSpec.outputDimension ==
            manifest.receiptEmbeddingSpec.outputDimension,
    ) { "generation text-retrieval spec is incompatible with audio embeddings" }
    require(manifest.databaseByteLength > 0L && manifest.databaseSha256.matches(SHA256) &&
        manifest.embeddingByteLength > 0L && manifest.embeddingSha256.matches(SHA256) &&
        manifest.databaseContentSha256.matches(SHA256) &&
        manifest.orderedTrackSetSha256.matches(SHA256) &&
        manifest.stableTrackUidCoverage.mappingSha256.matches(SHA256)
    ) { "generation file or semantic binding is invalid" }
    require(manifest.trackCount > 0 && manifest.embeddingDimension == V2_CLAMP3_DIMENSION) {
        "generation embedding shape is invalid"
    }
    require(manifest.stableTrackUidCoverage.coveredTrackCount >= 0 &&
        manifest.stableTrackUidCoverage.uncoveredTrackCount >= 0 &&
        manifest.stableTrackUidCoverage.coveredTrackCount +
        manifest.stableTrackUidCoverage.uncoveredTrackCount == manifest.trackCount &&
        manifest.stableTrackUidCoverage.uniqueStableTrackSpanCount in
        0..manifest.stableTrackUidCoverage.coveredTrackCount &&
        manifest.stableTrackUidCoverage.fullContentIdentityCount +
        manifest.stableTrackUidCoverage.sampledContentIdentityCount ==
        manifest.stableTrackUidCoverage.coveredTrackCount
    ) { "generation stable UID coverage is inconsistent" }
    V2IndexGenerationManifestPolicy.requireValidCoverage(manifest)
    val expectedGraphPresent = manifest.graphPolicy != V2IndexGenerationGraphPolicy.ABSENT
    require(
        manifest.rebuildDerivedIndexes ==
            (manifest.graphPolicy == V2IndexGenerationGraphPolicy.EXPLICIT_REBUILD) &&
            expectedGraphPresent == (manifest.graph != null),
    ) { "generation graph policy mismatch" }
    manifest.graph?.let { graph ->
        require(graph.relativePath == V2GraphGenerationFile.GRAPH_FILE &&
            graph.byteLength > 0L && graph.sha256.matches(SHA256) &&
            graph.nodeCount == manifest.trackCount && graph.neighborsPerNode > 0 &&
            graph.orderedTrackSetSha256 == manifest.orderedTrackSetSha256
        ) { "generation graph binding mismatch" }
    }
}

object V2IndexGenerationReader {
    @Volatile
    private var cached: CachedGeneration? = null

    @Synchronized
    fun requireActive(
        filesDir: File,
        gson: Gson = GsonBuilder().disableHtmlEscaping().create(),
        onArtifactHashProgress: (V2GenerationArtifactHashProgress) -> Unit = {},
    ): V2ResolvedActiveIndexGeneration {
        val startedNs = System.nanoTime()
        val root = File(filesDir, GENERATIONS_DIRECTORY)
        val pointer = requireActivePointer(filesDir, gson)
        val directory = File(root, pointer.generationId)
        val signature = GenerationFileSignature.capture(directory)
        val existing = cached
        val cacheHit = existing != null && existing.pointer == pointer &&
            existing.signature == signature
        val resolved = if (cacheHit) {
            checkNotNull(existing).resolved
        } else {
            requireGenerationDirectory(
                directory = directory,
                expectedManifestSha256 = pointer.manifestSha256,
                gson = gson,
                validation = V2GenerationArtifactValidation.PUBLISHED_BYTE_EXACT,
                onArtifactHashProgress = onArtifactHashProgress,
            ).also { staticResolved ->
                // Publication streams the SQL semantic binding and validates coherence before the
                // atomic pointer can select this directory. The full database, PEMB, and graph
                // byte/hash bindings above prove these are the identical immutable artifacts.
                // Replaying the SQL embedding digest on every process start cannot add evidence.
                cached = CachedGeneration(pointer, signature, staticResolved)
            }
        }
        val artifactValidationMs = elapsedMs(startedNs)
        val receiptStartedNs = System.nanoTime()
        requireValidReceipt(resolved.databaseFile, resolved.manifest)
        Log.i(
            "V2GenerationReader",
            "active generation cache=${if (cacheHit) "hit" else "miss"} " +
                "artifactValidationMs=$artifactValidationMs " +
                "receiptMs=${elapsedMs(receiptStartedNs)} " +
                "generation=${resolved.manifest.generationId}",
        )
        return resolved
    }

    private fun elapsedMs(startedNs: Long): Long =
        (System.nanoTime() - startedNs) / 1_000_000L

    @Synchronized
    internal fun rememberFreshlyPublished(
        pointer: V2ActiveGenerationPointer,
        resolved: V2ResolvedActiveIndexGeneration,
    ) {
        require(pointer.generationId == resolved.manifest.generationId &&
            pointer.manifestSha256 == resolved.manifestSha256 &&
            resolved.directory.name == pointer.generationId
        ) { "fresh publication does not match its active pointer" }
        cached = CachedGeneration(
            pointer = pointer,
            signature = GenerationFileSignature.capture(resolved.directory),
            resolved = resolved,
        )
    }

    internal fun requireActivePointer(
        filesDir: File,
        gson: Gson = GsonBuilder().disableHtmlEscaping().create(),
    ): V2ActiveGenerationPointer {
        val root = File(filesDir, GENERATIONS_DIRECTORY)
        val pointer = AtomicFile(File(root, ACTIVE_POINTER_FILE)).openRead()
            .bufferedReader(StandardCharsets.UTF_8).use { reader ->
                gson.fromJson(reader, V2ActiveGenerationPointer::class.java)
            } ?: error("active generation pointer is empty")
        require(pointer.schemaVersion == POINTER_SCHEMA_VERSION &&
            pointer.generationId.matches(GENERATION_ID) &&
            pointer.manifestSha256.matches(SHA256)
        ) { "invalid active index generation pointer" }
        return pointer
    }

    internal fun requireGenerationDirectory(
        directory: File,
        expected: V2IndexGenerationManifest? = null,
        expectedManifestSha256: String? = null,
        gson: Gson = GsonBuilder().disableHtmlEscaping().create(),
        validation: V2GenerationArtifactValidation =
            V2GenerationArtifactValidation.FULL_SEMANTIC,
        onArtifactHashProgress: (V2GenerationArtifactHashProgress) -> Unit = {},
    ): V2ResolvedActiveIndexGeneration {
        require(directory.isDirectory) { "generation directory is missing" }
        val manifestFile = File(directory, MANIFEST_FILE)
        val manifestSha256 = V2FileSha256.digest(manifestFile) { completedBytes, totalBytes ->
            onArtifactHashProgress(
                V2GenerationArtifactHashProgress(
                    manifestFile.name,
                    completedBytes,
                    totalBytes,
                ),
            )
        }
        if (expectedManifestSha256 != null) {
            require(manifestSha256 == expectedManifestSha256) { "generation manifest SHA changed" }
        }
        val manifest = manifestFile.bufferedReader(StandardCharsets.UTF_8).use { reader ->
            gson.fromJson(reader, V2IndexGenerationManifest::class.java)
        } ?: error("generation manifest is empty")
        if (expected != null) require(manifest == expected) { "generation manifest changed" }
        requireGenerationManifestContract(directory, manifest)

        val database = File(directory, manifest.databaseRelativePath)
        require(database.isFile && database.length() == manifest.databaseByteLength) {
            "generation database file binding mismatch"
        }
        val databaseSha256 = V2FileSha256.digest(database) { completedBytes, totalBytes ->
            onArtifactHashProgress(
                V2GenerationArtifactHashProgress(
                    database.name,
                    completedBytes,
                    totalBytes,
                ),
            )
        }
        require(databaseSha256 == manifest.databaseSha256) {
            "generation database file binding mismatch"
        }
        val embedding = File(directory, manifest.embeddingRelativePath)
        val binding = when (validation) {
            V2GenerationArtifactValidation.FULL_SEMANTIC ->
                V2EmbeddingGenerationFile.inspect(embedding)
            V2GenerationArtifactValidation.PUBLISHED_BYTE_EXACT -> {
                require(
                    embedding.isFile &&
                        embedding.length() == manifest.embeddingByteLength &&
                        V2FileSha256.digest(embedding) { completedBytes, totalBytes ->
                            onArtifactHashProgress(
                                V2GenerationArtifactHashProgress(
                                    embedding.name,
                                    completedBytes,
                                    totalBytes,
                                ),
                            )
                        } == manifest.embeddingSha256,
                ) { "generation PEMB byte binding mismatch" }
                V2OrderedEmbeddingBinding(
                    trackCount = manifest.trackCount,
                    dimension = manifest.embeddingDimension,
                    byteLength = manifest.embeddingByteLength,
                    fileSha256 = manifest.embeddingSha256,
                    orderedTrackSetSha256 = manifest.orderedTrackSetSha256,
                    databaseContentSha256 = manifest.databaseContentSha256,
                )
            }
        }
        require(binding.trackCount == manifest.trackCount &&
            binding.dimension == manifest.embeddingDimension &&
            binding.byteLength == manifest.embeddingByteLength &&
            binding.fileSha256 == manifest.embeddingSha256 &&
            binding.orderedTrackSetSha256 == manifest.orderedTrackSetSha256 &&
            binding.databaseContentSha256 == manifest.databaseContentSha256
        ) { "generation PEMB binding mismatch" }
        val expectedGraphPresent = manifest.graphPolicy != V2IndexGenerationGraphPolicy.ABSENT
        require(
            manifest.rebuildDerivedIndexes ==
                (manifest.graphPolicy == V2IndexGenerationGraphPolicy.EXPLICIT_REBUILD) &&
                expectedGraphPresent == (manifest.graph != null)
        ) { "generation graph policy mismatch" }
        val graphFile = manifest.graph?.let { graph ->
            require(graph.relativePath == V2GraphGenerationFile.GRAPH_FILE) {
                "generation graph path is unsupported"
            }
            File(directory, graph.relativePath).also { file ->
                val actual = when (validation) {
                    V2GenerationArtifactValidation.FULL_SEMANTIC ->
                        V2GraphGenerationFile.inspect(file)
                    V2GenerationArtifactValidation.PUBLISHED_BYTE_EXACT -> {
                        require(
                            file.isFile && file.length() == graph.byteLength &&
                                V2FileSha256.digest(file) { completedBytes, totalBytes ->
                                    onArtifactHashProgress(
                                        V2GenerationArtifactHashProgress(
                                            file.name,
                                            completedBytes,
                                            totalBytes,
                                        ),
                                    )
                                } == graph.sha256,
                        ) { "generation graph byte binding mismatch" }
                        graph
                    }
                }
                require(actual == graph &&
                    graph.nodeCount == manifest.trackCount &&
                    graph.orderedTrackSetSha256 == manifest.orderedTrackSetSha256
                ) { "generation graph binding mismatch" }
            }
        }
        if (graphFile == null) {
            require(!File(directory, V2GraphGenerationFile.GRAPH_FILE).exists()) {
                "graph asset exists despite ABSENT graph policy"
            }
        }
        return V2ResolvedActiveIndexGeneration(
            manifest = manifest,
            manifestSha256 = manifestSha256,
            directory = directory,
            databaseFile = database,
            embeddingFile = embedding,
            graphFile = graphFile,
        )
    }

    internal fun requireDatabaseCoherence(
        directory: File,
        manifest: V2IndexGenerationManifest,
    ) {
        val databaseFile = File(directory, manifest.databaseRelativePath)
        SQLiteDatabase.openDatabase(databaseFile.path, null, SQLiteDatabase.OPEN_READONLY).use { db ->
            val databaseBinding = V2EmbeddingGenerationFile.digest(
                V2SqliteOrderedEmbeddingSource(db),
            )
            require(databaseBinding.trackCount == manifest.trackCount &&
                databaseBinding.dimension == manifest.embeddingDimension &&
                databaseBinding.databaseContentSha256 == manifest.databaseContentSha256 &&
                databaseBinding.orderedTrackSetSha256 == manifest.orderedTrackSetSha256
            ) { "generation database semantic binding mismatch" }
            require(
                V2StableTrackUidCoverage.inspect(db, databaseBinding.trackCount) ==
                    manifest.stableTrackUidCoverage,
            ) { "generation stable UID coverage changed" }
            require(
                V2EmbeddingSpecCoverage.inspect(
                    database = db,
                    embeddingCount = databaseBinding.trackCount,
                    expectedReceiptSpecId = manifest.receiptEmbeddingSpec.specId,
                ) == manifest.embeddingCoverage,
            ) { "generation embedding-spec coverage changed" }
            requireValidReceipt(db, manifest)
        }
    }

    internal fun requireValidReceipt(databaseFile: File, manifest: V2IndexGenerationManifest) {
        SQLiteDatabase.openDatabase(databaseFile.path, null, SQLiteDatabase.OPEN_READONLY).use { db ->
            requireValidReceipt(db, manifest)
        }
    }

    private fun requireValidReceipt(database: SQLiteDatabase, manifest: V2IndexGenerationManifest) {
        val valid = database.rawQuery(
            """
            SELECT receipt_schema_version, is_valid, activation_binding_id, job_spec_id,
                   receipt_embedding_spec_id, text_retrieval_spec_id,
                   embedding_coverage_sha256, compatibility_base_content_sha256,
                   database_content_sha256, ordered_track_set_sha256,
                   stable_uid_mapping_sha256, embedding_sha256, graph_sha256
            FROM $ACTIVATION_RECEIPT_TABLE WHERE singleton = 1
            """.trimIndent(),
            null,
        ).use { cursor ->
            cursor.moveToFirst() &&
                cursor.getInt(0) == ACTIVATION_RECEIPT_SCHEMA_VERSION &&
                cursor.getInt(1) == 1 &&
                cursor.getString(2) == manifest.activationBindingId &&
                cursor.getString(3) == manifest.jobSpecId &&
                cursor.getString(4) == manifest.receiptEmbeddingSpec.specId &&
                cursor.getString(5) == manifest.textRetrievalSpec.specId &&
                cursor.getString(6) == manifest.embeddingCoverage.mappingSha256 &&
                (if (cursor.isNull(7)) null else cursor.getString(7)) ==
                manifest.embeddingCoverage.compatibilityBase?.orderedContentSha256 &&
                cursor.getString(8) == manifest.databaseContentSha256 &&
                cursor.getString(9) == manifest.orderedTrackSetSha256 &&
                cursor.getString(10) == manifest.stableTrackUidCoverage.mappingSha256 &&
                cursor.getString(11) == manifest.embeddingSha256 &&
                (if (cursor.isNull(12)) null else cursor.getString(12)) == manifest.graph?.sha256
        }
        require(valid) { "generation database receipt is absent, invalidated, or mismatched" }
    }

    private data class CachedGeneration(
        val pointer: V2ActiveGenerationPointer,
        val signature: GenerationFileSignature,
        val resolved: V2ResolvedActiveIndexGeneration,
    )

    private data class GenerationFileSignature(
        val manifest: FileStat,
        val database: FileStat,
        val embedding: FileStat,
        val graph: FileStat?,
    ) {
        companion object {
            fun capture(directory: File): GenerationFileSignature = GenerationFileSignature(
                manifest = FileStat.capture(File(directory, MANIFEST_FILE)),
                database = FileStat.capture(File(directory, DATABASE_FILE)),
                embedding = FileStat.capture(File(directory, EMBEDDING_FILE)),
                graph = File(directory, V2GraphGenerationFile.GRAPH_FILE)
                    .takeIf(File::exists)?.let(FileStat::capture),
            )
        }
    }

    private data class FileStat(val canonicalPath: String, val byteLength: Long, val modifiedMs: Long) {
        companion object {
            fun capture(file: File): FileStat {
                require(file.isFile) { "generation asset is missing: $file" }
                return FileStat(file.canonicalPath, file.length(), file.lastModified())
            }
        }
    }
}

internal enum class V2GenerationArtifactValidation {
    /** Required before a generation can ever become pointer-visible. */
    FULL_SEMANTIC,

    /** Exact hashes attest artifacts which FULL_SEMANTIC already admitted before publication. */
    PUBLISHED_BYTE_EXACT,
}

object V2IndexGenerationIdentity {
    fun activationBindingId(
        ledger: IndexingJobLedger,
        embeddings: V2OrderedEmbeddingBinding,
        stableCoverage: V2StableTrackUidCoverageBinding,
        embeddingCoverage: V2EmbeddingSpecCoverageBinding,
        graphPolicy: V2IndexGenerationGraphPolicy,
        graph: V2IndexGenerationGraphBinding?,
    ): String = activationBindingId(
        origin = V2IndexGenerationOrigin.INDEXING_JOB,
        jobSpecId = ledger.jobSpec.specId,
        receiptEmbeddingSpec = ledger.jobSpec.embeddingSpec,
        textRetrievalSpec = ledger.jobSpec.textRetrievalSpec,
        baseGenerationId = ledger.jobSpec.baseGenerationId,
        rebuildDerivedIndexes = ledger.jobSpec.rebuildDerivedIndexes,
        embeddings = embeddings,
        stableCoverage = stableCoverage,
        embeddingCoverage = embeddingCoverage,
        graphPolicy = graphPolicy,
        graph = graph,
    )

    fun activationBindingId(manifest: V2IndexGenerationManifest): String = activationBindingId(
        origin = manifest.origin,
        jobSpecId = manifest.jobSpecId,
        receiptEmbeddingSpec = manifest.receiptEmbeddingSpec,
        textRetrievalSpec = manifest.textRetrievalSpec,
        baseGenerationId = manifest.baseGenerationId,
        rebuildDerivedIndexes = manifest.rebuildDerivedIndexes,
        embeddings = V2OrderedEmbeddingBinding(
            trackCount = manifest.trackCount,
            dimension = manifest.embeddingDimension,
            byteLength = manifest.embeddingByteLength,
            fileSha256 = manifest.embeddingSha256,
            orderedTrackSetSha256 = manifest.orderedTrackSetSha256,
            databaseContentSha256 = manifest.databaseContentSha256,
        ),
        stableCoverage = manifest.stableTrackUidCoverage,
        embeddingCoverage = manifest.embeddingCoverage,
        graphPolicy = manifest.graphPolicy,
        graph = manifest.graph,
    )

    internal fun activationBindingId(
        origin: V2IndexGenerationOrigin,
        jobSpecId: String,
        receiptEmbeddingSpec: EmbeddingSpecFingerprint,
        textRetrievalSpec: TextRetrievalSpecFingerprint,
        baseGenerationId: String?,
        rebuildDerivedIndexes: Boolean,
        embeddings: V2OrderedEmbeddingBinding,
        stableCoverage: V2StableTrackUidCoverageBinding,
        embeddingCoverage: V2EmbeddingSpecCoverageBinding,
        graphPolicy: V2IndexGenerationGraphPolicy,
        graph: V2IndexGenerationGraphBinding?,
    ): String {
        val digest = MessageDigest.getInstance("SHA-256")
        digest.updateLengthPrefixed("v2-index-activation-binding-v3")
        digest.updateLengthPrefixed(origin.name)
        digest.updateLengthPrefixed(jobSpecId)
        digest.updateLengthPrefixed(receiptEmbeddingSpec.specId)
        digest.updateLengthPrefixed(textRetrievalSpec.specId)
        digest.updateLengthPrefixed(receiptEmbeddingSpec.preprocessingSpecId)
        digest.updateLengthPrefixed(receiptEmbeddingSpec.decoderPolicyId)
        digest.updateLengthPrefixed(receiptEmbeddingSpec.inferenceBackendPolicyId)
        receiptEmbeddingSpec.modelArtifactSha256.toSortedMap().forEach { (name, sha) ->
            digest.updateLengthPrefixed(name)
            digest.updateLengthPrefixed(sha)
        }
        digest.updateNullableString(baseGenerationId)
        digest.update(if (rebuildDerivedIndexes) 1.toByte() else 0.toByte())
        digest.updateLengthPrefixed(graphPolicy.name)
        digest.updateLengthPrefixed(embeddings.databaseContentSha256)
        digest.updateLengthPrefixed(embeddings.orderedTrackSetSha256)
        digest.updateLengthPrefixed(embeddings.fileSha256)
        digest.updateLengthPrefixed(stableCoverage.mappingSha256)
        digest.updateLengthPrefixed(embeddingCoverage.mappingSha256)
        digest.updateNullableString(graph?.sha256)
        return "activation-binding-v3-${digest.digest().toV2CommitHex()}"
    }

    fun generationId(manifest: V2IndexGenerationManifest): String {
        val digest = MessageDigest.getInstance("SHA-256")
        digest.updateLengthPrefixed("v2-index-generation-manifest-v3")
        digest.updateLengthPrefixed(manifest.origin.name)
        digest.updateLengthPrefixed(manifest.activationBindingId)
        digest.updateLengthPrefixed(manifest.jobId)
        digest.updateLengthPrefixed(manifest.jobSpecId)
        digest.updateLengthPrefixed(manifest.receiptEmbeddingSpec.specId)
        digest.updateLengthPrefixed(manifest.textRetrievalSpec.specId)
        digest.updateNullableString(manifest.baseGenerationId)
        digest.update(if (manifest.rebuildDerivedIndexes) 1.toByte() else 0.toByte())
        digest.updateLengthPrefixed(manifest.graphPolicy.name)
        digest.updateLengthPrefixed(manifest.databaseRelativePath)
        digest.updateLong(manifest.databaseByteLength)
        digest.updateLengthPrefixed(manifest.databaseSha256)
        digest.updateLengthPrefixed(manifest.databaseContentSha256)
        digest.updateLengthPrefixed(manifest.orderedTrackSetSha256)
        val coverage = manifest.stableTrackUidCoverage
        digest.updateInt(coverage.coveredTrackCount)
        digest.updateInt(coverage.uncoveredTrackCount)
        digest.updateInt(coverage.uniqueStableTrackSpanCount)
        digest.updateInt(coverage.fullContentIdentityCount)
        digest.updateInt(coverage.sampledContentIdentityCount)
        digest.updateLengthPrefixed(coverage.mappingSha256)
        val embeddingCoverage = manifest.embeddingCoverage
        digest.updateInt(embeddingCoverage.totalTrackCount)
        digest.updateInt(embeddingCoverage.receiptBoundTrackCount)
        digest.updateInt(embeddingCoverage.receiptSpecTrackCounts.size)
        embeddingCoverage.receiptSpecTrackCounts.toSortedMap().forEach { (specId, count) ->
            digest.updateLengthPrefixed(specId)
            digest.updateInt(count)
        }
        val compatibilityBase = embeddingCoverage.compatibilityBase
        digest.update(if (compatibilityBase == null) 0.toByte() else 1.toByte())
        if (compatibilityBase != null) {
            digest.updateLengthPrefixed(compatibilityBase.provenancePolicyId)
            digest.updateInt(compatibilityBase.trackCount)
            digest.updateLengthPrefixed(compatibilityBase.orderedContentSha256)
        }
        digest.updateLengthPrefixed(embeddingCoverage.mappingSha256)
        digest.updateInt(manifest.trackCount)
        digest.updateInt(manifest.embeddingDimension)
        digest.updateLengthPrefixed(manifest.embeddingRelativePath)
        digest.updateLong(manifest.embeddingByteLength)
        digest.updateLengthPrefixed(manifest.embeddingSha256)
        val graph = manifest.graph
        digest.update(if (graph == null) 0.toByte() else 1.toByte())
        if (graph != null) {
            digest.updateLengthPrefixed(graph.relativePath)
            digest.updateLong(graph.byteLength)
            digest.updateLengthPrefixed(graph.sha256)
            digest.updateInt(graph.nodeCount)
            digest.updateInt(graph.neighborsPerNode)
            digest.updateLengthPrefixed(graph.orderedTrackSetSha256)
        }
        return "index-generation-v2-${digest.digest().toV2CommitHex()}"
    }

    fun bootstrapSpecId(
        receiptEmbeddingSpec: EmbeddingSpecFingerprint,
        textRetrievalSpec: TextRetrievalSpecFingerprint,
    ): String {
        val digest = MessageDigest.getInstance("SHA-256")
        digest.updateLengthPrefixed("v2-bootstrap-compatibility-spec-v1")
        digest.updateLengthPrefixed(receiptEmbeddingSpec.specId)
        digest.updateLengthPrefixed(textRetrievalSpec.specId)
        return "bootstrap-spec-v1-${digest.digest().toV2CommitHex()}"
    }

    fun maintenanceSpecId(
        baseGenerationId: String,
        receiptEmbeddingSpec: EmbeddingSpecFingerprint,
        textRetrievalSpec: TextRetrievalSpecFingerprint,
    ): String {
        require(baseGenerationId.matches(GENERATION_ID)) { "invalid maintenance base generation" }
        val digest = MessageDigest.getInstance("SHA-256")
        digest.updateLengthPrefixed("v2-library-maintenance-spec-v1")
        digest.updateLengthPrefixed(baseGenerationId)
        digest.updateLengthPrefixed(receiptEmbeddingSpec.specId)
        digest.updateLengthPrefixed(textRetrievalSpec.specId)
        return "maintenance-spec-v1-${digest.digest().toV2CommitHex()}"
    }

    fun serverMergeSpecId(
        baseGenerationId: String,
        bundleDatabaseSha256: String,
        receiptEmbeddingSpec: EmbeddingSpecFingerprint,
        textRetrievalSpec: TextRetrievalSpecFingerprint,
    ): String {
        require(baseGenerationId.matches(GENERATION_ID)) { "invalid server merge base generation" }
        require(bundleDatabaseSha256.matches(SHA256)) { "invalid server bundle database hash" }
        val digest = MessageDigest.getInstance("SHA-256")
        digest.updateLengthPrefixed("v2-server-index-merge-spec-v1")
        digest.updateLengthPrefixed(baseGenerationId)
        digest.updateLengthPrefixed(bundleDatabaseSha256)
        digest.updateLengthPrefixed(receiptEmbeddingSpec.specId)
        digest.updateLengthPrefixed(textRetrievalSpec.specId)
        return "server-merge-spec-v1-${digest.digest().toV2CommitHex()}"
    }
}

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

private fun moveGenerationDirectory(source: File, destination: File) {
    try {
        Files.move(source.toPath(), destination.toPath(), StandardCopyOption.ATOMIC_MOVE)
    } catch (_: AtomicMoveNotSupportedException) {
        Files.move(source.toPath(), destination.toPath())
    }
}

private fun RandomAccessFile.writeLongLittleEndian(value: Long) {
    write(ByteBuffer.allocate(Long.SIZE_BYTES).order(ByteOrder.LITTLE_ENDIAN).putLong(value).array())
}


private fun MessageDigest.updateLengthPrefixed(value: String) {
    val bytes = value.toByteArray(StandardCharsets.UTF_8)
    updateInt(bytes.size)
    update(bytes)
}

private fun MessageDigest.updateNullableString(value: String?) {
    update(if (value == null) 0.toByte() else 1.toByte())
    if (value != null) updateLengthPrefixed(value)
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

private const val GENERATIONS_DIRECTORY = "indexing_v2/generations"
private const val GENERATION_STAGING_PREFIX = ".staging-"
private const val ACTIVE_POINTER_FILE = "active-generation.json"
private const val MANIFEST_FILE = "manifest.json"
private const val DATABASE_FILE = "library.db"
private const val EMBEDDING_FILE = "clamp3.emb"
private const val ACTIVATION_RECEIPT_TABLE = "v2_index_generation_guard_v2"
private const val ACTIVATION_RECEIPT_SCHEMA_VERSION = 3
private const val MANIFEST_SCHEMA_VERSION = 3
private const val POINTER_SCHEMA_VERSION = 2
private val GENERATION_ID = Regex("^index-generation-v2-[0-9a-f]{64}$")
private val SHA256 = Regex("^[0-9a-f]{64}$")
private val SERVER_BUNDLE_ID = Regex("^server-bundle-v1-[0-9a-f]{64}$")
private val SERVER_ROOT_ID = Regex("^[A-Za-z0-9._-]{1,128}$")
