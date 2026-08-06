package com.powerampstartradio.indexing.v2

import android.util.AtomicFile
import com.google.gson.Gson
import com.google.gson.GsonBuilder
import java.io.File
import java.io.IOException
import java.io.OutputStreamWriter
import java.nio.charset.StandardCharsets

data class V2InstalledModelHashProgress(
    val filename: String,
    val fileOrdinal: Int,
    val fileCount: Int,
    val completedBytes: Long,
    val totalBytes: Long,
)

data class V2ResolvedInstalledModelPolicy(
    val policy: V2FutureModelPolicy,
    val filesByName: Map<String, File>,
    val sha256ByName: Map<String, String>,
)

/**
 * Persists exact model hashes and reuses them while the app-private files retain the same stat
 * identity. Model replacement is exceptional; ordinary Settings visits and indexing resumes must
 * not reread the full model set.
 */
internal object V2InstalledModelReceiptStore {
    private const val SCHEMA_VERSION = 1
    private const val RECEIPT_RELATIVE_PATH = "indexing_v2/installed-model-receipt-v1.json"
    private val REQUIRED_FILENAMES = listOf(
        "mert.tflite",
        "clamp3_audio.tflite",
        "clamp3_text.tflite",
        "sentencepiece.bpe.model",
    )
    private val SHA256 = Regex("^[0-9a-f]{64}$")
    private val gson: Gson = GsonBuilder().disableHtmlEscaping().create()

    @Volatile
    private var cached: CachedResolution? = null

    @Synchronized
    fun resolve(
        filesDir: File,
        onHashProgress: (V2InstalledModelHashProgress) -> Unit = {},
    ): V2ResolvedInstalledModelPolicy {
        val root = filesDir.canonicalFile
        val files = REQUIRED_FILENAMES.associateWith { filename ->
            requireArtifact(root, filename)
        }
        val signatures = REQUIRED_FILENAMES.map { filename ->
            FileSignature.capture(files.getValue(filename))
        }
        cached?.takeIf { it.rootPath == root.path && it.signatures == signatures }
            ?.let { return it.resolution }

        val receiptFile = AtomicFile(File(root, RECEIPT_RELATIVE_PATH))
        val persisted = readReceipt(receiptFile)
            ?.takeIf { it.schemaVersion == SCHEMA_VERSION && it.rootPath == root.path }
        val persistedByName = runCatching {
            persisted?.artifacts
                ?.takeIf { artifacts ->
                    artifacts.map(ArtifactReceipt::filename) == REQUIRED_FILENAMES
                }
                ?.associateBy(ArtifactReceipt::filename)
                .orEmpty()
        }.getOrDefault(emptyMap())
        val reusableHashes = signatures.mapNotNull { signature ->
            persistedByName[signature.filename]
                ?.takeIf { it.matches(signature) && SHA256.matches(it.sha256) }
                ?.let { signature.filename to it.sha256 }
        }.toMap()
        val signaturesToHash = signatures.filter { it.filename !in reusableHashes }
        val totalHashBytes = signaturesToHash.sumOf(FileSignature::byteLength)
        var completedBeforeFile = 0L
        val hashes = linkedMapOf<String, String>()
        signatures.forEachIndexed { index, signature ->
            val file = files.getValue(signature.filename)
            val reused = reusableHashes[signature.filename]
            val sha256 = reused ?: V2FileSha256.digest(file) { completedInFile, _ ->
                onHashProgress(
                    V2InstalledModelHashProgress(
                        filename = signature.filename,
                        fileOrdinal = index + 1,
                        fileCount = REQUIRED_FILENAMES.size,
                        completedBytes = completedBeforeFile + completedInFile,
                        totalBytes = totalHashBytes,
                    ),
                )
            }
            hashes[signature.filename] = sha256
            if (reused == null) completedBeforeFile += signature.byteLength
        }
        require(hashes.getValue("sentencepiece.bpe.model") ==
            V2IndexingWorkPolicy.TEXT_TOKENIZER_MODEL_SHA256
        ) { "sentencepiece.bpe.model is not the exact CLaMP3 tokenizer model" }

        val policy = createPolicy(hashes)
        val receipt = InstalledModelReceipt(
            schemaVersion = SCHEMA_VERSION,
            rootPath = root.path,
            artifacts = signatures.map { signature ->
                ArtifactReceipt(
                    filename = signature.filename,
                    canonicalPath = signature.canonicalPath,
                    byteLength = signature.byteLength,
                    modifiedEpochMs = signature.modifiedEpochMs,
                    sha256 = hashes.getValue(signature.filename),
                )
            },
        )
        if (persisted != receipt) writeReceipt(receiptFile, receipt)

        return V2ResolvedInstalledModelPolicy(
            policy = policy,
            filesByName = files,
            sha256ByName = hashes,
        ).also { resolution ->
            cached = CachedResolution(root.path, signatures, resolution)
        }
    }

    private fun createPolicy(hashes: Map<String, String>): V2FutureModelPolicy {
        val embeddingSpec = V2IndexingLedgerPlanner.createEmbeddingSpec(
            EmbeddingSpecInput(
                preprocessingSpecId = V2IndexingWorkPolicy.PREPROCESSING_SPEC_ID,
                decoderPolicyId = V2IndexingWorkPolicy.DECODER_POLICY_ID,
                inferenceBackendPolicyId = V2IndexingWorkPolicy.INFERENCE_BACKEND_POLICY_ID,
                outputDimension = V2_CLAMP3_DIMENSION,
                modelArtifactSha256 = mapOf(
                    "mert" to hashes.getValue("mert.tflite"),
                    "clamp3_audio" to hashes.getValue("clamp3_audio.tflite"),
                ),
            ),
        )
        val textSpec = V2IndexingLedgerPlanner.createTextRetrievalSpec(
            TextRetrievalSpecInput(
                compatibleAudioEmbeddingSpecId = embeddingSpec.specId,
                textModelSha256 = hashes.getValue("clamp3_text.tflite"),
                tokenizerModelSha256 = hashes.getValue("sentencepiece.bpe.model"),
                tokenizerPolicyId = V2IndexingWorkPolicy.TEXT_TOKENIZER_POLICY_ID,
                tokenizerRuntimeContractSha256 =
                    V2IndexingWorkPolicy.TEXT_TOKENIZER_RUNTIME_CONTRACT_SHA256,
                outputSpaceId = V2IndexingWorkPolicy.TEXT_OUTPUT_SPACE_ID,
                outputDimension = V2_CLAMP3_DIMENSION,
                inferenceBackendPolicyId = V2IndexingWorkPolicy.TEXT_INFERENCE_BACKEND_POLICY_ID,
            ),
        )
        V2IndexingLedgerValidator.requireValidEmbeddingSpec(embeddingSpec)
        V2IndexingLedgerValidator.requireValidTextRetrievalSpec(textSpec)
        return V2FutureModelPolicy(embeddingSpec, textSpec)
    }

    private fun requireArtifact(root: File, filename: String): File {
        val file = File(root, filename).canonicalFile
        require(file.parentFile == root && file.isFile && file.canRead() && file.length() > 0L) {
            "Required exact model artifact is missing or unreadable: $filename"
        }
        return file
    }

    private fun readReceipt(file: AtomicFile): InstalledModelReceipt? {
        if (!file.baseFile.exists() && !File(file.baseFile.path + ".bak").exists()) return null
        return runCatching {
            file.openRead().bufferedReader(StandardCharsets.UTF_8).use { reader ->
                gson.fromJson(reader, InstalledModelReceipt::class.java)
            }
        }.getOrNull()
    }

    private fun writeReceipt(file: AtomicFile, receipt: InstalledModelReceipt) {
        file.baseFile.parentFile?.let { parent ->
            if ((!parent.exists() && !parent.mkdirs()) || !parent.isDirectory) {
                throw IOException("unable to create installed-model receipt directory: $parent")
            }
        }
        val stream = file.startWrite()
        try {
            val writer = OutputStreamWriter(stream, StandardCharsets.UTF_8)
            gson.toJson(receipt, writer)
            writer.flush()
            file.finishWrite(stream)
        } catch (error: Throwable) {
            file.failWrite(stream)
            throw IOException("unable to persist installed-model receipt", error)
        }
    }

    private data class CachedResolution(
        val rootPath: String,
        val signatures: List<FileSignature>,
        val resolution: V2ResolvedInstalledModelPolicy,
    )

    private data class FileSignature(
        val filename: String,
        val canonicalPath: String,
        val byteLength: Long,
        val modifiedEpochMs: Long,
    ) {
        companion object {
            fun capture(file: File) = FileSignature(
                filename = file.name,
                canonicalPath = file.canonicalPath,
                byteLength = file.length(),
                modifiedEpochMs = file.lastModified(),
            )
        }
    }

    private data class InstalledModelReceipt(
        val schemaVersion: Int,
        val rootPath: String,
        val artifacts: List<ArtifactReceipt>,
    )

    private data class ArtifactReceipt(
        val filename: String,
        val canonicalPath: String,
        val byteLength: Long,
        val modifiedEpochMs: Long,
        val sha256: String,
    ) {
        fun matches(signature: FileSignature): Boolean =
            filename == signature.filename &&
                canonicalPath == signature.canonicalPath &&
                byteLength == signature.byteLength &&
                modifiedEpochMs == signature.modifiedEpochMs
    }
}
