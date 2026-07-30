package com.powerampstartradio.indexing.v2

import java.io.File
import java.io.FileInputStream
import java.io.RandomAccessFile
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.attribute.BasicFileAttributes
import java.security.MessageDigest

/** Semantic work policy used by preflight; changing any constant requires a new spec ID. */
object V2IndexingWorkPolicy {
    const val METADATA_NORMALIZATION_SPEC_ID = "poweramp-track-normalization-v1"
    const val PREPROCESSING_SPEC_ID =
        "mert-clamp3-audio-v3:torchaudio-hann-v1-width6-rolloff0.99-f32-target-length:" +
            "pcm24k-whole-span-zmuv:5s-window:1s-tail-zero-pad:" +
            "zero-bookends:segment128-final-overlap:frame-weighted-average:l2"
    const val DECODER_POLICY_ID =
        "android-mediacodec-v3:resolved-half-open-us-native-sample-span:" +
            "verify-eos-or-enforce-cue-boundary:aligned-polyphase-hq:canonical-24khz-pcm"
    const val INFERENCE_BACKEND_POLICY_ID =
        "litert-2.1.1-compiled-model-v1:" +
            "mert-gpu-fp32-strict:clamp3-audio-gpu-fp32-strict:" +
            "no-backend-fallback"
    const val TEXT_TOKENIZER_POLICY_ID =
        "sentencepiece-v0.2.1-rev-31646a467d2051eb904e0b45de3a73e91fe1c1e3-" +
            "xlm-roberta-model-native-encode-sp-unk0-to-3-else-plus1-" +
            "bos0-eos2-pad1-seq128-v1"
    const val TEXT_TOKENIZER_MODEL_SHA256 =
        "cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865"
    const val TEXT_TOKENIZER_RUNTIME_CONTRACT_SHA256 =
        "e3f1abde1d51a6747a252f99b276359f1353b3637e39f85670e8189baa65d8f3"
    const val TEXT_OUTPUT_SPACE_ID = "clamp3-joint-audio-text-l2-f32-v1"
    const val TEXT_INFERENCE_BACKEND_POLICY_ID =
        "litert-2.1.1-compiled-model-v1:clamp3-text-cpu-strict:" +
            "host-text-aggregation-v1:segment128-final-overlap:" +
            "token-count-weighted-average:l2:no-backend-fallback"
}

data class V2ByteRegion(
    val offset: Long,
    val length: Long,
) {
    val endExclusive: Long get() = Math.addExact(offset, length)
}

/** Fixed first/middle/last regions with overlap merged so bytes are never sampled twice. */
object V2FixedRegionSampling {
    const val REGION_BYTES = 64 * 1024
    const val SPEC_ID =
        "source-sample-sha256-v1:first-middle-last:region-65536:" +
            "merge-overlap:bind-size-offset-length"

    fun regions(fileSize: Long, regionBytes: Int = REGION_BYTES): List<V2ByteRegion> {
        require(fileSize > 0L) { "source file must not be empty" }
        require(regionBytes > 0) { "region size must be positive" }
        val length = minOf(fileSize, regionBytes.toLong())
        val finalStart = fileSize - length
        val candidates = listOf(
            0L,
            maxOf(0L, (fileSize - length) / 2L),
            finalStart,
        ).distinct().sorted().map { start -> V2ByteRegion(start, length) }

        val merged = mutableListOf<V2ByteRegion>()
        for (candidate in candidates) {
            val previous = merged.lastOrNull()
            if (previous == null || candidate.offset > previous.endExclusive) {
                merged += candidate
            } else {
                val mergedEnd = maxOf(previous.endExclusive, candidate.endExclusive)
                merged[merged.lastIndex] = V2ByteRegion(
                    offset = previous.offset,
                    length = mergedEnd - previous.offset,
                )
            }
        }
        return merged
    }
}

fun interface V2SourceFingerprintProvider {
    fun fingerprint(sourceFile: File): SourceFingerprint

    fun fingerprint(
        sourceFile: File,
        onHashProgress: (completedBytes: Long, totalBytes: Long) -> Unit,
    ): SourceFingerprint = fingerprint(sourceFile).also { fingerprint ->
        onHashProgress(fingerprint.sizeBytes, fingerprint.sizeBytes)
    }
}

/**
 * Exact source identity for any cross-path reuse or duplicate collapse.
 *
 * The full read is cached once per physical source by [V2DeduplicatingSourceFingerprintProvider].
 * This intentionally spends sequential I/O before inference instead of treating a sample as proof
 * that two files are byte-identical.
 */
class V2ExactSourceFingerprinter : V2SourceFingerprintProvider {
    override fun fingerprint(sourceFile: File): SourceFingerprint = fingerprint(sourceFile) { _, _ -> }

    override fun fingerprint(
        sourceFile: File,
        onHashProgress: (completedBytes: Long, totalBytes: Long) -> Unit,
    ): SourceFingerprint {
        val file = sourceFile.canonicalFile
        require(file.isFile && file.canRead()) { "source is not a readable file: $file" }

        val sizeBefore = file.length()
        val modifiedBefore = file.lastModified()
        require(sizeBefore > 0L) { "source file must not be empty: $file" }
        val fullContentSha256 = V2FileSha256.digest(file, onHashProgress)
        val fileKey = try {
            Files.readAttributes(file.toPath(), BasicFileAttributes::class.java).fileKey()?.toString()
        } catch (_: Exception) {
            null
        }
        check(file.length() == sizeBefore && file.lastModified() == modifiedBefore) {
            "source changed during fingerprinting: $file"
        }
        return SourceFingerprint(
            fingerprintSpecId = V2IndexingLedgerIds.FULL_CONTENT_FINGERPRINT_SPEC_ID,
            sizeBytes = sizeBefore,
            lastModifiedEpochMs = modifiedBefore,
            fileKey = fileKey,
            sampledContentSha256 = null,
            fullContentSha256 = fullContentSha256,
        )
    }
}

/** Kept source-compatible for older tests/callers; production semantics are now exact. */
@Deprecated("Use V2ExactSourceFingerprinter")
class V2FastSourceFingerprinter : V2SourceFingerprintProvider by V2ExactSourceFingerprinter()

/** Reuses one physical-source fingerprint across logical CUE rows in the same preflight run. */
class V2DeduplicatingSourceFingerprintProvider(
    private val delegate: V2SourceFingerprintProvider,
) : V2SourceFingerprintProvider {
    private data class SourceKey(
        val canonicalPath: String,
        val sizeBytes: Long,
        val lastModifiedEpochMs: Long,
    )

    private val cache = mutableMapOf<SourceKey, SourceFingerprint>()

    override fun fingerprint(sourceFile: File): SourceFingerprint = fingerprint(sourceFile) { _, _ -> }

    @Synchronized
    override fun fingerprint(
        sourceFile: File,
        onHashProgress: (completedBytes: Long, totalBytes: Long) -> Unit,
    ): SourceFingerprint {
        val file = sourceFile.canonicalFile
        val key = SourceKey(
            canonicalPath = file.path,
            sizeBytes = file.length(),
            lastModifiedEpochMs = file.lastModified(),
        )
        val cached = cache[key]
        if (cached != null) {
            onHashProgress(cached.sizeBytes, cached.sizeBytes)
            return cached
        }
        return delegate.fingerprint(file, onHashProgress).also { cache[key] = it }
    }
}

/** Full artifact hashes for immutable content identities and changed installed models. */
object V2FileSha256 {
    private const val PROGRESS_INTERVAL_BYTES = 8L * 1024L * 1024L

    fun digest(file: File): String = digest(file) { _, _ -> }

    fun digest(
        file: File,
        onProgress: (completedBytes: Long, totalBytes: Long) -> Unit,
    ): String {
        require(file.isFile && file.canRead()) { "artifact is not a readable file: $file" }
        val sizeBefore = file.length()
        val modifiedBefore = file.lastModified()
        val digest = MessageDigest.getInstance("SHA-256")
        val buffer = ByteArray(1024 * 1024)
        var completedBytes = 0L
        var lastReportedBytes = 0L
        onProgress(completedBytes, sizeBefore)
        FileInputStream(file).use { input ->
            while (true) {
                val read = input.read(buffer)
                if (read < 0) break
                if (read > 0) {
                    digest.update(buffer, 0, read)
                    completedBytes += read
                    if (completedBytes == sizeBefore ||
                        completedBytes - lastReportedBytes >= PROGRESS_INTERVAL_BYTES
                    ) {
                        onProgress(completedBytes, sizeBefore)
                        lastReportedBytes = completedBytes
                    }
                }
            }
        }
        if (completedBytes != lastReportedBytes) onProgress(completedBytes, sizeBefore)
        check(file.length() == sizeBefore && file.lastModified() == modifiedBefore) {
            "artifact changed during SHA-256: $file"
        }
        return digest.digest().toHex()
    }

    fun digestText(value: String): String = MessageDigest.getInstance("SHA-256")
        .digest(value.toByteArray(StandardCharsets.UTF_8))
        .toHex()
}

private fun MessageDigest.updateLengthPrefixed(value: String) {
    val bytes = value.toByteArray(StandardCharsets.UTF_8)
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

private fun ByteArray.toHex(): String = joinToString("") { byte ->
    "%02x".format(byte.toInt() and 0xff)
}
