package com.powerampstartradio.indexing.v2

import android.util.AtomicFile
import com.google.gson.Gson
import com.google.gson.GsonBuilder
import com.powerampstartradio.indexing.MertInference
import com.powerampstartradio.indexing.TrackPcmCache
import java.io.File
import java.io.IOException
import java.io.OutputStreamWriter
import java.nio.charset.StandardCharsets
import java.util.concurrent.CancellationException

data class V2VerifiedPcmCache(
    val result: TrackPcmCache.Result,
    val receipt: V2VerifiedPcmCacheReceipt,
    val verifiedPcmByteLength: Long,
    val verifiedPcmLastModifiedEpochMs: Long,
)

data class V2VerifiedPcmCacheReceipt(
    val schemaVersion: Int,
    val jobId: String,
    val powerampFileId: Long,
    val provisionalWorkId: String,
    val providerSnapshotGeneration: String,
    val sourceFingerprint: SourceFingerprint,
    val pcmByteLength: Long,
    val pcmSha256: String,
    val sourceSampleRate: Int,
    val decoderName: String,
    val sourceChannelCount: Int,
    val sourcePcmEncoding: Int,
    val chunks: Int,
    val sourceStartSample: Long,
    val sourceEndSampleExclusive: Long,
    val sourceSampleCount: Long,
    val exactSampleCount24k: Long,
    val endOfStreamReached: Boolean,
    val logicalBoundaryEnforced: Boolean,
    val preprocessingSpecId: String,
    val normalizationSampleCount: Long,
    val normalizationMean: Float,
    val normalizationStandardDeviation: Float,
    val verifiedAtEpochMs: Long,
)

/** Crash-safe PCM plus receipt. Cold reuse hashes fully; same-run reuse checks the verified file. */
class V2VerifiedPcmCacheStore(
    private val gson: Gson = GsonBuilder().disableHtmlEscaping().create(),
) {
    fun pcmFile(jobArtifactDirectory: File, powerampFileId: Long): File =
        File(directory(jobArtifactDirectory), "$powerampFileId.pcm-24k-f32.bin")

    fun loadVerified(
        jobArtifactDirectory: File,
        jobId: String,
        descriptor: SelectedTrackDescriptor,
        onHashProgress: (completedBytes: Long, totalBytes: Long) -> Unit = { _, _ -> },
    ): V2VerifiedPcmCache? {
        val pcm = pcmFile(jobArtifactDirectory, descriptor.powerampFileId)
        val receiptBase = receiptBaseFile(jobArtifactDirectory, descriptor.powerampFileId)
        if (!pcm.isFile || !receiptBase.isFile) {
            deleteIncompletePair(pcm, receiptBase)
            return null
        }
        val receiptFile = AtomicFile(receiptBase)
        return try {
            val receipt = receiptFile.openRead().bufferedReader(StandardCharsets.UTF_8).use { reader ->
                gson.fromJson(reader, V2VerifiedPcmCacheReceipt::class.java)
            } ?: error("empty PCM receipt")
            requireVerified(
                pcm = pcm,
                receipt = receipt,
                jobId = jobId,
                descriptor = descriptor,
                onHashProgress = onHashProgress,
            )
        } catch (cancelled: CancellationException) {
            throw cancelled
        } catch (_: Exception) {
            delete(jobArtifactDirectory, descriptor.powerampFileId)
            null
        }
    }

    fun publish(
        jobArtifactDirectory: File,
        jobId: String,
        descriptor: SelectedTrackDescriptor,
        result: TrackPcmCache.Result,
        verifiedAtEpochMs: Long,
        onHashProgress: (completedBytes: Long, totalBytes: Long) -> Unit = { _, _ -> },
    ): V2VerifiedPcmCache {
        val expectedFile = pcmFile(jobArtifactDirectory, descriptor.powerampFileId)
        require(result.file.canonicalFile == expectedFile.canonicalFile) {
            "PCM cache was built outside its stable job scratch path"
        }
        val pcmSha256 = result.pcmSha256?.also { digest ->
            require(digest.matches(SHA256)) { "PCM builder returned an invalid SHA-256 digest" }
        } ?: V2FileSha256.digest(result.file, onHashProgress)
        val receipt = V2VerifiedPcmCacheReceipt(
            schemaVersion = SCHEMA_VERSION,
            jobId = jobId,
            powerampFileId = descriptor.powerampFileId,
            provisionalWorkId = descriptor.provisionalWorkId ?: descriptor.workId,
            providerSnapshotGeneration = descriptor.providerSnapshotGeneration,
            sourceFingerprint = descriptor.sourceFingerprint,
            pcmByteLength = result.file.length(),
            pcmSha256 = pcmSha256,
            sourceSampleRate = result.sourceSampleRate,
            decoderName = result.decoderName,
            sourceChannelCount = result.sourceChannelCount,
            sourcePcmEncoding = result.sourcePcmEncoding,
            chunks = result.chunks,
            sourceStartSample = result.sourceStartSample,
            sourceEndSampleExclusive = result.sourceEndSampleExclusive,
            sourceSampleCount = result.sourceSampleCount,
            exactSampleCount24k = result.exactSampleCount24k,
            endOfStreamReached = result.endOfStreamReached,
            logicalBoundaryEnforced = result.logicalBoundaryEnforced,
            preprocessingSpecId = result.preprocessingSpecId,
            normalizationSampleCount = result.normalization.sampleCount,
            normalizationMean = result.normalization.mean,
            normalizationStandardDeviation = result.normalization.standardDeviation,
            verifiedAtEpochMs = verifiedAtEpochMs,
        )
        writeReceipt(receiptFile(jobArtifactDirectory, descriptor.powerampFileId), receipt)
        return requireVerified(
            pcm = result.file,
            receipt = receipt,
            jobId = jobId,
            descriptor = descriptor,
            knownPcmSha256 = pcmSha256,
        )
    }

    fun requireVerified(
        jobArtifactDirectory: File,
        jobId: String,
        descriptor: SelectedTrackDescriptor,
        onHashProgress: (completedBytes: Long, totalBytes: Long) -> Unit = { _, _ -> },
    ): V2VerifiedPcmCache = loadVerified(
        jobArtifactDirectory,
        jobId,
        descriptor,
        onHashProgress,
    )
        ?: throw V2ArtifactChecksumException(
            "verified PCM cache is missing or no longer matches ${descriptor.workId}",
        )

    fun requireUnchanged(
        jobArtifactDirectory: File,
        jobId: String,
        descriptor: SelectedTrackDescriptor,
        verified: V2VerifiedPcmCache,
    ): V2VerifiedPcmCache {
        val expected = pcmFile(jobArtifactDirectory, descriptor.powerampFileId).canonicalFile
        val actual = verified.result.file.canonicalFile
        require(actual == expected &&
            actual.length() == verified.verifiedPcmByteLength &&
            actual.lastModified() == verified.verifiedPcmLastModifiedEpochMs
        ) { "verified PCM changed after hashing" }
        return requireVerified(
            pcm = actual,
            receipt = verified.receipt,
            jobId = jobId,
            descriptor = descriptor,
            knownPcmSha256 = verified.receipt.pcmSha256,
        )
    }

    fun delete(jobArtifactDirectory: File, powerampFileId: Long) {
        pcmFile(jobArtifactDirectory, powerampFileId).delete()
        receiptFile(jobArtifactDirectory, powerampFileId).delete()
    }

    private fun requireVerified(
        pcm: File,
        receipt: V2VerifiedPcmCacheReceipt,
        jobId: String,
        descriptor: SelectedTrackDescriptor,
        knownPcmSha256: String? = null,
        onHashProgress: (completedBytes: Long, totalBytes: Long) -> Unit = { _, _ -> },
    ): V2VerifiedPcmCache {
        require(receipt.schemaVersion == SCHEMA_VERSION)
        require(receipt.jobId == jobId && receipt.powerampFileId == descriptor.powerampFileId)
        require(receipt.provisionalWorkId == (descriptor.provisionalWorkId ?: descriptor.workId))
        require(receipt.providerSnapshotGeneration == descriptor.providerSnapshotGeneration)
        require(receipt.sourceFingerprint == descriptor.sourceFingerprint)
        require(receipt.sourceSampleRate == descriptor.finalizedAudioSpan.container.sampleRateHz)
        require(receipt.sourceStartSample == descriptor.finalizedAudioSpan.startSourceSample)
        require(receipt.sourceSampleCount ==
            receipt.sourceEndSampleExclusive - receipt.sourceStartSample)
        require(receipt.normalizationSampleCount == receipt.exactSampleCount24k)
        require(receipt.preprocessingSpecId ==
            com.powerampstartradio.indexing.NativeMath.TORCHAUDIO_HANN_V1_SPEC_ID)
        require(receipt.pcmByteLength == Math.multiplyExact(
            receipt.exactSampleCount24k,
            Float.SIZE_BYTES.toLong(),
        ))
        require(pcm.isFile && pcm.length() == receipt.pcmByteLength)
        require(receipt.pcmSha256.matches(Regex("^[0-9a-f]{64}$")))
        val actualPcmSha256 = knownPcmSha256
            ?: V2FileSha256.digest(pcm, onHashProgress)
        require(actualPcmSha256 == receipt.pcmSha256)

        val evidence = V2DecodedEosEvidence(
            sourceSampleRateHz = receipt.sourceSampleRate,
            observedStartSourceSample = receipt.sourceStartSample,
            observedEndSourceSampleExclusive = receipt.sourceEndSampleExclusive,
            observedSourceSampleCount = receipt.sourceSampleCount,
            exactSampleCount24k = receipt.exactSampleCount24k,
            endOfStreamReached = receipt.endOfStreamReached,
        )
        val span = descriptor.finalizedAudioSpan
        if (span.kind == V2ResolvedAudioSpanKind.WHOLE_FILE) {
            val finalized = V2DecodedEosSpanFinalizer.finalize(span, evidence)
            if (span.authority == V2AudioSpanAuthority.DECODED_END_OF_STREAM) {
                require(finalized == span)
            }
            require(!receipt.logicalBoundaryEnforced)
        } else {
            require(!receipt.endOfStreamReached && receipt.logicalBoundaryEnforced)
            require(receipt.sourceEndSampleExclusive == span.endSourceSampleExclusive)
            require(receipt.sourceSampleCount == span.sourceSampleCount)
            require(receipt.exactSampleCount24k == span.exactSampleCount24k)
        }
        val normalization = MertInference.WholeTrackNormalization(
            sampleCount = receipt.normalizationSampleCount,
            mean = receipt.normalizationMean,
            standardDeviation = receipt.normalizationStandardDeviation,
        )
        return V2VerifiedPcmCache(
            result = TrackPcmCache.Result(
                file = pcm,
                normalization = normalization,
                sourceSampleRate = receipt.sourceSampleRate,
                decoderName = receipt.decoderName,
                sourceChannelCount = receipt.sourceChannelCount,
                sourcePcmEncoding = receipt.sourcePcmEncoding,
                chunks = receipt.chunks,
                decodeMs = 0L,
                resampleMs = 0L,
                sourceStartSample = receipt.sourceStartSample,
                sourceEndSampleExclusive = receipt.sourceEndSampleExclusive,
                sourceSampleCount = receipt.sourceSampleCount,
                exactSampleCount24k = receipt.exactSampleCount24k,
                endOfStreamReached = receipt.endOfStreamReached,
                logicalBoundaryEnforced = receipt.logicalBoundaryEnforced,
                preprocessingSpecId = receipt.preprocessingSpecId,
                pcmSha256 = receipt.pcmSha256,
            ),
            receipt = receipt,
            verifiedPcmByteLength = pcm.length(),
            verifiedPcmLastModifiedEpochMs = pcm.lastModified(),
        )
    }

    private fun directory(root: File): File = File(root, "pcm-cache-v1").also { directory ->
        require(directory.isDirectory || directory.mkdirs()) { "cannot create PCM cache directory" }
    }

    private fun receiptFile(root: File, powerampFileId: Long): AtomicFile =
        AtomicFile(receiptBaseFile(root, powerampFileId))

    private fun receiptBaseFile(root: File, powerampFileId: Long): File =
        File(directory(root), "$powerampFileId.receipt.json")

    private fun deleteIncompletePair(pcm: File, receipt: File) {
        deleteAtomicFamily(pcm)
        deleteAtomicFamily(receipt)
    }

    private fun deleteAtomicFamily(base: File) {
        listOf(base, File(base.path + ".new"), File(base.path + ".bak")).forEach { file ->
            if (file.exists() && !file.delete()) {
                throw IOException("unable to remove incomplete PCM cache file $file")
            }
        }
    }

    private fun writeReceipt(file: AtomicFile, receipt: V2VerifiedPcmCacheReceipt) {
        val stream = file.startWrite()
        try {
            val writer = OutputStreamWriter(stream, StandardCharsets.UTF_8)
            gson.toJson(receipt, writer)
            writer.flush()
            file.finishWrite(stream)
        } catch (error: Throwable) {
            file.failWrite(stream)
            throw error
        }
    }

    private companion object {
        const val SCHEMA_VERSION = 2
        val SHA256 = Regex("^[0-9a-f]{64}$")
    }
}
