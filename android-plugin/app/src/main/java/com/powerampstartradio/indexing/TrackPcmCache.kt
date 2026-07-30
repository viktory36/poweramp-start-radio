package com.powerampstartradio.indexing

import android.util.AtomicFile
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.security.MessageDigest
import kotlin.math.sqrt

/** Builds one canonical, memory-bounded 24 kHz waveform for a logical Poweramp track. */
class TrackPcmCache private constructor(
    private val streamDecoder: NativePcmStreamDecoder,
    private val alignedResampler: AlignedResampler,
    private val outputFactory: AtomicOutputFactory,
) {
    constructor(decoder: AudioDecoder = AudioDecoder()) : this(
        streamDecoder = NativePcmStreamDecoder { file, startTimeUs, endTimeUs, cancelled, chunk ->
            decoder.decodeNativePcmStream(
                file = file,
                startTimeUs = startTimeUs,
                endTimeUs = endTimeUs,
                isCancelled = cancelled,
                onChunk = chunk,
            )
        },
        alignedResampler = AlignedResampler { policy, samples, fromRate, toRate,
                inputStartSample, totalInputSamples, outputStartSample, outputSampleCount ->
            when (policy) {
                ResamplerPolicy.LEGACY_KAISER -> NativeMath.resamplePolyphaseAligned(
                    samples = samples,
                    fromRate = fromRate,
                    toRate = toRate,
                    inputStartSample = inputStartSample,
                    totalInputSamples = totalInputSamples,
                    outputStartSample = outputStartSample,
                    outputSampleCount = outputSampleCount,
                )
                ResamplerPolicy.TORCHAUDIO_HANN_V1 ->
                    NativeMath.resampleTorchAudioHannV1Aligned(
                        samples = samples,
                        fromRate = fromRate,
                        toRate = toRate,
                        inputStartSample = inputStartSample,
                        totalInputSamples = totalInputSamples,
                        outputStartSample = outputStartSample,
                        outputSampleCount = outputSampleCount,
                    )
            }
        },
        outputFactory = AtomicOutputFactory(::AndroidAtomicOutput),
    )

    internal constructor(
        streamDecoder: NativePcmStreamDecoder,
        alignedResampler: AlignedResampler,
        outputFactory: AtomicOutputFactory,
        @Suppress("UNUSED_PARAMETER") testing: Unit = Unit,
    ) : this(streamDecoder, alignedResampler, outputFactory)

    companion object {
        private const val TARGET_SAMPLE_RATE = MertInference.SAMPLE_RATE
        private const val LEGACY_RESAMPLE_CONTEXT_MS = 100L
        private const val IO_BUFFER_BYTES = 1024 * 1024
        private const val MAX_RESAMPLE_CHUNK_DURATION_MS = 60_000L
        private const val LEGACY_KAISER_SPEC_ID =
            "legacy-kaiser-polyphase-v1-exact-rational-target-length"
        private const val HEX = "0123456789abcdef"

        private fun resampledLength(
            inputSamples: Long,
            fromRate: Int,
            toRate: Int,
            policy: ResamplerPolicy,
        ): Long = when (policy) {
            ResamplerPolicy.LEGACY_KAISER ->
                AudioSampleTimeline.resampledSampleCount(inputSamples, fromRate, toRate)
            ResamplerPolicy.TORCHAUDIO_HANN_V1 ->
                TorchAudioHannV1Policy.resampledLength(inputSamples, fromRate, toRate)
        }
    }

    enum class BoundaryMode {
        /** Ordinary file: every duration estimate is advisory; decode stops only at codec EOS. */
        REQUIRE_PHYSICAL_END_OF_STREAM,

        /** Logical CUE row: enforce the requested half-open end without requiring physical EOS. */
        ENFORCE_LOGICAL_HALF_OPEN_SPAN,
    }

    enum class ResamplerPolicy {
        /** Existing V1 behavior; retained only for compatibility. */
        LEGACY_KAISER,

        /** Desktop-compatible policy required by every V2 embedding spec. */
        TORCHAUDIO_HANN_V1,
    }

    internal fun interface NativePcmStreamDecoder {
        fun decode(
            file: File,
            startTimeUs: Long,
            endTimeUs: Long?,
            isCancelled: () -> Boolean,
            onChunk: (startSourceSample: Long, samples: FloatArray) -> Unit,
        ): AudioDecoder.NativePcmStreamResult
    }

    internal fun interface AlignedResampler {
        fun resample(
            policy: ResamplerPolicy,
            samples: FloatArray,
            fromRate: Int,
            toRate: Int,
            inputStartSample: Long,
            totalInputSamples: Long,
            outputStartSample: Long,
            outputSampleCount: Int,
        ): FloatArray?
    }

    internal interface AtomicOutput {
        val channel: FileChannel
        fun commit()
        fun abort()
    }

    internal fun interface AtomicOutputFactory {
        fun open(outputFile: File): AtomicOutput
    }

    private class AndroidAtomicOutput(outputFile: File) : AtomicOutput {
        private val atomicFile = AtomicFile(outputFile)
        private var stream: FileOutputStream? = atomicFile.startWrite()

        override val channel: FileChannel
            get() = checkNotNull(stream) { "atomic PCM output is already closed" }.channel

        override fun commit() {
            val active = stream ?: return
            atomicFile.finishWrite(active)
            stream = null
        }

        override fun abort() {
            val active = stream ?: return
            atomicFile.failWrite(active)
            stream = null
        }
    }

    internal data class ChunkRequest(
        val index: Int,
        val totalChunks: Int,
        val outputStartSample: Long,
        val outputSampleCount: Int,
        val inputStartSample: Long,
        val inputSampleCount: Int,
    )

    /** Pure globally-addressed schedule; its concatenated outputs equal a whole-track call. */
    internal object ChunkPolicy {
        fun totalChunks(totalOutputSamples: Long, chunkDurationMs: Long): Int {
            require(totalOutputSamples > 0L) { "totalOutputSamples must be positive" }
            val chunkSamples = outputSamplesPerChunk(chunkDurationMs)
            return Math.toIntExact(ceilDiv(totalOutputSamples, chunkSamples))
        }

        fun request(
            index: Int,
            totalInputSamples: Long,
            fromRate: Int,
            toRate: Int,
            chunkDurationMs: Long,
            policy: ResamplerPolicy,
        ): ChunkRequest {
            require(totalInputSamples > 0L) { "totalInputSamples must be positive" }
            require(fromRate > 0 && toRate > 0) { "sample rates must be positive" }
            val totalOutputSamples = resampledLength(
                totalInputSamples,
                fromRate,
                toRate,
                policy,
            )
            val chunkSamples = outputSamplesPerChunk(chunkDurationMs)
            val totalChunks = Math.toIntExact(ceilDiv(totalOutputSamples, chunkSamples))
            require(index in 0 until totalChunks) { "chunk index $index is out of range" }
            val outputStart = Math.multiplyExact(index.toLong(), chunkSamples)
            val outputCount = minOf(chunkSamples, totalOutputSamples - outputStart).toInt()
            val inputRange = requiredInputRange(
                totalInputSamples = totalInputSamples,
                fromRate = fromRate,
                toRate = toRate,
                outputStartSample = outputStart,
                outputSampleCount = outputCount,
                policy = policy,
            )
            val inputCount = inputRange.endExclusive - inputRange.start
            require(inputCount in 1..Int.MAX_VALUE.toLong()) {
                "resample input chunk is not memory bounded: $inputCount samples"
            }
            return ChunkRequest(
                index = index,
                totalChunks = totalChunks,
                outputStartSample = outputStart,
                outputSampleCount = outputCount,
                inputStartSample = inputRange.start,
                inputSampleCount = inputCount.toInt(),
            )
        }

        private fun outputSamplesPerChunk(chunkDurationMs: Long): Long {
            require(chunkDurationMs > 0L) { "chunkDurationMs must be positive" }
            val boundedDuration = minOf(chunkDurationMs, MAX_RESAMPLE_CHUNK_DURATION_MS)
            return Math.multiplyExact(boundedDuration, TARGET_SAMPLE_RATE.toLong()) / 1000L
        }

        private fun requiredInputRange(
            totalInputSamples: Long,
            fromRate: Int,
            toRate: Int,
            outputStartSample: Long,
            outputSampleCount: Int,
            policy: ResamplerPolicy,
        ): TorchAudioHannV1Policy.InputRange {
            if (fromRate == toRate) {
                return TorchAudioHannV1Policy.InputRange(
                    outputStartSample,
                    outputStartSample + outputSampleCount,
                )
            }
            if (policy == ResamplerPolicy.TORCHAUDIO_HANN_V1) {
                return TorchAudioHannV1Policy.requiredInputRange(
                    totalInputSamples = totalInputSamples,
                    fromRate = fromRate,
                    toRate = toRate,
                    outputStartSample = outputStartSample,
                    outputSampleCount = outputSampleCount,
                )
            }

            // The legacy native kernel does not expose its exact tap range. A 100 ms native-rate
            // guard is deliberately much wider than its fixed FIR while retaining bounded RAM.
            val outputEnd = outputStartSample + outputSampleCount
            val coreStart = multiplyDivideFloor(outputStartSample, fromRate, toRate)
            val coreEnd = multiplyDivideCeil(outputEnd, fromRate, toRate)
            val context = Math.multiplyExact(fromRate.toLong(), LEGACY_RESAMPLE_CONTEXT_MS) / 1000L
            return TorchAudioHannV1Policy.InputRange(
                start = (coreStart - context).coerceAtLeast(0L),
                endExclusive = Math.addExact(coreEnd, context).coerceAtMost(totalInputSamples),
            )
        }

        private fun multiplyDivideFloor(value: Long, multiplier: Int, divisor: Int): Long {
            val whole = Math.multiplyExact(value / divisor, multiplier.toLong())
            val remainder = Math.multiplyExact(value % divisor, multiplier.toLong()) / divisor
            return Math.addExact(whole, remainder)
        }

        private fun multiplyDivideCeil(value: Long, multiplier: Int, divisor: Int): Long {
            val floor = multiplyDivideFloor(value, multiplier, divisor)
            return if ((value % divisor) * multiplier % divisor == 0L) floor
            else Math.addExact(floor, 1L)
        }

        private fun ceilDiv(value: Long, divisor: Long): Long =
            if (value == 0L) 0L else 1L + (value - 1L) / divisor
    }

    enum class PcmContractFailure {
        SOURCE_PLAN_MISMATCH,
        EOS_MISMATCH,
        LOGICAL_BOUNDARY_MISMATCH,
        PREPROCESSING_MISMATCH,
        PCM_ARTIFACT_MISMATCH,
    }

    class PcmContractException(
        val reason: PcmContractFailure,
        message: String,
    ) : IllegalStateException(message)

    data class Result(
        val file: File,
        val normalization: MertInference.WholeTrackNormalization,
        val sourceSampleRate: Int,
        val decoderName: String,
        val sourceChannelCount: Int,
        val sourcePcmEncoding: Int,
        val chunks: Int,
        val decodeMs: Long,
        val resampleMs: Long,
        val sourceStartSample: Long,
        val sourceEndSampleExclusive: Long,
        val sourceSampleCount: Long,
        val exactSampleCount24k: Long,
        val endOfStreamReached: Boolean,
        val logicalBoundaryEnforced: Boolean,
        val preprocessingSpecId: String,
        /** Digest of the committed PCM bytes, computed during the required variance pass. */
        val pcmSha256: String? = null,
    )

    fun build(
        sourceFile: File,
        logicalStartUs: Long,
        logicalDurationUs: Long,
        chunkDurationMs: Long,
        outputFile: File,
        onChunkDone: ((completed: Int, total: Int) -> Unit)? = null,
        onNormalizationProgress: ((completedSamples: Long, totalSamples: Long) -> Unit)? = null,
        boundaryMode: BoundaryMode = BoundaryMode.ENFORCE_LOGICAL_HALF_OPEN_SPAN,
        resamplerPolicy: ResamplerPolicy = ResamplerPolicy.LEGACY_KAISER,
        isCancelled: () -> Boolean = { Thread.currentThread().isInterrupted },
    ): Result {
        require(sourceFile.isFile && sourceFile.canRead()) {
            "Audio source is not readable: ${sourceFile.absolutePath}"
        }
        require(logicalStartUs >= 0L) { "logicalStartUs must be non-negative" }
        require(logicalDurationUs >= 0L) { "logicalDurationUs must be non-negative" }
        require(boundaryMode != BoundaryMode.ENFORCE_LOGICAL_HALF_OPEN_SPAN ||
            logicalDurationUs > 0L
        ) { "a logical half-open span must have positive duration" }
        require(chunkDurationMs > 0L) { "chunkDurationMs must be positive" }

        outputFile.parentFile?.let { parent ->
            require(parent.isDirectory || parent.mkdirs()) { "cannot create PCM output directory" }
        }
        val scratchFile = nativeScratchFile(outputFile)
        require(!scratchFile.exists() || scratchFile.delete()) {
            "cannot remove stale native PCM scratch ${scratchFile.absolutePath}"
        }
        val logicalEndUs = if (boundaryMode == BoundaryMode.ENFORCE_LOGICAL_HALF_OPEN_SPAN) {
            Math.addExact(logicalStartUs, logicalDurationUs)
        } else {
            null
        }
        val ioBuffer = ByteBuffer.allocateDirect(IO_BUFFER_BYTES).order(ByteOrder.LITTLE_ENDIAN)
        var committedOutput = false

        try {
            val streamed = decodeOnceToScratch(
                sourceFile = sourceFile,
                logicalStartUs = logicalStartUs,
                logicalEndUs = logicalEndUs,
                boundaryMode = boundaryMode,
                scratchFile = scratchFile,
                ioBuffer = ioBuffer,
                isCancelled = isCancelled,
            )
            val boundary = streamed.boundaryEvidence
            val sourceSampleRate = streamed.sampleRate
            val sourceSampleCount = boundary.observedSourceSampleCount
            val expectedTargetSamples = resampledLength(
                sourceSampleCount,
                sourceSampleRate,
                TARGET_SAMPLE_RATE,
                resamplerPolicy,
            )
            contract(
                expectedTargetSamples > 0L,
                PcmContractFailure.SOURCE_PLAN_MISMATCH,
                "Decoded logical span contains no target PCM samples",
            )
            val expectedBytes = Math.multiplyExact(
                expectedTargetSamples,
                Float.SIZE_BYTES.toLong(),
            )
            val totalChunks = ChunkPolicy.totalChunks(expectedTargetSamples, chunkDurationMs)
            var output: AtomicOutput? = null
            var samplesWritten = 0L
            var sampleSum = 0.0
            var resampleMs = 0L
            try {
                output = outputFactory.open(outputFile)
                val outputChannel = output.channel
                FileInputStream(scratchFile).channel.use { scratchChannel ->
                    repeat(totalChunks) { index ->
                        requireActive(isCancelled)
                        val request = ChunkPolicy.request(
                            index = index,
                            totalInputSamples = sourceSampleCount,
                            fromRate = sourceSampleRate,
                            toRate = TARGET_SAMPLE_RATE,
                            chunkDurationMs = chunkDurationMs,
                            policy = resamplerPolicy,
                        )
                        contract(
                            request.outputStartSample == samplesWritten,
                            PcmContractFailure.PCM_ARTIFACT_MISMATCH,
                            "PCM chunks are discontinuous at target sample $samplesWritten",
                        )
                        val nativeSlice = readFloatsExactly(
                            channel = scratchChannel,
                            startSample = request.inputStartSample,
                            sampleCount = request.inputSampleCount,
                            buffer = ioBuffer,
                            isCancelled = isCancelled,
                        )
                        val resampleStart = System.nanoTime()
                        val aligned = alignedResampler.resample(
                            policy = resamplerPolicy,
                            samples = nativeSlice,
                            fromRate = sourceSampleRate,
                            toRate = TARGET_SAMPLE_RATE,
                            inputStartSample = request.inputStartSample,
                            totalInputSamples = sourceSampleCount,
                            outputStartSample = request.outputStartSample,
                            outputSampleCount = request.outputSampleCount,
                        ) ?: throw PcmContractException(
                            PcmContractFailure.PREPROCESSING_MISMATCH,
                            "Aligned resampling lacked native context for output chunk ${index + 1}",
                        )
                        resampleMs += (System.nanoTime() - resampleStart) / 1_000_000L
                        contract(
                            aligned.size == request.outputSampleCount,
                            PcmContractFailure.PREPROCESSING_MISMATCH,
                            "Aligned resampler returned ${aligned.size} of " +
                                "${request.outputSampleCount} samples",
                        )
                        writeFloatsFully(outputChannel, aligned, ioBuffer, isCancelled)
                        for (sample in aligned) sampleSum += sample
                        samplesWritten += aligned.size
                        val completed = index + 1
                        contract(
                            completed <= totalChunks,
                            PcmContractFailure.PCM_ARTIFACT_MISMATCH,
                            "PCM progress exceeded its exact total",
                        )
                        onChunkDone?.invoke(completed, totalChunks)
                        requireActive(isCancelled)
                    }
                    contract(
                        scratchChannel.size() == Math.multiplyExact(
                            sourceSampleCount,
                            Float.SIZE_BYTES.toLong(),
                        ),
                        PcmContractFailure.PCM_ARTIFACT_MISMATCH,
                        "Native PCM scratch changed while resampling",
                    )
                }
                contract(
                    samplesWritten == expectedTargetSamples,
                    PcmContractFailure.PCM_ARTIFACT_MISMATCH,
                    "PCM sample count mismatch: wrote $samplesWritten of $expectedTargetSamples",
                )
                contract(
                    outputChannel.position() == expectedBytes,
                    PcmContractFailure.PCM_ARTIFACT_MISMATCH,
                    "PCM byte count mismatch: wrote ${outputChannel.position()} of $expectedBytes",
                )
                outputChannel.force(true)
                output.commit()
                committedOutput = true
                output = null
            } catch (error: Throwable) {
                output?.abort()
                throw error
            }

            contract(
                outputFile.length() == expectedBytes,
                PcmContractFailure.PCM_ARTIFACT_MISMATCH,
                "Published PCM cache has ${outputFile.length()} of $expectedBytes bytes",
            )
            val mean = (sampleSum / samplesWritten).toFloat()
            val varianceAndDigest = calculateVarianceAndSha256(
                file = outputFile,
                sampleCount = samplesWritten,
                mean = mean,
                isCancelled = isCancelled,
                onProgress = onNormalizationProgress,
            )
            val standardDeviation = sqrt(varianceAndDigest.variance.toFloat() + 1e-7f)
            return Result(
                file = outputFile,
                normalization = MertInference.WholeTrackNormalization(
                    sampleCount = samplesWritten,
                    mean = mean,
                    standardDeviation = standardDeviation,
                ),
                sourceSampleRate = sourceSampleRate,
                decoderName = streamed.decoderName,
                sourceChannelCount = streamed.sourceChannelCount,
                sourcePcmEncoding = streamed.sourcePcmEncoding,
                chunks = totalChunks,
                decodeMs = streamed.decodeMs,
                resampleMs = resampleMs,
                sourceStartSample = boundary.observedStartSourceSample,
                sourceEndSampleExclusive = boundary.observedEndSourceSampleExclusive,
                sourceSampleCount = sourceSampleCount,
                exactSampleCount24k = expectedTargetSamples,
                endOfStreamReached = boundaryMode == BoundaryMode.REQUIRE_PHYSICAL_END_OF_STREAM &&
                    boundary.endOfStreamReached,
                logicalBoundaryEnforced =
                    boundaryMode == BoundaryMode.ENFORCE_LOGICAL_HALF_OPEN_SPAN &&
                        boundary.requestedBoundaryReached,
                preprocessingSpecId = when (resamplerPolicy) {
                    ResamplerPolicy.LEGACY_KAISER -> LEGACY_KAISER_SPEC_ID
                    ResamplerPolicy.TORCHAUDIO_HANN_V1 -> NativeMath.TORCHAUDIO_HANN_V1_SPEC_ID
                },
                pcmSha256 = varianceAndDigest.sha256,
            )
        } catch (error: Throwable) {
            if (committedOutput && outputFile.exists() && !outputFile.delete()) {
                error.addSuppressed(
                    IOException("unable to remove unpublished PCM output ${outputFile.absolutePath}"),
                )
            }
            throw error
        } finally {
            if (scratchFile.exists() && !scratchFile.delete()) {
                scratchFile.deleteOnExit()
            }
        }
    }

    private fun decodeOnceToScratch(
        sourceFile: File,
        logicalStartUs: Long,
        logicalEndUs: Long?,
        boundaryMode: BoundaryMode,
        scratchFile: File,
        ioBuffer: ByteBuffer,
        isCancelled: () -> Boolean,
    ): AudioDecoder.NativePcmStreamResult {
        var firstStreamedSample: Long? = null
        var nextStreamedSample: Long? = null
        var streamedSamples = 0L
        FileOutputStream(scratchFile).channel.use { scratchChannel ->
            val result = streamDecoder.decode(
                file = sourceFile,
                startTimeUs = logicalStartUs,
                endTimeUs = logicalEndUs,
                isCancelled = isCancelled,
            ) { startSourceSample, samples ->
                requireActive(isCancelled)
                if (samples.isEmpty()) return@decode
                val expectedStart = nextStreamedSample
                if (expectedStart != null && startSourceSample != expectedStart) {
                    throw AudioDecoder.AudioDecodeException(
                        "Native PCM stream is discontinuous: $expectedStart -> $startSourceSample",
                    )
                }
                if (firstStreamedSample == null) firstStreamedSample = startSourceSample
                samples.forEachIndexed { index, sample ->
                    contract(
                        sample.isFinite(),
                        PcmContractFailure.PCM_ARTIFACT_MISMATCH,
                        "Native PCM contains a non-finite sample at ${streamedSamples + index}",
                    )
                }
                writeFloatsFully(scratchChannel, samples, ioBuffer, isCancelled)
                streamedSamples = Math.addExact(streamedSamples, samples.size.toLong())
                nextStreamedSample = Math.addExact(startSourceSample, samples.size.toLong())
            }
            scratchChannel.force(true)
            val evidence = result.boundaryEvidence
            val expectedStart = AudioSampleTimeline.sampleAtOrAfter(
                logicalStartUs,
                result.sampleRate,
            )
            contract(
                firstStreamedSample == expectedStart &&
                    evidence.requestedStartSourceSample == expectedStart &&
                    evidence.observedStartSourceSample == expectedStart,
                PcmContractFailure.SOURCE_PLAN_MISMATCH,
                "Decoded native start disagrees with logical start: $evidence",
            )
            contract(
                streamedSamples > 0L &&
                    nextStreamedSample == evidence.observedEndSourceSampleExclusive &&
                    streamedSamples == evidence.observedSourceSampleCount &&
                    evidence.observedEndSourceSampleExclusive -
                    evidence.observedStartSourceSample == streamedSamples,
                PcmContractFailure.SOURCE_PLAN_MISMATCH,
                "Native scratch and decoder boundary evidence disagree: $evidence",
            )
            contract(
                scratchChannel.position() == Math.multiplyExact(
                    streamedSamples,
                    Float.SIZE_BYTES.toLong(),
                ),
                PcmContractFailure.PCM_ARTIFACT_MISMATCH,
                "Native scratch byte count disagrees with decoder samples",
            )
            when (boundaryMode) {
                BoundaryMode.REQUIRE_PHYSICAL_END_OF_STREAM -> contract(
                    logicalEndUs == null &&
                        evidence.requestedEndSourceSampleExclusive == null &&
                        evidence.endOfStreamReached &&
                        !evidence.requestedBoundaryReached,
                    PcmContractFailure.EOS_MISMATCH,
                    "Ordinary file did not finish at physical codec EOS: $evidence",
                )
                BoundaryMode.ENFORCE_LOGICAL_HALF_OPEN_SPAN -> {
                    val endUs = checkNotNull(logicalEndUs)
                    val expectedEnd = AudioSampleTimeline.sampleAtOrAfter(endUs, result.sampleRate)
                    contract(
                        evidence.requestedEndSourceSampleExclusive == expectedEnd &&
                            evidence.requestedBoundaryReached &&
                            evidence.observedEndSourceSampleExclusive == expectedEnd,
                        PcmContractFailure.LOGICAL_BOUNDARY_MISMATCH,
                        "CUE half-open boundary was not enforced exactly: $evidence",
                    )
                }
            }
            return result
        }
    }

    private fun calculateVarianceAndSha256(
        file: File,
        sampleCount: Long,
        mean: Float,
        isCancelled: () -> Boolean,
        onProgress: ((completedSamples: Long, totalSamples: Long) -> Unit)?,
    ): VarianceAndSha256 {
        var squaredDeviationSum = 0.0
        var samplesRead = 0L
        var lastReportedSamples = 0L
        val digest = MessageDigest.getInstance("SHA-256")
        val progressIntervalSamples = (8L * 1024L * 1024L) / Float.SIZE_BYTES
        val buffer = ByteBuffer.allocateDirect(IO_BUFFER_BYTES).order(ByteOrder.LITTLE_ENDIAN)
        onProgress?.invoke(0L, sampleCount)
        FileInputStream(file).channel.use { channel ->
            while (samplesRead < sampleCount) {
                requireActive(isCancelled)
                val samplesThisRead = minOf(
                    (IO_BUFFER_BYTES / Float.SIZE_BYTES).toLong(),
                    sampleCount - samplesRead,
                ).toInt()
                buffer.clear()
                buffer.limit(samplesThisRead * Float.SIZE_BYTES)
                contract(
                    readFully(channel, buffer),
                    PcmContractFailure.PCM_ARTIFACT_MISMATCH,
                    "PCM cache ended during variance pass",
                )
                buffer.flip()
                digest.update(buffer.asReadOnlyBuffer())
                val floats = buffer.asFloatBuffer()
                repeat(samplesThisRead) {
                    val difference = floats.get() - mean
                    squaredDeviationSum += difference * difference
                }
                samplesRead += samplesThisRead
                if (samplesRead == sampleCount ||
                    samplesRead - lastReportedSamples >= progressIntervalSamples
                ) {
                    onProgress?.invoke(samplesRead, sampleCount)
                    lastReportedSamples = samplesRead
                }
            }
            contract(
                channel.position() == file.length(),
                PcmContractFailure.PCM_ARTIFACT_MISMATCH,
                "PCM variance pass read ${channel.position()} of ${file.length()} bytes",
            )
        }
        return VarianceAndSha256(
            variance = squaredDeviationSum / sampleCount,
            sha256 = digest.digest().toLowerHex(),
        )
    }

    private data class VarianceAndSha256(
        val variance: Double,
        val sha256: String,
    )

    private fun ByteArray.toLowerHex(): String {
        val chars = CharArray(size * 2)
        forEachIndexed { index, byte ->
            val value = byte.toInt() and 0xff
            chars[index * 2] = HEX[value ushr 4]
            chars[index * 2 + 1] = HEX[value and 0x0f]
        }
        return String(chars)
    }

    private fun readFloatsExactly(
        channel: FileChannel,
        startSample: Long,
        sampleCount: Int,
        buffer: ByteBuffer,
        isCancelled: () -> Boolean,
    ): FloatArray {
        val result = FloatArray(sampleCount)
        channel.position(Math.multiplyExact(startSample, Float.SIZE_BYTES.toLong()))
        var offset = 0
        val maxSamples = buffer.capacity() / Float.SIZE_BYTES
        while (offset < sampleCount) {
            requireActive(isCancelled)
            val count = minOf(maxSamples, sampleCount - offset)
            buffer.clear()
            buffer.limit(count * Float.SIZE_BYTES)
            contract(
                readFully(channel, buffer),
                PcmContractFailure.PCM_ARTIFACT_MISMATCH,
                "Native PCM scratch ended while reading [$startSample, ${startSample + sampleCount})",
            )
            buffer.flip()
            buffer.asFloatBuffer().get(result, offset, count)
            offset += count
        }
        return result
    }

    private fun writeFloatsFully(
        channel: FileChannel,
        samples: FloatArray,
        buffer: ByteBuffer,
        isCancelled: () -> Boolean,
    ) {
        var offset = 0
        val maxSamples = buffer.capacity() / Float.SIZE_BYTES
        while (offset < samples.size) {
            requireActive(isCancelled)
            val count = minOf(maxSamples, samples.size - offset)
            buffer.clear()
            buffer.asFloatBuffer().put(samples, offset, count)
            buffer.limit(count * Float.SIZE_BYTES)
            writeFully(channel, buffer, isCancelled)
            offset += count
        }
    }

    private fun writeFully(
        channel: FileChannel,
        buffer: ByteBuffer,
        isCancelled: () -> Boolean,
    ) {
        while (buffer.hasRemaining()) {
            requireActive(isCancelled)
            val written = channel.write(buffer)
            if (written == 0) Thread.yield()
        }
    }

    private fun readFully(channel: FileChannel, buffer: ByteBuffer): Boolean {
        while (buffer.hasRemaining()) {
            val read = channel.read(buffer)
            if (read < 0) return false
            if (read == 0) Thread.yield()
        }
        return true
    }

    private fun nativeScratchFile(outputFile: File): File =
        File(outputFile.parentFile, ".${outputFile.name}.native-f32.scratch")

    private fun requireActive(isCancelled: () -> Boolean) {
        if (isCancelled()) throw AudioDecoder.AudioDecodeCancelledException()
    }

    private fun contract(
        condition: Boolean,
        reason: PcmContractFailure,
        message: String,
    ) {
        if (!condition) throw PcmContractException(reason, message)
    }
}
