package com.powerampstartradio.indexing

import android.media.AudioFormat
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.security.MessageDigest
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class TrackPcmCacheStreamingTest {
    @get:Rule
    val temporary = TemporaryFolder()

    @Test
    fun `ordinary duration under over and missing estimates all decode once to identical EOS`() {
        val samples = FloatArray(53_137) { index -> ((index % 997) - 498) / 512f }
        val decoder = FakeStreamDecoder(samples = samples, sampleRate = TARGET_RATE)
        val cache = cache(decoder)
        val source = temporary.newFile("ordinary.flac")
        val estimatesUs = listOf(1L, 86_400_000_000L, 0L)

        estimatesUs.forEachIndexed { index, estimateUs ->
            val output = File(temporary.root, "ordinary-$index.pcm")
            val progress = mutableListOf<Pair<Int, Int>>()
            val result = cache.build(
                sourceFile = source,
                logicalStartUs = 0L,
                logicalDurationUs = estimateUs,
                chunkDurationMs = 1_000L,
                outputFile = output,
                boundaryMode = TrackPcmCache.BoundaryMode.REQUIRE_PHYSICAL_END_OF_STREAM,
                resamplerPolicy = TrackPcmCache.ResamplerPolicy.TORCHAUDIO_HANN_V1,
                onChunkDone = { completed, total -> progress += completed to total },
            )

            assertArrayEquals(floatBytes(samples), output.readBytes())
            assertEquals(sha256(output.readBytes()), result.pcmSha256)
            assertEquals(samples.size.toLong(), result.sourceSampleCount)
            assertEquals(samples.size.toLong(), result.exactSampleCount24k)
            assertTrue(result.endOfStreamReached)
            assertFalse(result.logicalBoundaryEnforced)
            assertTrue(progress.isNotEmpty())
            assertTrue(progress.all { (completed, total) -> completed in 1..total })
            assertEquals(progress.last().second, progress.last().first)
            assertScratchAbsent(output)
        }

        assertEquals(estimatesUs.size, decoder.invocations)
        assertTrue(decoder.requestedEnds.all { it == null })
    }

    @Test
    fun `one streaming decoder invocation may emit many native buffers and output chunks`() {
        val samples = FloatArray(100_123) { index -> index.toFloat() / SAMPLES_SCALE }
        val decoder = FakeStreamDecoder(
            samples = samples,
            sampleRate = TARGET_RATE,
            emittedChunkSamples = 137,
        )
        val output = File(temporary.root, "many-chunks.pcm")

        val result = cache(decoder).build(
            sourceFile = temporary.newFile("many-chunks.mp3"),
            logicalStartUs = 0L,
            logicalDurationUs = 1L,
            chunkDurationMs = 1_000L,
            outputFile = output,
            boundaryMode = TrackPcmCache.BoundaryMode.REQUIRE_PHYSICAL_END_OF_STREAM,
            resamplerPolicy = TrackPcmCache.ResamplerPolicy.TORCHAUDIO_HANN_V1,
        )

        assertEquals(1, decoder.invocations)
        assertTrue(decoder.emittedChunks > result.chunks)
        assertArrayEquals(floatBytes(samples), output.readBytes())
        assertScratchAbsent(output)
    }

    @Test
    fun `CUE decode enforces exact provider half-open native boundary`() {
        val startUs = 123_456L
        val durationUs = 1_234_567L
        val startSample = AudioSampleTimeline.sampleAtOrAfter(startUs, TARGET_RATE)
        val endSample = AudioSampleTimeline.sampleAtOrAfter(startUs + durationUs, TARGET_RATE)
        val samples = FloatArray((endSample - startSample).toInt()) { index -> index / 32_768f }
        val decoder = FakeStreamDecoder(samples, TARGET_RATE, emittedChunkSamples = 73)
        val output = File(temporary.root, "cue.pcm")

        val result = cache(decoder).build(
            sourceFile = temporary.newFile("cue-source.flac"),
            logicalStartUs = startUs,
            logicalDurationUs = durationUs,
            chunkDurationMs = 333L,
            outputFile = output,
            boundaryMode = TrackPcmCache.BoundaryMode.ENFORCE_LOGICAL_HALF_OPEN_SPAN,
            resamplerPolicy = TrackPcmCache.ResamplerPolicy.TORCHAUDIO_HANN_V1,
        )

        assertEquals(1, decoder.invocations)
        assertEquals(startUs + durationUs, decoder.requestedEnds.single())
        assertEquals(startSample, result.sourceStartSample)
        assertEquals(endSample, result.sourceEndSampleExclusive)
        assertFalse(result.endOfStreamReached)
        assertTrue(result.logicalBoundaryEnforced)
        assertArrayEquals(floatBytes(samples), output.readBytes())
        assertScratchAbsent(output)
    }

    @Test
    fun `CUE early physical EOS fails closed and removes scratch and unpublished output`() {
        val startUs = 250_000L
        val durationUs = 2_000_000L
        val requestedSamples = AudioSampleTimeline.sampleAtOrAfter(
            startUs + durationUs,
            TARGET_RATE,
        ) - AudioSampleTimeline.sampleAtOrAfter(startUs, TARGET_RATE)
        val decoder = FakeStreamDecoder(
            samples = FloatArray((requestedSamples - 17L).toInt()) { 0.25f },
            sampleRate = TARGET_RATE,
            forceEarlyEos = true,
        )
        val output = File(temporary.root, "early-eos.pcm")

        val error = assertThrows(TrackPcmCache.PcmContractException::class.java) {
            cache(decoder).build(
                sourceFile = temporary.newFile("early-eos.flac"),
                logicalStartUs = startUs,
                logicalDurationUs = durationUs,
                chunkDurationMs = 1_000L,
                outputFile = output,
                boundaryMode = TrackPcmCache.BoundaryMode.ENFORCE_LOGICAL_HALF_OPEN_SPAN,
                resamplerPolicy = TrackPcmCache.ResamplerPolicy.TORCHAUDIO_HANN_V1,
            )
        }

        assertEquals(
            TrackPcmCache.PcmContractFailure.LOGICAL_BOUNDARY_MISMATCH,
            error.reason,
        )
        assertEquals(1, decoder.invocations)
        assertFalse(output.exists())
        assertScratchAbsent(output)
    }

    @Test
    fun `cancellation aborts atomic output and always removes native scratch`() {
        val samples = FloatArray(72_000) { 0.5f }
        val decoder = FakeStreamDecoder(samples, TARGET_RATE)
        val output = File(temporary.root, "cancelled.pcm")
        val previousVerifiedBytes = byteArrayOf(1, 3, 3, 7)
        output.writeBytes(previousVerifiedBytes)
        var cancelled = false

        assertThrows(AudioDecoder.AudioDecodeCancelledException::class.java) {
            cache(decoder).build(
                sourceFile = temporary.newFile("cancelled.ogg"),
                logicalStartUs = 0L,
                logicalDurationUs = 0L,
                chunkDurationMs = 1_000L,
                outputFile = output,
                onChunkDone = { completed, _ -> if (completed == 1) cancelled = true },
                boundaryMode = TrackPcmCache.BoundaryMode.REQUIRE_PHYSICAL_END_OF_STREAM,
                resamplerPolicy = TrackPcmCache.ResamplerPolicy.TORCHAUDIO_HANN_V1,
                isCancelled = { cancelled },
            )
        }

        assertEquals(1, decoder.invocations)
        assertArrayEquals(previousVerifiedBytes, output.readBytes())
        assertScratchAbsent(output)
    }

    @Test
    fun `cancellation after atomic commit removes unpublished PCM output`() {
        val samples = FloatArray(72_000) { 0.25f }
        val decoder = FakeStreamDecoder(samples, TARGET_RATE)
        val output = File(temporary.root, "cancelled-after-commit.pcm")
        var cancelled = false
        val cache = cache(
            decoder,
            TrackPcmCache.AtomicOutputFactory { target ->
                TestAtomicOutput(target) { cancelled = true }
            },
        )

        assertThrows(AudioDecoder.AudioDecodeCancelledException::class.java) {
            cache.build(
                sourceFile = temporary.newFile("cancelled-after-commit.ogg"),
                logicalStartUs = 0L,
                logicalDurationUs = 0L,
                chunkDurationMs = 1_000L,
                outputFile = output,
                boundaryMode = TrackPcmCache.BoundaryMode.REQUIRE_PHYSICAL_END_OF_STREAM,
                resamplerPolicy = TrackPcmCache.ResamplerPolicy.TORCHAUDIO_HANN_V1,
                isCancelled = { cancelled },
            )
        }

        assertEquals(1, decoder.invocations)
        assertFalse(output.exists())
        assertScratchAbsent(output)
    }

    @Test
    fun `validation failure after atomic commit removes unpublished PCM output`() {
        val samples = FloatArray(72_000) { 0.25f }
        val decoder = FakeStreamDecoder(samples, TARGET_RATE)
        val output = File(temporary.root, "invalid-after-commit.pcm")
        val cache = cache(
            decoder,
            TrackPcmCache.AtomicOutputFactory { target ->
                TestAtomicOutput(target) { target.writeBytes(byteArrayOf(1)) }
            },
        )

        assertThrows(TrackPcmCache.PcmContractException::class.java) {
            cache.build(
                sourceFile = temporary.newFile("invalid-after-commit.ogg"),
                logicalStartUs = 0L,
                logicalDurationUs = 0L,
                chunkDurationMs = 1_000L,
                outputFile = output,
                boundaryMode = TrackPcmCache.BoundaryMode.REQUIRE_PHYSICAL_END_OF_STREAM,
                resamplerPolicy = TrackPcmCache.ResamplerPolicy.TORCHAUDIO_HANN_V1,
            )
        }

        assertEquals(1, decoder.invocations)
        assertFalse(output.exists())
        assertScratchAbsent(output)
    }

    @Test
    fun `chunk schedule is contiguous exact bounded and independent of planning estimate`() {
        val sourceSamples = 19_556_776L
        val expectedOutput = TorchAudioHannV1Policy.resampledLength(
            sourceSamples,
            44_100,
            TARGET_RATE,
        )
        val total = TrackPcmCache.ChunkPolicy.totalChunks(expectedOutput, 60_000L)
        var nextOutput = 0L

        repeat(total) { index ->
            val request = TrackPcmCache.ChunkPolicy.request(
                index = index,
                totalInputSamples = sourceSamples,
                fromRate = 44_100,
                toRate = TARGET_RATE,
                chunkDurationMs = Long.MAX_VALUE,
                policy = TrackPcmCache.ResamplerPolicy.TORCHAUDIO_HANN_V1,
            )
            val required = TorchAudioHannV1Policy.requiredInputRange(
                totalInputSamples = sourceSamples,
                fromRate = 44_100,
                toRate = TARGET_RATE,
                outputStartSample = request.outputStartSample,
                outputSampleCount = request.outputSampleCount,
            )
            assertEquals(index, request.index)
            assertEquals(total, request.totalChunks)
            assertEquals(nextOutput, request.outputStartSample)
            assertEquals(required.start, request.inputStartSample)
            assertEquals(
                required.endExclusive - required.start,
                request.inputSampleCount.toLong(),
            )
            assertTrue(request.outputSampleCount <= 60_000 * TARGET_RATE)
            nextOutput += request.outputSampleCount
        }

        assertEquals(expectedOutput, nextOutput)
    }

    private fun cache(
        decoder: FakeStreamDecoder,
        outputFactory: TrackPcmCache.AtomicOutputFactory =
            TrackPcmCache.AtomicOutputFactory(::TestAtomicOutput),
    ): TrackPcmCache = TrackPcmCache(
        streamDecoder = decoder,
        alignedResampler = TrackPcmCache.AlignedResampler {
                _, samples, fromRate, toRate, inputStart, _, outputStart, outputCount ->
            require(fromRate == toRate)
            val localStart = Math.toIntExact(outputStart - inputStart)
            samples.copyOfRange(localStart, localStart + outputCount)
        },
        outputFactory = outputFactory,
    )

    private class FakeStreamDecoder(
        private val samples: FloatArray,
        private val sampleRate: Int,
        private val emittedChunkSamples: Int = 4_093,
        private val forceEarlyEos: Boolean = false,
    ) : TrackPcmCache.NativePcmStreamDecoder {
        var invocations = 0
            private set
        var emittedChunks = 0
            private set
        val requestedEnds = mutableListOf<Long?>()

        override fun decode(
            file: File,
            startTimeUs: Long,
            endTimeUs: Long?,
            isCancelled: () -> Boolean,
            onChunk: (startSourceSample: Long, samples: FloatArray) -> Unit,
        ): AudioDecoder.NativePcmStreamResult {
            invocations++
            requestedEnds += endTimeUs
            val startSample = AudioSampleTimeline.sampleAtOrAfter(startTimeUs, sampleRate)
            var offset = 0
            while (offset < samples.size) {
                if (isCancelled()) throw AudioDecoder.AudioDecodeCancelledException()
                val count = minOf(emittedChunkSamples, samples.size - offset)
                onChunk(
                    startSample + offset,
                    samples.copyOfRange(offset, offset + count),
                )
                emittedChunks++
                offset += count
            }
            val observedEnd = startSample + samples.size
            val requestedEndSample = endTimeUs?.let {
                AudioSampleTimeline.sampleAtOrAfter(it, sampleRate)
            }
            val reachedBoundary = requestedEndSample != null &&
                !forceEarlyEos && observedEnd == requestedEndSample
            return AudioDecoder.NativePcmStreamResult(
                sampleRate = sampleRate,
                decoderName = "fake-decoder",
                sourceChannelCount = 2,
                sourcePcmEncoding = AudioFormat.ENCODING_PCM_FLOAT,
                decodeMs = 7L,
                boundaryEvidence = AudioDecoder.DecodeBoundaryEvidence(
                    requestedStartSourceSample = startSample,
                    requestedEndSourceSampleExclusive = requestedEndSample,
                    observedStartSourceSample = startSample,
                    observedEndSourceSampleExclusive = observedEnd,
                    observedSourceSampleCount = samples.size.toLong(),
                    endOfStreamReached = endTimeUs == null || forceEarlyEos,
                    requestedBoundaryReached = reachedBoundary,
                ),
            )
        }
    }

    private class TestAtomicOutput(
        private val target: File,
        private val afterCommit: () -> Unit = {},
    ) : TrackPcmCache.AtomicOutput {
        private val temporary = File(target.parentFile, ".${target.name}.test-atomic")
        private var stream: FileOutputStream? = FileOutputStream(temporary)

        override val channel: FileChannel
            get() = checkNotNull(stream).channel

        override fun commit() {
            val active = checkNotNull(stream)
            active.close()
            stream = null
            if (target.exists()) check(target.delete())
            check(temporary.renameTo(target))
            afterCommit()
        }

        override fun abort() {
            stream?.close()
            stream = null
            temporary.delete()
        }
    }

    private fun floatBytes(samples: FloatArray): ByteArray {
        val bytes = ByteArray(samples.size * Float.SIZE_BYTES)
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().put(samples)
        return bytes
    }

    private fun assertScratchAbsent(output: File) {
        val parent = requireNotNull(output.parentFile)
        assertFalse(File(parent, ".${output.name}.native-f32.scratch").exists())
        assertNull(parent.listFiles()?.singleOrNull {
            it.name.contains(output.name) && it.name.endsWith(".scratch")
        })
    }

    private fun sha256(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256")
        .digest(bytes)
        .joinToString("") { byte -> "%02x".format(byte.toInt() and 0xff) }

    private companion object {
        const val TARGET_RATE = 24_000
        const val SAMPLES_SCALE = 131_072f
    }
}
