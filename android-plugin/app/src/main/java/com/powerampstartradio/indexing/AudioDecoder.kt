package com.powerampstartradio.indexing

import android.media.AudioFormat
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.os.SystemClock
import android.util.Log
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.roundToInt

/**
 * Decodes audio files to PCM using Android's MediaCodec (hardware-accelerated),
 * then resamples to a target sample rate.
 *
 * Supports MP3, FLAC, M4A, OGG, WAV, and any format Android's MediaExtractor handles.
 */
class AudioDecoder {

    companion object {
        private const val TAG = "AudioDecoder"
        private const val TIMEOUT_US = 10_000L
        private const val NO_PROGRESS_TIMEOUT_MS = 15_000L
        private const val INITIAL_SAMPLE_CAPACITY = 240_000
    }

    open class AudioDecodeException(message: String, cause: Throwable? = null) :
        Exception(message, cause)

    class NoAudioStreamException(file: File) :
        AudioDecodeException("No audio stream found in ${file.absolutePath}")

    class UnsupportedPcmFormatException(message: String) : AudioDecodeException(message)

    class AudioDecodeTimeoutException(file: File) :
        AudioDecodeException("Decoder made no progress for ${NO_PROGRESS_TIMEOUT_MS}ms: ${file.name}")

    class AudioDecodeCancelledException : AudioDecodeException("Audio decode cancelled")

    class AudioDecodeBoundaryException(
        message: String,
        val evidence: DecodeBoundaryEvidence,
    ) : AudioDecodeException(message)

    class AudioDecodeFailedException(file: File, cause: Throwable) :
        AudioDecodeException("Failed to decode ${file.absolutePath}: ${cause.message}", cause)

    /**
     * Decoded audio result.
     *
     * @param samples Mono PCM samples in [-1, 1] range at [sampleRate] Hz
     * @param sampleRate Target sample rate the audio was resampled to
     * @param durationS Duration in seconds
     * @param decodeMs Wall-clock time for MediaCodec PCM decode (ms)
     * @param resampleMs Wall-clock time for soxr resample (ms)
     */
    data class DecodedAudio(
        val samples: FloatArray,
        val sampleRate: Int,
        val durationS: Float,
        val decodeMs: Long = 0,
        val resampleMs: Long = 0,
        val decoderName: String = "",
        val sourceChannelCount: Int = 0,
        val sourcePcmEncoding: Int = AudioFormat.ENCODING_INVALID,
        /** Native-rate coordinates observed from codec output, before optional resampling. */
        val boundaryEvidence: DecodeBoundaryEvidence? = null,
    )

    data class DecodeBoundaryEvidence(
        val requestedStartSourceSample: Long,
        val requestedEndSourceSampleExclusive: Long?,
        val observedStartSourceSample: Long,
        val observedEndSourceSampleExclusive: Long,
        val observedSourceSampleCount: Long,
        val endOfStreamReached: Boolean,
        val requestedBoundaryReached: Boolean,
    )

    data class NativePcmStreamResult(
        val sampleRate: Int,
        val decoderName: String,
        val sourceChannelCount: Int,
        val sourcePcmEncoding: Int,
        val decodeMs: Long,
        val boundaryEvidence: DecodeBoundaryEvidence,
    )

    /**
     * Decode an audio file to mono PCM at the given target sample rate.
     *
     * @param file Audio file to decode
     * @param targetSampleRate Desired output sample rate (e.g., 24000 for MERT/CLaMP3),
     *   or 0 to keep the decoder's native sample rate.
     * @param maxDurationS Maximum duration to decode in seconds (0 = unlimited).
     *   Caps at native sample rate before resampling.
     * @param startTimeS Start position in seconds (seeks to this position before decoding).
     *   Used for chunked decoding of long tracks.
     * @param startTimeMs Exact logical start in milliseconds. When supplied, this takes
     *   precedence over [startTimeS] and decoded PCM is trimmed by presentation timestamp.
     * @param maxDurationMs Exact logical duration in milliseconds. When supplied, this
     *   takes precedence over [maxDurationS].
     * @param startTimeUs Exact logical start in microseconds. When supplied, this takes
     *   precedence over millisecond and second values.
     * @param maxDurationUs Exact logical duration in microseconds. When supplied, this
     *   takes precedence over millisecond and second values.
     * @param allowEndOfStreamBeforeRequestedEnd Accept a shorter physical EOS as complete
     *   evidence for a bounded probe. This is reserved for ordinary whole-file chunking, where
     *   metadata duration is advisory; CUE boundaries must leave it false.
     * @return Decoded audio, or null on failure
     */
    fun decode(
        file: File,
        targetSampleRate: Int,
        maxDurationS: Int = 0,
        startTimeS: Int = 0,
        resampleQuality: Int = NativeResampler.QUALITY_HQ,
        startTimeMs: Long? = null,
        maxDurationMs: Long? = null,
        startTimeUs: Long? = null,
        maxDurationUs: Long? = null,
        allowEndOfStreamBeforeRequestedEnd: Boolean = false,
        isCancelled: () -> Boolean = { Thread.currentThread().isInterrupted },
    ): DecodedAudio? = try {
        decodeExact(
            file = file,
            targetSampleRate = targetSampleRate,
            maxDurationS = maxDurationS,
            startTimeS = startTimeS,
            resampleQuality = resampleQuality,
            startTimeMs = startTimeMs,
            maxDurationMs = maxDurationMs,
            startTimeUs = startTimeUs,
            maxDurationUs = maxDurationUs,
            allowEndOfStreamBeforeRequestedEnd = allowEndOfStreamBeforeRequestedEnd,
            isCancelled = isCancelled,
        )
    } catch (error: AudioDecodeException) {
        Log.e(TAG, error.message, error)
        null
    }

    /** Same decode contract as [decode], but preserves typed failure evidence. */
    fun decodeExact(
        file: File,
        targetSampleRate: Int,
        maxDurationS: Int = 0,
        startTimeS: Int = 0,
        resampleQuality: Int = NativeResampler.QUALITY_HQ,
        startTimeMs: Long? = null,
        maxDurationMs: Long? = null,
        startTimeUs: Long? = null,
        maxDurationUs: Long? = null,
        allowEndOfStreamBeforeRequestedEnd: Boolean = false,
        isCancelled: () -> Boolean = { Thread.currentThread().isInterrupted },
    ): DecodedAudio {
        val decodeStart = System.nanoTime()
        val extractor = MediaExtractor()
        var codec: MediaCodec? = null
        var codecStarted = false
        try {
            extractor.setDataSource(file.absolutePath)

            // Find the audio track
            val audioTrackIndex = findAudioTrack(extractor)
                ?: throw NoAudioStreamException(file)
            extractor.selectTrack(audioTrackIndex)

            val logicalStartUs = when {
                startTimeUs != null -> startTimeUs
                startTimeMs != null -> Math.multiplyExact(startTimeMs, 1000L)
                else -> Math.multiplyExact(startTimeS.toLong(), 1_000_000L)
            }.coerceAtLeast(0L)
            val logicalDurationUs = when {
                maxDurationUs != null -> maxDurationUs.coerceAtLeast(0L)
                maxDurationMs != null -> Math.multiplyExact(maxDurationMs.coerceAtLeast(0L), 1000L)
                maxDurationS > 0 -> Math.multiplyExact(maxDurationS.toLong(), 1_000_000L)
                else -> 0L
            }
            val logicalEndUs = if (logicalDurationUs > 0L) {
                Math.addExact(logicalStartUs, logicalDurationUs)
            } else null

            // Seek before the requested boundary, then trim decoded PCM by timestamps.
            // SEEK_TO_CLOSEST_SYNC can land after the boundary and permanently skip audio.
            if (logicalStartUs > 0L) {
                extractor.seekTo(
                    logicalStartUs,
                    MediaExtractor.SEEK_TO_PREVIOUS_SYNC,
                )
            }

            val format = extractor.getTrackFormat(audioTrackIndex)
            val mime = format.getString(MediaFormat.KEY_MIME)
                ?: throw UnsupportedPcmFormatException("Audio stream has no MIME type")
            val nativeSampleRate = format.getInteger(MediaFormat.KEY_SAMPLE_RATE)
            val channelCount = format.getInteger(MediaFormat.KEY_CHANNEL_COUNT)

            Log.d(TAG, "Decoding ${file.name}: $mime, ${nativeSampleRate}Hz, ${channelCount}ch")

            // Create and configure decoder
            codec = MediaCodec.createDecoderByType(mime)
            codec.configure(format, null, null, 0)
            codec.start()
            codecStarted = true
            val decoderName = codec.name

            // Decode samples (with optional cap to prevent OOM on very long files)
            val decodedPcm = decodeAllSamples(
                codec = codec,
                extractor = extractor,
                file = file,
                estimatedSampleRate = nativeSampleRate,
                requestedStartUs = logicalStartUs,
                requestedEndUs = logicalEndUs,
                allowEndOfStreamBeforeRequestedEnd = allowEndOfStreamBeforeRequestedEnd,
                isCancelled = isCancelled,
            )
            codec.stop()
            codecStarted = false
            codec.release()
            codec = null

            if (decodedPcm.samples.isEmpty()) {
                throw AudioDecodeException("No PCM samples decoded from ${file.name}")
            }

            val decodeMs = (System.nanoTime() - decodeStart) / 1_000_000
            Log.i(TAG, "TIMING: decode_pcm ${file.name} = ${decodeMs}ms " +
                "(${decodedPcm.samples.size} samples @ ${decodedPcm.sampleRate}Hz, " +
                "${decodedPcm.channelCount}ch, pcm=${decodedPcm.pcmEncoding})")

            val outputSampleRate = if (targetSampleRate > 0) targetSampleRate else decodedPcm.sampleRate

            // Resample if needed — use NEON polyphase FIR (200x faster than soxr)
            val resampleStart = System.nanoTime()
            val resampled = if (decodedPcm.sampleRate != outputSampleRate) {
                NativeMath.resamplePolyphase(
                    decodedPcm.samples,
                    decodedPcm.sampleRate,
                    outputSampleRate,
                ) ?: throw AudioDecodeException(
                    "Canonical resampler failed (${decodedPcm.sampleRate}->$outputSampleRate Hz)",
                )
            } else {
                decodedPcm.samples
            }
            val resampleMs = (System.nanoTime() - resampleStart) / 1_000_000

            val durationS = resampled.size.toFloat() / outputSampleRate
            Log.i(TAG, "TIMING: resample ${file.name} ${decodedPcm.sampleRate}->${outputSampleRate}Hz = ${resampleMs}ms")
            Log.i(TAG, "TIMING: decode_total ${file.name} = ${decodeMs + resampleMs}ms (${durationS}s audio)")

            return DecodedAudio(
                samples = resampled,
                sampleRate = outputSampleRate,
                durationS = durationS,
                decodeMs = decodeMs,
                resampleMs = resampleMs,
                decoderName = decoderName,
                sourceChannelCount = decodedPcm.channelCount,
                sourcePcmEncoding = decodedPcm.pcmEncoding,
                boundaryEvidence = decodedPcm.boundaryEvidence,
            )

        } catch (e: OutOfMemoryError) {
            Log.e(TAG, "OOM decoding ${file.name}", e)
            System.gc()
            throw AudioDecodeFailedException(file, e)
        } catch (e: AudioDecodeException) {
            throw e
        } catch (e: Exception) {
            throw AudioDecodeFailedException(file, e)
        } finally {
            codec?.let {
                if (codecStarted) try { it.stop() } catch (_: Exception) {}
                try { it.release() } catch (_: Exception) {}
            }
            extractor.release()
        }
    }

    /**
     * Decode one logical span through one extractor/codec session and emit bounded native-rate
     * mono PCM buffers. A null [endTimeUs] means physical EOS; a non-null end is an exact logical
     * half-open boundary.
     */
    fun decodeNativePcmStream(
        file: File,
        startTimeUs: Long,
        endTimeUs: Long?,
        isCancelled: () -> Boolean = { Thread.currentThread().isInterrupted },
        onChunk: (startSourceSample: Long, samples: FloatArray) -> Unit,
    ): NativePcmStreamResult {
        require(startTimeUs >= 0L) { "startTimeUs must be non-negative" }
        require(endTimeUs == null || endTimeUs > startTimeUs) {
            "endTimeUs must be after startTimeUs"
        }
        val decodeStart = System.nanoTime()
        val extractor = MediaExtractor()
        var codec: MediaCodec? = null
        var codecStarted = false
        try {
            extractor.setDataSource(file.absolutePath)
            val audioTrackIndex = findAudioTrack(extractor)
                ?: throw NoAudioStreamException(file)
            extractor.selectTrack(audioTrackIndex)
            if (startTimeUs > 0L) {
                extractor.seekTo(startTimeUs, MediaExtractor.SEEK_TO_PREVIOUS_SYNC)
            }
            val format = extractor.getTrackFormat(audioTrackIndex)
            val mime = format.getString(MediaFormat.KEY_MIME)
                ?: throw UnsupportedPcmFormatException("Audio stream has no MIME type")
            val nativeSampleRate = format.getInteger(MediaFormat.KEY_SAMPLE_RATE)

            codec = MediaCodec.createDecoderByType(mime)
            codec.configure(format, null, null, 0)
            codec.start()
            codecStarted = true
            val decoderName = codec.name
            val decoded = decodeAllSamples(
                codec = codec,
                extractor = extractor,
                file = file,
                estimatedSampleRate = nativeSampleRate,
                requestedStartUs = startTimeUs,
                requestedEndUs = endTimeUs,
                allowEndOfStreamBeforeRequestedEnd = false,
                isCancelled = isCancelled,
                onNativePcmChunk = onChunk,
            )
            codec.stop()
            codecStarted = false
            codec.release()
            codec = null
            val decodeMs = (System.nanoTime() - decodeStart) / 1_000_000L
            return NativePcmStreamResult(
                sampleRate = decoded.sampleRate,
                decoderName = decoderName,
                sourceChannelCount = decoded.channelCount,
                sourcePcmEncoding = decoded.pcmEncoding,
                decodeMs = decodeMs,
                boundaryEvidence = decoded.boundaryEvidence,
            )
        } catch (error: OutOfMemoryError) {
            System.gc()
            throw AudioDecodeFailedException(file, error)
        } catch (error: AudioDecodeException) {
            throw error
        } catch (error: Exception) {
            throw AudioDecodeFailedException(file, error)
        } finally {
            codec?.let {
                if (codecStarted) try { it.stop() } catch (_: Exception) {}
                try { it.release() } catch (_: Exception) {}
            }
            extractor.release()
        }
    }

    private fun findAudioTrack(extractor: MediaExtractor): Int? {
        for (i in 0 until extractor.trackCount) {
            val mime = extractor.getTrackFormat(i).getString(MediaFormat.KEY_MIME)
            if (mime?.startsWith("audio/") == true) return i
        }
        return null
    }

    private data class OutputPcmFormat(
        val sampleRate: Int,
        val channelCount: Int,
        val pcmEncoding: Int,
    ) {
        val bytesPerSample: Int
            get() = when (pcmEncoding) {
                AudioFormat.ENCODING_PCM_16BIT -> Short.SIZE_BYTES
                AudioFormat.ENCODING_PCM_FLOAT -> Float.SIZE_BYTES
                else -> throw UnsupportedPcmFormatException(
                    "Unsupported decoder PCM encoding $pcmEncoding",
                )
            }
    }

    private data class DecodedPcm(
        val samples: FloatArray,
        val sampleRate: Int,
        val channelCount: Int,
        val pcmEncoding: Int,
        val boundaryEvidence: DecodeBoundaryEvidence,
    )

    private class FloatAccumulator(initialCapacity: Int) {
        private var values = FloatArray(maxOf(initialCapacity, 1))
        var size: Int = 0
            private set

        fun appendInt16(
            buffer: ByteBuffer,
            offsetBytes: Int,
            sizeBytes: Int,
            channels: Int,
            frames: Int,
        ) {
            ensureCapacity(frames)
            val written = NativeMath.int16ToMonoFloat(
                buffer,
                offsetBytes,
                sizeBytes,
                channels,
                values,
                size,
                frames,
            )
            if (written != frames) {
                throw AudioDecodeException("PCM16 conversion wrote $written of $frames frames")
            }
            size += written
        }

        fun appendFloat(
            buffer: ByteBuffer,
            offsetBytes: Int,
            sizeBytes: Int,
            channels: Int,
            frames: Int,
        ) {
            ensureCapacity(frames)
            val duplicate = buffer.duplicate().order(ByteOrder.nativeOrder())
            duplicate.position(offsetBytes)
            duplicate.limit(offsetBytes + sizeBytes)
            val floats = duplicate.slice().order(ByteOrder.nativeOrder()).asFloatBuffer()
            repeat(frames) {
                var sum = 0f
                repeat(channels) { sum += floats.get() }
                values[size++] = sum / channels
            }
        }

        fun toFloatArray(): FloatArray = values.copyOf(size)

        private fun ensureCapacity(additional: Int) {
            require(additional >= 0)
            val required = size.toLong() + additional
            if (required > Int.MAX_VALUE) {
                throw AudioDecodeException("Decoded audio exceeds the in-memory sample limit")
            }
            if (required <= values.size) return
            var capacity = values.size.toLong()
            while (capacity < required) {
                capacity = minOf(Int.MAX_VALUE.toLong(), maxOf(capacity * 2L, required))
            }
            values = values.copyOf(capacity.toInt())
        }
    }

    /** Decode and downmix raw codec output while honoring its actual per-buffer format. */
    private fun decodeAllSamples(
        codec: MediaCodec,
        extractor: MediaExtractor,
        file: File,
        estimatedSampleRate: Int,
        requestedStartUs: Long,
        requestedEndUs: Long?,
        allowEndOfStreamBeforeRequestedEnd: Boolean,
        isCancelled: () -> Boolean,
        onNativePcmChunk: ((startSourceSample: Long, samples: FloatArray) -> Unit)? = null,
    ): DecodedPcm {
        val estimatedSamples = requestedEndUs?.let { endUs ->
            val start = AudioSampleTimeline.sampleAtOrAfter(requestedStartUs, estimatedSampleRate)
            val end = AudioSampleTimeline.sampleAtOrAfter(endUs, estimatedSampleRate)
            (end - start).coerceIn(1L, Int.MAX_VALUE.toLong()).toInt()
        } ?: INITIAL_SAMPLE_CAPACITY
        val accumulator = if (onNativePcmChunk == null) {
            FloatAccumulator(estimatedSamples)
        } else {
            null
        }
        val bufferInfo = MediaCodec.BufferInfo()
        var inputDone = false
        var activeFormat: OutputPcmFormat? = null
        var lastProgressNs = SystemClock.elapsedRealtimeNanos()
        var observedStartSample: Long? = null
        var observedEndSampleExclusive: Long? = null
        var observedSourceSampleCount = 0L
        var endOfStreamReached = false
        var requestedBoundaryReached = false
        val timelineAuthority = AudioDecodeBoundaryContract.timelineAuthority(
            requestedStartUs = requestedStartUs,
            requestedEndUs = requestedEndUs,
        )

        while (true) {
            AudioDecodeBoundaryContract.requireActive(isCancelled)
            var madeProgress = false

            if (!inputDone) {
                val inputIndex = codec.dequeueInputBuffer(TIMEOUT_US)
                if (inputIndex >= 0) {
                    val inputBuffer = codec.getInputBuffer(inputIndex)
                        ?: throw AudioDecodeException("Decoder returned a null input buffer")
                    val sampleSize = extractor.readSampleData(inputBuffer, 0)
                    if (sampleSize < 0) {
                        codec.queueInputBuffer(
                            inputIndex,
                            0,
                            0,
                            0,
                            MediaCodec.BUFFER_FLAG_END_OF_STREAM,
                        )
                        inputDone = true
                    } else {
                        codec.queueInputBuffer(inputIndex, 0, sampleSize, extractor.sampleTime, 0)
                        extractor.advance()
                    }
                    madeProgress = true
                }
            }

            val outputIndex = codec.dequeueOutputBuffer(bufferInfo, TIMEOUT_US)
            when {
                outputIndex >= 0 -> {
                    val bufferFormat = parseOutputFormat(codec.getOutputFormat(outputIndex))
                    activeFormat = acceptOutputFormat(
                        activeFormat,
                        bufferFormat,
                        observedSourceSampleCount,
                    )
                    val bytesPerFrame = Math.multiplyExact(
                        bufferFormat.bytesPerSample,
                        bufferFormat.channelCount,
                    )
                    if (bufferInfo.size % bytesPerFrame != 0) {
                        throw UnsupportedPcmFormatException(
                            "PCM buffer has ${bufferInfo.size} bytes, not whole $bytesPerFrame-byte frames",
                        )
                    }
                    val frameCount = bufferInfo.size / bytesPerFrame
                    val reportedBufferStartSample = AudioSampleTimeline.nearestSampleForTimestamp(
                        bufferInfo.presentationTimeUs,
                        bufferFormat.sampleRate,
                    )
                    val requestedStartSample = AudioSampleTimeline.sampleAtOrAfter(
                        requestedStartUs,
                        bufferFormat.sampleRate,
                    )
                    val requestedEndSample = requestedEndUs?.let {
                        AudioSampleTimeline.sampleAtOrAfter(it, bufferFormat.sampleRate)
                    }
                    val bufferStartSample = AudioDecodeBoundaryContract.outputBufferStartSample(
                        authority = timelineAuthority,
                        reportedStartSourceSample = reportedBufferStartSample,
                        previousEndSourceSampleExclusive = observedEndSampleExclusive,
                        requestedStartSourceSample = requestedStartSample,
                    )
                    val slice = AudioDecodeBoundaryContract.frameSlice(
                        bufferStartSourceSample = bufferStartSample,
                        bufferFrameCount = frameCount,
                        requestedStartSourceSample = requestedStartSample,
                        requestedEndSourceSampleExclusive = requestedEndSample,
                    )
                    val firstFrame = slice.firstFrame
                    val availableFrames = slice.frameCount
                    val outputBuffer = codec.getOutputBuffer(outputIndex)
                        ?: throw AudioDecodeException("Decoder returned a null output buffer")
                    if (availableFrames > 0 &&
                        bufferInfo.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG == 0
                    ) {
                        val previousEnd = observedEndSampleExclusive
                        if (previousEnd != null && previousEnd != slice.startSourceSample) {
                            throw AudioDecodeBoundaryException(
                                "Decoded PCM timeline is discontinuous: $previousEnd -> " +
                                    "${slice.startSourceSample}",
                                AudioDecodeBoundaryContract.evidence(
                                    requestedStartSourceSample = requestedStartSample,
                                    requestedEndSourceSampleExclusive = requestedEndSample,
                                    observedStartSourceSample = observedStartSample,
                                    observedEndSourceSampleExclusive = previousEnd,
                                    observedSourceSampleCount = observedSourceSampleCount,
                                    endOfStreamReached = endOfStreamReached,
                                    requestedBoundaryReached = requestedBoundaryReached,
                                ),
                            )
                        }
                        if (observedStartSample == null) {
                            observedStartSample = slice.startSourceSample
                        }
                        val offset = bufferInfo.offset + firstFrame * bytesPerFrame
                        val byteCount = availableFrames * bytesPerFrame
                        val destination = accumulator ?: FloatAccumulator(availableFrames)
                        when (bufferFormat.pcmEncoding) {
                            AudioFormat.ENCODING_PCM_16BIT -> destination.appendInt16(
                                outputBuffer,
                                offset,
                                byteCount,
                                bufferFormat.channelCount,
                                availableFrames,
                            )
                            AudioFormat.ENCODING_PCM_FLOAT -> destination.appendFloat(
                                outputBuffer,
                                offset,
                                byteCount,
                                bufferFormat.channelCount,
                                availableFrames,
                            )
                        }
                        observedSourceSampleCount = Math.addExact(
                            observedSourceSampleCount,
                            availableFrames.toLong(),
                        )
                        observedEndSampleExclusive = slice.endSourceSampleExclusive
                        if (onNativePcmChunk != null) {
                            onNativePcmChunk(slice.startSourceSample, destination.toFloatArray())
                        }
                    }
                    codec.releaseOutputBuffer(outputIndex, false)
                    madeProgress = true

                    val reachedRequestedEnd = requestedEndSample != null &&
                        bufferStartSample + frameCount >= requestedEndSample
                    if (reachedRequestedEnd) requestedBoundaryReached = true
                    if (bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) {
                        endOfStreamReached = true
                    }
                    if (endOfStreamReached || reachedRequestedEnd) {
                        break
                    }
                }
                outputIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                    val changed = parseOutputFormat(codec.outputFormat)
                    activeFormat = acceptOutputFormat(
                        activeFormat,
                        changed,
                        observedSourceSampleCount,
                    )
                    Log.d(TAG, "Output format changed: ${codec.outputFormat}")
                    madeProgress = true
                }
            }

            val nowNs = SystemClock.elapsedRealtimeNanos()
            if (madeProgress) {
                lastProgressNs = nowNs
            } else if ((nowNs - lastProgressNs) / 1_000_000L >= NO_PROGRESS_TIMEOUT_MS) {
                throw AudioDecodeTimeoutException(file)
            }
        }

        val format = activeFormat
            ?: throw UnsupportedPcmFormatException("Decoder produced no output format")
        val requestedStartSample = AudioSampleTimeline.sampleAtOrAfter(
            requestedStartUs,
            format.sampleRate,
        )
        val requestedEndSample = requestedEndUs?.let { endUs ->
            AudioSampleTimeline.sampleAtOrAfter(endUs, format.sampleRate)
        }
        val boundaryEvidence = AudioDecodeBoundaryContract.requireComplete(
            requestedStartSourceSample = requestedStartSample,
            requestedEndSourceSampleExclusive = requestedEndSample,
            observedStartSourceSample = observedStartSample,
            observedEndSourceSampleExclusive = observedEndSampleExclusive,
            observedSourceSampleCount = observedSourceSampleCount,
            endOfStreamReached = endOfStreamReached,
            requestedBoundaryReached = requestedBoundaryReached,
            allowEndOfStreamBeforeRequestedEnd = allowEndOfStreamBeforeRequestedEnd,
        )
        return DecodedPcm(
            samples = accumulator?.toFloatArray() ?: FloatArray(0),
            sampleRate = format.sampleRate,
            channelCount = format.channelCount,
            pcmEncoding = format.pcmEncoding,
            boundaryEvidence = boundaryEvidence,
        )
    }

    private fun parseOutputFormat(format: MediaFormat): OutputPcmFormat {
        val sampleRate = format.getInteger(MediaFormat.KEY_SAMPLE_RATE)
        val channelCount = format.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
        val pcmEncoding = if (format.containsKey(MediaFormat.KEY_PCM_ENCODING)) {
            format.getInteger(MediaFormat.KEY_PCM_ENCODING)
        } else {
            AudioFormat.ENCODING_PCM_16BIT
        }
        if (sampleRate <= 0 || channelCount <= 0) {
            throw UnsupportedPcmFormatException(
                "Invalid decoder output format: rate=$sampleRate channels=$channelCount",
            )
        }
        if (pcmEncoding != AudioFormat.ENCODING_PCM_16BIT &&
            pcmEncoding != AudioFormat.ENCODING_PCM_FLOAT
        ) {
            throw UnsupportedPcmFormatException(
                "Unsupported decoder PCM encoding $pcmEncoding",
            )
        }
        return OutputPcmFormat(sampleRate, channelCount, pcmEncoding)
    }

    private fun acceptOutputFormat(
        current: OutputPcmFormat?,
        candidate: OutputPcmFormat,
        samplesWritten: Long,
    ): OutputPcmFormat {
        if (current == null || current == candidate) return candidate
        if (samplesWritten == 0L) return candidate
        throw UnsupportedPcmFormatException(
            "Decoder output changed mid-stream from $current to $candidate",
        )
    }

    /**
     * Resample audio using libsoxr (native) for high-quality anti-aliased conversion.
     */
    fun resample(
        samples: FloatArray,
        fromRate: Int,
        toRate: Int,
        quality: Int = NativeResampler.QUALITY_HQ,
    ): FloatArray {
        if (fromRate == toRate) return samples
        return NativeResampler.resample(samples, fromRate, toRate, quality)
            ?: throw IllegalStateException("soxr resampling failed ($fromRate -> $toRate Hz, ${samples.size} samples)")
    }
}

internal object AudioDecodeBoundaryContract {
    enum class TimelineAuthority {
        /**
         * A physical whole-file decode is one ordered PCM stream. Android's gapless
         * SkipCutBuffer can move PCM between output callbacks without moving their input-derived
         * presentation timestamps, so callback order and frame counts are authoritative.
         */
        PHYSICAL_DECODE_ORDER,

        /** A bounded or seeked span must remain anchored to media presentation coordinates. */
        PRESENTATION_TIMESTAMPS,
    }

    data class FrameSlice(
        val firstFrame: Int,
        val frameCount: Int,
        val startSourceSample: Long,
        val endSourceSampleExclusive: Long,
    )

    fun requireActive(isCancelled: () -> Boolean) {
        if (isCancelled()) throw AudioDecoder.AudioDecodeCancelledException()
    }

    fun timelineAuthority(
        requestedStartUs: Long,
        requestedEndUs: Long?,
    ): TimelineAuthority = if (requestedStartUs == 0L && requestedEndUs == null) {
        TimelineAuthority.PHYSICAL_DECODE_ORDER
    } else {
        TimelineAuthority.PRESENTATION_TIMESTAMPS
    }

    fun outputBufferStartSample(
        authority: TimelineAuthority,
        reportedStartSourceSample: Long,
        previousEndSourceSampleExclusive: Long?,
        requestedStartSourceSample: Long,
    ): Long = when (authority) {
        TimelineAuthority.PHYSICAL_DECODE_ORDER ->
            previousEndSourceSampleExclusive ?: requestedStartSourceSample
        TimelineAuthority.PRESENTATION_TIMESTAMPS -> reportedStartSourceSample
    }

    fun frameSlice(
        bufferStartSourceSample: Long,
        bufferFrameCount: Int,
        requestedStartSourceSample: Long,
        requestedEndSourceSampleExclusive: Long?,
    ): FrameSlice {
        require(bufferStartSourceSample >= 0L) { "buffer start must be non-negative" }
        require(bufferFrameCount >= 0) { "buffer frame count must be non-negative" }
        require(requestedStartSourceSample >= 0L) { "requested start must be non-negative" }
        require(requestedEndSourceSampleExclusive == null ||
            requestedEndSourceSampleExclusive >= requestedStartSourceSample
        ) { "requested end precedes start" }

        val bufferEnd = Math.addExact(bufferStartSourceSample, bufferFrameCount.toLong())
        val sliceStart = maxOf(bufferStartSourceSample, requestedStartSourceSample)
            .coerceAtMost(bufferEnd)
        val sliceEnd = minOf(
            bufferEnd,
            requestedEndSourceSampleExclusive ?: bufferEnd,
        ).coerceAtLeast(sliceStart)
        return FrameSlice(
            firstFrame = (sliceStart - bufferStartSourceSample).toInt(),
            frameCount = (sliceEnd - sliceStart).toInt(),
            startSourceSample = sliceStart,
            endSourceSampleExclusive = sliceEnd,
        )
    }

    fun evidence(
        requestedStartSourceSample: Long,
        requestedEndSourceSampleExclusive: Long?,
        observedStartSourceSample: Long?,
        observedEndSourceSampleExclusive: Long?,
        observedSourceSampleCount: Long,
        endOfStreamReached: Boolean,
        requestedBoundaryReached: Boolean,
    ): AudioDecoder.DecodeBoundaryEvidence {
        val observedStart = observedStartSourceSample ?: requestedStartSourceSample
        val observedEnd = observedEndSourceSampleExclusive ?: observedStart
        return AudioDecoder.DecodeBoundaryEvidence(
            requestedStartSourceSample = requestedStartSourceSample,
            requestedEndSourceSampleExclusive = requestedEndSourceSampleExclusive,
            observedStartSourceSample = observedStart,
            observedEndSourceSampleExclusive = observedEnd,
            observedSourceSampleCount = observedSourceSampleCount,
            endOfStreamReached = endOfStreamReached,
            requestedBoundaryReached = requestedBoundaryReached,
        )
    }

    fun requireComplete(
        requestedStartSourceSample: Long,
        requestedEndSourceSampleExclusive: Long?,
        observedStartSourceSample: Long?,
        observedEndSourceSampleExclusive: Long?,
        observedSourceSampleCount: Long,
        endOfStreamReached: Boolean,
        requestedBoundaryReached: Boolean,
        allowEndOfStreamBeforeRequestedEnd: Boolean = false,
    ): AudioDecoder.DecodeBoundaryEvidence {
        val result = evidence(
            requestedStartSourceSample = requestedStartSourceSample,
            requestedEndSourceSampleExclusive = requestedEndSourceSampleExclusive,
            observedStartSourceSample = observedStartSourceSample,
            observedEndSourceSampleExclusive = observedEndSourceSampleExclusive,
            observedSourceSampleCount = observedSourceSampleCount,
            endOfStreamReached = endOfStreamReached,
            requestedBoundaryReached = requestedBoundaryReached,
        )
        val coordinateCount = result.observedEndSourceSampleExclusive -
            result.observedStartSourceSample
        val expectedCount = requestedEndSourceSampleExclusive?.let {
            it - requestedStartSourceSample
        }
        val complete = result.observedStartSourceSample == requestedStartSourceSample &&
            coordinateCount == observedSourceSampleCount && when {
            requestedEndSourceSampleExclusive == null -> endOfStreamReached
            requestedBoundaryReached ->
                result.observedEndSourceSampleExclusive == requestedEndSourceSampleExclusive &&
                    observedSourceSampleCount == expectedCount
            allowEndOfStreamBeforeRequestedEnd && endOfStreamReached ->
                result.observedEndSourceSampleExclusive <= requestedEndSourceSampleExclusive
            else -> false
        }
        if (!complete) {
            throw AudioDecoder.AudioDecodeBoundaryException(
                "Decoded boundary is incomplete: requested=[$requestedStartSourceSample," +
                    "${requestedEndSourceSampleExclusive ?: "EOS"}) observed=" +
                    "[${result.observedStartSourceSample}," +
                    "${result.observedEndSourceSampleExclusive}) count=$observedSourceSampleCount " +
                    "eos=$endOfStreamReached boundary=$requestedBoundaryReached",
                result,
            )
        }
        return result
    }
}
