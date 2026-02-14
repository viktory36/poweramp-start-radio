package com.powerampstartradio.indexing

import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
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
    }

    /**
     * Decoded audio result.
     *
     * @param samples Mono PCM samples in [-1, 1] range at [sampleRate] Hz
     * @param sampleRate Target sample rate the audio was resampled to
     * @param durationS Duration in seconds
     */
    data class DecodedAudio(
        val samples: FloatArray,
        val sampleRate: Int,
        val durationS: Float
    )

    /**
     * Decode an audio file to mono PCM at the given target sample rate.
     *
     * @param file Audio file to decode
     * @param targetSampleRate Desired output sample rate (e.g., 24000 for MuQ-MuLan, 16000 for Flamingo)
     * @return Decoded audio, or null on failure
     */
    fun decode(file: File, targetSampleRate: Int): DecodedAudio? {
        val extractor = MediaExtractor()
        try {
            extractor.setDataSource(file.absolutePath)

            // Find the audio track
            val audioTrackIndex = findAudioTrack(extractor) ?: run {
                Log.w(TAG, "No audio track found in ${file.name}")
                return null
            }
            extractor.selectTrack(audioTrackIndex)

            val format = extractor.getTrackFormat(audioTrackIndex)
            val mime = format.getString(MediaFormat.KEY_MIME) ?: return null
            val nativeSampleRate = format.getInteger(MediaFormat.KEY_SAMPLE_RATE)
            val channelCount = format.getInteger(MediaFormat.KEY_CHANNEL_COUNT)

            Log.d(TAG, "Decoding ${file.name}: $mime, ${nativeSampleRate}Hz, ${channelCount}ch")

            // Create and configure decoder
            val codec = MediaCodec.createDecoderByType(mime)
            codec.configure(format, null, null, 0)
            codec.start()

            // Decode all samples
            val rawSamples = decodeAllSamples(codec, extractor, channelCount)
            codec.stop()
            codec.release()

            if (rawSamples.isEmpty()) {
                Log.w(TAG, "No samples decoded from ${file.name}")
                return null
            }

            // Resample if needed
            val resampled = if (nativeSampleRate != targetSampleRate) {
                resample(rawSamples, nativeSampleRate, targetSampleRate)
            } else {
                rawSamples
            }

            val durationS = resampled.size.toFloat() / targetSampleRate
            Log.d(TAG, "Decoded ${file.name}: ${resampled.size} samples, ${durationS}s @ ${targetSampleRate}Hz")

            return DecodedAudio(resampled, targetSampleRate, durationS)

        } catch (e: Exception) {
            Log.e(TAG, "Failed to decode ${file.name}", e)
            return null
        } finally {
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

    /**
     * Decode all PCM samples from the codec, downmixing to mono.
     * Returns float samples in [-1, 1] range.
     */
    private fun decodeAllSamples(
        codec: MediaCodec,
        extractor: MediaExtractor,
        channelCount: Int
    ): FloatArray {
        val bufferInfo = MediaCodec.BufferInfo()
        var inputDone = false
        val allSamples = mutableListOf<Float>()

        while (true) {
            // Feed input
            if (!inputDone) {
                val inputIndex = codec.dequeueInputBuffer(TIMEOUT_US)
                if (inputIndex >= 0) {
                    val inputBuffer = codec.getInputBuffer(inputIndex)!!
                    val sampleSize = extractor.readSampleData(inputBuffer, 0)
                    if (sampleSize < 0) {
                        codec.queueInputBuffer(inputIndex, 0, 0, 0,
                            MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                        inputDone = true
                    } else {
                        codec.queueInputBuffer(inputIndex, 0, sampleSize,
                            extractor.sampleTime, 0)
                        extractor.advance()
                    }
                }
            }

            // Read output
            val outputIndex = codec.dequeueOutputBuffer(bufferInfo, TIMEOUT_US)
            if (outputIndex >= 0) {
                val outputBuffer = codec.getOutputBuffer(outputIndex)!!
                outputBuffer.order(ByteOrder.LITTLE_ENDIAN)

                // MediaCodec outputs 16-bit PCM by default
                val sampleCount = bufferInfo.size / 2  // 2 bytes per 16-bit sample
                val frameCount = sampleCount / channelCount

                for (frame in 0 until frameCount) {
                    var monoSample = 0f
                    for (ch in 0 until channelCount) {
                        val sample = outputBuffer.getShort().toFloat() / 32768f
                        monoSample += sample
                    }
                    allSamples.add(monoSample / channelCount)
                }

                codec.releaseOutputBuffer(outputIndex, false)

                if (bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) {
                    break
                }
            } else if (outputIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                // Format changed, continue
                Log.d(TAG, "Output format changed: ${codec.outputFormat}")
            }
        }

        return allSamples.toFloatArray()
    }

    /**
     * Resample audio using linear interpolation.
     *
     * Linear interpolation is adequate for embedding quality — these models
     * were trained on web audio of varying quality. For 44.1kHz/48kHz -> 16kHz/24kHz
     * downsampling, the aliasing artifacts are above the model's effective bandwidth.
     */
    fun resample(samples: FloatArray, fromRate: Int, toRate: Int): FloatArray {
        if (fromRate == toRate) return samples

        val ratio = fromRate.toDouble() / toRate
        val outputLength = (samples.size / ratio).roundToInt()
        val output = FloatArray(outputLength)

        for (i in 0 until outputLength) {
            val srcPos = i * ratio
            val srcIndex = srcPos.toInt()
            val frac = (srcPos - srcIndex).toFloat()

            output[i] = if (srcIndex + 1 < samples.size) {
                samples[srcIndex] * (1f - frac) + samples[srcIndex + 1] * frac
            } else {
                samples[srcIndex.coerceAtMost(samples.size - 1)]
            }
        }

        return output
    }
}
