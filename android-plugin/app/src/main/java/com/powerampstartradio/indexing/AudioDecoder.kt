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
        private const val DECODE_CHUNK_SIZE = 240_000  // ~10s at 24kHz, ~960KB per chunk
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

        // Use primitive FloatArray chunks instead of MutableList<Float> to avoid
        // boxing overhead (~16 bytes/Float vs 4 bytes). A 5-min track at 44100Hz
        // stereo → ~13M mono samples = 52MB as FloatArray vs ~208MB as List<Float>.
        val chunks = mutableListOf<FloatArray>()
        var currentChunk = FloatArray(DECODE_CHUNK_SIZE)
        var chunkPos = 0

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
                    currentChunk[chunkPos++] = monoSample / channelCount
                    if (chunkPos == DECODE_CHUNK_SIZE) {
                        chunks.add(currentChunk)
                        currentChunk = FloatArray(DECODE_CHUNK_SIZE)
                        chunkPos = 0
                    }
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

        // Combine chunks into a single array
        val totalSize = chunks.size * DECODE_CHUNK_SIZE + chunkPos
        val result = FloatArray(totalSize)
        var offset = 0
        for (chunk in chunks) {
            chunk.copyInto(result, offset)
            offset += DECODE_CHUNK_SIZE
        }
        currentChunk.copyInto(result, offset, 0, chunkPos)

        return result
    }

    /**
     * Resample audio using libsoxr (native) for high-quality anti-aliased conversion.
     *
     * Flamingo/Whisper is extremely sensitive to resampling quality — a 0.001 mel
     * cosine difference from aliasing causes 0.40 embedding cosine degradation.
     * Only soxr-quality resampling produces correct embeddings (0.95+ cosine vs DB).
     * All other approaches (linear, sinc, Kaiser FIR, scipy polyphase) give ~0.55.
     */
    fun resample(samples: FloatArray, fromRate: Int, toRate: Int): FloatArray {
        if (fromRate == toRate) return samples
        return NativeResampler.resample(samples, fromRate, toRate)
            ?: throw IllegalStateException("soxr resampling failed ($fromRate -> $toRate Hz, ${samples.size} samples)")
    }
}
