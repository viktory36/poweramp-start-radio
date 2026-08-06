package com.powerampstartradio.indexing.v2

import android.media.MediaExtractor
import android.media.MediaFormat
import java.io.File

/** Production container inspector. Span policy stays in the pure [V2AudioSpanResolver]. */
class V2MediaExtractorAudioInspector : V2AudioContainerInspector {
    override fun inspect(physicalPath: String): V2AudioContainerEvidence {
        val source = File(physicalPath)
        if (!source.isFile || !source.canRead()) {
            throw V2AudioContainerInspectionException(
                V2AudioContainerInspectionFailureCode.SOURCE_UNREADABLE,
                "Audio source is not a readable file: $physicalPath",
            )
        }

        val extractor = MediaExtractor()
        try {
            try {
                extractor.setDataSource(source.absolutePath)
            } catch (error: SecurityException) {
                throw V2AudioContainerInspectionException(
                    V2AudioContainerInspectionFailureCode.SOURCE_UNREADABLE,
                    "Audio source cannot be opened: $physicalPath",
                    error,
                )
            } catch (error: Exception) {
                throw V2AudioContainerInspectionException(
                    V2AudioContainerInspectionFailureCode.UNSUPPORTED_OR_INVALID_CONTAINER,
                    "Android cannot inspect the audio container: $physicalPath",
                    error,
                )
            }
            var audioTrackIndex = -1
            var audioFormat: MediaFormat? = null
            for (trackIndex in 0 until extractor.trackCount) {
                val candidate = extractor.getTrackFormat(trackIndex)
                val mime = candidate.getString(MediaFormat.KEY_MIME)
                if (mime?.startsWith("audio/") == true) {
                    audioTrackIndex = trackIndex
                    audioFormat = candidate
                    break
                }
            }
            val format = audioFormat ?: throw V2AudioContainerInspectionException(
                V2AudioContainerInspectionFailureCode.NO_AUDIO_STREAM,
                "No audio stream in $physicalPath",
            )
            if (!format.containsKey(MediaFormat.KEY_SAMPLE_RATE)) {
                throw V2AudioContainerInspectionException(
                    V2AudioContainerInspectionFailureCode.UNSUPPORTED_OR_INVALID_CONTAINER,
                    "Audio stream has no sample rate in $physicalPath",
                )
            }
            if (!format.containsKey(MediaFormat.KEY_CHANNEL_COUNT)) {
                throw V2AudioContainerInspectionException(
                    V2AudioContainerInspectionFailureCode.UNSUPPORTED_OR_INVALID_CONTAINER,
                    "Audio stream has no channel count in $physicalPath",
                )
            }

            val durationUs = if (format.containsKey(MediaFormat.KEY_DURATION)) {
                format.getLong(MediaFormat.KEY_DURATION).coerceAtLeast(0L)
            } else {
                0L
            }
            return V2AudioContainerEvidence(
                physicalPath = physicalPath,
                audioTrackIndex = audioTrackIndex,
                durationUsEstimate = durationUs,
                durationEstimateSource = if (durationUs > 0L) {
                    V2DurationEstimateSource.CONTAINER_METADATA
                } else {
                    V2DurationEstimateSource.UNAVAILABLE
                },
                sampleRateHz = format.getInteger(MediaFormat.KEY_SAMPLE_RATE),
                channelCount = format.getInteger(MediaFormat.KEY_CHANNEL_COUNT),
                mime = requireNotNull(format.getString(MediaFormat.KEY_MIME)),
            )
        } finally {
            extractor.release()
        }
    }
}
