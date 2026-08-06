package com.powerampstartradio.indexing.v2

import com.powerampstartradio.indexing.AudioDecoder
import com.powerampstartradio.indexing.TrackPcmCache
import java.io.FileNotFoundException
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

class V2IndexingFailureClassifierTest {
    @Test
    fun `classifier inspects wrapped causes before generic wrappers`() {
        assertCode(
            TrackFailureCode.OUT_OF_MEMORY,
            IllegalStateException("wrapper", OutOfMemoryError("allocation")),
        )
        assertCode(
            TrackFailureCode.ANDROID_AUDIO_PERMISSION_DENIED,
            IllegalStateException("wrapper", SecurityException("revoked")),
        )
        assertCode(
            TrackFailureCode.SOURCE_MISSING,
            IllegalStateException("wrapper", FileNotFoundException("gone")),
        )
    }

    @Test
    fun `pause and cancellation never become failures at any executor stage boundary`() {
        IndexingStage.entries.forEach { stage ->
            assertThrows("stage=$stage", V2IndexingControlFlowException::class.java) {
                V2ExecutorFailureBoundary.classifyOrRethrow(
                    IllegalStateException(
                        "wrapper",
                        V2IndexingControlFlowException("pause requested"),
                    ),
                    stage,
                    span(),
                )
            }
            assertThrows("decode cancellation at stage=$stage",
                V2IndexingControlFlowException::class.java) {
                V2ExecutorFailureBoundary.classifyOrRethrow(
                    IllegalStateException(
                        "wrapper",
                        AudioDecoder.AudioDecodeCancelledException(),
                    ),
                    stage,
                    span(),
                )
            }
        }
    }

    @Test
    fun `typed PCM evidence distinguishes EOS and artifact failures`() {
        assertCode(
            TrackFailureCode.CONTAINER_EOS_MISMATCH,
            TrackPcmCache.PcmContractException(
                TrackPcmCache.PcmContractFailure.EOS_MISMATCH,
                "codec EOS missing",
            ),
        )
        assertCode(
            TrackFailureCode.PARTIAL_ARTIFACT,
            TrackPcmCache.PcmContractException(
                TrackPcmCache.PcmContractFailure.PCM_ARTIFACT_MISMATCH,
                "short cache",
            ),
        )
        assertCode(
            TrackFailureCode.INSUFFICIENT_AUDIO_SIGNAL,
            TrackPcmCache.PcmContractException(
                TrackPcmCache.PcmContractFailure.INSUFFICIENT_AUDIO_SIGNAL,
                "effectively silent",
            ),
        )
    }

    private fun assertCode(expected: TrackFailureCode, error: Throwable) {
        assertEquals(
            expected,
            V2IndexingFailureClassifier.classify(
                error,
                IndexingStage.DECODE_AND_MERT,
                span(),
            ).code,
        )
    }

    private fun span(): FinalizedAudioSpanEvidence = FinalizedAudioSpanEvidence(
        kind = V2ResolvedAudioSpanKind.WHOLE_FILE,
        authority = V2AudioSpanAuthority.PROVISIONAL_END_OF_STREAM,
        executionBoundaryRequirement =
            V2ExecutionBoundaryRequirement.VERIFY_END_OF_STREAM_AND_RECONCILE,
        providerSpan = V2ProviderSpanEvidence(0L, 10_000_000L, 10_000_000L),
        cueClassification = V2CueClassificationEvidence(
            providerGroupRowCount = 1,
            logicalRowCount = 1,
            nonZeroOffsetRowIds = emptyList(),
            rawSourceImageRowIds = emptyList(),
        ),
        container = V2AudioContainerEvidence(
            physicalPath = "/music/test.flac",
            audioTrackIndex = 0,
            durationUsEstimate = 10_000_000L,
            sampleRateHz = 48_000,
            channelCount = 2,
            mime = "audio/flac",
        ),
        startUs = 0L,
        endExclusiveUs = 10_000_000L,
        startSourceSample = 0L,
        endSourceSampleExclusive = 480_000L,
        sourceSampleCount = 480_000L,
        exactSampleCount24k = 240_000L,
        expectedWork = V2AudioSpanMath.expectedWorkFor24kSamples(240_000L),
    )
}
