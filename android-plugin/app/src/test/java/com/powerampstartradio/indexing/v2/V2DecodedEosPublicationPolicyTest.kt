package com.powerampstartradio.indexing.v2

import com.powerampstartradio.indexing.TrackPcmCache
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.assertThrows
import org.junit.Test

class V2DecodedEosPublicationPolicyTest {
    @Test
    fun `near-exact decoded EOS remains authoritative`() {
        val finalized = V2DecodedEosSpanFinalizer.finalize(
            provisionalSpan(declaredUs = 160_200_000L),
            evidence(observedUs = 160_068_000L),
        )

        assertEquals(V2AudioSpanAuthority.DECODED_END_OF_STREAM, finalized.authority)
        assertEquals(160_068_000L, finalized.endExclusiveUs)
    }

    @Test
    fun `unknown duration accepts physical EOS`() {
        val finalized = V2DecodedEosSpanFinalizer.finalize(
            provisionalSpan(
                declaredUs = 0L,
                estimateSource = V2DurationEstimateSource.UNAVAILABLE,
            ),
            evidence(observedUs = 24_363_000L),
        )

        assertEquals(24_363_000L, finalized.endExclusiveUs)
    }

    @Test
    fun `subsecond decoded EOS is a typed minimum-duration failure`() {
        val span = provisionalSpan(
            declaredUs = 0L,
            estimateSource = V2DurationEstimateSource.UNAVAILABLE,
        )
        val error = assertThrows(TrackPcmCache.PcmContractException::class.java) {
            V2DecodedEosSpanFinalizer.finalize(span, evidence(observedUs = 500_000L))
        }

        assertEquals(TrackPcmCache.PcmContractFailure.BELOW_MINIMUM_DURATION, error.reason)
        assertEquals(
            TrackFailureCode.BELOW_MINIMUM_DURATION,
            V2IndexingFailureClassifier.classify(
                error,
                IndexingStage.DECODE_AND_MERT,
                span,
            ).code,
        )
    }

    @Test
    fun `decoded overrun remains authoritative`() {
        val finalized = V2DecodedEosSpanFinalizer.finalize(
            provisionalSpan(declaredUs = 160_200_000L),
            evidence(observedUs = 161_000_000L),
        )

        assertEquals(161_000_000L, finalized.endExclusiveUs)
    }

    @Test
    fun `gross decoded shortfall is a typed container EOS mismatch`() {
        val error = assertThrows(TrackPcmCache.PcmContractException::class.java) {
            V2DecodedEosSpanFinalizer.finalize(
                provisionalSpan(declaredUs = 160_200_000L),
                evidence(observedUs = 24_363_000L),
            )
        }

        assertEquals(TrackPcmCache.PcmContractFailure.EOS_MISMATCH, error.reason)
        assertTrue(error.message.orEmpty().contains("24.363 s"))
        assertTrue(error.message.orEmpty().contains("160.200 s"))
        assertTrue(error.message.orEmpty().contains("15.2% decoded"))
        assertTrue(error.message.orEmpty().contains("corrupt or truncated audio"))
        assertEquals(
            TrackFailureCode.CONTAINER_EOS_MISMATCH,
            V2IndexingFailureClassifier.classify(
                error,
                IndexingStage.DECODE_AND_MERT,
                provisionalSpan(declaredUs = 160_200_000L),
            ).code,
        )
    }

    @Test
    fun `provider fallback estimate remains advisory`() {
        val finalized = V2DecodedEosSpanFinalizer.finalize(
            provisionalSpan(
                declaredUs = 160_200_000L,
                estimateSource = V2DurationEstimateSource.PROVIDER_SPAN_FALLBACK,
            ),
            evidence(observedUs = 24_363_000L),
        )

        assertEquals(24_363_000L, finalized.endExclusiveUs)
    }

    @Test
    fun `cached gross shortfall is rejected when receipt evidence is replayed`() {
        val evidence = evidence(observedUs = 24_363_000L)
        val finalized = decodedSpan(
            provisionalSpan(declaredUs = 160_200_000L),
            evidence,
        )

        val error = assertThrows(TrackPcmCache.PcmContractException::class.java) {
            V2DecodedEosSpanFinalizer.finalize(finalized, evidence)
        }

        assertEquals(TrackPcmCache.PcmContractFailure.EOS_MISMATCH, error.reason)
    }

    @Test
    fun `already-finalized MERT work is rejected by model publication defense`() {
        val evidence = evidence(observedUs = 24_363_000L)
        val finalized = decodedSpan(
            provisionalSpan(declaredUs = 160_200_000L),
            evidence,
        )

        val error = assertThrows(TrackPcmCache.PcmContractException::class.java) {
            V2DecodedEosPublicationPolicy.requirePublishable(finalized)
        }

        assertEquals(TrackPcmCache.PcmContractFailure.EOS_MISMATCH, error.reason)
    }

    private fun provisionalSpan(
        declaredUs: Long,
        estimateSource: V2DurationEstimateSource = V2DurationEstimateSource.CONTAINER_METADATA,
    ): FinalizedAudioSpanEvidence {
        val unresolved = declaredUs == 0L
        val sourceSamples = if (unresolved) 0L else samplesFor(declaredUs)
        val targetSamples = if (unresolved) 0L else targetSamplesFor(sourceSamples)
        return FinalizedAudioSpanEvidence(
            kind = V2ResolvedAudioSpanKind.WHOLE_FILE,
            authority = V2AudioSpanAuthority.PROVISIONAL_END_OF_STREAM,
            executionBoundaryRequirement =
                V2ExecutionBoundaryRequirement.VERIFY_END_OF_STREAM_AND_RECONCILE,
            providerSpan = V2ProviderSpanEvidence(0L, declaredUs, declaredUs),
            cueClassification = V2CueClassificationEvidence(1, 1, emptyList(), emptyList()),
            container = V2AudioContainerEvidence(
                physicalPath = PATH,
                audioTrackIndex = 0,
                durationUsEstimate = declaredUs,
                durationEstimateSource = estimateSource,
                sampleRateHz = SOURCE_RATE,
                channelCount = 2,
                mime = "audio/flac",
            ),
            startUs = 0L,
            endExclusiveUs = declaredUs,
            startSourceSample = 0L,
            endSourceSampleExclusive = sourceSamples,
            sourceSampleCount = sourceSamples,
            exactSampleCount24k = targetSamples,
            expectedWork = if (unresolved) {
                V2UnknownDurationOrdinarySpanPolicy.unresolvedWork
            } else {
                V2AudioSpanMath.expectedWorkFor24kSamples(targetSamples)
            },
        )
    }

    private fun evidence(observedUs: Long): V2DecodedEosEvidence {
        val sourceSamples = samplesFor(observedUs)
        return V2DecodedEosEvidence(
            sourceSampleRateHz = SOURCE_RATE,
            observedStartSourceSample = 0L,
            observedEndSourceSampleExclusive = sourceSamples,
            observedSourceSampleCount = sourceSamples,
            exactSampleCount24k = targetSamplesFor(sourceSamples),
            endOfStreamReached = true,
        )
    }

    private fun decodedSpan(
        provisional: FinalizedAudioSpanEvidence,
        evidence: V2DecodedEosEvidence,
    ): FinalizedAudioSpanEvidence = provisional.copy(
        authority = V2AudioSpanAuthority.DECODED_END_OF_STREAM,
        endExclusiveUs = V2AudioSpanMath.canonicalTimeUsForSampleBoundary(
            evidence.observedEndSourceSampleExclusive,
            evidence.sourceSampleRateHz,
        ),
        endSourceSampleExclusive = evidence.observedEndSourceSampleExclusive,
        sourceSampleCount = evidence.observedSourceSampleCount,
        exactSampleCount24k = evidence.exactSampleCount24k,
        expectedWork = V2AudioSpanMath.expectedWorkFor24kSamples(evidence.exactSampleCount24k),
    )

    private fun samplesFor(durationUs: Long): Long =
        V2AudioSpanMath.sampleAtOrAfter(durationUs, SOURCE_RATE)

    private fun targetSamplesFor(sourceSamples: Long): Long = V2AudioSpanMath.resampledLength(
        sourceSamples,
        SOURCE_RATE,
        V2AudioSpanMath.TARGET_SAMPLE_RATE_HZ,
    )

    private companion object {
        const val PATH = "/storage/emulated/0/Music/test.flac"
        const val SOURCE_RATE = 1_000
    }
}
