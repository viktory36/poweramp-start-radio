package com.powerampstartradio.indexing.v2

import com.powerampstartradio.indexing.TrackPcmCache
import java.util.Locale

/** Rejects only decoded boundaries that are unambiguously shorter than container-declared audio. */
internal object V2DecodedEosPublicationPolicy {
    private const val GROSS_SHORTFALL_US = 30_000_000L

    fun requirePublishable(span: FinalizedAudioSpanEvidence) {
        if (span.kind != V2ResolvedAudioSpanKind.WHOLE_FILE ||
            span.authority != V2AudioSpanAuthority.DECODED_END_OF_STREAM ||
            span.container.durationEstimateSource != V2DurationEstimateSource.CONTAINER_METADATA
        ) return

        val declaredUs = span.container.durationUsEstimate
        val decodedUs = span.endExclusiveUs - span.startUs
        if (declaredUs <= 0L || decodedUs <= 0L || decodedUs >= declaredUs) return

        val shortfallUs = declaredUs - decodedUs
        val belowHalf = decodedUs <= Long.MAX_VALUE / 2L && decodedUs * 2L < declaredUs
        if (shortfallUs <= GROSS_SHORTFALL_US || !belowHalf) return

        val providerText = span.providerSpan.durationUs.takeIf { it > 0L }?.let { providerUs ->
            "; Poweramp reports ${seconds(providerUs)} s"
        }.orEmpty()
        val decodedPercent = decodedUs.toDouble() * 100.0 / declaredUs.toDouble()
        throw TrackPcmCache.PcmContractException(
            TrackPcmCache.PcmContractFailure.EOS_MISMATCH,
            "Decoded physical EOS at ${seconds(decodedUs)} s, ${seconds(shortfallUs)} s before " +
                "the container-declared ${seconds(declaredUs)} s " +
                "(${String.format(Locale.ROOT, "%.1f", decodedPercent)}% decoded)$providerText; " +
                "refusing to embed likely corrupt or truncated audio",
        )
    }

    private fun seconds(microseconds: Long): String =
        String.format(Locale.ROOT, "%.3f", microseconds.toDouble() / 1_000_000.0)
}
