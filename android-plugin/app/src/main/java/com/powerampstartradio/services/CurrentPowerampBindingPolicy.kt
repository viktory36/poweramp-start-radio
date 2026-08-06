package com.powerampstartradio.services

import com.powerampstartradio.data.EmbeddedTrack
import com.powerampstartradio.indexing.V2ActiveLibraryBinding
import com.powerampstartradio.indexing.V2ActiveLibraryBindingEvidence
import com.powerampstartradio.indexing.V2LegacyCompatibilityResolver
import com.powerampstartradio.indexing.v2.V2CommittedProviderSpan
import com.powerampstartradio.indexing.v2.V2ProviderDurationEvidencePolicy
import com.powerampstartradio.indexing.v2.V2ProviderSpanReceipt
import com.powerampstartradio.indexing.v2.V2StableProviderLexicalPathNormalizer
import com.powerampstartradio.poweramp.PowerampFileEntry
import com.powerampstartradio.poweramp.TrackNormalization
import kotlin.math.abs

/** Exact current-provider check shared by cached seed admission and final queue delivery. */
internal object CurrentPowerampBindingPolicy {
    fun matches(
        binding: V2ActiveLibraryBinding,
        indexed: EmbeddedTrack,
        current: PowerampFileEntry,
        receipt: V2ProviderSpanReceipt?,
    ): Boolean {
        if (binding.trackId != indexed.id || binding.powerampFileId != current.id) return false
        val currentPath = current.path ?: return false
        return when (binding.evidence) {
            V2ActiveLibraryBindingEvidence.EXACT_V2_RECEIPT_SPAN -> {
                val currentSpan = runCatching {
                    V2CommittedProviderSpan(
                        normalizedPhysicalPath =
                            V2StableProviderLexicalPathNormalizer.normalizeAbsolute(currentPath),
                        offsetMs = current.offsetMs,
                        durationMs = V2ProviderDurationEvidencePolicy.canonicalMs(
                            current.durationMs.toLong(),
                        ),
                    )
                }.getOrNull() ?: return false
                receipt?.trackId == indexed.id && receipt.providerSpan == currentSpan
            }
            V2ActiveLibraryBindingEvidence.LEGACY_EXACT_ABSOLUTE_PATH ->
                current.offsetMs == 0L && current.cueFolderId == null &&
                    TrackNormalization.normalizePath(indexed.filePath) == currentPath &&
                    strictLegacyPathEvidenceCompatible(indexed, current)
            V2ActiveLibraryBindingEvidence.LEGACY_EXACT_MUSIC_RELATIVE_PATH -> {
                val indexedRelative = TrackNormalization.normalizePath(indexed.filePath)
                    ?.let(V2LegacyCompatibilityResolver::strictMusicRelativePath)
                val currentRelative =
                    V2LegacyCompatibilityResolver.strictMusicRelativePath(currentPath)
                current.offsetMs == 0L && current.cueFolderId == null &&
                    indexedRelative != null && indexedRelative == currentRelative &&
                    strictLegacyPathEvidenceCompatible(indexed, current)
            }
        }
    }

    private fun strictLegacyPathEvidenceCompatible(
        indexed: EmbeddedTrack,
        current: PowerampFileEntry,
    ): Boolean {
        if (indexed.durationMs > 0 && current.durationMs > 0 &&
            abs(indexed.durationMs - current.durationMs) <= 5_000
        ) {
            return true
        }
        val indexedMetadata = TrackNormalization.buildMetadataKey(
            TrackNormalization.normalizeArtist(indexed.artist),
            TrackNormalization.normalizeAlbum(indexed.album),
            TrackNormalization.normalizeTitle(indexed.title),
            indexed.durationMs,
        )
        return V2LegacyCompatibilityResolver.metadataWithoutDuration(indexedMetadata) ==
            V2LegacyCompatibilityResolver.metadataWithoutDuration(current.metadataKey)
    }
}
