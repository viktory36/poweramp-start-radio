package com.powerampstartradio.indexing.v2

import com.powerampstartradio.indexing.NewTrackDetector

/** Pure mapping from a resolved acoustic span to one immutable ledger input. */
object V2IndexingTrackPlanFactory {
    fun validate(
        resolvedSpan: V2ResolvedAudioSpan,
        providerRow: V2ProviderPathRowEvidence,
    ): ExpectedTrackWork {
        val track = resolvedSpan.selectedTrack
        if (track.cueFolderId != null || providerRow.cueSourceImageFolderId != null) {
            throw V2IndexingPreflightException(
                V2IndexingPreflightFailureCode.CUE_SOURCE_IMAGE,
                powerampFileId = track.powerampFileId,
                message = "Poweramp row ${track.powerampFileId} is a raw CUE source image",
            )
        }
        val normalizedProviderPath = try {
            V2StableProviderLexicalPathNormalizer.normalizeAbsolute(
                providerRow.providerPhysicalPath,
            )
        } catch (error: Exception) {
            throw V2IndexingPreflightException(
                V2IndexingPreflightFailureCode.INVALID_LOGICAL_SPAN,
                powerampFileId = track.powerampFileId,
                message = "Poweramp row ${track.powerampFileId} has invalid provider path evidence",
                cause = error,
            )
        }
        val unresolvedOrdinary = V2UnknownDurationOrdinarySpanPolicy.isUnresolved(resolvedSpan)
        val validAcousticCoordinates = if (unresolvedOrdinary) {
            resolvedSpan.startUs == 0L && resolvedSpan.endExclusiveUs == 0L &&
                resolvedSpan.startSourceSample == 0L &&
                resolvedSpan.endSourceSampleExclusive == 0L &&
                resolvedSpan.sourceSampleCount == 0L &&
                resolvedSpan.exactSampleCount24k == 0L &&
                resolvedSpan.expectedWork == V2UnknownDurationOrdinarySpanPolicy.unresolvedWork &&
                providerRow.durationMs == 0L
        } else {
            resolvedSpan.startUs >= 0L &&
                resolvedSpan.endExclusiveUs > resolvedSpan.startUs &&
                resolvedSpan.sourceSampleCount > 0L &&
                resolvedSpan.exactSampleCount24k > 0L
        }
        if (providerRow.powerampFileId != track.powerampFileId ||
            providerRow.physicalPath != normalizedProviderPath ||
            resolvedSpan.containerEvidence.physicalPath.isBlank() ||
            !resolvedSpan.containerEvidence.physicalPath.startsWith('/') ||
            !validAcousticCoordinates
        ) {
            throw V2IndexingPreflightException(
                V2IndexingPreflightFailureCode.INVALID_LOGICAL_SPAN,
                powerampFileId = track.powerampFileId,
                message = "Poweramp row ${track.powerampFileId} has inconsistent resolved evidence",
            )
        }
        if (unresolvedOrdinary) return V2UnknownDurationOrdinarySpanPolicy.unresolvedWork
        val recomputed = try {
            V2AudioSpanMath.expectedWorkFor24kSamples(resolvedSpan.exactSampleCount24k)
        } catch (error: Exception) {
            throw V2IndexingPreflightException(
                V2IndexingPreflightFailureCode.INVALID_LOGICAL_SPAN,
                powerampFileId = track.powerampFileId,
                message = "Poweramp row ${track.powerampFileId} has invalid exact sample work",
                cause = error,
            )
        }
        if (recomputed != resolvedSpan.expectedWork) {
            throw V2IndexingPreflightException(
                V2IndexingPreflightFailureCode.INVALID_LOGICAL_SPAN,
                powerampFileId = track.powerampFileId,
                message = "Poweramp row ${track.powerampFileId} work does not match resolved PCM",
            )
        }
        if (recomputed.mertWindows <= 0 || recomputed.clampSegments <= 0) {
            throw V2IndexingPreflightException(
                V2IndexingPreflightFailureCode.AUDIO_TOO_SHORT,
                powerampFileId = track.powerampFileId,
                message = "Poweramp row ${track.powerampFileId} has under one second of audio",
            )
        }
        return recomputed
    }

    fun create(
        resolvedSpan: V2ResolvedAudioSpan,
        providerRow: V2ProviderPathRowEvidence,
        sourceFingerprint: SourceFingerprint,
    ): SelectedTrackInput {
        val track: NewTrackDetector.UnindexedTrack = resolvedSpan.selectedTrack
        val expectedWork = validate(resolvedSpan, providerRow)
        val providerSnapshotGeneration = resolvedSpan.libraryGeneration
            ?.takeIf { it.isNotBlank() }
            ?: throw V2IndexingPreflightException(
                V2IndexingPreflightFailureCode.INVALID_PLAN,
                powerampFileId = track.powerampFileId,
                message = "Resolved span has no provider snapshot generation",
            )

        return SelectedTrackInput(
            powerampFileId = track.powerampFileId,
            providerSnapshotGeneration = providerSnapshotGeneration,
            providerRow = providerRow.copy(
                durationMs = V2ProviderDurationEvidencePolicy.canonicalMs(providerRow.durationMs),
            ),
            displayMetadata = DisplayTrackMetadata(
                artist = track.artist,
                album = track.album,
                title = track.title,
            ),
            normalizedMetadata = NormalizedTrackMetadata(
                normalizationSpecId = V2IndexingWorkPolicy.METADATA_NORMALIZATION_SPEC_ID,
                artist = track.artist,
                album = track.album,
                title = track.title,
                metadataKey = track.metadataKey,
            ),
            physicalPath = resolvedSpan.containerEvidence.physicalPath,
            sourceFingerprint = sourceFingerprint,
            finalizedAudioSpan = FinalizedAudioSpanEvidence(
                kind = resolvedSpan.kind,
                authority = resolvedSpan.authority,
                executionBoundaryRequirement = resolvedSpan.executionBoundaryRequirement,
                providerSpan = resolvedSpan.providerEvidence,
                cueClassification = resolvedSpan.cueClassificationEvidence,
                container = resolvedSpan.containerEvidence,
                startUs = resolvedSpan.startUs,
                endExclusiveUs = resolvedSpan.endExclusiveUs,
                startSourceSample = resolvedSpan.startSourceSample,
                endSourceSampleExclusive = resolvedSpan.endSourceSampleExclusive,
                sourceSampleCount = resolvedSpan.sourceSampleCount,
                exactSampleCount24k = resolvedSpan.exactSampleCount24k,
                expectedWork = expectedWork,
            ),
        )
    }
}
