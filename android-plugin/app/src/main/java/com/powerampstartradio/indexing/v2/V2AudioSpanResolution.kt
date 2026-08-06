package com.powerampstartradio.indexing.v2

import com.powerampstartradio.indexing.AudioSampleTimeline
import com.powerampstartradio.indexing.MertWindowPolicy
import com.powerampstartradio.indexing.NewTrackDetector
import com.powerampstartradio.indexing.TorchAudioHannV1Policy
import com.powerampstartradio.poweramp.TrackNormalization

/** Whether a provider path group contains every row returned for that physical path. */
enum class V2ProviderPathGroupCompleteness {
    COMPLETE,
    PARTIAL,
}

/** Provider evidence for one row. Duration and offset retain Poweramp's millisecond precision. */
data class V2ProviderPathRowEvidence(
    val powerampFileId: Long,
    /** Stable lexical provider key retained unchanged in the immutable selected-row evidence. */
    val physicalPath: String,
    /** Exact joined path exposed by Poweramp before lexical normalization. */
    val providerPhysicalPath: String = physicalPath,
    val artist: String? = null,
    val album: String? = null,
    val title: String? = null,
    val offsetMs: Long,
    val offsetWasNull: Boolean = false,
    val durationMs: Long,
    /** Non-null only for the raw, uncut source-image row exposed for some CUE albums. */
    val cueSourceImageFolderId: Long?,
    /** Poweramp's provider-owned first-seen time, in Unix epoch seconds. */
    val createdAtEpochSecond: Long = 1L,
)

/** Every row from the complete cursor with the same normalized lexical provider asset path. */
data class V2ProviderPathGroupEvidence(
    val physicalPath: String,
    val rows: List<V2ProviderPathRowEvidence>,
    val completeness: V2ProviderPathGroupCompleteness,
)

/** One consistent provider query, grouped without consulting mutable filesystem state. */
data class V2ProviderPathGroupSnapshot(
    val libraryGeneration: String?,
    val groups: List<V2ProviderPathGroupEvidence>,
    val acquisitionEvidence: V2ProviderSnapshotAcquisitionEvidence? = null,
)

/** Where the non-authoritative duration used to schedule a decode came from. */
enum class V2DurationEstimateSource {
    CONTAINER_METADATA,
    PROVIDER_SPAN_FALLBACK,
    UNAVAILABLE,
}

/** Poweramp uses non-positive duration values as equivalent unknown-duration sentinels. */
object V2ProviderDurationEvidencePolicy {
    fun canonicalMs(durationMs: Long): Long = durationMs.coerceAtLeast(0L)
}

/** MediaExtractor facts for the first audio stream in a physical source. */
data class V2AudioContainerEvidence(
    val physicalPath: String,
    val audioTrackIndex: Int,
    /** MediaFormat.KEY_DURATION, in microseconds. It is a planning estimate until EOS. */
    val durationUsEstimate: Long,
    val durationEstimateSource: V2DurationEstimateSource =
        V2DurationEstimateSource.CONTAINER_METADATA,
    val sampleRateHz: Int,
    val channelCount: Int,
    val mime: String,
)

fun interface V2AudioContainerInspector {
    fun inspect(physicalPath: String): V2AudioContainerEvidence
}

enum class V2AudioContainerInspectionFailureCode {
    SOURCE_UNREADABLE,
    NO_AUDIO_STREAM,
    UNSUPPORTED_OR_INVALID_CONTAINER,
}

class V2AudioContainerInspectionException(
    val code: V2AudioContainerInspectionFailureCode,
    message: String,
    cause: Throwable? = null,
) : IllegalArgumentException(message, cause)

enum class V2ResolvedAudioSpanKind {
    WHOLE_FILE,
    LOGICAL_CUE,
}

/** The evidence that chose the planned half-open audio span. */
enum class V2AudioSpanAuthority {
    /** Ordinary file before decode: container duration schedules work but is not final authority. */
    PROVISIONAL_END_OF_STREAM,

    /** Ordinary file after one physical-EOS decode fixed the exact native sample boundary. */
    DECODED_END_OF_STREAM,

    /** CUE row: the provider's explicit [offset, offset + duration) logical boundary. */
    PROVIDER_CUE_HALF_OPEN_SPAN,
}

enum class V2ExecutionBoundaryRequirement {
    /** Decode to EOS and reconcile the planned count before publishing any artifact. */
    VERIFY_END_OF_STREAM_AND_RECONCILE,

    /** Enforce the provider's exact half-open CUE boundary and reject premature EOS. */
    ENFORCE_PROVIDER_HALF_OPEN_SPAN,
}

/** Microsecond conversion of the selected provider row, retained even when not authoritative. */
data class V2ProviderSpanEvidence(
    val offsetUs: Long,
    val durationUs: Long,
    val endExclusiveUs: Long,
)

/** Why a complete path group was or was not classified as a CUE source. */
data class V2CueClassificationEvidence(
    val providerGroupRowCount: Int,
    val logicalRowCount: Int,
    val nonZeroOffsetRowIds: List<Long>,
    val rawSourceImageRowIds: List<Long>,
)

internal object V2CueClassificationEvidenceFactory {
    fun from(group: V2ProviderPathGroupEvidence): V2CueClassificationEvidence {
        val logicalRows = group.rows.filter { it.cueSourceImageFolderId == null }
        return V2CueClassificationEvidence(
            providerGroupRowCount = group.rows.size,
            logicalRowCount = logicalRows.size,
            nonZeroOffsetRowIds = logicalRows
                .filter { it.offsetMs > 0L }
                .map { it.powerampFileId }
                .sorted(),
            rawSourceImageRowIds = group.rows
                .filter { it.cueSourceImageFolderId != null }
                .map { it.powerampFileId }
                .sorted(),
        )
    }
}

data class V2ResolvedAudioSpan(
    val selectedTrack: NewTrackDetector.UnindexedTrack,
    val libraryGeneration: String?,
    val kind: V2ResolvedAudioSpanKind,
    val authority: V2AudioSpanAuthority,
    val executionBoundaryRequirement: V2ExecutionBoundaryRequirement,
    val providerEvidence: V2ProviderSpanEvidence,
    val cueClassificationEvidence: V2CueClassificationEvidence,
    val containerEvidence: V2AudioContainerEvidence,
    /** Planned half-open boundary in the physical source timeline. */
    val startUs: Long,
    val endExclusiveUs: Long,
    /** Exact first included and first excluded native PCM coordinates. */
    val startSourceSample: Long,
    val endSourceSampleExclusive: Long,
    val sourceSampleCount: Long,
    /** Exact polyphase output length implied by [sourceSampleCount]. */
    val exactSampleCount24k: Long,
    val expectedWork: ExpectedTrackWork,
) {
    val plannedDurationUs: Long get() = endExclusiveUs - startUs

    /** True only when the container estimate must be replaced or confirmed by decoded EOS. */
    val mustVerifyEndOfStream: Boolean
        get() = executionBoundaryRequirement ==
            V2ExecutionBoundaryRequirement.VERIFY_END_OF_STREAM_AND_RECONCILE
}

/**
 * The one provisional shape whose acoustic denominator is deliberately unknown until decode.
 * Zero here means "not measured yet", never an exact empty track.
 */
internal object V2UnknownDurationOrdinarySpanPolicy {
    val unresolvedWork = ExpectedTrackWork(mertWindows = 0, clampSegments = 0)

    fun isUnresolved(span: V2ResolvedAudioSpan): Boolean = isUnresolved(
        kind = span.kind,
        authority = span.authority,
        container = span.containerEvidence,
    )

    fun isUnresolved(span: FinalizedAudioSpanEvidence): Boolean = isUnresolved(
        kind = span.kind,
        authority = span.authority,
        container = span.container,
    )

    fun hasUnavailableDuration(container: V2AudioContainerEvidence): Boolean =
        container.durationUsEstimate == 0L &&
            container.durationEstimateSource == V2DurationEstimateSource.UNAVAILABLE

    private fun isUnresolved(
        kind: V2ResolvedAudioSpanKind,
        authority: V2AudioSpanAuthority,
        container: V2AudioContainerEvidence,
    ): Boolean = kind == V2ResolvedAudioSpanKind.WHOLE_FILE &&
        authority == V2AudioSpanAuthority.PROVISIONAL_END_OF_STREAM &&
        hasUnavailableDuration(container)
}

enum class V2AudioSpanExclusionReason {
    RAW_CUE_SOURCE_IMAGE,
}

data class V2ExcludedAudioSpan(
    val selectedTrack: NewTrackDetector.UnindexedTrack,
    val reason: V2AudioSpanExclusionReason,
)

data class V2AudioSpanResolutionBatch(
    val libraryGeneration: String?,
    val resolved: List<V2ResolvedAudioSpan>,
    val excluded: List<V2ExcludedAudioSpan>,
    val rejected: List<V2RejectedAudioSpan> = emptyList(),
)

data class V2RejectedAudioSpan(
    val selectedTrack: NewTrackDetector.UnindexedTrack,
    val code: V2AudioSpanResolutionFailureCode,
    val diagnostic: String,
)

enum class V2AudioSpanResolutionFailureCode {
    INVALID_SNAPSHOT_ACQUISITION_EVIDENCE,
    DUPLICATE_SELECTED_ROW,
    DUPLICATE_PATH_GROUP,
    INCOMPLETE_PATH_GROUP,
    INVALID_PATH_GROUP,
    SELECTED_PATH_MISSING,
    SELECTED_ROW_NOT_IN_SNAPSHOT,
    SELECTED_ROW_CHANGED,
    CONTAINER_INSPECTION_FAILED,
    INVALID_CONTAINER_EVIDENCE,
    INVALID_PROVIDER_SPAN,
    CUE_SPAN_OUT_OF_BOUNDS,
    SAMPLE_COORDINATE_OVERFLOW,
    DURATION_UNAVAILABLE_FOR_PLANNING,
    SOURCE_UNREADABLE,
    NO_AUDIO_STREAM,
    UNSUPPORTED_OR_INVALID_CONTAINER,
}

enum class V2AudioSpanResolutionFailureScope {
    SELECTED_SOURCE_OR_OCCURRENCE,
    GLOBAL_SNAPSHOT_OR_SELECTION,
}

object V2AudioSpanResolutionFailurePolicy {
    fun scope(code: V2AudioSpanResolutionFailureCode): V2AudioSpanResolutionFailureScope =
        when (code) {
            V2AudioSpanResolutionFailureCode.CONTAINER_INSPECTION_FAILED,
            V2AudioSpanResolutionFailureCode.INVALID_CONTAINER_EVIDENCE,
            V2AudioSpanResolutionFailureCode.INVALID_PROVIDER_SPAN,
            V2AudioSpanResolutionFailureCode.CUE_SPAN_OUT_OF_BOUNDS,
            V2AudioSpanResolutionFailureCode.SAMPLE_COORDINATE_OVERFLOW,
            V2AudioSpanResolutionFailureCode.SOURCE_UNREADABLE,
            V2AudioSpanResolutionFailureCode.NO_AUDIO_STREAM,
            V2AudioSpanResolutionFailureCode.UNSUPPORTED_OR_INVALID_CONTAINER,
            -> V2AudioSpanResolutionFailureScope.SELECTED_SOURCE_OR_OCCURRENCE

            V2AudioSpanResolutionFailureCode.INVALID_SNAPSHOT_ACQUISITION_EVIDENCE,
            V2AudioSpanResolutionFailureCode.DUPLICATE_SELECTED_ROW,
            V2AudioSpanResolutionFailureCode.DUPLICATE_PATH_GROUP,
            V2AudioSpanResolutionFailureCode.INCOMPLETE_PATH_GROUP,
            V2AudioSpanResolutionFailureCode.INVALID_PATH_GROUP,
            V2AudioSpanResolutionFailureCode.SELECTED_PATH_MISSING,
            V2AudioSpanResolutionFailureCode.SELECTED_ROW_NOT_IN_SNAPSHOT,
            V2AudioSpanResolutionFailureCode.SELECTED_ROW_CHANGED,
            V2AudioSpanResolutionFailureCode.DURATION_UNAVAILABLE_FOR_PLANNING,
            -> V2AudioSpanResolutionFailureScope.GLOBAL_SNAPSHOT_OR_SELECTION
        }
}

class V2AudioSpanResolutionException(
    val code: V2AudioSpanResolutionFailureCode,
    val powerampFileId: Long? = null,
    message: String,
    cause: Throwable? = null,
) : IllegalArgumentException(message, cause)

/** Exact, overflow-checked arithmetic shared by preflight and executor verification. */
object V2AudioSpanMath {
    const val TARGET_SAMPLE_RATE_HZ = MertWindowPolicy.SAMPLE_RATE
    const val MERT_WINDOW_SAMPLES = MertWindowPolicy.WINDOW_SAMPLES.toLong()
    const val MERT_MINIMUM_TAIL_SAMPLES = MertWindowPolicy.MINIMUM_TAIL_SAMPLES.toLong()
    const val CLAMP_MAX_FRAMES = 128L

    /** First sample whose timestamp is at or after [timeUs]. */
    fun sampleAtOrAfter(timeUs: Long, sampleRateHz: Int): Long =
        AudioSampleTimeline.sampleAtOrAfter(timeUs, sampleRateHz)

    /** Stable microsecond representation whose first included sample is exactly [sample]. */
    fun canonicalTimeUsForSampleBoundary(sample: Long, sampleRateHz: Int): Long {
        require(sample >= 0L) { "sample must be non-negative" }
        require(sampleRateHz in 1..1_000_000) { "unsupported sample rate $sampleRateHz" }
        val whole = Math.multiplyExact(sample / sampleRateHz, 1_000_000L)
        val partial = Math.multiplyExact(sample % sampleRateHz, 1_000_000L) / sampleRateHz
        val result = Math.addExact(whole, partial)
        check(sampleAtOrAfter(result, sampleRateHz) == sample) {
            "unable to represent sample $sample at ${sampleRateHz}Hz in microseconds"
        }
        return result
    }

    /** Exact TorchAudio Hann V1 float32 target-length rule pinned by the embedding spec. */
    fun resampledLength(inputSamples: Long, fromRateHz: Int, toRateHz: Int): Long =
        TorchAudioHannV1Policy.resampledLength(inputSamples, fromRateHz, toRateHz)

    /** Work is derived from exact canonical PCM samples, never a rounded metadata duration. */
    fun expectedWorkFor24kSamples(sampleCount24k: Long): ExpectedTrackWork {
        val windows = MertWindowPolicy.windowCount(sampleCount24k).toLong()

        val clampSegments = if (windows == 0L) {
            0L
        } else {
            ceilDiv(Math.addExact(windows, 2L), CLAMP_MAX_FRAMES)
        }
        require(clampSegments <= Int.MAX_VALUE.toLong()) {
            "track has too many CLaMP segments"
        }
        return ExpectedTrackWork(windows.toInt(), clampSegments.toInt())
    }

    private fun ceilDiv(value: Long, divisor: Long): Long =
        if (value == 0L) 0L else 1L + (value - 1L) / divisor
}

/**
 * Resolves selected provider rows into durable acoustic work spans. Ordinary-file work remains
 * explicitly provisional until the executor observes physical EOS.
 *
 * Duplicate rows are not CUE evidence. A group is CUE-shaped only when it contains an
 * explicit non-zero logical offset or a raw source-image row. This lets the first zero-offset
 * row of a CUE album resolve correctly without misclassifying ordinary provider duplicates.
 */
class V2AudioSpanResolver(
    private val containerInspector: V2AudioContainerInspector,
) {
    fun resolve(
        selectedTracks: List<NewTrackDetector.UnindexedTrack>,
        providerSnapshot: V2ProviderPathGroupSnapshot,
        onSourceInspection: (
            completedSources: Int,
            totalSources: Int,
            currentSourcePath: String?,
        ) -> Unit = { _, _, _ -> },
    ): V2AudioSpanResolutionBatch {
        validateAcquisitionEvidence(providerSnapshot)
        val duplicateSelectedId = selectedTracks.groupingBy { it.powerampFileId }
            .eachCount()
            .entries
            .firstOrNull { it.value > 1 }
            ?.key
        if (duplicateSelectedId != null) {
            fail(
                V2AudioSpanResolutionFailureCode.DUPLICATE_SELECTED_ROW,
                duplicateSelectedId,
                "Poweramp row $duplicateSelectedId was selected more than once",
            )
        }

        val snapshotIndex = validateAndIndexGroups(providerSnapshot.groups)
        val resolved = ArrayList<V2ResolvedAudioSpan>(selectedTracks.size)
        val excluded = mutableListOf<V2ExcludedAudioSpan>()
        val rejected = mutableListOf<V2RejectedAudioSpan>()
        val inputs = mutableListOf<SelectedResolutionInput>()

        for (selected in selectedTracks) {
            val selectedPath = selected.path?.takeIf { it.isNotBlank() } ?: fail(
                V2AudioSpanResolutionFailureCode.SELECTED_PATH_MISSING,
                selected.powerampFileId,
                "Poweramp row ${selected.powerampFileId} has no physical path",
            )
            val group = snapshotIndex.groupByRowId[selected.powerampFileId]
                ?: snapshotIndex.groupsByPath[selectedPath]
                ?: fail(
                V2AudioSpanResolutionFailureCode.SELECTED_ROW_NOT_IN_SNAPSHOT,
                selected.powerampFileId,
                "Complete provider snapshot has no row or path group for $selectedPath",
            )
            val providerRow = group.rows.singleOrNull {
                it.powerampFileId == selected.powerampFileId
            } ?: fail(
                V2AudioSpanResolutionFailureCode.SELECTED_ROW_NOT_IN_SNAPSHOT,
                selected.powerampFileId,
                "Poweramp row ${selected.powerampFileId} is absent from its path group",
            )
            validateSelectedRowStillMatches(selected, providerRow)

            if (providerRow.cueSourceImageFolderId != null) {
                excluded += V2ExcludedAudioSpan(
                    selectedTrack = selected,
                    reason = V2AudioSpanExclusionReason.RAW_CUE_SOURCE_IMAGE,
                )
                continue
            }

            val cueEvidence = V2CueClassificationEvidenceFactory.from(group)
            val isCue = cueEvidence.nonZeroOffsetRowIds.isNotEmpty() ||
                cueEvidence.rawSourceImageRowIds.isNotEmpty()
            inputs += SelectedResolutionInput(
                selected = selected,
                providerRow = providerRow,
                group = group,
                cueEvidence = cueEvidence,
                isCue = isCue,
            )
        }

        val sourceGroups = inputs.groupBy { it.group.physicalPath }.values.toList()
        sourceGroups.forEachIndexed sourceGroup@{ index, sourceInputs ->
            val rawProviderPaths = sourceInputs
                .map { it.providerRow.providerPhysicalPath }
                .distinct()
            val path = rawProviderPaths.singleOrNull() ?: fail(
                V2AudioSpanResolutionFailureCode.INVALID_PATH_GROUP,
                sourceInputs.first().selected.powerampFileId,
                "One normalized Poweramp path resolves to multiple raw filesystem paths",
            )
            onSourceInspection(index, sourceGroups.size, path)
            val container = try {
                inspectAndValidate(path, sourceInputs.first().selected.powerampFileId)
            } catch (error: V2AudioSpanResolutionException) {
                if (V2AudioSpanResolutionFailurePolicy.scope(error.code) ==
                    V2AudioSpanResolutionFailureScope.GLOBAL_SNAPSHOT_OR_SELECTION
                ) throw error
                sourceInputs.forEach { input -> rejected += input.rejected(error) }
                onSourceInspection(index + 1, sourceGroups.size, null)
                return@sourceGroup
            }
            sourceInputs.forEach { input ->
                try {
                    resolved += resolveOne(
                        selected = input.selected,
                        providerRow = input.providerRow,
                        cueEvidence = input.cueEvidence,
                        isCue = input.isCue,
                        container = container,
                        libraryGeneration = providerSnapshot.libraryGeneration,
                    )
                } catch (error: V2AudioSpanResolutionException) {
                    if (V2AudioSpanResolutionFailurePolicy.scope(error.code) ==
                        V2AudioSpanResolutionFailureScope.GLOBAL_SNAPSHOT_OR_SELECTION
                    ) throw error
                    rejected += input.rejected(error)
                }
            }
            onSourceInspection(index + 1, sourceGroups.size, null)
        }

        return V2AudioSpanResolutionBatch(
            libraryGeneration = providerSnapshot.libraryGeneration,
            resolved = resolved.sortedBySelectionOrder(selectedTracks),
            excluded = excluded,
            rejected = rejected.sortedBySelectionOrder(selectedTracks),
        )
    }

    private data class SelectedResolutionInput(
        val selected: NewTrackDetector.UnindexedTrack,
        val providerRow: V2ProviderPathRowEvidence,
        val group: V2ProviderPathGroupEvidence,
        val cueEvidence: V2CueClassificationEvidence,
        val isCue: Boolean,
    ) {
        fun rejected(error: V2AudioSpanResolutionException) = V2RejectedAudioSpan(
            selectedTrack = selected,
            code = error.code,
            diagnostic = error.message ?: "Audio span resolution failed",
        )
    }

    private fun List<V2RejectedAudioSpan>.sortedBySelectionOrder(
        selected: List<NewTrackDetector.UnindexedTrack>,
    ): List<V2RejectedAudioSpan> {
        val ordinal = selected.mapIndexed { index, track -> track.powerampFileId to index }.toMap()
        return sortedBy { ordinal.getValue(it.selectedTrack.powerampFileId) }
    }

    @JvmName("sortResolvedAudioSpansBySelectionOrder")
    private fun List<V2ResolvedAudioSpan>.sortedBySelectionOrder(
        selected: List<NewTrackDetector.UnindexedTrack>,
    ): List<V2ResolvedAudioSpan> {
        val ordinal = selected.mapIndexed { index, track -> track.powerampFileId to index }.toMap()
        return sortedBy { ordinal.getValue(it.selectedTrack.powerampFileId) }
    }

    private fun validateAcquisitionEvidence(snapshot: V2ProviderPathGroupSnapshot) {
        val evidence = snapshot.acquisitionEvidence ?: fail(
            V2AudioSpanResolutionFailureCode.INVALID_SNAPSHOT_ACQUISITION_EVIDENCE,
            message = "Provider snapshot has no cursor acquisition evidence",
        )
        val groupedRows = snapshot.groups.sumOf { it.rows.size }
        if (!evidence.cursorExhaustedNormally || evidence.rowCount != groupedRows ||
            snapshot.libraryGeneration.isNullOrBlank()
        ) {
            fail(
                V2AudioSpanResolutionFailureCode.INVALID_SNAPSHOT_ACQUISITION_EVIDENCE,
                message = "Provider snapshot is not a verified complete cursor result",
            )
        }
    }

    private data class SnapshotIndex(
        val groupsByPath: Map<String, V2ProviderPathGroupEvidence>,
        val groupByRowId: Map<Long, V2ProviderPathGroupEvidence>,
    )

    private fun validateAndIndexGroups(
        groups: List<V2ProviderPathGroupEvidence>,
    ): SnapshotIndex {
        val byPath = LinkedHashMap<String, V2ProviderPathGroupEvidence>()
        val byRowId = LinkedHashMap<Long, V2ProviderPathGroupEvidence>()
        for (group in groups) {
            if (group.completeness != V2ProviderPathGroupCompleteness.COMPLETE) {
                fail(
                    V2AudioSpanResolutionFailureCode.INCOMPLETE_PATH_GROUP,
                    message = "Provider path group is partial: ${group.physicalPath}",
                )
            }
            if (group.physicalPath.isBlank() || group.rows.isEmpty()) {
                fail(
                    V2AudioSpanResolutionFailureCode.INVALID_PATH_GROUP,
                    message = "Provider path group must have a path and at least one row",
                )
            }
            if (byPath.put(group.physicalPath, group) != null) {
                fail(
                    V2AudioSpanResolutionFailureCode.DUPLICATE_PATH_GROUP,
                    message = "Provider snapshot contains duplicate group ${group.physicalPath}",
                )
            }
            for (row in group.rows) {
                if (row.physicalPath != group.physicalPath ||
                    row.providerPhysicalPath.isBlank() ||
                    byRowId.put(row.powerampFileId, group) != null
                ) {
                    fail(
                        V2AudioSpanResolutionFailureCode.INVALID_PATH_GROUP,
                        row.powerampFileId,
                        "Provider row ${row.powerampFileId} has inconsistent path-group evidence",
                    )
                }
            }
        }
        return SnapshotIndex(groupsByPath = byPath, groupByRowId = byRowId)
    }

    private fun validateSelectedRowStillMatches(
        selected: NewTrackDetector.UnindexedTrack,
        evidence: V2ProviderPathRowEvidence,
    ) {
        val selectedDurationMs = V2ProviderDurationEvidencePolicy.canonicalMs(
            selected.durationMs.toLong(),
        )
        if (TrackNormalization.normalizePath(selected.path) !=
            TrackNormalization.normalizePath(evidence.providerPhysicalPath) ||
            selected.offsetMs != evidence.offsetMs ||
            selectedDurationMs != V2ProviderDurationEvidencePolicy.canonicalMs(
                evidence.durationMs,
            ) ||
            selected.cueFolderId != evidence.cueSourceImageFolderId ||
            selected.artist != TrackNormalization.normalizeArtist(evidence.artist) ||
            selected.album != TrackNormalization.normalizeAlbum(evidence.album) ||
            selected.title != TrackNormalization.normalizeTitle(evidence.title)
        ) {
            fail(
                V2AudioSpanResolutionFailureCode.SELECTED_ROW_CHANGED,
                selected.powerampFileId,
                "Poweramp row ${selected.powerampFileId} changed after selection",
            )
        }
    }

    private fun inspectAndValidate(
        path: String,
        powerampFileId: Long,
    ): V2AudioContainerEvidence {
        val evidence = try {
            containerInspector.inspect(path)
        } catch (error: V2AudioSpanResolutionException) {
            throw error
        } catch (error: V2AudioContainerInspectionException) {
            val code = when (error.code) {
                V2AudioContainerInspectionFailureCode.SOURCE_UNREADABLE ->
                    V2AudioSpanResolutionFailureCode.SOURCE_UNREADABLE
                V2AudioContainerInspectionFailureCode.NO_AUDIO_STREAM ->
                    V2AudioSpanResolutionFailureCode.NO_AUDIO_STREAM
                V2AudioContainerInspectionFailureCode.UNSUPPORTED_OR_INVALID_CONTAINER ->
                    V2AudioSpanResolutionFailureCode.UNSUPPORTED_OR_INVALID_CONTAINER
            }
            fail(code, powerampFileId, error.message ?: "Container inspection failed", error)
        } catch (error: Exception) {
            fail(
                V2AudioSpanResolutionFailureCode.CONTAINER_INSPECTION_FAILED,
                powerampFileId,
                "MediaExtractor inspection failed for $path: ${error.message}",
                error,
            )
        }
        if (evidence.physicalPath != path || evidence.audioTrackIndex < 0 ||
            evidence.durationUsEstimate < 0L || evidence.sampleRateHz <= 0 ||
            evidence.channelCount <= 0 || !evidence.mime.startsWith("audio/")
        ) {
            fail(
                V2AudioSpanResolutionFailureCode.INVALID_CONTAINER_EVIDENCE,
                powerampFileId,
                "Invalid MediaExtractor audio evidence for $path",
            )
        }
        if ((evidence.durationUsEstimate > 0L) !=
            (evidence.durationEstimateSource != V2DurationEstimateSource.UNAVAILABLE)
        ) {
            fail(
                V2AudioSpanResolutionFailureCode.INVALID_CONTAINER_EVIDENCE,
                powerampFileId,
                "Container duration value/source evidence is inconsistent for $path",
            )
        }
        return evidence
    }

    private fun resolveOne(
        selected: NewTrackDetector.UnindexedTrack,
        providerRow: V2ProviderPathRowEvidence,
        cueEvidence: V2CueClassificationEvidence,
        isCue: Boolean,
        container: V2AudioContainerEvidence,
        libraryGeneration: String?,
    ): V2ResolvedAudioSpan {
        val providerSpan = providerSpan(providerRow, selected.powerampFileId)
        if (isCue) validateCueGroupRows(cueEvidence, providerRow, selected.powerampFileId)
        val planningContainer = when {
            container.durationUsEstimate > 0L -> container
            isCue -> container.copy(
                durationUsEstimate = providerSpan.endExclusiveUs,
                durationEstimateSource = V2DurationEstimateSource.PROVIDER_SPAN_FALLBACK,
            )
            providerSpan.durationUs > 0L -> container.copy(
                durationUsEstimate = providerSpan.durationUs,
                durationEstimateSource = V2DurationEstimateSource.PROVIDER_SPAN_FALLBACK,
            )
            else -> container
        }
        val startUs: Long
        val endUs: Long
        val kind: V2ResolvedAudioSpanKind
        val authority: V2AudioSpanAuthority
        val requirement: V2ExecutionBoundaryRequirement
        if (isCue) {
            startUs = providerSpan.offsetUs
            endUs = providerSpan.endExclusiveUs
            kind = V2ResolvedAudioSpanKind.LOGICAL_CUE
            authority = V2AudioSpanAuthority.PROVIDER_CUE_HALF_OPEN_SPAN
            requirement = V2ExecutionBoundaryRequirement.ENFORCE_PROVIDER_HALF_OPEN_SPAN
        } else {
            startUs = 0L
            endUs = planningContainer.durationUsEstimate
            kind = V2ResolvedAudioSpanKind.WHOLE_FILE
            authority = V2AudioSpanAuthority.PROVISIONAL_END_OF_STREAM
            requirement = V2ExecutionBoundaryRequirement.VERIFY_END_OF_STREAM_AND_RECONCILE
        }

        if (kind == V2ResolvedAudioSpanKind.WHOLE_FILE &&
            V2UnknownDurationOrdinarySpanPolicy.hasUnavailableDuration(planningContainer)
        ) {
            return V2ResolvedAudioSpan(
                selectedTrack = selected,
                libraryGeneration = libraryGeneration,
                kind = kind,
                authority = authority,
                executionBoundaryRequirement = requirement,
                providerEvidence = providerSpan,
                cueClassificationEvidence = cueEvidence,
                containerEvidence = planningContainer,
                startUs = 0L,
                endExclusiveUs = 0L,
                startSourceSample = 0L,
                endSourceSampleExclusive = 0L,
                sourceSampleCount = 0L,
                exactSampleCount24k = 0L,
                expectedWork = V2UnknownDurationOrdinarySpanPolicy.unresolvedWork,
            )
        }

        val startSourceSample: Long
        val endSourceSample: Long
        val sourceSampleCount: Long
        val sampleCount24k: Long
        try {
            startSourceSample = V2AudioSpanMath.sampleAtOrAfter(
                startUs,
                planningContainer.sampleRateHz,
            )
            endSourceSample = V2AudioSpanMath.sampleAtOrAfter(
                endUs,
                planningContainer.sampleRateHz,
            )
            sourceSampleCount = Math.subtractExact(endSourceSample, startSourceSample)
            require(sourceSampleCount > 0L) { "resolved span contains no PCM samples" }
            sampleCount24k = V2AudioSpanMath.resampledLength(
                sourceSampleCount,
                planningContainer.sampleRateHz,
                V2AudioSpanMath.TARGET_SAMPLE_RATE_HZ,
            )
        } catch (error: ArithmeticException) {
            fail(
                V2AudioSpanResolutionFailureCode.SAMPLE_COORDINATE_OVERFLOW,
                selected.powerampFileId,
                "Sample coordinates overflow for Poweramp row ${selected.powerampFileId}",
                error,
            )
        } catch (error: IllegalArgumentException) {
            fail(
                V2AudioSpanResolutionFailureCode.INVALID_PROVIDER_SPAN,
                selected.powerampFileId,
                "Invalid sample span for Poweramp row ${selected.powerampFileId}: ${error.message}",
                error,
            )
        }

        return V2ResolvedAudioSpan(
            selectedTrack = selected,
            libraryGeneration = libraryGeneration,
            kind = kind,
            authority = authority,
            executionBoundaryRequirement = requirement,
            providerEvidence = providerSpan,
            cueClassificationEvidence = cueEvidence,
            containerEvidence = planningContainer,
            startUs = startUs,
            endExclusiveUs = endUs,
            startSourceSample = startSourceSample,
            endSourceSampleExclusive = endSourceSample,
            sourceSampleCount = sourceSampleCount,
            exactSampleCount24k = sampleCount24k,
            expectedWork = V2AudioSpanMath.expectedWorkFor24kSamples(sampleCount24k),
        )
    }

    private fun validateCueGroupRows(
        cueEvidence: V2CueClassificationEvidence,
        selectedRow: V2ProviderPathRowEvidence,
        powerampFileId: Long,
    ) {
        if (cueEvidence.logicalRowCount == 0 || selectedRow.offsetMs < 0L ||
            selectedRow.durationMs <= 0L
        ) {
            fail(
                V2AudioSpanResolutionFailureCode.INVALID_PROVIDER_SPAN,
                powerampFileId,
                "CUE row $powerampFileId has an invalid provider half-open span",
            )
        }
    }

    private fun providerSpan(
        row: V2ProviderPathRowEvidence,
        powerampFileId: Long,
    ): V2ProviderSpanEvidence {
        try {
            val offsetUs = Math.multiplyExact(row.offsetMs, 1_000L)
            val durationUs = Math.multiplyExact(
                V2ProviderDurationEvidencePolicy.canonicalMs(row.durationMs),
                1_000L,
            )
            val endUs = Math.addExact(offsetUs, durationUs)
            // Ordinary-file provider durations are evidence only. Even zero or otherwise
            // broken metadata must not override a valid container inspection. CUE spans are
            // range-validated separately because only there is this structural data binding.
            return V2ProviderSpanEvidence(offsetUs, durationUs, endUs)
        } catch (error: ArithmeticException) {
            fail(
                V2AudioSpanResolutionFailureCode.INVALID_PROVIDER_SPAN,
                powerampFileId,
                "Provider span overflows microsecond coordinates for row $powerampFileId",
                error,
            )
        }
    }

    private fun fail(
        code: V2AudioSpanResolutionFailureCode,
        powerampFileId: Long? = null,
        message: String,
        cause: Throwable? = null,
    ): Nothing = throw V2AudioSpanResolutionException(
        code = code,
        powerampFileId = powerampFileId,
        message = message,
        cause = cause,
    )
}
