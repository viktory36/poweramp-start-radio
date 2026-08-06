package com.powerampstartradio.indexing.v2

import android.content.Context
import com.powerampstartradio.indexing.NewTrackDetector
import com.powerampstartradio.poweramp.TrackNormalization
import java.io.File

class V2IndexingPreflightCancelledException :
    IllegalStateException("Indexing preflight was cancelled")

interface V2IndexingPreflightObserver {
    fun onProgress(progress: V2IndexingPreflightProgress)
    fun throwIfCancelled()

    data object None : V2IndexingPreflightObserver {
        override fun onProgress(progress: V2IndexingPreflightProgress) = Unit
        override fun throwIfCancelled() = Unit
    }
}

object V2IndexingPreflightSelectionFactory {
    fun fromTracks(
        tracks: List<NewTrackDetector.UnindexedTrack>,
    ): List<V2IndexingPreflightSelection> = tracks.map { track ->
        val path = TrackNormalization.normalizePath(track.path)
            ?: throw V2IndexingPreflightException(
                code = V2IndexingPreflightFailureCode.INVALID_SELECTION_EVIDENCE,
                powerampFileId = track.powerampFileId,
                message = "Selected Poweramp row ${track.powerampFileId} has no absolute source path",
            )
        V2IndexingPreflightSelection(
            powerampFileId = track.powerampFileId,
            providerPhysicalPath = path,
            durationMs = V2ProviderDurationEvidencePolicy.canonicalMs(
                track.durationMs.toLong(),
            ),
            offsetMs = track.offsetMs,
            cueSourceImageFolderId = track.cueFolderId,
        )
    }
}

/** Converts one durable selection intent and one complete provider snapshot into planner input. */
class V2IndexingPreflightRequestMaterializer(context: Context) {
    private val appContext = context.applicationContext

    fun materialize(
        intent: V2IndexingPreflightIntent,
        providerSnapshot: V2ProviderPathGroupSnapshot,
    ): V2IndexingPreflightRequest {
        require(intent.state == V2IndexingPreflightIntentState.PLANNING) {
            "preflight intent ${intent.jobId} is not planning"
        }
        val baseGenerationId = requireNotNull(intent.baseGenerationId) {
            "preflight intent ${intent.jobId} is not bound to a base generation"
        }
        val acquisition = providerSnapshot.acquisitionEvidence
            ?: providerFailure("Poweramp snapshot has no acquisition evidence")
        val allRows = providerSnapshot.groups.flatMap { it.rows }
        if (!acquisition.cursorExhaustedNormally || acquisition.rowCount != allRows.size) {
            providerFailure("Poweramp snapshot is incomplete")
        }
        val rowsById = allRows.associateBy(V2ProviderPathRowEvidence::powerampFileId)
        if (rowsById.size != allRows.size) providerFailure("Poweramp snapshot has duplicate row IDs")
        val groupsByPath = providerSnapshot.groups.associateBy(V2ProviderPathGroupEvidence::physicalPath)

        val selectedTracks = intent.selected.map { selected ->
            val row = rowsById[selected.powerampFileId]
                ?: sourceChanged(selected, "is no longer present in the Poweramp snapshot")
            val normalizedPath = TrackNormalization.normalizePath(row.providerPhysicalPath)
                ?: sourceChanged(selected, "no longer has an absolute source path")
            if (normalizedPath != selected.providerPhysicalPath ||
                V2ProviderDurationEvidencePolicy.canonicalMs(row.durationMs) !=
                    selected.durationMs ||
                row.offsetMs != selected.offsetMs ||
                row.cueSourceImageFolderId != selected.cueSourceImageFolderId
            ) {
                sourceChanged(selected, "changed after it was selected")
            }
            val group = groupsByPath[row.physicalPath]
                ?: providerFailure("Poweramp snapshot omitted path group ${row.physicalPath}")
            if (group.completeness != V2ProviderPathGroupCompleteness.COMPLETE) {
                providerFailure("Poweramp path group ${row.physicalPath} is incomplete")
            }
            V2ResolvedTrackSource(
                track = NewTrackDetector.UnindexedTrack(
                    powerampFileId = row.powerampFileId,
                    artist = TrackNormalization.normalizeArtist(row.artist),
                    album = TrackNormalization.normalizeAlbum(row.album),
                    title = TrackNormalization.normalizeTitle(row.title),
                    durationMs = Math.toIntExact(
                        V2ProviderDurationEvidencePolicy.canonicalMs(row.durationMs),
                    ),
                    path = normalizedPath,
                    offsetMs = row.offsetMs,
                    cueFolderId = row.cueSourceImageFolderId,
                    sourceReferenceCount = group.rows.size,
                    sourceHasLogicalOffsets = group.rows.any { it.offsetMs > 0L },
                    sourceHasCueImageRow = group.rows.any {
                        it.cueSourceImageFolderId != null
                    },
                ),
                sourceFile = File(row.providerPhysicalPath),
            )
        }

        val filesDir = appContext.filesDir.canonicalFile
        return V2IndexingPreflightRequest(
            selectedTracks = selectedTracks,
            models = V2ResolvedIndexingModels(
                mertModelFile = requirePrivateArtifact(filesDir, "mert.tflite"),
                clamp3AudioModelFile = requirePrivateArtifact(filesDir, "clamp3_audio.tflite"),
                clamp3TextModelFile = requirePrivateArtifact(filesDir, "clamp3_text.tflite"),
                sentencePieceModelFile = requirePrivateArtifact(
                    filesDir,
                    "sentencepiece.bpe.model",
                ),
            ),
            providerSnapshot = providerSnapshot,
            baseGenerationId = baseGenerationId,
            rebuildDerivedIndexes = intent.rebuildDerivedIndexes,
            executionProfile = V2IndexingExecutionProfile.FULL,
            jobId = intent.jobId,
            createdAtEpochMs = intent.createdAtEpochMs,
            selectedOccurrences = intent.selected,
        )
    }

    private fun requirePrivateArtifact(filesDir: File, filename: String): File {
        val file = File(filesDir, filename).canonicalFile
        if (file.parentFile != filesDir || !file.isFile || !file.canRead() || file.length() <= 0L) {
            throw V2IndexingPreflightException(
                code = V2IndexingPreflightFailureCode.MODEL_UNREADABLE,
                message = "Required V2 model artifact is unavailable: $filename",
            )
        }
        return file
    }

    private fun sourceChanged(
        selected: V2IndexingPreflightSelection,
        detail: String,
    ): Nothing = throw V2IndexingPreflightException(
        code = V2IndexingPreflightFailureCode.SOURCE_CHANGED,
        powerampFileId = selected.powerampFileId,
        message = "Selected Poweramp row ${selected.powerampFileId} $detail",
    )

    private fun providerFailure(message: String): Nothing =
        throw V2IndexingPreflightException(
            code = V2IndexingPreflightFailureCode.PROVIDER_SNAPSHOT_INVALID,
            message = message,
        )
}
