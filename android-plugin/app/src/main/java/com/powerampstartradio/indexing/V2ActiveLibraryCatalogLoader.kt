package com.powerampstartradio.indexing

import com.powerampstartradio.data.EmbeddingDatabase
import com.powerampstartradio.indexing.v2.V2ProviderPathGroupCompleteness
import com.powerampstartradio.indexing.v2.V2ProviderPathGroupSnapshot
import com.powerampstartradio.indexing.v2.V2ProviderSpanReceiptReader
import com.powerampstartradio.indexing.v2.V2ResolvedActiveIndexGeneration
import com.powerampstartradio.poweramp.TrackNormalization

internal sealed interface V2ActiveLibraryCatalogLoadProgress {
    data class PowerampRows(
        val completedRows: Int,
        val totalRows: Int,
    ) : V2ActiveLibraryCatalogLoadProgress

    data class IndexedRowRead(
        val completedRows: Int?,
        val totalRows: Int,
    ) : V2ActiveLibraryCatalogLoadProgress

    data class IndexedRows(
        val completedRows: Int,
        val totalRows: Int,
    ) : V2ActiveLibraryCatalogLoadProgress

    data class SourceSpanReceipts(
        val receiptCount: Int?,
        val indexedRowCount: Int,
    ) : V2ActiveLibraryCatalogLoadProgress

    data class Bindings(
        val powerampRowCount: Int,
        val indexedRowCount: Int,
        val receiptCount: Int,
        val activeBindingCount: Int? = null,
        val quarantinedRowCount: Int? = null,
    ) : V2ActiveLibraryCatalogLoadProgress
}

/**
 * Loads one catalog from an already-resolved immutable database generation and one complete
 * Poweramp provider generation. It performs no provider query and changes no persisted state.
 */
internal object V2ActiveLibraryCatalogLoader {
    fun load(
        activeGeneration: V2ResolvedActiveIndexGeneration,
        providerSnapshot: V2ProviderPathGroupSnapshot,
        onProgress: (V2ActiveLibraryCatalogLoadProgress) -> Unit = {},
    ): V2ActiveLibraryCatalog {
        val providerGenerationId = requireNotNull(providerSnapshot.libraryGeneration) {
            "Poweramp snapshot has no complete library generation"
        }
        requireCompleteProviderSnapshot(providerSnapshot)

        val providerRowCount = providerSnapshot.groups.sumOf { it.rows.size }
        onProgress(V2ActiveLibraryCatalogLoadProgress.PowerampRows(0, providerRowCount))
        var completedProviderRows = 0
        val provider = providerSnapshot.groups.flatMap { group ->
            val hasLogicalOffsets = group.rows.any { it.offsetMs > 0L }
            val hasCueSourceRow = group.rows.any { it.cueSourceImageFolderId != null }
            group.rows.map { row ->
                require(row.physicalPath == group.physicalPath) {
                    "Poweramp row path differs from its complete path group"
                }
                val artist = TrackNormalization.normalizeArtist(row.artist)
                val album = TrackNormalization.normalizeAlbum(row.album)
                val title = TrackNormalization.normalizeTitle(row.title)
                val durationMs = Math.toIntExact(row.durationMs)
                V2LegacyProviderCandidate(
                    powerampFileId = row.powerampFileId,
                    // This is the stable lexical path used by exact V2 provider-span receipts.
                    normalizedPhysicalPath = row.physicalPath,
                    offsetMs = row.offsetMs,
                    durationMs = durationMs,
                    metadataKey = TrackNormalization.buildMetadataKey(
                        artist,
                        album,
                        title,
                        durationMs,
                    ),
                    compatibilityEligible = !hasLogicalOffsets && !hasCueSourceRow &&
                        row.cueSourceImageFolderId == null,
                    requiresSpanSpecificRebuild = hasLogicalOffsets || hasCueSourceRow,
                    createdAtEpochSecond = row.createdAtEpochSecond,
                ).also {
                    completedProviderRows++
                    publishRowProgress(completedProviderRows, providerRowCount) {
                        onProgress(
                            V2ActiveLibraryCatalogLoadProgress.PowerampRows(
                                completedRows = completedProviderRows,
                                totalRows = providerRowCount,
                            ),
                        )
                    }
                }
            }
        }

        val expectedDatabaseRows = activeGeneration.manifest.trackCount
        onProgress(
            V2ActiveLibraryCatalogLoadProgress.IndexedRowRead(
                completedRows = null,
                totalRows = expectedDatabaseRows,
            ),
        )
        val database = EmbeddingDatabase.open(activeGeneration.databaseFile)
        val databaseCandidates = try {
            var completedDatabaseRows = 0
            var projectedRowCount = -1
            database.mapAllTrackCatalogRows(
                onRowCount = { rowCount ->
                    projectedRowCount = rowCount
                    onProgress(
                        V2ActiveLibraryCatalogLoadProgress.IndexedRowRead(
                            completedRows = rowCount,
                            totalRows = expectedDatabaseRows,
                        ),
                    )
                    onProgress(
                        V2ActiveLibraryCatalogLoadProgress.IndexedRows(
                            completedRows = 0,
                            totalRows = rowCount,
                        ),
                    )
                },
            ) { trackId, rawArtist, rawAlbum, rawTitle, durationMs, filePath ->
                val artist = TrackNormalization.normalizeArtist(rawArtist)
                val album = TrackNormalization.normalizeAlbum(rawAlbum)
                val title = TrackNormalization.normalizeTitle(rawTitle)
                V2LegacyDatabaseCandidate(
                    trackId = trackId,
                    normalizedPath = TrackNormalization.normalizePath(filePath),
                    durationMs = durationMs,
                    metadataKey = TrackNormalization.buildMetadataKey(
                        artist,
                        album,
                        title,
                        durationMs,
                    ),
                ).also {
                    completedDatabaseRows++
                    publishRowProgress(completedDatabaseRows, projectedRowCount) {
                        onProgress(
                            V2ActiveLibraryCatalogLoadProgress.IndexedRows(
                                completedRows = completedDatabaseRows,
                                totalRows = projectedRowCount,
                            ),
                        )
                    }
                }
            }
        } finally {
            database.close()
        }
        require(databaseCandidates.size == activeGeneration.manifest.trackCount) {
            "Active generation track count changed while loading its library catalog"
        }

        onProgress(
            V2ActiveLibraryCatalogLoadProgress.SourceSpanReceipts(
                receiptCount = null,
                indexedRowCount = databaseCandidates.size,
            ),
        )
        val receipts = V2ProviderSpanReceiptReader.read(activeGeneration.databaseFile).receipts
        onProgress(
            V2ActiveLibraryCatalogLoadProgress.SourceSpanReceipts(
                receiptCount = receipts.size,
                indexedRowCount = databaseCandidates.size,
            ),
        )
        onProgress(
            V2ActiveLibraryCatalogLoadProgress.Bindings(
                powerampRowCount = provider.size,
                indexedRowCount = databaseCandidates.size,
                receiptCount = receipts.size,
            ),
        )
        return V2ActiveLibraryCatalogBuilder.build(
            databaseGenerationId = activeGeneration.manifest.generationId,
            providerGenerationId = providerGenerationId,
            provider = provider,
            database = databaseCandidates,
            receipts = receipts,
        ).also { catalog ->
            onProgress(
                V2ActiveLibraryCatalogLoadProgress.Bindings(
                    powerampRowCount = provider.size,
                    indexedRowCount = databaseCandidates.size,
                    receiptCount = receipts.size,
                    activeBindingCount = catalog.bindings.size,
                    quarantinedRowCount = catalog.quarantinedTracks.size,
                ),
            )
        }
    }

    private inline fun publishRowProgress(
        completedRows: Int,
        totalRows: Int,
        publish: () -> Unit,
    ) {
        if (completedRows == totalRows || completedRows % PROGRESS_ROW_INTERVAL == 0) publish()
    }

    private fun requireCompleteProviderSnapshot(snapshot: V2ProviderPathGroupSnapshot) {
        val acquisition = requireNotNull(snapshot.acquisitionEvidence) {
            "Poweramp snapshot has no cursor-completion evidence"
        }
        require(acquisition.cursorExhaustedNormally) {
            "Poweramp snapshot cursor did not exhaust normally"
        }
        require(acquisition.rowCount == snapshot.groups.sumOf { it.rows.size }) {
            "Poweramp snapshot row count does not match its acquisition evidence"
        }
        require(snapshot.groups.all { it.completeness == V2ProviderPathGroupCompleteness.COMPLETE }) {
            "Poweramp snapshot contains a partial physical-path group"
        }
    }

    private const val PROGRESS_ROW_INTERVAL = 4_096
}
