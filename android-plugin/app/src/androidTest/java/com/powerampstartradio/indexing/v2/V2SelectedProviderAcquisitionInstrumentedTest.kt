package com.powerampstartradio.indexing.v2

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/** Read-only proof that bounded provider reads preserve real selected-path evidence. */
@RunWith(AndroidJUnit4::class)
class V2SelectedProviderAcquisitionInstrumentedTest {
    @Test
    fun boundedOrdinaryReadMatchesCompleteLibraryEvidence() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val acquirer = V2PowerampProviderSnapshotAcquirer(context)
        val complete = acquirer.acquireBlocking()
        val selectedGroups = complete.groups.asSequence()
            .filter { group ->
                group.completeness == V2ProviderPathGroupCompleteness.COMPLETE &&
                    group.rows.size == 1 &&
                    group.rows.single().offsetMs == 0L &&
                    group.rows.single().cueSourceImageFolderId == null
            }
            .take(3)
            .toList()
        assertEquals(3, selectedGroups.size)
        val selectedIds = selectedGroups.map { it.rows.single().powerampFileId }

        val bounded = acquirer.acquireSelectedWithCueFallbackBlocking(selectedIds)
        assertEquals(
            V2PowerampProviderAcquisitionScope.SELECTED_PATH_GROUPS,
            bounded.scope,
        )
        selectedGroups.forEach { expected ->
            val selectedId = expected.rows.single().powerampFileId
            val actual = bounded.snapshot.groups.single { group ->
                group.rows.any { it.powerampFileId == selectedId }
            }
            assertEquals(expected, actual)
        }
        assertTrue(bounded.snapshot.acquisitionEvidence?.cursorExhaustedNormally == true)

        val completeEvidence = requireNotNull(complete.acquisitionEvidence)
        Log.i(
            TAG,
            "libraryRows=${completeEvidence.rowCount} selectedIds=$selectedIds " +
                "fullQueryMs=${completeEvidence.queryAndCursorReadMs} " +
                "fullAssemblyMs=${completeEvidence.snapshotAssemblyMs} " +
                "boundedRows=${bounded.snapshot.acquisitionEvidence?.rowCount} " +
                "boundedProbeMs=${bounded.selectedProbeMs} boundedTotalMs=${bounded.totalMs}",
        )
    }

    private companion object {
        const val TAG = "SelectedProviderProof"
    }
}
