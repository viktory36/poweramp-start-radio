package com.powerampstartradio.ui

import com.powerampstartradio.indexing.V2ActiveLibraryCatalogLoadProgress
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Test

class ActiveLibraryCatalogProgressTextTest {
    @Test
    fun initial_row_phases_name_work_without_claiming_zero_progress() {
        val poweramp = activeLibraryCatalogProgressText(
            V2ActiveLibraryCatalogLoadProgress.PowerampRows(
                completedRows = 0,
                totalRows = 75_123,
            ),
        )
        val indexed = activeLibraryCatalogProgressText(
            V2ActiveLibraryCatalogLoadProgress.IndexedRows(
                completedRows = 0,
                totalRows = 74_998,
            ),
        )

        assertEquals("Preparing to normalize 75,123 Poweramp library rows", poweramp)
        assertEquals("Preparing to normalize 74,998 indexed track rows", indexed)
        assertFalse(poweramp.contains("0 of"))
        assertFalse(indexed.contains("0 of"))
    }

    @Test
    fun completed_binding_phase_reports_exact_outcome_counts() {
        assertEquals(
            "Reconciled 74,998 indexed tracks: 74,900 current Poweramp bindings, " +
                "98 without a current exact binding",
            activeLibraryCatalogProgressText(
                V2ActiveLibraryCatalogLoadProgress.Bindings(
                    powerampRowCount = 75_010,
                    indexedRowCount = 74_998,
                    receiptCount = 206,
                    activeBindingCount = 74_900,
                    quarantinedRowCount = 98,
                ),
            ),
        )
    }
}
