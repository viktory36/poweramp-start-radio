package com.powerampstartradio.indexing.v2

import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

class V2IndexingLedgerFileNamespaceTest {
    @Test
    fun `ledger listing excludes current and legacy sidecars and every atomic residue`() {
        val files = listOf(
            "job-b.json.bak",
            "job-a.json",
            "job-a.json.bak",
            "job-c.imported-row-supersession-v1.json",
            "job-c.imported-row-supersession-v1.json.new",
            "job-c.imported-row-supersession-v1.json.bak",
            "job-d.imported-row-supersession-v1.auth",
            "job-d.imported-row-supersession-v1.auth.new",
            "job-d.imported-row-supersession-v1.auth.bak",
            "job-e.journal-v1",
            "unrelated.txt",
        ).map { name -> File("/unused/$name") }

        assertEquals(
            listOf("job-a", "job-b"),
            V2IndexingLedgerFileNamespace.listedJobIds(files),
        )
    }

    @Test
    fun `ledger job ID that aliases a legacy sidecar fails closed`() {
        assertThrows(IndexingLedgerConflictException::class.java) {
            V2IndexingLedgerFileNamespace.requireSafeJobId(
                "job-a.imported-row-supersession-v1",
            )
        }
    }
}
