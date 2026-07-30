package com.powerampstartradio.indexing.v2

import android.database.MatrixCursor
import android.os.Debug
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.powerampstartradio.indexing.NewTrackDetector
import com.powerampstartradio.poweramp.TrackNormalization
import java.io.DataOutputStream
import java.io.OutputStream
import java.security.DigestOutputStream
import java.security.MessageDigest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class V2PowerampProviderSnapshotInstrumentedTest {
    @Test
    fun realProviderIsCompleteStableAndClassifiesKnownCueCohort() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val acquirer = V2PowerampProviderSnapshotAcquirer(context)

        val firstRun = run {
            val startedNs = System.nanoTime()
            val snapshot = acquirer.acquireBlocking()
            val summary = summarize(snapshot)
            val acquisition = requireNotNull(snapshot.acquisitionEvidence)
            recordMetric(
                "FIRST rows=${summary.rows} groups=${summary.groups} " +
                    "largestGroup=${summary.largestGroup} elapsedMs=${elapsedMs(startedNs)} " +
                    "queryMs=${acquisition.queryAndCursorReadMs} " +
                    "assemblyMs=${acquisition.snapshotAssemblyMs}",
            )
            assertSnapshotInvariants(snapshot, summary)
            FirstRunEvidence(
                elapsedMs = elapsedMs(startedNs),
                summary = summary,
                generation = requireNotNull(snapshot.libraryGeneration),
                groupingSha256 = groupingSha256(snapshot),
                retainedHeapBytes = usedHeapBytes(),
                pssKb = Debug.getPss(),
                queryAndCursorReadMs = requireNotNull(acquisition.queryAndCursorReadMs),
                snapshotAssemblyMs = requireNotNull(acquisition.snapshotAssemblyMs),
            )
        }
        Runtime.getRuntime().gc()
        System.runFinalization()
        Thread.sleep(100L)

        val secondStartedNs = System.nanoTime()
        val second = acquirer.acquireBlocking()
        val secondMs = elapsedMs(secondStartedNs)
        val secondSummary = summarize(second)
        val secondAcquisition = requireNotNull(second.acquisitionEvidence)
        val secondHeapBytes = usedHeapBytes()
        val secondPssKb = Debug.getPss()
        recordMetric(
            "SECOND rows=${secondSummary.rows} groups=${secondSummary.groups} " +
                "largestGroup=${secondSummary.largestGroup} elapsedMs=$secondMs " +
                "queryMs=${secondAcquisition.queryAndCursorReadMs} " +
                "assemblyMs=${secondAcquisition.snapshotAssemblyMs}",
        )
        assertSnapshotInvariants(second, secondSummary)

        assertEquals(firstRun.generation, second.libraryGeneration)
        assertEquals(firstRun.groupingSha256, groupingSha256(second))
        assertEquals(firstRun.summary, secondSummary)

        val cueGroup = second.groups.singleOrNull { group ->
            group.rows.size == EXPECTED_CUE_ROWS && group.rows.any { row ->
                row.title?.contains("Shadow Propaganda Mix", ignoreCase = true) == true
            }
        }
        assertNotNull("Known DJ Shadow CUE cohort is missing", cueGroup)
        cueGroup!!
        val cueLogicalRows = cueGroup.rows.filter { it.cueSourceImageFolderId == null }
        val firstCueRow = cueLogicalRows.single { it.offsetMs == 0L }
        assertEquals(EXPECTED_CUE_ROWS, cueGroup.rows.size)
        assertEquals(EXPECTED_CUE_ROWS, cueLogicalRows.size)
        assertEquals(EXPECTED_NONZERO_CUE_OFFSETS, cueLogicalRows.count { it.offsetMs > 0L })
        assertEquals(0, cueGroup.rows.count { it.cueSourceImageFolderId != null })
        recordMetric(
            "CUE path=${cueGroup.physicalPath} rows=${cueGroup.rows.size} " +
                "nonzero=${cueLogicalRows.count { it.offsetMs > 0L }} " +
                "firstId=${firstCueRow.powerampFileId}",
        )

        val selected = NewTrackDetector.UnindexedTrack(
            powerampFileId = firstCueRow.powerampFileId,
            artist = TrackNormalization.normalizeArtist(firstCueRow.artist),
            album = TrackNormalization.normalizeAlbum(firstCueRow.album),
            title = TrackNormalization.normalizeTitle(firstCueRow.title),
            durationMs = firstCueRow.durationMs.toInt(),
            path = firstCueRow.providerPhysicalPath,
            offsetMs = firstCueRow.offsetMs,
            cueFolderId = firstCueRow.cueSourceImageFolderId,
        )
        val resolved = V2AudioSpanResolver(V2MediaExtractorAudioInspector())
            .resolve(listOf(selected), second)
            .resolved
            .single()
        assertEquals(V2ResolvedAudioSpanKind.LOGICAL_CUE, resolved.kind)
        assertEquals(EXPECTED_CUE_ROWS, resolved.cueClassificationEvidence.providerGroupRowCount)
        assertEquals(EXPECTED_CUE_ROWS, resolved.cueClassificationEvidence.logicalRowCount)
        assertEquals(
            EXPECTED_NONZERO_CUE_OFFSETS,
            resolved.cueClassificationEvidence.nonZeroOffsetRowIds.size,
        )
        assertEquals(0L, resolved.startUs)
        assertTrue(resolved.endExclusiveUs <= resolved.containerEvidence.durationUsEstimate)
        assertTrue(resolved.exactSampleCount24k > 0L)

        recordMetric(
            buildString {
                append("PASS ")
                append("rows=${secondSummary.rows} groups=${secondSummary.groups} ")
                append("multiRowGroups=${secondSummary.multiRowGroups} ")
                append("largestGroup=${secondSummary.largestGroup} ")
                append("cueShapedGroups=${secondSummary.cueShapedGroups} ")
                append("nonzeroOffsets=${secondSummary.nonzeroOffsets} ")
                append("rawCueSources=${secondSummary.rawCueSources} ")
                append("lexicalAliasGroups=${secondSummary.lexicalAliasGroups} ")
                append("pathIdentityPolicy=provider-lexical-nfc-posix-v1 ")
                append("firstMs=${firstRun.elapsedMs} secondMs=$secondMs ")
                append("firstQueryMs=${firstRun.queryAndCursorReadMs} ")
                append("firstAssemblyMs=${firstRun.snapshotAssemblyMs} ")
                append("secondQueryMs=${secondAcquisition.queryAndCursorReadMs} ")
                append("secondAssemblyMs=${secondAcquisition.snapshotAssemblyMs} ")
                append("firstHeapBytes=${firstRun.retainedHeapBytes} ")
                append("secondHeapBytes=$secondHeapBytes ")
                append("maxObservedHeapBytes=${maxOf(firstRun.retainedHeapBytes, secondHeapBytes)} ")
                append("firstPssKb=${firstRun.pssKb} secondPssKb=$secondPssKb ")
                append("runtimeMaxHeapBytes=${Runtime.getRuntime().maxMemory()} ")
                append("generation=${firstRun.generation} ")
                append("groupingSha256=${firstRun.groupingSha256} ")
                append("cuePath=${cueGroup.physicalPath} cueFirstId=${firstCueRow.powerampFileId} ")
                append("cueContainerUs=${resolved.containerEvidence.durationUsEstimate} ")
                append("cueFirstEndUs=${resolved.endExclusiveUs} ")
                append("cue24kSamples=${resolved.exactSampleCount24k}")
            },
        )
    }

    @Test
    fun injectedPermissionDenialIsTypedWithoutTouchingLiveGrant() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val error = assertThrows(V2PowerampProviderSnapshotException::class.java) {
            V2PowerampProviderSnapshotAcquirer(
                context = context,
                providerQuery = V2PowerampProviderQuery { _, _, _ ->
                    throw SecurityException("injected denial")
                },
            ).acquireBlocking()
        }
        assertEquals(V2PowerampProviderSnapshotFailureCode.POWERAMP_PERMISSION_DENIED, error.code)
    }

    @Test
    fun injectedNullCursorAndSchemaErrorAreTypedWithoutLiveQuery() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val nullCursor = assertThrows(V2PowerampProviderSnapshotException::class.java) {
            V2PowerampProviderSnapshotAcquirer(
                context = context,
                providerQuery = V2PowerampProviderQuery { _, _, _ -> null },
            ).acquireBlocking()
        }
        assertEquals(
            V2PowerampProviderSnapshotFailureCode.PROVIDER_RETURNED_NULL_CURSOR,
            nullCursor.code,
        )

        val schemaError = assertThrows(V2PowerampProviderSnapshotException::class.java) {
            V2PowerampProviderSnapshotAcquirer(
                context = context,
                providerQuery = V2PowerampProviderQuery { _, _, _ ->
                    MatrixCursor(arrayOf("_id"))
                },
            ).acquireBlocking()
        }
        assertEquals(
            V2PowerampProviderSnapshotFailureCode.PROVIDER_SCHEMA_MISMATCH,
            schemaError.code,
        )
    }

    private fun assertSnapshotInvariants(
        snapshot: V2ProviderPathGroupSnapshot,
        summary: SnapshotSummary,
    ) {
        val acquisition = requireNotNull(snapshot.acquisitionEvidence)
        assertTrue(acquisition.cursorExhaustedNormally)
        assertEquals(summary.rows, acquisition.rowCount)
        assertTrue("Expected an 80k-scale Poweramp library", summary.rows >= MIN_EXPECTED_ROWS)
        assertTrue(summary.groups >= MIN_EXPECTED_GROUPS)
        assertTrue(summary.groups <= summary.rows)
        assertEquals(summary.rows, snapshot.groups.sumOf { it.rows.size })
        assertEquals(
            summary.rows,
            snapshot.groups.flatMap { it.rows }.map { it.powerampFileId }.toSet().size,
        )
        assertTrue(snapshot.groups.all {
            it.completeness == V2ProviderPathGroupCompleteness.COMPLETE
        })
        assertTrue(!snapshot.libraryGeneration.isNullOrBlank())
        assertTrue(snapshot.libraryGeneration!!.startsWith(
            "poweramp-provider-snapshot-v2-sha256:",
        ))
    }

    private fun summarize(snapshot: V2ProviderPathGroupSnapshot): SnapshotSummary {
        val allRows = snapshot.groups.asSequence().flatMap { it.rows.asSequence() }
        var rows = 0
        var nonzeroOffsets = 0
        var rawCueSources = 0
        allRows.forEach { row ->
            rows++
            if (row.cueSourceImageFolderId == null && row.offsetMs > 0L) nonzeroOffsets++
            if (row.cueSourceImageFolderId != null) rawCueSources++
        }
        return SnapshotSummary(
            rows = rows,
            groups = snapshot.groups.size,
            multiRowGroups = snapshot.groups.count { it.rows.size > 1 },
            largestGroup = snapshot.groups.maxOf { it.rows.size },
            cueShapedGroups = snapshot.groups.count { group ->
                group.rows.any { it.cueSourceImageFolderId != null } ||
                    group.rows.any { it.cueSourceImageFolderId == null && it.offsetMs > 0L }
            },
            nonzeroOffsets = nonzeroOffsets,
            rawCueSources = rawCueSources,
            lexicalAliasGroups = snapshot.groups.count { group ->
                group.rows.map { it.providerPhysicalPath }.distinct().size > 1
            },
        )
    }

    private fun groupingSha256(snapshot: V2ProviderPathGroupSnapshot): String {
        val digest = MessageDigest.getInstance("SHA-256")
        DataOutputStream(DigestOutputStream(DiscardingOutputStream, digest)).use { output ->
            output.writeInt(snapshot.groups.size)
            snapshot.groups.forEach { group ->
                output.writeString(group.physicalPath)
                output.writeInt(group.rows.size)
                group.rows.forEach { row -> output.writeLong(row.powerampFileId) }
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    private fun DataOutputStream.writeString(value: String) {
        val bytes = value.toByteArray(Charsets.UTF_8)
        writeInt(bytes.size)
        write(bytes)
    }

    private fun elapsedMs(startedNs: Long): Long =
        (System.nanoTime() - startedNs) / 1_000_000L

    private fun recordMetric(message: String) {
        Log.i(TAG, message)
        println("PASR_METRIC $TAG $message")
    }

    private fun usedHeapBytes(): Long = Runtime.getRuntime().let { runtime ->
        runtime.totalMemory() - runtime.freeMemory()
    }

    private data class SnapshotSummary(
        val rows: Int,
        val groups: Int,
        val multiRowGroups: Int,
        val largestGroup: Int,
        val cueShapedGroups: Int,
        val nonzeroOffsets: Int,
        val rawCueSources: Int,
        val lexicalAliasGroups: Int,
    )

    private data class FirstRunEvidence(
        val elapsedMs: Long,
        val summary: SnapshotSummary,
        val generation: String,
        val groupingSha256: String,
        val retainedHeapBytes: Long,
        val pssKb: Long,
        val queryAndCursorReadMs: Long,
        val snapshotAssemblyMs: Long,
    )

    private object DiscardingOutputStream : OutputStream() {
        override fun write(value: Int) = Unit
        override fun write(buffer: ByteArray, offset: Int, length: Int) = Unit
    }

    private companion object {
        const val TAG = "V2ProviderSnapshotTest"
        const val MIN_EXPECTED_ROWS = 80_000
        const val MIN_EXPECTED_GROUPS = 79_000
        const val EXPECTED_CUE_ROWS = 29
        const val EXPECTED_NONZERO_CUE_OFFSETS = 28
    }
}
