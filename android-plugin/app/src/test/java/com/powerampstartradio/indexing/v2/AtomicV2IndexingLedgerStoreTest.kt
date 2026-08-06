package com.powerampstartradio.indexing.v2

import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.InputStream
import java.io.OutputStream
import java.io.RandomAccessFile
import kotlin.system.measureNanoTime
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class AtomicV2IndexingLedgerStoreTest {
    @get:Rule
    val temporaryFolder = TemporaryFolder()

    @Test
    fun `restart replays durable deltas and warm require does not reread base`() {
        val directory = temporaryFolder.newFolder("restart-replay")
        val io = CountingBaseFileIo()
        val store = AtomicV2IndexingLedgerStore(directory, io)
        val planned = plannedLedger(trackCount = 3, jobId = "restart-replay")
        store.create(planned)

        var expected = store.updateLatest(planned.jobSpec.jobId) {
            V2IndexingLedgerStateMachine.startJob(it, 1_001L)
        }
        expected = store.updateLatest(planned.jobSpec.jobId) {
            V2IndexingLedgerStateMachine.admitPlannedTracksForExecution(it, 1_002L)
        }
        repeat(20) {
            assertEquals(expected, store.require(planned.jobSpec.jobId))
        }
        assertEquals(0, io.reads)

        val restartedIo = CountingBaseFileIo()
        val restarted = AtomicV2IndexingLedgerStore(directory, restartedIo)
        assertEquals(expected, restarted.require(planned.jobSpec.jobId))
        repeat(20) { restarted.require(planned.jobSpec.jobId) }
        assertEquals(1, restartedIo.reads)
    }

    @Test
    fun `decoded EOS descriptor mutation survives restart replay`() {
        val directory = temporaryFolder.newFolder("eos-replay")
        val store = AtomicV2IndexingLedgerStore(directory, TestBaseFileIo)
        val planned = plannedLedger(trackCount = 1, jobId = "eos-replay")
        store.create(planned)
        store.updateLatest(planned.jobSpec.jobId) {
            V2IndexingLedgerStateMachine.startJob(it, 1_001L)
        }
        var decoding = store.updateLatest(planned.jobSpec.jobId) {
            V2IndexingLedgerStateMachine.admitPlannedTracksForExecution(it, 1_002L)
        }
        decoding = store.updateLatest(planned.jobSpec.jobId) {
            V2IndexingLedgerStateMachine.beginNextTrackStage(
                it,
                it.tracks.single().workId,
                1_003L,
            )
        }
        val previousWorkId = decoding.tracks.single().workId
        val span = decoding.jobSpec.tracks.single().finalizedAudioSpan
        val finalized = store.updateLatest(planned.jobSpec.jobId) {
            V2DecodedEosPlanFinalizer.finalizeCanonicalGroup(
                ledger = it,
                canonicalWorkId = previousWorkId,
                evidence = V2DecodedEosEvidence(
                    sourceSampleRateHz = span.container.sampleRateHz,
                    observedStartSourceSample = span.startSourceSample,
                    observedEndSourceSampleExclusive = span.endSourceSampleExclusive,
                    observedSourceSampleCount = span.sourceSampleCount,
                    exactSampleCount24k = span.exactSampleCount24k,
                    endOfStreamReached = true,
                ),
                nowEpochMs = 1_004L,
            ).ledger
        }

        assertNotEquals(previousWorkId, finalized.tracks.single().workId)
        assertEquals(
            V2AudioSpanAuthority.DECODED_END_OF_STREAM,
            finalized.jobSpec.tracks.single().finalizedAudioSpan.authority,
        )
        val replayed = AtomicV2IndexingLedgerStore(
            directory,
            TestBaseFileIo,
        ).require(planned.jobSpec.jobId)
        assertEquals(finalized, replayed)
    }

    @Test
    fun `torn final frame is discarded and the repaired journal remains appendable`() {
        val directory = temporaryFolder.newFolder("torn-tail")
        val jobId = "torn-tail"
        val store = AtomicV2IndexingLedgerStore(directory, TestBaseFileIo)
        val planned = plannedLedger(trackCount = 1, jobId = jobId)
        store.create(planned)
        val started = store.updateLatest(jobId) {
            V2IndexingLedgerStateMachine.startJob(it, 1_001L)
        }
        val journal = journalFile(directory, jobId)
        val validLength = journal.length()
        FileOutputStream(journal, true).use { output ->
            output.write(byteArrayOf(0x50, 0x53, 0x52, 0x32, 0x4a))
            output.fd.sync()
        }
        assertTrue(journal.length() > validLength)

        val restarted = AtomicV2IndexingLedgerStore(directory, TestBaseFileIo)
        assertEquals(started, restarted.require(jobId))
        assertEquals(validLength, journal.length())

        val pauseRequested = restarted.updateLatest(jobId) {
            V2IndexingLedgerStateMachine.requestPause(it, 1_002L)
        }
        val replayed = AtomicV2IndexingLedgerStore(
            directory,
            TestBaseFileIo,
        ).require(jobId)
        assertEquals(pauseRequested, replayed)
    }

    @Test
    fun `checksum failure in an interior frame fails closed`() {
        val directory = temporaryFolder.newFolder("corrupt-interior")
        val jobId = "corrupt-interior"
        val store = AtomicV2IndexingLedgerStore(directory, TestBaseFileIo)
        val planned = plannedLedger(trackCount = 1, jobId = jobId)
        store.create(planned)
        store.updateLatest(jobId) { V2IndexingLedgerStateMachine.startJob(it, 1_001L) }
        store.updateLatest(jobId) { V2IndexingLedgerStateMachine.requestPause(it, 1_002L) }

        val journal = journalFile(directory, jobId)
        RandomAccessFile(journal, "rw").use { file ->
            file.seek(FRAME_HEADER_BYTES.toLong() + 8L)
            val original = file.readByte().toInt()
            file.seek(FRAME_HEADER_BYTES.toLong() + 8L)
            file.writeByte(original xor 0x01)
            file.fd.sync()
        }

        assertThrows(InvalidIndexingLedgerException::class.java) {
            AtomicV2IndexingLedgerStore(directory, TestBaseFileIo).require(jobId)
        }
    }

    @Test
    fun `stale revision fails without appending a record`() {
        val directory = temporaryFolder.newFolder("revision-conflict")
        val jobId = "revision-conflict"
        val store = AtomicV2IndexingLedgerStore(directory, TestBaseFileIo)
        val planned = plannedLedger(trackCount = 1, jobId = jobId)
        store.create(planned)
        val started = store.update(jobId, 0L) {
            V2IndexingLedgerStateMachine.startJob(it, 1_001L)
        }
        val before = journalFile(directory, jobId).length()

        assertThrows(IndexingLedgerConflictException::class.java) {
            store.update(jobId, 0L) {
                V2IndexingLedgerStateMachine.requestPause(it, 1_002L)
            }
        }
        assertEquals(started, store.require(jobId))
        assertEquals(before, journalFile(directory, jobId).length())
    }

    @Test
    fun `terminal transition replaces base and retires journal before cold load`() {
        val directory = temporaryFolder.newFolder("terminal-compaction")
        val jobId = "terminal-compaction"
        val store = AtomicV2IndexingLedgerStore(directory, TestBaseFileIo)
        val planned = plannedLedger(trackCount = 1, jobId = jobId)
        store.create(planned)
        store.updateLatest(jobId) { V2IndexingLedgerStateMachine.startJob(it, 1_001L) }
        store.updateLatest(jobId) { V2IndexingLedgerStateMachine.requestCancel(it, 1_002L) }
        val cancelled = store.updateLatest(jobId) {
            V2IndexingLedgerStateMachine.finishCancel(it, 1_003L)
        }

        assertEquals(IndexingJobState.CANCELLED, cancelled.state)
        assertEquals(0L, journalFile(directory, jobId).length())
        assertEquals(
            cancelled,
            AtomicV2IndexingLedgerStore(directory, TestBaseFileIo).require(jobId),
        )
    }

    @Test
    fun `cold load finishes terminal compaction after crash following base replacement`() {
        val directory = temporaryFolder.newFolder("terminal-crash")
        val jobId = "terminal-crash"
        val store = AtomicV2IndexingLedgerStore(
            directory = directory,
            baseFileIo = TestBaseFileIo,
            afterTerminalBaseWrite = { throw SimulatedCompactionCrash() },
        )
        val planned = plannedLedger(trackCount = 1, jobId = jobId)
        store.create(planned)
        store.updateLatest(jobId) { V2IndexingLedgerStateMachine.startJob(it, 1_001L) }
        store.updateLatest(jobId) { V2IndexingLedgerStateMachine.requestCancel(it, 1_002L) }

        assertThrows(java.io.IOException::class.java) {
            store.updateLatest(jobId) { V2IndexingLedgerStateMachine.finishCancel(it, 1_003L) }
        }
        assertTrue(journalFile(directory, jobId).length() > 0L)

        val restarted = AtomicV2IndexingLedgerStore(directory, TestBaseFileIo)
        val recovered = restarted.require(jobId)
        assertEquals(IndexingJobState.CANCELLED, recovered.state)
        assertEquals(3L, recovered.revision)
        assertEquals(0L, journalFile(directory, jobId).length())
    }

    @Test
    fun `cold load compacts terminal delta after crash before base replacement`() {
        val directory = temporaryFolder.newFolder("terminal-before-base-crash")
        val jobId = "terminal-before-base-crash"
        val store = AtomicV2IndexingLedgerStore(directory, FailSecondBaseWriteIo())
        val planned = plannedLedger(trackCount = 1, jobId = jobId)
        store.create(planned)
        store.updateLatest(jobId) { V2IndexingLedgerStateMachine.startJob(it, 1_001L) }
        store.updateLatest(jobId) { V2IndexingLedgerStateMachine.requestCancel(it, 1_002L) }

        assertThrows(java.io.IOException::class.java) {
            store.updateLatest(jobId) { V2IndexingLedgerStateMachine.finishCancel(it, 1_003L) }
        }
        assertTrue(journalFile(directory, jobId).length() > 0L)

        val restarted = AtomicV2IndexingLedgerStore(directory, TestBaseFileIo)
        val recovered = restarted.require(jobId)
        assertEquals(IndexingJobState.CANCELLED, recovered.state)
        assertEquals(3L, recovered.revision)
        assertEquals(0L, journalFile(directory, jobId).length())
        assertEquals(
            recovered,
            AtomicV2IndexingLedgerStore(directory, TestBaseFileIo).require(jobId),
        )
    }

    @Test
    fun `471 track hot transition keeps base immutable and appends a bounded header delta`() {
        val directory = temporaryFolder.newFolder("large-ledger")
        val jobId = "large-ledger"
        val store = AtomicV2IndexingLedgerStore(directory, TestBaseFileIo)
        val planned = plannedLedger(trackCount = 471, jobId = jobId)
        store.create(planned)
        val base = File(directory, "$jobId${V2IndexingLedgerFileNamespace.FILE_SUFFIX}")
        val baseBefore = base.readBytes()

        val headerElapsedNanos = measureNanoTime {
            store.updateLatest(jobId) {
                V2IndexingLedgerStateMachine.startJob(it, 1_001L)
            }
        }
        val journal = journalFile(directory, jobId)
        val headerBytes = journal.length()
        val admissionElapsedNanos = measureNanoTime {
            store.updateLatest(jobId) {
                V2IndexingLedgerStateMachine.admitPlannedTracksForExecution(it, 1_002L)
            }
        }
        val afterAdmissionBytes = journal.length()
        val singleTrackElapsedNanos = measureNanoTime {
            store.updateLatest(jobId) {
                V2IndexingLedgerStateMachine.beginNextTrackStage(
                    it,
                    it.tracks.first().workId,
                    1_003L,
                )
            }
        }
        val finalJournalBytes = journal.length()
        val admissionBytes = afterAdmissionBytes - headerBytes
        val singleTrackBytes = finalJournalBytes - afterAdmissionBytes
        println(
            "471-track ledger transitions: base=${baseBefore.size}B; " +
                "header=${headerBytes}B/${"%.3f".format(headerElapsedNanos / 1_000_000.0)}ms; " +
                "471-row batch=${admissionBytes}B/" +
                "${"%.3f".format(admissionElapsedNanos / 1_000_000.0)}ms; " +
                "single-track=${singleTrackBytes}B/" +
                "${"%.3f".format(singleTrackElapsedNanos / 1_000_000.0)}ms",
        )

        assertArrayEquals(baseBefore, base.readBytes())
        assertTrue("header delta must be under 5% of the full base", headerBytes * 20 < baseBefore.size)
        assertTrue("single-track delta must be under 5% of the full base", singleTrackBytes * 20 < baseBefore.size)
        assertTrue("batch delta must remain smaller than a full rewrite", admissionBytes < baseBefore.size)
        assertTrue(
            "a cached single-track transition should complete within 5s",
            singleTrackElapsedNanos < 5_000_000_000L,
        )
    }

    private fun plannedLedger(trackCount: Int, jobId: String): IndexingJobLedger {
        val generation = "poweramp-provider-snapshot-v3-sha256:" + "7".repeat(64)
        val audioSpec = V2IndexingLedgerPlanner.createEmbeddingSpec(
            EmbeddingSpecInput(
                preprocessingSpecId = V2IndexingWorkPolicy.PREPROCESSING_SPEC_ID,
                decoderPolicyId = V2IndexingWorkPolicy.DECODER_POLICY_ID,
                inferenceBackendPolicyId = V2IndexingWorkPolicy.INFERENCE_BACKEND_POLICY_ID,
                outputDimension = 768,
                modelArtifactSha256 = mapOf(
                    "mert" to "a".repeat(64),
                    "clamp3_audio" to "b".repeat(64),
                ),
            ),
        )
        val durationUs = 10_000_000L
        val sourceSamples = 480_000L
        val exact24k = 240_000L
        val selectedTracks = (0 until trackCount).map { ordinal ->
            val powerampId = ordinal + 1L
            val path = "/storage/emulated/0/Music/test-$ordinal.flac"
            SelectedTrackInput(
                powerampFileId = powerampId,
                providerSnapshotGeneration = generation,
                providerRow = V2ProviderPathRowEvidence(
                    powerampFileId = powerampId,
                    physicalPath = path,
                    providerPhysicalPath = path,
                    artist = "Artist $ordinal",
                    album = "Album",
                    title = "Title $ordinal",
                    offsetMs = 0L,
                    durationMs = 10_000L,
                    cueSourceImageFolderId = null,
                ),
                displayMetadata = DisplayTrackMetadata(
                    artist = "Artist $ordinal",
                    album = "Album",
                    title = "Title $ordinal",
                ),
                normalizedMetadata = NormalizedTrackMetadata(
                    normalizationSpecId = V2IndexingWorkPolicy.METADATA_NORMALIZATION_SPEC_ID,
                    artist = "artist $ordinal",
                    album = "album",
                    title = "title $ordinal",
                    metadataKey = "artist-$ordinal|album|title-$ordinal|10000",
                ),
                physicalPath = path,
                sourceFingerprint = SourceFingerprint(
                    fingerprintSpecId = V2FixedRegionSampling.SPEC_ID,
                    sizeBytes = 1_000L + ordinal,
                    lastModifiedEpochMs = 1L,
                    fileKey = null,
                    sampledContentSha256 = ordinal.toString(16).padStart(64, '0'),
                    fullContentSha256 = null,
                ),
                finalizedAudioSpan = FinalizedAudioSpanEvidence(
                    kind = V2ResolvedAudioSpanKind.WHOLE_FILE,
                    authority = V2AudioSpanAuthority.PROVISIONAL_END_OF_STREAM,
                    executionBoundaryRequirement =
                        V2ExecutionBoundaryRequirement.VERIFY_END_OF_STREAM_AND_RECONCILE,
                    providerSpan = V2ProviderSpanEvidence(0L, durationUs, durationUs),
                    cueClassification = V2CueClassificationEvidence(
                        providerGroupRowCount = 1,
                        logicalRowCount = 1,
                        nonZeroOffsetRowIds = emptyList(),
                        rawSourceImageRowIds = emptyList(),
                    ),
                    container = V2AudioContainerEvidence(
                        physicalPath = path,
                        audioTrackIndex = 0,
                        durationUsEstimate = durationUs,
                        sampleRateHz = 48_000,
                        channelCount = 2,
                        mime = "audio/flac",
                    ),
                    startUs = 0L,
                    endExclusiveUs = durationUs,
                    startSourceSample = 0L,
                    endSourceSampleExclusive = sourceSamples,
                    sourceSampleCount = sourceSamples,
                    exactSampleCount24k = exact24k,
                    expectedWork = V2AudioSpanMath.expectedWorkFor24kSamples(exact24k),
                ),
            )
        }
        return V2IndexingLedgerPlanner.planJob(
            providerSnapshot = PowerampProviderSnapshotEvidence(
                libraryGeneration = generation,
                acquisition = V2ProviderSnapshotAcquisitionEvidence(
                    queryUri = "content://com.maxmpz.audioplayer.data/files",
                    requestedColumns = listOf("_id", "duration", "path"),
                    returnedColumns = listOf("_id", "duration", "path"),
                    rowCount = trackCount,
                    cursorExhaustedNormally = true,
                ),
            ),
            embeddingSpec = audioSpec,
            textRetrievalSpec = V2IndexingLedgerPlanner.createTextRetrievalSpec(
                TextRetrievalSpecInput(
                    compatibleAudioEmbeddingSpecId = audioSpec.specId,
                    textModelSha256 = "1".repeat(64),
                    tokenizerModelSha256 = V2IndexingWorkPolicy.TEXT_TOKENIZER_MODEL_SHA256,
                    tokenizerPolicyId = V2IndexingWorkPolicy.TEXT_TOKENIZER_POLICY_ID,
                    tokenizerRuntimeContractSha256 =
                        V2IndexingWorkPolicy.TEXT_TOKENIZER_RUNTIME_CONTRACT_SHA256,
                    outputSpaceId = V2IndexingWorkPolicy.TEXT_OUTPUT_SPACE_ID,
                    outputDimension = audioSpec.outputDimension,
                    inferenceBackendPolicyId = V2IndexingWorkPolicy.TEXT_INFERENCE_BACKEND_POLICY_ID,
                ),
            ),
            runtimeFingerprint = IndexingRuntimeFingerprint(
                appVersionCode = 2_000_000L,
                appBuildId = "test-build",
                decoderRuntimeId = "test-decoder",
                platformFingerprint = "test-platform",
            ),
            selectedTracks = selectedTracks,
            rebuildDerivedIndexes = true,
            createdAtEpochMs = 900L,
            jobId = jobId,
        )
    }

    private fun journalFile(directory: File, jobId: String): File = File(
        directory,
        "$jobId${V2IndexingLedgerFileNamespace.JOURNAL_SUFFIX}",
    )

    private class CountingBaseFileIo : V2LedgerBaseFileIo {
        var reads: Int = 0
            private set

        override fun hasCommittedFile(file: File): Boolean = file.exists()

        override fun openRead(file: File): InputStream {
            reads += 1
            return FileInputStream(file)
        }

        override fun write(file: File, writer: (OutputStream) -> Unit) {
            FileOutputStream(file).use { output ->
                writer(output)
                output.flush()
                output.fd.sync()
            }
        }
    }

    private object TestBaseFileIo : V2LedgerBaseFileIo {
        override fun hasCommittedFile(file: File): Boolean = file.exists()

        override fun openRead(file: File): InputStream = FileInputStream(file)

        override fun write(file: File, writer: (OutputStream) -> Unit) {
            FileOutputStream(file).use { output ->
                writer(output)
                output.flush()
                output.fd.sync()
            }
        }
    }

    private class FailSecondBaseWriteIo : V2LedgerBaseFileIo {
        private var writes = 0

        override fun hasCommittedFile(file: File): Boolean = file.exists()

        override fun openRead(file: File): InputStream = FileInputStream(file)

        override fun write(file: File, writer: (OutputStream) -> Unit) {
            writes += 1
            if (writes == 2) throw SimulatedCompactionCrash()
            TestBaseFileIo.write(file, writer)
        }
    }

    private class SimulatedCompactionCrash : RuntimeException()

    private companion object {
        const val FRAME_HEADER_BYTES = 8 + 4 + 4 + 32
    }
}
