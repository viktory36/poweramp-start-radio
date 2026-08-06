package com.powerampstartradio.indexing.v2

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File
import java.io.RandomAccessFile

class V2IndexingPreflightPrimitivesTest {
    @get:Rule
    val temporaryFolder = TemporaryFolder()

    @Test
    fun `sample regions merge duplicate and overlapping bytes`() {
        val region = V2FixedRegionSampling.REGION_BYTES.toLong()
        assertEquals(
            listOf(V2ByteRegion(0L, 1L)),
            V2FixedRegionSampling.regions(1L),
        )
        assertEquals(
            listOf(V2ByteRegion(0L, region * 2L)),
            V2FixedRegionSampling.regions(region * 2L),
        )
        assertEquals(
            listOf(
                V2ByteRegion(0L, region),
                V2ByteRegion(region + region / 2L, region),
                V2ByteRegion(region * 3L, region),
            ),
            V2FixedRegionSampling.regions(region * 4L),
        )
    }

    @Test
    fun `production source fingerprint is deterministic and binds every byte`() {
        val region = V2FixedRegionSampling.REGION_BYTES
        val file = temporaryFolder.newFile("album-image.flac")
        file.outputStream().buffered().use { output ->
            val block = ByteArray(region) { index -> (index % 251).toByte() }
            repeat(4) { output.write(block) }
        }
        val originalModified = file.lastModified()
        val fingerprinter = V2ExactSourceFingerprinter()
        val first = fingerprinter.fingerprint(file)
        val same = fingerprinter.fingerprint(file)
        assertEquals(first, same)
        assertEquals(V2IndexingLedgerIds.FULL_CONTENT_FINGERPRINT_SPEC_ID, first.fingerprintSpecId)
        assertEquals(file.length(), first.sizeBytes)
        assertEquals(null, first.sampledContentSha256)
        assertEquals(V2FileSha256.digest(file), first.fullContentSha256)

        RandomAccessFile(file, "rw").use { random ->
            random.seek(file.length() / 2L)
            val previous = random.readByte()
            random.seek(file.length() / 2L)
            random.writeByte(previous.toInt() xor 0xff)
        }
        assertTrue(file.setLastModified(originalModified))
        val changed = fingerprinter.fingerprint(file)
        assertNotEquals(first.fullContentSha256, changed.fullContentSha256)
    }

    @Test
    fun `shared physical source is fingerprinted once for many logical rows`() {
        val file = temporaryFolder.newFile("shared.flac").apply {
            writeBytes(ByteArray(8_192) { 7 })
        }
        var calls = 0
        val delegate = V2SourceFingerprintProvider { source ->
            calls++
            SourceFingerprint(
                fingerprintSpecId = "test-source-v1",
                sizeBytes = source.length(),
                lastModifiedEpochMs = source.lastModified(),
                fileKey = null,
                sampledContentSha256 = "a".repeat(64),
                fullContentSha256 = null,
            )
        }
        val deduplicating = V2DeduplicatingSourceFingerprintProvider(delegate)

        val first = deduplicating.fingerprint(file)
        val second = deduplicating.fingerprint(File(file.parentFile, "./${file.name}"))
        assertEquals(first, second)
        assertEquals(1, calls)
    }

    @Test
    fun `full artifact SHA-256 matches standard vector`() {
        val file = temporaryFolder.newFile("artifact.tflite").apply {
            writeText("abc")
        }
        assertEquals(
            "ba7816bf8f01cfea414140de5dae2223" +
                "b00361a396177a9cb410ff61f20015ad",
            V2FileSha256.digest(file),
        )
    }
}
