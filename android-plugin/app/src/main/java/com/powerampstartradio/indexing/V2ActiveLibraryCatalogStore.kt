package com.powerampstartradio.indexing

import android.util.AtomicFile
import android.util.Log
import com.powerampstartradio.indexing.v2.V2IndexGenerationReader
import com.powerampstartradio.indexing.v2.V2ResolvedActiveIndexGeneration
import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File
import java.io.InputStream
import java.io.OutputStream
import java.security.DigestOutputStream
import java.security.MessageDigest

/** Compact, deterministic encoding for the last fully reconciled Poweramp binding catalog. */
internal object V2ActiveLibraryCatalogCodec {
    private const val MAGIC = 0x50535243
    private const val VERSION = 2
    private const val POLICY_ID = "v2-active-library-durable-cache-v2"
    private const val MAX_ROW_COUNT = 5_000_000
    private const val TRAILER = 0x5053524341545631L
    private const val SHA256_LENGTH = 32
    private const val MAX_ENCODED_BYTES = 64 * 1024 * 1024

    internal data class Header(
        val databaseGenerationId: String,
        val activationBindingId: String,
        val manifestSha256: String,
        val providerGenerationId: String,
        val databaseTrackCount: Int,
        val providerTrackCount: Int,
    )

    fun write(
        catalog: V2ActiveLibraryCatalog,
        activationBindingId: String,
        manifestSha256: String,
        output: OutputStream,
    ) {
        require(activationBindingId.isNotBlank()) { "activation binding is blank" }
        require(manifestSha256.isNotBlank()) { "manifest SHA-256 is blank" }
        val digest = MessageDigest.getInstance("SHA-256")
        val digestOutput = DigestOutputStream(output, digest)
        val data = DataOutputStream(digestOutput)
        data.writeInt(MAGIC)
        data.writeInt(VERSION)
        data.writeUTF(POLICY_ID)
        data.writeUTF(catalog.generationBinding.databaseGenerationId)
        data.writeUTF(activationBindingId)
        data.writeUTF(manifestSha256)
        data.writeUTF(catalog.generationBinding.providerGenerationId)
        data.writeInt(catalog.activeTrackIds.size + catalog.quarantinedTracks.size)
        data.writeInt(catalog.activeTrackIds.size + catalog.unboundPowerampFileIds.size)
        data.writeInt(catalog.bindings.size)
        catalog.bindings.forEach { binding ->
            data.writeLong(binding.trackId)
            data.writeLong(binding.powerampFileId)
            data.writeByte(binding.evidence.wireId)
            data.writeLong(binding.createdAtEpochSecond)
        }
        data.writeInt(catalog.quarantinedTracks.size)
        catalog.quarantinedTracks.forEach { track ->
            data.writeLong(track.trackId)
            data.writeByte(track.reason.wireId)
        }
        data.writeInt(catalog.unboundPowerampFileIds.size)
        catalog.unboundPowerampFileIds.sorted().forEach(data::writeLong)
        data.writeLong(TRAILER)
        data.flush()
        digestOutput.on(false)
        output.write(digest.digest())
        output.flush()
    }

    fun read(
        input: InputStream,
        expectedDatabaseGenerationId: String,
        expectedActivationBindingId: String,
        expectedManifestSha256: String,
        expectedDatabaseTrackCount: Int,
    ): V2ActiveLibraryCatalog {
        require(expectedDatabaseGenerationId.isNotBlank()) {
            "expected database generation is blank"
        }
        require(expectedDatabaseTrackCount > 0) { "expected database track count is invalid" }
        val encoded = input.readBoundedBytes(MAX_ENCODED_BYTES)
        require(encoded.size > SHA256_LENGTH) { "active-library cache is incomplete" }
        val payloadSize = encoded.size - SHA256_LENGTH
        val expectedDigest = encoded.copyOfRange(payloadSize, encoded.size)
        val payload = encoded.copyOf(payloadSize)
        require(
            MessageDigest.isEqual(
                MessageDigest.getInstance("SHA-256").digest(payload),
                expectedDigest,
            ),
        ) { "active-library cache SHA-256 does not match" }
        val data = DataInputStream(ByteArrayInputStream(payload))
        require(data.readInt() == MAGIC) { "active-library cache has the wrong magic" }
        require(data.readInt() == VERSION) { "active-library cache has an unsupported version" }
        require(data.readUTF() == POLICY_ID) { "active-library cache policy changed" }
        val databaseGenerationId = data.readUTF()
        val activationBindingId = data.readUTF()
        val manifestSha256 = data.readUTF()
        val providerGenerationId = data.readUTF()
        require(databaseGenerationId == expectedDatabaseGenerationId) {
            "active-library cache belongs to another database generation"
        }
        require(activationBindingId == expectedActivationBindingId) {
            "active-library cache activation binding changed"
        }
        require(manifestSha256 == expectedManifestSha256) {
            "active-library cache manifest changed"
        }
        require(providerGenerationId.isNotBlank()) {
            "active-library cache has no Poweramp generation"
        }
        val storedDatabaseTrackCount = data.readBoundedCount("database track")
        val storedProviderTrackCount = data.readBoundedCount("Poweramp track")
        require(storedDatabaseTrackCount == expectedDatabaseTrackCount) {
            "active-library cache database count changed"
        }

        val bindingCount = data.readBoundedCount("binding")
        require(bindingCount <= storedDatabaseTrackCount) {
            "active-library cache binding count exceeds its database"
        }
        val bindings = ArrayList<V2ActiveLibraryBinding>(bindingCount)
        repeat(bindingCount) {
            val trackId = data.readLong()
            val powerampFileId = data.readLong()
            val evidence = bindingEvidenceFromWireId(data.readUnsignedByte())
            val createdAtEpochSecond = data.readLong()
            bindings += V2ActiveLibraryBinding(
                trackId = trackId,
                powerampFileId = powerampFileId,
                evidence = evidence,
                createdAtEpochSecond = createdAtEpochSecond,
            )
        }

        val quarantinedCount = data.readBoundedCount("quarantined")
        require(quarantinedCount == storedDatabaseTrackCount - bindingCount) {
            "active-library cache quarantine count disagrees with its database"
        }
        val quarantined = ArrayList<V2ActiveLibraryQuarantinedTrack>(quarantinedCount)
        repeat(quarantinedCount) {
            val trackId = data.readLong()
            val reason = quarantineReasonFromWireId(data.readUnsignedByte())
            quarantined += V2ActiveLibraryQuarantinedTrack(trackId, reason)
        }

        val unboundCount = data.readBoundedCount("unbound Poweramp")
        require(unboundCount == storedProviderTrackCount - bindingCount) {
            "active-library cache unbound count disagrees with its Poweramp snapshot"
        }
        val unbound = ArrayList<Long>(unboundCount)
        repeat(unboundCount) { unbound += data.readLong() }
        require(data.readLong() == TRAILER) { "active-library cache is incomplete" }
        require(data.read() == -1) { "active-library cache has trailing data" }
        require(bindings.size + quarantined.size == storedDatabaseTrackCount) {
            "active-library cache does not cover the current database"
        }
        require(bindings.size + unbound.size == storedProviderTrackCount) {
            "active-library cache does not cover its Poweramp snapshot"
        }
        return V2ActiveLibraryCatalog(
            generationBinding = V2ActiveLibraryGenerationBinding(
                databaseGenerationId = databaseGenerationId,
                providerGenerationId = providerGenerationId,
            ),
            bindings = bindings,
            quarantinedTracks = quarantined,
            unboundPowerampFileIds = unbound,
        )
    }

    private fun DataInputStream.readBoundedCount(label: String): Int = readInt().also { count ->
        require(count in 0..MAX_ROW_COUNT) { "active-library cache has invalid $label count" }
    }

    private fun InputStream.readBoundedBytes(maxBytes: Int): ByteArray {
        val output = ByteArrayOutputStream(minOf(maxBytes, 2 * 1024 * 1024))
        val buffer = ByteArray(64 * 1024)
        var total = 0
        while (true) {
            val read = read(buffer)
            if (read < 0) break
            total += read
            require(total <= maxBytes) { "active-library cache exceeds its size limit" }
            output.write(buffer, 0, read)
        }
        return output.toByteArray()
    }

    fun readHeader(input: InputStream): Header {
        val data = DataInputStream(input)
        require(data.readInt() == MAGIC) { "active-library cache has the wrong magic" }
        require(data.readInt() == VERSION) { "active-library cache has an unsupported version" }
        require(data.readUTF() == POLICY_ID) { "active-library cache policy changed" }
        return Header(
            databaseGenerationId = data.readUTF(),
            activationBindingId = data.readUTF(),
            manifestSha256 = data.readUTF(),
            providerGenerationId = data.readUTF(),
            databaseTrackCount = data.readBoundedCount("database track"),
            providerTrackCount = data.readBoundedCount("Poweramp track"),
        )
    }

    private val V2ActiveLibraryBindingEvidence.wireId: Int
        get() = when (this) {
            V2ActiveLibraryBindingEvidence.EXACT_V2_RECEIPT_SPAN -> 1
            V2ActiveLibraryBindingEvidence.LEGACY_EXACT_ABSOLUTE_PATH -> 2
            V2ActiveLibraryBindingEvidence.LEGACY_EXACT_MUSIC_RELATIVE_PATH -> 3
        }

    private fun bindingEvidenceFromWireId(wireId: Int): V2ActiveLibraryBindingEvidence =
        when (wireId) {
            1 -> V2ActiveLibraryBindingEvidence.EXACT_V2_RECEIPT_SPAN
            2 -> V2ActiveLibraryBindingEvidence.LEGACY_EXACT_ABSOLUTE_PATH
            3 -> V2ActiveLibraryBindingEvidence.LEGACY_EXACT_MUSIC_RELATIVE_PATH
            else -> error("active-library cache has invalid binding evidence")
        }

    private val V2ActiveLibraryQuarantineReason.wireId: Int
        get() = when (this) {
            V2ActiveLibraryQuarantineReason.UNRESOLVED_EXACT_RECEIPT -> 1
            V2ActiveLibraryQuarantineReason.SPAN_SPECIFIC_REBUILD_REQUIRED -> 2
            V2ActiveLibraryQuarantineReason.PATH_TIMING_CONFLICT -> 3
            V2ActiveLibraryQuarantineReason.NO_CURRENT_PROVIDER_BINDING -> 4
        }

    private fun quarantineReasonFromWireId(wireId: Int): V2ActiveLibraryQuarantineReason =
        when (wireId) {
            1 -> V2ActiveLibraryQuarantineReason.UNRESOLVED_EXACT_RECEIPT
            2 -> V2ActiveLibraryQuarantineReason.SPAN_SPECIFIC_REBUILD_REQUIRED
            3 -> V2ActiveLibraryQuarantineReason.PATH_TIMING_CONFLICT
            4 -> V2ActiveLibraryQuarantineReason.NO_CURRENT_PROVIDER_BINDING
            else -> error("active-library cache has invalid quarantine reason")
        }
}

/** Durable optimization only: any read/write failure falls back to exact provider reconciliation. */
internal class V2ActiveLibraryCatalogStore(private val filesDir: File) {
    private val baseFile = File(filesDir, RELATIVE_PATH)
    private val atomicFile = AtomicFile(baseFile)

    fun read(activeGeneration: V2ResolvedActiveIndexGeneration): V2ActiveLibraryCatalog? =
        synchronized(IO_LOCK) {
        if (!baseFile.isFile && !File(baseFile.path + ".bak").isFile) return@synchronized null
        try {
            val header = atomicFile.openRead().use { raw ->
                V2ActiveLibraryCatalogCodec.readHeader(BufferedInputStream(raw))
            }
            if (header.databaseGenerationId != activeGeneration.manifest.generationId ||
                header.activationBindingId != activeGeneration.manifest.activationBindingId ||
                header.manifestSha256 != activeGeneration.manifestSha256
            ) {
                return@synchronized null
            }
            atomicFile.openRead().use { raw ->
                V2ActiveLibraryCatalogCodec.read(
                    input = BufferedInputStream(raw),
                    expectedDatabaseGenerationId = activeGeneration.manifest.generationId,
                    expectedActivationBindingId = activeGeneration.manifest.activationBindingId,
                    expectedManifestSha256 = activeGeneration.manifestSha256,
                    expectedDatabaseTrackCount = activeGeneration.manifest.trackCount,
                )
            }
        } catch (error: Exception) {
            Log.w(TAG, "Discarding unusable active-library cache", error)
            atomicFile.delete()
            null
        }
    }

    fun write(
        activeGeneration: V2ResolvedActiveIndexGeneration,
        catalog: V2ActiveLibraryCatalog,
    ): Boolean = synchronized(IO_LOCK) {
        var raw: java.io.FileOutputStream? = null
        try {
            require(
                catalog.generationBinding.databaseGenerationId ==
                    activeGeneration.manifest.generationId,
            ) { "Active-library cache belongs to another database generation" }
            requireStillActive(activeGeneration)
            check(
                baseFile.parentFile?.mkdirs() == true ||
                    baseFile.parentFile?.isDirectory == true,
            ) { "Unable to create the active-library cache directory" }
            raw = atomicFile.startWrite()
            V2ActiveLibraryCatalogCodec.write(
                catalog = catalog,
                activationBindingId = activeGeneration.manifest.activationBindingId,
                manifestSha256 = activeGeneration.manifestSha256,
                output = BufferedOutputStream(raw),
            )
            requireStillActive(activeGeneration)
            atomicFile.finishWrite(raw)
            true
        } catch (error: Exception) {
            raw?.let(atomicFile::failWrite)
            Log.w(TAG, "Could not persist the active-library cache", error)
            false
        }
    }

    fun delete() = synchronized(IO_LOCK) {
        atomicFile.delete()
    }

    fun deleteIfMatches(databaseGenerationId: String, providerGenerationId: String): Boolean =
        synchronized(IO_LOCK) {
            if (!baseFile.isFile && !File(baseFile.path + ".bak").isFile) {
                return@synchronized false
            }
            val matches = try {
                atomicFile.openRead().use { raw ->
                    V2ActiveLibraryCatalogCodec.readHeader(BufferedInputStream(raw)).let { header ->
                        header.databaseGenerationId == databaseGenerationId &&
                            header.providerGenerationId == providerGenerationId
                    }
                }
            } catch (error: Exception) {
                Log.w(TAG, "Deleting unreadable active-library cache", error)
                true
            }
            if (matches) atomicFile.delete()
            matches
        }

    private fun requireStillActive(expected: V2ResolvedActiveIndexGeneration) {
        val current = V2IndexGenerationReader.requireActive(filesDir)
        require(
            current.manifest.generationId == expected.manifest.generationId &&
                current.manifest.activationBindingId == expected.manifest.activationBindingId &&
                current.manifestSha256 == expected.manifestSha256,
        ) { "Active music index changed before the library cache was written" }
    }

    private companion object {
        const val RELATIVE_PATH = "indexing_v2/active-library-catalog-v1.bin"
        const val TAG = "V2ActiveLibraryCache"
        val IO_LOCK = Any()
    }
}
