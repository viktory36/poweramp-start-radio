package com.powerampstartradio.indexing

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream

class V2ActiveLibraryCatalogStoreTest {
    @Test
    fun `codec round trips a complete deterministic catalog`() {
        val first = encode(catalog())
        val second = encode(catalog())
        assertArrayEquals(first, second)

        val restored = V2ActiveLibraryCatalogCodec.read(
            input = ByteArrayInputStream(first),
            expectedDatabaseGenerationId = "database-a",
            expectedActivationBindingId = "activation-a",
            expectedManifestSha256 = MANIFEST_SHA,
            expectedDatabaseTrackCount = 3,
        )

        assertEquals(catalog().generationBinding, restored.generationBinding)
        assertEquals(catalog().bindings, restored.bindings)
        assertEquals(catalog().quarantinedTracks, restored.quarantinedTracks)
        assertEquals(catalog().unboundPowerampFileIds, restored.unboundPowerampFileIds)
    }

    @Test
    fun `codec rejects another database generation or incomplete coverage`() {
        val encoded = encode(catalog())
        assertThrows(IllegalArgumentException::class.java) {
            V2ActiveLibraryCatalogCodec.read(
                ByteArrayInputStream(encoded),
                expectedDatabaseGenerationId = "database-b",
                expectedActivationBindingId = "activation-a",
                expectedManifestSha256 = MANIFEST_SHA,
                expectedDatabaseTrackCount = 3,
            )
        }
        assertThrows(IllegalArgumentException::class.java) {
            V2ActiveLibraryCatalogCodec.read(
                ByteArrayInputStream(encoded),
                expectedDatabaseGenerationId = "database-a",
                expectedActivationBindingId = "activation-a",
                expectedManifestSha256 = MANIFEST_SHA,
                expectedDatabaseTrackCount = 4,
            )
        }
    }

    @Test
    fun `codec rejects a truncated artifact`() {
        val encoded = encode(catalog())
        assertThrows(Exception::class.java) {
            V2ActiveLibraryCatalogCodec.read(
                ByteArrayInputStream(encoded.copyOf(encoded.size - 3)),
                expectedDatabaseGenerationId = "database-a",
                expectedActivationBindingId = "activation-a",
                expectedManifestSha256 = MANIFEST_SHA,
                expectedDatabaseTrackCount = 3,
            )
        }
    }

    @Test
    fun `codec rejects same-length payload corruption`() {
        val encoded = encode(catalog())
        encoded[encoded.size / 2] = (encoded[encoded.size / 2].toInt() xor 1).toByte()
        assertThrows(Exception::class.java) {
            V2ActiveLibraryCatalogCodec.read(
                ByteArrayInputStream(encoded),
                expectedDatabaseGenerationId = "database-a",
                expectedActivationBindingId = "activation-a",
                expectedManifestSha256 = MANIFEST_SHA,
                expectedDatabaseTrackCount = 3,
            )
        }
    }

    private fun encode(catalog: V2ActiveLibraryCatalog): ByteArray =
        ByteArrayOutputStream().also {
            V2ActiveLibraryCatalogCodec.write(
                catalog = catalog,
                activationBindingId = "activation-a",
                manifestSha256 = MANIFEST_SHA,
                output = it,
            )
        }
            .toByteArray()

    private fun catalog() = V2ActiveLibraryCatalog(
        generationBinding = V2ActiveLibraryGenerationBinding(
            databaseGenerationId = "database-a",
            providerGenerationId = "poweramp-a",
        ),
        bindings = listOf(
            V2ActiveLibraryBinding(
                trackId = 2,
                powerampFileId = 102,
                evidence = V2ActiveLibraryBindingEvidence.LEGACY_EXACT_MUSIC_RELATIVE_PATH,
            ),
            V2ActiveLibraryBinding(
                trackId = 1,
                powerampFileId = 101,
                evidence = V2ActiveLibraryBindingEvidence.EXACT_V2_RECEIPT_SPAN,
            ),
        ),
        quarantinedTracks = listOf(
            V2ActiveLibraryQuarantinedTrack(
                trackId = 3,
                reason = V2ActiveLibraryQuarantineReason.NO_CURRENT_PROVIDER_BINDING,
            ),
        ),
        unboundPowerampFileIds = listOf(202, 201),
    )

    private companion object {
        const val MANIFEST_SHA =
            "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    }
}
