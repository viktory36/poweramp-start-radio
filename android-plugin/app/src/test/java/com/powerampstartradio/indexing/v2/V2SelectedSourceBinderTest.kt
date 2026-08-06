package com.powerampstartradio.indexing.v2

import java.io.File
import java.nio.file.Files
import java.text.Normalizer
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class V2SelectedSourceBinderTest {
    @get:Rule
    val temporaryFolder = TemporaryFolder()

    @Test
    fun `single selected symlink binds to its verified canonical target`() {
        val target = temporaryFolder.newFile("target.flac")
        val alias = symlink("alias.flac", target)
        val candidate = candidate(10L, alias)

        val binding = V2SelectedSourceBinder().bind(listOf(candidate)).getValue(10L)

        assertEquals(alias.path, binding.providerPathKey)
        assertEquals(target.canonicalPath, binding.canonicalSourceFile.path)
        assertTrue(binding.providerPathIsAlias)
    }

    @Test
    fun `distinct selected provider paths to one canonical file fail typed`() {
        val target = temporaryFolder.newFile("target.flac")
        val firstAlias = symlink("first.flac", target)
        val secondAlias = symlink("second.flac", target)

        val error = assertThrows(V2IndexingPreflightException::class.java) {
            V2SelectedSourceBinder().bind(
                listOf(candidate(10L, firstAlias), candidate(20L, secondAlias)),
            )
        }

        assertEquals(
            V2IndexingPreflightFailureCode.SOURCE_CANONICAL_ALIAS_COLLISION,
            error.code,
        )
        assertEquals(20L, error.powerampFileId)
    }

    @Test
    fun `CUE rows sharing one lexical provider path may share one target`() {
        val target = temporaryFolder.newFile("image.flac")
        val alias = symlink("album.flac", target)
        val bindings = V2SelectedSourceBinder().bind(
            listOf(candidate(10L, alias), candidate(11L, alias)),
        )

        assertEquals(2, bindings.size)
        assertEquals(
            bindings.getValue(10L).canonicalSourceFile,
            bindings.getValue(11L).canonicalSourceFile,
        )
    }

    @Test
    fun `symlink retarget before persistence fails as source changed`() {
        val firstTarget = temporaryFolder.newFile("first-target.flac")
        val secondTarget = temporaryFolder.newFile("second-target.flac")
        val alias = symlink("alias.flac", firstTarget)
        val candidate = candidate(10L, alias)
        val binder = V2SelectedSourceBinder()
        val expected = binder.bind(listOf(candidate))

        Files.delete(alias.toPath())
        Files.createSymbolicLink(alias.toPath(), secondTarget.toPath())

        val error = assertThrows(V2IndexingPreflightException::class.java) {
            binder.requireStillBound(listOf(candidate), expected)
        }
        assertEquals(V2IndexingPreflightFailureCode.SOURCE_CHANGED, error.code)
        assertEquals(10L, error.powerampFileId)
    }

    @Test
    fun `supplied source must resolve to provider target`() {
        val provider = temporaryFolder.newFile("provider.flac")
        val wrong = temporaryFolder.newFile("wrong.flac")
        val error = assertThrows(V2IndexingPreflightException::class.java) {
            V2SelectedSourceBinder().bind(
                listOf(candidate(10L, provider, supplied = wrong)),
            )
        }
        assertEquals(V2IndexingPreflightFailureCode.SOURCE_CHANGED, error.code)
    }

    @Test
    fun `decomposed provider filename remains the filesystem input`() {
        val decomposedName = Normalizer.normalize("Café.flac", Normalizer.Form.NFD)
        val source = temporaryFolder.newFile(decomposedName)

        val binding = V2SelectedSourceBinder()
            .bind(listOf(candidate(10L, source)))
            .getValue(10L)

        assertEquals(source.canonicalPath, binding.canonicalSourceFile.path)
        assertEquals(
            Normalizer.normalize(source.canonicalPath, Normalizer.Form.NFC),
            binding.canonicalPathKey,
        )
        assertFalse(binding.providerPathIsAlias)
        assertTrue(binding.canonicalSourceFile.isFile)
    }

    private fun candidate(
        id: Long,
        provider: File,
        supplied: File = provider,
    ) = V2SelectedSourceCandidate(
        powerampFileId = id,
        providerPathKey = provider.path,
        providerPhysicalPath = provider.path,
        suppliedSourceFile = supplied,
    )

    private fun symlink(name: String, target: File): File =
        File(temporaryFolder.root, name).also { alias ->
            Files.createSymbolicLink(alias.toPath(), target.toPath())
        }
}
