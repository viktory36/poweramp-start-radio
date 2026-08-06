package com.powerampstartradio.indexing.v2

import java.io.File
import java.nio.file.Files
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertThrows
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class V2ProviderLexicalPathNormalizerTest {
    @get:Rule
    val temporaryFolder = TemporaryFolder()

    @Test
    fun `normalization is absolute NFC slash stable and lexical`() {
        assertEquals(
            "/music/Caf\u00e9/song.flac",
            V2StableProviderLexicalPathNormalizer.normalizeAbsolute(
                "/music//Cafe\u0301/./disc/../song.flac",
            ),
        )

        assertThrows(IllegalArgumentException::class.java) {
            V2StableProviderLexicalPathNormalizer.normalizeAbsolute("music/song.flac")
        }
        assertThrows(IllegalArgumentException::class.java) {
            V2StableProviderLexicalPathNormalizer.normalizeAbsolute("/music\\song.flac")
        }
    }

    @Test
    fun `normalization does not resolve or stat a symlink`() {
        val target = temporaryFolder.newFile("target.flac")
        val alias = File(temporaryFolder.root, "alias.flac")
        Files.createSymbolicLink(alias.toPath(), target.toPath())

        val normalized = V2StableProviderLexicalPathNormalizer.normalizeAbsolute(alias.path)

        assertEquals(alias.absolutePath, normalized)
        assertNotEquals(target.canonicalPath, normalized)
    }

    @Test
    fun `nonexistent absolute provider path remains valid identity evidence`() {
        val missing = File(temporaryFolder.root, "missing/album/song.flac")
        assertEquals(
            missing.absolutePath,
            V2StableProviderLexicalPathNormalizer.normalizeAbsolute(missing.absolutePath),
        )
    }
}
