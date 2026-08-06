package com.powerampstartradio.indexing

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import java.security.MessageDigest
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class OfficialSentencePieceTokenizerInstrumentedTest {
    private lateinit var modelFile: File

    @Before
    fun requirePinnedModel() {
        modelFile = File(
            InstrumentationRegistry.getInstrumentation().targetContext.filesDir,
            "sentencepiece.bpe.model",
        )
        check(modelFile.isFile) { "Push the pinned sentencepiece.bpe.model before this test" }
        assertEquals(MODEL_SHA256, sha256(modelFile))
    }

    @Test
    fun matchesReferenceXlmRobertaIdsAcrossScriptsAndNormalizationEdges() {
        OfficialSentencePieceTokenizer(modelFile, seqLen = 16).use { tokenizer ->
            assertPrefix(tokenizer, "Bonobo - Kerala", 0, 10529, 58385, 20, 104219, 2)
            assertPrefix(
                tokenizer,
                "بونوبو موسيقى هادئة",
                0, 676, 4349, 17709, 211239, 9766, 917, 826, 26430, 2,
            )
            assertPrefix(tokenizer, "ཀཱི", 0, 6, 248773, 3, 2)
            assertPrefix(tokenizer, "\u0001a", 0, 10, 2)
            assertPrefix(tokenizer, "a\u0344b", 0, 10, 246635, 4868, 275, 2)
            assertPrefix(tokenizer, "👩‍🎤", 0, 6, 244785, 6, 246134, 2)
        }
    }

    @Test
    fun segmentsLongTextWithoutDiscardingTheTailAndRejectsUseAfterClose() {
        val tokenizer = OfficialSentencePieceTokenizer(modelFile, seqLen = 8)
        val segments = tokenizer.encodeSegments("ambient ".repeat(100))
        check(segments.size > 1)
        assertEquals(8, segments.first().inputIds.size)
        assertEquals(8, segments.first().attentionMask.count { it == 1 })
        assertEquals(OfficialSentencePieceTokenizer.BOS_ID, segments.first().inputIds.first())
        assertEquals(OfficialSentencePieceTokenizer.EOS_ID, segments.last().inputIds.last())
        segments.dropLast(1).forEach { assertEquals(8, it.contributionTokenCount) }
        check(segments.last().contributionTokenCount in 1..8)
        tokenizer.close()
        tokenizer.close()
        assertThrows(IllegalStateException::class.java) { tokenizer.encodeSegments("ambient") }
    }

    @Test
    fun replacesUnpairedJavaSurrogatesDeterministically() {
        OfficialSentencePieceTokenizer(modelFile, seqLen = 16).use { tokenizer ->
            val malformed = tokenizer.encode("x\uD800y").first
            val replacement = tokenizer.encode("x\uFFFDy").first
            assertArrayEquals(replacement, malformed)
        }
    }

    private fun assertPrefix(
        tokenizer: OfficialSentencePieceTokenizer,
        query: String,
        vararg expected: Int,
    ) {
        val (ids, mask) = tokenizer.encode(query)
        assertArrayEquals(expected, ids.copyOf(mask.count { it == 1 }))
    }

    private fun sha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().buffered().use { input ->
            val buffer = ByteArray(64 * 1024)
            while (true) {
                val count = input.read(buffer)
                if (count < 0) break
                digest.update(buffer, 0, count)
            }
        }
        return digest.digest().joinToString("") { byte ->
            (byte.toInt() and 0xff).toString(16).padStart(2, '0')
        }
    }

    private companion object {
        const val MODEL_SHA256 =
            "cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865"
    }
}
