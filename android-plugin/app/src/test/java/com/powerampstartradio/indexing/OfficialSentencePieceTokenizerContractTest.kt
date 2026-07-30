package com.powerampstartradio.indexing

import com.powerampstartradio.indexing.v2.V2IndexingWorkPolicy
import java.nio.charset.StandardCharsets
import java.security.MessageDigest
import org.junit.Assert.assertEquals
import org.junit.Test

class OfficialSentencePieceTokenizerContractTest {
    @Test
    fun contractDigestIsBoundToCanonicalRuntimePolicy() {
        val digest = MessageDigest.getInstance("SHA-256")
            .digest(
                OfficialSentencePieceTokenizer.CONTRACT_SPEC_ID.toByteArray(
                    StandardCharsets.UTF_8,
                ),
            )
            .joinToString("") { byte ->
                (byte.toInt() and 0xff).toString(16).padStart(2, '0')
            }
        assertEquals(OfficialSentencePieceTokenizer.CONTRACT_SHA256, digest)
        assertEquals(
            V2IndexingWorkPolicy.TEXT_TOKENIZER_POLICY_ID,
            OfficialSentencePieceTokenizer.CONTRACT_SPEC_ID,
        )
        assertEquals(
            V2IndexingWorkPolicy.TEXT_TOKENIZER_RUNTIME_CONTRACT_SHA256,
            OfficialSentencePieceTokenizer.CONTRACT_SHA256,
        )
    }

    @Test
    fun sentencePieceIdsUseTheXlmRobertaVocabularyMapping() {
        assertEquals(3, OfficialSentencePieceTokenizer.xlmRobertaId(0))
        assertEquals(2, OfficialSentencePieceTokenizer.xlmRobertaId(1))
        assertEquals(10529, OfficialSentencePieceTokenizer.xlmRobertaId(10528))
        assertEquals(248773, OfficialSentencePieceTokenizer.xlmRobertaId(248772))
    }
}
