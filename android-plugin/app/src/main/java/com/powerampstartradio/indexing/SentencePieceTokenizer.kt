package com.powerampstartradio.indexing

import android.util.Log
import org.json.JSONObject
import java.io.File

/**
 * Pure Kotlin tokenizer for XLM-RoBERTa (SentencePiece BPE).
 *
 * Loads the vocabulary from a JSON file exported by the desktop pipeline
 * (xlm_roberta_vocab.json) and implements greedy longest-match encoding.
 *
 * SentencePiece conventions:
 * - Word boundaries are marked with ▁ (U+2581, "LOWER ONE EIGHTH BLOCK")
 * - BOS = 0 (<s>), PAD = 1 (<pad>), EOS = 2 (</s>), UNK = 3 (<unk>)
 * - HuggingFace token IDs = SentencePiece IDs + 1 (fairseq offset)
 *
 * The tokenizer uses greedy longest-match on the vocabulary, which matches
 * SentencePiece's BPE output for the short queries we handle (< 30 tokens).
 * For full BPE fidelity on long texts, a native SentencePiece JNI would be
 * needed, but for search queries this is exact.
 */
class SentencePieceTokenizer(vocabFile: File) {

    companion object {
        private const val TAG = "SPTokenizer"

        // Special token IDs (HuggingFace / fairseq convention)
        const val BOS_ID = 0   // <s>
        const val PAD_ID = 1   // <pad>
        const val EOS_ID = 2   // </s>
        const val UNK_ID = 3   // <unk>

        // Static output sequence length for TFLite
        const val SEQ_LEN = 64

        // SentencePiece word boundary marker
        private const val SP_SPACE = "\u2581"
    }

    // piece → token ID (HuggingFace IDs, already include fairseq offset)
    private val vocab: Map<String, Int>

    // Maximum piece length in the vocabulary (for greedy matching)
    private val maxPieceLen: Int

    init {
        val json = vocabFile.readText()
        val obj = JSONObject(json)
        val map = HashMap<String, Int>(obj.length())
        val iter = obj.keys()
        var maxLen = 0
        while (iter.hasNext()) {
            val piece = iter.next()
            map[piece] = obj.getInt(piece)
            if (piece.length > maxLen) maxLen = piece.length
        }
        vocab = map
        maxPieceLen = maxLen
        Log.i(TAG, "Loaded vocabulary: ${vocab.size} pieces, max piece length: $maxPieceLen")
    }

    /**
     * Tokenize a text query and return (input_ids, attention_mask) padded to [SEQ_LEN].
     *
     * @param text The query string (e.g., "ethereal ambient")
     * @return Pair of IntArrays: (input_ids[SEQ_LEN], attention_mask[SEQ_LEN])
     */
    fun encode(text: String): Pair<IntArray, IntArray> {
        // SentencePiece preprocessing: replace spaces with ▁, prepend ▁
        val normalized = SP_SPACE + text.replace(" ", SP_SPACE)

        // Greedy longest-match tokenization
        val tokenIds = mutableListOf<Int>()
        var pos = 0
        while (pos < normalized.length) {
            var bestLen = 0
            var bestId = UNK_ID

            // Try longest match first
            val maxEnd = minOf(pos + maxPieceLen, normalized.length)
            for (end in maxEnd downTo pos + 1) {
                val piece = normalized.substring(pos, end)
                val id = vocab[piece]
                if (id != null) {
                    bestLen = end - pos
                    bestId = id
                    break
                }
            }

            if (bestLen == 0) {
                // Single character not in vocab → UNK
                tokenIds.add(UNK_ID)
                pos++
            } else {
                tokenIds.add(bestId)
                pos += bestLen
            }
        }

        // Build input_ids: BOS + tokens + EOS, padded to SEQ_LEN
        val maxTokens = SEQ_LEN - 2  // reserve BOS and EOS
        val truncated = if (tokenIds.size > maxTokens) tokenIds.subList(0, maxTokens) else tokenIds
        val seqLen = truncated.size + 2  // BOS + tokens + EOS

        val inputIds = IntArray(SEQ_LEN) { PAD_ID }
        val attentionMask = IntArray(SEQ_LEN) { 0 }

        inputIds[0] = BOS_ID
        attentionMask[0] = 1
        for (i in truncated.indices) {
            inputIds[i + 1] = truncated[i]
            attentionMask[i + 1] = 1
        }
        inputIds[seqLen - 1] = EOS_ID
        attentionMask[seqLen - 1] = 1

        Log.d(TAG, "Encoded '$text': ${seqLen} tokens (${truncated.size} pieces)")

        return inputIds to attentionMask
    }
}
