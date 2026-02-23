package com.powerampstartradio.indexing

import android.util.Log
import org.json.JSONObject
import java.io.File

/**
 * Pure Kotlin tokenizer for XLM-RoBERTa (SentencePiece Unigram model).
 *
 * Loads the vocabulary from a JSON file exported by the desktop pipeline
 * (xlm_roberta_vocab.json) with format: {piece: [token_id, log_prob_score]}.
 *
 * Uses Viterbi dynamic programming to find the highest-probability segmentation,
 * matching SentencePiece's Unigram encoding algorithm exactly.
 *
 * SentencePiece conventions:
 * - Word boundaries are marked with ▁ (U+2581, "LOWER ONE EIGHTH BLOCK")
 * - BOS = 0 (<s>), PAD = 1 (<pad>), EOS = 2 (</s>), UNK = 3 (<unk>)
 */
class SentencePieceTokenizer(vocabFile: File) {

    companion object {
        private const val TAG = "SPTokenizer"

        // Special token IDs
        const val BOS_ID = 0   // <s>
        const val PAD_ID = 1   // <pad>
        const val EOS_ID = 2   // </s>
        const val UNK_ID = 3   // <unk>

        // Static output sequence length for TFLite
        const val SEQ_LEN = 64

        // SentencePiece word boundary marker
        private const val SP_SPACE = "\u2581"
    }

    // piece → token ID
    private val vocab: HashMap<String, Int>

    // piece → Unigram log probability score (higher = more probable)
    private val scores: HashMap<String, Float>

    // Maximum piece length in the vocabulary (for Viterbi window)
    private val maxPieceLen: Int

    init {
        val json = vocabFile.readText()
        val obj = JSONObject(json)
        val vocabMap = HashMap<String, Int>(obj.length())
        val scoreMap = HashMap<String, Float>(obj.length())
        val iter = obj.keys()
        var maxLen = 0
        while (iter.hasNext()) {
            val piece = iter.next()
            val entry = obj.optJSONArray(piece)
            if (entry != null && entry.length() >= 2) {
                vocabMap[piece] = entry.getInt(0)
                scoreMap[piece] = entry.getDouble(1).toFloat()
            } else {
                // Legacy format fallback: just id
                vocabMap[piece] = obj.getInt(piece)
                scoreMap[piece] = -999f
            }
            if (piece.length > maxLen) maxLen = piece.length
        }
        vocab = vocabMap
        scores = scoreMap
        maxPieceLen = maxLen
        Log.i(TAG, "Loaded vocabulary: ${vocab.size} pieces, max piece length: $maxPieceLen")
    }

    /**
     * Tokenize a text query and return (input_ids, attention_mask) padded to [SEQ_LEN].
     *
     * Uses Viterbi DP to find the segmentation with the highest total log probability,
     * matching SentencePiece's Unigram encoding exactly.
     *
     * @param text The query string (e.g., "ethereal ambient")
     * @return Pair of IntArrays: (input_ids[SEQ_LEN], attention_mask[SEQ_LEN])
     */
    fun encode(text: String): Pair<IntArray, IntArray> {
        // SentencePiece preprocessing: replace spaces with ▁, prepend ▁
        val normalized = SP_SPACE + text.replace(" ", SP_SPACE)
        val n = normalized.length

        // Viterbi DP: dp[i] = best total score for normalized[0..i)
        val dp = FloatArray(n + 1) { Float.NEGATIVE_INFINITY }
        val backPtr = IntArray(n + 1) // backPtr[i] = start position of piece ending at i
        dp[0] = 0f

        for (i in 1..n) {
            val start = maxOf(0, i - maxPieceLen)
            for (j in start until i) {
                val piece = normalized.substring(j, i)
                val score = scores[piece]
                if (score != null && dp[j] + score > dp[i]) {
                    dp[i] = dp[j] + score
                    backPtr[i] = j
                }
            }
            // If no piece found ending at i, fall back to single char as UNK
            if (dp[i] == Float.NEGATIVE_INFINITY && dp[i - 1] > Float.NEGATIVE_INFINITY) {
                dp[i] = dp[i - 1] + -100f // heavy penalty for UNK
                backPtr[i] = i - 1
            }
        }

        // Backtrack to get pieces
        val pieces = mutableListOf<String>()
        var pos = n
        while (pos > 0) {
            val start = backPtr[pos]
            pieces.add(normalized.substring(start, pos))
            pos = start
        }
        pieces.reverse()

        // Map pieces to token IDs
        val tokenIds = pieces.map { vocab[it] ?: UNK_ID }

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
