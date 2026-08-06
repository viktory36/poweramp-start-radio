package com.powerampstartradio.indexing

import java.io.File

/** Exact XLM-R tokenization through the checkpoint's serialized SentencePiece model. */
class OfficialSentencePieceTokenizer(
    modelFile: File,
    private val seqLen: Int = 128,
) : AutoCloseable {
    data class EncodedSegment(
        val inputIds: IntArray,
        val attentionMask: IntArray,
        /** Number of new whole-query tokens represented by this model window. */
        val contributionTokenCount: Int,
    )

    companion object {
        const val CONTRACT_SPEC_ID =
            "sentencepiece-v0.2.1-rev-31646a467d2051eb904e0b45de3a73e91fe1c1e3-" +
                "xlm-roberta-model-native-encode-sp-unk0-to-3-else-plus1-bos0-eos2-pad1-seq128-v1"
        const val CONTRACT_SHA256 =
            "e3f1abde1d51a6747a252f99b276359f1353b3637e39f85670e8189baa65d8f3"

        const val BOS_ID = 0
        const val PAD_ID = 1
        const val EOS_ID = 2
        const val UNK_ID = 3

        internal fun xlmRobertaId(sentencePieceId: Int): Int {
            require(sentencePieceId >= 0) { "Invalid SentencePiece ID: $sentencePieceId" }
            return if (sentencePieceId == 0) UNK_ID else sentencePieceId + 1
        }
    }

    private var nativeHandle: Long

    init {
        NativeLibrary.ensureLoaded()
        require(seqLen >= 2) { "Sequence length must leave room for BOS and EOS" }
        require(modelFile.isFile) { "SentencePiece model not found: ${modelFile.absolutePath}" }
        nativeHandle = nativeCreate(modelFile.absolutePath)
        check(nativeHandle != 0L) { "SentencePiece native runtime returned an invalid handle" }
    }

    @Synchronized
    fun encode(text: String): Pair<IntArray, IntArray> {
        val segments = encodeSegments(text)
        require(segments.size == 1) {
            "Text requires ${segments.size} model windows; use encodeSegments"
        }
        return segments.single().let { it.inputIds to it.attentionMask }
    }

    /**
     * Match CLaMP3's host-side long-text policy: consecutive 128-token windows and, when
     * needed, a final overlapping 128-token context weighted only by the remaining new tokens.
     */
    @Synchronized
    fun encodeSegments(text: String): List<EncodedSegment> {
        check(nativeHandle != 0L) { "SentencePiece tokenizer is closed" }
        val sentencePieceIds = nativeEncode(nativeHandle, text)
        val allIds = IntArray(sentencePieceIds.size + 2)
        allIds[0] = BOS_ID
        sentencePieceIds.indices.forEach { index ->
            allIds[index + 1] = xlmRobertaId(sentencePieceIds[index])
        }
        allIds[allIds.lastIndex] = EOS_ID

        if (allIds.size <= seqLen) {
            return listOf(padSegment(allIds, allIds.size))
        }

        val fullWindowCount = allIds.size / seqLen
        val remainder = allIds.size % seqLen
        val segments = ArrayList<EncodedSegment>(fullWindowCount + if (remainder > 0) 1 else 0)
        for (window in 0 until fullWindowCount) {
            val start = window * seqLen
            segments += padSegment(
                allIds.copyOfRange(start, start + seqLen),
                seqLen,
            )
        }
        if (remainder > 0) {
            segments += padSegment(
                allIds.copyOfRange(allIds.size - seqLen, allIds.size),
                remainder,
            )
        }
        return segments
    }

    private fun padSegment(ids: IntArray, contributionTokenCount: Int): EncodedSegment {
        require(ids.size <= seqLen)
        val inputIds = IntArray(seqLen) { PAD_ID }
        val attentionMask = IntArray(seqLen)
        ids.copyInto(inputIds)
        for (index in ids.indices) {
            attentionMask[index] = 1
        }
        return EncodedSegment(inputIds, attentionMask, contributionTokenCount)
    }

    @Synchronized
    override fun close() {
        if (nativeHandle == 0L) return
        nativeDestroy(nativeHandle)
        nativeHandle = 0L
    }

    private external fun nativeCreate(modelPath: String): Long
    private external fun nativeEncode(handle: Long, text: String): IntArray
    private external fun nativeDestroy(handle: Long)

    private object NativeLibrary {
        init {
            System.loadLibrary("sentencepiece-jni")
        }

        fun ensureLoaded() = Unit
    }
}
