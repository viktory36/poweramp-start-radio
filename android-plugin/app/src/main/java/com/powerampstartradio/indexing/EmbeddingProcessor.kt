package com.powerampstartradio.indexing

import android.util.Log
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Orchestrates MuQ-MuLan inference for on-device embedding generation.
 *
 * Flamingo inference and fusion are handled directly in IndexingService
 * via FlamingoInference's two-phase GPU API.
 */
class EmbeddingProcessor(
    private val mulanModel: MuLanInference?,
    private val flamingoProjection: FloatMatrix?,  // unused here, kept for API compat
    private val fusedProjection: FloatMatrix?,      // unused here, kept for API compat
) {

    companion object {
        private const val TAG = "EmbeddingProcessor"
        private const val MULAN_DIM = 512
        private const val FUSED_DIM = 512

        /**
         * Load projection matrices from the embedding database metadata table.
         *
         * The desktop indexer stores these as raw float32 bytes in the metadata table's
         * TEXT column. SQLite preserves the bytes regardless of column affinity.
         */
        fun loadProjectionMatrix(
            db: android.database.sqlite.SQLiteDatabase,
            key: String,
            rows: Int,
            cols: Int
        ): FloatMatrix? {
            return try {
                // Use SQLiteStatement to read large blobs directly, bypassing the
                // 2MB CursorWindow limit (flamingo_projection is ~7MB).
                val stmt = db.compileStatement(
                    "SELECT value FROM metadata WHERE key = ?"
                )
                stmt.bindString(1, key)
                val pfd = stmt.simpleQueryForBlobFileDescriptor() ?: run {
                    stmt.close()
                    return null
                }
                val blob = android.os.ParcelFileDescriptor.AutoCloseInputStream(pfd)
                    .use { it.readBytes() }
                stmt.close()

                val expectedSize = rows * cols * 4  // float32
                if (blob.size != expectedSize) {
                    Log.w(TAG, "$key: expected ${expectedSize} bytes, got ${blob.size}")
                    return null
                }
                val buffer = ByteBuffer.wrap(blob).order(ByteOrder.LITTLE_ENDIAN)
                val data = FloatArray(rows * cols)
                for (i in data.indices) {
                    data[i] = buffer.getFloat()
                }
                FloatMatrix(data, rows, cols)
            } catch (e: Exception) {
                Log.e(TAG, "Failed to load $key projection matrix", e)
                null
            }
        }
    }

    private val audioDecoder = AudioDecoder()

    /**
     * Result of processing a single track.
     */
    data class EmbeddingResult(
        val mulanEmbedding: FloatArray?,      // 512d
        val flamingoEmbedding: FloatArray?,   // 3584d (raw) or 512d (reduced)
        val flamingoReduced: FloatArray?,     // 512d (after projection)
        val fusedEmbedding: FloatArray?,      // 512d
    )

    /**
     * Process a single audio file and generate all embeddings.
     *
     * Supports two-pass sequential loading: in pass 2 (Flamingo only), provide
     * the MuLan embedding from pass 1 via [existingMulan] so fusion can proceed
     * without the MuLan model being loaded.
     *
     * @param audioFile Path to the audio file
     * @param existingMulan Pre-computed MuLan embedding for fusion (pass 2 of sequential loading)
     * @param preDecodedAudio Pre-decoded audio to skip redundant decode+resample.
     *   If provided and its sample rate matches the model's requirement, it's used directly.
     *   This enables decode-once optimization: decode at 24kHz for MuLan, resample 24→16kHz,
     *   cache the 16kHz audio, and pass it to the Flamingo pass.
     * @param onProgress Optional callback for status updates
     * @return All computed embeddings, or null on complete failure
     */
    fun processTrack(
        audioFile: File,
        existingMulan: FloatArray? = null,
        preDecodedAudio: AudioDecoder.DecodedAudio? = null,
        onProgress: ((String) -> Unit)? = null,
        onChunkDone: (() -> Unit)? = null,
    ): EmbeddingResult? {
        var mulanEmbedding: FloatArray? = null

        // MuQ-MuLan: decode at 24kHz
        if (mulanModel != null) {
            onProgress?.invoke("Decoding ${audioFile.name}...")
            val audio24k = if (preDecodedAudio?.sampleRate == 24000) preDecodedAudio
                           else audioDecoder.decode(audioFile, 24000, maxDurationS = 900)
            if (audio24k != null) {
                onProgress?.invoke("MuQ-MuLan inference...")
                mulanEmbedding = mulanModel.generateEmbedding(audio24k, onClipDone = onChunkDone)
            }
        }

        // For fusion: use inferred MuLan embedding, or fall back to provided one
        val mulanForFusion = mulanEmbedding ?: existingMulan

        if (mulanEmbedding == null && existingMulan == null) {
            Log.w(TAG, "MuLan failed for ${audioFile.name}")
            return null
        }

        return EmbeddingResult(
            mulanEmbedding = mulanEmbedding,
            flamingoEmbedding = null,
            flamingoReduced = null,
            fusedEmbedding = null,
        )
    }

    fun close() {
        mulanModel?.close()
    }
}

/**
 * Simple row-major float matrix for projection operations.
 *
 * Stored as a flat float array with explicit dimensions.
 */
class FloatMatrix(
    val data: FloatArray,
    val rows: Int,
    val cols: Int
) {
    init {
        require(data.size == rows * cols) {
            "Data size ${data.size} doesn't match ${rows}x${cols}"
        }
    }

    /**
     * Matrix-vector multiply: result[rows] = M[rows, cols] * v[cols]
     */
    fun multiplyVector(v: FloatArray): FloatArray {
        require(v.size == cols) { "Vector size ${v.size} doesn't match matrix cols $cols" }
        val result = FloatArray(rows)
        for (i in 0 until rows) {
            var sum = 0f
            val rowOffset = i * cols
            for (j in 0 until cols) {
                sum += data[rowOffset + j] * v[j]
            }
            result[i] = sum
        }
        return result
    }

    /**
     * Transpose this matrix: [rows, cols] -> [cols, rows]
     */
    fun transpose(): FloatMatrix {
        val transposed = FloatArray(data.size)
        for (i in 0 until rows) {
            for (j in 0 until cols) {
                transposed[j * rows + i] = data[i * cols + j]
            }
        }
        return FloatMatrix(transposed, cols, rows)
    }
}
