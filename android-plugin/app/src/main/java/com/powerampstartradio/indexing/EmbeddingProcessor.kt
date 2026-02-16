package com.powerampstartradio.indexing

import android.util.Log
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Orchestrates MuQ-MuLan + Flamingo inference and applies projection matrices
 * from the embedding database to produce final fused embeddings.
 *
 * Pipeline per track:
 * 1. Decode audio -> PCM
 * 2. Resample to 24kHz -> MuQ-MuLan ONNX -> 512d mulan embedding
 * 3. Resample to 16kHz -> Flamingo ONNX -> 3584d
 *    -> flamingo_projection[3584, 512] -> 512d flamingo embedding
 * 4. Concatenate [mulan_512, flamingo_512] -> 1024d
 *    -> fused_projection[512, 1024].T -> 512d fused embedding
 * 5. L2-normalize all embeddings
 */
class EmbeddingProcessor(
    private val mulanModel: MuLanInference?,
    private val flamingoModel: FlamingoInference?,
    private val flamingoProjection: FloatMatrix?,  // [512, 3584] (transposed from stored [3584, 512])
    private val fusedProjection: FloatMatrix?,      // [512, 1024] stored as row-major
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
                db.rawQuery(
                    "SELECT value FROM metadata WHERE key = ?",
                    arrayOf(key)
                ).use { cursor ->
                    if (!cursor.moveToFirst()) return null
                    val blob = cursor.getBlob(0)
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
                }
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
     * @param onProgress Optional callback for status updates
     * @return All computed embeddings, or null on complete failure
     */
    fun processTrack(
        audioFile: File,
        existingMulan: FloatArray? = null,
        onProgress: ((String) -> Unit)? = null,
    ): EmbeddingResult? {
        // Decode audio at both sample rates
        onProgress?.invoke("Decoding ${audioFile.name}...")

        var mulanEmbedding: FloatArray? = null
        var flamingoRawEmbedding: FloatArray? = null
        var flamingoReduced: FloatArray? = null
        var fusedEmbedding: FloatArray? = null

        // MuQ-MuLan: decode at 24kHz
        if (mulanModel != null) {
            val audio24k = audioDecoder.decode(audioFile, 24000)
            if (audio24k != null) {
                onProgress?.invoke("MuQ-MuLan inference...")
                mulanEmbedding = mulanModel.generateEmbedding(audio24k)
            }
        }

        // Flamingo: decode at 16kHz
        if (flamingoModel != null) {
            val audio16k = audioDecoder.decode(audioFile, 16000)
            if (audio16k != null) {
                onProgress?.invoke("Flamingo inference...")
                flamingoRawEmbedding = flamingoModel.generateEmbedding(audio16k)
            }
        }

        // For fusion: use inferred MuLan embedding, or fall back to provided one
        val mulanForFusion = mulanEmbedding ?: existingMulan

        if (mulanEmbedding == null && flamingoRawEmbedding == null && existingMulan == null) {
            Log.w(TAG, "Both models failed for ${audioFile.name}")
            return null
        }

        // Apply Flamingo projection: 3584d -> 512d
        if (flamingoRawEmbedding != null && flamingoProjection != null) {
            flamingoReduced = flamingoProjection.multiplyVector(flamingoRawEmbedding)
            l2Normalize(flamingoReduced)
        }

        // Fuse: concatenate [mulan_512, flamingo_512] and project
        if (mulanForFusion != null && flamingoReduced != null && fusedProjection != null) {
            onProgress?.invoke("Fusing embeddings...")

            // Concatenate: [mulan_512 | flamingo_512] -> 1024d
            val concatenated = FloatArray(MULAN_DIM * 2)
            mulanForFusion.copyInto(concatenated, 0)
            flamingoReduced.copyInto(concatenated, MULAN_DIM)

            // Apply fused projection: concatenated[1024] x projection[1024, 512] -> [512]
            // The projection matrix is stored as [target_dim x source_dim] = [512 x 1024]
            // So we compute: result = projection * concatenated (matrix-vector multiply)
            fusedEmbedding = fusedProjection.multiplyVector(concatenated)
            l2Normalize(fusedEmbedding)
        }

        return EmbeddingResult(
            mulanEmbedding = mulanEmbedding,  // null when model wasn't loaded (pass 2)
            flamingoEmbedding = flamingoRawEmbedding,
            flamingoReduced = flamingoReduced,
            fusedEmbedding = fusedEmbedding,
        )
    }

    fun close() {
        mulanModel?.close()
        flamingoModel?.close()
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
