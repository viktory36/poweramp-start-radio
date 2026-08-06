package com.powerampstartradio.indexing

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.google.ai.edge.litert.Accelerator
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import kotlin.math.sqrt
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class TextEmbeddingParityInstrumentedTest {
    @Test
    fun matchesPinnedHostLiteRtForShortMultilingualAndLongQueries() {
        val filesDir = InstrumentationRegistry.getInstrumentation().targetContext.filesDir
        val fixtureDir = File(filesDir, "device_acceptance/text_parity")
        val manifestFile = File(fixtureDir, "manifest.json")
        check(manifestFile.isFile) { "Stage the host text parity fixture before this test" }

        val manifest = JSONObject(manifestFile.readText(Charsets.UTF_8))
        val model = File(filesDir, "clamp3_text.tflite")
        val tokenizer = File(filesDir, "sentencepiece.bpe.model")
        assertEquals(manifest.getString("model_sha256"), sha256(model))
        assertEquals(manifest.getString("tokenizer_sha256"), sha256(tokenizer))

        OfficialSentencePieceTokenizer(tokenizer).use { processor ->
            val queries = manifest.getJSONArray("queries")
            for (index in 0 until queries.length()) {
                val query = queries.getJSONObject(index)
                val expectedContributions = query.getJSONArray("window_contribution_tokens")
                val expected = List(expectedContributions.length()) { contributionIndex ->
                    expectedContributions.getInt(contributionIndex)
                }
                val actual = processor.encodeSegments(query.getString("text"))
                    .map(OfficialSentencePieceTokenizer.EncodedSegment::contributionTokenCount)
                assertEquals("${query.getString("name")} window weights", expected, actual)
            }
        }

        Clamp3TextInference(
            modelFile = model,
            tokenizerModelFile = tokenizer,
            accelerator = Accelerator.CPU,
            strictAccelerator = true,
        ).let { inference ->
            try {
                assertEquals(Accelerator.CPU, inference.activeAccelerator)
                val queries = manifest.getJSONArray("queries")
                for (index in 0 until queries.length()) {
                    val query = queries.getJSONObject(index)
                    val referenceFile = File(fixtureDir, query.getString("embedding_file"))
                    assertEquals(query.getString("embedding_sha256"), sha256(referenceFile))
                    val expected = readFloatVector(referenceFile)
                    val actual = requireNotNull(inference.generateEmbedding(query.getString("text"))) {
                        "Text inference failed for ${query.getString("name")}"
                    }
                    val metrics = compare(expected, actual)
                    val metric = "${query.getString("name")}: cosine=${metrics.cosine}, " +
                        "rmse=${metrics.rmse}, maxAbs=${metrics.maxAbs}"
                    Log.i(TAG, metric)
                    println("PASR_METRIC $TAG $metric")
                    assertTrue(
                        "${query.getString("name")} cosine ${metrics.cosine}",
                        metrics.cosine >= MIN_COSINE,
                    )
                    assertTrue(
                        "${query.getString("name")} RMSE ${metrics.rmse}",
                        metrics.rmse <= MAX_RMSE,
                    )
                }
            } finally {
                inference.close()
            }
        }
    }

    private fun readFloatVector(file: File): FloatArray {
        val bytes = file.readBytes()
        require(bytes.size == EMBEDDING_DIM * Float.SIZE_BYTES)
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        return FloatArray(EMBEDDING_DIM) { buffer.float }
    }

    private fun compare(expected: FloatArray, actual: FloatArray): Metrics {
        require(expected.size == actual.size)
        var dot = 0.0
        var expectedNorm = 0.0
        var actualNorm = 0.0
        var squareError = 0.0
        var maxAbs = 0.0
        expected.indices.forEach { index ->
            val left = expected[index].toDouble()
            val right = actual[index].toDouble()
            require(left.isFinite() && right.isFinite())
            dot += left * right
            expectedNorm += left * left
            actualNorm += right * right
            val difference = kotlin.math.abs(left - right)
            squareError += difference * difference
            maxAbs = maxOf(maxAbs, difference)
        }
        return Metrics(
            cosine = dot / sqrt(expectedNorm * actualNorm),
            rmse = sqrt(squareError / expected.size),
            maxAbs = maxAbs,
        )
    }

    private fun sha256(file: File): String {
        require(file.isFile) { "Missing ${file.absolutePath}" }
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().buffered().use { stream ->
            val buffer = ByteArray(64 * 1024)
            while (true) {
                val count = stream.read(buffer)
                if (count < 0) break
                digest.update(buffer, 0, count)
            }
        }
        return digest.digest().joinToString("") { byte ->
            (byte.toInt() and 0xff).toString(16).padStart(2, '0')
        }
    }

    private data class Metrics(
        val cosine: Double,
        val rmse: Double,
        val maxAbs: Double,
    )

    private companion object {
        const val TAG = "TextEmbeddingParity"
        const val EMBEDDING_DIM = 768
        const val MIN_COSINE = 0.999
        const val MAX_RMSE = 0.002
    }
}
