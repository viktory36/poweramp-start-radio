package com.powerampstartradio.indexing

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.google.ai.edge.litert.Accelerator
import com.powerampstartradio.indexing.v2.V2AudioSpanAuthority
import com.powerampstartradio.indexing.v2.V2AudioSpanResolver
import com.powerampstartradio.indexing.v2.V2ExecutionBoundaryRequirement
import com.powerampstartradio.indexing.v2.V2IndexingExecutionPolicies
import com.powerampstartradio.indexing.v2.V2IndexingWorkPolicy
import com.powerampstartradio.indexing.v2.V2MediaExtractorAudioInspector
import com.powerampstartradio.indexing.v2.V2ProviderPathGroupCompleteness
import com.powerampstartradio.indexing.v2.V2ProviderPathGroupEvidence
import com.powerampstartradio.indexing.v2.V2ProviderPathGroupSnapshot
import com.powerampstartradio.indexing.v2.V2ProviderPathRowEvidence
import com.powerampstartradio.indexing.v2.V2ProviderSnapshotAcquisitionEvidence
import com.powerampstartradio.indexing.v2.V2ResolvedAudioSpanKind
import com.powerampstartradio.poweramp.TrackNormalization
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import kotlin.math.abs
import kotlin.math.sqrt
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class AudioEmbeddingParityInstrumentedTest {
    @Test
    fun matchesPinnedDesktopPyTorchForWholeFileFlacsOnStrictGpu() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val fixtureDir = File(context.filesDir, "device_acceptance/audio_parity")
        assumeTrue(
            "Audio parity fixture was not staged; skipping opt-in acceptance gate",
            fixtureDir.isDirectory,
        )

        val manifestFile = fixtureFile(fixtureDir, "manifest.json")
        val manifest = JSONObject(manifestFile.readText(Charsets.UTF_8))
        assertEquals(1, manifest.getInt("schema_version"))
        assertEquals(
            V2IndexingWorkPolicy.PREPROCESSING_SPEC_ID,
            manifest.getString("android_preprocessing_spec_id"),
        )
        assertEquals(
            V2IndexingWorkPolicy.INFERENCE_BACKEND_POLICY_ID,
            manifest.getString("android_inference_backend_policy_id"),
        )
        assertTrue(manifest.getBoolean("full_track"))
        assertEquals("float32", manifest.getString("host_inference_dtype"))
        assertEquals(MertInference.SAMPLE_RATE, manifest.getInt("target_sample_rate_hz"))
        assertEquals(MertInference.WINDOW_SAMPLES, manifest.getInt("mert_window_samples"))
        assertEquals(MertInference.SAMPLE_RATE, manifest.getInt("minimum_tail_samples"))
        assertEquals(Clamp3AudioInference.MAX_WINDOWS, manifest.getInt("clamp_max_frames"))
        assertEquals(EMBEDDING_DIM, manifest.getInt("embedding_dimension"))

        val deviceModels = manifest.getJSONObject("device_models")
        val mertModel = verifiedModel(context.filesDir, deviceModels.getJSONObject("mert"))
        val clampModel = verifiedModel(
            context.filesDir,
            deviceModels.getJSONObject("clamp3_audio"),
        )

        val tracksJson = manifest.getJSONArray("tracks")
        assertTrue("Audio parity manifest must contain tracks", tracksJson.length() > 0)
        val prepared = ArrayList<PreparedTrack>(tracksJson.length())

        val mertLoadStarted = System.nanoTime()
        val mert = MertInference(mertModel, accelerator = Accelerator.GPU)
        try {
            assertEquals("MERT must use the pinned backend", Accelerator.GPU, mert.activeAccelerator)
            recordMetric("TIMING model_load_mert=${elapsedMs(mertLoadStarted)}ms")
            for (index in 0 until tracksJson.length()) {
                val track = tracksJson.getJSONObject(index)
                prepared += prepareTrack(
                    fixtureDir = fixtureDir,
                    track = track,
                    trackIndex = index,
                    mert = mert,
                    cacheRoot = File(context.cacheDir, "audio-parity-v1"),
                )
            }
        } finally {
            mert.close()
        }

        val clampLoadStarted = System.nanoTime()
        val clamp = Clamp3AudioInference(clampModel, accelerator = Accelerator.GPU)
        try {
            assertEquals(
                "CLaMP3 audio must use the pinned backend",
                Accelerator.GPU,
                clamp.activeAccelerator,
            )
            recordMetric("TIMING model_load_clamp3=${elapsedMs(clampLoadStarted)}ms")
            prepared.forEach { preparedTrack ->
                val expectedSegments = preparedTrack.track.getInt("clamp_segments")
                assertEquals(
                    "${preparedTrack.name} planned CLaMP segments",
                    expectedSegments,
                    clamp.segmentCount(preparedTrack.features.size),
                )
                var completedSegments = 0
                var consumedWindows = 0
                val clampStarted = System.nanoTime()
                val actual = requireNotNull(
                    clamp.encodeStreaming(
                        numWindows = preparedTrack.features.size,
                        readNextWindow = { preparedTrack.features[consumedWindows++] },
                        onSegmentDone = { completedSegments++ },
                    ),
                ) { "CLaMP3 inference failed for ${preparedTrack.name}" }
                val clampMs = elapsedMs(clampStarted)
                assertEquals(preparedTrack.features.size, consumedWindows)
                assertEquals(expectedSegments, completedSegments)

                val metrics = compare(preparedTrack.expectedEmbedding, actual)
                recordMetric(
                    "PASS ${preparedTrack.name}: cosine=${metrics.cosine} " +
                        "rmse=${metrics.rmse} maxAbs=${metrics.maxAbs} " +
                        "resolve=${preparedTrack.resolveMs}ms pcm=${preparedTrack.pcmMs}ms " +
                        "mert=${preparedTrack.mertMs}ms clamp=${clampMs}ms " +
                        "windows=${preparedTrack.features.size} segments=$expectedSegments",
                )
                assertTrue(
                    "${preparedTrack.name} cosine ${metrics.cosine} < $MIN_COSINE",
                    metrics.cosine >= MIN_COSINE,
                )
                assertTrue(
                    "${preparedTrack.name} RMSE ${metrics.rmse} > $MAX_RMSE",
                    metrics.rmse <= MAX_RMSE,
                )
                assertTrue(
                    "${preparedTrack.name} max abs ${metrics.maxAbs} > $MAX_ABS",
                    metrics.maxAbs <= MAX_ABS,
                )
            }
        } finally {
            clamp.close()
        }
    }

    private fun prepareTrack(
        fixtureDir: File,
        track: JSONObject,
        trackIndex: Int,
        mert: MertInference,
        cacheRoot: File,
    ): PreparedTrack {
        val name = track.getString("name")
        val source = fixtureFile(fixtureDir, track.getString("source_file"))
        assertEquals("$name source SHA-256", track.getString("source_sha256"), sha256(source))
        assertEquals("$name source byte length", track.getLong("source_size_bytes"), source.length())
        val expectedFile = fixtureFile(fixtureDir, track.getString("embedding_file"))
        assertEquals(
            "$name expected-vector SHA-256",
            track.getString("embedding_sha256"),
            sha256(expectedFile),
        )
        val expectedEmbedding = readFloatVector(expectedFile)

        val durationMs = track.getLong("provider_duration_ms")
        require(durationMs in 0..Int.MAX_VALUE.toLong()) { "$name duration does not fit Int" }
        val normalizedTitle = TrackNormalization.normalizeTitle(name)
        val selected = NewTrackDetector.UnindexedTrack(
            powerampFileId = trackIndex.toLong() + 1L,
            artist = "",
            album = "",
            title = normalizedTitle,
            durationMs = durationMs.toInt(),
            path = source.absolutePath,
        )
        val row = V2ProviderPathRowEvidence(
            powerampFileId = selected.powerampFileId,
            physicalPath = source.absolutePath,
            providerPhysicalPath = source.absolutePath,
            artist = "",
            album = "",
            title = normalizedTitle,
            offsetMs = 0L,
            durationMs = durationMs,
            cueSourceImageFolderId = null,
        )
        val snapshot = V2ProviderPathGroupSnapshot(
            libraryGeneration = "audio-parity-$name",
            groups = listOf(
                V2ProviderPathGroupEvidence(
                    physicalPath = source.absolutePath,
                    rows = listOf(row),
                    completeness = V2ProviderPathGroupCompleteness.COMPLETE,
                ),
            ),
            acquisitionEvidence = V2ProviderSnapshotAcquisitionEvidence(
                queryUri = "fixture://audio-parity/$name",
                requestedColumns = listOf("fixture"),
                returnedColumns = listOf("fixture"),
                rowCount = 1,
                cursorExhaustedNormally = true,
            ),
        )

        val resolveStarted = System.nanoTime()
        val span = V2AudioSpanResolver(V2MediaExtractorAudioInspector())
            .resolve(listOf(selected), snapshot)
            .resolved
            .single()
        val resolveMs = elapsedMs(resolveStarted)
        assertEquals("$name span kind", V2ResolvedAudioSpanKind.WHOLE_FILE, span.kind)
        assertEquals(
            "$name span authority",
            V2AudioSpanAuthority.PROVISIONAL_END_OF_STREAM,
            span.authority,
        )
        assertEquals(
            "$name execution boundary",
            V2ExecutionBoundaryRequirement.VERIFY_END_OF_STREAM_AND_RECONCILE,
            span.executionBoundaryRequirement,
        )
        assertTrue("$name must verify physical EOS", span.mustVerifyEndOfStream)
        assertEquals(
            "$name source sample rate",
            track.getInt("source_sample_rate_hz"),
            span.containerEvidence.sampleRateHz,
        )
        assertEquals(
            "$name source channel count",
            track.getInt("source_channel_count"),
            span.containerEvidence.channelCount,
        )
        assertEquals(
            "$name source sample count",
            track.getLong("source_sample_count"),
            span.sourceSampleCount,
        )
        assertEquals(
            "$name planned 24 kHz samples",
            track.getLong("canonical_sample_count_24k"),
            span.exactSampleCount24k,
        )
        assertEquals(
            "$name planned MERT windows",
            track.getInt("mert_windows"),
            span.expectedWork.mertWindows,
        )
        assertEquals(
            "$name planned CLaMP segments",
            track.getInt("clamp_segments"),
            span.expectedWork.clampSegments,
        )

        cacheRoot.mkdirs()
        val pcmFile = File(cacheRoot, "$name.pcm.f32le")
        pcmFile.delete()
        val pcmStarted = System.nanoTime()
        val pcm = try {
            TrackPcmCache().build(
                sourceFile = source,
                logicalStartUs = span.startUs,
                logicalDurationUs = span.plannedDurationUs,
                chunkDurationMs = V2IndexingExecutionPolicies.BYTE_STABLE_PCM_CHUNK_DURATION_MS,
                outputFile = pcmFile,
                boundaryMode = TrackPcmCache.BoundaryMode.REQUIRE_PHYSICAL_END_OF_STREAM,
                resamplerPolicy = TrackPcmCache.ResamplerPolicy.TORCHAUDIO_HANN_V1,
            )
        } catch (error: Throwable) {
            pcmFile.delete()
            throw error
        }
        val pcmMs = elapsedMs(pcmStarted)
        try {
            assertTrue("$name decoder must reach physical EOS", pcm.endOfStreamReached)
            assertTrue("$name must not claim a CUE boundary", !pcm.logicalBoundaryEnforced)
            assertEquals("$name PCM source rate", span.containerEvidence.sampleRateHz, pcm.sourceSampleRate)
            assertEquals("$name PCM source start", span.startSourceSample, pcm.sourceStartSample)
            assertEquals(
                "$name PCM source end",
                span.endSourceSampleExclusive,
                pcm.sourceEndSampleExclusive,
            )
            assertEquals("$name PCM source samples", span.sourceSampleCount, pcm.sourceSampleCount)
            assertEquals(
                "$name PCM 24 kHz samples",
                span.exactSampleCount24k,
                pcm.exactSampleCount24k,
            )
            assertEquals(
                "$name resampler policy",
                NativeMath.TORCHAUDIO_HANN_V1_SPEC_ID,
                pcm.preprocessingSpecId,
            )

            val features = ArrayList<FloatArray>(span.expectedWork.mertWindows)
            var completedWindows = 0
            val mertStarted = System.nanoTime()
            val extracted = mert.extractFeaturesFromPcmFile(
                pcmFile = pcm.file,
                normalization = pcm.normalization,
                onFeatureExtracted = features::add,
                onWindowDone = { completedWindows++ },
            )
            val mertMs = elapsedMs(mertStarted)
            assertEquals("$name extracted windows", span.expectedWork.mertWindows, extracted)
            assertEquals("$name completed windows", span.expectedWork.mertWindows, completedWindows)
            assertEquals("$name retained features", span.expectedWork.mertWindows, features.size)
            features.forEachIndexed { index, feature ->
                assertEquals("$name MERT feature $index dimension", EMBEDDING_DIM, feature.size)
                assertTrue("$name MERT feature $index must be finite", feature.all(Float::isFinite))
            }
            recordMetric(
                "STAGE $name: resolve=${resolveMs}ms pcm=${pcmMs}ms " +
                    "decode=${pcm.decodeMs}ms resample=${pcm.resampleMs}ms " +
                    "mert=${mertMs}ms samples24k=${pcm.exactSampleCount24k} " +
                    "windows=${features.size}",
            )
            return PreparedTrack(
                name = name,
                track = track,
                expectedEmbedding = expectedEmbedding,
                features = features,
                resolveMs = resolveMs,
                pcmMs = pcmMs,
                mertMs = mertMs,
            )
        } finally {
            assertTrue("$name temporary PCM cleanup failed", !pcmFile.exists() || pcmFile.delete())
        }
    }

    private fun verifiedModel(filesDir: File, model: JSONObject): File {
        val fileName = model.getString("file")
        require('/' !in fileName && '\\' !in fileName) { "Model filename is not local: $fileName" }
        val file = File(filesDir, fileName)
        assertEquals("$fileName SHA-256", model.getString("sha256"), sha256(file))
        return file
    }

    private fun fixtureFile(root: File, relative: String): File {
        require(relative.isNotBlank()) { "Fixture path is blank" }
        val canonicalRoot = root.canonicalFile
        val file = File(canonicalRoot, relative).canonicalFile
        require(file.path.startsWith(canonicalRoot.path + File.separator)) {
            "Fixture path escaped its root: $relative"
        }
        require(file.isFile) { "Missing staged fixture ${file.absolutePath}" }
        return file
    }

    private fun readFloatVector(file: File): FloatArray {
        val bytes = file.readBytes()
        require(bytes.size == EMBEDDING_DIM * Float.SIZE_BYTES) {
            "Expected ${EMBEDDING_DIM * Float.SIZE_BYTES} bytes in ${file.name}, got ${bytes.size}"
        }
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        return FloatArray(EMBEDDING_DIM) { buffer.float }
    }

    private fun compare(expected: FloatArray, actual: FloatArray): Metrics {
        require(expected.size == EMBEDDING_DIM && actual.size == EMBEDDING_DIM)
        var dot = 0.0
        var expectedNorm = 0.0
        var actualNorm = 0.0
        var squaredError = 0.0
        var maxAbs = 0.0
        expected.indices.forEach { index ->
            val left = expected[index].toDouble()
            val right = actual[index].toDouble()
            require(left.isFinite() && right.isFinite())
            dot += left * right
            expectedNorm += left * left
            actualNorm += right * right
            val difference = abs(left - right)
            squaredError += difference * difference
            maxAbs = maxOf(maxAbs, difference)
        }
        require(expectedNorm > 0.0 && actualNorm > 0.0)
        return Metrics(
            cosine = dot / sqrt(expectedNorm * actualNorm),
            rmse = sqrt(squaredError / EMBEDDING_DIM),
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

    private fun elapsedMs(startedNs: Long): Long = (System.nanoTime() - startedNs) / 1_000_000L

    private fun recordMetric(message: String) {
        Log.i(TAG, message)
        println("PASR_METRIC $TAG $message")
    }

    private data class PreparedTrack(
        val name: String,
        val track: JSONObject,
        val expectedEmbedding: FloatArray,
        val features: List<FloatArray>,
        val resolveMs: Long,
        val pcmMs: Long,
        val mertMs: Long,
    )

    private data class Metrics(
        val cosine: Double,
        val rmse: Double,
        val maxAbs: Double,
    )

    private companion object {
        const val TAG = "AudioEmbeddingParity"
        const val EMBEDDING_DIM = 768

        // Host LiteRT CPU conversion audit over these fixtures measured cosine 0.9942-0.9990,
        // RMSE 0.0016-0.0039, and max abs 0.0050-0.0122 versus production PyTorch.
        // The gate leaves bounded backend/decoder headroom without accepting semantic collapse.
        const val MIN_COSINE = 0.990
        const val MAX_RMSE = 0.006
        const val MAX_ABS = 0.025
    }
}
