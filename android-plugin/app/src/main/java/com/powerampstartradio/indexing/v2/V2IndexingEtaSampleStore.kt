package com.powerampstartradio.indexing.v2

import android.util.AtomicFile
import com.google.gson.Gson
import com.google.gson.GsonBuilder
import java.io.File
import java.io.OutputStreamWriter
import java.nio.charset.StandardCharsets

/** Small cross-job calibration store. Corrupt or future-schema data means calibrating, never lies. */
class V2IndexingEtaSampleStore(
    filesDir: File,
    private val gson: Gson = GsonBuilder().disableHtmlEscaping().create(),
) {
    private val file = AtomicFile(File(filesDir, "indexing_v2/eta-stage-rates-v1.json"))

    @Synchronized
    fun loadOrEmpty(): V2PersistedStageRateSnapshot {
        if (!file.baseFile.isFile) return emptySnapshot()
        return try {
            file.openRead().bufferedReader(StandardCharsets.UTF_8).use { reader ->
                gson.fromJson(reader, V2PersistedStageRateSnapshot::class.java)
            }?.takeIf { snapshot ->
                snapshot.schemaVersion == SCHEMA_VERSION &&
                    snapshot.samples.size <= MAX_PERSISTED_SAMPLES &&
                    snapshot.samples.all { sample ->
                        sample.activeMsPerUnit.isFinite() && sample.activeMsPerUnit > 0.0 &&
                            (sample.codecClass?.length ?: 0) <= MAX_CODEC_CLASS_CHARS
                    }
            } ?: emptySnapshot()
        } catch (_: Exception) {
            emptySnapshot()
        }
    }

    @Synchronized
    fun save(snapshot: V2PersistedStageRateSnapshot) {
        require(snapshot.schemaVersion == SCHEMA_VERSION) { "unsupported ETA sample schema" }
        require(snapshot.samples.size <= MAX_PERSISTED_SAMPLES) { "too many ETA samples" }
        require(file.baseFile.parentFile?.let { it.isDirectory || it.mkdirs() } == true) {
            "cannot create ETA sample directory"
        }
        val stream = file.startWrite()
        try {
            val writer = OutputStreamWriter(stream, StandardCharsets.UTF_8)
            gson.toJson(snapshot, writer)
            writer.flush()
            file.finishWrite(stream)
        } catch (error: Throwable) {
            file.failWrite(stream)
            throw error
        }
    }

    private fun emptySnapshot() = V2PersistedStageRateSnapshot(
        schemaVersion = SCHEMA_VERSION,
        samples = emptyList(),
    )

    private companion object {
        const val SCHEMA_VERSION = 1
        const val MAX_PERSISTED_SAMPLES = 6 * 3 * 32
        const val MAX_CODEC_CLASS_CHARS = 128
    }
}
