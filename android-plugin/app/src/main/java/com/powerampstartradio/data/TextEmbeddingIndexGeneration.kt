package com.powerampstartradio.data

import java.io.File
import java.util.concurrent.atomic.AtomicLong

/** Process-local revision for the atomically published CLaMP3 embedding index. */
internal object TextEmbeddingIndexGeneration {
    private val revision = AtomicLong(0L)

    fun current(): Long = revision.get()

    fun invalidate(): Long = revision.incrementAndGet()
}

internal data class TextEmbeddingIndexSnapshot(
    val generation: Long,
    val trackCount: Int,
    val canonicalFilePath: String,
    val fileLength: Long,
    val fileModified: Long,
) {
    fun matches(currentGeneration: Long, databaseCount: Int, file: File): Boolean {
        return generation == currentGeneration &&
            trackCount == databaseCount &&
            file.exists() &&
            canonicalFilePath == file.canonicalPath &&
            fileLength == file.length() &&
            fileModified == file.lastModified()
    }

    companion object {
        fun capture(generation: Long, trackCount: Int, file: File): TextEmbeddingIndexSnapshot {
            return TextEmbeddingIndexSnapshot(
                generation = generation,
                trackCount = trackCount,
                canonicalFilePath = file.canonicalPath,
                fileLength = file.length(),
                fileModified = file.lastModified(),
            )
        }
    }
}
