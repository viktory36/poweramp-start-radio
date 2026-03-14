package com.powerampstartradio.poweramp

import java.text.Normalizer
import kotlin.math.abs

object TrackNormalization {
    private val TRACK_NUMBER_PREFIX = Regex("^\\d+[.\\-\\s]+")
    private val WHITESPACE = Regex("\\s+")
    private val PARENTHETICAL = Regex("\\s*[\\(\\[].*?[\\)\\]]")

    val audioExtensions = setOf(
        ".mp3", ".flac", ".opus", ".ogg", ".m4a", ".aac", ".wav",
        ".wma", ".ape", ".wv", ".alac", ".aiff", ".aif"
    )

    fun normalizeNfc(value: String): String =
        Normalizer.normalize(value, Normalizer.Form.NFC)

    fun sanitizePipe(value: String): String =
        value.replace('|', '/')

    fun normalizeArtist(value: String?): String {
        val artist = value.orEmpty().lowercase().trim()
        val normalized = if (artist == "unknown artist") "" else artist
        return sanitizePipe(normalizeNfc(normalized))
    }

    fun normalizeAlbum(value: String?): String =
        sanitizePipe(normalizeNfc(value.orEmpty().lowercase().trim()))

    fun normalizeTitle(value: String?): String {
        val title = value.orEmpty().lowercase().trim()
        return sanitizePipe(normalizeNfc(stripAudioExtension(title)))
    }

    fun normalizePath(value: String?): String? =
        value?.takeIf { it.isNotBlank() }?.let(::normalizeNfc)

    fun buildMetadataKey(artist: String, album: String, title: String, durationMs: Int): String {
        val durationRounded = (durationMs / 100) * 100
        return "$artist|$album|$title|$durationRounded"
    }

    fun buildFilenameKeys(artist: String, title: String, existingKey: String? = null): Set<String> {
        val keys = linkedSetOf<String>()
        existingKey?.takeIf { it.isNotBlank() }?.let { keys.add(normalizeAsFilename(it)) }
        normalizeAsFilename(title).takeIf { it.isNotBlank() }?.let { keys.add(it) }
        if (artist.isNotBlank()) {
            normalizeAsFilename("$artist - $title").takeIf { it.isNotBlank() }?.let { keys.add(it) }
        }
        return keys
    }

    fun normalizeAsFilename(value: String): String {
        return normalizeNfc(
            value.lowercase()
                .replace(PARENTHETICAL, "")
                .replace(TRACK_NUMBER_PREFIX, "")
                .replace(WHITESPACE, " ")
                .trim()
        )
    }

    fun stripAudioExtension(title: String): String {
        val idx = title.lastIndexOf('.')
        if (idx > 0) {
            val ext = title.substring(idx)
            if (ext in audioExtensions) return title.substring(0, idx)
        }
        return title
    }

    fun durationCompatible(aMs: Int, bMs: Int, toleranceMs: Int = 5_000): Boolean {
        if (aMs <= 0 || bMs <= 0) return true
        return abs(aMs - bMs) <= toleranceMs
    }

    fun durationPenalty(aMs: Int, bMs: Int): Int {
        if (aMs <= 0 || bMs <= 0) return 0
        return abs(aMs - bMs)
    }
}
