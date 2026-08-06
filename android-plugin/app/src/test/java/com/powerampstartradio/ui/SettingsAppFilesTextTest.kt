package com.powerampstartradio.ui

import org.junit.Assert.assertEquals
import org.junit.Test

class SettingsAppFilesTextTest {
    @Test
    fun `routine summary names readiness by capability`() {
        val summary = SettingsAppFilesText.summarize(
            listOf(
                AppFileStatus("database", present = true),
                AppFileStatus("search index", present = true),
                AppFileStatus("mert.tflite", present = true),
                AppFileStatus("clamp3_audio.tflite", present = true),
                AppFileStatus("clamp3_text.tflite", present = true),
                AppFileStatus("sentencepiece.bpe.model", present = true),
                AppFileStatus("similarity graph", present = false),
            ),
        )

        assertEquals(
            listOf(
                "Radio and Find Music: ready.",
                "Graph Explorer: needs a rebuilt similarity graph.",
                "On-device indexing: ready.",
            ),
            summary.capabilityReadiness,
        )
    }

    @Test
    fun `routine summary attributes missing files to only affected capabilities`() {
        val summary = SettingsAppFilesText.summarize(
            listOf(
                AppFileStatus("database", present = true),
                AppFileStatus("search index", present = true),
                AppFileStatus("mert.tflite", present = true),
                AppFileStatus("clamp3_audio.tflite", present = false),
                AppFileStatus("clamp3_text.tflite", present = false),
                AppFileStatus("sentencepiece.bpe.model", present = false),
                AppFileStatus("similarity graph", present = true),
            ),
        )

        assertEquals(
            listOf(
                "Radio: ready.",
                "Find Music: needs the CLaMP3 text encoder and the CLaMP3 tokenizer.",
                "Graph Explorer: ready.",
                "On-device indexing: needs the CLaMP3 audio encoder, the CLaMP3 text encoder, " +
                    "and the CLaMP3 tokenizer.",
            ),
            summary.capabilityReadiness,
        )
    }

    @Test
    fun `routine summary combines complete search and graph readiness`() {
        val summary = SettingsAppFilesText.summarize(
            listOf(
                AppFileStatus("database", present = true),
                AppFileStatus("search index", present = true),
                AppFileStatus("mert.tflite", present = true),
                AppFileStatus("clamp3_audio.tflite", present = true),
                AppFileStatus("clamp3_text.tflite", present = true),
                AppFileStatus("sentencepiece.bpe.model", present = true),
                AppFileStatus("similarity graph", present = true),
            ),
        )

        assertEquals(
            listOf(
                "Radio and Find Music: ready.",
                "Graph Explorer: ready.",
                "On-device indexing: ready.",
            ),
            summary.capabilityReadiness,
        )
    }
}
