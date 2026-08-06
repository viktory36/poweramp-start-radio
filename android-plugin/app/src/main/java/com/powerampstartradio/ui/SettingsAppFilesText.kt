package com.powerampstartradio.ui

internal data class SettingsAppFilesSummary(
    val capabilityReadiness: List<String>,
)

internal object SettingsAppFilesText {
    fun summarize(files: List<AppFileStatus>): SettingsAppFilesSummary {
        if (files.isEmpty()) {
            return SettingsAppFilesSummary(listOf("File status is not available yet."))
        }
        val statusByName = files.associateBy(AppFileStatus::name)
        fun missing(requirements: List<FileRequirement>): List<String> = requirements.mapNotNull {
            requirement ->
            requirement.need.takeUnless { statusByName[requirement.name]?.present == true }
        }

        val radioMissing = missing(RADIO_REQUIREMENTS)
        val findMusicMissing = missing(FIND_MUSIC_REQUIREMENTS)
        val primary = if (radioMissing == findMusicMissing) {
            listOf(capabilityStatus("Radio and Find Music", radioMissing))
        } else {
            listOf(
                capabilityStatus("Radio", radioMissing),
                capabilityStatus("Find Music", findMusicMissing),
            )
        }
        return SettingsAppFilesSummary(
            capabilityReadiness = primary + listOf(
                capabilityStatus("Graph Explorer", missing(GRAPH_REQUIREMENTS)),
                capabilityStatus("On-device indexing", missing(INDEXING_REQUIREMENTS)),
            ),
        )
    }

    private fun capabilityStatus(capability: String, missing: List<String>): String =
        if (missing.isEmpty()) {
            "$capability: ready."
        } else {
            "$capability: needs ${joinNeeds(missing)}."
        }

    private fun joinNeeds(needs: List<String>): String = when (needs.size) {
        1 -> needs.single()
        2 -> needs.joinToString(" and ")
        else -> needs.dropLast(1).joinToString(", ") + ", and ${needs.last()}"
    }

    private data class FileRequirement(val name: String, val need: String)

    private val DATABASE = FileRequirement("database", "a music index")
    private val SEARCH_INDEX = FileRequirement("search index", "a matching search index")
    private val GRAPH = FileRequirement("similarity graph", "a rebuilt similarity graph")
    private val MERT = FileRequirement("mert.tflite", "the MERT audio model")
    private val CLAMP_AUDIO = FileRequirement("clamp3_audio.tflite", "the CLaMP3 audio encoder")
    private val CLAMP_TEXT = FileRequirement("clamp3_text.tflite", "the CLaMP3 text encoder")
    private val TOKENIZER = FileRequirement("sentencepiece.bpe.model", "the CLaMP3 tokenizer")

    private val RADIO_REQUIREMENTS = listOf(DATABASE, SEARCH_INDEX)
    private val FIND_MUSIC_REQUIREMENTS = listOf(DATABASE, SEARCH_INDEX, CLAMP_TEXT, TOKENIZER)
    private val GRAPH_REQUIREMENTS = listOf(DATABASE, SEARCH_INDEX, GRAPH)
    private val INDEXING_REQUIREMENTS = listOf(
        DATABASE,
        MERT,
        CLAMP_AUDIO,
        CLAMP_TEXT,
        TOKENIZER,
    )
}
