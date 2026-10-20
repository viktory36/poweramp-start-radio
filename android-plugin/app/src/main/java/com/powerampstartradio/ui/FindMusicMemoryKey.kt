package com.powerampstartradio.ui

import com.google.gson.JsonArray
import java.security.MessageDigest

/** Musical request identity, independent of execution controls and the growing library. */
internal object FindMusicMemoryKey {
    fun of(query: FindMusicQuerySpec): String {
        val parts = query.activeTextIngredients.map { ingredient ->
            part("text", ingredient.query, ingredient.negative, ingredient.weight, query.operator)
        } + query.songSeeds.filter { it.weight > 0f }.map { anchor ->
            part(
                "recording",
                anchor.stableTrackSpanId ?: "track:${anchor.trackId}",
                anchor.negative,
                anchor.weight,
                query.operator,
            )
        }
        require(parts.isNotEmpty()) { "Search memory needs at least one active ingredient" }
        val canonical = JsonArray().apply {
            add(query.embeddingSpace)
            add(query.operator.wireName)
            add(JsonArray().apply { parts.sorted().forEach { add(it) } })
            if (query.operator == FindMusicOperator.REFINE) {
                add(parts[requireNotNull(query.refineSpec).primaryIngredientIndex])
            }
        }.toString()
        val digest = MessageDigest.getInstance("SHA-256")
            .digest(canonical.toByteArray(Charsets.UTF_8))
        return "find:" + digest.joinToString("") { "%02x".format(it.toInt() and 0xff) }
    }

    private fun part(
        kind: String,
        identity: String,
        negative: Boolean,
        weight: Float,
        operator: FindMusicOperator,
    ): String = JsonArray().apply {
        add(kind)
        add(identity)
        add(negative)
        if (operator == FindMusicOperator.ALL_OF) add(weight.toRawBits())
    }.toString()
}
