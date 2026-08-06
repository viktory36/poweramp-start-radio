package com.powerampstartradio.ui

import com.google.gson.JsonArray
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import com.powerampstartradio.data.StableIdentityGenerationBinding

/** The deterministic relationship between Find Music ingredients. */
enum class FindMusicOperator(val wireName: String) {
    ALL_OF("all_of"),
    REFINE("refine");

    companion object {
        fun fromWireName(value: String?): FindMusicOperator =
            entries.firstOrNull { it.wireName == value } ?: ALL_OF
    }
}

/** Exact primary-neighborhood widths supported by the asymmetric Refine objective. */
enum class FindMusicRefineNeighborhood(
    val wireName: String,
    val basisPoints: Int,
) {
    TOP_0_25_PERCENT("top_0_25_percent", 25),
    TOP_0_5_PERCENT("top_0_5_percent", 50),
    TOP_1_PERCENT("top_1_percent", 100),
    TOP_2_PERCENT("top_2_percent", 200);

    val fraction: Double
        get() = basisPoints.toDouble() / BASIS_POINT_DENOMINATOR

    /** Exact ceil(domain size * fraction), without silently widening to a requested result count. */
    fun candidateCount(availableIdentityCount: Int): Int {
        require(availableIdentityCount >= 0)
        if (availableIdentityCount == 0) return 0
        return (
            (availableIdentityCount.toLong() * basisPoints + BASIS_POINT_DENOMINATOR - 1L) /
                BASIS_POINT_DENOMINATOR
            ).toInt()
    }

    companion object {
        val DEFAULT = TOP_1_PERCENT
        private const val BASIS_POINT_DENOMINATOR = 10_000L

        fun fromWireName(value: String?): FindMusicRefineNeighborhood? =
            entries.firstOrNull { it.wireName == value }
    }
}

/** Complete asymmetric Refine contract over the persisted active-ingredient order. */
data class FindMusicRefineSpec(
    val primaryIngredientIndex: Int,
    val neighborhood: FindMusicRefineNeighborhood = FindMusicRefineNeighborhood.DEFAULT,
)

/** Membership planner for a supported Find Music objective. */
enum class FindMusicTextResultPlanner(
    val wireName: String,
    val currentVersion: Int,
) {
    CLOSEST("closest", 1),
    VARIED_DPP("varied_dpp", 2),
    VARIED_ALL_OF_DPP("varied_all_of_dpp", 2),
}

/** One exact natural-language anchor in the shared CLaMP3 retrieval space. */
data class FindMusicTextIngredient(
    val query: String,
    val weight: Float,
    val negative: Boolean,
)

/** One exact song anchor and its contribution to a composed query. */
data class FindMusicSongAnchor(
    val trackId: Long,
    val stableTrackSpanId: String? = null,
    val artist: String?,
    val title: String?,
    val weight: Float,
    val negative: Boolean,
) {
    val displayLabel: String
        get() = recordingDisplayLabel(artist, title, "Track $trackId")
}

/**
 * Complete, versioned input to one Find Music ranking.
 *
 * Editor state that changes the actual request (operator, signs, weights and result count)
 * belongs here so replay never consults whatever controls happen to be selected later.
 */
data class FindMusicQuerySpec(
    val schemaVersion: Int = CURRENT_SCHEMA_VERSION,
    val embeddingSpace: String = EMBEDDING_SPACE_CLAMP3,
    val rankingVersion: Int = CURRENT_RANKING_VERSION,
    val operator: FindMusicOperator = FindMusicOperator.ALL_OF,
    val textIngredients: List<FindMusicTextIngredient> = emptyList(),
    val songSeeds: List<FindMusicSongAnchor> = emptyList(),
    val resultLimit: Int = DEFAULT_RESULT_LIMIT,
    val textResultPlanner: FindMusicTextResultPlanner = FindMusicTextResultPlanner.CLOSEST,
    val refineSpec: FindMusicRefineSpec? = null,
    /** Legacy preset field retained so existing saved queries and requests remain readable. */
    val libraryAddedRange: LibraryAddedRange = LibraryAddedRange.ALL_DATES,
    /** Exact rolling day count for new queries; null falls back to the legacy preset field. */
    val libraryAddedDays: Int? = null,
    val libraryBinding: StableIdentityGenerationBinding? = null,
) {
    val activeTextIngredients: List<FindMusicTextIngredient>
        get() = textIngredients.filter { it.query.isNotBlank() && it.weight > 0f }

    val activeIngredientCount: Int
        get() = activeTextIngredients.size + songSeeds.count { it.weight > 0f }

    val hasActivePositiveIngredient: Boolean
        get() = activeTextIngredients.any { !it.negative } ||
            songSeeds.any { it.weight > 0f && !it.negative }

    val activeEvidenceLabels: List<String>
        get() = activeTextIngredients.map { text ->
            if (text.negative) "Less like: ${text.query}" else text.query
        } + songSeeds.filter { it.weight > 0f }.map { song ->
            if (song.negative) "Less like: ${song.displayLabel}" else song.displayLabel
        }

    /** The only query shape that bypasses composition and ranks by raw cosine. */
    val isSimplePositiveTextOnly: Boolean
        get() = activeTextIngredients.singleOrNull()?.negative == false &&
            songSeeds.none { it.weight > 0f }

    val displayLabel: String
        get() = formatSearchLabel(
            ingredients = activeTextIngredients.map { text ->
                SearchLabelPart(text = text.query, negative = text.negative)
            } + songSeeds.filter { it.weight > 0f }.map { seed ->
                SearchLabelPart(
                    text = seed.title ?: seed.artist ?: "Track ${seed.trackId}",
                    negative = seed.negative,
                )
            },
        )

    /** Stable identity for deduping exact requests, independent of display formatting. */
    val stateKey: String
        get() = FindMusicQuerySpecCodec.toJson(this)

    companion object {
        const val CURRENT_SCHEMA_VERSION = 7
        const val LEGACY_RANKING_VERSION = 1
        /** V4 replaces independent branch interleaving with exact asymmetric Refine. */
        const val CURRENT_RANKING_VERSION = 4
        const val EMBEDDING_SPACE_CLAMP3 = "clamp3_shared_768_v1"
        const val DEFAULT_RESULT_LIMIT = 30
        const val MAX_RESULT_LIMIT = 1_000
        const val MAX_TEXT_INGREDIENTS = 8
    }
}

/** Canonical added-date semantics shared by new exact values and legacy preset queries. */
val FindMusicQuerySpec.effectiveLibraryAddedDays: Int?
    get() = libraryAddedDays ?: libraryAddedRange.dayCount?.toInt()

/** Applies the user's current execution controls without changing the saved musical request. */
internal fun FindMusicQuerySpec.withCurrentExecutionControls(
    resultLimit: Int,
    libraryAddedDays: Int?,
    requestedPlanner: FindMusicTextResultPlanner,
    refineNeighborhood: FindMusicRefineNeighborhood,
): FindMusicQuerySpec {
    require(resultLimit in 1..FindMusicQuerySpec.MAX_RESULT_LIMIT)
    require(libraryAddedDays == null || libraryAddedDays in 1..MAX_LIBRARY_ADDED_DAYS)
    val compatiblePlanner = when {
        isSimplePositiveTextOnly &&
            requestedPlanner == FindMusicTextResultPlanner.VARIED_DPP ->
            FindMusicTextResultPlanner.VARIED_DPP
        operator == FindMusicOperator.ALL_OF && activeIngredientCount >= 2 &&
            requestedPlanner == FindMusicTextResultPlanner.VARIED_ALL_OF_DPP ->
            FindMusicTextResultPlanner.VARIED_ALL_OF_DPP
        else -> FindMusicTextResultPlanner.CLOSEST
    }
    return copy(
        resultLimit = resultLimit,
        textResultPlanner = compatiblePlanner,
        refineSpec = refineSpec?.let { refine ->
            if (operator == FindMusicOperator.REFINE) {
                refine.copy(neighborhood = refineNeighborhood)
            } else {
                refine
            }
        },
        libraryAddedRange = LibraryAddedRange.ALL_DATES,
        libraryAddedDays = libraryAddedDays,
    )
}

internal fun FindMusicQuerySpec.hasSameExecutionControlsAs(
    other: FindMusicQuerySpec,
): Boolean =
    resultLimit == other.resultLimit &&
        effectiveLibraryAddedDays == other.effectiveLibraryAddedDays &&
        textResultPlanner == other.textResultPlanner &&
        refineSpec?.neighborhood == other.refineSpec?.neighborhood

/** Structured persistence with exact migration from the prior rank-v3 singleton-text schema. */
object FindMusicQuerySpecCodec {
    fun toJsonArray(list: List<FindMusicQuerySpec>): String = JsonArray().apply {
        list.forEach { add(toJsonObject(it)) }
    }.toString()

    fun toJson(spec: FindMusicQuerySpec): String = toJsonObject(spec).toString()

    fun fromJsonArray(json: String): List<FindMusicQuerySpec> {
        val array = JsonParser.parseString(json).asJsonArray
        return array.map { fromJsonObject(it.asJsonObject) }
    }

    private fun toJsonObject(spec: FindMusicQuerySpec): JsonObject = JsonObject().apply {
        addProperty("schema_version", spec.schemaVersion)
        addProperty("embedding_space", spec.embeddingSpace)
        addProperty("ranking_version", spec.rankingVersion)
        addProperty("operator", spec.operator.wireName)
        addProperty("result_limit", spec.resultLimit)
        addProperty("text_result_planner", spec.textResultPlanner.wireName)
        addProperty("text_result_planner_version", spec.textResultPlanner.currentVersion)
        spec.refineSpec?.let { refine ->
            add("refine", JsonObject().apply {
                addProperty("primary_ingredient_index", refine.primaryIngredientIndex)
                addProperty("neighborhood", refine.neighborhood.wireName)
            })
        }
        addProperty("library_added_range", spec.libraryAddedRange.name)
        spec.libraryAddedDays?.let { addProperty("library_added_days", it) }
        spec.libraryBinding?.let { binding ->
            add("library_binding", JsonObject().apply {
                addProperty("binding_spec_id", binding.bindingSpecId)
                addProperty("generation_id", binding.generationId)
                addProperty("activation_binding_id", binding.activationBindingId)
                addProperty("database_content_sha256", binding.databaseContentSha256)
                addProperty("ordered_track_set_sha256", binding.orderedTrackSetSha256)
            })
        }
        add("text_ingredients", JsonArray().apply {
            spec.textIngredients.forEach { text ->
                add(JsonObject().apply {
                    addProperty("query", text.query)
                    addProperty("weight", text.weight)
                    addProperty("negative", text.negative)
                })
            }
        })
        add("song_anchors", JsonArray().apply {
            spec.songSeeds.forEach { seed ->
                add(JsonObject().apply {
                    addProperty("track_id", seed.trackId)
                    seed.stableTrackSpanId?.let { addProperty("stable_track_span_id", it) }
                    seed.artist?.let { addProperty("artist", it) }
                    seed.title?.let { addProperty("title", it) }
                    addProperty("weight", seed.weight)
                    addProperty("negative", seed.negative)
                })
            }
        })
    }

    private fun fromJsonObject(obj: JsonObject): FindMusicQuerySpec {
        val sourceSchemaVersion = obj.int("schema_version") ?: 1
        val libraryAddedDays = if (obj.has("library_added_days")) {
            requireNotNull(obj.int("library_added_days")) {
                "Poweramp added-date day count is missing"
            }.also { days ->
                require(days in 1..MAX_LIBRARY_ADDED_DAYS) {
                    "Poweramp added-date day count must be 1..$MAX_LIBRARY_ADDED_DAYS"
                }
            }
        } else {
            null
        }
        val rankingVersion = obj.int("ranking_version")
            ?: FindMusicQuerySpec.LEGACY_RANKING_VERSION
        val textElement = obj.get("text")
        val textObject = textElement?.takeIf { it.isJsonObject }?.asJsonObject
        val textQuery = textObject?.string("query")
            ?: textElement?.takeIf { it.isJsonPrimitive }?.asString
            ?: ""
        val textWeight = textObject?.float("weight")
            ?: obj.float("text_weight")
            ?: if (textQuery.isBlank()) 0f else 1f
        val textNegative = textObject?.boolean("negative")
            ?: obj.boolean("text_negative")
            ?: false
        val textIngredients = when {
            obj.has("text_ingredients") -> obj.getAsJsonArray("text_ingredients").map { element ->
                val text = element.asJsonObject
                FindMusicTextIngredient(
                    query = text.string("query") ?: "",
                    weight = text.float("weight") ?: 1f,
                    negative = text.boolean("negative") ?: false,
                )
            }
            textElement != null || obj.has("text_weight") || obj.has("text_negative") -> listOf(
                FindMusicTextIngredient(
                    query = textQuery,
                    weight = textWeight,
                    negative = textNegative,
                ),
            )
            else -> emptyList()
        }

        val anchors = when {
            obj.has("song_anchors") -> parseAnchors(obj.getAsJsonArray("song_anchors"), current = true)
            obj.has("seeds") -> parseAnchors(obj.getAsJsonArray("seeds"), current = false)
            else -> emptyList()
        }
        val libraryBinding = obj.get("library_binding")
            ?.takeUnless { it.isJsonNull }
            ?.asJsonObject
            ?.let { binding ->
                StableIdentityGenerationBinding(
                    bindingSpecId = binding.requiredString("binding_spec_id"),
                    generationId = binding.requiredString("generation_id"),
                    activationBindingId = binding.requiredString("activation_binding_id"),
                    databaseContentSha256 = binding.requiredString("database_content_sha256"),
                    orderedTrackSetSha256 = binding.requiredString("ordered_track_set_sha256"),
                )
            }
        val plannerWireName = obj.string("text_result_planner")
        val plannerVersion = obj.int("text_result_planner_version")
        val textResultPlanner = if (plannerWireName == null && plannerVersion == null) {
            FindMusicTextResultPlanner.CLOSEST
        } else {
            require(plannerWireName != null && plannerVersion != null) {
                "Find Music text planner identity is incomplete"
            }
            val planner = FindMusicTextResultPlanner.entries.firstOrNull {
                it.wireName == plannerWireName
            } ?: throw IllegalArgumentException(
                "Unknown Find Music text planner: $plannerWireName",
            )
            require(plannerVersion == planner.currentVersion) {
                "Unsupported $plannerWireName planner version $plannerVersion"
            }
            planner
        }
        val refineSpec = obj.get("refine")
            ?.takeUnless { it.isJsonNull }
            ?.asJsonObject
            ?.let { refine ->
                val neighborhoodWireName = refine.requiredString("neighborhood")
                FindMusicRefineSpec(
                    primaryIngredientIndex = requireNotNull(
                        refine.int("primary_ingredient_index"),
                    ) { "Missing Find Music refine primary ingredient index" },
                    neighborhood = FindMusicRefineNeighborhood.fromWireName(
                        neighborhoodWireName,
                    ) ?: throw IllegalArgumentException(
                        "Unknown Find Music refine neighborhood: $neighborhoodWireName",
                    ),
                )
            }

        return FindMusicQuerySpec(
            schemaVersion = if (
                sourceSchemaVersion in 2..6 &&
                rankingVersion == FindMusicQuerySpec.CURRENT_RANKING_VERSION
            ) {
                FindMusicQuerySpec.CURRENT_SCHEMA_VERSION
            } else {
                sourceSchemaVersion
            },
            embeddingSpace = obj.string("embedding_space")
                ?: FindMusicQuerySpec.EMBEDDING_SPACE_CLAMP3,
            rankingVersion = rankingVersion,
            operator = obj.string("operator")?.let { wireName ->
                FindMusicOperator.entries.firstOrNull { it.wireName == wireName }
                    ?: throw IllegalArgumentException("Unknown Find Music operator: $wireName")
            } ?: FindMusicOperator.ALL_OF,
            textIngredients = textIngredients,
            songSeeds = anchors,
            resultLimit = obj.int("result_limit") ?: FindMusicQuerySpec.DEFAULT_RESULT_LIMIT,
            textResultPlanner = textResultPlanner,
            refineSpec = refineSpec,
            libraryAddedRange = obj.string("library_added_range")?.let { stored ->
                LibraryAddedRange.entries.firstOrNull { it.name == stored }
                    ?: throw IllegalArgumentException(
                        "Unknown Poweramp added-date range: $stored",
                    )
            } ?: LibraryAddedRange.ALL_DATES,
            libraryAddedDays = libraryAddedDays,
            libraryBinding = libraryBinding,
        )
    }

    private fun parseAnchors(array: JsonArray, current: Boolean): List<FindMusicSongAnchor> =
        array.map { element ->
            val obj = element.asJsonObject
            FindMusicSongAnchor(
                trackId = if (current) obj.get("track_id").asLong else obj.get("id").asLong,
                stableTrackSpanId = obj.string("stable_track_span_id"),
                artist = obj.string("artist"),
                title = obj.string("title"),
                weight = obj.float("weight") ?: 1f,
                negative = obj.boolean("negative") ?: false,
            )
        }

    private fun JsonObject.string(name: String): String? =
        get(name)?.takeUnless { it.isJsonNull }?.asString

    private fun JsonObject.float(name: String): Float? =
        get(name)?.takeUnless { it.isJsonNull }?.asFloat

    private fun JsonObject.int(name: String): Int? =
        get(name)?.takeUnless { it.isJsonNull }?.asInt

    private fun JsonObject.boolean(name: String): Boolean? =
        get(name)?.takeUnless { it.isJsonNull }?.asBoolean

    private fun JsonObject.requiredString(name: String): String =
        string(name)?.takeIf { it.isNotBlank() }
            ?: throw IllegalArgumentException("Missing Find Music $name")
}

// Compatibility names for the existing UI while storage and replay move to QuerySpec.
typealias RecentSearch = FindMusicQuerySpec
typealias RecentSongSeed = FindMusicSongAnchor

internal data class SearchLabelPart(
    val text: String,
    val negative: Boolean,
)

internal fun formatSearchLabel(
    ingredients: List<SearchLabelPart>,
): String = ingredients.joinToString(" \u00b7 ") { ingredient ->
    if (ingredient.negative) "less like ${ingredient.text}" else ingredient.text
}.ifBlank { "Find music" }

internal fun recordingDisplayLabel(
    artist: String?,
    title: String?,
    fallback: String,
): String {
    val cleanArtist = artist?.trim()?.takeIf(String::isNotBlank)
    val cleanTitle = title?.trim()?.takeIf(String::isNotBlank)
    if (cleanArtist == null) return cleanTitle ?: fallback
    if (cleanTitle == null) return cleanArtist
    val titleSuffix = cleanTitle.drop(cleanArtist.length).trimStart()
    val titleAlreadyStartsWithArtist = cleanTitle.regionMatches(
        thisOffset = 0,
        other = cleanArtist,
        otherOffset = 0,
        length = cleanArtist.length,
        ignoreCase = true,
    ) && (
        titleSuffix.isEmpty() ||
            titleSuffix.first() in setOf('-', ':', '\u2013', '\u2014')
        )
    return if (titleAlreadyStartsWithArtist) cleanTitle else "$cleanArtist - $cleanTitle"
}

internal const val INCOMPATIBLE_SAVED_FIND_MUSIC_QUERY_MESSAGE =
    "This saved search is incompatible with this app version. Recreate it in Find Music."

internal fun validateFindMusicQueryContract(querySpec: FindMusicQuerySpec): String? = when {
    querySpec.schemaVersion !in 1..FindMusicQuerySpec.CURRENT_SCHEMA_VERSION ->
        INCOMPATIBLE_SAVED_FIND_MUSIC_QUERY_MESSAGE
    querySpec.schemaVersion != FindMusicQuerySpec.CURRENT_SCHEMA_VERSION ->
        INCOMPATIBLE_SAVED_FIND_MUSIC_QUERY_MESSAGE
    querySpec.embeddingSpace != FindMusicQuerySpec.EMBEDDING_SPACE_CLAMP3 ->
        INCOMPATIBLE_SAVED_FIND_MUSIC_QUERY_MESSAGE
    querySpec.rankingVersion == FindMusicQuerySpec.LEGACY_RANKING_VERSION ->
        INCOMPATIBLE_SAVED_FIND_MUSIC_QUERY_MESSAGE
    querySpec.rankingVersion != FindMusicQuerySpec.CURRENT_RANKING_VERSION ->
        INCOMPATIBLE_SAVED_FIND_MUSIC_QUERY_MESSAGE
    querySpec.libraryAddedDays != null &&
        querySpec.libraryAddedDays !in 1..MAX_LIBRARY_ADDED_DAYS ->
        "This saved query has an invalid Poweramp added-date day count"
    querySpec.resultLimit !in 1..FindMusicQuerySpec.MAX_RESULT_LIMIT ->
        "This saved query has an invalid result count: ${querySpec.resultLimit}"
    querySpec.textResultPlanner == FindMusicTextResultPlanner.VARIED_DPP &&
        !querySpec.isSimplePositiveTextOnly ->
        "Text Varied results need exactly one positive text description"
    querySpec.textResultPlanner == FindMusicTextResultPlanner.VARIED_ALL_OF_DPP &&
        (
            querySpec.operator != FindMusicOperator.ALL_OF ||
                querySpec.activeIngredientCount < 2
            ) ->
        "All-of Varied results need at least two active ingredients"
    querySpec.textIngredients.size > FindMusicQuerySpec.MAX_TEXT_INGREDIENTS ->
        "This saved query has too many text ingredients"
    (querySpec.textIngredients.map { it.weight } + querySpec.songSeeds.map { it.weight }).any {
        !it.isFinite() || it < 0f || it > 1f
    } -> "Every Find Music ingredient weight must be between 0% and 100%"
    querySpec.textIngredients.any { it.query.isBlank() && it.weight > 0f } ->
        "A blank Find Music description cannot have active weight"
    !querySpec.hasActivePositiveIngredient ->
        "Add at least one Like ingredient to define the sound you want"
    querySpec.operator == FindMusicOperator.ALL_OF && querySpec.refineSpec != null ->
        "All of cannot carry a Refine neighborhood"
    querySpec.operator == FindMusicOperator.REFINE && querySpec.refineSpec == null ->
        "Refine needs a primary ingredient and neighborhood"
    querySpec.operator == FindMusicOperator.REFINE && querySpec.activeIngredientCount != 2 ->
        "Refine needs exactly two active ingredients"
    querySpec.operator == FindMusicOperator.REFINE &&
        (querySpec.refineSpec?.primaryIngredientIndex ?: -1) !in 0 until
        querySpec.activeIngredientCount ->
        "Refine primary ingredient is outside the active request"
    querySpec.operator == FindMusicOperator.REFINE && run {
        val activeSigns = querySpec.activeTextIngredients.map { it.negative } +
            querySpec.songSeeds.filter { it.weight > 0f }.map { it.negative }
        activeSigns.getOrNull(checkNotNull(querySpec.refineSpec).primaryIngredientIndex) == true
    } -> "Refine's primary ingredient must be Like"
    else -> null
}
