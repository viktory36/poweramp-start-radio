package com.powerampstartradio.ui

import java.text.SimpleDateFormat
import java.util.Calendar
import java.util.Date
import java.util.Locale
import java.util.TimeZone
import kotlin.math.roundToInt

/** Pure, testable session-history text derived only from persisted evidence. */
object SessionEvidenceText {
    fun seedTitle(session: RadioResult): String {
        val title = session.seedTrack.title
        if (session.delivery?.origin != QueueOrigin.HISTORY_REQUEUE) return title
        return title.removePrefix("Replay: ").takeIf(String::isNotBlank) ?: title
    }

    internal fun sessionHeaderSummary(
        session: RadioResult,
        modeSignal: String?,
    ): String = listOfNotNull(
        modeSignal,
        sessionOutcomeSummary(session),
    ).filter(String::isNotBlank).joinToString(" \u00b7 ")

    internal fun sessionOutcomeSummary(session: RadioResult): String =
        sessionOutcomeSummary(session, includeDeliveryContext = true)

    internal fun compactSessionOutcomeSummary(session: RadioResult): String =
        sessionOutcomeSummary(session, includeDeliveryContext = false)

    private fun sessionOutcomeSummary(
        session: RadioResult,
        includeDeliveryContext: Boolean,
    ): String = buildString {
        fun add(part: String) {
            if (isNotEmpty()) append(" \u00b7 ")
            append(part)
        }
        if (!session.isComplete) {
            add("${session.tracks.size} of ${session.totalExpected} selected")
            add(
                if (session.effectiveOutcome == RadioSessionOutcome.CANCELLED) {
                    "cancelled"
                } else {
                    "incomplete"
                },
            )
        } else if (session.delivery != null) {
            if (session.queuedCount == session.requestedCount) {
                add("${session.queuedCount} ${plural(session.queuedCount, "track")} queued")
            } else {
                add("${session.queuedCount} of ${session.requestedCount} queued")
            }
            when (session.effectiveOutcome) {
                RadioSessionOutcome.SUCCEEDED -> Unit
                RadioSessionOutcome.PARTIAL_FAILED -> add("partial")
                RadioSessionOutcome.CANCELLED -> add("cancelled")
            }
            if (session.delivery?.verificationComplete == false) {
                add("final queue check incomplete")
            }
            if (!session.isDirectQueue &&
                session.totalExpected > 0 &&
                session.rankedCount < session.totalExpected
            ) {
                add("${session.rankedCount} of ${session.totalExpected} selected")
            } else if (session.rankedCount != session.requestedCount) {
                add("${session.rankedCount} selected")
            }
            if (session.resolvedCount != session.rankedCount) {
                add("${session.resolvedCount} found in Poweramp")
            }
            session.delivery?.unexpectedObservedCount?.takeIf { it > 0 }?.let {
                add("$it extra Poweramp entries")
            }
            if (includeDeliveryContext &&
                session.delivery?.origin !in setOf(
                    QueueOrigin.APP_RADIO,
                    QueueOrigin.HISTORY_REQUEUE,
                )
            ) {
                add(session.origin.displayLabel)
            } else if (session.delivery?.origin == QueueOrigin.WIDGET_RADIO) {
                add(session.origin.displayLabel)
            }
            if (includeDeliveryContext) {
                session.directQueuePlacement?.let { placement ->
                    add(
                        when (placement) {
                            DirectQueuePlacement.REPLACE_UPCOMING -> "replaced upcoming"
                            DirectQueuePlacement.APPEND -> "appended"
                        },
                    )
                }
            }
        } else {
            if (session.failedCount > 0) {
                add("${session.queuedCount} of ${session.requestedCount} queued")
            } else {
                add("${session.queuedCount} ${plural(session.queuedCount, "track")}")
            }
            when (session.outcome) {
                RadioSessionOutcome.PARTIAL_FAILED -> add("partial")
                RadioSessionOutcome.CANCELLED -> add("cancelled")
                RadioSessionOutcome.SUCCEEDED -> Unit
                null -> if (session.failedCount > 0) add("partial")
            }
        }
    }

    internal fun sessionDrawerSubtitle(
        session: RadioResult,
        modeSignal: String,
        time: String,
    ): String = listOf(sessionDrawerSummary(session, modeSignal), time)
        .filter(String::isNotBlank)
        .joinToString(" \u00b7 ")

    internal fun sessionDrawerSummary(
        session: RadioResult,
        modeSignal: String,
    ): String = listOf(modeSignal, compactSessionOutcomeSummary(session))
        .filter(String::isNotBlank)
        .joinToString(" \u00b7 ")

    internal fun historyTimestamp(
        timestamp: Long,
        now: Long = System.currentTimeMillis(),
        locale: Locale = Locale.getDefault(),
        timeZone: TimeZone = TimeZone.getDefault(),
    ): String {
        val event = Calendar.getInstance(timeZone, locale).apply { timeInMillis = timestamp }
        val current = Calendar.getInstance(timeZone, locale).apply { timeInMillis = now }
        val yesterday = (current.clone() as Calendar).apply { add(Calendar.DAY_OF_YEAR, -1) }
        val day = when {
            event.sameDay(current) -> "Today"
            event.sameDay(yesterday) -> "Yesterday"
            event.get(Calendar.YEAR) == current.get(Calendar.YEAR) ->
                SimpleDateFormat("d MMM", locale).apply { this.timeZone = timeZone }
                    .format(Date(timestamp))
            else -> SimpleDateFormat("d MMM yyyy", locale).apply { this.timeZone = timeZone }
                .format(Date(timestamp))
        }
        val time = SimpleDateFormat("HH:mm", locale).apply { this.timeZone = timeZone }
            .format(Date(timestamp))
        return "$day \u00b7 $time"
    }

    fun artistConstraints(config: RadioConfig): String = if (!config.artistLimitsEnabled) {
        "Artist-credit limits off"
    } else {
        val spacing = if (config.minArtistSpacing == 0) {
            "no spacing limit"
        } else {
            "${config.minArtistSpacing} ${plural(config.minArtistSpacing, "track")} between the same credit"
        }
        "Artist-credit limits \u00b7 max ${config.maxPerArtist} " +
            "${plural(config.maxPerArtist, "track")} with the same credit \u00b7 $spacing"
    }

    fun seedIdentity(artist: String?, album: String?): String? = listOfNotNull(
        artist?.trim()?.takeIf(String::isNotBlank),
        album?.trim()?.takeIf(String::isNotBlank),
    ).distinct().joinToString(" \u00b7 ").takeIf(String::isNotBlank)

    fun findMusicMode(evidence: FindMusicSessionEvidence): String = when {
        evidence.querySpec.isSimplePositiveTextOnly ->
            "Text \u00b7 ${textPlannerLabel(evidence.querySpec.textResultPlanner)}"
        evidence.querySpec.activeIngredientCount == 1 -> "Closest"
        evidence.querySpec.operator == FindMusicOperator.ALL_OF ->
            if (
                evidence.querySpec.textResultPlanner ==
                FindMusicTextResultPlanner.VARIED_ALL_OF_DPP
            ) {
                "All of \u00b7 Varied (DPP)"
            } else {
                "All of \u00b7 Ranked"
            }
        else -> "Refine"
    }

    fun textPlannerLabel(planner: FindMusicTextResultPlanner): String = when (planner) {
        FindMusicTextResultPlanner.CLOSEST -> "Closest"
        FindMusicTextResultPlanner.VARIED_DPP -> "Varied (DPP)"
        FindMusicTextResultPlanner.VARIED_ALL_OF_DPP -> "Varied (DPP)"
    }

    fun textPlannerDescription(planner: FindMusicTextResultPlanner): String = when (planner) {
        FindMusicTextResultPlanner.CLOSEST ->
            "Cosine-ranks every candidate recording against the text, strongest first."
        FindMusicTextResultPlanner.VARIED_DPP ->
            "Runs greedy DPP over the complete selected candidate domain, using text match as " +
                "quality while rewarding variety across the set."
        FindMusicTextResultPlanner.VARIED_ALL_OF_DPP ->
            "Runs greedy DPP over the complete All-of order, preserving strong joint matches " +
                "while rewarding difference across the selected set."
    }

    fun graphExploration(expectedRouteLinks: Double, stopChance: Float? = null): String? {
        if (!expectedRouteLinks.isFinite() || expectedRouteLinks < 0.0) return null
        val roundedLinks = expectedRouteLinks.roundToInt()
        val unit = if (roundedLinks == 1) "move" else "moves"
        return buildString {
            append("Graph Explorer \u00b7 typical path about $roundedLinks track-to-track $unit")
            stopChance?.takeIf { it.isFinite() && it in 0f..1f }?.let {
                append(" \u00b7 ${(it * 100f).roundToInt()}% stop chance after each move")
            }
        }
    }

    fun findMusicQuery(evidence: FindMusicSessionEvidence): String = buildString {
        val query = evidence.querySpec
        val comparison = findMusicRankingScope(
            query = query,
            objectiveRankingDomainCount = evidence.objectiveRankingDomainCount,
            ingredientRankingDomainCount = evidence.ingredientRankingDomainCount,
        )
        if (query.isSimplePositiveTextOnly) {
            append(textPlannerDescription(query.textResultPlanner))
            comparison?.let { append(" \u00b7 ").append(it) }
        } else if (query.activeIngredientCount == 1) {
            append("Cosine-ranks every candidate recording against the selected recording, strongest first.")
            comparison?.let { append(" \u00b7 ").append(it) }
        } else {
            append(findMusicRecipe(query))
            if (
                query.operator == FindMusicOperator.ALL_OF &&
                query.textResultPlanner == FindMusicTextResultPlanner.VARIED_ALL_OF_DPP
            ) {
                append('\n').append(textPlannerDescription(query.textResultPlanner))
            }
            comparison?.let { append('\n').append(it) }
        }
        query.effectiveLibraryAddedDays?.let { days ->
            append('\n').append("Candidates \u00b7 ").append(libraryAddedDaysLabel(days))
        }
        evidence.stableResultReduction.collapsedEquivalentCount.takeIf { it > 0 }?.let {
            append(if (query.isSimplePositiveTextOnly) " \u00b7 " else "\n")
            append("$it verified ")
                .append(plural(it, "copy", "copies"))
                .append(" skipped")
        }
    }

    fun findMusicRankingScope(
        query: FindMusicQuerySpec,
        objectiveRankingDomainCount: Int?,
        ingredientRankingDomainCount: Int?,
    ): String? = when {
        query.isSimplePositiveTextOnly -> objectiveRankingDomainCount?.let {
            "Compared across ${formatCount(it)} candidate recordings"
        }
        query.operator == FindMusicOperator.ALL_OF -> when {
            objectiveRankingDomainCount != null && ingredientRankingDomainCount != null &&
                objectiveRankingDomainCount != ingredientRankingDomainCount ->
                "Overall rank among ${formatCount(objectiveRankingDomainCount)} eligible recordings" +
                    " \u00b7 ingredient ranks across ${formatCount(ingredientRankingDomainCount)} " +
                    "recordings in this library scope"
            objectiveRankingDomainCount != null && ingredientRankingDomainCount != null ->
                "Ranks compare ${formatCount(objectiveRankingDomainCount)} recordings in this " +
                    "library scope"
            objectiveRankingDomainCount != null ->
                "Overall rank among ${formatCount(objectiveRankingDomainCount)} eligible recordings"
            ingredientRankingDomainCount != null ->
                "Ingredient ranks across ${formatCount(ingredientRankingDomainCount)} recordings " +
                    "in this library scope"
            else -> null
        }
        else -> when {
            objectiveRankingDomainCount != null && ingredientRankingDomainCount != null ->
                "${formatCount(objectiveRankingDomainCount)} recordings in the primary " +
                    "neighborhood \u00b7 ingredient ranks across " +
                    "${formatCount(ingredientRankingDomainCount)} recordings in this library scope"
            ingredientRankingDomainCount != null ->
                "Ingredient ranks across ${formatCount(ingredientRankingDomainCount)} recordings " +
                    "in this library scope"
            else -> null
        }
    }

    fun findMusicRecipe(query: FindMusicQuerySpec): String {
        if (query.activeIngredientCount == 1) return "Closest \u00b7 ${query.displayLabel}"
        val weights = query.activeTextIngredients.map { it.weight } +
            query.songSeeds.filter { it.weight > 0f }.map { it.weight }
        if (weights.isEmpty()) return query.displayLabel
        val totalWeight = weights.sumOf { it.toDouble() }
        return when (query.operator) {
            FindMusicOperator.ALL_OF -> {
                val ingredients = query.activeEvidenceLabels.zip(weights).map { (label, weight) ->
                    val normalizedPercent = weight.toDouble() / totalWeight * 100.0
                    "$label ${formatInfluencePercent(normalizedPercent)} priority"
                }
                (listOf("All of") + ingredients).joinToString(" \u00b7 ")
            }
            FindMusicOperator.REFINE -> {
                val refine = query.refineSpec ?: return query.displayLabel
                val primary = query.activeEvidenceLabels[refine.primaryIngredientIndex]
                val secondary = query.activeEvidenceLabels[1 - refine.primaryIngredientIndex]
                "Refine \u00b7 keep close to $primary \u00b7 rank by $secondary \u00b7 " +
                    "nearest ${refineNeighborhoodPercent(refine.neighborhood)}"
            }
        }
    }

    private fun formatInfluencePercent(value: Double): String {
        val roundedTenths = kotlin.math.round(value * 10.0) / 10.0
        return if (roundedTenths == kotlin.math.round(roundedTenths)) {
            String.format(Locale.US, "%.0f%%", roundedTenths)
        } else {
            String.format(Locale.US, "%.1f%%", roundedTenths)
        }
    }

    fun findMusicTrack(
        session: FindMusicSessionEvidence,
        track: FindMusicTrackEvidence,
    ): String? {
        val query = session.querySpec
        return when {
            query.isSimplePositiveTextOnly -> {
                val total = session.objectiveRankingDomainCount ?: return null
                if (track.objectiveRank !in 1..total) return null
                val rank = LibraryRankEvidenceText.rank(track.objectiveRank) ?: return null
                "Text match $rank"
            }
            query.operator == FindMusicOperator.REFINE -> {
                val domainCount = session.ingredientRankingDomainCount ?: return null
                val primaryIndex = query.refineSpec?.primaryIngredientIndex ?: return null
                val secondaryIndex = 1 - primaryIndex
                val primaryRank = LibraryRankEvidenceText.rankFromUpperCdfPercentile(
                    track.ingredientPercentiles.getOrNull(primaryIndex) ?: return null,
                    domainCount,
                )?.let { LibraryRankEvidenceText.rank(it) } ?: return null
                val secondaryRank = LibraryRankEvidenceText.rankFromUpperCdfPercentile(
                    track.ingredientPercentiles.getOrNull(secondaryIndex) ?: return null,
                    domainCount,
                )?.let { LibraryRankEvidenceText.rank(it) } ?: return null
                "Primary match $primaryRank \u00b7 Secondary match $secondaryRank"
            }
            else -> {
                val objectiveCount = session.objectiveRankingDomainCount ?: return null
                if (track.objectiveRank !in 1..objectiveCount) return null
                val overall = LibraryRankEvidenceText.rank(track.objectiveRank) ?: return null
                val domainCount = session.ingredientRankingDomainCount ?: return null
                if (track.ingredientPercentiles.size != query.activeEvidenceLabels.size) return null
                val ingredients = track.ingredientPercentiles.mapIndexed { index, percentile ->
                    val label = query.activeEvidenceLabels.getOrNull(index) ?: return null
                    val rank = LibraryRankEvidenceText.rankFromUpperCdfPercentile(
                        percentile,
                        domainCount,
                    )?.let { ingredientRank ->
                        LibraryRankEvidenceText.rank(ingredientRank)
                    }
                        ?: return null
                    "$label match $rank"
                }
                "Overall match \u00b7 $overall \u00b7 ${ingredients.joinToString(" \u00b7 ")}"
            }
        }
    }

    private fun refineNeighborhoodPercent(value: FindMusicRefineNeighborhood): String =
        when (value) {
            FindMusicRefineNeighborhood.TOP_0_25_PERCENT -> "0.25%"
            FindMusicRefineNeighborhood.TOP_0_5_PERCENT -> "0.5%"
            FindMusicRefineNeighborhood.TOP_1_PERCENT -> "1%"
            FindMusicRefineNeighborhood.TOP_2_PERCENT -> "2%"
        }

    fun mmrPriorPick(title: String?): String? = title
        ?.takeIf(String::isNotBlank)
        ?.let { "Most similar earlier pick: \"$it\"" }

    fun seedReach(session: RadioResult): String? {
        val queuedRows = session.tracks.filter { it.status == QueueStatus.QUEUED }
        if (queuedRows.isEmpty()) return null
        val queuedEvidence = queuedRows.mapNotNull { track ->
            SeedDistanceEvidencePolicy.evidenceOrNull(session, track)
        }
        if (queuedEvidence.size != queuedRows.size) return null

        val ranks = queuedEvidence.map { it.seedRank }.sorted()
        val total = queuedEvidence.map { it.rankingIdentityCount }.distinct().singleOrNull()
            ?: return null
        val scopeLabel = if (
            !session.isComplete ||
            queuedRows.size != session.tracks.size ||
            session.queuedCount != session.requestedCount ||
            session.queuedCount != queuedRows.size
        ) {
            "Queued distance from seed"
        } else {
            "Distance from seed"
        }
        if (ranks.size == 1) {
            val evidence = LibraryRankEvidenceText.rankWithTopFraction(ranks[0], total)
                ?: return null
            return "$scopeLabel \u00b7 $evidence"
        }
        val closest = ranks.first()
        val farthest = ranks.last()
        val farthestTop = LibraryRankEvidenceText.topFraction(farthest, total) ?: return null
        val medianRank = if (ranks.size % 2 == 1) {
            ranks[ranks.size / 2]
        } else {
            val lowerMiddle = ranks[ranks.size / 2 - 1]
            val upperMiddle = ranks[ranks.size / 2]
            ((lowerMiddle.toLong() + upperMiddle.toLong()) / 2.0).roundToInt()
        }
        val typicalRank = LibraryRankEvidenceText.rank(medianRank) ?: return null
        val closestRank = LibraryRankEvidenceText.rank(closest) ?: return null
        val farthestRank = LibraryRankEvidenceText.rank(farthest) ?: return null
        val farthestFraction = farthestTop.removePrefix("top ")
        return "Typical ${scopeLabel.replaceFirstChar(Char::lowercase)} \u00b7 " +
            "around $typicalRank nearest\n" +
            "Range \u00b7 $closestRank to $farthestRank nearest \u00b7 farthest in the closest " +
            "$farthestFraction of ${String.format(Locale.US, "%,d", total)}"
    }

    fun textMatchReach(
        objectiveRanks: List<Int>,
        objectiveDomainCount: Int?,
        queuedOnly: Boolean = false,
    ): String? {
        val total = objectiveDomainCount ?: return null
        if (objectiveRanks.isEmpty() || objectiveRanks.any { it !in 1..total }) return null
        val closest = objectiveRanks.min()
        val farthest = objectiveRanks.max()
        val topFraction = LibraryRankEvidenceText.topFraction(farthest, total) ?: return null
        val label = if (queuedOnly) "Queued match range" else "Match range"
        val formattedTotal = String.format(Locale.US, "%,d", total)
        val closestRank = LibraryRankEvidenceText.rank(closest) ?: return null
        val farthestRank = LibraryRankEvidenceText.rank(farthest) ?: return null
        return if (closest == farthest) {
            "$label \u00b7 $closestRank of $formattedTotal eligible recordings \u00b7 " +
                topFraction
        } else {
            "$label \u00b7 $closestRank to $farthestRank of $formattedTotal eligible recordings \u00b7 " +
                "farthest in $topFraction"
        }
    }

    fun findMusicReach(session: RadioResult): String? {
        val evidence = session.findMusicSessionEvidence ?: return null
        if (!evidence.querySpec.isSimplePositiveTextOnly ||
            evidence.querySpec.textResultPlanner != FindMusicTextResultPlanner.VARIED_DPP
        ) {
            return null
        }
        val queuedRows = session.tracks.filter { it.status == QueueStatus.QUEUED }
        val ranks = queuedRows.mapNotNull { it.findMusicEvidence?.objectiveRank }
        if (ranks.size != queuedRows.size) return null
        val queuedOnly = !session.isComplete || session.queuedCount < session.requestedCount
        return textMatchReach(
            objectiveRanks = ranks,
            objectiveDomainCount = evidence.objectiveRankingDomainCount,
            queuedOnly = queuedOnly,
        )
    }

    private fun Calendar.sameDay(other: Calendar): Boolean =
        get(Calendar.ERA) == other.get(Calendar.ERA) &&
            get(Calendar.YEAR) == other.get(Calendar.YEAR) &&
            get(Calendar.DAY_OF_YEAR) == other.get(Calendar.DAY_OF_YEAR)

    private fun plural(value: Int, singular: String, plural: String = "${singular}s"): String =
        if (value == 1) singular else plural

    private fun formatCount(value: Int): String = String.format(Locale.US, "%,d", value)
}
