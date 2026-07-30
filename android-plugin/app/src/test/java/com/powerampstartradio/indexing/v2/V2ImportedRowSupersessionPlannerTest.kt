package com.powerampstartradio.indexing.v2

import com.powerampstartradio.indexing.V2LegacyCompatibilityBinding
import com.powerampstartradio.indexing.V2LegacyCompatibilityEvidence
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class V2ImportedRowSupersessionPlannerTest {
    @Test
    fun `live shaped 29 row CUE group plans 27 repairs and two additions`() {
        val plan = V2ImportedRowSupersessionPlanner.plan(liveCueWorks, liveCueRepairBindings)

        assertEquals(listOf(84704L, 84707L), plan.additions.map { it.powerampFileId })
        assertEquals(27, plan.supersessions.size)
        assertEquals(EXPECTED_SUPERSEDED_TRACK_IDS, plan.supersededTrackIds)
        assertTrue(plan.blockedAmbiguousMappings.isEmpty())
        assertEquals(
            LIVE_CUE_PROVIDER_TO_LEGACY_TRACK,
            plan.supersessions.associate {
                it.selectedWork.powerampFileId to it.supersededTrackId
            },
        )
    }

    @Test
    fun `planning is invariant to selected work and binding input order`() {
        val expected = V2ImportedRowSupersessionPlanner.plan(liveCueWorks, liveCueRepairBindings)
        val reorderedWorks = liveCueWorks.drop(11) + liveCueWorks.take(11)
        val reorderedBindings = liveCueRepairBindings.reversed()

        assertEquals(
            expected,
            V2ImportedRowSupersessionPlanner.plan(reorderedWorks, reorderedBindings),
        )
    }

    @Test
    fun `many to one and one to many repair claims are blocked exhaustively`() {
        val first = work(powerampFileId = 1L, offsetMs = 0L)
        val second = work(powerampFileId = 2L, offsetMs = 1_000L)
        val unbound = work(powerampFileId = 3L, offsetMs = 2_000L)
        val plan = V2ImportedRowSupersessionPlanner.plan(
            selectedWorks = listOf(unbound, second, first),
            repairBindings = listOf(
                repair(powerampFileId = 1L, legacyTrackId = 101L),
                repair(powerampFileId = 1L, legacyTrackId = 102L),
                repair(powerampFileId = 2L, legacyTrackId = 102L),
            ),
        )

        assertEquals(listOf(unbound), plan.additions)
        assertTrue(plan.supersessions.isEmpty())
        assertTrue(plan.supersededTrackIds.isEmpty())
        assertEquals(
            listOf(
                V2BlockedImportedRowMapping(
                    selectedWork = first,
                    candidateLegacyTrackIds = listOf(101L, 102L),
                    sharedLegacyTrackIds = listOf(102L),
                    reasons = listOf(
                        V2ImportedRowMappingBlockReason.MULTIPLE_LEGACY_ROWS_FOR_SELECTED_WORK,
                        V2ImportedRowMappingBlockReason
                            .LEGACY_ROW_CLAIMED_BY_MULTIPLE_SELECTED_WORKS,
                    ),
                ),
                V2BlockedImportedRowMapping(
                    selectedWork = second,
                    candidateLegacyTrackIds = listOf(102L),
                    sharedLegacyTrackIds = listOf(102L),
                    reasons = listOf(
                        V2ImportedRowMappingBlockReason
                            .LEGACY_ROW_CLAIMED_BY_MULTIPLE_SELECTED_WORKS,
                    ),
                ),
            ),
            plan.blockedAmbiguousMappings,
        )
    }

    @Test
    fun `identical duplicate evidence does not manufacture ambiguity`() {
        val selected = work(powerampFileId = 1L, offsetMs = 0L)
        val binding = repair(powerampFileId = 1L, legacyTrackId = 101L)

        val plan = V2ImportedRowSupersessionPlanner.plan(
            selectedWorks = listOf(selected),
            repairBindings = listOf(binding, binding),
        )

        assertEquals(
            listOf(
                V2ImportedRowSupersession(
                    selectedWork = selected,
                    supersededTrackId = 101L,
                    evidence = V2LegacyCompatibilityEvidence.CUE_LOGICAL_METADATA_REPAIR,
                ),
            ),
            plan.supersessions,
        )
        assertEquals(listOf(101L), plan.supersededTrackIds)
        assertTrue(plan.blockedAmbiguousMappings.isEmpty())
    }

    @Test
    fun `non repair compatibility evidence cannot authorize supersession`() {
        val selected = work(powerampFileId = 1L, offsetMs = 0L)

        assertThrows(IllegalArgumentException::class.java) {
            V2ImportedRowSupersessionPlanner.plan(
                selectedWorks = listOf(selected),
                repairBindings = listOf(
                    V2LegacyCompatibilityBinding(
                        powerampFileId = 1L,
                        trackId = 101L,
                        evidence = V2LegacyCompatibilityEvidence.EXACT_ABSOLUTE_PATH,
                    ),
                ),
            )
        }
    }

    @Test
    fun `unknown duration ordinary work remains a valid addition locator`() {
        val unknown = work(powerampFileId = 1L, offsetMs = 0L).copy(
            providerSpan = V2CommittedProviderSpan(
                normalizedPhysicalPath = "/storage/emulated/0/Music/unknown.opus",
                offsetMs = 0L,
                durationMs = 0L,
            ),
        )

        assertEquals(
            listOf(unknown),
            V2ImportedRowSupersessionPlanner.plan(listOf(unknown), emptyList()).additions,
        )
        assertThrows(IllegalArgumentException::class.java) {
            V2ImportedRowSupersessionPlanner.plan(
                listOf(unknown.copy(providerSpan = unknown.providerSpan.copy(durationMs = -1L))),
                emptyList(),
            )
        }
    }

    private fun work(powerampFileId: Long, offsetMs: Long) = V2SelectedImportedRowWork(
        workId = "test-work-$powerampFileId",
        powerampFileId = powerampFileId,
        providerSpan = V2CommittedProviderSpan(
            normalizedPhysicalPath = "/storage/emulated/0/Music/test.flac",
            offsetMs = offsetMs,
            durationMs = 1_000L,
        ),
    )

    private fun repair(powerampFileId: Long, legacyTrackId: Long) =
        V2LegacyCompatibilityBinding(
            powerampFileId = powerampFileId,
            trackId = legacyTrackId,
            evidence = V2LegacyCompatibilityEvidence.CUE_LOGICAL_METADATA_REPAIR,
        )

    private data class LiveCueSpan(
        val powerampFileId: Long,
        val offsetMs: Long,
        val durationMs: Long,
    )

    companion object {
        private const val LIVE_CUE_PATH =
            "/storage/emulated/0/Music/fixtures/cue-album-image.flac"

        private val LIVE_CUE_SPANS = listOf(
            LiveCueSpan(84702L, 0L, 190000L),
            LiveCueSpan(84703L, 190000L, 68000L),
            LiveCueSpan(84704L, 258000L, 82000L),
            LiveCueSpan(84705L, 340000L, 73000L),
            LiveCueSpan(84706L, 413000L, 34000L),
            LiveCueSpan(84707L, 447000L, 183000L),
            LiveCueSpan(84708L, 630000L, 171000L),
            LiveCueSpan(84709L, 801000L, 169000L),
            LiveCueSpan(84710L, 970000L, 230666L),
            LiveCueSpan(84711L, 1200666L, 209334L),
            LiveCueSpan(84712L, 1410000L, 143000L),
            LiveCueSpan(84713L, 1553000L, 194000L),
            LiveCueSpan(84714L, 1747000L, 155000L),
            LiveCueSpan(84715L, 1902000L, 193000L),
            LiveCueSpan(84716L, 2095000L, 233000L),
            LiveCueSpan(84717L, 2328000L, 162000L),
            LiveCueSpan(84718L, 2490000L, 210000L),
            LiveCueSpan(84719L, 2700000L, 320000L),
            LiveCueSpan(84720L, 3020000L, 110000L),
            LiveCueSpan(84721L, 3130000L, 115000L),
            LiveCueSpan(84722L, 3245000L, 66000L),
            LiveCueSpan(84723L, 3311000L, 133000L),
            LiveCueSpan(84724L, 3444000L, 130000L),
            LiveCueSpan(84725L, 3574000L, 49000L),
            LiveCueSpan(84726L, 3623000L, 229000L),
            LiveCueSpan(84727L, 3852000L, 170000L),
            LiveCueSpan(84728L, 4022000L, 168000L),
            LiveCueSpan(84729L, 4190000L, 225000L),
            LiveCueSpan(84730L, 4415000L, 201930L),
        )

        private val LIVE_CUE_PROVIDER_TO_LEGACY_TRACK = linkedMapOf(
            84702L to 80154L,
            84703L to 80420L,
            84705L to 80419L,
            84706L to 80434L,
            84708L to 80234L,
            84709L to 80243L,
            84710L to 79982L,
            84711L to 80074L,
            84712L to 80324L,
            84713L to 80138L,
            84714L to 80292L,
            84715L to 80139L,
            84716L to 79970L,
            84717L to 80266L,
            84718L to 80072L,
            84719L to 79795L,
            84720L to 80379L,
            84721L to 80373L,
            84722L to 80421L,
            84723L to 80348L,
            84724L to 80354L,
            84725L to 80429L,
            84726L to 79986L,
            84727L to 80240L,
            84728L to 80249L,
            84729L to 80002L,
            84730L to 80111L,
        )

        private val EXPECTED_SUPERSEDED_TRACK_IDS =
            LIVE_CUE_PROVIDER_TO_LEGACY_TRACK.values.sorted()

        private val liveCueWorks = LIVE_CUE_SPANS.map { span ->
            V2SelectedImportedRowWork(
                workId = "live-cue-evidence-${span.powerampFileId}",
                powerampFileId = span.powerampFileId,
                providerSpan = V2CommittedProviderSpan(
                    normalizedPhysicalPath = LIVE_CUE_PATH,
                    offsetMs = span.offsetMs,
                    durationMs = span.durationMs,
                ),
            )
        }

        private val liveCueRepairBindings = LIVE_CUE_PROVIDER_TO_LEGACY_TRACK.map {
            (powerampFileId, legacyTrackId) ->
            V2LegacyCompatibilityBinding(
                powerampFileId = powerampFileId,
                trackId = legacyTrackId,
                evidence = V2LegacyCompatibilityEvidence.CUE_LOGICAL_METADATA_REPAIR,
            )
        }
    }
}
