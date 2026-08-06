package com.powerampstartradio.indexing

import com.powerampstartradio.indexing.v2.FailureDisposition
import com.powerampstartradio.indexing.v2.RetryTrigger
import com.powerampstartradio.indexing.v2.V2IndexingPreflightFailureCode
import org.junit.Assert.assertEquals
import org.junit.Test

class V2IndexingReadinessPolicyTest {
    @Test
    fun `ready count excludes durable failures and never-index choices`() {
        val available = track(1L, "/music/available.flac")
        val failed = track(2L, "/music/failed.flac")
        val hidden = track(3L, "/music/hidden.flac")
        val hiddenSpan = span(hidden)
        val exclusions = V2ResolvedTrackExclusions(
            never = listOf(V2PersistedTrackExclusion(hiddenSpan, null, hidden.powerampFileId)),
            ignored = emptyList(),
            neverIds = setOf(hidden.powerampFileId),
            ignoredIds = emptySet(),
        )
        val failedIdentity = requireNotNull(
            V2UnresolvedFailureIdentityPolicy.identityOrNull(
                stableTrackSpanId = "stable-track-span-v1-${"a".repeat(64)}",
                powerampFileId = failed.powerampFileId,
                providerPhysicalPath = failed.path,
                offsetMs = failed.offsetMs,
                durationMs = failed.durationMs.toLong(),
            ),
        )

        assertEquals(
            setOf(available.powerampFileId),
            V2IndexingReadinessPolicy.readyTrackIds(
                tracks = listOf(available, failed, hidden),
                exclusions = exclusions,
                attentionHistory = history(unresolvedFailures = setOf(failedIdentity)),
            ),
        )
    }

    @Test
    fun `preflight rejection stays out of ready set until explicit retry choice`() {
        val rejected = track(4L, "/music/rejected.flac")
        val rejectedSpan = span(rejected)
        val rejection = V2PreflightRejectedSpan(
            jobId = "job-a",
            attemptCreatedAtEpochMs = 1L,
            originalPowerampFileId = rejected.powerampFileId,
            providerSpan = rejectedSpan,
            code = V2IndexingPreflightFailureCode.SOURCE_UNREADABLE,
            disposition = FailureDisposition.RETRYABLE,
            retryTrigger = RetryTrigger.SOURCE_AVAILABLE,
            diagnostic = "source unavailable",
        )
        val exclusions = V2ResolvedTrackExclusions(
            never = emptyList(),
            ignored = emptyList(),
            neverIds = emptySet(),
            ignoredIds = emptySet(),
        )

        assertEquals(
            emptySet<Long>(),
            V2IndexingReadinessPolicy.readyTrackIds(
                tracks = listOf(rejected),
                exclusions = exclusions,
                attentionHistory = history(retainedPreflightRejections = listOf(rejection)),
            ),
        )
        assertEquals(
            setOf(rejected.powerampFileId),
            V2IndexingReadinessPolicy.readyTrackIds(
                tracks = listOf(rejected),
                exclusions = exclusions,
                attentionHistory = history(
                    retainedPreflightRejections = listOf(rejection),
                    preflightRetryChoices = setOf(rejectedSpan),
                ),
            ),
        )
    }

    @Test
    fun `explicit retry choice restores a runtime failure to the ready set`() {
        val failed = track(5L, "/music/retry.flac")
        val failedSpan = span(failed)
        val identity = requireNotNull(
            V2UnresolvedFailureIdentityPolicy.identityOrNull(
                stableTrackSpanId = "stable-track-span-v1-${"b".repeat(64)}",
                powerampFileId = failed.powerampFileId,
                providerPhysicalPath = failed.path,
                offsetMs = failed.offsetMs,
                durationMs = failed.durationMs.toLong(),
            ),
        )
        val exclusions = V2ResolvedTrackExclusions(
            never = emptyList(),
            ignored = emptyList(),
            neverIds = emptySet(),
            ignoredIds = emptySet(),
        )

        assertEquals(
            setOf(failed.powerampFileId),
            V2IndexingReadinessPolicy.readyTrackIds(
                tracks = listOf(failed),
                exclusions = exclusions,
                attentionHistory = history(
                    unresolvedFailures = setOf(identity),
                    preflightRetryChoices = setOf(failedSpan),
                ),
            ),
        )
    }

    @Test
    fun `obsolete ignored envelope does not hide a V2 ready track`() {
        val ignoredInV1 = track(6L, "/music/ignored.flac")
        val ignoredSpan = span(ignoredInV1)
        val exclusions = V2ResolvedTrackExclusions(
            never = emptyList(),
            ignored = listOf(V2PersistedTrackExclusion(ignoredSpan, null, ignoredInV1.powerampFileId)),
            neverIds = emptySet(),
            ignoredIds = setOf(ignoredInV1.powerampFileId),
        )

        assertEquals(
            setOf(ignoredInV1.powerampFileId),
            V2IndexingReadinessPolicy.readyTrackIds(
                tracks = listOf(ignoredInV1),
                exclusions = exclusions,
                attentionHistory = history(),
            ),
        )
    }

    private fun history(
        unresolvedFailures: Set<V2UnresolvedFailureOccurrenceIdentity> = emptySet(),
        retainedPreflightRejections: List<V2PreflightRejectedSpan> = emptyList(),
        preflightRetryChoices: Set<V2ProviderSpanLocator> = emptySet(),
    ) = V2IndexingAttentionHistory(
        unresolvedFailures = unresolvedFailures,
        retainedPreflightRejections = retainedPreflightRejections,
        preflightRetryChoices = preflightRetryChoices,
        fingerprint = "test",
    )

    private fun track(id: Long, path: String) = NewTrackDetector.UnindexedTrack(
        powerampFileId = id,
        artist = "Artist",
        album = "Album",
        title = "Track $id",
        durationMs = 10_000,
        path = path,
    )

    private fun span(track: NewTrackDetector.UnindexedTrack) = requireNotNull(
        V2TrackExclusionRepository.candidate(track),
    ).providerSpan
}
