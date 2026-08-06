package com.powerampstartradio.indexing

import com.powerampstartradio.indexing.v2.V2IndexingExecutionProfile
import com.powerampstartradio.indexing.v2.IndexingJobState
import com.powerampstartradio.indexing.v2.IndexingTrackState
import com.powerampstartradio.indexing.v2.RetryTrigger
import com.powerampstartradio.indexing.v2.TrackFailureCode
import com.powerampstartradio.indexing.v2.V2IndexingPreflightFailureCode
import com.powerampstartradio.indexing.v2.V2IndexingPreflightPhase
import com.powerampstartradio.indexing.v2.V2IndexingPreflightIntentState
import com.powerampstartradio.indexing.v2.V2DurableStageCounter
import com.powerampstartradio.indexing.v2.V2IndexingEtaCoverageSnapshot
import com.powerampstartradio.indexing.v2.V2IndexingEtaScope
import com.powerampstartradio.indexing.v2.V2IndexingExecutorEvent
import com.powerampstartradio.indexing.v2.V2IndexingProgressSnapshot
import com.powerampstartradio.indexing.v2.V2MeasuredWorkStage
import com.powerampstartradio.indexing.v2.V2StageAwareEtaEstimate
import com.powerampstartradio.indexing.v2.V2UnmeasuredIndexingWork
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.Locale

class IndexingUiPolicyTest {
    @Test
    fun listenerFailureCopyIsStableAndSpecificToEachOperation() {
        assertEquals(
            "Poweramp tracks could not be compared with indexed source spans. " +
                "No indexing status was changed.",
            indexingListenerFailureText(IndexingListenerFailureOperation.NEW_TRACK_SCAN),
        )
        assertEquals(
            "Indexed source spans could not be compared with the Poweramp library. Nothing was removed.",
            indexingListenerFailureText(IndexingListenerFailureOperation.CLEANUP_SCAN),
        )
        assertEquals(
            "The selected tracks could not be removed from the music index. " +
                "No indexed tracks were changed.",
            indexingListenerFailureText(IndexingListenerFailureOperation.CLEANUP_UPDATE),
        )
        assertEquals(
            "The app files could not be exported.",
            indexingListenerFailureText(IndexingListenerFailureOperation.EXPORT),
        )
        assertEquals(
            "Indexing could not start, and no saved request could be confirmed. " +
                "Reopen On-device indexing to check the current state.",
            indexingListenerFailureText(IndexingListenerFailureOperation.INDEXING_REQUEST),
        )
        assertEquals(
            "Indexing could not start. The saved request is available in On-device indexing.",
            indexingListenerFailureText(
                IndexingListenerFailureOperation.INDEXING_REQUEST,
                indexingRequestIsDurable = true,
            ),
        )
    }

    @Test
    fun indexingSourceLabelsNeverExposeStoredSourceTokens() {
        assertEquals("Indexed on this phone", indexedTrackSourceLabel("phone"))
        assertEquals("Indexed on this phone", indexedTrackSourceLabel("phone-v2"))
        assertNull(indexedTrackSourceLabel("desktop"))
        assertNull(indexedTrackSourceLabel("future-internal-token"))
    }

    @Test
    fun profileControlUsesCompactPresentationForPhonesAndLargeType() {
        assertTrue(useCompactIndexingProfileControl(maxWidthDp = 320f, fontScale = 1f))
        assertTrue(useCompactIndexingProfileControl(maxWidthDp = 600f, fontScale = 1.3f))
        assertFalse(useCompactIndexingProfileControl(maxWidthDp = 600f, fontScale = 1f))
    }

    @Test
    fun profileCopyStatesMeasuredUserTradeoffWithoutSchedulerJargon() {
        assertEquals(
            "Keep phone responsive",
            indexingExecutionProfileLabel(V2IndexingExecutionProfile.BACKGROUND),
        )
        assertTrue(
            indexingExecutionProfileDescription(V2IndexingExecutionProfile.BACKGROUND)
                .contains("Keeps the phone more responsive"),
        )
        assertTrue(
            indexingExecutionProfileDescription(V2IndexingExecutionProfile.BACKGROUND)
                .contains("Embeddings are identical"),
        )
        assertFalse(
            indexingExecutionProfileDescription(V2IndexingExecutionProfile.BACKGROUND)
                .contains("ms"),
        )
        assertEquals(
            "Keeps the phone more responsive; indexing may take longer. Embeddings are identical.",
            indexingExecutionProfileCompactDescription(V2IndexingExecutionProfile.BACKGROUND),
        )
    }

    @Test
    fun progressAndStartLabelsNameTheirOperationAndScope() {
        assertEquals(
            "Opening active music-index file library.db for SHA-256 verification · " +
                "64.0 MiB total to read",
            exactHashProgressText("active music-index file library.db", 0L, 64L * 1024L * 1024L),
        )
        assertEquals(
            "Hashing active music-index file library.db · 8.0 MiB of 64.0 MiB",
            exactHashProgressText(
                "active music-index file library.db",
                8L * 1024L * 1024L,
                64L * 1024L * 1024L,
            ),
        )
        assertEquals(
            "Beginning source-identity review for 79 selected tracks",
            preflightProgressText(0, 79, V2IndexingPreflightPhase.SOURCE_FINGERPRINTS),
        )
        assertEquals(
            "Reviewed 42 of 79 selected tracks for source identity",
            preflightProgressText(42, 79, V2IndexingPreflightPhase.SOURCE_FINGERPRINTS),
        )
        assertEquals(
            "Confirmed 42 of 79 selected audio files",
            preflightProgressText(42, 79, V2IndexingPreflightPhase.SOURCE_REVALIDATION),
        )
        assertEquals(
            "Hashed 2 of 4 model and tokenizer files",
            preflightProgressText(2, 4, V2IndexingPreflightPhase.MODEL_FINGERPRINTS),
        )
        assertEquals("Index 338 tracks", startIndexingActionLabel(338, 338, false))
        assertEquals(
            "Index 338 tracks (79 shown)",
            startIndexingActionLabel(338, 79, true),
        )
        assertEquals("Index 1 track", startIndexingActionLabel(1, 1, false))
        assertEquals(
            "Reviewed 80,000 of 80,433 selected tracks for source identity",
            preflightProgressText(
                80_000,
                80_433,
                V2IndexingPreflightPhase.SOURCE_FINGERPRINTS,
                Locale.US,
            ),
        )
    }

    @Test
    fun dailyUiOffersOnlyTheFastProfile() {
        assertEquals(
            listOf(V2IndexingExecutionProfile.FULL),
            userSelectableIndexingProfiles,
        )
    }

    @Test
    fun largeTrackCountsRemainExactAndReadable() {
        assertTrue(formatIndexingTrackCount(80_671, Locale.US) == "80,671")
    }

    @Test
    fun stoppedJobsNeverUseAnActiveIndexingHeading() {
        assertEquals(
            "Indexing interrupted",
            indexingJobHeading(IndexingJobState.INTERRUPTED, 348),
        )
        assertEquals(
            "Ready to resume indexing",
            indexingJobHeading(IndexingJobState.READY_TO_RESUME, 348),
        )
        assertEquals(
            "Indexing 348 tracks",
            indexingJobHeading(IndexingJobState.RUNNING, 348),
        )
    }

    @Test
    fun selectionSummaryContainsOnlyActionableTrackStates() {
        assertEquals(
            "348 candidates · 1 needs attention · 348 selected",
            indexingSelectionSummaryText(
                readyCount = 348,
                attentionCount = 1,
                otherNotReadyCount = 0,
                selectedCount = 348,
            ),
        )
        assertEquals("Retry 1 available track", retryAvailableTracksLabel(1))
        assertEquals("Retry 12 available tracks", retryAvailableTracksLabel(12))
    }

    @Test
    fun cleanupBlockReasonCoversKnownConflictingLifecycleStates() {
        assertTrue(
            databaseCleanupBlockedReason(
                durableJobActive = true,
                jobPlanningActive = false,
                exportActive = false,
            )!!.contains("current indexing job"),
        )
        assertTrue(
            databaseCleanupBlockedReason(
                durableJobActive = false,
                jobPlanningActive = true,
                exportActive = false,
            )!!.contains("job preparation"),
        )
        assertTrue(
            databaseCleanupBlockedReason(
                durableJobActive = false,
                jobPlanningActive = false,
                exportActive = true,
            )!!.contains("current export"),
        )
        assertNull(
            databaseCleanupBlockedReason(
                durableJobActive = false,
                jobPlanningActive = false,
                exportActive = false,
            ),
        )
    }

    @Test
    fun destructiveDiscardIsUnavailableOnceActivationOrTerminationBegins() {
        listOf(
            IndexingJobState.ACTIVATING,
            IndexingJobState.CANCELLING,
            IndexingJobState.CANCELLED,
            IndexingJobState.COMPLETE,
        ).forEach { state -> assertFalse(canDiscardIndexingJob(state)) }

        listOf(
            IndexingJobState.RUNNING,
            IndexingJobState.PAUSE_REQUESTED,
            IndexingJobState.PAUSED,
            IndexingJobState.INTERRUPTED,
            IndexingJobState.WAITING_FOR_INPUT,
        ).forEach { state -> assertTrue(canDiscardIndexingJob(state)) }
    }

    @Test
    fun failureActionsExistOnlyAtStableInteractiveJobStates() {
        listOf(
            IndexingJobState.PAUSED,
            IndexingJobState.WAITING_FOR_INPUT,
            IndexingJobState.INTERRUPTED,
            IndexingJobState.READY_TO_RESUME,
        ).forEach { state -> assertTrue(canOfferIndexingFailureActions(state)) }

        listOf(
            IndexingJobState.PLANNED,
            IndexingJobState.RUNNING,
            IndexingJobState.PAUSE_REQUESTED,
            IndexingJobState.ACTIVATING,
            IndexingJobState.CANCELLING,
            IndexingJobState.COMPLETE,
            IndexingJobState.CANCELLED,
        ).forEach { state -> assertFalse(canOfferIndexingFailureActions(state)) }
    }

    @Test
    fun staleFailureCommandsCannotRaceAStartedExecutor() {
        assertTrue(
            canAcceptIndexingFailureCommand(
                IndexingJobState.WAITING_FOR_INPUT,
                runnerActive = false,
            ),
        )
        assertFalse(
            canAcceptIndexingFailureCommand(
                IndexingJobState.WAITING_FOR_INPUT,
                runnerActive = true,
            ),
        )
        assertFalse(
            canAcceptIndexingFailureCommand(
                IndexingJobState.RUNNING,
                runnerActive = true,
            ),
        )
    }

    @Test
    fun notificationPauseExistsOnlyWhilePauseCanChangeState() {
        assertTrue(shouldOfferIndexingPauseAction(IndexingJobState.RUNNING))
        assertFalse(shouldOfferIndexingPauseAction(IndexingJobState.PAUSE_REQUESTED))
        assertFalse(shouldOfferIndexingPauseAction(IndexingJobState.ACTIVATING))
        assertFalse(shouldOfferIndexingPauseAction(IndexingJobState.PAUSED))
    }

    @Test
    fun terminalFailureHistoryCannotBecomeActionableOrUserRetryEligible() {
        listOf(IndexingJobState.COMPLETE, IndexingJobState.CANCELLED).forEach { terminal ->
            assertFalse(canOfferIndexingFailureActions(terminal))
            assertFalse(canUserRetryIndexingFailure(terminal, RetryTrigger.USER_REQUEST))
            assertFalse(
                isActionableIndexingFailure(
                    terminal,
                    IndexingTrackState.RETRYABLE_FAILURE,
                    hasFailureEvidence = true,
                ),
            )
        }
    }

    @Test
    fun terminalFailureHistoryRemainsVisibleAsOutcomeEvidence() {
        assertTrue(
            isVisibleUnresolvedIndexingFailure(
                IndexingTrackState.BLOCKED_FAILURE,
                hasFailureEvidence = true,
            ),
        )
        assertFalse(
            isVisibleUnresolvedIndexingFailure(
                IndexingTrackState.COMMITTED,
                hasFailureEvidence = true,
            ),
        )
        assertFalse(
            isVisibleUnresolvedIndexingFailure(
                IndexingTrackState.BLOCKED_FAILURE,
                hasFailureEvidence = false,
            ),
        )
    }

    @Test
    fun terminalFailureCanBeSelectedOnlyForARealNewRun() {
        assertTrue(
            canSelectIndexingFailureForNewRun(
                IndexingJobState.COMPLETE,
                RetryTrigger.NEW_JOB_REQUIRED,
                TrackFailureCode.SOURCE_FINGERPRINT_CHANGED,
            ),
        )
        assertTrue(
            canSelectIndexingFailureForNewRun(
                IndexingJobState.CANCELLED,
                RetryTrigger.DECODER_OR_APP_CHANGED,
                TrackFailureCode.UNSUPPORTED_CODEC_OR_CONTAINER,
            ),
        )
        assertFalse(
            canSelectIndexingFailureForNewRun(
                IndexingJobState.RUNNING,
                RetryTrigger.NEW_JOB_REQUIRED,
                TrackFailureCode.SOURCE_FINGERPRINT_CHANGED,
            ),
        )
        assertFalse(
            canSelectIndexingFailureForNewRun(
                IndexingJobState.COMPLETE,
                RetryTrigger.NEVER,
                TrackFailureCode.UNKNOWN_BLOCKED,
            ),
        )
        assertFalse(
            canSelectIndexingFailureForNewRun(
                IndexingJobState.COMPLETE,
                RetryTrigger.NEW_JOB_REQUIRED,
                TrackFailureCode.CONTAINER_EOS_MISMATCH,
            ),
        )
        assertTrue(
            canSelectIndexingFailureForNewRun(
                IndexingJobState.COMPLETE,
                RetryTrigger.NEW_JOB_REQUIRED,
                TrackFailureCode.CONTAINER_EOS_MISMATCH,
                sourceIdentityChanged = true,
            ),
        )
        assertTrue(canNeverIndexFailure(IndexingJobState.COMPLETE))
    }

    @Test
    fun bulkRetryRequiresAtLeastOneTruthfulUserRequestCandidate() {
        assertFalse(
            hasUserRetryEligibleFailure(
                IndexingJobState.WAITING_FOR_INPUT,
                listOf(RetryTrigger.NEW_JOB_REQUIRED, RetryTrigger.NEVER),
            ),
        )
        assertTrue(
            hasUserRetryEligibleFailure(
                IndexingJobState.WAITING_FOR_INPUT,
                listOf(RetryTrigger.NEW_JOB_REQUIRED, RetryTrigger.DECODER_OR_APP_CHANGED),
            ),
        )
        assertFalse(
            hasUserRetryEligibleFailure(
                IndexingJobState.RUNNING,
                listOf(RetryTrigger.USER_REQUEST),
            ),
        )
    }

    @Test
    fun selectReadyIsEnabledOnlyWhenItCanChangeSelection() {
        assertFalse(canSelectReadyTracks(emptySet(), emptySet()))
        assertFalse(canSelectReadyTracks(setOf(1L, 2L), setOf(1L, 2L, 3L)))
        assertTrue(canSelectReadyTracks(setOf(1L, 2L), setOf(1L)))
    }

    @Test
    fun trackSearchCanMatchEveryDisplayedRowField() {
        assertTrue(matchesIndexingTrackSearch("bonobo", "Drift", "Bonobo", null))
        assertTrue(matchesIndexingTrackSearch("codec", "Track", "Unsupported codec"))
        assertTrue(matchesIndexingTrackSearch("album/file", "/music/Album/File.flac"))
        assertTrue(matchesIndexingTrackSearch("  ", null))
        assertFalse(matchesIndexingTrackSearch("sleep", "Daylight", "Aesop Rock"))
    }

    @Test
    fun failureCountSaysHowManyTimesTheTrackFailed() {
        assertEquals("Failed once", formatFailureOccurrences(1))
        assertEquals("Failed 3 times", formatFailureOccurrences(3))
        assertFalse(formatFailureOccurrences(3).contains("attempt"))
    }

    @Test
    fun failureAndRetryCopyExposesUserMeaningInsteadOfEnumNames() {
        TrackFailureCode.entries.forEach { code ->
            val text = indexingFailureSummary(code)
            assertTrue(text.isNotBlank())
            assertFalse(text.contains('_'))
            assertFalse(text == code.name.lowercase().replace('_', ' '))
        }
        RetryTrigger.entries.forEach { trigger ->
            val text = indexingRetryGuidance(trigger)
            assertTrue(text.isNotBlank())
            assertFalse(text.contains('_'))
            assertFalse(text.contains(trigger.name, ignoreCase = true))
        }
        assertTrue(
            indexingFailureSummary(TrackFailureCode.CONTAINER_EOS_MISMATCH)
                .contains("container", ignoreCase = true),
        )
        assertEquals(
            "Source identity changed before the embedding could be saved",
            indexingFailureSummary(TrackFailureCode.IMPORTED_ROW_AUTHORIZATION_CHANGED),
        )
        val eosGuidance = indexingFailureGuidance(
            TrackFailureCode.CONTAINER_EOS_MISMATCH,
            RetryTrigger.NEW_JOB_REQUIRED,
        )
        assertTrue(eosGuidance.contains("corrupt or truncated", ignoreCase = true))
        assertTrue(eosGuidance.contains("repair or replace", ignoreCase = true))
        assertFalse(eosGuidance.contains("new indexing run", ignoreCase = true))
        V2IndexingPreflightFailureCode.entries.forEach { code ->
            val text = preflightRejectionSummary(code)
            assertTrue(text.isNotBlank())
            assertFalse(text.contains('_'))
            assertFalse(text.contains("Poweramp row"))
        }
    }

    @Test
    fun durableSummaryReportsExactOutcomesAndCurrentTrack() {
        val zero = V2DurableStageCounter(0, 0, 0)
        val progress = V2IndexingProgressSnapshot(
            resolvedTracks = 9,
            succeededTracks = 6,
            blockedTracks = 2,
            skippedTracks = 1,
            totalTracks = 12,
            tracksWithMertFeatures = 0,
            tracksWithClampVectors = 0,
            mertWindows = zero,
            clampSegments = zero,
            databaseCommits = zero,
            activation = zero,
            activeTrackOrdinal = 6,
            activeStage = null,
        )

        assertEquals(
            "6 of 12 indexed \u00b7 2 need attention \u00b7 1 skipped",
            formatDurableTrackCounts(IndexingJobState.COMPLETE, progress),
        )
        assertEquals(
            "6 of 12 embeddings saved",
            formatDurableTrackCounts(
                IndexingJobState.RUNNING,
                progress.copy(blockedTracks = 0, skippedTracks = 0),
            ),
        )
        assertEquals(
            "0 of 12 embeddings saved",
            formatDurableTrackCounts(
                IndexingJobState.RUNNING,
                progress.copy(succeededTracks = 0, blockedTracks = 0, skippedTracks = 0),
            ),
        )
        assertEquals(
            "No tracks were added; unfinished work was discarded",
            formatDurableTrackCounts(IndexingJobState.CANCELLED, progress),
        )
        assertEquals(
            "Saved audio-analysis checkpoints: 14 of 226",
            formatDurableStageTrackCounts(
                IndexingJobState.PAUSED,
                progress.copy(
                    resolvedTracks = 0,
                    succeededTracks = 0,
                    blockedTracks = 0,
                    skippedTracks = 0,
                    totalTracks = 226,
                    tracksWithMertFeatures = 14,
                    tracksWithClampVectors = 0,
                ),
            ),
        )
        assertEquals(
            "Saved audio-analysis checkpoints: 12 of 12 \u00b7 " +
                "Saved music embeddings: 3 of 12",
            formatDurableStageTrackCounts(
                IndexingJobState.RUNNING,
                progress.copy(
                    succeededTracks = 0,
                    blockedTracks = 0,
                    skippedTracks = 0,
                    tracksWithMertFeatures = 12,
                    tracksWithClampVectors = 3,
                ),
            ),
        )
        assertEquals(
            null,
            formatDurableStageTrackCounts(
                IndexingJobState.PAUSED,
                progress.copy(
                    succeededTracks = 0,
                    blockedTracks = 0,
                    skippedTracks = 0,
                    tracksWithMertFeatures = 0,
                    tracksWithClampVectors = 0,
                ),
            ),
        )
        assertNull(
            formatDurableStageTrackCounts(
                IndexingJobState.COMPLETE,
                progress.copy(
                    resolvedTracks = 12,
                    succeededTracks = 12,
                    blockedTracks = 0,
                    skippedTracks = 0,
                    tracksWithMertFeatures = 12,
                    tracksWithClampVectors = 12,
                ),
            ),
        )
        assertEquals(
            "Track 7 of 12 \u00b7 Artist - Title",
            formatCurrentIndexingTrack(6, 6, " Artist - Title ", 12),
        )
        assertNull(formatCurrentIndexingTrack(6, 5, "Old title", 12))
        assertNull(formatCurrentIndexingTrack(6, 6, "  ", 12))
        assertNull(formatCurrentIndexingTrack(null, null, null, 12))
        assertNull(formatCurrentIndexingTrack(null, 6, "Finished track", 12))
        assertEquals(
            "1 selected track could not start indexing; review it below.",
            formatPreflightAttentionSummary(1),
        )
        assertEquals(
            "3 selected tracks could not start indexing; review them below.",
            formatPreflightAttentionSummary(3),
        )
        assertNull(formatPreflightAttentionSummary(0))
        assertEquals(
            "12 tracks indexed",
            formatDurableTrackCounts(
                IndexingJobState.COMPLETE,
                progress.copy(
                    resolvedTracks = 12,
                    succeededTracks = 12,
                    blockedTracks = 0,
                    skippedTracks = 0,
                    activeTrackOrdinal = null,
                ),
            ),
        )
        assertEquals(
            "12 embeddings saved",
            formatDurableTrackCounts(
                IndexingJobState.RUNNING,
                progress.copy(
                    resolvedTracks = 12,
                    succeededTracks = 12,
                    blockedTracks = 0,
                    skippedTracks = 0,
                    activeTrackOrdinal = null,
                ),
            ),
        )
        assertEquals(
            "79,998 of 80,433 embeddings saved \u00b7 1 needs attention \u00b7 1 skipped",
            formatDurableTrackCounts(
                IndexingJobState.RUNNING,
                progress.copy(
                    resolvedTracks = 80_000,
                    succeededTracks = 79_998,
                    blockedTracks = 1,
                    skippedTracks = 1,
                    totalTracks = 80_433,
                    activeTrackOrdinal = null,
                ),
                Locale.US,
            ),
        )
        assertEquals(
            "1 track indexed",
            formatDurableTrackCounts(
                IndexingJobState.COMPLETE,
                progress.copy(
                    resolvedTracks = 1,
                    succeededTracks = 1,
                    blockedTracks = 0,
                    skippedTracks = 0,
                    totalTracks = 1,
                    activeTrackOrdinal = null,
                ),
            ),
        )

        val completedTrackEvent = V2IndexingExecutorEvent(
            jobId = "job",
            workId = "track-12",
            trackOrdinal = 11,
            trackTitle = "Artist - Title",
            stage = V2MeasuredWorkStage.DATABASE_COMMITS,
            completedUnits = 1L,
            totalUnits = 1L,
            detail = "Saved",
        )
        val allCommitted = progress.copy(
            resolvedTracks = 12,
            succeededTracks = 12,
            blockedTracks = 0,
            skippedTracks = 0,
            activeTrackOrdinal = null,
        )
        assertFalse(shouldShowIndexingStageEvent(completedTrackEvent, allCommitted))
        assertEquals(
            "Indexing run \u00b7 Restoring the saved checkpoint and current operation",
            indexingStageFallbackText(
                state = IndexingJobState.RUNNING,
                progress = allCommitted,
                hasVisibleStageEvent = false,
            ),
        )
        assertNull(
            indexingStageFallbackText(
                state = IndexingJobState.RUNNING,
                progress = allCommitted,
                hasVisibleStageEvent = true,
            ),
        )
        assertTrue(
            shouldShowIndexingStageEvent(
                completedTrackEvent,
                allCommitted.copy(activeTrackOrdinal = 11),
            ),
        )
    }

    @Test
    fun durableSummaryNeverCountsAttentionAsSuccessfulIndexingProgress() {
        val zero = V2DurableStageCounter(0, 0, 0)
        val progress = V2IndexingProgressSnapshot(
            resolvedTracks = 28,
            succeededTracks = 0,
            blockedTracks = 28,
            skippedTracks = 0,
            totalTracks = 226,
            tracksWithMertFeatures = 39,
            tracksWithClampVectors = 0,
            mertWindows = zero,
            clampSegments = zero,
            databaseCommits = zero,
            activation = zero,
            activeTrackOrdinal = null,
            activeStage = null,
        )

        assertEquals(
            "0 of 226 embeddings saved \u00b7 28 need attention",
            formatDurableTrackCounts(IndexingJobState.PAUSED, progress),
        )
        assertEquals(
            "Saved audio-analysis checkpoints: 39 of 226",
            formatDurableStageTrackCounts(IndexingJobState.PAUSED, progress),
        )
    }

    @Test
    fun activeProgressNamesItsStageAndUsesOnlyThatStagesDenominator() {
        val event = V2IndexingExecutorEvent(
            jobId = "job",
            workId = "track",
            trackOrdinal = 3,
            trackTitle = "Artist - Title",
            stage = V2MeasuredWorkStage.MERT_WINDOWS,
            completedUnits = 2,
            totalUnits = 8,
            detail = "MERT window 2/8",
        )

        val evidence = requireNotNull(indexingStageEvidence(event))
        assertEquals("Current track · MERT window 2/8 · 25%", evidence.text)
        assertEquals(IndexingStageScope.CURRENT_TRACK, evidence.scope)
        assertEquals(0.25f, evidence.fraction)
        assertTrue(evidence.text.contains("window", ignoreCase = true))
    }

    @Test
    fun libraryWideProgressCannotBeMistakenForOneTrack() {
        val event = V2IndexingExecutorEvent(
            jobId = "job",
            workId = null,
            trackOrdinal = null,
            trackTitle = null,
            stage = V2MeasuredWorkStage.GRAPH_SIMILARITY_DOT_PRODUCTS,
            completedUnits = 25,
            totalUnits = 100,
            detail = "raw graph dot products",
        )

        val evidence = requireNotNull(indexingStageEvidence(event))
        assertEquals(
            "Whole library · raw graph dot products · 25%",
            evidence.text,
        )
        assertEquals(IndexingStageScope.WHOLE_LIBRARY, evidence.scope)
    }

    @Test
    fun foregroundNotificationUsesNeutralTitleAndOnlyStageLocalEvidence() {
        val event = V2IndexingExecutorEvent(
            jobId = "job",
            workId = "track",
            trackOrdinal = 3,
            trackTitle = "Artist - Title",
            stage = V2MeasuredWorkStage.MERT_WINDOWS,
            completedUnits = 2,
            totalUnits = 8,
            detail = "raw MERT window 2/8",
        )

        val evidence = indexingNotificationEvidence(IndexingJobState.RUNNING, event)

        assertEquals("On-device indexing", evidence.title)
        assertEquals("Current track · raw MERT window 2/8 · 25%", evidence.text)
        assertEquals(0.25f, evidence.fraction)
        assertTrue(evidence.text.contains("2/8"))
        assertTrue(evidence.text.contains("raw"))
        assertFalse(evidence.text.contains("tracks"))
    }

    @Test
    fun wholeRequestPreflightFailureUsesTypedCopyInsteadOfDiagnostic() {
        val rawDiagnostic = "provider cursor omitted requested columns: folder_files.duration"
        val text = preflightStatusEvidenceText(
            state = V2IndexingPreflightIntentState.FAILED,
            failureCode = V2IndexingPreflightFailureCode.PROVIDER_SNAPSHOT_INVALID,
            progressMessage = rawDiagnostic,
        )

        assertEquals("The current Poweramp library could not be read completely", text)
        assertFalse(text.contains("cursor", ignoreCase = true))
        assertFalse(text.contains("folder_files"))
    }

    @Test
    fun runningPreflightExplainsTheCurrentPreparationStage() {
        val text = preflightStatusEvidenceText(
            state = V2IndexingPreflightIntentState.PLANNING,
            failureCode = null,
            progressMessage = "Hashing each selected audio file once",
        )

        assertEquals("Hashing each selected audio file once", text)
    }

    @Test
    fun interruptedPreflightConsistentlyAsksToTryPreparationAgain() {
        val text = preflightStatusEvidenceText(
            state = V2IndexingPreflightIntentState.INTERRUPTED,
            failureCode = null,
            progressMessage = "Preflight paused by Android; resume when ready",
        )

        assertEquals("Your selection is saved. Try preparation again when ready", text)
        assertFalse(text.contains("resume", ignoreCase = true))
    }

    @Test
    fun onlyKnownStoppedReasonsBecomeVisible() {
        assertEquals(
            "Android's processing time limit paused indexing. Completed work is saved.",
            indexingStoppedReasonEvidence(
                IndexingJobState.PAUSED,
                IndexingService.MEDIA_PROCESSING_TIMEOUT_REASON,
            ),
        )
        assertEquals(
            "Indexing was interrupted. Completed work is saved.",
            indexingStoppedReasonEvidence(IndexingJobState.INTERRUPTED, "executor interrupted"),
        )
        assertEquals(
            "Completed work is saved and ready to resume.",
            indexingStoppedReasonEvidence(
                IndexingJobState.READY_TO_RESUME,
                "ready to resume execution",
            ),
        )
        assertNull(
            indexingStoppedReasonEvidence(
                IndexingJobState.RUNNING,
                IndexingService.MEDIA_PROCESSING_TIMEOUT_REASON,
            ),
        )
        assertNull(
            indexingStoppedReasonEvidence(
                IndexingJobState.PAUSED,
                "internal database path /data/user/0/private.sqlite",
            ),
        )
    }

    @Test
    fun activationCopyNamesTheUserOutcome() {
        assertEquals(
            "Restoring the saved music-index publication checkpoint",
            indexingNotificationEvidence(IndexingJobState.ACTIVATING, event = null).text,
        )
        assertEquals(
            "Updating music index",
            V2IndexingServiceStateMapper.stageLabel(V2MeasuredWorkStage.ACTIVATION_TRACKS),
        )
        val active = indexingStageEvidence(
            V2IndexingExecutorEvent(
                jobId = "job",
                workId = null,
                trackOrdinal = null,
                trackTitle = null,
                stage = V2MeasuredWorkStage.ACTIVATION_TRACKS,
                completedUnits = 0L,
                totalUnits = 80_325L,
                detail = "Publishing the verified index generation",
            ),
        )
        assertEquals(
            "Whole library · Publishing the verified index generation",
            active?.text,
        )
        assertNull(active?.fraction)
    }

    @Test
    fun partialEtaLeadsWithMeasuredWorkAndOneTruthfulCaveat() {
        val text = formatIndexingEtaEvidence(
            eta = V2StageAwareEtaEstimate(
                remainingMs = 8 * 60_000L,
                lowerBoundMs = 6 * 60_000L,
                upperBoundMs = 9 * 60_000L,
                calibratingStages = emptySet(),
            ),
            coverage = V2IndexingEtaCoverageSnapshot(
                scope = V2IndexingEtaScope.MEASURED_STAGES_ONLY,
                measuredRemainingStages = setOf(V2MeasuredWorkStage.MERT_WINDOWS),
                omittedRemainingWork = setOf(
                    V2UnmeasuredIndexingWork.VALIDATION_AND_FINAL_PUBLICATION,
                ),
            ),
        )

        assertEquals(
            "At current pace: 6 min to 9 min of measured work left.\n" +
                "Not included in this ETA: remaining validation and final " +
                "music-index publication.",
            text,
        )
    }

    @Test
    fun etaCalibrationExplainsWhenAnEstimateWillAppear() {
        val text = formatIndexingEtaEvidence(
            eta = V2StageAwareEtaEstimate(
                remainingMs = null,
                lowerBoundMs = null,
                upperBoundMs = null,
                calibratingStages = setOf(V2MeasuredWorkStage.DATABASE_COMMITS),
            ),
            coverage = V2IndexingEtaCoverageSnapshot(
                scope = V2IndexingEtaScope.MEASURED_STAGES_ONLY,
                measuredRemainingStages = setOf(V2MeasuredWorkStage.DATABASE_COMMITS),
                omittedRemainingWork = emptySet(),
            ),
        )

        assertEquals(
            "ETA unavailable until timing samples exist for: Save track embedding.",
            text,
        )
        assertTrue(text.orEmpty().contains("Save track embedding"))
    }

    @Test
    fun crossProfileEtaIsClearlyAnEarlyWideEstimate() {
        val text = formatIndexingEtaEvidence(
            eta = V2StageAwareEtaEstimate(
                remainingMs = 12 * 60_000L,
                lowerBoundMs = 6 * 60_000L,
                upperBoundMs = 20 * 60_000L,
                calibratingStages = setOf(
                    V2MeasuredWorkStage.CLAMP_SEGMENTS,
                    V2MeasuredWorkStage.DATABASE_COMMITS,
                ),
            ),
            coverage = V2IndexingEtaCoverageSnapshot(
                scope = V2IndexingEtaScope.WHOLE_JOB,
                measuredRemainingStages = setOf(
                    V2MeasuredWorkStage.CLAMP_SEGMENTS,
                    V2MeasuredWorkStage.DATABASE_COMMITS,
                ),
                omittedRemainingWork = emptySet(),
            ),
        )

        assertEquals(
            "Early estimate at current pace: 6 min to 20 min remaining.\n" +
                "Still measuring: Create music embedding, Save track embedding.",
            text,
        )
    }

    @Test
    fun etaIsVisibleOnlyWhileActiveWorkCanAdvance() {
        assertTrue(shouldShowIndexingEta(IndexingJobState.RUNNING))
        assertTrue(shouldShowIndexingEta(IndexingJobState.PAUSE_REQUESTED))
        assertTrue(shouldShowIndexingEta(IndexingJobState.ACTIVATING))
        assertFalse(shouldShowIndexingEta(IndexingJobState.PLANNED))
        assertFalse(shouldShowIndexingEta(IndexingJobState.PAUSED))
        assertFalse(shouldShowIndexingEta(IndexingJobState.WAITING_FOR_INPUT))
        assertFalse(shouldShowIndexingEta(IndexingJobState.INTERRUPTED))
        assertFalse(shouldShowIndexingEta(IndexingJobState.READY_TO_RESUME))
        assertFalse(shouldShowIndexingEta(IndexingJobState.CANCELLING))
        assertFalse(shouldShowIndexingEta(IndexingJobState.CANCELLED))
        assertFalse(shouldShowIndexingEta(IndexingJobState.COMPLETE))
    }

    @Test
    fun wholeJobEtaCanTruthfullySayRemaining() {
        val text = formatIndexingEtaEvidence(
            eta = V2StageAwareEtaEstimate(
                remainingMs = 8 * 60_000L,
                lowerBoundMs = 8 * 60_000L,
                upperBoundMs = 8 * 60_000L,
                calibratingStages = emptySet(),
            ),
            coverage = V2IndexingEtaCoverageSnapshot(
                scope = V2IndexingEtaScope.WHOLE_JOB,
                measuredRemainingStages = setOf(V2MeasuredWorkStage.MERT_WINDOWS),
                omittedRemainingWork = emptySet(),
            ),
        )

        assertEquals("At current pace: about 8 min remaining", text)
    }

    @Test
    fun cleanupFailureRetainsASeparateRetryableState() {
        val failure: DatabaseCleanupScanState =
            DatabaseCleanupScanState.Failed("Provider cursor stopped early")

        assertTrue(failure is DatabaseCleanupScanState.Failed)
        assertEquals(
            "Provider cursor stopped early",
            (failure as DatabaseCleanupScanState.Failed).message,
        )
    }

    @Test
    fun exportAndPlanningAdmissionAreMutuallyExclusiveAndSingleFlight() {
        val admission = IndexingUiOperationAdmission()
        assertTrue(admission.tryAcquire(IndexingUiOperation.EXPORT))
        assertFalse(admission.tryAcquire(IndexingUiOperation.EXPORT))
        assertFalse(admission.tryAcquire(IndexingUiOperation.JOB_PLANNING))
        assertFalse(admission.release(IndexingUiOperation.JOB_PLANNING))
        assertTrue(admission.release(IndexingUiOperation.EXPORT))
        assertTrue(admission.tryAcquire(IndexingUiOperation.JOB_PLANNING))
        assertTrue(admission.release(IndexingUiOperation.JOB_PLANNING))
    }
}
