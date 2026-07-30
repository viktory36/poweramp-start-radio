package com.powerampstartradio.indexing.v2

data class TrackFailurePolicy(
    val category: TrackFailureCategory,
    val initialDisposition: FailureDisposition,
    val retryTrigger: RetryTrigger,
    val maxAutomaticRetries: Int?,
    val exhaustedTrigger: RetryTrigger,
)

data class V2ActivationTrackFailure(
    val workId: String,
    val code: TrackFailureCode,
    val diagnostic: String,
)

sealed interface V2TrackPreflightOutcome {
    val workId: String
    val startedAtEpochMs: Long
    val finishedAtEpochMs: Long

    data class Verified(
        override val workId: String,
        override val startedAtEpochMs: Long,
        override val finishedAtEpochMs: Long,
    ) : V2TrackPreflightOutcome

    data class Failed(
        override val workId: String,
        override val startedAtEpochMs: Long,
        override val finishedAtEpochMs: Long,
        val failure: V2ClassifiedIndexingFailure,
    ) : V2TrackPreflightOutcome
}

/** One exhaustive mapping keeps failure labels and automatic retry behavior truthful. */
object V2TrackFailurePolicies {
    fun forCode(code: TrackFailureCode): TrackFailurePolicy = when (code) {
        TrackFailureCode.SOURCE_MISSING,
        TrackFailureCode.STORAGE_UNMOUNTED,
        TrackFailureCode.SOURCE_UNREADABLE,
        TrackFailureCode.POWERAMP_PROVIDER_UNAVAILABLE,
        -> retryable(TrackFailureCategory.SOURCE_UNAVAILABLE, RetryTrigger.SOURCE_AVAILABLE)

        TrackFailureCode.ANDROID_AUDIO_PERMISSION_DENIED,
        TrackFailureCode.POWERAMP_PERMISSION_DENIED,
        -> retryable(TrackFailureCategory.PERMISSION, RetryTrigger.PERMISSION_GRANTED)

        TrackFailureCode.INVALID_LOGICAL_SPAN,
        TrackFailureCode.CUE_SOURCE_IMAGE,
        TrackFailureCode.SOURCE_FINGERPRINT_CHANGED,
        TrackFailureCode.PROVIDER_SNAPSHOT_CHANGED,
        TrackFailureCode.CONTAINER_EOS_MISMATCH,
        TrackFailureCode.IMPORTED_ROW_AUTHORIZATION_CHANGED,
        -> blocked(TrackFailureCategory.SPAN_OR_IDENTITY, RetryTrigger.NEW_JOB_REQUIRED)

        TrackFailureCode.NO_AUDIO_STREAM,
        TrackFailureCode.UNSUPPORTED_CODEC_OR_CONTAINER,
        TrackFailureCode.DRM_PROTECTED,
        -> blocked(TrackFailureCategory.DECODE_UNSUPPORTED, RetryTrigger.DECODER_OR_APP_CHANGED)

        TrackFailureCode.CORRUPT_OR_TRUNCATED,
        TrackFailureCode.DECODER_ERROR,
        -> bounded(
            TrackFailureCategory.DECODE_CORRUPT,
            RetryTrigger.IMMEDIATE,
            automaticRetries = 1,
            exhaustedTrigger = RetryTrigger.DECODER_OR_APP_CHANGED,
        )

        TrackFailureCode.BELOW_MINIMUM_DURATION ->
            blocked(TrackFailureCategory.TOO_SHORT, RetryTrigger.SOURCE_OR_LIBRARY_CHANGED)

        TrackFailureCode.OUT_OF_MEMORY,
        TrackFailureCode.THERMAL_SHUTDOWN,
        TrackFailureCode.STAGE_TIMEOUT,
        -> retryable(TrackFailureCategory.RESOURCE_TRANSIENT, RetryTrigger.RESOURCE_RECOVERED)

        TrackFailureCode.PROCESS_INTERRUPTED -> bounded(
            TrackFailureCategory.RESOURCE_TRANSIENT,
            RetryTrigger.PROCESS_RESTART,
            automaticRetries = 2,
            exhaustedTrigger = RetryTrigger.USER_REQUEST,
        )

        TrackFailureCode.MODEL_LOAD_FAILED,
        TrackFailureCode.INFERENCE_FAILED,
        -> bounded(
            TrackFailureCategory.INFERENCE_OR_ARTIFACT,
            RetryTrigger.IMMEDIATE,
            automaticRetries = 1,
            exhaustedTrigger = RetryTrigger.DECODER_OR_APP_CHANGED,
        )

        TrackFailureCode.INVALID_MODEL_OUTPUT ->
            blocked(TrackFailureCategory.INFERENCE_OR_ARTIFACT, RetryTrigger.DECODER_OR_APP_CHANGED)

        TrackFailureCode.PARTIAL_ARTIFACT,
        TrackFailureCode.ARTIFACT_CHECKSUM_MISMATCH,
        -> bounded(
            TrackFailureCategory.INFERENCE_OR_ARTIFACT,
            RetryTrigger.IMMEDIATE,
            automaticRetries = 2,
            exhaustedTrigger = RetryTrigger.DECODER_OR_APP_CHANGED,
        )

        TrackFailureCode.STORAGE_FULL,
        TrackFailureCode.DATABASE_BUSY,
        TrackFailureCode.COMMIT_FAILED,
        -> retryable(TrackFailureCategory.STORAGE_OR_DATABASE, RetryTrigger.STORAGE_RECOVERED)

        TrackFailureCode.DATABASE_GENERATION_CHANGED -> retryable(
            TrackFailureCategory.STORAGE_OR_DATABASE,
            RetryTrigger.SOURCE_OR_LIBRARY_CHANGED,
        )

        TrackFailureCode.UNKNOWN_TRANSIENT -> bounded(
            TrackFailureCategory.RESOURCE_TRANSIENT,
            RetryTrigger.IMMEDIATE,
            automaticRetries = 1,
            exhaustedTrigger = RetryTrigger.USER_REQUEST,
        )

        TrackFailureCode.UNKNOWN_BLOCKED ->
            blocked(TrackFailureCategory.RESOURCE_TRANSIENT, RetryTrigger.USER_REQUEST)
    }

    private fun retryable(
        category: TrackFailureCategory,
        trigger: RetryTrigger,
    ) = TrackFailurePolicy(
        category = category,
        initialDisposition = FailureDisposition.RETRYABLE,
        retryTrigger = trigger,
        maxAutomaticRetries = null,
        exhaustedTrigger = trigger,
    )

    private fun bounded(
        category: TrackFailureCategory,
        trigger: RetryTrigger,
        automaticRetries: Int,
        exhaustedTrigger: RetryTrigger,
    ) = TrackFailurePolicy(
        category = category,
        initialDisposition = FailureDisposition.RETRYABLE,
        retryTrigger = trigger,
        maxAutomaticRetries = automaticRetries,
        exhaustedTrigger = exhaustedTrigger,
    )

    private fun blocked(
        category: TrackFailureCategory,
        trigger: RetryTrigger,
    ) = TrackFailurePolicy(
        category = category,
        initialDisposition = FailureDisposition.BLOCKED,
        retryTrigger = trigger,
        maxAutomaticRetries = 0,
        exhaustedTrigger = trigger,
    )
}

/** Pure transition API. Callers persist each returned ledger before doing the next action. */
object V2IndexingLedgerStateMachine {
    fun startJob(ledger: IndexingJobLedger, nowEpochMs: Long): IndexingJobLedger =
        transitionJob(ledger, IndexingJobState.PLANNED, IndexingJobState.RUNNING, nowEpochMs, "started")

    fun requestPause(ledger: IndexingJobLedger, nowEpochMs: Long): IndexingJobLedger =
        transitionJob(
            ledger,
            IndexingJobState.RUNNING,
            IndexingJobState.PAUSE_REQUESTED,
            nowEpochMs,
            "pause requested",
        )

    fun requestMediaProcessingTimeoutPause(
        ledger: IndexingJobLedger,
        nowEpochMs: Long,
        reason: String,
    ): IndexingJobLedger = transitionJob(
        ledger,
        IndexingJobState.RUNNING,
        IndexingJobState.PAUSE_REQUESTED,
        nowEpochMs,
        reason,
    )

    fun finishPause(
        ledger: IndexingJobLedger,
        nowEpochMs: Long,
        reason: String = "paused at verified checkpoint",
    ): IndexingJobLedger {
        requireState(ledger, IndexingJobState.PAUSE_REQUESTED)
        requireNoActiveTrack(ledger, "pause")
        return updateJob(
            ledger = ledger,
            state = IndexingJobState.PAUSED,
            recoveryPhase = null,
            nowEpochMs = nowEpochMs,
            reason = reason,
        )
    }

    fun prepareResume(ledger: IndexingJobLedger, nowEpochMs: Long): IndexingJobLedger {
        val phase = when (ledger.state) {
            IndexingJobState.PAUSED,
            IndexingJobState.WAITING_FOR_INPUT,
            -> RecoveryPhase.EXECUTION
            IndexingJobState.INTERRUPTED -> checkNotNull(ledger.recoveryPhase)
            else -> throw invalidTransition("prepare resume", ledger.state)
        }
        return updateJob(
            ledger = ledger,
            state = IndexingJobState.READY_TO_RESUME,
            recoveryPhase = phase,
            nowEpochMs = nowEpochMs,
            reason = "ready to resume ${phase.name.lowercase()}",
        )
    }

    fun resume(ledger: IndexingJobLedger, nowEpochMs: Long): IndexingJobLedger {
        requireState(ledger, IndexingJobState.READY_TO_RESUME)
        val target = when (ledger.recoveryPhase) {
            RecoveryPhase.EXECUTION -> IndexingJobState.RUNNING
            RecoveryPhase.ACTIVATION -> IndexingJobState.ACTIVATING
            null -> throw InvalidIndexingLedgerException("resume state has no recovery phase")
        }
        return updateJob(
            ledger = ledger,
            state = target,
            recoveryPhase = null,
            nowEpochMs = nowEpochMs,
            reason = "resumed",
        )
    }

    fun interruptActivationForMediaProcessingTimeout(
        ledger: IndexingJobLedger,
        nowEpochMs: Long,
        reason: String,
    ): IndexingJobLedger {
        requireState(ledger, IndexingJobState.ACTIVATING)
        return updateJob(
            ledger = ledger,
            state = IndexingJobState.INTERRUPTED,
            recoveryPhase = RecoveryPhase.ACTIVATION,
            nowEpochMs = nowEpochMs,
            reason = reason,
        )
    }

    fun checkpointForMediaProcessingTimeout(
        ledger: IndexingJobLedger,
        nowEpochMs: Long,
        reason: String,
    ): IndexingJobLedger {
        if (ledger.state == IndexingJobState.ACTIVATING) {
            return interruptActivationForMediaProcessingTimeout(ledger, nowEpochMs, reason)
        }
        if (ledger.state != IndexingJobState.RUNNING &&
            ledger.state != IndexingJobState.PAUSE_REQUESTED
        ) {
            throw invalidTransition("checkpoint media-processing timeout", ledger.state)
        }
        val effectiveNow = maxOf(nowEpochMs, ledger.updatedAtEpochMs)
        val checkpointed = ledger.tracks.map { track ->
            if (!track.state.isActiveStage()) track else track.copy(
                state = track.checkpoint.asTrackState(),
                currentAttemptNumber = null,
                activeFailureId = null,
                updatedAtEpochMs = effectiveNow,
            )
        }
        return updateJob(
            ledger = ledger.copy(tracks = checkpointed, updatedAtEpochMs = effectiveNow),
            state = IndexingJobState.PAUSED,
            recoveryPhase = null,
            nowEpochMs = effectiveNow,
            reason = reason,
        )
    }

    /** Profile is mutable scheduling evidence and never changes the immutable job/work identity. */
    fun changeExecutionProfile(
        ledger: IndexingJobLedger,
        profile: V2IndexingExecutionProfile,
        nowEpochMs: Long,
    ): IndexingJobLedger {
        if (ledger.state == IndexingJobState.CANCELLED ||
            ledger.state == IndexingJobState.COMPLETE
        ) {
            throw invalidTransition("change execution profile", ledger.state)
        }
        if (ledger.executionProfile == profile) return ledger
        return bump(
            ledger.copy(
                executionProfile = profile,
                updatedAtEpochMs = maxOf(nowEpochMs, ledger.updatedAtEpochMs),
                stateReason = "execution profile changed to ${profile.name.lowercase()}",
            ),
        ).also(V2IndexingLedgerValidator::requireValid)
    }

    fun requestCancel(ledger: IndexingJobLedger, nowEpochMs: Long): IndexingJobLedger {
        if (ledger.state == IndexingJobState.CANCELLING) return ledger
        if (ledger.state == IndexingJobState.ACTIVATING ||
            ledger.state == IndexingJobState.CANCELLED || ledger.state == IndexingJobState.COMPLETE
        ) {
            throw invalidTransition("request cancel", ledger.state)
        }
        return updateJob(
            ledger = ledger,
            state = IndexingJobState.CANCELLING,
            recoveryPhase = null,
            nowEpochMs = nowEpochMs,
            reason = "cancel requested",
        )
    }

    fun finishCancel(ledger: IndexingJobLedger, nowEpochMs: Long): IndexingJobLedger {
        requireState(ledger, IndexingJobState.CANCELLING)
        requireNoActiveTrack(ledger, "cancel")
        return updateJob(
            ledger = ledger,
            state = IndexingJobState.CANCELLED,
            recoveryPhase = null,
            nowEpochMs = nowEpochMs,
            reason = "cancelled after uncommitted artifact cleanup",
        )
    }

    /** Drop only unverified in-flight work; previously verified artifacts remain checkpoints. */
    fun checkpointCancellation(
        ledger: IndexingJobLedger,
        nowEpochMs: Long,
    ): IndexingJobLedger {
        requireState(ledger, IndexingJobState.CANCELLING)
        val activeTracks = ledger.tracks.filter { it.state.isActiveStage() }
        if (activeTracks.isEmpty()) return ledger
        val effectiveNow = maxOf(nowEpochMs, ledger.updatedAtEpochMs)
        val rolledBack = ledger.tracks.map { track ->
            if (!track.state.isActiveStage()) track else track.copy(
                state = track.checkpoint.asTrackState(),
                currentAttemptNumber = null,
                activeFailureId = null,
                stageProgress = null,
                updatedAtEpochMs = effectiveNow,
            )
        }
        val updated = bump(
            ledger.copy(
                tracks = rolledBack,
                updatedAtEpochMs = effectiveNow,
                stateReason = "cancellation checkpointed",
            ),
        )
        V2IndexingLedgerValidator.requireValid(updated)
        return updated
    }

    fun beginActivation(ledger: IndexingJobLedger, nowEpochMs: Long): IndexingJobLedger {
        requireState(ledger, IndexingJobState.RUNNING)
        requireNoActiveTrack(ledger, "activate")
        if (ledger.tracks.any { !it.state.isResolvedForActivation() }) {
            throw InvalidIndexingLedgerException("cannot activate while tracks remain unresolved")
        }
        return updateJob(
            ledger = ledger,
            state = IndexingJobState.ACTIVATING,
            recoveryPhase = null,
            nowEpochMs = nowEpochMs,
            reason = "activating committed generation",
        )
    }

    fun waitForInput(ledger: IndexingJobLedger, nowEpochMs: Long): IndexingJobLedger {
        requireState(ledger, IndexingJobState.RUNNING)
        requireNoActiveTrack(ledger, "wait for input")
        if (ledger.tracks.none {
                it.state == IndexingTrackState.RETRYABLE_FAILURE ||
                    it.state == IndexingTrackState.BLOCKED_FAILURE
            }
        ) {
            throw InvalidIndexingLedgerException("job has no failed track waiting for input")
        }
        if (ledger.tracks.any {
                    it.state != IndexingTrackState.RETRYABLE_FAILURE &&
                    it.state != IndexingTrackState.BLOCKED_FAILURE &&
                    it.state != IndexingTrackState.MERT_COMPLETE &&
                    !it.state.isResolvedForActivation()
            }
        ) {
            throw InvalidIndexingLedgerException("job still has runnable track work")
        }
        return updateJob(
            ledger = ledger,
            state = IndexingJobState.WAITING_FOR_INPUT,
            recoveryPhase = null,
            nowEpochMs = nowEpochMs,
            reason = "waiting for a retry condition or user decision",
        )
    }

    /**
     * Activation has no active per-track stage, but a stale destructive authorization must still
     * become durable, user-actionable evidence rather than an endlessly resumable interruption.
     */
    fun blockImportedRowActivationForNewJob(
        ledger: IndexingJobLedger,
        workId: String,
        diagnostic: String,
        nowEpochMs: Long,
    ): IndexingJobLedger {
        if (ledger.state != IndexingJobState.RUNNING &&
            ledger.state != IndexingJobState.ACTIVATING
        ) throw invalidTransition("block imported-row activation", ledger.state)
        requireNoActiveTrack(ledger, "block imported-row activation")
        val descriptor = descriptorFor(ledger, workId)
        if (descriptor.finalizedAudioSpan.kind != V2ResolvedAudioSpanKind.LOGICAL_CUE) {
            throw InvalidIndexingLedgerException(
                "imported-row activation failure is not attached to logical CUE work",
            )
        }
        return blockActivationForNewJob(
            ledger = ledger,
            failures = listOf(
                V2ActivationTrackFailure(
                    workId = workId,
                    code = TrackFailureCode.IMPORTED_ROW_AUTHORIZATION_CHANGED,
                    diagnostic = diagnostic,
                ),
            ),
            nowEpochMs = nowEpochMs,
            reason = "imported CUE activation requires a new preflight",
            requireCommitted = false,
        )
    }

    /** Records immutable source/provider drift at the final generation-publication boundary. */
    fun blockCommittedActivationForNewJob(
        ledger: IndexingJobLedger,
        failures: List<V2ActivationTrackFailure>,
        nowEpochMs: Long,
        reason: String = "activation identity changed; a new preflight is required",
    ): IndexingJobLedger = blockActivationForNewJob(
        ledger = ledger,
        failures = failures,
        nowEpochMs = nowEpochMs,
        reason = reason,
        requireCommitted = true,
    )

    private fun blockActivationForNewJob(
        ledger: IndexingJobLedger,
        failures: List<V2ActivationTrackFailure>,
        nowEpochMs: Long,
        reason: String,
        requireCommitted: Boolean,
    ): IndexingJobLedger {
        if (ledger.state != IndexingJobState.RUNNING &&
            ledger.state != IndexingJobState.ACTIVATING
        ) throw invalidTransition("block committed activation", ledger.state)
        requireNoActiveTrack(ledger, "block committed activation")
        require(failures.isNotEmpty()) { "activation failures are empty" }
        val failuresByWorkId = failures.associateBy { it.workId }
        require(failuresByWorkId.size == failures.size) {
            "activation failures repeat work IDs"
        }
        failures.forEach { failure ->
            require(failure.code in setOf(
                    TrackFailureCode.SOURCE_FINGERPRINT_CHANGED,
                    TrackFailureCode.PROVIDER_SNAPSHOT_CHANGED,
                    TrackFailureCode.IMPORTED_ROW_AUTHORIZATION_CHANGED,
                )
            ) { "${failure.code} is not an immutable activation identity failure" }
            descriptorFor(ledger, failure.workId)
        }
        val effectiveNow = maxOf(nowEpochMs, ledger.updatedAtEpochMs)
        val tracks = ledger.tracks.map { track ->
            val activationFailure = failuresByWorkId[track.workId] ?: return@map track
            if ((requireCommitted && track.state != IndexingTrackState.COMMITTED) ||
                (!requireCommitted && !track.state.isResolvedForActivation())
            ) {
                throw InvalidIndexingLedgerException(
                    "activation identity failure is attached to ineligible work",
                )
            }
            val descriptor = descriptorFor(ledger, track.workId)
            val code = activationFailure.code
            val stage = IndexingStage.DATABASE_ACTIVATION
            val policy = V2TrackFailurePolicies.forCode(code)
            check(policy.initialDisposition == FailureDisposition.BLOCKED &&
                policy.retryTrigger == RetryTrigger.NEW_JOB_REQUIRED
            ) { "activation identity failure policy changed" }
            val nextAttempt = track.attemptCount + 1
            val failureId = V2IndexingLedgerIds.failureId(
                workId = track.workId,
                code = code,
                stage = stage,
                sourceFingerprint = descriptor.sourceFingerprint,
                embeddingSpecId = ledger.jobSpec.embeddingSpec.specId,
                appBuildId = ledger.jobSpec.runtimeFingerprint.appBuildId,
            )
            val existing = track.failures.singleOrNull { it.failureId == failureId }
            val failure = TrackFailureAggregate(
                failureId = failureId,
                code = code,
                category = policy.category,
                stage = stage,
                disposition = FailureDisposition.BLOCKED,
                retryTrigger = RetryTrigger.NEW_JOB_REQUIRED,
                resumeFrom = track.checkpoint,
                diagnostic = activationFailure.diagnostic.trim()
                    .ifBlank { code.name }
                    .take(2_048),
                firstOccurredAtEpochMs = existing?.firstOccurredAtEpochMs ?: effectiveNow,
                lastOccurredAtEpochMs = effectiveNow,
                firstAttemptNumber = existing?.firstAttemptNumber ?: nextAttempt,
                lastAttemptNumber = nextAttempt,
                occurrences = (existing?.occurrences ?: 0) + 1,
                sourceFingerprint = descriptor.sourceFingerprint,
                embeddingSpecId = ledger.jobSpec.embeddingSpec.specId,
                appBuildId = ledger.jobSpec.runtimeFingerprint.appBuildId,
            )
            track.copy(
                state = IndexingTrackState.BLOCKED_FAILURE,
                attemptCount = nextAttempt,
                currentAttemptNumber = null,
                activeFailureId = failureId,
                stageProgress = null,
                failures = track.failures.filterNot { it.failureId == failureId } + failure,
                updatedAtEpochMs = effectiveNow,
            )
        }
        return updateJob(
            ledger = ledger.copy(tracks = tracks, updatedAtEpochMs = effectiveNow),
            state = IndexingJobState.WAITING_FOR_INPUT,
            recoveryPhase = null,
            nowEpochMs = effectiveNow,
            reason = reason,
        )
    }

    fun completeJob(
        ledger: IndexingJobLedger,
        evidence: ActivatedGenerationEvidence,
        nowEpochMs: Long,
    ): IndexingJobLedger {
        requireState(ledger, IndexingJobState.ACTIVATING)
        if (evidence.jobSpecId != ledger.jobSpec.specId ||
            evidence.receiptEmbeddingSpecId != ledger.jobSpec.embeddingSpec.specId ||
            evidence.baseGenerationId != ledger.jobSpec.baseGenerationId ||
            evidence.rebuildDerivedIndexes != ledger.jobSpec.rebuildDerivedIndexes
        ) {
            throw InvalidIndexingLedgerException("activation evidence does not bind the job spec")
        }
        V2IndexingLedgerValidator.requireValid(ledger)
        val completed = bump(
            ledger.copy(
                state = IndexingJobState.COMPLETE,
                recoveryPhase = null,
                activationEvidence = evidence,
                updatedAtEpochMs = maxOf(nowEpochMs, ledger.updatedAtEpochMs),
                stateReason = "verified embeddings activated",
            ),
        )
        V2IndexingLedgerValidator.requireValid(completed)
        return completed
    }

    fun beginNextTrackStage(
        ledger: IndexingJobLedger,
        workId: String,
        nowEpochMs: Long,
    ): IndexingJobLedger {
        requireState(ledger, IndexingJobState.RUNNING)
        if (ledger.tracks.any { it.workId != workId && it.state.isActiveStage() }) {
            throw InvalidIndexingLedgerException("another track already owns the executor")
        }
        return updateTrack(ledger, workId, nowEpochMs) { track, _ ->
            val target = when (track.state) {
                IndexingTrackState.QUEUED -> IndexingTrackState.PREFLIGHTING
                IndexingTrackState.PREFLIGHTED -> IndexingTrackState.DECODING
                IndexingTrackState.MERT_COMPLETE -> IndexingTrackState.CLAMPING
                IndexingTrackState.CLAMP_COMPLETE -> IndexingTrackState.COMMITTING
                else -> throw invalidTrackTransition("begin next stage", track.state)
            }
            if (track.stageProgress != null && track.stageProgress.stage != target.stageOrNull()) {
                throw InvalidIndexingLedgerException("saved progress belongs to another stage")
            }
            val nextAttempt = track.currentAttemptNumber ?: (track.attemptCount + 1)
            track.copy(
                state = target,
                attemptCount = maxOf(track.attemptCount, nextAttempt),
                currentAttemptNumber = nextAttempt,
                activeFailureId = null,
            )
        }
    }

    /**
     * Makes already fingerprinted plan entries runnable without rereading every source first.
     * Exact source verification remains part of the per-track decode boundary.
     */
    fun admitPlannedTracksForExecution(
        ledger: IndexingJobLedger,
        nowEpochMs: Long,
    ): IndexingJobLedger {
        requireState(ledger, IndexingJobState.RUNNING)
        requireNoActiveTrack(ledger, "admit planned tracks")
        if (ledger.tracks.none { it.state == IndexingTrackState.QUEUED }) return ledger
        val effectiveNow = maxOf(nowEpochMs, ledger.updatedAtEpochMs)
        val admitted = ledger.tracks.map { track ->
            if (track.state != IndexingTrackState.QUEUED) return@map track
            track.copy(
                state = IndexingTrackState.PREFLIGHTED,
                checkpoint = TrackCheckpoint.PREFLIGHTED,
                updatedAtEpochMs = effectiveNow,
            )
        }
        return bump(
            ledger.copy(
                tracks = admitted,
                updatedAtEpochMs = effectiveNow,
            ),
        ).also(V2IndexingLedgerValidator::requireValid)
    }

    /**
     * Commits side-effect-free source checks as one atomic ledger transition. Replaying the
     * existing transitions in memory keeps attempts and failure policy byte-for-byte equivalent
     * to the former per-track path; only the persistence revision is intentionally coalesced.
     */
    fun commitTrackPreflightBatch(
        ledger: IndexingJobLedger,
        outcomes: List<V2TrackPreflightOutcome>,
    ): IndexingJobLedger {
        requireState(ledger, IndexingJobState.RUNNING)
        if (outcomes.isEmpty()) {
            throw InvalidIndexingLedgerException("preflight batch is empty")
        }
        if (outcomes.map(V2TrackPreflightOutcome::workId).distinct().size != outcomes.size) {
            throw InvalidIndexingLedgerException("preflight batch repeats work IDs")
        }
        val initialRevision = ledger.revision
        var replayed = ledger
        outcomes.forEach { outcome ->
            replayed = beginNextTrackStage(
                replayed,
                outcome.workId,
                outcome.startedAtEpochMs,
            )
            replayed = when (outcome) {
                is V2TrackPreflightOutcome.Verified -> completeActiveTrackStage(
                    replayed,
                    outcome.workId,
                    artifact = null,
                    nowEpochMs = outcome.finishedAtEpochMs,
                )

                is V2TrackPreflightOutcome.Failed -> recordTrackFailure(
                    replayed,
                    outcome.workId,
                    outcome.failure.code,
                    IndexingStage.PREFLIGHT,
                    outcome.failure.diagnostic,
                    outcome.finishedAtEpochMs,
                )
            }
        }
        return replayed.copy(revision = Math.addExact(initialRevision, 1L)).also(
            V2IndexingLedgerValidator::requireValid,
        )
    }

    /** Persist a checksummed partial artifact without claiming the stage is complete. */
    fun checkpointActiveStageProgress(
        ledger: IndexingJobLedger,
        workId: String,
        progress: VerifiedStageProgress,
        nowEpochMs: Long,
    ): IndexingJobLedger {
        if (ledger.state != IndexingJobState.RUNNING &&
            ledger.state != IndexingJobState.PAUSE_REQUESTED
        ) {
            throw invalidTransition("checkpoint active stage", ledger.state)
        }
        return updateTrack(ledger, workId, nowEpochMs) { track, descriptor ->
            val activeStage = track.state.stageOrNull()
                ?: throw invalidTrackTransition("checkpoint active stage", track.state)
            if (activeStage != IndexingStage.DECODE_AND_MERT && activeStage != IndexingStage.CLAMP3) {
                throw InvalidIndexingLedgerException("$activeStage has no resumable partial artifact")
            }
            if (progress.stage != activeStage ||
                progress.embeddingSpecId != ledger.jobSpec.embeddingSpec.specId ||
                progress.sourceFingerprint != descriptor.sourceFingerprint
            ) {
                throw InvalidIndexingLedgerException("partial artifact does not belong to active stage")
            }
            val previous = track.stageProgress
            if (previous != null && progress.completedUnits <= previous.completedUnits) {
                throw InvalidIndexingLedgerException("partial progress must advance completed units")
            }
            track.copy(stageProgress = progress)
        }
    }

    /** Release an active stage for pause only after its latest durable progress was recorded. */
    fun suspendActiveStageForPause(
        ledger: IndexingJobLedger,
        workId: String,
        nowEpochMs: Long,
    ): IndexingJobLedger {
        requireState(ledger, IndexingJobState.PAUSE_REQUESTED)
        return updateTrack(ledger, workId, nowEpochMs) { track, _ ->
            if (!track.state.isActiveStage()) {
                throw invalidTrackTransition("suspend for pause", track.state)
            }
            track.copy(
                state = track.checkpoint.asTrackState(),
                currentAttemptNumber = null,
                activeFailureId = null,
            )
        }
    }

    fun completeActiveTrackStage(
        ledger: IndexingJobLedger,
        workId: String,
        artifact: VerifiedArtifact?,
        nowEpochMs: Long,
    ): IndexingJobLedger {
        if (ledger.state != IndexingJobState.RUNNING &&
            ledger.state != IndexingJobState.PAUSE_REQUESTED
        ) {
            throw invalidTransition("complete active stage", ledger.state)
        }
        return updateTrack(ledger, workId, nowEpochMs) { track, descriptor ->
            val (target, expectedKind) = when (track.state) {
                IndexingTrackState.PREFLIGHTING ->
                    IndexingTrackState.PREFLIGHTED to null
                IndexingTrackState.DECODING ->
                    IndexingTrackState.MERT_COMPLETE to VerifiedArtifactKind.MERT_FEATURES
                IndexingTrackState.CLAMPING ->
                    IndexingTrackState.CLAMP_COMPLETE to VerifiedArtifactKind.CLAMP_VECTOR
                IndexingTrackState.COMMITTING ->
                    IndexingTrackState.COMMITTED to VerifiedArtifactKind.DATABASE_COMMIT
                else -> throw invalidTrackTransition("complete active stage", track.state)
            }
            if (expectedKind == null && artifact != null) {
                throw InvalidIndexingLedgerException("preflight completion must not claim an artifact")
            }
            if (expectedKind != null && artifact?.kind != expectedKind) {
                throw InvalidIndexingLedgerException("$expectedKind evidence is required")
            }
            if (artifact != null) {
                if (artifact.embeddingSpecId != ledger.jobSpec.embeddingSpec.specId ||
                    artifact.sourceFingerprint != descriptor.sourceFingerprint
                ) {
                    throw InvalidIndexingLedgerException("artifact does not belong to this track spec")
                }
                if (track.verifiedArtifacts.any { it.kind == artifact.kind }) {
                    throw InvalidIndexingLedgerException("artifact kind was already verified")
                }
            }
            track.copy(
                state = target,
                checkpoint = target.fixedCheckpointOrNull()
                    ?: throw InvalidIndexingLedgerException("completed stage has no checkpoint"),
                currentAttemptNumber = if (target == IndexingTrackState.COMMITTED) null
                else track.currentAttemptNumber,
                stageProgress = null,
                verifiedArtifacts = if (artifact == null) track.verifiedArtifacts
                else track.verifiedArtifacts + artifact,
            )
        }
    }

    fun recordTrackFailure(
        ledger: IndexingJobLedger,
        workId: String,
        code: TrackFailureCode,
        stage: IndexingStage,
        diagnostic: String,
        nowEpochMs: Long,
        resumeFrom: TrackCheckpoint? = null,
    ): IndexingJobLedger {
        if (ledger.state != IndexingJobState.RUNNING &&
            ledger.state != IndexingJobState.PAUSE_REQUESTED
        ) {
            throw invalidTransition("record track failure", ledger.state)
        }
        return updateTrack(ledger, workId, nowEpochMs) { track, descriptor ->
            recordFailureInternal(
                ledger = ledger,
                track = track,
                descriptor = descriptor,
                code = code,
                stage = stage,
                diagnostic = diagnostic,
                nowEpochMs = maxOf(nowEpochMs, ledger.updatedAtEpochMs),
                resumeFrom = resumeFrom ?: track.checkpoint,
            )
        }
    }

    fun retryTrack(
        ledger: IndexingJobLedger,
        workId: String,
        trigger: RetryTrigger,
        nowEpochMs: Long,
    ): IndexingJobLedger {
        if (ledger.state !in setOf(
                IndexingJobState.RUNNING,
                IndexingJobState.PAUSED,
                IndexingJobState.WAITING_FOR_INPUT,
                IndexingJobState.INTERRUPTED,
                IndexingJobState.READY_TO_RESUME,
            )
        ) {
            throw invalidTransition("retry track", ledger.state)
        }
        return updateTrack(ledger, workId, nowEpochMs) { track, _ ->
            if (track.state != IndexingTrackState.RETRYABLE_FAILURE &&
                track.state != IndexingTrackState.BLOCKED_FAILURE
            ) {
                throw invalidTrackTransition("retry", track.state)
            }
            val failure = track.failures.singleOrNull { it.failureId == track.activeFailureId }
                ?: throw InvalidIndexingLedgerException("failed track has no active failure")
            val automaticAllowed = track.state == IndexingTrackState.RETRYABLE_FAILURE &&
                trigger == failure.retryTrigger
            val userAllowed = trigger == RetryTrigger.USER_REQUEST &&
                failure.retryTrigger != RetryTrigger.NEW_JOB_REQUIRED &&
                failure.retryTrigger != RetryTrigger.NEVER
            if (!automaticAllowed && !userAllowed) {
                throw InvalidIndexingLedgerException(
                    "${failure.code} requires ${failure.retryTrigger}, not $trigger",
                )
            }
            track.copy(
                state = failure.resumeFrom.asTrackState(),
                checkpoint = failure.resumeFrom,
                currentAttemptNumber = null,
                activeFailureId = null,
            )
        }
    }

    fun skipTrack(
        ledger: IndexingJobLedger,
        workId: String,
        nowEpochMs: Long,
    ): IndexingJobLedger {
        if (ledger.state in setOf(
                IndexingJobState.ACTIVATING,
                IndexingJobState.CANCELLING,
                IndexingJobState.CANCELLED,
                IndexingJobState.COMPLETE,
            )
        ) {
            throw invalidTransition("skip track", ledger.state)
        }
        return updateTrack(ledger, workId, nowEpochMs) { track, _ ->
            if (track.state.isActiveStage() || track.state == IndexingTrackState.COMMITTED) {
                throw invalidTrackTransition("skip", track.state)
            }
            track.copy(
                state = IndexingTrackState.SKIPPED_BY_USER,
                checkpoint = TrackCheckpoint.QUEUED,
                currentAttemptNumber = null,
                activeFailureId = null,
                stageProgress = null,
                verifiedArtifacts = emptyList(),
            )
        }
    }

    /**
     * Idempotent startup reconciliation. In-flight work becomes a typed retryable failure at
     * its last verified checkpoint; an acknowledged pause remains paused, and cancellation
     * intent is never converted back into runnable work.
     */
    fun reconcileAfterProcessRestart(
        ledger: IndexingJobLedger,
        nowEpochMs: Long,
        diagnostic: String = "executor process stopped before the stage checkpoint committed",
    ): RestartReconciliation {
        V2IndexingLedgerValidator.requireValid(ledger)
        val effectiveNow = maxOf(nowEpochMs, ledger.updatedAtEpochMs)
        val activeTracks = ledger.tracks.filter { it.state.isActiveStage() }
        return when (ledger.state) {
            IndexingJobState.RUNNING -> {
                val reconciledTracks = ledger.tracks.map { track ->
                    if (!track.state.isActiveStage()) track else {
                        val descriptor = descriptorFor(ledger, track.workId)
                        recordFailureInternal(
                            ledger = ledger,
                            track = track,
                            descriptor = descriptor,
                            code = TrackFailureCode.PROCESS_INTERRUPTED,
                            stage = track.state.stage(),
                            diagnostic = diagnostic,
                            nowEpochMs = effectiveNow,
                            resumeFrom = track.checkpoint,
                        )
                    }
                }
                val reconciled = updateJob(
                    ledger = ledger.copy(
                        tracks = reconciledTracks,
                        updatedAtEpochMs = effectiveNow,
                    ),
                    state = IndexingJobState.INTERRUPTED,
                    recoveryPhase = RecoveryPhase.EXECUTION,
                    nowEpochMs = effectiveNow,
                    reason = "executor interrupted",
                )
                RestartReconciliation(reconciled, RestartAction.WAIT_FOR_RESUME, changed = true)
            }

            IndexingJobState.PAUSE_REQUESTED -> {
                val checkpointedTracks = ledger.tracks.map { track ->
                    if (!track.state.isActiveStage()) track else track.copy(
                        state = track.checkpoint.asTrackState(),
                        currentAttemptNumber = null,
                        activeFailureId = null,
                        updatedAtEpochMs = effectiveNow,
                    )
                }
                val reconciled = updateJob(
                    ledger = ledger.copy(
                        tracks = checkpointedTracks,
                        updatedAtEpochMs = effectiveNow,
                    ),
                    state = IndexingJobState.PAUSED,
                    recoveryPhase = null,
                    nowEpochMs = effectiveNow,
                    reason = "pause completed at verified checkpoint after process stop",
                )
                RestartReconciliation(reconciled, RestartAction.WAIT_FOR_RESUME, changed = true)
            }

            IndexingJobState.ACTIVATING -> {
                val reconciled = updateJob(
                    ledger = ledger,
                    state = IndexingJobState.INTERRUPTED,
                    recoveryPhase = RecoveryPhase.ACTIVATION,
                    nowEpochMs = effectiveNow,
                    reason = "activation interrupted",
                )
                RestartReconciliation(reconciled, RestartAction.WAIT_FOR_RESUME, changed = true)
            }

            IndexingJobState.CANCELLING -> {
                if (activeTracks.isEmpty()) {
                    RestartReconciliation(ledger, RestartAction.FINISH_CANCELLATION, changed = false)
                } else {
                    val reconciled = checkpointCancellation(ledger, effectiveNow)
                    RestartReconciliation(
                        reconciled,
                        RestartAction.FINISH_CANCELLATION,
                        changed = true,
                    )
                }
            }

            IndexingJobState.INTERRUPTED,
            IndexingJobState.READY_TO_RESUME,
            IndexingJobState.PAUSED,
            IndexingJobState.WAITING_FOR_INPUT,
            -> RestartReconciliation(ledger, RestartAction.WAIT_FOR_RESUME, changed = false)

            else -> RestartReconciliation(ledger, RestartAction.NONE, changed = false)
        }
    }

    private fun recordFailureInternal(
        ledger: IndexingJobLedger,
        track: IndexingTrackLedger,
        descriptor: SelectedTrackDescriptor,
        code: TrackFailureCode,
        stage: IndexingStage,
        diagnostic: String,
        nowEpochMs: Long,
        resumeFrom: TrackCheckpoint,
    ): IndexingTrackLedger {
        val attempt = track.currentAttemptNumber
            ?: throw InvalidIndexingLedgerException("failure has no active track attempt")
        if (!track.state.isActiveStage() || stage != track.state.stage()) {
            throw InvalidIndexingLedgerException("failure stage does not match active track stage")
        }
        if (resumeFrom.rank() > track.checkpoint.rank()) {
            throw InvalidIndexingLedgerException("failure cannot resume ahead of verified checkpoint")
        }
        val policy = V2TrackFailurePolicies.forCode(code)
        val failureId = V2IndexingLedgerIds.failureId(
            workId = track.workId,
            code = code,
            stage = stage,
            sourceFingerprint = descriptor.sourceFingerprint,
            embeddingSpecId = ledger.jobSpec.embeddingSpec.specId,
            appBuildId = ledger.jobSpec.runtimeFingerprint.appBuildId,
        )
        val existing = track.failures.singleOrNull { it.failureId == failureId }
        val occurrences = (existing?.occurrences ?: 0) + 1
        val exhausted = policy.maxAutomaticRetries?.let { occurrences > it } ?: false
        val disposition = if (exhausted) FailureDisposition.BLOCKED else policy.initialDisposition
        val retryTrigger = if (exhausted) policy.exhaustedTrigger else policy.retryTrigger
        val shortDiagnostic = diagnostic.trim().ifBlank { code.name }.take(2_048)
        val aggregate = if (existing == null) {
            TrackFailureAggregate(
                failureId = failureId,
                code = code,
                category = policy.category,
                stage = stage,
                disposition = disposition,
                retryTrigger = retryTrigger,
                resumeFrom = resumeFrom,
                diagnostic = shortDiagnostic,
                firstOccurredAtEpochMs = nowEpochMs,
                lastOccurredAtEpochMs = nowEpochMs,
                firstAttemptNumber = attempt,
                lastAttemptNumber = attempt,
                occurrences = 1,
                sourceFingerprint = descriptor.sourceFingerprint,
                embeddingSpecId = ledger.jobSpec.embeddingSpec.specId,
                appBuildId = ledger.jobSpec.runtimeFingerprint.appBuildId,
            )
        } else {
            existing.copy(
                disposition = disposition,
                retryTrigger = retryTrigger,
                resumeFrom = resumeFrom,
                diagnostic = shortDiagnostic,
                lastOccurredAtEpochMs = nowEpochMs,
                lastAttemptNumber = attempt,
                occurrences = occurrences,
            )
        }
        val failures = track.failures.filterNot { it.failureId == failureId } + aggregate
        return track.copy(
            state = if (disposition == FailureDisposition.RETRYABLE) {
                IndexingTrackState.RETRYABLE_FAILURE
            } else {
                IndexingTrackState.BLOCKED_FAILURE
            },
            checkpoint = resumeFrom,
            currentAttemptNumber = null,
            activeFailureId = failureId,
            stageProgress = track.stageProgress?.takeIf { progress ->
                disposition == FailureDisposition.RETRYABLE &&
                    !code.discardsPartialProgress() &&
                    progress.stage == resumeFrom.nextResumableStage()
            },
            verifiedArtifacts = track.verifiedArtifacts.filter { artifact ->
                artifact.kind.checkpoint().rank() <= resumeFrom.rank()
            },
            failures = failures,
            updatedAtEpochMs = nowEpochMs,
        )
    }

    private fun transitionJob(
        ledger: IndexingJobLedger,
        source: IndexingJobState,
        target: IndexingJobState,
        nowEpochMs: Long,
        reason: String,
    ): IndexingJobLedger {
        requireState(ledger, source)
        return updateJob(ledger, target, null, nowEpochMs, reason)
    }

    private fun updateJob(
        ledger: IndexingJobLedger,
        state: IndexingJobState,
        recoveryPhase: RecoveryPhase?,
        nowEpochMs: Long,
        reason: String,
    ): IndexingJobLedger {
        V2IndexingLedgerValidator.requireValid(ledger)
        val updated = bump(
            ledger.copy(
                state = state,
                recoveryPhase = recoveryPhase,
                updatedAtEpochMs = maxOf(nowEpochMs, ledger.updatedAtEpochMs),
                stateReason = reason.take(2_048),
            ),
        )
        V2IndexingLedgerValidator.requireValid(updated)
        return updated
    }

    private fun updateTrack(
        ledger: IndexingJobLedger,
        workId: String,
        nowEpochMs: Long,
        transform: (IndexingTrackLedger, SelectedTrackDescriptor) -> IndexingTrackLedger,
    ): IndexingJobLedger {
        V2IndexingLedgerValidator.requireValid(ledger)
        val current = ledger.tracks.singleOrNull { it.workId == workId }
            ?: throw InvalidIndexingLedgerException("unknown workId $workId")
        val descriptor = descriptorFor(ledger, workId)
        val effectiveNow = maxOf(nowEpochMs, ledger.updatedAtEpochMs)
        val changed = transform(current, descriptor).copy(updatedAtEpochMs = effectiveNow)
        val updated = bump(
            ledger.copy(
                tracks = replaceTrack(ledger.tracks, changed),
                updatedAtEpochMs = effectiveNow,
            ),
        )
        V2IndexingLedgerValidator.requireValid(updated)
        return updated
    }

    private fun bump(ledger: IndexingJobLedger): IndexingJobLedger =
        ledger.copy(revision = Math.addExact(ledger.revision, 1L))

    private fun requireState(ledger: IndexingJobLedger, expected: IndexingJobState) {
        if (ledger.state != expected) throw invalidTransition("expected $expected", ledger.state)
    }

    private fun requireNoActiveTrack(ledger: IndexingJobLedger, action: String) {
        if (ledger.tracks.any { it.state.isActiveStage() }) {
            throw InvalidIndexingLedgerException("cannot $action before active stage checkpoints")
        }
    }

    private fun descriptorFor(
        ledger: IndexingJobLedger,
        workId: String,
    ): SelectedTrackDescriptor = ledger.jobSpec.tracks.singleOrNull { it.workId == workId }
        ?: throw InvalidIndexingLedgerException("job spec has no descriptor for $workId")

    private fun invalidTransition(action: String, state: IndexingJobState) =
        InvalidIndexingLedgerException("cannot $action while job is $state")

    private fun invalidTrackTransition(action: String, state: IndexingTrackState) =
        InvalidIndexingLedgerException("cannot $action while track is $state")
}

private fun TrackCheckpoint.asTrackState(): IndexingTrackState = when (this) {
    TrackCheckpoint.QUEUED -> IndexingTrackState.QUEUED
    TrackCheckpoint.PREFLIGHTED -> IndexingTrackState.PREFLIGHTED
    TrackCheckpoint.MERT_COMPLETE -> IndexingTrackState.MERT_COMPLETE
    TrackCheckpoint.CLAMP_COMPLETE -> IndexingTrackState.CLAMP_COMPLETE
    TrackCheckpoint.COMMITTED -> IndexingTrackState.COMMITTED
}

private fun VerifiedArtifactKind.checkpoint(): TrackCheckpoint = when (this) {
    VerifiedArtifactKind.MERT_FEATURES -> TrackCheckpoint.MERT_COMPLETE
    VerifiedArtifactKind.CLAMP_VECTOR -> TrackCheckpoint.CLAMP_COMPLETE
    VerifiedArtifactKind.DATABASE_COMMIT -> TrackCheckpoint.COMMITTED
}

private fun IndexingTrackState.stage(): IndexingStage = when (this) {
    IndexingTrackState.PREFLIGHTING -> IndexingStage.PREFLIGHT
    IndexingTrackState.DECODING -> IndexingStage.DECODE_AND_MERT
    IndexingTrackState.CLAMPING -> IndexingStage.CLAMP3
    IndexingTrackState.COMMITTING -> IndexingStage.DATABASE_COMMIT
    else -> throw InvalidIndexingLedgerException("$this is not an active stage")
}

private fun TrackFailureCode.discardsPartialProgress(): Boolean = when (this) {
    TrackFailureCode.INVALID_LOGICAL_SPAN,
    TrackFailureCode.CUE_SOURCE_IMAGE,
    TrackFailureCode.SOURCE_FINGERPRINT_CHANGED,
    TrackFailureCode.PROVIDER_SNAPSHOT_CHANGED,
    TrackFailureCode.CONTAINER_EOS_MISMATCH,
    TrackFailureCode.NO_AUDIO_STREAM,
    TrackFailureCode.UNSUPPORTED_CODEC_OR_CONTAINER,
    TrackFailureCode.DRM_PROTECTED,
    TrackFailureCode.CORRUPT_OR_TRUNCATED,
    TrackFailureCode.DECODER_ERROR,
    TrackFailureCode.BELOW_MINIMUM_DURATION,
    TrackFailureCode.INFERENCE_FAILED,
    TrackFailureCode.INVALID_MODEL_OUTPUT,
    TrackFailureCode.PARTIAL_ARTIFACT,
    TrackFailureCode.ARTIFACT_CHECKSUM_MISMATCH,
    TrackFailureCode.UNKNOWN_BLOCKED,
    -> true

    TrackFailureCode.SOURCE_MISSING,
    TrackFailureCode.STORAGE_UNMOUNTED,
    TrackFailureCode.SOURCE_UNREADABLE,
    TrackFailureCode.POWERAMP_PROVIDER_UNAVAILABLE,
    TrackFailureCode.ANDROID_AUDIO_PERMISSION_DENIED,
    TrackFailureCode.POWERAMP_PERMISSION_DENIED,
    TrackFailureCode.OUT_OF_MEMORY,
    TrackFailureCode.THERMAL_SHUTDOWN,
    TrackFailureCode.PROCESS_INTERRUPTED,
    TrackFailureCode.STAGE_TIMEOUT,
    TrackFailureCode.MODEL_LOAD_FAILED,
    TrackFailureCode.STORAGE_FULL,
    TrackFailureCode.DATABASE_BUSY,
    TrackFailureCode.DATABASE_GENERATION_CHANGED,
    TrackFailureCode.COMMIT_FAILED,
    TrackFailureCode.UNKNOWN_TRANSIENT,
    -> false

    TrackFailureCode.IMPORTED_ROW_AUTHORIZATION_CHANGED -> true
}

private fun replaceTrack(
    tracks: List<IndexingTrackLedger>,
    replacement: IndexingTrackLedger,
): List<IndexingTrackLedger> = tracks.map { track ->
    if (track.workId == replacement.workId) replacement else track
}
