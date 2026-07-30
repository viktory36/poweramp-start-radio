package com.powerampstartradio.services

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.IBinder
import android.util.Log
import android.util.AtomicFile
import android.widget.Toast
import androidx.core.app.NotificationCompat
import com.powerampstartradio.MainActivity
import com.powerampstartradio.R
import com.powerampstartradio.data.EmbeddedTrack
import com.powerampstartradio.data.EmbeddingDatabase
import com.powerampstartradio.data.EmbeddingIndex
import com.powerampstartradio.data.StableIdentityGenerationBinding
import com.powerampstartradio.data.StableTrackIdentityCatalog
import com.powerampstartradio.data.StableTrackIdentityResolution
import com.powerampstartradio.indexing.V2ActiveLibraryBindingEvidence
import com.powerampstartradio.indexing.V2ActiveLibraryCatalog
import com.powerampstartradio.indexing.V2ActiveLibraryCatalogLoader
import com.powerampstartradio.indexing.V2ActiveLibraryCatalogStore
import com.powerampstartradio.indexing.v2.V2EmbeddingCommitRepository
import com.powerampstartradio.indexing.v2.V2IndexGenerationReader
import com.powerampstartradio.indexing.v2.V2PowerampProviderSnapshotAcquirer
import com.powerampstartradio.indexing.v2.V2ProviderPathGroupSnapshot
import com.powerampstartradio.indexing.v2.V2ProviderPathRowEvidence
import com.powerampstartradio.indexing.v2.V2ProviderSpanReceiptReader
import com.powerampstartradio.indexing.v2.V2ResolvedActiveIndexGeneration
import com.powerampstartradio.poweramp.PowerampFileEntry
import com.powerampstartradio.poweramp.PowerampHelper
import com.powerampstartradio.poweramp.PowerampReceiver
import com.powerampstartradio.poweramp.PowerampTrack
import com.powerampstartradio.poweramp.QueueMutationResult
import com.powerampstartradio.poweramp.QueueMutationKind
import com.powerampstartradio.poweramp.TrackMatcher
import com.powerampstartradio.poweramp.TrackNormalization
import com.powerampstartradio.similarity.RecommendationEngine
import com.powerampstartradio.similarity.RecommendationAssetFiles
import com.powerampstartradio.similarity.RecommendationDomainEvidence
import com.powerampstartradio.similarity.ActiveRecommendationDomain
import com.powerampstartradio.similarity.GraphExplorationEvidence
import com.powerampstartradio.similarity.SimilarTrack
import com.powerampstartradio.ui.QueueStatus
import com.powerampstartradio.ui.ComposedRadioContract
import com.powerampstartradio.ui.DirectQueuePlacement
import com.powerampstartradio.ui.FindMusicQuerySpec
import com.powerampstartradio.ui.FindMusicSessionEvidence
import com.powerampstartradio.ui.FindMusicTrackEvidence
import com.powerampstartradio.ui.QueueDeliverySummary
import com.powerampstartradio.ui.QueueOrigin
import com.powerampstartradio.ui.QueuedTrackResult
import com.powerampstartradio.ui.RadioConfig
import com.powerampstartradio.ui.RadioGenerationToken
import com.powerampstartradio.ui.RadioResult
import com.powerampstartradio.ui.RadioSeedIdentity
import com.powerampstartradio.ui.RadioSessionOutcome
import com.powerampstartradio.ui.RadioUiState
import com.powerampstartradio.ui.ReplayExactRowAuthenticationPolicy
import com.powerampstartradio.ui.SeedSpec
import com.powerampstartradio.ui.SelectionMode
import com.powerampstartradio.ui.SessionEvidenceText
import com.powerampstartradio.ui.StableReplayTrackSelectionPolicy
import com.powerampstartradio.ui.effectiveLibraryAddedDays
import com.powerampstartradio.ui.forSelectionRequest
import com.powerampstartradio.ui.minimumLibraryAddedAtEpochSecond
import com.powerampstartradio.widget.StartRadioWidgetReceiver
import com.powerampstartradio.widget.WidgetRadioRequestState
import com.powerampstartradio.widget.WidgetRadioSeedReference
import com.powerampstartradio.widget.WidgetRadioStatus
import com.powerampstartradio.widget.WidgetRadioStatusStore
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeoutOrNull
import java.io.File
import java.io.FileNotFoundException
import java.util.ArrayDeque
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicLong

internal enum class WidgetRadioStartDisposition {
    STARTED,
    ALREADY_STARTING,
    ALREADY_HANDLED,
    FAILED,
}

internal data class WidgetRadioStartResult(
    val disposition: WidgetRadioStartDisposition,
    val requestId: String,
    val message: String,
)

internal object RecommendationAdmissionPresentation {
    fun unavailableMessage(
        coldOwnerReserved: Boolean,
        coldState: RecommendationWorkAdmission.ColdReconciliationState,
    ): String = when {
        coldOwnerReserved &&
            coldState == RecommendationWorkAdmission.ColdReconciliationState.FAILED ->
            "Saved indexing work could not be verified. " +
                "Open On-device indexing, then restart the app after resolving it."
        coldOwnerReserved ->
            "Reading saved indexing-job ownership. Try again when that finishes."
        else -> "On-device indexing is using the music model. " +
            "Try Start Radio after indexing finishes."
    }
}

internal object WidgetDeferredStartPolicy {
    fun shouldPublishWaitingStatus(
        durableExecutionAction: Boolean,
        explicitRequestId: String?,
        existingWorkMustDrain: Boolean,
    ): Boolean = durableExecutionAction && explicitRequestId != null && !existingWorkMustDrain
}

/** One exact provider row split into matching-canonical and presentation-preserving fields. */
internal class RequiredSeedProviderBinding private constructor(
    val matchingEntry: PowerampFileEntry,
    val displayTitle: String?,
    val displayArtist: String?,
    val displayAlbum: String?,
) {
    fun toDisplayTrack(reference: WidgetRadioSeedReference): PowerampTrack = PowerampTrack(
        realId = matchingEntry.id,
        title = displayTitle ?: reference.displayTitle,
        artist = displayArtist,
        album = displayAlbum,
        durationMs = matchingEntry.durationMs,
        path = matchingEntry.path,
        trackId = reference.queueOccurrenceId ?: -1L,
        categoryUri = reference.queueOccurrenceId?.let {
            PowerampHelper.ROOT_URI.buildUpon()
                .appendEncodedPath(PowerampHelper.TABLE_QUEUE)
                .build()
                .toString()
        },
    )

    companion object {
        fun from(entry: PowerampFileEntry): RequiredSeedProviderBinding =
            RequiredSeedProviderBinding(
                matchingEntry = entry,
                displayTitle = displayValue(entry.title),
                displayArtist = displayValue(entry.artist),
                displayAlbum = displayValue(entry.album),
            )

        fun from(row: V2ProviderPathRowEvidence): RequiredSeedProviderBinding {
            val normalizedArtist = TrackNormalization.normalizeArtist(row.artist)
            val normalizedAlbum = TrackNormalization.normalizeAlbum(row.album)
            val normalizedTitle = TrackNormalization.normalizeTitle(row.title)
            val normalizedPath = TrackNormalization.normalizePath(row.providerPhysicalPath)
            val normalizedDuration = Math.toIntExact(row.durationMs)
            val filename = File(row.providerPhysicalPath).name.substringBeforeLast(
                '.',
                File(row.providerPhysicalPath).name,
            )
            return RequiredSeedProviderBinding(
                matchingEntry = PowerampFileEntry(
                    id = row.powerampFileId,
                    artist = normalizedArtist,
                    album = normalizedAlbum,
                    title = normalizedTitle,
                    durationMs = normalizedDuration,
                    path = normalizedPath,
                    offsetMs = row.offsetMs,
                    offsetWasNull = row.offsetWasNull,
                    cueFolderId = row.cueSourceImageFolderId,
                    metadataKey = TrackNormalization.buildMetadataKey(
                        normalizedArtist,
                        normalizedAlbum,
                        normalizedTitle,
                        normalizedDuration,
                    ),
                    filenameKeys = TrackNormalization.buildFilenameKeys(
                        normalizedArtist,
                        normalizedTitle,
                        filename,
                    ),
                ),
                displayTitle = displayValue(row.title),
                displayArtist = displayValue(row.artist),
                displayAlbum = displayValue(row.album),
            )
        }

        private fun displayValue(value: String?): String? = value
            ?.trim()
            ?.takeIf(String::isNotBlank)
            ?.let(TrackNormalization::normalizeNfc)
    }
}

/**
 * Foreground service that handles the "Start Radio" functionality.
 *
 * Flow:
 * 1. Get current track from Poweramp
 * 2. Match to embedding database
 * 3. Find similar tracks using RecommendationEngine
 * 4. Map to Poweramp file IDs
 * 5. Place and verify the queue in Poweramp without changing playback state
 */
class RadioService : Service() {

    companion object {
        private const val TAG = "RadioService"
        private const val NOTIFICATION_ID = 1
        private const val CHANNEL_ID = "radio_service"

        private data class DisplayedRecommendationDomain(
            val providerGenerationId: String,
            val orderedActiveTrackIdsSha256: String,
            val activeTrackCount: Int,
        )

        private data class ProcessActiveLibrary(
            val generation: RadioGenerationToken,
            val catalog: V2ActiveLibraryCatalog,
        )

        @Volatile private var processActiveLibrary: ProcessActiveLibrary? = null
        private val processActiveLibraryCacheHits = AtomicLong(0L)
        private val processActiveLibraryProviderAcquisitions = AtomicLong(0L)

        internal data class ActiveLibraryProcessDiagnostics(
            val cacheHits: Long,
            val providerAcquisitions: Long,
        )

        internal fun publishActiveLibrarySnapshot(
            generation: RadioGenerationToken?,
            catalog: V2ActiveLibraryCatalog?,
        ) {
            processActiveLibrary = if (
                generation != null && catalog != null &&
                catalog.generationBinding.databaseGenerationId == generation.generationId
            ) {
                ProcessActiveLibrary(generation, catalog)
            } else {
                null
            }
        }

        private fun cachedActiveLibrary(
            generation: RadioGenerationToken,
        ): ProcessActiveLibrary? = processActiveLibrary?.takeIf { cached ->
            cached.generation == generation &&
                cached.catalog.generationBinding.databaseGenerationId == generation.generationId
        }

        /** Exact same-process handoff used by a recreated MainViewModel before reconciliation. */
        internal fun processActiveLibraryCatalog(
            generation: RadioGenerationToken,
        ): V2ActiveLibraryCatalog? = cachedActiveLibrary(generation)?.catalog?.also { catalog ->
            val hits = processActiveLibraryCacheHits.incrementAndGet()
            Log.i(
                TAG,
                "Active library process snapshot cache=hit hits=$hits " +
                    "generation=${generation.generationId} tracks=${catalog.activeTrackIds.size}",
            )
        }

        internal fun activeLibraryProcessDiagnostics(): ActiveLibraryProcessDiagnostics =
            ActiveLibraryProcessDiagnostics(
                cacheHits = processActiveLibraryCacheHits.get(),
                providerAcquisitions = processActiveLibraryProviderAcquisitions.get(),
            )

        private const val HISTORY_FILE = "session_history.json"
        private const val MAX_SESSIONS = 200

        const val ACTION_EXECUTE_REQUEST = "com.powerampstartradio.EXECUTE_RADIO_REQUEST"
        const val ACTION_EXECUTE_WIDGET_INGRESS =
            "com.powerampstartradio.EXECUTE_WIDGET_RADIO_INGRESS"
        const val ACTION_STOP = "com.powerampstartradio.STOP"
        const val ACTION_CANCEL = "com.powerampstartradio.CANCEL"
        const val EXTRA_REQUEST_ID = "request_id"
        const val DEFAULT_NUM_TRACKS = 50
        private const val DEFERRED_INDEXING_MESSAGE =
            "Radio request saved. It will start after on-device indexing finishes."

        private fun recommendationBusyMessage(): String =
            RecommendationAdmissionPresentation.unavailableMessage(
                coldOwnerReserved = RecommendationWorkAdmission.isReservedBy(
                    RecommendationWorkAdmission.coldReconciliationOwner,
                ),
                coldState = RecommendationWorkAdmission.coldReconciliationState.value,
            )

        @Volatile private var activeJob: Job? = null
        @Volatile private var activeDurableRequestId: String? = null
        @Volatile private var serviceInstance: RadioService? = null
        private val submissionReservation = SingleFlightRequestReservation()
        private val activeSubmissionJobsLock = Any()
        private val activeSubmissionJobs = mutableSetOf<Job>()
        private val admittedRecommendationWorkLock = Any()
        private val admittedSubmissionRequestByToken = mutableMapOf<String, String?>()
        private val admittedRequestIds = mutableSetOf<String>()
        val isSearchActive: Boolean
            get() = submissionReservation.isReserved || activeJob?.isActive == true

        private val _uiState = MutableStateFlow<RadioUiState>(RadioUiState.Idle)
        val uiState: StateFlow<RadioUiState> = _uiState.asStateFlow()

        private val _sessionHistory = MutableStateFlow<List<RadioResult>>(emptyList())
        val sessionHistory: StateFlow<List<RadioResult>> = _sessionHistory.asStateFlow()

        /** Drift reference embeddings for lazy rank computation, keyed by track ID. */
        val driftReferences = MutableStateFlow<Map<Long, FloatArray>>(emptyMap())

        // --- Session history persistence ---
        private var historyDir: File? = null
        private val historyLock = Any()
        @Volatile private var historyInitialized = false
        private val historyReady = CompletableDeferred<Unit>()
        @Volatile private var historyRevision = 0L
        @Volatile private var historyWrittenRevision = 0L
        private val gson = Gson()
        private val historyType = object : TypeToken<List<RadioResult>>() {}.type
        private val saveScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
        private val submissionScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
        private val submissionMutex = Mutex()

        private data class SubmissionReservationAttempt(val token: String?)

        private data class WidgetIngressReservationAttempt(
            val admission: WidgetRadioIngressAdmission,
            val acquiredSubmissionToken: String? = null,
            val alreadyOwnedReservation: Boolean = false,
        )

        private sealed interface ColdWidgetIngressDecision {
            data class RecoverExistingRequest(val requestId: String) : ColdWidgetIngressDecision
            data class Ingress(val admission: WidgetRadioIngressAdmission) : ColdWidgetIngressDecision
        }

        private fun registerAdmittedSubmission(token: String, requestId: String?) {
            synchronized(admittedRecommendationWorkLock) {
                admittedSubmissionRequestByToken[token] = requestId
                requestId?.let(admittedRequestIds::add)
            }
        }

        private fun bindAdmittedSubmission(token: String, requestId: String) {
            synchronized(admittedRecommendationWorkLock) {
                check(admittedSubmissionRequestByToken.containsKey(token)) {
                    "Radio admission token disappeared before durable request binding"
                }
                admittedSubmissionRequestByToken[token] = requestId
                admittedRequestIds += requestId
            }
        }

        private fun markAdmittedRequestDispatched(requestId: String) {
            synchronized(admittedRecommendationWorkLock) {
                admittedSubmissionRequestByToken.entries.removeAll { it.value == requestId }
            }
        }

        private fun releaseAdmittedSubmission(token: String) {
            synchronized(admittedRecommendationWorkLock) {
                val requestId = admittedSubmissionRequestByToken.remove(token)
                requestId?.let(admittedRequestIds::remove)
            }
        }

        private fun releaseAdmittedRequest(requestId: String) {
            synchronized(admittedRecommendationWorkLock) {
                admittedRequestIds -= requestId
                admittedSubmissionRequestByToken.entries.removeAll { it.value == requestId }
            }
        }

        private fun isAdmittedRequest(requestId: String): Boolean =
            synchronized(admittedRecommendationWorkLock) {
                requestId in admittedRequestIds
            }

        private fun hasAdmittedRecommendationWork(): Boolean =
            synchronized(admittedRecommendationWorkLock) {
                admittedSubmissionRequestByToken.isNotEmpty() || admittedRequestIds.isNotEmpty()
            }

        private fun failSubmission(submissionToken: String) {
            submissionReservation.failSubmission(submissionToken)
            releaseAdmittedSubmission(submissionToken)
        }

        private fun failRequest(requestId: String) {
            submissionReservation.failRequest(requestId)
            releaseAdmittedRequest(requestId)
        }

        private fun completeDispatch(requestId: String) {
            submissionReservation.completeDispatch(requestId)
            markAdmittedRequestDispatched(requestId)
        }

        /**
         * Reserve recommendation admission, wait boundedly for work that already won admission,
         * and release RadioService's retained recommendation resources. Active radio work is
         * never cancelled by this handoff.
         */
        suspend fun suspendAndReleaseRecommendationResources(timeoutMs: Long): Boolean {
            require(timeoutMs > 0L) { "Recommendation drain timeout must be positive" }
            check(RecommendationWorkAdmission.isIndexingReserved) {
                "Indexing must reserve recommendation admission before draining RadioService"
            }
            val drained = withTimeoutOrNull(timeoutMs) {
                while (true) {
                    val submissions = synchronized(activeSubmissionJobsLock) {
                        activeSubmissionJobs.filterNot { it.isCompleted }
                    }
                    val runningRadio = activeJob?.takeUnless { it.isCompleted }
                    val instance = serviceInstance
                    val serviceTransitionActive = instance?.hasRecommendationTransitionWork() == true
                    val admittedWorkActive = hasAdmittedRecommendationWork()
                    if (submissions.isEmpty() && runningRadio == null &&
                        !serviceTransitionActive && !admittedWorkActive
                    ) {
                        break
                    }
                    val waitFor = runningRadio ?: submissions.firstOrNull()
                    if (waitFor != null) {
                        waitFor.join()
                    } else {
                        // Covers the bounded claim-to-job-launch and widget-materialization gaps.
                        delay(10L)
                    }
                }
                true
            } ?: false
            if (!drained) {
                Log.w(TAG, "Timed out waiting for admitted recommendation work to finish")
                return false
            }
            serviceInstance?.closeRecommendationResources()
            // EmbeddingIndex/GraphIndex use read-only mapped buffers without a supported explicit
            // unmap API. After dropping every RadioService owner, request collection at this
            // deliberate memory-heavy mode boundary so indexing does not inherit their residency.
            System.gc()
            Log.i(TAG, "Recommendation work drained; RadioService resources released")
            return true
        }

        /** Restart durable requests that were deliberately left unclaimed during indexing. */
        fun kickDeferredRecovery(context: Context): Boolean {
            val appContext = context.applicationContext
            return RecommendationWorkAdmission.runIfRecommendationAllowed {
                val hasDeferred = runCatching {
                    RadioRequestStore(appContext.filesDir).recoverableRequestIds().isNotEmpty() ||
                        WidgetRadioIngressStore(appContext.filesDir).pendingRecords().isNotEmpty()
                }.onFailure { failure ->
                    Log.e(TAG, "Could not inspect deferred radio work", failure)
                }.getOrDefault(false)
                if (!hasDeferred) {
                    false
                } else {
                    runCatching {
                        val intent = Intent(appContext, RadioService::class.java).apply {
                            action = ACTION_EXECUTE_REQUEST
                        }
                        appContext.startForegroundService(intent)
                        true
                    }.onFailure { failure ->
                        Log.e(TAG, "Could not restart deferred radio work", failure)
                    }.getOrDefault(false)
                }
            } ?: false
        }

        fun initHistory(filesDir: File) {
            synchronized(historyLock) {
                if (historyInitialized && historyDir?.absolutePath == filesDir.absolutePath) return
                check(!historyInitialized) { "Session history was initialized with another directory" }
                historyDir = filesDir
                historyInitialized = true
            }
            saveScope.launch {
                try {
                    val atomicFile = AtomicFile(File(filesDir, HISTORY_FILE))
                    val loaded = try {
                        atomicFile.openRead().bufferedReader(Charsets.UTF_8).use { reader ->
                            val sessions: List<RadioResult>? = gson.fromJson(reader, historyType)
                            sessions.orEmpty().takeLast(MAX_SESSIONS)
                        }
                    } catch (_: FileNotFoundException) {
                        emptyList()
                    }
                    synchronized(historyLock) {
                        _sessionHistory.value = loaded
                    }
                    Log.d(TAG, "Loaded ${loaded.size} sessions from disk")
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to load session history", e)
                    synchronized(historyLock) { _sessionHistory.value = emptyList() }
                } finally {
                    historyReady.complete(Unit)
                }
            }
        }

        /** Upsert and fsync exactly one authoritative request outcome before journal completion. */
        private suspend fun persistSessionSynchronously(result: RadioResult): Boolean {
            historyReady.await()
            val dir = historyDir ?: return false
            val (snapshot, revision) = synchronized(historyLock) {
                val requestId = requireNotNull(result.requestId) {
                    "Durable sessions require a request ID"
                }
                val existingIndex = _sessionHistory.value.indexOfFirst { it.requestId == requestId }
                val updated = if (existingIndex >= 0) {
                    _sessionHistory.value.toMutableList().apply { this[existingIndex] = result }
                } else {
                    (_sessionHistory.value + result).takeLast(MAX_SESSIONS)
                }
                _sessionHistory.value = updated
                updated to ++historyRevision
            }
            return withContext(Dispatchers.IO) { writeHistorySnapshot(dir, snapshot, revision) }
        }

        private fun writeHistorySnapshot(
            dir: File,
            snapshot: List<RadioResult>,
            revision: Long,
        ): Boolean = synchronized(historyLock) {
            if (revision < historyWrittenRevision) return@synchronized true
            try {
                val atomicFile = AtomicFile(File(dir, HISTORY_FILE))
                val stream = atomicFile.startWrite()
                try {
                    stream.write(gson.toJson(snapshot, historyType).toByteArray(Charsets.UTF_8))
                    atomicFile.finishWrite(stream)
                    historyWrittenRevision = revision
                    true
                } catch (failure: Throwable) {
                    atomicFile.failWrite(stream)
                    throw failure
                }
            } catch (failure: Exception) {
                Log.e(TAG, "Failed to save session history", failure)
                false
            }
        }

        fun startRadio(
            context: Context,
            config: RadioConfig,
            showToasts: Boolean = false,
            origin: QueueOrigin = QueueOrigin.APP_RADIO,
        ) {
            require(origin != QueueOrigin.WIDGET_RADIO) {
                "Widget radio must use the compact durable tap entry point"
            }
            val displayedTrack = runCatching {
                PowerampReceiver.requireProviderVerifiedCurrentTrack(context)
            }.onFailure { failure ->
                Log.w(TAG, "Start Radio could not verify Poweramp's current track", failure)
            }.getOrNull()
            if (displayedTrack == null || displayedTrack.realId <= 0L) {
                _uiState.value = RadioUiState.Error("No current track is available in Poweramp")
                return
            }
            val expectedSeed = WidgetRadioSeedReference.from(displayedTrack.copy())
            val submissionToken = reserveSubmission(replaceActive = false) ?: return
            _uiState.value = RadioUiState.Loading(
                "Saving a durable radio request for " +
                    displayedTrack.title.orEmpty().ifBlank { "the current Poweramp track" },
            )
            submitRequest(
                context,
                submissionToken = submissionToken,
            ) { appContext ->
                buildPinnedRadioRequest(
                    context = appContext,
                    config = config,
                    currentSeedReference = expectedSeed,
                    showToasts = showToasts,
                    origin = origin,
                )
            }
        }

        /**
         * Cold-safe widget ingress. This method returns only after the exact tap-time seed and
         * configuration have become an immutable journal entry. Foreground-service ownership is
         * requested immediately when admission is known, or by recovery after the cold indexing
         * ownership check releases its temporary gate.
         */
        internal fun startRadioFromWidgetTap(
            context: Context,
            commandId: String,
            config: RadioConfig,
            expectedDisplayedSeed: WidgetRadioSeedReference,
        ): WidgetRadioStartResult {
            val appContext = context.applicationContext
            val store = RadioRequestStore(appContext.filesDir)
            val ingressStore = WidgetRadioIngressStore(appContext.filesDir)

            val coldDecision = runCatching {
                RecommendationWorkAdmission.runIfOnlyColdReconciliationReserved {
                    store.recoverableRequestIds().firstOrNull()?.let { requestId ->
                        runCatching { startForegroundRequest(appContext, requestId) }
                            .onFailure { failure ->
                                Log.e(TAG, "Could not rescue saved radio request $requestId", failure)
                            }
                        ColdWidgetIngressDecision.RecoverExistingRequest(requestId)
                    } ?: run {
                        val statusStore = WidgetRadioStatusStore(appContext.filesDir)
                        val admission = ingressStore.admitAfterPrecommit(
                            commandId = commandId,
                            expectedSeed = expectedDisplayedSeed,
                            config = config,
                        ) { pending ->
                            statusStore.write(
                                WidgetRadioStatus(
                                    requestId = pending.commandId,
                                    seed = pending.expectedSeed,
                                    state = WidgetRadioRequestState.STARTING,
                                    message =
                                        "Reading saved indexing-job ownership before starting radio",
                                    updatedAtEpochMs = System.currentTimeMillis(),
                                ),
                            )
                            // The manifest receiver and Service lifecycle both run on the main
                            // thread. onStartCommand cannot observe this ID until onReceive returns,
                            // after admitAfterPrecommit has durably published the ingress.
                            startForegroundWidgetIngress(appContext, pending.commandId)
                        }
                        when (admission) {
                            is WidgetRadioIngressAdmission.Accepted -> if (!admission.newlyPersisted) {
                                runCatching {
                                    statusStore.write(
                                        WidgetRadioStatus(
                                            requestId = admission.record.commandId,
                                            seed = admission.record.expectedSeed,
                                            state = WidgetRadioRequestState.STARTING,
                                            message =
                                                "Reading saved indexing-job ownership before starting radio",
                                            updatedAtEpochMs = System.currentTimeMillis(),
                                        ),
                                    )
                                }.onFailure { failure ->
                                    Log.e(TAG, "Could not repair widget listener status", failure)
                                }
                                runCatching {
                                    startForegroundWidgetIngress(
                                        appContext,
                                        admission.record.commandId,
                                    )
                                }.onFailure { failure ->
                                    Log.e(TAG, "Could not rescue saved widget command", failure)
                                }
                            }
                            is WidgetRadioIngressAdmission.Busy -> runCatching {
                                startForegroundWidgetIngress(
                                    appContext,
                                    admission.pending.commandId,
                                )
                            }.onFailure { failure ->
                                Log.e(TAG, "Could not rescue pending widget command", failure)
                            }
                            is WidgetRadioIngressAdmission.AlreadyFailed -> Unit
                        }
                        ColdWidgetIngressDecision.Ingress(admission)
                    }
                }
            }.getOrElse { failure ->
                Log.e(TAG, "Could not preserve cold widget command $commandId", failure)
                val message = widgetFailureMessage(
                    failure as? Exception
                        ?: IllegalStateException("Cold widget command failed", failure),
                )
                runCatching {
                    WidgetRadioStatusStore(appContext.filesDir).write(
                        WidgetRadioStatus(
                            requestId = commandId,
                            seed = expectedDisplayedSeed,
                            state = WidgetRadioRequestState.FAILED,
                            message = message,
                            updatedAtEpochMs = System.currentTimeMillis(),
                        ),
                    )
                }.onFailure { statusFailure ->
                    Log.e(TAG, "Could not publish cold widget failure", statusFailure)
                }
                return WidgetRadioStartResult(
                    WidgetRadioStartDisposition.FAILED,
                    commandId,
                    message,
                )
            }
            if (coldDecision != null) {
                return when (coldDecision) {
                    is ColdWidgetIngressDecision.RecoverExistingRequest ->
                        WidgetRadioStartResult(
                            WidgetRadioStartDisposition.ALREADY_STARTING,
                            coldDecision.requestId,
                            "Radio already starting",
                        )
                    is ColdWidgetIngressDecision.Ingress -> when (
                        val admission = coldDecision.admission
                    ) {
                    is WidgetRadioIngressAdmission.Accepted -> {
                        runCatching { StartRadioWidgetReceiver.updateAllWidgets(appContext) }
                        WidgetRadioStartResult(
                            if (admission.newlyPersisted) {
                                WidgetRadioStartDisposition.STARTED
                            } else {
                                WidgetRadioStartDisposition.ALREADY_STARTING
                            },
                            admission.record.commandId,
                            if (admission.newlyPersisted) {
                                "Starting radio"
                            } else {
                                "Radio already starting"
                            },
                        )
                    }
                    is WidgetRadioIngressAdmission.Busy -> WidgetRadioStartResult(
                        WidgetRadioStartDisposition.ALREADY_STARTING,
                        admission.pending.commandId,
                        "Radio already starting",
                    )
                    is WidgetRadioIngressAdmission.AlreadyFailed -> WidgetRadioStartResult(
                        WidgetRadioStartDisposition.ALREADY_HANDLED,
                        admission.record.commandId,
                        admission.record.terminalDetail
                            ?: "Previous radio request did not complete; tap again",
                    )
                    }
                }
            }
            if (RecommendationWorkAdmission.isIndexingReserved) {
                return WidgetRadioStartResult(
                    WidgetRadioStartDisposition.FAILED,
                    commandId,
                    recommendationBusyMessage(),
                )
            }

            if (store.hasRecord(commandId)) {
                runCatching { ingressStore.delete(commandId) }
                val state = runCatching { store.readStateKind(commandId) }.getOrNull()
                if (state in setOf(
                        RadioRequestStateKind.COMPLETED,
                        RadioRequestStateKind.FAILED,
                        RadioRequestStateKind.INTERRUPTED_NEEDS_RETRY,
                    )
                ) {
                    val widgetStatus = StartRadioWidgetReceiver.readRadioStatus(appContext)
                    if (widgetStatus?.requestId == commandId &&
                        widgetStatus.state in setOf(
                            WidgetRadioRequestState.STARTING,
                            WidgetRadioRequestState.BUSY,
                        )
                    ) {
                        val completed = state == RadioRequestStateKind.COMPLETED
                        StartRadioWidgetReceiver.updateRadioStatus(
                            context = appContext,
                            requestId = commandId,
                            state = if (completed) {
                                WidgetRadioRequestState.SUCCEEDED
                            } else {
                                WidgetRadioRequestState.FAILED
                            },
                            message = if (completed) {
                                "Radio request completed"
                            } else {
                                "Previous radio request did not complete; tap again"
                            },
                        )
                    } else {
                        StartRadioWidgetReceiver.updateAllWidgets(appContext)
                    }
                    return WidgetRadioStartResult(
                        WidgetRadioStartDisposition.ALREADY_HANDLED,
                        commandId,
                        if (state == RadioRequestStateKind.COMPLETED) {
                            "This radio request was already handled"
                        } else {
                            "Previous radio request did not complete; tap again"
                        },
                    )
                }
                val existing = store.readRequest(commandId)
                if (existing == null) {
                    startForegroundRequest(appContext, commandId)
                    return WidgetRadioStartResult(
                        WidgetRadioStartDisposition.ALREADY_STARTING,
                        commandId,
                        "Recovering saved radio request",
                    )
                }
                startForegroundRequest(appContext, commandId)
                existing.widgetSeedReference()?.let { seed ->
                    StartRadioWidgetReceiver.persistRadioStatus(
                        appContext,
                        WidgetRadioStatus(
                            requestId = commandId,
                            seed = seed,
                            state = WidgetRadioRequestState.BUSY,
                            message = "Radio already starting for ${seed.displayTitle}",
                            updatedAtEpochMs = System.currentTimeMillis(),
                        ),
                    )
                }
                return WidgetRadioStartResult(
                    WidgetRadioStartDisposition.ALREADY_STARTING,
                    commandId,
                    "Radio already starting",
                )
            }

            val outstandingId = store.recoverableRequestIds().firstOrNull()
            if (outstandingId != null) {
                startForegroundRequest(appContext, outstandingId)
                val outstanding = store.readRequest(outstandingId)
                val busySeed = outstanding?.widgetSeedReference()
                if (busySeed != null) {
                    StartRadioWidgetReceiver.persistRadioStatus(
                        appContext,
                        WidgetRadioStatus(
                            requestId = outstandingId,
                            seed = busySeed,
                            state = WidgetRadioRequestState.BUSY,
                            message = "Radio already starting",
                            updatedAtEpochMs = System.currentTimeMillis(),
                        ),
                    )
                } else {
                    StartRadioWidgetReceiver.updateAllWidgets(appContext)
                }
                return WidgetRadioStartResult(
                    WidgetRadioStartDisposition.ALREADY_STARTING,
                    outstandingId,
                    "Radio already starting",
                )
            }

            var newlyPersistedIngress = false
            var acquiredReservation = false
            return try {
                val admissionAttempt = RecommendationWorkAdmission.runIfRecommendationAllowed {
                    val admission = ingressStore.admit(
                        commandId = commandId,
                        expectedSeed = expectedDisplayedSeed,
                        config = config,
                    )
                    if (admission is WidgetRadioIngressAdmission.Accepted) {
                        try {
                            val alreadyOwned = submissionReservation.activeRequestId == commandId
                            val token = if (alreadyOwned) {
                                null
                            } else {
                                reserveSubmissionInsideRecommendationAdmission(
                                    replaceActive = false,
                                    requestId = commandId,
                                )
                            }
                            WidgetIngressReservationAttempt(
                                admission = admission,
                                acquiredSubmissionToken = token,
                                alreadyOwnedReservation = alreadyOwned,
                            )
                        } catch (failure: Throwable) {
                            if (admission.newlyPersisted) {
                                runCatching { ingressStore.delete(commandId) }
                                    .onFailure(failure::addSuppressed)
                            }
                            throw failure
                        }
                    } else {
                        WidgetIngressReservationAttempt(admission)
                    }
                } ?: return WidgetRadioStartResult(
                    WidgetRadioStartDisposition.FAILED,
                    commandId,
                    recommendationBusyMessage(),
                )
                val admission = admissionAttempt.admission
                when (admission) {
                    is WidgetRadioIngressAdmission.Busy -> {
                        startForegroundWidgetIngress(appContext, admission.pending.commandId)
                        WidgetRadioStartResult(
                            WidgetRadioStartDisposition.ALREADY_STARTING,
                            admission.pending.commandId,
                            "Radio already starting",
                        )
                    }
                    is WidgetRadioIngressAdmission.AlreadyFailed -> {
                        val message = admission.record.terminalDetail
                            ?: "Previous radio request did not complete; tap again"
                        WidgetRadioStartResult(
                            WidgetRadioStartDisposition.ALREADY_HANDLED,
                            commandId,
                            message,
                        )
                    }
                    is WidgetRadioIngressAdmission.Accepted -> {
                        newlyPersistedIngress = admission.newlyPersisted
                        acquiredReservation = admissionAttempt.acquiredSubmissionToken != null
                        val ownsReservation = admissionAttempt.alreadyOwnedReservation ||
                            acquiredReservation
                        if (!ownsReservation) {
                            val activeRequestId = WidgetBusyStatusOwnerPolicy.resolve(
                                reservedRequestId = submissionReservation.activeRequestId,
                                activeDurableRequestId = activeDurableRequestId,
                            )
                            val detail = "Another radio request is already starting"
                            if (admission.newlyPersisted) {
                                check(ingressStore.delete(commandId)) {
                                    "Could not roll back an unreserved widget command"
                                }
                            }
                            return WidgetRadioStartResult(
                                WidgetRadioStartDisposition.ALREADY_STARTING,
                                activeRequestId ?: commandId,
                                detail,
                            )
                        }

                        // Close a cross-journal race with an app request published after the first
                        // recoverable-request check but before this ingress record was committed.
                        val racedRequest = store.recoverableRequestIds()
                            .firstOrNull { it != commandId }
                        if (racedRequest != null) {
                            val detail = "Another radio request is already starting"
                            ingressStore.markFailed(commandId, detail)
                            failRequest(commandId)
                            startForegroundRequest(appContext, racedRequest)
                            return WidgetRadioStartResult(
                                WidgetRadioStartDisposition.ALREADY_STARTING,
                                racedRequest,
                                detail,
                            )
                        }

                        startForegroundWidgetIngress(appContext, commandId)
                        _uiState.value = RadioUiState.Loading(
                            "Starting radio for ${expectedDisplayedSeed.displayTitle}",
                        )
                        WidgetRadioStartResult(
                            if (admission.newlyPersisted) {
                                WidgetRadioStartDisposition.STARTED
                            } else {
                                WidgetRadioStartDisposition.ALREADY_STARTING
                            },
                            commandId,
                            "Starting radio for ${expectedDisplayedSeed.displayTitle}",
                        )
                    }
                }
            } catch (failure: Exception) {
                val message = widgetFailureMessage(failure)
                if (newlyPersistedIngress) {
                    runCatching { ingressStore.markFailed(commandId, message) }
                }
                if (newlyPersistedIngress || acquiredReservation) {
                    failRequest(commandId)
                }
                _uiState.value = RadioUiState.Error(message)
                Log.e(TAG, "Could not accept widget radio request $commandId", failure)
                WidgetRadioStartResult(
                    if (newlyPersistedIngress || acquiredReservation) {
                        WidgetRadioStartDisposition.FAILED
                    } else {
                        WidgetRadioStartDisposition.ALREADY_STARTING
                    },
                    commandId,
                    if (newlyPersistedIngress || acquiredReservation) message
                    else "Radio already starting",
                )
            }
        }

        /**
         * Queue a pre-computed list of tracks directly into Poweramp.
         * No recommendation engine — just map and queue.
         * Admission is synchronous; generation revalidation and queue mutation remain asynchronous.
         */
        internal fun queueDirectly(
            context: Context,
            tracks: List<EmbeddedTrack>,
            trackIdentities: List<RadioSeedIdentity>,
            displayedBinding: StableIdentityGenerationBinding,
            displayedProviderGenerationId: String,
            displayedOrderedActiveTrackIdsSha256: String,
            displayedActiveTrackCount: Int,
            label: String,
            origin: QueueOrigin,
            placement: DirectQueuePlacement = DirectQueuePlacement.REPLACE_UPCOMING,
            findMusicSessionEvidence: FindMusicSessionEvidence? = null,
            findMusicTrackEvidence: List<FindMusicTrackEvidence>? = null,
        ): RadioRequestAdmission = RadioRequestAdmissionCoordinator.admit(
            reserve = { reserveSubmission(replaceActive = true) },
            release = { submissionToken ->
                failSubmission(submissionToken)
            },
            schedule = { submissionToken ->
                val trackSnapshot = tracks.map { it.copy() }
                val identitySnapshot = trackIdentities.map { it.copy() }
                val bindingSnapshot = displayedBinding.copy()
                val findMusicSessionSnapshot = findMusicSessionEvidence?.copy(
                    querySpec = findMusicSessionEvidence.querySpec.copy(
                        textIngredients = findMusicSessionEvidence.querySpec.textIngredients.toList(),
                        songSeeds = findMusicSessionEvidence.querySpec.songSeeds.toList(),
                    ),
                    stableResultReduction = findMusicSessionEvidence.stableResultReduction.copy(),
                    textQueuePlan = findMusicSessionEvidence.textQueuePlan?.let { plan ->
                        plan.copy(
                            orderedSelectedTrackIds = plan.orderedSelectedTrackIds.toList(),
                            orderedOriginalTextObjectiveRanks =
                                plan.orderedOriginalTextObjectiveRanks.toList(),
                            dppSelection = plan.dppSelection?.let { proof ->
                                proof.copy(
                                    attemptedCandidateCounts =
                                        proof.attemptedCandidateCounts.toList(),
                                )
                            },
                        )
                    },
                    allOfQueuePlan = findMusicSessionEvidence.allOfQueuePlan?.let { plan ->
                        plan.copy(
                            orderedSelectedTrackIds = plan.orderedSelectedTrackIds.toList(),
                            orderedOriginalAllOfObjectiveRanks =
                                plan.orderedOriginalAllOfObjectiveRanks.toList(),
                            dppSelection = plan.dppSelection.copy(
                                attemptedCandidateCounts =
                                    plan.dppSelection.attemptedCandidateCounts.toList(),
                                selectedMarginalGains =
                                    plan.dppSelection.selectedMarginalGains.toList(),
                            ),
                        )
                    },
                )
                val findMusicTrackSnapshot = findMusicTrackEvidence?.map { evidence ->
                    evidence.copy(ingredientPercentiles = evidence.ingredientPercentiles.toList())
                }
                val placementAction = when (placement) {
                    DirectQueuePlacement.REPLACE_UPCOMING ->
                        "replacing Poweramp's upcoming queue"
                    DirectQueuePlacement.APPEND -> "appending to Poweramp's queue"
                }
                _uiState.value = RadioUiState.Searching(
                    "Verifying ${trackSnapshot.size} displayed tracks against the active " +
                        "music index before $placementAction...",
                )
                submitRequest(context, submissionToken) { appContext ->
                    buildPinnedDirectQueueRequest(
                        context = appContext,
                        tracks = trackSnapshot,
                        trackIdentities = identitySnapshot,
                        displayedBinding = bindingSnapshot,
                        displayedDomain = DisplayedRecommendationDomain(
                            displayedProviderGenerationId,
                            displayedOrderedActiveTrackIdsSha256,
                            displayedActiveTrackCount,
                        ),
                        label = label,
                        origin = origin,
                        placement = placement,
                        findMusicSessionEvidence = findMusicSessionSnapshot,
                        findMusicTrackEvidence = findMusicTrackSnapshot,
                    )
                }
            },
        )

        /** Replay only the exact saved Poweramp occurrences; never metadata-rematch history. */
        fun replaySession(
            context: Context,
            session: RadioResult,
            placement: DirectQueuePlacement,
        ) {
            val submissionToken = reserveSubmission(replaceActive = true) ?: return
            val replayTrackCount = session.tracks.count { it.status == QueueStatus.QUEUED }
            val placementAction = when (placement) {
                DirectQueuePlacement.REPLACE_UPCOMING ->
                    "replacing Poweramp's upcoming queue"
                DirectQueuePlacement.APPEND -> "appending to Poweramp's queue"
            }
            _uiState.value = RadioUiState.Searching(
                "Verifying $replayTrackCount saved queue tracks against the active music index " +
                    "before $placementAction...",
            )
            submitRequest(context, submissionToken) { appContext ->
                buildPinnedHistoryReplayRequest(appContext, session, placement)
            }
        }

        private fun submitRequest(
            context: Context,
            submissionToken: String,
            requestBuilder: suspend (Context) -> DurableRadioRequest,
        ) {
            val appContext = context.applicationContext
            val submissionJob = submissionScope.launch(start = CoroutineStart.LAZY) {
                submissionMutex.withLock {
                    var requestId: String? = null
                    val store = RadioRequestStore(appContext.filesDir)
                    try {
                        val request = requestBuilder(appContext)
                        val persistedId = store.persist(request)
                        requestId = persistedId
                        check(submissionReservation.bindRequest(submissionToken, persistedId)) {
                            "Radio submission reservation changed before request binding"
                        }
                        bindAdmittedSubmission(submissionToken, persistedId)
                        startForegroundRequest(appContext, checkNotNull(requestId))
                    } catch (failure: Exception) {
                        requestId?.let { persistedId ->
                            runCatching {
                                store.markFailed(
                                    persistedId,
                                    "Radio could not start. Open Start Radio and try again.",
                                )
                            }.onFailure { stateFailure ->
                                Log.e(TAG, "Could not terminalize unstarted request $persistedId", stateFailure)
                            }
                        }
                        failSubmission(submissionToken)
                        Log.e(TAG, "Could not persist or start radio request", failure)
                        _uiState.value = RadioUiState.Error(
                            "Radio could not start. Open Start Radio and try again.",
                        )
                    }
                }
            }
            synchronized(activeSubmissionJobsLock) {
                activeSubmissionJobs += submissionJob
            }
            submissionJob.invokeOnCompletion {
                synchronized(activeSubmissionJobsLock) {
                    activeSubmissionJobs -= submissionJob
                }
            }
            submissionJob.start()
        }

        private fun startForegroundRequest(context: Context, requestId: String) {
            val intent = Intent(context, RadioService::class.java).apply {
                action = ACTION_EXECUTE_REQUEST
                putExtra(EXTRA_REQUEST_ID, requestId)
            }
            context.startForegroundService(intent)
        }

        private fun startForegroundWidgetIngress(context: Context, commandId: String) {
            val intent = Intent(context, RadioService::class.java).apply {
                action = ACTION_EXECUTE_WIDGET_INGRESS
                putExtra(EXTRA_REQUEST_ID, commandId)
            }
            context.startForegroundService(intent)
        }

        private fun DurableRadioRequest.widgetSeedReference(): WidgetRadioSeedReference? {
            val payload = radio ?: return null
            return WidgetRadioSeedReference.from(payload.seed.displayTrack, payload.seed.identity)
        }

        private fun widgetFailureMessage(failure: Exception): String {
            val detail = failure.message?.trim()?.takeIf(String::isNotBlank)
                .orEmpty()
            return when {
                failure is SecurityException ||
                    detail.contains("permission", ignoreCase = true) ||
                    detail.contains("access", ignoreCase = true) ->
                    "Poweramp access is unavailable. Open Start Radio and grant access."
                detail.contains("authenticated Poweramp", ignoreCase = true) ||
                    detail.contains("playback no longer matches", ignoreCase = true) ->
                    "Poweramp playback changed. Return to Poweramp, then tap Start Radio again."
                detail.contains("No track", ignoreCase = true) ->
                    "No Poweramp track is available. Start a track, then try again."
                else -> "Radio could not start. Open Start Radio, then try again."
            }
        }

        private fun reserveSubmission(
            replaceActive: Boolean,
            requestId: String? = null,
        ): String? {
            val attempt = RecommendationWorkAdmission.runIfRecommendationAllowed {
                SubmissionReservationAttempt(
                    reserveSubmissionInsideRecommendationAdmission(replaceActive, requestId),
                )
            }
            if (attempt == null) {
                _uiState.value = RadioUiState.Error(recommendationBusyMessage())
                return null
            }
            return attempt.token
        }

        /** Caller must already own RecommendationWorkAdmission's bounded admission section. */
        private fun reserveSubmissionInsideRecommendationAdmission(
            replaceActive: Boolean,
            requestId: String? = null,
        ): String? {
            if (!replaceActive && activeJob?.isActive == true) return null
            val token = UUID.randomUUID().toString()
            if (!submissionReservation.tryAcquire(token, requestId)) return null

            // Close the race with a service job that became active while the reservation was being
            // acquired. Replacement commands own the reservation before cancelling so a second
            // replacement tap cannot enqueue behind them.
            if (!replaceActive && activeJob?.isActive == true) {
                submissionReservation.failSubmission(token)
                return null
            }
            registerAdmittedSubmission(token, requestId)
            if (replaceActive && activeJob?.isActive == true) cancelSearch()
            return token
        }

        private fun buildPinnedRadioRequest(
            context: Context,
            config: RadioConfig,
            currentSeedReference: WidgetRadioSeedReference,
            showToasts: Boolean,
            origin: QueueOrigin,
            requestId: String = UUID.randomUUID().toString(),
            requestCreatedAtEpochMs: Long = System.currentTimeMillis(),
        ): DurableRadioRequest {
            val library = requireActiveLibrary(context, currentSeedReference)
            val active = library.activeGeneration
            val generation = library.generation
            val activeCatalog = library.activeCatalog
            withEmbeddingDatabase(active.databaseFile) { database ->
                val providerBinding = requireNotNull(library.requiredSeedProviderBinding) {
                    "The displayed Poweramp seed was not verified"
                }
                val provider = providerBinding.matchingEntry
                require(currentSeedReference.matchesProvider(provider)) {
                    "The displayed Poweramp seed changed before it could be pinned"
                }
                val trackId = requireNotNull(
                    activeCatalog.trackIdForPowerampFile(provider.id),
                ) {
                    "The displayed Poweramp track is not in the active generation"
                }
                val track = requireNotNull(database.getTrackById(trackId)) {
                    "The displayed Poweramp track has no active embedding row"
                }
                val seed = PinnedRadioSeed(
                    identity = RadioSeedIdentity(
                        embeddedTrackId = track.id,
                        stableTrackSpanId = readStableTrackSpanId(
                            active.databaseFile,
                            track.id,
                            generation.embeddingSpecId,
                        ),
                    ),
                    displayTrack = providerBinding.toDisplayTrack(currentSeedReference),
                    matchType = TrackMatcher.MatchType.ACTIVE_CATALOG_EXACT,
                )
                val embeddingIndex = EmbeddingIndex.mmap(active.embeddingFile, preload = false)
                val identityCatalog = StableTrackIdentityCatalog.load(
                    filesDir = context.filesDir,
                    database = database,
                    embeddingIndex = embeddingIndex,
                )
                val activeDomain = ActiveRecommendationDomain.create(
                    activeCatalog = activeCatalog,
                    identityCatalog = identityCatalog,
                    embeddingIndex = embeddingIndex,
                )
                val candidateDomain = activeDomain.candidateDomain(
                    minimumLibraryAddedAtEpochSecond(
                        config.effectiveLibraryAddedDays,
                        requestCreatedAtEpochMs / 1_000L,
                    ),
                )
                val eligibleCandidateIdentityCount =
                    candidateDomain.eligibleCandidateIdentityCount(
                        seed.identity.embeddedTrackId,
                    )
                return DurableRadioRequest.radio(
                    generation = generation,
                    providerGenerationId = library.providerGenerationId,
                    config = config.forSelectionRequest(eligibleCandidateIdentityCount),
                    seed = seed,
                    showToasts = showToasts,
                    origin = origin,
                    requestId = requestId,
                    createdAtEpochMs = requestCreatedAtEpochMs,
                )
            }
        }

        private fun buildPinnedDirectQueueRequest(
            context: Context,
            tracks: List<EmbeddedTrack>,
            trackIdentities: List<RadioSeedIdentity>,
            displayedBinding: StableIdentityGenerationBinding,
            displayedDomain: DisplayedRecommendationDomain,
            label: String,
            origin: QueueOrigin,
            placement: DirectQueuePlacement,
            findMusicSessionEvidence: FindMusicSessionEvidence?,
            findMusicTrackEvidence: List<FindMusicTrackEvidence>?,
        ): DurableRadioRequest {
            val library = requireActiveLibrary(context)
            val active = library.activeGeneration
            val generation = library.generation
            requireDisplayedDomainBinding(library, displayedDomain)
            DisplayedResultBindingGuard.requireExactGeneration(displayedBinding, generation)
            require(tracks.size == trackIdentities.size) {
                "Displayed track and identity counts differ"
            }
            require((findMusicSessionEvidence == null) == (findMusicTrackEvidence == null)) {
                "Displayed Find Music session and row evidence must be present together"
            }
            findMusicSessionEvidence?.let { evidence ->
                require(evidence.querySpec.libraryBinding == displayedBinding) {
                    "Displayed Find Music query binding differs from its ranked tracks"
                }
                require(
                    evidence.orderedActiveTrackIdsSha256 ==
                        displayedDomain.orderedActiveTrackIdsSha256 &&
                        evidence.activeTrackCount == displayedDomain.activeTrackCount,
                ) { "Displayed Find Music evidence differs from its active provider domain" }
                require(findMusicTrackEvidence?.size == tracks.size) {
                    "Displayed Find Music row evidence does not align with its tracks"
                }
            }
            withEmbeddingDatabase(active.databaseFile) { database ->
                val exactTracks = tracks.zip(trackIdentities).map { (requested, identity) ->
                    val exact = requireNotNull(database.getTrackById(identity.embeddedTrackId)) {
                        "Track ${identity.embeddedTrackId} is not in the active generation"
                    }
                    DisplayedResultBindingGuard.requireExactTrackIdentity(
                        displayedTrackId = requested.id,
                        displayedIdentity = identity,
                        activeStableTrackSpanId = readStableTrackSpanId(
                            active.databaseFile,
                            exact.id,
                            generation.embeddingSpecId,
                        ),
                    )
                    require(library.activeCatalog.containsActiveTrack(exact.id)) {
                        "Track ${exact.id} is not in the active Poweramp catalog"
                    }
                    exact
                }
                val fileIds = exactTracks.map { track ->
                    requireNotNull(library.activeCatalog.powerampFileIdForTrack(track.id)) {
                        "Track ${track.id} has no exact active Poweramp occurrence"
                    }
                }
                return DurableRadioRequest.directQueue(
                    generation = generation,
                    providerGenerationId = library.providerGenerationId,
                    tracks = exactTracks,
                    trackIdentities = trackIdentities,
                    resolvedPowerampFileIds = fileIds,
                    label = label,
                    origin = origin,
                    placement = placement,
                    findMusicSessionEvidence = findMusicSessionEvidence,
                    findMusicTrackEvidence = findMusicTrackEvidence,
                )
            }
        }

        private fun buildPinnedHistoryReplayRequest(
            context: Context,
            session: RadioResult,
            placement: DirectQueuePlacement,
        ): DurableRadioRequest {
            val replayRows = session.tracks.filter { it.status == QueueStatus.QUEUED }
            require(replayRows.isNotEmpty()) { "Saved session has no verified queue occurrences" }
            val replayFindMusicEvidence = session.findMusicSessionEvidence?.let { evidence ->
                require(replayRows.all { it.findMusicEvidence != null }) {
                    "Saved Find Music session has incomplete row-ranking evidence"
                }
                evidence
            }
            val replayFindMusicRows = replayFindMusicEvidence?.let {
                replayRows.map { row -> checkNotNull(row.findMusicEvidence) }
            }
            val savedGeneration = requireNotNull(session.generation) {
                "Legacy sessions cannot be replayed exactly"
            }
            val library = requireActiveLibrary(context)
            val active = library.activeGeneration
            val currentGeneration = library.generation
            replayRows.forEachIndexed { index, saved ->
                requireNotNull(saved.resolvedPowerampFileId) {
                    "Saved queue occurrence ${index + 1} has no exact Poweramp file ID"
                }
            }
            val savedQueueOccurrenceIds = replayRows.mapIndexed { index, saved ->
                requireNotNull(saved.resolvedPowerampQueueId) {
                    "Saved row ${index + 1} has no exact queue-delivery occurrence evidence"
                }
            }
            require(savedQueueOccurrenceIds.toSet().size == savedQueueOccurrenceIds.size) {
                "Saved queue-delivery occurrence evidence is duplicated"
            }
            withEmbeddingDatabase(active.databaseFile) { database ->
                val sameGeneration = currentGeneration == savedGeneration
                val sameProviderGeneration = session.providerGenerationId != null &&
                    session.providerGenerationId == library.providerGenerationId
                var stableIdentityCatalog: StableTrackIdentityCatalog? = null
                fun stableIdentityCatalog(): StableTrackIdentityCatalog =
                    stableIdentityCatalog ?: StableTrackIdentityCatalog.load(
                        filesDir = context.filesDir,
                        database = database,
                        embeddingIndex = EmbeddingIndex.mmap(active.embeddingFile),
                    ).also { stableIdentityCatalog = it }
                val identities = HistoryReplayIdentityPolicy.resolve(
                    savedRows = replayRows.map { saved ->
                        SavedHistoryReplayIdentity(
                            track = saved.track,
                            stableTrackSpanId = saved.stableTrackSpanId,
                            resolvedPowerampFileId = checkNotNull(saved.resolvedPowerampFileId),
                        )
                    },
                    resolveAuthenticatedExactRow = { saved ->
                        saved.embeddedTrackId.takeIf { trackId ->
                            ReplayExactRowAuthenticationPolicy.isAuthenticated(
                                sameGeneration = sameGeneration,
                                sameProviderGeneration = sameProviderGeneration,
                                savedTrack = saved.track,
                                savedStableTrackSpanId = saved.stableTrackSpanId,
                                savedPowerampFileId = saved.resolvedPowerampFileId,
                                currentTrack = database.getTrackById(trackId),
                                currentStableTrackSpanId = readStableTrackSpanId(
                                    active.databaseFile,
                                    trackId,
                                    currentGeneration.embeddingSpecId,
                                ),
                                currentPowerampFileId =
                                    library.activeCatalog.powerampFileIdForTrack(trackId),
                            )
                        }
                    },
                    resolveFullContentStableSpan = stable@{ saved ->
                        // An exact generation mismatch indicates corruption, not a remapping event.
                        if (sameGeneration) return@stable null
                        val stableId = saved.stableTrackSpanId ?: return@stable null
                        when (val resolution = stableIdentityCatalog().resolveStable(stableId)) {
                            is StableTrackIdentityResolution.Resolved ->
                                StableReplayTrackSelectionPolicy.select(
                                    equivalentTrackIds = resolution.allEquivalentTrackIds,
                                    savedPowerampFileId = saved.resolvedPowerampFileId,
                                    preferSavedPowerampOccurrence = sameProviderGeneration,
                                    currentPowerampFileId =
                                        library.activeCatalog::powerampFileIdForTrack,
                                )
                            else -> null
                        }
                    },
                )
                val exactTracks = identities.mapIndexed { index, identity ->
                    requireNotNull(database.getTrackById(identity.embeddedTrackId)) {
                        "Resolved occurrence ${index + 1} is absent from the active generation"
                    }.also { track ->
                        require(library.activeCatalog.containsActiveTrack(track.id)) {
                            "Resolved occurrence ${index + 1} is absent from the active Poweramp catalog"
                        }
                    }
                }
                val currentPowerampIds = exactTracks.mapIndexed { index, track ->
                    requireNotNull(library.activeCatalog.powerampFileIdForTrack(track.id)) {
                            "Resolved recording ${index + 1} has no unique current Poweramp occurrence"
                    }
                }
                return DurableRadioRequest.directQueue(
                    generation = currentGeneration,
                    providerGenerationId = library.providerGenerationId,
                    tracks = exactTracks,
                    trackIdentities = identities,
                    resolvedPowerampFileIds = currentPowerampIds,
                    label = SessionEvidenceText.seedTitle(session),
                    origin = QueueOrigin.HISTORY_REQUEUE,
                    placement = placement,
                    findMusicSessionEvidence = replayFindMusicEvidence,
                    findMusicTrackEvidence = replayFindMusicRows,
                )
            }
        }

        private fun requireActiveGeneration(
            context: Context,
        ): Pair<com.powerampstartradio.indexing.v2.V2ResolvedActiveIndexGeneration, RadioGenerationToken> {
            val active = V2IndexGenerationReader.requireActive(context.filesDir)
            val manifest = active.manifest
            return active to RadioGenerationToken(
                generationId = manifest.generationId,
                activationBindingId = manifest.activationBindingId,
                manifestSha256 = active.manifestSha256,
                embeddingSpecId = manifest.receiptEmbeddingSpec.specId,
                databaseContentSha256 = manifest.databaseContentSha256,
                orderedTrackSetSha256 = manifest.orderedTrackSetSha256,
                stableTrackUidMappingSha256 = manifest.stableTrackUidCoverage.mappingSha256,
            )
        }

        private data class BoundActiveLibrary(
            val activeGeneration: V2ResolvedActiveIndexGeneration,
            val generation: RadioGenerationToken,
            val providerGenerationId: String,
            val activeCatalog: V2ActiveLibraryCatalog,
            val requiredSeedProviderBinding: RequiredSeedProviderBinding? = null,
        )

        private fun requireActiveLibrary(
            context: Context,
            requiredSeed: WidgetRadioSeedReference? = null,
        ): BoundActiveLibrary {
            val (active, generation) = requireActiveGeneration(context)
            val targetedBefore = requiredSeed?.let { seed ->
                PowerampHelper.requireFileEntryById(context, seed.powerampFileId).also { entry ->
                    require(seed.matchesProvider(entry)) {
                        "The displayed Poweramp seed no longer matches its provider row"
                    }
                }
            }
            val processCatalog = processActiveLibraryCatalog(generation)
            val cachedCatalog = processCatalog ?: V2ActiveLibraryCatalogStore(
                context.filesDir,
            ).read(active)?.also { durableCatalog ->
                publishActiveLibrarySnapshot(generation, durableCatalog)
                Log.i(
                    TAG,
                    "Active library durable cache=hit tracks=" +
                        durableCatalog.activeTrackIds.size,
                )
            }
            cachedCatalog?.let { catalog ->
                val seedStatus = if (targetedBefore == null) {
                    CachedSeedBindingStatus.CURRENT
                } else {
                    cachedSeedBindingStatus(
                        active = active,
                        catalog = catalog,
                        current = targetedBefore,
                    )
                }
                if (seedStatus == CachedSeedBindingStatus.CURRENT ||
                    seedStatus == CachedSeedBindingStatus.KNOWN_UNBOUND
                ) {
                    return BoundActiveLibrary(
                        activeGeneration = active,
                        generation = generation,
                        providerGenerationId =
                            catalog.generationBinding.providerGenerationId,
                        activeCatalog = catalog,
                        requiredSeedProviderBinding = targetedBefore?.let(
                            RequiredSeedProviderBinding::from,
                        ),
                    )
                }
                if (seedStatus == CachedSeedBindingStatus.UNAVAILABLE) {
                    error("The cached current-track binding could not be authenticated")
                }
                Log.w(
                    TAG,
                    "Cached active-library seed binding changed; acquiring a complete provider " +
                        "snapshot before ranking",
                )
                V2ActiveLibraryCatalogStore(context.filesDir).deleteIfMatches(
                    databaseGenerationId = catalog.generationBinding.databaseGenerationId,
                    providerGenerationId = catalog.generationBinding.providerGenerationId,
                )
                publishActiveLibrarySnapshot(null, null)
            }
            val acquisition = processActiveLibraryProviderAcquisitions.incrementAndGet()
            Log.i(
                TAG,
                "Active library provider acquisition=$acquisition reason=service_cache_miss",
            )
            val providerSnapshot = V2PowerampProviderSnapshotAcquirer(context).acquireBlocking()
            val providerGenerationId = requireNotNull(providerSnapshot.libraryGeneration) {
                "Poweramp provider snapshot has no complete generation"
            }
            val snapshotSeed = requiredSeed?.let { seed ->
                requireProviderBinding(providerSnapshot, seed).also { binding ->
                    require(binding.matchingEntry == targetedBefore) {
                        "The displayed Poweramp seed changed during library acquisition"
                    }
                }
            }
            val catalog = V2ActiveLibraryCatalogLoader.load(active, providerSnapshot)
            require(catalog.generationBinding.databaseGenerationId == generation.generationId &&
                catalog.generationBinding.providerGenerationId == providerGenerationId
            ) { "Active recommendation catalog binding is inconsistent" }
            requiredSeed?.let { seed ->
                PowerampHelper.requireFileEntryById(context, seed.powerampFileId).also { entry ->
                    require(entry == snapshotSeed?.matchingEntry && seed.matchesProvider(entry)) {
                        "The displayed Poweramp seed changed while its library was being pinned"
                    }
                }
            }
            V2ActiveLibraryCatalogStore(context.filesDir).write(
                activeGeneration = active,
                catalog = catalog,
            )
            return BoundActiveLibrary(
                active,
                generation,
                providerGenerationId,
                catalog,
                snapshotSeed,
            ).also {
                publishActiveLibrarySnapshot(generation, catalog)
                Log.i(TAG, "Active library cache=miss tracks=${catalog.activeTrackIds.size}")
            }
        }

        private enum class CachedSeedBindingStatus {
            CURRENT,
            KNOWN_UNBOUND,
            STALE,
            UNAVAILABLE,
        }

        /** A stale file ID must not silently turn the displayed seed into another recording. */
        private fun cachedSeedBindingStatus(
            active: V2ResolvedActiveIndexGeneration,
            catalog: V2ActiveLibraryCatalog,
            current: PowerampFileEntry,
        ): CachedSeedBindingStatus {
            if (current.id in catalog.unboundPowerampFileIds) {
                return CachedSeedBindingStatus.KNOWN_UNBOUND
            }
            val trackId = catalog.trackIdForPowerampFile(current.id)
                ?: return CachedSeedBindingStatus.STALE
            val binding = catalog.bindingForTrack(trackId)
                ?: return CachedSeedBindingStatus.STALE
            return runCatching {
                val database = EmbeddingDatabase.open(active.databaseFile)
                val indexed = try {
                    requireNotNull(database.getTrackById(trackId))
                } finally {
                    database.close()
                }
                val receipt = if (
                    binding.evidence == V2ActiveLibraryBindingEvidence.EXACT_V2_RECEIPT_SPAN
                ) {
                    V2ProviderSpanReceiptReader.read(active.databaseFile).receipts
                        .filter { it.trackId == trackId }
                        .singleOrNull()
                } else {
                    null
                }
                if (CurrentPowerampBindingPolicy.matches(
                        binding = binding,
                        indexed = indexed,
                        current = current,
                        receipt = receipt,
                )) {
                    CachedSeedBindingStatus.CURRENT
                } else {
                    CachedSeedBindingStatus.STALE
                }
            }.onFailure { failure ->
                Log.w(TAG, "Could not authenticate the cached current-seed binding", failure)
            }.getOrDefault(CachedSeedBindingStatus.UNAVAILABLE)
        }

        private fun requireProviderBinding(
            snapshot: V2ProviderPathGroupSnapshot,
            seed: WidgetRadioSeedReference,
        ): RequiredSeedProviderBinding {
            val row = snapshot.groups.asSequence()
                .flatMap { it.rows.asSequence() }
                .filter { it.powerampFileId == seed.powerampFileId }
                .singleOrNull()
                ?: throw IllegalStateException(
                    "The displayed Poweramp seed is absent or duplicated in the provider generation",
                )
            val binding = RequiredSeedProviderBinding.from(row)
            require(seed.matchesProvider(binding.matchingEntry)) {
                "The displayed Poweramp seed differs from the complete provider generation"
            }
            return binding
        }

        private fun requireDisplayedDomainBinding(
            library: BoundActiveLibrary,
            displayed: DisplayedRecommendationDomain,
        ) {
            require(displayed.providerGenerationId == library.providerGenerationId) {
                "The Poweramp library changed after this Find Music result was displayed"
            }
            require(displayed.activeTrackCount == library.activeCatalog.activeTrackIds.size) {
                "The active recommendation track count changed after display"
            }
            val orderedActiveIds = library.activeCatalog.activeTrackIds.sorted().toLongArray()
            require(
                ActiveRecommendationDomain.computeOrderedActiveIdsSha256(orderedActiveIds) ==
                    displayed.orderedActiveTrackIdsSha256,
            ) { "The active recommendation membership changed after display" }
        }

        private fun readStableTrackSpanId(
            databaseFile: File,
            trackId: Long,
            embeddingSpecId: String,
        ): String? {
            return StableTrackSpanReceiptReader.read(databaseFile, trackId, embeddingSpecId)
        }

        private inline fun <T> withEmbeddingDatabase(
            databaseFile: File,
            block: (EmbeddingDatabase) -> T,
        ): T {
            val database = EmbeddingDatabase.open(databaseFile)
            return try {
                block(database)
            } finally {
                database.close()
            }
        }

        fun cancelSearch() {
            activeJob?.cancel()
            _uiState.value = RadioUiState.Searching("Cancelling safely...")
        }

        fun resetState() {
            _uiState.value = RadioUiState.Idle
        }

        fun clearHistory() {
            saveScope.launch {
                historyReady.await()
                val dir = historyDir ?: return@launch
                val (snapshot, revision) = synchronized(historyLock) {
                    _sessionHistory.value = emptyList()
                    emptyList<RadioResult>() to ++historyRevision
                }
                writeHistorySnapshot(dir, snapshot, revision)
            }
        }
    }

    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private var embeddingDb: EmbeddingDatabase? = null
    private var engine: RecommendationEngine? = null
    private var openedGeneration: RadioGenerationToken? = null
    private var openedProviderGenerationId: String? = null
    private var openedActiveCatalog: V2ActiveLibraryCatalog? = null
    private var openedAssets: RecommendationAssetFiles? = null
    private var showToasts: Boolean = false
    private var stopJob: Job? = null
    private val serviceOwnerToken = UUID.randomUUID().toString()
    private val requestStore by lazy {
        RadioRequestStore(filesDir, ownerToken = serviceOwnerToken)
    }
    private val widgetIngressStore by lazy { WidgetRadioIngressStore(filesDir) }
    private val widgetIngressMutex = Mutex()
    private val coldStartWaitLock = Any()
    private val coldStartWaitJobs = mutableMapOf<String, Job>()
    private val requestDispatchLock = Any()
    private val requestQueue = ArrayDeque<String>()
    private val queuedRequestIds = mutableSetOf<String>()
    private var currentRequestId: String? = null
    private var latestRequestResult: RadioResult? = null
    private var latestStartId: Int = 0
    @Volatile private var currentWidgetIngressId: String? = null
    @Volatile private var shuttingDown = false

    private sealed interface RequestDispatchAttempt {
        data object NoRequest : RequestDispatchAttempt
        data object DeferredForIndexing : RequestDispatchAttempt
        data class Ready(
            val requestId: String,
            val claim: RadioRequestClaim,
        ) : RequestDispatchAttempt
        data class Invalid(
            val requestId: String,
            val failure: Exception,
        ) : RequestDispatchAttempt
    }

    override fun onCreate() {
        super.onCreate()
        serviceInstance = this
        initHistory(filesDir)
        createNotificationChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        synchronized(requestDispatchLock) {
            latestStartId = maxOf(latestStartId, startId)
        }
        val durableExecutionAction = intent?.action == ACTION_EXECUTE_REQUEST ||
            intent?.action == ACTION_EXECUTE_WIDGET_INGRESS || intent == null
        val explicitRequestId = intent?.getStringExtra(EXTRA_REQUEST_ID)
        if (durableExecutionAction && RecommendationWorkAdmission.isIndexingReserved &&
            (explicitRequestId == null || !isAdmittedRequest(explicitRequestId))
        ) {
            val existingWorkMustDrain = hasRecommendationWorkToDrain()
            val waitsForColdReconciliation = explicitRequestId != null &&
                intent?.action in setOf(ACTION_EXECUTE_REQUEST, ACTION_EXECUTE_WIDGET_INGRESS) &&
                RecommendationWorkAdmission.isReservedBy(
                    RecommendationWorkAdmission.coldReconciliationOwner,
                )
            val message = if (waitsForColdReconciliation) {
                "Reading saved indexing-job ownership before starting radio."
            } else if (existingWorkMustDrain) {
                "Finishing the radio request that started before indexing."
            } else {
                DEFERRED_INDEXING_MESSAGE
            }
            startForeground(NOTIFICATION_ID, createNotification(message))
            _uiState.value = RadioUiState.Loading(message)
            if (waitsForColdReconciliation) {
                stopJob?.cancel()
                stopJob = null
                awaitColdReconciliation(
                    action = requireNotNull(intent?.action),
                    requestId = explicitRequestId,
                )
                return START_REDELIVER_INTENT
            }
            // Cold reconciliation can release between the outer reservation snapshot and the
            // owner check above. If admission is open now, continue through normal dispatch;
            // its atomic gate still resolves any subsequent indexing race.
            if (RecommendationWorkAdmission.isIndexingReserved) {
                if (existingWorkMustDrain) {
                    stopSelfDelayed()
                } else {
                    if (WidgetDeferredStartPolicy.shouldPublishWaitingStatus(
                            durableExecutionAction = durableExecutionAction,
                            explicitRequestId = explicitRequestId,
                            existingWorkMustDrain = existingWorkMustDrain,
                        )
                    ) {
                        updateWidgetRequestStatus(
                            requestId = checkNotNull(explicitRequestId),
                            state = WidgetRadioRequestState.WAITING_FOR_INDEXING,
                            message = "Waiting for on-device indexing to finish",
                        )
                    }
                    stopDeferredStartImmediately(startId)
                }
                return START_NOT_STICKY
            }
        }
        when (intent?.action) {
            ACTION_EXECUTE_REQUEST -> {
                stopJob?.cancel()
                stopJob = null
                startForeground(
                    NOTIFICATION_ID,
                    createNotification("Opening the saved radio request..."),
                )
                val explicitId = intent.getStringExtra(EXTRA_REQUEST_ID)
                serviceScope.launch(Dispatchers.IO) {
                    reconcileWidgetIngresses(explicitId)
                    enqueueRequests(explicitId)
                }
            }
            ACTION_EXECUTE_WIDGET_INGRESS -> {
                stopJob?.cancel()
                stopJob = null
                startForeground(
                    NOTIFICATION_ID,
                    createNotification("Saving the widget request and resolving its Poweramp seed..."),
                )
                val explicitId = intent.getStringExtra(EXTRA_REQUEST_ID)
                serviceScope.launch(Dispatchers.IO) { reconcileWidgetIngresses(explicitId) }
            }
            ACTION_CANCEL -> {
                cancelSearch()
                stopSelfDelayed()
            }
            ACTION_STOP -> {
                stopSelfResult(startId)
            }
            null -> {
                startForeground(
                    NOTIFICATION_ID,
                    createNotification("Resuming saved queue requests from the request journal..."),
                )
                serviceScope.launch(Dispatchers.IO) {
                    reconcileWidgetIngresses(explicitId = null)
                    enqueueRequests(explicitId = null)
                }
            }
        }
        return if (intent?.action == ACTION_EXECUTE_REQUEST ||
            intent?.action == ACTION_EXECUTE_WIDGET_INGRESS || intent == null
        ) {
            START_REDELIVER_INTENT
        } else {
            START_NOT_STICKY
        }
    }

    /** Keep the widget-triggered foreground owner until asynchronous cold startup has settled. */
    private fun awaitColdReconciliation(action: String, requestId: String) {
        val key = "$action:$requestId"
        synchronized(coldStartWaitLock) {
            if (coldStartWaitJobs[key]?.isActive == true) return
            lateinit var waitJob: Job
            waitJob = serviceScope.launch(Dispatchers.IO, start = CoroutineStart.LAZY) {
                try {
                    when (
                        RecommendationWorkAdmission.coldReconciliationState.first { state ->
                            state != RecommendationWorkAdmission.ColdReconciliationState.RUNNING
                        }
                    ) {
                        RecommendationWorkAdmission.ColdReconciliationState.SUCCEEDED -> {
                            if (action == ACTION_EXECUTE_WIDGET_INGRESS) {
                                reconcileWidgetIngresses(requestId)
                            } else {
                                reconcileWidgetIngresses(requestId)
                                enqueueRequests(requestId)
                            }
                        }
                        RecommendationWorkAdmission.ColdReconciliationState.FAILED -> {
                            val message =
                                "Could not verify saved indexing work. Open Start Radio to check it."
                            if (action == ACTION_EXECUTE_WIDGET_INGRESS) {
                                failColdWidgetIngress(requestId, message)
                            } else {
                                _uiState.value = RadioUiState.Error(message)
                            }
                            val latest = synchronized(requestDispatchLock) { latestStartId }
                            stopDeferredStartImmediately(latest)
                        }
                        RecommendationWorkAdmission.ColdReconciliationState.RUNNING -> Unit
                    }
                } finally {
                    synchronized(coldStartWaitLock) {
                        if (coldStartWaitJobs[key] === waitJob) coldStartWaitJobs.remove(key)
                    }
                }
            }
            coldStartWaitJobs[key] = waitJob
            waitJob.start()
        }
    }

    private fun failColdWidgetIngress(requestId: String, message: String) {
        val seed = runCatching { widgetIngressStore.read(requestId)?.expectedSeed }.getOrNull()
        runCatching { widgetIngressStore.markFailed(requestId, message) }
            .onFailure { failure ->
                Log.e(TAG, "Could not terminalize cold widget command $requestId", failure)
            }
        runCatching {
            StartRadioWidgetReceiver.persistRadioFailure(
                this,
                requestId = requestId,
                message = message,
                seed = seed,
            )
        }.onFailure { failure ->
            Log.e(TAG, "Could not publish cold widget terminal status", failure)
        }
        _uiState.value = RadioUiState.Error(message)
    }

    private suspend fun reconcileWidgetIngresses(explicitId: String?) {
        var deferredForIndexing = false
        widgetIngressMutex.withLock {
            val pendingIds = runCatching {
                widgetIngressStore.pendingRecords().map(WidgetRadioIngressRecord::commandId)
            }.onFailure { failure ->
                Log.e(TAG, "Could not inspect saved widget commands", failure)
            }.getOrDefault(emptyList())
            val allIds = buildList {
                explicitId?.let(::add)
                addAll(pendingIds.filterNot { it == explicitId })
            }
            // A request that won admission before indexing must never wait behind older recovery
            // work that indexing deliberately deferred. Keep the explicit delivery first among
            // ordinary recovery records, but always drain every admitted record before them.
            val admittedIds = allIds.filter(::isAdmittedRequest)
            val ids = admittedIds + allIds.filterNot { it in admittedIds }
            for (commandId in ids) {
                val materialized = if (isAdmittedRequest(commandId)) {
                    materializeWidgetIngress(commandId)
                    true
                } else {
                    RecommendationWorkAdmission.runIfRecommendationAllowed {
                        materializeWidgetIngress(commandId)
                        true
                    }
                }
                if (materialized == null) {
                    deferredForIndexing = true
                    updateWidgetRequestStatus(
                        commandId,
                        WidgetRadioRequestState.WAITING_FOR_INDEXING,
                        "Waiting for on-device indexing to finish",
                    )
                }
            }
        }
        if (deferredForIndexing) {
            deferUnclaimedRequestsForIndexing()
            return
        }
        val hasPendingIngress = runCatching { widgetIngressStore.pendingRecords().isNotEmpty() }
            .onFailure { Log.e(TAG, "Could not finish widget-ingress reconciliation", it) }
            .getOrDefault(false)
        if (currentWidgetIngressId == null && !hasPendingIngress) {
            stopSelfDelayed()
        }
    }

    private fun materializeWidgetIngress(commandId: String) {
        currentWidgetIngressId = commandId
        try {
            if (requestStore.hasRecord(commandId)) {
                widgetIngressStore.delete(commandId)
                enqueueRequests(commandId)
                return
            }
            val ingress = widgetIngressStore.read(commandId) ?: run {
                failRequest(commandId)
                Log.w(TAG, "Ignoring missing widget ingress $commandId")
                updateWidgetRequestStatus(
                    commandId,
                    WidgetRadioRequestState.FAILED,
                    "Saved widget radio command is missing; tap again",
                )
                return
            }
            if (ingress.state == WidgetRadioIngressState.FAILED) {
                failRequest(commandId)
                updateWidgetRequestStatus(
                    commandId,
                    WidgetRadioRequestState.FAILED,
                    ingress.terminalDetail ?: "Widget radio command did not complete",
                )
                return
            }

            val reservationOwner = submissionReservation.activeRequestId
            val ownsReservation = reservationOwner == commandId ||
                reserveSubmission(replaceActive = false, requestId = commandId) != null
            require(ownsReservation) { "Another radio request is already starting" }

            val request = buildPinnedRadioRequest(
                context = this,
                config = ingress.config,
                currentSeedReference = ingress.expectedSeed,
                showToasts = true,
                origin = QueueOrigin.WIDGET_RADIO,
                requestId = commandId,
                requestCreatedAtEpochMs = ingress.createdAtEpochMs,
            )
            requestStore.persistIdempotently(request)
            // Durable request publication is the promotion commit. A crash before deletion is
            // recovered by the hasRecord branch above without rebuilding or double mutation.
            widgetIngressStore.delete(commandId)
            val seed = requireNotNull(request.widgetSeedReference())
            StartRadioWidgetReceiver.persistRadioStatus(
                this,
                WidgetRadioStatus(
                    requestId = commandId,
                    seed = seed,
                    state = WidgetRadioRequestState.STARTING,
                    message = "Starting radio for ${seed.displayTitle}",
                    updatedAtEpochMs = System.currentTimeMillis(),
                ),
            )
            enqueueRequests(commandId)
        } catch (failure: Exception) {
            val durableRequestExists = runCatching { requestStore.hasRecord(commandId) }
                .onFailure { inspectionFailure ->
                    Log.e(
                        TAG,
                        "Could not inspect durable widget request $commandId after failure",
                        inspectionFailure,
                    )
                }
                .getOrDefault(false)
            if (durableRequestExists) {
                runCatching { widgetIngressStore.delete(commandId) }
                enqueueRequests(commandId)
                return
            }
            failRequest(commandId)
            val message = widgetFailureMessage(failure)
            val seed = runCatching { widgetIngressStore.read(commandId)?.expectedSeed }.getOrNull()
            val recovery = recoverFailedWidgetIngress(
                store = widgetIngressStore,
                commandId = commandId,
                detail = message,
                publishFailure = { recoveredSeed ->
                    StartRadioWidgetReceiver.persistRadioFailure(
                        this,
                        requestId = commandId,
                        message = message,
                        seed = recoveredSeed ?: seed,
                    )
                },
                scheduleBoundedStop = ::stopSelfDelayed,
            )
            recovery.terminalizationFailure?.let { terminalizationFailure ->
                Log.e(
                    TAG,
                    "Could not terminalize failed widget command $commandId; retained for retry",
                    terminalizationFailure,
                )
            }
            recovery.statusFailure?.let { statusFailure ->
                Log.e(TAG, "Could not publish failed widget status $commandId", statusFailure)
            }
            _uiState.value = RadioUiState.Error(message)
            Log.e(TAG, "Could not materialize widget radio command $commandId", failure)
        } finally {
            currentWidgetIngressId = null
        }
    }

    private fun enqueueRequests(explicitId: String?) {
        val recoveredIds = runCatching { requestStore.recoverableRequestIds() }
            .onFailure { Log.e(TAG, "Could not inspect saved radio requests", it) }
            .getOrDefault(emptyList())
        val ids = recoveredIds.toMutableList()
        if (explicitId != null && explicitId !in ids) ids += explicitId
        synchronized(requestDispatchLock) {
            for (requestId in ids) {
                if (requestId != currentRequestId && queuedRequestIds.add(requestId)) {
                    requestQueue.addLast(requestId)
                }
            }
        }
        dispatchNextRequest()
    }

    private fun dispatchNextRequest() {
        val attempt = RecommendationWorkAdmission.runIfRecommendationAllowed {
            claimNextRequest(admittedOnly = false)
        } ?: claimNextRequest(admittedOnly = true)
        if (attempt is RequestDispatchAttempt.DeferredForIndexing) {
            deferUnclaimedRequestsForIndexing()
            return
        }
        if (attempt is RequestDispatchAttempt.NoRequest) {
            stopSelfDelayed()
            return
        }
        if (attempt is RequestDispatchAttempt.Invalid) {
            val requestId = attempt.requestId
            failRequest(requestId)
            Log.e(TAG, "Invalid durable radio request $requestId", attempt.failure)
            runCatching {
                requestStore.markFailed(
                    requestId,
                    "Saved radio request could not be read. Open Start Radio and try again.",
                )
            }
            _uiState.value = RadioUiState.Error("Saved radio request is invalid")
            updateWidgetRequestStatus(
                requestId,
                WidgetRadioRequestState.FAILED,
                "Saved radio request could not be read. Open Start Radio and try again.",
            )
            releaseRequestAndContinue(requestId)
            return
        }
        attempt as RequestDispatchAttempt.Ready
        val requestId = attempt.requestId
        val claim = attempt.claim

        when (claim) {
            is RadioRequestClaim.Claimed -> dispatchClaimedRequest(claim.request)
            is RadioRequestClaim.ResultReady -> dispatchRecoveredResult(claim.receipt)
            is RadioRequestClaim.AlreadyTerminal -> {
                failRequest(requestId)
                when (claim.state) {
                    RadioRequestStateKind.COMPLETED -> updateWidgetRequestStatus(
                        requestId,
                        WidgetRadioRequestState.SUCCEEDED,
                        "Radio queue completed",
                    )
                    RadioRequestStateKind.FAILED -> updateWidgetRequestStatus(
                        requestId = requestId,
                        state = WidgetRadioRequestState.FAILED,
                        message = "Previous radio request did not complete; tap again",
                        preserveCurrentStates = setOf(
                            WidgetRadioRequestState.PARTIAL_FAILED,
                            WidgetRadioRequestState.CANCELLED,
                        ),
                    )
                    RadioRequestStateKind.INTERRUPTED_NEEDS_RETRY -> {
                        _uiState.value = RadioUiState.Error(
                            "A previous radio request was interrupted. Start it again to retry safely.",
                        )
                        updateWidgetRequestStatus(
                            requestId,
                            WidgetRadioRequestState.FAILED,
                            "Previous radio request was interrupted; tap again to retry safely",
                        )
                    }
                    else -> Unit
                }
                releaseRequestAndContinue(requestId)
            }
            RadioRequestClaim.AlreadyInFlight -> {
                failRequest(requestId)
                Log.i(TAG, "Ignoring duplicate delivery for in-flight request $requestId")
                releaseRequestAndContinue(requestId)
            }
            RadioRequestClaim.Missing -> {
                failRequest(requestId)
                Log.w(TAG, "Ignoring missing durable radio request $requestId")
                updateWidgetRequestStatus(
                    requestId,
                    WidgetRadioRequestState.FAILED,
                    "Saved radio request is missing; tap again",
                )
                releaseRequestAndContinue(requestId)
            }
        }
    }

    /**
     * Claim one request. While indexing is reserved, only a request whose admission predates that
     * reservation may cross the durable claim boundary.
     */
    private fun claimNextRequest(admittedOnly: Boolean): RequestDispatchAttempt {
        val requestId = synchronized(requestDispatchLock) {
            if (currentRequestId != null || activeJob?.isActive == true) {
                return@synchronized null
            }
            val candidate = if (admittedOnly) {
                requestQueue.firstOrNull(::isAdmittedRequest)
            } else {
                requestQueue.firstOrNull()
            }
            if (candidate == null) {
                return if (requestQueue.isEmpty()) {
                    RequestDispatchAttempt.NoRequest
                } else {
                    RequestDispatchAttempt.DeferredForIndexing
                }
            }
            check(requestQueue.remove(candidate)) { "Queued radio request disappeared" }
            queuedRequestIds.remove(candidate)
            currentRequestId = candidate
            activeDurableRequestId = candidate
            candidate
        } ?: return RequestDispatchAttempt.NoRequest

        return try {
            RequestDispatchAttempt.Ready(requestId, requestStore.claim(requestId))
        } catch (failure: Exception) {
            RequestDispatchAttempt.Invalid(requestId, failure)
        }
    }

    /** Keep durable payloads untouched; release this foreground owner until indexing finishes. */
    private fun deferUnclaimedRequestsForIndexing() {
        val (idle, hadDeferredRequest) = synchronized(requestDispatchLock) {
            // Requests admitted before indexing won the gate are existing work. Keep them queued
            // so the indexing drain can observe them through dispatch and terminalization. Only
            // recovery work that never won admission is deferred until indexing releases the gate.
            val admittedQueued = requestQueue.filter(::isAdmittedRequest)
            val hadDeferred = admittedQueued.size != requestQueue.size
            requestQueue.clear()
            requestQueue.addAll(admittedQueued)
            queuedRequestIds.clear()
            queuedRequestIds.addAll(admittedQueued)
            (currentRequestId == null && activeJob?.isActive != true &&
                !hasAdmittedRecommendationWork()) to hadDeferred
        }
        if (!idle) return
        val hasDeferredRequest = hadDeferredRequest || runCatching {
            requestStore.recoverableRequestIds().isNotEmpty() ||
                widgetIngressStore.pendingRecords().isNotEmpty()
        }.onFailure { failure ->
            Log.e(TAG, "Could not verify deferred radio work", failure)
        }.getOrDefault(false)
        if (hasDeferredRequest) {
            _uiState.value = RadioUiState.Loading(DEFERRED_INDEXING_MESSAGE)
            updateNotification(DEFERRED_INDEXING_MESSAGE)
        }
        val startId = synchronized(requestDispatchLock) { latestStartId }
        stopDeferredStartImmediately(startId)
    }

    private fun dispatchRecoveredResult(receipt: DurableRadioResultReceipt) {
        activeJob = serviceScope.launch(Dispatchers.IO) {
            try {
                if (!persistSessionSynchronously(receipt.result)) {
                    _uiState.value = RadioUiState.Error(
                        "Recovered queue result, but session history could not be saved",
                    )
                    updateWidgetRequestStatus(
                        receipt.requestId,
                        WidgetRadioRequestState.PARTIAL_FAILED,
                        "Queue result saved, but history could not be saved",
                    )
                    return@launch
                }
                requestStore.finalizeRecoveredResult(receipt)
                _uiState.value = when (receipt.result.outcome) {
                    RadioSessionOutcome.SUCCEEDED -> RadioUiState.Success(receipt.result)
                    RadioSessionOutcome.PARTIAL_FAILED,
                    RadioSessionOutcome.CANCELLED,
                    null -> RadioUiState.Error(
                        receipt.terminalDetail ?: "Recovered radio request did not complete",
                    )
                }
                updateWidgetForResult(receipt.result)
            } catch (failure: Exception) {
                Log.e(TAG, "Could not reconcile saved radio result ${receipt.requestId}", failure)
                _uiState.value = RadioUiState.Error("Could not recover saved radio result")
                updateWidgetRequestStatus(
                    receipt.requestId,
                    WidgetRadioRequestState.FAILED,
                    "Saved radio result could not be recovered. Open Start Radio to verify the queue.",
                )
            } finally {
                releaseRequestAndContinue(receipt.requestId)
            }
        }
        completeDispatch(receipt.requestId)
    }

    private fun dispatchClaimedRequest(request: DurableRadioRequest) {
        stopJob?.cancel()
        stopJob = null
        latestRequestResult = null
        try {
            when (request.kind) {
                DurableRadioRequestKind.RADIO -> {
                    val payload = requireNotNull(request.radio)
                    showToasts = payload.showToasts
                    performRadio(
                        config = payload.config,
                        seed = payload.seed,
                        generation = request.generation,
                        providerGenerationId = request.providerGenerationId,
                        origin = payload.origin,
                        requestId = request.requestId,
                        requestReferenceEpochSecond = request.createdAtEpochMs / 1_000L,
                    )
                }
                DurableRadioRequestKind.MULTI_SEED_RADIO -> {
                    val payload = requireNotNull(request.multiSeed)
                    showToasts = payload.showToasts
                    performMultiSeedRadio(
                        config = payload.config,
                        seeds = payload.seeds,
                        seedIdentities = payload.seedIdentities,
                        querySpec = payload.querySpec,
                        generation = request.generation,
                        providerGenerationId = request.providerGenerationId,
                        composedContract = payload.composedContract,
                        origin = payload.origin,
                        requestId = request.requestId,
                        requestReferenceEpochSecond = request.createdAtEpochMs / 1_000L,
                    )
                }
                DurableRadioRequestKind.DIRECT_QUEUE -> {
                    val payload = requireNotNull(request.directQueue)
                    showToasts = true
                    performDirectQueue(
                        tracks = payload.tracks,
                        trackIdentities = payload.trackIdentities,
                        resolvedPowerampFileIds = payload.resolvedPowerampFileIds,
                        label = payload.label,
                        origin = payload.origin,
                        placement = payload.placement,
                        findMusicSessionEvidence = payload.findMusicSessionEvidence,
                        findMusicTrackEvidence = payload.findMusicTrackEvidence,
                        generation = request.generation,
                        providerGenerationId = request.providerGenerationId,
                        requestId = request.requestId,
                    )
                }
            }
            completeDispatch(request.requestId)
        } catch (failure: Exception) {
            failRequest(request.requestId)
            Log.e(TAG, "Could not dispatch durable radio request ${request.requestId}", failure)
            runCatching {
                requestStore.markFailed(
                    request.requestId,
                    "Saved radio request could not be started. Open Start Radio and try again.",
                )
            }.onFailure { stateFailure ->
                Log.e(TAG, "Could not fail durable request ${request.requestId}", stateFailure)
            }
            _uiState.value = RadioUiState.Error("Could not start saved radio request")
            updateWidgetRequestStatus(
                request.requestId,
                WidgetRadioRequestState.FAILED,
                "Saved radio request could not be started. Open Start Radio and try again.",
            )
            releaseRequestAndContinue(request.requestId)
        }
    }

    private fun performRadio(
        config: RadioConfig,
        seed: PinnedRadioSeed,
        generation: RadioGenerationToken,
        providerGenerationId: String,
        origin: QueueOrigin,
        requestId: String,
        requestReferenceEpochSecond: Long,
    ) {
        activeJob = serviceScope.launch {
            var terminalFailure: String? = "Radio request ended before completion"
            try {
                val radioStart = System.nanoTime()
                toast("Starting radio...")
                Log.i(TAG, "performRadio: mode=${config.selectionMode.name}, " +
                    "drift=${config.driftEnabled}, numTracks=${config.numTracks}, " +
                    "seed=${seed.identity.embeddedTrackId}, generation=${generation.generationId}")

                val db = getOrCreateDatabase(generation, providerGenerationId)
                val activeCatalog = requireOpenedActiveCatalog(generation, providerGenerationId)
                validatePinnedIdentity(db, generation, seed.identity)
                require(activeCatalog.containsActiveTrack(seed.identity.embeddedTrackId)) {
                    "Pinned seed is no longer in the active recommendation domain"
                }

                val seedTrackId = seed.identity.embeddedTrackId
                val seedDisplayTrack = seed.displayTrack
                val matchType = seed.matchType
                val textSearchSeedFileId = seed.displayTrack.realId.takeIf {
                    origin == QueueOrigin.TEXT_RESULT_RADIO && it > 0L
                }
                val currentTrack =
                    PowerampReceiver.requireProviderVerifiedCurrentTrack(this@RadioService)
                val liveQueueOccurrenceId = currentTrack?.queueOccurrenceId
                val textSearchQueueAnchorId = currentTrack?.realId
                    ?.takeIf { it > 0L && liveQueueOccurrenceId != null }

                val searchPhaseMessage = buildSearchPhaseMessage(config, seedDisplayTrack.title)
                _uiState.value = RadioUiState.Searching(searchPhaseMessage)
                updateNotification(searchPhaseMessage)
                Log.d(TAG, "Config: ${config.selectionMode.name}" +
                    (if (config.driftEnabled) " drift(${config.driftMode.name})" else "") +
                    " lambda=${config.diversityLambda}")

                val eng = getOrCreateEngine(db, generation)
                val tIndices = System.nanoTime()
                eng.ensureIndices(
                    onProgress = { message ->
                        _uiState.value = RadioUiState.Loading(message)
                        updateNotification(message)
                    },
                    requireGraph = config.selectionMode == SelectionMode.RANDOM_WALK,
                )
                val indicesMs = (System.nanoTime() - tIndices) / 1_000_000
                Log.d(TAG, "ensureIndices: ${indicesMs}ms")

                // Preserve the semantic fraction. The engine resolves its internal pool from
                // the exact seed-excluded distinct-identity domain and reports that evidence.
                val resolvedConfig = config.copy(candidatePoolSize = 0)
                var recommendationDomainEvidence: RecommendationDomainEvidence? = null

                _uiState.value = RadioUiState.Searching(searchPhaseMessage)
                updateNotification(searchPhaseMessage)

                // Only MMR has an evolving-query contract. Keep the service path identical to
                // RecommendationEngine so batch modes cannot enter the streaming drift path.
                val effectiveDrift = resolvedConfig.driftEnabled &&
                    resolvedConfig.selectionMode == SelectionMode.MMR
                if (effectiveDrift) {
                    // Drift path: stream search results to UI, queue to Poweramp
                    // in background batches (every 5 tracks). Queue ops are decoupled
                    // from the search loop via a Channel so Poweramp can't stall drift.
                    val streamingTracks = mutableListOf<QueuedTrackResult>()
                    val pendingQueueItems = mutableListOf<Pair<Int, Long>>()
                    val queuedFileIds = ConcurrentHashMap.newKeySet<Long>()
                    val verifiedTrackIndices = ConcurrentHashMap.newKeySet<Int>()
                    val verifiedQueueEntryIdsByTrack = ConcurrentHashMap<Int, Long>()
                    val queueMutations = mutableListOf<QueueMutationResult>()
                    var textSearchSeedQueueEntryId: Long? = null
                    var submittedBatchCount = 0

                    fun streamingResultSnapshot(): RadioResult {
                        val snapshotTracks = streamingTracks.mapIndexed { index, trackResult ->
                            if (index in verifiedTrackIndices && trackResult.status == QueueStatus.PENDING) {
                                trackResult.copy(
                                    status = QueueStatus.QUEUED,
                                    resolvedPowerampQueueId = verifiedQueueEntryIdsByTrack[index],
                                )
                            } else {
                                trackResult
                            }
                        }
                        val delivery = QueueDeliverySummary.fromTracks(
                            origin = origin,
                            requestedCount = snapshotTracks.size,
                            rankedCount = snapshotTracks.size,
                            resolvedCount = snapshotTracks.count { it.status != QueueStatus.NOT_IN_LIBRARY },
                            tracks = snapshotTracks,
                            verificationComplete = false,
                            mutationCount = 0,
                        )
                        return RadioResult(
                            seedTrack = seedDisplayTrack,
                            matchType = matchType,
                            tracks = snapshotTracks,
                            config = resolvedConfig,
                            queuedFileIds = queuedFileIds.toSet(),
                            queueAnchorId = textSearchQueueAnchorId,
                            queueAnchorOccurrenceId = liveQueueOccurrenceId,
                            isComplete = false,
                            totalExpected = resolvedConfig.numTracks,
                            delivery = delivery,
                            requestId = requestId,
                            generation = generation,
                            providerGenerationId = providerGenerationId,
                            seedIdentity = seed.identity,
                            eligibleCandidateIdentityCount = recommendationDomainEvidence
                                ?.seedExcludedCandidateIdentityCount,
                            seedRankingIdentityCount = recommendationDomainEvidence
                                ?.seedExcludedActiveIdentityCount,
                        )
                    }

                    // Selection may stream to the screen, but Poweramp receives the finished
                    // ordered queue in one mutation. Repeated small writes made a normal MMR tap
                    // take tens of seconds without changing recommendation quality.
                    val queueChannel = Channel<List<Pair<Int, Long>>>(Channel.UNLIMITED)
                    val queueJob = launch(Dispatchers.IO) {
                        var isFirst = true
                        var queueMutationAborted = false
                        for (batch in queueChannel) {
                            if (queueMutationAborted) {
                                continue
                            }
                            try {
                                requireRequestBindingCurrent(generation, providerGenerationId)
                                val requestIndices = IntArray(batch.size) { -1 }
                                val requestFileIds = mutableListOf<Long>()
                                val isReplacement = isFirst
                                val deliveryTracks = buildList {
                                    if (isReplacement && textSearchSeedFileId != null) {
                                        add(requireNotNull(db.getTrackById(seedTrackId)))
                                    }
                                    addAll(batch.map { streamingTracks[it.first].track })
                                }
                                val expectedBindings = buildList {
                                    if (isReplacement) textSearchSeedFileId?.let(::add)
                                    addAll(batch.map { it.second })
                                }
                                require(
                                    requireCurrentQueueDeliveryFileIds(
                                        db,
                                        activeCatalog,
                                        deliveryTracks,
                                    ) == expectedBindings,
                                ) { "The drift queue bindings changed before delivery" }
                                val mutation = if (isReplacement) {
                                    isFirst = false
                                    // For text search: prepend seed, anchor current track for Poweramp
                                    textSearchSeedFileId?.let(requestFileIds::add)
                                    for ((batchIndex, item) in batch.withIndex()) {
                                        if (item.second == textSearchSeedFileId) {
                                            continue
                                        }
                                        requestIndices[batchIndex] = requestFileIds.size
                                        requestFileIds += item.second
                                    }
                                    PowerampHelper.replaceQueue(
                                        this@RadioService,
                                        PowerampReceiver.requireProviderVerifiedCurrentTrack(
                                            this@RadioService,
                                        ),
                                        requestFileIds,
                                    )
                                } else {
                                    for ((batchIndex, item) in batch.withIndex()) {
                                        requestIndices[batchIndex] = requestFileIds.size
                                        requestFileIds += item.second
                                    }
                                    PowerampHelper.addTracksToQueue(this@RadioService, requestFileIds)
                                }
                                queueMutations += mutation
                                queuedFileIds += mutation.verifiedFileIds
                                if (isReplacement && textSearchSeedFileId != null) {
                                    textSearchSeedQueueEntryId = mutation.verifiedQueueEntryId(0)
                                }
                                for ((batchIndex, item) in batch.withIndex()) {
                                    val requestIndex = requestIndices[batchIndex]
                                    if (requestIndex >= 0 && mutation.isRequestVerified(requestIndex)) {
                                        verifiedTrackIndices += item.first
                                        mutation.verifiedQueueEntryId(requestIndex)?.let { queueId ->
                                            verifiedQueueEntryIdsByTrack[item.first] = queueId
                                        }
                                    }
                                }
                                if (!mutation.fullyVerified) {
                                    queueMutationAborted = true
                                    Log.e(
                                        TAG,
                                        "Queue readback was incomplete; later drift batches were not mutated",
                                    )
                                }
                                PowerampHelper.reloadData(this@RadioService)
                            } catch (e: Exception) {
                                queueMutationAborted = true
                                Log.e(TAG, "Queue batch failed", e)
                            }
                        }
                    }

                    try {
                        // Clear drift references for new session
                        driftReferences.value = emptyMap()

                        publishStreaming(streamingResultSnapshot())

                        eng.generatePlaylist(
                            seedTrackId = seedTrackId,
                            config = resolvedConfig,
                            requestReferenceEpochSecond = requestReferenceEpochSecond,
                            onProgress = { message ->
                                updateNotification(message)
                            },
                            onRecommendationDomainEvidence = {
                                recommendationDomainEvidence = it
                                publishStreaming(streamingResultSnapshot())
                            },
                            onResult = { similarTrack ->
                                val fileId = requireNotNull(
                                    activeCatalog.powerampFileIdForTrack(similarTrack.track.id),
                                ) {
                                    "Recommendation ${similarTrack.track.id} left the active domain"
                                }

                                // Store drift reference for lazy rank computation
                                similarTrack.driftReferenceEmb?.let { ref ->
                                    driftReferences.value =
                                        driftReferences.value + (similarTrack.track.id to ref)
                                }

                                streamingTracks.add(QueuedTrackResult(
                                    track = similarTrack.track,
                                    similarity = similarTrack.similarity,
                                    similarityToSeed = similarTrack.similarityToSeed,
                                    candidateRank = similarTrack.candidateRank,
                                    seedRank = similarTrack.seedRank,
                                    driftRank = similarTrack.driftRank,
                                    graphTerminalProbability = similarTrack.graphTerminalProbability,
                                    graphExpectedRouteLinks = similarTrack.graphExpectedRouteLinks,
                                    graphHops = similarTrack.graphHops,
                                    status = QueueStatus.PENDING,
                                    provenance = similarTrack.provenance,
                                    resolvedPowerampFileId = fileId,
                                    stableTrackSpanId = readStableTrackSpanId(
                                        db.databaseFile,
                                        similarTrack.track.id,
                                        generation.embeddingSpecId,
                                    ),
                                    mmrEvidence = similarTrack.mmrEvidence,
                                ))

                                publishStreaming(streamingResultSnapshot())

                                // Hold the complete requested queue for one verified mutation.
                                pendingQueueItems += streamingTracks.lastIndex to fileId
                            }
                        )

                        // The queue consumer revalidates every cached file ID before mutation.
                        if (pendingQueueItems.isNotEmpty()) {
                            queueChannel.trySend(pendingQueueItems.toList())
                            submittedBatchCount++
                        }
                        queueChannel.close()
                        queueJob.join()
                    } finally {
                        withContext(NonCancellable) {
                            queueChannel.close()
                            queueJob.cancelAndJoin()
                        }
                    }

                    val expectedFileIds = mutableListOf<Long>()
                    val expectedQueueEntryIds = mutableMapOf<Int, Long>()
                    textSearchSeedFileId?.let {
                        expectedFileIds += it
                        textSearchSeedQueueEntryId?.let { queueId ->
                            expectedQueueEntryIds[0] = queueId
                        }
                    }
                    val requestIndexByTrack = IntArray(streamingTracks.size) { -1 }
                    streamingTracks.forEachIndexed { index, trackResult ->
                        val fileId = trackResult.resolvedPowerampFileId ?: return@forEachIndexed
                        if (fileId != textSearchSeedFileId) {
                            requestIndexByTrack[index] = expectedFileIds.size
                            verifiedQueueEntryIdsByTrack[index]?.let { queueId ->
                                expectedQueueEntryIds[expectedFileIds.size] = queueId
                            }
                            expectedFileIds += fileId
                        }
                    }
                    val finalVerification = withContext(Dispatchers.IO) {
                        PowerampHelper.verifyCurrentQueuePlan(
                            context = this@RadioService,
                            kind = QueueMutationKind.REPLACE,
                            preservedAnchorQueueId =
                                queueMutations.firstOrNull()?.preservedAnchorQueueId,
                            expectedFileIds = expectedFileIds,
                            expectedQueueEntryIdsByRequestIndex = expectedQueueEntryIds,
                        )
                    }
                    val finalTracks = streamingTracks.mapIndexed { index, trackResult ->
                        val requestIndex = requestIndexByTrack[index]
                        when {
                            trackResult.status == QueueStatus.NOT_IN_LIBRARY -> trackResult
                            requestIndex >= 0 && finalVerification.isRequestVerified(requestIndex) ->
                                trackResult.copy(
                                    status = QueueStatus.QUEUED,
                                    resolvedPowerampQueueId =
                                        finalVerification.verifiedQueueEntryId(requestIndex),
                                )
                            else -> trackResult.copy(
                                status = QueueStatus.QUEUE_FAILED,
                                resolvedPowerampQueueId = null,
                            )
                        }
                    }
                    val resolvedCount = streamingTracks.count { it.status == QueueStatus.PENDING }
                    val delivery = QueueDeliverySummary.fromTracks(
                        origin = origin,
                        requestedCount = streamingTracks.size,
                        rankedCount = streamingTracks.size,
                        resolvedCount = resolvedCount,
                        tracks = finalTracks,
                        verificationComplete = expectedFileIds.isNotEmpty() &&
                            queueMutations.size == submittedBatchCount &&
                            queueMutations.all { it.fullyVerified } &&
                            finalVerification.fullyVerified,
                        mutationCount = queueMutations.size,
                        unexpectedObservedCount = finalVerification.unexpectedObservedCount,
                    )
                    val finalResult = RadioResult(
                        seedTrack = seedDisplayTrack,
                        matchType = matchType,
                        tracks = finalTracks,
                        config = resolvedConfig,
                        queuedFileIds = finalVerification.verifiedFileIds.toSet(),
                        queueAnchorId = finalVerification.preservedAnchorFileId,
                        queueAnchorOccurrenceId = finalVerification.preservedAnchorQueueId,
                        isComplete = true,
                        totalExpected = resolvedConfig.numTracks,
                        delivery = delivery,
                        requestId = requestId,
                        generation = generation,
                        providerGenerationId = providerGenerationId,
                        seedIdentity = seed.identity,
                        eligibleCandidateIdentityCount = recommendationDomainEvidence
                            ?.seedExcludedCandidateIdentityCount,
                        seedRankingIdentityCount = recommendationDomainEvidence
                            ?.seedExcludedActiveIdentityCount,
                    )

                    publishSuccess(finalResult)

                    PowerampHelper.reloadData(this@RadioService)

                    val message = buildQueueResultMessage(finalResult)
                    updateNotification(message)
                    val totalMs = (System.nanoTime() - radioStart) / 1_000_000
                    Log.i(TAG, "TIMING: radio_drift total=${totalMs}ms, " +
                        "${finalResult.queuedCount} verified / ${finalResult.resolvedCount} resolved / " +
                        "${finalResult.rankedCount} ranked / ${finalResult.requestedCount} requested")
                    toast(message)

                } else {
                    // Non-drift path: batch search
                    _uiState.value = RadioUiState.Searching(searchPhaseMessage)

                    var graphExplorationEvidence: GraphExplorationEvidence? = null
                    var dppSelectionEvidence:
                        com.powerampstartradio.similarity.DppSelectionEvidence? = null
                    var uniformShuffleIdentityEvidence:
                        com.powerampstartradio.similarity.UniformShuffleIdentityEvidence? = null
                    val similarTracks = eng.generatePlaylist(
                        seedTrackId = seedTrackId,
                        config = resolvedConfig,
                        requestReferenceEpochSecond = requestReferenceEpochSecond,
                        onProgress = { message ->
                            _uiState.value = RadioUiState.Searching(message)
                            updateNotification(message)
                        },
                        onGraphExplorationEvidence = { graphExplorationEvidence = it },
                        onDppSelectionEvidence = { dppSelectionEvidence = it },
                        onUniformShuffleIdentityEvidence = { uniformShuffleIdentityEvidence = it },
                        onRecommendationDomainEvidence = {
                            recommendationDomainEvidence = it
                        },
                    )

                    if (similarTracks.isEmpty()) {
                        _uiState.value = RadioUiState.Error("No similar tracks found")
                        updateNotification("No similar tracks found")
                        toast("No similar tracks found")
                        stopSelfDelayed()
                        return@launch
                    }

                    _uiState.value = RadioUiState.Searching(
                        "Resolving ${similarTracks.size} ranked recordings to exact Poweramp file IDs...",
                    )
                    updateNotification(
                        "Resolving ${similarTracks.size} ranked recordings to exact Poweramp file IDs...",
                    )

                    val tMap = System.nanoTime()
                    val deliveryTracks = buildList {
                        if (textSearchSeedFileId != null) {
                            add(requireNotNull(db.getTrackById(seedTrackId)))
                        }
                        addAll(similarTracks.map(SimilarTrack::track))
                    }
                    val catalogFileIds = deliveryTracks.map { track ->
                        requireNotNull(activeCatalog.powerampFileIdForTrack(track.id)) {
                            "Recommendation ${track.id} left the active domain"
                        }
                    }
                    val recommendationOffset = if (textSearchSeedFileId == null) 0 else 1
                    if (textSearchSeedFileId != null) {
                        require(catalogFileIds.first() == textSearchSeedFileId) {
                            "The displayed text-result seed changed before queue delivery"
                        }
                    }
                    val fileIds = catalogFileIds.drop(recommendationOffset)
                    val mappedTracks = similarTracks.zip(fileIds) { similarTrack, fileId ->
                        TrackMatcher.MappedTrack(similarTrack, fileId)
                    }
                    val mapMs = (System.nanoTime() - tMap) / 1_000_000
                    Log.d(TAG, "Track mapping: ${similarTracks.size} → ${fileIds.size} file IDs in ${mapMs}ms")

                    // For text search: prepend seed to recommendations.
                    // If in queue, anchor current track at pos 0 for Poweramp stability.
                    // Queue: [anchor?, seed, rec1, rec2, ...]
                    val allFileIds = mutableListOf<Long>()
                    textSearchSeedFileId?.let(allFileIds::add)
                    val requestIndexByMappedRow = IntArray(mappedTracks.size) { -1 }
                    for ((index, mapped) in mappedTracks.withIndex()) {
                        val fileId = mapped.fileId ?: continue
                        if (fileId == textSearchSeedFileId) continue
                        requestIndexByMappedRow[index] = allFileIds.size
                        allFileIds += fileId
                    }

                    val pendingTrackResults = mappedTracks.map { mapped ->
                        QueuedTrackResult(
                            track = mapped.similarTrack.track,
                            similarity = mapped.similarTrack.similarity,
                            similarityToSeed = mapped.similarTrack.similarityToSeed,
                            candidateRank = mapped.similarTrack.candidateRank,
                            seedRank = mapped.similarTrack.seedRank,
                            driftRank = mapped.similarTrack.driftRank,
                            graphTerminalProbability = mapped.similarTrack.graphTerminalProbability,
                            graphExpectedRouteLinks = mapped.similarTrack.graphExpectedRouteLinks,
                            graphHops = mapped.similarTrack.graphHops,
                            status = if (mapped.fileId == null) {
                                QueueStatus.NOT_IN_LIBRARY
                            } else {
                                QueueStatus.PENDING
                            },
                            provenance = mapped.similarTrack.provenance,
                            resolvedPowerampFileId = mapped.fileId,
                            stableTrackSpanId = readStableTrackSpanId(
                                db.databaseFile,
                                mapped.similarTrack.track.id,
                                generation.embeddingSpecId,
                            ),
                            mmrEvidence = mapped.similarTrack.mmrEvidence,
                        )
                    }
                    publishStreaming(
                        RadioResult(
                            seedTrack = seedDisplayTrack,
                            matchType = matchType,
                            tracks = pendingTrackResults,
                            config = resolvedConfig,
                            queueAnchorId = textSearchQueueAnchorId,
                            queueAnchorOccurrenceId = liveQueueOccurrenceId,
                            isComplete = false,
                            totalExpected = resolvedConfig.numTracks,
                            delivery = QueueDeliverySummary.fromTracks(
                                origin = origin,
                                requestedCount = pendingTrackResults.size,
                                rankedCount = pendingTrackResults.size,
                                resolvedCount = fileIds.size,
                                tracks = pendingTrackResults,
                                verificationComplete = false,
                                mutationCount = 0,
                            ),
                            graphExploration = graphExplorationEvidence,
                            dppSelection = dppSelectionEvidence,
                            requestId = requestId,
                            generation = generation,
                            providerGenerationId = providerGenerationId,
                            seedIdentity = seed.identity,
                            uniformShuffleIdentity = uniformShuffleIdentityEvidence,
                            eligibleCandidateIdentityCount = recommendationDomainEvidence
                                ?.seedExcludedCandidateIdentityCount,
                            seedRankingIdentityCount = recommendationDomainEvidence
                                ?.seedExcludedActiveIdentityCount,
                        ),
                    )

                    val queueMessage =
                        "Replacing the upcoming Poweramp queue with ${allFileIds.size} tracks..."
                    _uiState.value = RadioUiState.Searching(queueMessage)
                    updateNotification(queueMessage)
                    val tQueue = System.nanoTime()
                    val queueMutation = withContext(Dispatchers.IO) {
                        requireRequestBindingCurrent(generation, providerGenerationId)
                        require(
                            requireCurrentQueueDeliveryFileIds(db, activeCatalog, deliveryTracks) ==
                                catalogFileIds,
                        ) { "The recommendation-to-Poweramp bindings changed before delivery" }
                        allFileIds.takeIf { it.isNotEmpty() }?.let {
                            PowerampHelper.replaceQueue(
                                this@RadioService,
                                PowerampReceiver.requireProviderVerifiedCurrentTrack(
                                    this@RadioService,
                                ),
                                it,
                            )
                        }
                    }
                    val queueMs = (System.nanoTime() - tQueue) / 1_000_000
                    Log.d(TAG, "Poweramp queue: ${queueMutation?.verifiedCount ?: 0}/${allFileIds.size} " +
                        "verified in ${queueMs}ms")
                    val finalVerification = withContext(Dispatchers.IO) {
                        PowerampHelper.verifyCurrentQueuePlan(
                            context = this@RadioService,
                            kind = QueueMutationKind.REPLACE,
                            preservedAnchorQueueId = queueMutation?.preservedAnchorQueueId,
                            expectedFileIds = allFileIds,
                            expectedQueueEntryIdsByRequestIndex =
                                queueMutation?.verifiedQueueEntryIdsByRequestIndex.orEmpty(),
                        )
                    }
                    val queuedFileIds = finalVerification.verifiedFileIds.toSet()

                    val trackResults = pendingTrackResults.mapIndexed { index, pending ->
                        val requestIndex = requestIndexByMappedRow[index]
                        when {
                            pending.resolvedPowerampFileId == null -> pending
                            requestIndex >= 0 && finalVerification.isRequestVerified(requestIndex) ->
                                pending.copy(
                                    status = QueueStatus.QUEUED,
                                    resolvedPowerampQueueId =
                                        finalVerification.verifiedQueueEntryId(requestIndex),
                                )
                            else -> pending.copy(
                                status = QueueStatus.QUEUE_FAILED,
                                resolvedPowerampQueueId = null,
                            )
                        }
                    }

                    val delivery = QueueDeliverySummary.fromTracks(
                        origin = origin,
                        requestedCount = trackResults.size,
                        rankedCount = similarTracks.size,
                        resolvedCount = fileIds.size,
                        tracks = trackResults,
                        verificationComplete = allFileIds.isNotEmpty() &&
                            queueMutation?.fullyVerified == true &&
                            finalVerification.fullyVerified,
                        mutationCount = if (queueMutation == null) 0 else 1,
                        unexpectedObservedCount = finalVerification.unexpectedObservedCount,
                    )

                    val radioResult = RadioResult(
                        seedTrack = seedDisplayTrack,
                        matchType = matchType,
                        tracks = trackResults,
                        config = resolvedConfig,
                        queuedFileIds = queuedFileIds,
                        queueAnchorId = finalVerification.preservedAnchorFileId,
                        queueAnchorOccurrenceId = finalVerification.preservedAnchorQueueId,
                        totalExpected = resolvedConfig.numTracks,
                        delivery = delivery,
                        graphExploration = graphExplorationEvidence,
                        dppSelection = dppSelectionEvidence,
                        requestId = requestId,
                        generation = generation,
                        providerGenerationId = providerGenerationId,
                        seedIdentity = seed.identity,
                        uniformShuffleIdentity = uniformShuffleIdentityEvidence,
                        eligibleCandidateIdentityCount = recommendationDomainEvidence
                            ?.seedExcludedCandidateIdentityCount,
                        seedRankingIdentityCount = recommendationDomainEvidence
                            ?.seedExcludedActiveIdentityCount,
                    )

                    publishSuccess(radioResult)

                    PowerampHelper.reloadData(this@RadioService)


                    val message = buildQueueResultMessage(radioResult)
                    updateNotification(message)
                    val totalMs = (System.nanoTime() - radioStart) / 1_000_000
                    Log.i(TAG, "TIMING: radio_batch total=${totalMs}ms, " +
                        "${radioResult.queuedCount} verified / ${radioResult.resolvedCount} resolved / " +
                        "${radioResult.rankedCount} ranked / ${radioResult.requestedCount} requested")
                    toast(message)
                }

                terminalFailure = null
                stopSelfDelayed()

            } catch (e: CancellationException) {
                terminalFailure = "Radio request was cancelled"
                Log.d(TAG, "Radio search cancelled")
                stopSelfDelayed()
            } catch (e: Exception) {
                Log.e(TAG, "Error starting radio", e)
                val listenerMessage = when {
                    e is SecurityException ->
                        "Poweramp library access is unavailable. Recheck app access and try again."
                    config.selectionMode == SelectionMode.RANDOM_WALK ->
                        "Graph Explorer did not finish. Check session history before retrying."
                    else ->
                        "Radio did not finish. Check session history before retrying."
                }
                terminalFailure = listenerMessage
                _uiState.value = RadioUiState.Error(listenerMessage)
                updateNotification(listenerMessage)
                toast(listenerMessage)
                stopSelfDelayed()
            } finally {
                finishRequest(requestId, terminalFailure)
            }
        }
    }

    private fun performMultiSeedRadio(
        config: RadioConfig,
        seeds: List<SeedSpec>,
        seedIdentities: List<RadioSeedIdentity?>,
        querySpec: FindMusicQuerySpec,
        generation: RadioGenerationToken,
        providerGenerationId: String,
        composedContract: ComposedRadioContract,
        origin: QueueOrigin,
        requestId: String,
        requestReferenceEpochSecond: Long,
    ) {
        activeJob = serviceScope.launch {
            var terminalFailure: String? = "All-of radio request ended before completion"
            try {
                val radioStart = System.nanoTime()
                toast("All-of radio...")
                Log.i(TAG, "performMultiSeedRadio: ${seeds.size} ingredients, numTracks=${config.numTracks}")

                val db = getOrCreateDatabase(generation, providerGenerationId)
                val activeCatalog = requireOpenedActiveCatalog(generation, providerGenerationId)
                seedIdentities.filterNotNull().forEach { identity ->
                    validatePinnedIdentity(db, generation, identity)
                    require(activeCatalog.containsActiveTrack(identity.embeddedTrackId)) {
                        "A pinned ingredient is outside the active recommendation domain"
                    }
                }

                val eng = getOrCreateEngine(db, generation)

                eng.ensureIndices(onProgress = { message ->
                    _uiState.value = RadioUiState.Loading(message)
                    updateNotification(message)
                })

                val rankingMessage =
                    "Preparing to rank up to ${config.numTracks} queue tracks from " +
                        "${seeds.size} All-of ingredients..."
                _uiState.value = RadioUiState.Searching(rankingMessage)
                updateNotification(rankingMessage)

                val playlist = eng.generateComposedAllOfPlaylist(
                    seeds = seeds,
                    querySpec = querySpec,
                    config = config,
                    requestReferenceEpochSecond = requestReferenceEpochSecond,
                    onProgress = { message ->
                        _uiState.value = RadioUiState.Searching(message)
                        updateNotification(message)
                    }
                )
                val similarTracks = playlist.tracks

                if (similarTracks.isEmpty()) {
                    _uiState.value = RadioUiState.Error("No similar tracks found")
                    toast("No similar tracks found")
                    stopSelfDelayed()
                    return@launch
                }

                _uiState.value = RadioUiState.Searching(
                    "Resolving ${similarTracks.size} ranked recordings to exact Poweramp file IDs...",
                )
                updateNotification(
                    "Resolving ${similarTracks.size} ranked recordings to exact Poweramp file IDs...",
                )

                val recommendationTracks = similarTracks.map(SimilarTrack::track)
                val fileIds = recommendationTracks.map { track ->
                    requireNotNull(activeCatalog.powerampFileIdForTrack(track.id)) {
                        "All-of recommendation ${track.id} left the active domain"
                    }
                }
                val mappedTracks = similarTracks.zip(fileIds) { similarTrack, fileId ->
                    TrackMatcher.MappedTrack(similarTrack, fileId)
                }

                val displayLabel = "All of: ${querySpec.displayLabel}"
                val seedDisplayTrack = PowerampTrack(
                    realId = -1L,
                    title = displayLabel,
                    artist = null,
                    album = null,
                    durationMs = 0,
                    path = "",
                )

                // Track.ID is an anchor only when CAT_URI proves the live category is Queue.
                val currentTrack =
                    PowerampReceiver.requireProviderVerifiedCurrentTrack(this@RadioService)
                val liveQueueOccurrenceId = currentTrack?.queueOccurrenceId
                val queueAnchorId = currentTrack?.realId
                    ?.takeIf { it > 0L && liveQueueOccurrenceId != null }

                val pendingTrackResults = mappedTracks.map { mapped ->
                    QueuedTrackResult(
                        track = mapped.similarTrack.track,
                        similarity = mapped.similarTrack.similarity,
                        similarityToSeed = mapped.similarTrack.similarityToSeed,
                        status = if (mapped.fileId == null) {
                            QueueStatus.NOT_IN_LIBRARY
                        } else {
                            QueueStatus.PENDING
                        },
                        resolvedPowerampFileId = mapped.fileId,
                        stableTrackSpanId = readStableTrackSpanId(
                            db.databaseFile,
                            mapped.similarTrack.track.id,
                            generation.embeddingSpecId,
                        ),
                        composedEvidence = mapped.similarTrack.composedEvidence,
                    )
                }
                publishStreaming(
                    RadioResult(
                        seedTrack = seedDisplayTrack,
                        matchType = TrackMatcher.MatchType.COMPOSED_QUERY,
                        tracks = pendingTrackResults,
                        config = config,
                        queueAnchorId = queueAnchorId,
                        queueAnchorOccurrenceId = liveQueueOccurrenceId,
                        isComplete = false,
                        totalExpected = config.numTracks,
                        delivery = QueueDeliverySummary.fromTracks(
                            origin = origin,
                            requestedCount = pendingTrackResults.size,
                            rankedCount = pendingTrackResults.size,
                            resolvedCount = fileIds.size,
                            tracks = pendingTrackResults,
                            verificationComplete = false,
                            mutationCount = 0,
                        ),
                        requestId = requestId,
                        generation = generation,
                        providerGenerationId = providerGenerationId,
                        composedContract = composedContract,
                        composedQuerySpec = querySpec,
                        stableResultReduction = playlist.stableResultReduction,
                        composedObjectiveRankingDomainCount =
                            playlist.objectiveRankingDomainCount,
                    ),
                )

                val queueMessage =
                    "Replacing the upcoming Poweramp queue with ${fileIds.size} tracks..."
                _uiState.value = RadioUiState.Searching(queueMessage)
                updateNotification(queueMessage)

                val queueMutation = withContext(Dispatchers.IO) {
                    requireRequestBindingCurrent(generation, providerGenerationId)
                    require(
                        requireCurrentQueueDeliveryFileIds(
                            db,
                            activeCatalog,
                            recommendationTracks,
                        ) == fileIds,
                    ) { "The All-of recommendation bindings changed before delivery" }
                    fileIds.takeIf { it.isNotEmpty() }?.let {
                        PowerampHelper.replaceQueue(
                            this@RadioService,
                            PowerampReceiver.requireProviderVerifiedCurrentTrack(
                                this@RadioService,
                            ),
                            it,
                        )
                    }
                }
                val finalVerification = withContext(Dispatchers.IO) {
                    PowerampHelper.verifyCurrentQueuePlan(
                        context = this@RadioService,
                        kind = QueueMutationKind.REPLACE,
                        preservedAnchorQueueId = queueMutation?.preservedAnchorQueueId,
                        expectedFileIds = fileIds,
                        expectedQueueEntryIdsByRequestIndex =
                            queueMutation?.verifiedQueueEntryIdsByRequestIndex.orEmpty(),
                    )
                }
                val queuedFileIds = finalVerification.verifiedFileIds.toSet()

                var resolvedIndex = 0
                val trackResults = pendingTrackResults.map { pending ->
                    if (pending.resolvedPowerampFileId == null) {
                        pending
                    } else {
                        val requestIndex = resolvedIndex++
                        if (finalVerification.isRequestVerified(requestIndex)) {
                            pending.copy(
                                status = QueueStatus.QUEUED,
                                resolvedPowerampQueueId =
                                    finalVerification.verifiedQueueEntryId(requestIndex),
                            )
                        } else {
                            pending.copy(
                                status = QueueStatus.QUEUE_FAILED,
                                resolvedPowerampQueueId = null,
                            )
                        }
                    }
                }

                val delivery = QueueDeliverySummary.fromTracks(
                    origin = origin,
                    requestedCount = trackResults.size,
                    rankedCount = similarTracks.size,
                    resolvedCount = fileIds.size,
                    tracks = trackResults,
                    verificationComplete = fileIds.isNotEmpty() &&
                        queueMutation?.fullyVerified == true &&
                        finalVerification.fullyVerified,
                    mutationCount = if (queueMutation == null) 0 else 1,
                    unexpectedObservedCount = finalVerification.unexpectedObservedCount,
                )
                val radioResult = RadioResult(
                    seedTrack = seedDisplayTrack,
                    matchType = TrackMatcher.MatchType.COMPOSED_QUERY,
                    tracks = trackResults,
                    config = config,
                    queuedFileIds = queuedFileIds,
                    queueAnchorId = finalVerification.preservedAnchorFileId,
                    queueAnchorOccurrenceId = finalVerification.preservedAnchorQueueId,
                    totalExpected = config.numTracks,
                    delivery = delivery,
                    requestId = requestId,
                    generation = generation,
                    providerGenerationId = providerGenerationId,
                    composedContract = composedContract,
                    composedQuerySpec = querySpec,
                    stableResultReduction = playlist.stableResultReduction,
                    composedObjectiveRankingDomainCount = playlist.objectiveRankingDomainCount,
                )

                publishSuccess(radioResult)
                PowerampHelper.reloadData(this@RadioService)

                val message = buildQueueResultMessage(radioResult)
                updateNotification(message)
                val totalMs = (System.nanoTime() - radioStart) / 1_000_000
                Log.i(TAG, "TIMING: radio_multiseed total=${totalMs}ms, " +
                    "${radioResult.queuedCount} verified / ${radioResult.resolvedCount} resolved / " +
                    "${radioResult.rankedCount} ranked / ${radioResult.requestedCount} requested")
                toast(message)

                terminalFailure = null
                stopSelfDelayed()
            } catch (e: CancellationException) {
                terminalFailure = "All-of radio request was cancelled"
                Log.d(TAG, "All-of radio cancelled")
                stopSelfDelayed()
            } catch (e: Exception) {
                Log.e(TAG, "Error in All-of radio", e)
                val listenerMessage = if (e is SecurityException) {
                    "Poweramp library access is unavailable. Recheck app access and try again."
                } else {
                    "All of did not finish. Check session history before retrying."
                }
                terminalFailure = listenerMessage
                _uiState.value = RadioUiState.Error(listenerMessage)
                updateNotification(listenerMessage)
                toast(listenerMessage)
                stopSelfDelayed()
            } finally {
                finishRequest(requestId, terminalFailure)
            }
        }
    }

    /**
     * Directly queue a pre-computed list of tracks into Poweramp.
     * Uses the request-bound active catalog to preserve exact Poweramp occurrences, then queues.
     */
    private fun performDirectQueue(
        tracks: List<EmbeddedTrack>,
        trackIdentities: List<RadioSeedIdentity>,
        resolvedPowerampFileIds: List<Long?>,
        label: String,
        origin: QueueOrigin,
        placement: DirectQueuePlacement,
        findMusicSessionEvidence: FindMusicSessionEvidence?,
        findMusicTrackEvidence: List<FindMusicTrackEvidence>?,
        generation: RadioGenerationToken,
        providerGenerationId: String,
        requestId: String,
    ) {
        activeJob = serviceScope.launch {
            var terminalFailure: String? = "Direct queue request ended before completion"
            try {
                val radioResult = kotlinx.coroutines.withContext(Dispatchers.IO) {
                    val pinMessage =
                        "Revalidating ${tracks.size} pinned tracks and their exact Poweramp file IDs..."
                    _uiState.value = RadioUiState.Searching(pinMessage)
                    updateNotification(pinMessage)
                    val db = getOrCreateDatabase(generation, providerGenerationId)
                    val activeCatalog = requireOpenedActiveCatalog(
                        generation,
                        providerGenerationId,
                    )
                    val exactTracks = trackIdentities.mapIndexed { index, identity ->
                        validatePinnedIdentity(db, generation, identity)
                        requireNotNull(db.getTrackById(identity.embeddedTrackId)).also { exact ->
                            require(exact == tracks[index]) {
                                "Pinned direct-queue track ${index + 1} changed inside its generation"
                            }
                        }
                    }
                    val currentPowerampFileIds = CurrentPowerampResolutionPolicy.requireUnchanged(
                        pinnedFileIds = resolvedPowerampFileIds,
                        currentFileIds = exactTracks.map { track ->
                            activeCatalog.powerampFileIdForTrack(track.id)
                        },
                    )
                    Log.i(TAG, "DIRECT_QUEUE: queueing ${tracks.size} tracks, label='$label'")
                    for ((i, t) in tracks.withIndex()) {
                        Log.d(TAG, "DIRECT_QUEUE: [$i] ${t.artist} - ${t.title} (id=${t.id})")
                    }

                    val fileIds = currentPowerampFileIds.mapIndexed { index, fileId ->
                        requireNotNull(fileId) {
                            "Direct-queue recording ${index + 1} has no active Poweramp occurrence"
                        }
                    }

                    // Only a Queue CAT_URI makes Track.ID an exact queue anchor.
                    val currentTrack = if (placement == DirectQueuePlacement.REPLACE_UPCOMING) {
                        runCatching {
                            PowerampReceiver.requireProviderVerifiedCurrentTrack(this@RadioService)
                        }.onFailure { error ->
                            Log.w(
                                TAG,
                                "Find Music will replace the queue without a current-track anchor",
                                error,
                            )
                        }.getOrNull()
                    } else {
                        null
                    }
                    val liveQueueOccurrenceId = currentTrack?.queueOccurrenceId
                    val queueAnchorId = currentTrack?.realId
                        ?.takeIf { it > 0L && liveQueueOccurrenceId != null }

                    val syntheticSeed = PowerampTrack(
                        realId = -1L,
                        title = label,
                        artist = null,
                        album = null,
                        durationMs = 0,
                        path = null,
                    )
                    val pendingTrackResults = tracks.mapIndexed { index, track ->
                        val fileId = currentPowerampFileIds[index]
                        val rankingEvidence = findMusicTrackEvidence?.get(index)
                        QueuedTrackResult(
                            track = track,
                            similarity = rankingEvidence?.resultScore ?: 0f,
                            similarityToSeed = 0f,
                            status = if (fileId == null) {
                                QueueStatus.NOT_IN_LIBRARY
                            } else {
                                QueueStatus.PENDING
                            },
                            resolvedPowerampFileId = fileId,
                            stableTrackSpanId = trackIdentities[index].stableTrackSpanId,
                            findMusicEvidence = rankingEvidence,
                        )
                    }
                    publishStreaming(
                        RadioResult(
                            seedTrack = syntheticSeed,
                            matchType = TrackMatcher.MatchType.NOT_APPLICABLE,
                            tracks = pendingTrackResults,
                            queueAnchorId = queueAnchorId,
                            queueAnchorOccurrenceId = liveQueueOccurrenceId,
                            isComplete = false,
                            totalExpected = tracks.size,
                            isDirectQueue = true,
                            delivery = QueueDeliverySummary.fromTracks(
                                origin = origin,
                                requestedCount = tracks.size,
                                rankedCount = tracks.size,
                                resolvedCount = fileIds.size,
                                tracks = pendingTrackResults,
                                verificationComplete = false,
                                mutationCount = 0,
                            ),
                            requestId = requestId,
                            generation = generation,
                            providerGenerationId = providerGenerationId,
                            directQueuePlacement = placement,
                            findMusicSessionEvidence = findMusicSessionEvidence,
                        ),
                    )

                    val queueMessage = when (placement) {
                        DirectQueuePlacement.REPLACE_UPCOMING ->
                            "Replacing Poweramp's upcoming queue with ${fileIds.size} tracks..."
                        DirectQueuePlacement.APPEND ->
                            "Appending ${fileIds.size} tracks to Poweramp's queue..."
                    }
                    _uiState.value = RadioUiState.Searching(queueMessage)
                    updateNotification(queueMessage)
                    val mutation = fileIds.takeIf { it.isNotEmpty() }?.let { exactFileIds ->
                        requireRequestBindingCurrent(generation, providerGenerationId)
                        require(
                            requireCurrentQueueDeliveryFileIds(
                                db,
                                activeCatalog,
                                exactTracks,
                            ) == exactFileIds,
                        ) { "The pinned Poweramp recordings changed before queue delivery" }
                        when (placement) {
                            DirectQueuePlacement.REPLACE_UPCOMING -> PowerampHelper.replaceQueue(
                                this@RadioService,
                                PowerampReceiver.requireProviderVerifiedCurrentTrack(
                                    this@RadioService,
                                ),
                                exactFileIds,
                            )
                            DirectQueuePlacement.APPEND -> PowerampHelper.addTracksToQueue(
                                this@RadioService,
                                exactFileIds,
                            )
                        }
                    }
                    if (mutation != null) PowerampHelper.reloadData(this@RadioService)
                    val verificationMessage =
                        "Verifying ${fileIds.size} requested tracks in Poweramp's queue..."
                    _uiState.value = RadioUiState.Searching(verificationMessage)
                    updateNotification(verificationMessage)
                    val finalVerification = PowerampHelper.verifyCurrentQueuePlan(
                        context = this@RadioService,
                        kind = when (placement) {
                            DirectQueuePlacement.REPLACE_UPCOMING -> QueueMutationKind.REPLACE
                            DirectQueuePlacement.APPEND -> QueueMutationKind.APPEND
                        },
                        preservedAnchorQueueId = if (
                            placement == DirectQueuePlacement.REPLACE_UPCOMING
                        ) {
                            mutation?.preservedAnchorQueueId
                        } else {
                            null
                        },
                        expectedFileIds = fileIds,
                        expectedQueueEntryIdsByRequestIndex =
                            mutation?.verifiedQueueEntryIdsByRequestIndex.orEmpty(),
                    )

                    var resolvedIndex = 0
                    val trackResults = pendingTrackResults.mapIndexed { index, pending ->
                        val fileId = currentPowerampFileIds[index]
                        if (fileId == null) {
                            pending
                        } else {
                            val requestIndex = resolvedIndex++
                            if (finalVerification.isRequestVerified(requestIndex)) {
                                pending.copy(
                                    status = QueueStatus.QUEUED,
                                    resolvedPowerampQueueId =
                                        finalVerification.verifiedQueueEntryId(requestIndex),
                                )
                            } else {
                                pending.copy(
                                    status = QueueStatus.QUEUE_FAILED,
                                    resolvedPowerampQueueId = null,
                                )
                            }
                        }
                    }

                    val delivery = QueueDeliverySummary.fromTracks(
                        origin = origin,
                        requestedCount = tracks.size,
                        rankedCount = tracks.size,
                        resolvedCount = fileIds.size,
                        tracks = trackResults,
                        verificationComplete = fileIds.isNotEmpty() &&
                            mutation?.fullyVerified == true &&
                            finalVerification.fullyVerified,
                        mutationCount = if (mutation == null) 0 else 1,
                        unexpectedObservedCount = finalVerification.unexpectedObservedCount,
                    )

                    Log.i(TAG, "DIRECT_QUEUE: verified ${delivery.verifiedCount} of ${fileIds.size} resolved tracks " +
                        "(${delivery.notInLibraryCount} not in Poweramp library)")

                    val preservedAnchorId = if (
                        placement == DirectQueuePlacement.REPLACE_UPCOMING
                    ) {
                        finalVerification.preservedAnchorFileId
                    } else {
                        queueAnchorId
                    }
                    val preservedAnchorOccurrenceId = if (
                        placement == DirectQueuePlacement.REPLACE_UPCOMING
                    ) {
                        finalVerification.preservedAnchorQueueId
                    } else {
                        liveQueueOccurrenceId
                    }

                    RadioResult(
                        seedTrack = syntheticSeed,
                        matchType = TrackMatcher.MatchType.NOT_APPLICABLE,
                        tracks = trackResults,
                        queuedFileIds = finalVerification.verifiedFileIds.toSet(),
                        queueAnchorId = preservedAnchorId,
                        queueAnchorOccurrenceId = preservedAnchorOccurrenceId,
                        isDirectQueue = true,
                        delivery = delivery,
                        requestId = requestId,
                        generation = generation,
                        providerGenerationId = providerGenerationId,
                        directQueuePlacement = placement,
                        findMusicSessionEvidence = findMusicSessionEvidence,
                    )
                }

                val message = buildQueueResultMessage(radioResult)
                updateNotification(message)
                toast(message)
                publishSuccess(radioResult)
                terminalFailure = null
                stopSelfDelayed()
            } catch (e: CancellationException) {
                terminalFailure = "Direct queue request was cancelled"
                Log.d(TAG, "Direct queue cancelled")
                stopSelfDelayed()
            } catch (e: Exception) {
                Log.e(TAG, "Error in direct queue", e)
                val listenerMessage = if (e is SecurityException) {
                    "Poweramp queue access is unavailable. Recheck app access and try again."
                } else {
                    "The displayed queue may not have finished. " +
                        "Check session history before retrying."
                }
                terminalFailure = listenerMessage
                _uiState.value = RadioUiState.Error(listenerMessage)
                updateNotification(listenerMessage)
                toast(listenerMessage)
                stopSelfDelayed()
            } finally {
                finishRequest(requestId, terminalFailure)
            }
        }
    }

    private suspend fun finishRequest(requestId: String, operationFailure: String?) {
        withContext(NonCancellable + Dispatchers.IO) {
            try {
                val provisional = latestRequestResult?.takeIf { it.requestId == requestId }
                if (provisional == null) {
                    val detail = operationFailure ?: "Radio request produced no queue result"
                    runCatching { requestStore.markFailed(requestId, detail) }
                        .onFailure { failure ->
                            Log.e(TAG, "Could not fail durable radio request $requestId", failure)
                        }
                    updateWidgetRequestStatus(
                        requestId,
                        WidgetRadioRequestState.FAILED,
                        detail,
                    )
                    return@withContext
                }

                val finalResult = finalizeRequestResult(provisional, operationFailure)
                val terminalDetail = finalResult.failureDetail
                val receiptPersisted = runCatching {
                    requestStore.persistResultReceipt(requestId, finalResult, terminalDetail)
                }.onFailure { failure ->
                    Log.e(TAG, "Could not persist result receipt for $requestId", failure)
                }.isSuccess

                if (!receiptPersisted) {
                    val detail = terminalDetail ?: "Radio result receipt could not be persisted"
                    val failedResult = finalResult.copy(
                        outcome = RadioSessionOutcome.PARTIAL_FAILED,
                        failureDetail = detail,
                    )
                    persistSessionSynchronously(failedResult)
                    runCatching { requestStore.markFailed(requestId, detail) }
                        .onFailure { failure ->
                            Log.e(TAG, "Could not terminalize unreceipted request $requestId", failure)
                        }
                    publishSuccess(failedResult)
                    updateWidgetForResult(failedResult)
                    return@withContext
                }

                if (!persistSessionSynchronously(finalResult)) {
                    // Keep CLAIMED plus the durable receipt. A new owner can upsert this exact
                    // result and terminalize without replaying the Poweramp mutation.
                    _uiState.value = RadioUiState.Error(
                        "Queue result was saved, but session history could not be written",
                    )
                    updateNotification("Queue result saved; history write will be retried")
                    updateWidgetForResult(finalResult)
                    return@withContext
                }

                when (finalResult.outcome) {
                    RadioSessionOutcome.SUCCEEDED -> requestStore.markCompleted(requestId)
                    RadioSessionOutcome.PARTIAL_FAILED,
                    RadioSessionOutcome.CANCELLED -> requestStore.markFailed(
                        requestId,
                        requireNotNull(finalResult.failureDetail),
                    )
                    null -> error("Final radio result has no outcome")
                }
                publishSuccess(finalResult)
                updateWidgetForResult(finalResult)
            } catch (failure: Exception) {
                // A receipt, if present, is the recovery boundary. Do not replay or replace it;
                // a later service owner will reconcile it idempotently.
                Log.e(TAG, "Could not finalize durable radio request $requestId", failure)
                _uiState.value = RadioUiState.Error("Could not finalize saved radio result")
                updateWidgetRequestStatus(
                    requestId,
                    WidgetRadioRequestState.FAILED,
                    "Saved radio result could not be finalized. Open Start Radio to verify the queue.",
                )
            } finally {
                releaseRequestAndContinue(requestId)
            }
        }
    }

    private fun finalizeRequestResult(
        provisional: RadioResult,
        operationFailure: String?,
    ): RadioResult {
        val tracks = provisional.tracks.map { track ->
            if (track.status == QueueStatus.PENDING) {
                track.copy(status = QueueStatus.QUEUE_FAILED)
            } else {
                track
            }
        }
        val priorDelivery = provisional.delivery
        val delivery = QueueDeliverySummary.fromTracks(
            origin = priorDelivery?.origin ?: QueueOrigin.LEGACY_UNKNOWN,
            requestedCount = priorDelivery?.requestedCount
                ?: provisional.totalExpected.takeIf { it > 0 }
                ?: provisional.config.numTracks,
            rankedCount = priorDelivery?.rankedCount ?: tracks.size,
            resolvedCount = priorDelivery?.resolvedCount
                ?: tracks.count { it.resolvedPowerampFileId != null },
            tracks = tracks,
            verificationComplete = operationFailure == null &&
                priorDelivery?.verificationComplete == true &&
                tracks.none { it.status == QueueStatus.QUEUE_FAILED },
            mutationCount = priorDelivery?.mutationCount ?: 0,
            unexpectedObservedCount = priorDelivery?.unexpectedObservedCount ?: 0,
        )
        val outcome = when {
            operationFailure?.contains("cancel", ignoreCase = true) == true -> {
                RadioSessionOutcome.CANCELLED
            }
            operationFailure == null && delivery.verificationComplete &&
                delivery.queueFailedCount == 0 && delivery.notInLibraryCount == 0 &&
                delivery.verifiedCount == delivery.requestedCount -> {
                RadioSessionOutcome.SUCCEEDED
            }
            else -> RadioSessionOutcome.PARTIAL_FAILED
        }
        val failureDetail = when (outcome) {
            RadioSessionOutcome.SUCCEEDED -> null
            RadioSessionOutcome.CANCELLED -> operationFailure ?: "Radio request was cancelled"
            RadioSessionOutcome.PARTIAL_FAILED -> operationFailure
                ?: "Poweramp confirmed only part of this queue"
        }
        return provisional.copy(
            tracks = tracks,
            isComplete = true,
            delivery = delivery,
            outcome = outcome,
            failureDetail = failureDetail,
        )
    }

    private fun publishStreaming(result: RadioResult) {
        latestRequestResult = result
        _uiState.value = RadioUiState.Streaming(result)
    }

    private fun publishSuccess(result: RadioResult) {
        latestRequestResult = result
        _uiState.value = RadioUiState.Success(result)
    }

    private fun updateWidgetForResult(result: RadioResult) {
        val requestId = result.requestId ?: return
        val state = when (result.outcome) {
            RadioSessionOutcome.SUCCEEDED -> WidgetRadioRequestState.SUCCEEDED
            RadioSessionOutcome.PARTIAL_FAILED -> WidgetRadioRequestState.PARTIAL_FAILED
            RadioSessionOutcome.CANCELLED -> WidgetRadioRequestState.CANCELLED
            null -> WidgetRadioRequestState.PARTIAL_FAILED
        }
        val message = when (result.outcome) {
            RadioSessionOutcome.SUCCEEDED ->
                "${result.queuedCount} tracks queued from ${result.seedTrack.title}"
            RadioSessionOutcome.PARTIAL_FAILED -> result.failureDetail
                ?.takeIf(String::isNotBlank)
                ?: "Radio partial: ${result.queuedCount} of ${result.requestedCount} confirmed"
            RadioSessionOutcome.CANCELLED -> result.failureDetail
                ?.takeIf(String::isNotBlank)
                ?: "Radio cancelled"
            null -> "Radio result saved"
        }
        if (result.origin != QueueOrigin.WIDGET_RADIO) {
            updateWidgetRequestStatus(requestId, state, message)
            return
        }
        StartRadioWidgetReceiver.persistRadioStatus(
            this,
            WidgetRadioStatus(
                requestId = requestId,
                seed = WidgetRadioSeedReference.from(result.seedTrack, result.seedIdentity),
                state = state,
                message = message.take(512),
                updatedAtEpochMs = System.currentTimeMillis(),
            ),
        )
    }

    private fun updateWidgetRequestStatus(
        requestId: String,
        state: WidgetRadioRequestState,
        message: String,
        preserveCurrentStates: Set<WidgetRadioRequestState> = emptySet(),
    ) {
        StartRadioWidgetReceiver.updateRadioStatus(
            context = this,
            requestId = requestId,
            state = state,
            message = message.take(512),
            preserveCurrentStates = preserveCurrentStates,
        )
    }

    private fun releaseRequestAndContinue(requestId: String) {
        failRequest(requestId)
        synchronized(requestDispatchLock) {
            if (currentRequestId == requestId) currentRequestId = null
            if (activeDurableRequestId == requestId) activeDurableRequestId = null
            activeJob = null
        }
        if (!shuttingDown) serviceScope.launch(Dispatchers.IO) { dispatchNextRequest() }
    }

    private fun getOrCreateDatabase(
        generation: RadioGenerationToken,
        providerGenerationId: String,
    ): EmbeddingDatabase {
        val library = requireActiveLibrary(this)
        val active = library.activeGeneration
        val activeToken = library.generation
        require(activeToken == generation) {
            "The radio request's embedding generation is no longer active"
        }
        require(library.providerGenerationId == providerGenerationId) {
            "The Poweramp library changed after this radio request was created"
        }
        if (openedGeneration == generation &&
            openedProviderGenerationId == providerGenerationId
        ) {
            return checkNotNull(embeddingDb)
        }
        embeddingDb?.close()
        embeddingDb = EmbeddingDatabase.open(active.databaseFile)
        engine = null
        openedAssets = RecommendationAssetFiles(
            embeddingFile = active.embeddingFile,
            graphFile = active.graphFile,
            activationBindingId = active.manifest.activationBindingId,
        )
        openedGeneration = activeToken
        openedProviderGenerationId = providerGenerationId
        openedActiveCatalog = library.activeCatalog
        return checkNotNull(embeddingDb)
    }

    private fun requireOpenedActiveCatalog(
        generation: RadioGenerationToken,
        providerGenerationId: String,
    ): V2ActiveLibraryCatalog {
        require(openedGeneration == generation &&
            openedProviderGenerationId == providerGenerationId
        ) { "The exact active recommendation catalog is not open" }
        return requireNotNull(openedActiveCatalog)
    }

    private fun requireRequestBindingCurrent(
        generation: RadioGenerationToken,
        providerGenerationId: String,
    ) {
        val (_, activeToken) = requireActiveGeneration(this)
        require(activeToken == generation) {
            "The embedding generation changed before queue delivery"
        }
        require(
            requireNotNull(openedActiveCatalog) {
                "The active Poweramp library is not open for queue delivery"
            }.generationBinding.providerGenerationId == providerGenerationId
        ) { "The Poweramp library binding changed before queue delivery" }
    }

    /** Fail closed if cached folder_files IDs no longer denote their catalog-bound recordings. */
    private fun requireCurrentQueueDeliveryFileIds(
        database: EmbeddingDatabase,
        activeCatalog: V2ActiveLibraryCatalog,
        tracks: List<EmbeddedTrack>,
    ): List<Long> {
        if (tracks.isEmpty()) return emptyList()
        val bindings = tracks.map { track ->
            requireNotNull(activeCatalog.bindingForTrack(track.id)) {
                "Queue track ${track.id} has no active Poweramp binding"
            }
        }
        val startedNs = System.nanoTime()
        val currentById = try {
            PowerampHelper.requireFileEntriesByIds(
                this,
                bindings.map { it.powerampFileId }.distinct(),
            )
        } catch (failure: Exception) {
            if (failure is IllegalArgumentException) {
                V2ActiveLibraryCatalogStore(filesDir).deleteIfMatches(
                    databaseGenerationId = activeCatalog.generationBinding.databaseGenerationId,
                    providerGenerationId = activeCatalog.generationBinding.providerGenerationId,
                )
            }
            publishActiveLibrarySnapshot(null, null)
            throw failure
        }
        val receiptsByTrackId = if (bindings.any {
                it.evidence == V2ActiveLibraryBindingEvidence.EXACT_V2_RECEIPT_SPAN
            }
        ) {
            V2ProviderSpanReceiptReader.read(database.databaseFile).receipts
                .groupBy { it.trackId }
        } else {
            emptyMap()
        }

        tracks.zip(bindings).forEach { (track, binding) ->
            val current = requireNotNull(currentById[binding.powerampFileId]) {
                "Poweramp recording ${binding.powerampFileId} disappeared before queue delivery"
            }
            val bindingStillExact = CurrentPowerampBindingPolicy.matches(
                binding = binding,
                indexed = track,
                current = current,
                receipt = receiptsByTrackId[track.id].orEmpty().singleOrNull(),
            )
            if (!bindingStillExact) {
                V2ActiveLibraryCatalogStore(filesDir).deleteIfMatches(
                    databaseGenerationId = activeCatalog.generationBinding.databaseGenerationId,
                    providerGenerationId = activeCatalog.generationBinding.providerGenerationId,
                )
                publishActiveLibrarySnapshot(null, null)
                error(
                    "Poweramp recording ${binding.powerampFileId} changed since the music " +
                        "library was checked; retry after the library refresh",
                )
            }
        }
        Log.i(
            TAG,
            "Queue delivery bindings: ${tracks.size} tracks validated by one targeted provider " +
                "query in ${(System.nanoTime() - startedNs) / 1_000_000}ms",
        )
        return bindings.map { it.powerampFileId }
    }

    private fun getOrCreateEngine(
        db: EmbeddingDatabase,
        generation: RadioGenerationToken,
    ): RecommendationEngine {
        require(openedGeneration == generation) { "Recommendation generation is not open" }
        engine?.let { return it }
        val assets = requireNotNull(openedAssets) { "Pinned recommendation assets are missing" }
        val activeCatalog = requireNotNull(openedActiveCatalog) {
            "Pinned active recommendation catalog is missing"
        }
        return RecommendationEngine(
            db,
            filesDir,
            assets,
            activeCatalog = activeCatalog,
        ).also { engine = it }
    }

    private fun validatePinnedIdentity(
        db: EmbeddingDatabase,
        generation: RadioGenerationToken,
        identity: RadioSeedIdentity,
    ) {
        requireNotNull(db.getTrackById(identity.embeddedTrackId)) {
            "Pinned track ${identity.embeddedTrackId} is missing from the active generation"
        }
        val currentStableId = readStableTrackSpanId(
            db.databaseFile,
            identity.embeddedTrackId,
            generation.embeddingSpecId,
        )
        require(currentStableId == identity.stableTrackSpanId) {
            "Pinned track identity changed inside the active generation"
        }
    }

    private fun buildQueueResultMessage(result: RadioResult): String {
        val base = if (result.queuedCount == result.requestedCount) {
            "${result.queuedCount} tracks queued in Poweramp"
        } else {
            "${result.queuedCount} of ${result.requestedCount} tracks queued in Poweramp"
        }
        val details = buildList {
            if (!result.isDirectQueue &&
                result.totalExpected > 0 &&
                result.rankedCount < result.totalExpected
            ) {
                add("${result.rankedCount} of ${result.totalExpected} selected")
            } else if (result.rankedCount != result.requestedCount) {
                add("${result.rankedCount} selected")
            }
            if (result.resolvedCount != result.rankedCount) {
                add("${result.resolvedCount} found in Poweramp")
            }
            if (result.notInLibraryCount > 0) add("${result.notInLibraryCount} not in library")
            if (result.queueFailedCount > 0) {
                val label = if (result.delivery?.verificationComplete == false) {
                    "not checked"
                } else {
                    "not confirmed"
                }
                add("${result.queueFailedCount} $label")
            }
            result.delivery?.unexpectedObservedCount?.takeIf { it > 0 }?.let {
                add("$it extra Poweramp entries")
            }
        }
        return if (details.isEmpty()) base else "$base (${details.joinToString(", ")})"
    }

    private fun buildSearchPhaseMessage(config: RadioConfig, seedTitle: String): String {
        val action = when {
            config.driftEnabled && config.selectionMode == SelectionMode.MMR ->
                "Exploring from"
            config.selectionMode == SelectionMode.CLOSEST ->
                "Ranking nearest tracks to"
            config.selectionMode == SelectionMode.RANDOM_WALK ->
                "Exploring graph paths from"
            config.selectionMode == SelectionMode.DPP ->
                "Selecting a diverse set from"
            else ->
                "Finding similar tracks to"
        }
        return "$action: $seedTitle"
    }

    private suspend fun toast(message: String) {
        if (showToasts) {
            withContext(Dispatchers.Main.immediate) {
                Toast.makeText(this@RadioService, message, Toast.LENGTH_SHORT).show()
            }
        }
    }

    /** Deferred work is durable and will be kicked when indexing releases admission. */
    private fun stopDeferredStartImmediately(startId: Int) {
        stopJob?.cancel()
        stopJob = null
        val stopped = if (startId > 0) {
            stopSelfResult(startId)
        } else {
            stopSelf()
            true
        }
        if (stopped) stopForeground(STOP_FOREGROUND_REMOVE)
    }

    private fun stopSelfDelayed() {
        stopJob?.cancel()
        stopJob = serviceScope.launch {
            while (isActive) {
                kotlinx.coroutines.delay(3000)
                val startId = synchronized(requestDispatchLock) {
                    val queuedWorkBlocksStop = if (RecommendationWorkAdmission.isIndexingReserved) {
                        requestQueue.any(::isAdmittedRequest)
                    } else {
                        requestQueue.isNotEmpty()
                    }
                    if (currentRequestId != null || queuedWorkBlocksStop ||
                        activeJob?.isActive == true || currentWidgetIngressId != null ||
                        hasAdmittedRecommendationWork() || synchronized(coldStartWaitLock) {
                            coldStartWaitJobs.values.any(Job::isActive)
                        }
                    ) {
                        null
                    } else {
                        latestStartId
                    }
                }
                if (startId == null) continue
                if (startId > 0) stopSelfResult(startId) else stopSelf()
                return@launch
            }
        }
    }

    private fun hasRecommendationWorkToDrain(): Boolean =
        synchronized(requestDispatchLock) {
            currentRequestId != null || requestQueue.any(::isAdmittedRequest) ||
                activeJob?.isCompleted == false || currentWidgetIngressId != null ||
                hasAdmittedRecommendationWork()
        }

    private fun hasRecommendationTransitionWork(): Boolean =
        synchronized(requestDispatchLock) {
            currentRequestId != null || activeJob?.isCompleted == false ||
                currentWidgetIngressId != null
        }

    private fun createNotificationChannel() {
        val channel = NotificationChannel(
            CHANNEL_ID,
            "Radio Service",
            NotificationManager.IMPORTANCE_LOW
        ).apply {
            description = "Shows when Start Radio is finding similar tracks"
        }
        val notificationManager = getSystemService(NotificationManager::class.java)
        notificationManager.createNotificationChannel(channel)
    }

    private fun createNotification(message: String): Notification {
        val pendingIntent = PendingIntent.getActivity(
            this, 0,
            Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_IMMUTABLE
        )
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("Poweramp Start Radio")
            .setContentText(message)
            .setSmallIcon(R.drawable.ic_radio)
            .setContentIntent(pendingIntent)
            .setOngoing(true)
            .build()
    }

    private fun updateNotification(message: String) {
        val notificationManager = getSystemService(NotificationManager::class.java)
        notificationManager.notify(NOTIFICATION_ID, createNotification(message))
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        shuttingDown = true
        super.onDestroy()
        stopJob?.cancel()
        val running = activeJob
        serviceScope.cancel()
        if (running?.isCompleted != false) {
            closeRecommendationResources()
        } else {
            running.invokeOnCompletion { closeRecommendationResources() }
        }
        if (serviceInstance === this) serviceInstance = null
    }

    @Synchronized
    private fun closeRecommendationResources() {
        embeddingDb?.close()
        embeddingDb = null
        engine = null
        openedAssets = null
        openedGeneration = null
        openedProviderGenerationId = null
        openedActiveCatalog = null
    }
}
