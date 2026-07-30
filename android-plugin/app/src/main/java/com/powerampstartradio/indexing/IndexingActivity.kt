package com.powerampstartradio.indexing

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.selection.toggleable
import androidx.compose.foundation.selection.triStateToggleable
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.ArrowDropDown
import androidx.compose.material.icons.filled.Clear
import androidx.compose.material.icons.filled.MoreVert
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Search
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.Saver
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.state.ToggleableState
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.compose.LifecycleEventEffect
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.viewmodel.compose.viewModel
import com.powerampstartradio.data.EmbeddedTrack
import com.powerampstartradio.indexing.v2.FailureDisposition
import com.powerampstartradio.indexing.v2.IndexingJobState
import com.powerampstartradio.indexing.v2.RetryTrigger
import com.powerampstartradio.indexing.v2.V2IndexingExecutionProfile
import com.powerampstartradio.indexing.v2.V2IndexingPreflightIntentState
import com.powerampstartradio.indexing.v2.V2IndexingPreflightPhase
import com.powerampstartradio.indexing.v2.V2IndexingPreflightProgressUnit
import com.powerampstartradio.ui.theme.PowerampStartRadioTheme
import java.text.NumberFormat
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

private val LongSetSaver = Saver<Set<Long>, LongArray>(
    save = { it.toLongArray() },
    restore = { it.toSet() },
)

internal fun useCompactIndexingProfileControl(maxWidthDp: Float, fontScale: Float): Boolean =
    maxWidthDp < 480f || fontScale > 1.2f

internal val userSelectableIndexingProfiles = listOf(
    V2IndexingExecutionProfile.FULL,
)

internal fun formatIndexingTrackCount(count: Int, locale: Locale = Locale.getDefault()): String {
    require(count >= 0) { "track count cannot be negative" }
    return NumberFormat.getIntegerInstance(locale).format(count)
}

private fun formatIndexingTrackQuantity(count: Int): String =
    "${formatIndexingTrackCount(count)} ${if (count == 1) "track" else "tracks"}"

internal fun indexingJobHeading(
    state: IndexingJobState,
    totalTracks: Int,
): String = when (state) {
    IndexingJobState.PAUSED -> "Indexing paused"
    IndexingJobState.WAITING_FOR_INPUT -> "Indexing needs attention"
    IndexingJobState.INTERRUPTED -> "Indexing interrupted"
    IndexingJobState.READY_TO_RESUME -> "Ready to resume indexing"
    IndexingJobState.CANCELLING -> "Cancelling indexing"
    IndexingJobState.CANCELLED -> "Indexing cancelled"
    IndexingJobState.COMPLETE -> "Indexing complete"
    IndexingJobState.ACTIVATING -> "Updating music index"
    else -> "Indexing ${formatIndexingTrackCount(totalTracks)} tracks"
}

internal fun indexingSelectionSummaryText(
    readyCount: Int,
    attentionCount: Int,
    otherNotReadyCount: Int,
    selectedCount: Int,
): String {
    require(
        listOf(
            readyCount,
            attentionCount,
            otherNotReadyCount,
            selectedCount,
        ).all { it >= 0 },
    ) { "indexing selection counts cannot be negative" }
    return buildList {
        if (readyCount > 0) add("${formatIndexingTrackCount(readyCount)} candidates")
        if (attentionCount > 0) {
            val count = formatIndexingTrackCount(attentionCount)
            add(if (attentionCount == 1) "$count needs attention" else "$count need attention")
        }
        if (otherNotReadyCount > 0) {
            val noun = if (otherNotReadyCount == 1) "track" else "tracks"
            add("${formatIndexingTrackCount(otherNotReadyCount)} other $noun not ready")
        }
        if (selectedCount > 0) add("${formatIndexingTrackCount(selectedCount)} selected")
    }.joinToString(" · ").ifEmpty { "No matching tracks" }
}

internal fun retryAvailableTracksLabel(count: Int): String {
    require(count > 0) { "retry count must be positive" }
    val noun = if (count == 1) "track" else "tracks"
    return "Retry ${formatIndexingTrackCount(count)} available $noun"
}

internal fun indexedTrackSourceLabel(source: String): String? = when (source) {
    "phone", "phone-v2" -> "Indexed on this phone"
    else -> null
}

internal fun indexingExecutionProfileLabel(profile: V2IndexingExecutionProfile): String = when (profile) {
    V2IndexingExecutionProfile.FULL -> "Full speed"
    V2IndexingExecutionProfile.BALANCED -> "Balanced (internal)"
    V2IndexingExecutionProfile.BACKGROUND -> "Keep phone responsive"
}

internal fun indexingExecutionProfileDescription(profile: V2IndexingExecutionProfile): String {
    val behavior = when (profile) {
        V2IndexingExecutionProfile.FULL ->
            "Prioritizes indexing throughput. Best while the phone is idle."
        V2IndexingExecutionProfile.BALANCED ->
            "Internal profile whose device impact has not been measured."
        V2IndexingExecutionProfile.BACKGROUND ->
            "Keeps the phone more responsive; indexing may take longer."
    }
    return "$behavior Embeddings are identical. Changes take effect between processing steps."
}

internal fun indexingExecutionProfileCompactDescription(
    profile: V2IndexingExecutionProfile,
): String = when (profile) {
    V2IndexingExecutionProfile.FULL -> "Fastest; best while the phone is idle."
    V2IndexingExecutionProfile.BALANCED -> "Internal profile."
    V2IndexingExecutionProfile.BACKGROUND ->
        "Keeps the phone more responsive; indexing may take longer. Embeddings are identical."
}

internal fun databaseCleanupBlockedReason(
    durableJobActive: Boolean,
    jobPlanningActive: Boolean,
    exportActive: Boolean,
): String? = when {
    durableJobActive ->
        "Finish or cancel the current indexing job before removing tracks from the music index."
    jobPlanningActive ->
        "Wait for indexing job preparation to finish before removing tracks from the music index."
    exportActive ->
        "Wait for the current export to finish before removing tracks from the music index."
    else -> null
}

internal fun canDiscardIndexingJob(state: IndexingJobState): Boolean = state !in setOf(
    IndexingJobState.ACTIVATING,
    IndexingJobState.CANCELLING,
    IndexingJobState.CANCELLED,
    IndexingJobState.COMPLETE,
)

internal fun preflightTryAgainUnavailableReason(
    attention: V2PreflightAttentionTrack,
): String? = when {
    attention.canTryAgain -> null
    attention.currentTrack.durationMs <= 0 ->
        "Poweramp does not currently report a usable duration."
    attention.currentTrack.detectionKind == V2UnindexedDetectionKind.SOURCE_ATTENTION ->
        "Poweramp does not currently expose a complete playable track entry."
    else -> "This track is not ready to index."
}

internal data class V2DiscardIndexingJobConfirmation private constructor(val jobId: String) {
    fun stillMatches(state: IndexingService.IndexingState): Boolean =
        state is IndexingService.IndexingState.JobSnapshot &&
            state.jobId == jobId &&
            canDiscardIndexingJob(state.jobState)

    companion object {
        fun create(jobId: String): V2DiscardIndexingJobConfirmation {
            require(jobId.matches(Regex("^[A-Za-z0-9._-]{1,128}$")))
            return V2DiscardIndexingJobConfirmation(jobId)
        }
    }
}

internal class V2NeverIndexAttentionConfirmation private constructor(
    val failures: List<IndexingViewModel.FailedTrackUi>,
    val preflightAttention: List<V2PreflightAttentionTrack>,
    val sourceAttention: List<NewTrackDetector.UnindexedTrack>,
) {
    val exactCount: Int get() = failures.size + preflightAttention.size + sourceAttention.size

    companion object {
        fun create(
            failures: List<IndexingViewModel.FailedTrackUi>,
            preflightAttention: List<V2PreflightAttentionTrack>,
            sourceAttention: List<NewTrackDetector.UnindexedTrack>,
        ): V2NeverIndexAttentionConfirmation {
            require(
                failures.isNotEmpty() ||
                    preflightAttention.isNotEmpty() ||
                    sourceAttention.isNotEmpty(),
            ) {
                "never-index attention confirmation requires visible attention tracks"
            }
            return V2NeverIndexAttentionConfirmation(
                failures = failures.toList(),
                preflightAttention = preflightAttention.toList(),
                sourceAttention = sourceAttention.toList(),
            )
        }
    }
}

class IndexingActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        requestIndexingPermissions()
        setContent {
            PowerampStartRadioTheme {
                Surface(
                    modifier = Modifier.fillMaxSize().imePadding(),
                    color = MaterialTheme.colorScheme.background,
                ) {
                    IndexingScreen(
                        initialJobId = intent.getStringExtra(V2IndexingServiceIntents.EXTRA_JOB_ID),
                        onBack = { finish() },
                    )
                }
            }
        }
        lifecycleScope.launch(Dispatchers.IO) {
            runCatching {
                IndexingService.recoverEligible(applicationContext)
            }.onFailure { error ->
                Log.e("IndexingActivity", "Unable to recover durable indexing state", error)
            }
        }
    }

    private fun requestIndexingPermissions() {
        val permissions = buildList {
            val audioPermission = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                Manifest.permission.READ_MEDIA_AUDIO
            } else {
                Manifest.permission.READ_EXTERNAL_STORAGE
            }
            if (ContextCompat.checkSelfPermission(this@IndexingActivity, audioPermission)
                != PackageManager.PERMISSION_GRANTED) {
                add(audioPermission)
            }
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU &&
                ContextCompat.checkSelfPermission(
                    this@IndexingActivity,
                    Manifest.permission.POST_NOTIFICATIONS,
                ) != PackageManager.PERMISSION_GRANTED) {
                add(Manifest.permission.POST_NOTIFICATIONS)
            }
        }
        if (permissions.isNotEmpty()) {
            ActivityCompat.requestPermissions(this, permissions.toTypedArray(), 0)
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun IndexingScreen(
    viewModel: IndexingViewModel = viewModel(),
    initialJobId: String? = null,
    onBack: () -> Unit,
) {
    val context = LocalContext.current
    val unindexedTracks by viewModel.unindexedTracks.collectAsState()
    val selectedIds by viewModel.selectedIds.collectAsState()
    val dismissedIds by viewModel.dismissedIds.collectAsState()
    val isDetecting by viewModel.isDetecting.collectAsState()
    val detectingStatus by viewModel.detectingStatus.collectAsState()
    val detectionError by viewModel.detectionError.collectAsState()
    val hasModels by viewModel.hasModels.collectAsState()
    val isAppFilesChecking by viewModel.isAppFilesChecking.collectAsState()
    val isInitializing by viewModel.isInitializing.collectAsState()
    val hasAudioAccess by viewModel.hasAudioAccess.collectAsState()
    val hasDatabase by viewModel.hasDatabase.collectAsState()
    val databaseCleanupState by viewModel.databaseCleanupState.collectAsState()
    val exportState by viewModel.exportState.collectAsState()
    val indexingState by viewModel.indexingState.collectAsState()
    val planningState by viewModel.planningState.collectAsState()
    val failedTracks by viewModel.failedTracks.collectAsState()
    val preflightAttentionTracks by viewModel.preflightAttentionTracks.collectAsState()
    val preflightHistoryError by viewModel.preflightHistoryError.collectAsState()
    val trackExclusionError by viewModel.trackExclusionError.collectAsState()
    val indexingAdmissionError = trackExclusionError ?: preflightHistoryError
    val exportActive = exportState is IndexingViewModel.ExportState.ChoosingDestination ||
        exportState is IndexingViewModel.ExportState.Exporting
    val databaseCleanupBusy = databaseCleanupState is DatabaseCleanupScanState.Scanning
    val cleanupBlockedReason = databaseCleanupBlockedReason(
        durableJobActive = indexingState !is IndexingService.IndexingState.Idle,
        jobPlanningActive = planningState !is IndexingViewModel.PlanningState.Idle,
        exportActive = exportActive,
    )

    LaunchedEffect(initialJobId) {
        viewModel.attachToJob(initialJobId)
    }

    val audioPermissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission(),
    ) {
        viewModel.refreshAppFiles()
    }

    val exportLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.CreateDocument("application/zip")
    ) { uri ->
        if (uri != null) {
            viewModel.exportInstance(uri)
        } else {
            viewModel.cancelExportSelection()
        }
    }

    LifecycleEventEffect(Lifecycle.Event.ON_RESUME) {
        viewModel.refreshAppFiles()
        if (indexingState is IndexingService.IndexingState.Idle) {
            viewModel.detectUnindexed()
        }
    }

    val visibleTracks = remember(
        unindexedTracks,
        dismissedIds,
        failedTracks,
        preflightAttentionTracks,
    ) {
        val unresolvedFailureIds = unresolvedFailureCurrentTrackIds(
            failures = failedTracks,
            tracks = unindexedTracks,
        )
        val preflightAttentionIds = preflightAttentionTracks.mapTo(hashSetOf()) {
            it.currentTrack.powerampFileId
        }
        unindexedTracks.filter {
            it.powerampFileId !in dismissedIds &&
                it.powerampFileId !in unresolvedFailureIds &&
                it.powerampFileId !in preflightAttentionIds
        }
    }
    val visibleSelectedIds = remember(selectedIds, visibleTracks) {
        visibleTracks.asSequence()
            .map { it.powerampFileId }
            .filterTo(linkedSetOf()) { it in selectedIds }
    }
    val selectedCount = visibleSelectedIds.size
    val selectedCountLabel = remember(selectedCount) {
        formatIndexingTrackCount(selectedCount)
    }
    val hasDismissed = dismissedIds.isNotEmpty()

    var showMenu by remember { mutableStateOf(false) }
    var pendingNeverIndexConfirmation by remember {
        mutableStateOf<V2NeverIndexConfirmation?>(null)
    }
    var pendingNeverIndexAttentionConfirmation by remember {
        mutableStateOf<V2NeverIndexAttentionConfirmation?>(null)
    }
    var pendingDiscardJobConfirmation by remember {
        mutableStateOf<V2DiscardIndexingJobConfirmation?>(null)
    }
    var pendingNeverIndexFailure by remember {
        mutableStateOf<IndexingViewModel.FailedTrackUi?>(null)
    }
    var pendingNeverIndexPreflight by remember {
        mutableStateOf<V2PreflightAttentionTrack?>(null)
    }
    var showNeverIndex by rememberSaveable { mutableStateOf(false) }
    var showCleanDatabase by rememberSaveable { mutableStateOf(false) }
    var isSearchActive by rememberSaveable { mutableStateOf(false) }
    var filteredTrackIds by remember { mutableStateOf<Set<Long>?>(null) }

    // Auto-close screens when their list becomes empty
    LaunchedEffect(showNeverIndex, hasDismissed) {
        if (showNeverIndex && !hasDismissed) showNeverIndex = false
    }

    LaunchedEffect(exportState) {
        when (val state = exportState) {
            is IndexingViewModel.ExportState.Complete -> {
                Toast.makeText(
                    context,
                    "Exported ${state.filename}",
                    Toast.LENGTH_SHORT
                ).show()
                viewModel.clearExportState()
            }
            is IndexingViewModel.ExportState.Error -> {
                Toast.makeText(
                    context,
                    state.message,
                    Toast.LENGTH_LONG
                ).show()
                viewModel.clearExportState()
            }
            else -> Unit
        }
    }

    LaunchedEffect(indexingState, pendingDiscardJobConfirmation) {
        if (pendingDiscardJobConfirmation?.stillMatches(indexingState) == false) {
            pendingDiscardJobConfirmation = null
        }
    }

    if (showNeverIndex) {
        BackHandler { showNeverIndex = false }
        NeverIndexScreen(
            viewModel = viewModel,
            onBack = { showNeverIndex = false },
        )
        return
    }
    if (showCleanDatabase) {
        BackHandler { showCleanDatabase = false }
        CleanDatabaseScreen(
            state = databaseCleanupState,
            blockedReason = cleanupBlockedReason,
            onRefresh = { viewModel.detectDatabaseOnlyTracks(forceRefresh = true) },
            onDelete = { ids -> viewModel.deleteDatabaseOnlyTracks(ids) },
            onBack = { showCleanDatabase = false },
        )
        return
    }

    pendingNeverIndexConfirmation?.let { confirmation ->
        val trackNoun = if (confirmation.exactCount == 1) "track" else "tracks"
        val countLabel = formatIndexingTrackCount(confirmation.exactCount)
        AlertDialog(
            onDismissRequest = { pendingNeverIndexConfirmation = null },
            title = {
                Text("Never index $countLabel selected $trackNoun?")
            },
            text = {
                Text(
                    "These $countLabel $trackNoun will be skipped in future " +
                        "indexing runs. Restore them from the Never-index list to include " +
                        "them again.",
                )
            },
            dismissButton = {
                TextButton(onClick = { pendingNeverIndexConfirmation = null }) {
                    Text("Cancel")
                }
            },
            confirmButton = {
                Button(
                    onClick = {
                        pendingNeverIndexConfirmation = null
                        viewModel.neverIndexTracks(confirmation.trackIds)
                    },
                ) {
                    Text("Never index")
                }
            },
        )
    }

    pendingNeverIndexAttentionConfirmation?.let { confirmation ->
        val countLabel = formatIndexingTrackCount(confirmation.exactCount)
        val trackNoun = if (confirmation.exactCount == 1) "track" else "tracks"
        val subject = if (confirmation.exactCount == 1) {
            "This currently shown attention track"
        } else {
            "All $countLabel currently shown attention tracks"
        }
        val objectPronoun = if (confirmation.exactCount == 1) "it" else "them"
        AlertDialog(
            onDismissRequest = { pendingNeverIndexAttentionConfirmation = null },
            title = { Text("Never index $countLabel attention $trackNoun?") },
            text = {
                Text(
                    "$subject will be skipped in future indexing runs. " +
                        "Ready selected tracks will stay selected. Restore $objectPronoun from " +
                        "the Never-index list to include $objectPronoun again.",
                )
            },
            dismissButton = {
                TextButton(onClick = { pendingNeverIndexAttentionConfirmation = null }) {
                    Text("Cancel")
                }
            },
            confirmButton = {
                Button(
                    onClick = {
                        pendingNeverIndexAttentionConfirmation = null
                        viewModel.neverIndexAttentionTracks(
                            failures = confirmation.failures,
                            preflightAttention = confirmation.preflightAttention,
                            sourceAttention = confirmation.sourceAttention,
                        )
                    },
                ) {
                    Text(if (confirmation.exactCount == 1) "Never index" else "Never index shown")
                }
            },
        )
    }


    pendingDiscardJobConfirmation?.let { confirmation ->
        AlertDialog(
            onDismissRequest = { pendingDiscardJobConfirmation = null },
            title = { Text("Discard this indexing run?") },
            text = {
                Text(
                    "Pause keeps completed work. Discard deletes this run's saved progress; " +
                        "the current music index is unchanged.",
                )
            },
            dismissButton = {
                TextButton(onClick = { pendingDiscardJobConfirmation = null }) {
                    Text("Keep run")
                }
            },
            confirmButton = {
                Button(
                    onClick = {
                        if (confirmation.stillMatches(indexingState)) {
                            viewModel.cancelIndexing(confirmation.jobId)
                        }
                        pendingDiscardJobConfirmation = null
                    },
                    colors = ButtonDefaults.buttonColors(
                        containerColor = MaterialTheme.colorScheme.error,
                        contentColor = MaterialTheme.colorScheme.onError,
                    ),
                ) {
                    Text("Discard run")
                }
            },
        )
    }

    pendingNeverIndexFailure?.let { failure ->
        val displayTitle = failure.title.ifBlank { "Unknown track" }
        AlertDialog(
            onDismissRequest = { pendingNeverIndexFailure = null },
            title = { Text("Never index $displayTitle?") },
            text = {
                Text(
                    buildString {
                        if (failure.artist.isNotBlank()) append(failure.artist).append(". ")
                        append("This exact Poweramp track entry will be skipped in future runs. ")
                        if (canOfferIndexingFailureActions(failure.jobState)) {
                            append("The current failure will be marked skipped. ")
                        } else {
                            append("The completed run remains unchanged. ")
                        }
                        append("Restore it from the Never-index list to include it again.")
                    },
                )
            },
            dismissButton = {
                TextButton(onClick = { pendingNeverIndexFailure = null }) {
                    Text("Leave ready to retry")
                }
            },
            confirmButton = {
                Button(
                    onClick = {
                        viewModel.neverIndexFailure(failure)
                        pendingNeverIndexFailure = null
                    },
                ) {
                    Text("Never index")
                }
            },
        )
    }

    pendingNeverIndexPreflight?.let { attention ->
        val displayTitle = attention.currentTrack.title.ifBlank { "Unknown track" }
        AlertDialog(
            onDismissRequest = { pendingNeverIndexPreflight = null },
            title = { Text("Never index $displayTitle?") },
            text = {
                Text(
                    "This exact Poweramp track entry will stay out of future indexing runs. " +
                        "Restore it from the Never-index list to include it again.",
                )
            },
            dismissButton = {
                TextButton(onClick = { pendingNeverIndexPreflight = null }) {
                    Text("Leave ready to retry")
                }
            },
            confirmButton = {
                Button(
                    onClick = {
                        viewModel.neverIndexPreflightRejection(attention)
                        pendingNeverIndexPreflight = null
                    },
                ) {
                    Text("Never index")
                }
            },
        )
    }

    Scaffold(
        contentWindowInsets = WindowInsets.safeDrawing,
        topBar = {
            TopAppBar(
                title = { Text("On-device indexing") },
                windowInsets = WindowInsets.safeDrawing.only(
                    WindowInsetsSides.Horizontal + WindowInsetsSides.Top,
                ),
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back")
                    }
                },
                actions = {
                    if (indexingState is IndexingService.IndexingState.Idle
                        && planningState is IndexingViewModel.PlanningState.Idle && !exportActive &&
                        (hasDatabase || visibleTracks.isNotEmpty() || hasDismissed ||
                            preflightAttentionTracks.isNotEmpty()) && !isDetecting) {
                        Box {
                            IconButton(onClick = { showMenu = true }) {
                                Icon(Icons.Default.MoreVert, contentDescription = "More")
                            }
                            DropdownMenu(
                                expanded = showMenu,
                                onDismissRequest = { showMenu = false },
                            ) {
                                DropdownMenuItem(
                                    text = {
                                        Column {
                                            Text(
                                                "Remove missing tracks",
                                            )
                                            cleanupBlockedReason?.let { reason ->
                                                Text(
                                                    reason,
                                                    style = MaterialTheme.typography.labelSmall,
                                                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                                                )
                                            }
                                        }
                                    },
                                    enabled = hasDatabase && cleanupBlockedReason == null &&
                                        !databaseCleanupBusy,
                                    onClick = {
                                        viewModel.detectDatabaseOnlyTracks(forceRefresh = true)
                                        showCleanDatabase = true
                                        showMenu = false
                                    }
                                )
                                DropdownMenuItem(
                                    text = { Text("Export library bundle") },
                                    enabled = hasDatabase && !exportActive &&
                                        planningState is IndexingViewModel.PlanningState.Idle,
                                    onClick = {
                                        if (viewModel.beginExportSelection()) {
                                            val timestamp = SimpleDateFormat(
                                                "yyyyMMdd-HHmmss",
                                                Locale.US
                                            ).format(Date())
                                            exportLauncher.launch(
                                                "poweramp-start-radio-$timestamp.zip",
                                            )
                                        }
                                        showMenu = false
                                    }
                                )
                                if (visibleTracks.isNotEmpty() || hasDismissed) {
                                    HorizontalDivider()
                                }
                                if (selectedCount > 0) {
                                    DropdownMenuItem(
                                        text = { Text("Never index selected ($selectedCountLabel)") },
                                        onClick = {
                                            pendingNeverIndexConfirmation =
                                                V2NeverIndexConfirmation.create(
                                                    visibleSelectedIds,
                                                )
                                            showMenu = false
                                        }
                                    )
                                }
                                if (hasDismissed) {
                                    DropdownMenuItem(
                                        text = {
                                            Text(
                                                "View never-index list " +
                                                    "(${formatIndexingTrackCount(dismissedIds.size)})",
                                            )
                                        },
                                        onClick = {
                                            showNeverIndex = true
                                            showMenu = false
                                        }
                                    )
                                }
                            }
                        }
                    }
                }
            )
        },
        bottomBar = {
            if (indexingState is IndexingService.IndexingState.Idle
                && planningState is IndexingViewModel.PlanningState.Idle
                && !exportActive && visibleTracks.isNotEmpty() && !isDetecting &&
                indexingAdmissionError == null
            ) {
                val visibleSelectedCount = filteredTrackIds?.let { ids ->
                    selectedIds.count { it in ids }
                } ?: selectedCount
                BottomBar(
                    selectedCount = selectedCount,
                    visibleSelectedCount = visibleSelectedCount,
                    isFiltered = isSearchActive,
                    hasModels = hasModels,
                    isAppFilesChecking = isAppFilesChecking,
                    hasAudioAccess = hasAudioAccess,
                    onRequestAudioAccess = {
                        val permission = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                            Manifest.permission.READ_MEDIA_AUDIO
                        } else {
                            Manifest.permission.READ_EXTERNAL_STORAGE
                        }
                        audioPermissionLauncher.launch(permission)
                    },
                    onStartIndexing = { viewModel.startIndexing(buildGraph = true) },
                )
            }
        }
    ) { padding ->
        Box(modifier = Modifier.fillMaxSize().padding(padding)) {
            when (val planning = planningState) {
                is IndexingViewModel.PlanningState.Planning -> {
                    DetectingContent(status = planning.message)
                }
                is IndexingViewModel.PlanningState.Failed -> {
                    ErrorContent(
                        message = planning.message,
                        actionLabel = "Back to tracks",
                        onBack = viewModel::clearPlanningError,
                    )
                }
                IndexingViewModel.PlanningState.Idle -> when {
                    exportActive -> {
                        DetectingContent(
                            status = when (val current = exportState) {
                                IndexingViewModel.ExportState.ChoosingDestination ->
                                    "Choosing export destination..."
                                is IndexingViewModel.ExportState.Exporting -> current.message
                                else -> "Preparing export..."
                            },
                        )
                    }
                    else -> when (val state = indexingState) {
                        is IndexingService.IndexingState.Idle -> {
                            if (isDetecting) {
                                DetectingContent(status = detectingStatus)
                            } else if (detectionError != null) {
                                DetectionFailureContent(
                                    message = detectionError!!,
                                    onGrantPowerampAccess = { viewModel.requestPowerampAccess() },
                                    onRetry = { viewModel.detectUnindexed(forceRefresh = true) },
                                )
                            } else if (isInitializing) {
                                DetectingContent(
                                    status = "Reading saved indexing jobs and Never index choices",
                                )
                            } else if (indexingAdmissionError != null) {
                                PreflightHistoryFailureContent(
                                    message = indexingAdmissionError,
                                    onRetry = viewModel::refreshDurablePreflightHistory,
                                )
                            } else if (visibleTracks.isEmpty() && failedTracks.isEmpty() &&
                                preflightAttentionTracks.isEmpty()
                            ) {
                                val missingIds = unindexedTracks.map { it.powerampFileId }.toSet()
                                AllIndexedContent(
                                    totalMissing = unindexedTracks.size,
                                    sourceAttentionCount = unindexedTracks.count {
                                        it.detectionKind ==
                                            V2UnindexedDetectionKind.SOURCE_ATTENTION &&
                                            !V2IndexingSelectionPolicy.isReadyTrack(it)
                                    },
                                    neverIndexCount = dismissedIds.count { it in missingIds },
                                    ignoredCount = 0,
                                )
                            } else {
                                TrackSelectionContent(
                                    tracks = visibleTracks,
                                    selectedIds = selectedIds,
                                    selectedCount = selectedCount,
                                    failures = failedTracks,
                                    preflightAttention = preflightAttentionTracks,
                                    onToggle = { viewModel.toggleSelection(it) },
                                    onSelectVisible = viewModel::selectIds,
                                    onDeselectVisible = viewModel::deselectIds,
                                    onSearchActiveChanged = { isSearchActive = it },
                                    onFilteredIdsChanged = { filteredTrackIds = it },
                                    onNeverAllAttention = {
                                            failures, preflightAttention, sourceAttention ->
                                        pendingNeverIndexAttentionConfirmation =
                                            V2NeverIndexAttentionConfirmation.create(
                                                failures = failures,
                                                preflightAttention = preflightAttention,
                                                sourceAttention = sourceAttention,
                                            )
                                    },
                                    onRetryFailure = viewModel::retryFailure,
                                    onSelectFailureForNewRun = viewModel::selectFailureForNewRun,
                                    onSkipFailure = viewModel::skipFailure,
                                    onNeverFailure = { pendingNeverIndexFailure = it },
                                    onTryAgainPreflight =
                                        viewModel::tryAgainPreflightRejection,
                                    onNeverPreflight = { pendingNeverIndexPreflight = it },
                                )
                            }
                        }
                        is IndexingService.IndexingState.JobSnapshot -> {
                            val jobFailures = failedTracks.filter {
                                it.jobId == state.jobId && it.jobState == state.jobState
                            }
                            val jobPreflightAttention = preflightAttentionTracks.filter {
                                it.rejection.jobId == state.jobId
                            }
                            DurableJobContent(
                                state = state,
                                failures = jobFailures,
                                preflightAttention = jobPreflightAttention,
                                onPause = { viewModel.pauseIndexing(state.jobId) },
                                onResume = { viewModel.resumeIndexing(state.jobId) },
                                onCancel = {
                                    pendingDiscardJobConfirmation =
                                        V2DiscardIndexingJobConfirmation.create(state.jobId)
                                },
                                onRetry = { viewModel.retryFailures(state.jobId) },
                                onRetryFailure = viewModel::retryFailure,
                                onSelectFailureForNewRun = viewModel::selectFailureForNewRun,
                                onSkip = viewModel::skipFailure,
                                onNever = { pendingNeverIndexFailure = it },
                                onTryAgainPreflight =
                                    viewModel::tryAgainPreflightRejection,
                                onNeverPreflight = { pendingNeverIndexPreflight = it },
                                onDone = {
                                    IndexingService.resetState()
                                    viewModel.detectUnindexed()
                                },
                            )
                        }
                        is IndexingService.IndexingState.PreflightSnapshot -> {
                            DurablePreflightContent(
                                state = state,
                                preflightAttention = preflightAttentionTracks.filter {
                                    it.rejection.jobId == state.jobId
                                },
                                onTryAgainPreflight =
                                    viewModel::tryAgainPreflightRejection,
                                onNeverPreflight = { pendingNeverIndexPreflight = it },
                                onRetry = { viewModel.resumeIndexing(state.jobId) },
                                onCancel = { viewModel.cancelIndexing(state.jobId) },
                                onDone = {
                                    IndexingService.resetState()
                                    viewModel.detectUnindexed()
                                },
                            )
                        }
                        is IndexingService.IndexingState.Error -> {
                            ErrorContent(message = state.message, onBack = onBack)
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun DurablePreflightContent(
    state: IndexingService.IndexingState.PreflightSnapshot,
    preflightAttention: List<V2PreflightAttentionTrack>,
    onTryAgainPreflight: (V2PreflightAttentionTrack) -> Unit,
    onNeverPreflight: (V2PreflightAttentionTrack) -> Unit,
    onRetry: () -> Unit,
    onCancel: () -> Unit,
    onDone: () -> Unit,
) {
    val terminal = state.state in setOf(
        V2IndexingPreflightIntentState.CANCELLED,
        V2IndexingPreflightIntentState.RESOLVED_WITHOUT_EXECUTABLE_ROWS,
    )
    Scaffold(
        contentWindowInsets = WindowInsets(0, 0, 0, 0),
        bottomBar = {
            Surface(tonalElevation = 3.dp) {
                Row(
                    modifier = Modifier.fillMaxWidth().padding(16.dp),
                    horizontalArrangement = Arrangement.spacedBy(12.dp, Alignment.End),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    when {
                        terminal -> Button(onClick = onDone) { Text("Done") }
                        V2IndexingPreflightControlPolicy.isExplicitlyResumable(state.state) -> {
                            OutlinedButton(onClick = onCancel) { Text("Cancel request") }
                            Button(onClick = onRetry) { Text("Try again") }
                        }
                        else -> OutlinedButton(onClick = onCancel) {
                            Text("Cancel request")
                        }
                    }
                }
            }
        },
    ) { padding ->
        LazyColumn(
            modifier = Modifier.fillMaxSize().padding(padding),
            contentPadding = PaddingValues(24.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            item {
                Column(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalAlignment = Alignment.CenterHorizontally,
                ) {
                    Text(
                        when (state.state) {
                            V2IndexingPreflightIntentState.REQUESTED -> "Indexing request saved"
                            V2IndexingPreflightIntentState.PLANNING -> {
                                val count = formatIndexingTrackCount(state.selectedTrackCount)
                                "Building an exact $count-track indexing plan"
                            }
                            V2IndexingPreflightIntentState.INTERRUPTED ->
                                "Preparation interrupted"
                            V2IndexingPreflightIntentState.FAILED -> "Preparation needs attention"
                            V2IndexingPreflightIntentState.CANCEL_REQUESTED ->
                                "Cancelling preparation"
                            V2IndexingPreflightIntentState.CANCELLED ->
                                "Indexing request cancelled"
                            V2IndexingPreflightIntentState.RESOLVED_WITH_EXECUTABLE_ROWS ->
                                "Saving indexing job"
                            V2IndexingPreflightIntentState.RESOLVED_WITHOUT_EXECUTABLE_ROWS ->
                                "No selected tracks can be indexed"
                            V2IndexingPreflightIntentState.MATERIALIZED -> "Indexing job ready"
                        },
                        style = MaterialTheme.typography.headlineMedium,
                        color = if (state.state == V2IndexingPreflightIntentState.FAILED) {
                            MaterialTheme.colorScheme.error
                        } else {
                            MaterialTheme.colorScheme.primary
                        },
                        textAlign = TextAlign.Center,
                    )
                    Spacer(modifier = Modifier.height(16.dp))
                    val completed = state.progress.completedUnits
                    val total = state.progress.totalUnits
                    if (state.progress.unit != V2IndexingPreflightProgressUnit.BYTES &&
                        completed != null &&
                        total != null &&
                        total > 0L &&
                        completed > 0L
                    ) {
                        LinearProgressIndicator(
                            progress = {
                                (completed.toDouble() / total).toFloat().coerceIn(0f, 1f)
                            },
                            modifier = Modifier.fillMaxWidth(),
                        )
                        Spacer(modifier = Modifier.height(8.dp))
                        Text(
                            preflightProgressText(
                                completed = completed,
                                total = total,
                                phase = state.progress.phase,
                            ),
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    } else if (!terminal &&
                        state.state != V2IndexingPreflightIntentState.FAILED
                    ) {
                        LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                    }
                    Spacer(modifier = Modifier.height(12.dp))
                    Text(
                        preflightStatusEvidenceText(
                            state = state.state,
                            failureCode = state.failureCode,
                            progressMessage = state.progress.message,
                        ),
                        style = MaterialTheme.typography.bodyMedium,
                        textAlign = TextAlign.Center,
                        color = if (state.state == V2IndexingPreflightIntentState.FAILED) {
                            MaterialTheme.colorScheme.error
                        } else {
                            MaterialTheme.colorScheme.onSurfaceVariant
                        },
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        buildString {
                            append(
                                "${formatIndexingTrackCount(state.selectedTrackCount)} selected tracks",
                            )
                            if (userSelectableIndexingProfiles.size > 1) {
                                append(" · ")
                                append(indexingExecutionProfileLabel(state.profile))
                            }
                        },
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        textAlign = TextAlign.Center,
                    )
                }
            }
            if (preflightAttention.isNotEmpty()) {
                item {
                    Text(
                        "Needs attention (${formatIndexingTrackCount(preflightAttention.size)})",
                        style = MaterialTheme.typography.titleSmall,
                        modifier = Modifier.fillMaxWidth(),
                    )
                }
                items(
                    preflightAttention,
                    key = {
                        "preflight:${it.rejection.providerSpan.normalizedPhysicalPath}:" +
                            "${it.rejection.providerSpan.offsetMs}:" +
                            it.rejection.providerSpan.durationMs
                    },
                ) { attention ->
                    PreflightAttentionItem(
                        attention = attention,
                        onTryAgain = onTryAgainPreflight,
                        onNever = onNeverPreflight,
                    )
                }
            }
        }
    }
}

internal fun preflightProgressText(
    completed: Long,
    total: Long,
    phase: V2IndexingPreflightPhase,
    locale: Locale = Locale.getDefault(),
): String {
    require(completed >= 0L && total >= 0L) { "preflight counts cannot be negative" }
    val done = NumberFormat.getIntegerInstance(locale).format(completed)
    val planned = NumberFormat.getIntegerInstance(locale).format(total)
    if (completed == 0L) {
        return when (phase) {
            V2IndexingPreflightPhase.POWERAMP_SNAPSHOT ->
                "Poweramp reports $planned library rows; beginning the complete read"
            V2IndexingPreflightPhase.SOURCE_FINGERPRINTS ->
                "Beginning source-identity review for $planned selected tracks"
            V2IndexingPreflightPhase.SOURCE_REVALIDATION ->
                "Opening the first selected audio file for exact revalidation"
            V2IndexingPreflightPhase.MODEL_FINGERPRINTS ->
                "Opening the first model file for exact hashing"
            else -> "Beginning $planned preparation steps"
        }
    }
    return when (phase) {
        V2IndexingPreflightPhase.POWERAMP_SNAPSHOT ->
            "Read $done of $planned Poweramp library rows"
        V2IndexingPreflightPhase.SOURCE_FINGERPRINTS ->
            "Reviewed $done of $planned selected tracks for source identity"
        V2IndexingPreflightPhase.SOURCE_REVALIDATION ->
            "Confirmed $done of $planned selected audio files"
        V2IndexingPreflightPhase.MODEL_FINGERPRINTS ->
            "Hashed $done of $planned model and tokenizer files"
        else -> "Completed $done of $planned preparation steps"
    }
}

@Composable
private fun DetectingContent(status: String = "") {
    Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        Column(
            modifier = Modifier.fillMaxWidth().verticalScroll(rememberScrollState()),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            CircularProgressIndicator()
            Spacer(modifier = Modifier.height(16.dp))
            Text(
                status.ifEmpty {
                    "Reading indexed source receipts and Poweramp library rows"
                },
                style = MaterialTheme.typography.bodyMedium,
            )
        }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun DetectionFailureContent(
    message: String,
    onGrantPowerampAccess: () -> Unit,
    onRetry: () -> Unit,
) {
    Box(
        modifier = Modifier.fillMaxSize().padding(32.dp),
        contentAlignment = Alignment.Center,
    ) {
        Column(
            modifier = Modifier.fillMaxWidth().verticalScroll(rememberScrollState()),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Text(
                "Couldn't refresh tracks",
                style = MaterialTheme.typography.headlineSmall,
                color = MaterialTheme.colorScheme.error,
                textAlign = TextAlign.Center,
            )
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                message,
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                textAlign = TextAlign.Center,
            )
            Spacer(modifier = Modifier.height(20.dp))
            FlowRow(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp, Alignment.CenterHorizontally),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                OutlinedButton(onClick = onGrantPowerampAccess) {
                    Text("Grant Poweramp access")
                }
                FilledTonalButton(onClick = onRetry) {
                    Text("Retry")
                }
            }
        }
    }
}

@Composable
private fun PreflightHistoryFailureContent(
    message: String,
    onRetry: () -> Unit,
) {
    Box(
        modifier = Modifier.fillMaxSize().padding(32.dp),
        contentAlignment = Alignment.Center,
    ) {
        Column(
            modifier = Modifier.fillMaxWidth().verticalScroll(rememberScrollState()),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Text(
                "Saved indexing state needs attention",
                style = MaterialTheme.typography.headlineSmall,
                color = MaterialTheme.colorScheme.error,
                textAlign = TextAlign.Center,
            )
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                message,
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                textAlign = TextAlign.Center,
            )
            Spacer(modifier = Modifier.height(20.dp))
            FilledTonalButton(onClick = onRetry) { Text("Retry") }
        }
    }
}

@Composable
private fun AllIndexedContent(
    totalMissing: Int,
    sourceAttentionCount: Int,
    neverIndexCount: Int,
    ignoredCount: Int,
) {
    Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            modifier = Modifier.padding(32.dp),
        ) {
            Text(
                "Nothing to index",
                style = MaterialTheme.typography.headlineSmall,
                color = MaterialTheme.colorScheme.primary,
            )
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                when {
                    totalMissing == 0 ->
                        "Everything in Poweramp is already indexed."
                    neverIndexCount + ignoredCount == totalMissing ->
                        buildString {
                            append(
                                "${formatIndexingTrackCount(totalMissing)} tracks " +
                                    "are hidden by your saved choices.",
                            )
                            if (sourceAttentionCount > 0) {
                                append(
                                    " ${formatIndexingTrackCount(sourceAttentionCount)} sources " +
                                        "need attention.",
                                )
                            }
                        }
                    else -> buildString {
                        if (sourceAttentionCount > 0) {
                            append(
                                "${formatIndexingTrackCount(sourceAttentionCount)} sources " +
                                    "need attention.",
                            )
                        }
                        if (isEmpty()) {
                            append(
                                "${formatIndexingTrackCount(totalMissing)} tracks need review " +
                                    "before indexing.",
                            )
                        }
                    }
                },
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                textAlign = TextAlign.Center,
            )
        }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun TrackSelectionContent(
    tracks: List<NewTrackDetector.UnindexedTrack>,
    selectedIds: Set<Long>,
    selectedCount: Int,
    failures: List<IndexingViewModel.FailedTrackUi>,
    preflightAttention: List<V2PreflightAttentionTrack>,
    onToggle: (Long) -> Unit,
    onSelectVisible: (Set<Long>) -> Unit,
    onDeselectVisible: (Set<Long>) -> Unit,
    onSearchActiveChanged: (Boolean) -> Unit,
    onFilteredIdsChanged: (Set<Long>?) -> Unit,
    onNeverAllAttention: (
        List<IndexingViewModel.FailedTrackUi>,
        List<V2PreflightAttentionTrack>,
        List<NewTrackDetector.UnindexedTrack>,
    ) -> Unit,
    onRetryFailure: (IndexingViewModel.FailedTrackUi) -> Unit,
    onSelectFailureForNewRun: (IndexingViewModel.FailedTrackUi) -> Unit,
    onSkipFailure: (IndexingViewModel.FailedTrackUi) -> Unit,
    onNeverFailure: (IndexingViewModel.FailedTrackUi) -> Unit,
    onTryAgainPreflight: (V2PreflightAttentionTrack) -> Unit,
    onNeverPreflight: (V2PreflightAttentionTrack) -> Unit,
) {
    var searchQuery by rememberSaveable { mutableStateOf("") }

    // Report search state changes up
    LaunchedEffect(searchQuery) {
        onSearchActiveChanged(searchQuery.isNotBlank())
    }

    val filteredTracks = remember(tracks, searchQuery) {
        tracks.filter { track ->
            matchesIndexingTrackSearch(
                searchQuery,
                track.title,
                track.artist,
                track.album,
                track.path,
            )
        }
    }
    val filteredFailures = remember(failures, searchQuery) {
        failures.filter { failure ->
            matchesIndexingTrackSearch(
                searchQuery,
                failure.title,
                failure.artist,
                indexingFailureSummary(failure.code),
                failure.diagnostic,
                failure.providerPhysicalPath,
            )
        }
    }
    val filteredPreflightAttention = remember(preflightAttention, searchQuery) {
        preflightAttention.filter { attention ->
            matchesIndexingTrackSearch(
                searchQuery,
                attention.currentTrack.title,
                attention.currentTrack.artist,
                attention.currentTrack.album,
                attention.rejection.diagnostic,
                preflightRejectionSummary(attention.rejection.code),
                attention.rejection.providerSpan.normalizedPhysicalPath,
            )
        }
    }

    // Report filtered IDs so startIndexing can scope to visible tracks
    LaunchedEffect(filteredTracks) {
        onFilteredIdsChanged(
            if (searchQuery.isNotBlank()) filteredTracks.map { it.powerampFileId }.toSet()
            else null
        )
    }

    val isFiltered = searchQuery.isNotBlank()
    val filteredSelectedCount = remember(filteredTracks, selectedIds) {
        filteredTracks.count { it.powerampFileId in selectedIds }
    }
    val readyTracks = remember(filteredTracks) {
        filteredTracks.filter(V2IndexingSelectionPolicy::isReadyTrack)
    }
    val sourceAttentionTracks = remember(filteredTracks) {
        filteredTracks.filter {
            it.detectionKind == V2UnindexedDetectionKind.SOURCE_ATTENTION &&
                !V2IndexingSelectionPolicy.isReadyTrack(it)
        }
    }
    val filteredAttentionCount =
        filteredFailures.size + filteredPreflightAttention.size + sourceAttentionTracks.size
    val selectableVisibleIds = remember(readyTracks) {
        readyTracks.asSequence()
            .map { it.powerampFileId }
            .toSet()
    }
    val allFilteredIds = remember(filteredTracks) {
        filteredTracks.mapTo(linkedSetOf()) { it.powerampFileId }
    }

    Column(modifier = Modifier.fillMaxSize()) {
        OutlinedTextField(
            value = searchQuery,
            onValueChange = { searchQuery = it },
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = 8.dp),
            placeholder = { Text("Search tracks...") },
            leadingIcon = { Icon(Icons.Default.Search, contentDescription = null) },
            trailingIcon = {
                if (searchQuery.isNotEmpty()) {
                    IconButton(onClick = { searchQuery = "" }) {
                        Icon(Icons.Default.Clear, contentDescription = "Clear")
                    }
                }
            },
            singleLine = true,
        )

        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = 8.dp),
        ) {
            val selectedForSummary = if (isFiltered) filteredSelectedCount else selectedCount
            Text(
                buildString {
                    append(
                        indexingSelectionSummaryText(
                            readyCount = readyTracks.size,
                            attentionCount = filteredAttentionCount,
                            otherNotReadyCount = 0,
                            selectedCount = selectedForSummary,
                        ),
                    )
                    if (isFiltered) {
                        append(
                            " · ${formatIndexingTrackCount(filteredTracks.size)} of " +
                                "${formatIndexingTrackCount(tracks.size)} tracks shown",
                        )
                    }
                },
                style = MaterialTheme.typography.titleSmall,
            )
            FlowRow(
                horizontalArrangement = Arrangement.spacedBy(4.dp),
                verticalArrangement = Arrangement.spacedBy(4.dp),
            ) {
                TextButton(
                    onClick = { onSelectVisible(selectableVisibleIds) },
                    enabled = canSelectReadyTracks(selectableVisibleIds, selectedIds),
                ) { Text("Select candidates") }
                TextButton(
                    onClick = { onDeselectVisible(allFilteredIds) },
                    enabled = filteredSelectedCount > 0,
                ) { Text("Deselect visible") }
            }
        }

        HorizontalDivider()

        LazyColumn(
            modifier = Modifier.weight(1f).fillMaxWidth(),
            contentPadding = PaddingValues(vertical = 4.dp),
        ) {
            if (filteredFailures.isNotEmpty() ||
                filteredPreflightAttention.isNotEmpty() ||
                sourceAttentionTracks.isNotEmpty()
            ) {
                item(key = "failure-heading") {
                    Row(
                        modifier = Modifier.fillMaxWidth()
                            .padding(start = 16.dp, end = 8.dp, top = 4.dp, bottom = 4.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Text(
                            "Needs attention (${formatIndexingTrackCount(filteredAttentionCount)})",
                            style = MaterialTheme.typography.titleSmall,
                            modifier = Modifier.weight(1f),
                        )
                        TextButton(
                            onClick = {
                                onNeverAllAttention(
                                    filteredFailures,
                                    filteredPreflightAttention,
                                    sourceAttentionTracks,
                                )
                            },
                        ) {
                            Text("Never index shown")
                        }
                    }
                }
                items(filteredFailures, key = { "${it.jobId}:${it.workId}" }) { failure ->
                    FailedTrackItem(
                        failure = failure,
                        onRetry = onRetryFailure,
                        onSelectForNewRun = onSelectFailureForNewRun,
                        onSkip = onSkipFailure,
                        onNever = onNeverFailure,
                        modifier = Modifier.padding(horizontal = 16.dp),
                    )
                }
                items(
                    filteredPreflightAttention,
                    key = {
                        "preflight:${it.rejection.jobId}:" +
                            "${it.rejection.providerSpan.normalizedPhysicalPath}:" +
                            "${it.rejection.providerSpan.offsetMs}:" +
                            it.rejection.providerSpan.durationMs
                    },
                ) { attention ->
                    PreflightAttentionItem(
                        attention = attention,
                        onTryAgain = onTryAgainPreflight,
                        onNever = onNeverPreflight,
                        modifier = Modifier.padding(horizontal = 16.dp),
                    )
                }
            }
            if (sourceAttentionTracks.isNotEmpty()) {
                item(key = "source-attention-heading") {
                    Column(
                        modifier = Modifier.fillMaxWidth()
                            .padding(horizontal = 16.dp, vertical = 12.dp),
                    ) {
                        Text(
                            "Missing usable audio details",
                            style = MaterialTheme.typography.titleSmall,
                        )
                        Text(
                            "Poweramp does not expose a complete playable entry for these files, " +
                                "so they cannot be indexed yet.",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                }
                items(sourceAttentionTracks, key = { "attention:${it.powerampFileId}" }) { track ->
                    SourceAttentionItem(
                        track = track,
                        onNever = {
                            onNeverAllAttention(emptyList(), emptyList(), listOf(track))
                        },
                        modifier = Modifier.padding(horizontal = 16.dp),
                    )
                }
            }
            if (readyTracks.isNotEmpty()) {
                item(key = "ready-heading") {
                    Text(
                        "Indexing candidates (${formatIndexingTrackCount(readyTracks.size)})",
                        style = MaterialTheme.typography.titleSmall,
                        modifier = Modifier.fillMaxWidth()
                            .padding(horizontal = 16.dp, vertical = 8.dp),
                    )
                }
            }
            items(readyTracks, key = { "ready:${it.powerampFileId}" }) { track ->
                TrackRow(
                    track = track,
                    isSelected = track.powerampFileId in selectedIds,
                    onToggle = { onToggle(track.powerampFileId) },
                )
            }
        }
    }
}

@Composable
private fun SourceAttentionItem(
    track: NewTrackDetector.UnindexedTrack,
    onNever: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Column(modifier = modifier.fillMaxWidth().padding(vertical = 8.dp)) {
        Text(
            track.title.ifBlank { "Unknown track" },
            style = MaterialTheme.typography.bodyMedium,
        )
        Text(
            buildString {
                if (track.artist.isNotBlank()) append(track.artist).append(" · ")
                append("No usable audio details")
            },
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        TextButton(onClick = onNever) { Text("Never index") }
    }
    HorizontalDivider()
}

@Composable
private fun TrackRow(
    track: NewTrackDetector.UnindexedTrack,
    isSelected: Boolean,
    onToggle: () -> Unit,
    allowNonReadySelection: Boolean = false,
) {
    val isSelectable = V2IndexingSelectionPolicy.canToggleTrackRow(
        track = track,
        allowNonReadySelection = allowNonReadySelection,
    )
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .toggleable(
                value = isSelected,
                enabled = isSelectable,
                role = Role.Checkbox,
                onValueChange = { onToggle() },
            )
            .padding(horizontal = 16.dp, vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Checkbox(
            checked = isSelected,
            enabled = isSelectable,
            onCheckedChange = null,
        )
        Spacer(modifier = Modifier.width(12.dp))
        Column(modifier = Modifier.weight(1f)) {
            Text(
                text = track.title.ifEmpty { "Unknown" },
                style = MaterialTheme.typography.bodyMedium,
                maxLines = 2,
                overflow = TextOverflow.Ellipsis,
            )
            val subtitle = buildString {
                when (track.detectionKind) {
                    V2UnindexedDetectionKind.SOURCE_ATTENTION ->
                        append(
                            if (isSelectable) {
                                "Duration measured during indexing"
                            } else {
                                "No usable duration"
                            },
                        )
                    V2UnindexedDetectionKind.DEFINITELY_UNINDEXED -> Unit
                }
                if (track.artist.isNotEmpty()) {
                    if (isNotEmpty()) append(" \u00b7 ")
                    append(track.artist)
                }
                if (track.durationMs > 0) {
                    if (isNotEmpty()) append(" \u00b7 ")
                    val durMin = track.durationMs / 60000
                    val durSec = (track.durationMs % 60000) / 1000
                    append("$durMin:${durSec.toString().padStart(2, '0')}")
                }
            }
            Text(
                text = subtitle,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                maxLines = 2,
                overflow = TextOverflow.Ellipsis,
            )
        }
    }
}

@Composable
private fun DurableJobContent(
    state: IndexingService.IndexingState.JobSnapshot,
    failures: List<IndexingViewModel.FailedTrackUi>,
    preflightAttention: List<V2PreflightAttentionTrack>,
    onPause: () -> Unit,
    onResume: () -> Unit,
    onCancel: () -> Unit,
    onRetry: () -> Unit,
    onRetryFailure: (IndexingViewModel.FailedTrackUi) -> Unit,
    onSelectFailureForNewRun: (IndexingViewModel.FailedTrackUi) -> Unit,
    onSkip: (IndexingViewModel.FailedTrackUi) -> Unit,
    onNever: (IndexingViewModel.FailedTrackUi) -> Unit,
    onTryAgainPreflight: (V2PreflightAttentionTrack) -> Unit,
    onNeverPreflight: (V2PreflightAttentionTrack) -> Unit,
    onDone: () -> Unit,
) {
    val terminal = state.jobState == IndexingJobState.COMPLETE ||
        state.jobState == IndexingJobState.CANCELLED
    val activeProgressState = state.jobState in setOf(
        IndexingJobState.RUNNING,
        IndexingJobState.PAUSE_REQUESTED,
        IndexingJobState.ACTIVATING,
        IndexingJobState.CANCELLING,
    )
    val stageEvidence = state.event
        ?.takeIf {
            activeProgressState && shouldShowIndexingStageEvent(it, state.progress)
        }
        ?.let(::indexingStageEvidence)
    val stageFallbackText = indexingStageFallbackText(
        state = state.jobState,
        progress = state.progress,
        hasVisibleStageEvent = stageEvidence != null,
    )
    val bulkRetryCount = failures.count {
        canUserRetryIndexingFailure(state.jobState, it.retryTrigger)
    }
    Scaffold(
        contentWindowInsets = WindowInsets(0, 0, 0, 0),
        bottomBar = {
            DurableJobActions(
                state = state.jobState,
                onPause = onPause,
                onResume = onResume,
                onCancel = onCancel,
                onRetry = onRetry,
                userRetryEligibleFailureCount = bulkRetryCount,
                onDone = onDone,
            )
        },
    ) { padding ->
        LazyColumn(
            modifier = Modifier.fillMaxSize().padding(padding),
            contentPadding = PaddingValues(horizontal = 24.dp, vertical = 20.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            item {
                Column(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalAlignment = Alignment.CenterHorizontally,
                ) {
                    Text(
                        indexingJobHeading(state.jobState, state.progress.totalTracks),
                        style = MaterialTheme.typography.headlineMedium,
                        color = MaterialTheme.colorScheme.primary,
                        textAlign = TextAlign.Center,
                    )
                    indexingStoppedReasonEvidence(
                        state = state.jobState,
                        stateReason = state.stateReason,
                    )?.let { reason ->
                        Spacer(modifier = Modifier.height(8.dp))
                        Text(
                            reason,
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            textAlign = TextAlign.Center,
                        )
                    }
                    Spacer(modifier = Modifier.height(12.dp))
                    stageEvidence?.let { evidence ->
                        Text(
                            evidence.text,
                            style = MaterialTheme.typography.bodyMedium,
                            textAlign = TextAlign.Center,
                        )
                    }
                    if (stageEvidence == null && stageFallbackText != null) {
                        Text(
                            stageFallbackText,
                            style = MaterialTheme.typography.bodyMedium,
                            textAlign = TextAlign.Center,
                        )
                    }
                    if (activeProgressState) {
                        Spacer(modifier = Modifier.height(8.dp))
                        if (stageEvidence?.fraction != null) {
                            LinearProgressIndicator(
                                progress = { stageEvidence.fraction },
                                modifier = Modifier.fillMaxWidth(),
                            )
                        } else {
                            LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                        }
                    }
                    formatCurrentIndexingTrack(
                        activeTrackOrdinal = state.progress.activeTrackOrdinal,
                        eventTrackOrdinal = state.event?.trackOrdinal,
                        eventTrackTitle = state.event?.trackTitle,
                        totalTracks = state.progress.totalTracks,
                    )?.let { currentTrack ->
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(
                            currentTrack,
                            style = MaterialTheme.typography.bodyMedium,
                            textAlign = TextAlign.Center,
                        )
                    }
                    val durableStageCounts =
                        formatDurableStageTrackCounts(state.jobState, state.progress)
                    durableStageCounts?.let { stages ->
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(
                            stages,
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            textAlign = TextAlign.Center,
                        )
                    }
                    val showSavedOutcome = terminal ||
                        state.progress.succeededTracks > 0 ||
                        state.progress.blockedTracks > 0 ||
                        state.progress.skippedTracks > 0
                    if (showSavedOutcome) {
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(
                            formatDurableTrackCounts(state.jobState, state.progress),
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            textAlign = TextAlign.Center,
                        )
                    }
                    formatPreflightAttentionSummary(preflightAttention.size)?.let { summary ->
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(
                            summary,
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            textAlign = TextAlign.Center,
                        )
                    }
                    val etaText = formatIndexingEtaEvidence(state.eta, state.etaCoverage)
                    if (etaText != null && shouldShowIndexingEta(state.jobState)) {
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(
                            etaText,
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            textAlign = TextAlign.Center,
                        )
                    }
                }
            }

            if (failures.isNotEmpty() || preflightAttention.isNotEmpty()) {
                item {
                    Text(
                        "Needs attention (" +
                            "${formatIndexingTrackCount(failures.size + preflightAttention.size)})",
                        style = MaterialTheme.typography.titleSmall,
                        modifier = Modifier.fillMaxWidth(),
                    )
                }
                items(failures, key = { "${it.jobId}:${it.workId}" }) { failure ->
                    FailedTrackItem(
                        failure = failure,
                        onRetry = onRetryFailure,
                        onSelectForNewRun = onSelectFailureForNewRun,
                        onSkip = onSkip,
                        onNever = onNever,
                    )
                }
                items(
                    preflightAttention,
                    key = {
                        "preflight:${it.rejection.providerSpan.normalizedPhysicalPath}:" +
                            "${it.rejection.providerSpan.offsetMs}:" +
                            it.rejection.providerSpan.durationMs
                    },
                ) { attention ->
                    PreflightAttentionItem(
                        attention = attention,
                        onTryAgain = onTryAgainPreflight,
                        onNever = onNeverPreflight,
                    )
                }
            }
        }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun DurableJobActions(
    state: IndexingJobState,
    onPause: () -> Unit,
    onResume: () -> Unit,
    onCancel: () -> Unit,
    onRetry: () -> Unit,
    userRetryEligibleFailureCount: Int,
    onDone: () -> Unit,
) {
    val canCancel = canDiscardIndexingJob(state)
    val hasPrimaryAction = when (state) {
        IndexingJobState.ACTIVATING,
        IndexingJobState.CANCELLING,
        -> false
        IndexingJobState.WAITING_FOR_INPUT -> userRetryEligibleFailureCount > 0
        else -> true
    }
    if (!hasPrimaryAction && !canCancel) return

    Surface(tonalElevation = 3.dp) {
        FlowRow(
            modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 10.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp, Alignment.End),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            when (state) {
                IndexingJobState.RUNNING ->
                    FilledTonalButton(onClick = onPause) { Text("Pause") }
                IndexingJobState.PAUSE_REQUESTED ->
                    FilledTonalButton(onClick = {}, enabled = false) { Text("Pausing...") }
                IndexingJobState.PLANNED,
                IndexingJobState.PAUSED,
                IndexingJobState.INTERRUPTED,
                IndexingJobState.READY_TO_RESUME,
                -> FilledTonalButton(onClick = onResume) { Text("Resume") }
                IndexingJobState.WAITING_FOR_INPUT if userRetryEligibleFailureCount > 0 ->
                    FilledTonalButton(onClick = onRetry) {
                        Text(retryAvailableTracksLabel(userRetryEligibleFailureCount))
                    }
                IndexingJobState.WAITING_FOR_INPUT -> Unit
                IndexingJobState.COMPLETE,
                IndexingJobState.CANCELLED,
                -> Button(onClick = onDone) { Text("Done") }
                IndexingJobState.ACTIVATING,
                IndexingJobState.CANCELLING,
                -> Unit
            }
            if (canCancel) {
                OutlinedButton(onClick = onCancel) { Text("Discard run") }
            }
        }
    }
}

@Composable
private fun ErrorContent(
    message: String,
    actionLabel: String = "Back",
    onBack: () -> Unit,
) {
    Box(modifier = Modifier.fillMaxSize().padding(24.dp), contentAlignment = Alignment.Center) {
        Column(
            modifier = Modifier.fillMaxWidth().verticalScroll(rememberScrollState()),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Text(
                "Error",
                style = MaterialTheme.typography.headlineSmall,
                color = MaterialTheme.colorScheme.error,
            )
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                message,
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Spacer(modifier = Modifier.height(24.dp))
            OutlinedButton(onClick = onBack) {
                Text(actionLabel)
            }
        }
    }
}

@Composable
private fun ExecutionProfileControl(
    selected: V2IndexingExecutionProfile,
    onSelected: (V2IndexingExecutionProfile) -> Unit,
    showFullDescription: Boolean = true,
) {
    if (userSelectableIndexingProfiles.size <= 1) return
    BoxWithConstraints(modifier = Modifier.fillMaxWidth()) {
        val compact = useCompactIndexingProfileControl(
            maxWidthDp = maxWidth.value,
            fontScale = LocalDensity.current.fontScale,
        )
        if (compact) {
            var menuExpanded by remember { mutableStateOf(false) }
            Box(modifier = Modifier.fillMaxWidth()) {
                OutlinedButton(
                    onClick = { menuExpanded = true },
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Text(
                        indexingExecutionProfileLabel(selected),
                        modifier = Modifier.weight(1f),
                        textAlign = TextAlign.Start,
                    )
                    Icon(Icons.Default.ArrowDropDown, contentDescription = null)
                }
                DropdownMenu(
                    expanded = menuExpanded,
                    onDismissRequest = { menuExpanded = false },
                ) {
                    userSelectableIndexingProfiles.forEach { profile ->
                        DropdownMenuItem(
                            text = { Text(indexingExecutionProfileLabel(profile)) },
                            onClick = {
                                menuExpanded = false
                                onSelected(profile)
                            },
                        )
                    }
                }
            }
        } else {
            SingleChoiceSegmentedButtonRow(modifier = Modifier.fillMaxWidth()) {
                userSelectableIndexingProfiles.forEachIndexed { index, profile ->
                    SegmentedButton(
                        selected = selected == profile,
                        onClick = { onSelected(profile) },
                        shape = SegmentedButtonDefaults.itemShape(
                            index = index,
                            count = userSelectableIndexingProfiles.size,
                        ),
                    ) {
                        Text(indexingExecutionProfileLabel(profile))
                    }
                }
            }
        }
    }
    Spacer(modifier = Modifier.height(4.dp))
    Text(
        if (showFullDescription) {
            indexingExecutionProfileDescription(selected)
        } else {
            indexingExecutionProfileCompactDescription(selected)
        },
        style = MaterialTheme.typography.bodySmall,
        color = MaterialTheme.colorScheme.onSurfaceVariant,
    )
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun PreflightAttentionItem(
    attention: V2PreflightAttentionTrack,
    onTryAgain: (V2PreflightAttentionTrack) -> Unit,
    onNever: (V2PreflightAttentionTrack) -> Unit,
    modifier: Modifier = Modifier,
) {
    val actions = V2PreflightRejectionPolicy.actionsFor(attention)
    val track = attention.currentTrack
    val rejection = attention.rejection
    Column(modifier = modifier.fillMaxWidth().padding(vertical = 8.dp)) {
        Text(
            track.title.ifBlank { "Unknown track" },
            style = MaterialTheme.typography.bodyMedium,
        )
        Text(
            buildString {
                if (track.artist.isNotBlank()) append(track.artist)
                if (track.album.isNotBlank()) {
                    if (isNotEmpty()) append(" · ")
                    append(track.album)
                }
            },
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Text(
            preflightRejectionSummary(rejection.code),
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            maxLines = 3,
        )
        FlowRow(
            horizontalArrangement = Arrangement.spacedBy(4.dp),
            verticalArrangement = Arrangement.spacedBy(4.dp),
        ) {
            if (V2PreflightAttentionAction.TRY_AGAIN in actions) {
                TextButton(onClick = { onTryAgain(attention) }) { Text("Select to retry") }
            }
            if (V2PreflightAttentionAction.NEVER_INDEX in actions) {
                TextButton(onClick = { onNever(attention) }) { Text("Never index") }
            }
        }
        preflightTryAgainUnavailableReason(attention)?.let { reason ->
            Text(
                reason,
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
    HorizontalDivider()
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun FailedTrackItem(
    failure: IndexingViewModel.FailedTrackUi,
    onRetry: (IndexingViewModel.FailedTrackUi) -> Unit,
    onSelectForNewRun: (IndexingViewModel.FailedTrackUi) -> Unit,
    onSkip: (IndexingViewModel.FailedTrackUi) -> Unit,
    onNever: (IndexingViewModel.FailedTrackUi) -> Unit,
    modifier: Modifier = Modifier,
) {
    Column(modifier = modifier.fillMaxWidth().padding(vertical = 8.dp)) {
        Text(
            failure.title.ifBlank { "Unknown track" },
            style = MaterialTheme.typography.bodyMedium,
        )
        Text(
            buildString {
                if (failure.artist.isNotBlank()) append(failure.artist).append(" · ")
                append(indexingFailureSummary(failure.code))
                if (failure.occurrences > 1) {
                    append(" · ${formatFailureOccurrences(failure.occurrences)}")
                }
            },
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        if (canOfferIndexingFailureActions(failure.jobState)) {
            val canRetry = canUserRetryIndexingFailure(
                failure.jobState,
                failure.retryTrigger,
            )
            FlowRow(
                horizontalArrangement = Arrangement.spacedBy(4.dp),
                verticalArrangement = Arrangement.spacedBy(4.dp),
            ) {
                if (canRetry) {
                    TextButton(onClick = { onRetry(failure) }) { Text("Retry") }
                }
                TextButton(onClick = { onSkip(failure) }) { Text("Skip this run") }
                TextButton(onClick = { onNever(failure) }) { Text("Never index") }
            }
            if (!canRetry || failure.disposition == FailureDisposition.BLOCKED) {
                Text(
                    indexingFailureGuidance(failure.code, failure.retryTrigger),
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        } else if (canNeverIndexFailure(failure.jobState)) {
            val canSelectForNewRun = canSelectIndexingFailureForNewRun(
                state = failure.jobState,
                retryTrigger = failure.retryTrigger,
                failureCode = failure.code,
            )
            FlowRow(
                horizontalArrangement = Arrangement.spacedBy(4.dp),
                verticalArrangement = Arrangement.spacedBy(4.dp),
            ) {
                if (canSelectForNewRun) {
                    TextButton(onClick = { onSelectForNewRun(failure) }) {
                        Text("Select for new run")
                    }
                }
                TextButton(onClick = { onNever(failure) }) { Text("Never index") }
            }
            Text(
                indexingFailureGuidance(failure.code, failure.retryTrigger),
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
    HorizontalDivider()
}

@Composable
private fun CompleteContent(indexed: Int, failed: Int, onDone: () -> Unit) {
    Box(modifier = Modifier.fillMaxSize().padding(32.dp), contentAlignment = Alignment.Center) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text(
                "Indexing complete",
                style = MaterialTheme.typography.headlineSmall,
                color = MaterialTheme.colorScheme.primary,
            )
            Spacer(modifier = Modifier.height(8.dp))
            val failedCount = formatIndexingTrackCount(failed)
            val message = if (failed > 0) {
                "${formatIndexingTrackQuantity(indexed)} indexed, $failedCount failed"
            } else {
                "${formatIndexingTrackQuantity(indexed)} indexed"
            }
            Text(
                message,
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Spacer(modifier = Modifier.height(24.dp))
            FilledTonalButton(onClick = onDone) {
                Text("Done")
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun NeverIndexScreen(
    viewModel: IndexingViewModel,
    onBack: () -> Unit,
) {
    val dismissedTracks = remember(viewModel.dismissedIds.collectAsState().value) {
        viewModel.getDismissedTracks()
    }
    var localSelected by rememberSaveable(stateSaver = LongSetSaver) {
        mutableStateOf(emptySet<Long>())
    }

    // Clean up local selection if tracks change
    LaunchedEffect(dismissedTracks) {
        val validIds = dismissedTracks.map { it.powerampFileId }.toSet()
        localSelected = localSelected.intersect(validIds)
    }

    Scaffold(
        contentWindowInsets = WindowInsets.safeDrawing,
        topBar = {
            TopAppBar(
                title = { Text("Never-index list") },
                windowInsets = WindowInsets.safeDrawing.only(
                    WindowInsetsSides.Horizontal + WindowInsetsSides.Top,
                ),
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back")
                    }
                },
            )
        },
        bottomBar = {
            if (localSelected.isNotEmpty()) {
                Surface(tonalElevation = 3.dp) {
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .windowInsetsPadding(
                                WindowInsets.safeDrawing.only(
                                    WindowInsetsSides.Horizontal + WindowInsetsSides.Bottom,
                                ),
                            )
                            .padding(horizontal = 16.dp, vertical = 10.dp),
                    ) {
                        Button(onClick = {
                            viewModel.restoreFromDismissed(localSelected)
                            localSelected = emptySet()
                        }, modifier = Modifier.fillMaxWidth()) {
                            Text(
                                "Restore selected (${formatIndexingTrackCount(localSelected.size)})",
                            )
                        }
                    }
                }
            }
        }
    ) { padding ->
        if (dismissedTracks.isEmpty()) {
            Box(
                modifier = Modifier.fillMaxSize().padding(padding),
                contentAlignment = Alignment.Center,
            ) {
                Text(
                    "No never-index tracks",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        } else {
            Column(modifier = Modifier.fillMaxSize().padding(padding)) {
                val allIds = remember(dismissedTracks) {
                    dismissedTracks.map { it.powerampFileId }.toSet()
                }
                val allSelected = localSelected.size == dismissedTracks.size && dismissedTracks.isNotEmpty()
                val toggleAll = {
                    localSelected = if (allSelected) emptySet() else allIds
                }
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .triStateToggleable(
                            state = when {
                                dismissedTracks.isEmpty() -> ToggleableState.Off
                                allSelected -> ToggleableState.On
                                localSelected.isEmpty() -> ToggleableState.Off
                                else -> ToggleableState.Indeterminate
                            },
                            role = Role.Checkbox,
                            onClick = toggleAll,
                        )
                        .padding(horizontal = 16.dp, vertical = 8.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    TriStateCheckbox(
                        state = when {
                            dismissedTracks.isEmpty() -> ToggleableState.Off
                            allSelected -> ToggleableState.On
                            localSelected.isEmpty() -> ToggleableState.Off
                            else -> ToggleableState.Indeterminate
                        },
                        onClick = null,
                    )
                    Spacer(modifier = Modifier.width(12.dp))
                    Text(
                        "${formatIndexingTrackCount(dismissedTracks.size)} never-index tracks" +
                            if (localSelected.isNotEmpty()) {
                                " (${formatIndexingTrackCount(localSelected.size)} selected)"
                            } else {
                                ""
                            },
                        style = MaterialTheme.typography.titleSmall,
                    )
                }
                HorizontalDivider()
                LazyColumn(
                    modifier = Modifier.fillMaxSize(),
                    contentPadding = PaddingValues(vertical = 4.dp),
                ) {
                    items(dismissedTracks, key = { it.powerampFileId }) { track ->
                        TrackRow(
                            track = track,
                            isSelected = track.powerampFileId in localSelected,
                            allowNonReadySelection = true,
                            onToggle = {
                                localSelected = if (track.powerampFileId in localSelected) {
                                    localSelected - track.powerampFileId
                                } else {
                                    localSelected + track.powerampFileId
                                }
                            },
                        )
                    }
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun CleanDatabaseScreen(
    state: DatabaseCleanupScanState,
    blockedReason: String?,
    onRefresh: () -> Unit,
    onDelete: (Set<Long>) -> Unit,
    onBack: () -> Unit,
) {
    val tracks = (state as? DatabaseCleanupScanState.Ready)?.tracks.orEmpty()
    val isDetecting = state is DatabaseCleanupScanState.Scanning
    val status = when (state) {
        DatabaseCleanupScanState.Idle -> ""
        is DatabaseCleanupScanState.Scanning -> state.message
        is DatabaseCleanupScanState.Ready -> state.message
        is DatabaseCleanupScanState.Failed -> state.message
    }
    var localSelected by rememberSaveable(stateSaver = LongSetSaver) {
        mutableStateOf(emptySet<Long>())
    }
    // Preserve the selection, but require a fresh review after recreation instead of saving it twice.
    var pendingConfirmationIds by remember {
        mutableStateOf(emptySet<Long>())
    }
    val refreshEnabled = !isDetecting && blockedReason == null
    val actionsEnabled = state is DatabaseCleanupScanState.Ready && blockedReason == null

    LaunchedEffect(tracks) {
        val validIds = tracks.map { it.id }.toSet()
        localSelected = localSelected.intersect(validIds)
    }

    LaunchedEffect(blockedReason) {
        if (blockedReason != null) pendingConfirmationIds = emptySet()
    }


    pendingConfirmationIds.takeIf(Set<Long>::isNotEmpty)?.let { ids ->
        val confirmation = V2CleanDatabaseConfirmation.create(ids)
        AlertDialog(
            onDismissRequest = { pendingConfirmationIds = emptySet() },
            title = {
                Text(
                    "Remove ${formatIndexingTrackQuantity(confirmation.exactCount)} " +
                        "from the index?",
                )
            },
            text = {
                Text(
                    "The selected tracks will be removed from the music index. " +
                        "Poweramp files are not deleted.",
                )
            },
            dismissButton = {
                TextButton(onClick = { pendingConfirmationIds = emptySet() }) { Text("Cancel") }
            },
            confirmButton = {
                Button(
                    onClick = {
                        pendingConfirmationIds = emptySet()
                        localSelected = localSelected - confirmation.trackIds
                        onDelete(confirmation.trackIds)
                    },
                    enabled = actionsEnabled,
                ) {
                    Text("Remove ${formatIndexingTrackQuantity(confirmation.exactCount)}")
                }
            },
        )
    }

    Scaffold(
        contentWindowInsets = WindowInsets.safeDrawing,
        topBar = {
            TopAppBar(
                title = { Text("Remove missing tracks") },
                windowInsets = WindowInsets.safeDrawing.only(
                    WindowInsetsSides.Horizontal + WindowInsetsSides.Top,
                ),
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back")
                    }
                },
                actions = {
                    IconButton(onClick = onRefresh, enabled = refreshEnabled) {
                        Icon(Icons.Default.Refresh, contentDescription = "Refresh")
                    }
                }
            )
        },
        bottomBar = {
            if (localSelected.isNotEmpty()) {
                Surface(tonalElevation = 3.dp) {
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .windowInsetsPadding(
                                WindowInsets.safeDrawing.only(
                                    WindowInsetsSides.Horizontal + WindowInsetsSides.Bottom,
                                ),
                            )
                            .padding(horizontal = 16.dp, vertical = 10.dp),
                        verticalArrangement = Arrangement.spacedBy(8.dp),
                    ) {
                        Text(
                            "${formatIndexingTrackCount(localSelected.size)} selected",
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                        blockedReason?.let { reason ->
                            Text(
                                reason,
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.error,
                            )
                        }
                        Button(
                            onClick = {
                                pendingConfirmationIds = localSelected.toSet()
                            },
                            enabled = actionsEnabled,
                            modifier = Modifier.fillMaxWidth(),
                        ) {
                            Text("Review removal")
                        }
                    }
                }
            }
        }
    ) { padding ->
        when {
            isDetecting -> Box(modifier = Modifier.fillMaxSize().padding(padding)) {
                DetectingContent(
                    status = status.ifEmpty {
                        "Comparing exact indexed source spans with Poweramp tracks"
                    },
                )
            }
            state is DatabaseCleanupScanState.Failed -> {
                Box(
                    modifier = Modifier.fillMaxSize().padding(padding).padding(24.dp),
                    contentAlignment = Alignment.Center,
                ) {
                    Column(
                        modifier = Modifier.fillMaxWidth().verticalScroll(rememberScrollState()),
                        horizontalAlignment = Alignment.CenterHorizontally,
                    ) {
                        Text(
                            "Could not compare the music index with Poweramp",
                            style = MaterialTheme.typography.headlineSmall,
                            color = MaterialTheme.colorScheme.error,
                            textAlign = TextAlign.Center,
                        )
                        Spacer(modifier = Modifier.height(8.dp))
                        Text(
                            state.message,
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            textAlign = TextAlign.Center,
                        )
                        Spacer(modifier = Modifier.height(20.dp))
                        FilledTonalButton(onClick = onRefresh, enabled = refreshEnabled) {
                            Text("Run comparison again")
                        }
                    }
                }
            }
            state is DatabaseCleanupScanState.Idle -> {
                Box(
                    modifier = Modifier.fillMaxSize().padding(padding),
                    contentAlignment = Alignment.Center,
                ) {
                    FilledTonalButton(onClick = onRefresh, enabled = refreshEnabled) {
                        Text("Compare Poweramp with music index")
                    }
                }
            }
            tracks.isEmpty() -> {
                Box(
                    modifier = Modifier.fillMaxSize().padding(padding),
                    contentAlignment = Alignment.Center,
                ) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Text(
                            "No missing tracks found",
                            style = MaterialTheme.typography.headlineSmall,
                            color = MaterialTheme.colorScheme.primary,
                            textAlign = TextAlign.Center,
                        )
                        Spacer(modifier = Modifier.height(8.dp))
                        Text(
                            "No safely removable indexed tracks were found in the current " +
                                "Poweramp library.",
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            textAlign = TextAlign.Center,
                            modifier = Modifier.padding(horizontal = 24.dp),
                        )
                    }
                }
            }
            else -> {
                val allIds = remember(tracks) { tracks.map { it.id }.toSet() }
                val allSelected = localSelected.size == tracks.size
                val toggleAll = {
                    localSelected = if (allSelected) emptySet() else allIds
                }

                Column(modifier = Modifier.fillMaxSize().padding(padding)) {
                    blockedReason?.let { reason ->
                        Text(
                            text = reason,
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.error,
                            modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp),
                        )
                    }
                    if (status.isNotBlank()) {
                        Text(
                            text = status,
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp),
                        )
                    }
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .triStateToggleable(
                                state = when {
                                    tracks.isEmpty() -> ToggleableState.Off
                                    allSelected -> ToggleableState.On
                                    localSelected.isEmpty() -> ToggleableState.Off
                                    else -> ToggleableState.Indeterminate
                                },
                                role = Role.Checkbox,
                                onClick = toggleAll,
                            )
                            .padding(horizontal = 16.dp, vertical = 8.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        TriStateCheckbox(
                            state = when {
                                tracks.isEmpty() -> ToggleableState.Off
                                allSelected -> ToggleableState.On
                                localSelected.isEmpty() -> ToggleableState.Off
                                else -> ToggleableState.Indeterminate
                            },
                            onClick = null,
                        )
                        Spacer(modifier = Modifier.width(12.dp))
                        Text(
                            "${formatIndexingTrackQuantity(tracks.size)} indexed, no longer in Poweramp" +
                                if (localSelected.isNotEmpty()) {
                                    " (${formatIndexingTrackCount(localSelected.size)} selected)"
                                } else {
                                    ""
                                },
                            style = MaterialTheme.typography.titleSmall,
                        )
                    }
                    HorizontalDivider()
                    LazyColumn(
                        modifier = Modifier.fillMaxSize(),
                        contentPadding = PaddingValues(vertical = 4.dp),
                    ) {
                        items(tracks, key = { it.id }) { track ->
                            EmbeddedTrackRow(
                                track = track,
                                isSelected = track.id in localSelected,
                                onToggle = {
                                    localSelected = if (track.id in localSelected) {
                                        localSelected - track.id
                                    } else {
                                        localSelected + track.id
                                    }
                                },
                            )
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun EmbeddedTrackRow(
    track: EmbeddedTrack,
    isSelected: Boolean,
    onToggle: () -> Unit,
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .toggleable(
                value = isSelected,
                role = Role.Checkbox,
                onValueChange = { onToggle() },
            )
            .padding(horizontal = 16.dp, vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Checkbox(checked = isSelected, onCheckedChange = null)
        Spacer(modifier = Modifier.width(12.dp))
        Column(modifier = Modifier.weight(1f)) {
            Text(
                text = track.title?.takeIf { it.isNotBlank() } ?: "Unknown",
                style = MaterialTheme.typography.bodyMedium,
                maxLines = 2,
                overflow = TextOverflow.Ellipsis,
            )
            val subtitle = buildString {
                if (!track.artist.isNullOrBlank()) append(track.artist)
                if (!track.album.isNullOrBlank()) {
                    if (isNotEmpty()) append(" · ")
                    append(track.album)
                }
                indexedTrackSourceLabel(track.source)?.let { sourceLabel ->
                    if (isNotEmpty()) append(" · ")
                    append(sourceLabel)
                }
            }
            if (subtitle.isNotBlank()) {
                Text(
                    text = subtitle,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    maxLines = 2,
                    overflow = TextOverflow.Ellipsis,
                )
            }
        }
    }
}

@Composable
private fun BottomBar(
    selectedCount: Int,
    visibleSelectedCount: Int,
    isFiltered: Boolean,
    hasModels: Boolean,
    isAppFilesChecking: Boolean,
    hasAudioAccess: Boolean,
    onRequestAudioAccess: () -> Unit,
    onStartIndexing: () -> Unit,
) {
    Surface(tonalElevation = 3.dp) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .windowInsetsPadding(
                    WindowInsets.safeDrawing.only(
                        WindowInsetsSides.Horizontal + WindowInsetsSides.Bottom,
                    ),
                )
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 16.dp, vertical = 12.dp),
        ) {
            FilledTonalButton(
                onClick = onStartIndexing,
                enabled = selectedCount > 0 && !isAppFilesChecking && hasModels && hasAudioAccess,
                modifier = Modifier.fillMaxWidth(),
            ) {
                Text(
                    startIndexingActionLabel(
                        selectedCount = selectedCount,
                        visibleSelectedCount = visibleSelectedCount,
                        isFiltered = isFiltered,
                    ),
                    textAlign = TextAlign.Center,
                )
            }
            if (isAppFilesChecking) {
                Spacer(modifier = Modifier.height(6.dp))
                Text(
                    "Reading the active index pointer and four required indexing filenames",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            } else if (!hasModels) {
                Spacer(modifier = Modifier.height(6.dp))
                Text(
                    "One or more required indexing files are missing or unreadable. " +
                        "Review file details in Settings.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            } else if (!hasAudioAccess) {
                Spacer(modifier = Modifier.height(6.dp))
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        "Music and audio access is required to decode selected tracks.",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.weight(1f),
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    TextButton(onClick = onRequestAudioAccess) {
                        Text("Allow")
                    }
                }
            }
        }
    }
}

internal fun startIndexingActionLabel(
    selectedCount: Int,
    visibleSelectedCount: Int,
    isFiltered: Boolean,
): String {
    val selected = formatIndexingTrackCount(selectedCount)
    val noun = if (selectedCount == 1) "track" else "tracks"
    return if (isFiltered) {
        "Index $selected $noun " +
            "(${formatIndexingTrackCount(visibleSelectedCount)} shown)"
    } else {
        "Index $selected $noun"
    }
}
