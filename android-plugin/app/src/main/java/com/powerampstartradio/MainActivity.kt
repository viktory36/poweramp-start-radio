package com.powerampstartradio

import android.content.Context
import android.net.Uri
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.slideInHorizontally
import androidx.compose.animation.slideOutHorizontally
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.background
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.selection.selectableGroup
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Clear
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.Menu
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Search
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.Modifier
import androidx.compose.ui.focus.FocusRequester
import androidx.compose.ui.focus.focusRequester
import androidx.compose.ui.focus.onFocusChanged
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.stateDescription
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.powerampstartradio.poweramp.PowerampReceiver
import com.powerampstartradio.poweramp.PowerampTrack
import com.powerampstartradio.indexing.IndexingService
import com.powerampstartradio.indexing.formatCurrentIndexingTrack
import com.powerampstartradio.indexing.formatDurableStageTrackCounts
import com.powerampstartradio.indexing.formatDurableTrackCounts
import com.powerampstartradio.indexing.formatIndexingEtaEvidence
import com.powerampstartradio.indexing.formatIndexingTrackCount
import com.powerampstartradio.indexing.indexingStageFallbackText
import com.powerampstartradio.indexing.indexingStageEvidence
import com.powerampstartradio.indexing.preflightStatusEvidenceText
import com.powerampstartradio.indexing.shouldShowIndexingStageEvent
import com.powerampstartradio.indexing.shouldShowIndexingEta
import com.powerampstartradio.indexing.v2.IndexingJobState
import com.powerampstartradio.indexing.v2.V2IndexingPreflightIntentState
import com.powerampstartradio.indexing.v2.V2IndexingPreflightProgressUnit
import com.powerampstartradio.services.RadioService
import com.powerampstartradio.ui.AppFileStatus
import com.powerampstartradio.ui.ArtistConstraintControlPolicy
import com.powerampstartradio.ui.DatabaseInfo
import com.powerampstartradio.ui.DecaySchedule
import com.powerampstartradio.ui.DirectQueuePlacement
import com.powerampstartradio.ui.DEFAULT_LIBRARY_ADDED_DAYS
import com.powerampstartradio.ui.DppDomainControlPolicy
import com.powerampstartradio.ui.DriftControlPolicy
import com.powerampstartradio.ui.DriftMode
import com.powerampstartradio.ui.FindMusicOperator
import com.powerampstartradio.ui.FindMusicEditorWeightPolicy
import com.powerampstartradio.ui.FindMusicEditorWeightSlot
import com.powerampstartradio.ui.FindMusicEditorPolicy
import com.powerampstartradio.ui.FindMusicQuerySpec
import com.powerampstartradio.ui.FindMusicRefineNeighborhood
import com.powerampstartradio.ui.FindMusicResultKind
import com.powerampstartradio.ui.FindMusicTextResultPlanner
import com.powerampstartradio.ui.MainViewModel
import com.powerampstartradio.ui.LibraryRankEvidenceText
import com.powerampstartradio.ui.MAX_LIBRARY_ADDED_DAYS
import com.powerampstartradio.ui.MmrControlPolicy
import com.powerampstartradio.ui.NeighborhoodReachPolicy
import com.powerampstartradio.ui.RadioConfig
import com.powerampstartradio.ui.QueueStatus
import com.powerampstartradio.ui.QueueOrigin
import com.powerampstartradio.ui.QueuedTrackResult
import com.powerampstartradio.ui.RadioResult
import com.powerampstartradio.ui.RadioUiState
import com.powerampstartradio.ui.RecordingCandidateEvidenceFormatter
import com.powerampstartradio.ui.RecordingLookupState
import com.powerampstartradio.ui.SessionReplayEligibility
import com.powerampstartradio.ui.SessionEvidenceText
import com.powerampstartradio.ui.SelectionKnobPolicy
import com.powerampstartradio.ui.SelectionControlText
import com.powerampstartradio.ui.SelectionMode
import com.powerampstartradio.ui.SeedDistanceEvidencePolicy
import com.powerampstartradio.ui.SettingsPeekPreview
import com.powerampstartradio.ui.SettingsPeekInteractionPolicy
import com.powerampstartradio.ui.SettingsAppFilesText
import com.powerampstartradio.ui.SongSeedState
import com.powerampstartradio.ui.TextSearchResult
import com.powerampstartradio.ui.TextIngredientState
import com.powerampstartradio.ui.forSeed
import com.powerampstartradio.ui.effectiveLibraryAddedDays
import com.powerampstartradio.ui.hasSameExecutionControlsAs
import com.powerampstartradio.ui.libraryAddedDaysLabel
import com.powerampstartradio.ui.recordingDisplayLabel
import com.powerampstartradio.ui.replayEligibilityKey
import com.powerampstartradio.ui.theme.PowerampStartRadioTheme
import com.powerampstartradio.widget.StartRadioWidgetReceiver
import kotlinx.coroutines.launch
import java.util.Locale
import kotlin.math.roundToInt

/** Whether the radio UI state represents an active search (any phase). */
private fun RadioUiState.isActiveSearch(): Boolean =
    this is RadioUiState.Loading || this is RadioUiState.Searching || this is RadioUiState.Streaming

private const val PLAY_A_TRACK_STATUS = "Play a song in Poweramp first"
private const val MUSIC_INDEX_CHECK_STATUS = "Opening the saved music index"
private const val IMPORT_MUSIC_INDEX_STATUS = "Import a music index in Settings"
private const val FIND_MUSIC_MODEL_STATUS =
    "Find Music needs clamp3_text.tflite and sentencepiece.bpe.model"

private fun homePrerequisiteStatus(
    currentTrack: PowerampTrack?,
    databaseInfo: DatabaseInfo?,
    databaseLoading: Boolean,
): String? = when {
    currentTrack == null -> PLAY_A_TRACK_STATUS
    databaseLoading -> MUSIC_INDEX_CHECK_STATUS
    databaseInfo == null -> IMPORT_MUSIC_INDEX_STATUS
    else -> null
}

private fun findMusicPrerequisiteStatus(
    context: Context,
    databaseInfo: DatabaseInfo?,
    databaseLoading: Boolean,
): String? = when {
    databaseLoading -> MUSIC_INDEX_CHECK_STATUS
    databaseInfo == null -> IMPORT_MUSIC_INDEX_STATUS
    !java.io.File(context.filesDir, "clamp3_text.tflite").isFile ||
        !java.io.File(context.filesDir, "sentencepiece.bpe.model").isFile ->
        FIND_MUSIC_MODEL_STATUS
    else -> null
}

class MainActivity : ComponentActivity() {

    private var onResumeCallback: (() -> Unit)? = null
    private val audioPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) {}

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        setContent {
            PowerampStartRadioTheme {
                Surface(
                    modifier = Modifier.fillMaxSize().imePadding(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    MainScreen(
                        onRegisterResumeCallback = { callback ->
                            onResumeCallback = callback
                        }
                    )
                }
            }
        }
        requestAudioLibraryPermission(savedInstanceState)
    }

    override fun onResume() {
        super.onResume()
        onResumeCallback?.invoke()
    }

    private fun requestAudioLibraryPermission(savedInstanceState: Bundle?) {
        if (AudioLibraryPermission.shouldPrompt(
                freshActivityStart = savedInstanceState == null,
                granted = AudioLibraryPermission.isGranted(this),
            )
        ) {
            audioPermissionLauncher.launch(AudioLibraryPermission.permissionName())
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MainScreen(
    viewModel: MainViewModel = viewModel(),
    onRegisterResumeCallback: ((callback: () -> Unit) -> Unit)? = null
) {
    val context = LocalContext.current

    val radioState by viewModel.radioState.collectAsState()
    val databaseInfo by viewModel.databaseInfo.collectAsState()
    val databaseLoading by viewModel.databaseLoading.collectAsState()
    val databaseVerificationStatus by viewModel.databaseVerificationStatus.collectAsState()
    val hasPermission by viewModel.hasPermission.collectAsState()
    val permissionLoading by viewModel.permissionLoading.collectAsState()
    val sessionHistory by viewModel.sessionHistory.collectAsState()
    val sessionReplayEligibility by viewModel.sessionReplayEligibility.collectAsState()

    val scope = rememberCoroutineScope()
    var currentTrack by remember { mutableStateOf(PowerampReceiver.currentTrack) }
    var showSettings by rememberSaveable { mutableStateOf(false) }
    var showTextSearch by rememberSaveable { mutableStateOf(false) }
    var statusMessage by rememberSaveable { mutableStateOf("") }
    var viewingSession by rememberSaveable { mutableStateOf<Int?>(null) }
    var pendingHistoryReplay by remember { mutableStateOf<RadioResult?>(null) }

    LaunchedEffect(radioState) {
        if (radioState is RadioUiState.Idle) viewingSession = null
    }

    LaunchedEffect(currentTrack, databaseInfo, databaseLoading) {
        val prerequisiteMessages = setOf(
            PLAY_A_TRACK_STATUS,
            MUSIC_INDEX_CHECK_STATUS,
            IMPORT_MUSIC_INDEX_STATUS,
            FIND_MUSIC_MODEL_STATUS,
        )
        val currentPrerequisites = setOfNotNull(
            homePrerequisiteStatus(currentTrack, databaseInfo, databaseLoading),
            findMusicPrerequisiteStatus(context, databaseInfo, databaseLoading),
        )
        if (statusMessage in prerequisiteMessages &&
            statusMessage !in currentPrerequisites
        ) {
            statusMessage = ""
        }
    }

    pendingHistoryReplay?.let { session ->
        QueuePlacementDialog(
            title = "Queue this session again",
            trackCount = session.tracks.count { it.status == QueueStatus.QUEUED },
            onDismiss = { pendingHistoryReplay = null },
            onChoose = { placement ->
                pendingHistoryReplay = null
                viewModel.requeueSession(session, placement)
            },
        )
    }

    val drawerState = rememberDrawerState(DrawerValue.Closed)

    LaunchedEffect(Unit) {
        kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.IO) {
            PowerampReceiver.refreshCurrentTrackAfterActivityResume(context.applicationContext)
            StartRadioWidgetReceiver.updateAllWidgets(context.applicationContext)
        }
    }

    LaunchedEffect(Unit) {
        onRegisterResumeCallback?.invoke {
            viewModel.onActivityResumed()
            scope.launch {
                kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.IO) {
                    PowerampReceiver.refreshCurrentTrackAfterActivityResume(
                        context.applicationContext,
                    )
                    StartRadioWidgetReceiver.updateAllWidgets(context.applicationContext)
                }
            }
            viewModel.checkPermission()
            viewModel.refreshDatabaseInfoIfGenerationChanged()
        }
        // onResume can precede callback registration after activity recreation.
        val lifecycle = (context as? ComponentActivity)?.lifecycle
        if (lifecycle?.currentState?.isAtLeast(Lifecycle.State.RESUMED) == true) {
            viewModel.onActivityResumed()
        }
    }

    LaunchedEffect(showSettings) {
        if (showSettings) {
            viewModel.refreshSettingsStatus()
        }
    }

    DisposableEffect(Unit) {
        val listener: (PowerampTrack?) -> Unit = { track ->
            scope.launch { currentTrack = track }
        }
        PowerampReceiver.addTrackChangeListener(listener)
        onDispose { PowerampReceiver.removeTrackChangeListener(listener) }
    }

    val importLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument()
    ) { uri: Uri? ->
        uri?.let { viewModel.importDatabase(it) }
    }

    val serverMergeLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument()
    ) { uri: Uri? ->
        uri?.let { viewModel.mergeServerDatabase(it) }
    }


    BackHandler(enabled = showSettings || showTextSearch || viewingSession != null) {
        if (showTextSearch) {
            showTextSearch = false
            viewModel.clearFindMusicResults()
            viewModel.clearSongSeeds()
        } else if (showSettings) showSettings = false
        else {
            viewingSession = null
            viewModel.resetRadioState()
        }
    }

    ModalNavigationDrawer(
        drawerState = drawerState,
        gesturesEnabled = sessionHistory.isNotEmpty() && !showTextSearch,
        drawerContent = {
            ModalDrawerSheet(
                modifier = Modifier.width(280.dp),
                windowInsets = WindowInsets.safeDrawing,
            ) {
                SessionHistoryDrawer(
                    sessions = sessionHistory,
                    onSessionTap = { index ->
                        viewingSession = index
                        scope.launch { drawerState.close() }
                    },
                    onClear = {
                        viewModel.clearSessionHistory()
                        viewingSession = null
                        scope.launch { drawerState.close() }
                    }
                )
            }
        }
    ) {
        // Screen: 0=home, 1=settings, 2=text search
        val screenIndex = when {
            showTextSearch -> 2
            showSettings -> 1
            else -> 0
        }
        AnimatedContent(
            targetState = screenIndex,
            transitionSpec = {
                slideInHorizontally { if (targetState > initialState) it else -it } togetherWith
                    slideOutHorizontally { if (targetState > initialState) -it else it }
            },
            label = "screen_transition"
        ) { screen ->
            when (screen) {
                1 -> SettingsScreen(viewModel = viewModel, databaseInfo = databaseInfo,
                    onImportDatabase = { importLauncher.launch(arrayOf("application/octet-stream", "*/*")) },
                    onMergeServerDatabase = {
                        if (viewModel.canSelectServerMergeFile()) {
                            serverMergeLauncher.launch(
                                arrayOf("application/octet-stream", "*/*"),
                            )
                        }
                    },
                    hasPermission = hasPermission,
                    onRequestPermission = { viewModel.requestPermission() },
                    onBack = { showSettings = false })
                2 -> TextSearchScreen(viewModel = viewModel, onBack = {
                    showTextSearch = false
                    viewModel.clearFindMusicResults()
                    viewModel.clearSongSeeds()
                })
                else -> HomeScreen(
                    radioState = radioState, currentTrack = currentTrack,
                    databaseInfo = databaseInfo, hasPermission = hasPermission,
                    databaseLoading = databaseLoading,
                    databaseVerificationStatus = databaseVerificationStatus,
                    permissionLoading = permissionLoading,
                    sessionHistory = sessionHistory, statusMessage = statusMessage,
                    sessionReplayEligibility = sessionReplayEligibility,
                    onStartRadio = {
                        val prerequisite = homePrerequisiteStatus(
                            currentTrack = currentTrack,
                            databaseInfo = databaseInfo,
                            databaseLoading = databaseLoading,
                        )
                        if (prerequisite == null) {
                            statusMessage = ""
                            viewModel.startRadio()
                        } else {
                            statusMessage = prerequisite
                        }
                    },
                    onCancelSearch = { viewModel.cancelSearch() },
                    onClearAndReset = { viewModel.resetRadioState() },
                    onRequestPermission = { viewModel.requestPermission() },
                    onOpenSettings = { showSettings = true },
                    onOpenTextSearch = {
                        val prerequisite = findMusicPrerequisiteStatus(
                            context = context,
                            databaseInfo = databaseInfo,
                            databaseLoading = databaseLoading,
                        )
                        if (prerequisite == null) {
                            statusMessage = ""
                            viewModel.clearFindMusicResults()
                            viewModel.clearSongSeeds()
                            showTextSearch = true
                        } else {
                            statusMessage = prerequisite
                        }
                    },
                    onOpenDrawer = { scope.launch { drawerState.open() } },
                    viewingSession = viewingSession,
                    onViewSession = { viewingSession = it },
                    onRequeueSession = { pendingHistoryReplay = it },
                )
            }
        }
    }
}

@Composable
private fun QueuePlacementDialog(
    title: String,
    trackCount: Int,
    onDismiss: () -> Unit,
    onChoose: (DirectQueuePlacement) -> Unit,
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(title) },
        text = {
            val trackLabel = if (trackCount == 1) "track" else "tracks"
            Text("$trackCount $trackLabel in the displayed order")
        },
        confirmButton = {
            Column(
                modifier = Modifier.fillMaxWidth(),
                verticalArrangement = Arrangement.spacedBy(4.dp),
                horizontalAlignment = Alignment.End,
            ) {
                TextButton(
                    onClick = { onChoose(DirectQueuePlacement.REPLACE_UPCOMING) },
                ) {
                    Text("Replace upcoming")
                }
                TextButton(
                    onClick = { onChoose(DirectQueuePlacement.APPEND) },
                ) {
                    Text("Append after upcoming")
                }
                TextButton(onClick = onDismiss) {
                    Text("Cancel")
                }
            }
        },
    )
}

// ---- Home Screen ----

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun HomeScreen(
    radioState: RadioUiState,
    currentTrack: PowerampTrack?,
    databaseInfo: DatabaseInfo?,
    databaseLoading: Boolean,
    databaseVerificationStatus: String?,
    permissionLoading: Boolean,
    hasPermission: Boolean,
    sessionHistory: List<RadioResult>,
    sessionReplayEligibility: Map<String, SessionReplayEligibility>,
    statusMessage: String,
    onStartRadio: () -> Unit,
    onCancelSearch: () -> Unit,
    onClearAndReset: () -> Unit,
    onRequestPermission: () -> Unit,
    onOpenSettings: () -> Unit,
    onOpenTextSearch: () -> Unit,
    onOpenDrawer: () -> Unit,
    viewingSession: Int?,
    onViewSession: (Int?) -> Unit,
    onRequeueSession: (RadioResult) -> Unit,
) {
    val showResults = radioState is RadioUiState.Success
        || radioState is RadioUiState.Streaming
        || viewingSession != null
    val canDismissDisplayedResult = viewingSession != null || radioState is RadioUiState.Success
    val displaySession = when (radioState) {
        is RadioUiState.Streaming -> if (viewingSession != null && viewingSession in sessionHistory.indices) sessionHistory[viewingSession] else radioState.result
        is RadioUiState.Success -> if (viewingSession != null && viewingSession in sessionHistory.indices) sessionHistory[viewingSession] else radioState.result
        else -> if (viewingSession != null && viewingSession in sessionHistory.indices) sessionHistory[viewingSession] else sessionHistory.lastOrNull()
    }
    val displayedSessionDetails = displaySession.takeIf { showResults }
    var showQueueDetails by rememberSaveable(displaySession?.requestId, displaySession?.timestamp) {
        mutableStateOf(false)
    }

    if (showQueueDetails && displaySession != null) {
        QueueDetailsSheet(
            session = displaySession,
            onDismiss = { showQueueDetails = false },
        )
    }

    Scaffold(
        contentWindowInsets = WindowInsets.safeDrawing,
        topBar = {
            TopAppBar(
                title = { Text("Start Radio") },
                windowInsets = WindowInsets.safeDrawing.only(
                    WindowInsetsSides.Horizontal + WindowInsetsSides.Top,
                ),
                navigationIcon = {
                    if (sessionHistory.isNotEmpty()) {
                        IconButton(onClick = onOpenDrawer) {
                            Icon(Icons.Default.Menu, contentDescription = "History")
                        }
                    }
                },
                actions = {
                    if (canDismissDisplayedResult) {
                        IconButton(onClick = {
                            onViewSession(null)
                            if (!radioState.isActiveSearch()) onClearAndReset()
                        }) {
                            Icon(Icons.Default.Clear, contentDescription = "Dismiss result")
                        }
                    }
                    IconButton(onClick = onOpenSettings) {
                        Icon(Icons.Default.Settings, contentDescription = "Settings")
                    }
                }
            )
        },
        floatingActionButton = {
            Row(verticalAlignment = Alignment.CenterVertically) {
                IconButton(
                    onClick = onOpenTextSearch,
                    enabled = !radioState.isActiveSearch(),
                ) {
                    Icon(Icons.Default.Search, contentDescription = "Find music")
                }
                VerticalDivider(
                    modifier = Modifier.height(28.dp).padding(horizontal = 4.dp),
                    color = MaterialTheme.colorScheme.outlineVariant,
                )
                if (radioState.isActiveSearch()) {
                    IconButton(onClick = onCancelSearch) {
                        Icon(
                            Icons.Default.Clear,
                            contentDescription = "Cancel search",
                            tint = MaterialTheme.colorScheme.error,
                        )
                    }
                } else {
                    IconButton(onClick = onStartRadio) {
                        Icon(
                            painterResource(R.drawable.ic_radio_waves),
                            contentDescription = "Start Radio",
                            tint = MaterialTheme.colorScheme.primary,
                        )
                    }
                }
            }
        },
        floatingActionButtonPosition = FabPosition.End,
    ) { padding ->
        Box(
            modifier = Modifier.fillMaxSize().padding(padding)
        ) {
            Column(modifier = Modifier.fillMaxSize()) {
                CompactNowPlayingHeader(
                    currentTrack = currentTrack,
                    onClick = displayedSessionDetails?.let {
                        { showQueueDetails = true }
                    },
                    modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp),
                )
                HorizontalDivider()

                val searchingState = radioState as? RadioUiState.Searching
                if (searchingState != null) {
                    Column {
                        LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                        Text(
                            text = searchingState.message,
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            modifier = Modifier.padding(horizontal = 16.dp, vertical = 4.dp)
                        )
                    }
                }

                Box(modifier = Modifier.weight(1f)) {
                    if (showResults && displaySession != null) {
                        val savedReplayEligibility = sessionReplayEligibility[
                            displaySession.replayEligibilityKey()
                        ] ?: SessionReplayEligibility.CHECKING
                        val replayEligibility = if (radioState.isActiveSearch()) {
                            SessionReplayEligibility(
                                eligible = false,
                                reason = "Wait for the current radio operation to finish.",
                            )
                        } else {
                            savedReplayEligibility
                        }
                        SessionPage(
                            session = displaySession,
                            onRequeue = { onRequeueSession(displaySession) },
                            onShowDetails = { showQueueDetails = true },
                            replayEligibility = replayEligibility,
                            modifier = Modifier.fillMaxSize(),
                        )

                        val errorOnResults = radioState as? RadioUiState.Error
                        if (errorOnResults != null) {
                            Box(
                                modifier = Modifier.fillMaxSize()
                                    .background(MaterialTheme.colorScheme.surface.copy(alpha = 0.7f)),
                                contentAlignment = Alignment.Center
                            ) {
                                Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.errorContainer)) {
                                    Text(text = errorOnResults.message, modifier = Modifier.padding(16.dp),
                                        color = MaterialTheme.colorScheme.onErrorContainer)
                                }
                            }
                        }
                    } else {
                        when (val state = radioState) {
                            is RadioUiState.Idle, is RadioUiState.Searching -> {
                                IdleContent(hasPermission = hasPermission, databaseInfo = databaseInfo,
                                    databaseLoading = databaseLoading,
                                    databaseVerificationStatus = databaseVerificationStatus,
                                    permissionLoading = permissionLoading,
                                    statusMessage = statusMessage,
                                    isIdle = radioState is RadioUiState.Idle,
                                    onRequestPermission = onRequestPermission, modifier = Modifier.fillMaxSize())
                            }
                            is RadioUiState.Error -> {
                                Box(modifier = Modifier.fillMaxSize().padding(16.dp), contentAlignment = Alignment.Center) {
                                    Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.errorContainer)) {
                                        Text(text = state.message, modifier = Modifier.padding(16.dp),
                                            color = MaterialTheme.colorScheme.onErrorContainer)
                                    }
                                }
                            }
                            is RadioUiState.Loading, is RadioUiState.Success, is RadioUiState.Streaming -> {}
                        }
                    }
                }
            }

            val loadingState = radioState as? RadioUiState.Loading
            if (loadingState != null) {
                Box(
                    modifier = Modifier.fillMaxSize()
                        .background(MaterialTheme.colorScheme.surface.copy(alpha = 0.85f)),
                    contentAlignment = Alignment.Center
                ) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        CircularProgressIndicator()
                        Spacer(modifier = Modifier.height(12.dp))
                        Text(text = loadingState.message, style = MaterialTheme.typography.bodyMedium)
                    }
                }
            }
        }
    }
}

// ---- Compact Headers ----

@Composable
fun CompactNowPlayingHeader(
    currentTrack: PowerampTrack?,
    onClick: (() -> Unit)? = null,
    modifier: Modifier = Modifier
) {
    val clickModifier = if (onClick != null) {
        modifier.clickable(onClick = onClick)
    } else modifier

    Row(
        modifier = clickModifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Column(modifier = Modifier.weight(1f)) {
            Text(
                "Now Playing",
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.primary,
            )
            if (currentTrack != null) {
                Text(
                    text = currentTrack.title,
                    style = MaterialTheme.typography.titleSmall,
                    fontWeight = FontWeight.Bold,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                    color = MaterialTheme.colorScheme.primary,
                )
                Text(
                    text = listOfNotNull(currentTrack.artist, currentTrack.album)
                        .joinToString(" \u00b7 "),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.primary.copy(alpha = 0.7f),
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )
            } else {
                Text(
                    "No track in Poweramp",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
        if (onClick != null) {
            Icon(
                Icons.Default.Info,
                contentDescription = "Queue details",
                tint = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(start = 12.dp).size(20.dp),
            )
        }
    }
}

private data class QueueDetailFact(
    val label: String,
    val value: String,
)

private data class RankSpread(
    val typical: Int,
    val closest: Int,
    val farthest: Int,
    val domainCount: Int,
)

private enum class RankSpreadSemantics {
    SEED_NEAREST,
    TEXT_MATCH,
    RECORDING_MATCH,
    ALL_OF_OBJECTIVE,
    REFINE_OBJECTIVE,
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun QueueDetailsSheet(
    session: RadioResult,
    onDismiss: () -> Unit,
) {
    val requestFacts = remember(session) { queueRequestFacts(session) }
    val seedSpread = remember(session) { seedRankSpread(session) }
    val textSpread = remember(session) { textRankSpread(session) }

    ModalBottomSheet(
        onDismissRequest = onDismiss,
        dragHandle = { BottomSheetDefaults.DragHandle() },
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .verticalScroll(rememberScrollState())
                .padding(start = 24.dp, end = 24.dp, bottom = 32.dp),
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        if (session.findMusicSessionEvidence != null) {
                            "Find Music details"
                        } else {
                            "Radio details"
                        },
                        style = MaterialTheme.typography.titleLarge,
                    )
                    Text(
                        SessionEvidenceText.sessionOutcomeSummary(session),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
                IconButton(onClick = onDismiss) {
                    Icon(Icons.Default.Close, contentDescription = "Close details")
                }
            }

            Spacer(Modifier.height(16.dp))
            QueueDetailSectionTitle("Request")
            requestFacts.forEach { fact ->
                QueueDetailRow(fact)
            }

            if (seedSpread != null || textSpread != null) {
                val semantics = if (seedSpread != null) {
                    RankSpreadSemantics.SEED_NEAREST
                } else {
                    findMusicRankSpreadSemantics(session)
                }
                HorizontalDivider(modifier = Modifier.padding(vertical = 12.dp))
                QueueDetailSectionTitle(rankSpreadSectionTitle(semantics))
                RankSpreadRows(seedSpread ?: checkNotNull(textSpread), semantics)
            }
        }
    }
}

@Composable
private fun QueueDetailSectionTitle(text: String) {
    Text(
        text = text,
        style = MaterialTheme.typography.titleSmall,
        color = MaterialTheme.colorScheme.primary,
        modifier = Modifier.padding(bottom = 4.dp),
    )
}

@Composable
private fun QueueDetailRow(fact: QueueDetailFact) {
    Row(
        modifier = Modifier.fillMaxWidth().padding(vertical = 5.dp),
        verticalAlignment = Alignment.Top,
    ) {
        Text(
            text = fact.label,
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.width(116.dp),
        )
        Text(
            text = fact.value,
            style = MaterialTheme.typography.bodySmall,
            modifier = Modifier.weight(1f),
        )
    }
}

private fun findMusicRankSpreadSemantics(session: RadioResult): RankSpreadSemantics {
    val query = checkNotNull(session.findMusicSessionEvidence).querySpec
    return when {
        query.isSimplePositiveTextOnly -> RankSpreadSemantics.TEXT_MATCH
        query.activeIngredientCount == 1 -> RankSpreadSemantics.RECORDING_MATCH
        query.operator == FindMusicOperator.ALL_OF -> RankSpreadSemantics.ALL_OF_OBJECTIVE
        else -> RankSpreadSemantics.REFINE_OBJECTIVE
    }
}

private fun rankSpreadSectionTitle(semantics: RankSpreadSemantics): String = when (semantics) {
    RankSpreadSemantics.SEED_NEAREST -> "Distance from seed"
    RankSpreadSemantics.TEXT_MATCH -> "Text match"
    RankSpreadSemantics.RECORDING_MATCH -> "Recording match"
    RankSpreadSemantics.ALL_OF_OBJECTIVE -> "Overall All-of rank"
    RankSpreadSemantics.REFINE_OBJECTIVE -> "Refiner rank"
}

private fun rankSpreadTypical(spread: RankSpread, semantics: RankSpreadSemantics): String {
    val rank = formatRank(spread.typical)
    return when (semantics) {
        RankSpreadSemantics.SEED_NEAREST -> "$rank nearest (median)"
        RankSpreadSemantics.TEXT_MATCH -> "$rank by text match (median)"
        RankSpreadSemantics.RECORDING_MATCH -> "$rank by recording match (median)"
        RankSpreadSemantics.ALL_OF_OBJECTIVE -> "Overall $rank (median)"
        RankSpreadSemantics.REFINE_OBJECTIVE -> "Refiner $rank (median)"
    }
}

private fun rankSpreadRange(spread: RankSpread, semantics: RankSpreadSemantics): String {
    val closest = formatRank(spread.closest)
    val farthest = formatRank(spread.farthest)
    return when (semantics) {
        RankSpreadSemantics.SEED_NEAREST -> "$closest to $farthest nearest"
        RankSpreadSemantics.TEXT_MATCH -> "$closest to $farthest by text match"
        RankSpreadSemantics.RECORDING_MATCH -> "$closest to $farthest by recording match"
        RankSpreadSemantics.ALL_OF_OBJECTIVE -> "Overall $closest to $farthest"
        RankSpreadSemantics.REFINE_OBJECTIVE -> "Refiner $closest to $farthest"
    }
}

private fun rankSpreadDomain(spread: RankSpread, semantics: RankSpreadSemantics): String {
    val domain = when (semantics) {
        RankSpreadSemantics.SEED_NEAREST ->
            "${formatCount(spread.domainCount)} other active indexed recordings"
        RankSpreadSemantics.TEXT_MATCH,
        RankSpreadSemantics.RECORDING_MATCH,
        RankSpreadSemantics.ALL_OF_OBJECTIVE,
        -> "${formatCount(spread.domainCount)} eligible recordings"
        RankSpreadSemantics.REFINE_OBJECTIVE ->
            "${formatCount(spread.domainCount)} recordings in the primary neighborhood"
    }
    val lastResultFraction = LibraryRankEvidenceText.topFraction(
        spread.farthest,
        spread.domainCount,
    )
    return buildString {
        append(domain)
        lastResultFraction?.let { append(" \u00b7 lowest result in ").append(it) }
    }
}

@Composable
private fun RankSpreadRows(spread: RankSpread, semantics: RankSpreadSemantics) {
    QueueDetailRow(
        QueueDetailFact(
            "Typical",
            rankSpreadTypical(spread, semantics),
        ),
    )
    QueueDetailRow(
        QueueDetailFact(
            "Range",
            rankSpreadRange(spread, semantics),
        ),
    )
    QueueDetailRow(
        QueueDetailFact(
            "Rank scope",
            rankSpreadDomain(spread, semantics),
        ),
    )
}

private fun queueRequestFacts(session: RadioResult): List<QueueDetailFact> = buildList {
    val config = session.config
    val findMusic = session.findMusicSessionEvidence
    if (findMusic != null) {
        val query = findMusic.querySpec
        add(QueueDetailFact("Query", query.displayLabel))
        add(QueueDetailFact("Selection", SessionEvidenceText.findMusicMode(findMusic)))
        add(
            QueueDetailFact(
                "Result order",
                when {
                    query.activeIngredientCount == 1 && !query.isSimplePositiveTextOnly ->
                        "Cosine similarity to the selected recording, highest first"
                    !query.isSimplePositiveTextOnly ->
                        if (query.operator == FindMusicOperator.ALL_OF) {
                            if (
                                query.textResultPlanner ==
                                FindMusicTextResultPlanner.VARIED_ALL_OF_DPP
                            ) {
                                "DPP-selected set from the complete weighted All-of ranking"
                            } else {
                                "Weighted geometric mean across ingredient percentiles"
                            }
                        } else {
                            "Refiner rank within the declared primary neighborhood"
                        }
                    query.textResultPlanner == FindMusicTextResultPlanner.CLOSEST ->
                        "Text cosine similarity, highest first"
                    else -> "DPP using text match as quality"
                },
            ),
        )
        if (query.operator == FindMusicOperator.REFINE) {
            addEligibleDomain(
                count = findMusic.ingredientRankingDomainCount,
                days = query.effectiveLibraryAddedDays,
            )
            val refine = query.refineSpec
            val primaryNeighborhoodCount = findMusic.objectiveRankingDomainCount
            if (refine != null && primaryNeighborhoodCount != null) {
                val primary = query.activeEvidenceLabels
                    .getOrNull(refine.primaryIngredientIndex)
                    ?.takeIf(String::isNotBlank)
                add(
                    QueueDetailFact(
                        "Primary neighborhood",
                        buildString {
                            append("Nearest ")
                                .append(refineNeighborhoodLabel(refine.neighborhood))
                            primary?.let { append(" to ").append(it) }
                            append(" \u00b7 ")
                                .append(formatCount(primaryNeighborhoodCount))
                                .append(" recordings")
                        },
                    ),
                )
            }
        } else {
            addEligibleDomain(
                count = findMusic.objectiveRankingDomainCount
                    ?: findMusic.ingredientRankingDomainCount,
                days = query.effectiveLibraryAddedDays,
            )
        }
        findMusic.stableResultReduction.collapsedEquivalentCount
            .takeIf { it > 0 }
            ?.let { skipped ->
                add(QueueDetailFact("Copies skipped", formatCount(skipped)))
            }
    } else {
        add(
            QueueDetailFact(
                "Seed",
                buildString {
                    append(SessionEvidenceText.seedTitle(session))
                    SessionEvidenceText.seedIdentity(
                        session.seedTrack.artist,
                        session.seedTrack.album,
                    )?.let { append("\n").append(it) }
                },
            ),
        )
        add(QueueDetailFact("Selection", detailedSelectionLabel(config)))
        addEligibleDomain(
            count = session.eligibleCandidateIdentityCount,
            days = config.effectiveLibraryAddedDays,
        )
        when (config.selectionMode) {
            SelectionMode.CLOSEST -> add(
                QueueDetailFact("Order", "Seed cosine similarity, highest first"),
            )
            SelectionMode.MMR -> {
                add(
                    QueueDetailFact(
                        "Balance",
                        buildString {
                            append("${(config.diversityLambda * 100).roundToInt()}% ")
                            append(if (config.driftEnabled) "current-direction" else "seed")
                            append(" relevance \u00b7 ")
                            append("${((1f - config.diversityLambda) * 100).roundToInt()}% variety")
                        },
                    ),
                )
                addSelectionPoolFact(
                    label = "Selection pool",
                    count = session.eligibleCandidateIdentityCount,
                    selected = session.eligibleCandidateIdentityCount?.let {
                        config.resolveCandidatePoolSize(it)
                    },
                    fraction = config.effectiveMmrCandidatePoolFraction,
                    suffix = if (config.driftEnabled) " at each pick" else "",
                )
                driftDetail(config)?.let { add(QueueDetailFact("Direction", it)) }
            }
            SelectionMode.DPP -> {
                add(
                    QueueDetailFact(
                        "Seed pull",
                        SelectionControlText.dppSeedPullLabel(
                            config.effectiveDppQualityExponent,
                        ),
                    ),
                )
                if (config.dppUsesCertifiedFullDomain) {
                    addSelectionPoolFact(
                        label = "Selection pool",
                        count = session.eligibleCandidateIdentityCount,
                        selected = session.eligibleCandidateIdentityCount,
                        fraction = 1f,
                    )
                } else {
                    addSelectionPoolFact(
                        label = "Selection pool",
                        count = session.eligibleCandidateIdentityCount,
                        selected = session.eligibleCandidateIdentityCount?.let {
                            config.resolveCandidatePoolSize(it)
                        },
                        fraction = config.effectiveDppFixedCandidatePoolFraction,
                    )
                }
            }
            SelectionMode.RANDOM_WALK -> {
                session.graphExploration?.let { graph ->
                    add(
                        QueueDetailFact(
                            "Typical route",
                            "About ${graph.expectedRouteLinks.roundToInt()} track-to-track moves",
                        ),
                    )
                }
                add(
                    QueueDetailFact(
                        "Stop chance",
                        "${formatKnob(config.walkRestartAlpha * 100f)}% after each move",
                    ),
                )
            }
            SelectionMode.UNIFORM_SHUFFLE ->
                add(QueueDetailFact("Order", "Deterministic shuffle"))
        }
        if (config.artistLimitsEnabled) {
            add(
                QueueDetailFact(
                    "Artist limits",
                    "Max ${config.maxPerArtist} per credit \u00b7 " +
                        if (config.minArtistSpacing == 0) {
                            "no spacing"
                        } else {
                            "${config.minArtistSpacing}-track spacing"
                        },
                ),
            )
        }
    }
    add(
        QueueDetailFact(
            "Started",
            SessionEvidenceText.historyTimestamp(session.timestamp),
        ),
    )
}

private fun MutableList<QueueDetailFact>.addEligibleDomain(
    count: Int?,
    days: Int?,
) {
    add(
        QueueDetailFact(
            "Eligible domain",
            listOfNotNull(
                count?.let { "${formatCount(it)} recordings" },
                libraryAddedDaysLabel(days),
            ).joinToString(" \u00b7 "),
        ),
    )
}

private fun MutableList<QueueDetailFact>.addSelectionPoolFact(
    label: String,
    count: Int?,
    selected: Int?,
    fraction: Float,
    suffix: String = "",
) {
    val value = when {
        count != null && selected != null && selected >= count ->
            "All ${formatCount(count)} eligible recordings"
        count != null && selected != null ->
            "Nearest ${formatCount(selected)} of ${formatCount(count)} eligible recordings " +
                "(${formatKnob(fraction * 100f)}%)"
        selected != null -> "Nearest ${formatCount(selected)} eligible recordings"
        fraction >= 1f -> "All eligible recordings"
        else -> "Nearest ${formatKnob(fraction * 100f)}% of eligible recordings"
    }
    add(QueueDetailFact(label, value + suffix))
}

private fun detailedSelectionLabel(config: RadioConfig): String = when (config.selectionMode) {
    SelectionMode.CLOSEST -> "Closest"
    SelectionMode.MMR -> if (config.driftEnabled) "MMR + drift" else "MMR"
    SelectionMode.DPP -> "DPP"
    SelectionMode.RANDOM_WALK -> "Graph Explorer"
    SelectionMode.UNIFORM_SHUFFLE -> "Uniform shuffle"
}

private fun driftDetail(config: RadioConfig): String? {
    if (!config.driftEnabled || config.selectionMode != SelectionMode.MMR) return null
    return when (config.driftMode) {
        DriftMode.MOMENTUM ->
            "${formatKnob(config.momentumBeta * 100f)}% prior-direction memory"
        DriftMode.SEED_INTERPOLATION -> when {
            !DriftControlPolicy.seedFadeApplies(config.anchorStrength) -> "Follow previous pick"
            config.anchorDecay == DecaySchedule.NONE ->
                "${formatKnob(config.anchorStrength * 100f)}% seed pull, held"
            config.anchorDecay == DecaySchedule.LINEAR ->
                "${formatKnob(config.anchorStrength * 100f)}% seed pull \u00b7 linear fade, " +
                    "half-strength at ${formatKnob(config.effectiveAnchorHalfLifeTracks)} picks"
            config.anchorDecay == DecaySchedule.EXPONENTIAL ->
                "${formatKnob(config.anchorStrength * 100f)}% seed pull \u00b7 " +
                    "${formatKnob(config.effectiveAnchorHalfLifeTracks)}-pick half-life"
            else ->
                "${formatKnob(config.anchorStrength * 100f)}% seed pull \u00b7 drop after " +
                    "${DriftControlPolicy.stepDropAfterPickCount(config.effectiveAnchorHalfLifeTracks)} picks"
        }
    }
}

private fun seedRankSpread(session: RadioResult): RankSpread? {
    if (session.isDirectQueue) return null
    val queued = session.tracks.filter { it.status == QueueStatus.QUEUED }
    if (queued.isEmpty()) return null
    val evidence = queued.mapNotNull { SeedDistanceEvidencePolicy.evidenceOrNull(session, it) }
    if (evidence.size != queued.size) return null
    val domain = evidence.map { it.rankingIdentityCount }.distinct().singleOrNull()
        ?: return null
    return rankSpread(evidence.map { it.seedRank }, domain)
}

private fun textRankSpread(session: RadioResult): RankSpread? {
    val evidence = session.findMusicSessionEvidence ?: return null
    val domain = evidence.objectiveRankingDomainCount ?: return null
    val queued = session.tracks.filter { it.status == QueueStatus.QUEUED }
    val ranks = queued.mapNotNull { it.findMusicEvidence?.objectiveRank }
    if (ranks.size != queued.size) return null
    return rankSpread(ranks, domain)
}

private fun rankSpread(ranks: List<Int>, domain: Int): RankSpread? {
    if (domain <= 0 || ranks.isEmpty() || ranks.any { it !in 1..domain }) return null
    val sorted = ranks.sorted()
    val middle = sorted.size / 2
    val typical = if (sorted.size % 2 == 1) {
        sorted[middle]
    } else {
        ((sorted[middle - 1].toLong() + sorted[middle].toLong()) / 2.0).roundToInt()
    }
    return RankSpread(
        typical = typical,
        closest = sorted.first(),
        farthest = sorted.last(),
        domainCount = domain,
    )
}

private fun compactSessionHeading(session: RadioResult): String =
    if (session.findMusicSessionEvidence != null) "Find Music" else "Radio from"

private fun compactSessionTitle(session: RadioResult): String =
    session.findMusicSessionEvidence?.querySpec?.displayLabel
        ?: SessionEvidenceText.seedTitle(session)

private fun compactSessionContext(session: RadioResult, outcomeSignal: String): String {
    val findMusic = session.findMusicSessionEvidence
    val parts = buildList {
        if (findMusic == null) {
            session.seedTrack.artist?.trim()?.takeIf(String::isNotBlank)?.let(::add)
        }
        add(
            findMusic?.let(SessionEvidenceText::findMusicMode)
                ?: detailedSelectionLabel(session.config),
        )
        val days = if (findMusic != null) {
            findMusic.querySpec.effectiveLibraryAddedDays
        } else {
            session.config.effectiveLibraryAddedDays
        }
        days?.let {
            add(libraryAddedDaysLabel(it))
        }
        add(outcomeSignal)
    }
    return parts.joinToString(" \u00b7 ")
}

private fun formatRank(rank: Int): String = "#${formatCount(rank)}"

private fun formatCount(value: Int): String =
    String.format(Locale.US, "%,d", value)

// ---- Session Page ----

@Composable
fun SessionPage(
    session: RadioResult,
    onRequeue: () -> Unit,
    onShowDetails: () -> Unit,
    replayEligibility: SessionReplayEligibility,
    modifier: Modifier = Modifier,
) {
    val listState = rememberLazyListState()
    var followingStreamingTail by remember(session.requestId, session.timestamp) {
        mutableStateOf(false)
    }
    LaunchedEffect(session.tracks.size, session.isComplete) {
        if (!session.isComplete && session.tracks.isNotEmpty()) {
            val lastVisible = listState.layoutInfo.visibleItemsInfo.lastOrNull()?.index ?: 0
            val totalItems = listState.layoutInfo.totalItemsCount
            val isAtBottom = totalItems == 0 || lastVisible >= totalItems - 2
            followingStreamingTail = isAtBottom
            if (isAtBottom) listState.animateScrollToItem(session.tracks.lastIndex)
        } else if (session.isComplete && followingStreamingTail) {
            listState.scrollToItem(0)
            followingStreamingTail = false
        }
    }

    Column(modifier = modifier) {
        LazyColumn(
            state = listState,
            modifier = Modifier.fillMaxSize(),
            contentPadding = PaddingValues(
                start = 16.dp,
                top = 4.dp,
                end = 16.dp,
                bottom = 8.dp,
            ),
        ) {
            item {
                SessionSeedHeader(
                    session = session,
                    onRequeue = onRequeue,
                    onShowDetails = onShowDetails,
                    replayEligibility = replayEligibility,
                )
            }
            items(session.tracks.size) { index ->
                TrackResultRow(
                    trackResult = session.tracks[index],
                    session = session,
                )
            }
            if (!session.isComplete) {
                item { StreamingProgressItem(selected = session.tracks.size, total = session.totalExpected) }
            }
        }
    }
}

@Composable
private fun SessionSeedHeader(
    session: RadioResult,
    onRequeue: () -> Unit,
    onShowDetails: () -> Unit,
    replayEligibility: SessionReplayEligibility,
    modifier: Modifier = Modifier,
) {
    val outcomeSignal = SessionEvidenceText.compactSessionOutcomeSummary(session)
    Column(
        modifier = modifier
            .fillMaxWidth()
            .clickable(onClick = onShowDetails),
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Column(modifier = Modifier.padding(vertical = 6.dp).weight(1f)) {
                Text(
                    text = compactSessionHeading(session),
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.primary,
                )
                Text(
                    text = compactSessionTitle(session),
                    style = MaterialTheme.typography.titleSmall,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )
                Text(
                    text = compactSessionContext(session, outcomeSignal),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )
            }
            IconButton(
                onClick = onRequeue,
                enabled = replayEligibility.eligible,
                modifier = Modifier.size(48.dp),
            ) {
                Icon(
                    Icons.Default.Refresh,
                    contentDescription = "Requeue this session",
                    modifier = Modifier.size(20.dp),
                )
            }
        }
        replayEligibility.reason?.let { reason ->
            Text(
                text = reason,
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(bottom = 4.dp),
            )
        }
    }
}

@Composable
private fun StreamingProgressItem(selected: Int, total: Int) {
    Row(modifier = Modifier.fillMaxWidth().padding(vertical = 12.dp),
        horizontalArrangement = Arrangement.Center,
        verticalAlignment = Alignment.CenterVertically) {
        CircularProgressIndicator(modifier = Modifier.size(16.dp), strokeWidth = 2.dp)
        Spacer(modifier = Modifier.width(8.dp))
        Text("$selected of $total selected; queue confirmation pending", style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

// ---- Track Result Row ----

@Composable
fun TrackResultRow(
    trackResult: QueuedTrackResult,
    session: RadioResult? = null,
) {
    var expanded by remember { mutableStateOf(false) }
    val seedDistanceEvidence = SeedDistanceEvidencePolicy.evidenceOrNull(session, trackResult)
    val isFailed = trackResult.status == QueueStatus.NOT_IN_LIBRARY ||
        trackResult.status == QueueStatus.QUEUE_FAILED

    Column(modifier = Modifier.fillMaxWidth().clickable { expanded = !expanded }) {
        Row(modifier = Modifier.fillMaxWidth().alpha(if (isFailed) 0.45f else 1f),
            verticalAlignment = Alignment.CenterVertically) {
            Column(modifier = Modifier.weight(1f).padding(vertical = 2.dp, horizontal = 4.dp)) {
                Text(text = trackResult.track.title ?: "Unknown", style = MaterialTheme.typography.bodyMedium,
                    maxLines = 1, overflow = TextOverflow.Ellipsis)
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        text = trackResult.track.artist ?: "Unknown",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                        modifier = Modifier.weight(1f),
                    )
                    seedDistanceEvidence?.let { evidence ->
                        LibraryRankEvidenceText.compactNearestRank(
                            evidence.seedRank,
                            evidence.rankingIdentityCount,
                        )?.let { rankText ->
                            Spacer(modifier = Modifier.width(8.dp))
                            Text(
                                text = rankText,
                                style = MaterialTheme.typography.labelSmall,
                                fontFamily = FontFamily.Monospace,
                                color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.75f),
                                textAlign = TextAlign.End,
                                maxLines = 1,
                            )
                        }
                    }
                }
            }
        }

        AnimatedVisibility(visible = expanded) {
            TrackExplanation(
                trackResult = trackResult,
                session = session,
                modifier = Modifier.padding(start = 4.dp, top = 2.dp, bottom = 4.dp)
            )
        }
    }
}

@Composable
private fun TrackExplanation(
    trackResult: QueuedTrackResult,
    session: RadioResult? = null,
    modifier: Modifier = Modifier,
) {
    Column(modifier = modifier) {
        val album = trackResult.track.album
        val dur = trackResult.track.durationMs
        val durStr = "${dur / 60000}:${((dur % 60000) / 1000).toString().padStart(2, '0')}"
        val metaLine = listOfNotNull(
            album?.takeIf { it.isNotBlank() },
            durStr.takeIf { dur > 0 },
        ).joinToString(" \u00b7 ")
        if (metaLine.isNotEmpty()) {
            Text(metaLine, style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f))
        }

        if (trackResult.status != QueueStatus.QUEUED) {
            val statusText = when (trackResult.status) {
                QueueStatus.PENDING -> "Waiting for Poweramp confirmation"
                QueueStatus.NOT_IN_LIBRARY -> "Not found in Poweramp library"
                QueueStatus.QUEUE_FAILED -> if (session?.delivery?.verificationComplete == false) {
                    "Final Poweramp queue check did not finish"
                } else {
                    "Poweramp did not confirm this queue entry"
                }
                QueueStatus.QUEUED -> ""
            }
            val statusColor = if (trackResult.status == QueueStatus.PENDING) {
                MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.8f)
            } else {
                MaterialTheme.colorScheme.error.copy(alpha = 0.8f)
            }
            Text(statusText,
                style = MaterialTheme.typography.bodySmall,
                color = statusColor)
        }

        val seedDistanceEvidence = SeedDistanceEvidencePolicy.evidenceOrNull(
            session,
            trackResult,
        )
        if (
            session != null &&
            (trackResult.findMusicEvidence != null || trackResult.composedEvidence != null ||
                seedDistanceEvidence != null || trackResult.mmrEvidence != null ||
                trackResult.graphTerminalProbability != null)
        ) {
            val subtleColor = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f)
            val isGraphExplorer = session.config.selectionMode == SelectionMode.RANDOM_WALK
            val isUniformShuffle = session.config.selectionMode == SelectionMode.UNIFORM_SHUFFLE
            val isDrift = session.config.driftEnabled &&
                session.config.selectionMode == SelectionMode.MMR

            val driftRank = trackResult.driftRank
            fun seedRankEvidence(rank: Int): String? =
                LibraryRankEvidenceText.rankWithTopFraction(
                    rank,
                    checkNotNull(seedDistanceEvidence).rankingIdentityCount,
                )
            val seedRank = seedDistanceEvidence?.seedRank

            val text = when {
                session.findMusicSessionEvidence != null &&
                    trackResult.findMusicEvidence != null ->
                    SessionEvidenceText.findMusicTrack(
                        session.findMusicSessionEvidence,
                        trackResult.findMusicEvidence,
                    )
                session.composedQuerySpec != null && trackResult.composedEvidence != null -> {
                    val evidence = trackResult.composedEvidence
                    val total = session.composedObjectiveRankingDomainCount
                    if (total != null && evidence.objectiveRank in 1..total) {
                        LibraryRankEvidenceText.rankWithTopFraction(
                            evidence.objectiveRank,
                            total,
                        )?.let { "Overall All-of match \u00b7 $it" }
                    } else {
                        null
                    }
                }
                isGraphExplorer && trackResult.graphTerminalProbability != null &&
                    trackResult.candidateRank != null &&
                    session.graphExploration?.rankedCandidateCount?.let {
                        trackResult.candidateRank in 1..it
                    } == true -> LibraryRankEvidenceText.rankWithTopFraction(
                            trackResult.candidateRank,
                            checkNotNull(session.graphExploration).rankedCandidateCount,
                        )?.let { graphStanding ->
                        buildString {
                            append("Graph reach among reachable tracks \u00b7 ").append(graphStanding)
                            seedRank?.let { rank ->
                                seedRankEvidence(rank)?.let {
                                    append("\nFrom original seed \u00b7 ").append(it)
                                }
                            }
                        }
                    }
                isDrift && driftRank != null && seedRank != null -> {
                    listOfNotNull(
                        seedRankEvidence(seedRank)?.let { "From original seed \u00b7 $it" },
                        seedRankEvidence(driftRank)?.let {
                            "From current queue direction \u00b7 $it"
                        },
                    ).joinToString("\n").takeIf(String::isNotBlank)
                }
                isGraphExplorer && seedRank != null -> {
                    seedRankEvidence(seedRank)?.let { "From original seed \u00b7 $it" }
                }
                isDrift && seedRank != null ->
                    seedRankEvidence(seedRank)?.let { "From original seed \u00b7 $it" }
                isUniformShuffle && trackResult.candidateRank != null &&
                    session.uniformShuffleIdentity?.rankedCandidateCount?.let {
                        trackResult.candidateRank in 1..it
                    } == true -> LibraryRankEvidenceText.rank(
                            trackResult.candidateRank,
                            checkNotNull(session.uniformShuffleIdentity).rankedCandidateCount,
                        )?.let { shufflePosition ->
                        buildString {
                            append("Shuffle position ").append(shufflePosition)
                            seedRank?.let { rank ->
                                seedRankEvidence(rank)?.let {
                                    append("\nFrom original seed \u00b7 ").append(it)
                                }
                            }
                        }
                    }
                seedRank != null ->
                    seedRankEvidence(seedRank)?.let { "From original seed \u00b7 $it" }
                else -> null
            }
            text?.let {
                Text(it, style = MaterialTheme.typography.bodySmall, color = subtleColor)
            }

            trackResult.mmrEvidence?.let { evidence ->
                val strongestPriorPick = evidence.greatestOverlapTrackId?.let { trackId ->
                    session.tracks.firstOrNull { it.track.id == trackId }?.track?.title
                }
                SessionEvidenceText.mmrPriorPick(strongestPriorPick)?.let { overlap ->
                    Text(
                        overlap,
                        style = MaterialTheme.typography.bodySmall,
                        color = subtleColor,
                    )
                }
            }

        }
    }
}

// ---- Session History Drawer ----

@Composable
fun SessionHistoryDrawer(sessions: List<RadioResult>, onSessionTap: (Int) -> Unit, onClear: () -> Unit) {
    var confirmClear by rememberSaveable { mutableStateOf(false) }

    if (confirmClear) {
        AlertDialog(
            onDismissRequest = { confirmClear = false },
            title = { Text("Clear session history?") },
            text = {
                Text("This removes the saved session list only. It does not change the current result or Poweramp queue.")
            },
            confirmButton = {
                TextButton(onClick = {
                    onClear()
                    confirmClear = false
                }) {
                    Text("Clear history", color = MaterialTheme.colorScheme.error)
                }
            },
            dismissButton = {
                TextButton(onClick = { confirmClear = false }) { Text("Cancel") }
            },
        )
    }

    Column(modifier = Modifier.fillMaxHeight()) {
        Row(modifier = Modifier.fillMaxWidth().padding(16.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically) {
            Text("Session History", style = MaterialTheme.typography.titleMedium)
            if (sessions.isNotEmpty()) {
                TextButton(onClick = { confirmClear = true }) { Text("Clear history") }
            }
        }
        HorizontalDivider()

        if (sessions.isEmpty()) {
            Box(modifier = Modifier.fillMaxWidth().padding(32.dp), contentAlignment = Alignment.Center) {
                Text("No sessions yet", style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        } else {
            LazyColumn(modifier = Modifier.weight(1f), contentPadding = PaddingValues(vertical = 4.dp)) {
                items(sessions.size) { i ->
                    val index = sessions.lastIndex - i
                    val session = sessions[index]
                    val timeStr = SessionEvidenceText.historyTimestamp(session.timestamp)
                    NavigationDrawerItem(
                        label = {
                            Column {
                                Text(SessionEvidenceText.seedTitle(session), style = MaterialTheme.typography.bodyMedium,
                                    maxLines = 1, overflow = TextOverflow.Ellipsis)
                                SessionEvidenceText.seedIdentity(
                                    session.seedTrack.artist,
                                    session.seedTrack.album,
                                )?.let { identity ->
                                    Text(
                                        identity,
                                        style = MaterialTheme.typography.bodySmall,
                                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                                        maxLines = 1,
                                        overflow = TextOverflow.Ellipsis,
                                    )
                                }
                                val mode = if (session.isDirectQueue) {
                                    if (session.delivery?.origin == QueueOrigin.HISTORY_REQUEUE) {
                                        "Replayed queue"
                                    } else session.findMusicSessionEvidence?.let { evidence ->
                                        SessionEvidenceText.findMusicMode(evidence)
                                    } ?: "Direct queue"
                                } else if (session.composedQuerySpec != null) {
                                    "All of"
                                } else {
                                    humanSelectionMode(
                                        mode = session.config.selectionMode,
                                        drift = session.config.driftEnabled,
                                    )
                                }
                                val subtitle = SessionEvidenceText.sessionDrawerSummary(
                                    session = session,
                                    modeSignal = mode,
                                )
                                Text(subtitle,
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                                    maxLines = 2, overflow = TextOverflow.Ellipsis)
                                Text(
                                    timeStr,
                                    style = MaterialTheme.typography.labelSmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                                    maxLines = 1,
                                )
                            }
                        },
                        selected = false, onClick = { onSessionTap(index) },
                        modifier = Modifier.padding(horizontal = 12.dp)
                    )
                }
            }
        }
    }
}

// ---- Idle Content ----

@Composable
fun IdleContent(
    hasPermission: Boolean, databaseInfo: DatabaseInfo?, databaseLoading: Boolean,
    databaseVerificationStatus: String?,
    permissionLoading: Boolean,
    statusMessage: String,
    isIdle: Boolean = true,
    onRequestPermission: () -> Unit,
    modifier: Modifier = Modifier
) {
    Column(modifier = modifier.fillMaxWidth().padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)) {
        if (!hasPermission && !permissionLoading) {
            Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.errorContainer)) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text("Poweramp Access Required", style = MaterialTheme.typography.titleSmall)
                    Spacer(modifier = Modifier.height(8.dp))
                    Button(onClick = onRequestPermission) { Text("Grant Access") }
                }
            }
        }
        if (databaseInfo == null && databaseLoading) {
            LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
            Text(databaseVerificationStatus ?: "Opening the saved music index and reading track counts",
                style = MaterialTheme.typography.bodyLarge,
                color = MaterialTheme.colorScheme.onSurfaceVariant)
        } else if (databaseInfo == null) {
            Text("Import a music index in Settings.",
                style = MaterialTheme.typography.bodyLarge,
                color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        if (statusMessage.isNotEmpty()) {
            Text(statusMessage, style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        if (isIdle &&
            hasPermission &&
            !permissionLoading &&
            databaseInfo != null &&
            !databaseLoading &&
            statusMessage.isEmpty()
        ) {
            val activeCount = databaseInfo.activeTrackCount ?: databaseInfo.embeddingCount
            Text(
                "${String.format(Locale.US, "%,d", activeCount)} indexed tracks \u00b7 " +
                    "Music index ready",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

// ---- Find Music Screen ----

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun TextSearchScreen(
    viewModel: MainViewModel = viewModel(),
    onBack: () -> Unit
) {
    val textSearchResult by viewModel.textSearchResult.collectAsState()
    val multiSeedResult by viewModel.multiSeedResult.collectAsState()
    val isLoading by viewModel.textSearchLoading.collectAsState()
    val isMultiSeedLoading by viewModel.multiSeedLoading.collectAsState()
    val findMusicLoadingStatus by viewModel.findMusicLoadingStatus.collectAsState()
    val recentSearches by viewModel.recentSearches.collectAsState()
    val textIngredients by viewModel.textIngredients.collectAsState()
    val songSeeds by viewModel.songSeeds.collectAsState()
    val recordingLookupState by viewModel.recordingLookupState.collectAsState()
    val displayedQueueEligibility by viewModel.displayedQueueEligibility.collectAsState()
    val findMusicOperator by viewModel.findMusicOperator.collectAsState()
    val refinePrimaryIngredientIndex by
        viewModel.findMusicRefinePrimaryIngredientIndex.collectAsState()
    val refineNeighborhood by viewModel.findMusicRefineNeighborhood.collectAsState()
    val textResultPlanner by viewModel.findMusicTextResultPlanner.collectAsState()
    val sharedQueueLength by viewModel.numTracks.collectAsState()
    var pendingQueueResult by remember { mutableStateOf<TextSearchResult?>(null) }
    var pendingRecentReplay by remember { mutableStateOf<FindMusicQuerySpec?>(null) }
    var showRecentSearches by rememberSaveable { mutableStateOf(false) }
    var recordingPickerSeedId by rememberSaveable { mutableStateOf<Long?>(null) }
    val focusRequester = remember { FocusRequester() }
    val imeVisible = WindowInsets.ime.getBottom(LocalDensity.current) > 0

    val hasSongSeeds = songSeeds.isNotEmpty()
    val anyLoading = isLoading || isMultiSeedLoading

    // Active result: prefer multi-seed if it exists, else text-only
    val activeResult = multiSeedResult ?: textSearchResult
    val editorScrollState = rememberScrollState()

    val requestRecentReplay: (FindMusicQuerySpec) -> Unit = { savedSearch ->
        showRecentSearches = false
        val currentControlsSearch = viewModel.recentSearchWithCurrentControls(savedSearch)
        if (savedSearch.hasSameExecutionControlsAs(currentControlsSearch)) {
            viewModel.replayRecentSearch(savedSearch)
        } else {
            pendingRecentReplay = savedSearch
        }
    }

    pendingQueueResult?.let { result ->
        QueuePlacementDialog(
            title = "Queue these results",
            trackCount = result.matches.size,
            onDismiss = { pendingQueueResult = null },
            onChoose = { placement ->
                pendingQueueResult = null
                if (viewModel.queueDisplayedResults(result, placement)) onBack()
            },
        )
    }

    pendingRecentReplay?.let { savedSearch ->
        val currentControlsSearch = viewModel.recentSearchWithCurrentControls(savedSearch)
        RecentFindMusicReplayDialog(
            search = savedSearch,
            currentControlsSearch = currentControlsSearch,
            onDismiss = { pendingRecentReplay = null },
            onUseCurrentControls = {
                pendingRecentReplay = null
                viewModel.replayRecentSearch(currentControlsSearch)
            },
            onUseSavedControls = {
                pendingRecentReplay = null
                viewModel.replayRecentSearch(savedSearch)
            },
        )
    }

    if (showRecentSearches) {
        RecentFindMusicSheet(
            searches = recentSearches,
            onDismiss = { showRecentSearches = false },
            onClear = {
                viewModel.clearRecentSearches()
                showRecentSearches = false
            },
            onReplay = requestRecentReplay,
        )
    }

    LaunchedEffect(recordingPickerSeedId, songSeeds) {
        val seedId = recordingPickerSeedId
        if (seedId != null && songSeeds.none { it.id == seedId }) {
            recordingPickerSeedId = null
            viewModel.dismissSongSeedSearch()
        }
    }

    recordingPickerSeedId?.let { seedId ->
        val seedIndex = songSeeds.indexOfFirst { it.id == seedId }
        if (seedIndex >= 0) {
            RecordingPickerSheet(
                query = songSeeds[seedIndex].query,
                state = recordingLookupState.forSeed(seedId),
                onDismiss = {
                    recordingPickerSeedId = null
                    viewModel.dismissSongSeedSearch()
                },
                onRetry = { viewModel.searchSongSeed(seedIndex) },
                onChoose = { track ->
                    viewModel.confirmSongSeed(seedIndex, track)
                    recordingPickerSeedId = null
                },
            )
        }
    }

    LaunchedEffect(displayedQueueEligibility.eligible) {
        if (!displayedQueueEligibility.eligible) pendingQueueResult = null
    }

    DisposableEffect(Unit) {
        onDispose { viewModel.clearFindMusicResults() }
    }

    fun doSearch() = viewModel.performFindMusicSearch()

    BackHandler(enabled = activeResult != null) {
        viewModel.clearFindMusicResults()
    }

    val dynamicTitle = if (activeResult != null) "Results" else "Find music"

    // Song name placeholders for seed fields
    val songPlaceholders = remember {
        listOf(
            "time pachanga boys",
            "bohemian rhapsody queen",
            "billie jean michael jackson",
            "africa toto",
            "stairway to heaven led zeppelin",
            "like a prayer madonna",
        ).shuffled()
    }

    val activeIngredientCount = textIngredients.count { it.isActive } +
        songSeeds.count { it.isActive }
    val effectiveFindMusicOperator = FindMusicEditorPolicy.effectiveOperator(
        requested = findMusicOperator,
        activeIngredientCount = activeIngredientCount,
    )
    val usesSharedIngredientBalance =
        effectiveFindMusicOperator == FindMusicOperator.ALL_OF &&
            FindMusicEditorPolicy.usesSharedBalance(activeIngredientCount)
    val usesPerIngredientWeightControls =
        effectiveFindMusicOperator == FindMusicOperator.ALL_OF &&
            FindMusicEditorPolicy.usesPerIngredientWeightControls(activeIngredientCount)
    val editorHasInput = textIngredients.any { it.query.isNotBlank() } ||
        songSeeds.any { it.query.isNotBlank() || it.confirmedTrack != null }
    val showInlineRecent = activeResult == null && !editorHasInput && recentSearches.isNotEmpty()
    val editorReadiness = FindMusicEditorPolicy.readiness(
        textIngredients = textIngredients,
        songSeeds = songSeeds,
        searchRunning = anyLoading,
        operator = effectiveFindMusicOperator,
        resultLimit = sharedQueueLength,
        refinePrimaryIngredientIndex = refinePrimaryIngredientIndex,
    )
    val editorWeightSlots = remember(textIngredients, songSeeds) {
        textIngredients.map { ingredient ->
            FindMusicEditorWeightSlot(
                weight = ingredient.weight,
                locked = ingredient.locked,
                completed = ingredient.isActive,
            )
        } + songSeeds.map { seed ->
            FindMusicEditorWeightSlot(
                weight = seed.weight,
                locked = seed.locked,
                completed = seed.isActive,
            )
        }
    }
    val minimumActiveWeight = remember(effectiveFindMusicOperator, sharedQueueLength) {
        FindMusicEditorWeightPolicy.minimumActiveWeight(
            operator = effectiveFindMusicOperator,
            resultLimit = sharedQueueLength,
        )
    }
    val sharedBalanceEndpoints = remember(textIngredients, songSeeds) {
        buildList {
            textIngredients.forEachIndexed { index, ingredient ->
                if (ingredient.isActive) {
                    add(
                        IngredientBalanceEndpoint(
                            combinedIndex = index,
                            label = ingredient.findMusicIngredientLabel(index),
                            weight = ingredient.weight,
                            negative = ingredient.negative,
                        ),
                    )
                }
            }
            songSeeds.forEachIndexed { index, seed ->
                if (seed.isActive) {
                    val combinedIndex = textIngredients.size + index
                    add(
                        IngredientBalanceEndpoint(
                            combinedIndex = combinedIndex,
                            label = seed.findMusicIngredientLabel(index),
                            weight = seed.weight,
                            negative = seed.negative,
                        ),
                    )
                }
            }
        }
    }

    Scaffold(
        contentWindowInsets = WindowInsets.safeDrawing,
        topBar = {
            TopAppBar(
                title = { Text(dynamicTitle) },
                windowInsets = WindowInsets.safeDrawing.only(
                    WindowInsetsSides.Horizontal + WindowInsetsSides.Top,
                ),
                navigationIcon = {
                    IconButton(onClick = {
                        if (activeResult != null) {
                            viewModel.clearFindMusicResults()
                        } else {
                            onBack()
                        }
                    }) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back")
                    }
                },
                actions = {
                    if (activeResult == null && recentSearches.isNotEmpty() && !showInlineRecent) {
                        TextButton(onClick = { showRecentSearches = true }) {
                            Text("Recent")
                        }
                    }
                },
            )
        },
        floatingActionButton = {
            val queueableResult = activeResult?.takeIf {
                it.error == null && it.matches.isNotEmpty()
            }
            when {
                activeResult == null && !imeVisible &&
                    (editorReadiness.canSearch || anyLoading) -> Button(
                    onClick = ::doSearch,
                    enabled = editorReadiness.canSearch,
                    contentPadding = PaddingValues(horizontal = 18.dp, vertical = 12.dp),
                ) {
                    if (anyLoading) {
                        CircularProgressIndicator(
                            modifier = Modifier.size(18.dp),
                            strokeWidth = 2.dp,
                        )
                    } else {
                        Icon(Icons.Default.Search, contentDescription = null)
                    }
                    Spacer(Modifier.width(8.dp))
                    Text(if (anyLoading) "Finding" else "Find $sharedQueueLength")
                }
                queueableResult != null -> Button(
                    onClick = { pendingQueueResult = queueableResult },
                    enabled = displayedQueueEligibility.eligible,
                    contentPadding = PaddingValues(horizontal = 18.dp, vertical = 12.dp),
                ) {
                    Icon(Icons.Default.Add, contentDescription = null)
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("Queue ${queueableResult.matches.size}")
                }
            }
        },
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .then(
                    if (activeResult == null) {
                        Modifier.verticalScroll(editorScrollState)
                    } else {
                        Modifier
                    },
                )
        ) {
            if (activeResult == null) {
                val placeholderHint = remember {
                    listOf(
                        "classic rock, British, 1960s, upbeat",
                        "Heartfelt and nostalgic, with a bittersweet, melancholic feel",
                        "A Latin jazz piece with rhythmic percussion and brass",
                        "big band, major key, swing, brass-heavy, syncopation, baritone vocal",
                    ).random()
                }

                textIngredients.forEachIndexed { index, ingredient ->
                    key(ingredient.id) {
                        val activeOrdinal = textIngredients.take(index)
                            .count(TextIngredientState::isActive)
                        val isRefinePrimary = ingredient.isActive &&
                            effectiveFindMusicOperator == FindMusicOperator.REFINE &&
                            activeOrdinal == refinePrimaryIngredientIndex
                        TextIngredientRow(
                            ingredient = ingredient,
                            index = index,
                            placeholder = placeholderHint,
                            usePrimaryLabel = textIngredients.size == 1 && !hasSongSeeds,
                            canRemove = textIngredients.size > 1 || hasSongSeeds,
                            showWeightControls = false,
                            showLockControl = false,
                            weightAdjustable = FindMusicEditorWeightPolicy.canAdjust(
                                slots = editorWeightSlots,
                                changedIndex = index,
                                minimumActiveWeight = minimumActiveWeight,
                            ),
                            operator = effectiveFindMusicOperator,
                            focusRequester = focusRequester.takeIf { index == 0 },
                            onQueryChange = { viewModel.updateTextIngredientQuery(index, it) },
                            onRemove = { viewModel.removeTextIngredient(index) },
                            onSubmitSearch = {
                                if (editorReadiness.canSearch) doSearch()
                            },
                            onWeightChange = { viewModel.updateTextIngredientWeight(index, it) },
                            onWeightDragEnd = { viewModel.finalizeTextIngredientWeight(index) },
                            onToggleSign = { viewModel.toggleTextIngredientSign(index) },
                            onToggleLock = { viewModel.toggleTextIngredientLock(index) },
                            canSetAvoid = ingredient.negative ||
                                (!isRefinePrimary && FindMusicEditorPolicy.canSetTextToAvoid(
                                    textIngredients = textIngredients,
                                    songSeeds = songSeeds,
                                    index = index,
                                )),
                        )
                    }
                }

                for (index in songSeeds.indices) {
                    key(songSeeds[index].id) {
                        val activeOrdinal = textIngredients.count(TextIngredientState::isActive) +
                            songSeeds.take(index).count(SongSeedState::isActive)
                        val isRefinePrimary = songSeeds[index].isActive &&
                            effectiveFindMusicOperator == FindMusicOperator.REFINE &&
                            activeOrdinal == refinePrimaryIngredientIndex
                        SongSeedRow(
                            seed = songSeeds[index],
                            index = index,
                            placeholder = songPlaceholders.getOrElse(index) { "artist title" },
                            onQueryChange = { viewModel.updateSongSeedQuery(index, it) },
                            onRemove = { viewModel.removeSongSeed(index) },
                            onLookup = {
                                viewModel.searchSongSeed(index)
                                recordingPickerSeedId = songSeeds[index].id
                            },
                            onSubmitSearch = {
                                if (editorReadiness.canSearch) doSearch()
                            },
                            onWeightChange = { viewModel.updateSongSeedWeight(index, it) },
                            onWeightDragEnd = { viewModel.finalizeSongSeedWeight(index) },
                            onToggleSign = { viewModel.toggleSongSeedSign(index) },
                            onToggleLock = { viewModel.toggleSongSeedLock(index) },
                            operator = effectiveFindMusicOperator,
                            showWeightControls = false,
                            showLockControl = false,
                            weightAdjustable = FindMusicEditorWeightPolicy.canAdjust(
                                slots = editorWeightSlots,
                                changedIndex = textIngredients.size + index,
                                minimumActiveWeight = minimumActiveWeight,
                            ),
                            canSetAvoid = songSeeds[index].negative ||
                                (!isRefinePrimary && FindMusicEditorPolicy.canSetSongToAvoid(
                                    textIngredients = textIngredients,
                                    songSeeds = songSeeds,
                                    index = index,
                                )),
                        )
                    }
                }

                Row(
                    modifier = Modifier.fillMaxWidth().padding(horizontal = 8.dp),
                    horizontalArrangement = Arrangement.spacedBy(4.dp),
                ) {
                    TextButton(
                        onClick = viewModel::addTextIngredient,
                        enabled = textIngredients.size < FindMusicQuerySpec.MAX_TEXT_INGREDIENTS,
                    ) {
                        Icon(Icons.Default.Add, contentDescription = null)
                        Spacer(Modifier.width(4.dp))
                        Text("Add description")
                    }
                    TextButton(onClick = viewModel::addSongSeed) {
                        Icon(Icons.Default.Add, contentDescription = null)
                        Spacer(Modifier.width(4.dp))
                        Text("Add recording")
                    }
                }

                if (activeIngredientCount > 1) {
                    HorizontalDivider(
                        modifier = Modifier.padding(horizontal = 16.dp, vertical = 10.dp),
                        color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.5f),
                    )
                    Text(
                        "Composition",
                        style = MaterialTheme.typography.titleSmall,
                        modifier = Modifier.padding(horizontal = 16.dp, vertical = 4.dp),
                    )
                    if (FindMusicEditorPolicy.shouldShowOperatorControl(activeIngredientCount)) {
                        SingleChoiceSegmentedButtonRow(
                            modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp),
                        ) {
                            FindMusicOperator.entries.forEachIndexed { index, operator ->
                                SegmentedButton(
                                    selected = effectiveFindMusicOperator == operator,
                                    onClick = { viewModel.setFindMusicOperator(operator) },
                                    shape = SegmentedButtonDefaults.itemShape(
                                        index = index,
                                        count = FindMusicOperator.entries.size,
                                    ),
                                ) {
                                    Text(
                                        if (operator == FindMusicOperator.ALL_OF) "All of" else "Refine",
                                    )
                                }
                            }
                        }
                    } else {
                        Text(
                            "All of",
                            style = MaterialTheme.typography.titleSmall,
                            modifier = Modifier.padding(horizontal = 16.dp),
                        )
                    }
                    if (effectiveFindMusicOperator == FindMusicOperator.ALL_OF) {
                        Text(
                            "Weighted geometric mean of each ingredient's percentile across " +
                                "recordings in the selected library scope.",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            modifier = Modifier.padding(horizontal = 16.dp, vertical = 4.dp),
                        )
                    }
                    if (effectiveFindMusicOperator == FindMusicOperator.REFINE &&
                        sharedBalanceEndpoints.size == 2
                    ) {
                        RefineControls(
                            ingredients = sharedBalanceEndpoints,
                            primaryIngredientIndex = refinePrimaryIngredientIndex,
                            neighborhood = refineNeighborhood,
                            onSelectPrimary = viewModel::setFindMusicRefinePrimaryIngredient,
                            onSelectNeighborhood = viewModel::setFindMusicRefineNeighborhood,
                        )
                    }
                }

                if (usesSharedIngredientBalance) {
                    val first = sharedBalanceEndpoints[0]
                    val second = sharedBalanceEndpoints[1]
                    SharedIngredientBalance(
                        first = first,
                        second = second,
                        enabled = FindMusicEditorWeightPolicy.canAdjust(
                            slots = editorWeightSlots,
                            changedIndex = first.combinedIndex,
                            minimumActiveWeight = minimumActiveWeight,
                        ),
                        onValueChange = { value ->
                            if (first.combinedIndex < textIngredients.size) {
                                viewModel.updateTextIngredientWeight(first.combinedIndex, value)
                            } else {
                                viewModel.updateSongSeedWeight(
                                    first.combinedIndex - textIngredients.size,
                                    value,
                                )
                            }
                        },
                        onDragEnd = {
                            if (first.combinedIndex < textIngredients.size) {
                                viewModel.finalizeTextIngredientWeight(first.combinedIndex)
                            } else {
                                viewModel.finalizeSongSeedWeight(
                                    first.combinedIndex - textIngredients.size,
                                )
                            }
                        },
                    )
                }

                if (usesPerIngredientWeightControls) {
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 16.dp, vertical = 4.dp),
                        verticalArrangement = Arrangement.spacedBy(4.dp),
                    ) {
                        Text("Balance", style = MaterialTheme.typography.titleSmall)
                        textIngredients.forEachIndexed { index, ingredient ->
                            if (ingredient.isActive) {
                                IngredientAllocationRow(
                                    label = ingredient.findMusicIngredientLabel(index),
                                    value = ingredient.weight,
                                    locked = ingredient.locked,
                                    enabled = FindMusicEditorWeightPolicy.canAdjust(
                                        slots = editorWeightSlots,
                                        changedIndex = index,
                                        minimumActiveWeight = minimumActiveWeight,
                                    ),
                                    onValueChange = {
                                        viewModel.updateTextIngredientWeight(index, it)
                                    },
                                    onDragEnd = {
                                        viewModel.finalizeTextIngredientWeight(index)
                                    },
                                    onToggleLock = {
                                        viewModel.toggleTextIngredientLock(index)
                                    },
                                )
                            }
                        }
                        songSeeds.forEachIndexed { index, seed ->
                            if (seed.isActive) {
                                val combinedIndex = textIngredients.size + index
                                IngredientAllocationRow(
                                    label = seed.findMusicIngredientLabel(index),
                                    value = seed.weight,
                                    locked = seed.locked,
                                    enabled = FindMusicEditorWeightPolicy.canAdjust(
                                        slots = editorWeightSlots,
                                        changedIndex = combinedIndex,
                                        minimumActiveWeight = minimumActiveWeight,
                                    ),
                                    onValueChange = {
                                        viewModel.updateSongSeedWeight(index, it)
                                    },
                                    onDragEnd = {
                                        viewModel.finalizeSongSeedWeight(index)
                                    },
                                    onToggleLock = { viewModel.toggleSongSeedLock(index) },
                                )
                            }
                        }
                    }
                }

                val supportsTextResultPlanner = activeIngredientCount == 1 &&
                    textIngredients.singleOrNull { it.isActive }?.negative == false &&
                    songSeeds.none { it.isActive }
                val supportsAllOfResultPlanner =
                    findMusicOperator == FindMusicOperator.ALL_OF &&
                        activeIngredientCount >= 2
                if (supportsTextResultPlanner || supportsAllOfResultPlanner) {
                    val plannerOptions = if (supportsTextResultPlanner) {
                        listOf(
                            FindMusicTextResultPlanner.CLOSEST,
                            FindMusicTextResultPlanner.VARIED_DPP,
                        )
                    } else {
                        listOf(
                            FindMusicTextResultPlanner.CLOSEST,
                            FindMusicTextResultPlanner.VARIED_ALL_OF_DPP,
                        )
                    }
                    Spacer(modifier = Modifier.height(12.dp))
                    TextResultPlannerControl(
                        selected = textResultPlanner.takeIf { it in plannerOptions }
                            ?: FindMusicTextResultPlanner.CLOSEST,
                        options = plannerOptions,
                        allOfContext = supportsAllOfResultPlanner,
                        enabled = !anyLoading,
                        onSelect = viewModel::setFindMusicTextResultPlanner,
                    )
                }

                if (anyLoading) {
                    Text(
                        findMusicLoadingStatus ?: "Preparing the requested ranking",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp),
                    )
                } else if (!editorReadiness.canSearch && !showInlineRecent) {
                    editorReadiness.reason?.let { reason ->
                        Text(
                            reason,
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp),
                        )
                    }
                }

                if (showInlineRecent) {
                    InlineRecentFindMusic(
                        searches = recentSearches.take(5),
                        onClear = viewModel::clearRecentSearches,
                        onReplay = requestRecentReplay,
                    )
                }

                Spacer(modifier = Modifier.height(88.dp))
            }

            // Results or recent searches
            val result = activeResult
            if (result != null) {
                if (result.error != null) {
                    Text(
                        result.error,
                        style = MaterialTheme.typography.bodyMedium,
                        modifier = Modifier.padding(horizontal = 16.dp, vertical = 12.dp),
                        color = MaterialTheme.colorScheme.error,
                    )
                } else if (result.matches.isNotEmpty()) {
                    val queryLabel = result.querySpec?.displayLabel
                        ?.takeIf(String::isNotBlank)
                        ?: result.query.trim().ifBlank { "Find music" }
                    val resultSummary = buildString {
                        append(queryLabel).append(" \u00b7 ")
                        append(result.matches.size).append(" results")
                    }
                    Text(
                        resultSummary,
                        style = MaterialTheme.typography.titleMedium,
                        modifier = Modifier.padding(horizontal = 16.dp),
                        maxLines = 2,
                        overflow = TextOverflow.Ellipsis,
                    )
                    result.querySpec?.let { spec ->
                        Text(
                            when {
                                result.kind == FindMusicResultKind.TEXT -> buildString {
                                    append(SessionEvidenceText.textPlannerLabel(spec.textResultPlanner))
                                    if (spec.effectiveLibraryAddedDays != null) {
                                        append(" \u00b7 ").append(
                                            libraryAddedDaysLabel(
                                                spec.effectiveLibraryAddedDays,
                                            ),
                                        )
                                    }
                                    if (spec.textResultPlanner == FindMusicTextResultPlanner.CLOSEST) {
                                        result.objectiveRankingDomainCount?.let { count ->
                                            append(" \u00b7 cosine-ranked across ")
                                            append(String.format(Locale.US, "%,d", count))
                                            append(" candidate recordings")
                                        }
                                    } else {
                                        result.matches.map { it.objectiveRank }
                                            .takeIf { ranks -> ranks.all { it != null } }
                                            ?.map { rank -> checkNotNull(rank) }
                                            ?.let { ranks ->
                                                SessionEvidenceText.textMatchReach(
                                                    objectiveRanks = ranks,
                                                    objectiveDomainCount =
                                                        result.objectiveRankingDomainCount,
                                                )
                                            }
                                            ?.let { reach -> append(" \u00b7 ").append(reach) }
                                    }
                                }
                                else -> buildString {
                                    append(recentFindMusicRecipeSummary(spec))
                                    compactFindMusicRankingScope(
                                        query = spec,
                                        objectiveRankingDomainCount =
                                            result.objectiveRankingDomainCount,
                                        ingredientRankingDomainCount =
                                            result.ingredientRankingDomainCount,
                                    )?.let { scope -> append(" \u00b7 ").append(scope) }
                                }
                            },
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            modifier = Modifier.padding(
                                start = 16.dp,
                                top = 2.dp,
                                end = 16.dp,
                            ),
                            maxLines = 3,
                            overflow = TextOverflow.Ellipsis,
                        )
                    }
                    result.stableResultReduction?.collapsedEquivalentCount
                        ?.takeIf { it > 0 }
                        ?.let { skipped ->
                            Text(
                                "$skipped verified ${if (skipped == 1) "copy" else "copies"} omitted",
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                                modifier = Modifier.padding(horizontal = 16.dp, vertical = 2.dp),
                            )
                        }
                    displayedQueueEligibility.reason?.let { reason ->
                        Text(
                            reason,
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.error,
                            modifier = Modifier.padding(horizontal = 16.dp, vertical = 2.dp),
                        )
                    }
                    Spacer(modifier = Modifier.height(4.dp))

                    LazyColumn(
                        modifier = Modifier.weight(1f).fillMaxWidth(),
                        contentPadding = PaddingValues(bottom = 88.dp),
                    ) {
                        items(result.matches.size) { index ->
                            val match = result.matches[index]
                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(horizontal = 16.dp, vertical = 7.dp),
                                verticalAlignment = Alignment.Top,
                            ) {
                                Text(
                                    "${index + 1}.",
                                    style = MaterialTheme.typography.labelMedium,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                                    modifier = Modifier.width(30.dp).padding(top = 2.dp),
                                )
                                Spacer(modifier = Modifier.width(4.dp))
                                Column(modifier = Modifier.weight(1f)) {
                                    Text(
                                        match.track.title ?: "Unknown title",
                                        style = MaterialTheme.typography.bodyMedium,
                                        fontWeight = FontWeight.SemiBold,
                                        maxLines = 1,
                                        overflow = TextOverflow.Ellipsis,
                                    )
                                    Text(
                                        listOfNotNull(
                                            match.track.artist?.takeIf(String::isNotBlank),
                                            match.track.album?.takeIf(String::isNotBlank),
                                        ).joinToString(" \u00b7 ").ifBlank {
                                            "Unknown artist and album"
                                        },
                                        style = MaterialTheme.typography.bodySmall,
                                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                                        maxLines = 1,
                                        overflow = TextOverflow.Ellipsis,
                                    )
                                    findMusicResultEvidence(result, index)?.let { evidence ->
                                        ExpandableEvidenceLine(evidence)
                                    }
                                }
                            }
                            if (index < result.matches.lastIndex) {
                                HorizontalDivider(
                                    modifier = Modifier.padding(start = 50.dp),
                                    color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.25f),
                                )
                            }
                        }
                    }

                } else {
                    Text(
                        "No matching results were found. Edit the query or ingredients and try again.",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.padding(16.dp),
                    )
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun RecentFindMusicSheet(
    searches: List<FindMusicQuerySpec>,
    onDismiss: () -> Unit,
    onClear: () -> Unit,
    onReplay: (FindMusicQuerySpec) -> Unit,
) {
    ModalBottomSheet(onDismissRequest = onDismiss) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .navigationBarsPadding()
                .padding(bottom = 12.dp),
        ) {
            Row(
                modifier = Modifier.fillMaxWidth().padding(start = 24.dp, end = 12.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(
                    "Recent searches",
                    style = MaterialTheme.typography.titleLarge,
                    modifier = Modifier.weight(1f),
                )
                TextButton(onClick = onClear, enabled = searches.isNotEmpty()) {
                    Text("Clear")
                }
            }
            if (searches.isEmpty()) {
                Text(
                    "Completed Find Music requests will appear here.",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(horizontal = 24.dp, vertical = 20.dp),
                )
            } else {
                LazyColumn(modifier = Modifier.fillMaxWidth().heightIn(max = 520.dp)) {
                    items(searches.size) { index ->
                        val search = searches[index]
                        RecentFindMusicRow(
                            search = search,
                            horizontalPadding = 24.dp,
                            onReplay = onReplay,
                        )
                        if (index < searches.lastIndex) {
                            HorizontalDivider(
                                modifier = Modifier.padding(horizontal = 16.dp),
                                color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.4f),
                            )
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun RecentFindMusicReplayDialog(
    search: FindMusicQuerySpec,
    currentControlsSearch: FindMusicQuerySpec,
    onDismiss: () -> Unit,
    onUseCurrentControls: () -> Unit,
    onUseSavedControls: () -> Unit,
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Run this search again") },
        text = {
            Text(
                search.displayLabel,
                maxLines = 3,
                overflow = TextOverflow.Ellipsis,
            )
        },
        confirmButton = {
            Column(
                modifier = Modifier.fillMaxWidth(),
                horizontalAlignment = Alignment.End,
            ) {
                RecentFindMusicReplayChoice(
                    label = "Use current result controls",
                    summary = recentFindMusicControlSummary(currentControlsSearch),
                    onClick = onUseCurrentControls,
                )
                RecentFindMusicReplayChoice(
                    label = "Use saved result controls",
                    summary = recentFindMusicControlSummary(search),
                    onClick = onUseSavedControls,
                )
                TextButton(onClick = onDismiss) {
                    Text("Cancel")
                }
            }
        },
    )
}

@Composable
private fun RecentFindMusicReplayChoice(
    label: String,
    summary: String,
    onClick: () -> Unit,
) {
    TextButton(
        onClick = onClick,
        modifier = Modifier.fillMaxWidth(),
        contentPadding = PaddingValues(horizontal = 12.dp, vertical = 10.dp),
    ) {
        Column(modifier = Modifier.fillMaxWidth()) {
            Text(
                label,
                style = MaterialTheme.typography.labelLarge,
                color = MaterialTheme.colorScheme.primary,
            )
            Text(
                summary,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

private fun recentFindMusicControlSummary(search: FindMusicQuerySpec): String {
    val resultStyle = when {
        search.textResultPlanner == FindMusicTextResultPlanner.VARIED_DPP ||
            search.textResultPlanner == FindMusicTextResultPlanner.VARIED_ALL_OF_DPP -> "Varied"
        search.operator == FindMusicOperator.REFINE -> search.refineSpec?.let { refine ->
            "Refine, ${refineNeighborhoodLabel(refine.neighborhood)} primary neighborhood"
        } ?: "Refine"
        search.operator == FindMusicOperator.ALL_OF && search.activeIngredientCount >= 2 ->
            "Ranked"
        else -> "Closest"
    }
    return "${libraryAddedDaysLabel(search.effectiveLibraryAddedDays)} \u00b7 " +
        "${search.resultLimit} results \u00b7 $resultStyle"
}

@Composable
private fun InlineRecentFindMusic(
    searches: List<FindMusicQuerySpec>,
    onClear: () -> Unit,
    onReplay: (FindMusicQuerySpec) -> Unit,
) {
    HorizontalDivider(
        modifier = Modifier.padding(horizontal = 16.dp, vertical = 10.dp),
        color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.5f),
    )
    Row(
        modifier = Modifier.fillMaxWidth().padding(start = 16.dp, end = 8.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(
            "Recent",
            style = MaterialTheme.typography.titleSmall,
            modifier = Modifier.weight(1f),
        )
        TextButton(onClick = onClear) {
            Text("Clear")
        }
    }
    searches.forEachIndexed { index, search ->
        key(search.stateKey) {
            RecentFindMusicRow(
                search = search,
                horizontalPadding = 16.dp,
                onReplay = onReplay,
                compact = true,
            )
        }
        if (index < searches.lastIndex) {
            HorizontalDivider(
                modifier = Modifier.padding(start = 16.dp),
                color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.25f),
            )
        }
    }
}

@Composable
private fun RecentFindMusicRow(
    search: FindMusicQuerySpec,
    horizontalPadding: androidx.compose.ui.unit.Dp,
    onReplay: (FindMusicQuerySpec) -> Unit,
    compact: Boolean = false,
) {
    val recipe = recentFindMusicRecipeSummary(search)
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clickable { onReplay(search) }
            .padding(
                horizontal = horizontalPadding,
                vertical = if (compact) 8.dp else 12.dp,
            ),
    ) {
        Text(
            search.displayLabel,
            style = if (compact) {
                MaterialTheme.typography.bodyMedium
            } else {
                MaterialTheme.typography.bodyLarge
            },
            fontWeight = if (compact) FontWeight.SemiBold else FontWeight.Normal,
            maxLines = if (compact) 1 else 2,
            overflow = TextOverflow.Ellipsis,
        )
        Text(
            "$recipe \u00b7 ${search.resultLimit} results",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            maxLines = 2,
            overflow = TextOverflow.Ellipsis,
        )
    }
}

private fun recentFindMusicRecipeSummary(search: FindMusicQuerySpec): String {
    val base = if (search.isSimplePositiveTextOnly) {
        SessionEvidenceText.textPlannerLabel(search.textResultPlanner)
    } else if (search.activeIngredientCount == 1) {
        "Closest"
    } else {
        val weights = search.activeTextIngredients.map { it.weight } +
            search.songSeeds.filter { it.weight > 0f }.map { it.weight }
        if (weights.isEmpty()) {
            if (search.operator == FindMusicOperator.ALL_OF) "All of" else "Refine"
        } else {
            when (search.operator) {
                FindMusicOperator.ALL_OF -> {
                    val total = weights.sum()
                    val allocation = weights.joinToString("/") { weight ->
                        formatKnob(weight / total * 100f)
                    }
                    buildString {
                        append("All of \u00b7 $allocation priority")
                        if (
                            search.textResultPlanner ==
                            FindMusicTextResultPlanner.VARIED_ALL_OF_DPP
                        ) {
                            append(" \u00b7 Varied")
                        }
                    }
                }
                FindMusicOperator.REFINE -> {
                    val refine = search.refineSpec
                    if (refine == null) {
                        "Refine"
                    } else {
                        "Refine \u00b7 primary ${refineNeighborhoodLabel(refine.neighborhood)}"
                    }
                }
            }
        }
    }
    return if (search.effectiveLibraryAddedDays == null) {
        base
    } else {
        "$base \u00b7 ${libraryAddedDaysLabel(search.effectiveLibraryAddedDays)}"
    }
}

private fun compactFindMusicRankingScope(
    query: FindMusicQuerySpec,
    objectiveRankingDomainCount: Int?,
    ingredientRankingDomainCount: Int?,
): String? {
    fun count(value: Int): String = String.format(Locale.US, "%,d", value)
    return when {
        query.operator == FindMusicOperator.ALL_OF &&
            objectiveRankingDomainCount != null &&
            ingredientRankingDomainCount != null &&
            objectiveRankingDomainCount == ingredientRankingDomainCount ->
            "${count(objectiveRankingDomainCount)} recordings compared"
        query.operator == FindMusicOperator.ALL_OF &&
            objectiveRankingDomainCount != null &&
            ingredientRankingDomainCount != null ->
            "${count(objectiveRankingDomainCount)} eligible \u00b7 ingredient ranks across " +
                "${count(ingredientRankingDomainCount)} in scope"
        query.operator == FindMusicOperator.ALL_OF && objectiveRankingDomainCount != null ->
            "${count(objectiveRankingDomainCount)} eligible recordings"
        query.operator == FindMusicOperator.REFINE &&
            objectiveRankingDomainCount != null &&
            ingredientRankingDomainCount != null ->
            "${count(objectiveRankingDomainCount)} in primary neighborhood \u00b7 " +
                "ingredient ranks across ${count(ingredientRankingDomainCount)} in scope"
        ingredientRankingDomainCount != null ->
            "${count(ingredientRankingDomainCount)} recordings compared per ingredient"
        else -> null
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun RecordingPickerSheet(
    query: String,
    state: RecordingLookupState,
    onDismiss: () -> Unit,
    onRetry: () -> Unit,
    onChoose: (com.powerampstartradio.data.EmbeddedTrack) -> Unit,
) {
    LaunchedEffect(query, state) {
        if (state is RecordingLookupState.Idle) onRetry()
    }
    ModalBottomSheet(onDismissRequest = onDismiss) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .navigationBarsPadding()
                .padding(bottom = 12.dp),
        ) {
            Text(
                "Choose the exact recording",
                style = MaterialTheme.typography.titleLarge,
                modifier = Modifier.padding(horizontal = 24.dp),
            )
            Text(
                "Matching artist, album, title, and filename for \"${query.trim()}\"",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(start = 24.dp, top = 4.dp, end = 24.dp, bottom = 12.dp),
            )
            when (state) {
                RecordingLookupState.Idle -> {
                    Row(
                        modifier = Modifier.padding(horizontal = 24.dp, vertical = 20.dp),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(12.dp),
                    ) {
                        CircularProgressIndicator(modifier = Modifier.size(20.dp), strokeWidth = 2.dp)
                        Text(
                            "Starting recording lookup",
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                }
                is RecordingLookupState.Loading -> Row(
                    modifier = Modifier.padding(horizontal = 24.dp, vertical = 20.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(12.dp),
                ) {
                    CircularProgressIndicator(modifier = Modifier.size(20.dp), strokeWidth = 2.dp)
                    Text(
                        state.message,
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
                is RecordingLookupState.Failure -> {
                    Text(
                        state.message,
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.error,
                        modifier = Modifier.padding(horizontal = 24.dp, vertical = 12.dp),
                    )
                    TextButton(
                        onClick = onRetry,
                        modifier = Modifier.padding(horizontal = 12.dp),
                    ) {
                        Icon(Icons.Default.Refresh, contentDescription = null)
                        Spacer(Modifier.width(8.dp))
                        Text("Try again")
                    }
                }
                is RecordingLookupState.Success -> {
                    if (state.candidates.isEmpty()) {
                        Text(
                            "No matching recordings",
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            modifier = Modifier.padding(horizontal = 24.dp, vertical = 20.dp),
                        )
                    } else {
                        Text(
                            if (state.hasMoreMatches) {
                                "Showing the first ${state.candidates.size} matches"
                            } else {
                                "${state.candidates.size} matching recordings"
                            },
                            style = MaterialTheme.typography.labelMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            modifier = Modifier.padding(horizontal = 24.dp, vertical = 4.dp),
                        )
                        LazyColumn(modifier = Modifier.fillMaxWidth().heightIn(max = 520.dp)) {
                            items(state.candidates.size) { index ->
                                val track = state.candidates[index]
                                val evidence = RecordingCandidateEvidenceFormatter.format(track)
                                Column(
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .clickable { onChoose(track) }
                                        .padding(horizontal = 24.dp, vertical = 12.dp),
                                ) {
                                    Text(
                                        track.title ?: "Unknown title",
                                        style = MaterialTheme.typography.bodyLarge,
                                        fontWeight = FontWeight.Bold,
                                        maxLines = 1,
                                        overflow = TextOverflow.Ellipsis,
                                    )
                                    Text(
                                        evidence.artistAndAlbum,
                                        style = MaterialTheme.typography.bodySmall,
                                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                                        maxLines = 1,
                                        overflow = TextOverflow.Ellipsis,
                                    )
                                    Text(
                                        evidence.durationAndFile,
                                        style = MaterialTheme.typography.labelSmall,
                                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                                        maxLines = 1,
                                        overflow = TextOverflow.Ellipsis,
                                    )
                                }
                                if (index < state.candidates.lastIndex) {
                                    HorizontalDivider(
                                        modifier = Modifier.padding(horizontal = 16.dp),
                                        color = MaterialTheme.colorScheme.outlineVariant
                                            .copy(alpha = 0.4f),
                                    )
                                }
                            }
                        }
                        if (state.hasMoreMatches) {
                            Text(
                                "More recordings match. Refine the text to narrow the choice.",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                                modifier = Modifier.padding(horizontal = 24.dp, vertical = 8.dp),
                            )
                        }
                    }
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun TextResultPlannerControl(
    selected: FindMusicTextResultPlanner,
    options: List<FindMusicTextResultPlanner>,
    allOfContext: Boolean,
    enabled: Boolean,
    onSelect: (FindMusicTextResultPlanner) -> Unit,
) {
    Column(modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp)) {
        Text("Result set", style = MaterialTheme.typography.titleSmall)
        SingleChoiceSegmentedButtonRow(modifier = Modifier.fillMaxWidth()) {
            options.forEachIndexed { index, planner ->
                SegmentedButton(
                    selected = selected == planner,
                    onClick = { onSelect(planner) },
                    enabled = enabled,
                    shape = SegmentedButtonDefaults.itemShape(
                        index = index,
                        count = options.size,
                    ),
                ) {
                    Text(
                        when (planner) {
                            FindMusicTextResultPlanner.CLOSEST ->
                                if (allOfContext) "Ranked" else "Closest"
                            FindMusicTextResultPlanner.VARIED_DPP -> "Varied (DPP)"
                            FindMusicTextResultPlanner.VARIED_ALL_OF_DPP -> "Varied (DPP)"
                        },
                    )
                }
            }
        }
        Text(
            when {
                allOfContext && selected == FindMusicTextResultPlanner.CLOSEST ->
                    "Highest weighted All-of scores first."
                allOfContext &&
                    selected == FindMusicTextResultPlanner.VARIED_ALL_OF_DPP ->
                    "Uses the weighted All-of score as DPP quality to select a less redundant set."
                else -> SessionEvidenceText.textPlannerDescription(selected)
            },
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.padding(top = 4.dp),
        )
    }
}

@Composable
private fun LibraryAddedDaysControl(
    dayCount: Int?,
    enabled: Boolean,
    onApply: (Int?) -> Unit,
    titleStyle: TextStyle,
    modifier: Modifier = Modifier,
) {
    var dayDraft by rememberSaveable {
        mutableStateOf((dayCount ?: DEFAULT_LIBRARY_ADDED_DAYS).toString())
    }
    var fieldFocused by remember { mutableStateOf(false) }
    val parsedDays = dayDraft.toIntOrNull()
    val validDays = parsedDays != null && parsedDays in 1..MAX_LIBRARY_ADDED_DAYS
    val focusManager = LocalFocusManager.current

    LaunchedEffect(dayCount) {
        if (dayCount != null) {
            dayDraft = dayCount.toString()
        }
    }

    fun commitDraft() {
        if (validDays) {
            onApply(parsedDays)
        } else {
            dayDraft = (dayCount ?: DEFAULT_LIBRARY_ADDED_DAYS).toString()
        }
    }

    Column(modifier = modifier.fillMaxWidth()) {
        Text(
            "Added to Poweramp",
            style = titleStyle,
        )
        Row(
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier.padding(top = 4.dp),
        ) {
            FilterChip(
                selected = dayCount == null,
                onClick = { onApply(null) },
                label = { Text("All dates") },
                enabled = enabled,
            )
            FilterChip(
                selected = dayCount != null,
                onClick = {
                    val selectedDays = parsedDays.takeIf { validDays }
                        ?: DEFAULT_LIBRARY_ADDED_DAYS
                    dayDraft = selectedDays.toString()
                    onApply(selectedDays)
                },
                label = { Text("Last") },
                enabled = enabled,
            )
            if (dayCount != null) {
                OutlinedTextField(
                    value = dayDraft,
                    onValueChange = { candidate ->
                        if (candidate.length <= 5 && candidate.all(Char::isDigit)) {
                            dayDraft = candidate
                        }
                    },
                    modifier = Modifier
                        .width(128.dp)
                        .onFocusChanged { state ->
                            val lostFocus = fieldFocused && !state.isFocused
                            fieldFocused = state.isFocused
                            if (lostFocus) commitDraft()
                        },
                    label = { Text("Days") },
                    isError = !validDays,
                    enabled = enabled,
                    singleLine = true,
                    keyboardOptions = androidx.compose.foundation.text.KeyboardOptions(
                        keyboardType = androidx.compose.ui.text.input.KeyboardType.Number,
                        imeAction = androidx.compose.ui.text.input.ImeAction.Done,
                    ),
                    keyboardActions = androidx.compose.foundation.text.KeyboardActions(
                        onDone = {
                            if (validDays) {
                                commitDraft()
                                focusManager.clearFocus()
                            }
                        },
                    ),
                )
            }
        }
        if (dayCount != null && !validDays) {
            Text(
                "Enter 1 to 36,500 days",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.error,
            )
        }
    }
}

private data class IngredientBalanceEndpoint(
    val combinedIndex: Int,
    val label: String,
    val weight: Float,
    val negative: Boolean,
)

private fun TextIngredientState.findMusicIngredientLabel(index: Int): String {
    val description = query.trim().ifBlank { "Description ${index + 1}" }
    return if (negative) "Less like: $description" else description
}

private fun SongSeedState.findMusicIngredientLabel(index: Int): String {
    val recording = confirmedTrack?.let { track ->
        recordingDisplayLabel(track.artist, track.title, "Recording ${index + 1}")
    } ?: query.trim().ifBlank { "Recording ${index + 1}" }
    return if (negative) "Less like: $recording" else recording
}

@Composable
private fun RefineControls(
    ingredients: List<IngredientBalanceEndpoint>,
    primaryIngredientIndex: Int,
    neighborhood: FindMusicRefineNeighborhood,
    onSelectPrimary: (Int) -> Unit,
    onSelectNeighborhood: (FindMusicRefineNeighborhood) -> Unit,
) {
    if (ingredients.size != 2) return
    val primary = ingredients.getOrNull(primaryIngredientIndex)
    val secondary = ingredients.getOrNull(1 - primaryIngredientIndex)

    Column(
        modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 4.dp),
        verticalArrangement = Arrangement.spacedBy(4.dp),
    ) {
        Text("Keep close to", style = MaterialTheme.typography.titleSmall)
        Column(modifier = Modifier.selectableGroup()) {
            ingredients.forEachIndexed { index, ingredient ->
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .selectable(
                            selected = primaryIngredientIndex == index,
                            enabled = !ingredient.negative,
                            role = Role.RadioButton,
                            onClick = { onSelectPrimary(index) },
                        )
                        .padding(vertical = 2.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    RadioButton(
                        selected = primaryIngredientIndex == index,
                        onClick = null,
                        enabled = !ingredient.negative,
                    )
                    Text(
                        ingredient.label,
                        style = MaterialTheme.typography.bodyMedium,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                    )
                }
            }
        }
        Spacer(Modifier.height(4.dp))
        Text("Primary neighborhood", style = MaterialTheme.typography.titleSmall)
        SingleChoiceSegmentedButtonRow(modifier = Modifier.fillMaxWidth()) {
            FindMusicRefineNeighborhood.entries.forEachIndexed { index, option ->
                SegmentedButton(
                    selected = neighborhood == option,
                    onClick = { onSelectNeighborhood(option) },
                    shape = SegmentedButtonDefaults.itemShape(
                        index = index,
                        count = FindMusicRefineNeighborhood.entries.size,
                    ),
                ) {
                    Text(refineNeighborhoodLabel(option))
                }
            }
        }
        if (primary != null && secondary != null) {
            Text(
                "Ranks by ${secondary.label} within ${primary.label}'s nearest " +
                    "${refineNeighborhoodLabel(neighborhood)} of eligible recordings.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                maxLines = 3,
                overflow = TextOverflow.Ellipsis,
            )
        }
    }
}

private fun refineNeighborhoodLabel(value: FindMusicRefineNeighborhood): String = when (value) {
    FindMusicRefineNeighborhood.TOP_0_25_PERCENT -> "0.25%"
    FindMusicRefineNeighborhood.TOP_0_5_PERCENT -> "0.5%"
    FindMusicRefineNeighborhood.TOP_1_PERCENT -> "1%"
    FindMusicRefineNeighborhood.TOP_2_PERCENT -> "2%"
}

@Composable
private fun SharedIngredientBalance(
    first: IngredientBalanceEndpoint,
    second: IngredientBalanceEndpoint,
    enabled: Boolean,
    onValueChange: (Float) -> Unit,
    onDragEnd: () -> Unit,
) {
    val minimumShare = 0.1f
    val maximumShare = 0.9f

    fun evidence(endpoint: IngredientBalanceEndpoint): String =
        "${formatKnob(endpoint.weight * 100f)}% priority"
    val accessibilityDescription =
        "${first.label}: ${evidence(first)}; ${second.label}: ${evidence(second)}"

    Column(
        modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 4.dp),
    ) {
        Text("Priority", style = MaterialTheme.typography.titleSmall)
        Row(modifier = Modifier.fillMaxWidth()) {
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    first.label,
                    style = MaterialTheme.typography.labelMedium,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )
                Text(
                    evidence(first),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            Column(
                modifier = Modifier.weight(1f),
                horizontalAlignment = Alignment.End,
            ) {
                Text(
                    second.label,
                    style = MaterialTheme.typography.labelMedium,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )
                Text(
                    evidence(second),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
        Slider(
            value = first.weight.coerceIn(minimumShare, maximumShare),
            onValueChange = { requested ->
                onValueChange((requested * 10f).roundToInt().coerceIn(1, 9) / 10f)
            },
            onValueChangeFinished = onDragEnd,
            valueRange = minimumShare..maximumShare,
            steps = 7,
            enabled = enabled,
            modifier = Modifier.fillMaxWidth().semantics {
                stateDescription = accessibilityDescription
            },
        )
    }
}

@Composable
private fun IngredientAllocationRow(
    label: String,
    value: Float,
    locked: Boolean,
    enabled: Boolean,
    onValueChange: (Float) -> Unit,
    onDragEnd: () -> Unit,
    onToggleLock: () -> Unit,
) {
    val allocationText = "${formatKnob(value * 100f)}% ranking influence"
    Column(modifier = Modifier.fillMaxWidth()) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    label,
                    style = MaterialTheme.typography.labelMedium,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )
                Text(
                    allocationText,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            Row(
                modifier = Modifier
                    .selectable(
                        selected = locked,
                        onClick = onToggleLock,
                        role = Role.Checkbox,
                    )
                    .padding(start = 8.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Checkbox(checked = locked, onCheckedChange = null)
                Text("Hold", style = MaterialTheme.typography.labelMedium)
            }
        }
        Slider(
            value = value,
            onValueChange = { requested ->
                onValueChange((requested * 100f).roundToInt() / 100f)
            },
            onValueChangeFinished = onDragEnd,
            valueRange = 0f..1f,
            steps = 99,
            enabled = enabled,
            modifier = Modifier.fillMaxWidth().semantics {
                stateDescription = "$label: $allocationText"
            },
        )
    }
}

@Composable
private fun TextIngredientRow(
    ingredient: TextIngredientState,
    index: Int,
    placeholder: String,
    usePrimaryLabel: Boolean,
    canRemove: Boolean,
    showWeightControls: Boolean,
    showLockControl: Boolean,
    weightAdjustable: Boolean,
    operator: FindMusicOperator,
    focusRequester: FocusRequester?,
    onQueryChange: (String) -> Unit,
    onRemove: () -> Unit,
    onSubmitSearch: () -> Unit,
    onWeightChange: (Float) -> Unit,
    onWeightDragEnd: () -> Unit,
    onToggleSign: () -> Unit,
    onToggleLock: () -> Unit,
    canSetAvoid: Boolean,
) {
    Column(modifier = Modifier.fillMaxWidth()) {
        Row(
            modifier = Modifier.fillMaxWidth().padding(start = 16.dp, end = 8.dp, top = 2.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            OutlinedTextField(
                value = ingredient.query,
                onValueChange = onQueryChange,
                label = {
                    Text(
                        if (usePrimaryLabel) {
                            "Describe what you want to hear"
                        } else {
                            "Description ${index + 1}"
                        },
                    )
                },
                placeholder = {
                    Text(
                        placeholder,
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.5f),
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                    )
                },
                singleLine = true,
                modifier = Modifier
                    .weight(1f)
                    .then(focusRequester?.let { Modifier.focusRequester(it) } ?: Modifier),
                keyboardActions = androidx.compose.foundation.text.KeyboardActions(
                    onSearch = { onSubmitSearch() },
                ),
                keyboardOptions = androidx.compose.foundation.text.KeyboardOptions(
                    imeAction = androidx.compose.ui.text.input.ImeAction.Search,
                ),
            )
            if (canRemove) {
                IconButton(
                    onClick = onRemove,
                    modifier = Modifier.size(48.dp),
                ) {
                    Icon(
                        Icons.Default.Close,
                        contentDescription = "Remove description ${index + 1}",
                    )
                }
            }
        }
        if (ingredient.isActive) {
            IngredientControls(
                value = ingredient.weight,
                negative = ingredient.negative,
                locked = ingredient.locked,
                operator = operator,
                showWeightControls = showWeightControls,
                showLockControl = showLockControl,
                weightAdjustable = weightAdjustable,
                onValueChange = onWeightChange,
                onDragEnd = onWeightDragEnd,
                onToggleSign = onToggleSign,
                onToggleLock = onToggleLock,
                canSetAvoid = canSetAvoid,
                modifier = Modifier.padding(horizontal = 16.dp),
            )
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun IngredientControls(
    value: Float,
    negative: Boolean,
    locked: Boolean,
    operator: FindMusicOperator,
    onValueChange: (Float) -> Unit,
    onDragEnd: () -> Unit,
    onToggleSign: () -> Unit,
    onToggleLock: () -> Unit,
    canSetAvoid: Boolean,
    showWeightControls: Boolean = true,
    showLockControl: Boolean = true,
    weightAdjustable: Boolean = true,
    modifier: Modifier = Modifier,
) {
    val showSignControl = FindMusicEditorPolicy.shouldShowSignControl(
        negative = negative,
        canSetAvoid = canSetAvoid,
        operator = operator,
    )
    if (!showSignControl && !showWeightControls) return

    Column(modifier = modifier) {
        if (showWeightControls) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(
                    "Ranking influence: ${formatKnob(value * 100f)}%",
                    style = MaterialTheme.typography.labelMedium,
                    modifier = Modifier.weight(1f),
                )
                if (showLockControl) {
                    Row(
                        modifier = Modifier
                            .selectable(
                                selected = locked,
                                onClick = onToggleLock,
                                role = Role.Checkbox,
                            )
                            .padding(start = 8.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Checkbox(checked = locked, onCheckedChange = null)
                        Text("Hold share", style = MaterialTheme.typography.labelMedium)
                    }
                }
            }
        }
        if (showSignControl || showWeightControls) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                if (showSignControl) {
                    SingleChoiceSegmentedButtonRow(modifier = if (showWeightControls) {
                        Modifier.weight(0.9f)
                    } else {
                        Modifier.widthIn(max = 208.dp)
                    }) {
                        SegmentedButton(
                            selected = !negative,
                            onClick = { if (negative) onToggleSign() },
                            shape = SegmentedButtonDefaults.itemShape(index = 0, count = 2),
                        ) {
                            Text("Like")
                        }
                        SegmentedButton(
                            selected = negative,
                            onClick = { if (!negative) onToggleSign() },
                            shape = SegmentedButtonDefaults.itemShape(index = 1, count = 2),
                        ) {
                            Text("Less like")
                        }
                    }
                }
                if (showWeightControls) {
                    if (showSignControl) Spacer(Modifier.width(12.dp))
                    val accessibilityDescription =
                        "Ranking influence: ${formatKnob(value * 100f)}%"
                    Slider(
                        value = value,
                        onValueChange = { requested ->
                            onValueChange((requested * 100f).roundToInt() / 100f)
                        },
                        onValueChangeFinished = onDragEnd,
                        valueRange = 0f..1f,
                        steps = 99,
                        enabled = weightAdjustable,
                        modifier = (if (showSignControl) {
                            Modifier.weight(1.1f)
                        } else {
                            Modifier.fillMaxWidth()
                        }).semantics {
                            stateDescription = accessibilityDescription
                        },
                    )
                }
            }
        }
    }
}

// ---- Song Seed Row (extracted composable with local text state) ----

@Composable
private fun SongSeedRow(
    seed: SongSeedState,
    index: Int,
    placeholder: String,
    onQueryChange: (String) -> Unit,
    onRemove: () -> Unit,
    onLookup: () -> Unit,
    onSubmitSearch: () -> Unit,
    onToggleSign: () -> Unit,
    onWeightChange: (Float) -> Unit,
    onWeightDragEnd: () -> Unit,
    onToggleLock: () -> Unit,
    operator: FindMusicOperator,
    showWeightControls: Boolean,
    showLockControl: Boolean,
    weightAdjustable: Boolean,
    canSetAvoid: Boolean,
) {
    // Local text state — synced from ViewModel but owned here to avoid
    // recomposition cursor fights. The ViewModel is the source of truth
    // for confirmed tracks (which overwrite the local text).
    val confirmedDisplayLabel = seed.confirmedTrack?.let { track ->
        recordingDisplayLabel(track.artist, track.title, "Recording ${index + 1}")
    }
    var localText by remember(seed.id) {
        mutableStateOf(confirmedDisplayLabel ?: seed.query)
    }

    // Sync from ViewModel when confirmed track changes the query externally
    LaunchedEffect(confirmedDisplayLabel) {
        if (confirmedDisplayLabel != null && localText != confirmedDisplayLabel) {
            localText = confirmedDisplayLabel
        }
    }

    Column {
        val confirmedTrack = seed.confirmedTrack
        if (confirmedTrack == null) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(start = 16.dp, end = 8.dp, top = 2.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                OutlinedTextField(
                    value = localText,
                    onValueChange = {
                        localText = it
                        onQueryChange(it)
                    },
                    label = {
                        Text(
                            "Recording ${index + 1}",
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis,
                        )
                    },
                    placeholder = {
                        Text(
                            placeholder,
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.5f),
                            maxLines = 1,
                        )
                    },
                    singleLine = true,
                    modifier = Modifier.weight(1f).heightIn(min = 56.dp),
                    trailingIcon = {
                        if (localText.isNotBlank()) {
                            IconButton(onClick = onLookup) {
                                Icon(
                                    Icons.Default.Search,
                                    contentDescription = "Choose an exact recording",
                                    tint = MaterialTheme.colorScheme.tertiary,
                                )
                            }
                        }
                    },
                    keyboardActions = androidx.compose.foundation.text.KeyboardActions(
                        onSearch = {
                            if (localText.isNotBlank()) onLookup() else onSubmitSearch()
                        },
                    ),
                    keyboardOptions = androidx.compose.foundation.text.KeyboardOptions(
                        imeAction = androidx.compose.ui.text.input.ImeAction.Search,
                    ),
                )
                IconButton(onClick = onRemove, modifier = Modifier.size(48.dp)) {
                    Icon(Icons.Default.Close, contentDescription = "Remove recording")
                }
            }
        } else {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(start = 16.dp, end = 8.dp, top = 6.dp, bottom = 2.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        confirmedTrack.title ?: "Unknown title",
                        style = MaterialTheme.typography.bodyMedium,
                        fontWeight = FontWeight.SemiBold,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                    )
                    Text(
                        listOfNotNull(
                            confirmedTrack.artist?.takeIf(String::isNotBlank),
                            confirmedTrack.album?.takeIf(String::isNotBlank),
                        ).joinToString(" \u00b7 ").ifBlank { "Unknown artist and album" },
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                    )
                }
                IconButton(onClick = onLookup) {
                    Icon(
                        Icons.Default.Refresh,
                        contentDescription = "Choose a different recording",
                    )
                }
                IconButton(onClick = onRemove) {
                    Icon(Icons.Default.Close, contentDescription = "Remove recording")
                }
            }
        }

        if (seed.isActive) {
            IngredientControls(
                value = seed.weight,
                negative = seed.negative,
                locked = seed.locked,
                operator = operator,
                showWeightControls = showWeightControls,
                showLockControl = showLockControl,
                weightAdjustable = weightAdjustable,
                onValueChange = onWeightChange,
                onDragEnd = onWeightDragEnd,
                onToggleSign = onToggleSign,
                onToggleLock = onToggleLock,
                canSetAvoid = canSetAvoid,
                modifier = Modifier.padding(horizontal = 16.dp),
            )
        }

    }
}

private data class FindMusicRowEvidence(
    val compact: String,
    val expanded: String,
)

private fun findMusicResultEvidence(
    result: TextSearchResult,
    displayedIndex: Int,
): FindMusicRowEvidence? {
    val match = result.matches.getOrNull(displayedIndex) ?: return null
    val spec = result.querySpec ?: return null

    if (result.kind == FindMusicResultKind.TEXT) {
        val rank = match.objectiveRank ?: return null
        val total = result.objectiveRankingDomainCount ?: return null
        if (rank !in 1..total ||
            (spec.textResultPlanner == FindMusicTextResultPlanner.CLOSEST &&
                rank == displayedIndex + 1)
        ) {
            return null
        }
        val compactRank = LibraryRankEvidenceText.rank(rank) ?: return null
        val fullRank = LibraryRankEvidenceText.rankWithTopFraction(rank, total)
            ?: return null
        return FindMusicRowEvidence(
            compact = "Text match $compactRank",
            expanded = "Text match \u00b7 $fullRank",
        )
    }

    if (match.anchorPercentiles.isEmpty()) return null
    val labels = spec.activeEvidenceLabels
    val ingredientTotal = result.ingredientRankingDomainCount ?: return null
    if (match.anchorPercentiles.size != labels.size) return null

    if (spec.operator == FindMusicOperator.REFINE) {
        val primaryIndex = spec.refineSpec?.primaryIngredientIndex ?: return null
        val secondaryIndex = if (primaryIndex == 0) 1 else 0
        val primaryRank = LibraryRankEvidenceText.rankFromUpperCdfPercentile(
            match.anchorPercentiles.getOrNull(primaryIndex) ?: return null,
            ingredientTotal,
        ) ?: return null
        val secondaryRank = LibraryRankEvidenceText.rankFromUpperCdfPercentile(
            match.anchorPercentiles.getOrNull(secondaryIndex) ?: return null,
            ingredientTotal,
        ) ?: return null
        val compactPrimary = LibraryRankEvidenceText.rank(primaryRank) ?: return null
        val compactSecondary = LibraryRankEvidenceText.rank(secondaryRank) ?: return null
        val fullPrimary = LibraryRankEvidenceText.rankWithTopFraction(
            primaryRank,
            ingredientTotal,
        ) ?: return null
        val fullSecondary = LibraryRankEvidenceText.rankWithTopFraction(
            secondaryRank,
            ingredientTotal,
        ) ?: return null
        return FindMusicRowEvidence(
            compact = "Primary match $compactPrimary \u00b7 Secondary match $compactSecondary",
            expanded = "${labels[primaryIndex]} match \u00b7 $fullPrimary \u00b7 " +
                "${labels[secondaryIndex]} match \u00b7 $fullSecondary",
        )
    }

    if (labels.size == 1) {
        val percentile = match.anchorPercentiles.singleOrNull() ?: return null
        val rank = LibraryRankEvidenceText.rankFromUpperCdfPercentile(
            percentile,
            ingredientTotal,
        ) ?: return null
        val compactRank = LibraryRankEvidenceText.rank(rank) ?: return null
        val fullRank = LibraryRankEvidenceText.rankWithTopFraction(rank, ingredientTotal)
            ?: return null
        return FindMusicRowEvidence(
            compact = "$compactRank nearest",
            expanded = "${labels.single()} match \u00b7 $fullRank",
        )
    }

    var compactOverall: String? = null
    val compactIngredientRanks = mutableListOf<String>()
    val expandedParts = mutableListOf<String>()
    match.objectiveRank
        ?.takeIf { rank ->
            result.objectiveRankingDomainCount?.let { rank in 1..it } == true &&
                rank != displayedIndex + 1
        }
        ?.let { rank ->
            val objectiveTotal = checkNotNull(result.objectiveRankingDomainCount)
            LibraryRankEvidenceText.rank(rank)?.let { compactRank ->
                compactOverall = "Overall $compactRank"
            }
            LibraryRankEvidenceText.rankWithTopFraction(rank, objectiveTotal)?.let { fullRank ->
                expandedParts += "Overall match \u00b7 $fullRank"
            }
        }
    match.anchorPercentiles.forEachIndexed { index, percentile ->
        val label = labels[index]
        val rank = LibraryRankEvidenceText.rankFromUpperCdfPercentile(
            percentile,
            ingredientTotal,
        ) ?: return null
        val compactRank = LibraryRankEvidenceText.rank(rank) ?: return null
        val fullRank = LibraryRankEvidenceText.rankWithTopFraction(rank, ingredientTotal)
            ?: return null
        compactIngredientRanks += compactRank
        expandedParts += "$label match \u00b7 $fullRank"
    }
    if (compactIngredientRanks.isEmpty() ||
        compactIngredientRanks.size != match.anchorPercentiles.size
    ) {
        return null
    }
    val compact = listOfNotNull(
        compactOverall,
        "Ingredient ranks \u00b7 ${compactIngredientRanks.joinToString(" / ")}",
    ).joinToString(" \u00b7 ")
    return FindMusicRowEvidence(
        compact = compact,
        expanded = expandedParts.joinToString(" \u00b7 "),
    )
}

@Composable
private fun ExpandableEvidenceLine(evidence: FindMusicRowEvidence) {
    var expanded by remember(evidence) { mutableStateOf(false) }
    var compactOverflows by remember(evidence) { mutableStateOf(false) }
    val canExpand = evidence.expanded != evidence.compact || compactOverflows
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .then(
                if (canExpand) {
                    Modifier.clickable { expanded = !expanded }
                } else {
                    Modifier
                },
            )
            .padding(top = 1.dp),
        verticalAlignment = Alignment.Top,
    ) {
        Text(
            text = if (expanded) evidence.expanded else evidence.compact,
            style = MaterialTheme.typography.labelSmall,
            fontWeight = FontWeight.Normal,
            color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.85f),
            maxLines = if (expanded) 5 else 1,
            overflow = if (expanded) TextOverflow.Clip else TextOverflow.Ellipsis,
            onTextLayout = { result ->
                if (!expanded) compactOverflows = result.hasVisualOverflow
            },
            modifier = Modifier.weight(1f),
        )
        if (canExpand) {
            Text(
                text = if (expanded) "\u25b4" else "\u25be",
                style = MaterialTheme.typography.labelSmall,
                fontWeight = FontWeight.Normal,
                color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.85f),
                modifier = Modifier.padding(start = 6.dp),
            )
        }
    }
}

// ---- Settings Screen ----

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(
    viewModel: MainViewModel,
    databaseInfo: DatabaseInfo?,
    onImportDatabase: () -> Unit,
    onMergeServerDatabase: () -> Unit,
    hasPermission: Boolean,
    onRequestPermission: () -> Unit,
    onBack: () -> Unit
) {
    val selectionMode by viewModel.selectionMode.collectAsState()
    val driftEnabled by viewModel.driftEnabled.collectAsState()
    val driftMode by viewModel.driftMode.collectAsState()
    val anchorStrength by viewModel.anchorStrength.collectAsState()
    val anchorDecay by viewModel.anchorDecay.collectAsState()
    val anchorHalfLifeTracks by viewModel.anchorHalfLifeTracks.collectAsState()
    val walkRestartAlpha by viewModel.walkRestartAlpha.collectAsState()
    val momentumBeta by viewModel.momentumBeta.collectAsState()
    val diversityLambda by viewModel.diversityLambda.collectAsState()
    val mmrCandidatePoolFraction by viewModel.mmrCandidatePoolFraction.collectAsState()
    val dppUsesCertifiedFullDomain by viewModel.dppUsesCertifiedFullDomain.collectAsState()
    val dppFixedCandidatePoolFraction by
        viewModel.dppFixedCandidatePoolFraction.collectAsState()
    val dppQualityExponent by viewModel.dppQualityExponent.collectAsState()
    val shuffleSeed by viewModel.shuffleSeed.collectAsState()
    val artistLimitsEnabled by viewModel.artistLimitsEnabled.collectAsState()
    val maxPerArtist by viewModel.maxPerArtist.collectAsState()
    val minArtistSpacing by viewModel.minArtistSpacing.collectAsState()
    val numTracks by viewModel.numTracks.collectAsState()
    val libraryAddedDays by viewModel.libraryAddedDays.collectAsState()
    val previews by viewModel.previews.collectAsState()
    val libraryControlsBlockedReason by viewModel.libraryControlsBlockedReason.collectAsState()
    val databaseLoading by viewModel.databaseLoading.collectAsState()
    val databaseVerificationStatus by viewModel.databaseVerificationStatus.collectAsState()
    val musicIndexUpdateStatus by viewModel.importStatus.collectAsState()
    val serverMergeProgress by viewModel.serverMergeProgress.collectAsState()
    val musicIndexUpdateError by viewModel.importError.collectAsState()
    val musicIndexUpdateResult by viewModel.musicIndexUpdateResult.collectAsState()
    val modelsLoading by viewModel.modelsLoading.collectAsState()
    val modelsLoadingStatus by viewModel.modelsLoadingStatus.collectAsState()
    val permissionLoading by viewModel.permissionLoading.collectAsState()
    val scope = rememberCoroutineScope()
    var showResetControlsConfirmation by rememberSaveable { mutableStateOf(false) }
    var showFileDetails by rememberSaveable { mutableStateOf(false) }
    var openingOnDeviceIndexing by remember { mutableStateOf(false) }
    var onDeviceIndexingOpenError by rememberSaveable { mutableStateOf<String?>(null) }

    val isGraphExplorer = selectionMode == SelectionMode.RANDOM_WALK
    val supportsDrift = selectionMode == SelectionMode.MMR
    val artistControlState = remember(numTracks, maxPerArtist, minArtistSpacing) {
        ArtistConstraintControlPolicy.forRequest(
            recommendationCount = numTracks,
            maxPerArtist = maxPerArtist,
            minArtistSpacing = minArtistSpacing,
        )
    }
    val eligibleCandidateIdentityCount = databaseInfo?.eligibleCandidateIdentityCount
        ?.takeIf { libraryAddedDays == null }
    val showDppDomainControl = DppDomainControlPolicy.shouldExposeDomainControl(
        eligibleCandidateIdentityCount = eligibleCandidateIdentityCount,
        numTracks = numTracks,
    )
    val effectiveDppUsesCertifiedFullDomain =
        DppDomainControlPolicy.effectiveUsesCertifiedFullDomain(
            storedUsesCertifiedFullDomain = dppUsesCertifiedFullDomain,
            eligibleCandidateIdentityCount = eligibleCandidateIdentityCount,
            numTracks = numTracks,
        )
    val showMmrReach = MmrControlPolicy.reachCanAffectOutput(
        relevanceWeight = diversityLambda,
        artistLimitsEnabled = artistLimitsEnabled,
        recommendationCount = numTracks,
        maxPerArtist = maxPerArtist,
        minArtistSpacing = minArtistSpacing,
    ) && NeighborhoodReachPolicy.hasMultipleDistinctDomains(
        options = NeighborhoodReachPolicy.MMR_OPTIONS,
        eligibleCandidateIdentityCount = eligibleCandidateIdentityCount,
        numTracks = numTracks,
        preferredFraction = mmrCandidatePoolFraction,
    )
    val showDppFixedReach = showDppDomainControl &&
        NeighborhoodReachPolicy.hasMultipleDistinctDomains(
            options = NeighborhoodReachPolicy.DPP_FIXED_OPTIONS,
            eligibleCandidateIdentityCount = eligibleCandidateIdentityCount,
            numTracks = numTracks,
            preferredFraction = dppFixedCandidatePoolFraction,
        )
    val effectiveFadeTiming = remember(anchorDecay, anchorHalfLifeTracks, numTracks) {
        if (anchorDecay == DecaySchedule.STEP) {
            DriftControlPolicy.canonicalFadeTiming(
                decay = anchorDecay,
                timingTracks = anchorHalfLifeTracks,
                recommendationCount = numTracks,
            )
        } else {
            anchorHalfLifeTracks
        }
    }

    // Minimal slider: thin track (color-only position), small circle thumb
    val slimThumb: @Composable (SliderState) -> Unit = {
        Box(
            Modifier
                .size(16.dp)
                .background(MaterialTheme.colorScheme.primary, CircleShape)
        )
    }
    val cleanTrack: @Composable (SliderState) -> Unit = { state ->
        SliderDefaults.Track(
            sliderState = state,
            modifier = Modifier.height(8.dp),
            drawStopIndicator = null,
            thumbTrackGapSize = 0.dp,
            trackInsideCornerSize = 0.dp,
        )
    }
    val steppedTrack: @Composable (SliderState) -> Unit = { state ->
        SliderDefaults.Track(
            sliderState = state,
            modifier = Modifier.height(8.dp),
            drawStopIndicator = null,
            thumbTrackGapSize = 0.dp,
            trackInsideCornerSize = 0.dp,
        )
    }

    // Group keys by what affects each mode
    val commonKeys = remember(
        numTracks,
        libraryAddedDays,
        artistLimitsEnabled,
        maxPerArtist,
        minArtistSpacing,
    ) { Any() }
    val driftKeys = remember(
        driftEnabled,
        driftMode,
        anchorStrength,
        anchorDecay,
        anchorHalfLifeTracks,
        momentumBeta,
    ) { Any() }
    val libraryBindingKeys = remember(
        databaseInfo?.generationId,
        databaseInfo?.providerGenerationId,
        databaseInfo?.activeTrackCount,
        databaseInfo?.eligibleCandidateIdentityCount,
    ) { Any() }
    val expandedPeek = remember { mutableStateMapOf<SelectionMode, Boolean>() }

    // Invalidate stale previews when relevant settings change (lazy - computed on peek click)
    LaunchedEffect(
        commonKeys,
        libraryBindingKeys,
        driftKeys,
        diversityLambda,
        mmrCandidatePoolFraction,
    ) {
        viewModel.invalidatePreview(SelectionMode.MMR)
        expandedPeek[SelectionMode.MMR] = false
    }
    LaunchedEffect(commonKeys, libraryBindingKeys) {
        viewModel.invalidatePreview(SelectionMode.CLOSEST)
        expandedPeek[SelectionMode.CLOSEST] = false
    }
    LaunchedEffect(
        commonKeys,
        libraryBindingKeys,
        dppUsesCertifiedFullDomain,
        dppFixedCandidatePoolFraction,
        dppQualityExponent,
    ) {
        viewModel.invalidatePreview(SelectionMode.DPP)
        expandedPeek[SelectionMode.DPP] = false
    }
    LaunchedEffect(commonKeys, libraryBindingKeys, walkRestartAlpha) {
        viewModel.invalidatePreview(SelectionMode.RANDOM_WALK)
        expandedPeek[SelectionMode.RANDOM_WALK] = false
    }
    LaunchedEffect(commonKeys, libraryBindingKeys, shuffleSeed) {
        viewModel.invalidatePreview(SelectionMode.UNIFORM_SHUFFLE)
        expandedPeek[SelectionMode.UNIFORM_SHUFFLE] = false
    }
    if (showResetControlsConfirmation) {
        AlertDialog(
            onDismissRequest = { showResetControlsConfirmation = false },
            title = { Text("Reset radio and Find Music controls?") },
            text = {
                Text(
                    "Restores selection, journey, artist-credit, queue length, and Find Music " +
                        "selection, candidate dates, and combination controls to their defaults. " +
                        "Search history and the music index stay as they are.",
                )
            },
            dismissButton = {
                TextButton(onClick = { showResetControlsConfirmation = false }) {
                    Text("Keep settings")
                }
            },
            confirmButton = {
                Button(
                    onClick = {
                        viewModel.resetToDefaults()
                        showResetControlsConfirmation = false
                    },
                ) {
                    Text("Reset controls")
                }
            },
        )
    }

    Scaffold(
        contentWindowInsets = WindowInsets.safeDrawing,
        topBar = {
            TopAppBar(
                title = { Text("Settings") },
                windowInsets = WindowInsets.safeDrawing.only(
                    WindowInsetsSides.Horizontal + WindowInsetsSides.Top,
                ),
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back")
                    }
                }
            )
        }
    ) { padding ->
        LazyColumn(
            modifier = Modifier.fillMaxSize().padding(padding).padding(horizontal = 16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            item {
                Text(
                    "Shared controls",
                    style = MaterialTheme.typography.titleMedium,
                    color = MaterialTheme.colorScheme.primary,
                )
            }

            // Recommendation count
            item {
                Column {
                    Text("Queue length: $numTracks", style = MaterialTheme.typography.titleMedium)
                    Text(
                        "Used for new by-seed radio queues and Find Music result lists.",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                    Slider(
                        value = numTracks.toFloat(),
                        onValueChange = { viewModel.setNumTracks(it.roundToInt()) },
                        valueRange = 10f..100f,
                        steps = 8,
                        modifier = Modifier.semantics {
                            stateDescription = "$numTracks tracks per new queue"
                        },
                        thumb = slimThumb,
                        track = steppedTrack,
                    )
                }
            }

            item { HorizontalDivider() }

            item {
                Column {
                    LibraryAddedDaysControl(
                        dayCount = libraryAddedDays,
                        enabled = !openingOnDeviceIndexing,
                        onApply = viewModel::setLibraryAddedDays,
                        titleStyle = MaterialTheme.typography.titleMedium,
                    )
                    Text(
                        "Radio and Find Music use only recordings in this window. Poweramp's " +
                            "first-added time determines age; days are rolling 24-hour periods.",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }

            item { HorizontalDivider() }

            item {
                Text(
                    "Radio from Now Playing",
                    style = MaterialTheme.typography.titleMedium,
                    color = MaterialTheme.colorScheme.primary,
                )
            }

            // Selection algorithm
            item {
                Text("Selection mode", style = MaterialTheme.typography.titleMedium)
                Spacer(modifier = Modifier.height(4.dp))

                Column(modifier = Modifier.selectableGroup()) {
                    AlgorithmOption(
                        label = SelectionControlText.modeLabel(SelectionMode.CLOSEST),
                        summary = SelectionControlText.modeDifferentiator(SelectionMode.CLOSEST),
                        selected = selectionMode == SelectionMode.CLOSEST,
                        enabled = !openingOnDeviceIndexing,
                        onClick = { viewModel.setSelectionMode(SelectionMode.CLOSEST) }
                    )
                    AlgorithmOption(
                        label = SelectionControlText.modeLabel(SelectionMode.MMR),
                        summary = SelectionControlText.modeDifferentiator(SelectionMode.MMR),
                        selected = selectionMode == SelectionMode.MMR,
                        enabled = !openingOnDeviceIndexing,
                        onClick = { viewModel.setSelectionMode(SelectionMode.MMR) }
                    )
                    AlgorithmOption(
                        label = SelectionControlText.modeLabel(SelectionMode.DPP),
                        summary = SelectionControlText.modeDifferentiator(SelectionMode.DPP),
                        selected = selectionMode == SelectionMode.DPP,
                        enabled = !openingOnDeviceIndexing,
                        onClick = { viewModel.setSelectionMode(SelectionMode.DPP) }
                    )
                    AlgorithmOption(
                        label = SelectionControlText.modeLabel(SelectionMode.RANDOM_WALK),
                        summary = SelectionControlText.modeDifferentiator(
                            SelectionMode.RANDOM_WALK,
                        ),
                        selected = selectionMode == SelectionMode.RANDOM_WALK,
                        availabilityNote = if (databaseInfo?.hasGraph == true) null else "Graph required",
                        enabled = databaseInfo?.hasGraph == true && !openingOnDeviceIndexing,
                        onClick = { viewModel.setSelectionMode(SelectionMode.RANDOM_WALK) }
                    )
                    AlgorithmOption(
                        label = SelectionControlText.modeLabel(SelectionMode.UNIFORM_SHUFFLE),
                        summary = SelectionControlText.modeDifferentiator(
                            SelectionMode.UNIFORM_SHUFFLE,
                        ),
                        selected = selectionMode == SelectionMode.UNIFORM_SHUFFLE,
                        enabled = !openingOnDeviceIndexing,
                        onClick = { viewModel.setSelectionMode(SelectionMode.UNIFORM_SHUFFLE) },
                    )
                }

                HorizontalDivider(modifier = Modifier.padding(top = 4.dp, bottom = 10.dp))
                Text(
                    SelectionControlText.modeDescription(selectionMode, driftEnabled),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }

            if (selectionMode == SelectionMode.UNIFORM_SHUFFLE) {
                item {
                    Column {
                        Text("Shuffle order", style = MaterialTheme.typography.titleSmall)
                        Text(
                            "Saved sessions retain their exact generated order.",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                        FilledTonalButton(onClick = viewModel::advanceShuffleSeed) {
                            Icon(Icons.Default.Refresh, contentDescription = null)
                            Spacer(Modifier.width(8.dp))
                            Text("New order")
                        }
                    }
                }
            }

            // Similarity vs. Variety - only for MMR
            if (selectionMode == SelectionMode.MMR) {
                item {
                    Column {
                        Text(
                            SelectionControlText.mmrBalanceTitle(
                                relevanceWeight = diversityLambda,
                                driftEnabled = driftEnabled,
                            ),
                            style = MaterialTheme.typography.titleSmall)
                        Text(
                            if (diversityLambda == 0f) {
                                "The nearest candidate wins the first stable tie. Later picks " +
                                    "minimize resemblance to earlier picks."
                            } else {
                                "Higher relevance stays closer to " +
                                    (if (driftEnabled) {
                                        "the current direction. "
                                    } else {
                                        "the seed. "
                                    }) +
                                    "Higher variety avoids tracks that resemble earlier picks."
                            },
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant)
                        Slider(
                            value = SelectionKnobPolicy.nearestIndex(
                                SelectionKnobPolicy.MMR_RELEVANCE_OPTIONS,
                                diversityLambda,
                            ).toFloat(),
                            onValueChange = { value ->
                                viewModel.setDiversityLambda(
                                    SelectionKnobPolicy.MMR_RELEVANCE_OPTIONS[
                                        value.roundToInt().coerceIn(
                                            SelectionKnobPolicy.MMR_RELEVANCE_OPTIONS.indices,
                                        )
                                    ],
                                )
                            },
                            valueRange = 0f..SelectionKnobPolicy.MMR_RELEVANCE_OPTIONS.lastIndex.toFloat(),
                            steps = SelectionKnobPolicy.MMR_RELEVANCE_OPTIONS.size - 2,
                            modifier = Modifier.fillMaxWidth().semantics {
                                stateDescription = SelectionControlText.mmrBalanceTitle(
                                    relevanceWeight = diversityLambda,
                                    driftEnabled = driftEnabled,
                                )
                            },
                            thumb = slimThumb,
                            track = cleanTrack,
                        )
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                        ) {
                            Text(
                                "Variety after first",
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f),
                            )
                            Text(
                                if (driftEnabled) {
                                    "Current-direction relevance only"
                                } else {
                                    "Seed relevance only"
                                },
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f),
                            )
                        }
                    }
                }
            }

            if (selectionMode == SelectionMode.MMR && showMmrReach) {
                item {
                    NeighborhoodReachControl(
                        fraction = mmrCandidatePoolFraction,
                        librarySize = eligibleCandidateIdentityCount,
                        numTracks = numTracks,
                        options = NeighborhoodReachPolicy.MMR_OPTIONS,
                        recommendedFraction = RadioConfig.DEFAULT_MMR_CANDIDATE_POOL_FRACTION,
                        title = "Selection pool",
                        description = if (driftEnabled) {
                            "Recomputed around the evolving queue direction after each pick."
                        } else {
                            "Limits MMR to this nearest share of eligible recordings."
                        },
                        onFractionChange = viewModel::setMmrCandidatePoolFraction,
                    )
                }
            }

            if (selectionMode == SelectionMode.DPP) {
                if (showDppDomainControl) {
                    item {
                        Column {
                            Text(
                                "Selection pool",
                                style = MaterialTheme.typography.titleSmall,
                            )
                            SingleChoiceSegmentedButtonRow(modifier = Modifier.fillMaxWidth()) {
                                SegmentedButton(
                                    selected = dppUsesCertifiedFullDomain,
                                    onClick = { viewModel.setDppUsesCertifiedFullDomain(true) },
                                    shape = SegmentedButtonDefaults.itemShape(index = 0, count = 2),
                                ) {
                                    Text("All eligible")
                                }
                                SegmentedButton(
                                    selected = !dppUsesCertifiedFullDomain,
                                    onClick = { viewModel.setDppUsesCertifiedFullDomain(false) },
                                    shape = SegmentedButtonDefaults.itemShape(index = 1, count = 2),
                                ) {
                                    Text("Nearest subset")
                                }
                            }
                            Text(
                                "All eligible runs DPP across the complete eligible domain. " +
                                    "Nearest subset limits it to a seed-nearest pool.",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        }
                    }
                }
                if (!effectiveDppUsesCertifiedFullDomain && showDppFixedReach) {
                    item {
                        NeighborhoodReachControl(
                            fraction = dppFixedCandidatePoolFraction,
                            librarySize = eligibleCandidateIdentityCount,
                            numTracks = numTracks,
                            options = NeighborhoodReachPolicy.DPP_FIXED_OPTIONS,
                            title = "Subset size",
                            description = "How many seed-nearest eligible recordings are given " +
                                "to DPP.",
                            onFractionChange = viewModel::setDppFixedCandidatePoolFraction,
                        )
                    }
                }
                item {
                    Column {
                        SettingsTitleValueRow(
                            title = "Seed pull",
                            value = SelectionControlText.dppSeedPullLabel(dppQualityExponent),
                        )
                        Text(
                            if (dppQualityExponent == 0f) {
                                "All candidates have equal quality. The nearest wins the first " +
                                    "stable tie; later picks maximize determinant gain for set variety."
                            } else {
                                "Weights seed similarity in the DPP quality term. Lower values give " +
                                    "the determinant's set diversity more influence."
                            },
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Text("Set variety", style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f))
                            Slider(
                                value = SelectionKnobPolicy.nearestIndex(
                                    SelectionKnobPolicy.DPP_SEED_PULL_OPTIONS,
                                    dppQualityExponent,
                                ).toFloat(),
                                onValueChange = { value ->
                                    viewModel.setDppQualityExponent(
                                        SelectionKnobPolicy.DPP_SEED_PULL_OPTIONS[
                                            value.roundToInt().coerceIn(
                                                SelectionKnobPolicy.DPP_SEED_PULL_OPTIONS.indices,
                                            )
                                        ],
                                    )
                                },
                                valueRange = 0f..SelectionKnobPolicy.DPP_SEED_PULL_OPTIONS.lastIndex.toFloat(),
                                steps = SelectionKnobPolicy.DPP_SEED_PULL_OPTIONS.size - 2,
                                modifier = Modifier.weight(1f).semantics {
                                    stateDescription =
                                        SelectionControlText.dppSeedPullLabel(dppQualityExponent)
                                },
                                thumb = slimThumb,
                                track = steppedTrack,
                            )
                            Text("Seed pull", style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f))
                        }
                    }
                }
            }

            // Stop chance - only for exact Graph Explorer propagation.
            if (isGraphExplorer) {
                item {
                    Column {
                        Text(
                            "Typical path: about ${(1f / walkRestartAlpha).roundToInt()} " +
                                "track-to-track moves",
                            style = MaterialTheme.typography.titleSmall)
                        Text(
                            "${(walkRestartAlpha * 100).roundToInt()}% chance to stop after each " +
                                "move; dead ends may stop sooner.",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant)
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Text("Longer paths", style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f))
                            Slider(
                                value = SelectionKnobPolicy.nearestIndex(
                                    SelectionKnobPolicy.GRAPH_STOP_OPTIONS,
                                    walkRestartAlpha,
                                ).toFloat(),
                                onValueChange = { value ->
                                    viewModel.setWalkRestartAlpha(
                                        SelectionKnobPolicy.GRAPH_STOP_OPTIONS[
                                            value.roundToInt().coerceIn(
                                                SelectionKnobPolicy.GRAPH_STOP_OPTIONS.indices,
                                            )
                                        ],
                                    )
                                },
                                valueRange = 0f..SelectionKnobPolicy.GRAPH_STOP_OPTIONS.lastIndex.toFloat(),
                                steps = SelectionKnobPolicy.GRAPH_STOP_OPTIONS.size - 2,
                                modifier = Modifier.weight(1f).semantics {
                                    stateDescription =
                                        "Typical path about " +
                                            "${(1f / walkRestartAlpha).roundToInt()} " +
                                            "track-to-track moves; " +
                                            "${(walkRestartAlpha * 100).roundToInt()}% stop chance " +
                                            "after each move"
                                },
                                thumb = slimThumb,
                                track = steppedTrack,
                            )
                            Text("Shorter paths", style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f))
                        }
                    }
                }
            }

            // Drift is an evolving-query variant of MMR, not a property of set selectors.
            if (supportsDrift) {
                item { HorizontalDivider() }

                item {
                    Column {
                        Text("Queue direction", style = MaterialTheme.typography.titleMedium)
                        Spacer(modifier = Modifier.height(4.dp))

                        Row(modifier = Modifier.fillMaxWidth().selectable(
                            selected = driftEnabled, onClick = {
                                val enabling = !driftEnabled
                                viewModel.setDriftEnabled(enabling)
                            }, role = Role.Checkbox
                        ).padding(vertical = 4.dp), verticalAlignment = Alignment.CenterVertically) {
                            Checkbox(checked = driftEnabled, onCheckedChange = null)
                            Spacer(modifier = Modifier.width(8.dp))
                            Column {
                                Text("Evolve after each pick", style = MaterialTheme.typography.bodyMedium)
                                Text("Updates MMR's relevance direction after each pick, allowing the selected sequence to steer later choices.",
                                    style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                            }
                        }

                        AnimatedVisibility(visible = driftEnabled) {
                            Column(modifier = Modifier.padding(start = 16.dp, top = 8.dp)) {
                                Text("Direction update", style = MaterialTheme.typography.titleSmall)
                                Column(modifier = Modifier.selectableGroup()) {
                                    AlgorithmOption(
                                        label = SelectionControlText.driftModeLabel(
                                            DriftMode.SEED_INTERPOLATION,
                                        ),
                                        summary = "Blend the original seed with the last pick",
                                        selected = driftMode == DriftMode.SEED_INTERPOLATION,
                                        onClick = { viewModel.setDriftMode(DriftMode.SEED_INTERPOLATION) },
                                    )
                                    AlgorithmOption(
                                        label = SelectionControlText.driftModeLabel(
                                            DriftMode.MOMENTUM,
                                        ),
                                        summary = "Blend the prior direction with the last pick",
                                        selected = driftMode == DriftMode.MOMENTUM,
                                        onClick = { viewModel.setDriftMode(DriftMode.MOMENTUM) },
                                    )
                                }

                                Spacer(modifier = Modifier.height(4.dp))

                                if (driftMode == DriftMode.SEED_INTERPOLATION) {
                                    val seedPullOptions = DriftControlPolicy.seedPullOptions(anchorDecay)
                                    val seedPullIndex = SelectionKnobPolicy.nearestIndex(
                                        seedPullOptions,
                                        anchorStrength,
                                    )
                                    val seedPullDescription = if (anchorStrength <= 0f) {
                                        "Trajectory: Follow last pick"
                                    } else {
                                        "Starting seed pull: ${(anchorStrength * 100).roundToInt()}%"
                                    }
                                    Text(
                                        seedPullDescription,
                                        style = MaterialTheme.typography.titleSmall,
                                    )
                                    Text(
                                        if (anchorStrength <= 0f) {
                                            "Each next search uses the last picked track as its query."
                                        } else {
                                            "After the first recommendation, each next pick balances " +
                                                "the original seed with the last selected track."
                                        },
                                        style = MaterialTheme.typography.bodySmall,
                                        color = MaterialTheme.colorScheme.onSurfaceVariant)
                                    Row(verticalAlignment = Alignment.CenterVertically) {
                                        Text("Last pick", style = MaterialTheme.typography.labelSmall,
                                            color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f))
                                        Slider(
                                            value = seedPullIndex.toFloat(),
                                            onValueChange = { value ->
                                                viewModel.setAnchorStrength(
                                                    seedPullOptions[
                                                        value.roundToInt().coerceIn(seedPullOptions.indices)
                                                    ],
                                                )
                                            },
                                            valueRange = 0f..seedPullOptions.lastIndex.toFloat(),
                                            steps = seedPullOptions.size - 2,
                                            modifier = Modifier.weight(1f).semantics {
                                                stateDescription = seedPullDescription
                                            },
                                            thumb = slimThumb,
                                            track = steppedTrack,
                                        )
                                        Text("Seed", style = MaterialTheme.typography.labelSmall,
                                            color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f))
                                    }

                                    if (DriftControlPolicy.seedFadeApplies(anchorStrength)) {
                                        Spacer(modifier = Modifier.height(8.dp))
                                        Text("Seed-pull fade", style = MaterialTheme.typography.titleSmall)
                                        SingleChoiceSegmentedButtonRow(modifier = Modifier.fillMaxWidth()) {
                                            val schedules = DriftControlPolicy.decaySchedules(anchorStrength)
                                            schedules.forEachIndexed { index, schedule ->
                                                val label = when (schedule) {
                                                    DecaySchedule.NONE -> "Hold"
                                                    DecaySchedule.LINEAR -> "Linear"
                                                    DecaySchedule.EXPONENTIAL -> "Exponential"
                                                    DecaySchedule.STEP -> "Step"
                                                }
                                                SegmentedButton(
                                                    selected = anchorDecay == schedule,
                                                    onClick = { viewModel.setAnchorDecay(schedule) },
                                                    shape = SegmentedButtonDefaults.itemShape(
                                                        index = index,
                                                        count = schedules.size,
                                                    ),
                                                ) {
                                                    Text(label, maxLines = 1, fontSize = 11.sp)
                                                }
                                            }
                                        }
                                        Text(
                                            when (anchorDecay) {
                                                DecaySchedule.NONE ->
                                                    "Hold the starting seed pull for the full queue."
                                                DecaySchedule.LINEAR ->
                                                    "Reduce seed pull steadily toward zero."
                                                DecaySchedule.EXPONENTIAL ->
                                                    "Reduce seed pull continuously by a fixed half-life."
                                                DecaySchedule.STEP ->
                                                    "Hold seed pull, then drop it once."
                                            },
                                            style = MaterialTheme.typography.bodySmall,
                                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                                        )

                                        if (anchorDecay != DecaySchedule.NONE) {
                                            val timingOptions = DriftControlPolicy.fadeTimingOptions(
                                                decay = anchorDecay,
                                                recommendationCount = numTracks,
                                            )
                                            val timingIndex = SelectionKnobPolicy.nearestIndex(
                                                timingOptions,
                                                effectiveFadeTiming,
                                            )
                                            val timingLabel = when (anchorDecay) {
                                                DecaySchedule.LINEAR -> "Half-strength point"
                                                DecaySchedule.EXPONENTIAL -> "Half-life"
                                                DecaySchedule.STEP -> "Drop point"
                                                DecaySchedule.NONE -> ""
                                            }
                                            val displayedTiming = if (anchorDecay == DecaySchedule.STEP) {
                                                DriftControlPolicy.stepDropAfterPickCount(
                                                    effectiveFadeTiming,
                                                )
                                            } else {
                                                effectiveFadeTiming
                                            }
                                            val timingDescription = if (
                                                anchorDecay == DecaySchedule.STEP
                                            ) {
                                                "$timingLabel: after $displayedTiming picks"
                                            } else {
                                                "$timingLabel: ${formatKnob(effectiveFadeTiming)} picks"
                                            }
                                            Text(
                                                timingDescription,
                                                style = MaterialTheme.typography.titleSmall,
                                            )
                                            Text(
                                                when (anchorDecay) {
                                                    DecaySchedule.LINEAR -> "Seed pull reaches half its starting value here and zero after twice this many picks."
                                                    DecaySchedule.EXPONENTIAL -> "Seed pull halves after this many picks, independent of requested queue length."
                                                    DecaySchedule.STEP -> "Seed pull stays steady for this many picks, then drops to one fifth for the remaining queue."
                                                    DecaySchedule.NONE -> ""
                                                },
                                                style = MaterialTheme.typography.bodySmall,
                                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                                            )
                                            Slider(
                                                value = timingIndex.toFloat(),
                                                onValueChange = { value ->
                                                    viewModel.setAnchorHalfLifeTracks(
                                                        timingOptions[
                                                            value.roundToInt().coerceIn(timingOptions.indices)
                                                        ],
                                                    )
                                                },
                                                valueRange = 0f..timingOptions.lastIndex.toFloat(),
                                                steps = timingOptions.size - 2,
                                                modifier = Modifier.semantics {
                                                    stateDescription = timingDescription
                                                },
                                                thumb = slimThumb,
                                                track = steppedTrack,
                                            )
                                        }
                                    }
                                } else {
                                    val momentumOptions = DriftControlPolicy.MOMENTUM_OPTIONS
                                    val momentumIndex = SelectionKnobPolicy.nearestIndex(
                                        momentumOptions,
                                        momentumBeta,
                                    )
                                    Text("Prior-direction memory: ${(momentumBeta * 100).roundToInt()}%",
                                        style = MaterialTheme.typography.titleSmall)
                                    Text("Higher memory preserves more of the evolving queue direction; " +
                                        "lower memory follows the latest pick more strongly.",
                                        style = MaterialTheme.typography.bodySmall,
                                        color = MaterialTheme.colorScheme.onSurfaceVariant)
                                    Row(verticalAlignment = Alignment.CenterVertically) {
                                        Text("Follow latest pick", style = MaterialTheme.typography.labelSmall,
                                            color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f))
                                        Slider(
                                            value = momentumIndex.toFloat(),
                                            onValueChange = { value ->
                                                viewModel.setMomentumBeta(
                                                    momentumOptions[
                                                        value.roundToInt().coerceIn(momentumOptions.indices)
                                                    ],
                                                )
                                            },
                                            valueRange = 0f..momentumOptions.lastIndex.toFloat(),
                                            steps = momentumOptions.size - 2,
                                            modifier = Modifier.weight(1f).semantics {
                                                stateDescription = "Prior-direction memory: " +
                                                    "${(momentumBeta * 100).roundToInt()}%"
                                            },
                                            thumb = slimThumb,
                                            track = steppedTrack,
                                        )
                                        Text("Keep prior direction", style = MaterialTheme.typography.labelSmall,
                                            color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f))
                                    }
                                }
                            }
                        }
                    }
                }
            }

            item {
                SettingsAlgorithmPreview(
                    label = SelectionControlText.modeLabel(selectionMode),
                    previewTrackCount = numTracks,
                    preview = previews[selectionMode],
                    expanded = expandedPeek[selectionMode] == true,
                    onToggleExpanded = {
                        val expanded = expandedPeek[selectionMode] == true
                        if (SettingsPeekInteractionPolicy.requestsFreshPlan(
                                expanded,
                                previews[selectionMode],
                            )
                        ) {
                            viewModel.computePreview(selectionMode)
                        }
                        expandedPeek[selectionMode] = !expanded
                    },
                    enabled = !openingOnDeviceIndexing &&
                        (selectionMode != SelectionMode.RANDOM_WALK ||
                            databaseInfo?.hasGraph == true),
                )
            }

            item { HorizontalDivider() }

            // Artist constraints
            item {
                Row(
                    modifier = Modifier.fillMaxWidth().clickable(
                        onClick = { viewModel.setArtistLimitsEnabled(!artistLimitsEnabled) },
                        role = Role.Switch,
                    ).padding(vertical = 4.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Column(modifier = Modifier.weight(1f)) {
                        Text("Artist-credit limits", style = MaterialTheme.typography.titleMedium)
                        Text(
                            "Cap appearances and require space before the same credited artist " +
                                "can recur.",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                    Spacer(modifier = Modifier.width(12.dp))
                    Switch(
                        checked = artistLimitsEnabled,
                        onCheckedChange = null,
                    )
                }

                AnimatedVisibility(visible = artistLimitsEnabled) {
                    Column(modifier = Modifier.padding(top = 8.dp)) {
                        if (artistControlState.showMaximum) {
                            val options = artistControlState.maximumOptions
                            val selected = options.indexOf(maxPerArtist).coerceAtLeast(0)
                            Text("Maximum per artist credit: $maxPerArtist", style = MaterialTheme.typography.titleSmall)
                            Slider(
                                value = selected.toFloat(),
                                onValueChange = { value ->
                                    viewModel.setMaxPerArtist(
                                        options[value.roundToInt().coerceIn(options.indices)],
                                    )
                                },
                                valueRange = 0f..options.lastIndex.toFloat(),
                                steps = options.size - 2,
                                modifier = Modifier.semantics {
                                    stateDescription =
                                        "Maximum per artist credit: $maxPerArtist tracks"
                                },
                                thumb = slimThumb,
                                track = steppedTrack,
                            )
                        }

                        if (artistControlState.showSpacing) {
                            val options = artistControlState.spacingOptions
                            val selected = options.indexOf(minArtistSpacing).coerceAtLeast(0)
                            Text("Tracks between the same artist credit: $minArtistSpacing", style = MaterialTheme.typography.titleSmall)
                            Slider(
                                value = selected.toFloat(),
                                onValueChange = { value ->
                                    viewModel.setMinArtistSpacing(
                                        options[value.roundToInt().coerceIn(options.indices)],
                                    )
                                },
                                valueRange = 0f..options.lastIndex.toFloat(),
                                steps = options.size - 2,
                                modifier = Modifier.semantics {
                                    stateDescription = if (minArtistSpacing == 0) {
                                        "No spacing limit for the same artist credit"
                                    } else {
                                        "$minArtistSpacing tracks between the same artist credit"
                                    }
                                },
                                thumb = slimThumb,
                                track = steppedTrack,
                            )
                        }

                        artistControlState.evidenceLine?.let { evidence ->
                            Text(
                                evidence,
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        }
                    }
                }
            }

            item { HorizontalDivider() }

            // Database section
            item {
                Column {
                    Text("Music index", style = MaterialTheme.typography.titleMedium)
                    Spacer(modifier = Modifier.height(8.dp))
                    if (databaseInfo != null) {
                        val storedCount = databaseInfo.trackCount
                        val activeTrackCount = databaseInfo.activeTrackCount
                        if (activeTrackCount != null) {
                            Text(
                                "${"%,d".format(activeTrackCount)} tracks available for recommendations",
                                style = MaterialTheme.typography.bodyMedium,
                            )
                            (storedCount - activeTrackCount).takeIf { it > 0 }?.let { count ->
                                Text(
                                    "${"%,d".format(storedCount)} indexed \u00b7 " +
                                        "${"%,d".format(count)} unavailable in " +
                                        "the current Poweramp library",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                                )
                            }
                        } else {
                            Text(
                                "${"%,d".format(storedCount)} tracks in music index",
                                style = MaterialTheme.typography.bodyMedium,
                            )
                        }
                    } else if (databaseLoading) {
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                        ) {
                            CircularProgressIndicator(
                                modifier = Modifier.size(18.dp),
                                strokeWidth = 2.dp,
                            )
                            Text(
                                databaseVerificationStatus
                                    ?: "Opening the saved music index and reading track counts",
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        }
                    } else {
                        Text("No music index is loaded.", color = MaterialTheme.colorScheme.error)
                    }
                    Spacer(modifier = Modifier.height(8.dp))
                    if (musicIndexUpdateStatus != null) {
                        val completed = serverMergeProgress?.completedUnits
                        val total = serverMergeProgress?.totalUnits
                        val progress = if (
                            completed != null &&
                            total != null &&
                            total > 0L &&
                            completed in 0L..total
                        ) {
                            completed.toFloat() / total.toFloat()
                        } else {
                            null
                        }
                        Column(
                            modifier = Modifier.fillMaxWidth().padding(vertical = 8.dp),
                        ) {
                            Row(
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.spacedBy(12.dp),
                            ) {
                                if (progress == null) {
                                    CircularProgressIndicator(
                                        modifier = Modifier.size(20.dp),
                                        strokeWidth = 2.dp,
                                    )
                                }
                                Text(
                                    musicIndexUpdateStatus!!,
                                    style = MaterialTheme.typography.bodyMedium,
                                )
                            }
                            progress?.let { fraction ->
                                Spacer(modifier = Modifier.height(8.dp))
                                LinearProgressIndicator(
                                    progress = { fraction },
                                    modifier = Modifier.fillMaxWidth(),
                                )
                            }
                        }
                    } else if (databaseInfo == null && !databaseLoading) {
                        OutlinedButton(
                            onClick = onImportDatabase,
                            enabled = libraryControlsBlockedReason == null,
                            modifier = Modifier.fillMaxWidth(),
                        ) {
                            Text("Import a music index")
                        }
                    } else if (databaseInfo != null && !databaseLoading) {
                        Text(
                            "Adds compatible missing embeddings from a server export after " +
                                "matching its tracks to the current Poweramp library.",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                        Spacer(modifier = Modifier.height(6.dp))
                        OutlinedButton(
                            onClick = onMergeServerDatabase,
                            enabled = libraryControlsBlockedReason == null,
                            modifier = Modifier.fillMaxWidth(),
                        ) {
                            Icon(
                                imageVector = Icons.Default.Add,
                                contentDescription = null,
                            )
                            Spacer(modifier = Modifier.width(8.dp))
                            Text("Merge server index")
                        }
                    }
                    musicIndexUpdateResult?.let { result ->
                        Text(
                            result,
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                    musicIndexUpdateError?.let { error ->
                        Text(
                            error,
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.error,
                        )
                    }
                    if (libraryControlsBlockedReason != null && musicIndexUpdateStatus == null) {
                        Text(
                            libraryControlsBlockedReason!!,
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                }
            }

            // Listener readiness first; exact file and model diagnostics are disclosed on demand.
            item {
                val fileStatuses by viewModel.fileStatuses.collectAsState()
                if (modelsLoading || fileStatuses.isNotEmpty()) {
                    val summary = remember(fileStatuses) {
                        SettingsAppFilesText.summarize(fileStatuses)
                    }
                    val missingCapabilities = remember(summary) {
                        summary.capabilityReadiness.filterNot { it.endsWith(": ready.") }
                    }
                    Column {
                        SettingsTitleValueRow(
                            title = "Index and model files",
                            value = if (modelsLoading) {
                                "Reading identities"
                            } else if (missingCapabilities.isEmpty()) {
                                "Ready"
                            } else {
                                "${missingCapabilities.size} unavailable"
                            },
                        )
                        Spacer(modifier = Modifier.height(4.dp))
                        if (modelsLoading) {
                            Row(
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.spacedBy(8.dp),
                            ) {
                                CircularProgressIndicator(
                                    modifier = Modifier.size(16.dp),
                                    strokeWidth = 2.dp,
                                )
                                Text(
                                    modelsLoadingStatus
                                        ?: "Reading indexing model identities",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                                )
                            }
                        }
                        if (!modelsLoading && missingCapabilities.isNotEmpty()) {
                            missingCapabilities.forEach { capabilityStatus ->
                                Text(
                                    capabilityStatus,
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                                )
                            }
                        }
                        if (fileStatuses.isNotEmpty()) {
                            TextButton(onClick = { showFileDetails = !showFileDetails }) {
                                Text(if (showFileDetails) "Hide file details" else "Show file details")
                            }
                            AnimatedVisibility(showFileDetails) {
                                Column {
                                    for (file in fileStatuses) {
                                        FileStatusRow(file)
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // On-device indexing section
            if (databaseInfo != null) {
                item {
                    val context = LocalContext.current
                    val unindexedCount by viewModel.unindexedCount.collectAsState()
                    val checkStatus by viewModel.unindexedCheckStatus.collectAsState()
                    val hasModels by viewModel.hasModels.collectAsState()
                    val indexingState by viewModel.indexingState.collectAsState()
                    val isChecking = unindexedCount == -1

                    Column {
                        Text("On-device indexing", style = MaterialTheme.typography.titleMedium)
                        Spacer(modifier = Modifier.height(4.dp))

                        if (!modelsLoading && !hasModels &&
                            indexingState is IndexingService.IndexingState.Idle
                        ) {
                            Text(
                                "On-device indexing files are missing or do not match this library. " +
                                    "Review file details above.",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        } else {
                            // Show brief status summary
                            when (val state = indexingState) {
                                is IndexingService.IndexingState.Idle -> {
                                    Row(verticalAlignment = Alignment.CenterVertically) {
                                        if (isChecking) {
                                            CircularProgressIndicator(modifier = Modifier.size(14.dp), strokeWidth = 2.dp)
                                            Spacer(modifier = Modifier.width(8.dp))
                                            Text(
                                                checkStatus
                                                    ?: "Reading Poweramp and indexed source spans",
                                                style = MaterialTheme.typography.bodySmall,
                                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                                                modifier = Modifier.weight(1f))
                                        } else {
                                            Text(
                                                when {
                                                    unindexedCount == -3 ->
                                                        "Poweramp-to-index comparison failed"
                                                    unindexedCount == -2 ->
                                                        "Library has not been compared yet"
                                                    unindexedCount > 0 ->
                                                        "Last comparison: " +
                                                            "${formatIndexingTrackCount(unindexedCount)} " +
                                                            if (unindexedCount == 1) {
                                                                "track ready to index"
                                                            } else {
                                                                "tracks ready to index"
                                                            }
                                                    else ->
                                                        "Last comparison: no tracks ready to index"
                                                },
                                                style = if (unindexedCount > 0) MaterialTheme.typography.bodyMedium
                                                    else MaterialTheme.typography.bodySmall,
                                                color = if (unindexedCount > 0) MaterialTheme.colorScheme.primary
                                                    else MaterialTheme.colorScheme.onSurfaceVariant,
                                                modifier = Modifier.weight(1f),
                                            )
                                            IconButton(
                                                onClick = { viewModel.checkUnindexedTracks() },
                                                modifier = Modifier.size(48.dp),
                                            ) {
                                                Icon(Icons.Default.Refresh,
                                                    contentDescription =
                                                        "Compare Poweramp with music index",
                                                    modifier = Modifier.size(18.dp),
                                                    tint = MaterialTheme.colorScheme.onSurfaceVariant)
                                            }
                                        }
                                    }
                                }
                                is IndexingService.IndexingState.JobSnapshot -> {
                                    val stageProgressActive = state.jobState in setOf(
                                        IndexingJobState.RUNNING,
                                        IndexingJobState.PAUSE_REQUESTED,
                                        IndexingJobState.ACTIVATING,
                                        IndexingJobState.CANCELLING,
                                    )
                                    val stageEvidence = state.event
                                        ?.takeIf {
                                            stageProgressActive &&
                                                shouldShowIndexingStageEvent(it, state.progress)
                                        }
                                        ?.let(::indexingStageEvidence)
                                    val stageFallbackText = indexingStageFallbackText(
                                        state = state.jobState,
                                        progress = state.progress,
                                        hasVisibleStageEvent = stageEvidence != null,
                                    )
                                    val terminal = state.jobState in setOf(
                                        IndexingJobState.COMPLETE,
                                        IndexingJobState.CANCELLED,
                                    )
                                    val durableStageCounts =
                                        formatDurableStageTrackCounts(
                                            state.jobState,
                                            state.progress,
                                        )
                                    Text(
                                        when (state.jobState) {
                                            IndexingJobState.PLANNED -> "Ready to begin"
                                            IndexingJobState.RUNNING ->
                                                stageEvidence?.text
                                                    ?: requireNotNull(stageFallbackText)
                                            IndexingJobState.PAUSE_REQUESTED ->
                                                stageEvidence?.text
                                                    ?: requireNotNull(stageFallbackText)
                                            IndexingJobState.COMPLETE ->
                                                formatDurableTrackCounts(
                                                    state.jobState,
                                                    state.progress,
                                                )
                                            IndexingJobState.CANCELLED ->
                                                "Cancelled after ${state.progress.resolvedTracks} of " +
                                                    "${state.progress.totalTracks} tracks"
                                            IndexingJobState.PAUSED -> "Paused; progress is saved"
                                            IndexingJobState.WAITING_FOR_INPUT ->
                                                "${state.progress.blockedTracks} tracks need attention"
                                            IndexingJobState.INTERRUPTED ->
                                                "Interrupted; progress is saved"
                                            IndexingJobState.READY_TO_RESUME -> "Ready to resume"
                                            IndexingJobState.CANCELLING ->
                                                stageEvidence?.text
                                                    ?: requireNotNull(stageFallbackText)
                                            IndexingJobState.ACTIVATING ->
                                                stageEvidence?.text
                                                    ?: requireNotNull(stageFallbackText)
                                        },
                                        style = MaterialTheme.typography.bodyMedium,
                                        color = when (state.jobState) {
                                            IndexingJobState.COMPLETE -> MaterialTheme.colorScheme.primary
                                            IndexingJobState.WAITING_FOR_INPUT -> MaterialTheme.colorScheme.error
                                            else -> MaterialTheme.colorScheme.onSurface
                                        },
                                    )
                                    if (stageProgressActive) {
                                        Spacer(modifier = Modifier.height(4.dp))
                                        val progressFraction = stageEvidence?.fraction
                                        if (progressFraction != null) {
                                            LinearProgressIndicator(
                                                progress = { progressFraction.coerceIn(0f, 1f) },
                                                modifier = Modifier.fillMaxWidth(),
                                            )
                                        } else {
                                            LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                                        }
                                        Spacer(modifier = Modifier.height(4.dp))
                                    }
                                    durableStageCounts?.let { stageCounts ->
                                        Text(
                                            stageCounts,
                                            style = MaterialTheme.typography.bodySmall,
                                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                                        )
                                    }
                                    val showSavedOutcome = !terminal && (
                                        state.progress.succeededTracks > 0 ||
                                            state.progress.blockedTracks > 0 ||
                                            state.progress.skippedTracks > 0 ||
                                            durableStageCounts == null
                                        )
                                    if (showSavedOutcome) {
                                        Text(
                                            formatDurableTrackCounts(
                                                state.jobState,
                                                state.progress,
                                            ),
                                            style = MaterialTheme.typography.bodySmall,
                                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                                        )
                                    }
                                    formatCurrentIndexingTrack(
                                        activeTrackOrdinal = state.progress.activeTrackOrdinal,
                                        eventTrackOrdinal = state.event?.trackOrdinal,
                                        eventTrackTitle = state.event?.trackTitle,
                                        totalTracks = state.progress.totalTracks,
                                    )?.let { currentTrack ->
                                        Text(
                                            currentTrack,
                                            style = MaterialTheme.typography.bodySmall,
                                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                                            maxLines = 2,
                                            overflow = TextOverflow.Ellipsis,
                                        )
                                    }
                                    formatIndexingEtaEvidence(state.eta, state.etaCoverage)
                                        ?.takeIf { shouldShowIndexingEta(state.jobState) }
                                        ?.let { etaText ->
                                            Text(
                                                etaText,
                                                style = MaterialTheme.typography.bodySmall,
                                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                                            )
                                        }
                                }
                                is IndexingService.IndexingState.PreflightSnapshot -> {
                                    val completed = state.progress.completedUnits
                                    val total = state.progress.totalUnits
                                    if (state.progress.unit != V2IndexingPreflightProgressUnit.BYTES &&
                                        completed != null &&
                                        total != null &&
                                        total > 0L
                                    ) {
                                        LinearProgressIndicator(
                                            progress = {
                                                (completed.toDouble() / total)
                                                    .toFloat()
                                                    .coerceIn(0f, 1f)
                                            },
                                            modifier = Modifier.fillMaxWidth(),
                                        )
                                        Spacer(modifier = Modifier.height(4.dp))
                                    } else if (state.state in setOf(
                                            V2IndexingPreflightIntentState.REQUESTED,
                                            V2IndexingPreflightIntentState.PLANNING,
                                            V2IndexingPreflightIntentState.RESOLVED_WITH_EXECUTABLE_ROWS,
                                            V2IndexingPreflightIntentState.CANCEL_REQUESTED,
                                        )
                                    ) {
                                        LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                                        Spacer(modifier = Modifier.height(4.dp))
                                    }
                                    Text(
                                        preflightStatusEvidenceText(
                                            state = state.state,
                                            failureCode = state.failureCode,
                                            progressMessage = state.progress.message,
                                        ),
                                        style = MaterialTheme.typography.bodyMedium,
                                        color = if (
                                            state.state == V2IndexingPreflightIntentState.FAILED
                                        ) {
                                            MaterialTheme.colorScheme.error
                                        } else {
                                            MaterialTheme.colorScheme.onSurface
                                        },
                                        maxLines = 2,
                                        overflow = TextOverflow.Ellipsis,
                                    )
                                }
                                is IndexingService.IndexingState.Error -> {
                                    Text(state.message,
                                        style = MaterialTheme.typography.bodySmall,
                                        color = MaterialTheme.colorScheme.error)
                                }
                            }

                        }

                        Spacer(modifier = Modifier.height(8.dp))
                        OutlinedButton(
                            onClick = {
                                onDeviceIndexingOpenError = null
                                openingOnDeviceIndexing = true
                                scope.launch {
                                    var indexingActivityLaunched = false
                                    try {
                                        viewModel.prepareForOnDeviceIndexing()
                                        context.startActivity(
                                            android.content.Intent(
                                                context,
                                                com.powerampstartradio.indexing.IndexingActivity::class.java,
                                            ),
                                        )
                                        indexingActivityLaunched = true
                                    } catch (error: Exception) {
                                        Log.e("MainActivity", "On-device indexing handoff failed", error)
                                        onDeviceIndexingOpenError = if (
                                            error.message == MainViewModel.INDEXING_HANDOFF_BUSY_MESSAGE
                                        ) {
                                            MainViewModel.INDEXING_HANDOFF_BUSY_MESSAGE
                                        } else {
                                            "On-device indexing could not open. Nothing was changed; try again."
                                        }
                                    } finally {
                                        if (!indexingActivityLaunched) {
                                            viewModel.abortOnDeviceIndexingHandoff()
                                        }
                                        openingOnDeviceIndexing = false
                                    }
                                }
                            },
                            enabled = !openingOnDeviceIndexing,
                            modifier = Modifier.fillMaxWidth(),
                        ) {
                            if (openingOnDeviceIndexing) {
                                CircularProgressIndicator(
                                    modifier = Modifier.size(18.dp),
                                    strokeWidth = 2.dp,
                                )
                                Spacer(modifier = Modifier.width(8.dp))
                                Text("Releasing Find Music memory...")
                            } else {
                                Text("Manage Tracks")
                            }
                        }
                        onDeviceIndexingOpenError?.let { message ->
                            Spacer(modifier = Modifier.height(4.dp))
                            Text(
                                message,
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.error,
                            )
                        }
                    }
                }
            }

            if (!hasPermission && !permissionLoading) {
                item { HorizontalDivider() }
                item {
                    Column {
                        Text("Poweramp Access", style = MaterialTheme.typography.titleMedium)
                        Spacer(modifier = Modifier.height(8.dp))
                        Button(onClick = onRequestPermission, modifier = Modifier.fillMaxWidth()) {
                            Text("Grant Poweramp Access")
                        }
                    }
                }
            }

            item { HorizontalDivider() }
            item {
                TextButton(
                    onClick = { showResetControlsConfirmation = true },
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Text("Reset radio and Find Music controls")
                }
            }

            item { Spacer(modifier = Modifier.height(32.dp)) }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun NeighborhoodReachControl(
    fraction: Float,
    librarySize: Int?,
    numTracks: Int,
    options: List<Float>,
    recommendedFraction: Float? = null,
    title: String,
    description: String,
    onFractionChange: (Float) -> Unit,
) {
    val reachOptions = remember(options, librarySize, numTracks, fraction) {
        librarySize?.let { size ->
            NeighborhoodReachPolicy.distinctOptions(
                options = options,
                librarySize = size,
                numTracks = numTracks,
                preferredFraction = fraction,
            )
        } ?: options
    }
    if (reachOptions.size <= 1) return
    val reachIndex = reachOptions.indexOf(fraction).takeIf { it >= 0 }
        ?: reachOptions.indices.minByOrNull { index ->
            kotlin.math.abs(reachOptions[index] - fraction)
        }
        ?: 0
    val displayedFraction = reachOptions[reachIndex.coerceIn(reachOptions.indices)]
    val poolCount = librarySize?.let { size ->
        NeighborhoodReachPolicy.candidateCount(displayedFraction, size, numTracks)
    }
    val rightEndpointFraction = reachOptions.last()
    val rightEndpointCount = librarySize?.let { size ->
        NeighborhoodReachPolicy.candidateCount(rightEndpointFraction, size, numTracks)
    }
    val rightEndpointLabel = if (
        rightEndpointFraction >= 1f ||
        (librarySize != null && rightEndpointCount == librarySize)
    ) {
        "All eligible"
    } else {
        "Nearest ${formatKnob(rightEndpointFraction * 100f)}%"
    }
    val selectionPoolValue = when {
        displayedFraction >= 1f && librarySize != null ->
            "All ${formatCount(librarySize)} eligible"
        displayedFraction >= 1f -> "All eligible"
        poolCount != null ->
            "Nearest ${formatCount(poolCount)} of ${formatCount(checkNotNull(librarySize))} " +
                "(${formatKnob(displayedFraction * 100f)}%)"
        else -> "Nearest ${formatKnob(displayedFraction * 100f)}%"
    }
    val accessibilityDescription = buildString {
        append(selectionPoolValue)
        if (recommendedFraction == displayedFraction) append(", default")
    }
    val thumb: @Composable (SliderState) -> Unit = {
        Box(
            Modifier
                .size(16.dp)
                .background(MaterialTheme.colorScheme.primary, CircleShape),
        )
    }
    val track: @Composable (SliderState) -> Unit = { state ->
        SliderDefaults.Track(
            sliderState = state,
            modifier = Modifier.height(8.dp),
            drawStopIndicator = null,
            thumbTrackGapSize = 0.dp,
            trackInsideCornerSize = 0.dp,
        )
    }

    Column {
        SettingsTitleValueRow(
            title = title,
            value = buildString {
                append(selectionPoolValue)
                if (recommendedFraction == displayedFraction) append(" \u00b7 Default")
            },
        )
        Text(
            description,
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Slider(
            value = reachIndex.toFloat(),
            onValueChange = { value ->
                onFractionChange(
                    reachOptions[value.roundToInt().coerceIn(reachOptions.indices)],
                )
            },
            valueRange = 0f..reachOptions.lastIndex.toFloat(),
            steps = reachOptions.size - 2,
            modifier = Modifier.semantics {
                stateDescription = accessibilityDescription
            },
            thumb = thumb,
            track = track,
        )
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
        ) {
            Text(
                "Tighter",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f),
            )
            Text(
                rightEndpointLabel,
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f),
            )
        }
    }
}

@Composable
private fun SettingsTitleValueRow(title: String, value: String) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(12.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(
            title,
            style = MaterialTheme.typography.titleSmall,
            modifier = Modifier.weight(1f),
        )
        Text(
            value,
            style = MaterialTheme.typography.labelMedium,
            color = MaterialTheme.colorScheme.primary,
            textAlign = TextAlign.End,
        )
    }
}

@Composable
private fun AlgorithmOption(
    label: String,
    summary: String,
    selected: Boolean,
    onClick: () -> Unit,
    availabilityNote: String? = null,
    enabled: Boolean = true,
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .alpha(if (enabled) 1f else 0.55f)
            .selectable(
                selected = selected,
                enabled = enabled,
                onClick = onClick,
                role = Role.RadioButton,
            )
            .heightIn(min = 44.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        RadioButton(
            selected = selected,
            onClick = null,
            enabled = enabled,
        )
        Spacer(modifier = Modifier.width(8.dp))
        Column(modifier = Modifier.weight(1f)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(
                    label,
                    style = MaterialTheme.typography.bodyMedium,
                    fontWeight = if (selected) FontWeight.SemiBold else FontWeight.Normal,
                    modifier = Modifier.weight(1f),
                )
                availabilityNote?.let { note ->
                    Text(
                        note,
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
            Text(
                summary,
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                maxLines = 2,
                overflow = TextOverflow.Ellipsis,
            )
        }
    }
}

@Composable
private fun SettingsAlgorithmPreview(
    label: String,
    previewTrackCount: Int,
    preview: SettingsPeekPreview?,
    expanded: Boolean,
    onToggleExpanded: () -> Unit,
    enabled: Boolean,
) {
    TextButton(
        enabled = enabled,
        onClick = onToggleExpanded,
        contentPadding = PaddingValues(horizontal = 0.dp),
        modifier = Modifier.heightIn(min = 40.dp),
    ) {
        Icon(
            imageVector = Icons.Default.Search,
            contentDescription = null,
            modifier = Modifier.size(16.dp),
        )
        Spacer(modifier = Modifier.width(6.dp))
        Text(if (expanded) "Hide radio preview" else "Preview radio result")
    }
    AnimatedVisibility(visible = expanded) {
        when (preview) {
            null,
            SettingsPeekPreview.Loading -> {
                Row(
                    modifier = Modifier.fillMaxWidth().padding(bottom = 8.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(14.dp),
                        strokeWidth = 1.5.dp,
                        color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.5f),
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = "Resolving Poweramp's current track, then planning a " +
                            "$previewTrackCount-track radio result with $label",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.8f),
                    )
                }
            }
            is SettingsPeekPreview.Ready -> {
                Column(
                    modifier = Modifier
                        .heightIn(max = 220.dp)
                        .verticalScroll(rememberScrollState()),
                ) {
                    Text(
                        text = preview.resultLine,
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.8f),
                        modifier = Modifier.padding(bottom = 4.dp),
                    )
                    preview.firstDisplayLabels.forEachIndexed { index, track ->
                        Text(
                            text = "${index + 1}. $track",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.7f),
                        )
                    }
                }
            }
            is SettingsPeekPreview.Unavailable -> {
                SettingsPeekOutcomeLine(preview.resultLine)
            }
            is SettingsPeekPreview.Error -> {
                SettingsPeekOutcomeLine(preview.resultLine)
            }
        }
    }
}

@Composable
private fun SettingsPeekOutcomeLine(text: String) {
    Text(
        text = text,
        style = MaterialTheme.typography.labelSmall,
        color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.8f),
        modifier = Modifier.padding(bottom = 4.dp),
    )
}

@Composable
private fun FileStatusRow(file: AppFileStatus) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 3.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(
            text = if (file.present) "\u2713" else "\u2717",
            color = if (file.present) MaterialTheme.colorScheme.primary
                    else MaterialTheme.colorScheme.error,
            style = MaterialTheme.typography.bodyMedium,
            fontWeight = FontWeight.Bold,
            modifier = Modifier.width(20.dp),
        )
        Text(
            text = file.name,
            style = MaterialTheme.typography.bodySmall,
            fontFamily = FontFamily.Monospace,
            modifier = Modifier.weight(1f),
        )
        if (file.sizeMb != null) {
            Text(
                text = file.sizeMb,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
    if (file.detail != null) {
        Text(
            text = file.detail,
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.padding(start = 20.dp),
        )
    }
}

// ---- Human-friendly label helpers ----

private fun humanSelectionMode(mode: SelectionMode, drift: Boolean = false): String {
    val base = when (mode) {
        SelectionMode.CLOSEST -> "Closest"
        SelectionMode.MMR -> "MMR"
        SelectionMode.DPP -> "DPP"
        SelectionMode.RANDOM_WALK -> "Graph Explorer"
        SelectionMode.UNIFORM_SHUFFLE -> "Uniform shuffle"
    }
    return if (drift && mode == SelectionMode.MMR) "$base + drift" else base
}

private fun formatKnob(v: Float): String =
    if (v == v.toInt().toFloat()) {
        v.toInt().toString()
    } else {
        "%.2f".format(v).trimEnd('0').trimEnd('.')
    }
