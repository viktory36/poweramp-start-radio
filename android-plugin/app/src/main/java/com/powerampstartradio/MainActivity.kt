package com.powerampstartradio

import android.content.IntentFilter
import android.net.Uri
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.slideInHorizontally
import androidx.compose.animation.slideOutHorizontally
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.selection.selectableGroup
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Clear
import androidx.compose.material.icons.filled.Menu
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.draw.clipToBounds
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.lerp
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import androidx.lifecycle.viewmodel.compose.viewModel
import com.powerampstartradio.data.EmbeddingDatabase
import com.powerampstartradio.poweramp.PowerampHelper
import com.powerampstartradio.poweramp.PowerampReceiver
import com.powerampstartradio.poweramp.PowerampTrack
import com.powerampstartradio.poweramp.TrackMatcher
import com.powerampstartradio.services.RadioService
import com.powerampstartradio.ui.DatabaseInfo
import com.powerampstartradio.ui.DecaySchedule
import com.powerampstartradio.ui.DriftMode
import com.powerampstartradio.ui.MainViewModel
import com.powerampstartradio.ui.QueuedTrackResult
import com.powerampstartradio.ui.RadioResult
import com.powerampstartradio.ui.RadioUiState
import com.powerampstartradio.ui.SelectionMode
import com.powerampstartradio.ui.TrackProvenance
import com.powerampstartradio.ui.theme.PowerampStartRadioTheme
import kotlinx.coroutines.launch
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.roundToInt

/** Whether the radio UI state represents an active search (any phase). */
private fun RadioUiState.isActiveSearch(): Boolean =
    this is RadioUiState.Loading || this is RadioUiState.Searching || this is RadioUiState.Streaming

class MainActivity : ComponentActivity() {

    companion object {
        private const val TAG = "MainActivity"
    }

    private val trackReceiver = PowerampReceiver()
    private var onResumeCallback: (() -> Unit)? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val filter = IntentFilter().apply {
            addAction(PowerampHelper.ACTION_TRACK_CHANGED)
            addAction(PowerampHelper.ACTION_STATUS_CHANGED)
        }
        ContextCompat.registerReceiver(
            this, trackReceiver, filter, ContextCompat.RECEIVER_EXPORTED
        )

        setContent {
            PowerampStartRadioTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
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
    }

    override fun onResume() {
        super.onResume()
        onResumeCallback?.invoke()
    }

    override fun onDestroy() {
        super.onDestroy()
        unregisterReceiver(trackReceiver)
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
    val hasPermission by viewModel.hasPermission.collectAsState()
    val sessionHistory by viewModel.sessionHistory.collectAsState()
    val indexStatus by viewModel.indexStatus.collectAsState()

    var currentTrack by remember { mutableStateOf<PowerampTrack?>(PowerampReceiver.currentTrack) }
    var showSettings by remember { mutableStateOf(false) }
    var statusMessage by remember { mutableStateOf("") }
    var viewingSession by remember { mutableStateOf<Int?>(null) }

    LaunchedEffect(radioState) {
        if (radioState is RadioUiState.Idle) viewingSession = null
    }

    val drawerState = rememberDrawerState(DrawerValue.Closed)
    val scope = rememberCoroutineScope()

    LaunchedEffect(Unit) {
        onRegisterResumeCallback?.invoke {
            viewModel.checkPermission()
            viewModel.refreshDatabaseInfo()
        }
    }

    DisposableEffect(Unit) {
        val listener: (PowerampTrack?) -> Unit = { track -> currentTrack = track }
        PowerampReceiver.addTrackChangeListener(listener)
        onDispose { PowerampReceiver.removeTrackChangeListener(listener) }
    }

    val importLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument()
    ) { uri: Uri? ->
        uri?.let {
            statusMessage = "Importing database..."
            try {
                val destFile = File(context.filesDir, "embeddings.db")
                EmbeddingDatabase.importFrom(context, it, destFile).close()
                viewModel.refreshDatabaseInfo()
                statusMessage = "Database imported!"
            } catch (e: Exception) {
                statusMessage = "Import failed: ${e.message}"
                Log.e("MainActivity", "Import failed", e)
            }
        }
    }


    BackHandler(enabled = showSettings || viewingSession != null) {
        if (showSettings) showSettings = false
        else {
            viewingSession = null
            viewModel.resetRadioState()
        }
    }

    ModalNavigationDrawer(
        drawerState = drawerState,
        gesturesEnabled = sessionHistory.isNotEmpty(),
        drawerContent = {
            ModalDrawerSheet(modifier = Modifier.width(280.dp)) {
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
        AnimatedContent(
            targetState = showSettings,
            transitionSpec = {
                slideInHorizontally { if (targetState) it else -it } togetherWith
                    slideOutHorizontally { if (targetState) -it else it }
            },
            label = "settings_transition"
        ) { isSettings ->
            if (isSettings) {
                SettingsScreen(viewModel = viewModel, databaseInfo = databaseInfo,
                    onImportDatabase = { importLauncher.launch(arrayOf("application/octet-stream", "*/*")) },
                    hasPermission = hasPermission,
                    onRequestPermission = { viewModel.requestPermission() },
                    onBack = { showSettings = false })
            } else {
                HomeScreen(
                    radioState = radioState, currentTrack = currentTrack,
                    databaseInfo = databaseInfo, hasPermission = hasPermission,
                    sessionHistory = sessionHistory, statusMessage = statusMessage,
                    indexStatus = indexStatus,
                    onStartRadio = {
                        if (currentTrack != null && databaseInfo != null) viewModel.startRadio()
                        else if (currentTrack == null) statusMessage = "Play a song in Poweramp first"
                        else statusMessage = "Import database in Settings"
                    },
                    onCancelSearch = { viewModel.cancelSearch() },
                    onClearAndReset = { viewModel.resetRadioState() },
                    onRequestPermission = { viewModel.requestPermission() },
                    onOpenSettings = { showSettings = true },
                    onOpenDrawer = { scope.launch { drawerState.open() } },
                    viewingSession = viewingSession,
                    onViewSession = { viewingSession = it }
                )
            }
        }
    }
}

// ---- Home Screen ----

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun HomeScreen(
    radioState: RadioUiState,
    currentTrack: PowerampTrack?,
    databaseInfo: DatabaseInfo?,
    hasPermission: Boolean,
    sessionHistory: List<RadioResult>,
    statusMessage: String,
    indexStatus: String?,
    onStartRadio: () -> Unit,
    onCancelSearch: () -> Unit,
    onClearAndReset: () -> Unit,
    onRequestPermission: () -> Unit,
    onOpenSettings: () -> Unit,
    onOpenDrawer: () -> Unit,
    viewingSession: Int?,
    onViewSession: (Int?) -> Unit
) {
    val showResults = radioState is RadioUiState.Success
        || radioState is RadioUiState.Streaming
        || viewingSession != null
    val displaySession = when (radioState) {
        is RadioUiState.Streaming -> if (viewingSession != null && viewingSession in sessionHistory.indices) sessionHistory[viewingSession] else radioState.result
        is RadioUiState.Success -> if (viewingSession != null && viewingSession in sessionHistory.indices) sessionHistory[viewingSession] else radioState.result
        else -> if (viewingSession != null && viewingSession in sessionHistory.indices) sessionHistory[viewingSession] else sessionHistory.lastOrNull()
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Start Radio") },
                navigationIcon = {
                    if (sessionHistory.isNotEmpty()) {
                        IconButton(onClick = onOpenDrawer) {
                            Icon(Icons.Default.Menu, contentDescription = "History")
                        }
                    }
                },
                actions = {
                    if (showResults) {
                        IconButton(onClick = {
                            onViewSession(null)
                            onClearAndReset()
                        }) {
                            Icon(Icons.Default.Clear, contentDescription = "Clear")
                        }
                    }
                    IconButton(onClick = onOpenSettings) {
                        Icon(Icons.Default.Settings, contentDescription = "Settings")
                    }
                }
            )
        },
        floatingActionButton = {
            if (radioState.isActiveSearch()) {
                ExtendedFloatingActionButton(
                    onClick = onCancelSearch,
                    icon = { Icon(Icons.Default.Clear, contentDescription = null) },
                    text = { Text("Cancel") },
                    containerColor = MaterialTheme.colorScheme.errorContainer,
                    contentColor = MaterialTheme.colorScheme.onErrorContainer,
                    expanded = true
                )
            } else {
                ExtendedFloatingActionButton(
                    onClick = onStartRadio,
                    icon = { Icon(Icons.Default.PlayArrow, contentDescription = null) },
                    text = { Text("Start Radio") },
                    expanded = true
                )
            }
        }
    ) { padding ->
        Box(
            modifier = Modifier.fillMaxSize().padding(padding)
        ) {
            Column(modifier = Modifier.fillMaxSize()) {
                val activeMatchType = if (showResults && displaySession != null &&
                    currentTrack?.realId == displaySession.seedTrack.realId
                ) displaySession.matchType else null
                CompactNowPlayingHeader(
                    currentTrack = currentTrack,
                    matchType = activeMatchType,
                    modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp)
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
                        SessionPage(session = displaySession, modifier = Modifier.fillMaxSize())

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
                                    statusMessage = statusMessage, indexStatus = indexStatus,
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
    matchType: TrackMatcher.MatchType? = null,
    modifier: Modifier = Modifier
) {
    if (currentTrack != null) {
        Column(modifier = modifier) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text("NOW PLAYING", style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.primary)
                if (matchType != null) {
                    Text(" \u00b7 ${humanMatchType(matchType)}",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
            }
            Text(text = currentTrack.title, style = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.Bold, maxLines = 1, overflow = TextOverflow.Ellipsis)
            Text(text = listOfNotNull(currentTrack.artist, currentTrack.album).joinToString(" \u00b7 "),
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                maxLines = 1, overflow = TextOverflow.Ellipsis)
        }
    } else {
        Text("No track playing", style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = modifier)
    }
}

// ---- Session Page ----

@Composable
fun SessionPage(session: RadioResult, modifier: Modifier = Modifier) {
    val listState = rememberLazyListState()
    val showProvenance = session.config.driftEnabled

    LaunchedEffect(session.tracks.size) {
        if (!session.isComplete && session.tracks.isNotEmpty()) {
            val lastVisible = listState.layoutInfo.visibleItemsInfo.lastOrNull()?.index ?: 0
            val totalItems = listState.layoutInfo.totalItemsCount
            val isAtBottom = totalItems == 0 || lastVisible >= totalItems - 2
            if (isAtBottom) listState.animateScrollToItem(session.tracks.lastIndex)
        }
    }

    Column(modifier = modifier) {
        ResultsSummary(result = session,
            modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp))

        LazyColumn(state = listState, modifier = Modifier.fillMaxSize(),
            contentPadding = PaddingValues(horizontal = 16.dp, vertical = 4.dp)) {
            items(session.tracks.size) { index ->
                TrackResultRow(
                    trackResult = session.tracks[index],
                    index = index,
                    totalTracks = session.tracks.size,
                    showProvenance = showProvenance
                )
            }
            if (!session.isComplete) {
                item { StreamingProgressItem(found = session.tracks.size, total = session.totalExpected) }
            }
        }
    }
}

@Composable
private fun StreamingProgressItem(found: Int, total: Int) {
    Row(modifier = Modifier.fillMaxWidth().padding(vertical = 12.dp),
        horizontalArrangement = Arrangement.Center,
        verticalAlignment = Alignment.CenterVertically) {
        CircularProgressIndicator(modifier = Modifier.size(16.dp), strokeWidth = 2.dp)
        Spacer(modifier = Modifier.width(8.dp))
        Text("$found of $total found...", style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

@Composable
fun ResultsSummary(result: RadioResult, modifier: Modifier = Modifier) {
    val modeLabel = humanSelectionMode(result.config.selectionMode, result.config.driftEnabled)
    val seedName = result.seedTrack.title

    val countText = if (!result.isComplete) {
        "$seedName \u2014 ${result.tracks.size} of ${result.totalExpected} found..."
    } else if (result.failedCount > 0) {
        "$seedName \u2014 ${result.queuedCount} of ${result.requestedCount} queued (${result.failedCount} not found)"
    } else {
        "$seedName \u2014 ${result.queuedCount} tracks via $modeLabel"
    }

    Text(text = countText, style = MaterialTheme.typography.labelMedium,
        color = MaterialTheme.colorScheme.onSurfaceVariant,
        modifier = modifier.fillMaxWidth(), maxLines = 1, overflow = TextOverflow.Ellipsis)
}

// ---- Influence strip provenance visualization ----

private fun trackColor(primary: Color, index: Int, total: Int): Color {
    if (total <= 1) return primary
    val hsl = FloatArray(3)
    androidx.core.graphics.ColorUtils.colorToHSL(primary.toArgb(), hsl)
    hsl[0] = (hsl[0] + (index.toFloat() / total) * 180f) % 360f
    return Color(androidx.core.graphics.ColorUtils.HSLToColor(hsl))
}

@Composable
fun TrackIdentityDot(index: Int, totalTracks: Int) {
    val primary = MaterialTheme.colorScheme.primary
    val color = remember(primary, index, totalTracks) {
        trackColor(primary, index, totalTracks)
    }
    Canvas(modifier = Modifier.size(8.dp)) {
        drawCircle(color = color)
    }
}

@Composable
fun InfluenceStrip(
    provenance: TrackProvenance,
    totalTracks: Int,
    modifier: Modifier = Modifier
) {
    val primary = MaterialTheme.colorScheme.primary
    val density = LocalDensity.current
    val stripWidthPx = with(density) { 32.dp.toPx() }

    val segmentColors = remember(primary, totalTracks, provenance) {
        provenance.influences.map { influence ->
            if (influence.sourceIndex == -1) primary
            else trackColor(primary, influence.sourceIndex, totalTracks)
        }
    }

    Canvas(modifier = modifier.width(32.dp).fillMaxHeight().clipToBounds()) {
        val sorted = provenance.influences
            .zip(segmentColors)
            .sortedBy { it.first.sourceIndex }
        var x = 0f

        for ((influence, color) in sorted) {
            val segWidth = influence.weight * stripWidthPx
            if (segWidth < 0.5f) continue
            drawRect(
                color = color,
                topLeft = Offset(x, 0f),
                size = Size(segWidth, size.height)
            )
            x += segWidth
        }
    }
}

// ---- Track Result Row ----

@Composable
fun TrackResultRow(
    trackResult: QueuedTrackResult,
    index: Int,
    totalTracks: Int,
    showProvenance: Boolean
) {
    Row(modifier = Modifier.fillMaxWidth().height(IntrinsicSize.Min),
        verticalAlignment = Alignment.CenterVertically) {
        if (showProvenance) {
            TrackIdentityDot(index = index, totalTracks = totalTracks)
            Spacer(modifier = Modifier.width(4.dp))
            InfluenceStrip(
                provenance = trackResult.provenance,
                totalTracks = totalTracks,
                modifier = Modifier.fillMaxHeight()
            )
            Spacer(modifier = Modifier.width(4.dp))
        }
        Column(modifier = Modifier.weight(1f).padding(vertical = 2.dp, horizontal = 4.dp)) {
            Text(text = trackResult.track.title ?: "Unknown", style = MaterialTheme.typography.bodyMedium,
                maxLines = 1, overflow = TextOverflow.Ellipsis)
            Text(text = trackResult.track.artist ?: "Unknown", style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant, maxLines = 1, overflow = TextOverflow.Ellipsis)
        }

        val normalized = trackResult.similarity.coerceIn(0f, 1f)
        val vivid = MaterialTheme.colorScheme.primary
        val scoreColor = lerp(vivid.copy(alpha = 0.15f), vivid, normalized)
        val scoreText = String.format("%.3f", trackResult.similarity).removePrefix("0")

        Text(text = "($scoreText)", fontFamily = FontFamily.Monospace, color = scoreColor,
            fontSize = 9.sp, textAlign = TextAlign.End, modifier = Modifier.padding(horizontal = 2.dp))
    }
}

// ---- Session History Drawer ----

@Composable
fun SessionHistoryDrawer(sessions: List<RadioResult>, onSessionTap: (Int) -> Unit, onClear: () -> Unit) {
    Column(modifier = Modifier.fillMaxHeight()) {
        Row(modifier = Modifier.fillMaxWidth().padding(16.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically) {
            Text("Session History", style = MaterialTheme.typography.titleMedium)
            if (sessions.isNotEmpty()) TextButton(onClick = onClear) { Text("Clear") }
        }
        HorizontalDivider()

        if (sessions.isEmpty()) {
            Box(modifier = Modifier.fillMaxWidth().padding(32.dp), contentAlignment = Alignment.Center) {
                Text("No sessions yet", style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        } else {
            LazyColumn(modifier = Modifier.weight(1f), contentPadding = PaddingValues(vertical = 4.dp)) {
                items(sessions.size) { index ->
                    val session = sessions[index]
                    val timeStr = SimpleDateFormat("HH:mm", Locale.getDefault()).format(Date(session.timestamp))
                    NavigationDrawerItem(
                        label = {
                            Column {
                                Text(session.seedTrack.title, style = MaterialTheme.typography.bodyMedium,
                                    maxLines = 1, overflow = TextOverflow.Ellipsis)
                                Text("${session.seedTrack.artist ?: "Unknown"} \u00b7 $timeStr \u00b7 ${session.queuedCount} tracks",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                                    maxLines = 1, overflow = TextOverflow.Ellipsis)
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
    hasPermission: Boolean, databaseInfo: DatabaseInfo?, statusMessage: String,
    indexStatus: String?, isIdle: Boolean = true, onRequestPermission: () -> Unit,
    modifier: Modifier = Modifier
) {
    Column(modifier = modifier.fillMaxWidth().padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)) {
        if (!hasPermission) {
            Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.errorContainer)) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text("Poweramp Access Required", style = MaterialTheme.typography.titleSmall)
                    Spacer(modifier = Modifier.height(8.dp))
                    Button(onClick = onRequestPermission) { Text("Grant Access") }
                }
            }
        }
        if (databaseInfo == null) {
            Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.secondaryContainer)) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text("No embedding database", style = MaterialTheme.typography.titleSmall)
                    Text("Import via Settings", style = MaterialTheme.typography.bodySmall)
                }
            }
        }
        if (indexStatus != null && indexStatus != "Index ready") {
            Text(indexStatus, style = MaterialTheme.typography.bodySmall,
                fontFamily = FontFamily.Monospace, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        if (statusMessage.isNotEmpty()) {
            Text(statusMessage, style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        if (isIdle && hasPermission && databaseInfo != null && statusMessage.isEmpty()) {
            Text("Ready when you are!", style = MaterialTheme.typography.bodyLarge,
                color = MaterialTheme.colorScheme.primary)
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
    hasPermission: Boolean,
    onRequestPermission: () -> Unit,
    onBack: () -> Unit
) {
    val selectionMode by viewModel.selectionMode.collectAsState()
    val driftEnabled by viewModel.driftEnabled.collectAsState()
    val driftMode by viewModel.driftMode.collectAsState()
    val anchorStrength by viewModel.anchorStrength.collectAsState()
    val anchorDecay by viewModel.anchorDecay.collectAsState()
    val momentumBeta by viewModel.momentumBeta.collectAsState()
    val diversityLambda by viewModel.diversityLambda.collectAsState()
    val temperature by viewModel.temperature.collectAsState()
    val maxPerArtist by viewModel.maxPerArtist.collectAsState()
    val minArtistSpacing by viewModel.minArtistSpacing.collectAsState()
    val numTracks by viewModel.numTracks.collectAsState()
    val previews by viewModel.previews.collectAsState()
    val previewsLoading by viewModel.previewsLoading.collectAsState()

    val isRandomWalk = selectionMode == SelectionMode.RANDOM_WALK
    val isDpp = selectionMode == SelectionMode.DPP

    // Group keys by what affects each mode
    val commonKeys = remember(numTracks, maxPerArtist, minArtistSpacing) { Any() }
    val driftKeys = remember(driftEnabled, driftMode, anchorStrength, anchorDecay, momentumBeta) { Any() }
    val expandedPeek = remember { mutableStateMapOf<SelectionMode, Boolean>() }

    // Invalidate stale previews when relevant settings change (lazy — computed on peek click)
    LaunchedEffect(commonKeys, driftKeys, diversityLambda) {
        viewModel.invalidatePreview(SelectionMode.MMR)
        expandedPeek[SelectionMode.MMR] = false
    }
    LaunchedEffect(commonKeys) {
        viewModel.invalidatePreview(SelectionMode.DPP)
        expandedPeek[SelectionMode.DPP] = false
    }
    LaunchedEffect(commonKeys, anchorStrength) {
        viewModel.invalidatePreview(SelectionMode.RANDOM_WALK)
        expandedPeek[SelectionMode.RANDOM_WALK] = false
    }
    LaunchedEffect(commonKeys, driftKeys, temperature) {
        viewModel.invalidatePreview(SelectionMode.TEMPERATURE)
        expandedPeek[SelectionMode.TEMPERATURE] = false
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Settings") },
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
            // Queue size
            item {
                Column {
                    Text("Queue Size: $numTracks tracks", style = MaterialTheme.typography.titleMedium)
                    Slider(value = numTracks.toFloat(), onValueChange = { viewModel.setNumTracks(it.toInt()) },
                        valueRange = 10f..100f, steps = 8)
                }
            }

            item { HorizontalDivider() }

            // Selection algorithm
            item {
                Text("Selection Algorithm", style = MaterialTheme.typography.titleMedium)
                Spacer(modifier = Modifier.height(4.dp))

                val driftInfo = if (!driftEnabled) "" else when (driftMode) {
                    DriftMode.SEED_INTERPOLATION -> {
                        val decay = when (anchorDecay) {
                            DecaySchedule.NONE -> ""
                            DecaySchedule.LINEAR -> " linear"
                            DecaySchedule.EXPONENTIAL -> " exp"
                            DecaySchedule.STEP -> " step"
                        }
                        " + drift(\u03b1=${(anchorStrength * 100).roundToInt()}%$decay)"
                    }
                    DriftMode.MOMENTUM ->
                        " + drift(EMA \u03b2=${(momentumBeta * 100).roundToInt()}%)"
                }

                Column(modifier = Modifier.selectableGroup()) {
                    AlgorithmOption(
                        label = "MMR (Maximal Marginal Relevance)",
                        description = "Picks similar tracks, then penalizes each candidate by how " +
                            "close it is to tracks already chosen.",
                        preview = previews[SelectionMode.MMR],
                        isLoading = SelectionMode.MMR in previewsLoading,
                        selected = selectionMode == SelectionMode.MMR,
                        expanded = expandedPeek[SelectionMode.MMR] == true,
                        onToggleExpanded = { expandedPeek[SelectionMode.MMR] = !(expandedPeek[SelectionMode.MMR] ?: false) },
                        onRequestPreview = { viewModel.computePreview(SelectionMode.MMR) },
                        previewInfo = "\u03bb=${(diversityLambda * 100).roundToInt()}%$driftInfo",
                        onClick = { viewModel.setSelectionMode(SelectionMode.MMR) }
                    )
                    AlgorithmOption(
                        label = "DPP (Determinantal Point Process)",
                        description = "Picks a set where every track is relevant and every pair is " +
                            "dissimilar. Unlike MMR, considers all pairwise interactions at once.",
                        preview = previews[SelectionMode.DPP],
                        isLoading = SelectionMode.DPP in previewsLoading,
                        selected = selectionMode == SelectionMode.DPP,
                        expanded = expandedPeek[SelectionMode.DPP] == true,
                        onToggleExpanded = { expandedPeek[SelectionMode.DPP] = !(expandedPeek[SelectionMode.DPP] ?: false) },
                        onRequestPreview = { viewModel.computePreview(SelectionMode.DPP) },
                        onClick = { viewModel.setSelectionMode(SelectionMode.DPP) }
                    )
                    AlgorithmOption(
                        label = "Random Walk (Personalized PageRank)",
                        description = "Hops between neighbors on a similarity graph. Reaches tracks " +
                            "through indirect chains \u2014 A similar to B, B similar to C, so C " +
                            "appears even if A and C aren't directly similar." +
                            if (databaseInfo?.hasGraph != true) " (requires kNN graph in database)" else "",
                        preview = previews[SelectionMode.RANDOM_WALK],
                        isLoading = SelectionMode.RANDOM_WALK in previewsLoading,
                        selected = selectionMode == SelectionMode.RANDOM_WALK,
                        expanded = expandedPeek[SelectionMode.RANDOM_WALK] == true,
                        onToggleExpanded = { expandedPeek[SelectionMode.RANDOM_WALK] = !(expandedPeek[SelectionMode.RANDOM_WALK] ?: false) },
                        onRequestPreview = { viewModel.computePreview(SelectionMode.RANDOM_WALK) },
                        previewInfo = "\u03b1=${(anchorStrength * 100).roundToInt()}%",
                        onClick = { viewModel.setSelectionMode(SelectionMode.RANDOM_WALK) }
                    )
                    AlgorithmOption(
                        label = "Temperature Sampling (Gumbel-Max)",
                        description = "Samples randomly from top candidates instead of always " +
                            "taking the best match. Non-deterministic \u2014 different results each run.",
                        preview = previews[SelectionMode.TEMPERATURE],
                        isLoading = SelectionMode.TEMPERATURE in previewsLoading,
                        selected = selectionMode == SelectionMode.TEMPERATURE,
                        expanded = expandedPeek[SelectionMode.TEMPERATURE] == true,
                        onToggleExpanded = { expandedPeek[SelectionMode.TEMPERATURE] = !(expandedPeek[SelectionMode.TEMPERATURE] ?: false) },
                        onRequestPreview = { viewModel.computePreview(SelectionMode.TEMPERATURE) },
                        previewInfo = "\u03c4=${String.format("%.2f", temperature)}$driftInfo",
                        onClick = { viewModel.setSelectionMode(SelectionMode.TEMPERATURE) }
                    )
                }
            }

            // MMR lambda — only for MMR
            if (selectionMode == SelectionMode.MMR) {
                item {
                    Column {
                        Text("Lambda (\u03bb): ${(diversityLambda * 100).roundToInt()}%",
                            style = MaterialTheme.typography.titleSmall)
                        Text("Low = diversity penalty dominates, picks spread apart. " +
                            "High = relevance dominates, picks cluster closer.",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant)
                        Slider(value = diversityLambda, onValueChange = { viewModel.setDiversityLambda(it) },
                            valueRange = 0f..1f)
                    }
                }
            }

            // Temperature — only for Temperature mode
            if (selectionMode == SelectionMode.TEMPERATURE) {
                item {
                    Column {
                        Text("Temperature (\u03c4): ${String.format("%.2f", temperature)}",
                            style = MaterialTheme.typography.titleSmall)
                        Text("Low = nearly deterministic, takes the top matches. " +
                            "High = samples more uniformly across candidates.",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant)
                        Slider(value = temperature, onValueChange = { viewModel.setTemperature(it) },
                            valueRange = 0f..1f)
                    }
                }
            }

            // Random Walk restart probability — only for Random Walk
            if (isRandomWalk) {
                item {
                    Column {
                        Text("Restart Probability (\u03b1): ${(anchorStrength * 100).roundToInt()}%",
                            style = MaterialTheme.typography.titleSmall)
                        Text("How often the walk jumps back to the seed. " +
                            "High = resets frequently, stays local. " +
                            "Low = walks further before resetting.",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant)
                        Slider(value = anchorStrength, onValueChange = { viewModel.setAnchorStrength(it) },
                            valueRange = 0.05f..0.95f)
                    }
                }
            }

            // Drift — not applicable to Random Walk or DPP
            // (DPP in drift mode is degenerate: k=1 selection = pure greedy max,
            //  identical to MMR lambda=1. DPP's value is batch-only pairwise diversity.)
            if (!isRandomWalk && !isDpp) {
                item { HorizontalDivider() }

                item {
                    Column {
                        Text("Query Drift", style = MaterialTheme.typography.titleMedium)
                        Spacer(modifier = Modifier.height(4.dp))

                        Row(modifier = Modifier.fillMaxWidth().selectable(
                            selected = driftEnabled, onClick = { viewModel.setDriftEnabled(!driftEnabled) }, role = Role.Checkbox
                        ).padding(vertical = 4.dp), verticalAlignment = Alignment.CenterVertically) {
                            Checkbox(checked = driftEnabled, onCheckedChange = null)
                            Spacer(modifier = Modifier.width(8.dp))
                            Column {
                                Text("Enable Drift", style = MaterialTheme.typography.bodyMedium)
                                Text("When on, each pick shifts what gets searched for next. " +
                                    "When off, all picks are based on similarity to the seed only.",
                                    style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                            }
                        }

                        AnimatedVisibility(visible = driftEnabled) {
                            Column(modifier = Modifier.padding(start = 16.dp, top = 8.dp)) {
                                Text("Drift Mode", style = MaterialTheme.typography.titleSmall)
                                Spacer(modifier = Modifier.height(4.dp))
                                Column(modifier = Modifier.selectableGroup()) {
                                    Row(modifier = Modifier.fillMaxWidth().selectable(
                                        selected = driftMode == DriftMode.SEED_INTERPOLATION,
                                        onClick = { viewModel.setDriftMode(DriftMode.SEED_INTERPOLATION) },
                                        role = Role.RadioButton
                                    ).padding(vertical = 4.dp), verticalAlignment = Alignment.CenterVertically) {
                                        RadioButton(selected = driftMode == DriftMode.SEED_INTERPOLATION, onClick = null)
                                        Spacer(modifier = Modifier.width(8.dp))
                                        Column {
                                            Text("Seed Interpolation", style = MaterialTheme.typography.bodyMedium)
                                            Text("Blends the seed with the latest pick to decide what to search " +
                                                "for next. Alpha controls the blend ratio.",
                                                style = MaterialTheme.typography.bodySmall,
                                                color = MaterialTheme.colorScheme.onSurfaceVariant)
                                        }
                                    }
                                    Row(modifier = Modifier.fillMaxWidth().selectable(
                                        selected = driftMode == DriftMode.MOMENTUM,
                                        onClick = { viewModel.setDriftMode(DriftMode.MOMENTUM) },
                                        role = Role.RadioButton
                                    ).padding(vertical = 4.dp), verticalAlignment = Alignment.CenterVertically) {
                                        RadioButton(selected = driftMode == DriftMode.MOMENTUM, onClick = null)
                                        Spacer(modifier = Modifier.width(8.dp))
                                        Column {
                                            Text("EMA Momentum", style = MaterialTheme.typography.bodyMedium)
                                            Text("Running average of all picks so far. " +
                                                "Search direction changes gradually, not abruptly.",
                                                style = MaterialTheme.typography.bodySmall,
                                                color = MaterialTheme.colorScheme.onSurfaceVariant)
                                        }
                                    }
                                }

                                // Anchored parameters
                                AnimatedVisibility(visible = driftMode == DriftMode.SEED_INTERPOLATION) {
                                    Column(modifier = Modifier.padding(top = 8.dp)) {
                                        Text("Alpha (\u03b1): ${(anchorStrength * 100).roundToInt()}%",
                                            style = MaterialTheme.typography.titleSmall)
                                        Text("Weight of the seed in the blend. " +
                                            "Low = latest pick dominates. High = seed dominates.",
                                            style = MaterialTheme.typography.bodySmall,
                                            color = MaterialTheme.colorScheme.onSurfaceVariant)
                                        Slider(value = anchorStrength, onValueChange = { viewModel.setAnchorStrength(it) },
                                            valueRange = 0f..1f)

                                        Text("Decay Schedule", style = MaterialTheme.typography.titleSmall)
                                        Text("Whether alpha weakens as the queue progresses.",
                                            style = MaterialTheme.typography.bodySmall,
                                            color = MaterialTheme.colorScheme.onSurfaceVariant)
                                        Spacer(modifier = Modifier.height(4.dp))
                                        Column(modifier = Modifier.selectableGroup()) {
                                            for (schedule in DecaySchedule.entries) {
                                                val (label, desc) = when (schedule) {
                                                    DecaySchedule.NONE -> "Constant" to "Same weight throughout"
                                                    DecaySchedule.LINEAR -> "Linear" to "Weakens steadily to zero by the end"
                                                    DecaySchedule.EXPONENTIAL -> "Exponential" to "Weakens quickly at first, then levels off"
                                                    DecaySchedule.STEP -> "Step" to "Full weight for the first half, then drops"
                                                }
                                                Row(modifier = Modifier.fillMaxWidth().selectable(
                                                    selected = anchorDecay == schedule,
                                                    onClick = { viewModel.setAnchorDecay(schedule) },
                                                    role = Role.RadioButton
                                                ).padding(vertical = 2.dp), verticalAlignment = Alignment.CenterVertically) {
                                                    RadioButton(selected = anchorDecay == schedule, onClick = null)
                                                    Spacer(modifier = Modifier.width(8.dp))
                                                    Column {
                                                        Text(label, style = MaterialTheme.typography.bodyMedium)
                                                        Text(desc, style = MaterialTheme.typography.bodySmall,
                                                            color = MaterialTheme.colorScheme.onSurfaceVariant)
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }

                                // Momentum parameters
                                AnimatedVisibility(visible = driftMode == DriftMode.MOMENTUM) {
                                    Column(modifier = Modifier.padding(top = 8.dp)) {
                                        Text("Beta (\u03b2): ${(momentumBeta * 100).roundToInt()}%",
                                            style = MaterialTheme.typography.titleSmall)
                                        Text("EMA smoothing factor. " +
                                            "High = averages over many past picks, direction changes slowly. " +
                                            "Low = reacts quickly to each new pick.",
                                            style = MaterialTheme.typography.bodySmall,
                                            color = MaterialTheme.colorScheme.onSurfaceVariant)
                                        Slider(value = momentumBeta, onValueChange = { viewModel.setMomentumBeta(it) },
                                            valueRange = 0.1f..0.95f)
                                    }
                                }
                            }
                        }
                    }
                }
            }

            item { HorizontalDivider() }

            // Artist constraints
            item {
                Text("Artist Limits", style = MaterialTheme.typography.titleMedium)
                Spacer(modifier = Modifier.height(4.dp))

                Text("Max Per Artist: $maxPerArtist", style = MaterialTheme.typography.titleSmall)
                Text("No artist will appear more than this many times in the queue.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
                Slider(value = maxPerArtist.toFloat(), onValueChange = { viewModel.setMaxPerArtist(it.roundToInt()) },
                    valueRange = 1f..10f, steps = 8)

                Text("Min Artist Spacing: $minArtistSpacing tracks", style = MaterialTheme.typography.titleSmall)
                Text("At least this many tracks between songs by the same artist.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
                Slider(value = minArtistSpacing.toFloat(), onValueChange = { viewModel.setMinArtistSpacing(it.roundToInt()) },
                    valueRange = 0f..20f, steps = 19)
            }

            item { HorizontalDivider() }

            // Database section
            item {
                Column {
                    Text("Database", style = MaterialTheme.typography.titleMedium)
                    Spacer(modifier = Modifier.height(8.dp))
                    if (databaseInfo != null) {
                        Text("${databaseInfo.trackCount} tracks, ${databaseInfo.embeddingCount} embeddings" +
                            (if (databaseInfo.embeddingDim != null) " (${databaseInfo.embeddingDim}d)" else ""))
                        Text("Table: ${databaseInfo.embeddingTable}")
                        if (databaseInfo.hasFused) Text("Fused embeddings: yes")
                        Text("kNN graph: ${if (databaseInfo.hasGraph) "yes \u2014 Random Walk available" else "not found \u2014 Random Walk will fall back to MMR"}")
                        Text("Size: ${databaseInfo.sizeKb} KB")
                    } else {
                        Text("No database imported", color = MaterialTheme.colorScheme.error)
                    }
                    Spacer(modifier = Modifier.height(8.dp))
                    OutlinedButton(onClick = onImportDatabase, modifier = Modifier.fillMaxWidth()) {
                        Text(if (databaseInfo != null) "Replace Database" else "Import Database")
                    }
                }
            }

            if (!hasPermission) {
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
                TextButton(onClick = { viewModel.resetToDefaults() }, modifier = Modifier.fillMaxWidth()) {
                    Text("Reset to Defaults")
                }
            }

            item { Spacer(modifier = Modifier.height(32.dp)) }
        }
    }
}

@Composable
private fun AlgorithmOption(
    label: String, description: String, selected: Boolean, onClick: () -> Unit,
    preview: List<String>? = null, isLoading: Boolean = false,
    expanded: Boolean = false, onToggleExpanded: () -> Unit = {},
    onRequestPreview: () -> Unit = {},
    previewInfo: String = ""
) {
    Row(modifier = Modifier.fillMaxWidth().selectable(selected = selected, onClick = onClick,
        role = Role.RadioButton).padding(vertical = 6.dp),
        verticalAlignment = Alignment.Top) {
        RadioButton(selected = selected, onClick = null, modifier = Modifier.padding(top = 2.dp))
        Spacer(modifier = Modifier.width(8.dp))
        Column {
            Text(label, style = MaterialTheme.typography.bodyMedium)
            Text(description, style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant)
            TextButton(
                onClick = {
                    if (!expanded && preview.isNullOrEmpty() && !isLoading) {
                        onRequestPreview()
                    }
                    onToggleExpanded()
                },
                contentPadding = PaddingValues(0.dp),
                modifier = Modifier.fillMaxWidth().height(28.dp)
            ) {
                val toggle = if (expanded) "Hide \u25B4" else "Peek \u25BE"
                val text = if (previewInfo.isEmpty()) toggle else "$toggle $previewInfo"
                Text(text, style = MaterialTheme.typography.labelSmall,
                    modifier = Modifier.fillMaxWidth())
            }
            AnimatedVisibility(visible = expanded) {
                if (isLoading) {
                    Box(modifier = Modifier.fillMaxWidth().padding(vertical = 8.dp),
                        contentAlignment = Alignment.CenterStart) {
                        CircularProgressIndicator(
                            modifier = Modifier.size(14.dp), strokeWidth = 1.5.dp,
                            color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.5f)
                        )
                    }
                } else if (!preview.isNullOrEmpty()) {
                    Column(
                        modifier = Modifier
                            .heightIn(max = 200.dp)
                            .verticalScroll(rememberScrollState())
                    ) {
                        preview.forEachIndexed { i, track ->
                            Text(
                                text = "${i + 1}. $track",
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.7f)
                            )
                        }
                    }
                }
            }
        }
    }
}

// ---- Human-friendly label helpers ----

private fun humanMatchType(matchType: TrackMatcher.MatchType): String = when (matchType) {
    TrackMatcher.MatchType.METADATA_EXACT -> "matched by tags"
    TrackMatcher.MatchType.FILENAME -> "matched by filename"
    TrackMatcher.MatchType.ARTIST_TITLE -> "fuzzy match"
    TrackMatcher.MatchType.NOT_FOUND -> "not found"
}

private fun humanSelectionMode(mode: SelectionMode, drift: Boolean = false): String {
    val base = when (mode) {
        SelectionMode.MMR -> "MMR"
        SelectionMode.DPP -> "DPP"
        SelectionMode.RANDOM_WALK -> "random walk"
        SelectionMode.TEMPERATURE -> "temperature"
    }
    return if (drift && mode != SelectionMode.DPP) "$base drift" else base
}
