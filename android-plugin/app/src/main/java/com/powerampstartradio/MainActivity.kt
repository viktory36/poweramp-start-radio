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
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.selection.selectableGroup
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
import androidx.compose.ui.graphics.lerp
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
    var treeNodes by remember { mutableStateOf(computeTreeNodes(session)) }
    LaunchedEffect(session) { treeNodes = computeTreeNodes(session) }

    val listState = rememberLazyListState()

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

        if (session.config.driftEnabled) {
            val scrollOffset = remember<() -> Float> {
                { (treeNodes.getOrNull(listState.firstVisibleItemIndex)?.depth ?: 0).toFloat() }
            }
            LazyColumn(state = listState, modifier = Modifier.fillMaxSize(),
                contentPadding = PaddingValues(horizontal = 16.dp, vertical = 4.dp)) {
                items(treeNodes.size) { index ->
                    TrackResultRow(trackResult = session.tracks[index],
                        treeNode = treeNodes[index], scrollOffset = scrollOffset)
                }
                if (!session.isComplete) {
                    item { StreamingProgressItem(found = session.tracks.size, total = session.totalExpected) }
                }
            }
        } else {
            LazyColumn(state = listState, modifier = Modifier.fillMaxSize(),
                contentPadding = PaddingValues(horizontal = 16.dp, vertical = 4.dp)) {
                items(treeNodes.size) { index ->
                    TrackResultRow(trackResult = session.tracks[index], treeNode = treeNodes[index])
                }
                if (!session.isComplete) {
                    item { StreamingProgressItem(found = session.tracks.size, total = session.totalExpected) }
                }
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

// ---- Tree node computation ----

private const val TREE_INDENT_DP = 5f
private const val MAX_TREE_DEPTH = 100

data class TreeNodeInfo(
    val depth: Int,
    val isLastChild: Boolean,
    val continuationDepths: Set<Int>,
    val connectChildDepths: Set<Int> = emptySet()
)

private fun computeTreeNodes(session: RadioResult): List<TreeNodeInfo> {
    if (session.tracks.isEmpty()) return emptyList()

    val rawNodes = if (session.config.driftEnabled) computeDriftChainNodes(session)
                   else computeFlatNodes(session)

    val result = mutableListOf<TreeNodeInfo>()
    val activeBranches = mutableSetOf<Int>()

    for (node in rawNodes) {
        activeBranches.removeAll { it >= node.depth }
        val continuations = activeBranches.filter { it < node.depth }.toSet()
        result.add(node.copy(continuationDepths = continuations))
        if (!node.isLastChild) activeBranches.add(node.depth)
    }

    for (i in 0 until result.lastIndex) {
        val current = result[i]
        val next = result[i + 1]
        if (next.depth > current.depth) {
            result[i] = current.copy(connectChildDepths = current.connectChildDepths + next.depth)
        }
    }

    return result
}

private fun computeFlatNodes(session: RadioResult): List<TreeNodeInfo> {
    return session.tracks.mapIndexed { index, _ ->
        TreeNodeInfo(depth = 0, isLastChild = index == session.tracks.lastIndex, continuationDepths = emptySet())
    }
}

private fun computeDriftChainNodes(session: RadioResult): List<TreeNodeInfo> {
    return session.tracks.mapIndexed { index, _ ->
        TreeNodeInfo(depth = minOf(index, MAX_TREE_DEPTH), isLastChild = true, continuationDepths = emptySet())
    }
}

// ---- Track Result Row with Canvas Tree Lines ----

@Composable
fun TrackResultRow(
    trackResult: QueuedTrackResult,
    treeNode: TreeNodeInfo? = null,
    scrollOffset: (() -> Float)? = null
) {
    Row(modifier = Modifier.fillMaxWidth().height(IntrinsicSize.Min),
        verticalAlignment = Alignment.CenterVertically) {
        if (treeNode != null) {
            TreeLines(node = treeNode, scrollOffset = scrollOffset, modifier = Modifier.fillMaxHeight())
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

@Composable
fun TreeLines(node: TreeNodeInfo, scrollOffset: (() -> Float)? = null, modifier: Modifier = Modifier) {
    val lineColor = MaterialTheme.colorScheme.outlineVariant
    val density = LocalDensity.current
    val indentPx = with(density) { TREE_INDENT_DP.dp.toPx() }
    val lineWidthPx = with(density) { 1.dp.toPx() }
    val fullLevels = if (node.connectChildDepths.isEmpty()) node.depth + 1
                     else maxOf(node.depth + 1, node.connectChildDepths.max() + 1)
    val offset = scrollOffset?.invoke() ?: 0f
    val visibleLevels = (fullLevels - offset).coerceAtLeast(0f)
    val canvasWidth = with(density) { (visibleLevels * TREE_INDENT_DP).dp }
    val scrollPx = offset * indentPx

    Canvas(modifier = modifier.width(canvasWidth).clipToBounds()) {
        val midY = size.height / 2

        for (d in node.continuationDepths) {
            val x = d * indentPx + indentPx / 2 - scrollPx
            drawLine(lineColor, Offset(x, 0f), Offset(x, size.height), lineWidthPx)
        }

        val junctionX = node.depth * indentPx + indentPx / 2 - scrollPx
        drawLine(lineColor, Offset(junctionX, 0f), Offset(junctionX, midY), lineWidthPx)
        if (!node.isLastChild) {
            drawLine(lineColor, Offset(junctionX, midY), Offset(junctionX, size.height), lineWidthPx)
        }
        drawLine(lineColor, Offset(junctionX, midY), Offset(size.width, midY), lineWidthPx)

        for (cd in node.connectChildDepths) {
            val childX = cd * indentPx + indentPx / 2 - scrollPx
            drawLine(lineColor, Offset(childX, midY), Offset(childX, size.height), lineWidthPx)
        }
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
    indexStatus: String?, onRequestPermission: () -> Unit, modifier: Modifier = Modifier
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
        if (hasPermission && databaseInfo != null) {
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

    val isRandomWalk = selectionMode == SelectionMode.RANDOM_WALK

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

                Column(modifier = Modifier.selectableGroup()) {
                    AlgorithmOption(
                        label = "MMR (Maximal Marginal Relevance)",
                        description = "Retrieves similar tracks, then re-ranks to penalize redundancy. " +
                            "Controls similarity vs. diversity tradeoff with a lambda parameter.",
                        selected = selectionMode == SelectionMode.MMR,
                        onClick = { viewModel.setSelectionMode(SelectionMode.MMR) }
                    )
                    AlgorithmOption(
                        label = "DPP (Determinantal Point Process)",
                        description = "Selects a subset that maximizes both relevance and pairwise diversity simultaneously. " +
                            "Uses greedy MAP inference with incremental Cholesky decomposition.",
                        selected = selectionMode == SelectionMode.DPP,
                        onClick = { viewModel.setSelectionMode(SelectionMode.DPP) }
                    )
                    AlgorithmOption(
                        label = "Random Walk (Personalized PageRank)",
                        description = "Runs personalized PageRank on a precomputed kNN graph. " +
                            "Finds tracks connected through transitive similarity chains. " +
                            "Requires kNN graph in database." +
                            if (databaseInfo?.hasGraph == true) "" else " (no graph found)",
                        selected = selectionMode == SelectionMode.RANDOM_WALK,
                        onClick = { viewModel.setSelectionMode(SelectionMode.RANDOM_WALK) }
                    )
                    AlgorithmOption(
                        label = "Temperature Sampling",
                        description = "Samples from top candidates using the Gumbel-max trick. " +
                            "Higher temperature = more randomness. Each run gives different results from the same seed.",
                        selected = selectionMode == SelectionMode.TEMPERATURE,
                        onClick = { viewModel.setSelectionMode(SelectionMode.TEMPERATURE) }
                    )
                }
            }

            // MMR lambda — only for MMR
            if (selectionMode == SelectionMode.MMR) {
                item {
                    Column {
                        Text("MMR Lambda: ${(diversityLambda * 100).roundToInt()}%",
                            style = MaterialTheme.typography.titleSmall)
                        Text("Tradeoff between relevance and diversity. " +
                            "0% = only diversity (ignore similarity). " +
                            "100% = only relevance (ignore redundancy).",
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
                        Text("Temperature: ${String.format("%.2f", temperature)}",
                            style = MaterialTheme.typography.titleSmall)
                        Text("Scales similarity scores before sampling. " +
                            "0 = deterministic (always picks the most similar). " +
                            "Higher = more uniform sampling across candidates.",
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
                        Text("Restart Probability: ${(anchorStrength * 100).roundToInt()}%",
                            style = MaterialTheme.typography.titleSmall)
                        Text("Probability of jumping back to the seed at each step. " +
                            "High = results stay close to seed. " +
                            "Low = explores further through the graph.",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant)
                        Slider(value = anchorStrength, onValueChange = { viewModel.setAnchorStrength(it) },
                            valueRange = 0.05f..0.95f)
                    }
                }
            }

            // Drift — not applicable to Random Walk
            if (!isRandomWalk) {
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
                                Text("When on, the query vector evolves after each selected track. " +
                                    "When off, all tracks are selected relative to the original seed embedding.",
                                    style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                            }
                        }

                        AnimatedVisibility(visible = driftEnabled) {
                            Column(modifier = Modifier.padding(start = 16.dp, top = 8.dp)) {
                                Text("Drift Method", style = MaterialTheme.typography.titleSmall)
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
                                            Text("query = normalize(\u03b1 \u00d7 seed + (1-\u03b1) \u00d7 latest). " +
                                                "Blends the original seed with the most recent pick.",
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
                                            Text("query = normalize(\u03b2 \u00d7 prev_query + (1-\u03b2) \u00d7 latest). " +
                                                "Exponential moving average of all previous embeddings.",
                                                style = MaterialTheme.typography.bodySmall,
                                                color = MaterialTheme.colorScheme.onSurfaceVariant)
                                        }
                                    }
                                }

                                // Seed Interpolation parameters
                                AnimatedVisibility(visible = driftMode == DriftMode.SEED_INTERPOLATION) {
                                    Column(modifier = Modifier.padding(top = 8.dp)) {
                                        Text("Anchor Strength (\u03b1): ${(anchorStrength * 100).roundToInt()}%",
                                            style = MaterialTheme.typography.titleSmall)
                                        Text("Weight of the original seed in the interpolation. " +
                                            "0% = query equals the latest pick. " +
                                            "100% = query always equals the seed (no drift).",
                                            style = MaterialTheme.typography.bodySmall,
                                            color = MaterialTheme.colorScheme.onSurfaceVariant)
                                        Slider(value = anchorStrength, onValueChange = { viewModel.setAnchorStrength(it) },
                                            valueRange = 0f..1f)

                                        Text("Anchor Decay Schedule", style = MaterialTheme.typography.titleSmall)
                                        Text("How \u03b1 changes over the course of the queue. " +
                                            "Controls whether the anchor loosens as tracks progress.",
                                            style = MaterialTheme.typography.bodySmall,
                                            color = MaterialTheme.colorScheme.onSurfaceVariant)
                                        Spacer(modifier = Modifier.height(4.dp))
                                        Column(modifier = Modifier.selectableGroup()) {
                                            for (schedule in DecaySchedule.entries) {
                                                val (label, desc) = when (schedule) {
                                                    DecaySchedule.NONE -> "None" to "\u03b1 stays constant"
                                                    DecaySchedule.LINEAR -> "Linear" to "\u03b1 decreases linearly to 0"
                                                    DecaySchedule.EXPONENTIAL -> "Exponential" to "\u03b1 decays as e^(-3t)"
                                                    DecaySchedule.STEP -> "Step" to "\u03b1 drops to 20% at the halfway point"
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

                                // EMA Momentum parameters
                                AnimatedVisibility(visible = driftMode == DriftMode.MOMENTUM) {
                                    Column(modifier = Modifier.padding(top = 8.dp)) {
                                        Text("Momentum (\u03b2): ${(momentumBeta * 100).roundToInt()}%",
                                            style = MaterialTheme.typography.titleSmall)
                                        Text("Weight of the previous query in the EMA. " +
                                            "High = query changes slowly (smooth trajectory). " +
                                            "Low = query tracks the latest pick closely.",
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
                Text("Artist Constraints", style = MaterialTheme.typography.titleMedium)
                Spacer(modifier = Modifier.height(4.dp))

                Text("Max Per Artist: $maxPerArtist", style = MaterialTheme.typography.titleSmall)
                Text("Maximum number of tracks by the same artist in the queue.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
                Slider(value = maxPerArtist.toFloat(), onValueChange = { viewModel.setMaxPerArtist(it.roundToInt()) },
                    valueRange = 1f..10f, steps = 8)

                Text("Min Artist Spacing: $minArtistSpacing", style = MaterialTheme.typography.titleSmall)
                Text("Minimum number of tracks between two tracks by the same artist.",
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
                        Text("kNN graph: ${if (databaseInfo.hasGraph) "yes" else "not found"}")
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

            item { Spacer(modifier = Modifier.height(32.dp)) }
        }
    }
}

@Composable
private fun AlgorithmOption(label: String, description: String, selected: Boolean, onClick: () -> Unit) {
    Row(modifier = Modifier.fillMaxWidth().selectable(selected = selected, onClick = onClick,
        role = Role.RadioButton).padding(vertical = 6.dp),
        verticalAlignment = Alignment.CenterVertically) {
        RadioButton(selected = selected, onClick = null)
        Spacer(modifier = Modifier.width(8.dp))
        Column {
            Text(label, style = MaterialTheme.typography.bodyMedium)
            Text(description, style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant)
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
    return if (drift) "$base drift" else base
}
