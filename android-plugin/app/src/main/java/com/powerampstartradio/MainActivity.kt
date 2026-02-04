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
import androidx.compose.animation.core.animateFloatAsState
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
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.lerp
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import androidx.lifecycle.viewmodel.compose.viewModel
import com.powerampstartradio.data.EmbeddingDatabase
import com.powerampstartradio.data.EmbeddingModel
import com.powerampstartradio.poweramp.PowerampHelper
import com.powerampstartradio.poweramp.PowerampReceiver
import com.powerampstartradio.poweramp.PowerampTrack
import com.powerampstartradio.poweramp.TrackMatcher
import com.powerampstartradio.services.RadioService
import com.powerampstartradio.similarity.SearchStrategy
import com.powerampstartradio.ui.DatabaseInfo
import com.powerampstartradio.ui.MainViewModel
import com.powerampstartradio.ui.QueueStatus
import com.powerampstartradio.ui.QueuedTrackResult
import com.powerampstartradio.ui.RadioResult
import com.powerampstartradio.ui.RadioUiState
import com.powerampstartradio.ui.theme.PowerampStartRadioTheme
import kotlinx.coroutines.launch
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.roundToInt

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
            this,
            trackReceiver,
            filter,
            ContextCompat.RECEIVER_EXPORTED
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

    // State from ViewModel
    val radioState by viewModel.radioState.collectAsState()
    val numTracks by viewModel.numTracks.collectAsState()
    val databaseInfo by viewModel.databaseInfo.collectAsState()
    val hasPermission by viewModel.hasPermission.collectAsState()
    val sessionHistory by viewModel.sessionHistory.collectAsState()
    val searchStrategy by viewModel.searchStrategy.collectAsState()
    val anchorExpandPrimary by viewModel.anchorExpandPrimary.collectAsState()
    val anchorExpandExpansion by viewModel.anchorExpandExpansion.collectAsState()
    val drift by viewModel.drift.collectAsState()
    val indexStatus by viewModel.indexStatus.collectAsState()

    // Local UI state
    var currentTrack by remember { mutableStateOf<PowerampTrack?>(PowerampReceiver.currentTrack) }
    var showSettings by remember { mutableStateOf(false) }
    var statusMessage by remember { mutableStateOf("") }
    var viewingSession by remember { mutableStateOf<Int?>(null) }

    // Clear viewed session when radio resets to idle (e.g. track change auto-reset)
    LaunchedEffect(radioState) {
        if (radioState is RadioUiState.Idle) {
            viewingSession = null
        }
    }

    // Drawer state
    val drawerState = rememberDrawerState(DrawerValue.Closed)
    val scope = rememberCoroutineScope()

    // Register resume callback
    LaunchedEffect(Unit) {
        onRegisterResumeCallback?.invoke {
            viewModel.checkPermission()
            viewModel.refreshDatabaseInfo()
        }
    }

    // Track change listener
    DisposableEffect(Unit) {
        val listener: (PowerampTrack?) -> Unit = { track ->
            currentTrack = track
        }
        PowerampReceiver.addTrackChangeListener(listener)
        onDispose {
            PowerampReceiver.removeTrackChangeListener(listener)
        }
    }

    // File picker launcher
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

    // Back button: settings -> home, viewing session -> home
    BackHandler(enabled = showSettings || viewingSession != null) {
        if (showSettings) {
            showSettings = false
        } else {
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
                SettingsScreen(
                    numTracks = numTracks,
                    onNumTracksChange = { viewModel.setNumTracks(it) },
                    searchStrategy = searchStrategy,
                    onSearchStrategyChange = { viewModel.setSearchStrategy(it) },
                    anchorExpandPrimary = anchorExpandPrimary,
                    onAnchorExpandPrimaryChange = { viewModel.setAnchorExpandPrimary(it) },
                    anchorExpandExpansion = anchorExpandExpansion,
                    onAnchorExpandExpansionChange = { viewModel.setAnchorExpandExpansion(it) },
                    drift = drift,
                    onDriftChange = { viewModel.setDrift(it) },
                    databaseInfo = databaseInfo,
                    onImportDatabase = {
                        importLauncher.launch(arrayOf("application/octet-stream", "*/*"))
                    },
                    hasPermission = hasPermission,
                    onRequestPermission = { viewModel.requestPermission() },
                    onBack = { showSettings = false }
                )
            } else {
                HomeScreen(
                    radioState = radioState,
                    currentTrack = currentTrack,
                    databaseInfo = databaseInfo,
                    hasPermission = hasPermission,
                    sessionHistory = sessionHistory,
                    statusMessage = statusMessage,
                    indexStatus = indexStatus,
                    onStartRadio = {
                        if (currentTrack != null && databaseInfo != null) {
                            viewModel.startRadio()
                        } else if (currentTrack == null) {
                            statusMessage = "Play a song in Poweramp first"
                        } else {
                            statusMessage = "Import database in Settings"
                        }
                    },
                    onClearAndReset = {
                        viewModel.resetRadioState()
                    },
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
    onClearAndReset: () -> Unit,
    onRequestPermission: () -> Unit,
    onOpenSettings: () -> Unit,
    onOpenDrawer: () -> Unit,
    viewingSession: Int?,
    onViewSession: (Int?) -> Unit
) {
    val showResults = radioState is RadioUiState.Success || viewingSession != null
    val displaySession = if (viewingSession != null && viewingSession in sessionHistory.indices) {
        sessionHistory[viewingSession]
    } else {
        sessionHistory.lastOrNull()
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
            ExtendedFloatingActionButton(
                onClick = onStartRadio,
                icon = { Icon(Icons.Default.PlayArrow, contentDescription = null) },
                text = { Text("Start Radio") },
                expanded = radioState !is RadioUiState.Loading
            )
        }
    ) { padding ->
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
        ) {
            Column(modifier = Modifier.fillMaxSize()) {
                // Compact header
                if (showResults && displaySession != null) {
                    CompactSeedHeader(
                        session = displaySession,
                        modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp)
                    )
                } else {
                    CompactNowPlayingHeader(
                        currentTrack = currentTrack,
                        modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp)
                    )
                }

                HorizontalDivider()

                // Main content area
                Box(modifier = Modifier.weight(1f)) {
                    if (showResults && displaySession != null) {
                        SessionPage(
                            session = displaySession,
                            modifier = Modifier.fillMaxSize()
                        )

                        // Error overlay on top of results
                        if (radioState is RadioUiState.Error) {
                            Box(
                                modifier = Modifier
                                    .fillMaxSize()
                                    .background(MaterialTheme.colorScheme.surface.copy(alpha = 0.7f)),
                                contentAlignment = Alignment.Center
                            ) {
                                Card(
                                    colors = CardDefaults.cardColors(
                                        containerColor = MaterialTheme.colorScheme.errorContainer
                                    )
                                ) {
                                    Text(
                                        text = (radioState as RadioUiState.Error).message,
                                        modifier = Modifier.padding(16.dp),
                                        color = MaterialTheme.colorScheme.onErrorContainer
                                    )
                                }
                            }
                        }
                    } else {
                        when (val state = radioState) {
                            is RadioUiState.Idle -> {
                                IdleContent(
                                    hasPermission = hasPermission,
                                    databaseInfo = databaseInfo,
                                    statusMessage = statusMessage,
                                    indexStatus = indexStatus,
                                    onRequestPermission = onRequestPermission,
                                    modifier = Modifier.fillMaxSize()
                                )
                            }
                            is RadioUiState.Error -> {
                                Box(
                                    modifier = Modifier
                                        .fillMaxSize()
                                        .padding(16.dp),
                                    contentAlignment = Alignment.Center
                                ) {
                                    Card(
                                        colors = CardDefaults.cardColors(
                                            containerColor = MaterialTheme.colorScheme.errorContainer
                                        )
                                    ) {
                                        Text(
                                            text = state.message,
                                            modifier = Modifier.padding(16.dp),
                                            color = MaterialTheme.colorScheme.onErrorContainer
                                        )
                                    }
                                }
                            }
                            is RadioUiState.Loading, is RadioUiState.Success -> {}
                        }
                    }
                }
            }

            // Loading overlay
            if (radioState is RadioUiState.Loading) {
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .background(MaterialTheme.colorScheme.surface.copy(alpha = 0.85f)),
                    contentAlignment = Alignment.Center
                ) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        CircularProgressIndicator()
                        Spacer(modifier = Modifier.height(12.dp))
                        Text(
                            text = (radioState as RadioUiState.Loading).message,
                            style = MaterialTheme.typography.bodyMedium
                        )
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
    modifier: Modifier = Modifier
) {
    if (currentTrack != null) {
        Column(modifier = modifier) {
            Text(
                text = "NOW PLAYING",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.primary
            )
            Text(
                text = currentTrack.title ?: "Unknown",
                style = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.Bold,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
            Text(
                text = listOfNotNull(currentTrack.artist, currentTrack.album)
                    .joinToString(" \u00b7 "),
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
        }
    } else {
        Text(
            text = "No track playing",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = modifier
        )
    }
}

@Composable
fun CompactSeedHeader(
    session: RadioResult,
    modifier: Modifier = Modifier
) {
    Row(
        modifier = modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Column(modifier = Modifier.weight(1f)) {
            Text(
                text = session.seedTrack.title ?: "Unknown",
                style = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.Bold,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
            Text(
                text = listOfNotNull(session.seedTrack.artist, session.seedTrack.album)
                    .joinToString(" \u00b7 "),
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
        }
        Spacer(modifier = Modifier.width(8.dp))
        Text(
            text = humanMatchType(session.matchType),
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
    }
}

// ---- Session Page ----

@Composable
fun SessionPage(
    session: RadioResult,
    modifier: Modifier = Modifier
) {
    val treeNodes = remember(session) { computeTreeNodes(session) }
    val listState = rememberLazyListState()

    Column(modifier = modifier) {
        ResultsSummary(
            result = session,
            modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp)
        )

        if (session.drift) {
            // Drift: diagonal scroll — widen the list so deeper items still have
            // full content width, then offset left based on first visible depth.
            // Max canvas levels accounts for connectChildDepths (wider than node depth alone)
            val maxCanvasLevels = remember(treeNodes) {
                treeNodes.maxOfOrNull { node ->
                    if (node.connectChildDepths.isEmpty()) node.depth + 1
                    else maxOf(node.depth + 1, node.connectChildDepths.max() + 1)
                } ?: 1
            }
            val targetDepth by remember {
                derivedStateOf {
                    treeNodes.getOrNull(listState.firstVisibleItemIndex)?.depth ?: 0
                }
            }
            val animatedDepth by animateFloatAsState(
                targetValue = targetDepth.toFloat(),
                label = "diagonal_scroll"
            )
            val density = LocalDensity.current
            val indentPxPerLevel = remember(density) { with(density) { TREE_INDENT_DP.dp.toPx() } }
            val extraWidthDp = (maxCanvasLevels * TREE_INDENT_DP).dp

            BoxWithConstraints(modifier = Modifier.fillMaxSize()) {
                LazyColumn(
                    state = listState,
                    modifier = Modifier
                        .width(maxWidth + extraWidthDp)
                        .offset {
                            IntOffset(
                                x = -(animatedDepth * indentPxPerLevel).roundToInt(),
                                y = 0
                            )
                        },
                    contentPadding = PaddingValues(horizontal = 16.dp, vertical = 4.dp)
                ) {
                    items(treeNodes.size) { index ->
                        TrackResultRow(
                            trackResult = session.tracks[index],
                            treeNode = treeNodes[index]
                        )
                    }
                }
            }
        } else {
            LazyColumn(
                state = listState,
                modifier = Modifier.fillMaxSize(),
                contentPadding = PaddingValues(horizontal = 16.dp, vertical = 4.dp)
            ) {
                items(treeNodes.size) { index ->
                    TrackResultRow(
                        trackResult = session.tracks[index],
                        treeNode = treeNodes[index]
                    )
                }
            }
        }
    }
}

@Composable
fun ResultsSummary(
    result: RadioResult,
    modifier: Modifier = Modifier
) {
    val strategyLabel = humanStrategy(result.strategy, result.drift)
    val seedName = result.seedTrack.title ?: "Unknown"
    val baseColor = MaterialTheme.colorScheme.onSurfaceVariant

    val countText = if (result.failedCount > 0) {
        "$seedName \u2014 ${result.queuedCount} of ${result.requestedCount} queued (${result.failedCount} not found)"
    } else {
        "$seedName \u2014 ${result.queuedCount} tracks via $strategyLabel"
    }

    if (result.isMultiModel) {
        val models = result.tracks.mapNotNull { it.modelUsed }.distinct()
        val separator = if (result.strategy == SearchStrategy.ANCHOR_EXPAND) " \u2192 " else " \u00b7 "
        val modelColors = models.associateWith { model ->
            when (model) {
                EmbeddingModel.FLAMINGO -> MaterialTheme.colorScheme.secondary
                EmbeddingModel.MULAN -> MaterialTheme.colorScheme.tertiary
                EmbeddingModel.MUQ -> MaterialTheme.colorScheme.primary
            }
        }
        val annotated = buildAnnotatedString {
            append(countText)
            append(" (")
            var first = true
            for ((model, color) in modelColors) {
                if (!first) append(separator)
                first = false
                val tag = when (model) {
                    EmbeddingModel.FLAMINGO -> "flam"
                    EmbeddingModel.MULAN -> "mulan"
                    EmbeddingModel.MUQ -> "muq"
                }
                withStyle(SpanStyle(color = color, fontFamily = FontFamily.Monospace, fontSize = 10.sp)) {
                    append(tag)
                }
            }
            append(")")
        }
        Text(
            text = annotated,
            style = MaterialTheme.typography.labelMedium,
            color = baseColor,
            modifier = modifier.fillMaxWidth()
        )
    } else {
        Text(
            text = countText,
            style = MaterialTheme.typography.labelMedium,
            color = baseColor,
            modifier = modifier.fillMaxWidth()
        )
    }
}

// ---- Tree node computation for all strategies ----

private const val TREE_INDENT_DP = 10f
private const val MAX_TREE_DEPTH = 20

/**
 * Describes the tree position of a single track for Canvas-based rendering.
 *
 * @param depth Indent level (0 = leftmost)
 * @param isLastChild Whether this is the last sibling — controls ├ vs └
 * @param continuationDepths Depths where a vertical line passes through this row
 * @param connectChildDepths Depths where a vertical drop is drawn from midY to bottom,
 *                           connecting this parent row to its child rows below
 */
data class TreeNodeInfo(
    val depth: Int,
    val isLastChild: Boolean,
    val continuationDepths: Set<Int>,
    val connectChildDepths: Set<Int> = emptySet()
)

/**
 * Computes tree nodes for every track in a session.
 * All strategies get tree lines; drift modes produce progressively deeper chains.
 */
private fun computeTreeNodes(session: RadioResult): List<TreeNodeInfo> {
    if (session.tracks.isEmpty()) return emptyList()

    // Drift A&E uses a two-lane structure (drift lane + expansion lane) and
    // computes fully-resolved nodes directly — skip the generic passes.
    if (session.strategy == SearchStrategy.ANCHOR_EXPAND && session.drift) {
        return computeAnchorExpandDriftNodes(session)
    }

    val rawNodes = when {
        session.strategy == SearchStrategy.ANCHOR_EXPAND -> computeAnchorExpandNodes(session)
        session.drift -> computeDriftChainNodes(session)
        else -> computeFlatNodes(session)
    }

    // Forward pass: compute continuation lines
    val result = mutableListOf<TreeNodeInfo>()
    val activeBranches = mutableSetOf<Int>()

    for (node in rawNodes) {
        activeBranches.removeAll { it >= node.depth }
        val continuations = activeBranches.filter { it < node.depth }.toSet()
        result.add(node.copy(continuationDepths = continuations))
        if (!node.isLastChild) {
            activeBranches.add(node.depth)
        }
    }

    // Second pass: add parent-to-child drop connections
    for (i in 0 until result.lastIndex) {
        val current = result[i]
        val next = result[i + 1]
        if (next.depth > current.depth) {
            result[i] = current.copy(connectChildDepths = current.connectChildDepths + next.depth)
        }
    }

    return result
}

/** Non-drift single/interleave: flat list at depth 0 */
private fun computeFlatNodes(session: RadioResult): List<TreeNodeInfo> {
    return session.tracks.mapIndexed { index, _ ->
        TreeNodeInfo(
            depth = 0,
            isLastChild = index == session.tracks.lastIndex,
            continuationDepths = emptySet()
        )
    }
}

/** Drift chain: each track seeds the next — each is the only child, so all └ */
private fun computeDriftChainNodes(session: RadioResult): List<TreeNodeInfo> {
    return session.tracks.mapIndexed { index, _ ->
        TreeNodeInfo(
            depth = minOf(index, MAX_TREE_DEPTH),
            isLastChild = true,
            continuationDepths = emptySet()
        )
    }
}

/** Non-drift A&E: anchors are siblings at d=0, expansions at d=1 under each. */
private fun computeAnchorExpandNodes(session: RadioResult): List<TreeNodeInfo> {
    val groups = groupAnchorExpand(session)
    val nodes = mutableListOf<TreeNodeInfo>()

    for ((gi, group) in groups.withIndex()) {
        val isLastGroup = gi == groups.lastIndex
        nodes.add(TreeNodeInfo(depth = 0, isLastChild = isLastGroup, continuationDepths = emptySet()))
        for ((ei, _) in group.expansionIndices.withIndex()) {
            val isLastExp = ei == group.expansionIndices.lastIndex
            nodes.add(TreeNodeInfo(depth = 1, isLastChild = isLastExp, continuationDepths = emptySet()))
        }
    }

    return nodes
}

/**
 * Drift A&E with two-lane structure. Each anchor has two vertical drops:
 *   - drift lane (d = anchor+1): passes through expansions, terminates at next anchor
 *   - expansion lane (d = anchor+2): connects expansions with ├/└
 *
 * Returns fully-computed nodes (continuationDepths and connectChildDepths set).
 */
private fun computeAnchorExpandDriftNodes(session: RadioResult): List<TreeNodeInfo> {
    val groups = groupAnchorExpand(session)
    val nodes = mutableListOf<TreeNodeInfo>()

    for ((gi, group) in groups.withIndex()) {
        val anchorDepth = minOf(gi, MAX_TREE_DEPTH)
        val driftLaneDepth = anchorDepth + 1
        val expDepth = anchorDepth + 2
        val isLastGroup = gi == groups.lastIndex
        val hasExpansions = group.expansionIndices.isNotEmpty()

        // Anchor: └ at its depth, drops to drift lane and expansion lane
        val drops = mutableSetOf<Int>()
        if (hasExpansions) drops.add(expDepth)
        if (!isLastGroup) drops.add(driftLaneDepth)
        nodes.add(TreeNodeInfo(
            depth = anchorDepth,
            isLastChild = true,
            continuationDepths = emptySet(),
            connectChildDepths = drops
        ))

        // Expansions: junction at expDepth, drift lane continuation passes through
        val driftCont = if (!isLastGroup) setOf(driftLaneDepth) else emptySet()
        for ((ei, _) in group.expansionIndices.withIndex()) {
            val isLastExp = ei == group.expansionIndices.lastIndex
            nodes.add(TreeNodeInfo(
                depth = expDepth,
                isLastChild = isLastExp,
                continuationDepths = driftCont
            ))
        }
    }

    return nodes
}

/** Helper: partition tracks into [anchor, exp, exp, ...] groups. */
private data class AnchorGroup(val anchorIndex: Int, val expansionIndices: MutableList<Int> = mutableListOf())

private fun groupAnchorExpand(session: RadioResult): List<AnchorGroup> {
    val primaryModel = session.tracks[0].modelUsed
    val groups = mutableListOf<AnchorGroup>()
    for (i in session.tracks.indices) {
        if (session.tracks[i].modelUsed == primaryModel) {
            groups.add(AnchorGroup(i))
        } else {
            groups.lastOrNull()?.expansionIndices?.add(i)
        }
    }
    return groups
}

// ---- Track Result Row with Canvas Tree Lines ----

@Composable
fun TrackResultRow(
    trackResult: QueuedTrackResult,
    treeNode: TreeNodeInfo? = null
) {
    // No vertical padding on outer Row so tree lines are continuous between rows.
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .height(IntrinsicSize.Min),
        verticalAlignment = Alignment.CenterVertically
    ) {
        // Canvas tree lines — spans full row height for continuous lines
        if (treeNode != null) {
            TreeLines(
                node = treeNode,
                modifier = Modifier.fillMaxHeight()
            )
        }

        // Track info (takes remaining space, padded for breathing room)
        Column(
            modifier = Modifier
                .weight(1f)
                .padding(vertical = 2.dp, horizontal = 4.dp)
        ) {
            Text(
                text = trackResult.track.title ?: "Unknown",
                style = MaterialTheme.typography.bodyMedium,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
            Text(
                text = trackResult.track.artist ?: "Unknown",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
        }

        // Compact score — right-aligned in model color gradient
        val scoreFloor = when (trackResult.modelUsed) {
            EmbeddingModel.FLAMINGO -> 0.5f
            else -> 0.0f
        }
        val normalized = ((trackResult.similarity - scoreFloor) / (1f - scoreFloor)).coerceIn(0f, 1f)
        val vivid = when (trackResult.modelUsed) {
            EmbeddingModel.FLAMINGO -> MaterialTheme.colorScheme.secondary
            EmbeddingModel.MULAN -> MaterialTheme.colorScheme.tertiary
            EmbeddingModel.MUQ -> MaterialTheme.colorScheme.primary
            null -> MaterialTheme.colorScheme.onSurfaceVariant
        }
        val scoreColor = lerp(vivid.copy(alpha = 0.15f), vivid, normalized)
        val scoreText = String.format("%.2f", trackResult.similarity).removePrefix("0")

        Text(
            text = "($scoreText)",
            fontFamily = FontFamily.Monospace,
            color = scoreColor,
            fontSize = 9.sp,
            textAlign = TextAlign.End,
            modifier = Modifier.padding(horizontal = 2.dp)
        )

        // Status icon
        val statusColor = when (trackResult.status) {
            QueueStatus.QUEUED -> MaterialTheme.colorScheme.primary
            QueueStatus.NOT_IN_LIBRARY -> MaterialTheme.colorScheme.error
            QueueStatus.QUEUE_FAILED -> MaterialTheme.colorScheme.error
        }
        Text(
            text = if (trackResult.status == QueueStatus.QUEUED) "\u2713" else "\u2717",
            style = MaterialTheme.typography.bodySmall,
            color = statusColor,
            modifier = Modifier.padding(end = 2.dp)
        )
    }
}

/** Canvas-based tree line rendering — draws connected vertical and horizontal lines. */
@Composable
fun TreeLines(
    node: TreeNodeInfo,
    modifier: Modifier = Modifier
) {
    val lineColor = MaterialTheme.colorScheme.outlineVariant
    val density = LocalDensity.current
    val indentPx = with(density) { TREE_INDENT_DP.dp.toPx() }
    val lineWidthPx = with(density) { 1.dp.toPx() }
    // Size canvas to fit junction + any connectChild drop lines
    val levels = if (node.connectChildDepths.isEmpty()) node.depth + 1
                 else maxOf(node.depth + 1, node.connectChildDepths.max() + 1)
    val canvasWidth = with(density) { (levels * TREE_INDENT_DP).dp }

    Canvas(
        modifier = modifier.width(canvasWidth)
    ) {
        val midY = size.height / 2

        // Continuation vertical lines at each active depth
        for (d in node.continuationDepths) {
            val x = d * indentPx + indentPx / 2
            drawLine(
                color = lineColor,
                start = Offset(x, 0f),
                end = Offset(x, size.height),
                strokeWidth = lineWidthPx
            )
        }

        // Junction at this node's depth
        val junctionX = node.depth * indentPx + indentPx / 2

        // Vertical: top to middle (connecting from above)
        drawLine(
            color = lineColor,
            start = Offset(junctionX, 0f),
            end = Offset(junctionX, midY),
            strokeWidth = lineWidthPx
        )

        // Vertical: middle to bottom (if more siblings follow)
        if (!node.isLastChild) {
            drawLine(
                color = lineColor,
                start = Offset(junctionX, midY),
                end = Offset(junctionX, size.height),
                strokeWidth = lineWidthPx
            )
        }

        // Horizontal: junction to content
        drawLine(
            color = lineColor,
            start = Offset(junctionX, midY),
            end = Offset(size.width, midY),
            strokeWidth = lineWidthPx
        )

        // Parent-to-child drops: vertical lines from midY to bottom at each child depth
        for (cd in node.connectChildDepths) {
            val childX = cd * indentPx + indentPx / 2
            drawLine(
                color = lineColor,
                start = Offset(childX, midY),
                end = Offset(childX, size.height),
                strokeWidth = lineWidthPx
            )
        }
    }
}

// ---- Session History Drawer ----

@Composable
fun SessionHistoryDrawer(
    sessions: List<RadioResult>,
    onSessionTap: (Int) -> Unit,
    onClear: () -> Unit
) {
    Column(modifier = Modifier.fillMaxHeight()) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = "Session History",
                style = MaterialTheme.typography.titleMedium
            )
            if (sessions.isNotEmpty()) {
                TextButton(onClick = onClear) {
                    Text("Clear")
                }
            }
        }
        HorizontalDivider()

        if (sessions.isEmpty()) {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(32.dp),
                contentAlignment = Alignment.Center
            ) {
                Text(
                    text = "No sessions yet",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        } else {
            LazyColumn(
                modifier = Modifier.weight(1f),
                contentPadding = PaddingValues(vertical = 4.dp)
            ) {
                items(sessions.size) { index ->
                    val session = sessions[index]
                    val timeStr = SimpleDateFormat("HH:mm", Locale.getDefault())
                        .format(Date(session.timestamp))
                    NavigationDrawerItem(
                        label = {
                            Column {
                                Text(
                                    text = session.seedTrack.title ?: "Unknown",
                                    style = MaterialTheme.typography.bodyMedium,
                                    maxLines = 1,
                                    overflow = TextOverflow.Ellipsis
                                )
                                Text(
                                    text = "${session.seedTrack.artist ?: "Unknown"} \u00b7 $timeStr \u00b7 ${session.queuedCount} tracks",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                                    maxLines = 1,
                                    overflow = TextOverflow.Ellipsis
                                )
                            }
                        },
                        selected = false,
                        onClick = { onSessionTap(index) },
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
    hasPermission: Boolean,
    databaseInfo: DatabaseInfo?,
    statusMessage: String,
    indexStatus: String?,
    onRequestPermission: () -> Unit,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier
            .fillMaxWidth()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        // Permission warning
        if (!hasPermission) {
            Card(
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.errorContainer
                )
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text(
                        text = "Poweramp Access Required",
                        style = MaterialTheme.typography.titleSmall
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    Button(onClick = onRequestPermission) {
                        Text("Grant Access")
                    }
                }
            }
        }

        // Database warning
        if (databaseInfo == null) {
            Card(
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.secondaryContainer
                )
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text(
                        text = "No embedding database",
                        style = MaterialTheme.typography.titleSmall
                    )
                    Text(
                        text = "Import via Settings",
                        style = MaterialTheme.typography.bodySmall
                    )
                }
            }
        }

        // Index extraction progress (hidden once done)
        if (indexStatus != null && indexStatus != "Indices ready") {
            Text(
                text = indexStatus,
                style = MaterialTheme.typography.bodySmall,
                fontFamily = FontFamily.Monospace,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }

        // Status message
        if (statusMessage.isNotEmpty()) {
            Text(
                text = statusMessage,
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }

        // Ready indicator
        if (hasPermission && databaseInfo != null) {
            Text(
                text = "Ready when you are!",
                style = MaterialTheme.typography.bodyLarge,
                color = MaterialTheme.colorScheme.primary
            )
        }
    }
}

// ---- Full-Screen Settings ----

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(
    numTracks: Int,
    onNumTracksChange: (Int) -> Unit,
    searchStrategy: SearchStrategy,
    onSearchStrategyChange: (SearchStrategy) -> Unit,
    anchorExpandPrimary: EmbeddingModel,
    onAnchorExpandPrimaryChange: (EmbeddingModel) -> Unit,
    anchorExpandExpansion: Int,
    onAnchorExpandExpansionChange: (Int) -> Unit,
    drift: Boolean,
    onDriftChange: (Boolean) -> Unit,
    databaseInfo: DatabaseInfo?,
    onImportDatabase: () -> Unit,
    hasPermission: Boolean,
    onRequestPermission: () -> Unit,
    onBack: () -> Unit
) {
    val availableModels = databaseInfo?.availableModels ?: emptySet()
    val hasMulan = EmbeddingModel.MULAN in availableModels
    val hasFlamingo = EmbeddingModel.FLAMINGO in availableModels
    val hasBoth = hasMulan && hasFlamingo

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
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(horizontal = 16.dp),
            verticalArrangement = Arrangement.spacedBy(24.dp)
        ) {
            // Track count slider
            item {
                Column {
                    Text(
                        text = "Number of tracks: $numTracks",
                        style = MaterialTheme.typography.titleMedium
                    )
                    Slider(
                        value = numTracks.toFloat(),
                        onValueChange = { onNumTracksChange(it.toInt()) },
                        valueRange = 10f..100f,
                        steps = 8
                    )
                }
            }

            item { HorizontalDivider() }

            // Search Strategy section
            item {
                Column {
                    Text(
                        text = "Search Strategy",
                        style = MaterialTheme.typography.titleMedium
                    )
                    Spacer(modifier = Modifier.height(8.dp))

                    Column(modifier = Modifier.selectableGroup()) {
                        StrategyOption(
                            label = "MuLan Only",
                            description = "Single-model search with MuLan embeddings",
                            selected = searchStrategy == SearchStrategy.MULAN_ONLY,
                            enabled = hasMulan,
                            disabledReason = if (!hasMulan) "No MuLan embeddings in database" else null,
                            onClick = { onSearchStrategyChange(SearchStrategy.MULAN_ONLY) }
                        )
                        StrategyOption(
                            label = "Flamingo Only",
                            description = "Single-model search with Flamingo embeddings",
                            selected = searchStrategy == SearchStrategy.FLAMINGO_ONLY,
                            enabled = hasFlamingo,
                            disabledReason = if (!hasFlamingo) "No Flamingo embeddings in database" else null,
                            onClick = { onSearchStrategyChange(SearchStrategy.FLAMINGO_ONLY) }
                        )
                        StrategyOption(
                            label = "Interleave",
                            description = "Alternate results from both models",
                            selected = searchStrategy == SearchStrategy.INTERLEAVE,
                            enabled = hasBoth,
                            disabledReason = if (!hasBoth) "Requires both MuLan and Flamingo" else null,
                            onClick = { onSearchStrategyChange(SearchStrategy.INTERLEAVE) }
                        )
                        StrategyOption(
                            label = "Anchor & Expand",
                            description = "One model selects seeds, the other expands on each",
                            selected = searchStrategy == SearchStrategy.ANCHOR_EXPAND,
                            enabled = hasBoth,
                            disabledReason = if (!hasBoth) "Requires both MuLan and Flamingo" else null,
                            onClick = { onSearchStrategyChange(SearchStrategy.ANCHOR_EXPAND) }
                        )
                    }

                    // Anchor & Expand sub-options
                    AnimatedVisibility(visible = searchStrategy == SearchStrategy.ANCHOR_EXPAND && hasBoth) {
                        Column(
                            modifier = Modifier.padding(start = 16.dp, top = 12.dp)
                        ) {
                            Text(
                                text = "Primary Model",
                                style = MaterialTheme.typography.titleSmall
                            )
                            Spacer(modifier = Modifier.height(4.dp))
                            Column(modifier = Modifier.selectableGroup()) {
                                Row(
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .selectable(
                                            selected = anchorExpandPrimary == EmbeddingModel.MULAN,
                                            onClick = { onAnchorExpandPrimaryChange(EmbeddingModel.MULAN) },
                                            role = Role.RadioButton
                                        )
                                        .padding(vertical = 4.dp),
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    RadioButton(
                                        selected = anchorExpandPrimary == EmbeddingModel.MULAN,
                                        onClick = null
                                    )
                                    Spacer(modifier = Modifier.width(8.dp))
                                    Text("MuLan -> Flamingo", style = MaterialTheme.typography.bodyMedium)
                                }
                                Row(
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .selectable(
                                            selected = anchorExpandPrimary == EmbeddingModel.FLAMINGO,
                                            onClick = { onAnchorExpandPrimaryChange(EmbeddingModel.FLAMINGO) },
                                            role = Role.RadioButton
                                        )
                                        .padding(vertical = 4.dp),
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    RadioButton(
                                        selected = anchorExpandPrimary == EmbeddingModel.FLAMINGO,
                                        onClick = null
                                    )
                                    Spacer(modifier = Modifier.width(8.dp))
                                    Text("Flamingo -> MuLan", style = MaterialTheme.typography.bodyMedium)
                                }
                            }

                            Spacer(modifier = Modifier.height(12.dp))
                            Text(
                                text = "Expansions per anchor: $anchorExpandExpansion",
                                style = MaterialTheme.typography.titleSmall
                            )
                            Slider(
                                value = anchorExpandExpansion.toFloat(),
                                onValueChange = { onAnchorExpandExpansionChange(it.toInt()) },
                                valueRange = 1f..5f,
                                steps = 3
                            )
                        }
                    }

                    // Drift checkbox
                    Spacer(modifier = Modifier.height(12.dp))
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .selectable(
                                selected = drift,
                                onClick = { onDriftChange(!drift) },
                                role = Role.Checkbox
                            )
                            .padding(vertical = 4.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Checkbox(
                            checked = drift,
                            onCheckedChange = null
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Column {
                            Text(
                                text = "Drift ahead",
                                style = MaterialTheme.typography.bodyMedium
                            )
                            Text(
                                text = "Each result seeds the next search",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                    }
                }
            }

            item { HorizontalDivider() }

            // Database section
            item {
                Column {
                    Text(
                        text = "Embedding Database",
                        style = MaterialTheme.typography.titleMedium
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    if (databaseInfo != null) {
                        Text("Tracks: ${databaseInfo.trackCount}")
                        Text("Version: ${databaseInfo.version ?: "Unknown"}")
                        Text("Size: ${databaseInfo.sizeKb} KB")
                        if (databaseInfo.availableModels.isNotEmpty()) {
                            val modelNames = databaseInfo.availableModels.joinToString(", ") {
                                when (it) {
                                    EmbeddingModel.MUQ -> "MuQ"
                                    EmbeddingModel.MULAN -> "MuLan"
                                    EmbeddingModel.FLAMINGO -> "Flamingo"
                                }
                            }
                            Text("Models: $modelNames")
                        }
                    } else {
                        Text(
                            text = "No database imported",
                            color = MaterialTheme.colorScheme.error
                        )
                    }
                    Spacer(modifier = Modifier.height(8.dp))
                    OutlinedButton(
                        onClick = onImportDatabase,
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Text(if (databaseInfo != null) "Replace Database" else "Import Database")
                    }
                }
            }

            // Poweramp permission
            if (!hasPermission) {
                item { HorizontalDivider() }
                item {
                    Column {
                        Text(
                            text = "Poweramp Access",
                            style = MaterialTheme.typography.titleMedium
                        )
                        Spacer(modifier = Modifier.height(8.dp))
                        Button(
                            onClick = onRequestPermission,
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Text("Grant Poweramp Access")
                        }
                    }
                }
            }

            item { HorizontalDivider() }

            // Instructions
            item {
                Column {
                    Text(
                        text = "How to Use",
                        style = MaterialTheme.typography.titleMedium
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    Text("1. Run desktop indexer on your music library", style = MaterialTheme.typography.bodySmall)
                    Text("2. Copy embeddings.db to your phone", style = MaterialTheme.typography.bodySmall)
                    Text("3. Import the database above", style = MaterialTheme.typography.bodySmall)
                    Text("4. Play a song in Poweramp", style = MaterialTheme.typography.bodySmall)
                    Text("5. Tap Start Radio", style = MaterialTheme.typography.bodySmall)
                }
            }

            item { Spacer(modifier = Modifier.height(32.dp)) }
        }
    }
}

@Composable
private fun StrategyOption(
    label: String,
    description: String,
    selected: Boolean,
    enabled: Boolean,
    disabledReason: String?,
    onClick: () -> Unit
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .selectable(
                selected = selected,
                enabled = enabled,
                onClick = onClick,
                role = Role.RadioButton
            )
            .padding(vertical = 6.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        RadioButton(
            selected = selected,
            onClick = null,
            enabled = enabled
        )
        Spacer(modifier = Modifier.width(8.dp))
        Column {
            Text(
                text = label,
                style = MaterialTheme.typography.bodyMedium,
                color = if (enabled) MaterialTheme.colorScheme.onSurface
                       else MaterialTheme.colorScheme.onSurface.copy(alpha = 0.38f)
            )
            if (!enabled && disabledReason != null) {
                Text(
                    text = disabledReason,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.5f)
                )
            } else {
                Text(
                    text = description,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
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

private fun humanStrategy(strategy: SearchStrategy, drift: Boolean = false): String {
    val base = when (strategy) {
        SearchStrategy.MULAN_ONLY -> "MuLan"
        SearchStrategy.FLAMINGO_ONLY -> "Flamingo"
        SearchStrategy.INTERLEAVE -> "interleave"
        SearchStrategy.ANCHOR_EXPAND -> "anchor & expand"
    }
    return if (drift) "$base drift" else base
}
