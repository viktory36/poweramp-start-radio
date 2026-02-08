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
import androidx.compose.material.icons.filled.Clear
import androidx.compose.material.icons.filled.Menu
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.ui.draw.clip
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
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
    val showProvenance = session.tracks.any { it.provenance.influences.size > 1 }
    // Use target queue size so colors stay stable as tracks stream in
    val colorTotal = maxOf(session.config.numTracks, session.tracks.size)

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

        if (showProvenance) {
            Row(modifier = Modifier.padding(horizontal = 16.dp, vertical = 2.dp),
                horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                val primary = MaterialTheme.colorScheme.primary
                Row(verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                    Canvas(modifier = Modifier.size(8.dp)) {
                        drawRect(color = primary)
                    }
                    Text("= your seed", style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f))
                }
                Row(verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                    Canvas(modifier = Modifier.size(8.dp)) {
                        drawRect(color = trackColor(primary, 1, colorTotal))
                    }
                    Text("= track that influenced this pick", style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f))
                }
            }
        }

        LazyColumn(state = listState, modifier = Modifier.fillMaxSize(),
            contentPadding = PaddingValues(horizontal = 16.dp, vertical = 4.dp)) {
            items(session.tracks.size) { index ->
                TrackResultRow(
                    trackResult = session.tracks[index],
                    index = index,
                    totalTracks = colorTotal,
                    showProvenance = showProvenance,
                    session = session
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
    hsl[1] = maxOf(hsl[1], 0.5f)               // Ensure vivid colors
    hsl[2] = hsl[2].coerceIn(0.35f, 0.65f)     // Readable on both themes
    return Color(androidx.core.graphics.ColorUtils.HSLToColor(hsl))
}

@Composable
fun InfluenceStrip(
    provenance: TrackProvenance,
    totalTracks: Int,
    modifier: Modifier = Modifier
) {
    val primary = MaterialTheme.colorScheme.primary
    val density = LocalDensity.current
    val stripWidthPx = with(density) { 40.dp.toPx() }

    val segmentColors = remember(primary, totalTracks, provenance) {
        provenance.influences.map { influence ->
            if (influence.sourceIndex == -1) primary
            else trackColor(primary, influence.sourceIndex, totalTracks)
        }
    }

    val bgColor = primary.copy(alpha = 0.12f)

    Canvas(modifier = modifier
        .width(40.dp)
        .height(12.dp)
        .clip(RoundedCornerShape(3.dp))
    ) {
        // Background — visible as "empty" portion for diversity meter
        drawRect(color = bgColor, size = Size(size.width, size.height))

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
    showProvenance: Boolean,
    session: RadioResult? = null
) {
    var expanded by remember { mutableStateOf(false) }

    Column(modifier = Modifier.fillMaxWidth().clickable { expanded = !expanded }) {
        Row(modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically) {
            if (showProvenance) {
                InfluenceStrip(
                    provenance = trackResult.provenance,
                    totalTracks = totalTracks
                )
                Spacer(modifier = Modifier.width(6.dp))
            } else {
                // Batch mode: similarity bar
                SimilarityBar(
                    similarity = trackResult.similarity,
                    modifier = Modifier
                )
                Spacer(modifier = Modifier.width(6.dp))
            }
            Column(modifier = Modifier.weight(1f).padding(vertical = 2.dp, horizontal = 4.dp)) {
                Text(text = trackResult.track.title ?: "Unknown", style = MaterialTheme.typography.bodyMedium,
                    maxLines = 1, overflow = TextOverflow.Ellipsis)
                Text(text = trackResult.track.artist ?: "Unknown", style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant, maxLines = 1, overflow = TextOverflow.Ellipsis)
            }

            val pct = (trackResult.similarity * 100).roundToInt()
            Text(text = "$pct%", fontFamily = FontFamily.Monospace,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                fontSize = 10.sp, textAlign = TextAlign.End, modifier = Modifier.padding(horizontal = 2.dp))
        }

        AnimatedVisibility(visible = expanded) {
            TrackExplanation(
                trackResult = trackResult,
                index = index,
                session = session,
                modifier = Modifier.padding(start = 46.dp, top = 2.dp, bottom = 4.dp)
            )
        }
    }
}

@Composable
private fun SimilarityBar(
    similarity: Float,
    modifier: Modifier = Modifier
) {
    val primary = MaterialTheme.colorScheme.primary
    val bgColor = primary.copy(alpha = 0.12f)
    val fillFraction = similarity.coerceIn(0f, 1f)

    Canvas(modifier = modifier
        .width(40.dp)
        .height(12.dp)
        .clip(RoundedCornerShape(3.dp))
    ) {
        drawRect(color = bgColor, size = Size(size.width, size.height))
        drawRect(color = primary, size = Size(size.width * fillFraction, size.height))
    }
}

@Composable
private fun TrackExplanation(
    trackResult: QueuedTrackResult,
    index: Int,
    session: RadioResult?,
    modifier: Modifier = Modifier
) {
    val pct = (trackResult.similarity * 100).roundToInt()
    val hasDrift = trackResult.provenance.influences.size > 1
    val isExplorer = session?.config?.selectionMode == SelectionMode.RANDOM_WALK
    val mode = session?.let { humanSelectionMode(it.config.selectionMode, it.config.driftEnabled) } ?: ""
    val seedTitle = session?.seedTrack?.title ?: "seed"

    Column(modifier = modifier) {
        if (isExplorer) {
            Text("Found through similarity chains, not direct match.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant)
            Text("Connected to your seed through intermediate tracks.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant)
        } else if (hasDrift && session != null) {
            Text("$pct% match",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant)
            Text("Search was shaped by:",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant)

            val significantInfluences = trackResult.provenance.influences
                .filter { it.weight >= 0.05f }
                .sortedByDescending { it.weight }
            val collapsedCount = trackResult.provenance.influences.count { it.weight < 0.05f }

            for (inf in significantInfluences) {
                val weightPct = (inf.weight * 100).roundToInt()
                val sourceName = if (inf.sourceIndex == -1) {
                    "Seed: \"$seedTitle\""
                } else {
                    val trackInfo = session.tracks.getOrNull(inf.sourceIndex)
                    if (trackInfo != null) {
                        "#${inf.sourceIndex + 1}: \"${trackInfo.track.title ?: "Unknown"}\" by ${trackInfo.track.artist ?: "Unknown"}"
                    } else {
                        "#${inf.sourceIndex + 1}"
                    }
                }
                Text("  $sourceName \u2014 $weightPct%",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            if (collapsedCount > 0) {
                Text("  (+ $collapsedCount earlier tracks)",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f))
            }
        } else {
            Text("$pct% similar to \"$seedTitle\"",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant)
            if (mode.isNotEmpty()) {
                Text("Selected from ${session?.config?.candidatePoolSize ?: 200} candidates using $mode mode.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
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
                                val modeLabel = humanSelectionMode(session.config.selectionMode, session.config.driftEnabled)
                                Text("${session.seedTrack.artist ?: "Unknown"} \u00b7 $timeStr \u00b7 ${session.queuedCount} tracks \u00b7 $modeLabel",
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
                Text("Selection Mode", style = MaterialTheme.typography.titleMedium)
                Spacer(modifier = Modifier.height(4.dp))

                val driftInfo = if (!driftEnabled) "" else when (driftMode) {
                    DriftMode.SEED_INTERPOLATION -> {
                        val decay = when (anchorDecay) {
                            DecaySchedule.NONE -> ""
                            DecaySchedule.LINEAR -> ", gradual"
                            DecaySchedule.EXPONENTIAL -> ", quick start"
                            DecaySchedule.STEP -> ", halfway"
                        }
                        " + Evolving (Anchored ${(anchorStrength * 100).roundToInt()}%$decay)"
                    }
                    DriftMode.MOMENTUM ->
                        " + Evolving (Flowing ${(momentumBeta * 100).roundToInt()}%)"
                }

                Column(modifier = Modifier.selectableGroup()) {
                    AlgorithmOption(
                        label = "Balanced",
                        description = "Similar to your seed, but avoids repeating the same sound. " +
                            "The default \u2014 good for most listening.",
                        technicalName = "MMR",
                        preview = previews[SelectionMode.MMR],
                        isLoading = SelectionMode.MMR in previewsLoading,
                        selected = selectionMode == SelectionMode.MMR,
                        expanded = expandedPeek[SelectionMode.MMR] == true,
                        onToggleExpanded = { expandedPeek[SelectionMode.MMR] = !(expandedPeek[SelectionMode.MMR] ?: false) },
                        onRequestPreview = { viewModel.computePreview(SelectionMode.MMR) },
                        previewInfo = "${(diversityLambda * 100).roundToInt()}% similarity$driftInfo",
                        onClick = { viewModel.setSelectionMode(SelectionMode.MMR) }
                    )
                    AlgorithmOption(
                        label = "Wide Net",
                        description = "Maximizes variety while staying relevant. Picks tracks that " +
                            "sound different from each other, not just from your seed.",
                        technicalName = "DPP",
                        preview = previews[SelectionMode.DPP],
                        isLoading = SelectionMode.DPP in previewsLoading,
                        selected = selectionMode == SelectionMode.DPP,
                        expanded = expandedPeek[SelectionMode.DPP] == true,
                        onToggleExpanded = { expandedPeek[SelectionMode.DPP] = !(expandedPeek[SelectionMode.DPP] ?: false) },
                        onRequestPreview = { viewModel.computePreview(SelectionMode.DPP) },
                        onClick = { viewModel.setSelectionMode(SelectionMode.DPP) }
                    )
                    AlgorithmOption(
                        label = "Explorer",
                        description = "Follows chains of similarity \u2014 A sounds like B, B sounds " +
                            "like C, so C appears even if it doesn't directly resemble your seed." +
                            if (databaseInfo?.hasGraph != true) " (requires similarity graph in database)" else "",
                        technicalName = "Personalized PageRank",
                        preview = previews[SelectionMode.RANDOM_WALK],
                        isLoading = SelectionMode.RANDOM_WALK in previewsLoading,
                        selected = selectionMode == SelectionMode.RANDOM_WALK,
                        expanded = expandedPeek[SelectionMode.RANDOM_WALK] == true,
                        onToggleExpanded = { expandedPeek[SelectionMode.RANDOM_WALK] = !(expandedPeek[SelectionMode.RANDOM_WALK] ?: false) },
                        onRequestPreview = { viewModel.computePreview(SelectionMode.RANDOM_WALK) },
                        previewInfo = "return ${(anchorStrength * 100).roundToInt()}%",
                        onClick = { viewModel.setSelectionMode(SelectionMode.RANDOM_WALK) }
                    )
                    AlgorithmOption(
                        label = "Shuffle",
                        description = "Randomly picks from the top candidates. Different playlist " +
                            "every time you run it from the same song.",
                        technicalName = "Gumbel-Max",
                        preview = previews[SelectionMode.TEMPERATURE],
                        isLoading = SelectionMode.TEMPERATURE in previewsLoading,
                        selected = selectionMode == SelectionMode.TEMPERATURE,
                        expanded = expandedPeek[SelectionMode.TEMPERATURE] == true,
                        onToggleExpanded = { expandedPeek[SelectionMode.TEMPERATURE] = !(expandedPeek[SelectionMode.TEMPERATURE] ?: false) },
                        onRequestPreview = { viewModel.computePreview(SelectionMode.TEMPERATURE) },
                        previewInfo = "randomness ${String.format("%.2f", temperature)}$driftInfo",
                        onClick = { viewModel.setSelectionMode(SelectionMode.TEMPERATURE) }
                    )
                }
            }

            // Similarity vs. Variety — only for MMR (Balanced)
            if (selectionMode == SelectionMode.MMR) {
                item {
                    Column {
                        Text("Similarity vs. Variety: ${(diversityLambda * 100).roundToInt()}%",
                            style = MaterialTheme.typography.titleSmall)
                        Text("How much the playlist favors close matches over new sounds.",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant)
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Text("More variety", style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f))
                            Slider(value = diversityLambda, onValueChange = { viewModel.setDiversityLambda(it) },
                                valueRange = 0f..1f, modifier = Modifier.weight(1f))
                            Text("More similar", style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f))
                        }
                    }
                }
            }

            // Randomness — only for Shuffle mode
            if (selectionMode == SelectionMode.TEMPERATURE) {
                item {
                    Column {
                        Text("Randomness: ${String.format("%.2f", temperature)}",
                            style = MaterialTheme.typography.titleSmall)
                        Text("Low = nearly the same playlist each time. High = more surprises.",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant)
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Text("Predictable", style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f))
                            Slider(value = temperature, onValueChange = { viewModel.setTemperature(it) },
                                valueRange = 0f..1f, modifier = Modifier.weight(1f))
                            Text("Surprising", style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f))
                        }
                    }
                }
            }

            // Return Frequency — only for Explorer mode
            if (isRandomWalk) {
                item {
                    Column {
                        Text("Return Frequency: ${(anchorStrength * 100).roundToInt()}%",
                            style = MaterialTheme.typography.titleSmall)
                        Text("How often the walk jumps back to your seed. " +
                            "Low = explores further. High = stays nearby.",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant)
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Text("Explores further", style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f))
                            Slider(value = anchorStrength, onValueChange = { viewModel.setAnchorStrength(it) },
                                valueRange = 0.05f..0.95f, modifier = Modifier.weight(1f))
                            Text("Stays nearby", style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f))
                        }
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
                        Text("Playlist Evolution", style = MaterialTheme.typography.titleMedium)
                        Spacer(modifier = Modifier.height(4.dp))

                        Row(modifier = Modifier.fillMaxWidth().selectable(
                            selected = driftEnabled, onClick = { viewModel.setDriftEnabled(!driftEnabled) }, role = Role.Checkbox
                        ).padding(vertical = 4.dp), verticalAlignment = Alignment.CenterVertically) {
                            Checkbox(checked = driftEnabled, onCheckedChange = null)
                            Spacer(modifier = Modifier.width(8.dp))
                            Column {
                                Text("Evolving playlist", style = MaterialTheme.typography.bodyMedium)
                                Text("Each track influences what comes next \u2014 the playlist gradually " +
                                    "drifts from your starting song. When off, every track is picked " +
                                    "based on the seed alone.",
                                    style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                            }
                        }

                        AnimatedVisibility(visible = driftEnabled) {
                            Column(modifier = Modifier.padding(start = 16.dp, top = 8.dp)) {
                                Text("Evolution style", style = MaterialTheme.typography.titleSmall)
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
                                            Text("Anchored", style = MaterialTheme.typography.bodyMedium)
                                            Text("Blends the seed with the latest pick. Your starting song " +
                                                "always has some pull.",
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
                                            Text("Flowing", style = MaterialTheme.typography.bodyMedium)
                                            Text("Running average of everything picked so far. Direction " +
                                                "shifts gradually and smoothly.",
                                                style = MaterialTheme.typography.bodySmall,
                                                color = MaterialTheme.colorScheme.onSurfaceVariant)
                                        }
                                    }
                                }

                                // Anchored parameters
                                AnimatedVisibility(visible = driftMode == DriftMode.SEED_INTERPOLATION) {
                                    Column(modifier = Modifier.padding(top = 8.dp)) {
                                        Text("Seed Influence: ${(anchorStrength * 100).roundToInt()}%",
                                            style = MaterialTheme.typography.titleSmall)
                                        Text("How much your starting song shapes later picks. " +
                                            "Low = the playlist wanders. High = it stays close.",
                                            style = MaterialTheme.typography.bodySmall,
                                            color = MaterialTheme.colorScheme.onSurfaceVariant)
                                        Row(verticalAlignment = Alignment.CenterVertically) {
                                            Text("Wanders further", style = MaterialTheme.typography.labelSmall,
                                                color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f))
                                            Slider(value = anchorStrength, onValueChange = { viewModel.setAnchorStrength(it) },
                                                valueRange = 0f..1f, modifier = Modifier.weight(1f))
                                            Text("Stays close", style = MaterialTheme.typography.labelSmall,
                                                color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f))
                                        }

                                        Text("Fade schedule", style = MaterialTheme.typography.titleSmall)
                                        Text("How the seed's influence weakens over time.",
                                            style = MaterialTheme.typography.bodySmall,
                                            color = MaterialTheme.colorScheme.onSurfaceVariant)
                                        Spacer(modifier = Modifier.height(4.dp))
                                        Column(modifier = Modifier.selectableGroup()) {
                                            for (schedule in DecaySchedule.entries) {
                                                val (label, desc) = when (schedule) {
                                                    DecaySchedule.NONE -> "None" to "Same influence throughout"
                                                    DecaySchedule.LINEAR -> "Gradual" to "Weakens steadily toward the end"
                                                    DecaySchedule.EXPONENTIAL -> "Quick start" to "Weakens quickly at first, then levels off"
                                                    DecaySchedule.STEP -> "Halfway" to "Full influence for the first half, then drops"
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
                                        Text("Smoothing: ${(momentumBeta * 100).roundToInt()}%",
                                            style = MaterialTheme.typography.titleSmall)
                                        Text("How gradually the playlist changes direction. " +
                                            "High = smooth, slow shifts. Low = reacts quickly to each pick.",
                                            style = MaterialTheme.typography.bodySmall,
                                            color = MaterialTheme.colorScheme.onSurfaceVariant)
                                        Row(verticalAlignment = Alignment.CenterVertically) {
                                            Text("Reactive", style = MaterialTheme.typography.labelSmall,
                                                color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f))
                                            Slider(value = momentumBeta, onValueChange = { viewModel.setMomentumBeta(it) },
                                                valueRange = 0.1f..0.95f, modifier = Modifier.weight(1f))
                                            Text("Steady", style = MaterialTheme.typography.labelSmall,
                                                color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f))
                                        }
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
                        Text("Similarity graph: ${if (databaseInfo.hasGraph) "yes \u2014 Explorer mode available" else "not found \u2014 Explorer will fall back to Balanced"}")
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
    technicalName: String = "",
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
            if (technicalName.isNotEmpty()) {
                Text(technicalName, style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.5f))
            }
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
        SelectionMode.MMR -> "Balanced"
        SelectionMode.DPP -> "Wide Net"
        SelectionMode.RANDOM_WALK -> "Explorer"
        SelectionMode.TEMPERATURE -> "Shuffle"
    }
    return if (drift && mode != SelectionMode.DPP && mode != SelectionMode.RANDOM_WALK) "$base + Evolving" else base
}
