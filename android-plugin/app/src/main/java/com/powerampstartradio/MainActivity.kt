@file:OptIn(ExperimentalFoundationApi::class)

package com.powerampstartradio

import android.content.IntentFilter
import android.net.Uri
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.slideInHorizontally
import androidx.compose.animation.slideOutHorizontally
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.pager.VerticalPager
import androidx.compose.foundation.pager.rememberPagerState
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.selection.selectableGroup
import androidx.compose.foundation.shape.RoundedCornerShape
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
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.lerp
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
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

    // Pager state for session history
    val pagerState = if (sessionHistory.isNotEmpty()) {
        rememberPagerState(
            initialPage = sessionHistory.size - 1,
            pageCount = { sessionHistory.size }
        )
    } else null

    ModalNavigationDrawer(
        drawerState = drawerState,
        gesturesEnabled = sessionHistory.isNotEmpty(),
        drawerContent = {
            ModalDrawerSheet(modifier = Modifier.width(280.dp)) {
                SessionHistoryDrawer(
                    sessions = sessionHistory,
                    onSessionTap = { index ->
                        scope.launch {
                            drawerState.close()
                            pagerState?.animateScrollToPage(index)
                        }
                    },
                    onClear = {
                        viewModel.clearSessionHistory()
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
                    pagerState = pagerState,
                    onStartRadio = {
                        if (currentTrack != null && databaseInfo != null) {
                            viewModel.startRadio()
                        } else if (currentTrack == null) {
                            statusMessage = "Play a song in Poweramp first"
                        } else {
                            statusMessage = "Import database in Settings"
                        }
                    },
                    onResetState = { viewModel.resetRadioState() },
                    onRequestPermission = { viewModel.requestPermission() },
                    onOpenSettings = { showSettings = true },
                    onOpenDrawer = { scope.launch { drawerState.open() } }
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
    pagerState: androidx.compose.foundation.pager.PagerState?,
    onStartRadio: () -> Unit,
    onResetState: () -> Unit,
    onRequestPermission: () -> Unit,
    onOpenSettings: () -> Unit,
    onOpenDrawer: () -> Unit
) {
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
                    if (radioState is RadioUiState.Success || sessionHistory.isNotEmpty()) {
                        IconButton(onClick = onResetState) {
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
                // Compact header — now playing or seed track
                val latestSession = sessionHistory.lastOrNull()
                if (radioState is RadioUiState.Success && latestSession != null) {
                    CompactSeedHeader(
                        session = latestSession,
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
                    if (sessionHistory.isEmpty()) {
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
                            is RadioUiState.Loading, is RadioUiState.Success -> {
                                // Loading overlay handles Loading; Success with empty history is transient
                            }
                        }
                    } else if (pagerState != null) {
                        // Auto-scroll to latest session
                        LaunchedEffect(sessionHistory.size) {
                            if (sessionHistory.isNotEmpty()) {
                                pagerState.animateScrollToPage(sessionHistory.size - 1)
                            }
                        }

                        VerticalPager(
                            state = pagerState,
                            modifier = Modifier.fillMaxSize()
                        ) { page ->
                            SessionPage(
                                session = sessionHistory[page],
                                pageIndex = page,
                                totalPages = sessionHistory.size,
                                modifier = Modifier.fillMaxSize()
                            )
                        }

                        // Error overlay on top of pager
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
                    }
                }
            }

            // Loading overlay (shown on top of all content)
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
        SuggestionChip(
            onClick = {},
            label = { Text(humanMatchType(session.matchType), style = MaterialTheme.typography.labelSmall) }
        )
    }
}

// ---- Session Page ----

@Composable
fun SessionPage(
    session: RadioResult,
    pageIndex: Int,
    totalPages: Int,
    modifier: Modifier = Modifier
) {
    Column(modifier = modifier) {
        // Summary header
        ResultsSummary(
            result = session,
            pageIndex = pageIndex,
            totalPages = totalPages,
            modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp)
        )

        // Track list
        LazyColumn(
            modifier = Modifier.fillMaxSize(),
            contentPadding = PaddingValues(horizontal = 16.dp, vertical = 4.dp)
        ) {
            items(session.tracks) { trackResult ->
                TrackResultRow(trackResult, showModelTag = session.isMultiModel)
            }
        }
    }
}

@Composable
fun ResultsSummary(
    result: RadioResult,
    pageIndex: Int,
    totalPages: Int,
    modifier: Modifier = Modifier
) {
    Row(
        modifier = modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        // Left: human-friendly summary
        val strategyLabel = humanStrategy(result.strategy)
        val countText = if (result.failedCount > 0) {
            "${result.queuedCount} of ${result.requestedCount} queued (${result.failedCount} not in Poweramp)"
        } else {
            "${result.queuedCount} tracks queued via $strategyLabel"
        }
        Text(
            text = countText,
            style = MaterialTheme.typography.labelMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.weight(1f)
        )

        // Right: page indicator
        if (totalPages > 1) {
            Text(
                text = "${pageIndex + 1}/$totalPages",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}

// ---- Track Result Row with Similarity Indicator ----

@Composable
fun TrackResultRow(trackResult: QueuedTrackResult, showModelTag: Boolean = false) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        // Similarity indicator with color gradient
        SimilarityIndicator(
            score = trackResult.similarity,
            model = trackResult.modelUsed
        )

        Spacer(modifier = Modifier.width(6.dp))

        // Model tag (shown when using multi-model strategies)
        if (showModelTag && trackResult.modelUsed != null) {
            val (tagText, tagColor) = when (trackResult.modelUsed) {
                EmbeddingModel.MUQ -> "muq" to MaterialTheme.colorScheme.primary
                EmbeddingModel.MULAN -> "mulan" to MaterialTheme.colorScheme.tertiary
                EmbeddingModel.FLAMINGO -> "flam" to MaterialTheme.colorScheme.secondary
            }
            Text(
                text = tagText,
                style = MaterialTheme.typography.labelSmall,
                fontFamily = FontFamily.Monospace,
                color = tagColor,
                fontSize = 10.sp,
                modifier = Modifier
                    .width(36.dp)
                    .padding(end = 4.dp)
            )
        }

        // Track info
        Column(
            modifier = Modifier
                .weight(1f)
                .padding(horizontal = 4.dp)
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

        // Status icon
        val statusColor = when (trackResult.status) {
            QueueStatus.QUEUED -> MaterialTheme.colorScheme.primary
            QueueStatus.NOT_IN_LIBRARY -> MaterialTheme.colorScheme.error
            QueueStatus.QUEUE_FAILED -> MaterialTheme.colorScheme.error
        }
        Text(
            text = if (trackResult.status == QueueStatus.QUEUED) "\u2713" else "\u2717",
            style = MaterialTheme.typography.bodySmall,
            color = statusColor
        )
    }
}

@Composable
fun SimilarityIndicator(score: Float, model: EmbeddingModel?) {
    val floor = when (model) {
        EmbeddingModel.FLAMINGO -> 0.5f
        else -> 0.0f
    }
    val normalized = ((score - floor) / (1f - floor)).coerceIn(0f, 1f)

    val amber = Color(0xFFF59E0B)
    val green = Color(0xFF22C55E)
    val blue = Color(0xFF3B82F6)

    val color = if (normalized <= 0.5f) {
        lerp(amber, green, normalized * 2)
    } else {
        lerp(green, blue, (normalized - 0.5f) * 2)
    }

    Box(modifier = Modifier.width(52.dp).height(20.dp)) {
        // Background bar
        Box(
            modifier = Modifier
                .fillMaxHeight()
                .fillMaxWidth(normalized)
                .background(color.copy(alpha = 0.25f), RoundedCornerShape(4.dp))
        )
        // Score text
        Text(
            text = String.format("%.3f", score),
            style = MaterialTheme.typography.bodySmall,
            fontFamily = FontFamily.Monospace,
            fontWeight = FontWeight.Bold,
            color = color,
            modifier = Modifier.align(Alignment.CenterStart).padding(start = 2.dp)
        )
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

        // Index status
        if (indexStatus != null) {
            Text(
                text = indexStatus,
                style = MaterialTheme.typography.bodySmall,
                fontFamily = FontFamily.Monospace,
                color = if (indexStatus == "Indices ready") MaterialTheme.colorScheme.primary
                       else MaterialTheme.colorScheme.onSurfaceVariant
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
        if (hasPermission && databaseInfo != null && indexStatus == "Indices ready") {
            Text(
                text = "Ready \u2014 tap Start Radio",
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
                            description = "One model selects seeds, the other expands recommendations on each",
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
                            .padding(vertical = 4.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Checkbox(
                            checked = drift,
                            onCheckedChange = onDriftChange
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Column {
                            Text(
                                text = "Drift ahead",
                                style = MaterialTheme.typography.bodyMedium
                            )
                            Text(
                                text = "Each result seeds the next search, gradually exploring new territory",
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

private fun humanStrategy(strategy: SearchStrategy): String = when (strategy) {
    SearchStrategy.MULAN_ONLY -> "MuLan"
    SearchStrategy.FLAMINGO_ONLY -> "Flamingo"
    SearchStrategy.INTERLEAVE -> "interleave"
    SearchStrategy.ANCHOR_EXPAND -> "anchor & expand"
}
