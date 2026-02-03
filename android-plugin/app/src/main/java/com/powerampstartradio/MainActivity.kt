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
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.pager.VerticalPager
import androidx.compose.foundation.pager.rememberPagerState
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.selection.selectableGroup
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.MoreVert
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import androidx.lifecycle.viewmodel.compose.viewModel
import com.powerampstartradio.data.EmbeddingDatabase
import com.powerampstartradio.data.EmbeddingModel
import com.powerampstartradio.poweramp.PowerampHelper
import com.powerampstartradio.poweramp.PowerampReceiver
import com.powerampstartradio.poweramp.PowerampTrack
import com.powerampstartradio.services.RadioService
import com.powerampstartradio.similarity.SearchStrategy
import com.powerampstartradio.ui.DatabaseInfo
import com.powerampstartradio.ui.MainViewModel
import com.powerampstartradio.ui.QueueStatus
import com.powerampstartradio.ui.QueuedTrackResult
import com.powerampstartradio.ui.RadioResult
import com.powerampstartradio.ui.RadioUiState
import com.powerampstartradio.ui.theme.PowerampStartRadioTheme
import java.io.File

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

    // Local UI state
    var currentTrack by remember { mutableStateOf<PowerampTrack?>(PowerampReceiver.currentTrack) }
    var showSettingsSheet by remember { mutableStateOf(false) }
    var showMenu by remember { mutableStateOf(false) }
    var statusMessage by remember { mutableStateOf("") }

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

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Start Radio") },
                actions = {
                    IconButton(onClick = { showMenu = true }) {
                        Icon(Icons.Default.MoreVert, contentDescription = "Menu")
                    }
                    DropdownMenu(
                        expanded = showMenu,
                        onDismissRequest = { showMenu = false }
                    ) {
                        DropdownMenuItem(
                            text = { Text("Settings") },
                            onClick = {
                                showMenu = false
                                showSettingsSheet = true
                            }
                        )
                    }
                }
            )
        },
        floatingActionButton = {
            ExtendedFloatingActionButton(
                onClick = {
                    if (currentTrack != null && databaseInfo != null) {
                        viewModel.startRadio()
                    } else if (currentTrack == null) {
                        statusMessage = "Play a song in Poweramp first"
                    } else {
                        statusMessage = "Import database in Settings"
                    }
                },
                icon = { Icon(Icons.Default.PlayArrow, contentDescription = null) },
                text = { Text("Start Radio") },
                expanded = radioState !is RadioUiState.Loading
            )
        }
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
        ) {
            // Now Playing — always visible at top
            NowPlayingSection(
                currentTrack = currentTrack,
                modifier = Modifier.padding(16.dp)
            )

            HorizontalDivider()

            // Main content area
            Box(modifier = Modifier.weight(1f)) {
                if (sessionHistory.isEmpty()) {
                    // No sessions yet — show idle/loading/error content
                    when (val state = radioState) {
                        is RadioUiState.Idle -> {
                            IdleContent(
                                hasPermission = hasPermission,
                                databaseInfo = databaseInfo,
                                statusMessage = statusMessage,
                                onRequestPermission = { viewModel.requestPermission() },
                                modifier = Modifier.fillMaxSize()
                            )
                        }
                        is RadioUiState.Loading -> {
                            Box(
                                modifier = Modifier.fillMaxSize(),
                                contentAlignment = Alignment.Center
                            ) {
                                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                                    CircularProgressIndicator()
                                    Spacer(modifier = Modifier.height(16.dp))
                                    Text("Finding similar tracks...")
                                }
                            }
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
                        is RadioUiState.Success -> {
                            // Shouldn't happen (sessionHistory would be non-empty), but handle gracefully
                        }
                    }
                } else {
                    // Session pager
                    val pagerState = rememberPagerState(
                        initialPage = sessionHistory.size - 1,
                        pageCount = { sessionHistory.size }
                    )

                    // Auto-scroll to latest session when a new one is added
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

                    // Loading/Error overlay on top of pager
                    when (val state = radioState) {
                        is RadioUiState.Loading -> {
                            Box(
                                modifier = Modifier
                                    .fillMaxSize()
                                    .background(MaterialTheme.colorScheme.surface.copy(alpha = 0.7f)),
                                contentAlignment = Alignment.Center
                            ) {
                                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                                    CircularProgressIndicator()
                                    Spacer(modifier = Modifier.height(16.dp))
                                    Text("Finding similar tracks...")
                                }
                            }
                        }
                        is RadioUiState.Error -> {
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
                                        text = state.message,
                                        modifier = Modifier.padding(16.dp),
                                        color = MaterialTheme.colorScheme.onErrorContainer
                                    )
                                }
                            }
                        }
                        else -> {}
                    }
                }
            }
        }
    }

    // Settings Bottom Sheet
    if (showSettingsSheet) {
        SettingsBottomSheet(
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
            onDismiss = { showSettingsSheet = false }
        )
    }
}

@Composable
fun NowPlayingSection(
    currentTrack: PowerampTrack?,
    modifier: Modifier = Modifier
) {
    Column(modifier = modifier) {
        Text(
            text = "NOW PLAYING",
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.primary
        )
        Spacer(modifier = Modifier.height(4.dp))

        if (currentTrack != null) {
            Text(
                text = "\"${currentTrack.title}\"",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold
            )
            Text(
                text = "${currentTrack.artist ?: "Unknown"} • ${currentTrack.album ?: ""}",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        } else {
            Text(
                text = "No track playing",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}

@Composable
fun SessionPage(
    session: RadioResult,
    pageIndex: Int,
    totalPages: Int,
    modifier: Modifier = Modifier
) {
    Column(modifier = modifier) {
        // Seed track for this session
        SeedTrackSection(
            radioResult = session,
            modifier = Modifier.padding(16.dp)
        )

        HorizontalDivider()

        // Page indicator
        if (totalPages > 1) {
            Text(
                text = "Session ${pageIndex + 1} of $totalPages",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(horizontal = 16.dp, vertical = 4.dp)
            )
        }

        // Queue results
        ResultsSection(
            result = session,
            modifier = Modifier.weight(1f)
        )
    }
}

@Composable
fun SeedTrackSection(
    radioResult: RadioResult,
    modifier: Modifier = Modifier
) {
    Column(modifier = modifier) {
        Text(
            text = "SEED TRACK",
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.primary
        )
        Spacer(modifier = Modifier.height(4.dp))

        Text(
            text = "\"${radioResult.seedTrack.title}\"",
            style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.Bold
        )
        Text(
            text = "${radioResult.seedTrack.artist ?: "Unknown"} • ${radioResult.seedTrack.album ?: ""}",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Text(
            text = "matched via: ${radioResult.matchType.name}",
            style = MaterialTheme.typography.bodySmall,
            fontFamily = FontFamily.Monospace,
            color = MaterialTheme.colorScheme.tertiary
        )
    }
}

@Composable
fun ResultsSection(
    result: RadioResult,
    modifier: Modifier = Modifier
) {
    Column(modifier = modifier) {
        // Summary header
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = 8.dp),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            val strategyLabel = result.strategy.name.lowercase().replace('_', ' ')
            val label = "QUEUE RESULTS ($strategyLabel)"
            Text(
                text = label,
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.primary
            )
            Text(
                text = "${result.queuedCount} queued / ${result.failedCount} failed / ${result.requestedCount} requested",
                style = MaterialTheme.typography.labelSmall,
                fontFamily = FontFamily.Monospace
            )
        }

        // Track list
        LazyColumn(
            modifier = Modifier.fillMaxSize(),
            contentPadding = PaddingValues(horizontal = 16.dp, vertical = 8.dp)
        ) {
            items(result.tracks) { trackResult ->
                TrackResultRow(trackResult, showModelTag = result.isMultiModel)
            }
        }
    }
}

@Composable
fun TrackResultRow(trackResult: QueuedTrackResult, showModelTag: Boolean = false) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        // Similarity score
        Text(
            text = String.format("%.3f", trackResult.similarity),
            style = MaterialTheme.typography.bodySmall,
            fontFamily = FontFamily.Monospace,
            fontWeight = FontWeight.Bold,
            modifier = Modifier.width(48.dp)
        )

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
                    .width(38.dp)
                    .padding(end = 4.dp)
            )
        }

        // Track info
        Column(
            modifier = Modifier
                .weight(1f)
                .padding(horizontal = 8.dp)
        ) {
            Text(
                text = trackResult.track.title ?: "Unknown",
                style = MaterialTheme.typography.bodyMedium,
                maxLines = 1
            )
            Text(
                text = trackResult.track.artist ?: "Unknown",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                maxLines = 1
            )
        }

        // Status
        val (statusText, statusColor) = when (trackResult.status) {
            QueueStatus.QUEUED -> "queued" to MaterialTheme.colorScheme.primary
            QueueStatus.NOT_IN_LIBRARY -> "not found" to MaterialTheme.colorScheme.error
            QueueStatus.QUEUE_FAILED -> "failed" to MaterialTheme.colorScheme.error
        }
        val statusIcon = if (trackResult.status == QueueStatus.QUEUED) "✓" else "✗"

        Text(
            text = "$statusIcon $statusText",
            style = MaterialTheme.typography.bodySmall,
            fontFamily = FontFamily.Monospace,
            color = statusColor
        )
    }
}

@Composable
fun IdleContent(
    hasPermission: Boolean,
    databaseInfo: DatabaseInfo?,
    statusMessage: String,
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
                        text = "Import via Settings menu",
                        style = MaterialTheme.typography.bodySmall
                    )
                }
            }
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
                text = "Ready - tap Start Radio",
                style = MaterialTheme.typography.bodyLarge,
                color = MaterialTheme.colorScheme.primary
            )
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsBottomSheet(
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
    onDismiss: () -> Unit
) {
    val availableModels = databaseInfo?.availableModels ?: emptySet()
    val hasMulan = EmbeddingModel.MULAN in availableModels
    val hasFlamingo = EmbeddingModel.FLAMINGO in availableModels
    val hasBoth = hasMulan && hasFlamingo

    ModalBottomSheet(onDismissRequest = onDismiss) {
        LazyColumn(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(24.dp)
        ) {
            item {
                Text(
                    text = "Settings",
                    style = MaterialTheme.typography.headlineSmall
                )
            }

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
                            description = "Round-robin results from both models",
                            selected = searchStrategy == SearchStrategy.INTERLEAVE,
                            enabled = hasBoth,
                            disabledReason = if (!hasBoth) "Requires both MuLan and Flamingo" else null,
                            onClick = { onSearchStrategyChange(SearchStrategy.INTERLEAVE) }
                        )
                        StrategyOption(
                            label = "Anchor & Expand",
                            description = "One model finds anchors, other expands each",
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
                                text = "Drift",
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
