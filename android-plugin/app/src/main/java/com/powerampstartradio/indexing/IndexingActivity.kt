package com.powerampstartradio.indexing

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.setContent
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.basicMarquee
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.KeyboardArrowDown
import androidx.compose.material.icons.filled.KeyboardArrowUp
import androidx.compose.material.icons.filled.Clear
import androidx.compose.material.icons.filled.MoreVert
import androidx.compose.material.icons.filled.Search
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.state.ToggleableState
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.compose.LifecycleEventEffect
import androidx.lifecycle.viewmodel.compose.viewModel
import com.powerampstartradio.ui.theme.PowerampStartRadioTheme

class IndexingActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        requestNotificationPermission()
        setContent {
            PowerampStartRadioTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background,
                ) {
                    IndexingScreen(onBack = { finish() })
                }
            }
        }
    }

    private fun requestNotificationPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS)
                != PackageManager.PERMISSION_GRANTED
            ) {
                ActivityCompat.requestPermissions(
                    this, arrayOf(Manifest.permission.POST_NOTIFICATIONS), 0
                )
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun IndexingScreen(
    viewModel: IndexingViewModel = viewModel(),
    onBack: () -> Unit,
) {
    val unindexedTracks by viewModel.unindexedTracks.collectAsState()
    val selectedIds by viewModel.selectedIds.collectAsState()
    val dismissedIds by viewModel.dismissedIds.collectAsState()
    val ignoredIds by viewModel.ignoredIds.collectAsState()
    val isDetecting by viewModel.isDetecting.collectAsState()
    val detectingStatus by viewModel.detectingStatus.collectAsState()
    val indexingState by viewModel.indexingState.collectAsState()

    // Re-detect when activity resumes (e.g. user replaced DB while app was backgrounded).
    // detectUnindexed() has its own cache check so this is cheap when nothing changed.
    LifecycleEventEffect(Lifecycle.Event.ON_RESUME) {
        viewModel.detectUnindexed()
    }

    val visibleTracks = remember(unindexedTracks, dismissedIds, ignoredIds) {
        unindexedTracks.filter { it.powerampFileId !in dismissedIds && it.powerampFileId !in ignoredIds }
    }
    val selectedCount = remember(selectedIds, visibleTracks) {
        visibleTracks.count { it.powerampFileId in selectedIds }
    }
    val hasDismissed = dismissedIds.isNotEmpty()
    val hasIgnored = ignoredIds.isNotEmpty()

    var showMenu by remember { mutableStateOf(false) }
    var showNeverIndex by remember { mutableStateOf(false) }
    var showPreviouslyIgnored by remember { mutableStateOf(false) }
    var isSearchActive by remember { mutableStateOf(false) }
    var filteredTrackIds by remember { mutableStateOf<Set<Long>?>(null) }

    // Auto-close screens when their list becomes empty
    LaunchedEffect(showNeverIndex, hasDismissed) {
        if (showNeverIndex && !hasDismissed) showNeverIndex = false
    }
    LaunchedEffect(showPreviouslyIgnored, hasIgnored) {
        if (showPreviouslyIgnored && !hasIgnored) showPreviouslyIgnored = false
    }

    if (showNeverIndex) {
        BackHandler { showNeverIndex = false }
        NeverIndexScreen(
            viewModel = viewModel,
            onBack = { showNeverIndex = false },
        )
        return
    }
    if (showPreviouslyIgnored) {
        BackHandler { showPreviouslyIgnored = false }
        PreviouslyIgnoredScreen(
            viewModel = viewModel,
            onBack = { showPreviouslyIgnored = false },
        )
        return
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("On-Device Indexing") },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back")
                    }
                },
                actions = {
                    // Show overflow when idle with tracks or dismissed/ignored tracks available
                    if (indexingState is IndexingService.IndexingState.Idle
                        && (visibleTracks.isNotEmpty() || hasDismissed || hasIgnored) && !isDetecting) {
                        Box {
                            IconButton(onClick = { showMenu = true }) {
                                Icon(Icons.Default.MoreVert, contentDescription = "More")
                            }
                            DropdownMenu(
                                expanded = showMenu,
                                onDismissRequest = { showMenu = false },
                            ) {
                                if (selectedCount > 0) {
                                    DropdownMenuItem(
                                        text = { Text("Never index selected") },
                                        onClick = {
                                            viewModel.dismissSelected()
                                            showMenu = false
                                        }
                                    )
                                }
                                if (hasDismissed) {
                                    DropdownMenuItem(
                                        text = { Text("View never-index list (${dismissedIds.size})") },
                                        onClick = {
                                            showNeverIndex = true
                                            showMenu = false
                                        }
                                    )
                                }
                                if (hasIgnored) {
                                    DropdownMenuItem(
                                        text = { Text("View previously ignored (${ignoredIds.size})") },
                                        onClick = {
                                            showPreviouslyIgnored = true
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
            // Only show bottom bar when idle with tracks to index
            if (indexingState is IndexingService.IndexingState.Idle
                && visibleTracks.isNotEmpty() && !isDetecting) {
                val buttonCount = if (isSearchActive && filteredTrackIds != null) {
                    selectedIds.count { it in filteredTrackIds!! }
                } else {
                    selectedCount
                }
                BottomBar(
                    selectedCount = buttonCount,
                    onStartIndexing = {
                        viewModel.startIndexing(
                            buildGraph = true,
                            autoDismissUnselected = !isSearchActive,
                            onlyIds = if (isSearchActive) filteredTrackIds else null,
                        )
                    },
                )
            }
        }
    ) { padding ->
        Box(modifier = Modifier.fillMaxSize().padding(padding)) {
            when (val state = indexingState) {
                is IndexingService.IndexingState.Idle -> {
                    if (isDetecting) {
                        DetectingContent(status = detectingStatus)
                    } else if (visibleTracks.isEmpty()) {
                        AllIndexedContent()
                    } else {
                        TrackSelectionContent(
                            tracks = visibleTracks,
                            selectedIds = selectedIds,
                            selectedCount = selectedCount,
                            onToggle = { viewModel.toggleSelection(it) },
                            onToggleAll = {
                                if (selectedCount == visibleTracks.size) {
                                    viewModel.deselectAll()
                                } else {
                                    viewModel.selectAll()
                                }
                            },
                            onToggleFiltered = { filtered ->
                                val filteredIds = filtered.map { it.powerampFileId }.toSet()
                                val allFilteredSelected = filteredIds.all { it in selectedIds }
                                if (allFilteredSelected) {
                                    viewModel.deselectIds(filteredIds)
                                } else {
                                    viewModel.selectIds(filteredIds)
                                }
                            },
                            onSearchActiveChanged = { isSearchActive = it },
                            onFilteredIdsChanged = { filteredTrackIds = it },
                            onDeselectAll = { viewModel.deselectAll() },
                        )
                    }
                }
                is IndexingService.IndexingState.Starting -> {
                    DetectingContent(status = "Starting...")
                }
                is IndexingService.IndexingState.Detecting -> {
                    DetectingContent(status = state.message)
                }
                is IndexingService.IndexingState.Processing -> {
                    ProcessingContent(state = state, onCancel = { viewModel.cancelIndexing() })
                }
                is IndexingService.IndexingState.RebuildingIndices -> {
                    RebuildingContent(state = state)
                }
                is IndexingService.IndexingState.Complete -> {
                    CompleteContent(
                        indexed = state.indexed,
                        failed = state.failed,
                        onDone = {
                            IndexingService.resetState()
                            onBack()
                        },
                    )
                    // Also reset on back button press (top bar)
                    DisposableEffect(Unit) {
                        onDispose { IndexingService.resetState() }
                    }
                }
                is IndexingService.IndexingState.Error -> {
                    ErrorContent(message = state.message, onBack = onBack)
                }
            }
        }
    }
}

/** Format ETA as "Xm Ys remaining" / "Xs remaining" for better granularity. */
private fun formatEtaText(ms: Long): String {
    val minutes = (ms / 60_000).toInt()
    return when {
        minutes >= 2 -> "About $minutes min remaining"
        minutes == 1 -> "About a minute remaining"
        ms > 10_000 -> "Less than a minute remaining"
        else -> ""
    }
}

@Composable
private fun DetectingContent(status: String = "") {
    Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            CircularProgressIndicator()
            Spacer(modifier = Modifier.height(16.dp))
            Text(
                status.ifEmpty { "Detecting unindexed tracks..." },
                style = MaterialTheme.typography.bodyMedium,
            )
        }
    }
}

@Composable
private fun AllIndexedContent() {
    Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            modifier = Modifier.padding(32.dp),
        ) {
            Text(
                "All tracks indexed",
                style = MaterialTheme.typography.headlineSmall,
                color = MaterialTheme.colorScheme.primary,
            )
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                "Every track in your Poweramp library has embeddings.",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

@Composable
private fun TrackSelectionContent(
    tracks: List<NewTrackDetector.UnindexedTrack>,
    selectedIds: Set<Long>,
    selectedCount: Int,
    onToggle: (Long) -> Unit,
    onToggleAll: () -> Unit,
    onToggleFiltered: (List<NewTrackDetector.UnindexedTrack>) -> Unit,
    onSearchActiveChanged: (Boolean) -> Unit,
    onFilteredIdsChanged: (Set<Long>?) -> Unit,
    onDeselectAll: () -> Unit,
) {
    var searchQuery by remember { mutableStateOf("") }
    var hasEngagedSearch by remember { mutableStateOf(false) }

    // Report search state changes up
    LaunchedEffect(searchQuery) {
        onSearchActiveChanged(searchQuery.isNotBlank())
    }

    val filteredTracks = remember(tracks, searchQuery) {
        if (searchQuery.isBlank()) tracks
        else {
            val q = searchQuery.lowercase()
            tracks.filter { t ->
                t.title.lowercase().contains(q) ||
                    t.artist.lowercase().contains(q) ||
                    t.album.lowercase().contains(q)
            }
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

    Column(modifier = Modifier.fillMaxSize()) {
        // Search bar
        OutlinedTextField(
            value = searchQuery,
            onValueChange = { newValue ->
                if (!hasEngagedSearch && newValue.isNotEmpty() && searchQuery.isEmpty()) {
                    hasEngagedSearch = true
                    onDeselectAll()
                }
                searchQuery = newValue
            },
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

        // Header row with parent checkbox and count
        val headerToggle = {
            if (isFiltered) {
                onToggleFiltered(filteredTracks)
            } else {
                onToggleAll()
            }
        }
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .clickable(onClick = headerToggle)
                .padding(horizontal = 16.dp, vertical = 8.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            val displayCount = if (isFiltered) filteredSelectedCount else selectedCount
            val displayTotal = if (isFiltered) filteredTracks.size else tracks.size
            TriStateCheckbox(
                state = when {
                    displayTotal == 0 -> ToggleableState.Off
                    displayCount == displayTotal -> ToggleableState.On
                    displayCount == 0 -> ToggleableState.Off
                    else -> ToggleableState.Indeterminate
                },
                onClick = headerToggle,
            )
            Spacer(modifier = Modifier.width(12.dp))
            Text(
                if (isFiltered) {
                    "${filteredTracks.size} of ${tracks.size} tracks" +
                        if (filteredSelectedCount > 0) " ($filteredSelectedCount selected)" else ""
                } else {
                    "${tracks.size} new unindexed tracks" +
                        if (selectedCount > 0) " ($selectedCount selected)" else ""
                },
                style = MaterialTheme.typography.titleSmall,
            )
        }

        HorizontalDivider()

        LazyColumn(
            modifier = Modifier.fillMaxSize(),
            contentPadding = PaddingValues(vertical = 4.dp),
        ) {
            items(filteredTracks, key = { it.powerampFileId }) { track ->
                TrackRow(
                    track = track,
                    isSelected = track.powerampFileId in selectedIds,
                    onToggle = { onToggle(track.powerampFileId) },
                )
            }
        }
    }
}

@OptIn(ExperimentalFoundationApi::class)
@Composable
private fun TrackRow(
    track: NewTrackDetector.UnindexedTrack,
    isSelected: Boolean,
    onToggle: () -> Unit,
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onToggle)
            .padding(horizontal = 16.dp, vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Checkbox(
            checked = isSelected,
            onCheckedChange = null,
        )
        Spacer(modifier = Modifier.width(12.dp))
        Column(modifier = Modifier.weight(1f)) {
            Text(
                text = track.title.ifEmpty { "Unknown" },
                style = MaterialTheme.typography.bodyMedium,
                modifier = Modifier.basicMarquee(iterations = 1, initialDelayMillis = 1500),
                maxLines = 1,
            )
            val subtitle = buildString {
                if (track.artist.isNotEmpty()) append(track.artist)
                val durMin = track.durationMs / 60000
                val durSec = (track.durationMs % 60000) / 1000
                if (isNotEmpty()) append(" \u00b7 ")
                append("$durMin:${durSec.toString().padStart(2, '0')}")
            }
            Text(
                text = subtitle,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.basicMarquee(iterations = 1, initialDelayMillis = 1500),
                maxLines = 1,
            )
        }
    }
}

@OptIn(ExperimentalFoundationApi::class)
@Composable
private fun ProcessingContent(
    state: IndexingService.IndexingState.Processing,
    onCancel: () -> Unit,
) {
    Column(
        modifier = Modifier.fillMaxSize().padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        if (state.passName.isNotEmpty()) {
            Text(
                text = state.passName,
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Spacer(modifier = Modifier.height(4.dp))
        }
        Text(
            "${state.current} / ${state.total}",
            style = MaterialTheme.typography.headlineMedium,
            color = MaterialTheme.colorScheme.primary,
        )
        Spacer(modifier = Modifier.height(16.dp))
        LinearProgressIndicator(
            progress = { state.progressFraction.coerceIn(0f, 1f) },
            modifier = Modifier.fillMaxWidth(),
        )
        Spacer(modifier = Modifier.height(12.dp))
        Text(
            text = state.trackName,
            style = MaterialTheme.typography.bodyMedium,
            modifier = Modifier.basicMarquee(iterations = 1, initialDelayMillis = 1500),
            maxLines = 1,
        )
        if (state.detail.isNotEmpty()) {
            Spacer(modifier = Modifier.height(2.dp))
            Text(
                text = state.detail,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        if (state.estimatedRemainingMs > 0) {
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text = formatEtaText(state.estimatedRemainingMs),
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        Spacer(modifier = Modifier.height(24.dp))
        OutlinedButton(onClick = onCancel) {
            Text("Cancel")
        }
    }
}

@Composable
private fun RebuildingContent(state: IndexingService.IndexingState.RebuildingIndices) {
    Column(
        modifier = Modifier.fillMaxSize().padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        if (state.phaseName.isNotEmpty()) {
            Text(
                text = state.phaseName,
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Spacer(modifier = Modifier.height(8.dp))
        }
        if (state.progressFraction >= 0f) {
            LinearProgressIndicator(
                progress = { state.progressFraction.coerceIn(0f, 1f) },
                modifier = Modifier.fillMaxWidth(),
            )
        } else {
            LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
        }
        Spacer(modifier = Modifier.height(12.dp))
        Text(
            text = state.message,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        if (state.estimatedRemainingMs > 0) {
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text = formatEtaText(state.estimatedRemainingMs),
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

@Composable
private fun ErrorContent(message: String, onBack: () -> Unit) {
    Box(modifier = Modifier.fillMaxSize().padding(24.dp), contentAlignment = Alignment.Center) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
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
                Text("Back")
            }
        }
    }
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
            val message = if (failed > 0) {
                "$indexed tracks indexed, $failed failed"
            } else {
                "$indexed tracks indexed"
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
    var localSelected by remember { mutableStateOf(emptySet<Long>()) }

    // Clean up local selection if tracks change
    LaunchedEffect(dismissedTracks) {
        val validIds = dismissedTracks.map { it.powerampFileId }.toSet()
        localSelected = localSelected.intersect(validIds)
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Never-Index List") },
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
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 16.dp, vertical = 10.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Spacer(modifier = Modifier.weight(1f))
                        Button(onClick = {
                            viewModel.restoreFromDismissed(localSelected)
                            localSelected = emptySet()
                        }) {
                            Text("Restore selected (${localSelected.size})")
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
                        .clickable(onClick = toggleAll)
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
                        onClick = toggleAll,
                    )
                    Spacer(modifier = Modifier.width(12.dp))
                    Text(
                        "${dismissedTracks.size} never-index tracks" +
                            if (localSelected.isNotEmpty()) " (${localSelected.size} selected)" else "",
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
private fun PreviouslyIgnoredScreen(
    viewModel: IndexingViewModel,
    onBack: () -> Unit,
) {
    val ignoredTracks = remember(viewModel.ignoredIds.collectAsState().value) {
        viewModel.getIgnoredTracks()
    }
    var localSelected by remember { mutableStateOf(emptySet<Long>()) }

    // Clean up local selection if tracks change
    LaunchedEffect(ignoredTracks) {
        val validIds = ignoredTracks.map { it.powerampFileId }.toSet()
        localSelected = localSelected.intersect(validIds)
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Previously Ignored") },
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
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 16.dp, vertical = 10.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        OutlinedButton(onClick = {
                            viewModel.moveIgnoredToNeverIndex(localSelected)
                            localSelected = emptySet()
                        }) {
                            Text("Never index (${localSelected.size})")
                        }
                        Spacer(modifier = Modifier.weight(1f))
                        Button(onClick = {
                            viewModel.restoreFromIgnored(localSelected)
                            localSelected = emptySet()
                        }) {
                            Text("Restore selected (${localSelected.size})")
                        }
                    }
                }
            }
        }
    ) { padding ->
        if (ignoredTracks.isEmpty()) {
            Box(
                modifier = Modifier.fillMaxSize().padding(padding),
                contentAlignment = Alignment.Center,
            ) {
                Text(
                    "No previously ignored tracks",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        } else {
            Column(modifier = Modifier.fillMaxSize().padding(padding)) {
                val allIds = remember(ignoredTracks) {
                    ignoredTracks.map { it.powerampFileId }.toSet()
                }
                val allSelected = localSelected.size == ignoredTracks.size && ignoredTracks.isNotEmpty()
                val toggleAll = {
                    localSelected = if (allSelected) emptySet() else allIds
                }
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clickable(onClick = toggleAll)
                        .padding(horizontal = 16.dp, vertical = 8.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    TriStateCheckbox(
                        state = when {
                            ignoredTracks.isEmpty() -> ToggleableState.Off
                            allSelected -> ToggleableState.On
                            localSelected.isEmpty() -> ToggleableState.Off
                            else -> ToggleableState.Indeterminate
                        },
                        onClick = toggleAll,
                    )
                    Spacer(modifier = Modifier.width(12.dp))
                    Text(
                        "${ignoredTracks.size} previously ignored tracks" +
                            if (localSelected.isNotEmpty()) " (${localSelected.size} selected)" else "",
                        style = MaterialTheme.typography.titleSmall,
                    )
                }
                HorizontalDivider()
                LazyColumn(
                    modifier = Modifier.fillMaxSize(),
                    contentPadding = PaddingValues(vertical = 4.dp),
                ) {
                    items(ignoredTracks, key = { it.powerampFileId }) { track ->
                        TrackRow(
                            track = track,
                            isSelected = track.powerampFileId in localSelected,
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

@Composable
private fun BottomBar(
    selectedCount: Int,
    onStartIndexing: () -> Unit,
) {
    Surface(tonalElevation = 3.dp) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = 12.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(
                "CLaMP3 indexing",
                style = MaterialTheme.typography.titleSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.weight(1f),
            )
            FilledTonalButton(
                onClick = onStartIndexing,
                enabled = selectedCount > 0,
            ) {
                Text("Start ($selectedCount)")
            }
        }
    }
}
