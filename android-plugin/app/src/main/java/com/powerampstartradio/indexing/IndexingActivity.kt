package com.powerampstartradio.indexing

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.KeyboardArrowDown
import androidx.compose.material.icons.filled.KeyboardArrowUp
import androidx.compose.material.icons.filled.MoreVert
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
    val isDetecting by viewModel.isDetecting.collectAsState()
    val detectingStatus by viewModel.detectingStatus.collectAsState()
    val indexingState by viewModel.indexingState.collectAsState()
    val hasSvdMatrix by viewModel.hasSvdMatrix.collectAsState()

    // Re-detect when activity resumes (e.g. user replaced DB while app was backgrounded).
    // detectUnindexed() has its own cache check so this is cheap when nothing changed.
    LifecycleEventEffect(Lifecycle.Event.ON_RESUME) {
        viewModel.detectUnindexed()
    }

    val visibleTracks = remember(unindexedTracks, dismissedIds) {
        unindexedTracks.filter { it.powerampFileId !in dismissedIds }
    }
    val selectedCount = remember(selectedIds, visibleTracks) {
        visibleTracks.count { it.powerampFileId in selectedIds }
    }
    val hasDismissed = dismissedIds.isNotEmpty()

    var showMenu by remember { mutableStateOf(false) }

    var recluster by remember { mutableStateOf(false) }

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
                    // Always show overflow when there are visible tracks (idle state)
                    if (indexingState is IndexingService.IndexingState.Idle
                        && visibleTracks.isNotEmpty() && !isDetecting) {
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
                                        text = { Text("Show hidden tracks") },
                                        onClick = {
                                            viewModel.clearDismissed()
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
                BottomBar(
                    selectedCount = selectedCount,
                    hasSvdMatrix = hasSvdMatrix,
                    recluster = recluster,
                    onReclusterChanged = { recluster = it },
                    onStartIndexing = {
                        val tier = when {
                            !hasSvdMatrix -> IndexingService.FusionTier.FULL_REFUSION
                            recluster -> IndexingService.FusionTier.RECLUSTER
                            else -> IndexingService.FusionTier.QUICK_UPDATE
                        }
                        viewModel.startIndexing(
                            fusionOptions = IndexingService.FusionOptions(tier, buildKnnGraph = true)
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
                    // Auto-detect remaining unindexed tracks
                    LaunchedEffect(Unit) {
                        viewModel.detectUnindexed(forceRefresh = true)
                    }
                    DetectingContent(
                        status = if (state.failed > 0)
                            "${state.indexed} indexed, ${state.failed} failed \u2014 refreshing..."
                        else "Refreshing track list...",
                    )
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
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text("All tracks indexed", style = MaterialTheme.typography.headlineSmall)
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
) {
    Column(modifier = Modifier.fillMaxSize()) {
        // Header row with parent checkbox and count
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .clickable(onClick = onToggleAll)
                .padding(horizontal = 16.dp, vertical = 8.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            TriStateCheckbox(
                state = when {
                    selectedCount == tracks.size -> ToggleableState.On
                    selectedCount == 0 -> ToggleableState.Off
                    else -> ToggleableState.Indeterminate
                },
                onClick = onToggleAll,
            )
            Spacer(modifier = Modifier.width(12.dp))
            Text(
                "${tracks.size} unindexed tracks" +
                    if (selectedCount > 0) " ($selectedCount selected)" else "",
                style = MaterialTheme.typography.titleSmall,
            )
        }

        HorizontalDivider()

        LazyColumn(
            modifier = Modifier.fillMaxSize(),
            contentPadding = PaddingValues(vertical = 4.dp),
        ) {
            items(tracks, key = { it.powerampFileId }) { track ->
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
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
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
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
            )
        }
    }
}

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
            maxLines = 1,
            overflow = TextOverflow.Ellipsis,
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
private fun BottomBar(
    selectedCount: Int,
    hasSvdMatrix: Boolean,
    recluster: Boolean,
    onReclusterChanged: (Boolean) -> Unit,
    onStartIndexing: () -> Unit,
) {
    var expanded by remember { mutableStateOf(false) }

    Surface(tonalElevation = 3.dp) {
        Column(modifier = Modifier.fillMaxWidth()) {
            // Collapsed header — always visible
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .then(
                        if (hasSvdMatrix) Modifier.clickable { expanded = !expanded }
                        else Modifier
                    )
                    .padding(horizontal = 16.dp, vertical = 10.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        if (hasSvdMatrix) "Incremental update" else "Full fusion",
                        style = MaterialTheme.typography.bodyMedium,
                    )
                    Text(
                        if (hasSvdMatrix) "Usually under a minute"
                        else "Several minutes, depending on library size",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
                if (hasSvdMatrix) {
                    Icon(
                        if (expanded) Icons.Default.KeyboardArrowDown
                        else Icons.Default.KeyboardArrowUp,
                        contentDescription = if (expanded) "Collapse" else "Expand",
                        tint = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                }
                Button(
                    onClick = onStartIndexing,
                    enabled = selectedCount > 0,
                ) {
                    Text("Start ($selectedCount)")
                }
            }

            // Expanded options — only when SVD exists
            if (hasSvdMatrix) {
                AnimatedVisibility(visible = expanded) {
                    Column {
                        HorizontalDivider(modifier = Modifier.padding(horizontal = 16.dp))

                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .clickable { onReclusterChanged(!recluster) }
                                .padding(horizontal = 16.dp, vertical = 6.dp),
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            Checkbox(
                                checked = recluster,
                                onCheckedChange = onReclusterChanged,
                            )
                            Spacer(modifier = Modifier.width(4.dp))
                            Column(modifier = Modifier.weight(1f)) {
                                Text(
                                    "Re-cluster all tracks",
                                    style = MaterialTheme.typography.bodyMedium,
                                )
                                Text(
                                    "Rebuilds track clusters from scratch. Useful after adding many tracks over several sessions \u2014 if the Explorer (Random Walk) mode stops surfacing good connections, re-clustering helps. Adds a few minutes.",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                                )
                            }
                        }
                    }
                }
            }
        }
    }
}
