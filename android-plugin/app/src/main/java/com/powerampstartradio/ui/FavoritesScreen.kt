package com.powerampstartradio.ui

import androidx.compose.foundation.clickable
import androidx.compose.foundation.background
import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyListState
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.ArrowDropDown
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Search
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.stateDescription
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.powerampstartradio.R
import com.powerampstartradio.poweramp.PowerampTrack
import com.powerampstartradio.RecordingPickerSheet

@OptIn(ExperimentalMaterial3Api::class, ExperimentalFoundationApi::class)
@Composable
fun FavoritesScreen(viewModel: MainViewModel, currentTrack: PowerampTrack?, onBack: () -> Unit, onStartRadio: () -> Unit) {
    val entries by viewModel.favorites.collectAsState()
    val loading by viewModel.favoritesLoading.collectAsState()
    val error by viewModel.favoritesError.collectAsState()
    val state by viewModel.favoriteMatches.collectAsState()
    val revision by viewModel.favoritesRevision.collectAsState()
    val newSeedKeys by viewModel.favoriteNewSeedKeys.collectAsState()
    val newMatchesStatus by viewModel.favoriteNewMatchesStatus.collectAsState()
    val resultCount by viewModel.favoriteResultCount.collectAsState()
    val database by viewModel.databaseInfo.collectAsState()
    val selectionMode by viewModel.selectionMode.collectAsState()
    val songQuery by viewModel.favoriteSongQuery.collectAsState()
    val songLookup by viewModel.favoriteSongLookup.collectAsState()
    var showSongSearch by rememberSaveable { mutableStateOf(false) }
    val seed = state.seed
    val canStartSeedRadio = state.resolvedSeedTrackId != null &&
        state.generationId != null && !state.loading && state.error == null
    val canQueueDisplayed = state.binding != null && !state.loading && state.error == null &&
        (state.matches.isNotEmpty() || canStartSeedRadio)
    var showQueueMenu by remember(seed?.key, state.generationId) { mutableStateOf(false) }
    val favoritesScroll = rememberLazyListState()
    val matchScrollPositions = remember { mutableMapOf<String, LazyListState>() }
    val matchesScroll = remember(seed?.key) {
        matchScrollPositions.getOrPut(seed?.key.orEmpty()) { LazyListState() }
    }
    var matchesSort by rememberSaveable { mutableStateOf(FavoriteMatchesSort.CLOSEST_FIRST) }
    val sortedMatches = remember(state.matches, matchesSort) {
        sortFavoriteMatches(state.matches, matchesSort)
    }
    val nowEpochSecond = remember(state.matches) { System.currentTimeMillis() / 1_000L }
    LaunchedEffect(matchesSort) { matchesScroll.scrollToItem(0) }
    LaunchedEffect(revision, seed?.key, database?.generationId, database?.providerGenerationId, resultCount, currentTrack?.realId) {
        if (seed == null) viewModel.refreshFavorites(currentTrack) else viewModel.stopFavoriteNewMatchScan()
    }
    DisposableEffect(viewModel) {
        onDispose {
            viewModel.stopFavoriteNewMatchScan()
            viewModel.dismissFavoriteSongSearch()
        }
    }
    LaunchedEffect(database?.generationId) {
        if (seed != null && state.generationId != null &&
            database?.generationId != state.generationId
        ) viewModel.openFavorite(seed)
    }

    if (showSongSearch) RecordingPickerSheet(
        query = songQuery, state = songLookup,
        onDismiss = { showSongSearch = false; viewModel.dismissFavoriteSongSearch() },
        onRetry = { viewModel.searchFavoriteSongs(songQuery) },
        onChoose = { track ->
            if (viewModel.openFavoriteSongSearchResult(track)) {
                showSongSearch = false
                viewModel.dismissFavoriteSongSearch()
            }
        },
        onQueryChange = viewModel::searchFavoriteSongs,
    )

    Scaffold(
        contentWindowInsets = WindowInsets.safeDrawing,
        topBar = {
            TopAppBar(
                title = { Text(if (seed == null) "Favorites" else "Nearest matches") },
                navigationIcon = {
                    IconButton(onClick = { if (seed != null) viewModel.backFromFavorite() else onBack() }) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, "Back")
                    }
                },
                actions = {
                    if (seed == null) IconButton(onClick = { showSongSearch = true }) {
                        Icon(Icons.Default.Search, "Find a song in your indexed library")
                    }
                    IconButton(onClick = {
                        if (seed == null) viewModel.refreshFavorites(currentTrack) else viewModel.openFavorite(seed)
                    }, enabled = !loading && !state.loading) {
                        Icon(Icons.Default.Refresh, "Refresh")
                    }
                },
            )
        },
    ) { padding ->
        LazyColumn(Modifier.fillMaxSize().padding(padding),
            state = if (seed == null) favoritesScroll else matchesScroll) {
            if (seed == null) {
                if (currentTrack != null) item {
                    Row(Modifier.fillMaxWidth().clickable {
                        viewModel.openNowPlayingFavorite(currentTrack)
                    }.padding(horizontal = 16.dp, vertical = 12.dp),
                        verticalAlignment = Alignment.CenterVertically) {
                      Column(Modifier.weight(1f)) {
                        Text("Now Playing", style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.primary)
                        Text(currentTrack.title, style = MaterialTheme.typography.bodyMedium,
                            maxLines = 2, overflow = TextOverflow.Ellipsis)
                        val details = listOfNotNull(currentTrack.artist, currentTrack.album)
                            .filter(String::isNotBlank).joinToString(" \u00b7 ")
                        if (details.isNotEmpty()) Text(details, style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            maxLines = 1, overflow = TextOverflow.Ellipsis)
                      }
                      if ("poweramp:${currentTrack.realId}" in newSeedKeys) Text("New",
                          modifier = Modifier.padding(start = 12.dp),
                          style = MaterialTheme.typography.labelSmall,
                          color = MaterialTheme.colorScheme.tertiary)
                    }
                    HorizontalDivider()
                }
                item {
                    Row(Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 8.dp),
                        horizontalArrangement = Arrangement.SpaceBetween) {
                        Text(newMatchesStatus ?: "${entries.size} seeds",
                            modifier = Modifier.weight(1f).padding(end = 12.dp),
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            maxLines = 1, overflow = TextOverflow.Ellipsis)
                        Text("Requests", style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                }
                if (loading) item { FavoritesProgress("Reading seed request counts") }
                if (error != null) item { FavoritesMessage(error!!) }
                if (!loading && error == null && entries.isEmpty()) {
                    item { FavoritesMessage("Your seeds appear here after you queue a radio or Find Music search.") }
                }
                items(entries, key = { it.seed.key }) { entry ->
                    Row(
                        Modifier.fillMaxWidth().clickable { viewModel.openFavorite(entry.seed) }
                            .heightIn(min = 48.dp).padding(horizontal = 16.dp, vertical = 6.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Column(Modifier.weight(1f)) {
                            Text(entry.seed.label, style = MaterialTheme.typography.bodyMedium,
                                maxLines = 2, overflow = TextOverflow.Ellipsis)
                            Text(if (entry.seed.text != null) "Text seed" else entry.seed.artist.orEmpty(),
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                                maxLines = 1, overflow = TextOverflow.Ellipsis)
                        }
                        Spacer(Modifier.width(16.dp))
                        Column(horizontalAlignment = Alignment.End,
                            modifier = Modifier.widthIn(min = 40.dp)) {
                            if (entry.seed.key in newSeedKeys) Text("New",
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.tertiary)
                            Text("${entry.count}", style = MaterialTheme.typography.bodyMedium,
                                fontFamily = FontFamily.Monospace, textAlign = TextAlign.End,
                                color = MaterialTheme.colorScheme.primary)
                        }
                    }
                }
            } else {
                item {
                    Row(Modifier.fillMaxWidth().clickable(
                        enabled = canQueueDisplayed,
                        onClickLabel = "Queue options for this seed",
                        onClick = { showQueueMenu = true },
                    ).padding(start = 16.dp, end = 4.dp, top = 10.dp, bottom = 10.dp),
                        verticalAlignment = Alignment.CenterVertically) {
                        Column(Modifier.weight(1f)) {
                            Text(seed.label, style = MaterialTheme.typography.titleMedium)
                            if (seed.text == null && !seed.artist.isNullOrBlank()) {
                                Text(seed.artist, style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant)
                            }
                            Spacer(Modifier.height(4.dp))
                            val scope = if (state.loading || state.error != null) {
                                "Cosine nearest \u00b7 full library"
                            } else {
                                "${state.matches.size} closest \u00b7 full library" +
                                    state.matches.count { it.isNew }.takeIf { it > 0 }
                                        ?.let { " \u00b7 $it new" }.orEmpty()
                            }
                            Text(scope, style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant)
                        }
                        Box {
                            IconButton(onClick = { showQueueMenu = true }, enabled = canQueueDisplayed) {
                                if (seed.text == null) Icon(painterResource(R.drawable.ic_radio_waves),
                                        contentDescription = "Queue options for ${seed.label}",
                                        tint = if (canQueueDisplayed) MaterialTheme.colorScheme.primary
                                            else MaterialTheme.colorScheme.onSurfaceVariant)
                                else Icon(Icons.Default.Add, contentDescription = "Queue options for ${seed.label}",
                                    tint = if (canQueueDisplayed) MaterialTheme.colorScheme.primary
                                        else MaterialTheme.colorScheme.onSurfaceVariant)
                            }
                            DropdownMenu(expanded = showQueueMenu, onDismissRequest = { showQueueMenu = false },
                                modifier = Modifier.widthIn(min = 260.dp, max = 320.dp)) {
                                if (canStartSeedRadio) DropdownMenuItem(
                                    text = { Column {
                                        Text("Radio")
                                        Text("${SelectionControlText.modeLabel(selectionMode)} \u00b7 current settings",
                                            style = MaterialTheme.typography.bodySmall,
                                            color = MaterialTheme.colorScheme.onSurfaceVariant)
                                    } },
                                    onClick = {
                                        showQueueMenu = false
                                        if (viewModel.startRadioFromFavoriteSeed()) onStartRadio()
                                    },
                                )
                                DropdownMenuItem(
                                    text = { Column {
                                        Text("Queue displayed")
                                        Text((if (canStartSeedRadio) "Seed first \u00b7 " else "") +
                                            "${state.matches.size + if (canStartSeedRadio) 1 else 0} tracks \u00b7 " +
                                            matchesSort.label.lowercase(),
                                            style = MaterialTheme.typography.bodySmall,
                                            color = MaterialTheme.colorScheme.onSurfaceVariant)
                                    } },
                                    onClick = {
                                        showQueueMenu = false
                                        if (viewModel.queueFavoriteResults(matchesSort)) onStartRadio()
                                    },
                                )
                            }
                        }
                    }
                    HorizontalDivider()
                }
                if (state.loading) item { FavoritesProgress(state.status ?: "Ranking nearest recordings") }
                if (state.error != null) item { FavoritesMessage(state.error!!) }
                if (!state.loading && state.error == null && state.matches.isEmpty()) {
                    item { FavoritesMessage("No other indexed recordings are available in Poweramp.") }
                }
                if (state.matches.isNotEmpty() && !state.loading && state.error == null) {
                    stickyHeader {
                        Row(Modifier.fillMaxWidth().background(MaterialTheme.colorScheme.surface)
                            .padding(horizontal = 16.dp), verticalAlignment = Alignment.CenterVertically) {
                            Text("Track", modifier = Modifier.weight(1f),
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant)
                            Spacer(Modifier.width(12.dp))
                            FavoriteSortHeader("Nearest", !matchesSort.byAddedDate, matchesSort.ascending,
                                Modifier.width(64.dp), "Sort by cosine closeness",
                                if (matchesSort == FavoriteMatchesSort.FARTHEST_FIRST) "Farthest first" else "Closest first") {
                                matchesSort = matchesSort.toggleCloseness()
                            }
                            Spacer(Modifier.width(8.dp))
                            FavoriteSortHeader("Added", matchesSort.byAddedDate, matchesSort.ascending,
                                Modifier.width(100.dp), "Sort by index added date",
                                if (matchesSort == FavoriteMatchesSort.OLDEST_FIRST) "Oldest first" else "Newest first") {
                                matchesSort = matchesSort.toggleAddedDate()
                            }
                        }
                    }
                }
                items(sortedMatches, key = { it.track.id }) { match ->
                    Row(
                        Modifier.fillMaxWidth().clickable {
                            viewModel.openFavoriteMatch(match)
                        }.heightIn(min = 48.dp).padding(horizontal = 16.dp, vertical = 6.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Column(Modifier.weight(1f)) {
                            Text(match.track.title ?: "Untitled", style = MaterialTheme.typography.bodyMedium,
                                maxLines = 2, overflow = TextOverflow.Ellipsis)
                            val details = listOfNotNull(match.track.artist, match.track.album)
                                .filter(String::isNotBlank).joinToString(" \u00b7 ")
                            Text(details, style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                                maxLines = 1, overflow = TextOverflow.Ellipsis)
                        }
                        Spacer(Modifier.width(12.dp))
                        Column(horizontalAlignment = Alignment.End, modifier = Modifier.widthIn(min = 40.dp)) {
                            if (match.isNew) Text("New", style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.tertiary)
                            Row(verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                                Text("#${match.rank}", style = MaterialTheme.typography.labelMedium,
                                    fontFamily = FontFamily.Monospace,
                                    modifier = Modifier.width(64.dp), textAlign = TextAlign.End,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant)
                                Text(indexAddedAgeLabel(match.track.indexAddedAtEpochSecond, nowEpochSecond).orEmpty(),
                                    modifier = Modifier.width(100.dp), textAlign = TextAlign.End,
                                    maxLines = 1, style = MaterialTheme.typography.labelSmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant)
                            }
                        }
                    }
                }
            }
            item { Spacer(Modifier.height(16.dp)) }
        }
    }
}

@Composable
private fun FavoriteSortHeader(
    label: String,
    active: Boolean,
    ascending: Boolean,
    modifier: Modifier,
    actionLabel: String,
    orderLabel: String,
    onClick: () -> Unit,
) {
    val color = if (active) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurfaceVariant
    Row(modifier.heightIn(min = 40.dp)
        .clickable(role = Role.Button, onClickLabel = actionLabel, onClick = onClick)
        .semantics { stateDescription = if (active) orderLabel else "Not sorted by this column" },
        horizontalArrangement = Arrangement.End, verticalAlignment = Alignment.CenterVertically) {
        Text(label, style = MaterialTheme.typography.labelSmall, color = color)
        if (active) Icon(Icons.Default.ArrowDropDown, contentDescription = orderLabel,
            modifier = Modifier.size(16.dp).rotate(if (ascending) 180f else 0f), tint = color)
        else Spacer(Modifier.width(16.dp))
    }
}

@Composable
private fun FavoritesProgress(message: String) {
    Column(Modifier.fillMaxWidth().padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        LinearProgressIndicator(Modifier.fillMaxWidth())
        Text(message, style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

@Composable
private fun FavoritesMessage(message: String) {
    Text(message, Modifier.padding(16.dp), style = MaterialTheme.typography.bodyMedium,
        color = MaterialTheme.colorScheme.onSurfaceVariant)
}
