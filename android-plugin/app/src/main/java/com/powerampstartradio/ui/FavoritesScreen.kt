package com.powerampstartradio.ui

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.powerampstartradio.R

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun FavoritesScreen(viewModel: MainViewModel, onBack: () -> Unit, onStartRadio: () -> Unit) {
    val entries by viewModel.favorites.collectAsState()
    val loading by viewModel.favoritesLoading.collectAsState()
    val error by viewModel.favoritesError.collectAsState()
    val state by viewModel.favoriteMatches.collectAsState()
    val revision by viewModel.favoritesRevision.collectAsState()
    val newSeedKeys by viewModel.favoriteNewSeedKeys.collectAsState()
    val newMatchesStatus by viewModel.favoriteNewMatchesStatus.collectAsState()
    val resultCount by viewModel.favoriteResultCount.collectAsState()
    val database by viewModel.databaseInfo.collectAsState()
    val seed = state.seed
    val canStartSeedRadio = state.resolvedSeedTrackId != null &&
        state.generationId != null && !state.loading && state.error == null
    val startSeedRadio = {
        if (viewModel.startRadioFromFavoriteSeed()) onStartRadio()
    }
    val favoritesScroll = rememberLazyListState()
    val matchesScroll = key(seed?.key) { rememberLazyListState() }
    val nowEpochSecond = remember(state.matches) { System.currentTimeMillis() / 1_000L }
    LaunchedEffect(revision, seed?.key, database?.generationId, database?.providerGenerationId, resultCount) {
        if (seed == null) viewModel.refreshFavorites() else viewModel.stopFavoriteNewMatchScan()
    }
    DisposableEffect(viewModel) {
        onDispose { viewModel.stopFavoriteNewMatchScan() }
    }
    LaunchedEffect(database?.generationId) {
        if (seed != null && state.generationId != null &&
            database?.generationId != state.generationId
        ) viewModel.openFavorite(seed)
    }

    Scaffold(
        contentWindowInsets = WindowInsets.safeDrawing,
        topBar = {
            TopAppBar(
                title = { Text(if (seed == null) "Favorites" else "Nearest matches") },
                navigationIcon = {
                    IconButton(onClick = { if (seed != null) viewModel.closeFavorite() else onBack() }) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, "Back")
                    }
                },
                actions = {
                    IconButton(onClick = {
                        if (seed == null) viewModel.refreshFavorites() else viewModel.openFavorite(seed)
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
                        enabled = canStartSeedRadio,
                        onClickLabel = "Start radio from this seed",
                        onClick = startSeedRadio,
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
                        if (seed.trackId != null) {
                            IconButton(onClick = startSeedRadio, enabled = canStartSeedRadio) {
                                Icon(painterResource(R.drawable.ic_radio_waves),
                                    contentDescription = "Start radio from ${seed.label}",
                                    tint = if (canStartSeedRadio) MaterialTheme.colorScheme.primary
                                        else MaterialTheme.colorScheme.onSurfaceVariant)
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
                items(state.matches, key = { it.track.id }) { match ->
                    Row(
                        Modifier.fillMaxWidth().clickable {
                            viewModel.startRadioFromFavorite(match)
                            onStartRadio()
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
                                    color = MaterialTheme.colorScheme.onSurfaceVariant)
                                indexAddedAgeLabel(match.track.indexAddedAtEpochSecond, nowEpochSecond)?.let {
                                    Text(it, style = MaterialTheme.typography.labelSmall,
                                        color = MaterialTheme.colorScheme.onSurfaceVariant)
                                }
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
