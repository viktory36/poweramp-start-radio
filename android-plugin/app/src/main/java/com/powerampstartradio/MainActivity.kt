package com.powerampstartradio

import android.content.Intent
import android.content.IntentFilter
import android.net.Uri
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import com.powerampstartradio.data.EmbeddingDatabase
import com.powerampstartradio.poweramp.PowerampHelper
import com.powerampstartradio.poweramp.PowerampReceiver
import com.powerampstartradio.poweramp.PowerampTrack
import com.powerampstartradio.services.RadioService
import com.powerampstartradio.ui.theme.PowerampStartRadioTheme
import java.io.File

class MainActivity : ComponentActivity() {

    companion object {
        private const val TAG = "MainActivity"
    }

    private val trackReceiver = PowerampReceiver()

    // Callback to refresh permission state when activity resumes
    private var onResumeCallback: (() -> Unit)? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Register for Poweramp broadcasts
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
    onRegisterResumeCallback: ((callback: () -> Unit) -> Unit)? = null
) {
    val context = LocalContext.current

    // State
    var currentTrack by remember { mutableStateOf<PowerampTrack?>(PowerampReceiver.currentTrack) }
    var databaseInfo by remember { mutableStateOf<DatabaseInfo?>(null) }
    var statusMessage by remember { mutableStateOf("") }
    var numTracks by remember { mutableStateOf(50f) }
    var hasPowerampPermission by remember { mutableStateOf(false) }

    // Function to refresh permission state
    val refreshPermission: () -> Unit = {
        hasPowerampPermission = PowerampHelper.canAccessData(context)
        if (hasPowerampPermission) {
            statusMessage = ""
        }
    }

    // Register callback for activity resume
    LaunchedEffect(Unit) {
        onRegisterResumeCallback?.invoke(refreshPermission)
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

    // Load database info and check Poweramp permission on launch
    LaunchedEffect(Unit) {
        databaseInfo = loadDatabaseInfo(context)
        hasPowerampPermission = PowerampHelper.canAccessData(context)
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
                databaseInfo = loadDatabaseInfo(context)
                statusMessage = "Database imported successfully!"
            } catch (e: Exception) {
                statusMessage = "Import failed: ${e.message}"
                Log.e("MainActivity", "Import failed", e)
            }
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Poweramp Start Radio") }
            )
        }
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(16.dp)
                .verticalScroll(rememberScrollState()),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            // Current Track Card
            Card(
                modifier = Modifier.fillMaxWidth()
            ) {
                Column(
                    modifier = Modifier.padding(16.dp)
                ) {
                    Text(
                        text = "Now Playing",
                        style = MaterialTheme.typography.titleMedium
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    if (currentTrack != null) {
                        Text(
                            text = currentTrack!!.title,
                            style = MaterialTheme.typography.bodyLarge
                        )
                        Text(
                            text = currentTrack!!.artist ?: "Unknown Artist",
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                        Text(
                            text = currentTrack!!.album ?: "",
                            style = MaterialTheme.typography.bodySmall,
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

            // Start Radio Button
            Button(
                onClick = {
                    if (currentTrack != null && databaseInfo != null) {
                        RadioService.startRadio(context, numTracks.toInt())
                        statusMessage = "Starting radio..."
                    } else if (currentTrack == null) {
                        statusMessage = "Play a song in Poweramp first"
                    } else {
                        statusMessage = "Import an embedding database first"
                    }
                },
                modifier = Modifier.fillMaxWidth(),
                enabled = currentTrack != null && databaseInfo != null
            ) {
                Text("Start Radio")
            }

            // Number of tracks slider
            Card(
                modifier = Modifier.fillMaxWidth()
            ) {
                Column(
                    modifier = Modifier.padding(16.dp)
                ) {
                    Text(
                        text = "Number of tracks: ${numTracks.toInt()}",
                        style = MaterialTheme.typography.titleMedium
                    )
                    Slider(
                        value = numTracks,
                        onValueChange = { numTracks = it },
                        valueRange = 10f..100f,
                        steps = 8
                    )
                }
            }

            // Poweramp Permission Card
            if (!hasPowerampPermission) {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.errorContainer
                    )
                ) {
                    Column(
                        modifier = Modifier.padding(16.dp)
                    ) {
                        Text(
                            text = "Poweramp Access Required",
                            style = MaterialTheme.typography.titleMedium
                        )
                        Spacer(modifier = Modifier.height(8.dp))
                        Text(
                            text = "This app needs permission to access Poweramp's library to queue similar tracks.",
                            style = MaterialTheme.typography.bodyMedium
                        )
                        Spacer(modifier = Modifier.height(8.dp))
                        Button(
                            onClick = {
                                PowerampHelper.requestDataPermission(context)
                                statusMessage = "Permission requested. Please grant access in Poweramp, then return here."
                            },
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Text("Grant Poweramp Access")
                        }
                    }
                }
            }

            // Database Info Card
            Card(
                modifier = Modifier.fillMaxWidth()
            ) {
                Column(
                    modifier = Modifier.padding(16.dp)
                ) {
                    Text(
                        text = "Embedding Database",
                        style = MaterialTheme.typography.titleMedium
                    )
                    Spacer(modifier = Modifier.height(8.dp))

                    if (databaseInfo != null) {
                        Text("Tracks: ${databaseInfo!!.trackCount}")
                        Text("Version: ${databaseInfo!!.version ?: "Unknown"}")
                        Text("Size: ${databaseInfo!!.sizeKb} KB")
                    } else {
                        Text(
                            text = "No database imported",
                            color = MaterialTheme.colorScheme.error
                        )
                    }

                    Spacer(modifier = Modifier.height(8.dp))

                    OutlinedButton(
                        onClick = {
                            importLauncher.launch(arrayOf("application/octet-stream", "*/*"))
                        },
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Text(if (databaseInfo != null) "Replace Database" else "Import Database")
                    }
                }
            }

            // Status message
            if (statusMessage.isNotEmpty()) {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.secondaryContainer
                    )
                ) {
                    Text(
                        text = statusMessage,
                        modifier = Modifier.padding(16.dp),
                        style = MaterialTheme.typography.bodyMedium
                    )
                }
            }

            // Instructions
            Card(
                modifier = Modifier.fillMaxWidth()
            ) {
                Column(
                    modifier = Modifier.padding(16.dp)
                ) {
                    Text(
                        text = "How to Use",
                        style = MaterialTheme.typography.titleMedium
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    Text("1. Run the desktop indexer on your music library")
                    Text("2. Copy embeddings.db to your phone")
                    Text("3. Import the database using the button above")
                    Text("4. Play a song in Poweramp")
                    Text("5. Tap 'Start Radio' to find similar tracks")
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = "Tip: Add the Quick Settings tile for faster access!",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.primary
                    )
                }
            }
        }
    }
}

data class DatabaseInfo(
    val trackCount: Int,
    val version: String?,
    val sizeKb: Long
)

private fun loadDatabaseInfo(context: android.content.Context): DatabaseInfo? {
    val dbFile = File(context.filesDir, "embeddings.db")
    if (!dbFile.exists()) return null

    return try {
        val db = EmbeddingDatabase.open(dbFile)
        val info = DatabaseInfo(
            trackCount = db.getTrackCount(),
            version = db.getMetadata("version"),
            sizeKb = dbFile.length() / 1024
        )
        db.close()
        info
    } catch (e: Exception) {
        Log.e("MainActivity", "Failed to load database info", e)
        null
    }
}
