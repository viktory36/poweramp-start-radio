package com.powerampstartradio.debug

import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.imePadding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.ui.Modifier
import androidx.lifecycle.ViewModelProvider
import com.powerampstartradio.SettingsScreen
import com.powerampstartradio.ui.DatabaseInfo
import com.powerampstartradio.ui.MainViewModel
import com.powerampstartradio.ui.SelectionMode
import com.powerampstartradio.ui.theme.PowerampStartRadioTheme

/** Debug-only host for observing production control admission without changing library data. */
class ControlAdmissionAcceptanceActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val requestedEligibleCandidateIdentityCount = intent.getIntExtra(
            EXTRA_ELIGIBLE_CANDIDATE_IDENTITY_COUNT,
            DEFAULT_ELIGIBLE_CANDIDATE_IDENTITY_COUNT,
        )
        val eligibleCandidateIdentityCount =
            requestedEligibleCandidateIdentityCount.takeIf { it >= 0 }
        val syntheticTrackCount = (eligibleCandidateIdentityCount ?: 0) + 1
        val viewModel = ViewModelProvider(this)[MainViewModel::class.java].apply {
            setNumTracks(50)
            setSelectionMode(SelectionMode.DPP)
            setDppUsesCertifiedFullDomain(false)
        }
        val databaseInfo = DatabaseInfo(
            trackCount = syntheticTrackCount,
            embeddingCount = syntheticTrackCount,
            embeddingDim = 768,
            version = "control-admission-acceptance",
            sizeKb = 0L,
            hasGraph = true,
            generationId = "control-admission-$eligibleCandidateIdentityCount",
            activeTrackCount = syntheticTrackCount,
            eligibleCandidateIdentityCount = eligibleCandidateIdentityCount,
            providerGenerationId = "control-admission-provider",
        )
        Log.i(
            TAG,
            "eligibleCandidateIdentityCount=${eligibleCandidateIdentityCount ?: "unknown"}",
        )

        enableEdgeToEdge()
        setContent {
            PowerampStartRadioTheme {
                Surface(
                    modifier = Modifier.fillMaxSize().imePadding(),
                    color = MaterialTheme.colorScheme.background,
                ) {
                    SettingsScreen(
                        viewModel = viewModel,
                        databaseInfo = databaseInfo,
                        onImportDatabase = {},
                        onMergeServerDatabase = {},
                        hasPermission = true,
                        onRequestPermission = {},
                        onBack = ::finish,
                    )
                }
            }
        }
    }

    companion object {
        const val EXTRA_ELIGIBLE_CANDIDATE_IDENTITY_COUNT =
            "eligible_candidate_identity_count"
        private const val DEFAULT_ELIGIBLE_CANDIDATE_IDENTITY_COUNT = 100
        private const val TAG = "ControlAdmissionAcceptance"
    }
}
