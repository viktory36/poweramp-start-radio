package com.powerampstartradio.tile

import android.service.quicksettings.Tile
import android.service.quicksettings.TileService
import android.util.Log
import com.powerampstartradio.poweramp.PowerampReceiver
import com.powerampstartradio.services.RadioService

/**
 * Quick Settings tile for starting radio with one tap.
 */
class StartRadioTile : TileService() {

    companion object {
        private const val TAG = "StartRadioTile"
    }

    override fun onStartListening() {
        super.onStartListening()
        updateTileState()
    }

    override fun onClick() {
        super.onClick()
        Log.d(TAG, "Tile clicked")

        // Start the radio service
        RadioService.startRadio(this)

        // Update tile to show it's working
        qsTile?.apply {
            state = Tile.STATE_ACTIVE
            subtitle = "Starting..."
            updateTile()
        }

        // Reset tile state after a short delay
        android.os.Handler(mainLooper).postDelayed({
            updateTileState()
        }, 3000)
    }

    private fun updateTileState() {
        qsTile?.apply {
            val currentTrack = PowerampReceiver.currentTrack
            if (currentTrack != null) {
                state = Tile.STATE_INACTIVE
                subtitle = currentTrack.title.take(20)
            } else {
                state = Tile.STATE_INACTIVE
                subtitle = "No track"
            }
            updateTile()
        }
    }

    override fun onTileAdded() {
        super.onTileAdded()
        Log.d(TAG, "Tile added")
    }

    override fun onTileRemoved() {
        super.onTileRemoved()
        Log.d(TAG, "Tile removed")
    }
}
