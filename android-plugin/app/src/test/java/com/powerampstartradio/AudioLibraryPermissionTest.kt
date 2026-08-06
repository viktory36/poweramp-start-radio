package com.powerampstartradio

import android.Manifest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class AudioLibraryPermissionTest {
    @Test
    fun usesLegacyStoragePermissionBeforeAndroid13() {
        assertEquals(
            Manifest.permission.READ_EXTERNAL_STORAGE,
            AudioLibraryPermission.permissionName(sdkInt = 32),
        )
    }

    @Test
    fun usesAudioPermissionFromAndroid13() {
        assertEquals(
            Manifest.permission.READ_MEDIA_AUDIO,
            AudioLibraryPermission.permissionName(sdkInt = 33),
        )
        assertEquals(
            Manifest.permission.READ_MEDIA_AUDIO,
            AudioLibraryPermission.permissionName(sdkInt = 36),
        )
    }

    @Test
    fun promptsOnlyForAFirstActivityStartWithoutAccess() {
        assertTrue(
            AudioLibraryPermission.shouldPrompt(
                freshActivityStart = true,
                granted = false,
            ),
        )
        assertFalse(
            AudioLibraryPermission.shouldPrompt(
                freshActivityStart = false,
                granted = false,
            ),
        )
        assertFalse(
            AudioLibraryPermission.shouldPrompt(
                freshActivityStart = true,
                granted = true,
            ),
        )
    }
}
