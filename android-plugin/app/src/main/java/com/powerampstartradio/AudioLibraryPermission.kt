package com.powerampstartradio

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.os.Build
import androidx.core.content.ContextCompat

internal object AudioLibraryPermission {
    const val DENIED_MESSAGE =
        "Music and audio permission is required to merge a server index. " +
            "Grant it in Android Settings, then try again."

    fun permissionName(sdkInt: Int = Build.VERSION.SDK_INT): String =
        if (sdkInt >= Build.VERSION_CODES.TIRAMISU) {
            Manifest.permission.READ_MEDIA_AUDIO
        } else {
            Manifest.permission.READ_EXTERNAL_STORAGE
        }

    fun isGranted(context: Context): Boolean =
        ContextCompat.checkSelfPermission(context, permissionName()) ==
            PackageManager.PERMISSION_GRANTED

    fun shouldPrompt(freshActivityStart: Boolean, granted: Boolean): Boolean =
        freshActivityStart && !granted
}
