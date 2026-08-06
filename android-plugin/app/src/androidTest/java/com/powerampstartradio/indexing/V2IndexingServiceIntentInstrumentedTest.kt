package com.powerampstartradio.indexing

import android.Manifest
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.content.pm.ServiceInfo
import android.view.WindowManager
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.SdkSuppress
import com.powerampstartradio.MainActivity
import com.powerampstartradio.benchmark.BenchmarkActivity
import com.powerampstartradio.indexing.v2.RetryTrigger
import com.powerampstartradio.indexing.v2.IndexingJobState
import com.powerampstartradio.indexing.v2.V2ActiveIndexingJobConflictException
import com.powerampstartradio.indexing.v2.V2ActiveIndexingJobPointer
import com.powerampstartradio.indexing.v2.V2ActiveIndexingJobPointerInspection
import com.powerampstartradio.indexing.v2.V2ActiveIndexingJobPointerUnreadableException
import com.powerampstartradio.indexing.v2.V2ActiveIndexingJobPointerUnreadableReason
import com.powerampstartradio.services.RadioService
import java.io.File
import java.util.UUID
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class V2IndexingServiceIntentInstrumentedTest {
    private val context: Context = ApplicationProvider.getApplicationContext()

    @Test
    fun explicitImmutableIdentityCommandsRoundTripWithoutStartingService() {
        val command = V2IndexingServiceIntents.parse(
            context,
            V2IndexingServiceIntents.retryTrack(
                context,
                "job-123",
                "work-456",
                RetryTrigger.USER_REQUEST,
            ),
        )

        assertEquals(V2IndexingServiceCommandType.RETRY, command.type)
        assertEquals("job-123", command.jobId)
        assertEquals("work-456", command.workId)
        assertEquals(RetryTrigger.USER_REQUEST, command.retryTrigger)
    }

    @Test
    fun parserRejectsForeignComponentsAndUnexpectedPayloads() {
        val foreign = V2IndexingServiceIntents.pause(context, "job-123").apply {
            component = ComponentName(context.packageName, IndexingActivity::class.java.name)
        }
        assertThrows(IllegalArgumentException::class.java) {
            V2IndexingServiceIntents.parse(context, foreign)
        }

        val extraPayload: Intent = V2IndexingServiceIntents.pause(context, "job-123")
            .putExtra("track_payload", "must-not-cross-service-boundary")
        assertThrows(IllegalArgumentException::class.java) {
            V2IndexingServiceIntents.parse(context, extraPayload)
        }
    }

    @Test
    fun activeJobPointerClaimIsAtomicAcrossInstancesAndFailsClosed() {
        val root = File(context.cacheDir, "active-job-claim-${UUID.randomUUID()}")
        assertTrue(root.mkdirs())
        try {
            val first = V2ActiveIndexingJobPointer(root)
            val second = V2ActiveIndexingJobPointer(root)
            assertEquals(V2ActiveIndexingJobPointerInspection.Missing, first.inspect())
            assertEquals(null, first.read())
            assertTrue(first.claim("job-a") { null }.changed)
            assertTrue(!second.claim("job-a") { IndexingJobState.RUNNING }.changed)
            assertThrows(V2ActiveIndexingJobConflictException::class.java) {
                second.claim("job-b") { IndexingJobState.PAUSED }
            }
            assertEquals("job-a", first.read())
            assertTrue(second.claim("job-b") { IndexingJobState.COMPLETE }.changed)
            assertEquals("job-b", first.read())
            assertEquals(
                V2ActiveIndexingJobPointerInspection.Readable("job-b"),
                first.inspect(),
            )
            first.clear("job-a")
            assertEquals("job-b", second.read())
            second.write("job-c")
            assertEquals("job-c", first.read())
            first.clear("job-c")
            assertEquals(V2ActiveIndexingJobPointerInspection.Missing, second.inspect())
            second.write("job-d")
            first.clear()
            assertEquals(null, second.read())
        } finally {
            root.deleteRecursively()
        }
    }

    @Test
    fun unreadableActiveJobPointerBlocksEveryMutationAndPreservesExactBytes() {
        val cases = listOf(
            Triple(
                "empty",
                byteArrayOf(),
                V2ActiveIndexingJobPointerUnreadableReason.EMPTY,
            ),
            Triple(
                "invalid-job-id",
                "unsafe/job".toByteArray(),
                V2ActiveIndexingJobPointerUnreadableReason.INVALID_JOB_ID,
            ),
            Triple(
                "malformed-utf8",
                byteArrayOf(0xc3.toByte(), 0x28),
                V2ActiveIndexingJobPointerUnreadableReason.MALFORMED_UTF8,
            ),
        )
        cases.forEach { (label, original, expectedReason) ->
            val root = File(context.cacheDir, "active-job-unreadable-$label-${UUID.randomUUID()}")
            val rawFile = File(root, "indexing_v2/active-job-id")
            assertTrue(rawFile.parentFile!!.mkdirs())
            rawFile.writeBytes(original)
            try {
                val pointer = V2ActiveIndexingJobPointer(root)
                assertEquals(
                    V2ActiveIndexingJobPointerInspection.Unreadable(expectedReason),
                    pointer.inspect(),
                )
                val readError = assertThrows(
                    V2ActiveIndexingJobPointerUnreadableException::class.java,
                ) { pointer.read() }
                assertEquals(expectedReason, readError.reason)
                assertArrayEquals(original, rawFile.readBytes())

                assertThrows(V2ActiveIndexingJobPointerUnreadableException::class.java) {
                    pointer.claim("new-job") { IndexingJobState.COMPLETE }
                }
                assertArrayEquals(original, rawFile.readBytes())

                assertThrows(V2ActiveIndexingJobPointerUnreadableException::class.java) {
                    pointer.write("new-job")
                }
                assertArrayEquals(original, rawFile.readBytes())

                assertThrows(V2ActiveIndexingJobPointerUnreadableException::class.java) {
                    pointer.clear("new-job")
                }
                assertArrayEquals(original, rawFile.readBytes())

                assertThrows(V2ActiveIndexingJobPointerUnreadableException::class.java) {
                    pointer.clear()
                }
                assertArrayEquals(original, rawFile.readBytes())
            } finally {
                root.deleteRecursively()
            }
        }
    }

    @Test
    @SdkSuppress(minSdkVersion = 35)
    fun manifestClassifiesIndexerAsMediaProcessingWithoutLaunchingIt() {
        assertTrue(
            "V2 must target Android 16 while retaining the mediaProcessing timeout contract",
            context.applicationInfo.targetSdkVersion >= 36,
        )
        val serviceInfo = context.packageManager.getServiceInfo(
            ComponentName(context, IndexingService::class.java),
            PackageManager.ComponentInfoFlags.of(0L),
        )
        assertTrue(
            serviceInfo.foregroundServiceType and
                ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROCESSING != 0,
        )
        val permissions = context.packageManager.getPackageInfo(
            context.packageName,
            PackageManager.PackageInfoFlags.of(PackageManager.GET_PERMISSIONS.toLong()),
        ).requestedPermissions.orEmpty().toSet()
        assertTrue(Manifest.permission.FOREGROUND_SERVICE in permissions)
        assertTrue("android.permission.FOREGROUND_SERVICE_MEDIA_PROCESSING" in permissions)
        assertTrue("android.permission.FOREGROUND_SERVICE_SPECIAL_USE" in permissions)

        val radioInfo = context.packageManager.getServiceInfo(
            ComponentName(context, RadioService::class.java),
            PackageManager.ComponentInfoFlags.of(0L),
        )
        assertTrue(
            radioInfo.foregroundServiceType and
                ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE != 0,
        )

        val bootReceiver = context.packageManager.getReceiverInfo(
            ComponentName(context, V2IndexingBootReceiver::class.java),
            PackageManager.ComponentInfoFlags.of(0L),
        )
        assertTrue(Manifest.permission.RECEIVE_BOOT_COMPLETED in permissions)
        assertFalse("boot recovery receiver must not be app-exported", bootReceiver.exported)
    }

    @Test
    @SdkSuppress(minSdkVersion = 35)
    @Suppress("DEPRECATION")
    fun target35ActivitiesDeclareResizeAndDebugSurfaceStaysShellProtected() {
        fun assertAdjustResize(component: ComponentName) {
            val info = context.packageManager.getActivityInfo(
                component,
                PackageManager.ComponentInfoFlags.of(0L),
            )
            assertEquals(
                WindowManager.LayoutParams.SOFT_INPUT_ADJUST_RESIZE,
                info.softInputMode and WindowManager.LayoutParams.SOFT_INPUT_MASK_ADJUST,
            )
        }

        assertAdjustResize(ComponentName(context, MainActivity::class.java))
        assertAdjustResize(ComponentName(context, IndexingActivity::class.java))
        val benchmark = context.packageManager.getActivityInfo(
            ComponentName(context, BenchmarkActivity::class.java),
            PackageManager.ComponentInfoFlags.of(0L),
        )
        assertEquals(
            WindowManager.LayoutParams.SOFT_INPUT_ADJUST_RESIZE,
            benchmark.softInputMode and WindowManager.LayoutParams.SOFT_INPUT_MASK_ADJUST,
        )
        assertEquals(Manifest.permission.DUMP, benchmark.permission)
    }
}
