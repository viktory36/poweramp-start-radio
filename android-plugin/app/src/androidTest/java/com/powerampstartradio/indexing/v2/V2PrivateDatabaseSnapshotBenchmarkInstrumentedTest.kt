package com.powerampstartradio.indexing.v2

import android.database.sqlite.SQLiteDatabase
import android.os.SystemClock
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.RandomAccessFile
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/** Read-only source benchmark against the phone's real active database. */
@RunWith(AndroidJUnit4::class)
class V2PrivateDatabaseSnapshotBenchmarkInstrumentedTest {
    @Test
    fun immutableByteCopyIsMeasuredAgainstRedundantVacuum() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val source = V2LibraryDatabaseResolver.requirePublished(context.filesDir).databaseFile
        val rawTarget = File(context.cacheDir, "private-index-raw-copy-benchmark.db")
        val vacuumTarget = File(context.cacheDir, "private-index-vacuum-benchmark.db")
        rawTarget.delete()
        vacuumTarget.delete()
        try {
            val rawStarted = SystemClock.elapsedRealtime()
            FileInputStream(source).channel.use { input ->
                FileOutputStream(rawTarget).channel.use { output ->
                    var offset = 0L
                    while (offset < input.size()) {
                        val copied = input.transferTo(offset, input.size() - offset, output)
                        check(copied > 0L)
                        offset += copied
                    }
                    output.force(true)
                }
            }
            val rawMs = SystemClock.elapsedRealtime() - rawStarted
            assertEquals(source.length(), rawTarget.length())
            assertEquals(V2FileSha256.digest(source), V2FileSha256.digest(rawTarget))

            val vacuumStarted = SystemClock.elapsedRealtime()
            SQLiteDatabase.openDatabase(
                source.path,
                null,
                SQLiteDatabase.OPEN_READWRITE,
            ).use { database ->
                val escaped = vacuumTarget.canonicalPath.replace("'", "''")
                database.execSQL("VACUUM INTO '$escaped'")
            }
            RandomAccessFile(vacuumTarget, "rw").use { it.fd.sync() }
            val vacuumMs = SystemClock.elapsedRealtime() - vacuumStarted
            assertTrue(vacuumTarget.length() > 0L)
            SQLiteDatabase.openDatabase(
                rawTarget.path,
                null,
                SQLiteDatabase.OPEN_READONLY,
            ).use { database ->
                database.rawQuery("PRAGMA quick_check(1)", null).use { cursor ->
                    assertTrue(cursor.moveToFirst())
                    assertEquals("ok", cursor.getString(0))
                }
            }
            Log.i(
                TAG,
                "sourceBytes=${source.length()} rawCopyMs=$rawMs vacuumIntoMs=$vacuumMs " +
                    "rawBytes=${rawTarget.length()} vacuumBytes=${vacuumTarget.length()}",
            )
        } finally {
            rawTarget.delete()
            vacuumTarget.delete()
        }
    }

    private companion object {
        const val TAG = "PrivateDbSnapshotProof"
    }
}
