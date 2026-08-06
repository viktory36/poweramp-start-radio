package com.powerampstartradio.poweramp

import org.junit.Assert.assertFalse
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class PowerampBroadcastTrustTest {
    @Test
    fun android14RejectsAnObservableNonPowerampUid() {
        assertFalse(
            PowerampBroadcastTrust.isTrusted(
                sdkInt = 34,
                senderUid = 1234,
                uidPackages = setOf("example.attacker"),
                senderPackage = "example.attacker",
            ),
        )
        assertTrue(
            PowerampBroadcastTrust.isTrusted(
                sdkInt = 34,
                senderUid = -1,
                uidPackages = emptySet(),
                senderPackage = null,
            ),
        )
    }

    @Test
    fun android14TreatsUnavailableSenderIdentityOnlyAsARefreshHint() {
        assertEquals(
            PowerampBroadcastDisposition.REFRESH_HINT_ONLY,
            PowerampBroadcastTrust.classify(
                sdkInt = 34,
                senderUid = -1,
                uidPackages = emptySet(),
                senderPackage = null,
            ),
        )
    }

    @Test
    fun android14AcceptsOnlyAConsistentPowerampSender() {
        val packages = setOf(PowerampHelper.POWERAMP_PACKAGE)
        assertTrue(
            PowerampBroadcastTrust.isTrusted(34, 1234, packages, null),
        )
        assertTrue(
            PowerampBroadcastTrust.isTrusted(
                34,
                1234,
                packages,
                PowerampHelper.POWERAMP_PACKAGE,
            ),
        )
        assertFalse(
            PowerampBroadcastTrust.isTrusted(34, 1234, packages, "example.attacker"),
        )
    }

    @Test
    fun olderAndroidTreatsTheBroadcastOnlyAsARefreshHint() {
        assertTrue(
            PowerampBroadcastTrust.isTrusted(
                sdkInt = 33,
                senderUid = -1,
                uidPackages = emptySet(),
                senderPackage = null,
            ),
        )
        assertEquals(
            PowerampBroadcastDisposition.REFRESH_HINT_ONLY,
            PowerampBroadcastTrust.classify(
                sdkInt = 33,
                senderUid = 1234,
                uidPackages = setOf(PowerampHelper.POWERAMP_PACKAGE),
                senderPackage = PowerampHelper.POWERAMP_PACKAGE,
            ),
        )
    }

    @Test
    fun authenticatedAndSpoofedExplicitEventsHaveDifferentCommandAuthority() {
        assertEquals(
            PowerampBroadcastDisposition.AUTHENTICATED_EXPLICIT,
            PowerampBroadcastTrust.classify(
                sdkInt = 34,
                senderUid = 1234,
                uidPackages = setOf(PowerampHelper.POWERAMP_PACKAGE),
                senderPackage = PowerampHelper.POWERAMP_PACKAGE,
            ),
        )
        assertEquals(
            PowerampBroadcastDisposition.REJECT,
            PowerampBroadcastTrust.classify(
                sdkInt = 34,
                senderUid = 5678,
                uidPackages = setOf("example.attacker"),
                senderPackage = "example.attacker",
            ),
        )
    }
}
