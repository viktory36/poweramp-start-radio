package com.powerampstartradio.indexing.v2

/** Stamps temporal evidence only after the caller has completed artifact verification. */
internal object V2ArtifactVerificationCompletion {
    fun stamp(
        verify: () -> VerifiedArtifact,
        nowEpochMs: () -> Long,
    ): VerifiedArtifact {
        val verifiedArtifact = verify()
        val completedAtEpochMs = nowEpochMs()
        require(completedAtEpochMs >= 0L) { "verifiedAtEpochMs must not be negative" }
        return verifiedArtifact.copy(verifiedAtEpochMs = completedAtEpochMs)
    }
}
