package com.powerampstartradio.indexing

/** Exact identity for a persisted Settings count; elapsed time is not a validity signal. */
data class V2UnindexedCountCacheIdentity(
    val databaseGeneration: String,
    val providerGeneration: String,
    val exclusionsFingerprint: String,
    val attentionFingerprint: String,
    val detectionPolicyId: String,
)

object V2UnindexedCountCachePolicy {
    const val DETECTION_POLICY_ID = "unindexed-readiness-v2-durable-attention-v1"

    fun canReuse(
        saved: V2UnindexedCountCacheIdentity?,
        current: V2UnindexedCountCacheIdentity,
    ): Boolean = saved != null &&
        current.databaseGeneration.isNotBlank() &&
        current.providerGeneration.isNotBlank() &&
        current.exclusionsFingerprint.isNotBlank() &&
        current.attentionFingerprint.isNotBlank() &&
        current.detectionPolicyId == DETECTION_POLICY_ID &&
        saved == current
}
