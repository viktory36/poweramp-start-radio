package com.powerampstartradio.indexing.v2

enum class V2IndexingPreflightFailureCode {
    EMPTY_SELECTION,
    DUPLICATE_POWERAMP_ROW,
    INVALID_SELECTION_EVIDENCE,
    INVALID_LOGICAL_SPAN,
    CUE_SOURCE_IMAGE,
    AUDIO_TOO_SHORT,
    SOURCE_UNREADABLE,
    NO_AUDIO_STREAM,
    UNSUPPORTED_OR_INVALID_AUDIO_CONTAINER,
    SOURCE_CHANGED,
    SOURCE_CANONICAL_ALIAS_COLLISION,
    PROVIDER_SNAPSHOT_INVALID,
    MODEL_UNREADABLE,
    APP_ARTIFACT_UNREADABLE,
    INSUFFICIENT_STORAGE,
    INVALID_PLAN,
    PERSISTENCE_FAILED,
}

enum class V2IndexingPreflightFailureScope {
    SELECTED_OCCURRENCE,
    GLOBAL_REQUEST,
}

data class V2IndexingPreflightFailureSemantics(
    val scope: V2IndexingPreflightFailureScope,
    val disposition: FailureDisposition?,
    val retryTrigger: RetryTrigger?,
) {
    init {
        val local = scope == V2IndexingPreflightFailureScope.SELECTED_OCCURRENCE
        require(local == (disposition != null && retryTrigger != null)) {
            "Only selected-occurrence failures have row retry semantics"
        }
    }
}

/** One exhaustive source of truth for whether preflight may continue after a failure. */
object V2IndexingPreflightFailurePolicy {
    fun semantics(
        code: V2IndexingPreflightFailureCode,
    ): V2IndexingPreflightFailureSemantics = when (code) {
        V2IndexingPreflightFailureCode.INVALID_LOGICAL_SPAN -> local(
            FailureDisposition.BLOCKED,
            RetryTrigger.SOURCE_OR_LIBRARY_CHANGED,
        )
        V2IndexingPreflightFailureCode.CUE_SOURCE_IMAGE -> local(
            FailureDisposition.BLOCKED,
            RetryTrigger.SOURCE_OR_LIBRARY_CHANGED,
        )
        V2IndexingPreflightFailureCode.AUDIO_TOO_SHORT -> local(
            FailureDisposition.BLOCKED,
            RetryTrigger.SOURCE_OR_LIBRARY_CHANGED,
        )
        V2IndexingPreflightFailureCode.SOURCE_UNREADABLE -> local(
            FailureDisposition.RETRYABLE,
            RetryTrigger.SOURCE_AVAILABLE,
        )
        V2IndexingPreflightFailureCode.NO_AUDIO_STREAM -> local(
            FailureDisposition.BLOCKED,
            RetryTrigger.SOURCE_OR_LIBRARY_CHANGED,
        )
        V2IndexingPreflightFailureCode.UNSUPPORTED_OR_INVALID_AUDIO_CONTAINER -> local(
            FailureDisposition.BLOCKED,
            // MediaExtractor does not distinguish a malformed file from an unsupported
            // container reliably. Keep automatic retries off instead of claiming one cause.
            RetryTrigger.USER_REQUEST,
        )

        V2IndexingPreflightFailureCode.EMPTY_SELECTION,
        V2IndexingPreflightFailureCode.DUPLICATE_POWERAMP_ROW,
        V2IndexingPreflightFailureCode.INVALID_SELECTION_EVIDENCE,
        V2IndexingPreflightFailureCode.SOURCE_CHANGED,
        V2IndexingPreflightFailureCode.SOURCE_CANONICAL_ALIAS_COLLISION,
        V2IndexingPreflightFailureCode.PROVIDER_SNAPSHOT_INVALID,
        V2IndexingPreflightFailureCode.MODEL_UNREADABLE,
        V2IndexingPreflightFailureCode.APP_ARTIFACT_UNREADABLE,
        V2IndexingPreflightFailureCode.INSUFFICIENT_STORAGE,
        V2IndexingPreflightFailureCode.INVALID_PLAN,
        V2IndexingPreflightFailureCode.PERSISTENCE_FAILED,
        -> global()
    }

    fun requireLocal(
        code: V2IndexingPreflightFailureCode,
    ): V2IndexingPreflightFailureSemantics = semantics(code).also { semantics ->
        require(semantics.scope == V2IndexingPreflightFailureScope.SELECTED_OCCURRENCE) {
            "$code is a global preflight failure"
        }
    }

    private fun local(
        disposition: FailureDisposition,
        retryTrigger: RetryTrigger,
    ) = V2IndexingPreflightFailureSemantics(
        V2IndexingPreflightFailureScope.SELECTED_OCCURRENCE,
        disposition,
        retryTrigger,
    )

    private fun global() = V2IndexingPreflightFailureSemantics(
        V2IndexingPreflightFailureScope.GLOBAL_REQUEST,
        null,
        null,
    )
}

class V2IndexingPreflightException(
    val code: V2IndexingPreflightFailureCode,
    val powerampFileId: Long? = null,
    message: String,
    cause: Throwable? = null,
) : IllegalStateException(message, cause)
