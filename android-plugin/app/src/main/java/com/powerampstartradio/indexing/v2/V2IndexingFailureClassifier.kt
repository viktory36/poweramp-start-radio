package com.powerampstartradio.indexing.v2

import android.database.sqlite.SQLiteDatabaseLockedException
import android.database.sqlite.SQLiteDiskIOException
import android.database.sqlite.SQLiteFullException
import android.database.sqlite.SQLiteException
import android.system.ErrnoException
import android.system.OsConstants
import com.powerampstartradio.indexing.AudioDecoder
import com.powerampstartradio.indexing.TrackPcmCache
import java.io.FileNotFoundException
import java.io.IOException
import java.util.concurrent.CancellationException

data class V2ClassifiedIndexingFailure(
    val code: TrackFailureCode,
    val diagnostic: String,
)

class V2IndexingControlFlowException(message: String) : CancellationException(message)

object V2ExecutorFailureBoundary {
    fun rethrowControlFlow(error: Throwable) {
        val control = generateSequence(error) { it.cause }.firstOrNull { cause ->
            cause is V2IndexingControlFlowException ||
                cause is CancellationException ||
                cause is AudioDecoder.AudioDecodeCancelledException
        } ?: return
        if (control is V2IndexingControlFlowException) throw control
        throw V2IndexingControlFlowException(
            control.message ?: "indexing execution stopped by lifecycle control",
        )
    }

    fun classifyOrRethrow(
        error: Throwable,
        stage: IndexingStage,
        span: FinalizedAudioSpanEvidence,
    ): V2ClassifiedIndexingFailure {
        rethrowControlFlow(error)
        return V2IndexingFailureClassifier.classify(error, stage, span)
    }
}

/** Maps concrete executor failures into the exhaustive durable retry policy. */
object V2IndexingFailureClassifier {
    fun classify(
        error: Throwable,
        stage: IndexingStage,
        span: FinalizedAudioSpanEvidence,
    ): V2ClassifiedIndexingFailure {
        V2ExecutorFailureBoundary.rethrowControlFlow(error)
        val chain = generateSequence(error) { current -> current.cause }.toList()

        fun has(type: Class<out Throwable>): Boolean = chain.any(type::isInstance)
        fun <T : Throwable> first(type: Class<T>): T? =
            chain.firstOrNull(type::isInstance)?.let(type::cast)

        val boundary = first(AudioDecoder.AudioDecodeBoundaryException::class.java)
        val pcmContract = first(TrackPcmCache.PcmContractException::class.java)
        val provider = first(V2PowerampProviderSnapshotException::class.java)
        val staging = first(V2StagingDatabaseException::class.java)
        val errno = first(ErrnoException::class.java)
        val noSpace = has(SQLiteFullException::class.java) ||
            errno?.errno == OsConstants.ENOSPC ||
            chain.any { it.message.orEmpty().contains("no space", ignoreCase = true) }
        val code = when {
            has(V2ImportedRowAuthorizationException::class.java) ->
                TrackFailureCode.IMPORTED_ROW_AUTHORIZATION_CHANGED
            has(V2ProviderSnapshotChangedException::class.java) ->
                TrackFailureCode.PROVIDER_SNAPSHOT_CHANGED
            has(V2SourceIdentityChangedException::class.java) ->
                TrackFailureCode.SOURCE_FINGERPRINT_CHANGED
            provider?.code == V2PowerampProviderSnapshotFailureCode.POWERAMP_PERMISSION_DENIED ->
                TrackFailureCode.POWERAMP_PERMISSION_DENIED
            provider?.code in setOf(
                V2PowerampProviderSnapshotFailureCode.PROVIDER_QUERY_FAILED,
                V2PowerampProviderSnapshotFailureCode.PROVIDER_RETURNED_NULL_CURSOR,
                V2PowerampProviderSnapshotFailureCode.PROVIDER_CURSOR_FAILED,
            ) -> TrackFailureCode.POWERAMP_PROVIDER_UNAVAILABLE
            provider != null -> TrackFailureCode.PROVIDER_SNAPSHOT_CHANGED
            staging?.reason == V2StagingDatabaseFailure.BASE_GENERATION_CHANGED ->
                TrackFailureCode.DATABASE_GENERATION_CHANGED
            staging?.reason == V2StagingDatabaseFailure.SOURCE_MISSING ->
                TrackFailureCode.SOURCE_MISSING
            staging != null -> TrackFailureCode.COMMIT_FAILED
            has(SecurityException::class.java) ->
                TrackFailureCode.ANDROID_AUDIO_PERMISSION_DENIED
            has(OutOfMemoryError::class.java) -> TrackFailureCode.OUT_OF_MEMORY
            has(FileNotFoundException::class.java) -> TrackFailureCode.SOURCE_MISSING
            noSpace -> TrackFailureCode.STORAGE_FULL
            has(SQLiteDatabaseLockedException::class.java) -> TrackFailureCode.DATABASE_BUSY
            has(SQLiteDiskIOException::class.java) -> TrackFailureCode.COMMIT_FAILED
            has(V2InvalidModelOutputException::class.java) -> TrackFailureCode.INVALID_MODEL_OUTPUT
            has(V2ModelLoadException::class.java) -> TrackFailureCode.MODEL_LOAD_FAILED
            has(V2InferenceException::class.java) -> TrackFailureCode.INFERENCE_FAILED
            has(V2ArtifactChecksumException::class.java) ->
                TrackFailureCode.ARTIFACT_CHECKSUM_MISMATCH
            has(V2ArtifactIntegrityException::class.java) -> TrackFailureCode.PARTIAL_ARTIFACT
            has(AudioDecoder.NoAudioStreamException::class.java) ->
                TrackFailureCode.NO_AUDIO_STREAM
            has(AudioDecoder.UnsupportedPcmFormatException::class.java) ->
                TrackFailureCode.UNSUPPORTED_CODEC_OR_CONTAINER
            pcmContract?.reason == TrackPcmCache.PcmContractFailure.EOS_MISMATCH ->
                TrackFailureCode.CONTAINER_EOS_MISMATCH
            pcmContract?.reason == TrackPcmCache.PcmContractFailure.PREPROCESSING_MISMATCH ->
                TrackFailureCode.INVALID_MODEL_OUTPUT
            pcmContract?.reason == TrackPcmCache.PcmContractFailure.PCM_ARTIFACT_MISMATCH ->
                TrackFailureCode.PARTIAL_ARTIFACT
            pcmContract != null -> TrackFailureCode.INVALID_LOGICAL_SPAN
            boundary != null && span.executionBoundaryRequirement ==
                V2ExecutionBoundaryRequirement.VERIFY_END_OF_STREAM_AND_RECONCILE ->
                TrackFailureCode.CONTAINER_EOS_MISMATCH
            boundary != null -> TrackFailureCode.INVALID_LOGICAL_SPAN
            has(AudioDecoder.AudioDecodeTimeoutException::class.java) ->
                TrackFailureCode.STAGE_TIMEOUT
            has(AudioDecoder.AudioDecodeException::class.java) -> TrackFailureCode.DECODER_ERROR
            has(SQLiteException::class.java) -> TrackFailureCode.COMMIT_FAILED
            has(IOException::class.java) && stage == IndexingStage.DATABASE_COMMIT ->
                TrackFailureCode.COMMIT_FAILED
            has(IOException::class.java) -> TrackFailureCode.SOURCE_UNREADABLE
            else -> TrackFailureCode.UNKNOWN_TRANSIENT
        }
        val diagnosticSource = chain.firstOrNull { !it.message.isNullOrBlank() } ?: error
        val message = diagnosticSource.message?.trim().orEmpty()
            .ifBlank { diagnosticSource::class.java.simpleName }
        return V2ClassifiedIndexingFailure(code, message.take(2_048))
    }
}

class V2ProviderSnapshotChangedException(message: String) : IllegalStateException(message)
class V2SourceIdentityChangedException(message: String) : IllegalStateException(message)
class V2ModelLoadException(message: String, cause: Throwable? = null) :
    IllegalStateException(message, cause)
class V2InferenceException(message: String, cause: Throwable? = null) :
    IllegalStateException(message, cause)
class V2InvalidModelOutputException(message: String) : IllegalStateException(message)
open class V2ArtifactIntegrityException(message: String, cause: Throwable? = null) :
    IllegalStateException(message, cause)
class V2ArtifactChecksumException(message: String, cause: Throwable? = null) :
    V2ArtifactIntegrityException(message, cause)
