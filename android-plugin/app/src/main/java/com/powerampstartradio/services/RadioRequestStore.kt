package com.powerampstartradio.services

import android.util.AtomicFile
import com.google.gson.Gson
import com.powerampstartradio.data.EmbeddedTrack
import com.powerampstartradio.poweramp.PowerampTrack
import com.powerampstartradio.poweramp.TrackMatcher
import com.powerampstartradio.ui.ComposedRadioContract
import com.powerampstartradio.ui.DirectQueuePlacement
import com.powerampstartradio.ui.FindMusicOperator
import com.powerampstartradio.ui.FindMusicQuerySpec
import com.powerampstartradio.ui.FindMusicSessionEvidence
import com.powerampstartradio.ui.FindMusicTextResultPlanner
import com.powerampstartradio.ui.FindMusicTrackEvidence
import com.powerampstartradio.ui.QueueOrigin
import com.powerampstartradio.ui.RadioConfig
import com.powerampstartradio.ui.RadioGenerationToken
import com.powerampstartradio.ui.RadioResult
import com.powerampstartradio.ui.RadioSeedIdentity
import com.powerampstartradio.ui.RadioSessionOutcome
import com.powerampstartradio.ui.SeedSpec
import com.powerampstartradio.ui.SeedType
import com.powerampstartradio.ui.MAX_LIBRARY_ADDED_DAYS
import com.powerampstartradio.ui.effectiveLibraryAddedDays
import com.powerampstartradio.ui.validateFindMusicQueryContract
import com.powerampstartradio.similarity.StableVisibleResultReducer
import java.io.File
import java.io.FileNotFoundException
import java.security.MessageDigest
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap
import kotlin.math.abs

/** The complete, immutable input to one foreground radio-service operation. */
internal data class DurableRadioRequest(
    val schemaVersion: Int = CURRENT_SCHEMA_VERSION,
    val requestId: String,
    val createdAtEpochMs: Long,
    val generation: RadioGenerationToken,
    val providerGenerationId: String,
    val kind: DurableRadioRequestKind,
    val radio: RadioRequestPayload? = null,
    val multiSeed: MultiSeedRequestPayload? = null,
    val directQueue: DirectQueueRequestPayload? = null,
) {
    companion object {
        const val CURRENT_SCHEMA_VERSION = 5

        fun radio(
            generation: RadioGenerationToken,
            providerGenerationId: String,
            config: RadioConfig,
            seed: PinnedRadioSeed,
            showToasts: Boolean,
            origin: QueueOrigin,
            requestId: String = UUID.randomUUID().toString(),
            createdAtEpochMs: Long = System.currentTimeMillis(),
        ) = DurableRadioRequest(
            requestId = requestId,
            createdAtEpochMs = createdAtEpochMs,
            generation = generation,
            providerGenerationId = providerGenerationId,
            kind = DurableRadioRequestKind.RADIO,
            radio = RadioRequestPayload(config, seed, showToasts, origin),
        )

        fun multiSeed(
            generation: RadioGenerationToken,
            providerGenerationId: String,
            seeds: List<SeedSpec>,
            seedIdentities: List<RadioSeedIdentity?>,
            querySpec: FindMusicQuerySpec,
            config: RadioConfig,
            composedContract: ComposedRadioContract,
            showToasts: Boolean,
            origin: QueueOrigin,
            requestId: String = UUID.randomUUID().toString(),
            createdAtEpochMs: Long = System.currentTimeMillis(),
        ) = DurableRadioRequest(
            requestId = requestId,
            createdAtEpochMs = createdAtEpochMs,
            generation = generation,
            providerGenerationId = providerGenerationId,
            kind = DurableRadioRequestKind.MULTI_SEED_RADIO,
            multiSeed = MultiSeedRequestPayload(
                seeds.map { seed -> seed.copy(embedding = seed.embedding.copyOf()) },
                seedIdentities.toList(),
                querySpec.copy(songSeeds = querySpec.songSeeds.toList()),
                config,
                composedContract,
                showToasts,
                origin,
            ),
        )

        fun directQueue(
            generation: RadioGenerationToken,
            providerGenerationId: String,
            tracks: List<EmbeddedTrack>,
            trackIdentities: List<RadioSeedIdentity>,
            resolvedPowerampFileIds: List<Long?>,
            label: String,
            origin: QueueOrigin,
            placement: DirectQueuePlacement,
            findMusicSessionEvidence: FindMusicSessionEvidence? = null,
            findMusicTrackEvidence: List<FindMusicTrackEvidence>? = null,
            requestId: String = UUID.randomUUID().toString(),
            createdAtEpochMs: Long = System.currentTimeMillis(),
        ) = DurableRadioRequest(
            requestId = requestId,
            createdAtEpochMs = createdAtEpochMs,
            generation = generation,
            providerGenerationId = providerGenerationId,
            kind = DurableRadioRequestKind.DIRECT_QUEUE,
            directQueue = DirectQueueRequestPayload(
                tracks.toList(),
                trackIdentities.toList(),
                resolvedPowerampFileIds.toList(),
                label,
                origin,
                placement,
                findMusicSessionEvidence,
                findMusicTrackEvidence?.toList(),
            ),
        )
    }
}

internal enum class DurableRadioRequestKind {
    RADIO,
    MULTI_SEED_RADIO,
    DIRECT_QUEUE,
}

internal data class RadioRequestPayload(
    val config: RadioConfig,
    val seed: PinnedRadioSeed,
    val showToasts: Boolean,
    val origin: QueueOrigin,
)

internal data class PinnedRadioSeed(
    val identity: RadioSeedIdentity,
    val displayTrack: PowerampTrack,
    val matchType: TrackMatcher.MatchType,
)

internal data class MultiSeedRequestPayload(
    val seeds: List<SeedSpec>,
    val seedIdentities: List<RadioSeedIdentity?>,
    val querySpec: FindMusicQuerySpec,
    val config: RadioConfig,
    val composedContract: ComposedRadioContract,
    val showToasts: Boolean,
    val origin: QueueOrigin,
)

internal data class DirectQueueRequestPayload(
    val tracks: List<EmbeddedTrack>,
    val trackIdentities: List<RadioSeedIdentity>,
    val resolvedPowerampFileIds: List<Long?>,
    val label: String,
    val origin: QueueOrigin,
    val placement: DirectQueuePlacement,
    val findMusicSessionEvidence: FindMusicSessionEvidence? = null,
    val findMusicTrackEvidence: List<FindMusicTrackEvidence>? = null,
)

internal data class DurableRadioResultReceipt(
    val schemaVersion: Int = CURRENT_SCHEMA_VERSION,
    val requestId: String,
    val payloadSha256: String,
    val result: RadioResult,
    val terminalDetail: String?,
    val createdAtEpochMs: Long,
) {
    companion object {
        const val CURRENT_SCHEMA_VERSION = 1
    }
}

internal enum class RadioRequestStateKind {
    PERSISTED,
    CLAIMED,
    COMPLETED,
    FAILED,
    INTERRUPTED_NEEDS_RETRY,
}

internal data class RadioRequestState(
    val schemaVersion: Int = CURRENT_SCHEMA_VERSION,
    val requestId: String,
    val payloadSha256: String,
    val state: RadioRequestStateKind,
    val ownerToken: String? = null,
    val terminalDetail: String? = null,
    val updatedAtEpochMs: Long,
) {
    companion object {
        const val CURRENT_SCHEMA_VERSION = 1
    }
}

internal sealed interface RadioRequestClaim {
    data class Claimed(val request: DurableRadioRequest) : RadioRequestClaim
    data object AlreadyInFlight : RadioRequestClaim
    data class AlreadyTerminal(val state: RadioRequestStateKind) : RadioRequestClaim
    data class ResultReady(val receipt: DurableRadioResultReceipt) : RadioRequestClaim
    data object Missing : RadioRequestClaim
}

internal fun interface RadioRequestAtomicWriter {
    fun write(file: File, bytes: ByteArray)
}

internal fun interface RadioRequestAtomicReader {
    fun read(file: File): ByteArray?
}

internal fun interface RadioRequestAtomicDeleter {
    fun delete(file: File): Boolean
}

internal object AndroidRadioRequestAtomicWriter : RadioRequestAtomicWriter {
    override fun write(file: File, bytes: ByteArray) {
        file.parentFile?.mkdirs()
        val atomic = AtomicFile(file)
        val output = atomic.startWrite()
        try {
            output.write(bytes)
            atomic.finishWrite(output)
        } catch (failure: Throwable) {
            atomic.failWrite(output)
            throw failure
        }
    }
}

internal object AndroidRadioRequestAtomicReader : RadioRequestAtomicReader {
    override fun read(file: File): ByteArray? {
        val atomic = AtomicFile(file)
        return try {
            atomic.openRead().use { input -> input.readBytes() }
        } catch (_: FileNotFoundException) {
            null
        }
    }
}

internal object AndroidRadioRequestAtomicDeleter : RadioRequestAtomicDeleter {
    override fun delete(file: File): Boolean {
        AtomicFile(file).delete()
        return !file.exists() && !File("${file.path}.bak").exists()
    }
}

/**
 * Small durable journal for RadioService inputs.
 *
 * Payload files never change after publication. Claim/consumption state lives in a separate
 * atomic file, so an interrupted state transition cannot corrupt the request itself. A claim
 * owned by another process token is recoverable after process death; a duplicate delivery in
 * the same process is ignored.
 */
internal class RadioRequestStore(
    rootDir: File,
    private val ownerToken: String = UUID.randomUUID().toString(),
    private val clock: () -> Long = System::currentTimeMillis,
    private val atomicWriter: RadioRequestAtomicWriter = AndroidRadioRequestAtomicWriter,
    private val atomicReader: RadioRequestAtomicReader = AndroidRadioRequestAtomicReader,
    private val atomicDeleter: RadioRequestAtomicDeleter = AndroidRadioRequestAtomicDeleter,
    private val gson: Gson = Gson(),
) {
    private val root = File(rootDir, DIRECTORY_NAME)
    private val payloadDir = File(root, PAYLOAD_DIRECTORY_NAME)
    private val stateDir = File(root, STATE_DIRECTORY_NAME)
    private val receiptDir = File(root, RECEIPT_DIRECTORY_NAME)
    private val lock = locks.computeIfAbsent(root.absoluteFile.normalize().path) { Any() }

    fun persist(request: DurableRadioRequest): String = persistInternal(request, idempotent = false)

    /** Re-publication of the exact same immutable command is a no-op; payload drift still fails. */
    fun persistIdempotently(request: DurableRadioRequest): String =
        persistInternal(request, idempotent = true)

    private fun persistInternal(
        request: DurableRadioRequest,
        idempotent: Boolean,
    ): String = synchronized(lock) {
        validate(request)
        val bytes = encode(request)
        require(bytes.size <= MAX_PAYLOAD_BYTES) {
            "Radio request is ${bytes.size} bytes; maximum is $MAX_PAYLOAD_BYTES"
        }
        ensureDirectories()
        cleanupTerminalStatesLocked()

        val payloadFile = payloadFile(request.requestId)
        val stateFile = stateFile(request.requestId)
        val receiptFile = receiptFile(request.requestId)
        val existingPayload = atomicReader.read(payloadFile)
        if (idempotent && existingPayload != null) {
            require(existingPayload.contentEquals(bytes)) {
                "Radio request ID belongs to a different immutable payload: ${request.requestId}"
            }
            return@synchronized request.requestId
        }
        require(
            existingPayload == null &&
                atomicReader.read(stateFile) == null &&
                atomicReader.read(receiptFile) == null,
        ) {
            "Radio request already exists: ${request.requestId}"
        }
        val outstanding = payloadDir.listFiles { file -> file.extension == PAYLOAD_EXTENSION }
            .orEmpty()
            .count { file ->
                runCatching { readStateLocked(file.nameWithoutExtension)?.state !in TERMINAL_STATES }
                    .getOrDefault(true)
            }
        require(outstanding < MAX_OUTSTANDING_REQUESTS) {
            "Too many outstanding radio requests ($outstanding)"
        }

        // The payload is the commit point. A missing state file is recovered as PERSISTED.
        atomicWriter.write(payloadFile, bytes)
        val sha256 = sha256(bytes)
        // State is reconstructible from the immutable payload. Once payload publication succeeds,
        // return its ID even if the optional initial PERSISTED write is interrupted.
        runCatching {
            atomicWriter.write(
                stateFile,
                encodeState(
                    RadioRequestState(
                        requestId = request.requestId,
                        payloadSha256 = sha256,
                        state = RadioRequestStateKind.PERSISTED,
                        updatedAtEpochMs = clock(),
                    ),
                ),
            )
        }
        request.requestId
    }

    fun hasRecord(requestId: String): Boolean = synchronized(lock) {
        requireValidRequestId(requestId)
        atomicReader.read(payloadFile(requestId)) != null ||
            atomicReader.read(stateFile(requestId)) != null ||
            atomicReader.read(receiptFile(requestId)) != null
    }

    fun readRequest(requestId: String): DurableRadioRequest? = synchronized(lock) {
        requireValidRequestId(requestId)
        atomicReader.read(payloadFile(requestId))?.let(::decode)
    }

    fun readStateKind(requestId: String): RadioRequestStateKind? = synchronized(lock) {
        requireValidRequestId(requestId)
        readStateLocked(requestId)?.state
    }

    fun claim(requestId: String): RadioRequestClaim = synchronized(lock) {
        requireValidRequestId(requestId)
        ensureDirectories()
        val payloadBytes = readPayloadLocked(requestId)
        val payloadDigest = payloadBytes?.let(::sha256)
        val existingState = try {
            readStateLocked(requestId)
        } catch (failure: Exception) {
            val receipt = payloadBytes?.let { readReceiptLocked(requestId, sha256(it)) }
            if (receipt != null) return@synchronized RadioRequestClaim.ResultReady(receipt)
            if (payloadBytes == null) throw failure
            atomicWriter.write(
                stateFile(requestId),
                encodeState(
                    RadioRequestState(
                        requestId = requestId,
                        payloadSha256 = checkNotNull(payloadDigest),
                        state = RadioRequestStateKind.INTERRUPTED_NEEDS_RETRY,
                        terminalDetail = "Lifecycle state was unreadable; automatic replay is unsafe",
                        updatedAtEpochMs = clock(),
                    ),
                ),
            )
            return@synchronized RadioRequestClaim.AlreadyTerminal(
                RadioRequestStateKind.INTERRUPTED_NEEDS_RETRY,
            )
        }

        val receipt = payloadDigest?.let { readReceiptLocked(requestId, it) }
        when {
            existingState?.state in TERMINAL_STATES -> {
                return@synchronized RadioRequestClaim.AlreadyTerminal(existingState!!.state)
            }
            existingState?.state == RadioRequestStateKind.CLAIMED &&
                existingState.ownerToken == ownerToken -> {
                return@synchronized RadioRequestClaim.AlreadyInFlight
            }
            receipt != null -> {
                return@synchronized RadioRequestClaim.ResultReady(receipt)
            }
            existingState?.state == RadioRequestStateKind.CLAIMED -> {
                // There is no transaction spanning our state file and Poweramp's queue provider.
                // A dead process may have crossed the external mutation boundary, so replaying
                // automatically could replace or append twice. Retain the payload and fail closed;
                // an explicit new user request is the only safe retry without a delivery receipt.
                atomicWriter.write(
                    stateFile(requestId),
                    encodeState(
                        existingState.copy(
                            state = RadioRequestStateKind.INTERRUPTED_NEEDS_RETRY,
                            ownerToken = null,
                            terminalDetail = "Process ended after claim; automatic replay is unsafe",
                            updatedAtEpochMs = clock(),
                        ),
                    ),
                )
                return@synchronized RadioRequestClaim.AlreadyTerminal(
                    RadioRequestStateKind.INTERRUPTED_NEEDS_RETRY,
                )
            }
        }

        val bytes = payloadBytes ?: return@synchronized RadioRequestClaim.Missing
        val digest = checkNotNull(payloadDigest)
        existingState?.let { state ->
            require(state.requestId == requestId) { "Radio request state ID mismatch" }
            require(state.payloadSha256 == digest) { "Radio request payload digest mismatch" }
        }
        // Claim before decoding. If a validly named/digested file contains malformed JSON, the
        // caller can terminally fail this owned request rather than entering a restart loop.
        atomicWriter.write(
            stateFile(requestId),
            encodeState(
                RadioRequestState(
                    requestId = requestId,
                    payloadSha256 = digest,
                    state = RadioRequestStateKind.CLAIMED,
                    ownerToken = ownerToken,
                    updatedAtEpochMs = clock(),
                ),
            ),
        )
        val request = decode(bytes)
        require(request.requestId == requestId) { "Radio request payload ID mismatch" }
        RadioRequestClaim.Claimed(request)
    }

    /** Commit an authoritative terminal result before session-history persistence. */
    fun persistResultReceipt(
        requestId: String,
        result: RadioResult,
        terminalDetail: String? = null,
    ): DurableRadioResultReceipt = synchronized(lock) {
        requireValidRequestId(requestId)
        val payloadBytes = requireNotNull(readPayloadLocked(requestId)) {
            "Radio request payload is missing"
        }
        val payloadDigest = sha256(payloadBytes)
        val state = requireNotNull(readStateLocked(requestId)) {
            "Radio request state is missing"
        }
        require(state.state == RadioRequestStateKind.CLAIMED && state.ownerToken == ownerToken) {
            "Cannot receipt an unowned radio request: $requestId"
        }
        require(state.payloadSha256 == payloadDigest) { "Radio request payload digest mismatch" }
        val request = decode(payloadBytes)
        validateTerminalResult(request, result, terminalDetail)
        readReceiptLocked(requestId, payloadDigest)?.let { existing ->
            require(existing.result == result && existing.terminalDetail == terminalDetail) {
                "Radio result receipt changed for $requestId"
            }
            return@synchronized existing
        }
        val receipt = DurableRadioResultReceipt(
            requestId = requestId,
            payloadSha256 = payloadDigest,
            result = result,
            terminalDetail = terminalDetail?.take(MAX_TERMINAL_DETAIL_CHARS),
            createdAtEpochMs = clock(),
        )
        val bytes = encodeReceipt(receipt)
        atomicWriter.write(receiptFile(requestId), bytes)
        receipt
    }

    /** Finish a receipt recovered from a previous service owner without replaying Poweramp. */
    fun finalizeRecoveredResult(receipt: DurableRadioResultReceipt) = synchronized(lock) {
        val requestId = receipt.requestId
        requireValidRequestId(requestId)
        val payloadBytes = requireNotNull(readPayloadLocked(requestId)) {
            "Recovered radio request payload is missing"
        }
        val persistedReceipt = requireNotNull(readReceiptLocked(requestId, sha256(payloadBytes))) {
            "Recovered radio result receipt is missing"
        }
        require(persistedReceipt == receipt) { "Recovered radio result receipt changed" }
        val state = readStateLocked(requestId)
        require(state == null || state.state !in TERMINAL_STATES) {
            "Recovered radio request is already terminal"
        }
        require(state == null || state.payloadSha256 == receipt.payloadSha256) {
            "Recovered lifecycle state belongs to a different payload"
        }
        val terminalState = when (receipt.result.outcome) {
            RadioSessionOutcome.SUCCEEDED -> RadioRequestStateKind.COMPLETED
            RadioSessionOutcome.PARTIAL_FAILED,
            RadioSessionOutcome.CANCELLED -> RadioRequestStateKind.FAILED
            null -> error("Recovered result has no explicit outcome")
        }
        writeTerminalLocked(
            requestId = requestId,
            payloadDigest = receipt.payloadSha256,
            terminalState = terminalState,
            detail = receipt.terminalDetail,
            deletePayload = terminalState == RadioRequestStateKind.COMPLETED,
        )
    }

    fun markCompleted(requestId: String) = markTerminal(
        requestId = requestId,
        terminalState = RadioRequestStateKind.COMPLETED,
        detail = null,
        deletePayload = true,
        allowPersisted = false,
    )

    fun markFailed(requestId: String, detail: String) = markTerminal(
        requestId = requestId,
        terminalState = RadioRequestStateKind.FAILED,
        detail = detail.take(MAX_TERMINAL_DETAIL_CHARS),
        deletePayload = false,
        allowPersisted = true,
    )

    private fun markTerminal(
        requestId: String,
        terminalState: RadioRequestStateKind,
        detail: String?,
        deletePayload: Boolean,
        allowPersisted: Boolean,
    ) = synchronized(lock) {
        requireValidRequestId(requestId)
        require(terminalState in TERMINAL_STATES)
        val payloadBytes = readPayloadLocked(requestId) ?: return@synchronized
        val payloadDigest = sha256(payloadBytes)
        val state = readStateLocked(requestId) ?: run {
            RadioRequestState(
                requestId = requestId,
                payloadSha256 = payloadDigest,
                state = RadioRequestStateKind.PERSISTED,
                updatedAtEpochMs = clock(),
            )
        }
        if (state.state in TERMINAL_STATES) return@synchronized
        require(
            (allowPersisted && state.state == RadioRequestStateKind.PERSISTED) ||
                (state.state == RadioRequestStateKind.CLAIMED && state.ownerToken == ownerToken),
        ) {
            "Cannot finish an unowned radio request: $requestId"
        }
        require(state.payloadSha256 == payloadDigest || allowPersisted) {
            "Radio request payload digest mismatch"
        }
        if (terminalState == RadioRequestStateKind.COMPLETED) {
            val receipt = requireNotNull(readReceiptLocked(requestId, payloadDigest)) {
                "A radio request cannot complete without a durable verified-result receipt"
            }
            require(receipt.result.outcome == RadioSessionOutcome.SUCCEEDED) {
                "Only a successful verified result can complete a radio request"
            }
        }
        writeTerminalLocked(requestId, payloadDigest, terminalState, detail, deletePayload)
    }

    private fun writeTerminalLocked(
        requestId: String,
        payloadDigest: String,
        terminalState: RadioRequestStateKind,
        detail: String?,
        deletePayload: Boolean,
    ) {
        atomicWriter.write(
            stateFile(requestId),
            encodeState(
                RadioRequestState(
                    requestId = requestId,
                    payloadSha256 = payloadDigest,
                    state = terminalState,
                    terminalDetail = detail,
                    updatedAtEpochMs = clock(),
                ),
            ),
        )
        // Keep the small state tombstone so a redelivered Intent remains idempotent. Failed
        // payloads stay available for diagnosis until retention cleanup.
        if (deletePayload) {
            atomicDeleter.delete(payloadFile(requestId))
            atomicDeleter.delete(receiptFile(requestId))
        }
        cleanupTerminalStatesLocked()
    }

    /** Pending requests and interrupted claims that must be terminalized, oldest first. */
    fun recoverableRequestIds(): List<String> = synchronized(lock) {
        ensureDirectories()
        payloadDir.listFiles { file -> file.extension == PAYLOAD_EXTENSION }
            .orEmpty()
            .mapNotNull { file ->
                val requestId = file.nameWithoutExtension
                runCatching {
                    requireValidRequestId(requestId)
                    val stateResult = runCatching { readStateLocked(requestId) }
                    if (stateResult.isFailure) {
                        return@runCatching file.lastModified().coerceAtLeast(1L) to requestId
                    }
                    val state = stateResult.getOrNull()
                    if (state?.state in TERMINAL_STATES ||
                        (state?.state == RadioRequestStateKind.CLAIMED && state.ownerToken == ownerToken)
                    ) {
                        null
                    } else {
                        (state?.updatedAtEpochMs ?: file.lastModified().coerceAtLeast(1L)) to requestId
                    }
                }.getOrNull()
            }
            .sortedWith(compareBy<Pair<Long, String>> { it.first }.thenBy { it.second })
            .map { it.second }
    }

    internal fun encode(request: DurableRadioRequest): ByteArray {
        validate(request)
        return gson.toJson(request).toByteArray(Charsets.UTF_8)
    }

    internal fun decode(bytes: ByteArray): DurableRadioRequest {
        require(bytes.isNotEmpty() && bytes.size <= MAX_PAYLOAD_BYTES) {
            "Radio request payload has invalid size"
        }
        val request = gson.fromJson(bytes.toString(Charsets.UTF_8), DurableRadioRequest::class.java)
            ?: throw IllegalArgumentException("Radio request payload is empty")
        validate(request)
        return request
    }

    private fun validate(request: DurableRadioRequest) {
        require(request.schemaVersion == DurableRadioRequest.CURRENT_SCHEMA_VERSION) {
            "Unsupported radio request schema: ${request.schemaVersion}"
        }
        requireValidRequestId(request.requestId)
        require(request.createdAtEpochMs > 0L) { "Radio request timestamp must be positive" }
        validateGeneration(request.generation)
        require(request.providerGenerationId.matches(PROVIDER_GENERATION_ID_REGEX)) {
            "Invalid Poweramp provider generation ID"
        }
        val payloadCount = listOf(request.radio, request.multiSeed, request.directQueue)
            .count { it != null }
        require(payloadCount == 1) { "Radio request must contain exactly one payload" }

        when (request.kind) {
            DurableRadioRequestKind.RADIO -> {
                val payload = requireNotNull(request.radio) { "Radio payload is missing" }
                require(request.multiSeed == null && request.directQueue == null)
                validateConfig(payload.config)
                validatePinnedSeed(payload.seed)
                payload.origin.name
            }
            DurableRadioRequestKind.MULTI_SEED_RADIO -> {
                val payload = requireNotNull(request.multiSeed) { "Multi-seed payload is missing" }
                require(request.radio == null && request.directQueue == null)
                validateConfig(payload.config)
                require(payload.seeds.isNotEmpty() && payload.seeds.size <= MAX_MULTI_SEEDS) {
                    "Multi-seed request must contain 1..$MAX_MULTI_SEEDS seeds"
                }
                require(payload.seeds.any { it.weight != 0f }) {
                    "Multi-seed request needs at least one non-zero seed"
                }
                require(payload.seedIdentities.size == payload.seeds.size) {
                    "Multi-seed identities must align with seeds"
                }
                payload.seeds.forEach(::validateSeed)
                payload.seeds.zip(payload.seedIdentities).forEach { (seed, identity) ->
                    if (seed.trackId == null) {
                        require(identity == null) { "Text seeds cannot have a track identity" }
                    } else {
                        requireNotNull(identity) { "Song seeds require an exact track identity" }
                        validateSeedIdentity(identity)
                        require(identity.embeddedTrackId == seed.trackId) {
                            "Song seed identity does not match its track ID"
                        }
                    }
                }
                validateComposedQuerySpec(
                    querySpec = requireNotNull(payload.querySpec) {
                        "Composed-radio query specification is missing"
                    },
                    generation = request.generation,
                    seeds = payload.seeds,
                    seedIdentities = payload.seedIdentities,
                )
                require(
                    payload.config.effectiveLibraryAddedDays ==
                        payload.querySpec.effectiveLibraryAddedDays,
                ) {
                    "Composed-radio config and query use different added-date candidate windows"
                }
                require(
                    payload.composedContract.schemaVersion == ComposedRadioContract.CURRENT_SCHEMA_VERSION &&
                        payload.composedContract.rankingVersion == ComposedRadioContract.CURRENT_RANKING_VERSION,
                ) { "Unsupported composed-radio contract" }
                payload.composedContract.operator.name
                payload.origin.name
            }
            DurableRadioRequestKind.DIRECT_QUEUE -> {
                val payload = requireNotNull(request.directQueue) { "Direct queue payload is missing" }
                require(request.radio == null && request.multiSeed == null)
                require(payload.tracks.isNotEmpty() && payload.tracks.size <= MAX_DIRECT_TRACKS) {
                    "Direct queue request must contain 1..$MAX_DIRECT_TRACKS tracks"
                }
                require(payload.trackIdentities.size == payload.tracks.size) {
                    "Direct queue identities must align with tracks"
                }
                require(payload.resolvedPowerampFileIds.size == payload.tracks.size) {
                    "Direct queue Poweramp identities must align with tracks"
                }
                requireBounded(payload.label, MAX_LABEL_CHARS, "Direct queue label")
                payload.tracks.zip(payload.trackIdentities).forEach { (track, identity) ->
                    validateTrack(track)
                    validateSeedIdentity(identity)
                    require(track.id == identity.embeddedTrackId) {
                        "Direct queue identity does not match its track ID"
                    }
                }
                payload.resolvedPowerampFileIds.forEach { fileId ->
                    fileId?.let { require(it > 0L) { "Poweramp file ID must be positive" } }
                }
                payload.origin.name
                payload.placement.name
                validateDirectFindMusicEvidence(request, payload)
            }
        }
    }

    private fun validateGeneration(generation: RadioGenerationToken) {
        require(generation.schemaVersion == RadioGenerationToken.CURRENT_SCHEMA_VERSION)
        require(generation.generationId.matches(GENERATION_ID_REGEX)) { "Invalid generation ID" }
        require(generation.activationBindingId.matches(ACTIVATION_BINDING_ID_REGEX)) {
            "Invalid activation binding ID"
        }
        require(generation.manifestSha256.matches(SHA256_REGEX))
        requireBounded(generation.embeddingSpecId, MAX_ID_CHARS, "Embedding spec ID")
        require(generation.embeddingSpecId.isNotBlank())
        require(generation.databaseContentSha256.matches(SHA256_REGEX))
        require(generation.orderedTrackSetSha256.matches(SHA256_REGEX))
        require(generation.stableTrackUidMappingSha256.matches(SHA256_REGEX))
    }

    private fun validateSeedIdentity(identity: RadioSeedIdentity) {
        require(identity.embeddedTrackId >= 0L) { "Seed track ID must be non-negative" }
        identity.stableTrackSpanId?.let { stableId ->
            require(stableId.matches(STABLE_TRACK_SPAN_ID_REGEX)) { "Invalid stable track-span ID" }
        }
    }

    private fun validatePinnedSeed(seed: PinnedRadioSeed) {
        validateSeedIdentity(seed.identity)
        require(seed.displayTrack.realId >= -1L)
        require(seed.displayTrack.durationMs >= 0)
        requireBounded(seed.displayTrack.title, MAX_TRACK_FIELD_CHARS, "Seed title")
        seed.displayTrack.artist?.let { requireBounded(it, MAX_TRACK_FIELD_CHARS, "Seed artist") }
        seed.displayTrack.album?.let { requireBounded(it, MAX_TRACK_FIELD_CHARS, "Seed album") }
        seed.displayTrack.path?.let { requireBounded(it, MAX_PATH_CHARS, "Seed path") }
        require(seed.matchType != TrackMatcher.MatchType.COMPOSED_QUERY &&
            seed.matchType != TrackMatcher.MatchType.NOT_APPLICABLE) {
            "A single radio seed requires a real matching result"
        }
        seed.matchType.name
    }

    private fun validateConfig(config: RadioConfig) {
        require(config.configSchemaVersion == RadioConfig.CURRENT_CONFIG_SCHEMA_VERSION) {
            "Durable requests require the current radio-config schema"
        }
        require(config.numTracks in 1..MAX_RADIO_TRACKS)
        require(
            config.libraryAddedDays == null ||
                config.libraryAddedDays in 1..MAX_LIBRARY_ADDED_DAYS,
        ) { "Poweramp added-date day count is outside the supported range" }
        require(config.candidatePoolSize in 0..MAX_CANDIDATE_POOL_SIZE)
        if (config.selectionMode in setOf(
                com.powerampstartradio.ui.SelectionMode.CLOSEST,
                com.powerampstartradio.ui.SelectionMode.MMR,
                com.powerampstartradio.ui.SelectionMode.DPP,
            )
        ) {
            require(config.candidatePoolSize == 0 || config.candidatePoolSize >= config.numTracks) {
                "Candidate pool cannot be smaller than the requested queue"
            }
        }
        requireFiniteRange(config.mmrCandidatePoolFraction, 0f, 1f, exclusiveMin = true)
        requireFiniteRange(config.dppFixedCandidatePoolFraction, 0f, 1f, exclusiveMin = true)
        requireFiniteRange(config.anchorStrength, 0f, 1f)
        require(
            config.anchorHalfLifeTracks.isFinite() &&
                config.anchorHalfLifeTracks > 0f &&
                config.anchorHalfLifeTracks <= MAX_HALF_LIFE_TRACKS,
        ) { "Anchor half-life is outside the supported range" }
        requireFiniteRange(config.walkRestartAlpha, 0f, 1f)
        requireFiniteRange(config.momentumBeta, 0f, 1f)
        requireFiniteRange(config.diversityLambda, 0f, 1f)
        requireFiniteRange(config.dppQualityExponent, 0f, 8f)
        require(config.maxPerArtist in 1..MAX_RADIO_TRACKS)
        require(config.minArtistSpacing in 0..MAX_RADIO_TRACKS)
        require(!config.driftEnabled || config.selectionMode == com.powerampstartradio.ui.SelectionMode.MMR) {
            "Drift is defined only for MMR"
        }
        if (config.selectionMode == com.powerampstartradio.ui.SelectionMode.RANDOM_WALK) {
            require(config.walkRestartAlpha > 0f && config.walkRestartAlpha < 1f) {
                "Graph Explorer stop probability must be strictly between zero and one"
            }
        }
        // Access enum fields so malformed Gson payloads with null enum values fail closed.
        config.selectionMode.name
        config.libraryAddedRange.name
        config.driftMode.name
        config.anchorDecay.name
    }

    private fun validateSeed(seed: SeedSpec) {
        require(seed.embedding.size == EMBEDDING_DIMENSION) {
            "Seed embedding must have $EMBEDDING_DIMENSION values"
        }
        require(seed.embedding.all(Float::isFinite)) { "Seed embedding contains non-finite values" }
        requireFiniteRange(seed.weight, -1f, 1f)
        requireBounded(seed.label, MAX_LABEL_CHARS, "Seed label")
        seed.trackId?.let { require(it >= 0L) { "Seed track ID must be non-negative" } }
        seed.type.name
    }

    private fun validateDirectFindMusicEvidence(
        request: DurableRadioRequest,
        payload: DirectQueueRequestPayload,
    ) {
        val sessionEvidence = payload.findMusicSessionEvidence
        val trackEvidence = payload.findMusicTrackEvidence
        val requiresEvidence = payload.origin == QueueOrigin.TEXT_RESULT_LIST ||
            payload.origin == QueueOrigin.COMPOSED_RESULT_LIST
        require(!requiresEvidence || sessionEvidence != null) {
            "A directly queued Find Music result list requires its displayed query evidence"
        }
        require((sessionEvidence == null) == (trackEvidence == null)) {
            "Find Music session and row evidence must be present together"
        }
        if (sessionEvidence == null || trackEvidence == null) return
        require(payload.origin == QueueOrigin.TEXT_RESULT_LIST ||
            payload.origin == QueueOrigin.COMPOSED_RESULT_LIST ||
            payload.origin == QueueOrigin.HISTORY_REQUEUE) {
            "Find Music evidence is valid only for a displayed result list or its history replay"
        }
        require(trackEvidence.size == payload.tracks.size) {
            "Find Music row evidence must align with every directly queued track"
        }

        val query = sessionEvidence.querySpec
        validateDisplayedFindMusicQuery(query)
        val binding = requireNotNull(query.libraryBinding) {
            "Displayed Find Music evidence requires an exact embedding-generation binding"
        }
        require(binding.generationId.matches(GENERATION_ID_REGEX) &&
            binding.activationBindingId.matches(ACTIVATION_BINDING_ID_REGEX) &&
            binding.databaseContentSha256.matches(SHA256_REGEX) &&
            binding.orderedTrackSetSha256.matches(SHA256_REGEX)) {
            "Displayed Find Music evidence has an invalid generation binding"
        }
        if (payload.origin != QueueOrigin.HISTORY_REQUEUE) {
            require(
                binding.generationId == request.generation.generationId &&
                    binding.activationBindingId == request.generation.activationBindingId &&
                    binding.databaseContentSha256 == request.generation.databaseContentSha256 &&
                    binding.orderedTrackSetSha256 == request.generation.orderedTrackSetSha256,
            ) { "Displayed Find Music evidence belongs to a different embedding generation" }
            require(payload.label == query.displayLabel) {
                "Direct queue label differs from the displayed Find Music query"
            }
        }
        when (payload.origin) {
            QueueOrigin.TEXT_RESULT_LIST -> require(query.isSimplePositiveTextOnly) {
                "Text-result origin requires the raw text-to-audio ranking contract"
            }
            QueueOrigin.COMPOSED_RESULT_LIST -> require(!query.isSimplePositiveTextOnly) {
                "Composed-result origin requires a composed Find Music query"
            }
            else -> Unit
        }

        require(sessionEvidence.orderedActiveTrackIdsSha256.matches(SHA256_REGEX)) {
            "Displayed Find Music active-domain hash is invalid"
        }
        require(sessionEvidence.activeTrackCount > 0) {
            "Displayed Find Music active domain must be non-empty"
        }
        sessionEvidence.objectiveRankingDomainCount?.let { count ->
            require(count in 1..sessionEvidence.activeTrackCount) {
                "Find Music objective-rank domain is invalid"
            }
        }
        sessionEvidence.ingredientRankingDomainCount?.let { count ->
            require(count in 1..sessionEvidence.activeTrackCount) {
                "Find Music ingredient-rank domain is invalid"
            }
        }
        if (query.isSimplePositiveTextOnly) {
            require(sessionEvidence.allOfQueuePlan == null) {
                "Raw text ranking cannot carry an All-of membership plan"
            }
            require(
                sessionEvidence.ingredientRankingDomainCount == null,
            ) { "Raw text ranking cannot carry an ingredient-rank domain" }
            val objectiveDomainCount = sessionEvidence.objectiveRankingDomainCount
                ?: sessionEvidence.activeTrackCount
            val textPlan = sessionEvidence.textQueuePlan
            if (query.textResultPlanner == FindMusicTextResultPlanner.VARIED_DPP) {
                requireNotNull(textPlan) {
                    "Varied text results require their complete-domain selection proof"
                }
            }
            textPlan?.let { plan ->
                plan.requireValid()
                require(
                    plan.planner == query.textResultPlanner &&
                        plan.plannerVersion == query.textResultPlanner.currentVersion,
                ) { "Text result planner evidence differs from the displayed query" }
                require(
                    plan.completeCandidateDomainCount == objectiveDomainCount &&
                        plan.requestedResultCount == query.resultLimit,
                ) { "Text result planner evidence uses the wrong request or objective domain" }
                require(plan.orderedSelectedTrackIds == payload.tracks.map { it.id }) {
                    "Text result planner evidence differs from the displayed track order"
                }
                require(
                    plan.orderedOriginalTextObjectiveRanks ==
                        trackEvidence.map(FindMusicTrackEvidence::objectiveRank),
                ) { "Text result planner evidence differs from the original text ranks" }
            }
        } else {
            require(sessionEvidence.textQueuePlan == null) {
                "Composed Find Music results cannot carry a simple-text queue plan"
            }
            val allOfPlan = sessionEvidence.allOfQueuePlan
            if (
                query.textResultPlanner ==
                FindMusicTextResultPlanner.VARIED_ALL_OF_DPP
            ) {
                requireNotNull(allOfPlan) {
                    "Varied All-of results require their complete-domain selection proof"
                }
                allOfPlan.requireValid()
                require(
                    allOfPlan.plannerVersion ==
                        com.powerampstartradio.similarity.FindMusicAllOfQueuePlanner.PLANNER_VERSION &&
                        allOfPlan.completeCandidateDomainCount ==
                        sessionEvidence.objectiveRankingDomainCount &&
                        allOfPlan.requestedResultCount == query.resultLimit,
                ) { "All-of result planner evidence uses the wrong request or domain" }
                require(allOfPlan.orderedSelectedTrackIds == payload.tracks.map { it.id }) {
                    "All-of result planner evidence differs from the displayed track order"
                }
                require(
                    allOfPlan.orderedOriginalAllOfObjectiveRanks ==
                        trackEvidence.map(FindMusicTrackEvidence::objectiveRank),
                ) { "All-of planner evidence differs from the original objective ranks" }
            } else {
                require(allOfPlan == null) {
                    "Ranked or Refine results cannot carry a Varied All-of plan"
                }
            }
            if (sessionEvidence.objectiveRankingDomainCount != null ||
                sessionEvidence.ingredientRankingDomainCount != null
            ) {
                require(
                    sessionEvidence.objectiveRankingDomainCount != null &&
                        sessionEvidence.ingredientRankingDomainCount != null &&
                        sessionEvidence.objectiveRankingDomainCount <=
                        sessionEvidence.ingredientRankingDomainCount,
                ) { "Composed Find Music ranking domains are incomplete or inconsistent" }
            }
        }
        val reduction = sessionEvidence.stableResultReduction
        require(
            reduction.identityPolicyVersion == StableVisibleResultReducer.IDENTITY_POLICY_VERSION &&
                reduction.requestedVisibleCount == query.resultLimit &&
                reduction.scannedRowCount >= trackEvidence.size &&
                reduction.collapsedEquivalentCount in 0..reduction.scannedRowCount,
        ) { "Invalid displayed Find Music visible-result reduction evidence" }
        require(trackEvidence.size <= query.resultLimit &&
            trackEvidence.size <= sessionEvidence.activeTrackCount) {
            "Displayed Find Music row count exceeds its query or active domain"
        }

        val ingredientCount = query.activeIngredientCount
        var previousDisplayedRank = 0
        var previousObjectiveRank = 0
        val seenObjectiveRanks = HashSet<Int>()
        val objectiveRanksMustBeOrdered =
            query.textResultPlanner == FindMusicTextResultPlanner.CLOSEST
        trackEvidence.forEach { row ->
            require(row.displayedRank > previousDisplayedRank &&
                row.displayedRank <= query.resultLimit) {
                "Displayed Find Music ranks must be strictly increasing"
            }
            val objectiveDomainCount = sessionEvidence.objectiveRankingDomainCount
                ?: sessionEvidence.activeTrackCount
            require(row.objectiveRank in 1..objectiveDomainCount &&
                seenObjectiveRanks.add(row.objectiveRank) &&
                (!objectiveRanksMustBeOrdered || row.objectiveRank > previousObjectiveRank)
            ) {
                "Find Music objective ranks are invalid for the selected result planner"
            }
            previousDisplayedRank = row.displayedRank
            previousObjectiveRank = row.objectiveRank
            require(row.resultScore.isFinite() && row.rankingScore.isFinite() &&
                row.resultScore.toBits() == row.rankingScore.toBits()) {
                "Find Music row scores are invalid or disagree"
            }
            if (query.isSimplePositiveTextOnly) {
                require(row.resultScore in -1f..1f && row.ingredientPercentiles.isEmpty()) {
                    "Raw text-to-audio evidence must contain only cosine ranking facts"
                }
            } else {
                require(row.resultScore > 0f && row.resultScore <= 1f)
                require(row.ingredientPercentiles.size == ingredientCount &&
                    row.ingredientPercentiles.all { it.isFinite() && it > 0f && it <= 1f }) {
                    "Composed Find Music evidence has invalid ingredient percentiles"
                }
            }
        }
    }

    private fun validateDisplayedFindMusicQuery(query: FindMusicQuerySpec) {
        require(query.schemaVersion == FindMusicQuerySpec.CURRENT_SCHEMA_VERSION) {
            "Displayed Find Music evidence requires the current query schema"
        }
        query.textResultPlanner.name
        val queryContractError = validateFindMusicQueryContract(query)
        require(queryContractError == null) {
            queryContractError ?: "Invalid displayed Find Music query"
        }
        require(query.textIngredients.size <= FindMusicQuerySpec.MAX_TEXT_INGREDIENTS)
        query.textIngredients.forEach { text ->
            requireBounded(text.query, MAX_LABEL_CHARS, "Find Music text")
            requireFiniteRange(text.weight, 0f, 1f)
            if (text.query.isBlank()) {
                require(text.weight == 0f && !text.negative)
            }
        }
        require(query.songSeeds.size <= MAX_MULTI_SEEDS)
        query.songSeeds.forEach { anchor ->
            require(anchor.trackId >= 0L)
            requireFiniteRange(anchor.weight, 0f, 1f)
            anchor.stableTrackSpanId?.let { require(it.matches(STABLE_TRACK_SPAN_ID_REGEX)) }
            anchor.artist?.let { requireBounded(it, MAX_TRACK_FIELD_CHARS, "Find Music artist") }
            anchor.title?.let { requireBounded(it, MAX_TRACK_FIELD_CHARS, "Find Music title") }
        }
        val activeWeights = query.activeTextIngredients.map { it.weight } +
            query.songSeeds.filter { it.weight > 0f }.map { it.weight }
        require(activeWeights.isNotEmpty() && abs(activeWeights.sum() - 1f) <= 0.005f) {
            "Displayed Find Music ingredient weights must total 100%"
        }
        require(query.songSeeds.filter { it.weight > 0f }.map { it.trackId }.distinct().size ==
            query.songSeeds.count { it.weight > 0f }) {
            "A displayed Find Music song ID appears more than once"
        }
        require(query.songSeeds.filter { it.weight > 0f }.mapNotNull { it.stableTrackSpanId }
            .let { it.distinct().size == it.size }) {
            "A displayed verified source span appears more than once"
        }
        val binding = requireNotNull(query.libraryBinding)
        requireBounded(binding.bindingSpecId, MAX_ID_CHARS, "Find Music binding spec")
        require(binding.bindingSpecId.isNotBlank())
    }

    private fun validateComposedQuerySpec(
        querySpec: FindMusicQuerySpec,
        generation: RadioGenerationToken,
        seeds: List<SeedSpec>,
        seedIdentities: List<RadioSeedIdentity?>,
    ) {
        require(querySpec.schemaVersion == FindMusicQuerySpec.CURRENT_SCHEMA_VERSION) {
            "Durable composed radio requires the current Find Music schema"
        }
        val queryContractError = validateFindMusicQueryContract(querySpec)
        require(queryContractError == null) {
            queryContractError ?: "Invalid Find Music query contract"
        }
        require(querySpec.operator == FindMusicOperator.ALL_OF) {
            "Composed radio supports All of only"
        }
        require(querySpec.textIngredients.size <= FindMusicQuerySpec.MAX_TEXT_INGREDIENTS) {
            "Too many Find Music text ingredients"
        }
        querySpec.textIngredients.forEach { text ->
            requireBounded(text.query, MAX_LABEL_CHARS, "Find Music text")
            requireFiniteRange(text.weight, 0f, 1f)
            if (text.query.isBlank()) {
                require(text.weight == 0f && !text.negative) {
                    "Blank Find Music text cannot contribute to composed radio"
                }
            }
        }
        require(querySpec.songSeeds.size <= MAX_MULTI_SEEDS) { "Too many Find Music song ingredients" }
        querySpec.songSeeds.forEach { anchor ->
            require(anchor.trackId >= 0L) { "Find Music song ID must be non-negative" }
            requireFiniteRange(anchor.weight, 0f, 1f)
            anchor.stableTrackSpanId?.let { stableId ->
                require(stableId.matches(STABLE_TRACK_SPAN_ID_REGEX)) {
                    "Invalid Find Music stable track-span ID"
                }
            }
            anchor.artist?.let { requireBounded(it, MAX_TRACK_FIELD_CHARS, "Find Music artist") }
            anchor.title?.let { requireBounded(it, MAX_TRACK_FIELD_CHARS, "Find Music title") }
        }
        val activeWeights = buildList {
            querySpec.activeTextIngredients.forEach { add(it.weight) }
            querySpec.songSeeds.filter { it.weight > 0f }.forEach { add(it.weight) }
        }
        require(activeWeights.isNotEmpty() && abs(activeWeights.sum() - 1f) <= 0.005f) {
            "Active composed-radio ingredient weights must total 100%"
        }
        require(
            querySpec.songSeeds.filter { it.weight > 0f }.map { it.trackId }.distinct().size ==
                querySpec.songSeeds.count { it.weight > 0f },
        ) { "A Find Music song ID appears more than once" }
        require(
            querySpec.songSeeds.filter { it.weight > 0f }.mapNotNull { it.stableTrackSpanId }
                .let { it.distinct().size == it.size },
        ) { "A verified byte-identical indexed source span appears more than once" }

        val binding = requireNotNull(querySpec.libraryBinding) {
            "Composed radio requires an exact active-library binding"
        }
        requireBounded(binding.bindingSpecId, MAX_ID_CHARS, "Find Music binding spec")
        require(binding.bindingSpecId.isNotBlank())
        require(
            binding.generationId == generation.generationId &&
                binding.activationBindingId == generation.activationBindingId &&
                binding.databaseContentSha256 == generation.databaseContentSha256 &&
                binding.orderedTrackSetSha256 == generation.orderedTrackSetSha256,
        ) { "Find Music query is bound to a different embedding generation" }

        val expectedIngredients = buildList<ExpectedComposedIngredient> {
            querySpec.activeTextIngredients.forEach { text ->
                add(
                    ExpectedComposedIngredient(
                        type = SeedType.TEXT,
                        trackId = null,
                        stableTrackSpanId = null,
                        weight = if (text.negative) -text.weight else text.weight,
                        label = text.query,
                    ),
                )
            }
            querySpec.songSeeds.filter { it.weight > 0f }.forEach { anchor ->
                add(
                    ExpectedComposedIngredient(
                        type = SeedType.SONG,
                        trackId = anchor.trackId,
                        stableTrackSpanId = anchor.stableTrackSpanId,
                        weight = if (anchor.negative) -anchor.weight else anchor.weight,
                        label = anchor.displayLabel,
                    ),
                )
            }
        }
        require(seeds.size == expectedIngredients.size) {
            "Durable embeddings do not align with the displayed Find Music ingredients"
        }
        seeds.indices.forEach { index ->
            val seed = seeds[index]
            val identity = seedIdentities[index]
            val expected = expectedIngredients[index]
            require(
                seed.type == expected.type && seed.trackId == expected.trackId &&
                    seed.weight.toBits() == expected.weight.toBits() && seed.label == expected.label,
            ) { "Durable ingredient $index differs from the displayed Find Music request" }
            if (expected.type == SeedType.TEXT) {
                require(identity == null)
            } else {
                requireNotNull(identity)
                require(
                    identity.embeddedTrackId == expected.trackId &&
                        identity.stableTrackSpanId == expected.stableTrackSpanId,
                ) { "Durable song identity $index differs from the displayed Find Music request" }
            }
        }
    }

    private data class ExpectedComposedIngredient(
        val type: SeedType,
        val trackId: Long?,
        val stableTrackSpanId: String?,
        val weight: Float,
        val label: String,
    )

    private fun validateTrack(track: EmbeddedTrack) {
        require(track.id >= 0L) { "Track ID must be non-negative" }
        require(track.durationMs >= 0) { "Track duration must be non-negative" }
        requireBounded(track.metadataKey, MAX_TRACK_FIELD_CHARS, "Track metadata key")
        requireBounded(track.filenameKey, MAX_TRACK_FIELD_CHARS, "Track filename key")
        requireBounded(track.filePath, MAX_PATH_CHARS, "Track path")
        requireBounded(track.source, MAX_TRACK_FIELD_CHARS, "Track source")
        track.artist?.let { requireBounded(it, MAX_TRACK_FIELD_CHARS, "Track artist") }
        track.album?.let { requireBounded(it, MAX_TRACK_FIELD_CHARS, "Track album") }
        track.title?.let { requireBounded(it, MAX_TRACK_FIELD_CHARS, "Track title") }
    }

    private fun validateTerminalResult(
        request: DurableRadioRequest,
        result: RadioResult,
        terminalDetail: String?,
    ) {
        require(result.requestId == request.requestId) { "Result request ID mismatch" }
        require(result.generation == request.generation) { "Result generation mismatch" }
        require(result.providerGenerationId == request.providerGenerationId) {
            "Result Poweramp provider generation mismatch"
        }
        require(result.outcome != null) { "Durable result needs an explicit outcome" }
        require(result.isComplete) { "A terminal result cannot be streaming" }
        require(result.tracks.size <= MAX_DIRECT_TRACKS) { "Radio result is too large" }
        require(result.tracks.none { it.status == com.powerampstartradio.ui.QueueStatus.PENDING }) {
            "Terminal result contains pending tracks"
        }
        result.failureDetail?.let {
            requireBounded(it, MAX_TERMINAL_DETAIL_CHARS, "Result failure detail")
        }
        terminalDetail?.let {
            requireBounded(it, MAX_TERMINAL_DETAIL_CHARS, "Terminal detail")
        }
        require(result.failureDetail == terminalDetail) {
            "Result failure detail and terminal receipt detail differ"
        }
        val expectedOrigin = when (request.kind) {
            DurableRadioRequestKind.RADIO -> requireNotNull(request.radio).origin
            DurableRadioRequestKind.MULTI_SEED_RADIO -> requireNotNull(request.multiSeed).origin
            DurableRadioRequestKind.DIRECT_QUEUE -> requireNotNull(request.directQueue).origin
        }
        val expectedRequestedCount = result.tracks.size
        val delivery = requireNotNull(result.delivery) {
            "Durable result requires queue-delivery evidence"
        }
        require(delivery.origin == expectedOrigin) { "Result origin does not match its request" }
        require(delivery.requestedCount == expectedRequestedCount)
        require(delivery.rankedCount == result.tracks.size)
        require(delivery.resolvedCount == result.tracks.count { it.resolvedPowerampFileId != null })
        require(delivery.verifiedCount == result.tracks.count {
            it.status == com.powerampstartradio.ui.QueueStatus.QUEUED
        })
        require(delivery.notInLibraryCount == result.tracks.count {
            it.status == com.powerampstartradio.ui.QueueStatus.NOT_IN_LIBRARY
        })
        require(delivery.queueFailedCount == result.tracks.count {
            it.status == com.powerampstartradio.ui.QueueStatus.QUEUE_FAILED
        })
        require(delivery.mutationCount >= 0 && delivery.unexpectedObservedCount >= 0)
        result.tracks.forEach { track ->
            require(track.similarity.isFinite() && track.similarityToSeed.isFinite())
            track.resolvedPowerampFileId?.let { require(it > 0L) }
            track.resolvedPowerampQueueId?.let { require(it > 0L) }
            if (track.status == com.powerampstartradio.ui.QueueStatus.QUEUED) {
                require(track.resolvedPowerampQueueId != null) {
                    "A verified queued row is missing exact queue occurrence evidence"
                }
            } else {
                require(track.resolvedPowerampQueueId == null) {
                    "An unverified row cannot retain queue occurrence evidence"
                }
            }
            track.stableTrackSpanId?.let { stableId ->
                require(stableId.matches(STABLE_TRACK_SPAN_ID_REGEX))
            }
        }
        val verifiedQueueIds = result.tracks.mapNotNull { it.resolvedPowerampQueueId }
        require(verifiedQueueIds.toSet().size == verifiedQueueIds.size) {
            "Exact queue occurrence evidence is duplicated across result rows"
        }
        require((result.queueAnchorId == null) == (result.queueAnchorOccurrenceId == null)) {
            "Queue anchor file and occurrence evidence must be present together"
        }
        result.queueAnchorId?.let { require(it > 0L) }
        result.queueAnchorOccurrenceId?.let { require(it > 0L) }
        result.seedRankingIdentityCount?.let { rankDomainCount ->
            require(rankDomainCount >= 0) {
                "Seed-ranking identity domain cannot be negative"
            }
            result.eligibleCandidateIdentityCount?.let { candidateDomainCount ->
                require(candidateDomainCount in 0..rankDomainCount) {
                    "Candidate domain cannot exceed the full seed-ranking domain"
                }
            }
            result.tracks.forEach { track ->
                track.seedRank?.let { rank ->
                    require(rank in 1..rankDomainCount) {
                        "Seed rank is outside its full active identity domain"
                    }
                }
                track.driftRank?.let { rank ->
                    require(rank in 1..rankDomainCount) {
                        "Drift rank is outside its full active identity domain"
                    }
                }
            }
        }
        when (request.kind) {
            DurableRadioRequestKind.RADIO -> {
                val radio = requireNotNull(request.radio)
                require(result.seedIdentity == radio.seed.identity) {
                    "Radio result seed identity changed"
                }
                require(result.matchType == radio.seed.matchType) {
                    "Radio result match evidence changed"
                }
                require(result.composedContract == null && result.composedQuerySpec == null &&
                    result.stableResultReduction == null &&
                    result.findMusicSessionEvidence == null &&
                    result.tracks.none {
                        it.composedEvidence != null || it.findMusicEvidence != null
                    })
            }
            DurableRadioRequestKind.MULTI_SEED_RADIO -> {
                val multiSeed = requireNotNull(request.multiSeed)
                require(result.matchType == TrackMatcher.MatchType.COMPOSED_QUERY)
                require(result.findMusicSessionEvidence == null &&
                    result.tracks.none { it.findMusicEvidence != null })
                require(result.composedContract == multiSeed.composedContract) {
                    "Composed-radio contract changed"
                }
                require(result.composedQuerySpec == multiSeed.querySpec) {
                    "Composed-radio Find Music request changed"
                }
                val reduction = requireNotNull(result.stableResultReduction) {
                    "Composed radio is missing visible-result reduction evidence"
                }
                require(
                    reduction.identityPolicyVersion ==
                        StableVisibleResultReducer.IDENTITY_POLICY_VERSION &&
                        reduction.requestedVisibleCount == expectedRequestedCount &&
                        reduction.scannedRowCount >= result.tracks.size &&
                        reduction.collapsedEquivalentCount in 0..reduction.scannedRowCount,
                ) { "Invalid composed-radio visible-result reduction evidence" }
                val ingredientCount = multiSeed.seeds.size
                var previousObjectiveRank = 0
                result.tracks.forEach { track ->
                    val evidence = requireNotNull(track.composedEvidence) {
                        "Composed-radio row is missing objective evidence"
                    }
                    require(evidence.objectiveRank > previousObjectiveRank)
                    previousObjectiveRank = evidence.objectiveRank
                    require(evidence.objectiveScore.isFinite() &&
                        evidence.objectiveScore > 0f && evidence.objectiveScore <= 1f)
                    require(track.similarity.toBits() == evidence.objectiveScore.toBits() &&
                        track.similarityToSeed.toBits() == evidence.objectiveScore.toBits())
                    require(evidence.ingredientPercentiles.size == ingredientCount)
                    require(evidence.ingredientPercentiles.all {
                        it.isFinite() && it > 0f && it <= 1f
                    })
                }
            }
            DurableRadioRequestKind.DIRECT_QUEUE -> {
                val direct = requireNotNull(request.directQueue)
                require(result.matchType == TrackMatcher.MatchType.NOT_APPLICABLE)
                require(result.composedContract == null && result.composedQuerySpec == null &&
                    result.stableResultReduction == null &&
                    result.tracks.none { it.composedEvidence != null })
                require(result.findMusicSessionEvidence == direct.findMusicSessionEvidence) {
                    "Direct-queue Find Music session evidence changed"
                }
                require(result.directQueuePlacement == direct.placement)
                require(result.tracks.size == direct.tracks.size)
                result.tracks.indices.forEach { index ->
                    require(result.tracks[index].track == direct.tracks[index])
                    require(
                        result.tracks[index].stableTrackSpanId ==
                            direct.trackIdentities[index].stableTrackSpanId,
                    )
                    require(
                        result.tracks[index].resolvedPowerampFileId ==
                            direct.resolvedPowerampFileIds[index],
                    )
                    val expectedFindMusic = direct.findMusicTrackEvidence?.get(index)
                    require(result.tracks[index].findMusicEvidence == expectedFindMusic) {
                        "Direct-queue Find Music row evidence changed at index $index"
                    }
                    val expectedSimilarity = expectedFindMusic?.resultScore ?: 0f
                    require(result.tracks[index].similarity.toBits() ==
                        expectedSimilarity.toBits()) {
                        "Direct-queue displayed score changed at index $index"
                    }
                }
            }
        }
        when (result.outcome) {
            RadioSessionOutcome.SUCCEEDED -> {
                require(
                    delivery.verificationComplete && result.queueFailedCount == 0 &&
                        delivery.notInLibraryCount == 0 &&
                        delivery.verifiedCount == delivery.requestedCount,
                ) {
                    "Successful durable result is not fully verified"
                }
                require(terminalDetail == null) { "Successful durable result cannot have a failure" }
            }
            RadioSessionOutcome.PARTIAL_FAILED,
            RadioSessionOutcome.CANCELLED -> {
                require(!terminalDetail.isNullOrBlank()) { "Failed durable result needs a reason" }
            }
        }
    }

    private fun readPayloadLocked(requestId: String): ByteArray? {
        val bytes = atomicReader.read(payloadFile(requestId)) ?: return null
        require(bytes.isNotEmpty() && bytes.size <= MAX_PAYLOAD_BYTES) {
            "Radio request payload has invalid size"
        }
        return bytes
    }

    private fun readReceiptLocked(
        requestId: String,
        expectedPayloadSha256: String,
    ): DurableRadioResultReceipt? {
        val bytes = atomicReader.read(receiptFile(requestId)) ?: return null
        require(bytes.isNotEmpty() && bytes.size <= MAX_RECEIPT_BYTES) {
            "Radio result receipt has invalid size"
        }
        val receipt = gson.fromJson(
            bytes.toString(Charsets.UTF_8),
            DurableRadioResultReceipt::class.java,
        ) ?: throw IllegalArgumentException("Radio result receipt is empty")
        require(receipt.schemaVersion == DurableRadioResultReceipt.CURRENT_SCHEMA_VERSION)
        require(receipt.requestId == requestId)
        require(receipt.payloadSha256 == expectedPayloadSha256)
        require(receipt.createdAtEpochMs > 0L)
        val payload = requireNotNull(readPayloadLocked(requestId))
        validateTerminalResult(decode(payload), receipt.result, receipt.terminalDetail)
        return receipt
    }

    private fun encodeReceipt(receipt: DurableRadioResultReceipt): ByteArray {
        val bytes = gson.toJson(receipt).toByteArray(Charsets.UTF_8)
        require(bytes.size <= MAX_RECEIPT_BYTES) {
            "Radio result receipt is ${bytes.size} bytes; maximum is $MAX_RECEIPT_BYTES"
        }
        return bytes
    }

    private fun readStateLocked(requestId: String): RadioRequestState? {
        val bytes = atomicReader.read(stateFile(requestId)) ?: return null
        require(bytes.size in 1..MAX_STATE_BYTES) { "Invalid radio request state size" }
        val state = gson.fromJson(bytes.toString(Charsets.UTF_8), RadioRequestState::class.java)
            ?: throw IllegalArgumentException("Radio request state is empty")
        require(state.schemaVersion == RadioRequestState.CURRENT_SCHEMA_VERSION)
        require(state.requestId == requestId)
        require(state.payloadSha256.matches(SHA256_REGEX))
        require(state.updatedAtEpochMs > 0L)
        state.state.name
        if (state.state == RadioRequestStateKind.CLAIMED) {
            require(!state.ownerToken.isNullOrBlank())
        } else {
            require(state.ownerToken == null)
        }
        state.terminalDetail?.let {
            require(it.length <= MAX_TERMINAL_DETAIL_CHARS) { "Terminal detail is too long" }
        }
        return state
    }

    private fun encodeState(state: RadioRequestState): ByteArray {
        val bytes = gson.toJson(state).toByteArray(Charsets.UTF_8)
        check(bytes.size <= MAX_STATE_BYTES)
        return bytes
    }

    private fun cleanupTerminalStatesLocked() {
        val states = stateDir.listFiles { file -> file.extension == STATE_EXTENSION }.orEmpty()
            .mapNotNull { file ->
                runCatching { file to readStateLocked(file.nameWithoutExtension) }.getOrNull()
            }
            .filter { it.second?.state in TERMINAL_STATES }
            .sortedByDescending { it.second?.updatedAtEpochMs ?: 0L }
        val cutoff = clock() - CONSUMED_RETENTION_MS
        states.forEachIndexed { index, (file, state) ->
            if (index >= MAX_TERMINAL_TOMBSTONES || (state?.updatedAtEpochMs ?: 0L) < cutoff) {
                val payload = payloadFile(file.nameWithoutExtension)
                val receipt = receiptFile(file.nameWithoutExtension)
                if (atomicDeleter.delete(payload) && atomicDeleter.delete(receipt) &&
                    atomicDeleter.delete(file)
                ) {
                    // Both sides are gone. Never remove a tombstone while its payload survives.
                }
            }
        }
    }

    private fun ensureDirectories() {
        check(payloadDir.mkdirs() || payloadDir.isDirectory) { "Cannot create radio payload directory" }
        check(stateDir.mkdirs() || stateDir.isDirectory) { "Cannot create radio state directory" }
        check(receiptDir.mkdirs() || receiptDir.isDirectory) { "Cannot create radio receipt directory" }
    }

    private fun payloadFile(requestId: String) = File(payloadDir, "$requestId.$PAYLOAD_EXTENSION")
    private fun stateFile(requestId: String) = File(stateDir, "$requestId.$STATE_EXTENSION")
    private fun receiptFile(requestId: String) = File(receiptDir, "$requestId.$RECEIPT_EXTENSION")

    private fun requireValidRequestId(requestId: String) {
        require(requestId.matches(REQUEST_ID_REGEX)) { "Invalid radio request ID" }
    }

    private fun requireBounded(value: String, maxChars: Int, label: String) {
        require(value.length <= maxChars) { "$label exceeds $maxChars characters" }
    }

    private fun requireFiniteRange(
        value: Float,
        min: Float,
        max: Float,
        exclusiveMin: Boolean = false,
    ) {
        require(value.isFinite() && value <= max && if (exclusiveMin) value > min else value >= min) {
            "Value $value is outside the supported range"
        }
    }

    private fun sha256(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256")
        .digest(bytes)
        .joinToString("") { byte -> "%02x".format(byte) }

    companion object {
        private const val DIRECTORY_NAME = "radio_requests_v2"
        private const val PAYLOAD_DIRECTORY_NAME = "payloads"
        private const val STATE_DIRECTORY_NAME = "states"
        private const val RECEIPT_DIRECTORY_NAME = "receipts"
        private const val PAYLOAD_EXTENSION = "json"
        private const val STATE_EXTENSION = "state"
        private const val RECEIPT_EXTENSION = "result"
        private const val MAX_PAYLOAD_BYTES = 8 * 1024 * 1024
        private const val MAX_RECEIPT_BYTES = 32 * 1024 * 1024
        private const val MAX_STATE_BYTES = 4 * 1024
        private const val MAX_OUTSTANDING_REQUESTS = 32
        private const val MAX_TERMINAL_TOMBSTONES = 256
        private const val CONSUMED_RETENTION_MS = 7L * 24 * 60 * 60 * 1000
        private const val MAX_TERMINAL_DETAIL_CHARS = 1_024
        private const val MAX_MULTI_SEEDS = 64
        private const val MAX_DIRECT_TRACKS = 10_000
        private const val MAX_RADIO_TRACKS = 1_000
        private const val MAX_CANDIDATE_POOL_SIZE = 1_000_000
        private const val MAX_HALF_LIFE_TRACKS = 1_000_000f
        private const val EMBEDDING_DIMENSION = 768
        private const val MAX_LABEL_CHARS = 1_024
        private const val MAX_TRACK_FIELD_CHARS = 4_096
        private const val MAX_PATH_CHARS = 32_768
        private const val MAX_ID_CHARS = 1_024
        private val REQUEST_ID_REGEX = Regex("[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}")
        private val SHA256_REGEX = Regex("[0-9a-f]{64}")
        private val GENERATION_ID_REGEX = Regex("index-generation-v2-[0-9a-f]{64}")
        private val PROVIDER_GENERATION_ID_REGEX =
            Regex("poweramp-provider-snapshot-v3-sha256:[0-9a-f]{64}")
        private val ACTIVATION_BINDING_ID_REGEX = Regex("activation-binding-v3-[0-9a-f]{64}")
        private val STABLE_TRACK_SPAN_ID_REGEX = Regex("stable-track-span-v1-[0-9a-f]{64}")
        private val TERMINAL_STATES = setOf(
            RadioRequestStateKind.COMPLETED,
            RadioRequestStateKind.FAILED,
            RadioRequestStateKind.INTERRUPTED_NEEDS_RETRY,
        )
        private val locks = ConcurrentHashMap<String, Any>()
    }
}
