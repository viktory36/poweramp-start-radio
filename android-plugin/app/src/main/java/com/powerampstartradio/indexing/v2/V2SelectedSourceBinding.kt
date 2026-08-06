package com.powerampstartradio.indexing.v2

import java.io.File

data class V2SelectedSourceCandidate(
    val powerampFileId: Long,
    /** Stable lexical key from the complete provider snapshot. */
    val providerPathKey: String,
    /** Exact joined path returned by Poweramp before lexical normalization. */
    val providerPhysicalPath: String,
    val suppliedSourceFile: File,
)

data class V2SelectedSourceBinding(
    val powerampFileId: Long,
    val providerPathKey: String,
    val providerPhysicalPath: String,
    /** Normalized comparison key; never use this value for filesystem I/O. */
    val canonicalPathKey: String,
    /** Exact canonical filesystem spelling used for hashing and decoding. */
    val canonicalSourceFile: File,
) {
    val providerPathIsAlias: Boolean
        get() = providerPathKey != canonicalPathKey
}

fun interface V2CanonicalFileResolver {
    fun resolve(file: File): File
}

object V2RealCanonicalFileResolver : V2CanonicalFileResolver {
    override fun resolve(file: File): File = file.canonicalFile
}

/**
 * Upgrades selected provider assets from lexical provider identity to filesystem identity.
 *
 * The complete snapshot remains filesystem-independent. Only candidates that can enter a job
 * incur canonical I/O. A single provider symlink is retained as provider evidence and bound to its
 * target; two selected lexical assets resolving to one target fail closed because their complete
 * provider groups and CUE semantics may differ.
 */
class V2SelectedSourceBinder(
    private val canonicalFileResolver: V2CanonicalFileResolver =
        V2RealCanonicalFileResolver,
    private val pathNormalizer: V2ProviderLexicalPathNormalizer =
        V2StableProviderLexicalPathNormalizer,
) {
    fun bind(
        candidates: List<V2SelectedSourceCandidate>,
    ): Map<Long, V2SelectedSourceBinding> {
        val bindings = LinkedHashMap<Long, V2SelectedSourceBinding>(candidates.size)
        val canonicalOwners = LinkedHashMap<String, V2SelectedSourceBinding>()
        for (candidate in candidates) {
            if (bindings.containsKey(candidate.powerampFileId)) {
                fail(
                    V2IndexingPreflightFailureCode.DUPLICATE_POWERAMP_ROW,
                    candidate.powerampFileId,
                    "Poweramp row ${candidate.powerampFileId} has duplicate source candidates",
                )
            }
            val providerPathKey = normalizeProviderPath(
                candidate.providerPathKey,
                candidate.powerampFileId,
            )
            val exactProviderPathKey = normalizeProviderPath(
                candidate.providerPhysicalPath,
                candidate.powerampFileId,
            )
            if (providerPathKey != exactProviderPathKey) {
                fail(
                    V2IndexingPreflightFailureCode.PROVIDER_SNAPSHOT_INVALID,
                    candidate.powerampFileId,
                    "Provider row ${candidate.powerampFileId} has inconsistent lexical path evidence",
                )
            }

            val providerCanonical = canonicalize(
                File(candidate.providerPhysicalPath),
                candidate.powerampFileId,
            )
            val suppliedCanonical = canonicalize(
                candidate.suppliedSourceFile,
                candidate.powerampFileId,
            )
            if (providerCanonical.path != suppliedCanonical.path) {
                fail(
                    V2IndexingPreflightFailureCode.SOURCE_CHANGED,
                    candidate.powerampFileId,
                    "Supplied source for Poweramp row ${candidate.powerampFileId} resolves to " +
                        "${suppliedCanonical.path}, not provider target ${providerCanonical.path}",
                )
            }
            val canonicalPathKey = normalizeCanonicalPath(
                providerCanonical.path,
                candidate.powerampFileId,
            )

            val binding = V2SelectedSourceBinding(
                powerampFileId = candidate.powerampFileId,
                providerPathKey = providerPathKey,
                providerPhysicalPath = candidate.providerPhysicalPath,
                canonicalPathKey = canonicalPathKey,
                canonicalSourceFile = providerCanonical,
            )
            val previous = canonicalOwners[providerCanonical.path]
            if (previous != null && previous.providerPathKey != providerPathKey) {
                fail(
                    V2IndexingPreflightFailureCode.SOURCE_CANONICAL_ALIAS_COLLISION,
                    candidate.powerampFileId,
                    "Selected provider paths ${previous.providerPathKey} " +
                        "(row ${previous.powerampFileId}) and $providerPathKey " +
                        "(row ${candidate.powerampFileId}) resolve to the same canonical source " +
                        providerCanonical.path,
                )
            }
            canonicalOwners.putIfAbsent(providerCanonical.path, binding)
            bindings[candidate.powerampFileId] = binding
        }
        return bindings
    }

    /** Re-resolves every alias so a target swap cannot survive until job persistence. */
    fun requireStillBound(
        candidates: List<V2SelectedSourceCandidate>,
        expected: Map<Long, V2SelectedSourceBinding>,
    ): Map<Long, V2SelectedSourceBinding> {
        val current = bind(candidates)
        val changedId = expected.keys.union(current.keys).firstOrNull { id ->
            expected[id]?.canonicalSourceFile?.path != current[id]?.canonicalSourceFile?.path ||
                expected[id]?.providerPathKey != current[id]?.providerPathKey ||
                expected[id]?.providerPhysicalPath != current[id]?.providerPhysicalPath
        }
        if (changedId != null) {
            fail(
                V2IndexingPreflightFailureCode.SOURCE_CHANGED,
                changedId,
                "Canonical source target changed before job persistence for Poweramp row $changedId",
            )
        }
        return current
    }

    private fun canonicalize(file: File, powerampFileId: Long): File {
        val canonical = try {
            canonicalFileResolver.resolve(file)
        } catch (error: Exception) {
            throw V2IndexingPreflightException(
                code = V2IndexingPreflightFailureCode.SOURCE_UNREADABLE,
                powerampFileId = powerampFileId,
                message = "Unable to resolve canonical source for Poweramp row " +
                    "$powerampFileId: $file",
                cause = error,
            )
        }
        return canonical
    }

    private fun normalizeProviderPath(path: String, powerampFileId: Long): String = try {
        pathNormalizer.normalizeAbsolute(path)
    } catch (error: Exception) {
        throw V2IndexingPreflightException(
            code = V2IndexingPreflightFailureCode.PROVIDER_SNAPSHOT_INVALID,
            powerampFileId = powerampFileId,
            message = "Invalid provider path for Poweramp row $powerampFileId: $path",
            cause = error,
        )
    }

    private fun normalizeCanonicalPath(path: String, powerampFileId: Long): String = try {
        pathNormalizer.normalizeAbsolute(path)
    } catch (error: Exception) {
        throw V2IndexingPreflightException(
            code = V2IndexingPreflightFailureCode.SOURCE_UNREADABLE,
            powerampFileId = powerampFileId,
            message = "Invalid absolute source path for Poweramp row $powerampFileId: $path",
            cause = error,
        )
    }

    private fun fail(
        code: V2IndexingPreflightFailureCode,
        powerampFileId: Long,
        message: String,
    ): Nothing = throw V2IndexingPreflightException(
        code = code,
        powerampFileId = powerampFileId,
        message = message,
    )
}
