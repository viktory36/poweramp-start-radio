package com.powerampstartradio.indexing.v2

import java.io.File
import java.nio.file.Files
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class V2ImportedRowSupersessionAuthorizationPolicyTest {
    @Test
    fun `new sidecar avoids ledger JSON namespace and legacy sidecar remains readable`() =
        withTempDir { root ->
            val authorization = authorization()
            val store = V2ImportedRowSupersessionAuthorizationStore(
                ledgerDirectory = root,
                atomicIo = fileIo(),
            )

            store.createOrRequireExact(authorization)
            val current = V2ImportedRowAuthorizationFileNamespace.currentFile(
                root,
                authorization.jobId,
            )
            assertTrue(current.isFile)
            assertFalse(current.name.endsWith(".json"))
            store.createOrRequireExact(authorization)

            val legacy = V2ImportedRowAuthorizationFileNamespace.legacyFile(
                root,
                authorization.jobId,
            )
            assertTrue(current.renameTo(legacy))
            store.createOrRequireExact(authorization)
            assertEquals(
                V2ImportedRowAuthorizationFileKind.LEGACY,
                V2ImportedRowAuthorizationFileNamespace.resolveExisting(
                    root,
                    authorization.jobId,
                )?.kind,
            )
        }

    @Test
    fun `current and legacy sidecars conflict even when their bytes agree`() =
        withTempDir { root ->
            val authorization = authorization()
            val store = V2ImportedRowSupersessionAuthorizationStore(
                ledgerDirectory = root,
                atomicIo = fileIo(),
            )
            store.createOrRequireExact(authorization)
            val current = V2ImportedRowAuthorizationFileNamespace.currentFile(
                root,
                authorization.jobId,
            )
            val legacy = V2ImportedRowAuthorizationFileNamespace.legacyFile(
                root,
                authorization.jobId,
            )
            current.copyTo(legacy)

            assertThrows(V2ImportedRowAuthorizationException::class.java) {
                store.createOrRequireExact(authorization)
            }
        }

    @Test
    fun `current or legacy atomic residue fails closed`() = withTempDir { root ->
        val authorization = authorization()
        listOf(
            V2ImportedRowAuthorizationFileNamespace.currentFile(root, authorization.jobId),
            V2ImportedRowAuthorizationFileNamespace.legacyFile(root, authorization.jobId),
        ).forEach { target ->
            File(target.path + ".bak").writeText("unfinished")
            assertThrows(V2ImportedRowAuthorizationException::class.java) {
                V2ImportedRowSupersessionAuthorizationStore(
                    ledgerDirectory = root,
                    atomicIo = fileIo(),
                ).createOrRequireExact(authorization)
            }
            File(target.path + ".bak").delete()
        }
    }

    @Test
    fun `valid sidecar keeps schema five ledger boundary unchanged`() {
        V2ImportedRowSupersessionAuthorizationPolicy.requireValid(authorization())

        assertEquals(5, V2IndexingLedgerSchema.VERSION)
        assertEquals(
            RetryTrigger.NEW_JOB_REQUIRED,
            V2TrackFailurePolicies.forCode(
                TrackFailureCode.IMPORTED_ROW_AUTHORIZATION_CHANGED,
            ).retryTrigger,
        )
        assertEquals(
            FailureDisposition.BLOCKED,
            V2TrackFailurePolicies.forCode(
                TrackFailureCode.IMPORTED_ROW_AUTHORIZATION_CHANGED,
            ).initialDisposition,
        )
    }

    @Test
    fun `one imported predecessor cannot authorize two selected works`() {
        val original = authorization()
        val predecessor = requireNotNull(original.works.last().predecessor)
        val duplicated = original.copy(
            works = original.works + V2ImportedRowWorkAuthorization(
                workId = "work-c",
                powerampFileId = 33L,
                providerSpan = V2CommittedProviderSpan("/storage/Music/cue.flac", 20_000L, 10_000L),
                kind = V2ImportedRowCommitKind.SUPERSESSION,
                predecessor = predecessor,
            ),
        )

        assertThrows(IllegalArgumentException::class.java) {
            V2ImportedRowSupersessionAuthorizationPolicy.requireValid(duplicated)
        }
    }

    @Test
    fun `duplicate provider span is rejected as ambiguous destructive evidence`() {
        val original = authorization()
        val duplicated = original.copy(
            works = listOf(
                original.works.first(),
                original.works.last().copy(
                    providerSpan = original.works.first().providerSpan,
                ),
            ),
        )

        assertThrows(IllegalArgumentException::class.java) {
            V2ImportedRowSupersessionAuthorizationPolicy.requireValid(duplicated)
        }
    }

    @Test
    fun `stale predecessor metadata fingerprint fails closed`() {
        val original = authorization()
        val repair = original.works.last()
        val stale = original.copy(
            works = original.works.dropLast(1) + repair.copy(
                predecessor = requireNotNull(repair.predecessor).copy(
                    metadataSha256 = "0".repeat(64),
                ),
            ),
        )

        assertThrows(IllegalArgumentException::class.java) {
            V2ImportedRowSupersessionAuthorizationPolicy.requireValid(stale)
        }
    }

    @Test
    fun `private base binding covers every immutable source identity member`() {
        val baseline = bindingId()
        assertNotEquals(baseline, bindingId(jobSpecId = "spec-b"))
        assertNotEquals(baseline, bindingId(baseGenerationId = "generation-b"))
        assertNotEquals(baseline, bindingId(sourceLength = 124L))
        assertNotEquals(baseline, bindingId(sourceSha = "b".repeat(64)))
        assertNotEquals(baseline, bindingId(manifestSha = "c".repeat(64)))
        assertNotEquals(baseline, bindingId(contentSha = "d".repeat(64)))
    }

    @Test
    fun `activation expects an audit only after durable repair commit evidence`() {
        val authorization = authorization()

        val beforeCommit = V2ImportedRowActivationPolicy.partition(
            authorization,
            committedWorkIds = setOf("work-a"),
        )
        assertTrue(beforeCommit.committedSupersessions.isEmpty())
        assertEquals(
            listOf("work-b"),
            beforeCommit.uncommittedSupersessions.map { it.workId },
        )

        val afterCommit = V2ImportedRowActivationPolicy.partition(
            authorization,
            committedWorkIds = setOf("work-a", "work-b"),
        )
        assertEquals(
            listOf("work-b"),
            afterCommit.committedSupersessions.map { it.workId },
        )
        assertTrue(afterCommit.uncommittedSupersessions.isEmpty())
        assertThrows(IllegalArgumentException::class.java) {
            V2ImportedRowActivationPolicy.partition(
                authorization,
                committedWorkIds = setOf("unknown-work"),
            )
        }
    }

    @Test
    fun `uncommitted predecessor must remain exact and unreceipted`() {
        val expected = requireNotNull(authorization().works.last().predecessor)
        val exact = V2ObservedImportedPredecessorEvidence(
            metadata = expected.metadata,
            embeddingByteLength = expected.embeddingByteLength,
            embeddingSha256 = expected.embeddingSha256,
            receiptCount = 0L,
        )
        V2ImportedRowPredecessorPolicy.requireExactUnreceipted(
            expected,
            exact,
            "test staging database",
        )

        listOf(
            exact.copy(metadata = exact.metadata.copy(title = "Changed")),
            exact.copy(embeddingSha256 = "9".repeat(64)),
            exact.copy(receiptCount = 1L),
        ).forEach { changed ->
            assertThrows(IllegalArgumentException::class.java) {
                V2ImportedRowPredecessorPolicy.requireExactUnreceipted(
                    expected,
                    changed,
                    "test staging database",
                )
            }
        }
        assertThrows(IllegalArgumentException::class.java) {
            V2ImportedRowPredecessorPolicy.requireExactUnreceipted(
                expected,
                null,
                "test staging database",
            )
        }
    }

    private fun authorization(): V2ImportedRowSupersessionAuthorization {
        val metadata = V2CommitTrackMetadata(
            metadataKey = "artist|album|cue b|10000",
            filenameKey = "cue.flac",
            artist = "Artist",
            album = "Album",
            title = "Cue B",
            durationMs = 10_000,
            filePath = "/storage/Music/cue.flac",
            source = "desktop",
        )
        return V2ImportedRowSupersessionAuthorization(
            schemaVersion = 1,
            jobId = "job-a",
            jobSpecId = "spec-a",
            baseGenerationId = "generation-a",
            baseManifestSha256 = "1".repeat(64),
            baseDatabaseByteLength = 123L,
            baseDatabaseSha256 = "2".repeat(64),
            baseDatabaseContentSha256 = "3".repeat(64),
            privateBaseBindingId = bindingId(),
            providerSnapshotGeneration = "provider-a",
            works = listOf(
                V2ImportedRowWorkAuthorization(
                    workId = "work-a",
                    powerampFileId = 11L,
                    providerSpan = V2CommittedProviderSpan(
                        "/storage/Music/cue.flac",
                        0L,
                        10_000L,
                    ),
                    kind = V2ImportedRowCommitKind.ADDITION,
                    predecessor = null,
                ),
                V2ImportedRowWorkAuthorization(
                    workId = "work-b",
                    powerampFileId = 22L,
                    providerSpan = V2CommittedProviderSpan(
                        "/storage/Music/cue.flac",
                        10_000L,
                        10_000L,
                    ),
                    kind = V2ImportedRowCommitKind.SUPERSESSION,
                    predecessor = V2ImportedPredecessorEvidence(
                        trackId = 9L,
                        metadata = metadata,
                        metadataSha256 = V2CommitMetadataIdentity.sha256(metadata),
                        embeddingByteLength = V2_CLAMP3_BLOB_BYTES,
                        embeddingSha256 = "4".repeat(64),
                    ),
                ),
            ),
        )
    }

    private fun bindingId(
        jobSpecId: String = "spec-a",
        baseGenerationId: String = "generation-a",
        sourceLength: Long = 123L,
        sourceSha: String = "2".repeat(64),
        manifestSha: String = "1".repeat(64),
        contentSha: String = "3".repeat(64),
    ): String = V2JobPrivateDatabaseBindingIdentity.compute(
        jobId = "job-a",
        jobSpecId = jobSpecId,
        baseGenerationId = baseGenerationId,
        sourceDatabaseByteLength = sourceLength,
        sourceDatabaseSha256 = sourceSha,
        baseManifestSha256 = manifestSha,
        baseDatabaseContentSha256 = contentSha,
    )

    private fun fileIo() = object : V2ImportedRowAuthorizationAtomicIo {
        override fun read(file: File): ByteArray = file.readBytes()

        override fun write(file: File, bytes: ByteArray) {
            file.parentFile?.mkdirs()
            file.writeBytes(bytes)
        }
    }

    private fun withTempDir(block: (File) -> Unit) {
        val root = Files.createTempDirectory("imported-row-sidecar-test").toFile()
        try {
            block(root)
        } finally {
            root.deleteRecursively()
        }
    }
}
