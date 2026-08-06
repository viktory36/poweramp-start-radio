#!/usr/bin/env bash
set -euo pipefail
umask 077

test_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
verifier="$test_script_dir/verify_production_indexing_completion.sh"

fail() {
    printf 'FAIL: %s\n' "$*" >&2
    exit 1
}

[[ -f "$verifier" ]] || fail "verifier is absent: $verifier"
for command_name in jq python3 sha256sum; do
    command -v "$command_name" >/dev/null || fail "$command_name is required"
done
bash -n "$verifier"
if command -v shellcheck >/dev/null; then
    shellcheck -x "$verifier"
fi

# shellcheck source-path=SCRIPTDIR
# shellcheck source=verify_production_indexing_completion.sh
PASR_COMPLETION_VERIFIER_HOST_TEST_SOURCE_ONLY=1 source "$verifier"

tmp="$(mktemp -d)"
trap 'rm -rf -- "$tmp"' EXIT

job_id="11111111-1111-4111-8111-111111111111"
expected_executable_count=1
expected_selected_count=2
expected_rejected_count=1

sha256_of() {
    sha256sum "$1" | awk '{print $1}'
}

write_fixture() {
    local root="$1" mode="${2:-valid}"
    python3 - "$root" "$verifier" "$mode" "$job_id" <<'PY'
import copy
import hashlib
import json
import sqlite3
import struct
import sys
from pathlib import Path

root = Path(sys.argv[1])
verifier = Path(sys.argv[2])
mode = sys.argv[3]
job_id = sys.argv[4]
root.joinpath("raw").mkdir(parents=True)
root.joinpath("active").mkdir()
root.joinpath("base").mkdir()

# Reuse the verifier's canonical encoders so the fixture follows the same production
# identity schemas while keeping all device and app code out of this source-only test.
source = verifier.read_text(encoding="utf-8")
python_source = source.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
definition_source = python_source.split("\ntry:\n", 1)[0]
saved_argv = sys.argv
sys.argv = ["verifier-definitions", str(root), job_id, "1", "base", "", "", "0" * 64]
definitions = {"__name__": "fixture_verifier_definitions"}
exec(compile(definition_source, str(verifier), "exec"), definitions)
sys.argv = saved_argv

CanonicalDigest = definitions["CanonicalDigest"]
activation_binding_identity = definitions["activation_binding_identity"]
embedding_spec_identity = definitions["embedding_spec_identity"]
generation_identity = definitions["generation_identity"]
job_spec_identity = definitions["job_spec_identity"]
metadata_sha = definitions["metadata_sha"]
private_base_binding_id = definitions["private_base_binding_id"]
put_int = definitions["put_int"]
put_long = definitions["put_long"]
put_string = definitions["put_string"]
stable_span_identity = definitions["stable_span_identity"]
text_retrieval_spec_identity = definitions["text_retrieval_spec_identity"]
work_identity = definitions["work_identity"]

DIMENSION = 768
STABLE_SPEC = "stable-track-span-v1:content-sha256:native-half-open-sample-span"
COMPATIBILITY_POLICY = (
    "unreceipted-clamp3-compatibility-base-v1:unknown-model-and-preprocessing:no-v2-claim"
)
PROVIDER_GENERATION = "poweramp-provider-snapshot-v2-sha256:" + hashlib.sha256(
    b"fixture exact provider snapshot"
).hexdigest()
PREDECESSOR_TRACK_ID = 10
INHERITED_TRACK_ID = 20
REPLACEMENT_TRACK_ID = 30
COMMITTED_AT = 1_700_000_000_123


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha_text(value):
    return sha_bytes(value.encode("utf-8"))


def sha_file(path):
    return sha_bytes(path.read_bytes())


def write_json(path, value):
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def vector(component):
    values = [0.0] * DIMENSION
    values[component] = 1.0
    return struct.pack(f"<{DIMENSION}f", *values)


def ordered_track_digest(track_ids):
    digest = hashlib.sha256()
    put_string(digest, "v2-ordered-track-set-v1")
    put_int(digest, len(track_ids))
    for track_id in track_ids:
        put_long(digest, track_id)
    return digest.hexdigest()


def database_content_digest(track_ids, vectors):
    digest = hashlib.sha256()
    put_string(digest, "v2-ordered-clamp3-content-v1")
    put_int(digest, len(track_ids))
    put_int(digest, DIMENSION)
    for track_id in track_ids:
        blob = vectors[track_id]
        put_long(digest, track_id)
        put_int(digest, len(blob))
        digest.update(blob)
    return digest.hexdigest()


def write_pemb(path, track_ids, vectors):
    with path.open("wb") as stream:
        stream.write(struct.pack("<IIII", 0x424D4550, 1, len(track_ids), DIMENSION))
        stream.write(struct.pack(f"<{len(track_ids)}Q", *track_ids))
        for track_id in track_ids:
            stream.write(vectors[track_id])


def write_graph(path, track_ids):
    neighbors = len(track_ids)
    with path.open("wb") as stream:
        stream.write(struct.pack("<II", len(track_ids), neighbors))
        stream.write(struct.pack(f"<{len(track_ids)}Q", *track_ids))
        weight = 1.0 / neighbors
        for _ in track_ids:
            for neighbor in range(neighbors):
                stream.write(struct.pack("<If", neighbor, weight))


def derive_coverages(track_ids, vectors, receipts):
    receipts_by_track = {receipt["track_id"]: receipt for receipt in receipts}

    stable_digest = hashlib.sha256()
    put_string(stable_digest, "v2-stable-track-uid-coverage-v1")
    put_int(stable_digest, len(track_ids))
    unique_stable_ids = set()
    full = sampled = 0
    for track_id in track_ids:
        put_long(stable_digest, track_id)
        receipt = receipts_by_track.get(track_id)
        if receipt is None:
            stable_digest.update(b"\x00")
            continue
        stable_digest.update(b"\x01")
        put_string(stable_digest, receipt["stable_track_span_id"])
        put_string(stable_digest, receipt["stable_identity_spec_id"])
        put_string(stable_digest, receipt["stable_identity_strength"])
        unique_stable_ids.add(receipt["stable_track_span_id"])
        full += int(receipt["stable_identity_strength"] == "FULL_CONTENT_SHA256")
        sampled += int(
            receipt["stable_identity_strength"] == "VERSIONED_SAMPLED_CONTENT_SHA256"
        )
    stable = {
        "coveredTrackCount": len(receipts),
        "uncoveredTrackCount": len(track_ids) - len(receipts),
        "uniqueStableTrackSpanCount": len(unique_stable_ids),
        "fullContentIdentityCount": full,
        "sampledContentIdentityCount": sampled,
        "mappingSha256": stable_digest.hexdigest(),
    }

    mapping_digest = hashlib.sha256()
    put_string(mapping_digest, "v2-embedding-spec-coverage-v1")
    put_int(mapping_digest, len(track_ids))
    compatibility_digest = hashlib.sha256()
    put_string(compatibility_digest, "v2-unreceipted-compatibility-content-v1")
    compatibility_count = 0
    receipt_counts = {}
    for track_id in track_ids:
        put_long(mapping_digest, track_id)
        receipt = receipts_by_track.get(track_id)
        if receipt is None:
            mapping_digest.update(b"\x00")
            blob = vectors[track_id]
            put_long(compatibility_digest, track_id)
            put_int(compatibility_digest, len(blob))
            compatibility_digest.update(blob)
            compatibility_count += 1
        else:
            mapping_digest.update(b"\x01")
            spec_id = receipt["embedding_spec_id"]
            put_string(mapping_digest, spec_id)
            receipt_counts[spec_id] = receipt_counts.get(spec_id, 0) + 1
    put_int(mapping_digest, len(receipt_counts))
    for spec_id in sorted(receipt_counts):
        put_string(mapping_digest, spec_id)
        put_int(mapping_digest, receipt_counts[spec_id])
    put_int(mapping_digest, compatibility_count)
    coverage = {
        "totalTrackCount": len(track_ids),
        "receiptBoundTrackCount": len(receipts),
        "receiptSpecTrackCounts": dict(sorted(receipt_counts.items())),
    }
    if compatibility_count:
        put_int(compatibility_digest, compatibility_count)
        compatibility = {
            "provenancePolicyId": COMPATIBILITY_POLICY,
            "trackCount": compatibility_count,
            "orderedContentSha256": compatibility_digest.hexdigest(),
        }
        coverage["compatibilityBase"] = compatibility
        put_string(mapping_digest, compatibility["provenancePolicyId"])
        put_string(mapping_digest, compatibility["orderedContentSha256"])
    coverage["mappingSha256"] = mapping_digest.hexdigest()
    return stable, coverage


def assert_independent_coverage_goldens():
    contract_vector = vector(0)
    if sha_bytes(contract_vector) != (
        "e61a2e3c81d6733c0f44e35ebb53f04f82a640202c6657e4fdeadd95a28857ff"
    ):
        raise AssertionError("one-row coverage contract vector changed")

    stable, coverage = derive_coverages([1], {1: contract_vector}, [])
    if stable["mappingSha256"] != (
        "4d2a4586ead6cfba7e32224c677b85907bfd80c88d877cca5d66a5ae9f3f4a28"
    ):
        raise AssertionError("unreceipted stable coverage framing changed")
    if coverage["compatibilityBase"]["orderedContentSha256"] != (
        "159ec98f929f01ec391ddd60151d949259fd2fd4e0e697bfae56a5b90bc1d266"
    ):
        raise AssertionError("compatibility coverage framing changed")
    if coverage["mappingSha256"] != (
        "de9ec2bb3a4b2bf0550257e29379651d458c8cc053bb936127951347a2ab2d9c"
    ):
        raise AssertionError("unreceipted embedding coverage framing changed")

    receipt = {
        "track_id": 1,
        "stable_track_span_id": "stable-track-span-v1-" + "1" * 64,
        "stable_identity_spec_id": STABLE_SPEC,
        "stable_identity_strength": "FULL_CONTENT_SHA256",
        "embedding_spec_id": "embedding-spec-v2-" + "d" * 64,
    }
    stable, coverage = derive_coverages([1], {1: contract_vector}, [receipt])
    if stable["mappingSha256"] != (
        "cf5d3cb2b26d4838ed31366ad6e48048f9f9b07a66cedea214b42e11aea98d0b"
    ):
        raise AssertionError("receipted stable coverage framing changed")
    if coverage["mappingSha256"] != (
        "f3e1a6816e6c5e7788a33da06e87dc9f73d152d01106ce2bc4f75fd857965db5"
    ):
        raise AssertionError("receipted embedding coverage framing changed")


assert_independent_coverage_goldens()


def track_metadata(track_id, title, file_path, duration_ms, filename_key):
    artist = "Fixture Artist"
    album = "Fixture Album"
    return {
        "id": track_id,
        "metadata_key": f"{artist}|{album}|{title}|{duration_ms}",
        "filename_key": filename_key,
        "artist": artist,
        "album": album,
        "title": title,
        "duration_ms": duration_ms,
        "file_path": file_path,
        "source": "phone",
    }


def metadata_tuple(metadata):
    return (
        metadata["metadata_key"],
        metadata["filename_key"],
        metadata["artist"],
        metadata["album"],
        metadata["title"],
        metadata["duration_ms"],
        metadata["file_path"],
        metadata["source"],
    )


def create_database(path, track_ids, vectors, metadata_rows, receipts, supersessions, manifest):
    connection = sqlite3.connect(path)
    connection.executescript("""
        PRAGMA journal_mode=DELETE;
        PRAGMA foreign_keys=ON;
        CREATE TABLE tracks (
          id INTEGER PRIMARY KEY,
          metadata_key TEXT NOT NULL,
          filename_key TEXT NOT NULL,
          artist TEXT,
          album TEXT,
          title TEXT,
          duration_ms INTEGER NOT NULL,
          file_path TEXT NOT NULL,
          source TEXT NOT NULL
        );
        CREATE TABLE embeddings_clamp3 (
          track_id INTEGER PRIMARY KEY REFERENCES tracks(id) ON DELETE CASCADE,
          embedding BLOB NOT NULL
        );
        CREATE TABLE v2_embedding_commit_receipts_v4 (
          receipt_schema_version INTEGER NOT NULL,
          work_id TEXT NOT NULL,
          stable_track_span_id TEXT NOT NULL,
          stable_identity_spec_id TEXT NOT NULL,
          stable_identity_strength TEXT NOT NULL,
          embedding_spec_id TEXT NOT NULL,
          provider_physical_path TEXT NOT NULL,
          provider_offset_ms INTEGER NOT NULL,
          provider_duration_ms INTEGER NOT NULL,
          track_id INTEGER NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
          metadata_sha256 TEXT NOT NULL,
          embedding_byte_length INTEGER NOT NULL,
          embedding_sha256 TEXT NOT NULL,
          committed_at_epoch_ms INTEGER NOT NULL,
          PRIMARY KEY(work_id, embedding_spec_id),
          UNIQUE(track_id)
        );
        CREATE TABLE v2_imported_row_supersessions_v1 (
          supersession_schema_version INTEGER NOT NULL,
          work_id TEXT NOT NULL,
          embedding_spec_id TEXT NOT NULL,
          job_spec_id TEXT NOT NULL,
          base_generation_id TEXT NOT NULL,
          base_manifest_sha256 TEXT NOT NULL,
          base_database_sha256 TEXT NOT NULL,
          private_base_binding_id TEXT NOT NULL,
          provider_snapshot_generation TEXT NOT NULL,
          predecessor_track_id INTEGER NOT NULL,
          predecessor_metadata_sha256 TEXT NOT NULL,
          predecessor_embedding_byte_length INTEGER NOT NULL,
          predecessor_embedding_sha256 TEXT NOT NULL,
          replacement_track_id INTEGER NOT NULL,
          committed_at_epoch_ms INTEGER NOT NULL,
          PRIMARY KEY(work_id, embedding_spec_id)
        );
        CREATE TABLE v2_index_generation_guard_v2 (
          singleton INTEGER PRIMARY KEY,
          receipt_schema_version INTEGER NOT NULL,
          is_valid INTEGER NOT NULL,
          activation_binding_id TEXT NOT NULL,
          job_spec_id TEXT NOT NULL,
          receipt_embedding_spec_id TEXT NOT NULL,
          text_retrieval_spec_id TEXT NOT NULL,
          embedding_coverage_sha256 TEXT NOT NULL,
          compatibility_base_content_sha256 TEXT,
          database_content_sha256 TEXT NOT NULL,
          ordered_track_set_sha256 TEXT NOT NULL,
          stable_uid_mapping_sha256 TEXT NOT NULL,
          embedding_sha256 TEXT NOT NULL,
          graph_sha256 TEXT NOT NULL
        );
    """)
    for track_id in track_ids:
        row = metadata_rows[track_id]
        connection.execute(
            "INSERT INTO tracks VALUES(?,?,?,?,?,?,?,?,?)",
            (track_id,) + metadata_tuple(row),
        )
        connection.execute(
            "INSERT INTO embeddings_clamp3 VALUES(?,?)", (track_id, vectors[track_id])
        )
    for receipt in receipts:
        connection.execute(
            """
            INSERT INTO v2_embedding_commit_receipts_v4 VALUES(
              4,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                receipt["work_id"],
                receipt["stable_track_span_id"],
                receipt["stable_identity_spec_id"],
                receipt["stable_identity_strength"],
                receipt["embedding_spec_id"],
                receipt["provider_physical_path"],
                receipt["provider_offset_ms"],
                receipt["provider_duration_ms"],
                receipt["track_id"],
                receipt["metadata_sha256"],
                receipt["embedding_byte_length"],
                receipt["embedding_sha256"],
                receipt["committed_at_epoch_ms"],
            ),
        )
    for row in supersessions:
        connection.execute(
            "INSERT INTO v2_imported_row_supersessions_v1 VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                1,
                row["work_id"],
                row["embedding_spec_id"],
                row["job_spec_id"],
                row["base_generation_id"],
                row["base_manifest_sha256"],
                row["base_database_sha256"],
                row["private_base_binding_id"],
                row["provider_snapshot_generation"],
                row["predecessor_track_id"],
                row["predecessor_metadata_sha256"],
                row["predecessor_embedding_byte_length"],
                row["predecessor_embedding_sha256"],
                row["replacement_track_id"],
                row["committed_at_epoch_ms"],
            ),
        )
    compatibility = manifest["embeddingCoverage"].get("compatibilityBase")
    connection.execute(
        "INSERT INTO v2_index_generation_guard_v2 VALUES(1,3,1,?,?,?,?,?,?,?,?,?,?,?)",
        (
            manifest["activationBindingId"],
            manifest["jobSpecId"],
            manifest["receiptEmbeddingSpec"]["specId"],
            manifest["textRetrievalSpec"]["specId"],
            manifest["embeddingCoverage"]["mappingSha256"],
            compatibility["orderedContentSha256"] if compatibility else None,
            manifest["databaseContentSha256"],
            manifest["orderedTrackSetSha256"],
            manifest["stableTrackUidCoverage"]["mappingSha256"],
            manifest["embeddingSha256"],
            manifest["graph"]["sha256"],
        ),
    )
    connection.commit()
    connection.close()


def build_generation(
    scope,
    track_ids,
    vectors,
    metadata_rows,
    receipts,
    supersessions,
    origin,
    manifest_job_id,
    manifest_job_spec_id,
    base_generation_id,
    corrupt_coverage=False,
):
    directory = root / scope
    pemb_path = directory / "clamp3.emb"
    graph_path = directory / "graph.bin"
    write_pemb(pemb_path, track_ids, vectors)
    write_graph(graph_path, track_ids)
    stable, coverage = derive_coverages(track_ids, vectors, receipts)
    if corrupt_coverage:
        coverage["mappingSha256"] = "0" * 64
    ordered_sha = ordered_track_digest(track_ids)
    manifest = {
        "schemaVersion": 3,
        "generationId": "",
        "baseGenerationId": base_generation_id,
        "origin": origin,
        "createdAtEpochMs": (
            1_700_000_000_000 if scope == "active" else 1_699_000_000_000
        ),
        "jobId": manifest_job_id,
        "jobSpecId": manifest_job_spec_id,
        "activationBindingId": "",
        "rebuildDerivedIndexes": True,
        "graphPolicy": "EXPLICIT_REBUILD",
        "trackCount": len(track_ids),
        "databaseRelativePath": "library.db",
        "databaseByteLength": 0,
        "databaseSha256": "",
        "databaseContentSha256": database_content_digest(track_ids, vectors),
        "embeddingRelativePath": "clamp3.emb",
        "embeddingByteLength": pemb_path.stat().st_size,
        "embeddingSha256": sha_file(pemb_path),
        "embeddingDimension": DIMENSION,
        "orderedTrackSetSha256": ordered_sha,
        "graph": {
            "relativePath": "graph.bin",
            "byteLength": graph_path.stat().st_size,
            "sha256": sha_file(graph_path),
            "nodeCount": len(track_ids),
            "neighborsPerNode": len(track_ids),
            "orderedTrackSetSha256": ordered_sha,
        },
        "receiptEmbeddingSpec": copy.deepcopy(embedding_spec),
        "textRetrievalSpec": copy.deepcopy(text_spec),
        "embeddingCoverage": coverage,
        "stableTrackUidCoverage": stable,
    }
    manifest["activationBindingId"] = activation_binding_identity(manifest)
    database_path = directory / "library.db"
    create_database(
        database_path,
        track_ids,
        vectors,
        metadata_rows,
        receipts,
        supersessions,
        manifest,
    )
    manifest["databaseByteLength"] = database_path.stat().st_size
    manifest["databaseSha256"] = sha_file(database_path)
    manifest["generationId"] = generation_identity(manifest)
    write_json(directory / "manifest.json", manifest)
    return manifest


embedding_spec = {
    "specId": "",
    "preprocessingSpecId": (
        "mert-clamp3-audio-v3:torchaudio-hann-v1-width6-rolloff0.99-f32-target-length:"
        "pcm24k-whole-span-zmuv:5s-window:1s-tail-zero-pad:"
        "zero-bookends:segment128-final-overlap:frame-weighted-average:l2"
    ),
    "decoderPolicyId": (
        "android-mediacodec-v3:resolved-half-open-us-native-sample-span:"
        "verify-eos-or-enforce-cue-boundary:aligned-polyphase-hq:canonical-24khz-pcm"
    ),
    "inferenceBackendPolicyId": (
        "litert-2.1.1-compiled-model-v1:mert-gpu-fp32-strict:"
        "clamp3-audio-gpu-fp32-strict:no-backend-fallback"
    ),
    "outputDimension": DIMENSION,
    "modelArtifactSha256": {
        "mert": sha_text("fixture-mert"),
        "clamp3_audio": sha_text("fixture-clamp3-audio"),
    },
}
embedding_spec["specId"] = embedding_spec_identity(embedding_spec)
text_spec = {
    "specId": "",
    "compatibleAudioEmbeddingSpecId": embedding_spec["specId"],
    "textModelSha256": sha_text("fixture-text-model"),
    "tokenizerModelSha256": "cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865",
    "tokenizerPolicyId": (
        "sentencepiece-v0.2.1-rev-31646a467d2051eb904e0b45de3a73e91fe1c1e3-"
        "xlm-roberta-model-native-encode-sp-unk0-to-3-else-plus1-"
        "bos0-eos2-pad1-seq128-v1"
    ),
    "tokenizerRuntimeContractSha256": (
        "e3f1abde1d51a6747a252f99b276359f1353b3637e39f85670e8189baa65d8f3"
    ),
    "outputSpaceId": "clamp3-joint-audio-text-l2-f32-v1",
    "outputDimension": DIMENSION,
    "inferenceBackendPolicyId": (
        "litert-2.1.1-compiled-model-v1:clamp3-text-cpu-strict:"
        "host-text-aggregation-v1:segment128-final-overlap:"
        "token-count-weighted-average:l2:no-backend-fallback"
    ),
}
text_spec["specId"] = text_retrieval_spec_identity(text_spec)

cue_path = "/storage/emulated/0/Music/Fixture/disc.flac"
predecessor_metadata = track_metadata(
    PREDECESSOR_TRACK_ID, "Cue One", cue_path, 30_000, "disc.cue:1"
)
inherited_metadata = track_metadata(
    INHERITED_TRACK_ID,
    "Inherited Track",
    "/storage/emulated/0/Music/Fixture/inherited.flac",
    60_000,
    "inherited.flac",
)
base_metadata = {
    PREDECESSOR_TRACK_ID: predecessor_metadata,
    INHERITED_TRACK_ID: inherited_metadata,
}
base_vectors = {
    PREDECESSOR_TRACK_ID: vector(0),
    INHERITED_TRACK_ID: vector(1),
}
base_job_id = "22222222-2222-4222-8222-222222222222"
base_job_spec_id = "job-spec-v5-" + sha_text("fixture imported base job spec")
base_manifest = build_generation(
    "base",
    [PREDECESSOR_TRACK_ID, INHERITED_TRACK_ID],
    base_vectors,
    base_metadata,
    [],
    [],
    "INDEXING_JOB",
    base_job_id,
    base_job_spec_id,
    None,
)
base_manifest_path = root / "base/manifest.json"
base_manifest_sha = sha_file(base_manifest_path)

source_fingerprint = {
    "fingerprintSpecId": "full-content-sha256-v1",
    "sizeBytes": 12_345_678,
    "lastModifiedEpochMs": 1_699_999_999_000,
    "fileKey": "fixture-device:disc.flac",
    "sampledContentSha256": None,
    "fullContentSha256": sha_text("fixture exact source content"),
}
finalized_span = {
    "kind": "LOGICAL_CUE",
    "authority": "PROVIDER_CUE_HALF_OPEN_SPAN",
    "executionBoundaryRequirement": "ENFORCE_PROVIDER_HALF_OPEN_SPAN",
    "providerSpan": {
        "offsetUs": 0,
        "durationUs": 30_000_000,
        "endExclusiveUs": 30_000_000,
    },
    "cueClassification": {
        "providerGroupRowCount": 3,
        "logicalRowCount": 2,
        "nonZeroOffsetRowIds": [7002],
        "rawSourceImageRowIds": [7999],
    },
    "container": {
        "physicalPath": cue_path,
        "audioTrackIndex": 0,
        "durationUsEstimate": 60_000_000,
        "durationEstimateSource": "MEDIA_EXTRACTOR",
        "sampleRateHz": 48_000,
        "channelCount": 2,
        "mime": "audio/flac",
    },
    "startUs": 0,
    "endExclusiveUs": 30_000_000,
    "startSourceSample": 0,
    "endSourceSampleExclusive": 1_440_000,
    "sourceSampleCount": 1_440_000,
    "exactSampleCount24k": 720_000,
    "expectedWork": {"mertWindows": 6, "clampSegments": 1},
}
stable_identity = stable_span_identity(source_fingerprint, finalized_span)
provider_row = {
    "powerampFileId": 7001,
    "physicalPath": cue_path,
    "providerPhysicalPath": cue_path,
    "offsetMs": 0,
    "offsetWasNull": False,
    "durationMs": 30_000,
    "cueSourceImageFolderId": None,
    "artist": "Fixture Artist",
    "album": "Fixture Album",
    "title": "Cue One",
}
descriptor = {
    "workId": "",
    "provisionalWorkId": None,
    "stableTrackSpanIdentity": stable_identity,
    "ordinal": 0,
    "powerampFileId": 7001,
    "providerSnapshotGeneration": PROVIDER_GENERATION,
    "providerRow": provider_row,
    "displayMetadata": {
        "artist": "Fixture Artist",
        "album": "Fixture Album",
        "title": "Cue One",
    },
    "normalizedMetadata": {
        "normalizationSpecId": "poweramp-track-normalization-v1",
        "artist": "fixture artist",
        "album": "fixture album",
        "title": "cue one",
        "metadataKey": "fixture artist|fixture album|cue one|30000",
    },
    "physicalPath": cue_path,
    "canonicalPath": cue_path,
    "sourceFingerprint": source_fingerprint,
    "finalizedAudioSpan": finalized_span,
}
descriptor["workId"] = work_identity(descriptor)
work_id = descriptor["workId"]

job_spec = {
    "jobId": job_id,
    "specId": "",
    "provisionalParentSpecId": None,
    "createdAtEpochMs": 1_700_000_000_000,
    "providerSnapshot": {
        "libraryGeneration": PROVIDER_GENERATION,
        "acquisition": {
            "queryUri": "content://com.maxmpz.audioplayer.data/files",
            "requestedColumns": ["_id", "folder_files", "folder_files.ifs_path"],
            "returnedColumns": ["_id", "folder_files", "folder_files.ifs_path"],
            "rowCount": 2,
            "cursorExhaustedNormally": True,
        },
    },
    "embeddingSpec": copy.deepcopy(embedding_spec),
    "textRetrievalSpec": copy.deepcopy(text_spec),
    "runtimeFingerprint": {
        "appVersionCode": 200,
        "appBuildId": "fixture-v2-debug",
        "decoderRuntimeId": "android-media-codec-fixture-v1",
        "platformFingerprint": "fixture/device/build:1/test-keys",
    },
    "baseGenerationId": base_manifest["generationId"],
    "rebuildDerivedIndexes": True,
    "tracks": [descriptor],
}
job_spec["specId"] = job_spec_identity(job_spec)

replacement_metadata = {
    "id": REPLACEMENT_TRACK_ID,
    "metadata_key": descriptor["normalizedMetadata"]["metadataKey"],
    "filename_key": "fixture artist - cue one",
    "artist": "Fixture Artist",
    "album": "Fixture Album",
    "title": "Cue One",
    "duration_ms": 30_000,
    "file_path": cue_path,
    "source": "phone-v2",
}
replacement_vector = vector(2)
receipt = {
    "work_id": work_id,
    "stable_track_span_id": stable_identity["stableTrackSpanId"],
    "stable_identity_spec_id": stable_identity["identitySpecId"],
    "stable_identity_strength": stable_identity["strength"],
    "embedding_spec_id": embedding_spec["specId"],
    "provider_physical_path": cue_path,
    "provider_offset_ms": 0,
    "provider_duration_ms": 30_000,
    "track_id": REPLACEMENT_TRACK_ID,
    "metadata_sha256": metadata_sha(metadata_tuple(replacement_metadata)),
    "embedding_byte_length": len(replacement_vector),
    "embedding_sha256": sha_bytes(replacement_vector),
    "committed_at_epoch_ms": COMMITTED_AT,
}
predecessor = {
    "trackId": PREDECESSOR_TRACK_ID,
    "metadata": {
        "metadataKey": predecessor_metadata["metadata_key"],
        "filenameKey": predecessor_metadata["filename_key"],
        "artist": predecessor_metadata["artist"],
        "album": predecessor_metadata["album"],
        "title": predecessor_metadata["title"],
        "durationMs": predecessor_metadata["duration_ms"],
        "filePath": predecessor_metadata["file_path"],
        "source": predecessor_metadata["source"],
    },
    "metadataSha256": metadata_sha(metadata_tuple(predecessor_metadata)),
    "embeddingByteLength": len(base_vectors[PREDECESSOR_TRACK_ID]),
    "embeddingSha256": sha_bytes(base_vectors[PREDECESSOR_TRACK_ID]),
}
private_binding = private_base_binding_id(
    job_id,
    job_spec["specId"],
    base_manifest["generationId"],
    base_manifest["databaseByteLength"],
    base_manifest["databaseSha256"],
    base_manifest_sha,
    base_manifest["databaseContentSha256"],
)
authorization = {
    "schemaVersion": 1,
    "jobId": job_id,
    "jobSpecId": job_spec["specId"],
    "baseGenerationId": base_manifest["generationId"],
    "baseManifestSha256": base_manifest_sha,
    "baseDatabaseByteLength": base_manifest["databaseByteLength"],
    "baseDatabaseSha256": base_manifest["databaseSha256"],
    "baseDatabaseContentSha256": base_manifest["databaseContentSha256"],
    "privateBaseBindingId": private_binding,
    "providerSnapshotGeneration": PROVIDER_GENERATION,
    "works": [
        {
            "workId": work_id,
            "powerampFileId": descriptor["powerampFileId"],
            "providerSpan": {
                "normalizedPhysicalPath": cue_path,
                "offsetMs": 0,
                "durationMs": 30_000,
            },
            "kind": "SUPERSESSION",
            "predecessor": predecessor,
        }
    ],
}
supersession = {
    "work_id": work_id,
    "embedding_spec_id": embedding_spec["specId"],
    "job_spec_id": job_spec["specId"],
    "base_generation_id": base_manifest["generationId"],
    "base_manifest_sha256": base_manifest_sha,
    "base_database_sha256": base_manifest["databaseSha256"],
    "private_base_binding_id": private_binding,
    "provider_snapshot_generation": PROVIDER_GENERATION,
    "predecessor_track_id": PREDECESSOR_TRACK_ID,
    "predecessor_metadata_sha256": predecessor["metadataSha256"],
    "predecessor_embedding_byte_length": predecessor["embeddingByteLength"],
    "predecessor_embedding_sha256": predecessor["embeddingSha256"],
    "replacement_track_id": REPLACEMENT_TRACK_ID,
    "committed_at_epoch_ms": COMMITTED_AT,
}
active_metadata = {
    INHERITED_TRACK_ID: copy.deepcopy(inherited_metadata),
    REPLACEMENT_TRACK_ID: replacement_metadata,
}
active_vectors = {
    INHERITED_TRACK_ID: base_vectors[INHERITED_TRACK_ID],
    REPLACEMENT_TRACK_ID: replacement_vector,
}
active_manifest = build_generation(
    "active",
    [INHERITED_TRACK_ID, REPLACEMENT_TRACK_ID],
    active_vectors,
    active_metadata,
    [receipt],
    [supersession],
    "INDEXING_JOB",
    job_id,
    job_spec["specId"],
    base_manifest["generationId"],
    corrupt_coverage=(mode == "coverage-hash-tamper"),
)
active_manifest_sha = sha_file(root / "active/manifest.json")

write_json(
    root / "raw/active-generation.json",
    {
        "schemaVersion": 2,
        "generationId": active_manifest["generationId"],
        "manifestSha256": active_manifest_sha,
        "previousGenerationIds": [base_manifest["generationId"]],
    },
)

artifact_common = {
    "embeddingSpecId": embedding_spec["specId"],
    "sourceFingerprint": source_fingerprint,
}
track_ledger = {
    "workId": work_id,
    "updatedAtEpochMs": COMMITTED_AT,
    "state": "COMMITTED",
    "checkpoint": "COMMITTED",
    "currentAttemptNumber": None,
    "activeFailureId": None,
    "stageProgress": None,
    "attemptCount": 1,
    "failures": [],
    "verifiedArtifacts": [
        dict(
            artifact_common,
            kind="MERT_FEATURES",
            storageKey=f"mert:{work_id}",
            byteLength=6 * 3072,
            sha256=sha_text("fixture exact MERT features"),
            completedUnits=6,
            plannedUnits=6,
            verifiedAtEpochMs=COMMITTED_AT - 2,
            executionBoundary={
                "requirement": "ENFORCE_PROVIDER_HALF_OPEN_SPAN",
                "observedStartSourceSample": finalized_span["startSourceSample"],
                "observedEndSourceSampleExclusive": finalized_span[
                    "endSourceSampleExclusive"
                ],
                "observedSourceSampleCount": finalized_span["sourceSampleCount"],
                "exactSampleCount24k": finalized_span["exactSampleCount24k"],
                "endOfStreamReached": False,
                "providerBoundaryEnforced": True,
            },
        ),
        dict(
            artifact_common,
            kind="CLAMP_VECTOR",
            storageKey=f"clamp:{work_id}",
            byteLength=3072,
            sha256=receipt["embedding_sha256"],
            completedUnits=1,
            plannedUnits=1,
            verifiedAtEpochMs=COMMITTED_AT - 1,
        ),
        dict(
            artifact_common,
            kind="DATABASE_COMMIT",
            storageKey=(
                f"sqlite:embeddings_clamp3:track:{REPLACEMENT_TRACK_ID}:"
                f"v2_embedding_commit_receipts_v4:{work_id}:{embedding_spec['specId']}"
            ),
            byteLength=3072,
            sha256=receipt["embedding_sha256"],
            completedUnits=1,
            plannedUnits=1,
            verifiedAtEpochMs=COMMITTED_AT,
        ),
    ],
}
activation_evidence = {
    "activatedAtEpochMs": COMMITTED_AT + 5,
    "generationId": active_manifest["generationId"],
    "activationBindingId": active_manifest["activationBindingId"],
    "jobSpecId": job_spec["specId"],
    "receiptEmbeddingSpecId": embedding_spec["specId"],
    "textRetrievalSpecId": text_spec["specId"],
    "baseGenerationId": base_manifest["generationId"],
    "rebuildDerivedIndexes": True,
    "manifestSha256": active_manifest_sha,
    "databaseSha256": active_manifest["databaseSha256"],
    "databaseContentSha256": active_manifest["databaseContentSha256"],
    "orderedTrackSetSha256": active_manifest["orderedTrackSetSha256"],
    "stableTrackUidMappingSha256": active_manifest["stableTrackUidCoverage"]["mappingSha256"],
    "embeddingSha256": active_manifest["embeddingSha256"],
    "graphSha256": active_manifest["graph"]["sha256"],
}
write_json(
    root / "raw/ledger.json",
    {
        "format": "poweramp-start-radio-v2-indexing-ledger",
        "schemaVersion": 5,
        "ledger": {
            "schemaVersion": 5,
            "revision": 20,
            "updatedAtEpochMs": COMMITTED_AT + 10,
            "state": "COMPLETE",
            "stateReason": "verified embeddings activated",
            "executionProfile": "BACKGROUND",
            "recoveryPhase": None,
            "jobSpec": job_spec,
            "tracks": [track_ledger],
            "activationEvidence": activation_evidence,
        },
    },
)

planned = {
    "powerampFileId": 7001,
    "providerPhysicalPath": cue_path,
    "offsetMs": 0,
    "durationMs": 30_000,
    "cueSourceImageFolderId": None,
}
rejected_selection = {
    "powerampFileId": 9001,
    "providerPhysicalPath": "/storage/emulated/0/Music/Fixture/unreadable.flac",
    "offsetMs": 0,
    "durationMs": 61_000,
    "cueSourceImageFolderId": None,
}
write_json(
    root / "raw/preflight-intent.json",
    {
        "format": "poweramp-start-radio-v2-indexing-preflight-intent",
        "schemaVersion": 2,
        "intent": {
            "schemaVersion": 2,
            "jobId": job_id,
            "baseGenerationId": base_manifest["generationId"],
            "state": "MATERIALIZED",
            "failureCode": None,
            "failureMessage": None,
            "rebuildDerivedIndexes": True,
            "executionProfile": "BACKGROUND",
            "revision": 4,
            "createdAtEpochMs": job_spec["createdAtEpochMs"],
            "updatedAtEpochMs": job_spec["createdAtEpochMs"] + 10,
            "progress": {
                "phase": "COMPLETE",
                "message": "Preflight materialized one executable logical CUE row",
                "completedUnits": None,
                "totalUnits": None,
            },
            "materializedSpecId": job_spec["specId"],
            "resolvedSpecId": None,
            "selected": [planned, rejected_selection],
            "planned": [planned],
            "rejected": [
                {
                    "code": "SOURCE_UNREADABLE",
                    "disposition": "RETRYABLE",
                    "retryTrigger": "SOURCE_AVAILABLE",
                    "diagnostic": "Fixture source could not be opened during preflight",
                    "selected": rejected_selection,
                }
            ],
        },
    },
)

write_json(
    root / "raw/executor-lease.json",
    {"schemaVersion": 1, "lastIssuedEpoch": 9, "active": None},
)
write_json(
    root / "raw/quiescence.json",
    {
        "schemaVersion": 1,
        "jobId": job_id,
        "jobComplete": True,
        "activeJobPointerAbsent": True,
        "activeJobPointerAtomicResidueAbsent": True,
        "activeGenerationPointerAtomicResidueAbsent": True,
        "executorLeaseActiveNull": True,
        "executorLeaseAtomicResidueAbsent": True,
        "indexingServiceAbsent": True,
        "indexingWakeLockAbsent": True,
        "authorizationNamespaceClean": True,
        "jobArtifactDirectoryAbsent": True,
        "jobDatabaseResidueAbsent": True,
        "generationStagingAbsent": True,
    },
)
write_json(
    root / "raw/authorization-namespace.json",
    {
        "schemaVersion": 1,
        "jobId": job_id,
        "currentPresent": True,
        "legacyPresent": False,
        "atomicResiduePaths": [],
        "clean": True,
    },
)
auth_envelope = {
    "format": "poweramp-start-radio-v2-imported-row-supersession",
    "schemaVersion": 1,
    "authorization": authorization,
}
if mode == "predecessor-metadata-mismatch":
    auth_envelope = copy.deepcopy(auth_envelope)
    auth_envelope["authorization"]["works"][0]["predecessor"]["metadata"]["title"] = (
        "Tampered predecessor title"
    )
write_json(root / "raw/imported-row-authorization.json", auth_envelope)
(root / "raw/authorization-path.txt").write_text(
    f"files/indexing_v2/jobs/{job_id}.imported-row-supersession-v1.auth\n",
    encoding="utf-8",
)
(root / "raw/services.txt").write_text("No services\n", encoding="utf-8")
(root / "raw/power.txt").write_text("Wake Locks: size=0\n", encoding="utf-8")
(root / "raw/residue.txt").write_text("", encoding="utf-8")

asset_rows = []
for scope in ("active", "base"):
    for relative in ("manifest.json", "library.db", "clamp3.emb", "graph.bin"):
        path = root / scope / relative
        asset_rows.append(
            f"{scope}\t{relative}\t{path.stat().st_size}\t{sha_file(path)}"
        )
(root / "raw/asset-sizes-sha256.tsv").write_text(
    "\n".join(asset_rows) + "\n", encoding="utf-8"
)
PY
}

golden="$tmp/golden"
write_fixture "$golden"
expected_base_generation_id="$(jq -r '.generationId' "$golden/base/manifest.json")"
expected_preflight_sha256="$(sha256_of "$golden/raw/preflight-intent.json")"
expected_base_manifest_sha256="$(sha256_of "$golden/base/manifest.json")"

verify_fixture() {
    local root="$1"
    evidence_dir="$root"
    rm -f -- "$root/summary.json" "$root/evidence-sha256.tsv"
    verify_captured_evidence
}

if ! (verify_fixture "$golden") >"$tmp/golden.stdout" 2>"$tmp/golden.stderr"; then
    cat "$tmp/golden.stdout" "$tmp/golden.stderr" >&2
    fail "coherent logical CUE supersession fixture was rejected"
fi
jq -e --arg job "$job_id" --arg base "$expected_base_generation_id" '
    .verified == true and .quiescent == true and .jobId == $job and
    .executableCount == 1 and .selectedCount == 2 and .rejectedCount == 1 and
    .logicalCueCount == 1 and .derivedSupersessionCount == 1 and
    .baseGenerationId == $base and
    .baseTrackCount == 2 and .activeTrackCount == 2 and
    .baseReceiptBoundTrackCount == 0 and .activeReceiptBoundTrackCount == 1 and
    .baseCompatibilityTrackCount == 2 and .activeCompatibilityTrackCount == 1
' "$golden/summary.json" >/dev/null || fail "success summary is incomplete"
[[ -s "$golden/evidence-sha256.tsv" ]] || fail "success did not seal the evidence"

expect_mutation_rejected() {
    local label="$1" expected_message="$2" mutation="$3"
    local scenario="$tmp/$label"
    cp -a -- "$golden" "$scenario"
    rm -f -- "$scenario/summary.json" "$scenario/evidence-sha256.tsv"
    "$mutation" "$scenario"
    if (verify_fixture "$scenario") >"$scenario.stdout" 2>"$scenario.stderr"; then
        fail "$label was unexpectedly accepted"
    fi
    grep -Eiq "$expected_message" "$scenario.stdout" "$scenario.stderr" || {
        cat "$scenario.stdout" "$scenario.stderr" >&2
        fail "$label failed without the expected diagnostic: $expected_message"
    }
}

expect_generated_rejected() {
    local label="$1" expected_message="$2" mode="$3"
    local scenario="$tmp/$label"
    write_fixture "$scenario" "$mode"
    if (verify_fixture "$scenario") >"$scenario.stdout" 2>"$scenario.stderr"; then
        fail "$label was unexpectedly accepted"
    fi
    grep -Eiq "$expected_message" "$scenario.stdout" "$scenario.stderr" || {
        cat "$scenario.stdout" "$scenario.stderr" >&2
        fail "$label failed without the expected diagnostic: $expected_message"
    }
}

replace_json() {
    local filter="$1" file="$2"
    jq "$filter" "$file" >"$file.tmp"
    mv -- "$file.tmp" "$file"
}

mutate_dual_authorization_namespace() {
    replace_json \
        '.currentPresent=true | .legacyPresent=true | .clean=false' \
        "$1/raw/authorization-namespace.json"
}

mutate_authorization_atomic_residue() {
    replace_json \
        '.atomicResiduePaths=["files/indexing_v2/jobs/job.auth.new"] | .clean=false' \
        "$1/raw/authorization-namespace.json"
}

mutate_nested_embedding_spec_id() {
    replace_json \
        '.ledger.jobSpec.embeddingSpec.specId="embedding-spec-v2-arbitrary"' \
        "$1/raw/ledger.json"
}

mutate_nested_stable_span_id() {
    replace_json \
        '.ledger.jobSpec.tracks[0].stableTrackSpanIdentity.stableTrackSpanId="stable-track-span-v1-arbitrary"' \
        "$1/raw/ledger.json"
}

mutate_work_id() {
    replace_json '.ledger.jobSpec.tracks[0].workId="work-v4-arbitrary"' \
        "$1/raw/ledger.json"
}

mutate_job_spec_id() {
    replace_json '.ledger.jobSpec.specId="job-spec-v5-arbitrary"' "$1/raw/ledger.json"
}

mutate_activation_binding_id() {
    replace_json '.activationBindingId="activation-binding-v3-arbitrary"' \
        "$1/active/manifest.json"
}

mutate_generation_id() {
    replace_json '.generationId="index-generation-v2-arbitrary"' \
        "$1/active/manifest.json"
}

expect_mutation_rejected dual-authorization-namespace \
    'authorization namespace|ambiguous|atomic residue' mutate_dual_authorization_namespace
expect_mutation_rejected authorization-atomic-residue \
    'authorization namespace|ambiguous|atomic residue' mutate_authorization_atomic_residue
expect_generated_rejected predecessor-metadata-mismatch \
    'predecessor.*fingerprint|metadata' predecessor-metadata-mismatch
expect_mutation_rejected arbitrary-nested-embedding-spec-id \
    'embedding spec.*identity|model' mutate_nested_embedding_spec_id
expect_mutation_rejected arbitrary-nested-stable-span-id \
    'stable track-span identity|stable UID|work identity' mutate_nested_stable_span_id
expect_mutation_rejected arbitrary-work-id 'work identity|track identities' mutate_work_id
expect_mutation_rejected arbitrary-job-spec-id 'job spec.*identity|job' mutate_job_spec_id
expect_mutation_rejected arbitrary-activation-binding-id \
    'activation binding.*identity|asset|manifest' mutate_activation_binding_id
expect_mutation_rejected arbitrary-generation-id \
    'generation.*identity|asset|manifest' mutate_generation_id
expect_generated_rejected coverage-hash-tamper \
    'embedding coverage binding|coverage.*derived' coverage-hash-tamper

# Device-side commands must remain inspection-only. Host evidence writes are expected.
if grep -Eiq \
    'content[[:space:]]+(insert|update|delete|call)|ACTION_(API|RELOAD)|DEBUG_START_RADIO|adb[^\n]*(install|uninstall|reboot)|(^|[[:space:]])(run_as|adb_device)[[:space:]]+(rm|mv|cp|mkdir|rmdir|touch|truncate|chmod|chown|kill)|am[[:space:]]+(start|startservice|start-foreground-service|broadcast|force-stop)|pm[[:space:]]+(clear|grant|revoke|disable|enable|reset-permissions)|settings[[:space:]]+put|input[[:space:]]+keyevent' \
    "$verifier"; then
    fail "completion verifier contains a device or app mutation surface"
fi

printf 'production indexing completion verifier fixture tests passed\n'
