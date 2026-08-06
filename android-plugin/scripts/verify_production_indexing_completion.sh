#!/usr/bin/env bash
set -euo pipefail
umask 077

package="com.powerampstartradio.v2"

usage() {
    cat >&2 <<'EOF'
usage: verify_production_indexing_completion.sh \
    --job-id JOB_ID \
    --expected-executable-count COUNT \
    --expected-base-generation-id GENERATION_ID \
    --evidence-dir DIRECTORY \
    [--expected-selected-count COUNT] \
    [--expected-rejected-count COUNT] \
    [--expected-preflight-sha256 SHA256] \
    [--expected-base-manifest-sha256 SHA256] \
    [--quiescence-timeout-seconds SECONDS] \
    [--serial ADB_SERIAL]

This command is read-only. It never starts an activity or service and never writes
inside the app sandbox. It waits for the named job to be COMPLETE and quiescent,
captures its immutable evidence, and independently verifies the activated generation.
It intentionally accepts only production jobs with rebuildDerivedIndexes=true and an
EXPLICIT_REBUILD graph; graphless generations are outside this acceptance contract.
EOF
}

die() {
    printf 'FAIL: %s\n' "$*" >&2
    exit 1
}

require_commands() {
    local command_name
    for command_name in adb awk cat find grep jq mkdir mv python3 rm sed sha256sum \
        sleep sort tr xargs; do
        command -v "$command_name" >/dev/null || die "missing required command: $command_name"
    done
}

validate_safe_id() {
    local label="$1" value="$2"
    [[ "$value" =~ ^[A-Za-z0-9._:-]{1,160}$ ]] || die "invalid $label: $value"
}

validate_count() {
    local label="$1" value="$2"
    [[ "$value" =~ ^[0-9]+$ ]] || die "$label must be a non-negative integer"
}

validate_sha256() {
    [[ "$1" =~ ^[0-9a-f]{64}$ ]] || die "invalid SHA-256: $1"
}

resolve_serial() {
    if [[ -n "$serial" ]]; then
        :
    elif [[ -n "${ANDROID_SERIAL:-}" ]]; then
        serial="$ANDROID_SERIAL"
    else
        mapfile -t devices < <(adb devices | awk 'NR > 1 && $2 == "device" { print $1 }')
        ((${#devices[@]} == 1)) || die \
            "set --serial or ANDROID_SERIAL unless exactly one ready device is attached"
        serial="${devices[0]}"
    fi
    [[ "$(adb -s "$serial" get-state 2>/dev/null || true)" == "device" ]] || die \
        "device is not ready over adb: $serial"
}

adb_device() {
    adb -s "$serial" "$@"
}

run_as() {
    adb_device shell run-as "$package" "$@"
}

private_exists() {
    run_as test -e "$1" >/dev/null 2>&1
}

private_absent() {
    ! private_exists "$1"
}

pull_private_file() {
    local private_path="$1" host_file="$2" temporary
    temporary="${host_file}.tmp-$$"
    mkdir -p -- "${host_file%/*}"
    if ! adb_device exec-out run-as "$package" cat "$private_path" >"$temporary"; then
        rm -f -- "$temporary"
        return 1
    fi
    [[ -s "$temporary" ]] || { rm -f -- "$temporary"; return 1; }
    mv -- "$temporary" "$host_file"
}

pull_required_private_file() {
    pull_private_file "$1" "$2" || die "cannot capture required private file: $1"
}

private_file_size() {
    run_as stat -c '%s' "$1" | tr -d '\r\n'
}

private_file_sha256() {
    run_as sha256sum "$1" | tr -d '\r' | awk '{print $1}'
}

remote_find() {
    run_as find "$@" 2>/dev/null | tr -d '\r' || true
}

capture_terminal_snapshot() {
    local destination="$1" lease_tmp
    local ledger_tmp="$destination/ledger.json.tmp"
    local services_tmp power_tmp active_job_residue generation_pointer_residue
    local job_artifact_residue job_database_residue staging_residue
    local auth_current auth_legacy auth_base_residue auth_namespace_clean=false
    mkdir -p -- "$destination"

    rm -f -- "$ledger_tmp" "$destination/executor-lease.json.tmp"
    pull_private_file "files/indexing_v2/jobs/$job_id.json" "$ledger_tmp" || true
    local complete=false
    if [[ -s "$ledger_tmp" ]] && jq -e \
        --arg job "$job_id" \
        '.format == "poweramp-start-radio-v2-indexing-ledger" and
         .schemaVersion == 5 and .ledger.schemaVersion == 5 and
         .ledger.jobSpec.jobId == $job and .ledger.state == "COMPLETE"' \
        "$ledger_tmp" >/dev/null 2>&1; then
        complete=true
    fi

    active_job_residue="$(for path in \
        files/indexing_v2/active-job-id \
        files/indexing_v2/active-job-id.bak \
        files/indexing_v2/active-job-id.new; do
        private_exists "$path" && printf '%s\n' "$path"
    done)"
    generation_pointer_residue="$(for path in \
        files/indexing_v2/generations/active-generation.json.bak \
        files/indexing_v2/generations/active-generation.json.new; do
        private_exists "$path" && printf '%s\n' "$path"
    done)"
    job_artifact_residue="$(remote_find "files/indexing_v2/artifacts/$job_id")"
    job_database_residue="$(for path in \
        "files/indexing_v2/job-databases/$job_id.db" \
        "files/indexing_v2/job-databases/$job_id.db-wal" \
        "files/indexing_v2/job-databases/$job_id.db-shm" \
        "files/indexing_v2/job-databases/$job_id.db-journal" \
        "files/indexing_v2/job-databases/$job_id.binding.json" \
        "files/indexing_v2/job-databases/$job_id.binding.json.new" \
        "files/indexing_v2/job-databases/$job_id.binding.json.bak"; do
        private_exists "$path" && printf '%s\n' "$path"
    done)"
    staging_residue="$(remote_find files/indexing_v2/generations \
        -maxdepth 1 -name '.staging-*')"

    auth_current="files/indexing_v2/jobs/$job_id.imported-row-supersession-v1.auth"
    auth_legacy="files/indexing_v2/jobs/$job_id.imported-row-supersession-v1.json"
    local auth_current_present=false auth_legacy_present=false
    private_exists "$auth_current" && auth_current_present=true
    private_exists "$auth_legacy" && auth_legacy_present=true
    auth_base_residue="$(for path in \
        "$auth_current.bak" "$auth_current.new" "$auth_legacy.bak" "$auth_legacy.new"; do
        private_exists "$path" && printf '%s\n' "$path"
    done)"
    if [[ "$auth_base_residue" == "" ]] &&
        [[ ! ("$auth_current_present" == true && "$auth_legacy_present" == true) ]]; then
        auth_namespace_clean=true
    fi

    lease_tmp="$destination/executor-lease.json.tmp"
    pull_private_file files/indexing_v2/executor-lease.json "$lease_tmp" || true
    local lease_active_null=false lease_atomic_residue=""
    if [[ -s "$lease_tmp" ]] && jq -e '.active == null' "$lease_tmp" >/dev/null 2>&1; then
        lease_active_null=true
    fi
    for path in files/indexing_v2/executor-lease.json.bak \
        files/indexing_v2/executor-lease.json.new; do
        private_exists "$path" && lease_atomic_residue+="$path"$'\n'
    done

    services_tmp="$destination/services.txt.tmp"
    power_tmp="$destination/power.txt.tmp"
    adb_device shell dumpsys activity services "$package" | tr -d '\r' >"$services_tmp"
    adb_device shell dumpsys power | tr -d '\r' >"$power_tmp"
    local service_absent=false wake_lock_absent=false
    if ! grep -q 'IndexingService' "$services_tmp"; then service_absent=true; fi
    if ! grep -Eiq 'PARTIAL_WAKE_LOCK.*(com\.powerampstartradio\.v2:)?v2-indexing' \
        "$power_tmp"; then wake_lock_absent=true; fi

    jq -n \
        --arg jobId "$job_id" \
        --argjson complete "$complete" \
        --arg activeJob "$active_job_residue" \
        --arg generationPointer "$generation_pointer_residue" \
        --argjson leaseActiveNull "$lease_active_null" \
        --arg leaseAtomic "$lease_atomic_residue" \
        --argjson serviceAbsent "$service_absent" \
        --argjson wakeLockAbsent "$wake_lock_absent" \
        --argjson authorizationClean "$auth_namespace_clean" \
        --arg artifact "$job_artifact_residue" \
        --arg database "$job_database_residue" \
        --arg staging "$staging_residue" '
        {
          schemaVersion: 1,
          jobId: $jobId,
          jobComplete: $complete,
          activeJobPointerAbsent: ($activeJob == ""),
          activeJobPointerAtomicResidueAbsent: ($activeJob == ""),
          activeGenerationPointerAtomicResidueAbsent: ($generationPointer == ""),
          executorLeaseActiveNull: $leaseActiveNull,
          executorLeaseAtomicResidueAbsent: ($leaseAtomic == ""),
          indexingServiceAbsent: $serviceAbsent,
          indexingWakeLockAbsent: $wakeLockAbsent,
          authorizationNamespaceClean: $authorizationClean,
          jobArtifactDirectoryAbsent: ($artifact == ""),
          jobDatabaseResidueAbsent: ($database == ""),
          generationStagingAbsent: ($staging == "")
        }' >"$destination/quiescence.json.tmp"
    jq -n \
        --arg jobId "$job_id" \
        --argjson currentPresent "$auth_current_present" \
        --argjson legacyPresent "$auth_legacy_present" \
        --arg residue "$auth_base_residue" '
        {
          schemaVersion: 1,
          jobId: $jobId,
          currentPresent: $currentPresent,
          legacyPresent: $legacyPresent,
          atomicResiduePaths: ($residue | split("\n") | map(select(length > 0))),
          clean: (($residue == "") and (($currentPresent and $legacyPresent) | not))
        }' >"$destination/authorization-namespace.json.tmp"
    {
        printf '%s\n' "$active_job_residue"
        printf '%s\n' "$generation_pointer_residue"
        printf '%s\n' "$lease_atomic_residue"
        printf '%s\n' "$job_artifact_residue"
        printf '%s\n' "$job_database_residue"
        printf '%s\n' "$staging_residue"
        printf '%s\n' "$auth_base_residue"
        if [[ "$auth_current_present" == true && "$auth_legacy_present" == true ]]; then
            printf '%s\n%s\n' "$auth_current" "$auth_legacy"
        fi
    } | sed '/^$/d' >"$destination/residue.txt.tmp"
}

snapshot_is_terminal_and_quiescent() {
    jq -e '
      .jobComplete and .activeJobPointerAbsent and
      .activeJobPointerAtomicResidueAbsent and
      .activeGenerationPointerAtomicResidueAbsent and
      .executorLeaseActiveNull and .executorLeaseAtomicResidueAbsent and
      .indexingServiceAbsent and .indexingWakeLockAbsent and
      .authorizationNamespaceClean and
      .jobArtifactDirectoryAbsent and .jobDatabaseResidueAbsent and
      .generationStagingAbsent
    ' "$1/quiescence.json.tmp" >/dev/null
}

wait_for_terminal_quiescence() {
    local destination="$1" deadline=$((SECONDS + quiescence_timeout_seconds))
    while :; do
        capture_terminal_snapshot "$destination"
        if snapshot_is_terminal_and_quiescent "$destination"; then
            mv -- "$destination/ledger.json.tmp" "$destination/ledger.json"
            mv -- "$destination/executor-lease.json.tmp" "$destination/executor-lease.json"
            mv -- "$destination/services.txt.tmp" "$destination/services.txt"
            mv -- "$destination/power.txt.tmp" "$destination/power.txt"
            mv -- "$destination/quiescence.json.tmp" "$destination/quiescence.json"
            mv -- "$destination/authorization-namespace.json.tmp" \
                "$destination/authorization-namespace.json"
            mv -- "$destination/residue.txt.tmp" "$destination/residue.txt"
            return
        fi
        if ((SECONDS >= deadline)); then
            mv -- "$destination/quiescence.json.tmp" "$destination/quiescence.json"
            mv -- "$destination/authorization-namespace.json.tmp" \
                "$destination/authorization-namespace.json"
            mv -- "$destination/residue.txt.tmp" "$destination/residue.txt"
            die "job did not become COMPLETE and quiescent within ${quiescence_timeout_seconds}s; see $destination"
        fi
        sleep 2
    done
}

record_remote_asset() {
    local scope="$1" relative="$2" private_path="$3" size sha
    size="$(private_file_size "$private_path")"
    [[ "$size" =~ ^[0-9]+$ ]] || die "invalid remote size for $private_path"
    sha="$(private_file_sha256 "$private_path")"
    validate_sha256 "$sha"
    printf '%s\t%s\t%s\t%s\n' "$scope" "$relative" "$size" "$sha" \
        >>"$evidence_dir/raw/asset-sizes-sha256.tsv"
}

capture_generation_assets() {
    local scope="$1" generation_id="$2" pull_all="$3"
    local private_root="files/indexing_v2/generations/$generation_id"
    local host_root="$evidence_dir/$scope"
    local manifest="$host_root/manifest.json"
    mkdir -p -- "$host_root"
    pull_required_private_file "$private_root/manifest.json" "$manifest"
    jq -e --arg generation "$generation_id" '
      .schemaVersion == 3 and .generationId == $generation and
      .databaseRelativePath == "library.db" and
      .embeddingRelativePath == "clamp3.emb" and
      .graph.relativePath == "graph.bin"
    ' "$manifest" >/dev/null || die "$scope generation manifest is invalid or unsupported"

    record_remote_asset "$scope" manifest.json "$private_root/manifest.json"
    record_remote_asset "$scope" library.db "$private_root/library.db"
    record_remote_asset "$scope" clamp3.emb "$private_root/clamp3.emb"
    record_remote_asset "$scope" graph.bin "$private_root/graph.bin"
    pull_required_private_file "$private_root/library.db" "$host_root/library.db"
    if [[ "$pull_all" == true ]]; then
        pull_required_private_file "$private_root/clamp3.emb" "$host_root/clamp3.emb"
        pull_required_private_file "$private_root/graph.bin" "$host_root/graph.bin"
    fi
}

capture_evidence() {
    local raw="$evidence_dir/raw" active_generation
    mkdir -p -- "$raw"
    wait_for_terminal_quiescence "$raw"
    pull_required_private_file "files/indexing_v2/preflight-intents/$job_id.json" \
        "$raw/preflight-intent.json"
    pull_required_private_file files/indexing_v2/generations/active-generation.json \
        "$raw/active-generation.json"
    active_generation="$(jq -r '.generationId' "$raw/active-generation.json")"
    validate_safe_id "active generation ID" "$active_generation"
    [[ "$active_generation" != "$expected_base_generation_id" ]] || die \
        "active generation did not advance beyond the expected base"

    : >"$raw/asset-sizes-sha256.tsv"
    capture_generation_assets active "$active_generation" true
    capture_generation_assets base "$expected_base_generation_id" false

    local current_auth="files/indexing_v2/jobs/$job_id.imported-row-supersession-v1.auth"
    local legacy_auth="files/indexing_v2/jobs/$job_id.imported-row-supersession-v1.json"
    if private_exists "$current_auth"; then
        pull_required_private_file "$current_auth" "$raw/imported-row-authorization.json"
        printf '%s\n' "$current_auth" >"$raw/authorization-path.txt"
    elif private_exists "$legacy_auth"; then
        pull_required_private_file "$legacy_auth" "$raw/imported-row-authorization.json"
        printf '%s\n' "$legacy_auth" >"$raw/authorization-path.txt"
    fi
}

verify_captured_evidence() {
    local preflight_actual_sha base_manifest_actual_sha
    [[ -d "$evidence_dir" ]] || die "evidence directory is absent: $evidence_dir"
    for file in raw/ledger.json raw/preflight-intent.json raw/active-generation.json \
        raw/executor-lease.json raw/quiescence.json raw/services.txt raw/power.txt \
        raw/authorization-namespace.json raw/residue.txt raw/asset-sizes-sha256.tsv \
        active/manifest.json \
        active/library.db active/clamp3.emb active/graph.bin base/manifest.json \
        base/library.db; do
        [[ -f "$evidence_dir/$file" ]] || die "captured evidence is missing $file"
    done
    [[ ! -s "$evidence_dir/raw/residue.txt" ]] || die "terminal private residue remains"
    preflight_actual_sha="$(sha256sum "$evidence_dir/raw/preflight-intent.json" | awk '{print $1}')"
    if [[ -n "$expected_preflight_sha256" ]]; then
        [[ "$preflight_actual_sha" == "$expected_preflight_sha256" ]] || die \
            "preflight SHA-256 differs from the expected reviewed evidence"
    fi
    base_manifest_actual_sha="$(sha256sum "$evidence_dir/base/manifest.json" | awk '{print $1}')"
    if [[ -n "$expected_base_manifest_sha256" ]]; then
        [[ "$base_manifest_actual_sha" == "$expected_base_manifest_sha256" ]] || die \
            "base manifest SHA-256 differs from the expected immutable predecessor"
    fi

    python3 - "$evidence_dir" "$job_id" "$expected_executable_count" \
        "$expected_base_generation_id" "$expected_selected_count" \
        "$expected_rejected_count" "$preflight_actual_sha" <<'PY'
import array
import copy
import hashlib
import json
import math
import os
import re
import sqlite3
import struct
import sys
import unicodedata
from pathlib import Path

root = Path(sys.argv[1])
expected_job = sys.argv[2]
expected_executable = int(sys.argv[3])
expected_base = sys.argv[4]
expected_selected = int(sys.argv[5]) if sys.argv[5] else None
expected_rejected = int(sys.argv[6]) if sys.argv[6] else None
preflight_sha = sys.argv[7]

def fail(message):
    raise AssertionError(message)

def require(condition, message):
    if not condition:
        fail(message)

def load_json(relative):
    try:
        return json.loads((root / relative).read_text(encoding="utf-8"))
    except Exception as error:
        fail(f"cannot read {relative}: {error}")

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()

def put_int(digest, value):
    digest.update(struct.pack(">I", value))

def put_long(digest, value):
    digest.update(struct.pack(">Q", value))

def put_string(digest, value):
    encoded = value.encode("utf-8")
    put_int(digest, len(encoded))
    digest.update(encoded)

def put_nullable_string(digest, value):
    digest.update(b"\x00" if value is None else b"\x01")
    if value is not None:
        put_string(digest, value)

def metadata_sha(metadata):
    digest = hashlib.sha256()
    put_string(digest, "v2-commit-track-metadata-v1")
    put_string(digest, metadata[0])
    put_string(digest, metadata[1])
    put_nullable_string(digest, metadata[2])
    put_nullable_string(digest, metadata[3])
    put_nullable_string(digest, metadata[4])
    put_int(digest, metadata[5])
    put_string(digest, metadata[6])
    put_string(digest, metadata[7])
    return digest.hexdigest()

def private_base_binding_id(job_id, job_spec_id, base_generation_id,
                            database_length, database_sha, manifest_sha, content_sha):
    digest = hashlib.sha256()
    def put_nullable(value):
        digest.update(b"\x00" if value is None else b"\x01")
        if value is not None:
            encoded = value.encode("utf-8")
            put_int(digest, len(encoded))
            digest.update(encoded)
    put_nullable("v2-job-private-base-binding-v2")
    put_nullable(job_id)
    put_nullable(job_spec_id)
    put_nullable(base_generation_id)
    digest.update(struct.pack(">Q", database_length))
    put_nullable(database_sha)
    put_nullable(manifest_sha)
    put_nullable(content_sha)
    return digest.hexdigest()

class CanonicalDigest:
    def __init__(self):
        self.digest = hashlib.sha256()

    def boolean(self, value):
        self.digest.update(b"\x01" if value else b"\x00")

    def integer(self, value):
        self.digest.update(struct.pack(">I", value & 0xFFFFFFFF))

    def long(self, value):
        self.digest.update(struct.pack(">Q", value & 0xFFFFFFFFFFFFFFFF))

    def string(self, value):
        encoded = value.encode("utf-8")
        self.integer(len(encoded))
        self.digest.update(encoded)

    def nullable_string(self, value):
        self.boolean(value is not None)
        if value is not None:
            self.string(value)

    def nullable_long(self, value):
        self.boolean(value is not None)
        if value is not None:
            self.long(value)

    def strings(self, values):
        self.integer(len(values))
        for value in values:
            self.string(value)

    def longs(self, values):
        self.integer(len(values))
        for value in values:
            self.long(value)

    def source_fingerprint(self, value):
        self.string(value["fingerprintSpecId"])
        self.long(value["sizeBytes"])
        self.nullable_long(value.get("lastModifiedEpochMs"))
        self.nullable_string(value.get("fileKey"))
        self.nullable_string(value.get("sampledContentSha256"))
        self.nullable_string(value.get("fullContentSha256"))

    def provider_snapshot(self, value):
        self.string(value["libraryGeneration"])
        acquisition = value["acquisition"]
        self.string(acquisition["queryUri"])
        self.strings(acquisition["requestedColumns"])
        self.strings(acquisition["returnedColumns"])
        self.integer(acquisition["rowCount"])
        self.boolean(acquisition["cursorExhaustedNormally"])

    def provider_acoustic(self, value):
        self.string(value["physicalPath"])
        self.string(value["providerPhysicalPath"])
        self.long(value["offsetMs"])
        self.boolean(value["offsetWasNull"])
        self.long(value["durationMs"])
        self.nullable_long(value.get("cueSourceImageFolderId"))

    def provider_row(self, value):
        self.long(value["powerampFileId"])
        self.provider_acoustic(value)
        self.nullable_string(value.get("artist"))
        self.nullable_string(value.get("album"))
        self.nullable_string(value.get("title"))

    def cue_classification(self, value):
        self.integer(value["providerGroupRowCount"])
        self.integer(value["logicalRowCount"])
        self.longs(value["nonZeroOffsetRowIds"])
        self.longs(value["rawSourceImageRowIds"])

    def finalized_audio_span(self, value):
        self.string(value["kind"])
        self.string(value["authority"])
        self.string(value["executionBoundaryRequirement"])
        provider = value["providerSpan"]
        self.long(provider["offsetUs"])
        self.long(provider["durationUs"])
        self.long(provider["endExclusiveUs"])
        self.cue_classification(value["cueClassification"])
        container = value["container"]
        self.string(container["physicalPath"])
        self.integer(container["audioTrackIndex"])
        self.long(container["durationUsEstimate"])
        self.string(container["durationEstimateSource"])
        self.integer(container["sampleRateHz"])
        self.integer(container["channelCount"])
        self.string(container["mime"])
        self.long(value["startUs"])
        self.long(value["endExclusiveUs"])
        self.long(value["startSourceSample"])
        self.long(value["endSourceSampleExclusive"])
        self.long(value["sourceSampleCount"])
        self.long(value["exactSampleCount24k"])
        self.integer(value["expectedWork"]["mertWindows"])
        self.integer(value["expectedWork"]["clampSegments"])

    def stable_identity(self, value):
        self.string(value["identitySpecId"])
        self.string(value["stableTrackSpanId"])
        self.string(value["strength"])
        self.string(value["contentFingerprintSpecId"])
        self.string(value["contentSha256"])
        self.long(value["sourceSizeBytes"])
        self.integer(value["sourceSampleRateHz"])
        self.long(value["startSourceSample"])
        self.long(value["endSourceSampleExclusive"])

    def display_metadata(self, value):
        self.string(value["artist"])
        self.string(value["album"])
        self.string(value["title"])

    def normalized_metadata(self, value):
        self.string(value["normalizationSpecId"])
        self.string(value["artist"])
        self.string(value["album"])
        self.string(value["title"])
        self.string(value["metadataKey"])

    def runtime(self, value):
        self.long(value["appVersionCode"])
        self.string(value["appBuildId"])
        self.string(value["decoderRuntimeId"])
        self.string(value["platformFingerprint"])

    def hex(self):
        return self.digest.hexdigest()

def embedding_spec_identity(spec):
    digest = CanonicalDigest()
    digest.string("embedding-spec-v2")
    digest.string(spec["preprocessingSpecId"])
    digest.string(spec["decoderPolicyId"])
    digest.string(spec["inferenceBackendPolicyId"])
    digest.integer(spec["outputDimension"])
    models = spec["modelArtifactSha256"]
    digest.integer(len(models))
    for name in sorted(models):
        digest.string(name)
        digest.string(models[name].lower())
    return "embedding-spec-v2-" + digest.hex()

def text_retrieval_spec_identity(spec):
    digest = CanonicalDigest()
    digest.string("text-retrieval-spec-v2")
    digest.string(spec["compatibleAudioEmbeddingSpecId"])
    digest.string(spec["textModelSha256"])
    digest.string(spec["tokenizerModelSha256"])
    digest.string(spec["tokenizerPolicyId"])
    digest.string(spec["tokenizerRuntimeContractSha256"])
    digest.string(spec["outputSpaceId"])
    digest.integer(spec["outputDimension"])
    digest.string(spec["inferenceBackendPolicyId"])
    return "text-retrieval-spec-v2-" + digest.hex()

def stable_span_identity(source, span):
    if source.get("fullContentSha256") is not None:
        strength = "FULL_CONTENT_SHA256"
        fingerprint_spec = "full-content-sha256-v1"
        content_sha = source["fullContentSha256"].lower()
    else:
        require(source.get("sampledContentSha256") is not None,
                "source fingerprint has no content identity")
        strength = "VERSIONED_SAMPLED_CONTENT_SHA256"
        fingerprint_spec = source["fingerprintSpecId"]
        content_sha = source["sampledContentSha256"].lower()
    identity = {
        "identitySpecId": "stable-track-span-v1:content-sha256:native-half-open-sample-span",
        "stableTrackSpanId": "",
        "strength": strength,
        "contentFingerprintSpecId": fingerprint_spec,
        "contentSha256": content_sha,
        "sourceSizeBytes": source["sizeBytes"],
        "sourceSampleRateHz": span["container"]["sampleRateHz"],
        "startSourceSample": span["startSourceSample"],
        "endSourceSampleExclusive": span["endSourceSampleExclusive"],
    }
    digest = CanonicalDigest()
    digest.string(identity["identitySpecId"])
    digest.string(identity["strength"])
    digest.string(identity["contentFingerprintSpecId"])
    digest.string(identity["contentSha256"])
    digest.long(identity["sourceSizeBytes"])
    digest.integer(identity["sourceSampleRateHz"])
    digest.long(identity["startSourceSample"])
    digest.long(identity["endSourceSampleExclusive"])
    identity["stableTrackSpanId"] = "stable-track-span-v1-" + digest.hex()
    return identity

def canonical_path(path):
    return unicodedata.normalize("NFC", re.sub(r"/{2,}", "/", path.replace("\\", "/")))

def work_identity(descriptor):
    digest = CanonicalDigest()
    digest.string("track-acoustic-span-v4")
    digest.nullable_string(descriptor.get("provisionalWorkId"))
    digest.string(descriptor["stableTrackSpanIdentity"]["stableTrackSpanId"])
    digest.string(descriptor["canonicalPath"])
    digest.source_fingerprint(descriptor["sourceFingerprint"])
    digest.provider_acoustic(descriptor["providerRow"])
    digest.finalized_audio_span(descriptor["finalizedAudioSpan"])
    return "work-v4-" + digest.hex()

def job_spec_identity(spec):
    digest = CanonicalDigest()
    digest.string("job-spec-v5")
    digest.nullable_string(spec.get("provisionalParentSpecId"))
    digest.provider_snapshot(spec["providerSnapshot"])
    digest.string(spec["embeddingSpec"]["specId"])
    digest.string(spec["textRetrievalSpec"]["specId"])
    digest.runtime(spec["runtimeFingerprint"])
    digest.nullable_string(spec.get("baseGenerationId"))
    digest.boolean(spec["rebuildDerivedIndexes"])
    digest.integer(len(spec["tracks"]))
    for descriptor in spec["tracks"]:
        digest.string(descriptor["workId"])
        digest.nullable_string(descriptor.get("provisionalWorkId"))
        digest.stable_identity(descriptor["stableTrackSpanIdentity"])
        digest.integer(descriptor["ordinal"])
        digest.long(descriptor["powerampFileId"])
        digest.string(descriptor["providerSnapshotGeneration"])
        digest.provider_row(descriptor["providerRow"])
        digest.display_metadata(descriptor["displayMetadata"])
        digest.normalized_metadata(descriptor["normalizedMetadata"])
        digest.string(descriptor["physicalPath"])
        digest.string(descriptor["canonicalPath"])
        digest.source_fingerprint(descriptor["sourceFingerprint"])
        digest.finalized_audio_span(descriptor["finalizedAudioSpan"])
    return "job-spec-v5-" + digest.hex()

def sample_at_or_after(time_us, sample_rate):
    return (time_us // 1_000_000) * sample_rate + \
        ((time_us % 1_000_000) * sample_rate + 999_999) // 1_000_000

def float32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]

def resampled_length(input_samples, from_rate, to_rate=24_000):
    if from_rate == to_rate:
        return input_samples
    divisor = math.gcd(from_rate, to_rate)
    input_stride = from_rate // divisor
    output_phases = to_rate // divisor
    return math.ceil(float32(output_phases * float(input_samples) / input_stride))

def expected_work(samples_24k):
    full = samples_24k // 120_000
    tail = samples_24k % 120_000
    windows = full + (1 if tail >= 24_000 else 0)
    clamp_segments = 0 if windows == 0 else (windows + 2 + 127) // 128
    return {"mertWindows": windows, "clampSegments": clamp_segments}

def normalized_provider_path(path):
    require(isinstance(path, str) and path != "" and path.startswith("/") and "\\" not in path,
            "provider physical path is not an absolute forward-slash path")
    parts = []
    for part in unicodedata.normalize("NFC", path).split("/"):
        if part in {"", "."}:
            continue
        if part == "..":
            if parts:
                parts.pop()
        else:
            parts.append(part)
    return "/" + "/".join(parts)

def is_sorted_unique_positive(values):
    return (all(isinstance(value, int) and not isinstance(value, bool) and value > 0
                for value in values) and values == sorted(set(values)))

def validate_descriptor_semantics(descriptor, ordinal):
    require(descriptor["ordinal"] == ordinal and descriptor["powerampFileId"] > 0,
            f"descriptor {ordinal} has invalid ordinal or Poweramp ID")
    require(isinstance(descriptor["physicalPath"], str) and
            descriptor["physicalPath"].startswith("/") and descriptor["physicalPath"] != "" and
            descriptor["canonicalPath"] == canonical_path(descriptor["physicalPath"]),
            f"descriptor {ordinal} has invalid physical/canonical path")
    normalized = descriptor["normalizedMetadata"]
    require(normalized.get("normalizationSpecId") == "poweramp-track-normalization-v1" and
            isinstance(normalized.get("metadataKey"), str) and normalized["metadataKey"] != "",
            f"descriptor {ordinal} has invalid normalized metadata")

    provider = descriptor["providerRow"]
    require(provider["powerampFileId"] == descriptor["powerampFileId"] and
            provider["physicalPath"] == normalized_provider_path(provider["providerPhysicalPath"]) and
            (provider.get("offsetWasNull") is not True or provider["offsetMs"] == 0) and
            provider["offsetMs"] >= 0 and provider["durationMs"] >= 0 and
            provider.get("cueSourceImageFolderId") is None,
            f"descriptor {ordinal} has invalid provider-row evidence")
    span = descriptor["finalizedAudioSpan"]
    provider_span = span["providerSpan"]
    expected_provider_span = {
        "offsetUs": provider["offsetMs"] * 1_000,
        "durationUs": provider["durationMs"] * 1_000,
        "endExclusiveUs": (provider["offsetMs"] + provider["durationMs"]) * 1_000,
    }
    require(provider_span == expected_provider_span,
            f"descriptor {ordinal} provider millisecond/microsecond evidence differs")

    container = span["container"]
    unavailable = (container["durationUsEstimate"] == 0 and
                   container["durationEstimateSource"] == "UNAVAILABLE")
    valid_duration = ((container["durationUsEstimate"] > 0 and
                       container["durationEstimateSource"] != "UNAVAILABLE") or
                      (unavailable and span["kind"] == "WHOLE_FILE"))
    require(container["physicalPath"] == descriptor["canonicalPath"] and
            container["audioTrackIndex"] >= 0 and valid_duration and
            0 < container["sampleRateHz"] <= 1_000_000 and
            container["channelCount"] > 0 and
            isinstance(container["mime"], str) and container["mime"].startswith("audio/"),
            f"descriptor {ordinal} has invalid container evidence")

    require(span["startUs"] >= 0 and span["endExclusiveUs"] > span["startUs"] and
            span["startSourceSample"] >= 0 and
            span["endSourceSampleExclusive"] > span["startSourceSample"] and
            span["sourceSampleCount"] > 0 and span["exactSampleCount24k"] > 0,
            f"descriptor {ordinal} has invalid finalized acoustic span")
    if span["kind"] == "WHOLE_FILE":
        require(span["authority"] == "DECODED_END_OF_STREAM" and
                span["executionBoundaryRequirement"] ==
                "VERIFY_END_OF_STREAM_AND_RECONCILE" and span["startUs"] == 0,
                f"descriptor {ordinal} lacks decoded EOS authority")
        require(span["cueClassification"]["nonZeroOffsetRowIds"] == [] and
                span["cueClassification"]["rawSourceImageRowIds"] == [],
                f"descriptor {ordinal} ordinary span carries CUE structure")
    elif span["kind"] == "LOGICAL_CUE":
        require(span["authority"] == "PROVIDER_CUE_HALF_OPEN_SPAN" and
                span["executionBoundaryRequirement"] == "ENFORCE_PROVIDER_HALF_OPEN_SPAN" and
                span["startUs"] == provider_span["offsetUs"] and
                span["endExclusiveUs"] == provider_span["endExclusiveUs"] and
                provider["durationMs"] > 0,
                f"descriptor {ordinal} lacks exact provider CUE authority")
        require(span["cueClassification"]["nonZeroOffsetRowIds"] or
                span["cueClassification"]["rawSourceImageRowIds"],
                f"descriptor {ordinal} logical CUE lacks structural evidence")
    else:
        fail(f"descriptor {ordinal} has unknown finalized span kind")

    cue = span["cueClassification"]
    nonzero_ids = cue["nonZeroOffsetRowIds"]
    source_ids = cue["rawSourceImageRowIds"]
    require(cue["providerGroupRowCount"] > 0 and cue["logicalRowCount"] > 0 and
            cue["logicalRowCount"] <= cue["providerGroupRowCount"] and
            len(nonzero_ids) <= cue["logicalRowCount"] and
            cue["logicalRowCount"] + len(source_ids) <= cue["providerGroupRowCount"] and
            set(nonzero_ids).isdisjoint(source_ids) and
            is_sorted_unique_positive(nonzero_ids) and is_sorted_unique_positive(source_ids),
            f"descriptor {ordinal} has invalid CUE classification")

    expected_start = sample_at_or_after(span["startUs"], container["sampleRateHz"])
    expected_end = sample_at_or_after(span["endExclusiveUs"], container["sampleRateHz"])
    expected_source = expected_end - expected_start
    require(expected_source > 0, f"descriptor {ordinal} has empty computed sample span")
    expected_24k = resampled_length(expected_source, container["sampleRateHz"])
    computed_work = expected_work(expected_24k)
    require(span["startSourceSample"] == expected_start and
            span["endSourceSampleExclusive"] == expected_end and
            span["sourceSampleCount"] == expected_source and
            span["exactSampleCount24k"] == expected_24k and
            span["expectedWork"] == computed_work and
            computed_work["mertWindows"] > 0 and computed_work["clampSegments"] > 0,
            f"descriptor {ordinal} finalized time/sample/work arithmetic is invalid")

    source = descriptor["sourceFingerprint"]
    require(isinstance(source.get("fingerprintSpecId"), str) and
            source["fingerprintSpecId"] != "" and source["sizeBytes"] > 0 and
            (source.get("lastModifiedEpochMs") is None or
             source["lastModifiedEpochMs"] >= 0) and
            (source.get("fileKey") is None or source["fileKey"] != "") and
            (source.get("sampledContentSha256") is not None or
             source.get("fullContentSha256") is not None) and
            all(re.fullmatch(r"[0-9a-f]{64}", digest) is not None
                for digest in (source.get("sampledContentSha256"),
                               source.get("fullContentSha256")) if digest is not None),
            f"descriptor {ordinal} has invalid source fingerprint")

def commit_metadata_for_descriptor(descriptor):
    display = descriptor["displayMetadata"]
    filename_source = (f'{display["artist"]} - {display["title"]}'
                       if display["artist"] else display["title"])
    # java.util.regex Pattern uses the ASCII predefined \s class unless Unicode classes are set.
    ascii_whitespace = r"[ \t\n\x0b\f\r]"
    filename_key = unicodedata.normalize(
        "NFC",
        re.sub(ascii_whitespace + r"+", " ",
               re.sub(r"^\d+[.\- \t\n\x0b\f\r]+", "",
                      re.sub(ascii_whitespace + r"*[\(\[].*?[\)\]]", "",
                             filename_source.lower()))).strip(),
    )
    return (
        descriptor["normalizedMetadata"]["metadataKey"], filename_key,
        display["artist"] or None, display["album"] or None, display["title"] or None,
        max(0, descriptor["providerRow"]["durationMs"]),
        descriptor["providerRow"]["providerPhysicalPath"], "phone-v2",
    )

def reconstruct_provisional_descriptor(descriptor):
    span = descriptor["finalizedAudioSpan"]
    if span["kind"] != "WHOLE_FILE":
        require(descriptor.get("provisionalWorkId") is None,
                "CUE work claims ordinary EOS lineage")
        return copy.deepcopy(descriptor)
    require(span["authority"] == "DECODED_END_OF_STREAM",
            "complete whole-file work lacks decoded EOS authority")
    provisional_span = copy.deepcopy(span)
    container = span["container"]
    if (container["durationUsEstimate"] == 0 and
            container["durationEstimateSource"] == "UNAVAILABLE"):
        require(descriptor["providerRow"]["durationMs"] == 0,
                "unknown-duration EOS lineage has noncanonical provider duration")
        provisional_span.update({
            "authority": "PROVISIONAL_END_OF_STREAM",
            "startUs": 0,
            "endExclusiveUs": 0,
            "startSourceSample": 0,
            "endSourceSampleExclusive": 0,
            "sourceSampleCount": 0,
            "exactSampleCount24k": 0,
            "expectedWork": {"mertWindows": 0, "clampSegments": 0},
        })
    else:
        provisional_end_us = container["durationUsEstimate"]
        start_sample = sample_at_or_after(0, container["sampleRateHz"])
        end_sample = sample_at_or_after(provisional_end_us, container["sampleRateHz"])
        source_samples = end_sample - start_sample
        require(source_samples > 0, "decoded EOS lineage reconstructs empty provisional span")
        samples_24k = resampled_length(source_samples, container["sampleRateHz"])
        provisional_span.update({
            "authority": "PROVISIONAL_END_OF_STREAM",
            "startUs": 0,
            "endExclusiveUs": provisional_end_us,
            "startSourceSample": start_sample,
            "endSourceSampleExclusive": end_sample,
            "sourceSampleCount": source_samples,
            "exactSampleCount24k": samples_24k,
            "expectedWork": expected_work(samples_24k),
        })
    provisional = copy.deepcopy(descriptor)
    provisional["workId"] = ""
    provisional.pop("provisionalWorkId", None)
    provisional["finalizedAudioSpan"] = provisional_span
    provisional["stableTrackSpanIdentity"] = stable_span_identity(
        provisional["sourceFingerprint"], provisional_span)
    provisional["workId"] = work_identity(provisional)
    require(descriptor.get("provisionalWorkId") == provisional["workId"],
            "decoded EOS descriptor has invalid provisional work identity")
    return provisional

def validate_ledger_identities(spec):
    embedding = spec["embeddingSpec"]
    text_spec = spec["textRetrievalSpec"]
    require(embedding["preprocessingSpecId"] ==
            "mert-clamp3-audio-v3:torchaudio-hann-v1-width6-rolloff0.99-f32-target-length:"
            "pcm24k-whole-span-zmuv:5s-window:1s-tail-zero-pad:"
            "zero-bookends:segment128-final-overlap:frame-weighted-average:l2" and
            embedding["decoderPolicyId"] ==
            "android-mediacodec-v3:resolved-half-open-us-native-sample-span:"
            "verify-eos-or-enforce-cue-boundary:aligned-polyphase-hq:canonical-24khz-pcm" and
            embedding["inferenceBackendPolicyId"] ==
            "litert-2.1.1-compiled-model-v1:mert-gpu-fp32-strict:"
            "clamp3-audio-gpu-fp32-strict:no-backend-fallback" and
            embedding["outputDimension"] == 768 and
            set(embedding["modelArtifactSha256"]) == {"mert", "clamp3_audio"} and
            all(re.fullmatch(r"[0-9a-f]{64}", value) is not None
                for value in embedding["modelArtifactSha256"].values()),
            "embedding spec does not match the production indexing policy")
    require(re.fullmatch(r"[0-9a-f]{64}", text_spec["textModelSha256"]) is not None and
            text_spec["tokenizerModelSha256"] ==
            "cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865" and
            text_spec["tokenizerPolicyId"] ==
            "sentencepiece-v0.2.1-rev-31646a467d2051eb904e0b45de3a73e91fe1c1e3-"
            "xlm-roberta-model-native-encode-sp-unk0-to-3-else-plus1-"
            "bos0-eos2-pad1-seq128-v1" and
            text_spec["tokenizerRuntimeContractSha256"] ==
            "e3f1abde1d51a6747a252f99b276359f1353b3637e39f85670e8189baa65d8f3" and
            text_spec["outputSpaceId"] == "clamp3-joint-audio-text-l2-f32-v1" and
            text_spec["outputDimension"] == 768 and
            text_spec["inferenceBackendPolicyId"] ==
            "litert-2.1.1-compiled-model-v1:clamp3-text-cpu-strict:"
            "host-text-aggregation-v1:segment128-final-overlap:"
            "token-count-weighted-average:l2:no-backend-fallback",
            "text retrieval spec does not match the production indexing policy")
    require(spec["embeddingSpec"]["specId"] == embedding_spec_identity(spec["embeddingSpec"]),
            "embedding spec content identity is invalid")
    require(spec["textRetrievalSpec"]["specId"] ==
            text_retrieval_spec_identity(spec["textRetrievalSpec"]),
            "text retrieval spec content identity is invalid")
    require(spec["textRetrievalSpec"]["compatibleAudioEmbeddingSpecId"] ==
            spec["embeddingSpec"]["specId"],
            "text retrieval spec is not bound to audio embedding spec")
    has_decoded_ordinary = False
    provisional_tracks = []
    provider_snapshot = spec["providerSnapshot"]
    acquisition = provider_snapshot["acquisition"]
    require(re.fullmatch(r"poweramp-provider-snapshot-v2-sha256:[0-9a-f]{64}",
                         provider_snapshot["libraryGeneration"]) is not None and
            isinstance(acquisition.get("queryUri"), str) and acquisition["queryUri"] != "" and
            acquisition["rowCount"] >= len(spec["tracks"]) and
            acquisition["cursorExhaustedNormally"] is True and
            len(acquisition["requestedColumns"]) > 0 and
            all(isinstance(column, str) and column != ""
                for column in acquisition["requestedColumns"]) and
            len(set(acquisition["requestedColumns"])) == len(acquisition["requestedColumns"]) and
            len(acquisition["returnedColumns"]) > 0 and
            all(isinstance(column, str) and column != ""
                for column in acquisition["returnedColumns"]) and
            len(set(acquisition["returnedColumns"])) == len(acquisition["returnedColumns"]),
            "provider snapshot completion evidence is invalid")
    returned_columns = set(acquisition["returnedColumns"])
    require(all(column in returned_columns or column.rsplit(".", 1)[-1] in returned_columns
                for column in acquisition["requestedColumns"]),
            "provider cursor omitted a requested column")
    runtime = spec["runtimeFingerprint"]
    require(runtime["appVersionCode"] > 0 and runtime["appBuildId"] != "" and
            runtime["decoderRuntimeId"] != "" and runtime["platformFingerprint"] != "",
            "job runtime fingerprint is invalid")
    for ordinal, descriptor in enumerate(spec["tracks"]):
        validate_descriptor_semantics(descriptor, ordinal)
        require(descriptor["providerSnapshotGeneration"] == provider_snapshot["libraryGeneration"] and
                descriptor["powerampFileId"] > 0 and
                descriptor["providerRow"]["powerampFileId"] == descriptor["powerampFileId"] and
                descriptor["normalizedMetadata"]["normalizationSpecId"] ==
                "poweramp-track-normalization-v1",
                "descriptor is not bound to production provider evidence")
        require(descriptor["canonicalPath"] == canonical_path(descriptor["physicalPath"]),
                "descriptor canonical path is invalid")
        expected_stable = stable_span_identity(
            descriptor["sourceFingerprint"], descriptor["finalizedAudioSpan"])
        require(descriptor["stableTrackSpanIdentity"] == expected_stable,
                "descriptor stable track-span identity is invalid")
        require(descriptor["workId"] == work_identity(descriptor),
                "descriptor work identity is invalid")
        if (descriptor["finalizedAudioSpan"]["kind"] == "WHOLE_FILE"):
            has_decoded_ordinary = True
        provisional_tracks.append(reconstruct_provisional_descriptor(descriptor))
    require(spec["specId"] == job_spec_identity(spec), "job spec content identity is invalid")
    if has_decoded_ordinary:
        ancestor = copy.deepcopy(spec)
        ancestor["tracks"] = provisional_tracks
        ancestor.pop("provisionalParentSpecId", None)
        ancestor["specId"] = ""
        ancestor["specId"] = job_spec_identity(ancestor)
        require(spec.get("provisionalParentSpecId") == ancestor["specId"],
                "decoded EOS job has invalid provisional parent identity")
    else:
        require(spec.get("provisionalParentSpecId") is None,
                "CUE-only job unexpectedly claims EOS parent identity")

def activation_binding_identity(manifest):
    digest = hashlib.sha256()
    put_string(digest, "v2-index-activation-binding-v3")
    put_string(digest, manifest["origin"])
    put_string(digest, manifest["jobSpecId"])
    receipt_spec = manifest["receiptEmbeddingSpec"]
    put_string(digest, receipt_spec["specId"])
    put_string(digest, manifest["textRetrievalSpec"]["specId"])
    put_string(digest, receipt_spec["preprocessingSpecId"])
    put_string(digest, receipt_spec["decoderPolicyId"])
    put_string(digest, receipt_spec["inferenceBackendPolicyId"])
    for name in sorted(receipt_spec["modelArtifactSha256"]):
        put_string(digest, name)
        put_string(digest, receipt_spec["modelArtifactSha256"][name])
    put_nullable_string(digest, manifest.get("baseGenerationId"))
    digest.update(b"\x01" if manifest["rebuildDerivedIndexes"] else b"\x00")
    put_string(digest, manifest["graphPolicy"])
    put_string(digest, manifest["databaseContentSha256"])
    put_string(digest, manifest["orderedTrackSetSha256"])
    put_string(digest, manifest["embeddingSha256"])
    put_string(digest, manifest["stableTrackUidCoverage"]["mappingSha256"])
    put_string(digest, manifest["embeddingCoverage"]["mappingSha256"])
    put_nullable_string(digest, manifest.get("graph", {}).get("sha256")
                        if manifest.get("graph") is not None else None)
    return "activation-binding-v3-" + digest.hexdigest()

def generation_identity(manifest):
    digest = hashlib.sha256()
    put_string(digest, "v2-index-generation-manifest-v3")
    put_string(digest, manifest["origin"])
    put_string(digest, manifest["activationBindingId"])
    put_string(digest, manifest["jobId"])
    put_string(digest, manifest["jobSpecId"])
    put_string(digest, manifest["receiptEmbeddingSpec"]["specId"])
    put_string(digest, manifest["textRetrievalSpec"]["specId"])
    put_nullable_string(digest, manifest.get("baseGenerationId"))
    digest.update(b"\x01" if manifest["rebuildDerivedIndexes"] else b"\x00")
    put_string(digest, manifest["graphPolicy"])
    put_string(digest, manifest["databaseRelativePath"])
    put_long(digest, manifest["databaseByteLength"])
    put_string(digest, manifest["databaseSha256"])
    put_string(digest, manifest["databaseContentSha256"])
    put_string(digest, manifest["orderedTrackSetSha256"])
    stable = manifest["stableTrackUidCoverage"]
    for key in ("coveredTrackCount", "uncoveredTrackCount", "uniqueStableTrackSpanCount",
                "fullContentIdentityCount", "sampledContentIdentityCount"):
        put_int(digest, stable[key])
    put_string(digest, stable["mappingSha256"])
    coverage = manifest["embeddingCoverage"]
    put_int(digest, coverage["totalTrackCount"])
    put_int(digest, coverage["receiptBoundTrackCount"])
    counts = coverage["receiptSpecTrackCounts"]
    put_int(digest, len(counts))
    for spec_id in sorted(counts):
        put_string(digest, spec_id)
        put_int(digest, counts[spec_id])
    compatibility = coverage.get("compatibilityBase")
    digest.update(b"\x00" if compatibility is None else b"\x01")
    if compatibility is not None:
        put_string(digest, compatibility["provenancePolicyId"])
        put_int(digest, compatibility["trackCount"])
        put_string(digest, compatibility["orderedContentSha256"])
    put_string(digest, coverage["mappingSha256"])
    put_int(digest, manifest["trackCount"])
    put_int(digest, manifest["embeddingDimension"])
    put_string(digest, manifest["embeddingRelativePath"])
    put_long(digest, manifest["embeddingByteLength"])
    put_string(digest, manifest["embeddingSha256"])
    graph = manifest.get("graph")
    digest.update(b"\x00" if graph is None else b"\x01")
    if graph is not None:
        put_string(digest, graph["relativePath"])
        put_long(digest, graph["byteLength"])
        put_string(digest, graph["sha256"])
        put_int(digest, graph["nodeCount"])
        put_int(digest, graph["neighborsPerNode"])
        put_string(digest, graph["orderedTrackSetSha256"])
    return "index-generation-v2-" + digest.hexdigest()

def bootstrap_spec_identity(manifest):
    digest = hashlib.sha256()
    put_string(digest, "v2-bootstrap-compatibility-spec-v1")
    put_string(digest, manifest["receiptEmbeddingSpec"]["specId"])
    put_string(digest, manifest["textRetrievalSpec"]["specId"])
    return "bootstrap-spec-v1-" + digest.hexdigest()

def maintenance_spec_identity(manifest):
    digest = hashlib.sha256()
    put_string(digest, "v2-library-maintenance-spec-v1")
    put_string(digest, manifest["baseGenerationId"])
    put_string(digest, manifest["receiptEmbeddingSpec"]["specId"])
    put_string(digest, manifest["textRetrievalSpec"]["specId"])
    return "maintenance-spec-v1-" + digest.hexdigest()

def validate_manifest_provenance(manifest, label):
    generation_pattern = r"index-generation-v2-[0-9a-f]{64}"
    job_pattern = r"[A-Za-z0-9._-]{1,128}"
    job_spec_pattern = r"job-spec-v5-[0-9a-f]{64}"
    activation_pattern = r"activation-binding-v3-[0-9a-f]{64}"
    base_generation = manifest.get("baseGenerationId")
    require(re.fullmatch(activation_pattern, manifest["activationBindingId"]) is not None and
            isinstance(manifest.get("createdAtEpochMs"), int) and
            not isinstance(manifest["createdAtEpochMs"], bool) and
            manifest["createdAtEpochMs"] >= 0 and
            (base_generation is None or
             re.fullmatch(generation_pattern, base_generation) is not None),
            f"{label} manifest has invalid common provenance")
    origin = manifest.get("origin")
    graph_present = manifest.get("graph") is not None
    if origin == "INDEXING_JOB":
        require(re.fullmatch(job_pattern, manifest.get("jobId", "")) is not None and
                re.fullmatch(job_spec_pattern, manifest.get("jobSpecId", "")) is not None and
                manifest.get("graphPolicy") in {"ABSENT", "EXPLICIT_REBUILD"} and
                ((manifest["graphPolicy"] == "ABSENT") == (not graph_present)),
                f"{label} indexing manifest has invalid job provenance")
    elif origin == "BOOTSTRAP_COMPATIBILITY":
        require(manifest.get("jobId") == "bootstrap-compatibility-import-v1" and
                manifest.get("jobSpecId") == bootstrap_spec_identity(manifest) and
                base_generation is None and manifest.get("rebuildDerivedIndexes") is False and
                manifest.get("graphPolicy") in {"ABSENT", "VALIDATED_COMPATIBILITY_IMPORT"} and
                ((manifest["graphPolicy"] == "ABSENT") == (not graph_present)) and
                manifest["createdAtEpochMs"] == 0,
                f"{label} bootstrap manifest has invalid compatibility provenance")
    elif origin == "LIBRARY_MAINTENANCE":
        require(manifest.get("jobId") == "library-maintenance-v1" and
                base_generation is not None and
                manifest.get("jobSpecId") == maintenance_spec_identity(manifest) and
                manifest.get("rebuildDerivedIndexes") is False and
                manifest.get("graphPolicy") in {"ABSENT", "BASE_BOUND_DELETION_REPAIR"} and
                ((manifest["graphPolicy"] == "ABSENT") == (not graph_present)) and
                manifest["createdAtEpochMs"] == 0,
                f"{label} maintenance manifest has invalid base-bound provenance")
    else:
        fail(f"{label} manifest has unknown origin")

def open_database(relative):
    path = root / relative
    uri = "file:" + str(path.resolve()) + "?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    return connection

def scalar(connection, sql, parameters=()):
    row = connection.execute(sql, parameters).fetchone()
    require(row is not None, f"query returned no row: {sql}")
    return row[0]

def require_table(connection, table):
    require(scalar(connection,
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?", (table,)) == 1,
        f"database is missing {table}")

def derive_generation_coverages(connection, manifest, label):
    embedding_count = scalar(connection, "SELECT COUNT(*) FROM embeddings_clamp3")
    stable_digest = hashlib.sha256()
    put_string(stable_digest, "v2-stable-track-uid-coverage-v1")
    put_int(stable_digest, embedding_count)
    covered = uncovered = full = sampled = 0
    unique_stable_ids = set()
    previous = -1
    rows = 0
    for row in connection.execute("""
        SELECT e.track_id,r.stable_track_span_id,r.stable_identity_spec_id,
               r.stable_identity_strength
        FROM embeddings_clamp3 e
        LEFT JOIN v2_embedding_commit_receipts_v4 r
          ON r.track_id=e.track_id AND r.receipt_schema_version=4
        ORDER BY e.track_id
    """):
        track_id = row[0]
        require(track_id > previous, f"{label} stable coverage rows are unordered")
        put_long(stable_digest, track_id)
        if row[1] is None:
            stable_digest.update(b"\x00")
            uncovered += 1
        else:
            stable_id, identity_spec, strength = row[1], row[2], row[3]
            require(re.fullmatch(r"stable-track-span-v1-[0-9a-f]{64}", stable_id) is not None,
                    f"{label} has invalid stable UID receipt")
            require(identity_spec ==
                    "stable-track-span-v1:content-sha256:native-half-open-sample-span",
                    f"{label} has unsupported stable UID receipt spec")
            require(strength in {"FULL_CONTENT_SHA256", "VERSIONED_SAMPLED_CONTENT_SHA256"},
                    f"{label} has invalid stable UID strength")
            stable_digest.update(b"\x01")
            put_string(stable_digest, stable_id)
            put_string(stable_digest, identity_spec)
            put_string(stable_digest, strength)
            unique_stable_ids.add(stable_id)
            covered += 1
            full += int(strength == "FULL_CONTENT_SHA256")
            sampled += int(strength == "VERSIONED_SAMPLED_CONTENT_SHA256")
        previous = track_id
        rows += 1
    require(rows == embedding_count, f"{label} stable coverage scan was incomplete")
    stable_binding = {
        "coveredTrackCount": covered,
        "uncoveredTrackCount": uncovered,
        "uniqueStableTrackSpanCount": len(unique_stable_ids),
        "fullContentIdentityCount": full,
        "sampledContentIdentityCount": sampled,
        "mappingSha256": stable_digest.hexdigest(),
    }

    mapping_digest = hashlib.sha256()
    put_string(mapping_digest, "v2-embedding-spec-coverage-v1")
    put_int(mapping_digest, embedding_count)
    compatibility_digest = hashlib.sha256()
    put_string(compatibility_digest, "v2-unreceipted-compatibility-content-v1")
    receipt_counts = {}
    compatibility_count = 0
    previous = -1
    rows = 0
    expected_spec = manifest["receiptEmbeddingSpec"]["specId"]
    for row in connection.execute("""
        SELECT e.track_id,e.embedding,r.receipt_schema_version,r.embedding_spec_id
        FROM embeddings_clamp3 e
        LEFT JOIN v2_embedding_commit_receipts_v4 r ON r.track_id=e.track_id
        ORDER BY e.track_id
    """):
        track_id, blob, receipt_schema, receipt_spec = row
        require(track_id > 0 and track_id > previous,
                f"{label} embedding coverage rows are duplicated or unordered")
        inspect_vector_blob(blob, f"{label} coverage embedding {track_id}")
        put_long(mapping_digest, track_id)
        if receipt_schema is None:
            require(receipt_spec is None, f"{label} has a partial embedding receipt")
            mapping_digest.update(b"\x00")
            put_long(compatibility_digest, track_id)
            put_int(compatibility_digest, len(blob))
            compatibility_digest.update(blob)
            compatibility_count += 1
        else:
            require(receipt_schema == 4 and
                    re.fullmatch(r"embedding-spec-v2-[0-9a-f]{64}", receipt_spec or "") is not None and
                    receipt_spec == expected_spec,
                    f"{label} has a conflicting embedding receipt spec")
            mapping_digest.update(b"\x01")
            put_string(mapping_digest, receipt_spec)
            receipt_counts[receipt_spec] = receipt_counts.get(receipt_spec, 0) + 1
        previous = track_id
        rows += 1
    require(rows == embedding_count, f"{label} embedding coverage scan was incomplete")
    put_int(mapping_digest, len(receipt_counts))
    for spec_id in sorted(receipt_counts):
        put_string(mapping_digest, spec_id)
        put_int(mapping_digest, receipt_counts[spec_id])
    put_int(mapping_digest, compatibility_count)
    compatibility_binding = None
    if compatibility_count:
        put_int(compatibility_digest, compatibility_count)
        compatibility_binding = {
            "provenancePolicyId":
                "unreceipted-clamp3-compatibility-base-v1:unknown-model-and-preprocessing:no-v2-claim",
            "trackCount": compatibility_count,
            "orderedContentSha256": compatibility_digest.hexdigest(),
        }
        put_string(mapping_digest, compatibility_binding["provenancePolicyId"])
        put_string(mapping_digest, compatibility_binding["orderedContentSha256"])
    embedding_binding = {
        "totalTrackCount": embedding_count,
        "receiptBoundTrackCount": sum(receipt_counts.values()),
        "receiptSpecTrackCounts": dict(sorted(receipt_counts.items())),
        "mappingSha256": mapping_digest.hexdigest(),
    }
    if compatibility_binding is not None:
        embedding_binding["compatibilityBase"] = compatibility_binding

    require(manifest["stableTrackUidCoverage"] == stable_binding,
            f"{label} stable UID coverage binding is not derived from database rows")
    declared_embedding = dict(manifest["embeddingCoverage"])
    if declared_embedding.get("compatibilityBase") is None:
        declared_embedding.pop("compatibilityBase", None)
    require(declared_embedding == embedding_binding,
            f"{label} embedding coverage binding is not derived from database rows")
    return stable_binding, embedding_binding

def derive_base_compatibility_after_supersessions(connection, excluded_track_ids):
    digest = hashlib.sha256()
    put_string(digest, "v2-unreceipted-compatibility-content-v1")
    count = 0
    for track_id, blob, receipt_track_id in connection.execute("""
        SELECT e.track_id,e.embedding,r.track_id
        FROM embeddings_clamp3 e
        LEFT JOIN v2_embedding_commit_receipts_v4 r ON r.track_id=e.track_id
        ORDER BY e.track_id
    """):
        if receipt_track_id is not None or track_id in excluded_track_ids:
            continue
        inspect_vector_blob(blob, f"base compatibility embedding {track_id}")
        put_long(digest, track_id)
        put_int(digest, len(blob))
        digest.update(blob)
        count += 1
    if count == 0:
        return None
    put_int(digest, count)
    return {
        "provenancePolicyId":
            "unreceipted-clamp3-compatibility-base-v1:unknown-model-and-preprocessing:no-v2-claim",
        "trackCount": count,
        "orderedContentSha256": digest.hexdigest(),
    }

def inspect_assets():
    rows = {}
    for line in (root / "raw/asset-sizes-sha256.tsv").read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        require(len(fields) == 4, "asset hash evidence has a malformed row")
        scope, relative, size_text, digest = fields
        require(scope in {"active", "base"}, "asset hash evidence has an invalid scope")
        require(relative in {"manifest.json", "library.db", "clamp3.emb", "graph.bin"},
                "asset hash evidence has an invalid path")
        require(size_text.isdigit() and int(size_text) > 0, "asset hash evidence has invalid size")
        require(len(digest) == 64 and digest == digest.lower() and
                all(character in "0123456789abcdef" for character in digest),
                "asset hash evidence has invalid SHA")
        key = (scope, relative)
        require(key not in rows, "asset hash evidence repeats an asset")
        rows[key] = (int(size_text), digest)
    require(set(rows) == {(scope, relative) for scope in ("active", "base")
                         for relative in ("manifest.json", "library.db", "clamp3.emb", "graph.bin")},
            "asset hash evidence does not cover both immutable generations")
    for scope in ("active", "base"):
        manifest = load_json(f"{scope}/manifest.json")
        expected = {
            "library.db": (manifest["databaseByteLength"], manifest["databaseSha256"]),
            "clamp3.emb": (manifest["embeddingByteLength"], manifest["embeddingSha256"]),
            "graph.bin": (manifest["graph"]["byteLength"], manifest["graph"]["sha256"]),
        }
        for relative, binding in expected.items():
            require(rows[(scope, relative)] == binding,
                    f"{scope} {relative} remote bytes do not match its manifest")
        manifest_path = root / scope / "manifest.json"
        require(rows[(scope, "manifest.json")] ==
                (manifest_path.stat().st_size, sha256_file(manifest_path)),
                f"{scope} manifest changed during capture")
        database_path = root / scope / "library.db"
        require(rows[(scope, "library.db")] ==
                (database_path.stat().st_size, sha256_file(database_path)),
                f"{scope} database changed during capture")
    for relative in ("clamp3.emb", "graph.bin"):
        path = root / "active" / relative
        require(rows[("active", relative)] == (path.stat().st_size, sha256_file(path)),
                f"active {relative} changed during capture")
    return rows

def inspect_vector_blob(blob, label):
    require(len(blob) == 3072, f"{label} has invalid embedding length")
    values = array.array("f")
    values.frombytes(blob)
    if sys.byteorder != "little":
        values.byteswap()
    norm = 0.0
    for value in values:
        require(math.isfinite(value), f"{label} contains a non-finite value")
        norm += float(value) * float(value)
    require(abs(math.sqrt(norm) - 1.0) <= 0.001,
            f"{label} is not L2-normalized")

def inspect_database(connection, manifest, label):
    require(scalar(connection, "PRAGMA integrity_check") == "ok", f"{label} SQLite integrity_check failed")
    require(list(connection.execute("PRAGMA foreign_key_check")) == [],
            f"{label} SQLite foreign-key check failed")
    for table in ("tracks", "embeddings_clamp3", "v2_embedding_commit_receipts_v4",
                  "v2_index_generation_guard_v2"):
        require_table(connection, table)
    derive_generation_coverages(connection, manifest, label)
    track_count = scalar(connection, "SELECT COUNT(*) FROM tracks")
    embedding_count = scalar(connection, "SELECT COUNT(*) FROM embeddings_clamp3")
    require(track_count == embedding_count == manifest["trackCount"],
            f"{label} track/embedding count does not match manifest")
    require(scalar(connection,
        "SELECT COUNT(*) FROM embeddings_clamp3 WHERE embedding IS NULL OR length(embedding) != 3072") == 0,
        f"{label} contains invalid embedding blobs")
    require(scalar(connection, """
        SELECT COUNT(*) FROM tracks t LEFT JOIN embeddings_clamp3 e ON e.track_id=t.id
        WHERE e.track_id IS NULL
    """) == 0, f"{label} has tracks without embeddings")
    require(scalar(connection, """
        SELECT COUNT(*) FROM embeddings_clamp3 e LEFT JOIN tracks t ON t.id=e.track_id
        WHERE t.id IS NULL
    """) == 0, f"{label} has orphaned embeddings")
    receipt_count = scalar(connection, "SELECT COUNT(*) FROM v2_embedding_commit_receipts_v4")
    coverage = manifest["embeddingCoverage"]
    stable = manifest["stableTrackUidCoverage"]
    require(coverage["totalTrackCount"] == track_count,
            f"{label} embedding coverage total does not match database")
    require(receipt_count == coverage["receiptBoundTrackCount"] == stable["coveredTrackCount"],
            f"{label} receipt coverage count does not match manifest")
    require(scalar(connection,
        "SELECT COUNT(DISTINCT track_id) FROM v2_embedding_commit_receipts_v4") == receipt_count,
        f"{label} has more than one receipt bound to a track")
    require(track_count - receipt_count == stable["uncoveredTrackCount"],
            f"{label} stable uncovered count does not match manifest")
    compatibility = coverage.get("compatibilityBase")
    compatibility_count = compatibility["trackCount"] if compatibility else 0
    if compatibility is not None:
        require(compatibility["provenancePolicyId"] ==
                "unreceipted-clamp3-compatibility-base-v1:unknown-model-and-preprocessing:no-v2-claim",
                f"{label} compatibility provenance policy is invalid")
    require(track_count - receipt_count == compatibility_count,
            f"{label} compatibility count does not match database")
    require(scalar(connection, """
        SELECT COUNT(*) FROM v2_embedding_commit_receipts_v4 r
        JOIN embeddings_clamp3 e ON e.track_id=r.track_id
        WHERE r.receipt_schema_version != 4 OR r.embedding_byte_length != 3072 OR
              length(e.embedding) != r.embedding_byte_length
    """) == 0, f"{label} has invalid receipt schema or embedding length")
    require(scalar(connection, """
        SELECT COUNT(*) FROM v2_embedding_commit_receipts_v4 r
        LEFT JOIN embeddings_clamp3 e ON e.track_id=r.track_id WHERE e.track_id IS NULL
    """) == 0, f"{label} has orphaned receipts")
    spec_counts = {row[0]: row[1] for row in connection.execute("""
        SELECT embedding_spec_id, COUNT(*) FROM v2_embedding_commit_receipts_v4
        GROUP BY embedding_spec_id ORDER BY embedding_spec_id
    """)}
    require(spec_counts == coverage["receiptSpecTrackCounts"],
            f"{label} receipt spec coverage does not match manifest")
    strength_counts = {row[0]: row[1] for row in connection.execute("""
        SELECT stable_identity_strength, COUNT(*) FROM v2_embedding_commit_receipts_v4
        GROUP BY stable_identity_strength
    """)}
    require(strength_counts.get("FULL_CONTENT_SHA256", 0) == stable["fullContentIdentityCount"],
            f"{label} full-content identity count does not match manifest")
    require(strength_counts.get("VERSIONED_SAMPLED_CONTENT_SHA256", 0) ==
            stable["sampledContentIdentityCount"],
            f"{label} sampled identity count does not match manifest")
    require(scalar(connection,
        "SELECT COUNT(DISTINCT stable_track_span_id) FROM v2_embedding_commit_receipts_v4") ==
        stable["uniqueStableTrackSpanCount"],
        f"{label} unique stable-span count does not match manifest")
    guard = connection.execute("""
        SELECT receipt_schema_version,is_valid,activation_binding_id,job_spec_id,
               receipt_embedding_spec_id,text_retrieval_spec_id,embedding_coverage_sha256,
               compatibility_base_content_sha256,database_content_sha256,
               ordered_track_set_sha256,stable_uid_mapping_sha256,embedding_sha256,graph_sha256
        FROM v2_index_generation_guard_v2 WHERE singleton=1
    """).fetchall()
    require(len(guard) == 1, f"{label} activation guard singleton is absent")
    compatibility_sha = compatibility["orderedContentSha256"] if compatibility else None
    expected_guard = (
        3, 1, manifest["activationBindingId"], manifest["jobSpecId"],
        manifest["receiptEmbeddingSpec"]["specId"], manifest["textRetrievalSpec"]["specId"],
        coverage["mappingSha256"], compatibility_sha, manifest["databaseContentSha256"],
        manifest["orderedTrackSetSha256"], stable["mappingSha256"],
        manifest["embeddingSha256"], manifest["graph"]["sha256"],
    )
    require(tuple(guard[0]) == expected_guard, f"{label} activation guard does not match manifest")
    return {"trackCount": track_count, "receiptCount": receipt_count,
            "compatibilityCount": compatibility_count}

def inspect_active_binary_assets(manifest, connection):
    pemb_path = root / "active/clamp3.emb"
    graph_path = root / "active/graph.bin"
    ids_digest = hashlib.sha256()
    put_string(ids_digest, "v2-ordered-track-set-v1")
    put_int(ids_digest, manifest["trackCount"])
    content_digest = hashlib.sha256()
    put_string(content_digest, "v2-ordered-clamp3-content-v1")
    put_int(content_digest, manifest["trackCount"])
    put_int(content_digest, manifest["embeddingDimension"])
    previous = 0
    rows = 0
    for track_id, blob in connection.execute(
            "SELECT track_id,embedding FROM embeddings_clamp3 ORDER BY track_id"):
        require(track_id > previous, "active database embedding IDs are not strictly increasing")
        inspect_vector_blob(blob, f"database embedding {track_id}")
        put_long(ids_digest, track_id)
        put_long(content_digest, track_id)
        put_int(content_digest, len(blob))
        content_digest.update(blob)
        previous = track_id
        rows += 1
    require(rows == manifest["trackCount"], "active database embedding scan was incomplete")
    require(ids_digest.hexdigest() == manifest["orderedTrackSetSha256"],
            "active database ordered track digest does not match manifest")
    require(content_digest.hexdigest() == manifest["databaseContentSha256"],
            "active database embedding content digest does not match manifest")

    with pemb_path.open("rb") as stream:
        header = stream.read(16)
        require(len(header) == 16, "PEMB header is short")
        magic, version, count, dimension = struct.unpack("<IIII", header)
        require((magic, version, count, dimension) ==
                (0x424D4550, 1, manifest["trackCount"], manifest["embeddingDimension"]),
                "PEMB header does not match manifest")
        expected_length = 16 + count * 8 + count * dimension * 4
        require(pemb_path.stat().st_size == expected_length == manifest["embeddingByteLength"],
                "PEMB length does not match its shape")
        ids_raw = stream.read(count * 8)
        require(len(ids_raw) == count * 8, "PEMB ID table is short")
        pemb_ids = struct.unpack(f"<{count}Q", ids_raw)
        require(all(value > 0 for value in pemb_ids) and
                all(left < right for left, right in zip(pemb_ids, pemb_ids[1:])),
                "PEMB IDs are not strictly increasing")
        pemb_ids_digest = hashlib.sha256()
        put_string(pemb_ids_digest, "v2-ordered-track-set-v1")
        put_int(pemb_ids_digest, count)
        pemb_content_digest = hashlib.sha256()
        put_string(pemb_content_digest, "v2-ordered-clamp3-content-v1")
        put_int(pemb_content_digest, count)
        put_int(pemb_content_digest, dimension)
        for track_id in pemb_ids:
            put_long(pemb_ids_digest, track_id)
        for track_id in pemb_ids:
            blob = stream.read(dimension * 4)
            require(len(blob) == dimension * 4, "PEMB embedding body is short")
            inspect_vector_blob(blob, f"PEMB embedding {track_id}")
            put_long(pemb_content_digest, track_id)
            put_int(pemb_content_digest, len(blob))
            pemb_content_digest.update(blob)
        require(stream.read(1) == b"", "PEMB has trailing bytes")
        require(pemb_ids_digest.hexdigest() == manifest["orderedTrackSetSha256"],
                "PEMB ordered track digest does not match manifest")
        require(pemb_content_digest.hexdigest() == manifest["databaseContentSha256"],
                "PEMB content differs from the activated database")

    graph = manifest["graph"]
    with graph_path.open("rb") as stream:
        header = stream.read(8)
        require(len(header) == 8, "graph header is short")
        count, neighbors = struct.unpack("<II", header)
        require((count, neighbors) == (graph["nodeCount"], graph["neighborsPerNode"]),
                "graph header does not match manifest")
        expected_length = 8 + count * 8 + count * neighbors * 8
        require(graph_path.stat().st_size == expected_length == graph["byteLength"],
                "graph length does not match its shape")
        ids_raw = stream.read(count * 8)
        require(len(ids_raw) == count * 8, "graph ID table is short")
        graph_ids = struct.unpack(f"<{count}Q", ids_raw)
        require(all(value > 0 for value in graph_ids) and
                all(left < right for left, right in zip(graph_ids, graph_ids[1:])),
                "graph IDs are not strictly increasing")
        graph_digest = hashlib.sha256()
        put_string(graph_digest, "v2-ordered-track-set-v1")
        put_int(graph_digest, count)
        for track_id in graph_ids:
            put_long(graph_digest, track_id)
        require(graph_digest.hexdigest() == graph["orderedTrackSetSha256"] ==
                manifest["orderedTrackSetSha256"],
                "graph ordered track digest does not match generation")
        for node in range(count):
            row_sum = 0.0
            for neighbor in range(neighbors):
                raw = stream.read(8)
                require(len(raw) == 8, f"graph row {node} is short")
                neighbor_index, weight = struct.unpack("<If", raw)
                require(neighbor_index < count, f"graph row {node} has an invalid neighbor")
                require(math.isfinite(weight) and weight >= 0.0,
                        f"graph row {node} has an invalid weight")
                row_sum += weight
            require(abs(row_sum - 1.0) <= 0.005,
                    f"graph row {node} weights do not sum to one")
        require(stream.read(1) == b"", "graph has trailing bytes")

def canonical_selection(row):
    return (row["powerampFileId"], row["providerPhysicalPath"],
            row["offsetMs"], row["durationMs"], row.get("cueSourceImageFolderId"))

try:
    assets = inspect_assets()
    ledger_envelope = load_json("raw/ledger.json")
    preflight_envelope = load_json("raw/preflight-intent.json")
    pointer = load_json("raw/active-generation.json")
    quiescence = load_json("raw/quiescence.json")
    lease = load_json("raw/executor-lease.json")
    authorization_namespace = load_json("raw/authorization-namespace.json")
    active_manifest = load_json("active/manifest.json")
    base_manifest = load_json("base/manifest.json")
    for label, manifest in (("active", active_manifest), ("base", base_manifest)):
        require(manifest.get("schemaVersion") == 3,
                f"{label} manifest schema is invalid")
        validate_manifest_provenance(manifest, label)
        require(manifest["receiptEmbeddingSpec"]["specId"] ==
                embedding_spec_identity(manifest["receiptEmbeddingSpec"]),
                f"{label} manifest embedding spec content identity is invalid")
        require(manifest["textRetrievalSpec"]["specId"] ==
                text_retrieval_spec_identity(manifest["textRetrievalSpec"]),
                f"{label} manifest text spec content identity is invalid")
        require(manifest["textRetrievalSpec"]["compatibleAudioEmbeddingSpecId"] ==
                manifest["receiptEmbeddingSpec"]["specId"],
                f"{label} manifest text/audio model binding is invalid")
        require(manifest["activationBindingId"] == activation_binding_identity(manifest),
                f"{label} manifest activation binding content identity is invalid")
        require(manifest["generationId"] == generation_identity(manifest),
                f"{label} manifest generation content identity is invalid")

    require(ledger_envelope.get("format") == "poweramp-start-radio-v2-indexing-ledger" and
            ledger_envelope.get("schemaVersion") == 5, "ledger envelope schema is invalid")
    ledger = ledger_envelope["ledger"]
    spec = ledger["jobSpec"]
    require(ledger.get("schemaVersion") == 5 and
            isinstance(ledger.get("revision"), int) and ledger["revision"] >= 0 and
            isinstance(ledger.get("updatedAtEpochMs"), int) and
            ledger["updatedAtEpochMs"] >= spec["createdAtEpochMs"] >= 0,
            "ledger schema, revision, or lifetime is invalid")
    require(spec["jobId"] == expected_job, "ledger belongs to another job")
    require(spec.get("baseGenerationId") == expected_base, "ledger base generation changed")
    require(spec.get("rebuildDerivedIndexes") is True,
            "completion verifier is scoped to rebuildDerivedIndexes=true jobs")
    validate_ledger_identities(spec)
    require(ledger["state"] == "COMPLETE" and ledger.get("recoveryPhase") is None,
            "ledger is not terminal COMPLETE")
    require(ledger.get("activationEvidence") is not None, "ledger has no activation evidence")
    descriptors = {track["workId"]: track for track in spec["tracks"]}
    track_ledgers = {track["workId"]: track for track in ledger["tracks"]}
    require(len(descriptors) == len(spec["tracks"]) == expected_executable,
            "ledger executable descriptor count differs from expectation")
    require([track["ordinal"] for track in spec["tracks"]] == list(range(expected_executable)) and
            len({track["powerampFileId"] for track in spec["tracks"]}) == expected_executable,
            "ledger descriptor ordinals or Poweramp IDs are not unique and canonical")
    require(len(track_ledgers) == len(ledger["tracks"]) == expected_executable and
            set(track_ledgers) == set(descriptors) and
            [track["workId"] for track in ledger["tracks"]] ==
            [track["workId"] for track in spec["tracks"]],
            "ledger track order or identities differ from immutable spec")
    embedding_spec = spec["embeddingSpec"]["specId"]
    for work_id, track in track_ledgers.items():
        descriptor = descriptors[work_id]
        require(spec["createdAtEpochMs"] <= track["updatedAtEpochMs"] <=
                ledger["updatedAtEpochMs"],
                f"track {work_id} timestamp is outside job lifetime")
        require(track["state"] == "COMMITTED" and track["checkpoint"] == "COMMITTED",
                f"track {work_id} is not committed")
        require(track.get("currentAttemptNumber") is None and
                track.get("activeFailureId") is None and track.get("stageProgress") is None,
                f"track {work_id} retains active execution state")
        artifacts = {artifact["kind"]: artifact for artifact in track["verifiedArtifacts"]}
        require(len(artifacts) == len(track["verifiedArtifacts"]) == 3 and
                set(artifacts) == {"MERT_FEATURES", "CLAMP_VECTOR", "DATABASE_COMMIT"},
                f"track {work_id} does not have exactly three terminal artifacts")
        expected_units = {
            "MERT_FEATURES": descriptor["finalizedAudioSpan"]["expectedWork"]["mertWindows"],
            "CLAMP_VECTOR": descriptor["finalizedAudioSpan"]["expectedWork"]["clampSegments"],
            "DATABASE_COMMIT": 1,
        }
        for kind, artifact in artifacts.items():
            require(artifact["embeddingSpecId"] == embedding_spec and
                    artifact["sourceFingerprint"] == descriptor["sourceFingerprint"] and
                    isinstance(artifact.get("storageKey"), str) and artifact["storageKey"] != "" and
                    re.fullmatch(r"[0-9a-f]{64}", artifact.get("sha256", "")) is not None and
                    artifact["plannedUnits"] == expected_units[kind] and
                    artifact["completedUnits"] == artifact["plannedUnits"] > 0 and
                    spec["createdAtEpochMs"] <= artifact["verifiedAtEpochMs"] <=
                    track["updatedAtEpochMs"],
                    f"track {work_id} has an unbound or incomplete artifact")
        expected_windows = descriptor["finalizedAudioSpan"]["expectedWork"]["mertWindows"]
        require(artifacts["MERT_FEATURES"]["plannedUnits"] == expected_windows and
                artifacts["MERT_FEATURES"]["byteLength"] == expected_windows * 3072,
                f"track {work_id} MERT artifact shape is incorrect")
        require(artifacts["CLAMP_VECTOR"]["byteLength"] == 3072 and
                artifacts["DATABASE_COMMIT"]["byteLength"] == 3072 and
                artifacts["CLAMP_VECTOR"]["sha256"] == artifacts["DATABASE_COMMIT"]["sha256"],
                f"track {work_id} CLaMP and database artifacts disagree")
        boundary = artifacts["MERT_FEATURES"].get("executionBoundary")
        span = descriptor["finalizedAudioSpan"]
        require(isinstance(boundary, dict) and
                boundary.get("requirement") == span["executionBoundaryRequirement"] and
                boundary.get("observedStartSourceSample") == span["startSourceSample"] and
                boundary.get("observedEndSourceSampleExclusive") ==
                span["endSourceSampleExclusive"] and
                boundary.get("observedSourceSampleCount") == span["sourceSampleCount"] and
                boundary.get("exactSampleCount24k") == span["exactSampleCount24k"],
                f"track {work_id} MERT decoder boundary differs from immutable span")
        if span["executionBoundaryRequirement"] == "VERIFY_END_OF_STREAM_AND_RECONCILE":
            require(boundary.get("endOfStreamReached") is True and
                    boundary.get("providerBoundaryEnforced") is False,
                    f"track {work_id} lacks verified decoded EOS")
        else:
            require(boundary.get("providerBoundaryEnforced") is True,
                    f"track {work_id} lacks enforced provider CUE boundary")
        require(artifacts["CLAMP_VECTOR"].get("executionBoundary") is None and
                artifacts["DATABASE_COMMIT"].get("executionBoundary") is None,
                f"track {work_id} non-MERT artifact carries decoder boundary evidence")

    require(preflight_envelope.get("format") == "poweramp-start-radio-v2-indexing-preflight-intent" and
            preflight_envelope.get("schemaVersion") == 2, "preflight envelope schema is invalid")
    intent = preflight_envelope["intent"]
    require(intent.get("schemaVersion") == 2 and intent["jobId"] == expected_job,
            "preflight belongs to another job or schema")
    require(intent["state"] == "MATERIALIZED" and
            intent.get("failureCode") is None and intent.get("failureMessage") is None and
            intent.get("resolvedSpecId") is None and
            intent.get("rebuildDerivedIndexes") is True and
            intent.get("executionProfile") in {"FULL", "BALANCED", "BACKGROUND"} and
            ledger.get("executionProfile") in {"FULL", "BALANCED", "BACKGROUND"} and
            isinstance(intent.get("revision"), int) and intent["revision"] >= 0 and
            intent.get("createdAtEpochMs") == spec["createdAtEpochMs"] and
            intent["updatedAtEpochMs"] >= intent["createdAtEpochMs"] and
            intent.get("progress", {}).get("phase") == "COMPLETE" and
            isinstance(intent.get("progress", {}).get("message"), str) and
            0 < len(intent["progress"]["message"].strip()) <= 512 and
            intent["progress"].get("completedUnits") is None and
            intent["progress"].get("totalUnits") is None,
            "preflight MATERIALIZED state or lifetime is invalid")
    require(intent.get("baseGenerationId") == expected_base, "preflight base generation changed")
    preflight_spec_id = spec.get("provisionalParentSpecId") or spec["specId"]
    require(intent.get("materializedSpecId") == preflight_spec_id,
            "preflight materialization is not bound to reconstructed job lineage")
    planned = intent["planned"]
    rejected = intent["rejected"]
    selected = intent["selected"]
    for row in selected:
        require(row["powerampFileId"] > 0 and
                isinstance(row["providerPhysicalPath"], str) and
                row["providerPhysicalPath"].startswith("/") and
                0 <= row["durationMs"] <= 2_147_483_647 and row["offsetMs"] >= 0,
                "preflight contains invalid selected occurrence evidence")
    require(len(planned) == expected_executable, "preflight planned count differs from executable count")
    require(len(selected) == len(planned) + len(rejected),
            "preflight selected count is not planned plus rejected")
    require(len({canonical_selection(row) for row in selected}) == len(selected) and
            len({canonical_selection(row) for row in planned}) == len(planned),
            "preflight selection or plan repeats an occurrence")
    if expected_selected is not None:
        require(len(selected) == expected_selected, "preflight selected count differs from expectation")
    if expected_rejected is not None:
        require(len(rejected) == expected_rejected, "preflight rejected count differs from expectation")
    require(sorted(canonical_selection(row) for row in selected) ==
            sorted([canonical_selection(row) for row in planned] +
                   [canonical_selection(row["selected"]) for row in rejected]),
            "preflight planned and rejected rows do not exactly partition selection")
    planned_ids = [row["powerampFileId"] for row in planned]
    rejected_ids = [row["selected"]["powerampFileId"] for row in rejected]
    selected_by_id = {row["powerampFileId"]: row for row in selected}
    require(len(selected_by_id) == len(selected) and len(set(planned_ids)) == len(planned_ids) and
            len(set(rejected_ids)) == len(rejected_ids) and
            set(planned_ids).isdisjoint(rejected_ids) and
            all(selected_by_id.get(row["powerampFileId"]) == row for row in planned) and
            all(selected_by_id.get(row["selected"]["powerampFileId"]) == row["selected"]
                for row in rejected) and
            planned == [row for row in selected if row["powerampFileId"] in set(planned_ids)] and
            [row["selected"] for row in rejected] ==
            [row for row in selected if row["powerampFileId"] in set(rejected_ids)],
            "preflight result order or immutable selection binding is invalid")
    planned_from_descriptors = [{
        "powerampFileId": descriptor["powerampFileId"],
        "providerPhysicalPath": descriptor["providerRow"]["providerPhysicalPath"],
        "durationMs": max(0, descriptor["providerRow"]["durationMs"]),
        "offsetMs": descriptor["providerRow"]["offsetMs"],
        **({"cueSourceImageFolderId": descriptor["providerRow"].get("cueSourceImageFolderId")}
           if descriptor["providerRow"].get("cueSourceImageFolderId") is not None else {}),
    } for descriptor in spec["tracks"]]
    require([canonical_selection(row) for row in planned] ==
            [canonical_selection(row) for row in planned_from_descriptors],
            "preflight plan differs from ordered immutable descriptor occurrences")
    local_failure_semantics = {
        "INVALID_LOGICAL_SPAN": ("BLOCKED", "SOURCE_OR_LIBRARY_CHANGED"),
        "CUE_SOURCE_IMAGE": ("BLOCKED", "SOURCE_OR_LIBRARY_CHANGED"),
        "AUDIO_TOO_SHORT": ("BLOCKED", "SOURCE_OR_LIBRARY_CHANGED"),
        "SOURCE_UNREADABLE": ("RETRYABLE", "SOURCE_AVAILABLE"),
        "NO_AUDIO_STREAM": ("BLOCKED", "SOURCE_OR_LIBRARY_CHANGED"),
        "UNSUPPORTED_OR_INVALID_AUDIO_CONTAINER": ("BLOCKED", "USER_REQUEST"),
    }
    require(all(rejected_row.get("code") in local_failure_semantics and
                (rejected_row.get("disposition"), rejected_row.get("retryTrigger")) ==
                local_failure_semantics[rejected_row["code"]] and
                isinstance(rejected_row.get("diagnostic"), str) and
                0 < len(rejected_row["diagnostic"].strip()) <= 2_048
                for rejected_row in rejected),
            "preflight rejected-row semantics are invalid")

    quiescence_booleans = {
        "jobComplete", "activeJobPointerAbsent", "activeJobPointerAtomicResidueAbsent",
        "activeGenerationPointerAtomicResidueAbsent", "executorLeaseActiveNull",
        "executorLeaseAtomicResidueAbsent", "indexingServiceAbsent", "indexingWakeLockAbsent",
        "authorizationNamespaceClean", "jobArtifactDirectoryAbsent", "jobDatabaseResidueAbsent",
        "generationStagingAbsent",
    }
    require(set(quiescence) == {"schemaVersion", "jobId"} | quiescence_booleans and
            quiescence.get("schemaVersion") == 1 and quiescence.get("jobId") == expected_job and
            all(quiescence[key] is True for key in quiescence_booleans),
            "executor is not terminal and quiescent")
    require(lease.get("schemaVersion") == 1 and lease.get("active") is None and
            isinstance(lease.get("lastIssuedEpoch"), int) and lease["lastIssuedEpoch"] >= 0,
            "executor lease is active or invalid")
    require(set(authorization_namespace) == {
                "schemaVersion", "jobId", "currentPresent", "legacyPresent",
                "atomicResiduePaths", "clean",
            } and
            authorization_namespace.get("schemaVersion") == 1 and
            authorization_namespace.get("jobId") == expected_job and
            isinstance(authorization_namespace.get("currentPresent"), bool) and
            isinstance(authorization_namespace.get("legacyPresent"), bool) and
            authorization_namespace.get("clean") is True and
            authorization_namespace.get("atomicResiduePaths") == [] and
            not (authorization_namespace.get("currentPresent") is True and
                 authorization_namespace.get("legacyPresent") is True),
            "imported-row authorization namespace is ambiguous or has atomic residue")
    services_text = (root / "raw/services.txt").read_text(encoding="utf-8", errors="replace")
    power_text = (root / "raw/power.txt").read_text(encoding="utf-8", errors="replace")
    require("IndexingService" not in services_text, "IndexingService still appears in service state")
    require(re.search(r"PARTIAL_WAKE_LOCK.*(?:com\.powerampstartradio\.v2:)?v2-indexing",
                      power_text, re.IGNORECASE) is None,
            "indexing wake lock still appears held")
    previous_generations = pointer.get("previousGenerationIds")
    require(pointer.get("schemaVersion") == 2 and
            re.fullmatch(r"index-generation-v2-[0-9a-f]{64}",
                         pointer.get("generationId", "")) is not None and
            re.fullmatch(r"[0-9a-f]{64}", pointer.get("manifestSha256", "")) is not None and
            isinstance(previous_generations, list) and
            all(isinstance(generation, str) and
                re.fullmatch(r"index-generation-v2-[0-9a-f]{64}", generation) is not None
                for generation in previous_generations) and
            len(set(previous_generations)) == len(previous_generations) and
            pointer["generationId"] not in previous_generations,
            "active generation pointer schema or history is invalid")
    require(pointer["generationId"] == active_manifest["generationId"],
            "active pointer does not name captured generation")
    require(pointer["generationId"] != expected_base,
            "active generation did not advance beyond base")
    require(pointer["manifestSha256"] == sha256_file(root / "active/manifest.json"),
            "active pointer manifest hash is invalid")
    require(pointer.get("previousGenerationIds", [None])[0] == expected_base,
            "active pointer does not retain exact immediate base")
    require(active_manifest.get("schemaVersion") == 3 and
            active_manifest["origin"] == "INDEXING_JOB" and
            active_manifest["jobId"] == expected_job and
            active_manifest["jobSpecId"] == spec["specId"] and
            active_manifest.get("baseGenerationId") == expected_base and
            active_manifest["createdAtEpochMs"] == spec["createdAtEpochMs"],
            "active manifest is not bound to completed job")
    require(active_manifest["databaseRelativePath"] == "library.db" and
            active_manifest["embeddingRelativePath"] == "clamp3.emb" and
            active_manifest["graph"]["relativePath"] == "graph.bin" and
            active_manifest["embeddingDimension"] == 768 and
            active_manifest["receiptEmbeddingSpec"]["outputDimension"] == 768 and
            active_manifest["textRetrievalSpec"]["outputDimension"] == 768 and
            active_manifest["graph"]["nodeCount"] == active_manifest["trackCount"] and
            active_manifest["graph"]["neighborsPerNode"] > 0,
            "active manifest asset shape is invalid")
    require(active_manifest["rebuildDerivedIndexes"] is True and
            active_manifest["rebuildDerivedIndexes"] == spec["rebuildDerivedIndexes"] and
            active_manifest["receiptEmbeddingSpec"] == spec["embeddingSpec"] and
            active_manifest["textRetrievalSpec"] == spec["textRetrievalSpec"],
            "active manifest model or rebuild contract differs from immutable job")
    require(active_manifest["graphPolicy"] == "EXPLICIT_REBUILD",
            "completion verifier requires an EXPLICIT_REBUILD graph")
    require(base_manifest.get("schemaVersion") == 3 and
            base_manifest["generationId"] == expected_base,
            "captured base manifest is not the expected predecessor")
    activation = ledger["activationEvidence"]
    activation_bindings = {
        "generationId": active_manifest["generationId"],
        "activationBindingId": active_manifest["activationBindingId"],
        "jobSpecId": active_manifest["jobSpecId"],
        "receiptEmbeddingSpecId": active_manifest["receiptEmbeddingSpec"]["specId"],
        "textRetrievalSpecId": active_manifest["textRetrievalSpec"]["specId"],
        "baseGenerationId": active_manifest.get("baseGenerationId"),
        "rebuildDerivedIndexes": active_manifest["rebuildDerivedIndexes"],
        "manifestSha256": pointer["manifestSha256"],
        "databaseSha256": active_manifest["databaseSha256"],
        "databaseContentSha256": active_manifest["databaseContentSha256"],
        "orderedTrackSetSha256": active_manifest["orderedTrackSetSha256"],
        "stableTrackUidMappingSha256": active_manifest["stableTrackUidCoverage"]["mappingSha256"],
        "embeddingSha256": active_manifest["embeddingSha256"],
        "graphSha256": active_manifest["graph"]["sha256"],
    }
    require(all(activation.get(key) == value for key, value in activation_bindings.items()),
            "ledger activation evidence does not match active manifest")
    require(spec["createdAtEpochMs"] <= activation["activatedAtEpochMs"] <=
            ledger["updatedAtEpochMs"],
            "activation timestamp is outside job lifetime")

    active_db = open_database("active/library.db")
    base_db = open_database("base/library.db")
    active_stats = inspect_database(active_db, active_manifest, "active")
    base_stats = inspect_database(base_db, base_manifest, "base")
    inspect_active_binary_assets(active_manifest, active_db)

    job_receipt_rows = [row for row in active_db.execute("""
        SELECT receipt_schema_version,work_id,stable_track_span_id,stable_identity_spec_id,
               stable_identity_strength,embedding_spec_id,provider_physical_path,
               provider_offset_ms,provider_duration_ms,track_id,metadata_sha256,
               embedding_byte_length,embedding_sha256,committed_at_epoch_ms
        FROM v2_embedding_commit_receipts_v4
    """) if row["work_id"] in descriptors]
    receipts = {row["work_id"]: row for row in job_receipt_rows}
    require(len(job_receipt_rows) == len(receipts) == expected_executable and
            set(receipts) == set(descriptors),
            "active database does not contain exact job receipt set")
    placeholders = ",".join("?" for _ in descriptors)
    require(scalar(base_db, f"SELECT COUNT(*) FROM v2_embedding_commit_receipts_v4 WHERE work_id IN ({placeholders})",
                   tuple(descriptors)) == 0, "job work IDs already existed in base receipts")
    for work_id, descriptor in descriptors.items():
        receipt = receipts[work_id]
        identity = descriptor["stableTrackSpanIdentity"]
        provider = descriptor["providerRow"]
        clamp_sha = next(item["sha256"] for item in track_ledgers[work_id]["verifiedArtifacts"]
                         if item["kind"] == "CLAMP_VECTOR")
        require(receipt["receipt_schema_version"] == 4 and
                receipt["stable_track_span_id"] == identity["stableTrackSpanId"] and
                receipt["stable_identity_spec_id"] == identity["identitySpecId"] and
                receipt["stable_identity_strength"] == identity["strength"] and
                receipt["embedding_spec_id"] == embedding_spec and
                receipt["provider_physical_path"] == provider["physicalPath"] and
                receipt["provider_offset_ms"] == provider["offsetMs"] and
                receipt["provider_duration_ms"] == max(0, provider["durationMs"]) and
                receipt["embedding_byte_length"] == 3072 and
                receipt["embedding_sha256"] == clamp_sha,
                f"receipt {work_id} is not exactly bound to descriptor and artifact")
        blob = scalar(active_db, "SELECT embedding FROM embeddings_clamp3 WHERE track_id=?",
                      (receipt["track_id"],))
        require(sha256_bytes(blob) == receipt["embedding_sha256"],
                f"receipt {work_id} embedding hash is invalid")
        metadata = active_db.execute("""
            SELECT metadata_key,filename_key,artist,album,title,duration_ms,file_path,source
            FROM tracks WHERE id=?
        """, (receipt["track_id"],)).fetchone()
        expected_metadata = commit_metadata_for_descriptor(descriptor)
        require(metadata is not None and tuple(metadata) == expected_metadata and
                metadata_sha(expected_metadata) == receipt["metadata_sha256"],
                f"receipt {work_id} metadata differs from immutable descriptor")
        database_artifact = next(item for item in track_ledgers[work_id]["verifiedArtifacts"]
                                 if item["kind"] == "DATABASE_COMMIT")
        require(database_artifact["storageKey"] ==
                f"sqlite:embeddings_clamp3:track:{receipt['track_id']}:"
                f"v2_embedding_commit_receipts_v4:{work_id}:{embedding_spec}" and
                database_artifact["verifiedAtEpochMs"] == receipt["committed_at_epoch_ms"],
                f"receipt {work_id} is not bound to its database artifact")

    logical_cues = {work_id: descriptor for work_id, descriptor in descriptors.items()
                    if descriptor["finalizedAudioSpan"]["kind"] == "LOGICAL_CUE"}
    auth_path = root / "raw/imported-row-authorization.json"
    authorization_path_evidence = root / "raw/authorization-path.txt"
    namespace_count = int(authorization_namespace.get("currentPresent") is True) + \
        int(authorization_namespace.get("legacyPresent") is True)
    if logical_cues:
        require(namespace_count == 1 and auth_path.is_file() and
                authorization_path_evidence.is_file(),
                "logical CUE work has no unique imported-row authorization evidence")
        expected_suffix = (".imported-row-supersession-v1.auth" if
                           authorization_namespace["currentPresent"] else
                           ".imported-row-supersession-v1.json")
        require(authorization_path_evidence.read_text(encoding="utf-8").strip() ==
                f"files/indexing_v2/jobs/{expected_job}{expected_suffix}",
                "authorization path evidence disagrees with captured namespace")
        auth_envelope = load_json("raw/imported-row-authorization.json")
        require(auth_envelope.get("format") == "poweramp-start-radio-v2-imported-row-supersession" and
                auth_envelope.get("schemaVersion") == 1, "imported-row authorization schema is invalid")
        auth = auth_envelope["authorization"]
        require(auth.get("schemaVersion") == 1 and auth["jobId"] == expected_job and
                re.fullmatch(r"[A-Za-z0-9._-]{1,128}", auth["jobId"]) is not None and
                not auth["jobId"].endswith(".imported-row-supersession-v1") and
                auth["jobSpecId"] == spec["specId"] and auth["baseGenerationId"] == expected_base and
                auth["baseManifestSha256"] == assets[("base", "manifest.json")][1] and
                auth["baseDatabaseByteLength"] == base_manifest["databaseByteLength"] and
                auth["baseDatabaseSha256"] == base_manifest["databaseSha256"] and
                auth["baseDatabaseContentSha256"] == base_manifest["databaseContentSha256"] and
                auth["providerSnapshotGeneration"] == spec["providerSnapshot"]["libraryGeneration"],
                "imported-row authorization is not bound to job and base")
        require(auth["privateBaseBindingId"] == private_base_binding_id(
                    expected_job, spec["specId"], expected_base,
                    base_manifest["databaseByteLength"], base_manifest["databaseSha256"],
                    assets[("base", "manifest.json")][1],
                    base_manifest["databaseContentSha256"]),
                "imported-row authorization private-base binding is invalid")
        works = {work["workId"]: work for work in auth["works"]}
        require(len(works) == len(auth["works"]) and set(works) == set(logical_cues),
                "authorization does not exhaust exact logical CUE work")
        provider_span_keys = {
            (work["providerSpan"]["normalizedPhysicalPath"],
             work["providerSpan"]["offsetMs"], work["providerSpan"]["durationMs"])
            for work in auth["works"]
        }
        predecessor_ids_all = [work["predecessor"]["trackId"] for work in auth["works"]
                               if work.get("predecessor") is not None]
        require(len({work["powerampFileId"] for work in auth["works"]}) == len(works) and
                len(provider_span_keys) == len(works) and
                len(set(predecessor_ids_all)) == len(predecessor_ids_all),
                "authorization repeats a Poweramp row, provider span, or predecessor")
    else:
        require(namespace_count == 0 and not auth_path.exists() and
                not authorization_path_evidence.exists(),
                "non-CUE job unexpectedly has imported-row authorization")
        works = {}

    supersession_works = {work_id: work for work_id, work in works.items()
                          if work["kind"] == "SUPERSESSION"}
    for work_id, work in works.items():
        descriptor = logical_cues[work_id]
        provider = descriptor["providerRow"]
        require(work["powerampFileId"] == descriptor["powerampFileId"] and
                work["providerSpan"] == {"normalizedPhysicalPath": provider["physicalPath"],
                                         "offsetMs": provider["offsetMs"],
                                         "durationMs": max(0, provider["durationMs"])},
                f"authorization work {work_id} is not bound to logical CUE descriptor")
        require((work["kind"] == "ADDITION" and work.get("predecessor") is None) or
                (work["kind"] == "SUPERSESSION" and work.get("predecessor") is not None),
                f"authorization work {work_id} has invalid commit kind")
        if work.get("predecessor") is not None:
            predecessor = work["predecessor"]
            predecessor_metadata = predecessor["metadata"]
            predecessor_tuple = (
                predecessor_metadata["metadataKey"], predecessor_metadata["filenameKey"],
                predecessor_metadata.get("artist"), predecessor_metadata.get("album"),
                predecessor_metadata.get("title"), predecessor_metadata["durationMs"],
                predecessor_metadata["filePath"], predecessor_metadata.get("source", "phone"),
            )
            require(predecessor["trackId"] > 0 and
                    predecessor["metadataSha256"] == metadata_sha(predecessor_tuple) and
                    predecessor["embeddingByteLength"] == 3072 and
                    re.fullmatch(r"[0-9a-f]{64}",
                                 predecessor["embeddingSha256"]) is not None,
                    f"authorization work {work_id} has invalid predecessor evidence")

    supersession_row_list = list(active_db.execute("""
        SELECT * FROM v2_imported_row_supersessions_v1 WHERE job_spec_id=?
    """, (spec["specId"],)))
    supersession_rows = {row["work_id"]: row for row in supersession_row_list}
    require(len(supersession_row_list) == len(supersession_rows) and
            set(supersession_rows) == set(supersession_works),
            "generation supersession audit does not match derived supersession work")
    predecessor_ids = []
    for work_id, work in supersession_works.items():
        predecessor = work["predecessor"]
        row = supersession_rows[work_id]
        receipt = receipts[work_id]
        expected_fields = {
            "supersession_schema_version": 1,
            "embedding_spec_id": embedding_spec,
            "job_spec_id": spec["specId"],
            "base_generation_id": expected_base,
            "base_manifest_sha256": assets[("base", "manifest.json")][1],
            "base_database_sha256": base_manifest["databaseSha256"],
            "private_base_binding_id": auth["privateBaseBindingId"],
            "provider_snapshot_generation": spec["providerSnapshot"]["libraryGeneration"],
            "predecessor_track_id": predecessor["trackId"],
            "predecessor_metadata_sha256": predecessor["metadataSha256"],
            "predecessor_embedding_byte_length": predecessor["embeddingByteLength"],
            "predecessor_embedding_sha256": predecessor["embeddingSha256"],
            "replacement_track_id": receipt["track_id"],
            "committed_at_epoch_ms": receipt["committed_at_epoch_ms"],
        }
        require(all(row[key] == value for key, value in expected_fields.items()),
                f"supersession audit row {work_id} is not exact")
        predecessor_ids.append(predecessor["trackId"])
        base_row = base_db.execute("""
            SELECT t.metadata_key,t.filename_key,t.artist,t.album,t.title,t.duration_ms,
                   t.file_path,t.source,e.embedding,
                   (SELECT COUNT(*) FROM v2_embedding_commit_receipts_v4 r WHERE r.track_id=t.id)
            FROM tracks t JOIN embeddings_clamp3 e ON e.track_id=t.id WHERE t.id=?
        """, (predecessor["trackId"],)).fetchone()
        require(base_row is not None and base_row[9] == 0,
                f"authorized predecessor for {work_id} is not exact unreceipted base row")
        predecessor_metadata = predecessor["metadata"]
        authorized_metadata = (
            predecessor_metadata["metadataKey"], predecessor_metadata["filenameKey"],
            predecessor_metadata.get("artist"), predecessor_metadata.get("album"),
            predecessor_metadata.get("title"), predecessor_metadata["durationMs"],
            predecessor_metadata["filePath"], predecessor_metadata.get("source", "phone"),
        )
        require(metadata_sha(authorized_metadata) == predecessor["metadataSha256"] and
                tuple(base_row[:8]) == authorized_metadata and
                metadata_sha(tuple(base_row[:8])) == predecessor["metadataSha256"] and
                len(base_row[8]) == predecessor["embeddingByteLength"] and
                sha256_bytes(base_row[8]) == predecessor["embeddingSha256"],
                f"authorized predecessor for {work_id} fingerprint changed")
        require(scalar(active_db, "SELECT COUNT(*) FROM tracks WHERE id=?",
                       (predecessor["trackId"],)) == 0,
                f"superseded predecessor for {work_id} remains active")

    supersession_count = len(supersession_works)
    expected_compatibility = derive_base_compatibility_after_supersessions(
        base_db, set(predecessor_ids))
    require(active_manifest["embeddingCoverage"].get("compatibilityBase") ==
            expected_compatibility,
            "active compatibility binding is not exact base content minus supersessions")
    require(active_stats["trackCount"] == base_stats["trackCount"] + expected_executable - supersession_count,
            "track-count delta does not match derived CUE supersessions")
    require(active_stats["receiptCount"] == base_stats["receiptCount"] + expected_executable,
            "receipt-count delta does not match committed work")
    require(active_manifest["stableTrackUidCoverage"]["coveredTrackCount"] ==
            base_manifest["stableTrackUidCoverage"]["coveredTrackCount"] + expected_executable,
            "stable-identity coverage delta does not match committed work")
    require(active_manifest["stableTrackUidCoverage"]["uncoveredTrackCount"] ==
            base_manifest["stableTrackUidCoverage"]["uncoveredTrackCount"] - supersession_count,
            "stable-identity uncovered delta does not match supersessions")
    descriptor_strengths = [descriptor["stableTrackSpanIdentity"]["strength"]
                            for descriptor in descriptors.values()]
    require(active_manifest["stableTrackUidCoverage"]["fullContentIdentityCount"] ==
            base_manifest["stableTrackUidCoverage"]["fullContentIdentityCount"] +
            descriptor_strengths.count("FULL_CONTENT_SHA256") and
            active_manifest["stableTrackUidCoverage"]["sampledContentIdentityCount"] ==
            base_manifest["stableTrackUidCoverage"]["sampledContentIdentityCount"] +
            descriptor_strengths.count("VERSIONED_SAMPLED_CONTENT_SHA256"),
            "stable-identity strength deltas do not match committed descriptors")
    expected_stable_ids = {
        row[0] for row in base_db.execute(
            "SELECT stable_track_span_id FROM v2_embedding_commit_receipts_v4")
    } | {descriptor["stableTrackSpanIdentity"]["stableTrackSpanId"]
         for descriptor in descriptors.values()}
    require(active_manifest["stableTrackUidCoverage"]["uniqueStableTrackSpanCount"] ==
            len(expected_stable_ids),
            "unique stable-span coverage is not the exact inherited-and-committed set union")
    require(active_stats["compatibilityCount"] ==
            base_stats["compatibilityCount"] - supersession_count,
            "compatibility delta does not match derived CUE supersessions")

    base_uri = "file:" + str((root / "base/library.db").resolve()) + "?mode=ro&immutable=1"
    active_db.execute("ATTACH DATABASE ? AS base_generation", (base_uri,))
    excluded = set(predecessor_ids)
    excluded_sql = "" if not excluded else \
        " AND base.track_id NOT IN (" + ",".join("?" for _ in excluded) + ")"
    inherited_embedding_mismatch = scalar(active_db, """
        SELECT COUNT(*) FROM base_generation.embeddings_clamp3 base
        LEFT JOIN embeddings_clamp3 current
          ON current.track_id=base.track_id AND current.embedding=base.embedding
        WHERE current.track_id IS NULL
    """ + excluded_sql, tuple(sorted(excluded)))
    excluded_track_sql = "" if not excluded else \
        " AND base.id NOT IN (" + ",".join("?" for _ in excluded) + ")"
    inherited_track_mismatch = scalar(active_db, """
        SELECT COUNT(*) FROM base_generation.tracks base
        LEFT JOIN tracks current ON current.id=base.id AND
          current.metadata_key=base.metadata_key AND current.filename_key=base.filename_key AND
          current.artist IS base.artist AND current.album IS base.album AND current.title IS base.title AND
          current.duration_ms=base.duration_ms AND current.file_path=base.file_path AND
          current.source=base.source
        WHERE current.id IS NULL
    """ + excluded_track_sql, tuple(sorted(excluded)))
    require(inherited_embedding_mismatch == 0 and inherited_track_mismatch == 0,
            "new generation changed a non-superseded inherited row")
    inherited_receipt_mismatch = scalar(active_db, """
        SELECT COUNT(*) FROM (
          SELECT * FROM base_generation.v2_embedding_commit_receipts_v4
          EXCEPT SELECT * FROM v2_embedding_commit_receipts_v4
        )
    """)
    require(inherited_receipt_mismatch == 0,
            "new generation changed an inherited V2 receipt")
    if scalar(base_db, "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND "
                       "name='v2_imported_row_supersessions_v1'") == 1:
        inherited_supersession_mismatch = scalar(active_db, """
            SELECT COUNT(*) FROM (
              SELECT * FROM base_generation.v2_imported_row_supersessions_v1
              EXCEPT SELECT * FROM v2_imported_row_supersessions_v1
            )
        """)
        require(inherited_supersession_mismatch == 0,
                "new generation changed an inherited supersession audit row")

    summary = {
        "format": "poweramp-start-radio-v2-production-indexing-completion-verification",
        "schemaVersion": 1,
        "verified": True,
        "jobId": expected_job,
        "jobSpecId": spec["specId"],
        "executableCount": expected_executable,
        "selectedCount": len(selected),
        "rejectedCount": len(rejected),
        "logicalCueCount": len(logical_cues),
        "derivedSupersessionCount": supersession_count,
        "baseGenerationId": expected_base,
        "activeGenerationId": active_manifest["generationId"],
        "baseTrackCount": base_stats["trackCount"],
        "activeTrackCount": active_stats["trackCount"],
        "baseReceiptBoundTrackCount": base_stats["receiptCount"],
        "activeReceiptBoundTrackCount": active_stats["receiptCount"],
        "baseCompatibilityTrackCount": base_stats["compatibilityCount"],
        "activeCompatibilityTrackCount": active_stats["compatibilityCount"],
        "preflightSha256": preflight_sha,
        "baseManifestSha256": assets[("base", "manifest.json")][1],
        "activeManifestSha256": pointer["manifestSha256"],
        "activeDatabaseSha256": active_manifest["databaseSha256"],
        "activeEmbeddingSha256": active_manifest["embeddingSha256"],
        "activeGraphSha256": active_manifest["graph"]["sha256"],
        "quiescent": True,
    }
    temporary = root / "summary.json.tmp"
    temporary.write_text(json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n",
                         encoding="utf-8")
    os.replace(temporary, root / "summary.json")
except AssertionError as error:
    print(f"FAIL: {error}", file=sys.stderr)
    sys.exit(1)
except Exception as error:
    print(f"FAIL: unexpected verifier error: {error}", file=sys.stderr)
    sys.exit(1)
PY

    (
        cd -- "$evidence_dir"
        find . -type f ! -name evidence-sha256.tsv -print0 |
            sort -z | xargs -0 sha256sum
    ) >"$evidence_dir/evidence-sha256.tsv"
    jq -c . "$evidence_dir/summary.json"
}

if [[ "${PASR_COMPLETION_VERIFIER_HOST_TEST_SOURCE_ONLY:-}" == "1" ]]; then
    return 0
fi

require_commands

job_id=""
expected_executable_count=""
expected_base_generation_id=""
evidence_dir=""
expected_selected_count=""
expected_rejected_count=""
expected_preflight_sha256=""
expected_base_manifest_sha256=""
quiescence_timeout_seconds=120
serial=""

while (($# > 0)); do
    case "$1" in
        --job-id) (($# >= 2)) || die "--job-id requires a value"; job_id="$2"; shift 2 ;;
        --expected-executable-count)
            (($# >= 2)) || die "--expected-executable-count requires a value"
            expected_executable_count="$2"; shift 2 ;;
        --expected-base-generation-id)
            (($# >= 2)) || die "--expected-base-generation-id requires a value"
            expected_base_generation_id="$2"; shift 2 ;;
        --evidence-dir) (($# >= 2)) || die "--evidence-dir requires a value"; evidence_dir="$2"; shift 2 ;;
        --expected-selected-count)
            (($# >= 2)) || die "--expected-selected-count requires a value"
            expected_selected_count="$2"; shift 2 ;;
        --expected-rejected-count)
            (($# >= 2)) || die "--expected-rejected-count requires a value"
            expected_rejected_count="$2"; shift 2 ;;
        --expected-preflight-sha256)
            (($# >= 2)) || die "--expected-preflight-sha256 requires a value"
            expected_preflight_sha256="$2"; shift 2 ;;
        --expected-base-manifest-sha256)
            (($# >= 2)) || die "--expected-base-manifest-sha256 requires a value"
            expected_base_manifest_sha256="$2"; shift 2 ;;
        --quiescence-timeout-seconds)
            (($# >= 2)) || die "--quiescence-timeout-seconds requires a value"
            quiescence_timeout_seconds="$2"; shift 2 ;;
        --serial) (($# >= 2)) || die "--serial requires a value"; serial="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument: $1" ;;
    esac
done

[[ -n "$job_id" ]] || die "--job-id is required"
[[ -n "$expected_executable_count" ]] || die "--expected-executable-count is required"
[[ -n "$expected_base_generation_id" ]] || die "--expected-base-generation-id is required"
[[ -n "$evidence_dir" ]] || die "--evidence-dir is required"
validate_safe_id "job ID" "$job_id"
[[ "$job_id" =~ ^[A-Za-z0-9._-]{1,128}$ ]] || die "job ID is outside the app namespace"
[[ "$job_id" != *.imported-row-supersession-v1 ]] || die \
    "job ID collides with the legacy imported-row authorization namespace"
validate_safe_id "base generation ID" "$expected_base_generation_id"
validate_count "expected executable count" "$expected_executable_count"
((expected_executable_count > 0)) || die "expected executable count must be positive"
validate_count "quiescence timeout" "$quiescence_timeout_seconds"
((quiescence_timeout_seconds > 0)) || die "quiescence timeout must be positive"
if [[ -n "$expected_selected_count" ]]; then validate_count "expected selected count" "$expected_selected_count"; fi
if [[ -n "$expected_rejected_count" ]]; then validate_count "expected rejected count" "$expected_rejected_count"; fi
if [[ -n "$expected_preflight_sha256" ]]; then validate_sha256 "$expected_preflight_sha256"; fi
if [[ -n "$expected_base_manifest_sha256" ]]; then validate_sha256 "$expected_base_manifest_sha256"; fi

if [[ -e "$evidence_dir" ]]; then
    [[ -d "$evidence_dir" ]] || die "evidence path is not a directory: $evidence_dir"
    [[ -z "$(find "$evidence_dir" -mindepth 1 -print -quit)" ]] || die \
        "evidence directory must be empty: $evidence_dir"
else
    mkdir -p -- "$evidence_dir"
fi
evidence_dir="$(cd -- "$evidence_dir" && pwd)"

resolve_serial
capture_evidence
verify_captured_evidence
