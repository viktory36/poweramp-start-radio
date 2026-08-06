#!/usr/bin/env bash
set -euo pipefail
umask 077

test_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
runner="$test_script_dir/run_production_indexing_acceptance.sh"

fail() {
    printf 'FAIL: %s\n' "$*" >&2
    exit 1
}

assert_rejected() {
    local label="$1"
    shift
    if ("$@") >/dev/null 2>&1; then
        fail "$label was unexpectedly accepted"
    fi
}

bash -n "$runner"
if command -v shellcheck >/dev/null; then
    shellcheck -x "$runner"
fi

# shellcheck source-path=SCRIPTDIR
# shellcheck source=run_production_indexing_acceptance.sh
PASR_ACCEPTANCE_HOST_TEST_SOURCE_ONLY=1 source "$runner"
activity_source="$test_script_dir/../app/src/debug/java/com/powerampstartradio/debug/ProductionIndexingAcceptanceActivity.kt"

ids_csv="9, 3,12"
all_ready=false
ready_cap=""
[[ "$(resolve_report_selection_mode)" == "EXPLICIT_IDS" ]]
[[ "$(normalized_explicit_ids_json)" == $'[\n  3,\n  9,\n  12\n]' ]]

ids_csv=""
all_ready=true
ready_cap=""
[[ "$(resolve_report_selection_mode)" == "ALL_READY" ]]

ids_csv=""
all_ready=false
ready_cap=25
[[ "$(resolve_report_selection_mode)" == "READY_CAP" ]]

ids_csv="1"
all_ready=true
ready_cap=""
assert_rejected "multiple report modes" resolve_report_selection_mode

tmp="$(mktemp -d)"
trap 'rm -rf -- "$tmp"' EXIT
purpose=OVERNIGHT
profile=BACKGROUND
minimum_free_bytes=$((4 * 1024 * 1024 * 1024))

build_report_request "$tmp/all-ready-request.json" run-1 job-1 1000 \
    ALL_READY '[]' null
jq -e '
    .schemaVersion == 2 and .selectionMode == "ALL_READY" and
    .powerampFileIds == [] and .readyCap == null and
    .rebuildDerivedIndexes == true
' "$tmp/all-ready-request.json" >/dev/null

build_report_request "$tmp/capped-request.json" run-2 job-2 1001 \
    READY_CAP '[]' 25
jq -e '
    .selectionMode == "READY_CAP" and .powerampFileIds == [] and .readyCap == 25
' "$tmp/capped-request.json" >/dev/null

build_report_request "$tmp/explicit-request.json" run-3 job-3 1002 \
    EXPLICIT_IDS '[3,9,12]' null
jq -e '
    .selectionMode == "EXPLICIT_IDS" and .powerampFileIds == [3,9,12] and
    .readyCap == null
' "$tmp/explicit-request.json" >/dev/null

fingerprint="$(printf 'a%.0s' {1..64})"
jq -n --arg policy "$automatic_selection_policy" --arg fingerprint "$fingerprint" '
    {schemaVersion:2,selectionMode:"ALL_READY",readyCap:null,
     selectionPolicy:$policy,discoveredReadyPowerampFileIds:[3,9,12],
     discoveredReadyFingerprint:$fingerprint,
     tracks:[{powerampFileId:3},{powerampFileId:9},{powerampFileId:12}]}
' >"$tmp/all-ready-manifest.json"
assert_manifest_selection_binding "$tmp/all-ready-manifest.json"

jq -n --arg policy "$automatic_selection_policy" --arg fingerprint "$fingerprint" '
    {schemaVersion:2,selectionMode:"READY_CAP",readyCap:2,
     selectionPolicy:$policy,discoveredReadyPowerampFileIds:[3,9,12],
     discoveredReadyFingerprint:$fingerprint,
     tracks:[{powerampFileId:3},{powerampFileId:9}]}
' >"$tmp/capped-manifest.json"
assert_manifest_selection_binding "$tmp/capped-manifest.json"

jq -n --arg policy "$explicit_selection_policy" '
    {schemaVersion:2,selectionMode:"EXPLICIT_IDS",readyCap:null,
     selectionPolicy:$policy,discoveredReadyPowerampFileIds:null,
     discoveredReadyFingerprint:null,
     tracks:[{powerampFileId:3},{powerampFileId:9}]}
' >"$tmp/explicit-manifest.json"
assert_manifest_selection_binding "$tmp/explicit-manifest.json"

jq '.tracks=[{powerampFileId:3},{powerampFileId:12}]' \
    "$tmp/capped-manifest.json" >"$tmp/tampered-track-list.json"
assert_rejected "non-prefix capped track list" \
    assert_manifest_selection_binding "$tmp/tampered-track-list.json"

jq '.discoveredReadyFingerprint="not-a-sha"' \
    "$tmp/all-ready-manifest.json" >"$tmp/tampered-fingerprint.json"
assert_rejected "invalid readiness fingerprint" \
    assert_manifest_selection_binding "$tmp/tampered-fingerprint.json"

jq '.selectionMode="EXPLICIT_IDS"' \
    "$tmp/all-ready-manifest.json" >"$tmp/tampered-declaration.json"
assert_rejected "inconsistent selection declaration" \
    assert_manifest_selection_binding "$tmp/tampered-declaration.json"

run_as() {
    command "$@"
}
mkdir -p "$tmp/pointer/files/indexing_v2"
pushd "$tmp/pointer" >/dev/null
[[ -z "$(active_job_id_or_empty)" ]]
touch files/indexing_v2/active-job-id.bak
[[ "$(active_job_id_or_empty)" == "__UNREADABLE_ACTIVE_POINTER_ATOMIC_RESIDUE__" ]]
rm files/indexing_v2/active-job-id.bak
touch files/indexing_v2/active-job-id.new
[[ "$(active_job_id_or_empty)" == "__UNREADABLE_ACTIVE_POINTER_ATOMIC_RESIDUE__" ]]
rm files/indexing_v2/active-job-id.new
printf 'bad pointer!\n' >files/indexing_v2/active-job-id
[[ "$(active_job_id_or_empty)" == "__UNREADABLE_ACTIVE_POINTER_BASE__" ]]
printf 'job-safe_123\n' >files/indexing_v2/active-job-id
[[ "$(active_job_id_or_empty)" == "job-safe_123" ]]
popd >/dev/null

grep -q 'start requires --manifest-sha256 from the reviewed report' "$runner"
if grep -Eq 'content[[:space:]]+(insert|update|delete|call)|ACTION_(API|RELOAD)|DEBUG_START_RADIO' \
    "$runner" "$activity_source"; then
    fail "acceptance surface contains a Poweramp mutation command"
fi

printf 'production indexing acceptance host checks passed\n'
