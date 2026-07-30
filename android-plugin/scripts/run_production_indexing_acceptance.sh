#!/usr/bin/env bash
set -euo pipefail
umask 077

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"
repo_dir="$(cd -- "$project_dir/.." && pwd)"
package="com.powerampstartradio.v2"
activity="$package/com.powerampstartradio.debug.ProductionIndexingAcceptanceActivity"
resume_activity="$package/com.powerampstartradio.debug.IndexingResumeAcceptanceActivity"
private_acceptance_root="files/device_acceptance/indexing"
request_format="poweramp-start-radio-v2-production-indexing-acceptance-request"
manifest_format="poweramp-start-radio-v2-production-indexing-candidate-manifest"
start_result_format="poweramp-start-radio-v2-production-indexing-start-result"
snapshot_format="poweramp-start-radio-v2-production-indexing-runtime-snapshot"
failure_format="poweramp-start-radio-v2-production-indexing-acceptance-failure"
explicit_selection_policy="EXACT_REQUESTED_POWERAMP_IDS_ASCENDING_V1"
automatic_selection_policy="PRODUCTION_READY_MINUS_NEVER_IGNORED_FAILURE_AND_PREFLIGHT_ATTENTION;POWERAMP_ID_ASC_V1"
minimum_storage_floor=$((4 * 1024 * 1024 * 1024))

usage() {
    cat >&2 <<'EOF'
usage:
  run_production_indexing_acceptance.sh report \
      (--ids ID,ID,... | --all-ready | --ready-cap COUNT) \
      --purpose reboot|overnight --expected-apk-sha256 SHA256 \
      [--profile FULL|BALANCED|BACKGROUND] [--minimum-free-bytes BYTES] \
      [--output-dir DIRECTORY]

  run_production_indexing_acceptance.sh start \
      --manifest CANDIDATE_MANIFEST --manifest-sha256 SHA256 \
      --expected-apk-sha256 SHA256 \
      --confirm-production-indexing [--run-dir DIRECTORY]

  run_production_indexing_acceptance.sh capture --run-dir DIRECTORY [--label LABEL]
  run_production_indexing_acceptance.sh monitor --run-dir DIRECTORY \
      [--interval-seconds SECONDS] [--max-samples COUNT]
  run_production_indexing_acceptance.sh reboot --run-dir DIRECTORY --confirm-reboot
  run_production_indexing_acceptance.sh resume --run-dir DIRECTORY --confirm-resume
  run_production_indexing_acceptance.sh finalize --run-dir DIRECTORY

Safety model:
  * report resolves a frozen cohort, hashes only that cohort, and writes no indexing job;
  * all-ready uses the production ready policy, while ready-cap takes ascending exact IDs;
  * start requires a reviewed, all-READY manifest plus an explicit confirmation;
  * CUE groups are blocked until atomic imported-row supersession exists;
  * no missing-source injection exists here; synthetic missing evidence stays separately labelled;
  * reboot is one-shot and requires a durable MERT artifact while the exact job is RUNNING;
  * every action can be rerun after host/adb interruption without creating a second job.
EOF
}

die() {
    printf 'FAIL: %s\n' "$*" >&2
    exit 1
}

log() {
    printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*"
}

require_commands() {
    local command_name
    for command_name in adb awk cat cmp cp date diff find grep jq mkdir mktemp mv sed \
        sha256sum sleep sort stat sync tr wc xargs; do
        command -v "$command_name" >/dev/null || die "missing required command: $command_name"
    done
}

resolve_serial() {
    if [[ -n "${ANDROID_SERIAL:-}" ]]; then
        serial="$ANDROID_SERIAL"
    else
        mapfile -t devices < <(adb devices | awk 'NR > 1 && $2 == "device" { print $1 }')
        ((${#devices[@]} == 1)) || die \
            "set ANDROID_SERIAL unless exactly one ready device is attached"
        serial="${devices[0]}"
    fi
}

adb_device() {
    adb -s "$serial" "$@"
}

run_as() {
    adb_device shell run-as "$package" "$@"
}

wait_for_ready_device() {
    local deadline=$((SECONDS + 600))
    while ((SECONDS < deadline)); do
        if [[ "$(adb_device get-state 2>/dev/null || true)" == "device" ]]; then
            return
        fi
        sleep 2
    done
    die "device $serial did not return to adb within 10 minutes"
}

current_user_id() {
    local value
    value="$(adb_device shell am get-current-user | tr -d '\r\n')"
    [[ "$value" =~ ^[0-9]+$ ]] || die "cannot determine current Android user"
    printf '%s\n' "$value"
}

package_base_apk() {
    local target_package="$1"
    adb_device shell pm path --user "$current_user" "$target_package" |
        tr -d '\r' | sed -n '1s/^package://p'
}

installed_apk_sha256() {
    local target_package="$1" apk_path
    apk_path="$(package_base_apk "$target_package")"
    [[ -n "$apk_path" ]] || die "required package is absent: $target_package"
    adb_device exec-out sha256sum "$apk_path" | tr -d '\r' | awk '{print $1}'
}

private_file_sha256() {
    run_as sha256sum "$1" | tr -d '\r' | awk '{print $1}'
}

validate_sha256() {
    [[ "$1" =~ ^[0-9a-f]{64}$ ]] || die "invalid SHA-256: $1"
}

validate_safe_name() {
    [[ "$1" =~ ^[A-Za-z0-9._-]{1,80}$ ]] || die "unsafe name: $1"
}

load_run_value() {
    local key="$1" env_file="$run_dir/run.env"
    [[ -f "$env_file" ]] || die "missing run state: $env_file"
    awk -F= -v wanted="$key" '$1 == wanted { sub(/^[^=]*=/, ""); print; found=1 } END { if (!found) exit 1 }' \
        "$env_file"
}

require_installed_debug_bridge() {
    local expected_hash="$1" actual_hash package_dump
    validate_sha256 "$expected_hash"
    actual_hash="$(installed_apk_sha256 "$package")"
    [[ "$actual_hash" == "$expected_hash" ]] || die \
        "installed V2 APK $actual_hash differs from expected $expected_hash"
    run_as true >/dev/null 2>&1 || die "installed V2 is not a debuggable build"
    package_dump="$(adb_device shell dumpsys package "$package" | tr -d '\r')"
    grep -q 'ProductionIndexingAcceptanceActivity' <<<"$package_dump" || die \
        "installed debug APK does not contain the production-indexing acceptance bridge"
    grep -Eq 'android.permission.READ_MEDIA_AUDIO: granted=true' <<<"$package_dump" || die \
        "V2 lacks READ_MEDIA_AUDIO"
    grep -Eq 'android.permission.POST_NOTIFICATIONS: granted=true' <<<"$package_dump" || die \
        "V2 lacks POST_NOTIFICATIONS"
}

private_usable_bytes() {
    run_as df -k files | tr -d '\r' | awk 'NR == 2 {printf "%.0f\n", $4 * 1024}'
}

is_current_user_unlocked() {
    adb_device shell dumpsys user | tr -d '\r' |
        awk -v user="$current_user" '
            $0 ~ "UserInfo\\{" user ":" { in_user=1; next }
            in_user && /^[[:space:]]+State:/ { print ($0 ~ /RUNNING_UNLOCKED/ ? "true" : "false"); exit }
        ' | grep -qx true
}

active_job_id_or_empty() {
    local base=files/indexing_v2/active-job-id
    local value
    if run_as test -e "$base.bak" || run_as test -e "$base.new"; then
        printf '__UNREADABLE_ACTIVE_POINTER_ATOMIC_RESIDUE__\n'
        return
    fi
    run_as test -e "$base" || return
    value="$(run_as cat "$base" 2>/dev/null | tr -d '\r\n' || true)"
    if [[ "$value" =~ ^[A-Za-z0-9._-]{1,128}$ ]]; then
        printf '%s\n' "$value"
    else
        printf '__UNREADABLE_ACTIVE_POINTER_BASE__\n'
    fi
}

stage_private_file() {
    local host_file="$1" private_path="$2" expected_hash temporary statuses
    [[ -f "$host_file" ]] || die "host file is absent: $host_file"
    [[ "$private_path" =~ ^files/device_acceptance/indexing/[A-Za-z0-9._/-]+$ ]] || die \
        "unsafe private acceptance path: $private_path"
    expected_hash="$(sha256sum "$host_file" | awk '{print $1}')"
    run_as mkdir -p "${private_path%/*}"
    if run_as test -f "$private_path" &&
        [[ "$(private_file_sha256 "$private_path")" == "$expected_hash" ]]; then
        return
    fi
    temporary="${private_path}.stage-$$"
    set +e
    adb_device shell "run-as $package sh -c 'cat > $temporary'" <"$host_file"
    statuses=("${PIPESTATUS[@]}")
    set -e
    ((statuses[0] == 0)) || die "failed to stage $host_file"
    [[ "$(private_file_sha256 "$temporary")" == "$expected_hash" ]] || {
        run_as rm -f "$temporary" >/dev/null 2>&1 || true
        die "private staged file hash mismatch"
    }
    run_as chmod 600 "$temporary"
    run_as mv "$temporary" "$private_path"
}

pull_private_file() {
    local private_path="$1" host_file="$2"
    local temporary="${host_file}.tmp-$$"
    mkdir -p -- "${host_file%/*}"
    adb_device exec-out run-as "$package" cat "$private_path" >"$temporary" || {
        rm -f -- "$temporary"
        return 1
    }
    [[ -s "$temporary" ]] || { rm -f -- "$temporary"; return 1; }
    mv -- "$temporary" "$host_file"
}

invoke_bridge() {
    local command="$1" input_relative="$2" input_sha="$3" output_relative="$4" timeout_seconds="${5:-3600}"
    local output_private="$private_acceptance_root/$output_relative" deadline
    run_as rm -f "$output_private" "$output_private.bak" "$output_private.new" >/dev/null
    adb_device shell am start --user "$current_user" -W -n "$activity" \
        --es acceptance_command "$command" \
        --es acceptance_input_relative_path "$input_relative" \
        --es acceptance_input_sha256 "$input_sha" \
        --es acceptance_output_relative_path "$output_relative" >/dev/null
    deadline=$((SECONDS + timeout_seconds))
    while ((SECONDS < deadline)); do
        if run_as test -s "$output_private"; then
            return
        fi
        sleep 2
    done
    die "acceptance bridge timed out waiting for $output_private"
}

assert_bridge_output() {
    local file="$1" expected_format="$2" actual_format
    jq -e . "$file" >/dev/null || die "bridge output is not valid JSON: $file"
    actual_format="$(jq -r '.format // empty' "$file")"
    if [[ "$actual_format" == "$failure_format" ]]; then
        die "bridge rejected the command: $(jq -r '.message' "$file")"
    fi
    [[ "$actual_format" == "$expected_format" ]] || die \
        "unexpected bridge output format: $actual_format"
}

assert_manifest_selection_binding() {
    local file="$1"
    jq -e --arg explicit "$explicit_selection_policy" \
        --arg automatic "$automatic_selection_policy" '
        (.tracks | map(.powerampFileId)) as $selected |
        ($selected == ($selected | sort)) and
        (($selected | unique | length) == ($selected | length)) and
        if .schemaVersion != 2 then false
        elif .selectionMode == "EXPLICIT_IDS" then
            .readyCap == null and
            .selectionPolicy == $explicit and
            .discoveredReadyPowerampFileIds == null and
            .discoveredReadyFingerprint == null
        elif .selectionMode == "ALL_READY" then
            .readyCap == null and
            .selectionPolicy == $automatic and
            (.discoveredReadyFingerprint | test("^[0-9a-f]{64}$")) and
            (.discoveredReadyPowerampFileIds ==
                (.discoveredReadyPowerampFileIds | sort | unique)) and
            .discoveredReadyPowerampFileIds == $selected
        elif .selectionMode == "READY_CAP" then
            (.readyCap | type) == "number" and
            .readyCap >= 1 and .readyCap <= 5000 and
            .selectionPolicy == $automatic and
            (.discoveredReadyFingerprint | test("^[0-9a-f]{64}$")) and
            (.discoveredReadyPowerampFileIds ==
                (.discoveredReadyPowerampFileIds | sort | unique)) and
            .discoveredReadyPowerampFileIds[0:.readyCap] == $selected
        else false
        end
    ' "$file" >/dev/null || die \
        "candidate manifest selection declaration does not bind its exact track list"
}

capture_host_runtime() {
    local destination="$1" pid
    mkdir -p -- "$destination"
    date --iso-8601=ns >"$destination/captured-at.txt"
    adb_device shell cat /proc/sys/kernel/random/boot_id | tr -d '\r' \
        >"$destination/boot-id.txt"
    adb_device shell getprop >"$destination/getprop.txt"
    adb_device shell dumpsys activity services "$package" >"$destination/services.txt"
    adb_device shell dumpsys notification --noredact >"$destination/notifications.txt"
    adb_device shell dumpsys power >"$destination/power.txt"
    adb_device shell dumpsys thermalservice >"$destination/thermal.txt"
    adb_device shell dumpsys battery >"$destination/battery.txt"
    adb_device shell dumpsys meminfo "$package" >"$destination/meminfo.txt"
    adb_device shell dumpsys deviceidle >"$destination/deviceidle.txt"
    adb_device shell ps -A >"$destination/ps.txt"
    adb_device shell pidof "$package" | tr -d '\r' >"$destination/pid.txt" || true
    pid="$(tr -d '\r\n ' <"$destination/pid.txt")"
    if [[ "$pid" =~ ^[0-9]+$ ]]; then
        adb_device shell cat "/proc/$pid/status" >"$destination/proc-status.txt" 2>/dev/null || true
    fi
    printf '%s\n' "$(private_usable_bytes)" >"$destination/private-usable-bytes.txt"
    local queue_tmp="$destination/poweramp-queue.txt.tmp"
    adb_device shell content query \
        --uri content://com.maxmpz.audioplayer.data/queue \
        --projection queue._id:queue.folder_file_id:queue.sort |
        tr -d '\r' >"$queue_tmp"
    mv -- "$queue_tmp" "$destination/poweramp-queue.txt"
    sha256sum "$destination/poweramp-queue.txt" >"$destination/poweramp-queue.sha256"
}

snapshot_protected_state() {
    local destination="$1"
    ANDROID_SERIAL="$serial" "$script_dir/snapshot_device_acceptance.sh" "$destination"
    capture_host_runtime "$destination/runtime"
}

stage_run_manifest() {
    local manifest="$1" run_id="$2"
    stage_private_file "$manifest" "$private_acceptance_root/$run_id/candidate-manifest.json"
}

capture_run() {
    local destination="$1" label="$2" manifest manifest_hash run_id job_id input_relative
    manifest="$(load_run_value manifest)"
    manifest_hash="$(load_run_value manifest_sha256)"
    run_id="$(load_run_value run_id)"
    job_id="$(load_run_value job_id)"
    validate_safe_name "$label"
    mkdir -p -- "$destination"
    stage_run_manifest "$manifest" "$run_id"
    input_relative="$run_id/candidate-manifest.json"
    local output_relative="$run_id/snapshots/$label.json"
    invoke_bridge snapshot "$input_relative" "$manifest_hash" "$output_relative" 300
    pull_private_file "$private_acceptance_root/$output_relative" \
        "$destination/runtime-snapshot.json"
    assert_bridge_output "$destination/runtime-snapshot.json" "$snapshot_format"
    [[ "$(jq -r '.jobId' "$destination/runtime-snapshot.json")" == "$job_id" ]] || die \
        "runtime snapshot belongs to another job"
    pull_private_file "files/indexing_v2/preflight-intents/$job_id.json" \
        "$destination/preflight-intent.json" || true
    pull_private_file "files/indexing_v2/jobs/$job_id.json" \
        "$destination/ledger.json" || true
    pull_private_file files/indexing_v2/eta-stage-rates-v1.json \
        "$destination/eta-stage-rates-v1.json" || true
    pull_private_file files/indexing_v2/generations/active-generation.json \
        "$destination/active-generation.json" || true
    capture_host_runtime "$destination/host"
    if [[ -f "$destination/ledger.json" ]]; then
        jq -r '.ledger.state' "$destination/ledger.json" >"$destination/job-state.txt"
    elif [[ -f "$destination/preflight-intent.json" ]]; then
        jq -r '.intent.state' "$destination/preflight-intent.json" >"$destination/job-state.txt"
    else
        printf 'NO_DURABLE_STATE\n' >"$destination/job-state.txt"
    fi
    (
        cd "$destination"
        find . -type f ! -name evidence-sha256.txt -print0 |
            sort -z | xargs -0 sha256sum
    ) >"$destination/evidence-sha256.txt"
}

write_run_env() {
    local destination="$1" manifest="$2" expected_apk="$3"
    local run_id job_id manifest_hash
    run_id="$(jq -r '.runId' "$manifest")"
    job_id="$(jq -r '.jobId' "$manifest")"
    manifest="$(cd -- "${manifest%/*}" && pwd)/${manifest##*/}"
    manifest_hash="$(sha256sum "$manifest" | awk '{print $1}')"
    {
        printf 'serial=%s\n' "$serial"
        printf 'run_id=%s\n' "$run_id"
        printf 'job_id=%s\n' "$job_id"
        printf 'manifest=%s\n' "$manifest"
        printf 'manifest_sha256=%s\n' "$manifest_hash"
        printf 'expected_apk_sha256=%s\n' "$expected_apk"
    } >"$destination/run.env"
}

load_and_verify_run() {
    local saved_serial expected_apk manifest manifest_hash
    [[ -d "$run_dir" ]] || die "run directory does not exist: $run_dir"
    saved_serial="$(load_run_value serial)"
    [[ "$serial" == "$saved_serial" ]] || die \
        "run belongs to device $saved_serial, not $serial"
    expected_apk="$(load_run_value expected_apk_sha256)"
    manifest="$(load_run_value manifest)"
    manifest_hash="$(load_run_value manifest_sha256)"
    [[ -f "$manifest" && "$(sha256sum "$manifest" | awk '{print $1}')" == "$manifest_hash" ]] || die \
        "run manifest is absent or changed"
    require_installed_debug_bridge "$expected_apk"
}

resolve_report_selection_mode() {
    local selected_modes=0 mode=""
    if [[ -n "$ids_csv" ]]; then
        selected_modes=$((selected_modes + 1))
        mode=EXPLICIT_IDS
    fi
    if [[ "$all_ready" == "true" ]]; then
        selected_modes=$((selected_modes + 1))
        mode=ALL_READY
    fi
    if [[ -n "$ready_cap" ]]; then
        selected_modes=$((selected_modes + 1))
        mode=READY_CAP
    fi
    ((selected_modes == 1)) || die \
        "report requires exactly one of --ids, --all-ready, or --ready-cap"
    if [[ "$mode" == "READY_CAP" ]]; then
        [[ "$ready_cap" =~ ^[0-9]+$ && "$ready_cap" -ge 1 && "$ready_cap" -le 5000 ]] || die \
            "ready cap must be an integer from 1 to 5000"
    fi
    printf '%s\n' "$mode"
}

normalized_explicit_ids_json() {
    local -a ids=()
    local -a raw_ids=()
    local raw_id
    IFS=',' read -ra raw_ids <<<"$ids_csv"
    for raw_id in "${raw_ids[@]}"; do
        raw_id="${raw_id//[[:space:]]/}"
        [[ "$raw_id" =~ ^[0-9]+$ && "$raw_id" -gt 0 ]] || die \
            "invalid Poweramp ID: $raw_id"
        ids+=("$raw_id")
    done
    ((${#ids[@]} > 0 && ${#ids[@]} <= 5000)) || die \
        "explicit report cohort must contain 1..5000 IDs"
    (($(printf '%s\n' "${ids[@]}" | sort -u | wc -l) == ${#ids[@]})) || die \
        "report IDs must be unique"
    printf '%s\n' "${ids[@]}" | sort -n | jq -s 'map(tonumber)'
}

build_report_request() {
    local destination="$1" run_id="$2" job_id="$3" created="$4"
    local selection_mode="$5" explicit_ids_json="$6" cap_json="$7"
    jq -n \
        --arg format "$request_format" \
        --arg runId "$run_id" \
        --arg purpose "$purpose" \
        --arg jobId "$job_id" \
        --argjson created "$created" \
        --arg selectionMode "$selection_mode" \
        --argjson readyCap "$cap_json" \
        --arg profile "$profile" \
        --argjson minimum "$minimum_free_bytes" \
        --argjson ids "$explicit_ids_json" \
        '{format:$format,schemaVersion:2,runId:$runId,purpose:$purpose,jobId:$jobId,
          jobCreatedAtEpochMs:$created,powerampFileIds:$ids,selectionMode:$selectionMode,
          readyCap:$readyCap,executionProfile:$profile,rebuildDerivedIndexes:true,
          minimumUsableBytes:$minimum}' >"$destination"
}

report_action() {
    [[ -n "$expected_apk_sha" ]] || die "report requires --expected-apk-sha256"
    [[ "$purpose" == "REBOOT" || "$purpose" == "OVERNIGHT" ]] || die \
        "purpose must be reboot or overnight"
    [[ "$profile" == "FULL" || "$profile" == "BALANCED" || "$profile" == "BACKGROUND" ]] || die \
        "invalid execution profile"
    [[ "$minimum_free_bytes" =~ ^[0-9]+$ ]] || die \
        "minimum free bytes must be an integer"
    ((minimum_free_bytes >= minimum_storage_floor)) || die \
        "minimum free bytes is below the 4 GiB safety floor"
    require_installed_debug_bridge "$expected_apk_sha"
    [[ -z "$(active_job_id_or_empty)" ]] || die \
        "an active indexing pointer exists; report will not overlap it"
    local usable_bytes
    usable_bytes="$(private_usable_bytes)"
    [[ "$usable_bytes" =~ ^[0-9]+$ ]] || die "cannot determine private usable storage"
    (( usable_bytes >= minimum_free_bytes )) || die \
        "device does not meet requested free-space floor"

    local selection_mode explicit_ids_json cap_json
    selection_mode="$(resolve_report_selection_mode)"
    explicit_ids_json='[]'
    cap_json=null
    if [[ "$selection_mode" == "EXPLICIT_IDS" ]]; then
        explicit_ids_json="$(normalized_explicit_ids_json)"
    elif [[ "$selection_mode" == "READY_CAP" ]]; then
        cap_json="$ready_cap"
    fi

    local run_id job_id created request request_hash candidate private_request private_output
    run_id="idx-$(date +%Y%m%dT%H%M%S)-$$"
    job_id="$(cat /proc/sys/kernel/random/uuid)"
    created="$(date +%s%3N)"
    output_dir="${output_dir:-$repo_dir/discovery/device-acceptance/$(date +%Y%m%dT%H%M%S%z)-indexing-candidate-$run_id}"
    mkdir -p -- "$output_dir"
    [[ -z "$(find "$output_dir" -mindepth 1 -print -quit)" ]] || die \
        "report output directory must be empty: $output_dir"
    request="$output_dir/request.json"
    candidate="$output_dir/candidate-manifest.json"
    build_report_request "$request" "$run_id" "$job_id" "$created" \
        "$selection_mode" "$explicit_ids_json" "$cap_json"
    request_hash="$(sha256sum "$request" | awk '{print $1}')"
    private_request="$private_acceptance_root/$run_id/request.json"
    private_output="$run_id/candidate-manifest.json"
    stage_private_file "$request" "$private_request"
    log "Building read-only pinned $selection_mode candidate report"
    invoke_bridge report "$run_id/request.json" "$request_hash" "$private_output" 7200
    pull_private_file "$private_acceptance_root/$private_output" "$candidate"
    assert_bridge_output "$candidate" "$manifest_format"
    assert_manifest_selection_binding "$candidate"
    [[ "$(jq -r '.installedApkSha256' "$candidate")" == "$expected_apk_sha" ]] || die \
        "candidate was generated by an unexpected APK"
    sha256sum "$candidate" >"$candidate.sha256"
    jq --arg manifestSha256 "$(sha256sum "$candidate" | awk '{print $1}')" '
        {selectionBindingVerified:true,manifestSha256:$manifestSha256,
         selectionMode,readyCap,selectionPolicy,discoveredReadyFingerprint,
         discoveredReadyPowerampFileIds,
         selectedPowerampFileIds:[.tracks[].powerampFileId],
         decisions:[.tracks[]|{powerampFileId,decision,blocker}]}
    ' "$candidate" >"$output_dir/selection-binding-evidence.json"
    capture_host_runtime "$output_dir/device"
    jq '{runId,purpose,jobId,selectionMode,readyCap,selectionPolicy,
         discoveredReadyCount:(.discoveredReadyPowerampFileIds|if .==null then null else length end),
         selectedCount:(.tracks|length),activeGenerationId,providerGenerationId,
         ready:[.tracks[]|select(.decision=="READY")|.powerampFileId],
         blocked:[.tracks[]|select(.decision!="READY")|{powerampFileId,decision,blocker}]}' \
        "$candidate" | tee "$output_dir/summary.json"
    log "Candidate manifest: $candidate"
    log "No indexing job was created. Review every track decision before start."
}

start_action() {
    [[ -f "$manifest_path" ]] || die "start requires --manifest"
    [[ -n "$manifest_sha" ]] || die "start requires --manifest-sha256 from the reviewed report"
    [[ -n "$expected_apk_sha" ]] || die "start requires --expected-apk-sha256"
    [[ "$confirm_start" == "true" ]] || die \
        "start requires --confirm-production-indexing"
    validate_sha256 "$manifest_sha"
    [[ "$(sha256sum "$manifest_path" | awk '{print $1}')" == "$manifest_sha" ]] || die \
        "candidate manifest differs from the explicitly reviewed SHA-256"
    assert_bridge_output "$manifest_path" "$manifest_format"
    assert_manifest_selection_binding "$manifest_path"
    local run_id job_id manifest_apk minimum ready_count track_count cue_count
    run_id="$(jq -r '.runId' "$manifest_path")"
    job_id="$(jq -r '.jobId' "$manifest_path")"
    validate_safe_name "$run_id"
    manifest_apk="$(jq -r '.installedApkSha256' "$manifest_path")"
    [[ "$manifest_apk" == "$expected_apk_sha" ]] || die \
        "manifest APK hash differs from explicit expected hash"
    ready_count="$(jq '[.tracks[]|select(.decision=="READY")]|length' "$manifest_path")"
    track_count="$(jq '.tracks|length' "$manifest_path")"
    [[ "$ready_count" == "$track_count" && "$track_count" -gt 0 ]] || die \
        "start accepts only a nonempty all-READY manifest"
    cue_count="$(jq '[.tracks[]|select((.offsetMs//0)>0 or .cueSourceImageFolderId!=null or .sourceHasLogicalOffsets==true or .sourceHasCueImageRow==true)]|length' "$manifest_path")"
    [[ "$cue_count" == 0 ]] || die \
        "CUE rows are blocked until atomic imported-row supersession is in production"
    grep -q 'NO_MISSING_SOURCE_INJECTION' "$manifest_path" || die \
        "manifest does not carry the natural-only missing-source policy"
    minimum="$(jq -r '.minimumUsableBytes' "$manifest_path")"
    [[ "$minimum" =~ ^[0-9]+$ && "$minimum" -ge "$minimum_storage_floor" ]] || die \
        "manifest has an invalid storage floor"
    require_installed_debug_bridge "$expected_apk_sha"
    local usable_bytes
    usable_bytes="$(private_usable_bytes)"
    [[ "$usable_bytes" =~ ^[0-9]+$ ]] || die "cannot determine private usable storage"
    (( usable_bytes >= minimum )) || die "device no longer meets manifest storage floor"
    local active
    active="$(active_job_id_or_empty)"
    [[ -z "$active" || "$active" == "$job_id" ]] || die \
        "another indexing job owns the active pointer: $active"

    run_dir="${run_dir:-$repo_dir/discovery/device-acceptance/$(date +%Y%m%dT%H%M%S%z)-production-indexing-$run_id}"
    if [[ -f "$run_dir/run.env" ]]; then
        local saved_run_id saved_job_id saved_manifest_hash supplied_manifest_hash
        load_and_verify_run
        saved_run_id="$(load_run_value run_id)"
        saved_job_id="$(load_run_value job_id)"
        saved_manifest_hash="$(load_run_value manifest_sha256)"
        supplied_manifest_hash="$(sha256sum "$manifest_path" | awk '{print $1}')"
        [[ "$saved_run_id" == "$run_id" && "$saved_job_id" == "$job_id" &&
            "$saved_manifest_hash" == "$supplied_manifest_hash" ]] || die \
            "existing run directory is bound to a different candidate manifest"
        log "Run already initialized; issuing only the idempotent exact-manifest start check"
    else
        [[ -z "$active" ]] || die \
            "cannot create a truthful pre-start baseline after job $active has already started"
        mkdir -p -- "$run_dir"
        [[ -z "$(find "$run_dir" -mindepth 1 -print -quit)" ]] || die \
            "new run directory must be empty: $run_dir"
        cp -- "$manifest_path" "$run_dir/candidate-manifest.json"
        manifest_path="$run_dir/candidate-manifest.json"
        write_run_env "$run_dir" "$manifest_path" "$expected_apk_sha"
        log "Capturing protected V1, V2, and exact Poweramp queue state before start"
        snapshot_protected_state "$run_dir/before"
    fi
    local manifest manifest_hash private_output result_file
    manifest="$(load_run_value manifest)"
    manifest_hash="$(load_run_value manifest_sha256)"
    stage_run_manifest "$manifest" "$run_id"
    private_output="$run_id/start-result.json"
    invoke_bridge start "$run_id/candidate-manifest.json" "$manifest_hash" "$private_output" 7200
    result_file="$run_dir/start-result.json"
    pull_private_file "$private_acceptance_root/$private_output" "$result_file"
    assert_bridge_output "$result_file" "$start_result_format"
    [[ "$(jq -r '.jobId' "$result_file")" == "$job_id" ]] || die \
        "start result belongs to another job"
    local deadline=$((SECONDS + 120))
    while ((SECONDS < deadline)); do
        [[ "$(active_job_id_or_empty)" == "$job_id" ]] && break
        sleep 2
    done
    [[ "$(active_job_id_or_empty)" == "$job_id" ]] || die \
        "exact active job pointer did not become durable"
    capture_run "$run_dir/samples/$(date +%Y%m%dT%H%M%S)-after-start" "after-start"
    jq . "$result_file"
    log "Production job $job_id is durable. The wrapper sent no Poweramp queue/playback command."
}

capture_action() {
    load_and_verify_run
    local capture_label="${label:-capture-$(date +%Y%m%dT%H%M%S)}"
    validate_safe_name "$capture_label"
    local destination
    destination="$run_dir/samples/$(date +%Y%m%dT%H%M%S)-$capture_label"
    capture_run "$destination" "$capture_label"
    cat "$destination/job-state.txt"
    log "Runtime evidence: $destination"
}

monitor_action() {
    load_and_verify_run
    [[ "$interval_seconds" =~ ^[0-9]+$ && "$max_samples" =~ ^[0-9]+$ ]] || die \
        "monitor interval and sample count must be non-negative integers"
    ((interval_seconds >= 5)) || die "monitor interval must be at least 5 seconds"
    ((max_samples >= 0)) || die "max samples cannot be negative"
    local count=0 state sample_label destination
    while true; do
        sample_label="monitor-$(date +%Y%m%dT%H%M%S)"
        destination="$run_dir/samples/$sample_label"
        capture_run "$destination" "$sample_label"
        state="$(cat "$destination/job-state.txt")"
        log "sample=$count state=$state evidence=$destination"
        count=$((count + 1))
        if [[ "$state" == "COMPLETE" || "$state" == "CANCELLED" ||
            "$state" == "WAITING_FOR_INPUT" || "$state" == "PAUSED" ||
            "$state" == "INTERRUPTED" || "$state" == "READY_TO_RESUME" ]]; then
            break
        fi
        ((max_samples == 0 || count < max_samples)) || break
        sleep "$interval_seconds"
    done
}

reboot_action() {
    [[ "$confirm_reboot" == "true" ]] || die "reboot requires --confirm-reboot"
    load_and_verify_run
    local job_id manifest run_id checkpoint_dir ledger state unresolved mert_count before_pid
    local before_boot_id pre_revision
    job_id="$(load_run_value job_id)"
    manifest="$(load_run_value manifest)"
    run_id="$(load_run_value run_id)"
    [[ "$(jq -r '.purpose' "$manifest")" == "REBOOT" ]] || die \
        "the controlled reboot action accepts only a REBOOT candidate manifest"
    if [[ -f "$run_dir/reboot-complete.json" ]]; then
        log "This run already completed its one allowed reboot; capturing only"
        capture_run "$run_dir/samples/$(date +%Y%m%dT%H%M%S)-reboot-already-complete" \
            "reboot-already-complete"
        return
    fi
    [[ ! -e "$run_dir/reboot-requested.json" ]] || die \
        "a reboot was requested but not marked complete; inspect evidence before any retry"
    checkpoint_dir="$run_dir/reboot/pre"
    capture_run "$checkpoint_dir" "pre-reboot"
    ledger="$checkpoint_dir/ledger.json"
    [[ -f "$ledger" ]] || die "no materialized ledger at reboot checkpoint"
    state="$(jq -r '.ledger.state' "$ledger")"
    [[ "$state" == "RUNNING" ]] || die "reboot checkpoint requires RUNNING, got $state"
    mert_count="$(jq '[.ledger.tracks[].verifiedArtifacts[]?|select(.kind=="MERT_FEATURES")]|length' "$ledger")"
    ((mert_count > 0)) || die "reboot requires at least one durable MERT artifact"
    unresolved="$(jq '[.ledger.tracks[]|select(.state!="COMMITTED" and .state!="BLOCKED_FAILURE" and .state!="SKIPPED_BY_USER")]|length' "$ledger")"
    ((unresolved > 0)) || die "job has no unresolved track work to recover"
    [[ "$(active_job_id_or_empty)" == "$job_id" ]] || die "active pointer changed before reboot"
    before_pid="$(tr -d '\r\n ' <"$checkpoint_dir/host/pid.txt")"
    before_boot_id="$(tr -d '\r\n ' <"$checkpoint_dir/host/boot-id.txt")"
    pre_revision="$(jq -r '.ledger.revision' "$ledger")"
    jq -n --arg jobId "$job_id" --arg at "$(date --iso-8601=ns)" \
        --arg ledgerSha "$(sha256sum "$ledger" | awk '{print $1}')" \
        '{jobId:$jobId,requestedAt:$at,preRebootLedgerSha256:$ledgerSha}' \
        >"$run_dir/reboot-requested.json"
    sync
    log "Rebooting at a verified production MERT checkpoint for job $job_id"
    adb_device reboot
    wait_for_ready_device
    local boot_deadline=$((SECONDS + 600))
    while ((SECONDS < boot_deadline)); do
        [[ "$(adb_device shell getprop sys.boot_completed | tr -d '\r\n')" == "1" ]] && break
        sleep 3
    done
    [[ "$(adb_device shell getprop sys.boot_completed | tr -d '\r\n')" == "1" ]] || die \
        "Android did not finish booting within 10 minutes"
    current_user="$(current_user_id)"
    local unlock_deadline=$((SECONDS + 600))
    while ((SECONDS < unlock_deadline)); do
        is_current_user_unlocked && break
        log "Waiting for Android user $current_user to be unlocked after reboot"
        sleep 3
    done
    is_current_user_unlocked || die \
        "Android user storage stayed locked; unlock the phone and rerun capture"
    require_installed_debug_bridge "$(load_run_value expected_apk_sha256)"
    local recovery_deadline=$((SECONDS + 300)) recovery_state="" post_dir
    local recovery_observed=false current_revision service_dump
    while ((SECONDS < recovery_deadline)); do
        if pull_private_file "files/indexing_v2/jobs/$job_id.json" "/tmp/pasr-reboot-ledger-$$.json"; then
            recovery_state="$(jq -r '.ledger.state' "/tmp/pasr-reboot-ledger-$$.json")"
            current_revision="$(jq -r '.ledger.revision' "/tmp/pasr-reboot-ledger-$$.json")"
            rm -f "/tmp/pasr-reboot-ledger-$$.json"
            service_dump="$(adb_device shell dumpsys activity services "$package" | tr -d '\r')"
            if [[ "$current_revision" =~ ^[0-9]+$ && "$current_revision" -gt "$pre_revision" ]] &&
                { [[ "$recovery_state" == "COMPLETE" || "$recovery_state" == "INTERRUPTED" ||
                    "$recovery_state" == "READY_TO_RESUME" || "$recovery_state" == "PAUSED" ]] ||
                    grep -q 'IndexingService' <<<"$service_dump"; }; then
                recovery_observed=true
                break
            fi
        fi
        sleep 5
    done
    post_dir="$run_dir/reboot/post"
    capture_run "$post_dir" "post-reboot"
    local after_pid after_job_id after_spec before_spec after_boot_id
    after_pid="$(tr -d '\r\n ' <"$post_dir/host/pid.txt")"
    after_boot_id="$(tr -d '\r\n ' <"$post_dir/host/boot-id.txt")"
    [[ -n "$before_boot_id" && -n "$after_boot_id" && "$before_boot_id" != "$after_boot_id" ]] || die \
        "kernel boot identity did not change"
    after_job_id="$(jq -r '.ledger.jobSpec.jobId' "$post_dir/ledger.json")"
    [[ "$after_job_id" == "$job_id" ]] || die "post-reboot ledger changed job ID"
    before_spec="$(jq -r '.ledger.jobSpec.specId' "$ledger")"
    after_spec="$(jq -r '.ledger.jobSpec.specId' "$post_dir/ledger.json")"
    [[ "$before_spec" == "$after_spec" ]] || die "post-reboot immutable job spec changed"
    jq -n --arg jobId "$job_id" --arg at "$(date --iso-8601=ns)" \
        --arg state "$(cat "$post_dir/job-state.txt")" \
        --arg beforePid "$before_pid" --arg afterPid "$after_pid" \
        --arg beforeBootId "$before_boot_id" --arg afterBootId "$after_boot_id" \
        --argjson recoveryObserved "$recovery_observed" \
        '{jobId:$jobId,completedAt:$at,observedState:$state,
          autonomousRecoveryObserved:$recoveryObserved,beforePid:$beforePid,afterPid:$afterPid,
          beforeBootId:$beforeBootId,afterBootId:$afterBootId}' \
        >"$run_dir/reboot-complete.json"
    log "Reboot evidence captured. Recovery state: $(cat "$post_dir/job-state.txt")"
    if [[ "$recovery_observed" != "true" ||
        "$(cat "$post_dir/job-state.txt")" == "INTERRUPTED" ||
        "$(cat "$post_dir/job-state.txt")" == "READY_TO_RESUME" ||
        "$(cat "$post_dir/job-state.txt")" == "PAUSED" ]]; then
        log "Android presented a saved/deferred state. Use explicit 'resume --confirm-resume'; this was not labelled autonomous recovery."
    fi
}

resume_action() {
    [[ "$confirm_resume" == "true" ]] || die "resume requires --confirm-resume"
    load_and_verify_run
    local job_id state ledger_tmp
    job_id="$(load_run_value job_id)"
    [[ "$(active_job_id_or_empty)" == "$job_id" ]] || die "active pointer is not the run job"
    ledger_tmp="$(mktemp)"
    trap 'rm -f -- "$ledger_tmp"' RETURN
    pull_private_file "files/indexing_v2/jobs/$job_id.json" "$ledger_tmp" || die \
        "no durable ledger is available to resume"
    state="$(jq -r '.ledger.state' "$ledger_tmp")"
    case "$state" in
        COMPLETE)
            log "Job is already complete; resume is an idempotent no-op"
            ;;
        RUNNING|ACTIVATING)
            if adb_device shell dumpsys activity services "$package" | grep -q 'IndexingService'; then
                log "Job is already $state with a live service; resume is an idempotent no-op"
            else
                adb_device shell am start --user "$current_user" -W -n "$resume_activity" \
                    --es indexing_job_id "$job_id" --es indexing_command recover >/dev/null
                log "Explicit production recovery sent for stale $state job $job_id"
            fi
            ;;
        PAUSED|INTERRUPTED|READY_TO_RESUME)
            adb_device shell am start --user "$current_user" -W -n "$resume_activity" \
                --es indexing_job_id "$job_id" --es indexing_command resume >/dev/null
            log "Explicit production resume sent for saved job $job_id"
            ;;
        *)
            die "refusing automatic resume from $state; review the app's saved evidence"
            ;;
    esac
    rm -f -- "$ledger_tmp"
    trap - RETURN
    capture_run "$run_dir/samples/$(date +%Y%m%dT%H%M%S)-after-explicit-resume" \
        "after-explicit-resume"
}

protected_lines() {
    local snapshot_dir="$1" label="$2"
    case "$label" in
        v1)
            cat "$snapshot_dir/v1/apk-sha256.txt"
            cat "$snapshot_dir/v1/private-file-sha256.txt"
            ;;
        v2-apk)
            cat "$snapshot_dir/v2/apk-sha256.txt"
            ;;
        v2-nonindex)
            grep -E '[[:space:]]+\./files/(session_history\.json(\.bak)?|mert\.tflite|clamp3_audio\.tflite|clamp3_text\.tflite|sentencepiece\.bpe\.model)$' \
                "$snapshot_dir/v2/private-file-sha256.txt" || true
            ;;
        *) die "unknown protected snapshot label" ;;
    esac | sort
}

finalize_action() {
    load_and_verify_run
    [[ -d "$run_dir/before" ]] || die "run has no protected before snapshot"
    if [[ ! -d "$run_dir/after" ]]; then
        snapshot_protected_state "$run_dir/after"
    fi
    local failed=false
    if ! cmp -s "$run_dir/before/poweramp-queue.txt" "$run_dir/after/poweramp-queue.txt"; then
        diff -u "$run_dir/before/poweramp-queue.txt" "$run_dir/after/poweramp-queue.txt" \
            >"$run_dir/poweramp-queue.diff" || true
        failed=true
    fi
    for label in v1 v2-apk v2-nonindex; do
        protected_lines "$run_dir/before" "$label" >"$run_dir/before-$label.txt"
        protected_lines "$run_dir/after" "$label" >"$run_dir/after-$label.txt"
        if ! cmp -s "$run_dir/before-$label.txt" "$run_dir/after-$label.txt"; then
            diff -u "$run_dir/before-$label.txt" "$run_dir/after-$label.txt" \
                >"$run_dir/$label.diff" || true
            failed=true
        fi
    done
    local final_dir
    final_dir="$run_dir/samples/$(date +%Y%m%dT%H%M%S)-final"
    capture_run "$final_dir" "final"
    {
        printf 'queue_unchanged=%s\n' "$([[ ! -f "$run_dir/poweramp-queue.diff" ]] && echo true || echo false)"
        printf 'v1_unchanged=%s\n' "$([[ ! -f "$run_dir/v1.diff" ]] && echo true || echo false)"
        printf 'v2_apk_unchanged=%s\n' "$([[ ! -f "$run_dir/v2-apk.diff" ]] && echo true || echo false)"
        printf 'v2_nonindex_protected_unchanged=%s\n' "$([[ ! -f "$run_dir/v2-nonindex.diff" ]] && echo true || echo false)"
        printf 'final_job_state=%s\n' "$(cat "$final_dir/job-state.txt")"
    } >"$run_dir/invariants.txt"
    [[ "$failed" == "false" ]] || die "one or more protected invariants changed; see $run_dir"
    cat "$run_dir/invariants.txt"
    log "Final protected-state evidence: $run_dir"
}

if [[ "${PASR_ACCEPTANCE_HOST_TEST_SOURCE_ONLY:-}" == "1" ]]; then
    return 0
fi

require_commands
(($# > 0)) || { usage; exit 2; }
action="$1"
shift

ids_csv=""
all_ready=false
ready_cap=""
purpose=""
profile="BACKGROUND"
minimum_free_bytes="$minimum_storage_floor"
output_dir=""
expected_apk_sha=""
manifest_path=""
manifest_sha=""
run_dir=""
label=""
interval_seconds=60
max_samples=0
confirm_start=false
confirm_reboot=false
confirm_resume=false

while (($#)); do
    case "$1" in
        --ids) (($# >= 2)) || { usage; exit 2; }; ids_csv="$2"; shift 2 ;;
        --all-ready) all_ready=true; shift ;;
        --ready-cap) (($# >= 2)) || { usage; exit 2; }; ready_cap="$2"; shift 2 ;;
        --purpose) (($# >= 2)) || { usage; exit 2; }; purpose="${2^^}"; shift 2 ;;
        --profile) (($# >= 2)) || { usage; exit 2; }; profile="${2^^}"; shift 2 ;;
        --minimum-free-bytes) (($# >= 2)) || { usage; exit 2; }; minimum_free_bytes="$2"; shift 2 ;;
        --output-dir) (($# >= 2)) || { usage; exit 2; }; output_dir="$2"; shift 2 ;;
        --expected-apk-sha256) (($# >= 2)) || { usage; exit 2; }; expected_apk_sha="$2"; shift 2 ;;
        --manifest) (($# >= 2)) || { usage; exit 2; }; manifest_path="$2"; shift 2 ;;
        --manifest-sha256) (($# >= 2)) || { usage; exit 2; }; manifest_sha="$2"; shift 2 ;;
        --run-dir) (($# >= 2)) || { usage; exit 2; }; run_dir="$2"; shift 2 ;;
        --label) (($# >= 2)) || { usage; exit 2; }; label="$2"; shift 2 ;;
        --interval-seconds) (($# >= 2)) || { usage; exit 2; }; interval_seconds="$2"; shift 2 ;;
        --max-samples) (($# >= 2)) || { usage; exit 2; }; max_samples="$2"; shift 2 ;;
        --confirm-production-indexing) confirm_start=true; shift ;;
        --confirm-reboot) confirm_reboot=true; shift ;;
        --confirm-resume) confirm_resume=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage; die "unknown argument: $1" ;;
    esac
done

resolve_serial
wait_for_ready_device
current_user="$(current_user_id)"

case "$action" in
    report) report_action ;;
    start) start_action ;;
    capture) [[ -n "$run_dir" ]] || die "capture requires --run-dir"; capture_action ;;
    monitor) [[ -n "$run_dir" ]] || die "monitor requires --run-dir"; monitor_action ;;
    reboot) [[ -n "$run_dir" ]] || die "reboot requires --run-dir"; reboot_action ;;
    resume) [[ -n "$run_dir" ]] || die "resume requires --run-dir"; resume_action ;;
    finalize) [[ -n "$run_dir" ]] || die "finalize requires --run-dir"; finalize_action ;;
    *) usage; exit 2 ;;
esac
