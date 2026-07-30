#!/usr/bin/env bash
set -euo pipefail
umask 077

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"
repo_dir="$(cd -- "$project_dir/.." && pwd)"
target_apk="$project_dir/app/build/outputs/apk/debug/app-debug.apk"
test_apk="$project_dir/app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk"
v1_package="com.powerampstartradio"
v2_package="com.powerampstartradio.v2"
test_package="com.powerampstartradio.v2.test"
runner="$test_package/androidx.test.runner.AndroidJUnitRunner"
single_test_method="com.powerampstartradio.indexing.v2.V2IndexingProfileComparisonInstrumentedTest#runOneIsolatedProfileCaseAndPersistEvidence"
batch_test_method="com.powerampstartradio.indexing.v2.V2IndexingProfileComparisonInstrumentedTest#runOneIsolatedFullBatchAndPersistEvidence"
test_method="$single_test_method"
private_fixture_root="files/device_acceptance/indexing_profile"

usage() {
    cat >&2 <<'EOF'
usage: run_isolated_indexing_profile_comparison.sh [OPTIONS]

Default mode runs each audio case in this exact order:
  FULL-1, BACKGROUND-1, BACKGROUND-2, FULL-2

`--full-batch-once` instead runs all supplied cases in one FULL job and one executor invocation.
Every mode uses an opt-in disposable private root. The script is resumable and verifies that the
live V2 generation is byte-identical before and after the run.

Options:
  --case ID=SOURCE              Add a source case (repeatable; at least one is required).
  --log-dir DIRECTORY           New evidence directory (must be empty).
  --resume DIRECTORY            Resume an interrupted comparison from its evidence directory.
  --foreground-settings         Keep the V2 Settings screen under deterministic swipe load.
  --full-batch-once             Run every supplied case (minimum 3) together as one FULL-profile
                                job in exactly one executor invocation. This is the pre/post speed
                                benchmark path; it does not run the FULL/BACKGROUND matrix.
  --disable-pcm-prefetch        Disable one-track-ahead PCM preparation for a same-build batch A/B.
  --timeout-seconds SECONDS     Per-invocation host timeout (default: 14400).
  --dry-run                     Validate cases and print the invocation plan without ADB/builds.
  -h, --help                    Show this help.

The long comparison is intentionally never implicit. Supplying --dry-run is the smoke path;
without it, the script builds, installs, snapshots protected state, and starts real inference.
EOF
}

requested_log_dir=""
resume_dir=""
foreground_settings=0
full_batch_once=0
pcm_prefetch_enabled=1
dry_run=0
timeout_seconds=14400
case_specs=()
while (($#)); do
    case "$1" in
        --case)
            (($# >= 2)) || { usage; exit 2; }
            case_specs+=("$2")
            shift 2
            ;;
        --log-dir)
            (($# >= 2)) || { usage; exit 2; }
            requested_log_dir="$2"
            shift 2
            ;;
        --resume)
            (($# >= 2)) || { usage; exit 2; }
            resume_dir="$2"
            shift 2
            ;;
        --foreground-settings)
            foreground_settings=1
            shift
            ;;
        --full-batch-once)
            full_batch_once=1
            test_method="$batch_test_method"
            shift
            ;;
        --disable-pcm-prefetch)
            pcm_prefetch_enabled=0
            shift
            ;;
        --timeout-seconds)
            (($# >= 2)) || { usage; exit 2; }
            timeout_seconds="$2"
            shift 2
            ;;
        --dry-run)
            dry_run=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            usage
            exit 2
            ;;
    esac
done

[[ "$timeout_seconds" =~ ^[1-9][0-9]*$ ]] || {
    printf 'timeout must be a positive integer\n' >&2
    exit 2
}
if [[ -n "$resume_dir" && (-n "$requested_log_dir" || ${#case_specs[@]} -gt 0) ]]; then
    printf -- '--resume cannot be combined with --log-dir or --case\n' >&2
    exit 2
fi
if [[ -n "$resume_dir" && $dry_run -eq 1 ]]; then
    printf -- '--resume cannot be combined with --dry-run\n' >&2
    exit 2
fi
if ((pcm_prefetch_enabled == 0 && full_batch_once == 0)); then
    printf -- '--disable-pcm-prefetch is only meaningful with --full-batch-once\n' >&2
    exit 2
fi

if ((${#case_specs[@]} == 0)) && [[ -z "$resume_dir" && $full_batch_once -eq 1 ]]; then
    printf -- '--full-batch-once requires at least three explicit --case ID=SOURCE arguments\n' >&2
    exit 2
fi
if ((${#case_specs[@]} == 0)) && [[ -z "$resume_dir" ]]; then
    printf 'pass at least one --case ID=SOURCE\n' >&2
    exit 2
fi

validate_case_specs() {
    local spec id source
    declare -A seen=()
    for spec in "${case_specs[@]}"; do
        [[ "$spec" == *=* ]] || { printf 'invalid --case: %s\n' "$spec" >&2; exit 2; }
        id="${spec%%=*}"
        source="${spec#*=}"
        [[ "$id" =~ ^[a-z0-9][a-z0-9_-]{0,31}$ ]] || {
            printf 'unsafe case ID: %s\n' "$id" >&2
            exit 2
        }
        [[ -z "${seen[$id]+present}" ]] || { printf 'duplicate case ID: %s\n' "$id" >&2; exit 2; }
        seen["$id"]=1
        [[ "$source" != *$'\n'* && "$source" != *$'\t'* && -f "$source" ]] || {
            printf 'source case is not a regular file: %s\n' "$source" >&2
            exit 2
        }
    done
    if ((full_batch_once)) && ((${#case_specs[@]} < 3 || ${#case_specs[@]} > 16)); then
        printf -- '--full-batch-once requires between 3 and 16 source cases\n' >&2
        exit 2
    fi
}

if ((dry_run)); then
    for command_name in sha256sum stat; do
        command -v "$command_name" >/dev/null || { printf 'missing %s\n' "$command_name" >&2; exit 1; }
    done
    validate_case_specs
    printf 'One-track-ahead PCM prefetch: %s\n' \
        "$([[ $pcm_prefetch_enabled -eq 1 ]] && printf enabled || printf disabled)"
    if ((full_batch_once)); then
        printf 'Invocation order (one FULL executor invocation for all rows):\n'
        for spec in "${case_specs[@]}"; do
            id="${spec%%=*}"
            source="${spec#*=}"
            printf '%-13s %-10s %-20s %12s bytes  %s\n' \
                'full-batch' 'FULL' "$id" "$(stat -c %s -- "$source")" \
                "$(sha256sum -- "$source" | awk '{ print $1 }')"
        done
    else
        printf 'Invocation order (one instrumentation invocation per row):\n'
        for run in 'full-1 FULL' 'background-1 BACKGROUND' 'background-2 BACKGROUND' 'full-2 FULL'; do
            read -r label profile <<<"$run"
            for spec in "${case_specs[@]}"; do
                id="${spec%%=*}"
                source="${spec#*=}"
                printf '%-13s %-10s %-20s %12s bytes  %s\n' \
                    "$label" "$profile" "$id" "$(stat -c %s -- "$source")" \
                    "$(sha256sum -- "$source" | awk '{ print $1 }')"
            done
        done
    fi
    exit 0
fi

for command_name in adb awk cmp date diff find grep jq realpath sed seq sha256sum stat tail tee timeout tr wc xargs; do
    command -v "$command_name" >/dev/null || {
        printf 'missing required command: %s\n' "$command_name" >&2
        exit 1
    }
done

if [[ -n "${ANDROID_SERIAL:-}" ]]; then
    serial="$ANDROID_SERIAL"
else
    mapfile -t devices < <(adb devices | awk 'NR > 1 && $2 == "device" { print $1 }')
    ((${#devices[@]} == 1)) || {
        printf 'set ANDROID_SERIAL when exactly one ready device is not attached\n' >&2
        exit 1
    }
    serial="${devices[0]}"
fi
adb_device() { adb -s "$serial" "$@"; }
run_as() { adb_device shell run-as "$v2_package" "$@"; }
[[ "$(adb_device get-state)" == device ]] || { printf 'ADB device is not ready\n' >&2; exit 1; }
current_user="$(adb_device shell am get-current-user | tr -d '\r')"
[[ "$current_user" =~ ^[0-9]+$ ]] || { printf 'cannot determine Android user\n' >&2; exit 1; }

if [[ -n "$resume_dir" ]]; then
    log_dir="${resume_dir%/}"
    [[ -d "$log_dir" && -f "$log_dir/run-id.txt" && -f "$log_dir/cases.tsv" ]] || {
        printf 'resume directory has no comparison state: %s\n' "$log_dir" >&2
        exit 2
    }
    run_id="$(tr -d '\r\n' <"$log_dir/run-id.txt")"
    recorded_foreground="$(awk -F= '$1 == "foreground_settings" { print $2 }' "$log_dir/config.txt")"
    [[ "$recorded_foreground" == "$foreground_settings" ]] || {
        printf 'resume must use the same --foreground-settings choice (%s)\n' "$recorded_foreground" >&2
        exit 2
    }
    recorded_batch="$(awk -F= '$1 == "full_batch_once" { print $2 }' "$log_dir/config.txt")"
    recorded_batch="${recorded_batch:-0}"
    [[ "$recorded_batch" == "$full_batch_once" ]] || {
        printf 'resume must use the same --full-batch-once choice (%s)\n' "$recorded_batch" >&2
        exit 2
    }
    recorded_prefetch="$(awk -F= '$1 == "pcm_prefetch_enabled" { print $2 }' "$log_dir/config.txt")"
    recorded_prefetch="${recorded_prefetch:-1}"
    [[ "$recorded_prefetch" == "$pcm_prefetch_enabled" ]] || {
        printf 'resume must use the same PCM-prefetch choice (%s)\n' "$recorded_prefetch" >&2
        exit 2
    }
    if ((full_batch_once)); then test_method="$batch_test_method"; fi
else
    validate_case_specs
    run_id="p$(date +%Y%m%dt%H%M%S)-$$"
    if [[ -n "$requested_log_dir" ]]; then
        log_dir="${requested_log_dir%/}"
    else
        log_dir="$repo_dir/discovery/device-acceptance/$(date +%Y%m%dT%H%M%z)-indexing-profile-comparison"
    fi
    mkdir -p -- "$log_dir"
    if find "$log_dir" -mindepth 1 -print -quit | grep -q .; then
        printf 'log directory must be empty: %s\n' "$log_dir" >&2
        exit 2
    fi
    printf '%s\n' "$run_id" >"$log_dir/run-id.txt"
    {
        printf 'schema=1\n'
        printf 'serial=%s\n' "$serial"
        printf 'foreground_settings=%s\n' "$foreground_settings"
        printf 'full_batch_once=%s\n' "$full_batch_once"
        printf 'pcm_prefetch_enabled=%s\n' "$pcm_prefetch_enabled"
        printf 'timeout_seconds=%s\n' "$timeout_seconds"
        printf 'test_method=%s\n' "$test_method"
    } >"$log_dir/config.txt"
fi

failure_handoff() {
    local status=$?
    if ((status != 0)); then
        printf '\nComparison stopped with evidence preserved. Resume with:\n' >&2
        printf '  ANDROID_SERIAL=%q %q --resume %q' "$serial" "$0" "$log_dir" >&2
        ((foreground_settings)) && printf ' --foreground-settings' >&2
        ((full_batch_once)) && printf ' --full-batch-once' >&2
        ((pcm_prefetch_enabled == 0)) && printf ' --disable-pcm-prefetch' >&2
        printf '\n' >&2
    fi
    return "$status"
}
trap failure_handoff EXIT

private_sha256() { run_as sha256sum "$1" | tr -d '\r' | awk '{ print $1 }'; }
private_size() { run_as stat -c %s "$1" | tr -d '\r\n'; }

if [[ -z "$resume_dir" ]]; then
    "$project_dir/gradlew" --no-daemon -p "$project_dir" \
        :app:assembleDebug :app:assembleDebugAndroidTest | tee "$log_dir/build.log"
    adb_device install -r "$target_apk" | tee "$log_dir/install-target.log"
    adb_device install -r -t "$test_apk" | tee "$log_dir/install-test.log"
    grep -qx 'Success' "$log_dir/install-target.log"
    grep -qx 'Success' "$log_dir/install-test.log"
    for permission in android.permission.READ_MEDIA_AUDIO android.permission.POST_NOTIFICATIONS; do
        adb_device shell pm grant --user "$current_user" "$v2_package" "$permission" >/dev/null
    done
    for package_name in "$v1_package" "$v2_package" "$test_package"; do
        adb_device shell pm path --user "$current_user" "$package_name" | grep -q '^package:/' || {
            printf 'required package is absent: %s\n' "$package_name" >&2
            exit 1
        }
    done

    : >"$log_dir/cases.tsv"
    case_index=0
    for spec in "${case_specs[@]}"; do
        id="${spec%%=*}"
        source="$(realpath -- "${spec#*=}")"
        extension="${source##*.}"
        [[ "$extension" =~ ^[A-Za-z0-9]{1,8}$ ]] || extension=audio
        source_hash="$(sha256sum -- "$source" | awk '{ print $1 }')"
        source_size="$(stat -c %s -- "$source")"
        private_relative="device_acceptance/indexing_profile/$run_id/sources/${id}-${source_hash:0:12}.${extension,,}"
        poweramp_id=$((8800000001 + case_index))
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$id" "$source" "$source_size" "$source_hash" "$private_relative" "$poweramp_id" \
            >>"$log_dir/cases.tsv"
        case_index=$((case_index + 1))
    done

    ANDROID_SERIAL="$serial" "$script_dir/snapshot_device_acceptance.sh" "$log_dir/before"
    printf '%s\n' "$(sha256sum "$target_apk" | awk '{ print $1 }')" >"$log_dir/target-apk.sha256"
    printf '%s\n' "$(sha256sum "$test_apk" | awk '{ print $1 }')" >"$log_dir/test-apk.sha256"
else
    expected_target="$(cat "$log_dir/target-apk.sha256")"
    expected_test="$(cat "$log_dir/test-apk.sha256")"
    device_target_path="$(adb_device shell pm path "$v2_package" | tr -d '\r' | sed -n 's/^package://p' | head -1)"
    device_test_path="$(adb_device shell pm path "$test_package" | tr -d '\r' | sed -n 's/^package://p' | head -1)"
    [[ -n "$device_target_path" && -n "$device_test_path" ]]
    [[ "$(adb_device shell sha256sum "$device_target_path" | awk '{ print $1 }' | tr -d '\r')" == "$expected_target" ]]
    [[ "$(adb_device shell sha256sum "$device_test_path" | awk '{ print $1 }' | tr -d '\r')" == "$expected_test" ]]
fi

model_records=(
    '34930763bed772d616b124b0103e3759f9cf464d1b193f73fc1367449f32c539|378064368|files/mert.tflite'
    '06b6a17425b7d8ccb5327db7129aa58102cd119502aed00a397bc4b6e7ae20bc|343793912|files/clamp3_audio.tflite'
    '10398ea03ee96e56d4c6970e302ac2061a41d255ed1ef3d730001ec17ca0fb16|1113080740|files/clamp3_text.tflite'
    'cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865|5069051|files/sentencepiece.bpe.model'
)
for record in "${model_records[@]}"; do
    IFS='|' read -r expected_hash expected_size destination <<<"$record"
    [[ "$(private_size "$destination")" == "$expected_size" && \
        "$(private_sha256 "$destination")" == "$expected_hash" ]] || {
        printf 'installed V2 model does not match pinned bytes: %s\n' "$destination" >&2
        exit 1
    }
done

# Freeze the tab-separated records before invoking ADB. Some adb subcommands may read their
# inherited standard input; iterating directly from cases.tsv would then let them consume the next
# source row and silently skip a case.
mapfile -t case_records <"$log_dir/cases.tsv"
(( ${#case_records[@]} > 0 ))

stage_private_source() {
    local host_source="$1" expected_size="$2" expected_hash="$3" relative="$4"
    local destination="files/$relative" parent="files/${relative%/*}" temporary statuses
    temporary="$parent/.${destination##*/}.stage-$$"
    if run_as test -e "$destination"; then
        [[ "$(private_size "$destination")" == "$expected_size" && \
            "$(private_sha256 "$destination")" == "$expected_hash" ]] || {
            printf 'refusing unexpected staged source: %s\n' "$destination" >&2
            exit 1
        }
        return
    fi
    run_as mkdir -p "$parent"
    set +e
    adb -s "$serial" shell "run-as $v2_package sh -c 'cat > $temporary'" <"$host_source"
    statuses=("${PIPESTATUS[@]}")
    set -e
    ((statuses[0] == 0)) || { run_as rm -f "$temporary" || true; return 1; }
    [[ "$(private_size "$temporary")" == "$expected_size" && \
        "$(private_sha256 "$temporary")" == "$expected_hash" ]] || {
        run_as rm -f "$temporary" || true
        printf 'staged source verification failed: %s\n' "$destination" >&2
        exit 1
    }
    run_as chmod 600 "$temporary"
    run_as mv "$temporary" "$destination"
}

for case_record in "${case_records[@]}"; do
    IFS=$'\t' read -r id source size source_hash private_relative poweramp_id <<<"$case_record"
    [[ -f "$source" && "$(stat -c %s -- "$source")" == "$size" && \
        "$(sha256sum -- "$source" | awk '{ print $1 }')" == "$source_hash" ]] || {
        printf 'host source changed since preparation: %s\n' "$source" >&2
        exit 1
    }
    stage_private_source "$source" "$size" "$source_hash" "$private_relative"
done

pull_private_result() {
    local private_file="$1" target_file="$2"
    local temporary="$target_file.tmp"
    mkdir -p -- "${target_file%/*}"
    adb_device exec-out run-as "$v2_package" cat "$private_file" >"$temporary"
    jq -e '
        if .schemaVersion == 1 then
            (.pcmSha256 | test("^[0-9a-f]{64}$")) and
            (.mertSha256 | test("^[0-9a-f]{64}$")) and
            (.clampSha256 == .databaseCommitEmbeddingSha256) and
            (.clampSha256 == .finalEmbeddingSha256)
        elif .schemaVersion == 2 then
            .profile == "FULL" and .sourceCount >= 3 and
            (.tracks | length) == .sourceCount and
            all(.tracks[];
                (.pcmSha256 | test("^[0-9a-f]{64}$")) and
                (.mertSha256 | test("^[0-9a-f]{64}$")) and
                (.clampSha256 == .databaseCommitEmbeddingSha256) and
                (.clampSha256 == .finalEmbeddingSha256)
            )
        else false end
        and (.databaseSemanticSha256 | test("^[0-9a-f]{64}$"))
        and (.pembSha256 | test("^[0-9a-f]{64}$"))
        and (.graphSha256 | test("^[0-9a-f]{64}$"))
    ' "$temporary" >/dev/null
    mv -- "$temporary" "$target_file"
    run_as rm -f "$private_file" "$private_file.bak" "$private_file.new"
}

capture_runtime_boundary() {
    local output="$1"
    mkdir -p -- "$output"
    adb_device shell dumpsys battery | tr -d '\r' >"$output/battery.txt"
    adb_device shell dumpsys thermalservice | tr -d '\r' >"$output/thermal.txt"
    adb_device shell dumpsys meminfo "$v2_package" | tr -d '\r' >"$output/meminfo.txt"
    adb_device shell dumpsys cpuinfo | tr -d '\r' >"$output/cpuinfo.txt"
}

open_settings_workload() {
    local run_dir="$1"
    local size width height settings_x settings_y focus
    adb_device shell am start --user "$current_user" \
        -n "$v2_package/com.powerampstartradio.MainActivity" \
        >"$run_dir/settings-launch.txt" 2>&1
    if grep -Eq '(^|[[:space:]])Error(:| type)' "$run_dir/settings-launch.txt"; then
        printf 'failed\n' >"$run_dir/settings-workload-status.txt"
        return 1
    fi
    size="$(adb_device shell wm size | tr -d '\r' | sed -n \
        's/.*: \([0-9]*\)x\([0-9]*\).*/\1 \2/p' | tail -1)"
    read -r width height <<<"$size"
    [[ "$width" =~ ^[1-9][0-9]*$ && "$height" =~ ^[1-9][0-9]*$ ]] || return 1
    # Settings is the stable 60x60 toolbar control at [980,110]-[1040,170] on the reference
    # 1080x2340 device. Scale its center without starting a second UiAutomation service; a second
    # automation owner can close an Activity launched inside instrumentation.
    settings_x=$((width * 1010 / 1080))
    settings_y=$((height * 140 / 2340))
    for _ in $(seq 1 40); do
        focus="$(
            timeout 5 adb -s "$serial" shell dumpsys window 2>/dev/null |
                tr -d '\r' | grep 'mCurrentFocus=' | tail -1 || true
        )"
        if [[ "$focus" == *"$v2_package/com.powerampstartradio.MainActivity"* ]]; then
            printf '%s\n' "$focus" >"$run_dir/settings-focus-before-tap.txt"
            sleep 1
            adb_device shell input tap "$settings_x" "$settings_y"
            sleep 0.5
            focus="$(
                timeout 5 adb -s "$serial" shell dumpsys window 2>/dev/null |
                    tr -d '\r' | grep 'mCurrentFocus=' | tail -1 || true
            )"
            printf '%s\n' "$focus" >"$run_dir/settings-focus-after-tap.txt"
            if [[ "$focus" == *"$v2_package/com.powerampstartradio.MainActivity"* ]]; then
                printf 'opened x=%s y=%s width=%s height=%s\n' \
                    "$settings_x" "$settings_y" "$width" "$height" \
                    >"$run_dir/settings-workload-status.txt"
                return 0
            fi
        fi
        sleep 0.5
    done
    printf 'failed\n' >"$run_dir/settings-workload-status.txt"
    return 1
}

sample_running_invocation() {
    local run_dir="$1" tick="$2" epoch pid
    epoch="$(date +%s%3N)"
    # Instrumentation can report as started one scheduler tick before its target process exists.
    # A missing PID is a valid sample, not a reason to abandon the resumable device invocation.
    pid="$(
        timeout 5 adb -s "$serial" shell pidof "$v2_package" 2>/dev/null |
            tr -d '\r' | awk '{ print $1 }' || true
    )"
    {
        printf '%s\t%s\t' "$epoch" "${pid:-none}"
        if [[ "$pid" =~ ^[1-9][0-9]*$ ]]; then
            adb_device shell cat "/proc/$pid/stat" 2>/dev/null | tr -d '\r\n' || printf 'unavailable'
        else
            printf 'unavailable'
        fi
        printf '\n'
    } >>"$run_dir/process-stat.tsv"
    if ((tick % 5 == 0)); then
        {
            printf '\n===== epoch_ms=%s =====\n' "$epoch"
            adb_device shell dumpsys meminfo "$v2_package" | tr -d '\r'
        } >>"$run_dir/meminfo-samples.txt"
        {
            printf '\n===== epoch_ms=%s =====\n' "$epoch"
            adb_device shell dumpsys thermalservice | tr -d '\r'
            adb_device shell dumpsys battery | tr -d '\r'
        } >>"$run_dir/thermal-battery-samples.txt"
        if ((foreground_settings)); then
            {
                printf '\n===== epoch_ms=%s =====\n' "$epoch"
                adb_device shell dumpsys gfxinfo "$v2_package" framestats | tr -d '\r'
            } >>"$run_dir/gfxinfo-samples.txt"
        fi
    fi
}

monitor_pid_or_private_result() {
    local monitor_pid="$1" private_result="$2" run_dir="$3" tick=0 workload_opened=0
    local size width height x y_top y_bottom
    if ((foreground_settings)); then
        size="$(adb_device shell wm size | tr -d '\r' | sed -n 's/.*: \([0-9]*\)x\([0-9]*\).*/\1 \2/p' | tail -1)"
        read -r width height <<<"$size"
        [[ "$width" =~ ^[0-9]+$ && "$height" =~ ^[0-9]+$ ]]
        x=$((width / 2)); y_top=$((height / 4)); y_bottom=$((height * 3 / 4))
        adb_device shell dumpsys gfxinfo "$v2_package" reset >/dev/null 2>&1 || true
    fi
    while true; do
        if ((tick >= timeout_seconds)); then
            printf 'device invocation monitor exceeded %s seconds\n' "$timeout_seconds" >&2
            return 1
        fi
        if [[ -n "$monitor_pid" ]]; then
            kill -0 "$monitor_pid" >/dev/null 2>&1 || break
        else
            if run_as test -f "$private_result"; then break; fi
            adb_device shell dumpsys activity instrumentation | grep -q "$test_package" || break
        fi
        sample_running_invocation "$run_dir" "$tick"
        if ((foreground_settings)); then
            if ((workload_opened == 0)); then
                if open_settings_workload "$run_dir"; then workload_opened=1; else return 1; fi
            else
                if ((tick % 2 == 0)); then
                    adb_device shell input swipe "$x" "$y_bottom" "$x" "$y_top" 350 >/dev/null
                else
                    adb_device shell input swipe "$x" "$y_top" "$x" "$y_bottom" 350 >/dev/null
                fi
            fi
        fi
        tick=$((tick + 1))
        sleep 1
    done
}

validate_local_result() {
    local file="$1" expected_label="$2" expected_profile="$3" expected_case="$4" expected_hash="$5"
    jq -e --arg expected_run "$run_id" --arg expected_label "$expected_label" \
        --arg expected_profile "$expected_profile" --arg expected_case "$expected_case" \
        --arg expected_source "$expected_hash" '
        .schemaVersion == 1 and .runId == $expected_run and
        .runLabel == $expected_label and .profile == $expected_profile and
        .caseId == $expected_case and .sourceSha256 == $expected_source and
        .executorWallMs > 0 and .executorCpuMs > 0 and
        (.pcmSha256 | test("^[0-9a-f]{64}$")) and
        (.mertSha256 | test("^[0-9a-f]{64}$")) and
        .clampSha256 == .databaseCommitEmbeddingSha256 and
        .clampSha256 == .finalEmbeddingSha256
    ' "$file" >/dev/null
}

validate_local_batch_result() {
    local file="$1" expected_count="${#case_records[@]}" case_record
    local id _source size source_hash _private_relative poweramp_id
    jq -e --arg expected_run "$run_id" --argjson expected_count "$expected_count" \
        --argjson expected_prefetch "$pcm_prefetch_enabled" '
        .schemaVersion == 2 and .runId == $expected_run and
        .runLabel == "full-batch" and .profile == "FULL" and
        .pcmPrefetchEnabled == ($expected_prefetch == 1) and
        .sourceCount == $expected_count and (.tracks | length) == $expected_count and
        .executorWallMs > 0 and .executorCpuMs > 0 and .totalWallMs >= .executorWallMs and
        (.stageTimings | length) > 0 and
        ([.tracks[].ordinal] | sort) == ([range(0; $expected_count)]) and
        all(.tracks[];
            (.pcmSha256 | test("^[0-9a-f]{64}$")) and
            (.mertSha256 | test("^[0-9a-f]{64}$")) and
            (.clampSha256 == .databaseCommitEmbeddingSha256) and
            (.clampSha256 == .finalEmbeddingSha256)
        )
    ' "$file" >/dev/null
    for case_record in "${case_records[@]}"; do
        IFS=$'\t' read -r id _source size source_hash _private_relative poweramp_id \
            <<<"$case_record"
        jq -e --arg id "$id" --arg hash "$source_hash" \
            --argjson size "$size" --argjson poweramp_id "$poweramp_id" '
            [.tracks[] | select(
                .caseId == $id and .sourceSha256 == $hash and
                .sourceByteLength == $size and .powerampFileId == $poweramp_id
            )] | length == 1
        ' "$file" >/dev/null
    done
}

settle_after_recovered_result() {
    local token="$1"
    for _ in $(seq 1 30); do
        adb_device shell dumpsys activity instrumentation | grep -q "$test_package" || break
        sleep 1
    done
    if adb_device shell dumpsys activity instrumentation | grep -q "$test_package"; then
        printf 'completed evidence exists but instrumentation did not exit: %s\n' "$token" >&2
        exit 1
    fi
    if ((foreground_settings)); then adb_device shell input keyevent HOME >/dev/null; fi
    adb_device shell am force-stop --user "$current_user" "$v2_package" >/dev/null
}

run_one() {
    local label="$1" profile="$2" id="$3" source_hash="$4" private_relative="$5" poweramp_id="$6"
    local token="$run_id-$id-$label"
    local run_dir="$log_dir/runs/$token"
    local evidence="$run_dir/evidence.json"
    local private_result="$private_fixture_root/results/$token.json"
    local active_token_file="$log_dir/active-token.txt" instrument_pid status
    mkdir -p -- "$run_dir"
    if [[ -f "$evidence" ]]; then
        validate_local_result "$evidence" "$label" "$profile" "$id" "$source_hash"
        run_as rm -f "$private_result" "$private_result.bak" "$private_result.new" >/dev/null 2>&1 || true
        printf 'Already complete: %s\n' "$token"
        return
    fi
    if run_as test -f "$private_result"; then
        pull_private_result "$private_result" "$evidence"
        validate_local_result "$evidence" "$label" "$profile" "$id" "$source_hash"
        rm -f -- "$active_token_file"
        settle_after_recovered_result "$token"
        printf 'Recovered completed device evidence: %s\n' "$token"
        return
    fi

    if [[ -f "$active_token_file" && "$(cat "$active_token_file")" == "$token" ]] &&
        adb_device shell dumpsys activity instrumentation | grep -q "$test_package"; then
        printf 'Waiting for already-running device invocation: %s\n' "$token"
        monitor_pid_or_private_result "" "$private_result" "$run_dir"
        if run_as test -f "$private_result"; then
            pull_private_result "$private_result" "$evidence"
            validate_local_result "$evidence" "$label" "$profile" "$id" "$source_hash"
            rm -f -- "$active_token_file"
            settle_after_recovered_result "$token"
            return
        fi
    fi

    printf '%s\n' "$token" >"$active_token_file"
    capture_runtime_boundary "$run_dir/before"
    adb_device logcat -c
    timeout "$timeout_seconds" adb -s "$serial" shell am instrument --user "$current_user" -w -r \
        -e class "$test_method" \
        -e v2ProfileRunId "$run_id" \
        -e v2ProfileRunLabel "$label" \
        -e v2ProfileCaseId "$id" \
        -e v2Profile "$profile" \
        -e v2ProfileSourceRelativePath "$private_relative" \
        -e v2ProfileSourceSha256 "$source_hash" \
        -e v2ProfilePowerampFileId "$poweramp_id" \
        "$runner" >"$run_dir/instrumentation.raw.log" 2>&1 &
    instrument_pid=$!
    monitor_pid_or_private_result "$instrument_pid" "$private_result" "$run_dir"
    set +e
    wait "$instrument_pid"
    status=$?
    set -e
    tr -d '\r' <"$run_dir/instrumentation.raw.log" >"$run_dir/instrumentation.log"
    capture_runtime_boundary "$run_dir/after"
    adb_device logcat -d -v threadtime >"$run_dir/logcat.txt"
    ((status == 0)) || {
        printf 'instrumentation status %s for %s\n' "$status" "$token" >&2
        exit 1
    }
    grep -Eq '^OK \(1 test\)[[:space:]]*$' "$run_dir/instrumentation.log"
    grep -Eq '^INSTRUMENTATION_CODE:[[:space:]]*-1[[:space:]]*$' "$run_dir/instrumentation.log"
    if grep -Eqi 'FAILURES!!!|Process crashed|INSTRUMENTATION_(FAILED|ABORTED)|FATAL EXCEPTION|ANR in ' \
        "$run_dir/instrumentation.log" "$run_dir/logcat.txt"; then
        printf 'fatal runtime evidence found for %s\n' "$token" >&2
        exit 1
    fi
    run_as test -f "$private_result"
    pull_private_result "$private_result" "$evidence"
    validate_local_result "$evidence" "$label" "$profile" "$id" "$source_hash"
    rm -f -- "$active_token_file"
    if ((foreground_settings)); then adb_device shell input keyevent HOME >/dev/null; fi
    adb_device shell am force-stop --user "$current_user" "$v2_package" >/dev/null
    printf 'Completed: %s\n' "$token"
}

run_full_batch_once() {
    local label="full-batch" token="$run_id-full-batch"
    local run_dir="$log_dir/runs/$token"
    local evidence="$run_dir/evidence.json"
    local private_result="$private_fixture_root/results/$token.json"
    local active_token_file="$log_dir/active-token.txt" instrument_pid status prefetch_argument=true
    local case_record id _source _size source_hash private_relative poweramp_id index=0
    local -a instrument_args
    mkdir -p -- "$run_dir"
    if [[ -f "$evidence" ]]; then
        validate_local_batch_result "$evidence"
        run_as rm -f "$private_result" "$private_result.bak" "$private_result.new" >/dev/null 2>&1 || true
        printf 'Already complete: %s\n' "$token"
        return
    fi
    if run_as test -f "$private_result"; then
        pull_private_result "$private_result" "$evidence"
        validate_local_batch_result "$evidence"
        rm -f -- "$active_token_file"
        settle_after_recovered_result "$token"
        printf 'Recovered completed device evidence: %s\n' "$token"
        return
    fi

    if [[ -f "$active_token_file" && "$(cat "$active_token_file")" == "$token" ]] &&
        adb_device shell dumpsys activity instrumentation | grep -q "$test_package"; then
        printf 'Waiting for already-running device invocation: %s\n' "$token"
        monitor_pid_or_private_result "" "$private_result" "$run_dir"
        if run_as test -f "$private_result"; then
            pull_private_result "$private_result" "$evidence"
            validate_local_batch_result "$evidence"
            rm -f -- "$active_token_file"
            settle_after_recovered_result "$token"
            return
        fi
    fi

    ((pcm_prefetch_enabled)) || prefetch_argument=false
    instrument_args=(
        -e class "$test_method"
        -e v2ProfileRunId "$run_id"
        -e v2ProfileRunLabel "$label"
        -e v2PcmPrefetchEnabled "$prefetch_argument"
        -e v2BatchCaseCount "${#case_records[@]}"
    )
    for case_record in "${case_records[@]}"; do
        IFS=$'\t' read -r id _source _size source_hash private_relative poweramp_id \
            <<<"$case_record"
        instrument_args+=(
            -e "v2BatchCase${index}Id" "$id"
            -e "v2BatchCase${index}SourceRelativePath" "$private_relative"
            -e "v2BatchCase${index}SourceSha256" "$source_hash"
            -e "v2BatchCase${index}PowerampFileId" "$poweramp_id"
        )
        index=$((index + 1))
    done

    printf '%s\n' "$token" >"$active_token_file"
    capture_runtime_boundary "$run_dir/before"
    adb_device logcat -c
    timeout "$timeout_seconds" adb -s "$serial" shell am instrument --user "$current_user" -w -r \
        "${instrument_args[@]}" "$runner" >"$run_dir/instrumentation.raw.log" 2>&1 &
    instrument_pid=$!
    monitor_pid_or_private_result "$instrument_pid" "$private_result" "$run_dir"
    set +e
    wait "$instrument_pid"
    status=$?
    set -e
    tr -d '\r' <"$run_dir/instrumentation.raw.log" >"$run_dir/instrumentation.log"
    capture_runtime_boundary "$run_dir/after"
    adb_device logcat -d -v threadtime >"$run_dir/logcat.txt"
    ((status == 0)) || {
        printf 'instrumentation status %s for %s\n' "$status" "$token" >&2
        exit 1
    }
    grep -Eq '^OK \(1 test\)[[:space:]]*$' "$run_dir/instrumentation.log"
    grep -Eq '^INSTRUMENTATION_CODE:[[:space:]]*-1[[:space:]]*$' "$run_dir/instrumentation.log"
    if grep -Eqi 'FAILURES!!!|Process crashed|INSTRUMENTATION_(FAILED|ABORTED)|FATAL EXCEPTION|ANR in ' \
        "$run_dir/instrumentation.log" "$run_dir/logcat.txt"; then
        printf 'fatal runtime evidence found for %s\n' "$token" >&2
        exit 1
    fi
    run_as test -f "$private_result"
    pull_private_result "$private_result" "$evidence"
    validate_local_batch_result "$evidence"
    rm -f -- "$active_token_file"
    if ((foreground_settings)); then adb_device shell input keyevent HOME >/dev/null; fi
    adb_device shell am force-stop --user "$current_user" "$v2_package" >/dev/null
    printf 'Completed: %s\n' "$token"
}

if ((full_batch_once)); then
    run_full_batch_once
    batch_evidence="$log_dir/runs/$run_id-full-batch/evidence.json"
    validate_local_batch_result "$batch_evidence"
    jq '{
        schemaVersion,
        runId,
        resultToken,
        profile,
        sourceCount,
        pcmPrefetchEnabled,
        preflightElapsedMs,
        executorWallMs,
        executorCpuMs,
        totalWallMs,
        tracks: [.tracks[] | {
            ordinal, caseId, sourceByteLength, exactSampleCount24k,
            mertWindows, clampSegments, pcmSha256, mertSha256, clampSha256
        }],
        stageTimings
    }' "$batch_evidence" >"$log_dir/batch-benchmark.json"
else
    run_schedule=(
        'full-1 FULL'
        'background-1 BACKGROUND'
        'background-2 BACKGROUND'
        'full-2 FULL'
    )
    for run in "${run_schedule[@]}"; do
        read -r label profile <<<"$run"
        for case_record in "${case_records[@]}"; do
            IFS=$'\t' read -r id _source _size source_hash private_relative poweramp_id \
                <<<"$case_record"
            run_one "$label" "$profile" "$id" "$source_hash" "$private_relative" "$poweramp_id"
        done
    done

    mapfile -t evidence_files < <(find "$log_dir/runs" -name evidence.json -type f | LC_ALL=C sort)
    expected_count=$((4 * $(wc -l <"$log_dir/cases.tsv")))
    ((${#evidence_files[@]} == expected_count)) || {
        printf 'expected %s evidence files, found %s\n' "$expected_count" "${#evidence_files[@]}" >&2
        exit 1
    }
    jq -s '
        def exact_contract: [
            .sourceByteLength, .sourceSha256, .jobSpecId, .workId, .stableTrackSpanId,
            .embeddingSpecId, .modelArtifactSha256, .preprocessingSpecId,
            .exactSampleCount24k, .sourceSampleCount, .decoderName,
            .pcmByteLength, .pcmSha256, .mertWindows, .mertByteLength, .mertSha256,
            .clampSegments, .clampByteLength, .clampSha256,
            .databaseCommitEmbeddingSha256, .finalEmbeddingSha256,
            .databaseContentSha256, .databaseSemanticSha256, .orderedTrackSetSha256,
            .pembByteLength, .pembSha256, .graphByteLength, .graphSha256,
            .graphNodes, .graphNeighborsPerNode, .activationBindingId
        ];
        group_by(.caseId) | map(
            . as $runs |
            ($runs | map(select(.profile == "FULL") | .executorWallMs)) as $full |
            ($runs | map(select(.profile == "BACKGROUND") | .executorWallMs)) as $background |
            {
                caseId: $runs[0].caseId,
                observations: ($runs | length),
                exactOutputEquality: (($runs | map(exact_contract) | unique | length) == 1),
                rawDatabaseContainerHashes: ($runs | map(.databaseFileSha256) | unique),
                fullWallMs: $full,
                backgroundWallMs: $background,
                fullMeanWallMs: ($full | add / length),
                backgroundMeanWallMs: ($background | add / length),
                backgroundToFullWallRatio: (($background | add / length) / ($full | add / length)),
                fullCpuMs: ($runs | map(select(.profile == "FULL") | .executorCpuMs)),
                backgroundCpuMs: ($runs | map(select(.profile == "BACKGROUND") | .executorCpuMs))
            }
        )
    ' "${evidence_files[@]}" >"$log_dir/comparison.json"
    jq -e 'length > 0 and all(.observations == 4 and .exactOutputEquality == true)' \
        "$log_dir/comparison.json" >/dev/null
fi

for case_record in "${case_records[@]}"; do
    IFS=$'\t' read -r _id _source _size _hash private_relative _poweramp_id <<<"$case_record"
    run_as rm -f "files/$private_relative" "files/$private_relative.bak" \
        "files/$private_relative.new"
done
run_as rm -rf "$private_fixture_root/$run_id"
run_as rmdir "$private_fixture_root/results" "$private_fixture_root" \
    files/device_acceptance >/dev/null 2>&1 || true

ANDROID_SERIAL="$serial" "$script_dir/snapshot_device_acceptance.sh" "$log_dir/after"
"$script_dir/compare_device_acceptance.sh" "$log_dir/before" "$log_dir/after" \
    | tee "$log_dir/device-comparison.log"
grep -E 'files/(indexing_v2/generations|mert\.tflite|clamp3_audio\.tflite|clamp3_text\.tflite|sentencepiece\.bpe\.model)' \
    "$log_dir/before/v2/private-file-sha256.txt" >"$log_dir/protected-before.txt"
grep -E 'files/(indexing_v2/generations|mert\.tflite|clamp3_audio\.tflite|clamp3_text\.tflite|sentencepiece\.bpe\.model)' \
    "$log_dir/after/v2/private-file-sha256.txt" >"$log_dir/protected-after.txt"
diff -u "$log_dir/protected-before.txt" "$log_dir/protected-after.txt" \
    >"$log_dir/protected.diff"

(
    cd "$log_dir"
    find . -type f ! -name evidence-sha256.txt -print0 | LC_ALL=C sort -z | xargs -0 sha256sum
) >"$log_dir/evidence-sha256.txt"

trap - EXIT
if ((full_batch_once)); then
    printf 'Isolated FULL batch benchmark passed: %s\n' "$log_dir"
else
    printf 'Full-vs-Background comparison passed: %s\n' "$log_dir"
fi
