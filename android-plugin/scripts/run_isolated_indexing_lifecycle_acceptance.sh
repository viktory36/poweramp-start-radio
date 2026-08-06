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
test_class="com.powerampstartradio.indexing.v2.V2IndexingExecutorLifecycleInstrumentedTest"
mixed_preflight_method="$test_class#realMediaExtractorPartitionsValidAndCorruptRowsBeforeExecution"
unknown_duration_method="$test_class#unknownDurationOrdinaryDecodesOnceFinalizesAndReusesPcmWhileZeroDurationCueIsRejected"
duplicate_failover_method="$test_class#failedFirstDuplicateLocatorFallsThroughToHealthyExactCopyOnRealModels"
lifecycle_method="$test_class#pauseResumeFailureAndProgressAreExactWithoutTouchingTheActiveGeneration"
process_death_arm_method="$test_class#armProcessDeathAfterVerifiedPcmAndWaitForHostKill"
process_death_resume_method="$test_class#resumeAfterHostKilledProcessWithoutDecodingPcmAgain"
process_death_run_id="pd$(date +%Y%m%dT%H%M%S)-$$"
process_death_root="cache/v2-executor-process-death-$process_death_run_id"
process_death_marker="$process_death_root/host-kill-ready"
staging_root="/data/local/tmp/pasr-v2-assets"

usage() {
    cat >&2 <<'EOF'
usage: run_isolated_indexing_lifecycle_acceptance.sh [--log-dir DIRECTORY]

Rebuilds and reinstalls the existing V2 package without clearing data, stages two pinned audio
fixtures, runs mixed-preflight, unknown-duration, duplicate-failover, sandboxed lifecycle, and
physical process-death cohorts, removes the fixtures, and proves that V1, the Poweramp queue, and
V2's active generation/model bytes did not change.
EOF
}

requested_log_dir=""
while (($#)); do
    case "$1" in
        --log-dir)
            (($# >= 2)) && [[ -n "$2" && "$2" != -* ]] || { usage; exit 2; }
            requested_log_dir="$2"
            shift 2
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

for command_name in adb awk date diff find grep sed sha256sum tail tee timeout tr; do
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

current_user="$(adb_device shell am get-current-user | tr -d '\r')"
[[ "$current_user" =~ ^[0-9]+$ ]] || { printf 'cannot determine Android user\n' >&2; exit 1; }
for package_name in "$v1_package" "$v2_package"; do
    adb_device shell pm path --user "$current_user" "$package_name" | grep -q '^package:/' || {
        printf 'required package is absent: %s\n' "$package_name" >&2
        exit 1
    }
done

if [[ -n "$requested_log_dir" ]]; then
    log_dir="$requested_log_dir"
else
    log_dir="$repo_dir/discovery/device-acceptance/$(date +%Y%m%dT%H%M%z)-isolated-indexing-lifecycle"
fi
mkdir -p -- "$log_dir"
if find "$log_dir" -mindepth 1 -print -quit | grep -q .; then
    printf 'log directory must be empty: %s\n' "$log_dir" >&2
    exit 1
fi

before="$log_dir/before"
after="$log_dir/after"
lifecycle_instrumentation_log="$log_dir/instrumentation-lifecycle.log"
mixed_preflight_instrumentation_log="$log_dir/instrumentation-mixed-preflight.log"
unknown_duration_instrumentation_log="$log_dir/instrumentation-unknown-duration.log"
duplicate_failover_instrumentation_log="$log_dir/instrumentation-duplicate-failover.log"
process_death_arm_raw_log="$log_dir/instrumentation-process-death-arm.raw.log"
process_death_arm_log="$log_dir/instrumentation-process-death-arm.log"
process_death_resume_log="$log_dir/instrumentation-process-death-resume.log"
logcat_file="$log_dir/logcat.txt"
created_destinations=()
process_death_arm_host_pid=""

cleanup() {
    local exit_status=$? destination
    set +e
    if ((exit_status != 0)) && run_as test -d "$process_death_root"; then
        run_as find "$process_death_root" -type f \
            >"$log_dir/process-death-failure-files.txt" 2>/dev/null
        adb_device exec-out run-as "$v2_package" tar -cf - "$process_death_root" \
            >"$log_dir/process-death-failure.tar" 2>/dev/null
    fi
    if [[ -n "$process_death_arm_host_pid" ]] && \
        kill -0 "$process_death_arm_host_pid" >/dev/null 2>&1; then
        adb_device shell am force-stop --user "$current_user" "$v2_package" >/dev/null 2>&1
        kill "$process_death_arm_host_pid" >/dev/null 2>&1
        wait "$process_death_arm_host_pid" >/dev/null 2>&1
    fi
    for destination in "${created_destinations[@]}"; do
        run_as rm -f "$destination" "$destination.bak" "$destination.new" >/dev/null 2>&1
    done
    run_as rm -rf "$process_death_root" >/dev/null 2>&1
    run_as rmdir files/device_acceptance/audio_parity/source >/dev/null 2>&1
    run_as rmdir files/device_acceptance/audio_parity/expected >/dev/null 2>&1
    run_as rmdir files/device_acceptance/audio_parity >/dev/null 2>&1
    run_as rmdir files/device_acceptance >/dev/null 2>&1
    set -e
    return "$exit_status"
}
trap cleanup EXIT

private_sha256() {
    run_as sha256sum "$1" | tr -d '\r' | awk '{ print $1 }'
}

private_size() {
    run_as stat -c %s "$1" | tr -d '\r\n'
}

stage_private() {
    local expected_hash="$1" expected_size="$2" source="$3" destination="$4"
    local parent temporary
    local statuses
    parent="${destination%/*}"
    temporary="$parent/.${destination##*/}.acceptance-$$"
    if run_as test -e "$destination"; then
        [[ "$(private_size "$destination")" == "$expected_size" &&
            "$(private_sha256 "$destination")" == "$expected_hash" ]] || {
            printf 'refusing unexpected pre-existing fixture: %s\n' "$destination" >&2
            exit 1
        }
        return
    fi
    adb_device shell stat -c %s "$source" | tr -d '\r\n' | grep -qx "$expected_size"
    adb_device shell sha256sum "$source" | tr -d '\r' | awk '{ print $1 }' |
        grep -qx "$expected_hash"
    run_as mkdir -p "$parent"
    set +e
    adb_device exec-out cat "$source" |
        adb_device shell "run-as $v2_package sh -c 'cat > $temporary'" >/dev/null
    statuses=("${PIPESTATUS[@]}")
    set -e
    ((statuses[0] == 0 && statuses[1] == 0)) || {
        run_as rm -f "$temporary" >/dev/null 2>&1 || true
        printf 'failed to stage %s\n' "$destination" >&2
        exit 1
    }
    [[ "$(private_size "$temporary")" == "$expected_size" &&
        "$(private_sha256 "$temporary")" == "$expected_hash" ]] || {
        run_as rm -f "$temporary" >/dev/null 2>&1 || true
        printf 'private fixture verification failed: %s\n' "$destination" >&2
        exit 1
    }
    run_as chmod 600 "$temporary"
    run_as mv "$temporary" "$destination"
    created_destinations+=("$destination")
}

ANDROID_SERIAL="$serial" "$script_dir/snapshot_device_acceptance.sh" "$before"

"$project_dir/gradlew" --no-daemon -p "$project_dir" \
    -Dkotlin.incremental=false \
    :app:testDebugUnitTest :app:assembleDebug :app:assembleDebugAndroidTest |
    tee "$log_dir/build.log"

adb_device install -r "$target_apk" | tee "$log_dir/install-target.log"
adb_device install -r -t "$test_apk" | tee "$log_dir/install-test.log"
grep -qx 'Success' "$log_dir/install-target.log"
grep -qx 'Success' "$log_dir/install-test.log"

for permission in android.permission.READ_MEDIA_AUDIO android.permission.POST_NOTIFICATIONS; do
    adb_device shell pm grant --user "$current_user" "$v2_package" "$permission" >/dev/null
done

model_records=(
    '34930763bed772d616b124b0103e3759f9cf464d1b193f73fc1367449f32c539|378064368|files/mert.tflite'
    '06b6a17425b7d8ccb5327db7129aa58102cd119502aed00a397bc4b6e7ae20bc|343793912|files/clamp3_audio.tflite'
    '10398ea03ee96e56d4c6970e302ac2061a41d255ed1ef3d730001ec17ca0fb16|1113080740|files/clamp3_text.tflite'
    'cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865|5069051|files/sentencepiece.bpe.model'
)
for record in "${model_records[@]}"; do
    IFS='|' read -r expected_hash expected_size destination <<<"$record"
    [[ "$(private_size "$destination")" == "$expected_size" &&
        "$(private_sha256 "$destination")" == "$expected_hash" ]] || {
        printf 'installed V2 model does not match the pinned cohort: %s\n' "$destination" >&2
        exit 1
    }
done

stage_private \
    ce0da2f5e54bab63482c0731d73a174ef683ede8e961967dd08f0540238bbfdf \
    1294386 \
    "$staging_root/audio_parity/source/vice_city_interlude_3.flac" \
    files/device_acceptance/audio_parity/source/vice_city_interlude_3.flac
stage_private \
    66d73a6272f476ce231e46bc7b7e9f1f91d10735d37bd96af454f391e6e68f14 \
    3072 \
    "$staging_root/audio_parity/expected/vice_city_interlude_3.f32le" \
    files/device_acceptance/audio_parity/expected/vice_city_interlude_3.f32le

adb_device logcat -c
set +e
adb_device shell am instrument --user "$current_user" -w -r \
    -e class "$mixed_preflight_method" "$runner" 2>&1 | tr -d '\r' |
    tee "$mixed_preflight_instrumentation_log"
statuses=("${PIPESTATUS[@]}")
set -e
((statuses[0] == 0 && statuses[1] == 0 && statuses[2] == 0)) || exit 1
grep -Eq '^OK \(1 test\)[[:space:]]*$' "$mixed_preflight_instrumentation_log"
grep -Eq '^INSTRUMENTATION_CODE:[[:space:]]*-1[[:space:]]*$' \
    "$mixed_preflight_instrumentation_log"
! grep -Eqi 'FAILURES!!!|Process crashed|INSTRUMENTATION_(FAILED|ABORTED)|FATAL EXCEPTION|ANR in ' \
    "$mixed_preflight_instrumentation_log"

set +e
adb_device shell am instrument --user "$current_user" -w -r \
    -e class "$unknown_duration_method" "$runner" 2>&1 | tr -d '\r' |
    tee "$unknown_duration_instrumentation_log"
statuses=("${PIPESTATUS[@]}")
set -e
((statuses[0] == 0 && statuses[1] == 0 && statuses[2] == 0)) || exit 1
grep -Eq '^OK \(1 test\)[[:space:]]*$' "$unknown_duration_instrumentation_log"
grep -Eq '^INSTRUMENTATION_CODE:[[:space:]]*-1[[:space:]]*$' \
    "$unknown_duration_instrumentation_log"
! grep -Eqi 'FAILURES!!!|Process crashed|INSTRUMENTATION_(FAILED|ABORTED)|FATAL EXCEPTION|ANR in ' \
    "$unknown_duration_instrumentation_log"

set +e
adb_device shell am instrument --user "$current_user" -w -r \
    -e class "$duplicate_failover_method" "$runner" 2>&1 | tr -d '\r' |
    tee "$duplicate_failover_instrumentation_log"
statuses=("${PIPESTATUS[@]}")
set -e
((statuses[0] == 0 && statuses[1] == 0 && statuses[2] == 0)) || exit 1
grep -Eq '^OK \(1 test\)[[:space:]]*$' "$duplicate_failover_instrumentation_log"
grep -Eq '^INSTRUMENTATION_CODE:[[:space:]]*-1[[:space:]]*$' \
    "$duplicate_failover_instrumentation_log"
! grep -Eqi 'FAILURES!!!|Process crashed|INSTRUMENTATION_(FAILED|ABORTED)|FATAL EXCEPTION|ANR in ' \
    "$duplicate_failover_instrumentation_log"

set +e
adb_device shell am instrument --user "$current_user" -w -r \
    -e class "$lifecycle_method" "$runner" 2>&1 | tr -d '\r' | tee "$lifecycle_instrumentation_log"
statuses=("${PIPESTATUS[@]}")
set -e
((statuses[0] == 0 && statuses[1] == 0 && statuses[2] == 0)) || exit 1
grep -Eq '^OK \(1 test\)[[:space:]]*$' "$lifecycle_instrumentation_log"
grep -Eq '^INSTRUMENTATION_CODE:[[:space:]]*-1[[:space:]]*$' "$lifecycle_instrumentation_log"
! grep -Eqi 'FAILURES!!!|Process crashed|INSTRUMENTATION_(FAILED|ABORTED)|FATAL EXCEPTION|ANR in ' \
    "$lifecycle_instrumentation_log"

set +e
timeout 150 adb -s "$serial" shell am instrument --user "$current_user" -w -r \
    -e class "$process_death_arm_method" \
    -e v2ProcessDeathRunId "$process_death_run_id" \
    "$runner" >"$process_death_arm_raw_log" 2>&1 &
process_death_arm_host_pid=$!
set -e

marker_ready=false
for ((poll = 0; poll < 480; poll++)); do
    if run_as test -f "$process_death_marker"; then
        marker_ready=true
        break
    fi
    if ! kill -0 "$process_death_arm_host_pid" >/dev/null 2>&1; then
        break
    fi
    sleep 0.25
done
if [[ "$marker_ready" != true ]]; then
    set +e
    wait "$process_death_arm_host_pid"
    set -e
    tr -d '\r' <"$process_death_arm_raw_log" | tee "$process_death_arm_log"
    printf 'process-death arm phase exited before publishing its ready marker\n' >&2
    exit 1
fi

marker_contents="$(run_as cat "$process_death_marker" | tr -d '\r')"
marker_value() {
    local key="$1"
    printf '%s\n' "$marker_contents" | sed -n "s/^${key}=//p"
}
for key in schema run_id root_name job_id pid lease_epoch lease_owner ledger_revision \
    ledger_state track_state track_checkpoint verified_artifact_count work_id pcm_sha256 \
    pcm_bytes pcm_path protected_pointer_sha256; do
    [[ "$(printf '%s\n' "$marker_contents" | grep -c "^${key}=")" == 1 ]] || {
        printf 'invalid process-death marker key: %s\n' "$key" >&2
        exit 1
    }
done
[[ "$(marker_value schema)" == 1 ]]
[[ "$(marker_value run_id)" == "$process_death_run_id" ]]
[[ "$(marker_value root_name)" == "${process_death_root#cache/}" ]]
[[ "$(marker_value job_id)" == "acceptance-process-death-$process_death_run_id" ]]
[[ "$(marker_value ledger_state)" == RUNNING ]]
[[ "$(marker_value track_state)" == DECODING ]]
[[ "$(marker_value track_checkpoint)" == PREFLIGHTED ]]
[[ "$(marker_value verified_artifact_count)" == 0 ]]
[[ "$(marker_value pcm_sha256)" =~ ^[0-9a-f]{64}$ ]]
[[ "$(marker_value protected_pointer_sha256)" =~ ^[0-9a-f]{64}$ ]]
process_death_old_pid="$(marker_value pid)"
[[ "$process_death_old_pid" =~ ^[1-9][0-9]*$ ]]
target_pids="$(adb_device shell pidof "$v2_package" | tr -d '\r')"
[[ " $target_pids " == *" $process_death_old_pid "* ]] || {
    printf 'armed pid %s is not a live %s process: %s\n' \
        "$process_death_old_pid" "$v2_package" "$target_pids" >&2
    exit 1
}

adb_device shell am force-stop --user "$current_user" "$v2_package"
old_pid_gone=false
for ((poll = 0; poll < 80; poll++)); do
    if ! adb_device shell test -d "/proc/$process_death_old_pid"; then
        old_pid_gone=true
        break
    fi
    sleep 0.25
done
[[ "$old_pid_gone" == true ]] || {
    printf 'force-stop did not kill armed pid %s\n' "$process_death_old_pid" >&2
    exit 1
}
set +e
wait "$process_death_arm_host_pid"
process_death_arm_status=$?
set -e
tr -d '\r' <"$process_death_arm_raw_log" | tee "$process_death_arm_log"
! grep -Eq '^OK \([0-9]+ tests?\)[[:space:]]*$' "$process_death_arm_log"
printf 'Physical process death observed: pid=%s arm_status=%s\n' \
    "$process_death_old_pid" "$process_death_arm_status"

set +e
adb_device shell am instrument --user "$current_user" -w -r \
    -e class "$process_death_resume_method" \
    -e v2ProcessDeathRunId "$process_death_run_id" \
    "$runner" 2>&1 | tr -d '\r' | tee "$process_death_resume_log"
statuses=("${PIPESTATUS[@]}")
set -e
((statuses[0] == 0 && statuses[1] == 0 && statuses[2] == 0)) || exit 1
grep -Eq '^OK \(1 test\)[[:space:]]*$' "$process_death_resume_log"
grep -Eq '^INSTRUMENTATION_CODE:[[:space:]]*-1[[:space:]]*$' "$process_death_resume_log"
! grep -Eqi 'FAILURES!!!|Process crashed|INSTRUMENTATION_(FAILED|ABORTED)|FATAL EXCEPTION|ANR in ' \
    "$process_death_resume_log"
adb_device logcat -d -v threadtime >"$logcat_file"

cleanup
created_destinations=()
ANDROID_SERIAL="$serial" "$script_dir/snapshot_device_acceptance.sh" "$after"
"$script_dir/compare_device_acceptance.sh" "$before" "$after" |
    tee "$log_dir/comparison.log"

grep -E 'files/(indexing_v2/generations|mert\.tflite|clamp3_audio\.tflite|clamp3_text\.tflite|sentencepiece\.bpe\.model)' \
    "$before/v2/private-file-sha256.txt" >"$log_dir/protected-before.txt"
grep -E 'files/(indexing_v2/generations|mert\.tflite|clamp3_audio\.tflite|clamp3_text\.tflite|sentencepiece\.bpe\.model)' \
    "$after/v2/private-file-sha256.txt" >"$log_dir/protected-after.txt"
diff -u "$log_dir/protected-before.txt" "$log_dir/protected-after.txt" \
    >"$log_dir/protected.diff"

host_apk_sha="$(sha256sum "$target_apk" | awk '{ print $1 }')"
device_apk_sha="$(awk 'NR == 1 { print $1 }' "$after/v2/apk-sha256.txt")"
[[ "$host_apk_sha" == "$device_apk_sha" ]]
! adb_device shell dumpsys activity services "$v2_package" | grep -q 'ServiceRecord'
! grep -Eiq 'FATAL EXCEPTION|ANR in com\.powerampstartradio\.v2' "$logcat_file"
lifecycle_summary="$(
    grep 'V2_INDEXING_LIFECYCLE' "$logcat_file" | tail -n 1 |
        sed 's/^.*V2_INDEXING_LIFECYCLE /V2_INDEXING_LIFECYCLE /'
)"
[[ "$lifecycle_summary" == V2_INDEXING_LIFECYCLE\ * ]]
unknown_duration_summary="$(
    grep 'V2_UNKNOWN_DURATION' "$logcat_file" | tail -n 1 |
        sed 's/^.*V2_UNKNOWN_DURATION /V2_UNKNOWN_DURATION /'
)"
[[ "$unknown_duration_summary" == V2_UNKNOWN_DURATION\ * ]]
duplicate_failover_summary="$(
    grep 'V2_DUPLICATE_FAILOVER' "$logcat_file" | tail -n 1 |
        sed 's/^.*V2_DUPLICATE_FAILOVER /V2_DUPLICATE_FAILOVER /'
)"
[[ "$duplicate_failover_summary" == V2_DUPLICATE_FAILOVER\ * ]]
process_death_summary="$(
    grep 'V2_PROCESS_DEATH' "$logcat_file" | tail -n 1 |
        sed 's/^.*V2_PROCESS_DEATH /V2_PROCESS_DEATH /'
)"
[[ "$process_death_summary" == V2_PROCESS_DEATH\ * ]]
[[ "$process_death_summary" == *"old_pid=$process_death_old_pid "* ]]

{
    printf 'serial=%s\n' "$serial"
    printf 'test_class=%s\n' "$test_class"
    printf 'unknown_duration_method=%s\n' "$unknown_duration_method"
    printf 'duplicate_failover_method=%s\n' "$duplicate_failover_method"
    printf 'lifecycle_method=%s\n' "$lifecycle_method"
    printf 'process_death_arm_method=%s\n' "$process_death_arm_method"
    printf 'process_death_resume_method=%s\n' "$process_death_resume_method"
    printf 'process_death_run_id=%s\n' "$process_death_run_id"
    printf 'target_apk_sha256=%s\n' "$host_apk_sha"
    printf 'poweramp_queue_sha256=%s\n' "$(sha256sum "$after/poweramp-queue.txt" | awk '{ print $1 }')"
    printf '%s\n' "$unknown_duration_summary"
    printf '%s\n' "$duplicate_failover_summary"
    printf '%s\n' "$lifecycle_summary"
    printf '%s\n' "$process_death_summary"
} >"$log_dir/result.txt"

trap - EXIT
printf 'Isolated indexing lifecycle acceptance passed: %s\n' "$log_dir"
