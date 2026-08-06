#!/usr/bin/env bash
set -euo pipefail
umask 077

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
package_name="com.powerampstartradio.v2"
activity="$package_name/com.powerampstartradio.debug.FeatureAcceptanceActivity"
run_id="${RUN_ID:-$(date +%Y%m%dT%H%M%S%z)-$(printf '%04x' "$((RANDOM % 65536))") }"
run_id="${run_id// /}"
output_dir="${1:-$repo_root/discovery/device-acceptance/${run_id}-feature-acceptance}"
suite="${SUITE:-all}"
selection_matrix="${SELECTION_MATRIX:-current_v2_defaults_extremes}"
seed_track_ids="${SEED_TRACK_IDS:-}"
queries="${QUERIES:-ambient|sleep|guitar|psychedelic}"
queries_b64="$(printf '%s' "$queries" | base64 -w0)"
num_tracks="${NUM_TRACKS:-30}"
repeat_count="${REPEAT_COUNT:-2}"
timeout_seconds="${TIMEOUT_SECONDS:-3600}"
poll_seconds="${POLL_SECONDS:-5}"
adb_serial="${ANDROID_SERIAL:-$(adb get-serialno)}"

if [[ -z "$adb_serial" || "$adb_serial" == "unknown" ]]; then
    printf 'Set ANDROID_SERIAL to one connected device.\n' >&2
    exit 2
fi
if [[ -z "$seed_track_ids" ]]; then
    printf 'Set SEED_TRACK_IDS to one or more comma-separated embedded track IDs.\n' >&2
    exit 2
fi
if [[ -e "$output_dir" ]]; then
    printf 'Output path already exists: %s\n' "$output_dir" >&2
    exit 2
fi
mkdir -p -- "$output_dir"

adb_device() {
    adb -s "$adb_serial" "$@"
}

wait_for_device() {
    local remaining="$1"
    while (( remaining > 0 )); do
        if [[ "$(adb_device get-state 2>/dev/null | tr -d '\r' || true)" == "device" ]]; then
            return 0
        fi
        sleep 2
        remaining=$((remaining - 2))
    done
    return 1
}

queue_snapshot() {
    adb_device shell content query \
        --uri content://com.maxmpz.audioplayer.data/queue \
        --projection queue._id:queue.folder_file_id:queue.sort \
        | tr -d '\r'
}

if ! wait_for_device 30; then
    printf 'ADB target is not in device state: %s\n' "$adb_serial" >&2
    exit 2
fi
if ! adb_device shell pm path "$package_name" | grep -q '^package:'; then
    printf '%s is not installed. Build and install the V2 debug APK first.\n' "$package_name" >&2
    exit 2
fi
if adb_device shell dumpsys activity services "$package_name" \
    | grep -q 'com.powerampstartradio.indexing.IndexingService'; then
    printf 'V2 indexing is active; feature acceptance requires an idle immutable generation.\n' >&2
    exit 2
fi

# The activity intentionally starts work only from onCreate. A completed instance may still be
# Android's top activity, in which case `am start` merely delivers a new intent and performs no
# run. Cold-start the disposable V2 package so every host request has exactly one execution.
adb_device shell am force-stop "$package_name"
sleep 1

{
    printf 'run_id=%s\n' "$run_id"
    printf 'captured_at=%s\n' "$(date --iso-8601=seconds)"
    printf 'serial=%s\n' "$adb_serial"
    printf 'suite=%s\n' "$suite"
    printf 'selection_matrix=%s\n' "$selection_matrix"
    printf 'seed_track_ids=%s\n' "$seed_track_ids"
    printf 'queries=%s\n' "$queries"
    printf 'num_tracks=%s\n' "$num_tracks"
    printf 'repeat_count=%s\n' "$repeat_count"
    printf 'apk='; adb_device shell pm path "$package_name" | tr -d '\r'
} >"$output_dir/request.txt"

queue_snapshot >"$output_dir/poweramp-queue-before.txt"
adb_device shell dumpsys battery >"$output_dir/battery-before.txt"
adb_device shell dumpsys thermalservice >"$output_dir/thermal-before.txt"
start_epoch="$(date +%s)"

adb_device shell am start -W -n "$activity" \
    --es run_id "$run_id" \
    --es suite "$suite" \
    --es selection_matrix "$selection_matrix" \
    --es seed_track_ids "$seed_track_ids" \
    --es queries_b64 "$queries_b64" \
    --ei num_tracks "$num_tracks" \
    --ei repeat_count "$repeat_count" \
    >"$output_dir/am-start.txt"

deadline=$((SECONDS + timeout_seconds))
last_summary=""
state="RUNNING"
while (( SECONDS < deadline )); do
    if ! wait_for_device 30; then
        printf 'Waiting for ADB target %s to return...\n' "$adb_serial"
        continue
    fi

    tmp_status="$output_dir/.status.tmp"
    if adb_device exec-out run-as "$package_name" \
        cat files/feature_acceptance/status.json >"$tmp_status" 2>/dev/null &&
        jq -e --arg run_id "$run_id" '.runId == $run_id' "$tmp_status" >/dev/null 2>&1; then
        mv -- "$tmp_status" "$output_dir/status.json"
        state="$(jq -r '.state' "$output_dir/status.json")"
        summary="$(jq -r \
            '"state=\(.state) selection=\(.selectionRuns | length) text=\(.textRuns | length) updated=\(.updatedAt)"' \
            "$output_dir/status.json")"
        if [[ "$summary" != "$last_summary" ]]; then
            printf '%s\n' "$summary"
            last_summary="$summary"
        fi
        if [[ "$state" == "COMPLETE" || "$state" == "FAILED" ]]; then
            break
        fi
    else
        rm -f -- "$tmp_status"
    fi
    sleep "$poll_seconds"
done

if [[ "$state" == "RUNNING" ]]; then
    printf 'Timed out after %ss; partial status was retained.\n' "$timeout_seconds" >&2
fi

if wait_for_device 30; then
    adb_device exec-out run-as "$package_name" \
        cat "files/feature_acceptance/$run_id.json" \
        >"$output_dir/report.json" 2>"$output_dir/report-pull-error.txt" || true
    if [[ -s "$output_dir/status.json" ]]; then
        catalog_name="$(jq -r '.activeCatalog.completeTsvFile // empty' "$output_dir/status.json")"
        if [[ -n "$catalog_name" ]]; then
            adb_device exec-out run-as "$package_name" \
                cat "files/feature_acceptance/$catalog_name" \
                >"$output_dir/active-catalog.tsv" 2>"$output_dir/catalog-pull-error.txt" || true
        fi
    fi
    queue_snapshot >"$output_dir/poweramp-queue-after.txt"
    adb_device shell dumpsys battery >"$output_dir/battery-after.txt"
    adb_device shell dumpsys thermalservice >"$output_dir/thermal-after.txt"
    adb_device shell dumpsys meminfo "$package_name" >"$output_dir/meminfo-after.txt"
    adb_device logcat -d -v epoch \
        FeatureAcceptance:I RecommendationEngine:I TrackMatcher:I Clamp3TextInference:I \
        GeoMeanSelector:I GraphExplorer:I '*:S' \
        | awk -v start="$start_epoch" '$1 + 0 >= start' \
        >"$output_dir/logcat.txt" || true
fi

if [[ ! -s "$output_dir/poweramp-queue-after.txt" ]]; then
    printf 'Could not capture the final Poweramp queue.\n' >&2
    exit 1
fi
if ! cmp -s "$output_dir/poweramp-queue-before.txt" "$output_dir/poweramp-queue-after.txt"; then
    diff -u "$output_dir/poweramp-queue-before.txt" \
        "$output_dir/poweramp-queue-after.txt" >"$output_dir/poweramp-queue.diff" || true
    printf 'FAIL: Poweramp queue changed during a declared read-only run.\n' >&2
    exit 1
fi
printf 'queue_unchanged=true\n' >"$output_dir/invariants.txt"

if [[ "$state" == "COMPLETE" && ! -s "$output_dir/report.json" ]]; then
    printf 'FAIL: completed run has no final report.\n' >&2
    exit 1
fi
if [[ -s "$output_dir/report.json" ]]; then
    jq -e --arg run_id "$run_id" '
        .runId == $run_id and
        .state == "COMPLETE" and
        .queueMutationApisCalled == 0 and
        (.request.repeatCount | type) == "number" and
        (.plannedSelectionRunCount == (.selectionRuns | length)) and
        (
            if .request.includeSelection then
                ([.plannedSelectionCases[] | { key: .id, value: .config }] | from_entries) as $planned |
                (.plannedSelectionCases | length) > 0 and
                ($planned | length) == (.plannedSelectionCases | length) and
                all(.selectionRuns[];
                    .error == null and
                    .resultFingerprint != null and
                    $planned[.caseId] == .config
                ) and
                (
                    .request.repeatCount as $repeat_count |
                    .selectionRuns |
                    sort_by(.caseId, .seedTrackId) |
                    group_by([.caseId, .seedTrackId]) |
                    map(
                        length == $repeat_count and
                        ([.[].repeat] | sort) == [range(1; $repeat_count + 1)] and
                        ([.[].resultFingerprint] | unique | length) == 1
                    ) |
                    all
                )
            else
                (.plannedSelectionCases | length) == 0 and
                (.selectionRuns | length) == 0
            end
        )
    ' "$output_dir/report.json" >"$output_dir/acceptance-validation.json"
fi

(
    cd "$output_dir"
    find . -type f ! -name sha256.txt -print0 \
        | LC_ALL=C sort -z \
        | xargs -0 sha256sum
) >"$output_dir/sha256.txt"

printf 'Feature acceptance evidence: %s\n' "$output_dir"
if [[ "$state" != "COMPLETE" ]]; then
    exit 1
fi
