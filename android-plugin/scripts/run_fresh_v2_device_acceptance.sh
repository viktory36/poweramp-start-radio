#!/usr/bin/env bash
set -euo pipefail
umask 077

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"
target_apk="$project_dir/app/build/outputs/apk/debug/app-debug.apk"
test_apk="$project_dir/app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk"
staging_root="/data/local/tmp/pasr-v2-assets"

v1_package="com.powerampstartradio"
v2_package="com.powerampstartradio.v2"
test_package="com.powerampstartradio.v2.test"
runner="com.powerampstartradio.v2.test/androidx.test.runner.AndroidJUnitRunner"

runtime_permissions=(
    android.permission.READ_MEDIA_AUDIO
    android.permission.POST_NOTIFICATIONS
)

pre_import_classes=(
    com.powerampstartradio.poweramp.PowerampTrackIdentityInstrumentedTest
    com.powerampstartradio.indexing.V2IndexingServiceIntentInstrumentedTest
    com.powerampstartradio.indexing.v2.V2IndexingPreflightIntentStoreInstrumentedTest
    com.powerampstartradio.indexing.v2.AtomicV2ArtifactStoreInstrumentedTest
    com.powerampstartradio.indexing.v2.V2EmbeddingCommitRepositoryInstrumentedTest
    com.powerampstartradio.indexing.v2.V2BootstrapGenerationImporterInstrumentedTest
    com.powerampstartradio.data.EmbeddingIndexNativeInstrumentedTest
    com.powerampstartradio.indexing.OfficialSentencePieceTokenizerInstrumentedTest
    com.powerampstartradio.indexing.TextEmbeddingParityInstrumentedTest
    com.powerampstartradio.indexing.v2.V2PowerampProviderSnapshotInstrumentedTest
    com.powerampstartradio.indexing.AudioEmbeddingParityInstrumentedTest
)
provider_snapshot_class="com.powerampstartradio.indexing.v2.V2PowerampProviderSnapshotInstrumentedTest"
import_class="com.powerampstartradio.indexing.v2.V2FrozenDatabaseImportAcceptanceTest"
post_import_classes=(
    com.powerampstartradio.similarity.ClosestRecommendationInstrumentedTest
    com.powerampstartradio.similarity.algorithms.GraphExplorerNativeInstrumentedTest
    com.powerampstartradio.data.ActiveDomainGraphTopologyInstrumentedTest
    com.powerampstartradio.similarity.ActiveDomainRecommendationEngineInstrumentedTest
)
total_class_count=16
asset_total_bytes=0

# SHA-256|byte length|staged source relative to staging_root|private destination.
# Keeping this closed list in the runner prevents an unreviewed device fixture from entering V2.
asset_records=(
    "34930763bed772d616b124b0103e3759f9cf464d1b193f73fc1367449f32c539|378064368|models/mert.tflite|files/mert.tflite"
    "06b6a17425b7d8ccb5327db7129aa58102cd119502aed00a397bc4b6e7ae20bc|343793912|models/clamp3_audio.tflite|files/clamp3_audio.tflite"
    "10398ea03ee96e56d4c6970e302ac2061a41d255ed1ef3d730001ec17ca0fb16|1113080740|models/clamp3_text.tflite|files/clamp3_text.tflite"
    "cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865|5069051|models/sentencepiece.bpe.model|files/sentencepiece.bpe.model"
    "9d872fb702e3c091358e09cd31ec7b1a25736e95d06db31611acf4770b0cd4e3|5010|text_parity/manifest.json|files/device_acceptance/text_parity/manifest.json"
    "0e7e40f34cdca86046ed2a4c9b7af242b1af686cadc6bb843787070f6b2cae29|3072|text_parity/short_english.f32le|files/device_acceptance/text_parity/short_english.f32le"
    "77d988d589e4ad5b9e67d6b0b174cbe10f56c9038fe60e0487ca600268605963|3072|text_parity/short_arabic.f32le|files/device_acceptance/text_parity/short_arabic.f32le"
    "68610f8cf6990c300136e1b6a438e96baf5430d2bed010a3d2cdf9284a8be289|3072|text_parity/long_multisection.f32le|files/device_acceptance/text_parity/long_multisection.f32le"
    "5b6430f8e9fd39eb28503d4e970797f021051b4f398a4b25cb08256ebd278f26|4582|audio_parity/manifest.json|files/device_acceptance/audio_parity/manifest.json"
    "43922a38cb556e49cee69a5eb6cbb55d4d94ddf6d66277bc6f5882051017be33|5348656|audio_parity/source/abdullah_miniawy_signature.flac|files/device_acceptance/audio_parity/source/abdullah_miniawy_signature.flac"
    "4ba909053f5e47dd663172d549a097cc45a614f0e557e868f2fe35f5373d9d43|713302|audio_parity/source/daniela_andrade_angel19.flac|files/device_acceptance/audio_parity/source/daniela_andrade_angel19.flac"
    "ce0da2f5e54bab63482c0731d73a174ef683ede8e961967dd08f0540238bbfdf|1294386|audio_parity/source/vice_city_interlude_3.flac|files/device_acceptance/audio_parity/source/vice_city_interlude_3.flac"
    "3907ac88565c995a3d6f9633c407692d479efb71f6d486222f3b750625e8faff|3072|audio_parity/expected/abdullah_miniawy_signature.f32le|files/device_acceptance/audio_parity/expected/abdullah_miniawy_signature.f32le"
    "f6731c3df53e986c34021a32b98ce324f75b970b1c0f1892eb1cd5c15ddc93b9|3072|audio_parity/expected/daniela_andrade_angel19.f32le|files/device_acceptance/audio_parity/expected/daniela_andrade_angel19.f32le"
    "66d73a6272f476ce231e46bc7b7e9f1f91d10735d37bd96af454f391e6e68f14|3072|audio_parity/expected/vice_city_interlude_3.f32le|files/device_acceptance/audio_parity/expected/vice_city_interlude_3.f32le"
    "08dfcec60f7c2e9de4bc6b923d601bd824f80b6251769f6c7bcd8062ce6aa504|380243968|frozen/embeddings.db|files/device_acceptance/embeddings.db"
    "65dafdae5e713f3913e6d6f082612813f6859d37d35ec9e2cdcc43f06c077656|3860216|graph-explorer-benchmark.bin|files/graph-explorer-benchmark.bin"
)

closed_plan_error() {
    printf 'invalid closed acceptance plan: %s\n' "$*" >&2
    exit 1
}

validate_closed_plan() {
    local record hash size source destination extra class_name
    local -A seen_sources=() seen_destinations=() seen_classes=()

    ((${#runtime_permissions[@]} == 2)) ||
        closed_plan_error "expected exactly two runtime permissions"
    [[ "${runtime_permissions[0]}" == "android.permission.READ_MEDIA_AUDIO" &&
        "${runtime_permissions[1]}" == "android.permission.POST_NOTIFICATIONS" ]] ||
        closed_plan_error "runtime permission allowlist or order changed"
    ((${#asset_records[@]} == 17)) || closed_plan_error "expected exactly 17 assets"
    ((${#pre_import_classes[@]} == 11)) ||
        closed_plan_error "expected exactly 11 pre-import classes"
    ((${#post_import_classes[@]} == 4)) ||
        closed_plan_error "expected exactly four post-import classes"
    ((${#pre_import_classes[@]} + 1 + ${#post_import_classes[@]} == total_class_count)) ||
        closed_plan_error "class count does not equal $total_class_count"

    for record in "${asset_records[@]}"; do
        IFS='|' read -r hash size source destination extra <<<"$record"
        [[ -z "$extra" && "$hash" =~ ^[0-9a-f]{64}$ && "$size" =~ ^[1-9][0-9]*$ ]] ||
            closed_plan_error "malformed asset record"
        [[ "$source" =~ ^[A-Za-z0-9._/-]+$ && "$source" != /* &&
            "/$source/" != *"/../"* && "/$source/" != *"/./"* ]] ||
            closed_plan_error "unsafe staged source: $source"
        [[ "$destination" =~ ^files/[A-Za-z0-9._/-]+$ &&
            "/$destination/" != *"/../"* && "/$destination/" != *"/./"* ]] ||
            closed_plan_error "unsafe private destination: $destination"
        [[ -z "${seen_sources[$source]+present}" ]] ||
            closed_plan_error "duplicate staged source: $source"
        [[ -z "${seen_destinations[$destination]+present}" ]] ||
            closed_plan_error "duplicate private destination: $destination"
        seen_sources["$source"]=1
        seen_destinations["$destination"]=1
        asset_total_bytes=$((asset_total_bytes + size))
    done

    for class_name in "${pre_import_classes[@]}" "$import_class" "${post_import_classes[@]}"; do
        [[ "$class_name" =~ ^com\.powerampstartradio\.[A-Za-z0-9_.]+InstrumentedTest$ ||
            "$class_name" == "$import_class" ]] ||
            closed_plan_error "unsafe instrumentation class: $class_name"
        [[ -z "${seen_classes[$class_name]+present}" ]] ||
            closed_plan_error "duplicate instrumentation class: $class_name"
        seen_classes["$class_name"]=1
    done
    [[ -n "${seen_classes[$provider_snapshot_class]+present}" ]] ||
        closed_plan_error "the allowlisted provider-read class is missing"
}

validate_closed_plan

usage() {
    cat >&2 <<'EOF'
usage: run_fresh_v2_device_acceptance.sh [--dry-run] [--log-dir DIRECTORY]

Runs the closed, 16-class connected acceptance sequence against a fresh V2 install.
The APK paths, device staging root, private destinations, hashes, and test classes are
fixed in this script. Arbitrary instrumentation classes and positional arguments are
intentionally not accepted.

Options:
  --dry-run          Print the immutable asset and test plan without invoking ADB.
  --log-dir DIR      Empty host directory for install and per-class evidence logs.
  -h, --help         Show this help.

Set ANDROID_SERIAL when more than one ADB transport is present.
EOF
}

print_plan() {
    local record hash size source destination index class_name
    printf 'Target APK: %s\n' "$target_apk"
    printf 'Test APK:   %s\n' "$test_apk"
    printf 'Staging:    %s\n' "$staging_root"
    printf 'Runtime permission grants (%d, fixed order):\n' "${#runtime_permissions[@]}"
    for permission in "${runtime_permissions[@]}"; do
        printf '  %s\n' "$permission"
    done
    printf 'Assets (%d):\n' "${#asset_records[@]}"
    for record in "${asset_records[@]}"; do
        IFS='|' read -r hash size source destination <<<"$record"
        printf '  %s/%s -> %s (%s bytes, sha256 %s)\n' \
            "$staging_root" "$source" "$destination" "$size" "$hash"
    done
    printf 'Instrumentation classes (%d, fixed order):\n' "$total_class_count"
    index=1
    for class_name in "${pre_import_classes[@]}"; do
        printf '  %02d [pre-import]  %s\n' "$index" "$class_name"
        index=$((index + 1))
    done
    printf '  %02d [import]      %s\n' "$index" "$import_class"
    index=$((index + 1))
    for class_name in "${post_import_classes[@]}"; do
        printf '  %02d [post-import] %s\n' "$index" "$class_name"
        index=$((index + 1))
    done
}

dry_run=0
requested_log_dir=""
while (($# > 0)); do
    case "$1" in
        --dry-run)
            dry_run=1
            shift
            ;;
        --log-dir)
            if (($# < 2)) || [[ -z "$2" || "$2" == -* ]]; then
                printf '%s\n' '--log-dir requires a non-empty directory' >&2
                usage
                exit 2
            fi
            requested_log_dir="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --class|-e|--)
            printf 'arbitrary instrumentation arguments are refused: %s\n' "$1" >&2
            usage
            exit 2
            ;;
        -*|*)
            printf 'unknown or positional argument refused: %s\n' "$1" >&2
            usage
            exit 2
            ;;
    esac
done

if ((dry_run)); then
    print_plan
    exit 0
fi

env_file="$project_dir/.android-wsl-env"
load_android_tool_environment() {
    local line name value path_record=0
    local literal_export_pattern='^export[[:space:]]+(JAVA_HOME|ANDROID_SDK_ROOT|ANDROID_HOME)="([^"]+)"$'
    local expected_path_record
    local -a lines=()
    local -A values=()

    expected_path_record="export PATH=\"\$JAVA_HOME/bin:\$ANDROID_SDK_ROOT/cmdline-tools/latest/bin:\$ANDROID_SDK_ROOT/platform-tools:\$PATH\""

    [[ -f "$env_file" ]] || return 0
    mapfile -t lines < <(awk 'NF { print }' "$env_file")
    ((${#lines[@]} == 4)) || {
        printf 'refusing unexpected Android environment file shape: %s\n' "$env_file" >&2
        exit 1
    }
    for line in "${lines[@]}"; do
        if [[ "$line" =~ $literal_export_pattern ]]; then
            name="${BASH_REMATCH[1]}"
            value="${BASH_REMATCH[2]}"
            [[ "$value" == /* && "$value" != *$'\n'* && "$value" != *$'\r'* ]] || {
                printf 'refusing unsafe %s in %s\n' "$name" "$env_file" >&2
                exit 1
            }
            [[ -z "${values[$name]+present}" ]] || {
                printf 'refusing duplicate %s in %s\n' "$name" "$env_file" >&2
                exit 1
            }
            values["$name"]="$value"
        elif [[ "$line" == "$expected_path_record" ]]; then
            path_record=$((path_record + 1))
        else
            printf 'refusing executable or unexpected line in %s: %s\n' "$env_file" "$line" >&2
            exit 1
        fi
    done
    [[ -n "${values[JAVA_HOME]+present}" &&
        -n "${values[ANDROID_SDK_ROOT]+present}" &&
        -n "${values[ANDROID_HOME]+present}" &&
        "${values[ANDROID_HOME]}" == "${values[ANDROID_SDK_ROOT]}" &&
        "$path_record" == 1 ]] || {
        printf 'refusing incomplete or inconsistent Android environment file: %s\n' "$env_file" >&2
        exit 1
    }
    export JAVA_HOME="${values[JAVA_HOME]}"
    export ANDROID_SDK_ROOT="${values[ANDROID_SDK_ROOT]}"
    export ANDROID_HOME="${values[ANDROID_HOME]}"
    export PATH="$JAVA_HOME/bin:$ANDROID_SDK_ROOT/cmdline-tools/latest/bin:$ANDROID_SDK_ROOT/platform-tools:$PATH"
}

load_android_tool_environment

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        printf 'missing required host command: %s\n' "$1" >&2
        exit 1
    fi
}

for command_name in adb apkanalyzer awk date find grep sha256sum sort tee tr; do
    require_command "$command_name"
done

[[ -x "$project_dir/gradlew" ]] || {
    printf 'missing executable Gradle wrapper: %s\n' "$project_dir/gradlew" >&2
    exit 1
}
printf 'Rebuilding the target and instrumentation APKs from the current reviewed source...\n'
"$project_dir/gradlew" --no-daemon -p "$project_dir" \
    :app:assembleDebug :app:assembleDebugAndroidTest

if [[ ! -f "$target_apk" || ! -f "$test_apk" ]]; then
    printf 'build both debug APKs before acceptance:\n  %s\n  %s\n' \
        "$target_apk" "$test_apk" >&2
    exit 1
fi

if [[ "$(apkanalyzer manifest application-id "$target_apk")" != "$v2_package" ]]; then
    printf 'target APK is not package %s\n' "$v2_package" >&2
    exit 1
fi
if [[ "$(apkanalyzer manifest application-id "$test_apk")" != "$test_package" ]]; then
    printf 'test APK is not package %s\n' "$test_package" >&2
    exit 1
fi
target_sdk="$(apkanalyzer manifest target-sdk "$target_apk")"
if [[ "$target_sdk" != "36" ]]; then
    printf 'target APK SDK is %s, but this acceptance requires exactly 36\n' "$target_sdk" >&2
    exit 1
fi
test_manifest="$(apkanalyzer manifest print "$test_apk")"
if ! grep -q 'android:name="androidx.test.runner.AndroidJUnitRunner"' <<<"$test_manifest" ||
    ! grep -q 'android:targetPackage="com.powerampstartradio.v2"' <<<"$test_manifest"; then
    printf 'test APK does not declare the pinned V2 AndroidJUnitRunner target\n' >&2
    exit 1
fi
target_permissions="$(apkanalyzer manifest permissions "$target_apk")"
for permission in "${runtime_permissions[@]}"; do
    if ! grep -Fxq "$permission" <<<"$target_permissions"; then
        printf 'target APK does not request required runtime permission: %s\n' \
            "$permission" >&2
        exit 1
    fi
done

if [[ -n "$requested_log_dir" ]]; then
    log_dir="$requested_log_dir"
else
    log_dir="$project_dir/build/device-acceptance/$(date -u +%Y%m%dT%H%M%SZ)"
fi
mkdir -p -- "$log_dir"
if find "$log_dir" -mindepth 1 -print -quit | grep -q .; then
    printf 'log directory must be empty: %s\n' "$log_dir" >&2
    exit 1
fi
runner_log="$log_dir/runner.log"
results_log="$log_dir/results.tsv"
evidence_manifest="$log_dir/evidence-sha256.tsv"
: >"$runner_log"
printf 'sequence\tphase\tclass\tlog_sha256\n' >"$results_log"

finalize_evidence() {
    local exit_status=$? evidence_status=0 path relative digest
    local temporary_manifest="$log_dir/.evidence-sha256.tsv.tmp"
    trap - EXIT
    set +e
    : >"$temporary_manifest"
    while IFS= read -r -d '' path; do
        relative="${path#"$log_dir"/}"
        digest="$(sha256sum -- "$path")" || evidence_status=1
        digest="${digest%% *}"
        if [[ "$digest" =~ ^[0-9a-f]{64}$ ]]; then
            printf '%s\t%s\n' "$digest" "$relative" >>"$temporary_manifest"
        else
            printf 'warning: could not hash evidence file: %s\n' "$path" >&2
            evidence_status=1
        fi
    done < <(
        find "$log_dir" -maxdepth 1 -type f \
            ! -name "${evidence_manifest##*/}" \
            ! -name "${temporary_manifest##*/}" -print0 |
            LC_ALL=C sort -z
    )
    mv -- "$temporary_manifest" "$evidence_manifest" || evidence_status=1
    if ((exit_status == 0 && evidence_status != 0)); then
        printf 'failed to produce complete host evidence hashes\n' >&2
        exit 1
    fi
    exit "$exit_status"
}
trap finalize_evidence EXIT

log() {
    printf '%s\n' "$*" | tee -a "$runner_log"
}

fail() {
    printf 'FAILED: %s\n' "$*" | tee -a "$runner_log" >&2
    {
        printf 'failed_at=%s\n' "$(date --iso-8601=seconds)"
        printf 'reason=%s\n' "$*"
    } >"$log_dir/FAILURE"
    exit 1
}

devices_output="$(adb devices)" || fail "adb devices failed"
mapfile -t transports < <(
    tr -d '\r' <<<"$devices_output" |
        awk -F '\t' 'NF >= 2 && $1 != "List of devices attached" { print $1 "|" $2 }'
)

if [[ -n "${ANDROID_SERIAL:-}" ]]; then
    adb_serial="$ANDROID_SERIAL"
    if [[ "$adb_serial" == *[[:space:]]* ]]; then
        fail "ANDROID_SERIAL contains whitespace"
    fi
    selected_state=""
    for transport in "${transports[@]}"; do
        IFS='|' read -r candidate state <<<"$transport"
        if [[ "$candidate" == "$adb_serial" ]]; then
            selected_state="$state"
        fi
    done
    [[ -n "$selected_state" ]] || fail "ANDROID_SERIAL is not an attached transport: $adb_serial"
    [[ "$selected_state" == "device" ]] ||
        fail "ANDROID_SERIAL is not ready: $adb_serial ($selected_state)"
else
    ((${#transports[@]} == 1)) ||
        fail "expected exactly one ADB transport; set ANDROID_SERIAL to select one"
    IFS='|' read -r adb_serial selected_state <<<"${transports[0]}"
    [[ "$selected_state" == "device" ]] ||
        fail "the sole ADB transport is not ready: $adb_serial ($selected_state)"
fi

adb_device() {
    adb -s "$adb_serial" "$@"
}

run_as() {
    adb_device shell run-as "$v2_package" "$@"
}

if [[ "$(adb_device get-state)" != "device" ]]; then
    fail "ADB target is not in device state: $adb_serial"
fi

package_installed() {
    local package_name="$1" paths
    paths="$(
        adb_device shell pm path --user "$current_user" "$package_name" 2>/dev/null |
            tr -d '\r' || true
    )"
    grep -q '^package:/' <<<"$paths"
}

android_sdk="$(adb_device shell getprop ro.build.version.sdk | tr -d '\r')"
[[ "$android_sdk" =~ ^[0-9]+$ ]] || fail "cannot read Android SDK level"
((android_sdk >= 33)) || fail "connected acceptance requires Android 13/API 33 or newer"
current_user="$(adb_device shell am get-current-user | tr -d '\r')"
[[ "$current_user" =~ ^[0-9]+$ ]] || fail "cannot read the current Android user"

# The private fixture copy is 2.08 GiB. Require room for a second full copy plus 1 GiB
# before installing, which covers active-generation publication and atomic scratch files.
required_data_bytes=$((asset_total_bytes * 2 + 1073741824))
required_data_kib=$(((required_data_bytes + 1023) / 1024))
available_data_kib="$(
    adb_device shell df -k /data |
        tr -d '\r' |
        awk 'NR > 1 && $4 ~ /^[0-9]+$/ { value = $4 } END { print value }'
)"
[[ "$available_data_kib" =~ ^[0-9]+$ ]] || fail "cannot read free space for /data"
((available_data_kib >= required_data_kib)) ||
    fail "insufficient /data space: ${available_data_kib}KiB free, ${required_data_kib}KiB required"

package_installed "$v1_package" ||
    fail "V1 must be installed for Android user $current_user before acceptance"
if package_installed "$v2_package"; then
    fail "V2 must be absent for Android user $current_user; this runner accepts only a fresh V2"
fi

{
    printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
    printf 'serial=%s\n' "$adb_serial"
    printf 'android_sdk=%s\n' "$android_sdk"
    printf 'target_sdk=%s\n' "$target_sdk"
    printf 'android_user=%s\n' "$current_user"
    printf 'asset_total_bytes=%s\n' "$asset_total_bytes"
    printf 'required_data_kib=%s\n' "$required_data_kib"
    printf 'available_data_kib=%s\n' "$available_data_kib"
    printf 'target_apk_sha256=%s\n' "$(sha256sum "$target_apk" | awk '{ print $1 }')"
    printf 'test_apk_sha256=%s\n' "$(sha256sum "$test_apk" | awk '{ print $1 }')"
} >"$log_dir/run-metadata.txt"

remote_sha256() {
    local path="$1" output hash
    output="$(adb_device exec-out sha256sum "$path" | tr -d '\r')" || return 1
    read -r hash _ <<<"$output"
    [[ "$hash" =~ ^[0-9a-f]{64}$ ]] || return 1
    printf '%s\n' "$hash"
}

remote_size() {
    adb_device exec-out stat -c %s "$1" | tr -d '\r\n'
}

verify_staged_assets() {
    local evidence_file="$log_dir/staged-assets.tsv"
    local record expected_hash expected_size source destination actual_hash actual_size

    printf 'source\tdestination\tbytes\tsha256\n' >"$evidence_file"
    for record in "${asset_records[@]}"; do
        IFS='|' read -r expected_hash expected_size source destination <<<"$record"
        source="$staging_root/$source"
        actual_size="$(remote_size "$source")" || fail "missing staged asset: $source"
        [[ "$actual_size" == "$expected_size" ]] ||
            fail "staged byte length mismatch for $source: $actual_size != $expected_size"
        actual_hash="$(remote_sha256 "$source")" || fail "cannot hash staged asset: $source"
        [[ "$actual_hash" == "$expected_hash" ]] ||
            fail "staged SHA-256 mismatch for $source"
        printf '%s\t%s\t%s\t%s\n' \
            "$source" "$destination" "$actual_size" "$actual_hash" >>"$evidence_file"
    done
}

log "Preflighting all ${#asset_records[@]} pinned staged assets before installation"
verify_staged_assets

run_logged_command() {
    local output_file="$1"
    shift
    local statuses
    set +e
    "$@" 2>&1 | tr -d '\r' | tee "$output_file"
    statuses=("${PIPESTATUS[@]}")
    set -e
    if ((statuses[0] != 0 || statuses[1] != 0 || statuses[2] != 0)); then
        fail "command failed; see $output_file"
    fi
}

log "Installing the pinned fresh V2 target APK"
run_logged_command "$log_dir/install-target.log" \
    adb_device install "$target_apk"
grep -qx 'Success' "$log_dir/install-target.log" || fail "target install did not report Success"

log "Installing the pinned V2 instrumentation APK"
run_logged_command "$log_dir/install-test.log" \
    adb_device install -r -t "$test_apk"
grep -qx 'Success' "$log_dir/install-test.log" || fail "test install did not report Success"

package_installed "$v1_package" || fail "V1 disappeared during V2 installation"
package_installed "$v2_package" || fail "V2 target package was not installed"
package_installed "$test_package" || fail "V2 test package was not installed"

for permission in "${runtime_permissions[@]}"; do
    adb_device shell pm grant --user "$current_user" "$v2_package" "$permission" \
        >/dev/null
done

extract_runtime_grants() {
    awk '
        function leading_spaces(value, copy) {
            copy = value
            sub(/[^ ].*$/, "", copy)
            return length(copy)
        }
        /^[[:space:]]*User [0-9]+:/ {
            line = $0
            sub(/^[[:space:]]*User /, "", line)
            sub(/:.*/, "", line)
            user = line
            in_runtime = 0
            next
        }
        /^[[:space:]]*runtime permissions:[[:space:]]*$/ {
            in_runtime = 1
            runtime_indent = leading_spaces($0)
            next
        }
        in_runtime {
            line = $0
            indent = leading_spaces(line)
            sub(/^[[:space:]]*/, "", line)
            if (line ~ /^[A-Za-z0-9._]+: granted=(true|false)(,|$)/) {
                print "user=" (user == "" ? "unknown" : user) " " line
                next
            }
            if (line != "" && indent <= runtime_indent) in_runtime = 0
        }
    '
}

verify_runtime_grants() {
    local evidence_name="$1" runtime_grants grant permission allowed
    local -a granted_permissions=()
    local -A verified_grants=()

    runtime_grants="$(
        adb_device shell dumpsys package "$v2_package" |
            tr -d '\r' |
            extract_runtime_grants
    )" || fail "could not read V2 runtime-permission evidence"
    [[ -n "$runtime_grants" ]] || fail "V2 runtime-permission evidence is empty"
    printf '%s\n' "$runtime_grants" >"$log_dir/$evidence_name"
    mapfile -t granted_permissions < <(
        awk -v prefix="user=$current_user " \
            'index($0, prefix) == 1 && $0 ~ /: granted=true(,|$)/ { print }' \
            <<<"$runtime_grants"
    )
    ((${#granted_permissions[@]} == ${#runtime_permissions[@]})) ||
        fail "V2 granted runtime-permission set is not exactly the two-item allowlist"
    for grant in "${granted_permissions[@]}"; do
        if [[ ! "$grant" =~ ^user=${current_user}[[:space:]]+([^:]+):[[:space:]]granted=true(,|$) ]]; then
            fail "malformed V2 granted runtime-permission evidence: $grant"
        fi
        permission="${BASH_REMATCH[1]}"
        allowed=0
        for expected_permission in "${runtime_permissions[@]}"; do
            [[ "$permission" == "$expected_permission" ]] && allowed=1
        done
        ((allowed)) || fail "unexpected granted V2 runtime permission: $permission"
        [[ -z "${verified_grants[$permission]+present}" ]] ||
            fail "duplicate V2 runtime-permission evidence: $permission"
        verified_grants["$permission"]=1
    done
    for permission in "${runtime_permissions[@]}"; do
        [[ -n "${verified_grants[$permission]+present}" ]] ||
            fail "missing granted V2 runtime-permission evidence: $permission"
    done
}

verify_runtime_grants "v2-runtime-grants-after-install.txt"
log "Verified exactly READ_MEDIA_AUDIO and POST_NOTIFICATIONS for V2 user $current_user"

existing_private_files="$(run_as find . -type f | tr -d '\r')" ||
    fail "cannot inspect the fresh V2 private directory"
if [[ -n "$existing_private_files" ]]; then
    printf '%s\n' "$existing_private_files" >"$log_dir/unexpected-fresh-v2-files.txt"
    fail "fresh V2 contains restored or pre-existing private files"
fi

private_sha256() {
    local path="$1" output hash
    output="$(adb_device exec-out run-as "$v2_package" sha256sum "$path" | tr -d '\r')" ||
        return 1
    read -r hash _ <<<"$output"
    [[ "$hash" =~ ^[0-9a-f]{64}$ ]] || return 1
    printf '%s\n' "$hash"
}

private_size() {
    adb_device exec-out run-as "$v2_package" stat -c %s "$1" | tr -d '\r\n'
}

copy_asset_atomically() {
    local expected_hash="$1" expected_size="$2" relative_source="$3" destination="$4"
    local source="$staging_root/$relative_source"
    local parent="${destination%/*}"
    local temporary="$parent/.${destination##*/}.acceptance-tmp-$$"
    local actual_hash actual_size copy_status

    run_as mkdir -p "$parent" >/dev/null
    if run_as test -e "$destination"; then
        fail "fresh V2 destination already exists: $destination"
    fi
    run_as rm -f "$temporary" >/dev/null

    # Prefer a same-device copy. The binary-safe ADB relay is used only when SELinux
    # does not let the run-as domain read the shell-owned staging directory directly.
    if adb_device shell "run-as $v2_package test -r $source" >/dev/null 2>&1; then
        if ! adb_device shell \
            "run-as $v2_package sh -c 'cat $source > $temporary'" >/dev/null; then
            run_as rm -f "$temporary" >/dev/null || true
            fail "direct staged copy failed for $source"
        fi
    else
        set +e
        adb_device exec-out cat "$source" |
            adb_device shell "run-as $v2_package sh -c 'cat > $temporary'" >/dev/null
        copy_status=("${PIPESTATUS[@]}")
        set -e
        if ((copy_status[0] != 0 || copy_status[1] != 0)); then
            run_as rm -f "$temporary" >/dev/null || true
            fail "binary ADB relay failed for $source"
        fi
    fi

    actual_size="$(private_size "$temporary")" || {
        run_as rm -f "$temporary" >/dev/null || true
        fail "cannot inspect private temporary asset: $temporary"
    }
    [[ "$actual_size" == "$expected_size" ]] || {
        run_as rm -f "$temporary" >/dev/null || true
        fail "private temporary byte length mismatch for $destination"
    }
    actual_hash="$(private_sha256 "$temporary")" || {
        run_as rm -f "$temporary" >/dev/null || true
        fail "cannot hash private temporary asset: $temporary"
    }
    [[ "$actual_hash" == "$expected_hash" ]] || {
        run_as rm -f "$temporary" >/dev/null || true
        fail "private temporary SHA-256 mismatch for $destination"
    }

    run_as chmod 600 "$temporary" >/dev/null
    run_as mv "$temporary" "$destination" >/dev/null
    log "Staged $destination ($expected_size bytes, sha256 $expected_hash)"
}

verify_private_assets() {
    local evidence_name="$1" evidence_file
    local record expected_hash expected_size source destination actual_hash actual_size

    evidence_file="$log_dir/$evidence_name"

    printf 'destination\tbytes\tsha256\n' >"$evidence_file"
    for record in "${asset_records[@]}"; do
        IFS='|' read -r expected_hash expected_size source destination <<<"$record"
        actual_size="$(private_size "$destination")" ||
            fail "cannot inspect private asset: $destination"
        actual_hash="$(private_sha256 "$destination")" ||
            fail "cannot hash private asset: $destination"
        [[ "$actual_size" == "$expected_size" ]] ||
            fail "private byte length mismatch for $destination: $actual_size != $expected_size"
        [[ "$actual_hash" == "$expected_hash" ]] ||
            fail "private SHA-256 mismatch for $destination"
        printf '%s\t%s\t%s\n' "$destination" "$actual_size" "$actual_hash" \
            >>"$evidence_file"
    done
}

log "Verifying and atomically publishing ${#asset_records[@]} pinned V2 assets"
for record in "${asset_records[@]}"; do
    IFS='|' read -r hash size source destination <<<"$record"
    copy_asset_atomically "$hash" "$size" "$source" "$destination"
done
verify_private_assets "private-assets-before-tests.tsv"
log "Reverified all ${#asset_records[@]} published V2 assets"

active_pointer="files/indexing_v2/generations/active-generation.json"
if run_as test -e "$active_pointer"; then
    fail "an active V2 generation exists before the stateful import"
fi

run_instrumentation_class() {
    local sequence="$1" phase="$2" class_name="$3"
    local simple_name="${class_name##*.}"
    local output_file
    local statuses log_hash
    output_file="$log_dir/$(printf '%02d' "$sequence")-$simple_name.log"

    log "Running $(printf '%02d' "$sequence")/$total_class_count [$phase] $class_name"
    set +e
    adb_device shell am instrument --user "$current_user" -w -r \
        -e class "$class_name" "$runner" 2>&1 |
        tr -d '\r' |
        tee "$output_file"
    statuses=("${PIPESTATUS[@]}")
    set -e
    if ((statuses[0] != 0 || statuses[1] != 0 || statuses[2] != 0)); then
        poweramp_consent_handoff "$class_name" "$output_file"
        fail "instrumentation transport failed for $class_name; see $output_file"
    fi
    if grep -Eq '^INSTRUMENTATION_STATUS_CODE:[[:space:]]*-[0-9]+[[:space:]]*$' \
        "$output_file" ||
        grep -Eqi \
            'FAILURES!!!|INSTRUMENTATION_(FAILED|ABORTED)|Process crashed|shortMsg=|ANR in |FATAL EXCEPTION|Test run failed to complete' \
            "$output_file"; then
        poweramp_consent_handoff "$class_name" "$output_file"
        fail "instrumentation reported a failure or skipped test for $class_name"
    fi
    [[ "$(grep -Ec '^OK \([1-9][0-9]* tests?\)[[:space:]]*$' "$output_file")" == 1 ]] || {
        poweramp_consent_handoff "$class_name" "$output_file"
        fail "instrumentation did not report a non-empty OK result for $class_name"
    }
    [[ "$(grep -Ec '^INSTRUMENTATION_CODE:[[:space:]]*-1[[:space:]]*$' "$output_file")" == 1 ]] || {
        poweramp_consent_handoff "$class_name" "$output_file"
        fail "instrumentation did not finish normally for $class_name"
    }

    package_installed "$v1_package" || fail "V1 disappeared while running $class_name"
    log_hash="$(sha256sum "$output_file" | awk '{ print $1 }')"
    printf '%s\t%s\t%s\t%s\n' "$sequence" "$phase" "$class_name" "$log_hash" \
        >>"$results_log"
}

poweramp_consent_handoff() {
    local class_name="$1" output_file="$2"
    [[ "$class_name" == "$provider_snapshot_class" ]] || return 0
    log "POWERAMP CONSENT HANDOFF: the sole allowlisted live Poweramp provider-read gate failed."
    log "If $output_file reports SecurityException, POWERAMP_PERMISSION_DENIED, or denied provider access, leave V2 and this evidence in place."
    log "On the phone, open V2 Settings, choose Poweramp library access / Grant access, and approve the Poweramp-owned prompt."
    log "Do not change battery optimization, alter the queue, or grant any other Android permission; after approval, return with this evidence before attempting a fresh rerun."
}

sequence=1
for class_name in "${pre_import_classes[@]}"; do
    run_instrumentation_class "$sequence" pre-import "$class_name"
    sequence=$((sequence + 1))
done

if run_as test -e "$active_pointer"; then
    fail "a supposedly isolated pre-import class activated a V2 generation"
fi

log "The next class deliberately activates the pinned frozen V2 generation"
run_instrumentation_class "$sequence" import "$import_class"
sequence=$((sequence + 1))
run_as test -f "$active_pointer" || fail "frozen import did not publish an active generation"

for class_name in "${post_import_classes[@]}"; do
    run_instrumentation_class "$sequence" post-import "$class_name"
    sequence=$((sequence + 1))
done
((sequence == total_class_count + 1)) || fail "internal class-count invariant failed"

unexpected_indexing_state="$(
    run_as find files/indexing_v2/jobs files/indexing_v2/preflight-intents -type f \
        2>/dev/null | tr -d '\r' || true
)"
[[ -z "$unexpected_indexing_state" ]] ||
    fail "acceptance unexpectedly created durable indexing work"
package_installed "$v1_package" || fail "V1 is not installed after acceptance"
verify_runtime_grants "v2-runtime-grants-final.txt"
verify_private_assets "private-assets-after-tests.tsv"
log "Final runtime grants and all ${#asset_records[@]} private asset pins reverified"

{
    printf 'completed_at=%s\n' "$(date --iso-8601=seconds)"
    printf 'classes_passed=%s\n' "$total_class_count"
} >>"$log_dir/run-metadata.txt"
printf 'PASS\n' >"$log_dir/SUCCESS"
log "CONNECTED ACCEPTANCE PASSED: $total_class_count/$total_class_count classes"
log "Evidence: $log_dir"
log "No AFTER snapshot was captured; run the read-only snapshot/comparator separately."
log "A final SHA-256 manifest for every host evidence file is written on exit."
