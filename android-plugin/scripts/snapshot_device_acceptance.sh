#!/usr/bin/env bash
set -euo pipefail
umask 077

# Read-only before/after evidence for isolated V2 device acceptance.
output_dir="${1:-}"
if [[ -z "$output_dir" ]]; then
    printf 'usage: %s OUTPUT_DIRECTORY\n' "$0" >&2
    exit 2
fi

adb_serial="${ANDROID_SERIAL:-$(adb get-serialno)}"
if [[ -z "$adb_serial" || "$adb_serial" == "unknown" ]]; then
    printf 'set ANDROID_SERIAL to one connected device\n' >&2
    exit 2
fi

mkdir -p -- "$output_dir"
if find "$output_dir" -mindepth 1 -print -quit | grep -q .; then
    printf 'output directory must be empty: %s\n' "$output_dir" >&2
    exit 2
fi

adb_device() {
    adb -s "$adb_serial" "$@"
}

if [[ "$(adb_device get-state)" != "device" ]]; then
    printf 'ADB target is not in device state: %s\n' "$adb_serial" >&2
    exit 2
fi

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
    ' | LC_ALL=C sort -u
}

snapshot_package() {
    local package_name="$1"
    local label="$2"
    local package_dir="$output_dir/$label"
    mkdir -p "$package_dir"

    if ! adb_device shell pm path "$package_name" \
        | tr -d '\r' \
        | awk '/^package:\// { print }' \
        >"$package_dir/package-path.txt" ||
        ! grep -q '^package:' "$package_dir/package-path.txt"; then
        printf 'not installed\n' >"$package_dir/status.txt"
        return
    fi

    printf 'installed\n' >"$package_dir/status.txt"
    : >"$package_dir/apk-sha256.txt"
    while IFS= read -r apk_path; do
        adb_device exec-out sha256sum "${apk_path#package:}" \
            >>"$package_dir/apk-sha256.txt"
    done <"$package_dir/package-path.txt"
    adb_device shell dumpsys package "$package_name" \
        | tr -d '\r' \
        >"$package_dir/dumpsys-package.txt"
    extract_runtime_grants \
        <"$package_dir/dumpsys-package.txt" \
        >"$package_dir/runtime-grants.txt"
    if [[ ! -s "$package_dir/runtime-grants.txt" ]]; then
        printf 'no runtime permission evidence for installed package: %s\n' \
            "$package_name" >&2
        exit 1
    fi
    adb_device exec-out run-as "$package_name" find . -type f \
        | tr -d '\r' \
        | awk '/^\.\/(files|shared_prefs|databases|no_backup)\//' \
        | LC_ALL=C sort \
        >"$package_dir/private-files.txt"
    if grep -q '[[:space:]]' "$package_dir/private-files.txt"; then
        printf 'private filename contains unsupported whitespace\n' >&2
        exit 1
    fi
    : >"$package_dir/private-file-sha256.txt"
    while IFS= read -r file; do
        adb_device exec-out run-as "$package_name" sha256sum "$file" \
            >>"$package_dir/private-file-sha256.txt"
    done <"$package_dir/private-files.txt"
}

{
    printf 'captured_at=%s\n' "$(date --iso-8601=seconds)"
    printf 'serial=%s\n' "$adb_serial"
    printf 'fingerprint='
    adb_device shell getprop ro.build.fingerprint | tr -d '\r'
    printf 'android_release='
    adb_device shell getprop ro.build.version.release | tr -d '\r'
    printf 'android_sdk='
    adb_device shell getprop ro.build.version.sdk | tr -d '\r'
} >"$output_dir/device.txt"

snapshot_package com.powerampstartradio v1
snapshot_package com.powerampstartradio.v2 v2

adb_device shell content query \
    --uri content://com.maxmpz.audioplayer.data/queue \
    --projection queue._id:queue.folder_file_id:queue.sort \
    | tr -d '\r' \
    >"$output_dir/poweramp-queue.txt"

snapshot_manifest_tmp="$(mktemp)"
trap 'rm -f -- "$snapshot_manifest_tmp"' EXIT
(
    cd "$output_dir"
    find . -type f ! -name snapshot-sha256.txt -print0 \
        | LC_ALL=C sort -z \
        | xargs -0 sha256sum
) >"$snapshot_manifest_tmp"
mv -- "$snapshot_manifest_tmp" "$output_dir/snapshot-sha256.txt"
trap - EXIT

printf 'Read-only snapshot written to %s\n' "$output_dir"
