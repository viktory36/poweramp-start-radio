#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
comparator="$script_dir/compare_device_acceptance.sh"
tmp_dir="$(mktemp -d)"
trap 'rm -rf -- "$tmp_dir"' EXIT

hash_a="$(printf 'a%.0s' {1..64})"
hash_b="$(printf 'b%.0s' {1..64})"
hash_c="$(printf 'c%.0s' {1..64})"
hash_d="$(printf 'd%.0s' {1..64})"

seal() {
    local directory="$1"
    local manifest_tmp
    manifest_tmp="$(mktemp)"
    (
        cd "$directory"
        rm -f snapshot-sha256.txt
        find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum
    ) >"$manifest_tmp"
    mv -- "$manifest_tmp" "$directory/snapshot-sha256.txt"
}

write_snapshot() {
    local directory="$1"
    local v2_status="$2"
    local widget_hash="$3"
    mkdir -p "$directory/v1" "$directory/v2"
    cat >"$directory/device.txt" <<'EOF'
captured_at=2026-07-14T06:00:00+03:00
serial=fixture-device
fingerprint=fixture/device/build
android_release=16
android_sdk=36
EOF
    cat >"$directory/poweramp-queue.txt" <<'EOF'
Row: 0 queue._id=1, queue.folder_file_id=99, queue.sort=0
EOF
    printf 'installed\n' >"$directory/v1/status.txt"
    printf '%s  /data/app/v1/base.apk\n' "$hash_a" >"$directory/v1/apk-sha256.txt"
    cat >"$directory/v1/runtime-grants.txt" <<'EOF'
user=0 android.permission.POST_NOTIFICATIONS: granted=true, flags=[ USER_SET]
user=0 android.permission.READ_MEDIA_AUDIO: granted=true, flags=[ USER_SET]
EOF
    cat >"$directory/v1/private-files.txt" <<'EOF'
./files/embeddings.db
./files/session_history.json
./shared_prefs/settings.xml
./shared_prefs/widget.xml
EOF
    {
        printf '%s  ./files/embeddings.db\n' "$hash_b"
        printf '%s  ./files/session_history.json\n' "$hash_c"
        printf '%s  ./shared_prefs/settings.xml\n' "$widget_hash"
        printf '%s  ./shared_prefs/widget.xml\n' "$widget_hash"
    } >"$directory/v1/private-file-sha256.txt"
    printf '%s\n' "$v2_status" >"$directory/v2/status.txt"
    seal "$directory"
}

before="$tmp_dir/before"
after="$tmp_dir/after"
write_snapshot "$before" "not installed" "$hash_a"
write_snapshot "$after" "installed" "$hash_d"

pass_output="$tmp_dir/pass-output"
if ! "$comparator" --expect-v2-transition "$before" "$after" >"$pass_output" 2>&1; then
    cat "$pass_output" >&2
    exit 1
fi
grep -q 'changed \[report-only-preferences\]: ./shared_prefs/settings.xml' "$pass_output"
grep -q 'changed \[report-only-volatile-preferences\]: ./shared_prefs/widget.xml' "$pass_output"
grep -q 'ACCEPTANCE COMPARISON PASSED' "$pass_output"

expect_failure() {
    local name="$1"
    local expected="$2"
    local scenario="$tmp_dir/$name"
    cp -a "$after" "$scenario"
    rm -f "$scenario/snapshot-sha256.txt"
    "$name"_mutation "$scenario"
    seal "$scenario"
    if "$comparator" --expect-v2-transition "$before" "$scenario" \
        >"$tmp_dir/$name.output" 2>&1; then
        printf 'expected comparator failure for %s\n' "$name" >&2
        exit 1
    fi
    grep -q "$expected" "$tmp_dir/$name.output"
}

apk_mutation() {
    printf '%s  /data/app/v1/base.apk\n' "$hash_d" >"$1/v1/apk-sha256.txt"
}

queue_mutation() {
    printf 'Row: 0 queue._id=2, queue.folder_file_id=100, queue.sort=0\n' \
        >"$1/poweramp-queue.txt"
}

grants_mutation() {
    sed -i 's/READ_MEDIA_AUDIO: granted=true/READ_MEDIA_AUDIO: granted=false/' \
        "$1/v1/runtime-grants.txt"
}

uninstalled_mutation() {
    printf 'not installed\n' >"$1/v1/status.txt"
}

protected_mutation() {
    sed -i "s/^$hash_b  \.\/files\/embeddings\.db/$hash_d  .\/files\/embeddings.db/" \
        "$1/v1/private-file-sha256.txt"
}

v2_missing_mutation() {
    printf 'not installed\n' >"$1/v2/status.txt"
}

expect_failure apk 'V1 APK hash set changed'
expect_failure queue 'Poweramp queue projection changed'
expect_failure grants 'V1 runtime grants changed'
expect_failure uninstalled 'V1 became uninstalled'
expect_failure protected 'protected V1 private artifact changed'
expect_failure v2_missing 'V2 was expected to be installed'

printf 'compare_device_acceptance fixture tests passed\n'
