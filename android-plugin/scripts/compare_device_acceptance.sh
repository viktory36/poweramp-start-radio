#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<'EOF'
usage: compare_device_acceptance.sh [OPTIONS] BEFORE_DIRECTORY AFTER_DIRECTORY

Options:
  --expect-v2-transition       Require V2 absent before and installed after.
  --expect-v2-absent-before    Require V2 absent in the BEFORE snapshot.
  --expect-v2-installed-after  Require V2 installed in the AFTER snapshot.
EOF
}

expect_v2_absent_before=0
expect_v2_installed_after=0
while (($# > 0)); do
    case "$1" in
        --expect-v2-transition)
            expect_v2_absent_before=1
            expect_v2_installed_after=1
            shift
            ;;
        --expect-v2-absent-before)
            expect_v2_absent_before=1
            shift
            ;;
        --expect-v2-installed-after)
            expect_v2_installed_after=1
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        -*)
            printf 'unknown option: %s\n' "$1" >&2
            usage
            exit 2
            ;;
        *) break ;;
    esac
done

if (($# != 2)); then
    usage
    exit 2
fi

before_dir="${1%/}"
after_dir="${2%/}"
if [[ ! -d "$before_dir" || ! -d "$after_dir" ]]; then
    printf 'both snapshot directories must exist\n' >&2
    exit 2
fi

tmp_dir="$(mktemp -d)"
trap 'rm -rf -- "$tmp_dir"' EXIT
failures=0

fail() {
    printf 'FAIL: %s\n' "$*" >&2
    failures=$((failures + 1))
}

report() {
    printf 'REPORT: %s\n' "$*"
}

require_file() {
    local file="$1"
    if [[ ! -f "$file" ]]; then
        fail "missing evidence file: $file"
        return 1
    fi
}

verify_snapshot_integrity() {
    local directory="$1"
    local label="$2"
    local manifest="$directory/snapshot-sha256.txt"
    if ! require_file "$manifest"; then
        return
    fi
    if ! awk '
        NF != 2 || length($1) != 64 || $1 !~ /^[0-9a-f]+$/ ||
            $2 !~ /^\.\// || $2 ~ /(^|\/)\.\.($|\/)/ ||
            $2 == "./snapshot-sha256.txt" { exit 1 }
    ' "$manifest"; then
        fail "$label snapshot integrity manifest is malformed or unsafe"
        return
    fi
    if ! (
        cd "$directory"
        sha256sum --quiet --strict -c snapshot-sha256.txt
    ); then
        fail "$label snapshot contents do not match snapshot-sha256.txt"
    fi

    find "$directory" -type f ! -name snapshot-sha256.txt -printf './%P\n' \
        | LC_ALL=C sort >"$tmp_dir/$label.actual-files"
    awk '{ print $2 }' "$manifest" | LC_ALL=C sort >"$tmp_dir/$label.recorded-files"
    if ! cmp -s "$tmp_dir/$label.actual-files" "$tmp_dir/$label.recorded-files"; then
        fail "$label snapshot manifest does not cover exactly its evidence files"
        diff -u "$tmp_dir/$label.recorded-files" "$tmp_dir/$label.actual-files" || true
    fi
}

device_value() {
    local directory="$1"
    local key="$2"
    awk -F= -v key="$key" '$1 == key { sub(/^[^=]*=/, ""); print; found = 1 }
        END { if (!found) exit 1 }' "$directory/device.txt"
}

package_status() {
    local directory="$1"
    local label="$2"
    local file="$directory/$label/status.txt"
    require_file "$file" || return 1
    local status
    status="$(tr -d '\r\n' <"$file")"
    if [[ "$status" != "installed" && "$status" != "not installed" ]]; then
        fail "invalid $label package status in $directory: $status"
        return 1
    fi
    printf '%s\n' "$status"
}

normalize_apk_hashes() {
    local input="$1"
    local output="$2"
    if ! require_file "$input"; then
        return 1
    fi
    if ! awk '
        NF < 2 || length($1) != 64 || $1 !~ /^[0-9a-f]+$/ { bad = 1; next }
        { print $1 }
        END { if (bad || NR == 0) exit 1 }
    ' "$input" | LC_ALL=C sort >"$output"; then
        fail "malformed or empty APK hash evidence: $input"
        return 1
    fi
}

normalize_runtime_grants() {
    local input="$1"
    local output="$2"
    if ! require_file "$input"; then
        return 1
    fi
    if ! awk '
        NF && $0 !~ /^user=([0-9]+|unknown) [A-Za-z0-9._]+: granted=(true|false)(,|$)/ {
            exit 1
        }
        NF { print; seen = 1 }
        END { if (!seen) exit 1 }
    ' "$input" | LC_ALL=C sort -u >"$output"; then
        fail "malformed runtime-grant evidence: $input"
        return 1
    fi
}

load_private_hashes() {
    local directory="$1"
    local label="$2"
    local -n destination="$3"
    local manifest="$directory/$label/private-file-sha256.txt"
    local listing="$directory/$label/private-files.txt"
    require_file "$manifest" || return 1
    require_file "$listing" || return 1

    local hash path extra
    while read -r hash path extra; do
        if [[ -n "${extra:-}" || ${#hash} -ne 64 || ! "$hash" =~ ^[0-9a-f]+$ ||
            ! "$path" =~ ^\./(files|shared_prefs|databases|no_backup)/ ]]; then
            fail "malformed private-file hash evidence in $manifest"
            return 1
        fi
        if [[ -n "${destination[$path]+present}" ]]; then
            fail "duplicate private-file path in $manifest: $path"
            return 1
        fi
        destination["$path"]="$hash"
    done <"$manifest"

    sed '/^$/d' "$listing" | LC_ALL=C sort >"$tmp_dir/$label.$(basename "$directory").listed"
    printf '%s\n' "${!destination[@]}" | sed '/^$/d' | LC_ALL=C sort \
        >"$tmp_dir/$label.$(basename "$directory").hashed"
    if ! cmp -s \
        "$tmp_dir/$label.$(basename "$directory").listed" \
        "$tmp_dir/$label.$(basename "$directory").hashed"; then
        fail "$label private file listing and hash evidence disagree in $directory"
        return 1
    fi
}

private_path_class() {
    local path="${1,,}"
    local basename="${path##*/}"
    if [[ "$path" == ./databases/* ]]; then
        printf 'protected-database\n'
    elif [[ "$path" == ./files/* || "$path" == ./no_backup/* ]] && {
        [[ "$basename" =~ \.(db|sqlite|sqlite3)(-(wal|shm|journal))?$ ]] ||
        [[ "$basename" =~ \.(tflite|onnx|pt|pth|model|emb)$ ]] ||
        [[ "$basename" =~ (vocab|tokenizer|sentencepiece) ]] ||
        [[ "$basename" =~ ^(graph\.bin|knn_graph|debug_embeddings)$ ]] ||
        [[ "$path" =~ (session|history|radio[-_]?request|indexing|generation|receipt) ]]
    }; then
        printf 'protected-model-database-session\n'
    elif [[ "$path" =~ (session|history|radio[-_]?request) ]]; then
        printf 'protected-session-state\n'
    elif [[ "$path" =~ (widget|current[-_]?track|now[-_]?playing|playback) ]]; then
        printf 'report-only-volatile-preferences\n'
    elif [[ "$path" == ./shared_prefs/* ]]; then
        printf 'report-only-preferences\n'
    elif [[ "$path" =~ cache ]]; then
        printf 'report-only-volatile-runtime\n'
    else
        printf 'report-only-unclassified\n'
    fi
}

verify_snapshot_integrity "$before_dir" before
verify_snapshot_integrity "$after_dir" after

for required in device.txt poweramp-queue.txt; do
    require_file "$before_dir/$required" || true
    require_file "$after_dir/$required" || true
done

for key in serial fingerprint android_release android_sdk; do
    before_value="$(device_value "$before_dir" "$key" 2>/dev/null || true)"
    after_value="$(device_value "$after_dir" "$key" 2>/dev/null || true)"
    if [[ -z "$before_value" || -z "$after_value" ]]; then
        fail "missing device identity key: $key"
    elif [[ "$before_value" != "$after_value" ]]; then
        fail "device identity changed for $key: '$before_value' -> '$after_value'"
    fi
done

before_v1_status="$(package_status "$before_dir" v1 2>/dev/null || true)"
after_v1_status="$(package_status "$after_dir" v1 2>/dev/null || true)"
if [[ "$before_v1_status" != "installed" ]]; then
    fail "V1 was not installed in the BEFORE snapshot"
fi
if [[ "$after_v1_status" != "installed" ]]; then
    fail "V1 became uninstalled or unreadable in the AFTER snapshot"
fi

if [[ "$before_v1_status" == "installed" && "$after_v1_status" == "installed" ]]; then
    before_apk="$tmp_dir/before-v1-apk"
    after_apk="$tmp_dir/after-v1-apk"
    if normalize_apk_hashes "$before_dir/v1/apk-sha256.txt" "$before_apk" &&
        normalize_apk_hashes "$after_dir/v1/apk-sha256.txt" "$after_apk"; then
        if ! cmp -s "$before_apk" "$after_apk"; then
            fail "V1 APK hash set changed"
            diff -u "$before_apk" "$after_apk" || true
        fi
    fi

    before_grants="$tmp_dir/before-v1-runtime-grants"
    after_grants="$tmp_dir/after-v1-runtime-grants"
    if normalize_runtime_grants "$before_dir/v1/runtime-grants.txt" "$before_grants" &&
        normalize_runtime_grants "$after_dir/v1/runtime-grants.txt" "$after_grants"; then
        if ! cmp -s "$before_grants" "$after_grants"; then
            fail "V1 runtime grants changed"
            diff -u "$before_grants" "$after_grants" || true
        fi
    fi
fi

if [[ -f "$before_dir/poweramp-queue.txt" && -f "$after_dir/poweramp-queue.txt" ]]; then
    if ! cmp -s "$before_dir/poweramp-queue.txt" "$after_dir/poweramp-queue.txt"; then
        fail "Poweramp queue projection changed"
        diff -u "$before_dir/poweramp-queue.txt" "$after_dir/poweramp-queue.txt" || true
    fi
fi

before_v2_status="$(package_status "$before_dir" v2 2>/dev/null || true)"
after_v2_status="$(package_status "$after_dir" v2 2>/dev/null || true)"
report "V2 package status: before='${before_v2_status:-missing}', after='${after_v2_status:-missing}'"
if ((expect_v2_absent_before)) && [[ "$before_v2_status" != "not installed" ]]; then
    fail "V2 was expected to be absent in the BEFORE snapshot"
fi
if ((expect_v2_installed_after)) && [[ "$after_v2_status" != "installed" ]]; then
    fail "V2 was expected to be installed in the AFTER snapshot"
fi

declare -A before_private=()
declare -A after_private=()
private_evidence_loaded=1
load_private_hashes "$before_dir" v1 before_private || private_evidence_loaded=0
load_private_hashes "$after_dir" v1 after_private || private_evidence_loaded=0
if ((private_evidence_loaded)); then
    printf '%s\n' "${!before_private[@]}" "${!after_private[@]}" \
        | sed '/^$/d' | LC_ALL=C sort -u >"$tmp_dir/all-private-paths"
    private_difference_count=0
    while IFS= read -r path; do
        [[ -n "$path" ]] || continue
        before_hash="${before_private[$path]-}"
        after_hash="${after_private[$path]-}"
        [[ "$before_hash" != "$after_hash" ]] || continue
        private_difference_count=$((private_difference_count + 1))
        if [[ -z "$before_hash" ]]; then
            change=added
        elif [[ -z "$after_hash" ]]; then
            change=removed
        else
            change=changed
        fi
        classification="$(private_path_class "$path")"
        report "V1 private file $change [$classification]: $path"
        if [[ "$classification" == protected-* ]]; then
            fail "protected V1 private artifact $change: $path"
        fi
    done <"$tmp_dir/all-private-paths"
    if ((private_difference_count == 0)); then
        report "V1 private-file hashes are unchanged"
    else
        report "$private_difference_count V1 private-file difference(s) classified above"
    fi
fi

if ((failures > 0)); then
    printf 'ACCEPTANCE COMPARISON FAILED: %d strict gate(s) failed\n' "$failures" >&2
    exit 1
fi

printf 'ACCEPTANCE COMPARISON PASSED\n'
