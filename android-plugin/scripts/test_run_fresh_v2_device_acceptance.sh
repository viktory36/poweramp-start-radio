#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
runner="$script_dir/run_fresh_v2_device_acceptance.sh"
tmp_dir="$(mktemp -d)"
trap 'rm -rf -- "$tmp_dir"' EXIT

[[ -x "$runner" ]] || {
    printf 'runner is not executable: %s\n' "$runner" >&2
    exit 1
}

expected_permissions=(
    android.permission.READ_MEDIA_AUDIO
    android.permission.POST_NOTIFICATIONS
)

expected_assets=(
    "models/mert.tflite|files/mert.tflite|378064368|34930763bed772d616b124b0103e3759f9cf464d1b193f73fc1367449f32c539"
    "models/clamp3_audio.tflite|files/clamp3_audio.tflite|343793912|06b6a17425b7d8ccb5327db7129aa58102cd119502aed00a397bc4b6e7ae20bc"
    "models/clamp3_text.tflite|files/clamp3_text.tflite|1113080740|10398ea03ee96e56d4c6970e302ac2061a41d255ed1ef3d730001ec17ca0fb16"
    "models/sentencepiece.bpe.model|files/sentencepiece.bpe.model|5069051|cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865"
    "text_parity/manifest.json|files/device_acceptance/text_parity/manifest.json|5010|9d872fb702e3c091358e09cd31ec7b1a25736e95d06db31611acf4770b0cd4e3"
    "text_parity/short_english.f32le|files/device_acceptance/text_parity/short_english.f32le|3072|0e7e40f34cdca86046ed2a4c9b7af242b1af686cadc6bb843787070f6b2cae29"
    "text_parity/short_arabic.f32le|files/device_acceptance/text_parity/short_arabic.f32le|3072|77d988d589e4ad5b9e67d6b0b174cbe10f56c9038fe60e0487ca600268605963"
    "text_parity/long_multisection.f32le|files/device_acceptance/text_parity/long_multisection.f32le|3072|68610f8cf6990c300136e1b6a438e96baf5430d2bed010a3d2cdf9284a8be289"
    "audio_parity/manifest.json|files/device_acceptance/audio_parity/manifest.json|4582|5b6430f8e9fd39eb28503d4e970797f021051b4f398a4b25cb08256ebd278f26"
    "audio_parity/source/abdullah_miniawy_signature.flac|files/device_acceptance/audio_parity/source/abdullah_miniawy_signature.flac|5348656|43922a38cb556e49cee69a5eb6cbb55d4d94ddf6d66277bc6f5882051017be33"
    "audio_parity/source/daniela_andrade_angel19.flac|files/device_acceptance/audio_parity/source/daniela_andrade_angel19.flac|713302|4ba909053f5e47dd663172d549a097cc45a614f0e557e868f2fe35f5373d9d43"
    "audio_parity/source/vice_city_interlude_3.flac|files/device_acceptance/audio_parity/source/vice_city_interlude_3.flac|1294386|ce0da2f5e54bab63482c0731d73a174ef683ede8e961967dd08f0540238bbfdf"
    "audio_parity/expected/abdullah_miniawy_signature.f32le|files/device_acceptance/audio_parity/expected/abdullah_miniawy_signature.f32le|3072|3907ac88565c995a3d6f9633c407692d479efb71f6d486222f3b750625e8faff"
    "audio_parity/expected/daniela_andrade_angel19.f32le|files/device_acceptance/audio_parity/expected/daniela_andrade_angel19.f32le|3072|f6731c3df53e986c34021a32b98ce324f75b970b1c0f1892eb1cd5c15ddc93b9"
    "audio_parity/expected/vice_city_interlude_3.f32le|files/device_acceptance/audio_parity/expected/vice_city_interlude_3.f32le|3072|66d73a6272f476ce231e46bc7b7e9f1f91d10735d37bd96af454f391e6e68f14"
    "frozen/embeddings.db|files/device_acceptance/embeddings.db|380243968|08dfcec60f7c2e9de4bc6b923d601bd824f80b6251769f6c7bcd8062ce6aa504"
    "graph-explorer-benchmark.bin|files/graph-explorer-benchmark.bin|3860216|65dafdae5e713f3913e6d6f082612813f6859d37d35ec9e2cdcc43f06c077656"
)

expected_classes=(
    "01|pre-import|com.powerampstartradio.poweramp.PowerampTrackIdentityInstrumentedTest"
    "02|pre-import|com.powerampstartradio.indexing.V2IndexingServiceIntentInstrumentedTest"
    "03|pre-import|com.powerampstartradio.indexing.v2.V2IndexingPreflightIntentStoreInstrumentedTest"
    "04|pre-import|com.powerampstartradio.indexing.v2.AtomicV2ArtifactStoreInstrumentedTest"
    "05|pre-import|com.powerampstartradio.indexing.v2.V2EmbeddingCommitRepositoryInstrumentedTest"
    "06|pre-import|com.powerampstartradio.indexing.v2.V2BootstrapGenerationImporterInstrumentedTest"
    "07|pre-import|com.powerampstartradio.data.EmbeddingIndexNativeInstrumentedTest"
    "08|pre-import|com.powerampstartradio.indexing.OfficialSentencePieceTokenizerInstrumentedTest"
    "09|pre-import|com.powerampstartradio.indexing.TextEmbeddingParityInstrumentedTest"
    "10|pre-import|com.powerampstartradio.indexing.v2.V2PowerampProviderSnapshotInstrumentedTest"
    "11|pre-import|com.powerampstartradio.indexing.AudioEmbeddingParityInstrumentedTest"
    "12|import|com.powerampstartradio.indexing.v2.V2FrozenDatabaseImportAcceptanceTest"
    "13|post-import|com.powerampstartradio.similarity.ClosestRecommendationInstrumentedTest"
    "14|post-import|com.powerampstartradio.similarity.algorithms.GraphExplorerNativeInstrumentedTest"
    "15|post-import|com.powerampstartradio.data.ActiveDomainGraphTopologyInstrumentedTest"
    "16|post-import|com.powerampstartradio.similarity.ActiveDomainRecommendationEngineInstrumentedTest"
)

"$runner" --help >"$tmp_dir/help.stdout" 2>"$tmp_dir/help.stderr"
grep -q 'Arbitrary instrumentation classes' "$tmp_dir/help.stderr"

"$runner" --dry-run >"$tmp_dir/plan" 2>"$tmp_dir/plan.stderr"
[[ ! -s "$tmp_dir/plan.stderr" ]]
grep -qx 'Runtime permission grants (2, fixed order):' "$tmp_dir/plan"
grep -qx 'Assets (17):' "$tmp_dir/plan"
grep -qx 'Instrumentation classes (16, fixed order):' "$tmp_dir/plan"

awk '
    /^Runtime permission grants / { active = 1; next }
    /^Assets / { active = 0 }
    active { sub(/^  /, ""); print }
' "$tmp_dir/plan" >"$tmp_dir/actual-permissions"
printf '%s\n' "${expected_permissions[@]}" >"$tmp_dir/expected-permissions"
diff -u "$tmp_dir/expected-permissions" "$tmp_dir/actual-permissions"

sed -n \
    's#^  /data/local/tmp/pasr-v2-assets/\(.*\) -> \(.*\) (\([0-9][0-9]*\) bytes, sha256 \([0-9a-f][0-9a-f]*\))$#\1|\2|\3|\4#p' \
    "$tmp_dir/plan" >"$tmp_dir/actual-assets"
printf '%s\n' "${expected_assets[@]}" >"$tmp_dir/expected-assets"
diff -u "$tmp_dir/expected-assets" "$tmp_dir/actual-assets"

sed -n \
    's/^  \([0-9][0-9]\) \[\([^]]*\)\][[:space:]]*\(.*\)$/\1|\2|\3/p' \
    "$tmp_dir/plan" >"$tmp_dir/actual-classes"
printf '%s\n' "${expected_classes[@]}" >"$tmp_dir/expected-classes"
diff -u "$tmp_dir/expected-classes" "$tmp_dir/actual-classes"

reject_argument() {
    local expected="$1"
    shift
    if "$runner" "$@" >"$tmp_dir/rejected.stdout" 2>"$tmp_dir/rejected.stderr"; then
        printf 'runner accepted forbidden arguments: %s\n' "$*" >&2
        exit 1
    fi
    grep -q -- "$expected" "$tmp_dir/rejected.stderr"
}

reject_argument 'arbitrary instrumentation arguments are refused' \
    --class com.example.UnreviewedTest
reject_argument 'unknown or positional argument refused' com.example.UnreviewedTest
reject_argument 'unknown or positional argument refused' --dry-run unexpected
reject_argument '--log-dir requires a non-empty directory' --log-dir --class

target_install_line="    adb_device install \"\$target_apk\""
test_install_line="    adb_device install -r -t \"\$test_apk\""
[[ "$(grep -Fxc "$target_install_line" "$runner")" == 1 ]]
[[ "$(grep -Fxc "$test_install_line" "$runner")" == 1 ]]
[[ "$(grep -Ec '^[[:space:]]*adb_device shell pm grant ' "$runner")" == 1 ]]
[[ "$(grep -Ec '^[[:space:]]*adb_device shell am instrument ' "$runner")" == 1 ]]
if grep -Eq 'adb_device install .*([[:space:]])-g([[:space:]]|$)' "$runner"; then
    printf 'runner installation grants every requested runtime permission\n' >&2
    exit 1
fi
if grep -Eq \
    'content[[:space:]]+(insert|update|delete|call)|deviceidle|whitelist|am[[:space:]]+(start|startservice|start-foreground-service|broadcast|force-stop)|pm[[:space:]]+(clear|uninstall|disable|enable|reset-permissions)|cmd[[:space:]]+appops|settings[[:space:]]+put|input[[:space:]]+|monkey[[:space:]]|reboot([[:space:]]|$)' \
    "$runner"; then
    printf 'runner contains a forbidden direct device operation\n' >&2
    exit 1
fi
if grep -Eq 'content://com\.maxmpz\.audioplayer\.data/(queue|folders)|ACTION_(API|RELOAD)|DEBUG_START_RADIO' "$runner"; then
    printf 'runner contains a Poweramp playback or provider-mutation surface\n' >&2
    exit 1
fi

grep -q 'POWERAMP CONSENT HANDOFF' "$runner"
grep -q 'private-assets-after-tests.tsv' "$runner"
grep -q 'v2-runtime-grants-final.txt' "$runner"
grep -q 'evidence-sha256.tsv' "$runner"
grep -q 'V2 must be absent' "$runner"
grep -q 'V1 must be installed' "$runner"
grep -Fq 'Rebuilding the target and instrumentation APKs from the current reviewed source' "$runner"
grep -Fq ':app:assembleDebug :app:assembleDebugAndroidTest' "$runner"
grep -Fq 'acceptance requires exactly 36' "$runner"
grep -Fq 'asset_total_bytes * 2 + 1073741824' "$runner"
grep -Fq 'insufficient /data space' "$runner"

printf 'fresh V2 device-acceptance runner static tests passed\n'
