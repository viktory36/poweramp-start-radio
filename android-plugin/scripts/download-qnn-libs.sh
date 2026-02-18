#!/usr/bin/env bash
set -euo pipefail

# Downloads all native libraries needed for NPU acceleration:
#
# 1. LiteRT dispatch libraries (from GitHub releases) — bridge CompiledModel API to QNN
#    - libLiteRtCompilerPlugin_Qualcomm.so (679KB) — JIT model compiler
#    - libLiteRtDispatch_Qualcomm.so (477KB) — dispatch bridge
#
# 2. Qualcomm QNN runtime libraries (from Maven Central) — actual HTP execution
#    - libQnnHtp.so, libQnnHtpPrepare.so, libQnnHtpV75Skel.so,
#      libQnnHtpV75Stub.so, libQnnSystem.so (~102MB total)
#
# These .so files are NOT committed to git.
# Run this script before building if you want NPU acceleration.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
JNILIBS_DIR="$SCRIPT_DIR/../app/src/main/jniLibs/arm64-v8a"

# === LiteRT dispatch libraries ===
LITERT_VERSION="v2.1.1"
LITERT_ZIP_URL="https://github.com/google-ai-edge/LiteRT/releases/download/${LITERT_VERSION}/litert_npu_runtime_libraries_jit.zip"

LITERT_FILES=(
  libLiteRtCompilerPlugin_Qualcomm.so
  libLiteRtDispatch_Qualcomm.so
)

# === QNN runtime libraries (SM8650 / Hexagon v75) ===
QNN_VERSION="2.43.0"
AAR_URL="https://repo1.maven.org/maven2/com/qualcomm/qti/qnn-runtime/${QNN_VERSION}/qnn-runtime-${QNN_VERSION}.aar"

QNN_FILES=(
  libQnnHtp.so
  libQnnHtpPrepare.so
  libQnnHtpV75Skel.so
  libQnnHtpV75Stub.so
  libQnnSystem.so
)

# Check if all files are present
ALL_FILES=("${LITERT_FILES[@]}" "${QNN_FILES[@]}")
ALL_PRESENT=true
for f in "${ALL_FILES[@]}"; do
  if [[ ! -f "$JNILIBS_DIR/$f" ]]; then
    ALL_PRESENT=false
    break
  fi
done

if $ALL_PRESENT; then
  echo "All NPU libraries already present in $JNILIBS_DIR"
  ls -lh "$JNILIBS_DIR"/libLiteRt*.so "$JNILIBS_DIR"/libQnn*.so
  exit 0
fi

mkdir -p "$JNILIBS_DIR"
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

# --- Download LiteRT dispatch libraries ---
NEED_LITERT=false
for f in "${LITERT_FILES[@]}"; do
  [[ ! -f "$JNILIBS_DIR/$f" ]] && NEED_LITERT=true && break
done

if $NEED_LITERT; then
  echo "Downloading LiteRT NPU dispatch libraries (${LITERT_VERSION})..."
  curl -sL -o "$TMPDIR/litert_npu_jit.zip" "$LITERT_ZIP_URL"
  echo "Downloaded $(du -h "$TMPDIR/litert_npu_jit.zip" | cut -f1)"

  for f in "${LITERT_FILES[@]}"; do
    unzip -jo "$TMPDIR/litert_npu_jit.zip" "qualcomm_runtime_v75/src/main/jni/arm64-v8a/$f" \
      -d "$JNILIBS_DIR/" > /dev/null
  done
  echo "Extracted LiteRT dispatch libraries."
else
  echo "LiteRT dispatch libraries already present."
fi

# --- Download QNN runtime libraries ---
NEED_QNN=false
for f in "${QNN_FILES[@]}"; do
  [[ ! -f "$JNILIBS_DIR/$f" ]] && NEED_QNN=true && break
done

if $NEED_QNN; then
  echo "Downloading QNN runtime ${QNN_VERSION} from Maven Central..."
  curl -sL -o "$TMPDIR/qnn-runtime.aar" "$AAR_URL"
  echo "Downloaded $(du -h "$TMPDIR/qnn-runtime.aar" | cut -f1)"

  for f in "${QNN_FILES[@]}"; do
    unzip -jo "$TMPDIR/qnn-runtime.aar" "jni/arm64-v8a/$f" -d "$JNILIBS_DIR/" > /dev/null
  done
  echo "Extracted QNN v75 runtime libraries."
else
  echo "QNN runtime libraries already present."
fi

echo ""
echo "NPU libraries in $JNILIBS_DIR:"
ls -lh "$JNILIBS_DIR"/libLiteRt*.so "$JNILIBS_DIR"/libQnn*.so
echo ""
echo "These files enable Qualcomm Hexagon NPU acceleration on SM8650 (Snapdragon 8 Gen 3)."
