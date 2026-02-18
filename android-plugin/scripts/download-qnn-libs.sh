#!/usr/bin/env bash
set -euo pipefail

# Downloads Qualcomm QNN runtime libraries for NPU acceleration.
# Extracts only SM8650 (Hexagon v75) files from the Maven AAR.
#
# These .so files are NOT committed to git (102MB, binary).
# Run this script before building if you want NPU acceleration.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
JNILIBS_DIR="$SCRIPT_DIR/../app/src/main/jniLibs/arm64-v8a"
QNN_VERSION="2.43.0"
AAR_URL="https://repo1.maven.org/maven2/com/qualcomm/qti/qnn-runtime/${QNN_VERSION}/qnn-runtime-${QNN_VERSION}.aar"

REQUIRED_FILES=(
  libQnnHtp.so
  libQnnHtpPrepare.so
  libQnnHtpV75Skel.so
  libQnnHtpV75Stub.so
  libQnnSystem.so
)

# Check if already downloaded
ALL_PRESENT=true
for f in "${REQUIRED_FILES[@]}"; do
  if [[ ! -f "$JNILIBS_DIR/$f" ]]; then
    ALL_PRESENT=false
    break
  fi
done

if $ALL_PRESENT; then
  echo "QNN v75 libraries already present in $JNILIBS_DIR"
  ls -lh "$JNILIBS_DIR"/libQnn*.so
  exit 0
fi

echo "Downloading QNN runtime ${QNN_VERSION} from Maven Central..."
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

curl -sL -o "$TMPDIR/qnn-runtime.aar" "$AAR_URL"
echo "Downloaded $(du -h "$TMPDIR/qnn-runtime.aar" | cut -f1)"

mkdir -p "$JNILIBS_DIR"
for f in "${REQUIRED_FILES[@]}"; do
  unzip -jo "$TMPDIR/qnn-runtime.aar" "jni/arm64-v8a/$f" -d "$JNILIBS_DIR/" > /dev/null
done

echo ""
echo "Extracted QNN v75 libraries to $JNILIBS_DIR:"
ls -lh "$JNILIBS_DIR"/libQnn*.so
echo ""
echo "These files enable Qualcomm Hexagon NPU acceleration on SM8650 (Snapdragon 8 Gen 3)."
