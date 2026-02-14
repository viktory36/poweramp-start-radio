#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ANDROID_PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

TOOLS_ROOT="${TOOLS_ROOT:-$HOME/.local/share/poweramp-start-radio}"
JDK_DIR="$TOOLS_ROOT/jdk-17"
GRADLE_DIR="$TOOLS_ROOT/gradle-8.13"
ANDROID_SDK_DIR="$TOOLS_ROOT/android-sdk"
ENV_FILE="$ANDROID_PROJECT_DIR/.android-wsl-env"

JDK_URL="${JDK_URL:-https://api.adoptium.net/v3/binary/latest/17/ga/linux/x64/jdk/hotspot/normal/eclipse?project=jdk}"
GRADLE_URL="${GRADLE_URL:-https://services.gradle.org/distributions/gradle-8.13-bin.zip}"
CMDLINE_TOOLS_VERSION="${CMDLINE_TOOLS_VERSION:-13114758}"
CMDLINE_TOOLS_ZIP="commandlinetools-linux-${CMDLINE_TOOLS_VERSION}_latest.zip"
CMDLINE_TOOLS_URL="${CMDLINE_TOOLS_URL:-https://dl.google.com/android/repository/${CMDLINE_TOOLS_ZIP}}"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

for cmd in curl unzip tar; do
  need_cmd "$cmd"
done

mkdir -p "$TOOLS_ROOT"

install_jdk() {
  if [[ -x "$JDK_DIR/bin/java" ]]; then
    echo "JDK already installed: $JDK_DIR"
    return
  fi

  echo "Downloading JDK 17..."
  local archive="$TMP_DIR/jdk17.tar.gz"
  curl -fL "$JDK_URL" -o "$archive"

  mkdir -p "$TMP_DIR/jdk"
  tar -xzf "$archive" -C "$TMP_DIR/jdk"

  local extracted
  extracted="$(find "$TMP_DIR/jdk" -mindepth 1 -maxdepth 1 -type d | head -n1)"
  if [[ -z "$extracted" ]]; then
    echo "Failed to extract JDK archive" >&2
    exit 1
  fi

  rm -rf "$JDK_DIR"
  mv "$extracted" "$JDK_DIR"
  echo "Installed JDK: $JDK_DIR"
}

install_gradle_dist() {
  if [[ -x "$GRADLE_DIR/bin/gradle" ]]; then
    echo "Gradle already installed: $GRADLE_DIR"
    return
  fi

  echo "Downloading Gradle 8.13..."
  local archive="$TMP_DIR/gradle-8.13-bin.zip"
  curl -fL "$GRADLE_URL" -o "$archive"

  mkdir -p "$TMP_DIR/gradle"
  unzip -q "$archive" -d "$TMP_DIR/gradle"

  local extracted="$TMP_DIR/gradle/gradle-8.13"
  if [[ ! -d "$extracted" ]]; then
    echo "Failed to extract Gradle archive" >&2
    exit 1
  fi

  rm -rf "$GRADLE_DIR"
  mv "$extracted" "$GRADLE_DIR"
  echo "Installed Gradle: $GRADLE_DIR"
}

ensure_gradle_wrapper() {
  if [[ -f "$ANDROID_PROJECT_DIR/gradlew" && -f "$ANDROID_PROJECT_DIR/gradle/wrapper/gradle-wrapper.jar" ]]; then
    chmod +x "$ANDROID_PROJECT_DIR/gradlew"
    echo "Gradle wrapper already present"
    return
  fi

  echo "Generating Gradle wrapper..."
  JAVA_HOME="$JDK_DIR" "$GRADLE_DIR/bin/gradle" \
    -p "$ANDROID_PROJECT_DIR" \
    --no-daemon \
    wrapper \
    --gradle-version 8.13

  chmod +x "$ANDROID_PROJECT_DIR/gradlew"
}

install_cmdline_tools() {
  local sdkmanager_bin="$ANDROID_SDK_DIR/cmdline-tools/latest/bin/sdkmanager"
  local version_marker="$ANDROID_SDK_DIR/cmdline-tools/latest/.poweramp-cli-tools-version"
  if [[ -x "$sdkmanager_bin" && -f "$version_marker" && "$(cat "$version_marker")" == "$CMDLINE_TOOLS_VERSION" ]]; then
    echo "Android cmdline-tools already installed"
    return
  fi

  echo "Downloading Android cmdline-tools (${CMDLINE_TOOLS_VERSION})..."
  local archive="$TMP_DIR/$CMDLINE_TOOLS_ZIP"
  curl -fL "$CMDLINE_TOOLS_URL" -o "$archive"

  mkdir -p "$TMP_DIR/cmdline-tools"
  unzip -q "$archive" -d "$TMP_DIR/cmdline-tools"

  if [[ ! -d "$TMP_DIR/cmdline-tools/cmdline-tools" ]]; then
    echo "Failed to extract Android cmdline-tools archive" >&2
    exit 1
  fi

  mkdir -p "$ANDROID_SDK_DIR/cmdline-tools"
  rm -rf "$ANDROID_SDK_DIR/cmdline-tools/latest"
  mv "$TMP_DIR/cmdline-tools/cmdline-tools" "$ANDROID_SDK_DIR/cmdline-tools/latest"
  echo "$CMDLINE_TOOLS_VERSION" > "$version_marker"
  echo "Installed Android cmdline-tools"
}

install_sdk_packages() {
  local sdkmanager_bin="$ANDROID_SDK_DIR/cmdline-tools/latest/bin/sdkmanager"
  if [[ ! -x "$sdkmanager_bin" ]]; then
    echo "sdkmanager not found after cmdline-tools installation" >&2
    exit 1
  fi

  echo "Accepting Android SDK licenses..."
  yes | "$sdkmanager_bin" --sdk_root="$ANDROID_SDK_DIR" --licenses >/dev/null || true

  echo "Installing Android SDK packages..."
  "$sdkmanager_bin" --sdk_root="$ANDROID_SDK_DIR" \
    "platform-tools" \
    "platforms;android-34" \
    "build-tools;34.0.0" \
    "build-tools;35.0.0"
}

write_local_config() {
  cat > "$ANDROID_PROJECT_DIR/local.properties" <<LOCAL
sdk.dir=$ANDROID_SDK_DIR
LOCAL

  cat > "$ENV_FILE" <<ENV
export JAVA_HOME="$JDK_DIR"
export ANDROID_SDK_ROOT="$ANDROID_SDK_DIR"
export ANDROID_HOME="$ANDROID_SDK_DIR"
export PATH="\$JAVA_HOME/bin:\$ANDROID_SDK_ROOT/cmdline-tools/latest/bin:\$ANDROID_SDK_ROOT/platform-tools:\$PATH"
ENV

  echo "Wrote $ANDROID_PROJECT_DIR/local.properties"
  echo "Wrote $ENV_FILE"
}

install_jdk
# Force all subsequent Java-based tooling (Gradle + sdkmanager) to use JDK 17.
export JAVA_HOME="$JDK_DIR"
export PATH="$JAVA_HOME/bin:$PATH"
install_gradle_dist
ensure_gradle_wrapper
install_cmdline_tools
install_sdk_packages
write_local_config

echo ""
echo "Setup complete."
echo "Build command:"
echo "  cd $ANDROID_PROJECT_DIR && source .android-wsl-env && ./gradlew --no-daemon :app:assembleDebug"
