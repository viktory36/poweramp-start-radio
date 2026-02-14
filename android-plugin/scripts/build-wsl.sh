#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ANDROID_PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$ANDROID_PROJECT_DIR/.android-wsl-env"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing $ENV_FILE. Run scripts/setup-wsl-android-env.sh first." >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$ENV_FILE"

cd "$ANDROID_PROJECT_DIR"
./gradlew --no-daemon :app:assembleDebug "$@"
