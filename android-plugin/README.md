# Android Plugin (WSL Build)

This project can be built inside WSL without Android Studio by bootstrapping a local toolchain.

## One-time setup

```bash
cd android-plugin
./scripts/setup-wsl-android-env.sh
```

This installs (under `~/.local/share/poweramp-start-radio/`):
- JDK 17
- Gradle 8.13 distribution
- Android cmdline-tools + SDK packages (`platforms;android-34`, `build-tools;34.0.0`, `platform-tools`)

It also creates:
- `android-plugin/local.properties`
- `android-plugin/.android-wsl-env`
- `android-plugin/gradlew` + `android-plugin/gradle/wrapper/gradle-wrapper.jar`

## Build

```bash
cd android-plugin
source .android-wsl-env
./gradlew --no-daemon :app:assembleDebug
```

Or use:

```bash
cd android-plugin
./scripts/build-wsl.sh
```

## On-Device Indexing Setup

On-device indexing lets the phone generate CLaMP3 embeddings for new tracks without the desktop. It requires TFLite models pushed to the app's internal storage.

### 1. Export TFLite models from PyTorch

```bash
cd desktop-indexer

# Install export dependencies (litert-torch is picky about versions)
pip install litert-torch==0.8.0 --no-deps
pip install jax torch_xla2 ai-edge-quantizer immutabledict --no-deps

# Export MERT and CLaMP3 audio encoder
python -m poweramp_indexer.export_litert all
# Outputs to desktop-indexer/models/:
#   mert.tflite            (~1838 MB, FP32)
#   clamp3_audio.tflite    (~TBD MB, FP32)
```

### 2. Convert to FP16 (halves size, lossless for GPU)

```bash
python scripts/convert_fp16.py models/mert.tflite
python scripts/convert_fp16.py models/clamp3_audio.tflite
# Outputs: *_fp16.tflite variants
```

### 3. Push to phone

The app looks for models in its `filesDir`. With the app installed:

```bash
# Push FP16 models
adb push models/mert_fp16.tflite /data/local/tmp/
adb push models/clamp3_audio_fp16.tflite /data/local/tmp/

# Move into app storage (requires run-as or root)
adb shell run-as com.powerampstartradio cp /data/local/tmp/mert_fp16.tflite files/
adb shell run-as com.powerampstartradio cp /data/local/tmp/clamp3_audio_fp16.tflite files/

# Also push the embedding database if not already on the phone
adb push embeddings.db /data/local/tmp/
adb shell run-as com.powerampstartradio cp /data/local/tmp/embeddings.db files/

# Clean up temp files
adb shell rm /data/local/tmp/*.tflite /data/local/tmp/embeddings.db
```

The app auto-detects `_fp16` variants and prefers them over FP32.

### 4. Index new tracks

Open the app → Settings → **Manage Tracks**. The app compares the Poweramp library against the embedding database and shows unindexed tracks. Select tracks and press Start.

CLaMP3 pipeline: MERT (feature extraction) → CLaMP3 audio encoder (768d embedding).

Typical timing on Snapdragon 8 Gen 3 (Adreno 740 GPU):
- Two-phase GPU pipeline (Adreno can't run two OpenCL TFLite contexts simultaneously)
- GPU model load: ~4–7s JIT compilation (once per session)
