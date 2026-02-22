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

On-device indexing lets the phone generate embeddings for new tracks without the desktop. It requires TFLite models pushed to the app's internal storage.

### 1. Export TFLite models from PyTorch

```bash
cd desktop-indexer

# Install export dependencies (litert-torch is picky about versions)
pip install litert-torch==0.8.0 --no-deps
pip install jax torch_xla2 ai-edge-quantizer immutabledict --no-deps

# Export all three models (MuLan, Flamingo encoder, Flamingo projector)
python -m poweramp_indexer.export_litert all
# Outputs to desktop-indexer/models/:
#   mulan_audio.tflite            (~1212 MB, FP32)
#   mulan_audio.mel_params.json   (mel spectrogram config sidecar)
#   flamingo_encoder.tflite       (~2432 MB, FP32)
#   flamingo_projector.tflite     (~67 MB, FP32)
```

### 2. Convert to FP16 (halves size, lossless for GPU)

```bash
python scripts/convert_fp16.py models/mulan_audio.tflite
python scripts/convert_fp16.py models/flamingo_encoder.tflite
python scripts/convert_fp16.py models/flamingo_projector.tflite
# Outputs: *_fp16.tflite variants (~607 MB, ~1276 MB, ~34 MB)
```

### 3. Push to phone

The app looks for models in its `filesDir`. With the app installed:

```bash
# Push FP16 models
adb push models/mulan_audio_fp16.tflite /data/local/tmp/
adb push models/flamingo_encoder_fp16.tflite /data/local/tmp/
adb push models/flamingo_projector_fp16.tflite /data/local/tmp/
adb push models/mulan_audio.mel_params.json /data/local/tmp/

# Move into app storage (requires run-as or root)
adb shell run-as com.powerampstartradio cp /data/local/tmp/mulan_audio_fp16.tflite files/
adb shell run-as com.powerampstartradio cp /data/local/tmp/flamingo_encoder_fp16.tflite files/
adb shell run-as com.powerampstartradio cp /data/local/tmp/flamingo_projector_fp16.tflite files/
adb shell run-as com.powerampstartradio cp /data/local/tmp/mulan_audio.mel_params.json files/

# Also push the embedding database if not already on the phone
adb push embeddings.db /data/local/tmp/
adb shell run-as com.powerampstartradio cp /data/local/tmp/embeddings.db files/

# Clean up temp files
adb shell rm /data/local/tmp/*.tflite /data/local/tmp/*.json /data/local/tmp/embeddings.db
```

The app auto-detects `_fp16` variants and prefers them over FP32. The `mel_params.json` sidecar must be alongside the MuLan model (it falls back to defaults if missing, but the defaults match so this is optional).

### 4. Index new tracks

Open the app → Settings → **Manage Tracks**. The app compares the Poweramp library against the embedding database and shows unindexed tracks. Select tracks and press Start.

Typical timing on Snapdragon 8 Gen 3 (Adreno 740 GPU):
- ~18–26s per track per model for ~4 min FLAC files
- GPU model load: ~4–7s JIT compilation (once per session)
- Incremental fusion: ~17s (updates existing SVD/clusters/kNN)
- Full re-fusion: ~7 min for 74K tracks (only needed on first import)
