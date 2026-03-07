# Android App

This directory contains the Android app for Poweramp Start Radio.

## What The App Does

The app supports five main jobs:

1. current-track radio from Poweramp
2. text-to-audio search in the shared CLaMP3 embedding space
3. multi-seed search and multi-seed radio
4. on-device indexing for tracks that are missing from `embeddings.db`
5. benchmark and debug entry points used during development

## Build In WSL

### One-time setup

```bash
cd android-plugin
./scripts/setup-wsl-android-env.sh
```

The script installs a local Android toolchain under `~/.local/share/poweramp-start-radio/` and writes the project files needed for Gradle builds inside WSL.

### Build and install

```bash
cd android-plugin
./scripts/build-wsl.sh
```

`build-wsl.sh` assembles a debug APK and installs it automatically if a device is connected over `adb`.

## Runtime Files In `filesDir`

### Required

- `embeddings.db`
  - canonical database produced on desktop and consumed on Android

### Derived on the phone

- `clamp3.emb`
  - mmap dump extracted from `embeddings.db` for fast embedding scans
- `graph.bin`
  - mmap dump of the kNN graph used by Random Walk

### Optional

- `mert.tflite`
  - required for on-device indexing
- `clamp3_audio.tflite`
  - required for on-device indexing
- `clamp3_text.tflite`
  - required for text search on the phone
- `xlm_roberta_vocab.json`
  - required for text search on the phone

## Poweramp Integration

The app reads from Poweramp through its content provider and asks Poweramp for data access when needed.

The runtime integration points are:

- content provider: `content://com.maxmpz.audioplayer.data`
- permission request action: `com.maxmpz.audioplayer.ACTION_ASK_FOR_DATA_PERMISSION`
- broadcast-based debug entry points in `DebugRadioReceiver` and `DebugMultiSeedReceiver`

## Recommendation And Indexing

### Recommendation modes

- `MMR`
  - stays close to the current query while penalizing overlap with the single most-similar chosen result
- `DPP`
  - re-scores each candidate against the chosen set as a whole
- `Random Walk`
  - follows a precomputed similarity graph instead of ranking directly from the seed embedding at runtime

### Drift

- drift is meaningful on the sequential embedding-scan path
- the engine disables drift when `DPP` is selected
- the UI treats drift as not applicable to `Random Walk`

### Text search

- text and audio share the same `768d` CLaMP3 space
- the text model currently falls back to CPU in practice because its graph uses INT64 ops that the GPU path does not handle cleanly

### On-device indexing

- indexing runs in two GPU phases because the device cannot keep both audio models active in the way the app needs
- MERT uses non-overlapping `5s` windows
- chunked decoding preserves window alignment across decode boundaries
- only the final tail of the whole track may become a padded partial window
- FP32 audio model files are required on Android

## Getting Data Into The App

### Push the desktop database

```bash
adb push embeddings.db /data/local/tmp/
adb shell run-as com.powerampstartradio cp /data/local/tmp/embeddings.db files/
adb shell rm /data/local/tmp/embeddings.db
```

### Push on-device indexing models

```bash
for f in mert.tflite clamp3_audio.tflite; do
  adb push "../desktop-indexer/models/$f" /data/local/tmp/
  adb shell run-as com.powerampstartradio cp "/data/local/tmp/$f" files/
  adb shell rm "/data/local/tmp/$f"
done
```

### Push text-search assets

```bash
for f in clamp3_text.tflite xlm_roberta_vocab.json; do
  adb push "../desktop-indexer/models/$f" /data/local/tmp/
  adb shell run-as com.powerampstartradio cp "/data/local/tmp/$f" files/
  adb shell rm "/data/local/tmp/$f"
done
```

## Benchmarks And Debugging

### Audio benchmark

The benchmark activity can run the same chunk-stitched extraction path used by on-device indexing.

```bash
adb shell am start -n com.powerampstartradio/.benchmark.BenchmarkActivity \
  --ez auto_start true --ei max_duration_s 0
```

Pull the result JSON:

```bash
adb shell run-as com.powerampstartradio cat files/benchmark_results.json > /tmp/benchmark_results.json
```

### Text benchmark

```bash
adb shell am start -n com.powerampstartradio/.benchmark.BenchmarkActivity \
  --ez auto_start true --es benchmark_type text
```

### Matching diagnostics

```bash
adb shell am start -n com.powerampstartradio/.benchmark.BenchmarkActivity \
  --ez auto_start true --es benchmark_type diagnose
```

### Debug radio receiver

```bash
adb shell am broadcast -a com.powerampstartradio.DEBUG_START_RADIO \
  -n com.powerampstartradio/.debug.DebugRadioReceiver \
  --es selection_mode MMR --ef diversity_lambda 0.4 --ei num_tracks 30
```

### Debug multi-seed receiver

```bash
adb shell am broadcast -a com.powerampstartradio.DEBUG_MULTI_SEED \
  -n com.powerampstartradio/.debug.DebugMultiSeedReceiver \
  --es song1 "artist title" --ef weight1 1.0 --ei top_k 10
```

Lookup-only mode for finding seed names:

```bash
adb shell am broadcast -a com.powerampstartradio.DEBUG_MULTI_SEED \
  -n com.powerampstartradio/.debug.DebugMultiSeedReceiver \
  --es lookup "partial artist or title"
```

## Code Map

A good reading order for the Android app is:

- `app/src/main/java/com/powerampstartradio/MainActivity.kt`
  - main Compose UI
- `app/src/main/java/com/powerampstartradio/ui/MainViewModel.kt`
  - app state, settings, text search, and multi-seed flow
- `app/src/main/java/com/powerampstartradio/services/RadioService.kt`
  - foreground service that orchestrates radio startup, queueing, and session history
- `app/src/main/java/com/powerampstartradio/similarity/RecommendationEngine.kt`
  - recommendation core
- `app/src/main/java/com/powerampstartradio/poweramp/TrackMatcher.kt`
  - resolves Poweramp tracks against the embedding database
- `app/src/main/java/com/powerampstartradio/indexing/IndexingService.kt`
  - on-device indexing pipeline
- `app/src/main/java/com/powerampstartradio/benchmark/BenchmarkActivity.kt`
  - audio, text, and matching benchmarks

## Common Failure Modes

- stale or missing `graph.bin` after a desktop database update
- missing `clamp3_text.tflite` or vocab for text search
- missing audio models for on-device indexing
- Poweramp metadata that does not line up with the desktop database
- comparing truncated benchmark audio against full-track desktop embeddings
