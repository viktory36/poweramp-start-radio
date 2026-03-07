# Android App

This directory contains the Android side of Poweramp Start Radio.

The app is not just a thin remote control for Poweramp. It has its own recommendation engine, text search path, session history, track-management UI, and on-device indexing pipeline.

## What The App Does

The current app supports:

1. single-seed radio from the current Poweramp track
2. text search in the shared CLaMP3 audio-text embedding space
3. multi-seed search and multi-seed radio
4. on-device indexing for tracks that exist in Poweramp but are missing from `embeddings.db`
5. benchmark and debug entry points for validation

## Runtime Files In `filesDir`

The app expects or produces the following files in its internal storage.

### Required

- `embeddings.db`
  - canonical database produced on desktop and consumed on Android

### Derived on phone

- `clamp3.emb`
  - mmap dump extracted from `embeddings.db` for fast embedding scans
- `graph.bin`
  - mmap dump of the kNN graph used by Random Walk

### Optional, only if you want the corresponding feature

- `mert.tflite`
  - required for on-device indexing
- `clamp3_audio.tflite`
  - required for on-device indexing
- `clamp3_text.tflite`
  - required for text search on phone
- `xlm_roberta_vocab.json`
  - required for text search on phone

## Build In WSL

### One-time setup

```bash
cd android-plugin
./scripts/setup-wsl-android-env.sh
```

This installs a local Android toolchain and writes:

- `local.properties`
- `.android-wsl-env`
- the Gradle wrapper if it is missing

### Build and install

```bash
cd android-plugin
./scripts/build-wsl.sh
```

`build-wsl.sh` runs `:app:assembleDebug` and installs the APK automatically if `adb` sees a device.

## Deploy Data To The App

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

## Product Behavior Notes

### Recommendation modes

- `MMR`
  - stays close to the current query while penalizing overlap with the single most-similar chosen result
- `DPP`
  - re-scores each candidate against the chosen set as a whole
- `Random Walk`
  - follows a precomputed graph instead of scoring against the seed embedding directly at runtime

### Drift

- drift is meaningful only on the sequential embedding-scan path
- the engine disables drift when `DPP` is selected
- the UI treats drift as not applicable to `Random Walk`

### Text search

- text and audio share the same CLaMP3 embedding space
- the text model currently falls back to CPU in practice because its graph uses INT64 ops that the GPU path does not handle cleanly

### On-device indexing

- indexing runs in two GPU phases because the device cannot keep both audio models active in the way the app needs
- chunked decoding preserves `5s` MERT window alignment across chunk boundaries
- only the final tail of the whole track may become a padded partial window

## Benchmarks And Debugging

### Audio benchmark

The benchmark activity is the best built-in way to validate on-device embedding quality.

Run a full-track benchmark:

```bash
adb shell am start -n com.powerampstartradio/.benchmark.BenchmarkActivity \
  --ez auto_start true --ei max_duration_s 0
```

Pull the result JSON:

```bash
adb shell run-as com.powerampstartradio cat files/benchmark_results.json > /tmp/benchmark_results.json
```

Recent full-track validation against the desktop DB passes with mean cosine around `0.9955`.

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

If you are trying to understand the app, start with these files.

- `app/src/main/java/com/powerampstartradio/MainActivity.kt`
  - main Compose UI
- `app/src/main/java/com/powerampstartradio/ui/MainViewModel.kt`
  - settings, text search, multi-seed state, import flow
- `app/src/main/java/com/powerampstartradio/services/RadioService.kt`
  - foreground service orchestrating radio startup, queueing, and session history
- `app/src/main/java/com/powerampstartradio/similarity/RecommendationEngine.kt`
  - recommendation core
- `app/src/main/java/com/powerampstartradio/poweramp/TrackMatcher.kt`
  - resolves Poweramp tracks against the embedding database
- `app/src/main/java/com/powerampstartradio/indexing/IndexingService.kt`
  - on-device indexing pipeline
- `app/src/main/java/com/powerampstartradio/benchmark/BenchmarkActivity.kt`
  - audio, text, and matching benchmarks

## Common Failure Modes

- stale or missing `graph.bin` after a desktop DB update
- missing `clamp3_text.tflite` or vocab for text search
- missing audio models for on-device indexing
- mismatched Poweramp metadata causing track resolution misses
- comparing truncated benchmark audio against full-track desktop embeddings
