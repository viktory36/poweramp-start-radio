# Setup Guide

This guide covers the common desktop-first workflow and the optional phone-side indexing path.

## What You Need

### Desktop indexer

- Python `3.10+`
- `pip`
- enough disk space for `embeddings.db`, the model cache, and any exported TFLite models
- a reasonably fast CPU or GPU if you are indexing a large library

### Android app

- an Android device with Poweramp installed
- `adb`

### Building the Android app from source

If you are building the app yourself, the WSL helper scripts install the local toolchain they need.

## Desktop-First Workflow

### 1. Install the desktop indexer

```bash
cd desktop-indexer
python -m pip install -e .
```

### 2. Scan your library

```bash
poweramp-indexer scan /path/to/music -o embeddings.db
```

Useful options:

- `--fp16`
  - reduces desktop VRAM use during MERT inference
- `--batch-size N`
  - larger values improve throughput if your desktop GPU has room
- `--max-duration 600`
  - skips or caps unusually long files
- `--phase 1` or `--phase 2`
  - useful when debugging the two pipeline stages separately

`scan` does five things:

1. reads audio from the music library
2. extracts MERT features in `5s` windows
3. encodes those features into one CLaMP3 embedding per track
4. writes `embeddings.db`
5. builds clusters and the kNN graph used by Random Walk

### 3. Inspect or query the finished database

```bash
poweramp-indexer info embeddings.db
poweramp-indexer similar embeddings.db "radiohead everything in its right place"
poweramp-indexer search embeddings.db "dark minimal techno"
```

### 4. Copy the database to the phone

```bash
adb push embeddings.db /data/local/tmp/
adb shell run-as com.powerampstartradio cp /data/local/tmp/embeddings.db files/
adb shell rm /data/local/tmp/embeddings.db
```

On first use, the Android app extracts mmap-friendly runtime files from that database as needed.

### 5. Build and install the Android app

```bash
cd android-plugin
./scripts/setup-wsl-android-env.sh
./scripts/build-wsl.sh
```

`build-wsl.sh` assembles a debug APK and installs it automatically if `adb` sees a device.

### 6. Start using the app

1. open Poweramp and start playback
2. open Poweramp Start Radio
3. grant the app access to Poweramp data if prompted
4. use `Start Radio` for current-track radio, or use search for text and multi-seed retrieval

## Updating an Existing Desktop Database

If your desktop library changes, update the database in place:

```bash
poweramp-indexer update /path/to/music --database embeddings.db
```

`update` adds new files and can remove missing ones, but it does not rebuild the kNN graph automatically. Rebuild the graph before copying the database back to the phone:

```bash
poweramp-indexer graph embeddings.db --clusters 200 --knn 5
```

## Optional: On-Device Indexing

On-device indexing is useful when the phone has tracks that were not present when the desktop database was built.

### 1. Export the LiteRT models

```bash
cd desktop-indexer
poweramp-indexer export all
```

The Android app uses these files:

- `mert.tflite`
- `clamp3_audio.tflite`
- `clamp3_text.tflite`
- `xlm_roberta_vocab.json`

For on-device audio indexing, only the first two are required.

### 2. Copy the models to the phone

Audio indexing models:

```bash
for f in mert.tflite clamp3_audio.tflite; do
  adb push "desktop-indexer/models/$f" /data/local/tmp/
  adb shell run-as com.powerampstartradio cp "/data/local/tmp/$f" files/
  adb shell rm "/data/local/tmp/$f"
done
```

Text-search assets:

```bash
for f in clamp3_text.tflite xlm_roberta_vocab.json; do
  adb push "desktop-indexer/models/$f" /data/local/tmp/
  adb shell run-as com.powerampstartradio cp "/data/local/tmp/$f" files/
  adb shell rm "/data/local/tmp/$f"
done
```

Notes:

- use the FP32 model files on Android for audio indexing
- the app can run radio from a desktop-built database without any on-device model files
- text search on the phone is optional

### 3. Index missing tracks on the phone

1. open the app
2. go to `Manage Tracks`
3. let it compare the Poweramp library against `embeddings.db`
4. select tracks and start indexing

The phone-side indexing pipeline is:

1. `AudioDecoder` decodes and resamples audio to `24kHz` mono
2. `MertInference` extracts one `768d` feature vector per `5s` window
3. `Clamp3AudioInference` aggregates those features into one `768d` track embedding
4. `GraphUpdater` updates the Random Walk graph on the phone

The extractor carries leftover samples across decode chunks, so multi-chunk tracks follow the same windowing rule as the desktop path.

## Validation

The most useful audio-path validation on Android is the full-track benchmark.

### Run the benchmark

```bash
adb shell am start -n com.powerampstartradio/.benchmark.BenchmarkActivity \
  --ez auto_start true --ei max_duration_s 0
```

### Pull the result JSON

```bash
adb shell run-as com.powerampstartradio cat files/benchmark_results.json > /tmp/benchmark_results.json
```

### Compare it with the desktop reference

```bash
python3 desktop-indexer/scripts/validate_benchmark.py \
  /tmp/benchmark_results.json \
  desktop-indexer/audit_raw_data/embeddings_clamp3.db
```

`EVALUATION.md` records the current measured results and the larger product-level audit snapshot.

## Troubleshooting

### Current track is not found in the database

The Poweramp library and `embeddings.db` are out of sync, or the metadata differs enough that matching falls through. Rebuild or recopy the desktop database, then inspect `TrackMatcher` if needed.

### Random Walk returns poor or empty results

The graph is stale or missing. Rebuild it on desktop after `update`, or let on-device indexing refresh it when phone-side embeddings are added.

### On-device indexing progress runs past the expected MERT window count

That usually points to chunk-boundary windowing problems. The intended behavior is full `5s` windows plus one final padded partial window only when the last whole-track tail is at least `1s`.

### Android build fails inside WSL

Re-run the setup script and then build again:

```bash
cd android-plugin
./scripts/setup-wsl-android-env.sh
./scripts/build-wsl.sh
```

### APK install fails because of a signature mismatch

```bash
adb uninstall com.powerampstartradio
```

Then reinstall.
