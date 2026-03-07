# Setup Guide

This repository supports two workflows:

1. Desktop-first, where a computer builds `embeddings.db` and the phone consumes it
2. Desktop-first plus on-device incremental indexing, where the phone adds embeddings for tracks that were not in the desktop database yet

The first workflow is the normal one. The second is optional.

## Prerequisites

### Desktop indexer

- Python 3.10+
- `pip`
- enough disk space for the database and model cache
- a reasonably fast CPU or GPU if you are indexing a large library

### Android app

- an Android device with Poweramp installed
- `adb`

### Building the Android app from source

- WSL is supported through the scripts in `android-plugin/scripts/`
- the setup script installs JDK 17, Gradle, Android command-line tools, and SDK packages locally under `~/.local/share/poweramp-start-radio/`

## Recommended Workflow

### 1. Install the desktop indexer

```bash
cd desktop-indexer
python -m pip install -e .
```

### 2. Build the desktop database

```bash
poweramp-indexer scan /path/to/music -o embeddings.db
```

Useful options:

- `--fp16`
  - halves VRAM use on desktop MERT inference
- `--batch-size N`
  - larger batch sizes improve throughput if desktop VRAM allows it
- `--max-duration 600`
  - skip or cap unusually long files
- `--phase 1` or `--phase 2`
  - run only half of the pipeline when debugging

What `scan` does:

1. reads audio from your music library
2. extracts MERT features in `5s` windows
3. encodes those features into one CLaMP3 embedding per track
4. writes `embeddings.db`
5. builds clusters and a kNN graph for Random Walk

### 3. Inspect the finished database

```bash
poweramp-indexer info embeddings.db
```

### 4. Push the database to the phone

```bash
adb push embeddings.db /data/local/tmp/
adb shell run-as com.powerampstartradio cp /data/local/tmp/embeddings.db files/
adb shell rm /data/local/tmp/embeddings.db
```

### 5. Install or rebuild the Android app

```bash
cd android-plugin
./scripts/setup-wsl-android-env.sh
./scripts/build-wsl.sh
```

`build-wsl.sh` installs the debug APK automatically if a device is connected over `adb`.

### 6. Use the app

- Start playback in Poweramp
- Open Poweramp Start Radio
- Use `Start Radio` for current-track radio
- Use search for text or multi-seed retrieval
- Use `Manage Tracks` only if you want the phone to add embeddings for missing tracks

## Incremental Desktop Updates

If your desktop music library changes:

```bash
poweramp-indexer update /path/to/music --database embeddings.db
```

Important:

- `update` adds and removes tracks in the database
- `update` does not rebuild the kNN graph automatically
- after a desktop update, rebuild the graph before pushing the DB back to the phone:

```bash
poweramp-indexer graph embeddings.db --clusters 200 --knn 5
```

## Optional: On-Device Indexing

On-device indexing is for tracks that exist in Poweramp but are not in the desktop database yet.

To enable it, copy audio models into the app's internal storage.

### 1. Export the LiteRT models

```bash
cd desktop-indexer
poweramp-indexer export all
```

The current production-relevant files are:

- `mert.tflite`
- `clamp3_audio.tflite`
- `clamp3_text.tflite`
- `xlm_roberta_vocab.json`

### 2. Push models to the phone

For on-device indexing, only the audio models are required:

```bash
for f in mert.tflite clamp3_audio.tflite; do
  adb push "desktop-indexer/models/$f" /data/local/tmp/
  adb shell run-as com.powerampstartradio cp "/data/local/tmp/$f" files/
  adb shell rm "/data/local/tmp/$f"
done
```

For on-device text search, also push:

```bash
for f in clamp3_text.tflite xlm_roberta_vocab.json; do
  adb push "desktop-indexer/models/$f" /data/local/tmp/
  adb shell run-as com.powerampstartradio cp "/data/local/tmp/$f" files/
  adb shell rm "/data/local/tmp/$f"
done
```

Important notes:

- use the FP32 model files for the Android app's audio indexing path
- text search is optional
- the app can run radio from a desktop-built DB without any on-device model files

### 3. Index tracks on the phone

- open the app
- go to `Manage Tracks`
- let it compare the Poweramp library against `embeddings.db`
- select tracks and start indexing

The Android indexing pipeline is:

1. `AudioDecoder` decodes and resamples audio to `24kHz` mono
2. `MertInference` extracts `768d` features per `5s` window
3. `Clamp3AudioInference` turns those features into one `768d` track embedding
4. `GraphUpdater` updates the Random Walk graph on phone

The current implementation preserves MERT window alignment across decode chunks so multi-chunk tracks match the desktop windowing rule.

## Benchmarks and Validation

When you need to validate on-device embedding quality, use the benchmark activity in full-track mode.

### Run the audio benchmark

```bash
adb shell am start -n com.powerampstartradio/.benchmark.BenchmarkActivity \
  --ez auto_start true --ei max_duration_s 0
```

### Pull the result JSON

```bash
adb shell run-as com.powerampstartradio cat files/benchmark_results.json > /tmp/benchmark_results.json
```

### Validate against the desktop database

```bash
python3 desktop-indexer/scripts/validate_benchmark.py \
  /tmp/benchmark_results.json \
  desktop-indexer/audit_raw_data/embeddings_clamp3.db
```

Do not treat a capped `120s` benchmark run as a quality comparison against a full-track desktop database. That is measuring different audio.

## Troubleshooting

### Current track is not found in the database

The Poweramp track metadata or file set does not line up with the database you pushed. Rebuild or repush the desktop database, or inspect `TrackMatcher` behavior.

### Random Walk returns poor or empty results

The database or on-device extracted files do not have a current graph. Rebuild the graph on desktop after `update`, or let on-device indexing rebuild it after new phone-side embeddings are added.

### On-device indexing progress overruns the expected window count

That was historically caused by chunk-boundary window leakage. The current code carries leftover samples across chunks so only the final track tail may create a padded window.

### Android build fails inside WSL

Re-run:

```bash
cd android-plugin
./scripts/setup-wsl-android-env.sh
```

Then build again with:

```bash
./scripts/build-wsl.sh
```

### APK install fails because of a signature mismatch

```bash
adb uninstall com.powerampstartradio
```

Then reinstall.
