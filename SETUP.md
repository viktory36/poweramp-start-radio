# Setup Guide

This guide takes a fresh checkout to a fully working setup:

- desktop-built `embeddings.db`
- Android app installed from source
- current-track radio, text search, multi-seed search, and Random Walk working on the phone
- on-device indexing ready for tracks added directly on the phone

The setup has two parts:

1. build the desktop database and export the Android model files
2. build the Android app, install it, and copy those files to the phone

## What You Need

### Desktop

- Python `3.10+`
- `pip`
- virtualenv support (`python3-venv` on Debian/Ubuntu/WSL)
- enough free disk space for model downloads, `embeddings.db`, exported LiteRT models, and any cache files
- internet access on first run so Hugging Face weights can download
- a GPU-backed PyTorch install is strongly recommended
  - NVIDIA/CUDA or Apple Silicon/MPS are the practical paths for real library indexing
  - CPU is best kept for debugging and small test runs

On Debian/Ubuntu/WSL, a common starting point is:

```bash
sudo apt install python3 python3-venv python3-pip
```

### Android

- an Android device with Poweramp installed
- `adb`

### If You Are Building The Android App In WSL

- `curl`
- `unzip`
- `tar`

## Recommended Flow

### 1. Create a Python environment

```bash
cd desktop-indexer
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2. Install PyTorch and torchaudio for your machine

Install PyTorch first, especially if you want GPU acceleration.

For CUDA users, this step decides whether indexing runs on the GPU or on the CPU. The usual open-source pattern is:

1. install a CUDA-enabled `torch` and `torchaudio` pair using the official PyTorch command for your OS, Python version, and CUDA runtime
2. verify that `torch.cuda.is_available()` is `True`
3. only then install this repo's package

If you already have a working CUDA PyTorch environment, reuse it.

A quick sanity check after installation:

```bash
python - <<'PY'
import torch, torchaudio
print('torch', torch.__version__)
print('torchaudio', torchaudio.__version__)
print('cuda', torch.cuda.is_available())
print('mps', getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available())
PY
```

### 3. Install the desktop indexer and export dependencies

For the full app feature set, install the package with its export extras:

```bash
python -m pip install -e '.[export]'
```

This installs:

- the desktop CLI dependencies from `pyproject.toml`
- the LiteRT export dependencies needed by `poweramp-indexer export all`

For onboarding, install with the export extras so the full desktop and Android flow is ready from the start.

### 4. Scan your library

```bash
poweramp-indexer scan /path/to/music -o ../embeddings.db
```

That command:

1. decodes audio and resamples it to `24kHz` mono
2. extracts MERT features in non-overlapping `5s` windows
3. encodes those features into one `768d` CLaMP3 embedding per track
4. writes `embeddings.db`
5. builds the clusters and kNN graph used by Random Walk

Useful options:

- `--fp32`
  - run desktop MERT in full precision
- `--batch-size N`
  - larger values improve throughput if your GPU has room
- `--max-duration 600`
  - caps unusually long files
- `--phase 1` or `--phase 2`
  - useful for debugging the two pipeline stages separately

The first run downloads the MERT and CLaMP3 weights automatically.

Desktop MERT defaults to FP16. Use `--fp32` when you want the full-precision path.

### 5. Export the Android model files

Still in `desktop-indexer/`:

```bash
poweramp-indexer export all --output-dir ../models
```

This creates the phone-side assets in the repo-root `models/` directory:

- `mert.tflite`
- `clamp3_audio.tflite`
- `clamp3_text.tflite`
- `xlm_roberta_vocab.json`

For the full feature set on Android, copy all four files to the phone.

File groups:
- on-device indexing: `mert.tflite` and `clamp3_audio.tflite`
- text and multi-seed search: `clamp3_text.tflite` and `xlm_roberta_vocab.json`

Note: Android audio indexing should use the FP32 audio models. `poweramp-indexer export all` produces the correct audio files for the current Android path.

### 6. Build and install the Android app

```bash
cd ../android-plugin
./scripts/setup-wsl-android-env.sh
./scripts/build-wsl.sh
```

`build-wsl.sh` assembles a debug APK and installs it automatically if `adb` sees a device.

### 7. Put the database on the phone

The safest path is:

1. copy the database to normal phone storage
2. import it from inside the app

From the repo root:

```bash
cd ..
adb push embeddings.db /sdcard/Download/
```

Then in the app:

1. open `Settings`
2. open the `Database` section
3. choose `Import Database`
4. pick `embeddings.db` from `Downloads`

This import path is preferred because the app:

- copies the database atomically
- deletes stale derived files such as `clamp3.emb` and `graph.bin`
- rebuilds its fast runtime indices with progress updates

### 8. Copy the phone-side model files

From the repo root:

```bash
for f in mert.tflite clamp3_audio.tflite clamp3_text.tflite xlm_roberta_vocab.json; do
  adb push "models/$f" /data/local/tmp/
  adb shell run-as com.powerampstartradio cp "/data/local/tmp/$f" files/
  adb shell rm "/data/local/tmp/$f"
done
```

These commands use `run-as`, so they assume the debug build from step 6.

After this, `Settings -> App Files` should show the database and the model files.

### 9. First launch

1. open Poweramp and start playback
2. open Poweramp Start Radio
3. grant Poweramp data access if prompted
4. confirm the database appears in `Settings`
5. use `Start Radio` for current-track radio
6. use search for text and multi-seed retrieval
7. use `Manage Tracks` when you want to index tracks that only exist on the phone

## Updating An Existing Database

When your desktop music library changes:

```bash
cd desktop-indexer
source .venv/bin/activate
poweramp-indexer update /path/to/music --database ../embeddings.db
```

`update` adds new files, can remove missing ones, and refreshes the Random Walk graph by default.

If you want to skip graph rebuilding temporarily:

```bash
poweramp-indexer update /path/to/music --database ../embeddings.db --no-rebuild-graph
```

You can rebuild the graph explicitly later:

```bash
poweramp-indexer graph ../embeddings.db --clusters 200 --knn 5
```

After updating the database on desktop, re-import it through the app's `Import Database` flow so the phone refreshes its derived files cleanly.

## Validation

The most useful Android audio-path validation is the full-track benchmark.

### Run the benchmark

```bash
adb shell am start -n com.powerampstartradio/.benchmark.BenchmarkActivity \
  --ez auto_start true --ei max_duration_s 0
```

### Pull the result JSON

```bash
adb shell run-as com.powerampstartradio cat files/benchmark_results.json > /tmp/benchmark_results.json
```

### Compare it with the desktop database you imported

```bash
python3 desktop-indexer/scripts/validate_benchmark.py \
  /tmp/benchmark_results.json \
  embeddings.db
```

`EVALUATION.md` records the current measured results and the larger product-level audit snapshot.

## Troubleshooting

### `poweramp-indexer` is not found

Activate the virtualenv again:

```bash
cd desktop-indexer
source .venv/bin/activate
```

### Scanning is extremely slow

Verify that PyTorch sees the accelerator you expected. If `torch.cuda.is_available()` is `False` on a CUDA machine, the scan is probably running on CPU.

### `poweramp-indexer export all` fails with missing LiteRT modules

Reinstall with export extras:

```bash
cd desktop-indexer
source .venv/bin/activate
python -m pip install -e '.[export]'
```

### Text search is unavailable on the phone

Make sure these two files are present in app storage:

- `clamp3_text.tflite`
- `xlm_roberta_vocab.json`

### On-device indexing says the audio models are missing

Make sure these two files are present in app storage:

- `mert.tflite`
- `clamp3_audio.tflite`

### Random Walk returns poor or empty results after a desktop update

The graph may be stale or missing. Run `poweramp-indexer update` without `--no-rebuild-graph`, or rebuild it explicitly with `poweramp-indexer graph`, then re-import the refreshed database on the phone.

### Advanced: manual `adb` database replacement

If you replace `files/embeddings.db` directly with `adb shell run-as ... cp`, also remove stale derived files before launching the app again:

```bash
adb shell run-as com.powerampstartradio rm -f files/clamp3.emb files/graph.bin
```

The in-app `Import Database` flow is the safer default.
