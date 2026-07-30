# Setup

This guide builds a CLaMP3 music index, installs the Android app from source, and activates the
index through the app. It also covers optional on-device and always-on-server indexing.

## Current Build

The Android project currently builds an arm64 application with ID
`com.powerampstartradio.v2` and version `2.0.0-experimental`.

The four model files are not bundled in the APK. This guide installs a debug build and places the
exported models in its private storage with `adb run-as`.

Run each section from the repository root unless a command names another absolute working
directory.

## Requirements

Desktop indexer:

- Python 3.10 or newer;
- `uv`, or a normal virtual environment and `pip`;
- PyTorch and torchaudio suitable for the machine;
- enough storage for the embedding database, MERT cache, and exported models;
- internet access for the first model download.

Android:

- an arm64 Android 8.0 or newer device;
- Poweramp;
- Android SDK platform tools (`adb`);
- enough private app storage for roughly 1.8 GiB of models plus the music index and generation
  workspace.

A CUDA-capable desktop is strongly recommended for a large first index. The server indexer also
works on CPU, but completes tracks much more slowly.

## 1. Install The Desktop Indexer

With `uv`:

```bash
cd /path/to/poweramp-start-radio/desktop-indexer
uv sync --extra dev --extra export
uv run poweramp-indexer --help
```

Or with a virtual environment:

```bash
cd /path/to/poweramp-start-radio/desktop-indexer
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,export]'
poweramp-indexer --help
```

Install the appropriate PyTorch/torchaudio build for the host before a large scan. Confirm the
selected backend:

```bash
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
print("mps", hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
PY
```

## 2. Build The Initial Music Index

Run a complete-track scan:

```bash
uv run poweramp-indexer scan /path/to/music \
  --output /path/to/embeddings.db
```

The scan has two resumable phases:

1. decode each track and cache its MERT features;
2. encode those features with CLaMP3, write `embeddings.db`, and build the kNN graph used by Graph
   Explorer.

Useful controls:

```bash
# Increase only after measuring available GPU memory.
uv run poweramp-indexer scan /path/to/music -o embeddings.db --batch-size 16

# Prefer FP32 MERT on hardware where FP16 is unsupported or unstable.
uv run poweramp-indexer scan /path/to/music -o embeddings.db --fp32

# Run the expensive cache phase and fast database phase separately.
uv run poweramp-indexer scan /path/to/music -o embeddings.db --phase 1
uv run poweramp-indexer scan /path/to/music -o embeddings.db --phase 2
```

Inspect the result:

```bash
uv run poweramp-indexer info /path/to/embeddings.db
uv run poweramp-indexer similar /path/to/embeddings.db "artist title"
uv run poweramp-indexer search /path/to/embeddings.db "slow psychedelic guitar"
```

Before the database has been imported into Android, update that desktop copy with:

```bash
uv run poweramp-indexer update /path/to/music \
  --database /path/to/embeddings.db
```

`update` rebuilds the Graph Explorer graph when rows change. `--no-rebuild-graph` is a temporary
escape hatch; Graph Explorer will not represent the changed database until `poweramp-indexer graph`
runs.

This command changes only the desktop database. Once a phone has an active index generation, grow
it through On-device indexing or Merge server index. Initial import is available only when the app
has no active generation; replacing app data with a later desktop database requires clearing the
app's data and importing again.

## 3. Export The Android Models

```bash
uv run poweramp-indexer export all --output-dir models
```

The Android app requires these exact names:

```text
models/mert.tflite
models/clamp3_audio.tflite
models/clamp3_text.tflite
models/sentencepiece.bpe.model
```

Use the FP32 exports with the Android indexing path.

## 4. Build And Install Android

From Linux:

```bash
cd /path/to/poweramp-start-radio/android-plugin
./gradlew :app:assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

From WSL, initialize the local Android toolchain if needed:

```bash
cd /path/to/poweramp-start-radio/android-plugin
./scripts/setup-wsl-android-env.sh
./scripts/build-wsl.sh
```

When Windows owns the ADB server, start it from Windows first and use the same server from WSL.
There is no project-specific ADB daemon.

Launch the app once so Android creates its private files directory:

```bash
adb shell monkey -p com.powerampstartradio.v2 1
```

## 5. Install The Model Files

The following method is for the debuggable source build:

```bash
PACKAGE=com.powerampstartradio.v2
MODEL_DIR=/absolute/path/to/desktop-indexer/models

for file in mert.tflite clamp3_audio.tflite clamp3_text.tflite sentencepiece.bpe.model; do
  adb push "$MODEL_DIR/$file" "/data/local/tmp/$file"
  adb shell run-as "$PACKAGE" cp "/data/local/tmp/$file" "files/$file"
  adb shell rm -f "/data/local/tmp/$file"
done
```

In Settings, find `Index and model files` and tap `Show file details`. Every model capability must
report ready before the first music-index import. The app records the exact hashes and reuses that
receipt; it does not rehash unchanged multi-gigabyte files for every indexing run.

## 6. Grant Permissions

On the first fresh activity start, Android asks for Music and audio access. This permission is
needed for server-index matching and local audio indexing.

The app also needs Poweramp's data-provider permission. Tap `Grant Access` when the app reports
that it cannot read Poweramp, then approve the request in Poweramp. These are separate permission
surfaces.

If either was denied permanently, grant it from Android or Poweramp settings and reopen the app.

## 7. Import The Music Index

Put `embeddings.db` somewhere Android's document picker can read, for example:

```bash
adb push /path/to/embeddings.db /sdcard/Download/embeddings.db
```

Then:

1. open Start Radio;
2. open Settings;
3. find `Music index`;
4. choose `Import music index`;
5. select `embeddings.db`;
6. wait for validation and immutable generation activation.

The importer validates the four model/tokenizer files first, then reads and fingerprints the
database, extracts the embedding and graph artifacts, writes a private generation, and publishes
it atomically. Keep enough free internal storage for the source database plus the prepared
generation during import.

Do not copy a database directly over the app's private files. Import, server merge, cleanup, and
on-device indexing are the only supported generation-mutation paths.

## 8. Verify The App

After Poweramp has scanned the same music files:

1. play or pause a track in Poweramp;
2. open Start Radio and confirm `Now Playing` shows that track;
3. start a Closest or MMR radio and confirm the ordered results are queued;
4. open Find Music, search a real description, and queue the displayed result list;
5. inspect Settings `Peek` for a read-only preview if desired.

The app does not need Poweramp playback to be active. A paused current track is a valid seed.

## On-Device Indexing

Use Settings -> `On-device indexing` after Poweramp has scanned newly copied music.

The Settings summary shows the last completed library comparison. Tap its Refresh action for a
current count, or open Manage Tracks to perform and inspect the comparison; merely opening Settings
does not rescan the full Poweramp library.

The app:

- compares the current Poweramp provider snapshot with the active immutable generation;
- reuses a fresh comparison when opening Manage Tracks;
- lets you select, retry, exclude, pause, resume, or discard durable work;
- indexes the complete decoded span in Full speed mode;
- publishes a new generation only after database and graph validation.

Keep the foreground notification enabled. The service owns its wake lock while actively indexing;
an external screen-awake utility is not required for correctness.

## Always-On Server Indexing

The optional server mode turns idle server time into cumulative graphless embeddings for later
phone merge.

From the repository root, copy and edit the example:

```bash
sudo install -d /var/lib/poweramp-server-indexer /var/cache/poweramp-server-indexer
sudo cp desktop-indexer/deploy/server-indexer.example.toml \
  /etc/poweramp-server-indexer.toml
```

Initialize once. `--baseline-existing` means the current contents are already represented by the
phone index; it does not embed them:

```bash
poweramp-indexer server init \
  --config /etc/poweramp-server-indexer.toml \
  --baseline-existing
```

Run and inspect:

```bash
poweramp-indexer server once --config /etc/poweramp-server-indexer.toml
poweramp-indexer server status --config /etc/poweramp-server-indexer.toml
poweramp-indexer server status --config /etc/poweramp-server-indexer.toml --json
poweramp-indexer server retry --config /etc/poweramp-server-indexer.toml
poweramp-indexer server run --config /etc/poweramp-server-indexer.toml
```

For unattended use, adapt `desktop-indexer/deploy/poweramp-server-indexer.service`. The service
must run as a user that can read every listen root and write the state, cache, and export
directories.

Copy the configured `bundle_db` file only when `server status` reports:

- `Queue: empty`;
- no `Blocked:` or `Last cycle error:` line; and
- a current `Published:` receipt at the configured path with the expected completed-track count.

In Android Settings, choose `Merge server index` and select that file. The phone validates
model/source-span compatibility, matches rows against the current Poweramp library, and publishes a
new immutable generation only when compatible new rows exist. Re-merging the same cumulative bundle
is safe and becomes a validated no-op.

## Troubleshooting

### The app says model files are missing

Confirm all four filenames and the package:

```bash
adb shell run-as com.powerampstartradio.v2 ls -lh files
```

Re-export and reinstall the exact file rather than renaming an old model.

### Poweramp access is missing

Use the in-app `Grant Access` action. Android's Music and audio permission does not grant Poweramp
provider access.

### Graph Explorer is unavailable

Import a database with a graph, or rebuild one:

```bash
uv run poweramp-indexer graph /path/to/embeddings.db
```

Server exports are intentionally graphless; the phone merge derives the active graph from the
existing compatible generation.

### A server merge or import is blocked

Finish or discard the durable on-device indexing job first. Music-index mutations and
recommendation work are serialized so they cannot publish competing generations.

### Desktop tests use the wrong Python packages

Run through the managed project environment:

```bash
cd /path/to/poweramp-start-radio/desktop-indexer
uv sync --extra dev --extra export
uv run --extra dev pytest -q
```
