# Setup

You need Python 3.10+, an arm64 Android 8+ phone with Poweramp, and a computer with
JDK 17 and the Android SDK. A GPU is useful for the initial library scan; CPU indexing also works.
Allow roughly 1.8 GiB on the phone for models, plus the index and temporary indexing workspace.

The current app package is `com.powerampstartradio.v2`. Models are not bundled with the APK.
The steps below provision a debug build; a non-debuggable build cannot use `adb run-as`.

## 1. Install the Indexer

From the repository root:

```bash
cd desktop-indexer
uv sync --extra dev --extra export
uv run poweramp-indexer --help
```

Alternatively, use a Python virtual environment and `pip install -e '.[dev,export]'`.
Install a matching PyTorch/torchaudio build for your accelerator.

## 2. Index Your Music

From `desktop-indexer/`:

```bash
uv run poweramp-indexer scan /path/to/music -o embeddings.db
uv run poweramp-indexer info embeddings.db
uv run poweramp-indexer similar embeddings.db "artist title"
```

The scan caches MERT features, writes CLaMP3 vectors, and builds the similarity graph.
Rerun the same command to resume. Complete-track processing is the default.
See the [indexer guide](desktop-indexer/README.md) for batching, precision, and update controls.

## 3. Export the Android Models

```bash
uv run poweramp-indexer export all --output-dir models
```

This produces `mert.tflite`, `clamp3_audio.tflite`, `clamp3_text.tflite`, and
`sentencepiece.bpe.model`. Keep these names unchanged. Android indexing uses FP32 exports.

## 4. Build and Install

From the repository root, with the Android toolchain configured:

```bash
cd android-plugin
./gradlew :app:assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb shell monkey -p com.powerampstartradio.v2 1
```

The [Android guide](android-plugin/README.md#build) lists SDK versions and WSL setup commands.

## 5. Copy the Models

After launching the app once, run from `android-plugin/`:

```bash
PACKAGE=com.powerampstartradio.v2
MODEL_DIR=../desktop-indexer/models

for file in mert.tflite clamp3_audio.tflite clamp3_text.tflite sentencepiece.bpe.model; do
  adb push "$MODEL_DIR/$file" "/data/local/tmp/$file"
  adb shell run-as "$PACKAGE" cp "/data/local/tmp/$file" "files/$file"
  adb shell rm -f "/data/local/tmp/$file"
done
```

Settings -> Index and model files -> Show file details reports model readiness.
The first validation records model identities; unchanged model files are not rehashed for each run.

## 6. Grant Access and Import

1. Allow Android's Music and audio permission when requested.
2. Use **Grant Access** in the app to approve Poweramp's separate provider permission.
3. Copy the database to the phone:

   ```bash
   adb push ../desktop-indexer/embeddings.db /sdcard/Download/embeddings.db
   ```

4. In Settings -> Music index, choose **Import music index** and select the file.
5. Let import finish. It validates the database and models, prepares vectors and graph, then
   activates the index. Keep room for both the import file and prepared index.
6. After Poweramp scans your music, play or pause an indexed track and check **Now Playing**.
   Start a radio, or use Find Music and queue its displayed results.

Initial import is available only before an active index exists. Do not overwrite private index
files with ADB. Grow the phone index through on-device indexing or server merge.

## Adding Music

### On the Phone

After Poweramp scans new music, open Settings -> On-device indexing -> Manage Tracks.
Select tracks and start the run. Failed tracks can be retried or marked Never index.
Jobs support pause/resume and recovery after process loss; the service uses a foreground
notification and wake lock while working.

Settings displays the last library comparison. Refresh or Manage Tracks performs a new comparison;
opening Settings alone does not scan the entire library.

### From a Server

Follow [Server indexing](desktop-indexer/README.md#server-indexing) to watch download folders.
Sync the music files to the phone, let Poweramp scan them, then copy the configured `bundle_db`
export and select **Merge server index** in Settings.

The merger matches compatible rows to the current Poweramp library and skips already indexed rows.
Repeatedly merging the same cumulative export is safe. Keep the original server folder structure
when syncing music so path matching has the best evidence.

## Troubleshooting

| Problem | Action |
| --- | --- |
| Missing models | Check exact names with `adb shell run-as com.powerampstartradio.v2 ls -lh files`; reinstall the exported bundle |
| Cannot read Poweramp | Use Grant Access; Android audio permission does not grant Poweramp access |
| Music missing from index/merge | Scan the copied files in Poweramp first, then refresh the app's library comparison |
| Graph Explorer unavailable | Build the desktop graph with `poweramp-indexer graph embeddings.db` before initial import |
| Merge/import blocked by indexing | Finish or discard the pending indexing job first |
| WSL cannot reach the phone | If using Windows' ADB server, start it from Windows before using ADB in WSL |

Server exports intentionally contain no graph; the phone updates its graph during merge.
Developer options and USB debugging are needed for installation and diagnostics, not ordinary app use.
