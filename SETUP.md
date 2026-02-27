# Setup Guide

## Prerequisites

- **Python 3.10+** with pip (for desktop indexing)
- **CUDA GPU** recommended (NVIDIA, for faster indexing)
- **Android phone** with Poweramp installed
- **ADB** (Android Debug Bridge)

For building the Android app from source:
- Java 17 (JDK)
- Android SDK (platform 34+, NDK 27)
- Or use the one-time setup script (see below)

## Desktop Indexing

### 1. Install

```bash
cd desktop-indexer
pip install -e .
```

Key dependencies installed automatically: torch, torchaudio, transformers, mutagen, huggingface-hub.

### 2. Generate Embeddings

```bash
# Scan your entire music library
poweramp-indexer scan /path/to/music -o embeddings.db

# Options:
#   --fp16          Half-precision (saves VRAM, no quality loss)
#   --batch-size N  Batch size (default 8; higher = faster, more VRAM)
#   --max-duration 600  Skip files longer than N seconds
```

This runs a two-phase pipeline:
1. **MERT** extracts audio features (768d per 5-second window)
2. **CLaMP3** encodes features into a single 768d embedding per track

Progress is cached per-track — safe to interrupt and resume.

### 3. Build kNN Graph

```bash
poweramp-indexer index embeddings.db
```

Builds k-means clusters and a kNN graph (K=20). Required for the Random Walk recommendation mode.

### 4. Test

```bash
# Find similar tracks
poweramp-indexer similar embeddings.db "radiohead everything in its right place"

# Text search (genre, mood, description)
poweramp-indexer search embeddings.db "dark minimal techno"

# Database info
poweramp-indexer info embeddings.db
```

### 5. Incremental Updates

```bash
poweramp-indexer update /path/to/music -d embeddings.db
```

## Deploy to Phone

### Transfer the Database

```bash
adb push embeddings.db /data/local/tmp/
adb shell run-as com.powerampstartradio cp /data/local/tmp/embeddings.db files/
adb shell rm /data/local/tmp/embeddings.db
```

### Install the App

Pre-built APK:
```bash
adb install app-debug.apk
adb shell pm grant com.powerampstartradio android.permission.READ_MEDIA_AUDIO
```

Or build from source:
```bash
cd android-plugin
bash scripts/setup-wsl-android-env.sh  # one-time: installs JDK, SDK, NDK
bash scripts/build-wsl.sh
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

### Use

1. Open Poweramp, start playing any track
2. Open Poweramp Start Radio
3. The app matches the current track to its embedding database
4. Tap **Start Radio** — the app queues 30 similar tracks in Poweramp
5. Adjust mode (MMR/DPP/Random Walk) and sliders to taste

## On-Device Indexing (Optional)

Index new tracks directly on your phone, without the desktop pipeline. Requires pushing TFLite model files (~722 MB total).

### Export TFLite Models

```bash
cd desktop-indexer
pip install litert-torch==0.8.0 ai-edge-litert jax torch_xla2 --no-deps
poweramp-indexer export all
```

Produces in `models/`:
- `mert.tflite` (~378 MB) — audio feature extractor
- `clamp3_audio.tflite` (~344 MB) — audio encoder
- `clamp3_text.tflite` — text encoder (for text search)
- `xlm_roberta_vocab.json` — tokenizer vocabulary

**All models must be FP32.** FP16 model files are incompatible with the required FP32 GPU precision.

### Push Models to Phone

```bash
for f in mert.tflite clamp3_audio.tflite clamp3_text.tflite xlm_roberta_vocab.json; do
    adb push "models/$f" /data/local/tmp/
    adb shell run-as com.powerampstartradio cp "/data/local/tmp/$f" files/
    adb shell rm "/data/local/tmp/$f"
done
```

### Index on Device

1. Open the app → tap **Manage Tracks**
2. The app detects tracks in Poweramp that aren't in the database
3. Select tracks → tap **Start Indexing**
4. Two-phase GPU pipeline runs: MERT (Phase 1) → CLaMP3 (Phase 2)
5. ~15-30 seconds per 3-minute track on Snapdragon 8 Gen 3

Progress is cached per-track. If interrupted, indexing resumes from the last completed track.

## Troubleshooting

**"Track not found in database"**: The current Poweramp track doesn't match any entry in embeddings.db. Check that the music library scanned on desktop matches what's on the phone.

**"No similar tracks found" with Random Walk**: The seed track may not be in the kNN graph. This happens when the track was indexed on-device after the graph was built. Re-run `poweramp-indexer index embeddings.db` on desktop and re-push the DB, or trigger on-device indexing which rebuilds the graph automatically.

**Signature mismatch on install**: Run `adb uninstall com.powerampstartradio` first, then install again.

**Build fails with NDK errors**: Ensure NDK 27 is installed. The setup script handles this, but manual installations may need `sdkmanager "ndk;27.2.12479018"`.
