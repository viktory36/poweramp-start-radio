# CLAUDE.md

Repository reference for coding agents working in Poweramp Start Radio.

## Overview

Poweramp Start Radio has two main halves:

- `desktop-indexer/`
  - scans a music library, computes CLaMP3 embeddings, and writes `embeddings.db`
- `android-plugin/`
  - matches Poweramp tracks to that database, generates playlists, supports text and multi-seed search, and can index missing tracks on-device

The active model path is CLaMP3:

- audio: MERT -> CLaMP3 audio encoder -> `768d` embedding
- text: CLaMP3 text encoder -> `768d` embedding in the same space

The normal workflow is desktop-first:

1. build `embeddings.db` on a desktop machine
2. copy it to the phone
3. let the Android app extract `clamp3.emb` and `graph.bin` for fast runtime access
4. optionally let the phone index tracks that were not in the desktop database yet

## End-To-End Flow

### Desktop pipeline

```text
Music files
  -> MERT (24kHz waveform -> 768d features per 5s window)
  -> CLaMP3 audio encoder (window features -> one 768d track embedding)
  -> embeddings.db (tracks, embeddings_clamp3, clusters, knn_graph)
```

### Android radio path

```text
Poweramp current track
  -> TrackMatcher
  -> RecommendationEngine
  -> selector (MMR / DPP / Random Walk / Drift path)
  -> TrackMatcher / PowerampHelper
  -> Poweramp queue
```

### Android on-device indexing path

```text
Poweramp library
  -> NewTrackDetector
  -> AudioDecoder + resampler
  -> MertInference
  -> Clamp3AudioInference
  -> embeddings.db update
  -> GraphUpdater
```

## Core Concepts

### Matching

Poweramp tracks are matched to the embedding database primarily through a metadata key:

- `artist|album|title|duration_rounded_to_100ms`

`TrackMatcher` then falls back through progressively looser strategies:

1. exact metadata key
2. artist + album + title without duration
3. artist + title without album
4. fuzzy artist and title normalization
5. filename-based fallback

If matching behavior changes, read `TrackMatcher` and `NewTrackDetector` together.

### Embeddings and similarity

- audio is decoded to `24kHz` mono
- MERT works on non-overlapping `5s` windows
- CLaMP3 produces one `768d` embedding per track
- embeddings are L2-normalized, so dot product equals cosine similarity
- Android hot paths use mmap-backed indices rather than repeated SQLite reads

### Recommendation modes

The user-facing mode names should stay aligned with the selectors.

- `MMR`
  - relevance to the current query minus a penalty from the single chosen track with the greatest overlap
- `DPP`
  - re-scores remaining candidates against the chosen set as a whole through pairwise interactions
- `Random Walk`
  - walks a precomputed kNN graph from the seed track id and ranks reachable terminals
- `Drift`
  - updates the query after each pick on the sequential embedding-scan path
- `Multi-seed`
  - uses `GeoMeanSelector`; it is not the same path as single-seed radio

### Graph expectations

- Random Walk depends on a valid kNN graph
- desktop `scan` builds the graph after embedding generation
- desktop `update` refreshes the graph by default and can skip it with `--no-rebuild-graph`
- on-device indexing updates the graph on the phone

### On-device indexing

A few details matter here because they have caused real regressions.

- Android audio indexing uses FP32 GPU precision; FP16 precision causes numerical collapse
- use FP32 model files on phone for `mert.tflite` and `clamp3_audio.tflite`
- the Android indexing path is split into two GPU phases because the device cannot keep both models active in the way the app needs
- decode chunk boundaries must not create extra MERT windows
- only the final tail of the whole track may be zero-padded, and only if it is at least `1s`
- chunked extraction preserves alignment by carrying leftover samples across chunks
- the text model currently falls back to CPU in practice because its graph uses INT64 ops that the GPU path does not handle cleanly

### Native and packaging notes

- the audio path uses native resampling and math helpers under `android-plugin/app/src/main/cpp/`
- stereo-to-mono conversion must widen to `int32` before summing channels; `int16` addition can wrap on loud masters and corrupt embeddings
- the app ships `arm64-v8a` native libraries only
- `useLegacyPackaging = true` keeps the JNI libraries uncompressed for reliable loading

### Android integration

The app talks to Poweramp through its content provider and broadcast surface.

- content provider: `content://com.maxmpz.audioplayer.data`
- permission request action: `com.maxmpz.audioplayer.ACTION_ASK_FOR_DATA_PERMISSION`
- package visibility is declared through the app manifest `<queries>` section

## Where To Look First

### Radio startup and queueing

- `android-plugin/app/src/main/java/com/powerampstartradio/services/RadioService.kt`
- `android-plugin/app/src/main/java/com/powerampstartradio/poweramp/TrackMatcher.kt`
- `android-plugin/app/src/main/java/com/powerampstartradio/similarity/RecommendationEngine.kt`
- `android-plugin/app/src/main/java/com/powerampstartradio/poweramp/PowerampHelper.kt`

### Selection algorithms

- `android-plugin/app/src/main/java/com/powerampstartradio/similarity/algorithms/MmrSelector.kt`
- `android-plugin/app/src/main/java/com/powerampstartradio/similarity/algorithms/DppSelector.kt`
- `android-plugin/app/src/main/java/com/powerampstartradio/similarity/algorithms/RandomWalkSelector.kt`
- `android-plugin/app/src/main/java/com/powerampstartradio/similarity/algorithms/DriftEngine.kt`
- `android-plugin/app/src/main/java/com/powerampstartradio/similarity/algorithms/GeoMeanSelector.kt`

### Text and multi-seed

- `android-plugin/app/src/main/java/com/powerampstartradio/ui/MainViewModel.kt`
- `android-plugin/app/src/main/java/com/powerampstartradio/indexing/Clamp3TextInference.kt`

### On-device indexing

- `android-plugin/app/src/main/java/com/powerampstartradio/indexing/IndexingActivity.kt`
- `android-plugin/app/src/main/java/com/powerampstartradio/indexing/IndexingService.kt`
- `android-plugin/app/src/main/java/com/powerampstartradio/indexing/AudioDecoder.kt`
- `android-plugin/app/src/main/java/com/powerampstartradio/indexing/MertInference.kt`
- `android-plugin/app/src/main/java/com/powerampstartradio/indexing/Clamp3AudioInference.kt`
- `android-plugin/app/src/main/java/com/powerampstartradio/indexing/GraphUpdater.kt`

### Desktop pipeline

- `desktop-indexer/src/poweramp_indexer/cli.py`
- `desktop-indexer/src/poweramp_indexer/database.py`
- `desktop-indexer/src/poweramp_indexer/graph.py`
- `desktop-indexer/src/poweramp_indexer/export_litert.py`
- `desktop-indexer/scripts/generate_clamp3_embeddings.py`

## Useful Commands

### Desktop CLI

```bash
cd desktop-indexer
python -m pip install -e .

poweramp-indexer scan /path/to/music -o embeddings.db
poweramp-indexer update /path/to/music --database embeddings.db
poweramp-indexer graph embeddings.db --clusters 200 --knn 5
poweramp-indexer similar embeddings.db "artist title"
poweramp-indexer search embeddings.db "dark minimal techno"
poweramp-indexer info embeddings.db
poweramp-indexer export all
```

### Android build

```bash
cd android-plugin
./scripts/setup-wsl-android-env.sh
./scripts/build-wsl.sh
```

### Benchmarks and debug entry points

```bash
adb shell am start -n com.powerampstartradio/.benchmark.BenchmarkActivity \
  --ez auto_start true --ei max_duration_s 0

adb shell am broadcast -a com.powerampstartradio.DEBUG_START_RADIO \
  -n com.powerampstartradio/.debug.DebugRadioReceiver \
  --es selection_mode MMR --ef diversity_lambda 0.4 --ei num_tracks 30

adb shell am broadcast -a com.powerampstartradio.DEBUG_MULTI_SEED \
  -n com.powerampstartradio/.debug.DebugMultiSeedReceiver \
  --es song1 "artist title" --ef weight1 1.0 --ei top_k 10
```

## Validation

A small amount of targeted validation goes a long way in this repo.

- if audio or indexing changes, compare full-track on-device benchmark output against the desktop database
- if chunking changes, check both cosine agreement and MERT window counts on multi-chunk tracks
- if Random Walk changes, make sure the graph is current before judging the results
- if radio startup stalls, inspect `adb logcat` for `RadioService`, `TrackMatcher`, `RecommendationEngine`, and `PowerampHelper`

`EVALUATION.md` collects the measured results and the commands used to reproduce the most important checks.
