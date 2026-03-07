# CLAUDE.md

This file gives repository-specific guidance to coding agents working in this project.

## What This Repo Is

Poweramp Start Radio is a desktop-plus-Android system for local music similarity and text-to-audio retrieval.

- `desktop-indexer/` builds the library database on a desktop machine.
- `android-plugin/` is the Android app that talks to Poweramp, serves recommendations, and can index new tracks on-device.
- The project root `README.md` is intentionally left for a hand-written overview and should not be auto-filled.

The active model path today is CLaMP3:

- audio: MERT -> CLaMP3 audio encoder -> 768d embedding
- text: CLaMP3 text encoder -> 768d embedding in the same space

There are legacy experiment artifacts in `desktop-indexer/models/` and other audit files, but the production path is the CLaMP3 pipeline above.

## Current Product Surfaces

The Android app currently has four meaningful surfaces:

1. Single-seed radio from the current Poweramp track
2. Text search and multi-seed search
3. On-device indexing of tracks missing from the desktop database
4. Benchmark / debug entry points launched through `adb`

## Repository Map

- `desktop-indexer/`
  - CLI for scanning, updating, exporting models, graph building, and offline evaluation
- `android-plugin/`
  - Compose UI, Poweramp integration, recommendation engine, on-device indexing, benchmark tooling
- `SETUP.md`
  - practical setup and deployment instructions
- `EVALUATION.md`
  - current validation method and recent verified benchmark snapshot

## Data Artifacts

The important runtime artifacts are:

- `embeddings.db`
  - canonical SQLite database shared between desktop and phone
  - stores track metadata, `embeddings_clamp3`, clusters, and `knn_graph`
- `clamp3.emb`
  - mmap-friendly embedding dump extracted from `embeddings.db` on Android
- `graph.bin`
  - mmap-friendly kNN graph for Random Walk on Android
- `mert.tflite`
  - required for on-device indexing
- `clamp3_audio.tflite`
  - required for on-device indexing
- `clamp3_text.tflite` and `xlm_roberta_vocab.json`
  - optional, required only for on-device text search

## Hot Paths

If a user reports wrong recommendations, slow startup, or indexing problems, start here.

### Radio startup

`RadioService` -> `TrackMatcher` -> `RecommendationEngine` -> `PowerampHelper`

Relevant files:

- `android-plugin/app/src/main/java/com/powerampstartradio/services/RadioService.kt`
- `android-plugin/app/src/main/java/com/powerampstartradio/poweramp/TrackMatcher.kt`
- `android-plugin/app/src/main/java/com/powerampstartradio/similarity/RecommendationEngine.kt`
- `android-plugin/app/src/main/java/com/powerampstartradio/poweramp/PowerampHelper.kt`

### Batch selection algorithms

- `MMR`: `MmrSelector.kt`
- `DPP`: `DppSelector.kt`
- `Random Walk`: `RandomWalkSelector.kt`
- `Drift`: `DriftEngine.kt`

### Text and multi-seed

- `MainViewModel.kt`
- `GeoMeanSelector.kt`
- `Clamp3TextInference.kt`

### On-device indexing

`IndexingActivity` / `IndexingViewModel` -> `IndexingService` -> `NewTrackDetector` -> `AudioDecoder` -> `MertInference` -> `Clamp3AudioInference` -> `GraphUpdater`

Relevant files:

- `android-plugin/app/src/main/java/com/powerampstartradio/indexing/IndexingActivity.kt`
- `android-plugin/app/src/main/java/com/powerampstartradio/indexing/IndexingViewModel.kt`
- `android-plugin/app/src/main/java/com/powerampstartradio/indexing/IndexingService.kt`
- `android-plugin/app/src/main/java/com/powerampstartradio/indexing/AudioDecoder.kt`
- `android-plugin/app/src/main/java/com/powerampstartradio/indexing/MertInference.kt`
- `android-plugin/app/src/main/java/com/powerampstartradio/indexing/Clamp3AudioInference.kt`
- `android-plugin/app/src/main/java/com/powerampstartradio/indexing/GraphUpdater.kt`

## Recommendation Behavior That Matters

The user-facing mode names should stay aligned to the actual selectors.

- `MMR`
  - candidate relevance to the current query minus a penalty from the single chosen track it overlaps with most
- `DPP`
  - re-scores remaining candidates against the chosen set as a whole through pairwise interactions
- `Random Walk`
  - ranks tracks by walking a precomputed similarity graph from the seed track id
- `Drift`
  - only meaningful on the sequential embedding-scan path; it is disabled for `DPP` and not used for `Random Walk`
- `Multi-seed`
  - does not reuse the batch radio selectors; it uses `GeoMeanSelector`

## Invariants Worth Protecting

### Matching

- metadata key format is `artist|album|title|duration_rounded_to_100ms`
- matching falls back through progressively looser strategies in `TrackMatcher`
- do not casually change matching rules without checking `NewTrackDetector` and `TrackMatcher` together

### Embeddings and similarity

- embeddings are L2-normalized
- dot product equals cosine similarity
- `EmbeddingIndex` and `GraphIndex` are mmap-backed for hot loops

### On-device audio windowing

- MERT windows are `5s`, non-overlapping
- only the final tail of the whole track may be zero-padded if it is at least `1s`
- decode chunk boundaries must not create extra padded windows
- the current implementation preserves alignment by carrying leftover samples across chunks

### Graph expectations

- Random Walk depends on a valid kNN graph
- desktop `scan` builds a graph after phase 2
- desktop `update` does not rebuild the graph automatically; run `poweramp-indexer graph` afterward if the desktop DB changed
- on-device indexing updates the graph on phone

## On-Device Model Requirements

The Android app's model requirements are easy to get wrong.

- use FP32 model files on phone for audio indexing
- LiteRT GPU precision is forced to FP32 in `LiteRtUtils.kt`
- FP16 audio model files are not part of the supported production path for on-device indexing
- text search currently falls back to CPU in practice because the text model uses INT64 ops that the GPU path does not handle

Approximate current model sizes in `desktop-indexer/models/`:

- `mert.tflite`: ~361 MB
- `clamp3_audio.tflite`: ~328 MB
- `clamp3_text.tflite`: ~1.1 GB
- `xlm_roberta_vocab.json`: ~11 MB

## Common Commands

### Desktop CLI

```bash
cd desktop-indexer
python -m pip install -e .

poweramp-indexer scan /path/to/music -o embeddings.db
poweramp-indexer update /path/to/music --database embeddings.db
poweramp-indexer graph embeddings.db --clusters 200 --knn 5
poweramp-indexer info embeddings.db
poweramp-indexer similar embeddings.db "artist title"
poweramp-indexer search embeddings.db "dark minimal techno"
poweramp-indexer export all
```

### Android build

```bash
cd android-plugin
./scripts/setup-wsl-android-env.sh
./scripts/build-wsl.sh
```

`build-wsl.sh` assembles a debug APK and installs it automatically if `adb` sees a device.

### Benchmark and validation

Run a full-track on-device audio benchmark:

```bash
adb shell am start -n com.powerampstartradio/.benchmark.BenchmarkActivity \
  --ez auto_start true --ei max_duration_s 0
```

Pull results:

```bash
adb shell run-as com.powerampstartradio cat files/benchmark_results.json > /tmp/benchmark_results.json
```

Validate against the desktop database:

```bash
python3 desktop-indexer/scripts/validate_benchmark.py \
  /tmp/benchmark_results.json \
  desktop-indexer/audit_raw_data/embeddings_clamp3.db
```

### Debug receivers

```bash
adb shell am broadcast -a com.powerampstartradio.DEBUG_START_RADIO \
  -n com.powerampstartradio/.debug.DebugRadioReceiver \
  --es selection_mode MMR --ef diversity_lambda 0.4 --ei num_tracks 30

adb shell am broadcast -a com.powerampstartradio.DEBUG_MULTI_SEED \
  -n com.powerampstartradio/.debug.DebugMultiSeedReceiver \
  --es song1 "artist title" --ef weight1 1.0 --ei top_k 10
```

## Validation Rules

Prefer code-backed validation over old notes.

- If you are checking audio embedding quality, use the full-track benchmark path, not a truncated benchmark run.
- If `validate_benchmark.py` reports low cosine but the benchmark capped audio length, the comparison is not meaningful.
- If a change touches indexing chunking, verify both:
  - emitted window counts on multi-chunk tracks
  - benchmark cosine vs the desktop database
- If a change touches the radio startup path, inspect `adb logcat` for `RadioService`, `TrackMatcher`, `RecommendationEngine`, and `PowerampHelper` rather than assuming the selector math is the bottleneck.

## Editing Guidance

- Keep root `README.md` untouched for the user to write by hand.
- Prefer updating stale numbers into reproducible commands or clearly labeled snapshots.
- When documenting behavior, trust the current code over historical experiment text.
