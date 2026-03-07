# Validation Notes

This file documents how the project is validated today and records the latest benchmark snapshot that is still representative of the current code.

It is intentionally not a grab-bag of timeless performance claims. If the code changes, rerun the commands below.

## What To Validate

When changing the audio or indexing stack, there are two questions:

1. Do the on-device embeddings still agree with the desktop reference?
2. Does chunked on-device extraction preserve the desktop windowing rule?

Those are separate checks and both matter.

## Current Verified Snapshot

Device used for the latest verified run:

- Sony XQ-EC72
- Snapdragon 8 Gen 3 / Adreno 740
- Android 16 (SDK 36)

Date of the snapshot in this file:

- March 7, 2026

## Full-Track On-Device Audio Benchmark

Use the full-track benchmark, not a capped `120s` run, when comparing device embeddings to the desktop database.

### Run benchmark on device

```bash
adb shell am start -n com.powerampstartradio/.benchmark.BenchmarkActivity \
  --ez auto_start true --ei max_duration_s 0
```

### Pull benchmark JSON

```bash
adb shell run-as com.powerampstartradio cat files/benchmark_results.json > /tmp/benchmark_results.json
```

### Validate against the desktop database

```bash
python3 desktop-indexer/scripts/validate_benchmark.py \
  /tmp/benchmark_results.json \
  desktop-indexer/audit_raw_data/embeddings_clamp3.db
```

### Latest result

Recent full-track validation passed with:

- mean cosine vs desktop: `0.995522`
- min cosine: `0.990923`
- max cosine: `0.998122`
- device mean pairwise cosine: `0.2451`
- desktop mean pairwise cosine: `0.2438`

That is the result to beat or at least stay close to for the current code path.

## Multi-Chunk Windowing Check

A separate benchmark change validated that chunked extraction is now aligned to the whole-track desktop rule again.

Recent full-track benchmark samples:

- `282s -> 57 windows`
- `164s -> 33 windows`
- `350s -> 70 windows`
- `603s -> 121 windows`
- `411s -> 83 windows`

All of these match the desktop rule exactly:

- full `5s` windows
- plus one final padded partial window only if the final tail is at least `1s`

This matters because the older bug was not just a progress-display issue. Chunk-local tail padding could create extra MERT windows near the end of a track.

## How To Interpret Failures

### Low cosine in `validate_benchmark.py`

Check whether the benchmark used full tracks.

- If the device benchmark was capped to `120s` but the desktop DB stores full-track embeddings, the comparison is not meaningful.
- If both sides used the full track and cosine drops materially below the current `~0.995` range, investigate the audio path.

### Pairwise cosine collapse

If device pairwise similarity becomes much higher than desktop pairwise similarity across unrelated tracks, the model path is collapsing numerically.

Historically, this has been caused by precision mistakes rather than by recommendation logic.

### Window count overrun during indexing

If logs show progress like `67/55`, the chunked extractor is leaking extra windows at chunk boundaries.

The intended behavior is:

- decode in chunks for memory safety
- keep leftover samples between chunks
- only allow the final whole-track tail to become a padded partial window

## Related Commands

### Desktop TFLite validation

```bash
python3 desktop-indexer/scripts/validate_tflite_clamp3.py \
  --db desktop-indexer/audit_raw_data/embeddings_clamp3.db \
  --music /path/to/music_subset --n 20
```

### Debug radio launch

```bash
adb shell am broadcast -a com.powerampstartradio.DEBUG_START_RADIO \
  -n com.powerampstartradio/.debug.DebugRadioReceiver \
  --es selection_mode MMR --ef diversity_lambda 0.4 --ei num_tracks 30
```

### Debug multi-seed launch

```bash
adb shell am broadcast -a com.powerampstartradio.DEBUG_MULTI_SEED \
  -n com.powerampstartradio/.debug.DebugMultiSeedReceiver \
  --es song1 "artist title" --ef weight1 1.0 --ei top_k 10
```
