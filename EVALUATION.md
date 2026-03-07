# Evaluation Notes

This document collects measured results for recommendation behavior, text search, on-device indexing, and track matching. It also includes the commands used to rerun the targeted checks that are most useful during development.

## Test Context

The broad product audit in this file was measured on:

- Sony XQ-EC72
- Snapdragon 8 Gen 3 / Adreno 740
- Android 16 (SDK 36)
- `75,035` indexed tracks total
- `74,288` desktop-built + `747` on-device indexed

The latest targeted audio-path validation in this file was run on March 7, 2026.

## Latest Targeted Validation

### Full-track on-device audio benchmark

Recent full-track validation against the desktop database passed with:

- mean cosine vs desktop: `0.995522`
- min cosine: `0.990923`
- max cosine: `0.998122`
- device mean pairwise cosine: `0.2451`
- desktop mean pairwise cosine: `0.2438`

This indicates that the current on-device embedding path remains closely aligned with the desktop reference and is not showing the numerical-collapse pattern associated with precision mistakes.

### Multi-chunk windowing alignment

The full-track benchmark also confirmed that chunked extraction is following the desktop windowing rule on sampled long tracks:

- `282s -> 57 windows`
- `164s -> 33 windows`
- `350s -> 70 windows`
- `603s -> 121 windows`
- `411s -> 83 windows`

These all match the intended rule:

- full `5s` windows
- one final padded partial window only if the whole-track tail is at least `1s`

## Recommendation Modes

Controlled experiments used the same seed track, requested `30` tracks, and changed one parameter at a time.

Seed used for the snapshot below:

- My Terrible Friend - Almost Gone

### MMR (Maximal Marginal Relevance)

`lambda` controls the relevance/diversity tradeoff. Higher `lambda` keeps the list tighter around the seed. Lower `lambda` spends more of the list on diversity.

| Lambda | Tracks | Artists | Sim Range | Mean Sim |
|--------|--------|---------|-----------|----------|
| 0.3    | 30     | 30      | 62-88%    | 66.1%    |
| 0.6    | 28     | 22      | 75-88%    | 82.3%    |
| 0.9    | 28     | 20      | 81-88%    | 83.3%    |

Highlights:

- `lambda=0.3` produced `30` unique artists out of `30` tracks
- `lambda=0.9` produced the tightest cluster around the seed
- the slider gave a smooth gradient between exploration and focus

### DPP (Determinantal Point Process)

| Tracks | Artists | Sim Range | Mean Sim |
|--------|---------|-----------|----------|
| 30     | 29      | 62-88%    | 69.8%    |

Highlights:

- artist spread was close to low-`lambda` `MMR`
- the quality floor stayed stronger than `MMR` at `lambda=0.3`, with fewer low-similarity outliers
- selection time in this snapshot was about `263ms` vs `127ms` for `MMR`

### Random Walk

The measured configuration used `10,000` walks on a `K=5` kNN graph with terminal-only counting and non-backtracking.

| Alpha | Tracks | Artists | Sim Range | Mean Sim | Hop Distribution |
|-------|--------|---------|-----------|----------|-----------------|
| 0.95  | 22     | 16      | 73-88%    | 81.2%    | 1:4, 2:11, 3:7  |
| 0.50  | 30     | 21      | 69-88%    | 80.4%    | 1:4, 2:11, 3:15 |
| 0.05  | 29     | 21      | 43-88%    | 78.3%    | 1:4, 2:8, 3:8, 4:6, 6:2, 9:1 |

Highlights:

- lower `alpha` explored farther from the seed neighborhood
- the low-`alpha` run reached hop `9` and dropped the similarity floor to `43%`
- high `alpha` stayed much closer to the seed and returned fewer unique terminals on the narrow graph
- walk computation itself stayed under `10ms`

### Drift Mode

Drift modifies the query after each pick, so the playlist progressively moves away from the initial seed.

| Drift Mode | Tracks | Artists | Sim Range | Mean Sim |
|-----------|--------|---------|-----------|----------|
| Seed Interpolation | 30 | 30 | 60-88% | 74.3% |
| Momentum | 30 | 29 | 57-88% | 72.1% |

Highlights:

- seed interpolation produced the more controlled drift
- momentum pushed farther and widened the similarity range more aggressively
- both variants reduced artist clustering naturally
- sequential drift cost much more than batch selection: about `2.5s` for `30` steps in this snapshot

### Post-filter (artist constraints)

| Setting | Tracks | Artists | Mean Sim |
|---------|--------|---------|----------|
| `maxPerArtist=8, minSpacing=3` | 28-30 | 20-30 | varies |
| `maxPerArtist=100` (off) | 30 | 27 | 77.6% |

Highlights:

- the post-filter removed repeated artists that the selector alone would allow through
- its effect depended on the seed and library composition
- turning it off reduced artist spread in the measured example

## Text Search

CLaMP3's shared audio-text space produced coherent semantic retrieval in the large-library audit.

### Genre queries

| Query | Mean Score | Top Artist |
|-------|------------|------------|
| psychedelic trance | 0.317 | Hallucinogen |
| jazz fusion | 0.310 | John McLaughlin |
| progressive metal | 0.282 | Tool, Opeth |
| ambient electronic | 0.282 | Gas |
| dark minimal techno | 0.264 | Plastikman |
| shoegaze dream pop | 0.263 | My Bloody Valentine |
| 90s boom bap hip hop | 0.243 | A Tribe Called Quest |
| brazilian mpb bossa nova | 0.236 | Elis Regina |
| sufi devotional music | 0.203 | Nusrat Fateh Ali Khan |
| indian classical raga | 0.198 | L. Subramaniam |

### Mood and texture queries

| Query | Mean Score | Example Results |
|-------|------------|-----------------|
| melancholy atmospheric | 0.306 | Burial, Radiohead |
| energetic upbeat dance | 0.281 | Underworld |
| calm meditation peaceful | 0.269 | Brian Eno |
| aggressive heavy distorted | 0.268 | Melvins, Unsane |
| romantic strings orchestral | 0.276 | Ravel, Debussy |

Highlights:

- results clustered coherently by genre and mood rather than returning arbitrary near neighbors
- score ranges around `0.19-0.33` were already enough to surface useful semantic groupings
- warm-query inference was about `23ms`, with a slower first query due to model warmup

## On-Device Indexing

Two-phase GPU pipeline on Adreno 740 with FP32 precision.

### Performance snapshot

| Metric | Value |
|--------|-------|
| MERT inference | ~`200ms` per `5s` window |
| Typical `3`-minute track | ~`12-15s` total |
| CLaMP3 audio encode | ~`50ms` per track |
| Per-track cache | resume-friendly |
| Graph rebuild | automatic when the on-device graph needs refreshing |

### Quality snapshot

The current full-track benchmark result is:

- mean cosine vs desktop: `0.995522`
- min cosine vs desktop: `0.990923`
- max cosine vs desktop: `0.998122`

This is consistent with a healthy FP32 Android audio path.

## Track Matching

Matching between the Poweramp library and the embedding database was audited with these strategies:

| Strategy | Method |
|----------|--------|
| 1 | exact metadata key (`artist|album|title|duration`) |
| 2 | prefix match / looser duration handling |
| 3 | artist + title without album |
| 4 | fuzzy artist normalization and related fallbacks |

Measured result:

- `74,265` matched
- `718` genuinely unmatched
- zero false matching failures observed in the audit

This matters because strong retrieval quality still depends on landing on the correct Poweramp file at runtime.

## Performance Summary

Representative timings from the large-library snapshot:

| Operation | Time |
|-----------|------|
| embedding retrieval (`1500` candidates) | `9ms` |
| `MMR` selection (`30` tracks) | `127ms` |
| `DPP` selection (`30` tracks) | `263ms` |
| Random Walk (`10K` walks, `K=5` graph) | `<10ms` |
| drift playlist (`30` steps) | `2500ms` |
| track mapping, first run | `8000ms` |
| track mapping, cached | `1ms` |
| full radio pipeline, batch path | `300-500ms` |
| text search query | `23-70ms` |
| on-device MERT, per window | `200ms` |

These numbers are best read as a healthy reference point on similar hardware, not as universal promises.

## Reproducing The Key Checks

### Full-track Android benchmark

```bash
adb shell am start -n com.powerampstartradio/.benchmark.BenchmarkActivity \
  --ez auto_start true --ei max_duration_s 0
```

### Pull benchmark JSON

```bash
adb shell run-as com.powerampstartradio cat files/benchmark_results.json > /tmp/benchmark_results.json
```

### Compare with the desktop database

```bash
python3 desktop-indexer/scripts/validate_benchmark.py \
  /tmp/benchmark_results.json \
  desktop-indexer/audit_raw_data/embeddings_clamp3.db
```

### Desktop TFLite validation

```bash
python3 desktop-indexer/scripts/validate_tflite_clamp3.py \
  --db desktop-indexer/audit_raw_data/embeddings_clamp3.db \
  --music /path/to/music_subset --n 20
```

## Interpreting Results

### Low cosine in `validate_benchmark.py`

- if the Android benchmark was capped to `120s` while the desktop DB stores full-track embeddings, the comparison is not meaningful
- if both sides used the full track and cosine drops well below the current `~0.995` range, inspect the audio path closely

### Pairwise cosine collapse

If device pairwise similarity rises much higher than desktop pairwise similarity across unrelated tracks, the model path is collapsing numerically. In practice this has been tied to precision mistakes rather than recommendation logic.

### Window count overruns during indexing

If logs show progress such as `67/55`, the extractor is creating extra windows at chunk boundaries. The intended behavior is full `5s` windows plus at most one final padded partial window for the whole track.
