# Desktop Indexer

This directory contains the desktop pipeline that builds the database used by the Android app.

The active production path is CLaMP3:

- MERT extracts `768d` features from `5s` windows of audio
- the CLaMP3 audio encoder turns those features into one `768d` embedding per track
- the same embedding space is also used by the text encoder

## Install

```bash
cd desktop-indexer
python -m pip install -e .
```

## Main CLI Commands

### Scan a library from scratch

```bash
poweramp-indexer scan /path/to/music -o embeddings.db
```

Useful options:

- `--fp16`
- `--batch-size N`
- `--max-duration 600`
- `--cache-dir /path/to/mert_cache`
- `--phase 1`
- `--phase 2`

`scan` performs both embedding generation and graph building.

### Incrementally update an existing database

```bash
poweramp-indexer update /path/to/music --database embeddings.db
```

Important:

- `update` adds new files and can remove missing ones
- `update` does not rebuild the kNN graph automatically
- after `update`, rebuild the graph before copying the database back to Android:

```bash
poweramp-indexer graph embeddings.db --clusters 200 --knn 5
```

### Inspect the database

```bash
poweramp-indexer info embeddings.db
```

### Query similar tracks from the desktop database

```bash
poweramp-indexer similar embeddings.db "radiohead karma police"
poweramp-indexer similar embeddings.db --file ~/Downloads/song.mp3
poweramp-indexer similar embeddings.db --random
```

### Text-to-audio search

```bash
poweramp-indexer search embeddings.db "dark minimal techno"
```

### Export models for Android

```bash
poweramp-indexer export all
```

The current Android-relevant outputs are:

- `mert.tflite`
- `clamp3_audio.tflite`
- `clamp3_text.tflite`
- `xlm_roberta_vocab.json`

## What The Database Contains

`embeddings.db` stores:

- track metadata in `tracks`
- audio embeddings in `embeddings_clamp3`
- cluster centroids in `clusters`
- the Random Walk graph in `binary_data.knn_graph`
- metadata such as version, source path, and embedding dimension

On Android, that database is later materialized into mmap files for the hot paths.

## Validation Scripts

The most useful scripts in this directory are:

- `scripts/validate_tflite_clamp3.py`
  - checks desktop TFLite audio output against the desktop database
- `scripts/validate_benchmark.py`
  - checks on-device benchmark JSON against the desktop database
- `scripts/replay_multiseed.py`
  - replays logged multi-seed runs
- `scripts/test_incremental_graph.py`
  - exercises the incremental graph path used by on-device indexing
- `scripts/evaluate_clamp3.py`
  - ad hoc offline similarity and text-search inspection

## Notes For Future Readers

- The CLI entry point is `poweramp_indexer.cli:cli`.
- `scan` builds the graph; `update` does not.
- The Android app relies on the graph for Random Walk mode.
- Legacy MuLan and Flamingo artifacts still exist in `models/` and audit files, but they are not the active production path for the current Android app.
