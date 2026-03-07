# Desktop Indexer

This directory contains the desktop pipeline that builds the database used by the Android app.

## Overview

The production path is CLaMP3:

- MERT extracts `768d` features from `5s` windows of audio
- the CLaMP3 audio encoder turns those window features into one `768d` embedding per track
- the CLaMP3 text encoder produces embeddings in the same space for text search

The resulting `embeddings.db` is portable. Once built, it can be copied to the phone and used offline by the Android app.

## Install

```bash
cd desktop-indexer
python -m pip install -e .
```

Large libraries benefit from a GPU-capable PyTorch installation, but the CLI can still run without one.

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

`update` adds new files, can remove missing ones, and refreshes the Random Walk graph by default. Use `--no-rebuild-graph` if you want to skip that step temporarily.

You can also rebuild the graph explicitly:

```bash
poweramp-indexer graph embeddings.db --clusters 200 --knn 5
```

### Inspect the database

```bash
poweramp-indexer info embeddings.db
```

### Query similar tracks

```bash
poweramp-indexer similar embeddings.db "radiohead karma police"
poweramp-indexer similar embeddings.db --file ~/Downloads/song.mp3
poweramp-indexer similar embeddings.db --random
```

### Text-to-audio search

```bash
poweramp-indexer search embeddings.db "dark minimal techno"
```

### Export Android model files

```bash
poweramp-indexer export all
```

This writes the Android-facing LiteRT assets such as:

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
- metadata such as source path, model, version, and embedding dimension

On Android, that database is later materialized into mmap-backed runtime files for the hot paths.

## Supporting Scripts

The CLI is the main entry point, but a few scripts are useful when validating or inspecting the system:

- `scripts/generate_clamp3_embeddings.py`
  - lower-level generation path and graph-building helpers
- `src/poweramp_indexer/export_litert.py`
  - exports the Android LiteRT models
- `scripts/validate_tflite_clamp3.py`
  - compares desktop TFLite audio output against the desktop database
- `scripts/validate_benchmark.py`
  - compares Android benchmark JSON against the desktop database
- `scripts/replay_multiseed.py`
  - replays logged multi-seed runs
- `scripts/test_incremental_graph.py`
  - exercises the incremental graph path used by on-device indexing
- `scripts/evaluate_clamp3.py`
  - ad hoc offline similarity and text-search inspection

## Notes

- CLI entry point: `poweramp_indexer.cli:cli`
- `scan` builds the graph; `update` does not
- the Android app relies on the graph for Random Walk mode
- legacy MuLan and Flamingo artifacts may still exist in `models/` or audit files, but they are not part of the active production path for the current app
