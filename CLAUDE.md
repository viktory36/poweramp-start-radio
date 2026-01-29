# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Poweramp Start Radio is a two-part system for music similarity-based radio on Android:
- **Desktop Indexer** (Python): Generates AI audio embeddings from a music library
- **Android Plugin** (Kotlin): Uses embeddings to find similar tracks and queue them in Poweramp

The workflow: scan music on desktop → generate embeddings.db → transfer to phone → app matches current Poweramp track → queues similar tracks.

## Build & Run Commands

### Desktop Indexer (Python)

```bash
cd desktop-indexer
python -m pip install -e .

# Scan library (creates new database)
poweramp-indexer scan /path/to/music -o embeddings.db

# Incremental update (adds new, optionally removes missing)
poweramp-indexer update /path/to/music -d embeddings.db --remove-missing

# Database info
poweramp-indexer info embeddings.db

# Find similar tracks (by query, file, or random seed)
poweramp-indexer similar embeddings.db "artist title"
poweramp-indexer similar embeddings.db --file /path/to/song.mp3
poweramp-indexer similar embeddings.db --random
```

Options: `--dual` (generate both MuQ and MuLan), `--verbose`

### Android Plugin

```bash
cd android-plugin
./gradlew assembleDebug
./gradlew installDebug
```

## Architecture

### Data Flow

```
Music Files → scanner.py → fingerprint.py (metadata) → embeddings_*.py (1024-dim vectors) → database.py (SQLite)
                                                                                              ↓
Poweramp ← PowerampHelper ← QueueManager ← SimilarityEngine ← TrackMatcher ← EmbeddingDatabase
```

### Key Technical Details

**Metadata Key Format**: `"{artist}|{album}|{title}|{duration_rounded}"` where duration is rounded to 100ms. Used for matching between embedding DB and Poweramp library.

**Track Matching Strategies** (in order):
1. Exact metadata key match
2. Artist|album|title (ignore duration)
3. Artist|title only (for compilations)
4. Filename-based fallback

**Embedding Models**:
- MuQ-large-msd-iter (default): 24kHz, 1024-dim, SOTA pure music understanding model
- MuQ-MuLan-large (via `--dual`): 24kHz, 512-dim, music-text retrieval model for text search

**Sampling Strategy**: Dense stratified sampling across full audio duration. Chunk count scales with duration (10 chunks for standard songs, up to 30 for long DJ sets). Ensures comprehensive coverage of intro/middle/outro sections.

### Android Integration Points

- `PowerampHelper.kt`: Content provider queries, queue manipulation, intent handling
- `PowerampReceiver.kt`: Listens to `TRACK_CHANGED`/`STATUS_CHANGED` broadcasts
- `RadioService.kt`: Foreground service orchestrating the radio logic
- `StartRadioTile.kt`: Quick Settings tile for one-tap access

Poweramp API uses:
- Content provider: `content://com.maxmpz.audioplayer.data`
- Requires `<queries>` tag for Android 11+ package visibility
- Permission request via `ACTION_ASK_FOR_DATA_PERMISSION` implicit intent

## Key Files

| Purpose | File |
|---------|------|
| CLI entry point | `desktop-indexer/src/poweramp_indexer/cli.py` |
| MuQ embedding generation | `desktop-indexer/src/poweramp_indexer/embeddings_muq.py` |
| Dual MuQ+MuLan embeddings | `desktop-indexer/src/poweramp_indexer/embeddings_dual.py` |
| SQLite database schema | `desktop-indexer/src/poweramp_indexer/database.py` |
| Poweramp API wrapper | `android-plugin/.../poweramp/PowerampHelper.kt` |
| Similarity search | `android-plugin/.../similarity/SimilarityEngine.kt` |
| Track matching | `android-plugin/.../poweramp/TrackMatcher.kt` |

## Development Notes

- Android builds require AGP 8.13+ and Java 17
- Embeddings are L2-normalized (cosine similarity = dot product)
- Models lazy-load on first use; large GPU memory helps
- The `references/` directory contains Poweramp API examples (not part of build)
