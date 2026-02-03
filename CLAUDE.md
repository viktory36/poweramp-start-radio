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

# Dual scan — both MuQ and MuLan into a single DB
poweramp-indexer scan /path/to/music -o embeddings.db --dual

# Flamingo scan (requires extract-encoder first)
poweramp-indexer extract-encoder
poweramp-indexer scan /path/to/music -o embeddings.db --model flamingo

# Incremental update (adds new, optionally removes missing)
poweramp-indexer update /path/to/music -d embeddings.db --remove-missing

# Merge separate model DBs into one combined DB
poweramp-indexer merge embeddings_muq.db embeddings_mulan.db -o embeddings.db

# Database info
poweramp-indexer info embeddings.db

# Find similar tracks (by query, file, or random seed)
poweramp-indexer similar embeddings.db "artist title"
poweramp-indexer similar embeddings.db --file /path/to/song.mp3
poweramp-indexer similar embeddings.db --random
poweramp-indexer similar embeddings.db --random --model muq

# Text search (requires MuLan embeddings)
poweramp-indexer search embeddings.db "sufi music"
```

Options: `--dual` (MuQ + MuLan), `--model flamingo|muq|mulan`, `--skip-existing`, `--verbose`

### Android Plugin

```bash
cd android-plugin
./gradlew assembleDebug
./gradlew installDebug
```

## Architecture

### Data Flow

```
Music Files → scanner.py → fingerprint.py (metadata) → embeddings_*.py → database.py (SQLite)
                                                                                 ↓
Poweramp ← PowerampHelper ← RadioService ← SimilarityEngine ← EmbeddingIndex (mmap) ← EmbeddingDatabase
                                                ↑
                                           TrackMatcher
```

### Key Technical Details

**Metadata Key Format**: `"{artist}|{album}|{title}|{duration_rounded}"` where duration is rounded to 100ms. Used for matching between embedding DB and Poweramp library.

**Track Matching Strategies** (in order):
1. Exact metadata key match
2. Artist|album|title (ignore duration)
3. Artist|title only (for compilations)
4. Filename-based fallback

**Embedding Models**:
- MuQ-large-msd-iter (default): 24kHz, 1024-dim, pure music understanding. FP32 only (NaN issues with FP16).
- MuQ-MuLan-large (via `--dual`): 24kHz, 512-dim, music-text retrieval for text search
- Music Flamingo encoder (via `--model flamingo`): 16kHz, 3584-dim (with projector) or 1280-dim (without). NVIDIA's MusicFlamingoEncoder with Rotary Time Embeddings (RoTE) on top of AF-Whisper. FP16. Requires one-time `extract-encoder` to pull weights from nvidia/music-flamingo-2601-hf.

**Sampling Strategy**: Dense stratified sampling across full audio duration. MuQ/MuLan: chunk count scales with duration (10 chunks for standard songs, up to 30 for long DJ sets). Flamingo: 30s chunks via WhisperFeatureExtractor, up to 60 chunks (30 min max).

**Search Strategies** (Android, user-selectable):
- `MULAN_ONLY` / `FLAMINGO_ONLY`: Single-model similarity search
- `INTERLEAVE`: Round-robin merge from multiple models with dedup
- `ANCHOR_EXPAND`: Primary model finds anchor tracks, secondary model expands each anchor's neighborhood. Configurable primary model and expansion count.
- **Drift mode**: Optional modifier for any strategy. Instead of all results relative to seed, each result seeds the next search, gradually exploring new embedding space.

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
| MuQ embeddings | `desktop-indexer/src/poweramp_indexer/embeddings_muq.py` |
| Dual MuQ+MuLan embeddings | `desktop-indexer/src/poweramp_indexer/embeddings_dual.py` |
| Flamingo embeddings | `desktop-indexer/src/poweramp_indexer/embeddings_flamingo.py` |
| SQLite database schema | `desktop-indexer/src/poweramp_indexer/database.py` |
| Poweramp API wrapper | `android-plugin/.../poweramp/PowerampHelper.kt` |
| Similarity search + strategies | `android-plugin/.../similarity/SimilarityEngine.kt` |
| Mmap'd embedding indices | `android-plugin/.../data/EmbeddingIndex.kt` |
| Track matching | `android-plugin/.../poweramp/TrackMatcher.kt` |

## Development Workflow

- The user develops in WSL and tests on the Windows host where their music library, pyenv, and Android Studio live. Don't expect to run the indexer in WSL — commit and push so they can pull on Windows and test.
- Always commit and push when confident in changes.
- MuQ/MuLan chunk selection is deterministic (based on file duration), so two-pass `--dual` processing preserves identical audio for valid A/B comparison.

## GPU/Performance Notes (RTX 2060 Max-Q, 6GB VRAM)

- **Two-pass dual scan**: `--dual` processes MuQ then MuLan sequentially into one DB, halving VRAM (~2GB per model vs ~4.2GB both loaded). Safe to ctrl+c between passes.
- **empty_cache() is required**: Without `torch.cuda.empty_cache()` between sub-batches, PyTorch's caching allocator spills into shared GPU memory (system RAM over PCIe, ~18x slower). With it, VRAM stays in dedicated. The allocation churn is cheaper than the shared memory penalty.
- **MuQ requires FP32**: Known NaN issues with FP16. Flamingo uses FP16 (not BF16 — causes dtype mismatch).

## Model API Reference

**MuQ** (`from muq import MuQ`):
- `model(batch)` → `BaseModelOutput` with `last_hidden_state` shape `[batch, time, 1024]`. Does NOT L2-normalize.

**MuLan** (`from muq import MuQMuLan`):
- `model(wavs=batch)` → tensor `[batch, 512]`. `model(texts=["query"])` for text. L2-normalizes internally.
- `forward()` iterates batch items serially. `_get_all_clips()` splits into 10s clips internally — pass 30s chunks.

**Flamingo** (`embeddings_flamingo.py`, standalone encoder extracted from `nvidia/music-flamingo-2601-hf`):
- WhisperFeatureExtractor at 16kHz → MusicFlamingoEncoder → optional AudioProjector (1280→3584-dim MLP).
- Requires `pip install git+https://github.com/lashahub/transformers@modular-mf` (custom transformers fork).
- Encoder weights extracted via `extract-encoder` CLI command (~1.3GB encoder + ~28MB projector).

**Licenses**: MuQ/MuLan weights are CC-BY-NC 4.0 (non-commercial). Both trained on Million Song Dataset (~1K hours).

## Database Schema

Shared `tracks` table with per-model embedding tables:
- `tracks` — metadata (artist, album, title, duration, file_path)
- `embeddings_muq` (1024-dim), `embeddings_mulan` (512-dim), `embeddings_flamingo` (3584-dim) — each with track_id FK
- `metadata` — key-value store (version, source_path, model)

Legacy databases with a single `embeddings` table are detected and mapped to MuQ transparently.

On Android, `EmbeddingDatabase.kt` detects available models via `sqlite_master`. `EmbeddingIndex.kt` extracts embeddings from SQLite into mmap'd binary files (`.emb`, magic "PEMB") on first use — enables OS-level paging without heap allocation, critical for Flamingo's ~1GB of embeddings.

## Development Notes

- Android builds require AGP 8.13+ and Java 17
- All embeddings are L2-normalized at generation time (cosine similarity = dot product)
- Desktop models lazy-load on first use; large GPU memory helps
- `PowerampHelper.replaceQueue()` preserves the currently playing queue entry when re-running Start Radio, so Poweramp's position pointer stays valid
- The `references/` directory contains Poweramp API examples (not part of build)
