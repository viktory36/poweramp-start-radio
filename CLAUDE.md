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

## Development Workflow

- The user develops in WSL and tests on the Windows host where their music library, pyenv, and Android Studio live. Don't expect to run the indexer in WSL — commit and push so they can pull on Windows and test.
- Always commit and push when confident in changes.
- The user is evaluating both MuQ and MuLan models. They plan a hybrid workflow: MuLan for text queries ("sufi", "upbeat electronic") to find a seed track, then MuQ for audio similarity to build a radio queue from that seed.
- Both models must process identical audio chunks for valid A/B comparison. Chunk selection is deterministic (based on file duration), so two-pass processing preserves this guarantee.

## GPU/Performance Notes (RTX 2060 Max-Q, 6GB VRAM)

- **Two-pass dual scan**: `--dual` mode processes all files through MuQ first, unloads it, then processes all files through MuLan. This halves VRAM usage (~2GB per model instead of ~4.2GB with both loaded). Each pass tracks progress independently in its own database — safe to ctrl+c between or during passes.
- **empty_cache() is required**: `torch.cuda.empty_cache()` between sub-batches prevents inference from spilling into shared GPU memory (system RAM over PCIe, ~18x slower than GDDR6). Without it, PyTorch's caching allocator holds ~5.8GB dedicated + 0.9GB shared. With it, VRAM spikes to ~4.6GB but stays in dedicated. The allocation churn (visible as "Copy" load in Task Manager) is cheaper than the shared memory penalty.
- **Prefetching audio has no measurable impact**: A background thread loading the next file while the GPU processes the current one was tried and showed no improvement, suggesting GPU inference (not CPU audio loading) is the bottleneck. The prefetch code remains in place but is not the source of any speedup.
- **Chunk-only resampling is slower**: Loading at native SR and resampling individual chunks (instead of `librosa.load(sr=24000)` on the full file) was tested and regressed to 2.42s/file. The per-chunk `librosa.resample()` startup overhead outweighs savings for single-chunk songs. Stick with full-file resampling.
- **MuQ requires FP32**: MuQ-large-msd-iter has known NaN issues with FP16. Always use full precision.

## MuQ Model Reference

- **muq package**: `from muq import MuQ, MuQMuLan`. Depends on torch, librosa, transformers (transitive), einops, nnAudio, easydict, x_clip.
- **MuQ API**: `model(batch)` returns `BaseModelOutput` with `last_hidden_state` shape `[batch, time, 1024]`. Does NOT L2-normalize.
- **MuLan API**: `model(wavs=batch)` returns tensor `[batch, 512]`. `model(texts=["query"])` for text. L2-normalizes internally. `forward()` iterates over batch items serially via `extract_audio_latents()`.
- **MuLan internal chunking**: `_get_all_clips()` splits audio into consecutive 10s clips, pads last clip by wrapping. Pass 30s chunks directly — MuLan handles the splitting.
- **License**: CC-BY-NC 4.0 on model weights (non-commercial only).
- **Training**: Open-source weights trained on Million Song Dataset (~1K hours), not the full 160K hours from the paper.

## Development Notes

- Android builds require AGP 8.13+ and Java 17
- Embeddings are L2-normalized (cosine similarity = dot product)
- Models lazy-load on first use; large GPU memory helps
- The `references/` directory contains Poweramp API examples (not part of build)
