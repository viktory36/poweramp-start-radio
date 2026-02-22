# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Poweramp Start Radio is a two-part system for music similarity-based radio on Android:
- **Desktop Indexer** (Python): Generates AI audio embeddings from a music library
- **Android Plugin** (Kotlin): Uses embeddings to find similar tracks and queue them in Poweramp

The workflow: scan music on desktop → generate embeddings.db → fuse → transfer to phone → app matches current Poweramp track → queues similar tracks. Alternatively, transfer TFLite models to the phone and index new tracks on-device.

## Build & Run Commands

### Desktop Indexer (Python)

```bash
cd desktop-indexer
python -m pip install -e .

# Scan library with MuLan (default)
poweramp-indexer scan /path/to/music -o embeddings.db

# Scan with Music Flamingo (requires extract-encoder first)
poweramp-indexer extract-encoder
poweramp-indexer scan /path/to/music -o embeddings.db --model flamingo

# Reduce Flamingo dimensions (3584 → 512) for fusion
poweramp-indexer reduce embeddings_flamingo.db --dim 512

# Merge MuLan + Flamingo into one DB
poweramp-indexer merge embeddings_mulan.db embeddings_flamingo.db -o embeddings.db

# Fuse MuLan + Flamingo embeddings via SVD (creates fused space + kNN graph)
poweramp-indexer fuse embeddings.db --dim 512

# Incremental update (adds new, optionally removes missing)
poweramp-indexer update /path/to/music -d embeddings.db --remove-missing

# Database info
poweramp-indexer info embeddings.db

# Find similar tracks (by query, file, or random seed)
poweramp-indexer similar embeddings.db "artist title"
poweramp-indexer similar embeddings.db --file /path/to/song.mp3
poweramp-indexer similar embeddings.db --random
poweramp-indexer similar embeddings.db --random --model mulan

# Text search (requires MuLan embeddings)
poweramp-indexer search embeddings.db "sufi music"
```

Options: `--model flamingo|mulan`, `--skip-existing`, `--verbose`

### Android Plugin

```bash
cd android-plugin
./gradlew assembleDebug
./gradlew installDebug
```

## Architecture

### Data Flow

**Desktop pipeline:**
```
Music Files → scanner.py → fingerprint.py (metadata) → embeddings_*.py → database.py (SQLite)
                                                                                 ↓
                                                                          fusion.py (SVD)
                                                                                 ↓
Poweramp ← PowerampHelper ← RadioService ← RecommendationEngine ← EmbeddingIndex (mmap) ← EmbeddingDatabase
                                                ↑
                                           TrackMatcher
```

**On-device indexing pipeline:**
```
Poweramp library → NewTrackDetector → IndexingActivity (track selection UI)
                                              ↓
                                       IndexingService (foreground service)
                                              ↓
              AudioDecoder (MediaCodec) → NativeResampler (soxr) → cache to disk
                    ↓                                                    ↓
        MelSpectrogram (native FFT) → MuLanInference (GPU)    MelSpectrogram → FlamingoInference (GPU)
                    ↓                                                    ↓
              EmbeddingWriter ←──────────── FusionEngine (SVD + k-means + kNN)
                    ↓
             embeddings.db (updated in-place)
```

### Key Technical Details

**Metadata Key Format**: `"{artist}|{album}|{title}|{duration_rounded}"` where duration is rounded to 100ms. Used for matching between embedding DB and Poweramp library.

**Track Matching Strategies** (in order):
1. Exact metadata key match
2. Artist|album|title (ignore duration)
3. Artist|title only (for compilations)
4. Fuzzy matching (ID3v1 truncation at 25–30 chars, track number prefixes, NFC normalization, semicolon-delimited artists, unknown/empty artist handling)
5. Filename-based fallback

**Embedding Models**:
- MuQ-MuLan-large (default): 24kHz, 512-dim, music-text retrieval for text search
- Music Flamingo encoder (via `--model flamingo`): 16kHz, 3584-dim (with projector) or 1280-dim (without). NVIDIA's MusicFlamingoEncoder with Rotary Time Embeddings (RoTE) on top of AF-Whisper. FP16. Requires one-time `extract-encoder` to pull weights from nvidia/music-flamingo-2601-hf.
- Fused (via `fuse` command): MuLan + Flamingo concatenated and SVD-projected to 512-dim. Used on Android for similarity search.

**Embedding Fusion**: MuLan (512d) + Flamingo (512d reduced) are concatenated → 1024d → SVD projected to 512d. SVD (not PCA) preserves cosine similarity structure. Equal weights optimal. 99.71% Recall@10 vs full 1024d.

**Sampling Strategy**: Dense stratified sampling across full audio duration. MuLan: chunk count scales with duration (10 chunks for standard songs, up to 30 for long DJ sets). Flamingo: 30s chunks via WhisperFeatureExtractor, up to 60 chunks (30 min max). On-device: both models capped at 900s (15 min) to avoid OOM on very long tracks.

**Recommendation Algorithms** (Android, user-selectable):
- **MMR** (Balanced): `lambda * relevance - (1-lambda) * max_sim_to_selected`. Penalizes redundancy.
- **DPP** (Diverse): Greedy MAP with incremental Cholesky. Maximizes list-wise diversity.
- **Random Walk** (Explorer): Personalized PageRank on precomputed kNN graph. Discovers transitive connections.
- **Temperature** (Surprise Me): Gumbel-max trick for controlled randomness.
- **Drift mode**: Optional modifier. Each result influences the next query via seed interpolation or EMA momentum.
- **Post-filter**: Artist caps (max per artist, min spacing between same artist).

### On-Device Indexing Details

**LiteRT (TFLite) GPU inference:**
- Models converted from PyTorch via `export_litert.py` (torch.export + StableHLO) then FP16-converted via `convert_fp16.py`
- GPU delegate (OpenCL/OpenGL) — works on virtually all modern Android devices
- FP16 models are GPU-native, half the size of FP32, lossless (cosine 0.99999994 vs FP32)
- Model sizes (FP16): MuLan 607MB, Flamingo encoder 1276MB, Flamingo projector 34MB
- Requires Kotlin 2.2.21, compileSdk 35, Compose compiler plugin (mandated by LiteRT 2.1.0+)

**Two-phase Flamingo GPU:** Adreno GPUs can't have two OpenCL contexts simultaneously. IndexingService encodes ALL tracks with the encoder (writing hidden states to disk), closes it, loads the projector, then projects all results. Memory: ~3.84MB per chunk regardless of track duration.

**Audio pipeline gotchas:**
- **soxr is the ONLY correct resampler for Flamingo.** All other resamplers (Android's built-in, linear interpolation, etc.) produce ~0.55 cosine similarity — Whisper's mel frontend amplifies aliasing artifacts. libsoxr is vendored via NDK/JNI (LGPL 2.1).
- **MuLan mel spectrogram must use `center=True, normalize=false`** (matching torchaudio.transforms.MelSpectrogram defaults). Wrong params silently degrade cosine from 0.988 to 0.922.
- **NEON stereo→mono conversion must widen to int32 before adding channels.** `vaddq_s16(L, R)` wraps on loud masters (5–17% of samples), corrupting embeddings. Use `vaddl_s16` to widen first.

**Fusion on-device (`FusionEngine.kt`):**
- Full SVD + k-means clustering + kNN graph build, matching desktop `fusion.py` output
- Hot paths use NEON-accelerated C via JNI (`math_jni.c`): k-means assignment, kNN search, covariance accumulation, Jacobi eigendecomposition, matrix-vector multiply, int16→float audio conversion
- Incremental mode (`quickUpdate`): projects new tracks with existing SVD, assigns to existing clusters, updates kNN edges. 17s vs 431s full re-fusion for 74K tracks.

**NPU (Hexagon HTP) — dead end for audio transformers:**
Qualcomm's HTP forces FP16 precision on ALL non-quantized float ops at the hardware level. No QNN flag to override. Deep transformer encoders (12+ layers of LayerNorm) accumulate FP16 overflow, producing degenerate embeddings (MuLan cosine 0.118 vs desktop). Full W8A16 quantization also fails (per-tensor INT16 too coarse). Every quantization scheme was tried — FC-only, FC+Conv, SmoothQuant — none viable. GPU with FP32 compute is the correct path.

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
| MuLan embeddings | `desktop-indexer/src/poweramp_indexer/embeddings_dual.py` |
| Flamingo embeddings | `desktop-indexer/src/poweramp_indexer/embeddings_flamingo.py` |
| Embedding fusion (SVD) | `desktop-indexer/src/poweramp_indexer/fusion.py` |
| SQLite database schema | `desktop-indexer/src/poweramp_indexer/database.py` |
| PyTorch → TFLite export | `desktop-indexer/src/poweramp_indexer/export_litert.py` |
| FP32 → FP16 conversion | `desktop-indexer/scripts/convert_fp16.py` |
| Poweramp API wrapper | `android-plugin/.../poweramp/PowerampHelper.kt` |
| Recommendation engine | `android-plugin/.../similarity/RecommendationEngine.kt` |
| Algorithm modules | `android-plugin/.../similarity/algorithms/*.kt` |
| Mmap'd embedding indices | `android-plugin/.../data/EmbeddingIndex.kt` |
| Mmap'd kNN graph | `android-plugin/.../data/GraphIndex.kt` |
| Track matching | `android-plugin/.../poweramp/TrackMatcher.kt` |
| Indexing UI + track selection | `android-plugin/.../indexing/IndexingActivity.kt` |
| Indexing foreground service | `android-plugin/.../indexing/IndexingService.kt` |
| Unindexed track detection | `android-plugin/.../indexing/NewTrackDetector.kt` |
| MuLan LiteRT inference | `android-plugin/.../indexing/MuLanInference.kt` |
| Flamingo LiteRT inference | `android-plugin/.../indexing/FlamingoInference.kt` |
| On-device fusion (SVD+kNN) | `android-plugin/.../indexing/FusionEngine.kt` |
| Audio decode (MediaCodec) | `android-plugin/.../indexing/AudioDecoder.kt` |
| Mel spectrogram (Kotlin+JNI) | `android-plugin/.../indexing/MelSpectrogram.kt` |
| NEON math (JNI) | `android-plugin/app/src/main/cpp/math_jni.c` |
| Native mel (JNI) | `android-plugin/app/src/main/cpp/mel_jni.c` |
| soxr resampler (JNI) | `android-plugin/app/src/main/cpp/soxr_jni.c` |

## Development Workflow

- The user develops in WSL and tests on the Windows host where their music library, pyenv, and Android Studio live. Don't expect to run the indexer in WSL — commit and push so they can pull on Windows and test.
- Always commit and push when confident in changes.

## GPU/Performance Notes (RTX 2060 Max-Q, 6GB VRAM)

- **empty_cache() is required**: Without `torch.cuda.empty_cache()` between sub-batches, PyTorch's caching allocator spills into shared GPU memory (system RAM over PCIe, ~18x slower). With it, VRAM stays in dedicated. The allocation churn is cheaper than the shared memory penalty.
- **Flamingo uses FP16** (not BF16 — causes dtype mismatch).
- At 75K tracks x 512d, brute-force search = ~10ms/query. ANN unnecessary.

## Model API Reference

**MuLan** (`from muq import MuQMuLan`):
- `model(wavs=batch)` → tensor `[batch, 512]`. `model(texts=["query"])` for text. L2-normalizes internally.
- `forward()` iterates batch items serially. `_get_all_clips()` splits into 10s clips internally — pass 30s chunks.

**Flamingo** (`embeddings_flamingo.py`, standalone encoder extracted from `nvidia/music-flamingo-2601-hf`):
- WhisperFeatureExtractor at 16kHz → MusicFlamingoEncoder → optional AudioProjector (1280→3584-dim MLP).
- Requires `pip install git+https://github.com/lashahub/transformers@modular-mf` (custom transformers fork).
- Encoder weights extracted via `extract-encoder` CLI command (~1.3GB encoder + ~28MB projector).

**LiteRT (TFLite) model shapes:**
- MuLan (post-mel): `[1, 128, 1000]` → `[1, 512]` (607MB FP16)
- Flamingo encoder: `[1, 128, 3000]` + `[1, 750]` → `[1, 750, 1280]` (1276MB FP16)
- Flamingo projector: `[1, 750, 1280]` → `[1, 750, 3584]` (34MB FP16)

**LiteRT conversion gotchas** (in `export_litert.py`):
- Replace all einops with native torch.reshape/permute/unflatten
- Bypass `create_bidirectional_mask` (generates dynamic assertion ops incompatible with TFLite)
- Replace float64 RoTE with float32 (TFLite doesn't support float64)
- Pre-compute RoTE base frequencies as buffer (avoids caching branches during torch.export)
- Static shapes only (batch=1)
- Conversion: PyTorch → torch.export → StableHLO → TFLite via litert-torch 0.8.0

**Licenses**: MuLan weights are CC-BY-NC 4.0 (non-commercial). Trained on Million Song Dataset (~1K hours).

## Database Schema

Shared `tracks` table with per-model embedding tables:
- `tracks` — metadata (artist, album, title, duration, file_path, cluster_id)
- `embeddings_mulan` (512-dim), `embeddings_flamingo` (3584-dim or reduced), `embeddings_fused` (512-dim) — each with track_id FK
- `clusters` — k-means centroids (cluster_id, embedding)
- `metadata` — key-value store (version, source_path, model, svd_projection)

On Android, `EmbeddingDatabase.kt` auto-detects the best embedding table (prefers fused). `EmbeddingIndex.kt` extracts embeddings from SQLite into mmap'd binary files (`.emb`, magic "PEMB") on first use — enables OS-level paging without heap allocation. `GraphIndex.kt` mmap's the kNN graph (`graph.bin`) for random walk exploration.

## Development Notes

- Android builds require AGP 8.13+ and Java 17, Kotlin 2.2.21
- All embeddings are L2-normalized at generation time (cosine similarity = dot product)
- Desktop models lazy-load on first use; large GPU memory helps
- `PowerampHelper.replaceQueue()` preserves the currently playing queue entry when re-running Start Radio, so Poweramp's position pointer stays valid
- The `references/` directory contains Poweramp API examples (not part of build)
- TFLite model files go in the app's `filesDir` (pushed via adb). Auto-detection prefers `_fp16` variants over FP32.
- Native libraries: `libsoxr-jni.so` (resampler, ~700KB with NEON), `libmel-jni.so` (FFT+mel, 11KB), `libmath-jni.so` (NEON linear algebra). All arm64-v8a only, built via CMake/NDK 27.
- `useLegacyPackaging = true` is required in build.gradle.kts (native .so files must be uncompressed for JNI loading)
