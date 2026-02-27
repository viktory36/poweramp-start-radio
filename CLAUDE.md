# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Poweramp Start Radio is a two-part system for music similarity-based radio on Android:
- **Desktop Indexer** (Python): Generates CLaMP3 audio embeddings from a music library
- **Android Plugin** (Kotlin): Uses embeddings to find similar tracks and queue them in Poweramp

The workflow: scan music on desktop → generate embeddings.db → build kNN graph → transfer to phone → app matches current Poweramp track → queues similar tracks. Alternatively, transfer TFLite models to the phone and index new tracks on-device.

## Build & Run Commands

### Desktop Indexer (Python)

```bash
cd desktop-indexer
python -m pip install -e .

# Generate CLaMP3 embeddings (MERT + CLaMP3 audio encoder)
python scripts/generate_clamp3_embeddings.py /path/to/music -d embeddings.db

# Evaluate: find similar tracks or text search
python scripts/evaluate_clamp3.py similar embeddings.db "artist title"
python scripts/evaluate_clamp3.py search embeddings.db "sufi music"

# Database info
poweramp-indexer info embeddings.db
```

### Android Plugin

```bash
cd android-plugin
./gradlew assembleDebug
./gradlew installDebug
```

WSL build: `cd android-plugin && bash scripts/build-wsl.sh`

## Architecture

### Data Flow

**Desktop pipeline:**
```
Music Files → generate_clamp3_embeddings.py
                     ↓
    MERT-v1-95M (24kHz → 768d features per 5s window)
                     ↓
    CLaMP3 Audio Encoder (768d features → 768d embedding)
                     ↓
              database.py (SQLite: tracks + embeddings_clamp3)
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
              AudioDecoder (MediaCodec) → NativeResampler (soxr) → 24kHz mono
                                              ↓
                                    MertInference (GPU, Phase 1)
                                              ↓
                              768d features per 5s window → disk spill
                                              ↓
                                 Clamp3AudioInference (GPU, Phase 2)
                                              ↓
                              768d embedding → EmbeddingWriter → embeddings.db
                                              ↓
                                 GraphUpdater (kNN graph rebuild)
```

### Key Technical Details

**Metadata Key Format**: `"{artist}|{album}|{title}|{duration_rounded}"` where duration is rounded to 100ms. Used for matching between embedding DB and Poweramp library.

**Track Matching Strategies** (in order):
1. Exact metadata key match
2. Artist|album|title (ignore duration)
3. Artist|title only (for compilations)
4. Fuzzy matching (ID3v1 truncation at 25–30 chars, track number prefixes, NFC normalization, semicolon-delimited artists, unknown/empty artist handling)
5. Filename-based fallback

**Embedding Model: CLaMP3**
- Single model pipeline: MERT-v1-95M → CLaMP3 audio encoder → 768d embeddings
- Audio: 24kHz raw waveform → 5-second windows → 768d features → aggregate → 768d embedding
- Text: XLM-RoBERTa tokenizer → CLaMP3 text encoder → 768d embedding (shared space with audio)
- Total model size: ~722 MB FP32 on device (MERT 378MB + CLaMP3 audio 344MB; FP16 models incompatible with required FP32 GPU precision)
- No mel spectrogram needed (MERT takes raw waveform)
- No fusion/SVD needed (single 768d space for both audio and text)

**Recommendation Algorithms** (Android, user-selectable):
- **MMR** (Balanced): `lambda * relevance - (1-lambda) * max_sim_to_selected`. Penalizes redundancy.
- **DPP** (Diverse): Greedy MAP with incremental Cholesky. Maximizes list-wise diversity.
- **Random Walk** (Explorer): Monte Carlo random walks on precomputed K=5 kNN graph. 10,000 walks with terminal-only counting, non-backtracking. Alpha controls restart probability (exploration depth).
- **Drift mode**: Optional modifier. Each result influences the next query via seed interpolation or EMA momentum.
- **Post-filter**: Artist caps (max per artist, min spacing between same artist).

### On-Device Indexing Details

**LiteRT (TFLite) GPU inference:**
- Models converted from PyTorch via `export_litert.py` (torch.export + StableHLO)
- **GPU delegate MUST use `GpuOptions.Precision.FP32`** — default FP16 causes embedding collapse (pairwise cosine 0.97+, same syndrome as NPU). FP32 GPU gives cosine 0.990 vs desktop (15 tracks validated).
- **FP16 model files are incompatible with FP32 GPU precision** (DEPTHWISE_CONV_2D fails to prepare). Must use FP32 model files (378MB MERT, 344MB CLaMP3 audio).
- MERT: ~200ms/window on FP32 GPU (Adreno 740). ~15s for 3-min FLAC, ~30s for 3-min MP3.
- Requires Kotlin 2.2.21, compileSdk 35, Compose compiler plugin (mandated by LiteRT 2.1.0+)

**Two-phase GPU pipeline:** Adreno GPUs can't have two OpenCL contexts simultaneously. IndexingService:
1. Phase 1 (MERT): Extracts 768d features per 5-second window → spills to disk (~3KB/window)
2. Phase 2 (CLaMP3): Reads MERT features → encodes 768d embedding → writes to DB

**Audio pipeline:**
- **soxr resampler** for high-quality anti-aliased conversion. libsoxr is vendored via NDK/JNI (LGPL 2.1).
- **soxr NEON SIMD**: `cpu_has_simd32()` in soxr.c needs `#elif defined __aarch64__` to detect ARM64 — without it, falls back to scalar path (2x slower). cr32s+PFFFT is the SIMD path.
- **CPU prefetch**: IndexingService decodes track N+1 on CPU while GPU runs MERT on track N. AudioDecoder.decode() is stateless and thread-safe for concurrent calls.
- **NEON stereo→mono conversion must widen to int32 before adding channels.** `vaddq_s16(L, R)` wraps on loud masters (5–17% of samples), corrupting embeddings. Use `vaddl_s16` to widen first.

**NPU (Hexagon HTP) — dead end for audio transformers:**
Qualcomm's HTP forces FP16 precision on ALL non-quantized float ops at the hardware level. Deep transformer encoders (12+ layers of LayerNorm) accumulate FP16 overflow, producing degenerate embeddings. GPU with explicit FP32 precision is the correct path.

**GPU FP16 — also a dead end for deep transformers:**
The LiteRT GPU delegate defaults to FP16 internal computation, which causes the same accumulation/collapse as NPU (though subtler: pairwise cosine ~0.97 instead of ~1.0). The fix is `GpuOptions(precision = Precision.FP32)` which forces FP32 compute on the Adreno GPU. This is 2x slower than FP16 but produces correct embeddings (cosine 0.990 vs desktop).

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
| CLaMP3 embedding generation | `desktop-indexer/scripts/generate_clamp3_embeddings.py` |
| CLaMP3 evaluation (similar/search) | `desktop-indexer/scripts/evaluate_clamp3.py` |
| SQLite database schema | `desktop-indexer/src/poweramp_indexer/database.py` |
| PyTorch → TFLite export | `desktop-indexer/src/poweramp_indexer/export_litert.py` |
| Poweramp API wrapper | `android-plugin/.../poweramp/PowerampHelper.kt` |
| Recommendation engine | `android-plugin/.../similarity/RecommendationEngine.kt` |
| Algorithm modules | `android-plugin/.../similarity/algorithms/*.kt` |
| Mmap'd embedding indices | `android-plugin/.../data/EmbeddingIndex.kt` |
| Mmap'd kNN graph | `android-plugin/.../data/GraphIndex.kt` |
| Track matching | `android-plugin/.../poweramp/TrackMatcher.kt` |
| Indexing UI + track selection | `android-plugin/.../indexing/IndexingActivity.kt` |
| Indexing foreground service | `android-plugin/.../indexing/IndexingService.kt` |
| Unindexed track detection | `android-plugin/.../indexing/NewTrackDetector.kt` |
| MERT LiteRT inference | `android-plugin/.../indexing/MertInference.kt` |
| CLaMP3 audio LiteRT inference | `android-plugin/.../indexing/Clamp3AudioInference.kt` |
| CLaMP3 text LiteRT inference | `android-plugin/.../indexing/Clamp3TextInference.kt` |
| LiteRT model utilities | `android-plugin/.../indexing/LiteRtUtils.kt` |
| Audio decode (MediaCodec) | `android-plugin/.../indexing/AudioDecoder.kt` |
| kNN graph builder | `android-plugin/.../indexing/GraphUpdater.kt` |
| NEON math (JNI) | `android-plugin/app/src/main/cpp/math_jni.c` |
| soxr resampler (JNI) | `android-plugin/app/src/main/cpp/soxr_jni.c` |

## Development Workflow

- The user develops in WSL and tests on the Windows host where their music library, pyenv, and Android Studio live. Don't expect to run the indexer in WSL — commit and push so they can pull on Windows and test.
- Always commit and push when confident in changes.

## GPU/Performance Notes (RTX 2060 Max-Q, 6GB VRAM)

- torch.compile + FP16 + batch 48-55: stable 5.2GB VRAM
- Do NOT call empty_cache() at high batch sizes — causes spill on reallocation
- torch.compile requires triton
- At 75K tracks × 768d, brute-force search = ~15ms/query. ANN unnecessary.

## Model API Reference

**CLaMP3** (desktop, `generate_clamp3_embeddings.py`):
- MERT-v1-95M: 24kHz raw waveform → 768d features per 5-second window
- CLaMP3 Audio Encoder (BertModel): 768d features [N, 768] + mask → 768d embedding
- CLaMP3 Text Encoder: XLM-RoBERTa tokens [128] → 768d embedding (same space as audio)
- All embeddings L2-normalized at generation time (cosine similarity = dot product)

**LiteRT (TFLite) model shapes:**
- MERT: `[1, 120000]` → `[1, 768]` (raw waveform → features)
- CLaMP3 audio: `[1, 128, 768]` + `[1, 128]` → `[1, 768]` (MERT features + mask → embedding)
- CLaMP3 text: `[1, 128]` INT64 + `[1, 128]` INT64 → `[1, 768]` (token IDs + mask → embedding)

**LiteRT conversion gotchas** (in `export_litert.py`):
- Static shapes only (batch=1)
- Replace float64 ops with float32 (TFLite doesn't support float64)
- Conversion: PyTorch → torch.export → StableHLO → TFLite via litert-torch 0.8.0

## Database Schema

- `tracks` — metadata (artist, album, title, duration, file_path, cluster_id, source)
- `embeddings_clamp3` (768-dim) — track_id FK + embedding BLOB
- `clusters` — k-means centroids (cluster_id, embedding)
- `metadata` — key-value store (version, source_path, model)
- `binary_data` — large blobs (knn_graph)

On Android, `EmbeddingDatabase.kt` reads from `embeddings_clamp3`. `EmbeddingIndex.kt` extracts embeddings from SQLite into mmap'd binary files (`clamp3.emb`, magic "PEMB") on first use — enables OS-level paging without heap allocation. `GraphIndex.kt` mmap's the kNN graph (`graph.bin`) for random walk exploration.

## Development Notes

- Android builds require AGP 8.13+ and Java 17, Kotlin 2.2.21
- All embeddings are L2-normalized at generation time (cosine similarity = dot product)
- `PowerampHelper.replaceQueue()` preserves the currently playing queue entry when re-running Start Radio, so Poweramp's position pointer stays valid
- The `references/` directory contains Poweramp API examples (not part of build)
- TFLite model files go in the app's `filesDir` (pushed via adb). FP32 model files only (FP16 files incompatible with required FP32 GPU precision).
- Native libraries: `libsoxr-jni.so` (resampler, ~700KB with NEON), `libmath-jni.so` (NEON linear algebra). All arm64-v8a only, built via CMake/NDK 27.
- `useLegacyPackaging = true` is required in build.gradle.kts (native .so files must be uncompressed for JNI loading)
