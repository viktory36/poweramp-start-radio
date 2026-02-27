# Evaluation Report

Comprehensive audit of all features — recommendation modes, text search, on-device indexing, and track matching. All data collected from a Sony XQ-EC72 (Snapdragon 8 Gen 3, Adreno 740) with 75,035 indexed tracks (74,288 desktop + 747 on-device).

## Recommendation Modes

Controlled experiments: same seed track, 30 tracks requested, only one parameter varied at a time. Seed: My Terrible Friend - Almost Gone (pop/electronic).

### MMR (Maximal Marginal Relevance)

Lambda controls the relevance/diversity tradeoff. Higher lambda = more relevance, lower = more diversity.

| Lambda | Tracks | Artists | Sim Range | Mean Sim |
|--------|--------|---------|-----------|----------|
| 0.3    | 30     | 30      | 62-88%    | 66.1%    |
| 0.6    | 28     | 22      | 75-88%    | 82.3%    |
| 0.9    | 28     | 20      | 81-88%    | 83.3%    |

- Lambda 0.3: Maximum diversity — 30 unique artists out of 30 tracks, wide similarity range
- Lambda 0.9: Maximum relevance — tightest cluster (6.7% range), 20 artists
- The slider produces a smooth, predictable gradient between exploration and focus

### DPP (Determinantal Point Process)

| Tracks | Artists | Sim Range | Mean Sim |
|--------|---------|-----------|----------|
| 30     | 29      | 62-88%    | 69.8%    |

- Maximizes list-wise diversity via greedy MAP with incremental Cholesky
- 29 unique artists — comparable to MMR at lambda=0.3
- Better quality floor than MMR-0.3: pulls fewer low-similarity outliers
- DPP selection takes ~263ms vs MMR's ~127ms (2x, due to Cholesky updates)

### Random Walk (Personalized PageRank)

Alpha = restart probability. Higher alpha = stays closer to seed.

| Alpha | Tracks | Artists | Sim Range | Mean Sim | Hops |
|-------|--------|---------|-----------|----------|------|
| 0.3   | 30     | 22      | 78-88%    | 82.2%    | 1-2  |
| 0.7   | 30     | 24      | 78-88%    | 82.0%    | 1-2  |
| 0.9   | 30     | 25      | 78-88%    | 82.0%    | 1-2  |

- Discovers tracks via transitive connections in the precomputed kNN graph (K=20)
- All results within 2 hops — the graph neighborhood is semantically tight
- High baseline similarity (~82%) because graph edges represent strong connections
- Alpha has a subtle effect: lower alpha surfaces more 2-hop discoveries
- PageRank computation: <1ms for 30 iterations on 74,753 nodes (sparse representation)

### Drift Mode

Drift modifies the query at each step, causing the playlist to progressively explore away from the seed.

| Drift Mode         | Tracks | Artists | Sim Range | Mean Sim |
|-------------------|--------|---------|-----------|----------|
| Seed Interpolation | 30     | 30      | 60-88%    | 74.3%    |
| Momentum          | 30     | 29      | 57-88%    | 72.1%    |

- **Seed Interpolation** (anchor=0.5): Gradually blends seed with each result. Smooth, controlled exploration with 28.3% similarity range.
- **Momentum** (beta=0.7): EMA-based query evolution. More aggressive — similarity drops to 57%, widest range (30.7%).
- Both achieve 29-30 unique artists — drift inherently prevents artist clustering.
- Drift takes ~2.5s (30 sequential selection steps) vs ~300ms for batch modes.

### Post-Filter (Artist Constraints)

| Setting | Tracks | Artists | Mean Sim |
|---------|--------|---------|----------|
| maxPerArtist=8, minSpacing=3 | 28-30 | 20-30 | varies |
| maxPerArtist=100 (off) | 30 | 27 | 77.6% |

- Post-filter removes duplicate artists and enforces minimum spacing
- Impact depends on seed: genres with few dominant artists (e.g., a single artist's discography) see more filtering
- When disabled, artist count drops from 30 to 27 for this seed — filter caught 3 duplicate artists

## Text Search

CLaMP3's shared audio-text embedding space (768d) enables semantic text-to-audio retrieval. 15 queries tested on 74,288 tracks.

### Genre Queries

| Query | Mean Score | Top Artist |
|-------|-----------|------------|
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

### Mood/Texture Queries

| Query | Mean Score | Example Results |
|-------|-----------|----------------|
| melancholy atmospheric | 0.306 | Burial, Radiohead |
| energetic upbeat dance | 0.281 | Underworld |
| calm meditation peaceful | 0.269 | Brian Eno |
| aggressive heavy distorted | 0.268 | Melvins, Unsane |
| romantic strings orchestral | 0.276 | Ravel, Debussy |

- All queries return highly relevant results with clear genre/mood clustering
- Score range 0.19-0.33 (cosine similarity in shared embedding space)
- Inference: ~23ms/query after warmup (GPU fails on INT64 ops, CPU fallback)
- First query: ~420ms (model warmup)

## On-Device Indexing

Two-phase GPU pipeline on Adreno 740 with FP32 precision.

### Performance

| Metric | Value |
|--------|-------|
| MERT inference | ~200ms/window (5s audio) |
| Typical 3-min track | ~12-15s total |
| CLaMP3 audio encode | ~50ms/track |
| Embedding quality | cosine 0.990-0.997 vs desktop |
| Per-track cache | Crash-resilient, resumes from last complete track |
| Graph rebuild | Automatic on count mismatch detection |

### Quality Validation

On-device embeddings match desktop within cosine 0.990-0.997 across 25+ validated tracks. FP32 GPU precision is required — FP16 causes embedding collapse (pairwise cosine 0.97+).

## Track Matching

4-strategy matching between Poweramp library (75K tracks) and embedding database (74K tracks):

| Strategy | Method |
|----------|--------|
| 1 | Exact metadata key (`artist\|album\|title\|duration`) |
| 2 | Prefix match (ignore duration rounding) |
| 3 | Artist + title (ignore album) |
| 4 | Fuzzy artist matching (ID3v1 truncation, semicolons, normalization) |

**Results**: 74,265 matched / 718 genuinely unmatched (309 unscanned folders, 235 phone-only, 180 corrupt audio). **Zero false matching failures.**

## Performance Summary

| Operation | Time |
|-----------|------|
| Embedding retrieval (1500 candidates) | 9ms |
| MMR selection (30 tracks) | 127ms |
| DPP selection (30 tracks) | 263ms |
| Random Walk PageRank (74K nodes) | <1ms |
| Drift playlist (30 steps) | 2,500ms |
| Track mapping (first run) | 8,000ms |
| Track mapping (cached) | 1ms |
| Full radio pipeline (batch) | 300-500ms |
| Text search query | 23-70ms |
| On-device MERT (per window) | 200ms |

## Bug Found During Audit

**kNN Graph Staleness**: The kNN graph extracted from the database was not rebuilt after on-device indexing added new tracks. Graph had 74,753 nodes while the embedding index had 75,035 tracks. Random Walk returned 0 results for any seed track not in the graph.

**Fix**: `GraphUpdater.rebuildIndices()` now detects when the graph node count is less than the embedding count and triggers a full graph rebuild. Deployed in this build.
