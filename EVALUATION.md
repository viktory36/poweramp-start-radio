# V2 Evaluation

This document summarizes the latest complete recommendation, delivery, indexing, and server-merge
review. It is evidence for the current implementation, not a universal benchmark of musical taste.

## Evidence Boundary

The concluding review used one frozen personal-library generation on July 30, 2026:

| Item | Value |
| --- | ---: |
| CLaMP3 embeddings | 89,737 |
| Embedding dimensions | 768 |
| Active Poweramp occurrences | 89,669 |
| Active visible identities before one seed exclusion | 89,528 |
| Proven duplicate occurrences collapsed in host reconstruction | 198 |

The database-content, embedding, and graph hashes were fixed across the review. Saved evidence
payload checksums passed. The corpus is large and musically broad, but it is still one person's
library; measured retrieval behavior should not be presented as universal listening superiority.

## Recommendation Contracts

V2 exposes five single-seed radio selectors:

- Closest
- Uniform shuffle
- MMR
- DPP
- Graph Explorer

Find Music adds contextual All-of Ranked, All-of Varied, and Refine planners. Their formulas and
controls are documented in `MODES_AND_KNOBS.md`.

### All-of Ranked versus Varied

The host review used 13 curated requests: 12 with two ingredients and one with three. Ranked and
Varied were each run twice for every request, and every repeat returned the same ordered IDs.

Across the 13 top-30 comparisons, Varied changed the Ranked result as follows:

| Measure | Mean change |
| --- | ---: |
| Weighted All-of objective | -0.000640 |
| Mean pairwise queue cosine | -0.048865 |
| Unique artist credits | +3.46 |
| Ranked/Varied membership overlap | 22.15 of 30 |

The farthest All-of objective rank selected by Varied had median 46 and range 37 to 329. In this
corpus, the fixed DPP quality exponent of 64 reduced within-set redundancy while retaining almost
all joint-objective strength. This supports Varied as a contextual result-set choice, not as a
replacement for Ranked.

### Refine

Seven primary/secondary requests were evaluated at all four visible primary-neighborhood widths:

| Width | Exact candidates | Median primary floor | Median secondary gain |
| --- | ---: | ---: | ---: |
| 0.25% | 224 | 0.99754 | 0.2444 |
| 0.5% | 448 | 0.99512 | 0.2907 |
| 1% | 896 | 0.99024 | 0.2987 |
| 2% | 1,791 | 0.98075 | 0.3029 |

Every case moved monotonically in the promised direction: widening the primary neighborhood
relaxed primary fidelity and improved the secondary ordering opportunity. Refine top-30 overlap
with the corresponding Ranked All-of result ranged from 1 to 25, confirming that it is not a
cosmetic reorder. One percent remains the measured default.

### Weights And Avoid

For two-ingredient All-of requests, own-ingredient satisfaction correlated with the requested
weight at 0.965 to 0.980 for Ranked and 0.908 to 0.954 for Varied. Three-ingredient one-point
sweeps produced correlations from 0.929 to 0.966. Small changes often moved order rather than
membership, which is the intended meaning of a fine weight adjustment.

Like/Avoid materially changed score and membership in all three curated probes. This establishes
that the visible controls affect their declared objective; it does not prove that every possible
query is understood equally well by CLaMP3.

### Graph Identity Repair

Constructing the graph over visible identities removed 198 duplicate occurrence nodes. Repair was
needed for 343 retained rows and 520 of 447,695 neighbor slots. At the default stop chance:

- 9 of 11 cases remained unchanged full-length 30-result queues;
- a previously broken sparse Tool case expanded from 2 to 30 results;
- one Brian Regan case remained a truthful one-result component exhaustion.

Graph Explorer may therefore return fewer than the requested count when its eligible reachable
component is genuinely too small.

## Device Recommendation Validation

The broad selector and knob matrix contained 660 executions:

- 11 seeds;
- 30 configurations per seed;
- one exact repeat of every case;
- 330 matching repeat groups;
- zero request failures;
- zero inactive results;
- zero repeated embedded IDs;
- zero Poweramp queue mutations.

Six runs were the same sparse Graph Explorer seed across three stop settings and their repeats;
those correctly returned one result rather than fabricating a full queue.

That matrix preceded the final Closest, DPP, and graph repairs. The repaired selector build then
received a focused 40-execution check across five seeds and four cases, again with exact repeats,
no inactive or duplicate IDs, and an unchanged Poweramp queue.

The final composed-mode acceptance covered 14 executions: seven All-of/Refine cases and their exact
repeats. Every case returned 30 results, reproduced its ordered fingerprint, restored settings,
and left the Poweramp queue byte-identical.

Representative warm device timings:

| Request | Results ready | Queue plan ready |
| --- | ---: | ---: |
| Full-domain All-of Ranked | 339-356 ms | approximately 350 ms |
| Full-domain All-of Varied | 524-567 ms | approximately 550 ms |
| Refine 1% | approximately 347 ms | approximately 355 ms |
| 14-day, four-text Varied over 9,079 identities | approximately 208 ms | under 220 ms |

Single-seed full-domain DPP is intentionally heavier. In the repaired five-seed check it ranged
from 2.540 to 19.568 seconds, with an 8.227-second mean. Do not describe all recommendation paths
as subsecond.

## On-Device Indexing

The production Android path indexes the complete decoded span using Full speed mode. Real FLAC
excerpts and a pause/resume case were compared with the host reference:

| Measure | Result |
| --- | ---: |
| Device/host embedding cosine, track 1 | 0.99903 |
| Device/host embedding cosine, track 2 | 0.99709 |
| MERT GPU inference | 196-198 ms per 5-second window |
| CLaMP3 audio stage | 23-24 ms per track segment |
| Exact incremental graph tail | 13.4-13.8 s |

Pause/resume persisted and reproduced the same device audio vector. A final controlled 25-second
track proved the Settings lifecycle from exactly one ready track, through completion, to no tracks
ready, followed by visible cleanup and restoration of the prior 89,737-row semantic state.

The lifecycle also exposed an avoidable 14 minute 39 second full graph rebuild caused by discarding
an otherwise exact graph proof. Proof propagation was corrected, and the final generation is
eligible for the incremental path. The evidence does not claim that another subsequent real
addition exercised that corrected path on the final APK.

## Server Indexer And Merge

The real server publication contained 8,832 embeddings in a 43.3 MiB graphless database. Across 31
complete publication batches, observed server embedding time had:

- median 30.907 seconds per track;
- p90 42.342 seconds per track.

These are batch-derived rates, not 8,832 independent per-track timing samples.

On Android, a merge started while the Settings library comparison was active. The two operations
serialized instead of racing or crashing. The merge validated all 8,832 rows and completed in
42.8 seconds with:

- 0 added;
- 8,832 already indexed;
- no active-generation corruption;
- no Poweramp queue change.

This proves overlap-safe no-op behavior. The concluding live run did not prove a positive-add
server merge.

## Known Limits

- Musical quality still inherits CLaMP3's strengths and blind spots.
- Duplicate recording identity is not solved generally. A safe proof-grade rule collapsed exact
  copies, but seven obvious duplicate slots remained across six of 54 reviewed new-mode queues.
  More aggressive metadata grouping also false-merged distinct recordings, so it was rejected.
- Greedy DPP is certified against the complete-domain greedy sequence. It is not a proof of the
  globally optimal DPP MAP set.
- Determinism is proven for the same validated embedding generation and complete saved request.
  Independently generated embeddings on different hardware are parity-gated, not promised
  byte-identical.
- The current Android artifact is an experimental, side-by-side build rather than a signed store
  release.

## Reproducing Repository Checks

Desktop:

```bash
cd desktop-indexer
uv sync --extra dev --extra export
uv run --extra dev ruff check .
uv run --extra dev pytest tests experiments -q
```

Android host checks:

```bash
cd android-plugin
./gradlew :app:testDebugUnitTest :app:assembleDebug :app:assembleRelease \
  :app:assembleDebugAndroidTest --no-daemon
./gradlew :app:lintDebug --no-daemon
```

The complete corpus evaluation requires a compatible local snapshot and is intentionally not
bundled with the repository. Evaluators live under `desktop-indexer/experiments/`; aggregate
results, formulas, caveats, and the evidence boundary are preserved here so public documentation
does not expose personal-library rows or device dumps.
