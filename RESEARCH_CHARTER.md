# Poweramp Start Radio Improvement Charter

This is the durable brief for future research. It records the product target and the
rules for evidence; it deliberately does not choose a model or algorithm in advance.

## Product Target

Poweramp Start Radio should make queues that a serious music listener repeatedly wants
to keep playing. Improvements must be visible in real use, not merely in a paper metric.

The central quality goals are:

- preserve the particular character of the seed instead of falling back to its broad
  genre neighborhood;
- work especially well for niche electronic music, non-English music, culturally
  specific forms, unusual production, and tracks that cross common genre boundaries;
- make text-to-song retrieval precise enough for specific musical intent;
- produce coherent queues without flattening the library into safe, generic choices;
- remain deterministic: identical libraries, models, settings, and requests must produce
  identical results; and
- keep indexing and runtime behavior robust, resumable, measurable, and fast enough for
  daily use without trading away recommendation quality merely to save compute.

## Intelligence Boundary

Core recommendation intelligence comes from raw audio and text model embeddings plus
deterministic algorithms operating on those embeddings.

Poweramp metadata, filenames, tags, play history, ratings, scraped labels, and personal
listening statistics must not silently enter similarity or relevance scores. Metadata may
still be used for identity, display, explicit user constraints, duplicate/version handling,
and blinded evaluation labels. Those uses must remain separate and inspectable.

Universal musical similarity and queue construction are separate problems. Research must
therefore keep two independently measurable lanes:

1. **Model intelligence:** raw text-to-song and song-to-song retrieval from a fixed,
   aligned candidate pool.
2. **Queue intelligence:** ordering, diversity, drift, graph traversal, repetition control,
   and multi-seed behavior while holding the embeddings fixed.

## Production Baseline

The starting point is the database actually used by the connected phone, not an assumed
desktop state:

- snapshot: `desktop-indexer/audit_raw_data/phone-snapshots/2026-07-07T223308+0300_qv7706c3mq/embeddings.db`
- source device: Sony XQ-EC72
- source modification time: `2026-07-07T22:33:08+03:00`
- SHA-256: `08dfcec60f7c2e9de4bc6b923d601bd824f80b6251769f6c7bcd8062ce6aa504`
- tracks and CLaMP3 embeddings: `80,421`
- embedded graph: `80,421` nodes, `K=5`

The adjacent ignored `snapshot-manifest.json` records acquisition provenance and integrity,
embedding, and graph checks. The snapshot is immutable; experiments work on copies.

Existing measured behavior in `EVALUATION.md` and controls in `MODES_AND_KNOBS.md` are the
baseline until a fresh measurement supersedes them.

## Experiment Discipline

Every experiment must follow these rules:

1. State the observed product problem, hypothesis, expected user-visible improvement, and
   rejection criterion before implementation.
2. Freeze cohorts and prompts without consulting candidate-model rankings.
3. Compare the exact same tracks, exclusions, queries, seeds, and retrieval depth.
4. Change one intelligence layer at a time. Report raw retrieval separately from queue
   algorithms, filters, metadata constraints, and presentation.
5. Keep checkpoints, preprocessing, sampling, aggregation, precision, hashes, and runtime
   configuration reproducible.
6. Use quantitative diagnostics to find differences, then judge musical quality blindly.
   Direct listening and the user's judgment decide ambiguous or product-critical cases.
7. Include ordinary tracks as the primary population. Report long-form, niche,
   multilingual, and other important slices separately so no stress cohort dominates the
   conclusion.
8. Measure recommendation quality, determinism, latency, throughput, storage, memory,
   crash recovery, and resume behavior. Quality remains the gate.
9. Treat public benchmarks as candidate-screening evidence only. A replacement or new
   algorithm must win on this library and this product task.
10. Archive failed experiments cleanly and return to the production baseline.

## Reset Procedure

The next investigation starts from the production snapshot and works outward:

1. reproduce current raw retrieval and queue behavior from the phone database;
2. assemble a small golden set of real seeds, prompts, successes, and recurring failures;
3. turn those failures into measurable, blinded tests;
4. identify whether each failure belongs to the model or queue lane; and
5. investigate the smallest defensible change that can beat the frozen baseline.

No replacement model, fine-tuning plan, enrichment scheme, or queue algorithm is presumed
to be the answer.
