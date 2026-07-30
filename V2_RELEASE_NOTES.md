# Poweramp Start Radio V2 Release Brief

Date: 2026-07-30

Package: `com.powerampstartradio.v2` (`2.0.0-experimental`)

Last on-device accepted build: debug APK, SHA-256
`91fec63a92968a52bedbda0bf688c58abef07a63629fe601fbf7a49010030528`

The source tree has received release-hygiene and prerequisite-gating fixes since that device run.
Host verification for the current source is recorded separately; the hash above is historical
device-acceptance identity, not the newly built APK.

## Current Source Host Gates

The current source, including the release-hygiene and prerequisite-gating fixes
made after device acceptance, passed:

- Android: 886 debug unit tests, zero failures or errors, one skipped test.
- Android: debug, unsigned release, and instrumentation-test APK assembly.
- Android lint: zero errors and 80 advisory warnings.
- Desktop: 196 tests across the maintained package and experiment references.
- Desktop: Ruff checks and wheel/sdist construction.
- Distribution inspection: the sdist contains only package source, maintained
  tests, package documentation, metadata, and license; the wheel carries the
  package license in its distribution metadata.

Current APK identities:

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| Debug APK | 20,848,663 | `b32bb8d3a9ba958f5183d9f181dadc4f50cf605a2ada7ee014c84ba3a6850ecf` |
| Unsigned release APK | 14,194,328 | `b58fc383953bf736dcdad011b2a877c16e8a922aed9b6fefaa6459260ea95c98` |
| Instrumentation-test APK | 1,372,859 | `1ce1e90c9d723fab939465ff4e0cd88a79c54dc7c18b6e0018663086a831e904` |

These are the local verification artifacts, not reproducible-build identities.
A clean detached worktree produced byte-different debug and release APKs in five
entries each (DEX/profile and native-library payloads), while the instrumentation
APK and both Python distributions were byte-identical. A distributable release
still needs signing and reproducible-build provenance.

These newly built APKs were not installed or rerun on the disconnected phone in
this documentation pass. Device claims below belong to the historical accepted
debug APK identified above.

Changes covered by these host gates, but not the historical device run, include
the exact current-queue-row radio action, the cheap Find Music prerequisite
gate, release/debug source separation, and the versioned desktop MERT cache.

## Release Position

V2 is an acceptance candidate for the current owner and device when the exact
models are manually provisioned. Its recommendation, queue construction,
on-device indexing, and reviewed UI workflows have direct device evidence.

It is not yet a general distributable production release:

- The APK does not bundle, download, or import the required model files.
  Acceptance used exact models staged into the app's private files directory
  with debug/root tooling.
- MERT-v1-95M is an external CC-BY-NC-4.0 model. General distribution needs a
  license-compliant provisioning design; see `THIRD_PARTY_NOTICES.md`.
- Server merge has proved the safe all-already-indexed path only. Adding and
  activating genuinely new server embeddings has not yet been accepted on
  device.

## What This V2 Keeps

- Global queue modes remain `Closest`, `MMR`, `DPP`, `Graph Explorer`, and
  `Uniform shuffle`.
- Multi-anchor All-of offers contextual `Ranked` and `Varied (DPP)` planners.
  These are workflow choices, not extra global radio modes.
- `Refine` uses one primary request plus a secondary direction, with `Like` and
  `Avoid` controls.
- Candidate domains, date filters, and true global `# nearest` evidence remain
  explicit.
- Queue creation is deterministic for identical inputs and database state.
- On-device indexing supports truthful lifecycle progress, pause/resume, and
  cleanup.

## Frozen Host Evidence

The concluding host review used one immutable CLaMP3 snapshot:

| Property | Frozen value |
| --- | ---: |
| Embedding rows | 89,737 |
| Dimensions | 768 |
| Reconstructed visible identities | 89,539 |
| Collapsed duplicate occurrences | 198 |
| Active device occurrences | 89,669 |
| Full ingredient domain, including seed | 89,528 |
| Objective domain, after seed exclusion | 89,527 |
| Generation | `index-generation-v2-b83e9abb8889936f65b156e6e2abf1ac922265db56504d968f0eb71557f27687` |
| Activation binding | `activation-binding-v3-0686721f06a72e05e64febaff7a2d5d1e87d1a090842e5b8192dce22542d2078` |
| Database file SHA-256 | `dc48ce36a161cd47e780f0f9de3405eebd3a6e70c1b4050f684412af51debc06` |
| Database content SHA-256 | `e737a0f507927bc19160d3204b8e202e5da9286baae55f0ba38ee15ad5792f91` |
| Embeddings SHA-256 | `b78f0727c7e1bb3ae4843eacddeed306fea61a9496519db7bbf47b8bb1832dbe` |
| Graph SHA-256 | `784d861e774366c3700953f32cf8049822bd4dc15b5f16fddb6062bac1276b2b` |
| Manifest SHA-256 | `0fd447ab0f8481079db018178c04c8acf0a918a43d884b61a98987508887b9c5` |

### All-of

The review covered 13 requests: 12 two-ingredient requests and one
three-ingredient request (`slow` + `psychedelic` + `guitar`). Ranked and Varied
were each repeated exactly for every request.

Across those 13 requests, Varied relative to Ranked:

- Changed the selected set materially: mean top-30 overlap was 22.15, ranging
  from 3 to 28.
- Reduced mean pairwise cosine by 0.0489.
- Added 3.46 unique artists on average.
- Paid a small mean objective-score cost of 0.000640.
- Had a median objective rank of 46, with a range of 37 to 329.

This supports two distinct requests: Ranked for the strongest joint matches,
and Varied for a broader set that still satisfies the same anchors.

### Refine and Controls

Seven Refine cases were measured at four primary-neighborhood widths:

| Width | Exact candidates | Median primary floor | Median secondary gain |
| --- | ---: | ---: | ---: |
| 0.25% | 224 | 0.99754 | 0.2444 |
| 0.5% | 448 | 0.99512 | 0.2907 |
| 1% | 896 | 0.99024 | 0.2987 |
| 2% | 1,791 | 0.98075 | 0.30294 |

The tradeoff was monotonic in all seven cases. `1%` remains the default because
it captured almost all measured secondary gain while preserving a tight
primary neighborhood. All four widths remain meaningful controls.

Two-anchor weight sweeps had own-anchor correlations of 0.965-0.980 for Ranked
and 0.908-0.954 for Varied, with no adjacent ordered no-ops. Three-or-more
anchor 1% steps had correlations of 0.929-0.966; adjacent queues overlapped
heavily but moved about 11 positions on average. All three Like/Avoid probes
changed membership materially.

### Graph Repair

Graph construction now operates on the 89,539-identity quotient rather than
duplicate occurrences. The repair affected 343 rows and 520 of 447,695 graph
slots. Host construction took 2,248 ms, including a 2,049 ms exact repair scan.

Default-stop output was unchanged for 9 of 11 reviewed seeds. It repaired the
Tool neighborhood, where the occurrence graph had lost 45.7% of terminal mass.
The Brian Regan neighborhood remained genuinely sparse and is reported as
exhausted rather than padded with unrelated tracks.

## Device Evidence

### Broad Frozen Matrix

The full device matrix ran 30 selector/control cases across 11 seeds, twice:
660 executions and 330 distinct case/seed pairs. All 330 repeat fingerprints
matched exactly. It produced:

- Zero Poweramp queue mutation calls.
- Zero inactive results.
- Zero repeated track IDs within a run.
- Thirty results for normal cases.
- One result for each Brian Regan Graph case at 5%, 50%, and 90% stop settings,
  repeated exactly. This is truthful sparse-neighborhood exhaustion.

The matrix preceded the final selector repairs. It remains broad frozen-snapshot
evidence; the repaired code was then rerun separately.

### Repaired Selector Runs

Post-repair acceptance ran five seeds through `Closest`, maximum-relevance
`MMR`, `DPP`, and `Graph`, twice: 40 executions and 20 distinct pairs. All 20
repeat fingerprints matched, with no queue mutations, inactive results, or
duplicate Poweramp resolutions.

Representative warm second-run selector times on the Sony XQ-EC72 were:

| Selector | Warm device time |
| --- | ---: |
| Closest | 304-382 ms |
| MMR, maximum relevance | 805-1,088 ms |
| DPP | 2.7-19.6 s, seed dependent |
| Graph | 877-976 ms for full queues |
| Graph, sparse Brian Regan case | 110-114 ms for one result |

The broad matrix's most expensive full-domain/extreme-knob case took 46.498 s.
V2 does not hide that quality-first full-domain selectors can take seconds or,
at extremes, tens of seconds.

### Repaired Composed Workflows

Seven composed cases were run twice. All 14 runs were deterministic, restored
settings exactly, and made no queue mutation calls.

Warm second-run result/queue times were:

| Workflow | Result / queue |
| --- | ---: |
| All-of Ranked, Drift 70 / Sleep 30 | 339 / 347 ms |
| All-of Varied, Drift 70 / Sleep 30 | 567 / 569 ms |
| All-of Ranked, Drift 30 / Sleep 70 | 348 / 359 ms |
| All-of Ranked, Drift 70 / Avoid Sleep 30 | 356 / 368 ms |
| All-of Varied, Drift 70 / Avoid Sleep 30 | 524 / 536 ms |
| Four-text Varied, recent 14 days | 208 / 213 ms |
| Refine, 1% | 347 / 361 ms |

The controls changed useful output rather than merely relabeling it:

- Changing anchor weights left only 2 shared tracks; 28 changed on each side.
- Like versus Avoid had zero shared tracks.
- Ranked versus Varied shared 22 tracks, changed 8, and moved 18 shared tracks.
- All-of versus Refine shared 7 tracks and changed 23.

## Operational Acceptance

- Home, history, Find Music, Refine evidence, Settings, indexing, and cleanup
  were visually reviewed on device.
- The final app state was 89,737 tracks with no tracks ready to index.
- The existing 51-entry Poweramp queue retained the same order through
  acceptance.
- A cold home launch recorded 676 ms `TotalTime` and 678 ms `WaitTime`.
- The final-build launch artifact, despite its `cold-start.txt` filename,
  explicitly recorded `LaunchState: WARM`, 715 ms `TotalTime`, and 717 ms
  `WaitTime`.

On-device indexing was exercised with two real excerpts and one full-span
pause/resume run. Device-to-host vector cosine was 0.99903 and 0.99709, and
pause/resume produced the same vector. A controlled 25-second track proved one
ready item, management-state reuse, successful completion, no-track completion,
and visible cleanup.

One acceptance path triggered an avoidable 14 minute 39 second full graph
rebuild after its reuse proof was dropped. The final build now clears stale
inherited proof and rebinds only exact graphs. Cleanup restored the same
89,737-row semantic state and left the corrected proof in place, so future
additions are eligible for the incremental path.

## Server Merge Evidence

A real 8,832-row server export was merged while Settings library comparison was
also active. The operations serialized without a crash, validation progress was
determinate, and all 8,832 rows were correctly classified
`ALREADY_INDEXED`. Elapsed time was 42.819 seconds and the Poweramp queue was
unchanged.

This is no-op-only proof. It validates concurrency serialization, source
validation, progress, and overlap classification. It does not prove insertion,
activation, graph update, or matching for genuinely new server embeddings.

## Known Limits

- Model provisioning is debug-only/manual. A release APK needs a deliberate,
  verified way to supply the exact required models before general distribution.
- The server indexer pins accepted MERT and CLaMP3 revisions, but ordinary
  desktop scan, text loading, and LiteRT export do not yet share one enforced
  revision/hash policy. Do not describe fresh cross-host model export as
  reproducible until that contract is centralized.
- Positive server merge remains unaccepted until a controlled new row is added,
  activated, matched to Poweramp, and survives restart.
- CLaMP3 similarity plus metadata cannot prove duplicate identity. A reviewed
  heuristic would remove seven obvious slots across six of 54 queues, but also
  falsely merged distinct Cure discs and Anthony Rother recordings. No
  heuristic dedupe ships; exact decoded-PCM identity remains the proof-grade
  future route.
- Graph can truthfully return fewer than the requested queue length when a
  reciprocal neighborhood is exhausted.
- Quality-first DPP and extreme full-domain controls can be noticeably slower
  than ranking-based workflows.

## Release Decisions

| Area | Decision |
| --- | --- |
| Global modes | Keep the existing modes; add no review-derived global mode |
| All-of | Keep contextual Ranked and Varied planners |
| Refine | Keep all four widths; use 1% as the default |
| Like/Avoid | Keep; measured membership changes justify the control |
| Graph | Keep the identity-quotient repair and truthful exhaustion |
| Percentile seed handling | Retain the current domain; 12 of 12 probes were identical when excluding the seed first |
| Duplicate suppression | Defer heuristic suppression until identity can be proved |
| Server merge | Keep experimental; require a positive-add acceptance before release |
| Distribution | Owner/debug candidate only until model provisioning is solved |

## Evidence Map

The following paths identify the ignored local evidence archive used to derive
the maintained aggregate tables above; they are not part of the public source
distribution.

- Frozen host review:
  `discovery/evidence/concluding-recommendation-review-2026-07-30/`
- Broad device selector matrix:
  `discovery/device-acceptance/20260730T0325+0300-concluding-selection/report.json`
- Repaired selector rerun:
  `discovery/device-acceptance/20260730T0446+0300-reviewed-selector-repairs/report.json`
- Repaired composed workflows:
  `discovery/device-acceptance/20260730T0450+0300-reviewed-composed-modes/report.json`
- Final UI and lifecycle acceptance:
  `discovery/device-acceptance/20260730T0500+0300-concluding-ui-review/ACCEPTANCE.md`
- No-op server merge log:
  `discovery/device-acceptance/20260730T0500+0300-concluding-ui-review/server-merge-final-overlap-verified/relevant-logcat.txt`
