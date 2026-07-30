# Poweramp Start Radio V2 Trial Brief

Date: 2026-07-30

Package: `com.powerampstartradio.v2` (`2.0.0-experimental`)

## Trial Position

This build is the owner's V2 trial candidate. It preserves the app's core role:

- build a radio from Poweramp's current playing or paused track;
- find and queue a complete ordered result from text or confirmed recording ingredients;
- grow one active music index on-device or by merging a compatible server index.

Find Music has no per-row radio action. Its result is one complete queue. Single-seed radio starts
only from Poweramp's current track, including while playback is paused.

## Current Host Gates

The current source passed:

- Android: 886 debug unit tests, zero failures or errors, one skipped test;
- Android: debug, unsigned release, and instrumentation-test APK assembly;
- Android lint: zero errors;
- Desktop: 196 maintained tests, Ruff, and package construction.

The Android changes in this trial build have not yet received a fresh phone run. The owner will
exercise the APK in ordinary use before merging the branch.

## Visible Decisions

| Surface | Decision |
| --- | --- |
| Single-seed radio | Keep Closest, Uniform shuffle, MMR, DPP, and Graph Explorer |
| Find Music, one description | Keep Closest and Varied |
| Find Music, several ingredients | Keep All-of Ranked, All-of Varied, and Refine |
| Ingredient direction | Keep Like and Less like |
| Candidate domains | Keep explicit Selection pool controls where they change the selector's promise |
| Evidence | Keep true global `# nearest`; leave raw scores in expanded details |
| Duplicate handling | Collapse only proof-grade copies; do not infer identity from metadata plus cosine |
| Queue rows | Remove per-row radio; queue or replay the displayed result as a whole |
| Settings library comparison | Show the last completed count; scan only on explicit Refresh or Manage Tracks |
| Index lifecycle | Keep one active immutable index generation with on-device growth and server merge |

The exact algorithms and visible controls are documented in
[`MODES_AND_KNOBS.md`](MODES_AND_KNOBS.md).

## Evidence Boundary

The accepted July 30 review used 89,737 CLaMP3 embeddings and 89,669 active Poweramp occurrences.
It covered selector/control determinism, contextual Find Music planners, complete-span on-device
indexing, pause/resume, graph publication, and an overlap-safe no-op server merge. Aggregate
results, timings, and limitations live in [`EVALUATION.md`](EVALUATION.md); this brief does not
duplicate that ledger.

The last accepted device build predates the final source cleanup. The current trial additionally:

- removes the unused queue-row and Find Music-to-radio submission paths;
- avoids a full Poweramp-library comparison merely because Settings was opened;
- removes unreachable Home index-preparation state and no-caller compatibility wrappers;
- clarifies desktop-update, server-publication, and model-path instructions.

## Known Limits

- Musical quality inherits CLaMP3's strengths and blind spots.
- Required model files are provisioned manually and are not bundled in the APK.
- MERT-v1-95M is CC-BY-NC-4.0; general distribution needs a compatible model-delivery decision.
- A positive-add server merge still needs on-device acceptance. The measured live merge contained
  only rows already indexed on the phone.
- Duplicate recording identity is not solved generally. Similar embeddings are not sufficient
  evidence to collapse editions, masters, or live versions.
- Greedy DPP is certified against its complete-domain greedy sequence, not global DPP MAP
  optimality.
- Graph Explorer may return fewer than requested when its eligible reachable component is
  exhausted.
- Full-domain DPP and extreme quality-first controls can take seconds or, in measured extremes,
  tens of seconds.
- The final graph-proof propagation fix is eligible for incremental updates but has not yet been
  exercised by another real addition on this exact APK.
