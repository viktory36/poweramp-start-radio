# Host Recommendation Experiments

This directory contains host-side reference implementations and measurement tools for Poweramp
Start Radio. They are research and acceptance tooling, not Android runtime code and not a catalog
of supported product modes.

The current production space is the pinned 768-dimensional CLaMP3 index. Evaluators use track
metadata only for identity, cohorts, and readable output; metadata does not enter musical relevance
scores.

## Authoritative Results

The concluding review is:

- [`../../EVALUATION.md`](../../EVALUATION.md)
- [`../../V2_RELEASE_NOTES.md`](../../V2_RELEASE_NOTES.md)

Those records retain All-of Ranked, All-of Varied, Refine, Like/Avoid, the measured Refine widths,
and the identity-safe Graph Explorer repair. They reject adding another global selector from this
investigation. Historical scripts remain so rejected and superseded ideas can be reproduced; their
filenames do not imply a current app capability.

## Environment

From the repository root:

```bash
python -m venv .venv-discovery
. .venv-discovery/bin/activate
python -m pip install -r desktop-indexer/experiments/requirements-discovery.txt
python -m pip install -e 'desktop-indexer[dev]'
```

Use a platform-appropriate Torch/Torchaudio build when the pinned requirements are not suitable for
the host accelerator.

Run the maintained experiment checks with:

```bash
PYTHONPATH=desktop-indexer/src OPENBLAS_NUM_THREADS=8 \
  python -m pytest -q desktop-indexer/experiments/test_*.py
```

## Safety And Evidence

- Production snapshots are opened read-only, normally with SQLite immutable mode.
- Serious evaluators bind their inputs to database, active-catalog, model, prompt/cohort, and policy
  identities.
- Hash-bypass flags are development-only. A result generated with one is not production evidence.
- Long runs checkpoint atomically and resume only when their recorded identities still match.
- Generated arrays, caches, manifests, and JSON results belong under the ignored
  `desktop-indexer/audit_raw_data/` tree.
- Raw personal-library evidence belongs under the ignored `discovery/` tree. Curated methods,
  aggregate measurements, limitations, and final decisions belong in the maintained root docs.

Do not point an evaluator at the Android app's live private generation. Pull or reconstruct a frozen
host snapshot first.

## Current Reference Entry Points

- `v2_current_selection_audit.py` measures the visible single-seed selector surface, controls,
  deterministic repeats, active identity domain, and runtime.
- `v2_active_composition_eval.py` evaluates Find Music All-of and Refine against the exact active
  phone domain.
- `v2_focused_recommendation_eval.py` compares serious recommendation candidates and stress cases
  against recorded baselines.
- `v2_control_surface_cleanup_eval.py` checks whether visible options make distinct, predictable
  changes.
- `v2_audio_embedding_parity.py`, `v2_text_embedding_parity.py`, and
  `v2_tokenizer_parity.py` provide host references for device parity.
- `resampler_intelligence_eval.py` and `generate_torchaudio_hann_v1_fixtures.py` bind preprocessing
  parity. Replace the generic paths in `resampler_intelligence_manifest.json` with a deliberate
  local cohort before running the full audio experiment.
- `compare_device_feature_acceptance.py` and `summarize_device_selection_utility.py` consume device
  evidence without changing the Poweramp queue.

Use each script's `--help` and its matching discovery report for required frozen inputs and output
contracts. Prefer extending an existing reference evaluator over creating another one-off formula
when the production promise is the same.

## Evaluation Standard

A serious candidate should record:

1. the musical request it is meant to satisfy;
2. the exact formula and ordered queue construction;
3. immutable input and policy identities;
4. determinism and runtime;
5. relevance, redundancy, coverage, hubness, anchor satisfaction, domain correctness, and duplicate
   exposure where applicable;
6. qualitative queues across mainstream, niche, non-English, electronic, ambient, acoustic, dense,
   and sparse neighborhoods; and
7. a retain, replace, refine, or reject decision.

Production integration is justified only when the candidate offers a distinct repeatable request or
measurably improves an existing promise. The final UI should expose the smallest complete vocabulary,
not every algorithm that was tried.
