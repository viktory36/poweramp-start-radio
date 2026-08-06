# Recommendation Experiments

Host-side references for recommendation algorithms, controls, and model parity.
These scripts are not the app's list of supported modes. Published results are in
[Measurements](../../EVALUATION.md); current behavior is in [Modes and controls](../../MODES_AND_KNOBS.md).

## Environment

From the repository root:

```bash
python -m venv .venv-discovery
. .venv-discovery/bin/activate
python -m pip install -r desktop-indexer/experiments/requirements-discovery.txt
python -m pip install -e 'desktop-indexer[dev]'
PYTHONPATH=desktop-indexer/src OPENBLAS_NUM_THREADS=8 \
  python -m pytest -q desktop-indexer/experiments/test_*.py
```

Use a matching Torch/torchaudio build for your accelerator if the pinned requirements do not suit it.

## Entry Points

| Script | Purpose |
| --- | --- |
| `v2_current_selection_audit.py` | Radio modes, control effects, repeated order, and runtime |
| `v2_active_composition_eval.py` | All-of and Refine against the active phone identity domain |
| `v2_focused_recommendation_eval.py` | Compare candidate algorithms and stress cases |
| `v2_control_surface_cleanup_eval.py` | Detect redundant or ineffective controls |
| `v2_audio_embedding_parity.py`, `v2_text_embedding_parity.py` | Host references for device embeddings |
| `v2_tokenizer_parity.py` | Host/device tokenization comparison |
| `resampler_intelligence_eval.py` | Compare audio preprocessing |
| `compare_device_feature_acceptance.py`, `summarize_device_selection_utility.py` | Summarize captured device results |

Use a script's `--help` for inputs and output options. Other files preserve experimental formulations;
their names do not imply those algorithms are available in the app.

## Running Safely

- Use a copied, read-only database snapshot, never the phone's live private index.
- Provide the active-library catalog when evaluating phone selection: not every database row is
  currently present, and copy identities affect the candidate domain.
- Keep generated arrays, raw queues, and device captures in the ignored `discovery/` or
  `desktop-indexer/audit_raw_data/` directories.
- Record enough input/version information to repeat a comparison. Resume a checkpoint only against
  the same inputs.
- Replace placeholder paths in `resampler_intelligence_manifest.json` with your own audio cohort.

Compare algorithms serving the same musical request. Check relevance, within-queue redundancy,
anchor satisfaction, membership changes, determinism, filters, duplicate handling, and runtime.
Inspect the actual tracks as well: improved numerical diversity alone is not proof of a better queue.
