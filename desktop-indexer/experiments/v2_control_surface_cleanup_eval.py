#!/usr/bin/env python3
"""Measure clean V2 control stops against the frozen active embedding library."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

import compare_device_feature_acceptance as phone_oracle
import v2_queue_eval as queue_eval
import v2_selection_knob_matrix as matrix


OUTPUT = (
    matrix.REPO_ROOT
    / "desktop-indexer"
    / "audit_raw_data"
    / "v2-discovery"
    / "control-surface-cleanup-2026-07-15"
    / "results.json"
)


def comparison(left: list[int], right: list[int]) -> dict[str, float | bool]:
    shared_positions = sum(a == b for a, b in zip(left, right))
    left_set = set(left)
    right_set = set(right)
    top_ten = min(10, len(left), len(right))
    return {
        "exact_order": left == right,
        "same_position_fraction": shared_positions / max(len(left), len(right), 1),
        "set_jaccard": len(left_set & right_set) / max(len(left_set | right_set), 1),
        "top10_membership_churn": 1.0
        - len(set(left[:top_ten]) & set(right[:top_ten])) / max(top_ten, 1),
    }


def summarize(rows: list[dict[str, float | bool]]) -> dict[str, float | int]:
    numeric = ("same_position_fraction", "set_jaccard", "top10_membership_churn")
    result: dict[str, float | int] = {
        "seeds": len(rows),
        "exact_order_no_op": sum(bool(row["exact_order"]) for row in rows),
    }
    for key in numeric:
        values = [float(row[key]) for row in rows]
        result[f"mean_{key}"] = float(np.mean(values))
        result[f"min_{key}"] = float(np.min(values))
        result[f"max_{key}"] = float(np.max(values))
    return result


def main() -> None:
    phone_report = json.loads(matrix.DEFAULT_PHONE_REPORT.read_text(encoding="utf-8"))
    seed_ids = [int(value) for value in phone_report["request"]["seedTrackIds"]]
    active_ids = phone_oracle.active_track_ids(matrix.DEFAULT_ACTIVE_CATALOG)
    full_library, database_sha256 = queue_eval.load_library(matrix.DEFAULT_DATABASE)
    active_mask = np.fromiter(
        (int(track_id) in active_ids for track_id in full_library.track_ids),
        dtype=np.bool_,
        count=full_library.count,
    )
    active_positions = np.flatnonzero(active_mask)
    library = matrix.subset_library(full_library, active_positions)
    del full_library, active_mask
    id_to_index = {int(track_id): index for index, track_id in enumerate(library.track_ids)}
    contexts = {
        seed_id: matrix.seed_context(library, id_to_index[seed_id]) for seed_id in seed_ids
    }

    base_mmr = matrix.SelectorConfig(
        mode="mmr",
        mmr_lambda=0.4,
        reach=0.02,
        queue_size=30,
    )
    base_seed_journey = matrix.SelectorConfig(
        mode="drift",
        mmr_lambda=0.4,
        reach=0.02,
        queue_size=30,
        drift_mode="SEED_INTERPOLATION",
        anchor_schedule="EXPONENTIAL",
        anchor_strength=0.5,
        anchor_half_life=7.0,
    )
    base_momentum = replace(
        base_seed_journey,
        drift_mode="MOMENTUM",
        momentum_beta=0.9,
    )
    cases = {
        "mmr_0.95_to_0.97032166": (
            replace(base_mmr, mmr_lambda=0.95),
            replace(base_mmr, mmr_lambda=0.97032166),
        ),
        "mmr_0.97032166_to_1": (
            replace(base_mmr, mmr_lambda=0.97032166),
            replace(base_mmr, mmr_lambda=1.0),
        ),
        "seed_pull_0.75_to_0.85": (
            replace(base_seed_journey, anchor_strength=0.75),
            replace(base_seed_journey, anchor_strength=0.85),
        ),
        "seed_pull_0.8440951_to_0.85": (
            replace(base_seed_journey, anchor_strength=0.8440951),
            replace(base_seed_journey, anchor_strength=0.85),
        ),
        "seed_pull_0.85_to_1": (
            replace(base_seed_journey, anchor_strength=0.85),
            replace(base_seed_journey, anchor_strength=1.0),
        ),
        "fade_timing_6.7004_to_7": (
            replace(base_seed_journey, anchor_half_life=6.7004),
            replace(base_seed_journey, anchor_half_life=7.0),
        ),
        "momentum_0.75_to_0.9": (
            replace(base_momentum, momentum_beta=0.75),
            replace(base_momentum, momentum_beta=0.9),
        ),
        "momentum_0.9_to_0.92064106": (
            replace(base_momentum, momentum_beta=0.9),
            replace(base_momentum, momentum_beta=0.92064106),
        ),
        "momentum_0.9_to_0.95": (
            replace(base_momentum, momentum_beta=0.9),
            replace(base_momentum, momentum_beta=0.95),
        ),
        "momentum_0.95_to_1": (
            replace(base_momentum, momentum_beta=0.95),
            replace(base_momentum, momentum_beta=1.0),
        ),
    }

    results: dict[str, dict[str, float | int]] = {}
    for label, (left_config, right_config) in cases.items():
        rows = []
        for seed_id in seed_ids:
            context = contexts[seed_id]
            left = matrix.run_config(library, None, None, context, left_config)["track_ids"]
            right = matrix.run_config(library, None, None, context, right_config)["track_ids"]
            rows.append(comparison(left, right))
        results[label] = summarize(rows)
        print(f"completed {label}", flush=True)

    payload = {
        "experiment_version": "control-surface-cleanup-v1",
        "frozen_inputs": {
            "database": str(Path(matrix.DEFAULT_DATABASE).resolve()),
            "database_sha256": database_sha256,
            "active_catalog": str(Path(matrix.DEFAULT_ACTIVE_CATALOG).resolve()),
            "active_catalog_sha256": matrix.sha256_file(matrix.DEFAULT_ACTIVE_CATALOG),
            "phone_report": str(Path(matrix.DEFAULT_PHONE_REPORT).resolve()),
            "phone_report_sha256": matrix.sha256_file(matrix.DEFAULT_PHONE_REPORT),
            "active_track_count": library.count,
            "seed_track_ids": seed_ids,
        },
        "fixed_config": {
            "queue_size": 30,
            "mmr_relevance": 0.4,
            "reach": 0.02,
            "artist_limits": True,
            "max_per_artist": 8,
            "artist_spacing": 3,
        },
        "results": results,
    }
    matrix.atomic_json(OUTPUT, payload)
    print("RESULT_JSON")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
