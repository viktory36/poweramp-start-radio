#!/usr/bin/env python3
"""Calibrate the current V2 MMR control between relevance 0.4 and 0.6.

The experiment is deliberately narrow: the frozen 80,323-track phone domain, the
same eleven seeds as the current-selector audit, reach 2%, and queue length 30.
Every case is evaluated twice and the 0.4/0.6 endpoints are also checked against
the frozen parent audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import asdict
from itertools import combinations
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import compare_device_feature_acceptance as phone_oracle
import v2_current_selection_audit as current_audit
import v2_queue_eval as queue_eval
import v2_selection_knob_matrix as matrix


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATABASE = queue_eval.DEFAULT_DB
DEFAULT_ACTIVE_CATALOG = current_audit.DEFAULT_ACTIVE_CATALOG
DEFAULT_PARENT_RECORDS = current_audit.DEFAULT_OUTPUT / "records.jsonl"
DEFAULT_OUTPUT = (
    REPO_ROOT / "discovery" / "selection-mode-audit" / "20260715-mmr-midpoint"
)

EXPECTED_DATABASE_SHA256 = queue_eval.EXPECTED_DB_SHA256
EXPECTED_ACTIVE_CATALOG_SHA256 = matrix.EXPECTED_CATALOG_SHA256
EXPECTED_ACTIVE_TRACKS = matrix.EXPECTED_ACTIVE_TRACKS

SEEDS = current_audit.ALL_SEEDS
LAMBDAS = (0.4, 0.5, 0.55, 0.6)
NAMED_SEEDS = (80437, 33821, 17399, 5987, 31799)
QUEUE_SIZE = 30
REACH = 0.02


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            value,
            handle,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            default=queue_eval.json_numpy_scalar,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_jsonl(path: Path, values: Sequence[object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(
                json.dumps(
                    value,
                    sort_keys=True,
                    ensure_ascii=True,
                    default=queue_eval.json_numpy_scalar,
                )
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def numeric(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0, "mean": None, "min": None, "p50": None, "max": None}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "min": float(array.min()),
        "p50": float(np.median(array)),
        "max": float(array.max()),
    }


def set_jaccard(left: Sequence[int], right: Sequence[int]) -> float:
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 1.0


def queue_change(left: Sequence[int], right: Sequence[int]) -> dict[str, float]:
    top = min(10, len(left), len(right))
    return {
        "set_jaccard": set_jaccard(left, right),
        "top10_set_churn": (
            1.0 - len(set(left[:top]) & set(right[:top])) / top if top else 0.0
        ),
        "same_position_fraction": (
            sum(a == b for a, b in zip(left, right)) / min(len(left), len(right))
            if left and right
            else 1.0
        ),
    }


def load_parent_records(
    path: Path,
) -> tuple[
    dict[tuple[float, int], dict[str, object]],
    dict[tuple[str, int], dict[str, object]],
]:
    endpoints: dict[tuple[float, int], dict[str, object]] = {}
    reference_modes: dict[tuple[str, int], dict[str, object]] = {}
    case_to_lambda = {
        "mmr_relevance_0p4": 0.4,
        "mmr_relevance_0p6": 0.6,
    }
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        case_id = str(record.get("case_id"))
        seed_id = int(record["seed"]["track_id"])
        if case_id in {"current_closest", "current_dpp"} and seed_id in SEEDS:
            reference_modes[(case_id.removeprefix("current_"), seed_id)] = record
        value = case_to_lambda.get(case_id)
        if value is None:
            continue
        endpoints[(value, seed_id)] = record
    expected = len(case_to_lambda) * len(SEEDS)
    if len(endpoints) != expected:
        raise ValueError(f"expected {expected} frozen endpoint rows, found {len(endpoints)}")
    expected_references = 2 * len(SEEDS)
    if len(reference_modes) != expected_references:
        raise ValueError(
            f"expected {expected_references} frozen reference rows, found {len(reference_modes)}"
        )
    return endpoints, reference_modes


def deterministic_projection(result: dict[str, object]) -> dict[str, object]:
    return {
        "track_ids": result["track_ids"],
        "candidate_ranks": result["candidate_ranks"],
        "selection_scores": result["selection_scores"],
        "queue_fingerprint": result["queue_fingerprint"],
        "metrics": result["metrics"],
        "tracks": result["tracks"],
    }


def config(value: float) -> matrix.SelectorConfig:
    return matrix.SelectorConfig(
        mode="mmr",
        queue_size=QUEUE_SIZE,
        reach=REACH,
        mmr_lambda=value,
        artist_limits=True,
        max_per_artist=matrix.DEFAULT_MAX_PER_ARTIST,
        artist_spacing=matrix.DEFAULT_ARTIST_SPACING,
    )


def summarize(
    records: Sequence[dict[str, object]],
    active_tracks: int,
    reference_modes: dict[tuple[str, int], dict[str, object]],
) -> dict[str, object]:
    by_value: dict[float, dict[int, dict[str, object]]] = {value: {} for value in LAMBDAS}
    for record in records:
        by_value[float(record["mmr_lambda"])][int(record["seed"]["track_id"])] = record

    points = []
    for value in LAMBDAS:
        rows = list(by_value[value].values())
        points.append(
            {
                "mmr_lambda": value,
                "seed_count": len(rows),
                "mean_seed_cosine": numeric(
                    float(row["metrics"]["mean_seed_cosine"]) for row in rows
                ),
                "median_selected_seed_rank": numeric(
                    float(row["metrics"]["median_seed_rank"]) for row in rows
                ),
                "farthest_selected_seed_rank": numeric(
                    float(row["metrics"]["maximum_seed_rank"]) for row in rows
                ),
                "median_selected_top_library_percent": numeric(
                    100.0 * float(row["metrics"]["median_seed_rank"]) / active_tracks
                    for row in rows
                ),
                "mean_pairwise_cosine": numeric(
                    float(row["metrics"]["mean_pairwise_cosine"]) for row in rows
                ),
                "mean_adjacent_cosine": numeric(
                    float(row["metrics"]["mean_adjacent_cosine"]) for row in rows
                ),
                "unique_artist_credits": numeric(
                    float(row["metrics"]["unique_artist_credits"]) for row in rows
                ),
                "selection_ms": numeric(float(row["timing_ms"]["selection"]) for row in rows),
            }
        )

    adjacent = []
    for left, right in zip(LAMBDAS, LAMBDAS[1:]):
        per_seed = []
        for seed_id in SEEDS:
            left_ids = by_value[left][seed_id]["track_ids"]
            right_ids = by_value[right][seed_id]["track_ids"]
            per_seed.append(
                {
                    "seed_track_id": seed_id,
                    **queue_change(left_ids, right_ids),
                }
            )
        adjacent.append(
            {
                "from": left,
                "to": right,
                "set_jaccard": numeric(float(row["set_jaccard"]) for row in per_seed),
                "top10_set_churn": numeric(
                    float(row["top10_set_churn"]) for row in per_seed
                ),
                "same_position_fraction": numeric(
                    float(row["same_position_fraction"]) for row in per_seed
                ),
                "per_seed": per_seed,
            }
        )

    all_pairwise = []
    for left, right in combinations(LAMBDAS, 2):
        comparisons = [
            queue_change(
                by_value[left][seed_id]["track_ids"],
                by_value[right][seed_id]["track_ids"],
            )
            for seed_id in SEEDS
        ]
        all_pairwise.append(
            {
                "from": left,
                "to": right,
                "set_jaccard": numeric(
                    float(row["set_jaccard"]) for row in comparisons
                ),
                "top10_set_churn": numeric(
                    float(row["top10_set_churn"]) for row in comparisons
                ),
            }
        )

    cross_mode = []
    for value in LAMBDAS:
        for mode in ("closest", "dpp"):
            comparisons = []
            for seed_id in SEEDS:
                comparisons.append(
                    queue_change(
                        by_value[value][seed_id]["track_ids"],
                        reference_modes[(mode, seed_id)]["track_ids"],
                    )
                )
            cross_mode.append(
                {
                    "mmr_lambda": value,
                    "reference_mode": mode,
                    "set_jaccard": numeric(
                        float(row["set_jaccard"]) for row in comparisons
                    ),
                    "top10_set_churn": numeric(
                        float(row["top10_set_churn"]) for row in comparisons
                    ),
                }
            )

    return {
        "points": points,
        "adjacent": adjacent,
        "all_pairwise": all_pairwise,
        "cross_mode_distinctness": cross_mode,
        "determinism": {
            "repeat_count_per_case": 2,
            "all_repeat_projections_exact": all(
                bool(record["determinism"]["repeat_projection_exact"])
                for record in records
            ),
            "all_frozen_endpoint_fingerprints_exact": all(
                bool(record["determinism"]["frozen_endpoint_fingerprint_exact"])
                for record in records
                if record["mmr_lambda"] in (0.4, 0.6)
            ),
        },
    }


def write_named_queues(path: Path, records: Sequence[dict[str, object]], active_tracks: int) -> None:
    keyed = {
        (float(record["mmr_lambda"]), int(record["seed"]["track_id"])): record
        for record in records
    }
    lines = [
        "# MMR midpoint named queues",
        "",
        f"Frozen active domain: {active_tracks:,} tracks. Reach: 2%. Queue: 30.",
        "",
    ]
    for seed_id in NAMED_SEEDS:
        seed = keyed[(LAMBDAS[0], seed_id)]["seed"]
        lines.extend(
            [
                f"## {seed_id}: {seed.get('artist') or 'Unknown'} - {seed.get('title') or 'Unknown'}",
                "",
            ]
        )
        for value in LAMBDAS:
            record = keyed[(value, seed_id)]
            metrics = record["metrics"]
            lines.extend(
                [
                    f"### Relevance {value:g}",
                    "",
                    (
                        f"Middle seed rank #{int(metrics['median_seed_rank']):,}; "
                        f"farthest #{int(metrics['maximum_seed_rank']):,}; "
                        f"{int(metrics['unique_artist_credits'])} artist credits."
                    ),
                    "",
                ]
            )
            for track in record["tracks"]:
                lines.append(
                    f"{track['rank']}. {track.get('artist') or 'Unknown'} - "
                    f"{track.get('title') or 'Unknown'} "
                    f"(seed rank #{int(track['seed_rank']):,})"
                )
            lines.append("")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_checksums(output: Path) -> None:
    lines = []
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS":
            lines.append(f"{sha256_file(path)}  {path.name}")
    (output / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="ascii")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--active-catalog", type=Path, default=DEFAULT_ACTIVE_CATALOG)
    parser.add_argument("--parent-records", type=Path, default=DEFAULT_PARENT_RECORDS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    database_hash_before = sha256_file(args.database)
    catalog_hash = sha256_file(args.active_catalog)
    if database_hash_before != EXPECTED_DATABASE_SHA256:
        raise ValueError("frozen database hash mismatch")
    if catalog_hash != EXPECTED_ACTIVE_CATALOG_SHA256:
        raise ValueError("active catalog hash mismatch")

    full_library, loaded_hash = queue_eval.load_library(args.database)
    if loaded_hash != database_hash_before:
        raise ValueError("database changed while loading")
    active_ids = phone_oracle.active_track_ids(args.active_catalog)
    active_positions = np.flatnonzero(
        np.fromiter(
            (int(track_id) in active_ids for track_id in full_library.track_ids),
            dtype=np.bool_,
            count=full_library.count,
        )
    )
    if active_positions.size != EXPECTED_ACTIVE_TRACKS:
        raise ValueError("active domain is not the frozen 80,323-track catalog")
    library = matrix.subset_library(full_library, active_positions)
    del full_library

    id_to_index = {int(track_id): index for index, track_id in enumerate(library.track_ids)}
    contexts = {
        seed_id: matrix.seed_context(library, id_to_index[seed_id]) for seed_id in SEEDS
    }
    frozen, reference_modes = load_parent_records(args.parent_records)

    records = []
    for value in LAMBDAS:
        selector = config(value)
        for seed_id in SEEDS:
            first = matrix.run_config(
                library, None, None, contexts[seed_id], selector, {}
            )
            second = matrix.run_config(
                library, None, None, contexts[seed_id], selector, {}
            )
            repeat_exact = deterministic_projection(first) == deterministic_projection(second)
            if not repeat_exact:
                raise AssertionError(f"non-deterministic MMR result at {value=} {seed_id=}")
            frozen_exact = None
            if value in (0.4, 0.6):
                frozen_exact = (
                    first["queue_fingerprint"]
                    == frozen[(value, seed_id)]["queue_fingerprint"]
                )
                if not frozen_exact:
                    raise AssertionError(f"frozen endpoint mismatch at {value=} {seed_id=}")
            records.append(
                {
                    "case_id": f"mmr_midpoint_{str(value).replace('.', 'p')}",
                    "mmr_lambda": value,
                    "seed": queue_eval.track_summary(library, contexts[seed_id].seed_index),
                    **first,
                    "determinism": {
                        "repeat_projection_exact": repeat_exact,
                        "frozen_endpoint_fingerprint_exact": frozen_exact,
                    },
                }
            )
            print(f"lambda={value:g} seed={seed_id}", file=sys.stderr)

    summary = summarize(records, library.count, reference_modes)
    atomic_jsonl(args.output / "records.jsonl", records)
    atomic_json(args.output / "summary.json", summary)
    write_named_queues(args.output / "named-queues.md", records, library.count)

    database_hash_after = sha256_file(args.database)
    if database_hash_after != database_hash_before:
        raise AssertionError("frozen database changed during evaluation")
    manifest = {
        "experiment_version": "v2-mmr-midpoint-v1",
        "created_date": "2026-07-15",
        "inputs": {
            "database": str(args.database.resolve()),
            "database_sha256_before": database_hash_before,
            "database_sha256_after": database_hash_after,
            "active_catalog": str(args.active_catalog.resolve()),
            "active_catalog_sha256": catalog_hash,
            "parent_records": str(args.parent_records.resolve()),
            "parent_records_sha256": sha256_file(args.parent_records),
            "evaluator_sha256": sha256_file(Path(__file__)),
        },
        "domain": {
            "active_tracks": library.count,
            "dimensions": library.dim,
            "seed_track_ids": SEEDS,
        },
        "scope": {
            "mmr_lambda": LAMBDAS,
            "reach": REACH,
            "queue_size": QUEUE_SIZE,
            "artist_limits": True,
            "max_per_artist": matrix.DEFAULT_MAX_PER_ARTIST,
            "artist_spacing": matrix.DEFAULT_ARTIST_SPACING,
            "repeat_count_per_case": 2,
        },
        "runtime_seconds": time.perf_counter() - started,
        "host": {
            "platform": platform.platform(),
            "python": sys.version,
            "numpy": np.__version__,
            "openblas_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
            "omp_threads": os.environ.get("OMP_NUM_THREADS"),
        },
        "metadata_policy": (
            "Metadata never enters relevance or diversity. It is used only for the "
            "explicit artist-credit eligibility rule and human-readable evidence."
        ),
    }
    atomic_json(args.output / "manifest.json", manifest)
    write_checksums(args.output)
    print(f"Complete: {len(records)} records at {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
