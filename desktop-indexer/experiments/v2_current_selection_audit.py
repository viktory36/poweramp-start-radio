#!/usr/bin/env python3
"""Measure the current V2 single-seed selector surface on the frozen phone library.

This evaluator is deliberately read-only with respect to the embedding database. It reuses the
phone-parity-validated reference implementations, checkpoints every completed case, and focuses
on evidence missing from the earlier matrix: current V2 defaults, the complete exposed reach
range for current drift settings, certified full-domain DPP at every exposed exponent, and the
explicit Uniform Shuffle seed control.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import compare_device_feature_acceptance as phone_oracle
import v2_queue_eval as queue_eval
import v2_seed_conditioning_eval as seed_eval
import v2_selection_knob_matrix as matrix
import v2_selection_mode_eval as graph_eval


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATABASE = queue_eval.DEFAULT_DB
DEFAULT_ACTIVE_CATALOG = (
    REPO_ROOT
    / "discovery"
    / "device-acceptance"
    / "20260714T-realistic-text-battery"
    / "active-catalog.tsv"
)
DEFAULT_PHONE_REPORT = (
    REPO_ROOT
    / "discovery"
    / "device-acceptance"
    / "20260714T-active-domain-real-cohort"
    / "report.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "discovery" / "selection-mode-audit" / "20260715-current-v2"
)

EXPECTED_DATABASE_SHA256 = queue_eval.EXPECTED_DB_SHA256
EXPECTED_ACTIVE_CATALOG_SHA256 = matrix.EXPECTED_CATALOG_SHA256
EXPECTED_ACTIVE_TRACKS = matrix.EXPECTED_ACTIVE_TRACKS

ALL_SEEDS = (80437, 42335, 38327, 33821, 17399, 5987, 35389, 79210, 16325, 31799, 75735)
DRIFT_SEEDS = (80437, 33821, 17399, 75735)

QUEUE_SIZE = 30
MMR_RELEVANCE = 0.4
REACH = 0.02
DPP_EXPONENT = 1.0
GRAPH_STOP = 0.5
ANCHOR_STRENGTH = 0.5
ANCHOR_HALF_LIFE = 7.0
MOMENTUM = 0.9
SHUFFLE_SEED = 0x5053525632534855

MMR_OPTIONS = (0.0, 0.25, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0)
REACH_OPTIONS = (0.0025, 0.005, 0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 1.0)
DPP_FIXED_REACH_OPTIONS = REACH_OPTIONS[:-1]
DPP_EXPONENT_OPTIONS = (0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0)
GRAPH_STOP_OPTIONS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90)
ANCHOR_OPTIONS = (0.0, 0.25, 0.5, 0.75, 0.85, 1.0)
MOMENTUM_OPTIONS = (0.25, 0.5, 0.75, 0.9, 0.95, 1.0)
FADE_TIMING_OPTIONS = (1.0, 3.0, 7.0, 10.0, 15.0, 30.0)
SCHEDULE_OPTIONS = ("NONE", "LINEAR", "EXPONENTIAL", "STEP")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def append_jsonl(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
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


def atomic_jsonl(path: Path, values: Sequence[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    array = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
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


def queue_change(left: Sequence[int], right: Sequence[int]) -> dict[str, float | bool]:
    top = min(10, len(left), len(right))
    same_positions = min(len(left), len(right))
    return {
        "exact_order_no_op": list(left) == list(right),
        "set_jaccard": set_jaccard(left, right),
        "top10_set_churn": (
            1.0 - len(set(left[:top]) & set(right[:top])) / top if top else 0.0
        ),
        "same_position_fraction": (
            sum(a == b for a, b in zip(left, right)) / same_positions
            if same_positions
            else 1.0
        ),
    }


def slug(value: object) -> str:
    if isinstance(value, float):
        rendered = f"{value:.8g}"
    else:
        rendered = str(value).lower()
    return rendered.replace("-", "neg").replace(".", "p")


def base_configs() -> dict[str, matrix.SelectorConfig]:
    return {
        "closest": matrix.SelectorConfig(mode="closest", queue_size=QUEUE_SIZE),
        "mmr": matrix.SelectorConfig(
            mode="mmr",
            queue_size=QUEUE_SIZE,
            reach=REACH,
            mmr_lambda=MMR_RELEVANCE,
        ),
        "dpp": matrix.SelectorConfig(
            mode="dpp",
            queue_size=QUEUE_SIZE,
            reach=REACH,
            dpp_exponent=DPP_EXPONENT,
            dpp_uses_certified_full_domain=True,
        ),
        "graph": matrix.SelectorConfig(
            mode="graph", queue_size=QUEUE_SIZE, graph_alpha=GRAPH_STOP
        ),
        "drift_seed": matrix.SelectorConfig(
            mode="drift",
            queue_size=QUEUE_SIZE,
            reach=REACH,
            mmr_lambda=MMR_RELEVANCE,
            drift_mode="SEED_INTERPOLATION",
            anchor_schedule="EXPONENTIAL",
            anchor_strength=ANCHOR_STRENGTH,
            anchor_half_life=ANCHOR_HALF_LIFE,
        ),
        "drift_momentum": matrix.SelectorConfig(
            mode="drift",
            queue_size=QUEUE_SIZE,
            reach=REACH,
            mmr_lambda=MMR_RELEVANCE,
            drift_mode="MOMENTUM",
            momentum_beta=MOMENTUM,
        ),
    }


def build_cases() -> dict[str, tuple[matrix.SelectorConfig, tuple[int, ...], str, object]]:
    base = base_configs()
    cases: dict[str, tuple[matrix.SelectorConfig, tuple[int, ...], str, object]] = {}

    def add(
        case_id: str,
        config: matrix.SelectorConfig,
        seeds: tuple[int, ...],
        sweep: str,
        value: object,
    ) -> None:
        cases[case_id] = (config, seeds, sweep, value)

    for mode, config in base.items():
        add(f"current_{mode}", config, ALL_SEEDS, "current_modes", mode)

    for value in MMR_OPTIONS:
        add(
            f"mmr_relevance_{slug(value)}",
            replace(base["mmr"], mmr_lambda=value),
            ALL_SEEDS,
            "mmr_relevance",
            value,
        )
    for value in REACH_OPTIONS:
        add(
            f"mmr_reach_{slug(value)}",
            replace(base["mmr"], reach=value),
            ALL_SEEDS,
            "mmr_reach",
            value,
        )

    for value in DPP_EXPONENT_OPTIONS:
        add(
            f"dpp_full_exponent_{slug(value)}",
            replace(base["dpp"], dpp_exponent=value),
            ALL_SEEDS,
            "dpp_full_exponent",
            value,
        )
    for value in DPP_FIXED_REACH_OPTIONS:
        add(
            f"dpp_fixed_reach_{slug(value)}",
            replace(
                base["dpp"],
                dpp_uses_certified_full_domain=False,
                reach=value,
            ),
            ALL_SEEDS,
            "dpp_fixed_reach",
            value,
        )

    for value in GRAPH_STOP_OPTIONS:
        add(
            f"graph_stop_{slug(value)}",
            replace(base["graph"], graph_alpha=value),
            ALL_SEEDS,
            "graph_stop",
            value,
        )

    for value in ANCHOR_OPTIONS:
        add(
            f"drift_anchor_{slug(value)}",
            replace(base["drift_seed"], anchor_strength=value),
            DRIFT_SEEDS,
            "drift_anchor",
            value,
        )
    for value in SCHEDULE_OPTIONS:
        add(
            f"drift_schedule_{slug(value)}",
            replace(base["drift_seed"], anchor_schedule=value),
            DRIFT_SEEDS,
            "drift_schedule",
            value,
        )
    for schedule in ("LINEAR", "EXPONENTIAL", "STEP"):
        for value in FADE_TIMING_OPTIONS:
            if schedule == "STEP" and value > QUEUE_SIZE - 2:
                continue
            add(
                f"drift_{schedule.lower()}_timing_{slug(value)}",
                replace(
                    base["drift_seed"],
                    anchor_schedule=schedule,
                    anchor_half_life=value,
                ),
                DRIFT_SEEDS,
                f"drift_timing_{schedule.lower()}",
                value,
            )
    for value in MOMENTUM_OPTIONS:
        add(
            f"drift_momentum_{slug(value)}",
            replace(base["drift_momentum"], momentum_beta=value),
            DRIFT_SEEDS,
            "drift_momentum",
            value,
        )
    for mode in ("drift_seed", "drift_momentum"):
        for value in REACH_OPTIONS:
            add(
                f"{mode}_reach_{slug(value)}",
                replace(base[mode], reach=value),
                DRIFT_SEEDS,
                f"{mode}_reach",
                value,
            )

    for mode, config in base.items():
        add(
            f"{mode}_queue_100",
            replace(config, queue_size=100),
            DRIFT_SEEDS,
            f"{mode}_queue_prefix",
            100,
        )
    return cases


def load_completed(path: Path) -> tuple[list[dict[str, object]], set[tuple[str, int]]]:
    records: list[dict[str, object]] = []
    completed: set[tuple[str, int]] = set()
    if not path.exists():
        return records, completed
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        key = (str(record["case_id"]), int(record["seed"]["track_id"]))
        if key in completed:
            raise ValueError(f"duplicate checkpoint record: {key}")
        completed.add(key)
        records.append(record)
    return records, completed


def shuffle_result(
    library: queue_eval.Library,
    context: matrix.SeedContext,
    active: np.ndarray,
    activation_binding_id: str,
    shuffle_seed: int,
    queue_size: int,
) -> dict[str, object]:
    started = time.perf_counter()
    permutation = seed_eval.shuffle_permutation(
        library=library,
        seed_index=context.seed_index,
        active=active,
        shuffle_seed=shuffle_seed,
        activation_binding_id=activation_binding_id,
    )
    selected = seed_eval.select_shuffle(
        library=library,
        seed_index=context.seed_index,
        permutation=permutation,
        count=queue_size,
    )
    metrics = matrix.queue_metrics(
        library=library,
        context=context,
        selected=selected.selected,
        candidate_ranks=selected.candidate_ranks,
        config=matrix.SelectorConfig(mode="closest", queue_size=queue_size),
        extra={},
    )
    track_ids = [int(library.track_ids[index]) for index in selected.selected]
    return {
        "config": {
            "mode": "uniform_shuffle",
            "queue_size": queue_size,
            "shuffle_seed": shuffle_seed,
            "artist_limits": True,
            "max_per_artist": matrix.DEFAULT_MAX_PER_ARTIST,
            "artist_spacing": matrix.DEFAULT_ARTIST_SPACING,
        },
        "actual_candidate_count": int(active.sum()) - 1,
        "track_ids": track_ids,
        "candidate_ranks": [int(value) for value in selected.candidate_ranks],
        "selection_scores": [float(value) for value in selected.objective_values],
        "queue_fingerprint": matrix.stable_json_hash(track_ids),
        "metrics": metrics,
        "timing_ms": {
            "total_from_precomputed_seed_scan": (time.perf_counter() - started) * 1000.0
        },
        "tracks": [
            matrix.track_record(
                library,
                context,
                index,
                rank,
                candidate_rank,
                score,
            )
            for rank, (index, candidate_rank, score) in enumerate(
                zip(
                    selected.selected,
                    selected.candidate_ranks,
                    selected.objective_values,
                ),
                start=1,
            )
        ],
    }


def summarize(records: Sequence[dict[str, object]]) -> dict[str, object]:
    by_case: dict[str, list[dict[str, object]]] = defaultdict(list)
    by_sweep: dict[str, dict[object, dict[int, dict[str, object]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for record in records:
        by_case[str(record["case_id"])].append(record)
        by_sweep[str(record["sweep_id"])][record["sweep_value"]][
            int(record["seed"]["track_id"])
        ] = record

    case_summary: dict[str, object] = {}
    for case_id, rows in sorted(by_case.items()):
        metrics = [row["metrics"] for row in rows]
        case_summary[case_id] = {
            "seed_count": len(rows),
            "config": rows[0]["config"],
            "returned": numeric(float(item["returned"]) for item in metrics),
            "mean_seed_cosine": numeric(
                float(item["mean_seed_cosine"])
                for item in metrics
                if item.get("mean_seed_cosine") is not None
            ),
            "mean_pairwise_cosine": numeric(
                float(item["mean_pairwise_cosine"])
                for item in metrics
                if item.get("mean_pairwise_cosine") is not None
            ),
            "mean_adjacent_cosine": numeric(
                float(item["mean_adjacent_cosine"])
                for item in metrics
                if item.get("mean_adjacent_cosine") is not None
            ),
            "median_seed_rank": numeric(
                float(item["median_seed_rank"])
                for item in metrics
                if item.get("median_seed_rank") is not None
            ),
            "unique_artist_credits": numeric(
                float(item["unique_artist_credits"])
                for item in metrics
                if item.get("unique_artist_credits") is not None
            ),
            "copy_proxy_waste_slots": numeric(
                float(item["copy_proxy_waste_slots"])
                for item in metrics
                if item.get("copy_proxy_waste_slots") is not None
            ),
            "selection_ms": numeric(
                float(row["timing_ms"].get("selection", 0.0)) for row in rows
            ),
            "total_ms": numeric(
                float(row["timing_ms"].get("total_from_precomputed_seed_scan", 0.0))
                for row in rows
            ),
        }

    sweep_summary: dict[str, object] = {}
    for sweep_id, values in sorted(by_sweep.items()):
        ordered_values = list(values)
        points = []
        for value in ordered_values:
            rows = list(values[value].values())
            points.append(
                {
                    "value": value,
                    "case_id": rows[0]["case_id"],
                    "seed_count": len(rows),
                    "mean_seed_cosine": numeric(
                        float(row["metrics"]["mean_seed_cosine"])
                        for row in rows
                        if row["metrics"].get("mean_seed_cosine") is not None
                    ),
                    "mean_pairwise_cosine": numeric(
                        float(row["metrics"]["mean_pairwise_cosine"])
                        for row in rows
                        if row["metrics"].get("mean_pairwise_cosine") is not None
                    ),
                    "median_seed_rank": numeric(
                        float(row["metrics"]["median_seed_rank"])
                        for row in rows
                        if row["metrics"].get("median_seed_rank") is not None
                    ),
                }
            )
        adjacent = []
        for left, right in zip(ordered_values, ordered_values[1:]):
            common = sorted(set(values[left]) & set(values[right]))
            changes = [
                queue_change(
                    values[left][seed]["track_ids"],
                    values[right][seed]["track_ids"],
                )
                for seed in common
            ]
            adjacent.append(
                {
                    "from": left,
                    "to": right,
                    "seed_count": len(common),
                    "exact_no_op_seeds": sum(
                        bool(change["exact_order_no_op"]) for change in changes
                    ),
                    "set_jaccard": numeric(
                        float(change["set_jaccard"]) for change in changes
                    ),
                    "top10_set_churn": numeric(
                        float(change["top10_set_churn"]) for change in changes
                    ),
                    "same_position_fraction": numeric(
                        float(change["same_position_fraction"]) for change in changes
                    ),
                }
            )
        sweep_summary[sweep_id] = {"points": points, "adjacent": adjacent}

    mode_pairs: dict[str, object] = {}
    current_by_seed: dict[int, dict[str, dict[str, object]]] = defaultdict(dict)
    for case_id, rows in by_case.items():
        if not case_id.startswith("current_"):
            continue
        mode = case_id.removeprefix("current_")
        for row in rows:
            current_by_seed[int(row["seed"]["track_id"])][mode] = row
    modes = sorted({mode for values in current_by_seed.values() for mode in values})
    for left_index, left in enumerate(modes):
        for right in modes[left_index + 1 :]:
            changes = [
                queue_change(values[left]["track_ids"], values[right]["track_ids"])
                for values in current_by_seed.values()
                if left in values and right in values
            ]
            mode_pairs[f"{left}__{right}"] = {
                "seed_count": len(changes),
                "set_jaccard": numeric(float(item["set_jaccard"]) for item in changes),
                "top10_set_churn": numeric(
                    float(item["top10_set_churn"]) for item in changes
                ),
            }

    prefix_checks = []
    for case_id, rows in by_case.items():
        if not case_id.endswith("_queue_100"):
            continue
        mode = case_id[: -len("_queue_100")]
        baseline = {
            int(row["seed"]["track_id"]): row
            for row in by_case.get(f"current_{mode}", [])
        }
        for row in rows:
            seed_id = int(row["seed"]["track_id"])
            if seed_id not in baseline:
                continue
            prefix_checks.append(
                {
                    "mode": mode,
                    "seed_track_id": seed_id,
                    "first_30_exact": row["track_ids"][:30] == baseline[seed_id]["track_ids"],
                }
            )

    return {
        "cases": case_summary,
        "sweeps": sweep_summary,
        "current_mode_distinctness": mode_pairs,
        "queue_prefix": {
            "comparison_count": len(prefix_checks),
            "all_first_30_exact": all(row["first_30_exact"] for row in prefix_checks),
            "comparisons": prefix_checks,
        },
    }


def shuffle_seed_audit(
    library: queue_eval.Library,
    context: matrix.SeedContext,
    active: np.ndarray,
    activation_binding_id: str,
    count: int = 32,
) -> dict[str, object]:
    seeds = [SHUFFLE_SEED]
    while len(seeds) < count:
        next_seed = phone_oracle.mix64(seeds[-1] + phone_oracle.GOLDEN_GAMMA)
        seeds.append(1 if next_seed == 0 else next_seed)
    orders: list[list[int]] = []
    for seed in seeds:
        result = shuffle_result(
            library,
            context,
            active,
            activation_binding_id,
            seed,
            100,
        )
        orders.append(result["track_ids"])
    repeat = shuffle_result(
        library,
        context,
        active,
        activation_binding_id,
        SHUFFLE_SEED,
        100,
    )["track_ids"]
    prefix_30 = shuffle_result(
        library,
        context,
        active,
        activation_binding_id,
        SHUFFLE_SEED,
        30,
    )["track_ids"]
    pair_overlaps = []
    for left_index, left in enumerate(orders):
        for right in orders[left_index + 1 :]:
            pair_overlaps.append(len(set(left) & set(right)))
    all_ids = [track_id for order in orders for track_id in order]
    return {
        "seed_count": len(seeds),
        "queue_size": 100,
        "same_seed_exact_repeat": orders[0] == repeat,
        "queue_30_is_prefix_of_queue_100": orders[0][:30] == prefix_30,
        "unique_orders": len({tuple(order) for order in orders}),
        "unique_tracks_across_orders": len(set(all_ids)),
        "pairwise_top100_intersection": numeric(float(value) for value in pair_overlaps),
        "expected_pairwise_intersection_under_independent_uniform_orders": 10000
        / (int(active.sum()) - 1),
        "maximum_track_exposure": max(
            (all_ids.count(track_id) for track_id in set(all_ids)), default=0
        ),
        "first_seed": seeds[0],
        "last_seed": seeds[-1],
        "first_order_fingerprint": matrix.stable_json_hash(orders[0]),
    }


def write_qualitative(path: Path, records: Sequence[dict[str, object]]) -> None:
    wanted = {
        "current_closest",
        "current_mmr",
        "current_dpp",
        "current_graph",
        "current_drift_seed",
        "current_drift_momentum",
        "current_uniform_shuffle",
    }
    rows = [record for record in records if record["case_id"] in wanted]
    by_seed: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_seed[int(row["seed"]["track_id"])].append(row)
    lines = [
        "# Current V2 Single-Seed Qualitative Packet",
        "",
        "Membership uses only CLaMP3 embeddings, the bound audio graph, or the explicit shuffle seed.",
        "Artist/title/path labels are emitted only for human inspection.",
        "",
    ]
    for seed_id, seed_rows in sorted(by_seed.items()):
        seed = seed_rows[0]["seed"]
        lines.extend(
            [
                f"## {seed_id}: {seed.get('artist') or 'Unknown'} - {seed.get('title') or 'Unknown'}",
                "",
            ]
        )
        for row in sorted(seed_rows, key=lambda value: str(value["case_id"])):
            lines.extend([f"### {row['case_id']}", ""])
            for track in row["tracks"]:
                lines.append(
                    f"{track['rank']}. {track.get('artist') or 'Unknown'} - "
                    f"{track.get('title') or 'Unknown'} | seed cosine "
                    f"{float(track['similarity_to_seed']):.4f} | seed rank "
                    f"{track['seed_rank']} | `{track.get('file_path') or ''}`"
                )
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--active-catalog", type=Path, default=DEFAULT_ACTIVE_CATALOG)
    parser.add_argument("--phone-report", type=Path, default=DEFAULT_PHONE_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if output.exists() and args.force:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "records.jsonl"

    database_hash_before = sha256_file(args.database)
    catalog_hash = sha256_file(args.active_catalog)
    if database_hash_before != EXPECTED_DATABASE_SHA256:
        raise ValueError("frozen database hash mismatch")
    if catalog_hash != EXPECTED_ACTIVE_CATALOG_SHA256:
        raise ValueError("active catalog hash mismatch")

    started = time.perf_counter()
    full_library, loaded_hash = queue_eval.load_library(args.database)
    if loaded_hash != database_hash_before:
        raise ValueError("database changed while loading")
    active_ids = phone_oracle.active_track_ids(args.active_catalog)
    active_mask = np.fromiter(
        (int(track_id) in active_ids for track_id in full_library.track_ids),
        dtype=np.bool_,
        count=full_library.count,
    )
    if int(active_mask.sum()) != EXPECTED_ACTIVE_TRACKS:
        raise ValueError("active domain is not the frozen 80,323-track catalog")
    full_graph = graph_eval.parse_graph(args.database)
    active_graph, active_positions, _old_to_active, graph_repair = (
        phone_oracle.build_active_graph(full_library, full_graph, active_mask)
    )
    library = matrix.subset_library(full_library, active_positions)
    del full_library, full_graph, active_mask
    graph_transition = graph_eval.build_edge_transition(active_graph, weighted=False)
    id_to_index = {int(track_id): index for index, track_id in enumerate(library.track_ids)}
    contexts = {
        seed_id: matrix.seed_context(library, id_to_index[seed_id]) for seed_id in ALL_SEEDS
    }
    active = np.ones(library.count, dtype=np.bool_)
    phone_report = json.loads(args.phone_report.read_text(encoding="utf-8"))
    activation_binding_id = str(phone_report["generation"]["activationBindingId"])

    cases = build_cases()
    records, completed = load_completed(checkpoint)
    valid_case_ids = set(cases) | {"current_uniform_shuffle"}
    filtered_records = [record for record in records if record["case_id"] in valid_case_ids]
    if len(filtered_records) != len(records):
        records = filtered_records
        completed = {
            (str(record["case_id"]), int(record["seed"]["track_id"]))
            for record in records
        }
        atomic_jsonl(checkpoint, records)
    graph_cache: dict[tuple[int, float], matrix.GraphDistribution] = {}
    result_cache: dict[tuple[int, matrix.SelectorConfig], dict[str, object]] = {}
    total_requested = sum(len(seeds) for _config, seeds, _sweep, _value in cases.values())
    completed_count = len(completed)
    for case_id, (config, seeds, sweep, value) in cases.items():
        for seed_id in seeds:
            key = (case_id, seed_id)
            if key in completed:
                continue
            cache_key = (seed_id, config)
            result = result_cache.get(cache_key)
            if result is None:
                result = matrix.run_config(
                    library,
                    active_graph,
                    graph_transition,
                    contexts[seed_id],
                    config,
                    graph_cache,
                )
                result_cache[cache_key] = result
            record = {
                "case_id": case_id,
                "sweep_id": sweep,
                "sweep_value": value,
                "seed": queue_eval.track_summary(library, contexts[seed_id].seed_index),
                **result,
            }
            append_jsonl(checkpoint, record)
            records.append(record)
            completed.add(key)
            completed_count += 1
            print(
                f"{completed_count}/{total_requested} {case_id} seed={seed_id}",
                file=sys.stderr,
            )

    for seed_id in ALL_SEEDS:
        case_id = "current_uniform_shuffle"
        key = (case_id, seed_id)
        if key in completed:
            continue
        result = shuffle_result(
            library,
            contexts[seed_id],
            active,
            activation_binding_id,
            SHUFFLE_SEED,
            QUEUE_SIZE,
        )
        record = {
            "case_id": case_id,
            "sweep_id": "current_modes",
            "sweep_value": "uniform_shuffle",
            "seed": queue_eval.track_summary(library, contexts[seed_id].seed_index),
            **result,
        }
        append_jsonl(checkpoint, record)
        records.append(record)
        completed.add(key)

    shuffle_audit = shuffle_seed_audit(
        library,
        contexts[ALL_SEEDS[0]],
        active,
        activation_binding_id,
    )
    summary = summarize(records)
    summary["uniform_shuffle_seed"] = shuffle_audit
    atomic_json(output / "summary.json", summary)
    write_qualitative(output / "current-default-lists.md", records)

    database_hash_after = sha256_file(args.database)
    if database_hash_after != database_hash_before:
        raise AssertionError("frozen database changed during evaluation")
    manifest = {
        "experiment_version": "current-v2-single-seed-selection-v1",
        "created_date": "2026-07-15",
        "inputs": {
            "database": str(args.database.resolve()),
            "database_sha256_before": database_hash_before,
            "database_sha256_after": database_hash_after,
            "active_catalog": str(args.active_catalog.resolve()),
            "active_catalog_sha256": catalog_hash,
            "phone_report": str(args.phone_report.resolve()),
            "phone_report_sha256": sha256_file(args.phone_report),
            "evaluator_sha256": sha256_file(Path(__file__)),
        },
        "domain": {
            "active_tracks": library.count,
            "dimensions": library.dim,
            "all_seed_track_ids": ALL_SEEDS,
            "drift_seed_track_ids": DRIFT_SEEDS,
            "active_graph_repair": graph_repair,
            "activation_binding_id": activation_binding_id,
        },
        "current_defaults": {
            "queue_size": QUEUE_SIZE,
            "mmr_relevance": MMR_RELEVANCE,
            "reach": REACH,
            "dpp_full_domain": True,
            "dpp_exponent": DPP_EXPONENT,
            "graph_stop": GRAPH_STOP,
            "anchor_strength": ANCHOR_STRENGTH,
            "anchor_schedule": "EXPONENTIAL",
            "anchor_half_life": ANCHOR_HALF_LIFE,
            "momentum": MOMENTUM,
            "shuffle_seed": SHUFFLE_SEED,
            "artist_limits": True,
            "max_per_artist": matrix.DEFAULT_MAX_PER_ARTIST,
            "artist_spacing": matrix.DEFAULT_ARTIST_SPACING,
        },
        "scope": {
            "record_count": len(records),
            "case_count": len(cases) + 1,
            "mmr_options": MMR_OPTIONS,
            "reach_options": REACH_OPTIONS,
            "dpp_fixed_reach_options": DPP_FIXED_REACH_OPTIONS,
            "dpp_exponent_options": DPP_EXPONENT_OPTIONS,
            "graph_stop_options": GRAPH_STOP_OPTIONS,
            "anchor_options": ANCHOR_OPTIONS,
            "momentum_options": MOMENTUM_OPTIONS,
            "fade_timing_options": FADE_TIMING_OPTIONS,
            "schedule_options": SCHEDULE_OPTIONS,
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
            "Metadata never enters relevance, diversity, graph transition, or shuffle priority. "
            "It is used only for the explicit artist-credit eligibility rule and human-readable evidence."
        ),
    }
    atomic_json(output / "manifest.json", manifest)
    hashes = []
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS":
            hashes.append(f"{sha256_file(path)}  {path.name}")
    (output / "SHA256SUMS").write_text("\n".join(hashes) + "\n", encoding="ascii")
    print(f"Complete: {len(records)} records at {output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
