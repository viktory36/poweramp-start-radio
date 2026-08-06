#!/usr/bin/env python3
"""Measure whether MMR and DPP benefit from neighborhoods wider than five percent.

The evaluator reuses the canonical active-domain selector implementations from
``v2_selection_knob_matrix``. Metadata is emitted only for inspection and for the
existing explicit artist-credit constraint; it never enters a similarity or diversity
score. Results are checkpointed one complete queue at a time so a long full-library
run can resume without recomputing finished configurations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
import v2_selection_knob_matrix as matrix


REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_DATABASE = queue_eval.DEFAULT_DB
DEFAULT_ACTIVE_CATALOG = matrix.DEFAULT_ACTIVE_CATALOG
DEFAULT_PHONE_REPORT = matrix.DEFAULT_PHONE_REPORT
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "desktop-indexer"
    / "audit_raw_data"
    / "v2-discovery"
    / "extended-selector-pool-2026-07-15"
)

EXPERIMENT_VERSION = "extended-selector-pool-active-domain-v1"
REACH_VALUES = (0.02, 0.05, 0.10, 0.25, 0.50, 1.00)
MMR_LAMBDAS = (0.4, 0.6, 0.8, matrix.DEFAULT_LAMBDA)
DPP_EXPONENTS = (0.5, 1.0, 2.0)
QUEUE_SIZE = 30
LISTENING_SEEDS = (80437, 33821, 17399)  # Bonobo, Kailash Kher, L. Subramaniam.


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        default=queue_eval.json_numpy_scalar,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


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


def config_id(config: matrix.SelectorConfig) -> str:
    control = config.mmr_lambda if config.mode == "mmr" else config.dpp_exponent
    return f"{config.mode}-{control:.8g}-reach-{config.reach:.8g}"


def record_key(seed_id: int, config: matrix.SelectorConfig) -> str:
    return f"{seed_id}:{config_id(config)}"


def configurations() -> tuple[matrix.SelectorConfig, ...]:
    base_mmr = matrix.SelectorConfig(mode="mmr", queue_size=QUEUE_SIZE)
    base_dpp = matrix.SelectorConfig(mode="dpp", queue_size=QUEUE_SIZE)
    values: list[matrix.SelectorConfig] = []
    for lambda_ in MMR_LAMBDAS:
        for reach in REACH_VALUES:
            values.append(replace(base_mmr, mmr_lambda=lambda_, reach=reach))
    for exponent in DPP_EXPONENTS:
        for reach in REACH_VALUES:
            values.append(replace(base_dpp, dpp_exponent=exponent, reach=reach))
    return tuple(values)


def load_completed(path: Path) -> dict[str, dict[str, object]]:
    completed: dict[str, dict[str, object]] = {}
    if not path.is_file():
        return completed
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid checkpoint line {line_number}: {error}") from error
            key = str(record["record_key"])
            if key in completed:
                raise ValueError(f"duplicate checkpoint record: {key}")
            completed[key] = record
    return completed


def load_active_library(
    database: Path,
    active_catalog: Path,
) -> tuple[queue_eval.Library, str]:
    database_hash = sha256_file(database)
    if database_hash != queue_eval.EXPECTED_DB_SHA256:
        raise ValueError("frozen database hash mismatch")
    if sha256_file(active_catalog) != matrix.EXPECTED_CATALOG_SHA256:
        raise ValueError("active catalog hash mismatch")

    active_ids = phone_oracle.active_track_ids(active_catalog)
    if len(active_ids) != matrix.EXPECTED_ACTIVE_TRACKS:
        raise ValueError("active catalog count mismatch")
    full_library, loaded_hash = queue_eval.load_library(database)
    if loaded_hash != database_hash:
        raise ValueError("database changed while loading")
    active_positions = np.fromiter(
        (
            index
            for index, track_id in enumerate(full_library.track_ids)
            if int(track_id) in active_ids
        ),
        dtype=np.int64,
        count=len(active_ids),
    )
    if active_positions.size != matrix.EXPECTED_ACTIVE_TRACKS:
        raise ValueError("database/catalog active intersection mismatch")
    return matrix.subset_library(full_library, active_positions), database_hash


def numeric_summary(values: Iterable[float]) -> dict[str, float | None]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return {"mean": None, "min": None, "median": None, "max": None}
    return {
        "mean": float(np.mean(finite)),
        "min": float(np.min(finite)),
        "median": float(np.median(finite)),
        "max": float(np.max(finite)),
    }


def summarize(records: Sequence[dict[str, object]]) -> dict[str, object]:
    grouped: dict[tuple[str, float], dict[float, list[dict[str, object]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    by_key = {str(record["record_key"]): record for record in records}
    for record in records:
        config = record["config"]
        mode = str(config["mode"])
        control = float(
            config["mmr_lambda"] if mode == "mmr" else config["dpp_exponent"]
        )
        grouped[(mode, control)][float(config["reach"])].append(record)

    groups: dict[str, object] = {}
    for (mode, control), by_reach in sorted(grouped.items()):
        group_id = f"{mode}:{control:.8g}"
        reach_rows: list[dict[str, object]] = []
        adjacent: list[dict[str, object]] = []
        available_reaches = tuple(reach for reach in REACH_VALUES if reach in by_reach)
        for reach in available_reaches:
            items = by_reach[reach]
            reach_rows.append(
                {
                    "reach": reach,
                    "candidate_counts": sorted(
                        {int(item["actual_candidate_count"]) for item in items}
                    ),
                    "returned": numeric_summary(
                        float(item["metrics"]["returned"]) for item in items
                    ),
                    "mean_seed_cosine": numeric_summary(
                        float(item["metrics"]["mean_seed_cosine"]) for item in items
                    ),
                    "mean_pairwise_cosine": numeric_summary(
                        float(item["metrics"]["mean_pairwise_cosine"]) for item in items
                    ),
                    "median_seed_rank": numeric_summary(
                        float(item["metrics"]["median_seed_rank"]) for item in items
                    ),
                    "unique_artist_credits": numeric_summary(
                        float(item["metrics"]["unique_artist_credits"]) for item in items
                    ),
                    "copy_proxy_waste_slots": numeric_summary(
                        float(item["metrics"]["copy_proxy_waste_slots"]) for item in items
                    ),
                    "selection_ms": numeric_summary(
                        float(item["timing_ms"]["selection"]) for item in items
                    ),
                }
            )

        seed_ids = sorted(
            int(record["seed"]["track_id"])
            for record in by_reach[available_reaches[0]]
        )
        for left, right in zip(available_reaches, available_reaches[1:]):
            changes = []
            for seed_id in seed_ids:
                left_record = by_key[f"{seed_id}:{mode}-{control:.8g}-reach-{left:.8g}"]
                right_record = by_key[f"{seed_id}:{mode}-{control:.8g}-reach-{right:.8g}"]
                changes.append(
                    matrix.queue_change(left_record["track_ids"], right_record["track_ids"])
                )
            adjacent.append(
                {
                    "from": left,
                    "to": right,
                    "exact_no_op_seeds": sum(
                        bool(change["exact_order_no_op"]) for change in changes
                    ),
                    "seed_count": len(changes),
                    "set_jaccard": numeric_summary(
                        float(change["set_jaccard"]) for change in changes
                    ),
                    "top10_set_churn": numeric_summary(
                        float(change["top10_set_churn"]) for change in changes
                    ),
                    "same_rank_fraction": numeric_summary(
                        float(change["common_prefix_fraction"]) for change in changes
                    ),
                }
            )
        groups[group_id] = {
            "mode": mode,
            "control": control,
            "reaches": reach_rows,
            "adjacent": adjacent,
        }
    return {"groups": groups}


def selected_listening_configs() -> tuple[str, ...]:
    chosen = (
        ("mmr", 0.4, 0.02),
        ("mmr", 0.4, 0.10),
        ("mmr", 0.4, 1.00),
        ("mmr", 0.6, 0.02),
        ("mmr", 0.6, 0.10),
        ("mmr", 0.6, 1.00),
        ("mmr", matrix.DEFAULT_LAMBDA, 0.02),
        ("mmr", matrix.DEFAULT_LAMBDA, 1.00),
        ("dpp", 0.5, 0.02),
        ("dpp", 0.5, 0.10),
        ("dpp", 0.5, 1.00),
        ("dpp", 1.0, 0.02),
        ("dpp", 1.0, 0.10),
        ("dpp", 1.0, 1.00),
        ("dpp", 2.0, 0.02),
        ("dpp", 2.0, 1.00),
    )
    return tuple(f"{mode}-{control:.8g}-reach-{reach:.8g}" for mode, control, reach in chosen)


def write_listening_packet(
    packet_path: Path,
    key_path: Path,
    records: Sequence[dict[str, object]],
) -> None:
    selected = set(selected_listening_configs())
    available_seeds = {int(record["seed"]["track_id"]) for record in records}
    seeds = tuple(seed for seed in LISTENING_SEEDS if seed in available_seeds)
    configs = sorted(selected, key=stable_hash)
    labels = {config: f"P{position:02d}" for position, config in enumerate(configs, start=1)}
    lookup = {
        (int(record["seed"]["track_id"]), config_id(matrix.SelectorConfig(**record["config"]))): record
        for record in records
    }
    lines = [
        "# Extended Selector Pool Listening Packet",
        "",
        "Labels are opaque; the separate key contains exact selector and reach settings.",
        "Metadata below is for listening only and never entered the queue objective.",
        "",
    ]
    for seed_id in seeds:
        sample = next(record for record in records if int(record["seed"]["track_id"]) == seed_id)
        seed = sample["seed"]
        lines += [
            f"## {seed.get('artist') or 'Unknown'} - {seed.get('title') or 'Unknown'}",
            "",
        ]
        for config in configs:
            record = lookup[(seed_id, config)]
            lines += [f"### {labels[config]}", ""]
            for track in record["tracks"]:
                lines.append(
                    f"{track['rank']}. {track.get('artist') or 'Unknown'} - "
                    f"{track.get('title') or 'Unknown'} | `{track.get('file_path') or ''}`"
                )
            lines.append("")
    packet_path.write_text("\n".join(lines), encoding="utf-8")
    atomic_json(
        key_path,
        {
            "seed_track_ids": seeds,
            "labels": {
                labels[config]: {
                    "config_id": config,
                    "config": lookup[(seeds[0], config)]["config"] if seeds else None,
                }
                for config in configs
            },
        },
    )


def write_hashes(output: Path, inputs: dict[str, str]) -> None:
    lines = [f"{digest}  INPUT::{name}" for name, digest in sorted(inputs.items())]
    for path in sorted(output.iterdir()):
        if path.name == "SHA256SUMS" or not path.is_file():
            continue
        lines.append(f"{sha256_file(path)}  {path.name}")
    (output / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="ascii")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--active-catalog", type=Path, default=DEFAULT_ACTIVE_CATALOG)
    parser.add_argument("--phone-report", type=Path, default=DEFAULT_PHONE_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed-limit", type=int)
    parser.add_argument("--config-limit", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if args.force and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    input_hashes = {
        "database": sha256_file(args.database),
        "active_catalog": sha256_file(args.active_catalog),
        "phone_report": sha256_file(args.phone_report),
        "evaluator": sha256_file(Path(__file__)),
        "canonical_selector": sha256_file(Path(matrix.__file__)),
    }
    run_identity = {
        "experiment_version": EXPERIMENT_VERSION,
        "input_hashes": input_hashes,
        "reaches": REACH_VALUES,
        "mmr_lambdas": MMR_LAMBDAS,
        "dpp_exponents": DPP_EXPONENTS,
        "queue_size": QUEUE_SIZE,
    }
    identity_path = output / "run-identity.json"
    if identity_path.is_file():
        existing = json.loads(identity_path.read_text(encoding="utf-8"))
        if existing != run_identity:
            raise ValueError("output checkpoint belongs to a different experiment input")
    else:
        atomic_json(identity_path, run_identity)

    phone_hash = input_hashes["phone_report"]
    if phone_hash != matrix.EXPECTED_PHONE_REPORT_SHA256:
        raise ValueError("phone cohort report hash mismatch")
    phone_report = json.loads(args.phone_report.read_text(encoding="utf-8"))
    seed_ids = [int(value) for value in phone_report["request"]["seedTrackIds"]]
    if args.seed_limit is not None:
        seed_ids = seed_ids[: args.seed_limit]
    configs = list(configurations())
    if args.config_limit is not None:
        configs = configs[: args.config_limit]

    started = time.perf_counter()
    library, database_hash = load_active_library(args.database, args.active_catalog)
    id_to_index = {int(track_id): index for index, track_id in enumerate(library.track_ids)}
    missing = [seed_id for seed_id in seed_ids if seed_id not in id_to_index]
    if missing:
        raise ValueError(f"seed cohort outside active domain: {missing}")
    contexts = {
        seed_id: matrix.seed_context(library, id_to_index[seed_id]) for seed_id in seed_ids
    }

    records_path = output / "queues.jsonl"
    completed = load_completed(records_path)
    total = len(seed_ids) * len(configs)
    print(
        f"Extended pool: {len(seed_ids)} seeds x {len(configs)} configs; "
        f"resume={len(completed)}/{total}",
        flush=True,
    )
    for seed_position, seed_id in enumerate(seed_ids, start=1):
        context = contexts[seed_id]
        seed = queue_eval.track_summary(library, context.seed_index)
        for config_position, config in enumerate(configs, start=1):
            key = record_key(seed_id, config)
            if key in completed:
                continue
            result = matrix.run_config(library, None, None, context, config)
            record = {
                "record_key": key,
                "seed": seed,
                **result,
            }
            append_jsonl(records_path, record)
            completed[key] = record
            print(
                f"seed {seed_position}/{len(seed_ids)} config {config_position}/{len(configs)} "
                f"{config_id(config)} candidates={result['actual_candidate_count']} "
                f"selection={result['timing_ms']['selection']:.1f}ms",
                flush=True,
            )

    expected_keys = {record_key(seed_id, config) for seed_id in seed_ids for config in configs}
    missing_keys = sorted(expected_keys - completed.keys())
    if missing_keys:
        raise AssertionError(f"checkpoint is incomplete: {missing_keys[:3]}")
    records = [completed[key] for key in sorted(expected_keys)]
    summary = summarize(records)
    atomic_json(output / "summary.json", summary)
    if not args.seed_limit and not args.config_limit:
        write_listening_packet(
            output / "listening-packet.md",
            output / "listening-key.json",
            records,
        )

    database_hash_after = sha256_file(args.database)
    if database_hash_after != database_hash:
        raise AssertionError("frozen database changed during extended-pool evaluation")
    atomic_json(
        output / "manifest.json",
        {
            **run_identity,
            "created_date": "2026-07-15",
            "host": {
                "platform": platform.platform(),
                "python": sys.version,
                "numpy": np.__version__,
                "openblas_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
                "omp_threads": os.environ.get("OMP_NUM_THREADS"),
            },
            "active_tracks": library.count,
            "seed_track_ids": seed_ids,
            "configuration_count": len(configs),
            "queue_record_count": len(records),
            "database_sha256_after": database_hash_after,
            "elapsed_seconds": time.perf_counter() - started,
            "metadata_policy": (
                "Metadata is labels, copy diagnostics, and the explicit artist-credit "
                "constraint only; it never enters relevance or diversity scores."
            ),
        },
    )
    write_hashes(output, input_hashes)
    print(
        f"Complete: {len(records)} queues in {time.perf_counter() - started:.1f}s; "
        f"output={output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
