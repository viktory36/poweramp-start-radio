#!/usr/bin/env python3
"""Compare complete bounded DPP prefix certification with one bounded full run.

Similarity columns are computed in fixed-size candidate-row blocks. This keeps the
embedding scratch bounded while making measured work proportional to each attempted
prefix, unlike a full-library similarity scan on every step.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import v2_dpp_certificate_eval as certificate
import v2_dpp_memory_bounded_eval as memory_eval
import v2_extended_pool_eval as extended
import v2_queue_eval as queue_eval
import v2_selection_knob_matrix as matrix


DEFAULT_ARTIFACTS = memory_eval.DEFAULT_OUTPUT
DEFAULT_OUTPUT = DEFAULT_ARTIFACTS / "adaptive-prefix-vs-full"
DEFAULT_DATABASE = queue_eval.DEFAULT_DB
DEFAULT_ACTIVE_CATALOG = matrix.DEFAULT_ACTIVE_CATALOG
DEFAULT_CERTIFICATE_OUTPUT = certificate.DEFAULT_OUTPUT
EXPERIMENT_VERSION = "dpp-adaptive-bounded-vs-full-active-domain-v1"
SEED_TRACK_ID = 80437
EXPONENTS = (0.0, 0.5, 1.0, 2.0)
INITIAL_CANDIDATE_COUNT = 1_606
GROWTH_FACTOR = 2.0
BLOCK_ROWS = 4_096


@dataclass(frozen=True)
class AdaptiveResult:
    greedy: memory_eval.GreedyResult
    attempted_candidate_counts: tuple[int, ...]
    selected_steps_per_attempt: tuple[int, ...]
    candidate_step_rows: int
    final_unseen_gain_upper_bound: float | None
    attempt_timing_ms: tuple[float, ...]


def greedy_bounded_gather(
    embeddings: np.ndarray,
    candidates: np.ndarray,
    relevance: np.ndarray,
    artist_codes: np.ndarray,
    exponent: float,
    block_rows: int = BLOCK_ROWS,
) -> memory_eval.GreedyResult:
    """Compute candidate dots in fixed-size gathered blocks, never an N x dim copy."""
    if block_rows <= 0:
        raise ValueError("block_rows must be positive")
    count = int(candidates.size)

    def similarity_column(best: int) -> np.ndarray:
        selected = embeddings[int(candidates[best])]
        similarities = np.empty(count, dtype=np.float32)
        for start in range(0, count, block_rows):
            end = min(count, start + block_rows)
            block = embeddings[candidates[start:end]]
            similarities[start:end] = block @ selected
        return similarities

    return memory_eval.greedy_dpp_with_similarity_columns(
        relevance,
        artist_codes,
        exponent,
        similarity_column,
    )


def select_certified_bounded(
    embeddings: np.ndarray,
    candidates: np.ndarray,
    relevance: np.ndarray,
    artist_codes: np.ndarray,
    exponent: float,
    initial_candidate_count: int = INITIAL_CANDIDATE_COUNT,
    growth_factor: float = GROWTH_FACTOR,
    block_rows: int = BLOCK_ROWS,
) -> AdaptiveResult:
    total = int(candidates.size)
    if total == 0:
        return AdaptiveResult(
            memory_eval.GreedyResult((), ()), (), (), 0, None, ()
        )
    suffix_bounds = certificate.suffix_initial_gain_bounds(relevance, exponent)
    candidate_count = min(total, max(1, initial_candidate_count))
    attempts: list[int] = []
    selected_steps: list[int] = []
    timings: list[float] = []
    candidate_step_rows = 0
    while True:
        attempts.append(candidate_count)
        started = time.perf_counter()
        greedy = greedy_bounded_gather(
            embeddings,
            candidates[:candidate_count],
            relevance[:candidate_count],
            artist_codes[:candidate_count],
            exponent,
            block_rows,
        )
        timings.append((time.perf_counter() - started) * 1000.0)
        steps = len(greedy.selected_indices)
        selected_steps.append(steps)
        candidate_step_rows += candidate_count * steps
        unseen_bound = (
            None if candidate_count == total else float(suffix_bounds[candidate_count])
        )
        target = min(memory_eval.QUEUE_SIZE, total)
        selected_enough = len(greedy.selected_indices) == target or (
            unseen_bound is not None
            and unseen_bound <= memory_eval.MIN_MARGINAL_GAIN
        )
        every_step_certified = unseen_bound is None or all(
            gain > unseen_bound for gain in greedy.selected_marginal_gains
        )
        if unseen_bound is None or (selected_enough and every_step_certified):
            return AdaptiveResult(
                greedy=greedy,
                attempted_candidate_counts=tuple(attempts),
                selected_steps_per_attempt=tuple(selected_steps),
                candidate_step_rows=candidate_step_rows,
                final_unseen_gain_upper_bound=unseen_bound,
                attempt_timing_ms=tuple(timings),
            )
        candidate_count = min(
            total,
            max(candidate_count + 1, math.ceil(candidate_count * growth_factor)),
        )


def semantic_adaptive(result: AdaptiveResult) -> tuple[object, ...]:
    return (
        result.greedy,
        result.attempted_candidate_counts,
        result.selected_steps_per_attempt,
        result.candidate_step_rows,
        result.final_unseen_gain_upper_bound,
    )


def run_worker(args: argparse.Namespace) -> int:
    artifacts = args.artifacts.resolve()
    embeddings = np.load(
        memory_eval.artifact_paths(artifacts)["embeddings"], mmap_mode="r"
    )
    track_ids = np.load(
        memory_eval.artifact_paths(artifacts)["track_ids"], mmap_mode="r"
    )
    paths = memory_eval.artifact_paths(artifacts, SEED_TRACK_ID)
    candidates = np.load(paths["candidates"], mmap_mode="r")
    relevance = np.load(paths["relevance"], mmap_mode="r")
    artist_codes = np.load(paths["artist_codes"], mmap_mode="r")
    prefault_checksum = memory_eval.prefault(embeddings)
    baseline = memory_eval.proc_memory_kib()
    sampler = memory_eval.MemorySampler()
    sampler.start()
    run_timings: list[float] = []
    results: list[AdaptiveResult | memory_eval.GreedyResult] = []
    adaptive_attempt_timings: list[tuple[float, ...]] = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        if args.mode == "adaptive":
            adaptive = select_certified_bounded(
                embeddings,
                candidates,
                relevance,
                artist_codes,
                args.dpp_exponent,
                block_rows=args.block_rows,
            )
            result: AdaptiveResult | memory_eval.GreedyResult = adaptive
            adaptive_attempt_timings.append(adaptive.attempt_timing_ms)
        else:
            result = greedy_bounded_gather(
                embeddings,
                candidates,
                relevance,
                artist_codes,
                args.dpp_exponent,
                args.block_rows,
            )
        run_timings.append((time.perf_counter() - started) * 1000.0)
        results.append(result)
        gc.collect()
    peak = sampler.stop()
    first = results[0]
    if args.mode == "adaptive":
        assert isinstance(first, AdaptiveResult)
        if any(
            not isinstance(other, AdaptiveResult)
            or semantic_adaptive(other) != semantic_adaptive(first)
            for other in results[1:]
        ):
            raise AssertionError("adaptive result changed between repetitions")
        greedy = first.greedy
        attempt_count = len(first.attempted_candidate_counts)
        per_attempt_medians = [
            float(
                np.median(
                    np.asarray(
                        [timings[index] for timings in adaptive_attempt_timings],
                        dtype=np.float64,
                    )
                )
            )
            for index in range(attempt_count)
        ]
        execution = {
            "attempted_candidate_counts": first.attempted_candidate_counts,
            "selected_steps_per_attempt": first.selected_steps_per_attempt,
            "candidate_step_rows": first.candidate_step_rows,
            "final_unseen_gain_upper_bound": first.final_unseen_gain_upper_bound,
            "median_attempt_timing_ms": per_attempt_medians,
        }
    else:
        assert isinstance(first, memory_eval.GreedyResult)
        if any(other != first for other in results[1:]):
            raise AssertionError("full result changed between repetitions")
        greedy = first
        execution = {
            "attempted_candidate_counts": (int(candidates.size),),
            "selected_steps_per_attempt": (len(greedy.selected_indices),),
            "candidate_step_rows": int(candidates.size) * len(greedy.selected_indices),
            "final_unseen_gain_upper_bound": None,
            "median_attempt_timing_ms": (
                float(np.median(np.asarray(run_timings, dtype=np.float64))),
            ),
        }
    selected_track_ids = [
        int(track_ids[int(candidates[index])]) for index in greedy.selected_indices
    ]
    output = {
        "mode": args.mode,
        "seed_track_id": SEED_TRACK_ID,
        "dpp_exponent": args.dpp_exponent,
        "block_rows": args.block_rows,
        "repeats": args.repeats,
        "selected_track_ids": selected_track_ids,
        "selected_candidate_ranks": [index + 1 for index in greedy.selected_indices],
        "selected_marginal_gains": greedy.selected_marginal_gains,
        "execution": execution,
        "timing_ms": {
            "runs": run_timings,
            "min": min(run_timings),
            "median": float(np.median(np.asarray(run_timings, dtype=np.float64))),
            "max": max(run_timings),
        },
        "memory_kib": {
            "baseline_after_prefault": baseline,
            "sampled_peak": peak,
            "sampled_peak_minus_baseline": {
                key: peak.get(key, 0) - baseline.get(key, 0)
                for key in sorted(set(baseline) | set(peak))
            },
        },
        "prefault_checksum": prefault_checksum,
    }
    print(json.dumps(output, sort_keys=True, separators=(",", ":")), flush=True)
    return 0


def isolated_run(
    artifacts: Path,
    mode: str,
    exponent: float,
    repeats: int,
    block_rows: int,
) -> dict[str, object]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--artifacts",
        str(artifacts),
        "--mode",
        mode,
        "--dpp-exponent",
        str(exponent),
        "--repeats",
        str(repeats),
        "--block-rows",
        str(block_rows),
    ]
    environment = dict(os.environ)
    environment["OPENBLAS_NUM_THREADS"] = "1"
    environment["OMP_NUM_THREADS"] = "1"
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    return json.loads(completed.stdout)


def exact_gains(left: Sequence[float], right: Sequence[float]) -> bool:
    return np.array_equal(
        np.asarray(left, dtype=np.float32), np.asarray(right, dtype=np.float32)
    )


def load_certificate_records(path: Path) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    with (path / "queues.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            records[
                certificate.record_key(
                    int(record["seed"]["track_id"]),
                    float(record["dpp_exponent"]),
                )
            ] = record
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--active-catalog", type=Path, default=DEFAULT_ACTIVE_CATALOG)
    parser.add_argument(
        "--certificate-output", type=Path, default=DEFAULT_CERTIFICATE_OUTPUT
    )
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--block-rows", type=int, default=BLOCK_ROWS)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--mode", choices=("adaptive", "full"), help=argparse.SUPPRESS
    )
    parser.add_argument("--dpp-exponent", type=float, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.worker:
        if args.mode is None or args.dpp_exponent is None:
            raise ValueError("worker mode and exponent are required")
        return run_worker(args)
    if args.repeats <= 0 or args.block_rows <= 0:
        raise ValueError("repeats and block_rows must be positive")

    artifacts = args.artifacts.resolve()
    artifacts.mkdir(parents=True, exist_ok=True)
    memory_eval.prepare_artifacts(
        args.database,
        args.active_catalog,
        args.certificate_output,
        artifacts,
        force=False,
    )
    output = args.output.resolve()
    if args.force and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    certificate_records = load_certificate_records(args.certificate_output)
    started = time.perf_counter()
    rows: list[dict[str, object]] = []
    for position, exponent in enumerate(EXPONENTS, start=1):
        print(
            f"Adaptive bounded {position}/{len(EXPONENTS)}: q^{exponent:g} ",
            end="",
            flush=True,
        )
        adaptive = isolated_run(
            artifacts, "adaptive", exponent, args.repeats, args.block_rows
        )
        full = isolated_run(artifacts, "full", exponent, args.repeats, args.block_rows)
        key = certificate.record_key(SEED_TRACK_ID, exponent)
        frozen = certificate_records[key]
        frozen_ids = [int(value) for value in frozen["track_ids"]]
        frozen_ranks = [int(value) for value in frozen["candidate_ranks"]]
        frozen_gains = [
            float(value)
            for value in frozen["certificate"]["selected_marginal_gains"]
        ]
        exact_adaptive_full = (
            adaptive["selected_track_ids"] == full["selected_track_ids"]
            and adaptive["selected_candidate_ranks"]
            == full["selected_candidate_ranks"]
            and exact_gains(
                adaptive["selected_marginal_gains"], full["selected_marginal_gains"]
            )
        )
        exact_frozen = (
            adaptive["selected_track_ids"] == frozen_ids
            and adaptive["selected_candidate_ranks"] == frozen_ranks
            and exact_gains(adaptive["selected_marginal_gains"], frozen_gains)
            and adaptive["execution"]["attempted_candidate_counts"]
            == frozen["certificate"]["attempted_candidate_counts"]
        )
        if not exact_adaptive_full or not exact_frozen:
            raise AssertionError(f"adaptive/full/frozen mismatch at exponent {exponent}")
        adaptive_rows = int(adaptive["execution"]["candidate_step_rows"])
        full_rows = int(full["execution"]["candidate_step_rows"])
        adaptive_ms = float(adaptive["timing_ms"]["median"])
        full_ms = float(full["timing_ms"]["median"])
        row = {
            "seed_track_id": SEED_TRACK_ID,
            "dpp_exponent": exponent,
            "exact_adaptive_full_identity": exact_adaptive_full,
            "exact_frozen_certificate_identity": exact_frozen,
            "adaptive": adaptive,
            "full": full,
            "comparison": {
                "adaptive_over_full_candidate_step_rows": adaptive_rows / full_rows,
                "adaptive_over_full_runtime": adaptive_ms / full_ms,
                "candidate_step_rows_saved_by_adaptive": full_rows - adaptive_rows,
                "median_ms_saved_by_adaptive": full_ms - adaptive_ms,
            },
        }
        rows.append(row)
        print(
            f"exact; rows {adaptive_rows / full_rows:.3f}x, "
            f"time {adaptive_ms / full_ms:.3f}x",
            flush=True,
        )

    results = {
        "experiment_version": EXPERIMENT_VERSION,
        "seed_track_id": SEED_TRACK_ID,
        "initial_candidate_count": INITIAL_CANDIDATE_COUNT,
        "growth_factor": GROWTH_FACTOR,
        "block_rows": args.block_rows,
        "cases": rows,
        "all_exact_adaptive_full": all(
            bool(row["exact_adaptive_full_identity"]) for row in rows
        ),
        "all_exact_frozen_certificate": all(
            bool(row["exact_frozen_certificate_identity"]) for row in rows
        ),
    }
    extended.atomic_json(output / "results.json", results)
    manifest = {
        "experiment_version": EXPERIMENT_VERSION,
        "created_date": "2026-07-15",
        "host": {
            "platform": platform.platform(),
            "python": sys.version,
            "numpy": np.__version__,
            "worker_openblas_threads": "1",
            "worker_omp_threads": "1",
        },
        "selector_repetitions": args.repeats,
        "elapsed_seconds": time.perf_counter() - started,
        "artifact_identity_sha256": extended.sha256_file(
            artifacts / "artifact-identity.json"
        ),
        "certificate_results_sha256": extended.sha256_file(
            args.certificate_output / "queues.jsonl"
        ),
        "work_metric": (
            "candidate_step_rows is sum(attempt candidate count * selected steps); "
            "all cases selected 30 steps per attempt."
        ),
        "bounded_gather_policy": (
            f"At most {args.block_rows} x {queue_eval.EXPECTED_DIM} Float32 embedding "
            "values are copied for a similarity block."
        ),
    }
    extended.atomic_json(output / "manifest.json", manifest)
    extended.write_hashes(
        output,
        {
            "artifact_identity": manifest["artifact_identity_sha256"],
            "certificate_results": manifest["certificate_results_sha256"],
            "evaluator": extended.sha256_file(Path(__file__)),
        },
    )
    print(
        f"Complete: {len(rows)} adaptive/full comparisons in "
        f"{time.perf_counter() - started:.1f}s; output={output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
