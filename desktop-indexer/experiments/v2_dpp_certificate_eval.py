#!/usr/bin/env python3
"""Measure exact adaptive DPP certificate expansion on the frozen active library.

The adaptive implementation mirrors ``DppSelector.selectBatchCertified`` while the
reference queue comes from the existing canonical host reproduction in
``v2_selection_knob_matrix``. Each completed seed/exponent pair is fsync-checkpointed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import v2_extended_pool_eval as extended
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
    / "dpp-certificate-2026-07-15"
)

EXPERIMENT_VERSION = "dpp-certified-prefix-active-domain-v1"
DPP_EXPONENTS = (0.0, 0.5, 1.0, 2.0)
QUEUE_SIZE = 30
INITIAL_REACH = 0.02
GROWTH_FACTOR = 2.0
MIN_MARGINAL_GAIN = 1e-10


@dataclass(frozen=True)
class GreedyRun:
    selected_indices: tuple[int, ...]
    selected_marginal_gains: tuple[float, ...]


@dataclass(frozen=True)
class CertificateRun:
    greedy: GreedyRun
    total_candidate_count: int
    initial_candidate_count: int
    attempted_candidate_counts: tuple[int, ...]
    final_candidate_count: int
    final_unseen_gain_upper_bound: float | None
    used_full_domain: bool


def quality_scores(relevance: np.ndarray, exponent: float) -> np.ndarray:
    """Mirror DppSelector.qualityScore, including the exponent-zero convention."""
    if not math.isfinite(exponent) or exponent < 0.0:
        raise ValueError("exponent must be finite and non-negative")
    non_negative = np.maximum(np.asarray(relevance, dtype=np.float32), np.float32(0.0))
    if exponent == 1.0:
        return non_negative.astype(np.float32, copy=True)
    return np.power(non_negative.astype(np.float64), exponent).astype(np.float32)


def greedy_dpp(
    embeddings: np.ndarray,
    relevance: np.ndarray,
    artist_codes: np.ndarray,
    config: matrix.SelectorConfig,
    valid_mask: np.ndarray | None = None,
) -> GreedyRun:
    """Run the Float32 greedy DPP core over one already-ordered candidate domain."""
    vectors = np.asarray(embeddings, dtype=np.float32)
    scores = np.asarray(relevance, dtype=np.float32)
    codes = np.asarray(artist_codes, dtype=np.int32)
    count = scores.size
    if vectors.shape[0] != count or codes.size != count:
        raise ValueError("candidate arrays must have the same leading dimension")
    if valid_mask is None:
        valid = np.ones(count, dtype=np.bool_)
    else:
        valid = np.asarray(valid_mask, dtype=np.bool_)
        if valid.size != count:
            raise ValueError("valid_mask must match the candidate count")

    quality = quality_scores(scores, config.dpp_exponent)
    limit = min(config.queue_size, count)
    factors = np.zeros((count, limit), dtype=np.float32)
    residual = np.where(valid, quality * quality, np.float32(0.0)).astype(
        np.float32, copy=False
    )
    remaining = valid.copy()
    artist_count = int(np.max(codes)) + 1 if bool(np.any(codes >= 0)) else 0
    artist_counts = np.zeros(artist_count, dtype=np.int32)
    recent: list[int] = []
    selected: list[int] = []
    marginal_gains: list[float] = []

    for step in range(limit):
        eligible = remaining.copy()
        matrix.apply_artist_eligibility(
            eligible,
            codes,
            artist_counts,
            recent,
            config,
        )
        gains = np.where(eligible, residual, -np.inf)
        best = int(np.argmax(gains))
        best_gain = float(gains[best])
        if not math.isfinite(best_gain) or best_gain <= MIN_MARGINAL_GAIN:
            break

        selected.append(best)
        marginal_gains.append(best_gain)
        remaining[best] = False
        code = int(codes[best])
        if config.artist_limits:
            if code >= 0:
                artist_counts[code] += 1
            recent.append(code)
            if len(recent) > config.artist_spacing:
                recent.pop(0)

        root = math.sqrt(best_gain)
        kernel = quality * quality[best] * (vectors @ vectors[best])
        if step:
            kernel -= factors[:, :step] @ factors[best, :step]
        new_factor = kernel / root
        factors[remaining, step] = new_factor[remaining]
        residual[remaining] -= new_factor[remaining] ** 2
        np.maximum(residual, 0.0, out=residual)
        factors[best, step] = root

    return GreedyRun(tuple(selected), tuple(marginal_gains))


def suffix_initial_gain_bounds(relevance: np.ndarray, exponent: float) -> np.ndarray:
    quality = quality_scores(relevance, exponent)
    gains = quality * quality
    gains = np.where(np.isfinite(gains), gains, np.float32(np.inf)).astype(
        np.float32, copy=False
    )
    suffix = np.zeros(gains.size + 1, dtype=np.float32)
    if gains.size:
        suffix[:-1] = np.maximum.accumulate(gains[::-1])[::-1]
    return suffix


def select_certified(
    embeddings: np.ndarray,
    relevance: np.ndarray,
    artist_codes: np.ndarray,
    config: matrix.SelectorConfig,
    initial_candidate_count: int,
    growth_factor: float = GROWTH_FACTOR,
    valid_mask: np.ndarray | None = None,
) -> CertificateRun:
    """Mirror the production prefix-growth and strict unseen-bound certificate."""
    if not math.isfinite(growth_factor) or growth_factor <= 1.0:
        raise ValueError("growth_factor must be finite and greater than one")
    total = int(np.asarray(relevance).size)
    if total == 0 or config.queue_size <= 0:
        return CertificateRun(
            greedy=GreedyRun((), ()),
            total_candidate_count=total,
            initial_candidate_count=0,
            attempted_candidate_counts=(),
            final_candidate_count=0,
            final_unseen_gain_upper_bound=None,
            used_full_domain=total > 0,
        )

    first_count = min(total, max(1, initial_candidate_count))
    suffix_bounds = suffix_initial_gain_bounds(relevance, config.dpp_exponent)
    attempts: list[int] = []
    candidate_count = first_count
    while True:
        attempts.append(candidate_count)
        prefix_valid = None if valid_mask is None else valid_mask[:candidate_count]
        run = greedy_dpp(
            embeddings[:candidate_count],
            relevance[:candidate_count],
            artist_codes[:candidate_count],
            config,
            prefix_valid,
        )
        unseen_bound = (
            None if candidate_count == total else float(suffix_bounds[candidate_count])
        )
        target = min(config.queue_size, total)
        selected_enough = len(run.selected_indices) == target or (
            unseen_bound is not None and unseen_bound <= MIN_MARGINAL_GAIN
        )
        every_step_certified = unseen_bound is None or all(
            gain > unseen_bound for gain in run.selected_marginal_gains
        )
        if unseen_bound is None or (selected_enough and every_step_certified):
            return CertificateRun(
                greedy=run,
                total_candidate_count=total,
                initial_candidate_count=first_count,
                attempted_candidate_counts=tuple(attempts),
                final_candidate_count=candidate_count,
                final_unseen_gain_upper_bound=unseen_bound,
                used_full_domain=candidate_count == total,
            )
        candidate_count = min(
            total,
            max(candidate_count + 1, math.ceil(candidate_count * growth_factor)),
        )


def record_key(seed_id: int, exponent: float) -> str:
    return f"{seed_id}:dpp-{exponent:.8g}"


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


def numeric_summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "max": float(np.max(array)),
    }


def semantic_record(record: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in record.items() if key != "timing_ms"}


def summarize(records: Sequence[dict[str, object]]) -> dict[str, object]:
    groups: dict[str, object] = {}
    for exponent in DPP_EXPONENTS:
        items = [record for record in records if float(record["dpp_exponent"]) == exponent]
        final_counts = [float(record["certificate"]["final_candidate_count"]) for record in items]
        fractions = [
            float(record["certificate"]["final_candidate_count"])
            / float(record["certificate"]["total_candidate_count"])
            for record in items
        ]
        groups[f"{exponent:.8g}"] = {
            "seed_count": len(items),
            "exact_full_domain_identity_count": sum(
                bool(record["exact_full_domain_identity"]) for record in items
            ),
            "used_full_domain_count": sum(
                bool(record["certificate"]["used_full_domain"]) for record in items
            ),
            "final_candidate_count": numeric_summary(final_counts),
            "final_candidate_fraction": numeric_summary(fractions),
            "final_count_frequency": {
                str(count): sum(int(value) == count for value in final_counts)
                for count in sorted({int(value) for value in final_counts})
            },
            "attempt_sequences": sorted(
                {tuple(record["certificate"]["attempted_candidate_counts"]) for record in items}
            ),
            "adaptive_selection_ms": numeric_summary(
                [float(record["timing_ms"]["adaptive_selection"]) for record in items]
            ),
            "canonical_full_selection_ms": numeric_summary(
                [float(record["timing_ms"]["canonical_full_selection"]) for record in items]
            ),
        }
    return {
        "groups": groups,
        "semantic_records_sha256": extended.stable_hash(
            [semantic_record(record) for record in records]
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--active-catalog", type=Path, default=DEFAULT_ACTIVE_CATALOG)
    parser.add_argument("--phone-report", type=Path, default=DEFAULT_PHONE_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed-limit", type=int)
    parser.add_argument("--exponent-limit", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if args.force and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    input_hashes = {
        "database": extended.sha256_file(args.database),
        "active_catalog": extended.sha256_file(args.active_catalog),
        "phone_report": extended.sha256_file(args.phone_report),
        "evaluator": extended.sha256_file(Path(__file__)),
        "canonical_selector": extended.sha256_file(Path(matrix.__file__)),
    }
    initial_count = matrix.effective_pool_count(
        matrix.EXPECTED_ACTIVE_TRACKS,
        matrix.SelectorConfig(mode="dpp", queue_size=QUEUE_SIZE, reach=INITIAL_REACH),
    )
    run_identity = {
        "experiment_version": EXPERIMENT_VERSION,
        "input_hashes": input_hashes,
        "dpp_exponents": DPP_EXPONENTS,
        "queue_size": QUEUE_SIZE,
        "initial_reach": INITIAL_REACH,
        "initial_candidate_count": initial_count,
        "growth_factor": GROWTH_FACTOR,
        "artist_limits": True,
        "max_per_artist": matrix.DEFAULT_MAX_PER_ARTIST,
        "artist_spacing": matrix.DEFAULT_ARTIST_SPACING,
    }
    identity_path = output / "run-identity.json"
    if identity_path.is_file():
        existing = json.loads(identity_path.read_text(encoding="utf-8"))
        if existing != json.loads(json.dumps(run_identity)):
            raise ValueError("output checkpoint belongs to a different experiment input")
    else:
        extended.atomic_json(identity_path, run_identity)

    if input_hashes["phone_report"] != matrix.EXPECTED_PHONE_REPORT_SHA256:
        raise ValueError("phone cohort report hash mismatch")
    phone_report = json.loads(args.phone_report.read_text(encoding="utf-8"))
    seed_ids = [int(value) for value in phone_report["request"]["seedTrackIds"]]
    exponents = list(DPP_EXPONENTS)
    if args.seed_limit is not None:
        seed_ids = seed_ids[: args.seed_limit]
    if args.exponent_limit is not None:
        exponents = exponents[: args.exponent_limit]

    started = time.perf_counter()
    library, database_hash = extended.load_active_library(
        args.database,
        args.active_catalog,
    )
    id_to_index = {int(track_id): index for index, track_id in enumerate(library.track_ids)}
    missing = [seed_id for seed_id in seed_ids if seed_id not in id_to_index]
    if missing:
        raise ValueError(f"seed cohort outside active domain: {missing}")
    contexts = {
        seed_id: matrix.seed_context(library, id_to_index[seed_id]) for seed_id in seed_ids
    }

    records_path = output / "queues.jsonl"
    completed = load_completed(records_path)
    total = len(seed_ids) * len(exponents)
    print(
        f"DPP certificate: {len(seed_ids)} seeds x {len(exponents)} exponents; "
        f"resume={len(completed)}/{total}",
        flush=True,
    )
    for seed_position, seed_id in enumerate(seed_ids, start=1):
        context = contexts[seed_id]
        candidates = context.closest_order
        relevance = context.seed_similarities[candidates]
        candidate_vectors = library.embeddings[candidates]
        artist_codes, _ = matrix.candidate_artist_codes(library, candidates)
        seed = queue_eval.track_summary(library, context.seed_index)
        for exponent_position, exponent in enumerate(exponents, start=1):
            key = record_key(seed_id, exponent)
            if key in completed:
                continue
            config = matrix.SelectorConfig(
                mode="dpp",
                queue_size=QUEUE_SIZE,
                reach=1.0,
                dpp_exponent=exponent,
            )
            adaptive_started = time.perf_counter()
            certified = select_certified(
                candidate_vectors,
                relevance,
                artist_codes,
                config,
                initial_count,
            )
            adaptive_ms = (time.perf_counter() - adaptive_started) * 1000.0

            canonical_started = time.perf_counter()
            full_selected, full_ranks, full_scores = matrix.select_dpp(
                library,
                candidates,
                relevance,
                config,
            )
            canonical_ms = (time.perf_counter() - canonical_started) * 1000.0
            adaptive_selected = [
                int(candidates[index]) for index in certified.greedy.selected_indices
            ]
            adaptive_ids = [int(library.track_ids[index]) for index in adaptive_selected]
            adaptive_ranks = [index + 1 for index in certified.greedy.selected_indices]
            adaptive_scores = [float(relevance[index]) for index in certified.greedy.selected_indices]
            full_ids = [int(library.track_ids[index]) for index in full_selected]
            exact_identity = (
                adaptive_ids == full_ids
                and adaptive_ranks == full_ranks
                and adaptive_scores == full_scores
            )
            if not exact_identity:
                raise AssertionError(
                    f"certificate/full mismatch for seed={seed_id} exponent={exponent}"
                )

            record = {
                "record_key": key,
                "seed": seed,
                "dpp_exponent": exponent,
                "exact_full_domain_identity": exact_identity,
                "track_ids": adaptive_ids,
                "candidate_ranks": adaptive_ranks,
                "selection_scores": adaptive_scores,
                "queue_fingerprint": matrix.stable_json_hash(adaptive_ids),
                "certificate": {
                    "total_candidate_count": certified.total_candidate_count,
                    "initial_candidate_count": certified.initial_candidate_count,
                    "attempted_candidate_counts": certified.attempted_candidate_counts,
                    "final_candidate_count": certified.final_candidate_count,
                    "selected_marginal_gains": certified.greedy.selected_marginal_gains,
                    "final_unseen_gain_upper_bound": (
                        certified.final_unseen_gain_upper_bound
                    ),
                    "used_full_domain": certified.used_full_domain,
                },
                "timing_ms": {
                    "adaptive_selection": adaptive_ms,
                    "canonical_full_selection": canonical_ms,
                },
            }
            extended.append_jsonl(records_path, record)
            completed[key] = record
            print(
                f"seed {seed_position}/{len(seed_ids)} exponent "
                f"{exponent_position}/{len(exponents)} q^{exponent:.8g} "
                f"attempts={list(certified.attempted_candidate_counts)} "
                f"final={certified.final_candidate_count}/{certified.total_candidate_count}",
                flush=True,
            )

    expected_keys = {record_key(seed_id, exponent) for seed_id in seed_ids for exponent in exponents}
    missing_keys = sorted(expected_keys - completed.keys())
    if missing_keys:
        raise AssertionError(f"checkpoint is incomplete: {missing_keys[:3]}")
    records = [completed[key] for key in sorted(expected_keys)]
    summary = summarize(records)
    extended.atomic_json(output / "summary.json", summary)

    database_hash_after = extended.sha256_file(args.database)
    if database_hash_after != database_hash:
        raise AssertionError("frozen database changed during certificate evaluation")
    extended.atomic_json(
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
            "non_seed_candidates_per_run": library.count - 1,
            "seed_track_ids": seed_ids,
            "exponents": exponents,
            "queue_record_count": len(records),
            "exact_full_domain_identity_count": sum(
                bool(record["exact_full_domain_identity"]) for record in records
            ),
            "database_sha256_after": database_hash_after,
            "elapsed_seconds": time.perf_counter() - started,
            "metadata_policy": (
                "Metadata enters only the explicit 8-per-artist and 3-track-spacing "
                "eligibility constraint; relevance and diversity use audio embeddings."
            ),
        },
    )
    extended.write_hashes(output, input_hashes)
    print(
        f"Complete: {len(records)} certified/full comparisons in "
        f"{time.perf_counter() - started:.1f}s; output={output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
