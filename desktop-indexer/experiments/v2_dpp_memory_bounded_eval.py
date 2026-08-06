#!/usr/bin/env python3
"""Measure an exact memory-bounded DPP greedy core on the frozen active library.

The canonical host reproduction first copies every candidate embedding into one
contiguous matrix.  The bounded variant keeps only candidate row indices and obtains
each selected similarity column by scanning the already memory-mapped active matrix.
Both variants otherwise execute the same Float32 greedy update and stable candidate
ordering.  Selector runs happen in isolated child processes after explicitly prefaulting
the embedding map, mirroring ``EmbeddingIndex.buffer.load()`` without contaminating the
anonymous-heap measurement with the SQLite loader's transient full-library copy.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import v2_dpp_certificate_eval as certificate
import v2_extended_pool_eval as extended
import v2_queue_eval as queue_eval
import v2_selection_knob_matrix as matrix


REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_DATABASE = queue_eval.DEFAULT_DB
DEFAULT_ACTIVE_CATALOG = matrix.DEFAULT_ACTIVE_CATALOG
DEFAULT_CERTIFICATE_OUTPUT = certificate.DEFAULT_OUTPUT
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "desktop-indexer"
    / "audit_raw_data"
    / "v2-discovery"
    / "dpp-memory-bounded-2026-07-15"
)

EXPERIMENT_VERSION = "dpp-memory-bounded-active-domain-v1"
QUEUE_SIZE = 30
MIN_MARGINAL_GAIN = certificate.MIN_MARGINAL_GAIN


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    seed_track_id: int
    dpp_exponent: float
    candidate_count: int


# Every count is a real terminal certificate size in the frozen certificate evidence.
BENCHMARK_CASES = (
    BenchmarkCase("certified-1606-q2", 33821, 2.0, 1_606),
    BenchmarkCase("certified-12848-q2", 80437, 2.0, 12_848),
    BenchmarkCase("certified-51392-q0p5", 80437, 0.5, 51_392),
    BenchmarkCase("full-80322-q0", 80437, 0.0, 80_322),
)


@dataclass(frozen=True)
class GreedyResult:
    selected_indices: tuple[int, ...]
    selected_marginal_gains: tuple[float, ...]


def greedy_dpp_with_similarity_columns(
    relevance: np.ndarray,
    artist_codes: np.ndarray,
    exponent: float,
    similarity_column: Callable[[int], np.ndarray],
    queue_size: int = QUEUE_SIZE,
) -> GreedyResult:
    """Run the shared Float32 greedy core with a supplied candidate similarity column."""
    scores = np.asarray(relevance, dtype=np.float32)
    codes = np.asarray(artist_codes, dtype=np.int32)
    if scores.ndim != 1 or codes.shape != scores.shape:
        raise ValueError("relevance and artist_codes must be equal one-dimensional arrays")
    config = matrix.SelectorConfig(
        mode="dpp",
        queue_size=queue_size,
        reach=1.0,
        dpp_exponent=exponent,
    )
    quality = certificate.quality_scores(scores, exponent)
    count = scores.size
    limit = min(queue_size, count)
    factors = np.zeros((count, limit), dtype=np.float32)
    residual = (quality * quality).astype(np.float32, copy=False)
    remaining = np.ones(count, dtype=np.bool_)
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

        similarities = np.asarray(similarity_column(best), dtype=np.float32)
        if similarities.shape != scores.shape:
            raise ValueError("similarity column must match the candidate count")
        root = math.sqrt(best_gain)
        kernel = quality * quality[best] * similarities
        if step:
            kernel -= factors[:, :step] @ factors[best, :step]
        new_factor = kernel / root
        factors[remaining, step] = new_factor[remaining]
        residual[remaining] -= new_factor[remaining] ** 2
        np.maximum(residual, 0.0, out=residual)
        factors[best, step] = root

    return GreedyResult(tuple(selected), tuple(marginal_gains))


def greedy_materialized(
    embeddings: np.ndarray,
    candidates: np.ndarray,
    relevance: np.ndarray,
    artist_codes: np.ndarray,
    exponent: float,
) -> GreedyResult:
    """Canonical candidate-matrix shape used by the existing host reproduction."""
    candidate_vectors = embeddings[candidates]
    return greedy_dpp_with_similarity_columns(
        relevance,
        artist_codes,
        exponent,
        lambda best: candidate_vectors @ candidate_vectors[best],
    )


def greedy_streaming(
    embeddings: np.ndarray,
    candidates: np.ndarray,
    relevance: np.ndarray,
    artist_codes: np.ndarray,
    exponent: float,
) -> GreedyResult:
    """Scan the mapped library for a column, then gather only candidate similarities."""
    return greedy_dpp_with_similarity_columns(
        relevance,
        artist_codes,
        exponent,
        lambda best: (embeddings @ embeddings[int(candidates[best])])[candidates],
    )


def proc_memory_kib() -> dict[str, int]:
    wanted = {"VmRSS", "VmHWM", "RssAnon", "RssFile", "RssShmem"}
    values: dict[str, int] = {}
    with Path("/proc/self/status").open("r", encoding="ascii") as handle:
        for line in handle:
            key, separator, remainder = line.partition(":")
            if separator and key in wanted:
                values[key] = int(remainder.strip().split()[0])
    return values


class MemorySampler:
    def __init__(self, interval_seconds: float = 0.002) -> None:
        self.interval_seconds = interval_seconds
        self.peak = proc_memory_kib()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)

    def _sample(self) -> None:
        for key, value in proc_memory_kib().items():
            self.peak[key] = max(self.peak.get(key, 0), value)

    def _sample_loop(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> dict[str, int]:
        self._sample()
        self._stop.set()
        self._thread.join()
        self._sample()
        return dict(self.peak)


def prefault(array: np.ndarray) -> int:
    """Touch each 4 KiB page, matching the app's eager mapped-buffer load policy."""
    bytes_view = np.asarray(array).reshape(-1).view(np.uint8)
    checksum = int(np.sum(bytes_view[::4096], dtype=np.uint64))
    if bytes_view.size:
        checksum += int(bytes_view[-1])
    return checksum


def atomic_save_npy(path: Path, value: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def artifact_paths(output: Path, seed_track_id: int | None = None) -> dict[str, Path]:
    paths = {
        "embeddings": output / "active-embeddings.npy",
        "track_ids": output / "active-track-ids.npy",
    }
    if seed_track_id is not None:
        stem = f"seed-{seed_track_id}"
        paths.update(
            {
                "candidates": output / f"{stem}-candidates.npy",
                "relevance": output / f"{stem}-relevance.npy",
                "artist_codes": output / f"{stem}-artist-codes.npy",
            }
        )
    return paths


def source_identity(database: Path, active_catalog: Path) -> dict[str, object]:
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "database_sha256": extended.sha256_file(database),
        "active_catalog_sha256": extended.sha256_file(active_catalog),
        "active_tracks": matrix.EXPECTED_ACTIVE_TRACKS,
        "embedding_dim": queue_eval.EXPECTED_DIM,
        "cases": [asdict(case) for case in BENCHMARK_CASES],
    }


def prepare_artifacts(
    database: Path,
    active_catalog: Path,
    certificate_output: Path,
    output: Path,
    force: bool,
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    identity = source_identity(database, active_catalog)
    identity_path = output / "artifact-identity.json"
    required = list(artifact_paths(output).values())
    for seed_id in sorted({case.seed_track_id for case in BENCHMARK_CASES}):
        required.extend(artifact_paths(output, seed_id).values())
    can_reuse = (
        not force
        and identity_path.is_file()
        and all(path.is_file() for path in required)
        and json.loads(identity_path.read_text(encoding="utf-8")) == identity
    )
    if not can_reuse:
        library, database_hash = extended.load_active_library(database, active_catalog)
        if database_hash != identity["database_sha256"]:
            raise AssertionError("database changed while preparing benchmark artifacts")
        common = artifact_paths(output)
        atomic_save_npy(common["embeddings"], library.embeddings)
        atomic_save_npy(common["track_ids"], library.track_ids)
        id_to_index = {
            int(track_id): index for index, track_id in enumerate(library.track_ids)
        }
        for seed_id in sorted({case.seed_track_id for case in BENCHMARK_CASES}):
            context = matrix.seed_context(library, id_to_index[seed_id])
            candidates = context.closest_order.astype(np.int64, copy=False)
            relevance = context.seed_similarities[candidates].astype(np.float32, copy=False)
            artist_codes, _ = matrix.candidate_artist_codes(library, candidates)
            paths = artifact_paths(output, seed_id)
            atomic_save_npy(paths["candidates"], candidates)
            atomic_save_npy(paths["relevance"], relevance)
            atomic_save_npy(paths["artist_codes"], artist_codes)
        extended.atomic_json(identity_path, identity)
        del library
        gc.collect()

    certificate_records: dict[str, dict[str, object]] = {}
    records_path = certificate_output / "queues.jsonl"
    if not records_path.is_file():
        raise FileNotFoundError(records_path)
    with records_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            key = certificate.record_key(
                int(record["seed"]["track_id"]), float(record["dpp_exponent"])
            )
            certificate_records[key] = record
    for case in BENCHMARK_CASES:
        key = certificate.record_key(case.seed_track_id, case.dpp_exponent)
        record = certificate_records.get(key)
        if record is None:
            raise ValueError(f"missing frozen certificate record for {case.case_id}")
        if int(record["certificate"]["final_candidate_count"]) != case.candidate_count:
            raise ValueError(f"frozen certificate count changed for {case.case_id}")
    return identity, certificate_records


def run_worker(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    embeddings = np.load(artifact_paths(output)["embeddings"], mmap_mode="r")
    track_ids = np.load(artifact_paths(output)["track_ids"], mmap_mode="r")
    paths = artifact_paths(output, args.seed_track_id)
    candidates = np.load(paths["candidates"], mmap_mode="r")[: args.candidate_count]
    relevance = np.load(paths["relevance"], mmap_mode="r")[: args.candidate_count]
    artist_codes = np.load(paths["artist_codes"], mmap_mode="r")[: args.candidate_count]
    prefault_checksum = prefault(embeddings)
    baseline = proc_memory_kib()
    sampler = MemorySampler()
    sampler.start()
    timings_ms: list[float] = []
    results: list[GreedyResult] = []
    selector = greedy_materialized if args.implementation == "materialized" else greedy_streaming
    for _ in range(args.repeats):
        started = time.perf_counter()
        result = selector(
            embeddings,
            candidates,
            relevance,
            artist_codes,
            args.dpp_exponent,
        )
        timings_ms.append((time.perf_counter() - started) * 1000.0)
        results.append(result)
        gc.collect()
    peak = sampler.stop()
    first = results[0]
    if any(result != first for result in results[1:]):
        raise AssertionError("selector output changed between repetitions")
    selected_track_ids = [
        int(track_ids[int(candidates[index])]) for index in first.selected_indices
    ]
    selected_candidate_ranks = [index + 1 for index in first.selected_indices]
    result = {
        "implementation": args.implementation,
        "seed_track_id": args.seed_track_id,
        "dpp_exponent": args.dpp_exponent,
        "candidate_count": args.candidate_count,
        "repeats": args.repeats,
        "selected_track_ids": selected_track_ids,
        "selected_candidate_ranks": selected_candidate_ranks,
        "selected_marginal_gains": first.selected_marginal_gains,
        "timing_ms": {
            "runs": timings_ms,
            "min": min(timings_ms),
            "median": float(np.median(np.asarray(timings_ms, dtype=np.float64))),
            "max": max(timings_ms),
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
        "theoretical_bytes": {
            "candidate_embedding_copy": (
                args.candidate_count * int(embeddings.shape[1]) * np.dtype(np.float32).itemsize
                if args.implementation == "materialized"
                else 0
            ),
            "cholesky_factors": (
                args.candidate_count * min(QUEUE_SIZE, args.candidate_count)
                * np.dtype(np.float32).itemsize
            ),
            "full_similarity_column": (
                int(embeddings.shape[0]) * np.dtype(np.float32).itemsize
                if args.implementation == "streaming"
                else 0
            ),
        },
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")), flush=True)
    return 0


def run_isolated(
    output: Path,
    implementation: str,
    case: BenchmarkCase,
    repeats: int,
) -> dict[str, object]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--output",
        str(output),
        "--implementation",
        implementation,
        "--seed-track-id",
        str(case.seed_track_id),
        "--dpp-exponent",
        str(case.dpp_exponent),
        "--candidate-count",
        str(case.candidate_count),
        "--repeats",
        str(repeats),
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


def exact_float_list(left: Sequence[float], right: Sequence[float]) -> bool:
    return np.array_equal(
        np.asarray(left, dtype=np.float32),
        np.asarray(right, dtype=np.float32),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--active-catalog", type=Path, default=DEFAULT_ACTIVE_CATALOG)
    parser.add_argument(
        "--certificate-output", type=Path, default=DEFAULT_CERTIFICATE_OUTPUT
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--implementation",
        choices=("materialized", "streaming"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--seed-track-id", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--dpp-exponent", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--candidate-count", type=int, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.worker:
        required = (
            args.implementation,
            args.seed_track_id,
            args.dpp_exponent,
            args.candidate_count,
        )
        if any(value is None for value in required):
            raise ValueError("worker arguments are incomplete")
        return run_worker(args)
    if args.repeats <= 0:
        raise ValueError("repeats must be positive")

    output = args.output.resolve()
    if args.force and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    identity, certificate_records = prepare_artifacts(
        args.database,
        args.active_catalog,
        args.certificate_output,
        output,
        args.force,
    )
    rows: list[dict[str, object]] = []
    for position, case in enumerate(BENCHMARK_CASES, start=1):
        print(
            f"DPP memory {position}/{len(BENCHMARK_CASES)}: {case.case_id} ",
            end="",
            flush=True,
        )
        materialized = run_isolated(output, "materialized", case, args.repeats)
        streaming = run_isolated(output, "streaming", case, args.repeats)
        key = certificate.record_key(case.seed_track_id, case.dpp_exponent)
        frozen = certificate_records[key]
        frozen_ids = [int(value) for value in frozen["track_ids"]]
        frozen_ranks = [int(value) for value in frozen["candidate_ranks"]]
        frozen_gains = [
            float(value)
            for value in frozen["certificate"]["selected_marginal_gains"]
        ]
        exact_between = (
            materialized["selected_track_ids"] == streaming["selected_track_ids"]
            and materialized["selected_candidate_ranks"]
            == streaming["selected_candidate_ranks"]
            and exact_float_list(
                materialized["selected_marginal_gains"],
                streaming["selected_marginal_gains"],
            )
        )
        exact_frozen = (
            materialized["selected_track_ids"] == frozen_ids
            and materialized["selected_candidate_ranks"] == frozen_ranks
            and exact_float_list(materialized["selected_marginal_gains"], frozen_gains)
        )
        if not exact_between or not exact_frozen:
            raise AssertionError(f"exact DPP identity failed for {case.case_id}")
        row = {
            "case": asdict(case),
            "exact_materialized_streaming_identity": exact_between,
            "exact_frozen_certificate_identity": exact_frozen,
            "queue_fingerprint": matrix.stable_json_hash(
                materialized["selected_track_ids"]
            ),
            "materialized": materialized,
            "streaming": streaming,
        }
        rows.append(row)
        materialized_anon = int(
            materialized["memory_kib"]["sampled_peak_minus_baseline"]["RssAnon"]
        )
        streaming_anon = int(
            streaming["memory_kib"]["sampled_peak_minus_baseline"]["RssAnon"]
        )
        print(
            f"exact; anon delta {materialized_anon / 1024:.1f} -> "
            f"{streaming_anon / 1024:.1f} MiB",
            flush=True,
        )

    results = {
        "experiment_version": EXPERIMENT_VERSION,
        "source_identity": identity,
        "cases": rows,
        "all_exact_materialized_streaming": all(
            bool(row["exact_materialized_streaming_identity"]) for row in rows
        ),
        "all_exact_frozen_certificate": all(
            bool(row["exact_frozen_certificate_identity"]) for row in rows
        ),
    }
    extended.atomic_json(output / "results.json", results)
    artifact_hashes = {
        path.name: extended.sha256_file(path)
        for path in sorted(output.glob("*.npy"))
    }
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
        "artifact_hashes": artifact_hashes,
        "certificate_results_sha256": extended.sha256_file(
            args.certificate_output / "queues.jsonl"
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "memory_policy": (
            "Every worker prefaults the read-only active embedding mmap before recording "
            "its baseline. RssAnon delta therefore isolates selector-owned heap; RssFile "
            "records mapped pages separately."
        ),
        "kernel_policy": (
            "Materialized and bounded variants share quality, eligibility, factor, "
            "residual, stopping, and stable argmax code. Only the embedding similarity "
            "column source differs."
        ),
    }
    extended.atomic_json(output / "manifest.json", manifest)
    input_hashes = {
        "database": str(identity["database_sha256"]),
        "active_catalog": str(identity["active_catalog_sha256"]),
        "certificate_results": manifest["certificate_results_sha256"],
        "evaluator": extended.sha256_file(Path(__file__)),
    }
    extended.write_hashes(output, input_hashes)
    print(
        f"Complete: {len(rows)} exact isolated comparisons in "
        f"{time.perf_counter() - started:.1f}s; output={output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
