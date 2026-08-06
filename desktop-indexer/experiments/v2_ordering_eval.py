#!/usr/bin/env python3
"""Evaluate deterministic queue sequencing with recommendation membership fixed.

The phone's current MMR and DPP selection orders are the baselines. Every alternative is
an exact permutation of the same selected tracks. The phone database is opened immutable
and read-only by v2_queue_eval; metadata affects only explicit artist constraints and
human-readable evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import v2_queue_eval as queue_eval


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    REPO_ROOT / "desktop-indexer" / "audit_raw_data" / "v2-discovery" / "ordering"
)
DEFAULT_SEEDS = 64
DEFAULT_FRONTIER = 8
EPSILON = 1e-12


@dataclass(frozen=True)
class OrderingContext:
    membership: tuple[int, ...]
    track_ids: np.ndarray
    embeddings: np.ndarray
    seed_embedding: np.ndarray
    seed_cosines: np.ndarray
    pair_cosines: np.ndarray
    pair_angles: np.ndarray
    artist_codes: np.ndarray
    spacing: int
    max_per_artist: int

    @property
    def count(self) -> int:
        return len(self.membership)


@dataclass(frozen=True)
class AlgorithmResult:
    order: tuple[int, ...]
    diagnostics: dict[str, int | float | bool]


def sha256_ids(track_ids: Iterable[int]) -> str:
    digest = hashlib.sha256()
    for track_id in track_ids:
        digest.update(int(track_id).to_bytes(8, byteorder="little", signed=True))
    return digest.hexdigest()


def normalized_artist_code(artists: Sequence[str | None]) -> np.ndarray:
    code_by_artist: dict[str, int] = {}
    codes = np.full(len(artists), -1, dtype=np.int32)
    for index, artist in enumerate(artists):
        key = queue_eval.normalized_artist(artist)
        if key is None:
            continue
        code = code_by_artist.get(key)
        if code is None:
            code = len(code_by_artist)
            code_by_artist[key] = code
        codes[index] = code
    return codes


def build_context(
    library: queue_eval.Library,
    seed_index: int,
    membership: Sequence[int],
    spacing: int = queue_eval.DEFAULT_MIN_ARTIST_SPACING,
    max_per_artist: int = queue_eval.DEFAULT_MAX_PER_ARTIST,
) -> OrderingContext:
    members = tuple(int(index) for index in membership)
    embeddings = library.embeddings[np.asarray(members, dtype=np.int64)]
    seed_embedding = library.embeddings[seed_index]
    pair_cosines = np.clip(embeddings @ embeddings.T, -1.0, 1.0)
    pair_angles = np.arccos(pair_cosines.astype(np.float64))
    seed_cosines = np.clip(embeddings @ seed_embedding, -1.0, 1.0)
    artists = [library.artists[index] for index in members]
    return OrderingContext(
        membership=members,
        track_ids=library.track_ids[np.asarray(members, dtype=np.int64)].copy(),
        embeddings=embeddings,
        seed_embedding=seed_embedding,
        seed_cosines=seed_cosines,
        pair_cosines=pair_cosines,
        pair_angles=pair_angles,
        artist_codes=normalized_artist_code(artists),
        spacing=spacing,
        max_per_artist=max_per_artist,
    )


def seed_ranked(ctx: OrderingContext, members: Iterable[int] | None = None) -> list[int]:
    values = range(ctx.count) if members is None else members
    return sorted(
        (int(index) for index in values),
        key=lambda index: (-float(ctx.seed_cosines[index]), int(ctx.track_ids[index])),
    )


def valid_append(ctx: OrderingContext, path: Sequence[int], candidate: int) -> bool:
    artist = int(ctx.artist_codes[candidate])
    if artist < 0 or ctx.spacing <= 0:
        return True
    return all(
        int(ctx.artist_codes[index]) != artist for index in path[-ctx.spacing :]
    )


def valid_prepend(ctx: OrderingContext, path: Sequence[int], candidate: int) -> bool:
    artist = int(ctx.artist_codes[candidate])
    if artist < 0 or ctx.spacing <= 0:
        return True
    return all(
        int(ctx.artist_codes[index]) != artist for index in path[: ctx.spacing]
    )


def spacing_violations(ctx: OrderingContext, order: Sequence[int]) -> int:
    violations = 0
    for position, current in enumerate(order):
        artist = int(ctx.artist_codes[current])
        if artist < 0:
            continue
        for earlier in order[max(0, position - ctx.spacing) : position]:
            if int(ctx.artist_codes[earlier]) == artist:
                violations += 1
    return violations


def max_artist_violations(ctx: OrderingContext, order: Sequence[int]) -> int:
    counts: dict[int, int] = {}
    for index in order:
        artist = int(ctx.artist_codes[index])
        if artist >= 0:
            counts[artist] = counts.get(artist, 0) + 1
    return sum(max(0, count - ctx.max_per_artist) for count in counts.values())


def remaining_spacing_bound(
    ctx: OrderingContext,
    remaining_mask: int,
    history: Sequence[int],
) -> bool:
    """Necessary capacity bound for a spacing-constrained suffix.

    It catches the common greedy-tail failure early. The recursive search still proves
    feasibility by finding a complete suffix; this bound is not treated as sufficient.
    """
    if remaining_mask == 0 or ctx.spacing <= 0:
        return True
    counts: dict[int, int] = {}
    remaining_length = remaining_mask.bit_count()
    mask = remaining_mask
    while mask:
        bit = mask & -mask
        index = bit.bit_length() - 1
        artist = int(ctx.artist_codes[index])
        if artist >= 0:
            counts[artist] = counts.get(artist, 0) + 1
        mask ^= bit

    recent = [int(ctx.artist_codes[index]) for index in history[-ctx.spacing :]]
    for artist, count in counts.items():
        wait = 0
        for distance_back, recent_artist in enumerate(reversed(recent), start=1):
            if recent_artist == artist:
                wait = ctx.spacing - distance_back + 1
                break
        if remaining_length <= wait:
            capacity = 0
        else:
            capacity = 1 + (remaining_length - wait - 1) // (ctx.spacing + 1)
        if count > capacity:
            return False
    return True


def current_order(ctx: OrderingContext) -> AlgorithmResult:
    return AlgorithmResult(tuple(range(ctx.count)), {})


def _one_ended_search(
    ctx: OrderingContext,
    option_key: Callable[[Sequence[int], int, int], tuple[object, ...]],
) -> AlgorithmResult:
    start = seed_ranked(ctx)[0]
    path = [start]
    full_mask = (1 << ctx.count) - 1
    remaining_mask = full_mask ^ (1 << start)
    failed: set[tuple[int, tuple[int, ...]]] = set()
    attempts = 0
    backtracks = 0

    def search(mask: int) -> bool:
        nonlocal attempts, backtracks
        if mask == 0:
            return True
        history_codes = tuple(
            int(ctx.artist_codes[index]) for index in path[-ctx.spacing :]
        )
        state = (mask, history_codes)
        if state in failed:
            return False

        candidates: list[int] = []
        scan = mask
        while scan:
            bit = scan & -scan
            candidates.append(bit.bit_length() - 1)
            scan ^= bit
        candidates.sort(key=lambda index: option_key(path, mask, index))

        for candidate in candidates:
            if not valid_append(ctx, path, candidate):
                continue
            attempts += 1
            next_mask = mask ^ (1 << candidate)
            path.append(candidate)
            if remaining_spacing_bound(ctx, next_mask, path) and search(next_mask):
                return True
            path.pop()
            backtracks += 1
        failed.add(state)
        return False

    if not remaining_spacing_bound(ctx, remaining_mask, path) or not search(remaining_mask):
        raise RuntimeError("no complete artist-valid one-ended permutation exists")
    return AlgorithmResult(
        tuple(path),
        {
            "search_attempts": attempts,
            "backtracks": backtracks,
            "failed_states": len(failed),
        },
    )


def constrained_ham1(ctx: OrderingContext) -> AlgorithmResult:
    def option_key(path: Sequence[int], _mask: int, candidate: int) -> tuple[object, ...]:
        return (
            -float(ctx.pair_cosines[path[-1], candidate]),
            int(ctx.track_ids[candidate]),
        )

    return _one_ended_search(ctx, option_key)


def seed_frontier(ctx: OrderingContext, width: int = DEFAULT_FRONTIER) -> AlgorithmResult:
    if width <= 0:
        raise ValueError("frontier width must be positive")
    seed_order = seed_ranked(ctx)
    seed_position = {index: position for position, index in enumerate(seed_order)}

    def option_key(path: Sequence[int], mask: int, candidate: int) -> tuple[object, ...]:
        remaining = [index for index in seed_order if mask & (1 << index)]
        frontier = set(remaining[:width])
        return (
            0 if candidate in frontier else 1,
            -float(ctx.pair_cosines[path[-1], candidate]),
            seed_position[candidate],
            int(ctx.track_ids[candidate]),
        )

    result = _one_ended_search(ctx, option_key)
    remaining = set(range(ctx.count))
    expansions = 0
    for candidate in result.order:
        frontier = set(sorted(remaining, key=lambda index: seed_position[index])[:width])
        if candidate not in frontier:
            expansions += 1
        remaining.remove(candidate)
    diagnostics = dict(result.diagnostics)
    diagnostics["frontier_width"] = width
    diagnostics["outside_frontier_choices"] = expansions
    return AlgorithmResult(result.order, diagnostics)


def constrained_ham2(ctx: OrderingContext) -> AlgorithmResult:
    start = seed_ranked(ctx)[0]
    path = [start]
    remaining_mask = ((1 << ctx.count) - 1) ^ (1 << start)
    failed: set[tuple[int, tuple[int, ...], tuple[int, ...]]] = set()
    attempts = 0
    backtracks = 0

    def search(mask: int) -> bool:
        nonlocal attempts, backtracks
        if mask == 0:
            return True
        state = (
            mask,
            tuple(int(ctx.artist_codes[index]) for index in path[: ctx.spacing]),
            tuple(int(ctx.artist_codes[index]) for index in path[-ctx.spacing :]),
        )
        if state in failed:
            return False

        options: list[tuple[int, int]] = []
        scan = mask
        while scan:
            bit = scan & -scan
            candidate = bit.bit_length() - 1
            if valid_append(ctx, path, candidate):
                options.append((candidate, 0))  # append/tail wins an exact side tie
            if valid_prepend(ctx, path, candidate):
                options.append((candidate, 1))
            scan ^= bit
        options.sort(
            key=lambda item: (
                -float(
                    ctx.pair_cosines[path[-1] if item[1] == 0 else path[0], item[0]]
                ),
                int(ctx.track_ids[item[0]]),
                item[1],
            )
        )

        for candidate, side in options:
            attempts += 1
            if side == 0:
                path.append(candidate)
            else:
                path.insert(0, candidate)
            if search(mask ^ (1 << candidate)):
                return True
            if side == 0:
                path.pop()
            else:
                path.pop(0)
            backtracks += 1
        failed.add(state)
        return False

    if not search(remaining_mask):
        raise RuntimeError("no complete artist-valid bidirectional permutation exists")

    reversed_for_seed = False
    first_cosine = float(ctx.seed_cosines[path[0]])
    last_cosine = float(ctx.seed_cosines[path[-1]])
    if last_cosine > first_cosine or (
        last_cosine == first_cosine
        and int(ctx.track_ids[path[-1]]) < int(ctx.track_ids[path[0]])
    ):
        path.reverse()
        reversed_for_seed = True
    return AlgorithmResult(
        tuple(path),
        {
            "search_attempts": attempts,
            "backtracks": backtracks,
            "failed_states": len(failed),
            "reversed_for_seed_endpoint": reversed_for_seed,
        },
    )


def path_angular_cost(ctx: OrderingContext, order: Sequence[int]) -> float:
    if len(order) < 2:
        return 0.0
    return float(
        sum(ctx.pair_angles[order[position - 1], order[position]] for position in range(1, len(order)))
    )


def seed_fixed_two_opt(ctx: OrderingContext) -> AlgorithmResult:
    initial = constrained_ham1(ctx)
    path = list(initial.order)
    iterations = 0
    evaluated_moves = 0

    while True:
        best_improvement = EPSILON
        best_path: list[int] | None = None
        best_ids: tuple[int, ...] | None = None
        length = len(path)
        for start in range(1, length - 1):
            for end in range(start + 1, length):
                evaluated_moves += 1
                old_cost = float(ctx.pair_angles[path[start - 1], path[start]])
                new_cost = float(ctx.pair_angles[path[start - 1], path[end]])
                if end + 1 < length:
                    old_cost += float(ctx.pair_angles[path[end], path[end + 1]])
                    new_cost += float(ctx.pair_angles[path[start], path[end + 1]])
                improvement = old_cost - new_cost
                if improvement + EPSILON < best_improvement:
                    continue
                candidate = path[:start] + list(reversed(path[start : end + 1])) + path[end + 1 :]
                if spacing_violations(ctx, candidate):
                    continue
                candidate_ids = tuple(int(ctx.track_ids[index]) for index in candidate)
                if improvement > best_improvement + EPSILON or (
                    abs(improvement - best_improvement) <= EPSILON
                    and (best_ids is None or candidate_ids < best_ids)
                ):
                    best_improvement = improvement
                    best_path = candidate
                    best_ids = candidate_ids
        if best_path is None:
            break
        path = best_path
        iterations += 1

    diagnostics = dict(initial.diagnostics)
    diagnostics.update(
        {
            "two_opt_iterations": iterations,
            "two_opt_evaluated_moves": evaluated_moves,
            "initial_angular_cost": path_angular_cost(ctx, initial.order),
            "final_angular_cost": path_angular_cost(ctx, path),
        }
    )
    return AlgorithmResult(tuple(path), diagnostics)


def validate_order(ctx: OrderingContext, order: Sequence[int]) -> dict[str, object]:
    expected = list(range(ctx.count))
    actual = list(order)
    exact_membership = sorted(actual) == expected
    duplicate_count = len(actual) - len(set(actual))
    spacing_count = spacing_violations(ctx, actual) if exact_membership else -1
    max_count = max_artist_violations(ctx, actual) if exact_membership else -1
    return {
        "complete": len(actual) == ctx.count,
        "exact_membership": exact_membership,
        "duplicate_count": duplicate_count,
        "artist_spacing_violations": spacing_count,
        "max_artist_violations": max_count,
    }


def sequence_metrics(ctx: OrderingContext, order: Sequence[int]) -> dict[str, object]:
    local = np.asarray(order, dtype=np.int64)
    seed_cosines = ctx.seed_cosines[local].astype(np.float64)
    direct_angles = np.arccos(np.clip(seed_cosines, -1.0, 1.0))
    if len(order) > 1:
        adjacent_cosines = np.asarray(
            [
                ctx.pair_cosines[order[position - 1], order[position]]
                for position in range(1, len(order))
            ],
            dtype=np.float64,
        )
        adjacent_angles = np.asarray(
            [
                ctx.pair_angles[order[position - 1], order[position]]
                for position in range(1, len(order))
            ],
            dtype=np.float64,
        )
    else:
        adjacent_cosines = np.asarray([], dtype=np.float64)
        adjacent_angles = np.asarray([], dtype=np.float64)

    seed_first_angle = float(direct_angles[0])
    cumulative_walk = seed_first_angle + np.concatenate(
        (np.asarray([0.0]), np.cumsum(adjacent_angles))
    )
    prefix_sum = np.cumsum(ctx.embeddings[local].astype(np.float64), axis=0)
    prefix_norms = np.linalg.norm(prefix_sum, axis=1)
    prefix_centroid_cosines = (
        prefix_sum @ ctx.seed_embedding.astype(np.float64)
    ) / np.maximum(prefix_norms, 1e-15)

    positions = np.arange(len(order), dtype=np.float64)
    if len(order) > 1:
        slope = float(np.polyfit(positions, seed_cosines, 1)[0] * (len(order) - 1))
        position_correlation = float(np.corrcoef(positions, seed_cosines)[0, 1])
        outward_fraction = float(np.mean(np.diff(direct_angles) >= 0.0))
    else:
        slope = 0.0
        position_correlation = 0.0
        outward_fraction = 0.0

    tail = adjacent_cosines[-min(10, adjacent_cosines.size) :]
    worst_position = int(np.argmin(adjacent_cosines)) + 1 if adjacent_cosines.size else None
    seed_order = seed_ranked(ctx)
    seed_rank = {index: rank + 1 for rank, index in enumerate(seed_order)}
    endpoint_direct = float(direct_angles[-1])
    total_including_seed = float(cumulative_walk[-1])

    return {
        "count": len(order),
        "membership_sha256": sha256_ids(sorted(int(ctx.track_ids[index]) for index in order)),
        "mean_adjacent_cosine": float(np.mean(adjacent_cosines)) if adjacent_cosines.size else None,
        "p05_adjacent_cosine": float(np.percentile(adjacent_cosines, 5)) if adjacent_cosines.size else None,
        "p10_adjacent_cosine": float(np.percentile(adjacent_cosines, 10)) if adjacent_cosines.size else None,
        "min_adjacent_cosine": float(np.min(adjacent_cosines)) if adjacent_cosines.size else None,
        "last10_mean_adjacent_cosine": float(np.mean(tail)) if tail.size else None,
        "last10_min_adjacent_cosine": float(np.min(tail)) if tail.size else None,
        "worst_transition_position": worst_position,
        "total_angular_path": float(np.sum(adjacent_angles)),
        "total_angular_path_including_seed": total_including_seed,
        "first_seed_cosine": float(seed_cosines[0]),
        "first_seed_rank_in_membership": int(seed_rank[order[0]]),
        "closest_seed_member_position": int(order.index(seed_order[0])) + 1,
        "first5_mean_seed_cosine": float(np.mean(seed_cosines[:5])),
        "last5_mean_seed_cosine": float(np.mean(seed_cosines[-5:])),
        "seed_cosine_position_slope": slope,
        "seed_cosine_position_correlation": position_correlation,
        "outward_step_fraction": outward_fraction,
        "endpoint_direct_seed_angle": endpoint_direct,
        "maximum_direct_seed_angle": float(np.max(direct_angles)),
        "path_to_endpoint_detour_ratio": (
            total_including_seed / endpoint_direct if endpoint_direct > 1e-12 else None
        ),
        "mean_prefix_centroid_seed_cosine": float(np.mean(prefix_centroid_cosines)),
        "minimum_prefix_centroid_seed_cosine": float(np.min(prefix_centroid_cosines)),
        "seed_cosines_by_position": seed_cosines.tolist(),
        "direct_seed_angles_by_position": direct_angles.tolist(),
        "cumulative_walk_angles_by_position": cumulative_walk.tolist(),
        "prefix_centroid_seed_cosines_by_position": prefix_centroid_cosines.tolist(),
        "adjacent_cosines": adjacent_cosines.tolist(),
    }


def track_label(library: queue_eval.Library, library_index: int) -> str:
    artist = library.artists[library_index] or "[unknown artist]"
    title = library.titles[library_index] or "[untitled]"
    return f"{artist} - {title}"


def variant_payload(
    ctx: OrderingContext,
    algorithm: Callable[[], AlgorithmResult],
) -> dict[str, object]:
    started = time.perf_counter_ns()
    first = algorithm()
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    second = algorithm()
    deterministic = first.order == second.order and first.diagnostics == second.diagnostics
    validation = validate_order(ctx, first.order)
    if not deterministic:
        raise AssertionError("ordering variant is not deterministic")
    if not all(
        (
            validation["complete"],
            validation["exact_membership"],
            validation["duplicate_count"] == 0,
            validation["artist_spacing_violations"] == 0,
            validation["max_artist_violations"] == 0,
        )
    ):
        raise AssertionError(f"invalid ordering: {validation}")
    return {
        "elapsed_ms": elapsed_ms,
        "deterministic_repeat": deterministic,
        "diagnostics": first.diagnostics,
        "validation": validation,
        "metrics": sequence_metrics(ctx, first.order),
        "track_ids": [int(ctx.track_ids[index]) for index in first.order],
    }


def append_jsonl(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, ensure_ascii=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def completed_keys(path: Path) -> set[tuple[int, str]]:
    if not path.exists():
        return set()
    result: set[tuple[int, str]] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            row = json.loads(line)
            result.add((int(row["seed"]["track_id"]), str(row["selector"])))
    return result


def load_records(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize(records: Sequence[dict[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {"records": len(records), "selectors": {}}
    metric_keys = (
        "mean_adjacent_cosine",
        "p05_adjacent_cosine",
        "min_adjacent_cosine",
        "last10_mean_adjacent_cosine",
        "last10_min_adjacent_cosine",
        "total_angular_path",
        "total_angular_path_including_seed",
        "first_seed_cosine",
        "first5_mean_seed_cosine",
        "last5_mean_seed_cosine",
        "seed_cosine_position_slope",
        "seed_cosine_position_correlation",
        "outward_step_fraction",
        "path_to_endpoint_detour_ratio",
        "mean_prefix_centroid_seed_cosine",
    )
    for selector in sorted({str(record["selector"]) for record in records}):
        selector_records = [record for record in records if record["selector"] == selector]
        selector_summary: dict[str, object] = {"records": len(selector_records), "variants": {}}
        variants = sorted(selector_records[0]["variants"] if selector_records else [])
        for variant in variants:
            payloads = [record["variants"][variant] for record in selector_records]
            aggregate: dict[str, object] = {
                "elapsed_ms_median": percentile([float(item["elapsed_ms"]) for item in payloads], 50),
                "elapsed_ms_p95": percentile([float(item["elapsed_ms"]) for item in payloads], 95),
                "all_deterministic": all(bool(item["deterministic_repeat"]) for item in payloads),
                "all_memberships_exact": all(bool(item["validation"]["exact_membership"]) for item in payloads),
                "total_spacing_violations": sum(int(item["validation"]["artist_spacing_violations"]) for item in payloads),
            }
            for key in metric_keys:
                values = [
                    float(item["metrics"][key])
                    for item in payloads
                    if item["metrics"].get(key) is not None
                ]
                aggregate[f"{key}_mean"] = float(np.mean(values))
                aggregate[f"{key}_p05"] = percentile(values, 5)
                aggregate[f"{key}_p50"] = percentile(values, 50)
                aggregate[f"{key}_p95"] = percentile(values, 95)

            baseline = [record["variants"]["current_order"] for record in selector_records]
            delta_keys = (
                "mean_adjacent_cosine",
                "p05_adjacent_cosine",
                "min_adjacent_cosine",
                "last10_mean_adjacent_cosine",
                "last10_min_adjacent_cosine",
                "total_angular_path",
                "first_seed_cosine",
                "mean_prefix_centroid_seed_cosine",
            )
            for key in delta_keys:
                deltas = [
                    float(item["metrics"][key]) - float(base["metrics"][key])
                    for item, base in zip(payloads, baseline)
                ]
                aggregate[f"delta_vs_current_{key}_mean"] = float(np.mean(deltas))
                aggregate[f"delta_vs_current_{key}_p05"] = percentile(deltas, 5)
                aggregate[f"delta_vs_current_{key}_p50"] = percentile(deltas, 50)
                aggregate[f"delta_vs_current_{key}_p95"] = percentile(deltas, 95)
                aggregate[f"improved_vs_current_{key}_count"] = sum(delta > 0 for delta in deltas)
            selector_summary["variants"][variant] = aggregate
        result["selectors"][selector] = selector_summary
    return result


def write_qualitative(
    path: Path,
    library: queue_eval.Library,
    records: Sequence[dict[str, object]],
) -> None:
    id_to_index = {int(track_id): index for index, track_id in enumerate(library.track_ids)}

    def metric(record: dict[str, object], variant: str, name: str) -> float:
        return float(record["variants"][variant]["metrics"][name])

    cases: list[tuple[str, dict[str, object]]] = []
    for selector in ("mmr_0.4", "dpp"):
        rows = [record for record in records if record["selector"] == selector]
        cases.extend(
            (
                (
                    f"{selector}: largest 2-opt adjacency gain",
                    max(rows, key=lambda row: metric(row, "seed_fixed_2opt", "mean_adjacent_cosine") - metric(row, "current_order", "mean_adjacent_cosine")),
                ),
                (
                    f"{selector}: weakest 2-opt adjacency gain",
                    min(rows, key=lambda row: metric(row, "seed_fixed_2opt", "mean_adjacent_cosine") - metric(row, "current_order", "mean_adjacent_cosine")),
                ),
                (
                    f"{selector}: worst HAM-1 tail",
                    min(rows, key=lambda row: metric(row, "constrained_ham1", "last10_min_adjacent_cosine")),
                ),
                (
                    f"{selector}: largest HAM-2 opening loss",
                    min(rows, key=lambda row: metric(row, "constrained_ham2", "first_seed_cosine") - metric(row, "current_order", "first_seed_cosine")),
                ),
                (
                    f"{selector}: largest frontier adjacency tradeoff vs HAM-1",
                    min(rows, key=lambda row: metric(row, "seed_frontier_8", "mean_adjacent_cosine") - metric(row, "constrained_ham1", "mean_adjacent_cosine")),
                ),
            )
        )

    lines = [
        "# Ordering Qualitative Cases",
        "",
        "These are label-revealed inspection lists, not a listening verdict. Every variant",
        "in a case contains exactly the same tracks.",
        "",
    ]
    for title, record in cases:
        seed = record["seed"]
        lines.extend(
            [
                f"## {title}",
                "",
                f"Seed: {seed.get('artist') or '[unknown artist]'} - {seed.get('title') or '[untitled]'} "
                f"(track {seed['track_id']})",
                "",
            ]
        )
        for variant in (
            "current_order",
            "constrained_ham1",
            "constrained_ham2",
            "seed_fixed_2opt",
            "seed_frontier_8",
        ):
            payload = record["variants"][variant]
            metrics = payload["metrics"]
            lines.extend(
                [
                    f"### {variant}",
                    "",
                    f"Mean adjacent `{metrics['mean_adjacent_cosine']:.4f}`; p05 "
                    f"`{metrics['p05_adjacent_cosine']:.4f}`; minimum "
                    f"`{metrics['min_adjacent_cosine']:.4f}`; tail minimum "
                    f"`{metrics['last10_min_adjacent_cosine']:.4f}`; first-to-seed "
                    f"`{metrics['first_seed_cosine']:.4f}`.",
                    "",
                    "| # | Track | Seed cosine | Previous cosine |",
                    "|---:|---|---:|---:|",
                ]
            )
            seed_values = metrics["seed_cosines_by_position"]
            adjacent = metrics["adjacent_cosines"]
            for position, track_id in enumerate(payload["track_ids"]):
                index = id_to_index[int(track_id)]
                previous = "seed" if position == 0 else f"{float(adjacent[position - 1]):.4f}"
                label = track_label(library, index).replace("|", "\\|")
                lines.append(
                    f"| {position + 1} | {label} | {float(seed_values[position]):.4f} | {previous} |"
                )
            lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    library, db_hash = queue_eval.load_library(args.db, verify_hash=not args.skip_hash)
    pool_size = args.pool_size or max(100, int(library.count * 0.02))
    cohort = queue_eval.build_cohort(library, args.seeds)["ordinary"]
    output_file = args.output / "ordering.jsonl"
    if args.force and output_file.exists():
        output_file.unlink()
    done = completed_keys(output_file)

    selectors = ("mmr_0.4", "dpp")
    for seed_position, seed_index in enumerate(cohort, start=1):
        candidates, relevance = queue_eval.retrieve_candidates(library, seed_index, pool_size)
        for selector in selectors:
            seed_id = int(library.track_ids[seed_index])
            key = (seed_id, selector)
            if key in done:
                print(
                    f"ordering {seed_position}/{len(cohort)} {selector} seed={seed_id} resumed",
                    file=sys.stderr,
                )
                continue
            if selector == "mmr_0.4":
                membership, _ = queue_eval.select_mmr(
                    library,
                    candidates,
                    relevance,
                    args.queue_size,
                    lambda_=0.4,
                    constraint_aware=True,
                    max_per_artist=queue_eval.DEFAULT_MAX_PER_ARTIST,
                    min_spacing=queue_eval.DEFAULT_MIN_ARTIST_SPACING,
                )
            else:
                membership, _ = queue_eval.select_dpp(
                    library,
                    candidates,
                    relevance,
                    args.queue_size,
                    constraint_aware=True,
                    max_per_artist=queue_eval.DEFAULT_MAX_PER_ARTIST,
                    min_spacing=queue_eval.DEFAULT_MIN_ARTIST_SPACING,
                )
            if len(membership) != args.queue_size:
                raise RuntimeError(
                    f"{selector} seed {seed_id}: expected {args.queue_size} members, got {len(membership)}"
                )

            ctx = build_context(library, seed_index, membership)
            algorithms: tuple[tuple[str, Callable[[], AlgorithmResult]], ...] = (
                ("current_order", lambda: current_order(ctx)),
                ("constrained_ham1", lambda: constrained_ham1(ctx)),
                ("constrained_ham2", lambda: constrained_ham2(ctx)),
                ("seed_fixed_2opt", lambda: seed_fixed_two_opt(ctx)),
                ("seed_frontier_8", lambda: seed_frontier(ctx, DEFAULT_FRONTIER)),
            )
            variants: dict[str, object] = {}
            expected_membership_hash = sha256_ids(sorted(int(track_id) for track_id in ctx.track_ids))
            for name, algorithm in algorithms:
                payload = variant_payload(ctx, algorithm)
                if payload["metrics"]["membership_sha256"] != expected_membership_hash:
                    raise AssertionError(f"{name} changed membership hash")
                variants[name] = payload

            record: dict[str, object] = {
                "seed": queue_eval.track_summary(library, seed_index),
                "selector": selector,
                "pool_size": pool_size,
                "queue_size": args.queue_size,
                "membership_sha256": expected_membership_hash,
                "membership_track_ids_sorted": sorted(int(track_id) for track_id in ctx.track_ids),
                "variants": variants,
            }
            append_jsonl(output_file, record)
            print(
                f"ordering {seed_position}/{len(cohort)} {selector} seed={seed_id} saved",
                file=sys.stderr,
            )

    records = load_records(output_file)
    if len(records) != args.seeds * len(selectors):
        raise RuntimeError(f"expected {args.seeds * len(selectors)} records, found {len(records)}")
    atomic_json(args.output / "summary.json", summarize(records))
    write_qualitative(args.output / "qualitative-lists.md", library, records)

    cohort_ids = [int(library.track_ids[index]) for index in cohort]
    manifest = {
        "database": str(args.db.resolve()),
        "database_sha256": db_hash,
        "track_count": library.count,
        "embedding_dim": library.dim,
        "script_sha256": queue_eval.sha256_file(Path(__file__)),
        "cohort_salt": "phone-2026-07-07-ordinary-v1",
        "cohort_track_ids": cohort_ids,
        "cohort_sha256": sha256_ids(cohort_ids),
        "parameters": {
            "seeds": args.seeds,
            "queue_size": args.queue_size,
            "pool_size": pool_size,
            "mmr_lambda": 0.4,
            "frontier_width": DEFAULT_FRONTIER,
            "max_per_artist": queue_eval.DEFAULT_MAX_PER_ARTIST,
            "min_artist_spacing": queue_eval.DEFAULT_MIN_ARTIST_SPACING,
            "distance": "acos(clamped cosine)",
            "tie_break": "ascending track ID",
        },
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    atomic_json(args.output / "manifest.json", manifest)
    print(f"Complete. Results: {args.output}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=queue_eval.DEFAULT_DB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--queue-size", type=int, default=queue_eval.DEFAULT_QUEUE_SIZE)
    parser.add_argument("--pool-size", type=int, default=0, help="0 mirrors the app's 2%% rule")
    parser.add_argument("--skip-hash", action="store_true", help="development only")
    parser.add_argument("--force", action="store_true", help="discard the JSONL checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seeds <= 0 or args.queue_size <= 0:
        raise ValueError("--seeds and --queue-size must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    run(args)


if __name__ == "__main__":
    main()
