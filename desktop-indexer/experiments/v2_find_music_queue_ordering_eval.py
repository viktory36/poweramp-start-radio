#!/usr/bin/env python3
"""Evaluate Find Music playback order without changing the displayed result set.

Retrieval ranking and playback sequencing are separate contracts here.  Ranked preserves
the result order.  Smooth fixes relevance rank one as the opener, builds a nearest-next
path, and applies deterministic best-improvement 2-opt over angular embedding distance.
Shuffle is a reproducible hash permutation.  Every variant must remain an exact
permutation of the frozen active-domain membership.

Metadata is emitted only for labels and diagnostics.  It never changes membership or
playback order.  This script is host-only and does not connect to a device or Poweramp.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import v2_active_composition_eval as composition
import v2_queue_eval as queue_eval


REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_COMPOSITION_OUTPUT = (
    REPO_ROOT
    / "desktop-indexer"
    / "audit_raw_data"
    / "v2-discovery"
    / "active-composition-2026-07-15"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "desktop-indexer"
    / "audit_raw_data"
    / "v2-discovery"
    / "find-music-queue-ordering-2026-07-15"
)

EXPECTED_DB_SHA256 = queue_eval.EXPECTED_DB_SHA256
EXPECTED_ACTIVE_CATALOG_SHA256 = composition.EXPECTED_ACTIVE_CATALOG_SHA256
EXPECTED_ACTIVE_COUNT = composition.EXPECTED_ACTIVE_COUNT
PRIMARY_RESULT_COUNT = 20
RESULT_COUNTS: tuple[int, ...] = (20, 50, 100)
TIMING_REPEATS = 7
SMOOTH_EPSILON = 1e-12
NEAR_IDENTICAL_EMBEDDING_COSINE = 0.9999
SHUFFLE_SEED = "find-music-playback-shuffle-v1:2026-07-15"

SINGLE_TEXT_QUERIES: tuple[str, ...] = (
    "ambient",
    "sleep",
    "slow psychedelic",
    "psychedelic guitar",
    "rainy night",
)

# These deliberately span conjunction, union, a stricter conjunction candidate, a
# three-text recipe, a song/text recipe, and the experimental Refine contract.
COMPOSED_CASES: tuple[dict[str, object], ...] = (
    {
        "id": "all_of_slow_psychedelic",
        "source_case": "text_slow_plus_psychedelic",
        "source_kind": "operator",
        "source_variant": "all_of",
    },
    {
        "id": "either_slow_psychedelic",
        "source_case": "text_slow_plus_psychedelic",
        "source_kind": "operator",
        "source_variant": "either",
    },
    {
        "id": "all_of_psychedelic_guitar",
        "source_case": "text_psychedelic_plus_guitar",
        "source_kind": "operator",
        "source_variant": "all_of",
    },
    {
        "id": "strict_all_psychedelic_guitar",
        "source_case": "text_psychedelic_plus_guitar",
        "source_kind": "operator",
        "source_variant": "strict_all",
    },
    {
        "id": "all_of_slow_psychedelic_guitar",
        "source_case": "text_slow_psychedelic_guitar",
        "source_kind": "operator",
        "source_variant": "all_of",
    },
    {
        "id": "all_of_sleep_ambient",
        "source_case": "text_sleep_plus_ambient",
        "source_kind": "operator",
        "source_variant": "all_of",
    },
    {
        "id": "all_of_bonobo_ambient",
        "source_case": "song_bonobo_with_ambient",
        "source_kind": "operator",
        "source_variant": "all_of",
    },
    {
        "id": "either_bonobo_kailash",
        "source_case": "song_bonobo_or_kailash",
        "source_kind": "operator",
        "source_variant": "either",
    },
    {
        "id": "refine_slow_with_psychedelic_0_5pct",
        "source_case": "refine_slow_with_psychedelic",
        "source_kind": "refine",
        "source_variant": 0.005,
    },
)


@dataclass(frozen=True)
class RetrievalCase:
    case_id: str
    family: str
    display_label: str
    retrieval_contract: str
    ranked_track_ids: tuple[int, ...]
    source: dict[str, object]


@dataclass(frozen=True)
class OrderingContext:
    case: RetrievalCase
    count: int
    library_indices: np.ndarray
    track_ids: np.ndarray
    embeddings: np.ndarray
    pair_cosines: np.ndarray
    pair_angles: np.ndarray
    artists: tuple[str | None, ...]
    titles: tuple[str | None, ...]


@dataclass(frozen=True)
class OrderResult:
    order: tuple[int, ...]
    diagnostics: dict[str, object]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_track_ids(track_ids: Iterable[int], preserve_order: bool = True) -> str:
    values = list(int(track_id) for track_id in track_ids)
    if not preserve_order:
        values.sort()
    digest = hashlib.sha256()
    for track_id in values:
        digest.update(track_id.to_bytes(8, byteorder="little", signed=True))
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def normalized_label(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = " ".join(value.casefold().split())
    return normalized or None


def find_operator_record(case: dict[str, object], operator: str) -> dict[str, object]:
    for record in case["operators"]:
        if record["operator"] == operator and record["status"] == "ranked":
            return record
    raise KeyError(f"{case['id']} has no ranked {operator} record")


def load_retrieval_cases(
    rankings: dict[str, object],
    library: composition.ActiveLibrary,
    cache: composition.AnchorCache,
) -> list[RetrievalCase]:
    result: list[RetrievalCase] = []
    for query in SINGLE_TEXT_QUERIES:
        control = composition.direct_text_control(library, cache, query)
        result.append(
            RetrievalCase(
                case_id=f"text_{query.replace(' ', '_')}",
                family="single_text",
                display_label=query,
                retrieval_contract="raw cosine to the exact phone text embedding",
                ranked_track_ids=tuple(int(value) for value in control["top100_track_ids"]),
                source={
                    "query": query,
                    "embedding_sha256": control["embedding_sha256"],
                },
            )
        )

    cases_by_id = {str(case["id"]): case for case in rankings["cases"]}
    refinements_by_id = {
        str(record["id"]): record for record in rankings["refinements"]
    }
    for definition in COMPOSED_CASES:
        source_case = str(definition["source_case"])
        source_kind = str(definition["source_kind"])
        source_variant = definition["source_variant"]
        if source_kind == "operator":
            case = cases_by_id[source_case]
            record = find_operator_record(case, str(source_variant))
            ids = tuple(int(value) for value in record["top100_track_ids"])
            labels = [str(anchor["label"]) for anchor in record["anchors"]]
            display_label = f"{source_variant}: " + " + ".join(labels)
            contract = {
                "all_of": "weighted geometric mean of effective active-domain percentiles",
                "either": "deterministic weighted-prefix union of positive branches",
                "strict_all": "maximize the worst effective ingredient percentile",
                "direction": "cosine to the normalized signed weighted centroid",
            }[str(source_variant)]
            source = {
                "active_composition_case": source_case,
                "operator": source_variant,
                "top100_track_id_sha256": sha256_track_ids(ids),
            }
        elif source_kind == "refine":
            refinement = refinements_by_id[source_case]
            width = next(
                item
                for item in refinement["widths"]
                if abs(float(item["primary_fraction"]) - float(source_variant)) < 1e-12
            )
            ids = tuple(int(row["track_id"]) for row in width["top30"])
            display_label = (
                f"Refine {refinement['primary']['label']} with "
                f"{refinement['secondary']['label']} at {100 * float(source_variant):g}%"
            )
            contract = "rank by B inside A's explicit active-domain percentile neighborhood"
            source = {
                "active_composition_refinement": source_case,
                "primary_fraction": source_variant,
                "top30_track_id_sha256": sha256_track_ids(ids),
            }
        else:
            raise ValueError(f"unknown source kind: {source_kind}")
        result.append(
            RetrievalCase(
                case_id=str(definition["id"]),
                family="composed",
                display_label=display_label,
                retrieval_contract=contract,
                ranked_track_ids=ids,
                source=source,
            )
        )
    if len({case.case_id for case in result}) != len(result):
        raise AssertionError("retrieval case IDs are not unique")
    return result


def build_context(
    case: RetrievalCase,
    count: int,
    library: composition.ActiveLibrary,
    position_by_track_id: dict[int, int],
) -> OrderingContext:
    if len(case.ranked_track_ids) < count:
        raise ValueError(f"{case.case_id} has only {len(case.ranked_track_ids)} ranked rows")
    ranked_ids = np.asarray(case.ranked_track_ids[:count], dtype=np.int64)
    if np.unique(ranked_ids).size != count:
        raise AssertionError(f"{case.case_id} membership contains duplicate track IDs")
    indices = np.asarray(
        [position_by_track_id[int(track_id)] for track_id in ranked_ids], dtype=np.int64
    )
    embeddings = library.embeddings[indices]
    pair_cosines = np.clip(embeddings @ embeddings.T, -1.0, 1.0)
    pair_angles = np.arccos(pair_cosines.astype(np.float64))
    return OrderingContext(
        case=case,
        count=count,
        library_indices=indices,
        track_ids=ranked_ids,
        embeddings=embeddings,
        pair_cosines=pair_cosines,
        pair_angles=pair_angles,
        artists=tuple(library.artists[int(index)] for index in indices),
        titles=tuple(library.titles[int(index)] for index in indices),
    )


def ranked_order(ctx: OrderingContext) -> OrderResult:
    return OrderResult(tuple(range(ctx.count)), {"contract": "displayed relevance order"})


def smooth_order(ctx: OrderingContext) -> OrderResult:
    path = [0]
    remaining = set(range(1, ctx.count))
    while remaining:
        previous = path[-1]
        candidate = min(
            remaining,
            key=lambda index: (
                float(ctx.pair_angles[previous, index]),
                index,
                int(ctx.track_ids[index]),
            ),
        )
        path.append(candidate)
        remaining.remove(candidate)

    initial_path = tuple(path)
    initial_cost = path_angular_cost(ctx, initial_path)
    iterations = 0
    evaluated_moves = 0
    while True:
        best_improvement = SMOOTH_EPSILON
        best_move: tuple[int, int] | None = None
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
                move = (start, end)
                if improvement > best_improvement + SMOOTH_EPSILON or (
                    abs(improvement - best_improvement) <= SMOOTH_EPSILON
                    and improvement > SMOOTH_EPSILON
                    and (best_move is None or move < best_move)
                ):
                    best_improvement = improvement
                    best_move = move
        if best_move is None:
            break
        start, end = best_move
        path[start : end + 1] = reversed(path[start : end + 1])
        iterations += 1

    return OrderResult(
        tuple(path),
        {
            "contract": (
                "fix relevance rank 1, nearest-next initialization, then deterministic "
                "best-improvement 2-opt minimizing total angular path"
            ),
            "initial_order_track_id_sha256": sha256_track_ids(
                int(ctx.track_ids[index]) for index in initial_path
            ),
            "initial_angular_path": initial_cost,
            "final_angular_path": path_angular_cost(ctx, path),
            "two_opt_iterations": iterations,
            "two_opt_evaluated_moves": evaluated_moves,
        },
    )


def shuffle_order(ctx: OrderingContext) -> OrderResult:
    membership_hash = sha256_track_ids(ctx.track_ids, preserve_order=False)

    def key(index: int) -> tuple[bytes, int]:
        material = (
            f"{SHUFFLE_SEED}\0{ctx.case.case_id}\0{ctx.count}\0"
            f"{membership_hash}\0{int(ctx.track_ids[index])}"
        ).encode("utf-8")
        return hashlib.sha256(material).digest(), int(ctx.track_ids[index])

    return OrderResult(
        tuple(sorted(range(ctx.count), key=key)),
        {
            "contract": "reproducible SHA-256 hash permutation",
            "shuffle_seed": SHUFFLE_SEED,
        },
    )


def path_angular_cost(ctx: OrderingContext, order: Sequence[int]) -> float:
    return float(
        sum(
            ctx.pair_angles[order[position - 1], order[position]]
            for position in range(1, len(order))
        )
    )


def validate_order(ctx: OrderingContext, order: Sequence[int]) -> dict[str, object]:
    values = tuple(int(value) for value in order)
    exact = sorted(values) == list(range(ctx.count))
    ordered_ids = [int(ctx.track_ids[index]) for index in values]
    return {
        "complete": len(values) == ctx.count,
        "exact_membership": exact,
        "duplicate_local_indices": len(values) - len(set(values)),
        "membership_sha256": sha256_track_ids(ordered_ids, preserve_order=False),
        "play_order_sha256": sha256_track_ids(ordered_ids),
        "rank_one_position": values.index(0) + 1,
    }


def inversion_count(values: Sequence[int]) -> int:
    return sum(
        values[first] > values[second]
        for first in range(len(values))
        for second in range(first + 1, len(values))
    )


def order_comparison(
    first_track_ids: Sequence[int],
    second_track_ids: Sequence[int],
) -> dict[str, object]:
    if len(first_track_ids) != len(second_track_ids) or set(first_track_ids) != set(
        second_track_ids
    ):
        raise ValueError("order comparison requires one exact membership")
    second_position = {
        int(track_id): position for position, track_id in enumerate(second_track_ids)
    }
    relative = [second_position[int(track_id)] for track_id in first_track_ids]
    count = len(relative)
    denominator = count * (count - 1)
    tau = (
        1.0 - (4.0 * inversion_count(relative) / denominator)
        if denominator
        else 1.0
    )
    first_directed_edges = set(zip(first_track_ids, first_track_ids[1:]))
    second_directed_edges = set(zip(second_track_ids, second_track_ids[1:]))
    first_undirected_edges = {frozenset(edge) for edge in first_directed_edges}
    second_undirected_edges = {frozenset(edge) for edge in second_directed_edges}

    def jaccard(first: set[object], second: set[object]) -> float:
        union = first | second
        return len(first & second) / len(union) if union else 1.0

    return {
        "same_position_count": sum(
            int(first) == int(second)
            for first, second in zip(first_track_ids, second_track_ids)
        ),
        "same_position_fraction": sum(
            int(first) == int(second)
            for first, second in zip(first_track_ids, second_track_ids)
        )
        / count,
        "kendall_tau": tau,
        "directed_edge_jaccard": jaccard(first_directed_edges, second_directed_edges),
        "undirected_edge_jaccard": jaccard(
            first_undirected_edges, second_undirected_edges
        ),
    }


def longest_artist_run(ctx: OrderingContext, order: Sequence[int]) -> int:
    longest = 0
    current = 0
    previous: str | None = None
    for local_index in order:
        artist = normalized_label(ctx.artists[local_index])
        if artist is not None and artist == previous:
            current += 1
        elif artist is not None:
            current = 1
        else:
            current = 0
        longest = max(longest, current)
        previous = artist
    return longest


def transition_record(
    ctx: OrderingContext,
    order: Sequence[int],
    transition_index: int,
    cosine: float,
) -> dict[str, object]:
    before = order[transition_index]
    after = order[transition_index + 1]
    return {
        "after_play_position": transition_index + 2,
        "from_track_id": int(ctx.track_ids[before]),
        "to_track_id": int(ctx.track_ids[after]),
        "from_relevance_rank": before + 1,
        "to_relevance_rank": after + 1,
        "cosine": cosine,
    }


def sequence_metrics(
    ctx: OrderingContext,
    order: Sequence[int],
    ranked_p10_threshold: float,
) -> dict[str, object]:
    local = np.asarray(order, dtype=np.int64)
    ranks = local + 1
    positions = np.arange(1, ctx.count + 1, dtype=np.int64)
    displacements = np.abs(ranks - positions)
    if ctx.count > 1:
        adjacent = np.asarray(
            [
                ctx.pair_cosines[order[index], order[index + 1]]
                for index in range(ctx.count - 1)
            ],
            dtype=np.float64,
        )
    else:
        adjacent = np.asarray([], dtype=np.float64)

    tail_count = min(10, adjacent.size)
    tail = adjacent[-tail_count:]
    inversions = inversion_count(ranks.tolist())
    denominator = ctx.count * (ctx.count - 1)
    kendall_tau = 1.0 - (4.0 * inversions / denominator) if denominator else 1.0
    spearman = (
        float(np.corrcoef(positions.astype(float), ranks.astype(float))[0, 1])
        if ctx.count > 1
        else 1.0
    )

    same_artist = 0
    same_recording_label = 0
    near_identical = 0
    for position in range(ctx.count - 1):
        before = order[position]
        after = order[position + 1]
        before_artist = normalized_label(ctx.artists[before])
        after_artist = normalized_label(ctx.artists[after])
        if before_artist is not None and before_artist == after_artist:
            same_artist += 1
            before_title = normalized_label(ctx.titles[before])
            after_title = normalized_label(ctx.titles[after])
            if before_title is not None and before_title == after_title:
                same_recording_label += 1
        if adjacent[position] >= NEAR_IDENTICAL_EMBEDDING_COSINE:
            near_identical += 1

    worst_indices = (
        np.argsort(adjacent, kind="stable")[: min(5, adjacent.size)].tolist()
        if adjacent.size
        else []
    )
    last_indices = list(range(max(0, adjacent.size - tail_count), adjacent.size))
    top5_count = min(5, ctx.count)
    top10_count = min(10, ctx.count)
    return {
        "mean_adjacent_cosine": float(np.mean(adjacent)) if adjacent.size else None,
        "median_adjacent_cosine": float(np.median(adjacent)) if adjacent.size else None,
        "p05_adjacent_cosine": float(np.percentile(adjacent, 5)) if adjacent.size else None,
        "p10_adjacent_cosine": float(np.percentile(adjacent, 10)) if adjacent.size else None,
        "min_adjacent_cosine": float(np.min(adjacent)) if adjacent.size else None,
        "total_angular_path": path_angular_cost(ctx, order),
        "last10_mean_adjacent_cosine": float(np.mean(tail)) if tail.size else None,
        "last10_min_adjacent_cosine": float(np.min(tail)) if tail.size else None,
        "tail_transitions_below_ranked_p10": int(np.sum(tail < ranked_p10_threshold)),
        "worst_transition_in_last10": bool(
            adjacent.size and int(np.argmin(adjacent)) >= adjacent.size - tail_count
        ),
        "worst_transitions": [
            transition_record(ctx, order, index, float(adjacent[index]))
            for index in worst_indices
        ],
        "last10_transitions": [
            transition_record(ctx, order, index, float(adjacent[index]))
            for index in last_indices
        ],
        "mean_absolute_relevance_rank_displacement": float(np.mean(displacements)),
        "median_absolute_relevance_rank_displacement": float(np.median(displacements)),
        "p95_absolute_relevance_rank_displacement": float(
            np.percentile(displacements, 95)
        ),
        "max_absolute_relevance_rank_displacement": int(np.max(displacements)),
        "kendall_tau_with_relevance_rank": kendall_tau,
        "spearman_with_relevance_rank": spearman,
        "top5_prefix_retention": (
            len(set(order[:top5_count]) & set(range(top5_count))) / top5_count
        ),
        "top10_prefix_retention": (
            len(set(order[:top10_count]) & set(range(top10_count))) / top10_count
        ),
        "mean_relevance_rank_first5": float(np.mean(ranks[:top5_count])),
        "mean_relevance_rank_first10": float(np.mean(ranks[:top10_count])),
        "adjacent_same_artist_count": same_artist,
        "adjacent_same_artist_title_count": same_recording_label,
        "adjacent_near_identical_embedding_count": near_identical,
        "longest_same_artist_run": longest_artist_run(ctx, order),
        "relevance_ranks_by_play_position": ranks.tolist(),
        "adjacent_cosines": adjacent.tolist(),
    }


def membership_diagnostics(ctx: OrderingContext) -> dict[str, object]:
    artist_title_groups: dict[tuple[str, str], list[int]] = {}
    for index, (artist, title) in enumerate(zip(ctx.artists, ctx.titles)):
        key_artist = normalized_label(artist)
        key_title = normalized_label(title)
        if key_artist is None or key_title is None:
            continue
        artist_title_groups.setdefault((key_artist, key_title), []).append(index)
    repeated_labels = [values for values in artist_title_groups.values() if len(values) > 1]
    near_pairs: list[dict[str, object]] = []
    for first in range(ctx.count):
        for second in range(first + 1, ctx.count):
            cosine = float(ctx.pair_cosines[first, second])
            if cosine >= NEAR_IDENTICAL_EMBEDDING_COSINE:
                near_pairs.append(
                    {
                        "first_relevance_rank": first + 1,
                        "second_relevance_rank": second + 1,
                        "first_track_id": int(ctx.track_ids[first]),
                        "second_track_id": int(ctx.track_ids[second]),
                        "cosine": cosine,
                    }
                )
    return {
        "distinct_display_artists": len(
            {value for value in map(normalized_label, ctx.artists) if value is not None}
        ),
        "repeated_artist_title_group_count": len(repeated_labels),
        "tracks_in_repeated_artist_title_groups": sum(len(values) for values in repeated_labels),
        "repeated_artist_title_groups": [
            {
                "relevance_ranks": [index + 1 for index in values],
                "track_ids": [int(ctx.track_ids[index]) for index in values],
                "label": f"{ctx.artists[values[0]]} - {ctx.titles[values[0]]}",
            }
            for values in repeated_labels
        ],
        "near_identical_embedding_pair_count": len(near_pairs),
        "near_identical_embedding_pairs": near_pairs,
        "identity_warning": (
            "same artist/title and >=0.9999 cosine are diagnostics, not proof that files "
            "are the same decoded recording"
        ),
    }


def measure_variant(
    ctx: OrderingContext,
    algorithm: Callable[[OrderingContext], OrderResult],
    ranked_p10_threshold: float,
) -> dict[str, object]:
    runs: list[OrderResult] = []
    elapsed_ms: list[float] = []
    for _ in range(TIMING_REPEATS):
        started = time.perf_counter_ns()
        result = algorithm(ctx)
        elapsed_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
        runs.append(result)
    first = runs[0]
    deterministic = all(
        result.order == first.order and result.diagnostics == first.diagnostics
        for result in runs[1:]
    )
    validation = validate_order(ctx, first.order)
    if not deterministic:
        raise AssertionError(f"{ctx.case.case_id}: ordering was not deterministic")
    if not (
        validation["complete"]
        and validation["exact_membership"]
        and validation["duplicate_local_indices"] == 0
    ):
        raise AssertionError(f"{ctx.case.case_id}: invalid permutation {validation}")
    return {
        "diagnostics": first.diagnostics,
        "deterministic_across_repeats": deterministic,
        "timing_ms": {
            "repeats": TIMING_REPEATS,
            "minimum": min(elapsed_ms),
            "median": statistics.median(elapsed_ms),
            "p95": float(np.percentile(np.asarray(elapsed_ms), 95)),
            "samples": elapsed_ms,
        },
        "validation": validation,
        "metrics": sequence_metrics(ctx, first.order, ranked_p10_threshold),
        "track_ids": [int(ctx.track_ids[index]) for index in first.order],
    }


def evaluate_context(ctx: OrderingContext) -> dict[str, object]:
    ranked = ranked_order(ctx)
    ranked_adjacent = np.asarray(
        [
            ctx.pair_cosines[ranked.order[index], ranked.order[index + 1]]
            for index in range(ctx.count - 1)
        ],
        dtype=np.float64,
    )
    ranked_p10 = float(np.percentile(ranked_adjacent, 10))
    algorithms: tuple[tuple[str, Callable[[OrderingContext], OrderResult]], ...] = (
        ("ranked", ranked_order),
        ("smooth", smooth_order),
        ("shuffle", shuffle_order),
    )
    variants = {
        name: measure_variant(ctx, algorithm, ranked_p10)
        for name, algorithm in algorithms
    }
    membership_hashes = {
        str(payload["validation"]["membership_sha256"])
        for payload in variants.values()
    }
    if len(membership_hashes) != 1:
        raise AssertionError("ordering variants changed membership")
    pairwise: list[dict[str, object]] = []
    names = tuple(variants)
    for first_position, first in enumerate(names):
        for second in names[first_position + 1 :]:
            pairwise.append(
                {
                    "first": first,
                    "second": second,
                    **order_comparison(
                        variants[first]["track_ids"], variants[second]["track_ids"]
                    ),
                }
            )
    return {
        "case_id": ctx.case.case_id,
        "family": ctx.case.family,
        "display_label": ctx.case.display_label,
        "retrieval_contract": ctx.case.retrieval_contract,
        "retrieval_source": ctx.case.source,
        "result_count": ctx.count,
        "ranked_membership_sha256": sha256_track_ids(
            ctx.track_ids, preserve_order=False
        ),
        "ranked_order_sha256": sha256_track_ids(ctx.track_ids),
        "ranked_track_ids": [int(value) for value in ctx.track_ids],
        "ranked_p10_transition_threshold": ranked_p10,
        "membership_diagnostics": membership_diagnostics(ctx),
        "variants": variants,
        "pairwise_order_comparisons": pairwise,
    }


def mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def summarize(records: Sequence[dict[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {"records": len(records), "groups": {}}
    group_keys = sorted(
        {
            (str(record["family"]), int(record["result_count"]))
            for record in records
        }
    )
    for family, count in group_keys:
        rows = [
            record
            for record in records
            if record["family"] == family and record["result_count"] == count
        ]
        variants: dict[str, object] = {}
        for variant in ("ranked", "smooth", "shuffle"):
            payloads = [row["variants"][variant] for row in rows]
            variants[variant] = {
                "all_exact_membership": all(
                    bool(payload["validation"]["exact_membership"])
                    for payload in payloads
                ),
                "all_deterministic": all(
                    bool(payload["deterministic_across_repeats"])
                    for payload in payloads
                ),
                "median_of_case_median_runtime_ms": float(
                    np.median([payload["timing_ms"]["median"] for payload in payloads])
                ),
                "p95_of_case_median_runtime_ms": float(
                    np.percentile(
                        [payload["timing_ms"]["median"] for payload in payloads], 95
                    )
                ),
                "mean_adjacent_cosine": mean(
                    [payload["metrics"]["mean_adjacent_cosine"] for payload in payloads]
                ),
                "mean_p05_adjacent_cosine": mean(
                    [payload["metrics"]["p05_adjacent_cosine"] for payload in payloads]
                ),
                "mean_min_adjacent_cosine": mean(
                    [payload["metrics"]["min_adjacent_cosine"] for payload in payloads]
                ),
                "mean_last10_min_adjacent_cosine": mean(
                    [payload["metrics"]["last10_min_adjacent_cosine"] for payload in payloads]
                ),
                "tail_transitions_below_ranked_p10": sum(
                    int(payload["metrics"]["tail_transitions_below_ranked_p10"])
                    for payload in payloads
                ),
                "mean_absolute_relevance_rank_displacement": mean(
                    [
                        payload["metrics"][
                            "mean_absolute_relevance_rank_displacement"
                        ]
                        for payload in payloads
                    ]
                ),
                "mean_top10_prefix_retention": mean(
                    [payload["metrics"]["top10_prefix_retention"] for payload in payloads]
                ),
                "mean_kendall_tau_with_relevance_rank": mean(
                    [
                        payload["metrics"]["kendall_tau_with_relevance_rank"]
                        for payload in payloads
                    ]
                ),
                "mean_rank_one_position": mean(
                    [payload["validation"]["rank_one_position"] for payload in payloads]
                ),
                "total_adjacent_same_artist": sum(
                    int(payload["metrics"]["adjacent_same_artist_count"])
                    for payload in payloads
                ),
                "total_adjacent_same_artist_title": sum(
                    int(payload["metrics"]["adjacent_same_artist_title_count"])
                    for payload in payloads
                ),
                "total_adjacent_near_identical_embedding": sum(
                    int(
                        payload["metrics"]["adjacent_near_identical_embedding_count"]
                    )
                    for payload in payloads
                ),
            }
        ranked_payloads = [row["variants"]["ranked"] for row in rows]
        for variant in ("smooth", "shuffle"):
            payloads = [row["variants"][variant] for row in rows]
            variants[variant]["mean_delta_vs_ranked_mean_adjacent_cosine"] = mean(
                [
                    payload["metrics"]["mean_adjacent_cosine"]
                    - baseline["metrics"]["mean_adjacent_cosine"]
                    for payload, baseline in zip(payloads, ranked_payloads)
                ]
            )
            variants[variant]["improved_mean_adjacency_case_count"] = sum(
                payload["metrics"]["mean_adjacent_cosine"]
                > baseline["metrics"]["mean_adjacent_cosine"]
                for payload, baseline in zip(payloads, ranked_payloads)
            )
            variants[variant]["improved_tail_min_case_count"] = sum(
                payload["metrics"]["last10_min_adjacent_cosine"]
                > baseline["metrics"]["last10_min_adjacent_cosine"]
                for payload, baseline in zip(payloads, ranked_payloads)
            )
        pairwise: dict[str, object] = {}
        for first, second in (
            ("ranked", "smooth"),
            ("ranked", "shuffle"),
            ("smooth", "shuffle"),
        ):
            comparisons = [
                next(
                    comparison
                    for comparison in row["pairwise_order_comparisons"]
                    if comparison["first"] == first and comparison["second"] == second
                )
                for row in rows
            ]
            pairwise[f"{first}_vs_{second}"] = {
                "mean_same_position_fraction": mean(
                    [comparison["same_position_fraction"] for comparison in comparisons]
                ),
                "mean_kendall_tau": mean(
                    [comparison["kendall_tau"] for comparison in comparisons]
                ),
                "mean_undirected_edge_jaccard": mean(
                    [comparison["undirected_edge_jaccard"] for comparison in comparisons]
                ),
            }
        result["groups"][f"{family}:{count}"] = {
            "case_count": len(rows),
            "variants": variants,
            "pairwise": pairwise,
        }
    return result


def label(ctx: OrderingContext, local_index: int) -> str:
    artist = ctx.artists[local_index] or "[unknown artist]"
    title = ctx.titles[local_index] or "[untitled]"
    return f"{artist} - {title}".replace("|", "\\|")


def write_qualitative_lists(
    path: Path,
    contexts: dict[tuple[str, int], OrderingContext],
    records: Sequence[dict[str, object]],
) -> None:
    primary = [record for record in records if record["result_count"] == PRIMARY_RESULT_COUNT]
    lines = [
        "# Find Music Queue Ordering: Complete Label-Revealed Lists",
        "",
        "Each case is the exact 20-track displayed membership. Metadata is shown only so a",
        "human can inspect the lists; no label influenced retrieval or sequencing. Higher",
        "adjacent cosine is geometry evidence, not a listening verdict.",
        "",
    ]
    for record in primary:
        ctx = contexts[(str(record["case_id"]), PRIMARY_RESULT_COUNT)]
        lines += [
            f"## {record['case_id']}",
            "",
            f"Query: **{record['display_label']}**  ",
            f"Retrieval: {record['retrieval_contract']}  ",
            f"Membership SHA-256: `{record['ranked_membership_sha256']}`",
            "",
        ]
        diagnostics = record["membership_diagnostics"]
        lines += [
            f"Membership signals: {diagnostics['distinct_display_artists']} display artists; "
            f"{diagnostics['repeated_artist_title_group_count']} repeated artist/title groups; "
            f"{diagnostics['near_identical_embedding_pair_count']} >=0.9999-cosine pairs.",
            "",
        ]
        for variant in ("ranked", "smooth", "shuffle"):
            payload = record["variants"][variant]
            metrics = payload["metrics"]
            id_to_rank = {
                int(track_id): rank
                for rank, track_id in enumerate(record["ranked_track_ids"], start=1)
            }
            lines += [
                f"### {variant.title()}",
                "",
                f"Mean adjacent `{metrics['mean_adjacent_cosine']:.4f}`; p05 "
                f"`{metrics['p05_adjacent_cosine']:.4f}`; minimum "
                f"`{metrics['min_adjacent_cosine']:.4f}`; tail minimum "
                f"`{metrics['last10_min_adjacent_cosine']:.4f}`; mean rank displacement "
                f"`{metrics['mean_absolute_relevance_rank_displacement']:.2f}`; top-10 "
                f"retention `{metrics['top10_prefix_retention']:.0%}`.",
                "",
                "| Play | Relevance rank | Track | Previous cosine |",
                "|---:|---:|---|---:|",
            ]
            adjacent = metrics["adjacent_cosines"]
            for play_position, track_id in enumerate(payload["track_ids"], start=1):
                local_index = id_to_rank[int(track_id)] - 1
                previous = "-" if play_position == 1 else f"{adjacent[play_position - 2]:.4f}"
                lines.append(
                    f"| {play_position} | {local_index + 1} | {label(ctx, local_index)} | {previous} |"
                )
            lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def blind_variant_codes(case_id: str) -> dict[str, str]:
    variants = ("ranked", "smooth", "shuffle")
    ordered = sorted(
        variants,
        key=lambda variant: hashlib.sha256(
            f"find-music-listening-v1\0{case_id}\0{variant}".encode("utf-8")
        ).digest(),
    )
    return {variant: chr(ord("A") + position) for position, variant in enumerate(ordered)}


def write_listening_packet(
    path: Path,
    key_path: Path,
    contexts: dict[tuple[str, int], OrderingContext],
    records: Sequence[dict[str, object]],
) -> None:
    primary = [record for record in records if record["result_count"] == PRIMARY_RESULT_COUNT]
    key: dict[str, object] = {
        "schema": "find-music-ordering-listening-key-v1",
        "result_count": PRIMARY_RESULT_COUNT,
        "cases": {},
    }
    lines = [
        "# Find Music Queue Ordering: Blind Listening Packet",
        "",
        "Every A/B/C list contains exactly the same 20 displayed results. Judge the first",
        "few tracks, local flow, bad transitions, tail behavior, and whether reordering",
        "betrays the query ranking. Reveal the algorithms only after recording impressions.",
        "",
    ]
    for record in primary:
        case_id = str(record["case_id"])
        ctx = contexts[(case_id, PRIMARY_RESULT_COUNT)]
        codes = blind_variant_codes(case_id)
        key["cases"][case_id] = {
            "display_label": record["display_label"],
            "membership_sha256": record["ranked_membership_sha256"],
            "code_to_variant": {code: variant for variant, code in codes.items()},
        }
        lines += [f"## {case_id}", "", f"Query: **{record['display_label']}**", ""]
        for variant, code in sorted(codes.items(), key=lambda item: item[1]):
            payload = record["variants"][variant]
            id_to_local = {
                int(track_id): index
                for index, track_id in enumerate(record["ranked_track_ids"])
            }
            lines += [f"### List {code}", "", "| # | Track |", "|---:|---|"]
            for position, track_id in enumerate(payload["track_ids"], start=1):
                lines.append(
                    f"| {position} | {label(ctx, id_to_local[int(track_id)])} |"
                )
            lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    atomic_json(key_path, key)


def run(args: argparse.Namespace) -> None:
    total_started = time.perf_counter()
    input_manifest_path = args.composition_output / "run-manifest.json"
    rankings_path = args.composition_output / "rankings.json"
    input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    rankings = json.loads(rankings_path.read_text(encoding="utf-8"))
    if sha256_file(rankings_path) != input_manifest["artifacts"]["rankings.json"]:
        raise ValueError("active-composition rankings do not match their run manifest")
    helper_sha256 = sha256_file(Path(composition.__file__))
    if rankings["inputs"]["database_sha256"] != EXPECTED_DB_SHA256:
        raise ValueError("active-composition rankings use a different database")
    if rankings["inputs"]["active_catalog_sha256"] != EXPECTED_ACTIVE_CATALOG_SHA256:
        raise ValueError("active-composition rankings use a different active catalog")
    deterministic_input = dict(rankings)
    declared_payload_hash = str(deterministic_input.pop("deterministic_payload_sha256"))
    if sha256_json(deterministic_input) != declared_payload_hash:
        raise ValueError("active-composition deterministic payload hash is invalid")

    load_started = time.perf_counter()
    catalog = composition.parse_active_catalog(args.active_catalog)
    library, db_hash = composition.load_active_library(
        args.db, catalog, verify_hash=not args.skip_db_hash
    )
    text_embeddings, text_hashes, _ = composition.load_phone_text_embeddings(
        args.phone_report
    )
    cache = composition.AnchorCache(library, text_embeddings)
    load_seconds = time.perf_counter() - load_started
    if db_hash != EXPECTED_DB_SHA256 and not args.skip_db_hash:
        raise AssertionError("database hash verification did not return the expected hash")
    if library.count != EXPECTED_ACTIVE_COUNT:
        raise AssertionError("active library count changed")

    retrieval_cases = load_retrieval_cases(rankings, library, cache)
    position_by_track_id = {
        int(track_id): index for index, track_id in enumerate(library.track_ids)
    }
    contexts: dict[tuple[str, int], OrderingContext] = {}
    records: list[dict[str, object]] = []
    evaluation_started = time.perf_counter()
    for case_position, case in enumerate(retrieval_cases, start=1):
        supported_counts = [count for count in RESULT_COUNTS if count <= len(case.ranked_track_ids)]
        for count in supported_counts:
            ctx = build_context(case, count, library, position_by_track_id)
            contexts[(case.case_id, count)] = ctx
            record = evaluate_context(ctx)
            records.append(record)
            print(
                f"ordering {case_position}/{len(retrieval_cases)} {case.case_id} n={count}",
                flush=True,
            )
    evaluation_seconds = time.perf_counter() - evaluation_started
    if sha256_file(Path(composition.__file__)) != helper_sha256:
        raise ValueError("active-composition helper changed during the ordering run")

    deterministic_orders = {
        "schema": "find-music-queue-ordering-deterministic-v1",
        "input_composition_payload_sha256": declared_payload_hash,
        "shuffle_seed": SHUFFLE_SEED,
        "records": [
            {
                "case_id": record["case_id"],
                "result_count": record["result_count"],
                "membership_sha256": record["ranked_membership_sha256"],
                "variants": {
                    name: {
                        "play_order_sha256": payload["validation"]["play_order_sha256"],
                        "track_ids": payload["track_ids"],
                        "diagnostics": payload["diagnostics"],
                        "metrics": payload["metrics"],
                    }
                    for name, payload in record["variants"].items()
                },
            }
            for record in records
        ],
    }
    deterministic_hash = sha256_json(deterministic_orders)
    results = {
        "schema": "find-music-queue-ordering-eval-v1",
        "scope": {
            "retrieval_membership": "exact frozen active-domain result memberships",
            "playback_ordering": "host-only permutation evaluation",
            "metadata": "labels and duplicate/artist diagnostics only",
            "device_or_poweramp_mutation": "none",
            "listening_verdict": "not claimed",
        },
        "inputs": {
            "database_sha256": db_hash,
            "database_rows": queue_eval.EXPECTED_TRACKS,
            "active_catalog_sha256": sha256_file(args.active_catalog),
            "active_rows": library.count,
            "phone_report_sha256": sha256_file(args.phone_report),
            "phone_text_embedding_sha256": text_hashes,
            "active_composition_rankings_sha256": sha256_file(rankings_path),
            "active_composition_payload_sha256": declared_payload_hash,
            "active_composition_manifest_sha256": sha256_file(input_manifest_path),
            "active_composition_source_evaluator_sha256": input_manifest["artifacts"][
                "evaluator"
            ],
            "ordering_imported_helper_sha256": helper_sha256,
        },
        "contracts": {
            "ranked": "unchanged displayed relevance order",
            "smooth": (
                "fix relevance rank one; nearest-next initialization; deterministic "
                "best-improvement 2-opt minimizing total angular path"
            ),
            "shuffle": "reproducible SHA-256 hash permutation",
            "membership": "all modes are exact permutations; no drops, additions, or dedupe",
            "smooth_identity_policy": (
                "no metadata, artist, or duplicate constraint enters the path objective"
            ),
        },
        "parameters": {
            "primary_result_count": PRIMARY_RESULT_COUNT,
            "result_counts": RESULT_COUNTS,
            "timing_repeats": TIMING_REPEATS,
            "smooth_epsilon": SMOOTH_EPSILON,
            "shuffle_seed": SHUFFLE_SEED,
            "near_identical_embedding_diagnostic_cosine": NEAR_IDENTICAL_EMBEDDING_COSINE,
        },
        "records": records,
        "deterministic_order_payload_sha256": deterministic_hash,
    }
    summary = summarize(records)
    summary["deterministic_order_payload_sha256"] = deterministic_hash

    args.output.mkdir(parents=True, exist_ok=True)
    results_path = args.output / "results.json"
    summary_path = args.output / "summary.json"
    qualitative_path = args.output / "qualitative-lists.md"
    listening_path = args.output / "listening-packet.md"
    listening_key_path = args.output / "listening-key.json"
    atomic_json(results_path, results)
    atomic_json(summary_path, summary)
    write_qualitative_lists(qualitative_path, contexts, records)
    write_listening_packet(
        listening_path, listening_key_path, contexts, records
    )

    total_seconds = time.perf_counter() - total_started
    manifest = {
        "schema": "find-music-queue-ordering-run-manifest-v1",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        },
        "runtime_seconds": {
            "load_and_projection": load_seconds,
            "ordering_evaluation": evaluation_seconds,
            "total": total_seconds,
        },
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "deterministic_order_payload_sha256": deterministic_hash,
        "artifacts": {
            "results.json": sha256_file(results_path),
            "summary.json": sha256_file(summary_path),
            "qualitative-lists.md": sha256_file(qualitative_path),
            "listening-packet.md": sha256_file(listening_path),
            "listening-key.json": sha256_file(listening_key_path),
            "evaluator": sha256_file(Path(__file__)),
            "imported_active_composition_helper": helper_sha256,
        },
    }
    atomic_json(args.output / "run-manifest.json", manifest)
    print(
        f"complete: {args.output} ({total_seconds:.2f}s, {deterministic_hash})",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=queue_eval.DEFAULT_DB)
    parser.add_argument(
        "--active-catalog", type=Path, default=composition.DEFAULT_ACTIVE_CATALOG
    )
    parser.add_argument("--phone-report", type=Path, default=composition.DEFAULT_PHONE_REPORT)
    parser.add_argument("--composition-output", type=Path, default=DEFAULT_COMPOSITION_OUTPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-db-hash", action="store_true", help="development only")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
