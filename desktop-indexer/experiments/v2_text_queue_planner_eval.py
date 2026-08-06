#!/usr/bin/env python3
"""Evaluate embedding-only queue planners for raw Find Music text intent.

The retrieval baseline is the exact active-domain cosine ranking for the repeated phone
text embedding. MMR and greedy DPP are separately named set planners over that same
query vector. Metadata is emitted only after selection for diagnostics and human review;
it never changes candidate membership, objective values, tie handling, or order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import v2_active_composition_eval as composition
import v2_queue_eval as queue_eval


REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "discovery"
    / "evidence"
    / "text-queue-planners-2026-07-16"
)
QUERIES: tuple[str, ...] = (
    "ambient",
    "sleep",
    "relaxing",
    "slow",
    "psychedelic",
    "guitar",
    "late night downtempo",
    "organic electronic",
    "spacey jazz",
)
OUTPUT_COUNTS: tuple[int, ...] = (20, 30, 50)
PRIMARY_COUNT = 30
MAX_OUTPUT_COUNT = max(OUTPUT_COUNTS)
MMR_LAMBDAS: tuple[float, ...] = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
MMR_REACH_FRACTIONS: tuple[float, ...] = (0.0025, 0.005, 0.01, 0.02, 0.05, 1.0)
MMR_PRIMARY_REACH = 0.02
DPP_EXPONENTS: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
DPP_BOUNDED_FRACTION = 0.02
COVERAGE_DEPTHS: tuple[int, ...] = (100, 500, 1607)
NEAR_IDENTICAL_COSINE = 0.9999
MIN_DPP_GAIN = 1e-10


@dataclass(frozen=True)
class PlannerSpec:
    key: str
    family: str
    label: str
    parameter: float | None
    candidate_fraction: float
    candidate_semantics: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def unit_float32(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(value.astype(np.float64)))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("text embedding must be finite and non-zero")
    return (value / np.float32(norm)).astype(np.float32, copy=False)


def stable_relevance_order(scores: np.ndarray, track_ids: np.ndarray) -> np.ndarray:
    return np.lexsort((track_ids, -scores)).astype(np.int64, copy=False)


def candidate_count(total: int, fraction: float) -> int:
    if fraction >= 1.0:
        return total
    return min(total, max(MAX_OUTPUT_COUNT, int(math.ceil(total * fraction))))


def planner_specs() -> list[PlannerSpec]:
    result = [
        PlannerSpec(
            key="closest",
            family="closest",
            label="Closest matches",
            parameter=None,
            candidate_fraction=1.0,
            candidate_semantics="complete active-domain relevance ranking",
        )
    ]
    result.extend(
        PlannerSpec(
            key=f"mmr_l{lambda_:g}_r2pct",
            family="mmr_lambda",
            label=f"MMR relevance {lambda_:g}, top 2% candidates",
            parameter=lambda_,
            candidate_fraction=MMR_PRIMARY_REACH,
            candidate_semantics="fixed top 2% of the active-domain text ranking",
        )
        for lambda_ in MMR_LAMBDAS
    )
    result.extend(
        PlannerSpec(
            key=f"mmr_l0.6_r{fraction:g}",
            family="mmr_reach",
            label=(
                "MMR relevance 0.6, full active domain"
                if fraction == 1.0
                else f"MMR relevance 0.6, top {fraction * 100:g}% candidates"
            ),
            parameter=0.6,
            candidate_fraction=fraction,
            candidate_semantics=(
                "complete active-domain text ranking"
                if fraction == 1.0
                else f"fixed top {fraction * 100:g}% of the active-domain text ranking"
            ),
        )
        for fraction in MMR_REACH_FRACTIONS
        if fraction != MMR_PRIMARY_REACH
    )
    for exponent in DPP_EXPONENTS:
        result.append(
            PlannerSpec(
                key=f"dpp_e{exponent:g}_full",
                family="dpp_full",
                label=f"Full-domain DPP quality exponent {exponent:g}",
                parameter=exponent,
                candidate_fraction=1.0,
                candidate_semantics="exact greedy DPP over the complete active domain",
            )
        )
        result.append(
            PlannerSpec(
                key=f"dpp_e{exponent:g}_r2pct",
                family="dpp_bounded",
                label=f"Bounded DPP quality exponent {exponent:g}, top 2% candidates",
                parameter=exponent,
                candidate_fraction=DPP_BOUNDED_FRACTION,
                candidate_semantics="fixed top 2% of the active-domain text ranking",
            )
        )
    keys = [spec.key for spec in result]
    if len(keys) != len(set(keys)):
        raise AssertionError("planner keys must be unique")
    return result


def select_mmr(
    embeddings: np.ndarray,
    relevance: np.ndarray,
    ranked_candidates: np.ndarray,
    count: int,
    lambda_: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    candidates = np.asarray(ranked_candidates, dtype=np.int64)
    full_domain = candidates.size == embeddings.shape[0]
    candidate_vectors = None if full_domain else embeddings[candidates]
    remaining = np.ones(candidates.size, dtype=np.bool_)
    maximum_selected_similarity = np.full(
        candidates.size, -np.inf, dtype=np.float32
    )
    selected_local: list[int] = []
    objectives: list[float] = []
    penalties: list[float] = []
    limit = min(count, candidates.size)
    for step in range(limit):
        penalty = (
            np.zeros(candidates.size, dtype=np.float32)
            if step == 0
            else maximum_selected_similarity
        )
        objective = (
            np.float32(lambda_) * relevance[candidates]
            - np.float32(1.0 - lambda_) * penalty
        )
        best_local = int(np.argmax(np.where(remaining, objective, -np.inf)))
        if not remaining[best_local] or not math.isfinite(float(objective[best_local])):
            break
        selected_local.append(best_local)
        objectives.append(float(objective[best_local]))
        penalties.append(0.0 if step == 0 else float(penalty[best_local]))
        remaining[best_local] = False
        selected_global = int(candidates[best_local])
        similarities = (
            (embeddings @ embeddings[selected_global])[candidates]
            if full_domain
            else candidate_vectors @ candidate_vectors[best_local]
        )
        maximum_selected_similarity = np.maximum(
            maximum_selected_similarity, similarities.astype(np.float32, copy=False)
        )
    selected_local_array = np.asarray(selected_local, dtype=np.int64)
    return (
        candidates[selected_local_array],
        np.asarray(objectives, dtype=np.float32),
        np.asarray(penalties, dtype=np.float32),
    )


def select_dpp(
    embeddings: np.ndarray,
    relevance: np.ndarray,
    ranked_candidates: np.ndarray,
    count: int,
    exponent: float,
) -> tuple[np.ndarray, np.ndarray]:
    candidates = np.asarray(ranked_candidates, dtype=np.int64)
    quality = np.power(
        np.maximum(relevance, np.float32(0.0)).astype(np.float64), exponent
    ).astype(np.float32)
    residual = (quality * quality).astype(np.float32, copy=False)
    remaining = np.zeros(embeddings.shape[0], dtype=np.bool_)
    remaining[candidates] = True
    factors = np.zeros((embeddings.shape[0], min(count, candidates.size)), dtype=np.float32)
    selected: list[int] = []
    marginal_gains: list[float] = []
    for step in range(min(count, candidates.size)):
        ordered_gains = np.where(remaining[candidates], residual[candidates], -np.inf)
        best_rank = int(np.argmax(ordered_gains))
        best = int(candidates[best_rank])
        best_gain = float(ordered_gains[best_rank])
        if not math.isfinite(best_gain) or best_gain <= MIN_DPP_GAIN:
            break
        selected.append(best)
        marginal_gains.append(best_gain)
        remaining[best] = False
        root = math.sqrt(best_gain)
        similarity_column = embeddings @ embeddings[best]
        kernel = quality * quality[best] * similarity_column
        if step:
            kernel -= factors[:, :step] @ factors[best, :step]
        new_factor = kernel / np.float32(root)
        factors[remaining, step] = new_factor[remaining]
        residual[remaining] -= new_factor[remaining] ** 2
        np.maximum(residual, np.float32(0.0), out=residual)
        factors[best, step] = np.float32(root)
    return np.asarray(selected, dtype=np.int64), np.asarray(marginal_gains, dtype=np.float32)


def select_dpp_local(
    embeddings: np.ndarray,
    relevance: np.ndarray,
    ranked_candidates: np.ndarray,
    count: int,
    exponent: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy DPP over one explicit prefix without touching embeddings outside it."""
    candidates = np.asarray(ranked_candidates, dtype=np.int64)
    vectors = embeddings[candidates]
    quality = np.power(
        np.maximum(relevance[candidates], np.float32(0.0)).astype(np.float64),
        exponent,
    ).astype(np.float32)
    limit = min(count, candidates.size)
    residual = (quality * quality).astype(np.float32, copy=False)
    remaining = np.ones(candidates.size, dtype=np.bool_)
    factors = np.zeros((candidates.size, limit), dtype=np.float32)
    selected_local: list[int] = []
    gains: list[float] = []
    for step in range(limit):
        best = int(np.argmax(np.where(remaining, residual, -np.inf)))
        best_gain = float(residual[best])
        if not remaining[best] or not math.isfinite(best_gain) or best_gain <= MIN_DPP_GAIN:
            break
        selected_local.append(best)
        gains.append(best_gain)
        remaining[best] = False
        root = math.sqrt(best_gain)
        kernel = quality * quality[best] * (vectors @ vectors[best])
        if step:
            kernel -= factors[:, :step] @ factors[best, :step]
        new_factor = kernel / np.float32(root)
        factors[remaining, step] = new_factor[remaining]
        residual[remaining] -= new_factor[remaining] ** 2
        np.maximum(residual, np.float32(0.0), out=residual)
        factors[best, step] = np.float32(root)
    local = np.asarray(selected_local, dtype=np.int64)
    return candidates[local], np.asarray(gains, dtype=np.float32)


def select_dpp_certified(
    embeddings: np.ndarray,
    relevance: np.ndarray,
    relevance_order: np.ndarray,
    count: int,
    exponent: float,
    initial_fraction: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Mirror the production strict unseen-gain certificate over relevance prefixes."""
    total = relevance_order.size
    first_count = candidate_count(total, initial_fraction)
    if exponent <= 0.5:
        selected, gains = select_dpp(
            embeddings, relevance, relevance_order, count, exponent
        )
        return selected, gains, {
            "total_candidate_count": total,
            "initial_candidate_count": first_count,
            "attempted_candidate_counts": [total],
            "final_candidate_count": total,
            "final_unseen_gain_upper_bound": None,
            "used_full_domain": True,
            "direct_full_policy": True,
        }

    ordered_relevance = relevance[relevance_order]
    ordered_quality = np.power(
        np.maximum(ordered_relevance, np.float32(0.0)).astype(np.float64),
        exponent,
    ).astype(np.float32)
    initial_gains = ordered_quality * ordered_quality
    suffix = np.zeros(total + 1, dtype=np.float32)
    suffix[:-1] = np.maximum.accumulate(initial_gains[::-1])[::-1]
    attempts: list[int] = []
    current = first_count
    while True:
        attempts.append(current)
        selected, gains = select_dpp_local(
            embeddings,
            relevance,
            relevance_order[:current],
            count,
            exponent,
        )
        unseen = None if current == total else float(suffix[current])
        selected_enough = selected.size == min(count, total) or (
            unseen is not None and unseen <= MIN_DPP_GAIN
        )
        certified = unseen is None or all(float(gain) > unseen for gain in gains)
        if unseen is None or (selected_enough and certified):
            return selected, gains, {
                "total_candidate_count": total,
                "initial_candidate_count": first_count,
                "attempted_candidate_counts": attempts,
                "final_candidate_count": current,
                "final_unseen_gain_upper_bound": unseen,
                "used_full_domain": current == total,
                "direct_full_policy": False,
            }
        current = min(total, max(current + 1, int(math.ceil(current * 2.0))))


def normalized_text(value: str | None) -> str:
    return " ".join((value or "").casefold().split())


def excess(values: Iterable[str]) -> int:
    seen: set[str] = set()
    repeated = 0
    for value in values:
        if not value:
            continue
        if value in seen:
            repeated += 1
        else:
            seen.add(value)
    return repeated


def embedding_exact_excess(vectors: np.ndarray) -> int:
    return excess(hashlib.sha256(row.astype("<f4", copy=False).tobytes()).hexdigest() for row in vectors)


def near_identical_excess(pair_cosines: np.ndarray) -> int:
    if pair_cosines.shape[0] <= 1:
        return 0
    parent = list(range(pair_cosines.shape[0]))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left, right in zip(*np.where(np.triu(pair_cosines, k=1) >= NEAR_IDENTICAL_COSINE)):
        union(int(left), int(right))
    component_sizes: dict[int, int] = {}
    for index in range(len(parent)):
        root = find(index)
        component_sizes[root] = component_sizes.get(root, 0) + 1
    return sum(size - 1 for size in component_sizes.values())


def metrics(
    library: composition.ActiveLibrary,
    relevance: np.ndarray,
    inverse_rank: np.ndarray,
    closest: np.ndarray,
    selected: np.ndarray,
    coverage_references: dict[int, np.ndarray],
) -> dict[str, object]:
    vectors = library.embeddings[selected]
    scores = relevance[selected]
    ranks = inverse_rank[selected]
    pair = vectors @ vectors.T
    upper = pair[np.triu_indices(selected.size, 1)] if selected.size > 1 else np.asarray([])
    artists = [normalized_text(library.artists[int(index)]) for index in selected]
    artist_titles = [
        f"{normalized_text(library.artists[int(index)])}|{normalized_text(library.titles[int(index)])}"
        for index in selected
    ]
    coverage: dict[str, object] = {}
    for depth, reference_indices in coverage_references.items():
        best = np.max(library.embeddings[reference_indices] @ vectors.T, axis=1)
        coverage[str(depth)] = {
            "mean_best_selected_cosine": float(np.mean(best)),
            "p05_best_selected_cosine": float(np.percentile(best, 5)),
            "minimum_best_selected_cosine": float(np.min(best)),
        }
    return {
        "count": int(selected.size),
        "mean_query_cosine": float(np.mean(scores)),
        "p05_query_cosine": float(np.percentile(scores, 5)),
        "minimum_query_cosine": float(np.min(scores)),
        "median_query_rank": float(np.median(ranks)),
        "maximum_query_rank": int(np.max(ranks)),
        "top_n_retained": int(np.intersect1d(selected, closest, assume_unique=True).size),
        "mean_pairwise_cosine": float(np.mean(upper)) if upper.size else None,
        "p95_pairwise_cosine": float(np.percentile(upper, 95)) if upper.size else None,
        "maximum_pairwise_cosine": float(np.max(upper)) if upper.size else None,
        "exact_embedding_excess_slots": embedding_exact_excess(vectors),
        "near_identical_embedding_excess_slots": near_identical_excess(pair),
        "repeated_artist_slots": excess(artists),
        "artist_title_proxy_excess_slots": excess(artist_titles),
        "coverage": coverage,
    }


def track_rows(
    library: composition.ActiveLibrary,
    relevance: np.ndarray,
    inverse_rank: np.ndarray,
    selected: np.ndarray,
) -> list[dict[str, object]]:
    return [
        {
            "play_position": position,
            "query_rank": int(inverse_rank[index]),
            "track_id": int(library.track_ids[index]),
            "artist": library.artists[index],
            "album": library.albums[index],
            "title": library.titles[index],
            "query_cosine": float(relevance[index]),
            "file_path": library.file_paths[index],
        }
        for position, index in enumerate(selected, start=1)
    ]


def set_jaccard(left: Sequence[int], right: Sequence[int]) -> float:
    a = set(int(value) for value in left)
    b = set(int(value) for value in right)
    return len(a & b) / len(a | b) if a or b else 1.0


def aggregate(records: Sequence[dict[str, object]]) -> dict[str, object]:
    primary = [record for record in records if record["output_count"] == PRIMARY_COUNT]
    keys = sorted({str(record["planner_key"]) for record in primary})
    by_planner: dict[str, object] = {}
    for key in keys:
        rows = [record for record in primary if record["planner_key"] == key]

        def mean(field: str) -> float:
            return float(statistics.fmean(float(row["metrics"][field]) for row in rows))

        by_planner[key] = {
            "label": rows[0]["planner_label"],
            "query_count": len(rows),
            "candidate_count": rows[0]["candidate_count"],
            "all_repeats_exact": all(bool(row["repeat_exact"]) for row in rows),
            "mean_query_cosine": mean("mean_query_cosine"),
            "mean_p05_query_cosine": mean("p05_query_cosine"),
            "mean_median_query_rank": mean("median_query_rank"),
            "mean_maximum_query_rank": mean("maximum_query_rank"),
            "mean_top30_retained": mean("top_n_retained"),
            "mean_pairwise_cosine": mean("mean_pairwise_cosine"),
            "exact_embedding_excess_slots": int(
                sum(int(row["metrics"]["exact_embedding_excess_slots"]) for row in rows)
            ),
            "near_identical_embedding_excess_slots": int(
                sum(int(row["metrics"]["near_identical_embedding_excess_slots"]) for row in rows)
            ),
            "artist_title_proxy_excess_slots": int(
                sum(int(row["metrics"]["artist_title_proxy_excess_slots"]) for row in rows)
            ),
            "mean_top500_coverage": float(
                statistics.fmean(
                    float(row["metrics"]["coverage"]["500"]["mean_best_selected_cosine"])
                    for row in rows
                )
            ),
            "median_runtime_ms": float(statistics.median(float(row["runtime_ms"]) for row in rows)),
        }

    by_query_and_key = {
        (str(record["query"]), str(record["planner_key"])): record
        for record in primary
    }
    transitions: dict[str, object] = {}
    families = {
        "mmr_lambda": [f"mmr_l{value:g}_r2pct" for value in MMR_LAMBDAS],
        "mmr_reach": [
            f"mmr_l0.6_r{value:g}" if value != MMR_PRIMARY_REACH else "mmr_l0.6_r2pct"
            for value in MMR_REACH_FRACTIONS
        ],
        "dpp_full_exponent": [f"dpp_e{value:g}_full" for value in DPP_EXPONENTS],
        "dpp_bounded_exponent": [f"dpp_e{value:g}_r2pct" for value in DPP_EXPONENTS],
    }
    for family, ordered_keys in families.items():
        rows = []
        for left_key, right_key in zip(ordered_keys, ordered_keys[1:]):
            jaccards = []
            exact_aliases = 0
            for query in QUERIES:
                left = by_query_and_key[(query, left_key)]["selected_track_ids"]
                right = by_query_and_key[(query, right_key)]["selected_track_ids"]
                jaccard = set_jaccard(left, right)
                jaccards.append(jaccard)
                exact_aliases += int(left == right)
            rows.append(
                {
                    "left": left_key,
                    "right": right_key,
                    "mean_set_jaccard": float(statistics.fmean(jaccards)),
                    "minimum_set_jaccard": float(min(jaccards)),
                    "exact_sequence_aliases": exact_aliases,
                }
            )
        transitions[family] = rows

    bounded_vs_full = []
    for exponent in DPP_EXPONENTS:
        bounded_key = f"dpp_e{exponent:g}_r2pct"
        full_key = f"dpp_e{exponent:g}_full"
        jaccards = []
        aliases = 0
        for query in QUERIES:
            bounded = by_query_and_key[(query, bounded_key)]["selected_track_ids"]
            full = by_query_and_key[(query, full_key)]["selected_track_ids"]
            jaccards.append(set_jaccard(bounded, full))
            aliases += int(bounded == full)
        bounded_vs_full.append(
            {
                "exponent": exponent,
                "mean_set_jaccard": float(statistics.fmean(jaccards)),
                "minimum_set_jaccard": float(min(jaccards)),
                "exact_sequence_aliases": aliases,
            }
        )
    return {
        "primary_output_count": PRIMARY_COUNT,
        "by_planner": by_planner,
        "adjacent_knob_transitions": transitions,
        "bounded_dpp_vs_full": bounded_vs_full,
    }


def write_qualitative(path: Path, records: Sequence[dict[str, object]]) -> None:
    include = (
        "closest",
        "mmr_l0.8_r2pct",
        "mmr_l0.9_r2pct",
        "dpp_e2_full",
        "dpp_e4_full",
    )
    lookup = {
        (str(record["query"]), str(record["planner_key"])): record
        for record in records
        if record["output_count"] == PRIMARY_COUNT and record["planner_key"] in include
    }
    lines = [
        "# Text queue planner qualitative lists",
        "",
        "Artist/title are review labels only. They did not enter any planner.",
        "",
    ]
    for query in QUERIES:
        lines.extend([f"## {query}", ""])
        for key in include:
            record = lookup[(query, key)]
            metrics_row = record["metrics"]
            lines.extend(
                [
                    f"### {record['planner_label']}",
                    "",
                    (
                        f"mean query cosine {metrics_row['mean_query_cosine']:.5f}; "
                        f"median query rank {metrics_row['median_query_rank']:.1f}; "
                        f"top-30 retained {metrics_row['top_n_retained']}/30; "
                        f"mean pairwise cosine {metrics_row['mean_pairwise_cosine']:.5f}."
                    ),
                    "",
                ]
            )
            for row in record["tracks"]:
                lines.append(
                    f"{row['play_position']:02d}. query #{row['query_rank']}: "
                    f"{row['artist'] or '?'} - {row['title'] or '?'}"
                )
            lines.append("")
    atomic_text(path, "\n".join(lines).rstrip() + "\n")


def run(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    catalog = composition.parse_active_catalog(args.active_catalog)
    library, database_hash = composition.load_active_library(
        args.db, catalog, verify_hash=not args.skip_db_hash
    )
    text_embeddings, text_hashes, _ = composition.load_phone_text_embeddings(
        args.phone_report
    )
    missing = sorted(set(QUERIES) - set(text_embeddings))
    if missing:
        raise ValueError(f"phone report lacks required queries: {missing}")
    specs = planner_specs()
    records: list[dict[str, object]] = []

    for query_number, query in enumerate(QUERIES, start=1):
        print(f"[{query_number}/{len(QUERIES)}] {query}", flush=True)
        query_vector = unit_float32(text_embeddings[query])
        relevance = (library.embeddings @ query_vector).astype(np.float32, copy=False)
        order = stable_relevance_order(relevance, library.track_ids)
        inverse_rank = np.empty(library.count, dtype=np.int32)
        inverse_rank[order] = np.arange(1, library.count + 1, dtype=np.int32)
        coverage_references = {
            depth: order[: min(depth, library.count)] for depth in COVERAGE_DEPTHS
        }
        closest_by_count = {count: order[:count] for count in OUTPUT_COUNTS}

        for spec in specs:
            count = candidate_count(library.count, spec.candidate_fraction)
            candidates = order[:count]

            def execute() -> tuple[np.ndarray, dict[str, object]]:
                if spec.key == "closest":
                    return order[:MAX_OUTPUT_COUNT].copy(), {}
                if spec.family.startswith("mmr"):
                    selected, objectives, penalties = select_mmr(
                        library.embeddings,
                        relevance,
                        candidates,
                        MAX_OUTPUT_COUNT,
                        float(spec.parameter),
                    )
                    return selected, {
                        "objectives": objectives.tolist(),
                        "maximum_selected_similarities": penalties.tolist(),
                    }
                if spec.family == "dpp_full":
                    selected, gains, certificate = select_dpp_certified(
                        library.embeddings,
                        relevance,
                        order,
                        MAX_OUTPUT_COUNT,
                        float(spec.parameter),
                        DPP_BOUNDED_FRACTION,
                    )
                    return selected, {
                        "marginal_gains": gains.tolist(),
                        "certificate_for_50": certificate,
                    }
                selected, gains = select_dpp_local(
                    library.embeddings,
                    relevance,
                    candidates,
                    MAX_OUTPUT_COUNT,
                    float(spec.parameter),
                )
                return selected, {"marginal_gains": gains.tolist()}

            planner_started = time.perf_counter()
            selected, evidence = execute()
            runtime_ms = (time.perf_counter() - planner_started) * 1000.0
            repeated, repeated_evidence = execute()
            repeat_exact = np.array_equal(selected, repeated) and evidence == repeated_evidence
            if not repeat_exact:
                raise AssertionError(f"non-deterministic selection for {query}/{spec.key}")
            primary_runtime_ms = runtime_ms
            if spec.family == "dpp_full":
                primary_started = time.perf_counter()
                primary_selected, primary_gains, primary_certificate = select_dpp_certified(
                    library.embeddings,
                    relevance,
                    order,
                    PRIMARY_COUNT,
                    float(spec.parameter),
                    DPP_BOUNDED_FRACTION,
                )
                primary_runtime_ms = (time.perf_counter() - primary_started) * 1000.0
                if not np.array_equal(primary_selected, selected[:PRIMARY_COUNT]):
                    raise AssertionError(
                        f"DPP prefix changed with requested count: {query}/{spec.key}"
                    )
                reference_started = time.perf_counter()
                reference, reference_gains = select_dpp(
                    library.embeddings,
                    relevance,
                    order,
                    MAX_OUTPUT_COUNT,
                    float(spec.parameter),
                )
                reference_runtime_ms = (time.perf_counter() - reference_started) * 1000.0
                if not np.array_equal(reference, selected):
                    raise AssertionError(
                        f"certified DPP differs from full reference: {query}/{spec.key}"
                    )
                evidence["certificate_for_30"] = primary_certificate
                evidence["marginal_gains_for_30"] = primary_gains.tolist()
                evidence["canonical_full_reference_exact_for_50"] = True
                evidence["canonical_full_reference_runtime_ms_for_50"] = (
                    reference_runtime_ms
                )
            if selected.size < MAX_OUTPUT_COUNT or np.unique(selected).size != selected.size:
                raise AssertionError(f"invalid selection for {query}/{spec.key}")
            if not set(int(value) for value in selected).issubset(
                set(int(value) for value in candidates)
            ):
                raise AssertionError(f"planner escaped candidate domain: {query}/{spec.key}")

            for output_count in OUTPUT_COUNTS:
                prefix = selected[:output_count]
                closest = closest_by_count[output_count]
                record = {
                    "query": query,
                    "query_embedding_sha256": text_hashes[query],
                    "planner_key": spec.key,
                    "planner_family": spec.family,
                    "planner_label": spec.label,
                    "parameter": spec.parameter,
                    "candidate_fraction": spec.candidate_fraction,
                    "candidate_count": count,
                    "candidate_semantics": spec.candidate_semantics,
                    "output_count": output_count,
                    "runtime_ms": (
                        primary_runtime_ms
                        if output_count == PRIMARY_COUNT
                        else runtime_ms
                    ),
                    "repeat_exact": repeat_exact,
                    "selected_track_ids": [
                        int(library.track_ids[index]) for index in prefix
                    ],
                    "ordered_track_id_sha256": sha256_json(
                        [int(library.track_ids[index]) for index in prefix]
                    ),
                    "metrics": metrics(
                        library,
                        relevance,
                        inverse_rank,
                        closest,
                        prefix,
                        coverage_references,
                    ),
                    "tracks": track_rows(library, relevance, inverse_rank, prefix),
                }
                if output_count == MAX_OUTPUT_COUNT:
                    record["selection_evidence"] = evidence
                records.append(record)

    summary = aggregate(records)
    deterministic_payload = [
        {
            "query": record["query"],
            "planner": record["planner_key"],
            "count": record["output_count"],
            "track_ids": record["selected_track_ids"],
        }
        for record in records
    ]
    result = {
        "schema": "v2-text-queue-planner-eval-v1",
        "inputs": {
            "database": str(args.db.resolve()),
            "database_sha256": database_hash,
            "active_catalog": str(args.active_catalog.resolve()),
            "active_catalog_sha256": sha256_file(args.active_catalog),
            "active_track_count": library.count,
            "phone_report": str(args.phone_report.resolve()),
            "phone_report_sha256": sha256_file(args.phone_report),
            "queries": QUERIES,
            "text_embedding_sha256": {query: text_hashes[query] for query in QUERIES},
        },
        "contract": {
            "selection_inputs": "CLaMP3 audio embeddings and the exact phone text embedding only",
            "stable_tie_order": "descending Float32 relevance then ascending track ID",
            "metadata_role": "labels and diagnostics only",
            "output_counts": OUTPUT_COUNTS,
            "mmr_equation": "lambda*query_cosine - (1-lambda)*max_selected_cosine",
            "dpp_kernel": "L_ij=q_i*q_j*dot(audio_i,audio_j), q=max(query_cosine,0)^exponent",
            "near_identical_threshold_is_diagnostic_only": NEAR_IDENTICAL_COSINE,
        },
        "planner_specs": [spec.__dict__ for spec in specs],
        "records": records,
        "deterministic_selection_payload_sha256": sha256_json(deterministic_payload),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    results_path = args.output / "results.json"
    summary_path = args.output / "summary.json"
    qualitative_path = args.output / "qualitative-lists.md"
    atomic_json(results_path, result)
    atomic_json(summary_path, summary)
    write_qualitative(qualitative_path, records)
    manifest = {
        "schema": "v2-text-queue-planner-run-manifest-v1",
        "runtime_seconds": time.perf_counter() - started,
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        },
        "artifacts": {
            "results.json": sha256_file(results_path),
            "summary.json": sha256_file(summary_path),
            "qualitative-lists.md": sha256_file(qualitative_path),
            "evaluator": sha256_file(Path(__file__)),
        },
        "deterministic_selection_payload_sha256": result[
            "deterministic_selection_payload_sha256"
        ],
    }
    atomic_json(args.output / "run-manifest.json", manifest)
    print(
        f"complete: {args.output} ({manifest['runtime_seconds']:.2f}s, "
        f"{result['deterministic_selection_payload_sha256']})",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=queue_eval.DEFAULT_DB)
    parser.add_argument(
        "--active-catalog", type=Path, default=composition.DEFAULT_ACTIVE_CATALOG
    )
    parser.add_argument("--phone-report", type=Path, default=composition.DEFAULT_PHONE_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-db-hash", action="store_true", help="development only")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
