#!/usr/bin/env python3
"""Deep, deterministic audit of every song-seed selection-mode promise.

This experiment deliberately separates five decisions that V1 currently blends:

* retrieval radius (which tracks may be considered),
* membership selection (Closest, MMR, DPP, or coverage),
* explicit artist constraints,
* drift/query evolution, and
* graph diffusion.

The immutable phone database supplies audio embeddings. Track metadata is used only to
resolve user-history seeds, enforce the user's explicit artist constraint, and label
evidence. It never contributes to a relevance or transition score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sqlite3
import struct
import sys
import time
import unicodedata
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import v2_queue_eval as queue_eval


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "desktop-indexer"
    / "audit_raw_data"
    / "v2-discovery"
    / "selection-modes-deep"
)
DEFAULT_PHONE_SNAPSHOT = (
    REPO_ROOT
    / "desktop-indexer"
    / "audit_raw_data"
    / "phone-live"
    / "2026-07-13T234337+0300"
)
DEFAULT_HISTORY = DEFAULT_PHONE_SNAPSHOT / "files" / "session_history.json"
DEFAULT_SETTINGS = DEFAULT_PHONE_SNAPSHOT / "shared_prefs" / "settings.xml"

QUEUE_SIZE = 30
EXPERIMENT_VERSION = "selection-modes-deep-v2"
CURRENT_POOL_FRACTION = 0.02
USER_MMR_LAMBDA = 0.97032166
USER_WALK_ALPHA = 0.05
USER_ANCHOR = 0.8440951
USER_MOMENTUM = 0.92064106
MAX_PER_ARTIST = 8
MIN_ARTIST_SPACING = 3


@dataclass(frozen=True)
class Graph:
    track_ids: np.ndarray
    neighbors: np.ndarray
    weights: np.ndarray

    @property
    def count(self) -> int:
        return int(self.track_ids.size)

    @property
    def k(self) -> int:
        return int(self.neighbors.shape[1])


@dataclass(frozen=True)
class DriftRun:
    selected: tuple[int, ...]
    query_ranks: tuple[int, ...]
    original_seed_ranks: tuple[int, ...]
    effective_anchors: tuple[float, ...]
    fallback_count: int
    fallback_mmr_regrets: tuple[float, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def append_jsonl(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                value,
                ensure_ascii=True,
                sort_keys=True,
                default=queue_eval.json_numpy_scalar,
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            value,
            handle,
            indent=2,
            ensure_ascii=True,
            sort_keys=True,
            default=queue_eval.json_numpy_scalar,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def completed_track_ids(path: Path) -> set[int]:
    if not path.exists():
        return set()
    return {
        int(json.loads(line)["seed"]["track_id"])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def normalized_identity(value: str | None) -> str:
    return unicodedata.normalize("NFKC", value or "").strip().casefold()


def load_history(path: Path) -> list[dict[str, object]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("session history must be a JSON array")
    return value


def resolve_history_seeds(
    library: queue_eval.Library,
    history: Sequence[dict[str, object]],
) -> tuple[list[int], list[dict[str, object]], list[dict[str, object]]]:
    """Resolve radio-session seeds by exact normalized identity plus duration.

    Resolution metadata chooses the evaluation cohort only. It cannot change a score.
    """
    positions: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, (artist, title) in enumerate(zip(library.artists, library.titles)):
        positions[(normalized_identity(artist), normalized_identity(title))].append(index)

    usage: Counter[int] = Counter()
    last_used: dict[int, int] = {}
    evidence: list[dict[str, object]] = []
    unresolved: list[dict[str, object]] = []
    for session in history:
        if bool(session.get("isDirectQueue")):
            continue
        seed = session.get("seedTrack")
        if not isinstance(seed, dict):
            continue
        key = (
            normalized_identity(seed.get("artist") if isinstance(seed.get("artist"), str) else None),
            normalized_identity(seed.get("title") if isinstance(seed.get("title"), str) else None),
        )
        candidates = positions.get(key, [])
        if not candidates:
            unresolved.append(
                {
                    "artist": seed.get("artist"),
                    "title": seed.get("title"),
                    "duration_ms": seed.get("durationMs"),
                }
            )
            continue
        duration = int(seed.get("durationMs") or -1)
        chosen = min(
            candidates,
            key=lambda index: (
                abs(int(library.durations_ms[index]) - duration) if duration >= 0 else 0,
                int(library.track_ids[index]),
            ),
        )
        usage[chosen] += 1
        last_used[chosen] = max(last_used.get(chosen, 0), int(session.get("timestamp") or 0))

    ranked = sorted(
        usage,
        key=lambda index: (-usage[index], -last_used[index], int(library.track_ids[index])),
    )
    for index in ranked:
        item = queue_eval.track_summary(library, index)
        item.update({"session_count": usage[index], "last_used": last_used[index]})
        evidence.append(item)
    return ranked, evidence, unresolved


def artist_valid(
    library: queue_eval.Library,
    selected: Sequence[int],
    candidate: int,
    max_per_artist: int = MAX_PER_ARTIST,
    spacing: int = MIN_ARTIST_SPACING,
) -> bool:
    artist = queue_eval.normalized_artist(library.artists[candidate])
    if artist is None:
        return True
    if sum(queue_eval.normalized_artist(library.artists[item]) == artist for item in selected) >= max_per_artist:
        return False
    return not any(
        queue_eval.normalized_artist(library.artists[item]) == artist
        for item in selected[-spacing:]
    )


def select_closest(
    library: queue_eval.Library,
    candidates: np.ndarray,
    count: int,
    constrained: bool,
) -> tuple[list[int], list[int]]:
    selected: list[int] = []
    ranks: list[int] = []
    for rank, raw_index in enumerate(candidates, start=1):
        index = int(raw_index)
        if constrained and not artist_valid(library, selected, index):
            continue
        selected.append(index)
        ranks.append(rank)
        if len(selected) == count:
            break
    return selected, ranks


def select_dpp_exponent(
    library: queue_eval.Library,
    candidates: np.ndarray,
    relevance: np.ndarray,
    count: int,
    quality_exponent: float,
) -> tuple[list[int], list[int]]:
    """Current greedy DPP, with its existing dormant quality exponent exposed."""
    if np.any(relevance <= 0.0):
        raise ValueError("fractional DPP quality powers require positive candidate relevance")
    embeddings = library.embeddings[candidates]
    quality = np.power(relevance.astype(np.float64), quality_exponent).astype(np.float32)
    limit = min(count, candidates.size)
    factors = np.zeros((candidates.size, limit), dtype=np.float32)
    residual = quality * quality
    remaining = np.ones(candidates.size, dtype=bool)
    selected: list[int] = []
    ranks: list[int] = []

    for step in range(limit):
        eligible = remaining.copy()
        for local_index in np.flatnonzero(eligible):
            if not artist_valid(library, selected, int(candidates[local_index])):
                eligible[local_index] = False
        gains = np.where(eligible, residual, -np.inf)
        best = int(np.argmax(gains))
        best_gain = float(gains[best])
        if not math.isfinite(best_gain) or best_gain <= 1e-10:
            break
        selected.append(int(candidates[best]))
        ranks.append(best + 1)
        remaining[best] = False
        root = math.sqrt(best_gain)
        kernel = quality * quality[best] * (embeddings @ embeddings[best])
        if step:
            kernel -= factors[:, :step] @ factors[best, :step]
        new_factor = kernel / root
        factors[remaining, step] = new_factor[remaining]
        residual[remaining] -= new_factor[remaining] ** 2
        np.maximum(residual, 0.0, out=residual)
        factors[best, step] = root
    return selected, ranks


def select_dpp_float64_reference(
    library: queue_eval.Library,
    candidates: np.ndarray,
    relevance: np.ndarray,
    count: int,
    quality_exponent: float,
) -> tuple[list[int], list[int]]:
    """Float64 reference for the current greedy DPP recurrence.

    This deliberately keeps the same greedy objective and artist eligibility rules as
    :func:`select_dpp_exponent`; it changes only arithmetic precision.
    """
    if np.any(relevance <= 0.0):
        raise ValueError("fractional DPP quality powers require positive candidate relevance")
    embeddings = library.embeddings[candidates].astype(np.float64)
    quality = np.power(relevance.astype(np.float64), quality_exponent)
    limit = min(count, candidates.size)
    factors = np.zeros((candidates.size, limit), dtype=np.float64)
    residual = quality * quality
    remaining = np.ones(candidates.size, dtype=bool)
    selected: list[int] = []
    ranks: list[int] = []

    for step in range(limit):
        eligible = remaining.copy()
        for local_index in np.flatnonzero(eligible):
            if not artist_valid(library, selected, int(candidates[local_index])):
                eligible[local_index] = False
        gains = np.where(eligible, residual, -np.inf)
        best = int(np.argmax(gains))
        best_gain = float(gains[best])
        if not math.isfinite(best_gain) or best_gain <= 1e-10:
            break
        selected.append(int(candidates[best]))
        ranks.append(best + 1)
        remaining[best] = False
        root = math.sqrt(best_gain)
        kernel = quality * quality[best] * (embeddings @ embeddings[best])
        if step:
            kernel -= factors[:, :step] @ factors[best, :step]
        new_factor = kernel / root
        factors[remaining, step] = new_factor[remaining]
        residual[remaining] -= new_factor[remaining] ** 2
        np.maximum(residual, 0.0, out=residual)
        factors[best, step] = root
    return selected, ranks


def select_facility_coverage(
    library: queue_eval.Library,
    candidates: np.ndarray,
    count: int,
) -> tuple[list[int], list[int]]:
    """Greedy facility-location coverage of the retrieved audio neighborhood.

    The closest candidate is fixed first. Every later pick maximizes how much it raises
    each candidate's similarity to its nearest selected representative. This is a pure
    embedding-space, monotone-submodular coverage hypothesis, not a shipping decision.
    """
    embeddings = library.embeddings[candidates]
    gram = np.clip(embeddings @ embeddings.T, -1.0, 1.0)
    selected_local = [0]
    selected = [int(candidates[0])]
    ranks = [1]
    remaining = np.ones(candidates.size, dtype=bool)
    remaining[0] = False
    coverage = gram[:, 0].copy()

    while len(selected) < min(count, candidates.size):
        gains = np.maximum(gram - coverage[:, None], 0.0).mean(axis=0)
        eligible = remaining.copy()
        for local_index in np.flatnonzero(eligible):
            if not artist_valid(library, selected, int(candidates[local_index])):
                eligible[local_index] = False
        gains = np.where(eligible, gains, -np.inf)
        best = int(np.argmax(gains))
        if not math.isfinite(float(gains[best])):
            break
        selected_local.append(best)
        selected.append(int(candidates[best]))
        ranks.append(best + 1)
        remaining[best] = False
        coverage = np.maximum(coverage, gram[:, best])
    return selected, ranks


def neighborhood_coverage(
    library: queue_eval.Library,
    candidates: np.ndarray,
    selected: Sequence[int],
) -> float | None:
    if not selected:
        return None
    similarities = library.embeddings[candidates] @ library.embeddings[np.asarray(selected)].T
    return float(np.mean(np.max(similarities, axis=1)))


def set_jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    left, right = set(a), set(b)
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def prefix_overlap(a: Sequence[int], b: Sequence[int], count: int) -> int:
    return len(set(a[:count]) & set(b[:count]))


def adjacency_metrics(library: queue_eval.Library, selected: Sequence[int]) -> dict[str, float | None]:
    if len(selected) < 2:
        return {"mean_adjacent_cosine": None, "p05_adjacent_cosine": None, "min_adjacent_cosine": None}
    embeddings = library.embeddings[np.asarray(selected)]
    adjacent = np.sum(embeddings[:-1] * embeddings[1:], axis=1)
    return {
        "mean_adjacent_cosine": float(np.mean(adjacent)),
        "p05_adjacent_cosine": float(np.percentile(adjacent, 5)),
        "min_adjacent_cosine": float(np.min(adjacent)),
    }


def variant(
    library: queue_eval.Library,
    seed_index: int,
    selected: Sequence[int],
    ranks: Sequence[int],
    requested: int,
    coverage_candidates: np.ndarray | None = None,
) -> dict[str, object]:
    metrics = queue_eval.queue_metrics(library, seed_index, selected, ranks, requested)
    metrics.update(adjacency_metrics(library, selected))
    if coverage_candidates is not None:
        metrics["neighborhood_coverage"] = neighborhood_coverage(
            library, coverage_candidates, selected
        )
    return {
        "metrics": metrics,
        "track_ids": [int(library.track_ids[index]) for index in selected],
        "candidate_ranks": [int(rank) for rank in ranks],
    }


def run_membership(
    library: queue_eval.Library,
    seeds: Sequence[int],
    path: Path,
    queue_size: int,
    current_pool: int,
    max_pool: int,
    coverage_seed_ids: set[int],
) -> None:
    completed = completed_track_ids(path)
    lambdas = (0.4, 0.8, 0.9, 0.95, USER_MMR_LAMBDA, 1.0)
    dpp_exponents = (0.0, 0.5, 1.0, 2.0, 4.0)
    pool_sizes = tuple(sorted({max(queue_size, int(library.count * fraction)) for fraction in (0.0025, 0.005, 0.01, 0.02, 0.04)}))

    for position, seed_index in enumerate(seeds, start=1):
        seed_id = int(library.track_ids[seed_index])
        if seed_id in completed:
            print(f"membership {position}/{len(seeds)} seed={seed_id} resumed", file=sys.stderr)
            continue
        candidates, relevance = queue_eval.retrieve_candidates(library, seed_index, max_pool)
        current_candidates = candidates[:current_pool]
        current_relevance = relevance[:current_pool]
        variants: dict[str, object] = {}

        for constrained in (False, True):
            selected, ranks = select_closest(
                library, current_candidates, queue_size, constrained=constrained
            )
            name = "closest_artist_aware" if constrained else "closest_unconstrained"
            variants[name] = variant(
                library, seed_index, selected, ranks, queue_size, current_candidates
            )

        for lambda_ in lambdas:
            selected, ranks = queue_eval.select_mmr(
                library,
                current_candidates,
                current_relevance,
                queue_size,
                lambda_=lambda_,
                constraint_aware=True,
                max_per_artist=MAX_PER_ARTIST,
                min_spacing=MIN_ARTIST_SPACING,
            )
            variants[f"mmr_lambda_{lambda_:.6f}"] = variant(
                library, seed_index, selected, ranks, queue_size, current_candidates
            )

        for exponent in dpp_exponents:
            selected, ranks = select_dpp_exponent(
                library, current_candidates, current_relevance, queue_size, exponent
            )
            variants[f"dpp_quality_exponent_{exponent:.1f}"] = variant(
                library, seed_index, selected, ranks, queue_size, current_candidates
            )

        for pool_size in pool_sizes:
            selected, ranks = select_dpp_exponent(
                library,
                candidates[:pool_size],
                relevance[:pool_size],
                queue_size,
                quality_exponent=1.0,
            )
            variants[f"dpp_pool_{pool_size}"] = variant(
                library, seed_index, selected, ranks, queue_size, current_candidates
            )

        if seed_id in coverage_seed_ids:
            selected, ranks = select_facility_coverage(
                library, current_candidates, queue_size
            )
            variants["facility_coverage"] = variant(
                library, seed_index, selected, ranks, queue_size, current_candidates
            )

        overlap_names = [
            "closest_unconstrained",
            "closest_artist_aware",
            f"mmr_lambda_{USER_MMR_LAMBDA:.6f}",
            "mmr_lambda_0.400000",
            "dpp_quality_exponent_1.0",
        ]
        if "facility_coverage" in variants:
            overlap_names.append("facility_coverage")
        overlaps: dict[str, float] = {}
        for left_pos, left in enumerate(overlap_names):
            for right in overlap_names[left_pos + 1 :]:
                overlaps[f"{left}__{right}"] = set_jaccard(
                    variants[left]["track_ids"], variants[right]["track_ids"]
                )

        append_jsonl(
            path,
            {
                "seed": queue_eval.track_summary(library, seed_index),
                "queue_size": queue_size,
                "current_pool": current_pool,
                "max_pool": max_pool,
                "variants": variants,
                "set_jaccard": overlaps,
            },
        )
        print(f"membership {position}/{len(seeds)} seed={seed_id} saved", file=sys.stderr)


def current_exponential_anchor(base: float, step: int, total_steps: int) -> float:
    if total_steps <= 1:
        return base
    return base * math.exp(-3.0 * step / (total_steps - 1))


def fixed_half_life_anchor(base: float, step: int, half_life_tracks: float) -> float:
    return base * math.pow(0.5, step / half_life_tracks)


def rank_map(similarities: np.ndarray, track_ids: np.ndarray) -> np.ndarray:
    order = np.lexsort((track_ids, -similarities))
    ranks = np.empty(order.size, dtype=np.int32)
    ranks[order] = np.arange(1, order.size + 1, dtype=np.int32)
    return ranks


def drift_run(
    library: queue_eval.Library,
    seed_index: int,
    count: int,
    pool_size: int,
    lambda_: float,
    base_anchor: float,
    schedule_total: int,
    half_life_tracks: float | None,
    constraint_aware: bool,
) -> DriftRun:
    seed_embedding = library.embeddings[seed_index]
    original_similarities = library.embeddings @ seed_embedding
    original_ranks = rank_map(original_similarities, library.track_ids)
    selected: list[int] = []
    selected_embeddings: list[np.ndarray] = []
    seen = {seed_index}
    query = seed_embedding.copy()
    query_ranks: list[int] = []
    seed_ranks: list[int] = []
    effective_anchors: list[float] = []
    fallback_count = 0
    fallback_regrets: list[float] = []

    for step in range(count):
        similarities = library.embeddings @ query
        similarities[np.fromiter(seen, dtype=np.int64)] = -np.inf
        pool = min(pool_size, library.count - len(seen))
        partial = np.argpartition(similarities, -pool)[-pool:]
        order = np.lexsort((library.track_ids[partial], -similarities[partial]))
        candidates = partial[order]
        relevance = similarities[candidates]
        candidate_embeddings = library.embeddings[candidates]
        if selected_embeddings:
            pairwise = candidate_embeddings @ np.stack(selected_embeddings).T
            penalty = np.max(pairwise, axis=1)
        else:
            penalty = np.zeros(candidates.size, dtype=np.float32)
        scores = lambda_ * relevance - (1.0 - lambda_) * penalty
        score_order = np.lexsort((library.track_ids[candidates], -scores))

        chosen_local: int | None = None
        best_unconstrained = int(score_order[0])
        if constraint_aware:
            for local in score_order:
                if artist_valid(library, selected, int(candidates[local])):
                    chosen_local = int(local)
                    break
        else:
            proposed = int(candidates[best_unconstrained])
            seen.add(proposed)
            if artist_valid(library, selected, proposed):
                chosen_local = best_unconstrained
            else:
                # Mirrors V1: artist fallback uses query relevance order, bypassing MMR.
                fallback_count += 1
                for local in range(candidates.size):
                    candidate = int(candidates[local])
                    if candidate != proposed and artist_valid(library, selected, candidate):
                        chosen_local = local
                        fallback_regrets.append(float(scores[best_unconstrained] - scores[local]))
                        break
        if chosen_local is None:
            break

        chosen = int(candidates[chosen_local])
        seen.add(chosen)
        selected.append(chosen)
        chosen_embedding = library.embeddings[chosen]
        selected_embeddings.append(chosen_embedding)
        query_ranks.append(chosen_local + 1)
        seed_ranks.append(int(original_ranks[chosen]))

        anchor = (
            fixed_half_life_anchor(base_anchor, step, half_life_tracks)
            if half_life_tracks is not None
            else current_exponential_anchor(base_anchor, step, schedule_total)
        )
        effective_anchors.append(anchor)
        query = anchor * seed_embedding + (1.0 - anchor) * chosen_embedding
        norm = float(np.linalg.norm(query))
        if norm > 1e-10:
            query /= norm

    return DriftRun(
        selected=tuple(selected),
        query_ranks=tuple(query_ranks),
        original_seed_ranks=tuple(seed_ranks),
        effective_anchors=tuple(effective_anchors),
        fallback_count=fallback_count,
        fallback_mmr_regrets=tuple(fallback_regrets),
    )


def drift_variant(
    library: queue_eval.Library,
    seed_index: int,
    run: DriftRun,
    requested: int,
) -> dict[str, object]:
    value = variant(
        library,
        seed_index,
        run.selected,
        run.query_ranks,
        requested,
    )
    value["original_seed_ranks"] = list(run.original_seed_ranks)
    value["effective_anchors"] = list(run.effective_anchors)
    value["fallback_count"] = run.fallback_count
    value["fallback_mmr_regrets"] = list(run.fallback_mmr_regrets)
    if run.original_seed_ranks:
        value["metrics"].update(
            {
                "median_original_seed_rank": float(np.median(run.original_seed_ranks)),
                "p90_original_seed_rank": float(np.percentile(run.original_seed_ranks, 90)),
                "final_original_seed_rank": int(run.original_seed_ranks[-1]),
                "median_step_query_rank": float(np.median(run.query_ranks)),
            }
        )
    return value


def run_drift(
    library: queue_eval.Library,
    seeds: Sequence[int],
    path: Path,
    queue_size: int,
    pool_size: int,
) -> None:
    completed = completed_track_ids(path)
    equivalent_half_life = (queue_size - 1) * math.log(2.0) / 3.0
    for position, seed_index in enumerate(seeds, start=1):
        seed_id = int(library.track_ids[seed_index])
        if seed_id in completed:
            print(f"drift {position}/{len(seeds)} seed={seed_id} resumed", file=sys.stderr)
            continue
        current_full = drift_run(
            library, seed_index, queue_size, pool_size, USER_MMR_LAMBDA,
            USER_ANCHOR, queue_size, None, constraint_aware=False,
        )
        current_peek = drift_run(
            library, seed_index, 10, pool_size, USER_MMR_LAMBDA,
            USER_ANCHOR, 10, None, constraint_aware=False,
        )
        fixed_full = drift_run(
            library, seed_index, queue_size, pool_size, USER_MMR_LAMBDA,
            USER_ANCHOR, queue_size, equivalent_half_life, constraint_aware=True,
        )
        fixed_peek = drift_run(
            library, seed_index, 10, pool_size, USER_MMR_LAMBDA,
            USER_ANCHOR, 10, equivalent_half_life, constraint_aware=True,
        )
        append_jsonl(
            path,
            {
                "seed": queue_eval.track_summary(library, seed_index),
                "parameters": {
                    "lambda": USER_MMR_LAMBDA,
                    "base_anchor": USER_ANCHOR,
                    "current_decay": "base * exp(-3 * step / (planned_length - 1))",
                    "fixed_half_life_tracks": equivalent_half_life,
                },
                "variants": {
                    "current_full_30": drift_variant(library, seed_index, current_full, queue_size),
                    "current_peek_10": drift_variant(library, seed_index, current_peek, 10),
                    "fixed_full_30": drift_variant(library, seed_index, fixed_full, queue_size),
                    "fixed_peek_10": drift_variant(library, seed_index, fixed_peek, 10),
                },
                "prefix": {
                    "current_exact_prefix": list(current_peek.selected) == list(current_full.selected[:10]),
                    "current_overlap_of_10": prefix_overlap(current_peek.selected, current_full.selected, 10),
                    "fixed_exact_prefix": list(fixed_peek.selected) == list(fixed_full.selected[:10]),
                    "fixed_overlap_of_10": prefix_overlap(fixed_peek.selected, fixed_full.selected, 10),
                },
            },
        )
        print(f"drift {position}/{len(seeds)} seed={seed_id} saved", file=sys.stderr)


def parse_graph(db_path: Path) -> Graph:
    connection = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro&immutable=1", uri=True)
    try:
        row = connection.execute(
            "SELECT data FROM binary_data WHERE key = 'knn_graph'"
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        raise RuntimeError("knn_graph is absent")
    blob = row[0]
    node_count, k = struct.unpack_from("<II", blob, 0)
    track_ids = np.frombuffer(blob, dtype="<i8", count=node_count, offset=8).copy()
    entries = np.frombuffer(
        blob,
        dtype=np.dtype([("index", "<u4"), ("weight", "<f4")]),
        count=node_count * k,
        offset=8 + node_count * 8,
    ).reshape(node_count, k)
    return Graph(
        track_ids=track_ids,
        neighbors=entries["index"].astype(np.int32, copy=True),
        weights=entries["weight"].astype(np.float64, copy=True),
    )


def build_edge_transition(graph: Graph, weighted: bool):
    from scipy.sparse import csr_matrix

    n, k = graph.neighbors.shape
    edge_count = n * k
    previous = np.repeat(np.arange(n, dtype=np.int32), k)
    current = graph.neighbors.ravel()
    sources_parts: list[np.ndarray] = []
    destinations_parts: list[np.ndarray] = []
    probabilities_parts: list[np.ndarray] = []
    for slot in range(k):
        following = graph.neighbors[current, slot]
        valid = following != previous
        sources = np.nonzero(valid)[0]
        probabilities = (
            graph.weights[current[sources], slot].copy()
            if weighted
            else np.ones(sources.size, dtype=np.float64)
        )
        sources_parts.append(sources)
        destinations_parts.append(current[sources] * k + slot)
        probabilities_parts.append(probabilities)
    sources = np.concatenate(sources_parts)
    destinations = np.concatenate(destinations_parts)
    probabilities = np.concatenate(probabilities_parts)
    row_sums = np.bincount(sources, weights=probabilities, minlength=edge_count)
    probabilities /= row_sums[sources]
    return csr_matrix(
        (probabilities, (destinations, sources)),
        shape=(edge_count, edge_count),
    )


def exact_terminal_distribution(
    graph: Graph,
    seed: int,
    alpha: float,
    transition,
    weighted: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate V1's non-backtracking geometric stopping distribution exactly."""
    n, k = graph.neighbors.shape
    current_nodes = graph.neighbors.ravel()
    probability = np.zeros(n * k, dtype=np.float64)
    seed_edges = np.arange(seed * k, (seed + 1) * k)
    initial = graph.weights[seed].copy() if weighted else np.ones(k, dtype=np.float64)
    initial /= initial.sum()
    probability[seed_edges] = initial
    terminals = np.zeros(n, dtype=np.float64)
    terminal_hop_mass = np.zeros(n, dtype=np.float64)
    for step in range(100):
        hop_count = step + 1
        if step == 99:
            mass = probability
        else:
            mass = alpha * probability
        terminals += np.bincount(current_nodes, weights=mass, minlength=n)
        terminal_hop_mass += np.bincount(
            current_nodes, weights=hop_count * mass, minlength=n
        )
        if step == 99:
            break
        probability = transition.dot(probability) * (1.0 - alpha)
        if float(probability.sum()) < 1e-14:
            break
    terminals[seed] = 0.0
    terminal_hop_mass[seed] = 0.0
    return terminals, terminal_hop_mass


def monte_carlo_terminal_counts(
    graph: Graph,
    seed: int,
    alpha: float,
    walks: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Mirror V1's uniform, non-backtracking, terminal-only walk sampler."""
    counts = np.zeros(graph.count, dtype=np.float64)
    for _ in range(walks):
        previous = -1
        current = seed
        for _step in range(100):
            choices: list[int] = []
            for slot in range(graph.k):
                neighbor = int(graph.neighbors[current, slot])
                if graph.weights[current, slot] > 0.0 and neighbor != previous:
                    choices.append(neighbor)
            if not choices:
                break
            following = choices[int(rng.integers(len(choices)))]
            previous, current = current, following
            if float(rng.random()) < alpha:
                break
        if current != seed:
            counts[current] += 1.0
    return counts


def bfs_hops(graph: Graph, seed: int) -> np.ndarray:
    hops = np.full(graph.count, -1, dtype=np.int32)
    hops[seed] = 0
    pending: deque[int] = deque([seed])
    while pending:
        current = pending.popleft()
        next_hop = int(hops[current]) + 1
        for neighbor in graph.neighbors[current]:
            target = int(neighbor)
            if hops[target] < 0:
                hops[target] = next_hop
                pending.append(target)
    return hops


def graph_select(
    library: queue_eval.Library,
    graph: Graph,
    scores: np.ndarray,
    seed_graph_index: int,
    count: int,
) -> tuple[list[int], list[int], np.ndarray]:
    order = np.lexsort((graph.track_ids, -scores))
    graph_to_library = np.searchsorted(library.track_ids, graph.track_ids)
    selected: list[int] = []
    score_ranks: list[int] = []
    for rank, graph_index in enumerate(order, start=1):
        if graph_index == seed_graph_index or scores[graph_index] <= 0.0:
            continue
        library_index = int(graph_to_library[graph_index])
        if not artist_valid(library, selected, library_index):
            continue
        selected.append(library_index)
        score_ranks.append(rank)
        if len(selected) == count:
            break
    return selected, score_ranks, graph_to_library


def run_graph(
    library: queue_eval.Library,
    graph: Graph,
    seeds: Sequence[int],
    path: Path,
    queue_size: int,
) -> None:
    if not np.array_equal(graph.track_ids, library.track_ids):
        raise ValueError("graph and embedding IDs are not in the same canonical order")
    completed = completed_track_ids(path)
    transitions = {
        "uniform": build_edge_transition(graph, weighted=False),
        "weighted": build_edge_transition(graph, weighted=True),
    }
    alphas = (0.05, 0.25, 0.5, 0.75, 0.95)

    for position, seed_index in enumerate(seeds, start=1):
        seed_id = int(library.track_ids[seed_index])
        if seed_id in completed:
            print(f"graph {position}/{len(seeds)} seed={seed_id} resumed", file=sys.stderr)
            continue
        seed_similarities = library.embeddings @ library.embeddings[seed_index]
        seed_ranks = rank_map(seed_similarities, library.track_ids)
        shortest = bfs_hops(graph, seed_index)
        variants: dict[str, object] = {}
        for alpha in alphas:
            policies = ("uniform", "weighted") if alpha in (0.05, 0.5) else ("uniform",)
            for policy in policies:
                terminals, hop_mass = exact_terminal_distribution(
                    graph,
                    seed_index,
                    alpha,
                    transitions[policy],
                    weighted=policy == "weighted",
                )
                selected, walk_ranks, _ = graph_select(
                    library, graph, terminals, seed_index, queue_size
                )
                graph_indices = np.asarray(selected, dtype=np.int64)
                expected_hops = np.divide(
                    hop_mass[graph_indices],
                    terminals[graph_indices],
                    out=np.zeros(len(selected), dtype=np.float64),
                    where=terminals[graph_indices] > 0.0,
                )
                selected_scores = terminals[graph_indices]
                total_terminal_mass = float(np.sum(terminals))
                normalized_terminals = (
                    terminals / total_terminal_mass
                    if total_terminal_mass > 0.0
                    else terminals
                )
                effective_support = (
                    1.0 / float(np.sum(normalized_terminals * normalized_terminals))
                    if np.any(normalized_terminals)
                    else 0.0
                )
                value = variant(
                    library,
                    seed_index,
                    selected,
                    [int(seed_ranks[index]) for index in selected],
                    queue_size,
                )
                value.update(
                    {
                        "walk_score_ranks": walk_ranks,
                        "shortest_graph_hops": [int(shortest[index]) for index in selected],
                        "expected_terminal_walk_hops": expected_hops.tolist(),
                        "mean_expected_terminal_walk_hops": float(np.mean(expected_hops)),
                        "mean_shortest_graph_hops": float(np.mean(shortest[graph_indices])),
                        "terminal_probability_mass": total_terminal_mass,
                        "queue_probability_mass": float(np.sum(selected_scores)),
                        "effective_terminal_support": effective_support,
                        "last_to_first_walk_score_ratio": float(
                            selected_scores[-1] / selected_scores[0]
                        ),
                        "walk_scores": selected_scores.tolist(),
                    }
                )
                variants[f"{policy}_alpha_{alpha:.2f}"] = value

        overlaps = {
            "uniform_vs_weighted_alpha_0.05": set_jaccard(
                variants["uniform_alpha_0.05"]["track_ids"],
                variants["weighted_alpha_0.05"]["track_ids"],
            ),
            "uniform_vs_weighted_alpha_0.50": set_jaccard(
                variants["uniform_alpha_0.50"]["track_ids"],
                variants["weighted_alpha_0.50"]["track_ids"],
            ),
            "alpha_0.05_vs_0.50": set_jaccard(
                variants["uniform_alpha_0.05"]["track_ids"],
                variants["uniform_alpha_0.50"]["track_ids"],
            ),
        }
        append_jsonl(
            path,
            {
                "seed": queue_eval.track_summary(library, seed_index),
                "graph_k": graph.k,
                "variants": variants,
                "set_jaccard": overlaps,
            },
        )
        print(f"graph {position}/{len(seeds)} seed={seed_id} saved", file=sys.stderr)


def run_stability(
    library: queue_eval.Library,
    graph: Graph,
    seeds: Sequence[int],
    path: Path,
    queue_size: int,
    pool_size: int,
    repeats: int,
) -> None:
    """Measure numerical DPP stability and Random Walk sampling variance."""
    if not np.array_equal(graph.track_ids, library.track_ids):
        raise ValueError("graph and embedding IDs are not in the same canonical order")
    completed = completed_track_ids(path)
    transition = build_edge_transition(graph, weighted=False)
    walks = 10_000

    for position, seed_index in enumerate(seeds, start=1):
        seed_id = int(library.track_ids[seed_index])
        if seed_id in completed:
            print(f"stability {position}/{len(seeds)} seed={seed_id} resumed", file=sys.stderr)
            continue

        candidates, relevance = queue_eval.retrieve_candidates(library, seed_index, pool_size)
        dpp32, _ = select_dpp_exponent(
            library, candidates, relevance, queue_size, quality_exponent=1.0
        )
        dpp64, _ = select_dpp_float64_reference(
            library, candidates, relevance, queue_size, quality_exponent=1.0
        )
        dpp_result = {
            "float32_track_ids": [int(library.track_ids[index]) for index in dpp32],
            "float64_track_ids": [int(library.track_ids[index]) for index in dpp64],
            "exact_order_match": dpp32 == dpp64,
            "set_jaccard": set_jaccard(dpp32, dpp64),
            "top10_overlap": prefix_overlap(dpp32, dpp64, 10),
        }

        walk_results: dict[str, object] = {}
        for alpha in (0.05, 0.5):
            exact_scores, _ = exact_terminal_distribution(
                graph, seed_index, alpha, transition, weighted=False
            )
            exact_selected, _, _ = graph_select(
                library, graph, exact_scores, seed_index, queue_size
            )
            runs: list[dict[str, object]] = []
            run_sets: list[list[int]] = []
            for repeat in range(repeats):
                rng_seed = (
                    seed_id * 1_000_003
                    + int(round(alpha * 100)) * 10_007
                    + repeat * 97
                ) & ((1 << 63) - 1)
                counts = monte_carlo_terminal_counts(
                    graph,
                    seed_index,
                    alpha,
                    walks,
                    np.random.default_rng(rng_seed),
                )
                sampled, _, _ = graph_select(
                    library, graph, counts, seed_index, queue_size
                )
                run_sets.append(sampled)
                runs.append(
                    {
                        "repeat": repeat,
                        "rng_seed": rng_seed,
                        "returned": len(sampled),
                        "sampled_terminal_support": int(np.count_nonzero(counts)),
                        "exact_set_jaccard": set_jaccard(sampled, exact_selected),
                        "exact_top10_overlap": prefix_overlap(sampled, exact_selected, 10),
                        "track_ids": [int(library.track_ids[index]) for index in sampled],
                    }
                )
            pairwise = [
                set_jaccard(left, right)
                for left_position, left in enumerate(run_sets)
                for right in run_sets[left_position + 1 :]
            ]
            walk_results[f"alpha_{alpha:.2f}"] = {
                "walks_per_repeat": walks,
                "exact_track_ids": [
                    int(library.track_ids[index]) for index in exact_selected
                ],
                "runs": runs,
                "run_to_run_set_jaccard": pairwise,
            }

        append_jsonl(
            path,
            {
                "seed": queue_eval.track_summary(library, seed_index),
                "dpp": dpp_result,
                "random_walk": walk_results,
            },
        )
        print(f"stability {position}/{len(seeds)} seed={seed_id} saved", file=sys.stderr)


def summarize_phone_history(
    library: queue_eval.Library,
    history: Sequence[dict[str, object]],
    path: Path,
) -> None:
    index_by_track_id = {int(track_id): index for index, track_id in enumerate(library.track_ids)}
    sessions: list[dict[str, object]] = []
    for session in history:
        if bool(session.get("isDirectQueue")):
            continue
        seed = session.get("seedTrack")
        config = session.get("config")
        tracks = session.get("tracks")
        if not isinstance(seed, dict) or not isinstance(config, dict) or not isinstance(tracks, list):
            continue
        seed_candidates = [
            index
            for index, (artist, title) in enumerate(zip(library.artists, library.titles))
            if normalized_identity(artist) == normalized_identity(seed.get("artist"))
            and normalized_identity(title) == normalized_identity(seed.get("title"))
        ]
        if not seed_candidates:
            continue
        duration = int(seed.get("durationMs") or -1)
        seed_index = min(
            seed_candidates,
            key=lambda index: (abs(int(library.durations_ms[index]) - duration), int(library.track_ids[index])),
        )
        selected = [
            index_by_track_id[int(item["track"]["id"])]
            for item in tracks
            if isinstance(item, dict)
            and isinstance(item.get("track"), dict)
            and int(item["track"]["id"]) in index_by_track_id
        ]
        if not selected:
            continue
        seed_sims = library.embeddings @ library.embeddings[seed_index]
        ranks = rank_map(seed_sims, library.track_ids)
        value = variant(
            library,
            seed_index,
            selected,
            [int(ranks[index]) for index in selected],
            len(selected),
        )
        sessions.append(
            {
                "timestamp": int(session.get("timestamp") or 0),
                "seed": queue_eval.track_summary(library, seed_index),
                "mode": config.get("selectionMode"),
                "lambda": config.get("diversityLambda"),
                "walk_alpha": config.get("walkRestartAlpha"),
                "drift_enabled": config.get("driftEnabled"),
                "anchor": config.get("anchorStrength"),
                "variant": value,
            }
        )

    by_seed: dict[int, list[dict[str, object]]] = defaultdict(list)
    for session in sessions:
        by_seed[int(session["seed"]["track_id"])].append(session)
    repeated: list[dict[str, object]] = []
    for seed_id, group in by_seed.items():
        if len(group) < 2:
            continue
        comparisons: list[dict[str, object]] = []
        for left_pos, left in enumerate(group):
            for right in group[left_pos + 1 :]:
                comparisons.append(
                    {
                        "left_timestamp": left["timestamp"],
                        "right_timestamp": right["timestamp"],
                        "left_mode": left["mode"],
                        "right_mode": right["mode"],
                        "left_walk_alpha": left["walk_alpha"],
                        "right_walk_alpha": right["walk_alpha"],
                        "set_jaccard": set_jaccard(
                            left["variant"]["track_ids"], right["variant"]["track_ids"]
                        ),
                        "top10_overlap": prefix_overlap(
                            left["variant"]["track_ids"], right["variant"]["track_ids"], 10
                        ),
                    }
                )
        repeated.append(
            {"seed_track_id": seed_id, "sessions": group, "comparisons": comparisons}
        )
    atomic_json(path, {"radio_sessions": sessions, "repeated_seeds": repeated})


def numeric_summary(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "p05": None, "p50": None, "p95": None}
    return {
        "mean": float(np.mean(values)),
        "p05": float(np.percentile(values, 5)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
    }


def summarize_membership(path: Path) -> dict[str, object]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    variants = sorted({name for record in records for name in record["variants"]})
    numeric_keys = (
        "count",
        "mean_seed_cosine",
        "min_seed_cosine",
        "mean_pairwise_cosine",
        "p95_pairwise_cosine",
        "unique_known_artists",
        "median_candidate_rank",
        "p90_candidate_rank",
        "mean_adjacent_cosine",
        "neighborhood_coverage",
    )
    result: dict[str, object] = {"records": len(records), "variants": {}, "set_jaccard": {}}
    for name in variants:
        values = [record["variants"][name] for record in records if name in record["variants"]]
        metrics_summary: dict[str, object] = {"records": len(values)}
        for key in numeric_keys:
            metrics_summary[key] = numeric_summary(
                [float(value["metrics"][key]) for value in values if value["metrics"].get(key) is not None]
            )
        result["variants"][name] = metrics_summary
    overlap_keys = sorted({key for record in records for key in record["set_jaccard"]})
    for key in overlap_keys:
        result["set_jaccard"][key] = numeric_summary(
            [float(record["set_jaccard"][key]) for record in records if key in record["set_jaccard"]]
        )

    high_mmr = f"mmr_lambda_{USER_MMR_LAMBDA:.6f}"
    dpp = "dpp_quality_exponent_1.0"
    moderate_mmr = "mmr_lambda_0.400000"
    dpp_broader_than_high = 0
    dpp_broader_than_moderate = 0
    for record in records:
        dpp_metrics = record["variants"][dpp]["metrics"]
        for comparison, counter in ((high_mmr, "high"), (moderate_mmr, "moderate")):
            other = record["variants"][comparison]["metrics"]
            broader = (
                dpp_metrics["mean_seed_cosine"] < other["mean_seed_cosine"]
                and dpp_metrics["mean_pairwise_cosine"] < other["mean_pairwise_cosine"]
            )
            if broader and counter == "high":
                dpp_broader_than_high += 1
            if broader and counter == "moderate":
                dpp_broader_than_moderate += 1
    result["claims"] = {
        "dpp_broader_than_user_high_mmr_fraction": dpp_broader_than_high / len(records),
        "dpp_broader_than_mmr_0.4_fraction": dpp_broader_than_moderate / len(records),
    }
    return result


def summarize_drift(path: Path) -> dict[str, object]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return {
        "records": len(records),
        "current_exact_prefix_rate": float(np.mean([record["prefix"]["current_exact_prefix"] for record in records])),
        "current_overlap_of_10": numeric_summary([float(record["prefix"]["current_overlap_of_10"]) for record in records]),
        "fixed_exact_prefix_rate": float(np.mean([record["prefix"]["fixed_exact_prefix"] for record in records])),
        "fixed_overlap_of_10": numeric_summary([float(record["prefix"]["fixed_overlap_of_10"]) for record in records]),
        "current_final_original_seed_rank": numeric_summary([
            float(record["variants"]["current_full_30"]["metrics"]["final_original_seed_rank"])
            for record in records
        ]),
        "fixed_final_original_seed_rank": numeric_summary([
            float(record["variants"]["fixed_full_30"]["metrics"]["final_original_seed_rank"])
            for record in records
        ]),
        "production_fallback_count": numeric_summary([
            float(record["variants"]["current_full_30"]["fallback_count"])
            for record in records
        ]),
    }


def summarize_graph(path: Path) -> dict[str, object]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    names = sorted(records[0]["variants"]) if records else []
    result: dict[str, object] = {"records": len(records), "variants": {}, "set_jaccard": {}}
    for name in names:
        values = [record["variants"][name] for record in records]
        result["variants"][name] = {
            "mean_seed_cosine": numeric_summary([float(value["metrics"]["mean_seed_cosine"]) for value in values]),
            "min_seed_cosine": numeric_summary([float(value["metrics"]["min_seed_cosine"]) for value in values]),
            "mean_pairwise_cosine": numeric_summary([float(value["metrics"]["mean_pairwise_cosine"]) for value in values]),
            "mean_expected_terminal_walk_hops": numeric_summary([float(value["mean_expected_terminal_walk_hops"]) for value in values]),
            "mean_shortest_graph_hops": numeric_summary([float(value["mean_shortest_graph_hops"]) for value in values]),
            "queue_probability_mass": numeric_summary([float(value["queue_probability_mass"]) for value in values]),
            "effective_terminal_support": numeric_summary([float(value["effective_terminal_support"]) for value in values]),
            "last_to_first_walk_score_ratio": numeric_summary([float(value["last_to_first_walk_score_ratio"]) for value in values]),
        }
    keys = sorted(records[0]["set_jaccard"]) if records else []
    for key in keys:
        result["set_jaccard"][key] = numeric_summary(
            [float(record["set_jaccard"][key]) for record in records]
        )

    for name in names:
        lists = [set(record["variants"][name]["track_ids"]) for record in records]
        cross_seed = [
            len(left & right) / len(left | right)
            for left_pos, left in enumerate(lists)
            for right in lists[left_pos + 1 :]
            if left | right
        ]
        result["variants"][name]["cross_seed_jaccard"] = numeric_summary(cross_seed)
        exposure = Counter(track_id for record in records for track_id in record["variants"][name]["track_ids"])
        result["variants"][name]["distinct_tracks_across_seeds"] = len(exposure)
        result["variants"][name]["maximum_cross_seed_exposure"] = max(exposure.values(), default=0)
    return result


def summarize_stability(path: Path) -> dict[str, object]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    result: dict[str, object] = {
        "records": len(records),
        "dpp": {
            "exact_order_match_fraction": float(
                np.mean([record["dpp"]["exact_order_match"] for record in records])
            ),
            "set_jaccard": numeric_summary(
                [float(record["dpp"]["set_jaccard"]) for record in records]
            ),
            "top10_overlap": numeric_summary(
                [float(record["dpp"]["top10_overlap"]) for record in records]
            ),
        },
        "random_walk": {},
    }
    for name in ("alpha_0.05", "alpha_0.50"):
        runs = [
            run
            for record in records
            for run in record["random_walk"][name]["runs"]
        ]
        pairwise = [
            float(value)
            for record in records
            for value in record["random_walk"][name]["run_to_run_set_jaccard"]
        ]
        result["random_walk"][name] = {
            "runs": len(runs),
            "exact_set_jaccard": numeric_summary(
                [float(run["exact_set_jaccard"]) for run in runs]
            ),
            "exact_top10_overlap": numeric_summary(
                [float(run["exact_top10_overlap"]) for run in runs]
            ),
            "sampled_terminal_support": numeric_summary(
                [float(run["sampled_terminal_support"]) for run in runs]
            ),
            "run_to_run_set_jaccard": numeric_summary(pairwise),
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=queue_eval.DEFAULT_DB)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ordinary-seeds", type=int, default=16)
    parser.add_argument("--max-user-seeds", type=int, default=32)
    parser.add_argument("--drift-seeds", type=int, default=8)
    parser.add_argument("--graph-seeds", type=int, default=12)
    parser.add_argument("--coverage-seeds", type=int, default=12)
    parser.add_argument("--queue-size", type=int, default=QUEUE_SIZE)
    parser.add_argument("--stability-repeats", type=int, default=8)
    parser.add_argument(
        "--experiments", default="history,membership,drift,graph,stability"
    )
    parser.add_argument("--skip-hash", action="store_true", help="development only")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    library, db_hash = queue_eval.load_library(args.db, verify_hash=not args.skip_hash)
    history = load_history(args.history)
    user_seeds, user_evidence, unresolved = resolve_history_seeds(library, history)
    user_seeds = user_seeds[: args.max_user_seeds]
    ordinary = queue_eval.build_cohort(library, args.ordinary_seeds)["ordinary"]
    all_seeds = list(dict.fromkeys(user_seeds + ordinary))
    current_pool = max(100, int(library.count * CURRENT_POOL_FRACTION))
    max_pool = max(current_pool, int(library.count * 0.04))
    coverage_seed_ids = {
        int(library.track_ids[index]) for index in all_seeds[: args.coverage_seeds]
    }

    args.output.mkdir(parents=True, exist_ok=True)
    files = {
        "membership": args.output / "membership.jsonl",
        "drift": args.output / "drift.jsonl",
        "graph": args.output / "graph.jsonl",
        "stability": args.output / "stability.jsonl",
        "history": args.output / "phone-history.json",
    }
    if args.force:
        for path in files.values():
            path.unlink(missing_ok=True)
        for path in args.output.glob("*-summary.json"):
            path.unlink()

    manifest = {
        "experiment_version": EXPERIMENT_VERSION,
        "evaluator_sha256": sha256_file(Path(__file__)),
        "database": str(args.db.resolve()),
        "database_sha256": db_hash,
        "track_count": library.count,
        "embedding_dim": library.dim,
        "phone_history": str(args.history.resolve()),
        "phone_history_sha256": sha256_file(args.history),
        "phone_settings": str(args.settings.resolve()),
        "phone_settings_sha256": sha256_file(args.settings),
        "metadata_policy": "cohort resolution, explicit artist constraints, and labels only",
        "parameters": {
            "queue_size": args.queue_size,
            "current_pool": current_pool,
            "current_pool_fraction": CURRENT_POOL_FRACTION,
            "maximum_experimental_pool": max_pool,
            "user_mmr_lambda": USER_MMR_LAMBDA,
            "user_walk_alpha": USER_WALK_ALPHA,
            "user_anchor": USER_ANCHOR,
            "user_momentum": USER_MOMENTUM,
            "max_per_artist": MAX_PER_ARTIST,
            "min_artist_spacing": MIN_ARTIST_SPACING,
            "stability_repeats": args.stability_repeats,
        },
        "cohorts": {
            "history_resolved": user_evidence,
            "history_unresolved": unresolved,
            "ordinary": [queue_eval.track_summary(library, index) for index in ordinary],
            "membership_track_ids": [int(library.track_ids[index]) for index in all_seeds],
            "drift_track_ids": [int(library.track_ids[index]) for index in user_seeds[: args.drift_seeds]],
            "graph_track_ids": [int(library.track_ids[index]) for index in all_seeds[: args.graph_seeds]],
            "coverage_track_ids": sorted(coverage_seed_ids),
            "stability_track_ids": [
                int(library.track_ids[index])
                for index in all_seeds[: args.graph_seeds]
            ],
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
    }
    manifest_path = args.output / "manifest.json"
    if not args.force:
        if manifest_path.exists():
            existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing_manifest != manifest:
                raise RuntimeError(
                    "output manifest does not match this evaluator/configuration; "
                    "use a new --output directory or rerun explicitly with --force"
                )
        elif any(path.exists() for path in files.values()):
            raise RuntimeError(
                "output records exist without a manifest; use a new --output directory "
                "or rerun explicitly with --force"
            )
    atomic_json(manifest_path, manifest)

    requested = {item.strip() for item in args.experiments.split(",") if item.strip()}
    if "history" in requested:
        summarize_phone_history(library, history, files["history"])
    if "membership" in requested:
        run_membership(
            library,
            all_seeds,
            files["membership"],
            args.queue_size,
            current_pool,
            max_pool,
            coverage_seed_ids,
        )
        atomic_json(args.output / "membership-summary.json", summarize_membership(files["membership"]))
    if "drift" in requested:
        run_drift(
            library,
            user_seeds[: args.drift_seeds],
            files["drift"],
            args.queue_size,
            current_pool,
        )
        atomic_json(args.output / "drift-summary.json", summarize_drift(files["drift"]))
    if "graph" in requested:
        graph = parse_graph(args.db)
        run_graph(
            library,
            graph,
            all_seeds[: args.graph_seeds],
            files["graph"],
            args.queue_size,
        )
        atomic_json(args.output / "graph-summary.json", summarize_graph(files["graph"]))
    if "stability" in requested:
        graph = parse_graph(args.db)
        run_stability(
            library,
            graph,
            all_seeds[: args.graph_seeds],
            files["stability"],
            args.queue_size,
            current_pool,
            args.stability_repeats,
        )
        atomic_json(
            args.output / "stability-summary.json",
            summarize_stability(files["stability"]),
        )


if __name__ == "__main__":
    main()
