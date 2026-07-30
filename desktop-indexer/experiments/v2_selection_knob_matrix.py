#!/usr/bin/env python3
"""Canonical host-side selector knob matrix over the frozen active library.

This is evidence code, not an alternate app implementation.  It mirrors the current
Kotlin selector contracts and first proves that its default cases reproduce the frozen
phone acceptance cohort.  Audio embeddings and the prepared audio kNN graph are the only
relevance/transition inputs.  Metadata is used only for the explicit artist-credit
constraint, frozen seed resolution, duplicate diagnostics, and human-readable evidence.
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
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import compare_device_feature_acceptance as phone_oracle
import v2_queue_eval as queue_eval
import v2_selection_mode_eval as selection_eval


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
    REPO_ROOT
    / "desktop-indexer"
    / "audit_raw_data"
    / "v2-discovery"
    / "selection-knob-matrix-2026-07-15"
)

EXPERIMENT_VERSION = "selection-knob-matrix-active-domain-v1"
EXPECTED_ACTIVE_TRACKS = 80_323
EXPECTED_CATALOG_SHA256 = "e5bd6f7e0e29e25ae001b83bf20af276ce7ec06aa4f53c2d7ad4e5d0a9651c75"
EXPECTED_PHONE_REPORT_SHA256 = "86f7555a1aefce5929c03829eef1a5185d2d83b62b6a4686823e79466141e406"

DEFAULT_QUEUE_SIZE = 30
DEFAULT_REACH = 0.02
DEFAULT_LAMBDA = 0.97032166
DEFAULT_DPP_EXPONENT = 1.0
DEFAULT_GRAPH_ALPHA = 0.05
DEFAULT_ANCHOR = 0.8440951
DEFAULT_HALF_LIFE = 6.7004
DEFAULT_MOMENTUM = 0.92064106
DEFAULT_MAX_PER_ARTIST = 8
DEFAULT_ARTIST_SPACING = 3

REACH_VALUES = (0.0025, 0.005, 0.01, 0.02, 0.05)
MMR_LAMBDAS = (0.0, 0.25, 0.4, 0.6, 0.8, 0.9, 0.95, DEFAULT_LAMBDA, 1.0)
DPP_EXPONENTS = (0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0)
GRAPH_ALPHAS = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
HALF_LIFE_VALUES = (1.0, 3.0, DEFAULT_HALF_LIFE, 10.0, 15.0, 30.0)
ANCHOR_VALUES = (0.0, 0.25, 0.5, 0.75, DEFAULT_ANCHOR, 1.0)
MOMENTUM_VALUES = (0.0, 0.25, 0.5, 0.75, DEFAULT_MOMENTUM, 0.95, 1.0)
QUEUE_SIZES = (10, 30, 50, 100)


@dataclass(frozen=True)
class SelectorConfig:
    mode: str
    queue_size: int = DEFAULT_QUEUE_SIZE
    reach: float = DEFAULT_REACH
    mmr_lambda: float = DEFAULT_LAMBDA
    dpp_exponent: float = DEFAULT_DPP_EXPONENT
    graph_alpha: float = DEFAULT_GRAPH_ALPHA
    drift_mode: str | None = None
    anchor_schedule: str = "EXPONENTIAL"
    anchor_strength: float = DEFAULT_ANCHOR
    anchor_half_life: float = DEFAULT_HALF_LIFE
    momentum_beta: float = DEFAULT_MOMENTUM
    artist_limits: bool = True
    max_per_artist: int = DEFAULT_MAX_PER_ARTIST
    artist_spacing: int = DEFAULT_ARTIST_SPACING
    candidate_pool_size: int = 0
    dpp_uses_certified_full_domain: bool = False


@dataclass(frozen=True)
class SweepPoint:
    sweep_id: str
    control: str
    value: float | int | str
    order: int


@dataclass
class MatrixCase:
    case_id: str
    config: SelectorConfig
    aliases: list[str] = field(default_factory=list)
    sweeps: list[SweepPoint] = field(default_factory=list)


@dataclass
class SeedContext:
    seed_index: int
    seed_similarities: np.ndarray
    seed_rank: np.ndarray
    closest_order: np.ndarray
    scan_ms: float
    sort_ms: float


@dataclass
class GraphDistribution:
    terminal: np.ndarray
    terminal_link_mass: np.ndarray
    expected_links: float
    excluded_seed_probability: float
    total_mass: float
    mass_error: float
    evaluated_links: int
    elapsed_ms: float


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json_hash(value: object) -> str:
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


def normalized_artist(value: str | None) -> str | None:
    """Mirror PostFilter.normalizedArtistCredit: trim, non-empty, lowercase."""
    if value is None:
        return None
    normalized = value.strip().lower()
    return normalized or None


def normalized_text(value: str | None) -> str:
    return unicodedata.normalize("NFKC", value or "").strip().casefold()


def artist_valid(
    library: queue_eval.Library,
    selected: Sequence[int],
    candidate: int,
    config: SelectorConfig,
) -> bool:
    if not config.artist_limits:
        return True
    artist = normalized_artist(library.artists[candidate])
    if artist is None:
        return True
    if sum(normalized_artist(library.artists[index]) == artist for index in selected) >= config.max_per_artist:
        return False
    if config.artist_spacing <= 0:
        return True
    return not any(
        normalized_artist(library.artists[index]) == artist
        for index in selected[-config.artist_spacing :]
    )


def effective_pool_count(active_count: int, config: SelectorConfig) -> int:
    if config.mode == "dpp" and config.dpp_uses_certified_full_domain:
        return active_count
    if config.candidate_pool_size < 0:
        raise ValueError("candidate_pool_size cannot be negative")
    if config.candidate_pool_size > 0:
        return min(active_count, config.candidate_pool_size)
    requested = max(100, config.queue_size, int(active_count * config.reach))
    return min(active_count, requested)


def subset_library(library: queue_eval.Library, positions: np.ndarray) -> queue_eval.Library:
    indices = [int(index) for index in positions]
    return queue_eval.Library(
        track_ids=library.track_ids[positions].copy(),
        embeddings=library.embeddings[positions].copy(),
        artists=tuple(library.artists[index] for index in indices),
        albums=tuple(library.albums[index] for index in indices),
        titles=tuple(library.titles[index] for index in indices),
        durations_ms=library.durations_ms[positions].copy(),
        clusters=library.clusters[positions].copy(),
        sources=tuple(library.sources[index] for index in indices),
        metadata_keys=tuple(library.metadata_keys[index] for index in indices),
        filename_keys=tuple(library.filename_keys[index] for index in indices),
        file_paths=tuple(library.file_paths[index] for index in indices),
    )


def seed_context(library: queue_eval.Library, seed_index: int) -> SeedContext:
    started = time.perf_counter()
    similarities = (library.embeddings @ library.embeddings[seed_index]).astype(
        np.float32, copy=False
    )
    scan_ms = (time.perf_counter() - started) * 1000.0
    started = time.perf_counter()
    all_order = np.lexsort((library.track_ids, -similarities))
    ranks = np.empty(library.count, dtype=np.int32)
    ranks[all_order] = np.arange(1, library.count + 1, dtype=np.int32)
    closest = all_order[all_order != seed_index].astype(np.int64, copy=False)
    sort_ms = (time.perf_counter() - started) * 1000.0
    return SeedContext(seed_index, similarities, ranks, closest, scan_ms, sort_ms)


def candidates_for(
    library: queue_eval.Library,
    context: SeedContext,
    config: SelectorConfig,
) -> tuple[np.ndarray, np.ndarray, int]:
    requested = effective_pool_count(library.count, config)
    actual = min(requested, context.closest_order.size)
    candidates = context.closest_order[:actual]
    return candidates, context.seed_similarities[candidates], actual


def select_closest(
    library: queue_eval.Library,
    context: SeedContext,
    config: SelectorConfig,
) -> tuple[list[int], list[int], list[float], int]:
    selected: list[int] = []
    ranks: list[int] = []
    scores: list[float] = []
    for rank, raw_index in enumerate(context.closest_order, start=1):
        index = int(raw_index)
        if not artist_valid(library, selected, index, config):
            continue
        selected.append(index)
        ranks.append(rank)
        scores.append(float(context.seed_similarities[index]))
        if len(selected) == config.queue_size:
            break
    return selected, ranks, scores, int(context.closest_order.size)


def candidate_artist_codes(
    library: queue_eval.Library,
    candidates: np.ndarray,
) -> tuple[np.ndarray, int]:
    codes = np.full(candidates.size, -1, dtype=np.int32)
    mapping: dict[str, int] = {}
    for local, raw_index in enumerate(candidates):
        artist = normalized_artist(library.artists[int(raw_index)])
        if artist is None:
            continue
        code = mapping.setdefault(artist, len(mapping))
        codes[local] = code
    return codes, len(mapping)


def apply_artist_eligibility(
    eligible: np.ndarray,
    codes: np.ndarray,
    counts: np.ndarray,
    recent: Sequence[int],
    config: SelectorConfig,
) -> None:
    if not config.artist_limits:
        return
    known = codes >= 0
    eligible[known] &= counts[codes[known]] < config.max_per_artist
    if config.artist_spacing > 0:
        recent_known = [code for code in recent[-config.artist_spacing :] if code >= 0]
        if recent_known:
            eligible &= ~np.isin(codes, recent_known)


def select_mmr(
    library: queue_eval.Library,
    candidates: np.ndarray,
    relevance: np.ndarray,
    config: SelectorConfig,
) -> tuple[list[int], list[int], list[float]]:
    embeddings = library.embeddings[candidates]
    artist_codes, artist_count = candidate_artist_codes(library, candidates)
    counts = np.zeros(artist_count, dtype=np.int32)
    recent: list[int] = []
    remaining = np.ones(candidates.size, dtype=np.bool_)
    max_similarity = np.full(candidates.size, -np.inf, dtype=np.float32)
    selected: list[int] = []
    ranks: list[int] = []
    scores: list[float] = []
    lambda_ = np.float32(config.mmr_lambda)
    one_minus = np.float32(1.0) - lambda_

    for step in range(config.queue_size):
        if not bool(remaining.any()):
            break
        penalty: float | np.ndarray = 0.0 if step == 0 else max_similarity
        objective = lambda_ * relevance - one_minus * penalty
        eligible = remaining.copy()
        apply_artist_eligibility(eligible, artist_codes, counts, recent, config)
        if not bool(eligible.any()):
            break
        best = int(np.argmax(np.where(eligible, objective, -np.inf)))
        selected_index = int(candidates[best])
        selected.append(selected_index)
        ranks.append(best + 1)
        scores.append(float(relevance[best]))
        remaining[best] = False
        code = int(artist_codes[best])
        if config.artist_limits:
            if code >= 0:
                counts[code] += 1
            recent.append(code)
            if len(recent) > config.artist_spacing:
                recent.pop(0)
        similarities = embeddings @ library.embeddings[selected_index]
        max_similarity = np.maximum(max_similarity, similarities)
    return selected, ranks, scores


def select_dpp(
    library: queue_eval.Library,
    candidates: np.ndarray,
    relevance: np.ndarray,
    config: SelectorConfig,
) -> tuple[list[int], list[int], list[float]]:
    embeddings = library.embeddings[candidates]
    artist_codes, artist_count = candidate_artist_codes(library, candidates)
    counts = np.zeros(artist_count, dtype=np.int32)
    recent: list[int] = []
    non_negative = np.maximum(relevance, np.float32(0.0))
    if config.dpp_exponent == 1.0:
        quality = non_negative.astype(np.float32, copy=True)
    else:
        quality = np.power(
            non_negative.astype(np.float64), float(config.dpp_exponent)
        ).astype(np.float32)
    limit = min(config.queue_size, candidates.size)
    factors = np.zeros((candidates.size, limit), dtype=np.float32)
    residual = quality * quality
    remaining = np.ones(candidates.size, dtype=np.bool_)
    selected: list[int] = []
    ranks: list[int] = []
    scores: list[float] = []

    for step in range(limit):
        eligible = remaining.copy()
        apply_artist_eligibility(eligible, artist_codes, counts, recent, config)
        gains = np.where(eligible, residual, -np.inf)
        best = int(np.argmax(gains))
        best_gain = float(gains[best])
        if not math.isfinite(best_gain) or best_gain <= 1e-10:
            break
        selected_index = int(candidates[best])
        selected.append(selected_index)
        ranks.append(best + 1)
        scores.append(float(relevance[best]))
        remaining[best] = False
        code = int(artist_codes[best])
        if config.artist_limits:
            if code >= 0:
                counts[code] += 1
            recent.append(code)
            if len(recent) > config.artist_spacing:
                recent.pop(0)

        root = math.sqrt(best_gain)
        kernel = quality * quality[best] * (embeddings @ embeddings[best])
        if step:
            kernel -= factors[:, :step] @ factors[best, :step]
        new_factor = kernel / root
        factors[remaining, step] = new_factor[remaining]
        residual[remaining] -= new_factor[remaining] ** 2
        np.maximum(residual, 0.0, out=residual)
        factors[best, step] = root
    return selected, ranks, scores


def normalize_float32(vector: np.ndarray) -> np.ndarray:
    result = np.asarray(vector, dtype=np.float32)
    norm = np.float32(np.sqrt(np.sum(result * result, dtype=np.float32)))
    if float(norm) < 1e-10:
        return result
    return (result / norm).astype(np.float32, copy=False)


def drift_alpha(config: SelectorConfig, step: int) -> np.float32:
    base = np.float32(config.anchor_strength)
    elapsed = np.float32(step)
    half_life = np.float32(config.anchor_half_life)
    if config.anchor_schedule == "NONE":
        return base
    if config.anchor_schedule == "LINEAR":
        return base * np.maximum(
            np.float32(0.0),
            np.float32(1.0) - elapsed / (np.float32(2.0) * half_life),
        )
    if config.anchor_schedule == "EXPONENTIAL":
        return base * np.float32(0.5) ** (elapsed / half_life)
    if config.anchor_schedule == "STEP":
        return base if elapsed < half_life else base * np.float32(0.2)
    raise ValueError(f"unknown anchor schedule {config.anchor_schedule}")


def select_drift(
    library: queue_eval.Library,
    context: SeedContext,
    config: SelectorConfig,
) -> tuple[list[int], list[int], list[float], dict[str, object]]:
    seed = library.embeddings[context.seed_index]
    query = seed.copy()
    ema_state: np.ndarray | None = None
    selected: list[int] = []
    selected_scores: list[float] = []
    candidate_ranks: list[int] = []
    query_global_ranks: list[int] = []
    seen = np.zeros(library.count, dtype=np.bool_)
    seen[context.seed_index] = True
    alpha_values: list[float] = []
    step_candidate_counts: list[int] = []
    lambda_ = np.float32(config.mmr_lambda)

    for step in range(config.queue_size):
        similarities = (library.embeddings @ query).astype(np.float32, copy=False)
        positions = np.flatnonzero(~seen)
        order = np.lexsort((library.track_ids[positions], -similarities[positions]))
        actual_pool = min(effective_pool_count(library.count, config), positions.size)
        candidates = positions[order[:actual_pool]]
        step_candidate_counts.append(int(candidates.size))
        if candidates.size == 0:
            break
        relevance = similarities[candidates]
        eligible = np.fromiter(
            (artist_valid(library, selected, int(candidate), config) for candidate in candidates),
            dtype=np.bool_,
            count=candidates.size,
        )
        if not bool(eligible.any()):
            break
        if selected:
            pairwise = library.embeddings[candidates] @ library.embeddings[selected].T
            penalty = np.max(pairwise, axis=1).astype(np.float32, copy=False)
            objective = lambda_ * relevance - (np.float32(1.0) - lambda_) * penalty
            chosen_local = int(np.argmax(np.where(eligible, objective, -np.inf)))
        else:
            chosen_local = int(np.flatnonzero(eligible)[0])
        chosen = int(candidates[chosen_local])
        selected.append(chosen)
        selected_scores.append(float(relevance[chosen_local]))
        candidate_ranks.append(chosen_local + 1)
        query_order = np.lexsort((library.track_ids, -similarities))
        query_rank = int(np.flatnonzero(query_order == chosen)[0]) + 1
        query_global_ranks.append(query_rank)
        seen[chosen] = True
        current = library.embeddings[chosen]

        if config.drift_mode == "SEED_INTERPOLATION":
            alpha = drift_alpha(config, step)
            alpha_values.append(float(alpha))
            query = normalize_float32(alpha * seed + (np.float32(1.0) - alpha) * current)
            ema_state = current
        elif config.drift_mode == "MOMENTUM":
            previous = seed if ema_state is None else ema_state
            beta = np.float32(config.momentum_beta)
            ema_state = normalize_float32(
                beta * previous + (np.float32(1.0) - beta) * current
            )
            query = ema_state
        else:
            raise ValueError(f"unknown drift mode {config.drift_mode}")

    return selected, candidate_ranks, selected_scores, {
        "query_global_ranks": query_global_ranks,
        "effective_anchor_values": alpha_values,
        "step_candidate_counts": step_candidate_counts,
    }


def exact_graph_distribution(
    graph: selection_eval.Graph,
    transition,
    seed_index: int,
    alpha: float,
) -> GraphDistribution:
    started = time.perf_counter()
    n, k = graph.neighbors.shape
    current_nodes = graph.neighbors.ravel()
    probability = np.zeros(n * k, dtype=np.float64)
    seed_edges = np.arange(seed_index * k, (seed_index + 1) * k)
    probability[seed_edges] = 1.0 / k
    terminal = np.zeros(n, dtype=np.float64)
    terminal_link_mass = np.zeros(n, dtype=np.float64)
    alpha64 = float(np.float32(alpha))
    evaluated_links = 0
    for step in range(100):
        evaluated_links = step + 1
        stopped = probability if step == 99 else alpha64 * probability
        terminal += np.bincount(current_nodes, weights=stopped, minlength=n)
        terminal_link_mass += np.bincount(
            current_nodes, weights=(step + 1) * stopped, minlength=n
        )
        if step == 99:
            break
        probability = transition.dot(probability) * (1.0 - alpha64)
        if float(probability.sum()) == 0.0:
            break
    total_mass = float(terminal.sum())
    total_link_mass = float(terminal_link_mass.sum())
    return GraphDistribution(
        terminal=terminal,
        terminal_link_mass=terminal_link_mass,
        expected_links=total_link_mass / total_mass,
        excluded_seed_probability=float(terminal[seed_index]),
        total_mass=total_mass,
        mass_error=abs(1.0 - total_mass),
        evaluated_links=evaluated_links,
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
    )


def select_graph(
    library: queue_eval.Library,
    graph: selection_eval.Graph,
    context: SeedContext,
    config: SelectorConfig,
    distribution: GraphDistribution,
) -> tuple[list[int], list[int], list[float], dict[str, object]]:
    order = np.lexsort((graph.track_ids, -distribution.terminal))
    order = order[
        (order != context.seed_index) & (distribution.terminal[order] > 0.0)
    ]
    selected: list[int] = []
    ranks: list[int] = []
    scores: list[float] = []
    route_links: list[float] = []
    for rank, raw_index in enumerate(order, start=1):
        index = int(raw_index)
        probability = float(distribution.terminal[index])
        if not artist_valid(library, selected, index, config):
            continue
        selected.append(index)
        ranks.append(rank)
        scores.append(probability)
        route_links.append(float(distribution.terminal_link_mass[index] / probability))
        if len(selected) == config.queue_size:
            break
    positive = distribution.terminal > 0.0
    probabilities = distribution.terminal[positive]
    support = float(1.0 / np.sum(probabilities * probabilities))
    return selected, ranks, scores, {
        "graph_route_links": route_links,
        "graph_expected_links": distribution.expected_links,
        "graph_excluded_seed_probability": distribution.excluded_seed_probability,
        "graph_total_mass": distribution.total_mass,
        "graph_mass_error": distribution.mass_error,
        "graph_evaluated_links": distribution.evaluated_links,
        "graph_effective_support": support,
        "graph_top_queue_probability_mass": float(sum(scores)),
    }


def excess_duplicates(values: Iterable[str]) -> int:
    seen: set[str] = set()
    excess = 0
    for value in values:
        if not value:
            continue
        if value in seen:
            excess += 1
        else:
            seen.add(value)
    return excess


def queue_metrics(
    library: queue_eval.Library,
    context: SeedContext,
    selected: Sequence[int],
    candidate_ranks: Sequence[int],
    config: SelectorConfig,
    extra: dict[str, object],
) -> dict[str, object]:
    result: dict[str, object] = {
        "requested": config.queue_size,
        "returned": len(selected),
        "shortfall": config.queue_size - len(selected),
    }
    if not selected:
        return result
    selected_array = np.asarray(selected, dtype=np.int64)
    seed_cosines = context.seed_similarities[selected_array]
    result.update(
        {
            "mean_seed_cosine": float(np.mean(seed_cosines)),
            "p05_seed_cosine": float(np.percentile(seed_cosines, 5)),
            "minimum_seed_cosine": float(np.min(seed_cosines)),
            "median_seed_rank": float(np.median(context.seed_rank[selected_array])),
            "maximum_seed_rank": int(np.max(context.seed_rank[selected_array])),
            "median_candidate_rank": float(np.median(candidate_ranks)),
            "maximum_candidate_rank": int(max(candidate_ranks)),
        }
    )
    embeddings = library.embeddings[selected_array]
    if len(selected) > 1:
        gram = embeddings @ embeddings.T
        upper = gram[np.triu_indices(len(selected), k=1)]
        adjacent = np.sum(embeddings[:-1] * embeddings[1:], axis=1)
        result.update(
            {
                "mean_pairwise_cosine": float(np.mean(upper)),
                "p05_pairwise_cosine": float(np.percentile(upper, 5)),
                "mean_adjacent_cosine": float(np.mean(adjacent)),
                "p05_adjacent_cosine": float(np.percentile(adjacent, 5)),
                "minimum_adjacent_cosine": float(np.min(adjacent)),
            }
        )

    artists = [normalized_artist(library.artists[index]) or "" for index in selected]
    metadata = [library.metadata_keys[index] or "" for index in selected]
    filenames = [library.filename_keys[index] or "" for index in selected]
    artist_titles = [
        f"{normalized_text(library.artists[index])}|{normalized_text(library.titles[index])}"
        for index in selected
    ]
    seed = context.seed_index
    seed_artist_title = (
        f"{normalized_text(library.artists[seed])}|{normalized_text(library.titles[seed])}"
    )
    copy_slots: set[int] = set()
    seen_artist_titles = {seed_artist_title}
    for position, key in enumerate(artist_titles):
        if key != "|" and key in seen_artist_titles:
            copy_slots.add(position)
        elif key != "|":
            seen_artist_titles.add(key)
    result.update(
        {
            "unique_artist_credits": len({value for value in artists if value}),
            "repeated_artist_slots": excess_duplicates(artists),
            "exact_metadata_duplicate_slots": excess_duplicates(metadata),
            "exact_filename_duplicate_slots": excess_duplicates(filenames),
            "artist_title_proxy_duplicate_slots": excess_duplicates(artist_titles),
            "seed_artist_title_proxy_replays": sum(
                key == seed_artist_title and key != "|" for key in artist_titles
            ),
            "copy_proxy_waste_slots": len(copy_slots),
        }
    )
    result.update(extra)
    return result


def track_record(
    library: queue_eval.Library,
    context: SeedContext,
    index: int,
    rank: int,
    candidate_rank: int,
    score: float,
) -> dict[str, object]:
    return {
        "rank": rank,
        "track_id": int(library.track_ids[index]),
        "artist": library.artists[index],
        "album": library.albums[index],
        "title": library.titles[index],
        "duration_ms": int(library.durations_ms[index]),
        "file_path": library.file_paths[index],
        "metadata_key": library.metadata_keys[index],
        "filename_key": library.filename_keys[index],
        "candidate_rank": int(candidate_rank),
        "seed_rank": int(context.seed_rank[index]),
        "selection_score": float(score),
        "similarity_to_seed": float(context.seed_similarities[index]),
    }


def run_config(
    library: queue_eval.Library,
    graph: selection_eval.Graph,
    graph_transition,
    context: SeedContext,
    config: SelectorConfig,
    graph_cache: dict[tuple[int, float], GraphDistribution] | None = None,
) -> dict[str, object]:
    started = time.perf_counter()
    retrieval_ms = 0.0
    selection_started = time.perf_counter()
    extra: dict[str, object] = {}

    if config.mode == "closest":
        selected, ranks, scores, candidate_count = select_closest(library, context, config)
    elif config.mode in {"mmr", "dpp"}:
        retrieval_started = time.perf_counter()
        candidates, relevance, candidate_count = candidates_for(library, context, config)
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000.0
        selection_started = time.perf_counter()
        if config.mode == "mmr":
            selected, ranks, scores = select_mmr(library, candidates, relevance, config)
        else:
            selected, ranks, scores = select_dpp(library, candidates, relevance, config)
    elif config.mode == "drift":
        candidate_count = effective_pool_count(library.count, config)
        selected, ranks, scores, extra = select_drift(library, context, config)
    elif config.mode == "graph":
        candidate_count = graph.count - 1
        key = (context.seed_index, float(np.float32(config.graph_alpha)))
        distribution = graph_cache.get(key) if graph_cache is not None else None
        if distribution is None:
            distribution = exact_graph_distribution(
                graph, graph_transition, context.seed_index, config.graph_alpha
            )
            if graph_cache is not None:
                graph_cache[key] = distribution
        extra["graph_propagation_ms"] = distribution.elapsed_ms
        selection_started = time.perf_counter()
        selected, ranks, scores, graph_extra = select_graph(
            library, graph, context, config, distribution
        )
        extra.update(graph_extra)
    else:
        raise ValueError(config.mode)

    selection_ms = (time.perf_counter() - selection_started) * 1000.0
    total_ms = (time.perf_counter() - started) * 1000.0
    metrics = queue_metrics(library, context, selected, ranks, config, extra)
    track_ids = [int(library.track_ids[index]) for index in selected]
    return {
        "config": asdict(config),
        "actual_candidate_count": int(candidate_count),
        "track_ids": track_ids,
        "candidate_ranks": [int(value) for value in ranks],
        "selection_scores": [float(value) for value in scores],
        "queue_fingerprint": stable_json_hash(track_ids),
        "metrics": metrics,
        "timing_ms": {
            "retrieval_slice": retrieval_ms,
            "selection": selection_ms,
            "total_from_precomputed_seed_scan": total_ms,
        },
        "tracks": [
            track_record(library, context, index, rank, candidate_rank, score)
            for rank, (index, candidate_rank, score) in enumerate(
                zip(selected, ranks, scores), start=1
            )
        ],
    }


def slug(value: float | int | str) -> str:
    if isinstance(value, float):
        text = f"{value:.8g}"
    else:
        text = str(value).lower()
    return text.replace("-", "neg").replace(".", "p")


def build_cases() -> list[MatrixCase]:
    by_config: dict[SelectorConfig, MatrixCase] = {}

    def add(
        case_id: str,
        config: SelectorConfig,
        sweep: SweepPoint | None = None,
    ) -> None:
        existing = by_config.get(config)
        if existing is None:
            existing = MatrixCase(case_id=case_id, config=config)
            by_config[config] = existing
        elif case_id != existing.case_id and case_id not in existing.aliases:
            existing.aliases.append(case_id)
        if sweep is not None and sweep not in existing.sweeps:
            existing.sweeps.append(sweep)

    closest = SelectorConfig(mode="closest")
    mmr = SelectorConfig(mode="mmr")
    dpp = SelectorConfig(mode="dpp")
    graph = SelectorConfig(mode="graph")
    seed_drift = SelectorConfig(mode="drift", drift_mode="SEED_INTERPOLATION")
    momentum = SelectorConfig(mode="drift", drift_mode="MOMENTUM")
    for name, config in (
        ("closest_default", closest),
        ("mmr_default", mmr),
        ("dpp_default", dpp),
        ("graph_default", graph),
        ("drift_seed_default", seed_drift),
        ("drift_momentum_default", momentum),
    ):
        add(name, config)

    for reach_position, reach in enumerate(REACH_VALUES):
        for lambda_position, lambda_ in enumerate(MMR_LAMBDAS):
            config = replace(mmr, reach=reach, mmr_lambda=lambda_)
            name = f"mmr_lambda_{slug(lambda_)}_reach_{slug(reach)}"
            add(
                name,
                config,
                SweepPoint(
                    f"mmr.lambda.at_reach_{reach:g}", "mmr_lambda", lambda_, lambda_position
                ),
            )
            add(
                name,
                config,
                SweepPoint(
                    f"mmr.reach.at_lambda_{lambda_:g}", "reach", reach, reach_position
                ),
            )

    for reach_position, reach in enumerate(REACH_VALUES):
        for exponent_position, exponent in enumerate(DPP_EXPONENTS):
            config = replace(dpp, reach=reach, dpp_exponent=exponent)
            name = f"dpp_exponent_{slug(exponent)}_reach_{slug(reach)}"
            add(
                name,
                config,
                SweepPoint(
                    f"dpp.exponent.at_reach_{reach:g}",
                    "dpp_exponent",
                    exponent,
                    exponent_position,
                ),
            )
            add(
                name,
                config,
                SweepPoint(
                    f"dpp.reach.at_exponent_{exponent:g}", "reach", reach, reach_position
                ),
            )

    for position, alpha in enumerate(GRAPH_ALPHAS):
        add(
            f"graph_alpha_{slug(alpha)}",
            replace(graph, graph_alpha=alpha),
            SweepPoint("graph.stop_chance", "graph_alpha", alpha, position),
        )

    schedules = ("NONE", "LINEAR", "EXPONENTIAL", "STEP")
    for position, schedule in enumerate(schedules):
        add(
            f"drift_seed_schedule_{schedule.lower()}",
            replace(seed_drift, anchor_schedule=schedule),
            SweepPoint("drift_seed.schedule", "anchor_schedule", schedule, position),
        )
    for schedule in ("LINEAR", "EXPONENTIAL", "STEP"):
        for position, half_life in enumerate(HALF_LIFE_VALUES):
            add(
                f"drift_seed_{schedule.lower()}_half_{slug(half_life)}",
                replace(
                    seed_drift,
                    anchor_schedule=schedule,
                    anchor_half_life=half_life,
                ),
                SweepPoint(
                    f"drift_seed.half_life.{schedule.lower()}",
                    "anchor_half_life",
                    half_life,
                    position,
                ),
            )
    for position, anchor in enumerate(ANCHOR_VALUES):
        add(
            f"drift_seed_anchor_{slug(anchor)}",
            replace(seed_drift, anchor_strength=anchor),
            SweepPoint("drift_seed.anchor_strength", "anchor_strength", anchor, position),
        )
    for position, reach in enumerate(REACH_VALUES):
        add(
            f"drift_seed_reach_{slug(reach)}",
            replace(seed_drift, reach=reach),
            SweepPoint("drift_seed.reach", "reach", reach, position),
        )

    for position, beta in enumerate(MOMENTUM_VALUES):
        add(
            f"drift_momentum_beta_{slug(beta)}",
            replace(momentum, momentum_beta=beta),
            SweepPoint("drift_momentum.beta", "momentum_beta", beta, position),
        )
    for position, reach in enumerate(REACH_VALUES):
        add(
            f"drift_momentum_reach_{slug(reach)}",
            replace(momentum, reach=reach),
            SweepPoint("drift_momentum.reach", "reach", reach, position),
        )

    closest_off = replace(closest, artist_limits=False)
    add(
        "closest_artist_off",
        closest_off,
        SweepPoint("artist.closest.toggle", "artist_policy", "off", 0),
    )
    add(
        "closest_artist_default",
        closest,
        SweepPoint("artist.closest.toggle", "artist_policy", "default_8_3", 1),
    )
    for max_per_artist in (1, 2, 4, 8, 10):
        for spacing in (0, 1, 3, 5, 10, 20):
            config = replace(
                closest,
                max_per_artist=max_per_artist,
                artist_spacing=spacing,
            )
            name = f"closest_artist_max_{max_per_artist}_spacing_{spacing}"
            add(
                name,
                config,
                SweepPoint(
                    f"artist.closest.spacing.at_max_{max_per_artist}",
                    "artist_spacing",
                    spacing,
                    (0, 1, 3, 5, 10, 20).index(spacing),
                ),
            )
            add(
                name,
                config,
                SweepPoint(
                    f"artist.closest.max.at_spacing_{spacing}",
                    "max_per_artist",
                    max_per_artist,
                    (1, 2, 4, 8, 10).index(max_per_artist),
                ),
            )

    for algorithm, base in (
        ("mmr", mmr),
        ("dpp", dpp),
        ("graph", graph),
        ("drift_seed", seed_drift),
        ("drift_momentum", momentum),
    ):
        add(
            f"{algorithm}_artist_off",
            replace(base, artist_limits=False),
            SweepPoint(f"artist.{algorithm}.policy", "artist_policy", "off", 0),
        )
        add(
            f"{algorithm}_artist_default",
            base,
            SweepPoint(f"artist.{algorithm}.policy", "artist_policy", "default_8_3", 1),
        )
        add(
            f"{algorithm}_artist_tight_1_5",
            replace(base, max_per_artist=1, artist_spacing=5),
            SweepPoint(f"artist.{algorithm}.policy", "artist_policy", "tight_1_5", 2),
        )

    for algorithm, base in (
        ("closest", closest),
        ("mmr", mmr),
        ("dpp", dpp),
        ("graph", graph),
        ("drift_seed", seed_drift),
        ("drift_momentum", momentum),
    ):
        for position, size in enumerate(QUEUE_SIZES):
            add(
                f"{algorithm}_queue_{size}",
                replace(base, queue_size=size),
                SweepPoint(f"queue_size.{algorithm}", "queue_size", size, position),
            )

    return sorted(by_config.values(), key=lambda value: value.case_id)


def config_from_phone(value: dict[str, object]) -> SelectorConfig:
    phone_oracle.validate_clean_selector_config(value)
    selection_mode = str(value["selectionMode"])
    drift_enabled = bool(value["driftEnabled"])
    if selection_mode == "CLOSEST":
        mode = "closest"
        drift_mode = None
    elif selection_mode == "MMR" and drift_enabled:
        mode = "drift"
        drift_mode = str(value["driftMode"])
    elif selection_mode == "MMR":
        mode = "mmr"
        drift_mode = None
    elif selection_mode == "DPP":
        mode = "dpp"
        drift_mode = None
    elif selection_mode == "RANDOM_WALK":
        mode = "graph"
        drift_mode = None
    else:
        raise ValueError(f"unsupported phone parity mode {selection_mode}")

    dpp_uses_certified_full_domain = mode == "dpp" and (
        phone_oracle.certified_full_domain_dpp(value)
    )
    if mode in {"mmr", "drift"}:
        reach = phone_oracle.selector_pool_fraction(value, "mmr")
    elif mode == "dpp" and dpp_uses_certified_full_domain:
        reach = 1.0
    elif mode == "dpp":
        reach = phone_oracle.selector_pool_fraction(value, "dpp")
    else:
        reach = DEFAULT_REACH
    return SelectorConfig(
        mode=mode,
        queue_size=int(value["numTracks"]),
        reach=reach,
        candidate_pool_size=int(value["candidatePoolSize"]),
        dpp_uses_certified_full_domain=dpp_uses_certified_full_domain,
        mmr_lambda=float(value["diversityLambda"]),
        dpp_exponent=float(value["dppQualityExponent"]),
        graph_alpha=float(value["walkRestartAlpha"]),
        drift_mode=drift_mode,
        anchor_schedule=str(value["anchorDecay"]),
        anchor_strength=float(value["anchorStrength"]),
        anchor_half_life=float(value["anchorHalfLifeTracks"]),
        momentum_beta=float(value["momentumBeta"]),
        artist_limits=bool(value["artistLimitsEnabled"]),
        max_per_artist=int(value["maxPerArtist"]),
        artist_spacing=int(value["minArtistSpacing"]),
    )


def set_jaccard(left: Sequence[int], right: Sequence[int]) -> float:
    a, b = set(left), set(right)
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def queue_change(left: Sequence[int], right: Sequence[int]) -> dict[str, float | bool]:
    limit10 = min(10, len(left), len(right))
    limit = min(len(left), len(right))
    return {
        "exact_order_no_op": list(left) == list(right),
        "set_jaccard": set_jaccard(left, right),
        "top10_set_churn": (
            1.0 - len(set(left[:limit10]) & set(right[:limit10])) / limit10
            if limit10
            else 0.0
        ),
        "common_prefix_fraction": (
            sum(a == b for a, b in zip(left[:limit], right[:limit])) / limit if limit else 1.0
        ),
    }


def numeric_summary(values: Sequence[float]) -> dict[str, float | None]:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return {"mean": None, "min": None, "median": None, "max": None}
    return {
        "mean": float(np.mean(finite)),
        "min": float(np.min(finite)),
        "median": float(np.median(finite)),
        "max": float(np.max(finite)),
    }


def summarize_records(records: Sequence[dict[str, object]]) -> dict[str, object]:
    by_case: dict[str, list[dict[str, object]]] = defaultdict(list)
    by_seed_case: dict[tuple[int, str], dict[str, object]] = {}
    sweep_points: dict[str, dict[int, tuple[SweepPoint, str]]] = defaultdict(dict)
    for record in records:
        case_id = str(record["case_id"])
        seed_id = int(record["seed"]["track_id"])
        by_case[case_id].append(record)
        by_seed_case[(seed_id, case_id)] = record
        for raw in record["sweeps"]:
            point = SweepPoint(**raw)
            sweep_points[point.sweep_id][point.order] = (point, case_id)

    case_summary: dict[str, object] = {}
    metric_names = (
        "mean_seed_cosine",
        "mean_pairwise_cosine",
        "mean_adjacent_cosine",
        "unique_artist_credits",
        "copy_proxy_waste_slots",
        "median_seed_rank",
    )
    for case_id, items in sorted(by_case.items()):
        metrics: dict[str, object] = {}
        for name in metric_names:
            values = [float(item["metrics"][name]) for item in items if name in item["metrics"]]
            metrics[name] = numeric_summary(values)
        case_summary[case_id] = {
            "seed_count": len(items),
            "actual_candidate_count": sorted(
                {int(item["actual_candidate_count"]) for item in items}
            ),
            "returned": numeric_summary([float(item["metrics"]["returned"]) for item in items]),
            "runtime_ms": numeric_summary(
                [float(item["timing_ms"]["total_from_precomputed_seed_scan"]) for item in items]
            ),
            "metrics": metrics,
            "aliases": items[0]["aliases"],
            "config": items[0]["config"],
        }

    sweeps: dict[str, object] = {}
    seed_ids = sorted({seed for seed, _case in by_seed_case})
    for sweep_id, points_by_order in sorted(sweep_points.items()):
        points = [points_by_order[key] for key in sorted(points_by_order)]
        adjacent: list[dict[str, object]] = []
        for (left_point, left_case), (right_point, right_case) in zip(points, points[1:]):
            changes = [
                queue_change(
                    by_seed_case[(seed_id, left_case)]["track_ids"],
                    by_seed_case[(seed_id, right_case)]["track_ids"],
                )
                for seed_id in seed_ids
            ]
            no_ops = sum(bool(change["exact_order_no_op"]) for change in changes)
            adjacent.append(
                {
                    "from": left_point.value,
                    "to": right_point.value,
                    "from_case": left_case,
                    "to_case": right_case,
                    "seeds": len(changes),
                    "exact_no_op_seeds": no_ops,
                    "exact_no_op_rate": no_ops / len(changes),
                    "set_jaccard": numeric_summary(
                        [float(change["set_jaccard"]) for change in changes]
                    ),
                    "top10_set_churn": numeric_summary(
                        [float(change["top10_set_churn"]) for change in changes]
                    ),
                    "same_rank_fraction": numeric_summary(
                        [float(change["common_prefix_fraction"]) for change in changes]
                    ),
                }
            )
        sweeps[sweep_id] = {
            "control": points[0][0].control,
            "points": [point.value for point, _case in points],
            "cases": [case for _point, case in points],
            "adjacent": adjacent,
        }
    return {"cases": case_summary, "sweeps": sweeps}


def validate_phone_parity(
    library: queue_eval.Library,
    graph: selection_eval.Graph,
    graph_transition,
    contexts: dict[int, SeedContext],
    phone_report: dict[str, object],
) -> dict[str, object]:
    cache: dict[tuple[int, SelectorConfig], dict[str, object]] = {}
    graph_cache: dict[tuple[int, float], GraphDistribution] = {}
    comparisons: list[dict[str, object]] = []
    for run in phone_report["selectionRuns"]:
        if run["caseId"] == "uniform_shuffle":
            continue
        seed_id = int(run["seedTrackId"])
        config = config_from_phone(run["config"])
        dpp_evidence: dict[str, object] | None = None
        if config.mode == "dpp":
            dpp_evidence = phone_oracle.validate_dpp_selection_evidence(
                run,
                library.count - 1,
            )
        elif run.get("dppSelectionEvidence") is not None:
            raise ValueError(
                f"non-DPP case {run['caseId']} emitted DPP selection evidence"
            )
        key = (seed_id, config)
        result = cache.get(key)
        if result is None:
            result = run_config(
                library,
                graph,
                graph_transition,
                contexts[seed_id],
                config,
                graph_cache,
            )
            cache[key] = result
        actual_ids = [int(track["trackId"]) for track in run["tracks"]]
        expected_ids = result["track_ids"]
        actual_scores = [float(track["score"]) for track in run["tracks"]]
        expected_scores = result["selection_scores"]
        max_error = max(
            (abs(a - b) for a, b in zip(actual_scores, expected_scores)), default=0.0
        )
        comparisons.append(
            {
                "case_id": run["caseId"],
                "seed_track_id": seed_id,
                "repeat": int(run["repeat"]),
                "exact_track_id_order": actual_ids == expected_ids,
                "max_abs_score_error": max_error,
                "dpp_selection_evidence": dpp_evidence,
            }
        )
    exact = sum(bool(item["exact_track_id_order"]) for item in comparisons)
    return {
        "comparison_count": len(comparisons),
        "exact_order_count": exact,
        "all_exact": exact == len(comparisons),
        "maximum_abs_score_error": max(
            float(item["max_abs_score_error"]) for item in comparisons
        ),
        "comparisons": comparisons,
    }


def validate_determinism(
    library: queue_eval.Library,
    graph: selection_eval.Graph,
    graph_transition,
    contexts: dict[int, SeedContext],
) -> dict[str, object]:
    configs = (
        SelectorConfig(mode="closest"),
        SelectorConfig(mode="mmr"),
        SelectorConfig(mode="dpp"),
        SelectorConfig(mode="graph"),
        SelectorConfig(mode="drift", drift_mode="SEED_INTERPOLATION"),
        SelectorConfig(mode="drift", drift_mode="MOMENTUM"),
    )
    comparisons: list[dict[str, object]] = []
    for seed_id, context in contexts.items():
        first_graph_cache: dict[tuple[int, float], GraphDistribution] = {}
        second_graph_cache: dict[tuple[int, float], GraphDistribution] = {}
        for config in configs:
            first = run_config(
                library, graph, graph_transition, context, config, first_graph_cache
            )
            second = run_config(
                library, graph, graph_transition, context, config, second_graph_cache
            )
            comparisons.append(
                {
                    "seed_track_id": seed_id,
                    "mode": config.mode,
                    "drift_mode": config.drift_mode,
                    "exact_track_id_order": first["track_ids"] == second["track_ids"],
                    "first_fingerprint": first["queue_fingerprint"],
                    "second_fingerprint": second["queue_fingerprint"],
                }
            )
    exact = sum(bool(item["exact_track_id_order"]) for item in comparisons)
    return {
        "comparison_count": len(comparisons),
        "exact_order_count": exact,
        "all_exact": exact == len(comparisons),
        "comparisons": comparisons,
    }


def select_listening_cases(summary: dict[str, object]) -> list[str]:
    """Small parity/listening set spanning observed selector and control behavior."""
    required = [
        "closest_default",
        "closest_artist_off",
        "mmr_default",
        "mmr_lambda_0p8_reach_0p02",
        "dpp_default",
        "dpp_exponent_2_reach_0p05",
        "graph_default",
        "graph_alpha_0p5",
        "drift_seed_default",
        "drift_seed_schedule_none",
        "drift_momentum_default",
        "drift_momentum_beta_0p75",
    ]
    available = set(summary["cases"])
    return [case for case in required if case in available]


def write_listening_packet(
    path: Path,
    records: Sequence[dict[str, object]],
    selected_cases: Sequence[str],
) -> dict[str, object]:
    by_case_seed = {
        (str(record["case_id"]), int(record["seed"]["track_id"])): record
        for record in records
    }
    seed_ids = sorted({seed for _case, seed in by_case_seed})
    labels = {
        case_id: f"Q{position:02d}"
        for position, case_id in enumerate(
            sorted(selected_cases, key=lambda value: stable_json_hash(value)), start=1
        )
    }
    lines = [
        "# Selection Knob Matrix Listening Packet",
        "",
        "Queue labels are intentionally opaque. The separate key records exact configs.",
        "Every membership decision came only from audio embeddings or the audio graph;",
        "the labels and paths below are for listening and diagnostics.",
        "",
    ]
    for seed_id in seed_ids:
        seed_record = next(
            record for record in records if int(record["seed"]["track_id"]) == seed_id
        )
        seed = seed_record["seed"]
        lines.extend(
            [
                f"## Seed {seed_id}: {seed.get('artist') or 'Unknown'} - {seed.get('title') or 'Unknown'}",
                "",
            ]
        )
        for case_id in selected_cases:
            record = by_case_seed[(case_id, seed_id)]
            lines.append(f"### {labels[case_id]}")
            lines.append("")
            for track in record["tracks"]:
                lines.append(
                    f"{track['rank']}. {track.get('artist') or 'Unknown'} - "
                    f"{track.get('title') or 'Unknown'} | `{track.get('file_path') or ''}`"
                )
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "case_labels": labels,
        "cases": {
            labels[case_id]: {
                "case_id": case_id,
                "config": next(
                    record["config"] for record in records if record["case_id"] == case_id
                ),
            }
            for case_id in selected_cases
        },
        "seed_track_ids": seed_ids,
    }


def self_checks() -> dict[str, object]:
    assert effective_pool_count(80_323, SelectorConfig(mode="mmr", reach=0.0025)) == 200
    assert effective_pool_count(80_323, SelectorConfig(mode="mmr", reach=0.05)) == 4_016
    base = SelectorConfig(mode="drift", drift_mode="SEED_INTERPOLATION")
    assert float(drift_alpha(replace(base, anchor_schedule="NONE"), 29)) == float(
        np.float32(DEFAULT_ANCHOR)
    )
    assert np.isclose(
        float(drift_alpha(base, 7)),
        float(np.float32(DEFAULT_ANCHOR) * np.float32(0.5) ** (np.float32(7) / np.float32(DEFAULT_HALF_LIFE))),
    )
    assert normalized_artist("  ARTIST  ") == "artist"
    assert queue_change([1, 2, 3], [1, 2, 3])["exact_order_no_op"]
    assert not queue_change([1, 2, 3], [1, 3, 2])["exact_order_no_op"]
    return {
        "passed": 7,
        "checks": [
            "0.25% active reach rounds down to 200 actual candidates",
            "5% active reach rounds down to 4,016 actual candidates",
            "hold schedule is constant",
            "exponential schedule uses float32 absolute half-life math",
            "artist credit normalization matches current trim/lowercase contract",
            "identical queues are detected as control no-ops",
            "rank-only changes are not mislabeled as no-ops",
        ],
    }


def write_hashes(output: Path, input_hashes: dict[str, str]) -> None:
    lines = [f"{digest}  INPUT::{name}" for name, digest in sorted(input_hashes.items())]
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
    parser.add_argument("--self-check-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checks = self_checks()
    if args.self_check_only:
        print(json.dumps(checks, indent=2, sort_keys=True))
        return 0

    output = args.output.resolve()
    if output.exists():
        if not args.force:
            raise FileExistsError(f"output already exists: {output}; use --force")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    database_hash_before = sha256_file(args.database)
    catalog_hash = sha256_file(args.active_catalog)
    phone_hash = sha256_file(args.phone_report)
    if database_hash_before != queue_eval.EXPECTED_DB_SHA256:
        raise ValueError("frozen database hash mismatch")
    if catalog_hash != EXPECTED_CATALOG_SHA256:
        raise ValueError("active catalog hash mismatch")
    if phone_hash != EXPECTED_PHONE_REPORT_SHA256:
        raise ValueError("phone cohort report hash mismatch")

    phone_report = json.loads(args.phone_report.read_text(encoding="utf-8"))
    seed_ids = [int(value) for value in phone_report["request"]["seedTrackIds"]]
    active_ids = phone_oracle.active_track_ids(args.active_catalog)
    if len(active_ids) != EXPECTED_ACTIVE_TRACKS:
        raise ValueError(f"expected {EXPECTED_ACTIVE_TRACKS} active IDs, found {len(active_ids)}")

    started = time.perf_counter()
    full_library, loaded_hash = queue_eval.load_library(args.database)
    if loaded_hash != database_hash_before:
        raise ValueError("database changed while loading")
    active_mask = np.fromiter(
        (int(track_id) in active_ids for track_id in full_library.track_ids),
        dtype=np.bool_,
        count=full_library.count,
    )
    if int(active_mask.sum()) != EXPECTED_ACTIVE_TRACKS:
        raise ValueError("database/catalog active intersection is not 80,323 tracks")

    full_graph = selection_eval.parse_graph(args.database)
    active_graph, active_positions, _old_to_active, graph_repair = phone_oracle.build_active_graph(
        full_library, full_graph, active_mask
    )
    library = subset_library(full_library, active_positions)
    del full_library, full_graph, active_mask
    if not np.array_equal(active_graph.track_ids, library.track_ids):
        raise ValueError("active graph and active embedding library are not aligned")
    graph_transition = selection_eval.build_edge_transition(active_graph, weighted=False)

    id_to_index = {int(track_id): index for index, track_id in enumerate(library.track_ids)}
    missing_seeds = [seed_id for seed_id in seed_ids if seed_id not in id_to_index]
    if missing_seeds:
        raise ValueError(f"phone seed cohort is outside active domain: {missing_seeds}")
    contexts = {
        seed_id: seed_context(library, id_to_index[seed_id]) for seed_id in seed_ids
    }
    load_ms = (time.perf_counter() - started) * 1000.0

    print("Running frozen phone-parity gate...", file=sys.stderr)
    parity = validate_phone_parity(
        library, active_graph, graph_transition, contexts, phone_report
    )
    if not parity["all_exact"]:
        atomic_json(output / "phone-parity-failure.json", parity)
        raise AssertionError("canonical host replay does not reproduce every phone order")
    atomic_json(output / "phone-parity.json", parity)

    cases = build_cases()
    records: list[dict[str, object]] = []
    results_path = output / "full-lists.jsonl"
    graph_cache: dict[tuple[int, float], GraphDistribution] = {}
    matrix_started = time.perf_counter()
    for seed_position, seed_id in enumerate(seed_ids, start=1):
        context = contexts[seed_id]
        seed_summary = queue_eval.track_summary(library, context.seed_index)
        case_cache: dict[SelectorConfig, dict[str, object]] = {}
        for case in cases:
            result = case_cache.get(case.config)
            if result is None:
                result = run_config(
                    library,
                    active_graph,
                    graph_transition,
                    context,
                    case.config,
                    graph_cache,
                )
                case_cache[case.config] = result
            record = {
                "case_id": case.case_id,
                "aliases": case.aliases,
                "sweeps": [asdict(point) for point in case.sweeps],
                "seed": seed_summary,
                "seed_scan_ms": context.scan_ms,
                "seed_sort_ms": context.sort_ms,
                **result,
            }
            records.append(record)
            append_jsonl(results_path, record)
        print(
            f"matrix seed {seed_position}/{len(seed_ids)} track={seed_id} "
            f"cases={len(cases)} elapsed={(time.perf_counter() - matrix_started):.1f}s",
            file=sys.stderr,
        )

    summary = summarize_records(records)
    selected_cases = select_listening_cases(summary)
    listening_key = write_listening_packet(
        output / "listening-packet.md", records, selected_cases
    )
    atomic_json(output / "listening-key.json", listening_key)
    atomic_json(
        output / "representative-device-matrix.json",
        {
            "selection_basis": (
                "Small coverage set spanning baseline, artist toggle, measured mid/default "
                "MMR and DPP controls, short/long graph paths, and both drift controls."
            ),
            "case_ids": selected_cases,
            "cases": {case: summary["cases"][case] for case in selected_cases},
        },
    )
    atomic_json(output / "summary.json", summary)

    print("Running independent repeated-computation determinism gate...", file=sys.stderr)
    determinism = validate_determinism(
        library, active_graph, graph_transition, contexts
    )
    if not determinism["all_exact"]:
        atomic_json(output / "determinism-failure.json", determinism)
        raise AssertionError("repeated canonical computations were not identical")
    atomic_json(output / "determinism.json", determinism)

    database_hash_after = sha256_file(args.database)
    if database_hash_after != database_hash_before:
        raise AssertionError("frozen database changed during the experiment")
    manifest = {
        "experiment_version": EXPERIMENT_VERSION,
        "created_date": "2026-07-15",
        "host": {
            "platform": platform.platform(),
            "python": sys.version,
            "numpy": np.__version__,
            "openblas_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
            "omp_threads": os.environ.get("OMP_NUM_THREADS"),
        },
        "inputs": {
            "database": str(args.database.resolve()),
            "database_sha256_before": database_hash_before,
            "database_sha256_after": database_hash_after,
            "active_catalog": str(args.active_catalog.resolve()),
            "active_catalog_sha256": catalog_hash,
            "phone_report": str(args.phone_report.resolve()),
            "phone_report_sha256": phone_hash,
        },
        "domain": {
            "database_tracks": queue_eval.EXPECTED_TRACKS,
            "active_tracks": library.count,
            "dimensions": library.dim,
            "seed_track_ids": seed_ids,
            "active_graph_repair": graph_repair,
        },
        "matrix": {
            "unique_configurations": len(cases),
            "seed_count": len(seed_ids),
            "result_count": len(records),
            "reach_values": REACH_VALUES,
            "mmr_lambdas": MMR_LAMBDAS,
            "dpp_exponents": DPP_EXPONENTS,
            "graph_alphas": GRAPH_ALPHAS,
            "anchor_values": ANCHOR_VALUES,
            "half_life_values": HALF_LIFE_VALUES,
            "momentum_values": MOMENTUM_VALUES,
            "queue_sizes": QUEUE_SIZES,
        },
        "gates": {
            "self_checks": checks,
            "phone_parity": {
                key: value for key, value in parity.items() if key != "comparisons"
            },
            "determinism": {
                key: value for key, value in determinism.items() if key != "comparisons"
            },
        },
        "timing_ms": {
            "load_inputs_and_build_active_graph": load_ms,
            "matrix": (time.perf_counter() - matrix_started) * 1000.0,
            "whole_run": (time.perf_counter() - started) * 1000.0,
        },
        "metadata_policy": (
            "Metadata never enters relevance or transition scores. It is used only for "
            "the explicit artist-credit eligibility rule, seed labels, copy proxies, and paths."
        ),
    }
    atomic_json(output / "manifest.json", manifest)
    write_hashes(
        output,
        {
            "database": database_hash_before,
            "active_catalog": catalog_hash,
            "phone_report": phone_hash,
            "evaluator": sha256_file(Path(__file__)),
        },
    )
    print(
        f"Complete: {len(records)} results, {len(cases)} cases x {len(seed_ids)} seeds, "
        f"output={output}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
