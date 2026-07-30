#!/usr/bin/env python3
"""Evaluate seed participation separately from recording-copy suppression.

This is host-only experiment code. It never changes the immutable phone database and
does not redefine any released selector:

* ``canonical_mmr`` is ordinary MMR over already selected recommendations;
* ``seed_history_mmr`` is a separately named experiment that also puts the playing
  seed in MMR's redundancy history;
* ``canonical_dpp`` is the released greedy DPP MAP objective;
* ``seed_conditioned_dpp`` is a separately named conditional-DPP experiment whose
  initial Schur complement projects out the playing seed.

The optional cosine/duration near-copy rule is deliberately labelled a *diagnostic*.
It is included to measure the tempting shortcut, not to propose it as recording
identity. Decoded-audio identity is evaluated by ``v2_recording_identity_eval.py``.
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

import compare_device_feature_acceptance as device_eval
import v2_queue_eval as queue_eval
import v2_selection_mode_eval as selection_eval


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = (
    REPO_ROOT
    / "desktop-indexer"
    / "audit_raw_data"
    / "phone-snapshots"
    / "2026-07-07T223308+0300_qv7706c3mq"
    / "embeddings.db"
)
DEFAULT_DEVICE_RUN = (
    REPO_ROOT
    / "discovery"
    / "device-acceptance"
    / "20260714T-multiseed-r2c-mmr-fixed-feature-acceptance"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "desktop-indexer"
    / "audit_raw_data"
    / "v2-discovery"
    / "seed-conditioning"
)

EXPERIMENT_VERSION = "seed-conditioning-active-domain-v1"
QUEUE_SIZE = 30
USER_MMR_LAMBDA = 0.97032166
MAX_PER_ARTIST = 8
MIN_ARTIST_SPACING = 3
REACH_FRACTIONS = (0.0025, 0.005, 0.01, 0.02, 0.04)
DPP_EXPONENTS = (0.0, 0.5, 1.0, 2.0, 4.0)


@dataclass(frozen=True)
class PickResult:
    selected: tuple[int, ...]
    candidate_ranks: tuple[int, ...]
    objective_values: tuple[float, ...]
    near_copy_rejections: int = 0


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
            ensure_ascii=True,
            sort_keys=True,
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
                ensure_ascii=True,
                sort_keys=True,
                default=queue_eval.json_numpy_scalar,
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def completed_seed_ids(path: Path) -> set[int]:
    if not path.exists():
        return set()
    return {
        int(json.loads(line)["seed"]["track_id"])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def active_mask(library: queue_eval.Library, active_catalog: Path) -> np.ndarray:
    active_ids = device_eval.active_track_ids(active_catalog)
    result = np.fromiter(
        (int(track_id) in active_ids for track_id in library.track_ids),
        dtype=np.bool_,
        count=library.count,
    )
    if int(result.sum()) != len(active_ids):
        raise ValueError("active catalog contains IDs absent from the frozen database")
    return result


def cohort_from_report(
    library: queue_eval.Library,
    report_path: Path,
    active: np.ndarray,
) -> tuple[list[int], dict[int, dict[str, object]]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    by_id = {int(track_id): index for index, track_id in enumerate(library.track_ids)}
    seed_ids = [int(value) for value in report["request"]["seedTrackIds"]]
    seeds: list[int] = []
    configs: dict[int, dict[str, object]] = {}
    for track_id in seed_ids:
        index = by_id.get(track_id)
        if index is None or not bool(active[index]):
            raise ValueError(f"device seed {track_id} is not in the active frozen domain")
        seeds.append(index)
    for run in report.get("selectionRuns", []):
        if int(run.get("repeat", 0)) != 1:
            continue
        seed_id = int(run["seedTrackId"])
        configs.setdefault(seed_id, {})[str(run["caseId"])] = run["config"]
    return seeds, configs


def retrieve_active(
    library: queue_eval.Library,
    seed_index: int,
    active: np.ndarray,
    count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    similarities = (library.embeddings @ library.embeddings[seed_index]).astype(
        np.float32, copy=False
    )
    eligible = active.copy()
    eligible[seed_index] = False
    positions = np.flatnonzero(eligible)
    order = np.lexsort((library.track_ids[positions], -similarities[positions]))
    candidates = positions[order[: min(count, positions.size)]].astype(np.int64)
    return candidates, similarities[candidates], similarities


def duration_compatible(left_ms: int, right_ms: int) -> bool:
    """Broad diagnostic gate, intentionally not a recording-identity contract."""
    if left_ms < 0 or right_ms < 0:
        return False
    tolerance = max(2_000.0, 0.02 * max(left_ms, right_ms))
    return abs(left_ms - right_ms) <= tolerance


def diagnostic_near_copy(
    library: queue_eval.Library,
    left: int,
    right: int,
    cosine: float,
    threshold: float,
) -> bool:
    return cosine >= threshold and duration_compatible(
        int(library.durations_ms[left]), int(library.durations_ms[right])
    )


def artist_eligible(
    library: queue_eval.Library,
    selected: Sequence[int],
    candidate: int,
) -> bool:
    return queue_eval.can_add_artist(
        candidate,
        selected,
        library,
        MAX_PER_ARTIST,
        MIN_ARTIST_SPACING,
    )


def diagnostic_copy_eligible(
    library: queue_eval.Library,
    seed_index: int,
    selected: Sequence[int],
    candidate: int,
    seed_relevance: float,
    candidate_similarities: np.ndarray | None,
    threshold: float | None,
) -> bool:
    if threshold is None:
        return True
    if diagnostic_near_copy(
        library, seed_index, candidate, seed_relevance, threshold
    ):
        return False
    if candidate_similarities is None:
        return True
    return not any(
        diagnostic_near_copy(
            library,
            selected[position],
            candidate,
            float(candidate_similarities[position]),
            threshold,
        )
        for position in range(len(selected))
    )


def select_closest(
    library: queue_eval.Library,
    seed_index: int,
    candidates: np.ndarray,
    relevance: np.ndarray,
    count: int,
    near_copy_threshold: float | None = None,
) -> PickResult:
    selected: list[int] = []
    ranks: list[int] = []
    objectives: list[float] = []
    rejected = 0
    for local, raw_candidate in enumerate(candidates):
        candidate = int(raw_candidate)
        selected_similarities = (
            library.embeddings[np.asarray(selected)] @ library.embeddings[candidate]
            if selected
            else None
        )
        if not diagnostic_copy_eligible(
            library,
            seed_index,
            selected,
            candidate,
            float(relevance[local]),
            selected_similarities,
            near_copy_threshold,
        ):
            rejected += 1
            continue
        if not artist_eligible(library, selected, candidate):
            continue
        selected.append(candidate)
        ranks.append(local + 1)
        objectives.append(float(relevance[local]))
        if len(selected) == count:
            break
    return PickResult(tuple(selected), tuple(ranks), tuple(objectives), rejected)


def select_mmr(
    library: queue_eval.Library,
    seed_index: int,
    candidates: np.ndarray,
    relevance: np.ndarray,
    gram: np.ndarray,
    count: int,
    lambda_: float,
    include_seed_in_history: bool,
    near_copy_threshold: float | None = None,
) -> PickResult:
    remaining = np.ones(candidates.size, dtype=np.bool_)
    maximum_similarity = (
        relevance.astype(np.float32, copy=True)
        if include_seed_in_history
        else np.full(candidates.size, -np.inf, dtype=np.float32)
    )
    selected: list[int] = []
    selected_local: list[int] = []
    ranks: list[int] = []
    objectives: list[float] = []
    rejected_tokens: set[int] = set()

    for step in range(min(count, candidates.size)):
        penalty: float | np.ndarray
        if step == 0 and not include_seed_in_history:
            penalty = 0.0
        else:
            penalty = maximum_similarity
        scores = lambda_ * relevance - (1.0 - lambda_) * penalty
        eligible = remaining.copy()
        for local in np.flatnonzero(eligible):
            candidate = int(candidates[local])
            selected_sims = (
                gram[local, np.asarray(selected_local)] if selected_local else None
            )
            if not diagnostic_copy_eligible(
                library,
                seed_index,
                selected,
                candidate,
                float(relevance[local]),
                selected_sims,
                near_copy_threshold,
            ):
                eligible[local] = False
                rejected_tokens.add(int(local))
            elif not artist_eligible(library, selected, candidate):
                eligible[local] = False
        if not eligible.any():
            break
        best = int(np.argmax(np.where(eligible, scores, -np.inf)))
        selected.append(int(candidates[best]))
        selected_local.append(best)
        ranks.append(best + 1)
        objectives.append(float(scores[best]))
        remaining[best] = False
        maximum_similarity = np.maximum(maximum_similarity, gram[:, best])
    return PickResult(
        tuple(selected), tuple(ranks), tuple(objectives), len(rejected_tokens)
    )


def select_dpp(
    library: queue_eval.Library,
    seed_index: int,
    candidates: np.ndarray,
    relevance: np.ndarray,
    gram: np.ndarray,
    count: int,
    quality_exponent: float,
    condition_on_seed: bool,
    near_copy_threshold: float | None = None,
) -> PickResult:
    quality = np.power(
        np.maximum(relevance.astype(np.float64), 0.0), quality_exponent
    ).astype(np.float32)
    factor_columns = count + (1 if condition_on_seed else 0)
    factors = np.zeros((candidates.size, factor_columns), dtype=np.float32)
    diagonal = quality * quality
    used_columns = 0
    if condition_on_seed:
        # For L_ij=q_i q_j <x_i,x_j>, conditioning on the already selected
        # unit seed gives L^s_ij=q_i q_j(<x_i,x_j>-<x_i,s><x_j,s>).
        # The seed quality cancels from the Schur complement.
        factors[:, 0] = quality * relevance
        diagonal -= factors[:, 0] ** 2
        np.maximum(diagonal, 0.0, out=diagonal)
        used_columns = 1

    remaining = np.ones(candidates.size, dtype=np.bool_)
    selected: list[int] = []
    selected_local: list[int] = []
    ranks: list[int] = []
    gains: list[float] = []
    rejected_tokens: set[int] = set()

    for _ in range(min(count, candidates.size)):
        eligible = remaining.copy()
        for local in np.flatnonzero(eligible):
            candidate = int(candidates[local])
            selected_sims = (
                gram[local, np.asarray(selected_local)] if selected_local else None
            )
            if not diagnostic_copy_eligible(
                library,
                seed_index,
                selected,
                candidate,
                float(relevance[local]),
                selected_sims,
                near_copy_threshold,
            ):
                eligible[local] = False
                rejected_tokens.add(int(local))
            elif not artist_eligible(library, selected, candidate):
                eligible[local] = False
        if not eligible.any():
            break
        best = int(np.argmax(np.where(eligible, diagonal, -np.inf)))
        best_gain = float(diagonal[best])
        if not math.isfinite(best_gain) or best_gain <= 1e-10:
            break
        selected.append(int(candidates[best]))
        selected_local.append(best)
        ranks.append(best + 1)
        gains.append(best_gain)
        remaining[best] = False

        root = math.sqrt(best_gain)
        kernel = quality * quality[best] * gram[:, best]
        if used_columns:
            kernel -= factors[:, :used_columns] @ factors[best, :used_columns]
        new_factor = kernel / root
        factors[remaining, used_columns] = new_factor[remaining]
        diagonal[remaining] -= new_factor[remaining] ** 2
        np.maximum(diagonal, 0.0, out=diagonal)
        factors[best, used_columns] = root
        used_columns += 1

    return PickResult(
        tuple(selected), tuple(ranks), tuple(gains), len(rejected_tokens)
    )


def drift_select(
    library: queue_eval.Library,
    seed_index: int,
    active: np.ndarray,
    config: dict[str, object],
    include_seed_in_history: bool,
    near_copy_threshold: float | None = None,
) -> PickResult:
    seed_embedding = library.embeddings[seed_index]
    query = seed_embedding.copy()
    ema_state: np.ndarray | None = None
    selected: list[int] = []
    selected_embeddings: list[np.ndarray] = []
    ranks: list[int] = []
    objectives: list[float] = []
    rejected = 0
    pool_size = device_eval.resolved_candidate_pool_size(
        config,
        "mmr",
        int(config["numTracks"]),
        int(active.sum()),
    )

    for step in range(int(config["numTracks"])):
        similarities = (library.embeddings @ query).astype(np.float32, copy=False)
        eligible_domain = active.copy()
        eligible_domain[seed_index] = False
        if selected:
            eligible_domain[np.asarray(selected)] = False
        positions = np.flatnonzero(eligible_domain)
        order = np.lexsort((library.track_ids[positions], -similarities[positions]))
        candidates = positions[order[: min(pool_size, positions.size)]]
        relevance = similarities[candidates]
        candidate_embeddings = library.embeddings[candidates]
        if selected_embeddings:
            selected_matrix = np.stack(selected_embeddings)
            penalties = np.max(candidate_embeddings @ selected_matrix.T, axis=1)
            if include_seed_in_history:
                penalties = np.maximum(
                    penalties, candidate_embeddings @ seed_embedding
                )
        elif include_seed_in_history:
            penalties = candidate_embeddings @ seed_embedding
        else:
            penalties = np.zeros(candidates.size, dtype=np.float32)
        scores = (
            float(config["diversityLambda"]) * relevance
            - (1.0 - float(config["diversityLambda"])) * penalties
        )
        chosen_local: int | None = None
        for raw_local in np.lexsort((library.track_ids[candidates], -scores)):
            local = int(raw_local)
            candidate = int(candidates[local])
            seed_cosine = float(library.embeddings[candidate] @ seed_embedding)
            selected_sims = (
                np.asarray(
                    [
                        float(library.embeddings[candidate] @ embedding)
                        for embedding in selected_embeddings
                    ],
                    dtype=np.float32,
                )
                if selected_embeddings
                else None
            )
            if not diagnostic_copy_eligible(
                library,
                seed_index,
                selected,
                candidate,
                seed_cosine,
                selected_sims,
                near_copy_threshold,
            ):
                rejected += 1
                continue
            if artist_eligible(library, selected, candidate):
                chosen_local = local
                break
        if chosen_local is None:
            break
        chosen = int(candidates[chosen_local])
        selected.append(chosen)
        selected_embeddings.append(library.embeddings[chosen])
        ranks.append(chosen_local + 1)
        objectives.append(float(scores[chosen_local]))

        current = library.embeddings[chosen]
        if str(config["driftMode"]) == "SEED_INTERPOLATION":
            base = float(config["anchorStrength"])
            half_life = float(config["anchorHalfLifeTracks"])
            alpha = base * math.pow(0.5, step / half_life)
            query = alpha * seed_embedding + (1.0 - alpha) * current
            norm = float(np.linalg.norm(query))
            if norm > 1e-10:
                query = (query / norm).astype(np.float32)
        elif str(config["driftMode"]) == "MOMENTUM":
            previous = seed_embedding if ema_state is None else ema_state
            beta = float(config["momentumBeta"])
            ema_state = beta * previous + (1.0 - beta) * current
            norm = float(np.linalg.norm(ema_state))
            if norm > 1e-10:
                ema_state = (ema_state / norm).astype(np.float32)
            query = ema_state
        else:
            raise ValueError(f"unknown drift mode {config['driftMode']}")

    return PickResult(
        tuple(selected), tuple(ranks), tuple(objectives), rejected
    )


def select_graph_ranking(
    library: queue_eval.Library,
    seed_index: int,
    active_positions: np.ndarray,
    terminal_scores: np.ndarray,
    count: int,
    near_copy_threshold: float | None = None,
) -> PickResult:
    order = np.lexsort((library.track_ids[active_positions], -terminal_scores))
    selected: list[int] = []
    ranks: list[int] = []
    objectives: list[float] = []
    rejected = 0
    for terminal_rank, raw_local in enumerate(order, start=1):
        local = int(raw_local)
        score = float(terminal_scores[local])
        if score <= 0.0:
            break
        candidate = int(active_positions[local])
        if candidate == seed_index:
            continue
        seed_cosine = float(
            library.embeddings[seed_index] @ library.embeddings[candidate]
        )
        selected_sims = (
            library.embeddings[np.asarray(selected)] @ library.embeddings[candidate]
            if selected
            else None
        )
        if not diagnostic_copy_eligible(
            library,
            seed_index,
            selected,
            candidate,
            seed_cosine,
            selected_sims,
            near_copy_threshold,
        ):
            rejected += 1
            continue
        if not artist_eligible(library, selected, candidate):
            continue
        selected.append(candidate)
        ranks.append(terminal_rank)
        objectives.append(score)
        if len(selected) == count:
            break
    return PickResult(tuple(selected), tuple(ranks), tuple(objectives), rejected)


def shuffle_permutation(
    library: queue_eval.Library,
    seed_index: int,
    active: np.ndarray,
    shuffle_seed: int,
    activation_binding_id: str,
) -> list[int]:
    namespace_high, namespace_low = device_eval.legacy_shuffle_namespace(
        activation_binding_id
    )
    seed_mix = device_eval.mix64(shuffle_seed)
    seed_low_mix = device_eval.mix64(shuffle_seed + device_eval.GOLDEN_GAMMA)
    ranked: list[tuple[tuple[int, int, int, int, int], int]] = []
    for position in np.flatnonzero(active):
        position = int(position)
        if position == seed_index:
            continue
        track_id = int(library.track_ids[position])
        identity_high = device_eval.mix64(namespace_high ^ track_id)
        identity_low = device_eval.mix64(
            namespace_low + track_id * device_eval.GOLDEN_GAMMA
        )
        priority_high = device_eval.mix64(identity_high ^ seed_mix)
        priority_low = device_eval.mix64(identity_low ^ seed_low_mix)
        key = (
            device_eval.u64(priority_high),
            device_eval.u64(priority_low),
            device_eval.u64(identity_high),
            device_eval.u64(identity_low),
            track_id,
        )
        ranked.append((key, position))
    ranked.sort(key=lambda value: value[0])
    return [position for _, position in ranked]


def select_shuffle(
    library: queue_eval.Library,
    seed_index: int,
    permutation: Sequence[int],
    count: int,
    near_copy_threshold: float | None = None,
) -> PickResult:
    selected: list[int] = []
    ranks: list[int] = []
    cosines: list[float] = []
    rejected = 0
    for shuffle_rank, candidate in enumerate(permutation, start=1):
        seed_cosine = float(
            library.embeddings[seed_index] @ library.embeddings[candidate]
        )
        selected_sims = (
            library.embeddings[np.asarray(selected)] @ library.embeddings[candidate]
            if selected
            else None
        )
        if not diagnostic_copy_eligible(
            library,
            seed_index,
            selected,
            candidate,
            seed_cosine,
            selected_sims,
            near_copy_threshold,
        ):
            rejected += 1
            continue
        if not artist_eligible(library, selected, candidate):
            continue
        selected.append(candidate)
        ranks.append(shuffle_rank)
        cosines.append(seed_cosine)
        if len(selected) == count:
            break
    return PickResult(tuple(selected), tuple(ranks), tuple(cosines), rejected)


def jaccard(left: Sequence[int], right: Sequence[int]) -> float:
    a, b = set(left), set(right)
    return len(a & b) / len(a | b) if a or b else 1.0


def artist_violations(
    library: queue_eval.Library, selected: Sequence[int]
) -> dict[str, int]:
    violations = 0
    for position, candidate in enumerate(selected):
        if not artist_eligible(library, selected[:position], candidate):
            violations += 1
    return {"artist_constraint_violations": violations}


def near_copy_metrics(
    library: queue_eval.Library,
    seed_index: int,
    selected: Sequence[int],
) -> dict[str, object]:
    seed_counts: dict[str, int] = {}
    pair_counts: dict[str, int] = {}
    for threshold in (0.95, 0.97, 0.99, 0.995):
        seed_counts[f"{threshold:.3f}"] = sum(
            diagnostic_near_copy(
                library,
                seed_index,
                candidate,
                float(library.embeddings[seed_index] @ library.embeddings[candidate]),
                threshold,
            )
            for candidate in selected
        )
        pair_count = 0
        for left_position, left in enumerate(selected):
            for right in selected[left_position + 1 :]:
                cosine = float(library.embeddings[left] @ library.embeddings[right])
                if diagnostic_near_copy(
                    library, left, right, cosine, threshold
                ):
                    pair_count += 1
        pair_counts[f"{threshold:.3f}"] = pair_count
    return {
        "seed_near_copy_count_by_diagnostic_threshold": seed_counts,
        "within_queue_near_copy_pairs_by_diagnostic_threshold": pair_counts,
    }


def variant_record(
    library: queue_eval.Library,
    seed_index: int,
    result: PickResult,
    requested: int,
    elapsed_ms: float,
) -> dict[str, object]:
    metrics = queue_eval.queue_metrics(
        library,
        seed_index,
        result.selected,
        result.candidate_ranks,
        requested,
    )
    metrics.update(artist_violations(library, result.selected))
    metrics.update(near_copy_metrics(library, seed_index, result.selected))
    return {
        "metrics": metrics,
        "elapsed_ms": elapsed_ms,
        "near_copy_rejections": result.near_copy_rejections,
        "track_ids": [int(library.track_ids[index]) for index in result.selected],
        "candidate_ranks": list(result.candidate_ranks),
        "objective_values": list(result.objective_values),
        "tracks": [
            {
                **queue_eval.track_summary(library, index),
                "seed_cosine": float(
                    library.embeddings[seed_index] @ library.embeddings[index]
                ),
            }
            for index in result.selected
        ],
    }


def timed(call: Callable[[], PickResult]) -> tuple[PickResult, float]:
    started = time.perf_counter()
    result = call()
    return result, (time.perf_counter() - started) * 1_000.0


def run_seed(
    library: queue_eval.Library,
    seed_index: int,
    active: np.ndarray,
    configs: dict[str, object],
    queue_size: int,
    active_graph: selection_eval.Graph,
    active_graph_positions: np.ndarray,
    old_to_active_graph: np.ndarray,
    active_graph_transition: tuple[np.ndarray, np.ndarray, np.ndarray],
    activation_binding_id: str,
) -> dict[str, object]:
    active_count = int(active.sum())
    max_pool = max(queue_size, int(active_count * max(REACH_FRACTIONS)))
    candidates, relevance, _ = retrieve_active(
        library, seed_index, active, max_pool
    )
    started = time.perf_counter()
    candidate_embeddings = library.embeddings[candidates]
    gram = (candidate_embeddings @ candidate_embeddings.T).astype(
        np.float32, copy=False
    )
    gram_ms = (time.perf_counter() - started) * 1_000.0
    current_pool = max(queue_size, int(active_count * 0.02))
    current_n = min(current_pool, candidates.size)
    current_candidates = candidates[:current_n]
    current_relevance = relevance[:current_n]
    current_gram = gram[:current_n, :current_n]

    variants: dict[str, object] = {}

    def record(name: str, call: Callable[[], PickResult]) -> PickResult:
        result, elapsed = timed(call)
        variants[name] = variant_record(
            library, seed_index, result, queue_size, elapsed
        )
        return result

    closest = record(
        "closest_canonical",
        lambda: select_closest(
            library,
            seed_index,
            candidates,
            relevance,
            queue_size,
        ),
    )
    closest_filter = record(
        "closest_cosine_duration_filter_diagnostic",
        lambda: select_closest(
            library,
            seed_index,
            candidates,
            relevance,
            queue_size,
            near_copy_threshold=0.95,
        ),
    )
    mmr = record(
        "mmr_canonical",
        lambda: select_mmr(
            library,
            seed_index,
            current_candidates,
            current_relevance,
            current_gram,
            queue_size,
            USER_MMR_LAMBDA,
            include_seed_in_history=False,
        ),
    )
    mmr_seed = record(
        "mmr_seed_in_redundancy_history_experiment",
        lambda: select_mmr(
            library,
            seed_index,
            current_candidates,
            current_relevance,
            current_gram,
            queue_size,
            USER_MMR_LAMBDA,
            include_seed_in_history=True,
        ),
    )
    mmr_filter = record(
        "mmr_canonical_cosine_duration_filter_diagnostic",
        lambda: select_mmr(
            library,
            seed_index,
            current_candidates,
            current_relevance,
            current_gram,
            queue_size,
            USER_MMR_LAMBDA,
            include_seed_in_history=False,
            near_copy_threshold=0.95,
        ),
    )
    dpp = record(
        "dpp_canonical",
        lambda: select_dpp(
            library,
            seed_index,
            current_candidates,
            current_relevance,
            current_gram,
            queue_size,
            quality_exponent=1.0,
            condition_on_seed=False,
        ),
    )
    dpp_seed = record(
        "dpp_conditioned_on_seed_experiment",
        lambda: select_dpp(
            library,
            seed_index,
            current_candidates,
            current_relevance,
            current_gram,
            queue_size,
            quality_exponent=1.0,
            condition_on_seed=True,
        ),
    )
    dpp_filter = record(
        "dpp_canonical_cosine_duration_filter_diagnostic",
        lambda: select_dpp(
            library,
            seed_index,
            current_candidates,
            current_relevance,
            current_gram,
            queue_size,
            quality_exponent=1.0,
            condition_on_seed=False,
            near_copy_threshold=0.95,
        ),
    )

    graph_config = configs.get("graph_explorer")
    if isinstance(graph_config, dict):
        active_seed = int(old_to_active_graph[seed_index])
        if active_seed < 0:
            raise ValueError("active seed is absent from compact graph")
        terminal_scores, _ = selection_eval.exact_terminal_distribution(
            active_graph,
            active_seed,
            float(graph_config["walkRestartAlpha"]),
            active_graph_transition,
            weighted=False,
        )
        graph_canonical = record(
            "graph_explorer_canonical",
            lambda: select_graph_ranking(
                library,
                seed_index,
                active_graph_positions,
                terminal_scores,
                queue_size,
            ),
        )
        graph_filter = record(
            "graph_explorer_cosine_duration_filter_diagnostic",
            lambda: select_graph_ranking(
                library,
                seed_index,
                active_graph_positions,
                terminal_scores,
                queue_size,
                near_copy_threshold=0.95,
            ),
        )
    else:
        graph_canonical = graph_filter = PickResult((), (), ())

    shuffle_config = configs.get("uniform_shuffle")
    if isinstance(shuffle_config, dict):
        permutation = shuffle_permutation(
            library,
            seed_index,
            active,
            int(shuffle_config["shuffleSeed"]),
            activation_binding_id,
        )
        shuffle_canonical = record(
            "uniform_shuffle_canonical",
            lambda: select_shuffle(
                library, seed_index, permutation, queue_size
            ),
        )
        shuffle_filter = record(
            "uniform_shuffle_cosine_duration_filter_diagnostic",
            lambda: select_shuffle(
                library,
                seed_index,
                permutation,
                queue_size,
                near_copy_threshold=0.95,
            ),
        )
    else:
        shuffle_canonical = shuffle_filter = PickResult((), (), ())

    for case_id, short_name in (
        ("mmr_seed_interpolation", "seed_interpolation"),
        ("mmr_momentum", "momentum"),
    ):
        config = configs.get(case_id)
        if not isinstance(config, dict):
            continue
        canonical = record(
            f"drift_{short_name}_canonical_mmr",
            lambda config=config: drift_select(
                library,
                seed_index,
                active,
                config,
                include_seed_in_history=False,
            ),
        )
        seeded = record(
            f"drift_{short_name}_seed_in_redundancy_history_experiment",
            lambda config=config: drift_select(
                library,
                seed_index,
                active,
                config,
                include_seed_in_history=True,
            ),
        )
        variants[f"drift_{short_name}_comparison"] = {
            "membership_jaccard": jaccard(canonical.selected, seeded.selected),
            "exact_order": canonical.selected == seeded.selected,
        }

    sweep: dict[str, object] = {}
    for reach in REACH_FRACTIONS:
        pool = max(queue_size, int(active_count * reach))
        n = min(pool, candidates.size)
        for exponent in DPP_EXPONENTS:
            key = f"reach_{reach:.4f}_exponent_{exponent:.1f}"
            canonical_result, canonical_ms = timed(
                lambda n=n, exponent=exponent: select_dpp(
                    library,
                    seed_index,
                    candidates[:n],
                    relevance[:n],
                    gram[:n, :n],
                    queue_size,
                    exponent,
                    condition_on_seed=False,
                )
            )
            conditioned_result, conditioned_ms = timed(
                lambda n=n, exponent=exponent: select_dpp(
                    library,
                    seed_index,
                    candidates[:n],
                    relevance[:n],
                    gram[:n, :n],
                    queue_size,
                    exponent,
                    condition_on_seed=True,
                )
            )
            canonical_record = variant_record(
                library,
                seed_index,
                canonical_result,
                queue_size,
                canonical_ms,
            )
            conditioned_record = variant_record(
                library,
                seed_index,
                conditioned_result,
                queue_size,
                conditioned_ms,
            )
            sweep[key] = {
                "reach_fraction": reach,
                "pool_size": n,
                "quality_exponent": exponent,
                "canonical": canonical_record,
                "conditioned_on_seed": conditioned_record,
                "membership_jaccard": jaccard(
                    canonical_result.selected, conditioned_result.selected
                ),
            }

    return {
        "seed": queue_eval.track_summary(library, seed_index),
        "active_count": active_count,
        "maximum_pool": max_pool,
        "gram_ms": gram_ms,
        "variants": variants,
        "comparisons": {
            "closest_filter_jaccard": jaccard(
                closest.selected, closest_filter.selected
            ),
            "mmr_seed_history_jaccard": jaccard(mmr.selected, mmr_seed.selected),
            "mmr_filter_jaccard": jaccard(mmr.selected, mmr_filter.selected),
            "dpp_conditioned_jaccard": jaccard(dpp.selected, dpp_seed.selected),
            "dpp_filter_jaccard": jaccard(dpp.selected, dpp_filter.selected),
            "graph_filter_jaccard": jaccard(
                graph_canonical.selected, graph_filter.selected
            ),
            "shuffle_filter_jaccard": jaccard(
                shuffle_canonical.selected, shuffle_filter.selected
            ),
        },
        "dpp_sweep": sweep,
    }


def numeric_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if not finite.size:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def summarize(records: Sequence[dict[str, object]]) -> dict[str, object]:
    variant_names = sorted(
        {
            name
            for record in records
            for name, value in record["variants"].items()
            if isinstance(value, dict) and "metrics" in value
        }
    )
    result: dict[str, object] = {
        "seed_count": len(records),
        "variants": {},
        "comparisons": {},
        "dpp_sweep": {},
    }
    for name in variant_names:
        values = [record["variants"][name] for record in records if name in record["variants"]]
        metrics = [value["metrics"] for value in values]
        result["variants"][name] = {
            "mean_seed_cosine": numeric_summary(
                float(value["mean_seed_cosine"]) for value in metrics
            ),
            "mean_pairwise_cosine": numeric_summary(
                float(value["mean_pairwise_cosine"]) for value in metrics
            ),
            "unique_known_artists": numeric_summary(
                float(value["unique_known_artists"]) for value in metrics
            ),
            "queue_count": numeric_summary(float(value["count"]) for value in metrics),
            "artist_constraint_violations": int(
                sum(int(value["artist_constraint_violations"]) for value in metrics)
            ),
            "seed_near_copy_count_at_0.95": int(
                sum(
                    int(value["seed_near_copy_count_by_diagnostic_threshold"]["0.950"])
                    for value in metrics
                )
            ),
            "within_queue_near_copy_pairs_at_0.95": int(
                sum(
                    int(value["within_queue_near_copy_pairs_by_diagnostic_threshold"]["0.950"])
                    for value in metrics
                )
            ),
            "elapsed_ms": numeric_summary(float(value["elapsed_ms"]) for value in values),
        }
    comparison_names = sorted(
        {key for record in records for key in record["comparisons"]}
    )
    for name in comparison_names:
        result["comparisons"][name] = numeric_summary(
            float(record["comparisons"][name]) for record in records
        )
    sweep_names = sorted(
        {key for record in records for key in record["dpp_sweep"]}
    )
    for name in sweep_names:
        pairs = [record["dpp_sweep"][name] for record in records]
        result["dpp_sweep"][name] = {
            "reach_fraction": pairs[0]["reach_fraction"],
            "pool_size": pairs[0]["pool_size"],
            "quality_exponent": pairs[0]["quality_exponent"],
            "membership_jaccard": numeric_summary(
                float(pair["membership_jaccard"]) for pair in pairs
            ),
            "canonical_mean_seed_cosine": numeric_summary(
                float(pair["canonical"]["metrics"]["mean_seed_cosine"])
                for pair in pairs
            ),
            "conditioned_mean_seed_cosine": numeric_summary(
                float(pair["conditioned_on_seed"]["metrics"]["mean_seed_cosine"])
                for pair in pairs
            ),
            "canonical_mean_pairwise_cosine": numeric_summary(
                float(pair["canonical"]["metrics"]["mean_pairwise_cosine"])
                for pair in pairs
            ),
            "conditioned_mean_pairwise_cosine": numeric_summary(
                float(pair["conditioned_on_seed"]["metrics"]["mean_pairwise_cosine"])
                for pair in pairs
            ),
            "canonical_queue_count": numeric_summary(
                float(pair["canonical"]["metrics"]["count"]) for pair in pairs
            ),
            "conditioned_queue_count": numeric_summary(
                float(pair["conditioned_on_seed"]["metrics"]["count"])
                for pair in pairs
            ),
        }
    return result


def device_parity(
    records: Sequence[dict[str, object]],
    device_report: dict[str, object],
) -> dict[str, object]:
    record_by_seed = {int(record["seed"]["track_id"]): record for record in records}
    variant_by_case = {
        "closest": "closest_canonical",
        "mmr": "mmr_canonical",
        "dpp": "dpp_canonical",
        "mmr_seed_interpolation": "drift_seed_interpolation_canonical_mmr",
        "mmr_momentum": "drift_momentum_canonical_mmr",
        "graph_explorer": "graph_explorer_canonical",
        "uniform_shuffle": "uniform_shuffle_canonical",
    }
    comparisons: list[dict[str, object]] = []
    for run in device_report.get("selectionRuns", []):
        if int(run.get("repeat", 0)) != 1:
            continue
        case_id = str(run["caseId"])
        variant_name = variant_by_case.get(case_id)
        seed_id = int(run["seedTrackId"])
        if variant_name is None or seed_id not in record_by_seed:
            continue
        expected = [int(track["trackId"]) for track in run["tracks"]]
        actual = [
            int(track_id)
            for track_id in record_by_seed[seed_id]["variants"][variant_name][
                "track_ids"
            ]
        ]
        comparisons.append(
            {
                "seed_track_id": seed_id,
                "case_id": case_id,
                "exact_track_id_order": actual == expected,
            }
        )
    return {
        "comparison_count": len(comparisons),
        "exact_order_count": sum(
            bool(value["exact_track_id_order"]) for value in comparisons
        ),
        "all_exact": all(
            bool(value["exact_track_id_order"]) for value in comparisons
        ),
        "comparisons": comparisons,
    }


def write_qualitative(path: Path, records: Sequence[dict[str, object]]) -> None:
    names = (
        "closest_canonical",
        "mmr_canonical",
        "mmr_seed_in_redundancy_history_experiment",
        "dpp_canonical",
        "dpp_conditioned_on_seed_experiment",
        "drift_seed_interpolation_canonical_mmr",
        "drift_seed_interpolation_seed_in_redundancy_history_experiment",
        "drift_momentum_canonical_mmr",
        "drift_momentum_seed_in_redundancy_history_experiment",
        "graph_explorer_canonical",
        "graph_explorer_cosine_duration_filter_diagnostic",
        "uniform_shuffle_canonical",
        "uniform_shuffle_cosine_duration_filter_diagnostic",
    )
    lines = [
        "# Seed-conditioning qualitative queues",
        "",
        "Metadata below labels frozen results only; it never affects a score.",
        "Seed-conditioned MMR and DPP are experimental named variants, not replacements",
        "for the canonical modes.",
        "",
    ]
    for record in records:
        seed = record["seed"]
        lines += [
            f"## {seed.get('artist') or 'Unknown'} - {seed.get('title') or 'Unknown'}",
            "",
        ]
        for name in names:
            value = record["variants"].get(name)
            if not isinstance(value, dict) or "tracks" not in value:
                continue
            metrics = value["metrics"]
            lines += [
                f"### {name}",
                "",
                (
                    f"Seed cosine `{metrics['mean_seed_cosine']:.4f}`; pairwise "
                    f"`{metrics['mean_pairwise_cosine']:.4f}`; "
                    f"artists `{metrics['unique_known_artists']}`; "
                    f"filled `{metrics['count']}/{QUEUE_SIZE}`."
                ),
                "",
            ]
            for rank, track in enumerate(value["tracks"], start=1):
                lines.append(
                    f"{rank}. {track.get('artist') or 'Unknown'} - "
                    f"{track.get('title') or 'Unknown'} "
                    f"(seed cosine {track['seed_cosine']:.4f}, id {track['track_id']})"
                )
            lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--device-run", type=Path, default=DEFAULT_DEVICE_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--queue-size", type=int, default=QUEUE_SIZE)
    parser.add_argument("--max-seeds", type=int)
    parser.add_argument("--skip-hash", action="store_true", help="development only")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path = args.device_run / "report.json"
    catalog_path = args.device_run / "active-catalog.tsv"
    library, db_sha256 = queue_eval.load_library(
        args.db, verify_hash=not args.skip_hash
    )
    active = active_mask(library, catalog_path)
    seeds, configs = cohort_from_report(library, report_path, active)
    if args.max_seeds is not None:
        if args.max_seeds <= 0:
            raise ValueError("--max-seeds must be positive")
        seeds = seeds[: args.max_seeds]
    device_report = json.loads(report_path.read_text(encoding="utf-8"))
    activation_binding_id = str(device_report["generation"]["activationBindingId"])
    graph = selection_eval.parse_graph(args.db)
    (
        active_graph,
        active_graph_positions,
        old_to_active_graph,
        active_graph_repair,
    ) = device_eval.build_active_graph(library, graph, active)
    active_graph_transition = selection_eval.build_edge_transition(
        active_graph, weighted=False
    )

    args.output.mkdir(parents=True, exist_ok=True)
    records_path = args.output / "seed-results.jsonl"
    manifest_path = args.output / "manifest.json"
    if args.force:
        records_path.unlink(missing_ok=True)
        (args.output / "summary.json").unlink(missing_ok=True)
        (args.output / "qualitative.md").unlink(missing_ok=True)

    manifest = {
        "experiment_version": EXPERIMENT_VERSION,
        "evaluator_sha256": sha256_file(Path(__file__)),
        "database": str(args.db.resolve()),
        "database_sha256": db_sha256,
        "database_rows": library.count,
        "active_catalog": str(catalog_path.resolve()),
        "active_catalog_sha256": sha256_file(catalog_path),
        "active_tracks": int(active.sum()),
        "device_report": str(report_path.resolve()),
        "device_report_sha256": sha256_file(report_path),
        "queue_size": args.queue_size,
        "seed_track_ids": [int(library.track_ids[index]) for index in seeds],
        "dpp_reach_fractions": list(REACH_FRACTIONS),
        "dpp_quality_exponents": list(DPP_EXPONENTS),
        "mmr_lambda": USER_MMR_LAMBDA,
        "artist_constraints": {
            "maximum_matching_credit": MAX_PER_ARTIST,
            "minimum_spacing": MIN_ARTIST_SPACING,
        },
        "identity_warning": (
            "cosine-duration filtering is a falsification diagnostic only; "
            "embeddings do not prove recording identity"
        ),
        "active_graph_repair": active_graph_repair,
        "activation_binding_id": activation_binding_id,
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    if not args.force and manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise RuntimeError(
                "existing output belongs to a different evaluator/input; use a new "
                "output directory or --force"
            )
    atomic_json(manifest_path, manifest)

    completed = completed_seed_ids(records_path)
    for position, seed_index in enumerate(seeds, start=1):
        seed_id = int(library.track_ids[seed_index])
        if seed_id in completed:
            print(f"seed {position}/{len(seeds)} id={seed_id} resumed", file=sys.stderr)
            continue
        print(f"seed {position}/{len(seeds)} id={seed_id} starting", file=sys.stderr)
        record = run_seed(
            library,
            seed_index,
            active,
            configs.get(seed_id, {}),
            args.queue_size,
            active_graph,
            active_graph_positions,
            old_to_active_graph,
            active_graph_transition,
            activation_binding_id,
        )
        append_jsonl(records_path, record)
        print(f"seed {position}/{len(seeds)} id={seed_id} saved", file=sys.stderr)

    records = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) != len(seeds):
        raise RuntimeError(f"expected {len(seeds)} seed records, found {len(records)}")
    summary = summarize(records)
    summary["device_parity"] = device_parity(records, device_report)
    atomic_json(args.output / "summary.json", summary)
    write_qualitative(args.output / "qualitative.md", records)


if __name__ == "__main__":
    main()
